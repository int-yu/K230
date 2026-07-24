"""K230 钢球距离判断程序。

原理：

    distance_cm = real_diameter_cm * focal_px / pixel_diameter

钢球直径已知为 1 cm，因此只需要 YOLO 检测框的像素直径和相机焦距即可估算
距离。检测、模型路径、显示和 Sensor 配置全部复用 detectors.steelball。
"""

import gc
import os
import sys
import time

# CanMV 按绝对路径启动子目录脚本时不会自动加入项目根目录。
for _path in ("/sdcard/K230", "/sdcard"):
    if _path not in sys.path:
        sys.path.append(_path)

from core.draw_utils import draw_osd_status, draw_osd_text
from detectors.steelball import (
    PipeLine,
    YOLO11,
    _apply_sensor_orientation,
    _create_sensor,
    normalize_detections,
    resolve_display_config,
    resolve_kmodel_path,
)
from config import (
    STEELBALL_CONFIDENCE_THRESHOLD,
    STEELBALL_DISTANCE_CALIBRATION_SCALE,
    STEELBALL_DISTANCE_CENTER_COLOR,
    STEELBALL_DISTANCE_DEBUG_PRINT,
    STEELBALL_DISTANCE_DEBUG_RAW,
    STEELBALL_DISTANCE_DIAMETER_MODE,
    STEELBALL_DISTANCE_FOCAL_X,
    STEELBALL_DISTANCE_FOCAL_Y,
    STEELBALL_DISTANCE_GC_INTERVAL,
    STEELBALL_DISTANCE_MIN_PIXEL_DIAMETER,
    STEELBALL_DISTANCE_PRINT_INTERVAL,
    STEELBALL_DISTANCE_SMOOTH_ALPHA,
    STEELBALL_DISTANCE_TEXT_COLOR,
    STEELBALL_DISTANCE_USE_SMOOTHING,
    STEELBALL_LABELS,
    STEELBALL_MAX_BOXES_NUM,
    STEELBALL_MODEL_INPUT_SIZE,
    STEELBALL_NMS_THRESHOLD,
    STEELBALL_REAL_DIAMETER_CM,
    STEELBALL_RGB888P_SIZE,
)


def _scaled_focal(display_size, rgb888p_size=STEELBALL_RGB888P_SIZE):
    """把 640x480 内参焦距缩放到 YOLO/OSD 输出坐标尺度。"""

    display_width = float(display_size[0])
    display_height = float(display_size[1])
    source_width = float(rgb888p_size[0])
    source_height = float(rgb888p_size[1])
    return (
        STEELBALL_DISTANCE_FOCAL_X * display_width / source_width,
        STEELBALL_DISTANCE_FOCAL_Y * display_height / source_height,
    )


def estimate_steelball_distance(
    detection,
    display_size,
    real_diameter_cm=STEELBALL_REAL_DIAMETER_CM,
    diameter_mode=STEELBALL_DISTANCE_DIAMETER_MODE,
):
    """根据一个钢球检测框估算距离，返回扩展后的检测结果。"""

    width = float(detection["bbox_w"])
    height = float(detection["bbox_h"])
    if width < STEELBALL_DISTANCE_MIN_PIXEL_DIAMETER:
        return None
    if height < STEELBALL_DISTANCE_MIN_PIXEL_DIAMETER:
        return None

    focal_x, focal_y = _scaled_focal(display_size)
    distance_x = real_diameter_cm * focal_x / width
    distance_y = real_diameter_cm * focal_y / height

    if diameter_mode == "width":
        raw_distance_cm = distance_x
        pixel_diameter = width
    elif diameter_mode == "height":
        raw_distance_cm = distance_y
        pixel_diameter = height
    elif diameter_mode == "min":
        if width <= height:
            raw_distance_cm = distance_x
            pixel_diameter = width
        else:
            raw_distance_cm = distance_y
            pixel_diameter = height
    else:
        raw_distance_cm = (distance_x + distance_y) / 2.0
        pixel_diameter = (width + height) / 2.0

    distance_cm = raw_distance_cm * STEELBALL_DISTANCE_CALIBRATION_SCALE

    result = dict(detection)
    result.update({
        "distance_cm": distance_cm,
        "distance_m": distance_cm / 100.0,
        "raw_distance_cm": raw_distance_cm,
        "pixel_diameter": pixel_diameter,
        "distance_x_cm": distance_x,
        "distance_y_cm": distance_y,
        "calibration_scale": STEELBALL_DISTANCE_CALIBRATION_SCALE,
        "diameter_mode": diameter_mode,
    })
    return result


def select_best_detection(detections):
    """选择置信度最高、面积更大的钢球。"""

    best = None
    best_key = None
    for detection in detections:
        area = float(detection["bbox_w"]) * float(detection["bbox_h"])
        key = (float(detection["confidence"]), area)
        if best is None or key > best_key:
            best = detection
            best_key = key
    return best


class SteelballDistanceEstimator:
    """只负责检测框到距离的计算和平滑，不负责 YOLO 推理。"""

    def __init__(
        self,
        display_size,
        use_smoothing=STEELBALL_DISTANCE_USE_SMOOTHING,
        smooth_alpha=STEELBALL_DISTANCE_SMOOTH_ALPHA,
    ):
        self.display_size = display_size
        self.use_smoothing = bool(use_smoothing)
        self.smooth_alpha = float(smooth_alpha)
        self._smooth_distance_cm = None

    def process(self, detections):
        best = select_best_detection(detections)
        if best is None:
            self._smooth_distance_cm = None
            return None

        result = estimate_steelball_distance(best, self.display_size)
        if result is None:
            self._smooth_distance_cm = None
            return None

        if self.use_smoothing:
            if self._smooth_distance_cm is None:
                self._smooth_distance_cm = result["distance_cm"]
            else:
                alpha = self.smooth_alpha
                self._smooth_distance_cm = (
                    self._smooth_distance_cm * (1.0 - alpha) +
                    result["distance_cm"] * alpha
                )
            result["raw_distance_cm"] = result["distance_cm"]
            result["distance_cm"] = self._smooth_distance_cm
            result["distance_m"] = self._smooth_distance_cm / 100.0
        return result


def _draw_best_target(osd_img, target):
    """在 OSD 上绘制最佳目标中心和距离文字。"""

    if target is None:
        draw_osd_text(
            osd_img,
            "Dist: --",
            x=5,
            y=64,
            size=26,
            color=STEELBALL_DISTANCE_TEXT_COLOR,
        )
        return

    center_x = int(target["center_x"])
    center_y = int(target["center_y"])
    try:
        osd_img.draw_cross(
            center_x,
            center_y,
            color=STEELBALL_DISTANCE_CENTER_COLOR,
            size=8,
            thickness=2,
        )
    except Exception:
        pass

    text_x = max(0, min(int(target["x1"]), int(target["x2"]) - 80))
    text_y = max(64, int(target["y1"]) - 30)
    draw_osd_text(
        osd_img,
        "Dist: {:.1f}cm".format(target["distance_cm"]),
        x=text_x,
        y=text_y,
        size=24,
        color=STEELBALL_DISTANCE_TEXT_COLOR,
    )


def _short_repr(value, limit=160):
    try:
        text = repr(value)
    except Exception:
        text = str(value)
    if len(text) > limit:
        return text[:limit] + "..."
    return text


def run_steelball_distance():
    """运行实时钢球检测和距离判断演示。"""

    if PipeLine is None or YOLO11 is None:
        raise RuntimeError("当前环境缺少 libs.PipeLine 或 libs.YOLO，只能在 CanMV/K230 上运行")

    print("初始化钢球距离判断")
    kmodel_path = resolve_kmodel_path()
    display_target, display_size, display_mode = resolve_display_config()
    print("模型路径：" + kmodel_path)
    print("显示目标：" + display_target)
    print("钢球直径：{:.2f}cm".format(STEELBALL_REAL_DIAMETER_CM))
    print("检测输入：" + str(STEELBALL_RGB888P_SIZE) + "，模型输入：" + str(STEELBALL_MODEL_INPUT_SIZE))

    pl = None
    yolo = None
    frame_count = 0
    clock = time.clock()

    try:
        pl = PipeLine(
            rgb888p_size=STEELBALL_RGB888P_SIZE,
            display_size=display_size,
            display_mode=display_mode,
        )

        sensor = _create_sensor()
        pl.create(sensor=sensor)
        _apply_sensor_orientation(sensor)
        display_size = pl.get_display_size()
        estimator = SteelballDistanceEstimator(display_size)

        yolo = YOLO11(
            task_type="detect",
            mode="video",
            kmodel_path=kmodel_path,
            labels=STEELBALL_LABELS,
            rgb888p_size=STEELBALL_RGB888P_SIZE,
            model_input_size=STEELBALL_MODEL_INPUT_SIZE,
            display_size=display_size,
            conf_thresh=STEELBALL_CONFIDENCE_THRESHOLD,
            nms_thresh=STEELBALL_NMS_THRESHOLD,
            max_boxes_num=STEELBALL_MAX_BOXES_NUM,
            debug_mode=0,
        )
        yolo.config_preprocess()
        print("初始化完成")

        while True:
            try:
                os.exitpoint()
            except Exception:
                pass

            clock.tick()
            frame_count += 1

            frame = pl.get_frame()
            dets = yolo.run(frame)

            yolo.draw_result(dets, pl.osd_img)
            detections = normalize_detections(dets)
            target = estimator.process(detections)

            fps = clock.fps()
            draw_osd_status(
                pl.osd_img,
                fps=fps,
                count=len(detections),
                count_label="Num",
            )
            _draw_best_target(pl.osd_img, target)
            pl.show_image()

            if (
                STEELBALL_DISTANCE_GC_INTERVAL > 0 and
                frame_count % STEELBALL_DISTANCE_GC_INTERVAL == 0
            ):
                gc.collect()

            if (
                STEELBALL_DISTANCE_PRINT_INTERVAL > 0 and
                frame_count % STEELBALL_DISTANCE_PRINT_INTERVAL == 0
            ):
                if target is None:
                    print("FPS: {:.1f}, Num: {}, Dist: --".format(
                        fps,
                        len(detections),
                    ))
                else:
                    if STEELBALL_DISTANCE_DEBUG_PRINT:
                        print(
                            "FPS: {:.1f}, Num: {}, Dist: {:.1f}cm, "
                            "bbox=({:.1f},{:.1f}), dpx={:.1f}, "
                            "dx={:.1f}, dy={:.1f}, mode={}".format(
                                fps,
                                len(detections),
                                target["distance_cm"],
                                target["bbox_w"],
                                target["bbox_h"],
                                target["pixel_diameter"],
                                target["distance_x_cm"],
                                target["distance_y_cm"],
                                target["diameter_mode"],
                            )
                        )
                    else:
                        print("FPS: {:.1f}, Num: {}, Dist: {:.1f}cm".format(
                            fps,
                            len(detections),
                            target["distance_cm"],
                        ))
                    if STEELBALL_DISTANCE_DEBUG_RAW:
                        print("raw dets: {}".format(_short_repr(dets)))

    except Exception as exc:
        print("程序发生错误")
        try:
            sys.print_exception(exc)
        except Exception:
            print(exc)

    finally:
        print("释放资源")
        if yolo is not None:
            try:
                yolo.deinit()
            except Exception:
                pass

        if pl is not None:
            try:
                pl.destroy()
            except Exception:
                pass

        gc.collect()
        print("程序结束")


if __name__ == "__main__":
    run_steelball_distance()
