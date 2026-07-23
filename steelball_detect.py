"""K230 钢球 YOLO11 检测程序。

上板使用方式：
1. 把本文件复制到 /sdcard/steelball_detect.py。
2. 把 yolo11n_det_320.kmodel 复制到 /sdcard/yolo11n_det_320.kmodel，
   或复制到 /sdcard/7839/yolo11n_det_320.kmodel。
3. 在 CanMV IDE 中运行 /sdcard/steelball_detect.py。

程序功能：
- 使用 7839 目录中的 YOLO11 模型检测钢球。
- 在屏幕上绘制检测框。
- 在左上角显示当前帧检测数量。
"""

from libs.PipeLine import PipeLine
from libs.YOLO import YOLO11
from media.sensor import Sensor

import gc
import os
import sys
import time

from config import (
    BOARD_DISPLAY_HEIGHT,
    BOARD_DISPLAY_MODE,
    BOARD_DISPLAY_WIDTH,
    CAMERA_FPS,
    CAMERA_HMIRROR,
    CAMERA_ID,
    CAMERA_SOURCE_HEIGHT,
    CAMERA_SOURCE_WIDTH,
    CAMERA_VFLIP,
    DISPLAY_TARGET,
    DISPLAY_TARGET_BOARD,
    DISPLAY_TARGET_IDE,
    IDE_DISPLAY_HEIGHT,
    IDE_DISPLAY_MODE,
    IDE_DISPLAY_WIDTH,
    STEELBALL_CONFIDENCE_THRESHOLD,
    STEELBALL_KMODEL_PATH,
    STEELBALL_KMODEL_PATH_CANDIDATES,
    STEELBALL_LABELS,
    STEELBALL_MAX_BOXES_NUM,
    STEELBALL_MODEL_INPUT_SIZE,
    STEELBALL_NMS_THRESHOLD,
    STEELBALL_RGB888P_SIZE,
)


# 这些只是显示和调试细节，不放到 config.py，避免配置文件膨胀。
COUNT_TEXT_SIZE = 32
COUNT_TEXT_COLOR = (255, 255, 255, 255)
COUNT_BACKGROUND_COLOR = (160, 0, 0, 0)
PRINT_INTERVAL = 10
GC_INTERVAL = 1


def _path_exists(path):
    """判断文件路径是否可以访问。"""

    try:
        os.stat(path)
        return True
    except Exception:
        return False


def resolve_display_config(display_target=None):
    """根据统一显示目标开关，生成 PipeLine 需要的显示参数。"""

    if display_target is None:
        display_target = DISPLAY_TARGET

    if display_target == DISPLAY_TARGET_BOARD:
        return (
            display_target,
            [BOARD_DISPLAY_WIDTH, BOARD_DISPLAY_HEIGHT],
            BOARD_DISPLAY_MODE,
        )

    if display_target == DISPLAY_TARGET_IDE:
        return (
            display_target,
            [IDE_DISPLAY_WIDTH, IDE_DISPLAY_HEIGHT],
            IDE_DISPLAY_MODE,
        )

    raise ValueError("不支持的显示目标：{}，可用值为 board 或 ide".format(display_target))


def resolve_kmodel_path():
    """从候选路径中选择第一个可访问的 kmodel 文件。"""

    checked = []
    for path in STEELBALL_KMODEL_PATH_CANDIDATES:
        checked.append(path)
        if _path_exists(path):
            return path

    if STEELBALL_KMODEL_PATH not in checked:
        checked.append(STEELBALL_KMODEL_PATH)
        if _path_exists(STEELBALL_KMODEL_PATH):
            return STEELBALL_KMODEL_PATH

    raise RuntimeError("找不到钢球检测模型：" + ", ".join(checked))


def _safe_len(obj):
    """安全获取对象长度，失败时返回 None。"""

    try:
        return len(obj)
    except Exception:
        return None


def _shape(obj):
    """安全获取数组 shape，兼容 ulab/numpy 风格结果。"""

    try:
        return obj.shape
    except Exception:
        return None


def _is_detection_row(row):
    """判断一行数据是否像 YOLO 检测框：x1、y1、x2、y2、置信度、类别。"""

    row_len = _safe_len(row)
    if row_len is None or row_len < 6:
        return False

    try:
        float(row[4])
        int(row[5])
        return True
    except Exception:
        return False


def _count_from_array_shape(obj):
    """根据数组 shape 推断检测数量。"""

    shape = _shape(obj)
    if shape is None:
        return None

    try:
        ndim = len(shape)
    except Exception:
        return None

    if ndim >= 2:
        return int(shape[0])

    if ndim == 1:
        width = int(shape[0])
        if width == 0:
            return 0
        if width >= 6:
            return 1

    return None


def count_detections(dets):
    """统计当前帧检测数量，兼容不同 YOLO11 库的返回结构。"""

    if dets is None:
        return 0

    count = _count_from_array_shape(dets)
    if count is not None:
        return count

    if _is_detection_row(dets):
        return 1

    dets_len = _safe_len(dets)
    if dets_len is None or dets_len == 0:
        return 0

    # 少数后处理函数会返回 (boxes, scores, ids)，此时第一项才是 Nx4 检测框数组。
    if dets_len in (2, 3):
        try:
            first = dets[0]
            first_shape = _shape(first)
            if first_shape is not None and len(first_shape) >= 2:
                return int(first_shape[0])
        except Exception:
            pass

    try:
        first = dets[0]
        if _is_detection_row(first):
            return int(dets_len)
    except Exception:
        pass

    # 最后的保守兜底：只统计结构像检测框的行。
    count = 0
    try:
        for row in dets:
            if _is_detection_row(row):
                count += 1
        return count
    except Exception:
        return 0


def draw_count(osd_img, count):
    """在 OSD 左上角绘制当前帧检测数量。"""

    text = "Num: " + str(count)

    try:
        osd_img.draw_rectangle(
            0,
            0,
            150,
            40,
            color=COUNT_BACKGROUND_COLOR,
            thickness=1,
            fill=True,
        )
    except Exception:
        pass

    try:
        osd_img.draw_string_advanced(
            5,
            4,
            COUNT_TEXT_SIZE,
            text,
            color=COUNT_TEXT_COLOR,
        )
    except Exception:
        osd_img.draw_string_advanced(5, 4, COUNT_TEXT_SIZE, text)


def _create_sensor():
    """按项目公共摄像头配置创建 Sensor。"""

    try:
        return Sensor(
            id=CAMERA_ID,
            width=CAMERA_SOURCE_WIDTH,
            height=CAMERA_SOURCE_HEIGHT,
            fps=CAMERA_FPS,
        )
    except TypeError:
        return Sensor(width=CAMERA_SOURCE_WIDTH, height=CAMERA_SOURCE_HEIGHT)


def _apply_sensor_orientation(sensor):
    """应用项目公共摄像头方向配置。"""

    try:
        sensor.set_hmirror(CAMERA_HMIRROR)
    except Exception:
        pass

    try:
        sensor.set_vflip(CAMERA_VFLIP)
    except Exception:
        pass


def run_steelball_detect():
    """运行实时钢球检测演示。"""

    print("初始化钢球 YOLO11 检测")
    kmodel_path = resolve_kmodel_path()
    display_target, display_size, display_mode = resolve_display_config()
    print("模型路径：" + kmodel_path)
    print("显示目标：" + display_target)
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
            count = count_detections(dets)

            yolo.draw_result(dets, pl.osd_img)
            draw_count(pl.osd_img, count)
            pl.show_image()

            if GC_INTERVAL > 0 and frame_count % GC_INTERVAL == 0:
                gc.collect()

            if PRINT_INTERVAL > 0 and frame_count % PRINT_INTERVAL == 0:
                print("num=" + str(count) + " fps=" + str("%.1f" % clock.fps()))

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
    run_steelball_detect()
