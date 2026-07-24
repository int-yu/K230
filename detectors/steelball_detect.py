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

import gc
import os
import sys
import time

# CanMV 按绝对路径启动子目录脚本时不会自动加入项目根目录。
# 同时兼容把整个项目放在 /sdcard/K230 或直接放在 /sdcard。
for _path in ("/sdcard/K230", "/sdcard"):
    if _path not in sys.path:
        sys.path.append(_path)

try:
    from libs.PipeLine import PipeLine
    from libs.YOLO import YOLO11
    from media.sensor import Sensor
except Exception:
    PipeLine = None
    YOLO11 = None
    Sensor = None

from core.draw_utils import draw_osd_status

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
    STEELBALL_MAX_ASPECT_RATIO,
    STEELBALL_MAX_BOXES_NUM,
    STEELBALL_MODEL_INPUT_SIZE,
    STEELBALL_NMS_THRESHOLD,
    STEELBALL_RGB888P_SIZE,
)


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


def _as_list(value):
    """把 ulab/numpy 数组或元组转换为普通列表，失败时保持原值。"""

    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if tolist is not None:
        try:
            return tolist()
        except Exception:
            pass
    return value


def _is_sequence(value):
    return isinstance(value, (list, tuple)) or _shape(value) is not None


def _to_float(value):
    try:
        return float(value)
    except Exception:
        return None


def _valid_label(class_id):
    """判断类别 id 是否属于钢球类别。"""

    if class_id is None:
        return True
    try:
        return int(class_id) in STEELBALL_LABELS
    except Exception:
        return False


def _make_detection(x1, y1, x2, y2, confidence, class_id=None):
    """生成统一检测结果；坐标或类别无效时返回 None。"""

    x1 = _to_float(x1)
    y1 = _to_float(y1)
    x2 = _to_float(x2)
    y2 = _to_float(y2)
    confidence = _to_float(confidence)
    if (
        x1 is None or y1 is None or x2 is None or y2 is None or
        confidence is None
    ):
        return None
    if confidence < 0.0 or confidence > 1.5:
        return None
    if not _valid_label(class_id):
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    width = x2 - x1
    height = y2 - y1
    if width <= 0 or height <= 0:
        return None
    aspect_ratio = max(width / height, height / width)
    if aspect_ratio > STEELBALL_MAX_ASPECT_RATIO:
        return None
    if confidence < STEELBALL_CONFIDENCE_THRESHOLD:
        return None

    class_id = 0 if class_id is None else int(class_id)
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "bbox": (x1, y1, x2, y2),
        "rect": (x1, y1, width, height),
        "bbox_w": width,
        "bbox_h": height,
        "aspect_ratio": aspect_ratio,
        "center_x": (x1 + x2) / 2.0,
        "center_y": (y1 + y2) / 2.0,
        "confidence": confidence,
        "class_id": class_id,
        "label": STEELBALL_LABELS.get(class_id, str(class_id)),
    }


def _make_detection_xywh(x, y, width, height, confidence, class_id=None):
    """根据 x, y, w, h 生成统一检测结果。"""

    x = _to_float(x)
    y = _to_float(y)
    width = _to_float(width)
    height = _to_float(height)
    if x is None or y is None or width is None or height is None:
        return None
    if width <= 0 or height <= 0:
        return None
    return _make_detection(
        x,
        y,
        x + width,
        y + height,
        confidence,
        class_id,
    )


def _make_detection_from_box(box, confidence, class_id=None, prefer_xywh=False):
    """根据 4 元 box 生成统一检测结果。

    01Studio YOLO 的三元组返回值中，box 实测为 [x, y, w, h]。
    其他路径保留对 [x1, y1, x2, y2] 的兼容。
    """

    box = _as_list(box)
    if box is None or len(box) < 4:
        return None

    if prefer_xywh:
        detection = _make_detection_xywh(
            box[0],
            box[1],
            box[2],
            box[3],
            confidence,
            class_id,
        )
        if detection is not None:
            return detection

    if (
        _to_float(box[2]) is not None and
        _to_float(box[3]) is not None and
        _to_float(box[0]) is not None and
        _to_float(box[1]) is not None and
        (float(box[2]) <= float(box[0]) or float(box[3]) <= float(box[1]))
    ):
        detection = _make_detection_xywh(
            box[0],
            box[1],
            box[2],
            box[3],
            confidence,
            class_id,
        )
        if detection is not None:
            return detection

    detection = _make_detection(box[0], box[1], box[2], box[3], confidence, class_id)
    if detection is not None:
        return detection

    return _make_detection_xywh(box[0], box[1], box[2], box[3], confidence, class_id)


def _parse_detection_dict(item):
    """解析 dict 形式检测结果。"""

    try:
        confidence = item.get("confidence", item.get("score", item.get("conf", 1.0)))
        class_id = item.get("class_id", item.get("class", item.get("cls", 0)))

        if "bbox" in item:
            box = _as_list(item["bbox"])
        elif "box" in item:
            box = _as_list(item["box"])
        elif "rect" in item:
            rect = _as_list(item["rect"])
            if rect is not None and len(rect) >= 4:
                return _make_detection(
                    rect[0],
                    rect[1],
                    float(rect[0]) + float(rect[2]),
                    float(rect[1]) + float(rect[3]),
                    confidence,
                    class_id,
                )
            return None
        else:
            box = None

        if box is not None and len(box) >= 4:
            return _make_detection_from_box(box, confidence, class_id)

        if (
            "x1" in item and "y1" in item and
            "x2" in item and "y2" in item
        ):
            return _make_detection(
                item["x1"],
                item["y1"],
                item["x2"],
                item["y2"],
                confidence,
                class_id,
            )

        if "x" in item and "y" in item and "w" in item and "h" in item:
            return _make_detection(
                item["x"],
                item["y"],
                float(item["x"]) + float(item["w"]),
                float(item["y"]) + float(item["h"]),
                confidence,
                class_id,
            )
    except Exception:
        return None
    return None


def _parse_detection_row(row):
    """解析常见 YOLO 行格式。

    支持：
    - [class_id, score, x1, y1, x2, y2]
    - [x1, y1, x2, y2, score, class_id]
    """

    row = _as_list(row)
    if row is None or not isinstance(row, list) or len(row) < 6:
        return None

    class_first = _make_detection(row[2], row[3], row[4], row[5], row[1], row[0])
    if class_first is not None:
        return class_first

    class_box_score_last = _make_detection(
        row[1],
        row[2],
        row[3],
        row[4],
        row[5],
        row[0],
    )
    if class_box_score_last is not None:
        return class_box_score_last

    try:
        class_first_rect = _make_detection(
            row[2],
            row[3],
            float(row[2]) + float(row[4]),
            float(row[3]) + float(row[5]),
            row[1],
            row[0],
        )
        if class_first_rect is not None:
            return class_first_rect
    except Exception:
        pass

    try:
        class_rect_score_last = _make_detection(
            row[1],
            row[2],
            float(row[1]) + float(row[3]),
            float(row[2]) + float(row[4]),
            row[5],
            row[0],
        )
        if class_rect_score_last is not None:
            return class_rect_score_last
    except Exception:
        pass

    box_first = _make_detection(row[0], row[1], row[2], row[3], row[4], row[5])
    if box_first is not None:
        return box_first

    try:
        box_first_rect = _make_detection(
            row[0],
            row[1],
            float(row[0]) + float(row[2]),
            float(row[1]) + float(row[3]),
            row[4],
            row[5],
        )
        if box_first_rect is not None:
            return box_first_rect
    except Exception:
        pass

    return None


def _looks_like_boxes(value):
    value = _as_list(value)
    count = _safe_len(value)
    if value is None or count is None or count <= 0:
        return False
    try:
        first = _as_list(value[0])
        return first is not None and len(first) >= 4
    except Exception:
        return False


def _looks_like_scores(value, expected_count):
    value = _as_list(value)
    count = _safe_len(value)
    if count is None or count != expected_count:
        return False
    try:
        for index in range(count):
            score = _to_float(value[index])
            if score is None:
                return False
            if score < 0.0 or score > 1.5:
                return False
        return True
    except Exception:
        return False


def _score_candidate_quality(value):
    """分数列表候选质量；用于区分类别 id 列表和 confidence 列表。"""

    value = _as_list(value)
    count = _safe_len(value)
    if count is None or count <= 0:
        return -1.0
    total = 0.0
    above_threshold = 0
    non_integer = 0
    try:
        for index in range(count):
            score = _to_float(value[index])
            if score is None:
                return -1.0
            total += score
            if score >= STEELBALL_CONFIDENCE_THRESHOLD:
                above_threshold += 1
            if abs(score - int(score)) > 0.000001:
                non_integer += 1
    except Exception:
        return -1.0
    return above_threshold * 10.0 + non_integer + total / float(count)


def _normalize_triplet_detections(dets):
    """解析 (boxes, scores, ids)、(ids, scores, boxes) 等分离返回值。"""

    dets = _as_list(dets)
    if dets is None or not isinstance(dets, list) or len(dets) not in (2, 3):
        return None

    boxes = None
    box_index = -1
    box_count = None
    for index in range(len(dets)):
        if _looks_like_boxes(dets[index]):
            boxes = _as_list(dets[index])
            box_index = index
            box_count = _safe_len(boxes)
            break

    if boxes is None or box_count is None:
        return None

    scores = None
    score_index = -1
    score_quality = -1.0
    for index in range(len(dets)):
        if index == box_index:
            continue
        if _looks_like_scores(dets[index], box_count):
            current_quality = _score_candidate_quality(dets[index])
            if current_quality > score_quality:
                scores = _as_list(dets[index])
                score_index = index
                score_quality = current_quality

    if scores is None:
        return None

    ids = None
    for index in range(len(dets)):
        if index != box_index and index != score_index:
            ids = _as_list(dets[index])
            break

    results = []
    ids_len = _safe_len(ids)
    for index in range(box_count):
        box = _as_list(boxes[index])
        if box is None or len(box) < 4:
            continue
        class_id = 0
        if ids is not None and ids_len is not None and index < ids_len:
            class_id = ids[index]
        detection = _make_detection_from_box(
            box,
            scores[index],
            class_id,
            prefer_xywh=True,
        )
        if detection is not None:
            results.append(detection)
    return results


def normalize_detections(dets):
    """把 YOLO 返回结果标准化为检测框列表。

    只统计成功解析且通过置信度、类别和坐标检查的检测框；不再把未知容器长度
    当成数量，避免左上角恒显示 Num: 2 或 Num: 3。
    """

    if dets is None:
        return []

    triplet = _normalize_triplet_detections(dets)
    if triplet is not None:
        return triplet

    if isinstance(dets, dict):
        detection = _parse_detection_dict(dets)
        return [] if detection is None else [detection]

    row_detection = _parse_detection_row(dets)
    if row_detection is not None:
        return [row_detection]

    dets = _as_list(dets)
    if dets is None or not _is_sequence(dets):
        return []

    if isinstance(dets, list) and len(dets) == 1:
        nested = _as_list(dets[0])
        if nested is not dets[0] or isinstance(nested, (list, tuple)):
            nested_results = normalize_detections(nested)
            if nested_results:
                return nested_results

    results = []
    try:
        for item in dets:
            detection = None
            if isinstance(item, dict):
                detection = _parse_detection_dict(item)
            else:
                detection = _parse_detection_row(item)
            if detection is not None:
                results.append(detection)
    except Exception:
        return []
    return results


def count_detections(dets):
    """统计真实检测框数量。"""

    return len(normalize_detections(dets))


def _create_sensor():
    """按项目公共摄像头配置创建 Sensor。"""

    if Sensor is None:
        raise RuntimeError("当前环境缺少 media.sensor.Sensor，只能在 CanMV/K230 上运行")

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

    if PipeLine is None or YOLO11 is None:
        raise RuntimeError("当前环境缺少 libs.PipeLine 或 libs.YOLO，只能在 CanMV/K230 上运行")

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

            yolo.draw_result(dets, pl.osd_img)
            detections = normalize_detections(dets)
            count = len(detections)

            fps = clock.fps()
            draw_osd_status(pl.osd_img, fps=fps, count=count, count_label="Num")
            pl.show_image()

            if GC_INTERVAL > 0 and frame_count % GC_INTERVAL == 0:
                gc.collect()

            if PRINT_INTERVAL > 0 and frame_count % PRINT_INTERVAL == 0:
                print("FPS: {:.1f}, Num: {}".format(fps, count))

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
