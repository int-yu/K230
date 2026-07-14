"""可移植的 K230 彩色光点检测模块。

推荐在 num.py、tangle.py 或其他主程序中这样使用：

    from color import ColorSpotDetector

    # 放在主循环外，只初始化一次。
    color_detector = ColorSpotDetector()  # 参数默认从 config.py 读取

    # 放在取得 frame = image.to_numpy_ref() 之后。
    spot = color_detector.process(frame)  # 检测并绘制中心点

    if spot is not None:
        print(spot["center_x"], spot["center_y"])

运行时切换颜色：

    color_detector.set_color("green")

自定义颜色范围：

    custom_ranges = (((0, 100, 120), (12, 255, 255)),)
    color_detector.set_color(hsv_ranges=custom_ranges)

模块的检测核心不依赖 CameraIO、Display 或主程序全局变量。旧版的
detect_color_spot()、draw_spot_center() 和 run_color_tracking() 仍然保留。
"""

import math

import cv2

from config import (
    COLOR_DRAW_COLOR,
    COLOR_DRAW_CROSS_SIZE,
    COLOR_DRAW_RADIUS,
    COLOR_MAX_AREA,
    COLOR_MIN_AREA,
    COLOR_MIN_CONFIDENCE,
    COLOR_PRESET_HSV_RANGES,
    COLOR_TARGET,
)


# 保留旧名称，已有代码仍可从 color 导入 COLOR_HSV_RANGES。
COLOR_HSV_RANGES = COLOR_PRESET_HSV_RANGES


def _resolve_hsv_ranges(target_color, hsv_ranges):
    """返回已校验的预设或自定义 HSV 范围。"""
    if hsv_ranges is None:
        color_name = str(target_color).lower()
        if color_name not in COLOR_HSV_RANGES:
            raise ValueError(
                "未知颜色：{}，可用颜色：{}，或传入 hsv_ranges".format(
                    target_color,
                    ", ".join(COLOR_HSV_RANGES.keys()),
                )
            )
        ranges = COLOR_HSV_RANGES[color_name]
    else:
        color_name = "custom"
        ranges = hsv_ranges

    normalized = []
    for item in ranges:
        if len(item) != 2 or len(item[0]) != 3 or len(item[1]) != 3:
            raise ValueError("每个 HSV 范围必须是 ((H,S,V), (H,S,V))")

        lower = tuple(int(value) for value in item[0])
        upper = tuple(int(value) for value in item[1])
        if lower[0] < 0 or upper[0] > 179:
            raise ValueError("HSV 的 H 范围必须在 0..179")
        if min(lower[1], lower[2]) < 0:
            raise ValueError("HSV 的 S、V 下限不能小于 0")
        if max(upper[1], upper[2]) > 255:
            raise ValueError("HSV 的 S、V 上限不能大于 255")
        if (
            lower[0] > upper[0] or
            lower[1] > upper[1] or
            lower[2] > upper[2]
        ):
            raise ValueError("HSV 范围下限不能大于上限")
        normalized.append((lower, upper))

    if not normalized:
        raise ValueError("hsv_ranges 不能为空")
    return color_name, tuple(normalized)


def _validate_detector_limits(min_area, max_area, min_confidence):
    if min_area < 0:
        raise ValueError("min_area 不能小于 0")
    if max_area is not None and max_area < min_area:
        raise ValueError("max_area 不能小于 min_area")
    if min_confidence < 0 or min_confidence > 1:
        raise ValueError("min_confidence 必须在 0..1")


def _find_contours(mask):
    """兼容 OpenCV 两返回值和三返回值格式。"""
    result = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(result) == 2:
        return result[0]
    return result[1]


def _contour_center(contour, bbox):
    """优先使用轮廓矩，失败时退回外接框中心。"""
    x, y, width, height = bbox
    try:
        moments = cv2.moments(contour)
        m00 = moments["m00"]
        if m00:
            return (
                int(moments["m10"] / m00),
                int(moments["m01"] / m00),
            )
    except Exception:
        pass
    return (x + width // 2, y + height // 2)


def _spot_confidence(contour, area, bbox):
    """根据圆度和轮廓紧凑度计算 0..1 的结构评分。"""
    perimeter = cv2.arcLength(contour, True)
    width = bbox[2]
    height = bbox[3]
    if perimeter <= 0 or width <= 0 or height <= 0:
        return 0.0

    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    fill_ratio = area / float(width * height)
    circularity = min(1.0, max(0.0, circularity))
    fill_ratio = min(1.0, max(0.0, fill_ratio))
    return circularity * fill_ratio


def _make_color_mask(frame, hsv_ranges):
    """生成颜色掩膜，并自动合并红色等跨越 H 边界的范围。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = None
    for lower, upper in hsv_ranges:
        current = cv2.inRange(hsv, lower, upper)
        if mask is None:
            mask = current
        else:
            mask = cv2.bitwise_or(mask, current)
    return mask


def _detect_with_ranges(
    frame,
    hsv_ranges,
    min_area,
    max_area,
    min_confidence,
):
    mask = _make_color_mask(frame, hsv_ranges)
    contours = _find_contours(mask)

    best = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        bbox = cv2.boundingRect(contour)
        confidence = _spot_confidence(contour, area, bbox)
        if confidence < min_confidence or confidence <= 0:
            continue

        if (
            best is None or
            confidence > best["confidence"] or
            (
                confidence == best["confidence"] and
                area > best["area"]
            )
        ):
            center_x, center_y = _contour_center(contour, bbox)
            best = {
                # x/y 保持旧接口兼容；center_x/center_y 便于接入 tangle。
                "x": center_x,
                "y": center_y,
                "center_x": center_x,
                "center_y": center_y,
                "confidence": confidence,
                "area": area,
                "bbox": bbox,
            }

    return best


class ColorSpotDetector:
    """可重复使用的彩色光点检测器。

    不传参数时全部使用 config.py；也可只覆盖 target_color 或 min_area。
    颜色范围只在初始化或 set_color() 时解析，不会在每帧重复处理。
    """

    def __init__(
        self,
        target_color=COLOR_TARGET,
        min_area=COLOR_MIN_AREA,
        max_area=COLOR_MAX_AREA,
        hsv_ranges=None,
        min_confidence=COLOR_MIN_CONFIDENCE,
        draw_color=COLOR_DRAW_COLOR,
        draw_radius=COLOR_DRAW_RADIUS,
        draw_cross_size=COLOR_DRAW_CROSS_SIZE,
    ):
        _validate_detector_limits(min_area, max_area, min_confidence)
        if len(draw_color) != 3:
            raise ValueError("draw_color 必须包含 3 个颜色通道")
        if draw_radius <= 0 or draw_cross_size <= 0:
            raise ValueError("绘制半径和十字尺寸必须大于 0")
        self.min_area = min_area
        self.max_area = max_area
        self.min_confidence = min_confidence
        self.draw_color = tuple(draw_color)
        self.draw_radius = int(draw_radius)
        self.draw_cross_size = int(draw_cross_size)
        self.target_color = None
        self.hsv_ranges = None
        self.last_spot = None
        self.set_color(target_color, hsv_ranges)

    def set_color(self, target_color=COLOR_TARGET, hsv_ranges=None):
        """切换预设颜色或自定义 HSV 范围，并返回 self。"""
        color_name, normalized = _resolve_hsv_ranges(
            target_color,
            hsv_ranges,
        )
        self.target_color = color_name
        self.hsv_ranges = normalized
        self.last_spot = None
        return self

    def detect(self, frame):
        """检测当前帧中结构评分最高的目标，不修改画面。"""
        self.last_spot = _detect_with_ranges(
            frame,
            self.hsv_ranges,
            self.min_area,
            self.max_area,
            self.min_confidence,
        )
        return self.last_spot

    def draw(self, frame, spot=None, color=None):
        """绘制指定目标；spot 省略时绘制最近一次检测结果。"""
        if spot is None:
            spot = self.last_spot
        if color is None:
            color = self.draw_color
        draw_spot_center(
            frame,
            spot,
            color,
            radius=self.draw_radius,
            cross_size=self.draw_cross_size,
        )
        return spot

    def process(self, frame, draw=True):
        """检测一帧并按需绘制中心点，返回目标字典或 None。"""
        spot = self.detect(frame)
        if draw:
            self.draw(frame, spot)
        return spot


def detect_color_spot(
    frame,
    target_color=COLOR_TARGET,
    hsv_ranges=None,
    min_area=COLOR_MIN_AREA,
    max_area=COLOR_MAX_AREA,
    min_confidence=COLOR_MIN_CONFIDENCE,
):
    """兼容旧代码的一次性检测函数。

    连续视频建议改用 ColorSpotDetector，避免每帧重复解析颜色范围。
    """
    _validate_detector_limits(min_area, max_area, min_confidence)
    unused_color_name, ranges = _resolve_hsv_ranges(
        target_color,
        hsv_ranges,
    )
    return _detect_with_ranges(
        frame,
        ranges,
        min_area,
        max_area,
        min_confidence,
    )


def draw_spot_center(
    frame,
    spot,
    color=COLOR_DRAW_COLOR,
    radius=COLOR_DRAW_RADIUS,
    cross_size=COLOR_DRAW_CROSS_SIZE,
):
    """在目标中心绘制圆环和十字；spot 为 None 时不做任何操作。"""
    if spot is None:
        return None

    if "center_x" in spot:
        center_x = spot["center_x"]
    else:
        center_x = spot["x"]
    if "center_y" in spot:
        center_y = spot["center_y"]
    else:
        center_y = spot["y"]
    center = (center_x, center_y)
    cv2.circle(frame, center, radius, color, 2)
    cv2.line(
        frame,
        (center_x - cross_size, center_y),
        (center_x + cross_size, center_y),
        color,
        1,
    )
    cv2.line(
        frame,
        (center_x, center_y - cross_size),
        (center_x, center_y + cross_size),
        color,
        1,
    )
    return spot


def run_color_tracking(
    target_color=COLOR_TARGET,
    hsv_ranges=None,
    min_area=COLOR_MIN_AREA,
    max_area=COLOR_MAX_AREA,
    display_target=None,
):
    """独立运行示例；CameraIO 仅在调用本函数时才导入。"""
    import gc
    import sys

    from camera_io import CameraIO, DISPLAY_TARGET_BOARD

    if display_target is None:
        display_target = DISPLAY_TARGET_BOARD

    camera = None
    frame_count = 0
    detector = ColorSpotDetector(
        target_color=target_color,
        hsv_ranges=hsv_ranges,
        min_area=min_area,
        max_area=max_area,
    )

    try:
        print("================================")
        print("K230 彩色光点检测")
        print("追踪颜色：{}".format(detector.target_color))
        print("HSV 范围：{}".format(detector.hsv_ranges))
        print("显示目标：{}".format(display_target))
        print("================================")

        camera = CameraIO(display_target=display_target)
        camera.initialize()

        while True:
            image = camera.snapshot()
            frame = image.to_numpy_ref()
            detector.process(frame)
            camera.show_image(image)

            frame_count += 1
            if frame_count % 30 == 0:
                gc.collect()

            del frame
            del image

    except KeyboardInterrupt:
        print("用户停止程序")
    except Exception as error:
        sys.print_exception(error)
    finally:
        print("正在释放资源")
        if camera is not None:
            camera.deinitialize()
        gc.collect()
        print("程序结束")


if __name__ == "__main__":
    run_color_tracking()
