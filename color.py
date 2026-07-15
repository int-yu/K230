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

模块的检测核心不依赖 CameraIO、Display 或主程序全局变量。
draw_spot_center() 和 run_color_tracking() 可用于自定义绘制和独立演示。
"""

import math

import cv2

from config import (
    COLOR_DRAW_COLOR,
    COLOR_DRAW_CROSS_SIZE,
    COLOR_DRAW_RADIUS,
    COLOR_HIGHLIGHT_KERNEL_SIZE,
    COLOR_HIGHLIGHT_MAX_SATURATION,
    COLOR_HIGHLIGHT_MIN_VALUE,
    COLOR_INCLUDE_HIGHLIGHT,
    COLOR_MAX_AREA,
    COLOR_MIN_AREA,
    COLOR_MIN_CONFIDENCE,
    COLOR_PRESET_HSV_RANGES,
    COLOR_TARGET,
)


def _resolve_hsv_ranges(target_color, hsv_ranges):
    """返回已校验的预设或自定义 HSV 范围。"""
    if hsv_ranges is None:
        color_name = str(target_color).lower()
        if color_name not in COLOR_PRESET_HSV_RANGES:
            raise ValueError(
                "未知颜色：{}，可用颜色：{}，或传入 hsv_ranges".format(
                    target_color,
                    ", ".join(COLOR_PRESET_HSV_RANGES.keys()),
                )
            )
        ranges = COLOR_PRESET_HSV_RANGES[color_name]
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


def _make_highlight_kernel(
    include_highlight,
    max_saturation,
    min_value,
    kernel_size,
):
    """校验高亮补全参数，并只在启用时创建一次形态学核。"""
    if max_saturation < 0 or max_saturation > 255:
        raise ValueError("highlight_max_saturation 必须在 0..255")
    if min_value < 0 or min_value > 255:
        raise ValueError("highlight_min_value 必须在 0..255")
    if kernel_size <= 0 or kernel_size % 2 == 0:
        raise ValueError("highlight_kernel_size 必须是正奇数")
    if not include_highlight:
        return None
    return cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (int(kernel_size), int(kernel_size)),
    )


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


def _make_hsv_range_mask(hsv, hsv_ranges):
    """根据一个或多个 HSV 范围生成掩膜。"""
    mask = None
    for lower, upper in hsv_ranges:
        current = cv2.inRange(hsv, lower, upper)
        if mask is None:
            mask = current
        else:
            mask = cv2.bitwise_or(mask, current)
    return mask


def _make_base_color_mask(frame, hsv_ranges):
    """生成全图 HSV 和基础颜色掩膜，不在全图执行高亮补全。"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = _make_hsv_range_mask(hsv, hsv_ranges)
    return hsv, mask


def _best_spot_from_mask(
    mask,
    min_area,
    max_area,
    min_confidence,
    offset_x=0,
    offset_y=0,
):
    """返回掩膜中结构评分最高的候选，并把局部坐标映射到原图。"""
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
                "x": center_x + offset_x,
                "y": center_y + offset_y,
                "center_x": center_x + offset_x,
                "center_y": center_y + offset_y,
                "confidence": confidence,
                "area": area,
                "bbox": (
                    bbox[0] + offset_x,
                    bbox[1] + offset_y,
                    bbox[2],
                    bbox[3],
                ),
            }
    return best


def _expanded_roi(bbox, image_width, image_height, margin):
    """把候选框扩展 margin 像素并裁剪到画面范围。"""
    x, y, width, height = bbox
    x1 = max(0, int(x) - margin)
    y1 = max(0, int(y) - margin)
    x2 = min(image_width, int(x + width) + margin)
    y2 = min(image_height, int(y + height) + margin)
    return (x1, y1, x2, y2)


def _complete_highlight_in_candidate_roi(
    hsv,
    hsv_ranges,
    candidate,
    highlight_kernel,
    highlight_max_saturation,
    highlight_min_value,
    min_area,
    max_area,
    min_confidence,
):
    """只在当前帧最佳基础候选的 ROI 内补全过曝高亮像素。"""
    image_height = int(hsv.shape[0])
    image_width = int(hsv.shape[1])
    kernel_radius = int(highlight_kernel.shape[0]) // 2
    x1, y1, x2, y2 = _expanded_roi(
        candidate["bbox"],
        image_width,
        image_height,
        kernel_radius,
    )
    if x2 <= x1 or y2 <= y1:
        return None

    hsv_roi = hsv[y1:y2, x1:x2]
    # 不复用已经传给 findContours() 的全图掩膜，兼容可能原地修改
    # 输入的 OpenCV 固件；ROI 很小，重新生成基础颜色掩膜开销有限。
    base_roi = _make_hsv_range_mask(hsv_roi, hsv_ranges)
    highlight_mask = cv2.inRange(
        hsv_roi,
        (0, 0, int(highlight_min_value)),
        (179, int(highlight_max_saturation), 255),
    )
    color_neighborhood = cv2.dilate(base_roi, highlight_kernel)
    highlight_mask = cv2.bitwise_and(
        highlight_mask,
        color_neighborhood,
    )
    completed_mask = cv2.bitwise_or(base_roi, highlight_mask)
    best = _best_spot_from_mask(
        completed_mask,
        min_area,
        max_area,
        min_confidence,
        offset_x=x1,
        offset_y=y1,
    )

    del completed_mask
    del color_neighborhood
    del highlight_mask
    del base_roi
    del hsv_roi
    return best


def _detect_with_ranges(
    frame,
    hsv_ranges,
    min_area,
    max_area,
    min_confidence,
    highlight_kernel=None,
    highlight_max_saturation=COLOR_HIGHLIGHT_MAX_SATURATION,
    highlight_min_value=COLOR_HIGHLIGHT_MIN_VALUE,
):
    hsv, base_mask = _make_base_color_mask(frame, hsv_ranges)

    # 高亮补全前只按基础颜色结构评分选一个候选。此处暂不应用最终
    # min_confidence，避免白色过曝中心让基础颜色环的评分暂时偏低。
    base_candidate = _best_spot_from_mask(
        base_mask,
        min_area,
        max_area,
        0.0,
    )
    if base_candidate is None:
        return None

    if highlight_kernel is None:
        if base_candidate["confidence"] < min_confidence:
            return None
        return base_candidate

    return _complete_highlight_in_candidate_roi(
        hsv,
        hsv_ranges,
        base_candidate,
        highlight_kernel,
        highlight_max_saturation,
        highlight_min_value,
        min_area,
        max_area,
        min_confidence,
    )


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
        include_highlight=COLOR_INCLUDE_HIGHLIGHT,
        highlight_max_saturation=COLOR_HIGHLIGHT_MAX_SATURATION,
        highlight_min_value=COLOR_HIGHLIGHT_MIN_VALUE,
        highlight_kernel_size=COLOR_HIGHLIGHT_KERNEL_SIZE,
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
        self.include_highlight = bool(include_highlight)
        self.highlight_max_saturation = int(highlight_max_saturation)
        self.highlight_min_value = int(highlight_min_value)
        self.highlight_kernel_size = int(highlight_kernel_size)
        self._highlight_kernel = _make_highlight_kernel(
            self.include_highlight,
            self.highlight_max_saturation,
            self.highlight_min_value,
            self.highlight_kernel_size,
        )
        self.draw_color = tuple(draw_color)
        self.draw_radius = int(draw_radius)
        self.draw_cross_size = int(draw_cross_size)
        self.target_color = None
        self.hsv_ranges = None
        self.last_spot = None
        self._target_valid = False
        self._offset_x = 0
        self._offset_y = 0
        self.set_color(target_color, hsv_ranges)

    def _update_target_state(self, frame, spot):
        """更新供串口直接读取的当前帧目标状态。"""
        if spot is None:
            self._target_valid = False
            self._offset_x = 0
            self._offset_y = 0
            return

        self._target_valid = True
        self._offset_x = int(frame.shape[1]) // 2 - int(spot["center_x"])
        self._offset_y = int(frame.shape[0]) // 2 - int(spot["center_y"])

    def set_color(self, target_color=COLOR_TARGET, hsv_ranges=None):
        """切换预设颜色或自定义 HSV 范围，并返回 self。"""
        color_name, normalized = _resolve_hsv_ranges(
            target_color,
            hsv_ranges,
        )
        self.target_color = color_name
        self.hsv_ranges = normalized
        self.last_spot = None
        self._target_valid = False
        self._offset_x = 0
        self._offset_y = 0
        return self

    def detect(self, frame):
        """检测当前帧中结构评分最高的目标，不修改画面。"""
        self.last_spot = _detect_with_ranges(
            frame,
            self.hsv_ranges,
            self.min_area,
            self.max_area,
            self.min_confidence,
            self._highlight_kernel,
            self.highlight_max_saturation,
            self.highlight_min_value,
        )
        self._update_target_state(frame, self.last_spot)
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
            offset_x=(
                self._offset_x if spot is self.last_spot else None
            ),
            offset_y=(
                self._offset_y if spot is self.last_spot else None
            ),
        )
        return spot

    def process(self, frame, draw=True):
        """检测一帧并按需绘制中心点，返回目标字典或 None。"""
        spot = self.detect(frame)
        if draw:
            self.draw(frame, spot)
        return spot


def draw_spot_center(
    frame,
    spot,
    color=COLOR_DRAW_COLOR,
    radius=COLOR_DRAW_RADIUS,
    cross_size=COLOR_DRAW_CROSS_SIZE,
    offset_x=None,
    offset_y=None,
):
    """绘制目标、画面中心、连线和目标相对画面中心的偏差。"""
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
    image_width = int(frame.shape[1])
    image_height = int(frame.shape[0])
    image_center_x = image_width // 2
    image_center_y = image_height // 2
    image_center = (image_center_x, image_center_y)
    if offset_x is None:
        offset_x = image_center_x - center_x
    if offset_y is None:
        offset_y = image_center_y - center_y

    cv2.line(frame, image_center, center, color, 1)
    cv2.line(
        frame,
        (image_center_x - cross_size, image_center_y),
        (image_center_x + cross_size, image_center_y),
        color,
        1,
    )
    cv2.line(
        frame,
        (image_center_x, image_center_y - cross_size),
        (image_center_x, image_center_y + cross_size),
        color,
        1,
    )
    cv2.circle(frame, image_center, 3, color, -1)
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
    label_x = min(max(0, center_x + 8), max(0, image_width - 145))
    label_y = max(20, center_y - 8)
    cv2.putText(
        frame,
        "X:{} Y:{}".format(offset_x, offset_y),
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        2,
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
    import time

    from camera_io import CameraIO, DISPLAY_TARGET_IDE

    if display_target is None:
        display_target = DISPLAY_TARGET_IDE

    camera = None
    tracking_uart = None
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
        from uart_io import TrackingUART
        tracking_uart = TrackingUART().initialize()
        clock = time.clock()
        while True:
            clock.tick()
            image = camera.snapshot()
            frame = image.to_numpy_ref()
            detector.process(frame)
            tracking_uart.send_target(
                detector._target_valid,
                detector._offset_x,
                detector._offset_y,
                frame_id=frame_count,
            )
            fps = clock.fps()
            cv2.putText(
                frame,
                "FPS: {:.1f}".format(fps),
                (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
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
        if tracking_uart is not None:
            tracking_uart.deinitialize()
        if camera is not None:
            camera.deinitialize()
        gc.collect()
        print("程序结束")


if __name__ == "__main__":
    run_color_tracking()
