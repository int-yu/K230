"""K230 color spot detector.

Detect the most likely colored light spot in a frame and draw its center.
The default target is red; preset colors or custom HSV ranges can be passed
to ``detect_color_spot`` and ``run_color_tracking``.
"""

import gc
import math
import sys

import cv2

from camera_io import CameraIO, DISPLAY_TARGET_BOARD


# OpenCV HSV: H is 0..179, S and V are 0..255.  These presets are starting
# values and should be adjusted for the actual light source and environment.
COLOR_HSV_RANGES = {
    "red": (
        ((0, 120, 120), (10, 255, 255)),
        ((170, 120, 120), (179, 255, 255)),
    ),
    "green": (
        ((35, 80, 80), (85, 255, 255)),
    ),
    "blue": (
        ((90, 80, 80), (135, 255, 255)),
    ),
    "yellow": (
        ((20, 100, 100), (35, 255, 255)),
    ),
}


def _resolve_hsv_ranges(target_color, hsv_ranges):
    """Return validated HSV ranges for a preset or a custom color."""
    if hsv_ranges is None:
        color_name = str(target_color).lower()
        if color_name not in COLOR_HSV_RANGES:
            raise ValueError(
                "未知颜色：{}，可用颜色：{}，或传入 hsv_ranges".format(
                    target_color, ", ".join(COLOR_HSV_RANGES.keys())
                )
            )
        ranges = COLOR_HSV_RANGES[color_name]
    else:
        ranges = hsv_ranges

    normalized = []
    for item in ranges:
        if len(item) != 2 or len(item[0]) != 3 or len(item[1]) != 3:
            raise ValueError("每个 HSV 范围必须是 ((H,S,V), (H,S,V))")

        lower = tuple(int(value) for value in item[0])
        upper = tuple(int(value) for value in item[1])
        if lower[0] < 0 or upper[0] > 179:
            raise ValueError("HSV 的 H 范围必须在 0..179")
        if min(lower[1], lower[2]) < 0 or max(upper[1], upper[2]) > 255:
            raise ValueError("HSV 的 S、V 范围必须在 0..255")
        if lower[0] > upper[0] or lower[1] > upper[1] or lower[2] > upper[2]:
            raise ValueError("HSV 范围下限不能大于上限")
        normalized.append((lower, upper))

    if not normalized:
        raise ValueError("hsv_ranges 不能为空")
    return tuple(normalized)


def _find_contours(mask):
    """Handle both OpenCV two-value and three-value return formats."""
    result = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(result) == 2:
        return result[0]
    return result[1]


def _contour_center(contour, bbox):
    """Use contour moments, falling back to the bounding-box center."""
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
    """Score a light spot by circularity and contour compactness."""
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
    """Build one binary mask, including split hue ranges such as red."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    mask = None
    for lower, upper in hsv_ranges:
        current = cv2.inRange(hsv, lower, upper)
        if mask is None:
            mask = current
        else:
            mask = cv2.bitwise_or(mask, current)
    return mask


def detect_color_spot(
    frame,
    target_color="red",
    hsv_ranges=None,
    min_area=8,
    max_area=None,
):
    """Detect the highest-confidence colored light spot.

    Args:
        frame: RGB image returned by ``image.to_numpy_ref()``.
        target_color: Preset name: red, green, blue or yellow.
        hsv_ranges: Optional custom ranges in the form
            ``(((h1, s1, v1), (h2, s2, v2)), ...)``.
        min_area: Ignore contours smaller than this many pixels.
        max_area: Optional maximum contour area; ``None`` disables the limit.

    Returns:
        ``None`` if no candidate is found, otherwise a dictionary containing
        x, y, confidence, area and bbox.
    """
    if min_area < 0:
        raise ValueError("min_area 不能小于 0")
    if max_area is not None and max_area < min_area:
        raise ValueError("max_area 不能小于 min_area")

    ranges = _resolve_hsv_ranges(target_color, hsv_ranges)
    mask = _make_color_mask(frame, ranges)
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
        if confidence <= 0:
            continue

        if best is None or confidence > best["confidence"] or (
            confidence == best["confidence"] and area > best["area"]
        ):
            center_x, center_y = _contour_center(contour, bbox)
            best = {
                "x": center_x,
                "y": center_y,
                "confidence": confidence,
                "area": area,
                "bbox": bbox,
            }

    return best


def draw_spot_center(frame, spot, color=(255, 255, 255)):
    """Draw a visible ring and cross at the detected center."""
    if spot is None:
        return

    center = (spot["x"], spot["y"])
    cv2.circle(frame, center, 7, color, 2)
    cv2.line(
        frame,
        (center[0] - 10, center[1]),
        (center[0] + 10, center[1]),
        color,
        1,
    )
    cv2.line(
        frame,
        (center[0], center[1] - 10),
        (center[0], center[1] + 10),
        color,
        1,
    )


def run_color_tracking(
    target_color="red",
    hsv_ranges=None,
    min_area=8,
    max_area=None,
    display_target=DISPLAY_TARGET_BOARD,
):
    """Run continuous color-spot detection and display the marked image."""
    camera = None
    frame_count = 0
    try:
        ranges = _resolve_hsv_ranges(target_color, hsv_ranges)
        print("================================")
        print("K230 彩色光点检测")
        print("追踪颜色：{}".format(target_color))
        print("HSV 范围：{}".format(ranges))
        print("显示目标：{}".format(display_target))
        print("================================")

        camera = CameraIO(display_target=display_target)
        camera.initialize()

        while True:
            image = camera.snapshot()
            frame = image.to_numpy_ref()
            spot = detect_color_spot(
                frame,
                target_color=target_color,
                hsv_ranges=ranges,
                min_area=min_area,
                max_area=max_area,
            )
            draw_spot_center(frame, spot)
            camera.show_image(image)

            frame_count += 1
            if frame_count % 30 == 0:
                gc.collect()

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
    run_color_tracking(target_color="red")
