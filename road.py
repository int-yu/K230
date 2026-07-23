"""K230 T 型、十字型和黑色双排虚线结束符检测模块。

调用方式：

    from road import RoadSymbolDetector

    detector = RoadSymbolDetector()
    result = detector.process(frame)

模块不做 HSV 红色识别。场地前景只有红色和黑色，因此 T/十字使用
绿色通道 Otsu 提取前景，再选择真正横跨左右并向下延伸的主连通轮廓。
黑色双排虚线使用独立 RGB 黑色阈值检测。没有当前帧结果时返回 None，
不保留上一帧结果，不执行预测或时间平滑。
"""

import cv2

try:
    import ulab.numpy as np
except ImportError:
    import numpy as np

import sys

# CanMV 按绝对路径启动脚本时不会把脚本所在目录加入 sys.path，
# 会导致 import config 失败。这里补上，重复导入不会重复追加。
if "/sdcard/K230" not in sys.path:
    sys.path.append("/sdcard/K230")

from config import (
    ROAD_ARM_MIN_OCCUPANCY,
    ROAD_BLACK_MAX_VALUE,
    ROAD_BLACK_MORPH_KERNEL_SIZE,
    ROAD_CROSS_TOP_MAX_RATIO,
    ROAD_DEMO_GC_INTERVAL,
    ROAD_DASH_MAX_ASPECT_RATIO,
    ROAD_DASH_MAX_HEIGHT_RATIO,
    ROAD_DASH_MAX_ROW_SEPARATION_RATIO,
    ROAD_DASH_MAX_WIDTH_RATIO,
    ROAD_DASH_MIN_AREA_RATIO,
    ROAD_DASH_MIN_ASPECT_RATIO,
    ROAD_DASH_MIN_COUNT_PER_ROW,
    ROAD_DASH_MIN_FILL_RATIO,
    ROAD_DASH_MIN_HEIGHT_RATIO,
    ROAD_DASH_MIN_HORIZONTAL_OVERLAP,
    ROAD_DASH_MIN_ROW_SEPARATION_RATIO,
    ROAD_DASH_MIN_WIDTH_RATIO,
    ROAD_DASH_ROW_TOLERANCE_RATIO,
    ROAD_DETECT_HEIGHT,
    ROAD_DETECT_WIDTH,
    ROAD_DRAW_DASH_COLOR,
    ROAD_DRAW_ENDPOINT_COLOR,
    ROAD_DRAW_FONT_SCALE,
    ROAD_DRAW_INTERSECTION_COLOR,
    ROAD_DRAW_POINT_RADIUS,
    ROAD_DRAW_SEGMENT_COLOR,
    ROAD_DRAW_TEXT_COLOR,
    ROAD_DRAW_THICKNESS,
    ROAD_END_MIN_PATH_LENGTH_RATIO,
    ROAD_FOREGROUND_MORPH_KERNEL_SIZE,
    ROAD_PATH_CORRIDOR_HALF_WIDTH_RATIO,
    ROAD_PATH_MIN_PIXELS_RATIO,
    ROAD_ROUTE_DOWN_MIN_RATIO,
    ROAD_ROUTE_LEFT_MAX_RATIO,
    ROAD_ROUTE_MAX_TOP_RATIO,
    ROAD_ROUTE_MIN_AREA_RATIO,
    ROAD_ROUTE_RIGHT_MIN_RATIO,
    ROAD_ROI_CENTER,
    ROAD_ROI_DOWN,
    ROAD_ROI_LEFT,
    ROAD_ROI_RIGHT,
    ROAD_ROI_UP,
)


_ARM_ROIS = (
    ("up", ROAD_ROI_UP),
    ("down", ROAD_ROI_DOWN),
    ("left", ROAD_ROI_LEFT),
    ("right", ROAD_ROI_RIGHT),
    ("center", ROAD_ROI_CENTER),
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


def _clamp(value, lower, upper):
    return min(upper, max(lower, int(round(value))))


def _ratio_rect(width, height, ratio):
    x1 = _clamp(ratio[0] * width, 0, max(0, width - 1))
    y1 = _clamp(ratio[1] * height, 0, max(0, height - 1))
    x2 = _clamp(ratio[2] * width, x1 + 1, width)
    y2 = _clamp(ratio[3] * height, y1 + 1, height)
    return (x1, y1, x2, y2)


def _pixel_rect(width, height, x1, y1, x2, y2):
    x1 = _clamp(x1, 0, max(0, width - 1))
    y1 = _clamp(y1, 0, max(0, height - 1))
    x2 = _clamp(x2, x1 + 1, width)
    y2 = _clamp(y2, y1 + 1, height)
    return (x1, y1, x2, y2)


def _mask_occupancy(mask, rect):
    x1, y1, x2, y2 = rect
    area = (x2 - x1) * (y2 - y1)
    if area <= 0:
        return 0.0
    return cv2.countNonZero(mask[y1:y2, x1:x2]) / float(area)


def _mask_centroid(mask, rect):
    x1, y1, x2, y2 = rect
    roi = mask[y1:y2, x1:x2]
    if cv2.countNonZero(roi) <= 0:
        return None
    moments = cv2.moments(roi)
    m00 = moments["m00"]
    if not m00:
        return None
    return (
        x1 + moments["m10"] / m00,
        y1 + moments["m01"] / m00,
    )


def _mask_bounds(mask, rect, min_contour_area):
    """返回区域内合格连通块的整体边界。"""
    x1, y1, x2, y2 = rect
    contours = _find_contours(mask[y1:y2, x1:x2])
    left = None
    top = None
    right = None
    bottom = None
    for contour in contours:
        if cv2.contourArea(contour) < min_contour_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        current_left = x1 + x
        current_top = y1 + y
        current_right = current_left + width - 1
        current_bottom = current_top + height - 1
        left = current_left if left is None else min(left, current_left)
        top = current_top if top is None else min(top, current_top)
        right = current_right if right is None else max(right, current_right)
        bottom = (
            current_bottom if bottom is None else max(bottom, current_bottom)
        )
    if left is None:
        return None
    return (left, top, right, bottom)


def _line_intersection(first, second):
    x1, y1 = first[0]
    x2, y2 = first[1]
    x3, y3 = second[0]
    x4, y4 = second[1]
    denominator = (x1 - x2) * (y3 - y4) - (
        (y1 - y2) * (x3 - x4)
    )
    if abs(denominator) < 0.000001:
        return None
    first_cross = x1 * y2 - y1 * x2
    second_cross = x3 * y4 - y3 * x4
    return (
        (
            first_cross * (x3 - x4) -
            (x1 - x2) * second_cross
        ) / denominator,
        (
            first_cross * (y3 - y4) -
            (y1 - y2) * second_cross
        ) / denominator,
    )


def _point_at_x(line, x):
    x1, y1 = line[0]
    x2, y2 = line[1]
    if abs(x2 - x1) < 0.000001:
        return (x, (y1 + y2) * 0.5)
    ratio = (x - x1) / float(x2 - x1)
    return (x, y1 + ratio * (y2 - y1))


def _point_at_y(line, y):
    x1, y1 = line[0]
    x2, y2 = line[1]
    if abs(y2 - y1) < 0.000001:
        return ((x1 + x2) * 0.5, y)
    ratio = (y - y1) / float(y2 - y1)
    return (x1 + ratio * (x2 - x1), y)


def _integer_point(point, width, height):
    return (
        _clamp(point[0], 0, width - 1),
        _clamp(point[1], 0, height - 1),
    )


def _normalized_presence_score(value, threshold):
    if threshold <= 0:
        return 1.0
    return min(1.0, max(0.0, value / (threshold * 2.0)))


def _scaled_point(point, scale_x, scale_y):
    return (
        int(round(point[0] * scale_x)),
        int(round(point[1] * scale_y)),
    )


def _scale_result(result, scale_x, scale_y):
    if result is None or (scale_x == 1.0 and scale_y == 1.0):
        return result
    result["center_x"] = int(round(result["center_x"] * scale_x))
    result["center_y"] = int(round(result["center_y"] * scale_y))
    if result["intersection"] is not None:
        result["intersection"] = _scaled_point(
            result["intersection"],
            scale_x,
            scale_y,
        )
    result["endpoints"] = {
        name: _scaled_point(point, scale_x, scale_y)
        for name, point in result["endpoints"].items()
    }
    result["segments"] = tuple(
        (
            _scaled_point(start, scale_x, scale_y),
            _scaled_point(end, scale_x, scale_y),
        )
        for start, end in result["segments"]
    )
    result["dash_lines"] = tuple(
        (
            _scaled_point(start, scale_x, scale_y),
            _scaled_point(end, scale_x, scale_y),
        )
        for start, end in result["dash_lines"]
    )
    return result


class RoadSymbolDetector:
    """识别 T、十字和黑色双排虚线 END。"""

    def __init__(
        self,
        detect_width=ROAD_DETECT_WIDTH,
        detect_height=ROAD_DETECT_HEIGHT,
        arm_min_occupancy=ROAD_ARM_MIN_OCCUPANCY,
        black_max_value=ROAD_BLACK_MAX_VALUE,
        draw_segment_color=ROAD_DRAW_SEGMENT_COLOR,
        draw_endpoint_color=ROAD_DRAW_ENDPOINT_COLOR,
        draw_intersection_color=ROAD_DRAW_INTERSECTION_COLOR,
        draw_dash_color=ROAD_DRAW_DASH_COLOR,
        draw_text_color=ROAD_DRAW_TEXT_COLOR,
        draw_thickness=ROAD_DRAW_THICKNESS,
        draw_point_radius=ROAD_DRAW_POINT_RADIUS,
        draw_font_scale=ROAD_DRAW_FONT_SCALE,
    ):
        if detect_width <= 0 or detect_height <= 0:
            raise ValueError("检测分辨率必须大于 0")
        if arm_min_occupancy <= 0 or arm_min_occupancy > 1:
            raise ValueError("arm_min_occupancy 必须在 0..1")
        if black_max_value < 0 or black_max_value > 255:
            raise ValueError("black_max_value 必须在 0..255")
        if draw_thickness <= 0 or draw_point_radius <= 0:
            raise ValueError("绘制线宽和点半径必须大于 0")

        self.detect_width = int(detect_width)
        self.detect_height = int(detect_height)
        self.arm_min_occupancy = float(arm_min_occupancy)
        self.black_max_value = int(black_max_value)
        self.draw_segment_color = tuple(draw_segment_color)
        self.draw_endpoint_color = tuple(draw_endpoint_color)
        self.draw_intersection_color = tuple(draw_intersection_color)
        self.draw_dash_color = tuple(draw_dash_color)
        self.draw_text_color = tuple(draw_text_color)
        self.draw_thickness = int(draw_thickness)
        self.draw_point_radius = int(draw_point_radius)
        self.draw_font_scale = float(draw_font_scale)

        self._foreground_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (
                ROAD_FOREGROUND_MORPH_KERNEL_SIZE,
                ROAD_FOREGROUND_MORPH_KERNEL_SIZE,
            ),
        )
        self._black_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (ROAD_BLACK_MORPH_KERNEL_SIZE, ROAD_BLACK_MORPH_KERNEL_SIZE),
        )
        self._route_mask = np.zeros(
            (self.detect_height, self.detect_width),
            dtype=np.uint8,
        )
        self.last_result = None
        self.last_foreground_threshold = 0.0
        self._target_valid = False
        self._offset_x = 0
        self._offset_y = 0

    def _update_target_state(self, frame, result):
        if result is None:
            self._target_valid = False
            self._offset_x = 0
            self._offset_y = 0
            return
        self._target_valid = True
        self._offset_x = int(frame.shape[1]) // 2 - int(result["center_x"])
        self._offset_y = int(frame.shape[0]) // 2 - int(result["center_y"])

    def _foreground_mask(self, frame):
        """以绿色通道 Otsu 提取红色和黑色共同前景。"""
        green_channel = frame[:, :, 1]
        threshold_value, foreground = cv2.threshold(
            green_channel,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
        foreground = cv2.morphologyEx(
            foreground,
            cv2.MORPH_CLOSE,
            self._foreground_kernel,
        )
        return float(threshold_value), foreground

    def _route_candidate(self, foreground, width, height):
        """选择横跨左右、经过中心附近并向下延伸的主连通轮廓。"""
        frame_area = float(width * height)
        best = None
        for contour in _find_contours(foreground):
            area = cv2.contourArea(contour)
            if area < frame_area * ROAD_ROUTE_MIN_AREA_RATIO:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            right = x + box_width - 1
            bottom = y + box_height - 1
            if x > width * ROAD_ROUTE_LEFT_MAX_RATIO:
                continue
            if right < width * ROAD_ROUTE_RIGHT_MIN_RATIO:
                continue
            if bottom < height * ROAD_ROUTE_DOWN_MIN_RATIO:
                continue
            if y > height * ROAD_ROUTE_MAX_TOP_RATIO:
                continue

            score = (
                box_width / float(width) +
                box_height / float(height) +
                min(1.0, area / frame_area * 4.0)
            )
            if best is None or score > best["score"]:
                best = {
                    "contour": contour,
                    "bbox": (x, y, box_width, box_height),
                    "area": float(area),
                    "score": float(score),
                }
        return best

    def _candidate_mask(self, height, width, contour):
        """复用固定缓冲区，只保留当前主轮廓。"""
        cv2.rectangle(
            self._route_mask,
            (0, 0),
            (width - 1, height - 1),
            0,
            -1,
        )
        cv2.drawContours(self._route_mask, [contour], -1, 255, -1)
        return self._route_mask

    def _arm_scores(self, route_mask, width, height):
        scores = {}
        for name, ratio in _ARM_ROIS:
            scores[name] = _mask_occupancy(
                route_mask,
                _ratio_rect(width, height, ratio),
            )
        return scores

    @staticmethod
    def _horizontal_line(route_mask, width, height):
        left = _mask_centroid(
            route_mask,
            _ratio_rect(width, height, ROAD_ROI_LEFT),
        )
        right = _mask_centroid(
            route_mask,
            _ratio_rect(width, height, ROAD_ROI_RIGHT),
        )
        if left is None or right is None:
            return None
        return (left, right)

    @staticmethod
    def _cross_vertical_line(route_mask, width, height):
        upper = _mask_centroid(
            route_mask,
            _ratio_rect(width, height, ROAD_ROI_UP),
        )
        lower = _mask_centroid(
            route_mask,
            _ratio_rect(width, height, ROAD_ROI_DOWN),
        )
        if upper is None or lower is None:
            return None
        return (upper, lower)

    @staticmethod
    def _vertical_line(route_mask, width, height, preferred_x=None):
        if preferred_x is None:
            x1 = width * 0.20
            x2 = width * 0.80
        else:
            half_width = width * ROAD_PATH_CORRIDOR_HALF_WIDTH_RATIO
            x1 = preferred_x - half_width
            x2 = preferred_x + half_width
        upper = _mask_centroid(
            route_mask,
            _pixel_rect(
                width,
                height,
                x1,
                height * 0.48,
                x2,
                height * 0.72,
            ),
        )
        lower = _mask_centroid(
            route_mask,
            _pixel_rect(
                width,
                height,
                x1,
                height * 0.72,
                x2,
                height,
            ),
        )
        if upper is not None and lower is not None:
            return (upper, lower)
        available = upper if upper is not None else lower
        if available is None:
            return None
        return (
            (available[0], max(0.0, available[1] - 1.0)),
            (available[0], min(height - 1.0, available[1] + 1.0)),
        )

    def _route_result(
        self,
        symbol,
        route_mask,
        candidate,
        width,
        height,
        foreground_threshold,
    ):
        arm_scores = self._arm_scores(route_mask, width, height)
        required = ("left", "right", "down", "center")
        if any(
            arm_scores[name] < self.arm_min_occupancy
            for name in required
        ):
            return None
        if symbol == "cross" and (
            arm_scores["up"] < self.arm_min_occupancy
        ):
            return None
        if symbol == "t" and (
            arm_scores["up"] >= self.arm_min_occupancy
        ):
            return None

        horizontal = self._horizontal_line(route_mask, width, height)
        vertical = (
            self._cross_vertical_line(route_mask, width, height)
            if symbol == "cross"
            else self._vertical_line(route_mask, width, height)
        )
        if horizontal is None or vertical is None:
            return None
        intersection = _line_intersection(horizontal, vertical)
        if intersection is None:
            return None
        intersection = _integer_point(intersection, width, height)

        x, y, box_width, box_height = candidate["bbox"]
        left_x = x
        right_x = x + box_width - 1
        up_y = y
        down_y = y + box_height - 1
        endpoints = {
            "left": _integer_point(
                _point_at_x(horizontal, left_x),
                width,
                height,
            ),
            "right": _integer_point(
                _point_at_x(horizontal, right_x),
                width,
                height,
            ),
            "down": _integer_point(
                _point_at_y(vertical, down_y),
                width,
                height,
            ),
        }
        directions = ("left", "right", "down")
        if symbol == "cross":
            endpoints["up"] = _integer_point(
                _point_at_y(vertical, up_y),
                width,
                height,
            )
            directions = ("up", "down", "left", "right")

        confidence_parts = [
            _normalized_presence_score(
                arm_scores[name],
                self.arm_min_occupancy,
            )
            for name in directions
        ]
        confidence_parts.append(
            _normalized_presence_score(
                arm_scores["center"],
                self.arm_min_occupancy,
            )
        )
        if symbol == "t":
            confidence_parts.append(
                1.0 - min(
                    1.0,
                    arm_scores["up"] / self.arm_min_occupancy,
                )
            )
        confidence = sum(confidence_parts) / len(confidence_parts)
        segments = tuple(
            (intersection, endpoints[name]) for name in directions
        )
        return {
            "symbol": symbol,
            "center_x": intersection[0],
            "center_y": intersection[1],
            "confidence": float(confidence),
            "intersection": intersection,
            "endpoints": endpoints,
            "segments": segments,
            "dash_lines": (),
            "arm_scores": arm_scores,
            "foreground_threshold": foreground_threshold,
        }

    def _dash_candidates(self, black_mask, width, height):
        frame_area = float(width * height)
        candidates = []
        for index, contour in enumerate(_find_contours(black_mask)):
            area = cv2.contourArea(contour)
            if area < frame_area * ROAD_DASH_MIN_AREA_RATIO:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width <= 0 or box_height <= 0:
                continue
            if not (
                width * ROAD_DASH_MIN_WIDTH_RATIO <= box_width <=
                width * ROAD_DASH_MAX_WIDTH_RATIO
            ):
                continue
            if not (
                height * ROAD_DASH_MIN_HEIGHT_RATIO <= box_height <=
                height * ROAD_DASH_MAX_HEIGHT_RATIO
            ):
                continue
            aspect = box_width / float(box_height)
            if not (
                ROAD_DASH_MIN_ASPECT_RATIO <= aspect <=
                ROAD_DASH_MAX_ASPECT_RATIO
            ):
                continue
            fill = area / float(box_width * box_height)
            if fill < ROAD_DASH_MIN_FILL_RATIO:
                continue
            candidates.append({
                "index": index,
                "x": x,
                "y": y,
                "w": box_width,
                "h": box_height,
                "cx": x + box_width * 0.5,
                "cy": y + box_height * 0.5,
                "fill": fill,
            })
        return candidates

    @staticmethod
    def _dash_rows(candidates, height):
        tolerance = height * ROAD_DASH_ROW_TOLERANCE_RATIO
        rows = []
        signatures = set()
        for seed in candidates:
            first_group = [
                item
                for item in candidates
                if abs(item["cy"] - seed["cy"]) <= tolerance
            ]
            if len(first_group) < ROAD_DASH_MIN_COUNT_PER_ROW:
                continue
            mean_y = sum(item["cy"] for item in first_group) / len(
                first_group
            )
            group = [
                item
                for item in first_group
                if abs(item["cy"] - mean_y) <= tolerance
            ]
            if len(group) < ROAD_DASH_MIN_COUNT_PER_ROW:
                continue
            signature = tuple(sorted(item["index"] for item in group))
            if signature in signatures:
                continue
            signatures.add(signature)
            group.sort(key=lambda item: item["cx"])
            x1 = min(item["x"] for item in group)
            x2 = max(item["x"] + item["w"] - 1 for item in group)
            mean_y = sum(item["cy"] for item in group) / len(group)
            spread = max(abs(item["cy"] - mean_y) for item in group)
            mean_fill = sum(item["fill"] for item in group) / len(group)
            rows.append({
                "items": group,
                "indices": signature,
                "count": len(group),
                "x1": x1,
                "x2": x2,
                "y": mean_y,
                "spread": spread,
                "mean_fill": mean_fill,
            })
        return rows

    @staticmethod
    def _dash_row_line(row, width, height):
        first = row["items"][0]
        last = row["items"][-1]
        fit = (
            (first["cx"], first["cy"]),
            (last["cx"], last["cy"]),
        )
        if abs(last["cx"] - first["cx"]) < 0.000001:
            fit = ((row["x1"], row["y"]), (row["x2"], row["y"]))
        return (
            _integer_point(_point_at_x(fit, row["x1"]), width, height),
            _integer_point(_point_at_x(fit, row["x2"]), width, height),
        )

    def _end_candidate(self, black_mask, width, height):
        candidates = self._dash_candidates(black_mask, width, height)
        rows = self._dash_rows(candidates, height)
        best = None
        min_separation = height * ROAD_DASH_MIN_ROW_SEPARATION_RATIO
        max_separation = height * ROAD_DASH_MAX_ROW_SEPARATION_RATIO
        tolerance = height * ROAD_DASH_ROW_TOLERANCE_RATIO

        for first_index in range(len(rows)):
            for second_index in range(first_index + 1, len(rows)):
                first = rows[first_index]
                second = rows[second_index]
                if set(first["indices"]).intersection(second["indices"]):
                    continue
                top = first if first["y"] < second["y"] else second
                bottom = second if top is first else first
                separation = bottom["y"] - top["y"]
                if not min_separation <= separation <= max_separation:
                    continue
                overlap = max(
                    0.0,
                    min(top["x2"], bottom["x2"]) -
                    max(top["x1"], bottom["x1"]),
                )
                smaller_span = min(
                    top["x2"] - top["x1"],
                    bottom["x2"] - bottom["x1"],
                )
                if smaller_span <= 0:
                    continue
                overlap_ratio = overlap / float(smaller_span)
                if overlap_ratio < ROAD_DASH_MIN_HORIZONTAL_OVERLAP:
                    continue

                count_score = min(
                    1.0,
                    min(top["count"], bottom["count"]) /
                    float(ROAD_DASH_MIN_COUNT_PER_ROW + 2),
                )
                consistency = 1.0 - min(
                    1.0,
                    max(top["spread"], bottom["spread"]) /
                    max(1.0, tolerance),
                )
                fill_score = min(
                    1.0,
                    (top["mean_fill"] + bottom["mean_fill"]) * 0.5,
                )
                confidence = (
                    count_score * 0.30 +
                    overlap_ratio * 0.30 +
                    consistency * 0.25 +
                    fill_score * 0.15
                )
                if best is None or confidence > best["confidence"]:
                    dash_lines = (
                        self._dash_row_line(top, width, height),
                        self._dash_row_line(bottom, width, height),
                    )
                    center_x = int(round((
                        min(top["x1"], bottom["x1"]) +
                        max(top["x2"], bottom["x2"])
                    ) * 0.5))
                    center_y = int(round((top["y"] + bottom["y"]) * 0.5))
                    best = {
                        "confidence": float(min(1.0, confidence)),
                        "center_x": center_x,
                        "center_y": center_y,
                        "dash_lines": dash_lines,
                    }
        return best

    def _end_path(self, foreground, end_candidate, width, height):
        """在下排虚线到画面底部之间估计中央路径。"""
        bottom_line = end_candidate["dash_lines"][1]
        terminal_y = int(round((bottom_line[0][1] + bottom_line[1][1]) * 0.5))
        vertical = self._vertical_line(
            foreground,
            width,
            height,
            preferred_x=end_candidate["center_x"],
        )
        if vertical is None:
            return None
        center_x = _point_at_y(vertical, terminal_y)[0]
        half_width = width * ROAD_PATH_CORRIDOR_HALF_WIDTH_RATIO
        corridor = _pixel_rect(
            width,
            height,
            center_x - half_width,
            terminal_y,
            center_x + half_width,
            height,
        )
        roi = foreground[
            corridor[1]:corridor[3],
            corridor[0]:corridor[2],
        ]
        if cv2.countNonZero(roi) < (
            width * height * ROAD_PATH_MIN_PIXELS_RATIO
        ):
            return None
        bounds = _mask_bounds(foreground, corridor, 2.0)
        if bounds is None:
            return None
        terminal = _integer_point(
            _point_at_y(vertical, terminal_y),
            width,
            height,
        )
        near = _integer_point(
            _point_at_y(vertical, bounds[3]),
            width,
            height,
        )
        if near[1] - terminal[1] < (
            height * ROAD_END_MIN_PATH_LENGTH_RATIO
        ):
            return None
        return {"near": near, "terminal": terminal}

    def _end_result(
        self,
        end_candidate,
        foreground,
        width,
        height,
        foreground_threshold,
    ):
        endpoints = self._end_path(
            foreground,
            end_candidate,
            width,
            height,
        )
        segments = ()
        if endpoints is None:
            endpoints = {}
        else:
            segments = ((endpoints["near"], endpoints["terminal"]),)
        return {
            "symbol": "end",
            "center_x": end_candidate["center_x"],
            "center_y": end_candidate["center_y"],
            "confidence": end_candidate["confidence"],
            "intersection": None,
            "endpoints": endpoints,
            "segments": segments,
            "dash_lines": end_candidate["dash_lines"],
            "arm_scores": {
                "up": 0.0,
                "down": 0.0,
                "left": 0.0,
                "right": 0.0,
                "center": 0.0,
            },
            "foreground_threshold": foreground_threshold,
        }

    def _detect_working(self, frame):
        height = int(frame.shape[0])
        width = int(frame.shape[1])
        threshold_value, foreground = self._foreground_mask(frame)
        self.last_foreground_threshold = threshold_value

        # T/十字结构成立时不再生成黑色掩膜；END 场景的分散黑块不会
        # 形成同时横跨左右并向下延伸的单个连通轮廓。
        route_candidate = self._route_candidate(foreground, width, height)
        if route_candidate is not None:
            route_mask = self._candidate_mask(
                height,
                width,
                route_candidate["contour"],
            )
            symbol = (
                "cross"
                if route_candidate["bbox"][1] <= (
                    height * ROAD_CROSS_TOP_MAX_RATIO
                )
                else "t"
            )
            route_result = self._route_result(
                symbol,
                route_mask,
                route_candidate,
                width,
                height,
                threshold_value,
            )
            if route_result is not None:
                return route_result

        black_mask = cv2.inRange(
            frame,
            (0, 0, 0),
            (
                self.black_max_value,
                self.black_max_value,
                self.black_max_value,
            ),
        )
        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_CLOSE,
            self._black_kernel,
        )
        end_candidate = self._end_candidate(black_mask, width, height)
        if end_candidate is None:
            return None
        return self._end_result(
            end_candidate,
            foreground,
            width,
            height,
            threshold_value,
        )

    def detect(self, frame):
        """检测当前 RGB 帧，不修改输入画面。"""
        if frame is None or len(frame.shape) != 3:
            raise ValueError("frame 必须是 RGB 三通道图像")
        image_height = int(frame.shape[0])
        image_width = int(frame.shape[1])
        if image_width <= 1 or image_height <= 1:
            raise ValueError("frame 尺寸无效")

        if (
            image_width == self.detect_width and
            image_height == self.detect_height
        ):
            working = frame
        else:
            working = cv2.resize(
                frame,
                (self.detect_width, self.detect_height),
                interpolation=cv2.INTER_AREA,
            )
        result = self._detect_working(working)
        result = _scale_result(
            result,
            image_width / float(self.detect_width),
            image_height / float(self.detect_height),
        )
        self.last_result = result
        self._update_target_state(frame, result)
        return result

    def draw(self, frame, result=None):
        """绘制指定结果；省略 result 时绘制最近一次检测结果。"""
        if result is None:
            result = self.last_result
        if result is None:
            return None

        for start, end in result["segments"]:
            cv2.line(
                frame,
                start,
                end,
                self.draw_segment_color,
                self.draw_thickness,
            )
        for start, end in result["dash_lines"]:
            cv2.line(
                frame,
                start,
                end,
                self.draw_dash_color,
                self.draw_thickness,
            )
        for name, point in result["endpoints"].items():
            cv2.circle(
                frame,
                point,
                self.draw_point_radius,
                self.draw_endpoint_color,
                2,
            )
            cv2.putText(
                frame,
                name.upper(),
                (point[0] + 5, max(15, point[1] - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.draw_font_scale * 0.75,
                self.draw_endpoint_color,
                1,
            )
        if result["intersection"] is not None:
            cv2.circle(
                frame,
                result["intersection"],
                self.draw_point_radius,
                self.draw_intersection_color,
                -1,
            )

        if result["symbol"] == "end":
            label_x = max(0, result["center_x"] - 35)
            label_y = max(20, result["center_y"] - 10)
        else:
            label_x = 5
            label_y = 50
        cv2.putText(
            frame,
            "{} {:.2f}".format(
                result["symbol"].upper(),
                result["confidence"],
            ),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.draw_font_scale,
            self.draw_text_color,
            self.draw_thickness,
        )
        return result

    def process(self, frame, draw=True):
        """检测一帧并按需绘制，返回结果字典或 None。"""
        result = self.detect(frame)
        if draw and result is not None:
            self.draw(frame, result)
        return result


def run_road_demo(display_target=None):
    """使用 CameraIO 按统一显示目标运行寻路符号演示。"""
    import gc
    import sys
    import time

    from camera_io import CameraIO

    if display_target is None:
        from config import DISPLAY_TARGET
        display_target = DISPLAY_TARGET
    camera = None
    detector = RoadSymbolDetector()
    frame_count = 0

    try:
        print("================================")
        print("K230 寻路符号检测")
        print("状态：T / CROSS / END")
        print("检测分辨率：{}x{}".format(
            detector.detect_width,
            detector.detect_height,
        ))
        print("显示目标：{}".format(display_target))
        print("================================")
        camera = CameraIO(display_target=display_target)
        camera.initialize()
        clock = time.clock()

        while True:
            clock.tick()
            image = camera.snapshot()
            frame = image.to_numpy_ref()
            detector.process(frame)
            cv2.putText(
                frame,
                "FPS: {:.1f}".format(clock.fps()),
                (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                ROAD_DRAW_TEXT_COLOR,
                2,
            )
            camera.show_image(image)

            frame_count += 1
            if frame_count % ROAD_DEMO_GC_INTERVAL == 0:
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
    run_road_demo()
