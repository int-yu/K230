"""K230 红色寻路符号和黑色双排虚线结束符检测模块。

推荐调用方式：

    from road import RoadSymbolDetector

    detector = RoadSymbolDetector()
    result = detector.process(frame)
    if result is not None:
        print(result["symbol"], result["confidence"])

检测器不保留上一帧结果作为当前帧输出，也不执行预测或时间平滑。
``detect()`` 只检测，``draw()`` 只绘制，``process()`` 组合两者。
"""

import cv2

from config import (
    ROAD_ARM_MIN_OCCUPANCY,
    ROAD_BLACK_MAX_VALUE,
    ROAD_BLACK_MORPH_KERNEL_SIZE,
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
    ROAD_DRAW_DASH_COLOR,
    ROAD_DRAW_ENDPOINT_COLOR,
    ROAD_DRAW_FONT_SCALE,
    ROAD_DRAW_INTERSECTION_COLOR,
    ROAD_DRAW_POINT_RADIUS,
    ROAD_DRAW_SEGMENT_COLOR,
    ROAD_DRAW_TEXT_COLOR,
    ROAD_DRAW_THICKNESS,
    ROAD_PATH_CORRIDOR_HALF_WIDTH_RATIO,
    ROAD_PATH_MIN_PIXELS_RATIO,
    ROAD_RED_HSV_RANGES,
    ROAD_RED_MIN_CONTOUR_AREA_RATIO,
    ROAD_RED_MORPH_KERNEL_SIZE,
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
    """把 (x0, y0, x1, y1) 比例转换为裁剪后的像素矩形。"""
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
    """返回矩形内非零像素质心；区域为空时返回 None。"""
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
    """返回区域内合格红色连通块的整体边界。"""
    x1, y1, x2, y2 = rect
    roi = mask[y1:y2, x1:x2]
    contours = _find_contours(roi)
    left = None
    top = None
    right = None
    bottom = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_contour_area:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        current_left = x1 + x
        current_top = y1 + y
        current_right = current_left + width - 1
        current_bottom = current_top + height - 1
        left = current_left if left is None else min(left, current_left)
        top = current_top if top is None else min(top, current_top)
        right = (
            current_right if right is None else max(right, current_right)
        )
        bottom = (
            current_bottom if bottom is None else max(bottom, current_bottom)
        )
    if left is None:
        return None
    return (left, top, right, bottom)


def _line_intersection(first, second):
    """返回两条无限直线交点；近平行时返回 None。"""
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


def _make_range_mask(hsv, ranges):
    mask = None
    for lower, upper in ranges:
        current = cv2.inRange(hsv, lower, upper)
        if mask is None:
            mask = current
        else:
            mask = cv2.bitwise_or(mask, current)
    return mask


def _normalized_presence_score(value, threshold):
    """把占用率换算为结构评分；该值不是概率。"""
    if threshold <= 0:
        return 1.0
    return min(1.0, max(0.0, value / (threshold * 2.0)))


class RoadSymbolDetector:
    """红色 T/十字/普通路径与黑色双排虚线结束符检测器。"""

    def __init__(
        self,
        red_hsv_ranges=ROAD_RED_HSV_RANGES,
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
        if not red_hsv_ranges:
            raise ValueError("red_hsv_ranges 不能为空")
        if arm_min_occupancy <= 0 or arm_min_occupancy > 1:
            raise ValueError("arm_min_occupancy 必须在 0..1")
        if black_max_value < 0 or black_max_value > 255:
            raise ValueError("black_max_value 必须在 0..255")
        if draw_thickness <= 0 or draw_point_radius <= 0:
            raise ValueError("绘制线宽和点半径必须大于 0")

        self.red_hsv_ranges = tuple(red_hsv_ranges)
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

        self._red_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (ROAD_RED_MORPH_KERNEL_SIZE, ROAD_RED_MORPH_KERNEL_SIZE),
        )
        self._black_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (ROAD_BLACK_MORPH_KERNEL_SIZE, ROAD_BLACK_MORPH_KERNEL_SIZE),
        )
        self.last_result = None
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

    def _arm_scores(self, red_mask, width, height):
        scores = {}
        for name, ratio in _ARM_ROIS:
            scores[name] = _mask_occupancy(
                red_mask,
                _ratio_rect(width, height, ratio),
            )
        return scores

    def _horizontal_line(self, red_mask, width, height):
        left = _mask_centroid(
            red_mask,
            _ratio_rect(width, height, ROAD_ROI_LEFT),
        )
        right = _mask_centroid(
            red_mask,
            _ratio_rect(width, height, ROAD_ROI_RIGHT),
        )
        if left is None or right is None:
            return None
        return (left, right)

    def _vertical_line(self, red_mask, width, height, preferred_x=None):
        if preferred_x is None:
            x1 = width * 0.20
            x2 = width * 0.80
        else:
            half_width = width * ROAD_PATH_CORRIDOR_HALF_WIDTH_RATIO
            x1 = preferred_x - half_width
            x2 = preferred_x + half_width

        upper_rect = _pixel_rect(
            width,
            height,
            x1,
            height * 0.48,
            x2,
            height * 0.72,
        )
        lower_rect = _pixel_rect(
            width,
            height,
            x1,
            height * 0.72,
            x2,
            height,
        )
        upper = _mask_centroid(red_mask, upper_rect)
        lower = _mask_centroid(red_mask, lower_rect)
        if upper is not None and lower is not None:
            return (upper, lower)
        available = upper if upper is not None else lower
        if available is None:
            full_rect = _pixel_rect(width, height, x1, 0, x2, height)
            available = _mask_centroid(red_mask, full_rect)
        if available is None:
            return None
        return (
            (available[0], max(0.0, available[1] - 1.0)),
            (available[0], min(height - 1.0, available[1] + 1.0)),
        )

    def _cross_vertical_line(self, red_mask, width, height):
        """用十字的上、下方向 ROI 质心拟合纵向中心线。"""
        upper = _mask_centroid(
            red_mask,
            _ratio_rect(width, height, ROAD_ROI_UP),
        )
        lower = _mask_centroid(
            red_mask,
            _ratio_rect(width, height, ROAD_ROI_DOWN),
        )
        if upper is None or lower is None:
            return None
        return (upper, lower)

    def _direction_endpoint(
        self,
        red_mask,
        width,
        height,
        direction,
        line,
        min_contour_area,
    ):
        ratio = {
            "up": ROAD_ROI_UP,
            "down": ROAD_ROI_DOWN,
            "left": ROAD_ROI_LEFT,
            "right": ROAD_ROI_RIGHT,
        }[direction]
        bounds = _mask_bounds(
            red_mask,
            _ratio_rect(width, height, ratio),
            min_contour_area,
        )
        if bounds is None:
            return None
        if direction == "left":
            point = _point_at_x(line, bounds[0])
        elif direction == "right":
            point = _point_at_x(line, bounds[2])
        elif direction == "up":
            point = _point_at_y(line, bounds[1])
        else:
            point = _point_at_y(line, bounds[3])
        return _integer_point(point, width, height)

    def _red_symbol_result(
        self,
        symbol,
        red_mask,
        arm_scores,
        width,
        height,
    ):
        horizontal = self._horizontal_line(red_mask, width, height)
        vertical = (
            self._cross_vertical_line(red_mask, width, height)
            if symbol == "cross"
            else self._vertical_line(red_mask, width, height)
        )
        if horizontal is None or vertical is None:
            return None
        intersection = _line_intersection(horizontal, vertical)
        if intersection is None:
            intersection = (
                (horizontal[0][0] + horizontal[1][0]) * 0.5,
                (vertical[0][1] + vertical[1][1]) * 0.5,
            )
        intersection = _integer_point(intersection, width, height)

        directions = (
            ("left", "right", "down")
            if symbol == "t"
            else ("up", "down", "left", "right")
        )
        min_area = max(
            2.0,
            width * height * ROAD_RED_MIN_CONTOUR_AREA_RATIO,
        )
        endpoints = {}
        for direction in directions:
            line = (
                horizontal
                if direction in ("left", "right")
                else vertical
            )
            endpoint = self._direction_endpoint(
                red_mask,
                width,
                height,
                direction,
                line,
                min_area,
            )
            if endpoint is None:
                return None
            endpoints[direction] = endpoint

        required = [arm_scores[name] for name in directions]
        required.append(arm_scores["center"])
        confidence_parts = [
            _normalized_presence_score(value, self.arm_min_occupancy)
            for value in required
        ]
        if symbol == "t":
            absence = 1.0 - min(
                1.0,
                arm_scores["up"] / self.arm_min_occupancy,
            )
            confidence_parts.append(absence)
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
        }

    def _vertical_path_result(
        self,
        red_mask,
        width,
        height,
        arm_scores,
        preferred_x=None,
    ):
        vertical = self._vertical_line(
            red_mask,
            width,
            height,
            preferred_x=preferred_x,
        )
        if vertical is None:
            return None
        center_at_middle = _point_at_y(vertical, height * 0.5)[0]
        half_width = width * ROAD_PATH_CORRIDOR_HALF_WIDTH_RATIO
        corridor = _pixel_rect(
            width,
            height,
            center_at_middle - half_width,
            0,
            center_at_middle + half_width,
            height,
        )
        if cv2.countNonZero(
            red_mask[corridor[1]:corridor[3], corridor[0]:corridor[2]]
        ) < width * height * ROAD_PATH_MIN_PIXELS_RATIO:
            return None
        min_area = max(
            2.0,
            width * height * ROAD_RED_MIN_CONTOUR_AREA_RATIO,
        )
        bounds = _mask_bounds(red_mask, corridor, min_area)
        if bounds is None:
            return None
        far = _integer_point(_point_at_y(vertical, bounds[1]), width, height)
        near = _integer_point(
            _point_at_y(vertical, bounds[3]),
            width,
            height,
        )
        confidence = min(
            1.0,
            max(
                _normalized_presence_score(
                    arm_scores["down"],
                    self.arm_min_occupancy,
                ),
                _normalized_presence_score(
                    arm_scores["up"],
                    self.arm_min_occupancy,
                ),
            ),
        )
        return {
            "near": near,
            "far": far,
            "confidence": confidence,
        }

    def _dash_candidates(self, black_mask, width, height):
        frame_area = float(width * height)
        candidates = []
        contours = _find_contours(black_mask)
        for index in range(len(contours)):
            contour = contours[index]
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

    def _dash_rows(self, candidates, height):
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

    def _dash_row_line(self, row, width, height):
        items = row["items"]
        first = items[0]
        last = items[-1]
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

    def _end_result(
        self,
        end_candidate,
        red_mask,
        arm_scores,
        width,
        height,
    ):
        path = self._vertical_path_result(
            red_mask,
            width,
            height,
            arm_scores,
            preferred_x=end_candidate["center_x"],
        )
        # 双排黑块必须和中央红色路径同时存在，避免把背景中的两排
        # 矩形物体误判为结束符，也保证 end 结果始终带 near/terminal。
        if path is None:
            return None
        endpoints = {
            "near": path["near"],
            "terminal": path["far"],
        }
        segments = ((path["near"], path["far"]),)
        return {
            "symbol": "end",
            "center_x": end_candidate["center_x"],
            "center_y": end_candidate["center_y"],
            "confidence": end_candidate["confidence"],
            "intersection": None,
            "endpoints": endpoints,
            "segments": segments,
            "dash_lines": end_candidate["dash_lines"],
            "arm_scores": arm_scores,
        }

    def _line_result(self, red_mask, arm_scores, width, height):
        path = self._vertical_path_result(
            red_mask,
            width,
            height,
            arm_scores,
        )
        if path is None:
            return None
        near = path["near"]
        far = path["far"]
        center = (
            int(round((near[0] + far[0]) * 0.5)),
            int(round((near[1] + far[1]) * 0.5)),
        )
        return {
            "symbol": "line",
            "center_x": center[0],
            "center_y": center[1],
            "confidence": float(path["confidence"]),
            "intersection": None,
            "endpoints": {"near": near, "far": far},
            "segments": ((near, far),),
            "dash_lines": (),
            "arm_scores": arm_scores,
        }

    def detect(self, frame):
        """检测当前帧，不修改输入画面；无有效红色路径时返回 None。"""
        if frame is None or len(frame.shape) != 3:
            raise ValueError("frame 必须是 RGB 三通道图像")
        height = int(frame.shape[0])
        width = int(frame.shape[1])
        if width <= 1 or height <= 1:
            raise ValueError("frame 尺寸无效")

        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        red_mask = _make_range_mask(hsv, self.red_hsv_ranges)
        red_mask = cv2.morphologyEx(
            red_mask,
            cv2.MORPH_CLOSE,
            self._red_kernel,
        )
        black_mask = cv2.inRange(
            hsv,
            (0, 0, 0),
            (179, 255, self.black_max_value),
        )
        black_mask = cv2.morphologyEx(
            black_mask,
            cv2.MORPH_CLOSE,
            self._black_kernel,
        )
        arm_scores = self._arm_scores(red_mask, width, height)

        result = None
        end_candidate = self._end_candidate(black_mask, width, height)
        if end_candidate is not None:
            result = self._end_result(
                end_candidate,
                red_mask,
                arm_scores,
                width,
                height,
            )
        if result is None:
            threshold = self.arm_min_occupancy
            present = {
                name: arm_scores[name] >= threshold
                for name in arm_scores
            }
            if all(
                present[name]
                for name in ("up", "down", "left", "right", "center")
            ):
                result = self._red_symbol_result(
                    "cross",
                    red_mask,
                    arm_scores,
                    width,
                    height,
                )
            elif (
                present["left"] and
                present["right"] and
                present["down"] and
                present["center"] and
                not present["up"]
            ):
                result = self._red_symbol_result(
                    "t",
                    red_mask,
                    arm_scores,
                    width,
                    height,
                )
            elif (
                present["down"] and
                (present["up"] or present["center"])
            ):
                result = self._line_result(
                    red_mask,
                    arm_scores,
                    width,
                    height,
                )

        self.last_result = result
        self._update_target_state(frame, result)
        return result

    def draw(self, frame, result=None):
        """绘制指定结果；result 省略时绘制最近一次检测结果。"""
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
        for name in result["endpoints"]:
            point = result["endpoints"][name]
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

        label = result["symbol"].upper()
        if result["symbol"] == "end":
            label_x = max(0, result["center_x"] - 35)
            label_y = max(20, result["center_y"] - 10)
        else:
            label_x = 5
            label_y = 50
        cv2.putText(
            frame,
            "{} {:.2f}".format(label, result["confidence"]),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.draw_font_scale,
            self.draw_text_color,
            self.draw_thickness,
        )
        return result

    def process(self, frame, draw=True):
        """检测一帧并按需绘制，返回统一结果字典或 None。"""
        result = self.detect(frame)
        if draw and result is not None:
            self.draw(frame, result)
        return result


def run_road_demo(display_target=None):
    """使用 CameraIO 和 CanMV IDE 显示运行寻路符号演示。"""
    import gc
    import sys
    import time

    from camera_io import CameraIO, DISPLAY_TARGET_IDE

    if display_target is None:
        display_target = DISPLAY_TARGET_IDE
    camera = None
    detector = RoadSymbolDetector()
    frame_count = 0

    try:
        print("================================")
        print("K230 寻路符号检测")
        print("状态：T / CROSS / END / LINE")
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
    run_road_demo()
