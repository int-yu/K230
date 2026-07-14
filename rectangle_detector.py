"""黑色外框、白色内部目标检测。

检测器只负责单帧图像分析，不保存目标位置，也不使用 ROI 或运动预测。
主程序可独立决定是否保留历史目标、如何显示以及如何发送串口数据。
"""

import math
import time

import cv2

from config import (
    RECTANGLE_APPROX_RATIO,
    RECTANGLE_BINARY_THRESHOLD,
    RECTANGLE_DETECT_HEIGHT,
    RECTANGLE_DETECT_WIDTH,
    RECTANGLE_MAX_BORDER_ASYMMETRY,
    RECTANGLE_MAX_CENTER_OFFSET_RATIO,
    RECTANGLE_MAX_COUNT,
    RECTANGLE_MAX_INNER_AREA_RATIO,
    RECTANGLE_MIN_AREA,
    RECTANGLE_MIN_CONFIDENCE,
    RECTANGLE_MIN_HEIGHT,
    RECTANGLE_MIN_INNER_AREA_RATIO,
    RECTANGLE_MIN_WIDTH,
    RECTANGLE_USE_MORPH_CLOSE,
    RECTANGLE_USE_OTSU,
)


# 置信度是结构评分，不是统计概率。四项权重之和为 1。
GEOMETRY_WEIGHT = 0.30
CENTER_ALIGNMENT_WEIGHT = 0.25
BORDER_UNIFORMITY_WEIGHT = 0.25
INTERIOR_AREA_WEIGHT = 0.20


def _clamp(value, minimum=0.0, maximum=1.0):
    return min(maximum, max(minimum, value))


def _ticks_ms():
    try:
        return time.ticks_ms()
    except AttributeError:
        return int(time.time() * 1000)


def _ticks_diff(new_value, old_value):
    try:
        return time.ticks_diff(new_value, old_value)
    except AttributeError:
        return new_value - old_value


def _find_contours(binary, retrieval_mode):
    result = cv2.findContours(
        binary,
        retrieval_mode,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(result) == 2:
        return result[0], result[1]
    return result[1], result[2]


def _is_hierarchy_row(value):
    try:
        int(value[0])
        int(value[1])
        int(value[2])
        int(value[3])
        return True
    except Exception:
        return False


def _normalize_hierarchy(hierarchy, contour_count):
    """把 [1,N,4]、[N,4] 或扁平数据统一为可按轮廓索引的行。"""
    try:
        shape = hierarchy.shape
    except Exception:
        shape = None

    if shape is not None:
        dimension_count = len(shape)
        if dimension_count == 3:
            return hierarchy[0]
        if dimension_count == 2:
            return hierarchy
        if dimension_count == 1:
            return tuple(
                tuple(
                    int(hierarchy[index * 4 + column])
                    for column in range(4)
                )
                for index in range(contour_count)
            )

    try:
        first_item = hierarchy[0]
    except Exception:
        raise ValueError("findContours 返回了无法读取的 hierarchy")

    if _is_hierarchy_row(first_item):
        return hierarchy

    try:
        if _is_hierarchy_row(first_item[0]):
            return first_item
    except Exception:
        pass

    try:
        return tuple(
            tuple(
                int(hierarchy[index * 4 + column])
                for column in range(4)
            )
            for index in range(contour_count)
        )
    except Exception:
        raise ValueError("无法解析 findContours 返回的 hierarchy")


def _contour_points(approx):
    points = []
    for point in approx:
        try:
            x = int(point[0][0])
            y = int(point[0][1])
        except Exception:
            x = int(point[0])
            y = int(point[1])
        points.append((x, y))
    return tuple(points)


def _angle_score(points):
    """返回四边形接近直角的程度；透视场景中只作为辅助特征。"""
    if len(points) != 4:
        return 0.0

    cosine_sum = 0.0
    for index in range(4):
        previous_point = points[(index - 1) % 4]
        current_point = points[index]
        next_point = points[(index + 1) % 4]

        vector_1_x = previous_point[0] - current_point[0]
        vector_1_y = previous_point[1] - current_point[1]
        vector_2_x = next_point[0] - current_point[0]
        vector_2_y = next_point[1] - current_point[1]

        dot_product = (
            vector_1_x * vector_2_x +
            vector_1_y * vector_2_y
        )
        length_product = math.sqrt(
            (vector_1_x * vector_1_x + vector_1_y * vector_1_y) *
            (vector_2_x * vector_2_x + vector_2_y * vector_2_y)
        )
        if length_product <= 0:
            return 0.0
        cosine_sum += abs(dot_product / length_product)

    return _clamp(1.0 - cosine_sum / 4.0)


def _quadrilateral_center(points, bounding_box):
    """用两条对角线交点表示透视四边形的中心。"""
    x, y, width, height = bounding_box
    fallback = (x + width / 2.0, y + height / 2.0)
    if len(points) != 4:
        return fallback

    x1, y1 = points[0]
    x2, y2 = points[2]
    x3, y3 = points[1]
    x4, y4 = points[3]

    denominator = (
        (x1 - x2) * (y3 - y4) -
        (y1 - y2) * (x3 - x4)
    )
    if abs(denominator) < 0.000001:
        return fallback

    determinant_1 = x1 * y2 - y1 * x2
    determinant_2 = x3 * y4 - y3 * x4
    center_x = (
        determinant_1 * (x3 - x4) -
        (x1 - x2) * determinant_2
    ) / denominator
    center_y = (
        determinant_1 * (y3 - y4) -
        (y1 - y2) * determinant_2
    ) / denominator
    return (center_x, center_y)


def draw_frame_outline(frame, rectangle, color, thickness=2):
    """按检测器返回的四个角点绘制目标外框。"""
    points = rectangle["points"]
    for index in range(4):
        cv2.line(
            frame,
            points[index],
            points[(index + 1) % 4],
            color,
            thickness,
        )


class BlackWhiteFrameDetector:
    """在缩小图像中检测置信度最高的黑框白心四边形。"""

    def __init__(
        self,
        detect_width=RECTANGLE_DETECT_WIDTH,
        detect_height=RECTANGLE_DETECT_HEIGHT,
        binary_threshold=RECTANGLE_BINARY_THRESHOLD,
        use_otsu=RECTANGLE_USE_OTSU,
        use_morph_close=RECTANGLE_USE_MORPH_CLOSE,
        min_area=RECTANGLE_MIN_AREA,
        min_width=RECTANGLE_MIN_WIDTH,
        min_height=RECTANGLE_MIN_HEIGHT,
        approx_ratio=RECTANGLE_APPROX_RATIO,
        max_candidates=RECTANGLE_MAX_COUNT,
        min_confidence=RECTANGLE_MIN_CONFIDENCE,
        min_inner_area_ratio=RECTANGLE_MIN_INNER_AREA_RATIO,
        max_inner_area_ratio=RECTANGLE_MAX_INNER_AREA_RATIO,
        max_center_offset_ratio=RECTANGLE_MAX_CENTER_OFFSET_RATIO,
        max_border_asymmetry=RECTANGLE_MAX_BORDER_ASYMMETRY,
    ):
        if detect_width <= 0 or detect_height <= 0:
            raise ValueError("检测分辨率必须大于 0")
        if binary_threshold < 0 or binary_threshold > 255:
            raise ValueError("灰度阈值必须在 0..255")
        if min_area <= 0 or min_width <= 0 or min_height <= 0:
            raise ValueError("最小面积和尺寸必须大于 0")
        if approx_ratio <= 0 or approx_ratio >= 1:
            raise ValueError("多边形拟合比例必须在 0..1 之间")
        if max_candidates <= 0:
            raise ValueError("最大候选数量必须大于 0")
        if min_confidence < 0 or min_confidence > 1:
            raise ValueError("最低置信度必须在 0..1")
        if min_inner_area_ratio <= 0:
            raise ValueError("内部面积比例下限必须大于 0")
        if max_inner_area_ratio <= min_inner_area_ratio:
            raise ValueError("内部面积比例上限必须大于下限")
        if max_center_offset_ratio <= 0:
            raise ValueError("中心偏移比例必须大于 0")
        if max_border_asymmetry <= 0:
            raise ValueError("边框不对称上限必须大于 0")

        self.detect_width = detect_width
        self.detect_height = detect_height
        self.binary_threshold = binary_threshold
        self.use_otsu = use_otsu
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.approx_ratio = approx_ratio
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.min_inner_area_ratio = min_inner_area_ratio
        self.max_inner_area_ratio = max_inner_area_ratio
        self.max_center_offset_ratio = max_center_offset_ratio
        self.max_border_asymmetry = max_border_asymmetry

        self.close_kernel = None
        if use_morph_close:
            self.close_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (3, 3),
            )

        self.last_threshold = 0.0
        self.last_contour_count = 0
        self.last_candidate_count = 0
        self.last_detection_ms = 0

    def detect(self, frame):
        """返回最佳目标字典；没有合格目标时返回 None。"""
        start_ms = _ticks_ms()
        self.last_candidate_count = 0

        source_height = int(frame.shape[0])
        source_width = int(frame.shape[1])
        scale_x = source_width / float(self.detect_width)
        scale_y = source_height / float(self.detect_height)

        if (
            source_width == self.detect_width and
            source_height == self.detect_height
        ):
            small_frame = frame
        else:
            small_frame = cv2.resize(
                frame,
                (self.detect_width, self.detect_height),
                interpolation=cv2.INTER_AREA,
            )

        if len(small_frame.shape) == 2:
            gray = small_frame
        else:
            gray = cv2.cvtColor(small_frame, cv2.COLOR_RGB2GRAY)

        threshold_mode = cv2.THRESH_BINARY_INV
        threshold_value = self.binary_threshold
        if self.use_otsu:
            threshold_mode |= cv2.THRESH_OTSU
            threshold_value = 0

        self.last_threshold, binary = cv2.threshold(
            gray,
            threshold_value,
            255,
            threshold_mode,
        )

        if self.close_kernel is not None:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                self.close_kernel,
            )

        contours, hierarchy = _find_contours(binary, cv2.RETR_TREE)
        self.last_contour_count = len(contours)

        if hierarchy is None or not contours:
            self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
            return None

        hierarchy = _normalize_hierarchy(hierarchy, len(contours))

        area_scale = scale_x * scale_y
        minimum_detect_area = self.min_area / area_scale
        candidate_indices = self._collect_candidate_indices(
            contours,
            hierarchy,
            minimum_detect_area,
        )

        best = None
        best_detect_area = 0.0
        for detect_area, outer_index in candidate_indices:
            self.last_candidate_count += 1
            candidate = self._evaluate_outer_contour(
                contours,
                hierarchy,
                outer_index,
                source_width,
                source_height,
                scale_x,
                scale_y,
            )
            if candidate is None:
                continue

            if (
                best is None or
                candidate["confidence"] > best["confidence"] or
                (
                    candidate["confidence"] == best["confidence"] and
                    detect_area > best_detect_area
                )
            ):
                best = candidate
                best_detect_area = detect_area

        self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
        if best is None or best["confidence"] < self.min_confidence:
            return None
        return best

    def _collect_candidate_indices(
        self,
        contours,
        hierarchy,
        minimum_detect_area,
    ):
        candidates = []
        for index, contour in enumerate(contours):
            row = hierarchy[index]
            first_child = int(row[2])
            if first_child < 0:
                continue

            area = cv2.contourArea(contour)
            if area < minimum_detect_area:
                continue
            candidates.append((area, index))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:self.max_candidates]

    def _largest_child_index(self, contours, hierarchy, outer_index):
        child_index = int(hierarchy[outer_index][2])
        largest_index = -1
        largest_area = 0.0

        visited = 0
        while child_index >= 0 and visited < len(contours):
            child_area = cv2.contourArea(contours[child_index])
            if child_area > largest_area:
                largest_area = child_area
                largest_index = child_index

            child_index = int(hierarchy[child_index][0])
            visited += 1

        return largest_index

    def _evaluate_outer_contour(
        self,
        contours,
        hierarchy,
        outer_index,
        source_width,
        source_height,
        scale_x,
        scale_y,
    ):
        outer_contour = contours[outer_index]
        inner_index = self._largest_child_index(
            contours,
            hierarchy,
            outer_index,
        )
        if inner_index < 0:
            return None
        inner_contour = contours[inner_index]

        outer_area = cv2.contourArea(outer_contour)
        inner_area = cv2.contourArea(inner_contour)
        if outer_area <= 0 or inner_area <= 0:
            return None

        inner_area_ratio = inner_area / outer_area
        if not (
            self.min_inner_area_ratio <= inner_area_ratio <=
            self.max_inner_area_ratio
        ):
            return None

        outer_approx = self._approximate_quadrilateral(outer_contour)
        inner_approx = self._approximate_quadrilateral(inner_contour)
        if outer_approx is None or inner_approx is None:
            return None

        outer_box = cv2.boundingRect(outer_approx)
        inner_box = cv2.boundingRect(inner_approx)
        if self._touches_detection_border(outer_box):
            return None
        if (
            outer_box[2] * scale_x < self.min_width or
            outer_box[3] * scale_y < self.min_height
        ):
            return None

        border_uniformity = self._border_uniformity(outer_box, inner_box)
        if border_uniformity is None:
            return None

        outer_points = _contour_points(outer_approx)
        inner_points = _contour_points(inner_approx)
        outer_center = _quadrilateral_center(outer_points, outer_box)
        inner_center = _quadrilateral_center(inner_points, inner_box)
        center_score = self._center_alignment_score(
            outer_center,
            inner_center,
            outer_box,
        )
        if center_score is None:
            return None

        geometry_score = (
            _angle_score(outer_points) +
            _angle_score(inner_points)
        ) / 2.0

        interior_score = _clamp(
            inner_area_ratio / self.max_inner_area_ratio
        )

        confidence = (
            geometry_score * GEOMETRY_WEIGHT +
            center_score * CENTER_ALIGNMENT_WEIGHT +
            border_uniformity * BORDER_UNIFORMITY_WEIGHT +
            interior_score * INTERIOR_AREA_WEIGHT
        )

        full_points = tuple(
            self._scale_point(
                point,
                scale_x,
                scale_y,
                source_width,
                source_height,
            )
            for point in outer_points
        )
        full_center = self._scale_point(
            outer_center,
            scale_x,
            scale_y,
            source_width,
            source_height,
        )

        x_values = [point[0] for point in full_points]
        y_values = [point[1] for point in full_points]
        x = min(x_values)
        y = min(y_values)
        width = max(x_values) - x
        height = max(y_values) - y

        return {
            "x": x,
            "y": y,
            "w": width,
            "h": height,
            "center_x": full_center[0],
            "center_y": full_center[1],
            "points": full_points,
            "confidence": confidence,
            "inner_area_ratio": inner_area_ratio,
            "geometry_score": geometry_score,
            "center_score": center_score,
            "border_score": border_uniformity,
        }

    def _approximate_quadrilateral(self, contour):
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None

        approx = cv2.approxPolyDP(
            contour,
            self.approx_ratio * perimeter,
            True,
        )
        if len(approx) != 4 or not cv2.isContourConvex(approx):
            return None
        return approx

    def _border_uniformity(self, outer_box, inner_box):
        outer_x, outer_y, outer_width, outer_height = outer_box
        inner_x, inner_y, inner_width, inner_height = inner_box

        gaps = (
            inner_x - outer_x,
            inner_y - outer_y,
            (outer_x + outer_width) - (inner_x + inner_width),
            (outer_y + outer_height) - (inner_y + inner_height),
        )
        if min(gaps) < 0:
            return None

        average_gap = sum(gaps) / 4.0
        if average_gap <= 0:
            return None

        asymmetry = (max(gaps) - min(gaps)) / average_gap
        if asymmetry > self.max_border_asymmetry:
            return None
        return _clamp(1.0 - asymmetry / self.max_border_asymmetry)

    def _touches_detection_border(self, bounding_box):
        """排除由暗色背景产生的整幅画面外轮廓。"""
        x, y, width, height = bounding_box
        return (
            x <= 0 or
            y <= 0 or
            x + width >= self.detect_width - 1 or
            y + height >= self.detect_height - 1
        )

    def _center_alignment_score(self, outer_center, inner_center, outer_box):
        offset_x = outer_center[0] - inner_center[0]
        offset_y = outer_center[1] - inner_center[1]
        offset = math.sqrt(offset_x * offset_x + offset_y * offset_y)

        diagonal = math.sqrt(
            outer_box[2] * outer_box[2] +
            outer_box[3] * outer_box[3]
        )
        if diagonal <= 0:
            return None

        offset_ratio = offset / diagonal
        if offset_ratio > self.max_center_offset_ratio:
            return None
        return _clamp(
            1.0 - offset_ratio / self.max_center_offset_ratio
        )

    @staticmethod
    def _scale_point(
        point,
        scale_x,
        scale_y,
        source_width,
        source_height,
    ):
        x = int(point[0] * scale_x + 0.5)
        y = int(point[1] * scale_y + 0.5)
        x = min(source_width - 1, max(0, x))
        y = min(source_height - 1, max(0, y))
        return (x, y)
