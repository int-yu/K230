"""细铅笔线黑色方框检测模块。

自适应二值图只负责生成闭合四边形候选，不要求内外轮廓成对。
检测器随后回到原始灰度图，沿候选四条边的法线测量真实暗线宽度。
多个合格方框同时出现时，优先返回线宽中位数最小的方框。
"""

import math
import time

import cv2

import sys

# CanMV 按绝对路径启动脚本时不会把脚本所在目录加入 sys.path，
# 会导致 import config 失败。这里补上，重复导入不会重复追加。
for _path in ("/sdcard/K230", "/sdcard"):
    if _path not in sys.path:
        sys.path.append(_path)

from config import (
    PENCIL_RECTANGLE_ADAPTIVE_BLOCK_SIZE,
    PENCIL_RECTANGLE_ADAPTIVE_C,
    PENCIL_RECTANGLE_APPROX_RATIOS,
    PENCIL_RECTANGLE_BLUR_KERNEL_SIZE,
    PENCIL_RECTANGLE_CLOSE_KERNEL_HEIGHT,
    PENCIL_RECTANGLE_CLOSE_KERNEL_WIDTH,
    PENCIL_RECTANGLE_DETECT_HEIGHT,
    PENCIL_RECTANGLE_DETECT_WIDTH,
    PENCIL_RECTANGLE_DRAW_CENTER_COLOR,
    PENCIL_RECTANGLE_DRAW_COLOR,
    PENCIL_RECTANGLE_DRAW_POINT_RADIUS,
    PENCIL_RECTANGLE_DRAW_THICKNESS,
    PENCIL_RECTANGLE_FALLBACK_THRESHOLD,
    PENCIL_RECTANGLE_GC_INTERVAL,
    PENCIL_RECTANGLE_GEOMETRY_WEIGHT,
    PENCIL_RECTANGLE_MAX_BORDER_THICKNESS,
    PENCIL_RECTANGLE_MAX_COUNT,
    PENCIL_RECTANGLE_MIN_AREA,
    PENCIL_RECTANGLE_MIN_BORDER_THICKNESS,
    PENCIL_RECTANGLE_MIN_CONFIDENCE,
    PENCIL_RECTANGLE_MIN_EDGE_CONTRAST,
    PENCIL_RECTANGLE_MIN_GEOMETRY_SCORE,
    PENCIL_RECTANGLE_MIN_HEIGHT,
    PENCIL_RECTANGLE_MIN_PARALLEL_SCORE,
    PENCIL_RECTANGLE_MIN_STRAIGHTNESS_SCORE,
    PENCIL_RECTANGLE_MIN_THICKNESS_UNIFORMITY,
    PENCIL_RECTANGLE_MIN_WIDTH,
    PENCIL_RECTANGLE_PRINT_INTERVAL,
    PENCIL_RECTANGLE_PARALLEL_WEIGHT,
    PENCIL_RECTANGLE_PROFILE_DARK_RATIO,
    PENCIL_RECTANGLE_PROFILE_END_COUNT,
    PENCIL_RECTANGLE_PROFILE_SAMPLE_COUNT,
    PENCIL_RECTANGLE_PROFILE_SAMPLE_MARGIN,
    PENCIL_RECTANGLE_PROFILE_SCAN_RADIUS,
    PENCIL_RECTANGLE_PROFILE_SEARCH_RADIUS,
    PENCIL_RECTANGLE_STRAIGHTNESS_WEIGHT,
    PENCIL_RECTANGLE_THICKNESS_TIE_PX,
    PENCIL_RECTANGLE_USE_ADAPTIVE_THRESHOLD,
    PENCIL_RECTANGLE_USE_ELLIPSE_CLOSE_KERNEL,
)

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


def _find_contours(binary):
    try:
        result = cv2.findContours(
            binary,
            cv2.RETR_LIST,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    except Exception:
        # RETR_TREE 同样返回全部层级轮廓；这里只取轮廓，不读取层级关系。
        result = cv2.findContours(
            binary,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE,
        )
    if len(result) == 2:
        contours = result[0]
    else:
        contours = result[1]
    return contours


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


def _order_corners_clockwise(points):
    center_x = sum(point[0] for point in points) / 4.0
    center_y = sum(point[1] for point in points) / 4.0
    clockwise = sorted(
        points,
        key=lambda point: math.atan2(
            point[1] - center_y,
            point[0] - center_x,
        ),
    )
    top_left_index = min(
        range(4),
        key=lambda index: clockwise[index][0] + clockwise[index][1],
    )
    return tuple(
        clockwise[(top_left_index + index) % 4]
        for index in range(4)
    )


def _point_distance(point_a, point_b):
    delta_x = point_a[0] - point_b[0]
    delta_y = point_a[1] - point_b[1]
    return math.sqrt(delta_x * delta_x + delta_y * delta_y)


def _median(values):
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _angle_score(points):
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


def _parallel_score(points):
    directions = []
    for index in range(4):
        point_a = points[index]
        point_b = points[(index + 1) % 4]
        delta_x = point_b[0] - point_a[0]
        delta_y = point_b[1] - point_a[1]
        length = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        if length <= 0:
            return 0.0
        directions.append((delta_x / length, delta_y / length))

    first_pair = abs(
        directions[0][0] * directions[2][0] +
        directions[0][1] * directions[2][1]
    )
    second_pair = abs(
        directions[1][0] * directions[3][0] +
        directions[1][1] * directions[3][1]
    )
    return _clamp((first_pair + second_pair) / 2.0)


def _draw_polygon(frame, points, color, thickness):
    for index in range(4):
        cv2.line(
            frame,
            points[index],
            points[(index + 1) % 4],
            color,
            thickness,
        )


class PencilRectangleDetector:
    """检测细黑线方框，并在多个候选中选择边框最细的一个。"""

    def __init__(
        self,
        detect_width=PENCIL_RECTANGLE_DETECT_WIDTH,
        detect_height=PENCIL_RECTANGLE_DETECT_HEIGHT,
        use_adaptive_threshold=PENCIL_RECTANGLE_USE_ADAPTIVE_THRESHOLD,
        adaptive_block_size=PENCIL_RECTANGLE_ADAPTIVE_BLOCK_SIZE,
        adaptive_c=PENCIL_RECTANGLE_ADAPTIVE_C,
        fallback_threshold=PENCIL_RECTANGLE_FALLBACK_THRESHOLD,
        blur_kernel_size=PENCIL_RECTANGLE_BLUR_KERNEL_SIZE,
        close_kernel_width=PENCIL_RECTANGLE_CLOSE_KERNEL_WIDTH,
        close_kernel_height=PENCIL_RECTANGLE_CLOSE_KERNEL_HEIGHT,
        use_ellipse_close_kernel=(
            PENCIL_RECTANGLE_USE_ELLIPSE_CLOSE_KERNEL
        ),
        min_area=PENCIL_RECTANGLE_MIN_AREA,
        min_width=PENCIL_RECTANGLE_MIN_WIDTH,
        min_height=PENCIL_RECTANGLE_MIN_HEIGHT,
        max_candidates=PENCIL_RECTANGLE_MAX_COUNT,
        approx_ratios=PENCIL_RECTANGLE_APPROX_RATIOS,
        min_border_thickness=PENCIL_RECTANGLE_MIN_BORDER_THICKNESS,
        max_border_thickness=PENCIL_RECTANGLE_MAX_BORDER_THICKNESS,
        min_thickness_uniformity=PENCIL_RECTANGLE_MIN_THICKNESS_UNIFORMITY,
        thickness_tie_px=PENCIL_RECTANGLE_THICKNESS_TIE_PX,
        profile_sample_count=PENCIL_RECTANGLE_PROFILE_SAMPLE_COUNT,
        profile_sample_margin=PENCIL_RECTANGLE_PROFILE_SAMPLE_MARGIN,
        profile_scan_radius=PENCIL_RECTANGLE_PROFILE_SCAN_RADIUS,
        profile_search_radius=PENCIL_RECTANGLE_PROFILE_SEARCH_RADIUS,
        profile_end_count=PENCIL_RECTANGLE_PROFILE_END_COUNT,
        profile_dark_ratio=PENCIL_RECTANGLE_PROFILE_DARK_RATIO,
        min_edge_contrast=PENCIL_RECTANGLE_MIN_EDGE_CONTRAST,
        straightness_weight=PENCIL_RECTANGLE_STRAIGHTNESS_WEIGHT,
        geometry_weight=PENCIL_RECTANGLE_GEOMETRY_WEIGHT,
        parallel_weight=PENCIL_RECTANGLE_PARALLEL_WEIGHT,
        min_straightness_score=PENCIL_RECTANGLE_MIN_STRAIGHTNESS_SCORE,
        min_geometry_score=PENCIL_RECTANGLE_MIN_GEOMETRY_SCORE,
        min_parallel_score=PENCIL_RECTANGLE_MIN_PARALLEL_SCORE,
        min_confidence=PENCIL_RECTANGLE_MIN_CONFIDENCE,
        draw_color=PENCIL_RECTANGLE_DRAW_COLOR,
        draw_center_color=PENCIL_RECTANGLE_DRAW_CENTER_COLOR,
        draw_thickness=PENCIL_RECTANGLE_DRAW_THICKNESS,
        draw_point_radius=PENCIL_RECTANGLE_DRAW_POINT_RADIUS,
    ):
        if detect_width <= 0 or detect_height <= 0:
            raise ValueError("检测分辨率必须大于 0")
        if adaptive_block_size < 3 or adaptive_block_size % 2 == 0:
            raise ValueError("自适应阈值块尺寸必须是大于等于 3 的奇数")
        if blur_kernel_size <= 0 or blur_kernel_size % 2 == 0:
            raise ValueError("模糊核尺寸必须是正奇数")
        if close_kernel_width <= 0 or close_kernel_height <= 0:
            raise ValueError("闭运算核宽度和高度必须大于 0")
        if min_area <= 0 or min_width <= 0 or min_height <= 0:
            raise ValueError("最小面积和尺寸必须大于 0")
        if max_candidates <= 0 or not approx_ratios:
            raise ValueError("候选数量和拟合比例无效")
        if not 0 < min_border_thickness < max_border_thickness:
            raise ValueError("边框线宽范围无效")
        if not 0 < min_thickness_uniformity <= 1:
            raise ValueError("线宽一致性必须在 0..1")
        if thickness_tie_px < 0:
            raise ValueError("线宽并列容差不能小于 0")
        if profile_sample_count <= 0:
            raise ValueError("每边灰度采样数量必须大于 0")
        if not 0 <= profile_sample_margin < 0.5:
            raise ValueError("灰度采样边缘留白必须在 0..0.5")
        if profile_scan_radius <= 0:
            raise ValueError("灰度剖面扫描半径必须大于 0")
        if not 0 <= profile_search_radius <= profile_scan_radius:
            raise ValueError("暗线搜索半径必须在扫描半径以内")
        if not 0 < profile_end_count <= profile_scan_radius:
            raise ValueError("灰度剖面端点数量无效")
        if not 0 < profile_dark_ratio < 1:
            raise ValueError("暗线阈值比例必须在 0..1")
        if min_edge_contrast <= 0:
            raise ValueError("最低边缘灰度差必须大于 0")
        if not 0 <= min_straightness_score <= 1:
            raise ValueError("最低直线分数必须在 0..1")
        if not 0 <= min_geometry_score <= 1:
            raise ValueError("最低四角分数必须在 0..1")
        if not 0 <= min_parallel_score <= 1:
            raise ValueError("最低平行分数必须在 0..1")
        if not 0 <= min_confidence <= 1:
            raise ValueError("最低置信度必须在 0..1")
        confidence_weight_sum = (
            straightness_weight + geometry_weight + parallel_weight
        )
        if (
            straightness_weight < 0 or
            geometry_weight < 0 or
            parallel_weight < 0 or
            confidence_weight_sum <= 0
        ):
            raise ValueError("四边形置信度权重无效")
        if draw_thickness <= 0 or draw_point_radius <= 0:
            raise ValueError("绘制尺寸必须大于 0")

        self.detect_width = int(detect_width)
        self.detect_height = int(detect_height)
        self.use_adaptive_threshold = bool(use_adaptive_threshold)
        self.adaptive_block_size = int(adaptive_block_size)
        self.adaptive_c = adaptive_c
        self.fallback_threshold = fallback_threshold
        self.blur_kernel_size = int(blur_kernel_size)
        self.close_kernel_width = int(close_kernel_width)
        self.close_kernel_height = int(close_kernel_height)
        self.use_ellipse_close_kernel = bool(use_ellipse_close_kernel)
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.max_candidates = int(max_candidates)
        self.approx_ratios = tuple(approx_ratios)
        self.min_border_thickness = min_border_thickness
        self.max_border_thickness = max_border_thickness
        self.min_thickness_uniformity = min_thickness_uniformity
        self.thickness_tie_px = thickness_tie_px
        self.profile_sample_count = int(profile_sample_count)
        self.profile_sample_margin = profile_sample_margin
        self.profile_scan_radius = int(profile_scan_radius)
        self.profile_search_radius = int(profile_search_radius)
        self.profile_end_count = int(profile_end_count)
        self.profile_dark_ratio = profile_dark_ratio
        self.min_edge_contrast = min_edge_contrast
        self.min_straightness_score = min_straightness_score
        self.min_geometry_score = min_geometry_score
        self.min_parallel_score = min_parallel_score
        self.min_confidence = min_confidence
        self.straightness_weight = (
            straightness_weight / confidence_weight_sum
        )
        self.geometry_weight = geometry_weight / confidence_weight_sum
        self.parallel_weight = parallel_weight / confidence_weight_sum
        self.draw_color = draw_color
        self.draw_center_color = draw_center_color
        self.draw_thickness = int(draw_thickness)
        self.draw_point_radius = int(draw_point_radius)

        close_kernel_shape = cv2.MORPH_RECT
        if self.use_ellipse_close_kernel:
            try:
                close_kernel_shape = cv2.MORPH_ELLIPSE
            except Exception:
                close_kernel_shape = cv2.MORPH_RECT
        try:
            self.close_kernel = cv2.getStructuringElement(
                close_kernel_shape,
                (self.close_kernel_width, self.close_kernel_height),
            )
        except Exception:
            # 部分 CanMV OpenCV 固件可能不提供椭圆核，退回矩形核继续运行。
            self.close_kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT,
                (self.close_kernel_width, self.close_kernel_height),
            )
        self.last_source = "none"
        self.last_threshold = 0.0
        self.last_contour_count = 0
        self.last_candidate_count = 0
        self.last_detection_ms = 0
        self._target_valid = False
        self._offset_x = 0
        self._offset_y = 0

    def _approx_quadrilateral(self, contour):
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            return None
        for ratio in self.approx_ratios:
            approx = cv2.approxPolyDP(contour, ratio * perimeter, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                return approx
        return None

    def _make_binary(self, gray):
        if self.use_adaptive_threshold:
            try:
                binary = cv2.adaptiveThreshold(
                    gray,
                    255,
                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                    cv2.THRESH_BINARY_INV,
                    self.adaptive_block_size,
                    self.adaptive_c,
                )
                self.last_source = "adaptive"
                self.last_threshold = -1.0
                return binary
            except Exception:
                pass

        threshold_mode = cv2.THRESH_BINARY_INV
        threshold_value = self.fallback_threshold
        if threshold_value <= 0:
            threshold_mode |= cv2.THRESH_OTSU
            threshold_value = 0
        self.last_threshold, binary = cv2.threshold(
            gray,
            threshold_value,
            255,
            threshold_mode,
        )
        self.last_source = "otsu"
        return binary

    def _touches_border(self, bounding_box):
        x, y, width, height = bounding_box
        return (
            x <= 1 or
            y <= 1 or
            x + width >= self.detect_width - 2 or
            y + height >= self.detect_height - 2
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

    def _sample_profile(self, gray, center_x, center_y, normal_x, normal_y):
        height = int(gray.shape[0])
        width = int(gray.shape[1])
        values = []
        for offset in range(
            -self.profile_scan_radius,
            self.profile_scan_radius + 1,
        ):
            x = int(center_x + normal_x * offset + 0.5)
            y = int(center_y + normal_y * offset + 0.5)
            if x < 0 or y < 0 or x >= width or y >= height:
                return None
            values.append(int(gray[y, x]))
        return values

    def _measure_profile(self, profile):
        end_count = self.profile_end_count
        background = max(
            _median(profile[:end_count]),
            _median(profile[-end_count:]),
        )
        center_index = self.profile_scan_radius
        search_start = center_index - self.profile_search_radius
        search_end = center_index + self.profile_search_radius
        darkest_index = min(
            range(search_start, search_end + 1),
            key=lambda index: profile[index],
        )
        darkest_value = profile[darkest_index]
        contrast = background - darkest_value
        if contrast <= 0:
            return (0.0, contrast)

        dark_threshold = background - contrast * self.profile_dark_ratio
        first_dark = darkest_index
        last_dark = darkest_index
        while first_dark > 0 and profile[first_dark - 1] <= dark_threshold:
            first_dark -= 1
        while (
            last_dark + 1 < len(profile) and
            profile[last_dark + 1] <= dark_threshold
        ):
            last_dark += 1
        return (last_dark - first_dark + 1, contrast)

    def _measure_side(self, gray, point_a, point_b):
        delta_x = point_b[0] - point_a[0]
        delta_y = point_b[1] - point_a[1]
        side_length = math.sqrt(delta_x * delta_x + delta_y * delta_y)
        if side_length <= 0:
            return None

        normal_x = -delta_y / side_length
        normal_y = delta_x / side_length
        thicknesses = []
        contrasts = []
        if self.profile_sample_count == 1:
            fractions = (0.5,)
        else:
            usable_fraction = 1.0 - 2.0 * self.profile_sample_margin
            fractions = tuple(
                self.profile_sample_margin +
                usable_fraction * index / (self.profile_sample_count - 1)
                for index in range(self.profile_sample_count)
            )

        for fraction in fractions:
            center_x = point_a[0] + delta_x * fraction
            center_y = point_a[1] + delta_y * fraction
            profile = self._sample_profile(
                gray,
                center_x,
                center_y,
                normal_x,
                normal_y,
            )
            if profile is None:
                return None
            thickness, contrast = self._measure_profile(profile)
            thicknesses.append(thickness)
            contrasts.append(contrast)

        return (_median(thicknesses), _median(contrasts))

    def _evaluate_candidate(
        self,
        gray,
        contour,
        approx,
        source_width,
        source_height,
        scale_x,
        scale_y,
    ):
        area = abs(cv2.contourArea(contour))
        contour_perimeter = cv2.arcLength(contour, True)
        if area <= 0 or contour_perimeter <= 0:
            return None

        detect_points = _order_corners_clockwise(_contour_points(approx))
        quadrilateral_perimeter = sum(
            _point_distance(
                detect_points[index],
                detect_points[(index + 1) % 4],
            )
            for index in range(4)
        )
        straightness_score = _clamp(
            quadrilateral_perimeter / contour_perimeter
        )
        geometry_score = _angle_score(detect_points)
        parallel_score = _parallel_score(detect_points)
        if straightness_score < self.min_straightness_score:
            return None
        if geometry_score < self.min_geometry_score:
            return None
        if parallel_score < self.min_parallel_score:
            return None

        side_measurements = []
        for index in range(4):
            measurement = self._measure_side(
                gray,
                detect_points[index],
                detect_points[(index + 1) % 4],
            )
            if measurement is None:
                return None
            side_measurements.append(measurement)

        detect_thicknesses = tuple(
            measurement[0] for measurement in side_measurements
        )
        edge_contrasts = tuple(
            measurement[1] for measurement in side_measurements
        )
        min_edge_contrast = min(edge_contrasts)
        if min_edge_contrast < self.min_edge_contrast:
            return None

        min_detect_thickness = min(detect_thicknesses)
        max_detect_thickness = max(detect_thicknesses)
        if max_detect_thickness <= 0:
            return None
        thickness_uniformity = min_detect_thickness / max_detect_thickness
        if thickness_uniformity < self.min_thickness_uniformity:
            return None

        average_scale = (scale_x + scale_y) / 2.0
        border_thickness = _median(detect_thicknesses) * average_scale
        if not (
            self.min_border_thickness <= border_thickness <=
            self.max_border_thickness
        ):
            return None

        confidence = _clamp(
            straightness_score * self.straightness_weight +
            geometry_score * self.geometry_weight +
            parallel_score * self.parallel_weight
        )
        if confidence < self.min_confidence:
            return None

        points = tuple(
            self._scale_point(
                point,
                scale_x,
                scale_y,
                source_width,
                source_height,
            )
            for point in detect_points
        )
        center_x = int(sum(point[0] for point in points) / 4.0 + 0.5)
        center_y = int(sum(point[1] for point in points) / 4.0 + 0.5)
        x_values = [point[0] for point in points]
        y_values = [point[1] for point in points]
        x = min(x_values)
        y = min(y_values)
        width = max(x_values) - x
        height = max(y_values) - y

        return {
            "x": x,
            "y": y,
            "w": width,
            "h": height,
            "bbox": (x, y, width, height),
            "center_x": center_x,
            "center_y": center_y,
            "points": points,
            "border_thickness": border_thickness,
            "side_thicknesses": tuple(
                value * average_scale for value in detect_thicknesses
            ),
            "thickness_uniformity": thickness_uniformity,
            "edge_contrasts": edge_contrasts,
            "min_edge_contrast": min_edge_contrast,
            "mean_edge_contrast": _median(edge_contrasts),
            "straightness_score": straightness_score,
            "geometry_score": geometry_score,
            "parallel_score": parallel_score,
            "confidence": confidence,
            "area": area * scale_x * scale_y,
            "source": self.last_source,
        }

    def detect(self, frame):
        """返回边框最细的合格方框；没有目标时返回 None。"""
        start_ms = _ticks_ms()
        self.last_contour_count = 0
        self.last_candidate_count = 0
        self.last_source = "none"

        source_height = int(frame.shape[0])
        source_width = int(frame.shape[1])
        scale_x = source_width / float(self.detect_width)
        scale_y = source_height / float(self.detect_height)
        if (
            source_width == self.detect_width and
            source_height == self.detect_height
        ):
            detect_frame = frame
        else:
            detect_frame = cv2.resize(
                frame,
                (self.detect_width, self.detect_height),
                interpolation=cv2.INTER_AREA,
            )

        if len(detect_frame.shape) == 2:
            gray = detect_frame
        else:
            gray = cv2.cvtColor(detect_frame, cv2.COLOR_RGB2GRAY)
        profile_gray = gray
        if self.blur_kernel_size > 1:
            threshold_gray = cv2.GaussianBlur(
                gray,
                (self.blur_kernel_size, self.blur_kernel_size),
                0,
            )
        else:
            threshold_gray = gray

        binary = self._make_binary(threshold_gray)
        if self.close_kernel_width > 1 or self.close_kernel_height > 1:
            binary = cv2.morphologyEx(
                binary,
                cv2.MORPH_CLOSE,
                self.close_kernel,
            )

        contours = _find_contours(binary)
        self.last_contour_count = len(contours)
        best = None
        area_scale = scale_x * scale_y
        min_detect_area = self.min_area / area_scale
        contour_indices = []
        for index, contour in enumerate(contours):
            area = abs(cv2.contourArea(contour))
            if area < min_detect_area:
                continue

            bounding_box = cv2.boundingRect(contour)
            if self._touches_border(bounding_box):
                continue
            if (
                bounding_box[2] * scale_x < self.min_width or
                bounding_box[3] * scale_y < self.min_height
            ):
                continue
            contour_indices.append((area, index))

        contour_indices.sort(reverse=True)
        contour_indices = contour_indices[:self.max_candidates]
        for _, contour_index in contour_indices:
            contour = contours[contour_index]
            approx = self._approx_quadrilateral(contour)
            if approx is None:
                continue

            self.last_candidate_count += 1
            candidate = self._evaluate_candidate(
                profile_gray,
                contour,
                approx,
                source_width,
                source_height,
                scale_x,
                scale_y,
            )
            if candidate is None:
                continue

            if (
                best is None or
                candidate["border_thickness"] <
                best["border_thickness"] - self.thickness_tie_px or
                (
                    abs(
                        candidate["border_thickness"] -
                        best["border_thickness"]
                    ) <= self.thickness_tie_px and
                    candidate["confidence"] > best["confidence"]
                )
            ):
                best = candidate

        self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
        if best is None:
            self._target_valid = False
            self._offset_x = 0
            self._offset_y = 0
            return None

        self._target_valid = True
        self._offset_x = source_width // 2 - best["center_x"]
        self._offset_y = source_height // 2 - best["center_y"]
        return best

    def draw(self, frame, result):
        """绘制所选四边形、角点、中心和灰度剖面估算线宽。"""
        if result is None:
            return None

        _draw_polygon(
            frame,
            result["points"],
            self.draw_color,
            self.draw_thickness,
        )
        for point in result["points"]:
            cv2.circle(
                frame,
                point,
                self.draw_point_radius,
                self.draw_color,
                -1,
            )

        center = (result["center_x"], result["center_y"])
        cv2.circle(
            frame,
            center,
            self.draw_point_radius + 1,
            self.draw_center_color,
            -1,
        )
        label_x = min(max(0, result["x"]), int(frame.shape[1]) - 230)
        label_y = max(20, result["y"] - 8)
        cv2.putText(
            frame,
            "thin={:.1f}px conf={:.2f}".format(
                result["border_thickness"],
                result["confidence"],
            ),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            self.draw_color,
            self.draw_thickness,
        )
        return result

    def process(self, frame, draw=True):
        """检测一帧；默认绘制被选中的最细边框方框。"""
        result = self.detect(frame)
        if draw and result is not None:
            self.draw(frame, result)
        return result


def run_pencil_rectangle_demo(
    display_target=None,
    hold_ms=None,
    move_ms=None,
    uart_send_period_ms=None,
):
    """运行细铅笔框检测、四角轨迹显示和追踪串口应用。"""
    import gc
    import sys

    from core.camera_io import CameraIO
    from detectors.rectangle_corner_cycle import (
        CORNER_HOLD_MS,
        CORNER_MOVE_MS,
        CORNER_NAMES,
        UART_SEND_PERIOD_MS,
        CornerCycleController,
        _ticks_ms as corner_ticks_ms,
        draw_corner_cycle,
        draw_image_center,
    )
    from core.uart_io import TrackingUART

    if display_target is None:
        from config import DISPLAY_TARGET
        display_target = DISPLAY_TARGET
    if hold_ms is None:
        hold_ms = CORNER_HOLD_MS
    if move_ms is None:
        move_ms = CORNER_MOVE_MS
    if uart_send_period_ms is None:
        uart_send_period_ms = UART_SEND_PERIOD_MS

    camera = None
    tracking_uart = None
    detector = PencilRectangleDetector()
    controller = CornerCycleController(
        hold_ms=hold_ms,
        move_ms=move_ms,
    )

    try:
        print("初始化细铅笔线方框检测器")
        print("初始化四角轨迹追踪串口")
        tracking_uart = TrackingUART(
            send_period_ms=uart_send_period_ms,
        ).initialize()
        print(
            "UART{}：TX=GPIO{}，RX=GPIO{}，{} baud，周期={}ms".format(
                tracking_uart.uart_id,
                tracking_uart.tx_pin,
                tracking_uart.rx_pin,
                tracking_uart.baudrate,
                tracking_uart.send_period_ms,
            )
        )
        print("等待对端串口握手")
        tracking_uart.wait_for_handshake()
        print("串口握手完成")

        camera = CameraIO(display_target=display_target)
        camera.initialize()
        clock = time.clock()
        frame_count = 0
        print("检测规则：多个方框中选择边框最细的合格方框")
        print("四角顺序：TL -> TR -> BR -> BL -> TL")
        print("停留={}ms，移动={}ms".format(hold_ms, move_ms))

        while True:
            clock.tick()
            image = camera.snapshot()
            frame = image.to_numpy_ref()
            now_ms = corner_ticks_ms()
            result = detector.process(frame, draw=False)

            draw_image_center(frame)
            state = None
            if result is not None:
                detector.draw(frame, result)
                state = controller.update(result["points"], now_ms)
                draw_corner_cycle(frame, result, state)
                tracking_uart.send_target(
                    True,
                    state["offset_x"],
                    state["offset_y"],
                    frame_id=frame_count,
                    now_ms=now_ms,
                )
            else:
                controller.mark_target_lost(now_ms)
                tracking_uart.send_target(
                    False,
                    0,
                    0,
                    frame_id=frame_count,
                    now_ms=now_ms,
                )
                cv2.putText(
                    frame,
                    "TARGET LOST",
                    (5, 55),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.70,
                    (255, 0, 0),
                    2,
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

            if frame_count % PENCIL_RECTANGLE_PRINT_INTERVAL == 0:
                if state is None:
                    print(
                        "target=none fps={:.1f} contours={} candidates={}".format(
                            fps,
                            detector.last_contour_count,
                            detector.last_candidate_count,
                        )
                    )
                else:
                    print(
                        "thin={:.2f}px phase={} point={} x={} y={} "
                        "confidence={:.3f} geometry={:.3f} "
                        "parallel={:.3f} contrast={:.1f} fps={:.1f}".format(
                            result["border_thickness"],
                            state["phase"],
                            CORNER_NAMES[state["target_index"]],
                            state["offset_x"],
                            state["offset_y"],
                            result["confidence"],
                            result["geometry_score"],
                            result["parallel_score"],
                            result["min_edge_contrast"],
                            fps,
                        )
                    )
                print(
                    "detect={}ms total={:.1f}ms fps={:.1f}".format(
                        detector.last_detection_ms,
                        1000.0 / fps,
                        fps,
                    )
                )
            del frame
            del image
            if frame_count % PENCIL_RECTANGLE_GC_INTERVAL == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("用户停止程序")
    except Exception as error:
        print("程序发生错误")
        sys.print_exception(error)
    finally:
        print("释放资源")
        if tracking_uart is not None:
            tracking_uart.deinitialize()
        if camera is not None:
            camera.deinitialize()
        gc.collect()
        print("程序结束")


if __name__ == "__main__":
    run_pencil_rectangle_demo()
