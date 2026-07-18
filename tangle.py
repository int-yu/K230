"""黑框白心方框检测模块，以及可直接运行的 K230 追踪示例。

检测器只分析单帧图像，不保存历史位置，也不使用 ROI 或运动预测。
候选的核心特征是：四边形边界内侧较亮、外侧较暗。这样即使黑框外侧
与深色衣服连成一片，黑框和白色内容之间的内沿仍然可以被检测。
"""

import math
import time

import cv2

import sys

# CanMV 按绝对路径启动脚本时不会把脚本所在目录加入 sys.path，
# 会导致 import config 失败。这里补上，重复导入不会重复追加。
if "/sdcard/K230" not in sys.path:
    sys.path.append("/sdcard/K230")

from config import (
    IMAGE_CENTER_X,
    IMAGE_CENTER_Y,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RECTANGLE_APPROX_RATIOS,
    RECTANGLE_BINARY_THRESHOLD,
    RECTANGLE_CANNY_HIGH_RATIO,
    RECTANGLE_CANNY_LOW_RATIO,
    RECTANGLE_DRAW_CENTER_COLOR,
    RECTANGLE_DRAW_CENTER_RADIUS,
    RECTANGLE_DRAW_COLOR,
    RECTANGLE_DRAW_THICKNESS,
    RECTANGLE_DETECT_HEIGHT,
    RECTANGLE_DETECT_WIDTH,
    RECTANGLE_EDGE_SAMPLE_COUNT,
    RECTANGLE_EDGE_SAMPLE_MAX_OFFSET,
    RECTANGLE_EDGE_SAMPLE_MIN_OFFSET,
    RECTANGLE_EDGE_SAMPLE_OFFSET_RATIO,
    RECTANGLE_EDGE_TARGET_CONTRAST,
    RECTANGLE_MAX_COUNT,
    RECTANGLE_MIN_AREA,
    RECTANGLE_MIN_CONFIDENCE,
    RECTANGLE_MIN_HEIGHT,
    RECTANGLE_MIN_MEAN_EDGE_CONTRAST,
    RECTANGLE_MIN_SIDE_EDGE_CONTRAST,
    RECTANGLE_MIN_WIDTH,
    RECTANGLE_STRONG_CONFIDENCE,
    RECTANGLE_USE_CANNY_FALLBACK,
    RECTANGLE_USE_OTSU,
    RECTANGLE_LOST_FRAME_LIMIT,
    TANGLE_CENTER_CROSS_SIZE,
    TANGLE_GC_INTERVAL,
    TANGLE_PRINT_INTERVAL,
)


GEOMETRY_WEIGHT = 0.35
EDGE_CONTRAST_WEIGHT = 0.45
EDGE_CONSISTENCY_WEIGHT = 0.20


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
    result = cv2.findContours(
        binary,
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(result) == 2:
        return result[0]
    return result[1]


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


def _point_distance(point_a, point_b):
    delta_x = point_a[0] - point_b[0]
    delta_y = point_a[1] - point_b[1]
    return math.sqrt(delta_x * delta_x + delta_y * delta_y)


def _angle_score(points):
    """返回四边形接近矩形的程度；透视形变只会降低部分分数。"""
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
    """按检测器返回的四个角点绘制目标内沿。"""
    points = rectangle["points"]
    for index in range(4):
        cv2.line(
            frame,
            points[index],
            points[(index + 1) % 4],
            color,
            thickness,
        )


class RectangleDetector:
    """检测黑框与白色内容交界形成的凸四边形。"""

    def __init__(
        self,
        detect_width=RECTANGLE_DETECT_WIDTH,
        detect_height=RECTANGLE_DETECT_HEIGHT,
        binary_threshold=RECTANGLE_BINARY_THRESHOLD,
        use_otsu=RECTANGLE_USE_OTSU,
        min_area=RECTANGLE_MIN_AREA,
        min_width=RECTANGLE_MIN_WIDTH,
        min_height=RECTANGLE_MIN_HEIGHT,
        approx_ratios=RECTANGLE_APPROX_RATIOS,
        max_candidates=RECTANGLE_MAX_COUNT,
        min_confidence=RECTANGLE_MIN_CONFIDENCE,
        strong_confidence=RECTANGLE_STRONG_CONFIDENCE,
        min_mean_edge_contrast=RECTANGLE_MIN_MEAN_EDGE_CONTRAST,
        min_side_edge_contrast=RECTANGLE_MIN_SIDE_EDGE_CONTRAST,
        edge_target_contrast=RECTANGLE_EDGE_TARGET_CONTRAST,
        edge_sample_count=RECTANGLE_EDGE_SAMPLE_COUNT,
        edge_sample_offset_ratio=RECTANGLE_EDGE_SAMPLE_OFFSET_RATIO,
        edge_sample_min_offset=RECTANGLE_EDGE_SAMPLE_MIN_OFFSET,
        edge_sample_max_offset=RECTANGLE_EDGE_SAMPLE_MAX_OFFSET,
        use_canny_fallback=RECTANGLE_USE_CANNY_FALLBACK,
        canny_low_ratio=RECTANGLE_CANNY_LOW_RATIO,
        canny_high_ratio=RECTANGLE_CANNY_HIGH_RATIO,
        draw_color=RECTANGLE_DRAW_COLOR,
        draw_thickness=RECTANGLE_DRAW_THICKNESS,
        draw_center_color=RECTANGLE_DRAW_CENTER_COLOR,
        draw_center_radius=RECTANGLE_DRAW_CENTER_RADIUS,
    ):
        if detect_width <= 0 or detect_height <= 0:
            raise ValueError("检测分辨率必须大于 0")
        if binary_threshold < 0 or binary_threshold > 255:
            raise ValueError("灰度阈值必须在 0..255")
        if min_area <= 0 or min_width <= 0 or min_height <= 0:
            raise ValueError("最小面积和尺寸必须大于 0")
        if not approx_ratios:
            raise ValueError("至少需要一个四边形拟合比例")
        if max_candidates <= 0:
            raise ValueError("最大候选数量必须大于 0")
        if not 0 <= min_confidence <= strong_confidence <= 1:
            raise ValueError("置信度阈值必须满足 0 <= 最低 <= 强候选 <= 1")
        if min_mean_edge_contrast <= 0 or min_side_edge_contrast <= 0:
            raise ValueError("边缘对比度阈值必须大于 0")
        if edge_target_contrast <= min_mean_edge_contrast:
            raise ValueError("目标边缘对比度必须大于最低平均对比度")
        if edge_sample_count < 3:
            raise ValueError("每条边至少需要 3 个采样点")
        if not 0 < edge_sample_offset_ratio < 0.5:
            raise ValueError("边缘采样偏移比例必须在 0..0.5")
        if not 0 < edge_sample_min_offset <= edge_sample_max_offset:
            raise ValueError("边缘采样偏移范围无效")
        if canny_low_ratio <= 0 or canny_high_ratio <= canny_low_ratio:
            raise ValueError("Canny 阈值比例无效")
        if draw_thickness <= 0 or draw_center_radius <= 0:
            raise ValueError("绘制线宽和中心点半径必须大于 0")

        self.detect_width = detect_width
        self.detect_height = detect_height
        self.binary_threshold = binary_threshold
        self.use_otsu = use_otsu
        self.min_area = min_area
        self.min_width = min_width
        self.min_height = min_height
        self.approx_ratios = tuple(approx_ratios)
        self.max_candidates = max_candidates
        self.min_confidence = min_confidence
        self.strong_confidence = strong_confidence
        self.min_mean_edge_contrast = min_mean_edge_contrast
        self.min_side_edge_contrast = min_side_edge_contrast
        self.edge_target_contrast = edge_target_contrast
        self.edge_sample_count = edge_sample_count
        self.edge_sample_offset_ratio = edge_sample_offset_ratio
        self.edge_sample_min_offset = edge_sample_min_offset
        self.edge_sample_max_offset = edge_sample_max_offset
        self.use_canny_fallback = use_canny_fallback
        self.canny_low_ratio = canny_low_ratio
        self.canny_high_ratio = canny_high_ratio
        self.draw_color = draw_color
        self.draw_thickness = draw_thickness
        self.draw_center_color = draw_center_color
        self.draw_center_radius = draw_center_radius

        self.last_threshold = 0.0
        self.last_canny_low = 0
        self.last_canny_high = 0
        self.last_contour_count = 0
        self.last_candidate_count = 0
        self.last_detection_ms = 0
        self.last_source = "none"
        self._target_valid = False
        self._offset_x = 0
        self._offset_y = 0

    def _update_target_state(self, frame, target):
        """更新供串口直接读取的当前帧目标状态。"""
        if target is None:
            self._target_valid = False
            self._offset_x = 0
            self._offset_y = 0
            return

        self._target_valid = True
        self._offset_x = int(frame.shape[1]) // 2 - int(target["center_x"])
        self._offset_y = int(frame.shape[0]) // 2 - int(target["center_y"])

    def detect(self, frame):
        """返回最佳目标字典；没有合格目标时返回 None。"""
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

        threshold_mode = cv2.THRESH_BINARY
        threshold_value = self.binary_threshold
        if self.use_otsu:
            threshold_mode |= cv2.THRESH_OTSU
            threshold_value = 0

        self.last_threshold, bright_mask = cv2.threshold(
            gray,
            threshold_value,
            255,
            threshold_mode,
        )

        area_scale = scale_x * scale_y
        minimum_detect_area = self.min_area / area_scale
        best = self._best_from_binary(
            bright_mask,
            gray,
            "bright",
            minimum_detect_area,
            source_width,
            source_height,
            scale_x,
            scale_y,
        )

        if (
            self.use_canny_fallback and
            (best is None or best["confidence"] < self.strong_confidence)
        ):
            canny_threshold_base = max(1.0, self.last_threshold)
            self.last_canny_low = max(
                10,
                int(canny_threshold_base * self.canny_low_ratio),
            )
            self.last_canny_high = max(
                self.last_canny_low + 1,
                int(canny_threshold_base * self.canny_high_ratio),
            )
            edges = cv2.Canny(
                gray,
                self.last_canny_low,
                self.last_canny_high,
            )
            edge_best = self._best_from_binary(
                edges,
                gray,
                "canny",
                minimum_detect_area,
                source_width,
                source_height,
                scale_x,
                scale_y,
            )
            if (
                edge_best is not None and
                (
                    best is None or
                    edge_best["confidence"] > best["confidence"]
                )
            ):
                best = edge_best

        self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
        if best is None or best["confidence"] < self.min_confidence:
            self._update_target_state(frame, None)
            return None
        self.last_source = best["source"]
        self._update_target_state(frame, best)
        return best

    def draw(self, frame, result):
        """把一次 detect() 的结果绘制到原始帧上。"""
        if result is None:
            return None

        draw_frame_outline(
            frame,
            result,
            self.draw_color,
            self.draw_thickness,
        )
        cv2.circle(
            frame,
            (result["center_x"], result["center_y"]),
            self.draw_center_radius,
            self.draw_center_color,
            -1,
        )
        return result

    def process(self, frame, draw=True):
        """检测一帧；默认同时绘制边框和中心点。"""
        result = self.detect(frame)
        if draw and result is not None:
            self.draw(frame, result)
        return result

    def _best_from_binary(
        self,
        binary,
        gray,
        source_name,
        minimum_detect_area,
        source_width,
        source_height,
        scale_x,
        scale_y,
    ):
        contours = _find_contours(binary)
        self.last_contour_count += len(contours)
        candidates = self._collect_quadrilaterals(
            contours,
            minimum_detect_area,
        )

        best = None
        best_area = 0.0
        for detect_area, approx in candidates:
            self.last_candidate_count += 1
            candidate = self._evaluate_quadrilateral(
                approx,
                gray,
                source_name,
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
                    detect_area > best_area
                )
            ):
                best = candidate
                best_area = detect_area
        return best

    def _collect_quadrilaterals(self, contours, minimum_detect_area):
        candidates = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minimum_detect_area:
                continue

            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue

            approx = None
            for ratio in self.approx_ratios:
                current = cv2.approxPolyDP(
                    contour,
                    ratio * perimeter,
                    True,
                )
                if len(current) == 4 and cv2.isContourConvex(current):
                    approx = current
                    break
            if approx is not None:
                candidates.append((area, approx))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[:self.max_candidates]

    def _evaluate_quadrilateral(
        self,
        approx,
        gray,
        source_name,
        source_width,
        source_height,
        scale_x,
        scale_y,
    ):
        bounding_box = cv2.boundingRect(approx)
        if self._touches_detection_border(bounding_box):
            return None
        if (
            bounding_box[2] * scale_x < self.min_width or
            bounding_box[3] * scale_y < self.min_height
        ):
            return None

        detect_points = _contour_points(approx)
        edge_statistics = self._edge_contrast_statistics(
            gray,
            detect_points,
        )
        if edge_statistics is None:
            return None

        mean_contrast = edge_statistics[0]
        min_side_contrast = edge_statistics[1]
        side_contrasts = edge_statistics[2]
        if mean_contrast < self.min_mean_edge_contrast:
            return None
        if min_side_contrast < self.min_side_edge_contrast:
            return None

        geometry_score = _angle_score(detect_points)
        contrast_score = _clamp(
            (mean_contrast - self.min_mean_edge_contrast) /
            (self.edge_target_contrast - self.min_mean_edge_contrast)
        )
        edge_consistency_score = _clamp(
            min_side_contrast / mean_contrast
        )
        confidence = (
            geometry_score * GEOMETRY_WEIGHT +
            contrast_score * EDGE_CONTRAST_WEIGHT +
            edge_consistency_score * EDGE_CONSISTENCY_WEIGHT
        )

        detect_center = _quadrilateral_center(
            detect_points,
            bounding_box,
        )
        full_points = tuple(
            self._scale_point(
                point,
                scale_x,
                scale_y,
                source_width,
                source_height,
            )
            for point in detect_points
        )
        full_center = self._scale_point(
            detect_center,
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
            "geometry_score": geometry_score,
            "contrast_score": contrast_score,
            "edge_consistency_score": edge_consistency_score,
            "mean_edge_contrast": mean_contrast,
            "min_edge_contrast": min_side_contrast,
            "side_edge_contrasts": side_contrasts,
            "source": source_name,
        }

    def _edge_contrast_statistics(self, gray, points):
        center_x = sum(point[0] for point in points) / 4.0
        center_y = sum(point[1] for point in points) / 4.0
        side_contrasts = []

        for index in range(4):
            point_a = points[index]
            point_b = points[(index + 1) % 4]
            edge_length = _point_distance(point_a, point_b)
            if edge_length < 4:
                return None

            offset = edge_length * self.edge_sample_offset_ratio
            offset = min(self.edge_sample_max_offset, offset)
            offset = max(self.edge_sample_min_offset, offset)

            side_sum = 0.0
            valid_samples = 0
            for sample_index in range(self.edge_sample_count):
                fraction = (
                    (sample_index + 1.0) /
                    (self.edge_sample_count + 1.0)
                )
                edge_x = point_a[0] + (point_b[0] - point_a[0]) * fraction
                edge_y = point_a[1] + (point_b[1] - point_a[1]) * fraction

                inward_x = center_x - edge_x
                inward_y = center_y - edge_y
                inward_length = math.sqrt(
                    inward_x * inward_x + inward_y * inward_y
                )
                if inward_length <= 0:
                    continue
                inward_x /= inward_length
                inward_y /= inward_length

                inside_x = int(edge_x + inward_x * offset + 0.5)
                inside_y = int(edge_y + inward_y * offset + 0.5)
                outside_x = int(edge_x - inward_x * offset + 0.5)
                outside_y = int(edge_y - inward_y * offset + 0.5)
                if not self._sample_points_are_valid(
                    gray,
                    inside_x,
                    inside_y,
                    outside_x,
                    outside_y,
                ):
                    continue

                inside_value = int(gray[inside_y, inside_x])
                outside_value = int(gray[outside_y, outside_x])
                side_sum += inside_value - outside_value
                valid_samples += 1

            if valid_samples < self.edge_sample_count - 1:
                return None
            side_contrasts.append(side_sum / valid_samples)

        mean_contrast = sum(side_contrasts) / 4.0
        return (
            mean_contrast,
            min(side_contrasts),
            tuple(side_contrasts),
        )

    @staticmethod
    def _sample_points_are_valid(
        gray,
        inside_x,
        inside_y,
        outside_x,
        outside_y,
    ):
        width = int(gray.shape[1])
        height = int(gray.shape[0])
        return (
            0 <= inside_x < width and
            0 <= inside_y < height and
            0 <= outside_x < width and
            0 <= outside_y < height
        )

    def _touches_detection_border(self, bounding_box):
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


class TargetHoldState:
    """只在显示端短暂保留目标，不向串口发送历史坐标。"""

    def __init__(self, lost_frame_limit):
        self.lost_frame_limit = lost_frame_limit
        self.last_target = None
        self.lost_frame_count = 0

    def update(self, current_target):
        if current_target is not None:
            self.last_target = current_target
            self.lost_frame_count = 0
            return current_target, False

        self.lost_frame_count += 1
        if (
            self.last_target is not None and
            self.lost_frame_count <= self.lost_frame_limit
        ):
            return self.last_target, True

        self.last_target = None
        return None, False

    def clear(self):
        self.last_target = None
        self.lost_frame_count = 0


def draw_image_center(frame):
    """绘制画面中心十字。"""
    color = (255, 255, 255)
    cv2.line(
        frame,
        (IMAGE_CENTER_X - TANGLE_CENTER_CROSS_SIZE, IMAGE_CENTER_Y),
        (IMAGE_CENTER_X + TANGLE_CENTER_CROSS_SIZE, IMAGE_CENTER_Y),
        color,
        2,
    )
    cv2.line(
        frame,
        (IMAGE_CENTER_X, IMAGE_CENTER_Y - TANGLE_CENTER_CROSS_SIZE),
        (IMAGE_CENTER_X, IMAGE_CENTER_Y + TANGLE_CENTER_CROSS_SIZE),
        color,
        2,
    )
    cv2.circle(frame, (IMAGE_CENTER_X, IMAGE_CENTER_Y), 3, color, -1)


def target_relative_offset(target):
    """计算目标中心相对于画面中心的坐标。"""
    return (
        IMAGE_CENTER_X - target["center_x"],
        IMAGE_CENTER_Y - target["center_y"],
    )


def draw_tracking_target(frame, target, offset_x, offset_y, is_held):
    """绘制追踪示例中的目标连线和相对坐标。"""
    center = (target["center_x"], target["center_y"])
    box_color = (255, 255, 0) if is_held else RECTANGLE_DRAW_COLOR
    draw_frame_outline(frame, target, box_color, RECTANGLE_DRAW_THICKNESS)
    cv2.circle(
        frame,
        center,
        RECTANGLE_DRAW_CENTER_RADIUS,
        RECTANGLE_DRAW_CENTER_COLOR,
        -1,
    )
    cv2.line(
        frame,
        (IMAGE_CENTER_X, IMAGE_CENTER_Y),
        center,
        RECTANGLE_DRAW_CENTER_COLOR,
        1,
    )

    label_x = min(max(0, center[0] + 8), IMAGE_WIDTH - 145)
    label_y = max(20, center[1] - 8)
    cv2.putText(
        frame,
        "X:{} Y:{}".format(offset_x, offset_y),
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2,
    )


def print_tracking_status(
    frame_count,
    target,
    offset_x,
    offset_y,
    is_held,
    fps,
    detector,
    hold_state,
):
    """低频输出诊断信息，避免终端打印拖慢主循环。"""
    if target is None:
        print(
            "frame={} 未检测到目标 fps={:.1f} detect={}ms "
            "threshold={:.1f} candidates={} source={}".format(
                frame_count,
                fps,
                detector.last_detection_ms,
                detector.last_threshold,
                detector.last_candidate_count,
                detector.last_source,
            )
        )
        return

    state = "保持" if is_held else "实时"
    print(
        "frame={} x={} y={} confidence={:.3f} state={} "
        "lost={}/{} fps={:.1f} detect={}ms threshold={:.1f} "
        "candidates={} edge={:.1f}/{:.1f} source={}".format(
            frame_count,
            offset_x,
            offset_y,
            target["confidence"],
            state,
            hold_state.lost_frame_count,
            hold_state.lost_frame_limit,
            fps,
            detector.last_detection_ms,
            detector.last_threshold,
            detector.last_candidate_count,
            target["mean_edge_contrast"],
            target["min_edge_contrast"],
            target["source"],
        )
    )


def run_rectangle_tracking():
    """直接运行 tangle.py 时使用的完整摄像头、串口追踪示例。"""
    import gc
    import sys

    from camera_io import CameraIO, DISPLAY_TARGET_IDE
    from color import ColorSpotDetector
    from uart_io import TrackingUART

    camera = None
    tracking_uart = None
    hold_state = None

    try:
        print("初始化追踪串口")
        tracking_uart = TrackingUART().initialize()
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

        print("初始化黑框白心检测器")
        detector = RectangleDetector()
        color_detector = ColorSpotDetector()
        hold_state = TargetHoldState(RECTANGLE_LOST_FRAME_LIMIT)

        print("初始化 CSI2 摄像头")
        camera = CameraIO(display_target=DISPLAY_TARGET_IDE)
        camera.initialize()

        clock = time.clock()
        frame_count = 0
        print("初始化完成")
        print(
            "画面中心：({}, {})，检测分辨率：{}x{}".format(
                IMAGE_CENTER_X,
                IMAGE_CENTER_Y,
                detector.detect_width,
                detector.detect_height,
            )
        )

        while True:
            clock.tick()
            image = camera.snapshot()
            frame = image.to_numpy_ref()

            # 保留现有彩色光点组合；方框检测本身不依赖 color.py。
            color_detector.process(frame)
            current_target = detector.process(frame, draw=False)
            display_target, target_is_held = hold_state.update(current_target)

            draw_image_center(frame)
            display_offset_x = None
            display_offset_y = None
            if display_target is not None:
                display_offset_x, display_offset_y = target_relative_offset(
                    display_target,
                )
                draw_tracking_target(
                    frame,
                    display_target,
                    display_offset_x,
                    display_offset_y,
                    target_is_held,
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
            tracking_uart.send_target(
                color_detector._target_valid,
                color_detector._offset_x,
                color_detector._offset_y,
                frame_id=frame_count,
            )
            camera.show_image(image)
            frame_count += 1

            if frame_count % TANGLE_PRINT_INTERVAL == 0:
                print_tracking_status(
                    frame_count,
                    display_target,
                    display_offset_x,
                    display_offset_y,
                    target_is_held,
                    fps,
                    detector,
                    hold_state,
                )

            del frame
            del image
            if frame_count % TANGLE_GC_INTERVAL == 0:
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
        if hold_state is not None:
            hold_state.clear()
        gc.collect()
        print("程序结束")


if __name__ == "__main__":
    run_rectangle_tracking()
