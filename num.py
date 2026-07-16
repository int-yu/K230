"""K230 打印数字识别模块，以及可直接运行的摄像头示例。

模块接口与 color.py、tangle.py 保持一致：

    detector = DigitDetector()
    result = detector.process(frame)

没有候选数字时返回 None；检测到候选时返回包含整串文本和逐位结果的字典。
"""

import gc
import os
import time

import cv2

try:
    import ulab.numpy as np
except ImportError:
    import numpy as np

from config import (
    DIGIT_BLUR_KERNEL_SIZE,
    DIGIT_BROKEN_EIGHT_MIN_ASPECT_RATIO,
    DIGIT_DRAW_RECOGNIZED_COLOR,
    DIGIT_DRAW_SUMMARY,
    DIGIT_DRAW_TEXT_COLOR,
    DIGIT_DRAW_THICKNESS,
    DIGIT_DRAW_UNKNOWN_COLOR,
    DIGIT_FRAME_BORDER_MARGIN,
    DIGIT_LINE_CENTER_TOLERANCE_RATIO,
    DIGIT_LINE_MIN_HEIGHT_RATIO,
    DIGIT_LOCAL_BLUR_KERNEL_SIZE,
    DIGIT_LOCAL_MORPH_KERNEL_SIZE,
    DIGIT_MATCH_THRESHOLD,
    DIGIT_MAX_AREA,
    DIGIT_MAX_ASPECT_RATIO,
    DIGIT_MAX_COUNT,
    DIGIT_MAX_HEIGHT,
    DIGIT_MAX_WIDTH,
    DIGIT_MIN_AREA,
    DIGIT_MIN_ASPECT_RATIO,
    DIGIT_MIN_FILL_RATIO,
    DIGIT_MIN_HEIGHT,
    DIGIT_MIN_WIDTH,
    DIGIT_MORPH_KERNEL_SIZE,
    DIGIT_NORMALIZED_HEIGHT,
    DIGIT_NORMALIZED_MARGIN,
    DIGIT_NORMALIZED_WIDTH,
    DIGIT_ROI_MARGIN_DIVISOR,
    DIGIT_ROI_MIN_MARGIN,
    DIGIT_SHAPE_MAX_DISTANCE,
    DIGIT_STRONG_TEMPLATE_THRESHOLD,
    DIGIT_TEMPLATE_DIR,
    DIGIT_TEMPLATE_DIR_CANDIDATES,
)


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


def find_contours(binary_image):
    """兼容 OpenCV 两种 findContours 返回格式。"""
    result = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(result) == 2:
        return result[0]
    return result[1]


def find_contours_with_hierarchy(binary_image):
    """返回轮廓和层级，并兼容不同 OpenCV 的返回结构。"""
    result = cv2.findContours(
        binary_image,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    if len(result) == 2:
        contours, hierarchy = result
    else:
        _, contours, hierarchy = result

    if hierarchy is None:
        return contours, None

    shape = getattr(hierarchy, "shape", ())
    if len(shape) >= 3:
        hierarchy = hierarchy[0]
    return contours, hierarchy


def _hierarchy_link(hierarchy, index, column):
    """安全读取层级项；层级缺失或格式异常时返回 -1。"""
    if hierarchy is None or index < 0:
        return -1
    try:
        return int(hierarchy[index][column])
    except (IndexError, TypeError, ValueError):
        return -1


def _hierarchy_depth(hierarchy, index):
    """计算轮廓层级深度，并防止损坏层级造成越界或死循环。"""
    try:
        max_steps = len(hierarchy)
    except (TypeError, AttributeError):
        return 0

    depth = 0
    parent = _hierarchy_link(hierarchy, index, 3)
    steps = 0
    while parent >= 0 and steps < max_steps:
        depth += 1
        steps += 1
        parent = _hierarchy_link(hierarchy, parent, 3)
    return depth


def _direct_child_indices(hierarchy, index):
    """返回指定轮廓的直接子轮廓索引。"""
    try:
        max_steps = len(hierarchy)
    except (TypeError, AttributeError):
        return ()

    children = []
    child = _hierarchy_link(hierarchy, index, 2)
    steps = 0
    while child >= 0 and steps < max_steps:
        children.append(child)
        steps += 1
        child = _hierarchy_link(hierarchy, child, 0)
    return tuple(children)


def _make_clean_digit_binary(gray, blur_kernel_size, morph_kernel):
    """使用统一参数生成黑底白字二值图。"""
    blurred = cv2.GaussianBlur(
        gray,
        (blur_kernel_size, blur_kernel_size),
        0,
    )
    threshold_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    clean_binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        morph_kernel,
    )
    return threshold_value, clean_binary


def crop_foreground(binary_image):
    """裁出黑底白字二值图中的全部前景笔画。"""
    contours = find_contours(binary_image)
    if contours is None or len(contours) == 0:
        return None

    min_x = int(binary_image.shape[1])
    min_y = int(binary_image.shape[0])
    max_x = 0
    max_y = 0
    found = False

    for contour in contours:
        if cv2.contourArea(contour) < 2:
            continue
        x, y, width, height = cv2.boundingRect(contour)
        min_x = min(min_x, x)
        min_y = min(min_y, y)
        max_x = max(max_x, x + width)
        max_y = max(max_y, y + height)
        found = True

    if not found or max_x <= min_x or max_y <= min_y:
        return None
    return binary_image[min_y:max_y, min_x:max_x]


def normalize_digit(
    binary_image,
    normalized_width=DIGIT_NORMALIZED_WIDTH,
    normalized_height=DIGIT_NORMALIZED_HEIGHT,
    normalized_margin=DIGIT_NORMALIZED_MARGIN,
):
    """保持宽高比，把数字前景居中到固定大小的黑色画布。"""
    cropped = crop_foreground(binary_image)
    if cropped is None:
        return None

    source_height = int(cropped.shape[0])
    source_width = int(cropped.shape[1])
    usable_width = normalized_width - normalized_margin * 2
    usable_height = normalized_height - normalized_margin * 2
    if source_width <= 0 or source_height <= 0:
        return None
    if usable_width <= 0 or usable_height <= 0:
        raise ValueError("数字标准画布小于或等于边距")

    scale = min(
        usable_width / float(source_width),
        usable_height / float(source_height),
    )
    new_width = max(1, int(source_width * scale))
    new_height = max(1, int(source_height * scale))
    resized = cv2.resize(
        cropped,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA,
    )

    canvas = np.zeros(
        (normalized_height, normalized_width),
        dtype=np.uint8,
    )
    offset_x = (normalized_width - new_width) // 2
    offset_y = (normalized_height - new_height) // 2
    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width,
    ] = resized
    return canvas


def prepare_template(
    template_image,
    normalized_width=DIGIT_NORMALIZED_WIDTH,
    normalized_height=DIGIT_NORMALIZED_HEIGHT,
    normalized_margin=DIGIT_NORMALIZED_MARGIN,
    blur_kernel_size=DIGIT_BLUR_KERNEL_SIZE,
    morph_kernel=None,
):
    """使用实时检测的同一套流程生成标准模板。"""
    if template_image is None:
        return None
    if len(template_image.shape) == 3:
        gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = template_image
    if morph_kernel is None:
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (DIGIT_MORPH_KERNEL_SIZE, DIGIT_MORPH_KERNEL_SIZE),
        )
    _, clean_binary = _make_clean_digit_binary(
        gray,
        blur_kernel_size,
        morph_kernel,
    )
    return normalize_digit(
        clean_binary,
        normalized_width,
        normalized_height,
        normalized_margin,
    )


def resolve_template_dir(template_dir=None, candidate_dirs=None):
    """返回包含 0.png 的模板目录；找不到时列出全部尝试路径。"""
    if template_dir is not None:
        directories = (template_dir,)
    else:
        directories = tuple(
            candidate_dirs
            if candidate_dirs is not None
            else DIGIT_TEMPLATE_DIR_CANDIDATES
        )
        if DIGIT_TEMPLATE_DIR not in directories:
            directories = (DIGIT_TEMPLATE_DIR,) + directories

    tried_paths = []
    for directory in directories:
        probe_path = "{}/0.png".format(directory.rstrip("/"))
        tried_paths.append(probe_path)
        try:
            os.stat(probe_path)
            return directory.rstrip("/")
        except Exception:
            pass

    raise RuntimeError(
        "找不到数字模板，已尝试：{}".format(", ".join(tried_paths))
    )


def load_templates(
    template_dir=None,
    normalized_width=DIGIT_NORMALIZED_WIDTH,
    normalized_height=DIGIT_NORMALIZED_HEIGHT,
    normalized_margin=DIGIT_NORMALIZED_MARGIN,
    verbose=False,
    blur_kernel_size=DIGIT_BLUR_KERNEL_SIZE,
    morph_kernel=None,
):
    """一次性加载并预处理 0 到 9 的模板。"""
    resolved_dir = resolve_template_dir(template_dir)
    loaded_templates = []
    if morph_kernel is None:
        morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (DIGIT_MORPH_KERNEL_SIZE, DIGIT_MORPH_KERNEL_SIZE),
        )
    if verbose:
        print("数字模板目录：{}".format(resolved_dir))

    for digit in range(10):
        template_path = "{}/{}.png".format(resolved_dir, digit)
        try:
            os.stat(template_path)
        except Exception:
            raise RuntimeError("找不到数字模板：{}".format(template_path))

        template_image = cv2.imread(template_path, cv2.IMREAD_COLOR)
        if template_image is None:
            raise RuntimeError("无法读取数字模板：{}".format(template_path))

        normalized_template = prepare_template(
            template_image,
            normalized_width,
            normalized_height,
            normalized_margin,
            blur_kernel_size,
            morph_kernel,
        )
        if normalized_template is None:
            raise RuntimeError("数字模板处理失败：{}".format(template_path))
        loaded_templates.append(normalized_template)

        if verbose:
            print("模板{}加载完成".format(digit))
        del template_image

    gc.collect()
    return tuple(loaded_templates), resolved_dir


class DigitDetector:
    """基于轮廓分割和模板匹配的打印数字检测器。"""

    def __init__(
        self,
        template_dir=None,
        match_threshold=DIGIT_MATCH_THRESHOLD,
        normalized_width=DIGIT_NORMALIZED_WIDTH,
        normalized_height=DIGIT_NORMALIZED_HEIGHT,
        normalized_margin=DIGIT_NORMALIZED_MARGIN,
        min_area=DIGIT_MIN_AREA,
        max_area=DIGIT_MAX_AREA,
        min_width=DIGIT_MIN_WIDTH,
        max_width=DIGIT_MAX_WIDTH,
        min_height=DIGIT_MIN_HEIGHT,
        max_height=DIGIT_MAX_HEIGHT,
        min_aspect_ratio=DIGIT_MIN_ASPECT_RATIO,
        max_aspect_ratio=DIGIT_MAX_ASPECT_RATIO,
        max_count=DIGIT_MAX_COUNT,
        blur_kernel_size=DIGIT_BLUR_KERNEL_SIZE,
        morph_kernel_size=DIGIT_MORPH_KERNEL_SIZE,
        roi_margin_divisor=DIGIT_ROI_MARGIN_DIVISOR,
        roi_min_margin=DIGIT_ROI_MIN_MARGIN,
        draw_recognized_color=DIGIT_DRAW_RECOGNIZED_COLOR,
        draw_unknown_color=DIGIT_DRAW_UNKNOWN_COLOR,
        draw_text_color=DIGIT_DRAW_TEXT_COLOR,
        draw_thickness=DIGIT_DRAW_THICKNESS,
        draw_summary=DIGIT_DRAW_SUMMARY,
        verbose=False,
        local_blur_kernel_size=DIGIT_LOCAL_BLUR_KERNEL_SIZE,
        local_morph_kernel_size=DIGIT_LOCAL_MORPH_KERNEL_SIZE,
        min_fill_ratio=DIGIT_MIN_FILL_RATIO,
        frame_border_margin=DIGIT_FRAME_BORDER_MARGIN,
        line_center_tolerance_ratio=DIGIT_LINE_CENTER_TOLERANCE_RATIO,
        line_min_height_ratio=DIGIT_LINE_MIN_HEIGHT_RATIO,
        shape_max_distance=DIGIT_SHAPE_MAX_DISTANCE,
        broken_eight_min_aspect_ratio=(
            DIGIT_BROKEN_EIGHT_MIN_ASPECT_RATIO
        ),
        strong_template_threshold=DIGIT_STRONG_TEMPLATE_THRESHOLD,
    ):
        if normalized_width <= 0 or normalized_height <= 0:
            raise ValueError("数字标准尺寸必须大于 0")
        if normalized_margin < 0:
            raise ValueError("数字标准边距不能小于 0")
        if min_area <= 0 or max_area < min_area:
            raise ValueError("数字面积范围无效")
        if min_width <= 0 or max_width < min_width:
            raise ValueError("数字宽度范围无效")
        if min_height <= 0 or max_height < min_height:
            raise ValueError("数字高度范围无效")
        if min_aspect_ratio <= 0 or max_aspect_ratio < min_aspect_ratio:
            raise ValueError("数字宽高比范围无效")
        if max_count <= 0:
            raise ValueError("最大数字数量必须大于 0")
        if blur_kernel_size <= 0 or blur_kernel_size % 2 == 0:
            raise ValueError("高斯核尺寸必须是正奇数")
        if morph_kernel_size <= 0:
            raise ValueError("形态学核尺寸必须大于 0")
        if (
            local_blur_kernel_size <= 0 or
            local_blur_kernel_size % 2 == 0
        ):
            raise ValueError("局部高斯核尺寸必须是正奇数")
        if local_morph_kernel_size <= 0:
            raise ValueError("局部形态学核尺寸必须大于 0")
        if roi_margin_divisor <= 0 or roi_min_margin < 0:
            raise ValueError("数字 ROI 边距参数无效")
        if not 0.0 <= min_fill_ratio <= 1.0:
            raise ValueError("数字最小填充率必须在 0 到 1 之间")
        if frame_border_margin < 0:
            raise ValueError("画面边缘留白不能小于 0")
        if line_center_tolerance_ratio <= 0:
            raise ValueError("同行中心容差必须大于 0")
        if not 0.0 < line_min_height_ratio <= 1.0:
            raise ValueError("同行最小高度比例必须在 0 到 1 之间")
        if shape_max_distance <= 0:
            raise ValueError("形状最大距离必须大于 0")
        if broken_eight_min_aspect_ratio <= 0:
            raise ValueError("断笔 8 的最小宽高比必须大于 0")
        if not 0.0 <= strong_template_threshold <= 1.0:
            raise ValueError("强模板阈值必须在 0 到 1 之间")
        if draw_thickness <= 0:
            raise ValueError("绘制线宽必须大于 0")

        self.match_threshold = match_threshold
        self.normalized_width = normalized_width
        self.normalized_height = normalized_height
        self.normalized_margin = normalized_margin
        self.min_area = min_area
        self.max_area = max_area
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.max_count = max_count
        self.blur_kernel_size = blur_kernel_size
        self.local_blur_kernel_size = local_blur_kernel_size
        self.roi_margin_divisor = roi_margin_divisor
        self.roi_min_margin = roi_min_margin
        self.min_fill_ratio = min_fill_ratio
        self.frame_border_margin = frame_border_margin
        self.line_center_tolerance_ratio = line_center_tolerance_ratio
        self.line_min_height_ratio = line_min_height_ratio
        self.shape_max_distance = shape_max_distance
        self.broken_eight_min_aspect_ratio = (
            broken_eight_min_aspect_ratio
        )
        self.strong_template_threshold = strong_template_threshold
        self.draw_recognized_color = draw_recognized_color
        self.draw_unknown_color = draw_unknown_color
        self.draw_text_color = draw_text_color
        self.draw_thickness = draw_thickness
        self.draw_summary = draw_summary
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (morph_kernel_size, morph_kernel_size),
        )
        self.local_morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (local_morph_kernel_size, local_morph_kernel_size),
        )

        self.templates, self.template_dir = load_templates(
            template_dir,
            normalized_width,
            normalized_height,
            normalized_margin,
            verbose,
            blur_kernel_size,
            self.morph_kernel,
        )
        self.template_contours = self._build_template_contours()
        self.last_threshold = 0.0
        self.last_contour_count = 0
        self.last_detection_ms = 0
        self._target_valid = False
        self._offset_x = 0
        self._offset_y = 0

    def _update_target_state(self, frame, result):
        """更新供串口直接读取的当前帧目标状态。"""
        if result is None:
            self._target_valid = False
            self._offset_x = 0
            self._offset_y = 0
            return

        self._target_valid = True
        self._offset_x = int(frame.shape[1]) // 2 - int(result["center_x"])
        self._offset_y = int(frame.shape[0]) // 2 - int(result["center_y"])

    def _build_template_contours(self):
        """提取模板外轮廓，供抗旋转、抗透视的形状匹配使用。"""
        template_contours = []
        for template in self.templates:
            contours, hierarchy = find_contours_with_hierarchy(template)
            eligible = []
            for index in range(len(contours)):
                if _hierarchy_depth(hierarchy, index) % 2 != 0:
                    continue
                if cv2.contourArea(contours[index]) >= 2:
                    eligible.append(index)

            if eligible:
                best_index = max(
                    eligible,
                    key=lambda item: cv2.contourArea(contours[item]),
                )
                template_contours.append(contours[best_index])
            else:
                template_contours.append(None)
        return tuple(template_contours)

    def _valid_hole_indices(
        self,
        contours,
        hierarchy,
        contour_index,
        contour_area,
        width,
        height,
    ):
        """过滤极小层级噪声，只保留可能属于数字的内部孔洞。"""
        holes = []
        min_hole_area = max(15.0, contour_area * 0.02)
        min_hole_width = width * 0.10
        min_hole_height = height * 0.10
        for child_index in _direct_child_indices(
            hierarchy,
            contour_index,
        ):
            child_area = cv2.contourArea(contours[child_index])
            _, _, child_width, child_height = cv2.boundingRect(
                contours[child_index]
            )
            if (
                child_area >= min_hole_area and
                child_width >= min_hole_width and
                child_height >= min_hole_height
            ):
                holes.append(child_index)
        return tuple(holes)

    def _valid_digit_box(
        self,
        x,
        y,
        width,
        height,
        area,
        image_width,
        image_height,
    ):
        if area < self.min_area or area > self.max_area:
            return False
        if width < self.min_width or width > self.max_width:
            return False
        if height < self.min_height or height > self.max_height:
            return False
        if height <= 0:
            return False
        aspect_ratio = width / float(height)
        if not self.min_aspect_ratio <= aspect_ratio <= self.max_aspect_ratio:
            return False
        fill_ratio = area / float(width * height)
        if fill_ratio < self.min_fill_ratio:
            return False
        margin = self.frame_border_margin
        return (
            x >= margin and
            y >= margin and
            x + width <= image_width - margin and
            y + height <= image_height - margin
        )

    def _score_template(self, normalized_digit, digit):
        match_result = cv2.matchTemplate(
            normalized_digit,
            self.templates[digit],
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_value, _, _ = cv2.minMaxLoc(match_result)
        del match_result
        return float(max_value)

    def _recognize(self, normalized_digit):
        best_digit = -1
        best_score = -1.0
        for digit in range(10):
            score = self._score_template(normalized_digit, digit)
            if score > best_score:
                best_score = score
                best_digit = digit

        if best_score < self.match_threshold:
            return -1, best_score
        return best_digit, best_score

    def _shape_distance(self, contour, digit):
        template_contour = self.template_contours[digit]
        match_shapes = getattr(cv2, "matchShapes", None)
        if template_contour is None or match_shapes is None:
            return None
        try:
            return float(
                match_shapes(
                    contour,
                    template_contour,
                    cv2.CONTOURS_MATCH_I1,
                    0,
                )
            )
        except (AttributeError, TypeError, ValueError):
            return None

    def _best_shape_match(self, contour, allowed_digits):
        best_digit = -1
        best_distance = None
        for digit in allowed_digits:
            distance = self._shape_distance(contour, digit)
            if distance is None:
                continue
            if best_distance is None or distance < best_distance:
                best_digit = digit
                best_distance = distance
        return best_digit, best_distance

    def _classify_local_roi(self, gray, x, y, width, height):
        """在单个候选 ROI 中重新二值化，修复反光造成的细小断笔。"""
        image_height = int(gray.shape[0])
        image_width = int(gray.shape[1])
        margin_x = max(
            self.roi_min_margin,
            width // self.roi_margin_divisor,
        )
        margin_y = max(
            self.roi_min_margin,
            height // self.roi_margin_divisor,
        )
        roi_left = max(0, x - margin_x)
        roi_top = max(0, y - margin_y)
        roi_right = min(image_width, x + width + margin_x)
        roi_bottom = min(image_height, y + height + margin_y)
        local_gray = gray[roi_top:roi_bottom, roi_left:roi_right]
        if (
            int(local_gray.shape[0]) <= 0 or
            int(local_gray.shape[1]) <= 0
        ):
            return None

        _, local_binary = _make_clean_digit_binary(
            local_gray,
            self.local_blur_kernel_size,
            self.local_morph_kernel,
        )
        contours, hierarchy = find_contours_with_hierarchy(local_binary)
        min_local_area = max(40.0, width * height * 0.05)
        eligible = []
        for index in range(len(contours)):
            if _hierarchy_depth(hierarchy, index) % 2 != 0:
                continue
            if cv2.contourArea(contours[index]) >= min_local_area:
                eligible.append(index)
        if not eligible:
            return None

        contour_index = max(
            eligible,
            key=lambda item: cv2.contourArea(contours[item]),
        )
        contour = contours[contour_index]
        contour_area = cv2.contourArea(contour)
        local_x, local_y, local_width, local_height = cv2.boundingRect(
            contour
        )
        if local_width <= 0 or local_height <= 0:
            return None

        holes = self._valid_hole_indices(
            contours,
            hierarchy,
            contour_index,
            contour_area,
            local_width,
            local_height,
        )
        hole_count = len(holes)
        if hole_count >= 2:
            allowed_digits = (8,)
        elif hole_count == 1:
            allowed_digits = (0, 4, 6, 9)
        else:
            allowed_digits = (1, 2, 3, 5, 7)

        shape_digit, decision_distance = self._best_shape_match(
            contour,
            allowed_digits,
        )
        chosen_digit = shape_digit

        if hole_count >= 2:
            chosen_digit = 8
        elif hole_count == 1:
            largest_hole = max(
                holes,
                key=lambda item: cv2.contourArea(contours[item]),
            )
            _, hole_y, _, hole_height = cv2.boundingRect(
                contours[largest_hole]
            )
            relative_hole_y = (
                hole_y + hole_height * 0.5 - local_y
            ) / float(local_height)
            local_aspect_ratio = local_width / float(local_height)

            if relative_hole_y > 0.53:
                chosen_digit = 6
            elif (
                relative_hole_y < 0.47 and
                local_aspect_ratio >= (
                    self.broken_eight_min_aspect_ratio
                )
            ):
                # 裂纹或卡片边线可能破坏 8 的下孔；宽轮廓和上孔可恢复它。
                chosen_digit = 8

        normalized_digit = normalize_digit(
            local_binary,
            self.normalized_width,
            self.normalized_height,
            self.normalized_margin,
        )
        if normalized_digit is None:
            return None

        template_digit, best_template_score = self._recognize(
            normalized_digit
        )
        if (
            template_digit in allowed_digits and
            best_template_score >= self.strong_template_threshold
        ):
            chosen_digit = template_digit
            template_score = best_template_score
        elif chosen_digit < 0:
            chosen_digit = template_digit
            template_score = best_template_score
        else:
            template_score = self._score_template(
                normalized_digit,
                chosen_digit,
            )

        chosen_distance = (
            self._shape_distance(contour, chosen_digit)
            if chosen_digit >= 0
            else None
        )
        shape_score = -1.0
        if chosen_distance is not None:
            shape_score = 1.0 / (1.0 + max(0.0, chosen_distance))
        decision_score = -1.0
        if decision_distance is not None:
            decision_score = 1.0 / (
                1.0 + max(0.0, decision_distance)
            )
        confidence = max(template_score, shape_score, decision_score)

        if (
            decision_distance is not None and
            decision_distance > self.shape_max_distance and
            template_score < self.match_threshold
        ):
            return None

        recognized = (
            chosen_digit >= 0 and
            confidence >= self.match_threshold
        )
        return {
            "value": chosen_digit if recognized else -1,
            "confidence": float(confidence),
            "hole_count": hole_count,
            "shape_distance": (
                float(chosen_distance)
                if chosen_distance is not None
                else None
            ),
        }

    def _same_digit_line(self, first, second):
        first_height = float(first["h"])
        second_height = float(second["h"])
        height_ratio = min(first_height, second_height) / max(
            first_height,
            second_height,
        )
        if height_ratio < self.line_min_height_ratio:
            return False
        center_distance = abs(first["center_y"] - second["center_y"])
        return center_distance <= (
            self.line_center_tolerance_ratio *
            max(first_height, second_height)
        )

    def _select_best_digit_line(self, digits):
        """只保留面积最大且高度一致的一行数字，排除屏幕文字和污迹。"""
        if not digits:
            return []

        best_group = []
        best_score = -1.0
        for anchor in digits:
            group = [
                item
                for item in digits
                if self._same_digit_line(anchor, item)
            ]
            group_area = sum(item["area"] for item in group)
            group_score = group_area * (
                1.0 + 0.45 * (len(group) - 1)
            )
            if group_score > best_score:
                best_group = group
                best_score = group_score

        best_group.sort(key=lambda item: item["x"])
        return best_group[:self.max_count]

    def detect(self, frame):
        """识别一帧并返回整串结果；没有数字候选时返回 None。"""
        start_ms = _ticks_ms()
        image_height = int(frame.shape[0])
        image_width = int(frame.shape[1])

        if len(frame.shape) == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.last_threshold, clean_binary = _make_clean_digit_binary(
            gray,
            self.blur_kernel_size,
            self.morph_kernel,
        )
        contours, hierarchy = find_contours_with_hierarchy(clean_binary)
        self.last_contour_count = len(contours)

        digits = []
        for contour_index in range(len(contours)):
            if _hierarchy_depth(hierarchy, contour_index) % 2 != 0:
                continue
            contour = contours[contour_index]
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            if not self._valid_digit_box(
                x,
                y,
                width,
                height,
                area,
                image_width,
                image_height,
            ):
                continue

            local_result = self._classify_local_roi(
                gray,
                x,
                y,
                width,
                height,
            )
            if local_result is None:
                continue

            value = local_result["value"]
            digits.append({
                "value": value,
                "text": str(value) if value >= 0 else "?",
                "recognized": value >= 0,
                "confidence": local_result["confidence"],
                "x": x,
                "y": y,
                "w": width,
                "h": height,
                "area": float(area),
                "center_x": x + width // 2,
                "center_y": y + height // 2,
                "bbox": (x, y, width, height),
                "hole_count": local_result["hole_count"],
                "shape_distance": local_result["shape_distance"],
            })

        digits = self._select_best_digit_line(digits)

        self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
        if not digits:
            self._update_target_state(frame, None)
            return None

        x_min = min(item["x"] for item in digits)
        y_min = min(item["y"] for item in digits)
        x_max = max(item["x"] + item["w"] for item in digits)
        y_max = max(item["y"] + item["h"] for item in digits)
        confidence = sum(
            min(1.0, max(0.0, item["confidence"]))
            for item in digits
        ) / len(digits)

        result = {
            "text": "".join(item["text"] for item in digits),
            "digits": digits,
            "count": len(digits),
            "recognized_count": sum(
                1 for item in digits if item["recognized"]
            ),
            "confidence": confidence,
            "center_x": (x_min + x_max) // 2,
            "center_y": (y_min + y_max) // 2,
            "x": x_min,
            "y": y_min,
            "w": x_max - x_min,
            "h": y_max - y_min,
            "bbox": (x_min, y_min, x_max - x_min, y_max - y_min),
            "threshold": float(self.last_threshold),
        }
        self._update_target_state(frame, result)
        return result

    def draw(self, frame, result):
        """把一次 detect() 的逐位结果绘制到原始帧上。"""
        if result is None:
            return None

        for digit in result["digits"]:
            color = (
                self.draw_recognized_color
                if digit["recognized"]
                else self.draw_unknown_color
            )
            x = digit["x"]
            y = digit["y"]
            width = digit["w"]
            height = digit["h"]
            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                color,
                self.draw_thickness,
            )
            text_y = y - 8
            if text_y < 20:
                text_y = y + 22
            cv2.putText(
                frame,
                "{} {:.2f}".format(
                    digit["text"],
                    digit["confidence"],
                ),
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                self.draw_thickness,
            )

        if self.draw_summary:
            cv2.putText(
                frame,
                "Digits: {}".format(result["text"]),
                (5, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                self.draw_text_color,
                self.draw_thickness,
            )
        return result

    def process(self, frame, draw=True):
        """识别一帧；默认同时绘制数字框、结果和置信度。"""
        result = self.detect(frame)
        if draw and result is not None:
            self.draw(frame, result)
        return result


def run_digit_demo():
    """直接运行 num.py 时使用的完整摄像头识别示例。"""
    import sys

    from camera_io import CameraIO, DISPLAY_TARGET_IDE

    camera = None
    detector = None
    try:
        print("================================")
        print("K230 OpenCV打印数字识别")
        print("摄像头：CSI2")
        print("显示方式：CanMV IDE")
        print("================================")

        detector = DigitDetector(verbose=True)
        camera = CameraIO(display_target=DISPLAY_TARGET_IDE)
        camera.initialize()
        clock = time.clock()
        frame_count = 0
        print("初始化完成，请将打印数字放到摄像头前")

        while True:
            clock.tick()
            image = camera.snapshot()
            frame = image.to_numpy_ref()
            result = detector.process(frame)

            cv2.putText(
                frame,
                "FPS: {:.1f}".format(clock.fps()),
                (5, 58),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )
            count = result["count"] if result is not None else 0
            cv2.putText(
                frame,
                "Count: {}".format(count),
                (5, 86),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2,
            )
            camera.show_image(image)
            frame_count += 1

            if frame_count % 30 == 0:
                text = result["text"] if result is not None else ""
                print(
                    "FPS: {:.2f}, Digits: {}, Count: {}".format(
                        clock.fps(),
                        text,
                        count,
                    )
                )

            del frame
            del image
            if frame_count % 30 == 0:
                gc.collect()

    except KeyboardInterrupt:
        print("用户停止程序")
    except Exception as error:
        print("程序发生错误")
        sys.print_exception(error)
    finally:
        print("正在释放资源")
        if camera is not None:
            camera.deinitialize()
        if detector is not None:
            detector.templates = ()
        gc.collect()
        print("程序结束")


if __name__ == "__main__":
    run_digit_demo()
