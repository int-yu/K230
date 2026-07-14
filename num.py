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
    DIGIT_DRAW_RECOGNIZED_COLOR,
    DIGIT_DRAW_SUMMARY,
    DIGIT_DRAW_TEXT_COLOR,
    DIGIT_DRAW_THICKNESS,
    DIGIT_DRAW_UNKNOWN_COLOR,
    DIGIT_MATCH_THRESHOLD,
    DIGIT_MAX_AREA,
    DIGIT_MAX_ASPECT_RATIO,
    DIGIT_MAX_COUNT,
    DIGIT_MAX_HEIGHT,
    DIGIT_MAX_WIDTH,
    DIGIT_MIN_AREA,
    DIGIT_MIN_ASPECT_RATIO,
    DIGIT_MIN_HEIGHT,
    DIGIT_MIN_WIDTH,
    DIGIT_MORPH_KERNEL_SIZE,
    DIGIT_NORMALIZED_HEIGHT,
    DIGIT_NORMALIZED_MARGIN,
    DIGIT_NORMALIZED_WIDTH,
    DIGIT_ROI_MARGIN_DIVISOR,
    DIGIT_ROI_MIN_MARGIN,
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
):
    """把白底黑字模板转换为标准的黑底白字模板。"""
    if template_image is None:
        return None
    if len(template_image.shape) == 3:
        gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = template_image
    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
    )
    return normalize_digit(
        binary,
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
):
    """一次性加载并预处理 0 到 9 的模板。"""
    resolved_dir = resolve_template_dir(template_dir)
    loaded_templates = []
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
        )
        if normalized_template is None:
            raise RuntimeError("数字模板处理失败：{}".format(template_path))
        loaded_templates.append(normalized_template)

        if verbose:
            print("模板{}加载完成".format(digit))
        del template_image

    gc.collect()
    return tuple(loaded_templates), resolved_dir


def box_is_inside(box_a, box_b):
    ax, ay, aw, ah = box_a[0], box_a[1], box_a[2], box_a[3]
    bx, by, bw, bh = box_b[0], box_b[1], box_b[2], box_b[3]
    return (
        ax >= bx and
        ay >= by and
        ax + aw <= bx + bw and
        ay + ah <= by + bh
    )


def remove_contained_boxes(boxes):
    """删除完全包含在更大候选框中的内部轮廓。"""
    filtered = []
    for index_a in range(len(boxes)):
        contained = False
        for index_b in range(len(boxes)):
            if index_a == index_b:
                continue
            if box_is_inside(boxes[index_a], boxes[index_b]):
                area_a = boxes[index_a][2] * boxes[index_a][3]
                area_b = boxes[index_b][2] * boxes[index_b][3]
                if area_a < area_b:
                    contained = True
                    break
        if not contained:
            filtered.append(boxes[index_a])
    return filtered


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
        if roi_margin_divisor <= 0 or roi_min_margin < 0:
            raise ValueError("数字 ROI 边距参数无效")
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
        self.roi_margin_divisor = roi_margin_divisor
        self.roi_min_margin = roi_min_margin
        self.draw_recognized_color = draw_recognized_color
        self.draw_unknown_color = draw_unknown_color
        self.draw_text_color = draw_text_color
        self.draw_thickness = draw_thickness
        self.draw_summary = draw_summary
        self.morph_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (morph_kernel_size, morph_kernel_size),
        )

        self.templates, self.template_dir = load_templates(
            template_dir,
            normalized_width,
            normalized_height,
            normalized_margin,
            verbose,
        )
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
        return (
            x >= 0 and
            y >= 0 and
            x + width <= image_width and
            y + height <= image_height
        )

    def _recognize(self, normalized_digit):
        best_digit = -1
        best_score = -1.0
        for digit in range(10):
            match_result = cv2.matchTemplate(
                normalized_digit,
                self.templates[digit],
                cv2.TM_CCOEFF_NORMED,
            )
            _, max_value, _, _ = cv2.minMaxLoc(match_result)
            if max_value > best_score:
                best_score = max_value
                best_digit = digit
            del match_result

        if best_score < self.match_threshold:
            return -1, best_score
        return best_digit, best_score

    def detect(self, frame):
        """识别一帧并返回整串结果；没有数字候选时返回 None。"""
        start_ms = _ticks_ms()
        image_height = int(frame.shape[0])
        image_width = int(frame.shape[1])

        if len(frame.shape) == 2:
            gray = frame
        else:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self.blur_kernel_size, self.blur_kernel_size),
            0,
        )
        self.last_threshold, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU,
        )
        clean_binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            self.morph_kernel,
        )
        contours = find_contours(clean_binary)
        self.last_contour_count = len(contours)

        boxes = []
        for contour in contours:
            area = cv2.contourArea(contour)
            x, y, width, height = cv2.boundingRect(contour)
            if self._valid_digit_box(
                x,
                y,
                width,
                height,
                area,
                image_width,
                image_height,
            ):
                boxes.append((x, y, width, height, area))

        boxes = remove_contained_boxes(boxes)
        boxes.sort(key=lambda box: box[0])
        boxes = boxes[:self.max_count]

        digits = []
        for x, y, width, height, area in boxes:
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
            if roi_right <= roi_left or roi_bottom <= roi_top:
                continue

            digit_roi = clean_binary[roi_top:roi_bottom, roi_left:roi_right]
            normalized_digit = normalize_digit(
                digit_roi,
                self.normalized_width,
                self.normalized_height,
                self.normalized_margin,
            )
            if normalized_digit is None:
                continue

            value, score = self._recognize(normalized_digit)
            digits.append({
                "value": value,
                "text": str(value) if value >= 0 else "?",
                "recognized": value >= 0,
                "confidence": float(score),
                "x": x,
                "y": y,
                "w": width,
                "h": height,
                "area": float(area),
                "center_x": x + width // 2,
                "center_y": y + height // 2,
                "bbox": (x, y, width, height),
            })

            del normalized_digit
            del digit_roi

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
