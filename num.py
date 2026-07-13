
"""
实验名称：OpenCV 打印数字识别
开发板：01Studio CanMV K230 V3P0
固件：CanMV v1.8-2
摄像头：CSI2，Sensor id=2
显示方式：CanMV IDE

识别方式：
    OpenCV轮廓分割 + 模板匹配

模板目录：
    /sdcard/digit_templates/0.png
    /sdcard/digit_templates/1.png
    ...
    /sdcard/digit_templates/9.png
"""

import time
import sys
import gc

import cv2
import ulab.numpy as np

from camera_io import CameraIO
from config import (
    DIGIT_MATCH_THRESHOLD as MATCH_THRESHOLD,
    DIGIT_MAX_AREA as MAX_DIGIT_AREA,
    DIGIT_MAX_ASPECT_RATIO as MAX_ASPECT_RATIO,
    DIGIT_MAX_COUNT as MAX_DIGIT_COUNT,
    DIGIT_MAX_HEIGHT as MAX_DIGIT_HEIGHT,
    DIGIT_MAX_WIDTH as MAX_DIGIT_WIDTH,
    DIGIT_MIN_AREA as MIN_DIGIT_AREA,
    DIGIT_MIN_ASPECT_RATIO as MIN_ASPECT_RATIO,
    DIGIT_MIN_HEIGHT as MIN_DIGIT_HEIGHT,
    DIGIT_MIN_WIDTH as MIN_DIGIT_WIDTH,
    DIGIT_NORMALIZED_HEIGHT as NORMALIZED_HEIGHT,
    DIGIT_NORMALIZED_MARGIN as NORMALIZED_MARGIN,
    DIGIT_NORMALIZED_WIDTH as NORMALIZED_WIDTH,
    DIGIT_TEMPLATE_DIR as TEMPLATE_DIR,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    NUM_DISPLAY_FPS,
    NUM_DISPLAY_HEIGHT,
    NUM_DISPLAY_MODE,
    NUM_DISPLAY_QUALITY,
    NUM_DISPLAY_TO_IDE,
    NUM_DISPLAY_WIDTH,
    NUM_DISPLAY_X,
    NUM_DISPLAY_Y,
)


camera = None
templates = []


# ============================================================
# 兼容不同cv2版本的findContours返回格式
# ============================================================

def find_contours(binary_image):
    result = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # OpenCV常见返回：
    # contours, hierarchy
    if len(result) == 2:
        return result[0], result[1]

    # 某些版本返回：
    # image, contours, hierarchy
    return result[1], result[2]


# ============================================================
# 裁剪二值图中真正的数字区域
# ============================================================

def crop_foreground(binary_image):
    """
    输入：
        黑色背景、白色数字的二值图。

    返回：
        只包含数字主体的裁剪图。
    """

    contours, hierarchy = find_contours(binary_image)

    if contours is None or len(contours) == 0:
        return None

    # 将所有白色轮廓合并起来计算总外接范围。
    #
    # 不能只取最大轮廓，因为数字4、5等在某些字体和
    # 二值化结果中可能被分成多个部分。
    min_x = binary_image.shape[1]
    min_y = binary_image.shape[0]
    max_x = 0
    max_y = 0

    found = False

    for contour in contours:
        area = cv2.contourArea(contour)

        if area < 2:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if x < min_x:
            min_x = x

        if y < min_y:
            min_y = y

        if x + width > max_x:
            max_x = x + width

        if y + height > max_y:
            max_y = y + height

        found = True

    if not found:
        return None

    if max_x <= min_x or max_y <= min_y:
        return None

    cropped = binary_image[
        min_y:max_y,
        min_x:max_x
    ]

    return cropped


# ============================================================
# 将数字等比例缩放并居中
# ============================================================

def normalize_digit(binary_image):
    """
    将任意大小的数字区域，等比例缩放后放到固定画布中。

    输入：
        黑色背景、白色数字。

    输出：
        NORMALIZED_WIDTH × NORMALIZED_HEIGHT 的标准数字图。
    """

    cropped = crop_foreground(binary_image)

    if cropped is None:
        return None

    source_height = cropped.shape[0]
    source_width = cropped.shape[1]

    if source_width <= 0 or source_height <= 0:
        return None

    usable_width = NORMALIZED_WIDTH - NORMALIZED_MARGIN * 2
    usable_height = NORMALIZED_HEIGHT - NORMALIZED_MARGIN * 2

    scale_width = usable_width / source_width
    scale_height = usable_height / source_height

    # 使用较小的比例，保持原始宽高比
    scale = min(scale_width, scale_height)

    new_width = int(source_width * scale)
    new_height = int(source_height * scale)

    if new_width < 1:
        new_width = 1

    if new_height < 1:
        new_height = 1

    resized = cv2.resize(
        cropped,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # 创建黑色背景画布
    canvas = np.zeros(
        (NORMALIZED_HEIGHT, NORMALIZED_WIDTH),
        dtype=np.uint8
    )

    offset_x = (NORMALIZED_WIDTH - new_width) // 2
    offset_y = (NORMALIZED_HEIGHT - new_height) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width
    ] = resized

    return canvas


# ============================================================
# 预处理模板
# ============================================================

def prepare_template(template_image):
    """
    将电脑生成的白底黑字模板处理为：
        黑色背景、白色数字、固定尺寸。
    """

    if template_image is None:
        return None

    # cv2.imread通常返回BGR彩色图
    if len(template_image.shape) == 3:
        gray = cv2.cvtColor(
            template_image,
            cv2.COLOR_BGR2GRAY
        )
    else:
        gray = template_image

    # 白底黑字变为黑底白字
    threshold_value, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
    )

    normalized = normalize_digit(binary)

    return normalized


# ============================================================
# 加载0～9模板
# ============================================================

def load_templates():
    loaded_templates = []

    for digit in range(10):
        template_path = "{}/{}.png".format(
            TEMPLATE_DIR,
            digit
        )

        try:
            os.stat(template_path)
        except Exception:
            raise RuntimeError(
                "找不到数字模板：{}".format(template_path)
            )

        template_image = cv2.imread(
            template_path,
            cv2.IMREAD_COLOR
        )

        if template_image is None:
            raise RuntimeError(
                "无法读取数字模板：{}".format(template_path)
            )

        normalized_template = prepare_template(
            template_image
        )

        if normalized_template is None:
            raise RuntimeError(
                "数字模板处理失败：{}".format(template_path)
            )

        loaded_templates.append(normalized_template)

        print("模板{}加载完成".format(digit))

        del template_image

    gc.collect()

    return loaded_templates


# ============================================================
# 模板匹配识别
# ============================================================

def recognize_digit(normalized_digit):
    """
    将当前数字与0～9模板逐一比较。

    返回：
        best_digit：最相似的数字
        best_score：最高相似度
    """

    best_digit = -1
    best_score = -1.0

    for digit in range(10):
        result = cv2.matchTemplate(
            normalized_digit,
            templates[digit],
            cv2.TM_CCOEFF_NORMED
        )

        min_value, max_value, min_location, max_location = \
            cv2.minMaxLoc(result)

        if max_value > best_score:
            best_score = max_value
            best_digit = digit

        del result

    if best_score < MATCH_THRESHOLD:
        return -1, best_score

    return best_digit, best_score


# ============================================================
# 判断轮廓是否可能是数字
# ============================================================

def valid_digit_box(x, y, width, height, area):
    if area < MIN_DIGIT_AREA:
        return False

    if area > MAX_DIGIT_AREA:
        return False

    if width < MIN_DIGIT_WIDTH:
        return False

    if height < MIN_DIGIT_HEIGHT:
        return False

    if width > MAX_DIGIT_WIDTH:
        return False

    if height > MAX_DIGIT_HEIGHT:
        return False

    if height <= 0:
        return False

    aspect_ratio = width / height

    if aspect_ratio < MIN_ASPECT_RATIO:
        return False

    if aspect_ratio > MAX_ASPECT_RATIO:
        return False

    if x < 0 or y < 0:
        return False

    if x + width > IMAGE_WIDTH:
        return False

    if y + height > IMAGE_HEIGHT:
        return False

    return True


# ============================================================
# 判断两个框是否存在包含关系
#
# 数字8、0、6、9内部可能产生额外轮廓。
# RETR_EXTERNAL通常已经能避免，但这里再加一层保护。
# ============================================================

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


# ============================================================
# 主程序
# ============================================================

try:
    print("================================")
    print("K230 OpenCV打印数字识别")
    print("摄像头：CSI2")
    print("显示方式：CanMV IDE")
    print("================================")

    # --------------------------------------------------------
    # 加载模板
    # --------------------------------------------------------

    templates = load_templates()

    print("全部数字模板加载完成")

    camera = CameraIO(
        display_mode=NUM_DISPLAY_MODE,
        display_width=NUM_DISPLAY_WIDTH,
        display_height=NUM_DISPLAY_HEIGHT,
        display_fps=NUM_DISPLAY_FPS,
        to_ide=NUM_DISPLAY_TO_IDE,
        display_x=NUM_DISPLAY_X,
        display_y=NUM_DISPLAY_Y,
        quality=NUM_DISPLAY_QUALITY,
    )
    camera.initialize()

    clock = time.clock()
    frame_count = 0

    print("初始化完成，请将打印数字放到摄像头前")

    # ========================================================
    # 主循环
    # ========================================================

    while True:
        clock.tick()

        # ----------------------------------------------------
        # 获取摄像头图像
        # ----------------------------------------------------

        img = camera.snapshot()
        frame = img.to_numpy_ref()

        # ----------------------------------------------------
        # RGB转灰度
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        # ----------------------------------------------------
        # 轻微模糊，减少噪声
        # ----------------------------------------------------

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # ----------------------------------------------------
        # Otsu自动二值化
        #
        # 白纸上的黑字会变成：
        #   黑色背景
        #   白色数字
        # ----------------------------------------------------

        threshold_value, binary = cv2.threshold(
            blurred,
            0,
            255,
            cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU
        )

        # ----------------------------------------------------
        # 闭运算连接数字中可能断开的笔画
        # ----------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )

        clean_binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel
        )

        # ----------------------------------------------------
        # 查找数字轮廓
        # ----------------------------------------------------

        contours, hierarchy = find_contours(
            clean_binary
        )

        digit_boxes = []

        for contour in contours:
            area = cv2.contourArea(contour)

            x, y, width, height = cv2.boundingRect(
                contour
            )

            if not valid_digit_box(
                x,
                y,
                width,
                height,
                area
            ):
                continue

            digit_boxes.append(
                (x, y, width, height, area)
            )

        # 删除被其他框完全包含的小框
        digit_boxes = remove_contained_boxes(
            digit_boxes
        )

        # 按横坐标从左到右排序
        digit_boxes.sort(
            key=lambda box: box[0]
        )

        if len(digit_boxes) > MAX_DIGIT_COUNT:
            digit_boxes = digit_boxes[:MAX_DIGIT_COUNT]

        recognized_text = ""

        # ----------------------------------------------------
        # 逐个识别数字
        # ----------------------------------------------------

        for box in digit_boxes:
            x = box[0]
            y = box[1]
            width = box[2]
            height = box[3]

            # 给数字四周增加少量边距
            margin_x = max(2, width // 12)
            margin_y = max(2, height // 12)

            roi_left = max(0, x - margin_x)
            roi_top = max(0, y - margin_y)

            roi_right = min(
                IMAGE_WIDTH,
                x + width + margin_x
            )

            roi_bottom = min(
                IMAGE_HEIGHT,
                y + height + margin_y
            )

            if roi_right <= roi_left:
                continue

            if roi_bottom <= roi_top:
                continue

            # 从二值图中裁剪当前数字
            digit_roi = clean_binary[
                roi_top:roi_bottom,
                roi_left:roi_right
            ]

            normalized_digit = normalize_digit(
                digit_roi
            )

            if normalized_digit is None:
                continue

            digit, score = recognize_digit(
                normalized_digit
            )

            # ------------------------------------------------
            # 根据识别结果设置显示内容
            # ------------------------------------------------

            if digit >= 0:
                recognized_text += str(digit)

                label = "{} {:.2f}".format(
                    digit,
                    score
                )

                box_color = (0, 255, 0)

            else:
                recognized_text += "?"

                label = "? {:.2f}".format(
                    score
                )

                box_color = (255, 0, 0)

            # ------------------------------------------------
            # 绘制数字框
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + width, y + height),
                box_color,
                2
            )

            # 文字优先显示在框上方
            text_y = y - 8

            if text_y < 20:
                text_y = y + 22

            # 显示数字和匹配分数
            cv2.putText(
                frame,
                label,
                (x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                box_color,
                2
            )

            del digit_roi
            del normalized_digit

        # ----------------------------------------------------
        # 显示整串结果
        # ----------------------------------------------------

        cv2.putText(
            frame,
            "Digits: {}".format(recognized_text),
            (5, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "FPS: {:.1f}".format(clock.fps()),
            (5, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Count: {}".format(len(digit_boxes)),
            (5, 86),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        # ----------------------------------------------------
        # 输出到CanMV IDE
        # ----------------------------------------------------

        camera.show_image(img)

        frame_count += 1

        if frame_count % 30 == 0:
            print(
                "FPS: {:.2f}, Digits: {}, Count: {}".format(
                    clock.fps(),
                    recognized_text,
                    len(digit_boxes)
                )
            )

        # ----------------------------------------------------
        # 释放当前帧临时对象
        # ----------------------------------------------------

        del hierarchy
        del contours
        del digit_boxes
        del clean_binary
        del binary
        del kernel
        del blurred
        del gray
        del frame
        del img

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

    templates = []

    gc.collect()

    print("程序结束")
