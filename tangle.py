"""
K230 OpenCV 最佳矩形检测、中心点定位与短暂丢失保持

平台：01Studio CanMV K230 V3P0
固件：CanMV v1.8-2
摄像头：CSI2
显示：板载 3.5 寸 ST7701 屏幕

功能：
1. 检测画面中的矩形候选
2. 计算每个矩形的可信度
3. 选择可信度最高的矩形
4. 在板载屏幕绘制最佳矩形、中心点和画面中心
5. 在终端输出目标中心相对于画面中心的坐标
6. 短时间漏检时继续保留上一次检测结果

相对坐标规则：
    画面中心为 (0, 0)
    向右为 x 正方向
    向左为 x 负方向
    向上为 y 正方向
    向下为 y 负方向
"""

import time
import sys
import gc
import math

import cv2

from camera_io import CameraIO
from config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RECTANGLE_APPROX_RATIO as APPROX_RATIO,
    RECTANGLE_CANNY_HIGH as CANNY_HIGH,
    RECTANGLE_CANNY_LOW as CANNY_LOW,
    RECTANGLE_LOST_FRAME_LIMIT as LOST_FRAME_LIMIT,
    RECTANGLE_MAX_COUNT as MAX_RECT_COUNT,
    RECTANGLE_MIN_AREA as MIN_AREA,
    RECTANGLE_MIN_CONFIDENCE as MIN_CONFIDENCE,
    RECTANGLE_MIN_HEIGHT as MIN_HEIGHT,
    RECTANGLE_MIN_WIDTH as MIN_WIDTH,
    TANGLE_DISPLAY_FPS,
    TANGLE_DISPLAY_HEIGHT as LCD_HEIGHT,
    TANGLE_DISPLAY_MODE,
    TANGLE_DISPLAY_TO_IDE,
    TANGLE_DISPLAY_WIDTH as LCD_WIDTH,
    TANGLE_DISPLAY_X as DISPLAY_X,
    TANGLE_DISPLAY_Y as DISPLAY_Y,
)


# 摄像头画面中心坐标
IMAGE_CENTER_X = IMAGE_WIDTH // 2
IMAGE_CENTER_Y = IMAGE_HEIGHT // 2

# 上一次有效目标
last_rectangle = None

# 上一次有效目标的可信度
last_confidence = 0.0

# 已经连续漏检的帧数
lost_frame_count = 0


# ============================================================
# 输出参数
# ============================================================

# 每隔多少帧向终端输出一次坐标
PRINT_INTERVAL = 10

# 目标中心点圆半径
CENTER_POINT_RADIUS = 6

# 摄像头画面中心十字长度
CENTER_CROSS_SIZE = 12


camera = None


# ============================================================
# 获取多边形中的四个角点
# ============================================================

def get_four_points(approx):
    """
    将 OpenCV 多边形数据转换为四个二维坐标。

    返回：
        [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
    """

    points = []

    for point in approx:
        # 不同 OpenCV 移植版本可能返回：
        # [[x, y]]
        # 或 [x, y]
        try:
            x = int(point[0][0])
            y = int(point[0][1])

        except Exception:
            x = int(point[0])
            y = int(point[1])

        points.append((x, y))

    return points


# ============================================================
# 计算三个点构成角的余弦绝对值
# ============================================================

def angle_cosine(point_a, point_b, point_c):
    """
    计算角 ABC 的余弦绝对值。

    point_b 是角的顶点。

    当角度接近 90° 时：
        cos(90°) = 0

    所以返回值越接近 0，说明越接近直角。
    """

    vector_1_x = point_a[0] - point_b[0]
    vector_1_y = point_a[1] - point_b[1]

    vector_2_x = point_c[0] - point_b[0]
    vector_2_y = point_c[1] - point_b[1]

    dot_product = (
        vector_1_x * vector_2_x +
        vector_1_y * vector_2_y
    )

    length_1 = math.sqrt(
        vector_1_x * vector_1_x +
        vector_1_y * vector_1_y
    )

    length_2 = math.sqrt(
        vector_2_x * vector_2_x +
        vector_2_y * vector_2_y
    )

    if length_1 <= 0 or length_2 <= 0:
        return 1.0

    cosine = dot_product / (
        length_1 * length_2
    )

    return abs(cosine)


# ============================================================
# 将四个角点按顺时针排列
# ============================================================

def sort_points_clockwise(points):
    """
    根据角点相对于几何中心的方向排序。

    排序后，相邻元素对应矩形中的相邻顶点，
    方便后面依次计算四个角。
    """

    center_x = (
        sum(point[0] for point in points) /
        len(points)
    )

    center_y = (
        sum(point[1] for point in points) /
        len(points)
    )

    return sorted(
        points,
        key=lambda point: math.atan2(
            point[1] - center_y,
            point[0] - center_x
        )
    )


# ============================================================
# 计算矩形可信度
# ============================================================

def calculate_rectangle_confidence(
    contour,
    approx,
    x,
    y,
    width,
    height
):
    """
    返回 0～1 的矩形可信度。

    可信度包括：

    1. 填充率
       轮廓面积 / 水平外接框面积

    2. 角度规则度
       四个角越接近 90°，分数越高
    """

    bounding_area = width * height

    if bounding_area <= 0:
        return 0.0

    contour_area = cv2.contourArea(contour)

    # --------------------------------------------------------
    # 填充率
    # --------------------------------------------------------

    fill_ratio = contour_area / bounding_area

    if fill_ratio < 0:
        fill_ratio = 0.0

    if fill_ratio > 1:
        fill_ratio = 1.0

    # --------------------------------------------------------
    # 四角规则度
    # --------------------------------------------------------

    points = get_four_points(approx)

    if len(points) != 4:
        return 0.0

    points = sort_points_clockwise(points)

    cosine_sum = 0.0

    for index in range(4):
        previous_point = points[(index - 1) % 4]
        current_point = points[index]
        next_point = points[(index + 1) % 4]

        cosine_sum += angle_cosine(
            previous_point,
            current_point,
            next_point
        )

    average_cosine = cosine_sum / 4.0

    # 90°的余弦为 0
    # 因此用 1 - 平均余弦作为角度得分
    angle_score = 1.0 - average_cosine

    if angle_score < 0:
        angle_score = 0.0

    if angle_score > 1:
        angle_score = 1.0

    # 填充率和角度各占 50%
    confidence = (
        fill_ratio * 0.5 +
        angle_score * 0.5
    )

    return confidence


# ============================================================
# 绘制摄像头画面中心
# ============================================================

def draw_image_center(frame):
    """
    在摄像头画面中心绘制白色十字和中心点。
    """

    # 横线
    cv2.line(
        frame,
        (
            IMAGE_CENTER_X - CENTER_CROSS_SIZE,
            IMAGE_CENTER_Y
        ),
        (
            IMAGE_CENTER_X + CENTER_CROSS_SIZE,
            IMAGE_CENTER_Y
        ),
        (255, 255, 255),
        2
    )

    # 竖线
    cv2.line(
        frame,
        (
            IMAGE_CENTER_X,
            IMAGE_CENTER_Y - CENTER_CROSS_SIZE
        ),
        (
            IMAGE_CENTER_X,
            IMAGE_CENTER_Y + CENTER_CROSS_SIZE
        ),
        (255, 255, 255),
        2
    )

    # 中心点
    cv2.circle(
        frame,
        (
            IMAGE_CENTER_X,
            IMAGE_CENTER_Y
        ),
        3,
        (255, 255, 255),
        -1
    )


# ============================================================
# 主程序
# ============================================================

try:
    print("初始化 CSI2 摄像头")

    camera = CameraIO(
        display_mode=TANGLE_DISPLAY_MODE,
        display_width=LCD_WIDTH,
        display_height=LCD_HEIGHT,
        display_fps=TANGLE_DISPLAY_FPS,
        to_ide=TANGLE_DISPLAY_TO_IDE,
        display_x=DISPLAY_X,
        display_y=DISPLAY_Y,
    )
    camera.initialize()

    clock = time.clock()
    frame_count = 0

    print("初始化完成")

    print(
        "摄像头画面中心：({}, {})".format(
            IMAGE_CENTER_X,
            IMAGE_CENTER_Y
        )
    )

    print(
        "LCD 显示偏移：x={}, y={}".format(
            DISPLAY_X,
            DISPLAY_Y
        )
    )

    print(
        "目标连续丢失超过 {} 帧后清除".format(
            LOST_FRAME_LIMIT
        )
    )

    # ========================================================
    # 主循环
    # ========================================================

    while True:
        clock.tick()

        # ----------------------------------------------------
        # 获取摄像头图像
        # ----------------------------------------------------

        img = camera.snapshot()

        # 获取 RGB888 ndarray 引用
        frame = img.to_numpy_ref()

        # ----------------------------------------------------
        # 灰度化
        # ----------------------------------------------------

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_RGB2GRAY
        )

        # ----------------------------------------------------
        # 高斯滤波
        # ----------------------------------------------------

        blurred = cv2.GaussianBlur(
            gray,
            (5, 5),
            0
        )

        # ----------------------------------------------------
        # Canny 边缘检测
        # ----------------------------------------------------

        edges = cv2.Canny(
            blurred,
            CANNY_LOW,
            CANNY_HIGH
        )

        # ----------------------------------------------------
        # 膨胀边缘
        # ----------------------------------------------------

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (3, 3)
        )

        edges = cv2.dilate(
            edges,
            kernel,
            iterations=1
        )

        # ----------------------------------------------------
        # 查找最外层轮廓
        # ----------------------------------------------------

        contours, hierarchy = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        rectangle_count = 0

        # 当前帧检测到的最佳矩形
        best_rectangle = None
        best_confidence = 0.0

        # 标记当前显示的目标是否来自历史保留
        target_is_held = False

        # ====================================================
        # 遍历所有候选轮廓
        # ====================================================

        for contour in contours:
            if rectangle_count >= MAX_RECT_COUNT:
                break

            area = cv2.contourArea(contour)

            if area < MIN_AREA:
                continue

            perimeter = cv2.arcLength(
                contour,
                True
            )

            if perimeter <= 0:
                continue

            # 将轮廓拟合为多边形
            approx = cv2.approxPolyDP(
                contour,
                APPROX_RATIO * perimeter,
                True
            )

            # 矩形应当有四个顶点
            if len(approx) != 4:
                continue

            # 矩形应当是凸四边形
            if not cv2.isContourConvex(approx):
                continue

            x, y, width, height = cv2.boundingRect(
                approx
            )

            # 尺寸过滤
            if (
                width < MIN_WIDTH or
                height < MIN_HEIGHT
            ):
                continue

            # 坐标边界过滤
            if x < 0 or y < 0:
                continue

            if x + width > IMAGE_WIDTH:
                continue

            if y + height > IMAGE_HEIGHT:
                continue

            rectangle_count += 1

            confidence = calculate_rectangle_confidence(
                contour,
                approx,
                x,
                y,
                width,
                height
            )

            # 保存当前帧可信度最高的矩形
            if confidence > best_confidence:
                best_confidence = confidence

                best_rectangle = {
                    "x": x,
                    "y": y,
                    "w": width,
                    "h": height,
                    "approx": approx,
                    "confidence": confidence
                }

        # ====================================================
        # 目标短暂保留逻辑
        # ====================================================

        if (
            best_rectangle is not None and
            best_confidence >= MIN_CONFIDENCE
        ):
            # 当前帧正常检测到了有效目标
            last_rectangle = best_rectangle
            last_confidence = best_confidence

            # 清零连续漏检计数
            lost_frame_count = 0

            target_is_held = False

        else:
            # 当前帧没有检测到有效目标
            lost_frame_count += 1

            if (
                last_rectangle is not None and
                lost_frame_count <= LOST_FRAME_LIMIT
            ):
                # 漏检时间还没有超过限制
                # 继续使用上一次目标
                best_rectangle = last_rectangle
                best_confidence = last_confidence

                target_is_held = True

            else:
                # 连续漏检超过限制，真正清除目标
                best_rectangle = None
                best_confidence = 0.0

                last_rectangle = None
                last_confidence = 0.0

                target_is_held = False

        # ====================================================
        # 绘制摄像头画面中心
        # ====================================================

        draw_image_center(frame)

        # 默认没有有效坐标
        relative_x = None
        relative_y = None

        # ====================================================
        # 绘制当前目标
        # ====================================================

        if best_rectangle is not None:
            x = best_rectangle["x"]
            y = best_rectangle["y"]
            width = best_rectangle["w"]
            height = best_rectangle["h"]
            approx = best_rectangle["approx"]

            # ------------------------------------------------
            # 目标中心的绝对像素坐标
            # ------------------------------------------------

            target_center_x = (
                x + width // 2
            )

            target_center_y = (
                y + height // 2
            )

            # ------------------------------------------------
            # 目标中心相对于画面中心的坐标
            #
            # x：
            #   向右为正
            #   向左为负
            #
            # y：
            #   向上为正
            #   向下为负
            # ------------------------------------------------

            relative_x = (
                target_center_x -
                IMAGE_CENTER_X
            )

            relative_y = (
                IMAGE_CENTER_Y -
                target_center_y
            )

            # ------------------------------------------------
            # 根据目标来源决定颜色
            #
            # 正常检测：
            #   绿色框
            #
            # 历史保留：
            #   黄色框
            # ------------------------------------------------

            if target_is_held:
                box_color = (255, 255, 0)
            else:
                box_color = (0, 255, 0)

            # ------------------------------------------------
            # 绘制拟合四边形
            # ------------------------------------------------

            cv2.polylines(
                frame,
                [approx],
                True,
                (255, 0, 0),
                3
            )

            # ------------------------------------------------
            # 绘制水平外接框
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (
                    x + width,
                    y + height
                ),
                box_color,
                2
            )

            # ------------------------------------------------
            # 绘制目标中心点
            # ------------------------------------------------

            cv2.circle(
                frame,
                (
                    target_center_x,
                    target_center_y
                ),
                CENTER_POINT_RADIUS,
                (255, 255, 0),
                -1
            )

            # ------------------------------------------------
            # 从画面中心画线到目标中心
            # ------------------------------------------------

            cv2.line(
                frame,
                (
                    IMAGE_CENTER_X,
                    IMAGE_CENTER_Y
                ),
                (
                    target_center_x,
                    target_center_y
                ),
                (255, 255, 0),
                2
            )

            # ------------------------------------------------
            # 设置文字位置
            # ------------------------------------------------

            label_y = y - 10

            if label_y < 20:
                label_y = y + 22

            # ------------------------------------------------
            # 显示可信度或保持状态
            # ------------------------------------------------

            if target_is_held:
                state_text = "HOLD {}/{}".format(
                    lost_frame_count,
                    LOST_FRAME_LIMIT
                )

            else:
                state_text = "Conf: {:.2f}".format(
                    best_confidence
                )

            cv2.putText(
                frame,
                state_text,
                (x, label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                box_color,
                2
            )

            # ------------------------------------------------
            # 显示相对坐标
            # ------------------------------------------------

            coordinate_text_x = (
                target_center_x + 8
            )

            coordinate_text_y = (
                target_center_y - 8
            )

            # 防止文字超出右边界
            if coordinate_text_x > IMAGE_WIDTH - 150:
                coordinate_text_x = max(
                    0,
                    target_center_x - 145
                )

            # 防止文字超出顶部
            if coordinate_text_y < 20:
                coordinate_text_y = (
                    target_center_y + 25
                )

            cv2.putText(
                frame,
                "X:{} Y:{}".format(
                    relative_x,
                    relative_y
                ),
                (
                    coordinate_text_x,
                    coordinate_text_y
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2
            )

        else:
            # ------------------------------------------------
            # 没有目标
            # ------------------------------------------------

            cv2.putText(
                frame,
                "No valid rectangle",
                (5, 115),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 0, 0),
                2
            )

        # ====================================================
        # 显示基本信息
        # ====================================================

        cv2.putText(
            frame,
            "FPS: {:.1f}".format(
                clock.fps()
            ),
            (5, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "Candidates: {}".format(
                rectangle_count
            ),
            (5, 55),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Best: {:.2f}".format(
                best_confidence
            ),
            (5, 82),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 0),
            2
        )

        # 显示当前连续漏检帧数
        cv2.putText(
            frame,
            "Lost: {}/{}".format(
                lost_frame_count,
                LOST_FRAME_LIMIT
            ),
            (5, 109),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # ====================================================
        # 显示到板载屏幕
        # ====================================================

        camera.show_image(img)

        frame_count += 1

        # ====================================================
        # 终端输出相对坐标
        # ====================================================

        if frame_count % PRINT_INTERVAL == 0:
            if relative_x is not None:
                if target_is_held:
                    state = "保持"
                else:
                    state = "实时"

                print(
                    "中心相对坐标: x={}, y={}, "
                    "confidence={:.3f}, "
                    "state={}, lost={}/{}".format(
                        relative_x,
                        relative_y,
                        best_confidence,
                        state,
                        lost_frame_count,
                        LOST_FRAME_LIMIT
                    )
                )

            else:
                print(
                    "未检测到目标，"
                    "lost={}/{}".format(
                        lost_frame_count,
                        LOST_FRAME_LIMIT
                    )
                )

        # ====================================================
        # 释放当前帧临时变量
        # ====================================================

        del hierarchy
        del contours
        del edges
        del kernel
        del blurred
        del gray
        del frame
        del img

        # 每 30 帧进行一次垃圾回收
        if frame_count % 30 == 0:
            gc.collect()


except KeyboardInterrupt:
    print("用户停止程序")


except Exception as error:
    print("程序发生错误")
    sys.print_exception(error)


finally:
    print("释放资源")

    if camera is not None:
        camera.deinitialize()

    last_rectangle = None

    gc.collect()

    print("程序结束")
