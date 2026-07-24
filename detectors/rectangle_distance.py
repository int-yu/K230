"""方框距离检测程序。

检测流程符合本项目现有逻辑：

1. 复用 detectors.rectangle.RectangleDetector 检测黑框白心方框。
2. 复用 detectors.rectangle_corner_cycle.order_corners_clockwise 统一四角顺序。
3. 绘图只使用 cv2 和 core.draw_utils。
4. PnP 优先调用 cv2.solvePnP；如果 CanMV 固件没有暴露该函数，则使用本文件内的
   平面矩形 homography 后备测距。

返回距离单位为 cm。实际距离是否准确主要取决于：

- RECTANGLE_DISTANCE_OBJECT_WIDTH_CM / HEIGHT_CM 是否等于实物尺寸；
- RECTANGLE_DISTANCE_CAMERA_MATRIX 是否匹配当前摄像头、镜头和采集分辨率；
- RectangleDetector 返回的四个角点是否稳定落在同一个实际方框边界上。
"""

import gc
import math
import time

import cv2
import sys

# CanMV 按绝对路径启动子目录脚本时不会自动加入项目根目录。
for _path in ("/sdcard/K230", "/sdcard"):
    if _path not in sys.path:
        sys.path.append(_path)

from config import (
    DISPLAY_TARGET,
    RECTANGLE_DISTANCE_CAMERA_MATRIX,
    RECTANGLE_DISTANCE_DEMO_GC_INTERVAL,
    RECTANGLE_DISTANCE_DEMO_PRINT_INTERVAL,
    RECTANGLE_DISTANCE_DIST_COEFFS,
    RECTANGLE_DISTANCE_DRAW_ERROR_COLOR,
    RECTANGLE_DISTANCE_DRAW_POINT_COLOR,
    RECTANGLE_DISTANCE_DRAW_POINT_RADIUS,
    RECTANGLE_DISTANCE_DRAW_RECTANGLE_COLOR,
    RECTANGLE_DISTANCE_DRAW_ROI,
    RECTANGLE_DISTANCE_DRAW_ROI_COLOR,
    RECTANGLE_DISTANCE_DRAW_TEXT_COLOR,
    RECTANGLE_DISTANCE_DRAW_THICKNESS,
    RECTANGLE_DISTANCE_MIN_DISTANCE_CM,
    RECTANGLE_DISTANCE_OBJECT_HEIGHT_CM,
    RECTANGLE_DISTANCE_OBJECT_WIDTH_CM,
    RECTANGLE_DISTANCE_ROI,
)
from core.draw_utils import draw_cv_fps, draw_cv_text
from detectors.rectangle_corner_cycle import order_corners_clockwise
from detectors.rectangle import RectangleDetector, draw_frame_outline


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


def _clip(value, low, high):
    return max(low, min(high, value))


def _roi_ratio_to_rect(roi, width, height):
    x0 = int(_clip(float(roi[0]), 0.0, 1.0) * width)
    y0 = int(_clip(float(roi[1]), 0.0, 1.0) * height)
    x1 = int(_clip(float(roi[2]), 0.0, 1.0) * width)
    y1 = int(_clip(float(roi[3]), 0.0, 1.0) * height)
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    x0 = _clip(x0, 0, width - 1)
    y0 = _clip(y0, 0, height - 1)
    x1 = _clip(x1, x0 + 1, width)
    y1 = _clip(y1, y0 + 1, height)
    return (x0, y0, x1, y1)


def _point_inside_rect(point, rect):
    x, y = point
    x0, y0, x1, y1 = rect
    return x0 <= x <= x1 and y0 <= y <= y1


def _object_points(width_cm, height_cm):
    half_width = float(width_cm) / 2.0
    half_height = float(height_cm) / 2.0
    return (
        (-half_width, -half_height, 0.0),
        (half_width, -half_height, 0.0),
        (half_width, half_height, 0.0),
        (-half_width, half_height, 0.0),
    )


def _to_plain_nested(value):
    if isinstance(value, (list, tuple)):
        return [_to_plain_nested(item) for item in value]
    return float(value)


def _as_cv_array(value):
    """尽量生成 cv2.solvePnP 能接受的数组。

    PC OpenCV 使用 numpy；CanMV 上常见是 ulab.numpy。这里按可用性依次尝试，
    不把 numpy 作为上板必需依赖。
    """

    try:
        import numpy as np
        dtype = getattr(np, "float32", None)
        return np.array(value, dtype=dtype)
    except Exception:
        pass

    try:
        import ulab.numpy as np
        return np.array(value)
    except Exception:
        pass

    return _to_plain_nested(value)


def _call_cv2_solve_pnp(image_points, camera_matrix, dist_coeffs, width_cm, height_cm):
    """调用 cv2.solvePnP；固件未暴露时返回 None。"""

    solve_pnp = getattr(cv2, "solvePnP", None)
    if solve_pnp is None:
        return None

    object_points = _object_points(width_cm, height_cm)
    try:
        success, _rvec, tvec = solve_pnp(
            _as_cv_array(object_points),
            _as_cv_array(image_points),
            _as_cv_array((
                camera_matrix[0:3],
                camera_matrix[3:6],
                camera_matrix[6:9],
            )),
            _as_cv_array(dist_coeffs),
        )
        if not success:
            return None
        tx = float(tvec[0][0] if hasattr(tvec[0], "__len__") else tvec[0])
        ty = float(tvec[1][0] if hasattr(tvec[1], "__len__") else tvec[1])
        tz = float(tvec[2][0] if hasattr(tvec[2], "__len__") else tvec[2])
        distance_cm = math.sqrt(tx * tx + ty * ty + tz * tz)
        return {
            "distance_cm": distance_cm,
            "translation_cm": (tx, ty, tz),
            "method": "cv2.solvePnP",
        }
    except Exception:
        return None


def _normalize_image_point(point, camera_matrix, dist_coeffs):
    """把像素点转为归一化相机坐标，并用简单迭代消除径向/切向畸变。

    后备 homography PnP 使用归一化坐标求解。若畸变系数不可用，则等价于
    (x - cx) / fx、(y - cy) / fy。
    """

    fx = float(camera_matrix[0])
    fy = float(camera_matrix[4])
    cx = float(camera_matrix[2])
    cy = float(camera_matrix[5])
    x = (float(point[0]) - cx) / fx
    y = (float(point[1]) - cy) / fy

    coeffs = tuple(float(value) for value in dist_coeffs)
    if len(coeffs) < 4:
        return (x, y)

    k1 = coeffs[0]
    k2 = coeffs[1]
    p1 = coeffs[2]
    p2 = coeffs[3]
    k3 = coeffs[4] if len(coeffs) > 4 else 0.0

    distorted_x = x
    distorted_y = y
    for _ in range(5):
        r2 = x * x + y * y
        radial = 1.0 + k1 * r2 + k2 * r2 * r2 + k3 * r2 * r2 * r2
        if abs(radial) < 0.000001:
            break
        delta_x = 2.0 * p1 * x * y + p2 * (r2 + 2.0 * x * x)
        delta_y = p1 * (r2 + 2.0 * y * y) + 2.0 * p2 * x * y
        x = (distorted_x - delta_x) / radial
        y = (distorted_y - delta_y) / radial
    return (x, y)


def _solve_linear_system(matrix, vector):
    """高斯消元求解 Ax=b；matrix 为小规模 8x8，适合 MicroPython。"""

    size = len(vector)
    a = [list(row) + [float(vector[index])] for index, row in enumerate(matrix)]

    for column in range(size):
        pivot_row = column
        pivot_abs = abs(a[column][column])
        for row in range(column + 1, size):
            value_abs = abs(a[row][column])
            if value_abs > pivot_abs:
                pivot_abs = value_abs
                pivot_row = row
        if pivot_abs < 0.000000001:
            raise ValueError("homography 线性方程退化")
        if pivot_row != column:
            a[column], a[pivot_row] = a[pivot_row], a[column]

        pivot = a[column][column]
        for item in range(column, size + 1):
            a[column][item] /= pivot

        for row in range(size):
            if row == column:
                continue
            factor = a[row][column]
            if factor == 0:
                continue
            for item in range(column, size + 1):
                a[row][item] -= factor * a[column][item]

    return [a[row][size] for row in range(size)]


def _homography_from_four_points(object_points_2d, image_points_2d):
    """求平面物体坐标到归一化图像坐标的 3x3 homography。"""

    matrix = []
    vector = []
    for (object_x, object_y), (image_x, image_y) in zip(
        object_points_2d,
        image_points_2d,
    ):
        matrix.append([
            object_x, object_y, 1.0,
            0.0, 0.0, 0.0,
            -image_x * object_x, -image_x * object_y,
        ])
        vector.append(image_x)
        matrix.append([
            0.0, 0.0, 0.0,
            object_x, object_y, 1.0,
            -image_y * object_x, -image_y * object_y,
        ])
        vector.append(image_y)

    h = _solve_linear_system(matrix, vector)
    return (
        (h[0], h[1], h[2]),
        (h[3], h[4], h[5]),
        (h[6], h[7], 1.0),
    )


def _column_norm(homography, column):
    x = homography[0][column]
    y = homography[1][column]
    z = homography[2][column]
    return math.sqrt(x * x + y * y + z * z)


def _fallback_planar_pnp(image_points, camera_matrix, dist_coeffs, width_cm, height_cm):
    """不依赖 cv2.solvePnP 的平面矩形测距。

    这个后备算法利用平面标定关系：

        H = K [r1 r2 t]

    因为物体坐标以方框中心为原点，所以 t 的长度就是相机到方框中心的距离。
    """

    half_width = float(width_cm) / 2.0
    half_height = float(height_cm) / 2.0
    object_points_2d = (
        (-half_width, -half_height),
        (half_width, -half_height),
        (half_width, half_height),
        (-half_width, half_height),
    )
    normalized_points = tuple(
        _normalize_image_point(point, camera_matrix, dist_coeffs)
        for point in image_points
    )
    homography = _homography_from_four_points(object_points_2d, normalized_points)
    norm_1 = _column_norm(homography, 0)
    norm_2 = _column_norm(homography, 1)
    if norm_1 <= 0 or norm_2 <= 0:
        raise ValueError("homography 归一化失败")

    scale = 2.0 / (norm_1 + norm_2)
    tx = homography[0][2] * scale
    ty = homography[1][2] * scale
    tz = homography[2][2] * scale
    distance_cm = math.sqrt(tx * tx + ty * ty + tz * tz)
    return {
        "distance_cm": distance_cm,
        "translation_cm": (tx, ty, tz),
        "method": "planar_homography",
    }


def estimate_rectangle_distance(
    points,
    camera_matrix=RECTANGLE_DISTANCE_CAMERA_MATRIX,
    dist_coeffs=RECTANGLE_DISTANCE_DIST_COEFFS,
    object_width_cm=RECTANGLE_DISTANCE_OBJECT_WIDTH_CM,
    object_height_cm=RECTANGLE_DISTANCE_OBJECT_HEIGHT_CM,
):
    """根据四个角点估计方框中心距离。

    points 可以是任意顺序的四角点；函数内部会排序为左上、右上、右下、左下。
    """

    ordered_points = order_corners_clockwise(points)
    camera_matrix = tuple(float(value) for value in camera_matrix)
    dist_coeffs = tuple(float(value) for value in dist_coeffs)

    pnp_result = _call_cv2_solve_pnp(
        ordered_points,
        camera_matrix,
        dist_coeffs,
        object_width_cm,
        object_height_cm,
    )
    if pnp_result is None:
        pnp_result = _fallback_planar_pnp(
            ordered_points,
            camera_matrix,
            dist_coeffs,
            object_width_cm,
            object_height_cm,
        )

    pnp_result["points"] = ordered_points
    return pnp_result


class RectangleDistanceDetector:
    """检测方框并估算相机到方框中心的距离。"""

    def __init__(
        self,
        rectangle_detector=None,
        roi=RECTANGLE_DISTANCE_ROI,
        object_width_cm=RECTANGLE_DISTANCE_OBJECT_WIDTH_CM,
        object_height_cm=RECTANGLE_DISTANCE_OBJECT_HEIGHT_CM,
        camera_matrix=RECTANGLE_DISTANCE_CAMERA_MATRIX,
        dist_coeffs=RECTANGLE_DISTANCE_DIST_COEFFS,
        min_distance_cm=RECTANGLE_DISTANCE_MIN_DISTANCE_CM,
    ):
        if len(roi) != 4:
            raise ValueError("RECTANGLE_DISTANCE_ROI 必须包含 4 个比例值")
        if object_width_cm <= 0 or object_height_cm <= 0:
            raise ValueError("实际方框宽高必须大于 0 cm")
        if len(camera_matrix) != 9:
            raise ValueError("相机内参矩阵必须包含 9 个数")

        self.rectangle_detector = rectangle_detector or RectangleDetector()
        self.roi = tuple(roi)
        self.object_width_cm = float(object_width_cm)
        self.object_height_cm = float(object_height_cm)
        self.camera_matrix = tuple(float(value) for value in camera_matrix)
        self.dist_coeffs = tuple(float(value) for value in dist_coeffs)
        self.min_distance_cm = float(min_distance_cm)

        self.last_detection_ms = 0
        self.last_error = None
        self.last_result = None

    def detect(self, frame):
        """检测一帧；没有合格方框或测距失败时返回 None。"""

        start_ms = _ticks_ms()
        self.last_error = None

        frame_height = int(frame.shape[0])
        frame_width = int(frame.shape[1])
        search_roi = _roi_ratio_to_rect(self.roi, frame_width, frame_height)

        rectangle = self.rectangle_detector.detect(frame)
        if rectangle is None:
            self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
            self.last_result = None
            return None

        center = (int(rectangle["center_x"]), int(rectangle["center_y"]))
        if not _point_inside_rect(center, search_roi):
            self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
            self.last_result = None
            return None

        try:
            pnp_result = estimate_rectangle_distance(
                rectangle["points"],
                camera_matrix=self.camera_matrix,
                dist_coeffs=self.dist_coeffs,
                object_width_cm=self.object_width_cm,
                object_height_cm=self.object_height_cm,
            )
        except Exception as error:
            self.last_error = str(error)
            self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
            self.last_result = None
            return None

        distance_cm = float(pnp_result["distance_cm"])
        if distance_cm <= self.min_distance_cm:
            self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
            self.last_result = None
            return None

        result = dict(rectangle)
        rect = (
            int(result["x"]),
            int(result["y"]),
            int(result["w"]),
            int(result["h"]),
        )
        result.update({
            "distance_cm": distance_cm,
            "distance_m": distance_cm / 100.0,
            "translation_cm": pnp_result.get("translation_cm"),
            "pnp_method": pnp_result.get("method"),
            "points": pnp_result["points"],
            "corners": pnp_result["points"],
            "rect": rect,
            "search_roi": search_roi,
            "rectangle": rectangle,
            "error": None,
        })
        self.last_detection_ms = _ticks_diff(_ticks_ms(), start_ms)
        self.last_result = result
        return result

    def draw(self, frame, result):
        frame_height = int(frame.shape[0])
        frame_width = int(frame.shape[1])
        search_roi = _roi_ratio_to_rect(self.roi, frame_width, frame_height)

        if RECTANGLE_DISTANCE_DRAW_ROI:
            x0, y0, x1, y1 = search_roi
            cv2.rectangle(
                frame,
                (x0, y0),
                (x1, y1),
                RECTANGLE_DISTANCE_DRAW_ROI_COLOR,
                RECTANGLE_DISTANCE_DRAW_THICKNESS,
            )

        if result is None:
            text = "No Rect Found"
            if self.last_error:
                text = "PnP: {}".format(self.last_error[:24])
            draw_cv_text(
                frame,
                text,
                5,
                53,
                color=RECTANGLE_DISTANCE_DRAW_ERROR_COLOR,
            )
            return None

        draw_frame_outline(
            frame,
            result,
            RECTANGLE_DISTANCE_DRAW_RECTANGLE_COLOR,
            RECTANGLE_DISTANCE_DRAW_THICKNESS,
        )
        for point in result["points"]:
            px = int(point[0])
            py = int(point[1])
            radius = int(RECTANGLE_DISTANCE_DRAW_POINT_RADIUS)
            cv2.line(
                frame,
                (px - radius, py),
                (px + radius, py),
                RECTANGLE_DISTANCE_DRAW_POINT_COLOR,
                RECTANGLE_DISTANCE_DRAW_THICKNESS,
            )
            cv2.line(
                frame,
                (px, py - radius),
                (px, py + radius),
                RECTANGLE_DISTANCE_DRAW_POINT_COLOR,
                RECTANGLE_DISTANCE_DRAW_THICKNESS,
            )

        draw_cv_text(
            frame,
            "Dist: {:.2f}cm".format(result["distance_cm"]),
            5,
            53,
            color=RECTANGLE_DISTANCE_DRAW_TEXT_COLOR,
        )
        draw_cv_text(
            frame,
            "PnP: {}".format(result["pnp_method"]),
            5,
            81,
            color=RECTANGLE_DISTANCE_DRAW_TEXT_COLOR,
            scale=0.5,
            thickness=1,
        )
        return result

    def process(self, frame, draw=True):
        result = self.detect(frame)
        if draw:
            self.draw(frame, result)
        return result


def run_rectangle_distance_demo(display_target=None):
    """运行摄像头、显示和方框距离检测示例。"""

    from core.camera_io import CameraIO

    if display_target is None:
        display_target = DISPLAY_TARGET

    camera = None
    detector = RectangleDistanceDetector()

    try:
        print("================================")
        print("K230 方框距离检测")
        print("显示目标：{}".format(display_target))
        print("实际方框：{:.1f}cm x {:.1f}cm".format(
            RECTANGLE_DISTANCE_OBJECT_WIDTH_CM,
            RECTANGLE_DISTANCE_OBJECT_HEIGHT_CM,
        ))
        print("================================")

        camera = CameraIO(display_target=display_target)
        camera.initialize()
        clock = time.clock()
        frame_count = 0

        while True:
            clock.tick()
            image = camera.snapshot()
            frame = image.to_numpy_ref()

            result = detector.process(frame)
            fps = clock.fps()
            draw_cv_fps(frame, fps)
            camera.show_image(image)

            frame_count += 1
            if frame_count % RECTANGLE_DISTANCE_DEMO_PRINT_INTERVAL == 0:
                if result is None:
                    print("frame={} lost error={} fps={:.1f} detect={}ms".format(
                        frame_count,
                        detector.last_error,
                        fps,
                        detector.last_detection_ms,
                    ))
                else:
                    print(
                        "frame={} distance={:.2f}cm method={} confidence={:.3f} "
                        "fps={:.1f} detect={}ms".format(
                            frame_count,
                            result["distance_cm"],
                            result["pnp_method"],
                            result["confidence"],
                            fps,
                            detector.last_detection_ms,
                        )
                    )

            if frame_count % RECTANGLE_DISTANCE_DEMO_GC_INTERVAL == 0:
                gc.collect()
    except Exception:
        print("程序发生错误")
        import sys as _sys
        _sys.print_exception(_sys.exc_info()[1])
    finally:
        print("释放资源")
        if camera is not None:
            camera.deinitialize()
        print("程序结束")


# 历史公开名称别名。
SquareDistanceDetector = RectangleDistanceDetector
run_square_distance_demo = run_rectangle_distance_demo


if __name__ == "__main__":
    run_rectangle_distance_demo()
