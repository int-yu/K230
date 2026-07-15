"""K230 方框四角顺时针循环追踪应用。

运行流程：
    左上停留 3 秒 -> 3 秒移动到右上 -> 停留 3 秒 ->
    3 秒移动到右下 -> 停留 3 秒 -> 3 秒移动到左下 ->
    停留 3 秒 -> 3 秒回到左上，然后循环。

本文件是独立应用，不修改 RectangleDetector 或 TrackingUART 的职责。
"""

import gc
import math
import sys
import time

import cv2

from config import IMAGE_HEIGHT, IMAGE_WIDTH
from tangle import RectangleDetector, draw_frame_outline


CORNER_HOLD_MS = 3000
CORNER_MOVE_MS = 3000
UART_SEND_PERIOD_MS = 10

PRINT_INTERVAL = 30
GC_INTERVAL = 60

CORNER_NAMES = ("TL", "TR", "BR", "BL")
CORNER_COLORS = (
    (255, 0, 0),
    (0, 255, 0),
    (0, 128, 255),
    (255, 0, 255),
)
OUTLINE_COLOR = (0, 255, 0)
ACTIVE_COLOR = (255, 255, 0)
CENTER_COLOR = (255, 255, 255)


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


def order_corners_clockwise(points):
    """把四个角统一排列为左上、右上、右下、左下。"""
    if points is None or len(points) != 4:
        raise ValueError("方框必须包含 4 个角点")

    normalized = tuple(
        (int(point[0]), int(point[1]))
        for point in points
    )
    center_x = sum(point[0] for point in normalized) / 4.0
    center_y = sum(point[1] for point in normalized) / 4.0

    # 图像坐标 y 向下，atan2 角度递增方向正好是屏幕上的顺时针。
    clockwise = sorted(
        normalized,
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


class CornerCycleController:
    """只管理四角循环时间，不负责检测、摄像头或串口。"""

    def __init__(
        self,
        hold_ms=CORNER_HOLD_MS,
        move_ms=CORNER_MOVE_MS,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
    ):
        if hold_ms < 0:
            raise ValueError("角点停留时间不能小于 0")
        if move_ms <= 0:
            raise ValueError("角点移动时间必须大于 0")
        if image_width <= 0 or image_height <= 0:
            raise ValueError("画面尺寸必须大于 0")

        self.hold_ms = int(hold_ms)
        self.move_ms = int(move_ms)
        self.image_center_x = int(image_width) // 2
        self.image_center_y = int(image_height) // 2
        self._elapsed_valid_ms = 0
        self._last_update_ms = None
        self._last_frame_valid = False

    def reset(self):
        """从左上角停留阶段重新开始。"""
        self._elapsed_valid_ms = 0
        self._last_update_ms = None
        self._last_frame_valid = False

    def mark_target_lost(self, now_ms=None):
        """暂停循环计时；重新检测到目标后从当前阶段继续。"""
        if now_ms is None:
            now_ms = _ticks_ms()
        self._last_update_ms = now_ms
        self._last_frame_valid = False

    def update(self, points, now_ms=None):
        """根据当前四角返回本帧应该发送和绘制的轨迹点。"""
        if now_ms is None:
            now_ms = _ticks_ms()

        corners = order_corners_clockwise(points)
        if self._last_update_ms is not None and self._last_frame_valid:
            delta_ms = _ticks_diff(now_ms, self._last_update_ms)
            if delta_ms > 0:
                self._elapsed_valid_ms += delta_ms

        self._last_update_ms = now_ms
        self._last_frame_valid = True

        phase_span_ms = self.hold_ms + self.move_ms
        cycle_ms = phase_span_ms * 4
        cycle_elapsed_ms = self._elapsed_valid_ms % cycle_ms
        from_index = int(cycle_elapsed_ms // phase_span_ms)
        phase_elapsed_ms = cycle_elapsed_ms % phase_span_ms
        to_index = (from_index + 1) % 4

        if phase_elapsed_ms < self.hold_ms:
            phase = "hold"
            progress = 0.0
            active_x, active_y = corners[from_index]
            remaining_ms = self.hold_ms - phase_elapsed_ms
            target_index = from_index
        else:
            phase = "move"
            move_elapsed_ms = phase_elapsed_ms - self.hold_ms
            progress = min(1.0, move_elapsed_ms / float(self.move_ms))
            start_x, start_y = corners[from_index]
            end_x, end_y = corners[to_index]
            active_x = int(start_x + (end_x - start_x) * progress + 0.5)
            active_y = int(start_y + (end_y - start_y) * progress + 0.5)
            remaining_ms = self.move_ms - move_elapsed_ms
            target_index = to_index

        return {
            "corners": corners,
            "center_x": active_x,
            "center_y": active_y,
            "offset_x": self.image_center_x - active_x,
            "offset_y": self.image_center_y - active_y,
            "phase": phase,
            "progress": progress,
            "from_index": from_index,
            "target_index": target_index,
            "remaining_ms": int(remaining_ms),
            "cycle_elapsed_ms": int(cycle_elapsed_ms),
        }


def draw_image_center(frame):
    center = (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2)
    cv2.line(
        frame,
        (center[0] - 10, center[1]),
        (center[0] + 10, center[1]),
        CENTER_COLOR,
        2,
    )
    cv2.line(
        frame,
        (center[0], center[1] - 10),
        (center[0], center[1] + 10),
        CENTER_COLOR,
        2,
    )


def draw_corner_cycle(frame, rectangle, state):
    """绘制方框、四个已排序角点和当前插值点。"""
    draw_frame_outline(frame, rectangle, OUTLINE_COLOR, thickness=2)

    for index, point in enumerate(state["corners"]):
        color = CORNER_COLORS[index]
        cv2.circle(frame, point, 6, color, -1)
        label_x = min(IMAGE_WIDTH - 55, max(0, point[0] + 7))
        label_y = min(IMAGE_HEIGHT - 5, max(18, point[1] - 7))
        cv2.putText(
            frame,
            "{} {}".format(index + 1, CORNER_NAMES[index]),
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            2,
        )

    active_point = (state["center_x"], state["center_y"])
    cv2.circle(frame, active_point, 9, ACTIVE_COLOR, 2)
    cv2.circle(frame, active_point, 3, ACTIVE_COLOR, -1)
    cv2.line(
        frame,
        (IMAGE_WIDTH // 2, IMAGE_HEIGHT // 2),
        active_point,
        ACTIVE_COLOR,
        1,
    )

    if state["phase"] == "hold":
        status = "HOLD {} {:.1f}s".format(
            CORNER_NAMES[state["from_index"]],
            state["remaining_ms"] / 1000.0,
        )
    else:
        status = "MOVE {}>{} {:.1f}s".format(
            CORNER_NAMES[state["from_index"]],
            CORNER_NAMES[state["target_index"]],
            state["remaining_ms"] / 1000.0,
        )

    cv2.putText(
        frame,
        status,
        (5, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        ACTIVE_COLOR,
        2,
    )
    cv2.putText(
        frame,
        "X:{} Y:{}".format(state["offset_x"], state["offset_y"]),
        (5, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        ACTIVE_COLOR,
        2,
    )


def run_corner_cycle(
    display_target=None,
    hold_ms=CORNER_HOLD_MS,
    move_ms=CORNER_MOVE_MS,
    uart_send_period_ms=UART_SEND_PERIOD_MS,
):
    """运行完整摄像头、显示和四角串口循环应用。"""
    from camera_io import CameraIO, DISPLAY_TARGET_IDE
    from uart_io import TrackingUART

    if display_target is None:
        display_target = DISPLAY_TARGET_IDE

    camera = None
    tracking_uart = None
    controller = CornerCycleController(
        hold_ms=hold_ms,
        move_ms=move_ms,
    )
    detector = RectangleDetector()

    try:
        print("初始化四角循环串口")
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

        print("初始化方框检测器和摄像头")
        camera = CameraIO(display_target=display_target)
        camera.initialize()
        clock = time.clock()
        frame_count = 0
        print("四角顺序：TL -> TR -> BR -> BL -> TL")
        print("停留={}ms，移动={}ms".format(hold_ms, move_ms))

        while True:
            clock.tick()
            image = camera.snapshot()
            frame = image.to_numpy_ref()
            now_ms = _ticks_ms()
            rectangle = detector.process(frame, draw=False)

            draw_image_center(frame)
            state = None
            if rectangle is not None:
                state = controller.update(rectangle["points"], now_ms)
                draw_corner_cycle(frame, rectangle, state)
                tracking_uart.send_target(
                    True,
                    state["offset_x"],
                    state["offset_y"],
                    frame_id=frame_count,
                )
            else:
                controller.mark_target_lost(now_ms)
                tracking_uart.send_target(
                    False,
                    0,
                    0,
                    frame_id=frame_count,
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
                CENTER_COLOR,
                2,
            )
            camera.show_image(image)
            frame_count += 1

            if frame_count % PRINT_INTERVAL == 0:
                if state is None:
                    print("frame={} target=lost fps={:.1f}".format(
                        frame_count,
                        fps,
                    ))
                else:
                    print(
                        "frame={} phase={} point={} x={} y={} fps={:.1f}".format(
                            frame_count,
                            state["phase"],
                            CORNER_NAMES[state["target_index"]],
                            state["offset_x"],
                            state["offset_y"],
                            fps,
                        )
                    )

            del frame
            del image
            if frame_count % GC_INTERVAL == 0:
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
    run_corner_cycle()
