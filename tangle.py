"""K230 黑框白心方框追踪与串口输出。

平台：01Studio CanMV K230 V3P0
摄像头：CSI2
显示：由 CameraIO 选择 CanMV IDE

串口协议：T,frame,valid,x,y\n
坐标原点位于画面中心，x 向右为正，y 向上为正。
"""

import gc
import sys
import time

import cv2
from machine import FPIOA, UART

from camera_io import CameraIO, DISPLAY_TARGET_IDE
from config import (
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    RECTANGLE_LOST_FRAME_LIMIT,
    TRACK_UART_BAUDRATE,
    TRACK_UART_ID,
    TRACK_UART_RX_PIN,
    TRACK_UART_TX_PIN,
)
from rectangle_detector import BlackWhiteFrameDetector, draw_frame_outline


IMAGE_CENTER_X = IMAGE_WIDTH // 2
IMAGE_CENTER_Y = IMAGE_HEIGHT // 2

PRINT_INTERVAL = 10
GC_INTERVAL = 60
CENTER_POINT_RADIUS = 6
CENTER_CROSS_SIZE = 12


camera = None
tracking_uart = None


class TargetHoldState:
    """仅负责显示端的短暂丢失保持，不向串口发送历史坐标。"""

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


def initialize_tracking_uart():
    """初始化用于向单片机发送追踪数据的串口。"""
    fpioa = FPIOA()

    if TRACK_UART_ID == 1:
        uart_channel = UART.UART1
        tx_function = FPIOA.UART1_TXD
        rx_function = FPIOA.UART1_RXD
    elif TRACK_UART_ID == 2:
        uart_channel = UART.UART2
        tx_function = FPIOA.UART2_TXD
        rx_function = FPIOA.UART2_RXD
    else:
        raise ValueError("当前程序仅配置 UART1 或 UART2")

    fpioa.set_function(TRACK_UART_TX_PIN, tx_function)
    fpioa.set_function(TRACK_UART_RX_PIN, rx_function)

    return UART(
        uart_channel,
        baudrate=TRACK_UART_BAUDRATE,
        bits=UART.EIGHTBITS,
        parity=UART.PARITY_NONE,
        stop=UART.STOPBITS_ONE,
    )


def send_target_offset(frame_id, valid, offset_x, offset_y):
    """发送一行 ASCII：T,frame,valid,x,y。"""
    if not valid:
        offset_x = 0
        offset_y = 0

    packet = "T,{},{},{},{}\n".format(
        int(frame_id),
        1 if valid else 0,
        int(offset_x),
        int(offset_y),
    )
    tracking_uart.write(packet)


def draw_image_center(frame):
    """绘制画面中心十字。"""
    color = (255, 255, 255)
    cv2.line(
        frame,
        (IMAGE_CENTER_X - CENTER_CROSS_SIZE, IMAGE_CENTER_Y),
        (IMAGE_CENTER_X + CENTER_CROSS_SIZE, IMAGE_CENTER_Y),
        color,
        2,
    )
    cv2.line(
        frame,
        (IMAGE_CENTER_X, IMAGE_CENTER_Y - CENTER_CROSS_SIZE),
        (IMAGE_CENTER_X, IMAGE_CENTER_Y + CENTER_CROSS_SIZE),
        color,
        2,
    )
    cv2.circle(frame, (IMAGE_CENTER_X, IMAGE_CENTER_Y), 3, color, -1)


def target_relative_offset(target):
    """计算目标中心相对于画面中心的坐标。"""
    relative_x = target["center_x"] - IMAGE_CENTER_X
    relative_y = IMAGE_CENTER_Y - target["center_y"]
    return relative_x, relative_y


def draw_target(frame, target, relative_x, relative_y, is_held):
    """绘制目标外框、中心点和必要坐标，避免大量调试文字。"""
    center = (target["center_x"], target["center_y"])
    box_color = (255, 255, 0) if is_held else (0, 255, 0)

    draw_frame_outline(frame, target, box_color, thickness=2)
    cv2.circle(
        frame,
        center,
        CENTER_POINT_RADIUS,
        (255, 255, 0),
        -1,
    )
    cv2.line(
        frame,
        (IMAGE_CENTER_X, IMAGE_CENTER_Y),
        center,
        (255, 255, 0),
        1,
    )

    label_x = min(max(0, center[0] + 8), IMAGE_WIDTH - 145)
    label_y = max(20, center[1] - 8)
    cv2.putText(
        frame,
        "X:{} Y:{}".format(relative_x, relative_y),
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2,
    )


def print_tracking_status(
    frame_count,
    target,
    relative_x,
    relative_y,
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
            relative_x,
            relative_y,
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


try:
    print("初始化追踪串口")
    tracking_uart = initialize_tracking_uart()
    print(
        "UART{}：TX=GPIO{}，RX=GPIO{}，{} baud".format(
            TRACK_UART_ID,
            TRACK_UART_TX_PIN,
            TRACK_UART_RX_PIN,
            TRACK_UART_BAUDRATE,
        )
    )

    print("初始化黑框白心检测器")
    detector = BlackWhiteFrameDetector()
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

        current_target = detector.detect(frame)
        display_target, target_is_held = hold_state.update(current_target)

        draw_image_center(frame)
        relative_x = None
        relative_y = None

        if display_target is not None:
            relative_x, relative_y = target_relative_offset(display_target)
            draw_target(
                frame,
                display_target,
                relative_x,
                relative_y,
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

        target_valid = current_target is not None
        send_target_offset(
            frame_count,
            target_valid,
            relative_x if target_valid else 0,
            relative_y if target_valid else 0,
        )

        camera.show_image(image)
        frame_count += 1

        if frame_count % PRINT_INTERVAL == 0:
            print_tracking_status(
                frame_count,
                display_target,
                relative_x,
                relative_y,
                target_is_held,
                fps,
                detector,
                hold_state,
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
        try:
            tracking_uart.deinit()
        except Exception:
            pass

    if camera is not None:
        camera.deinitialize()

    try:
        hold_state.clear()
    except Exception:
        pass

    gc.collect()
    print("程序结束")
