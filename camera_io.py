"""K230 摄像头、显示和 MediaManager 生命周期管理。"""

import gc
import os
import time

from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager

from config import (
    CAMERA_FPS,
    CAMERA_HMIRROR,
    CAMERA_ID,
    CAMERA_PIXEL_FORMAT,
    CAMERA_SOURCE_HEIGHT,
    CAMERA_SOURCE_WIDTH,
    CAMERA_VFLIP,
    DISPLAY_MODE_ST7701,
    DISPLAY_MODE_VIRT,
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
    TANGLE_DISPLAY_FPS,
    TANGLE_DISPLAY_HEIGHT,
    TANGLE_DISPLAY_MODE,
    TANGLE_DISPLAY_TO_IDE,
    TANGLE_DISPLAY_WIDTH,
    TANGLE_DISPLAY_X,
    TANGLE_DISPLAY_Y,
)


DISPLAY_TARGET_BOARD = "board"
DISPLAY_TARGET_IDE = "ide"


class CameraIO:
    """统一管理摄像头采集、显示输出和媒体资源。"""

    def __init__(self, display_target=DISPLAY_TARGET_BOARD):
        self.display_target = display_target
        self._configure_display(display_target)

        self.sensor = None
        self._running = False
        self._resources_active = False

    def initialize(self):
        """初始化摄像头、显示和媒体缓冲区。"""

        if self._resources_active or self.sensor is not None:
            raise RuntimeError("CameraIO 已经初始化")

        self._resources_active = True

        try:
            self.sensor = Sensor(
                id=CAMERA_ID,
                width=CAMERA_SOURCE_WIDTH,
                height=CAMERA_SOURCE_HEIGHT,
                fps=CAMERA_FPS,
            )

            self.sensor.reset()
            self.sensor.set_hmirror(CAMERA_HMIRROR)
            self.sensor.set_vflip(CAMERA_VFLIP)
            self.sensor.set_framesize(
                width=IMAGE_WIDTH,
                height=IMAGE_HEIGHT,
            )
            self.sensor.set_pixformat(
                self._resolve_pixel_format(CAMERA_PIXEL_FORMAT)
            )

            self._initialize_display()
            MediaManager.init()
            self.sensor.run()
            self._running = True

        except Exception:
            self.deinitialize()
            raise

        return self

    def snapshot(self):
        """获取一帧图像。"""

        if not self._running:
            raise RuntimeError("CameraIO 尚未初始化")

        return self.sensor.snapshot()

    def show_image(self, image):
        """按照当前显示配置输出图像。"""

        if not self._running:
            raise RuntimeError("CameraIO 尚未初始化")

        if self.display_x == 0 and self.display_y == 0:
            Display.show_image(image)
        else:
            Display.show_image(
                image,
                x=self.display_x,
                y=self.display_y,
            )

    def deinitialize(self):
        """按安全顺序释放摄像头、显示和媒体资源。"""

        if not self._resources_active and self.sensor is None:
            return

        if self.sensor is not None:
            try:
                self.sensor.stop()
            except Exception:
                pass

        self._running = False

        try:
            Display.deinit()
        except Exception:
            pass

        try:
            os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
        except Exception:
            pass

        self._sleep_ms(100)

        try:
            MediaManager.deinit()
        except Exception:
            pass

        self.sensor = None
        self._resources_active = False
        gc.collect()

    def _initialize_display(self):
        if self.display_mode == DISPLAY_MODE_ST7701:
            Display.init(
                Display.ST7701,
                width=self.display_width,
                height=self.display_height,
                fps=self.display_fps,
                to_ide=self.to_ide,
            )
            return

        if self.display_mode == DISPLAY_MODE_VIRT:
            if self.quality is None:
                Display.init(
                    Display.VIRT,
                    width=self.display_width,
                    height=self.display_height,
                    fps=self.display_fps,
                    to_ide=self.to_ide,
                )
            else:
                Display.init(
                    Display.VIRT,
                    width=self.display_width,
                    height=self.display_height,
                    fps=self.display_fps,
                    to_ide=self.to_ide,
                    quality=self.quality,
                )
            return

        raise ValueError(
            "不支持的显示模式: {}".format(self.display_mode)
        )

    def _configure_display(self, display_target):
        """根据显示目标加载板载屏幕或 CanMV IDE 配置。"""

        if display_target == DISPLAY_TARGET_BOARD:
            self.display_mode = TANGLE_DISPLAY_MODE
            self.display_width = TANGLE_DISPLAY_WIDTH
            self.display_height = TANGLE_DISPLAY_HEIGHT
            self.display_fps = TANGLE_DISPLAY_FPS
            self.to_ide = TANGLE_DISPLAY_TO_IDE
            self.display_x = TANGLE_DISPLAY_X
            self.display_y = TANGLE_DISPLAY_Y
            self.quality = None
            return

        if display_target == DISPLAY_TARGET_IDE:
            self.display_mode = NUM_DISPLAY_MODE
            self.display_width = NUM_DISPLAY_WIDTH
            self.display_height = NUM_DISPLAY_HEIGHT
            self.display_fps = NUM_DISPLAY_FPS
            self.to_ide = NUM_DISPLAY_TO_IDE
            self.display_x = NUM_DISPLAY_X
            self.display_y = NUM_DISPLAY_Y
            self.quality = NUM_DISPLAY_QUALITY
            return

        raise ValueError(
            "不支持的显示目标: {}，可用值为 board 或 ide".format(
                display_target
            )
        )

    @staticmethod
    def _resolve_pixel_format(pixel_format):
        if pixel_format == "RGB888":
            return Sensor.RGB888

        if pixel_format == "RGB565":
            return Sensor.RGB565

        raise ValueError(
            "不支持的摄像头像素格式: {}".format(pixel_format)
        )

    @staticmethod
    def _sleep_ms(milliseconds):
        try:
            time.sleep_ms(milliseconds)
        except AttributeError:
            time.sleep(milliseconds / 1000.0)
