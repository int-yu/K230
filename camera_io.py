"""K230 摄像头、显示和 MediaManager 生命周期管理。"""

import gc
import os
import time

from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager

from config import (
    BOARD_DISPLAY_FPS,
    BOARD_DISPLAY_HEIGHT,
    BOARD_DISPLAY_MODE,
    BOARD_DISPLAY_TO_IDE,
    BOARD_DISPLAY_WIDTH,
    BOARD_DISPLAY_X,
    BOARD_DISPLAY_Y,
    CAMERA_FPS,
    CAMERA_HMIRROR,
    CAMERA_ID,
    CAMERA_PIXEL_FORMAT,
    CAMERA_SOURCE_HEIGHT,
    CAMERA_SOURCE_WIDTH,
    CAMERA_VFLIP,
    DISPLAY_TARGET as DEFAULT_DISPLAY_TARGET,
    DISPLAY_TARGET_BOARD,
    DISPLAY_TARGET_IDE,
    DISPLAY_MODE_ST7701,
    DISPLAY_MODE_VIRT,
    IDE_DISPLAY_FPS,
    IDE_DISPLAY_HEIGHT,
    IDE_DISPLAY_MODE,
    IDE_DISPLAY_QUALITY,
    IDE_DISPLAY_TO_IDE,
    IDE_DISPLAY_WIDTH,
    IDE_DISPLAY_X,
    IDE_DISPLAY_Y,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
)


class CameraIO:
    """统一管理摄像头采集、显示输出和媒体资源。"""

    def __init__(self, display_target=None):
        if display_target is None:
            display_target = DEFAULT_DISPLAY_TARGET
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
            self.display_mode = BOARD_DISPLAY_MODE
            self.display_width = BOARD_DISPLAY_WIDTH
            self.display_height = BOARD_DISPLAY_HEIGHT
            self.display_fps = BOARD_DISPLAY_FPS
            self.to_ide = BOARD_DISPLAY_TO_IDE
            self.display_x = BOARD_DISPLAY_X
            self.display_y = BOARD_DISPLAY_Y
            self.quality = None
            return

        if display_target == DISPLAY_TARGET_IDE:
            self.display_mode = IDE_DISPLAY_MODE
            self.display_width = IDE_DISPLAY_WIDTH
            self.display_height = IDE_DISPLAY_HEIGHT
            self.display_fps = IDE_DISPLAY_FPS
            self.to_ide = IDE_DISPLAY_TO_IDE
            self.display_x = IDE_DISPLAY_X
            self.display_y = IDE_DISPLAY_Y
            self.quality = IDE_DISPLAY_QUALITY
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
