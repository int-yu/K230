"""K230 摄像头、显示和 MediaManager 生命周期管理。"""

import gc
import os
import time

from media.sensor import Sensor
from media.display import Display
from media.media import MediaManager

import sys

# CanMV 按绝对路径启动脚本时不会把脚本所在目录加入 sys.path，
# 会导致 import config 失败。这里补上，重复导入不会重复追加。
if "/sdcard/K230" not in sys.path:
    sys.path.append("/sdcard/K230")

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
    WIFI_RTSP_CONNECT_TIMEOUT_S,
    WIFI_RTSP_ENABLED,
    WIFI_RTSP_EXCLUSIVE_DISPLAY,
    WIFI_RTSP_REQUIRED,
)


DISPLAY_TARGET_BOARD = "board"
DISPLAY_TARGET_IDE = "ide"


class CameraIO:
    """统一管理摄像头采集、显示输出和媒体资源。"""

    def __init__(
        self,
        display_target=DISPLAY_TARGET_BOARD,
        enable_rtsp=None,
        rtsp_required=None,
        rtsp_service_factory=None,
    ):
        self.display_target = display_target
        self.enable_rtsp = (
            WIFI_RTSP_ENABLED if enable_rtsp is None else bool(enable_rtsp)
        )
        self.rtsp_required = (
            WIFI_RTSP_REQUIRED if rtsp_required is None
            else bool(rtsp_required)
        )
        self._configure_display(display_target)
        self._rtsp_exclusive_display = False
        self._apply_rtsp_display_policy()
        self._rtsp_service_factory = rtsp_service_factory
        self._rtsp_service = None
        self._rtsp_error = None
        self.sensor = None
        self._running = False
        self._resources_active = False
        self._display_initialized = False
        self._media_initialized = False

    def initialize(self):
        """初始化摄像头、显示和媒体缓冲区。"""

        if self._resources_active or self.sensor is not None:
            raise RuntimeError("CameraIO 已经初始化")

        self._resources_active = True

        try:
            self._prepare_rtsp()
            try:
                self._initialize_camera_resources()
            except Exception as error:
                if (
                    self._rtsp_service is None or
                    self.rtsp_required or
                    not self._rtsp_exclusive_display
                ):
                    raise
                self._rtsp_error = (
                    "Wi-Fi RTSP display path startup failed: {}".format(error)
                )
                self._stop_rtsp_service()
                self._fallback_to_normal_display()
            self._start_rtsp()
        except Exception:
            try:
                self.deinitialize()
            except Exception as cleanup_error:
                print("CameraIO cleanup incomplete: {}".format(cleanup_error))
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

    @property
    def rtsp_active(self):
        return bool(
            self._rtsp_service is not None and
            self._rtsp_service.active
        )

    @property
    def rtsp_url(self):
        if self._rtsp_service is None:
            return None
        return self._rtsp_service.rtsp_url

    @property
    def rtsp_error(self):
        if self._rtsp_service is not None:
            worker_error = getattr(
                self._rtsp_service,
                "worker_error",
                None,
            )
            if worker_error:
                return worker_error
            service_error = getattr(self._rtsp_service, "last_error", None)
            if service_error:
                return service_error
        return self._rtsp_error

    def _prepare_rtsp(self):
        if not self.enable_rtsp:
            return
        try:
            if self._rtsp_service_factory is None:
                from wifi_rtsp import create_default_wifi_rtsp_service
                service = create_default_wifi_rtsp_service(
                    WIFI_RTSP_CONNECT_TIMEOUT_S
                )
            else:
                service = self._rtsp_service_factory()
            self._rtsp_service = service
            prepare = getattr(service, "prepare", None)
            if prepare is not None:
                prepare()
            if self._rtsp_exclusive_display:
                print(
                    "Wi-Fi RTSP mode: IDE Preview disabled; "
                    "web stream keeps final annotations"
                )
        except Exception as error:
            self._rtsp_error = self._rtsp_failure_message(error)
            self._stop_rtsp_service()
            if self.rtsp_required:
                raise
            self._restore_normal_display_policy()
            print("Wi-Fi RTSP unavailable; continuing: {}".format(
                self._rtsp_error
            ))

    def _start_rtsp(self):
        if self._rtsp_service is None:
            return
        try:
            start_stream = getattr(self._rtsp_service, "start_stream", None)
            if start_stream is None:
                self._rtsp_service.initialize(
                    self.display_width,
                    self.display_height,
                )
            else:
                start_stream(self.display_width, self.display_height)
            self._rtsp_error = None
            print("Wi-Fi RTSP started: {}".format(
                self._rtsp_service.rtsp_url
            ))
        except Exception as error:
            self._rtsp_error = self._rtsp_failure_message(error)
            self._stop_rtsp_service()
            if self.rtsp_required:
                raise

            # RTSP mode suppresses IDE Preview to avoid two consumers sharing
            # the same Display writeback channel. Rebuild the original media
            # path if stream startup fails, so optional RTSP remains fail-open.
            self._fallback_to_normal_display()

    def deinitialize(self):
        """按安全顺序释放摄像头、显示和媒体资源。"""

        if (
            not self._resources_active and
            self.sensor is None and
            self._rtsp_service is None
        ):
            return True

        if self._rtsp_service is not None:
            self._stop_rtsp_service()

        self._release_camera_resources()
        self._resources_active = False
        gc.collect()
        return True

    def _initialize_camera_resources(self):
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
        self._display_initialized = True
        MediaManager.init()
        self._media_initialized = True
        self.sensor.run()
        self._running = True

    def _release_camera_resources(self):
        if self.sensor is not None:
            try:
                self.sensor.stop()
                self.sensor = None
                self._running = False
            except Exception as error:
                raise RuntimeError(
                    "camera media cleanup failed; refusing unsafe restart: "
                    "Sensor.stop: {}".format(error)
                )

        display_released = False
        if self._display_initialized:
            try:
                Display.deinit()
                self._display_initialized = False
                display_released = True
            except Exception as error:
                raise RuntimeError(
                    "camera media cleanup failed; refusing unsafe restart: "
                    "Display.deinit: {}".format(error)
                )

        if display_released:
            try:
                os.exitpoint(os.EXITPOINT_ENABLE_SLEEP)
            except Exception:
                pass

            self._sleep_ms(100)

        if self._media_initialized:
            try:
                MediaManager.deinit()
                self._media_initialized = False
            except Exception as error:
                raise RuntimeError(
                    "camera media cleanup failed; refusing unsafe restart: "
                    "MediaManager.deinit: {}".format(error)
                )
        self.sensor = None
        self._running = False
        return True

    def _stop_rtsp_service(self):
        if self._rtsp_service is None:
            return True
        stopped = self._rtsp_service.deinitialize()
        if stopped is False:
            error = getattr(self._rtsp_service, "last_error", None)
            raise RuntimeError(
                error or (
                    "RTSP worker is still running; media resources were "
                    "not released. Power-cycle the K230 before running again."
                )
            )
        self._rtsp_service = None
        return True

    def _rtsp_failure_message(self, error):
        if self._rtsp_service is not None:
            service_error = getattr(
                self._rtsp_service,
                "last_error",
                None,
            )
            if service_error:
                return service_error
        return str(error)

    def _apply_rtsp_display_policy(self):
        if (
            self.enable_rtsp and
            WIFI_RTSP_EXCLUSIVE_DISPLAY and
            self.display_target == DISPLAY_TARGET_IDE
        ):
            # Display.VIRT owns the IDE framebuffer path even when callers pass
            # to_ide=False on affected firmware. Use the physical display path
            # so RTSP is the only Display writeback consumer.
            self._configure_display(DISPLAY_TARGET_BOARD)
            self._rtsp_exclusive_display = True

    def _restore_normal_display_policy(self):
        self._configure_display(self.display_target)
        self._rtsp_exclusive_display = False

    def _fallback_to_normal_display(self):
        self._release_camera_resources()
        self._restore_normal_display_policy()
        self._initialize_camera_resources()
        print("Wi-Fi RTSP unavailable; continuing: {}".format(
            self._rtsp_error
        ))

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
