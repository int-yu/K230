"""K230 按需拍照并保存到 TF 卡。

收到 MSPM0 发来的 CAPTURE 帧后，在主循环的下一帧保存当前画面。
照片不通过串口回传，事后从 TF 卡拷贝。
"""

import os

import sys

# CanMV 按绝对路径启动脚本时不会把脚本所在目录加入 sys.path，
# 会导致 import config 失败。这里补上，重复导入不会重复追加。
for _path in ("/sdcard/K230", "/sdcard"):
    if _path not in sys.path:
        sys.path.append(_path)

from config import (
    CAPTURE_FILE_PREFIX,
    CAPTURE_FILE_SUFFIX,
    CAPTURE_JPEG_QUALITY,
    CAPTURE_MAX_PENDING,
    CAPTURE_MESSAGE_ACK,
    CAPTURE_MESSAGE_REQUEST,
    CAPTURE_SAVE_DIR,
    CAPTURE_WARMUP_FRAMES,
)


class CaptureService:
    """按需保存当前画面到 TF 卡。

    只负责计数与写文件，不持有摄像头和串口，生命周期由调用方管理。
    """

    def __init__(
        self,
        save_dir=CAPTURE_SAVE_DIR,
        prefix=CAPTURE_FILE_PREFIX,
        suffix=CAPTURE_FILE_SUFFIX,
        quality=CAPTURE_JPEG_QUALITY,
        max_pending=CAPTURE_MAX_PENDING,
    ):
        self.save_dir = save_dir
        self.prefix = prefix
        self.suffix = suffix
        self.quality = quality
        self.max_pending = max_pending

        self._pending = 0
        self._next_index = 1

        self._ensure_dir()
        self._next_index = self._scan_next_index()

    @property
    def pending(self):
        return self._pending

    @property
    def next_index(self):
        return self._next_index

    def _ensure_dir(self):
        try:
            os.mkdir(self.save_dir)
        except OSError:
            # 目录已存在时忽略；真正不可写会在写文件时报错。
            pass

    def _scan_next_index(self):
        """扫描已有文件，返回下一个可用编号，避免覆盖历史素材。"""
        largest = 0
        try:
            names = os.listdir(self.save_dir)
        except OSError:
            return 1

        head = self.prefix + "_"
        for name in names:
            if not name.startswith(head) or not name.endswith(self.suffix):
                continue
            digits = name[len(head):len(name) - len(self.suffix)]
            if not digits.isdigit():
                continue
            value = int(digits)
            if value > largest:
                largest = value
        return largest + 1

    def _build_path(self, index):
        return "{}/{}_{:04d}{}".format(
            self.save_dir, self.prefix, index, self.suffix)

    def handle_frames(self, frames):
        """从 poll() 的返回值中挑出 CAPTURE 帧，累加待拍张数。

        返回本次实际新增的张数（受 max_pending 夹紧后的结果）。
        非 CAPTURE 帧会被忽略。
        """
        added = 0
        for message_type, _sequence, payload in frames:
            if message_type != CAPTURE_MESSAGE_REQUEST or len(payload) != 1:
                continue
            count = payload[0]
            if count <= 0:
                continue
            added += count

        if added <= 0:
            return 0

        # 先算剩余容量，只增加实际能容纳的张数，使返回值与 _pending 增量一致。
        capacity = self.max_pending - self._pending
        if added > capacity:
            added = capacity
        self._pending += added
        return added

    def save(self, image):
        """保存一张，返回本张编号；失败返回 None。

        实测该固件的 image.save() 不支持 rgb888 格式（报
        'current format not support save function!'），因此改用
        compressed() 取 JPEG 字节后自行写文件。
        """
        index = self._next_index
        path = self._build_path(index)

        try:
            try:
                data = image.compressed(quality=self.quality)
            except TypeError:
                # 部分固件的 compressed() 不接受 quality 关键字。
                data = image.compressed()
            with open(path, "wb") as handle:
                handle.write(data)
        except Exception:
            # TF 卡未挂载、已写满或路径不可写。删掉可能残留的半截
            # 文件，否则它会占用编号并在卡上留下打不开的坏图。
            try:
                os.remove(path)
            except Exception:
                pass
            # 不推进编号，让下一张重用同一个编号。
            return None

        self._next_index = index + 1
        return index

    def update(self, image):
        """主循环每帧调用。有待拍时保存一张。

        返回 (本次保存张数, 最后一张编号)；没有保存时返回 (0, 0)。
        """
        if self._pending <= 0:
            return (0, 0)

        index = self.save(image)
        if index is None:
            self._pending -= 1
            return (0, 0)

        self._pending -= 1
        return (1, index)


def run_capture_demo(display_target=None, enable_uart=True, warmup_frames=None):
    """独立拍照主程序：等 MSPM0 的 CAPTURE 帧，存图并回 ACK。"""
    import time

    from core.camera_io import CameraIO, DISPLAY_TARGET_IDE
    from core.uart_io import TrackingUART

    if display_target is None:
        display_target = DISPLAY_TARGET_IDE
    if warmup_frames is None:
        warmup_frames = CAPTURE_WARMUP_FRAMES

    service = CaptureService()
    camera = CameraIO(display_target=display_target).initialize()
    tracking_uart = None

    try:
        # sensor 刚启动时自动曝光尚未收敛,前若干帧是全黑的。
        # 实测初始化后立刻 snapshot() 存出来就是黑图,因此先空跑一段。
        # 不能省:K230 刚开机 MSPM0 就发 CAPTURE 时,存下的会是黑图。
        for _ in range(warmup_frames):
            camera.snapshot()

        if enable_uart:
            tracking_uart = TrackingUART().initialize()
            tracking_uart.wait_for_handshake()

        clock = time.clock()
        while True:
            clock.tick()
            image = camera.snapshot()

            if tracking_uart is not None:
                frames = tracking_uart.poll()
                service.handle_frames(frames)
                saved, last_index = service.update(image)
                if saved > 0:
                    tracking_uart.send_frame(
                        CAPTURE_MESSAGE_ACK,
                        bytes((1, last_index & 0xFF,
                               (last_index >> 8) & 0xFF)),
                    )

            frame = image.to_numpy_ref()
            _draw_status(frame, service, clock.fps())
            camera.show_image(image)
    finally:
        if tracking_uart is not None:
            tracking_uart.deinitialize()
        camera.deinitialize()


def _draw_status(frame, service, fps):
    """在画面左上角画待拍张数、下一编号和 FPS。"""
    import cv2

    cv2.putText(
        frame,
        "CAP pend {} next {} fps {:.1f}".format(
            service.pending, service.next_index, fps),
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        1,
    )


if __name__ == "__main__":
    run_capture_demo()
