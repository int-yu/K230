"""Local-only K230 RTSP web viewer with one shared FFmpeg upstream."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import threading
import time
import urllib.parse
import webbrowser
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import BinaryIO, Iterator


HOST = "127.0.0.1"
DEFAULT_PORT = 8765
DEFAULT_FPS = 20
FIRST_FRAME_TIMEOUT_S = 10.0
IDLE_STOP_GRACE_S = 2.0
BOUNDARY = b"frame"
JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"
MAX_JPEG_BUFFER = 8 * 1024 * 1024
MAX_REQUEST_BODY = 8 * 1024
ROOT = Path(__file__).resolve().parent
INDEX_FILE = ROOT / "index.html"


class StreamError(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


class StreamConflict(StreamError):
    def __init__(self, message: str):
        super().__init__(409, message)


def validate_rtsp_url(value: str) -> str:
    """Validate and return a complete RTSP URL."""

    value = value.strip()
    parsed = urllib.parse.urlsplit(value)
    if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
        raise ValueError("请输入完整的 rtsp:// 地址")
    return value


def sanitize_message(value: str) -> str:
    """Hide possible RTSP credentials before showing FFmpeg errors."""

    value = re.sub(r"(rtsp://)[^/@\s]+@", r"\1***@", value)
    return value.strip()[-4000:]


def find_ffmpeg() -> str:
    path = shutil.which("ffmpeg")
    if path is None:
        raise RuntimeError("未找到 FFmpeg，请先安装 FFmpeg 并加入 PATH")
    return path


def build_ffmpeg_command(
    ffmpeg_path: str,
    rtsp_url: str,
    fps: int = DEFAULT_FPS,
) -> list[str]:
    if fps < 1 or fps > 60:
        raise ValueError("fps 必须在 1 到 60 之间")
    return [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-rtsp_transport",
        "tcp",
        "-rw_timeout",
        "5000000",
        "-fflags",
        "nobuffer",
        "-flags",
        "low_delay",
        "-i",
        rtsp_url,
        "-an",
        "-vf",
        "fps={}".format(fps),
        "-f",
        "image2pipe",
        "-vcodec",
        "mjpeg",
        "-q:v",
        "5",
        "pipe:1",
    ]


def iter_jpeg_frames(
    stream: BinaryIO,
    chunk_size: int = 64 * 1024,
) -> Iterator[bytes]:
    buffer = bytearray()
    read_chunk = getattr(stream, "read1", stream.read)
    while True:
        chunk = read_chunk(chunk_size)
        if not chunk:
            return
        buffer.extend(chunk)

        while True:
            start = buffer.find(JPEG_START)
            if start < 0:
                if len(buffer) > 1:
                    del buffer[:-1]
                break

            end = buffer.find(JPEG_END, start + len(JPEG_START))
            if end < 0:
                if start > 0:
                    del buffer[:start]
                if len(buffer) > MAX_JPEG_BUFFER:
                    del buffer[:-1]
                break

            end += len(JPEG_END)
            frame = bytes(buffer[start:end])
            del buffer[:end]
            yield frame


class SharedRtspStream:
    """Own exactly one FFmpeg process and broadcast its latest JPEG frame."""

    def __init__(
        self,
        ffmpeg_path: str,
        fps: int,
        popen_factory=subprocess.Popen,
        first_frame_timeout_s: float = FIRST_FRAME_TIMEOUT_S,
        idle_stop_grace_s: float = IDLE_STOP_GRACE_S,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.fps = fps
        self.popen_factory = popen_factory
        self.first_frame_timeout_s = float(first_frame_timeout_s)
        self.idle_stop_grace_s = float(idle_stop_grace_s)

        self.condition = threading.Condition()
        self.control_lock = threading.Lock()
        self.state = "idle"
        self.url = None
        self.generation = 0
        self.process = None
        self.latest_frame = None
        self.frame_id = 0
        self.last_error = None
        self.subscribers = 0
        self.stderr_tail = deque(maxlen=64)
        self.idle_stop_timer = None

    def start(self, rtsp_url: str) -> dict:
        rtsp_url = validate_rtsp_url(rtsp_url)
        with self.control_lock:
            with self.condition:
                if (
                    self.url == rtsp_url and
                    self.state in ("starting", "streaming")
                ):
                    self._cancel_idle_timer_locked()
                    self._schedule_idle_stop_locked(
                        self.first_frame_timeout_s + 2.0
                    )
                    return self._status_locked()
                if self.subscribers and self.url != rtsp_url:
                    raise StreamConflict("当前画面仍有浏览器订阅，请先停止")

            self._stop_process_locked("restarting")
            command = build_ffmpeg_command(
                self.ffmpeg_path,
                rtsp_url,
                self.fps,
            )
            try:
                process = self.popen_factory(
                    command,
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    shell=False,
                )
            except OSError as error:
                raise StreamError(500, "无法启动 FFmpeg：{}".format(error))

            with self.condition:
                self.generation += 1
                generation = self.generation
                self.process = process
                self.url = rtsp_url
                self.state = "starting"
                self.latest_frame = None
                self.frame_id = 0
                self.last_error = None
                self.stderr_tail.clear()
                self._cancel_idle_timer_locked()
                self._schedule_idle_stop_locked(
                    self.first_frame_timeout_s + 2.0
                )
                self.condition.notify_all()

            threading.Thread(
                target=self._read_stderr,
                args=(process, generation),
                daemon=True,
            ).start()
            threading.Thread(
                target=self._read_frames,
                args=(process, generation),
                daemon=True,
            ).start()
            threading.Thread(
                target=self._watch_first_frame,
                args=(process, generation),
                daemon=True,
            ).start()
            return self.status()

    def stop(self) -> dict:
        with self.control_lock:
            self._stop_process_locked("stopped")
        return self.status()

    def status(self) -> dict:
        with self.condition:
            return self._status_locked()

    def open_subscription(self, generation: int) -> tuple[int, bytes]:
        deadline = time.monotonic() + self.first_frame_timeout_s + 1.0
        with self.control_lock:
            with self.condition:
                if generation != self.generation:
                    raise StreamError(409, "播放会话已过期，请重新开始")
                if self.process is None or self.state in ("idle", "stopped"):
                    raise StreamError(503, "RTSP 拉流未启动")
                self._cancel_idle_timer_locked()
                self.subscribers += 1

        with self.condition:
            try:
                while True:
                    if generation != self.generation:
                        raise StreamError(409, "播放会话已切换")
                    if self.latest_frame is not None:
                        return self.frame_id, self.latest_frame
                    if self.state == "failed":
                        raise StreamError(502, self.last_error or "RTSP 拉流失败")
                    if self.state in ("idle", "stopped"):
                        raise StreamError(503, "RTSP 拉流未启动")
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise StreamError(504, "等待首帧超时")
                    self.condition.wait(min(remaining, 0.5))
            except Exception:
                self.subscribers -= 1
                self._schedule_idle_stop_locked()
                raise

    def next_frame(
        self,
        generation: int,
        last_frame_id: int,
        timeout_s: float = 15.0,
    ) -> tuple[int, bytes] | None:
        deadline = time.monotonic() + timeout_s
        with self.condition:
            while True:
                if generation != self.generation:
                    raise StreamError(409, "播放会话已切换")
                if self.frame_id > last_frame_id and self.latest_frame is not None:
                    return self.frame_id, self.latest_frame
                if self.state == "failed":
                    raise StreamError(502, self.last_error or "RTSP 拉流失败")
                if self.state in ("idle", "stopped"):
                    return None
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self.condition.wait(min(remaining, 0.5))

    def close_subscription(self) -> None:
        with self.condition:
            if self.subscribers > 0:
                self.subscribers -= 1
            self._schedule_idle_stop_locked()

    def _read_frames(self, process, generation: int) -> None:
        try:
            if process.stdout is None:
                raise RuntimeError("FFmpeg stdout 不可用")
            for frame in iter_jpeg_frames(process.stdout):
                with self.condition:
                    if process is not self.process or generation != self.generation:
                        return
                    self.latest_frame = frame
                    self.frame_id += 1
                    self.state = "streaming"
                    self.condition.notify_all()
        except Exception as error:
            self._mark_failed(process, generation, str(error))
            return

        with self.condition:
            if process is not self.process or generation != self.generation:
                return
            if self.state not in ("idle", "stopped", "failed"):
                error = self._stderr_message_locked()
                self.state = "failed"
                self.last_error = error or "FFmpeg 已退出，未收到更多视频帧"
                self.condition.notify_all()

    def _read_stderr(self, process, generation: int) -> None:
        if process.stderr is None:
            return
        while True:
            chunk = process.stderr.read(1024)
            if not chunk:
                return
            text = chunk.decode("utf-8", errors="replace")
            with self.condition:
                if process is not self.process or generation != self.generation:
                    return
                self.stderr_tail.append(text)

    def _watch_first_frame(self, process, generation: int) -> None:
        deadline = time.monotonic() + self.first_frame_timeout_s
        with self.condition:
            while True:
                if process is not self.process or generation != self.generation:
                    return
                if self.latest_frame is not None or self.state != "starting":
                    return
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self.state = "failed"
                    error = self._stderr_message_locked()
                    self.last_error = error or "等待 K230 首帧超时（10 秒）"
                    self.condition.notify_all()
                    break
                self.condition.wait(min(remaining, 0.25))
        self._terminate_process(process)

    def _mark_failed(self, process, generation: int, message: str) -> None:
        with self.condition:
            if process is not self.process or generation != self.generation:
                return
            self.state = "failed"
            self.last_error = sanitize_message(
                self._stderr_message_locked() or message
            )
            self.condition.notify_all()

    def _stop_process_locked(self, final_state: str) -> None:
        with self.condition:
            process = self.process
            self.process = None
            self.state = final_state
            self.url = None
            self.latest_frame = None
            self.last_error = None
            self._cancel_idle_timer_locked()
            self.condition.notify_all()
        if process is not None:
            self._terminate_process(process)
        with self.condition:
            if self.state == final_state:
                self.state = "idle"
                self.condition.notify_all()

    @staticmethod
    def _terminate_process(process) -> None:
        try:
            running = process.poll() is None
        except OSError:
            running = False
        if running:
            try:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        pass
            except OSError:
                pass
        for stream in (process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except OSError:
                    pass

    def _status_locked(self) -> dict:
        return {
            "state": self.state,
            "generation": self.generation,
            "subscribers": self.subscribers,
            "frame_id": self.frame_id,
            "error": self.last_error,
        }

    def _stderr_message_locked(self) -> str:
        return sanitize_message("".join(self.stderr_tail))

    def _schedule_idle_stop_locked(self, delay_s=None) -> None:
        if self.subscribers != 0 or self.process is None:
            return
        self._cancel_idle_timer_locked()
        if delay_s is None:
            delay_s = self.idle_stop_grace_s
        generation = self.generation
        process = self.process
        timer = threading.Timer(
            delay_s,
            self._stop_if_idle,
            args=(generation, process),
        )
        timer.daemon = True
        self.idle_stop_timer = timer
        timer.start()

    def _cancel_idle_timer_locked(self) -> None:
        if self.idle_stop_timer is not None:
            self.idle_stop_timer.cancel()
            self.idle_stop_timer = None

    def _stop_if_idle(self, generation, process) -> None:
        with self.control_lock:
            with self.condition:
                if (
                    self.subscribers != 0 or
                    generation != self.generation or
                    process is not self.process
                ):
                    return
                self.idle_stop_timer = None
            self._stop_process_locked("stopped")


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, handler, ffmpeg_path: str, fps: int):
        super().__init__(address, handler)
        self.stream = SharedRtspStream(ffmpeg_path, fps)

    def server_close(self) -> None:
        self.stream.stop()
        super().server_close()


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/":
            self._serve_index()
            return
        if parsed.path == "/stream.mjpg":
            self._serve_stream(parsed.query)
            return
        if parsed.path == "/api/status":
            self._send_json(200, self.server.stream.status())
            return
        if parsed.path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        if parsed.path == "/favicon.ico":
            self.send_response(204)
            self.end_headers()
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path == "/api/start":
            try:
                payload = self._read_json()
                status = self.server.stream.start(payload.get("url", ""))
                self._send_json(200, status)
            except ValueError as error:
                self._send_json(400, {"error": str(error)})
            except StreamError as error:
                self._send_json(error.status, {"error": str(error)})
            return
        if parsed.path == "/api/stop":
            self._send_json(200, self.server.stream.stop())
            return
        self.send_error(404, "Not found")

    def _serve_index(self) -> None:
        try:
            content = INDEX_FILE.read_bytes()
        except OSError as error:
            self.send_error(500, "Cannot read index.html: {}".format(error))
            return
        self._send_bytes(200, content, "text/html; charset=utf-8")

    def _serve_stream(self, query: str) -> None:
        parameters = urllib.parse.parse_qs(query)
        try:
            generation = int(parameters.get("generation", ["0"])[0])
            frame_id, frame = self.server.stream.open_subscription(generation)
        except (ValueError, StreamError) as error:
            status = error.status if isinstance(error, StreamError) else 400
            self._send_json(status, {"error": str(error)})
            return

        try:
            self.send_response(200)
            self.send_header(
                "Content-Type",
                "multipart/x-mixed-replace; boundary={}".format(
                    BOUNDARY.decode("ascii")
                ),
            )
            self.send_header("Cache-Control", "no-store, no-cache")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            while frame is not None:
                self._write_frame(frame)
                next_item = self.server.stream.next_frame(
                    generation,
                    frame_id,
                )
                if next_item is None:
                    break
                frame_id, frame = next_item
        except (
            BrokenPipeError,
            ConnectionResetError,
            ConnectionAbortedError,
            StreamError,
        ):
            pass
        finally:
            self.server.stream.close_subscription()

    def _write_frame(self, frame: bytes) -> None:
        self.wfile.write(b"--" + BOUNDARY + b"\r\n")
        self.wfile.write(b"Content-Type: image/jpeg\r\n")
        self.wfile.write(
            "Content-Length: {}\r\n\r\n".format(len(frame)).encode("ascii")
        )
        self.wfile.write(frame)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _read_json(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("无效的请求长度")
        if length < 1 or length > MAX_REQUEST_BODY:
            raise ValueError("请求内容为空或过大")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("请求必须是 UTF-8 JSON")
        if not isinstance(value, dict):
            raise ValueError("请求 JSON 必须是对象")
        return value

    def _send_json(self, status: int, value: dict) -> None:
        content = json.dumps(value, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, content, "application/json; charset=utf-8")

    def _send_bytes(self, status: int, content: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, message: str, *args) -> None:
        safe_path = urllib.parse.urlsplit(self.path).path
        print("[viewer] {} {}".format(self.command, safe_path))


def open_browser_later(url: str) -> None:
    timer = threading.Timer(0.6, lambda: webbrowser.open(url))
    timer.daemon = True
    timer.start()


def main() -> int:
    parser = argparse.ArgumentParser(description="Local K230 RTSP web viewer")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS)
    parser.add_argument("--open", action="store_true", dest="open_browser")
    args = parser.parse_args()

    if args.port < 1 or args.port > 65535:
        parser.error("--port must be between 1 and 65535")
    if args.fps < 1 or args.fps > 60:
        parser.error("--fps must be between 1 and 60")

    ffmpeg_path = find_ffmpeg()
    server = ViewerServer((HOST, args.port), ViewerHandler, ffmpeg_path, args.fps)
    page_url = "http://{}:{}/".format(HOST, args.port)

    print("K230 RTSP 本地网页查看器")
    print("仅本机访问：{}".format(page_url))
    print("FFmpeg：{}".format(ffmpeg_path))
    print("同一时间只会建立一个 K230 RTSP 连接")
    print("按 Ctrl+C 停止")
    if args.open_browser:
        open_browser_later(page_url)

    try:
        server.serve_forever(poll_interval=0.2)
    except KeyboardInterrupt:
        print("\n正在停止查看器")
    finally:
        server.shutdown()
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
