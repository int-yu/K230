"""Hardware-independent tests for the local RTSP web viewer."""

import io
import threading
import time

import pytest

from server import (
    SharedRtspStream,
    StreamConflict,
    build_ffmpeg_command,
    iter_jpeg_frames,
    sanitize_message,
    validate_rtsp_url,
)


class BlockingStream:
    def __init__(self, chunks=()):
        self.chunks = list(chunks)
        self.closed = False
        self.condition = threading.Condition()

    def read(self, size=-1):
        with self.condition:
            while not self.chunks and not self.closed:
                self.condition.wait(0.05)
            if self.chunks:
                return self.chunks.pop(0)
            return b""

    read1 = read

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()


class FakeProcess:
    def __init__(self, stdout_chunks=(), stderr_chunks=()):
        self.stdout = BlockingStream(stdout_chunks)
        self.stderr = BlockingStream(stderr_chunks)
        self.returncode = None
        self.terminate_calls = 0
        self.kill_calls = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        self.returncode = 0
        self.stdout.close()
        self.stderr.close()

    def kill(self):
        self.kill_calls += 1
        self.returncode = -9
        self.stdout.close()
        self.stderr.close()

    def wait(self, timeout=None):
        return self.returncode


class FakePopenFactory:
    def __init__(self, processes):
        self.processes = list(processes)
        self.calls = []

    def __call__(self, command, **kwargs):
        self.calls.append((command, kwargs))
        return self.processes.pop(0)


def wait_until(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError("condition was not reached")


def test_validate_rtsp_url_and_sanitize_credentials():
    value = "rtsp://192.168.1.25:8554/test"
    assert validate_rtsp_url(value) == value
    with pytest.raises(ValueError):
        validate_rtsp_url("http://192.168.1.25/video")
    assert "user:secret" not in sanitize_message(
        "failed rtsp://user:secret@192.168.1.25/test"
    )


def test_iter_jpeg_frames_ignores_noise_and_split_reads():
    frame_a = b"\xff\xd8alpha\xff\xd9"
    frame_b = b"\xff\xd8beta\xff\xd9"
    stream = io.BytesIO(b"noise" + frame_a + b"junk" + frame_b)
    assert list(iter_jpeg_frames(stream, chunk_size=3)) == [frame_a, frame_b]


def test_build_ffmpeg_command_uses_tcp_read_timeout_and_no_shell_string():
    command = build_ffmpeg_command(
        "C:/ffmpeg.exe",
        "rtsp://192.168.1.25:8554/test",
        fps=15,
    )
    assert isinstance(command, list)
    assert "tcp" in command
    assert "-rw_timeout" in command
    assert "fps=15" in command
    assert command[-1] == "pipe:1"


def test_duplicate_start_and_two_subscribers_share_one_process():
    frame = b"\xff\xd8frame\xff\xd9"
    process = FakeProcess(stdout_chunks=[frame])
    factory = FakePopenFactory([process])
    stream = SharedRtspStream(
        "C:/ffmpeg.exe",
        20,
        popen_factory=factory,
        first_frame_timeout_s=0.5,
        idle_stop_grace_s=5,
    )

    first = stream.start("rtsp://192.168.1.25:8554/test")
    wait_until(lambda: stream.status()["state"] == "streaming")
    second = stream.start("rtsp://192.168.1.25:8554/test")
    one = stream.open_subscription(first["generation"])
    two = stream.open_subscription(first["generation"])

    assert first["generation"] == second["generation"]
    assert one[1] == frame and two[1] == frame
    assert len(factory.calls) == 1
    assert stream.status()["subscribers"] == 2
    stream.close_subscription()
    stream.close_subscription()
    stream.stop()


def test_different_url_is_rejected_while_subscribed():
    frame = b"\xff\xd8frame\xff\xd9"
    factory = FakePopenFactory([FakeProcess(stdout_chunks=[frame])])
    stream = SharedRtspStream(
        "C:/ffmpeg.exe",
        20,
        popen_factory=factory,
        first_frame_timeout_s=0.5,
        idle_stop_grace_s=5,
    )
    status = stream.start("rtsp://192.168.1.25:8554/test")
    wait_until(lambda: stream.status()["state"] == "streaming")
    stream.open_subscription(status["generation"])

    with pytest.raises(StreamConflict):
        stream.start("rtsp://192.168.1.26:8554/test")

    assert len(factory.calls) == 1
    stream.close_subscription()
    stream.stop()


def test_first_frame_timeout_stops_process_and_exposes_sanitized_error():
    process = FakeProcess(
        stderr_chunks=[b"rtsp://user:secret@192.168.1.25 invalid\n"]
    )
    factory = FakePopenFactory([process])
    stream = SharedRtspStream(
        "C:/ffmpeg.exe",
        20,
        popen_factory=factory,
        first_frame_timeout_s=0.05,
        idle_stop_grace_s=5,
    )

    stream.start("rtsp://192.168.1.25:8554/test")
    wait_until(lambda: stream.status()["state"] == "failed")
    value = stream.status()

    assert process.terminate_calls == 1
    assert "user:secret" not in value["error"]
    assert "invalid" in value["error"]


def test_stale_idle_timer_cannot_stop_new_generation():
    frame = b"\xff\xd8frame\xff\xd9"
    first_process = FakeProcess(stdout_chunks=[frame])
    second_process = FakeProcess(stdout_chunks=[frame])
    factory = FakePopenFactory([first_process, second_process])
    stream = SharedRtspStream(
        "C:/ffmpeg.exe",
        20,
        popen_factory=factory,
        first_frame_timeout_s=0.5,
        idle_stop_grace_s=5,
    )
    first = stream.start("rtsp://192.168.1.25:8554/test")
    wait_until(lambda: stream.status()["state"] == "streaming")
    second = stream.start("rtsp://192.168.1.26:8554/test")
    wait_until(lambda: stream.status()["state"] == "streaming")

    stream._stop_if_idle(first["generation"], first_process)

    assert stream.status()["generation"] == second["generation"]
    assert second_process.terminate_calls == 0
    stream.stop()
