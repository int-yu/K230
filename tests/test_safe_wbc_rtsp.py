"""Desktop tests for the bounded K230 WBC RTSP implementation."""

import os
import sys

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from safe_wbc_rtsp import SafeWbcRtsp


class FakeClock:
    def __init__(self, on_sleep=None):
        self.now_ms = 0
        self.on_sleep = on_sleep

    def ticks_ms(self):
        return self.now_ms

    @staticmethod
    def ticks_diff(current, start):
        return current - start

    def sleep_ms(self, milliseconds):
        self.now_ms += milliseconds
        if self.on_sleep is not None:
            self.on_sleep()


class FakeThread:
    def __init__(self):
        self.targets = []

    def start_new_thread(self, target, args):
        self.targets.append(target)
        return len(self.targets)


class FailWorkerStartThenCleanupThread(FakeThread):
    def start_new_thread(self, target, args):
        self.targets.append(target)
        if len(self.targets) == 1:
            raise RuntimeError("thread start failed")
        target()
        return len(self.targets)


class FakeOS:
    def exitpoint(self):
        return None


class FakeDisplay:
    def __init__(self):
        self.writeback_calls = []
        self.dump_values = []
        self.release_calls = []
        self.fail_disable = False

    @staticmethod
    def inited():
        return True

    @staticmethod
    def width():
        return 640

    @staticmethod
    def height():
        return 480

    def writeback(self, enabled):
        self.writeback_calls.append(bool(enabled))
        if not enabled and self.fail_disable:
            return False
        return True

    def writeback_dump(self, timeout):
        if self.dump_values:
            return self.dump_values.pop(0)
        return None

    def writeback_release(self, frame):
        self.release_calls.append(frame)


class FakeStreamData:
    def __init__(self):
        self.pack_cnt = 1
        self.phy_addr = [123]
        self.data_size = [456]


class FakeEncoder:
    PAYLOAD_TYPE_H264 = 1
    H264_PROFILE_MAIN = 2

    def __init__(self):
        self.events = []
        self.send_calls = []
        self.get_calls = []
        self.release_calls = []

    def SetOutBufs(self, count, width, height):
        self.events.append(("buffers", count, width, height))

    def Create(self, attr):
        self.events.append(("create", attr))

    def Start(self):
        self.events.append("start")

    def SendFrame(self, frame, timeout):
        self.send_calls.append((frame, timeout))
        return 0

    def GetStream(self, stream, timeout):
        self.get_calls.append(timeout)
        if timeout == 0:
            return -1
        return 0

    def ReleaseStream(self, stream):
        self.release_calls.append(stream)

    def Stop(self):
        self.events.append("stop")

    def Destroy(self):
        self.events.append("destroy")


class FakeRtspServer:
    def __init__(self):
        self.events = []
        self.send_calls = []

    def rtspserver_init(self, port):
        self.events.append(("init", port))

    def rtspserver_createsession(self, session, video_type, audio):
        self.events.append(("session", session, video_type, audio))

    def rtspserver_start(self):
        self.events.append("start")

    def rtspserver_getrtspurl(self, session):
        return "rtsp://192.168.1.25:8554/{}".format(session)

    def rtspserver_sendvideodata_byphyaddr(
        self, session, address, size, timeout
    ):
        self.send_calls.append((session, address, size, timeout))
        return 0

    def rtspserver_stop(self):
        self.events.append("stop")

    def rtspserver_deinit(self):
        self.events.append("deinit")


def make_wbc(clock=None, thread_module=None, **kwargs):
    display = FakeDisplay()
    encoder = FakeEncoder()
    rtsp = FakeRtspServer()
    thread = thread_module or FakeThread()
    wbc = SafeWbcRtsp(
        display=display,
        encoder_factory=lambda: encoder,
        channel_attr_factory=lambda *args, **kwargs: (args, kwargs),
        stream_data_factory=FakeStreamData,
        rtsp_server_factory=lambda: rtsp,
        h264_media_type=99,
        align_up=lambda value, alignment: value,
        thread_module=thread,
        os_module=FakeOS(),
        time_module=clock or FakeClock(),
        **kwargs,
    )
    return wbc, display, encoder, rtsp, thread


def test_start_configures_official_session_and_resets_thread_state():
    wbc, display, encoder, rtsp, thread = make_wbc()

    wbc.configure(640, 480).start()

    assert wbc.state == SafeWbcRtsp.STATE_RUNNING
    assert display.writeback_calls == [True]
    assert ("init", 8554) in rtsp.events
    assert ("session", "test", 99, False) in rtsp.events
    assert wbc.get_rtsp_url() == "rtsp://192.168.1.25:8554/test"
    assert thread.targets[0] is not None
    assert encoder.events[0] == ("buffers", 16, 640, 480)


def test_send_frame_uses_only_finite_timeouts_and_releases_stream():
    wbc, _, encoder, rtsp, _ = make_wbc()
    wbc.configure(640, 480).start()

    assert wbc._send_video_frame("frame") is True

    assert encoder.send_calls == [("frame", 100)]
    assert encoder.get_calls == [100]
    assert rtsp.send_calls == [("test", 123, 456, 100)]
    assert len(encoder.release_calls) == 1


def test_worker_releases_writeback_frame_and_always_marks_exit():
    holder = {}

    def stop_after_one_iteration():
        holder["wbc"]._run_requested = False

    clock = FakeClock(on_sleep=stop_after_one_iteration)
    wbc, display, _, _, thread = make_wbc(clock)
    holder["wbc"] = wbc
    display.dump_values.append("frame")
    wbc.configure(640, 480).start()

    thread.targets[0]()

    assert display.release_calls == ["frame"]
    assert wbc._thread_alive is False
    assert wbc._thread_over is True
    assert wbc.active is False
    assert wbc.state == SafeWbcRtsp.STATE_STOPPING


def test_stop_timeout_does_not_destroy_resources_while_worker_is_alive():
    wbc, display, encoder, rtsp, _ = make_wbc()
    wbc.configure(640, 480).start()

    assert wbc.stop(timeout_ms=40) is False

    assert wbc.state == SafeWbcRtsp.STATE_STOPPING
    assert display.writeback_calls == [True]
    assert "stop" not in encoder.events
    assert "destroy" not in encoder.events
    assert "stop" not in rtsp.events


def test_poll_stop_cleans_after_worker_finishes():
    wbc, display, encoder, rtsp, thread = make_wbc()
    wbc.configure(640, 480).start()
    assert wbc.stop(timeout_ms=40) is False
    wbc._thread_alive = False
    wbc._thread_over = True

    assert wbc.poll_stop() is False
    thread.targets[1]()
    assert wbc.poll_stop() is True

    assert display.writeback_calls == [True, False]
    assert "stop" in encoder.events
    assert "destroy" in encoder.events
    assert "stop" in rtsp.events
    assert "deinit" in rtsp.events


def test_cleanup_failure_keeps_resource_flags_and_reports_unsafe_stop():
    wbc, display, encoder, rtsp, _ = make_wbc()
    wbc.configure(640, 480).start()
    wbc._thread_alive = False
    wbc._thread_over = True
    display.fail_disable = True

    assert wbc.stop(timeout_ms=40) is False
    thread = wbc._thread
    thread.targets[1]()
    assert wbc.poll_stop() is False

    assert wbc.state == SafeWbcRtsp.STATE_STOPPING
    assert wbc._writeback_enabled is True
    assert "returned False" in wbc.last_error
    assert "stop" not in encoder.events
    assert "stop" not in rtsp.events


def test_worker_reports_sustained_missing_writeback_frames():
    wbc, _, _, _, thread = make_wbc(max_empty_frames=1)
    wbc.configure(640, 480).start()

    thread.targets[0]()

    assert wbc.active is False
    assert "no frame" in wbc.worker_error


def test_configure_rejects_stopping_state_with_live_resources():
    wbc, _, _, _, _ = make_wbc()
    wbc.configure(640, 480).start()
    wbc.state = SafeWbcRtsp.STATE_STOPPING

    with pytest.raises(RuntimeError, match="running or stopping"):
        wbc.configure(640, 480)


def test_start_failure_uses_bounded_cleanup_worker():
    thread = FailWorkerStartThenCleanupThread()
    wbc, display, encoder, rtsp, _ = make_wbc(thread_module=thread)
    wbc.configure(640, 480)

    with pytest.raises(RuntimeError, match="thread start failed"):
        wbc.start()

    assert wbc.state == SafeWbcRtsp.STATE_STOPPED
    assert display.writeback_calls == [True, False]
    assert "stop" in encoder.events
    assert "destroy" in encoder.events
    assert "stop" in rtsp.events
    assert "deinit" in rtsp.events
