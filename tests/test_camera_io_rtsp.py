"""CameraIO optional RTSP lifecycle tests with fake K230 media modules."""

import importlib
import os
import sys
import types

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_camera_io(monkeypatch):
    events = []

    class FakeSensor:
        RGB888 = 1
        RGB565 = 2

        def __init__(self, **kwargs):
            events.append("sensor_create")

        def reset(self):
            events.append("sensor_reset")

        def set_hmirror(self, value):
            events.append("sensor_hmirror")

        def set_vflip(self, value):
            events.append("sensor_vflip")

        def set_framesize(self, **kwargs):
            events.append("sensor_framesize")

        def set_pixformat(self, value):
            events.append("sensor_pixformat")

        def run(self):
            events.append("sensor_run")

        def stop(self):
            events.append("sensor_stop")

        def snapshot(self):
            events.append("snapshot")
            return "frame"

    class FakeDisplay:
        ST7701 = 1
        VIRT = 2
        fail_deinit = False
        fail_st7701_init = False

        @classmethod
        def init(cls, *args, **kwargs):
            events.append("display_init")
            events.append(("display_init_args", kwargs))
            events.append(("display_init_call", args, kwargs))
            if args == (cls.ST7701,) and cls.fail_st7701_init:
                raise RuntimeError("st7701 init failed")

        @classmethod
        def show_image(cls, *args, **kwargs):
            events.append("display_show")

        @classmethod
        def deinit(cls):
            events.append("display_deinit")
            if cls.fail_deinit:
                raise RuntimeError("display deinit failed")

    class FakeMediaManager:
        @classmethod
        def init(cls):
            events.append("media_init")

        @classmethod
        def deinit(cls):
            events.append("media_deinit")

    media_package = types.ModuleType("media")
    sensor_module = types.ModuleType("media.sensor")
    display_module = types.ModuleType("media.display")
    media_module = types.ModuleType("media.media")
    sensor_module.Sensor = FakeSensor
    display_module.Display = FakeDisplay
    media_module.MediaManager = FakeMediaManager
    monkeypatch.setitem(sys.modules, "media", media_package)
    monkeypatch.setitem(sys.modules, "media.sensor", sensor_module)
    monkeypatch.setitem(sys.modules, "media.display", display_module)
    monkeypatch.setitem(sys.modules, "media.media", media_module)
    sys.modules.pop("camera_io", None)
    camera_io = importlib.import_module("camera_io")
    camera_io.CameraIO._sleep_ms = staticmethod(
        lambda milliseconds: events.append("sleep")
    )
    return camera_io, events


class WorkingStream:
    def __init__(self, events):
        self.events = events
        self.active = False
        self.rtsp_url = None
        self.last_error = None

    def prepare(self):
        self.events.append("rtsp_prepare")
        return self

    def start_stream(self, width, height):
        self.events.append(("rtsp_start", width, height))
        self.active = True
        self.rtsp_url = "rtsp://192.168.137.25:8554/test"
        return self

    def deinitialize(self):
        self.events.append("rtsp_stop")
        self.active = False
        self.rtsp_url = None
        return True


class FailingStream(WorkingStream):
    def start_stream(self, width, height):
        self.events.append(("rtsp_start", width, height))
        self.last_error = "Wi-Fi RTSP startup failed: test failure"
        raise RuntimeError(self.last_error)


class PrepareFailingStream(WorkingStream):
    def prepare(self):
        self.events.append("rtsp_prepare")
        self.last_error = "Wi-Fi RTSP startup failed: prepare failure"
        raise RuntimeError(self.last_error)


class StuckStoppingStream(WorkingStream):
    def deinitialize(self):
        self.events.append("rtsp_stop_pending")
        self.last_error = "RTSP worker did not stop"
        return False


def test_disabled_rtsp_does_not_construct_service(monkeypatch):
    camera_io, events = load_camera_io(monkeypatch)

    def forbidden_factory():
        raise AssertionError("RTSP factory should not run")

    camera = camera_io.CameraIO(
        display_target=camera_io.DISPLAY_TARGET_IDE,
        enable_rtsp=False,
        rtsp_service_factory=forbidden_factory,
    ).initialize()
    assert camera.snapshot() == "frame"
    assert camera.rtsp_active is False
    display_args = [item[1] for item in events if isinstance(item, tuple)
                    and item[0] == "display_init_args"]
    assert display_args == [{
        "width": 640,
        "height": 480,
        "fps": 30,
        "to_ide": True,
        "quality": 80,
    }]
    camera.deinitialize()
    assert "sensor_stop" in events


def test_successful_rtsp_starts_after_media_and_stops_before_sensor(monkeypatch):
    camera_io, events = load_camera_io(monkeypatch)
    stream = WorkingStream(events)
    camera = camera_io.CameraIO(
        display_target=camera_io.DISPLAY_TARGET_IDE,
        enable_rtsp=True,
        rtsp_service_factory=lambda: stream,
    ).initialize()
    assert camera.rtsp_active is True
    assert camera.rtsp_url == "rtsp://192.168.137.25:8554/test"
    assert events.index("rtsp_prepare") < events.index("display_init")
    assert events.index("media_init") < events.index(("rtsp_start", 800, 480))
    assert events.index("sensor_run") < events.index(("rtsp_start", 800, 480))
    display_args = [item[1] for item in events if isinstance(item, tuple)
                    and item[0] == "display_init_args"]
    assert display_args[0]["to_ide"] is False
    assert "quality" not in display_args[0]
    display_calls = [item for item in events if isinstance(item, tuple)
                     and item[0] == "display_init_call"]
    assert display_calls[0][1] == (camera_io.Display.ST7701,)
    camera.deinitialize()
    assert events.index("rtsp_stop") < events.index("sensor_stop")
    assert events.index("rtsp_stop") < events.index("display_deinit")


def test_fail_open_keeps_camera_running(monkeypatch):
    camera_io, events = load_camera_io(monkeypatch)
    stream = FailingStream(events)
    camera = camera_io.CameraIO(
        display_target=camera_io.DISPLAY_TARGET_IDE,
        enable_rtsp=True,
        rtsp_required=False,
        rtsp_service_factory=lambda: stream,
    ).initialize()
    assert camera.snapshot() == "frame"
    assert camera.rtsp_active is False
    assert "test failure" in camera.rtsp_error
    display_args = [item[1] for item in events if isinstance(item, tuple)
                    and item[0] == "display_init_args"]
    assert [item["to_ide"] for item in display_args] == [False, True]
    assert "sensor_stop" in events
    assert camera.snapshot() == "frame"
    camera.deinitialize()
    assert events.count("rtsp_stop") == 1
    assert events.count("sensor_stop") == 2


def test_prepare_failure_uses_original_ide_display_without_media_restart(
    monkeypatch,
):
    camera_io, events = load_camera_io(monkeypatch)
    stream = PrepareFailingStream(events)
    camera = camera_io.CameraIO(
        display_target=camera_io.DISPLAY_TARGET_IDE,
        enable_rtsp=True,
        rtsp_required=False,
        rtsp_service_factory=lambda: stream,
    ).initialize()

    display_args = [item[1] for item in events if isinstance(item, tuple)
                    and item[0] == "display_init_args"]
    assert [item["to_ide"] for item in display_args] == [True]
    assert events.count("sensor_create") == 1
    assert "prepare failure" in camera.rtsp_error
    camera.deinitialize()


def test_stuck_rtsp_stop_does_not_destroy_live_media(monkeypatch):
    camera_io, events = load_camera_io(monkeypatch)
    stream = StuckStoppingStream(events)
    camera = camera_io.CameraIO(
        display_target=camera_io.DISPLAY_TARGET_IDE,
        enable_rtsp=True,
        rtsp_service_factory=lambda: stream,
    ).initialize()

    with pytest.raises(RuntimeError, match="did not stop"):
        camera.deinitialize()

    assert "rtsp_stop_pending" in events
    assert "sensor_stop" not in events
    assert "display_deinit" not in events
    assert "media_deinit" not in events


def test_fail_open_refuses_restart_when_old_display_did_not_release(
    monkeypatch,
):
    camera_io, events = load_camera_io(monkeypatch)
    camera_io.Display.fail_deinit = True
    stream = FailingStream(events)
    camera = camera_io.CameraIO(
        display_target=camera_io.DISPLAY_TARGET_IDE,
        enable_rtsp=True,
        rtsp_required=False,
        rtsp_service_factory=lambda: stream,
    )

    with pytest.raises(RuntimeError, match="refusing unsafe restart"):
        camera.initialize()

    assert events.count("sensor_create") == 1
    assert events.count("display_init") == 1


def test_rtsp_board_display_init_failure_falls_back_to_ide(monkeypatch):
    camera_io, events = load_camera_io(monkeypatch)
    camera_io.Display.fail_st7701_init = True
    stream = WorkingStream(events)
    camera = camera_io.CameraIO(
        display_target=camera_io.DISPLAY_TARGET_IDE,
        enable_rtsp=True,
        rtsp_required=False,
        rtsp_service_factory=lambda: stream,
    ).initialize()

    display_calls = [item for item in events if isinstance(item, tuple)
                     and item[0] == "display_init_call"]
    assert display_calls[0][1] == (camera_io.Display.ST7701,)
    assert display_calls[1][1] == (camera_io.Display.VIRT,)
    assert display_calls[1][2]["to_ide"] is True
    assert events.count("sensor_create") == 2
    assert events.count("rtsp_stop") == 1
    assert not any(isinstance(item, tuple) and item[0] == "rtsp_start"
                   for item in events)
    assert "st7701 init failed" in camera.rtsp_error
    assert camera.snapshot() == "frame"
    camera.deinitialize()


def test_required_rtsp_failure_releases_camera_and_raises(monkeypatch):
    camera_io, events = load_camera_io(monkeypatch)
    stream = FailingStream(events)
    camera = camera_io.CameraIO(
        display_target=camera_io.DISPLAY_TARGET_IDE,
        enable_rtsp=True,
        rtsp_required=True,
        rtsp_service_factory=lambda: stream,
    )
    with pytest.raises(RuntimeError, match="test failure"):
        camera.initialize()
    assert "rtsp_stop" in events
    assert events.count("rtsp_stop") == 1
    assert "sensor_stop" in events
    assert "display_deinit" in events
    assert "media_deinit" in events
