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

        @classmethod
        def init(cls, *args, **kwargs):
            events.append("display_init")

        @classmethod
        def show_image(cls, *args, **kwargs):
            events.append("display_show")

        @classmethod
        def deinit(cls):
            events.append("display_deinit")

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

    def initialize(self, width, height):
        self.events.append(("rtsp_start", width, height))
        self.active = True
        self.rtsp_url = "rtsp://192.168.137.25:8554/test"
        return self

    def deinitialize(self):
        self.events.append("rtsp_stop")
        self.active = False
        self.rtsp_url = None


class FailingStream(WorkingStream):
    def initialize(self, width, height):
        self.events.append(("rtsp_start", width, height))
        self.last_error = "Wi-Fi RTSP startup failed: test failure"
        raise RuntimeError(self.last_error)


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
    assert events.index("media_init") < events.index(("rtsp_start", 640, 480))
    assert events.index("sensor_run") < events.index(("rtsp_start", 640, 480))
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
    assert "sensor_stop" not in events
    camera.deinitialize()
    assert events.count("rtsp_stop") == 1
    assert "sensor_stop" in events


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
