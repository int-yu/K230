"""WifiRtspService tests that do not require K230 hardware."""

import importlib
import os
import sys
import types

import pytest


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class FakeClock:
    def __init__(self):
        self.now_ms = 0

    def ticks_ms(self):
        return self.now_ms

    @staticmethod
    def ticks_diff(current, start):
        return current - start

    def sleep_ms(self, milliseconds):
        self.now_ms += milliseconds


class FakeWLAN:
    def __init__(self, connect_after_checks=0):
        self.connect_after_checks = connect_after_checks
        self.check_count = 0
        self.active_calls = []
        self.connect_calls = []
        self.disconnect_calls = 0

    def active(self, state):
        self.active_calls.append(bool(state))

    def connect(self, ssid, password):
        self.connect_calls.append((ssid, password))

    def isconnected(self):
        self.check_count += 1
        return self.check_count > self.connect_after_checks

    def ifconfig(self):
        return ("192.168.137.25", "255.255.255.0",
                "192.168.137.1", "192.168.137.1")

    def disconnect(self):
        self.disconnect_calls += 1


class FakeRtspServer:
    def get_rtsp_url(self):
        return "rtsp://192.168.137.25:8554/test"


class FakeWBC:
    def __init__(self, fail_start=False, fail_stop=False):
        self.fail_start = fail_start
        self.fail_stop = fail_stop
        self.configure_calls = []
        self.start_calls = 0
        self.stop_calls = 0
        self.rtspserver = FakeRtspServer()
        self.active = False
        self.worker_error = None

    def configure(self, width, height):
        self.configure_calls.append((width, height))

    def start(self):
        self.start_calls += 1
        if self.fail_start:
            raise RuntimeError("start failed")
        self.active = True

    def stop(self):
        self.stop_calls += 1
        if self.fail_stop:
            raise RuntimeError("stop failed")
        self.active = False


def test_import_does_not_load_board_network_or_wbc(monkeypatch):
    monkeypatch.delitem(sys.modules, "network", raising=False)
    monkeypatch.delitem(sys.modules, "libs.WBCRtsp", raising=False)
    sys.modules.pop("wifi_rtsp", None)
    importlib.import_module("wifi_rtsp")
    assert "network" not in sys.modules
    assert "libs.WBCRtsp" not in sys.modules


def test_config_defaults_keep_rtsp_disabled():
    import config
    assert config.WIFI_RTSP_ENABLED is False
    assert config.WIFI_RTSP_REQUIRED is False
    assert config.WIFI_RTSP_CONNECT_TIMEOUT_S == 15
    assert config.WIFI_RTSP_EXCLUSIVE_DISPLAY is True
    assert config.WIFI_RTSP_ENCODE_TIMEOUT_MS > 0
    assert config.WIFI_RTSP_STREAM_TIMEOUT_MS > 0
    assert config.WIFI_RTSP_SEND_TIMEOUT_MS > 0
    assert config.WIFI_RTSP_STOP_TIMEOUT_MS > 0
    assert config.WIFI_RTSP_MAX_EMPTY_FRAMES > 0


def test_load_wifi_credentials_reads_ignored_module(monkeypatch):
    module = types.ModuleType("test_wifi_secrets")
    module.WIFI_SSID = "phone-hotspot"
    module.WIFI_PASSWORD = "12345678"
    monkeypatch.setitem(sys.modules, "test_wifi_secrets", module)
    wifi_rtsp = importlib.import_module("wifi_rtsp")
    assert wifi_rtsp.load_wifi_credentials("test_wifi_secrets") == (
        "phone-hotspot", "12345678")


def test_initialize_connects_and_starts_wbc_then_cleans_once():
    wifi_rtsp = importlib.import_module("wifi_rtsp")
    wlan = FakeWLAN()
    wbc = FakeWBC()
    service = wifi_rtsp.WifiRtspService(
        "phone-hotspot", "12345678",
        wlan_factory=lambda: wlan, wbc=wbc, time_module=FakeClock(),
    )
    assert service.initialize(640, 480) is service
    assert service.active is True
    assert service.ip_address == "192.168.137.25"
    assert service.rtsp_url == "rtsp://192.168.137.25:8554/test"
    assert wlan.connect_calls == [("phone-hotspot", "12345678")]
    assert wbc.configure_calls == [(640, 480)]
    assert wbc.start_calls == 1
    service.deinitialize()
    service.deinitialize()
    assert wbc.stop_calls == 1
    assert wlan.disconnect_calls == 1
    assert wlan.active_calls == [True, False]
    assert service.active is False
    assert service.rtsp_url is None


def test_wbc_stop_failure_keeps_dependencies_intact_for_safe_retry():
    wifi_rtsp = importlib.import_module("wifi_rtsp")
    wlan = FakeWLAN()
    wbc = FakeWBC(fail_stop=True)
    service = wifi_rtsp.WifiRtspService(
        "phone-hotspot", "12345678",
        wlan_factory=lambda: wlan, wbc=wbc, time_module=FakeClock(),
    )
    service.initialize(640, 480)

    assert service.deinitialize() is False

    assert wbc.stop_calls == 1
    assert wlan.disconnect_calls == 0
    assert wlan.active_calls == [True]
    assert service.active is False
    assert service.rtsp_url == "rtsp://192.168.137.25:8554/test"


def test_connection_timeout_never_starts_wbc():
    wifi_rtsp = importlib.import_module("wifi_rtsp")
    wlan = FakeWLAN(connect_after_checks=999)
    wbc = FakeWBC()
    service = wifi_rtsp.WifiRtspService(
        "phone-hotspot", "12345678", connect_timeout_s=0.2,
        wlan_factory=lambda: wlan, wbc=wbc, time_module=FakeClock(),
    )
    with pytest.raises(RuntimeError, match="连接热点超时"):
        service.initialize(640, 480)
    assert "连接热点超时" in service.last_error
    assert service.active is False
    assert wbc.start_calls == 0
    assert wbc.stop_calls == 0
    assert wlan.disconnect_calls == 1


def test_wbc_start_failure_does_not_call_hanging_stop():
    wifi_rtsp = importlib.import_module("wifi_rtsp")
    wlan = FakeWLAN()
    wbc = FakeWBC(fail_start=True)
    service = wifi_rtsp.WifiRtspService(
        "phone-hotspot", "12345678",
        wlan_factory=lambda: wlan, wbc=wbc, time_module=FakeClock(),
    )
    with pytest.raises(RuntimeError, match="start failed"):
        service.initialize(640, 480)
    assert service.active is False
    assert wbc.start_calls == 1
    assert wbc.stop_calls == 0
    assert wlan.disconnect_calls == 1


def test_service_active_reflects_safe_worker_failure():
    wifi_rtsp = importlib.import_module("wifi_rtsp")
    wlan = FakeWLAN()
    wbc = FakeWBC()
    service = wifi_rtsp.WifiRtspService(
        "phone-hotspot", "12345678",
        wlan_factory=lambda: wlan, wbc=wbc, time_module=FakeClock(),
    ).initialize(640, 480)
    wbc.worker_error = "RTSP worker failed: encoder timeout"
    wbc.active = False

    assert service.active is False
    assert "encoder timeout" in service.last_error

    wbc.active = True
    service.deinitialize()
