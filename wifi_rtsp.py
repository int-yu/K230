"""Fail-safe lifecycle wrapper for K230 Wi-Fi STA and official WBC RTSP."""

import time


class WifiRtspService:
    """Connect a hotspot and serve the final display via official WBCRtsp."""

    def __init__(self, ssid, password, connect_timeout_s=15,
                 wlan_factory=None, wbc=None, time_module=None):
        if not isinstance(ssid, str) or not ssid.strip():
            raise ValueError("WIFI_SSID cannot be empty")
        if not isinstance(password, str) or not password:
            raise ValueError("WIFI_PASSWORD cannot be empty")
        if connect_timeout_s <= 0:
            raise ValueError("connect_timeout_s must be greater than 0")
        self.ssid = ssid.strip()
        self.password = password
        self.connect_timeout_s = float(connect_timeout_s)
        self._wlan_factory = wlan_factory
        self._wbc = wbc
        self._time = time_module or time
        self._wlan = None
        self._wbc_started = False
        self.active = False
        self.ip_address = None
        self.rtsp_url = None
        self.last_error = None

    def initialize(self, width, height):
        if self.active:
            return self
        self.last_error = None
        try:
            self._wlan = self._create_wlan()
            active_method = getattr(self._wlan, "active", None)
            if active_method is not None:
                active_method(True)
            self._wlan.connect(self.ssid, self.password)
            self._wait_for_connection()
            network_config = self._wlan.ifconfig()
            ip_address = network_config[0]
            if not ip_address or ip_address == "0.0.0.0":
                raise RuntimeError("connected but DHCP did not provide a valid IP")
            wbc = self._get_wbc()
            wbc.configure(int(width), int(height))
            wbc.start()
            self._wbc_started = True
            self.ip_address = str(ip_address)
            self.rtsp_url = self._resolve_rtsp_url()
            self.active = True
            return self
        except Exception as error:
            message = "Wi-Fi RTSP startup failed: {}".format(error)
            self.deinitialize()
            self.last_error = message
            raise RuntimeError(message)

    def deinitialize(self):
        if self._wbc_started:
            try:
                self._wbc.stop()
            except Exception as error:
                if self.last_error is None:
                    self.last_error = "WBC RTSP stop failed: {}".format(error)
            self._wbc_started = False
        if self._wlan is not None:
            try:
                self._wlan.disconnect()
            except Exception:
                pass
            active_method = getattr(self._wlan, "active", None)
            if active_method is not None:
                try:
                    active_method(False)
                except Exception:
                    pass
            self._wlan = None
        self.active = False
        self.ip_address = None
        self.rtsp_url = None

    def _create_wlan(self):
        if self._wlan_factory is not None:
            return self._wlan_factory()
        try:
            import network
        except ImportError:
            raise RuntimeError("firmware is missing the network module with WLAN support")
        if not hasattr(network, "WLAN") or not hasattr(network, "STA_IF"):
            raise RuntimeError("the current network module does not support WLAN STA")
        return network.WLAN(network.STA_IF)

    def _get_wbc(self):
        if self._wbc is not None:
            return self._wbc
        try:
            from libs.WBCRtsp import WBCRtsp
        except ImportError:
            raise RuntimeError("firmware is missing libs.WBCRtsp")
        self._wbc = WBCRtsp
        return self._wbc

    def _wait_for_connection(self):
        start_ms = self._ticks_ms()
        timeout_ms = int(self.connect_timeout_s * 1000)
        while not self._wlan.isconnected():
            if self._ticks_diff(self._ticks_ms(), start_ms) >= timeout_ms:
                raise RuntimeError(
                    "连接热点超时（{} 秒）".format(self.connect_timeout_s)
                )
            self._sleep_ms(100)

    def _resolve_rtsp_url(self):
        server = getattr(self._wbc, "rtspserver", None)
        getter = getattr(server, "get_rtsp_url", None)
        if getter is not None:
            try:
                url = getter()
                if url:
                    return str(url)
            except Exception:
                pass
        return "rtsp://{}:8554/test".format(self.ip_address)

    def _ticks_ms(self):
        ticks_ms = getattr(self._time, "ticks_ms", None)
        if ticks_ms is not None:
            return ticks_ms()
        return int(self._time.time() * 1000)

    def _ticks_diff(self, current, start):
        ticks_diff = getattr(self._time, "ticks_diff", None)
        if ticks_diff is not None:
            return ticks_diff(current, start)
        return current - start

    def _sleep_ms(self, milliseconds):
        sleep_ms = getattr(self._time, "sleep_ms", None)
        if sleep_ms is not None:
            sleep_ms(milliseconds)
        else:
            self._time.sleep(milliseconds / 1000.0)


def load_wifi_credentials(module_name="wifi_secrets"):
    try:
        module = __import__(module_name)
    except ImportError:
        raise RuntimeError(
            "missing {}.py; copy wifi_secrets.example.py and set credentials".format(
                module_name
            )
        )
    ssid = getattr(module, "WIFI_SSID", None)
    password = getattr(module, "WIFI_PASSWORD", None)
    if not isinstance(ssid, str) or not ssid.strip():
        raise RuntimeError("{}.WIFI_SSID cannot be empty".format(module_name))
    if not isinstance(password, str) or not password:
        raise RuntimeError("{}.WIFI_PASSWORD cannot be empty".format(module_name))
    return ssid.strip(), password


def create_default_wifi_rtsp_service(connect_timeout_s=15):
    ssid, password = load_wifi_credentials()
    return WifiRtspService(ssid, password, connect_timeout_s=connect_timeout_s)
