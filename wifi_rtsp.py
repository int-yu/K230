"""Fail-safe lifecycle wrapper for K230 Wi-Fi STA and official WBC RTSP."""

import time


class WifiRtspService:
    """Connect a hotspot and serve the final display via bounded WBC RTSP."""

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
        self._prepared = False
        self._active = False
        self.ip_address = None
        self.rtsp_url = None
        self.last_error = None

    def initialize(self, width, height):
        if self.active:
            return self
        self.prepare()
        return self.start_stream(width, height)

    def prepare(self):
        """Connect Wi-Fi and load the safe RTSP implementation before media init."""

        if self._prepared:
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
            self._get_wbc()
            self.ip_address = str(ip_address)
            self.rtsp_url = "rtsp://{}:8554/test".format(self.ip_address)
            self._prepared = True
            return self
        except Exception as error:
            message = "Wi-Fi RTSP startup failed: {}".format(error)
            self.deinitialize()
            self.last_error = message
            raise RuntimeError(message)

    @property
    def active(self):
        if not self._active:
            return False
        if self._wbc_started:
            wbc_active = getattr(self._wbc, "active", None)
            if wbc_active is False:
                worker_error = getattr(self._wbc, "worker_error", None)
                if worker_error:
                    self.last_error = worker_error
                return False
        return True

    @active.setter
    def active(self, value):
        self._active = bool(value)

    @property
    def worker_error(self):
        if self._wbc is None:
            return None
        return getattr(self._wbc, "worker_error", None)

    def start_stream(self, width, height):
        """Start display writeback after Display and MediaManager are ready."""

        if self.active:
            return self
        if not self._prepared:
            raise RuntimeError("prepare Wi-Fi RTSP before starting the stream")
        try:
            wbc = self._get_wbc()
            wbc.configure(int(width), int(height))
            wbc.start()
            self._wbc_started = True
            self.rtsp_url = self._resolve_rtsp_url()
            self.active = True
            return self
        except Exception as error:
            message = "Wi-Fi RTSP startup failed: {}".format(error)
            self.deinitialize()
            self.last_error = message
            raise RuntimeError(message)

    def deinitialize(self):
        """Stop safely; return False rather than destroying live media resources."""

        if self._wbc_started:
            try:
                stopped = self._wbc.stop()
            except Exception as error:
                self.last_error = "WBC RTSP stop failed: {}".format(error)
                self.active = False
                return False
            if stopped is False:
                self.last_error = getattr(
                    self._wbc,
                    "last_error",
                    None,
                ) or "WBC RTSP worker is still stopping"
                self.active = False
                return False
            self._wbc_started = False
        self._disconnect_wlan()
        self._prepared = False
        self.active = False
        self.ip_address = None
        self.rtsp_url = None
        return True

    def poll_deinitialize(self):
        """Retry a deferred safe stop without ever forcing media destruction."""

        if self._wbc_started:
            poll_stop = getattr(self._wbc, "poll_stop", None)
            if poll_stop is None or poll_stop() is False:
                return False
            self._wbc_started = False
        self._disconnect_wlan()
        self._prepared = False
        self.active = False
        self.ip_address = None
        self.rtsp_url = None
        return True

    def _disconnect_wlan(self):
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
            from config import (
                WIFI_RTSP_BIT_RATE_KBPS,
                WIFI_RTSP_ENCODE_TIMEOUT_MS,
                WIFI_RTSP_FRAME_INTERVAL_MS,
                WIFI_RTSP_MAX_EMPTY_FRAMES,
                WIFI_RTSP_SEND_TIMEOUT_MS,
                WIFI_RTSP_STOP_TIMEOUT_MS,
                WIFI_RTSP_STREAM_TIMEOUT_MS,
                WIFI_RTSP_WRITEBACK_TIMEOUT_MS,
            )
            from safe_wbc_rtsp import SafeWbcRtsp
        except ImportError:
            raise RuntimeError("safe_wbc_rtsp.py or its configuration is missing")
        self._wbc = SafeWbcRtsp(
            encode_timeout_ms=WIFI_RTSP_ENCODE_TIMEOUT_MS,
            stream_timeout_ms=WIFI_RTSP_STREAM_TIMEOUT_MS,
            send_timeout_ms=WIFI_RTSP_SEND_TIMEOUT_MS,
            writeback_timeout_ms=WIFI_RTSP_WRITEBACK_TIMEOUT_MS,
            frame_interval_ms=WIFI_RTSP_FRAME_INTERVAL_MS,
            stop_timeout_ms=WIFI_RTSP_STOP_TIMEOUT_MS,
            bit_rate_kbps=WIFI_RTSP_BIT_RATE_KBPS,
            max_empty_frames=WIFI_RTSP_MAX_EMPTY_FRAMES,
        )
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
        getter = getattr(self._wbc, "get_rtsp_url", None)
        if getter is not None:
            try:
                url = getter()
                if url:
                    return str(url)
            except Exception:
                pass
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
