"""Bounded, restart-safe K230 display writeback RTSP pipeline."""

import time


class SafeWbcRtsp:
    """Serve Display writeback as H.264 without unbounded media waits."""

    STATE_NEW = "new"
    STATE_CONFIGURED = "configured"
    STATE_RUNNING = "running"
    STATE_STOPPING = "stopping"
    STATE_STOPPED = "stopped"

    def __init__(
        self,
        display=None,
        encoder_factory=None,
        channel_attr_factory=None,
        stream_data_factory=None,
        rtsp_server_factory=None,
        h264_media_type=None,
        align_up=None,
        thread_module=None,
        os_module=None,
        time_module=None,
        encode_timeout_ms=100,
        stream_timeout_ms=100,
        send_timeout_ms=100,
        writeback_timeout_ms=100,
        frame_interval_ms=50,
        stop_timeout_ms=2000,
        bit_rate_kbps=2048,
        max_empty_frames=100,
    ):
        self._display = display
        self._encoder_factory = encoder_factory
        self._channel_attr_factory = channel_attr_factory
        self._stream_data_factory = stream_data_factory
        self._rtsp_server_factory = rtsp_server_factory
        self._h264_media_type = h264_media_type
        self._align_up = align_up
        self._thread = thread_module
        self._os = os_module
        self._time = time_module or time

        self.encode_timeout_ms = self._positive_timeout(
            "encode_timeout_ms", encode_timeout_ms
        )
        self.stream_timeout_ms = self._positive_timeout(
            "stream_timeout_ms", stream_timeout_ms
        )
        self.send_timeout_ms = self._positive_timeout(
            "send_timeout_ms", send_timeout_ms
        )
        self.writeback_timeout_ms = self._positive_timeout(
            "writeback_timeout_ms", writeback_timeout_ms
        )
        self.frame_interval_ms = self._positive_timeout(
            "frame_interval_ms", frame_interval_ms
        )
        self.stop_timeout_ms = self._positive_timeout(
            "stop_timeout_ms", stop_timeout_ms
        )
        if bit_rate_kbps <= 0:
            raise ValueError("bit_rate_kbps must be greater than 0")
        self.bit_rate_kbps = int(bit_rate_kbps)
        if max_empty_frames <= 0:
            raise ValueError("max_empty_frames must be greater than 0")
        self.max_empty_frames = int(max_empty_frames)

        self.session_name = "test"
        self.port = 8554
        self.width = None
        self.height = None
        self.rtspserver = None
        self.last_error = None
        self.state = self.STATE_NEW

        self._encoder = None
        self._encoder_created = False
        self._encoder_started = False
        self._rtsp_initialized = False
        self._rtsp_started = False
        self._writeback_enabled = False
        self._run_requested = False
        self._thread_alive = False
        self._thread_over = True
        self._consecutive_frame_failures = 0
        self._max_consecutive_frame_failures = 20
        self._consecutive_empty_frames = 0
        self._cleanup_started = False
        self._cleanup_alive = False
        self._cleanup_done = True
        self._cleanup_success = True

    @property
    def active(self):
        return self.state == self.STATE_RUNNING and self._thread_alive

    @property
    def worker_error(self):
        return self.last_error

    def configure(self, width, height):
        """Load board dependencies and prepare the fixed H.264 session."""

        if self.state in (self.STATE_RUNNING, self.STATE_STOPPING):
            raise RuntimeError("cannot configure while RTSP is running or stopping")
        if (
            self._thread_alive or
            self._cleanup_alive or
            self._writeback_enabled or
            self._encoder_created or
            self._encoder_started or
            self._rtsp_initialized or
            self._rtsp_started
        ):
            raise RuntimeError("cannot configure while RTSP resources remain active")
        if int(width) <= 0 or int(height) <= 0:
            raise ValueError("RTSP width and height must be greater than 0")

        self._load_board_dependencies()
        inited = getattr(self._display, "inited", None)
        if inited is not None and not inited():
            raise RuntimeError("start WBC only after Display.init()")

        display_width = self._read_display_dimension("width", int(width))
        display_height = self._read_display_dimension("height", int(height))
        self.width = int(self._align_up(display_width, 16))
        self.height = int(display_height)
        self._encoder = self._encoder_factory()
        self.rtspserver = self._rtsp_server_factory()
        self.last_error = None
        self._cleanup_started = False
        self._cleanup_alive = False
        self._cleanup_done = True
        self._cleanup_success = True
        self.state = self.STATE_CONFIGURED
        return self

    def start(self):
        """Start writeback, encoder, RTSP server and the bounded worker."""

        if self.state == self.STATE_RUNNING:
            return self
        if self.state == self.STATE_STOPPING:
            raise RuntimeError("RTSP worker is still stopping")
        if self.state != self.STATE_CONFIGURED:
            raise RuntimeError("configure RTSP before start")

        self.last_error = None
        self._run_requested = False
        self._thread_alive = False
        self._thread_over = False
        try:
            if not self._display.writeback(True):
                raise RuntimeError("start display writeback failed")
            self._writeback_enabled = True
            self._start_media_pipeline()
            self._run_requested = True
            self._thread_alive = True
            self._consecutive_frame_failures = 0
            self._consecutive_empty_frames = 0
            self._cleanup_started = False
            self._cleanup_alive = False
            self._cleanup_done = False
            self._cleanup_success = False
            self.state = self.STATE_RUNNING
            self._thread.start_new_thread(self._worker, ())
            return self
        except Exception as start_error:
            self._run_requested = False
            self._thread_alive = False
            self._thread_over = True
            self.state = self.STATE_STOPPING
            self._cleanup_started = False
            self._cleanup_alive = False
            self._cleanup_done = False
            self._cleanup_success = False
            cleanup_started = self._begin_cleanup()
            cleanup_success = (
                cleanup_started and
                self._wait_for_cleanup(self.stop_timeout_ms)
            )
            if cleanup_success:
                self.state = self.STATE_STOPPED
                raise
            cleanup_error = self.last_error or "cleanup did not complete"
            raise RuntimeError(
                "RTSP startup failed: {}; cleanup incomplete: {}".format(
                    start_error,
                    cleanup_error,
                )
            )

    def stop(self, timeout_ms=None):
        """Request worker exit and clean up only after it actually exits."""

        if self.state in (self.STATE_NEW, self.STATE_STOPPED):
            return True
        if timeout_ms is None:
            timeout_ms = self.stop_timeout_ms
        timeout_ms = self._positive_timeout("timeout_ms", timeout_ms)

        self._run_requested = False
        deadline = self._ticks_add(self._ticks_ms(), timeout_ms)
        while self._thread_alive:
            if self._ticks_diff(self._ticks_ms(), deadline) >= 0:
                self.state = self.STATE_STOPPING
                self.last_error = (
                    "RTSP worker did not stop within {} ms; "
                    "media resources were left intact"
                ).format(timeout_ms)
                return False
            self._sleep_ms(20)

        if not self._begin_cleanup():
            self.state = self.STATE_STOPPING
            return False
        if not self._wait_for_cleanup(timeout_ms):
            self.state = self.STATE_STOPPING
            return False
        self.state = self.STATE_STOPPED
        return True

    def poll_stop(self):
        """Finish deferred cleanup after a previously timed-out stop."""

        if self._thread_alive:
            return False
        if not self._begin_cleanup():
            self.state = self.STATE_STOPPING
            return False
        if self._cleanup_alive:
            return False
        if not self._cleanup_done or not self._cleanup_success:
            self.state = self.STATE_STOPPING
            return False
        self.state = self.STATE_STOPPED
        return True

    def get_rtsp_url(self):
        if self.rtspserver is None:
            return None
        getter = getattr(self.rtspserver, "rtspserver_getrtspurl", None)
        if getter is None:
            return None
        return getter(self.session_name)

    def _worker(self):
        try:
            while self._run_requested:
                exitpoint = getattr(self._os, "exitpoint", None)
                if exitpoint is not None:
                    exitpoint()
                frame = self._display.writeback_dump(
                    self.writeback_timeout_ms
                )
                if frame:
                    self._consecutive_empty_frames = 0
                    try:
                        if self._send_video_frame(frame):
                            self._consecutive_frame_failures = 0
                        else:
                            self._consecutive_frame_failures += 1
                            if (
                                self._consecutive_frame_failures >=
                                self._max_consecutive_frame_failures
                            ):
                                raise RuntimeError(
                                    "encoder produced no usable stream for {} "
                                    "consecutive frames".format(
                                        self._consecutive_frame_failures
                                    )
                                )
                    finally:
                        self._release_writeback_frame(frame)
                else:
                    self._consecutive_empty_frames += 1
                    if self._consecutive_empty_frames >= self.max_empty_frames:
                        raise RuntimeError(
                            "display writeback returned no frame for {} "
                            "consecutive attempts".format(
                                self._consecutive_empty_frames
                            )
                        )
                self._sleep_ms(self.frame_interval_ms)
        except BaseException as error:
            self.last_error = "RTSP worker failed: {}".format(error)
        finally:
            self._run_requested = False
            self._thread_alive = False
            self._thread_over = True
            if self.state == self.STATE_RUNNING:
                self.state = self.STATE_STOPPING

    def _begin_cleanup(self):
        if self._cleanup_started:
            return True
        self._cleanup_started = True
        self._cleanup_alive = True
        self._cleanup_done = False
        self._cleanup_success = False
        try:
            self._thread.start_new_thread(self._cleanup_worker, ())
            return True
        except Exception as error:
            self._cleanup_alive = False
            self._cleanup_done = True
            self.last_error = "cannot start RTSP cleanup worker: {}".format(error)
            return False

    def _cleanup_worker(self):
        try:
            self._cleanup_success = self._cleanup_resources()
        except BaseException as error:
            self.last_error = "RTSP cleanup worker failed: {}".format(error)
            self._cleanup_success = False
        finally:
            self._cleanup_alive = False
            self._cleanup_done = True

    def _wait_for_cleanup(self, timeout_ms):
        cleanup_deadline = self._ticks_add(self._ticks_ms(), timeout_ms)
        while self._cleanup_alive:
            if self._ticks_diff(self._ticks_ms(), cleanup_deadline) >= 0:
                self.last_error = (
                    "RTSP native cleanup did not finish within {} ms; "
                    "media resources were left intact"
                ).format(timeout_ms)
                return False
            self._sleep_ms(20)
        return self._cleanup_done and self._cleanup_success

    def _send_video_frame(self, frame_info):
        ret = self._encoder.SendFrame(
            frame_info,
            timeout=self.encode_timeout_ms,
        )
        if ret != 0:
            return False

        stream_data = self._stream_data_factory()
        ret = self._encoder.GetStream(
            stream_data,
            timeout=self.stream_timeout_ms,
        )
        if ret != 0:
            return False

        try:
            for pack_index in range(stream_data.pack_cnt):
                send_ret = self.rtspserver.rtspserver_sendvideodata_byphyaddr(
                    self.session_name,
                    stream_data.phy_addr[pack_index],
                    stream_data.data_size[pack_index],
                    self.send_timeout_ms,
                )
                if send_ret not in (None, 0):
                    return False
            return True
        finally:
            self._encoder.ReleaseStream(stream_data)

    def _release_writeback_frame(self, frame):
        release = getattr(self._display, "writeback_release", None)
        if release is not None:
            release(frame)

    def _start_media_pipeline(self):
        self._encoder.SetOutBufs(16, self.width, self.height)
        channel_attr = self._channel_attr_factory(
            self._encoder.PAYLOAD_TYPE_H264,
            self._encoder.H264_PROFILE_MAIN,
            self.width,
            self.height,
            bit_rate=self.bit_rate_kbps,
        )
        self._encoder.Create(channel_attr)
        self._encoder_created = True
        self.rtspserver.rtspserver_init(self.port)
        self._rtsp_initialized = True
        self.rtspserver.rtspserver_createsession(
            self.session_name,
            self._h264_media_type,
            False,
        )
        self.rtspserver.rtspserver_start()
        self._rtsp_started = True
        self._encoder.Start()
        self._encoder_started = True

    def _cleanup_resources(self):
        if self._thread_alive:
            return False

        if self._writeback_enabled:
            try:
                result = self._display.writeback(False)
            except Exception as error:
                self._set_cleanup_error("Display.writeback(False)", error)
                return False
            if result is False:
                self._set_cleanup_error(
                    "Display.writeback(False)",
                    "returned False",
                )
                return False
            self._writeback_enabled = False

        if self._encoder is not None:
            self._drain_encoder()
            if self._encoder_started:
                try:
                    self._encoder.Stop()
                except Exception as error:
                    self._set_cleanup_error("Encoder.Stop", error)
                    return False
                self._encoder_started = False
            if self._encoder_created:
                try:
                    self._encoder.Destroy()
                except Exception as error:
                    self._set_cleanup_error("Encoder.Destroy", error)
                    return False
                self._encoder_created = False

        if self.rtspserver is not None:
            if self._rtsp_started:
                try:
                    self.rtspserver.rtspserver_stop()
                except Exception as error:
                    self._set_cleanup_error("RTSP stop", error)
                    return False
                self._rtsp_started = False
            if self._rtsp_initialized:
                try:
                    self.rtspserver.rtspserver_deinit()
                except Exception as error:
                    self._set_cleanup_error("RTSP deinit", error)
                    return False
                self._rtsp_initialized = False
        return True

    def _set_cleanup_error(self, operation, error):
        self.last_error = "{} failed: {}".format(operation, error)

    def _drain_encoder(self):
        if self._encoder is None or not self._encoder_created:
            return
        for _ in range(16):
            stream_data = self._stream_data_factory()
            try:
                ret = self._encoder.GetStream(stream_data, timeout=0)
            except Exception:
                return
            if ret != 0:
                return
            try:
                self._encoder.ReleaseStream(stream_data)
            except Exception:
                return

    def _load_board_dependencies(self):
        if (
            self._display is not None and
            self._encoder_factory is not None and
            self._channel_attr_factory is not None and
            self._stream_data_factory is not None and
            self._rtsp_server_factory is not None and
            self._h264_media_type is not None and
            self._align_up is not None and
            self._thread is not None and
            self._os is not None
        ):
            return

        try:
            import _thread
            import multimedia as mm
            import os
            from _media import Display
            from media.vencoder import ChnAttrStr, Encoder, StreamData
            from mpp import ALIGN_UP
        except ImportError as error:
            raise RuntimeError(
                "firmware is missing safe RTSP dependencies: {}".format(error)
            )

        self._display = Display
        self._encoder_factory = Encoder
        self._channel_attr_factory = ChnAttrStr
        self._stream_data_factory = StreamData
        self._rtsp_server_factory = mm.rtsp_server
        self._h264_media_type = mm.multi_media_type.media_h264
        self._align_up = ALIGN_UP
        self._thread = _thread
        self._os = os

    def _read_display_dimension(self, name, fallback):
        getter = getattr(self._display, name, None)
        if getter is None:
            return fallback
        value = getter()
        return fallback if value is None else value

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

    def _ticks_add(self, current, delta):
        ticks_add = getattr(self._time, "ticks_add", None)
        if ticks_add is not None:
            return ticks_add(current, delta)
        return current + delta

    def _sleep_ms(self, milliseconds):
        sleep_ms = getattr(self._time, "sleep_ms", None)
        if sleep_ms is not None:
            sleep_ms(milliseconds)
        else:
            self._time.sleep(milliseconds / 1000.0)

    @staticmethod
    def _positive_timeout(name, value):
        value = int(value)
        if value <= 0:
            raise ValueError("{} must be greater than 0".format(name))
        return value
