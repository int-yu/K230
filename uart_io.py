"""K230 UART 生命周期、二进制数据帧、握手和目标偏差协议。"""

import time

from config import (
    UART_BAUDRATE,
    UART_HANDSHAKE_PERIOD_MS,
    UART_ID,
    UART_RX_PIN,
    UART_SEND_PERIOD_MS,
    UART_TX_PIN,
)


UART_FRAME_MAGIC_0 = 0xAA
UART_FRAME_MAGIC_1 = 0x55
UART_FRAME_VERSION = 0x01
UART_FRAME_MAX_PAYLOAD = 32

UART_MESSAGE_READY = 0x01
UART_MESSAGE_READY_ACK = 0x02
UART_MESSAGE_TARGET = 0x10

UART_TEST_TARGET_VALID = True
UART_TEST_TARGET_X = 123
UART_TEST_TARGET_Y = -45


def _ticks_ms():
    try:
        return time.ticks_ms()
    except AttributeError:
        return int(time.time() * 1000)


def _ticks_diff(new_value, old_value):
    try:
        return time.ticks_diff(new_value, old_value)
    except AttributeError:
        return new_value - old_value


def _crc8(data):
    """计算 CRC-8/ATM，生成多项式为 0x07，初始值为 0。"""
    crc = 0
    for value in data:
        crc ^= int(value)
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ 0x07) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _encode_int16(value):
    value = int(value)
    if value < -32768 or value > 32767:
        raise ValueError("目标偏差必须在 int16 范围内")
    unsigned_value = value & 0xFFFF
    return bytes((unsigned_value & 0xFF, (unsigned_value >> 8) & 0xFF))


class UARTIO:
    """可复用的 UART1/UART2 生命周期与原始读写封装。

    send_period_ms 是两次周期发送之间的最小时间。write() 始终立即发送，
    write_periodic() 才会应用该周期。
    """

    def __init__(
        self,
        uart_id,
        tx_pin,
        rx_pin,
        baudrate,
        send_period_ms=0,
    ):
        if uart_id not in (1, 2):
            raise ValueError("当前模块仅支持 UART1 或 UART2")
        if tx_pin < 0 or rx_pin < 0:
            raise ValueError("UART GPIO 编号不能小于 0")
        if tx_pin == rx_pin:
            raise ValueError("UART TX 和 RX 不能使用同一个 GPIO")
        if baudrate <= 0:
            raise ValueError("UART 波特率必须大于 0")
        if send_period_ms < 0:
            raise ValueError("UART 发送周期不能小于 0")

        self.uart_id = int(uart_id)
        self.tx_pin = int(tx_pin)
        self.rx_pin = int(rx_pin)
        self.baudrate = int(baudrate)
        self.send_period_ms = int(send_period_ms)

        self._fpioa = None
        self._uart = None
        self._last_send_ms = None

    @property
    def is_initialized(self):
        return self._uart is not None

    @property
    def last_send_ms(self):
        return self._last_send_ms

    def initialize(self, machine_module=None):
        """映射 GPIO 并初始化 UART，成功后返回 self。

        machine_module 只用于电脑端模拟测试；K230 上不需要传入。
        """
        if self.is_initialized:
            raise RuntimeError("UARTIO 已经初始化")

        if machine_module is None:
            import machine as machine_module

        fpioa_class = machine_module.FPIOA
        uart_class = machine_module.UART
        if self.uart_id == 1:
            uart_channel = uart_class.UART1
            tx_function = fpioa_class.UART1_TXD
            rx_function = fpioa_class.UART1_RXD
        else:
            uart_channel = uart_class.UART2
            tx_function = fpioa_class.UART2_TXD
            rx_function = fpioa_class.UART2_RXD

        self._fpioa = fpioa_class()
        try:
            self._fpioa.set_function(self.tx_pin, tx_function)
            self._fpioa.set_function(self.rx_pin, rx_function)
            self._uart = uart_class(
                uart_channel,
                baudrate=self.baudrate,
                bits=uart_class.EIGHTBITS,
                parity=uart_class.PARITY_NONE,
                stop=uart_class.STOPBITS_ONE,
            )
        except Exception:
            self._uart = None
            self._fpioa = None
            raise

        self._last_send_ms = None
        return self

    def _require_initialized(self):
        if not self.is_initialized:
            raise RuntimeError("UARTIO 尚未初始化")

    def set_send_period(self, send_period_ms):
        """修改周期发送的最小间隔，并重新开始计时。"""
        if send_period_ms < 0:
            raise ValueError("UART 发送周期不能小于 0")
        self.send_period_ms = int(send_period_ms)
        self._last_send_ms = None
        return self

    def ready_to_send(self, now_ms=None):
        """当前是否满足周期发送条件。第一次调用始终允许发送。"""
        if self.send_period_ms == 0 or self._last_send_ms is None:
            return True
        if now_ms is None:
            now_ms = _ticks_ms()
        return _ticks_diff(now_ms, self._last_send_ms) >= self.send_period_ms

    def write(self, data):
        """立即发送字符串或字节，不受发送周期限制。"""
        self._require_initialized()
        return self._uart.write(data)

    def write_periodic(self, data, force=False, now_ms=None):
        """满足周期时发送，已发送返回 True，未到周期返回 False。"""
        self._require_initialized()
        if now_ms is None:
            now_ms = _ticks_ms()
        if not force and not self.ready_to_send(now_ms):
            return False
        self.write(data)
        self._last_send_ms = now_ms
        return True

    def any(self):
        """返回接收缓冲区中可读取的数据量。"""
        self._require_initialized()
        return self._uart.any()

    def read(self, size=None):
        """读取接收数据；size 省略时读取当前可用数据。"""
        self._require_initialized()
        if size is None:
            return self._uart.read()
        return self._uart.read(size)

    def readline(self):
        """读取一行接收数据。"""
        self._require_initialized()
        return self._uart.readline()

    def deinitialize(self):
        """安全释放 UART；允许重复调用。"""
        if self._uart is not None:
            try:
                self._uart.deinit()
            except Exception:
                pass
        self._uart = None
        self._fpioa = None
        self._last_send_ms = None

    close = deinitialize


class TrackingUART(UARTIO):
    """使用统一二进制帧完成双向握手并发送目标偏差。"""

    def __init__(
        self,
        uart_id=UART_ID,
        tx_pin=UART_TX_PIN,
        rx_pin=UART_RX_PIN,
        baudrate=UART_BAUDRATE,
        send_period_ms=UART_SEND_PERIOD_MS,
        handshake_period_ms=UART_HANDSHAKE_PERIOD_MS,
    ):
        if handshake_period_ms <= 0:
            raise ValueError("握手重发周期必须大于 0")

        UARTIO.__init__(
            self,
            uart_id,
            tx_pin,
            rx_pin,
            baudrate,
            send_period_ms,
        )
        self.handshake_period_ms = int(handshake_period_ms)
        self._next_sequence = 0
        self._ready_sequence = self._allocate_sequence()
        self._last_ready_ms = None
        self._peer_ready_received = False
        self._ready_ack_received = False
        self._rx_buffer = bytearray()

    @property
    def handshake_complete(self):
        return self._peer_ready_received and self._ready_ack_received

    def _allocate_sequence(self):
        sequence = self._next_sequence
        self._next_sequence = (self._next_sequence + 1) & 0xFF
        return sequence

    def _build_frame(self, message_type, payload=b"", sequence=None):
        payload = bytes(payload)
        if len(payload) > UART_FRAME_MAX_PAYLOAD:
            raise ValueError("UART 数据区超过最大长度")
        if sequence is None:
            sequence = self._allocate_sequence()
        sequence = int(sequence) & 0xFF

        body = bytes((
            UART_FRAME_VERSION,
            int(message_type) & 0xFF,
            sequence,
            len(payload),
        )) + payload
        return bytes((UART_FRAME_MAGIC_0, UART_FRAME_MAGIC_1)) + \
            body + bytes((_crc8(body),))

    def send_frame(self, message_type, payload=b"", sequence=None):
        """立即发送一帧，返回实际发送的 bytes。"""
        self._require_initialized()
        frame = self._build_frame(message_type, payload, sequence)
        self.write(frame)
        return frame

    @staticmethod
    def _find_magic(buffer):
        for index in range(max(0, len(buffer) - 1)):
            if (
                buffer[index] == UART_FRAME_MAGIC_0 and
                buffer[index + 1] == UART_FRAME_MAGIC_1
            ):
                return index
        return -1

    def _extract_frames(self):
        frames = []
        minimum_size = 7

        while len(self._rx_buffer) >= 2:
            magic_index = self._find_magic(self._rx_buffer)
            if magic_index < 0:
                if self._rx_buffer[-1] == UART_FRAME_MAGIC_0:
                    del self._rx_buffer[:-1]
                else:
                    self._rx_buffer = bytearray()
                break
            if magic_index > 0:
                del self._rx_buffer[:magic_index]
            if len(self._rx_buffer) < minimum_size:
                break

            payload_length = self._rx_buffer[5]
            if payload_length > UART_FRAME_MAX_PAYLOAD:
                del self._rx_buffer[0]
                continue

            frame_size = minimum_size + payload_length
            if len(self._rx_buffer) < frame_size:
                break

            body_end = 6 + payload_length
            body = bytes(self._rx_buffer[2:body_end])
            received_crc = self._rx_buffer[body_end]
            if (
                self._rx_buffer[2] != UART_FRAME_VERSION or
                _crc8(body) != received_crc
            ):
                del self._rx_buffer[0]
                continue

            frames.append((
                self._rx_buffer[3],
                self._rx_buffer[4],
                bytes(self._rx_buffer[6:body_end]),
            ))
            del self._rx_buffer[:frame_size]

        return frames

    def _handle_frame(self, message_type, sequence, payload):
        if message_type == UART_MESSAGE_READY and len(payload) == 0:
            self._peer_ready_received = True
            self.send_frame(
                UART_MESSAGE_READY_ACK,
                bytes((sequence,)),
            )
        elif (
            message_type == UART_MESSAGE_READY_ACK and
            len(payload) == 1 and
            payload[0] == self._ready_sequence
        ):
            self._ready_ack_received = True

    def poll(self):
        """读取并解析所有可用数据，返回本次收到的有效帧列表。"""
        self._require_initialized()
        if self.any() > 0:
            received = self.read()
            if received:
                self._rx_buffer.extend(received)

        frames = self._extract_frames()
        for message_type, sequence, payload in frames:
            self._handle_frame(message_type, sequence, payload)
        return frames

    def update_handshake(self, now_ms=None):
        """处理握手并按周期重发 READY；完成后返回 True。"""
        self._require_initialized()
        if now_ms is None:
            now_ms = _ticks_ms()

        self.poll()
        if (
            not self._ready_ack_received and
            (
                self._last_ready_ms is None or
                _ticks_diff(now_ms, self._last_ready_ms) >=
                self.handshake_period_ms
            )
        ):
            self.send_frame(
                UART_MESSAGE_READY,
                sequence=self._ready_sequence,
            )
            self._last_ready_ms = now_ms
        return self.handshake_complete

    def send_target(
        self,
        valid,
        offset_x,
        offset_y,
        frame_id=None,
        force=False,
        now_ms=None,
    ):
        """周期发送 TARGET 帧；握手未完成或未到周期时返回 None。"""
        self._require_initialized()
        if now_ms is None:
            now_ms = _ticks_ms()
        if not self.handshake_complete:
            self.update_handshake(now_ms)
            return None
        if not force and not self.ready_to_send(now_ms):
            return None

        if frame_id is None:
            frame_id = self._allocate_sequence()
        else:
            frame_id = int(frame_id) & 0xFF

        valid_value = 1 if valid else 0
        if valid_value:
            send_x = int(offset_x)
            send_y = int(offset_y)
        else:
            send_x = 0
            send_y = 0

        payload = bytes((valid_value,)) + \
            _encode_int16(send_x) + _encode_int16(send_y)
        frame = self.send_frame(
            UART_MESSAGE_TARGET,
            payload,
            sequence=frame_id,
        )
        self._last_send_ms = now_ms
        return frame

    def reset_frame_id(self, frame_id=0):
        """设置下一次自动发送使用的帧号。"""
        self._next_sequence = int(frame_id) & 0xFF
        return self

    def deinitialize(self):
        UARTIO.deinitialize(self)
        self._last_ready_ms = None
        self._peer_ready_received = False
        self._ready_ack_received = False
        self._rx_buffer = bytearray()

    close = deinitialize


def run_uart_handshake_test():
    """等待天猛星握手，成功后持续发送固定目标测试帧。"""
    uart = TrackingUART().initialize()
    print("UART test: waiting for MSPM0 handshake...")
    try:
        while not uart.update_handshake():
            time.sleep_ms(10)

        print("UART test: handshake complete")
        print("TARGET valid=1 x={} y={}".format(
            UART_TEST_TARGET_X,
            UART_TEST_TARGET_Y,
        ))
        while True:
            uart.poll()
            uart.send_target(
                UART_TEST_TARGET_VALID,
                UART_TEST_TARGET_X,
                UART_TEST_TARGET_Y,
            )
            time.sleep_ms(10)
    finally:
        uart.deinitialize()


if __name__ == "__main__":
    run_uart_handshake_test()
