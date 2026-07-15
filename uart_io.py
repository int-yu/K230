"""K230 UART 生命周期、限速发送和目标偏差协议。"""

import time

from config import (
    UART_BAUDRATE,
    UART_ID,
    UART_PACKET_PREFIX,
    UART_RX_PIN,
    UART_SEND_PERIOD_MS,
    UART_TX_PIN,
)


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
    """发送有效标志与目标相对偏差的 UART。

    数据格式：T,frame,valid,x,y\n
    valid 为 0 时，x 和 y 强制发送 0；接收端应先判断 valid。
    """

    def __init__(
        self,
        uart_id=UART_ID,
        tx_pin=UART_TX_PIN,
        rx_pin=UART_RX_PIN,
        baudrate=UART_BAUDRATE,
        packet_prefix=UART_PACKET_PREFIX,
        send_period_ms=UART_SEND_PERIOD_MS,
    ):
        if (
            not isinstance(packet_prefix, str) or
            not packet_prefix or
            "," in packet_prefix or
            "\n" in packet_prefix
        ):
            raise ValueError("UART 数据包前缀不能为空，且不能包含逗号或换行")

        UARTIO.__init__(
            self,
            uart_id,
            tx_pin,
            rx_pin,
            baudrate,
            send_period_ms,
        )
        self.packet_prefix = packet_prefix
        self._next_frame_id = 0

    def send_target(
        self,
        valid,
        offset_x,
        offset_y,
        frame_id=None,
        force=False,
        now_ms=None,
    ):
        """周期发送当前目标；未到周期返回 None，否则返回数据包。"""
        self._require_initialized()
        if now_ms is None:
            now_ms = _ticks_ms()
        if not force and not self.ready_to_send(now_ms):
            return None

        if frame_id is None:
            frame_id = self._next_frame_id
            self._next_frame_id += 1
        else:
            frame_id = int(frame_id)

        valid_value = 1 if valid else 0
        if valid_value:
            send_x = int(offset_x)
            send_y = int(offset_y)
        else:
            send_x = 0
            send_y = 0

        packet = "{},{},{},{},{}\n".format(
            self.packet_prefix,
            frame_id,
            valid_value,
            send_x,
            send_y,
        )
        self.write_periodic(packet, force=True, now_ms=now_ms)
        return packet

    def reset_frame_id(self, frame_id=0):
        """设置下一次自动发送使用的帧号。"""
        self._next_frame_id = int(frame_id)
        return self
