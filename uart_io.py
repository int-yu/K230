"""K230 UART 生命周期、原始读写和目标偏差协议。

导入本模块不会占用 GPIO 或初始化 UART。硬件只在 initialize() 中加载，
因此其他模块可以在电脑上导入和测试。
"""

from config import (
    UART_BAUDRATE,
    UART_ID,
    UART_PACKET_PREFIX,
    UART_RX_PIN,
    UART_TX_PIN,
)


class TrackingUART:
    """可复用的 K230 UART1/UART2 封装。

    send_target() 使用 ASCII 协议：

        T,frame,valid,x,y\n

    valid 为 0 时，x 和 y 会强制发送 0。接收端应先判断 valid，
    只有 valid 为 1 时才使用相对坐标。
    """

    def __init__(
        self,
        uart_id=UART_ID,
        tx_pin=UART_TX_PIN,
        rx_pin=UART_RX_PIN,
        baudrate=UART_BAUDRATE,
        packet_prefix=UART_PACKET_PREFIX,
    ):
        if uart_id not in (1, 2):
            raise ValueError("当前模块仅支持 UART1 或 UART2")
        if tx_pin < 0 or rx_pin < 0:
            raise ValueError("UART GPIO 编号不能小于 0")
        if tx_pin == rx_pin:
            raise ValueError("UART TX 和 RX 不能使用同一个 GPIO")
        if baudrate <= 0:
            raise ValueError("UART 波特率必须大于 0")
        if (
            not isinstance(packet_prefix, str) or
            not packet_prefix or
            "," in packet_prefix or
            "\n" in packet_prefix
        ):
            raise ValueError("UART 数据包前缀不能为空，且不能包含逗号或换行")

        self.uart_id = int(uart_id)
        self.tx_pin = int(tx_pin)
        self.rx_pin = int(rx_pin)
        self.baudrate = int(baudrate)
        self.packet_prefix = str(packet_prefix)

        self._fpioa = None
        self._uart = None
        self._next_frame_id = 0

    @property
    def is_initialized(self):
        return self._uart is not None

    def initialize(self, machine_module=None):
        """映射 GPIO 并初始化 UART，成功后返回 self。

        machine_module 仅用于电脑端模拟测试；K230 上不需要传入。
        """
        if self.is_initialized:
            raise RuntimeError("TrackingUART 已经初始化")

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
        return self

    def _require_initialized(self):
        if not self.is_initialized:
            raise RuntimeError("TrackingUART 尚未初始化")

    def write(self, data):
        """发送字符串或字节，并返回底层 UART.write() 的结果。"""
        self._require_initialized()
        return self._uart.write(data)

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

    def send_target(
        self,
        valid,
        offset_x,
        offset_y,
        frame_id=None,
    ):
        """发送有效标志和目标相对偏差，返回实际发送的字符串。"""
        self._require_initialized()
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
        self.write(packet)
        return packet

    def reset_frame_id(self, frame_id=0):
        """设置下一次自动发送使用的帧号。"""
        self._next_frame_id = int(frame_id)
        return self

    def deinitialize(self):
        """安全释放 UART；允许重复调用。"""
        if self._uart is not None:
            try:
                self._uart.deinit()
            except Exception:
                pass
        self._uart = None
        self._fpioa = None

    close = deinitialize
