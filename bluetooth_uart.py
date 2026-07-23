"""K230 蓝牙串口接收模块。

使用 UART2 连接蓝牙透明串口模块。

只接收以下6种单字符指令：
1、2、3、4、s、p

支持：
1. 一次接收一个指令；
2. 一次接收多个指令；
3. 自动忽略空格、回车和换行；
4. 自动缓存尚未取出的指令；
5. 不通过蓝牙发送任何数据。
"""

import sys

# CanMV 按绝对路径启动脚本时不会把脚本所在目录加入 sys.path，
# 会导致 import config 失败。这里补上，重复导入不会重复追加。
if "/sdcard/K230" not in sys.path:
    sys.path.append("/sdcard/K230")

from config import (
    BLUETOOTH_UART_BAUDRATE,
    BLUETOOTH_UART_ID,
    BLUETOOTH_UART_RX_PIN,
    BLUETOOTH_UART_SEND_PERIOD_MS,
    BLUETOOTH_UART_TX_PIN,
)
from uart_io import UARTIO


class BluetoothUART(UARTIO):
    """K230蓝牙透明串口接收类。"""

    # 允许接收的指令
    VALID_COMMANDS = "1234sp"

    def __init__(
        self,
        uart_id=BLUETOOTH_UART_ID,
        tx_pin=BLUETOOTH_UART_TX_PIN,
        rx_pin=BLUETOOTH_UART_RX_PIN,
        baudrate=BLUETOOTH_UART_BAUDRATE,
        send_period_ms=BLUETOOTH_UART_SEND_PERIOD_MS,
    ):
        UARTIO.__init__(
            self,
            uart_id,
            tx_pin,
            rx_pin,
            baudrate,
            send_period_ms,
        )
        self.initialize()
        # 存放已经接收到、但还没有被程序取走的指令
        self._command_buffer = []

    def _read_uart_data(self, size=None):
        """从UART读取原始数据。

        返回：
            bytes或str：读取到的数据；
            None：当前没有数据。
        """
        try:
            data = self.read(size)
        except Exception as error:
            print("Bluetooth UART read error:", error)
            return None

        if data is None:
            return None

        # 某些UART实现可能返回空字节
        if isinstance(data, (bytes, bytearray)):
            if len(data) == 0:
                return None

        elif isinstance(data, str):
            if len(data) == 0:
                return None

        return data

    def _decode_data(self, data):
        """把UART数据转换成字符串。

        返回：
            str：转换后的字符串；
            None：转换失败。
        """
        if data is None:
            return None

        # UART直接返回字符串
        if isinstance(data, str):
            return data

        # UART通常返回bytes或bytearray
        try:
            return bytes(data).decode("utf-8")
        except Exception as error:
            print("Bluetooth data decode error:", error)
            return None

    def _parse_commands(self, text):
        """从字符串中提取有效指令。

        例如：
            "12\\r\\n3xsp" -> ["1", "2", "3", "s", "p"]

        参数：
            text：待解析字符串。

        返回：
            有效指令列表。
        """
        commands = []

        if text is None:
            return commands

        for character in text:
            if character in self.VALID_COMMANDS:
                commands.append(character)

            # 忽略蓝牙串口助手可能自动添加的字符
            elif character in ("\r", "\n", " ", "\t"):
                continue

            else:
                print("忽略未知蓝牙数据:", character)

        return commands

    def poll(self, size=None):
        """检查UART并把新指令加入内部缓存。

        参数：
            size：本次最多读取的字节数。
                  None表示读取UART当前可用的数据。

        返回：
            int：本次新接收到的有效指令数量。
        """
        data = self._read_uart_data(size)

        if data is None:
            return 0

        text = self._decode_data(data)

        if text is None:
            return 0

        commands = self._parse_commands(text)

        if commands:
            self._command_buffer.extend(commands)

        return len(commands)

    def receive(self, size=None):
        """读取本次收到的全部有效指令。

        返回：
            list：有效指令列表，例如：
                  ["1", "2", "3", "4", "s", "p"]

            None：当前没有收到有效指令。

        注意：
            调用后会一次取出当前缓存中的全部指令。
        """
        # 先读取UART中的新数据
        self.poll(size)

        if not self._command_buffer:
            return None

        # 复制当前缓存
        commands = self._command_buffer[:]

        # 清空缓存，避免重复处理
        self._command_buffer.clear()

        return commands

    def receive_one(self):
        """每次只返回一个指令。

        返回：
            str：一个有效指令，例如 "1" 或 "s"；
            None：当前没有有效指令。

        如果一次收到多个指令，其余指令会保存在内部，
        等待后续调用。
        """
        # 缓存为空时才读取UART，避免覆盖已有指令
        if not self._command_buffer:
            self.poll()

        if not self._command_buffer:
            return None

        return self._command_buffer.pop(0)

    def available(self):
        """返回当前缓存中尚未处理的指令数量。"""
        self.poll()
        return len(self._command_buffer)

    def clear(self):
        """清空蓝牙接收缓存。"""
        self._command_buffer.clear()

        # 同时读取并丢弃UART中可能残留的数据
        try:
            while True:
                data = self.read()
                if not data:
                    break
        except Exception:
            pass


if __name__ == "__main__":
    # 测试蓝牙串口接收功能
    bluetooth_uart = BluetoothUART()

    print("蓝牙串口接收测试开始，请发送指令：1、2、3、4、s、p")
    print("按 Ctrl+C 停止测试。")

    try:
        while True:
            commands = bluetooth_uart.receive()
            if commands:
                print("收到指令:", commands)
    except KeyboardInterrupt:
        print("蓝牙串口接收测试结束。")
    finally:
        bluetooth_uart.deinitialize()
