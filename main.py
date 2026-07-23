"""K230 根目录自动运行入口。

上板自动运行通常需要根目录存在 main.py。需要切换题目时，只改这里的
导入和最后一行调用；各检测模块自己的 run_xxx_demo() 仍保留在原文件中，
便于单独测试。
"""

from detectors.tangle import run_rectangle_tracking


run_rectangle_tracking()
