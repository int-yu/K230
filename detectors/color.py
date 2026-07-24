"""历史兼容入口：请优先使用 detectors.color_spot。"""

import sys

for _path in ("/sdcard/K230", "/sdcard"):
    if _path not in sys.path:
        sys.path.append(_path)

from detectors.color_spot import *


if __name__ == "__main__":
    run_color_tracking()
