"""历史兼容入口：请优先使用 detectors.rectangle_distance。"""

import sys

for _path in ("/sdcard/K230", "/sdcard"):
    if _path not in sys.path:
        sys.path.append(_path)

from detectors.rectangle_distance import *


if __name__ == "__main__":
    run_square_distance_demo()
