"""K230 红线巡线模块。

调用方式：

    from line import LineTrackDetector

    detector = LineTrackDetector()
    result = detector.process(frame)

模块把画面下部整体缩放一次，再切成若干条水平带。每条带内对红色掩膜
做列投影，取连续列段（run）的中心作为红线在该带的位置，由近及远逐带
传递以避开路口横向分支。红色判据为 RGB 通道差分，与 road.py 的绿色
通道 Otsu 相反：巡线必须把红线和黑色数字、黑色墙线分开。

没有当前帧结果时返回 None，不保留上一帧结果，不做预测或时间平滑。
"""

import cv2

try:
    import ulab.numpy as np
except ImportError:
    import numpy as np

import sys

# CanMV 按绝对路径启动脚本时不会把脚本所在目录加入 sys.path，
# 会导致 import config 失败。这里补上，重复导入不会重复追加。
if "/sdcard/K230" not in sys.path:
    sys.path.append("/sdcard/K230")

from config import (
    LINE_BAND_COUNT,
    LINE_BAND_HEIGHT,
    LINE_CONTINUITY_REF_RATIO,
    LINE_DEMO_GC_INTERVAL,
    LINE_DEMO_PRINT_INTERVAL,
    LINE_DETECT_WIDTH,
    LINE_EDGE_MIN_COLUMN_COUNT,
    LINE_DRAW_AXIS_COLOR,
    LINE_DRAW_BAND_COLOR,
    LINE_DRAW_BAND_LABELS,
    LINE_DRAW_CENTER_COLOR,
    LINE_DRAW_DATA_LINE_HEIGHT,
    LINE_DRAW_DATA_ORIGIN,
    LINE_DRAW_DATA_OVERLAY,
    LINE_DRAW_FONT_SCALE,
    LINE_DRAW_JUNCTION_COLOR,
    LINE_DRAW_LOST_COLOR,
    LINE_DRAW_PATH_COLOR,
    LINE_DRAW_POINT_RADIUS,
    LINE_DRAW_TEXT_COLOR,
    LINE_DRAW_THICKNESS,
    LINE_JUNCTION_CONFIRM_FRAMES,
    LINE_JUNCTION_MASS_RATIO,
    LINE_JUNCTION_MIN_BANDS,
    LINE_JUNCTION_SIDE_MIN_OFFSET_RATIO,
    LINE_JUNCTION_WIDTH_RATIO,
    LINE_MAIN_MIN_COLUMN_COUNT,
    LINE_MIN_VALID_BANDS,
    LINE_RED_MIN_DIFF,
    LINE_RED_MIN_VALUE,
    LINE_ROI_BOTTOM_RATIO,
    LINE_ROI_TOP_RATIO,
    LINE_RUN_MAX_GAP,
    LINE_RUN_MAX_WIDTH_RATIO,
    LINE_RUN_MIN_WIDTH,
)


# junction_flags 的位定义，与 uart_io.send_line() 的 PAYLOAD 一致。
JUNCTION_FLAG_PRESENT = 0x01
JUNCTION_FLAG_LEFT = 0x02
JUNCTION_FLAG_RIGHT = 0x04
JUNCTION_FLAG_LOST = 0x08


def describe_junction_flags(flags):
    """把 junction_flags 展开成可读文本，用于画面和终端。"""
    flags = int(flags)
    if not flags:
        return "-"
    names = []
    if flags & JUNCTION_FLAG_PRESENT:
        names.append("JUNC")
    if flags & JUNCTION_FLAG_LEFT:
        names.append("L")
    if flags & JUNCTION_FLAG_RIGHT:
        names.append("R")
    if flags & JUNCTION_FLAG_LOST:
        names.append("LOST")
    return " ".join(names)


def format_result(result):
    """把一帧结果压成一行文本，画面和终端共用同一份格式。"""
    if result is None:
        return "LOST  valid=0"
    offsets = []
    for index, offset in enumerate(result["offsets"]):
        if result["band_valid"][index]:
            offsets.append("{:+d}".format(offset))
        else:
            offsets.append("--")
    return (
        "b[{}]  mass {:.2f}@b{}  conf {:.2f}  bands {}/{}  {}".format(
            " ".join(offsets),
            result["mass_ratio"],
            result["junction_band"],
            result["confidence"],
            result["valid_band_count"],
            len(result["bands"]),
            describe_junction_flags(result["junction_flags"]),
        )
    )


def _median(values):
    """ulab 的 median 支持情况不一致，这里用排序取中值。"""
    ordered = sorted(values)
    count = len(ordered)
    if count == 0:
        return 0.0
    middle = count // 2
    if count % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) * 0.5


class LineTrackDetector:
    """在画面下部的若干条水平带内跟踪红色引导线。"""

    def __init__(
        self,
        roi_top_ratio=LINE_ROI_TOP_RATIO,
        roi_bottom_ratio=LINE_ROI_BOTTOM_RATIO,
        detect_width=LINE_DETECT_WIDTH,
        band_count=LINE_BAND_COUNT,
        band_height=LINE_BAND_HEIGHT,
        red_min_diff=LINE_RED_MIN_DIFF,
        red_min_value=LINE_RED_MIN_VALUE,
        main_min_column_count=LINE_MAIN_MIN_COLUMN_COUNT,
        edge_min_column_count=LINE_EDGE_MIN_COLUMN_COUNT,
        run_min_width=LINE_RUN_MIN_WIDTH,
        run_max_gap=LINE_RUN_MAX_GAP,
        run_max_width_ratio=LINE_RUN_MAX_WIDTH_RATIO,
        min_valid_bands=LINE_MIN_VALID_BANDS,
        junction_mass_ratio=LINE_JUNCTION_MASS_RATIO,
        junction_width_ratio=LINE_JUNCTION_WIDTH_RATIO,
        junction_min_bands=LINE_JUNCTION_MIN_BANDS,
        draw_band_color=LINE_DRAW_BAND_COLOR,
        draw_center_color=LINE_DRAW_CENTER_COLOR,
        draw_lost_color=LINE_DRAW_LOST_COLOR,
        draw_path_color=LINE_DRAW_PATH_COLOR,
        draw_junction_color=LINE_DRAW_JUNCTION_COLOR,
        draw_axis_color=LINE_DRAW_AXIS_COLOR,
        draw_text_color=LINE_DRAW_TEXT_COLOR,
        draw_thickness=LINE_DRAW_THICKNESS,
        draw_point_radius=LINE_DRAW_POINT_RADIUS,
        draw_font_scale=LINE_DRAW_FONT_SCALE,
        draw_data=LINE_DRAW_DATA_OVERLAY,
        draw_band_labels=LINE_DRAW_BAND_LABELS,
    ):
        if not 0.0 <= roi_top_ratio < roi_bottom_ratio <= 1.0:
            raise ValueError("必须满足 0 <= roi_top < roi_bottom <= 1")
        if detect_width <= 0 or band_count <= 0 or band_height <= 0:
            raise ValueError("检测宽度、带数和带高必须大于 0")
        if not 0 < edge_min_column_count <= main_min_column_count <= band_height:
            raise ValueError("必须满足 0 < 边缘阈值 <= 主线阈值 <= 带高")
        if not 0 < run_max_width_ratio <= 1.0:
            raise ValueError("run_max_width_ratio 必须在 0..1")
        if min_valid_bands < 1 or min_valid_bands > band_count:
            raise ValueError("min_valid_bands 必须在 1..band_count")
        if junction_mass_ratio <= 1.0:
            raise ValueError("junction_mass_ratio 必须大于 1")

        self.roi_top_ratio = float(roi_top_ratio)
        self.roi_bottom_ratio = float(roi_bottom_ratio)
        self.detect_width = int(detect_width)
        self.band_count = int(band_count)
        self.band_height = int(band_height)
        self.detect_height = self.band_count * self.band_height
        self.red_min_diff = int(red_min_diff)
        self.red_min_value = int(red_min_value)
        self.main_min_column_count = int(main_min_column_count)
        self.edge_min_column_count = int(edge_min_column_count)
        self.run_min_width = int(run_min_width)
        self.run_max_gap = int(run_max_gap)
        self.run_max_width_ratio = float(run_max_width_ratio)
        self.min_valid_bands = int(min_valid_bands)
        self.junction_mass_ratio = float(junction_mass_ratio)
        self.junction_width_ratio = float(junction_width_ratio)
        self.junction_min_bands = int(junction_min_bands)
        self.draw_band_color = tuple(draw_band_color)
        self.draw_center_color = tuple(draw_center_color)
        self.draw_lost_color = tuple(draw_lost_color)
        self.draw_path_color = tuple(draw_path_color)
        self.draw_junction_color = tuple(draw_junction_color)
        self.draw_axis_color = tuple(draw_axis_color)
        self.draw_text_color = tuple(draw_text_color)
        self.draw_thickness = int(draw_thickness)
        self.draw_point_radius = int(draw_point_radius)
        self.draw_font_scale = float(draw_font_scale)
        self.draw_data = bool(draw_data)
        self.draw_band_labels = bool(draw_band_labels)

        # CanMV 的精简版 OpenCV 没有 cv2.reduce，列投影只能用数组求和。
        # 不同固件的 numpy/ulab 对 axis 参数支持不一致，这里探测一次，
        # 之后每帧直接走选中的实现，不在主循环里做判断。
        self._axis_sum = self._probe_axis_sum()

        self.last_result = None
        self._target_valid = False
        self._offset_x = 0
        self._offset_y = 0

    @staticmethod
    def _probe_axis_sum():
        probe = np.zeros((2, 3), dtype=np.uint8)
        try:
            result = np.sum(probe, axis=0)
        except (TypeError, AttributeError, ValueError):
            return False
        try:
            return len(result) == 3
        except TypeError:
            return False

    # ------------------------------------------------------------
    # 串口状态
    # ------------------------------------------------------------

    def _update_target_state(self, frame, result):
        if result is None:
            self._target_valid = False
            self._offset_x = 0
            self._offset_y = 0
            return
        self._target_valid = True
        self._offset_x = int(frame.shape[1]) // 2 - int(result["center_x"])
        self._offset_y = int(frame.shape[0]) // 2 - int(result["center_y"])

    # ------------------------------------------------------------
    # 红色掩膜
    # ------------------------------------------------------------

    def _red_mask(self, roi):
        """RGB 通道差分，输出 0/1 掩膜。

        cv2.subtract 对 uint8 做饱和减法，负数截到 0，因此不需要先转
        有符号类型；直接用 `-` 会下溢成很大的正数。
        """
        red, green, blue = cv2.split(roi)
        green_diff = cv2.subtract(red, green)
        blue_diff = cv2.subtract(red, blue)
        mask = cv2.threshold(
            green_diff, self.red_min_diff, 1, cv2.THRESH_BINARY,
        )[1]
        mask = cv2.bitwise_and(
            mask,
            cv2.threshold(
                blue_diff, self.red_min_diff, 1, cv2.THRESH_BINARY,
            )[1],
        )
        mask = cv2.bitwise_and(
            mask,
            cv2.threshold(
                red, self.red_min_value, 1, cv2.THRESH_BINARY,
            )[1],
        )
        return mask

    # ------------------------------------------------------------
    # 列投影与 run
    # ------------------------------------------------------------

    def _column_counts(self, band_mask):
        """沿竖直方向求和，得到每一列的红色像素数。

        沿第 0 维（行）求和，输出一行，索引即列号，也就是 x 坐标。
        """
        if self._axis_sum:
            counts = np.sum(band_mask, axis=0)
        else:
            # 逐行相加，只用最基础的数组加法。掩膜是 0/1，带高远小于
            # 255，累加不会溢出 uint8。
            counts = band_mask[0]
            for row in range(1, self.band_height):
                counts = counts + band_mask[row]
        # 后面要逐列扫描。逐个索引数组元素每次都要装箱，先转成 Python
        # 列表再遍历，实测快约 3 倍，在解释执行的板端差距只会更大。
        try:
            return counts.tolist()
        except AttributeError:
            return list(counts)

    def _scan_counts(self, counts):
        """单次扫描同时得到主线 run 和红色整体横向范围。

        主线基本竖直，会填满整条带的高度，列计数接近 band_height；
        路口横线只占带内很少几行，列计数远低于此。因此用较高的
        main_min_column_count 提取 run 时，横线不会并进主线，主线中心
        在路口也不会被拽偏。较低的 edge_min_column_count 只用来测量
        红色向左右延伸到哪里，供分支判定使用。

        空隙不超过 run_max_gap 的两段 run 视为同一段，用于吸收胶带
        反光和褶皱造成的细小断点，因此不需要每帧做形态学运算。
        """
        spans = []
        start = None
        last = None
        edge_first = -1
        edge_last = -1
        for index in range(self.detect_width):
            count = counts[index]
            if count >= self.edge_min_column_count:
                if edge_first < 0:
                    edge_first = index
                edge_last = index
            if count < self.main_min_column_count:
                continue
            if start is None:
                start = index
            elif index - last - 1 > self.run_max_gap:
                spans.append((start, last))
                start = index
            last = index
        if start is not None:
            spans.append((start, last))

        runs = []
        for run_start, run_end in spans:
            width = run_end - run_start + 1
            if width < self.run_min_width:
                continue
            runs.append({
                "start": run_start,
                "end": run_end,
                "width": width,
                "center": (run_start + run_end) * 0.5,
            })
        return runs, edge_first, edge_last

    def _select_run(self, runs, preferred_center):
        """选择主线所在的 run。

        宽度超过上限的 run 是路口横线，不作为主线中心，但仍留在
        runs 里参与路口判定。全部超宽时退回最接近参照中心的一个。
        """
        max_width = self.detect_width * self.run_max_width_ratio
        best = None
        best_distance = None
        for run in runs:
            if run["width"] > max_width:
                continue
            distance = abs(run["center"] - preferred_center)
            if best_distance is None or distance < best_distance:
                best = run
                best_distance = distance
        if best is not None:
            return best
        for run in runs:
            distance = abs(run["center"] - preferred_center)
            if best_distance is None or distance < best_distance:
                best = run
                best_distance = distance
        return best

    # ------------------------------------------------------------
    # 路口
    # ------------------------------------------------------------

    def _junction_flags(self, bands):
        """按逐带红色超量比判定路口，倾斜的横线不会削弱该判据。

        预期红量由各带主线 run 宽度的中位数推算，代表"这条带如果只有
        主线，应该有多少红色像素"。横线横穿整幅画面，会让它所在的那
        一条带远超预期；弯道时主线自身变宽，预期量同步抬高，比值仍在
        1 附近，因此不会误判。

        比值必须逐带取最大，不能把 5 条带加总：横线通常只落在一到两
        条带里，求和会把它稀释掉。
        """
        widths = [band["width"] for band in bands if band["valid"]]
        if not widths:
            return JUNCTION_FLAG_LOST, 0.0, -1
        median_width = _median(widths)
        expected = median_width * self.band_height
        if expected <= 0:
            return 0, 0.0, -1

        best_ratio = 0.0
        best_index = -1
        wide_bands = 0
        for band in bands:
            ratio = band["mass"] / float(expected)
            band["mass_ratio"] = ratio
            if ratio > best_ratio:
                best_ratio = ratio
                best_index = band["index"]
            # 红色远超预期的带即使没有主线也要计入：T 型路口的横线所在
            # 带正是这种情况，主线到此为止，带内只剩横穿的红色。
            if (
                ratio >= self.junction_mass_ratio or
                (
                    band["valid"] and (
                        band["width"] >
                        median_width * self.junction_width_ratio or
                        band["run_count"] > 1
                    )
                )
            ):
                wide_bands += 1

        flags = 0
        if (
            best_ratio >= self.junction_mass_ratio and
            wide_bands >= self.junction_min_bands
        ):
            flags |= JUNCTION_FLAG_PRESENT
            side = self.detect_width * LINE_JUNCTION_SIDE_MIN_OFFSET_RATIO
            # 倾斜的横线会被带边界切成两段，落在相邻两条带里，其中一
            # 段只向左、另一段只向右延伸。因此左右分支必须在所有超量
            # 带上累计，只看比值最大的那一条会漏掉另外半边。
            for band in bands:
                if band["mass_ratio"] < self.junction_mass_ratio:
                    continue
                if band["edge_first"] >= 0 and (
                    band["edge_first"] < band["center"] - side
                ):
                    flags |= JUNCTION_FLAG_LEFT
                if band["edge_last"] >= 0 and (
                    band["edge_last"] > band["center"] + side
                ):
                    flags |= JUNCTION_FLAG_RIGHT
        return flags, best_ratio, best_index

    # ------------------------------------------------------------
    # 检测
    # ------------------------------------------------------------

    def _confidence(self, bands):
        valid = [band for band in bands if band["valid"]]
        coverage = len(valid) / float(self.band_count)
        if len(valid) < 2:
            return coverage * 0.6
        jumps = []
        for index in range(1, len(valid)):
            jumps.append(abs(valid[index]["center"] - valid[index - 1]["center"]))
        reference = self.detect_width * LINE_CONTINUITY_REF_RATIO
        continuity = 1.0 - min(1.0, (sum(jumps) / len(jumps)) / reference)
        return coverage * 0.6 + continuity * 0.4

    def detect(self, frame):
        """检测当前 RGB 帧，不修改输入画面。"""
        if frame is None or len(frame.shape) != 3:
            raise ValueError("frame 必须是 RGB 三通道图像")
        image_height = int(frame.shape[0])
        image_width = int(frame.shape[1])
        if image_width <= 1 or image_height <= 1:
            raise ValueError("frame 尺寸无效")

        top = int(round(image_height * self.roi_top_ratio))
        bottom = int(round(image_height * self.roi_bottom_ratio))
        top = max(0, min(top, image_height - 2))
        bottom = max(top + 1, min(bottom, image_height))

        # 切片是视图，不复制大图；整个感兴趣区域只缩放一次。
        #
        # 这里必须避开 INTER_AREA。它的开销由源像素数决定，会把整个
        # ROI 读一遍做面积平均，实测比 INTER_LINEAR 慢 15 倍以上，
        # 直接抵消掉只取下部区域省下的时间。红线有 50 像素以上宽，
        # 通道差分阈值又极具选择性，缩放时的抗锯齿没有实际价值。
        roi = frame[top:bottom, :, :]
        working = cv2.resize(
            roi,
            (self.detect_width, self.detect_height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask = self._red_mask(working)

        scale_x = image_width / float(self.detect_width)
        band_pixel_height = (bottom - top) / float(self.band_count)

        bands = []
        preferred_center = self.detect_width * 0.5
        for index in range(self.band_count):
            # index 0 是最下面、也就是离车最近的一条带。
            row_end = self.detect_height - index * self.band_height
            row_start = row_end - self.band_height
            band_mask = mask[row_start:row_end, :]
            counts = self._column_counts(band_mask)
            runs, edge_first, edge_last = self._scan_counts(counts)
            selected = self._select_run(runs, preferred_center)
            center_y = bottom - (index + 0.5) * band_pixel_height
            band = {
                "index": index,
                "runs": runs,
                "run_count": len(runs),
                "mass": float(cv2.countNonZero(band_mask)),
                "mass_ratio": 0.0,
                "edge_first": edge_first,
                "edge_last": edge_last,
                "valid": selected is not None,
                "center": selected["center"] if selected else preferred_center,
                "width": selected["width"] if selected else 0,
                "center_x": int(round(
                    (selected["center"] if selected else preferred_center) *
                    scale_x
                )),
                "center_y": int(round(center_y)),
                "offset": 0,
            }
            band["offset"] = image_width // 2 - band["center_x"]
            if selected is not None:
                preferred_center = selected["center"]
            bands.append(band)

        valid_count = sum(1 for band in bands if band["valid"])
        if valid_count < self.min_valid_bands:
            self.last_result = None
            self._update_target_state(frame, None)
            return None

        flags, mass_ratio, junction_band = self._junction_flags(bands)
        near = bands[0] if bands[0]["valid"] else next(
            band for band in bands if band["valid"]
        )
        result = {
            "center_x": near["center_x"],
            "center_y": near["center_y"],
            "confidence": float(self._confidence(bands)),
            "bands": tuple(bands),
            "offsets": tuple(
                band["offset"] if band["valid"] else 0 for band in bands
            ),
            "band_valid": tuple(band["valid"] for band in bands),
            "valid_band_count": valid_count,
            "mass_ratio": float(mass_ratio),
            "junction_flags": int(flags),
            "junction": bool(flags & JUNCTION_FLAG_PRESENT),
            "junction_band": int(junction_band),
            "roi_top": top,
            "roi_bottom": bottom,
        }
        self.last_result = result
        self._update_target_state(frame, result)
        return result

    # ------------------------------------------------------------
    # 绘制
    # ------------------------------------------------------------

    def draw(self, frame, result=None):
        """绘制指定结果；省略 result 时绘制最近一次检测结果。"""
        if result is None:
            result = self.last_result
        if result is None:
            return None

        image_height = int(frame.shape[0])
        image_width = int(frame.shape[1])
        center_x = image_width // 2

        cv2.line(
            frame,
            (center_x, result["roi_top"]),
            (center_x, image_height - 1),
            self.draw_axis_color,
            1,
        )

        band_pixel_height = (
            result["roi_bottom"] - result["roi_top"]
        ) / float(self.band_count)
        for index in range(self.band_count + 1):
            y = int(round(result["roi_bottom"] - index * band_pixel_height))
            y = max(0, min(y, image_height - 1))
            cv2.line(frame, (0, y), (image_width - 1, y), self.draw_band_color, 1)

        previous = None
        for band in result["bands"]:
            color = (
                self.draw_center_color if band["valid"] else self.draw_lost_color
            )
            point = (band["center_x"], band["center_y"])
            cv2.circle(frame, point, self.draw_point_radius, color, -1)
            if band["valid"]:
                if previous is not None:
                    cv2.line(
                        frame,
                        previous,
                        point,
                        self.draw_path_color,
                        self.draw_thickness,
                    )
                previous = point

            if not self.draw_band_labels:
                continue
            # 每条带的偏差直接标在该带的点旁边，一眼看出哪条带偏多少。
            # 贴近右边界时改标到点的左侧，避免文字被画面截断。
            label = "b{} {:+d}".format(band["index"], band["offset"]) \
                if band["valid"] else "b{} --".format(band["index"])
            label_x = point[0] + self.draw_point_radius + 4
            if label_x > image_width - 70:
                label_x = point[0] - self.draw_point_radius - 66
            cv2.putText(
                frame,
                label,
                (max(0, label_x), min(image_height - 3, point[1] + 4)),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.draw_font_scale * 0.8,
                color,
                1,
            )

        self._draw_junction(frame, result, image_width)

        if self.draw_data:
            self._draw_data_block(frame, result)
        return result

    def _draw_junction(self, frame, result, image_width):
        """把判为路口的那条带和它的红色横向范围标出来。

        属于几何层，不受 draw_data 控制：路口是本模块的主要输出之一，
        上车后仍然需要一眼确认它标在哪里。
        """
        index = result["junction_band"]
        if not result["junction"] or index < 0:
            return
        band = result["bands"][index]
        if band["edge_first"] < 0:
            return
        scale_x = image_width / float(self.detect_width)
        left = int(round(band["edge_first"] * scale_x))
        right = int(round(band["edge_last"] * scale_x))
        y = band["center_y"]
        # 横线在这条带里的实际横向范围，端点各画一小段竖线便于看清。
        cv2.line(
            frame, (left, y), (right, y),
            self.draw_junction_color, self.draw_thickness,
        )
        for x in (left, right):
            cv2.line(
                frame, (x, y - 10), (x, y + 10),
                self.draw_junction_color, self.draw_thickness,
            )
        cv2.putText(
            frame,
            "JUNCTION b{}".format(index),
            (max(0, min(left, image_width - 150)), max(14, y - 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.draw_font_scale,
            self.draw_junction_color,
            self.draw_thickness,
        )

    def _draw_data_block(self, frame, result):
        """在画面左上角列出本帧的全部关键数据。"""
        offsets = []
        for index, offset in enumerate(result["offsets"]):
            if result["band_valid"][index]:
                offsets.append("{:+5d}".format(offset))
            else:
                offsets.append("   --")
        lines = [
            "b0..b4 {}".format(" ".join(offsets)),
            "mass {:.2f} @b{}   conf {:.2f}   bands {}/{}".format(
                result["mass_ratio"],
                result["junction_band"],
                result["confidence"],
                result["valid_band_count"],
                len(result["bands"]),
            ),
            "flags {:02X}  {}".format(
                result["junction_flags"],
                describe_junction_flags(result["junction_flags"]),
            ),
        ]
        origin_x, origin_y = LINE_DRAW_DATA_ORIGIN
        for index, text in enumerate(lines):
            cv2.putText(
                frame,
                text,
                (origin_x, origin_y + index * LINE_DRAW_DATA_LINE_HEIGHT),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.draw_font_scale,
                self.draw_text_color,
                self.draw_thickness,
            )

    def draw_lost(self, frame):
        """本帧没有检测结果时的提示。detect() 返回 None 时由调用方使用。"""
        if not self.draw_data:
            return None
        origin_x, origin_y = LINE_DRAW_DATA_ORIGIN
        cv2.putText(
            frame,
            "LINE LOST  valid=0",
            (origin_x, origin_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.draw_font_scale,
            self.draw_lost_color,
            self.draw_thickness,
        )
        return None

    def process(self, frame, draw=True):
        """检测一帧并按需绘制，返回结果字典或 None。"""
        result = self.detect(frame)
        if draw and result is not None:
            self.draw(frame, result)
        return result


class JunctionConfirmState:
    """连续多帧确认路口，避免单帧抖动就切换状态。

    只服务于主程序的状态切换，不属于检测器，也不会把历史结果当成当前
    帧的检测结果发送。
    """

    def __init__(self, confirm_frames=LINE_JUNCTION_CONFIRM_FRAMES):
        if confirm_frames < 1:
            raise ValueError("confirm_frames 必须大于等于 1")
        self.confirm_frames = int(confirm_frames)
        self.streak = 0

    def update(self, result):
        if result is None or not result["junction"]:
            self.streak = 0
            return False
        self.streak += 1
        return self.streak >= self.confirm_frames

    def reset(self):
        self.streak = 0


def run_line_demo(
    display_target=None,
    enable_uart=False,
    draw_data=True,
    print_interval=LINE_DEMO_PRINT_INTERVAL,
):
    """使用 CameraIO 运行红线巡线演示，并按需发送 LINE 帧。

    enable_uart 默认关闭。开启后会阻塞等待与单片机握手，且该等待没有
    超时；单独调视觉时若单片机没接或没在跑，程序会停在握手上不出画面。
    需要发送 LINE 帧时显式传入 True。

    draw_data 控制画面上的数据叠加层和每条带的偏差标注。调试时留 True
    看数值，实际上车传 False 关掉，省下每帧十几次 putText。
    print_interval 是终端打印同一份数据的帧间隔，传 0 关闭。
    """
    import gc
    import sys
    import time

    from camera_io import CameraIO, DISPLAY_TARGET_IDE

    if display_target is None:
        display_target = DISPLAY_TARGET_IDE
    camera = None
    tracking_uart = None
    detector = LineTrackDetector(
        draw_data=draw_data,
        draw_band_labels=draw_data,
    )
    junction_state = JunctionConfirmState()
    frame_count = 0

    try:
        print("================================")
        print("K230 红线巡线")
        print("检测区域：画面高度 {:.2f}~{:.2f}".format(
            detector.roi_top_ratio,
            detector.roi_bottom_ratio,
        ))
        print("检测分辨率：{}x{}（{} 条带，每条 {} 行）".format(
            detector.detect_width,
            detector.detect_height,
            detector.band_count,
            detector.band_height,
        ))
        print("显示目标：{}".format(display_target))
        print("串口发送：{}".format("开" if enable_uart else "关"))
        print("数据叠加：{}".format("开" if draw_data else "关"))
        print("================================")

        if enable_uart:
            from uart_io import TrackingUART

            tracking_uart = TrackingUART().initialize()
            print("等待与单片机完成握手")
            tracking_uart.wait_for_handshake()
            print("握手完成")

        camera = CameraIO(display_target=display_target)
        camera.initialize()
        clock = time.clock()

        while True:
            clock.tick()
            image = camera.snapshot()
            frame = image.to_numpy_ref()
            result = detector.process(frame)
            if result is None:
                detector.draw_lost(frame)

            if tracking_uart is not None:
                tracking_uart.send_line(result)

            if print_interval and frame_count % print_interval == 0:
                print(format_result(result))

            if junction_state.update(result):
                # TODO: 切换到 num.py 的 DigitDetector 识别病房号，
                # 并用 MSG_TYPE_DIGIT 上报。识别结束后调用
                # junction_state.reset() 回到巡线。
                cv2.putText(
                    frame,
                    "JUNCTION CONFIRMED",
                    (
                        LINE_DRAW_DATA_ORIGIN[0],
                        LINE_DRAW_DATA_ORIGIN[1] +
                        3 * LINE_DRAW_DATA_LINE_HEIGHT + 6,
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    LINE_DRAW_TEXT_COLOR,
                    2,
                )

            cv2.putText(
                frame,
                "FPS: {:.1f}".format(clock.fps()),
                (5, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                LINE_DRAW_AXIS_COLOR,
                2,
            )
            camera.show_image(image)

            frame_count += 1
            if frame_count % LINE_DEMO_GC_INTERVAL == 0:
                gc.collect()
            del frame
            del image

    except KeyboardInterrupt:
        print("用户停止程序")
    except Exception as error:
        sys.print_exception(error)
    finally:
        print("正在释放资源")
        if camera is not None:
            camera.deinitialize()
        if tracking_uart is not None:
            tracking_uart.deinitialize()
        gc.collect()
        print("程序结束")


if __name__ == "__main__":
    run_line_demo()
