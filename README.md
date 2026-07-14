# K230 视觉检测模块说明

本项目把摄像头生命周期、检测参数和各类视觉算法分开。检测模块统一接收一帧 RGB 图像，不在导入时初始化摄像头、显示器或串口，因此可以在 `num.py`、`tangle.py` 或新的主程序中组合使用。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `config.py` | 摄像头、显示、串口和各检测器的默认参数 |
| `camera_io.py` | `Sensor`、`Display`、`MediaManager` 生命周期 |
| `color.py` | 彩色光点检测，导出 `ColorSpotDetector` |
| `tangle.py` | 黑框白心方框检测，导出 `RectangleDetector`；直接运行时也是完整追踪程序 |
| `num.py` | 打印数字检测，导出 `DigitDetector`；直接运行时也是完整识别程序 |

原 `rectangle_detector.py` 已合并进 `tangle.py`，不再需要上传。

## 统一调用形式

所有检测器都遵循同一套接口：

```python
from color import ColorSpotDetector
from tangle import RectangleDetector
from num import DigitDetector

# 主循环外初始化一次。数字模板也只会在这里加载一次。
color_detector = ColorSpotDetector()
rectangle_detector = RectangleDetector()
digit_detector = DigitDetector()

# 获取 frame 后调用；process 默认会在 frame 上绘图。
spot = color_detector.process(frame)
rectangle = rectangle_detector.process(frame)
digit_result = digit_detector.process(frame)
```

三个公共方法的约定如下：

- `detect(frame)`：只检测，不绘图；有结果时返回字典，没有结果时返回 `None`。
- `draw(frame, result)`：只绘制已有结果，并返回该结果。
- `process(frame, draw=True)`：调用 `detect`，并在 `draw=True` 时调用 `draw`，返回检测结果。

需要自己控制绘制顺序时，使用：

```python
result = detector.process(frame, draw=False)
if result is not None:
    detector.draw(frame, result)
```

`frame` 应为 `image.to_numpy_ref()` 返回的 RGB 图像。检测器应在主循环外创建，不能每帧重复创建，否则会重复分配内存，数字模块还会重复加载模板。

## 彩色光点模块

```python
from color import ColorSpotDetector

color_detector = ColorSpotDetector(target_color="red")
spot = color_detector.process(frame)

if spot is not None:
    x = spot["center_x"]
    y = spot["center_y"]
    confidence = spot["confidence"]
```

主要返回字段：

| 字段 | 含义 |
| --- | --- |
| `center_x`, `center_y` | 光点中心坐标 |
| `confidence` | 当前候选的圆度与外接框填充率乘积 |
| `x`, `y` | 与 `center_x`、`center_y` 相同的兼容字段 |
| `bbox` | 外接矩形 `(x, y, w, h)` |
| `area` | 光点轮廓面积 |

运行期间可以用 `set_color()` 切换预设颜色，也可以在构造函数或 `set_color()` 中传入自定义 HSV 范围。

## 方框检测模块

```python
from tangle import RectangleDetector

rectangle_detector = RectangleDetector()
rectangle = rectangle_detector.process(frame)

if rectangle is not None:
    x = rectangle["center_x"]
    y = rectangle["center_y"]
    confidence = rectangle["confidence"]
```

主要返回字段：

| 字段 | 含义 |
| --- | --- |
| `center_x`, `center_y` | 方框两条对角线交点 |
| `confidence` | 几何、内外边缘对比度及四边一致性的加权分数 |
| `points` | 四边形的四个角点 |
| `x`, `y`, `w`, `h` | 四角点形成的外接范围 |
| `mean_edge_contrast` | 四条边的平均“内亮外暗”灰度差 |
| `min_edge_contrast` | 四条边中最低的灰度差 |
| `source` | 候选来源，`bright` 或 `canny` |

`RectangleDetector` 每帧独立检测，不使用历史位置、ROI 或运动预测。`tangle.py` 直接运行时的 `TargetHoldState` 只服务于示例画面显示，不属于检测器，也不会把历史坐标作为有效串口数据发送。

## 数字检测模块

```python
from num import DigitDetector

# 默认自动探测 config.py 中的模板目录。
digit_detector = DigitDetector()
result = digit_detector.process(frame)

if result is not None:
    text = result["text"]
    digits = result["digits"]
    count = result["count"]
    confidence = result["confidence"]
```

电脑离线测试或使用自定义模板目录时：

```python
digit_detector = DigitDetector(template_dir="_digit_templates")
```

整串返回字段：

| 字段 | 含义 |
| --- | --- |
| `text` | 从左到右的识别文本，低于阈值的位用 `?` 表示 |
| `digits` | 每一位数字的结果字典列表 |
| `count` | 数字候选数量 |
| `recognized_count` | 达到匹配阈值的数量 |
| `confidence` | 各候选模板匹配分数截取到 0..1 后的平均值 |
| `center_x`, `center_y`, `bbox` | 整串数字的中心和外接矩形 |

每个 `digits` 元素包含：

- `value`：识别成功时为 `0..9`，未达到阈值时为 `-1`。
- `text`：数字字符或 `?`。
- `recognized`：是否达到匹配阈值。
- `confidence`：该位与最佳模板的归一化相关系数。
- `center_x`、`center_y`、`x`、`y`、`w`、`h`、`bbox`、`area`：位置和轮廓信息。

## 参数管理规则

模块默认参数统一放在 `config.py`，并按模块使用前缀：

- `COLOR_...`：彩色光点。
- `RECTANGLE_...`：方框。
- `DIGIT_...`：数字。

构造函数参数用于临时覆盖默认值。例如：

```python
rectangle_detector = RectangleDetector(min_confidence=0.55)
digit_detector = DigitDetector(match_threshold=0.40)
```

修改阈值时应基于真实图片重新验证。不同检测器的 `confidence` 计算方法不同，只能用于该检测器内部候选排序或阈值判断，不能直接横向比较。

## 新检测模块规范

扩充新模块时，建议按以下骨架实现：

```python
from config import MY_TARGET_MIN_CONFIDENCE


class MyTargetDetector:
    def __init__(self, min_confidence=MY_TARGET_MIN_CONFIDENCE):
        self.min_confidence = min_confidence
        # 只初始化一次可复用的模板、核或缓冲区。

    def detect(self, frame):
        # 不绘图，不初始化硬件。
        # 没有结果返回 None；有结果返回至少包含
        # center_x、center_y、confidence 的字典。
        return None

    def draw(self, frame, result):
        if result is None:
            return None
        # 在 frame 上绘制。
        return result

    def process(self, frame, draw=True):
        result = self.detect(frame)
        if draw and result is not None:
            self.draw(frame, result)
        return result
```

新增模块还应遵守这些约束：

1. 模块导入时不能自动启动摄像头、显示器、串口或死循环。
2. 硬件演示入口放在 `run_xxx_demo()` 中，并由 `if __name__ == "__main__":` 调用。
3. 可复用对象在构造函数中创建一次，不在 `detect()` 中反复加载模板或创建固定形态学核。
4. `detect()` 不修改输入画面；所有可视化集中在 `draw()`。
5. 无目标必须返回 `None`，不能返回上一帧目标冒充当前检测结果。
6. 返回字典至少提供 `center_x`、`center_y` 和含义明确的 `confidence`；模块专属字段写入本文档。
7. 新参数进入 `config.py` 并使用模块前缀，构造函数保留同名覆盖入口。
8. 多个检测器组合时只获取一次 `frame`，依次处理同一帧；如需控制覆盖顺序，先全部 `detect()`，再按顺序 `draw()`。

## 上传到 K230

方框功能至少需要：

- `tangle.py`
- `color.py`（仅直接运行 `tangle.py` 的组合示例需要）
- `config.py`
- `camera_io.py`（仅直接运行完整摄像头示例需要）

数字功能至少需要：

- `num.py`
- `config.py`
- 数字模板目录及 `0.png` 到 `9.png`
- `camera_io.py`（仅直接运行完整摄像头示例需要）

作为模块导入时，调用方自行负责摄像头、显示和串口生命周期。
