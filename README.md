# K230 视觉检测模块说明

本项目把摄像头生命周期、检测参数和各类视觉算法分开。检测模块统一接收一帧 RGB 图像，不在导入时初始化摄像头、显示器或串口，因此可以在 `num.py`、`tangle.py` 或新的主程序中组合使用。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `config.py` | 摄像头、显示、串口和各检测器的默认参数 |
| `camera_io.py` | `Sensor`、`Display`、`MediaManager` 生命周期 |
| `uart_io.py` | FPIOA、UART 生命周期、原始读写和目标偏差协议 |
| `bluetooth_uart.py` | UART2 蓝牙单字符指令接收与缓存，导出 `BluetoothUART` |
| `color.py` | 彩色光点检测，导出 `ColorSpotDetector` |
| `tangle.py` | 黑框白心方框检测，导出 `RectangleDetector`；直接运行时也是完整追踪程序 |
| `pencil_rectangle.py` | 细铅笔线方框检测，导出 `PencilRectangleDetector`；多重方框中选择估算边框最细者 |
| `corner_cycle.py` | 独立的方框四角顺时针停留、移动和串口输出应用 |
| `num.py` | 打印数字检测，导出 `DigitDetector`；直接运行时也是完整识别程序 |

原 `rectangle_detector.py` 已合并进 `tangle.py`，不再需要上传。

## 统一调用形式

所有检测器都遵循同一套接口：

```python
from color import ColorSpotDetector
from tangle import RectangleDetector
from pencil_rectangle import PencilRectangleDetector
from num import DigitDetector

# 主循环外初始化一次。数字模板也只会在这里加载一次。
color_detector = ColorSpotDetector()
rectangle_detector = RectangleDetector()
pencil_rectangle_detector = PencilRectangleDetector()
digit_detector = DigitDetector()

# 获取 frame 后调用；process 默认会在 frame 上绘图。
spot = color_detector.process(frame)
rectangle = rectangle_detector.process(frame)
pencil_rectangle = pencil_rectangle_detector.process(frame)
digit_result = digit_detector.process(frame)
```

三个公共方法的约定如下：

- `detect(frame)`：只检测，不绘图；有结果时返回字典，没有结果时返回 `None`。
- `draw(frame, result)`：只绘制已有结果，并返回该结果。
- `process(frame, draw=True)`：调用 `detect`，并在 `draw=True` 时调用 `draw`，返回检测结果。

每个检测器还在当前帧检测结束时原地更新三个供串口发送使用的私有属性：

- `_target_valid`：当前帧是否真实检测到目标。
- `_offset_x`：画面中心横坐标减去目标中心横坐标。
- `_offset_y`：画面中心纵坐标减去目标中心纵坐标。

未检测到目标时，这三个属性固定为 `False, 0, 0`，不会保留上一帧坐标。属性只保存布尔值和整数，不复制图像，也不额外创建结果容器。

需要自己控制绘制顺序时，使用：

```python
result = detector.process(frame, draw=False)
if result is not None:
    detector.draw(frame, result)
```

`frame` 应为 `image.to_numpy_ref()` 返回的 RGB 图像。检测器应在主循环外创建，不能每帧重复创建，否则会重复分配内存，数字模块还会重复加载模板。

## 摄像头与显示

`CameraIO` 统一管理 `Sensor`、`Display` 和 `MediaManager`。构造时选择板载屏幕或 CanMV IDE，初始化和释放各调用一次：

```python
from camera_io import CameraIO, DISPLAY_TARGET_BOARD, DISPLAY_TARGET_IDE

camera = CameraIO(display_target=DISPLAY_TARGET_IDE).initialize()

try:
    image = camera.snapshot()
    frame = image.to_numpy_ref()
    camera.show_image(image)
finally:
    camera.deinitialize()
```

板载屏幕和 IDE 的分辨率、位置、帧率、质量参数分别由 `config.py` 中的 `TANGLE_DISPLAY_...` 和 `NUM_DISPLAY_...` 管理。当前摄像头同时启用水平镜像和垂直翻转，等效于旋转 180°。

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

直接发送红点偏差时，不在主循环重复计算坐标：

```python
tracking_uart.send_target(
    color_detector._target_valid,
    color_detector._offset_x,
    color_detector._offset_y,
)
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

高亮红点的中心可能因过曝变成低饱和白色。当前检测器先在全图基础颜色掩膜中选择 `confidence` 最高的一个候选，再把该候选外接框按高亮核半径扩展为当前帧 ROI；高亮阈值、膨胀、掩膜合并和补全后的轮廓评分只在这个 ROI 内执行。它不依赖上一帧位置，因此目标快速移动不会跑出历史搜索区域；远离基础颜色候选的普通白色物体也不会被并入目标。该策略适用于画面干净、只需要处理基础评分最高候选的场景；如果现场出现多个强颜色干扰物，最终结果仍取决于补全前的基础 `confidence` 排序。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `COLOR_INCLUDE_HIGHLIGHT` | `True` | 是否对当前帧基础评分最高候选启用 ROI 过曝中心补全。 |
| `COLOR_HIGHLIGHT_MAX_SATURATION` | `90` | 高亮中心允许的最大HSV饱和度。 |
| `COLOR_HIGHLIGHT_MIN_VALUE` | `220` | 高亮中心要求的最低HSV亮度。 |
| `COLOR_HIGHLIGHT_KERNEL_SIZE` | `15` | 搜索颜色区域邻近高亮像素的椭圆核尺寸，必须是正奇数。 |

`ColorSpotDetector.process(frame)` 默认绘制目标中心、画面中心、两点连线和 `X/Y` 相对偏差。偏差定义与串口一致：`画面中心坐标 - 目标中心坐标`。

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

直接发送方框偏差：

```python
tracking_uart.send_target(
    rectangle_detector._target_valid,
    rectangle_detector._offset_x,
    rectangle_detector._offset_y,
)
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

## 细铅笔线方框模块

`PencilRectangleDetector` 面向白色纸面上的闭合细黑线方框。它与检测黑色电工胶带框的 `RectangleDetector` 相互独立：自适应二值图只负责生成闭合凸四边形候选，每个轮廓独立参与检测，不再要求外轮廓和内轮廓成对。候选形成后，检测器回到原始灰度图，沿四条边的法线测量真实暗线宽度。多个合格方框同时出现时，按线宽中位数从小到大选择，不按面积、轮廓层级或画面位置选择。

```python
from pencil_rectangle import PencilRectangleDetector

# 主循环外初始化一次。
pencil_detector = PencilRectangleDetector()

# 获取 RGB frame 后调用；默认在画面上标出四角、中心和估算线宽。
rectangle = pencil_detector.process(frame)

if rectangle is not None:
    x = rectangle["center_x"]
    y = rectangle["center_y"]
    confidence = rectangle["confidence"]
    border_thickness = rectangle["border_thickness"]
    corners = rectangle["points"]
```

主要返回字段：

| 字段 | 含义 |
| --- | --- |
| `center_x`, `center_y` | 所选四边形的中心 |
| `points` | 按左上、右上、右下、左下排列的候选四角点 |
| `border_thickness` | 四条边灰度剖面线宽的中位数，单位为原图像素 |
| `side_thicknesses` | 四条边各自的估算线宽 |
| `thickness_uniformity` | 最细边与最粗边线宽之比，越接近 1 越均匀 |
| `edge_contrasts` | 四条边灰度剖面的暗线对比度 |
| `min_edge_contrast` | 四条边中的最低暗线对比度 |
| `straightness_score` | 原轮廓周长与拟合四边形周长的一致程度 |
| `geometry_score` | 四个角接近直角的程度 |
| `parallel_score` | 两组对边的平行程度 |
| `confidence` | 只由直线程度、四角几何和对边平行程度组成，不包含线宽或覆盖率 |

直接发送所选细框的中心偏差：

```python
tracking_uart.send_target(
    pencil_detector._target_valid,
    pencil_detector._offset_x,
    pencil_detector._offset_y,
)
```

直接运行主程序时，会复用 `corner_cycle.py` 的 `CornerCycleController` 和绘制函数，在所选细框的四个角之间执行 `TL -> TR -> BR -> BL -> TL` 轨迹，并通过 `TrackingUART` 发送当前插值点相对画面中心的偏差：

```python
import pencil_rectangle

pencil_rectangle.run_pencil_rectangle_demo()
```

停留、移动和串口周期可以覆盖：

```python
pencil_rectangle.run_pencil_rectangle_demo(
    hold_ms=3000,
    move_ms=3000,
    uart_send_period_ms=10,
)
```

检测到细框时，主程序先用 `points` 更新四角轨迹，再发送 `valid=1` 和轨迹点偏差；目标丢失时发送 `valid=0,x=0,y=0`，并暂停轨迹计时。重新检测到方框后从暂停位置继续。检测器本身仍只负责单帧检测和绘制，导入 `PencilRectangleDetector` 不会初始化串口或轨迹控制器。

细线在降采样后容易消失，所以默认使用 `640x480` 全分辨率检测；它通常会比 `RectangleDetector` 的 `320x240` 检测更耗时。若设备帧率不足，可覆盖 `detect_width`、`detect_height`，但必须用真实图片确认目标线仍能形成闭合四边形。自适应阈值和默认 `7x5` 椭圆闭运算核只负责生成候选，闭运算后的二值线宽不参与最终线宽比较。

每条边默认在避开角点的位置取 5 个法线灰度剖面，在候选边附近寻找最暗位置，以两端较亮背景和暗线中点阈值测量连续暗带宽度。每条边取采样宽度中位数，整个方框再取四边中位数。程序没有“四边有效覆盖率”评分；但四条边都必须达到最低灰度差，避免把没有黑线的纸张边缘当成目标。

`confidence` 按 `35:25:15` 组合四边直线程度、四角几何和对边平行程度，初始化时会自动归一化。最终仍先选择 `border_thickness` 最小的候选；线宽差不超过 `PENCIL_RECTANGLE_THICKNESS_TIE_PX` 时才比较 `confidence`。这不会保留上一帧目标，也不会引入位置预测。

模块已使用 `pic` 目录中的 13 张实拍图片校准。在这批图片中，新方法选择细铅笔框 13 次、选择粗胶带框 0 次；铅笔框测得约 `1.0~2.0 px`，胶带候选测得约 `7.0~8.0 px`。该结果只代表当前拍摄距离、光照和线宽；现场条件改变后仍应按下面顺序调整：

- `PENCIL_RECTANGLE_ADAPTIVE_BLOCK_SIZE`、`PENCIL_RECTANGLE_ADAPTIVE_C`：局部二值化范围和灵敏度。
- `PENCIL_RECTANGLE_CLOSE_KERNEL_WIDTH`、`CLOSE_KERNEL_HEIGHT`：小断点连接强度；过大会把相邻线条粘连。
- `PENCIL_RECTANGLE_USE_ELLIPSE_CLOSE_KERNEL`：默认使用椭圆核，减小连接相邻物体的概率；固件不支持时会自动退回矩形核。
- `PENCIL_RECTANGLE_MIN_AREA`、`MIN_WIDTH`、`MIN_HEIGHT`：排除文字和小型四边形干扰。
- `PENCIL_RECTANGLE_PROFILE_SCAN_RADIUS`、`PROFILE_SEARCH_RADIUS`：控制法线扫描范围和候选边附近的暗线搜索范围。
- `PENCIL_RECTANGLE_PROFILE_DARK_RATIO`、`MIN_EDGE_CONTRAST`：控制暗带宽度阈值和最低黑白灰度差。
- `PENCIL_RECTANGLE_MIN_BORDER_THICKNESS`、`MAX_BORDER_THICKNESS`、`MIN_THICKNESS_UNIFORMITY`：控制线宽范围和四边一致性。
- `PENCIL_RECTANGLE_STRAIGHTNESS_WEIGHT`、`GEOMETRY_WEIGHT`、`PARALLEL_WEIGHT`：调整纯几何置信度的组成。

需要降低灰度剖面采样开销时，可以先用下面这组覆盖参数。它在当前 13 张实拍图中仍保持 `13/13` 选择细框，但 K230 上的实际帧率提升应以板端日志为准：

```python
pencil_detector = PencilRectangleDetector(
    max_candidates=12,
    profile_sample_count=3,
    profile_scan_radius=12,
    profile_search_radius=5,
    profile_end_count=3,
)
```

## 方框四角循环应用

`corner_cycle.py` 是独立应用，不把轨迹状态加入 `RectangleDetector` 或 `TrackingUART`。它只调用现有方框检测和串口接口，内部完成四角排序、计时和线性插值。

直接在 K230 上运行：

```python
import corner_cycle

corner_cycle.run_corner_cycle()
```

默认循环为：

1. 左上角 `TL` 停留 3 秒。
2. 用 3 秒从左上角线性移动到右上角 `TR`。
3. 右上角停留 3 秒，再用 3 秒移动到右下角 `BR`。
4. 右下角停留 3 秒，再用 3 秒移动到左下角 `BL`。
5. 左下角停留 3 秒，再用 3 秒回到左上角，然后重复。

四角在每帧中统一排序为 `TL -> TR -> BR -> BL`，不直接依赖 OpenCV 返回轮廓点的起点和方向。屏幕会显示四个角点编号、当前插值点、阶段剩余时间和相对坐标。

串口发送的仍是：

```text
T,frame,valid,x,y\n
```

其中 `x = 画面中心X - 当前轨迹点X`，`y = 画面中心Y - 当前轨迹点Y`。方框丢失时发送 `valid=0,x=0,y=0`，循环计时暂停；重新检测到方框后从暂停位置继续，不发送历史角点坐标。

可以在调用时调整时间和串口最小周期：

```python
corner_cycle.run_corner_cycle(
    hold_ms=3000,
    move_ms=3000,
    uart_send_period_ms=10,
)
```

移动时间由系统毫秒计时保证，但坐标只会在主循环处理完新画面时更新。当前摄像头为 30 FPS，因此一次 3 秒移动通常最多约有 90 个画面更新点，并不是每 10 ms 生成一个新图像坐标。

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

需要发送整串数字目标的位置时，也使用 `digit_detector` 的同名三个私有属性。

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

## 串口模块

`uart_io.py` 导出 `TrackingUART`。默认配置是：

- UART1
- TX：GPIO3
- RX：GPIO4
- 波特率：115200
- 周期发送最小间隔：10 ms
- 8 数据位、无校验、1 停止位

接线时，K230 的 TX（GPIO3）连接单片机 RX，K230 的 RX（GPIO4）连接单片机 TX，两块板还需要共地。如果只需要 K230 单向发送，可以不连接 K230 的 RX，但 `TrackingUART` 仍会按配置完成 GPIO4 的映射。

最少调用方式：

```python
from tangle import RectangleDetector
from uart_io import TrackingUART

# 主循环外初始化一次。
rectangle_detector = RectangleDetector()
tracking_uart = TrackingUART(send_period_ms=10).initialize()

# 主循环中直接传入检测器本帧更新的私有状态。
tracking_uart.send_target(
    rectangle_detector._target_valid,
    rectangle_detector._offset_x,
    rectangle_detector._offset_y,
)

# 程序退出时释放。
tracking_uart.deinitialize()
```

`send_period_ms` 是两次实际发送之间的最小间隔。主循环可以每帧调用
`send_target()`；未到周期时方法返回 `None`，不会写 UART。到达周期时返回
实际数据包字符串。需要忽略周期立即发送时传入 `force=True`。

```python
packet = tracking_uart.send_target(
    rectangle_detector._target_valid,
    rectangle_detector._offset_x,
    rectangle_detector._offset_y,
)

if packet is not None:
    print("本次已发送：{}".format(packet))
```

需要注意：10 ms 对应最高 100 次/秒。当前摄像头配置为 30 FPS，相邻画面
约 33 ms，因此每帧调用时仍然会每帧发送。如果目的是降低 30 FPS 主循环的
发送次数，应把周期设为大于约 33 ms，例如 50 ms 或 100 ms。

`send_target()` 的数据格式为一行 ASCII：

```text
T,frame,valid,x,y\n
```

例如：

```text
T,25,1,-18,7
```

字段含义：

| 字段 | 含义 |
| --- | --- |
| `T` | 数据包类型，可通过 `packet_prefix` 修改 |
| `frame` | 帧号；不传 `frame_id` 时由模块从 0 自动递增 |
| `valid` | `1` 表示当前帧有目标，`0` 表示无目标；接收端应先判断此字段 |
| `x`, `y` | 目标相对画面中心的偏差 |

当 `valid=0` 时，模块会强制把 `x`、`y` 发送为 0。此时的 `0,0` 只是无效包占位值，单片机不能把它解释为“目标位于画面中心”。需要显式使用主程序帧号时：

```python
tracking_uart.send_target(
    rectangle_detector._target_valid,
    rectangle_detector._offset_x,
    rectangle_detector._offset_y,
    frame_id=frame_count,
)
```

临时修改串口参数不需要改 `config.py`：

```python
tracking_uart = TrackingUART(
    uart_id=1,
    tx_pin=3,
    rx_pin=4,
    baudrate=230400,
    send_period_ms=50,
).initialize()
```

除目标协议外，模块也提供 `write()`、`write_periodic()`、`any()`、`read()` 和 `readline()`，可以用于自定义协议。`write()` 始终立即发送，`write_periodic()` 才应用周期：

```python
tracking_uart.write("HELLO\n")

if tracking_uart.any() > 0:
    received = tracking_uart.read()
```

不要在每一帧中重新创建或初始化 `TrackingUART`。导入 `uart_io.py` 不会占用 GPIO；只有调用 `initialize()` 才会加载 `machine` 并初始化硬件。

## 蓝牙串口模块

`bluetooth_uart.py` 导出 `BluetoothUART`，通过 UART2 接收蓝牙透明串口传来的单字符指令。当前只接受 `1`、`2`、`3`、`4`、`s`、`p`，自动忽略空格、制表符、回车和换行；其他字符会被报告并丢弃。该模块不通过蓝牙发送数据。

默认配置为：

- UART2
- TX：GPIO11
- RX：GPIO12
- 波特率：9600

不同蓝牙模块的数据模式波特率可能不同，必须以你的蓝牙模块配置为准。GPIO11/12 同时也是 I2C2 引脚，使用蓝牙 UART2 时不能再把同一组引脚用于 I2C2。

接线方式：K230 GPIO11（TX）连接蓝牙模块 RX，K230 GPIO12（RX）连接蓝牙模块 TX，并连接 GND。模块供电电压和 UART IO 电平应以蓝牙模块规格为准。

```python
from bluetooth_uart import BluetoothUART

# BluetoothUART 构造时会自动初始化 UART2。
bluetooth = BluetoothUART(baudrate=9600)

commands = bluetooth.receive()
if commands is not None:
    for command in commands:
        print("收到指令：", command)

bluetooth.deinitialize()
```

`receive()` 返回本次缓存的全部有效指令列表；没有指令时返回 `None`。需要逐个处理时使用 `receive_one()`：

```python
command = bluetooth.receive_one()
if command == "s":
    print("start")
elif command == "p":
    print("pause")
```

`available()` 会先读取 UART，再返回缓存中的指令数量；`clear()` 同时清空软件缓存和当前 UART 残留数据。追踪串口使用 UART1（GPIO3/4），蓝牙接收使用 UART2（GPIO11/12），两者可以同时工作。程序退出时应调用 `deinitialize()`。

## 参数管理规则

模块默认参数统一放在 `config.py`，并按模块使用前缀：

- `COLOR_...`：彩色光点。
- `RECTANGLE_...`：方框。
- `PENCIL_RECTANGLE_...`：细铅笔线方框。
- `DIGIT_...`：数字。
- `UART_...`：串口和目标偏差协议。
- `BLUETOOTH_UART_...`：蓝牙指令接收串口。

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
        self._target_valid = False
        self._offset_x = 0
        self._offset_y = 0

    def _update_target_state(self, frame, result):
        if result is None:
            self._target_valid = False
            self._offset_x = 0
            self._offset_y = 0
            return
        self._target_valid = True
        self._offset_x = int(frame.shape[1]) // 2 - int(result["center_x"])
        self._offset_y = int(frame.shape[0]) // 2 - int(result["center_y"])

    def detect(self, frame):
        # 不绘图，不初始化硬件。
        # 没有结果返回 None；有结果返回至少包含
        # center_x、center_y、confidence 的字典。
        result = None
        self._update_target_state(frame, result)
        return result

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
9. 需要发送到串口的本帧状态优先封装为检测器的私有标量属性，并在 `detect()` 内与结果同时更新；调用方直接把属性传给 `TrackingUART`，不要重复计算。只有确认不会引入图像复制、大对象分配或明显降低帧率时才采用此方式。
10. 有显示输出的可运行主程序应在主循环外创建 `time.clock()`，每帧调用 `tick()`，并在 `show_image()` 前把当前 `FPS` 绘制到画面上；检测器模块本身不负责统计主循环帧率。

## 上传到 K230

彩色光点功能至少需要：

- `color.py`
- `config.py`

直接运行彩色光点摄像头示例还需要 `camera_io.py`。

方框功能至少需要：

- `tangle.py`
- `color.py`（仅直接运行 `tangle.py` 的组合示例需要）
- `config.py`
- `camera_io.py`（仅直接运行完整摄像头示例需要）
- `uart_io.py`（直接运行 `tangle.py` 的串口追踪示例需要）

蓝牙串口功能至少需要：

- `bluetooth_uart.py`
- `uart_io.py`
- `config.py`

四角循环应用至少需要：

- `corner_cycle.py`
- `tangle.py`
- `uart_io.py`
- `camera_io.py`
- `config.py`

细铅笔线方框功能至少需要：

- `pencil_rectangle.py`
- `config.py`

直接运行 `pencil_rectangle.py` 的四角轨迹主程序还需要：

- `corner_cycle.py`
- `tangle.py`（`corner_cycle.py` 当前使用其通用四边形绘制函数）
- `uart_io.py`
- `camera_io.py`

数字功能至少需要：

- `num.py`
- `config.py`
- 数字模板目录及 `0.png` 到 `9.png`
- `camera_io.py`（仅直接运行完整摄像头示例需要）

作为模块导入时，调用方自行负责摄像头、显示和串口生命周期。
