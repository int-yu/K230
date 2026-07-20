# K230 视觉检测模块说明

本项目把摄像头生命周期、检测参数和各类视觉算法分开。检测模块统一接收一帧 RGB 图像，不在导入时初始化摄像头、显示器或串口，因此可以在 `num.py`、`tangle.py` 或新的主程序中组合使用。

## 文件职责

| 文件 | 职责 |
| --- | --- |
| `config.py` | 摄像头、显示、串口和各检测器的默认参数 |
| `camera_io.py` | `Sensor`、`Display`、`MediaManager` 生命周期 |
| `wifi_rtsp.py` | 可选 Wi-Fi STA 与安全 H.264 RTSP 生命周期；默认关闭，失败回退原视觉流程 |
| `safe_wbc_rtsp.py` | 项目内有限超时的 WBC/VENC/RTSP 实现，避免固件示例的无限等待 |
| `local_rtsp_viewer/` | 仅本机使用的网页查看器；全局只建立一个 FFmpeg/RTSP 上游 |
| `uart_io.py` | FPIOA、UART 生命周期、二进制数据帧、双向握手和目标偏差协议；可直接运行固定信号测试 |
| `bluetooth_uart.py` | UART2 蓝牙单字符指令接收与缓存，导出 `BluetoothUART` |
| `color.py` | 彩色光点检测，导出 `ColorSpotDetector` |
| `road.py` | T/十字主连通轮廓和黑色双排虚线结束符检测，导出 `RoadSymbolDetector` |
| `line.py` | 红色引导线分带循迹和路口提示，导出 `LineTrackDetector`；直接运行时也是完整循迹程序 |
| `tangle.py` | 黑框白心方框检测，导出 `RectangleDetector`；直接运行时也是完整追踪程序 |
| `pencil_rectangle.py` | 细铅笔线方框检测，导出 `PencilRectangleDetector`；多重方框中选择估算边框最细者 |
| `corner_cycle.py` | 独立的方框四角顺时针停留、移动和串口输出应用 |
| `num.py` | 打印数字检测，导出 `DigitDetector`；直接运行时也是完整识别程序 |
| `capture.py` | 按需拍照并保存到 TF 卡，导出 `CaptureService`；直接运行时等待 CAPTURE 帧并存图 |

原 `rectangle_detector.py` 已合并进 `tangle.py`，不再需要上传。

## 统一调用形式

所有检测器都遵循同一套接口：

```python
from color import ColorSpotDetector
from road import RoadSymbolDetector
from line import LineTrackDetector
from tangle import RectangleDetector
from pencil_rectangle import PencilRectangleDetector
from num import DigitDetector

# 主循环外初始化一次。数字模板也只会在这里加载一次。
color_detector = ColorSpotDetector()
road_detector = RoadSymbolDetector()
line_detector = LineTrackDetector()
rectangle_detector = RectangleDetector()
pencil_rectangle_detector = PencilRectangleDetector()
digit_detector = DigitDetector()

# 获取 frame 后调用；process 默认会在 frame 上绘图。
spot = color_detector.process(frame)
road_result = road_detector.process(frame)
line_result = line_detector.process(frame)
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

板载屏幕和 IDE 的分辨率、位置、帧率、质量参数分别由 `config.py` 中的 `BOARD_DISPLAY_...` 和 `IDE_DISPLAY_...` 管理。它们是所有摄像头程序共享的显示配置，不再与 `tangle.py` 或 `num.py` 绑定。当前摄像头同时启用水平镜像和垂直翻转，等效于旋转 180°。

## Wi-Fi RTSP 标注画面推流

`wifi_rtsp.py` 使用 CanMV 固件内置的 `network.WLAN`、Display writeback 和 K230 硬件 H.264 VENC，把 `CameraIO.show_image()` 的最终显示画面推到局域网。检测框、文字、状态和 FPS 已经画在 Display 上，因此网页画面与原 IDE 画面内容一致并包含全部批注。

实板验证发现 VS Code IDE Preview 与 WBC RTSP 同时消费 Display writeback 会在运行约一分钟后卡住媒体管线。因此 `WIFI_RTSP_ENABLED = True` 时，默认由 `WIFI_RTSP_EXCLUSIVE_DISPLAY = True` 切换到板载 ST7701 显示路径并关闭 IDE Preview，只保留一条 WBC 消费链；RTSP 关闭或启动失败时恢复原来的 IDE Preview。这是当前固件下的稳定性保护，不是网页限制。

项目内 `safe_wbc_rtsp.py` 保持官方协议参数：

- H.264、无音频、约 2048 kbps。
- 端口 `8554`、会话名 `test`。
- 当前 Display 的实际宽高；默认 RTSP 独占模式使用板载 `800x480`，其中 `640x480` 识别画面居中显示。

### 配置热点

复制示例文件：

```text
wifi_secrets.example.py -> wifi_secrets.py
```

填写 2.4 GHz 热点：

```python
WIFI_SSID = "your-2.4g-hotspot"
WIFI_PASSWORD = "your-hotspot-password"
```

`wifi_secrets.py` 已加入 `.gitignore`，不要提交真实密码。把它和 `safe_wbc_rtsp.py`、`wifi_rtsp.py`、`camera_io.py`、`config.py` 一起上传到 `/sdcard/K230`。
在 `config.py` 中开启：

```python
WIFI_RTSP_ENABLED = True
WIFI_RTSP_REQUIRED = False
WIFI_RTSP_EXCLUSIVE_DISPLAY = True
```

现有检测程序不需要改主循环。任一程序创建并初始化 `CameraIO` 后会自动尝试连接热点；成功时终端打印类似：

```text
Wi-Fi RTSP started: rtsp://192.168.137.25:8554/test
```

电脑连接同一热点后，可以双击 `local_rtsp_viewer/start_viewer.bat`，在仅本机开放的网页中输入该地址。查看器全局只维护一个 FFmpeg 上游，重复点击不会累积 K230 连接，10 秒没有首帧会显示具体错误。

也可以在 VLC 的“打开网络串流”或 ffplay 中打开该地址：

```powershell
ffplay -fflags nobuffer -flags low_delay rtsp://192.168.137.25:8554/test
```

### 失败降级

默认 `WIFI_RTSP_REQUIRED = False`。缺少 `wifi_secrets.py`、固件媒体依赖不完整、密码错误、连接超时、DHCP 失败或 RTSP 启动失败时，终端会打印以 `Wi-Fi RTSP unavailable; continuing:` 开头的消息。若失败发生在媒体启动前，直接使用原 IDE Preview；若发生在 Display 初始化后，`CameraIO` 会释放第一次媒体资源并以原 IDE 模式重新初始化。

`safe_wbc_rtsp.py` 对 `SendFrame`、`GetStream` 和 RTSP 发送都使用有限超时，并把停止等待限制为 2 秒。如果工作线程仍未退出，它不会继续销毁 Display/MediaManager，而会报告必须断电重启，避免再次运行时访问已经释放的媒体对象。

只有专用程序明确要求“没有 RTSP 就不能运行”时才设为：

```python
WIFI_RTSP_REQUIRED = True
```

### 固件能力检查

如果终端报告缺少模块，可在板端 REPL 检查：

```python
import network
import multimedia
from media.vencoder import Encoder, ChnAttrStr, StreamData
from _media import Display
print("WLAN/RTSP/VENC available")
```

缺少任一模块时需要升级到包含无线网络、Display writeback、VENC 和 RTSP server 的 CanMV K230 固件。不要从其他固件复制 `_media`、`mpp` 或 VENC 二进制模块。

### 帧率与温升验证

RTSP 使用硬件 H.264 编码，但 WBC、内存搬运和 Wi-Fi 仍会增加负载。安全实现默认约 20 FPS 编码节奏，优先保证检测主循环和可停止性。部署前对同一个算法分别连续运行至少 10 分钟：

1. `WIFI_RTSP_ENABLED = False`，记录平均 FPS 和温度。
2. `WIFI_RTSP_ENABLED = True`，连接电脑播放器后记录平均 FPS 和温度。
3. 记录实际降幅、网页帧率和温度；不同固件、热点与算法负载会有明显差异，不再假定固定 10% 降幅。

RTSP 开启时 IDE Preview 会自动关闭，以避免同一 Display writeback 被双路消费。终端文字输出仍保留；网页看到的是同一张经过 `show_image()` 的最终标注画面。关闭 RTSP 后无需改主程序，IDE Preview 按原配置恢复。

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

## 寻路符号模块

`road.py` 只识别 T 型、十字型和黑色双排虚线结束符，不再返回普通 `line`。场地标记只有红色和黑色，因此 T/十字不再执行 HSV 红色识别：检测器对 RGB 图像的绿色通道执行 Otsu 反向二值化，把红色和黑色统一作为前景，再选择真正横跨左右、经过中心附近并向下延伸的单个主轮廓。与路线不连通的黑色数字、边框和远处物体不会参与 T/十字几何计算。

默认先把 `640×480` 输入缩放到 `320×240` 检测，所有中心、端点和线段坐标会自动映射回输入画面尺寸。模块不做骨架化，不保存上一帧结果。

```python
from road import RoadSymbolDetector

# 主循环外初始化一次。
road_detector = RoadSymbolDetector()

# frame 为 RGB 图像；默认同时绘制识别结果。
result = road_detector.process(frame)

if result is not None:
    symbol = result["symbol"]
    confidence = result["confidence"]
```

检测顺序为：

```text
T/十字主连通轮廓 → 黑色 END → None
```

- `t`：主轮廓横跨左右、向下延伸，但没有到达上方区域；返回 `left/right/down`。
- `cross`：同一个主轮廓横跨左右并同时延伸到上、下方；返回 `up/down/left/right`。
- `end`：没有 T/十字主轮廓时才生成 RGB 黑色掩膜；至少三块矩形黑块组成一行，两行满足间距和横向重合要求。返回两条 `dash_lines`，并在能够提取中央路径时返回 `near/terminal`。

T 和十字只使用选中主轮廓内部的像素拟合水平、垂直中心线。`intersection` 是两条中心线的交点，`segments` 从交点连接到各方向最远可见端点。黑色背景物体即使落入五个方向区域，只要没有和路线形成同一个主轮廓，就不会把 T 误判成十字。结束符只绘制两条黑块行中心线、`END` 标签和可用的中央路径，不逐块绘制外接框。

统一返回字段：

| 字段 | 含义 |
| --- | --- |
| `symbol` | `t`、`cross` 或 `end` |
| `center_x`, `center_y` | T/十字为交点；结束符为两排虚线整体中心 |
| `confidence` | 当前符号的结构评分，范围为 `0..1`，不是概率，也不能与其他检测器横向比较 |
| `intersection` | T/十字交点；其他状态为 `None` |
| `endpoints` | 各状态对应的方向端点字典 |
| `segments` | 需要绘制的中心线段，每段为 `(起点, 终点)` |
| `dash_lines` | `end` 的上、下两条黑块行中心线；其他状态为空元组 |
| `arm_scores` | 选中主轮廓在 `up/down/left/right/center` 五区的占用率；END 固定为 0 |
| `foreground_threshold` | 当前帧绿色通道 Otsu 自动阈值，便于上板诊断光照 |

需要自己控制绘制顺序或验证检测不修改画面时：

```python
result = road_detector.process(frame, draw=False)
if result is not None:
    road_detector.draw(frame, result)
```

主要调参项：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `ROAD_DETECT_WIDTH/HEIGHT` | `320/240` | 内部检测分辨率；结果自动映射回输入画面 |
| `ROAD_FOREGROUND_MORPH_KERNEL_SIZE` | `5` | 红黑统一前景的闭运算核边长 |
| `ROAD_ROUTE_MIN_AREA_RATIO` | `0.002` | T/十字主轮廓最小面积比例 |
| `ROAD_CROSS_TOP_MAX_RATIO` | `0.32` | 主轮廓顶部进入该高度比例以内时判为十字 |
| `ROAD_BLACK_MAX_VALUE` | `90` | RGB 三通道都不超过该值时视为结束符黑块 |
| `ROAD_DASH_MIN_COUNT_PER_ROW` | `3` | 每排结束虚线至少需要的黑块数 |
| `ROAD_DASH_MIN_ROW_SEPARATION_RATIO` | `0.07` | 两排中心的最小纵向间距，占画面高度比例 |
| `ROAD_DASH_MAX_ROW_SEPARATION_RATIO` | `0.30` | 两排中心的最大纵向间距，占画面高度比例 |
| `ROAD_DASH_MIN_HORIZONTAL_OVERLAP` | `0.55` | 两排横向覆盖相对较短一排的最低重合率 |
| `ROAD_PATH_CORRIDOR_HALF_WIDTH_RATIO` | `0.12` | END 中央路径搜索走廊半宽，占画面宽度比例 |
| `ROAD_END_MIN_PATH_LENGTH_RATIO` | `0.12` | END 返回 `near/terminal` 所需的最小向下路径长度 |

可直接运行 `road.py` 调用 `run_road_demo()`。演示使用 `CameraIO(display_target=DISPLAY_TARGET_IDE)`，并在送入 IDE 显示前绘制当前主循环 FPS。

当前实拍目录回归结果：T 全部文件 `50/50`、按内容去重 `17/17`；十字全部文件 `14/14`、去重 `11/11`；结束符全部文件 `5/5`、去重 `4/4`。全部回归均在默认 `320×240` 内部检测分辨率下完成。

## 红线巡线模块

`line.py` 面向 2021 年电赛 F 题的红色引导线。它把画面下部切成 5 条水平带，每条带
内求出红线的横向位置，等效于一列沿前进方向排开的虚拟灰度传感器：最近一条带给
单片机做 PID，最远一条带提供弯道和路口的前瞻。

它与 `road.py` 分工不同，不能互相替代：`road.py` 用绿色通道 Otsu 把红色和黑色
**合并**为前景，用于识别 T/十字/END 的整体结构；巡线必须把红线和黑色数字纸、黑色
墙线**分开**，因此使用 RGB 通道差分。

```python
from line import LineTrackDetector

# 主循环外初始化一次。
line_detector = LineTrackDetector()

# frame 为 RGB 图像；默认同时绘制识别结果。
result = line_detector.process(frame)

if result is not None:
    near_offset = result["offsets"][0]
    junction = result["junction"]
```

红色判据为三个通道的关系，不做 HSV 转换：

```text
红 = (R - G > LINE_RED_MIN_DIFF) 且 (R - B > LINE_RED_MIN_DIFF) 且 (R > LINE_RED_MIN_VALUE)
```

黑线和白布的 `R-G` 都接近 0，只有红色的差值很大。默认阈值由 `pic/line` 的 29 张
实拍图标定：红线像素的 `R-G` 第 1 百分位为 61，非红像素的第 99 百分位为 0，
阈值 50 落在这段空白区间中间。

检测流程为：

```text
裁剪画面下部 → 缩放一次到 160x60 → 通道差分掩膜 → 逐带列投影 → run 提取 → 路口判定
```

「列投影」指沿竖直方向把一条带的 12 行求和压成 1 行，得到 160 个数，第 `i` 个数
是第 `i` 列中红色像素的个数，索引即 x 坐标。取连续列段（run）的中心作为红线位置，
不使用整行质心：路口横线会把质心拽偏，run 不会。

扫描使用两个阈值。主线基本竖直，会填满整条带的 12 行，列计数接近 12；路口横线只
占带内很少几行，列计数只有 3~5。因此高阈值提取的 run 不会把横线并进主线，主线
中心在路口不会被带偏；低阈值只用来测量红色向左右延伸到哪里，供分支判定使用。

多个 run 时，最近一条带选最靠近画面中心的，其余各带选中心最接近上一条带的那个。
近带锚定主线，横向分支抢不走远带。

统一返回字段：

| 字段 | 含义 |
| --- | --- |
| `center_x`, `center_y` | 最近一条有效带的红线中心 |
| `offsets` | 5 条带的偏差元组，`画面中心X - 带中心X`，单位为原图像素 |
| `band_valid` | 5 条带各自是否检测到主线 |
| `valid_band_count` | 有效带数量 |
| `bands` | 每条带的完整信息，含 `runs`、`mass_ratio`、`edge_first/edge_last` |
| `confidence` | 有效带比例与带间连续性的加权分数，范围 `0..1`，不能与其他检测器横向比较 |
| `mass_ratio` | 逐带红色超量比的最大值，路口判定的主判据 |
| `junction_flags` | 路口状态位 |
| `junction` | `junction_flags` 的 bit0 是否置位 |
| `junction_band` | 超量比最大的带序号，可粗略反映路口距离 |
| `roi_top`, `roi_bottom` | 本帧检测区域在原图中的上下边界 |

`junction_flags` 的位定义：

| 位 | 含义 |
| --- | --- |
| bit0 | 接近路口 |
| bit1 | 左侧存在分支 |
| bit2 | 右侧存在分支 |
| bit3 | 丢线 |

路口判定不额外调用 `RoadSymbolDetector`，只复用已有的列投影统计：

```text
预期红量 = 各带主线 run 宽度的中位数 x 带高
逐带超量比 = 该带红色像素数 / 预期红量
```

比值必须**逐带取最大**，不能把 5 条带加总——横线通常只落在一到两条带里，求和会
把它稀释掉。该判据不受横线倾斜影响（横线无论多斜都要横穿整个视野，斜只是把红色
像素重新分配到不同的带），也不会把弯道误判成路口（弯道时主线自身变宽，分子分母
一起变大，比值仍在 1 附近）。

统计异常带时**不要求该带有主线**：T 型路口横线所在的那条带主线已经到头，带内只剩
横穿的红色，这正是最典型的路口带。左右分支要在所有超量带上累计，因为倾斜的横线
会被带边界切成两段落进相邻两条带。

主要调参项：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `LINE_ROI_TOP_RATIO` | `0.32` | 检测区域上沿占画面高度比例 |
| `LINE_DETECT_WIDTH` | `160` | 检测区域缩放后的宽度 |
| `LINE_BAND_COUNT` | `5` | 水平带数量，带间无空隙 |
| `LINE_BAND_HEIGHT` | `12` | 每条带缩放后的行数 |
| `LINE_RED_MIN_DIFF` | `50` | `R-G` 和 `R-B` 的最小差值 |
| `LINE_RED_MIN_VALUE` | `70` | 红色像素的最低 R 通道值 |
| `LINE_MAIN_MIN_COLUMN_COUNT` | `8` | 提取主线 run 的列计数阈值 |
| `LINE_EDGE_MIN_COLUMN_COUNT` | `2` | 测量红色横向范围的列计数阈值 |
| `LINE_RUN_MAX_GAP` | `2` | run 内允许合并的空列间隔 |
| `LINE_MIN_VALID_BANDS` | `2` | 低于该有效带数时返回 `None` |
| `LINE_JUNCTION_MASS_RATIO` | `1.50` | 判为路口的红色超量比 |
| `LINE_JUNCTION_CONFIRM_FRAMES` | `3` | 主程序切数字识别的连续确认帧数 |
| `LINE_DRAW_DATA_OVERLAY` | `True` | 画面左上角的数据块，上车应关 |
| `LINE_DRAW_BAND_LABELS` | `True` | 每条带旁边的偏差标注，上车应关 |
| `LINE_DEMO_PRINT_INTERVAL` | `30` | 终端打印数据的帧间隔，`0` 为关闭 |

`LINE_JUNCTION_MASS_RATIO` 的默认值来自实拍标定，三类场景完全分开：直道
`1.00~1.22`、终点 `1.05~1.35`、路口 `1.77~6.06`，阈值 `1.50` 落在空隙中间。在
29 张实拍图上路口召回 `25/25`，终点误报 `0/4`，直道误报 `0/29`。

### 缩放方式不要改回 INTER_AREA

`detect()` 使用 `INTER_LINEAR`。`INTER_AREA` 的开销由**源像素数**决定，它会把整个
检测区域读一遍做面积平均，实测比 `INTER_LINEAR` 慢 15 倍以上，会把只取画面下部
省下的时间全部抵消掉。红线有 50 像素以上宽，通道差分阈值又极具选择性，缩放时的
抗锯齿没有实际价值。改动这一行前请先测帧率。

同理，列投影结果在遍历前会先 `tolist()`。逐个索引数组元素每次都要装箱，实测比先
转列表慢约 3 倍，在解释执行的板端差距只会更大。

### 列投影不要改用 cv2.reduce

CanMV 固件上的精简版 OpenCV **没有 `cv2.reduce`**，用了会在上板运行时抛
`AttributeError: 'module' object has no attribute 'reduce'`。列投影使用
`np.sum(band_mask, axis=0)`，它在桌面上也比 `cv2.reduce` 更快。构造函数会探测一次
当前固件是否支持 `axis` 参数并缓存结果，不支持时退回逐行相加，两条路径结果一致。

板端使用精简版 OpenCV，新增代码前应确认所用函数在其他模块中已经出现过。当前已经
在板上验证可用的有：`resize`、`split`、`subtract`、`threshold`、`bitwise_and`、
`countNonZero`、`line`、`circle`、`putText`、`inRange`、`morphologyEx`、
`findContours`、`contourArea`、`boundingRect`、`moments`、`drawContours`、
`rectangle`、`getStructuringElement`。

### 直接运行

```python
import line

line.run_line_demo()
```

默认在 IDE 画面上叠加本帧的全部关键数据：

```text
b0..b4  +176  +174  +170  +166  +164      <- 五条带的偏差，无效带显示 --
mass 3.30 @b4   conf 0.99   bands 5/5     <- 超量比及其所在带、置信度、有效带数
flags 07  JUNC L R                        <- 路口标志位及其展开文本
```

同时在每条带的中心点旁边标出该带自己的偏差（如 `b2 +170`），有效带用绿色、无效带
用红色，一眼看出哪条带偏多少、哪条带丢了线。贴近画面右边界时标注自动改到点的左侧。
丢线帧显示 `LINE LOST  valid=0`。

判为路口时，会用洋红色在**横线所在的那条带**上画出红色的实际横向范围，两端各加一段
竖线标出延伸到哪里，并标注 `JUNCTION b4`。这一层属于几何层，**不受 `draw_data`
控制**：路口是本模块的主要输出之一，上车后仍然需要一眼确认它标在画面的哪个位置。

终端每 `LINE_DEMO_PRINT_INTERVAL` 帧打印同一份数据，格式由 `format_result()` 生成，
画面和终端共用：

```text
b[+176 +174 +170 +166 +164]  mass 3.30@b4  conf 0.99  bands 5/5  JUNC L R
```

**实际上车应该关掉叠加层。** 它每帧要多做八次 `putText`，实测占检测加绘制总耗时的
约 30%（0.060 ms / 0.198 ms）：

```python
line.run_line_demo(draw_data=False)      # 关掉画面数据和带标注
line.run_line_demo(print_interval=0)     # 关掉终端打印
```

作为模块使用时通过构造函数控制，两个开关互相独立：

```python
line_detector = LineTrackDetector(draw_data=False, draw_band_labels=False)
```

`enable_uart` 默认为 `False`，只做视觉和显示，方便单独调参。需要发送 LINE 帧时
显式开启：

```python
line.run_line_demo(enable_uart=True)
```

开启后会先创建 `TrackingUART` 并阻塞等待握手。**该等待没有超时**：单片机没接或
没在跑时，程序会停在握手上，画面不会出现。默认关闭就是为了避免把这种情况误认为
程序崩溃。

主循环用 `JunctionConfirmState` 做连续帧确认，`junction` 连续 `3` 帧成立后进入切换
数字识别的分支。该分支目前是 `TODO`，只在画面上显示 `JUNCTION`，接入 `num.py` 的
位置已经标出。`JunctionConfirmState` 只服务于状态切换，不属于检测器，也不会把历史
结果当成当前帧的检测结果发送。

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

串口发送使用统一 TARGET 二进制帧，PAYLOAD 为：

```text
valid:u8 | x:int16_LE | y:int16_LE
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

模板加载和全图粗检测使用相同的基础预处理顺序：

```text
灰度化 → 高斯模糊 → Otsu 二值化 → 闭运算 → 前景裁剪 → 等比例归一化
```

两条路径共用 `DIGIT_BLUR_KERNEL_SIZE` 和 `DIGIT_MORPH_KERNEL_SIZE`。修改模糊或形态学参数时，模板会在 `DigitDetector` 初始化阶段按新参数重新预处理，避免模板笔画粗细与现场数字不一致。原始模板文件仍应使用白底黑字图片。

粗候选产生后，检测器只在候选 ROI 内使用 `DIGIT_LOCAL_BLUR_KERNEL_SIZE` 和 `DIGIT_LOCAL_MORPH_KERNEL_SIZE` 再做一次局部 Otsu。较大的局部闭运算核用于修复反光造成的细小断笔，不会对整帧重复执行。分类顺序如下：

1. 高相关性的清晰数字优先采用模板结果。
2. 透视压缩明显时，使用轮廓形状距离辅助分类。
3. 使用内部孔洞数量和位置区分 `0/4/6/8/9`，并兼容裂纹破坏下半孔的 `8`。
4. 最后只保留中心高度和字符高度一致的主要数字行，排除 FPS 文字、十字线和小污迹。

与实拍环境关系最大的参数是 `DIGIT_MAX_ASPECT_RATIO`、`DIGIT_MIN_FILL_RATIO`、`DIGIT_LINE_MIN_HEIGHT_RATIO`、`DIGIT_SHAPE_MAX_DISTANCE` 和 `DIGIT_BROKEN_EIGHT_MIN_ASPECT_RATIO`。调整前应使用带正确标签的实拍图做整批回归，不建议只根据单帧降低 `DIGIT_MATCH_THRESHOLD`。

整串返回字段：

| 字段 | 含义 |
| --- | --- |
| `text` | 从左到右的识别文本，低于阈值的位用 `?` 表示 |
| `digits` | 每一位数字的结果字典列表 |
| `count` | 数字候选数量 |
| `recognized_count` | 达到匹配阈值的数量 |
| `confidence` | 各候选识别分数截取到 0..1 后的平均值；不是概率 |
| `center_x`, `center_y`, `bbox` | 整串数字的中心和外接矩形 |

每个 `digits` 元素包含：

- `value`：识别成功时为 `0..9`，未达到阈值时为 `-1`。
- `text`：数字字符或 `?`。
- `recognized`：是否达到匹配阈值。
- `confidence`：模板相关性和轮廓形状分数中的较强证据，不是概率。
- `hole_count`：局部修复后检测到的有效内部孔洞数量。
- `shape_distance`：与最终数字模板轮廓的形状距离，越小越相似；不支持形状匹配时为 `None`。
- `center_x`、`center_y`、`x`、`y`、`w`、`h`、`bbox`、`area`：位置和轮廓信息。

## 拍照存图模块

`capture.py` 导出 `CaptureService`，收到 MSPM0 发来的 CAPTURE 帧（`TYPE=0x20`）后，在主循环的下一帧把当前画面保存到 TF 卡，然后回一个 `CAPTURE_ACK` 帧（`TYPE=0x21`）。照片不通过串口回传，事后拔卡拷贝。

### 固件限制与存图实现（上板实测结论，勿改）

在真实 K230 板子上以 rgb888（640x480）格式实测，`image.save()` 的两种调用方式均失败：

```
image.save(path, quality=95)  -> OSError: current format not support save function!
image.save(path)              -> OSError: current format not support save function!
```

**该固件的 `image.save()` 不支持 rgb888 格式**，无论是否传 quality 参数，结果一致。唯一可用路径是 `image.compressed(quality=...)` 返回 JPEG 字节后自行写文件，实测返回正常 JPEG 数据（54969 字节）。

因此 `save()` 的实现为：先调 `image.compressed(quality=...)` 取字节，再用 `open(path, "wb")` 写文件。部分固件的 `compressed()` 不接受 quality 关键字时退回无参调用。

**失败时的行为**：若 `compressed()` 抛异常或写文件失败，会尝试 `os.remove(path)` 删除可能残留的半截文件（`open()` 成功但 `write()` 中途失败时会产生半截文件，不删则下次扫描会把它计入编号，留下打不开的坏图），然后返回 `None`，并且不推进 `_next_index`，让下一次尝试重用同一个编号。

**不要把实现改回 `image.save()`**——该固件对 rgb888 图像调用 `image.save()` 必然报 `current format not support save function!`，这是上板实测得出的结论，不是推测。

### 公共方法

| 方法 | 说明 |
| --- | --- |
| `__init__(save_dir, prefix, suffix, quality, max_pending)` | 构造并扫描已有文件，确定下一个编号，不启动任何硬件 |
| `handle_frames(frames)` | 从 `poll()` 的返回值中过滤 `0x20` 帧，累加待拍张数，返回本次新增张数 |
| `update(image)` | 主循环每帧调用；有待拍时保存一张，返回 `(saved_count, last_index)`；无待拍返回 `(0, 0)` |
| `save(image)` | 保存一张到 `save_dir/prefix_NNNN.suffix`，返回本张编号；失败返回 `None` |

### 属性

| 属性 | 说明 |
| --- | --- |
| `pending` | 当前待拍张数（只读） |
| `next_index` | 下一张照片的编号（只读） |

### 与 `poll()` 配合的用法示例

```python
from capture import CaptureService
from uart_io import TrackingUART
from camera_io import CameraIO, DISPLAY_TARGET_IDE
from config import CAPTURE_MESSAGE_ACK

service = CaptureService()
camera = CameraIO(display_target=DISPLAY_TARGET_IDE).initialize()
tracking_uart = TrackingUART().initialize()
tracking_uart.wait_for_handshake()

while True:
    image = camera.snapshot()
    frames = tracking_uart.poll()
    service.handle_frames(frames)
    saved, last_index = service.update(image)
    if saved > 0:
        tracking_uart.send_frame(
            CAPTURE_MESSAGE_ACK,
            bytes((1, last_index & 0xFF, (last_index >> 8) & 0xFF)),
        )
    camera.show_image(image)
```

### 冷启动预热（上板实测结论，勿省）

sensor 初始化完成后，自动曝光算法尚未收敛，最初若干帧是全黑图像。实测初始化后立刻调用 `snapshot()` 保存，得到的是全黑文件；预热 30 帧后保存，颜色正常。

`run_capture_demo()` 在握手之前先空跑 `CAPTURE_WARMUP_FRAMES` 帧：

```python
for _ in range(warmup_frames):
    camera.snapshot()
```

**预热放在握手前**：握手是 K230 向 MSPM0 宣告"我已就绪"的信号。若握手在预热后发出，MSPM0 收到 READY_ACK 时 sensor 已经收敛，第一张 CAPTURE 请求不会拍到黑图。若颠倒顺序，MSPM0 可能在 sensor 尚未收敛时立即发出 CAPTURE，导致存下黑图。

**不要删掉这段预热**——这是 K230 sensor 的固有行为，不是代码冗余。

### `to_numpy_ref()` 通道顺序（上板实测结论，勿改）

`image.to_numpy_ref()` 返回的 numpy 数组是 **RGB** 通道顺序，而 `cv2.imwrite` 期望 **BGR**。若直接将该数组传给 `cv2.imwrite`，存出来的图像红蓝互换。如需走 cv2 写图，必须先 `cvtColor(frame, cv2.COLOR_RGB2BGR)`。存图路径（`CaptureService.save()`）已改用 `image.compressed()` 绕过此问题，无需 `to_numpy_ref()`。

### `CAPTURE_` 参数表

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `CAPTURE_SAVE_DIR` | `/sdcard/pic` | TF 卡照片保存目录 |
| `CAPTURE_FILE_PREFIX` | `cap` | 文件名前缀 |
| `CAPTURE_FILE_SUFFIX` | `.jpg` | 文件扩展名 |
| `CAPTURE_JPEG_QUALITY` | `95` | JPEG 保存质量（0–100） |
| `CAPTURE_MAX_PENDING` | `20` | 单次最多累积的待拍张数上限 |
| `CAPTURE_MESSAGE_REQUEST` | `0x20` | MSPM0 发来的拍照请求帧 TYPE |
| `CAPTURE_MESSAGE_ACK` | `0x21` | K230 回复的拍照确认帧 TYPE |
| `CAPTURE_WARMUP_FRAMES` | `30` | 初始化后空跑帧数，等待自动曝光收敛（实测前若干帧全黑） |

## 串口模块

`uart_io.py` 导出 `TrackingUART`。默认配置是：

- UART1
- TX：GPIO3
- RX：GPIO4
- 波特率：115200
- 周期发送最小间隔：10 ms
- READY 重发周期：100 ms
- 启动握手轮询间隔：10 ms
- 8 数据位、无校验、1 停止位

接线时，K230 的 TX（GPIO3）连接单片机 PA22（UART2 RX），K230 的 RX（GPIO4）连接单片机 PA21（UART2 TX），两块板还需要共地。当前协议需要双向握手，TX、RX 都必须连接。

最少调用方式：

```python
from tangle import RectangleDetector
from uart_io import TrackingUART

# 主循环外初始化一次，并等待双方完成 READY/READY_ACK。
rectangle_detector = RectangleDetector()
tracking_uart = TrackingUART(send_period_ms=10).initialize()
tracking_uart.wait_for_handshake()

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
实际数据帧 `bytes`。需要忽略周期立即发送时传入 `force=True`。

```python
packet = tracking_uart.send_target(
    rectangle_detector._target_valid,
    rectangle_detector._offset_x,
    rectangle_detector._offset_y,
)

if packet is not None:
    print("本次发送字节数：{}".format(len(packet)))
```

需要注意：10 ms 对应最高 100 次/秒。当前摄像头配置为 30 FPS，相邻画面
约 33 ms，因此每帧调用时仍然会每帧发送。如果目的是降低 30 FPS 主循环的
发送次数，应把周期设为大于约 33 ms，例如 50 ms 或 100 ms。

所有消息使用同一种二进制帧：

```text
AA 55 | VER | TYPE | SEQ | LEN | PAYLOAD | CRC8
```

| 字段 | 长度 | 含义 |
| --- | ---: | --- |
| `AA 55` | 2 | 固定帧头，用于丢字节后重新同步 |
| `VER` | 1 | 协议版本，当前为 `0x01` |
| `TYPE` | 1 | `READY=0x01`、`READY_ACK=0x02`、`TARGET=0x10`、`LINE=0x11` |
| `SEQ` | 1 | 帧序号，达到 255 后回到 0 |
| `LEN` | 1 | PAYLOAD 长度，当前最大 32 |
| `PAYLOAD` | LEN | 消息数据 |
| `CRC8` | 1 | CRC-8/ATM，多项式 `0x07`，校验范围为 VER 到 PAYLOAD |

TARGET 的 PAYLOAD 固定为 5 字节：

```text
valid:u8 | offset_x:int16_LE | offset_y:int16_LE
```

当 `valid=0` 时，模块会强制把 `x`、`y` 发送为 0。此时的 `0,0` 只是无效包占位值，单片机不能把它解释为“目标位于画面中心”。需要显式使用主程序帧号时：

```python
tracking_uart.send_target(
    rectangle_detector._target_valid,
    rectangle_detector._offset_x,
    rectangle_detector._offset_y,
    frame_id=frame_count,
)
```

巡线使用独立的 LINE 帧，`TYPE` 为 `0x11`，PAYLOAD 定长 12 字节，整帧 19 字节：

```text
valid:u8 | b0:int16_LE | b1:int16_LE | b2:int16_LE | b3:int16_LE | b4:int16_LE
        | junction_flags:u8
```

`b0` 最近、`b4` 最远，单位为原图像素，符号与 TARGET 一致。直接把检测器本帧的返回
值传进去，不需要自己拆字段：

```python
from line import LineTrackDetector
from uart_io import TrackingUART

line_detector = LineTrackDetector()
tracking_uart = TrackingUART().initialize()
tracking_uart.wait_for_handshake()

result = line_detector.process(frame)
tracking_uart.send_line(result)
```

`result` 为 `None` 时会发送 `valid=0`、五个偏差全 0、`junction_flags` 置 bit3 的
丢线帧。和 TARGET 一样，这里的 `0` 只是无效占位值，单片机必须走丢线保护逻辑，
不能把它当成「红线位于画面中心」喂给 PID。

单片机侧只需要在现有解析循环里加一个分支，握手和 TARGET 都不用改：

```c
case 0x11:                                          /* LINE */
    valid = payload[0];
    for (int i = 0; i < 5; i++) {
        offset[i] = (int16_t)(payload[1 + i * 2] |
                              (payload[2 + i * 2] << 8));
    }
    junction_flags = payload[11];
    break;
```

`UART_MESSAGE_DIGIT = 0x12` 已经预留给后续的病房号上报，当前未实现。

临时修改串口参数不需要改 `config.py`：

```python
tracking_uart = TrackingUART(
    uart_id=1,
    tx_pin=3,
    rx_pin=4,
    baudrate=230400,
    send_period_ms=50,
    handshake_period_ms=100,
).initialize()
```

`wait_for_handshake()` 只在程序启动阶段使用，内部调用 `update_handshake()`，周期发送 READY 并处理 READY/READY_ACK。只有既收到对方 READY、又收到对方对本机 READY 的 ACK，`handshake_complete` 才为真。双方都必须主动发送 READY，不能采用“先等待接收、收到后才发送”的流程，否则会互相死锁。该方法默认一直等待，不设置超时。

握手完成后，主循环继续调用 `send_target()` 即可。该方法会先处理接收缓冲区；如果对端因首个 READY_ACK 丢失而继续重发 READY，K230 会再次回复 READY_ACK，避免出现 K230 已完成而对端仍停在 WAIT 的单边握手状态。完整的单边复位重连暂不在当前协议范围内。

直接在 K230 上运行 `uart_io.py` 会执行通信测试：握手成功前保持等待，成功后每 10 ms 尝试发送固定 `valid=1、x=123、y=-45` TARGET 帧。天猛星 OLED 最后一行应显示：

```text
K:1 X:+0123 Y:-0045
```

除目标协议外，模块也提供 `send_frame()`、`poll()`、`write()`、`any()` 和 `read()`，供自定义双向协议使用。标准目标发送程序的业务循环不需要单独调用 `poll()`，因为 `send_target()` 已经在发送前处理接收数据：

```python
tracking_uart.send_frame(0x20, b"custom payload")
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

## 板端模块导入路径

CanMV 按绝对路径启动脚本时，不会把脚本所在目录加入 `sys.path`，`import config`
会失败。所有导入 `config` 的模块都在导入前补上了板端目录：

```python
import sys

if "/sdcard/K230" not in sys.path:
    sys.path.append("/sdcard/K230")
```

这段必须放在 `from config import ...` **之前**，路径先就位才能导入。重复导入模块
不会重复追加。模块实际存放位置不是 `/sdcard/K230` 时，需要同步修改这个字面量。

## 参数管理规则

模块默认参数统一放在 `config.py`，分为公共参数和程序特有参数。

公共参数由多个程序共同使用：

- `CAMERA_...`、`IMAGE_...`：摄像头、处理分辨率和画面中心。
- `BOARD_DISPLAY_...`、`IDE_DISPLAY_...`：板载屏幕与 CanMV IDE 显示配置。
- `UART_...`：所有目标追踪程序共用的串口、发送周期和启动握手参数。

程序特有参数按模块前缀分区：

- `COLOR_...`：彩色光点。
- `ROAD_...`：T/十字主轮廓和黑色双排虚线结束符。
- `LINE_...`：红线分带循迹和路口判定。
- `RECTANGLE_...`：方框。
- `TANGLE_...`：方框追踪演示的打印、回收和绘制参数。
- `CORNER_CYCLE_...`：四角轨迹的停留、移动、串口周期和绘制参数。
- `PENCIL_RECTANGLE_...`：细铅笔线方框。
- `DIGIT_...`：数字。
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

Wi-Fi RTSP 标注画面推流还需要：

- `wifi_rtsp.py`
- `safe_wbc_rtsp.py`
- `wifi_secrets.py`（由示例复制并填写，不提交 Git）
- `camera_io.py`
- `config.py`

彩色光点功能至少需要：

- `color.py`
- `config.py`

直接运行彩色光点摄像头示例还需要 `camera_io.py`。

寻路符号功能至少需要：

- `road.py`
- `config.py`

直接运行 `road.py` 的摄像头演示还需要 `camera_io.py`。

红线巡线功能至少需要：

- `line.py`
- `config.py`

直接运行 `line.py` 的循迹主程序还需要 `camera_io.py` 和 `uart_io.py`；
传入 `enable_uart=False` 时不需要 `uart_io.py`。

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

拍照存图功能至少需要：

- `capture.py`
- `config.py`

直接运行 `capture.py` 的存图主程序还需要 `camera_io.py` 和 `uart_io.py`；
传入 `enable_uart=False` 时不需要 `uart_io.py`。

作为模块导入时，调用方自行负责摄像头、显示和串口生命周期。
