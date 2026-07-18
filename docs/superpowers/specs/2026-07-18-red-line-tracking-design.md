# K230 红线巡线模块设计

面向 2021 年全国大学生电子设计竞赛 F 题「智能送药小车」。走廊地面居中一条
`1.5~1.8cm` 红色胶带引导线，路口贴黑色数字纸标识病房号，药房和病房门口是黑白
相间虚线。本设计只负责**沿红线连续行驶所需的横向偏差**，以及路口的粗略提示。

已有的 `road.py` 负责 T/十字/END 的符号识别，本模块不替代它，也不调用它。

## 目标与非目标

目标：

- 从单帧 RGB 图像求出红线在 5 个不同前瞻距离上的横向偏差。
- 给出航向角和路口标志。
- 通过现有 `TrackingUART` 的帧格式发给 TI 板。
- 稳态处理像素量显著低于整帧检测，保证主循环帧率。

非目标：

- 不做 PID、不做速度规划，控制律留在 TI 板。
- 不做数字识别。本次只做出路口触发和状态机骨架，数字分支留接入点。
- 不保留上一帧结果做预测或时间平滑。单帧无结果就返回 `None`。

## 文件划分

| 文件 | 改动 |
| --- | --- |
| `line.py` | 新增，导出 `LineTrackDetector`；直接运行时是完整循迹主程序 |
| `config.py` | 新增 `LINE_...` 参数区 |
| `uart_io.py` | 新增 `MSG_TYPE_LINE = 0x11` 和 `send_line()`；不改握手，不改 TARGET |
| `README.md` | 新增「红线巡线模块」章节、参数前缀说明、上传清单 |

`line.py` 遵守 README「新检测模块规范」全部 10 条：导入时不启动摄像头、显示器
或串口；`detect()` 不修改输入画面；无目标返回 `None`；可复用缓冲区在构造函数中
创建一次；同时维护 `_target_valid`、`_offset_x`、`_offset_y` 三个私有属性，取最近
一条带的偏差，使本模块也能直接喂现有 TARGET 帧。

## 红色提取

采用 RGB 通道差分，不使用 HSV：

```text
红 = (R - G > LINE_RED_MIN_DIFF) 且 (R - B > LINE_RED_MIN_DIFF) 且 (R > LINE_RED_MIN_VALUE)
```

理由：

- `road.py` 用绿通道 Otsu 把红和黑**合并**为前景，那是符号识别需要的。巡线需要
  相反的能力，必须把红线与黑色数字纸、黑色墙线分开。通道差分天然做到这点：黑线
  和白地的 `R-G` 都接近 0。
- 比 `cvtColor` 转 HSV 快，且不像固定 HSV 阈值那样受色温影响。

不做形态学开闭运算。胶带反光造成的细小断点由后面 run 合并时的 `LINE_RUN_MAX_GAP`
吸收，省掉每帧的形态学开销。

## 感兴趣区域：只处理 5 条水平窄带

不对整帧做任何处理。在画面下半部由近及远取 5 条水平窄带，位置由
`LINE_BAND_RATIOS` 给出，每项为 `(y1, y2)` 占画面高度的比例。

每条带的处理顺序：

```text
按比例切片(view，不复制大图) → resize 到 160×8 → 通道差分掩膜 → 列投影 → run 提取
```

处理像素量：

```text
5 × 160 × 8 = 6400 px
对比 road.py 整帧 320×240 = 76800 px，约为 1/12
```

这是本模块帧率优势的唯一来源，实现时不得改为「先缩放整帧再取带」。

## 列投影与 run 提取

**带是水平的**：整幅宽、8 行高的横条。列投影指沿竖直方向把 8 行求和压成 1 行，
得到 160 个数，第 `i` 个数是第 `i` 列中红色像素的个数。结果按列索引，即按 x 坐标
排开。纵向 8 行不含额外信息，只用于抗噪。

```text
                 x=0                                            x=159
 一条带(8行高)  ┌──────────────────────────────────────────────────┐
                │ . . . . . . . . . ████ . . . . . . . . . . . . . │  红色掩膜
                │ . . . . . . . . . ████ . . . . . . . . . . . . . │
                └──────────────────────────────────────────────────┘
                         ↓ 沿竖直方向求和（8行 → 1行）
 列投影(160个数) [0 0 0 0 0 0 0 0 0 3 8 8 6 1 0 0 0 0 0 0 0 0 0 0 0]
                                    └──run──┘
                            run 中心 x  ← 红线在这条带的位置
                            run 宽度    ← 用于路口判定
```

实现使用 `cv2.reduce(mask, 0, cv2.REDUCE_SUM)`，其中 `0` 表示沿第 0 维（行）求和。

run 提取规则：

1. 列计数 `>= LINE_BAND_MIN_COLUMN_COUNT` 的列视为被占用。
2. 相邻被占用列之间空隙不超过 `LINE_RUN_MAX_GAP` 时合并为同一个 run。
3. 宽度小于 `LINE_RUN_MIN_WIDTH` 的 run 丢弃。
4. 宽度超过 `画面宽 × LINE_RUN_MAX_WIDTH_RATIO` 的 run 不作为主线中心，但仍参与
   路口判定。

**不使用整行质心**。十字路口的横向分支会把质心拽偏，run 不会。

多个 run 时的选择：由近及远逐带传递，第 0 条带（最近、车头正前方）选**最靠近画面
中心**的 run，其余各带选**中心最接近上一条带所选 run 中心**的那个。近带锚定主线，
横向分支抢不走远带。

每条带产出：`y`、`center_x`、`width`、`run_count`、`valid`。

## 偏差、航向与置信度

- `offset_i = 画面中心X − 带 i 的 center_x`。符号与现有全部模块一致。
- `heading`：由最近有效带与最远有效带的中心连线求 `atan2(dx, dy)`，单位 0.1 度，
  取值范围裁到 int16。有效带不足 2 条时为 0。
- `confidence`：`有效带比例 × 0.6 + 带间连续性 × 0.4`。连续性为
  `1 − min(1, 平均相邻带中心跳变 / (画面宽 × LINE_CONTINUITY_REF_RATIO))`。
  该值只用于本模块内部阈值判断，不与其他检测器横向比较。

有效带数少于 `LINE_MIN_VALID_BANDS` 时 `detect()` 返回 `None`。

## 路口判定

复用已生成的掩膜统计，不额外调用 `RoadSymbolDetector`，稳态零额外开销。

- 某带 `width > LINE_JUNCTION_WIDTH_RATIO × 各带宽度中位数` → 该带出现横向分支。
- 某带 `run_count >= 2` → 该带出现分叉。
- 满足上述任一条件的带数 `>= LINE_JUNCTION_MIN_BANDS` 时判为接近路口。

左右侧由分支 run 中心相对主线中心的偏移方向判定，偏移量需超过
`画面宽 × LINE_JUNCTION_SIDE_MIN_OFFSET_RATIO` 才置位。

`junction_flags:u8` 按位定义：

| 位 | 含义 |
| --- | --- |
| bit0 | 接近路口 |
| bit1 | 左侧存在分支 |
| bit2 | 右侧存在分支 |
| bit3 | 丢线（有效带数不足） |
| bit4~7 | 保留，固定为 0 |

该标志只负责「接近路口了，减速并准备切数字识别」，不承担精确的 T 型与十字分类。
精确分类仍应交给 `road.py`。

## 串口协议

沿用现有帧格式，走同一个 `send_frame()`，不改握手流程，不改 TARGET：

```text
AA 55 | VER | TYPE | SEQ | LEN | PAYLOAD | CRC8
```

新增 `MSG_TYPE_LINE = 0x11`，PAYLOAD 定长 14 字节：

```text
valid:u8 | b0:int16_LE | b1:int16_LE | b2:int16_LE | b3:int16_LE | b4:int16_LE
        | heading:int16_LE | junction_flags:u8
```

- `b0` 最近（画面最下、车头正前方，供 TI 板 PID 使用），`b4` 最远（前瞻）。
- 单条带无效时该字段发 0，同时 `junction_flags` 不置 bit3。
- `valid=0`（整体丢线）时全部数值字段强制发 0，`junction_flags` 置 bit3。TI 板不得
  把 `0` 解释为「红线位于画面中心」。

`send_line()` 的周期控制、`force` 参数和返回值语义与 `send_target()` 完全一致：未到
周期返回 `None`，到周期返回实际发送的 `bytes`。

保留 `MSG_TYPE_DIGIT = 0x12` 编号给后续病房号上报，本次不实现。

## 主程序状态机

`line.py` 直接运行时的主循环：

```text
TRACKING：每帧 process() → send_line()
          junction_flags bit0 连续 LINE_JUNCTION_CONFIRM_FRAMES 帧置位 → 切 DIGIT
DIGIT：   本次留 TODO 和明确接入点，不实现
```

`enable_uart=False` 时跳过 `TrackingUART` 的创建和握手，只做视觉和显示，便于单独
调参。默认 `enable_uart=True`，启动时阻塞等待 `wait_for_handshake()`。

主循环在循环外创建 `time.clock()`，每帧 `tick()`，在 `show_image()` 前把 FPS 画到
画面上，与 `road.py` 的演示保持一致。

## 绘制

`draw()` 绘制内容：

- 5 条带的边界线。
- 每条带所选 run 的中心点，有效为一种颜色，无效为另一种。
- 相邻有效带中心之间的连线，即拟合出的路线。
- 画面中心竖线，用于目测偏差。
- 文本行：`b0` 偏差、`heading`、`junction_flags`、`confidence`。

`detect()` 不得修改输入画面。

## 参数

全部进入 `config.py` 的 `LINE_...` 区，构造函数保留同名覆盖入口。

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `LINE_DETECT_WIDTH` | `160` | 每条带缩放后的宽度 |
| `LINE_BAND_DETECT_HEIGHT` | `8` | 每条带缩放后的高度 |
| `LINE_BAND_RATIOS` | 见下 | 5 条带的 `(y1, y2)` 高度比例，由近及远 |
| `LINE_RED_MIN_DIFF` | `40` | `R-G` 和 `R-B` 的最小差值 |
| `LINE_RED_MIN_VALUE` | `60` | 红色像素的最低 R 通道值 |
| `LINE_BAND_MIN_COLUMN_COUNT` | `3` | 一列至少多少红色像素才算被占用 |
| `LINE_RUN_MIN_WIDTH` | `2` | run 的最小列宽 |
| `LINE_RUN_MAX_GAP` | `2` | run 内允许合并的空列间隔 |
| `LINE_RUN_MAX_WIDTH_RATIO` | `0.55` | 超过该宽度比例的 run 不作为主线中心 |
| `LINE_MIN_VALID_BANDS` | `2` | 低于该有效带数时返回 `None` |
| `LINE_CONTINUITY_REF_RATIO` | `0.25` | 连续性评分的跳变参考宽度比例 |
| `LINE_JUNCTION_WIDTH_RATIO` | `2.2` | 判为横向分支的宽度倍数 |
| `LINE_JUNCTION_MIN_BANDS` | `2` | 判为路口所需的异常带数 |
| `LINE_JUNCTION_SIDE_MIN_OFFSET_RATIO` | `0.12` | 判定左右分支的最小偏移比例 |
| `LINE_JUNCTION_CONFIRM_FRAMES` | `3` | 主程序切数字识别的连续确认帧数 |
| `LINE_DEMO_GC_INTERVAL` | `60` | 演示主循环的垃圾回收间隔 |
| `LINE_DRAW_...` | — | 绘制颜色、线宽、点半径、字号 |

`LINE_BAND_RATIOS` 默认值：

```python
LINE_BAND_RATIOS = (
    (0.92, 1.00),
    (0.82, 0.90),
    (0.72, 0.80),
    (0.62, 0.70),
    (0.52, 0.60),
)
```

## 验证方式

1. **桌面端单元验证**：用合成图像直接调 `detect()`，覆盖直线居中、直线偏左、直线
   偏右、十字路口、完全丢线、单侧分支六种情况，断言 `offset` 符号、`junction_flags`
   位和 `None` 返回。合成图可在测试中用 numpy 直接构造，不依赖实拍。
2. **实拍回归**：本仓库目前没有红线实拍图片。上板前需要拍一批带标注的图片补上
   回归，阈值调整必须基于整批实拍验证，不允许只根据单帧改阈值。这是当前设计已知
   的缺口。
3. **板端帧率**：运行 `line.py` 演示，记录画面上的 FPS，与 `road.py` 演示对比。若
   未明显高于 `road.py`，说明 ROI 切片实现有误，需检查是否误对整帧做了缩放。
4. **协议验证**：用 `send_line()` 发固定值，确认 TI 板收到的 14 字节 PAYLOAD 与
   CRC8 校验通过。

## 已知取舍

- 路口判定走的是低成本方案，会把弯道内侧的红线加宽误判为路口。主程序用连续帧确认
  缓解，精确分类仍需 `road.py`。
- 通道差分对偏橙或偏粉的地面反光敏感，阈值需现场标定。
- 不做时间平滑，单帧丢线立即上报 `valid=0`，抗抖动责任在 TI 板。
