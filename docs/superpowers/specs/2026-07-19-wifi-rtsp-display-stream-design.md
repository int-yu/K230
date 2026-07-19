# K230 Wi-Fi RTSP 批注画面推流设计

## 目标

- CanMV-K230-LP4 V3.0 以 STA 模式连接 2.4 GHz 热点。
- 电脑通过 RTSP 查看与 IDE 最终预览一致的画面，包括检测框、文字、状态和 FPS。
- 使用 K230 的 WBC 显示回写与硬件 VENC H.264 编码，避免 Python 逐帧 JPEG 压缩。
- 默认不启用网络功能；启用后任一网络或推流步骤失败时，原有摄像头、检测、串口、拍照和 IDE 预览继续运行。
- 保留 VS Code CanMV 扩展原有的 USB IDE 帧缓冲预览，不修改扩展。

## 非目标

- 不修改 VS Code CanMV 扩展以直接播放 RTSP。
- 不发送音频。
- 不提升现有 `640x480 @ 30 FPS` 摄像头和处理分辨率。
- 不修改任何检测算法、串口协议或拍照协议。
- 不承诺零帧率损失；通过硬件编码、默认关闭和可选双路预览把额外负载限制在可控范围内。

## 方案选择

采用显示回写推流：

```text
Sensor -> RGB 检测 -> 绘制批注 -> Display/IDE 帧缓冲
                                      |
                                      +-> WBC -> VENC H.264 -> RTSP -> Wi-Fi
```

直接将 Sensor 绑定到 VENC 虽然开销更低，但编码发生在 Python 绘制之前，网络端看不到检测框和 FPS，不满足目标。修改 VS Code 扩展接收 RTSP 会扩大范围，并且扩展已有稳定的 USB Preview 通道，因此不采用。

## 文件划分

| 文件 | 改动 |
| --- | --- |
| `wifi_rtsp.py` | 新增 `WifiRtspService`，负责 WLAN STA、连接超时、WBC RTSP 启停、状态和安全清理 |
| `wifi_secrets.example.py` | 新增不含真实凭据的热点配置模板 |
| `.gitignore` | 忽略 `wifi_secrets.py`，避免热点密码进入版本库 |
| `config.py` | 新增 `WIFI_RTSP_...` 默认参数，功能默认关闭 |
| `camera_io.py` | 增加可选推流生命周期钩子；默认调用方式和行为不变 |
| `tests/test_wifi_rtsp.py` | 使用假 WLAN、时钟和 WBC 验证纯 Python 状态与失败路径 |
| `README.md` | 说明启用方式、上传文件、RTSP 地址、双路预览和性能取舍 |

检测器文件、`uart_io.py`、`bluetooth_uart.py` 和 `capture.py` 不修改业务逻辑。

## 配置

`config.py` 增加以下默认值：

- `WIFI_RTSP_ENABLED = False`：保持所有现有程序默认行为不变。
- `WIFI_RTSP_REQUIRED = False`：网络推流失败时继续运行本地功能。
- `WIFI_RTSP_CONNECT_TIMEOUT_S = 15`：热点连接最长等待时间。
- `WIFI_RTSP_PORT = 8554`：RTSP 服务端口。
- `WIFI_RTSP_SESSION = "test"`：RTSP 会话名。
- `WIFI_RTSP_WIDTH = IMAGE_WIDTH`、`WIFI_RTSP_HEIGHT = IMAGE_HEIGHT`：保持 `640x480`。
- `WIFI_RTSP_KEEP_IDE_PREVIEW = True`：调试时保留 VS Code Preview；性能优先时可关闭 USB Preview。

真实 `WIFI_SSID` 和 `WIFI_PASSWORD` 放在 `wifi_secrets.py`。仓库只提供 `wifi_secrets.example.py`，用户复制并填写后上传到 `/sdcard/K230`。

## `WifiRtspService` 边界

模块导入时不加载 `network` 或 `libs.WBCRtsp`，不连接热点，不启动线程或硬件。只有 `initialize()` 才延迟导入板端模块。

公共行为：

- `initialize(width, height)`：连接热点、取得 DHCP 地址、配置并启动 WBC RTSP，成功后返回自身。
- `deinitialize()`：停止 WBC RTSP、断开 WLAN；可重复调用，不抛出清理异常。
- `active`：只有 WLAN 和 RTSP 都启动成功时为 `True`。
- `ip_address`：成功后保存 K230 的局域网地址。
- `rtsp_url`：成功后为 `rtsp://<ip>:<port>/<session>`，未启动时为 `None`。
- `last_error`：失败时保存简短错误，便于终端诊断。

构造函数允许注入 WLAN、WBC 和时间实现，桌面测试不依赖 K230 硬件。默认路径使用板端真实模块。

## `CameraIO` 集成

现有 `CameraIO(display_target=...)` 调用保持不变。启用配置后，`CameraIO.initialize()` 在 Display、MediaManager 和 Sensor 成功启动之后创建 `WifiRtspService`。

- 推流启动成功：终端打印 IP 和 RTSP URL，主循环继续原样调用 `snapshot()`、绘图和 `show_image()`。
- 推流启动失败且 `WIFI_RTSP_REQUIRED=False`：打印一次错误，安全清理未完成的网络资源，继续现有程序。
- 推流启动失败且 `WIFI_RTSP_REQUIRED=True`：按当前 `CameraIO` 初始化异常路径释放全部资源并重新抛出，供需要强制联网的专用程序使用。
- `CameraIO.deinitialize()` 先停止 RTSP，再停止 Sensor、Display 和 MediaManager，避免 WBC 读取已经释放的显示缓冲。

由于 WBC 捕获的是 `show_image()` 的最终显示结果，检测器无需感知 RTSP，所有已经画到帧上的批注自然进入网络流。

## VS Code Preview

CanMV 扩展的 Preview 使用 USB IDE 帧缓冲，不使用 RTSP。`WIFI_RTSP_KEEP_IDE_PREVIEW=True` 时继续保留当前 `Display.VIRT(..., to_ide=True)` 行为，并同时提供 RTSP URL。

双路输出会增加内存与传输负载。性能优先部署时将 `WIFI_RTSP_KEEP_IDE_PREVIEW=False`，不再通过 USB 发送 IDE Preview，但 RTSP 仍包含同一套最终批注。

## 故障隔离

以下失败在默认配置下都不能终止原有视觉程序：

- `wifi_secrets.py` 缺失或凭据为空。
- 固件没有 `network.WLAN`、`libs.WBCRtsp` 或相应驱动。
- 连接热点超时、认证失败或 DHCP 未取得有效 IP。
- WBC 配置或 RTSP 启动失败。
- 清理阶段的 RTSP 停止、WLAN 断开失败。

失败后 `active=False`、`rtsp_url=None`，终端只打印一次可操作的错误信息。失败路径不修改检测结果、不替换摄像头帧、不改变串口发送周期。

## 性能策略

- 固定 `640x480`，不新增 720P/1080P 配置。
- 使用硬件 H.264 VENC，不使用 Python JPEG 循环。
- 不启用音频。
- 默认关闭；只有用户明确启用才占用资源。
- 保留双路调试与仅 RTSP 的性能模式。
- README 要求在同一算法上分别记录关闭和开启 RTSP 的平均 FPS；目标是平均 FPS 降幅不超过约 10%，该数值是验收目标而非芯片保证。
- 板端温升必须通过实际硬件持续运行观察；桌面测试不宣称温度结果。

## 测试与验收

桌面自动测试覆盖：

1. 模块导入不触发硬件初始化。
2. WLAN 成功连接后生成正确 IP 和 RTSP URL。
3. 热点连接超时不启动 WBC，并保存错误状态。
4. WBC 启动失败时断开 WLAN并回到非活动状态。
5. `deinitialize()` 可重复调用，单个清理异常不阻止后续清理。
6. fail-open 模式下 `CameraIO` 的原有初始化和显示路径保持可用。

板端验收：

1. K230 和电脑连接同一 2.4 GHz 热点。
2. VS Code Preview 继续显示原有带批注画面。
3. VLC 或 ffplay 打开打印出的 RTSP URL，画面内容和批注与 IDE 一致。
4. 主动填写错误密码时，RTSP 启动失败，但原检测、串口和 IDE Preview 继续工作。
5. 连续运行至少 10 分钟，记录关闭/开启 RTSP 的平均 FPS、稳定性和实际温升。

## 已知限制

- 能否同时使用 IDE VIRT 输出和 WBC RTSP 取决于板端固件是否包含对应模块及驱动；代码会检测并安全降级，但桌面环境不能代替上板验证。
- 手机热点可能启用客户端隔离，导致电脑无法访问 K230；此时改用普通路由器或 Windows 移动热点。
- 2.4 GHz 网络质量不足时，电脑端可能出现延迟和丢帧，即使 K230 本地检测 FPS 未明显下降。
