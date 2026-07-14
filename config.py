"""K230 摄像头、显示和检测参数。"""


# ============================================================
# 共享摄像头参数
# ============================================================

CAMERA_ID = 2
CAMERA_SOURCE_WIDTH = 1280
CAMERA_SOURCE_HEIGHT = 960
CAMERA_FPS = 30
CAMERA_PIXEL_FORMAT = "RGB888"

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# 当前画面需要同时水平镜像和垂直翻转，等效于旋转 180°。
CAMERA_HMIRROR = True
CAMERA_VFLIP = True


# ============================================================
# 方框追踪串口参数（tangle.py）
# ============================================================

# 01Studio CanMV K230：UART1 TX=GPIO3，RX=GPIO4。
TRACK_UART_ID = 1
TRACK_UART_TX_PIN = 3
TRACK_UART_RX_PIN = 4
TRACK_UART_BAUDRATE = 115200


# ============================================================
# 显示模式
# ============================================================

DISPLAY_MODE_ST7701 = "st7701"
DISPLAY_MODE_VIRT = "virt"


# tangle.py：板载 3.5 寸 ST7701 屏幕
TANGLE_DISPLAY_MODE = DISPLAY_MODE_ST7701
TANGLE_DISPLAY_WIDTH = 800
TANGLE_DISPLAY_HEIGHT = 480
TANGLE_DISPLAY_FPS = 30
TANGLE_DISPLAY_TO_IDE = False
TANGLE_DISPLAY_X = (TANGLE_DISPLAY_WIDTH - IMAGE_WIDTH) // 2
TANGLE_DISPLAY_Y = (TANGLE_DISPLAY_HEIGHT - IMAGE_HEIGHT) // 2


# num.py：CanMV IDE 虚拟显示
NUM_DISPLAY_MODE = DISPLAY_MODE_VIRT
NUM_DISPLAY_WIDTH = IMAGE_WIDTH
NUM_DISPLAY_HEIGHT = IMAGE_HEIGHT
NUM_DISPLAY_FPS = 30
NUM_DISPLAY_TO_IDE = True
NUM_DISPLAY_QUALITY = 80
NUM_DISPLAY_X = 0
NUM_DISPLAY_Y = 0


# ============================================================
# 矩形检测参数（tangle.py）
# ============================================================

# 在较低分辨率上检测，最终坐标自动映射回 IMAGE_WIDTH × IMAGE_HEIGHT。
RECTANGLE_DETECT_WIDTH = 320
RECTANGLE_DETECT_HEIGHT = 240

# 0 表示使用 Otsu 自动阈值；非 0 时使用给定灰度阈值。
RECTANGLE_BINARY_THRESHOLD = 0
RECTANGLE_USE_OTSU = True

# 是否用 3×3 闭运算连接轻微断裂的黑框。完整黑框建议保持 False 以提高帧率。
RECTANGLE_USE_MORPH_CLOSE = False

RECTANGLE_MIN_AREA = 1500
RECTANGLE_APPROX_RATIO = 0.025
RECTANGLE_MIN_WIDTH = 30
RECTANGLE_MIN_HEIGHT = 30
RECTANGLE_MAX_COUNT = 12
RECTANGLE_MIN_CONFIDENCE = 0.55

# 白色内部面积相对于黑框外轮廓面积的允许范围。
RECTANGLE_MIN_INNER_AREA_RATIO = 0.50
RECTANGLE_MAX_INNER_AREA_RATIO = 0.92

# 内外轮廓中心最大偏移量，相对于外框对角线长度。
RECTANGLE_MAX_CENTER_OFFSET_RATIO = 0.12

# 四侧边框宽度差异上限：(最大宽度 - 最小宽度) / 平均宽度。
RECTANGLE_MAX_BORDER_ASYMMETRY = 1.50

# 连续多少帧未检测到目标后，才判定目标丢失。
RECTANGLE_LOST_FRAME_LIMIT = 5


# ============================================================
# 数字识别参数（num.py）
# ============================================================

DIGIT_TEMPLATE_DIR = "/digit_templates"
DIGIT_TEMPLATE_DIR_CANDIDATES = (
    DIGIT_TEMPLATE_DIR,
    "/sdcard/digit_templates",
)

DIGIT_NORMALIZED_WIDTH = 48
DIGIT_NORMALIZED_HEIGHT = 64
DIGIT_NORMALIZED_MARGIN = 4

DIGIT_MIN_AREA = 150
DIGIT_MAX_AREA = 50000
DIGIT_MIN_WIDTH = 5
DIGIT_MIN_HEIGHT = 20
DIGIT_MAX_WIDTH = 250
DIGIT_MAX_HEIGHT = 400
DIGIT_MIN_ASPECT_RATIO = 0.05
DIGIT_MAX_ASPECT_RATIO = 1.40
DIGIT_MAX_COUNT = 12

DIGIT_MATCH_THRESHOLD = 0.35
