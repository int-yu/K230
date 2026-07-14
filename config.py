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

RECTANGLE_MIN_AREA = 1500
RECTANGLE_MIN_WIDTH = 30
RECTANGLE_MIN_HEIGHT = 30
RECTANGLE_MAX_COUNT = 24

# 依次尝试多个轮廓拟合精度，兼顾清晰边缘和轻微运动模糊。
RECTANGLE_APPROX_RATIOS = (0.015, 0.025, 0.040, 0.060)

# 不再使用内部面积比例。沿四边形的四条边分别采样：内侧应明显比外侧亮。
RECTANGLE_EDGE_SAMPLE_COUNT = 5
RECTANGLE_EDGE_SAMPLE_OFFSET_RATIO = 0.05
RECTANGLE_EDGE_SAMPLE_MIN_OFFSET = 3.0
RECTANGLE_EDGE_SAMPLE_MAX_OFFSET = 8.0
RECTANGLE_MIN_MEAN_EDGE_CONTRAST = 20.0
RECTANGLE_MIN_SIDE_EDGE_CONTRAST = 18.0
RECTANGLE_EDGE_TARGET_CONTRAST = 80.0

RECTANGLE_MIN_CONFIDENCE = 0.60
RECTANGLE_STRONG_CONFIDENCE = 0.85

# 亮区边界没有得到强候选时才执行 Canny，降低正常帧的计算量。
RECTANGLE_USE_CANNY_FALLBACK = True
RECTANGLE_CANNY_LOW_RATIO = 0.40
RECTANGLE_CANNY_HIGH_RATIO = 1.20

# 连续多少帧未检测到目标后，才判定目标丢失。
RECTANGLE_LOST_FRAME_LIMIT = 5


# ============================================================
# 彩色光点检测参数（color.py）
# ============================================================

# ColorSpotDetector() 不传参数时使用这些默认值。
COLOR_TARGET = "red"
COLOR_MIN_AREA = 8
COLOR_MAX_AREA = None
COLOR_MIN_CONFIDENCE = 0.0

# OpenCV HSV：H 为 0..179，S/V 为 0..255。
# 红色跨越 H 的首尾，因此需要两个范围。
COLOR_PRESET_HSV_RANGES = {
    "red": (
        ((0, 120, 120), (10, 255, 255)),
        ((170, 120, 120), (179, 255, 255)),
    ),
    "green": (
        ((35, 80, 80), (85, 255, 255)),
    ),
    "blue": (
        ((90, 80, 80), (135, 255, 255)),
    ),
    "yellow": (
        ((20, 100, 100), (35, 255, 255)),
    ),
}

COLOR_DRAW_COLOR = (255, 255, 255)
COLOR_DRAW_RADIUS = 7
COLOR_DRAW_CROSS_SIZE = 10


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
