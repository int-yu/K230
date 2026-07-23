"""K230 程序通用绘制辅助函数。

普通 OpenCV 程序使用 `draw_cv_*`；AI Pipeline 的 OSD 图层使用
`draw_osd_*`。函数内部尽量兼容 CanMV MicroPython，绘制失败时不会中断主程序。
"""


def draw_cv_text(
    frame,
    text,
    x=5,
    y=25,
    color=(255, 255, 255),
    scale=0.65,
    thickness=2,
):
    """在 OpenCV RGB 图像上绘制文字。"""

    try:
        import cv2
        cv2.putText(
            frame,
            text,
            (int(x), int(y)),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            thickness,
        )
    except Exception:
        pass


def draw_cv_fps(frame, fps, x=5, y=25, color=(255, 255, 255)):
    """按统一格式绘制帧率。"""

    draw_cv_text(frame, "FPS: {:.1f}".format(fps), x=x, y=y, color=color)


def draw_cv_count(frame, count, label="Count", x=5, y=53, color=(0, 255, 0)):
    """按统一格式绘制数量。"""

    draw_cv_text(frame, "{}: {}".format(label, count), x=x, y=y, color=color)


def draw_osd_text(
    osd_img,
    text,
    x=5,
    y=4,
    size=28,
    color=(255, 255, 255, 255),
):
    """在 PipeLine OSD 图层上绘制文字。"""

    try:
        osd_img.draw_string_advanced(
            int(x),
            int(y),
            int(size),
            text,
            color=color,
        )
    except Exception:
        try:
            osd_img.draw_string_advanced(int(x), int(y), int(size), text)
        except Exception:
            pass


def draw_osd_status(
    osd_img,
    fps=None,
    count=None,
    count_label="Num",
    x=5,
    y=4,
    line_height=30,
    size=28,
    text_color=(255, 255, 255, 255),
    background_color=(160, 0, 0, 0),
):
    """在 OSD 左上角绘制 FPS 和数量，格式与其它主程序保持一致。"""

    lines = []
    if fps is not None:
        lines.append("FPS: {:.1f}".format(fps))
    if count is not None:
        lines.append("{}: {}".format(count_label, count))

    if not lines:
        return

    try:
        osd_img.draw_rectangle(
            0,
            0,
            170,
            max(40, int(line_height) * len(lines) + 8),
            color=background_color,
            thickness=1,
            fill=True,
        )
    except Exception:
        pass

    for index, line in enumerate(lines):
        draw_osd_text(
            osd_img,
            line,
            x=x,
            y=y + index * line_height,
            size=size,
            color=text_color,
        )
