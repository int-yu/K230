"""CaptureService 的编号与待拍计数逻辑测试，不依赖 K230 硬件。"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from capture import CaptureService

FAKE_JPEG = b"\xff\xd8fake"


class FakeImage:
    """记录 compressed() 调用的假图像对象，返回假 JPEG 字节。"""

    def __init__(self):
        self.compressed_calls = 0

    def compressed(self, quality=None):
        self.compressed_calls += 1
        return FAKE_JPEG


def test_start_index_is_one_on_empty_dir(tmp_path):
    service = CaptureService(save_dir=str(tmp_path))
    image = FakeImage()
    service.handle_frames([(0x20, 0, bytes((1,)))])
    saved, last_index = service.update(image)
    assert saved == 1
    assert last_index == 1
    expected_path = tmp_path / "cap_0001.jpg"
    assert expected_path.exists()
    assert expected_path.read_bytes() == FAKE_JPEG


def test_index_continues_after_existing_files(tmp_path):
    (tmp_path / "cap_0007.jpg").write_bytes(b"")
    service = CaptureService(save_dir=str(tmp_path))
    image = FakeImage()
    service.handle_frames([(0x20, 0, bytes((1,)))])
    saved, last_index = service.update(image)
    assert last_index == 8


def test_burst_saves_one_per_update(tmp_path):
    service = CaptureService(save_dir=str(tmp_path))
    image = FakeImage()
    service.handle_frames([(0x20, 0, bytes((3,)))])
    assert service.pending == 3
    service.update(image)
    assert service.pending == 2
    service.update(image)
    service.update(image)
    assert service.pending == 0
    assert image.compressed_calls == 3


def test_non_capture_frames_are_ignored(tmp_path):
    service = CaptureService(save_dir=str(tmp_path))
    added = service.handle_frames([(0x01, 0, b""), (0x10, 1, bytes(5))])
    assert added == 0
    assert service.pending == 0


def test_pending_is_clamped(tmp_path):
    service = CaptureService(save_dir=str(tmp_path), max_pending=5)
    service.handle_frames([(0x20, 0, bytes((20,)))])
    assert service.pending == 5


def test_update_without_pending_saves_nothing(tmp_path):
    service = CaptureService(save_dir=str(tmp_path))
    image = FakeImage()
    saved, last_index = service.update(image)
    assert saved == 0
    assert last_index == 0
    assert image.compressed_calls == 0


class FailingImage:
    """compressed() 始终抛 OSError 的假图像对象，用于测试写入失败路径。"""

    def compressed(self, quality=None):
        raise OSError("TF 卡未挂载")


def test_save_failure_decrements_pending_and_does_not_advance_index(tmp_path):
    """compressed() 抛 OSError 时 update() 应返回 (0, 0)、pending 递减、next_index 不推进。"""
    service = CaptureService(save_dir=str(tmp_path))
    # 先累加两张待拍
    service.handle_frames([(0x20, 0, bytes((2,)))])
    assert service.pending == 2
    initial_index = service.next_index

    image = FailingImage()
    saved, last_index = service.update(image)

    # 失败时返回 (0, 0)
    assert saved == 0
    assert last_index == 0
    # pending 减了一（消耗一次机会，不无限重试）
    assert service.pending == 1
    # next_index 没有推进（文件没写成，编号留给下次）
    assert service.next_index == initial_index


def test_write_failure_after_compressed_success_does_not_advance_index(tmp_path, monkeypatch):
    """compressed() 成功但写文件失败时，应返回 None 且不推进编号。

    覆盖 open() 成功但 write() 失败会留下半截文件的场景：
    失败路径上会尝试 os.remove() 删除残留，不推进编号。
    """
    service = CaptureService(save_dir=str(tmp_path))
    service.handle_frames([(0x20, 0, bytes((1,)))])
    initial_index = service.next_index

    # 让 open() 在以写模式调用时抛 OSError，模拟磁盘已满或路径不可写。
    original_open = open

    def failing_open(path, mode="r", *args, **kwargs):
        if "w" in str(mode) and "b" in str(mode):
            raise OSError("磁盘已满")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr("builtins.open", failing_open)

    image = FakeImage()
    result = service.save(image)

    assert result is None
    assert service.next_index == initial_index


def test_handle_frames_returns_actual_added_when_clamped(tmp_path):
    """夹紧时 handle_frames() 应返回实际新增量，而不是请求量。"""
    service = CaptureService(save_dir=str(tmp_path), max_pending=20)
    # 先占用 18 张，剩余容量 2
    service.handle_frames([(0x20, 0, bytes((18,)))])
    assert service.pending == 18

    # 请求 5 张，但容量只剩 2
    added = service.handle_frames([(0x20, 0, bytes((5,)))])

    # 应返回实际新增量 2，而不是请求量 5
    assert added == 2
    assert service.pending == 20
