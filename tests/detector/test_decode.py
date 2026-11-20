import cv2
import numpy as np
import pytest

from detector.inference_core import DecodeError, StageTimer, decode, preprocess


def encode(ext: str, h: int = 120, w: int = 160) -> bytes:
    img = np.random.default_rng(0).integers(0, 255, (h, w, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(ext, img)
    assert ok
    return buf.tobytes()


@pytest.mark.parametrize("ext", [".jpg", ".png"])
def test_valid_images_decode_to_bgr(ext):
    img = decode(encode(ext))
    assert img.shape == (120, 160, 3)
    assert img.dtype == np.uint8


@pytest.mark.parametrize(
    "data",
    [b"", b"not an image at all", np.random.default_rng(1).bytes(4096)],
    ids=["empty", "text", "random"],
)
def test_garbage_raises_decode_error(data):
    with pytest.raises(DecodeError):
        decode(data)


def test_truncated_jpeg_header_raises_decode_error():
    with pytest.raises(DecodeError):
        decode(encode(".jpg")[:20])


def test_one_bad_image_does_not_fail_the_batch():
    good = encode(".jpg")
    pre = preprocess([good, b"corrupt", good, b""])
    assert pre.batch.shape == (2, 3, 640, 640)
    assert pre.indices == [0, 2]
    assert sorted(pre.failures) == [1, 3]
    assert all(isinstance(e, DecodeError) for e in pre.failures.values())
    assert len(pre.metas) == 2


def test_all_bad_gives_no_batch():
    pre = preprocess([b"x", b"y"])
    assert pre.batch is None
    assert sorted(pre.failures) == [0, 1]


def test_preprocess_times_decode_and_preprocess_separately():
    timer = StageTimer()
    preprocess([encode(".jpg"), b"bad"], timer=timer)
    assert timer.counts["decode"] == 2  # failed decodes are timed too
    assert set(timer.totals) == {"decode", "preprocess"}
