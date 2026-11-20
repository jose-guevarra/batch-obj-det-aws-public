import numpy as np

from detector.inference_core import to_batch


def test_shape_dtype_and_range():
    imgs = [np.full((640, 640, 3), 255, np.uint8), np.zeros((640, 640, 3), np.uint8)]
    batch = to_batch(imgs)
    assert batch.shape == (2, 3, 640, 640)
    assert batch.dtype == np.float32
    assert batch.flags["C_CONTIGUOUS"]
    assert batch[0].min() == batch[0].max() == 1.0
    assert batch[1].max() == 0.0


def test_bgr_becomes_rgb_channels_first():
    img = np.zeros((4, 4, 3), np.uint8)
    img[..., 0] = 10  # blue
    img[..., 2] = 250  # red
    batch = to_batch([img])
    np.testing.assert_allclose(batch[0, 0], 250 / 255)  # channel 0 = red
    np.testing.assert_allclose(batch[0, 2], 10 / 255)  # channel 2 = blue
