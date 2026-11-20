import numpy as np
import pytest

from detector.inference_core import IMG_SIZE, PAD_VALUE, letterbox


@pytest.mark.parametrize(
    ("h", "w", "pad_x", "pad_y"),
    [
        (480, 640, 0, 80),  # wide: pad top/bottom
        (640, 427, 106, 0),  # tall: 213 px of padding, odd -> 106 left, 107 right
        (500, 500, 0, 0),  # square, scaled up
        (640, 640, 0, 0),  # already the right size
        (1, 3000, 0, 319),  # very thin: resized side kept at least 1 px; 639 px pad -> 319 top
    ],
)
def test_output_shape_and_padding(h, w, pad_x, pad_y):
    out, meta = letterbox(np.zeros((h, w, 3), np.uint8))
    assert out.shape == (IMG_SIZE, IMG_SIZE, 3)
    assert (meta.orig_h, meta.orig_w) == (h, w)
    assert (meta.pad_x, meta.pad_y) == (pad_x, pad_y)


def test_padding_pixels_are_gray():
    out, meta = letterbox(np.zeros((480, 640, 3), np.uint8))
    assert (out[: meta.pad_y] == PAD_VALUE).all()
    assert (out[-meta.pad_y :] == PAD_VALUE).all()
    assert (out[meta.pad_y : -meta.pad_y] == 0).all()


def test_box_round_trips_to_original_pixels():
    h, w = 375, 500
    _, meta = letterbox(np.zeros((h, w, 3), np.uint8))
    box = np.array([50.0, 60.0, 400.0, 300.0])  # xyxy in the original image
    gain = np.array([meta.gain_x, meta.gain_y] * 2)
    pad = np.array([meta.pad_x, meta.pad_y] * 2)
    boxed = box * gain + pad
    assert (boxed[[0, 2]] <= 640).all() and (boxed[[1, 3]] <= 640).all()
    back = (boxed - pad) / gain
    np.testing.assert_allclose(back, box, atol=1.0)


def test_image_content_is_centered():
    img = np.full((320, 640, 3), 200, np.uint8)  # scales 2x to 640x1280 -> 320x640 fits: r=1
    out, meta = letterbox(img)
    assert meta.gain_x == meta.gain_y == 1.0
    assert (out[meta.pad_y : meta.pad_y + 320] == 200).all()
