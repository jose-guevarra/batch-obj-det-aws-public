import cv2
import numpy as np
import pytest

from detector.render import class_color, draw, render_jpeg


def test_draw_marks_the_box_and_keeps_the_input():
    img = np.full((200, 300, 3), 128, np.uint8)
    out = draw(img, [{"box": [50, 60, 150, 160], "score": 0.9, "class_id": 0}])
    assert out.shape == img.shape
    assert (img == 128).all()  # input untouched
    assert (out[100, 50] == class_color(0)).all()  # left edge of the box
    assert (out[100, 100] == 128).all()  # inside stays clear
    assert (out[190, 290] == 128).all()  # far corner untouched


def test_class_colors_differ():
    assert len({class_color(i) for i in range(80)}) == 80


def test_render_jpeg_round_trip():
    img = np.zeros((64, 64, 3), np.uint8)
    data = cv2.imencode(".png", img)[1].tobytes()
    out = render_jpeg(data, [{"box": [0, 0, 63, 63], "score": 0.5, "class_id": 3}])
    decoded = cv2.imdecode(np.frombuffer(out, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (64, 64, 3)
    with pytest.raises(ValueError):
        render_jpeg(b"nope", [])
