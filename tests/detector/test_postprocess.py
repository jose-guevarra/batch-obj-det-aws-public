import numpy as np
import pytest
import torch

from detector.inference_core import LetterboxMeta, postprocess

NC = 80
IDENTITY = LetterboxMeta(orig_h=640, orig_w=640, gain_x=1.0, gain_y=1.0, pad_x=0, pad_y=0)


def raw_from(anchors: list[tuple[tuple[float, float, float, float], dict[int, float]]]):
    """Build a (1, 84, N) tensor: each anchor is (xywh, {class_id: score})."""
    out = torch.zeros(1, 4 + NC, len(anchors))
    for n, (xywh, scores) in enumerate(anchors):
        out[0, :4, n] = torch.tensor(xywh)
        for cls, score in scores.items():
            out[0, 4 + cls, n] = score
    return out


def test_confidence_threshold():
    raw = raw_from([((100, 100, 50, 50), {0: 0.9}), ((300, 300, 50, 50), {0: 0.1})])
    (det,) = postprocess(raw, [IDENTITY], conf=0.25)
    assert len(det) == 1
    assert det.scores[0] == pytest.approx(0.9)


def test_xywh_becomes_xyxy():
    raw = raw_from([((100, 200, 40, 60), {3: 0.8})])
    (det,) = postprocess(raw, [IDENTITY])
    np.testing.assert_allclose(det.boxes[0], [80, 170, 120, 230])
    assert det.class_ids.tolist() == [3]


def test_nms_removes_same_class_duplicate_but_keeps_other_class():
    raw = raw_from(
        [
            ((100, 100, 50, 50), {0: 0.9}),
            ((102, 101, 50, 50), {0: 0.8}),  # duplicate person -> suppressed
            ((101, 100, 50, 50), {2: 0.7}),  # same spot, car -> kept
        ]
    )
    (det,) = postprocess(raw, [IDENTITY])
    assert sorted(det.class_ids.tolist()) == [0, 2]


def test_multi_label_keeps_every_class_above_conf():
    raw = raw_from([((100, 100, 50, 50), {0: 0.6, 1: 0.5, 2: 0.1})])
    (single,) = postprocess(raw, [IDENTITY], conf=0.25, multi_label=False)
    (multi,) = postprocess(raw, [IDENTITY], conf=0.25, multi_label=True)
    assert single.class_ids.tolist() == [0]
    assert sorted(multi.class_ids.tolist()) == [0, 1]


def test_max_det():
    anchors = [((20 + 30 * i, 20, 10, 10), {0: 0.5 + i / 100}) for i in range(20)]
    (det,) = postprocess(raw_from(anchors), [IDENTITY], max_det=5)
    assert len(det) == 5
    assert det.scores[0] == pytest.approx(0.69)  # highest first


def test_boxes_map_back_to_original_pixels_and_clip():
    # 1280x960 image letterboxed to 640x480 + 80 px top padding.
    meta = LetterboxMeta(orig_h=960, orig_w=1280, gain_x=0.5, gain_y=0.5, pad_x=0, pad_y=80)
    raw = raw_from(
        [
            ((100, 180, 40, 40), {0: 0.9}),  # letterbox xyxy 80,160,120,200
            ((630, 90, 40, 40), {1: 0.9}),  # runs off the right edge and into the top pad
        ]
    )
    (det,) = postprocess(raw, [meta])
    by_cls = dict(zip(det.class_ids.tolist(), det.boxes, strict=True))
    np.testing.assert_allclose(by_cls[0], [160, 160, 240, 240])
    np.testing.assert_allclose(by_cls[1], [1220, 0, 1280, 60])


def test_each_image_uses_its_own_meta():
    raw = torch.cat([raw_from([((100, 100, 20, 20), {0: 0.9})])] * 2)
    small = LetterboxMeta(orig_h=320, orig_w=320, gain_x=2.0, gain_y=2.0, pad_x=0, pad_y=0)
    a, b = postprocess(raw, [IDENTITY, small])
    np.testing.assert_allclose(a.boxes[0], [90, 90, 110, 110])
    np.testing.assert_allclose(b.boxes[0], [45, 45, 55, 55])


def test_no_detections_gives_empty_arrays():
    (det,) = postprocess(raw_from([((100, 100, 50, 50), {0: 0.01})]), [IDENTITY])
    assert len(det) == 0
    assert det.boxes.shape == (0, 4)
    assert det.to_records() == []


def test_meta_count_must_match_batch():
    with pytest.raises(ValueError, match="2 rows but 1 metas"):
        postprocess(torch.zeros(2, 84, 10), [IDENTITY])


def test_half_precision_input():
    raw = raw_from([((100, 100, 50, 50), {5: 0.9})]).half()
    (det,) = postprocess(raw, [IDENTITY])
    assert det.boxes.dtype == np.float32
    assert det.to_records()[0]["class_name"] == "bus"
