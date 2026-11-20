"""Slow checks against the real model and COCO files: uv run pytest -m slow"""

import numpy as np
import pytest

from detector.evaluation import EvalSettings, list_images, run_eval
from detector.inference_core import postprocess, preprocess

pytestmark = pytest.mark.slow


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of one box a (4,) against boxes b (N, 4), xyxy."""
    x1, y1 = np.maximum(a[0], b[:, 0]), np.maximum(a[1], b[:, 1])
    x2, y2 = np.minimum(a[2], b[:, 2]), np.minimum(a[3], b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda x: (x[..., 2] - x[..., 0]) * (x[..., 3] - x[..., 1])  # noqa: E731
    return inter / (area(a) + area(b) - inter)


def test_matches_ultralytics_predict(coco_files, weights, torch_runtime):
    """Our preprocessing + postprocessing reproduce Ultralytics' own predictions."""
    from ultralytics import YOLO

    images, _ = coco_files
    paths = list_images(images, limit=8)
    pre = preprocess([p.read_bytes() for p in paths])
    ours = postprocess(torch_runtime.infer(pre.batch), pre.metas, conf=0.25, iou=0.7)

    model = YOLO(str(weights))
    for path, det in zip(paths, ours, strict=True):
        ref = model.predict(
            str(path), imgsz=640, conf=0.25, iou=0.7, rect=False,
            device=torch_runtime.device, verbose=False,
        )[0].boxes  # fmt: skip
        ref_boxes, ref_scores = ref.xyxy.cpu().numpy(), ref.conf.cpu().numpy()
        ref_cls = ref.cls.cpu().numpy().astype(int)

        # Detections right at the 0.25 threshold may flip with float noise.
        assert abs(len(det) - len(ref_scores)) <= 1, path.name
        matched = 0
        for box, score, cls in zip(det.boxes, det.scores, det.class_ids, strict=True):
            same = ref_cls == cls
            if not same.any():
                continue
            ious = iou(box, ref_boxes[same])
            best = ious.argmax()
            if ious[best] > 0.99 and abs(ref_scores[same][best] - score) < 2e-3:
                matched += 1
        assert matched >= len(det) - 1, f"{path.name}: {matched}/{len(det)} matched"


def test_map_smoke(coco_files, torch_runtime):
    images, annotations = coco_files
    summary, _ = run_eval(torch_runtime, images, annotations, EvalSettings(limit=100))
    assert summary["processed"] == 100
    assert summary["metrics"]["map50_95"] > 0.45
