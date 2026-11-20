"""ONNX Runtime (CPU backend) against PyTorch on real images.

Needs models/yolov8m.onnx from the GPU image: `make -C infra onnx-local`.
"""

from pathlib import Path

import numpy as np
import pytest

from detector.inference_core import postprocess, preprocess

ONNX = Path(__file__).resolve().parents[2] / "models/yolov8m.onnx"


@pytest.mark.slow
def test_onnx_matches_pytorch(coco_files, torch_runtime):
    if not ONNX.is_file():
        pytest.skip(f"{ONNX} missing: run make -C infra onnx-local")
    from detector.runtimes import OnnxRuntime

    images, _ = coco_files
    paths = sorted(images.glob("*.jpg"))[:20]
    pre = preprocess([p.read_bytes() for p in paths])
    onnx = OnnxRuntime(ONNX, "cpu")

    ours = postprocess(onnx.infer(pre.batch), pre.metas)
    ref = postprocess(torch_runtime.infer(pre.batch).cpu(), pre.metas)

    for a, b in zip(ours, ref, strict=True):
        assert len(a.boxes) == len(b.boxes)
        np.testing.assert_array_equal(a.class_ids, b.class_ids)
        np.testing.assert_allclose(a.boxes, b.boxes, atol=1.0)  # within a pixel
        np.testing.assert_allclose(a.scores, b.scores, atol=0.01)
