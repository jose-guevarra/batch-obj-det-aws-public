"""End to end with the real model: COCO images -> moto S3/SQS/DynamoDB -> worker."""

from pathlib import Path

import numpy as np
import pytest

from detector.inference_core import postprocess, preprocess

pytestmark = pytest.mark.slow

ROOT = Path(__file__).resolve().parents[2]
IMAGES = ROOT / "data/datasets/coco/val2017"
WEIGHTS = ROOT / "models/yolov8m.pt"


@pytest.fixture(scope="module")
def real_runtime():
    if not IMAGES.is_dir() or not WEIGHTS.is_file():
        pytest.skip("needs data/datasets/coco/val2017 and models/yolov8m.pt")
    from detector.runtimes import TorchRuntime

    return TorchRuntime(WEIGHTS, device="auto")


def test_worker_matches_direct_inference(env, real_runtime):
    paths = sorted(IMAGES.glob("*.jpg"))[:20]
    names = [p.name for p in paths]
    for p in paths:
        env.s3.put_object(
            Bucket="detector-batch-test", Key=f"input/coco/{p.name}", Body=p.read_bytes()
        )
    env.store.create_batch("coco", len(paths), "yolov8m")
    env.send("coco", names)

    env.runtime = real_runtime
    worker = env.worker(batch_size=8)
    while worker.run_once() is not None:
        pass

    batch = env.store.get_batch("coco")
    assert (batch["completed"], batch["failed"]) == (20, 0)
    assert env.queue_count() == 0

    pre = preprocess([p.read_bytes() for p in paths])
    expected = postprocess(real_runtime.infer(pre.batch), pre.metas)
    for name, det in zip(names, expected, strict=True):
        item = env.store.get_image("coco", name)
        assert item["num_detections"] == len(det), name
        got = sorted(item["detections"], key=lambda d: -d["score"])
        for d, box, score, cls in zip(got, det.boxes, det.scores, det.class_ids, strict=True):
            assert d["class_id"] == int(cls)
            np.testing.assert_allclose(d["box"], box, atol=0.02)
            assert d["score"] == pytest.approx(float(score), abs=1e-4)
