import json

from detector.finalize import finalize
from detector.store import ImageResult
from detector.worker import Worker
from tests.worker.conftest import BUCKET


def worker_with_finalizer(env, calls, **overrides):
    def finalizer(batch_id):
        calls.append(batch_id)
        finalize(env.store, env.s3, BUCKET, batch_id)

    return Worker(
        env.config(**overrides), env.runtime, env.sqs, env.s3, env.store, finalizer=finalizer
    )


def drain(worker):
    while worker.run_once() is not None:
        pass


def s3_text(env, key):
    return env.s3.get_object(Bucket=BUCKET, Key=key)["Body"].read().decode()


def test_completing_commit_finalizes_the_batch_once(env):
    env.store.create_batch("b1", 20, "yolov8m")
    names = env.upload("b1", 19)
    env.s3.put_object(Bucket=BUCKET, Key="input/b1/bad.jpg", Body=b"not an image")
    env.send("b1", names + ["bad.jpg"])
    calls = []

    drain(worker_with_finalizer(env, calls, batch_size=8))

    assert calls == ["b1"]  # only the last commit completed the batch
    batch = env.store.get_batch("b1")
    assert batch["status"] == "done"
    lines = [json.loads(x) for x in s3_text(env, "results/b1/results.jsonl").splitlines()]
    assert [x["image_name"] for x in lines] == sorted(names + ["bad.jpg"])
    summary = json.loads(s3_text(env, "results/b1/summary.json"))
    assert (summary["images"], summary["completed"], summary["failed"]) == (20, 19, 1)
    assert summary["detections"] == 19
    assert summary["detections_per_class"] == {"person": 19}
    assert summary["failures"][0]["image_name"] == "bad.jpg"
    assert summary["workers"] == 1
    assert batch["detections"] == 19

    # Already done: neither the worker path nor a direct call runs it again.
    assert finalize(env.store, env.s3, BUCKET, "b1") is None


def test_incomplete_batch_is_not_finalized(env):
    env.store.create_batch("b1", 10, "yolov8m")
    env.send("b1", env.upload("b1", 4))
    calls = []
    drain(worker_with_finalizer(env, calls))
    assert calls == []
    assert finalize(env.store, env.s3, BUCKET, "b1") is None
    assert env.store.get_batch("b1")["status"] == "pending"


def test_duplicate_part_lines_keep_the_committed_one(env):
    env.store.create_batch("b1", 2, "yolov8m")
    meta = {"runtime": "fake", "precision": "fp32"}
    det = [{"box": [1, 2, 3, 4], "score": 0.5, "class_id": 2}]
    env.store.commit(
        "b1",
        [ImageResult("a.jpg", "done", det), ImageResult("b.jpg", "done", det)],
        worker_id="w2",
        **meta,
    )
    # w1 wrote a part line for a.jpg, then crashed before committing; w2 committed both.
    stale = {"batch_id": "b1", "image_name": "a.jpg", "status": "failed", "worker_id": "w1"}
    good = [
        {"batch_id": "b1", "image_name": n, "status": "done", "detections": det, "worker_id": "w2"}
        for n in ("a.jpg", "b.jpg")
    ]
    env.s3.put_object(
        Bucket=BUCKET, Key="results/b1/part-w1-x-000001.jsonl", Body=json.dumps(stale)
    )
    env.s3.put_object(
        Bucket=BUCKET,
        Key="results/b1/part-w2-y-000001.jsonl",
        Body="\n".join(json.dumps(g) for g in good),
    )

    summary = finalize(env.store, env.s3, BUCKET, "b1")

    lines = [json.loads(x) for x in s3_text(env, "results/b1/results.jsonl").splitlines()]
    assert [(x["image_name"], x["status"], x["worker_id"]) for x in lines] == [
        ("a.jpg", "done", "w2"),
        ("b.jpg", "done", "w2"),
    ]
    assert summary["detections_per_class"] == {"car": 2}


def test_missing_part_line_is_rebuilt_from_dynamodb(env):
    env.store.create_batch("b1", 1, "yolov8m")
    env.store.commit(
        "b1",
        [ImageResult.failed("a.jpg", "decode error")],
        worker_id="w1",
        runtime="fake",
        precision="fp32",
    )
    summary = finalize(env.store, env.s3, BUCKET, "b1")
    assert summary["failures"] == [{"image_name": "a.jpg", "error": "decode error"}]


def test_finalizer_error_does_not_fail_the_batch(env):
    env.store.create_batch("b1", 3, "yolov8m")
    env.send("b1", env.upload("b1", 3))

    def broken(_):
        raise RuntimeError("lambda down")

    worker = Worker(env.config(), env.runtime, env.sqs, env.s3, env.store, finalizer=broken)
    drain(worker)
    assert env.store.get_batch("b1")["completed"] == 3
    assert env.queue_count() == 0
