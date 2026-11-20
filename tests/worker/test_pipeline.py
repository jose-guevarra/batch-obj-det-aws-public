"""The pipelined worker (prepare / infer / finish in three threads) against moto."""

import pytest

from detector.worker import WorkerConfig


def run(worker):
    """Pipeline until a poll finds the queue empty; reports in commit order."""
    return worker.run_pipelined(idle_exit=True)


def test_pipelined_matches_serial_results(env):
    env.store.create_batch("b1", 40, "yolov8m")
    env.send("b1", env.upload("b1", 40))

    reports = run(env.worker(batch_size=16))

    assert sorted(r.images for r in reports) == [8, 16, 16]
    assert all(r.pipelined for r in reports)
    assert all("wait_input" in r.stages_ms and "load" in r.stages_ms for r in reports)
    assert all(r.t_receive <= r.t_written for r in reports)
    batch = env.store.get_batch("b1")
    assert (batch["completed"], batch["failed"]) == (40, 0)
    assert len(env.part_files("b1")) == 40
    assert env.queue_count() == 0


def test_pipelined_failures_are_isolated(env):
    env.store.create_batch("b1", 4, "yolov8m")
    names = env.upload("b1", 2) + env.upload("b1", 1, start=2, data=b"not a jpeg")
    env.send("b1", [*names, "never-uploaded.jpg"])

    reports = run(env.worker(batch_size=16))

    assert sum(r.processed for r in reports) == 2
    assert sum(r.failed for r in reports) == 2
    batch = env.store.get_batch("b1")
    assert (batch["completed"], batch["failed"]) == (2, 2)
    assert env.store.get_image("b1", "img0002.jpg")["error"].startswith("decode failed")
    assert env.store.get_image("b1", "never-uploaded.jpg")["error"] == "object missing from S3"
    assert len(env.dlq_messages()) == 2
    assert env.queue_count() == 0


def test_failed_inference_leaves_only_that_batch_for_retry(env):
    env.store.create_batch("b1", 24, "yolov8m")
    env.send("b1", env.upload("b1", 24))
    calls = []

    def fail_second_batch():
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("CUDA error: an illegal memory access was encountered")

    env.runtime.on_infer = fail_second_batch

    reports = run(env.worker(batch_size=8))

    # Batches 1 and 3 committed; batch 2's 8 messages stay on the queue for SQS to retry.
    assert len(reports) == 2
    assert env.store.get_batch("b1")["completed"] == 16
    assert env.queue_count() == 8


def test_stop_finishes_the_batches_in_flight(env):
    env.store.create_batch("b1", 48, "yolov8m")
    env.send("b1", env.upload("b1", 48))
    worker = env.worker(batch_size=8)
    env.runtime.on_infer = worker.request_stop  # SIGTERM during the first inference

    reports = worker.run_pipelined()  # no idle_exit: only the stop ends it

    # Everything that was received is either committed and deleted, or still on the queue;
    # nothing is deleted without its result.
    committed = env.store.get_batch("b1")["completed"]
    assert committed == sum(r.processed for r in reports) >= 8
    assert sum(r.deleted for r in reports) == committed
    assert env.queue_count() == 48 - committed
    assert committed < 48  # it stopped instead of draining the queue


def test_config_reads_runtime_pipeline_and_eval_settings():
    base = {"QUEUE_URL": "q", "DLQ_URL": "d", "BUCKET": "b", "TABLE": "t"}
    cfg = WorkerConfig.from_env(
        base | {"RUNTIME": "onnx-trt", "PIPELINE": "1", "MULTI_LABEL": "true", "CONF": "0.001"}
    )
    assert (cfg.runtime, cfg.pipeline, cfg.multi_label, cfg.conf) == (
        "onnx-trt", True, True, 0.001
    )  # fmt: skip
    defaults = WorkerConfig.from_env(base)
    assert (defaults.runtime, defaults.pipeline, defaults.multi_label) == ("pytorch", False, False)
    with pytest.raises(ValueError, match="RUNTIME"):
        WorkerConfig.from_env(base | {"RUNTIME": "tensorflow"})
