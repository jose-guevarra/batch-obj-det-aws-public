import json
import signal

import pytest
from botocore.exceptions import ClientError

from detector.store import BatchStore
from detector.worker import Worker, WorkerConfig
from tests.worker.conftest import BUCKET, s3_event, wait_for_retry


def drain(worker, max_batches: int = 20):
    reports = []
    for _ in range(max_batches):
        report = worker.run_once()
        if report is None:
            break
        reports.append(report)
    return reports


def test_happy_path(env):
    env.store.create_batch("b1", 40, "yolov8m")
    env.send("b1", env.upload("b1", 40))

    reports = drain(env.worker(batch_size=16))

    assert [r.images for r in reports] == [16, 16, 8]
    assert env.runtime.calls == [16, 16, 8]
    batch = env.store.get_batch("b1")
    assert (batch["completed"], batch["failed"]) == (40, 0)
    assert env.queue_count() == 0
    lines = env.part_files("b1")
    assert len(lines) == 40
    assert {line["status"] for line in lines} == {"done"}
    first = env.store.get_image("b1", "img0000.jpg")
    assert first["num_detections"] == 1
    # The fake box is centered in the letterbox; mapped back it's inside the 64x48 image.
    x1, y1, x2, y2 = first["detections"][0]["box"]
    assert 0 <= x1 < x2 <= 64 and 0 <= y1 < y2 <= 48


def test_rerun_skips_finished_images(env):
    env.store.create_batch("b1", 20, "yolov8m")
    names = env.upload("b1", 20)
    env.send("b1", names)
    drain(env.worker())
    calls_before = list(env.runtime.calls)

    env.send("b1", names)  # same images again
    reports = drain(env.worker())

    assert env.runtime.calls == calls_before  # no new inference
    assert sum(r.skipped for r in reports) == 20
    assert env.store.get_batch("b1")["completed"] == 20
    assert env.queue_count() == 0


def test_corrupt_image_is_isolated(env):
    env.store.create_batch("b1", 16, "yolov8m")
    names = env.upload("b1", 15)
    names += env.upload("b1", 1, start=15, data=b"definitely not a jpeg")
    env.send("b1", names)

    (report,) = drain(env.worker())

    assert (report.processed, report.failed, report.dead_lettered) == (15, 1, 1)
    batch = env.store.get_batch("b1")
    assert (batch["completed"], batch["failed"]) == (15, 1)
    bad = env.store.get_image("b1", "img0015.jpg")
    assert bad["status"] == "failed" and "decode failed" in bad["error"]
    (dlq,) = env.dlq_messages()
    assert "decode failed" in dlq["MessageAttributes"]["error"]["StringValue"]
    assert "img0015.jpg" in dlq["Body"]
    assert env.queue_count() == 0


def test_missing_object_fails_permanently(env):
    env.store.create_batch("b1", 2, "yolov8m")
    env.send("b1", env.upload("b1", 1) + ["never-uploaded.jpg"])

    (report,) = drain(env.worker())

    assert (report.processed, report.failed) == (1, 1)
    assert env.store.get_image("b1", "never-uploaded.jpg")["error"] == "object missing from S3"
    assert env.queue_count() == 0


def test_bad_message_goes_to_dlq(env):
    env.send_body("this is not an S3 event")
    env.send_body(s3_event(BUCKET, "results/b1/part.jsonl"))
    env.send_body(json.dumps({"Event": "s3:TestEvent"}))

    assert env.worker().run_once() is None  # nothing processable

    dlq = env.dlq_messages()
    assert len(dlq) == 2  # the test event is just deleted
    assert env.queue_count() == 0


class FlakyS3:
    """Wraps the S3 client; GetObject on one key raises a throttling error."""

    def __init__(self, s3, bad_key: str):
        self._s3, self._bad_key = s3, bad_key

    def get_object(self, **kw):
        if kw["Key"] == self._bad_key:
            raise ClientError({"Error": {"Code": "SlowDown", "Message": "slow"}}, "GetObject")
        return self._s3.get_object(**kw)

    def __getattr__(self, name):
        return getattr(self._s3, name)


def test_temporary_error_leaves_message_for_retry(env_fast_retry):
    env = env_fast_retry
    env.store.create_batch("b1", 4, "yolov8m")
    env.send("b1", env.upload("b1", 4))
    flaky = FlakyS3(env.s3, "input/b1/img0002.jpg")

    report = env.worker(s3=flaky).run_once()

    assert (report.processed, report.retry, report.deleted) == (3, 1, 3)
    assert env.store.get_batch("b1")["completed"] == 3
    assert env.queue_count() == 1  # the throttled image is still queued

    # A healthy worker picks it up once its visibility timeout runs out.
    wait_for_retry()
    drain(env.worker())
    assert env.store.get_batch("b1")["completed"] == 4
    assert env.queue_count() == 0


def test_missing_batch_record_is_retried_not_lost(env_fast_retry):
    env = env_fast_retry
    env.send("b9", env.upload("b9", 2))

    report = env.worker().run_once()
    assert (report.retry, report.deleted) == (2, 0)

    env.store.create_batch("b9", 2, "yolov8m")  # record shows up late
    wait_for_retry()
    drain(env.worker())
    assert env.store.get_batch("b9")["completed"] == 2


def test_batch_stops_at_batch_size(env):
    env.store.create_batch("b1", 25, "yolov8m")
    env.send("b1", env.upload("b1", 25))
    report = env.worker(batch_size=8).run_once()
    assert report.images == 8


def test_partial_batch_after_max_wait(env):
    env.store.create_batch("b1", 3, "yolov8m")
    env.send("b1", env.upload("b1", 3))
    report = env.worker(batch_size=32, max_wait_s=0.05).run_once()
    assert report.images == 3


def test_crash_between_part_file_and_commit(env_fast_retry, monkeypatch):
    env = env_fast_retry
    env.store.create_batch("b1", 5, "yolov8m")
    env.send("b1", env.upload("b1", 5))
    worker = env.worker()

    real_commit = BatchStore.commit

    def crash(*a, **kw):
        raise RuntimeError("worker killed")

    monkeypatch.setattr(BatchStore, "commit", crash)
    with pytest.raises(RuntimeError):
        worker.run_once()
    monkeypatch.setattr(BatchStore, "commit", real_commit)

    wait_for_retry()
    drain(env.worker())
    assert env.store.get_batch("b1")["completed"] == 5
    assert env.queue_count() == 0
    # Two part files hold the same 5 images; finalize keeps one line per image.
    lines = env.part_files("b1")
    assert len(lines) == 10 and len({line["image_name"] for line in lines}) == 5


def test_duplicate_delivery_is_counted_once(env):
    env.store.create_batch("b1", 6, "yolov8m")
    names = env.upload("b1", 6)
    env.send("b1", names)
    env.send("b1", names)  # S3 delivered every event twice

    reports = drain(env.worker(batch_size=16))

    assert sum(r.messages for r in reports) == 12
    assert env.runtime.calls == [6]  # each image inferred once
    assert env.store.get_batch("b1")["completed"] == 6
    assert env.queue_count() == 0


def test_two_workers_committing_the_same_images(env):
    env.store.create_batch("b1", 4, "yolov8m")
    names = env.upload("b1", 4)
    env.send("b1", names)
    w1 = env.worker(worker_id="w1")
    w2 = env.worker(worker_id="w2")

    # w2's skip check runs before w1 commits, so both process the images.
    real_existing = env.store.existing
    env.store.existing = lambda *a, **kw: set()
    w1.run_once()
    env.send("b1", names)
    report = w2.run_once()
    env.store.existing = real_existing

    assert report.skipped == 4
    assert env.store.get_batch("b1")["completed"] == 4


def test_sigterm_finishes_current_batch_then_stops(env):
    env.store.create_batch("b1", 40, "yolov8m")
    env.send("b1", env.upload("b1", 40))
    worker = env.worker(batch_size=16)
    env.runtime.on_infer = lambda: signal.raise_signal(signal.SIGTERM)

    previous = signal.getsignal(signal.SIGTERM)
    try:
        worker.run_forever()
    finally:
        signal.signal(signal.SIGTERM, previous)

    assert env.runtime.calls == [16]  # stopped after the batch in progress
    assert env.store.get_batch("b1")["completed"] == 16
    assert env.queue_count() == 24


def test_config_from_env():
    cfg = WorkerConfig.from_env(
        {
            "QUEUE_URL": "q",
            "DLQ_URL": "d",
            "BUCKET": "b",
            "TABLE": "t",
            "BATCH_SIZE": "32",
            "HALF": "true",
            "WORKER_ID": "gpu-1",
        }
    )
    assert (cfg.batch_size, cfg.half, cfg.worker_id, cfg.conf) == (32, True, "gpu-1", 0.25)
    assert (cfg.trt_prebuild, cfg.trt_cache_dir) == (True, "/tmp/trt-cache")
    off = WorkerConfig.from_env(
        {"QUEUE_URL": "q", "DLQ_URL": "d", "BUCKET": "b", "TABLE": "t",
         "TRT_PREBUILD": "0", "TRT_CACHE_DIR": "/tmp/trt-cache/ab-a"}
    )  # fmt: skip
    assert (off.trt_prebuild, off.trt_cache_dir) == (False, "/tmp/trt-cache/ab-a")
    with pytest.raises(ValueError, match="QUEUE_URL"):
        WorkerConfig.from_env({})
    with pytest.raises(ValueError, match="BATCH_SIZE"):
        WorkerConfig.from_env({"QUEUE_URL": "q", "DLQ_URL": "d", "BUCKET": "b", "TABLE": "t",
                               "BATCH_SIZE": "128"})  # fmt: skip


def test_worker_class_is_importable_without_model():
    assert Worker.__name__ == "Worker"


class FakeMonitor:
    def __init__(self):
        self.takes = 0

    def take(self):
        self.takes += 1
        return {"util_mean": 50.0, "samples": 4, "mem_mb": 1200}


def test_batch_log_has_wall_clock_times_and_gpu_window(env, caplog):
    env.store.create_batch("b1", 3, "yolov8m")
    env.send("b1", env.upload("b1", 3))
    ticks = iter([100.0, 101.5, 102.0])
    monitor = FakeMonitor()
    worker = Worker(
        env.config(),
        env.runtime,
        env.sqs,
        env.s3,
        env.store,
        wall=lambda: next(ticks),
        gpu_monitor=monitor,
    )

    with caplog.at_level("INFO", logger="detector.worker"):
        report = worker.run_once()

    assert (report.t_receive, report.t_written) == (100.0, 101.5)
    assert report.gpu == {"util_mean": 50.0, "samples": 4, "mem_mb": 1200}
    logged = json.loads(next(r.message for r in caplog.records if '"event": "batch"' in r.message))
    assert logged["t_receive"] <= logged["t_written"]
    assert logged["gpu"]["util_mean"] == 50.0

    # An idle poll discards the GPU window so idle time doesn't count toward the next batch.
    assert worker.run_once() is None
    assert monitor.takes == 2


def test_cpu_worker_logs_no_gpu_fields(env):
    env.store.create_batch("b1", 1, "yolov8m")
    env.send("b1", env.upload("b1", 1))
    report = env.worker().run_once()
    assert report.gpu is None
    assert report.t_receive <= report.t_written
