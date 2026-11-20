from types import SimpleNamespace

import pytest

from detector.inference_core import DecodeError, decode
from detector.ops import (
    credentials_env,
    list_images,
    load_deployment,
    s5cmd_commands,
    wait_for_batch,
    write_corrupt_files,
)
from detector.store import ImageResult


class FakeCloudFormation:
    OUTPUTS = {
        "DetectorBatch-dev": {
            "BucketName": "bkt",
            "QueueUrl": "q",
            "DlqUrl": "d",
            "ClusterName": "detector-cluster-dev",
            "CpuServiceName": "detector-worker-cpu-dev",
            "CpuAsgName": "asg",
            "CpuLogGroup": "/detector/dev/worker-cpu",
            "CpuTaskFamily": "detector-worker-cpu-dev",
            "CpuCapacityProvider": "detector-cpu-cp-dev",
            "GpuServiceName": "detector-worker-gpu-dev",
            "GpuAsgName": "gasg",
            "GpuLogGroup": "/detector/dev/worker-gpu",
            "GpuTaskFamily": "detector-worker-gpu-dev",
            "GpuCapacityProvider": "detector-gpu-cp-dev",
        },
        "DetectorData-dev": {"TableName": "detector-results-dev"},
    }

    def describe_stacks(self, StackName):
        outs = [{"OutputKey": k, "OutputValue": v} for k, v in self.OUTPUTS[StackName].items()]
        return {"Stacks": [{"Outputs": outs}]}


def test_load_deployment_reads_both_stacks():
    d = load_deployment(FakeCloudFormation(), "dev")
    assert (d.bucket, d.table) == ("bkt", "detector-results-dev")
    assert set(d.workers) == {"cpu", "gpu"}
    gpu = d.workers["gpu"]
    assert (gpu.service, gpu.task_family, gpu.capacity_provider, gpu.log_group) == (
        "detector-worker-gpu-dev", "detector-worker-gpu-dev", "detector-gpu-cp-dev",
        "/detector/dev/worker-gpu",
    )  # fmt: skip


def test_load_deployment_before_gpu_workers_existed():
    cf = FakeCloudFormation()
    cf.OUTPUTS = {
        **cf.OUTPUTS,
        "DetectorBatch-dev": {
            k: v for k, v in cf.OUTPUTS["DetectorBatch-dev"].items() if not k.startswith("Gpu")
        },
    }
    assert set(load_deployment(cf, "dev").workers) == {"cpu"}


def test_list_images_sorted_and_limited(tmp_path):
    for name in ["b.jpg", "a.png", "c.txt", "d.JPG"]:
        (tmp_path / name).write_bytes(b"x")
    assert [p.name for p in list_images(tmp_path)] == ["a.png", "b.jpg", "d.JPG"]
    assert [p.name for p in list_images(tmp_path, limit=2)] == ["a.png", "b.jpg"]


def test_corrupt_files_do_not_decode(tmp_path):
    paths = write_corrupt_files(tmp_path, 3)
    assert [p.name for p in paths] == ["corrupt-000.jpg", "corrupt-001.jpg", "corrupt-002.jpg"]
    for p in paths:
        with pytest.raises(DecodeError):
            decode(p.read_bytes())


def test_s5cmd_commands(tmp_path):
    files = [tmp_path / "000000000139.jpg", tmp_path / "corrupt-000.jpg"]
    text = s5cmd_commands(files, "bkt", "coco-1")
    assert text.splitlines() == [
        f"cp {files[0]} s3://bkt/input/coco-1/000000000139.jpg",
        f"cp {files[1]} s3://bkt/input/coco-1/corrupt-000.jpg",
    ]
    with pytest.raises(ValueError, match="batch id"):
        s5cmd_commands(files, "bkt", "bad id")


def test_credentials_env_includes_session_token():
    creds = SimpleNamespace(access_key="AK", secret_key="SK", token="TOK")
    session = SimpleNamespace(
        region_name="us-west-2",
        get_credentials=lambda: SimpleNamespace(get_frozen_credentials=lambda: creds),
    )
    assert credentials_env(session) == {
        "AWS_ACCESS_KEY_ID": "AK",
        "AWS_SECRET_ACCESS_KEY": "SK",
        "AWS_SESSION_TOKEN": "TOK",
        "AWS_REGION": "us-west-2",
    }


def test_wait_for_batch_until_every_image_is_accounted_for(env):
    env.store.create_batch("b1", 3, "yolov8m")
    meta = {"worker_id": "w", "runtime": "fake", "precision": "fp32"}
    steps = iter(
        [
            lambda: env.store.commit("b1", [ImageResult("a.jpg", "done")], **meta),
            lambda: env.store.commit(
                "b1", [ImageResult("b.jpg", "done"), ImageResult.failed("c.jpg", "x")], **meta
            ),
        ]
    )
    seen = []
    batch = wait_for_batch(
        env.store,
        "b1",
        poll_s=0,
        on_progress=lambda b: seen.append(b["completed"] + b["failed"]),
        sleep=lambda _: next(steps)(),
    )
    assert seen == [0, 1, 3]
    assert (batch["completed"], batch["failed"]) == (2, 1)


def test_wait_for_batch_times_out(env):
    env.store.create_batch("b1", 1, "yolov8m")
    with pytest.raises(TimeoutError):
        wait_for_batch(env.store, "b1", timeout_s=0, sleep=lambda _: None)


def test_wait_for_missing_batch(env):
    with pytest.raises(LookupError):
        wait_for_batch(env.store, "nope", sleep=lambda _: None)
