import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from detector.benchmark import (
    attach_map,
    bench_batch_id,
    best_gpu_batch_size,
    latest_per_setup,
    pipelined,
    plot_runtimes,
    plot_sweep,
    render_report,
    report_table,
    runtime_rows,
    setup_label,
)
from detector.messages import BATCH_ID_RE
from detector.ops import (
    HOURLY_USD,
    Workers,
    parse_log_events,
    run_metrics,
    run_worker_task,
    running_tasks,
    s5cmd_copy_commands,
    s5cmd_upload_commands,
    scaling_max,
    set_scaling_max,
    wait_for_alarm_clear,
)


def batch_line(t0, t1, processed, failed=0, gpu=None, **stages):
    return {
        "event": "batch",
        "processed": processed,
        "failed": failed,
        "skipped": 0,
        "t_receive": t0,
        "t_written": t1,
        "stages_ms": stages or {"inference": 100.0},
        "gpu": gpu,
    }


# Metrics ---------------------------------------------------------------------------


def test_run_metrics_times_first_receive_to_last_write():
    batches = [
        {"event": "startup", "device": "cuda"},
        batch_line(
            1000.0,
            1002.0,
            16,
            inference=200.0,
            decode=40.0,
            gpu={"util_mean": 60.0, "samples": 4, "mem_mb": 900},
        ),
        batch_line(
            1002.0,
            1005.0,
            15,
            failed=1,
            inference=220.0,
            decode=44.0,
            gpu={"util_mean": 90.0, "samples": 6, "mem_mb": 1100},
        ),
        batch_line(1005.0, 1005.5, 0),  # nothing processed (all skipped): ignored
    ]
    m = run_metrics(batches, hourly_usd=0.526)

    assert (m["images"], m["processed"], m["failed"], m["batches"]) == (32, 31, 1, 2)
    assert m["seconds"] == 5.0
    assert m["images_per_s"] == 6.4
    # 0.526 $/h * 5 s / 32 images * 1000
    assert m["cost_per_1000_usd"] == pytest.approx(0.526 * 5 / 3600 / 32 * 1000, abs=1e-5)
    assert m["stage_ms_per_batch"] == {"inference": 210.0, "decode": 42.0}
    assert m["stage_ms_per_image"]["inference"] == pytest.approx(420 / 32, abs=0.01)
    assert m["gpu_util_mean"] == 78.0  # sample-weighted: (60*4 + 90*6) / 10
    assert m["gpu_mem_mb_peak"] == 1100


def test_run_metrics_cpu_has_no_gpu_fields():
    m = run_metrics([batch_line(0.0, 1.0, 1)], hourly_usd=HOURLY_USD["c6i.2xlarge"])
    assert m["gpu_util_mean"] is None and m["gpu_mem_mb_peak"] is None


def test_run_metrics_needs_work():
    with pytest.raises(ValueError):
        run_metrics([{"event": "startup"}], hourly_usd=1.0)


def test_parse_log_events_reads_worker_json_lines():
    lines = [
        "2026-11-07 16:44:50,766 INFO worker fcab-1 started: pytorch",
        '2026-11-07 16:44:50,800 INFO {"event": "startup", "device": "cuda"}',
        '2026-11-07 16:45:20,956 INFO {"event": "batch", "processed": 1}',
        '2026-11-07 16:45:21,000 INFO {"event": "batch", "processed": ',  # truncated
    ]
    assert parse_log_events(lines) == [
        {"event": "startup", "device": "cuda"},
        {"event": "batch", "processed": 1},
    ]


# Ids and S3 commands --------------------------------------------------------------


def test_bench_batch_id_is_a_valid_batch_id():
    bid = bench_batch_id("gpu", 16, "fp32", now=datetime(2026, 11, 7, 17, 0, tzinfo=UTC))
    assert bid == "bench-gpu-bs16-fp32-20261107-170000"
    assert BATCH_ID_RE.match(bid)


def test_server_side_copy_into_input():
    cmds = s5cmd_copy_commands(["a.jpg", "b.jpg"], "bkt", "datasets/coco-val2017/", "bench-1")
    assert cmds.splitlines() == [
        "cp s3://bkt/datasets/coco-val2017/a.jpg s3://bkt/input/bench-1/a.jpg",
        "cp s3://bkt/datasets/coco-val2017/b.jpg s3://bkt/input/bench-1/b.jpg",
    ]
    with pytest.raises(ValueError):
        s5cmd_copy_commands(["a.jpg"], "bkt", "datasets", "bad id!")


def test_staging_never_goes_under_input():
    files = [Path("/tmp/x/000001.jpg")]
    assert s5cmd_upload_commands(files, "bkt", "datasets/coco-val2017") == (
        "cp /tmp/x/000001.jpg s3://bkt/datasets/coco-val2017/000001.jpg\n"
    )
    with pytest.raises(ValueError):
        s5cmd_upload_commands(files, "bkt", "input/staged")


# Scaling and tasks (stub clients) ----------------------------------------------------


class Recorder:
    def __init__(self, **responses):
        self.calls = []
        self.responses = responses

    def __getattr__(self, name):
        def call(**kwargs):
            self.calls.append((name, kwargs))
            return self.responses.get(name, {})

        return call


def test_cap_at_zero_also_stops_the_service():
    appscaling, ecs = Recorder(), Recorder()
    set_scaling_max(appscaling, ecs, "c", "svc", 0)
    ((name, kwargs),) = appscaling.calls
    assert name == "register_scalable_target"
    assert (kwargs["ResourceId"], kwargs["MinCapacity"], kwargs["MaxCapacity"]) == (
        "service/c/svc",
        0,
        0,
    )
    assert ecs.calls == [("update_service", {"cluster": "c", "service": "svc", "desiredCount": 0})]

    appscaling, ecs = Recorder(), Recorder()
    set_scaling_max(appscaling, ecs, "c", "svc", 2)
    assert appscaling.calls[0][1]["MaxCapacity"] == 2
    assert ecs.calls == []  # restoring doesn't touch the desired count


def test_scaling_max_reads_the_target():
    with_target = Recorder(describe_scalable_targets={"ScalableTargets": [{"MaxCapacity": 2}]})
    assert scaling_max(with_target, "c", "svc") == 2
    assert (
        scaling_max(Recorder(describe_scalable_targets={"ScalableTargets": []}), "c", "s") is None
    )


def test_run_worker_task_overrides_env_on_the_kinds_capacity_provider():
    ecs = Recorder(run_task={"tasks": [{"taskArn": "arn:aws:ecs:r:1:task/c/abc123"}]})
    workers = Workers("gpu", "svc", "asg", "/lg", "detector-worker-gpu-dev", "detector-gpu-cp-dev")
    arn = run_worker_task(ecs, "c", workers, {"BATCH_SIZE": "32"}, started_by="benchmark")
    assert arn.endswith("/abc123")
    ((_, kwargs),) = ecs.calls
    assert kwargs["taskDefinition"] == "detector-worker-gpu-dev"
    assert kwargs["capacityProviderStrategy"] == [
        {"capacityProvider": "detector-gpu-cp-dev", "weight": 1}
    ]
    (override,) = kwargs["overrides"]["containerOverrides"]
    assert override == {"name": "worker", "environment": [{"name": "BATCH_SIZE", "value": "32"}]}
    assert kwargs["startedBy"] == "benchmark"

    failing = Recorder(run_task={"tasks": [], "failures": [{"reason": "RESOURCE:GPU"}]})
    with pytest.raises(RuntimeError, match="RESOURCE:GPU"):
        run_worker_task(failing, "c", workers, {}, started_by="benchmark")


# Report ------------------------------------------------------------------------------


def record(
    kind,
    bs,
    ips,
    cost,
    created="2026-11-07T17:00:00+00:00",
    util=None,
    precision="fp32",
    runtime="pytorch",
    pipelined=None,
):
    rec = {
        "kind": kind,
        "runtime": runtime,
        "precision": precision,
        "batch_size": bs,
        "instance_type": "g4dn.xlarge" if kind == "gpu" else "c6i.2xlarge",
        "created_at": created,
        "metrics": {
            "images": 5000,
            "failed": 0,
            "seconds": 5000 / ips,
            "images_per_s": ips,
            "cost_per_1000_usd": cost,
            "gpu_util_mean": util,
            "stage_ms_per_image": {"decode": 4.0, "inference": 1000 / ips},
        },
    }
    if pipelined is not None:  # absent in runs from before milestone 6
        rec["pipelined"] = pipelined
    return rec


SWEEP = [
    record("gpu", 1, 20.0, 0.0073, util=40),
    record("gpu", 8, 60.0, 0.0024, util=85),
    record("gpu", 16, 64.0, 0.0023, util=90),
    record("cpu", 1, 3.0, 0.0315),
]


def test_latest_run_wins_and_rows_are_in_report_order():
    old = record("gpu", 16, 10.0, 0.01, created="2026-11-06T00:00:00+00:00")
    rows = latest_per_setup([*SWEEP, old])
    assert [(r["kind"], r["batch_size"]) for r in rows] == [
        ("cpu", 1),
        ("gpu", 1),
        ("gpu", 8),
        ("gpu", 16),
    ]
    assert rows[-1]["metrics"]["images_per_s"] == 64.0


def test_report_table_and_best_batch_size():
    rows = latest_per_setup(SWEEP)
    table = report_table(rows)
    assert "| ECS workers, CPU (`c6i.2xlarge`, PyTorch, unoptimized) | FP32 | 1 | 3.0 |" in table
    assert (
        "| ECS workers, GPU (`g4dn.xlarge`), PyTorch | FP32 | 16 | 64.0 | 1 min 18 s (5,000) |"
        in table
    )
    assert "| 90% | — | 0 |" in table
    assert best_gpu_batch_size(rows) == 16
    # FP16 rows aren't part of the sweep's "best".
    assert best_gpu_batch_size([*rows, record("gpu", 32, 99.0, 0.001, precision="fp16")]) == 16


def test_render_report_and_chart(tmp_path):
    rows = latest_per_setup(SWEEP)
    chart = tmp_path / "img" / "chart.png"
    assert plot_sweep(rows, chart)
    assert chart.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    text = render_report(rows, {"sweep": "img/chart.png"})
    assert "![" in text and "img/chart.png" in text
    assert "**batch size 16**" in text
    assert "| GPU PyTorch FP32 | 8 |" in text  # stage table
    assert not plot_sweep([record("cpu", 1, 3.0, 0.03)], tmp_path / "none.png")


def test_record_is_json_serializable():
    json.dumps(SWEEP)


class FakeCloudWatch:
    def __init__(self, states):
        self.states = list(states)

    def describe_alarms(self, AlarmNames):
        state = self.states.pop(0) if len(self.states) > 1 else self.states[0]
        return {"MetricAlarms": [{"AlarmName": AlarmNames[0], "StateValue": state}]}


def test_wait_for_alarm_clear():
    cw = FakeCloudWatch(["ALARM", "ALARM", "OK"])
    assert wait_for_alarm_clear(cw, "a", sleep=lambda _: None)
    assert cw.states == ["OK"]
    stuck = FakeCloudWatch(["ALARM"])
    assert not wait_for_alarm_clear(stuck, "a", timeout_s=0, sleep=lambda _: None)


def test_bench_batch_id_variants():
    now = datetime(2026, 11, 9, 8, 0, tzinfo=UTC)
    bid = bench_batch_id(
        "gpu", 64, "fp16", runtime="onnx-trt", pipelined=True, eval_run=True, now=now
    )
    assert bid == "bench-gpu-onnx-trt-bs64-fp16-pipe-eval-20261109-080000"
    assert BATCH_ID_RE.match(bid)


PIPELINED = [
    record("gpu", 32, 45.0, 0.0032, pipelined=True, util=80),
    record("gpu", 64, 50.0, 0.0029, pipelined=True, util=88),
    record("gpu", 64, 70.0, 0.0021, precision="fp16", pipelined=True),
    record("gpu", 64, 90.0, 0.0016, precision="fp16", runtime="onnxruntime-trt", pipelined=True),
]


def test_pipelined_rows_are_separate_setups_and_best():
    rows = latest_per_setup([*SWEEP, *PIPELINED])
    labels = [setup_label(r) for r in rows]
    assert labels[0].startswith("ECS workers, CPU")
    assert "ECS workers, GPU (`g4dn.xlarge`), ONNX Runtime (TensorRT), pipelined" in labels
    # Serial rows come before pipelined ones; a pre-milestone-6 run counts as serial.
    assert [pipelined(r) for r in rows] == sorted(pipelined(r) for r in rows)
    assert best_gpu_batch_size(rows) == 16
    assert best_gpu_batch_size(rows, pipelined_rows=True) == 64
    assert {(r["runtime"], r["precision"]) for r in runtime_rows(rows)} == {
        ("pytorch", "fp32"),
        ("pytorch", "fp16"),
        ("onnxruntime-trt", "fp16"),
    }


def test_attach_map_uses_newest_eval_per_setup():
    evals = [
        {
            "kind": "gpu",
            "runtime": "pytorch",
            "precision": "fp32",
            "created_at": "1",
            "metrics": {"map50_95": 0.4},
        },
        {
            "kind": "gpu",
            "runtime": "pytorch",
            "precision": "fp32",
            "created_at": "2",
            "metrics": {"map50_95": 0.5025},
        },
    ]
    rows = attach_map([record("gpu", 8, 1, 1), record("gpu", 8, 1, 1, precision="fp16")], evals)
    assert [r.get("map50_95") for r in rows] == [0.5025, None]
    assert "| 0.5025 |" in report_table(rows)


def test_runtime_chart(tmp_path):
    rows = latest_per_setup([*SWEEP, *PIPELINED])
    assert plot_runtimes(rows, tmp_path / "runtimes.png")
    assert plot_sweep(rows, tmp_path / "sweep.png")  # two series: serial and pipelined
    assert not plot_runtimes(latest_per_setup(SWEEP[:1]), tmp_path / "none.png")


class FakeEcs:
    """list_tasks by desired status; describe_tasks reports each task's last status."""

    def __init__(self, tasks):  # arn -> (desired, last)
        self.tasks = tasks

    def get_paginator(self, name):
        tasks = self.tasks

        class Pager:
            def paginate(self, cluster, desiredStatus):
                yield {"taskArns": [a for a, (d, _) in tasks.items() if d == desiredStatus]}

        return Pager()

    def describe_tasks(self, cluster, tasks):
        return {"tasks": [{"taskArn": a, "lastStatus": self.tasks[a][1]} for a in tasks]}


def test_a_stopping_task_still_counts_as_running():
    ecs = FakeEcs(
        {
            "running": ("RUNNING", "RUNNING"),
            "starting": ("RUNNING", "PROVISIONING"),
            "stopping": ("STOPPED", "RUNNING"),  # told to stop, still finishing its batch
            "gone": ("STOPPED", "STOPPED"),
        }
    )
    assert sorted(running_tasks(ecs, "c")) == ["running", "starting", "stopping"]


def test_report_appends_notes():
    rows = latest_per_setup(SWEEP)
    text = render_report(rows, {}, notes="## Notes\n\nTensorRT is an open issue.\n")
    assert text.rstrip().endswith("TensorRT is an open issue.")
    assert "## Notes" not in render_report(rows, {})


def test_report_explains_worker_modes_before_the_results():
    text = render_report(latest_per_setup(SWEEP), {})
    assert text.index("## One step at a time vs. pipelined") < text.index("## Results")
