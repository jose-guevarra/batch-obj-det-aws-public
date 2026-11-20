"""Benchmark rows: run records, mAP results, the report tables and charts.

scripts/benchmark.py runs the AWS side (fill the queue, start one worker, read its logs);
this module holds the parts that don't touch AWS, so they're unit-tested.

A run record is one JSON file in benchmarks/runs/, one report row. An eval record is one
JSON file in benchmarks/evals/: the mAP of one (kind, runtime, precision), joined onto every
row with that setup (batch size and pipelining don't change the model's output).
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SWEEP_BATCH_SIZES = (1, 8, 16, 32, 64)
STAGES = (
    "receive",
    "skip_check",
    "download",  # serial worker before milestone 6
    "decode",
    "load",  # download + decode + letterbox in the pool
    "preprocess",
    "wait_input",  # pipelined: inference waiting for the next prepared batch
    "inference",
    "postprocess",
    "write",
    "ack",
)
RUNTIME_LABELS = {
    "pytorch": "PyTorch",
    "onnxruntime-cuda": "ONNX Runtime (CUDA)",
    "onnxruntime-trt": "ONNX Runtime (TensorRT)",
}
RUNTIME_ORDER = list(RUNTIME_LABELS)

# Chart colors (validated: lightness band, chroma, CVD separation, contrast on the light
# surface). SERIES_2 is only used next to SERIES.
SERIES = "#2a78d6"
SERIES_2 = "#eb6834"
MUTED = "#898781"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"


def bench_batch_id(
    kind: str,
    batch_size: int,
    precision: str,
    *,
    runtime: str = "pytorch",
    pipelined: bool = False,
    eval_run: bool = False,
    now: datetime | None = None,
) -> str:
    """bench-gpu[-onnx-trt]-bs64-fp16[-pipe][-eval]-20261108-170000 (a valid batch id)."""
    stamp = (now or datetime.now(UTC)).strftime("%Y%m%d-%H%M%S")
    parts = ["bench", kind]
    if runtime != "pytorch":
        parts.append(runtime)
    parts += [f"bs{batch_size}", precision]
    if pipelined:
        parts.append("pipe")
    if eval_run:
        parts.append("eval")
    return "-".join([*parts, stamp])


def pipelined(record: dict) -> bool:
    return bool(record.get("pipelined", False))  # runs before milestone 6 were serial


def setup_key(record: dict) -> tuple:
    return (
        record["kind"],
        record["runtime"],
        record["precision"],
        record["batch_size"],
        pipelined(record),
    )


def _order(r: dict) -> tuple:
    runtime = RUNTIME_ORDER.index(r["runtime"]) if r["runtime"] in RUNTIME_ORDER else 9
    return (
        {"cpu": 0, "gpu": 1}.get(r["kind"], 9),
        pipelined(r),
        runtime,
        r["precision"] != "fp32",
        r["batch_size"],
    )


def latest_per_setup(records: Iterable[dict]) -> list[dict]:
    """The newest record for each setup, in report order."""
    latest: dict[tuple, dict] = {}
    for r in records:
        key = setup_key(r)
        if key not in latest or r["created_at"] > latest[key]["created_at"]:
            latest[key] = r
    return sorted(latest.values(), key=_order)


def load_records(directory: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(Path(directory).glob("*.json"))]


def attach_map(records: Sequence[dict], evals: Iterable[dict]) -> list[dict]:
    """Copies of records with map50_95 from the newest eval of the same kind/runtime/precision."""
    best: dict[tuple, dict] = {}
    for e in evals:
        key = (e["kind"], e["runtime"], e["precision"])
        if key not in best or e["created_at"] > best[key]["created_at"]:
            best[key] = e
    out = []
    for r in records:
        e = best.get((r["kind"], r["runtime"], r["precision"]))
        out.append({**r, "map50_95": e["metrics"]["map50_95"]} if e else dict(r))
    return out


def setup_label(r: dict) -> str:
    if r["kind"] == "cpu":
        return f"ECS workers, CPU (`{r['instance_type']}`, PyTorch, unoptimized)"
    runtime = RUNTIME_LABELS.get(r["runtime"], r["runtime"])
    label = f"ECS workers, GPU (`{r['instance_type']}`), {runtime}"
    return label + (", pipelined" if pipelined(r) else "")


def short_label(r: dict) -> str:
    runtime = RUNTIME_LABELS.get(r["runtime"], r["runtime"])
    mode = ", pipelined" if pipelined(r) else ""
    return f"{r['kind'].upper()} {runtime} {r['precision'].upper()}{mode}"


def _fmt(value: Any, spec: str, none: str = "—") -> str:
    return none if value is None else format(value, spec)


def _duration(seconds: float) -> str:
    m, s = divmod(round(seconds), 60)
    return f"{m} min {s:02d} s" if m else f"{s} s"


def report_table(records: Sequence[dict]) -> str:
    """The design doc's benchmark table, one row per record."""
    head = (
        "| Setup | Precision | Batch size | Images/sec | Time for N images | "
        "Cost per 1,000 images | Avg GPU utilization | mAP50-95 | Failed images |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for r in records:
        m = r["metrics"]
        util = m["gpu_util_mean"]
        util_text = "—" if util is None else f"{util:.0f}%"
        rows.append(
            f"| {setup_label(r)} | {r['precision'].upper()} | {r['batch_size']} "
            f"| {_fmt(m['images_per_s'], '.1f')} | {_duration(m['seconds'])} ({m['images']:,}) "
            f"| ${m['cost_per_1000_usd']:.4f} | {util_text} "
            f"| {_fmt(r.get('map50_95'), '.4f')} | {m['failed']} |"
        )
    return head + "\n".join(rows) + "\n"


def stage_table(records: Sequence[dict]) -> str:
    """Mean milliseconds per image in each stage, per row (where the time goes).

    In pipelined rows the stages overlap, so they add up to more than the time per image.
    """
    stages = [s for s in STAGES if any(s in r["metrics"]["stage_ms_per_image"] for r in records)]
    head = "| Setup | Batch size | " + " | ".join(stages) + " |\n"
    head += "| --- | --- |" + " --- |" * len(stages) + "\n"
    rows = []
    for r in records:
        per = r["metrics"]["stage_ms_per_image"]
        cells = " | ".join(_fmt(per.get(s), ".1f") for s in stages)
        rows.append(f"| {short_label(r)} | {r['batch_size']} | {cells} |")
    return head + "\n".join(rows) + "\n"


def _sweep(records: Sequence[dict], pipelined_rows: bool) -> list[dict]:
    """GPU PyTorch FP32 rows (the batch-size sweep) of one mode, by batch size."""
    return sorted(
        (
            r
            for r in records
            if (r["kind"], r["runtime"], r["precision"]) == ("gpu", "pytorch", "fp32")
            and pipelined(r) == pipelined_rows
        ),
        key=lambda r: r["batch_size"],
    )


def best_gpu_batch_size(records: Sequence[dict], pipelined_rows: bool = False) -> int | None:
    """Lowest cost per 1,000 images among the GPU PyTorch FP32 rows ("best from sweep")."""
    sweep = _sweep(records, pipelined_rows)
    if not sweep:
        return None
    return min(sweep, key=lambda r: r["metrics"]["cost_per_1000_usd"])["batch_size"]


def runtime_rows(records: Sequence[dict]) -> list[dict]:
    """The runtime comparison at the best batch size: every GPU setup run at it.

    The batch size is the pipelined sweep's best (or the serial one's, before pipelining).
    """
    bs = best_gpu_batch_size(records, True) or best_gpu_batch_size(records, False)
    return [r for r in records if r["kind"] == "gpu" and r["batch_size"] == bs]


WORKER_MODES = """## One step at a time vs. pipelined

The rows and charts compare two ways the same worker can process a batch. Both run the same
model on the same images, so they produce the same detections and the same mAP. Only the timing
differs. Each batch goes through three steps:

- **Prepare:** receive messages from SQS, skip images that already have results, then download,
  decode and letterbox each image (on the CPU).
- **Infer:** run the model on the GPU, then postprocess (NMS, boxes back to image coordinates).
- **Finish:** write the results part file to S3, commit results and counters to DynamoDB, delete
  the messages.

**One step at a time** (`PIPELINE=0`, used for the batch-size sweep and the CPU baseline) runs
the three steps in sequence. The GPU works only during *infer*; during *prepare* and *finish* it
sits idle.

**Pipelined** (`PIPELINE=1`, the GPU service's default) runs each step in its own thread, with a
queue holding one batch between steps. While the GPU infers batch N, the CPU is already
preparing batch N+1, and batch N−1 is being written. At most three batches are in flight at
once.

**One step at a time:** the GPU (green) is busy only during each batch's *infer*.

```mermaid
---
displayMode: compact
config:
  gantt:
    leftPadding: 150
    rightPadding: 90
    fontSize: 16
    sectionFontSize: 16
    barHeight: 30
    barGap: 6
  themeCSS: '.tick text { font-size: 14px; }'
  themeVariables:
    critBkgColor: '#2e7d32'
    critBorderColor: '#1b5e20'
---
gantt
    dateFormat x
    axisFormat t+%-L
    tickInterval 2millisecond
    section Time
        begin  :milestone, 0, 0
        finish :milestone, 12, 12
    section Worker
        batch 1 · prepare :0, 2
        batch 1 · infer :crit, 2, 4
        batch 1 · finish :4, 6
        batch 2 · prepare :6, 8
        batch 2 · infer :crit, 8, 10
        batch 2 · finish :10, 12
```

**Pipelined:** each step runs in its own thread, so the GPU goes straight from one batch to the
next.

```mermaid
---
displayMode: compact
config:
  gantt:
    leftPadding: 150
    rightPadding: 90
    fontSize: 16
    sectionFontSize: 16
    barHeight: 30
    barGap: 6
  themeCSS: '.tick text { font-size: 14px; }'
  themeVariables:
    critBkgColor: '#2e7d32'
    critBorderColor: '#1b5e20'
---
gantt
    dateFormat x
    axisFormat t+%-L
    tickInterval 2millisecond
    section Time
        begin  :milestone, 0, 0
        finish :milestone, 12, 12
    section Prepare thread
        batch 1 · prepare :0, 2
        batch 2 · prepare :2, 4
        batch 3 · prepare :4, 6
        batch 4 · prepare :6, 8
        batch 5 · prepare :8, 10
        batch 6 · prepare :10, 12
    section Infer thread
        batch 1 · infer :crit, 2, 4
        batch 2 · infer :crit, 4, 6
        batch 3 · infer :crit, 6, 8
        batch 4 · infer :crit, 8, 10
        batch 5 · infer :crit, 10, 12
    section Finish thread
        batch 1 · finish :4, 6
        batch 2 · finish :6, 8
        batch 3 · finish :8, 10
        batch 4 · finish :10, 12
```

In the same time, one step at a time finishes 2 batches and the pipelined worker finishes 4,
because the GPU stays busy instead of waiting on prepare and finish.

The total time for a batch's own steps barely changes; what changes is that they overlap. A
pipelined worker's throughput is set by its slowest step, not the sum of all three, so
pipelining helps most when prepare and finish take about as long as inference.

To see whether it worked, check **Avg GPU utilization** in the results and `wait_input` in the
stage table. `wait_input` is how long the GPU sat waiting for the next prepared batch. It's only
measured in pipelined rows; in one-step-at-a-time rows that idle time is simply the prepare and
finish stages.

In the stage table, a pipelined row's stage times add up to more than its time per image,
because the stages run at the same time.
"""


def render_report(records: Sequence[dict], charts: dict[str, str], notes: str = "") -> str:
    """charts: {"sweep": relative path, "runtimes": relative path} for the charts that exist;
    notes: Markdown appended at the end (benchmarks/notes.md), for findings tables can't show."""
    parts = [
        "# Benchmark results\n",
        "Generated by `scripts/benchmark.py report` from `benchmarks/runs/` (timed rows) and "
        "`benchmarks/evals/` (mAP). Every row is one worker on one instance, YOLOv8m on 5,000 "
        "COCO val2017 images, timed from the first message received to the last result "
        "written (instance start, image pull and model warm-up excluded). Cost counts only "
        "On-Demand instance time.\n",
        WORKER_MODES,
    ]
    if "sweep" in charts:
        parts.append(
            f"![Images per second and cost per 1,000 images by batch size]({charts['sweep']})\n"
        )
    if "runtimes" in charts:
        parts.append(
            f"![Images per second and cost per 1,000 images by runtime]({charts['runtimes']})\n"
        )
    parts += ["## Results\n", report_table(records)]
    for mode, name in ((False, "one step at a time"), (True, "pipelined")):
        best = best_gpu_batch_size(records, mode)
        if best is not None:
            parts.append(
                f"Best from sweep, {name} (lowest cost per 1,000 images, GPU PyTorch FP32): "
                f"**batch size {best}**.\n"
            )
    parts += ["## Time per image by stage (ms)\n", stage_table(records)]
    parts.append(
        "`receive` includes waiting for SQS; `inference` is synchronized with the GPU, so it's "
        "the real compute time. In pipelined rows the stages run in parallel threads, so they "
        "add up to more than the time per image; `wait_input` is how long the GPU waited for "
        "the next batch. mAP50-95 comes from separate eval runs (confidence 0.001, multi-label, "
        "as Ultralytics validates) and is scored only where the runtime or precision changes.\n"
    )
    if notes.strip():
        parts.append(notes)
    return "\n\n".join(p.strip("\n") for p in parts) + "\n"


# Charts ------------------------------------------------------------------------------


def _style(ax, title: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=TEXT)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.set_axisbelow(True)


def _finish(fig, axes, title: str, path: Path) -> None:
    import matplotlib.pyplot as plt

    fig.suptitle(title, fontsize=12, color=TEXT)
    fig.tight_layout()
    fig.texts[0].set(x=axes[0].get_position().x0, ha="left")  # align with the panel titles
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


PANELS = [
    ("Images per second", "images_per_s", "{:.0f}", max),
    ("Cost per 1,000 images (USD)", "cost_per_1000_usd", "${:.4f}", min),
]


def plot_sweep(records: Sequence[dict], path: Path) -> bool:
    """Images/s and cost per 1,000 by batch size: one panel per measure (no dual axis).

    One line per worker mode (one step at a time, pipelined) that has a sweep; the CPU row,
    if present, is a dashed reference line. Returns False if there's no sweep.
    """
    series = [
        (name, color, rows)
        for name, color, rows in (
            ("One step at a time", SERIES, _sweep(records, False)),
            ("Pipelined", SERIES_2, _sweep(records, True)),
        )
        if rows
    ]
    if not series:
        return False
    cpu = next((r for r in records if r["kind"] == "cpu"), None)
    plt = _pyplot()

    sizes = sorted({r["batch_size"] for _, _, rows in series for r in rows})
    x_of = {s: i for i, s in enumerate(sizes)}  # batch sizes as evenly spaced categories
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=150, facecolor=SURFACE)
    for ax, (title, key, fmt, pick) in zip(axes, PANELS, strict=True):
        _style(ax, title)
        top = 0.0
        first: dict[int, float] = {}  # the first series' values, to keep labels apart
        for i, (name, color, rows) in enumerate(series):
            xs = [x_of[r["batch_size"]] for r in rows]
            ys = [r["metrics"][key] for r in rows]
            top = max(top, *ys)
            if i == 0:
                first = dict(zip(xs, ys, strict=True))
            ax.plot(
                xs,
                ys,
                color=color,
                linewidth=2,
                marker="o",
                markersize=7,
                markeredgecolor=SURFACE,
                markeredgewidth=2,
                zorder=3,
                label=name,
            )
            best = ys.index(pick(ys))  # label only each series' best point
            # Above the point, unless it sits below the first series there (then below).
            below = i > 0 and ys[best] < first.get(xs[best], float("-inf"))
            ax.annotate(
                fmt.format(ys[best]),
                (xs[best], ys[best]),
                textcoords="offset points",
                xytext=(0, -15 if below else 9),
                ha="center",
                fontsize=9,
                color=TEXT,
            )
        # The CPU reference line only when it fits without flattening the GPU curves.
        ref = cpu["metrics"][key] if cpu else None
        if ref is not None and ref > 3 * top:
            ax.text(
                1.0,
                1.0,
                f"CPU baseline {fmt.format(ref)} (off scale)",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
            ref = None
        if ref is not None:
            ax.axhline(ref, color=MUTED, linewidth=1.5, linestyle=(0, (4, 3)), zorder=2)
            ax.annotate(
                f"CPU baseline {fmt.format(ref)}",
                (len(sizes) - 1, ref),
                textcoords="offset points",
                xytext=(0, 5),
                ha="right",
                fontsize=8,
                color=TEXT_SECONDARY,
            )
        ax.set_xticks(range(len(sizes)), [str(s) for s in sizes])
        ax.set_xlabel("Batch size", fontsize=9, color=TEXT_SECONDARY)
        ax.set_ylim(bottom=0, top=max(top, ref or 0) * 1.18)
        if key == "cost_per_1000_usd":
            ax.yaxis.set_major_formatter(lambda v, _: f"${v:.3f}")
        ax.grid(axis="y", color=GRID, linewidth=0.8)
    if len(series) > 1:
        axes[0].legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY, loc="upper left")
    gpu = series[0][2][0]["instance_type"]
    _finish(fig, axes, f"YOLOv8m PyTorch FP32 on {gpu} (NVIDIA T4), one worker", path)
    return True


def plot_runtimes(records: Sequence[dict], path: Path) -> bool:
    """Images/s and cost per 1,000 for each GPU setup at the best batch size (bars).

    One series, so one color; every bar is labeled with its value (few bars, and the values
    are the point). Returns False with fewer than two setups to compare.
    """
    rows = runtime_rows(records)
    if len(rows) < 2:
        return False
    plt = _pyplot()
    labels = [
        f"{RUNTIME_LABELS.get(r['runtime'], r['runtime'])} {r['precision'].upper()}"
        + ("" if pipelined(r) else ", one step at a time")
        for r in rows
    ]
    ys = list(range(len(rows)))[::-1]  # first row at the top
    fig, axes = plt.subplots(
        1, 2, figsize=(10, 0.55 * len(rows) + 1.6), dpi=150, facecolor=SURFACE, sharey=True
    )
    for ax, (title, key, fmt, _) in zip(axes, PANELS, strict=True):
        _style(ax, title)
        values = [r["metrics"][key] for r in rows]
        ax.barh(ys, values, height=0.6, color=SERIES, zorder=3)
        for y, v in zip(ys, values, strict=True):
            ax.annotate(
                fmt.format(v),
                (v, y),
                textcoords="offset points",
                xytext=(4, 0),
                va="center",
                fontsize=8,
                color=TEXT,
            )
        ax.set_xlim(0, max(values) * 1.22)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        if key == "cost_per_1000_usd":
            ax.xaxis.set_major_formatter(lambda v, _: f"${v:.3f}")
    axes[0].set_yticks(ys, labels, fontsize=8, color=TEXT)
    gpu, bs = rows[0]["instance_type"], rows[0]["batch_size"]
    _finish(fig, axes, f"YOLOv8m on {gpu} (NVIDIA T4) at batch {bs}, one worker", path)
    return True
