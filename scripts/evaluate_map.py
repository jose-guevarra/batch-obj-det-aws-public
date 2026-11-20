#!/usr/bin/env python3
"""Score YOLOv8m on COCO val2017 with the shared inference core.

    uv run python scripts/evaluate_map.py --limit 500
    uv run python scripts/evaluate_map.py --device cpu --batch-size 8

Defaults match Ultralytics validation (conf 0.001, IoU 0.7, max 300, multi-label), so the
full-set mAP50-95 is comparable to the published 50.2. Results go to results/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from detector.evaluation import EvalSettings, run_eval
from detector.runtimes import TorchRuntime, resolve_weights

ROOT = Path(__file__).resolve().parents[1]
COCO = ROOT / "data/datasets/coco"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--images", type=Path, default=COCO / "val2017")
    p.add_argument("--annotations", type=Path, default=COCO / "annotations/instances_val2017.json")
    p.add_argument("--model", default="yolov8m")
    p.add_argument("--device", default="auto", help="auto, cpu, mps or cuda")
    p.add_argument("--half", action="store_true", help="FP16 (GPU only)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--limit", type=int, default=None, help="first N images (sorted)")
    p.add_argument("--conf", type=float, default=0.001)
    p.add_argument("--iou", type=float, default=0.7)
    p.add_argument("--max-det", type=int, default=300)
    p.add_argument("--single-label", action="store_true", help="best class only per anchor")
    p.add_argument("--out", type=Path, default=None, help="summary JSON path")
    args = p.parse_args()

    runtime = TorchRuntime(resolve_weights(args.model), device=args.device, half=args.half)
    settings = EvalSettings(
        batch_size=args.batch_size,
        limit=args.limit,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        multi_label=not args.single_label,
    )
    summary, detections = run_eval(runtime, args.images, args.annotations, settings)

    out = args.out or ROOT / "results" / (
        f"eval-{runtime.name}-{runtime.device}-{runtime.precision}-bs{args.batch_size}"
        f"-n{summary['images']}.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n")
    out.with_suffix(".detections.json").write_text(json.dumps(detections))

    m = summary["metrics"]
    print(f"\n{runtime.name} on {runtime.device} ({runtime.precision}), batch {args.batch_size}")
    print(f"images: {summary['processed']}/{summary['images']}  failed: {len(summary['failed'])}")
    print(f"mAP50-95: {m['map50_95']:.4f}   mAP50: {m['map50']:.4f}")
    print(f"throughput: {summary['images_per_s']} images/s ({summary['elapsed_s']} s)")
    print("per batch (ms):")
    for stage, t in summary["stages"].items():
        print(f"  {stage:<12} {t['per_batch_ms']:>9.2f}")
    print(f"\nsummary: {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
