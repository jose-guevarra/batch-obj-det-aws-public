"""Score mAP50-95 on COCO val2017.

Two sources of detections, scored the same way:
  - run_eval(): a runtime over the local images (any machine);
  - part_lines_to_coco(): the result part files of an AWS eval run (benchmark.py --eval),
    so runtimes that need the GPU image (ONNX Runtime CUDA / TensorRT) get scored too.
"""

from __future__ import annotations

import contextlib
import io
import json
import platform
import time
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from detector.inference_core import COCO80_TO_91, StageTimer, postprocess, preprocess
from detector.runtimes import Runtime


@dataclass(frozen=True)
class EvalSettings:
    """Defaults match Ultralytics validation, so mAP is comparable to the published 50.2."""

    batch_size: int = 16
    limit: int | None = None
    conf: float = 0.001
    iou: float = 0.7
    max_det: int = 300
    multi_label: bool = True


def list_images(images_dir: Path, limit: int | None = None) -> list[Path]:
    """Sorted, so a --limit subset is the same every run."""
    paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in {".jpg", ".png"})
    return paths[:limit] if limit else paths


def run_eval(
    runtime: Runtime, images_dir: Path, annotations: Path, settings: EvalSettings
) -> tuple[dict, list[dict]]:
    """Returns (summary, COCO-format detections)."""
    paths = list_images(images_dir, settings.limit)
    if not paths:
        raise FileNotFoundError(f"no images in {images_dir}")

    timer = StageTimer(sync=runtime.sync)
    detections: list[dict] = []
    failures: dict[str, str] = {}
    batches = 0

    start = time.perf_counter()
    for i in range(0, len(paths), settings.batch_size):
        chunk = paths[i : i + settings.batch_size]
        with timer.stage("read"):
            raw = [p.read_bytes() for p in chunk]
        pre = preprocess(raw, timer=timer)
        for idx, err in pre.failures.items():
            failures[chunk[idx].name] = str(err)
        if pre.batch is None:
            continue
        with timer.stage("inference"):
            out = runtime.infer(pre.batch)
        with timer.stage("postprocess"):
            dets = postprocess(
                out,
                pre.metas,
                conf=settings.conf,
                iou=settings.iou,
                max_det=settings.max_det,
                multi_label=settings.multi_label,
            )
        batches += 1
        for idx, det in zip(pre.indices, dets, strict=True):
            detections += coco_detections(
                int(chunk[idx].stem), zip(det.boxes, det.scores, det.class_ids, strict=True)
            )
    elapsed = time.perf_counter() - start

    image_ids = [int(p.stem) for p in paths]
    metrics = coco_map(annotations, detections, image_ids)
    processed = len(paths) - len(failures)

    import ultralytics

    summary = {
        "runtime": runtime.name,
        "device": runtime.device,
        "precision": runtime.precision,
        "settings": asdict(settings),
        "images": len(paths),
        "processed": processed,
        "failed": failures,
        "detections": len(detections),
        "metrics": metrics,
        "elapsed_s": round(elapsed, 3),
        "images_per_s": round(processed / elapsed, 2),
        "stages": timer.summary(max(batches, 1)),
        "versions": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "ultralytics": ultralytics.__version__,
        },
    }
    return summary, detections


def coco_detections(image_id: int, dets: Iterable[tuple]) -> list[dict]:
    """(xyxy box, score, class 0-79) per detection -> COCO result records (xywh, 91 ids)."""
    out = []
    for (x1, y1, x2, y2), score, cls in dets:
        out.append(
            {
                "image_id": image_id,
                "category_id": COCO80_TO_91[int(cls)],
                "bbox": [round(float(v), 2) for v in (x1, y1, x2 - x1, y2 - y1)],
                "score": round(float(score), 5),
            }
        )
    return out


def part_lines_to_coco(lines: Iterable[str]) -> tuple[list[dict], list[int], list[str]]:
    """Worker part-file lines (JSON) -> (COCO detections, image ids scored, failed names).

    The image id is the COCO file name's number (000000000139.jpg -> 139). Failed images
    (corrupt, missing) are returned separately and left out of the scored ids.
    """
    detections: list[dict] = []
    image_ids: list[int] = []
    failed: list[str] = []
    for line in lines:
        if not line.strip():
            continue
        record = json.loads(line)
        name = record["image_name"]
        if record["status"] != "done":
            failed.append(name)
            continue
        image_id = int(Path(name).stem)
        image_ids.append(image_id)
        detections += coco_detections(
            image_id, ((d["box"], d["score"], d["class_id"]) for d in record["detections"])
        )
    return detections, sorted(set(image_ids)), failed


def coco_map(annotations: Path, detections: list[dict], image_ids: list[int]) -> dict:
    """Official COCO metrics for the given images (pycocotools)."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if not detections:
        return {"map50_95": 0.0, "map50": 0.0}
    quiet = io.StringIO()
    with contextlib.redirect_stdout(quiet):
        gt = COCO(str(annotations))
        dt = gt.loadRes(detections)
        ev = COCOeval(gt, dt, "bbox")
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    s = ev.stats
    return {
        "map50_95": round(float(s[0]), 4),
        "map50": round(float(s[1]), 4),
        "map75": round(float(s[2]), 4),
        "map_small": round(float(s[3]), 4),
        "map_medium": round(float(s[4]), 4),
        "map_large": round(float(s[5]), 4),
    }
