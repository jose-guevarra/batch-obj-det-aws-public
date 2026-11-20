"""Preprocessing and postprocessing shared by every runtime (PyTorch, ONNX Runtime, TensorRT).

Every benchmark row runs this exact code, so differences between rows come from the
runtime, the precision or the batch size, never from the input or the box decoding.

Pipeline for one batch:
    raw bytes --decode--> BGR image --letterbox--> 640x640 --to_batch--> (B, 3, 640, 640) float32
    model output (B, 84, 8400) --postprocess--> per-image Detections in original pixels

The letterbox, box rescaling and NMS follow Ultralytics 8.4 (LetterBox with auto=False,
ops.scale_boxes, nms.non_max_suppression) so results match its published mAP.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch
import torchvision

from detector.coco import COCO80_TO_91, COCO_NAMES  # noqa: F401  (re-exported)

IMG_SIZE = 640
PAD_VALUE = 114
MAX_NMS = 30_000  # candidates kept before NMS
MAX_WH = 7680  # per-class box offset so one NMS call never mixes classes


class DecodeError(ValueError):
    """The bytes are not a decodable image. Retrying won't help: send it to the DLQ."""


@dataclass(frozen=True)
class LetterboxMeta:
    """How one image was letterboxed; enough to map boxes back to original pixels."""

    orig_h: int
    orig_w: int
    gain_x: float  # resized width / original width
    gain_y: float  # resized height / original height
    pad_x: int  # left padding
    pad_y: int  # top padding


@dataclass(frozen=True)
class Detections:
    """Detections for one image. Boxes are xyxy in original image pixels."""

    boxes: np.ndarray  # float32 (N, 4)
    scores: np.ndarray  # float32 (N,)
    class_ids: np.ndarray  # int64 (N,), YOLO indices 0-79

    def __len__(self) -> int:
        return len(self.scores)

    def to_records(self) -> list[dict]:
        """Plain Python records for JSON, Parquet or DynamoDB."""
        return [
            {
                "box": [round(float(v), 2) for v in box],
                "score": round(float(score), 5),
                "class_id": int(cls),
                "class_name": COCO_NAMES[int(cls)],
            }
            for box, score, cls in zip(self.boxes, self.scores, self.class_ids, strict=True)
        ]


def decode(data: bytes) -> np.ndarray:
    """Decode image bytes to a BGR uint8 array (H, W, 3)."""
    if not data:
        raise DecodeError("empty input")
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None or img.size == 0:
        raise DecodeError("not a decodable image")
    return img


def letterbox(img: np.ndarray, size: int = IMG_SIZE) -> tuple[np.ndarray, LetterboxMeta]:
    """Resize keeping aspect ratio, then pad to exactly size x size, centered."""
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    new_w, new_h = max(round(w * r), 1), max(round(h * r), 1)
    if (new_w, new_h) != (w, h):
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    dw, dh = (size - new_w) / 2, (size - new_h) / 2
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.copyMakeBorder(
        img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(PAD_VALUE,) * 3
    )
    meta = LetterboxMeta(
        orig_h=h, orig_w=w, gain_x=new_w / w, gain_y=new_h / h, pad_x=left, pad_y=top
    )
    return img, meta


def to_batch(images: Sequence[np.ndarray]) -> np.ndarray:
    """Letterboxed BGR uint8 images -> contiguous RGB float32 (B, 3, H, W) in [0, 1]."""
    batch = np.stack(images)[..., ::-1].transpose(0, 3, 1, 2)
    return np.ascontiguousarray(batch, dtype=np.float32) / 255.0


@dataclass
class Preprocessed:
    batch: np.ndarray | None  # (B, 3, 640, 640) for the images that decoded, or None
    metas: list[LetterboxMeta]  # one per row of batch
    indices: list[int]  # position of each batch row in the input list
    failures: dict[int, DecodeError]  # input position -> why it failed


def preprocess(
    raw: Sequence[bytes], size: int = IMG_SIZE, timer: StageTimer | None = None
) -> Preprocessed:
    """Decode and letterbox each image on its own, so one bad file never fails the batch."""
    timer = timer or StageTimer()
    images, metas, indices, failures = [], [], [], {}
    for i, data in enumerate(raw):
        try:
            with timer.stage("decode"):
                img = decode(data)
        except DecodeError as e:
            failures[i] = e
            continue
        with timer.stage("preprocess"):
            boxed, meta = letterbox(img, size)
        images.append(boxed)
        metas.append(meta)
        indices.append(i)
    with timer.stage("preprocess"):
        batch = to_batch(images) if images else None
    return Preprocessed(batch=batch, metas=metas, indices=indices, failures=failures)


def _xywh2xyxy(x: torch.Tensor) -> torch.Tensor:
    xy, wh = x[:, :2], x[:, 2:4] / 2
    return torch.cat((xy - wh, xy + wh), dim=1)


def _empty() -> Detections:
    return Detections(
        boxes=np.zeros((0, 4), np.float32),
        scores=np.zeros(0, np.float32),
        class_ids=np.zeros(0, np.int64),
    )


def postprocess(
    raw: torch.Tensor,
    metas: Sequence[LetterboxMeta],
    conf: float = 0.25,
    iou: float = 0.7,
    max_det: int = 300,
    multi_label: bool = False,
) -> list[Detections]:
    """Raw model output (B, 4 + classes, anchors) -> per-image Detections.

    multi_label=True keeps every class above conf for each anchor (Ultralytics' validation
    setting, needed to reproduce its mAP); False keeps only the best class (production).
    """
    if raw.shape[0] != len(metas):
        raise ValueError(f"batch has {raw.shape[0]} rows but {len(metas)} metas")
    preds = raw.float().transpose(-1, -2)  # (B, anchors, 4 + classes)
    out = []
    for pred, meta in zip(preds, metas, strict=True):
        cls_scores = pred[:, 4:]
        pred = pred[cls_scores.amax(1) > conf]
        if not len(pred):
            out.append(_empty())
            continue
        boxes, cls_scores = _xywh2xyxy(pred[:, :4]), pred[:, 4:]

        if multi_label:
            i, j = torch.where(cls_scores > conf)
            boxes, scores, classes = boxes[i], cls_scores[i, j], j
        else:
            scores, classes = cls_scores.max(1)
            keep = scores > conf
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        if len(scores) > MAX_NMS:
            top = scores.argsort(descending=True)[:MAX_NMS]
            boxes, scores, classes = boxes[top], scores[top], classes[top]

        offsets = classes[:, None].to(boxes.dtype) * MAX_WH
        keep = torchvision.ops.nms(boxes + offsets, scores, iou)[:max_det]
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]

        boxes = boxes.cpu().numpy().copy()
        boxes[:, [0, 2]] = ((boxes[:, [0, 2]] - meta.pad_x) / meta.gain_x).clip(0, meta.orig_w)
        boxes[:, [1, 3]] = ((boxes[:, [1, 3]] - meta.pad_y) / meta.gain_y).clip(0, meta.orig_h)
        out.append(
            Detections(
                boxes=boxes.astype(np.float32),
                scores=scores.cpu().numpy().astype(np.float32),
                class_ids=classes.cpu().numpy().astype(np.int64),
            )
        )
    return out


@dataclass
class StageTimer:
    """Wall-clock time per pipeline stage.

    Pass sync (e.g. torch.cuda.synchronize) so asynchronous GPU work is finished, and
    charged to the right stage, before each stage's clock stops.
    """

    sync: Callable[[], None] | None = None
    totals: dict[str, float] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            if self.sync is not None:
                self.sync()
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - start
            self.counts[name] = self.counts.get(name, 0) + 1

    def summary(self, batches: int) -> dict[str, dict[str, float]]:
        """Total seconds and mean milliseconds per batch for each stage."""
        return {
            name: {"total_s": round(total, 4), "per_batch_ms": round(1000 * total / batches, 3)}
            for name, total in self.totals.items()
        }
