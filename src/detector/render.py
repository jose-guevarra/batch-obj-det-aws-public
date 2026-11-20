"""Draw detections on an image (OpenCV only, no torch: runs in the API's Lambda)."""

from __future__ import annotations

import colorsys
from collections.abc import Sequence

import cv2
import numpy as np

from detector.coco import COCO_NAMES

JPEG_QUALITY = 90


def class_color(class_id: int) -> tuple[int, int, int]:
    """A fixed, well-spread BGR color per class (golden-ratio hues)."""
    hue = (class_id * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.85, 0.95)
    return int(b * 255), int(g * 255), int(r * 255)


def draw(image: np.ndarray, detections: Sequence[dict]) -> np.ndarray:
    """A copy of a BGR image with boxes and "class score" labels. Boxes are xyxy pixels."""
    out = image.copy()
    h, w = out.shape[:2]
    scale = max(0.4, min(w, h) / 900)
    thickness = max(1, round(min(w, h) / 300))
    for d in sorted(detections, key=lambda d: d["score"]):  # best boxes drawn last, on top
        x1, y1, x2, y2 = (int(round(float(v))) for v in d["box"])
        cls = int(d["class_id"])
        color = class_color(cls)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        label = f"{COCO_NAMES[cls]} {float(d['score']):.2f}"
        (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        ty = y1 - 2 if y1 - th - base - 2 >= 0 else y1 + th + base + 2
        cv2.rectangle(out, (x1, ty - th - base), (x1 + tw + 2, ty + 1), color, -1)
        cv2.putText(
            out, label, (x1 + 1, ty - base + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1,
            cv2.LINE_AA,
        )  # fmt: skip
    return out


def render_jpeg(data: bytes, detections: Sequence[dict]) -> bytes:
    """Encoded image bytes in, annotated JPEG bytes out. ValueError if it can't be decoded."""
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("not a decodable image")
    ok, buf = cv2.imencode(
        ".jpg", draw(image, detections), [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        raise ValueError("JPEG encoding failed")
    return buf.tobytes()
