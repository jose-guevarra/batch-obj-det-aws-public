"""Pure helpers for the Streamlit app (unit-tested; no Streamlit imports)."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from pathlib import Path

from detector.coco import COCO_NAMES

RECENT_FILE = Path.home() / ".detector-ui.json"
MAX_RECENT = 10
PHASES = ("waiting for a GPU worker", "processing", "finalizing", "done")
CSV_COLUMNS = (
    "image_name", "status", "class_id", "class_name", "score", "x1", "y1", "x2", "y2", "error",
)  # fmt: skip


def results_to_csv(jsonl: str) -> str:
    """results.jsonl -> CSV with one row per detection (one row for an image with none)."""
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(CSV_COLUMNS)
    for raw in jsonl.splitlines():
        if not raw.strip():
            continue
        line = json.loads(raw)
        name, status = line["image_name"], line["status"]
        detections = line.get("detections") or []
        if status != "done" or not detections:
            writer.writerow([name, status, "", "", "", "", "", "", "", line.get("error") or ""])
            continue
        for d in detections:
            x1, y1, x2, y2 = d["box"]
            cls = int(d["class_id"])
            writer.writerow([name, status, cls, COCO_NAMES[cls], d["score"], x1, y1, x2, y2, ""])
    return out.getvalue()


def phase(status: dict) -> tuple[str, str]:
    """(phase, detail) for the progress block, from GET /batches/{id}."""
    s = status["status"]
    expected = status.get("expected_count")
    done = status["completed"] + status["failed"]
    if s == "rejected":
        return "rejected", status.get("error") or "the batch was rejected"
    if s == "done":
        return "done", "results are ready"
    if s == "finalizing" or (expected is not None and done >= expected):
        return "finalizing", "combining results and writing the summary"
    if expected is None:
        return "extracting zip", "the unzip function is counting and extracting the images"
    if done == 0:
        return (
            "waiting for a GPU worker",
            "a cold start from zero takes about 4-5 minutes before the first results",
        )
    return "processing", f"{expected - done} images to go"


def parse_time(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def elapsed_s(status: dict, now: datetime | None = None) -> float | None:
    """Seconds since the batch was created; frozen at the last result once it's done."""
    created = parse_time(status.get("created_at"))
    if created is None:
        return None
    end = parse_time(status.get("updated_at")) if status["status"] == "done" else None
    return ((end or now or datetime.now(UTC)) - created).total_seconds()


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = max(0, int(seconds))
    h, rest = divmod(seconds, 3600)
    m, s = divmod(rest, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s" if m else f"{s}s"


def fmt_bytes(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def default_batch_id(now: datetime | None = None) -> str:
    return f"ui-{(now or datetime.now(UTC)):%Y%m%d-%H%M%S}"


def matches(image: dict, status: str, classes: Iterable[str], min_detections: int) -> bool:
    """The image browser's filters: status (all | done | failed), any of classes, count."""
    if status != "all" and image["status"] != status:
        return False
    wanted = set(classes)
    if wanted and not wanted & set(image.get("classes", [])):
        return False
    return image.get("num_detections", 0) >= min_detections


def filtered_page(
    fetch: Callable[[str | None], dict],
    cursor: str | None,
    keep: Callable[[dict], bool],
    size: int,
    max_requests: int = 20,
) -> tuple[list[dict], str | None]:
    """Up to `size` matching images from `cursor` on, and the cursor for the next page.

    fetch(cursor) returns one API page ({"images", "next_cursor"}). The API's cursor is an
    image name (results come in name order), so the next page can start right after the
    last image shown, even in the middle of an API page.
    """
    found: list[dict] = []
    for _ in range(max_requests):
        page = fetch(cursor)
        for image in page["images"]:
            if keep(image):
                found.append(image)
                if len(found) == size:
                    return found, image["image_name"]
        cursor = page["next_cursor"]
        if cursor is None:
            return found, None
    return found, cursor


class RecentBatches:
    """The last few batch IDs this machine opened, newest first, kept in a small JSON file."""

    def __init__(self, path: Path = RECENT_FILE) -> None:
        self.path = path

    def load(self) -> list[str]:
        try:
            ids = json.loads(self.path.read_text())
        except OSError, ValueError:
            return []
        return [i for i in ids if isinstance(i, str)][:MAX_RECENT] if isinstance(ids, list) else []

    def add(self, batch_id: str) -> list[str]:
        ids = [batch_id] + [i for i in self.load() if i != batch_id]
        ids = ids[:MAX_RECENT]
        try:
            self.path.write_text(json.dumps(ids))
        except OSError:
            pass
        return ids
