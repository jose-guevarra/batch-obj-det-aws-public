import csv
import io
import json
from datetime import UTC, datetime

from ui.helpers import (
    RecentBatches,
    default_batch_id,
    elapsed_s,
    filtered_page,
    fmt_bytes,
    fmt_duration,
    matches,
    phase,
    results_to_csv,
)


def status(**kw):
    base = {"status": "pending", "expected_count": 10, "completed": 0, "failed": 0}
    return base | kw


def test_results_to_csv_one_row_per_detection():
    lines = [
        {
            "image_name": "a.jpg",
            "status": "done",
            "detections": [
                {"box": [1, 2, 3, 4], "score": 0.9, "class_id": 0},
                {"box": [5, 6, 7, 8], "score": 0.5, "class_id": 16},
            ],
        },
        {"image_name": "b.jpg", "status": "done", "detections": []},
        {"image_name": "c.jpg", "status": "failed", "error": "decode error"},
    ]
    text = results_to_csv("\n".join(json.dumps(x) for x in lines) + "\n")
    rows = list(csv.DictReader(io.StringIO(text)))
    assert [(r["image_name"], r["class_name"]) for r in rows] == [
        ("a.jpg", "person"),
        ("a.jpg", "dog"),
        ("b.jpg", ""),
        ("c.jpg", ""),
    ]
    assert rows[0]["x2"] == "3" and rows[3]["error"] == "decode error"


def test_phase():
    assert phase(status())[0] == "waiting for a GPU worker"
    assert phase(status(completed=3))[0] == "processing"
    assert phase(status(completed=9, failed=1))[0] == "finalizing"
    assert phase(status(status="finalizing", completed=10))[0] == "finalizing"
    assert phase(status(status="done", completed=10))[0] == "done"
    assert phase(status(expected_count=None))[0] == "extracting zip"
    assert phase(status(status="rejected", error="zip rejected: x")) == (
        "rejected",
        "zip rejected: x",
    )


def test_elapsed_freezes_when_done():
    s = status(created_at="2026-11-12T10:00:00.000+00:00", updated_at="2026-11-12T10:05:00+00:00")
    now = datetime(2026, 11, 12, 10, 10, tzinfo=UTC)
    assert elapsed_s(s, now) == 600
    assert elapsed_s(s | {"status": "done"}, now) == 300
    assert elapsed_s(status(), now) is None


def test_formatting():
    assert [fmt_duration(x) for x in (None, 5, 312, 7300)] == ["-", "5s", "5m 12s", "2h 01m"]
    assert [fmt_bytes(x) for x in (12, 4300, 4_300_000)] == ["12 B", "4.2 KB", "4.1 MB"]
    assert default_batch_id(datetime(2026, 11, 12, 15, 30, tzinfo=UTC)) == "ui-20261112-153000"


def img(name, status="done", classes=(), n=0):
    return {"image_name": name, "status": status, "classes": list(classes), "num_detections": n}


def test_matches():
    a = img("a", classes=["dog", "person"], n=3)
    assert matches(a, "all", [], 0)
    assert matches(a, "done", ["cat", "dog"], 3)
    assert not matches(a, "failed", [], 0)
    assert not matches(a, "all", ["cat"], 0)
    assert not matches(a, "all", [], 4)


def test_filtered_page_spans_api_pages_and_resumes_mid_page():
    names = [f"{i:02d}" for i in range(10)]
    calls = []

    def fetch(cursor):
        calls.append(cursor)
        start = 0 if cursor is None else names.index(cursor) + 1
        chunk = names[start : start + 4]
        nxt = chunk[-1] if start + 4 < len(names) else None
        return {"images": [img(n, n=int(n)) for n in chunk], "next_cursor": nxt}

    def even(image):
        return image["num_detections"] % 2 == 0

    page, nxt = filtered_page(fetch, None, even, size=3)
    assert [i["image_name"] for i in page] == ["00", "02", "04"]
    assert nxt == "04"
    page, nxt = filtered_page(fetch, nxt, even, size=3)
    assert [i["image_name"] for i in page] == ["06", "08"]
    assert nxt is None


def test_recent_batches(tmp_path):
    recent = RecentBatches(tmp_path / "recent.json")
    assert recent.load() == []
    for i in range(12):
        recent.add(f"b{i}")
    recent.add("b5")
    ids = recent.load()
    assert ids[0] == "b5" and len(ids) == 10 and ids.count("b5") == 1
    (tmp_path / "recent.json").write_text("not json")
    assert recent.load() == []
