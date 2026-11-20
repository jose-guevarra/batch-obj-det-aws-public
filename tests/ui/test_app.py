"""Smoke tests: the pages render with the API mocked (no network, no AWS)."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from ui import helpers
from ui.client import Api

APP = str(Path(__file__).parents[2] / "ui" / "app.py")

DONE = {
    "batch_id": "b1",
    "status": "done",
    "source": "ui",
    "expected_count": 2,
    "completed": 1,
    "failed": 1,
    "percent": 100.0,
    "created_at": "2026-11-12T10:00:00.000+00:00",
    "updated_at": "2026-11-12T10:05:00.000+00:00",
    "summary": {
        "images": 2,
        "completed": 1,
        "failed": 1,
        "detections": 2,
        "detections_per_class": {"person": 1, "dog": 1},
        "mean_score": 0.7,
        "failures": [{"image_name": "c.jpg", "error": "decode error"}],
        "total_s": 300.0,
        "estimated_cost_usd": 0.0123,
        "cost_note": "an estimate",
    },
    "results_url": "https://example.invalid/results.jsonl",
}
RESULTS = (
    '{"image_name": "a.jpg", "status": "done", "detections": '
    '[{"box": [1, 2, 3, 4], "score": 0.9, "class_id": 0}]}\n'
    '{"image_name": "c.jpg", "status": "failed", "error": "decode error"}\n'
)


@pytest.fixture
def mocked(monkeypatch, tmp_path):
    monkeypatch.setenv("API_URL", "https://api.example.invalid/prod/")
    monkeypatch.setattr(helpers, "RECENT_FILE", tmp_path / "recent.json")
    monkeypatch.setattr(helpers.RecentBatches.__init__, "__defaults__", (tmp_path / "recent.json",))
    statuses = {"b1": DONE, "p1": DONE | {"status": "pending", "completed": 0, "failed": 0}}
    monkeypatch.setattr(Api, "health", lambda self: True)
    monkeypatch.setattr(Api, "batch", lambda self, batch_id: statuses[batch_id])
    monkeypatch.setattr(Api, "download_text", lambda self, url: RESULTS)
    monkeypatch.setattr(
        Api,
        "images",
        lambda self, batch_id, cursor=None, limit=100: {
            "images": [
                {"image_name": "a.jpg", "status": "done", "num_detections": 1,
                 "top_classes": ["person"], "classes": ["person"], "error": None},
                {"image_name": "c.jpg", "status": "failed", "num_detections": 0,
                 "top_classes": [], "classes": [], "error": "decode error"},
            ],
            "next_cursor": None,
        },
    )  # fmt: skip


def test_new_batch_page_renders(mocked):
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert at.radio[0].options == ["Images", "Zip file"]
    assert at.text_input(key="new_batch_id").value.startswith("ui-")
    (button,) = [b for b in at.button if b.label == "Upload and process"]
    assert button.disabled  # nothing picked yet
    assert any("API" in c.value for c in at.sidebar.caption)


def test_done_batch_page_shows_summary_and_images(mocked):
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["batch"] = "b1"
    at.switch_page("pages/batch.py").run()
    assert not at.exception, at.exception
    assert at.header[0].value == "b1"
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Completed"] == "1" and metrics["Est. cost"] == "$0.0123"
    labels = [b.label for b in at.button]
    assert labels.count("View") == 2


def test_running_batch_page_shows_progress(mocked):
    at = AppTest.from_file(APP, default_timeout=30)
    at.query_params["batch"] = "p1"
    at.switch_page("pages/batch.py").run()
    assert not at.exception, at.exception
    assert any("waiting for a GPU worker" in m.value for m in at.markdown)
