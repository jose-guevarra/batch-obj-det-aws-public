import json

from detector.finalize import finalize
from detector.store import ImageResult
from tests.worker.conftest import BUCKET, jpeg

META = {"worker_id": "w1", "runtime": "fake", "precision": "fp32"}
PERSON = [{"box": [4.0, 4.0, 40.0, 30.0], "score": 0.9, "class_id": 0}]


def commit(api, batch_id, results):
    api.env.store.commit(batch_id, results, **META)


def test_health(api):
    assert api.get("/health").json() == {"ok": True}


def test_create_batch_returns_presigned_puts(api):
    r = api.post("/batches", json={"files": ["a.jpg", "b.png"], "batch_id": "t1"})
    assert r.status_code == 201
    body = r.json()
    assert body["batch_id"] == "t1"
    assert [u["name"] for u in body["uploads"]] == ["a.jpg", "b.png"]
    assert "input/t1/a.jpg" in body["uploads"][0]["url"]
    assert "X-Amz-Signature" in body["uploads"][0]["url"]
    batch = api.env.store.get_batch("t1")
    assert (batch["expected_count"], batch["source"]) == (2, "ui")
    assert api.post("/batches", json={"files": ["a.jpg"], "batch_id": "t1"}).status_code == 409


def test_create_zip_batch_has_no_count(api):
    body = api.post("/batches", json={"zip": True}).json()
    assert body["batch_id"].startswith("ui-")
    assert f"upload/{body['batch_id']}/batch.zip" in body["uploads"][0]["url"]
    status = api.get(f"/batches/{body['batch_id']}").json()
    assert status["expected_count"] is None and status["percent"] is None


def test_create_batch_validation(api):
    for payload in (
        {},
        {"files": []},
        {"files": ["a.jpg"], "zip": True},
        {"files": ["../a.jpg"]},
        {"files": ["notes.txt"]},
        {"files": [".hidden.jpg"]},
        {"files": ["a.jpg", "A.JPG"]},
        {"files": ["a.jpg"], "batch_id": "bad id"},
        {"files": [f"{i}.jpg" for i in range(1001)]},
    ):
        assert api.post("/batches", json=payload).status_code == 422, payload


def test_status_progress_and_finalize_safety_net(api):
    api.post("/batches", json={"files": ["a.jpg", "b.jpg"], "batch_id": "t1"})
    commit(api, "t1", [ImageResult("a.jpg", "done", PERSON)])
    s = api.get("/batches/t1").json()
    assert (s["status"], s["completed"], s["percent"]) == ("pending", 1, 50.0)
    assert api.finalize_calls == []

    commit(api, "t1", [ImageResult.failed("b.jpg", "bad")])
    s = api.get("/batches/t1").json()
    assert s["finalize_requested"] is True
    assert api.finalize_calls == ["t1"]

    finalize(api.env.store, api.env.s3, BUCKET, "t1")
    s = api.get("/batches/t1").json()
    assert s["status"] == "done"
    assert s["summary"]["detections_per_class"] == {"person": 1}
    assert "results/t1/results.jsonl" in s["results_url"]
    assert api.finalize_calls == ["t1"]  # not requested again once done


def test_unknown_batch_is_404(api):
    assert api.get("/batches/nope").status_code == 404
    assert api.get("/batches/nope/images").status_code == 404
    assert api.get("/batches/nope/images/a.jpg/annotated").status_code == 404


def test_list_and_get_images(api):
    api.post("/batches", json={"files": ["a.jpg", "b.jpg", "c.jpg"], "batch_id": "t1"})
    dog_and_people = PERSON * 2 + [{"box": [1, 1, 5, 5], "score": 0.4, "class_id": 16}]
    commit(
        api,
        "t1",
        [
            ImageResult("a.jpg", "done", dog_and_people),
            ImageResult("b.jpg", "done", []),
            ImageResult.failed("c.jpg", "decode error"),
        ],
    )
    page = api.get("/batches/t1/images", params={"limit": 2}).json()
    assert [i["image_name"] for i in page["images"]] == ["a.jpg", "b.jpg"]
    assert page["images"][0]["top_classes"] == ["person", "dog"]
    assert page["images"][0]["num_detections"] == 3
    rest = api.get("/batches/t1/images", params={"cursor": page["next_cursor"]}).json()
    assert rest["images"][0]["error"] == "decode error"

    one = api.get("/batches/t1/images/a.jpg").json()
    assert one["detections"][0]["class_name"] == "person"
    assert "input/t1/a.jpg" in one["original_url"]


def test_annotated_renders_once_then_serves_the_cache(api):
    api.post("/batches", json={"files": ["a.jpg", "c.jpg"], "batch_id": "t1"})
    api.env.s3.put_object(Bucket=BUCKET, Key="input/t1/a.jpg", Body=jpeg(1))
    commit(api, "t1", [ImageResult("a.jpg", "done", PERSON), ImageResult.failed("c.jpg", "x")])

    first = api.get("/batches/t1/images/a.jpg/annotated").json()
    assert first["cached"] is False
    assert "rendered/t1/a.jpg" in first["url"]
    head = api.env.s3.head_object(Bucket=BUCKET, Key="rendered/t1/a.jpg")
    assert head["ContentType"] == "image/jpeg"
    second = api.get("/batches/t1/images/a.jpg/annotated").json()
    assert second["cached"] is True

    assert api.get("/batches/t1/images/c.jpg/annotated").status_code == 409
    assert api.get("/batches/t1/images/zzz.jpg/annotated").status_code == 404


def test_rejected_zip_shows_its_error(api):
    api.post("/batches", json={"zip": True, "batch_id": "z1"})
    api.env.store.reject_batch("z1", "zip rejected: bad")
    s = api.get("/batches/z1").json()
    assert (s["status"], s["error"]) == ("rejected", "zip rejected: bad")
    assert json.dumps(s)  # serializable
