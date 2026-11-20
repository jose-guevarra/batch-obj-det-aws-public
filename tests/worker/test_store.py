import pytest

from detector import store as store_mod
from detector.store import BatchExists, BatchNotFound, ImageResult, is_complete

META = {"worker_id": "w1", "runtime": "fake", "precision": "fp32"}


def done(name: str, n: int = 1) -> ImageResult:
    dets = [{"box": [1.005, 2.5, 30.0, 40.25], "score": 0.91234567, "class_id": 0}] * n
    return ImageResult(image_name=name, status="done", detections=dets)


def test_create_batch_refuses_duplicate(env):
    env.store.create_batch("b1", 10, "yolov8m")
    with pytest.raises(BatchExists):
        env.store.create_batch("b1", 10, "yolov8m")
    batch = env.store.get_batch("b1")
    assert (batch["expected_count"], batch["completed"], batch["failed"]) == (10, 0, 0)
    assert batch["status"] == "pending"


def test_commit_writes_items_and_counts(env):
    env.store.create_batch("b1", 3, "yolov8m")
    out = env.store.commit(
        "b1", [done("a.jpg", 2), done("b.jpg"), ImageResult.failed("c.jpg", "bad")], **META
    )
    assert (out.written, out.skipped) == (3, 0)
    batch = env.store.get_batch("b1")
    assert (batch["completed"], batch["failed"]) == (2, 1)
    a = env.store.get_image("b1", "a.jpg")
    assert a["num_detections"] == 2
    assert a["detections"][0] == {"box": [1.0, 2.5, 30, 40.25], "score": 0.91235, "class_id": 0}
    assert a["worker_id"] == "w1"
    assert env.store.get_image("b1", "c.jpg")["error"] == "bad"


def test_recommit_is_skipped_and_not_counted(env):
    env.store.create_batch("b1", 2, "yolov8m")
    results = [done("a.jpg"), done("b.jpg")]
    env.store.commit("b1", results, **META)
    out = env.store.commit("b1", results, **META)
    assert (out.written, out.skipped) == (0, 2)
    assert env.store.get_batch("b1")["completed"] == 2


def test_partial_overlap_counts_only_new(env):
    env.store.create_batch("b1", 4, "yolov8m")
    env.store.commit("b1", [done("a.jpg"), done("b.jpg")], **META)
    out = env.store.commit(
        "b1", [done("b.jpg"), done("c.jpg"), ImageResult.failed("d.jpg", "x")], **META
    )
    assert (out.written, out.skipped) == (2, 1)
    batch = env.store.get_batch("b1")
    assert (batch["completed"], batch["failed"]) == (3, 1)


def test_missing_batch_record(env):
    with pytest.raises(BatchNotFound):
        env.store.commit("nope", [done("a.jpg")], **META)
    assert env.store.get_image("nope", "a.jpg") is None  # nothing half-written


def test_large_commit_splits_into_transactions(env):
    env.store.create_batch("b1", 250, "yolov8m")
    out = env.store.commit("b1", [done(f"{i}.jpg") for i in range(250)], **META)
    assert out.written == 250
    assert env.store.get_batch("b1")["completed"] == 250


def test_existing_handles_more_than_100_keys(env):
    env.store.create_batch("b1", 150, "yolov8m")
    env.store.commit("b1", [done(f"{i}.jpg") for i in range(0, 150, 2)], **META)
    found = env.store.existing("b1", [f"{i}.jpg" for i in range(150)] + ["x.jpg"])
    assert found == {f"{i}.jpg" for i in range(0, 150, 2)}


def test_existing_is_per_batch(env):
    env.store.create_batch("b1", 1, "yolov8m")
    env.store.create_batch("b2", 1, "yolov8m")
    env.store.commit("b1", [done("a.jpg")], **META)
    assert env.store.existing("b2", ["a.jpg"]) == set()


# Unknown counts, finalize claim, listing -------------------------------------


def test_zip_batch_starts_without_a_count_and_sets_it_once(env):
    env.store.create_batch("z1", None, "yolov8m", source="zip")
    batch = env.store.get_batch("z1")
    assert "expected_count" not in batch
    assert not is_complete(batch)
    assert env.store.set_expected_count("z1", 5) is True
    assert env.store.set_expected_count("z1", 7) is False
    assert env.store.get_batch("z1")["expected_count"] == 5


def test_set_expected_count_needs_a_record(env):
    with pytest.raises(BatchNotFound):
        env.store.set_expected_count("nope", 3)


def test_claim_finalize_only_when_complete_and_only_once(env):
    env.store.create_batch("b1", 2, "yolov8m")
    env.store.commit("b1", [done("a.jpg")], **META)
    assert env.store.claim_finalize("b1") is False  # 1 of 2
    env.store.commit("b1", [ImageResult.failed("b.jpg", "bad")], **META)
    assert env.store.claim_finalize("b1") is True
    assert env.store.claim_finalize("b1") is False
    assert env.store.get_batch("b1")["status"] == "finalizing"
    env.store.finish_finalize("b1", {"detections": 1, "mean_score": 0.91})
    batch = env.store.get_batch("b1")
    assert batch["status"] == "done"
    assert (batch["detections"], batch["mean_score"]) == (1, 0.91)
    assert env.store.claim_finalize("b1") is False


def test_stale_finalize_claim_can_be_taken_again(env, monkeypatch):
    env.store.create_batch("b1", 1, "yolov8m")
    env.store.commit("b1", [done("a.jpg")], **META)
    assert env.store.claim_finalize("b1") is True
    monkeypatch.setattr(store_mod, "FINALIZE_STALE_S", -1)  # every claim is now stale
    assert env.store.claim_finalize("b1") is True


def test_empty_batch_is_complete(env):
    env.store.create_batch("z1", None, "yolov8m", source="zip")
    env.store.set_expected_count("z1", 0)
    assert env.store.claim_finalize("z1") is True


def test_list_images_pages_in_name_order(env):
    env.store.create_batch("b1", 5, "yolov8m")
    names = [f"img{i}.jpg" for i in range(5)]
    env.store.commit(
        "b1", [done(n) for n in names[:4]] + [ImageResult.failed(names[4], "x")], **META
    )
    env.store.create_batch("b2", 1, "yolov8m")
    env.store.commit("b2", [done("other.jpg")], **META)

    page1, cursor = env.store.list_images("b1", limit=3)
    assert [i["image_name"] for i in page1] == names[:3]
    page2, cursor2 = env.store.list_images("b1", limit=3, cursor=cursor)
    assert [i["image_name"] for i in page2] == names[3:]
    assert page2[-1]["status"] == "failed"
    assert cursor2 is None or env.store.list_images("b1", limit=3, cursor=cursor2)[0] == []
    assert {i["image_name"] for i in env.store.all_images("b1")} == set(names)
