import io
import zipfile

import pytest

from detector import unzip
from detector.unzip import ZipRejected, batch_id_from_key, extract, plan_entries
from tests.worker.conftest import BUCKET, jpeg


def make_zip(entries: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            if name.endswith("/"):
                zf.writestr(zipfile.ZipInfo(name), b"")
            else:
                zf.writestr(name, data)
    return buf.getvalue()


def put_zip(env, batch_id: str, data: bytes) -> str:
    key = f"upload/{batch_id}/batch.zip"
    env.s3.put_object(Bucket=BUCKET, Key=key, Body=data)
    return key


def input_names(env, batch_id: str) -> list[str]:
    resp = env.s3.list_objects_v2(Bucket=BUCKET, Prefix=f"input/{batch_id}/")
    return sorted(o["Key"].removeprefix(f"input/{batch_id}/") for o in resp.get("Contents", []))


def test_batch_id_from_key():
    assert batch_id_from_key("upload/z1/batch.zip") == "z1"
    for bad in ("upload/z1/x.tar", "input/z1/a.zip", "upload/z1/sub/a.zip", "upload/b@d/a.zip"):
        with pytest.raises(ValueError):
            batch_id_from_key(bad)


def test_extracts_images_flattened_and_counts_them_first(env, tmp_path):
    env.store.create_batch("z1", None, "yolov8m", source="zip")
    data = make_zip(
        {
            "photos/": b"",
            "photos/a.jpg": jpeg(1),
            "photos/b.PNG": jpeg(2),
            "other/a.jpg": jpeg(3),  # same base name: becomes a-2.jpg
            "../../etc/evil.jpg": jpeg(4),  # traversal: flattened to evil.jpg
            "notes.txt": b"hello",
            "__MACOSX/photos/._a.jpg": b"junk",
            ".hidden.jpg": jpeg(5),
        }
    )
    out = extract(env.store, env.s3, BUCKET, put_zip(env, "z1", data), str(tmp_path))
    assert out["images"] == 4
    assert input_names(env, "z1") == ["a-2.jpg", "a.jpg", "b.PNG", "evil.jpg"]
    assert env.store.get_batch("z1")["expected_count"] == 4
    head = env.s3.head_object(Bucket=BUCKET, Key="input/z1/b.PNG")
    assert head["ContentType"] == "image/png"


def test_zip_without_a_record_creates_one(env, tmp_path):
    extract(
        env.store, env.s3, BUCKET, put_zip(env, "z2", make_zip({"a.jpg": jpeg()})), str(tmp_path)
    )
    batch = env.store.get_batch("z2")
    assert (batch["expected_count"], batch["source"]) == (1, "zip")


def test_empty_zip_counts_zero(env, tmp_path):
    env.store.create_batch("z3", None, "yolov8m", source="zip")
    extract(
        env.store, env.s3, BUCKET, put_zip(env, "z3", make_zip({"x.txt": b"no"})), str(tmp_path)
    )
    assert env.store.get_batch("z3")["expected_count"] == 0
    assert input_names(env, "z3") == []


def test_broken_zip_rejects_the_batch(env, tmp_path):
    env.store.create_batch("z4", None, "yolov8m", source="zip")
    out = extract(env.store, env.s3, BUCKET, put_zip(env, "z4", b"not a zip"), str(tmp_path))
    assert out["event"] == "unzip_rejected"
    batch = env.store.get_batch("z4")
    assert batch["status"] == "rejected"
    assert "expected_count" not in batch


def test_limits(monkeypatch):
    infos = [zipfile.ZipInfo(f"{i}.jpg") for i in range(4)]
    monkeypatch.setattr(unzip, "MAX_ENTRIES", 3)
    with pytest.raises(ZipRejected):
        plan_entries(infos)
    monkeypatch.setattr(unzip, "MAX_ENTRIES", 10)
    for info in infos:
        info.file_size = 10
    monkeypatch.setattr(unzip, "MAX_TOTAL_BYTES", 25)
    with pytest.raises(ZipRejected):
        plan_entries(infos)
