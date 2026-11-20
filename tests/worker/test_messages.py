import json

import pytest

from detector.messages import BadMessage, parse_input_key, parse_message, results_key
from tests.worker.conftest import s3_event


def msg(body: str, mid: str = "m1") -> dict:
    return {"Body": body, "MessageId": mid, "ReceiptHandle": f"rh-{mid}"}


def test_parses_s3_event():
    (ref,) = parse_message(msg(s3_event("bkt", "input/b1/000000000139.jpg")))
    assert (ref.bucket, ref.batch_id, ref.image_name) == ("bkt", "b1", "000000000139.jpg")
    assert ref.key == "input/b1/000000000139.jpg"
    assert (ref.message_id, ref.receipt_handle) == ("m1", "rh-m1")


def test_url_encoded_keys_are_decoded():
    (ref,) = parse_message(msg(s3_event("bkt", "input/b1/my photo+1.jpg")))
    assert ref.image_name == "my photo+1.jpg"


def test_nested_names_are_kept():
    (ref,) = parse_message(msg(s3_event("bkt", "input/b1/sub/dir/a.png")))
    assert ref.image_name == "sub/dir/a.png"


def test_multiple_records():
    refs = parse_message(msg(s3_event("bkt", "input/b1/a.jpg", "input/b2/b.jpg")))
    assert [(r.batch_id, r.image_name) for r in refs] == [("b1", "a.jpg"), ("b2", "b.jpg")]


def test_s3_test_event_is_empty():
    body = json.dumps({"Service": "Amazon S3", "Event": "s3:TestEvent", "Bucket": "bkt"})
    assert parse_message(msg(body)) == []


def test_non_create_events_are_ignored():
    body = json.loads(s3_event("bkt", "input/b1/a.jpg"))
    body["Records"][0]["eventName"] = "ObjectRemoved:Delete"
    assert parse_message(msg(json.dumps(body))) == []


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        json.dumps({"hello": "world"}),
        json.dumps({"Records": [{"eventName": "ObjectCreated:Put"}]}),
        s3_event("bkt", "results/b1/part-1.jsonl"),
        s3_event("bkt", "input/b1/"),
        s3_event("bkt", "input/bad id!/a.jpg"),
        json.dumps([1, 2]),
    ],
    ids=["text", "no-records", "malformed", "wrong-prefix", "folder", "bad-batch-id", "list"],
)
def test_bad_messages(body):
    with pytest.raises(BadMessage):
        parse_message(msg(body))


def test_key_helpers():
    assert parse_input_key("input/run_1-a/x.jpg") == ("run_1-a", "x.jpg")
    assert results_key("b1", "w1", 7) == "results/b1/part-w1-000007.jsonl"
