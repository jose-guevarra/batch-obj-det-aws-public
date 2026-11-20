"""S3 key layout and SQS message parsing.

Bucket layout (one top-level prefix per purpose, so the S3 -> SQS notification can
filter on input/ and the worker's own writes never queue messages):

    input/{batch_id}/{image_name}      images to process
    upload/{batch_id}/{name}.zip       zips, extracted into input/
    results/{batch_id}/part-*.jsonl    worker output, combined at finalize
    rendered/{batch_id}/{image_name}   annotated images, rendered on demand
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from urllib.parse import unquote_plus

INPUT_PREFIX = "input/"
UPLOAD_PREFIX = "upload/"
RESULTS_PREFIX = "results/"
RENDERED_PREFIX = "rendered/"

_BATCH_ID = r"[A-Za-z0-9_-]{1,64}"
BATCH_ID_RE = re.compile(rf"^{_BATCH_ID}$")
_INPUT_KEY_RE = re.compile(rf"^input/(?P<batch_id>{_BATCH_ID})/(?P<name>[^/].*)$")


class BadMessage(ValueError):
    """The message can never be processed (not JSON, not an S3 event, wrong key)."""


@dataclass(frozen=True)
class ImageRef:
    bucket: str
    key: str
    batch_id: str
    image_name: str
    message_id: str
    receipt_handle: str


def parse_input_key(key: str) -> tuple[str, str]:
    """input/{batch_id}/{image_name} -> (batch_id, image_name)."""
    m = _INPUT_KEY_RE.match(key)
    if not m or m["name"].endswith("/"):
        raise BadMessage(f"key is not under input/{{batch_id}}/: {key!r}")
    return m["batch_id"], m["name"]


def input_key(batch_id: str, image_name: str) -> str:
    return f"{INPUT_PREFIX}{batch_id}/{image_name}"


def results_key(batch_id: str, worker_id: str, seq: int) -> str:
    return f"{RESULTS_PREFIX}{batch_id}/part-{worker_id}-{seq:06d}.jsonl"


def part_prefix(batch_id: str) -> str:
    return f"{RESULTS_PREFIX}{batch_id}/part-"


def combined_results_key(batch_id: str) -> str:
    return f"{RESULTS_PREFIX}{batch_id}/results.jsonl"


def summary_key(batch_id: str) -> str:
    return f"{RESULTS_PREFIX}{batch_id}/summary.json"


def upload_key(batch_id: str, name: str = "batch.zip") -> str:
    return f"{UPLOAD_PREFIX}{batch_id}/{name}"


def rendered_key(batch_id: str, image_name: str) -> str:
    return f"{RENDERED_PREFIX}{batch_id}/{image_name}"


def parse_message(msg: dict) -> list[ImageRef]:
    """An SQS message carrying an S3 event notification -> the images it names.

    Returns [] for S3's test event (sent when the notification is created).
    """
    try:
        body = json.loads(msg["Body"])
    except (KeyError, TypeError, json.JSONDecodeError) as e:
        raise BadMessage(f"body is not JSON: {e}") from None

    if isinstance(body, dict) and body.get("Event") == "s3:TestEvent":
        return []
    records = body.get("Records") if isinstance(body, dict) else None
    if not records or not isinstance(records, list):
        raise BadMessage("not an S3 event notification (no Records)")

    refs = []
    for record in records:
        try:
            if not str(record["eventName"]).startswith("ObjectCreated:"):
                continue
            bucket = record["s3"]["bucket"]["name"]
            key = unquote_plus(record["s3"]["object"]["key"])
        except (KeyError, TypeError) as e:
            raise BadMessage(f"malformed S3 record: missing {e}") from None
        batch_id, name = parse_input_key(key)
        refs.append(
            ImageRef(
                bucket=bucket,
                key=key,
                batch_id=batch_id,
                image_name=name,
                message_id=msg["MessageId"],
                receipt_handle=msg["ReceiptHandle"],
            )
        )
    return refs
