"""DynamoDB access for batch records and per-image results.

Table layout (keys pk / sk):
    BATCH#{batch_id} / META               expected_count, completed, failed, status, ...
    BATCH#{batch_id} / IMG#{image_name}   status (done | failed), detections, error, ...

A batch's status goes pending -> finalizing -> done, or to rejected for a broken zip.
expected_count is missing until it's known: a zip batch gets it from the unzip step,
before any of its images are queued.

Results and the batch counters are committed together in one transaction per chunk:
each image is a conditional put (only if not already written) and the counters are
incremented by exactly the number of images that transaction wrote. A crash or a
duplicate delivery can therefore never count an image twice or leave it uncounted.
"""

from __future__ import annotations

import random
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from typing import TYPE_CHECKING, Any

from boto3.dynamodb.types import TypeDeserializer, TypeSerializer
from botocore.exceptions import ClientError

if TYPE_CHECKING:
    from detector.inference_core import Detections

MAX_TRANSACT_ITEMS = 100
CHUNK = MAX_TRANSACT_ITEMS - 1  # images per transaction; the last action is the counter
MAX_BATCH_GET = 100
MAX_ATTEMPTS = 6
FINALIZE_STALE_S = 15 * 60  # a finalizing claim older than this can be taken again
RETRYABLE_CODES = {
    "TransactionConflictException",
    "ThrottlingException",
    "ProvisionedThroughputExceededException",
    "RequestLimitExceeded",
    "InternalServerError",
}
RETRYABLE_REASONS = {"TransactionConflict", "ThrottlingError", "ProvisionedThroughputExceeded"}

_ser = TypeSerializer()
_deser = TypeDeserializer()


class BatchNotFound(LookupError):
    """No batch record: images arrived for a batch that was never created."""


class BatchExists(ValueError):
    """create_batch was called with a batch_id that is already in use."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _pk(batch_id: str) -> str:
    return f"BATCH#{batch_id}"


def _img_sk(image_name: str) -> str:
    return f"IMG#{image_name}"


def _key(batch_id: str, sk: str) -> dict:
    return {"pk": {"S": _pk(batch_id)}, "sk": {"S": sk}}


def _dec(x: float, places: int) -> Decimal:
    return Decimal(str(round(float(x), places)))


def _plain(value: Any) -> Any:
    """Decimals from DynamoDB -> int or float, recursively."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, list):
        return [_plain(v) for v in value]
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    return value


def _to_ddb(value: Any) -> Any:
    """floats -> Decimal (DynamoDB has no float type), recursively."""
    if isinstance(value, float):
        return Decimal(str(value))
    if isinstance(value, list):
        return [_to_ddb(v) for v in value]
    if isinstance(value, dict):
        return {k: _to_ddb(v) for k, v in value.items()}
    return value


def is_complete(batch: dict) -> bool:
    """Every image accounted for. A batch whose count isn't known yet is never complete."""
    expected = batch.get("expected_count")
    return expected is not None and batch["completed"] + batch["failed"] >= expected


def _backoff(attempt: int) -> None:
    time.sleep(min(2.0, 0.05 * 2**attempt) * random.uniform(0.5, 1.0))


def _chunks(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@dataclass(frozen=True)
class ImageResult:
    """The outcome for one image, as written to DynamoDB and the part file."""

    image_name: str
    status: str  # "done" or "failed"
    detections: list[dict] = field(default_factory=list)
    error: str | None = None

    @classmethod
    def done(cls, image_name: str, det: Detections) -> ImageResult:
        return cls(
            image_name=image_name,
            status="done",
            detections=[
                {"box": r["box"], "score": r["score"], "class_id": r["class_id"]}
                for r in det.to_records()
            ],
        )

    @classmethod
    def failed(cls, image_name: str, error: str) -> ImageResult:
        return cls(image_name=image_name, status="failed", error=error[:1000])

    def to_json(self) -> dict:
        out = {"image_name": self.image_name, "status": self.status}
        if self.status == "done":
            out["detections"] = self.detections
        else:
            out["error"] = self.error
        return out


@dataclass(frozen=True)
class CommitResult:
    written: int  # newly committed (and counted) by this call
    skipped: int  # already written earlier (not counted again)


class BatchStore:
    def __init__(self, client: Any, table_name: str) -> None:
        self.client = client
        self.table = table_name

    # Batch records ---------------------------------------------------------

    def create_batch(
        self, batch_id: str, expected_count: int | None, model: str, source: str = "script"
    ) -> None:
        """Create the batch record. expected_count None means "not known yet" (a zip)."""
        item: dict[str, Any] = {
            "pk": _pk(batch_id),
            "sk": "META",
            "completed": 0,
            "failed": 0,
            "status": "pending",
            "model": model,
            "source": source,
            "created_at": _now(),
        }
        if expected_count is not None:
            item["expected_count"] = expected_count
        try:
            self.client.put_item(
                TableName=self.table,
                Item={k: _ser.serialize(v) for k, v in item.items()},
                ConditionExpression="attribute_not_exists(pk)",
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                raise BatchExists(f"batch {batch_id!r} already exists") from None
            raise

    def set_expected_count(self, batch_id: str, expected_count: int) -> bool:
        """Set the count once. False if the batch already has one; BatchNotFound if no record."""
        try:
            self.client.update_item(
                TableName=self.table,
                Key=_key(batch_id, "META"),
                UpdateExpression="SET expected_count = :n, updated_at = :now",
                ConditionExpression="attribute_exists(pk) AND attribute_not_exists(expected_count)",
                ExpressionAttributeValues={
                    ":n": {"N": str(expected_count)},
                    ":now": {"S": _now()},
                },
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] != "ConditionalCheckFailedException":
                raise
        if self.get_batch(batch_id) is None:
            raise BatchNotFound(f"no batch record for {batch_id!r}")
        return False

    def get_batch(self, batch_id: str) -> dict | None:
        resp = self.client.get_item(
            TableName=self.table, Key=_key(batch_id, "META"), ConsistentRead=True
        )
        item = resp.get("Item")
        return _plain({k: _deser.deserialize(v) for k, v in item.items()}) if item else None

    def get_image(self, batch_id: str, image_name: str) -> dict | None:
        resp = self.client.get_item(
            TableName=self.table, Key=_key(batch_id, _img_sk(image_name)), ConsistentRead=True
        )
        item = resp.get("Item")
        return _plain({k: _deser.deserialize(v) for k, v in item.items()}) if item else None

    def reject_batch(self, batch_id: str, reason: str) -> None:
        """A batch that can't be processed at all (for example, a broken zip)."""
        self.client.update_item(
            TableName=self.table,
            Key=_key(batch_id, "META"),
            UpdateExpression="SET #s = :rejected, #e = :reason, updated_at = :now",
            ConditionExpression="attribute_exists(pk)",
            ExpressionAttributeNames={"#s": "status", "#e": "error"},
            ExpressionAttributeValues={
                ":rejected": {"S": "rejected"},
                ":reason": {"S": reason[:1000]},
                ":now": {"S": _now()},
            },
        )

    # Finalize --------------------------------------------------------------

    def claim_finalize(self, batch_id: str) -> bool:
        """Move a complete batch from pending to finalizing. True for exactly one caller.

        Counts only ever grow and expected_count is set once, so a batch read as complete
        stays complete; the condition only has to stop a second claim. A claim older than
        FINALIZE_STALE_S (a finalize run that crashed) can be taken again.
        """
        batch = self.get_batch(batch_id)
        if batch is None or not is_complete(batch):
            return False
        now = datetime.now(UTC)
        stale = (now - timedelta(seconds=FINALIZE_STALE_S)).isoformat(timespec="milliseconds")
        try:
            self.client.update_item(
                TableName=self.table,
                Key=_key(batch_id, "META"),
                UpdateExpression="SET #s = :finalizing, finalize_started = :now",
                ConditionExpression=(
                    "#s = :pending OR (#s = :finalizing AND finalize_started < :stale)"
                ),
                ExpressionAttributeNames={"#s": "status"},
                ExpressionAttributeValues={
                    ":pending": {"S": "pending"},
                    ":finalizing": {"S": "finalizing"},
                    ":now": {"S": now.isoformat(timespec="milliseconds")},
                    ":stale": {"S": stale},
                },
            )
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                return False
            raise

    def finish_finalize(self, batch_id: str, fields: dict[str, Any]) -> None:
        """Mark a finalizing batch done and store a few summary fields on its record."""
        values = {":done": "done", ":finalizing": "finalizing", ":now": _now()}
        sets = ["#s = :done", "finalized_at = :now"]
        for i, (name, value) in enumerate(fields.items()):
            sets.append(f"{name} = :f{i}")
            values[f":f{i}"] = value
        self.client.update_item(
            TableName=self.table,
            Key=_key(batch_id, "META"),
            UpdateExpression="SET " + ", ".join(sets),
            ConditionExpression="#s = :finalizing",
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={k: _ser.serialize(_to_ddb(v)) for k, v in values.items()},
        )

    # Image listing ---------------------------------------------------------

    def list_images(
        self, batch_id: str, limit: int = 24, cursor: str | None = None
    ) -> tuple[list[dict], str | None]:
        """One page of a batch's image results, in name order. Returns (items, next_cursor)."""
        kwargs: dict[str, Any] = {
            "TableName": self.table,
            "KeyConditionExpression": "pk = :pk AND begins_with(sk, :img)",
            "ExpressionAttributeValues": {":pk": {"S": _pk(batch_id)}, ":img": {"S": "IMG#"}},
            "Limit": limit,
        }
        if cursor:
            kwargs["ExclusiveStartKey"] = _key(batch_id, _img_sk(cursor))
        resp = self.client.query(**kwargs)
        items = [_plain({k: _deser.deserialize(v) for k, v in i.items()}) for i in resp["Items"]]
        last = resp.get("LastEvaluatedKey")
        return items, last["sk"]["S"].removeprefix("IMG#") if last else None

    def all_images(self, batch_id: str) -> Iterator[dict]:
        """Every image record of a batch (no detections: status, worker, error, time)."""
        kwargs: dict[str, Any] = {
            "TableName": self.table,
            "KeyConditionExpression": "pk = :pk AND begins_with(sk, :img)",
            "ExpressionAttributeValues": {":pk": {"S": _pk(batch_id)}, ":img": {"S": "IMG#"}},
            "ProjectionExpression": "image_name, #s, worker_id, #e, processed_at",
            "ExpressionAttributeNames": {"#s": "status", "#e": "error"},
        }
        while True:
            resp = self.client.query(**kwargs)
            for i in resp["Items"]:
                yield _plain({k: _deser.deserialize(v) for k, v in i.items()})
            if "LastEvaluatedKey" not in resp:
                return
            kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]

    # Skip-finished check ---------------------------------------------------

    def existing(self, batch_id: str, image_names: Iterable[str]) -> set[str]:
        """Which of these images already have a result (done or failed)."""
        names = list(dict.fromkeys(image_names))
        found: set[str] = set()
        for chunk in _chunks(names, MAX_BATCH_GET):
            request = {
                self.table: {
                    "Keys": [_key(batch_id, _img_sk(n)) for n in chunk],
                    "ProjectionExpression": "sk",
                    "ConsistentRead": True,
                }
            }
            attempt = 0
            while request:
                resp = self.client.batch_get_item(RequestItems=request)
                for item in resp.get("Responses", {}).get(self.table, []):
                    found.add(item["sk"]["S"].removeprefix("IMG#"))
                request = resp.get("UnprocessedKeys") or {}
                if request:
                    attempt += 1
                    if attempt >= MAX_ATTEMPTS:
                        raise RuntimeError("BatchGetItem kept returning UnprocessedKeys")
                    _backoff(attempt)
        return found

    # Commit ----------------------------------------------------------------

    def commit(
        self,
        batch_id: str,
        results: Sequence[ImageResult],
        *,
        worker_id: str,
        runtime: str,
        precision: str,
    ) -> CommitResult:
        """Write results and count them exactly once. Safe to call again with the same images."""
        meta = {"worker_id": worker_id, "runtime": runtime, "precision": precision}
        written = skipped = 0
        for chunk in _chunks(list(results), CHUNK):
            w, s = self._commit_chunk(batch_id, list(chunk), meta)
            written += w
            skipped += s
        return CommitResult(written=written, skipped=skipped)

    def _put(self, batch_id: str, r: ImageResult, meta: dict, now: str) -> dict:
        item: dict[str, Any] = {
            "pk": _pk(batch_id),
            "sk": _img_sk(r.image_name),
            "image_name": r.image_name,
            "status": r.status,
            "processed_at": now,
            **meta,
        }
        if r.status == "done":
            item["num_detections"] = len(r.detections)
            item["detections"] = [
                {
                    "box": [_dec(v, 2) for v in d["box"]],
                    "score": _dec(d["score"], 5),
                    "class_id": int(d["class_id"]),
                }
                for d in r.detections
            ]
        else:
            item["error"] = r.error or "unknown error"
        return {
            "Put": {
                "TableName": self.table,
                "Item": {k: _ser.serialize(v) for k, v in item.items()},
                "ConditionExpression": "attribute_not_exists(sk)",
            }
        }

    def _counter(self, batch_id: str, done: int, failed: int, now: str) -> dict:
        return {
            "Update": {
                "TableName": self.table,
                "Key": _key(batch_id, "META"),
                "UpdateExpression": "ADD completed :d, failed :f SET updated_at = :now",
                "ConditionExpression": "attribute_exists(pk)",
                "ExpressionAttributeValues": {
                    ":d": {"N": str(done)},
                    ":f": {"N": str(failed)},
                    ":now": {"S": now},
                },
            }
        }

    def _commit_chunk(
        self, batch_id: str, pending: list[ImageResult], meta: dict
    ) -> tuple[int, int]:
        skipped = 0
        attempt = 0
        while pending:
            now = _now()
            done = sum(r.status == "done" for r in pending)
            items = [self._put(batch_id, r, meta, now) for r in pending]
            items.append(self._counter(batch_id, done, len(pending) - done, now))
            try:
                self.client.transact_write_items(TransactItems=items)
                return len(pending), skipped
            except ClientError as e:
                code = e.response["Error"]["Code"]
                if code == "TransactionCanceledException":
                    reasons = [r.get("Code") for r in e.response.get("CancellationReasons", [])]
                    if len(reasons) == len(items) and reasons[-1] == "ConditionalCheckFailed":
                        raise BatchNotFound(f"no batch record for {batch_id!r}") from None
                    dup = {i for i, c in enumerate(reasons[:-1]) if c == "ConditionalCheckFailed"}
                    if dup:
                        # Already written by an earlier attempt or another worker: drop and retry.
                        skipped += len(dup)
                        pending = [r for i, r in enumerate(pending) if i not in dup]
                        continue
                    if not RETRYABLE_REASONS.intersection(reasons):
                        raise
                elif code not in RETRYABLE_CODES:
                    raise
                attempt += 1
                if attempt >= MAX_ATTEMPTS:
                    raise
                _backoff(attempt)
        return 0, skipped
