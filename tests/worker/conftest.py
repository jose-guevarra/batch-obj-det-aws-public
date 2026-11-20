import json
from dataclasses import dataclass
from urllib.parse import quote_plus

import boto3
import cv2
import numpy as np
import pytest
import torch
from moto import mock_aws

from detector.store import BatchStore
from detector.worker import Worker, WorkerConfig

REGION = "us-west-2"
BUCKET = "detector-batch-test"
TABLE = "detector-results-test"


@pytest.fixture(autouse=True)
def _aws_env(monkeypatch):
    # Fake credentials so nothing can reach real AWS.
    for k, v in {
        "AWS_ACCESS_KEY_ID": "testing",
        "AWS_SECRET_ACCESS_KEY": "testing",
        "AWS_SESSION_TOKEN": "testing",
        "AWS_DEFAULT_REGION": REGION,
    }.items():
        monkeypatch.setenv(k, v)
    monkeypatch.delenv("AWS_PROFILE", raising=False)


class FakeRuntime:
    """Returns one confident 'person' box in the middle of every letterboxed image."""

    name = "fake"
    device = "cpu"
    precision = "fp32"

    def __init__(self):
        self.calls: list[int] = []
        self.on_infer = None  # optional hook, e.g. to request a stop mid-batch

    def sync(self):
        pass

    def infer(self, batch):
        self.calls.append(len(batch))
        if self.on_infer:
            self.on_infer()
        out = torch.zeros(len(batch), 84, 1)
        out[:, :4, 0] = torch.tensor([320.0, 320.0, 100.0, 100.0])
        out[:, 4, 0] = 0.9
        return out


def jpeg(seed: int = 0, h: int = 48, w: int = 64) -> bytes:
    img = np.random.default_rng(seed).integers(0, 255, (h, w, 3), dtype=np.uint8)
    return cv2.imencode(".jpg", img)[1].tobytes()


def s3_event(bucket: str, *keys: str) -> str:
    return json.dumps(
        {
            "Records": [
                {
                    "eventName": "ObjectCreated:Put",
                    "s3": {"bucket": {"name": bucket}, "object": {"key": quote_plus(k)}},
                }
                for k in keys
            ]
        }
    )


@dataclass
class Env:
    sqs: object
    s3: object
    ddb: object
    store: BatchStore
    queue_url: str
    dlq_url: str
    runtime: FakeRuntime

    def config(self, **overrides) -> WorkerConfig:
        base = dict(
            queue_url=self.queue_url,
            dlq_url=self.dlq_url,
            bucket=BUCKET,
            table=TABLE,
            batch_size=16,
            max_wait_s=0.2,
            wait_time_s=0,
            worker_id="w1",
            download_threads=4,
        )
        return WorkerConfig(**(base | overrides))

    def worker(self, s3=None, **overrides) -> Worker:
        return Worker(self.config(**overrides), self.runtime, self.sqs, s3 or self.s3, self.store)

    def upload(self, batch_id: str, n: int, start: int = 0, data: bytes | None = None) -> list[str]:
        names = [f"img{i:04d}.jpg" for i in range(start, start + n)]
        for i, name in enumerate(names):
            body = data if data is not None else jpeg(start + i)
            self.s3.put_object(Bucket=BUCKET, Key=f"input/{batch_id}/{name}", Body=body)
        return names

    def send(self, batch_id: str, names: list[str]) -> None:
        for name in names:
            self.send_body(s3_event(BUCKET, f"input/{batch_id}/{name}"))

    def send_body(self, body: str) -> None:
        self.sqs.send_message(QueueUrl=self.queue_url, MessageBody=body)

    def queue_count(self, url: str | None = None) -> int:
        attrs = self.sqs.get_queue_attributes(
            QueueUrl=url or self.queue_url,
            AttributeNames=["ApproximateNumberOfMessages", "ApproximateNumberOfMessagesNotVisible"],
        )["Attributes"]
        return int(attrs["ApproximateNumberOfMessages"]) + int(
            attrs["ApproximateNumberOfMessagesNotVisible"]
        )

    def dlq_messages(self) -> list[dict]:
        out = []
        while True:
            resp = self.sqs.receive_message(
                QueueUrl=self.dlq_url, MaxNumberOfMessages=10, MessageAttributeNames=["All"]
            )
            if not resp.get("Messages"):
                return out
            out.extend(resp["Messages"])

    def part_files(self, batch_id: str) -> list[dict]:
        resp = self.s3.list_objects_v2(Bucket=BUCKET, Prefix=f"results/{batch_id}/")
        lines = []
        for obj in resp.get("Contents", []):
            body = self.s3.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read().decode()
            lines.extend(json.loads(line) for line in body.splitlines() if line)
        return lines


def _make_env(visibility_timeout: int) -> Env:
    sqs = boto3.client("sqs", region_name=REGION)
    s3 = boto3.client("s3", region_name=REGION)
    ddb = boto3.client("dynamodb", region_name=REGION)
    s3.create_bucket(Bucket=BUCKET, CreateBucketConfiguration={"LocationConstraint": REGION})
    dlq_url = sqs.create_queue(QueueName="images-dlq")["QueueUrl"]
    dlq_arn = sqs.get_queue_attributes(QueueUrl=dlq_url, AttributeNames=["QueueArn"])["Attributes"][
        "QueueArn"
    ]
    queue_url = sqs.create_queue(
        QueueName="images",
        Attributes={
            "VisibilityTimeout": str(visibility_timeout),
            "RedrivePolicy": json.dumps({"deadLetterTargetArn": dlq_arn, "maxReceiveCount": "3"}),
        },
    )["QueueUrl"]
    ddb.create_table(
        TableName=TABLE,
        KeySchema=[
            {"AttributeName": "pk", "KeyType": "HASH"},
            {"AttributeName": "sk", "KeyType": "RANGE"},
        ],
        AttributeDefinitions=[
            {"AttributeName": "pk", "AttributeType": "S"},
            {"AttributeName": "sk", "AttributeType": "S"},
        ],
        BillingMode="PAY_PER_REQUEST",
    )
    return Env(sqs, s3, ddb, BatchStore(ddb, TABLE), queue_url, dlq_url, FakeRuntime())


@pytest.fixture
def env():
    with mock_aws():
        yield _make_env(visibility_timeout=30)


RETRY_VISIBILITY_S = 1


@pytest.fixture
def env_fast_retry():
    """Visibility timeout 1 s: a message left for retry comes back after wait_for_retry()."""
    with mock_aws():
        yield _make_env(visibility_timeout=RETRY_VISIBILITY_S)


def wait_for_retry() -> None:
    import time

    time.sleep(RETRY_VISIBILITY_S + 0.1)
