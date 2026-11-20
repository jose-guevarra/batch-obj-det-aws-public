"""Extract a zip from upload/{batch_id}/ into input/{batch_id}/ (Lambda: detector.unzip.handler).

The batch's expected_count is set from the zip's contents *before* any image is written,
so the batch can't look complete early. Names are flattened to their base names (which
also rules out path traversal); base names that repeat get a -2, -3 ... suffix, so every
counted image lands as its own object. Hidden files, macOS __MACOSX entries and anything
that isn't an image are skipped. Limits on entry count and total size guard against
zip bombs.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import zipfile
from collections.abc import Iterable
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote_plus

from detector.messages import BATCH_ID_RE, UPLOAD_PREFIX, input_key
from detector.store import BatchNotFound, BatchStore

log = logging.getLogger(__name__)
log.setLevel(logging.INFO)

IMAGE_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
    ".tif": "image/tiff",
    ".tiff": "image/tiff",
}
MAX_ENTRIES = 10_000
MAX_TOTAL_BYTES = 8 * 1024**3
UPLOAD_THREADS = 16
MAX_IN_FLIGHT = 64


class ZipRejected(ValueError):
    """The zip breaks a limit; nothing is extracted."""


def batch_id_from_key(key: str) -> str:
    """upload/{batch_id}/{name}.zip -> batch_id."""
    parts = key.split("/")
    if len(parts) != 3 or f"{parts[0]}/" != UPLOAD_PREFIX or not parts[2].endswith(".zip"):
        raise ValueError(f"not an upload/{{batch_id}}/*.zip key: {key!r}")
    if not BATCH_ID_RE.match(parts[1]):
        raise ValueError(f"bad batch id in {key!r}")
    return parts[1]


def plan_entries(infos: Iterable[zipfile.ZipInfo]) -> list[tuple[zipfile.ZipInfo, str]]:
    """The image entries to extract and their flattened, unique names."""
    chosen: list[tuple[zipfile.ZipInfo, str]] = []
    used: set[str] = set()
    total = 0
    for info in infos:
        path = PurePosixPath(info.filename.replace("\\", "/"))
        name = path.name
        if info.is_dir() or "__MACOSX" in path.parts or not name or name.startswith("."):
            continue
        suffix = PurePosixPath(name).suffix.lower()
        if suffix not in IMAGE_TYPES:
            continue
        stem, n = name[: -len(suffix)], 2
        while name.lower() in used:
            name = f"{stem}-{n}{suffix}"
            n += 1
        used.add(name.lower())
        chosen.append((info, name))
        total += info.file_size
        if len(chosen) > MAX_ENTRIES:
            raise ZipRejected(f"more than {MAX_ENTRIES} images")
        if total > MAX_TOTAL_BYTES:
            raise ZipRejected(f"more than {MAX_TOTAL_BYTES // 1024**3} GB uncompressed")
    return chosen


def extract(store: BatchStore, s3: Any, bucket: str, key: str, workdir: str) -> dict:
    batch_id = batch_id_from_key(key)
    local = os.path.join(workdir, "batch.zip")
    s3.download_file(bucket, key, local)
    zf = None
    try:
        zf = zipfile.ZipFile(local)
        entries = plan_entries(zf.infolist())
    except (zipfile.BadZipFile, ZipRejected) as e:
        if zf is not None:
            zf.close()
        reason = f"zip rejected: {e}"
        try:
            store.reject_batch(batch_id, reason)
        except Exception:
            log.exception("could not mark %s rejected", batch_id)
        log.warning(json.dumps({"event": "unzip_rejected", "batch_id": batch_id, "reason": reason}))
        return {"event": "unzip_rejected", "batch_id": batch_id, "reason": reason}
    with zf:
        count = len(entries)
        try:
            counted = store.set_expected_count(batch_id, count)
        except BatchNotFound:
            # Uploaded without POST /batches: create the record here, count included.
            store.create_batch(batch_id, count, "yolov8m", source="zip")
            counted = True
        if not counted:
            log.info(json.dumps({"event": "unzip_recount_skipped", "batch_id": batch_id}))
        with ThreadPoolExecutor(UPLOAD_THREADS) as pool:
            pending: set[Future] = set()
            for info, name in entries:
                data = zf.read(info)
                pending.add(
                    pool.submit(
                        s3.put_object,
                        Bucket=bucket,
                        Key=input_key(batch_id, name),
                        Body=data,
                        ContentType=IMAGE_TYPES[PurePosixPath(name).suffix.lower()],
                    )
                )
                if len(pending) >= MAX_IN_FLIGHT:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for f in done:
                        f.result()
            for f in pending:
                f.result()
    out = {"event": "unzipped", "batch_id": batch_id, "images": count}
    log.info(json.dumps(out))
    return out


def handler(event: dict, _context: Any) -> dict:
    import boto3

    store = BatchStore(boto3.client("dynamodb"), os.environ["TABLE"])
    s3 = boto3.client("s3")
    results = []
    for record in event.get("Records", []):
        bucket = record["s3"]["bucket"]["name"]
        key = unquote_plus(record["s3"]["object"]["key"])
        with tempfile.TemporaryDirectory() as workdir:
            results.append(extract(store, s3, bucket, key, workdir))
    return {"results": results}
