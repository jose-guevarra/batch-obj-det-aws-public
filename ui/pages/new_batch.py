"""New batch: pick images or a zip, create the batch, upload through presigned URLs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from detector.messages import BATCH_ID_RE
from ui.client import ApiError
from ui.common import open_batch, recent, require_api
from ui.helpers import default_batch_id, fmt_bytes

IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"]
MAX_FILES = 1000
UPLOAD_THREADS = 8

api = require_api()
st.header("New batch")

mode = st.radio("Upload", ["Images", "Zip file"], horizontal=True, label_visibility="collapsed")
if mode == "Images":
    files = st.file_uploader("Images", type=IMAGE_TYPES, accept_multiple_files=True) or []
else:
    zip_file = st.file_uploader("Zip file", type=["zip"])
    files = [zip_file] if zip_file else []

too_many = mode == "Images" and len(files) > MAX_FILES
if files:
    total = sum(f.size for f in files)
    st.caption(f"{len(files)} file{'s' if len(files) != 1 else ''}, {fmt_bytes(total)}")
if too_many:
    st.error(f"At most {MAX_FILES} images here. For larger sets, use `scripts/upload_batch.py`.")

if "new_batch_id" not in st.session_state:
    st.session_state["new_batch_id"] = default_batch_id()
batch_id = st.text_input(
    "Batch ID", key="new_batch_id", help="Letters, digits, - and _; at most 64 characters."
).strip()
bad_id = not BATCH_ID_RE.match(batch_id)
if bad_id:
    st.error("Batch ID: letters, digits, - and _ only, at most 64 characters.")

st.info(
    "Uploading starts a GPU worker from zero: about 4-5 minutes before the first results, "
    "and about $0.10-0.15 per cold start. Viewing results later doesn't start anything.",
    icon=":material/info:",
)


def upload_all(uploads: list[dict], by_name: dict) -> list[tuple[str, str]]:
    """PUT every file to its presigned URL. Returns (name, error) for the ones that failed."""
    bar = st.progress(0.0, text=f"Uploading 0/{len(uploads)}")
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(UPLOAD_THREADS) as pool:
        futures = {
            pool.submit(api.upload, u["url"], by_name[u["name"]].getvalue()): u["name"]
            for u in uploads
        }
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                future.result()
            except ApiError as e:
                failed.append((futures[future], str(e)))
            bar.progress(done / len(uploads), text=f"Uploading {done}/{len(uploads)}")
    return failed


if st.button("Upload and process", type="primary", disabled=not files or too_many or bad_id):
    try:
        if mode == "Images":
            created = api.create_batch(files=[f.name for f in files], batch_id=batch_id)
            by_name = {f.name: f for f in files}
        else:
            created = api.create_batch(zip=True, batch_id=batch_id)
            by_name = {"batch.zip": files[0]}
    except ApiError as e:
        st.error(str(e))
        st.stop()
    new_id = created["batch_id"]
    recent().add(new_id)
    failed = upload_all(created["uploads"], by_name)
    del st.session_state["new_batch_id"]  # a fresh default for the next batch
    if not failed:
        open_batch(new_id)
    st.session_state["failed_uploads"] = {
        "batch_id": new_id,
        "uploads": [u for u in created["uploads"] if u["name"] in {n for n, _ in failed}],
    }

retry = st.session_state.get("failed_uploads")
if retry and retry["uploads"]:
    names = [u["name"] for u in retry["uploads"]]
    st.warning(
        f"{len(names)} upload(s) to batch {retry['batch_id']} failed. The batch can't finish "
        "until they're uploaded: retry while the files are still selected above (the upload "
        "links last an hour)."
    )
    st.caption(", ".join(names[:20]) + (" ..." if len(names) > 20 else ""))
    by_name = (
        {f.name: f for f in files} if mode == "Images" else {"batch.zip": files[0]} if files else {}
    )
    missing = [n for n in names if n not in by_name]
    c1, c2 = st.columns(2)
    if c1.button("Retry failed uploads", disabled=bool(missing)):
        still = upload_all(retry["uploads"], by_name)
        if not still:
            del st.session_state["failed_uploads"]
            open_batch(retry["batch_id"])
        retry["uploads"] = [u for u in retry["uploads"] if u["name"] in {n for n, _ in still}]
        st.rerun()
    if c2.button("Open the batch anyway"):
        open_batch(retry["batch_id"])
