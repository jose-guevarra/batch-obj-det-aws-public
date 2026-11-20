"""Batch: live progress while it runs, then summary, downloads and the image browser."""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from ui.client import Api, ApiError
from ui.common import STATUS_ICONS, recent, require_api
from ui.helpers import (
    PHASES,
    elapsed_s,
    filtered_page,
    fmt_duration,
    matches,
    phase,
    results_to_csv,
)

GRID_COLS, GRID_ROWS = 4, 6
PAGE_SIZE = GRID_COLS * GRID_ROWS
FINAL = {"done", "rejected"}

api = require_api()
batch_id = st.query_params.get("batch") or st.session_state.get("batch_id")
if not batch_id:
    st.header("Batch")
    st.info("Create a batch, or open one from the sidebar.")
    st.stop()
st.session_state["batch_id"] = batch_id
if st.query_params.get("batch") != batch_id:
    st.query_params["batch"] = batch_id


def fetch_status() -> dict:
    try:
        return api.batch(batch_id)
    except ApiError as e:
        st.error("No such batch." if e.status == 404 else str(e))
        st.stop()


def header(status: dict) -> None:
    icon = STATUS_ICONS.get(status["status"], "⚪")
    st.header(f"{batch_id}")
    created = (status.get("created_at") or "")[:19].replace("T", " ")
    st.caption(
        f"{icon} **{status['status']}** · created {created} UTC · "
        f"elapsed {fmt_duration(elapsed_s(status))} · source {status.get('source') or '-'}"
    )


def progress_block(status: dict) -> None:
    expected = status.get("expected_count")
    done = status["completed"] + status["failed"]
    name, detail = phase(status)
    with st.container(border=True):
        if expected:
            pct = status.get("percent") or 0.0
            st.progress(min(1.0, pct / 100), text=f"{done} / {expected} ({pct:.0f}%)")
        else:
            st.progress(0.0, text="counting images...")
        c = st.columns(4)
        c[0].metric("Completed", status["completed"])
        c[1].metric("Failed", status["failed"])
        c[2].metric("Remaining", "-" if expected is None else max(0, expected - done))
        c[3].metric("Elapsed", fmt_duration(elapsed_s(status)))
        steps = [f"**{p}**" if p == name else p for p in PHASES]
        st.markdown("Phase: " + " → ".join(steps))
        st.caption(detail)


status = fetch_status()
recent().add(batch_id)
header(status)

if status["status"] == "rejected":
    st.error(status.get("error") or "This batch was rejected.")
    st.stop()

if status["status"] not in FINAL:

    @st.fragment(run_every=3)
    def live() -> None:
        current = fetch_status()
        if current["status"] in FINAL:
            st.rerun()  # the whole page, which now shows the results
        progress_block(current)

    live()
    st.stop()


# Done -----------------------------------------------------------------------------
summary = status["summary"]


@st.cache_data(show_spinner="Downloading results...")
def results_files(_api: Api, batch_id: str, url: str) -> tuple[str, str]:
    jsonl = _api.download_text(url)
    return jsonl, results_to_csv(jsonl)


m = st.columns(6)
m[0].metric("Completed", summary["completed"])
m[1].metric("Failed", summary["failed"])
m[2].metric("Detections", f"{summary['detections']:,}")
mean = summary.get("mean_score")
m[3].metric("Mean confidence", f"{mean:.2f}" if mean is not None else "-")
m[4].metric("Total time", fmt_duration(summary.get("total_s")))
m[5].metric("Est. cost", f"${summary['estimated_cost_usd']:.4f}", help=summary.get("cost_note"))

left, right = st.columns([3, 2])
with left:
    st.subheader("Detections per class")
    per_class = list(summary["detections_per_class"].items())[:15]
    if per_class:
        df = pd.DataFrame(per_class, columns=["class", "detections"])
        chart = (
            alt.Chart(df)
            .mark_bar()
            .encode(
                x=alt.X("detections:Q", title="Detections"),
                y=alt.Y("class:N", sort="-x", title=None),
                tooltip=["class", "detections"],
            )
            .properties(height=max(120, 24 * len(df)))
        )
        st.altair_chart(chart, width="stretch")
    else:
        st.caption("No detections.")
with right:
    st.subheader("Downloads")
    jsonl, csv_text = results_files(api, batch_id, status["results_url"])
    st.download_button(
        "results.csv", csv_text, f"{batch_id}-results.csv", "text/csv", icon=":material/download:"
    )
    st.download_button(
        "results.jsonl",
        jsonl,
        f"{batch_id}-results.jsonl",
        "application/x-ndjson",
        icon=":material/download:",
    )
    if summary["failures"]:
        with st.expander(f"Failed images ({summary['failed']})"):
            st.dataframe(pd.DataFrame(summary["failures"]), hide_index=True, width="stretch")


# Image browser ----------------------------------------------------------------------
st.subheader("Images")
f1, f2, f3 = st.columns([1, 3, 1])
status_filter = f1.selectbox("Status", ["all", "done", "failed"])
class_filter = f2.multiselect("Classes (any of)", list(summary["detections_per_class"]))
min_dets = f3.number_input("Min. detections", min_value=0, value=0, step=1)

# One stack of page-start cursors per batch and filter combination.
filters = (batch_id, status_filter, tuple(class_filter), int(min_dets))
if st.session_state.get("browse_filters") != filters:
    st.session_state["browse_filters"] = filters
    st.session_state["browse_starts"] = [None]
    st.session_state.pop("selected_image", None)
starts: list[str | None] = st.session_state["browse_starts"]


@st.cache_data(ttl=300, show_spinner=False)
def page_of(_api: Api, filters: tuple, start: str | None) -> tuple[list[dict], str | None]:
    batch, status_f, classes, min_d = filters
    return filtered_page(
        lambda cursor: _api.images(batch, cursor=cursor, limit=100),
        start,
        lambda image: matches(image, status_f, classes, min_d),
        PAGE_SIZE,
    )


images, next_start = page_of(api, filters, starts[-1])
if not images:
    st.caption("No images match.")
for row in range(0, len(images), GRID_COLS):
    cols = st.columns(GRID_COLS)
    for col, image in zip(cols, images[row : row + GRID_COLS], strict=False):
        with col.container(border=True):
            st.markdown(f"**{image['image_name']}**")
            if image["status"] == "done":
                top = ", ".join(image["top_classes"]) or "nothing found"
                st.caption(f"{image['num_detections']} detections · {top}")
            else:
                st.caption(f"❌ failed: {(image.get('error') or '')[:60]}")
            if st.button("View", key=f"view-{image['image_name']}"):
                st.session_state["selected_image"] = image["image_name"]

p1, p2, p3 = st.columns([1, 1, 4])
if p1.button("← Previous", disabled=len(starts) == 1):
    starts.pop()
    st.rerun()
if p2.button("Next →", disabled=next_start is None):
    starts.append(next_start)
    st.rerun()
p3.caption(f"Page {len(starts)}")


# Detail panel -----------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def annotated(_api: Api, batch_id: str, name: str) -> dict:
    return _api.annotated(batch_id, name)


selected = st.session_state.get("selected_image")
if selected:
    st.divider()
    try:
        info = api.image(batch_id, selected)
    except ApiError as e:
        st.error(str(e))
        st.stop()
    st.subheader(selected)
    left, right = st.columns([3, 2])
    with left:
        show_original = info["status"] != "done" or st.toggle("Show the original")
        if show_original:
            st.image(info["original_url"], width="stretch")
        else:
            try:
                with st.spinner("Rendering..."):
                    ann = annotated(api, batch_id, selected)
            except ApiError as e:
                st.error(str(e))
            else:
                st.image(ann["url"], width="stretch")
                st.caption(
                    "Served from the cache (rendered earlier)."
                    if ann["cached"]
                    else f"Rendered just now, in {ann['render_ms']} ms; cached from now on."
                )
    with right:
        if info["status"] != "done":
            st.error(f"Failed: {info.get('error')}")
        elif info["detections"]:
            rows = [
                {
                    "class": d["class_name"],
                    "score": round(d["score"], 3),
                    "box (x1, y1, x2, y2)": ", ".join(f"{v:.0f}" for v in d["box"]),
                }
                for d in sorted(info["detections"], key=lambda d: -d["score"])
            ]
            st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        else:
            st.caption("No detections above the confidence threshold.")
        st.caption(f"Processed by {info.get('worker_id')} at {info.get('processed_at')}")
