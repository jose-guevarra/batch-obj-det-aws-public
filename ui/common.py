"""Streamlit pieces shared by the pages: the API client, the sidebar, navigation."""

from __future__ import annotations

import streamlit as st

from detector.messages import BATCH_ID_RE
from ui.client import Api, ApiError, api_from_env
from ui.helpers import RecentBatches

BATCH_PAGE = "pages/batch.py"
STATUS_ICONS = {"pending": "🟡", "finalizing": "🔵", "done": "🟢", "rejected": "🔴"}


@st.cache_resource
def get_api() -> Api | None:
    return api_from_env()


def require_api() -> Api:
    api = get_api()
    if api is None:
        st.error("API_URL isn't set. Start the UI from infra/ with `make ui`.")
        st.stop()
    return api


@st.cache_data(ttl=30, show_spinner=False)
def api_healthy(_api: Api, url: str) -> bool:
    return _api.health()


@st.cache_data(ttl=15, show_spinner=False)
def batch_status(_api: Api, url: str, batch_id: str) -> str:
    try:
        return _api.batch(batch_id)["status"]
    except ApiError as e:
        return "missing" if e.status == 404 else "?"


def recent() -> RecentBatches:
    return RecentBatches()


def open_batch(batch_id: str) -> None:
    recent().add(batch_id)
    st.session_state["batch_id"] = batch_id
    st.switch_page(BATCH_PAGE, query_params={"batch": batch_id})


def sidebar(api: Api) -> None:
    with st.sidebar:
        st.title("Batch detector")
        ok = api_healthy(api, api.base_url)
        st.caption(f"{'🟢' if ok else '🔴'} API: `{api.base_url}`")
        if not ok:
            st.caption(
                "Can't reach the API. If your IP changed, update `api_allowed_cidrs` in "
                "`infra/environments.conf.json` and run `make deploy`."
            )

        st.subheader("Recent batches")
        ids = recent().load()
        if not ids:
            st.caption("None yet.")
        for batch_id in ids:
            status = batch_status(api, api.base_url, batch_id) if ok else "?"
            icon = STATUS_ICONS.get(status, "⚪")
            if st.button(
                f"{icon} {batch_id} · {status}", key=f"recent-{batch_id}", width="stretch"
            ):
                open_batch(batch_id)

        with st.form("open-batch", border=False):
            batch_id = st.text_input("Open a batch by ID", placeholder="coco-20261106-120000")
            if st.form_submit_button("Open") and batch_id.strip():
                if BATCH_ID_RE.match(batch_id.strip()):
                    open_batch(batch_id.strip())
                else:
                    st.error("Letters, digits, - and _ only.")
