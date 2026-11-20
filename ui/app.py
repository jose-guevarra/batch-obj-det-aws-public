"""Streamlit front end for the batch detector. Run from infra/ with `make ui`.

Talks only to the batch API (API_URL); it holds no AWS credentials.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # so `ui.*` imports work

import streamlit as st  # noqa: E402

from ui.common import require_api, sidebar  # noqa: E402

st.set_page_config(page_title="Batch detector", page_icon="🔎", layout="wide")

api = require_api()
page = st.navigation(
    [
        st.Page("pages/new_batch.py", title="New batch", icon=":material/upload:", default=True),
        st.Page("pages/batch.py", title="Batch", icon=":material/monitoring:"),
    ]
)
sidebar(api)
page.run()
