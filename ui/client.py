"""A small client for the batch API (detector.api). No AWS credentials: plain HTTPS.

The API only answers requests from the IP addresses in api_allowed_cidrs; everything
else gets 403 from API Gateway. Uploads go straight to S3 through presigned URLs.
"""

from __future__ import annotations

import os
from typing import Any

import requests

TIMEOUT_S = 35  # the API function's own limit is 29 s
IP_HINT = (
    "Your IP isn't allowed. Update api_allowed_cidrs in infra/environments.conf.json "
    "and run make deploy."
)


class ApiError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


def _message(r: requests.Response) -> str:
    if r.status_code == 403:
        return IP_HINT
    try:
        detail = r.json().get("detail") or r.json().get("message")
    except ValueError:
        detail = None
    if isinstance(detail, list):  # FastAPI validation errors
        detail = "; ".join(str(d.get("msg", d)) for d in detail)
    return str(detail or f"HTTP {r.status_code}")


class Api:
    def __init__(self, base_url: str, session: requests.Session | None = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.http = session or requests.Session()

    def _call(self, method: str, path: str, **kwargs: Any) -> dict:
        try:
            r = self.http.request(method, f"{self.base_url}{path}", timeout=TIMEOUT_S, **kwargs)
        except requests.RequestException as e:
            raise ApiError(0, f"Can't reach the API: {e}") from None
        if r.status_code >= 400:
            raise ApiError(r.status_code, _message(r))
        return r.json()

    def health(self) -> bool:
        try:
            return self._call("GET", "/health").get("ok") is True
        except ApiError:
            return False

    def create_batch(
        self, *, files: list[str] | None = None, zip: bool = False, batch_id: str | None = None
    ) -> dict:
        body: dict[str, Any] = {"zip": True} if zip else {"files": files}
        if batch_id:
            body["batch_id"] = batch_id
        return self._call("POST", "/batches", json=body)

    def batch(self, batch_id: str) -> dict:
        return self._call("GET", f"/batches/{batch_id}")

    def images(self, batch_id: str, cursor: str | None = None, limit: int = 100) -> dict:
        params: dict[str, Any] = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return self._call("GET", f"/batches/{batch_id}/images", params=params)

    def image(self, batch_id: str, name: str) -> dict:
        return self._call("GET", f"/batches/{batch_id}/images/{requests.utils.quote(name)}")

    def annotated(self, batch_id: str, name: str) -> dict:
        quoted = requests.utils.quote(name)
        return self._call("GET", f"/batches/{batch_id}/images/{quoted}/annotated")

    def upload(self, url: str, data: bytes, attempts: int = 2) -> None:
        """PUT to a presigned URL, retried once."""
        for attempt in range(attempts):
            try:
                r = self.http.put(url, data=data, timeout=120)
                if r.status_code < 300:
                    return
                error = f"S3 returned HTTP {r.status_code}"
            except requests.RequestException as e:
                error = str(e)
            if attempt == attempts - 1:
                raise ApiError(0, error)

    def download_text(self, url: str) -> str:
        r = self.http.get(url, timeout=120)
        r.raise_for_status()
        return r.text


def api_from_env() -> Api | None:
    url = os.environ.get("API_URL", "").strip()
    return Api(url) if url else None
