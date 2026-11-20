import pytest
from fastapi.testclient import TestClient

from detector.api import Backend, app, aws_backend, presign_client
from tests.worker.conftest import BUCKET, REGION, _aws_env, env  # noqa: F401  (fixtures)


@pytest.fixture
def api(env):  # noqa: F811
    calls: list[str] = []
    backend = Backend(
        store=env.store,
        s3=env.s3,
        presign=presign_client(REGION),
        bucket=BUCKET,
        finalize=calls.append,
    )
    app.dependency_overrides[aws_backend] = lambda: backend
    client = TestClient(app)
    client.env = env
    client.finalize_calls = calls
    yield client
    app.dependency_overrides.clear()
