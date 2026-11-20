import base64
import importlib.util
import sys
from pathlib import Path

import aws_cdk as cdk
import pytest

from config import load_config
from stacks import build_stacks

ROOT = Path(__file__).parents[2]
EXAMPLE = ROOT / "infra" / "environments.conf.example.json"
CACHE_REPO = "123456789012.dkr.ecr.us-west-2.amazonaws.com/detector-build-cache-dev"
ASSETS_REPO = "cdk-hnb659fds-container-assets-123456789012-us-west-2"


def load_script():
    spec = importlib.util.spec_from_file_location(
        "ci_publish_images", ROOT / "scripts" / "ci_publish_images.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script()


@pytest.fixture(scope="module")
def cdk_out() -> Path:
    app = cdk.App(context={"ci_cache": "1"})
    build_stacks(app, load_config("dev", EXAMPLE))
    return Path(app.synth().directory)


def test_load_jobs_reads_each_image_once(script, cdk_out):
    jobs = script.load_jobs(cdk_out)
    by_file = {Path(j.dockerfile).name: j for j in jobs}
    assert set(by_file) == {"Dockerfile.gpu", "Dockerfile.cpu", "Dockerfile.api"}
    gpu, api = by_file["Dockerfile.gpu"], by_file["Dockerfile.api"]
    assert (gpu.platform, api.platform) == ("linux/amd64", "linux/arm64")
    assert gpu.repository == ASSETS_REPO and len(gpu.tag) == 64
    assert gpu.role_arn == (
        "arn:aws:iam::123456789012:role/cdk-hnb659fds-image-publishing-role-123456789012-us-west-2"
    )
    assert gpu.cache_from[0]["params"]["ref"] == f"{CACHE_REPO}:gpu"
    assert gpu.context.is_dir() and (gpu.context / gpu.dockerfile).is_file()


def test_build_command_pushes_with_the_layer_cache_and_no_local_load(script, cdk_out):
    gpu = next(j for j in script.load_jobs(cdk_out) if j.dockerfile.endswith("gpu"))
    cmd = script.build_command(gpu, "123456789012.dkr.ecr.us-west-2.amazonaws.com")
    assert cmd[:3] == ["docker", "buildx", "build"]
    assert "--push" in cmd and "--load" not in cmd
    assert "--provenance=false" in cmd  # a plain image manifest, as ECS and Lambda expect
    assert cmd[cmd.index("--platform") + 1] == "linux/amd64"
    assert cmd[cmd.index("--tag") + 1].endswith(f"/{ASSETS_REPO}:{gpu.tag}")
    assert cmd[cmd.index("--cache-from") + 1] == f"type=registry,ref={CACHE_REPO}:gpu"
    assert cmd[cmd.index("--cache-to") + 1] == (
        f"type=registry,ref={CACHE_REPO}:gpu,mode=max,image-manifest=true,oci-mediatypes=true"
    )
    assert cmd[-1] == str(gpu.context)


class FakeEcr:
    class exceptions:  # noqa: N801  (mirrors the boto3 client attribute)
        class ImageNotFoundException(Exception):
            pass

    def __init__(self, existing: set[str]):
        self.existing = existing

    def describe_images(self, repositoryName, imageIds):  # noqa: N803
        if imageIds[0]["imageTag"] not in self.existing:
            raise self.exceptions.ImageNotFoundException()
        return {}

    def get_authorization_token(self):
        token = base64.b64encode(b"AWS:secret").decode()
        return {
            "authorizationData": [
                {"authorizationToken": token, "proxyEndpoint": "https://1.dkr.ecr.x.amazonaws.com"}
            ]
        }


def test_publish_builds_only_missing_images(script, cdk_out):
    jobs = script.load_jobs(cdk_out)
    existing = {jobs[0].tag}
    calls = []

    class Session:
        def client(self, name):
            assert name == "ecr"
            return FakeEcr(existing)

    def run(cmd, check, input=None):  # noqa: A002
        calls.append((cmd, input))

    pushed = script.publish(jobs, session_for=lambda role, region: Session(), run=run)
    assert len(pushed) == len(jobs) - 1
    logins = [c for c in calls if c[0][:2] == ["docker", "login"]]
    builds = [c for c in calls if c[0][:3] == ["docker", "buildx", "build"]]
    assert len(builds) == len(jobs) - 1
    assert all(inp == b"secret" for _, inp in logins)  # the password goes on stdin
    assert all(jobs[0].tag not in " ".join(cmd) for cmd, _ in builds)
