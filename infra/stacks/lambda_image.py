"""Lambda functions built from docker/Dockerfile.api (arm64, no torch).

The status API, finalize and unzip functions share one image; each picks its handler
with CMD. Every function constructed here builds the same asset, so it's built and
pushed once.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import Duration, RemovalPolicy, Size
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_logs as logs
from constructs import Construct

from config import EnvConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
DOCKERFILE = "docker/Dockerfile.api"


CI_CACHE_CONTEXT = "ci_cache"  # cdk deploy -c ci_cache=1: CI keeps Docker layers in ECR


def build_cache_repo_uri(cfg: EnvConfig) -> str:
    return f"{cfg.account}.dkr.ecr.{cfg.region}.amazonaws.com/{cfg.name('build-cache')}"


def build_cache(scope: Construct, cfg: EnvConfig, name: str) -> dict:
    """Registry cache options for an image asset, only when CI asks for them.

    The cache is a tag (<name>) in the build-cache ECR repo (DetectorCi-<env>), whose tags
    are mutable so every build can overwrite it; the CDK assets repo's tags are immutable.
    A fresh CI runner pulls the layers that didn't change instead of rebuilding them (the
    CUDA layers take ~10 minutes). Without the context flag nothing changes, so local
    builds work as before.
    """
    if scope.node.try_get_context(CI_CACHE_CONTEXT) not in ("1", "true", True):
        return {}
    ref = f"{build_cache_repo_uri(cfg)}:{name}"
    return {
        "cache_from": [ecr_assets.DockerCacheOption(type="registry", params={"ref": ref})],
        "cache_to": ecr_assets.DockerCacheOption(
            type="registry",
            params={
                "ref": ref,
                "mode": "max",
                "image-manifest": "true",
                "oci-mediatypes": "true",
            },
        ),
    }


def image_function(
    scope: Construct,
    construct_id: str,
    *,
    cfg: EnvConfig,
    name: str,
    handler: str,
    memory_mib: int,
    timeout: Duration,
    environment: dict[str, str],
    ephemeral_gib: int | None = None,
) -> lambda_.DockerImageFunction:
    log_group = logs.LogGroup(
        scope,
        f"{construct_id}Logs",
        log_group_name=f"/detector/{cfg.env_name}/{name}",
        retention=logs.RetentionDays.TWO_WEEKS,
        removal_policy=RemovalPolicy.DESTROY,
    )
    return lambda_.DockerImageFunction(
        scope,
        construct_id,
        function_name=cfg.name(name),
        code=lambda_.DockerImageCode.from_image_asset(
            str(REPO_ROOT),
            file=DOCKERFILE,
            platform=ecr_assets.Platform.LINUX_ARM64,
            cmd=[handler],
            **build_cache(scope, cfg, "api"),
        ),
        architecture=lambda_.Architecture.ARM_64,
        memory_size=memory_mib,
        timeout=timeout,
        ephemeral_storage_size=Size.gibibytes(ephemeral_gib) if ephemeral_gib else None,
        environment=environment,
        log_group=log_group,
    )
