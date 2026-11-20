#!/usr/bin/env python3
"""Build and push the CDK app's Docker image assets straight to ECR (CI only).

    uv run python scripts/ci_publish_images.py infra/cdk.out

cdk deploy builds each image into the local Docker image store, then pushes it. For the
GPU image that needs ~45-50 GB of disk at once (build state, exported layers and the
unpacked image), more than a standard GitHub runner has. This script reads the synthesized
asset manifests (infra/cdk.out/*.assets.json) and, for every image whose tag isn't in ECR
yet, runs `docker buildx build --push` with the asset's own Dockerfile, platform and layer
cache, on a docker-container builder: nothing is unpacked locally. cdk deploy then finds
every image already in ECR and only deploys.

Credentials: the caller's (the CI deploy role), which may assume the CDK image-publishing
role named in each destination.
"""

from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import boto3


@dataclass(frozen=True)
class ImageJob:
    name: str  # displayName, for the log
    context: Path  # the staged asset directory in cdk.out
    dockerfile: str
    platform: str | None
    cache_from: tuple[dict, ...]
    cache_to: dict | None
    repository: str
    tag: str
    region: str
    role_arn: str | None

    def image_uri(self, registry: str) -> str:
        return f"{registry}/{self.repository}:{self.tag}"


def load_jobs(cdk_out: Path) -> list[ImageJob]:
    """Every (repository, tag) the asset manifests ask for, once."""
    jobs: dict[tuple[str, str], ImageJob] = {}
    for manifest in sorted(cdk_out.glob("*.assets.json")):
        for asset_hash, image in json.loads(manifest.read_text()).get("dockerImages", {}).items():
            source = image["source"]
            if "executable" in source:
                raise SystemExit(f"{asset_hash}: executable image assets aren't supported")
            for dest in image["destinations"].values():
                key = (dest["repositoryName"], dest["imageTag"])
                role = dest.get("assumeRoleArn")
                jobs.setdefault(
                    key,
                    ImageJob(
                        name=image.get("displayName", asset_hash[:12]),
                        context=cdk_out / source["directory"],
                        dockerfile=source.get("dockerFile", "Dockerfile"),
                        platform=source.get("platform"),
                        cache_from=tuple(source.get("cacheFrom", ())),
                        cache_to=source.get("cacheTo"),
                        repository=dest["repositoryName"],
                        tag=dest["imageTag"],
                        region=dest["region"],
                        role_arn=role.replace("${AWS::Partition}", "aws") if role else None,
                    ),
                )
    return list(jobs.values())


def cache_arg(option: dict) -> str:
    params = ",".join(f"{k}={v}" for k, v in option.get("params", {}).items())
    return f"type={option['type']}" + (f",{params}" if params else "")


def build_command(job: ImageJob, registry: str) -> list[str]:
    cmd = ["docker", "buildx", "build", "--push", "--provenance=false", "--sbom=false"]
    cmd += ["--file", str(job.context / job.dockerfile), "--tag", job.image_uri(registry)]
    if job.platform:
        cmd += ["--platform", job.platform]
    for option in job.cache_from:
        cmd += ["--cache-from", cache_arg(option)]
    if job.cache_to:
        cmd += ["--cache-to", cache_arg(job.cache_to)]
    return [*cmd, str(job.context)]


def image_exists(ecr: Any, job: ImageJob) -> bool:
    try:
        ecr.describe_images(repositoryName=job.repository, imageIds=[{"imageTag": job.tag}])
        return True
    except ecr.exceptions.ImageNotFoundException:
        return False


def publishing_session(role_arn: str | None, region: str) -> boto3.Session:
    if not role_arn:
        return boto3.Session(region_name=region)
    creds = boto3.client("sts", region_name=region).assume_role(
        RoleArn=role_arn, RoleSessionName="ci-publish-images"
    )["Credentials"]
    return boto3.Session(
        aws_access_key_id=creds["AccessKeyId"],
        aws_secret_access_key=creds["SecretAccessKey"],
        aws_session_token=creds["SessionToken"],
        region_name=region,
    )


def docker_login(ecr: Any, run: Callable = subprocess.run) -> str:
    """Log Docker in to the account's registry; returns the registry host."""
    data = ecr.get_authorization_token()["authorizationData"][0]
    user, password = base64.b64decode(data["authorizationToken"]).decode().split(":", 1)
    registry = data["proxyEndpoint"].removeprefix("https://")
    run(
        ["docker", "login", "--username", user, "--password-stdin", registry],
        input=password.encode(),
        check=True,
    )
    return registry


def publish(
    jobs: Iterable[ImageJob],
    session_for: Callable[[str | None, str], Any] = publishing_session,
    run: Callable = subprocess.run,
) -> list[str]:
    """Build and push the missing images. Returns the URIs it pushed."""
    pushed = []
    for job in jobs:
        ecr = session_for(job.role_arn, job.region).client("ecr")
        if image_exists(ecr, job):
            print(f"{job.name}: {job.repository}:{job.tag[:12]} already in ECR", flush=True)
            continue
        registry = docker_login(ecr, run)
        print(f"{job.name}: building and pushing {job.tag[:12]}", flush=True)
        run(build_command(job, registry), check=True)
        pushed.append(job.image_uri(registry))
    return pushed


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("cdk_out", type=Path, help="the synthesized cloud assembly (infra/cdk.out)")
    args = p.parse_args()
    jobs = load_jobs(args.cdk_out)
    if not jobs:
        sys.exit(f"no Docker image assets in {args.cdk_out}")
    pushed = publish(jobs)
    print(f"{len(pushed)} image(s) pushed, {len(jobs) - len(pushed)} already in ECR")


if __name__ == "__main__":
    main()
