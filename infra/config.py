"""Load and validate per-environment deployment settings.

The settings live in ``infra/environments.conf.json`` (gitignored), keyed by
environment name. ``infra/environments.conf.example.json`` shows the shape.
Set ``ENVIRONMENTS_CONF`` to read a file from somewhere else.
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import dataclass, fields
from pathlib import Path

ENV_VAR = "ENVIRONMENTS_CONF"
DEFAULT_PATH = Path(__file__).parent / "environments.conf.json"

_ENV_NAME_RE = re.compile(r"^[a-z][a-z0-9]{0,9}$")
# owner/name, optionally with GitHub's immutable IDs (owner@123/name@456), as GitHub puts
# them in OIDC subjects: repo:jose-guevarra@55925602/batch-obj-det-aws@1447631464:...
_GITHUB_REPO_RE = re.compile(r"^[A-Za-z0-9-]{1,39}(@[0-9]+)?/[A-Za-z0-9._-]{1,100}(@[0-9]+)?$")
_ACCOUNT_RE = re.compile(r"^\d{12}$")
_REGION_RE = re.compile(r"^[a-z]{2}(-[a-z]+)+-\d$")


class ConfigError(ValueError):
    """The environments config is missing, malformed or invalid."""


@dataclass(frozen=True)
class EnvConfig:
    env_name: str
    account: str
    region: str
    retain_data: bool
    api_allowed_cidrs: tuple[str, ...]
    model: str
    gpu_instance_type: str
    cpu_instance_type: str
    gpu_max_instances: int
    cpu_max_instances: int
    # The only repo whose Actions runs may assume the CI roles, exactly as it appears in the
    # OIDC subject: owner@id/name@id when GitHub includes immutable IDs (see README, CI/CD).
    github_repo: str

    def name(self, base: str) -> str:
        """Name for a resource that needs an explicit, environment-unique name."""
        return f"detector-{base}-{self.env_name}"

    @property
    def bucket_name(self) -> str:
        """S3 names are global across all accounts, so include account and region."""
        return f"detector-batch-{self.env_name}-{self.account}-{self.region}"


# Keys expected in each environment's JSON object, with their JSON types.
_SCHEMA: dict[str, type] = {
    "account": str,
    "region": str,
    "retain_data": bool,
    "api_allowed_cidrs": list,
    "model": str,
    "gpu_instance_type": str,
    "cpu_instance_type": str,
    "gpu_max_instances": int,
    "cpu_max_instances": int,
    "github_repo": str,
}
assert set(_SCHEMA) == {f.name for f in fields(EnvConfig)} - {"env_name"}


def _check_type(env_name: str, key: str, value: object, expected: type) -> None:
    # bool is a subclass of int; don't accept true/false where a count is expected.
    ok = isinstance(value, expected) and not (expected is int and isinstance(value, bool))
    if not ok:
        raise ConfigError(
            f"{env_name}.{key}: expected {expected.__name__}, got {type(value).__name__}"
        )


def parse_env(env_name: str, raw: object) -> EnvConfig:
    """Validate one environment's settings and return them as an EnvConfig."""
    if not _ENV_NAME_RE.match(env_name):
        raise ConfigError(
            f"environment name {env_name!r} must be lowercase letters and digits, "
            "start with a letter, and be at most 10 characters"
        )
    if not isinstance(raw, dict):
        raise ConfigError(f"{env_name}: expected an object, got {type(raw).__name__}")

    missing = sorted(set(_SCHEMA) - set(raw))
    unknown = sorted(set(raw) - set(_SCHEMA))
    if missing:
        raise ConfigError(f"{env_name}: missing keys: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{env_name}: unknown keys: {', '.join(unknown)}")

    for key, expected in _SCHEMA.items():
        _check_type(env_name, key, raw[key], expected)

    if not _ACCOUNT_RE.match(raw["account"]):
        raise ConfigError(f"{env_name}.account: must be 12 digits, got {raw['account']!r}")
    if not _REGION_RE.match(raw["region"]):
        raise ConfigError(f"{env_name}.region: not a valid AWS region: {raw['region']!r}")

    cidrs = raw["api_allowed_cidrs"]
    if not cidrs:
        raise ConfigError(f"{env_name}.api_allowed_cidrs: must list at least one CIDR")
    for cidr in cidrs:
        if not isinstance(cidr, str):
            raise ConfigError(f"{env_name}.api_allowed_cidrs: entries must be strings")
        try:
            ipaddress.ip_network(cidr, strict=True)
        except ValueError as e:
            raise ConfigError(f"{env_name}.api_allowed_cidrs: {e}") from None

    if not _GITHUB_REPO_RE.match(raw["github_repo"]):
        raise ConfigError(
            f"{env_name}.github_repo: expected owner/name or owner@id/name@id, "
            f"got {raw['github_repo']!r}"
        )

    for key in ("gpu_max_instances", "cpu_max_instances"):
        if raw[key] < 0:
            raise ConfigError(f"{env_name}.{key}: must be 0 or more")

    return EnvConfig(
        env_name=env_name,
        account=raw["account"],
        region=raw["region"],
        retain_data=raw["retain_data"],
        api_allowed_cidrs=tuple(cidrs),
        model=raw["model"],
        gpu_instance_type=raw["gpu_instance_type"],
        cpu_instance_type=raw["cpu_instance_type"],
        gpu_max_instances=raw["gpu_max_instances"],
        cpu_max_instances=raw["cpu_max_instances"],
        github_repo=raw["github_repo"],
    )


def load_config(env_name: str, path: str | Path | None = None) -> EnvConfig:
    """Read the environments file and return the settings for ``env_name``."""
    if path is None:
        path = os.environ.get(ENV_VAR) or DEFAULT_PATH
    path = Path(path)
    if not path.is_file():
        raise ConfigError(
            f"{path} not found. Copy infra/environments.conf.example.json to "
            f"infra/environments.conf.json and fill it in, or set {ENV_VAR}."
        )
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"{path}: invalid JSON: {e}") from None
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected an object keyed by environment name")
    if env_name not in data:
        known = ", ".join(sorted(data)) or "none"
        raise ConfigError(f"environment {env_name!r} not in {path} (defined: {known})")
    return parse_env(env_name, data[env_name])
