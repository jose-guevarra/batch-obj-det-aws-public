import json
from pathlib import Path

import pytest

from config import ConfigError, load_config, parse_env

EXAMPLE = Path(__file__).parents[2] / "infra" / "environments.conf.example.json"


def example_dev() -> dict:
    return json.loads(EXAMPLE.read_text())["dev"]


def test_example_file_loads():
    cfg = load_config("dev", EXAMPLE)
    assert cfg.env_name == "dev"
    assert cfg.region == "us-west-2"
    assert cfg.api_allowed_cidrs == ("203.0.113.7/32",)


def test_resource_names_carry_env_suffix():
    cfg = load_config("dev", EXAMPLE)
    assert cfg.name("results") == "detector-results-dev"
    assert cfg.bucket_name == "detector-batch-dev-123456789012-us-west-2"


def test_env_var_overrides_path(tmp_path, monkeypatch):
    path = tmp_path / "conf.json"
    path.write_text(EXAMPLE.read_text())
    monkeypatch.setenv("ENVIRONMENTS_CONF", str(path))
    assert load_config("dev").account == "123456789012"


def test_unknown_environment_lists_defined_ones():
    with pytest.raises(ConfigError, match=r"'nope' not in .*defined: dev"):
        load_config("nope", EXAMPLE)


def test_missing_file_explains_how_to_create_it(tmp_path):
    with pytest.raises(ConfigError, match="Copy infra/environments.conf.example.json"):
        load_config("dev", tmp_path / "absent.json")


def test_missing_key_rejected():
    raw = example_dev()
    del raw["model"]
    with pytest.raises(ConfigError, match="missing keys: model"):
        parse_env("dev", raw)


def test_unknown_key_rejected():
    raw = example_dev() | {"sagemaker_model": "yolov8n"}
    with pytest.raises(ConfigError, match="unknown keys: sagemaker_model"):
        parse_env("dev", raw)


@pytest.mark.parametrize("account", ["12345", "12345678901a", 123456789012])
def test_bad_account_rejected(account):
    raw = example_dev() | {"account": account}
    with pytest.raises(ConfigError, match="account"):
        parse_env("dev", raw)


@pytest.mark.parametrize("cidrs", [[], ["10.0.0.1/24"], ["not-a-cidr"], [42]])
def test_bad_cidrs_rejected(cidrs):
    raw = example_dev() | {"api_allowed_cidrs": cidrs}
    with pytest.raises(ConfigError, match="api_allowed_cidrs"):
        parse_env("dev", raw)


def test_bool_not_accepted_as_count():
    raw = example_dev() | {"gpu_max_instances": True}
    with pytest.raises(ConfigError, match="gpu_max_instances: expected int"):
        parse_env("dev", raw)


@pytest.mark.parametrize("name", ["Dev", "1dev", "dev-1", "averylongenvname"])
def test_bad_env_name_rejected(name):
    with pytest.raises(ConfigError, match="environment name"):
        parse_env(name, example_dev())


@pytest.mark.parametrize(
    "repo",
    [
        "jose-guevarra",
        "a/b/c",
        "owner/",
        "/name",
        "owner/na me",
        "*/batch-obj-det-aws",
        "owner@/name",
        "owner@12a/name",
    ],
)
def test_github_repo_must_be_owner_slash_name(repo):
    with pytest.raises(ConfigError, match="github_repo"):
        parse_env("dev", example_dev() | {"github_repo": repo})


@pytest.mark.parametrize("repo", ["owner/name", "owner@55925602/name@1447631464"])
def test_github_repo_accepts_names_and_immutable_ids(repo):
    assert parse_env("dev", example_dev() | {"github_repo": repo}).github_repo == repo
