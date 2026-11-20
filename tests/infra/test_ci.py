import json
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from config import load_config
from stacks import build_stacks

EXAMPLE = Path(__file__).parents[2] / "infra" / "environments.conf.example.json"
ACCOUNT, REGION = "123456789012", "us-west-2"
REPO = "jose-guevarra/batch-obj-det-aws"
OIDC = "token.actions.githubusercontent.com"


@pytest.fixture(scope="module")
def ci() -> Template:
    return Template.from_stack(build_stacks(cdk.App(), load_config("dev", EXAMPLE)).ci)


def bootstrap(kind: str) -> str:
    return f"arn:aws:iam::{ACCOUNT}:role/cdk-hnb659fds-{kind}-role-{ACCOUNT}-{REGION}"


def role(ci: Template, name: str) -> dict:
    (props,) = [
        r["Properties"]
        for r in ci.find_resources("AWS::IAM::Role").values()
        if r["Properties"].get("RoleName") == name
    ]
    return props


def policy_statements(ci: Template, prefix: str) -> list[dict]:
    (policy,) = [
        p for k, p in ci.find_resources("AWS::IAM::Policy").items() if k.startswith(prefix)
    ]
    return policy["Properties"]["PolicyDocument"]["Statement"]


def trust(props: dict) -> dict:
    (statement,) = props["AssumeRolePolicyDocument"]["Statement"]
    return statement


def test_build_cache_repo_is_mutable_and_writable_by_the_image_publishing_role(ci):
    (repo,) = ci.find_resources("AWS::ECR::Repository").values()
    props = repo["Properties"]
    assert props["RepositoryName"] == "detector-build-cache-dev"
    assert props["ImageTagMutability"] == "MUTABLE"  # every build overwrites its cache tag
    rules = json.loads(props["LifecyclePolicy"]["LifecyclePolicyText"])["rules"]
    assert rules[0]["selection"]["tagStatus"] == "untagged"
    (statement,) = props["RepositoryPolicyText"]["Statement"]
    assert statement["Principal"] == {"AWS": bootstrap("image-publishing")}
    assert "ecr:PutImage" in statement["Action"] and "ecr:BatchGetImage" in statement["Action"]
    assert not any(a in ("ecr:*", "*") or "Delete" in a for a in statement["Action"])
    assert "BuildCacheRepo" in ci.find_outputs("*")


def test_provider_is_imported_not_created(ci):
    ci.resource_count_is("Custom::AWSCDKOpenIdConnectProvider", 0)
    ci.resource_count_is("AWS::IAM::OIDCProvider", 0)
    assert len(ci.find_resources("AWS::IAM::Role")) == 2


@pytest.mark.parametrize(
    "name, sub",
    [
        ("detector-gh-deploy-dev", f"repo:{REPO}:environment:dev"),
        ("detector-gh-pr-dev", f"repo:{REPO}:pull_request"),
    ],
)
def test_trust_is_exact(ci, name, sub):
    statement = trust(role(ci, name))
    assert statement["Action"] == "sts:AssumeRoleWithWebIdentity"
    assert statement["Principal"]["Federated"] == f"arn:aws:iam::{ACCOUNT}:oidc-provider/{OIDC}"
    assert statement["Condition"] == {
        "StringEquals": {f"{OIDC}:aud": "sts.amazonaws.com", f"{OIDC}:sub": sub}
    }
    assert "*" not in json.dumps(statement["Condition"])


def test_deploy_role_may_only_assume_the_bootstrap_roles(ci):
    (statement,) = policy_statements(ci, "DeployRole")
    assert statement["Action"] == "sts:AssumeRole"
    assert sorted(statement["Resource"]) == sorted(
        bootstrap(k) for k in ("deploy", "file-publishing", "image-publishing", "lookup")
    )
    assert role(ci, "detector-gh-deploy-dev")["MaxSessionDuration"] == 7200


def test_pr_role_may_only_assume_the_lookup_role(ci):
    (statement,) = policy_statements(ci, "PrRole")
    assert statement == {
        "Action": "sts:AssumeRole",
        "Effect": "Allow",
        "Resource": bootstrap("lookup"),
    }


def test_outputs(ci):
    assert {"DeployRoleArn", "PrRoleArn"} <= set(ci.find_outputs("*"))


def _docker_sources(ci_cache: str | None) -> dict[str, dict]:
    app = cdk.App(context={"ci_cache": ci_cache} if ci_cache else None)
    build_stacks(app, load_config("dev", EXAMPLE))
    directory = Path(app.synth().directory)
    sources = {}
    for manifest in directory.glob("*.assets.json"):
        for h, image in json.loads(manifest.read_text()).get("dockerImages", {}).items():
            sources[h] = image["source"]
    return sources


def test_ci_cache_flag_adds_a_registry_cache_and_keeps_asset_hashes():
    plain, cached = _docker_sources(None), _docker_sources("1")
    assert set(plain) == set(cached) and len(plain) == 3  # same hashes: shared ECR tags
    assert all("cacheFrom" not in s and "cacheTo" not in s for s in plain.values())
    refs = set()
    for source in cached.values():
        (cache_from,) = source["cacheFrom"]
        ref = cache_from["params"]["ref"]
        assert source["cacheTo"]["params"] == {
            "ref": ref, "mode": "max", "image-manifest": "true", "oci-mediatypes": "true"
        }  # fmt: skip
        repo, tag = ref.rsplit(":", 1)
        assert repo == f"{ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/detector-build-cache-dev"
        refs.add(tag)
    assert refs == {"gpu", "cpu", "api"}


def test_trust_uses_github_repo_verbatim_with_immutable_ids():
    from dataclasses import replace

    repo = "jose-guevarra@55925602/batch-obj-det-aws@1447631464"
    cfg = replace(load_config("dev", EXAMPLE), github_repo=repo)
    t = Template.from_stack(build_stacks(cdk.App(), cfg).ci)
    subs = {
        trust(role(t, n))["Condition"]["StringEquals"][f"{OIDC}:sub"]
        for n in ("detector-gh-deploy-dev", "detector-gh-pr-dev")
    }
    assert subs == {f"repo:{repo}:environment:dev", f"repo:{repo}:pull_request"}
