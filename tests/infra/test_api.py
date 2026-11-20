import json
from dataclasses import replace
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from config import load_config
from stacks import build_stacks

EXAMPLE = Path(__file__).parents[2] / "infra" / "environments.conf.example.json"
CIDRS = ("198.51.100.7/32", "203.0.113.0/24")


@pytest.fixture(scope="module")
def api() -> Template:
    cfg = replace(load_config("dev", EXAMPLE), api_allowed_cidrs=CIDRS)
    return Template.from_stack(build_stacks(cdk.App(), cfg).api)


def test_rest_api_only_allows_the_configured_cidrs(api):
    (rest,) = api.find_resources("AWS::ApiGateway::RestApi").values()
    props = rest["Properties"]
    assert props["Name"] == "detector-api-dev"
    assert props["EndpointConfiguration"] == {"Types": ["REGIONAL"]}
    (statement,) = props["Policy"]["Statement"]
    assert statement["Effect"] == "Allow"
    assert statement["Action"] == "execute-api:Invoke"
    assert statement["Condition"] == {"IpAddress": {"aws:SourceIp": list(CIDRS)}}


def test_proxy_to_the_api_function(api):
    (fn,) = [
        r["Properties"]
        for r in api.find_resources("AWS::Lambda::Function").values()
        if r["Properties"].get("FunctionName") == "detector-api-dev"
    ]
    assert fn["ImageConfig"]["Command"] == ["detector.api.handler"]
    assert fn["Architectures"] == ["arm64"]
    assert fn["Timeout"] == 29
    assert set(fn["Environment"]["Variables"]) == {"BUCKET", "TABLE", "FINALIZE_FUNCTION"}
    methods = api.find_resources("AWS::ApiGateway::Method")
    assert {m["Properties"]["Integration"]["Type"] for m in methods.values()} == {"AWS_PROXY"}
    api.has_resource_properties("AWS::ApiGateway::Stage", {"StageName": "prod"})
    assert "ApiUrl" in api.find_outputs("*")


def test_api_role_permissions(api):
    (policy,) = [
        p for name, p in api.find_resources("AWS::IAM::Policy").items() if name.startswith("ApiFn")
    ]
    text = json.dumps(policy["Properties"]["PolicyDocument"])
    for prefix in ("input", "upload", "results", "rendered"):
        assert f"/{prefix}/*" in text
    assert "lambda:InvokeFunction" in text
    assert '"s3:*"' not in text and '"dynamodb:*"' not in text
