import json
from dataclasses import replace
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Match, Template

from config import load_config
from stacks import EnvStacks, build_stacks
from stacks.detector_batch import MAX_RECEIVE_COUNT

EXAMPLE = Path(__file__).parents[2] / "infra" / "environments.conf.example.json"

# CloudFormation property that holds the explicit name, per resource type.
NAME_PROPS = {
    "AWS::S3::Bucket": "BucketName",
    "AWS::DynamoDB::GlobalTable": "TableName",
    "AWS::SQS::Queue": "QueueName",
    "AWS::ECS::Cluster": "ClusterName",
    "AWS::ECS::Service": "ServiceName",
    "AWS::ECS::CapacityProvider": "Name",
    "AWS::ECS::TaskDefinition": "Family",
    "AWS::EC2::SecurityGroup": "GroupName",
    "AWS::CloudWatch::Alarm": "AlarmName",
    "AWS::Lambda::Function": "FunctionName",
    "AWS::ApiGateway::RestApi": "Name",
}


def build(**overrides) -> EnvStacks:
    cfg = replace(load_config("dev", EXAMPLE), **overrides)
    return build_stacks(cdk.App(), cfg)


def explicit_names(template: Template) -> list[str]:
    names = []
    for rtype, prop in NAME_PROPS.items():
        for resource in template.find_resources(rtype).values():
            name = resource["Properties"].get(prop)
            if name is not None:
                names.append(name)
    return sorted(names)


def test_stack_names_end_with_env_suffix():
    stacks = build()
    assert [s.stack_name for s in (stacks.data, stacks.batch, stacks.api, stacks.ci)] == [
        "DetectorData-dev",
        "DetectorBatch-dev",
        "DetectorApi-dev",
        "DetectorCi-dev",
    ]


def test_each_stack_owns_its_resources():
    stacks = build()
    assert explicit_names(Template.from_stack(stacks.data)) == ["detector-results-dev"]
    assert explicit_names(Template.from_stack(stacks.batch)) == [
        "detector-batch-dev-123456789012-us-west-2",
        "detector-cluster-dev",
        "detector-cpu-cp-dev",
        "detector-finalize-dev",
        "detector-gpu-backlog-dev",
        "detector-gpu-cp-dev",
        "detector-gpu-idle-dev",
        "detector-images-dev",
        "detector-images-dlq-dev",
        "detector-unzip-dev",
        "detector-worker-cpu-dev",
        "detector-worker-cpu-dev",  # task definition family and service share the name
        "detector-worker-gpu-dev",
        "detector-worker-gpu-dev",
        "detector-worker-sg-dev",
    ]
    assert explicit_names(Template.from_stack(stacks.api)) == [
        "detector-api-dev",
        "detector-api-dev",
    ]


def test_batch_stack_deploys_after_data():
    stacks = build()
    assert stacks.batch.dependencies == [stacks.data]
    assert stacks.api.dependencies == [stacks.batch]  # and so after data, too


def test_queue_redrives_to_dlq():
    Template.from_stack(build().batch).has_resource_properties(
        "AWS::SQS::Queue",
        {
            "QueueName": "detector-images-dev",
            "RedrivePolicy": {
                "maxReceiveCount": MAX_RECEIVE_COUNT,
                "deadLetterTargetArn": Match.any_value(),
            },
        },
    )


@pytest.mark.parametrize(
    ("retain", "policy", "auto_delete"), [(False, "Delete", 1), (True, "Retain", 0)]
)
def test_retain_data_controls_deletion(retain, policy, auto_delete):
    stacks = build(retain_data=retain)
    batch = Template.from_stack(stacks.batch)
    batch.has_resource("AWS::S3::Bucket", {"DeletionPolicy": policy})
    batch.resource_count_is("Custom::S3AutoDeleteObjects", auto_delete)
    Template.from_stack(stacks.data).has_resource(
        "AWS::DynamoDB::GlobalTable", {"DeletionPolicy": policy}
    )


def test_synth_needs_no_context_lookups():
    """CI synthesizes without AWS credentials: no stack may need a lookup (AZs, AMIs ...)."""
    app = cdk.App()
    build_stacks(app, load_config("dev", EXAMPLE))
    manifest = json.loads((Path(app.synth().directory) / "manifest.json").read_text())
    assert not manifest.get("missing"), manifest.get("missing")
