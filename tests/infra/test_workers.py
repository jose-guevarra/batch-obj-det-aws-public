import json
from pathlib import Path

import aws_cdk as cdk
import pytest
from aws_cdk.assertions import Template

from config import load_config
from stacks import build_stacks
from stacks.worker_service import BACKLOG_FOR_MAX, SCALE_IN_MINUTES, TABLE_ACTIONS

EXAMPLE = Path(__file__).parents[2] / "infra" / "environments.conf.example.json"


@pytest.fixture(scope="module")
def batch() -> Template:
    return Template.from_stack(build_stacks(cdk.App(), load_config("dev", EXAMPLE)).batch)


def test_only_input_prefix_triggers_the_queue_and_only_zips_the_unzip_function(batch):
    (notifications,) = batch.find_resources("Custom::S3BucketNotifications").values()
    config = notifications["Properties"]["NotificationConfiguration"]
    assert set(config) == {"QueueConfigurations", "LambdaFunctionConfigurations"}
    (queue,) = config["QueueConfigurations"]
    assert queue["Events"] == ["s3:ObjectCreated:*"]
    assert queue["Filter"] == {"Key": {"FilterRules": [{"Name": "prefix", "Value": "input/"}]}}
    (fn,) = config["LambdaFunctionConfigurations"]
    assert fn["Events"] == ["s3:ObjectCreated:*"]
    rules = {r["Name"]: r["Value"] for r in fn["Filter"]["Key"]["FilterRules"]}
    assert rules == {"prefix": "upload/", "suffix": ".zip"}
    assert fn["LambdaFunctionArn"]["Fn::GetAtt"][0].startswith("UnzipFn")


def test_network_has_no_nat_and_free_gateway_endpoints(batch):
    batch.resource_count_is("AWS::EC2::NatGateway", 0)
    endpoints = batch.find_resources("AWS::EC2::VPCEndpoint")
    assert {e["Properties"]["VpcEndpointType"] for e in endpoints.values()} == {"Gateway"}
    services = json.dumps([e["Properties"]["ServiceName"] for e in endpoints.values()])
    assert ".s3" in services and ".dynamodb" in services
    batch.resource_count_is("AWS::EC2::Subnet", 2)


def test_worker_security_group_has_no_inbound(batch):
    (sg,) = batch.find_resources("AWS::EC2::SecurityGroup").values()
    assert "SecurityGroupIngress" not in sg["Properties"]
    batch.resource_count_is("AWS::EC2::SecurityGroupIngress", 0)


def _by_prefix(batch: Template, rtype: str, prefix: str) -> dict:
    (resource,) = [r for k, r in batch.find_resources(rtype).items() if k.startswith(prefix)]
    return resource["Properties"]


def _container(batch: Template, kind: str) -> tuple[dict, dict]:
    taskdef = _by_prefix(batch, "AWS::ECS::TaskDefinition", f"{kind}WorkersTaskDef")
    (container,) = taskdef["ContainerDefinitions"]
    return taskdef, container


@pytest.mark.parametrize(
    ("kind", "instance_type", "volume_gib", "ami"),
    [
        ("Cpu", "c6i.2xlarge", 30, "amazonlinux2023recommended"),
        ("Gpu", "g4dn.xlarge", 60, "amazonlinux2023gpurecommended"),
    ],
)
def test_worker_instances(batch, kind, instance_type, volume_gib, ami):
    asg = _by_prefix(batch, "AWS::AutoScaling::AutoScalingGroup", f"{kind}WorkersAsg")
    assert (asg["MinSize"], asg["MaxSize"], asg["NewInstancesProtectedFromScaleIn"]) == (
        "0",
        "2",
        True,
    )
    data = _by_prefix(batch, "AWS::EC2::LaunchTemplate", f"{kind}WorkersLaunchTemplate")[
        "LaunchTemplateData"
    ]
    assert data["InstanceType"] == instance_type
    assert data["MetadataOptions"] == {"HttpTokens": "required"}
    assert data["BlockDeviceMappings"][0]["Ebs"]["VolumeSize"] == volume_gib
    # The AMI comes from the ECS-optimized SSM parameter (GPU variant for GPU workers).
    assert ami in data["ImageId"]["Ref"]

    cp = _by_prefix(batch, "AWS::ECS::CapacityProvider", f"{kind}WorkersCapacityProvider")
    provider = cp["AutoScalingGroupProvider"]
    assert provider["ManagedScaling"]["Status"] == "ENABLED"
    assert provider["ManagedScaling"]["TargetCapacity"] == 100
    assert provider["ManagedTerminationProtection"] == "ENABLED"


def test_cpu_task_and_service(batch):
    taskdef, container = _container(batch, "Cpu")
    assert taskdef["NetworkMode"] == "bridge"
    assert taskdef["Family"] == "detector-worker-cpu-dev"
    assert container["StopTimeout"] == 120
    assert "ResourceRequirements" not in container
    env = {e["Name"]: e["Value"] for e in container["Environment"]}
    assert set(env) >= {"QUEUE_URL", "DLQ_URL", "BUCKET", "TABLE", "AWS_DEFAULT_REGION"}
    assert (env["DEVICE"], env["BATCH_SIZE"], env["MODEL"]) == ("cpu", "1", "yolov8m")
    assert (env["RUNTIME"], env["PIPELINE"], env["HALF"]) == ("pytorch", "0", "0")
    assert env["AWS_DEFAULT_REGION"] == "us-west-2"
    assert env["FINALIZE_FUNCTION"]["Ref"].startswith("FinalizeFn")
    log_group_ref = container["LogConfiguration"]["Options"]["awslogs-group"]["Ref"]
    assert log_group_ref.startswith("CpuWorkersLogs")
    batch.has_resource_properties(
        "AWS::Logs::LogGroup", {"LogGroupName": "/detector/dev/worker-cpu", "RetentionInDays": 14}
    )
    service = _by_prefix(batch, "AWS::ECS::Service", "CpuWorkersService")
    assert service["DesiredCount"] == 0
    assert service["DeploymentConfiguration"]["DeploymentCircuitBreaker"] == {
        "Enable": True,
        "Rollback": True,
    }


def test_gpu_task_and_service(batch):
    taskdef, container = _container(batch, "Gpu")
    assert taskdef["Family"] == "detector-worker-gpu-dev"
    assert container["ResourceRequirements"] == [{"Type": "GPU", "Value": "1"}]
    assert container["Cpu"] <= 4096 and container["MemoryReservation"] <= 15 * 1024
    env = {e["Name"]: e["Value"] for e in container["Environment"]}
    assert (env["DEVICE"], env["BATCH_SIZE"], env["RUNTIME"], env["PIPELINE"], env["HALF"]) == (
        "cuda",
        "32",
        "pytorch",
        "1",
        "1",
    )
    batch.has_resource_properties(
        "AWS::Logs::LogGroup", {"LogGroupName": "/detector/dev/worker-gpu", "RetentionInDays": 14}
    )
    service = _by_prefix(batch, "AWS::ECS::Service", "GpuWorkersService")
    assert (service["ServiceName"], service["DesiredCount"]) == ("detector-worker-gpu-dev", 0)
    # The two kinds run separate images (CPU-only torch vs. CUDA torch).
    assert json.dumps(container["Image"]) != json.dumps(_container(batch, "Cpu")[1]["Image"])


def test_outputs_for_benchmark_tasks(batch):
    outputs = set(batch.find_outputs("*"))
    for kind in ("Cpu", "Gpu"):
        for name in ("ServiceName", "AsgName", "LogGroup", "TaskFamily", "CapacityProvider"):
            assert f"{kind}{name}" in outputs


def test_task_role_is_least_privilege(batch):
    policies = batch.find_resources("AWS::IAM::Policy")
    task_policies = [p for name, p in policies.items() if "TaskDefTaskRole" in name]
    assert len(task_policies) == 2  # one per worker kind, same permissions
    statements = task_policies[0]["Properties"]["PolicyDocument"]["Statement"]
    actions = set()
    for s in statements:
        actions |= set(s["Action"] if isinstance(s["Action"], list) else [s["Action"]])
    assert set(TABLE_ACTIONS) <= actions
    assert {"sqs:ReceiveMessage", "sqs:DeleteMessage", "sqs:SendMessage"} <= actions
    assert "s3:PutObject" in actions
    assert not any(a in {"*", "s3:*", "dynamodb:*", "sqs:*"} for a in actions)
    assert not any(a.startswith("dynamodb:Delete") or a == "dynamodb:Scan" for a in actions)
    # Reads are limited to input/, writes to results/.
    text = json.dumps(statements)
    assert "/input/*" in text and "/results/*" in text


def test_only_gpu_workers_scale_on_the_backlog(batch):
    (target,) = batch.find_resources("AWS::ApplicationAutoScaling::ScalableTarget").values()
    props = target["Properties"]
    assert (props["MinCapacity"], props["MaxCapacity"]) == (0, 2)
    assert props["ScalableDimension"] == "ecs:service:DesiredCount"
    assert "GpuWorkersService" in json.dumps(props["ResourceId"])

    alarms = {
        a["Properties"]["AlarmName"]: a["Properties"]
        for a in batch.find_resources("AWS::CloudWatch::Alarm").values()
    }
    assert set(alarms) == {"detector-gpu-backlog-dev", "detector-gpu-idle-dev"}
    busy, idle = alarms["detector-gpu-backlog-dev"], alarms["detector-gpu-idle-dev"]
    assert (busy["Threshold"], busy["EvaluationPeriods"]) == (1, 1)
    assert (idle["Threshold"], idle["EvaluationPeriods"]) == (0, SCALE_IN_MINUTES)
    assert idle["TreatMissingData"] == "breaching"
    assert any(m.get("Expression") == "visible + inflight" for m in busy["Metrics"])

    policies = batch.find_resources("AWS::ApplicationAutoScaling::ScalingPolicy")
    steps = sorted(
        (
            p["Properties"]["StepScalingPolicyConfiguration"]["StepAdjustments"]
            for p in policies.values()
        ),
        key=len,
    )
    assert steps[0] == [{"MetricIntervalUpperBound": 0, "ScalingAdjustment": 0}]
    assert steps[1] == [
        {
            "MetricIntervalLowerBound": 0,
            "MetricIntervalUpperBound": BACKLOG_FOR_MAX - 1,
            "ScalingAdjustment": 1,
        },
        {"MetricIntervalLowerBound": BACKLOG_FOR_MAX - 1, "ScalingAdjustment": 2},
    ]


def test_gpu_task_keeps_tensorrt_engines_in_a_shared_volume(batch):
    gpu_taskdef, gpu = _container(batch, "Gpu")
    (volume,) = gpu_taskdef["Volumes"]
    assert volume["Name"] == "detector-gpu-cache-dev"
    assert volume["DockerVolumeConfiguration"] == {
        "Autoprovision": True,
        "Driver": "local",
        "Scope": "shared",
    }
    assert gpu["MountPoints"] == [
        {
            "ContainerPath": "/tmp/trt-cache",
            "SourceVolume": "detector-gpu-cache-dev",
            "ReadOnly": False,
        }
    ]
    cpu_taskdef, cpu = _container(batch, "Cpu")
    assert "Volumes" not in cpu_taskdef or not cpu_taskdef["Volumes"]
    assert not cpu.get("MountPoints")


def test_workers_may_invoke_finalize_and_nothing_else(batch):
    policies = batch.find_resources("AWS::IAM::Policy")
    for name, policy in policies.items():
        if "TaskDefTaskRole" not in name:
            continue
        statements = policy["Properties"]["PolicyDocument"]["Statement"]
        invoke = [s for s in statements if "lambda:InvokeFunction" in json.dumps(s["Action"])]
        assert len(invoke) == 1
        assert all("FinalizeFn" in json.dumps(r) for r in invoke[0]["Resource"])


def test_lambda_functions(batch):
    fns = {
        p["Properties"]["FunctionName"]: p["Properties"]
        for p in batch.find_resources("AWS::Lambda::Function").values()
        if "FunctionName" in p["Properties"]
    }
    finalize, unzip = fns["detector-finalize-dev"], fns["detector-unzip-dev"]
    assert finalize["Architectures"] == unzip["Architectures"] == ["arm64"]
    assert finalize["ImageConfig"]["Command"] == ["detector.finalize.handler"]
    assert unzip["ImageConfig"]["Command"] == ["detector.unzip.handler"]
    assert (finalize["Timeout"], unzip["Timeout"]) == (300, 900)
    assert unzip["EphemeralStorage"] == {"Size": 4096}
    for name in ("finalize", "unzip"):
        batch.has_resource_properties(
            "AWS::Logs::LogGroup", {"LogGroupName": f"/detector/dev/{name}", "RetentionInDays": 14}
        )
