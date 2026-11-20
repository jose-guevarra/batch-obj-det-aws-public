"""One kind of worker (cpu or gpu): instances, capacity provider, ECS service, optional
queue-driven scaling and least-privilege IAM.

Scaling chain: queue backlog -> CloudWatch alarm -> step scaling sets the service's task count
-> the capacity provider adds or removes EC2 instances to fit the tasks.

A service built with autoscale=False stays at 0 tasks; its task definition is started
directly (ECS RunTask) by scripts/benchmark.py.
"""

from __future__ import annotations

from aws_cdk import Duration, RemovalPolicy
from aws_cdk import aws_applicationautoscaling as appscaling
from aws_cdk import aws_autoscaling as autoscaling
from aws_cdk import aws_cloudwatch as cloudwatch
from aws_cdk import aws_cloudwatch_actions as cw_actions
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_iam as iam
from aws_cdk import aws_logs as logs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_sqs as sqs
from constructs import Construct

from config import EnvConfig

STOP_TIMEOUT = Duration.seconds(120)  # SIGTERM -> finish the current batch -> exit
SCALE_IN_MINUTES = 5  # backlog must be 0 this long before workers are removed
BACKLOG_FOR_MAX = 2000  # backlog at which every allowed worker runs

# What TransactWriteItems, BatchGetItem and GetItem need on the results table.
TABLE_ACTIONS = [
    "dynamodb:GetItem",
    "dynamodb:BatchGetItem",
    "dynamodb:PutItem",
    "dynamodb:UpdateItem",
    "dynamodb:ConditionCheckItem",
]


class WorkerService(Construct):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        cfg: EnvConfig,
        kind: str,
        cluster: ecs.Cluster,
        vpc: ec2.IVpc,
        security_group: ec2.ISecurityGroup,
        instance_type: str,
        max_instances: int,
        image: ecs.ContainerImage,
        cpu_units: int,
        memory_reservation_mib: int,
        environment: dict[str, str],
        queue: sqs.IQueue,
        dlq: sqs.IQueue,
        bucket: s3.IBucket,
        table: dynamodb.ITableV2,
        gpu: bool = False,
        root_volume_gib: int = 30,
        autoscale: bool = True,
        cache_path: str | None = None,
    ) -> None:
        super().__init__(scope, construct_id)

        # Instances --------------------------------------------------------------
        instance_role = iam.Role(
            self,
            "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            managed_policies=[
                iam.ManagedPolicy.from_aws_managed_policy_name(
                    "service-role/AmazonEC2ContainerServiceforEC2Role"
                )
            ],
        )
        launch_template = ec2.LaunchTemplate(
            self,
            "LaunchTemplate",
            instance_type=ec2.InstanceType(instance_type),
            machine_image=ecs.EcsOptimizedImage.amazon_linux2023(
                ecs.AmiHardwareType.GPU if gpu else ecs.AmiHardwareType.STANDARD
            ),
            user_data=ec2.UserData.for_linux(),  # the capacity provider adds the cluster join
            role=instance_role,
            security_group=security_group,
            associate_public_ip_address=True,  # public subnets, no NAT gateway
            require_imdsv2=True,  # hop limit 1: containers can't use the instance role
            block_devices=[
                ec2.BlockDevice(
                    device_name="/dev/xvda",
                    volume=ec2.BlockDeviceVolume.ebs(
                        root_volume_gib, volume_type=ec2.EbsDeviceVolumeType.GP3, encrypted=True
                    ),
                )
            ],
        )
        self.asg = autoscaling.AutoScalingGroup(
            self,
            "Asg",
            vpc=vpc,
            vpc_subnets=ec2.SubnetSelection(subnet_type=ec2.SubnetType.PUBLIC),
            launch_template=launch_template,
            min_capacity=0,
            max_capacity=max_instances,
            new_instances_protected_from_scale_in=True,  # managed termination protection
        )
        self.capacity_provider = ecs.AsgCapacityProvider(
            self,
            "CapacityProvider",
            capacity_provider_name=cfg.name(f"{kind}-cp"),
            auto_scaling_group=self.asg,
            enable_managed_scaling=True,
            target_capacity_percent=100,
            enable_managed_termination_protection=True,
        )
        cluster.add_asg_capacity_provider(self.capacity_provider)

        # Task --------------------------------------------------------------------
        self.log_group = logs.LogGroup(
            self,
            "Logs",
            log_group_name=f"/detector/{cfg.env_name}/worker-{kind}",
            retention=logs.RetentionDays.TWO_WEEKS,
            removal_policy=RemovalPolicy.DESTROY,
        )
        self.task_definition = ecs.Ec2TaskDefinition(
            self,
            "TaskDef",
            family=cfg.name(f"worker-{kind}"),
            network_mode=ecs.NetworkMode.BRIDGE,  # no inbound traffic at all
        )
        container = self.task_definition.add_container(
            "worker",
            image=image,
            cpu=cpu_units,
            memory_reservation_mib=memory_reservation_mib,
            gpu_count=1 if gpu else None,
            environment=environment,
            stop_timeout=STOP_TIMEOUT,
            logging=ecs.LogDrivers.aws_logs(stream_prefix="worker", log_group=self.log_group),
        )
        if cache_path:
            # A Docker volume shared by every task on the instance: TensorRT engines built by
            # one task (minutes on a T4) are reused by the next instead of rebuilt.
            volume = cfg.name(f"{kind}-cache")
            self.task_definition.add_volume(
                name=volume,
                docker_volume_configuration=ecs.DockerVolumeConfiguration(
                    scope=ecs.Scope.SHARED, autoprovision=True, driver="local"
                ),
            )
            container.add_mount_points(
                ecs.MountPoint(container_path=cache_path, source_volume=volume, read_only=False)
            )
        role = self.task_definition.task_role
        queue.grant_consume_messages(role)
        dlq.grant_send_messages(role)
        bucket.grant_read(role, "input/*")
        bucket.grant_put(role, "results/*")
        role.add_to_principal_policy(
            iam.PolicyStatement(actions=TABLE_ACTIONS, resources=[table.table_arn])
        )

        # Service -----------------------------------------------------------------
        self.service = ecs.Ec2Service(
            self,
            "Service",
            service_name=cfg.name(f"worker-{kind}"),
            cluster=cluster,
            task_definition=self.task_definition,
            desired_count=0,  # scaling sets it from the queue backlog
            min_healthy_percent=0,  # a deploy may stop the old task first (one per instance)
            max_healthy_percent=100,
            capacity_provider_strategies=[
                ecs.CapacityProviderStrategy(
                    capacity_provider=self.capacity_provider.capacity_provider_name, weight=1
                )
            ],
            circuit_breaker=ecs.DeploymentCircuitBreaker(rollback=True),
        )

        if autoscale:
            self._scale_on_backlog(cfg, kind, cluster, queue, max_instances)

    def _scale_on_backlog(
        self, cfg: EnvConfig, kind: str, cluster: ecs.Cluster, queue: sqs.IQueue, max_instances: int
    ) -> None:
        """Backlog (visible + in-flight messages) -> step scaling of the task count.

        Step scaling, not target tracking: target tracking can't start from 0 tasks.
        """
        minute = Duration.minutes(1)
        self.backlog = cloudwatch.MathExpression(
            label="Backlog (visible + in flight)",
            expression="visible + inflight",
            using_metrics={
                "visible": queue.metric_approximate_number_of_messages_visible(
                    period=minute, statistic="Maximum"
                ),
                "inflight": queue.metric_approximate_number_of_messages_not_visible(
                    period=minute, statistic="Maximum"
                ),
            },
            period=minute,
        )
        target = appscaling.ScalableTarget(
            self,
            "ScalableTarget",
            service_namespace=appscaling.ServiceNamespace.ECS,
            resource_id=f"service/{cluster.cluster_name}/{self.service.service_name}",
            scalable_dimension="ecs:service:DesiredCount",
            min_capacity=0,
            max_capacity=max_instances,
        )
        target.node.add_dependency(self.service)

        scale_out = appscaling.StepScalingAction(
            self,
            "ScaleOut",
            scaling_target=target,
            adjustment_type=appscaling.AdjustmentType.EXACT_CAPACITY,
            metric_aggregation_type=appscaling.MetricAggregationType.MAXIMUM,
            cooldown=minute,
        )
        # Bounds are relative to the alarm threshold (1 message).
        scale_out.add_adjustment(lower_bound=0, upper_bound=BACKLOG_FOR_MAX - 1, adjustment=1)
        scale_out.add_adjustment(lower_bound=BACKLOG_FOR_MAX - 1, adjustment=max_instances)
        cloudwatch.Alarm(
            self,
            "BacklogAlarm",
            alarm_name=cfg.name(f"{kind}-backlog"),
            alarm_description=f"Images waiting for {kind} workers",
            metric=self.backlog,
            threshold=1,
            comparison_operator=cloudwatch.ComparisonOperator.GREATER_THAN_OR_EQUAL_TO_THRESHOLD,
            evaluation_periods=1,
            treat_missing_data=cloudwatch.TreatMissingData.NOT_BREACHING,
        ).add_alarm_action(cw_actions.ApplicationScalingAction(scale_out))

        scale_in = appscaling.StepScalingAction(
            self,
            "ScaleIn",
            scaling_target=target,
            adjustment_type=appscaling.AdjustmentType.EXACT_CAPACITY,
            metric_aggregation_type=appscaling.MetricAggregationType.MAXIMUM,
            cooldown=minute,
        )
        scale_in.add_adjustment(upper_bound=0, adjustment=0)
        cloudwatch.Alarm(
            self,
            "IdleAlarm",
            alarm_name=cfg.name(f"{kind}-idle"),
            alarm_description=f"No work for {kind} workers for {SCALE_IN_MINUTES} minutes",
            metric=self.backlog,
            threshold=0,
            comparison_operator=cloudwatch.ComparisonOperator.LESS_THAN_OR_EQUAL_TO_THRESHOLD,
            evaluation_periods=SCALE_IN_MINUTES,
            # SQS stops publishing metrics for long-idle queues: missing means idle.
            treat_missing_data=cloudwatch.TreatMissingData.BREACHING,
        ).add_alarm_action(cw_actions.ApplicationScalingAction(scale_in))
