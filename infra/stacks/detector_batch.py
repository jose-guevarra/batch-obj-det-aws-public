"""DetectorBatch-<env>: image bucket, queues, network, ECS cluster, workers, and the
finalize and unzip functions.

The bucket and queue share this stack so S3 can notify the queue directly; across
stacks that notification would create a dependency cycle. The unzip function is here
for the same reason (S3 notifies it), and finalize because the workers invoke it. The
results table lives in DetectorData-<env> and is passed in.
"""

from __future__ import annotations

from pathlib import Path

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_ec2 as ec2
from aws_cdk import aws_ecr_assets as ecr_assets
from aws_cdk import aws_ecs as ecs
from aws_cdk import aws_s3 as s3
from aws_cdk import aws_s3_notifications as s3n
from aws_cdk import aws_sqs as sqs
from constructs import Construct

from config import EnvConfig
from stacks.lambda_image import build_cache, image_function
from stacks.worker_service import WorkerService

TRT_CACHE_DIR = "/tmp/trt-cache"  # detector.runtimes.TRT_CACHE_DIR inside the GPU image

REPO_ROOT = Path(__file__).resolve().parents[2]

# A corrupt image is sent to the DLQ by the worker right away; this limit only
# applies to errors that may be temporary (throttling, timeouts).
MAX_RECEIVE_COUNT = 3

# One task fills one instance, so there is one worker per instance.
# CPU: c6i.2xlarge (8 vCPU, 16 GiB). Benchmark-only (RunTask); it doesn't scale on the queue.
CPU_TASK_CPU_UNITS = 7168
CPU_TASK_MEMORY_MIB = 12288
CPU_BATCH_SIZE = 1  # the design doc's CPU row
# GPU: g4dn.xlarge (4 vCPU, 16 GiB, one T4). Owns the queue: scales on its backlog.
GPU_TASK_CPU_UNITS = 3584
GPU_TASK_MEMORY_MIB = 12288
# Pipelined PyTorch FP16 at batch 32: the lowest cost per 1,000 images measured, with mAP50-95
# 0.5021 vs. 0.5026 in FP32 (docs/benchmark.md).
GPU_BATCH_SIZE = 32
GPU_HALF = True
GPU_ROOT_VOLUME_GIB = 60  # GPU AMI plus a ~5 GB CUDA image


class DetectorBatchStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        cfg: EnvConfig,
        table: dynamodb.ITableV2,
        **kwargs,
    ) -> None:
        self._azs = [f"{cfg.region}a", f"{cfg.region}b"]
        super().__init__(scope, construct_id, **kwargs)

        self.table = table

        # Storage and queues ------------------------------------------------------
        self.bucket = s3.Bucket(
            self,
            "Bucket",
            bucket_name=cfg.bucket_name,
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
            enforce_ssl=True,
            removal_policy=RemovalPolicy.RETAIN if cfg.retain_data else RemovalPolicy.DESTROY,
            auto_delete_objects=not cfg.retain_data,
        )

        self.dlq = sqs.Queue(
            self,
            "ImagesDlq",
            queue_name=cfg.name("images-dlq"),
            retention_period=Duration.days(14),
            enforce_ssl=True,
        )

        self.queue = sqs.Queue(
            self,
            "Images",
            queue_name=cfg.name("images"),
            # Must exceed collect + infer + write for the largest batch (64).
            # Revisit after the first GPU sweep.
            visibility_timeout=Duration.minutes(5),
            receive_message_wait_time=Duration.seconds(20),
            retention_period=Duration.days(4),
            enforce_ssl=True,
            dead_letter_queue=sqs.DeadLetterQueue(
                max_receive_count=MAX_RECEIVE_COUNT, queue=self.dlq
            ),
        )

        # Only images under input/ queue work; results/, rendered/ and upload/ never do.
        self.bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.SqsDestination(self.queue),
            s3.NotificationKeyFilter(prefix="input/"),
        )

        # Batch completion and zip uploads (Lambda) ------------------------------
        data_env = {"BUCKET": self.bucket.bucket_name, "TABLE": table.table_name}
        # Combines part files into results.jsonl and writes summary.json, once per batch
        # (a conditional claim on the batch record). Invoked asynchronously by workers.
        self.finalize_fn = image_function(
            self,
            "FinalizeFn",
            cfg=cfg,
            name="finalize",
            handler="detector.finalize.handler",
            memory_mib=1024,
            timeout=Duration.minutes(5),
            environment=data_env,
        )
        table.grant_read_write_data(self.finalize_fn)
        self.bucket.grant_read_write(self.finalize_fn, "results/*")

        # Extracts upload/{batch_id}/*.zip into input/{batch_id}/ (which queues the images).
        self.unzip_fn = image_function(
            self,
            "UnzipFn",
            cfg=cfg,
            name="unzip",
            handler="detector.unzip.handler",
            memory_mib=2048,
            timeout=Duration.minutes(15),
            ephemeral_gib=4,
            environment=data_env,
        )
        table.grant_read_write_data(self.unzip_fn)
        self.bucket.grant_read(self.unzip_fn, "upload/*")
        self.bucket.grant_put(self.unzip_fn, "input/*")
        self.bucket.add_event_notification(
            s3.EventType.OBJECT_CREATED,
            s3n.LambdaDestination(self.unzip_fn),
            s3.NotificationKeyFilter(prefix="upload/", suffix=".zip"),
        )

        # Network: public subnets, no NAT gateway (it would bill ~$32/month idle).
        # Instances accept no inbound traffic; S3 and DynamoDB go through free
        # gateway endpoints, SQS / ECR / CloudWatch Logs through the internet gateway.
        self.vpc = ec2.Vpc(
            self,
            "Vpc",
            vpc_name=cfg.name("vpc"),
            availability_zones=self.availability_zones,
            nat_gateways=0,
            subnet_configuration=[
                ec2.SubnetConfiguration(
                    name="public", subnet_type=ec2.SubnetType.PUBLIC, cidr_mask=24
                )
            ],
            gateway_endpoints={
                "S3": ec2.GatewayVpcEndpointOptions(service=ec2.GatewayVpcEndpointAwsService.S3),
                "DynamoDB": ec2.GatewayVpcEndpointOptions(
                    service=ec2.GatewayVpcEndpointAwsService.DYNAMODB
                ),
            },
        )
        self.worker_sg = ec2.SecurityGroup(
            self,
            "WorkerSg",
            vpc=self.vpc,
            security_group_name=cfg.name("worker-sg"),
            description="Detector workers: no inbound, all outbound",
            allow_all_outbound=True,
        )

        self.cluster = ecs.Cluster(
            self,
            "Cluster",
            cluster_name=cfg.name("cluster"),
            vpc=self.vpc,
            container_insights_v2=ecs.ContainerInsights.DISABLED,
        )

        shared = dict(
            cfg=cfg,
            cluster=self.cluster,
            vpc=self.vpc,
            security_group=self.worker_sg,
            queue=self.queue,
            dlq=self.dlq,
            bucket=self.bucket,
            table=table,
        )

        # CPU workers: benchmark baseline only -----------------------------------
        self.cpu_workers = WorkerService(
            self,
            "CpuWorkers",
            kind="cpu",
            instance_type=cfg.cpu_instance_type,
            max_instances=cfg.cpu_max_instances,
            image=self._image("CpuImage", "docker/Dockerfile.cpu", cfg, "cpu"),
            cpu_units=CPU_TASK_CPU_UNITS,
            memory_reservation_mib=CPU_TASK_MEMORY_MIB,
            environment=self._worker_env(
                cfg, device="cpu", batch_size=CPU_BATCH_SIZE, pipeline=False, half=False
            ),
            autoscale=False,
            **shared,
        )

        # GPU workers: process everything that lands in input/ ---------------------
        self.gpu_workers = WorkerService(
            self,
            "GpuWorkers",
            kind="gpu",
            instance_type=cfg.gpu_instance_type,
            max_instances=cfg.gpu_max_instances,
            image=self._image("GpuImage", "docker/Dockerfile.gpu", cfg, "gpu"),
            cpu_units=GPU_TASK_CPU_UNITS,
            memory_reservation_mib=GPU_TASK_MEMORY_MIB,
            # Pipelined: the next batch is prepared and the last one written while the
            # GPU works. Benchmarks override RUNTIME / HALF / PIPELINE / BATCH_SIZE per run.
            environment=self._worker_env(
                cfg, device="cuda", batch_size=GPU_BATCH_SIZE, pipeline=True, half=GPU_HALF
            ),
            gpu=True,
            root_volume_gib=GPU_ROOT_VOLUME_GIB,
            autoscale=True,
            cache_path=TRT_CACHE_DIR,
            **shared,
        )

        for workers in (self.cpu_workers, self.gpu_workers):
            self.finalize_fn.grant_invoke(workers.task_definition.task_role)

        CfnOutput(self, "BucketName", value=self.bucket.bucket_name)
        CfnOutput(self, "FinalizeFunction", value=self.finalize_fn.function_name)
        CfnOutput(self, "QueueUrl", value=self.queue.queue_url)
        CfnOutput(self, "DlqUrl", value=self.dlq.queue_url)
        CfnOutput(self, "ClusterName", value=self.cluster.cluster_name)
        for kind, workers in (("Cpu", self.cpu_workers), ("Gpu", self.gpu_workers)):
            CfnOutput(self, f"{kind}ServiceName", value=workers.service.service_name)
            CfnOutput(self, f"{kind}AsgName", value=workers.asg.auto_scaling_group_name)
            CfnOutput(self, f"{kind}LogGroup", value=workers.log_group.log_group_name)
            CfnOutput(self, f"{kind}TaskFamily", value=workers.task_definition.family)
            CfnOutput(
                self,
                f"{kind}CapacityProvider",
                value=workers.capacity_provider.capacity_provider_name,
            )

    @property
    def availability_zones(self) -> list[str]:
        """Fixed AZs instead of a context lookup, so synth needs no AWS credentials (CI)."""
        return list(self._azs)

    def _image(
        self, construct_id: str, dockerfile: str, cfg: EnvConfig, cache_name: str
    ) -> ecs.ContainerImage:
        asset = ecr_assets.DockerImageAsset(
            self,
            construct_id,
            directory=str(REPO_ROOT),
            file=dockerfile,
            platform=ecr_assets.Platform.LINUX_AMD64,
            **build_cache(self, cfg, cache_name),
        )
        return ecs.ContainerImage.from_docker_image_asset(asset)

    def _worker_env(
        self, cfg: EnvConfig, *, device: str, batch_size: int, pipeline: bool, half: bool
    ) -> dict[str, str]:
        return {
            "QUEUE_URL": self.queue.queue_url,
            "DLQ_URL": self.dlq.queue_url,
            "BUCKET": self.bucket.bucket_name,
            "TABLE": self.table.table_name,
            "AWS_DEFAULT_REGION": cfg.region,
            "MODEL": cfg.model,
            "DEVICE": device,
            "BATCH_SIZE": str(batch_size),
            "MAX_WAIT_S": "1",
            "RUNTIME": "pytorch",
            "HALF": "1" if half else "0",
            "PIPELINE": "1" if pipeline else "0",
            "FINALIZE_FUNCTION": self.finalize_fn.function_name,
        }
