"""DetectorData-<env>: the DynamoDB table for per-image results and batch status."""

from __future__ import annotations

from aws_cdk import CfnOutput, RemovalPolicy, Stack
from aws_cdk import aws_dynamodb as dynamodb
from constructs import Construct

from config import EnvConfig


class DataStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, cfg: EnvConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        # Single table: per-image results and the per-batch status record share
        # a batch_id partition. Item layout is defined in milestone 3.
        self.table = dynamodb.TableV2(
            self,
            "Results",
            table_name=cfg.name("results"),
            partition_key=dynamodb.Attribute(name="pk", type=dynamodb.AttributeType.STRING),
            sort_key=dynamodb.Attribute(name="sk", type=dynamodb.AttributeType.STRING),
            billing=dynamodb.Billing.on_demand(),
            removal_policy=RemovalPolicy.RETAIN if cfg.retain_data else RemovalPolicy.DESTROY,
        )

        CfnOutput(self, "TableName", value=self.table.table_name)
