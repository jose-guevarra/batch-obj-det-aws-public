"""DetectorApi-<env>: the status and rendering API (FastAPI on Lambda, behind API Gateway).

A REST API (not an HTTP API) because only REST APIs take a resource policy: it allows
execute-api:Invoke only from cfg.api_allowed_cidrs, so requests from anywhere else get
403 before the function runs. There's no other authentication; see docs/design.md.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, Stack
from aws_cdk import aws_apigateway as apigw
from aws_cdk import aws_dynamodb as dynamodb
from aws_cdk import aws_iam as iam
from aws_cdk import aws_lambda as lambda_
from aws_cdk import aws_s3 as s3
from constructs import Construct

from config import EnvConfig
from stacks.lambda_image import image_function

STAGE = "prod"


class ApiStack(Stack):
    def __init__(
        self,
        scope: Construct,
        construct_id: str,
        *,
        cfg: EnvConfig,
        bucket: s3.IBucket,
        table: dynamodb.ITableV2,
        finalize_fn: lambda_.IFunction,
        **kwargs,
    ) -> None:
        super().__init__(scope, construct_id, **kwargs)

        self.function = image_function(
            self,
            "ApiFn",
            cfg=cfg,
            name="api",
            handler="detector.api.handler",
            memory_mib=1024,
            timeout=Duration.seconds(29),  # API Gateway's integration limit
            environment={
                "BUCKET": bucket.bucket_name,
                "TABLE": table.table_name,
                "FINALIZE_FUNCTION": finalize_fn.function_name,
            },
        )
        fn = self.function
        table.grant_read_write_data(fn)  # batch records (create) and image results (read)
        # Presigned PUTs are signed with this role, so it needs the put permissions itself.
        bucket.grant_put(fn, "input/*")
        bucket.grant_put(fn, "upload/*")
        bucket.grant_read(fn, "input/*")  # originals, for rendering
        bucket.grant_read(fn, "results/*")  # summary.json, results.jsonl
        bucket.grant_read_write(fn, "rendered/*")
        finalize_fn.grant_invoke(fn)

        policy = iam.PolicyDocument(
            statements=[
                iam.PolicyStatement(
                    effect=iam.Effect.ALLOW,
                    principals=[iam.AnyPrincipal()],
                    actions=["execute-api:Invoke"],
                    resources=["execute-api:/*"],
                    conditions={"IpAddress": {"aws:SourceIp": list(cfg.api_allowed_cidrs)}},
                )
            ]
        )
        self.api = apigw.LambdaRestApi(
            self,
            "Api",
            rest_api_name=cfg.name("api"),
            handler=fn,
            proxy=True,
            policy=policy,
            endpoint_types=[apigw.EndpointType.REGIONAL],
            cloud_watch_role=False,
            deploy_options=apigw.StageOptions(
                stage_name=STAGE, throttling_rate_limit=20, throttling_burst_limit=40
            ),
        )

        CfnOutput(self, "ApiUrl", value=self.api.url)
        CfnOutput(self, "ApiFunction", value=fn.function_name)
