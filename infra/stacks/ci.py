"""DetectorCi-<env>: IAM roles that GitHub Actions assumes through OIDC (no stored keys).

    deploy role  only jobs in the GitHub environment "dev" of cfg.github_repo (the deploy
                 job; the environment allows the develop branch only); may assume
                 the CDK bootstrap roles (deploy, file and image publishing, lookup), so
                 CloudFormation does the work through them
    PR role      only pull-request runs of cfg.github_repo; may assume the lookup role
                 alone, which is enough for `cdk diff --method template` and nothing else
    build cache  an ECR repo for CI's Docker layer cache (one tag per image), with mutable
                 tags: the CDK assets repo's tags are immutable, so a cache tag there could
                 be written only once. The image-publishing role, which pushes during
                 cdk deploy, may read and write it through the repository policy.

The account's GitHub OIDC provider already exists (one per URL per account), so it's
imported. Deploy this stack locally once (make deploy-ci); CI deploys the others.
"""

from __future__ import annotations

from aws_cdk import CfnOutput, Duration, RemovalPolicy, Stack
from aws_cdk import aws_ecr as ecr
from aws_cdk import aws_iam as iam
from constructs import Construct

from config import EnvConfig

GITHUB_OIDC_HOST = "token.actions.githubusercontent.com"
CDK_QUALIFIER = "hnb659fds"  # the default bootstrap qualifier
DEPLOY_ENVIRONMENT = "dev"  # the GitHub environment the deploy job runs in
BOOTSTRAP_ROLES = ("deploy", "file-publishing", "image-publishing", "lookup")


def bootstrap_role_arn(cfg: EnvConfig, kind: str) -> str:
    return (
        f"arn:aws:iam::{cfg.account}:role/"
        f"cdk-{CDK_QUALIFIER}-{kind}-role-{cfg.account}-{cfg.region}"
    )


class CiStack(Stack):
    def __init__(self, scope: Construct, construct_id: str, *, cfg: EnvConfig, **kwargs) -> None:
        super().__init__(scope, construct_id, **kwargs)

        provider = iam.OpenIdConnectProvider.from_open_id_connect_provider_arn(
            self,
            "GithubOidc",
            f"arn:aws:iam::{cfg.account}:oidc-provider/{GITHUB_OIDC_HOST}",
        )

        def github_principal(sub: str) -> iam.WebIdentityPrincipal:
            return iam.WebIdentityPrincipal(
                provider.open_id_connect_provider_arn,
                conditions={
                    "StringEquals": {
                        f"{GITHUB_OIDC_HOST}:aud": "sts.amazonaws.com",
                        f"{GITHUB_OIDC_HOST}:sub": sub,
                    }
                },
            )

        repo = cfg.github_repo
        self.deploy_role = iam.Role(
            self,
            "DeployRole",
            role_name=cfg.name("gh-deploy"),
            description=f"GitHub Actions deploys from {repo}, environment {DEPLOY_ENVIRONMENT}",
            # A job that runs in an environment gets this subject (not the branch ref).
            assumed_by=github_principal(f"repo:{repo}:environment:{DEPLOY_ENVIRONMENT}"),
            max_session_duration=Duration.hours(2),  # a cold GPU image build and push
        )
        self.deploy_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole"],
                resources=[bootstrap_role_arn(cfg, kind) for kind in BOOTSTRAP_ROLES],
            )
        )

        self.pr_role = iam.Role(
            self,
            "PrRole",
            role_name=cfg.name("gh-pr"),
            description=f"GitHub Actions pull-request checks for {repo} (cdk diff only)",
            assumed_by=github_principal(f"repo:{repo}:pull_request"),
            max_session_duration=Duration.hours(1),
        )
        self.pr_role.add_to_policy(
            iam.PolicyStatement(
                actions=["sts:AssumeRole"], resources=[bootstrap_role_arn(cfg, "lookup")]
            )
        )

        self.build_cache = ecr.Repository(
            self,
            "BuildCache",
            repository_name=cfg.name("build-cache"),
            image_tag_mutability=ecr.TagMutability.MUTABLE,
            removal_policy=RemovalPolicy.DESTROY,
            empty_on_delete=True,
            lifecycle_rules=[
                ecr.LifecycleRule(
                    description="Cache manifests replaced by a newer build",
                    tag_status=ecr.TagStatus.UNTAGGED,
                    max_image_age=Duration.days(3),
                )
            ],
        )
        self.build_cache.add_to_resource_policy(
            iam.PolicyStatement(
                principals=[iam.ArnPrincipal(bootstrap_role_arn(cfg, "image-publishing"))],
                actions=[
                    "ecr:BatchCheckLayerAvailability",
                    "ecr:BatchGetImage",
                    "ecr:GetDownloadUrlForLayer",
                    "ecr:InitiateLayerUpload",
                    "ecr:UploadLayerPart",
                    "ecr:CompleteLayerUpload",
                    "ecr:PutImage",
                ],
            )
        )

        CfnOutput(self, "BuildCacheRepo", value=self.build_cache.repository_uri)
        CfnOutput(self, "DeployRoleArn", value=self.deploy_role.role_arn)
        CfnOutput(self, "PrRoleArn", value=self.pr_role.role_arn)
