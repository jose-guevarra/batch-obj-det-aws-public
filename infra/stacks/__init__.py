"""All stacks for one environment, wired together."""

from __future__ import annotations

from dataclasses import dataclass

import aws_cdk as cdk
from constructs import Construct

from config import EnvConfig
from stacks.api import ApiStack
from stacks.ci import CiStack
from stacks.data import DataStack
from stacks.detector_batch import DetectorBatchStack


@dataclass(frozen=True)
class EnvStacks:
    data: DataStack
    batch: DetectorBatchStack
    api: ApiStack
    ci: CiStack


def build_stacks(scope: Construct, cfg: EnvConfig) -> EnvStacks:
    """Create DetectorData-<env>, DetectorBatch-<env>, DetectorApi-<env> and DetectorCi-<env>."""
    env = cdk.Environment(account=cfg.account, region=cfg.region)

    data = DataStack(scope, f"DetectorData-{cfg.env_name}", cfg=cfg, env=env)
    batch = DetectorBatchStack(
        scope, f"DetectorBatch-{cfg.env_name}", cfg=cfg, table=data.table, env=env
    )
    # Explicit so deploy and destroy order is right before the batch stack
    # references the table (worker grants) in later milestones.
    batch.add_stack_dependency(data)

    api = ApiStack(
        scope,
        f"DetectorApi-{cfg.env_name}",
        cfg=cfg,
        bucket=batch.bucket,
        table=data.table,
        finalize_fn=batch.finalize_fn,
        env=env,
    )
    api.add_stack_dependency(batch)

    # GitHub Actions roles; independent of the others (deployed locally, once).
    ci = CiStack(scope, f"DetectorCi-{cfg.env_name}", cfg=cfg, env=env)

    return EnvStacks(data=data, batch=batch, api=api, ci=ci)
