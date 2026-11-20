#!/usr/bin/env python3
"""CDK app. Pick the environment with context: cdk deploy -c env=dev"""

from __future__ import annotations

import aws_cdk as cdk

from config import ConfigError, load_config
from stacks import build_stacks

app = cdk.App()

env_name = app.node.try_get_context("env")
if not env_name:
    raise SystemExit("Pass the environment as context, for example: cdk synth -c env=dev")
try:
    cfg = load_config(env_name)
except ConfigError as e:
    raise SystemExit(f"Config error: {e}") from None

build_stacks(app, cfg)

cdk.Tags.of(app).add("Project", "batch-obj-det-aws")
cdk.Tags.of(app).add("Environment", cfg.env_name)

app.synth()
