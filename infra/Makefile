# CDK commands for the batch detector stacks. Run from infra/:
#   make deploy              # dev, with $AWS_PROFILE (or the default profile)
#   make diff ENV=dev PROFILE=other-profile

ENV     ?= dev
PROFILE ?= $(or $(AWS_PROFILE),default)
# Empty = every stack. synth takes no --all flag; diff/deploy/destroy need it.
STACKS  ?=
ALL_OR_STACKS = $(or $(STACKS),--all)

ROOT := $(abspath ..)
# Project-local CDK CLI (infra/package.json), installed by `make install`.
CDK  := ./node_modules/.bin/cdk
CDK_ARGS := -c env=$(ENV) --profile $(PROFILE)

# Account and region come from environments.conf.json (evaluated only when used).
TARGET = $(shell uv run --quiet python -c "from config import load_config; c = load_config('$(ENV)'); print(c.account, c.region)")
ACCOUNT = $(word 1,$(TARGET))
REGION  = $(word 2,$(TARGET))

# Batch operations (scripts/upload_batch.py): make upload N=100 CORRUPT=3
N       ?= 100
CORRUPT ?= 0
BATCH   ?=
KIND    ?= gpu
CLUSTER  = detector-cluster-$(ENV)
# The GPU service owns the queue; CPU workers only run as benchmark tasks.
GPU_SVC  = detector-worker-gpu-$(ENV)
SCALABLE = --service-namespace ecs --scalable-dimension ecs:service:DesiredCount \
           --resource-id service/$(CLUSTER)/$(GPU_SVC)
GPU_MAX  = $(shell uv run --quiet python -c "from config import load_config; print(load_config('$(ENV)').gpu_max_instances)")

# Benchmarks (scripts/benchmark.py): make bench KIND=gpu BS=64 RUNTIME=onnx-trt HALF=1
BS      ?= 64
HALF    ?=
RUNTIME ?= pytorch
# PIPELINE=1 / PIPELINE=0 forces the worker mode; empty = pipelined on GPU, serial on CPU.
PIPELINE ?=
IMAGES  ?= 5000
BENCH    = cd $(ROOT) && uv run python scripts/benchmark.py --env $(ENV) --profile $(PROFILE)
RUN_ARGS = --kind $(KIND) --batch-size $(BS) --runtime $(RUNTIME) --images $(IMAGES) $(if $(HALF),--half) \
           $(if $(filter 1,$(PIPELINE)),--pipeline) $(if $(filter 0,$(PIPELINE)),--no-pipeline)

.DEFAULT_GOAL := help
.PHONY: help install check-cdk cdk-version check-config login whoami bootstrap synth diff deploy destroy \
        outputs test lint format clean upload rerun status logs scale-down scale-restore \
        bench-stage bench bench-eval bench-sweep bench-m6 bench-trt-ab bench-report onnx-local api-url ui logs-fn deploy-ci ci-outputs

help: ## Show this help
	@grep -E '^[a-z0-9-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  %-14s %s\n", $$1, $$2}'
	@echo
	@echo "  Variables: ENV=$(ENV) PROFILE=$(PROFILE) STACKS='$(STACKS)' (empty = all)"

install: ## Install Python dependencies (uv sync) and the local CDK CLI (npm ci)
	cd $(ROOT) && uv sync
	npm ci

check-cdk:
	@test -x $(CDK) || { echo "CDK CLI not found in infra/node_modules: run make install"; exit 1; }

cdk-version: check-cdk ## Show the local CDK CLI version
	$(CDK) --version

check-config:
	@test -f environments.conf.json || test -n "$$ENVIRONMENTS_CONF" || { \
	  echo "Missing infra/environments.conf.json: cp environments.conf.example.json environments.conf.json"; exit 1; }

login: ## Log in with AWS SSO
	aws sso login --profile $(PROFILE)

whoami: ## Show the AWS identity for PROFILE
	aws sts get-caller-identity --profile $(PROFILE)

bootstrap: check-cdk check-config ## Bootstrap CDK in ENV's account and region (once)
	$(CDK) bootstrap aws://$(ACCOUNT)/$(REGION) --profile $(PROFILE)

synth: check-cdk check-config ## Synthesize CloudFormation templates into cdk.out/
	$(CDK) synth $(STACKS) $(CDK_ARGS) --quiet

diff: check-cdk check-config ## Show changes against the deployed stacks
	$(CDK) diff $(ALL_OR_STACKS) $(CDK_ARGS)

deploy: check-cdk check-config test ## Run tests, then deploy (asks before IAM/security changes)
	$(CDK) deploy $(ALL_OR_STACKS) $(CDK_ARGS) --outputs-file cdk.out/outputs-$(ENV).json

deploy-ci: check-cdk check-config test ## Deploy DetectorCi-<env> (the GitHub Actions roles); local only
	$(CDK) deploy DetectorCi-$(ENV) $(CDK_ARGS)

ci-outputs: check-config ## Print the GitHub Actions role ARNs (for the repo variables)
	@aws cloudformation describe-stacks --stack-name DetectorCi-$(ENV) \
	  --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table \
	  --profile $(PROFILE) --region $(REGION)

destroy: check-cdk check-config ## Delete the stacks (asks to confirm; with retain_data false, data is deleted too)
	$(CDK) destroy $(ALL_OR_STACKS) $(CDK_ARGS)

outputs: check-config ## Print the deployed stack outputs (bucket, table, queue URLs)
	@for s in DetectorData DetectorBatch DetectorApi; do \
	  aws cloudformation describe-stacks --stack-name $$s-$(ENV) \
	    --query 'Stacks[0].Outputs[].[OutputKey,OutputValue]' --output table \
	    --profile $(PROFILE) --region $(REGION) || exit 1; \
	done

upload: ## Upload the first N COCO images (+ CORRUPT garbage files) as a new batch and wait
	cd $(ROOT) && uv run python scripts/upload_batch.py --env $(ENV) --profile $(PROFILE) \
	  --limit $(N) --corrupt $(CORRUPT) $(if $(BATCH),--batch-id $(BATCH)) --wait

rerun: ## Re-upload the same files to BATCH (all should be skipped): make rerun BATCH=<id> N=100
	@test -n "$(BATCH)" || { echo "usage: make rerun BATCH=<batch id> N=<n> [CORRUPT=<k>]"; exit 1; }
	cd $(ROOT) && uv run python scripts/upload_batch.py --env $(ENV) --profile $(PROFILE) \
	  --limit $(N) --corrupt $(CORRUPT) --batch-id $(BATCH) --rerun

status: ## Batch counters, queue depths, workers and instances: make status [BATCH=<id>]
	cd $(ROOT) && uv run python scripts/batch_status.py $(BATCH) --env $(ENV) --profile $(PROFILE)

logs: ## Follow worker logs: make logs [KIND=gpu|cpu]
	aws logs tail /detector/$(ENV)/worker-$(KIND) --follow --since 30m --profile $(PROFILE) --region $(REGION)

logs-fn: ## Follow a Lambda function's logs: make logs-fn FN=api|finalize|unzip
	aws logs tail /detector/$(ENV)/$(or $(FN),api) --follow --since 30m --profile $(PROFILE) --region $(REGION)

api-url: check-config ## Print the batch API's URL (DetectorApi-<env> output)
	@aws cloudformation describe-stacks --stack-name DetectorApi-$(ENV) \
	  --query "Stacks[0].Outputs[?OutputKey=='ApiUrl'].OutputValue" --output text \
	  --profile $(PROFILE) --region $(REGION)

ui: check-config ## Start the Streamlit UI on http://localhost:8501 (API_URL=... skips the lookup)
	@url="$${API_URL:-$$($(MAKE) -s api-url)}"; \
	  test -n "$$url" -a "$$url" != "None" || { echo "No API URL: deploy DetectorApi-$(ENV) first"; exit 1; }; \
	  echo "API: $$url"; \
	  cd $(ROOT) && API_URL="$$url" uv run streamlit run ui/app.py --server.headless true

scale-down: ## Emergency stop: cap GPU workers at 0 and stop benchmark tasks (undo: scale-restore)
	aws application-autoscaling register-scalable-target $(SCALABLE) --min-capacity 0 --max-capacity 0 \
	  --profile $(PROFILE) --region $(REGION)
	aws ecs update-service --cluster $(CLUSTER) --service $(GPU_SVC) --desired-count 0 \
	  --profile $(PROFILE) --region $(REGION) --query 'service.desiredCount'
	@for t in $$(aws ecs list-tasks --cluster $(CLUSTER) --started-by benchmark --query 'taskArns[]' \
	    --output text --profile $(PROFILE) --region $(REGION)); do \
	  aws ecs stop-task --cluster $(CLUSTER) --task $$t --reason "make scale-down" \
	    --profile $(PROFILE) --region $(REGION) --query 'task.taskArn' --output text; \
	done
	@echo "Instances terminate once their tasks stop (capacity provider). Restore: make scale-restore"

scale-restore: ## Let GPU workers scale from the queue again (0..gpu_max_instances)
	aws application-autoscaling register-scalable-target $(SCALABLE) --min-capacity 0 \
	  --max-capacity $(GPU_MAX) --profile $(PROFILE) --region $(REGION)

bench-stage: ## Upload COCO val2017 to datasets/ once (not queued), for benchmark runs
	$(BENCH) stage

bench: ## One timed row: make bench [KIND=gpu] [BS=64] [RUNTIME=onnx-trt] [HALF=1] [PIPELINE=0|1]
	$(BENCH) run $(RUN_ARGS)

bench-eval: ## Score mAP50-95 for a setup: make bench-eval [KIND=gpu] [RUNTIME=...] [HALF=1]
	$(BENCH) run $(RUN_ARGS) --eval

bench-sweep: ## GPU PyTorch FP32 rows at batch 1..64 [PIPELINE=1], then the report
	$(BENCH) sweep --images $(IMAGES) $(if $(filter 1,$(PIPELINE)),--pipeline) $(if $(filter 0,$(PIPELINE)),--no-pipeline)

bench-m6: ## Milestone 6: pipelined sweep, FP16 / ONNX Runtime rows, mAP evals, report
	$(BENCH) milestone6 --images $(IMAGES)

bench-trt-ab: ## TensorRT A/B (500 images): engine built in the worker vs. in a subprocess, + PyTorch FP16
	$(BENCH) trt-ab

bench-report: ## Write docs/benchmark.md and the charts from benchmarks/
	$(BENCH) report

onnx-local: ## Copy the ONNX models out of the locally built GPU image into models/ (parity tests)
	cd $(ROOT) && docker buildx build --platform linux/amd64 -f docker/Dockerfile.gpu -t detector-gpu:local . -q >/dev/null
	cd $(ROOT) && id=$$(docker create --platform linux/amd64 detector-gpu:local) && \
	  docker cp $$id:/app/models/yolov8m.onnx models/ && docker cp $$id:/app/models/yolov8m-fp16.onnx models/; \
	  status=$$?; docker rm $$id >/dev/null; exit $$status

test: ## Run the infra tests
	cd $(ROOT) && uv run pytest tests/infra -q

lint: ## Lint and check formatting
	cd $(ROOT) && uv run ruff check infra tests/infra && uv run ruff format --check infra tests/infra

format: ## Fix lint issues and format
	cd $(ROOT) && uv run ruff check --fix infra tests/infra && uv run ruff format infra tests/infra

clean: ## Remove cdk.out and Python caches
	rm -rf cdk.out
	find . $(ROOT)/tests/infra -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf $(ROOT)/.pytest_cache $(ROOT)/.ruff_cache
