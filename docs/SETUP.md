# Setup and deployment

How to deploy the batch detector into your own AWS account, run batches, use the UI, reproduce the benchmark and, optionally, set up CI/CD. For what the project is and what it measured, see the [README](../README.md); for the reasoning behind each choice, see [design.md](design.md).

Every operational command is a `make` target in `infra/`; run `make help` there to list them. Commands below run from `infra/` unless noted.

## 1. What gets deployed and what it costs

Four CloudFormation stacks per environment (`dev` here):

| Stack | Holds |
| --- | --- |
| `DetectorData-<env>` | DynamoDB results table |
| `DetectorBatch-<env>` | S3 bucket (S3 → SQS notification on `input/`), queue and dead-letter queue, VPC (public subnets, no NAT, S3/DynamoDB gateway endpoints), ECS cluster, GPU workers (`g4dn.xlarge` capacity provider, service, step scaling on queue backlog), CPU workers (`c6i.2xlarge`, benchmark tasks only), the finalize and unzip functions; deploys after the data stack |
| `DetectorApi-<env>` | The batch API: FastAPI on Lambda behind an API Gateway REST API whose resource policy allows only `api_allowed_cidrs`; deploys last |
| `DetectorCi-<env>` | Optional. GitHub Actions OIDC roles and the CI build-cache ECR repo ([section 11](#11-cicd-with-github-actions-optional)) |

**Cost:**
- **Idle:** no workers run until images arrive, and there's no NAT gateway, so an idle deployment costs only storage (S3, ECR images, DynamoDB).
- **Workers,** only while there's work (us-west-2 On-Demand): `g4dn.xlarge` at $0.526 an hour, `c6i.2xlarge` at $0.34.
- **A GPU cold start** costs about $0.10–0.15. From zero, ECS launches 2 GPU instances for the first task and removes the idle one about 15 minutes later.
- **The full benchmark sweep** plus the CPU row costs about $1–2 of instance time.

**Before you start, check your EC2 quotas** in the target region (*Service Quotas → Amazon EC2*):
- **Running On-Demand G and VT instances** (counted in vCPUs): a `g4dn.xlarge` needs 4, and a cold start briefly launches 2, so ask for at least 8. New accounts often start at 0, and a request can take a day.
- **Running On-Demand Standard instances:** a `c6i.2xlarge` needs 8 vCPUs, and only for the CPU benchmark row.

## 2. Prerequisites

- **Python 3.14** and [uv](https://docs.astral.sh/uv/). 3.15 isn't supported yet because ONNX Runtime has no 3.15 wheels.
- **AWS CLI v2** with credentials for the target account (an SSO profile, or any other credentials the CLI can use).
- **Node.js 22 or 24,** for the AWS CDK CLI. `make install` installs the CLI into `infra/node_modules`; no global install is needed.
- **Docker** with buildx. The worker images are built for `linux/amd64` (emulated on Apple silicon); the Lambda image is built for `arm64`.
- **s5cmd,** for uploading image sets to S3.
- **wget and unzip,** for the COCO download.
- **Optional:** the GitHub CLI (`gh`) for the CI/CD setup.

## 3. Install

```bash
git clone https://github.com/jose-guevarra/batch-obj-det-aws.git
cd batch-obj-det-aws/infra
make install     # uv sync + npm ci (project-local CDK CLI)
```

## 4. Configure

```bash
cp environments.conf.example.json environments.conf.json   # then fill it in
```

`infra/environments.conf.json` is gitignored. Each top-level key is an environment name; the Makefile uses `ENV=dev` by default.

| Key | Meaning |
| --- | --- |
| `account`, `region` | Where the environment deploys |
| `retain_data` | `true` keeps the bucket and table when the stack is deleted; `false` deletes them |
| `api_allowed_cidrs` | IP ranges allowed to reach the batch API (find your IP with `curl -s https://checkip.amazonaws.com` and write it as `x.x.x.x/32`). This allowlist is the API's only access control. |
| `model` | YOLO model name (`yolov8m`) |
| `gpu_instance_type`, `cpu_instance_type` | Worker instance types |
| `gpu_max_instances`, `cpu_max_instances` | Auto Scaling caps; these limit cost, not the EC2 quota |
| `github_repo` | Only for CI/CD: the GitHub repository whose Actions runs may assume the CI roles ([section 11](#11-cicd-with-github-actions-optional)) |

**AWS profile:** every target passes `--profile $(PROFILE)`. `PROFILE` defaults to `$AWS_PROFILE`, or to `default` if that isn't set. Either export it once or pass it per command:

```bash
export AWS_PROFILE=my-dev-profile
make whoami                          # aws sts get-caller-identity: check the account
make deploy PROFILE=other-profile    # or per command
```

The Python scripts in `scripts/` take `--profile` too, and fall back to `$AWS_PROFILE` and then boto3's default credential chain.

## 5. Bootstrap and deploy

```bash
make login       # aws sso login (skip if you don't use SSO)
make bootstrap   # once per account and region (read from the config)
make deploy STACKS="DetectorData-dev DetectorBatch-dev DetectorApi-dev"
make outputs     # bucket, table, queue URLs, API URL
```

`make deploy` runs the infra tests, builds the three images and runs `cdk deploy`. It asks for confirmation before IAM or security-group changes.

**Name the three application stacks** unless you're setting up CI/CD. Plain `make deploy` (empty `STACKS`) deploys every stack, including `DetectorCi-<env>`, which imports an existing GitHub OIDC provider and fails in an account without one. Section 11 covers that stack.

**The first deploy takes a while.** The CPU image is about 715 MB, but the GPU image is about 7 GB compressed because it carries CUDA, cuDNN and TensorRT. On Apple silicon, both worker images are built under emulation. The first push of the GPU image to ECR takes about 35 minutes at 20 Mbit/s upload. Later deploys push only the layers that changed, usually just `src/`.

Deploys never start workers: the services stay at 0 tasks until images arrive in `input/`.

## 6. Get the data

The benchmark and the `make upload` command use the COCO 2017 validation set (5,000 images, about 1 GB, plus annotations for mAP scoring). Run the download from the repository root:

```bash
cd data && bash dl_data.sh     # → data/datasets/coco/val2017/ and annotations/
```

Only the images go to S3; the annotations stay local and are used only to score mAP.

## 7. Run a batch

```bash
make upload N=100 CORRUPT=3     # new batch: writes the batch record, then s5cmd to input/<batch_id>/, and waits
make status BATCH=<batch_id>    # counters, queue and dead-letter depth, workers, instances
make logs                       # follow GPU worker logs (one JSON line per batch); KIND=cpu for CPU
make rerun BATCH=<batch_id> N=100 CORRUPT=3   # same files again: all skipped, counters unchanged
```

`CORRUPT=3` mixes in 3 garbage files, which show up as failed images and in the dead-letter queue.

**How scaling behaves:**
- **The GPU workers own the queue.** Uploads to `input/` start them; the CPU workers never start on their own.
- **Scale-up:** workers start at 0. About a minute after the first upload, the service asks for one task, the capacity provider launches an instance and the worker starts. That takes a few minutes from cold, longer for the first pull of the GPU image.
- **Scale-in:** after 5 minutes with an empty queue, the service returns to 0 (about 6 minutes in practice), and ECS removes the idle instances over the next 15–20 minutes.

**Emergency stop:** `make scale-down` caps the GPU workers at 0 whatever the queue holds and stops any benchmark task. `make scale-restore` undoes it.

### Bucket layout

| Prefix | Contents |
| --- | --- |
| `input/{batch_id}/` | Images to process; the only prefix that queues SQS messages |
| `upload/{batch_id}/` | Zips, extracted into `input/` |
| `results/{batch_id}/` | Worker part files (`part-<worker>-<run>-<seq>.jsonl`), combined into `results.jsonl` and `summary.json` at finalize |
| `rendered/{batch_id}/` | Annotated images, rendered on demand |
| `datasets/coco-val2017/` | The benchmark set, staged once; each run copies it into `input/` server-side |

## 8. Use the UI and the API

```bash
make ui                    # looks up the API URL, then Streamlit on http://localhost:8501
make api-url               # just the URL; API_URL=<url> make ui skips the lookup
make logs-fn FN=finalize   # follow a function's logs (api, finalize or unzip)
```

The Streamlit app runs on your machine and talks only to the API. It needs no AWS credentials; `make ui` uses them once, to read the URL from the stack outputs.

- **New batch:** pick images (up to 1,000) or one zip, then upload. The browser uploads straight to S3 through presigned URLs. Use `make upload` for larger sets such as COCO.
- **Batch page:** a progress bar while the batch runs (waiting for a GPU worker → processing → finalizing → done), then:
  - the summary: counts, detections per class, mean confidence, time and estimated cost;
  - `results.csv` and `results.jsonl` downloads;
  - the failed images;
  - an image browser. Viewing an image renders it with its boxes the first time and serves the cached copy after that.
- **What costs money:** only uploads, because they start a GPU worker (about 4–5 minutes and $0.10–0.15 per cold start). Status, browsing, rendering and downloads run on Lambda, cost fractions of a cent and have no idle cost.

**API endpoints:**

| Method and path | Does |
| --- | --- |
| `GET /health` | Health check |
| `POST /batches` | Creates a batch from `{"files": [...]}` or `{"zip": true}` and returns presigned upload URLs |
| `GET /batches/{id}` | Status, progress, the summary and a results link when done |
| `GET /batches/{id}/images?cursor=` | One page of results |
| `GET /batches/{id}/images/{name}` | Detections and the original's URL |
| `GET /batches/{id}/images/{name}/annotated` | Rendered once, then cached under `rendered/` |

## 9. Reproduce the benchmark

```bash
make bench-stage                # once: upload COCO val2017 to datasets/ (not queued)
make bench-sweep                # GPU PyTorch FP32 rows at batch 1..64 (PIPELINE=1 for pipelined), then the report
make bench KIND=cpu BS=1        # the CPU baseline row
make bench BS=32 RUNTIME=onnx-trt HALF=1   # any single row (IMAGES=500 for a quick check)
make bench-eval RUNTIME=onnx-trt HALF=1    # score mAP50-95 for a setup (benchmarks/evals/)
make bench-m6                   # pipelined sweep, FP16 / ONNX Runtime rows, mAP evals
make bench-trt-ab               # the TensorRT A/B experiment (500 images per run)
make bench-report               # docs/benchmark.md and the charts in docs/img/
```

Every row follows the same rules:
- The GPU service is capped at 0 for the run, so no other worker takes messages.
- The 5,000 staged images are copied into `input/<batch_id>/` server-side, and the worker starts only once the queue holds every message.
- Exactly one worker task runs (ECS RunTask with overrides such as `BATCH_SIZE`, `HALF`, `RUNTIME` and `PIPELINE`).
- The time runs from the first message received to the last result written, read from the worker's logs.

Each run is saved as `benchmarks/runs/<batch_id>.json`. The GPU service's scaling is restored afterwards, even after an error or Ctrl-C.

**TensorRT knobs** (for the open issue in the README):
- **`RUNTIME=onnx-trt`** runs ONNX Runtime's TensorRT backend. The engine takes about 7.5 minutes to build on a T4 and is cached per instance in `TRT_CACHE_DIR` (default `/tmp/trt-cache`, a Docker volume shared by the tasks on an instance).
- **`TRT_PREBUILD`** (default on) builds the engine in a short-lived subprocess before the worker loads it; `TRT_PREBUILD=0` builds it inside the worker process instead.
- **`TRT_OPTIONS`** takes a JSON object of `trt_*` provider options that override the defaults, e.g. `{"trt_cuda_graph_enable": false}`.
- **`THREAD_PROFILE=1`** logs CPU % per thread with every batch.
- **The startup log line** reports `prebuild_s`, `engine_from_cache` and `warmup_s`.

## 10. Worker configuration reference

`python -m detector.worker` reads `QUEUE_URL`, `DLQ_URL`, `BUCKET` and `TABLE` (set by the stacks) and these optional variables:

| Variable | Meaning |
| --- | --- |
| `BATCH_SIZE`, `MAX_WAIT_S` | Images per batch; how long to wait for a full batch |
| `DEVICE`, `HALF` | `cuda` or `cpu`; `HALF=1` runs FP16 |
| `RUNTIME` | `pytorch` (default), `onnx-cuda` or `onnx-trt`. The ONNX models are exported when the GPU image is built. |
| `PIPELINE` | `1` (GPU default) runs prepare, infer and finish in three threads; `0` runs them one after another |
| `CONF`, `IOU`, `MAX_DET`, `MULTI_LABEL` | Postprocessing (eval runs use conf 0.001, multi-label, like Ultralytics validation) |
| `WORKER_ID` | Name used in part-file names and logs |
| `FINALIZE_FUNCTION` | Function invoked asynchronously when a commit completes a batch (empty: off) |
| `TRT_CACHE_DIR`, `TRT_PREBUILD`, `TRT_OPTIONS`, `THREAD_PROFILE` | TensorRT and profiling, as in section 9 |

**The GPU service default** is pipelined PyTorch FP16 at batch 32, the cheapest setup measured.

**Logs:** every batch logs one JSON line (`"event": "batch"`) with:
- the counts and per-stage times (`stages_ms`);
- wall-clock `t_receive` and `t_written`;
- on GPU, the mean utilization and peak memory over the batch (sampled with NVML every 500 ms).

At startup, the worker logs one `"event": "startup"` line with the runtime, versions, GPU, driver and warm-up time.

## 11. CI/CD with GitHub Actions (optional)

GitHub Actions authenticates to AWS through OIDC: each run assumes an IAM role for a short-lived session, and no AWS keys are stored in GitHub.

| Workflow | When | What |
| --- | --- | --- |
| `ci.yml` | Pull request into `develop` | The checks, then `cdk diff` with the read-only PR role, posted as a PR comment |
| `deploy.yml` | Push to `develop` | The checks; then, **only if the commit is a merged PR's merge commit**, `cdk deploy` of DetectorData, DetectorBatch and DetectorApi with the deploy role |
| `checks.yml` | Reused by both | Lint, unit tests (summary in the job, JUnit file as an artifact), `cdk synth` |

**Setup, in your fork:**

1. **Create the GitHub OIDC provider** if the account doesn't have one yet. An account can have only one per URL, and `DetectorCi` imports it.

   ```bash
   aws iam create-open-id-connect-provider \
     --url https://token.actions.githubusercontent.com --client-id-list sts.amazonaws.com
   ```

2. **Set `github_repo`** in `environments.conf.json` to your repository, exactly as GitHub writes it in the OIDC subject. Newer repositories use immutable IDs, `owner@<owner id>/name@<repo id>`, while older tokens may still use `owner/name`. If unsure, deploy, let a workflow fail, and read the subject from the failed `AssumeRoleWithWebIdentity` event (`AccessDenied`) in CloudTrail.

3. **Deploy the CI stack locally** (CI never deploys it):

   ```bash
   make deploy-ci     # DetectorCi-dev: the deploy and PR roles, and the build-cache ECR repo
   make ci-outputs    # the two role ARNs
   ```

4. **Configure the repository:**
   - **Environment `dev`:** *Settings → Environments → New environment*. Under *Deployment branches and tags*, choose *Selected branches* and add `develop`. The deploy role trusts only jobs in this environment, so this rule keeps other branches out. Optionally, add a required reviewer to approve each deploy.
   - **Secret `ENVIRONMENTS_CONF_JSON`:** the contents of `infra/environments.conf.json`.
   - **Variables `AWS_DEPLOY_ROLE_ARN` and `AWS_PR_ROLE_ARN`:** from `make ci-outputs`. Optionally `AWS_REGION` (default `us-west-2`).

   ```bash
   gh secret set ENVIRONMENTS_CONF_JSON < infra/environments.conf.json
   gh variable set AWS_DEPLOY_ROLE_ARN --body arn:aws:iam::<account>:role/detector-gh-deploy-dev
   gh variable set AWS_PR_ROLE_ARN --body arn:aws:iam::<account>:role/detector-gh-pr-dev
   ```

**What to expect:**
- **Images build on the runner,** without a GPU, and go straight to ECR. `scripts/ci_publish_images.py` reads the synthesized asset manifests and runs `docker buildx build --push` for each image not yet in ECR, because loading the GPU image into the runner's Docker store would need more disk than the runner has. `cdk deploy` then finds every image already in ECR.
- **Builds reuse cached layers** from the `detector-build-cache-<env>` ECR repo, so a code-only change rebuilds only the `src` layers. An image already in ECR (same content hash, from CI or a local `make deploy`) isn't rebuilt at all.
- **The first CI deploy** builds the GPU image from scratch, in about 10–15 minutes.
- **A direct push to `develop`** runs the checks and skips the deploy.
- **A local `make deploy`** still works. If you change `api_allowed_cidrs`, update the secret too.

## 12. Local development

```bash
uv run pytest            # fast tests, including the worker against moto (no AWS, no weights)
uv run pytest -m slow    # real model: Ultralytics parity, 100-image mAP, end-to-end worker
uv run ruff check . && uv run ruff format --check .
```

**Local accuracy check:** `src/detector/inference_core.py` holds the preprocessing (decode, fixed 640×640 letterbox) and postprocessing (NMS, box rescaling) that every runtime shares. `scripts/evaluate_map.py` runs it over COCO val2017 and scores mAP50-95 with pycocotools:

```bash
uv run python scripts/evaluate_map.py              # all 5,000 images, auto device (MPS on Apple silicon)
uv run python scripts/evaluate_map.py --limit 500 --device cpu
```

The YOLOv8m weights download to `models/` on first use, and summaries go to `results/`. The defaults match Ultralytics validation (conf 0.001, IoU 0.7, max 300 detections, multi-label). Expected result: mAP50-95 0.5025 on all 5,000 images (published: 0.502).

## 13. Tear down

```bash
make scale-down                                                   # optional: stop any running workers first
make destroy STACKS="DetectorApi-dev DetectorBatch-dev DetectorData-dev"
make destroy STACKS=DetectorCi-dev                                 # if you deployed it
```

With `retain_data: false`, the bucket and table are deleted with their stacks; with `true`, they're kept and must be deleted by hand. The CDK bootstrap stack (`CDKToolkit`) and its staging bucket and ECR repository stay until you delete them.

## 14. Troubleshooting

- **403 from the API, or a red dot in the UI's sidebar:** your IP isn't in `api_allowed_cidrs` (it may have changed). Update the config and redeploy (`make deploy STACKS=DetectorApi-dev`), and update the CI secret if you use one.
- **The upload finished, but no worker starts:** check `make status`. If the ECS task stays in `PROVISIONING`, the capacity provider can't launch a `g4dn.xlarge`; look for a quota or capacity error in the Auto Scaling group's activity history ([section 1](#1-what-gets-deployed-and-what-it-costs)).
- **Scaling seems slow:** the backlog alarm waits on SQS metrics in CloudWatch, which lag by about a minute. Expect about 3.5–4.5 minutes from upload to a running worker.
- **The first deploy seems stuck on the GPU image:** it's pushing about 7 GB; see [section 5](#5-bootstrap-and-deploy).
- **A CI deploy fails with `AccessDenied` on `AssumeRoleWithWebIdentity`:** the `github_repo` subject doesn't match what GitHub sent, or the job isn't in the `dev` environment ([section 11](#11-cicd-with-github-actions-optional)).
