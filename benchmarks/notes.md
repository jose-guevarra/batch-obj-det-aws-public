## Notes

### TensorRT FP16: slower end to end than its inference suggests (open)

TensorRT has the fastest inference measured: 5.4 ms per image at batch 32, against 10 ms for PyTorch FP16. On 5,000 images it still reaches only 53.7 images/s with the GPU 28% busy. The cause is outside the model:

- **Loading slows down.** While TensorRT runs, image loading (download, decode, letterbox) takes 11–14 ms per image instead of about 6, so the GPU waits for input.
- **It isn't CPU load.** Per-thread profiles (`THREAD_PROFILE=1`) show the process using about 110% of the 4 vCPUs' 400%, less than PyTorch FP16's 190%. The loader threads are blocked, not busy.
- **Ruled out: building the engine inside the worker process.** An A/B on one instance (2026-11-12, 500 images each, `make bench-trt-ab`, `benchmarks/smoke/`) gave each run a new, empty cache folder:

  | Run | images/s | GPU busy | Load, ms/image | Engine |
  | --- | --- | --- | --- | --- |
  | A: built inside the worker | 52.8 | 40% | 12.6 | built in-process |
  | B: built in a subprocess, then loaded | 55.6 | 44% | 11.4 | loaded from cache |
  | C: PyTorch FP16 reference | 73.4 | 69% | 6.0 | – |

  B's worker built nothing (`engine_from_cache: true`) and was still slow, so the in-process build isn't the cause. The workers now build in a subprocess anyway (`TRT_PREBUILD`, on by default): it doesn't change throughput, but the startup line shows whether the engine came from the cache.
- **Still unexplained:** one earlier run reached 77.6 images/s with loading at 5.9 ms per image. Its task loaded an engine an earlier task had built and warmed up in 1.9 s. Every slow run (52.8–57.7 images/s) spent about 7.5 minutes building an engine in its task first, in-process or in a subprocess. So the slowdown may follow a build anywhere in the task, or the fast run may be an outlier. The test that would tell them apart, a run that loads a cached engine with no build in the task, wasn't run.
- **Already in place:**
  - CUDA graphs (`trt_cuda_graph_enable`) with fixed input and output buffers. mAP is unchanged at 0.5022.
  - A per-instance engine cache volume, so only the first task on an instance spends about 7.5 minutes building the engine.
  - ONNX Runtime limited to one non-spinning CPU thread.
  - The engine build in a subprocess (`TRT_PREBUILD`) and the `engine_from_cache` check.
- **Status:** left open. The report keeps the 53.7 images/s row; PyTorch FP16 remains the default.

### Other notes

- **Experiments vs. report rows:** 500-image runs (experiments, in `benchmarks/smoke/`) run slower per image than 5,000-image rows, because pipeline fill and drain weigh more. Compare them only with each other.
- **Cold starts:** ECS managed scaling launches 2 GPU instances for the first task from zero. The idle one is removed about 15 minutes later.
