"""Inference runtimes behind one interface.

A runtime only turns a preprocessed batch into raw model output (B, 84, 8400); all
preprocessing and postprocessing lives in detector.inference_core, so every runtime is
scored on the same pipeline.

    pytorch     TorchRuntime: the Ultralytics model called directly, FP32 or FP16
    onnx-cuda   OnnxRuntime, CUDA backend: yolov8m.onnx (FP32) or yolov8m-fp16.onnx
    onnx-trt    OnnxRuntime, TensorRT backend: yolov8m.onnx, FP16 through TensorRT

The ONNX files are exported when the GPU image is built (docker/Dockerfile.gpu).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Protocol

import numpy as np
import torch

from detector.inference_core import IMG_SIZE

# Repo-local models/ in development; containers set MODELS_DIR to the baked-in weights.
DEFAULT_MODELS_DIR = Path(
    os.environ.get("MODELS_DIR") or Path(__file__).resolve().parents[2] / "models"
)


class Runtime(Protocol):
    name: str
    device: str
    precision: str

    def sync(self) -> None:
        """Block until queued device work is done (for timing)."""

    def infer(self, batch: np.ndarray) -> torch.Tensor:
        """(B, 3, 640, 640) float32 in [0, 1] -> raw output (B, 84, 8400)."""

    def warmup(self, batch_size: int = 1) -> None:
        """Run dummy batches so one-time setup stays out of the timings."""

    def describe(self) -> dict[str, object]:
        """Versions and device details for the worker's startup log."""


def pick_device(device: str = "auto") -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_weights(name: str = "yolov8m", models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """Path to <name>.pt in models/, downloading it from Ultralytics' release once."""
    from ultralytics.utils.downloads import attempt_download_asset

    path = Path(models_dir) / f"{name}.pt"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        attempt_download_asset(path)
    if not path.is_file():
        raise FileNotFoundError(f"could not download weights to {path}")
    return path


class TorchRuntime:
    """YOLO model called directly (no Ultralytics predict), FP32 or FP16."""

    name = "pytorch"

    def __init__(self, weights: str | Path, device: str = "auto", half: bool = False) -> None:
        from ultralytics import YOLO

        self.device = pick_device(device)
        if half and self.device == "cpu":
            raise ValueError("FP16 needs a GPU (cuda or mps)")
        self.precision = "fp16" if half else "fp32"
        self._dtype = torch.float16 if half else torch.float32

        model = YOLO(str(weights)).model.fuse().eval().to(self.device)
        self.model = model.half() if half else model
        self.warmup()

    def sync(self) -> None:
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def describe(self) -> dict[str, object]:
        """Versions and device details for the worker's startup log."""
        info: dict[str, object] = {
            "runtime": self.name,
            "precision": self.precision,
            "device": self.device,
            "torch": torch.__version__,
        }
        if self.device.startswith("cuda"):
            info |= {
                "cuda": torch.version.cuda,
                "device_name": torch.cuda.get_device_name(),
                "capability": "sm_{}{}".format(*torch.cuda.get_device_capability()),
                "arch_list": torch.cuda.get_arch_list(),
            }
        return info

    @torch.inference_mode()
    def infer(self, batch: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(batch).to(self.device, dtype=self._dtype, non_blocking=True)
        out = self.model(x)
        # The Detect head returns (inference_output, raw_feature_maps) in eval mode.
        return out[0] if isinstance(out, list | tuple) else out

    def warmup(self, batch_size: int = 1) -> None:
        """One dummy pass so first-call setup doesn't land in the timings."""
        self.infer(np.zeros((batch_size, 3, IMG_SIZE, IMG_SIZE), np.float32))
        self.sync()


# ONNX Runtime -----------------------------------------------------------------------

# YOLOv8 detection output for a 640x640 input: 4 box values + 80 class scores, for the
# 80x80 + 40x40 + 20x20 anchors of strides 8, 16 and 32.
OUT_SHAPE = (4 + 80, sum((IMG_SIZE // s) ** 2 for s in (8, 16, 32)))
# Default engine and timing cache directory: a Docker volume shared by the tasks on one
# instance (infra/stacks/worker_service.py), so an engine is built once per instance.
TRT_CACHE_DIR = "/tmp/trt-cache"
TRT_PREBUILD_TIMEOUT_S = 30 * 60
ONNX_INPUT = "images"  # the input name in Ultralytics' ONNX export


def onnx_providers(
    provider: str,
    *,
    half: bool,
    batch_size: int,
    input_name: str = ONNX_INPUT,
    cache_dir: str = TRT_CACHE_DIR,
    trt_options: dict | None = None,
) -> list[tuple[str, dict]]:
    """Execution providers in priority order for provider = cuda | trt | cpu."""
    if provider == "cpu":
        return [("CPUExecutionProvider", {})]
    cuda = ("CUDAExecutionProvider", {"device_id": 0})
    if provider == "cuda":
        return [cuda]
    if provider == "trt":
        shape = f"x3x{IMG_SIZE}x{IMG_SIZE}"
        trt = {
            "device_id": 0,
            "trt_fp16_enable": half,
            "trt_engine_cache_enable": True,
            "trt_engine_cache_path": cache_dir,
            "trt_timing_cache_enable": True,
            "trt_timing_cache_path": cache_dir,
            # One engine covers every batch from 1 (a partial last batch) to batch_size.
            "trt_profile_min_shapes": f"{input_name}:1{shape}",
            "trt_profile_opt_shapes": f"{input_name}:{batch_size}{shape}",
            "trt_profile_max_shapes": f"{input_name}:{batch_size}{shape}",
            # Replay each batch as one CUDA graph: fewer CPU-side launches per batch. Needs
            # fixed input and output buffers: see OnnxRuntime._infer_static.
            "trt_cuda_graph_enable": True,
        }
        trt |= trt_options or {}  # experiments (TRT_OPTIONS) override the defaults
        # Nodes TensorRT can't take fall back to the CUDA backend.
        return [("TensorrtExecutionProvider", trt), cuda]
    raise ValueError(f"unknown ONNX Runtime provider {provider!r}")


def parse_trt_options(text: str) -> dict:
    """TRT_OPTIONS: JSON TensorRT provider options, e.g. {"trt_cuda_graph_enable": true}.

    Only trt_* keys with scalar values: a typo should fail at startup, not be ignored.
    """
    if not text.strip():
        return {}
    options = json.loads(text)
    if not isinstance(options, dict):
        raise ValueError("TRT_OPTIONS must be a JSON object")
    for key, value in options.items():
        if not key.startswith("trt_") or not isinstance(value, bool | int | float | str):
            raise ValueError(f"bad TRT_OPTIONS entry {key!r}: {value!r}")
    return options


def onnx_session_options(ort, provider: str):
    """Session options; on GPU, ONNX Runtime gets one non-spinning CPU thread.

    With a GPU backend ONNX Runtime does almost no CPU work, but by default it starts a
    thread per core that spin-waits. On a 4-vCPU g4dn those threads compete with the
    worker's download / decode / letterbox pool, and the GPU ends up waiting for input
    (seen in the TensorRT smoke test: 36% GPU busy).
    """
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if provider != "cpu":
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    return options


class OnnxRuntime:
    """An exported ONNX model in ONNX Runtime, on the CUDA or TensorRT backend (or CPU).

    On GPU the input and output are bound to torch CUDA tensors (IO binding), so the
    (B, 84, 8400) output stays on the GPU and postprocess() runs there, as with PyTorch.
    """

    NAMES = {"cuda": "onnxruntime-cuda", "trt": "onnxruntime-trt", "cpu": "onnxruntime-cpu"}

    def __init__(
        self,
        model_path: str | Path,
        provider: str = "cuda",
        half: bool = False,
        batch_size: int = 1,
        trt_options: dict | None = None,
        cache_dir: str = TRT_CACHE_DIR,
    ) -> None:
        import onnxruntime as ort

        self.name = self.NAMES[provider]
        self.device = "cpu" if provider == "cpu" else "cuda"
        self.precision = "fp16" if half else "fp32"
        self.model_path = Path(model_path)
        providers = onnx_providers(
            provider, half=half, batch_size=batch_size, cache_dir=cache_dir, trt_options=trt_options
        )
        self._provider_options = providers[0][1]
        self.batch_size = batch_size
        # A CUDA graph replays the memory addresses it captured, so its input and output must
        # live in the same buffers every run, at the full batch size.
        self._graph = bool(self._provider_options.get("trt_cuda_graph_enable"))
        self._io = None
        self.build_info: dict[str, object] = {}  # TensorRT: prebuild and cache-hit details
        if provider != "cpu":
            # Load the CUDA and cuDNN libraries from the nvidia-* pip packages, the same
            # ones torch uses (the image has no system CUDA install).
            ort.preload_dlls()
        if provider == "trt":
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
        options = onnx_session_options(ort, provider)
        self.session = ort.InferenceSession(str(model_path), options, providers=providers)
        wanted, active = providers[0][0], self.session.get_providers()
        if active[0] != wanted:
            # Never fall back silently: a CPU-run "TensorRT" row would be meaningless.
            raise RuntimeError(f"{wanted} isn't available; ONNX Runtime chose {active}")
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self._ort_version = ort.__version__

    def sync(self) -> None:
        if self.device == "cuda":
            torch.cuda.synchronize()

    def describe(self) -> dict[str, object]:
        info: dict[str, object] = {
            "runtime": self.name,
            "precision": self.precision,
            "device": self.device,
            "onnxruntime": self._ort_version,
            "providers": self.session.get_providers(),
            "provider_options": self._provider_options,
            "model_file": self.model_path.name,
            **self.build_info,
        }
        if self.device == "cuda":
            info |= {"device_name": torch.cuda.get_device_name()}
        return info

    @torch.inference_mode()
    def infer(self, batch: np.ndarray) -> torch.Tensor:
        if self.device == "cpu":
            out = self.session.run([self.output_name], {self.input_name: batch})[0]
            return torch.from_numpy(out)
        if self._graph:
            return self._infer_static(batch)
        x = torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)).to("cuda")
        torch.cuda.current_stream().synchronize()  # the copy must land before ORT reads x
        out = torch.empty((x.shape[0], *OUT_SHAPE), dtype=torch.float32, device="cuda")
        io = self._bind(x, out)
        self.session.run_with_iobinding(io)
        io.synchronize_outputs()
        return out

    def _infer_static(self, batch: np.ndarray) -> torch.Tensor:
        """Fixed full-size buffers for CUDA graphs; a partial batch is padded to full size."""
        n = len(batch)
        if n > self.batch_size:
            raise ValueError(f"batch of {n} exceeds the engine's batch size {self.batch_size}")
        if self._io is None:
            self._x = torch.zeros((self.batch_size, 3, IMG_SIZE, IMG_SIZE), device="cuda")
            self._out = torch.empty((self.batch_size, *OUT_SHAPE), device="cuda")
            self._io = self._bind(self._x, self._out)
        self._x[:n].copy_(torch.from_numpy(np.ascontiguousarray(batch, dtype=np.float32)))
        torch.cuda.current_stream().synchronize()  # the copy must land before ORT reads it
        self.session.run_with_iobinding(self._io)
        self._io.synchronize_outputs()
        return self._out[:n].clone()  # the next run overwrites the buffer

    def _bind(self, x: torch.Tensor, out: torch.Tensor):
        io = self.session.io_binding()
        io.bind_input(self.input_name, "cuda", 0, np.float32, tuple(x.shape), x.data_ptr())
        io.bind_output(self.output_name, "cuda", 0, np.float32, tuple(out.shape), out.data_ptr())
        return io

    def warmup(self, batch_size: int = 1) -> None:
        """Build TensorRT engines / pick CUDA kernels before the first timed batch."""
        for n in {1, batch_size}:  # the TensorRT profile's two ends
            self.infer(np.zeros((n, 3, IMG_SIZE, IMG_SIZE), np.float32))
        self.sync()


def onnx_model_path(model: str, half: bool, provider: str, models_dir: Path) -> Path:
    """FP16 weights for the CUDA backend; TensorRT builds FP16 itself from the FP32 file."""
    fp16_file = half and provider == "cuda"
    return Path(models_dir) / (f"{model}-fp16.onnx" if fp16_file else f"{model}.onnx")


def cache_snapshot(cache_dir: str | Path) -> dict[str, tuple[int, int]]:
    """File name -> (size, mtime_ns) for everything in a TensorRT cache directory."""
    path = Path(cache_dir)
    if not path.is_dir():
        return {}
    return {f.name: (f.stat().st_size, f.stat().st_mtime_ns) for f in path.iterdir() if f.is_file()}


def built_files(before: dict, after: dict) -> list[str]:
    """Engine files created or rewritten between two snapshots (the timing cache may be
    rewritten even when the engine loads from the cache, so it doesn't count)."""
    return sorted(
        name
        for name, stat in after.items()
        if name.endswith(".engine") and before.get(name) != stat
    )


def prebuild_command(
    model_path: str | Path,
    *,
    half: bool,
    batch_size: int,
    cache_dir: str,
    trt_options: dict | None = None,
) -> list[str]:
    cmd = [
        sys.executable, "-m", "detector.trt_prebuild",
        "--model", str(model_path),
        "--batch-size", str(batch_size),
        "--cache-dir", cache_dir,
    ]  # fmt: skip
    if half:
        cmd.append("--half")
    if trt_options:
        cmd += ["--options", json.dumps(trt_options)]
    return cmd


def prebuild_trt(cmd: list[str], run=subprocess.run) -> float:
    """Build (or load) the TensorRT engine in a short-lived process; seconds it took.

    Building inside the worker process slowed its image loading for the rest of the run
    (benchmarks/notes.md); a separate process takes the builder's memory, threads and GPU
    state with it when it exits. Its output goes to the worker's log.
    """
    started = time.monotonic()
    run(cmd, check=True, timeout=TRT_PREBUILD_TIMEOUT_S)
    return round(time.monotonic() - started, 1)


def make_runtime(
    kind: str,
    *,
    model: str = "yolov8m",
    device: str = "auto",
    half: bool = False,
    batch_size: int = 1,
    models_dir: Path = DEFAULT_MODELS_DIR,
    trt_options: dict | None = None,
    trt_prebuild: bool = False,
    trt_cache_dir: str = TRT_CACHE_DIR,
    run=subprocess.run,
) -> Runtime:
    """The runtime the worker's RUNTIME setting names, warmed up for batch_size.

    For onnx-trt with trt_prebuild, the engine is built in a subprocess first, and this
    process loads it from trt_cache_dir; build_info says whether it did (engine_from_cache).
    """
    if kind == "pytorch":
        runtime: Runtime = TorchRuntime(resolve_weights(model, models_dir), device, half)
        runtime.warmup(batch_size)
        return runtime
    if kind not in ("onnx-cuda", "onnx-trt"):
        raise ValueError(f"unknown runtime {kind!r}")
    provider = "cuda" if kind == "onnx-cuda" else "trt"
    path = onnx_model_path(model, half, provider, models_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing: the ONNX files are built into the GPU image "
            "(locally: make onnx-local)"
        )
    info: dict[str, object] = {}
    if provider == "trt":
        info["trt_cache_dir"] = trt_cache_dir
        info["prebuild_s"] = (
            prebuild_trt(
                prebuild_command(
                    path,
                    half=half,
                    batch_size=batch_size,
                    cache_dir=trt_cache_dir,
                    trt_options=trt_options,
                ),
                run=run,
            )
            if trt_prebuild
            else None
        )
    before = cache_snapshot(trt_cache_dir) if provider == "trt" else {}
    onnx = OnnxRuntime(
        path,
        provider,
        half=half,
        batch_size=batch_size,
        trt_options=trt_options,
        cache_dir=trt_cache_dir,
    )
    onnx.warmup(batch_size)
    if provider == "trt":
        built = built_files(before, cache_snapshot(trt_cache_dir))
        info["engine_from_cache"] = not built
        info["engines_built_in_process"] = built
    onnx.build_info = info
    return onnx
