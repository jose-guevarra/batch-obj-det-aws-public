"""Runtime selection and ONNX Runtime options (no GPU needed)."""

from pathlib import Path

import pytest

from detector.runtimes import make_runtime, onnx_model_path, onnx_providers


def test_tensorrt_provider_options():
    (trt_name, trt), (cuda_name, _) = onnx_providers("trt", half=True, batch_size=64)
    assert (trt_name, cuda_name) == ("TensorrtExecutionProvider", "CUDAExecutionProvider")
    assert trt["trt_fp16_enable"] is True
    assert trt["trt_engine_cache_enable"] and trt["trt_engine_cache_path"] == "/tmp/trt-cache"
    # One engine covers batch 1 (a partial last batch) up to the configured batch size.
    assert trt["trt_profile_min_shapes"] == "images:1x3x640x640"
    assert trt["trt_profile_opt_shapes"] == trt["trt_profile_max_shapes"] == "images:64x3x640x640"
    assert trt["trt_cuda_graph_enable"] is True  # see OnnxRuntime._infer_static
    assert onnx_providers("trt", half=False, batch_size=8)[0][1]["trt_fp16_enable"] is False


def test_cuda_and_cpu_providers():
    assert onnx_providers("cuda", half=True, batch_size=8) == [
        ("CUDAExecutionProvider", {"device_id": 0})
    ]
    assert onnx_providers("cpu", half=False, batch_size=1) == [("CPUExecutionProvider", {})]
    with pytest.raises(ValueError):
        onnx_providers("rocm", half=False, batch_size=1)


@pytest.mark.parametrize(
    ("half", "provider", "expected"),
    [(False, "cuda", "yolov8m.onnx"), (True, "cuda", "yolov8m-fp16.onnx"),
     (True, "trt", "yolov8m.onnx"), (False, "trt", "yolov8m.onnx")],
)  # fmt: skip
def test_onnx_model_file(half, provider, expected):
    # TensorRT makes its own FP16 engine from the FP32 file; the CUDA backend needs FP16 weights.
    assert onnx_model_path("yolov8m", half, provider, Path("/m")) == Path("/m") / expected


def test_make_runtime_errors(tmp_path):
    with pytest.raises(ValueError, match="unknown runtime"):
        make_runtime("tensorflow", models_dir=tmp_path)
    with pytest.raises(FileNotFoundError, match="make onnx-local"):
        make_runtime("onnx-trt", half=True, models_dir=tmp_path)


def test_gpu_sessions_use_one_quiet_cpu_thread():
    import onnxruntime as ort

    from detector.runtimes import onnx_session_options

    gpu = onnx_session_options(ort, "trt")
    assert (gpu.intra_op_num_threads, gpu.inter_op_num_threads) == (1, 1)
    assert gpu.get_session_config_entry("session.intra_op.allow_spinning") == "0"
    cpu = onnx_session_options(ort, "cpu")  # local parity tests keep ONNX Runtime's defaults
    assert cpu.intra_op_num_threads == 0


def test_trt_options_override_defaults():
    from detector.runtimes import parse_trt_options

    extra = parse_trt_options(
        '{"trt_cuda_graph_enable": true, "trt_builder_optimization_level": 2}'
    )
    (_, trt), _ = onnx_providers("trt", half=True, batch_size=32, trt_options=extra)
    assert trt["trt_cuda_graph_enable"] is True and trt["trt_builder_optimization_level"] == 2
    assert trt["trt_fp16_enable"] is True  # defaults stay unless overridden
    assert parse_trt_options("") == {}
    for bad in ('["trt_fp16_enable"]', '{"fp16": true}', '{"trt_x": [1]}'):
        with pytest.raises(ValueError):
            parse_trt_options(bad)


def test_stack_mounts_the_cache_where_the_runtime_writes():
    from detector.runtimes import TRT_CACHE_DIR
    from stacks.detector_batch import TRT_CACHE_DIR as mounted

    assert mounted == TRT_CACHE_DIR


def test_output_shape_for_640_input():
    from detector.runtimes import OUT_SHAPE

    assert OUT_SHAPE == (84, 8400)


# TensorRT prebuild in a subprocess ---------------------------------------------------


class FakeTrt:
    """Stands in for OnnxRuntime: writes an engine on warmup unless one is cached."""

    instances: list[FakeTrt] = []

    def __init__(self, path, provider, *, half, batch_size, trt_options, cache_dir):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)  # as OnnxRuntime does
        self.kwargs = dict(provider=provider, half=half, batch_size=batch_size, cache_dir=cache_dir)
        self.build_info = {}
        FakeTrt.instances.append(self)

    def warmup(self, batch_size):
        engine = self.cache_dir / "model_fp16_sm75.engine"
        if not engine.exists():
            engine.write_bytes(b"engine")
        (self.cache_dir / "timing.cache").write_bytes(b"rewritten every time")


@pytest.fixture
def fake_trt(tmp_path, monkeypatch):
    from detector import runtimes

    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "yolov8m.onnx").write_bytes(b"onnx")
    monkeypatch.setattr(runtimes, "OnnxRuntime", FakeTrt)
    FakeTrt.instances = []
    return tmp_path


def test_prebuild_runs_in_a_subprocess_then_loads_from_the_cache(fake_trt):
    cache = fake_trt / "cache"
    cache.mkdir()
    calls = []

    def fake_run(cmd, check, timeout):
        calls.append(cmd)
        (cache / "model_fp16_sm75.engine").write_bytes(b"built by the subprocess")

    rt = make_runtime(
        "onnx-trt", half=True, batch_size=32, models_dir=fake_trt / "models",
        trt_prebuild=True, trt_cache_dir=str(cache), run=fake_run,
        trt_options={"trt_builder_optimization_level": 3},
    )  # fmt: skip
    (cmd,) = calls
    assert cmd[1:3] == ["-m", "detector.trt_prebuild"]
    assert cmd[cmd.index("--batch-size") + 1] == "32"
    assert cmd[cmd.index("--cache-dir") + 1] == str(cache)
    assert "--half" in cmd
    assert '"trt_builder_optimization_level": 3' in cmd[cmd.index("--options") + 1]
    assert rt.kwargs["cache_dir"] == str(cache)
    assert rt.build_info["engine_from_cache"] is True
    assert rt.build_info["engines_built_in_process"] == []
    assert isinstance(rt.build_info["prebuild_s"], float)


def test_without_prebuild_the_engine_is_built_in_process(fake_trt):
    cache = fake_trt / "cache"
    rt = make_runtime(
        "onnx-trt", half=True, batch_size=32, models_dir=fake_trt / "models",
        trt_prebuild=False, trt_cache_dir=str(cache), run=None,
    )  # fmt: skip
    assert rt.build_info == {
        "trt_cache_dir": str(cache),
        "prebuild_s": None,
        "engine_from_cache": False,
        "engines_built_in_process": ["model_fp16_sm75.engine"],
    }


def test_built_files_ignores_the_timing_cache_and_unchanged_engines():
    from detector.runtimes import built_files

    before = {"a.engine": (10, 1), "timing.cache": (5, 1)}
    assert built_files(before, {"a.engine": (10, 1), "timing.cache": (6, 2)}) == []
    assert built_files(before, {"a.engine": (11, 2)}) == ["a.engine"]
    assert built_files({}, {"b.engine": (1, 1)}) == ["b.engine"]


def test_prebuild_failure_stops_startup(fake_trt):
    import subprocess

    def failing(cmd, check, timeout):
        raise subprocess.CalledProcessError(1, cmd)

    with pytest.raises(subprocess.CalledProcessError):
        make_runtime(
            "onnx-trt", half=True, batch_size=8, models_dir=fake_trt / "models",
            trt_prebuild=True, trt_cache_dir=str(fake_trt / "c"), run=failing,
        )  # fmt: skip


def test_worker_default_cache_dir_matches_the_runtime():
    from detector.runtimes import TRT_CACHE_DIR
    from detector.worker import WorkerConfig

    assert WorkerConfig(queue_url="q", dlq_url="d", bucket="b", table="t").trt_cache_dir == (
        TRT_CACHE_DIR
    )
