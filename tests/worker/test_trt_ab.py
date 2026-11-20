import importlib.util
import sys
from pathlib import Path

SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark.py"


def load_script():
    spec = importlib.util.spec_from_file_location("benchmark_script", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up while it loads
    spec.loader.exec_module(module)
    return module


def test_ab_specs_isolate_the_two_tensorrt_runs():
    bench = load_script()
    a, b, c = bench.trt_ab_specs("20261112-120000", images=500, batch_size=32)
    env_a, env_b, env_c = (s.environment() for s in (a, b, c))
    assert (a.runtime, b.runtime, c.runtime) == ("onnx-trt", "onnx-trt", "pytorch")
    assert all(s.smoke and s.half and s.pipelined and s.images == 500 for s in (a, b, c))
    assert (env_a["TRT_PREBUILD"], env_b["TRT_PREBUILD"]) == ("0", "1")
    # Fresh, distinct caches inside the shared volume: both runs build from nothing.
    assert env_a["TRT_CACHE_DIR"] != env_b["TRT_CACHE_DIR"]
    for env in (env_a, env_b):
        assert env["TRT_CACHE_DIR"].startswith("/tmp/trt-cache/ab-20261112-120000")
    assert "TRT_CACHE_DIR" not in env_c
    assert all(
        e["THREAD_PROFILE"] == "1" and e["BATCH_SIZE"] == "32" for e in (env_a, env_b, env_c)
    )


def test_print_ab(capsys):
    bench = load_script()
    record = {
        "warmup_s": 451.0,
        "startup": {"prebuild_s": 440.2, "engine_from_cache": True},
        "metrics": {
            "images_per_s": 77.6,
            "gpu_util_mean": 61.0,
            "stage_ms_per_image": {"load": 5.9, "wait_input": 1.0, "inference": 5.4},
        },
    }
    bench.print_ab(bench.trt_ab_specs("x")[:1], [record])
    out = capsys.readouterr().out
    assert "77.6" in out and "True" in out
