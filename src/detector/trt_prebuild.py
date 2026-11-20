"""Build the TensorRT engine for the worker in a separate, short-lived process.

    python -m detector.trt_prebuild --model /app/models/yolov8m.onnx --batch-size 32 \\
        --half --cache-dir /tmp/trt-cache

Creates the same ONNX Runtime TensorRT session the worker will (detector.runtimes:
onnx_providers), warms it up at both ends of the batch profile so the engine and timing
cache are written to --cache-dir, prints one JSON line and exits. The worker then loads
the engine from the cache. With an engine already in the cache, this just loads it.
"""

from __future__ import annotations

import argparse
import json
import time

from detector.runtimes import OnnxRuntime, cache_snapshot, parse_trt_options


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, required=True)
    p.add_argument("--half", action="store_true")
    p.add_argument("--cache-dir", required=True)
    p.add_argument("--options", default="", help="JSON TensorRT provider options")
    args = p.parse_args(argv)

    started = time.monotonic()
    runtime = OnnxRuntime(
        args.model,
        "trt",
        half=args.half,
        batch_size=args.batch_size,
        trt_options=parse_trt_options(args.options),
        cache_dir=args.cache_dir,
    )
    runtime.warmup(args.batch_size)
    print(
        json.dumps(
            {
                "event": "trt_prebuild",
                "build_s": round(time.monotonic() - started, 1),
                "cache_files": sorted(cache_snapshot(args.cache_dir)),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
