"""GPU utilization sampling with NVML (the counters nvidia-smi reports), in-process.

A background thread samples the GPU every interval_s; take() returns the mean since the
previous take() and starts a new window. The worker calls it once per batch, so each
batch log line carries the utilization over that batch.

On a machine without an NVIDIA GPU (or without the driver), start() returns False and
take() returns None: the worker logs no GPU fields and keeps running.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

log = logging.getLogger("detector.gpu_monitor")


class GpuMonitor:
    def __init__(self, index: int = 0, interval_s: float = 0.5, nvml: Any = None) -> None:
        self.index = index
        self.interval_s = interval_s
        self._nvml = nvml  # injectable for tests; pynvml by default
        self._handle: Any = None
        self._lock = threading.Lock()
        self._util_sum = 0.0
        self._samples = 0
        self._mem_mb = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> bool:
        """Begin sampling. False (and no thread) if NVML isn't available."""
        try:
            if self._nvml is None:
                import pynvml

                self._nvml = pynvml
            self._nvml.nvmlInit()
            self._handle = self._nvml.nvmlDeviceGetHandleByIndex(self.index)
        except Exception as e:  # no GPU, no driver, or no pynvml
            log.info("GPU monitoring off: %s", e)
            self._handle = None
            return False
        self._thread = threading.Thread(target=self._run, name="gpu-monitor", daemon=True)
        self._thread.start()
        return True

    @property
    def active(self) -> bool:
        return self._handle is not None

    def info(self) -> dict[str, str]:
        """Driver version and GPU name, for the worker's startup log."""
        if not self.active:
            return {}
        return {
            "driver": _text(self._nvml.nvmlSystemGetDriverVersion()),
            "gpu": _text(self._nvml.nvmlDeviceGetName(self._handle)),
        }

    def sample(self) -> None:
        """Take one sample now (the thread calls this every interval_s)."""
        util = self._nvml.nvmlDeviceGetUtilizationRates(self._handle).gpu
        mem = self._nvml.nvmlDeviceGetMemoryInfo(self._handle).used / 2**20
        with self._lock:
            self._util_sum += util
            self._samples += 1
            self._mem_mb = max(self._mem_mb, mem)

    def take(self) -> dict[str, float] | None:
        """Mean utilization (%) and peak memory (MiB) since the last take(); resets the window."""
        if not self.active:
            return None
        with self._lock:
            samples, util_sum, mem = self._samples, self._util_sum, self._mem_mb
            self._samples, self._util_sum, self._mem_mb = 0, 0.0, 0.0
        if not samples:
            return None
        return {"util_mean": round(util_sum / samples, 1), "samples": samples, "mem_mb": round(mem)}

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def _run(self) -> None:
        while not self._stop.wait(self.interval_s):
            try:
                self.sample()
            except Exception as e:
                log.warning("GPU sample failed: %s", e)


def _text(value: str | bytes) -> str:
    return value.decode() if isinstance(value, bytes) else value
