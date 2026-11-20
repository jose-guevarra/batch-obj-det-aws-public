from types import SimpleNamespace

from detector.gpu_monitor import GpuMonitor


class FakeNvml:
    def __init__(self, utils, fail_init=False):
        self.utils = list(utils)
        self.fail_init = fail_init

    def nvmlInit(self):
        if self.fail_init:
            raise RuntimeError("NVML Shared Library Not Found")

    def nvmlDeviceGetHandleByIndex(self, index):
        return f"gpu{index}"

    def nvmlDeviceGetUtilizationRates(self, handle):
        return SimpleNamespace(gpu=self.utils.pop(0))

    def nvmlDeviceGetMemoryInfo(self, handle):
        return SimpleNamespace(used=2048 * 2**20)

    def nvmlSystemGetDriverVersion(self):
        return b"580.95.05"

    def nvmlDeviceGetName(self, handle):
        return "Tesla T4"


def test_take_returns_mean_since_last_take_and_resets():
    monitor = GpuMonitor(interval_s=3600, nvml=FakeNvml([80, 100, 40]))
    assert monitor.start()
    try:
        monitor.sample()
        monitor.sample()
        assert monitor.take() == {"util_mean": 90.0, "samples": 2, "mem_mb": 2048}
        assert monitor.take() is None  # nothing sampled since
        monitor.sample()
        assert monitor.take()["util_mean"] == 40.0
        assert monitor.info() == {"driver": "580.95.05", "gpu": "Tesla T4"}
    finally:
        monitor.stop()


def test_without_nvml_monitoring_is_off_and_harmless():
    monitor = GpuMonitor(nvml=FakeNvml([], fail_init=True))
    assert monitor.start() is False
    assert not monitor.active
    assert monitor.take() is None
    assert monitor.info() == {}
    monitor.stop()
