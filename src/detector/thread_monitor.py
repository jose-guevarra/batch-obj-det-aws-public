"""Per-thread CPU use from /proc (Linux), for finding what competes with image loading.

take() returns each thread name's CPU use since the previous take(), as a percentage of one
core, plus the whole process. The worker calls it once per batch when THREAD_PROFILE=1, so
every batch log line shows where the instance's vCPUs went during that batch. Threads with
the same name are summed, so the loader pool shows as one entry.

Python 3.14 gives OS threads their Python names (truncated to 15 characters), so the
worker's own threads are recognizable: prepare, finish, ThreadPoolExecutor-*, and the main
thread as python. ONNX Runtime, TensorRT and CUDA threads appear under their own names.

Off Linux (no /proc/self/task) start() returns False and take() returns None.
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

TICKS_PER_S = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def _read_stat(text: str) -> int:
    """utime + stime (clock ticks) from a /proc/<pid>/task/<tid>/stat line."""
    # The thread name is in parentheses and may contain spaces; fields follow the last ")".
    fields = text[text.rindex(")") + 2 :].split()
    return int(fields[11]) + int(fields[12])  # utime, stime (fields 14 and 15 of stat)


class ThreadMonitor:
    def __init__(
        self,
        task_dir: str | Path = "/proc/self/task",
        top: int = 8,
        clock: Callable[[], float] = time.monotonic,
        ticks_per_s: int = TICKS_PER_S,
    ) -> None:
        self.task_dir = Path(task_dir)
        self.top = top
        self.clock = clock
        self.ticks_per_s = ticks_per_s
        self._last: dict[str, tuple[str, int]] | None = None  # tid -> (name, ticks)
        self._t = 0.0

    def start(self) -> bool:
        if not self.task_dir.is_dir():
            return False
        self._last, self._t = self._read(), self.clock()
        return True

    @property
    def active(self) -> bool:
        return self._last is not None

    def take(self) -> dict | None:
        """{"process_pct": ..., "by_name": {name: pct}} since the last take(); resets."""
        if self._last is None:
            return None
        now, t = self._read(), self.clock()
        elapsed = t - self._t
        if elapsed <= 0:
            return None
        by_name: dict[str, float] = defaultdict(float)
        for tid, (name, ticks) in now.items():
            before = self._last.get(tid, (name, 0))[1]  # a new thread started from 0
            by_name[name] += 100 * (ticks - before) / self.ticks_per_s / elapsed
        self._last, self._t = now, t
        top = sorted(by_name.items(), key=lambda kv: -kv[1])[: self.top]
        return {
            "process_pct": round(sum(by_name.values()), 1),
            "by_name": {name: round(pct, 1) for name, pct in top if pct >= 0.5},
        }

    def _read(self) -> dict[str, tuple[str, int]]:
        threads = {}
        for entry in self.task_dir.iterdir():
            try:
                name = (entry / "comm").read_text().strip()
                threads[entry.name] = (name, _read_stat((entry / "stat").read_text()))
            except OSError, ValueError, IndexError:
                continue  # the thread exited while we read it
        return threads
