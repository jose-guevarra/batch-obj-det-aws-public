from detector.thread_monitor import ThreadMonitor, _read_stat


def stat(utime: int, stime: int, comm: str = "x") -> str:
    # pid (comm) state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
    return f"123 ({comm}) S 1 1 1 0 -1 0 0 0 0 0 {utime} {stime} 0 0 20 0 1 0\n"


def write_thread(task_dir, tid, name, utime, stime):
    d = task_dir / str(tid)
    d.mkdir(exist_ok=True)
    (d / "comm").write_text(name + "\n")
    (d / "stat").write_text(stat(utime, stime, name))


def test_read_stat_handles_names_with_spaces_and_parens():
    assert _read_stat(stat(7, 3, comm="cuda (evt) handler")) == 10


def test_cpu_percent_per_thread_name(tmp_path):
    clock = iter([0.0, 2.0, 4.0])
    write_thread(tmp_path, 1, "python", 0, 0)
    write_thread(tmp_path, 2, "ThreadPoolExecu", 0, 0)
    write_thread(tmp_path, 3, "ThreadPoolExecu", 0, 0)
    monitor = ThreadMonitor(tmp_path, clock=lambda: next(clock), ticks_per_s=100)
    assert monitor.start()

    # 2 s later: main thread used 1 s (50%), the two pool threads 1.5 s each (150% together).
    write_thread(tmp_path, 1, "python", 80, 20)
    write_thread(tmp_path, 2, "ThreadPoolExecu", 100, 50)
    write_thread(tmp_path, 3, "ThreadPoolExecu", 150, 0)
    write_thread(tmp_path, 4, "onnxruntime", 200, 0)  # started during the window: 100%
    window = monitor.take()
    assert window == {
        "process_pct": 300.0,
        "by_name": {"ThreadPoolExecu": 150.0, "onnxruntime": 100.0, "python": 50.0},
    }

    # The next window counts only what happened since.
    write_thread(tmp_path, 1, "python", 80, 30)
    assert monitor.take()["by_name"] == {"python": 5.0}


def test_without_proc_it_is_off(tmp_path):
    monitor = ThreadMonitor(tmp_path / "missing")
    assert monitor.start() is False
    assert monitor.take() is None


def test_worker_logs_thread_window(env):
    from detector.worker import Worker

    class FakeThreads:
        def take(self):
            return {"process_pct": 250.0, "by_name": {"prepare": 90.0}}

    env.store.create_batch("b1", 2, "yolov8m")
    env.send("b1", env.upload("b1", 2))
    worker = Worker(
        env.config(), env.runtime, env.sqs, env.s3, env.store, thread_monitor=FakeThreads()
    )
    assert worker.run_once().threads == {"process_pct": 250.0, "by_name": {"prepare": 90.0}}
    assert env.worker().run_once() is None  # no monitor, nothing to log
