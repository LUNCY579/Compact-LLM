import pytest

from experiments.reasoning_posttraining_v1.resource_wait import ResourceWait


class Clock:
    def __init__(self):
        self.now = 0.

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def test_no_model_work_until_the_unchanged_guard_passes(tmp_path):
    clock, calls, operations = Clock(), [], []
    def check(**kwargs):
        calls.append(kwargs)
        if clock.now < 4:
            raise RuntimeError("host RAM guard refused; need 1.5 GiB, have 1.2 GiB")
        return {"available_host_bytes": 2 * 2**30}
    gate = ResourceWait(check, tmp_path / "wait", total_seconds=10, episode_seconds=8,
                        poll_seconds=2, clock=clock, sleep=clock.sleep)
    value = gate(before_load=False, phase="qualify")
    operations.append("one model operation")
    assert len(calls) == 3 and all(c == calls[0] for c in calls)
    assert operations == ["one model operation"] and value["available_host_bytes"] == 2 * 2**30
    assert gate.summary()["wait_seconds"] == 4
    assert gate.summary()["model_operation_retries"] == 0


def test_exhausted_wait_never_runs_the_dependent_operation(tmp_path):
    clock, operations = Clock(), []
    def check(**kw):
        raise RuntimeError("free GPU memory guard refused")
    gate = ResourceWait(check, tmp_path / "wait", total_seconds=8, episode_seconds=4,
                        poll_seconds=2, clock=clock, sleep=clock.sleep)
    with pytest.raises(RuntimeError, match="wait exhausted"):
        gate()
        operations.append("must not run")
    assert operations == [] and clock.now == 4


def test_wait_limit_is_shared_across_episodes(tmp_path):
    clock = Clock()
    blocked_until = [2]
    def check(**kw):
        if clock.now < blocked_until[0]:
            raise RuntimeError("host RAM guard refused")
        return {}
    gate = ResourceWait(check, tmp_path / "wait", total_seconds=5, episode_seconds=4,
                        poll_seconds=1, clock=clock, sleep=clock.sleep)
    gate()
    blocked_until[0] = 20
    with pytest.raises(RuntimeError, match="wait exhausted"):
        gate()
    assert gate.summary()["wait_seconds"] == 5


def test_nonresource_failures_propagate_without_waiting(tmp_path):
    clock = Clock()
    def check(**kw):
        raise RuntimeError("CUDA unavailable")
    gate = ResourceWait(check, tmp_path / "wait", clock=clock, sleep=clock.sleep)
    with pytest.raises(RuntimeError, match="CUDA unavailable"):
        gate()
    assert clock.now == 0


def test_oversleep_is_recorded_and_never_authorizes_late_work(tmp_path):
    clock = Clock()
    def check(**kw):
        if clock.now == 0:
            raise RuntimeError("host RAM guard refused")
        return {}
    def delayed_sleep(_seconds):
        clock.now += 10
    gate = ResourceWait(check, tmp_path / "wait", total_seconds=5, episode_seconds=4,
                        poll_seconds=1, clock=clock, sleep=delayed_sleep)
    with pytest.raises(RuntimeError, match="wait exhausted"):
        gate()
    assert gate.summary()["wait_seconds"] == 10, "actual overrun must not be clamped away"
