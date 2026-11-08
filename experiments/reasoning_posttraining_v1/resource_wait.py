"""Bounded backpressure at the unchanged resource gate, never a model-operation retry.

Only a registered worker can opt in. The wrapped guard alone decides whether work
may run. Transient RAM/VRAM refusal pauses before the next operation; other failures
propagate. Waiting counts against the foreground watchdog and journal wall budget.
"""
from __future__ import annotations

from pathlib import Path
import time

from experiments.model_quality_gpu_v1.integrity import new_dir, write_new


class ResourceWait:
    def __init__(self, check, directory: Path, *, total_seconds=240, episode_seconds=90,
                 poll_seconds=2, clock=time.monotonic, sleep=time.sleep):
        if not 0 < poll_seconds <= episode_seconds <= total_seconds:
            raise ValueError("invalid bounded resource-wait policy")
        self.check, self.clock, self.sleep = check, clock, sleep
        self.directory = new_dir(Path(directory))
        self.total_limit, self.episode_limit, self.poll = total_seconds, episode_seconds, poll_seconds
        self.waited = 0.0
        self.episodes = 0

    def __call__(self, **kwargs):
        start = None
        checks = []
        first_refusal = None
        while True:
            now = self.clock()
            elapsed = 0.0 if start is None else now - start
            if start is not None and (elapsed >= self.episode_limit or self.waited + elapsed >= self.total_limit):
                self._finish(elapsed, checks, "EXHAUSTED", kwargs)
                raise RuntimeError("bounded resource wait exhausted; no model operation was retried") from first_refusal
            try:
                value = self.check(**kwargs)
            except RuntimeError as exc:
                if not str(exc).startswith(("host RAM guard refused", "free GPU memory guard refused")):
                    raise
                if start is None:
                    start, first_refusal = self.clock(), exc
                elapsed = self.clock() - start
                checks.append({"elapsed_seconds": elapsed, "refusal": str(exc)})
                # Evidence is written before sleeping, including if the watchdog later kills us.
                write_new(self.directory / f"episode-{self.episodes:03d}-check-{len(checks):03d}.json",
                          {"phase_arguments": kwargs, **checks[-1], "model_work_permitted": False})
                remaining = min(self.episode_limit - elapsed, self.total_limit - self.waited - elapsed)
                if remaining <= 0:
                    continue
                self.sleep(min(self.poll, remaining))
                continue
            if start is not None:
                elapsed = self.clock() - start
                if elapsed >= self.episode_limit or self.waited + elapsed >= self.total_limit:
                    self._finish(elapsed, checks, "EXHAUSTED", kwargs)
                    raise RuntimeError("bounded resource wait exhausted before model work could resume") from first_refusal
                self._finish(elapsed, checks, "GUARD_PASSED", kwargs)
                first_refusal = None
            return value

    def _finish(self, elapsed, checks, outcome, arguments):
        self.waited += elapsed
        write_new(self.directory / f"episode-{self.episodes:03d}-result.json",
                  {"wait_seconds": elapsed, "total_wait_seconds": self.waited,
                   "checks": len(checks), "outcome": outcome, "phase_arguments": arguments,
                   "guard_thresholds_changed": False})
        self.episodes += 1

    def summary(self):
        return {"episodes": self.episodes, "wait_seconds": self.waited,
                "total_limit_seconds": self.total_limit, "episode_limit_seconds": self.episode_limit,
                "guard_thresholds_changed": False, "model_operation_retries": 0}
