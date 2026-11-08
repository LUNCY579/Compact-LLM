"""Fail-closed, read-only two-lane scheduling. Never launches a subprocess/model.

An admitted wave is a resource plan, not native qualification or execution authority.
Running jobs consume capacity; one GPU job maximum across both lanes.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

STATUSES = {"pending", "running", "complete", "blocked", "failed"}


def validate(tasks: list[dict]) -> dict[str, dict]:
    by_id = {t["task_id"]: t for t in tasks}
    if len(by_id) != len(tasks):
        raise ValueError("duplicate task IDs")
    for task in tasks:
        if task["status"] not in STATUSES:
            raise ValueError("invalid task status")
        if task["lane"] not in {"FAST", "MEDIUM", "HEAVY"}:
            raise ValueError("invalid lane")
        for field in ("cpu_cores", "ram_gib", "gpu_gib", "expected_seconds"):
            value = task[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"invalid {field}")
        if task["expected_seconds"] <= 0 or task["cpu_cores"] < 1:
            raise ValueError("duration and CPU allocation must be positive")
        if task["lane"] == "FAST" and task["expected_seconds"] > 1800:
            raise ValueError("FAST task exceeds 30 minutes")
        for dependency in task["dependency"]:
            if dependency not in by_id:
                raise ValueError(f"unknown dependency {dependency}")
        if task["status"] == "complete" and not task.get("evidence_receipt"):
            raise ValueError("completion requires evidence receipt")
    visited, active = set(), set()

    def visit(key):
        if key in active:
            raise ValueError("dependency cycle")
        if key in visited:
            return
        active.add(key)
        for dep in by_id[key]["dependency"]:
            visit(dep)
        active.remove(key)
        visited.add(key)

    for key in by_id:
        visit(key)
    return by_id


def plan(queue: dict, available: dict) -> dict:
    tasks = queue["tasks"]
    by_id = validate(tasks)
    floors = queue["resource_policy"]
    # These inherited PRLC minima cannot be weakened by editing a queue.
    minima = {"host_free_gib": 7.4, "worker_free_gib": 6.0, "gpu_free_gib": 4.0, "disk_free_gib": 4.0}
    if any(floors.get(k, 0) < v for k, v in minima.items()):
        raise ValueError("resource floor below inherited minimum")
    for key in ("cpu_cores", "ram_free_gib", "gpu_free_gib", "disk_free_gib"):
        value = available.get(key)
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"missing/invalid resource measurement: {key}")
    cpu = available["cpu_cores"]
    # Measurements are current free memory; running-job usage is already excluded.
    ram = available["ram_free_gib"]
    gpu = available["gpu_free_gib"]
    running = [t for t in tasks if t["status"] == "running"]
    cpu -= sum(t["cpu_cores"] for t in running)
    gpu_busy = any(t["gpu_gib"] > 0 for t in running)
    if sum(t["gpu_gib"] > 0 for t in running) > 1:
        raise ValueError("multiple GPU workloads already marked running")
    chosen, held = [], {}
    # Fast preparation always accompanies an independent heavy admission if it fits.
    order = sorted(tasks, key=lambda t: (t["lane"] != "FAST", t.get("priority", 100), t["task_id"]))
    for task in order:
        key = task["task_id"]
        if task["status"] != "pending":
            continue
        reasons = []
        if task.get("blocker"):
            reasons.append(task["blocker"])
        if any(by_id[dep]["status"] != "complete" for dep in task["dependency"]):
            reasons.append("dependency incomplete")
        if task["cpu_cores"] > cpu:
            reasons.append("CPU reservation")
        if task["ram_gib"] > ram:
            reasons.append("RAM reservation")
        if task["gpu_gib"]:
            if gpu_busy:
                reasons.append("one GPU workload limit")
            if not task.get("native_gate_receipt"):
                reasons.append("missing scoped native gate receipt")
            if ram < floors["host_free_gib"] or ram - task["ram_gib"] < floors["worker_free_gib"]:
                reasons.append("inherited RAM admission/worker floor")
            if gpu < max(task["gpu_gib"], floors["gpu_free_gib"]):
                reasons.append("GPU admission floor")
            if available["disk_free_gib"] < floors["disk_free_gib"]:
                reasons.append("disk admission floor")
        if reasons:
            held[key] = reasons
            continue
        chosen.append(key)
        cpu -= task["cpu_cores"]
        ram -= task["ram_gib"]
        if task["gpu_gib"]:
            gpu -= task["gpu_gib"]
            gpu_busy = True
    return {"schema": 1, "mode": "PLAN_ONLY_NO_LAUNCH", "admitted_wave": chosen, "held": held,
            "running": [t["task_id"] for t in running], "remaining_cpu": cpu, "remaining_ram_gib": ram,
            "note": "Recheck identities, receipts, clean source and live guards immediately before native launch."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(plan(json.loads(args.queue.read_text()), json.loads(args.resources.read_text())), indent=2))


if __name__ == "__main__":
    main()
