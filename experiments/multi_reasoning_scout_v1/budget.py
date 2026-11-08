"""No-torch preload guards and conservative finite inference budget."""
from __future__ import annotations
import importlib.metadata
import math
import shutil
import subprocess
import time
from .core import RECORDS

GIB = 2 ** 30
LIMITS = {"preload_host_gib": 6, "preload_gpu_gib": 6, "minimum_disk_gib": 8,
          "running_host_gib": 1.5, "running_gpu_free_gib": .5,
          "max_gpu_allocated_gib": 6, "max_host_rss_gib": 6,
          "max_disk_bytes_per_model": 128 * 2**20,
          "max_seconds_per_model": 1800, "max_calls_per_model": 1322,
          "max_generated_tokens_per_model": 115232, "max_models": 2,
          "batch_size": 1, "model_loads_per_model": 1,
          "training_steps": 0, "retries": 0, "cloud_spend": 0}
PINS = {"torch": "2.10.0", "transformers": "4.57.1", "safetensors": "0.7.0"}


def runtime_versions():
    return {n: importlib.metadata.version(n) for n in (*PINS, "tokenizers", "numpy", "psutil")}


def snapshot(native=False):
    import psutil
    output = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,memory.free,memory.total,utilization.gpu", "--format=csv,noheader,nounits"
    ], text=True, timeout=10).strip().splitlines()
    if len(output) != 1:
        raise RuntimeError("exactly one visible GPU required")
    name, free, total, util = [x.strip() for x in output[0].split(",")]
    result = {"time_unix": time.time(), "host_available_bytes": psutil.virtual_memory().available,
              "host_rss_bytes": psutil.Process().memory_info().rss,
              "disk_free_bytes": shutil.disk_usage(RECORDS).free, "gpu_name": name,
              "gpu_free_bytes": int(free) * 2**20, "gpu_total_bytes": int(total) * 2**20,
              "gpu_utilization_percent": int(util), "gpu_allocated_bytes": 0, "gpu_peak_bytes": 0}
    if native:
        import torch
        free_bytes, _ = torch.cuda.mem_get_info()
        result.update(gpu_free_bytes=free_bytes, gpu_allocated_bytes=torch.cuda.memory_allocated(),
                      gpu_peak_bytes=torch.cuda.max_memory_allocated())
    return result


def check(value, *, preload=False):
    required = ("host_available_bytes", "gpu_free_bytes", "disk_free_bytes", "host_rss_bytes", "gpu_allocated_bytes", "gpu_peak_bytes")
    if any(type(value.get(k)) not in (int, float) or not math.isfinite(value[k]) or value[k] < 0 for k in required):
        raise RuntimeError("RESOURCE_GUARD: missing or nonfinite measurement")
    floors = {"host_available_bytes": LIMITS["preload_host_gib" if preload else "running_host_gib"] * GIB,
              "gpu_free_bytes": LIMITS["preload_gpu_gib" if preload else "running_gpu_free_gib"] * GIB,
              "disk_free_bytes": LIMITS["minimum_disk_gib"] * GIB}
    failures = [f"{k}: {value[k]/GIB:.3f} GiB < {v/GIB:.3f} GiB" for k, v in floors.items() if value[k] < v]
    for key, limit in (("gpu_allocated_bytes", "max_gpu_allocated_gib"), ("gpu_peak_bytes", "max_gpu_allocated_gib"), ("host_rss_bytes", "max_host_rss_gib")):
        if value[key] > LIMITS[limit] * GIB:
            failures.append(f"{key} exceeds {limit}")
    if failures:
        raise RuntimeError("RESOURCE_GUARD: " + "; ".join(failures))
    return value


class Meter:
    def __init__(self, out, sample=snapshot):
        self.out, self.sample = out, sample
        self.started = time.monotonic()
        self.calls = self.allowance = 0
        self.peak_rss = self.peak_gpu = 0
        self.last_check = 0

    def guard(self, *, force=False):
        now = time.monotonic()
        if now - self.started > LIMITS["max_seconds_per_model"]:
            raise RuntimeError("wall-time ceiling")
        if force or now - self.last_check >= 1:
            value = check(self.sample(native=True))
            self.last_check = now
            self.peak_rss = max(self.peak_rss, value["host_rss_bytes"])
            self.peak_gpu = max(self.peak_gpu, value["gpu_peak_bytes"])
            size = sum(p.stat().st_size for p in self.out.rglob("*") if p.is_file())
            if size > LIMITS["max_disk_bytes_per_model"]:
                raise RuntimeError("output disk ceiling")

    def reserve(self, tokens):
        if type(tokens) is not int or tokens <= 0:
            raise ValueError("positive integer token allowance required")
        if self.calls + 1 > LIMITS["max_calls_per_model"] or self.allowance + tokens > LIMITS["max_generated_tokens_per_model"]:
            raise RuntimeError("inference budget exhausted")
        self.calls += 1
        self.allowance += tokens
        self.guard(force=True)

    def report(self):
        return {"generation_calls": self.calls, "reserved_generated_tokens": self.allowance,
                "wall_seconds": time.monotonic() - self.started,
                "peak_host_rss_bytes_sampled": self.peak_rss, "peak_gpu_allocated_bytes": self.peak_gpu}


def estimate():
    return {"models": 2, "model_loads": 2, "cases_per_model": 120,
            "scored_CHECK_cases": 96, "SMOKE": 8, "CAL": 16,
            "unique_calls_per_item": 11, "primary_A3_calls": 5, "A2_calls": 5,
            "primary_A3_token_allowance": 320, "A1_token_allowance": 320,
            "unique_generation_token_allowance_per_item": 960,
            "total_max_generation_tokens": 2 * LIMITS["max_generated_tokens_per_model"],
            "maximum_worker_wall_seconds": 3600,
            "rough_envelope": "Hard 30 minutes/model; actual speed unknown until smoke. No automatic retry or extension.",
            "disk_output_ceiling_bytes": 2 * LIMITS["max_disk_bytes_per_model"],
            "limits": LIMITS, "compute_matched": False,
            "accounting": "R0 shared across A0/A2/A3; S1 diagnostic adds one call outside five-call A3; logical per-arm and unique actual totals both reported"}
