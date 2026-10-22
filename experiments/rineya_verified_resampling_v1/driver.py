"""Windows supervised native entry points. The worker never opens an oracle file."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

from .contracts import (ROOT, directory_bytes, file_sha, read_json, require, source_identity, write_new)
from .datasets import load_study
from .runner import collect


def native_launch(action: str, study_dir: Path, model_dir: Path, out: Path, *,
                  qualification: Path | None = None, allow_model_load: bool = False,
                  allow_inference: bool = False, resume: bool = False,
                  max_attempts: int | None = None, abandon_interrupted: bool = False) -> dict:
    require(action in {"qualify", "collect"}, "unknown native action")
    require(allow_model_load and (action != "collect" or allow_inference), "explicit native permissions missing")
    require(os.name == "nt", "native RVS requires the registered Windows host; CPU demo works elsewhere")
    study, _ = load_study(study_dir)
    require(study["dataset"]["kind"] == "centralized_bridge", "fixture data cannot enter native experiment")
    source_identity(native=True)
    if action == "collect":
        require(qualification is not None, "native qualification receipt required")
        q = read_json(qualification / "qualification.json")
        require(q["status"] == "PASS_NATIVE_INFERENCE_ONLY" and q["study_id"] == study["study_id"],
                "qualification does not match study")
        require(q["code_digest"] == source_identity(native=True)["code_digest"], "qualification source drift")
    require(resume or not out.exists(), "output already exists; explicit resume required")
    require(not resume or (action == "collect" and (out / "run.json").exists()), "no collection to resume")
    out.mkdir(parents=True, exist_ok=True)
    from experiments.opallo_native_collector_v1.supervisor import run_bounded
    common = Path(subprocess.check_output(["git", "rev-parse", "--git-common-dir"], cwd=ROOT, text=True).strip())
    common = (ROOT / common).resolve() if not common.is_absolute() else common.resolve()
    lock = common / "rineya-central-gpu0.lock"
    nonce = uuid.uuid4().hex
    with lock.open("x", encoding="utf-8") as f:
        f.write(nonce)
    try:
        active = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,process_name",
                                          "--format=csv,noheader,nounits"], text=True, timeout=15).strip()
        require(not any(term in active.lower() for term in ("python", "torch", "llama", "ollama", "vllm", "train", "comfy")),
                "competing model process detected; do not kill unrelated processes")
        segments = out / "segments"
        segments.mkdir(exist_ok=True)
        used = 0.0
        existing = sorted(p for p in segments.iterdir() if p.is_dir())
        for directory in existing:
            require((directory / "SUPERVISOR.json").exists(), "unaccounted prior supervisor segment; investigate before resume")
            used += read_json(directory / "SUPERVISOR.json")["elapsed_seconds"]
        invoked_at, attempted_now = time.monotonic(), 0
        maximum = study["slots"] if max_attempts is None else max_attempts
        require(type(maximum) is int and 1 <= maximum <= study["slots"], "invalid attempt allowance")
        while True:
            require(directory_bytes(out) < study["config"]["max_output_bytes"], "aggregate output storage budget")
            remaining = study["config"]["total_wall_seconds"] - used - (time.monotonic()-invoked_at)
            require(remaining >= 30, "total native wall budget exhausted")
            segment = segments / f"{len(list(segments.iterdir())):04d}"
            segment.mkdir()
            request = {"action": action, "study": str(study_dir.resolve()), "model": str(model_dir.resolve()),
                       "out": str(out.resolve()), "segment": str(segment.resolve()), "parent_pid": os.getpid(),
                       "nonce": nonce, "qualification": str(qualification.resolve()) if qualification else None,
                       "max_attempts": min(study["config"]["segment_attempts"], maximum-attempted_now),
                       "resume": (out / "run.json").exists(), "abandon_interrupted": abandon_interrupted,
                       "code_digest": source_identity(native=True)["code_digest"]}
            write_new(segment / "request.json", request)
            write_new(segment / "gpu_inventory.json", {"observed": active.splitlines()})
            command = [sys.executable, "-B", "-m", "experiments.rineya_verified_resampling_v1",
                       "_native-worker", "--request", str((segment / "request.json").resolve()),
                       "--go", str((segment / "GO.json").resolve())]
            run_bounded(command, ROOT, segment, seconds=min(study["config"]["segment_seconds"], remaining),
                        min_free_gib=6.0, max_rss_gib=4.0,
                        go={"request_sha256": file_sha(segment / "request.json"), "nonce": nonce})
            result = read_json(segment / "worker_result.json")
            if action == "qualify":
                require(result["status"] == "PASS_NATIVE_INFERENCE_ONLY", "qualification failed")
                write_new(out / "qualification.json", read_json(segment / "qualification.json"))
                return result
            attempted_now += result["newly_started_this_segment"]
            abandon_interrupted = False
            if result["status"] != "PAUSED" or attempted_now >= maximum:
                return result
    finally:
        if lock.exists() and lock.read_text() == nonce:
            lock.unlink()


def native_worker(request_path: Path, go_path: Path) -> dict:
    request = read_json(request_path)
    started = time.monotonic()
    from experiments.opallo_native_collector_v1.supervisor import contain_worker
    contain_worker(request["parent_pid"])
    while not go_path.exists():
        require(time.monotonic()-started < 30, "supervisor GO timeout")
        time.sleep(0.01)
    go = read_json(go_path)
    require(go == {"request_sha256": file_sha(request_path), "nonce": request["nonce"]}, "GO binding mismatch")
    require(source_identity(native=True)["code_digest"] == request["code_digest"], "source changed after launch")
    study, public = load_study(Path(request["study"]))
    require(study["dataset"]["kind"] == "centralized_bridge", "native fixture escalation prohibited")
    segment, output = Path(request["segment"]), Path(request["out"])
    from .native import NativeBackend
    backend = NativeBackend(Path(request["model"]), segment, study["config"], allow_model_load=True)
    if request["action"] == "qualify":
        result = backend.qualify(study, public, segment)
    else:
        q = read_json(Path(request["qualification"]) / "qualification.json")
        require(q["status"] == "PASS_NATIVE_INFERENCE_ONLY" and q["study_id"] == study["study_id"]
                and q["code_digest"] == backend.source["code_digest"] and q["backend"] == backend.identity,
                "qualification source/model/runtime/host drift")
        _, prompt_digest = backend.bind_prompts(public)
        require(prompt_digest == q["public_token_ids_digest"], "qualified prompt tokenization drift")
        from .runner import Journal
        before = Journal(output / "events.jsonl", study["config"]["max_output_bytes"])
        started_slots = len(before.records)
        pause_seconds = max(1.0, study["config"]["segment_seconds"] - (time.monotonic()-started) - 120)
        result = collect(study, public, backend, output, resume=request["resume"],
                         max_attempts=request["max_attempts"], abandon_interrupted=request["abandon_interrupted"],
                         pause_after_seconds=pause_seconds)
        result["newly_started_this_segment"] = result["started"]-started_slots
    write_new(segment / "worker_result.json", result)
    return result
