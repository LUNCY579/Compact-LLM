"""Create-only phase ledger, shared run claim, and Git-before-check interlock."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

from .contract import (Blocked, CLOSED, COLLECTOR, committed_file, digest, git, pins, read,
                       require, reviewed_inputs, safe_child, screen, sha, verify_source, write_new)
from .native import parity, validate_row
from .supervisor import run_bounded

PLANS = {
    "qualify": [("on8", "calibration", False, 0, 8, True),
                ("off8", "calibration", False, 0, 8, False),
                ("repeat8", "calibration", False, 0, 8, True)],
    "calibrate": [("baseline56", "calibration", False, 8, 64, True),
                  ("formatted64", "calibration", True, 0, 64, True)],
    "check": [("baseline128", "check", False, 0, 128, True),
              ("formatted128", "check", True, 0, 128, True)],
}
COUNTS = {"qualify": 24, "calibrate": 120, "check": 256}


def requests_for(plan: tuple) -> list[dict]:
    _, split, formatted, begin, end, _ = plan
    return screen().prompts(split, formatted)[begin:end]


def git_common(repo: Path) -> Path:
    path = Path(git(repo, "rev-parse", "--git-common-dir").decode().strip())
    return path.resolve() if path.is_absolute() else (repo / path).resolve()


def initialize(repo: Path, root: Path, model_dir: Path, commit: str, auth_path: str) -> None:
    repo, root, model_dir = repo.resolve(), root.resolve(), model_dir.resolve()
    require(Path(git(repo, "rev-parse", "--show-toplevel").decode().strip()).resolve() == repo,
            "repository root required")
    require(not root.exists(), "session directory exists; no resume/reset")
    auth, recipe = reviewed_inputs(repo, commit, auth_path)
    from .contract import asset_manifest
    assets = asset_manifest(model_dir)
    require(not root.is_relative_to(model_dir), "session cannot be inside official model assets")
    claim = git_common(repo) / "opallo-native-claims" / (pins()["subject_commit"] + ".json")
    require(not claim.exists(), "this subject already has a run claim; failure requires separate reviewed disposition")
    manifest = {"collector": COLLECTOR, "protocol": pins()["protocol"], "subject_commit": pins()["subject_commit"],
                "run_id": auth["run_id"], "repo": str(repo), "root": str(root), "model_dir": str(model_dir),
                "authorization_commit": commit, "authorization_path": auth_path,
                "authorization_sha256": digest(auth), "recipe": recipe, "assets": assets,
                "run_claim_path": str(claim), **CLOSED}
    # Claim FIRST: an interrupted initialization cannot silently mint a replacement run.
    write_new(claim, {"run_id": auth["run_id"], "root": str(root), "session_sha256": digest(manifest)})
    root.mkdir(parents=True, exist_ok=False, mode=0o700)
    write_new(root / "SESSION.json", manifest)


def load_session(root: Path) -> dict:
    verify_source()
    manifest = read(root / "SESSION.json")
    require(manifest["root"] == str(root.resolve()), "session moved/copied to a new path")
    claim = read(Path(manifest["run_claim_path"]))
    require(claim == {"run_id": manifest["run_id"], "root": manifest["root"],
                      "session_sha256": digest(manifest)}, "run claim mismatch")
    auth, recipe = reviewed_inputs(Path(manifest["repo"]), manifest["authorization_commit"], manifest["authorization_path"])
    require(digest(auth) == manifest["authorization_sha256"] and recipe == manifest["recipe"], "authorization/recipe drift")
    return manifest


def remaining(root: Path) -> float:
    import psutil
    clock = read(root / "CLOCK.json")
    require(abs(psutil.boot_time() - clock["boot_time"]) < 1.0, "host reboot changes chronology")
    wall, mono = time.time() - clock["wall"], time.monotonic() - clock["monotonic"]
    require(wall >= 0 and mono >= 0, "clock rollback detected")
    left = 3600.0 - max(wall, mono)
    require(left > 0, "whole-screen 3600-second limit reached; no clock/budget reset")
    return left


def verify_phase(root: Path, phase: str) -> dict:
    value = read(root / phase / "RESULT.json")
    require(value.get("status") == "PASS_CAPTURE_MECHANICS" and value.get("phase") == phase,
            f"{phase} is incomplete or blocked")
    require(read(root / phase / "START.json")["generations_reserved"] == COUNTS[phase], "reservation mismatch")
    require(value.get("generations_completed_this_phase") == COUNTS[phase] and
            value.get("optimizer_steps") == 0 and all(value.get(k) is False for k in CLOSED),
            "phase counts or authority changed")
    actual_inventory = {path.relative_to(root).as_posix(): sha(path)
                        for path in sorted((root / phase).rglob("*"))
                        if path.is_file() and path != root / phase / "RESULT.json"}
    require(value.get("artifact_sha256") == actual_inventory, "phase artifact inventory incomplete or changed")
    for relative, expected in value["artifact_sha256"].items():
        path = safe_child(root, relative)
        require(path.is_file() and sha(path) == expected, f"prior phase evidence changed: {relative}")
    return value


def verify_freeze(root: Path, manifest: dict, commit: str, relative: str, *, active: bool = False) -> dict:
    verify_phase(root, "calibrate")
    require(active or not (root / "check" / "START.json").exists(), "check has already been opened")
    repo = Path(manifest["repo"])
    path = committed_file(repo, commit, relative)
    packet = read(root / "calibrate" / "FREEZE_PACKET.json")
    require(path.read_bytes() == (root / "calibrate" / "FREEZE_PACKET.json").read_bytes(), "Git freeze packet bytes differ")
    require(read(path) == packet and packet["session_sha256"] == digest(manifest), "freeze/session mismatch")
    git(repo, "merge-base", "--is-ancestor", manifest["authorization_commit"], commit)
    return {"freeze_commit": commit, "freeze_path": relative, "freeze_sha256": sha(path),
            "session_sha256": digest(manifest), "observed_before_check_wall": time.time(), **CLOSED}


def freeze(root: Path, commit: str, relative: str) -> None:
    manifest = load_session(root)
    remaining(root)
    write_new(root / "GIT_FREEZE.json", verify_freeze(root, manifest, commit, relative))


def ready(root: Path, phase: str, manifest: dict) -> None:
    require(phase in PLANS, "unregistered native phase")
    require(not (root / phase).exists(), "phase has already been reserved; silent retry forbidden")
    for previous in list(PLANS)[:list(PLANS).index(phase)]:
        verify_phase(root, previous)
    if phase == "qualify":
        require(not (root / "CLOCK.json").exists(), "whole-screen clock already started")
    if phase == "check":
        saved = read(root / "GIT_FREEZE.json")
        actual = verify_freeze(root, manifest, saved["freeze_commit"], saved["freeze_path"])
        for key in ("freeze_commit", "freeze_path", "freeze_sha256", "session_sha256"):
            require(saved.get(key) == actual[key], "chronological Git freeze changed")
    spent = sum(read(root / p / "START.json")["generations_reserved"]
                for p in PLANS if (root / p / "START.json").exists())
    require(spent + COUNTS[phase] <= 400, "400-generation screen budget exceeded")


def verify_active_worker(root: Path, manifest: dict, phase: str, label: str) -> None:
    """Recheck authorization/chronology inside worker; private CLI is not a bypass."""
    require(phase in PLANS and label in [plan[0] for plan in PLANS[phase]], "unregistered worker")
    current = root / phase
    require(not (current / "RESULT.json").exists() and not (current / "BLOCKED.json").exists(),
            "phase is finished or blocked; no worker retry")
    start = read(current / "START.json")
    require(start.get("phase") == phase and start.get("session_sha256") == digest(manifest) and
            start.get("generations_reserved") == COUNTS[phase], "worker lacks phase reservation")
    for previous in list(PLANS)[:list(PLANS).index(phase)]:
        verify_phase(root, previous)
    spent = sum(read(root / p / "START.json")["generations_reserved"]
                for p in PLANS if (root / p / "START.json").exists())
    require(spent <= 400, "worker generation budget exceeded")
    if phase == "check":
        saved = read(root / "GIT_FREEZE.json")
        actual = verify_freeze(root, manifest, saved["freeze_commit"], saved["freeze_path"], active=True)
        require(start.get("git_freeze_receipt_sha256") == sha(root / "GIT_FREEZE.json"), "freeze changed after check opening")
        for key in ("freeze_commit", "freeze_path", "freeze_sha256", "session_sha256"):
            require(saved.get(key) == actual[key], "worker Git freeze mismatch")
        require(saved["observed_before_check_wall"] <= start["started_wall"], "freeze occurred after check opened")
    labels = [plan[0] for plan in PLANS[phase]]
    for previous in labels[:labels.index(label)]:
        completed = read(current / previous / "CAPTURE.json")
        require(completed.get("status") == "PASS_NATIVE_JOB_MECHANICS" and
                completed.get("requests_sha256") == pins()["job_prompts_sha256"][phase][previous],
                "previous capture job is not complete")
    remaining(root)


def validate_job_capture(value: dict, requests: list[dict], root: Path, confidence: bool) -> None:
    require(value.get("kind") == "native_capture" and value.get("status") == "PASS_NATIVE_JOB_MECHANICS",
            "native job did not complete")
    require(value.get("requests_sha256") == digest(requests), "job prompt digest mismatch")
    identity = value.get("identity", {})
    require(identity.get("unknown_single_token") is True and identity.get("unknown_decoded") == "UNKNOWN" and
            identity.get("unknown_encoded_ids") == [identity.get("unknown_token_id")], "UNKNOWN qualification missing")
    require(identity.get("weight_sha256") == pins()["assets"]["model.safetensors"]["sha256"], "wrong model identity")
    require(isinstance(value.get("rows"), list) and len(value["rows"]) == len(requests), "incomplete job capture")
    for row, prompt in zip(value["rows"], requests):
        validate_row(row, prompt, identity, root, confidence)


def full_capture(root: Path, split: str, formatted: bool, parts: list[dict]) -> dict:
    identity = parts[0]["identity"]
    require(all(p["identity"] == identity for p in parts), "cross-job identity mismatch")
    s = screen()
    capture = {"protocol": s.PROTOCOL, "kind": "native_capture", "split": split, "formatted": formatted,
               "prompts_sha256": s.digest(s.prompts(split, formatted)), "identity": identity,
               "rows": [row for part in parts for row in part["rows"]]}
    s.validate_capture(capture, split, formatted)
    for row, request in zip(capture["rows"], s.prompts(split, formatted)):
        validate_row(row, request, identity, root, True)
    return capture


def run_job(root: Path, manifest: dict, phase: str, plan: tuple) -> dict:
    label, split, formatted, _, _, confidence = plan
    out = root / phase / label
    out.mkdir()
    requests = requests_for(plan)
    task = {"phase": phase, "label": label, "split": split, "formatted": formatted,
            "confidence": confidence, "requests": requests, "requests_sha256": digest(requests),
            "session_sha256": digest(manifest)}
    write_new(out / "REQUEST.json", task)
    command = [sys.executable, "-m", "experiments.opallo_native_collector_v1.cli", "_worker",
               "--session", str(root), "--phase", phase, "--label", label, "--parent-pid", str(os.getpid())]
    run_bounded(command, Path(manifest["repo"]), out, seconds=min(900.0, remaining(root)),
                min_free_gib=3.0, max_rss_gib=8.0,
                go={"parent_pid": os.getpid(), "task_sha256": digest(task), "session_sha256": digest(manifest)})
    value = read(out / "CAPTURE.json")
    validate_job_capture(value, requests, root, confidence)
    require(value["identity"]["runtime_sha256"] == manifest["recipe"]["runtime_sha256"] and
            value["identity"]["recipe_sha256"] == digest(manifest["recipe"]), "job runtime mismatch")
    return value


def execute_phase(root: Path, phase: str) -> dict:
    """Exactly one attempt per phase; unsuccessful phases remain consumed."""
    import psutil
    manifest = load_session(root)
    ready(root, phase, manifest)
    # Cooperative lock is shared across clones for this OS user, not just output dirs.
    lock = Path.home() / ".cache" / "rineya" / "opallo-native-host.lock"
    write_new(lock, {"pid": os.getpid(), "session_sha256": digest(manifest)})
    out = root / phase
    try:
        out.mkdir(exist_ok=False)
        if phase == "qualify":
            write_new(root / "CLOCK.json", {"wall": time.time(), "monotonic": time.monotonic(), "boot_time": psutil.boot_time()})
        write_new(out / "START.json", {"phase": phase, "session_sha256": digest(manifest),
                                       "generations_reserved": COUNTS[phase], "started_wall": time.time(),
                                       "git_freeze_receipt_sha256": sha(root / "GIT_FREEZE.json") if phase == "check" else None})
        jobs = [run_job(root, manifest, phase, plan) for plan in PLANS[phase]]
        identities = [j["identity"] for j in jobs]
        if phase != "qualify":
            identities.append(read(root / "qualify" / "on8" / "CAPTURE.json")["identity"])
        require(all(i == identities[0] for i in identities), "runtime/model/tokenizer identity drift between captures")
        if phase == "qualify":
            parity(*(j["rows"] for j in jobs))
        elif phase == "calibrate":
            reused = read(root / "qualify" / "on8" / "CAPTURE.json")
            validate_job_capture(reused, requests_for(PLANS["qualify"][0]), root, True)
            bc = full_capture(root, "calibration", False, [reused, jobs[0]])
            fc = full_capture(root, "calibration", True, [jobs[1]])
            write_new(out / "baseline.json", bc)
            write_new(out / "formatted.json", fc)
            spec = screen().calibrate(bc, fc)  # Unchanged B-only selector; F diagnostic.
            write_new(out / "spec.json", spec)
            write_new(out / "FREEZE_PACKET.json", {"collector": COLLECTOR, "protocol": pins()["protocol"],
                "session_sha256": digest(manifest), "qualification_receipt_sha256": sha(root / "qualify" / "RESULT.json"),
                "baseline_calibration_sha256": digest(bc), "formatted_calibration_sha256": digest(fc),
                "baseline_calibration_bytes_sha256": sha(out / "baseline.json"),
                "formatted_calibration_bytes_sha256": sha(out / "formatted.json"),
                "spec": spec, "spec_sha256": digest(spec), "generations_reserved_so_far": 144, **CLOSED})
        else:
            bc = full_capture(root, "check", False, [jobs[0]])
            fc = full_capture(root, "check", True, [jobs[1]])
            write_new(out / "baseline.json", bc)
            write_new(out / "formatted.json", fc)
        remaining(root)
        artifacts = {p.relative_to(root).as_posix(): sha(p) for p in sorted(out.rglob("*")) if p.is_file()}
        result = {"collector": COLLECTOR, "phase": phase, "status": "PASS_CAPTURE_MECHANICS",
                  "artifact_sha256": artifacts, "generations_completed_this_phase": COUNTS[phase],
                  "optimizer_steps": 0, "native_efficacy_qualified": False, **CLOSED}
        write_new(out / "RESULT.json", result)
        return result
    except BaseException as exc:
        if out.exists() and not (out / "BLOCKED.json").exists():
            write_new(out / "BLOCKED.json", {"status": "BLOCKED", "phase": phase,
                "error_type": type(exc).__name__, "error": str(exc), "retry_authorized": False, **CLOSED})
        raise
    finally:
        # A killed supervisor leaves its lock; it is never auto-cleared on next entry.
        lock.unlink(missing_ok=True)


def score(root: Path) -> dict:
    manifest = load_session(root)
    for phase in PLANS:
        verify_phase(root, phase)
    saved = read(root / "GIT_FREEZE.json")
    committed_file(Path(manifest["repo"]), saved["freeze_commit"], saved["freeze_path"])
    packet = read(root / "calibrate" / "FREEZE_PACKET.json")
    require(sha(safe_child(Path(manifest["repo"]), saved["freeze_path"])) == saved["freeze_sha256"], "freeze changed")
    require(packet["session_sha256"] == digest(manifest), "freeze/session drift")
    # Consume the one-time scoring opportunity before reading check capture contents.
    write_new(root / "SCORE_START.json", {"session_sha256": digest(manifest), "wall": time.time()})
    bc, fc = read(root / "calibrate" / "baseline.json"), read(root / "calibrate" / "formatted.json")
    b, f = read(root / "check" / "baseline.json"), read(root / "check" / "formatted.json")
    result = screen().replay(read(root / "calibrate" / "spec.json"), bc, fc, b, f)
    write_new(root / "SCREEN_RESULT.json", result)
    return result
