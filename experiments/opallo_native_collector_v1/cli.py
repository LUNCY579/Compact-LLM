"""OPALLO-NATIVE-001: inventory -> reviewed init -> qualify -> calibrate -> Git freeze -> check.

No automatic dependency installation, checkpoint download, training, retry or gate waiver.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
import time
from pathlib import Path

from .contract import (Blocked, CLOSED, COLLECTOR, append_event, digest, pins, read, require,
                       source_manifest, verify_source, write_new)


def inventory(out: Path, model_dir: Path | None) -> dict:
    verify_source()
    packages = {}
    for name in ("torch", "transformers", "tokenizers", "safetensors", "numpy", "psutil"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    blockers = ["HUMAN_192_TARGET_REVIEW_NOT_ATTESTED", "FRESH_UNION_OVERLAP_AUDIT_NOT_ATTESTED",
                "ISOLATED_RUNTIME_RECIPE_NOT_FROZEN", "NATIVE_EIGHT_ROW_PARITY_UNEXECUTED",
                "HOST_SPECIFIC_NATIVE_SUPERVISOR_QUALIFICATION_PENDING"]
    assets = None
    if model_dir is not None:
        try:
            from .contract import asset_manifest
            assets = asset_manifest(model_dir)
        except (Blocked, OSError, ValueError) as exc:
            blockers.append("OFFICIAL_ASSET_CHECK_BLOCKED:" + str(exc))
    else:
        blockers.append("OFFICIAL_LOCAL_OPALLO_ASSETS_NOT_PROVISIONED")
    if any(value is None for value in packages.values()):
        blockers.append("RUNTIME_PACKAGES_MISSING")
    if packages["transformers"] != "4.51.3":
        blockers.append("ISOLATED_TRANSFORMERS_4_51_3_NOT_AVAILABLE")
    value = {"collector": COLLECTOR, "status": "BLOCKED_BEFORE_NATIVE", "protocol": pins()["protocol"],
             "subject_commit": pins()["subject_commit"], "python": sys.version, "platform": platform.platform(),
             "packages": packages, "assets": assets, "blockers": blockers,
             "collector_implemented": True, "native_collector_qualified": False,
             "source_sha256": source_manifest(), "pretrained_load_attempts": 0,
             "pretrained_forward_calls": 0, "native_generations": 0, "optimizer_steps": 0,
             "check_model_outputs_accessed": False, "threshold_selected": False,
             "retention_executed": False, **CLOSED}
    write_new(out, value)
    return value


def templates(out: Path) -> None:
    from .contract import screen
    verify_source()
    out.mkdir(parents=True, exist_ok=False)
    rows = screen().fixture("calibration") + screen().fixture("check")
    human = {"status": "PENDING", "reviewer_name": None, "reviewed_at_utc": None,
             "fixture_canonical_sha256": pins()["fixture_canonical_sha256"],
             "rows": [{"id": r["id"], "scenario_id": r["scenario_id"], "decision": "PENDING",
                       "complete_value_correct": None, "field_presence_correct": None,
                       "unambiguous_contract": None, "notes": None} for r in rows]}
    audit = {"status": "PENDING", "reviewer_name": None, "reviewed_at_utc": None,
             "fixture_canonical_sha256": pins()["fixture_canonical_sha256"],
             "unresolved_findings": None, "exclusions": [], "sources": [
                 {"category": category, "source_commit": None, "path": None, "sha256": None,
                  "rows": None, "normalized_full_prompt_matches": None,
                  "normalized_latest_user_matches": None, "near_template_review": "PENDING"}
                 for category in ("training", "historical_demo", "exposed_panel", "repair_selection")]}
    auth = {"status": "PENDING", "collector": COLLECTOR, "protocol": pins()["protocol"],
            "subject_commit": pins()["subject_commit"], "run_id": None, "host_quiescent": None,
            "source_sha256": source_manifest(), "recipe": {"path": None, "sha256": None},
            "human_review": {"path": None, "sha256": None}, "overlap_audit": {"path": None, "sha256": None}, **CLOSED}
    for name, value in (("human_review.pending.json", human), ("overlap_audit.pending.json", audit),
                        ("authorization.pending.json", auth)):
        write_new(out / name, value)
    write_new(out / "review_rows.json", {"rows": rows, "status": "PENDING_HUMAN_REVIEW", **CLOSED})


def worker(root: Path, phase: str, label: str, parent_pid: int) -> None:
    from .supervisor import contain_worker
    contain_worker(parent_pid)
    verify_source()
    from .session import PLANS, load_session, verify_active_worker
    plan = [p for p in PLANS.get(phase, []) if p[0] == label]
    require(len(plan) == 1, "unregistered capture job")
    out = root / phase / label
    deadline = time.monotonic() + 10
    while not (out / "GO.json").exists():
        require(time.monotonic() < deadline and os.getppid() == parent_pid, "worker GO/parent unavailable")
        time.sleep(0.01)
    manifest = load_session(root)
    verify_active_worker(root, manifest, phase, label)
    task, go = read(out / "REQUEST.json"), read(out / "GO.json")
    require(go == {"parent_pid": parent_pid, "task_sha256": digest(task), "session_sha256": digest(manifest)}, "supervisor handoff mismatch")
    require(task["session_sha256"] == digest(manifest), "task/session mismatch")
    _, split, formatted, begin, end, confidence = plan[0]
    require(task["phase"] == phase and task["label"] == label and task["split"] == split and
            task["formatted"] is formatted and task["confidence"] is confidence, "job policy mismatch")
    requests = task["requests"]
    require(len(requests) == end - begin and digest(requests) == task["requests_sha256"], "request count/digest mismatch")
    require(task["requests_sha256"] == pins()["job_prompts_sha256"][phase][label], "unregistered prompt bytes")
    # The native worker receives prompt-only records. It never invokes a fixture key or scorer.
    for request in requests:
        require(set(request) == {"id", "messages"}, "gold/metadata present in worker input")
    from .native import NativeEngine, capture_one, render_input, validate_row
    from .supervisor import memory_snapshot
    rows = []
    write_new(out / "WORKER_STARTED.json", {"pid": os.getpid(), "session_sha256": digest(manifest),
                                           "requests_sha256": task["requests_sha256"], "wall": time.time()})
    try:
        require(memory_snapshot()[0] >= 3.0, "host memory blocks model load")
        append_event(out / "events.jsonl", {"event": "model_load_start", "wall": time.time()})
        engine = NativeEngine(Path(manifest["model_dir"]), manifest["recipe"])
        write_new(out / "IDENTITY.json", engine.identity)
        for index, request in enumerate(requests):
            require(os.getppid() == parent_pid, "native supervisor exited")
            require(memory_snapshot()[0] >= 3.0, "host memory floor")
            append_event(out / "events.jsonl", {"event": "generation_start", "id": request["id"],
                                               "ordinal": index, "confidence": confidence})
            row = capture_one(engine, request, root, f"{phase}/{label}/signals/{index:03d}", confidence)
            # Retain failed/unterminated output BEFORE treating the job as blocked.
            write_new(out / "rows" / f"{index:03d}.json", row)
            validate_row(row, request, engine.identity, root, confidence)
            text, ids = render_input(engine.tokenizer, request, engine.vocabulary)
            require(row["rendered_prompt"] == text and row["input_ids"] == ids, "input replay differs")
            for field, tokens in (("prediction", row["content_ids"]), ("decoded_with_eos", row["generated_ids"])):
                require(row[field] == engine.tokenizer.decode(tokens, skip_special_tokens=False,
                                                               clean_up_tokenization_spaces=False), "decode replay differs")
            rows.append(row)
            append_event(out / "events.jsonl", {"event": "generation_complete", "id": request["id"],
                                               "generation_steps": row["generation_steps"]})
        engine.verify_unchanged()
        write_new(out / "CAPTURE.json", {"collector": COLLECTOR, "kind": "native_capture",
            "status": "PASS_NATIVE_JOB_MECHANICS", "requests_sha256": task["requests_sha256"],
            "identity": engine.identity, "rows": rows, "optimizer_steps": 0, **CLOSED})
    except BaseException as exc:
        write_new(out / "WORKER_BLOCKED.json", {"status": "BLOCKED", "error_type": type(exc).__name__,
            "error": str(exc), "completed_generations": len(rows), "retry_authorized": False, **CLOSED})
        raise


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    cmd = commands.add_parser("inventory")
    cmd.add_argument("--out", type=Path, required=True)
    cmd.add_argument("--model-dir", type=Path)
    cmd = commands.add_parser("templates")
    cmd.add_argument("--out", type=Path, required=True)
    cmd = commands.add_parser("recipe")
    cmd.add_argument("--device", choices=("cpu", "cuda:0"), required=True)
    cmd.add_argument("--dtype", choices=("float32", "bfloat16"), required=True)
    cmd.add_argument("--threads", type=int, default=2)
    cmd.add_argument("--out", type=Path, required=True)
    cmd = commands.add_parser("init")
    for arg in ("repo", "session", "model-dir"):
        cmd.add_argument("--" + arg, type=Path, required=True)
    cmd.add_argument("--authorization-commit", required=True)
    cmd.add_argument("--authorization-path", required=True)
    for name in ("qualify", "calibrate", "check", "score", "freeze", "_worker"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--session", type=Path, required=True)
        if name == "freeze":
            cmd.add_argument("--commit", required=True)
            cmd.add_argument("--packet-path", required=True)
        if name == "_worker":
            cmd.add_argument("--phase", required=True)
            cmd.add_argument("--label", required=True)
            cmd.add_argument("--parent-pid", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "inventory":
            result = inventory(args.out, args.model_dir)
            print(result["status"])
            return 2
        if args.command == "templates":
            templates(args.out)
        elif args.command == "recipe":
            verify_source()
            from .native import make_recipe
            write_new(args.out, make_recipe(args.device, args.dtype, args.threads))
        elif args.command == "_worker":
            worker(args.session.resolve(), args.phase, args.label, args.parent_pid)
        else:
            from .session import execute_phase, freeze, initialize, score
            if args.command == "init":
                initialize(args.repo, args.session, args.model_dir, args.authorization_commit, args.authorization_path)
            elif args.command == "freeze":
                freeze(args.session.resolve(), args.commit, args.packet_path)
            elif args.command == "score":
                score(args.session.resolve())
            else:
                execute_phase(args.session.resolve(), args.command)
        return 0
    except (Blocked, OSError, ValueError, KeyError, TypeError) as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
