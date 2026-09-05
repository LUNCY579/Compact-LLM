"""Immutable, resumable Track B activation capture and per-depth-pair fitting."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
PROGRAMME = ROOT.parent
sys.path.insert(0, str(PROGRAMME))

from shared.hashing import canonical_sha256, sha256_file, source_files, write_json
from shared.guards import guard_non_sealed_path, require_runtime, require_within
from cross_scale.src.depth_pair_screen import OBJECTIVES, depth_pairs, run_pair, select_matrix, validate_bundle


def utc():
    return datetime.now(timezone.utc).isoformat()


def configure(path):
    config = json.loads(guard_non_sealed_path(path).read_text(encoding="utf-8"))
    require_runtime(config)
    if config["track"] != "cross_scale" or config["objectives"] != list(OBJECTIVES):
        raise ValueError("wrong track or missing registered bridge controls")
    if config["generation_enabled"] or config["qwen8b_eligible"]:
        raise ValueError("generation and 8B are closed in the alignment protocol")
    if config["delta_steps"] < 1 or config["delta_learning_rate"] <= 0:
        raise ValueError("invalid delta fitting budget")
    if config["supa_layers"] != [8, 12, 16, 19] or config["donor_layers"] != [12, 16, 20, 24]:
        raise ValueError("the preregistered 16-pair depth matrix is required")
    for key in ("delta_learning_rate", "ridge_relative", "gradient_clip", "max_prompt_seconds",
                "max_capture_seconds", "max_pair_seconds", "max_rss_gib", "min_available_gib"):
        if not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"invalid positive configuration value: {key}")
    for key in ("delta_steps", "calibration_max_tokens"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"invalid integer configuration value: {key}")
    for key in ("state_anchor_weight", "delta_cosine_weight"):
        if not math.isfinite(config[key]) or config[key] < 0:
            raise ValueError(f"invalid objective weight: {key}")
    for bounds in (config["hidden_rms_ratio_bounds"], config["mechanism_gate"]["residual_ratio_bounds"]):
        if len(bounds) != 2 or not all(math.isfinite(v) for v in bounds) or not 0 < bounds[0] < bounds[1]:
            raise ValueError("invalid registered RMS bounds")
    for key, value in config["mechanism_gate"].items():
        if key != "residual_ratio_bounds" and (not math.isfinite(value) or not 0 < value <= 1):
            raise ValueError(f"invalid mechanism gate: {key}")
    if config["holdout_fraction"] != .25:
        raise ValueError("the frozen 96/32 scenario partition is required")
    return config


def run_directories(run_id):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("invalid run identifier")
    return (require_within(ROOT/"evidence"/"01_alignment_matrix"/run_id, ROOT/"evidence"),
            require_within(ROOT/"local"/run_id, ROOT/"local"))


def exclusive_tensor(path, value):
    import torch
    with path.open("xb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())


def calibration_rows(dataset):
    from shared.datasets import alignment_prompts
    rows = [{**row, "track": "cross_scale", "split": "calibration", "alignment_partition": partition}
            for partition in ("fit", "holdout") for row in alignment_prompts(dataset, partition=partition)]
    fit = {r["scenario_id"] for r in rows if r["alignment_partition"] == "fit"}
    heldout = {r["scenario_id"] for r in rows if r["alignment_partition"] == "holdout"}
    if len(fit) != 96 or len(heldout) != 32 or fit & heldout:
        raise ValueError("capture requires the preregistered 96-fit/32-heldout independent scenarios")
    return rows


def validate_receipt(directory, result, config, registration, arm):
    """Started evidence stays immutable; final evidence may only extend it."""
    started = json.loads((directory/"started.json").read_text(encoding="utf-8"))
    if (result.get("status") not in {"COMPLETE", "FAILED", "ABANDONED"} or started.get("status") != "STARTED"
            or result.get("track") != "cross_scale" or result.get("arm") != arm
            or result.get("config_sha256") != canonical_sha256(config)
            or result.get("registration_sha256") != canonical_sha256(registration)
            or result.get("source_sha256") != canonical_sha256(registration["sources"])
            or result.get("sources") != registration["sources"]):
        raise ValueError("invalid or incompatible terminal receipt")
    mutable = {"status", "peak_rss_bytes"}
    if any(result.get(key) != value for key, value in started.items() if key not in mutable):
        raise ValueError("final receipt rewrote its started contract")
    if result["status"] == "FAILED" and not (directory/"failure.json").is_file():
        raise ValueError("failed attempt has no failure receipt")
    if result["status"] == "ABANDONED" and result.get("original_process_absent") is not True:
        raise ValueError("abandonment lacks verified process absence")


def abandon_pair_attempt(run_id, config, registration, supa_layer, donor_layer, attempt):
    """Explicit interrupted-worker closure; preserve every original evidence byte."""
    import psutil
    run_directories(run_id)
    if (supa_layer, donor_layer) not in depth_pairs(config) or attempt < 0:
        raise ValueError("invalid pair attempt")
    directory = require_within(ROOT/"evidence"/"02_bridge_objectives"/run_id/
                               f"S{supa_layer}_D{donor_layer}"/f"attempt_{attempt:03d}", ROOT/"evidence")
    if (directory/"run_manifest.json").exists() or (directory/"abandonment.json").exists():
        raise ValueError("attempt already has terminal evidence")
    started = json.loads((directory/"started.json").read_text(encoding="utf-8"))
    # PID reuse is conservatively treated as live; never close an uncertain worker.
    if type(started.get("pid")) is not int or started["pid"] <= 0 or psutil.pid_exists(started["pid"]):
        raise RuntimeError("original worker PID is live or cannot be identified")
    result = {**started, "status": "ABANDONED", "original_process_absent": True,
              "abandoned_utc": utc(), "reason": "explicit recovery after verified original PID absence"}
    validate_receipt(directory, result, config, registration, "pair")
    write_json(directory/"abandonment.json", result)
    return directory/"abandonment.json"


def read_capture(run_id, config, registration, *, metadata_only=False):
    import torch
    evidence, local = run_directories(run_id)
    manifest = json.loads((evidence/"capture.json").read_text(encoding="utf-8"))
    validate_receipt(evidence, manifest, config, registration, "capture")
    if manifest != json.loads((evidence/"run_manifest.json").read_text(encoding="utf-8")):
        raise ValueError("capture final receipts disagree")
    path = require_within(local/"activations.pt", local)
    if (manifest["status"] != "COMPLETE" or manifest["config_sha256"] != canonical_sha256(config)
            or manifest["registration_sha256"] != canonical_sha256(registration)
            or manifest["activation_sha256"] != sha256_file(path)):
        raise ValueError("capture hash/config/registration mismatch")
    dataset = guard_non_sealed_path(manifest["dataset_manifest"])
    registered_path = dataset.relative_to(PROGRAMME.parents[1]).as_posix()
    if registration["dataset_manifests"].get(registered_path) != manifest["dataset_manifest_sha256"]:
        raise ValueError("capture dataset was not registered for this track")
    if sha256_file(dataset) != manifest["dataset_manifest_sha256"]:
        raise ValueError("capture dataset manifest changed")
    expected = calibration_rows(dataset)
    if (source_files("cross_scale") != registration["sources"]
            or manifest.get("canonical_fixture_bit_identical") is not True):
        raise ValueError("capture source or qualification mismatch")
    if metadata_only:
        if (manifest["fit_prompts"] != sum(row["alignment_partition"] == "fit" for row in expected)
                or manifest["holdout_prompts"] != sum(row["alignment_partition"] == "holdout" for row in expected)
                or set(manifest["heldout_scenarios"]) != {row["scenario_id"] for row in expected if row["alignment_partition"] == "holdout"}):
            raise ValueError("capture metadata does not cover the registered calibration partition")
        # Selection needs immutable hashes and recorded metrics; never allocate the
        # multi-gigabyte activation cache just to aggregate already completed pairs.
        return manifest, None
    bundle = torch.load(path, weights_only=True, map_location="cpu")
    validate_bundle(bundle, config)
    from cross_scale.src.donor_layers import strip_calibration_rows
    if (bundle["rows"] != strip_calibration_rows(expected)
            or bundle["train_prompt_mask"] != [row["alignment_partition"] == "fit" for row in expected]
            or bundle.get("canonical_fixture_bit_identical") is not True
            or manifest.get("canonical_fixture_bit_identical") is not True
            or bundle.get("config_sha256") != canonical_sha256(config)
            or bundle.get("sources") != registration["sources"]
            or source_files("cross_scale") != registration["sources"]
            or bundle.get("source_sha256") != manifest.get("source_sha256")
            or bundle["donor_identity"] != manifest["donor_identity"]
            or bundle["supa_identity"] != manifest["supa_identity"]):
        raise ValueError("capture source, calibration coverage or identity mismatch")
    return manifest, bundle


def collect_completed_pairs(run_id, config, registration, capture_hash, local):
    reports, sources = [], []
    for i, j in depth_pairs(config):
        pair_root = require_within(ROOT/"evidence"/"02_bridge_objectives"/run_id/f"S{i}_D{j}", ROOT/"evidence")
        completed = []
        for attempt in sorted(pair_root.glob("attempt_*")):
            attempt = require_within(attempt, pair_root)
            final = attempt/"run_manifest.json"
            if not final.is_file():
                final = attempt/"abandonment.json"
                if not final.is_file():
                    raise ValueError(f"unfinished attempt blocks aggregation: {attempt}")
            result = json.loads(final.read_text(encoding="utf-8"))
            validate_receipt(attempt, result, config, registration, "pair")
            if result["status"] == "COMPLETE":
                completed.append((attempt, result))
            sources.append({"path": str(attempt), "status": result["status"],
                            "manifest_sha256": sha256_file(final)})
        if len(completed) != 1:
            raise ValueError(f"exactly one complete attempt is required for S{i}/D{j}")
        attempt, result = completed[0]
        if (result["capture_sha256"] != capture_hash
                or result["metrics_sha256"] != sha256_file(attempt/"metrics.json")
                or result["bridge_sha256"] != sha256_file(require_within(result["bridge_path"], local))):
            raise ValueError("pair evidence hash/config/capture mismatch")
        report = json.loads((attempt/"metrics.json").read_text(encoding="utf-8"))
        if (report["supa_layer"], report["donor_layer"]) != (i, j):
            raise ValueError("pair receipt contains another depth pair")
        reports.append(report)
    return reports, sources


def baseline_receipt(config, registration, arm):
    import psutil
    import torch
    import transformers
    return {"status": "STARTED", "track": "cross_scale", "arm": arm,
            "branch": subprocess.check_output(["git", "-C", str(PROGRAMME.parents[1]), "branch", "--show-current"], text=True).strip(),
            "commit_sha": subprocess.check_output(["git", "-C", str(PROGRAMME.parents[1]), "rev-parse", "HEAD"], text=True).strip(),
            "config_sha256": canonical_sha256(config), "registration_sha256": canonical_sha256(registration),
            "source_sha256": canonical_sha256(registration["sources"]), "sources": registration["sources"],
            "registration": registration, "seed": config["seed"], "threads": config["threads"],
            "device": config["device"], "dtype": config["dtype"], "cache_mode": "none",
            "python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
            "pid": os.getpid(), "start_utc": utc(), "peak_rss_bytes": psutil.Process().memory_info().rss,
            "model_execution": arm != "aggregate", "generation_count": 0}


def execute(args):
    import psutil
    import torch
    import transformers
    from shared.manifests import verify_registration
    config = configure(args.config)
    if args.action == "preflight":
        print(json.dumps({"pairs": depth_pairs(config), "objectives": list(OBJECTIVES),
                          "matrix_cells": len(depth_pairs(config))*len(OBJECTIVES),
                          "models_loaded": 0, "generation_licensed": False}, indent=2))
        return 0
    registration = verify_registration("cross_scale", args.config)
    if args.action == "abandon-pair":
        print(abandon_pair_attempt(args.run_id, config, registration, args.supa_layer, args.donor_layer, args.attempt))
        return 0
    if torch.__version__ != "2.12.1+cu126" or transformers.__version__ != "5.14.1":
        raise RuntimeError("unqualified runtime; use historical Chat-Bubble/.venv python")
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    evidence, local = run_directories(args.run_id)
    if args.action == "capture":
        output = evidence
        if output.exists():
            read_capture(args.run_id, config, registration)
            print("Existing complete capture verified; no overwrite.")
            return 0
        output.mkdir(parents=True, exist_ok=False)
        local.mkdir(parents=True, exist_ok=False)
    elif args.action == "pair":
        if (args.supa_layer, args.donor_layer) not in depth_pairs(config) or args.attempt < 0:
            raise ValueError("unregistered pair or invalid attempt")
        output = require_within(ROOT/"evidence"/"02_bridge_objectives"/args.run_id/f"S{args.supa_layer}_D{args.donor_layer}"/f"attempt_{args.attempt:03d}", ROOT/"evidence")
        output.mkdir(parents=True, exist_ok=False)
    else:
        if args.attempt < 0:
            raise ValueError("invalid aggregate attempt")
        output = require_within(ROOT/"evidence"/"01_alignment_matrix"/args.run_id/"aggregate"/f"attempt_{args.attempt:03d}", ROOT/"evidence")
        output.mkdir(parents=True, exist_ok=False)
    receipt = baseline_receipt(config, registration, args.action)
    started = time.perf_counter()
    write_json(output/"started.json", receipt)
    try:
        if args.action == "capture":
            from shared.model_loading import require_available_memory
            from cross_scale.src.donor_layers import capture_bundle
            require_available_memory(config)
            dataset = guard_non_sealed_path(args.dataset_manifest)
            registered_path = dataset.relative_to(PROGRAMME.parents[1]).as_posix()
            if registration["dataset_manifests"].get(registered_path) != sha256_file(dataset):
                raise ValueError("capture requires the registered Track B dataset manifest")
            rows = calibration_rows(dataset)
            bundle = capture_bundle(rows, config)
            validate_bundle(bundle, config)
            if bundle["sources"] != registration["sources"]:
                raise RuntimeError("activation source differs from registration")
            exclusive_tensor(local/"activations.pt", bundle)
            receipt.update(activation_sha256=sha256_file(local/"activations.pt"),
                           dataset_manifest=str(dataset), dataset_manifest_sha256=sha256_file(dataset),
                           donor_identity=bundle["donor_identity"], supa_identity=bundle["supa_identity"],
                           supa_files=bundle["supa_files"],
                           canonical_fixture_bit_identical=bundle["canonical_fixture_bit_identical"],
                           fit_prompts=sum(bundle["train_prompt_mask"]), holdout_prompts=sum(not v for v in bundle["train_prompt_mask"]),
                           heldout_scenarios=bundle["heldout_scenarios"], truncation=bundle["lengths"])
        elif args.action == "pair":
            from shared.model_loading import require_available_memory
            from cross_scale.src.donor_layers import donor_identity, load_donor_layer
            from armr.provenance import checkpoint_files
            require_available_memory(config)
            capture, bundle = read_capture(args.run_id, config, registration)
            supa_path = guard_non_sealed_path(bundle["supa_identity"]["model_path"])
            if checkpoint_files(supa_path) != bundle["supa_files"]:
                raise ValueError("canonical Supa checkpoint differs from capture")
            donor_path, donor_before = donor_identity(config)
            if donor_before != bundle["donor_identity"]:
                raise ValueError("donor differs from activation capture")
            layer, rope, _ = load_donor_layer(config, args.donor_layer, donor_path)
            report, artifacts = run_pair(bundle, config, args.supa_layer, args.donor_layer, layer, rope)
            _, donor_after = donor_identity(config)
            if donor_before != donor_after:
                raise ValueError("donor checkpoint mutation during fitting")
            if checkpoint_files(supa_path) != bundle["supa_files"]:
                raise ValueError("canonical Supa checkpoint mutation during fitting")
            artifact_dir = local/"pairs"/f"S{args.supa_layer}_D{args.donor_layer}"/f"attempt_{args.attempt:03d}"
            artifact_dir.mkdir(parents=True, exist_ok=False)
            exclusive_tensor(artifact_dir/"bridges.pt", artifacts)
            write_json(output/"metrics.json", report)
            receipt.update(capture_sha256=sha256_file(evidence/"capture.json"),
                           dataset_manifest_sha256=capture["dataset_manifest_sha256"],
                           donor_identity=donor_before, supa_identity=bundle["supa_identity"],
                           activation_sha256=capture["activation_sha256"],
                           metrics_sha256=sha256_file(output/"metrics.json"),
                           bridge_path=str(artifact_dir/"bridges.pt"), bridge_sha256=sha256_file(artifact_dir/"bridges.pt"))
        else:
            capture, _ = read_capture(args.run_id, config, registration, metadata_only=True)
            reports, sources = collect_completed_pairs(args.run_id, config, registration,
                                                       sha256_file(evidence/"capture.json"), local)
            decision = select_matrix(reports, config)
            write_json(output/"matrix.json", {"pairs": reports, "sources": sources})
            write_json(output/"selection.json", decision)
            receipt.update(selection=decision, dataset_manifest_sha256=capture["dataset_manifest_sha256"],
                           matrix_sha256=sha256_file(output/"matrix.json"))
        if canonical_sha256(verify_registration("cross_scale", args.config)) != canonical_sha256(registration):
            raise RuntimeError("registration changed during execution")
        receipt["status"] = "COMPLETE"
    except BaseException as error:
        receipt["status"] = "FAILED"
        write_json(output/"failure.json", {"type": type(error).__name__, "error": str(error), "traceback": traceback.format_exc()})
        raise
    finally:
        memory = psutil.Process().memory_info()
        receipt.update(end_utc=utc(), wall_seconds=time.perf_counter()-started,
                       peak_rss_bytes=getattr(memory, "peak_wset", memory.rss))
        write_json(output/"run_manifest.json", receipt)
        if args.action == "capture" and receipt["status"] == "COMPLETE":
            write_json(output/"capture.json", receipt)
    print(f"COMPLETE {output}")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT/"configs/alignment_v2.json")
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("preflight")
    capture = actions.add_parser("capture")
    capture.add_argument("--run-id", required=True)
    capture.add_argument("--dataset-manifest", type=Path, required=True)
    pair = actions.add_parser("pair")
    pair.add_argument("--run-id", required=True)
    pair.add_argument("--supa-layer", type=int, required=True)
    pair.add_argument("--donor-layer", type=int, required=True)
    pair.add_argument("--attempt", type=int, default=0)
    abandon = actions.add_parser("abandon-pair", help="close an interrupted attempt only after its PID is absent")
    abandon.add_argument("--run-id", required=True)
    abandon.add_argument("--supa-layer", type=int, required=True)
    abandon.add_argument("--donor-layer", type=int, required=True)
    abandon.add_argument("--attempt", type=int, required=True)
    aggregate = actions.add_parser("aggregate")
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--attempt", type=int, default=0)
    args = parser.parse_args()
    if args.action in ("capture", "pair"):
        from shared.timing import machine_lease
        with machine_lease():
            return execute(args)
    return execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
