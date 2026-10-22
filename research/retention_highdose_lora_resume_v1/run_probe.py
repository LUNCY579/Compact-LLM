"""Run one frozen diagnostic under the existing watchdog and bind its model bytes.

The runner uses only the standard library. It never imports the scorer or torch,
retries an attempt, removes files, or reports endpoint scores. Every --out must
be new; the legacy probe label must also have no pre-existing result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys

BASE = Path("C:/Users/clclc/BusinessProject")
PILOT = BASE / "ChatBubble-retention-doselora-resume-20260920"
PYTHON = BASE / ".venv-rab001-cu126/Scripts/python.exe"
RAB = PILOT / "experiments/architecture_benchmark_001"
WRAPPER = PILOT / "docs/model-quality/audit-2026-09-17/supervised_probe.py"
SUPERVISOR_CODE = PILOT / "experiments/model_quality_gpu_v1/supervisor.py"
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt")
PREFIXES = {"process": "PROCESS", "retention": "RETENTION_PROBE", "long_context": "LONGCTX"}
BANK_NAMES = {"process-transition-probe-001": "dose", "select-views-001": "select"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def regular(path: Path) -> Path:
    require(path.is_file() and not path.is_symlink(), f"missing/nonregular file: {path}")
    return path


def sha(path: Path) -> str:
    with regular(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path: Path) -> dict:
    value = json.loads(regular(path).read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), f"expected JSON object: {path}")
    return value


def native_digest(value: object) -> str:
    """MQ native integrity.digest uses unescaped Unicode, unlike RAB's digest."""
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write(path: Path, value: dict) -> None:
    encoded = json.dumps(value, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())


def copy_new(source: Path, destination: Path) -> str:
    """Copy exact bytes, without overwriting either an old result or a failed copy."""
    before = sha(source)
    with regular(source).open("rb") as src, destination.open("xb") as dst:
        for chunk in iter(lambda: src.read(1024 * 1024), b""):
            dst.write(chunk)
        dst.flush()
        os.fsync(dst.fileno())
    require(sha(source) == sha(destination) == before, "raw result changed while copying")
    return before


def implementation_digest() -> str:
    """Exactly gates.implementation_digest, without importing its torch dependency."""
    files = list(RAB.rglob("*.py")) + [RAB / name for name in
            ("protocol.json", "source_lock.json", "requirements.txt", "requirements-native.txt")]
    for name in read(RAB / "protocol.json").get("amendments", []):
        path = RAB / name
        require(path.resolve().is_relative_to(RAB.resolve()), "protocol amendment escapes RAB")
        regular(path)
        files.append(path)
    manifest = {p.relative_to(RAB).as_posix(): sha(p)
                for p in sorted(files, key=lambda p: p.relative_to(RAB).as_posix())}
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def row_ids(path: Path) -> set[str]:
    result = set()
    with regular(path).open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            value = row.get("id")
            require(isinstance(value, str) and value and value not in result,
                    f"missing or duplicate row ID in {path}")
            result.add(value)
    require(bool(result), f"empty instrument: {path}")
    return result


def training_snapshot(gate: dict, receipts: tuple[dict, dict]) -> dict:
    """Bind the optimizer trajectory to this frozen plan and its actual data files.

    Reproduce controller.presentations' row/order/token/update accounting with
    streamed JSON and the registered 2048-token update grouping, without model
    imports. Current files and checkpoint metadata must agree before probing.
    """
    plan_path = Path(gate["plan_path"]).resolve()
    require(sha(plan_path) == gate["plan_file_sha256"], "training plan file differs from frozen build gate")
    plan = read(plan_path)
    prepared = Path(gate["prepared"]).resolve()
    contract_path = prepared / "DATA_CONTRACT.json"
    contract = read(contract_path)
    plan_digest, contract_sha = native_digest(plan), sha(contract_path)
    require(contract.get("experiment") == plan.get("experiment"), "contract/plan experiment differs")
    files = contract.get("files")
    require(isinstance(files, dict) and "microfit-train.jsonl" in files, "training data contract is incomplete")
    hashes = {}
    for name, expected in files.items():
        path = (prepared / name).resolve()
        require(path.is_relative_to(prepared), "data contract file escapes prepared directory")
        hashes[str(path)] = sha(path)
        require(hashes[str(path)] == expected, f"prepared training artifact changed: {name}")
    order, parents, lengths, supervised = [], set(), [], 0
    with regular(prepared / "microfit-train.jsonl").open(encoding="utf-8-sig") as stream:
        for line in stream:
            require(len(line) <= 2**24, "training row exceeds native reader bound")
            row = json.loads(line)
            require(row.get("source_split") == "train" and row.get("mode") == "direct"
                    and row.get("truncated") is False, "training row is not untruncated direct TRAIN")
            require(isinstance(row.get("id"), str) and isinstance(row.get("parent_id"), str),
                    "training row identity missing")
            tokens, labels = row.get("ids"), row.get("labels")
            require(isinstance(tokens, list) and isinstance(labels, list) and len(tokens) == len(labels)
                    and 0 < len(tokens) <= 2048, "training token/label inventory is invalid")
            order.append(row["id"])
            parents.add(row["parent_id"])
            lengths.append(len(tokens))
            supervised += sum(token != -100 for token in labels[1:])
    require(order == plan.get("row_order_per_exposure") and len(order) == len(set(order)),
            "prepared training row order differs from frozen plan")
    require(len(parents) == plan.get("unique_worlds") == gate.get("training_parent_count"),
            "training parent inventory differs from frozen gate/plan")
    require(len(order) == gate.get("training_rows") == contract.get("rows"),
            "training row count differs from frozen gate/contract")
    exposures = plan.get("exposures_per_world")
    require(type(exposures) is int and exposures > 0, "invalid frozen exposure count")
    repeated = lengths * exposures
    groups, current, used = [], [], 0
    for index, length in enumerate(repeated):
        if current and used + length > 2048:
            groups.append(current)
            current, used = [], 0
        current.append(index)
        used += length
    if current:
        groups.append(current)
    require(len(repeated) == plan.get("presentations") and sum(repeated) == plan.get("processed_tokens")
            and supervised * exposures == plan.get("supervised_tokens") and groups == plan.get("update_groups"),
            "prepared token/exposure/update inventory differs from frozen plan")
    for bank in gate["banks"].values():
        path = Path(bank["path"]).resolve()
        require(sha(path) == bank["sha256"], "registered acquisition bank changed")
        bank_parents = set()
        with regular(path).open(encoding="utf-8-sig") as stream:
            for line in stream:
                parent = json.loads(line).get("parent_id")
                require(isinstance(parent, str) and parent, "acquisition bank parent ID missing")
                bank_parents.add(parent)
        require(len(bank_parents) == bank["parents"] and
                len(parents & bank_parents) == bank["training_parent_overlap"],
                "live train/evaluation parent overlap differs from build gate")
    for receipt in receipts:
        binding = receipt.get("checkpoint_binding")
        require(isinstance(binding, dict) and native_digest(binding) == receipt.get("checkpoint_binding_sha256"),
                "checkpoint binding body/hash mismatch")
        require(binding.get("plan_sha256") == plan_digest, "arm trained under a different frozen plan")
        require(binding.get("data_contract_sha256") == contract_sha, "arm trained under a different data contract")
        require(binding.get("experiment") == plan.get("experiment"), "arm training experiment differs")
        source = binding.get("source", {})
        require(source.get("assets") == gate["source_assets_sha256"] and
                source.get("sha256") == native_digest(gate["source_assets_sha256"]),
                "checkpoint binding source assets differ")
        progress = receipt.get("checkpoint_progress", {})
        require(progress.get("updates") == progress.get("next_group") == len(groups)
                and progress.get("processed_tokens") == plan["processed_tokens"]
                and progress.get("supervised_tokens") == plan["supervised_tokens"],
                "arm checkpoint is not the complete registered training trajectory")
    return {"plan_path": str(plan_path), "plan_file_sha256": gate["plan_file_sha256"],
            "plan_sha256": plan_digest, "prepared": str(prepared),
            "data_contract_path": str(contract_path), "data_contract_sha256": contract_sha,
            "data_files_sha256": hashes, "updates": len(groups),
            "processed_tokens": plan["processed_tokens"], "supervised_tokens": plan["supervised_tokens"]}


def arm_snapshot(arm: Path, gate: dict) -> dict:
    """Compare both arm receipts and live sidecars against the frozen source assets."""
    arm_path, materialization_path = arm / "ARM.json", arm / "MATERIALIZATION.json"
    arm_receipt, materialization = read(arm_path), read(materialization_path)
    training = training_snapshot(gate, (arm_receipt, materialization))
    expected = gate["source_assets_sha256"]
    require(set(expected) == {"model.safetensors", *SIDECARS}, "incomplete frozen source asset map")
    source = Path(gate["source"]).resolve()
    weight = sha(arm / "model.safetensors")
    require(arm_receipt.get("arm_weight_sha256") == materialization.get("out_weight_sha256") == weight,
            "arm/MATERIALIZATION/live weight identity mismatch")
    for key in ("checkpoint_manifest_sha256", "checkpoint_binding_sha256", "alpha"):
        require(key in arm_receipt and key in materialization and arm_receipt[key] == materialization[key],
                f"arm/MATERIALIZATION provenance differs: {key}")
    for receipt in (arm_receipt, materialization):
        require(Path(receipt.get("source", "")).resolve() == source, "arm source path differs from build gate")
        require(receipt.get("source_assets_sha256") == expected, "arm source assets differ from build gate")
        require(receipt.get("source_weight_sha256") == expected["model.safetensors"], "arm source weight differs")
        require(receipt.get("stored_dtype") == "float32", "expected a registered FP32 arm")
        for key in ("arm_weight_sha256", "out_weight_sha256"):
            if key in receipt:
                require(receipt[key] == weight, f"conflicting arm weight field: {key}")
    sidecars = {name: sha(arm / name) for name in SIDECARS}
    source_sidecars = {name: sha(source / name) for name in SIDECARS}
    require(sidecars == source_sidecars == {name: expected[name] for name in SIDECARS},
            "arm/source sidecar hash mismatch")
    return {"weight_sha256": weight, "arm_receipt_sha256": sha(arm_path),
            "materialization_sha256": sha(materialization_path),
            "sidecar_sha256": sidecars, "source_sidecar_sha256": source_sidecars,
            "training": training}


def inputs(kind: str, instrument: str, gate: dict, digest: str) -> tuple[Path, list[Path]]:
    if kind == "process":
        bank = gate["banks"][BANK_NAMES[instrument]]
        require(type(bank.get("training_parent_overlap")) is int and bank["training_parent_overlap"] == 0,
                "BLOCKED_TRAIN_EVAL_PARENT_OVERLAP: process bank is not disjoint")
        views = Path(bank["path"]).resolve()
        require(sha(views) == bank["sha256"], "process views differ from frozen build gate")
        return views, []
    if kind == "retention":
        return RAB / "data/monitor_probe.jsonl", [
            RAB / "evidence/retention-design-v1/PROBE_LABEL_AUDIT.json",
            RAB / f"outputs/retention-{digest[:8]}/B28/retention.json"]
    suite = PILOT / "Rineya_Long_Context_Demo_v1/data"
    return suite / "all_prompts.jsonl", [suite / "all_answer_key.jsonl"]


def verify_result(raw: dict, kind: str, label: str, arm: Path,
                  views_sha: str, expected_ids: set[str], digest: str, worker: Path,
                  weight_sha: str) -> dict:
    require(raw.get("label") == label, "raw result label differs")
    recorded_dir = raw.get({"process": "model_dir", "retention": "target",
                            "long_context": "model_directory"}[kind])
    require(isinstance(recorded_dir, str) and recorded_dir, "raw result has no arm directory")
    if kind == "retention":
        require(recorded_dir.startswith("hf:"), "retention must use the verified hf: materialization loader")
        recorded_dir = recorded_dir[3:]
        identity = raw.get("hf_candidate_identity")
        require(isinstance(identity, dict) and identity.get("weight_sha256") == weight_sha,
                "retention HF loader weight differs from live before/after arm identity")
    path = Path(recorded_dir)
    require((path if path.is_absolute() else PILOT / path).resolve() == arm, "raw result arm differs")
    field = "reference_digest" if kind == "retention" else "implementation_digest"
    require(raw.get(field) == digest, "raw scorer digest differs from before/after implementation")
    extra = {}
    if kind == "process":
        require(raw.get("views_sha256") == views_sha, "raw process views hash differs")
        require(raw.get("checkpoint") is None, "probe unexpectedly applied a checkpoint")
        predictions = raw.get("predictions")
        require(isinstance(predictions, list), "process predictions missing")
        ids = [r.get("id") for r in predictions]
        require(len(ids) == len(set(ids)) and set(ids) == expected_ids, "incomplete process prediction IDs")
        require(raw.get("views") == len(expected_ids), "process view count differs")
        require(raw.get("max_new_tokens") == 128, "process generation limit differs from registered 128")
    elif kind == "retention":
        require(raw.get("probe_sha256") == views_sha, "raw retention panel hash differs")
        require(raw.get("probe_rows") == len(expected_ids), "retention panel count differs")
        prediction_path = worker / "retention_predictions.json"
        saved = read(prediction_path)
        predictions = saved.get("predictions")
        require(isinstance(predictions, list) and len(predictions) == len(expected_ids),
                "wrapper did not preserve every retention prediction")
        ids = [r.get("id") for r in predictions]
        require(len(ids) == len(set(ids)) and set(ids) == expected_ids, "incomplete retention prediction IDs")
        extra = {"retention_predictions_path": str(prediction_path),
                 "retention_predictions_sha256": sha(prediction_path)}
    else:
        require(raw.get("checkpoint_applied") is None, "long-context probe unexpectedly applied a checkpoint")
        require(set(raw.get("predictions", {})) == expected_ids and
                set(raw.get("per_row_correct", {})) == expected_ids,
                "incomplete long-context prediction IDs")
        require(raw.get("rows_total") == len(expected_ids), "long-context row count differs")
    return extra


def run(args: argparse.Namespace) -> int:
    out = args.out.absolute()
    # No failure output is allowed to alter an earlier attempt.
    require(not out.exists() and not out.is_symlink(), "--out must be a fresh attempt directory")
    out.mkdir(parents=True, exist_ok=False)
    phase, code = "preflight", None
    try:
        require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", args.label) is not None,
                "label must be a simple bounded filename component")
        arm, gate_path = args.arm.resolve(), args.build_gate.resolve()
        gate_hash, gate = sha(gate_path), read(gate_path)
        require(str(gate.get("status", "")).startswith("PASS"), "build gate is not PASS")
        regular(PYTHON)
        script = PILOT / "experiments/rab001_exploration" / f"{args.kind}_probe.py"
        expected_script = gate["probe_script_sha256"][args.kind]
        require(sha(script) == expected_script, "probe script differs from build gate")
        require(sha(WRAPPER) == gate["wrapper_sha256"], "supervised wrapper differs from build gate")
        if "supervisor_sha256" in gate:
            require(sha(SUPERVISOR_CODE) == gate["supervisor_sha256"], "supervisor differs from build gate")
        digest = implementation_digest()
        if "implementation_digest" in gate:
            require(gate["implementation_digest"] == digest, "scorer differs from build gate")
        views, additional = inputs(args.kind, args.instrument, gate, digest)
        expected_ids = row_ids(views)
        if args.kind == "long_context":
            require(row_ids(additional[0]) == expected_ids, "long-context answer key IDs differ")
        views_hash = sha(views)
        additional_hashes = {str(p): sha(p) for p in additional}
        before = arm_snapshot(arm, gate)
        code_hashes = {str(p): sha(p) for p in (script, WRAPPER, SUPERVISOR_CODE, Path(__file__))}
        legacy = RAB / "evidence/retention-design-v1" / f"{PREFIXES[args.kind]}_{args.label}.json"
        require(not legacy.exists() and not legacy.is_symlink(), "legacy label already has a result")
        worker = out / "worker"
        arguments = ([args.label, "hf:" + str(arm), "cuda"] if args.kind == "retention" else
                     ["--model", str(arm), "--label", args.label, "--device", "cuda"])
        if args.kind == "process":
            require(gate.get("process_max_new_tokens", 128) == 128,
                    "build gate process generation limit differs from registered 128")
            arguments += ["--instrument", args.instrument, "--max-new-tokens", "128"]
        command = [str(PYTHON), str(WRAPPER), "--kind", args.kind, "--out", str(worker),
                   "--timeout", "900", "--expected-script-sha", expected_script, "--", *arguments]
        write(out / "PRECHECK.json", {"schema": 1, "status": "PRECHECK_PASSED",
              "created_utc": datetime.now(timezone.utc).isoformat(), "kind": args.kind,
              "arm_dir": str(arm), "arm_before": before, "build_gate_path": str(gate_path),
              "build_gate_sha256": gate_hash, "implementation_digest": digest,
              "views_path": str(views), "views_sha256": views_hash,
              "additional_input_sha256": additional_hashes, "code_sha256": code_hashes,
              "command": command, "cwd": str(PILOT), "quality_claim": False})
        phase = "supervised_probe"
        options = {"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}
        with (out / "launcher.log").open("xb") as log:
            code = subprocess.run(command, cwd=PILOT, stdout=log, stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, check=False, **options).returncode
        phase = "verify_terminal"
        require(code == 0, "supervised wrapper returned nonzero; see retained worker receipts")
        supervisor_path, terminal_path = worker / "supervisor.json", worker / "probe_terminal.json"
        supervisor, terminal = read(supervisor_path), read(terminal_path)
        require(supervisor.get("status") == "EXITED" and type(supervisor.get("returncode")) is int
                and supervisor["returncode"] == 0, "supervisor did not establish successful exit")
        require(terminal.get("status") == "COMPLETED" and terminal.get("complete") is True
                and type(terminal.get("system_exit_code")) is int and terminal["system_exit_code"] == 0
                and terminal.get("failure") is None and terminal.get("script_unchanged") is True
                and terminal.get("script_hash_error") is None
                and terminal.get("script_sha256_after") == expected_script,
                "probe terminal is incomplete, failed, or changed")
        require(Path(terminal.get("result", "")).resolve() == legacy.resolve(), "terminal points to unexpected raw result")
        phase = "verify_identity_after"
        after = arm_snapshot(arm, gate)
        require(after == before, "arm identity changed during probe")
        require(sha(gate_path) == gate_hash, "build gate changed during probe")
        require({path: sha(Path(path)) for path in code_hashes} == code_hashes,
                "probe/wrapper/supervisor/runner code changed during execution")
        require(implementation_digest() == digest, "scorer implementation changed during execution")
        require(sha(views) == views_hash, "instrument views changed during probe")
        require({path: sha(Path(path)) for path in additional_hashes} == additional_hashes,
                "additional instrument inputs changed during probe")
        phase = "seal_result"
        copied = out / "result.json"
        result_hash = copy_new(legacy, copied)
        raw = read(copied)
        if args.kind == "process":
            require(raw.get("record") == "PROCESS_TRANSITION_PROBE" and
                    raw.get("instrument") == args.instrument, "raw process record/instrument differs")
        extra = verify_result(raw, args.kind, args.label, arm, views_hash, expected_ids, digest, worker,
                              before["weight_sha256"])
        require(sha(copied) == sha(legacy) == result_hash, "raw result changed before binding")
        receipt = {"schema": 1, "status": "COMPLETE", "complete": True, "returncode": 0,
            "kind": args.kind, "label": args.label, "created_utc": datetime.now(timezone.utc).isoformat(),
            "raw_result_path": str(copied), "raw_result_sha256": result_hash,
            "original_raw_result_path": str(legacy), "original_raw_result_sha256": result_hash,
            "arm_dir": str(arm), "arm_receipt_path": str(arm / "ARM.json"),
            "arm_receipt_sha256": before["arm_receipt_sha256"],
            "materialization_path": str(arm / "MATERIALIZATION.json"),
            "materialization_sha256": before["materialization_sha256"],
            "weight_path": str(arm / "model.safetensors"), "weight_sha256": before["weight_sha256"],
            "weight_sha256_before": before["weight_sha256"], "weight_sha256_after": after["weight_sha256"],
            "views_path": str(views), "views_sha256": views_hash, "implementation_digest": digest,
            "script_path": str(script), "script_sha256": expected_script,
            "wrapper_path": str(WRAPPER), "wrapper_sha256": code_hashes[str(WRAPPER)],
            "supervisor_path": str(supervisor_path), "supervisor_sha256": sha(supervisor_path),
            "terminal_path": str(terminal_path), "terminal_sha256": sha(terminal_path),
            "supervisor_code_path": str(SUPERVISOR_CODE), "supervisor_code_sha256": code_hashes[str(SUPERVISOR_CODE)],
            "sidecar_sha256_before": before["sidecar_sha256"], "sidecar_sha256_after": after["sidecar_sha256"],
            "source_sidecar_sha256_before": before["source_sidecar_sha256"],
            "source_sidecar_sha256_after": after["source_sidecar_sha256"],
            "build_gate_path": str(gate_path), "build_gate_sha256": gate_hash,
            "training_snapshot_before": before["training"], "training_snapshot_after": after["training"],
            "additional_input_sha256_before": additional_hashes, "additional_input_sha256_after": additional_hashes,
            "quality_claim": False, "release_permitted": False, **extra}
        write(out / "PROBE_WEIGHT_BINDING.json", receipt)
        print(json.dumps({"status": "COMPLETE", "attempt": str(out), "kind": args.kind,
                          "binding": str(out / "PROBE_WEIGHT_BINDING.json")}), flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        failure = {"schema": 1, "status": "FAILED", "complete": False,
                   "returncode": code if type(code) is int and code != 0 else 1,
                   "phase": phase, "failure_type": type(exc).__name__, "message": str(exc),
                   "quality_claim": False, "release_permitted": False}
        write(out / "FAILURE.json", failure)
        print(json.dumps({"status": "FAILED", "attempt": str(out), "phase": phase,
                          "message": str(exc)}), file=sys.stderr, flush=True)
        return failure["returncode"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--kind", choices=tuple(PREFIXES), required=True)
    parser.add_argument("--instrument", choices=tuple(BANK_NAMES), default="process-transition-probe-001")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--build-gate", type=Path, required=True)
    return run(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
