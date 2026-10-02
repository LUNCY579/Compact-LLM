"""Frozen DEV -> selection lock -> single-use fresh confirmation state machine."""
from __future__ import annotations

import math
from pathlib import Path
from .common import code_hash, digest, exclusive, read, require, signed, verify, write_new
from .diagnostics import assert_disjoint, build_suite, suite_hash, validate_suite
from .metrics import eligible, grade, paired_comparison, summarize, trajectory
from .registry import audit_training_overlap, freeze_registry, verify_registry

DEFAULT_POLICY = {"min_clean_accuracy": .5, "max_retention_damage_fraction": .05,
                  "min_reference_solved": 8, "min_acquisition_delta": 0.0}
DEFAULT_NATIVE = {"device": "cpu", "dtype": "float32", "cpu_threads": 2,
                  "prompt_format": "plain", "max_input_tokens": 512, "max_new_tokens": 16,
                  "min_free_ram_gib": 7.4, "min_free_vram_gib": 7.4, "min_free_disk_gib": 1.0}


def validate_config(policy: dict, native: dict) -> None:
    require(set(policy) == set(DEFAULT_POLICY), "Unknown/missing selection policy fields")
    for key in ("min_clean_accuracy", "max_retention_damage_fraction", "min_acquisition_delta"):
        require(type(policy[key]) in (int, float) and math.isfinite(policy[key]) and 0 <= policy[key] <= 1, "Invalid policy threshold")
    require(type(policy["min_reference_solved"]) is int and policy["min_reference_solved"] > 0, "Invalid reference support")
    require(set(native) == set(DEFAULT_NATIVE), "Unknown/missing runtime fields")
    require(native["device"] in {"cpu", "cuda"}, "Only explicit CPU or CUDA:0 admitted")
    require(native["dtype"] in {"float32", "bfloat16"}, "Unsupported dtype")
    require(native["device"] != "cpu" or native["dtype"] == "float32", "CPU requires float32")
    require(native["prompt_format"] in {"plain", "chat"}, "Unsupported prompt format")
    for key, limit in (("cpu_threads", 8), ("max_input_tokens", 2048), ("max_new_tokens", 64)):
        require(type(native[key]) is int and 0 < native[key] <= limit, "Invalid compute budget")
    for key in ("min_free_ram_gib", "min_free_vram_gib", "min_free_disk_gib"):
        require(type(native[key]) in (int, float) and math.isfinite(native[key]) and native[key] >= DEFAULT_NATIVE[key], "Resource guards may not be weakened")


def initialize(run: Path, spec: dict | None = None, *, mechanical: bool = False,
               policy: dict | None = None, seed: int = 17, per_family: int = 24) -> dict:
    require(not run.exists(), "Run directory already exists; choose a fresh identity")
    policy = dict(DEFAULT_POLICY if policy is None else policy)
    native = dict(DEFAULT_NATIVE if spec is None else spec.get("runtime", DEFAULT_NATIVE))
    validate_config(policy, native)
    rows = build_suite("dev", seed, per_family)
    if mechanical:
        require(spec is None, "Mechanical fixtures cannot bind real snapshots")
        registry = {"trajectory_id": "MECHANICAL_FIXTURE_NOT_A_MODEL", "training": {"prompt_hashes": [], "parent_ids": []},
                    "checkpoints": [{"id": name, "step": step, "kind": "mechanical", "asset": {"content_sha256": digest(name)}}
                                    for name, step in (("reference", 0), ("intermediate", 100), ("final", 200))]}
    else:
        require(spec is not None, "Native run needs an explicit snapshot specification")
        registry = freeze_registry(spec)
        audit_training_overlap(rows, registry)
    manifest = signed({"schema": 1, "experiment": "GD-001", "code_sha256": code_hash(),
                       "evidence_kind": "mechanical_fixture" if mechanical else "native_pilot",
                       "registry": registry, "policy": policy, "runtime": native,
                       "dev_seed": seed, "per_family": per_family, "dev_suite_sha256": suite_hash(rows),
                       "confirm_policy": "single-use; fresh template families; no existing sealed banks",
                       "permits_model_promotion": False})
    run.mkdir(parents=True)
    write_new(run / "manifest.json", manifest)
    write_new(run / "development_suite.json", rows)
    return manifest


def state(run: Path) -> tuple[dict, list[dict]]:
    manifest = verify(read(run / "manifest.json"))
    require(manifest["experiment"] == "GD-001" and manifest["schema"] == 1, "Unknown manifest")
    require(manifest["code_sha256"] == code_hash(), "Code changed after registration; start a new run")
    validate_config(manifest["policy"], manifest["runtime"])
    rows = read(run / "development_suite.json")
    validate_suite(rows, "dev")
    require(suite_hash(rows) == manifest["dev_suite_sha256"], "Development data changed")
    return manifest, rows


def fixture_prediction(row: dict, checkpoint: dict) -> dict:
    """Deliberately gold-aware fixture; impossible to label as native evidence."""
    index = int(row["parent_id"].split(":")[-1])
    stage = checkpoint["step"]
    if row["family"] == "retention":
        correct = True
    elif row["condition"] == "clean":
        correct = stage == 200 or index % (4 if stage == 0 else 10) != 0
    else:
        correct = index % (2 if stage == 100 else 4) == 1
    return {"id": row["id"], "text": row["answer"] if correct else row["distractor"],
            "gold_logp": -.1 if correct else -4.0, "distractor_logp": -4.0 if correct else -.1,
            "gold_tokens": 1, "distractor_tokens": 1, "stop_reason": "eos"}


def result_path(run: Path, split: str, checkpoint: dict) -> Path:
    return run / split / (checkpoint["id"] + ".json")


def load_result(run: Path, manifest: dict, rows: list[dict], split: str, checkpoint: dict) -> dict:
    result = verify(read(result_path(run, split, checkpoint)))
    for key, value in (("manifest_sha256", manifest["sha256"]), ("suite_sha256", suite_hash(rows)),
                       ("checkpoint_id", checkpoint["id"]), ("checkpoint_sha256", checkpoint["asset"]["content_sha256"]),
                       ("evidence_kind", manifest["evidence_kind"]), ("split", split)):
        require(result.get(key) == value, f"Result provenance mismatch: {key}")
    grade(rows, result["predictions"])
    return result


def invariant_runtime(runtime: dict) -> dict:
    return {k: v for k, v in runtime.items() if k != "parameters_with_adapter"}


def evaluate(run: Path, manifest: dict, rows: list[dict], split: str,
             checkpoints: list[dict], allow_native: bool) -> list[dict]:
    require(split in {"dev", "confirm"}, "Unknown evaluation split")
    native = manifest["evidence_kind"] == "native_pilot"
    require(not native or allow_native, "Native inference requires --allow-native; no implicit model loads")
    if native:
        from .native import NativeEvaluator, device_lock
        verify_registry(manifest["registry"])
        evaluator = NativeEvaluator(manifest["registry"], manifest["runtime"])
    results = []
    # The first development reference establishes the runtime contract for confirmation.
    existing = run / "dev" / (manifest["registry"]["checkpoints"][0]["id"] + ".json")
    runtime_contract = invariant_runtime(verify(read(existing))["runtime"]) if existing.exists() else None
    def one(checkpoint):
        nonlocal runtime_contract
        destination = result_path(run, split, checkpoint)
        if destination.exists():
            result = load_result(run, manifest, rows, split, checkpoint)
        else:
            if native:
                try:
                    runtime = evaluator.load(checkpoint)
                    predictions = [evaluator.predict(r) for r in rows]
                    resources = evaluator.resources()
                finally:
                    evaluator.close()
            else:
                resources = {"measured_native_resources": False}
                runtime = {"backend": "gold-aware-mechanical-fixture-NOT-A-MODEL"}
                predictions = [fixture_prediction(r, checkpoint) for r in rows]
            grade(rows, predictions)
            result = signed({"manifest_sha256": manifest["sha256"], "suite_sha256": suite_hash(rows),
                             "checkpoint_id": checkpoint["id"], "checkpoint_sha256": checkpoint["asset"]["content_sha256"],
                             "evidence_kind": manifest["evidence_kind"], "split": split,
                             "runtime": runtime, "resources": resources, "predictions": predictions})
            current = invariant_runtime(runtime)
            require(runtime_contract is None or current == runtime_contract, "Runtime changed within campaign")
            write_new(destination, result)
        current = invariant_runtime(result["runtime"])
        require(runtime_contract is None or current == runtime_contract, "Mixed runtime evidence")
        runtime_contract = current
        results.append(result)
    if native:
        with device_lock(manifest["runtime"]):
            for checkpoint in checkpoints:
                one(checkpoint)
        verify_registry(manifest["registry"])
    else:
        for checkpoint in checkpoints:
            one(checkpoint)
    return results


def evaluate_development(run: Path, allow_native: bool = False) -> list[dict]:
    with exclusive(run / "BUSY.lock"):
        manifest, rows = state(run)
        require(not (run / "selection.json").exists(), "Selection already locked; development cannot be reopened")
        return evaluate(run, manifest, rows, "dev", manifest["registry"]["checkpoints"], allow_native)


def select(run: Path) -> dict:
    with exclusive(run / "BUSY.lock"):
        manifest, rows = state(run)
        require(not (run / "selection.json").exists(), "Selection is immutable")
        require(not (run / "confirmation.STARTED.json").exists(), "Confirmation already spent")
        checkpoints = manifest["registry"]["checkpoints"]
        results = [load_result(run, manifest, rows, "dev", c) for c in checkpoints]
        require(all(invariant_runtime(r["runtime"]) == invariant_runtime(results[0]["runtime"]) for r in results), "Mixed runtime evidence")
        reference = results[0]["predictions"]
        summaries = {c["id"]: summarize(rows, r["predictions"], reference) for c, r in zip(checkpoints, results)}
        baseline = summaries[checkpoints[0]["id"]]
        gates = {c["id"]: eligible(summaries[c["id"]], baseline, manifest["policy"])[1] for c in checkpoints[1:]}
        admissible = [c for c in checkpoints[1:] if not gates[c["id"]]]
        chosen = {"reference": checkpoints[0]["id"], "final": checkpoints[-1]["id"]}
        if admissible:
            for label, score in (("ordinary", "ordinary_macro"), ("generalization", "robust_macro")):
                chosen[label] = min(admissible, key=lambda c: (-summaries[c["id"]][score], c["step"], c["id"]))["id"]
        selection = signed({"manifest_sha256": manifest["sha256"], "status": "LOCKED" if admissible else "BLOCKED_NO_ELIGIBLE_CHECKPOINT",
                            "chosen": chosen, "development_receipts": {r["checkpoint_id"]: r["sha256"] for r in results},
                            "evidence_kind": manifest["evidence_kind"], "tie_break": "earlier step, then ID",
                            "policy_sha256": digest(manifest["policy"]), "permits_model_promotion": False})
        write_new(run / "development_report.json", {"evidence_kind": manifest["evidence_kind"], "summaries": summaries,
                  "gate_failures": gates, "trajectory": trajectory(rows, [r["predictions"] for r in results])})
        write_new(run / "selection.json", selection)
        return selection


def confirm(run: Path, seed: int, *, allow_confirmation: bool = False, allow_native: bool = False) -> dict:
    require(allow_confirmation, "Fresh confirmation access requires --allow-confirmation")
    with exclusive(run / "BUSY.lock"):
        manifest, dev = state(run)
        require(manifest["evidence_kind"] != "native_pilot" or allow_native, "Native inference not authorized")
        selection = verify(read(run / "selection.json"))
        require(selection["manifest_sha256"] == manifest["sha256"] and selection["policy_sha256"] == digest(manifest["policy"]), "Selection is not bound to this run")
        require(selection["status"] == "LOCKED", "No admissible checkpoint; confirmation blocked")
        require(type(seed) is int and seed >= 0 and seed != manifest["dev_seed"], "Supply a distinct independent confirmation seed")
        checkpoints = manifest["registry"]["checkpoints"]
        for c in checkpoints:
            result = load_result(run, manifest, dev, "dev", c)
            require(result["sha256"] == selection["development_receipts"][c["id"]], "Development receipt changed after selection")
        require(not (run / "confirmation.STARTED.json").exists() and not (run / "confirmation_suite.json").exists() and not (run / "confirm").exists(), "Confirmation already spent or partially exposed; no retry")
        # Mark ANY_READ before generating/accessing a single confirmation item.
        write_new(run / "confirmation.STARTED.json", signed({"selection_sha256": selection["sha256"], "seed": seed,
                  "meaning": "Confirmation spent on any subsequent outcome, including error or interruption"}))
        rows = build_suite("confirm", seed, manifest["per_family"])
        assert_disjoint(dev, rows)
        audit_training_overlap(rows, manifest["registry"])
        write_new(run / "confirmation_suite.json", rows)
        chosen_ids = set(selection["chosen"].values())
        selected = [c for c in checkpoints if c["id"] in chosen_ids]
        require(len(selected) == len(chosen_ids), "Unknown selected checkpoint")
        results = evaluate(run, manifest, rows, "confirm", selected, allow_native)
        predictions = {r["checkpoint_id"]: r["predictions"] for r in results}
        choices = selection["chosen"]
        reference = predictions[choices["reference"]]
        summaries = {name: summarize(rows, p, reference) for name, p in predictions.items()}
        comparisons = {label + "_minus_final": paired_comparison(rows, predictions[choices[label]], predictions[choices["final"]])
                       for label in ("generalization", "ordinary")}
        report = signed({"evidence_kind": manifest["evidence_kind"], "selection_sha256": selection["sha256"],
                         "suite_sha256": suite_hash(rows), "chosen": choices, "summaries": summaries,
                         "comparisons": comparisons, "primary_contrast": "generalization_minus_final",
                         "generalization_checkpoint_gate_failures": eligible(summaries[choices["generalization"]], summaries[choices["reference"]], manifest["policy"])[1],
                         "confirmation_receipts": {r["checkpoint_id"]: r["sha256"] for r in results},
                         "permits_model_promotion": False,
                         "limitations": ["fresh synthetic diagnostics, not a broad capability benchmark", "single training trajectory; no cross-seed replication", "exact overlap audit does not establish semantic decontamination", "behavioral changes do not identify a causal mechanism"]})
        write_new(run / "confirmation_report.json", report)
        return report
