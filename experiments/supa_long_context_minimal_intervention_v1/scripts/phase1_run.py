#!/usr/bin/env python3
"""Phase 1: tiny late-attention adapter, three arms, one frozen evaluation suite.

    Arm A  frozen canonical Supa-A1.1, no training
    Arm B  {q,k,v,o}_proj LoRA on layers 17,18,19
    Arm C  the same, NoRA-init

Arms B and C share dataset, data order, seed, rank, optimizer, LR, scheduler,
context length, accumulation, step count and precision. Only the adapter
initialization differs.

Evaluation is at FULL CONTEXT for every arm -- this measures whether the
*checkpoint* improved, which is a different question from Phase 0's.

    python scripts/phase1_run.py --arms A --smoke
    python scripts/phase1_run.py --arms B --max-steps 300
    python scripts/phase1_run.py --combine

Layers 0, 1 and 2 are never trainable; the runner asserts this rather than
trusting the config.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import sys
import uuid
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from slcmi.adapters import (  # noqa: E402
    DEFAULT_ALPHA,
    DEFAULT_DROPOUT,
    DEFAULT_LAYERS,
    DEFAULT_RANK,
    PROTECTED_LAYERS,
    build_adapter,
    count_trainable,
    verify_adapter,
)
from slcmi.evaluation import (  # noqa: E402
    MAX_NEW_TOKENS,
    SEED,
    evaluate_rows,
    is_malformed,
    load_model,
    normalize,
    score,
)
from slcmi.preservation import (  # noqa: E402
    CORPUS_DIR,
    PRESERVATION_CLASSES,
    PRESERVATION_PER_CLASS,
    build_preservation_set,
    evaluate_preservation,
    score_preservation,
)
from slcmi.provenance import (  # noqa: E402
    CORES,
    DEMO_SUITE,
    FROZEN_HASH_ALGORITHM,
    ExclusiveFileLock,
    Run,
    append_jsonl,
    checkpoint_files,
    frozen_parameter_hash,
    read_jsonl,
    read_jsonl_prefix,
    sha256,
    source_hashes,
    trainable_parameter_receipt,
    write_json,
    write_jsonl,
)
from slcmi.stats import (  # noqa: E402
    mcnemar_exact,
    paired_bootstrap_delta,
    relative_error_reduction,
    summarise,
)
from slcmi.training import train  # noqa: E402
from slcmi.checkpoint_resume import fingerprint, validate_progress  # noqa: E402
from slcmi.training_data import (  # noqa: E402
    LCR_SFT,
    build_dataset,
    load_packs,
    sample_curriculum,
)

MAX_LENGTH = 3840
REQUIRED_ARMS = ("A", "B", "C")
PUBLISHED_BASELINE = {
    "a11": {"showcase": 45, "stability_test": 125, "all": 170},
    "a1": {"showcase": 43, "stability_test": 125, "all": 168},
}
TRAIN_PACKS = 600
DEV_PACKS = 40
SMOKE_TRAIN_PACKS = 8
SMOKE_DEV_PACKS = 4
SMOKE_STEPS = 6


def target_items(smoke: bool) -> list[dict]:
    """The long-context target set: the demo suite at FULL context."""
    rows = read_jsonl(DEMO_SUITE)
    if smoke:
        rows = rows[:32]
    return [
        {
            "example_id": r["id"],
            "split": r["split"],
            "variant": r["variant"],
            "task_type": r["task_type"],
            "answerable": r["answerable"],
            "condition": "A_full",
            "context": r["context"],
            "question": r["question"],
            "accepted_answers": r["accepted_answers"],
        }
        for r in rows
    ]


def run_arm(
    arm: str,
    *,
    core: dict,
    device: str,
    smoke: bool,
    max_steps: int,
    grad_accum: int,
    lr: float,
    train_set,
    dev_set,
    targets,
    preservation,
    out: Path,
    run: Run,
    dataset_fingerprint: dict,
    arm_out: Path,
    identity: dict,
    resume_from_checkpoint: bool = False,
) -> dict:
    print(f"\n=== ARM {arm} ===")
    run.model_load_started()
    tokenizer, model = load_model(core["path"], device, dtype=core["dtype"], gate=True)
    run.model_load_completed()
    info: dict = {"arm": arm, "loaded_core_path": str(core["path"]),
                  "loaded_dtype": str(next(model.parameters()).dtype).removeprefix("torch."),
                  "dataset_fingerprint": dataset_fingerprint}
    run.resource_extra.update(trainable_params=0, optimizer_steps_attempted=0,
                              optimizer_steps_applied=0, steps_skipped_by_scaler=0,
                              steps_skipped_by_safety_check=0, examples_consumed=0)
    assert info["loaded_dtype"] == core["dtype"], "loaded dtype differs from selected core"

    if arm == "A":
        info["training"] = None
        info["adapter"] = {"kind": "none", **count_trainable(model)}
        frozen_before = frozen_parameter_hash(model)
        write_json(out / "frozen_parameters_before.json", {"algorithm": FROZEN_HASH_ALGORITHM, "sha256": frozen_before})
        seal_identity(arm_out, identity, "frozen_parameter_sha256", frozen_before)
        initializer = trainable_parameter_receipt(model)
        write_json(out / "adapter_initialization.json", initializer)
        seal_initialization(arm_out, identity, initializer)
    else:
        kind = "lora" if arm == "B" else "nora"
        probe = tokenizer("CONTEXT\nprobe.\n\nQUESTION\nwhat?\n\nANSWER",
                          return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            reference_logits = model(**probe).logits.float().clone()

        model, adapter_info = build_adapter(
            model, kind, DEFAULT_LAYERS, DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT, SEED
        )
        # This is the existing loop's adapter dtype, made explicit before
        # fingerprinting the actual initializer that the loop will receive.
        for parameter in model.parameters():
            if parameter.requires_grad:
                parameter.data = parameter.data.float()
        checks = verify_adapter(model, kind, DEFAULT_LAYERS, reference_logits, probe)
        print(f"  trainable {adapter_info['trainable_params']:,} params "
              f"({adapter_info['trainable_fraction'] * 100:.4f}%) on layers {DEFAULT_LAYERS}")
        print(f"  invariant checks: {'PASS' if checks['all_passed'] else 'FAIL'}")
        if not checks["all_passed"]:
            raise RuntimeError(f"adapter invariants failed: {checks}")
        assert not (set(checks["trained_layers"]) & set(PROTECTED_LAYERS)), "stem touched"

        # Serialisation and merge parity are verified once, on CPU, by
        # scripts/phase1_verify_adapters.py. Repeating the deep-copy here would
        # put a second 596M-parameter model on a GPU that other work already
        # keeps above 7 GB.

        info["adapter"] = {**adapter_info, "kind": kind, "checks": checks}
        run.resource_extra["trainable_params"] = adapter_info["trainable_params"]
        frozen_before = frozen_parameter_hash(model)
        write_json(out / "frozen_parameters_before.json", {"algorithm": FROZEN_HASH_ALGORITHM, "sha256": frozen_before})
        seal_identity(arm_out, identity, "frozen_parameter_sha256", frozen_before)
        initializer = trainable_parameter_receipt(model)
        write_json(out / "adapter_initialization.json", initializer)
        seal_initialization(arm_out, identity, initializer)
        def record_progress(record: dict) -> None:
            # Flush each attempted update, so an OOM/interruption leaves evidence.
            append_jsonl(out / "training_progress.jsonl", record)
            if record.get("event") == "optimizer_attempt":
                initial = run.manifest["resumed_from_committed_training"]
                cumulative = {
                    "optimizer_steps_attempted": record["attempt"],
                    "optimizer_steps_applied": record["optimizer_updates_applied"],
                    "steps_skipped_by_scaler": record.get("steps_skipped_by_scaler_total", 0),
                    "steps_skipped_by_safety_check": record.get("steps_skipped_by_safety_check_total", 0),
                    "examples_consumed": record["examples_consumed"],
                }
                run.resource_extra.update({key: value - initial[key] for key, value in cumulative.items()})

        # A missing latest.pt is recoverable only before this durable barrier.
        # Once train was entered, lack of a committed state is ambiguous.
        write_json(out / "training_entry.json", {"run_id": identity["run_id"],
                   "checkpoint_fingerprints": identity["checkpoint_fingerprints"],
                   "total_max_steps": max_steps, "resume_from_checkpoint": resume_from_checkpoint,
                   "entered_utc": datetime.now(timezone.utc).isoformat()})
        info["training"] = train(
            model,
            train_set,
            device=device,
            max_steps=max_steps,
            grad_accum=grad_accum,
            lr=lr,
            seed=SEED,
            eval_set=dev_set,
            eval_every=max(1, max_steps // 4),
            progress_callback=record_progress,
            checkpoint_dir=arm_out / "checkpoints",
            checkpoint_fingerprints=identity["checkpoint_fingerprints"],
            resume_from_checkpoint=resume_from_checkpoint,
            gradient_checkpointing=True,
        )
        model.eval()

    assert model.config.abstain_logit_bias, "abstention gate must be on for evaluation"

    print("  target eval (full context)...")
    target_records = evaluate_rows(tokenizer, model, targets, device=device)
    write_jsonl(out / f"arm_{arm}_target.jsonl", target_records, overwrite=True)
    print("  preservation eval...")
    preservation_records = evaluate_preservation(tokenizer, model, preservation, device=device)
    frozen_after = frozen_parameter_hash(model)
    info["frozen_base"] = {
        "algorithm": FROZEN_HASH_ALGORITHM, "before": frozen_before,
        "after": frozen_after, "unchanged": frozen_before == frozen_after,
    }
    if frozen_before != frozen_after:
        raise RuntimeError("frozen base parameters changed during the arm")
    info["evaluation_gate_on"] = bool(model.config.abstain_logit_bias)
    info["checkpoint_fingerprints"] = identity["checkpoint_fingerprints"]
    info["run_id"] = identity["run_id"]
    info["segment_id"] = run.manifest["segment_id"]

    write_jsonl(out / f"arm_{arm}_target.jsonl", target_records, overwrite=True)
    write_jsonl(out / f"arm_{arm}_preservation.jsonl", preservation_records, overwrite=True)
    write_jsonl(out / "per_example.jsonl", [
        *({"evaluation": "target", "arm": arm, **r} for r in target_records),
        *({"evaluation": "preservation", "arm": arm, **r} for r in preservation_records),
    ])

    info["target"] = summarise([r["correct"] for r in target_records])
    info["preservation"] = {
        "matched": summarise([r["matched"] for r in preservation_records]),
        "exact": summarise([r["exact"] for r in preservation_records]),
        "mean_response_words": round(
            statistics.mean(r["response_words"] for r in preservation_records), 2
        ),
    }
    info["target_malformed_rate"] = round(
        sum(r["malformed"] for r in target_records) / len(target_records), 4
    )
    print(f"  target {info['target']['correct']}/{info['target']['n']} "
          f"= {info['target']['accuracy_points']:.1f}%   "
          f"preservation {info['preservation']['matched']['accuracy_points']:.1f}%")

    write_json(out / f"arm_{arm}_info.json", info, overwrite=True)
    if arm != "A":
        model.save_pretrained(out / "adapter")
    # The caller reconciles all final physical counters from the committed
    # transaction, including a commit whose progress callback never arrived.
    run.resource_extra["trainable_params"] = info["adapter"]["trainable_params"]

    # Strip the adapter modules out of the base before dropping references,
    # otherwise the LoRA layers keep the base graph alive.
    if arm != "A":
        try:
            model = model.unload()
        except Exception:
            model = model.get_base_model()
    del model, tokenizer
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception as exc:
        run.metric_errors["cuda_cleanup"] = f"{type(exc).__name__}: {exc}"

    return info, target_records, preservation_records


def decide(arms: dict, target_by_arm: dict, preservation_by_arm: dict) -> dict:
    """Apply the preregistered Phase 1 gate."""
    missing = [arm for arm in REQUIRED_ARMS if arm not in arms]
    if missing:
        return {"verdict": "INCOMPLETE", "missing_arms": missing,
                "winner": None, "phase_2_licensed": False,
                "reason": "all three completed arms are required for a decision"}

    base_ids = [r["example_id"] for r in target_by_arm["A"]]
    base_scores = [r["correct"] for r in target_by_arm["A"]]
    base_acc = sum(base_scores) / len(base_scores)
    base_pres = [r["matched"] for r in preservation_by_arm["A"]]
    base_pres_acc = sum(base_pres) / len(base_pres)

    verdicts = {}
    for arm in ("B", "C"):
        if arm not in arms:
            continue
        by_id = {r["example_id"]: r["correct"] for r in target_by_arm[arm]}
        scores = [by_id[i] for i in base_ids]
        acc = sum(scores) / len(scores)
        pres = [r["matched"] for r in preservation_by_arm[arm]]
        pres_acc = sum(pres) / len(pres)

        gain_points = 100 * (acc - base_acc)
        rer = relative_error_reduction(base_acc, acc)
        regression_points = 100 * (base_pres_acc - pres_acc)
        exact_base = Fraction(sum(base_scores), len(base_scores))
        exact_acc = Fraction(sum(scores), len(scores))
        exact_gain = 100 * (exact_acc - exact_base)
        exact_rer = (exact_acc - exact_base) / (1 - exact_base) if exact_base < 1 else Fraction(0)
        exact_regression = 100 * (Fraction(sum(base_pres), len(base_pres)) - Fraction(sum(pres), len(pres)))
        meets_gain = exact_gain >= 2 or exact_rer >= Fraction(1, 20)
        meets_preservation = exact_regression <= 1
        positive = meets_gain and meets_preservation

        verdicts[arm] = {
            "target_accuracy_points": round(100 * acc, 2),
            "target_gain_points": round(gain_points, 2),
            "relative_error_reduction": round(rer, 4),
            "preservation_accuracy_points": round(100 * pres_acc, 2),
            "preservation_regression_points": round(regression_points, 2),
            "meets_gain_gate": bool(meets_gain),
            "meets_preservation_gate": bool(meets_preservation),
            "meaningfully_positive": bool(positive),
            "paired_bootstrap_vs_A": paired_bootstrap_delta(base_scores, scores),
            "mcnemar_vs_A": mcnemar_exact(base_scores, scores),
        }

    if "B" in verdicts and "C" in verdicts:
        b_id = {r["example_id"]: r["correct"] for r in target_by_arm["B"]}
        c_id = {r["example_id"]: r["correct"] for r in target_by_arm["C"]}
        verdicts["C_vs_B"] = {
            **paired_bootstrap_delta([b_id[i] for i in base_ids], [c_id[i] for i in base_ids]),
            "mcnemar": mcnemar_exact(
                [b_id[i] for i in base_ids], [c_id[i] for i in base_ids]
            ),
        }

    positives = [a for a in ("B", "C") if verdicts.get(a, {}).get("meaningfully_positive")]
    if not positives:
        winner = "FROZEN"
        phase2 = False
    elif len(positives) == 1:
        winner = "LoRA" if positives[0] == "B" else "NoRA-INIT"
        phase2 = True
    else:
        best = max(positives, key=lambda a: verdicts[a]["target_gain_points"])
        winner = "LoRA" if best == "B" else "NoRA-INIT"
        phase2 = True

    return {
        "gate": {
            "gain": ">= 2 absolute points on the target composite OR >= 5% relative error reduction",
            "preservation": "average preservation regression <= 1 absolute point",
        },
        "baseline_target_accuracy_points": round(100 * base_acc, 2),
        "baseline_preservation_accuracy_points": round(100 * base_pres_acc, 2),
        "arms": verdicts,
        "winner": winner,
        "phase_2_licensed": phase2,
        "phase_2_note": (
            "Phase 2 runs only on a clearly positive adapter. A null or negative "
            "Phase 1 stops the programme and does not open a layer sweep."
        ),
    }


def _digest_rows(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def _unique_by_id(rows: list[dict], label: str) -> dict:
    by_id = {r["example_id"]: r for r in rows}
    if len(by_id) != len(rows):
        raise ValueError(f"duplicate example IDs in {label}")
    return by_id


def rescore_arm(out: Path, arm: str, targets: list[dict], preservation: list[dict]):
    """Recompute every decision score from raw predictions and current references."""
    rows = read_jsonl(out / "per_example.jsonl")
    if any(r.get("arm") != arm or r.get("evaluation") not in ("target", "preservation") for r in rows):
        raise ValueError(f"invalid typed evaluation records for arm {arm}")
    groups = {kind: [r for r in rows if r["evaluation"] == kind]
              for kind in ("target", "preservation")}
    rescored = {}
    for kind, expected in (("target", targets), ("preservation", preservation)):
        actual = _unique_by_id(groups[kind], f"{arm}/{kind}")
        expected_by_id = _unique_by_id(expected, f"expected {kind}")
        if set(actual) != set(expected_by_id):
            raise ValueError(f"row IDs/counts do not match evaluation set: {arm}/{kind}")
        result = []
        for item in expected:
            record = dict(actual[item["example_id"]])
            raw = record["raw_prediction"]
            if kind == "target":
                for field in ("split", "variant", "task_type", "answerable", "condition"):
                    if record.get(field) != item[field]:
                        raise ValueError(f"target metadata mismatch: {arm}/{item['example_id']}/{field}")
                record.update(correct=bool(score(raw, item["accepted_answers"])),
                              normalized_prediction=normalize(raw), malformed=is_malformed(raw))
            else:
                if record.get("reference") != item["reference"] or record.get("task_family") != item["task_family"]:
                    raise ValueError(f"preservation reference mismatch: {arm}/{item['example_id']}")
                record.update(score_preservation(raw, item))
            result.append(record)
        rescored[kind] = result
    return rescored["target"], rescored["preservation"]


def baseline_validity(records: list[dict], core: str, smoke: bool) -> dict:
    if smoke:
        return {"checked": False, "matches": None, "note": "smoke subset; no published baseline gate"}
    actual = {split: sum(r["correct"] for r in records if r["split"] == split)
              for split in ("showcase", "stability_test")}
    actual["all"] = sum(r["correct"] for r in records)
    expected = PUBLISHED_BASELINE[core]
    return {"checked": True, "matches": len(records) == 256 and actual == expected,
            "published": expected, "recomputed": actual, "n": len(records)}


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_arm(arm_out: Path, arm: str, config: dict, core: dict, *, resume: bool) -> dict:
    """Validate identity before constructing a model. Caller must hold arm lock."""
    occupied = arm_out.exists() and any(arm_out.iterdir())
    if occupied and not resume:
        raise FileExistsError(f"refusing nonempty arm directory: {arm_out}; use --resume for explicit recovery")
    if resume and not occupied:
        raise FileNotFoundError(f"--resume requires an existing arm identity: {arm_out}")
    expected = {"schema_version": 3, "config": {**config, "arm": arm},
                "source_sha256": source_hashes(), "canonical_core": str(core["path"]),
                "checkpoint_sha256": checkpoint_files(core["path"])}
    if not expected["checkpoint_sha256"]:
        raise ValueError("canonical core has no hashable checkpoint files")
    # Pin encoding, gate, precision and the effective fixed loop defaults as
    # scientific inputs, as well as every source/configuration file involved.
    expected["scientific_recipe"] = {
        **expected["config"], "source_sha256": expected["source_sha256"],
        "initialization": {"A": "none", "B": "lora", "C": "nora"}[arm],
        "tokenizer_policy": {"trust_remote_code": True, "local_files_only": True,
                             "encoding_implementation": "slcmi.training_data.build_dataset"},
        "optimizer": "AdamW", "weight_decay": 0.0, "max_grad_norm": 1.0, "warmup_ratio": 0.1,
        "scheduler": "linear warmup then cosine; applied optimizer updates only",
        "gradient_checkpointing": True, "adapter_dtype": "float32",
        "training_gate": "disabled only within training", "evaluation_gate": "canonical enabled",
        "eval_every": max(1, config["max_steps"] // 4),
        "resume_policy": "latest validated atomic transaction; original total attempt budget",
    }
    identity_path = arm_out / "identity.json"
    if resume:
        if not identity_path.is_file():
            raise ValueError("no durable arm identity; old non-checkpointed runs cannot be resumed")
        identity = _read_json(identity_path)
        for key, value in expected.items():
            if identity.get(key) != value:
                raise ValueError(f"resume identity mismatch: {key}")
        if arm != "A" and not (arm_out / "checkpoints/latest.pt").is_file():
            if any((arm_out / "segments").glob("*/training_entry.json")):
                raise ValueError("training was entered but latest.pt is missing; progress cannot be recovered safely")
            # All physical starts must prove the same source and immutable run.
            segments = sorted((arm_out / "segments").glob("*/segment_identity.json"))
            actual_dirs = list((arm_out / "segments").glob("*/"))
            if len(segments) != len(actual_dirs):
                raise ValueError("missing segment identity; zero progress is not proven")
            for path in segments:
                saved = _read_json(path)
                if (saved.get("run_id") != identity["run_id"]
                        or saved.get("source_sha256") != expected["source_sha256"]
                        or saved.get("config") != expected["config"]
                        or saved.get("checkpoint_sha256_before") != expected["checkpoint_sha256"]):
                    raise ValueError("segment identity does not prove a pre-training interruption")
        return identity
    identity = {**expected, "run_id": uuid.uuid4().hex,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    write_json(identity_path, identity)
    return identity


def seal_identity(arm_out: Path, identity: dict, field: str, value) -> None:
    """Write a previously unavailable initializer/data fact once, then compare."""
    if field in identity:
        if identity[field] != value:
            raise ValueError(f"resume identity mismatch: {field}")
        return
    if any((arm_out / "segments").glob("*/training_entry.json")):
        raise ValueError(f"cannot add missing identity field after training began: {field}")
    identity[field] = value
    write_json(arm_out / "identity.json", identity, overwrite=True)


def seal_initialization(arm_out: Path, identity: dict, receipt: dict) -> None:
    seal_identity(arm_out, identity, "adapter_initialization", receipt)
    data = identity["dataset_fingerprint"]
    fingerprints = {
        "recipe_sha256": fingerprint({"scientific_recipe": identity["scientific_recipe"],
                                      "dataset_fingerprint": data}),
        "train_data_sha256": data["train"], "eval_data_sha256": data["dev"],
        "base_checkpoint_sha256": fingerprint(identity["checkpoint_sha256"]),
        "adapter_initialization_sha256": receipt["sha256"],
    }
    seal_identity(arm_out, identity, "checkpoint_fingerprints", fingerprints)


def committed_checkpoint(arm_out: Path, identity: dict, train_set: list[dict] | None = None) -> dict | None:
    """Read only latest.pt, with safe CPU deserialization and accounting checks.

    Full optimizer/model/RNG validation and restoration belongs to train(). This
    read reconciles observability, and cannot authorize further model updates.
    """
    path = arm_out / "checkpoints/latest.pt"
    if not path.is_file():
        return None
    encoded = _read_json(arm_out / "encoded_data.json")
    if (fingerprint(encoded["train"]) != identity["dataset_fingerprint"]["train"]
            or fingerprint(encoded["dev"]) != identity["dataset_fingerprint"]["dev"]):
        raise ValueError("immutable encoded data fingerprint mismatch")
    if train_set is None:
        train_set = encoded["train"]
    elif train_set != encoded["train"]:
        raise ValueError("reconstructed encoded training data differs from saved input")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["identity"]["caller"] != identity.get("checkpoint_fingerprints"):
        raise ValueError("latest checkpoint caller fingerprint mismatch")
    config = identity["config"]
    interval = max(1, config["max_steps"] // 4)
    progress = payload["progress"]
    validate_progress(progress, max_steps=config["max_steps"], grad_accum=config["grad_accum"],
                      eval_due=lambda step: step % interval == 0 or step == config["max_steps"],
                      train_set=train_set)
    receipt = {"sha256": sha256(path), "completed_attempts": progress["completed_attempts"],
               "pending_validation": progress["pending_validation"],
               "trajectory_sha256": fingerprint(progress["trajectory"]),
               "evals_sha256": fingerprint(progress["evals"])}
    # Rebuild from the checkpoint, never append segment event files. A commit
    # may survive a hard kill that happened before its callback was delivered.
    validation = {record["step"]: record for record in progress["evals"]}
    events = []
    for record in progress["trajectory"]:
        events.append({"event": "optimizer_attempt", **record})
        if record["step"] in validation:
            events.append({"event": "validation", **validation[record["step"]]})
    write_jsonl(arm_out / "training_progress.jsonl", events, overwrite=True)
    write_json(arm_out / "committed_checkpoint.json", {"receipt": receipt, "progress": progress}, overwrite=True)
    return {"receipt": receipt, "progress": progress}


DOSE_FIELDS = {
    "optimizer_steps_attempted": "completed_attempts", "optimizer_steps_applied": "applied_steps",
    "steps_skipped_by_scaler": "skipped_steps", "steps_skipped_by_safety_check": "safety_skipped_steps",
    "examples_consumed": "consumed",
}


def segment_manifests(arm_out: Path) -> list[dict]:
    result = []
    for directory in sorted((arm_out / "segments").glob("*/")):
        path = directory / "run_manifest.json"
        terminal = path.is_file()
        if not terminal:
            path = directory / "started_manifest.json"
            if not path.is_file():
                path = directory / "segment_identity.json"
        manifest = _read_json(path)
        if not terminal:
            manifest = {**manifest, "status": "INTERRUPTED", "end_utc": None,
                        "resource_accounting_complete": False}
        ledger_path = directory / "model_load.json"
        ledger = _read_json(ledger_path) if ledger_path.is_file() else {
            "model_loads": 0, "model_loads_upper_bound": 0, "model_load_attempts": 0, "model_load_seconds": 0}
        resource = manifest.get("resource", {})
        if not terminal:
            log = directory / "training_progress.jsonl"
            events = read_jsonl_prefix(log) if log.is_file() else []
            attempts = [r for r in events if r.get("event") == "optimizer_attempt"]
            initial = manifest.get("resumed_from_committed_training", {})
            if attempts:
                last = attempts[-1]
                observed = {"optimizer_steps_attempted": last["attempt"],
                            "optimizer_steps_applied": last["optimizer_updates_applied"],
                            "steps_skipped_by_scaler": last["steps_skipped_by_scaler_total"],
                            "steps_skipped_by_safety_check": last["steps_skipped_by_safety_check_total"],
                            "examples_consumed": last["examples_consumed"]}
                resource.update({key: value - initial.get(key, 0) for key, value in observed.items()})
            else:
                resource.update({key: 0 for key in DOSE_FIELDS})
        complete = terminal and resource.get("resource_accounting_complete", True)
        resource = {**resource, "model_loads": ledger["model_loads"],
                    "model_loads_upper_bound": ledger["model_loads_upper_bound"],
                    "model_load_attempts": ledger["model_load_attempts"],
                    "model_load_seconds": ledger.get("model_load_seconds"),
                    "resource_accounting_complete": bool(complete and ledger.get("status") != "LOADING")}
        for key in ("wall_seconds", "peak_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
            resource.setdefault(key, None)
        result.append({**manifest, "segment_id": directory.name, "resource": resource})
    return result


def publish_arm(arm_out: Path, identity: dict, committed: dict | None, *, complete: bool,
                final_segment: Path | None = None) -> dict:
    """Publish a logical receipt while retaining every physical segment intact."""
    segments = segment_manifests(arm_out)
    progress = committed["progress"] if committed else {}
    resource = aggregate_resources(segments)
    physical_dose = {key: resource[key] for key in DOSE_FIELDS}
    resource.update({key: progress.get(value, 0) for key, value in DOSE_FIELDS.items()})
    resource.update(segment_observed_dose_lower_bound=physical_dose,
                    trainable_params=sum(math.prod(p["shape"])
                                         for p in identity.get("adapter_initialization", {}).get("parameters", [])),
                    resource_scope="physical segment wall/load costs; logical committed dose counted once",
                    unknown_segment_ids=[m["segment_id"] for m in segments
                                         if not m["resource"]["resource_accounting_complete"]],
                    uncommitted_work_may_be_missing=any(m["status"] != "COMPLETE" for m in segments))
    if not segments:
        raise ValueError("cannot publish an arm without a physical segment")
    if complete and identity["config"]["arm"] != "A":
        if (not committed or progress["completed_attempts"] != identity["config"]["max_steps"]
                or progress["pending_validation"] is not None):
            raise ValueError("arm cannot complete without its full committed attempt budget and validation")
    last = segments[-1]
    if complete:
        if final_segment is None or last["status"] != "COMPLETE":
            raise ValueError("logical completion requires a completed physical segment")
        arm = identity["config"]["arm"]
        for name in (f"arm_{arm}_info.json", f"arm_{arm}_target.jsonl",
                     f"arm_{arm}_preservation.jsonl", "per_example.jsonl", "dataset.json"):
            shutil.copyfile(final_segment / name, arm_out / name)
        if (final_segment / "adapter").is_dir():
            shutil.copytree(final_segment / "adapter", arm_out / "adapter", dirs_exist_ok=True)
    integrity = [m for m in segments if "checkpoint_sha256_after" in m]
    unchanged = bool(integrity) and all(
        m.get("canonical_checkpoint_unchanged") is True
        and m["checkpoint_sha256_before"] == identity["checkpoint_sha256"]
        and m["checkpoint_sha256_after"] == identity["checkpoint_sha256"] for m in integrity)
    manifest = {
        "schema_version": 3, "stage": f"arm_{identity['config']['arm']}",
        "status": "COMPLETE" if complete and unchanged else "FAILED",
        "config": identity["config"], "run_id": identity["run_id"],
        "start_utc": identity["created_utc"], "end_utc": last.get("end_utc"),
        "source_sha256": identity["source_sha256"], "canonical_core": identity["canonical_core"],
        "checkpoint_sha256_before": identity["checkpoint_sha256"],
        "checkpoint_sha256_after": last.get("checkpoint_sha256_after"),
        "canonical_checkpoint_unchanged": unchanged,
        "canonical_hash_coverage": "all observed boundaries; interrupted segment ends remain unobserved",
        "checkpoint_fingerprints": identity.get("checkpoint_fingerprints"),
        "checkpoint_receipt": committed["receipt"] if committed else None,
        "committed_training": {key: progress.get(value, 0) for key, value in DOSE_FIELDS.items()},
        "segment_ids": [m["segment_id"] for m in segments], "resume_count": len(segments) - 1,
        "resource": resource, "decision": {"winner": None, "phase_2_licensed": False},
    }
    write_json(arm_out / "run_manifest.json", manifest, overwrite=True)
    write_json(arm_out / "resource.json", resource, overwrite=True)
    return manifest


def aggregate_resources(manifests: list[dict]) -> dict:
    resources = [m["resource"] for m in manifests if "resource" in m]
    summed = ("wall_seconds", "model_loads", "model_loads_upper_bound", "model_load_attempts", "model_load_seconds", "optimizer_steps_attempted",
              "optimizer_steps_applied", "steps_skipped_by_scaler", "steps_skipped_by_safety_check",
              "examples_consumed")
    peaks = ("peak_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
    return {
        **{key: sum(r.get(key) or 0 for r in resources) for key in summed},
        **{key: max((r.get(key) or 0 for r in resources), default=0) for key in peaks},
        "resource_accounting_complete": all(r.get("resource_accounting_complete", True) for r in resources),
        "numeric_costs_are_lower_bounds": any(not r.get("resource_accounting_complete", True) for r in resources),
        "unknown_segment_ids": [segment for r in resources for segment in r.get("unknown_segment_ids", [])],
        "incomplete_metric_fields": sorted({key for r in resources for key in (*summed, *peaks)
                                             if key in r and r[key] is None}
                                            | {key for r in resources for key in r.get("incomplete_metric_fields", [])}),
        "trainable_params_by_arm": {m["config"]["arm"]: m.get("resource", {}).get("trainable_params", 0)
                                     for m in manifests},
        "resource_scope": "sum of arm process costs (including failed arms); maximum arm memory peaks",
    }


def validate_lifecycle_receipt(arm_dir: Path, arm: str, manifest: dict, info: dict) -> None:
    """A final receipt must represent one immutable, completed logical arm."""
    if manifest.get("schema_version") != 3:
        if manifest.get("config", {}).get("schema_version") == 3 or (arm_dir / "segments").exists():
            raise ValueError(f"missing durable lifecycle schema for arm {arm}")
        return
    identity = _read_json(arm_dir / "identity.json")
    for key in ("config", "source_sha256", "canonical_core", "checkpoint_fingerprints", "run_id"):
        if identity.get(key) != manifest.get(key):
            raise ValueError(f"logical arm identity mismatch: {arm}/{key}")
    if (identity.get("checkpoint_sha256") != manifest["checkpoint_sha256_before"]
            or identity.get("dataset_fingerprint") != info["dataset_fingerprint"]
            or identity.get("frozen_parameter_sha256") != info["frozen_base"]["before"]
            or info.get("run_id") != identity["run_id"]):
        raise ValueError(f"logical scientific identity mismatch for arm {arm}")
    expected = {
        "recipe_sha256": fingerprint({"scientific_recipe": identity["scientific_recipe"],
                                      "dataset_fingerprint": identity["dataset_fingerprint"]}),
        "train_data_sha256": identity["dataset_fingerprint"]["train"],
        "eval_data_sha256": identity["dataset_fingerprint"]["dev"],
        "base_checkpoint_sha256": fingerprint(identity["checkpoint_sha256"]),
        "adapter_initialization_sha256": identity["adapter_initialization"]["sha256"],
    }
    if identity["checkpoint_fingerprints"] != expected or info.get("checkpoint_fingerprints") != expected:
        raise ValueError(f"five checkpoint fingerprints mismatch for arm {arm}")
    segments = segment_manifests(arm_dir)
    ids = [m["segment_id"] for m in segments]
    if (not ids or ids != manifest.get("segment_ids") or manifest.get("resume_count") != len(ids) - 1
            or segments[-1]["status"] != "COMPLETE" or info.get("segment_id") != ids[-1]):
        raise ValueError(f"physical segment inventory/completion mismatch for arm {arm}")
    for segment in segments:
        for key in ("config", "source_sha256", "canonical_core", "run_id"):
            if segment.get(key) != identity[key]:
                raise ValueError(f"physical segment identity mismatch: {arm}/{key}")
        if (segment["checkpoint_sha256_before"] != identity["checkpoint_sha256"]
                or segment["resource"]["model_load_attempts"] > 1):
            raise ValueError(f"physical model/core invariant mismatch for arm {arm}")
        if "checkpoint_sha256_after" in segment and (
                segment["checkpoint_sha256_after"] != identity["checkpoint_sha256"]
                or segment.get("canonical_checkpoint_unchanged") is not True):
            raise ValueError(f"physical canonical checkpoint changed for arm {arm}")
    final_segment = arm_dir / "segments" / ids[-1]
    for name in (f"arm_{arm}_info.json", f"arm_{arm}_target.jsonl", f"arm_{arm}_preservation.jsonl", "per_example.jsonl", "dataset.json"):
        if sha256(arm_dir / name) != sha256(final_segment / name):
            raise ValueError(f"logical output differs from completed segment: {arm}/{name}")
    aggregated = aggregate_resources(segments)
    for key in ("wall_seconds", "model_loads", "model_loads_upper_bound", "peak_rss_bytes",
                "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes", "resource_accounting_complete"):
        if manifest["resource"].get(key) != aggregated[key]:
            raise ValueError(f"physical resource aggregation mismatch: {arm}/{key}")
    if arm == "A":
        if info["training"] is not None or any(manifest["committed_training"].values()):
            raise ValueError("frozen Arm A has an optimizer trajectory")
        return
    path = arm_dir / "checkpoints/latest.pt"
    receipt = manifest.get("checkpoint_receipt") or {}
    if receipt.get("sha256") != sha256(path):
        raise ValueError(f"final latest checkpoint hash mismatch for arm {arm}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["identity"]["caller"] != expected:
        raise ValueError(f"final latest checkpoint identity mismatch for arm {arm}")
    progress = state["progress"]
    saved = _read_json(arm_dir / "committed_checkpoint.json")
    training = info["training"]
    if (saved != {"receipt": receipt, "progress": progress}
            or progress["completed_attempts"] != manifest["config"]["max_steps"]
            or progress["pending_validation"] is not None
            or progress["trajectory"] != training["trajectory"] or progress["evals"] != training["validation"]
            or receipt["trajectory_sha256"] != fingerprint(progress["trajectory"])
            or receipt["evals_sha256"] != fingerprint(progress["evals"])):
        raise ValueError(f"full committed training trajectory mismatch for arm {arm}")
    if (len(progress["trajectory"]) != progress["completed_attempts"]
            or [r["attempt"] for r in progress["trajectory"]] != list(range(1, progress["completed_attempts"] + 1))
            or not training["gradient_checkpointing"] or not training["gate_restored_after_training"]):
        raise ValueError(f"fixed training budget or gate/checkpointing mismatch for arm {arm}")
    validation = {record["step"]: record for record in progress["evals"]}
    events = []
    for record in progress["trajectory"]:
        events.append({"event": "optimizer_attempt", **record})
        if record["step"] in validation:
            events.append({"event": "validation", **validation[record["step"]]})
    if read_jsonl(arm_dir / "training_progress.jsonl") != events:
        raise ValueError(f"logical training progress differs from committed history for arm {arm}")
    for key, value in DOSE_FIELDS.items():
        if manifest["resource"].get(key) != progress[value] or manifest["committed_training"].get(key) != progress[value]:
            raise ValueError(f"logical committed dose mismatch: {arm}/{key}")


def combine_stage(out: Path, config: dict, core: dict, targets: list[dict], preservation: list[dict]) -> dict:
    """Validate completed, compatible arm receipts, then rescore their raw records."""
    with Run(out.name, config, results_dir=out.parent, core_path=core["path"]) as run:
        write_json(out / "summary.json", {
            "config": config, "status": "PARTIAL", "completed_arms": [],
            "missing_arms": list(REQUIRED_ARMS), "arms": {},
            "decision": {"verdict": "INCOMPLETE", "winner": None, "phase_2_licensed": False,
                         "reason": "compatibility validation in progress"},
        }, overwrite=True)
        write_jsonl(out / "per_example.jsonl", [], overwrite=True)
        arms, target_by_arm, preservation_by_arm, manifests = {}, {}, {}, []
        expected_sources = source_hashes()
        expected_checkpoint = checkpoint_files(core["path"])
        dataset_fingerprint = None
        for arm in REQUIRED_ARMS:
            arm_dir = out / f"arm_{arm}"
            path = arm_dir / "run_manifest.json"
            identity_path = arm_dir / "identity.json"
            # A hard kill cannot publish a terminal arm receipt. Reconstruct
            # partial accounting from physical evidence before combining, even
            # when no explicit resume has yet been requested.
            if identity_path.is_file() and (arm_dir / "segments").exists():
                old = _read_json(path) if path.is_file() else {}
                if old.get("status") != "COMPLETE":
                    identity = _read_json(identity_path)
                    observed = committed_checkpoint(arm_dir, identity) if arm != "A" else None
                    publish_arm(arm_dir, identity, observed, complete=False)
            if not path.is_file():
                continue
            manifest = json.loads(path.read_text(encoding="utf-8"))
            manifests.append(manifest)
            if manifest["status"] != "COMPLETE":
                continue
            arm_config = dict(manifest["config"])
            if arm_config.pop("arm", None) != arm or arm_config != config:
                raise ValueError(f"configuration mismatch for arm {arm}")
            if manifest["source_sha256"] != expected_sources:
                raise ValueError(f"source hashes differ for arm {arm}")
            if (manifest.get("canonical_checkpoint_unchanged") is not True
                    or manifest["checkpoint_sha256_before"] != expected_checkpoint
                    or manifest["checkpoint_sha256_after"] != expected_checkpoint
                    or manifest["canonical_core"] != str(core["path"])):
                raise ValueError(f"canonical checkpoint mismatch for arm {arm}")
            info = json.loads((arm_dir / f"arm_{arm}_info.json").read_text(encoding="utf-8"))
            if (info["loaded_core_path"] != str(core["path"]) or info["loaded_dtype"] != core["dtype"]
                    or not info.get("evaluation_gate_on") or not info["frozen_base"]["unchanged"]
                    or info["frozen_base"]["before"] != info["frozen_base"]["after"]):
                raise ValueError(f"loaded model or frozen-base invariant mismatch for arm {arm}")
            fingerprint = info["dataset_fingerprint"]
            if (fingerprint.get("target") != _digest_rows(targets)
                    or fingerprint.get("preservation") != _digest_rows(preservation)
                    or fingerprint.get("target_count") != len(targets)
                    or fingerprint.get("preservation_count") != len(preservation)):
                raise ValueError(f"evaluation fingerprint differs for arm {arm}")
            if dataset_fingerprint is None:
                dataset_fingerprint = fingerprint
            elif fingerprint != dataset_fingerprint:
                raise ValueError(f"sampled curriculum/evaluation differs for arm {arm}")
            validate_lifecycle_receipt(arm_dir, arm, manifest, info)
            target, preserved = rescore_arm(arm_dir, arm, targets, preservation)
            info["target"] = summarise([r["correct"] for r in target])
            info["preservation"] = {
                "matched": summarise([r["matched"] for r in preserved]),
                "exact": summarise([r["exact"] for r in preserved]),
                "mean_response_words": round(statistics.mean(r["response_words"] for r in preserved), 2),
            }
            arms[arm], target_by_arm[arm], preservation_by_arm[arm] = info, target, preserved
        decision = decide(arms, target_by_arm, preservation_by_arm)
        validity = baseline_validity(target_by_arm["A"], config["core"], config["smoke"]) if "A" in arms else {"checked": False}
        complete = set(arms) == set(REQUIRED_ARMS)
        status = "COMPLETE" if complete else "PARTIAL"
        if validity.get("checked") and not validity["matches"]:
            status = "INVALID"
            decision = {"verdict": "INVALID", "winner": None, "phase_2_licensed": False,
                        "reason": "frozen Arm A fails the published baseline validity gate"}
        if config["smoke"]:
            decision["phase_2_licensed"] = False
            decision["phase_2_note"] = "smoke is a mechanics check and cannot license Phase 2"
        summary = {"config": config, "status": status, "completed_arms": sorted(arms),
                   "missing_arms": [a for a in REQUIRED_ARMS if a not in arms],
                   "arms": arms, "baseline_validity": validity, "decision": decision}
        write_json(out / "summary.json", summary, overwrite=True)
        write_jsonl(out / "per_example.jsonl", [r for arm in REQUIRED_ARMS if arm in arms
                    for r in [*target_by_arm[arm], *preservation_by_arm[arm]]], overwrite=True)
        run.finish(status, {"completed_arms": sorted(arms), "baseline_validity": validity,
                            **decision}, resource_overrides=aggregate_resources(manifests))
    return summary


def execute_arm(arm_out: Path, arm: str, config: dict, core: dict,
                targets: list[dict], preservation: list[dict], *, resume: bool) -> None:
    """One physical process segment. No restart ever constructs a second base."""
    with ExclusiveFileLock(arm_out.parent / f".arm_{arm}.lock"):
        identity = prepare_arm(arm_out, arm, config, core, resume=resume)
        existing = arm_out / "run_manifest.json"
        if resume and existing.is_file() and _read_json(existing).get("status") == "COMPLETE":
            return  # Completed recovery is idempotent; combine revalidates it.
        committed = committed_checkpoint(arm_out, identity) if arm != "A" else None
        initial_progress = dict(committed["progress"]) if committed else {}
        segment_id = f"{len(list((arm_out / 'segments').glob('*/'))) + 1:06d}-{uuid.uuid4().hex[:12]}"
        segment = arm_out / "segments" / segment_id
        header = {
            "schema_version": 3, "status": "PREPARING", "segment_id": segment_id,
            "run_id": identity["run_id"], "config": identity["config"],
            "source_sha256": identity["source_sha256"], "canonical_core": identity["canonical_core"],
            "checkpoint_sha256_before": identity["checkpoint_sha256"],
            "start_utc": datetime.now(timezone.utc).isoformat(), "resume_requested": resume,
            "resumed_from_attempt": initial_progress.get("completed_attempts", 0),
            "resumed_from_checkpoint_sha256": committed["receipt"]["sha256"] if committed else None,
        }
        write_json(segment / "segment_identity.json", header)
        complete = False
        try:
            with Run(segment_id, identity["config"], results_dir=arm_out / "segments",
                     core_path=core["path"]) as run:
                if (run.manifest["source_sha256"] != identity["source_sha256"]
                        or run.checkpoint_before != identity["checkpoint_sha256"]):
                    raise ValueError("source or core changed while preparing the physical segment")
                run.manifest.update({**header, "status": "STARTED"})
                write_json(segment / "started_manifest.json", run.manifest, overwrite=True)
                packs = load_packs(LCR_SFT)
                train_raw = sample_curriculum(packs, "train", config["train_packs"])
                dev_raw = sample_curriculum(packs, "dev", config["dev_packs"])
                tokenizer = AutoTokenizer.from_pretrained(core["path"], trust_remote_code=True,
                                                         local_files_only=True)
                train_set, train_stats = build_dataset(tokenizer, train_raw, MAX_LENGTH)
                dev_set, dev_stats = build_dataset(tokenizer, dev_raw, MAX_LENGTH)
                if train_stats["n_dropped_too_long"] or dev_stats["n_dropped_too_long"]:
                    raise RuntimeError("fixed curriculum would drop overlength examples")
                data_identity = {
                    "train": fingerprint(list(train_set)), "dev": fingerprint(list(dev_set)),
                    "target": _digest_rows(targets), "preservation": _digest_rows(preservation),
                    "train_count": len(train_set), "dev_count": len(dev_set),
                    "target_count": len(targets), "preservation_count": len(preservation),
                }
                seal_identity(arm_out, identity, "dataset_fingerprint", data_identity)
                encoded = {"train": train_set, "dev": dev_set}
                encoded_path = arm_out / "encoded_data.json"
                if encoded_path.is_file():
                    if _read_json(encoded_path) != encoded:
                        raise ValueError("reconstructed encoded data differs from immutable input artifact")
                else:
                    if any((arm_out / "segments").glob("*/training_entry.json")):
                        raise ValueError("encoded data artifact is missing after training entry")
                    write_json(encoded_path, encoded)
                write_json(segment / "dataset.json", {"train": train_stats, "dev": dev_stats,
                                                       "fingerprint": data_identity})
                del tokenizer, packs, train_raw, dev_raw
                gc.collect()
                committed = committed_checkpoint(arm_out, identity, train_set) if arm != "A" else None
                initial_progress = dict(committed["progress"]) if committed else {}
                run.manifest.update(
                    resumed_from_attempt=initial_progress.get("completed_attempts", 0),
                    resumed_from_checkpoint_sha256=committed["receipt"]["sha256"] if committed else None,
                    resumed_from_committed_training={key: initial_progress.get(value, 0)
                                                    for key, value in DOSE_FIELDS.items()},
                )
                write_json(segment / "started_manifest.json", run.manifest, overwrite=True)
                try:
                    info, _, _ = run_arm(
                        arm, core=core, device=config["device"], smoke=config["smoke"],
                        max_steps=config["max_steps"], grad_accum=config["grad_accum"], lr=config["lr"],
                        train_set=train_set, dev_set=dev_set, targets=targets, preservation=preservation,
                        out=segment, run=run, dataset_fingerprint=data_identity,
                        arm_out=arm_out, identity=identity, resume_from_checkpoint=bool(committed),
                    )
                finally:
                    # CPU-only reconciliation runs even after a sticky CUDA
                    # failure. Do not replace the original error if state read
                    # also fails; the failed receipt keeps both observations.
                    active_error = sys.exc_info()[0] is not None
                    try:
                        committed = committed_checkpoint(arm_out, identity, train_set) if arm != "A" else None
                        end_progress = committed["progress"] if committed else {}
                        run.manifest.update(
                            end_committed_attempt=end_progress.get("completed_attempts", 0),
                            checkpoint_receipt=committed["receipt"] if committed else None,
                        )
                        for key, value in DOSE_FIELDS.items():
                            run.resource_extra[key] = end_progress.get(value, 0) - initial_progress.get(value, 0)
                        run.resource_extra["physical_training_cost_is_lower_bound"] = active_error
                        if committed:
                            write_json(segment / "committed_checkpoint.json", committed)
                    except Exception as exc:
                        run.manifest["checkpoint_reconciliation_error"] = f"{type(exc).__name__}: {exc}"
                        if not active_error:
                            raise
                if arm != "A":
                    progress = committed["progress"] if committed else {}
                    if (progress.get("completed_attempts") != config["max_steps"]
                            or progress.get("pending_validation") is not None
                            or info["training"]["trajectory"] != progress["trajectory"]
                            or info["training"]["validation"] != progress["evals"]):
                        raise ValueError("training output does not match the full committed final checkpoint")
                run.finish("COMPLETE", {"arm": arm, "frozen_base_unchanged": info["frozen_base"]["unchanged"],
                                        "evaluation_gate_on": info["evaluation_gate_on"]})
                complete = True
        finally:
            publish_arm(arm_out, identity, committed, complete=complete,
                        final_segment=segment if complete else None)


GPU_EXECUTION_ENV = {
    "CUDA_VISIBLE_DEVICES": "0", "CUDA_LAUNCH_BLOCKING": "1",
    "PYTORCH_ALLOC_CONF": "backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8",
    "PYTORCH_CUDA_ALLOC_CONF": "backend:native,garbage_collection_threshold:0.7,per_process_memory_fraction:0.8",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--core", default="a11", choices=sorted(CORES))
    parser.add_argument("--arms", nargs="+", choices=REQUIRED_ARMS)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true",
                        help="explicitly recover the same immutable arm from latest.pt (total budget unchanged)")
    parser.add_argument("--combine", action="store_true",
                        help="validate and rescore all three isolated arm directories")
    args = parser.parse_args()
    if not args.combine and (args.arms is None or len(args.arms) != 1):
        parser.error("execute exactly one arm per process: --arms A, --arms B, or --arms C")
    if args.combine and args.arms is not None and set(args.arms) != set(REQUIRED_ARMS):
        parser.error("--combine always requires A B C; omit --arms")
    if args.max_steps < 1 or args.grad_accum < 1 or args.lr <= 0:
        parser.error("steps, accumulation and learning rate must be positive")
    if args.max_steps != 300 or args.grad_accum != 8 or args.lr != 1e-4:
        parser.error("the fixed recipe requires --max-steps 300 --grad-accum 8 --lr 0.0001; smoke uses six total attempts")
    if args.resume and args.combine:
        parser.error("--resume applies to one explicit arm, not --combine")
    if args.device == "cuda":
        mismatched = {key: os.environ.get(key) for key, value in GPU_EXECUTION_ENV.items()
                      if os.environ.get(key) != value}
        if mismatched:
            parser.error(f"fixed GPU execution environment differs: {mismatched}; set the frozen controller policy")

    core = CORES[args.core]
    max_steps = SMOKE_STEPS if args.smoke else args.max_steps
    stage = "phase1_smoke" if args.smoke else "phase1"
    if args.core != "a11":
        stage += f"_{args.core}"
    out = ROOT / "results" / stage
    if out.exists() and any(out.glob("arm_*_info.json")):
        raise FileExistsError(f"legacy shared-arm artifacts remain in {out}; archive/remove this stale stage first")
    config = {
        "schema_version": 3, "stage": stage, "smoke": args.smoke,
        "core": args.core, "core_label": core["label"], "core_path": str(core["path"]),
        "dtype": core["dtype"], "device": args.device, "required_arms": list(REQUIRED_ARMS),
        "execution_environment": {key: os.environ.get(key) for key in GPU_EXECUTION_ENV},
        "trainable_layers": list(DEFAULT_LAYERS), "protected_layers": list(PROTECTED_LAYERS),
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "rank": DEFAULT_RANK, "alpha": DEFAULT_ALPHA, "dropout": DEFAULT_DROPOUT,
        "max_steps": max_steps, "step_budget_unit": "optimizer attempts",
        "grad_accum": args.grad_accum, "lr": args.lr, "max_length": MAX_LENGTH,
        "seed": SEED, "train_packs": SMOKE_TRAIN_PACKS if args.smoke else TRAIN_PACKS,
        "dev_packs": SMOKE_DEV_PACKS if args.smoke else DEV_PACKS,
        "curriculum": str(LCR_SFT), "target_suite": str(DEMO_SUITE),
        "external_scorer_sha256": sha256(DEMO_SUITE.parents[1] / "scripts/evaluate_predictions.py"),
        "data_sha256": {
            "curriculum": sha256(LCR_SFT), "target_suite": sha256(DEMO_SUITE),
            **{f"preservation_{cls}": sha256(CORPUS_DIR / f"{cls}.jsonl")
               for cls in PRESERVATION_CLASSES},
        },
        "target_context": "FULL", "preservation_classes": list(PRESERVATION_CLASSES),
        "preservation_per_class": 8 if args.smoke else PRESERVATION_PER_CLASS,
        "eval_max_new_tokens": MAX_NEW_TOKENS,
    }
    targets = target_items(args.smoke)
    preservation = build_preservation_set(per_class=config["preservation_per_class"])
    # Held by every Phase 1 invocation, across stages and selected cores. OS
    # ownership releases on hard kill; no stale PID-file decision is involved.
    with ExclusiveFileLock(ROOT / "results/.phase1_model.lock"):
        if args.combine:
            summary = combine_stage(out, config, core, targets, preservation)
        else:
            arm = args.arms[0]
            arm_out = out / f"arm_{arm}"
            if not args.resume and arm_out.exists() and any(arm_out.iterdir()):
                raise FileExistsError(f"refusing nonempty arm directory: {arm_out}; use --resume")
            previous = combine_stage(out, config, core, targets, preservation)
            if previous["status"] == "INVALID":
                raise RuntimeError("existing baseline is INVALID; adapter runs are not interpretable")
            try:
                execute_arm(arm_out, arm, config, core, targets, preservation, resume=args.resume)
            finally:
                # Even failed/OOM recovery removes any stale COMPLETE decision.
                summary = combine_stage(out, config, core, targets, preservation)
    print(f"{stage}: {summary['status']}; completed={summary['completed_arms']}; "
          f"missing={summary['missing_arms']}; winner={summary['decision']['winner']}; "
          f"phase_2_licensed={summary['decision']['phase_2_licensed']}")
    if summary["status"] == "INVALID":
        raise SystemExit("baseline validity gate failed")


if __name__ == "__main__":
    main()
