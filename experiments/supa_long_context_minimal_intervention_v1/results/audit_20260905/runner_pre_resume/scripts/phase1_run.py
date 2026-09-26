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
import statistics
import sys
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
    Run,
    checkpoint_files,
    frozen_parameter_hash,
    read_jsonl,
    sha256,
    source_hashes,
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
) -> dict:
    print(f"\n=== ARM {arm} ===")
    run.model_loads += 1
    tokenizer, model = load_model(core["path"], device, dtype=core["dtype"], gate=True)
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
    else:
        kind = "lora" if arm == "B" else "nora"
        probe = tokenizer("CONTEXT\nprobe.\n\nQUESTION\nwhat?\n\nANSWER",
                          return_tensors="pt", add_special_tokens=False).to(device)
        with torch.no_grad():
            reference_logits = model(**probe).logits.float().clone()

        model, adapter_info = build_adapter(
            model, kind, DEFAULT_LAYERS, DEFAULT_RANK, DEFAULT_ALPHA, DEFAULT_DROPOUT, SEED
        )
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
        def record_progress(record: dict) -> None:
            # Flush each attempted update, so an OOM/interruption leaves evidence.
            with (out / "training_progress.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
            if record.get("event") == "optimizer_attempt":
                run.resource_extra.update(
                    trainable_params=adapter_info["trainable_params"],
                    optimizer_steps_attempted=record["attempt"],
                    optimizer_steps_applied=record["optimizer_updates_applied"],
                    steps_skipped_by_scaler=record.get("steps_skipped_by_scaler_total", 0),
                    steps_skipped_by_safety_check=record.get("steps_skipped_by_safety_check_total", 0),
                    examples_consumed=record["examples_consumed"],
                )

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
        )
        model.eval()

    assert model.config.abstain_logit_bias, "abstention gate must be on for evaluation"

    print("  target eval (full context)...")
    target_records = evaluate_rows(tokenizer, model, targets, device=device)
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
    training = info["training"] or {}
    run.resource_extra.update(
        trainable_params=info["adapter"]["trainable_params"],
        optimizer_steps_attempted=training.get("max_steps", 0),
        optimizer_steps_applied=training.get("steps_applied", 0),
        steps_skipped_by_scaler=training.get("steps_skipped_by_scaler", 0),
        steps_skipped_by_safety_check=training.get("steps_skipped_by_safety_check", 0),
        examples_consumed=training.get("examples_consumed", 0),
    )

    # Strip the adapter modules out of the base before dropping references,
    # otherwise the LoRA layers keep the base graph alive.
    if arm != "A":
        try:
            model = model.unload()
        except Exception:
            model = model.get_base_model()
    del model, tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

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


def aggregate_resources(manifests: list[dict]) -> dict:
    resources = [m["resource"] for m in manifests if "resource" in m]
    summed = ("wall_seconds", "model_loads", "optimizer_steps_attempted",
              "optimizer_steps_applied", "steps_skipped_by_scaler", "steps_skipped_by_safety_check",
              "examples_consumed")
    peaks = ("peak_rss_bytes", "peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes")
    return {
        **{key: sum(r.get(key, 0) for r in resources) for key in summed},
        **{key: max((r.get(key, 0) for r in resources), default=0) for key in peaks},
        "trainable_params_by_arm": {m["config"]["arm"]: m.get("resource", {}).get("trainable_params", 0)
                                     for m in manifests},
        "resource_scope": "sum of arm process costs (including failed arms); maximum arm memory peaks",
    }


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--core", default="a11", choices=sorted(CORES))
    parser.add_argument("--arms", nargs="+", choices=REQUIRED_ARMS)
    parser.add_argument("--device", default="cuda", choices=("cuda", "cpu"))
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--combine", action="store_true",
                        help="validate and rescore all three isolated arm directories")
    args = parser.parse_args()
    if not args.combine and (args.arms is None or len(args.arms) != 1):
        parser.error("execute exactly one arm per process: --arms A, --arms B, or --arms C")
    if args.combine and args.arms is not None and set(args.arms) != set(REQUIRED_ARMS):
        parser.error("--combine always requires A B C; omit --arms")
    if args.max_steps < 1 or args.grad_accum < 1 or args.lr <= 0:
        parser.error("steps, accumulation and learning rate must be positive")

    core = CORES[args.core]
    max_steps = SMOKE_STEPS if args.smoke else args.max_steps
    stage = "phase1_smoke" if args.smoke else "phase1"
    if args.core != "a11":
        stage += f"_{args.core}"
    out = ROOT / "results" / stage
    if out.exists() and any(out.glob("arm_*_info.json")):
        raise FileExistsError(f"legacy shared-arm artifacts remain in {out}; archive/remove this stale stage first")
    config = {
        "schema_version": 2, "stage": stage, "smoke": args.smoke,
        "core": args.core, "core_label": core["label"], "core_path": str(core["path"]),
        "dtype": core["dtype"], "device": args.device, "required_arms": list(REQUIRED_ARMS),
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
    if args.combine:
        summary = combine_stage(out, config, core, targets, preservation)
    else:
        arm = args.arms[0]
        arm_out = out / f"arm_{arm}"
        if arm_out.exists() and any(arm_out.iterdir()):
            raise FileExistsError(f"refusing nonempty arm directory: {arm_out}")
        previous = combine_stage(out, config, core, targets, preservation)
        if previous["status"] == "INVALID":
            raise RuntimeError("existing baseline is INVALID; adapter runs are not interpretable")
        try:
            with Run(f"arm_{arm}", {**config, "arm": arm}, results_dir=out,
                     core_path=core["path"], refuse_nonempty=True) as run:
                packs = load_packs(LCR_SFT)
                train_raw = sample_curriculum(packs, "train", config["train_packs"])
                dev_raw = sample_curriculum(packs, "dev", config["dev_packs"])
                tokenizer = AutoTokenizer.from_pretrained(core["path"], trust_remote_code=True,
                                                         local_files_only=True)
                train_set, train_stats = build_dataset(tokenizer, train_raw, MAX_LENGTH)
                dev_set, dev_stats = build_dataset(tokenizer, dev_raw, MAX_LENGTH)
                if train_stats["n_dropped_too_long"] or dev_stats["n_dropped_too_long"]:
                    raise RuntimeError("fixed curriculum would drop overlength examples")
                fingerprint = {
                    "train": _digest_rows(train_set), "dev": _digest_rows(dev_set),
                    "target": _digest_rows(targets), "preservation": _digest_rows(preservation),
                    "train_count": len(train_set), "dev_count": len(dev_set),
                    "target_count": len(targets), "preservation_count": len(preservation),
                }
                del tokenizer, packs, train_raw, dev_raw
                gc.collect()
                info, _, _ = run_arm(
                    arm, core=core, device=args.device, smoke=args.smoke,
                    max_steps=max_steps, grad_accum=args.grad_accum, lr=args.lr,
                    train_set=train_set, dev_set=dev_set, targets=targets,
                    preservation=preservation, out=run.out, run=run,
                    dataset_fingerprint=fingerprint,
                )
                write_json(run.out / "dataset.json", {"train": train_stats, "dev": dev_stats,
                                                      "fingerprint": fingerprint})
                run.finish("COMPLETE", {"arm": arm, "frozen_base_unchanged": info["frozen_base"]["unchanged"],
                                        "evaluation_gate_on": info["evaluation_gate_on"]})
        finally:
            # Even a failed/OOM arm removes any stale root COMPLETE decision.
            summary = combine_stage(out, config, core, targets, preservation)
    print(f"{stage}: {summary['status']}; completed={summary['completed_arms']}; "
          f"missing={summary['missing_arms']}; winner={summary['decision']['winner']}; "
          f"phase_2_licensed={summary['decision']['phase_2_licensed']}")
    if summary["status"] == "INVALID":
        raise SystemExit("baseline validity gate failed")


if __name__ == "__main__":
    main()
