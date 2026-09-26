#!/usr/bin/env python3
"""Phase 0, step 2: score every cached context condition under one frozen model.

One model load serves all conditions. Model, checkpoint, decoding, output
length, scorer, prompt and seed policy are identical across A/B/C/D -- the only
thing that varies is the context string.

    python scripts/phase0_run_eval.py [--smoke] [--device cuda]

Validity gate: on the full run, condition A must reproduce the published
Supa-A1.1 result (45/64 Showcase, 125/192 Stability, 170/256 combined) or the
stage is reported as INVALID.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from slcmi.evaluation import (  # noqa: E402
    MAX_NEW_TOKENS,
    SEED,
    evaluate_rows,
    load_model,
)
from slcmi.provenance import (  # noqa: E402
    CORES,
    Run,
    read_jsonl,
    state_hash,
    write_json,
    write_jsonl,
)
from slcmi.stats import mcnemar_exact, paired_bootstrap_delta, summarise  # noqa: E402
from slcmi.suites import SUITES  # noqa: E402

CONDITION_ORDER = ["A_full", "B_query", "C_random", "D_oracle"]

# Published expected values for every (core, gate) cell, from each core's own
# CORE.json / README. A1.1 all256 ungated is 41 + 121 = 162, stated in the
# README's results table.
PUBLISHED_BASELINE = {
    ("a11", True): {"showcase": 45, "stability_test": 125, "all": 170},
    ("a11", False): {"showcase": 41, "stability_test": 121, "all": 162},
    ("a1", True): {"showcase": 43, "stability_test": 125, "all": 168},
    ("a1", False): None,  # no published ungated figure for the bfloat16 store
}


def analyse(records: list[dict], full_run: bool, gated: bool = True,
            expected: dict | None = None) -> dict:
    by_condition: dict[str, dict[str, dict]] = defaultdict(dict)
    for r in records:
        by_condition[r["condition"]][r["example_id"]] = r

    conditions = [c for c in CONDITION_ORDER if c in by_condition]

    # Two evaluation frames. The "all" frame uses every row; the
    # "oracle_eligible" frame is restricted to rows where D exists, so D can be
    # compared against A/B/C on identical examples.
    all_ids = sorted(by_condition["A_full"])
    oracle_ids = sorted(by_condition.get("D_oracle", {}))

    frames: dict[str, list[str]] = {"all": all_ids}
    if oracle_ids:
        frames["oracle_eligible"] = oracle_ids

    out: dict = {"conditions_present": conditions, "frames": {}}

    for frame_name, ids in frames.items():
        present = [c for c in conditions if all(i in by_condition[c] for i in ids)]
        scores = {c: [by_condition[c][i]["correct"] for i in ids] for c in present}

        headline = {c: summarise(scores[c]) for c in present}
        contrasts = {}
        for label, (base, arm) in {
            "B_minus_A": ("A_full", "B_query"),
            "B_minus_C": ("C_random", "B_query"),
            "C_minus_A": ("A_full", "C_random"),
            "D_minus_A": ("A_full", "D_oracle"),
            "D_minus_B": ("B_query", "D_oracle"),
        }.items():
            if base in present and arm in present:
                contrasts[label] = {
                    **paired_bootstrap_delta(scores[base], scores[arm]),
                    "mcnemar": mcnemar_exact(scores[base], scores[arm]),
                }

        by_family = {}
        for c in present:
            fam: dict[str, list[bool]] = defaultdict(list)
            for i in ids:
                fam[by_condition[c][i]["task_type"]].append(by_condition[c][i]["correct"])
            by_family[c] = {k: summarise(v) for k, v in sorted(fam.items())}

        by_variant = {}
        for c in present:
            var: dict[str, list[bool]] = defaultdict(list)
            for i in ids:
                var[by_condition[c][i]["variant"]].append(by_condition[c][i]["correct"])
            by_variant[c] = {k: summarise(v) for k, v in sorted(var.items())}

        out["frames"][frame_name] = {
            "n_examples": len(ids),
            "headline": headline,
            "contrasts": contrasts,
            "by_task_type": by_family,
            "by_variant": by_variant,
            "malformed_rate": {
                c: round(
                    sum(by_condition[c][i]["malformed"] for i in ids) / len(ids), 4
                )
                for c in present
            },
            "mean_prompt_tokens": {
                c: round(statistics.mean(by_condition[c][i]["prompt_tokens"] for i in ids), 1)
                for c in present
            },
            "mean_inference_seconds": {
                c: round(
                    statistics.mean(by_condition[c][i]["inference_seconds"] for i in ids), 4
                )
                for c in present
            },
        }

    if not gated:
        out["baseline_validity"] = {
            "checked": False,
            "note": (
                "This suite has no published Supa-A1.1 baseline, so no validity "
                "gate applies. It is confirmatory; the gated demo suite is primary."
            ),
        }
        return out

    # Validity gate against the published core.
    a = by_condition["A_full"]
    splits: dict[str, int] = defaultdict(int)
    for i in all_ids:
        if a[i]["correct"]:
            splits[a[i]["split"]] += 1
    reproduced = {
        "showcase": splits.get("showcase", 0),
        "stability_test": splits.get("stability_test", 0),
        "all": sum(splits.values()),
    }
    out["baseline_validity"] = {
        "published": expected,
        "reproduced": reproduced,
        "matches": (reproduced == expected) if (full_run and expected) else None,
        "checked": bool(full_run and expected),
        "note": (
            "Condition A is the canonical published protocol for this (core, gate) "
            "cell. On the full 256-row run it must reproduce the published counts "
            "exactly, or every downstream contrast is suspect."
        ),
    }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--suite", default="demo", choices=sorted(SUITES))
    parser.add_argument("--core", default="a11", choices=sorted(CORES))
    parser.add_argument("--gate", default="on", choices=("on", "off"))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    suite = SUITES[args.suite]
    core = CORES[args.core]
    gate_on = args.gate == "on"
    expected = PUBLISHED_BASELINE.get((args.core, gate_on)) if suite.key == "demo" else None
    gated = suite.key == "demo"
    base = "phase0_conditions" if gated else f"phase0_conditions_{suite.key}"
    suffix = "_smoke" if args.smoke else ""
    conditions_path = ROOT / f"results/{base}{suffix}/conditions.jsonl"
    if not conditions_path.is_file():
        raise SystemExit(
            f"missing {conditions_path}; run phase0_build_conditions.py "
            f"--suite {suite.key}{' --smoke' if args.smoke else ''} first"
        )

    items = read_jsonl(conditions_path)
    stem = "phase0_eval" if gated else f"phase0_eval_{suite.key}"
    if not (args.core == "a11" and gate_on):
        stem += f"_{args.core}_{'gated' if gate_on else 'ungated'}"
    stage = stem + suffix
    config = {
        "stage": stage,
        "suite": suite.key,
        "suite_role": suite.role,
        "suite_note": suite.note,
        "oracle_localisation_mode": suite.oracle_mode,
        "arm": f"frozen_{args.core}_{'gated' if gate_on else 'ungated'}",
        "core": args.core,
        "core_label": core["label"],
        "core_path": str(core["path"]),
        "abstention_gate": "on" if gate_on else "off (abstain_logit_bias=0.0, in memory only)",
        "conditions": CONDITION_ORDER,
        "device": args.device,
        "dtype": core["dtype"],
        "max_new_tokens": MAX_NEW_TOKENS,
        "do_sample": False,
        "seed": SEED,
        "enable_thinking": False,
        "scorer": "Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py::normalize + strict exact",
        "n_condition_rows": len(items),
    }

    with Run(stage, config, core_path=core["path"]) as run:
        tokenizer, model = load_model(core["path"], args.device, core["dtype"], gate_on)
        assert bool(model.config.abstain_logit_bias) == gate_on, "gate state not as requested"
        run.model_loads = 1
        before = state_hash(model)

        records: list[dict] = []
        for condition in CONDITION_ORDER:
            subset = [i for i in items if i["condition"] == condition]
            if not subset:
                print(f"  {condition}: no rows (unavailable by construction)")
                continue
            print(f"  {condition}: {len(subset)} rows")
            records.extend(evaluate_rows(tokenizer, model, subset, device=args.device))

        after = state_hash(model)
        if before != after:
            raise RuntimeError("model weights changed during inference-only evaluation")

        summary = analyse(records, full_run=not args.smoke, gated=gated, expected=expected)
        write_jsonl(run.out / "per_example.jsonl", records, overwrite=True)
        write_json(run.out / "summary.json", summary, overwrite=True)
        resource = run.finish(
            "COMPLETE",
            {
                "model_state_hash_unchanged": True,
                "model_state_hash": before,
                "summary": summary,
            },
        )

    frame = summary["frames"]["all"]
    print("\n--- headline (all rows) ---")
    for c, s in frame["headline"].items():
        print(f"  {c:9s} {s['correct']:4d}/{s['n']:<4d} = {s['accuracy_points']:5.1f}%")
    if "oracle_eligible" in summary["frames"]:
        print("\n--- headline (oracle-eligible frame) ---")
        for c, s in summary["frames"]["oracle_eligible"]["headline"].items():
            print(f"  {c:9s} {s['correct']:4d}/{s['n']:<4d} = {s['accuracy_points']:5.1f}%")
    print("\nbaseline validity:", summary["baseline_validity"]["reproduced"],
          "matches:", summary["baseline_validity"]["matches"])
    print(f"wall {resource['wall_seconds']}s  peak GPU {resource['peak_gpu_allocated_bytes']/1e6:.0f} MB"
          f"  peak RSS {resource['peak_rss_bytes']/1e9:.2f} GB")
    print(f"wrote {run.out}")

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
