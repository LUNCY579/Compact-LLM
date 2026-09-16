"""RRP-009: transitions plus rehearsal, to test whether the forgetting can be bought back.

RRP-007 established the asymmetry this lane exists for. The new skill replicates almost
exactly across three seeds -- 289/384 every time, every family clearing its floor -- while the
retention gate passed on seeds 17 and 29 and FAILED on seed 23 (panel 58, references retained
35 of 47). Retained was 37 / 35 / 37: never once above the 37 minimum.

Rehearsal is the standard intervention for exactly that failure: keep showing the model the
old distribution while it learns the new one. This module builds the mixture.

**The exclusion is the part that makes the test honest.** The rehearsal source is the v3
preservation slice, whose ids (`rineya_grnc_v2_*`) have ZERO overlap with the 150-row panel --
but 750 of its rows carry one of the panel's own three `reply_class` values
(`percentage_reasoning`, `ratio_proportion`, `time_duration`). Those 750 are dropped. Training
on the panel's own task types and then reporting a panel improvement is tuning against the
retention instrument, which QUARANTINE.json bars. With them gone, any panel movement is
transfer from the other 59 classes rather than practice on the tested three.

Restricted further to single-turn (system, user, assistant) rows so every row compiles through
the already-qualified `compile_row` path. The mixture introduces no second serialization mode.
The multi-turn rows are a declared scope limit, not an oversight.

The transition half is built by `process_arm.build` at the RRP-008 D2 draw, so it is the SAME
corpus as the D2 rung and the comparison isolates rehearsal rather than confounding it with a
different transition sample.

Pre-registration, both comparisons, four predictions and the analysis rules are in
`forge-v2-prereg-001/RRP009_MIXTURE_PREREG.md`, written before anything here was built.

Torch-free at module level. `compile_row` needs a tokenizer, which the caller supplies.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import digest, write_new

from . import budget, process_arm
from .serialization import compile_row
from .sources import stable_hash

EXPERIMENT = "RRP-009-PROCESS-MIX"
SALT = "RRP-009-process-mix"

# The three classes the 150-row arithmetic panel scores. Rehearsal must not contain them.
PANEL_REPLY_CLASSES = frozenset({"percentage_reasoning", "ratio_proportion", "time_duration"})

LIMITS = dict(process_arm.LIMITS)
EXCLUDES = process_arm.EXCLUDES
budget.register(EXPERIMENT, ceiling=LIMITS, phases=("train", "resume"), excludes=EXCLUDES)


def rehearsal_pool(path: Path) -> tuple[list[dict], dict]:
    """Single-turn preservation rows, with every panel reply_class removed."""
    kept, dropped_panel, dropped_multiturn = [], 0, 0
    classes = collections.Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        reply_class = (row.get("metadata") or {}).get("reply_class")
        if reply_class in PANEL_REPLY_CLASSES:
            dropped_panel += 1
            continue
        roles = [m["role"] for m in row["messages"]]
        if roles != ["system", "user", "assistant"]:
            dropped_multiturn += 1
            continue
        kept.append(row)
        classes[reply_class] += 1
    if dropped_panel == 0:
        raise ValueError("expected panel reply_classes in the preservation slice; found none, "
                         "so the exclusion is not doing anything and the source is wrong")
    audit = {"kept": len(kept), "dropped_panel_reply_class": dropped_panel,
             "dropped_multi_turn": dropped_multiturn,
             "panel_reply_classes_excluded": sorted(PANEL_REPLY_CLASSES),
             "distinct_classes_kept": len(classes)}
    return kept, audit


def compile_rehearsal(pool: list[dict], tokenizer, *, token_budget: int,
                      max_sequence: int = process_arm.TRAIN_WINDOW) -> tuple[list[dict], dict]:
    """Take deterministically-ordered rows until the processed-token budget is met."""
    ordered = sorted(pool, key=lambda r: (digest([SALT, "rehearsal", r["id"]]), r["id"]))
    compiled, used, rejected = [], 0, 0
    for row in ordered:
        if used >= token_budget:
            break
        prompt = [m for m in row["messages"] if m["role"] in ("system", "user")]
        final = next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        try:
            unit = compile_row(prompt, final, tokenizer, max_sequence=max_sequence)
        except ValueError:
            rejected += 1
            continue
        reply_class = (row.get("metadata") or {}).get("reply_class")
        compiled.append({**unit, "id": f"rehearsal:{row['id']}", "parent_id": row["id"],
                         "family": f"rehearsal_{reply_class}", "source_split": "train",
                         "source_release": "v3_preservation_slice",
                         "prompt": prompt, "reference": final, "reference_text": final})
        used += unit["processed_tokens"]
    return compiled, {"rehearsal_rows": len(compiled), "rehearsal_processed_tokens": used,
                      "rehearsal_token_budget": token_budget, "rejected_overlength": rejected}


def interleave(transitions: list[dict], rehearsal: list[dict]) -> list[dict]:
    """One deterministic order over the union. No block structure the schedule could exploit."""
    return sorted(transitions + rehearsal,
                  key=lambda r: (digest([SALT, "order", r["id"]]), r["id"]))


def build(binding: dict, views: Path, rehearsal_path: Path, tokenizer, *,
          per_family: int, exposures: int = 1,
          selection_seed: int = process_arm.SEED) -> tuple[list[dict], dict, dict]:
    transitions, _, transition_report = process_arm.build(
        binding, views, tokenizer, per_family=per_family, exposures=exposures,
        selection_seed=selection_seed, experiment=EXPERIMENT)
    transition_tokens = sum(len(r["ids"]) for r in transitions)

    pool, pool_audit = rehearsal_pool(rehearsal_path)
    rehearsal, rehearsal_audit = compile_rehearsal(pool, tokenizer,
                                                   token_budget=transition_tokens)
    rows = interleave(transitions, rehearsal)

    # The probe measures the NEW skill, so it is drawn from the transition half only.
    plan = process_arm.build_plan(rows, exposures=exposures, experiment=EXPERIMENT)
    plan["probe_row_ids"] = process_arm.probe_subset(transitions, per_family=8)

    report = {
        "record": "RRP009_PROCESS_MIX_BUILD",
        "schema": 1,
        "experiment": EXPERIMENT,
        "what_this_is": ("One-step transitions interleaved with v3 preservation rehearsal, to "
                         "test whether rehearsal buys back the retention RRP-007 spent."),
        "salt": SALT,
        "train_window": transition_report["train_window"],
        "held_out_parents": transition_report["held_out_parents"],
        "transition_rows": len(transitions),
        "transition_processed_tokens": transition_tokens,
        "transition_draw": {"per_family": per_family, "selection_seed": selection_seed,
                            "identical_to": "RRP-008 D2 rung"},
        "rehearsal_source": str(rehearsal_path),
        "rehearsal_source_sha256": stable_hash(rehearsal_path),
        "rehearsal_pool_audit": pool_audit,
        "rehearsal_audit": rehearsal_audit,
        "rehearsal_never_scored": True,
        "total_rows": len(rows),
        "processed_tokens": plan["processed_tokens"],
        "supervised_tokens": plan["supervised_tokens"],
        "probe_drawn_from": "transition half only",
        "quality_claim": False,
        "confirmation": "NOT_ESTABLISHED",
    }
    return rows, plan, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-prepared", type=Path, required=True)
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--per-family", type=int, required=True)
    parser.add_argument("--exposures", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=process_arm.SEED)
    args = parser.parse_args(argv)

    from experiments.model_quality_gpu_v1.native import load_tokenizer

    binding = json.loads(Path(args.binding).read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(args.source)
    rows, plan, report = build(binding, args.views, args.rehearsal, tokenizer,
                               per_family=args.per_family, exposures=args.exposures,
                               selection_seed=args.selection_seed)
    process_arm.write_prepared(args.out_prepared, rows, report)
    write_new(args.out_plan, plan)

    pa = report["rehearsal_pool_audit"]
    print(f"rehearsal pool      : {pa['kept']} kept "
          f"(dropped {pa['dropped_panel_reply_class']} panel-class, "
          f"{pa['dropped_multi_turn']} multi-turn) across {pa['distinct_classes_kept']} classes")
    print(f"transitions         : {report['transition_rows']} rows / "
          f"{report['transition_processed_tokens']} tokens")
    print(f"rehearsal           : {report['rehearsal_audit']['rehearsal_rows']} rows / "
          f"{report['rehearsal_audit']['rehearsal_processed_tokens']} tokens")
    print(f"total               : {report['total_rows']} rows / {plan['processed_tokens']} "
          f"processed, {plan['supervised_tokens']} supervised, "
          f"{len(plan['update_groups'])} update groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
