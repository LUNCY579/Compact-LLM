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
PANEL_PATH = Path(__file__).resolve().parents[2] / "experiments/architecture_benchmark_001/data/monitor_probe.jsonl"

# The three classes the 150-row arithmetic panel scores. Rehearsal must not contain them.
PANEL_REPLY_CLASSES = frozenset({"percentage_reasoning", "ratio_proportion", "time_duration"})

# RRP-011 arms. A filter changes WHICH rehearsal rows are eligible and nothing else: same
# transition draw, same token budget, same recipe. Registered in
# forge-v2-prereg-001/RRP011_COMPOSITION_PREREG.md before any arm was built.
ABSTENTION_CLASSES = frozenset({
    "insufficient_information", "clarification", "clarification_request",
    "safety_boundary", "boundary_setting", "uncertainty", "knowledge_boundary"})
REHEARSAL_FILTERS = {
    "all": lambda m: True,
    "communication": lambda m: m.get("capability_group") == "communication",
    "reasoning": lambda m: m.get("capability_group") == "reasoning",
    "integrated": lambda m: m.get("capability_group") == "integrated",
    "abstention": lambda m: m.get("reply_class") in ABSTENTION_CLASSES,
    # RRP-012 amendment 1: insufficient_information (206 rows) is BOTH an abstention class
    # and capability_group == reasoning, so a plain abstention+reasoning blend double-counts
    # exactly the class the experiment cares most about. This component makes the two
    # disjoint by construction.
    "reasoning_minus_abstention": lambda m: (m.get("capability_group") == "reasoning"
                                             and m.get("reply_class") not in ABSTENTION_CLASSES),
}

LIMITS = dict(process_arm.LIMITS)
EXCLUDES = process_arm.EXCLUDES
budget.register(EXPERIMENT, ceiling=LIMITS, phases=("train", "resume"), excludes=EXCLUDES)


def rehearsal_pool(path: Path, *, select: str = "all",
                   historical_reproduction: bool = False) -> tuple[list[dict], dict]:
    """Single-turn preservation rows, panel classes removed, then the arm's filter.

    The panel exclusion is unconditional and runs FIRST, so no filter can reintroduce a
    class the panel scores. The filter then narrows whatever survives.
    """
    if select not in REHEARSAL_FILTERS:
        raise ValueError(f"unregistered rehearsal filter {select!r}; "
                         f"expected one of {sorted(REHEARSAL_FILTERS)}")
    eligible = REHEARSAL_FILTERS[select]
    kept, dropped_panel, dropped_multiturn, dropped_filter = [], 0, 0, 0
    classes = collections.Counter()
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        meta = row.get("metadata") or {}
        reply_class = meta.get("reply_class")
        if reply_class in PANEL_REPLY_CLASSES:
            dropped_panel += 1
            continue
        roles = [m["role"] for m in row["messages"]]
        if roles != ["system", "user", "assistant"]:
            dropped_multiturn += 1
            continue
        if not eligible(meta):
            dropped_filter += 1
            continue
        kept.append(row)
        classes[reply_class] += 1
    if dropped_panel == 0:
        raise ValueError("expected panel reply_classes in the preservation slice; found none, "
                         "so the exclusion is not doing anything and the source is wrong")
    if historical_reproduction:
        text_contact = {"status": "SKIPPED_FOR_EXPLICIT_HISTORICAL_REPRODUCTION",
                        "current_gate_eligible": False}
    else:
        from .text_contact import audit_and_filter
        panel_sha = stable_hash(PANEL_PATH)
        panel = [json.loads(line) for line in PANEL_PATH.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
        kept, text_contact = audit_and_filter(kept, panel)
        if stable_hash(PANEL_PATH) != panel_sha:
            raise ValueError("retention panel changed during text-contact exclusion")
        text_contact = {**text_contact, "panel_path": str(PANEL_PATH), "panel_sha256": panel_sha}
        classes = collections.Counter((r.get("metadata") or {}).get("reply_class") for r in kept)
    if not kept:
        raise ValueError(f"rehearsal filter {select!r} selected no rows")
    audit = {"kept": len(kept), "dropped_panel_reply_class": dropped_panel,
             "dropped_multi_turn": dropped_multiturn, "dropped_by_filter": dropped_filter,
             "filter": select,
             "panel_reply_classes_excluded": sorted(PANEL_REPLY_CLASSES),
             "text_contact_audit": text_contact,
             "distinct_classes_kept": len(classes),
             "classes_kept": dict(sorted(classes.items()))}
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
    # A narrow filter may hold fewer tokens than the budget. Under-dosing silently would
    # confound composition with dose -- the one thing RRP-011 must not do -- so refuse.
    if used < token_budget:
        raise ValueError(
            f"rehearsal pool exhausted at {used} of {token_budget} processed tokens "
            f"({len(compiled)} rows). Running this arm would vary dose as well as "
            f"composition; widen the filter or lower the transition dose instead.")
    return compiled, {"rehearsal_rows": len(compiled), "rehearsal_processed_tokens": used,
                      "rehearsal_token_budget": token_budget, "rejected_overlength": rejected}


def interleave(transitions: list[dict], rehearsal: list[dict]) -> list[dict]:
    """One deterministic order over the union. No block structure the schedule could exploit."""
    return sorted(transitions + rehearsal,
                  key=lambda r: (digest([SALT, "order", r["id"]]), r["id"]))


def build(binding: dict, views: Path, rehearsal_path: Path, tokenizer, *,
          per_family: int, exposures: int = 1,
          selection_seed: int = process_arm.SEED,
          select: str = "all", also_exclude_views=(),
          historical_reproduction: bool = False) -> tuple[list[dict], dict, dict]:
    transitions, _, transition_report = process_arm.build(
        binding, views, tokenizer, per_family=per_family, exposures=exposures,
        selection_seed=selection_seed, experiment=EXPERIMENT,
        also_exclude_views=also_exclude_views,
        historical_reproduction=historical_reproduction)
    transition_tokens = sum(len(r["ids"]) for r in transitions)

    # A "+"-joined selector is an RRP-012 blend: the rehearsal budget is split EVENLY by
    # processed tokens across the named filters, not by pool size. Equal weighting is what
    # "span every capability you intend to keep" means operationally; letting the larger pool
    # dominate would not test that rule as stated.
    parts = select.split("+")
    if len(parts) == 1:
        pool, pool_audit = rehearsal_pool(rehearsal_path, select=select,
                                          historical_reproduction=historical_reproduction)
        rehearsal, rehearsal_audit = compile_rehearsal(pool, tokenizer,
                                                       token_budget=transition_tokens)
    else:
        share = transition_tokens // len(parts)
        rehearsal, pool_audit, rehearsal_audit = [], {}, {"components": {}}
        for i, part in enumerate(parts):
            # The last component absorbs the integer-division remainder.
            budget_i = transition_tokens - share * (len(parts) - 1) if i == len(parts) - 1 else share
            sub_pool, sub_pool_audit = rehearsal_pool(rehearsal_path, select=part,
                                                      historical_reproduction=historical_reproduction)
            sub, sub_audit = compile_rehearsal(sub_pool, tokenizer, token_budget=budget_i)
            rehearsal.extend(sub)
            pool_audit[part] = sub_pool_audit
            rehearsal_audit["components"][part] = sub_audit
        rehearsal_audit["rehearsal_rows"] = len(rehearsal)
        rehearsal_audit["rehearsal_processed_tokens"] = sum(r["processed_tokens"] for r in rehearsal)
        rehearsal_audit["rehearsal_token_budget"] = transition_tokens
        if len({r["id"] for r in rehearsal}) != len(rehearsal):
            raise ValueError("blend components overlap; a row would be trained twice")
    rows = interleave(transitions, rehearsal)

    # The probe measures the NEW skill, so it is drawn from the transition half only.
    plan = process_arm.build_plan(rows, exposures=exposures, experiment=EXPERIMENT)
    plan["probe_row_ids"] = process_arm.probe_subset(transitions, per_family=8)
    if "scored_instrument_exclusion_audit" in transition_report:
        plan["scored_instrument_exclusion_audit"] = transition_report["scored_instrument_exclusion_audit"]

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
                            "parent_ids_sha256": digest(sorted(r["parent_id"] for r in transitions)),
                            "historical_draw_identity": "NOT_CLAIMED; current instrument exclusions change the draw"},
        "rehearsal_source": str(rehearsal_path),
        "rehearsal_source_sha256": stable_hash(rehearsal_path),
        "rehearsal_pool_audit": pool_audit,
        "rehearsal_audit": rehearsal_audit,
        "rehearsal_filter": select,
        "rehearsal_never_scored": True,
        "total_rows": len(rows),
        "processed_tokens": plan["processed_tokens"],
        "supervised_tokens": plan["supervised_tokens"],
        "probe_drawn_from": "transition half only",
        "quality_claim": False,
        "confirmation": "NOT_ESTABLISHED",
    }
    for key in ("scored_instruments_excluded", "scored_instrument_exclusion_audit"):
        if key in transition_report:
            report[key] = transition_report[key]
    return rows, plan, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--exclude-views", type=Path, action="append", default=[],
                        help="Additional scored instruments to exclude. Must include the frozen "
                             "SELECT-VIEWS-001 instrument for current-gate training.")
    parser.add_argument("--rehearsal", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out-prepared", type=Path, required=True)
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--per-family", type=int, required=True)
    parser.add_argument("--exposures", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=process_arm.SEED)
    parser.add_argument("--rehearsal-select", default="all",
                        help="RRP-011 arm: which rehearsal rows are eligible. A '+'-joined "
                             "value is an RRP-012 blend and splits the rehearsal budget "
                             "EVENLY by processed tokens across the named filters, e.g. "
                             "abstention+reasoning")
    args = parser.parse_args(argv)

    from experiments.model_quality_gpu_v1.native import load_tokenizer

    binding = json.loads(Path(args.binding).read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(args.source)
    rows, plan, report = build(binding, args.views, args.rehearsal, tokenizer,
                               per_family=args.per_family, exposures=args.exposures,
                               selection_seed=args.selection_seed,
                               select=args.rehearsal_select,
                               also_exclude_views=args.exclude_views)
    process_arm.write_prepared(args.out_prepared, rows, report)
    write_new(args.out_plan, plan)

    pa = report["rehearsal_pool_audit"]
    print(f"rehearsal filter    : {report['rehearsal_filter']}")
    # A blend's pool audit is keyed by component, not flat. Summing the components keeps
    # this line meaningful instead of raising KeyError three lines into a successful build.
    if "kept" not in pa:
        for name, sub in pa.items():
            print(f"  component {name:<30} {sub['kept']:>5} eligible")
        pa = {"kept": sum(v["kept"] for v in pa.values()),
              "dropped_panel_reply_class": max(v["dropped_panel_reply_class"] for v in pa.values()),
              "dropped_multi_turn": max(v["dropped_multi_turn"] for v in pa.values()),
              "distinct_classes_kept": sum(v["distinct_classes_kept"] for v in pa.values())}
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
