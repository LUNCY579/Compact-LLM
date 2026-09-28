"""SELECT-VIEWS-001: an acquisition instrument that is admissible for SELECTION at birth.

GATE_V2_PREREG.md established that the gate cannot be extended with any existing instrument.
The long-context and abstention suites are barred by `QUARANTINE.json` (`SPENT`, bars include
`SELECTION`), and the existing 384-view acquisition probe is barred by its own declaration --
all 26 `PROCESS_*.json` files carry `"never for selection"` and all 384 views carry
`view_role: CANDIDATE_NOT_ADMITTED_OR_TRAINED`. A bar stated in the instrument binds as firmly
as one in the registry.

This module builds the replacement. The difference that matters is not the content -- it is the
same task, drawn from the same release, rendered through the same rule strings -- but the
DECLARATION. An instrument's admissibility is fixed when it is created and cannot be granted
retroactively, so this one says `ADMITTED_FOR_SELECTION` from the first byte it ever writes.

**Why historical training on a parent does not disqualify it.** Every arm in this programme
trains `--source Supa` -- verified across all 27 runner invocations -- so no arm inherits weights
from another. A parent trained in D4 has no effect whatsoever on a fresh arm started from the
frozen source. The contamination that matters is only within the arm being judged, and that is
handled by excluding this instrument's parents from that arm's training draw, exactly as the
existing probe is excluded. This matters practically: of 1,155 `prerequisite_plan` parents, 1,026
have been trained by some historical arm and 128 sit in the existing probe, leaving exactly
**one** untouched. Requiring never-trained parents would make the instrument unbuildable.

What IS required, and enforced here:
  - disjoint from the existing probe's 384 parents, so the two instruments stay independent;
  - one view per parent, 128 per family, matching the existing probe's geometry so the view
    remains the independent unit and no parent-level clustering correction is needed;
  - the identical per-family system rule, read from the existing probe rather than restated,
    so the two instruments measure the same task and cannot drift;
  - floors computed here and published BEFORE any model call.

Torch-free. No model is loaded and no tokenizer is required: views are rendered as chat messages
and scored by `process_probe.py`, which does its own tokenization.
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import digest, write_new

from . import process_arm

EXPERIMENT = "SELECT-VIEWS-001"

# A DIFFERENT salt from process_arm.SALT on purpose. It changes both which positive step is
# taken per parent and which parents are drawn, so this instrument is not a re-rendering of the
# existing probe under a new name.
SALT = "SELECT-VIEWS-001-acquisition-gate"

# The whole point of the module. This value must NOT assert "never for selection".
VIEW_ROLE = "ADMITTED_FOR_SELECTION"

PER_FAMILY = 128
ID_PREFIX = "RRP-select"


def render(row: dict, rules: dict[str, str]) -> dict:
    """One view, byte-compatible with what `process_probe.py` already consumes.

    The prompt is built exactly as `process_arm.compile_training` builds it -- same key order,
    same separators -- because a scored view whose serialization differs from the trained one
    would measure serialization rather than acquisition.
    """
    prompt = [{"role": "system", "content": rules[row["family"]]},
              {"role": "user", "content": json.dumps(
                  {"before": row["before"], "event": row["event"]},
                  sort_keys=True, separators=(",", ":"))}]
    return {
        "family": row["family"],
        "gold_verified_by": "oracle.replay_transition",
        "id": f"{ID_PREFIX}:{row['id']}",
        "parent_id": row["parent_id"],
        "prompt": prompt,
        "reference": json.dumps(row["proposed_after"], sort_keys=True, separators=(",", ":")),
        "source_id": row["id"],
        "source_split": row["source_split"],
        "view_role": VIEW_ROLE,
    }


def floors(rows: list[dict]) -> dict:
    """Shortcut floors, per family, computed before any model call.

    Two answer-agnostic predictors, matching `PROCESS_PROBE_FLOORS.md`:
      copy_before        -- emit `before` unchanged. Reads the prompt only.
      constant_majority  -- always emit the single commonest reference. Reads nothing.

    The binding floor is the STRONGER of the two, per family. It is not uniform across families
    and an aggregate quoted without the per-family split repeats CL-040.
    """
    per_family: dict[str, dict] = {}
    by_family: dict[str, list[dict]] = collections.defaultdict(list)
    for row in rows:
        by_family[row["family"]].append(row)
    for family, group in sorted(by_family.items()):
        canon = [json.dumps(r["proposed_after"], sort_keys=True, separators=(",", ":"))
                 for r in group]
        before = [json.dumps(r["before"], sort_keys=True, separators=(",", ":")) for r in group]
        copy_before = sum(1 for b, c in zip(before, canon) if b == c)
        counts = collections.Counter(canon)
        constant_majority = max(counts.values()) if counts else 0
        n = len(group)
        binding, predictor = ((copy_before, "copy_before")
                              if copy_before >= constant_majority
                              else (constant_majority, "constant_majority"))
        per_family[family] = {
            "n": n,
            "copy_before": copy_before,
            "constant_majority": constant_majority,
            "binding_floor_count": binding,
            "binding_floor_rate": round(binding / n, 4) if n else 0.0,
            "binding_floor_predictor": predictor,
        }
    return per_family


def build(binding: dict, existing_views: Path, *, per_family: int = PER_FAMILY,
          salt: str = SALT) -> tuple[list[dict], dict]:
    """Draw a fresh, selection-admissible partition disjoint from the existing probe."""
    rules = process_arm.rules_from_views(existing_views)
    excluded = process_arm.held_out_parents(existing_views)
    per_parent, label_audit = process_arm.verified_positives(binding, excluded, salt=salt)

    available = {f: len(p) for f, p in sorted(per_parent.items())}
    short = {f: n for f, n in available.items() if n < per_family}
    if short:
        raise ValueError(f"not enough parents outside the existing probe for {per_family} per "
                         f"family: {short}. Lower per_family or widen the source rather than "
                         f"silently drawing fewer.")

    chosen = process_arm.select(per_parent, per_family=per_family, salt=salt)
    views = [render(row, rules) for row in chosen]

    overlap = sorted({v["parent_id"] for v in views} & excluded)
    if overlap:
        raise ValueError(f"drew parents that are already in the existing probe: {overlap[:5]}")
    taken = collections.Counter(v["family"] for v in views)
    if set(taken.values()) != {per_family}:
        raise ValueError(f"unbalanced draw: {dict(taken)}")
    if len({v["parent_id"] for v in views}) != len(views):
        raise ValueError("a parent appears twice; the view is not the independent unit")

    report = {
        "record": "SELECT_VIEWS_001_BUILD",
        "schema": 1,
        "experiment": EXPERIMENT,
        "what_this_is": (
            "A fresh one-step state-transition acquisition instrument, declared admissible for "
            "SELECTION at creation. It exists because every acquisition instrument that "
            "predates it declares itself never-for-selection."),
        "view_role": VIEW_ROLE,
        "admissible_uses": ["SELECTION", "DEV"],
        "inadmissible_uses": ["TRAIN", "CONFIRMATION"],
        "why_not_confirmation": (
            "An instrument is spent for confirmation the moment it is used for selection. This "
            "one is built to be selected on, so it can never confirm."),
        "salt": salt,
        "per_family_requested": per_family,
        "per_family_taken": dict(sorted(taken.items())),
        "views": len(views),
        "distinct_parents": len({v["parent_id"] for v in views}),
        "disjoint_from_existing_probe": True,
        "existing_probe_parents_excluded": len(excluded),
        "available_parents_after_excluding_existing_probe": available,
        "source_label_audit": label_audit,
        "rules_taken_from": str(existing_views),
        "floors_published_before_any_model_call": floors(chosen),
        "decision_rule": (
            "a core clears a family only if the two-sided exact binomial 95% lower bound on "
            "canonical_correct exceeds that family's binding floor, where the binding floor is "
            "the STRONGER of copy_before and constant_majority"),
        "model_calls": 0,
        "optimizer_steps": 0,
        "training_admitted": False,
        "limitations": [
            "TRAIN-split derivative views; ADMITTED_FOR_SELECTION, never for confirmation",
            "single-step competence does not establish multi-step search or planning",
            "prerequisite_plan positives cover applicable actions only",
            "parents may have been trained by HISTORICAL arms; this is sound only because every "
            "arm trains from the frozen source and none inherits weights, so the exclusion that "
            "matters is from the training draw of the arm being judged",
            "exact-match scoring; strict vs canonical separates presentation from arithmetic",
        ],
    }
    return views, report


def write(out_dir: Path, views: list[dict], report: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "SELECT_VIEWS.jsonl"
    path.write_text("".join(json.dumps(v, sort_keys=True) + "\n" for v in views),
                    encoding="utf-8")
    from .sources import stable_hash
    report = {**report, "views_sha256": stable_hash(path)}
    write_new(out_dir / "BUILD.json", report)
    write_new(out_dir / "FLOORS.json", {
        "record": "SELECT_VIEWS_001_FLOORS",
        "experiment": EXPERIMENT,
        "published_before_any_model_call": True,
        "views_sha256": report["views_sha256"],
        "decision_rule": report["decision_rule"],
        "per_family": report["floors_published_before_any_model_call"],
    })


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--existing-views", type=Path, required=True,
                        help="the CANDIDATE_VIEWS.jsonl of the existing probe. Its parents are "
                             "excluded and its per-family system rules are reused verbatim.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-family", type=int, default=PER_FAMILY)
    args = parser.parse_args(argv)

    binding = json.loads(Path(args.binding).read_text(encoding="utf-8"))
    views, report = build(binding, args.existing_views, per_family=args.per_family)
    write(args.out, views, report)

    print(f"views                 : {report['views']} "
          f"({report['distinct_parents']} distinct parents)")
    print(f"per family taken      : {report['per_family_taken']}")
    print(f"view_role             : {report['view_role']}")
    print(f"disjoint from probe   : {report['disjoint_from_existing_probe']} "
          f"({report['existing_probe_parents_excluded']} parents excluded)")
    print("floors published before any model call:")
    for family, f in sorted(report["floors_published_before_any_model_call"].items()):
        print(f"  {family:<20} binding {f['binding_floor_count']}/{f['n']} = "
              f"{f['binding_floor_rate']:.4f} ({f['binding_floor_predictor']})  "
              f"[copy_before {f['copy_before']}, constant_majority {f['constant_majority']}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
