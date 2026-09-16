"""RRP-007 arm A1: teach one state transition, because neither core can do one.

PROCESS_TRANSITION_FLOOR.md measured Supa and Opallo on 384 frozen one-step transition
views and found no family cleared on either core, both **below** the shortcut floor:

    ledger_replay      supa 14/128 = 0.1094   floor 0.2422 copy_before        clears at 40
    register_machine   supa  0/128 = 0.0000   floor 0.0547 copy_before        clears at 12
    prerequisite_plan  supa  0/128 = 0.0000   floor 0.0703 constant_majority  clears at 15

The decomposition says what to target. On `ledger_replay` 114 of 128 predictions already
carry the EXACT key set and 36 miss exactly one value: the structure is learned and the
arithmetic on the touched field is not. On `register_machine` the contract is never reached
at all (0 correct key sets; it echoes its own input). On `prerequisite_plan` unwrapping the
`{"after": N}` envelope post-hoc lifts it only 0 -> 1, so the bitmask update itself is wrong.

So this arm is supervised fine-tuning on the SAME task form the probe scores, drawn from
parents the probe never touched. It changes the DATA, not the recipe: blocks 8-27, fp32,
CpuAdamW, the frozen 1e-5 WSD schedule, 2048 window -- every knob identical to the qualified
RRP-001 inventory.

Why this target rather than the v2 response contract:

  * it needs no window amendment. Rows are ~150 processed tokens against a 2048 qualified
    window, where the v2 reason view retains 0 of 2,000 ledger_replay rows at 2048;
  * the floor is published per family and both cores sit BELOW it, so a null is informative
    rather than uninterpretable -- there is room to move that a shortcut cannot fake;
  * every one of the 36,752 source labels was independently re-derived by
    `oracle.replay_transition` and agreed, and this module re-derives them again rather than
    trusting the recorded audit.

**Disjointness is the load-bearing property.** The 384 probe parents are excluded before
selection and the exclusion is verified afterwards. Training and evaluation share the task
and the rule text; they share no world.

The rule text is READ FROM THE FROZEN VIEWS rather than restated here, so the trained task
and the scored task cannot drift apart by a character. Only the per-family system rule is
taken -- never a view's data.

Torch-free at module level. `compile_row` needs a tokenizer, which the caller supplies.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
from pathlib import Path

from experiments.model_quality_gpu_v1.config import Design
from experiments.model_quality_gpu_v1.data import group_updates
from experiments.model_quality_gpu_v1.integrity import digest, write_new

from . import budget
from .arm import frozen_optimizer
from .serialization import compile_row
from .sources import import_forge_v2, stable_hash

EXPERIMENT = "RRP-007-PROCESS-A1"
OBJECTIVE = "assistant_response_plus_eos"
SALT = "RRP-007-process-a1"
SEED = 17
TRAIN_WINDOW = 2048
FAMILIES = ("ledger_replay", "prerequisite_plan", "register_machine")

# This arm's own ceiling. It does not widen RRP-001's, and a grant may sit anywhere at or
# below it. Sized from measured quantities, not guessed:
#   processed tokens -- 512 parents x 3 families x ~150 processed tokens is ~230k; 600k
#     leaves headroom for one retry at the same dose and nothing more.
#   wall seconds -- ~118 update groups at the measured 3.7 s/update is ~440 s, plus load.
#   disk -- engine.py checkpoints at group 2 and at the final group only, reserving
#     sum(numel) * 12 + 1 MiB each. For the 314,618,880-parameter tail that is ~3.52 GiB per
#     checkpoint, so two fit inside 8 GiB. The host had 33 GB free when this was written.
#   model calls -- the engine probes baseline/midpoint/final over 24 probe rows = 72, plus
#     the zero-update native qualification.
LIMITS = {
    "total_owned_work_wall_seconds": 3600,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 2400,
    "processed_training_tokens_all_attempts": 600000,
    "evaluation_and_qualification_model_calls": 240,
    "generated_tokens_all_candidates": 30720,
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 8 * 2**30,
    "adaptive_model_selection_rounds": 0,
    "model_worker_launches": 4,
    "retries": 1,
    "cloud_or_teacher_spend": 0,
}
EXCLUDES = ("RL", "larger_teacher", "cloud", "release", "architectural_change")
budget.register(EXPERIMENT, ceiling=LIMITS, phases=("train", "resume"), excludes=EXCLUDES)


def rules_from_views(views_path: Path) -> dict[str, str]:
    """The per-family system rule, taken from the frozen scored views.

    Restating the rule here would let the trained task and the scored task drift. Only the
    system string is read; no view's `before`, `event` or `reference` is touched.
    """
    rules: dict[str, str] = {}
    for line in Path(views_path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        system = row["prompt"][0]["content"]
        if rules.setdefault(row["family"], system) != system:
            raise ValueError(f"family {row['family']} has more than one system rule")
    if sorted(rules) != sorted(FAMILIES):
        raise ValueError(f"expected exactly {FAMILIES}, found {sorted(rules)}")
    return rules


def held_out_parents(views_path: Path) -> frozenset[str]:
    return frozenset(json.loads(l)["parent_id"]
                     for l in Path(views_path).read_text(encoding="utf-8").splitlines()
                     if l.strip())


def verified_positives(binding: dict, exclude: frozenset[str]) -> tuple[dict, dict]:
    """Every label re-derived by the release's own oracle. Positives only, minus held-out."""
    root = Path(binding["root"])
    relative = "process/train_transitions.jsonl.gz"
    source = root / relative
    if stable_hash(source) != binding["files"][relative]["sha256"]:
        raise ValueError("process release bytes changed")
    modules = import_forge_v2(binding)
    replay = modules["oracle"].replay_transition

    per_parent: dict[str, dict[str, dict]] = collections.defaultdict(dict)
    counts = collections.Counter()
    for line in gzip.open(source, "rt", encoding="utf-8"):
        row = json.loads(line)
        if row["source_split"] != "train" or row["parent_id"] != row["group_id"]:
            raise ValueError("non-TRAIN or broken parent identity")
        expected = replay(row["family"], row["before"], row["event"])
        if (expected == row["proposed_after"]) != row["valid"]:
            raise ValueError("process label fails the independent transition checker")
        counts[(row["family"], row["valid"])] += 1
        if not row["valid"] or row["parent_id"] in exclude:
            continue
        # One step per parent, chosen by a salted hash so the draw is deterministic and
        # independent of file order. Parent diversity beats extra steps from one world.
        best = per_parent[row["family"]].get(row["parent_id"])
        if best is None or digest([SALT, "step", row["id"]]) < digest([SALT, "step", best["id"]]):
            per_parent[row["family"]][row["parent_id"]] = row
    return per_parent, {"labels_checked": sum(counts.values()),
                        "labels_by_family": {f"{k[0]}|{k[1]}": v for k, v in sorted(counts.items())}}


def select(per_parent: dict, *, per_family: int) -> list[dict]:
    chosen = []
    for family in sorted(per_parent):
        pool = sorted(per_parent[family].values(),
                      key=lambda r: (digest([SALT, "parent", r["parent_id"]]), r["parent_id"]))
        chosen.extend(pool[:per_family])
    # One global deterministic order, independent of family grouping.
    return sorted(chosen, key=lambda r: (digest([SALT, "order", SEED, r["id"]]), r["id"]))


def compile_training(chosen: list[dict], rules: dict[str, str], tokenizer, *,
                     max_sequence: int = TRAIN_WINDOW) -> tuple[list[dict], list[dict]]:
    compiled, rejected = [], []
    for row in chosen:
        prompt = [{"role": "system", "content": rules[row["family"]]},
                  {"role": "user", "content": json.dumps(
                      {"before": row["before"], "event": row["event"]},
                      sort_keys=True, separators=(",", ":"))}]
        final = json.dumps(row["proposed_after"], sort_keys=True, separators=(",", ":"))
        try:
            unit = compile_row(prompt, final, tokenizer, max_sequence=max_sequence)
        except ValueError as exc:
            rejected.append({"id": row["id"], "reason": str(exc)})
            continue
        compiled.append({**unit, "id": row["id"], "parent_id": row["parent_id"],
                         "family": row["family"], "source_split": "train"})
    return compiled, rejected


def probe_subset(compiled: list[dict], *, per_family: int = 8) -> list[str]:
    buckets = collections.defaultdict(list)
    for row in compiled:
        buckets[row["family"]].append(row["id"])
    chosen = []
    for family in sorted(buckets):
        chosen.extend(sorted(buckets[family],
                             key=lambda i: digest([SALT, "probe", i]))[:per_family])
    return chosen


def build_plan(compiled: list[dict], *, exposures: int, max_new_tokens: int = 128,
               probe_per_family: int = 8, train_window: int = TRAIN_WINDOW) -> dict:
    """Computed from the compiled bytes so it cannot drift from them."""
    design = Design()
    repeated = [r for _ in range(exposures) for r in compiled]
    return {
        "experiment": EXPERIMENT,
        "objective": OBJECTIVE,
        "row_order_per_exposure": [r["id"] for r in compiled],
        "unique_worlds": len({r["parent_id"] for r in compiled}),
        "exposures_per_world": exposures,
        "presentations": len(repeated),
        "processed_tokens": sum(len(r["ids"]) for r in repeated),
        "supervised_tokens": sum(sum(t != -100 for t in r["labels"][1:]) for r in repeated),
        "update_groups": group_updates(repeated, design.update_tokens),
        "max_sequence": train_window,
        "max_new_tokens": max_new_tokens,
        "precision": "fp32",
        "trainable": "blocks8-27",
        "selection_seed": design.selection_seed,
        "order_seed": design.order_seed,
        "optimizer": frozen_optimizer(design),
        "probe_row_ids": probe_subset(compiled, per_family=probe_per_family),
    }


def write_prepared(directory: Path, compiled: list[dict], report: dict) -> dict:
    """Create-only prepared directory in the layout controller.inputs() verifies."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError(f"{directory} exists; prepared data is create-only")
    directory.mkdir(parents=True)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in compiled)
    (directory / "microfit-train.jsonl").write_bytes(body.encode("utf-8"))
    write_new(directory / "ARM_BUILD.json", report)
    contract = {
        "experiment": EXPERIMENT,
        "files": {"microfit-train.jsonl": stable_hash(directory / "microfit-train.jsonl"),
                  "ARM_BUILD.json": stable_hash(directory / "ARM_BUILD.json")},
        "rows": len(compiled),
        "source_release": "forge_v2",
        "source_split": "train",
        "source_view": "process/train_transitions.jsonl.gz positives only",
        "train_window": report["train_window"],
        "confirmation": "NOT_ESTABLISHED",
        "human_data_review": "NOT_PERFORMED",
        "held_out_parents_excluded": report["held_out_parents"],
        "disjointness_verified_against": ["process-transition-probe-001/CANDIDATE_VIEWS.jsonl"],
    }
    write_new(directory / "DATA_CONTRACT.json", contract)
    return contract


def build(binding: dict, views: Path, tokenizer, *, per_family: int, exposures: int,
          train_window: int = TRAIN_WINDOW) -> tuple[list[dict], dict, dict]:
    rules = rules_from_views(views)
    excluded = held_out_parents(views)
    per_parent, label_audit = verified_positives(binding, excluded)
    chosen = select(per_parent, per_family=per_family)
    compiled, rejected = compile_training(chosen, rules, tokenizer, max_sequence=train_window)
    if not compiled:
        raise ValueError(f"no process row fits the {train_window} training window")

    # Belt and braces: the exclusion above is enforced by construction, and this is what
    # catches a mistake in it. Training the scored worlds would invalidate the whole arm.
    collision = sorted({r["parent_id"] for r in compiled} & excluded)
    if collision:
        raise ValueError(f"training set collides with the scored views: {collision[:5]}")

    plan = build_plan(compiled, exposures=exposures, train_window=train_window)
    taken = collections.Counter(r["family"] for r in compiled)
    report = {
        "record": "RRP007_PROCESS_A1_BUILD",
        "schema": 1,
        "experiment": EXPERIMENT,
        "what_this_is": ("Supervised fine-tuning data for one-step state transitions, "
                         "selected from Forge v2 process positives on parents the scored "
                         "views never touched."),
        "salt": SALT, "seed": SEED,
        "train_window": train_window,
        "source_label_audit": label_audit,
        "held_out_parents": len(excluded),
        "available_parents_after_exclusion": {f: len(p) for f, p in sorted(per_parent.items())},
        "per_family_requested": per_family,
        "per_family_taken": dict(sorted(taken.items())),
        "compiled": len(compiled),
        "rejected_overlength": len(rejected),
        "rejected": rejected[:32],
        "processed_tokens": plan["processed_tokens"],
        "supervised_tokens": plan["supervised_tokens"],
        "scored_instrument": "process-transition-probe-001/CANDIDATE_VIEWS.jsonl",
        "rule_text_source": "read from the frozen scored views; never restated",
        "positives_only_note": ("valid=False rows carry a WRONG proposed_after by construction "
                                "and are excluded: they are usable for a binary judgement arm, "
                                "never as a generation target."),
        "quality_claim": False,
        "confirmation": "NOT_ESTABLISHED",
    }
    return compiled, plan, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binding", type=Path, required=True,
                        help="a1-prepared-003/SOURCE_BINDING.json")
    parser.add_argument("--views", type=Path, required=True,
                        help="process-transition-probe-001/CANDIDATE_VIEWS.jsonl")
    parser.add_argument("--source", type=Path, required=True, help="the Supa tokenizer assets")
    parser.add_argument("--out-prepared", type=Path, required=True)
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--per-family", type=int, required=True)
    parser.add_argument("--exposures", type=int, default=1)
    args = parser.parse_args(argv)

    from experiments.model_quality_gpu_v1.native import load_tokenizer

    binding = json.loads(Path(args.binding).read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(args.source)
    compiled, plan, report = build(binding, args.views, tokenizer,
                                   per_family=args.per_family, exposures=args.exposures)
    write_prepared(args.out_prepared, compiled, report)
    write_new(args.out_plan, plan)

    print(f"labels re-derived and agreed : {report['source_label_audit']['labels_checked']}")
    print(f"held-out parents excluded    : {report['held_out_parents']}")
    print(f"available after exclusion    : {report['available_parents_after_exclusion']}")
    print(f"compiled                     : {report['compiled']} "
          f"(rejected overlength {report['rejected_overlength']})")
    print(f"per family taken             : {report['per_family_taken']}")
    print(f"tokens                       : {plan['processed_tokens']} processed, "
          f"{plan['supervised_tokens']} supervised, {len(plan['update_groups'])} update groups")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
