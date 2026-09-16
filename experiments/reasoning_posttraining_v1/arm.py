"""RRP-003 arm A1: build the v2 response-contract training set, its contract, and its plan.

The arm targets the three stacked deficits the baseline isolated, all at once, because the
repair experiment already proved that fixing one alone buys nothing (CL-033):

  syntax    strict JSON with quoted array elements   -- 87 of 230 predictions failed only here
  schema    the right keys for this family           -- 31 repaired rows landed on `schema`
  contract  emit {"status": ...} when unanswerable   -- 34 landed on `answer_requires_status`

Underneath all three sits the planning itself, which v1 says is partly present (0.2553 against
a 0.1411 floor) while v2 says is absent at its contract (0.0174 against 0.4391).

Deliberately the SIMPLEST arm that addresses all three: supervised fine-tuning on Forge v2
`data/rl/train`, prompt -> gold JSON, with the recipe held byte-identical to the qualified
RRP-001 inventory. The arm changes the DATA, not the optimizer, precision, trainable tail or
window. One variable at a time.

Window choice, and it is forced rather than preferred: training at 2048 is already qualified
(RRP-001) and inference at 4096 is already qualified (CL-027), while training at 4096 is NOT
and would need its own backward-pass qualification on a host where RAM is already the binding
constraint. So the arm trains at 2048 on the third of v2 train that fits, and evaluates at
4096. That mismatch is a real confound, not a detail, and it is turned into a measurement: the
evaluation is additionally split by whether the row fits 2048, which reads out length
generalization directly instead of hiding it.

Torch-free at module level. `compile_row` needs a tokenizer, which the caller supplies.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import re
import unicodedata
from pathlib import Path

from experiments.model_quality_gpu_v1.config import Design
from experiments.model_quality_gpu_v1.data import group_updates
from experiments.model_quality_gpu_v1.integrity import digest, write_new

from . import quarantine
from .instrument import v2_stratum
from .serialization import compile_row
from .sources import stable_hash

ARM_EXPERIMENT = "RRP-003-ARM-A1"
OBJECTIVE = "assistant_response_plus_eos"
SALT = "RRP-003-arm-a1"
SEED = 17
TRAIN_WINDOW = 2048          # already qualified by RRP-001; not widened
EVAL_WINDOW = 4096           # already qualified for inference only (CL-027)

# The frozen optimizer design, restated so a mismatch is a build failure here rather than a
# refusal three minutes into a GPU launch. controller.design_for compares against this exactly.
def frozen_optimizer(design: Design) -> dict:
    return {"betas": [design.beta1, design.beta2], "epsilon": design.epsilon,
            "gradient_clip": design.clip_norm, "name": "MQ CpuAdamW",
            "peak_lr": 1e-5, "schedule": "WSD20/60/20_to_0.1_peak",
            "weight_decay": design.weight_decay}


def _rows(root: Path, rel: str):
    with gzip.open(Path(root) / rel, "rt", encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


def _norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def prompt_keys(prompt: list[dict]) -> dict[str, str]:
    """The same four axes the preservation admission audit uses, so the two are comparable."""
    if (not isinstance(prompt, list) or not prompt or
            any(not isinstance(m, dict) or not isinstance(m.get("content"), str)
                or not m.get("role") for m in prompt)):
        raise ValueError("leakage audit requires a valid prompt")
    turns = [m["content"] for m in prompt if m["role"] == "user"]
    if not turns or any(not text.strip() for text in turns):
        raise ValueError("leakage audit requires a nonempty user prompt")
    out = {}
    for name, text in (("first_user_turn", turns[0]), ("last_user_turn", turns[-1])):
        normal = _norm(text)
        out[name] = digest(normal)
        out[name + "_digit_blinded"] = digest(re.sub(r"\d", "#", normal))
    return out


def select(root: Path, *, per_bucket: int, exclude_ids: frozenset[str] = frozenset(),
           seed: int = SEED) -> tuple[list[str], dict, dict, dict]:
    """Stratified selection over (family, status) from v2 TRAIN.

    Stratifying matters for the same reason it did for the DEV bank: UNKNOWN appears in only
    two v2 families and IMPOSSIBLE in four, so a flat sample would leave the arm barely any
    abstention signal and confound status with family.
    """
    policy = {r["id"]: r for r in _rows(root, "data/rl/train.jsonl.gz")}
    meta = {m["id"]: m for m in _rows(root, "metadata/train.jsonl.gz")}
    if set(policy) != set(meta):
        raise ValueError("v2 train data/metadata id sets differ")

    buckets: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for world_id, m in meta.items():
        if world_id in exclude_ids:
            continue
        buckets[(m["family"], v2_stratum(m["reference"]))].append(world_id)

    chosen: list[str] = []
    per_bucket_taken = {}
    for key in sorted(buckets):
        pool = sorted(buckets[key], key=lambda i: (digest([SALT, seed, i]), i))
        take = pool[:per_bucket]
        per_bucket_taken["|".join(key)] = len(take)
        chosen.extend(take)
    return chosen, policy, meta, per_bucket_taken


def compile_training(chosen, policy, meta, tokenizer, *, max_sequence: int = TRAIN_WINDOW):
    """Compile to the native training contract. Overlength rows are REJECTED, never truncated."""
    compiled, rejected = [], []
    for world_id in chosen:
        row, m = policy[world_id], meta[world_id]
        try:
            got = compile_row(row["prompt"], m["reference_text"], tokenizer,
                              max_sequence=max_sequence)
        except ValueError as exc:
            rejected.append({"id": world_id, "family": m["family"],
                             "stratum": v2_stratum(m["reference"]), "reason": str(exc)})
            continue
        compiled.append({
            "id": world_id,
            "parent_id": m.get("group_id", world_id),
            "family": m["family"],
            "stratum": v2_stratum(m["reference"]),
            "source_split": "train",
            "source_release": "forge_v2",
            "mode": got["mode"],
            "prompt": row["prompt"],
            "reference_text": m["reference_text"],
            "reference": m["reference_text"],
            "ids": got["ids"], "prompt_ids": got["prompt_ids"], "labels": got["labels"],
            "answer_start": got["answer_start"], "answer_end": got["answer_end"],
            "processed_tokens": got["processed_tokens"],
            "supervised_tokens": got["supervised_tokens"],
            "truncated": got["truncated"],
            "case_sha256": m["layout"]["case_sha256"],
            "prompt_sha256": m["prompt_sha256"],
        })
    return compiled, rejected


def disjointness(compiled: list[dict], banks: dict[str, list[dict]]) -> dict:
    """ZERO tolerance against every DEV bank. Any hit is a build failure, not a warning.

    The release asserts train/validation disjointness. The programme's rule is that an
    assertion in a document is not evidence, so it is re-measured here on four axes.
    """
    if not compiled or not banks or any(not rows for rows in banks.values()):
        raise ValueError("training and every DEV bank must be nonempty")
    ids = {r["id"] for r in compiled}
    parents = {r["parent_id"] for r in compiled}
    cases = {r["case_sha256"] for r in compiled}
    axes = list(prompt_keys(compiled[0]["prompt"]))
    train_keys = {a: set() for a in axes}
    for r in compiled:
        for a, v in prompt_keys(r["prompt"]).items():
            train_keys[a].add(v)

    results = []
    for name, rows in sorted(banks.items()):
        hits = collections.Counter()
        for row in rows:
            if row.get("id") in ids:
                hits["id"] += 1
            if any(row.get(key) in parents for key in ("parent_id", "group_id", "id")):
                hits["parent_id"] += 1
            if row.get("case_sha256") in cases:
                hits["case_sha256"] += 1
            keys = prompt_keys(row.get("prompt"))
            for a, v in keys.items():
                if v in train_keys[a]:
                    hits[a] += 1
        results.append({"bank": name, "rows": len(rows), "collisions": dict(hits)})
    clean = all(not r["collisions"] for r in results)
    return {"axes": ["id", "parent_id", "case_sha256", *axes], "tolerance": "ZERO",
            "against": results, "clean": clean}


def probe_subset(compiled: list[dict], *, per_bucket: int) -> list[str]:
    """A small stratified probe for the engine's three in-training evaluations.

    The engine probes at baseline, midpoint and final over whatever row set it is given.
    RRP-001's probe was its whole 32-world microfit, which is cheap. Probing an arm's whole
    training set would cost thousands of evaluation calls to measure MEMORIZATION, which is
    not the question. So the probe is a stratified handful, and the held-out banks are what
    answer the actual question.
    """
    buckets = collections.defaultdict(list)
    for row in compiled:
        buckets[(row["family"], row["stratum"])].append(row["id"])
    chosen = []
    for key in sorted(buckets):
        pool = sorted(buckets[key], key=lambda i: (digest([SALT, "probe", i]), i))
        chosen.extend(pool[:per_bucket])
    return chosen


def build_plan(compiled: list[dict], *, exposures: int, max_new_tokens: int = 128,
               probe_per_bucket: int = 8, train_window: int = TRAIN_WINDOW) -> dict:
    """The frozen plan, computed from the compiled bytes so it cannot drift from them."""
    design = Design()
    repeated = [r for _ in range(exposures) for r in compiled]
    processed = sum(len(r["ids"]) for r in repeated)
    supervised = sum(sum(t != -100 for t in r["labels"][1:]) for r in repeated)
    groups = group_updates(repeated, design.update_tokens)
    return {
        "experiment": ARM_EXPERIMENT,
        "objective": OBJECTIVE,
        "row_order_per_exposure": [r["id"] for r in compiled],
        "unique_worlds": len({r["parent_id"] for r in compiled}),
        "exposures_per_world": exposures,
        "presentations": len(repeated),
        "processed_tokens": processed,
        "supervised_tokens": supervised,
        "update_groups": groups,
        "max_sequence": train_window,
        "max_new_tokens": max_new_tokens,
        "precision": "fp32",
        "trainable": "blocks8-27",
        "selection_seed": design.selection_seed,
        "order_seed": design.order_seed,
        "optimizer": frozen_optimizer(design),
        "probe_row_ids": probe_subset(compiled, per_bucket=probe_per_bucket),
    }


def write_prepared(directory: Path, compiled: list[dict], report: dict) -> dict:
    """Create-only prepared directory in the layout controller.inputs() verifies."""
    directory = Path(directory)
    if directory.exists():
        raise ValueError(f"{directory} exists; prepared data is create-only")
    directory.mkdir(parents=True)
    body = "".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n" for r in compiled)
    # The filename controller.presentations() reads. Kept identical so the arm reuses the
    # already-qualified loading path rather than introducing a second one.
    (directory / "microfit-train.jsonl").write_bytes(body.encode("utf-8"))
    write_new(directory / "ARM_BUILD.json", report)
    contract = {
        "experiment": ARM_EXPERIMENT,
        "files": {"microfit-train.jsonl": stable_hash(directory / "microfit-train.jsonl"),
                  "ARM_BUILD.json": stable_hash(directory / "ARM_BUILD.json")},
        "rows": len(compiled),
        "source_release": "forge_v2",
        "source_split": "train",
        # Single source of truth: the build report already carries the window the rows were
        # actually compiled at. Re-deriving it here is how the two can disagree.
        "train_window": report["train_window"],
        "confirmation": "NOT_ESTABLISHED",
        "human_data_review": "NOT_PERFORMED",
        "disjointness_verified_against": sorted(report["disjointness"]["against"][i]["bank"]
                                                for i in range(len(report["disjointness"]["against"]))),
    }
    write_new(directory / "DATA_CONTRACT.json", contract)
    return contract


def cap_per_bucket(compiled: list[dict], *, per_bucket: int) -> list[dict]:
    """Balance the buckets AFTER compilation, which is the only place balance is real.

    Only about a third of v2 TRAIN fits the 2048 training window, and the fit rate varies
    enormously by family -- `evidence_chain` contributes 4 of 887 ANSWER rows. Capping before
    compilation therefore produces a set whose measured composition is nothing like the
    requested one: a first attempt asked for 60 per bucket and got 275 rows with one family
    represented by a single row. Selecting generously and capping afterwards makes the
    registered balance hold on the rows that actually train.
    """
    buckets = collections.defaultdict(list)
    for row in compiled:
        buckets[(row["family"], row["stratum"])].append(row)
    kept = []
    for key in sorted(buckets):
        kept.extend(buckets[key][:per_bucket])
    # Preserve the deterministic global order rather than the bucket-walk order.
    order = {row["id"]: i for i, row in enumerate(compiled)}
    return sorted(kept, key=lambda r: order[r["id"]])


def build(v2_root: Path, tokenizer, registry: dict, banks: dict[str, list[dict]], *,
          per_bucket: int, exposures: int,
          candidate_multiple: int = 6,
          train_window: int = TRAIN_WINDOW) -> tuple[list[dict], dict, dict]:
    """Select, compile, balance, prove disjoint, and plan. No model call, no budget."""
    # Every DEV id is excluded up front, so disjointness is enforced by construction as well
    # as verified afterwards. Belt and braces on purpose: the verification is what catches a
    # mistake in the exclusion.
    dev_ids = frozenset(row["id"] for rows in banks.values() for row in rows
                        if isinstance(row.get("id"), str))
    # Select generously, because only ~a third will fit the training window, then cap.
    candidates = per_bucket * candidate_multiple
    chosen, policy, meta, per_bucket_offered = select(v2_root, per_bucket=candidates,
                                                      exclude_ids=dev_ids)
    quarantine.refuse(chosen, registry, use="TRAIN")

    fitting, rejected = compile_training(chosen, policy, meta, tokenizer,
                                        max_sequence=train_window)
    if not fitting:
        raise ValueError(f"no v2 train row fits the {train_window} training window")
    compiled = cap_per_bucket(fitting, per_bucket=per_bucket)
    per_bucket_taken = dict(collections.Counter(
        f"{r['family']}|{r['stratum']}" for r in compiled))
    short = {k: v for k, v in per_bucket_taken.items() if v < per_bucket}
    audit = disjointness(compiled, banks)
    if not audit["clean"]:
        raise ValueError(f"training set collides with a DEV bank: {audit['against']}")

    plan = build_plan(compiled, exposures=exposures, train_window=train_window)
    report = {
        "record": "RRP003_ARM_A1_BUILD",
        "schema": 1,
        "experiment": ARM_EXPERIMENT,
        "what_this_is": ("Supervised fine-tuning data for the v2 response contract and its "
                         "reasoning, selected from Forge v2 TRAIN only."),
        "salt": SALT, "seed": SEED,
        "train_window": train_window,
        "evaluation_window": EVAL_WINDOW,
        "window_note": (("Training at 2048 is already qualified (RRP-001) and inference at "
                         "4096 is already qualified (CL-027). The train/eval window mismatch is "
                         "a real confound and is reported as a length-generalization split, not "
                         "hidden.") if train_window == TRAIN_WINDOW else
                        (f"Training at {train_window} is NOT qualified by RRP-001. This build is "
                         "only admissible inside a registered window-qualification experiment "
                         "that measures peak memory at the maximum planned shape; it carries no "
                         "training-quality authority.")),
        "candidates_offered": len(chosen),
        "candidates_that_fit": len(fitting),
        "compiled": len(compiled),
        "fit_rate": round(len(fitting) / len(chosen), 4),
        "rejected_overlength": len(rejected),
        "rejected": rejected[:32],
        "per_bucket_requested": per_bucket,
        "candidate_multiple": candidate_multiple,
        "per_bucket_offered": per_bucket_offered,
        "per_bucket_taken": per_bucket_taken,
        "buckets_short_of_the_request": short,
        "buckets_short_note": (f"a bucket short of the request has no more rows that fit "
                               f"{train_window}. "
                               "evidence_chain is the extreme case -- 4 of 887 ANSWER and 9 of "
                               "1113 UNKNOWN fit -- so the arm barely sees it and NO per-family "
                               "claim about that family will be interpretable."),
        "composition": {
            "per_family": dict(collections.Counter(r["family"] for r in compiled)),
            "per_stratum": dict(collections.Counter(r["stratum"] for r in compiled)),
            "unique_parents": len({r["parent_id"] for r in compiled}),
            "unique_cases": len({r["case_sha256"] for r in compiled}),
        },
        "token_inventory": {
            "processed_tokens_per_exposure": sum(len(r["ids"]) for r in compiled),
            "supervised_tokens_per_exposure": sum(r["supervised_tokens"] for r in compiled),
            "exposures": exposures,
            "processed_tokens_total": plan["processed_tokens"],
            "supervised_tokens_total": plan["supervised_tokens"],
            "update_groups": len(plan["update_groups"]),
        },
        "disjointness": audit,
        "quarantine_sha256": registry["sha256"],
        "model_calls": 0,
        "quality_claim": False,
    }
    return compiled, plan, report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--quarantine", type=Path, required=True)
    parser.add_argument("--bank", type=Path, action="append", required=True,
                        help="a DEV bank jsonl to prove disjointness against; repeatable")
    parser.add_argument("--out-prepared", type=Path, required=True)
    parser.add_argument("--out-plan", type=Path, required=True)
    parser.add_argument("--per-bucket", type=int, required=True)
    parser.add_argument("--exposures", type=int, required=True)
    parser.add_argument("--train-window", type=int, default=TRAIN_WINDOW,
                        help="training sequence window; anything other than the qualified "
                             "2048 is admissible only inside a registered window "
                             "qualification")
    args = parser.parse_args(argv)

    from experiments.model_quality_gpu_v1.native import load_tokenizer

    registry = quarantine.load(args.quarantine)
    banks = {}
    for path in args.bank:
        banks[Path(path).stem] = [json.loads(l) for l in
                                  Path(path).read_text(encoding="utf-8").splitlines()
                                  if l.strip()]
    tokenizer = load_tokenizer(args.source)
    compiled, plan, report = build(args.v2_root, tokenizer, registry, banks,
                                   per_bucket=args.per_bucket, exposures=args.exposures,
                                   train_window=args.train_window)
    contract = write_prepared(args.out_prepared, compiled, report)
    write_new(args.out_plan, plan)

    print(f"offered {report['candidates_offered']} candidates, "
          f"{report['candidates_that_fit']} fit {report['train_window']} "
          f"(rate {report['fit_rate']}), "
          f"kept {report['compiled']} after capping at {report['per_bucket_requested']}/bucket")
    if report["buckets_short_of_the_request"]:
        print(f"buckets SHORT of the request: {report['buckets_short_of_the_request']}")
    print(f"per stratum : {report['composition']['per_stratum']}")
    print(f"per family  : {report['composition']['per_family']}")
    print(f"tokens      : {plan['processed_tokens']} processed, "
          f"{plan['supervised_tokens']} supervised, "
          f"{len(plan['update_groups'])} updates over {args.exposures} exposures")
    print(f"disjointness: clean={report['disjointness']['clean']} "
          f"against {[b['bank'] for b in report['disjointness']['against']]}")
    print(f"contract    : {contract['files']}")
    print(f"plan digest : {digest(plan)}")


if __name__ == "__main__":
    main()
