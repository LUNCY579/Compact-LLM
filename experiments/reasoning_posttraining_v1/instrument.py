"""DEV bank assembly, strata, positive control and model-free shortcut baselines.

Nothing here loads a model, touches CUDA or spends budget. Its job is to make a model
number interpretable: a bank is only usable once the independent exact solver scores
100% on it, and a model result is only meaningful once reported beside the shortcut
floor that any trivial strategy already achieves.

Forge v1 encodes the answer as a plain string, and an unanswerable world carries a
literal sentinel as its answer. Those sentinels mean different things and are kept apart.
`BOTH` is a genuine logical verdict, not an abstention.
"""
from __future__ import annotations

import collections
import json
import re
from pathlib import Path

from experiments.model_quality_gpu_v1.integrity import digest, read_json

from . import budget
from .serialization import compile_row
from .sources import import_forge_v1_validator

# ---------------------------------------------------------------- RRP-002 baseline lane
#
# A held-out measurement of an EXISTING checkpoint. It performs no optimizer step, writes
# no checkpoint, and selects no model, so processed_training_tokens and
# adaptive_model_selection_rounds are pinned at zero and the journal will refuse any
# charge against them. Registered as its own experiment rather than widening RRP-001's
# envelope, whose adaptive round is already spent.
BASELINE_EXPERIMENT = "RRP-002-BASELINE"
BASELINE_LIMITS = {
    "total_owned_work_wall_seconds": 5400,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 3600,
    "processed_training_tokens_all_attempts": 0,
    "evaluation_and_qualification_model_calls": 1600,
    "generated_tokens_all_candidates": 131072,
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 268435456,
    "adaptive_model_selection_rounds": 0,
    "model_worker_launches": 4,
    "retries": 2,
    "cloud_or_teacher_spend": 0,
}
BASELINE_PHASES = ("baseline",)
BASELINE_EXCLUDES = ("broad_SFT", "RL", "larger_teacher", "cloud", "release",
                     "architectural_change", "training")

budget.register(BASELINE_EXPERIMENT, ceiling=BASELINE_LIMITS, phases=BASELINE_PHASES,
                excludes=BASELINE_EXCLUDES)

# Amendment A2. The original envelope allowed 4 launches and 2 retries, sized for a clean
# run. Three launches were spent and two of them died on THIS AGENT'S implementation
# defects, not on anything the science required: a missing torch.no_grad in evaluate(),
# and a NameError from a lazily-imported symbol that was not imported. Compute was barely
# touched -- 260 of 3600 GPU-seconds and 241 of 1600 evaluation calls -- so only the
# launch and retry counters are raised. Every other lane keeps its original value, and
# training, model selection, RL and cloud stay pinned at zero. This is a SUCCESSOR with
# its own journal rather than an edit to the exhausted one, so the original hash-chained
# spend history stays immutable.
BASELINE_A2_EXPERIMENT = "RRP-002-BASELINE-A2"
BASELINE_A2_LIMITS = {**BASELINE_LIMITS, "model_worker_launches": 8, "retries": 6}

budget.register(BASELINE_A2_EXPERIMENT, ceiling=BASELINE_A2_LIMITS,
                phases=BASELINE_PHASES, excludes=BASELINE_EXCLUDES)

# Amendment A3 corrects a SCOPE defect, not a compute shortfall. A2 declares
# evaluation_bank = RRP2_DEV_V1_BANK.jsonl and scopes itself to that bank, but run
# baseline-v2-4k-001 evaluated RRP2-DEV-V2 under the A2 lane. The numeric envelope was
# enforced throughout and the kind of work was exactly what A2 authorized, but the declared
# bank was not the bank used and no amendment recorded the widening (CL-045).
#
# So A3 widens the declared scope to the three registered DEV banks and its ceiling is A2's
# REMAINDER, read off the A2 journal. A2 + A3 together authorize exactly what A2 alone
# authorized: the net new compute is zero on every lane. Correcting a governance defect must
# not be a route to a bigger envelope.
BASELINE_A3_EXPERIMENT = "RRP-002-BASELINE-A3"
BASELINE_A3_LIMITS = {
    "total_owned_work_wall_seconds": 3942,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 2143,
    "processed_training_tokens_all_attempts": 0,
    "evaluation_and_qualification_model_calls": 1029,
    "generated_tokens_all_candidates": 59008,
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 260046848,
    "adaptive_model_selection_rounds": 0,
    "model_worker_launches": 6,
    "retries": 5,
    "cloud_or_teacher_spend": 0,
}

budget.register(BASELINE_A3_EXPERIMENT, ceiling=BASELINE_A3_LIMITS,
                phases=BASELINE_PHASES, excludes=BASELINE_EXCLUDES)

# ---------------------------------------------------------------- RRP-003 arm A1 lane
#
# THE FIRST LANE IN THIS PROGRAMME AUTHORIZED TO TRAIN. Every lane before it listed
# "training" in excludes and pinned processed_training_tokens_all_attempts at 0; this one
# does neither, and that is the whole point of registering it separately rather than
# widening an existing envelope. Everything else stays pinned: no RL, no cloud or teacher
# spend, no model selection round, no release.
#
# Sized from measurement, not preference. The pool measurement found 5,297 of 16,000 v2
# TRAIN rows fit the qualified 2048 window at a mean 1,603 processed tokens, and RRP-001
# measured ~3.7 s per optimizer update. A 60-per-bucket dose probe is ~733 rows, ~1.18M
# processed tokens and ~574 updates, i.e. roughly 35 minutes of GPU. The ceiling below
# covers about two such attempts plus their held-out evaluations, and nothing more: a first
# arm should be a dose probe that can fail cheaply, not a two-hour commitment on a host
# where RAM is already the binding constraint.
ARM_A1_EXPERIMENT = "RRP-003-ARM-A1"
ARM_A1_LIMITS = {
    "total_owned_work_wall_seconds": 16000,
    "model_worker_wall_seconds_and_conservative_gpu_seconds": 12000,
    "processed_training_tokens_all_attempts": 4000000,
    "evaluation_and_qualification_model_calls": 3000,
    "generated_tokens_all_candidates": 262144,
    "on_policy_rl_rollout_tokens": 0,
    "new_disk_bytes": 10737418240,
    "adaptive_model_selection_rounds": 0,
    "model_worker_launches": 4,
    "retries": 3,
    "cloud_or_teacher_spend": 0,
}
ARM_A1_PHASES = ("train", "resume")
# "training" is deliberately ABSENT. "broad_SFT" too: this is bounded, registered SFT on a
# declared TRAIN split, which is what the arm is for. Everything else stays barred.
ARM_A1_EXCLUDES = ("RL", "larger_teacher", "cloud", "release", "architectural_change")

budget.register(ARM_A1_EXPERIMENT, ceiling=ARM_A1_LIMITS, phases=ARM_A1_PHASES,
                excludes=ARM_A1_EXCLUDES)

# Distinct meanings, deliberately not collapsed into one "unanswerable" bucket.
SENTINELS = {
    "UNKNOWN": "information required to answer is absent",
    "IMPOSSIBLE": "no valid ordering/assignment exists",
    "UNREACHABLE": "no path exists between the endpoints",
    "INCONSISTENT": "the supplied premises contradict each other",
}
# Answers that look like sentinels but are real verdicts.
VERDICTS = ("BOTH", "SUPPORTED", "REFUTED")
# Families whose gold is prose; the release scorer refuses to grade them.
PROSE_FAMILIES = ("communication", "grounded_summary")


def load_v1(bindings: Path):
    """Import the v1 read-only text solver after re-checking its recorded hash."""
    return import_forge_v1_validator(read_json(bindings)["forge_v1"])


def stratum(reference: str) -> str:
    """ANSWER unless the whole answer is one of the four unanswerable sentinels."""
    text = reference.strip()
    if text in SENTINELS:
        return text
    return "ANSWER"


def canonical(text: str) -> str:
    """The frozen primitive target contract: whitespace-normalized, nothing else."""
    return " ".join(text.strip().split())


def tolerant(text: str) -> str:
    """Contract-tolerant form: also drop one trailing sentence mark and outer quotes.

    Case is preserved on purpose. Identifiers like RoomDMKTM are case-bearing, so
    lowercasing would manufacture credit rather than tolerate formatting.
    """
    value = canonical(text)
    value = value.strip('"').strip("'").strip()
    while value[-1:] in (".", "!", "?"):
        value = value[:-1].rstrip()
    return value


def score(prediction: str, reference: str) -> dict:
    """Strict and tolerant correctness as separately observable quantities."""
    strict = canonical(prediction) == canonical(reference)
    loose = tolerant(prediction) == tolerant(reference)
    return {"strict": strict, "tolerant": loose,
            # A tolerant-only hit is a presentation failure, not a reasoning failure.
            "presentation_only": loose and not strict,
            "stratum": stratum(reference),
            "predicted_sentinel": canonical(prediction) in SENTINELS,
            "reference_sentinel": canonical(reference) in SENTINELS}


# ---------------------------------------------------------------- bank assembly

def build_bank(root: Path, split: str, *, families: tuple[str, ...], per_family: int,
               solver, seed: int = 17, exclude_ids: frozenset[str] = frozenset()) -> list[dict]:
    """Select `per_family` worlds per family from one split, validating every row.

    Selection is by a stable digest of (bank tag, seed, id), so it is reproducible and
    independent of file order, and it never consults an answer or a model outcome.
    """
    rows = {r["id"]: r for r in solver.iter_jsonl(root / "data" / f"{split}.jsonl.gz")}
    metas = {m["id"]: m for m in solver.iter_jsonl(root / "metadata" / f"{split}.jsonl.gz")}
    if set(rows) != set(metas):
        raise ValueError("data/metadata id sets differ")

    chosen: list[dict] = []
    for family in families:
        pool = [i for i, m in metas.items()
                if m["family"] == family and i not in exclude_ids]
        if len(pool) < per_family:
            raise ValueError(f"insufficient {family} worlds: {len(pool)}/{per_family}")
        pool.sort(key=lambda i: digest(["RRP-002-bank", seed, i]))
        for world_id in pool[:per_family]:
            row, meta = rows[world_id], metas[world_id]
            # The release's own row validator: it re-solves the prompt with the text
            # oracle, re-checks every hash and word count, and refuses leaked metadata.
            solver.validate_row(row, meta)
            messages = row["messages"]
            chosen.append({
                "id": world_id,
                "group_id": meta.get("group_id", world_id),
                "family": family,
                "split": split,
                "source_release": "forge_v1",
                "prompt": [messages[0], messages[1]],
                "reference": messages[2]["content"],
                "stratum": stratum(messages[2]["content"]),
                "answer_category": meta["answer_category"],
                "prompt_sha256": meta["prompt_sha256"],
                "case_sha256": meta["case_sha256"],
                "actual_user_words": meta["actual_user_words"],
                "core_record_count": meta["core_record_count"],
                "added_distractor_blocks": meta.get("added_distractor_blocks"),
                "position_mode": meta.get("position_mode"),
                "difficulty": meta.get("difficulty"),
            })
    return chosen


# ---------------------------------------------------------------- Forge v2 lane
#
# v2 is where the stratified abstention analysis belongs: it carries a structured
# reference.status with UNKNOWN and IMPOSSIBLE spanning several families, which v1
# structurally cannot supply (CL-019). Its answers are strict JSON objects, and four of
# its eight families are optimization problems where a DIFFERENT plan of equal cost is
# genuinely correct -- so it must be scored with the release's audited
# rewards.score_response and NEVER by text-matching reference_text (CL-021).
V2_FAMILIES = ("budget_selection", "continuous_replan", "deadline_schedule", "evidence_chain",
               "ledger_replay", "prerequisite_plan", "register_machine", "temporal_route")
V2_STATUSES = ("ANSWER", "UNKNOWN", "IMPOSSIBLE")
# An empty/degenerate witness per answer shape, for the model-free floor.
V2_EMPTY_WITNESS = {
    "path": {"cost": 0, "path": []},
    "actions": {"actions": [], "cost": 0},
    "order": {"finish": 0, "order": []},
    "items": {"cost": 0, "items": [], "value": 0},
    "value": {"value": 0},
    "answer": {"answer": "S0"},
    "status": {"status": "UNKNOWN"},
}


def v2_stratum(reference: dict) -> str:
    """ANSWER unless the reference carries an explicit status sentinel."""
    return reference.get("status", "ANSWER")


def _v2_shape(reference: dict) -> str:
    for key in ("path", "actions", "order", "items", "value", "answer", "status"):
        if key in reference:
            return key
    raise ValueError(f"unrecognized v2 answer shape: {sorted(reference)}")


def build_bank_v2(root: Path, split: str, *, per_bucket: int, seed: int = 17,
                  exclude_ids: frozenset[str] = frozenset()) -> list[dict]:
    """Select `per_bucket` worlds for every family x status bucket present in `split`.

    Stratifying on family x status is deliberate: UNKNOWN appears in only two v2 families
    and IMPOSSIBLE in four, so a flat sample would leave the abstention strata thin and
    family-confounded exactly as v1's are.
    """
    import gzip

    def rows(rel):
        with gzip.open(root / rel, "rt", encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    yield json.loads(line)

    policy = {r["id"]: r for r in rows(f"data/rl/{split}.jsonl.gz")}
    meta = {m["id"]: m for m in rows(f"metadata/{split}.jsonl.gz")}
    if set(policy) != set(meta):
        raise ValueError("v2 data/metadata id sets differ")

    buckets: dict[tuple[str, str], list[str]] = collections.defaultdict(list)
    for world_id, m in meta.items():
        if world_id in exclude_ids:
            continue
        buckets[(m["family"], v2_stratum(m["reference"]))].append(world_id)

    chosen: list[dict] = []
    for key in sorted(buckets):
        pool = sorted(buckets[key], key=lambda i: digest(["RRP-002-bank-v2", seed, i]))
        for world_id in pool[:per_bucket]:
            p, m = policy[world_id], meta[world_id]
            if [x["role"] for x in p["prompt"]] != ["system", "user"]:
                raise ValueError("v2 policy prompt must be exactly system+user")
            # What the model can actually read. This MUST be the raw concatenated content:
            # json.dumps() escapes the inner quotes, so a needle of {"status":"UNKNOWN"}
            # could never match a haystack of {\"status\":\"UNKNOWN\"}. That defect made this
            # check report 0 leaks where there were 144 (CL-040).
            visible = "\n".join(x["content"] for x in p["prompt"])
            # An id in the prompt is a genuine build defect: refuse it.
            if world_id in visible:
                raise ValueError(f"world id leaked into the v2 prompt: {world_id}")
            # The gold text is DIFFERENT. Forge v2 states the abstention contract in its own
            # rule text -- "If either limit has no approved record, return
            # {"status":"UNKNOWN"}" -- so for an unanswerable world the complete gold object
            # is printed to the model by design. That is the release telling the model the
            # contract, not a selection error, and refusing it would make the bank
            # unbuildable. So it is MEASURED and carried on the row, and it obliges the floor
            # to include `copy_prompt_sentinel`. A bank property that is recorded and priced
            # in is honest; one that is asserted absent is not.
            prints_gold = m["reference_text"] in visible
            chosen.append({
                "id": world_id, "group_id": m["group_id"], "family": m["family"],
                "split": split, "source_release": "forge_v2",
                "prompt": p["prompt"],
                "stratum": v2_stratum(m["reference"]),
                "answer_shape": _v2_shape(m["reference"]),
                "prompt_prints_its_gold": prints_gold,
                "prompt_status_sentinel": _prompt_sentinel(p["prompt"]),
                # verifier-only below this line; never placed in a model-visible prompt
                "reference": m["reference"], "reference_text": m["reference_text"],
                "world": m["world"], "certificate": m.get("certificate"),
                "prompt_sha256": m["prompt_sha256"],
                "case_sha256": m["layout"]["case_sha256"],
                "layout": m["layout"]["layout"],
                "distractor_records": m["layout"]["distractor_records"],
                "input_words": m["layout"]["input_words"],
                "generation_tier": m["generation"]["tier"],
                "transfer": m["generation"]["transfer"],
            })
    return chosen


def positive_control_v2(bank: list[dict], modules) -> dict:
    """The shipped gold witness must verify on every row, via the audited verifier."""
    rewards = modules["rewards"]
    wrong, reasons = [], collections.Counter()
    for row in bank:
        got = rewards.score_response(row["world"], row["reference_text"],
                                     certificate=row.get("certificate"))
        reasons[got["reason"]] += 1
        if not got["correct"]:
            wrong.append({"id": row["id"], "family": row["family"],
                          "reason": got["reason"], "reference_text": row["reference_text"]})
    return {"record": "RRP002_V2_POSITIVE_CONTROL_GOLD_WITNESS",
            "verifier": "forge_v2 forge_rl.rewards.score_response -> oracle.verify_solution",
            "rows": len(bank), "verified": len(bank) - len(wrong), "failed": len(wrong),
            "reasons": dict(sorted(reasons.items())),
            "status": "PASS" if not wrong else "FAIL", "failures": wrong[:20],
            "note": ("audit_answer returns a certificate rather than a plan for the "
                     "optimization families, so it cannot act as a predictor; the gold-witness "
                     "check is the available exact control."),
            "model_calls": 0}


_BARE_IN_ARRAY = re.compile(r'(?<=[\[,])\s*([A-Za-z][A-Za-z0-9_]*)\s*(?=[,\]])')


def v2_repair(text: str) -> tuple[str, list[str]]:
    """One minimal, auditable repair: quote bare identifiers inside JSON arrays.

    The measured failure it targets is exactly `{"path":[N3,N1],"cost":12}`, which is
    structurally correct with the right keys and plausible values and fails only because
    the array elements are unquoted. Measured on the 230 v2 baseline predictions: 91 failed
    to parse and 87 of those are repaired by this rule (CL-041 corrects an earlier 70).

    This is a DIAGNOSTIC channel and must never be the headline number: primary accuracy
    stays pass@1 under the declared strict-JSON policy. It exists to separate a
    presentation failure from a reasoning failure (bottleneck B-3), not to inflate a score.
    Nothing here consults a reference, and no value is ever altered -- only quoting.
    """
    applied = []
    fixed = _BARE_IN_ARRAY.sub(lambda m: f'"{m.group(1)}"', text)
    if fixed != text:
        applied.append("quoted_bare_identifiers_in_arrays")
    return fixed, applied


def score_v2(prediction: str, row: dict, modules) -> dict:
    """One prediction, scored by the audited verifier. Channels stay separate."""
    got = modules["rewards"].score_response(row["world"], prediction,
                                            certificate=row.get("certificate"))
    predicted_status = None
    if got["format_valid"]:
        try:
            predicted_status = json.loads(prediction.strip()).get("status")
        except (ValueError, AttributeError):
            predicted_status = None
    return {"correct": got["correct"], "format_valid": got["format_valid"],
            "schema_valid": got["schema_valid"], "feasible": got["feasible"],
            "reason": got["reason"], "reward": got["reward"],
            "stratum": row["stratum"], "predicted_status": predicted_status}


# Keyed on answer_shape, which is DERIVED FROM THE REFERENCE. It is therefore
# oracle-assisted, not answer-agnostic, and must never count toward the shortcut floor.
# Kept as a diagnostic answering "what would a trivial witness score if the answer shape
# were handed to you?" It was caught only because a test asserts the honest baselines are
# invariant to stripping the reference -- though that test was itself too weak to catch a
# NEW leak of this class, since it did not strip answer_shape (CL-043).
V2_ORACLE_ASSISTED = ("oracle_shape_empty_witness",)

# The v2 rule text prints the abstention contract as a complete JSON object. Measured on
# RRP2-DEV-V2: 288 of 336 prompts print exactly ONE `{"status":"X"}` literal and 48 print
# none, and for every one of the 144 unanswerable rows the printed literal IS the gold. So
# copying it is a pure prompt-surface shortcut -- the direct v2 analogue of v1's
# `copy_question_sentinel` (CL-022), which the v2 floor was missing entirely (CL-040).
_V2_SENTINEL = re.compile(r'\{"status"\s*:\s*"([A-Z_]+)"\}')
# Declared fallback for the 48 prompts that print no literal, so the predictor stays a total
# function of the visible prompt and never consults a reference.
V2_SENTINEL_FALLBACK = '{"value":0}'


def _prompt_sentinel(prompt: list[dict]) -> str | None:
    """The single status literal the prompt prints, or None. Reads only the prompt."""
    found = _V2_SENTINEL.findall("\n".join(x["content"] for x in prompt))
    return found[-1] if found else None


def shortcut_predictions_v2(bank: list[dict], *, train_majority: str) -> dict[str, list[str]]:
    """v2 predictions. Every name outside V2_ORACLE_ASSISTED is strictly answer-agnostic,
    reading only the model-visible prompt and a TRAIN-fitted constant."""
    out: dict[str, list[str]] = collections.defaultdict(list)
    for row in bank:
        question = row["prompt"][1]["content"]
        out["constant_unknown"].append(json.dumps({"status": "UNKNOWN"}))
        out["constant_impossible"].append(json.dumps({"status": "IMPOSSIBLE"}))
        out["constant_train_majority"].append(train_majority)
        out["zero_value"].append(json.dumps({"value": 0}))
        # ORACLE-ASSISTED, quarantined by V2_ORACLE_ASSISTED: the shape comes from the
        # reference, so this is a diagnostic and not part of any floor.
        out["oracle_shape_empty_witness"].append(
            json.dumps(V2_EMPTY_WITNESS[row["answer_shape"]]))
        # Prose: establishes that the format channel really does floor at zero.
        out["prose_non_json"].append("The answer is zero.")
        # Copy the status literal the prompt itself prints. Answer-agnostic -- it reads the
        # prompt and nothing else -- and it is the strongest honest shortcut on this bank, so
        # it belongs IN the floor. Omitting it understated the v2 floor by 14 points (CL-040).
        sentinel = _prompt_sentinel(row["prompt"])
        out["copy_prompt_sentinel"].append(
            json.dumps({"status": sentinel}, separators=(",", ":")) if sentinel
            else V2_SENTINEL_FALLBACK)
        # Echo an integer out of the QUERY line without solving anything.
        match = re.search(r"QUERY\s*(\{.*\})", question)
        value = 0
        if match:
            numbers = _INT.findall(match.group(1))
            value = int(numbers[-1]) if numbers else 0
        out["copy_query_integer"].append(json.dumps({"value": value}))
    return dict(out)


def summarize_v2(bank: list[dict], predictions: list[str], modules) -> dict:
    """Stratified v2 summary keeping format, schema, feasibility and semantics apart."""
    per_stratum = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    per_family = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    per_shape = collections.defaultdict(lambda: {"n": 0, "correct": 0})
    reasons = collections.Counter()
    fmt = schema = 0
    false_abstain = unsupported = wrong_sentinel = unparseable = 0
    for row, pred in zip(bank, predictions, strict=True):
        got = score_v2(pred, row, modules)
        reasons[got["reason"]] += 1
        fmt += got["format_valid"]
        schema += got["schema_valid"]
        for table, key in ((per_stratum, row["stratum"]), (per_family, row["family"]),
                           (per_shape, row["answer_shape"])):
            table[key]["n"] += 1
            table[key]["correct"] += got["correct"]
        truth, guess = row["stratum"], got["predicted_status"]
        if truth == "ANSWER" and guess in ("UNKNOWN", "IMPOSSIBLE"):
            false_abstain += 1
        elif truth != "ANSWER" and guess is None:
            # `predicted_status is None` covers TWO different things, and booking both as an
            # unsupported answer overstates the model's confidence: it reads a row that did
            # not parse at all as a deliberate answer to an unanswerable world. The v2
            # baseline's 99/99 was 34 parse failures plus 65 real unsupported answers
            # (CL-042). Kept apart.
            if got["format_valid"]:
                unsupported += 1
            else:
                unparseable += 1
        elif truth != "ANSWER" and guess is not None and guess != truth:
            wrong_sentinel += 1
    total = len(bank)
    correct = sum(v["correct"] for v in per_stratum.values())
    return {
        "rows": total, "correct": correct,
        "accuracy": correct / total if total else None,
        "format_valid": fmt, "schema_valid": schema,
        "false_abstention_on_answerable": false_abstain,
        "unsupported_answer_on_unanswerable": unsupported,
        "unparseable_on_unanswerable": unparseable,
        "wrong_sentinel_on_unanswerable": wrong_sentinel,
        "per_stratum": {k: dict(v) for k, v in sorted(per_stratum.items())},
        "per_family": {k: dict(v) for k, v in sorted(per_family.items())},
        "per_answer_shape": {k: dict(v) for k, v in sorted(per_shape.items())},
        "verifier_reasons": dict(sorted(reasons.items())),
        "cluster_unit": "family; paired variants share group_id",
        "scored_by": "forge_rl.rewards.score_response (alternative optima accepted)",
        "quality_claim": False,
    }


def qualify_evaluation_window(model, tokenizer, window: int, reserve, guard, *,
                              keep: int = 8) -> dict:
    """Qualify an EVALUATION window wider than the trained 2048. Forward-only.

    This qualifies inference at `window` tokens: cached/uncached agreement, causality and
    finite logits at full length, with the source parameters proven unchanged. It does
    NOT qualify training at that window -- no backward pass is run and no optimizer state
    is allocated, so a training claim at this window would need its own qualification.

    Only the last `keep` positions' logits are materialized. A full logit tensor at 4096
    tokens is 4096 x 151936 x 4 bytes, about 2.5 GiB, and two of them plus the fp32 model
    exceed this host's free VRAM. The kept suffix tests the property that matters: that
    position `window` is computed correctly given `window - 1` tokens of real context.
    """
    # Imported lazily and TOGETHER: this module is imported by the foreground controller
    # and must stay torch-free at module level, so every torch-dependent name has to be
    # resolved here. parameter_digest lives in a torch module and was missed once, which
    # cost a supervised launch to a NameError.
    import torch

    from experiments.model_quality_gpu_v1.native import parameter_digest

    if type(window) is not int or not 2048 < window <= 16384:
        raise ValueError("evaluation-window qualification covers 2048 < window <= 16384")
    device = next(model.parameters()).device
    before = parameter_digest(model)

    # A separately named mechanical shape fixture, never truncated task evidence.
    ordinary = tokenizer.encode("x y", add_special_tokens=False)
    if not ordinary or tokenizer.eos_token_id in ordinary:
        raise ValueError("shape fixture token identity invalid")
    sequence = (ordinary * (window // len(ordinary) + 1))[:window]
    sequence[-1] = tokenizer.eos_token_id
    ids = torch.tensor([sequence], device=device)

    reserve({"evaluation_and_qualification_model_calls": 4})
    model.eval()
    guard()
    with torch.no_grad():
        direct = model(input_ids=ids, use_cache=False, return_dict=True,
                       logits_to_keep=keep).logits
        if not torch.isfinite(direct).all():
            raise FloatingPointError("nonfinite logits at the candidate evaluation window")
        guard()
        # Same sequence, streamed through the KV cache in chunks, then compared on the
        # same kept suffix. This is the RoPE/position-handling check at full length.
        past, tail = None, None
        step = 512
        for start in range(0, window, step):
            part = model(input_ids=ids[:, start:start + step], past_key_values=past,
                         use_cache=True, return_dict=True, logits_to_keep=keep)
            past = part.past_key_values
            tail = part.logits
            guard()
        cached = tail
        torch.testing.assert_close(direct, cached, atol=2e-3, rtol=2e-3)
        cache_error = float((direct - cached).abs().max())
        del past, tail, cached
        # Causality: perturbing a token INSIDE the kept suffix must not move the logits
        # that precede it.
        changed = ids.clone()
        changed[0, -3] = tokenizer.encode("8", add_special_tokens=False)[0]
        perturbed = model(input_ids=changed, use_cache=False, return_dict=True,
                          logits_to_keep=keep).logits
        torch.testing.assert_close(direct[:, :keep - 3], perturbed[:, :keep - 3],
                                   atol=0, rtol=0)
        if bool((direct[:, keep - 3:] == perturbed[:, keep - 3:]).all()):
            raise ValueError("perturbation had no effect; the probe is not exercising context")
        del direct, perturbed, changed

    peak = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else None
    after = parameter_digest(model)
    if before != after:
        raise RuntimeError("window qualification changed source parameters")
    return {"phase": "RRP_NATIVE_EVALUATION_WINDOW_QUALIFICATION",
            "status": "PASS", "window": window, "kept_logit_positions": keep,
            "optimizer_steps": 0, "backward_passes": 0,
            "cached_vs_uncached_max_abs": cache_error,
            "source_state_before": before, "source_state_after": after,
            "peak_vram_bytes": peak,
            "checks": ["finite_logits_at_window", "cached_uncached_agreement_at_window",
                       "causality_within_kept_suffix", "context_actually_used",
                       "frozen_source_unchanged"],
            "qualifies": "INFERENCE ONLY at this window",
            "does_not_qualify": ("training at this window; no backward pass was run and no "
                                 "optimizer state was allocated"),
            "quality_claim": False}


def compile_bank(bank: list[dict], tokenizer, *, max_sequence: int = 2048,
                 completion_reserve: int = 128,
                 reference_field: str = "reference",
                 family_field: str = "family",
                 stratum_field: str = "stratum") -> tuple[list[dict], list[dict]]:
    """Tokenize the bank for native evaluation. Overlength rows are REJECTED, not truncated.

    Returns (evaluable, rejected). The caller must report both denominators: at a 2048
    window only about 65% of Forge v1 validation fits, so the evaluable subset is
    systematically shorter than the bank and cannot be described as the release
    distribution.

    `family_field` and `stratum_field` exist because the three bank kinds name their grouping
    and stratification columns differently, and hardcoding the Forge names made the
    preservation bank die with KeyError: 'family' after a 35-second launch (CL-048). They are
    passed EXPLICITLY by the caller rather than sniffed, and they default to the Forge names so
    the v1/v2 paths are byte-for-byte unchanged.
    """
    evaluable, rejected = [], []
    for row in bank:
        if family_field not in row or stratum_field not in row:
            raise ValueError(
                f"row {row.get('id')!r} has no {family_field!r}/{stratum_field!r}; pass the "
                f"field names this bank kind actually uses instead of relying on a default")
        try:
            compiled = compile_row(row["prompt"], row[reference_field], tokenizer,
                                   max_sequence=max_sequence,
                                   completion_reserve=completion_reserve)
        except ValueError as exc:
            rejected.append({"id": row["id"], "family": row[family_field],
                             "stratum": row[stratum_field], "reason": str(exc),
                             "actual_user_words": row.get("actual_user_words")})
            continue
        evaluable.append({
            "id": row["id"], "parent_id": row.get("group_id", row["id"]),
            "family": row[family_field], "stratum": row[stratum_field],
            "reference": row[reference_field], "prompt": row["prompt"],
            "ids": compiled["ids"], "prompt_ids": compiled["prompt_ids"],
            "labels": compiled["labels"], "source_split": "validation",
            "processed_tokens": compiled["processed_tokens"],
            "supervised_tokens": compiled["supervised_tokens"],
        })
    return evaluable, rejected


def positive_control(bank: list[dict], solver) -> dict:
    """The independent text solver must reproduce every gold. Anything less is a broken bank."""
    wrong, by_family = [], collections.Counter()
    for row in bank:
        got = solver.solve_prompt(row["family"], row["prompt"][1]["content"])
        if canonical(got) != canonical(row["reference"]):
            wrong.append({"id": row["id"], "family": row["family"],
                          "solver": got, "reference": row["reference"]})
        else:
            by_family[row["family"]] += 1
    return {"record": "RRP002_POSITIVE_CONTROL_EXACT_SOLVER",
            "solver": "forge_v1 scripts/verify.py solve_prompt (visible text only)",
            "rows": len(bank), "solved": len(bank) - len(wrong), "failed": len(wrong),
            "per_family_solved": dict(sorted(by_family.items())),
            "status": "PASS" if not wrong else "FAIL",
            "failures": wrong[:20],
            "meaning": ("A bank the exact solver cannot solve is broken. A broken instrument "
                        "is not a model failure."),
            "model_calls": 0}


# ---------------------------------------------------------------- shortcut baselines

_IDENT = re.compile(r"\b[A-Z][A-Za-z]*[A-Z0-9]\w*\b")
_INT = re.compile(r"-?\d+")


def _sections(row: dict, solver) -> tuple[str, str]:
    """(records-and-rules text, question text) exactly as the release splits it."""
    return solver.parts(row["prompt"][1]["content"])


def _cue_terms(question: str) -> list[str]:
    """Identifiers in the question, EXCLUDING the sentinel words the instruction leaks.

    Every v1 question ends with wording like "Return the value only or UNKNOWN.", so the
    literal sentinel is always present as a capitalized token. Copying it is not a cue
    strategy, it is exploiting the instruction text, and that is measured separately by
    `copy_question_sentinel`.
    """
    return [t for t in _IDENT.findall(question)
            if t not in SENTINELS and t not in VERDICTS]


def _question_sentinel(question: str) -> str:
    """The sentinel the question itself offers, if any. Makes the surface leak explicit."""
    offered = [t for t in _IDENT.findall(question) if t in SENTINELS]
    return offered[-1] if offered else ""


def shortcut_predictions(bank: list[dict], solver, *, train_majority: str) -> dict[str, list[str]]:
    """Model-free predictions. None of these may read a reference.

    `train_majority` must be fitted on TRAIN only and passed in, so this function cannot
    peek at the bank it is scoring.
    """
    out: dict[str, list[str]] = collections.defaultdict(list)
    for row in bank:
        body, question = _sections(row, solver)
        cues = set(_cue_terms(question))
        body_idents = [t for t in _IDENT.findall(body) if t not in cues]
        body_ints = _INT.findall(body)

        out["constant_unknown"].append("UNKNOWN")
        out["constant_train_majority"].append(train_majority)
        out["last_mentioned"].append(body_idents[-1] if body_idents else "")
        out["first_mentioned"].append(body_idents[0] if body_idents else "")
        out["last_number"].append(body_ints[-1] if body_ints else "")
        # cue_copy: echo the question's own salient identifier back as the answer
        terms = _cue_terms(question)
        out["cue_copy"].append(terms[-1] if terms else "")
        # The instruction text names the sentinel, so copying it is free. Measured
        # separately, because it sets the real floor for any abstention claim.
        out["copy_question_sentinel"].append(_question_sentinel(question))
        # signed-ledger specific: opening balance plus only the final delta (RRP-001 B-1)
        out["last_delta_only"].append(_last_delta_only(body, question))
    return dict(out)


def _last_delta_only(body: str, question: str) -> str:
    account = re.search(r"account (\w+)", question)
    if account is None:
        return ""
    name = account.group(1)
    opening = re.search(rf"Opening: account {name} held (-?\d+) units", body)
    deltas = [int(d) for a, d, s in
              re.findall(r"Entry: receipt \w+; account (\w+); delta ([+-]\d+); state (\w+)\.", body)
              if a == name and s == "POSTED"]
    if opening is None or not deltas:
        return "UNKNOWN"
    return str(int(opening.group(1)) + deltas[-1])


def sibling_collision(bank: list[dict], predictions: list[str]) -> dict:
    """Does an error equal another world's gold IN THE SAME FAMILY? Tests RRP-001 B-1.

    Returns the raw rate only. There is deliberately NO analytic chance level here: the
    probability that a wrong answer coincidentally equals a sibling gold depends entirely
    on the predictor's output distribution, and any closed-form estimate would be
    arbitrary. The comparison baseline must be supplied empirically by running this on the
    answer-agnostic shortcut predictors, which cannot be doing memorization interference
    by construction. Compare a candidate model's rate against those.
    """
    golds = collections.defaultdict(list)
    for row in bank:
        golds[row["family"]].append(canonical(row["reference"]))
    errors = collisions = sentinel_collisions = 0
    examples = []
    for row, pred in zip(bank, predictions, strict=True):
        own = canonical(row["reference"])
        got = canonical(pred)
        if got == own:
            continue
        errors += 1
        siblings = {g for g in golds[row["family"]] if g != own}
        if got in siblings:
            # A sentinel is a generic abstention shared by many worlds. Landing on it is
            # not evidence of recalling a SPECIFIC sibling world, so it is counted apart.
            if got in SENTINELS:
                sentinel_collisions += 1
            else:
                collisions += 1
                if len(examples) < 10:
                    examples.append({"id": row["id"], "family": row["family"],
                                     "predicted": got, "own_gold": own})
    return {"errors": errors,
            "specific_sibling_collisions": collisions,
            "collision_rate": (collisions / errors) if errors else None,
            "sentinel_collisions_excluded": sentinel_collisions,
            "sentinel_collision_rate": (sentinel_collisions / errors) if errors else None,
            "examples": examples,
            "why_sentinels_are_excluded": ("A constant-UNKNOWN predictor otherwise scores a "
                                           "0.707 collision rate on this bank purely because "
                                           "UNKNOWN is also a sibling gold in most families. "
                                           "That is abstention, not interference."),
            "baseline_required": ("Compare against this same statistic computed on the "
                                  "answer-agnostic shortcut predictors. A model rate at or "
                                  "below the shortcut rate refutes the B-1 interference reading."),
            "no_analytic_chance_level": ("Intentionally omitted: it would depend on the "
                                         "predictor's output distribution and any formula "
                                         "would be arbitrary.")}


def summarize(bank: list[dict], predictions: list[str]) -> dict:
    """Stratified, family-clustered summary. Denominators are always explicit."""
    per_stratum = collections.defaultdict(lambda: {"n": 0, "strict": 0, "tolerant": 0})
    per_family = collections.defaultdict(lambda: {"n": 0, "strict": 0})
    presentation_only = false_abstain = unsupported_answer = 0
    for row, pred in zip(bank, predictions, strict=True):
        got = score(pred, row["reference"])
        s = per_stratum[got["stratum"]]
        s["n"] += 1
        s["strict"] += got["strict"]
        s["tolerant"] += got["tolerant"]
        f = per_family[row["family"]]
        f["n"] += 1
        f["strict"] += got["strict"]
        presentation_only += got["presentation_only"]
        if got["reference_sentinel"] and not got["predicted_sentinel"]:
            unsupported_answer += 1
        if got["predicted_sentinel"] and not got["reference_sentinel"]:
            false_abstain += 1
    total = len(bank)
    strict = sum(v["strict"] for v in per_stratum.values())
    return {
        "rows": total,
        "strict_correct": strict,
        "strict_accuracy": strict / total if total else None,
        "tolerant_correct": sum(v["tolerant"] for v in per_stratum.values()),
        "presentation_only_gain": presentation_only,
        "false_abstention_on_answerable": false_abstain,
        "unsupported_answer_on_unanswerable": unsupported_answer,
        "per_stratum": {k: dict(v) for k, v in sorted(per_stratum.items())},
        "per_family": {k: dict(v) for k, v in sorted(per_family.items())},
        "cluster_unit": "family (paired variants share group_id)",
        "quality_claim": False,
    }
