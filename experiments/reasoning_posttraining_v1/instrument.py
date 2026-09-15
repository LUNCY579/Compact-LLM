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
    import torch

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
                 completion_reserve: int = 128) -> tuple[list[dict], list[dict]]:
    """Tokenize the bank for native evaluation. Overlength rows are REJECTED, not truncated.

    Returns (evaluable, rejected). The caller must report both denominators: at a 2048
    window only about 65% of Forge v1 validation fits, so the evaluable subset is
    systematically shorter than the bank and cannot be described as the release
    distribution.
    """
    evaluable, rejected = [], []
    for row in bank:
        try:
            compiled = compile_row(row["prompt"], row["reference"], tokenizer,
                                   max_sequence=max_sequence,
                                   completion_reserve=completion_reserve)
        except ValueError as exc:
            rejected.append({"id": row["id"], "family": row["family"],
                             "stratum": row["stratum"], "reason": str(exc),
                             "actual_user_words": row.get("actual_user_words")})
            continue
        evaluable.append({
            "id": row["id"], "parent_id": row.get("group_id", row["id"]),
            "family": row["family"], "stratum": row["stratum"],
            "reference": row["reference"], "prompt": row["prompt"],
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
