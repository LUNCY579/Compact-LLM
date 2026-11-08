"""Stdlib-only, prospective E1 schedule and optimizer-boundary accounting.

This module neither loads data files nor imports a tokenizer/model/training stack.
The caller must supply freshly validated, bound TRAIN records. It never accesses
CHECK/FINAL, runs an optimizer, restores a checkpoint, or grants native execution.
The registered loss is mean completion-token CE per sequence, followed by the
mean of the 20 sequence losses in an optimizer update (microbatch size one).

External identities use these meanings:
* parent_sha256: exact unchanged Titan weight digest;
* model_sha256: PRIMARY_STUDENT_RECEIPT.json byte digest;
* data_sha256: normalized records plus their file-binding aggregate digest;
* source_sha256: critical planner/binder/config source inventory digest;
* config_sha256: frozen E1 planning configuration byte digest.
The outer manifest records the Git commit separately. A resume cursor validates
schedule/accounting only; it cannot establish optimizer, checkpoint, or native
training-RNG integrity. The shuffle RNG is bound separately from training RNG.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import hashlib
import json
import platform
import random
import re
import sys


SEED = 101
OPTIMIZER_STEPS = 256
ACQUISITION_PER_STEP = 16
REPLAY_PER_STEP = 4
SEQUENCES_PER_STEP = 20
WARMUP_STEPS = 13  # ceil(0.05 * 256), prospectively fixed; no configurable dose.
PEAK_LEARNING_RATE = 1e-4
CHECKPOINT_STEPS = (0, 64, 128, 192, 256)
COMPONENT_COUNTS = {
    "acquisition": 4096,
    "deterministic_replay": 768,
    "language_replay": 256,
}
ROW_FIELDS = frozenset({
    "key", "component", "role", "parent_id", "nonpadding_tokens", "completion_tokens",
})
IDENTITY_FIELDS = frozenset({
    "parent_sha256", "model_sha256", "data_sha256", "source_sha256", "config_sha256",
})
MAX_NONPADDING_TOKENS = 5_242_880
MAX_SEQUENCE_TOKENS = 1024
MAX_COMPLETION_TOKENS = 128


def canonical_bytes(value) -> bytes:
    """Unambiguous UTF-8 JSON used for every digest, with nonfinite values refused."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def sha256_json(value) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _identities(identities: dict) -> dict:
    _require(type(identities) is dict and set(identities) == IDENTITY_FIELDS,
             "exact five external identity digests required")
    for name, digest in identities.items():
        _require(type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest) is not None,
                 f"invalid SHA256 identity: {name}")
    return dict(identities)


def schedule_config() -> dict:
    """Return a new copy of the fixed prospective contract; accepts no overrides."""
    return {
        "schema_version": 1,
        "seed": SEED,
        "confirmation_seeds": "211/307 refused: no separately bound passing DEV screen implemented",
        "epochs": 1,
        "component_rows": dict(COMPONENT_COUNTS),
        "optimizer_steps": OPTIMIZER_STEPS,
        "microbatch_sequences": 1,
        "sequences_per_optimizer_step": SEQUENCES_PER_STEP,
        "acquisition_per_optimizer_step": ACQUISITION_PER_STEP,
        "replay_per_optimizer_step": REPLAY_PER_STEP,
        "interleave": ["acquisition"] * 4 + ["replay"],
        "interleave_repeats_per_step": 4,
        "family_quotas_per_step": None,
        "loss": {
            "token_reduction": "mean CE over completion tokens including EOS in each sequence",
            "sequence_reduction": "arithmetic mean of 20 sequence losses per optimizer step",
            "prompt_and_padding": "masked from the objective",
            "implementation_status": "PROPOSED_NO_TRAINING_LOOP",
        },
        "optimizer": {
            "name": "AdamW", "peak_learning_rate": PEAK_LEARNING_RATE,
            "betas": [0.9, 0.999], "epsilon": 1e-8, "weight_decay": 0,
            "gradient_clip_norm": 1.0,
        },
        "learning_rate": {
            "warmup_steps": WARMUP_STEPS,
            "rounding": "ceil(0.05 * 256)",
            "index": "1-based optimizer update; LR applied to that update",
            "warmup_formula": "peak * u / 13 for 1 <= u <= 13",
            "decay_formula": "peak * (256 - u) / 243 for 14 <= u <= 256",
            "final_update_learning_rate": 0.0,
        },
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "dev_evaluation_steps": list(CHECKPOINT_STEPS),
        "candidate_selection": "final checkpoint at optimizer step 256 only",
        "held_out_access": {"CHECK": False, "FINAL": False},
        "limits": {
            "max_sequence_nonpadding_tokens": MAX_SEQUENCE_TOKENS,
            "max_sequence_completion_tokens": MAX_COMPLETION_TOKENS,
            "max_nonpadding_training_tokens_per_seed": MAX_NONPADDING_TOKENS,
            "max_gpu_hours_per_seed": 24,
            "max_wall_hours_per_seed": 24,
            "max_generated_tokens_per_seed_including_failures": 262_144,
            "teacher_calls": 0,
        },
    }


def rng_contract() -> dict:
    """Bind the scheduling algorithm and Python runtime, without training RNG claims."""
    return {
        "schema_version": 1,
        "algorithm": "stdlib random.Random / MT19937 / shuffle",
        "seed": SEED,
        "python_implementation": platform.python_implementation(),
        "python_version": list(sys.version_info[:3]),
        "random_state_version": random.Random.VERSION,
        "input_order": "preserve bound record order within acquisition and combined replay",
        "calls": ["shuffle(acquisition)", "shuffle(combined_replay)"],
        "resume": "index immutable planned rows; never reshuffle",
    }


def learning_rate(update: int) -> float:
    _require(type(update) is int and 1 <= update <= OPTIMIZER_STEPS,
             "learning rate requires a 1-based update in 1..256")
    if update == WARMUP_STEPS:
        return PEAK_LEARNING_RATE
    if update < WARMUP_STEPS:
        return PEAK_LEARNING_RATE * update / WARMUP_STEPS
    return PEAK_LEARNING_RATE * (OPTIMIZER_STEPS - update) / (OPTIMIZER_STEPS - WARMUP_STEPS)


def _validated_rows(rows) -> list[dict]:
    _require(type(rows) in (list, tuple), "bound rows must be an ordered list or tuple")
    _require(len(rows) == sum(COMPONENT_COUNTS.values()), "exactly 5120 TRAIN rows required")
    clean, keys, acquisition_parents, parent_components = [], set(), set(), {}
    counts = Counter()
    for row in rows:
        _require(type(row) is dict and set(row) == ROW_FIELDS, "bound row schema mismatch")
        for field in ("key", "parent_id"):
            _require(type(row[field]) is str and bool(row[field].strip()), f"invalid row {field}")
        _require(row["role"] == "TRAIN", "schedule accepts TRAIN only; held-out roles remain closed")
        component = row["component"]
        _require(type(component) is str and component in COMPONENT_COUNTS, "unknown TRAIN component")
        _require(row["key"] not in keys, "duplicate TRAIN row key")
        keys.add(row["key"])
        parent = row["parent_id"]
        if component == "acquisition":
            _require(parent not in acquisition_parents, "duplicate acquisition parent")
            acquisition_parents.add(parent)
        _require(parent_components.get(parent, component) == component,
                 "parent cannot cross acquisition/replay component boundaries")
        parent_components[parent] = component
        for field, ceiling in (("nonpadding_tokens", MAX_SEQUENCE_TOKENS),
                               ("completion_tokens", MAX_COMPLETION_TOKENS)):
            _require(type(row[field]) is int and 1 <= row[field] <= ceiling,
                     f"invalid or over-ceiling {field}; target truncation forbidden")
        _require(row["completion_tokens"] < row["nonpadding_tokens"],
                 "sequence must contain both prompt and completion tokens")
        counts[component] += 1
        clean.append(dict(row))
    _require(dict(counts) == COMPONENT_COUNTS, "TRAIN component dose must be exactly 4096/768/256")
    _require(sum(row["nonpadding_tokens"] for row in clean) <= MAX_NONPADDING_TOKENS,
             "nonpadding training-token ceiling exceeded")
    return clean


def build_plan(rows, *, identities: dict, seed: int = SEED) -> dict:
    """Build a deterministic prospective plan from validated, source-bound records.

    Every record is used once, including replay variants sharing their parent.
    Input order is part of the binding; changing it changes the plan. The caller
    freezes the protocol/config before producing a production plan artifact.
    """
    _require(type(seed) is int and seed == SEED,
             "initial E1 phase permits seed 101 only; passing confirmation DEV screen is unbound")
    external = _identities(identities)
    ordered_rows = _validated_rows(rows)
    acquisition = [row for row in ordered_rows if row["component"] == "acquisition"]
    replay = [row for row in ordered_rows if row["component"] != "acquisition"]
    rng = random.Random(seed)
    rng.shuffle(acquisition)
    rng.shuffle(replay)
    config, shuffle_contract = schedule_config(), rng_contract()
    totals_by_component = {
        component: {
            "rows": sum(row["component"] == component for row in ordered_rows),
            "nonpadding_tokens": sum(row["nonpadding_tokens"] for row in ordered_rows
                                     if row["component"] == component),
            "completion_tokens": sum(row["completion_tokens"] for row in ordered_rows
                                     if row["component"] == component),
        } for component in COMPONENT_COUNTS
    }
    steps = []
    cumulative_nonpadding = cumulative_completion = 0
    cumulative_counts = {component: 0 for component in COMPONENT_COUNTS}
    for index in range(OPTIMIZER_STEPS):
        batch = []
        for group in range(4):
            start = index * ACQUISITION_PER_STEP + group * 4
            batch.extend(acquisition[start:start + 4])
            batch.append(replay[index * REPLAY_PER_STEP + group])
        nonpadding = sum(row["nonpadding_tokens"] for row in batch)
        completion = sum(row["completion_tokens"] for row in batch)
        counts = {component: sum(row["component"] == component for row in batch)
                  for component in COMPONENT_COUNTS}
        cumulative_nonpadding += nonpadding
        cumulative_completion += completion
        for component, count in counts.items():
            cumulative_counts[component] += count
        update = index + 1
        steps.append({
            "optimizer_step": update,
            "row_keys": [row["key"] for row in batch],
            "component_rows": counts,
            "nonpadding_tokens": nonpadding,
            "completion_tokens": completion,
            "cumulative_component_rows": dict(cumulative_counts),
            "cumulative_nonpadding_tokens": cumulative_nonpadding,
            "cumulative_completion_tokens": cumulative_completion,
            "learning_rate": learning_rate(update),
            "learning_rate_fraction_of_peak": {
                "numerator": update if update <= WARMUP_STEPS else OPTIMIZER_STEPS - update,
                "denominator": WARMUP_STEPS if update <= WARMUP_STEPS else OPTIMIZER_STEPS - WARMUP_STEPS,
            },
            "checkpoint_after": update in CHECKPOINT_STEPS,
            "dev_evaluation_after": update in CHECKPOINT_STEPS,
        })
    plan = {
        "schema_version": 1,
        "status": "PROPOSED_PLAN_ONLY_NO_NATIVE_EXECUTION",
        "identities": external,
        "schedule_config": config,
        "schedule_config_sha256": sha256_json(config),
        "rng_contract": shuffle_contract,
        "rng_contract_sha256": sha256_json(shuffle_contract),
        "shuffle_final_state_sha256": sha256_json(rng.getstate()),
        "ordered_rows_sha256": sha256_json(ordered_rows),
        "ordered_rows": ordered_rows,
        "totals": {
            "rows": len(ordered_rows),
            "optimizer_steps": len(steps),
            "nonpadding_tokens": cumulative_nonpadding,
            "completion_tokens": cumulative_completion,
            "maximum_sequence_nonpadding_tokens": max(row["nonpadding_tokens"] for row in ordered_rows),
            "maximum_sequence_completion_tokens": max(row["completion_tokens"] for row in ordered_rows),
            "components": totals_by_component,
        },
        "checkpoint_steps": list(CHECKPOINT_STEPS),
        "dev_evaluation_steps": list(CHECKPOINT_STEPS),
        "final_candidate_optimizer_step": OPTIMIZER_STEPS,
        "steps": steps,
        "resume_scope": "schedule/accounting only; native checkpoint and optimizer/RNG integrity unverified",
    }
    plan["plan_sha256"] = sha256_json(plan)
    return plan


def validate_plan(plan: dict, rows=None, *, identities: dict | None = None) -> dict:
    """Rebuild and compare all fields; pass fresh rows/identities to rebind sources.

    Omitting rows/identities checks self-consistency against the embedded binding,
    not whether files on disk still match. This module never reads those files.
    """
    _require(type(plan) is dict, "plan must be an object")
    _require("ordered_rows" in plan and "identities" in plan, "plan binding missing")
    expected = build_plan(plan["ordered_rows"] if rows is None else rows,
                          identities=plan["identities"] if identities is None else identities)
    _require(canonical_bytes(plan) == canonical_bytes(expected),
             "plan differs from exact bound rows, identities, schedule, or RNG contract")
    return plan


def _cursor(plan: dict, completed_optimizer_steps: int) -> dict:
    _require(type(completed_optimizer_steps) is int and 0 <= completed_optimizer_steps <= OPTIMIZER_STEPS,
             "resume requires a completed optimizer-boundary cursor in 0..256")
    previous = plan["steps"][completed_optimizer_steps - 1] if completed_optimizer_steps else None
    return {
        "schema_version": 1,
        "kind": "E1_OPTIMIZER_BOUNDARY_ACCOUNTING_ONLY",
        "identities": deepcopy(plan["identities"]),
        "plan_sha256": plan["plan_sha256"],
        "schedule_config_sha256": plan["schedule_config_sha256"],
        "rng_contract_sha256": plan["rng_contract_sha256"],
        "ordered_rows_sha256": plan["ordered_rows_sha256"],
        "completed_optimizer_steps": completed_optimizer_steps,
        "next_optimizer_step": completed_optimizer_steps + 1 if completed_optimizer_steps < OPTIMIZER_STEPS else None,
        "next_sequence_index": completed_optimizer_steps * SEQUENCES_PER_STEP,
        "accumulated_sequences": 0,
        "consumed_rows": completed_optimizer_steps * SEQUENCES_PER_STEP,
        "consumed_component_rows": deepcopy(previous["cumulative_component_rows"]) if previous else {
            component: 0 for component in COMPONENT_COUNTS
        },
        "consumed_nonpadding_tokens": previous["cumulative_nonpadding_tokens"] if previous else 0,
        "consumed_completion_tokens": previous["cumulative_completion_tokens"] if previous else 0,
        "native_checkpoint_integrity": "UNVERIFIED_BY_SCHEDULE_CONTRACT",
    }


def make_resume_cursor(plan: dict, completed_optimizer_steps: int) -> dict:
    """Describe a prospective boundary; this is not evidence that updates occurred."""
    validate_plan(plan)
    return _cursor(plan, completed_optimizer_steps)


def validate_resume_cursor(cursor: dict, plan: dict, *, rows=None,
                           identities: dict | None = None) -> int | None:
    """Reject partial/inconsistent prefixes and return the next 1-based update.

    Supply fresh bound rows/identities to reject changed sources. Native callers
    must separately validate checkpoint tensors, optimizer state and training RNG.
    """
    validate_plan(plan, rows, identities=identities)
    _require(type(cursor) is dict and "completed_optimizer_steps" in cursor, "invalid resume cursor")
    expected = _cursor(plan, cursor["completed_optimizer_steps"])
    _require(canonical_bytes(cursor) == canonical_bytes(expected),
             "resume cursor is not the exact optimizer-boundary schedule/token prefix")
    return expected["next_optimizer_step"]
