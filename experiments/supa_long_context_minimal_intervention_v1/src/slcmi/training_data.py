"""Phase 1 training curriculum, rendered from existing repository data.

Source: ``Rineya_LCR_SFT_v1/data/lcr_sft_v1.jsonl`` -- 8000 rows over 2000
scenario packs, split train 7200 / dev 800, ten reasoning task types crossed
with four variants (base, paraphrase, counterfactual, unanswerable). It is
declared ``sft_train_disjoint_from_rineya_long_context_demo_v1`` and that was
verified here: 0 overlapping contexts, 0 overlapping questions and 0 overlapping
titles against the 256-row evaluation suite.

Two renderings are applied, both mechanical, both preregistered:

**Protocol alignment.** LCR ships a verbose system message ("Brevity is
optional") and long sentence answers (median 16 words). The evaluation suite
demands the shortest exact answer (median 2 words) and is scored by strict exact
match. Training on the raw form would teach verbosity and Phase 1 would measure
answer length rather than reasoning. So rows are re-rendered under the
*evaluation's own* system prompt and user template -- which is already
character-identical between the two datasets -- with a short target.

**Short target extraction.** For answerable rows the target is the single
``must_include`` span that does not already appear in the question. This is
exact, not heuristic: it resolves to exactly one span on all 6000 answerable
rows. For unanswerable rows the target is ``UNKNOWN``, matching the suite's
accepted answer.

Whole scenario packs are sampled together so the grouped-family structure
(same scenario seen as base / paraphrase / counterfactual / unanswerable)
survives into training, which is what forces comparison rather than lookup.
"""

from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Sequence

from .evaluation import SYSTEM_PROMPT, USER_TEMPLATE
from .provenance import REPO

LCR_SFT = (
    REPO.parent
    / "Chat-Bubble"
    / "Rineya_LCR_SFT_v1"
    / "Rineya_LCR_SFT_v1"
    / "data"
    / "lcr_sft_v1.jsonl"
)

UNKNOWN_SURFACE = "UNKNOWN"
CURRICULUM_SEED = 20260905


def short_target(row: dict) -> str:
    """The evaluation-protocol answer for one LCR row."""
    if not row["answerable"]:
        return UNKNOWN_SURFACE
    question = row["question"].lower()
    residual = [m for m in row["must_include"] if m.lower() not in question]
    if len(residual) != 1:
        raise ValueError(
            f"{row['id']}: expected exactly one residual must_include span, got {residual!r}"
        )
    return residual[0]


def render(row: dict) -> dict:
    return {
        "example_id": row["id"],
        "scenario_id": row["scenario_id"],
        "split": row["split"],
        "variant": row["variant"],
        "task_type": row["task_type"],
        "needle_profile": row["needle_profile"],
        "length_bucket": row["length_bucket"],
        "answerable": row["answerable"],
        "system": SYSTEM_PROMPT,
        "user": USER_TEMPLATE.format(context=row["context"], question=row["question"]),
        "target": short_target(row),
    }


def load_packs(path: Path = LCR_SFT) -> dict[str, list[dict]]:
    """Group every row by scenario id, preserving file order within a pack."""
    import json

    packs: dict[str, list[dict]] = defaultdict(list)
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            packs[row["scenario_id"]].append(row)
    return dict(packs)


def sample_curriculum(
    packs: dict[str, list[dict]],
    split: str,
    n_packs: int,
    seed: int = CURRICULUM_SEED,
) -> list[dict]:
    """Stratified sample of whole packs, balanced across task types.

    Returns rendered rows in a fixed, seeded order. Arms B and C consume this
    identical sequence, so data and data order are matched by construction.
    """
    eligible = {
        sid: rows for sid, rows in packs.items() if all(r["split"] == split for r in rows)
    }
    by_task: dict[str, list[str]] = defaultdict(list)
    for sid, rows in eligible.items():
        by_task[rows[0]["task_type"]].append(sid)

    rng = random.Random(seed)
    for sids in by_task.values():
        sids.sort()
        rng.shuffle(sids)

    tasks = sorted(by_task)
    chosen: list[str] = []
    cursor = 0
    while len(chosen) < n_packs:
        task = tasks[cursor % len(tasks)]
        pool = by_task[task]
        index = cursor // len(tasks)
        if index < len(pool):
            chosen.append(pool[index])
        elif all(cursor // len(tasks) >= len(by_task[t]) for t in tasks):
            break
        cursor += 1

    rendered = [render(row) for sid in chosen for row in eligible[sid]]
    rng.shuffle(rendered)
    return rendered


def tokenize_example(
    tokenizer: Any,
    example: dict,
    max_length: int,
) -> dict | None:
    """Prompt-masked causal-LM example.

    Loss is taken on the answer tokens and the turn terminator only; every
    prompt token is masked to -100. Examples longer than ``max_length`` are
    dropped rather than truncated -- truncating a long-context reasoning item
    would silently delete the evidence it is testing.
    """
    messages = [
        {"role": "system", "content": example["system"]},
        {"role": "user", "content": example["user"]},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids
    answer_ids = tokenizer(
        example["target"] + "<|im_end|>", add_special_tokens=False
    ).input_ids

    input_ids = prompt_ids + answer_ids
    if len(input_ids) > max_length:
        return None
    labels = [-100] * len(prompt_ids) + list(answer_ids)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "n_prompt_tokens": len(prompt_ids),
        "n_answer_tokens": len(answer_ids),
    }


def build_dataset(
    tokenizer: Any,
    examples: Sequence[dict],
    max_length: int,
    log: Callable[[str], None] = print,
) -> tuple[list[dict], dict]:
    kept: list[dict] = []
    dropped = 0
    for example in examples:
        encoded = tokenize_example(tokenizer, example, max_length)
        if encoded is None:
            dropped += 1
            continue
        kept.append({**example, **encoded})

    lengths = [len(k["input_ids"]) for k in kept]
    stats = {
        "n_input": len(examples),
        "n_kept": len(kept),
        "n_dropped_too_long": dropped,
        "max_length": max_length,
        "tokens_min": min(lengths) if lengths else 0,
        "tokens_median": sorted(lengths)[len(lengths) // 2] if lengths else 0,
        "tokens_mean": round(sum(lengths) / len(lengths), 1) if lengths else 0,
        "tokens_max": max(lengths) if lengths else 0,
        "answer_tokens_mean": (
            round(sum(k["n_answer_tokens"] for k in kept) / len(kept), 2) if kept else 0
        ),
        "variants": dict(
            sorted(
                (v, sum(1 for k in kept if k["variant"] == v))
                for v in {k["variant"] for k in kept}
            )
        ),
        "task_types": dict(
            sorted(
                (t, sum(1 for k in kept if k["task_type"] == t))
                for t in {k["task_type"] for k in kept}
            )
        ),
    }
    log(f"    dataset: kept {stats['n_kept']}, dropped {dropped}, "
        f"median {stats['tokens_median']} tokens")
    return kept, stats
