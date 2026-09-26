"""Short/general preservation probe for Phase 1.

Purpose: detect obvious regression outside the long-context target, so a target
gain is never reported as a general capability gain.

Source: ``Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0``
``by_class`` slices. Two TINY-length-mode classes are used, one with numeric
and one with short textual replies, so both are scoreable mechanically rather
than by judgement:

* ``counting_sets``     -- set counting with exclusions (numeric reply)
* ``conditional_rules`` -- rule application with a stated exception (textual reply)

Class choice is empirical, not aesthetic. A preservation metric pinned at 0%
cannot detect a regression, because there is no headroom below it. Probing the
frozen core on 24 items of each candidate class gave: counting_sets 42%,
conditional_rules 38-42%, arithmetic_chain 12%, and contradiction_detection,
constraint_selection, clarification_request and acknowledgment_update all 0%.
The two retained classes sit near 40%, so a regression has room to show.

These are short-context conversational items with no long document, so they sit
squarely outside the training distribution of the Phase 1 curriculum. That is
the point: they are a regression tripwire, not a second target.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Any, Sequence

from .provenance import REPO

CORPUS_DIR = (
    REPO
    / "Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0"
    / "data"
    / "sft"
    / "by_class"
)

PRESERVATION_CLASSES = ("counting_sets", "conditional_rules")
PRESERVATION_PER_CLASS = 100
PRESERVATION_SEED = 20260905

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def normalise_short(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower().rstrip(".")).strip()


def build_preservation_set(
    classes: Sequence[str] = PRESERVATION_CLASSES,
    per_class: int = PRESERVATION_PER_CLASS,
    seed: int = PRESERVATION_SEED,
) -> list[dict]:
    """Sample a fixed, seeded preservation set of prompt/reference pairs."""
    rng = random.Random(seed)
    items: list[dict] = []
    for cls in classes:
        path = CORPUS_DIR / f"{cls}.jsonl"
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows.sort(key=lambda r: r["id"])
        for row in rng.sample(rows, min(per_class, len(rows))):
            messages = row["messages"]
            if messages[-1]["role"] != "assistant":
                continue
            reference = messages[-1]["content"]
            items.append(
                {
                    "example_id": row["id"],
                    "task_family": cls,
                    "messages": messages[:-1],
                    "reference": reference,
                    "reference_numbers": _NUM_RE.findall(reference),
                }
            )
    items.sort(key=lambda r: r["example_id"])
    return items


def score_preservation(prediction: str, item: dict) -> dict:
    """Headline ``matched``, plus the stricter ``exact`` kept for reference.

    ``matched`` accommodates the two answer shapes these classes use, and the
    model's habit of showing work before stating the answer:

    * numeric reference -> compare the **last** number in each. The model
      replies ``"73 + 11 - 3 + 4 = 85"`` where the reference is ``"85."``; that
      is correct, and comparing first numbers would score it wrong against 73.
    * textual reference -> the normalised reference must appear in the
      normalised prediction.
    """
    predicted = normalise_short(prediction)
    reference = normalise_short(item["reference"])
    predicted_numbers = _NUM_RE.findall(prediction)
    reference_numbers = item["reference_numbers"]

    if reference_numbers:
        matched = bool(predicted_numbers) and predicted_numbers[-1] == reference_numbers[-1]
    else:
        matched = bool(reference) and reference in predicted

    return {
        "matched": bool(matched),
        "exact": predicted == reference,
        "has_reference_numbers": bool(reference_numbers),
        "response_words": len(re.findall(r"\b[\w'-]+\b", prediction)),
    }


def evaluate_preservation(
    tokenizer: Any,
    model: Any,
    items: Sequence[dict],
    *,
    device: str = "cuda",
    max_new_tokens: int = 24,
) -> list[dict]:
    import time

    import torch

    records: list[dict] = []
    for item in items:
        text = tokenizer.apply_chat_template(
            item["messages"],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(device)
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        elapsed = time.perf_counter() - t0
        raw = tokenizer.decode(out[0][enc.input_ids.shape[1]:], skip_special_tokens=True)
        records.append(
            {
                "example_id": item["example_id"],
                "task_family": item["task_family"],
                "prompt_tokens": int(enc.input_ids.shape[1]),
                "raw_prediction": raw,
                "reference": item["reference"],
                **score_preservation(raw, item),
                "inference_seconds": round(elapsed, 4),
            }
        )
    return records
