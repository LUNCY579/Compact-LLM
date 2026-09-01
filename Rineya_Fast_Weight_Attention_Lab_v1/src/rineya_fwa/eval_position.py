"""Position-conditioned retrieval eval. Unsealed lab fixtures only."""

from __future__ import annotations

import random
from typing import Any

import torch

from rineya_fwa.data import COLOURS, FILLER_SENTENCE, SYSTEM, _place

POSITIONS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


def build_position_suite(
    *,
    lengths: list[int],
    positions: tuple[float, ...] = POSITIONS,
    n_per_cell: int = 2,
    seed: int = 11,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    items: list[dict[str, Any]] = []
    i = 0
    for length in lengths:
        filler_n = max(8, length // 12)
        for pos in positions:
            for _ in range(n_per_cell):
                key = f"PX-{rng.randint(10000, 99999)}"
                colour = rng.choice(COLOURS)
                fact = f"The registry key {key} is assigned colour {colour}."
                ctx = _place(fact, filler_n, pos, rng)
                question = f"What colour is registry key {key}?"
                items.append(
                    {
                        "id": f"POS-{length}-{pos:.2f}-{i:04d}",
                        "prompt": (
                            f"{SYSTEM}\n\nCONTEXT\n{ctx}\n\nQUESTION\n{question}\n\nANSWER"
                        ),
                        "messages": [
                            {"role": "system", "content": SYSTEM},
                            {
                                "role": "user",
                                "content": f"CONTEXT\n{ctx}\n\nQUESTION\n{question}\n\nANSWER",
                            },
                        ],
                        "answer": colour,
                        "position": pos,
                        "target_length": length,
                    }
                )
                i += 1
    return items


def normalize(text: str) -> str:
    return text.strip().splitlines()[0].strip().lower()


def greedy_answer(model: Any, tokenizer: Any, messages: list[dict[str, str]], max_new: int = 16) -> str:
    device = next(model.parameters()).device
    input_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_tensors="pt"
    )
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(0)
    input_ids = input_ids.to(device)
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_new,
            do_sample=False,
            use_cache=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )
    new_tokens = out[0, input_ids.shape[1] :]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def score_suite(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    *,
    max_new: int = 16,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for item in items:
        raw = greedy_answer(model, tokenizer, item["messages"], max_new=max_new)
        pred = normalize(raw)
        gold = normalize(item["answer"])
        abstain = pred == "unknown"
        malformed = pred not in {gold, "unknown"} and pred not in COLOURS
        exact = pred == gold
        rows.append(
            {
                "id": item["id"],
                "position": item["position"],
                "target_length": item["target_length"],
                "gold": gold,
                "pred": pred,
                "raw": raw,
                "exact": exact,
                "abstain": abstain,
                "malformed": malformed,
            }
        )

    def _mean(xs: list[bool]) -> float:
        return sum(xs) / len(xs) if xs else float("nan")

    by_pos: dict[str, float] = {}
    for pos in POSITIONS:
        cell = [r["exact"] for r in rows if abs(r["position"] - pos) < 1e-9]
        by_pos[str(pos)] = _mean(cell)
    middle = [r["exact"] for r in rows if r["position"] in (0.25, 0.50, 0.75)]
    edge = [r["exact"] for r in rows if r["position"] in (0.05, 0.10, 0.90, 0.95)]
    by_len: dict[str, float] = {}
    lengths = sorted({r["target_length"] for r in rows})
    for length in lengths:
        cell = [r["exact"] for r in rows if r["target_length"] == length]
        by_len[str(length)] = _mean(cell)
    return {
        "n": len(rows),
        "exact": _mean([r["exact"] for r in rows]),
        "abstain_rate": _mean([r["abstain"] for r in rows]),
        "malformed_rate": _mean([r["malformed"] for r in rows]),
        "by_position": by_pos,
        "by_length": by_len,
        "middle_exact": _mean(middle),
        "edge_exact": _mean(edge),
        "middle_edge_gap": _mean(middle) - _mean(edge),
        "rows": rows,
    }
