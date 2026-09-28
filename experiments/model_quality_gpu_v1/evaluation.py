"""Raw bounded greedy outputs and declared response contracts; no substring credit."""
from __future__ import annotations

from decimal import Decimal, InvalidOperation
import re
import torch
from .data import targets
from .losses import chunked_sum
from .native import hidden_states


def classify(text: str, specification: dict | None) -> dict:
    literal_unknown = text.strip() == "UNKNOWN"
    if specification is None:
        return {"correct": None, "scorer": "not_scored", "literal_unknown": literal_unknown}
    if set(specification) != {"scorer", "answers", "answerable"}:
        raise ValueError("evaluation requires an explicit complete scoring contract")
    if (type(specification["answerable"]) is not bool or not isinstance(specification["answers"], list)
            or not specification["answers"] or any(not isinstance(a, str) or not a.strip()
                                                   for a in specification["answers"])):
        raise ValueError("invalid scoring contract")
    if not specification["answerable"] and specification["answers"] != ["UNKNOWN"]:
        raise ValueError("literal-UNKNOWN tasks require the exact declared sentinel")
    scorer = specification["scorer"]
    if scorer == "exact":
        correct = text.strip() in [a.strip() for a in specification["answers"]]
    elif scorer == "number":
        if not specification["answerable"]:
            raise ValueError("number scorer cannot define missing-evidence tasks")
        pattern = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)"
        if not all(re.fullmatch(pattern, a.strip()) for a in specification["answers"]):
            raise ValueError("numeric gold must be finite bare decimal")
        correct = bool(re.fullmatch(pattern, text.strip())) and any(
            Decimal(text.strip()) == Decimal(a.strip()) for a in specification["answers"])
    else:
        raise ValueError("no implicit scorer for summaries, explanations or open-ended text")
    return {"correct": bool(correct), "scorer": scorer, "literal_unknown": literal_unknown,
            "answerable": specification["answerable"],
            "false_abstention": specification["answerable"] and literal_unknown}


def greedy(model, row, tokenizer, eos_ids: list[int], max_new=128) -> dict:
    device = next(model.parameters()).device
    current = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
    generated, past, terminated = [], None, False
    with torch.no_grad():
        for _ in range(max_new):
            output = model(input_ids=current, past_key_values=past, use_cache=True,
                           return_dict=True)
            past = output.past_key_values
            token = int(output.logits[0, -1].argmax())
            if token in eos_ids:
                terminated = True
                break
            generated.append(token)
            current = torch.tensor([[token]], dtype=torch.long, device=device)
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return {"id": row["id"], "family": row["family"], "prediction": text,
            "generated_ids": generated, "terminated": terminated,
            "limit_hit": not terminated, "prompt_tokens": len(row["prompt_ids"]),
            "scores": classify(text, row.get("evaluation"))}


def evaluate(model, rows: list[dict], chunk_size: int, *, tokenizer=None, eos_ids=None) -> dict:
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    total, tokens, records = 0.0, 0, []
    try:
        with torch.no_grad():
            for row in rows:
                ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
                labels = torch.tensor([targets(row, "response")], dtype=torch.long, device=device)
                loss, count = chunked_sum(hidden_states(model, ids), model.lm_head.weight,
                                          labels, chunk_size, recompute=False)
                total += float(loss); tokens += count
                if tokenizer is not None:
                    records.append(greedy(model, row, tokenizer, eos_ids))
        return {"response_ce": total / tokens, "response_targets": tokens,
                "generated": tokenizer is not None, "records": records,
                "role": "DEVELOPMENT_ONLY_NO_PROMOTION"}
    finally:
        model.train(was_training)
