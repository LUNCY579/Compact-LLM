"""FP32 logit and hidden trajectory measurements (no learned quality proxy)."""
from __future__ import annotations

import unicodedata
from typing import Any

import torch


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(-1), b.float().reshape(-1)
    return float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())


def logit_metrics(base: torch.Tensor, value: torch.Tensor, k: int = 10,
                  gold: int | None = None) -> dict[str, Any]:
    a, b = base.float(), value.float()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise FloatingPointError("nonfinite logits")
    la, lb = a.log_softmax(-1), b.log_softmax(-1)
    ka, kb = a.topk(min(k, a.numel())).indices, b.topk(min(k, b.numel())).indices
    va, vb = a.topk(2).values, b.topk(2).values
    return {
        "logit_cosine": cosine(a, b),
        "kl_base_to_arm": max(0.0, float((la.exp() * (la - lb)).sum().item())),
        "max_abs_logit_delta": float((a - b).abs().max().item()),
        "top1_agreement": int(a.argmax() == b.argmax()),
        "topk_overlap": len(set(ka.tolist()) & set(kb.tolist())) / len(ka),
        "margin_change": float(((vb[0] - vb[1]) - (va[0] - va[1])).item()),
        "gold_token": gold,
        "gold_rank": None if gold is None else int((b > b[gold]).sum().item()) + 1,
        "baseline_gold_rank": None if gold is None else int((a > a[gold]).sum().item()) + 1,
    }


def hidden_metrics(value: torch.Tensor, normal: torch.Tensor,
                   previous: torch.Tensor, previous_delta: torch.Tensor | None,
                   application: int) -> tuple[dict[str, Any], torch.Tensor]:
    h, ref, prev = value.float(), normal.float(), previous.float()
    if not torch.isfinite(h).all():
        raise FloatingPointError("nonfinite hidden state")
    delta = h - prev
    rms = float(h.square().mean().sqrt().item())
    ref_rms = float(ref.square().mean().sqrt().item())
    dn = float(delta.norm().item())
    return {
        "application": application, "rms": rms, "l2": float(h.norm().item()),
        "variance": float(h.var(unbiased=False).item()),
        "rms_ratio_to_normal": rms / max(ref_rms, 1e-12),
        "cosine_to_normal": cosine(h, ref),
        "delta_to_normal_l2": float((h - ref).norm().item()),
        "delta_l2": dn, "cosine_to_previous": cosine(h, prev),
        "rho": None if previous_delta is None else dn / (float(previous_delta.norm().item()) + 1e-12),
        "delta_cosine": None if previous_delta is None else cosine(delta, previous_delta),
    }, delta


def output_flags(text: str, ids: list[int]) -> dict[str, Any]:
    scripts: dict[str, int] = {}
    for char in text:
        if char.isalpha():
            script = unicodedata.name(char, "UNNAMED").split()[0]
            scripts[script] = scripts.get(script, 0) + 1
    nonlatin = sum(v for k, v in scripts.items() if k != "LATIN")
    loop = any(len(ids) >= max(6, 3 * p) and ids[-p:] == ids[-2*p:-p] == ids[-3*p:-2*p]
               for p in range(1, 5))
    replacement = text.count("\ufffd")
    controls = sum(unicodedata.category(c) == "Cc" and c not in "\t\r\n" for c in text)
    script_corruption = nonlatin >= 3 and nonlatin / max(1, len(text)) >= .25
    return {
        "empty": not text.strip(), "replacement_count": replacement,
        "replacement_rate": replacement / max(1, len(text)),
        "control_count": controls, "script_counts": scripts,
        "nonlatin_letter_rate": nonlatin / max(1, len(text)),
        "script_corruption": script_corruption, "repetition_loop": loop,
        "invalid": not text.strip() or replacement > 0 or controls > 0 or script_corruption or loop,
        "char_length": len(text), "output_tokens": len(ids),
    }


def canonical_answer(case: dict[str, Any]) -> str | None:
    """Only exact/numeric/JSON labels supply a gold continuation, not keywords."""
    import json
    s = case["scorer"]
    if s["type"] == "exact" and len(s["answers"]) == 1:
        return str(s["answers"][0])
    if s["type"] == "numeric":
        return str(s["value"])
    if s["type"] == "json_exact":
        return json.dumps(s["value"], separators=(",", ":"), ensure_ascii=False)
    return None
