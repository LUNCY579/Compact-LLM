"""Bounded deterministic full-prefix generation and paired measurements."""
from __future__ import annotations

import time
from typing import Any

import psutil
import torch

from rineya_cross_scale.evaluation.scoring import score_response

from .hooks import Reapplication, SafetyError
from .metrics import canonical_answer, logit_metrics, output_flags


def encode(tokenizer: Any, case: dict) -> torch.Tensor:
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": case["prompt"]}], tokenize=False,
        add_generation_prompt=True, enable_thinking=False)
    return tokenizer(rendered, return_tensors="pt")["input_ids"]


def forward_logits(model: Any, ids: torch.Tensor) -> torch.Tensor:
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False,
                logits_to_keep=1)
    if out.past_key_values is not None:
        raise SafetyError("unexpected output cache")
    logits = out.logits[0, -1].float().clone()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("nonfinite logits")
    return logits


def evaluate_case(model: Any, tokenizer: Any, case: dict, config: dict,
                  wrapper: Reapplication, base: torch.Tensor | None = None) -> tuple[dict, torch.Tensor]:
    start = time.perf_counter()
    ids = encode(tokenizer, case)
    wrapper.records.clear()
    wrapper.calls = 0
    wrapper.min_rms_ratio, wrapper.max_rms_ratio = 1.0, 1.0
    wrapper.capture = True
    logits = forward_logits(model, ids)
    first = logits.clone()
    hidden = list(wrapper.records)
    wrapper.capture = False
    prefill = time.perf_counter() - start
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    token_ids: list[int] = []
    cap = min(config["max_new_tokens"], case.get("max_new_tokens", config["max_new_tokens"]))
    finish, text = "token_cap", ""
    token_positions = ids.shape[1]
    decode_started = time.perf_counter()
    for step in range(cap):
        token = int(logits.argmax().item())
        token_ids.append(token)
        text = tokenizer.decode(token_ids, skip_special_tokens=True)
        flags = output_flags(text, token_ids)
        if token in eos:
            finish = "eos"
            break
        if flags["repetition_loop"]:
            finish = "repetition_loop"
            break
        if flags["script_corruption"] or flags["control_count"] or flags["replacement_count"] >= 3:
            finish = "script_or_unicode_corruption"
            break
        if time.perf_counter() - start > config["max_prompt_seconds"]:
            finish = "timeout"
            break
        mem = psutil.Process().memory_info().rss
        if mem > config["max_rss_gib"] * 1024**3:
            raise SafetyError("RSS budget exceeded")
        if step + 1 < cap:
            ids = torch.cat((ids, torch.tensor([[token]], dtype=ids.dtype)), dim=1)
            token_positions += ids.shape[1]
            logits = forward_logits(model, ids)
    decode_seconds = time.perf_counter() - decode_started
    generation_calls = wrapper.calls
    answer = canonical_answer(case)
    gold_ids = [] if answer is None else tokenizer.encode(answer, add_special_tokens=False)
    nll = None
    if gold_ids and finish in ("eos", "token_cap"):
        prompt_ids = encode(tokenizer, case)
        full = torch.cat((prompt_ids, torch.tensor([gold_ids], dtype=prompt_ids.dtype)), dim=1)
        result = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False)
        gold_logits = result.logits[0, prompt_ids.shape[1]-1:-1].float()
        if not torch.isfinite(gold_logits).all():
            raise FloatingPointError("nonfinite gold logits")
        nll = float(torch.nn.functional.cross_entropy(gold_logits, torch.tensor(gold_ids)).item())
    row = {
        "case_id": case["case_id"], "family": case["family"], "response": text,
        "token_ids": token_ids, **output_flags(text, token_ids), "termination": finish,
        "eos_emitted": finish == "eos", "truncated": finish == "token_cap",
        "safety_stopped": finish not in ("eos", "token_cap"),
        "score": score_response(text, case["scorer"])["score"],
        "format_score": score_response(text, case["format_scorer"])["score"] if case.get("format_scorer") else None,
        "gold_nll": nll, "gold_answer": answer,
        "prefill_seconds": prefill, "decode_seconds": decode_seconds,
        "decode_tokens_per_second": max(0, len(token_ids)-1) / max(decode_seconds, 1e-12),
        "wall_seconds": time.perf_counter() - start,
        "processed_token_positions": token_positions, "layer_calls_generation": generation_calls,
        "hidden": hidden,
        "trajectory_min_rms_ratio": wrapper.min_rms_ratio,
        "trajectory_max_rms_ratio": wrapper.max_rms_ratio,
        **logit_metrics(first if base is None else base, first, config["top_k"], gold_ids[0] if gold_ids else None),
    }
    # Dominant decoder linear matmuls + output projection; excludes attention,
    # normalization, instrumentation, and optional teacher-forced NLL forward.
    c = model.config
    hd, inter = c.hidden_size, c.intermediate_size
    q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    per_layer = 2 * (hd*q + 2*hd*kv + q*hd + 3*hd*inter)
    depth = c.num_hidden_layers + wrapper.extra_layer_applications
    row["generation_dense_matmul_flops_estimate"] = int(token_positions * depth * per_layer + len(token_ids) * 2 * hd * c.vocab_size)
    return row, first
