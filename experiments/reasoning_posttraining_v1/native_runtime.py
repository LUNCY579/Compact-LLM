"""Actual-checkpoint qualification/evaluation; imported only by an approved worker."""
from __future__ import annotations
from collections import Counter
import time

import torch
from torch.nn import functional as F

from experiments.model_quality_gpu_v1.losses import chunked_sum
from experiments.model_quality_gpu_v1.native import hidden_states, parameter_digest, select_tail
from .serialization import compile_row


def canonical_answer(text: str) -> str:
    # This is the frozen primitive-training target contract, not a prose judge.
    return " ".join(text.strip().split())


@torch.no_grad()
def evaluate(model, rows, tokenizer, eos_ids, label, reserve, guard, *, max_new=128):
    """Pure inference: fixed-probe CE plus greedy decode. Never needs gradients.

    The decorator is load-bearing, not decoration. engine.microfit's probe() already
    wraps this in no_grad, but a caller that forgot would build a full autograd graph
    across 28 layers of eager attention and exhaust VRAM on the first long row -- which
    is exactly how the first RRP-002 baseline attempt died, with the free-memory guard
    reporting 0.00 GiB. Nesting no_grad is harmless, so the guarantee lives here.
    """
    model.eval()
    device = next(model.parameters()).device
    records, ce_sum, ce_targets = [], 0.0, 0
    for row in rows:
        # Charge a worst-case generation allowance before either CE or generation.
        # Actual tokens/forwards are separately retained in the prediction record.
        reserve({"evaluation_and_qualification_model_calls": 1,
                 "generated_tokens_all_candidates": max_new})
        guard()
        ids = torch.tensor([row["ids"]], dtype=torch.long, device=device)
        labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
        loss, n = chunked_sum(hidden_states(model, ids), model.lm_head.weight, labels, 64, recompute=False)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite fixed-probe loss")
        ce_sum += float(loss); ce_targets += n
        current = torch.tensor([row["prompt_ids"]], dtype=torch.long, device=device)
        emitted, past, terminated, calls = [], None, False, 0
        started = time.monotonic()
        for _ in range(max_new):
            guard()
            output = model(input_ids=current, past_key_values=past, use_cache=True,
                           return_dict=True, logits_to_keep=1)
            calls += 1; past = output.past_key_values
            token = int(output.logits[0, -1].argmax())
            emitted.append(token)
            if token in eos_ids:
                terminated = True
                break
            current = torch.tensor([[token]], dtype=torch.long, device=device)
        public_ids = emitted[:-1] if terminated else emitted
        # Never hide malformed generated control tokens by decoding them away.
        prediction = tokenizer.decode(public_ids, skip_special_tokens=False)
        correct = terminated and canonical_answer(prediction) == canonical_answer(row["reference"])
        records.append({"id": row["id"], "parent_id": row["parent_id"], "family": row["family"],
            "prediction": prediction, "generated_ids_including_terminal": emitted,
            "canonical_training_target_match": bool(correct),
            "strict_target_text_match": terminated and prediction.strip() == row["reference"].strip(),
            "literal_unknown": prediction.strip() == "UNKNOWN", "reference": row["reference"],
            "reference_is_whole_unknown": row["reference"] == "UNKNOWN", "terminated": terminated,
            "truncated": not terminated, "generated_tokens_including_terminal": len(emitted),
            "generation_forward_calls": calls, "fixed_ce_forward_calls": 1,
            "prefill_tokens": len(row["prompt_ids"]), "fixed_ce_input_tokens": len(row["ids"]),
            "generation_wall_seconds": time.monotonic() - started, "reasoning_tokens": 0})
        del output, past, loss, ids, labels, current
    return {"scope": "FIXED_TRAIN_ONLY_MICROFIT_PROBE", "label": label, "rows": len(records),
            "canonical_correct": sum(r["canonical_training_target_match"] for r in records),
            "terminated": sum(r["terminated"] for r in records), "response_ce": ce_sum / ce_targets,
            "loss_bearing_tokens": ce_targets, "records": records,
            "generated_tokens_actual": sum(r["generated_tokens_including_terminal"] for r in records),
            "generation_allowance_reserved": len(records) * max_new,
            "semantic_generalization_or_preservation_claim": False}


def qualify(model, tokenizer, rows, reserve, guard):
    named = select_tail(model, 20)
    unique = sum(p.numel() for p in model.parameters())
    if unique != 596049920 or len(named) != 220 or sum(p.numel() for _, p in named) != 314618880:
        raise ValueError("actual native unique/trainable inventory differs")
    if model.lm_head.weight.data_ptr() != model.model.embed_tokens.weight.data_ptr():
        raise ValueError("native tied vocabulary storage differs")
    if any(p.dtype != torch.float32 for p in model.parameters()):
        raise ValueError("qualification requires actual FP32 computation")
    for row in rows:
        rebuilt = compile_row(row["prompt"], row["reference"], tokenizer, max_sequence=2048, completion_reserve=128)
        for key in ("ids", "prompt_ids", "labels", "answer_start", "answer_end"):
            if rebuilt[key] != row[key]:
                raise ValueError("native tokenizer does not reproduce the approved targets")
    before = parameter_digest(model)
    prompt = [{"role": "system", "content": "Answer with the requested number."},
              {"role": "user", "content": "Reply with 7."}]
    probe = compile_row(prompt, "7", tokenizer)
    tokens = len(probe["ids"])
    if tokens > 64:
        raise ValueError("short native parity probe exceeds its bound")
    reserve({"evaluation_and_qualification_model_calls": 5 + (tokens + 2) // 3,
             "processed_training_tokens_all_attempts": 5 * tokens + 2048})
    device = next(model.parameters()).device
    ids = torch.tensor([probe["ids"]], device=device)
    labels = torch.tensor([probe["labels"]], device=device)
    model.eval(); guard()
    with torch.no_grad():
        direct = model(input_ids=ids, use_cache=False, return_dict=True).logits
        h = hidden_states(model, ids)
        torch.testing.assert_close(F.linear(h, model.lm_head.weight), direct, atol=0, rtol=0)
        loss, count = chunked_sum(h, model.lm_head.weight, labels, 2, recompute=False)
        reference = F.cross_entropy(direct[:, :-1].reshape(-1, direct.shape[-1]), labels[:, 1:].reshape(-1),
                                    ignore_index=-100, reduction="sum")
        torch.testing.assert_close(loss, reference, atol=2e-4, rtol=2e-4)
        pieces, past = [], None
        for start in range(0, tokens, 3):
            guard()
            part = model(input_ids=ids[:, start:start + 3], past_key_values=past, use_cache=True, return_dict=True)
            pieces.append(part.logits); past = part.past_key_values
        cached = torch.cat(pieces, dim=1)
        torch.testing.assert_close(direct, cached, atol=2e-4, rtol=2e-4)
        cache_error = float((direct - cached).abs().max())
        changed = ids.clone(); changed[0, -3] = tokenizer.encode("8", add_special_tokens=False)[0]
        perturbed = model(input_ids=changed, use_cache=False, return_dict=True).logits
        torch.testing.assert_close(direct[:, :-3], perturbed[:, :-3], atol=0, rtol=0)
    del direct, h, loss, reference, pieces, cached, past, part, changed, perturbed
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    gradient_records = []
    for name, inp, target in [("native_chat", ids, labels), ("synthetic_2048_shape_on_real_source", None, None)]:
        if inp is None:
            # A separately named mechanical shape fixture, not truncated task evidence.
            ordinary = tokenizer.encode("x y", add_special_tokens=False)
            if not ordinary or any(t in (tokenizer.eos_token_id,) for t in ordinary):
                raise ValueError("shape fixture token identity invalid")
            sequence = (ordinary * (2048 // len(ordinary) + 1))[:2048]
            sequence[-1] = tokenizer.eos_token_id
            inp = torch.tensor([sequence], device=device)
            target = torch.full_like(inp, -100); target[0, -2:] = inp[0, -2:]
        guard(); model.train(); model.zero_grad(set_to_none=True)
        loss, n = chunked_sum(hidden_states(model, inp), model.lm_head.weight, target, 64)
        (loss / n).backward()
        finite = sum(p.grad is not None and bool(torch.isfinite(p.grad).all()) for _, p in named)
        active = sum(p.grad is not None and bool(p.grad.abs().sum() > 0) for _, p in named)
        if finite != 220 or (name == "native_chat" and active != 220):
            raise ValueError("native trainable gradient participation failed")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise ValueError("frozen parameter received a gradient")
        gradient_records.append({"probe": name, "tokens": inp.shape[1], "finite_tensors": finite,
                                 "nonzero_gradient_tensors": active})
        model.zero_grad(set_to_none=True)
        del loss
    model.eval(); guard()
    after = parameter_digest(model)
    if before != after:
        raise ValueError("zero-update qualification changed source parameters")
    return {"phase": "RRP_NATIVE_ZERO_UPDATE_QUALIFICATION", "status": "PASS", "optimizer_steps": 0,
            "unique_parameters": unique, "trainable_parameters": 314618880, "trainable_tensors": 220,
            "source_state_before": before, "source_state_after": after, "cached_max_abs": cache_error,
            "native_template_rows_reproduced": len(rows), "gradient_probes": gradient_records,
            "qualified_training_shape_tokens": 2048, "padding_or_packing": "UNSUPPORTED",
            "checks": ["native_hidden_logits", "cached_logits", "causality", "response_eos_masks",
                       "loss_value", "finite_gradient_participation", "frozen_weights", "tied_storage", "shape_2048"],
            "native_gradient_equivalence": "NOT_CLAIMED; full value/gradient parity covered separately on CPU fixtures",
            "quality_permission": False, "release_permission": False}
