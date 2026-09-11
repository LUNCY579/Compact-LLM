"""Strict path comparisons and bounded numerical diagnostics, never quality evidence.

Argmax agreement and a tensor-global scale bound are diagnostics, not substitutes for
an elementwise comparison. FP64 experiments test a hypothesis; they cannot prove the
absence of an implementation defect or qualify an FP32 serving path.
"""
from __future__ import annotations
import math
from pathlib import Path
import torch
from torch import Tensor
from .gates import GateClosed, context, protocol, write_once
from .source import audit_source, sha256_file


def compare_logits(expected: Tensor, actual: Tensor, *, atol: float, rtol: float) -> dict:
    if expected.shape != actual.shape or expected.ndim < 2 or expected.shape[-1] < 2 or not expected.numel():
        raise ValueError('nonempty matching logits with vocabulary >= 2 required')
    if any(not math.isfinite(x) or x < 0 for x in (atol, rtol)):
        raise ValueError('finite nonnegative tolerances required')
    finite = bool(torch.isfinite(expected).all() and torch.isfinite(actual).all())
    result = {'finite': finite, 'elements': expected.numel(), 'positions': expected.numel() // expected.shape[-1],
              'atol': atol, 'rtol': rtol, 'strict_elementwise_pass': False,
              'argmax_exact': False, 'elementwise_violations': None}
    if not finite:
        # JSON must not contain NaN/Infinity, and a NaN must never look like zero violations.
        return result
    a, b = expected.detach().reshape(-1, expected.shape[-1]), actual.detach().reshape(-1, actual.shape[-1])
    delta = (a-b).abs()
    bound = atol + rtol*a.abs()
    violations = delta > bound
    count = int(violations.sum())
    margin = a.topk(2, dim=-1).values
    margin = margin[:, 0] - margin[:, 1]
    per_position_error = delta.max(-1).values
    tv_sum, kl_sum = 0.0, 0.0
    # Chunk the probability metrics; full-vocabulary FP64 temporaries otherwise dominate RAM.
    for i in range(0, a.shape[0], 4):
        la = a[i:i+4].double().log_softmax(-1)
        lb = b[i:i+4].double().log_softmax(-1)
        pa, pb = la.exp(), lb.exp()
        tv_sum += ((pa-pb).abs().sum(-1)*0.5).sum().item()
        kl_sum += (pa*(la-lb)).sum(-1).sum().item()
    scale = float(a.abs().max())
    result.update(strict_elementwise_pass=count == 0, elementwise_violations=count,
                  violating_positions=int(violations.any(-1).sum()),
                  max_abs_error=float(delta.max()), mean_abs_error=float(delta.double().mean()),
                  scale=scale, within_global_scaled_tolerance_diagnostic=bool(delta.max() <= atol + rtol*scale),
                  argmax_exact=bool(torch.equal(a.argmax(-1), b.argmax(-1))),
                  argmax_disagreements=int((a.argmax(-1) != b.argmax(-1)).sum()),
                  minimum_top1_margin=float(margin.min()),
                  margin_certified_positions=int((margin > 2*per_position_error).sum()),
                  mean_total_variation=tv_sum/a.shape[0], mean_KL_expected_to_actual=kl_sum/a.shape[0])
    return result


def _chunks(length: int, size: int) -> list[tuple[int, int]]:
    if type(size) is not int or size < 1:
        raise ValueError('chunk sizes must be positive integers')
    return [(i, min(i+size, length)) for i in range(0, length, size)]


def compare_paths(model, ids: Tensor, *, chunk_sizes=(1, 7, 16), atol=1e-4, rtol=1e-4,
                  attention_mask: Tensor | None = None, position_ids: Tensor | None = None,
                  segment_ids: Tensor | None = None) -> dict:
    """Compare tail-only, several chunk schedules, prefix and same-shape causal isolation.

    Uses one unmodified full-sequence forward as the canonical reference. All supplied
    metadata is sliced with the current chunk. Invalid padded positions are not scored.
    """
    if model.training or torch.is_grad_enabled():
        raise ValueError('path diagnostic requires eval and inference_mode')
    if ids.shape[1] < 2:
        raise ValueError('at least two tokens required for path diagnostics')
    sizes = list(dict.fromkeys(chunk_sizes))
    if not sizes or any(type(s) is not int or s < 1 for s in sizes):
        raise ValueError('at least one positive chunk size required')
    valid = torch.ones_like(ids, dtype=torch.bool) if attention_mask is None else attention_mask.bool()
    if not valid.any() or not valid[:, -1].all():
        raise ValueError('diagnostic needs valid positions and a valid final physical token')
    metadata = {k: v for k, v in {'attention_mask': attention_mask, 'position_ids': position_ids,
                                 'segment_ids': segment_ids}.items() if v is not None}
    full = model(ids, **metadata).logits
    def compare(a, b):
        return compare_logits(a, b, atol=atol, rtol=rtol)
    tail = model(ids, **metadata, use_cache=True, last_position_only=True).logits
    result = {'tail_only': compare(full[:, -1:], tail), 'chunks': {}}
    schedules = {f'size_{size}': _chunks(ids.shape[1], size) for size in sizes}
    cut = max(1, ids.shape[1]//2)
    schedules['halves'] = [(0, cut), (cut, ids.shape[1])]
    for label, schedule in schedules.items():
        cache, parts = None, []
        for start, end in schedule:
            step = model(ids[:, start:end],
                         **{k: v[:, start:end] for k, v in metadata.items()},
                         past=cache, use_cache=True)
            parts.append(step.logits)
            cache = step.cache
        result['chunks'][label] = compare(full[valid], torch.cat(parts, dim=1)[valid])
        del parts, cache, step
    prefix = model(ids[:, :cut], **{k: v[:, :cut] for k, v in metadata.items()}).logits
    prefix_valid = valid[:, :cut]
    if prefix_valid.any():
        result['prefix'] = compare(full[:, :cut][prefix_valid], prefix[prefix_valid])
    else:
        raise ValueError('diagnostic prefix contains no valid positions')
    # Same physical shape isolates causal leakage from chunk-size GEMM rounding effects.
    changed = ids.clone()
    changed[:, cut:] = (changed[:, cut:] + 1) % model.config.vocab_size
    perturbed = model(changed, **metadata).logits
    result['causal_suffix_isolation_exact'] = bool(torch.equal(full[:, :cut][prefix_valid],
                                                               perturbed[:, :cut][prefix_valid]))
    result['canonical_logits_finite'] = bool(torch.isfinite(full).all())
    comparisons = [result['tail_only'], result['prefix'], *result['chunks'].values()]
    result['all_strict_paths_pass'] = bool(result['canonical_logits_finite']
        and result['causal_suffix_isolation_exact'] and all(x['strict_elementwise_pass'] for x in comparisons))
    result['all_decode_choices_exact'] = all(x['argmax_exact'] for x in comparisons)
    return result


def continuation_probe(model, ids: Tensor, *, steps: int, atol: float, rtol: float) -> dict:
    """Cached-vs-full next-token comparison at identical autoregressive prefixes.

    Feed the canonical full-forward choice to both paths. The first disagreement is
    also the first free-running divergence; comparisons after it are teacher-forced.
    """
    if type(steps) is not int or steps < 0 or model.training or torch.is_grad_enabled():
        raise ValueError('eval/inference mode and nonnegative continuation steps required')
    seq, current, cache, observations = ids.clone(), ids, None, []
    for step in range(steps):
        expected = model(seq).logits[:, -1:]
        actual = model(current, past=cache, use_cache=True, last_position_only=True)
        metric = compare_logits(expected, actual.logits, atol=atol, rtol=rtol)
        metric['step'] = step
        observations.append(metric)
        current = expected[:, -1].argmax(-1, keepdim=True)
        seq = torch.cat((seq, current), -1)
        cache = actual.cache
    return {'steps': observations,
            'first_choice_divergence_step': next((m['step'] for m in observations if not m['argmax_exact']), None),
            'comparison_contexts': 'identical canonical prefixes; not a benchmark',
            'all_strict_paths_pass': all(m['strict_elementwise_pass'] for m in observations)}


def diagnose(source: Path, checkpoint: Path | None, pilot: Path, out: Path, device: str,
             *, arm: str, rows: int = 8, precision: str = 'float32', continuation_steps: int = 8,
             memory_precision: str = 'inherit') -> dict:
    from .run import _directory, _load, _load_tokenizer, _prompt, _restore, load_rows
    from .budget import presets
    if type(rows) is not int or not 1 <= rows <= 24 or precision not in ('float32', 'float64'):
        raise GateClosed('diagnostics are bounded to 1..24 pilot rows and FP32/FP64')
    if memory_precision not in ('inherit', 'float64'):
        raise GateClosed('memory accumulation diagnostic is inherit or float64 only')
    if type(continuation_steps) is not int or not 0 <= continuation_steps <= 24:
        raise GateClosed('continuation diagnostic is bounded to 0..24 steps')
    audit = audit_source(source)
    ctx = context(audit, pilot, device)
    data = load_rows(pilot, 'pilot')[:rows]
    if arm != 'B28' and checkpoint is None:
        raise GateClosed('converted-arm diagnostic requires a local saved checkpoint directory')
    if checkpoint is not None and out.resolve().is_relative_to(checkpoint.resolve()):
        raise GateClosed('diagnostic outputs must not modify checkpoint directories')
    _directory(out, source)
    bound_checkpoint = None
    if checkpoint is None:
        model, tokenizer, _ = _load(source, device)
    else:
        bound_checkpoint = {name: sha256_file(checkpoint/name) for name in ('config.json', 'model.safetensors')}
        model = _restore(checkpoint, device)
        tokenizer = _load_tokenizer(source)
    if model.config != presets()[protocol()['primary_arms'][arm]]:
        raise GateClosed('checkpoint shape does not match the declared diagnostic arm')
    model.to(dtype=getattr(torch, precision)).eval()
    # HOTFIX-002: raise accumulation for the CED shared-memory read only. A plain arm has
    # no such read, so this is inert there; that asymmetry is the point of the diagnostic.
    model.memory_accumulation_dtype = torch.float64 if memory_precision == 'float64' else None
    model_has_memory = model.memory is not None
    p = protocol()['G1']['parity_policy']
    observations = []
    with torch.inference_mode():
        for row in data:
            ids = _prompt(tokenizer, row, device)
            if ids.shape[1] + continuation_steps > protocol()['decoding']['max_context']:
                raise GateClosed('diagnostic continuation exceeds registered context; no truncation')
            observations.append({'id': row['id'], 'group_id': row['group_id'], 'tokens': ids.shape[1],
                'paths': compare_paths(model, ids, chunk_sizes=p['chunk_sizes'], atol=p['atol'], rtol=p['rtol']),
                'continuation': continuation_probe(model, ids, steps=continuation_steps, atol=p['atol'], rtol=p['rtol'])})
    if sha256_file(pilot) != ctx['data_sha256'] or audit_source(source)['audit_digest'] != audit['audit_digest']:
        raise GateClosed('source or diagnostic data changed during execution')
    if checkpoint is not None and bound_checkpoint != {
        name: sha256_file(checkpoint/name) for name in bound_checkpoint}:
        raise GateClosed('diagnosed checkpoint changed during execution')
    result = {'schema': 2, 'evidence_kind': 'numerical_diagnostic_only', 'arm': arm, 'context': ctx,
              'diagnostic_precision': precision, 'memory_accumulation_precision': memory_precision,
              'memory_accumulation_applies_to_this_arm': model_has_memory,
              'checkpoint_artifacts': bound_checkpoint, 'rows': observations,
              'status': 'OBSERVATIONS_RECORDED', 'licenses_training': False, 'licenses_efficacy_claim': False,
              'all_strict_paths_pass': all(r['paths']['all_strict_paths_pass'] and
                  r['continuation']['all_strict_paths_pass'] for r in observations),
              'cause': 'UNDETERMINED; compare paired FP32/FP64 and ordinary controls; no automatic causal attribution',
              'boundary': 'This report cannot replace, upgrade or authorize a G0/G1/G2 receipt.'}
    return write_once(out/'NUMERICS.json', result)
