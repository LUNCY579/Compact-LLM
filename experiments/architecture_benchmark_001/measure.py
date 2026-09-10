"""Paired group statistics and actual reference timings, not deployment estimates."""
from __future__ import annotations
import math
import random
import statistics
import time
import torch
from .model import DenseReference


def paired_group_interval(control: list[dict], candidate: list[dict], *, seed: int = 17,
                          resamples: int = 2000, confidence: float = 0.95) -> dict:
    if resamples < 100 or not 0 < confidence < 1:
        raise ValueError('invalid bootstrap specification')
    def index(rows):
        out = {}
        for row in rows:
            if row['id'] in out or type(row['correct']) is not bool or not row['group_id']:
                raise ValueError('duplicate ID, invalid outcome or missing resampling group')
            out[row['id']] = row
        return out
    a, b = index(control), index(candidate)
    if set(a) != set(b) or not a:
        raise ValueError('paired outcomes must have identical nonempty ID coverage')
    groups = {}
    for key in sorted(a):
        if a[key]['group_id'] != b[key]['group_id']:
            raise ValueError('paired group identities changed')
        groups.setdefault(a[key]['group_id'], []).append(int(b[key]['correct'])-int(a[key]['correct']))
    if len(groups) < 2:
        raise ValueError('at least two independent groups are required')
    effects = [statistics.mean(v) for _, v in sorted(groups.items())]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(effects, k=len(effects))) for _ in range(resamples))
    tail = (1-confidence)/2
    return {'estimand': 'equal_group_weighted_accuracy_difference', 'effect': statistics.mean(effects),
            'lower': draws[math.floor(tail*(resamples-1))], 'upper': draws[math.ceil((1-tail)*(resamples-1))],
            'groups': len(effects), 'rows': len(a), 'resamples': resamples, 'confidence': confidence,
            'seed': seed, 'note': 'One seed contrast only; report all registered training seeds separately. No automatic promotion.'}


def profile_reference(model: DenseReference, ids: torch.Tensor, *, new_tokens: int = 4, repeats: int = 5) -> dict:
    if repeats < 2 or new_tokens < 1 or ids.shape[1]+new_tokens > model.config.max_position_embeddings:
        raise ValueError('invalid profile budget')
    device = ids.device
    def sync():
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
    was_training = model.training
    model.eval()
    prefill, decode, total = [], [], []
    kv_bytes = 0
    try:
        with torch.inference_mode():
            # Separate warmup; never include initialization in TTFT.
            model(ids, use_cache=True, last_position_only=True)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            for _ in range(repeats):
                sync(); t0 = time.perf_counter()
                out = model(ids, use_cache=True, last_position_only=True)
                token = out.logits[:, -1].argmax(-1, keepdim=True)
                sync(); t1 = time.perf_counter()
                latencies = []
                for _ in range(new_tokens):
                    sync(); start = time.perf_counter()
                    out = model(token, past=out.cache, use_cache=True, last_position_only=True)
                    token = out.logits[:, -1].argmax(-1, keepdim=True)
                    sync(); latencies.append(time.perf_counter()-start)
                prefill.append(t1-t0); decode.extend(latencies); total.append(time.perf_counter()-t0)
                pairs = list(out.cache.encoder) + ([] if out.cache.memory is None else [out.cache.memory])
                kv_bytes = sum(t.numel()*t.element_size() for pair in pairs for t in pair)
    finally:
        model.train(was_training)
    def summarize(values):
        vals = sorted(values)
        return {'p50_seconds': statistics.median(vals), 'p95_seconds': vals[math.ceil(0.95*len(vals))-1]}
    return {'evidence_kind': 'reference_runtime_diagnostic', 'device': str(device),
            'dtype': str(model.embed_tokens.weight.dtype), 'prompt_tokens': ids.shape[1],
            'decode_steps': new_tokens, 'batch': ids.shape[0], 'repeats': repeats,
            'prefill_plus_first_argmax': summarize(prefill), 'decode_step_plus_argmax': summarize(decode),
            'total': summarize(total), 'persistent_kv_tensor_bytes': kv_bytes,
            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None,
            'licenses_quality_or_deployment_claim': False,
            'boundary': 'Explicit repeated-head eager reference; validation overhead included; CPU RSS not measured.'}
