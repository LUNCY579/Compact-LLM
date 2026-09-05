"""Greedy full-prefix generation; deadlines also cover prefill, EOS and NLL.

Deadlines are checked at every forward boundary. A native torch forward cannot
be interrupted safely in-process; overrun is a failed receipt, never a success.
"""
import time
from . import legacy
from armr.evaluation import encode, forward_logits
from armr.metrics import output_flags, logit_metrics
from .scoring import canonical_answer, score_response


def evaluate_case(model, tokenizer, case, config, wrapper, base=None):
    import psutil
    import torch
    start = time.perf_counter()
    process = psutil.Process()
    peak_rss = 0

    def boundary():
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        if time.perf_counter() - start > config['max_prompt_seconds']:
            raise TimeoutError('registered per-example deadline exceeded (including prefill/NLL)')
        if peak_rss > config['max_rss_gib'] * 2**30:
            raise MemoryError('registered RSS limit exceeded')

    wrapper.records.clear()
    wrapper.calls = 0
    wrapper.min_rms_ratio = wrapper.max_rms_ratio = 1.0
    wrapper.capture = True
    ids = encode(tokenizer, case)
    if ids.shape[1] > config.get('max_input_tokens', 1024):
        raise RuntimeError('input token cap exceeded; no truncation of question permitted')
    boundary()
    logits = forward_logits(model, ids)
    boundary()
    first = logits.clone()
    hidden = list(wrapper.records)
    wrapper.capture = False
    prefill_seconds = time.perf_counter() - start
    eos_ids = model.generation_config.eos_token_id
    eos = set(eos_ids if isinstance(eos_ids, list) else [eos_ids])
    tokens = []
    cap = min(config['max_new_tokens'], case.get('max_new_tokens', config['max_new_tokens']))
    if cap <= 0:
        raise ValueError('invalid token cap')
    positions = ids.shape[1]
    finish = 'token_cap'
    decode_start = time.perf_counter()
    text = ''
    for step in range(cap):
        boundary()
        token = int(logits.argmax())
        tokens.append(token)
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        flags = output_flags(text, tokens)
        if flags['repetition_loop'] or flags['script_corruption'] or flags['control_count'] or flags['replacement_count']:
            raise RuntimeError('malformed generation: repetition/script/Unicode guard')
        if token in eos:
            finish = 'eos'
            break
        if step + 1 < cap:
            ids = torch.cat((ids, torch.tensor([[token]], dtype=ids.dtype)), dim=1)
            positions += ids.shape[1]
            boundary()
            logits = forward_logits(model, ids)
            boundary()
    boundary()
    decode_seconds = time.perf_counter() - decode_start
    generation_calls = wrapper.calls
    answer = canonical_answer(case)
    gold = tokenizer.encode(answer, add_special_tokens=False) if answer is not None else []
    nll = None
    if gold:
        prompt = encode(tokenizer, case)
        full = torch.cat((prompt, torch.tensor([gold], dtype=prompt.dtype)), dim=1)
        boundary()
        output = model(input_ids=full, attention_mask=torch.ones_like(full), use_cache=False)
        boundary()
        if output.past_key_values is not None:
            raise RuntimeError('unexpected NLL cache')
        values = output.logits[0, prompt.shape[1]-1:-1].float()
        if not torch.isfinite(values).all():
            raise FloatingPointError('nonfinite answer logits')
        nll = float(torch.nn.functional.cross_entropy(values, torch.tensor(gold)))
    score = score_response(text, case['scorer'])
    c = model.config
    q, kv = c.num_attention_heads * c.head_dim, c.num_key_value_heads * c.head_dim
    per_layer = 2 * (2*c.hidden_size*q + 2*c.hidden_size*kv + 3*c.hidden_size*c.intermediate_size)
    extra = getattr(wrapper, 'extra_matmul_flops_per_token', wrapper.extra_layer_applications * per_layer)
    bounds = config.get('hidden_rms_ratio_bounds', [.25, 4.0])
    if wrapper.min_rms_ratio < bounds[0] or wrapper.max_rms_ratio > bounds[1]:
        raise RuntimeError('registered hidden RMS guard exceeded')
    boundary()
    return {**{k: case[k] for k in ('example_id','case_id','scenario_id','family','split','track','variant','template_id','template_unseen_in_engineering') if k in case},
        'response': text, 'token_ids': tokens, 'termination': finish, 'eos_emitted': finish == 'eos',
        'truncated': finish == 'token_cap', 'safety_stopped': False, **output_flags(text, tokens),
        'score': float(score['score']), 'correct': bool(score['score'] == 1),
        'score_ambiguous': score.get('ambiguous', False), 'gold_nll': nll,
        'prefill_seconds': prefill_seconds, 'decode_seconds': decode_seconds,
        'wall_seconds': time.perf_counter()-start, 'peak_rss_bytes': peak_rss,
        'processed_token_positions': positions, 'layer_calls_generation': generation_calls,
        'hidden': hidden, 'trajectory_min_rms_ratio': wrapper.min_rms_ratio,
        'trajectory_max_rms_ratio': wrapper.max_rms_ratio,
        'generation_dense_matmul_flops_estimate': int(positions*(c.num_hidden_layers*per_layer+extra)+len(tokens)*2*c.hidden_size*c.vocab_size),
        'flops_excludes': ['attention', 'normalization', 'instrumentation', 'teacher_forced_NLL'],
        **logit_metrics(first if base is None else base, first, config['top_k'], gold[0] if gold else None)}, first
