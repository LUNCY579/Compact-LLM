"""S0 decoding, teacher forcing, KV-cache parity and resumable reads (COMP-RS-001 Amendment 1, sections 8, 12, 16).

Pure functions over a loaded causal LM and token-ID lists. Device-agnostic: every tensor is created on the model's
device and no CUDA call is made unless that device is CUDA. Nothing here loads a model, takes the GPU lease,
builds an optimizer or authorises a stage.

Decoding paths (transformers 4.57.1, asserted before every generate call):
- greedy_decode: batch size 1, do_sample=False, use_cache=True, output_logits=True. raw_step_logprobs is
  log_softmax of the UNPROCESSED float32 step logits, gathered at the generated token.
- sample_group: one unpadded batch of G identical prompts, do_sample=True, temperature 1.0, top-p 1.0, top-k 0,
  repetition penalty 1.0 (the Q2 sampler). Raw step logprobs are streamed by the only logits processor in the
  list, so memory is O(G*V) rather than O(steps*G*V).
Both paths pass a complete GenerationConfig with use_model_defaults=False, then check the processor list and
stopping criteria that generate actually built (recorded as 'observed'), and replay every token: greedy tokens
must equal the raw-logit argmax, sampled tokens must equal draws from softmax(raw logits) by an independently
seeded torch.Generator. SAMPLER_SEMANTICS and SEED_PATH record why each check is needed.

teacher_forced_logprobs runs the decoder once over the (optionally left-padded) batch and applies the LM head
only at the hidden states that predict completion tokens (positions P-1 .. P+C-2). cached_logprobs is the
incremental KV-cache path. parity compares any two of them. ResumableJsonl is the per-row-hashed append-only
read journal of section 12.
"""
from __future__ import annotations

import base64
import contextlib
import datetime
import hashlib
import json
import os
from pathlib import Path

import torch

from .preflight import digest, require

TRANSFORMERS_VERSION = '4.57.1'
EOS_IDS = (151645, 151643)   # tokenizer receipt: generation_eos_ids (<|im_end|>, <|endoftext|>)
PAD_ID = 151643              # tokenizer receipt: pad (equal to an EOS ID, so masks are positional, never ID-based)
MAX_NEW_TOKENS = 128
SEED_LIMIT = 2 ** 63
SAMPLER = {'do_sample': True, 'temperature': 1.0, 'top_k': 0, 'top_p': 1.0, 'repetition_penalty': 1.0}
RECORDER = '_RawStepRecorder'
STOPPING = ['MaxLengthCriteria', 'EosTokenCriteria']

SAMPLER_SEMANTICS = (
    'transformers 4.57.1 generation/utils.py. _get_logits_processor adds RepetitionPenaltyLogitsProcessor iff '
    'repetition_penalty is not None and != 1.0 (L1141), and, only when do_sample, TemperatureLogitsWarper iff '
    'temperature is not None and != 1.0 (L1266), TopKLogitsWarper iff top_k is not None and != 0 (L1268), '
    'TopPLogitsWarper iff top_p is not None and < 1.0 (L1272); min_p None, typical_p 1.0 and epsilon/eta cutoffs '
    '0.0 add nothing. top_k=0 therefore disables top-k; omitting it leaves GenerationConfig\'s global default '
    'top_k=50 (configuration_utils.py L357), a live top-50 truncation. When a GenerationConfig object is passed '
    'with use_model_defaults=None and the model generation config is from >= 4.50, every field equal to the '
    'global default (temperature 1.0, top_p 1.0) is replaced by the model generation config value (L1761-1783); '
    'the Supa generation_config.json holds temperature 0.6, top_k 20, top_p 0.95, so this module always passes '
    'use_model_defaults=False with a complete config. _sample stores output_logits as the float32 copy of '
    'outputs.logits[:, -1] BEFORE the processors (L2800, L2810), then samples torch.multinomial(softmax(processed '
    'scores), num_samples=1) with no generator argument (L2828-2829); greedy is torch.argmax(processed scores). '
    'With the processor list reduced to the pass-through recorder, processed scores are the raw logits, so the '
    'sampling distribution is the raw policy softmax; this is checked per token by replay, not assumed.')

SEED_PATH = (
    'transformers 4.57.1 _sample passes no generator to torch.multinomial, so the draw consumes the default '
    'generator of the device holding the logits. Per call, inside torch.random.fork_rng (devices=[] on CPU, '
    '[index] on CUDA, so the caller\'s RNG state is restored afterwards), that default generator is seeded with '
    'the integer seed: torch.default_generator.manual_seed(seed) on CPU, torch.cuda.manual_seed(seed) under '
    'torch.cuda.device(index) on CUDA. No other RNG consumer runs during generate (eval mode is required). A '
    'separate torch.Generator(device) seeded with the same integer replays softmax(raw logits) -> '
    'torch.multinomial(., 1, generator=replay) at every step, and every sampled token of every sequence must '
    'equal its replay draw.')


def generation_contract(max_new_tokens=MAX_NEW_TOKENS, eos_ids=EOS_IDS, pad_id=PAD_ID):
    """The decoding settings and recorded semantics, for the S0 runtime freeze / decoding_sha256 preimage."""
    eos, pad = _eos_pad(eos_ids, pad_id)
    contract = {
        'transformers': TRANSFORMERS_VERSION, 'max_new_tokens': _cap(max_new_tokens),
        'eos_token_ids': list(eos), 'pad_token_id': pad,
        'greedy': {'do_sample': False, 'num_beams': 1, 'batch_size': 1, 'use_cache': True,
                   'raw_logits_source': 'generate(output_logits=True) float32 step logits, cross-checked '
                                        'bitwise against the pass-through recorder'},
        'sampler': dict(SAMPLER, num_beams=1, use_cache=True, batch='one unpadded batch of G identical prompts',
                        raw_logits_source='pass-through logits processor (sole processor, streamed)'),
        'use_model_defaults': False, 'logits_processors': [RECORDER], 'stopping_criteria': list(STOPPING),
        'sampler_semantics': SAMPLER_SEMANTICS, 'seed_path': SEED_PATH,
        'seed_range': [0, SEED_LIMIT - 1]}
    return dict(contract, contract_sha256=digest(contract))


# ---------------------------------------------------------------- validation helpers

def _cap(max_new_tokens):
    require(type(max_new_tokens) is int and max_new_tokens > 0, 'max_new_tokens must be a positive int')
    return max_new_tokens


def _eos_pad(eos_ids, pad_id):
    require(type(eos_ids) in (list, tuple) and len(eos_ids) > 0 and len(set(eos_ids)) == len(eos_ids)
            and all(type(x) is int and x >= 0 for x in eos_ids), 'EOS inventory')
    require(type(pad_id) is int and pad_id >= 0, 'pad ID')
    return tuple(eos_ids), pad_id


def _ids(values, vocab, what):
    if isinstance(values, torch.Tensor):
        require(values.ndim == 1 and values.dtype == torch.long, what + ': 1-D long tensor required')
        values = values.tolist()
    require(type(values) in (list, tuple) and len(values) > 0, what + ': non-empty ID list required')
    require(all(type(t) is int and 0 <= t < vocab for t in values), what + ': IDs must be ints in vocabulary')
    return list(values)


def _device(model):
    devices = {p.device for p in model.parameters()}
    require(len(devices) == 1, 'model parameters span devices: ' + str(sorted(map(str, devices))))
    device = devices.pop()
    require(device.type in ('cpu', 'cuda'), 'unsupported device: ' + str(device))
    return device


def _vocab(model):
    vocab = model.config.vocab_size
    require(type(vocab) is int and vocab > 0, 'model vocabulary size')
    require(model.get_output_embeddings().weight.shape[0] == vocab, 'LM head rows differ from vocab_size')
    return vocab


def _require_generation_runtime(model):
    import transformers
    require(transformers.__version__ == TRANSFORMERS_VERSION,
            'recorded generation semantics are for transformers ' + TRANSFORMERS_VERSION + ', found '
            + transformers.__version__)
    require(not model.training, 'generation requires eval mode (no dropout, no gradient-checkpoint cache override)')


# ---------------------------------------------------------------- generation internals

_RECORDER_CLASS = None


def _recorder_class():
    """A pass-through LogitsProcessor that streams raw selected logprobs and replays each step's token choice."""
    global _RECORDER_CLASS
    if _RECORDER_CLASS is None:
        from transformers import LogitsProcessor

        class _RawStepRecorder(LogitsProcessor):
            def __init__(self, mode, prompt_length, generator=None):
                require(mode in ('greedy', 'sample'), 'recorder mode')
                require(mode == 'greedy' or generator is not None, 'sampling replay needs a generator')
                self.mode, self.prompt_length, self.generator = mode, prompt_length, generator
                self.selected, self.predicted, self.pending = [], [], None

            def __call__(self, input_ids, scores):
                require(input_ids.shape[1] == self.prompt_length + len(self.predicted), 'recorder step misaligned')
                require(scores.dtype == torch.float32 and scores.ndim == 2, 'raw step logits must be float32 2-D')
                if self.pending is not None:
                    self.selected.append(self.pending.gather(-1, input_ids[:, -1:]).squeeze(-1))
                self.pending = scores.log_softmax(-1)
                if self.mode == 'greedy':
                    self.predicted.append(scores.argmax(-1))
                else:
                    self.predicted.append(torch.multinomial(scores.softmax(-1), 1,
                                                            generator=self.generator).squeeze(1))
                return scores

            def finish(self, sequences):
                require(self.pending is not None
                        and sequences.shape[1] == self.prompt_length + len(self.predicted),
                        'recorder did not see every decode step')
                self.selected.append(self.pending.gather(-1, sequences[:, -1:]).squeeze(-1))
                self.pending = None
                return torch.stack(self.selected), torch.stack(self.predicted)

        _RECORDER_CLASS = _RawStepRecorder
    return _RECORDER_CLASS


_CONFIG_FIELDS = ('do_sample', 'num_beams', 'num_return_sequences', 'temperature', 'top_k', 'top_p', 'min_p',
                  'typical_p', 'epsilon_cutoff', 'eta_cutoff', 'repetition_penalty', 'no_repeat_ngram_size',
                  'bad_words_ids', 'suppress_tokens', 'begin_suppress_tokens', 'renormalize_logits',
                  'max_new_tokens', 'max_length', 'min_length', 'min_new_tokens', 'eos_token_id', 'pad_token_id',
                  'bos_token_id', 'use_cache', 'output_logits', 'output_scores', 'return_dict_in_generate',
                  'cache_implementation', 'prefill_chunk_size')


def _config_record(config):
    def plain(value):
        if isinstance(value, (list, tuple)):
            return [plain(v) for v in value]
        require(value is None or type(value) in (bool, int, float, str), 'unrecordable generation field')
        return value
    return {name: plain(getattr(config, name, None)) for name in _CONFIG_FIELDS}


@contextlib.contextmanager
def _observe_generate(model):
    """Record the logits processors, stopping criteria and prepared config that generate() actually builds."""
    names = ('_get_logits_processor', '_get_stopping_criteria')
    require(not any(name in vars(model) for name in names), 'generation hooks already patched on this model')
    seen = {'processors': [], 'config': [], 'stopping': []}
    real_processors, real_stopping = model._get_logits_processor, model._get_stopping_criteria

    def processors(*args, **kwargs):
        result = real_processors(*args, **kwargs)
        seen['processors'].append([type(p).__name__ for p in result])
        seen['config'].append(_config_record(kwargs['generation_config'] if 'generation_config' in kwargs
                                             else args[0]))
        return result

    def stopping(*args, **kwargs):
        result = real_stopping(*args, **kwargs)
        seen['stopping'].append([type(c).__name__ for c in result])
        return result

    model._get_logits_processor, model._get_stopping_criteria = processors, stopping
    try:
        yield seen
    finally:
        for name in names:
            if name in vars(model):
                delattr(model, name)


def _checked_observation(seen, expected):
    require(len(seen['processors']) == len(seen['config']) == len(seen['stopping']) == 1,
            'expected exactly one observed generate() preparation')
    processors, config, stopping = seen['processors'][0], seen['config'][0], seen['stopping'][0]
    require(processors == [RECORDER], 'generate built extra logits processors: ' + repr(processors))
    require(stopping == STOPPING, 'unexpected stopping criteria: ' + repr(stopping))
    for key, value in expected.items():
        require(config[key] == value, 'effective generation field ' + key + ' = ' + repr(config[key])
                + ', expected ' + repr(value))
    return {'logits_processors': processors, 'stopping_criteria': stopping, 'effective_config': config}


@contextlib.contextmanager
def _seeded_default_generator(device, seed):
    """Seed the default generator that transformers' torch.multinomial consumes; restore the caller's state."""
    if device.type == 'cuda':
        index = device.index
        require(type(index) is int, 'CUDA device index')
        with torch.random.fork_rng(devices=[index], device_type='cuda'):
            with torch.cuda.device(index):
                torch.cuda.manual_seed(seed)
            yield
    else:
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            yield


def _truncate(tokens, eos, pad, cap):
    """Cut a generated row at its first EOS (inclusive); a row without EOS must have hit the cap."""
    first = next((j for j, t in enumerate(tokens) if t in eos), None)
    if first is None:
        require(len(tokens) == cap, 'no EOS and fewer than max_new_tokens tokens')
        return tokens, 'length'
    require(all(t == pad for t in tokens[first + 1:]), 'tokens after the first EOS are not forced padding')
    return tokens[:first + 1], 'eos'


def _generation_config(**fields):
    from transformers import GenerationConfig
    return GenerationConfig(**fields)


# ---------------------------------------------------------------- public decoding API

def greedy_decode(model, prompt_ids, max_new_tokens=MAX_NEW_TOKENS, eos_ids=EOS_IDS, pad_id=PAD_ID):
    """Greedy decode one prompt (batch 1). The completion includes the first EOS; stop_reason is 'eos'|'length'."""
    from transformers import LogitsProcessorList

    _require_generation_runtime(model)
    eos, pad = _eos_pad(eos_ids, pad_id)
    cap = _cap(max_new_tokens)
    device, vocab = _device(model), _vocab(model)
    prompt = _ids(prompt_ids, vocab, 'prompt')
    config = _generation_config(do_sample=False, num_beams=1, num_return_sequences=1, max_new_tokens=cap,
                                eos_token_id=list(eos), pad_token_id=pad, use_cache=True,
                                return_dict_in_generate=True, output_logits=True, output_scores=False)
    recorder = _recorder_class()('greedy', len(prompt))
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with _observe_generate(model) as seen, torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config,
                             logits_processor=LogitsProcessorList([recorder]), use_model_defaults=False)
        observed = _checked_observation(seen, {'do_sample': False, 'num_beams': 1, 'max_new_tokens': cap,
                                               'eos_token_id': list(eos), 'pad_token_id': pad,
                                               'use_cache': True, 'output_logits': True})
        generated = out.sequences[0, len(prompt):]
        steps = generated.shape[0]
        require(steps == len(out.logits) > 0, 'one raw logit row per generated token')
        raw = torch.cat(out.logits, dim=0)
        require(raw.shape == (steps, vocab) and raw.dtype == torch.float32, 'raw logit shape/dtype')
        selected = torch.cat([step.log_softmax(-1) for step in out.logits]).gather(
            -1, generated[:, None]).squeeze(-1)
        require(torch.equal(raw.argmax(-1), generated), 'greedy token differs from the raw-logit argmax')
        streamed, predicted = recorder.finish(out.sequences)
        require(torch.equal(streamed[:, 0], selected) and torch.equal(predicted[:, 0], generated),
                'streamed raw logprobs differ from output_logits')
        tokens = generated.tolist()
        logprobs = selected.tolist()
    completion, reason = _truncate(tokens, eos, pad, cap)
    require(len(completion) == steps, 'batch-1 greedy generated past its first EOS')
    return {'completion_ids': completion, 'stop_reason': reason, 'generated_tokens': len(completion),
            'raw_step_logprobs': logprobs,
            'decoder': dict(observed, mode='greedy', transformers=TRANSFORMERS_VERSION,
                            attn_implementation=model.config._attn_implementation,
                            raw_logits_source='output_logits', argmax_replay_verified=True)}


def sample_group(model, prompt_ids, G, seed, max_new_tokens=MAX_NEW_TOKENS, eos_ids=EOS_IDS, pad_id=PAD_ID):
    """G sampled completions of one prompt in one unpadded batch, under the Q2 sampler (see SEED_PATH)."""
    from transformers import LogitsProcessorList

    _require_generation_runtime(model)
    eos, pad = _eos_pad(eos_ids, pad_id)
    cap = _cap(max_new_tokens)
    require(type(G) is int and G > 0, 'group size G must be a positive int')
    require(type(seed) is int and 0 <= seed < SEED_LIMIT, 'seed must be an int in [0, 2^63)')
    device, vocab = _device(model), _vocab(model)
    prompt = _ids(prompt_ids, vocab, 'prompt')
    config = _generation_config(**SAMPLER, num_beams=1, num_return_sequences=1, max_new_tokens=cap,
                                eos_token_id=list(eos), pad_token_id=pad, use_cache=True,
                                return_dict_in_generate=True, output_logits=False, output_scores=False)
    replay = torch.Generator(device=device)
    replay.manual_seed(seed)
    recorder = _recorder_class()('sample', len(prompt), replay)
    ids = torch.tensor([prompt] * G, dtype=torch.long, device=device)
    with _observe_generate(model) as seen, _seeded_default_generator(device, seed), torch.inference_mode():
        out = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), generation_config=config,
                             logits_processor=LogitsProcessorList([recorder]), use_model_defaults=False)
        observed = _checked_observation(seen, dict(SAMPLER, num_beams=1, min_p=None, typical_p=1.0,
                                                   max_new_tokens=cap, eos_token_id=list(eos),
                                                   pad_token_id=pad, use_cache=True, output_logits=False))
        streamed, predicted = recorder.finish(out.sequences)
        generated = out.sequences[:, len(prompt):]
        steps = generated.shape[1]
        require(generated.shape[0] == G and streamed.shape == predicted.shape == (steps, G),
                'group shape')
        rows, logprobs, draws = generated.tolist(), streamed.T.tolist(), predicted.T.tolist()
    completions = []
    for tokens, row_logprobs, row_draws in zip(rows, logprobs, draws):
        completion, reason = _truncate(tokens, eos, pad, cap)
        n = len(completion)
        require(row_draws[:n] == completion, 'sampled token differs from its seeded replay draw')
        completions.append({'completion_ids': completion, 'stop_reason': reason, 'generated_tokens': n,
                            'raw_step_logprobs': row_logprobs[:n]})
    return {'seed': seed, 'group_size': G, 'decode_steps': steps, 'completions': completions,
            'sampler': dict(observed, mode='sample', transformers=TRANSFORMERS_VERSION,
                            attn_implementation=model.config._attn_implementation,
                            raw_logits_source='pass-through recorder', seed_path=SEED_PATH,
                            replay_verified=True)}


# ---------------------------------------------------------------- teacher forcing and cache

def left_pad_batch(prompt_ids_list, completion_ids_list, pad_id, device='cpu', vocab=None):
    """Left-pad prompt+completion rows. attention_mask is bool; position_ids count real tokens from 0 (pads: 1)."""
    require(type(prompt_ids_list) in (list, tuple) and type(completion_ids_list) in (list, tuple)
            and len(prompt_ids_list) == len(completion_ids_list) > 0, 'batch lists')
    require(type(pad_id) is int and pad_id >= 0, 'pad ID')
    limit = vocab if vocab is not None else 2 ** 62
    prompts = [_ids(p, limit, 'prompt') for p in prompt_ids_list]
    completions = [_ids(c, limit, 'completion') for c in completion_ids_list]
    lengths = [len(p) + len(c) for p, c in zip(prompts, completions)]
    width = max(lengths)
    offsets = [width - n for n in lengths]
    ids = torch.tensor([[pad_id] * o + p + c for o, p, c in zip(offsets, prompts, completions)],
                       dtype=torch.long, device=device)
    positions = torch.arange(width, device=device)
    attention = positions[None, :] >= torch.tensor(offsets, device=device)[:, None]
    position_ids = attention.long().cumsum(-1) - 1
    position_ids.masked_fill_(~attention, 1)
    return {'input_ids': ids, 'attention_mask': attention, 'position_ids': position_ids,
            'prompt_lengths': [len(p) for p in prompts], 'completion_lengths': [len(c) for c in completions],
            'offsets': offsets}


def _head_logprobs(head, hidden, targets):
    return head(hidden).float().log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1)


def teacher_forced_logprobs(model, prompt_ids_list, completion_ids_list, pad_id, padding_side='left',
                            return_argmax=False, checkpoint_head=False):
    """Per-sequence selected-token logprobs from one decoder pass; the LM head runs only at completion positions.

    padding_side='left' left-pads the batch with an attention mask and real-token position IDs. 'none' runs the
    batch unpadded with no attention mask, and refuses rows of different total length (a single row always
    qualifies: this is the individual reference path). Gradients flow when grad mode is enabled. With
    return_argmax, also returns the per-position argmax token of the same logits.

    Under grad, autograd keeps each row's (C, V) float32 log-softmax for backward (8 x 128 x 151936 x 4 B =
    0.58 GiB for a full G = 8 group). checkpoint_head=True keeps only the row's hidden states and recomputes that
    row's head in backward (torch.utils.checkpoint, non-reentrant), one row at a time; values are unchanged.
    It is ignored when grad is disabled and cannot be combined with return_argmax.
    """
    require(padding_side in ('left', 'none'), "padding_side must be 'left' or 'none'")
    require(not (checkpoint_head and return_argmax), 'checkpoint_head is a training path; no argmax there')
    device, vocab = _device(model), _vocab(model)
    batch = left_pad_batch(prompt_ids_list, completion_ids_list, pad_id, device=device, vocab=vocab)
    if padding_side == 'none':
        require(not any(batch['offsets']), "padding_side='none' needs rows of equal total length")
        hidden = model.model(input_ids=batch['input_ids'], use_cache=False).last_hidden_state
    else:
        hidden = model.model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'].long(),
                             position_ids=batch['position_ids'], use_cache=False).last_hidden_state
    head = model.get_output_embeddings()
    recompute = checkpoint_head and torch.is_grad_enabled()
    logprobs, argmax = [], []
    for i, (offset, p, c) in enumerate(zip(batch['offsets'], batch['prompt_lengths'],
                                           batch['completion_lengths'])):
        start = offset + p - 1
        rows = hidden[i, start:start + c]
        targets = batch['input_ids'][i, start + 1:start + 1 + c]
        if recompute:
            from torch.utils.checkpoint import checkpoint
            logprobs.append(checkpoint(_head_logprobs, head, rows, targets, use_reentrant=False))
        elif return_argmax:
            logits = head(rows).float()
            logprobs.append(logits.log_softmax(-1).gather(-1, targets[:, None]).squeeze(-1))
            argmax.append(logits.argmax(-1))
        else:
            logprobs.append(_head_logprobs(head, rows, targets))
    for values in logprobs:
        require(bool(torch.isfinite(values).all()), 'nonfinite teacher-forced logprob')
    return (logprobs, argmax) if return_argmax else logprobs


def cached_logprobs(model, prompt_ids, completion_ids, prefill=True, return_argmax=False):
    """Selected logprobs by incremental decoding with past_key_values, one completion token per forward.

    prefill=True runs the prompt in one forward (as generate does); prefill=False feeds the prompt one token at
    a time too. Only the last position's logits are computed at each step.
    """
    device, vocab = _device(model), _vocab(model)
    prompt = _ids(prompt_ids, vocab, 'prompt')
    completion = _ids(completion_ids, vocab, 'completion')

    def step(tokens, cache):
        out = model(input_ids=torch.tensor([tokens], dtype=torch.long, device=device), past_key_values=cache,
                    use_cache=True, logits_to_keep=1)
        return out.logits[0, -1].float(), out.past_key_values

    cache = None
    for chunk in ([prompt] if prefill else [[t] for t in prompt]):
        last, cache = step(chunk, cache)
    logprobs, argmax = [], []
    for j, token in enumerate(completion):
        logprobs.append(last.log_softmax(-1)[token])
        argmax.append(last.argmax())
        if j + 1 < len(completion):
            last, cache = step([token], cache)
    require(cache.get_seq_length() == len(prompt) + len(completion) - 1, 'KV cache length')
    values = torch.stack(logprobs)
    require(bool(torch.isfinite(values).all()), 'nonfinite cached logprob')
    return (values, torch.stack(argmax)) if return_argmax else values


def _pieces(value):
    if isinstance(value, torch.Tensor):
        return [value.detach().to('cpu', torch.float64)]
    require(type(value) in (list, tuple) and len(value) > 0, 'parity input must be a tensor or non-empty list')
    if all(type(v) in (int, float) for v in value):
        return [torch.tensor(value, dtype=torch.float64)]
    require(all(isinstance(v, torch.Tensor) or type(v) in (list, tuple) for v in value),
            'parity input mixes scalars and sequences')
    return [p for v in value for p in _pieces(v)]


def parity(a, b):
    """Compare two logprob/score collections of identical structure (tensors, lists, or ragged per-sequence lists).

    max_abs is over every element. For per-token scalars argmax_equal is None (no vocabulary axis); for score rows
    (last axis = vocabulary) it is True iff every row has the same argmax. bitwise_equal compares exact values.
    """
    left, right = _pieces(a), _pieces(b)
    require(len(left) == len(right) and all(x.shape == y.shape for x, y in zip(left, right)),
            'parity inputs differ in structure')
    require(all(x.ndim == left[0].ndim for x in left) and left[0].ndim in (1, 2), 'parity rank')
    x, y = torch.cat(left), torch.cat(right)
    require(x.numel() > 0 and bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all()),
            'parity inputs must be finite and non-empty')
    argmax_equal = bool(torch.equal(x.argmax(-1), y.argmax(-1))) if x.ndim == 2 else None
    return {'n': int(x.shape[0]), 'max_abs': float((x - y).abs().max()), 'argmax_equal': argmax_equal,
            'bitwise_equal': bool(torch.equal(x, y))}


# ---------------------------------------------------------------- resumable read journal

def _canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def _parse_line(raw: bytes):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate JSON key')
            result[key] = value
        return result

    def invalid(value):
        raise ValueError('nonfinite JSON: ' + value)
    return json.loads(raw.decode('ascii'), object_pairs_hook=unique, parse_constant=invalid)


class ResumableJsonl:
    """Append-only JSONL read journal: one canonical JSON row per line, each carrying the sha256 of its content.

    - append(row) writes one LF-terminated canonical line (sorted keys, ASCII, no NaN) in one write, then flush
      and fsync. The row's `key` value must be a new non-empty string; `bind` fields (e.g. arm digest,
      decoding_sha256) are added to every row and must match on reopen.
    - Reopening verifies every stored line: canonical bytes, sha256 of the row without HASH_FIELD, unique keys,
      bind fields. Any failure refuses (ValueError); nothing is repaired.
    - The single exception is a trailing fragment without its LF (an interrupted write): it is never counted as a
      row, it is copied into '<name>.recovery.jsonl' (base64 plus sha256, itself hashed and fsynced) and only then
      truncated. `recovered` holds that record; a complete line that fails verification is never truncated.
    - done_ids lists completed keys in file order; next_index(planned) is where an interrupted read resumes, and
      refuses unless the journal is exactly a prefix of the planned order.
    Single writer per file: an append refuses if the file length changed behind this object's back.
    """
    HASH_FIELD = 'row_sha256'

    def __init__(self, path, key='id', bind=None):
        self.path = Path(path)
        require(type(key) is str and key and key != self.HASH_FIELD, 'journal key field')
        bind = dict(bind or {})
        require(self.HASH_FIELD not in bind and key not in bind, 'bind fields collide with key/hash')
        _canonical(bind)
        self.key, self.bind = key, bind
        self.recovered = None
        self._ids, self._seen = [], set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, 'xb'):
                pass
        self._size = self._load()

    @property
    def recovery_path(self):
        return self.path.with_name(self.path.name + '.recovery.jsonl')

    def _verified(self, data: bytes):
        """Verify every LF-terminated line of `data`; yield (key, stored row)."""
        seen = set()
        for number, raw in enumerate(data.split(b'\n')[:-1], start=1):
            try:
                row = _parse_line(raw)
            except ValueError as error:
                raise ValueError('unparseable journal line %d: %s' % (number, error)) from error
            require(type(row) is dict and _canonical(row) == raw, 'non-canonical journal line %d' % number)
            body = dict(row)
            stored = body.pop(self.HASH_FIELD, None)
            require(type(stored) is str and stored == digest(body), 'row hash mismatch at journal line %d' % number)
            rid = body.get(self.key)
            require(type(rid) is str and rid, 'journal line %d has no key' % number)
            require(rid not in seen, 'duplicate key at journal line %d: %s' % (number, rid))
            for name, value in self.bind.items():
                require(name in body and body[name] == value,
                        'bind field %s differs at journal line %d' % (name, number))
            seen.add(rid)
            yield rid, row

    def _load(self):
        with open(self.path, 'rb') as stream:
            data = stream.read()
        end = data.rfind(b'\n') + 1
        for rid, _ in self._verified(data[:end]):
            self._ids.append(rid)
            self._seen.add(rid)
        fragment = data[end:]
        if fragment:
            self._record_truncation(end, fragment)
            with open(self.path, 'r+b') as stream:
                stream.truncate(end)
                stream.flush()
                os.fsync(stream.fileno())
        return end

    def _record_truncation(self, offset, fragment):
        record = {'event': 'TRUNCATED_PARTIAL_LAST_LINE', 'journal': self.path.name, 'offset': offset,
                  'fragment_bytes': len(fragment), 'fragment_sha256': hashlib.sha256(fragment).hexdigest(),
                  'fragment_b64': base64.b64encode(fragment).decode('ascii'), 'rows_verified': len(self._ids),
                  'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
        record['record_sha256'] = digest(record)
        with open(self.recovery_path, 'ab') as stream:
            stream.write(_canonical(record) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        self.recovered = record

    @property
    def done_ids(self):
        return list(self._ids)

    def __len__(self):
        return len(self._ids)

    def __contains__(self, rid):
        return rid in self._seen

    def next_index(self, planned_ids):
        planned = list(planned_ids)
        require(len(set(planned)) == len(planned), 'planned IDs repeat')
        require(planned[:len(self._ids)] == self._ids, 'journal is not a prefix of the planned read order')
        return len(self._ids)

    def append(self, row):
        require(type(row) is dict, 'journal row must be a dict')
        require(self.HASH_FIELD not in row, 'row already carries ' + self.HASH_FIELD)
        rid = row.get(self.key)
        require(type(rid) is str and rid, 'row key must be a non-empty string')
        require(rid not in self._seen, 'row already in journal: ' + rid)
        for name, value in self.bind.items():
            require(name not in row or row[name] == value, 'row contradicts bind field ' + name)
        body = dict(row, **self.bind)
        stored = dict(body)
        stored[self.HASH_FIELD] = digest(body)
        line = _canonical(stored) + b'\n'
        with open(self.path, 'ab') as stream:
            require(os.fstat(stream.fileno()).st_size == self._size, 'journal changed behind this writer')
            stream.write(line)
            stream.flush()
            os.fsync(stream.fileno())
        self._size += len(line)
        self._ids.append(rid)
        self._seen.add(rid)
        return stored

    def rows(self):
        """Re-read and re-verify the whole journal; returns the stored rows (with their hash field) in order."""
        with open(self.path, 'rb') as stream:
            data = stream.read()
        require(data.endswith(b'\n') or not data, 'journal has a partial last line; reopen it to recover')
        rows = list(self._verified(data))
        require([rid for rid, _ in rows] == self._ids, "journal rows differ from this writer's record")
        return [row for _, row in rows]
