"""S0 decoding, teacher forcing, cache parity and the resumable journal (Amendment 1 sections 8, 12, 16).

CPU FP32 only, on a tiny random Qwen3 (vocab 151936, hidden 32, 2 layers) saved as BF16 safetensors with a
config.json in the real format. The real Supa weights are never opened; only its tokenizer/config files are read.

The tiny model is crafted so that EOS behaviour is testable: every token carries a constant 0.1 in dimension 1,
and the EOS row points the other way (-8 there, +24 in dimension 0), so EOS is unlikely in ordinary contexts.
TRIGGER's row is aligned with the final hidden state that is specific to PLAIN (PLAIN minus CHAT), plus 1.0 in
dimension 0, so after PLAIN the model prefers TRIGGER and after TRIGGER it emits EOS. Outcomes (pinned by
test_fixture_construction): greedy PLAIN -> [TRIGGER, EOS]; greedy CHAT -> [tok, EOS]; greedy HELLO never
stops before the cap; sampled PLAIN groups mix EOS stops at several lengths with length stops.
"""
from __future__ import annotations

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from experiments.maxrl_comp002_pretraining_v1 import s0_generate as g
from experiments.maxrl_comp002_pretraining_v1.mechanics import completion_labels, selected_logprobs
from experiments.maxrl_comp002_pretraining_v1.preflight import digest

REAL = Path(r"C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze")
TINY = {'hidden_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
        'head_dim': 8, 'intermediate_size': 64, 'max_window_layers': 2}
IM_END, ENDOFTEXT = 151645, 151643
EOS = (IM_END, ENDOFTEXT)
PAD = ENDOFTEXT
TRIGGER = 90000
TOL = 1e-5
MESSAGES = [{'role': 'system', 'content': 'You are a concise assistant.'},
            {'role': 'user', 'content': 'Reply with the JSON object {"answer": [1]}.'}]


def _craft(model, plain, chat):
    embed = model.model.embed_tokens.weight
    with torch.no_grad():
        embed[:, 1] = 0.1
        embed[IM_END].zero_()
        embed[IM_END, 0], embed[IM_END, 1] = 24.0, -8.0
        embed[TRIGGER].zero_()
        h_plain = model.model(input_ids=torch.tensor([plain])).last_hidden_state[0, -1]
        h_chat = model.model(input_ids=torch.tensor([chat])).last_hidden_state[0, -1]
        direction = h_plain - h_chat
        direction[:2] = 0
        embed[TRIGGER] = direction / direction.norm() * 3.8
        embed[TRIGGER, 0] = 1.0


def _load_fp32(root):
    """Stream the stored BF16 tensors into an FP32 model; lm_head stays tied to the input embedding."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    config = Qwen3Config.from_pretrained(str(root))
    model = Qwen3ForCausalLM(config)
    stored = load_file(str(root / 'model.safetensors'))
    assert all(t.dtype == torch.bfloat16 for t in stored.values())
    missing, unexpected = model.load_state_dict({k: v.float() for k, v in stored.items()}, strict=False)
    assert list(missing) == ['lm_head.weight'] and not unexpected
    assert model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    assert next(model.parameters()).dtype == torch.float32
    return model.eval()


@pytest.fixture(scope='module')
def tiny(tmp_path_factory):
    if not (REAL / 'config.json').is_file() or not (REAL / 'tokenizer.json').is_file():
        pytest.skip('Supa tokenizer/config files absent: ' + str(REAL))
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    root = tmp_path_factory.mktemp('tiny_qwen3')
    real = json.loads((REAL / 'config.json').read_text(encoding='utf-8'))
    with open(root / 'config.json', 'x', encoding='utf-8', newline='\n') as stream:
        json.dump(dict(real, **TINY), stream, indent=2, sort_keys=True)
        stream.write('\n')
    tokenizer = AutoTokenizer.from_pretrained(str(REAL), local_files_only=True, trust_remote_code=False)
    rendered = tokenizer.apply_chat_template(MESSAGES, tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False)
    prompts = {'plain': tokenizer('The quick brown fox jumps over', add_special_tokens=False)['input_ids'],
               'chat': tokenizer(rendered, add_special_tokens=False)['input_ids'],
               'hello': tokenizer('Hello world', add_special_tokens=False)['input_ids']}
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        seed_model = Qwen3ForCausalLM(Qwen3Config.from_pretrained(str(root))).eval()
    _craft(seed_model, prompts['plain'], prompts['chat'])
    state = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in seed_model.state_dict().items()
             if k != 'lm_head.weight'}
    save_file(state, str(root / 'model.safetensors'), metadata={'format': 'pt'})
    del seed_model, state
    return SimpleNamespace(model=_load_fp32(root), root=root, prompts=prompts)


def _rows(n=3):
    return [{'id': 'row-%d' % i, 'completion_ids': [i, i + 1, IM_END], 'raw_step_logprobs': [-0.1 * i, -1e-7, -3.25],
             'stop_reason': 'eos'} for i in range(n)]


# ---------------------------------------------------------------- the model under test

def test_fixture_construction(tiny):
    """The stored checkpoint has the real layout, and the crafted EOS behaviour holds after the BF16 round trip."""
    config = json.loads((tiny.root / 'config.json').read_text(encoding='utf-8'))
    assert config['vocab_size'] == 151936 and config['tie_word_embeddings'] is True
    assert config['model_type'] == 'qwen3' and config['torch_dtype'] == 'bfloat16'
    assert {k: config[k] for k in TINY} == TINY
    assert 'lm_head.weight' not in load_file(str(tiny.root / 'model.safetensors'))
    plain = g.greedy_decode(tiny.model, tiny.prompts['plain'], max_new_tokens=8)
    chat = g.greedy_decode(tiny.model, tiny.prompts['chat'], max_new_tokens=8)
    hello = g.greedy_decode(tiny.model, tiny.prompts['hello'], max_new_tokens=8)
    assert plain['completion_ids'] == [TRIGGER, IM_END] and plain['stop_reason'] == 'eos'
    assert len(chat['completion_ids']) == 2 and chat['completion_ids'][-1] == IM_END
    assert chat['completion_ids'][0] != TRIGGER and chat['stop_reason'] == 'eos'
    assert hello['stop_reason'] == 'length' and hello['generated_tokens'] == 8
    assert not set(hello['completion_ids']) & set(EOS)


# ---------------------------------------------------------------- greedy vs teacher forcing vs cache (T3/T4)

@pytest.mark.parametrize('name', ['plain', 'chat', 'hello'])
def test_greedy_raw_logprobs_equal_teacher_forced_and_cache(tiny, name):
    prompt = tiny.prompts[name]
    result = g.greedy_decode(tiny.model, prompt, max_new_tokens=8)
    completion = result['completion_ids']
    assert result['generated_tokens'] == len(completion) == len(result['raw_step_logprobs'])
    if result['stop_reason'] == 'eos':
        assert completion[-1] in EOS and not set(completion[:-1]) & set(EOS)
    with torch.inference_mode():
        (single,), (single_argmax,) = g.teacher_forced_logprobs(tiny.model, [prompt], [completion], PAD,
                                                                padding_side='none', return_argmax=True)
        batch = g.teacher_forced_logprobs(tiny.model, [tiny.prompts['chat'], prompt], [[5, 6, 7, IM_END], completion],
                                          PAD, padding_side='left')
        cached, cached_argmax = g.cached_logprobs(tiny.model, prompt, completion, return_argmax=True)
        stepped = g.cached_logprobs(tiny.model, prompt, completion, prefill=False)
    for other in (single, batch[1], cached, stepped):
        check = g.parity(result['raw_step_logprobs'], other)
        assert check['n'] == len(completion) and check['max_abs'] <= TOL and check['argmax_equal'] is None
    assert single_argmax.tolist() == completion and cached_argmax.tolist() == completion


def test_greedy_records_the_decoder_it_ran(tiny):
    result = g.greedy_decode(tiny.model, tiny.prompts['plain'], max_new_tokens=8)
    decoder = result['decoder']
    assert decoder['logits_processors'] == [g.RECORDER]
    assert decoder['stopping_criteria'] == ['MaxLengthCriteria', 'EosTokenCriteria']
    config = decoder['effective_config']
    assert config['do_sample'] is False and config['num_beams'] == 1 and config['use_cache'] is True
    assert config['eos_token_id'] == list(EOS) and config['pad_token_id'] == PAD
    assert config['max_new_tokens'] == 8 and config['max_length'] == len(tiny.prompts['plain']) + 8
    assert decoder['raw_logits_source'] == 'output_logits' and decoder['attn_implementation'] == 'sdpa'
    json.dumps(result, allow_nan=False)


def test_greedy_refuses_training_mode_other_versions_and_bad_ids(tiny, monkeypatch):
    tiny.model.train()
    try:
        with pytest.raises(ValueError, match='eval mode'):
            g.greedy_decode(tiny.model, tiny.prompts['plain'], max_new_tokens=2)
    finally:
        tiny.model.eval()
    for bad in ([], [True, 3], [151936], [-1]):
        with pytest.raises(ValueError, match='prompt'):
            g.greedy_decode(tiny.model, bad, max_new_tokens=2)
    with pytest.raises(ValueError, match='max_new_tokens'):
        g.greedy_decode(tiny.model, tiny.prompts['plain'], max_new_tokens=0)
    monkeypatch.setattr(g, 'TRANSFORMERS_VERSION', '4.56.0')
    with pytest.raises(ValueError, match='transformers 4.56.0'):
        g.greedy_decode(tiny.model, tiny.prompts['plain'], max_new_tokens=2)
    assert '_get_logits_processor' not in vars(tiny.model)


# ---------------------------------------------------------------- teacher forcing under left padding (T5)

BATCH_COMPLETIONS = [[TRIGGER, IM_END], [11, 12, 13, 14, 15, 16], [7, 8, ENDOFTEXT], [9, IM_END]]
BATCH_REASONS = ['eos', 'length', 'eos', 'eos']


def _batch_prompts(tiny):
    return [tiny.prompts['plain'], tiny.prompts['chat'], tiny.prompts['hello'], tiny.prompts['hello'][:1]]


def test_left_padded_batch_equals_individual_rows(tiny):
    prompts = _batch_prompts(tiny)
    with torch.inference_mode():
        batch = g.teacher_forced_logprobs(tiny.model, prompts, BATCH_COMPLETIONS, PAD, padding_side='left')
        singles = [g.teacher_forced_logprobs(tiny.model, [p], [c], PAD, padding_side='none')[0]
                   for p, c in zip(prompts, BATCH_COMPLETIONS)]
        check = g.parity(batch, singles)
        assert check['n'] == sum(map(len, BATCH_COMPLETIONS)) and check['max_abs'] <= TOL
        # control: the same padded batch with the pads attended is far from the individual rows
        padded = g.left_pad_batch(prompts, BATCH_COMPLETIONS, PAD)
        assert padded['offsets'][1] == 0 and all(padded['offsets'][i] > 0 for i in (0, 2, 3))
        leaked = tiny.model(input_ids=padded['input_ids'][3:], use_cache=False).logits[0].log_softmax(-1)
        start = padded['offsets'][3] + padded['prompt_lengths'][3] - 1
        row = leaked[start:start + 2].gather(-1, torch.tensor(BATCH_COMPLETIONS[3])[:, None]).squeeze(-1)
        assert g.parity(row, singles[3])['max_abs'] > 1e-3
    with pytest.raises(ValueError, match="equal total length"):
        g.teacher_forced_logprobs(tiny.model, prompts, BATCH_COMPLETIONS, PAD, padding_side='none')
    with pytest.raises(ValueError, match='padding_side'):
        g.teacher_forced_logprobs(tiny.model, prompts, BATCH_COMPLETIONS, PAD, padding_side='right')
    with pytest.raises(ValueError, match='completion'):
        g.teacher_forced_logprobs(tiny.model, prompts[:1], [[]], PAD)


def test_labels_include_eos_and_exclude_padding(tiny):
    """mechanics.completion_labels/selected_logprobs on full logits select exactly teacher_forced's positions."""
    prompts = _batch_prompts(tiny)
    padded = g.left_pad_batch(prompts, BATCH_COMPLETIONS, PAD)
    ids, attention = padded['input_ids'], padded['attention_mask']
    assert attention.dtype == torch.bool
    labels = completion_labels(ids, attention, padded['prompt_lengths'], padded['completion_lengths'],
                               list(EOS), BATCH_REASONS, 6)
    with torch.inference_mode():
        logits = tiny.model(input_ids=ids, attention_mask=attention.long(), position_ids=padded['position_ids'],
                            use_cache=False).logits
        reference = selected_logprobs(logits, ids, labels)
        ours = g.teacher_forced_logprobs(tiny.model, prompts, BATCH_COMPLETIONS, PAD)
    check = g.parity(torch.cat(ours), reference)
    assert check['n'] == reference.numel() == sum(map(len, BATCH_COMPLETIONS)) and check['max_abs'] <= TOL
    pads = ~attention
    assert bool((ids[pads] == PAD).all()) and bool((labels[pads] == -100).all())  # PAD == an EOS id, masked
    for i, (completion, reason) in enumerate(zip(BATCH_COMPLETIONS, BATCH_REASONS)):
        kept = labels[i][labels[i] != -100].tolist()
        assert kept == completion
        assert (kept[-1] in EOS) == (reason == 'eos')


def test_teacher_forced_applies_the_head_only_at_completion_positions_and_keeps_gradients(tiny):
    prompts = _batch_prompts(tiny)
    seen = []
    handle = tiny.model.lm_head.register_forward_hook(lambda m, inp, out: seen.append(inp[0].shape[:-1].numel()))
    try:
        with torch.enable_grad():
            logprobs = g.teacher_forced_logprobs(tiny.model, prompts, BATCH_COMPLETIONS, PAD)
            assert all(lp.requires_grad for lp in logprobs)
            loss = -sum(lp.sum() for lp in logprobs)
            loss.backward()
    finally:
        handle.remove()
    assert sum(seen) == sum(map(len, BATCH_COMPLETIONS))
    grad = tiny.model.model.embed_tokens.weight.grad
    assert grad is not None and bool(torch.isfinite(grad).all()) and float(grad.abs().sum()) > 0
    tiny.model.zero_grad(set_to_none=True)


def test_checkpoint_head_saves_no_vocab_wide_tensor_and_changes_nothing(tiny):
    prompts = _batch_prompts(tiny)

    def run(flag):
        saved = []

        def pack(tensor):
            saved.append(tuple(tensor.shape))
            return tensor
        with torch.enable_grad(), torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
            logprobs = g.teacher_forced_logprobs(tiny.model, prompts, BATCH_COMPLETIONS, PAD, checkpoint_head=flag)
            loss = -sum(lp.sum() for lp in logprobs)
        loss.backward()
        grads = {n: p.grad.clone() for n, p in tiny.model.named_parameters() if p.grad is not None}
        tiny.model.zero_grad(set_to_none=True)
        wide = [s for s in saved if len(s) == 2 and s[-1] == 151936]
        return [lp.detach() for lp in logprobs], grads, wide

    plain_values, plain_grads, plain_wide = run(False)
    ckpt_values, ckpt_grads, ckpt_wide = run(True)
    # unpatched: each row saves its (C, V) log-softmax (gather refers to the same tensor) and the (H, V) head
    # weight view; checkpointed: nothing vocabulary-wide is saved outside the recomputed region
    activations = {s[0] for s in plain_wide if s[0] != TINY['hidden_size']}
    assert activations == {len(c) for c in BATCH_COMPLETIONS} and ckpt_wide == []
    assert g.parity(plain_values, ckpt_values)['bitwise_equal'] is True
    assert plain_grads.keys() == ckpt_grads.keys() and plain_grads
    assert max(float((plain_grads[n] - ckpt_grads[n]).abs().max()) for n in plain_grads) <= 1e-7
    with pytest.raises(ValueError, match='checkpoint_head'):
        g.teacher_forced_logprobs(tiny.model, prompts, BATCH_COMPLETIONS, PAD, checkpoint_head=True,
                                  return_argmax=True)


def test_greedy_refuses_a_generate_output_that_is_not_the_raw_argmax(tiny):
    """If generate's token were not the argmax of the raw logits it returns, the read is refused."""
    real = tiny.model.generate

    def tampered(*args, **kwargs):
        out = real(*args, **kwargs)
        first = out.logits[0].clone()
        first[0, 5] = first.max() + 1.0
        out.logits = (first,) + tuple(out.logits[1:])
        return out
    tiny.model.generate = tampered
    try:
        with pytest.raises(ValueError, match='raw-logit argmax'):
            g.greedy_decode(tiny.model, tiny.prompts['plain'], max_new_tokens=4)
    finally:
        del tiny.model.generate
    assert g.greedy_decode(tiny.model, tiny.prompts['plain'], max_new_tokens=4)['completion_ids'] == [TRIGGER, IM_END]


# ---------------------------------------------------------------- sampling (Q2 sampler)

def test_sampled_group_is_reproducible_seed_sensitive_and_rng_neutral(tiny):
    prompt = tiny.prompts['plain']
    torch.manual_seed(1)
    before = torch.get_rng_state()
    first = g.sample_group(tiny.model, prompt, 8, 12345, max_new_tokens=6)
    assert torch.equal(torch.get_rng_state(), before)  # the caller's RNG is restored
    torch.manual_seed(2)  # a different global state does not change the draw
    again = g.sample_group(tiny.model, prompt, 8, 12345, max_new_tokens=6)
    other = g.sample_group(tiny.model, prompt, 8, 12346, max_new_tokens=6)
    assert first == again
    ids = lambda r: [c['completion_ids'] for c in r['completions']]
    assert ids(first) != ids(other)
    reasons = [c['stop_reason'] for c in first['completions']]
    assert 'eos' in reasons and 'length' in reasons
    assert len({len(c) for c in ids(first) if c[-1] in EOS}) >= 2  # EOS at more than one position
    for c in first['completions']:
        assert c['generated_tokens'] == len(c['completion_ids']) == len(c['raw_step_logprobs'])
        if c['stop_reason'] == 'length':
            assert len(c['completion_ids']) == 6 and not set(c['completion_ids']) & set(EOS)
        else:
            assert c['completion_ids'][-1] in EOS and not set(c['completion_ids'][:-1]) & set(EOS)
    assert first['decode_steps'] == max(len(c) for c in ids(first))
    sampler = first['sampler']
    assert sampler['logits_processors'] == [g.RECORDER] and sampler['replay_verified'] is True
    assert {k: sampler['effective_config'][k] for k in g.SAMPLER} == g.SAMPLER
    assert sampler['seed_path'] == g.SEED_PATH
    with torch.inference_mode():
        forced = g.teacher_forced_logprobs(tiny.model, [prompt] * 8, ids(first), PAD)
    check = g.parity([c['raw_step_logprobs'] for c in first['completions']], forced)
    assert check['max_abs'] <= TOL and check['n'] == sum(map(len, ids(first)))
    json.dumps(first, allow_nan=False)
    for bad in (-1, 2 ** 63, 1.5, True):
        with pytest.raises(ValueError, match='seed'):
            g.sample_group(tiny.model, prompt, 2, bad, max_new_tokens=2)
    with pytest.raises(ValueError, match='group size'):
        g.sample_group(tiny.model, prompt, 0, 1, max_new_tokens=2)


def _direct_generate(model, prompt, G, seed, cap, **overrides):
    """transformers' own sampling with the default CPU generator seeded as SEED_PATH states."""
    from transformers import GenerationConfig

    fields = dict(g.SAMPLER, num_beams=1, max_new_tokens=cap, eos_token_id=list(EOS), pad_token_id=PAD,
                  use_cache=True, return_dict_in_generate=True, output_logits=True)
    fields.update(overrides)
    ids = torch.tensor([prompt] * G)
    with torch.random.fork_rng(devices=[]), torch.inference_mode():
        torch.default_generator.manual_seed(seed)
        return model.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                              generation_config=GenerationConfig(**fields), use_model_defaults=False)


def test_seed_reaches_sampling_through_the_default_generator_and_stream_equals_output_logits(tiny):
    prompt, G, seed = tiny.prompts['plain'], 8, 777
    ours = g.sample_group(tiny.model, prompt, G, seed, max_new_tokens=6)
    direct = _direct_generate(tiny.model, prompt, G, seed, 6)
    rows = direct.sequences[:, len(prompt):].tolist()
    raw = torch.stack([step.log_softmax(-1) for step in direct.logits])  # (steps, G, V)
    for i, c in enumerate(ours['completions']):
        n = c['generated_tokens']
        assert rows[i][:n] == c['completion_ids']
        streamed = torch.tensor(c['raw_step_logprobs'], dtype=torch.float32)
        from_output_logits = raw[:n, i].gather(-1, torch.tensor(c['completion_ids'])[:, None]).squeeze(-1)
        assert torch.equal(streamed, from_output_logits)


def test_replay_refuses_a_seed_that_did_not_reach_the_sampler(tiny, monkeypatch):
    real = g._seeded_default_generator

    @contextlib.contextmanager
    def off_by_one(device, seed):
        with real(device, seed + 1):
            yield
    monkeypatch.setattr(g, '_seeded_default_generator', off_by_one)
    with pytest.raises(ValueError, match='replay'):
        g.sample_group(tiny.model, tiny.prompts['plain'], 8, 12345, max_new_tokens=6)
    assert '_get_logits_processor' not in vars(tiny.model)


@pytest.mark.parametrize('call,extra', [('sample', {'temperature': 0.7}), ('sample', {'top_k': 20}),
                                        ('greedy', {'repetition_penalty': 1.3})])
def test_an_active_processor_is_refused(tiny, monkeypatch, call, extra):
    real = g._generation_config
    monkeypatch.setattr(g, '_generation_config', lambda **fields: real(**dict(fields, **extra)))
    with pytest.raises(ValueError, match='extra logits processors'):
        if call == 'sample':
            g.sample_group(tiny.model, tiny.prompts['plain'], 2, 5, max_new_tokens=2)
        else:
            g.greedy_decode(tiny.model, tiny.prompts['plain'], max_new_tokens=2)


def _ranks(model, prompt, tokens):
    with torch.inference_mode():
        logits = model(input_ids=torch.tensor([prompt])).logits[0, -1]
    return [int((logits > logits[t]).sum()) for t in tokens]


def test_top_k_zero_really_disables_top_k(tiny):
    """Near-uniform first step: top-k 0 draws far outside the top 50; the same seed under top-k 50 cannot."""
    prompt, G, seed = tiny.prompts['hello'], 16, 7
    ours = g.sample_group(tiny.model, prompt, G, seed, max_new_tokens=1)
    firsts = [c['completion_ids'][0] for c in ours['completions']]
    ranks = _ranks(tiny.model, prompt, firsts)
    assert sum(r >= 50 for r in ranks) >= 12 and max(ranks) >= 1000
    assert ours['sampler']['effective_config']['top_k'] == 0
    control = _direct_generate(tiny.model, prompt, G, seed, 1, top_k=50)
    control_ranks = _ranks(tiny.model, prompt, control.sequences[:, len(prompt)].tolist())
    assert max(control_ranks) < 50
    # omitting top_k leaves transformers' global default of 50 in force
    from transformers import GenerationConfig
    assert GenerationConfig().top_k == 50


def test_model_generation_defaults_do_not_leak(tiny):
    """The Supa generation_config.json (T 0.6, top-k 20, top-p 0.95) attached to the model changes nothing."""
    from transformers import GenerationConfig

    prompt = tiny.prompts['hello']
    plain_run = g.sample_group(tiny.model, prompt, 16, 7, max_new_tokens=3)
    original = tiny.model.generation_config
    supa = GenerationConfig.from_pretrained(str(REAL))
    assert (supa.temperature, supa.top_k, supa.top_p) == (0.6, 20, 0.95)
    tiny.model.generation_config = supa
    try:
        attached = g.sample_group(tiny.model, prompt, 16, 7, max_new_tokens=3)
        assert attached['completions'] == plain_run['completions']
        assert {k: attached['sampler']['effective_config'][k] for k in g.SAMPLER} == g.SAMPLER
        # the trap use_model_defaults=False avoids: defaults-equal fields silently take the model's values
        trap, _ = tiny.model._prepare_generation_config(
            GenerationConfig(do_sample=True, temperature=1.0, top_k=0, top_p=1.0), use_model_defaults=None)
        assert (trap.temperature, trap.top_k, trap.top_p) == (0.6, 0, 0.95)
    finally:
        tiny.model.generation_config = original


def test_generation_contract_binds_the_semantics():
    contract = g.generation_contract()
    body = {k: v for k, v in contract.items() if k != 'contract_sha256'}
    assert contract['contract_sha256'] == digest(body)
    assert contract['sampler_semantics'] == g.SAMPLER_SEMANTICS and contract['seed_path'] == g.SEED_PATH
    assert contract['eos_token_ids'] == list(EOS) and contract['pad_token_id'] == PAD
    assert contract['max_new_tokens'] == 128 and contract['use_model_defaults'] is False
    assert g.generation_contract(max_new_tokens=64)['contract_sha256'] != contract['contract_sha256']


def test_truncate_rules():
    assert g._truncate([5, IM_END, PAD, PAD], EOS, PAD, 4) == ([5, IM_END], 'eos')
    assert g._truncate([5, ENDOFTEXT, PAD], EOS, PAD, 3) == ([5, ENDOFTEXT], 'eos')
    assert g._truncate([5, 6, 7], EOS, PAD, 3) == ([5, 6, 7], 'length')
    with pytest.raises(ValueError, match='fewer than max_new_tokens'):
        g._truncate([5, 6], EOS, PAD, 3)
    with pytest.raises(ValueError, match='forced padding'):
        g._truncate([5, IM_END, 9], EOS, PAD, 3)


# ---------------------------------------------------------------- parity

def test_parity_reports_structure_argmax_and_exactness():
    a = torch.tensor([[0.0, 1.0, 0.5], [2.0, 0.0, 0.0]])
    b = a.clone()
    b[0, 2] = 0.75
    rows = g.parity(a, b)
    assert rows == {'n': 2, 'max_abs': 0.25, 'argmax_equal': True, 'bitwise_equal': False}
    b[1, 1] = 3.0
    assert g.parity(a, b)['argmax_equal'] is False
    ragged = g.parity([[-1.0, -2.0], torch.tensor([-3.0])], [torch.tensor([-1.0, -2.0]), [-3.0]])
    assert ragged == {'n': 3, 'max_abs': 0.0, 'argmax_equal': None, 'bitwise_equal': True}
    with pytest.raises(ValueError, match='structure'):
        g.parity([[-1.0, -2.0], [-3.0]], [[-1.0], [-2.0, -3.0]])
    with pytest.raises(ValueError, match='finite'):
        g.parity([float('nan')], [0.0])


# ---------------------------------------------------------------- resumable journal

def test_journal_round_trip_bind_and_resume_index(tmp_path):
    path = tmp_path / 'reads' / 'arm-B' / 'guard_dev.jsonl'
    bind = {'arm_digest': 'a' * 64, 'decoding_sha256': 'd' * 64}
    journal = g.ResumableJsonl(path, bind=bind)
    assert path.is_file() and path.read_bytes() == b'' and journal.done_ids == []
    stored = [journal.append(row) for row in _rows(3)]
    assert all(s[g.ResumableJsonl.HASH_FIELD] == digest({k: v for k, v in s.items()
                                                          if k != g.ResumableJsonl.HASH_FIELD}) for s in stored)
    data = path.read_bytes()
    assert b'\r' not in data and data.count(b'\n') == 3
    reopened = g.ResumableJsonl(path, bind=bind)
    assert reopened.done_ids == ['row-0', 'row-1', 'row-2'] and reopened.recovered is None
    assert reopened.rows() == stored and 'row-1' in reopened and len(reopened) == 3
    planned = ['row-0', 'row-1', 'row-2', 'row-3', 'row-4']
    assert reopened.next_index(planned) == 3
    with pytest.raises(ValueError, match='prefix'):
        reopened.next_index(['row-1', 'row-0', 'row-2', 'row-3'])
    with pytest.raises(ValueError, match='already in journal'):
        reopened.append(_rows(1)[0])
    with pytest.raises(ValueError, match='bind field arm_digest'):
        g.ResumableJsonl(path, bind=dict(bind, arm_digest='b' * 64))
    with pytest.raises(ValueError, match='contradicts bind'):
        reopened.append(dict(_rows(4)[3], arm_digest='c' * 64))
    with pytest.raises(ValueError, match='already carries'):
        reopened.append(dict(_rows(4)[3], row_sha256='0' * 64))


def test_journal_resumes_after_a_simulated_crash(tmp_path):
    path = tmp_path / 'crash.jsonl'
    journal = g.ResumableJsonl(path)
    rows = _rows(5)
    for row in rows[:3]:
        journal.append(row)
    intact = path.read_bytes()
    partial = g._canonical(dict(rows[3], row_sha256=digest(rows[3])))[:37]  # killed mid-write, no LF
    with open(path, 'ab') as stream:
        stream.write(partial)
    resumed = g.ResumableJsonl(path)
    assert resumed.done_ids == ['row-0', 'row-1', 'row-2']
    assert path.read_bytes() == intact
    record = resumed.recovered
    assert record['event'] == 'TRUNCATED_PARTIAL_LAST_LINE' and record['offset'] == len(intact)
    assert record['fragment_bytes'] == 37 and record['rows_verified'] == 3
    logged = [json.loads(line) for line in resumed.recovery_path.read_bytes().splitlines()]
    assert len(logged) == 1 and logged[0] == record
    assert record['record_sha256'] == digest({k: v for k, v in record.items() if k != 'record_sha256'})
    assert resumed.next_index([r['id'] for r in rows]) == 3
    for row in rows[3:]:
        resumed.append(row)
    final = g.ResumableJsonl(path)
    assert final.done_ids == [r['id'] for r in rows] and final.recovered is None
    assert len(resumed.recovery_path.read_bytes().splitlines()) == 1


def test_journal_refuses_a_mismatched_hash_and_repairs_nothing(tmp_path):
    path = tmp_path / 'tampered.jsonl'
    journal = g.ResumableJsonl(path)
    for row in _rows(3):
        journal.append(row)
    lines = path.read_bytes().split(b'\n')
    row = json.loads(lines[1])
    row['stop_reason'] = 'length'  # content changed, stored hash kept
    lines[1] = g._canonical(row)
    tampered = b'\n'.join(lines)
    path.write_bytes(tampered)
    with pytest.raises(ValueError, match='row hash mismatch at journal line 2'):
        g.ResumableJsonl(path)
    assert path.read_bytes() == tampered and not (tmp_path / 'tampered.jsonl.recovery.jsonl').exists()


@pytest.mark.parametrize('damage', ['crlf', 'middle_fragment', 'duplicate'])
def test_journal_refuses_damage_that_is_not_a_trailing_fragment(tmp_path, damage):
    path = tmp_path / 'damaged.jsonl'
    journal = g.ResumableJsonl(path)
    for row in _rows(3):
        journal.append(row)
    lines = path.read_bytes().split(b'\n')[:-1]
    if damage == 'crlf':
        lines[0] += b'\r'
        expected = 'non-canonical journal line 1'
    elif damage == 'middle_fragment':
        lines[1] = lines[1][:20]
        expected = 'unparseable journal line 2'
    else:
        lines.append(lines[0])
        expected = 'duplicate key at journal line 4'
    damaged = b'\n'.join(lines) + b'\n'
    path.write_bytes(damaged)
    with pytest.raises(ValueError, match=expected):
        g.ResumableJsonl(path)
    assert path.read_bytes() == damaged


def test_journal_refuses_a_second_writer(tmp_path):
    path = tmp_path / 'two.jsonl'
    first, second = g.ResumableJsonl(path), g.ResumableJsonl(path)
    first.append(_rows(1)[0])
    with pytest.raises(ValueError, match='changed behind this writer'):
        second.append(_rows(2)[1])
    assert g.ResumableJsonl(path).done_ids == ['row-0']


def test_no_cuda_was_initialised():
    assert not torch.cuda.is_initialized()
