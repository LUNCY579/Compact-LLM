"""S0-C source competence, S0-E enrollment and the churn floor (Amendment 1 section 16; sections 8.3, 8.5, 12).

CPU only: no test creates a CUDA tensor or initialises CUDA (asserted after every test), takes the real GPU lease or
reads the real host budget (fakes are injected, and registered-mode refusals are checked with the pinned loaders
patched to fail), and the real Supa weights are never opened. The model is a tiny random Qwen3 (vocab 151936,
hidden 32, 2 layers, heads 4/2, head_dim 8, intermediate 64, tied) whose config.json is the real Supa config with
the sizes overridden, saved as BF16 safetensors in tmp_path with a Q1-format asset receipt. Real inputs read: the
Supa tokenizer/config files, manifest.json and the CALIBRATION and guard_dev canonical, policy and oracle files.

The CLIs run end to end on row subsets (calibration_plan / guard_dev_plan replaced), at the registered
max_new_tokens = 128. Chosen completions are replaced by constructed ones (oracle references, a special token in
the body, UNKNOWN) keyed by the rendered prompt, so rewards, diagnostics and the canary-bound re-score are
exercised on real rows. canary.run_canary_on is memoised per argument set for this module (it is deterministic;
the real canary still runs once per distinct call).
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from experiments.maxrl_comp002_pretraining_v1 import arm_build, canary, scoped_scorer
from experiments.maxrl_comp002_pretraining_v1 import s0_churn as churn
from experiments.maxrl_comp002_pretraining_v1 import s0_competence as comp
from experiments.maxrl_comp002_pretraining_v1 import s0_enroll as enroll
from experiments.maxrl_comp002_pretraining_v1 import s0_generate as g
from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as s0
from experiments.maxrl_comp002_pretraining_v1.preflight import digest, sha
from experiments.maxrl_composable_skills_v2 import behaviour, corpus
from experiments.maxrl_verifiable_data_v1 import maxrl

TINY = {'hidden_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
        'head_dim': 8, 'intermediate_size': 64, 'max_window_layers': 2}
IM_END, ENDOFTEXT, IM_START, THINK = 151645, 151643, 151644, 151667   # <think> is a NON-special added token
UNRENDERABLE_ID = 151900
REGISTERED_SCOPE_SHA = '1acd416dea7bcf61cd439fa54665975ffc34c047acc0dfde2a37c1fac370140f'


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(autouse=True)
def torch_state():
    """Restore the process-global torch settings configure() changes; CUDA must never initialise."""
    saved = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.get_num_threads(), os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    yield
    torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
    torch.set_float32_matmul_precision(saved[2])
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[3], saved[4]
    torch.set_num_threads(saved[5])
    if saved[6] is None:
        os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
    else:
        os.environ['CUBLAS_WORKSPACE_CONFIG'] = saved[6]
    assert not torch.cuda.is_initialized()


@pytest.fixture(scope='module', autouse=True)
def memoised_canary():
    """canary.run_canary_on is deterministic; run it once per distinct argument set in this module."""
    real, cache = canary.run_canary_on, {}

    def cached(sections, **kwargs):
        if kwargs.get('reference_receipt') is not None:
            return real(sections, **kwargs)
        key = (tuple((bank.role, bank.bank_sha256, oracle_sha) for bank, _, oracle_sha in sections),
               tuple(kwargs.get('signatures', canary.SIGNATURES)), digest(kwargs.get('scope_receipts') or {}),
               kwargs.get('expected_code_bundle'), bool(kwargs.get('at_freeze')),
               bool(kwargs.get('legacy_exemption')))
        if key not in cache:
            cache[key] = real(sections, **kwargs)
        return cache[key]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(canary, 'run_canary_on', cached)
        yield cache


@pytest.fixture(scope='module')
def real_config():
    if not (s0.MODEL_DIR / 'config.json').is_file() or not (s0.MODEL_DIR / 'tokenizer.json').is_file():
        pytest.skip('Supa tokenizer/config files absent: ' + str(s0.MODEL_DIR))
    return json.loads((s0.MODEL_DIR / 'config.json').read_text(encoding='utf-8'))


def _create(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as stream:
        stream.write(data)


def _write_json(path: Path, value) -> Path:
    _create(path, (json.dumps(value, indent=2, sort_keys=True) + '\n').encode('ascii'))
    return path


@pytest.fixture(scope='module')
def tiny_dir(tmp_path_factory, real_config):
    """A tiny random Qwen3 in the real layout (BF16, no lm_head) and its Q1-format asset receipt."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    root = tmp_path_factory.mktemp('comp_enroll_tiny')
    model_dir = root / 'tiny'
    config = dict(real_config, **TINY)
    _write_json(model_dir / 'config.json', config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260929)
        model = Qwen3ForCausalLM(Qwen3Config(**config))
    state = {k: v.detach().to(torch.bfloat16).contiguous() for k, v in model.state_dict().items()
             if k != 'lm_head.weight'}
    del model
    save_file(state, str(model_dir / 'model.safetensors'), metadata={'format': 'pt'})
    receipt = {'stage': 'Q1_ASSET_CANDIDATE', 'model_path': str(model_dir),
               'assets': {name: {'bytes': (model_dir / name).stat().st_size, 'sha256': sha(model_dir / name)}
                          for name in ('model.safetensors', 'config.json')},
               'architecture': {key: config[key] for key in s0.ARCHITECTURE_KEYS},
               'model_header': {'tensors': len(state), 'dtypes': ['BF16']}}
    asset = _write_json(root / 'asset_receipt.json', receipt)
    return SimpleNamespace(root=model_dir, asset=asset, weights_sha256=receipt['assets']['model.safetensors']['sha256'])


@pytest.fixture(scope='module')
def tokenizer(real_config):
    return s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)


@pytest.fixture(scope='module')
def cal(comp_rs_data):
    return comp.load_calibration(comp_rs_data)


@pytest.fixture(scope='module')
def guard_bank(comp_rs_data):
    return scoped_scorer.load_role(comp_rs_data, 'guard_dev')


@pytest.fixture(scope='module')
def sections(comp_rs_data, cal, guard_bank):
    return comp.canary_sections(comp_rs_data, cal['bank'], guard_bank)


@pytest.fixture(scope='module')
def oracles(sections):
    return {bank.role: table for bank, table, _ in sections}


def cal_subset(rows_by_skill):
    return [rows_by_skill[skill][0] for skill in comp.SKILLS]


def guard_subset(bank):
    """The first row of each of the first three families, in bank order."""
    seen, rows = set(), []
    for row in bank:
        family = row['verifier']['spec']['family']
        if family not in seen:
            seen.add(family)
            rows.append(row)
        if len(rows) == 3:
            break
    return rows


def ref_ids(tokenizer, oracles, row) -> list:
    reference = oracles[row['role']][row['id']]['references'][0]
    ids = tokenizer(reference, add_special_tokens=False)['input_ids']
    assert tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False) == reference
    return ids


def completion(ids) -> dict:
    return {'completion_ids': list(ids), 'stop_reason': 'eos' if ids[-1] in s0.EOS_IDS else 'length',
            'generated_tokens': len(ids), 'raw_step_logprobs': [0.0] * len(ids)}


class Injector:
    """Wrap sample_group / greedy_decode: replace chosen completions, keyed by the rendered prompt's digest."""

    def __init__(self, tokenizer, rows, plan, fail_at=None, error=RuntimeError, perturb=None):
        self.by_digest = {digest(s0.render(tokenizer, row['prompt'])): row['id'] for row in rows}
        self.plan, self.fail_at, self.error, self.perturb = plan, fail_at, error, perturb
        self.calls, self.real_sample, self.real_greedy = 0, g.sample_group, g.greedy_decode
        self.sample_index = {}

    def _tick(self):
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise self.error('simulated crash')

    def sample_group(self, model, prompt_ids, G, seed, max_new_tokens=128, **kwargs):
        self._tick()
        result = self.real_sample(model, prompt_ids, G, seed, max_new_tokens=max_new_tokens, **kwargs)
        rid = self.by_digest[digest(list(prompt_ids))]
        planned = self.plan.get(rid, {})
        if G == 1:                                    # C13: one call per sample; count the sample index per prompt
            k = self.sample_index.get(rid, 0)
            self.sample_index[rid] = k + 1
            if k in planned:
                result['completions'][0] = completion(planned[k])
        else:
            for k, ids in planned.items():
                result['completions'][k] = completion(ids)
        return result

    def greedy_decode(self, model, prompt_ids, max_new_tokens=128, **kwargs):
        self._tick()
        result = self.real_greedy(model, prompt_ids, max_new_tokens=max_new_tokens, **kwargs)
        rid = self.by_digest[digest(list(prompt_ids))]
        if rid in self.plan:
            result.update(completion(self.plan[rid]))
        if self.perturb == rid:                       # one changed first token: same length and stop reason
            ids = list(result['completion_ids'])
            assert ids[0] not in s0.EOS_IDS
            ids[0] = 9707 if ids[0] != 9707 else 9708
            result = dict(result, completion_ids=ids)
        return result


class FakeLease:
    def __init__(self):
        self.LOCK = Path('Z:/fake-temp') / s0.LOCK_NAME
        self.events = []

    def acquire(self, timeout_s, label):
        self.events.append(('acquire', label))
        return 'token-1'

    def release(self, token):
        self.events.append(('release', token))


def recording_writes(mp, lease):
    """Record every receipt write in the lease's event list, so tests see it happen while the lease is held."""
    real = s0.write_receipt

    def write(path, value, *, repo):
        lease.events.append(('write', Path(path).name))
        return real(path, value, repo=repo)
    mp.setattr(s0, 'write_receipt', write)


class FakeHost:
    class LowMemoryAbort(RuntimeError):
        pass

    def __init__(self, free=64.0, abort_at=None):
        self.free, self.abort_at, self.calls = free, abort_at, []

    def snapshot(self):
        return {'available': True, 'free_phys_gib': self.free, 'total_phys_gib': 128.0}

    def require(self, gib, label):
        self.calls.append(('require', gib, label))
        if self.free < gib:
            raise SystemExit('REFUSING TO START ' + label)
        return self.snapshot()

    def guard(self, gib, label):
        self.calls.append(('guard', gib, label))
        if self.abort_at is not None and sum(c[0] == 'guard' for c in self.calls) == self.abort_at:
            raise self.LowMemoryAbort(label + ' standing down')


def common_argv(tiny_dir, out, *extra, floor=True):
    argv = ['--device', 'cpu', '--test-mode', '--out', str(out), '--model-dir', str(tiny_dir.root),
            '--asset-receipt', str(tiny_dir.asset), *extra]
    return argv + (['--host-floor-gib', '0.5'] if floor else [])


def _lines(path: Path) -> list:
    return [json.loads(line) for line in path.read_bytes().decode('ascii').splitlines()]


def _receipt(path: Path) -> dict:
    raw = path.read_bytes()
    assert b'\r' not in raw and raw.endswith(b'\n')
    raw.decode('ascii')
    return json.loads(raw)


# ------------------------------------------------------------------------------------------------ plan
def test_comp_seed_is_the_registered_integer_form():
    for rid in ('MC2-a', 'MC2-5e1b21cc8e571c03a8ebd81a7614090728f1f9ea021d7db2aa88d657d113235b'):
        expected = int(hashlib.sha256(('COMP-RS-001-S0-COMP|' + rid).encode()).hexdigest()[:16], 16) % 2 ** 63
        assert comp.comp_seed(rid) == expected and 0 <= expected < 2 ** 63
    pilot = int(hashlib.sha256(b'COMP-RS-001-S0-PILOT-SEED|MC2-a').hexdigest()[:16], 16) % 2 ** 63
    assert comp.comp_seed('MC2-a') != pilot
    with pytest.raises(ValueError):
        comp.comp_seed('')


def test_calibration_plan_reproduces_the_frozen_scope_receipt(cal):
    receipt = json.loads(comp.CALIBRATION_SCOPE.read_bytes())
    assert sha(comp.CALIBRATION_SCOPE) == REGISTERED_SCOPE_SHA == comp.CALIBRATION_SCOPE_SHA
    plan = comp.calibration_plan(cal['rows_by_skill'])
    assert len(plan) == 450 and comp.is_full_plan(plan, cal['rows_by_skill'])
    assert [row['skills'][0] for row in plan] == ['R'] * 150 + ['S'] * 150 + ['P'] * 150
    for skill in comp.SKILLS:
        rows = [row for row in plan if row['skills'] == [skill]]
        ids = [row['id'] for row in rows]
        assert ids == sorted(ids) and all(row['tag'] == 'core' and row['role'] == 'calibration' for row in rows)
        assert digest(ids) == receipt['signatures'][skill]['ids_sha256']
        assert digest(sorted({row['parent_id'] for row in rows})) == receipt['signatures'][skill]['parents_sha256']
    assert cal['record']['binding']['signatures'] == {
        s: {'binding': 'full', 'rows': 150, 'rows_in_receipt': 150, 'receipt_rows': 150} for s in 'PRS'}
    assert not comp.is_full_plan(plan[:-1], cal['rows_by_skill'])
    assert not comp.is_full_plan(plan[150:300] + plan[:150] + plan[300:], cal['rows_by_skill'])


def test_load_calibration_refuses_a_drifted_scope_receipt(comp_rs_data, tmp_path):
    receipt = json.loads(comp.CALIBRATION_SCOPE.read_bytes())
    copy = tmp_path / 'copy.json'
    _create(copy, json.dumps(receipt, indent=1, sort_keys=True).encode('ascii'))   # same content, other bytes
    with pytest.raises(ValueError, match='file sha differs'):
        comp.load_calibration(comp_rs_data, copy)             # pinned sha: any other bytes are refused
    assert comp.load_calibration(comp_rs_data, copy, expected_file_sha256=None)['record']['scope_receipt_digest'] == \
        digest(receipt)
    for skill, field, value in (('S', 'ids_sha256', '0' * 64), ('P', 'parents_sha256', '0' * 64),
                                ('R', 'rows', 149)):
        tampered = json.loads(json.dumps(receipt))
        tampered['signatures'][skill][field] = value
        path = _write_json(tmp_path / ('%s_%s.json' % (skill, field)), tampered)
        with pytest.raises(ValueError, match='core ' + skill):
            comp.load_calibration(comp_rs_data, path, expected_file_sha256=None)
    other = dict(receipt, canonical_calibration_sha256='1' * 64)
    with pytest.raises(ValueError, match='another CALIBRATION bank'):
        comp.load_calibration(comp_rs_data, _write_json(tmp_path / 'bank.json', other), expected_file_sha256=None)


def test_check_policy_binds_messages_and_prompt_hash(comp_rs_data, cal):
    rows = cal_subset(cal['rows_by_skill'])
    record = comp.check_policy(comp_rs_data, 'calibration', rows)
    assert record['rows_checked'] == 3 and record['file'] == 'policy/calibration.jsonl'
    edited = dict(rows[0], prompt=[dict(rows[0]['prompt'][0]), dict(rows[0]['prompt'][1], content='changed')])
    with pytest.raises(ValueError, match='policy messages differ'):
        comp.check_policy(comp_rs_data, 'calibration', [edited])
    with pytest.raises(ValueError, match='prompt_hash mismatch'):
        comp.check_policy(comp_rs_data, 'calibration', [dict(rows[1], prompt_hash='0' * 64)])


# ------------------------------------------------------------------------------------------------ responses
def test_response_rule_strips_only_the_terminal_eos(tokenizer, cal, oracles):
    row = cal['rows_by_skill']['R'][0]
    ids = ref_ids(tokenizer, oracles, row)
    reference = oracles['calibration'][row['id']]['references'][0]
    for eos in s0.EOS_IDS:
        text = comp.response_text(tokenizer, ids + [eos], 'eos')
        assert text == {'response': reference, 'body_special_ids': [], 'body_unknown_ids': [], 'lenient_response': None}
        assert corpus.reward(row, text['response']) == 1
    assert comp.response_text(tokenizer, ids, 'length')['response'] == reference
    # a special token inside the body keeps its text: never stripped into a success
    special = comp.response_text(tokenizer, ids + [IM_START, IM_END], 'eos')
    assert special['response'] == reference + '<|im_start|>' and special['body_special_ids'] == [IM_START]
    assert special['lenient_response'] == reference
    think = comp.response_text(tokenizer, ids + [THINK, IM_END], 'eos')   # not special: both rules keep it
    assert think == {'response': reference + '<think>', 'body_special_ids': [], 'body_unknown_ids': [],
                     'lenient_response': None}
    assert corpus.reward(row, special['response']) == 0 and corpus.reward(row, special['lenient_response']) == 1
    # an unrenderable ID (decode drops it silently) becomes a visible placeholder
    plain = {'skip_special_tokens': False, 'clean_up_tokenization_spaces': False}
    assert tokenizer.decode(ids + [UNRENDERABLE_ID], **plain) == reference
    hole = comp.response_text(tokenizer, ids[:2] + [UNRENDERABLE_ID] + ids[2:] + [IM_END], 'eos')
    assert hole['body_unknown_ids'] == [UNRENDERABLE_ID]
    assert hole['response'] == (tokenizer.decode(ids[:2], **plain) + '<|unrenderable:%d|>' % UNRENDERABLE_ID
                                + tokenizer.decode(ids[2:], **plain))
    assert hole['lenient_response'] == reference and corpus.reward(row, hole['response']) == 0
    assert comp.response_text(tokenizer, [IM_END], 'eos')['response'] == ''
    for bad_ids, reason in ((ids, 'eos'), (ids[:1] + [IM_END] + ids[1:] + [IM_END], 'eos'), (ids + [IM_END], 'length'),
                            (ids, 'stop'), ([], 'length'), ([-1], 'length')):
        with pytest.raises(ValueError):
            comp.response_text(tokenizer, bad_ids, reason)


def test_score_completion_is_the_scorer_strict_path(tokenizer, cal, oracles):
    row = cal['rows_by_skill']['S'][0]
    ids = ref_ids(tokenizer, oracles, row)
    good = comp.response_text(tokenizer, ids + [IM_END], 'eos')
    scored = comp.score_completion(row, completion(ids + [IM_END]), good)
    assert scored == {'reward': 1, 'content_pass': True, 'normalization': 'answer_object',
                      'format_only_failure': False, 'subtype': None, 'unknown_answer': False, 'lenient_reward': None}
    special = comp.response_text(tokenizer, ids + [IM_START, IM_END], 'eos')
    scored = comp.score_completion(row, completion(ids + [IM_START, IM_END]), special)
    assert scored['reward'] == 0 and scored['lenient_reward'] == 1 and scored['subtype'] == 'PROSE_WRAPPED'
    unknown_ids = tokenizer('{"answer":"UNKNOWN"}', add_special_tokens=False)['input_ids'] + [IM_END]
    unknown = comp.score_completion(row, completion(unknown_ids),
                                    comp.response_text(tokenizer, unknown_ids, 'eos'))
    assert unknown['reward'] == 0 and unknown['unknown_answer'] is True


def test_unknown_answer_uses_the_content_contract():
    assert comp.unknown_answer('{"answer":"UNKNOWN"}') and comp.unknown_answer('UNKNOWN')
    assert comp.unknown_answer('```json\n{"answer":"UNKNOWN"}\n```')
    for text in ('{"answer":[1]}', '{"answer":"unknown"}', 'garbage', '', '{"answer":"UNKNOWN","alt":[1]}'):
        assert not comp.unknown_answer(text)


# ------------------------------------------------------------------------------------------------ rules
def test_no_signal_rule_on_constructed_counts():
    def rule(r, s, p, n=150):
        return comp.no_signal_rule({'R': {'mixed': r, 'prompts': n}, 'S': {'mixed': s, 'prompts': n},
                                    'P': {'mixed': p, 'prompts': n}})
    assert rule(15, 15, 0)['outcome'] == 'SIGNAL'                   # exactly m_min passes; P is not gated
    assert rule(14, 150, 150)['outcome'] == 'STOP_NO_SIGNAL' and rule(14, 150, 150)['below_m_min'] == ['R']
    assert rule(150, 14, 150)['below_m_min'] == ['S']
    assert rule(0, 0, 0)['below_m_min'] == ['R', 'S']
    result = rule(20, 30, 1)
    assert result['skills']['P'] == {'mixed': 1, 'prompts': 150, 'mixed_fraction': 1 / 150, 'gated': False,
                                     'below_m_min': True}
    assert result['m_min'] == 0.1 and result['delta_acq_rows'] == 13 and result['recorded_only'] == ['P']
    assert result['s1_first_skill'] == 'R' and rule(40, 30, 0)['s1_first_skill'] == 'S'
    assert rule(30, 30, 0)['s1_first_skill'] == 'R'                  # tie: R
    two = {'R': {'mixed': 1, 'prompts': 10}, 'S': {'mixed': 1, 'prompts': 10}}
    assert comp.no_signal_rule(two)['outcome'] == 'SIGNAL'
    for bad in ({'R': {'mixed': 1, 'prompts': 10}},
                {'R': {'mixed': 11, 'prompts': 10}, 'S': {'mixed': 1, 'prompts': 10}},
                {'R': {'mixed': 1, 'prompts': 0}, 'S': {'mixed': 0, 'prompts': 1}},
                {'R': {'mixed': 1.0, 'prompts': 10}, 'S': {'mixed': 1, 'prompts': 10}},
                {'R': {'mixed': 1, 'prompts': 10}, 'S': {'mixed': 1, 'prompts': 10}, 'RS': {'mixed': 1, 'prompts': 1}}):
        with pytest.raises(ValueError):
            comp.no_signal_rule(bad)


def _synthetic_row(skill, rewards, *, stop='eos', subtype=None, unknown=False, lenient=None, tokens=5):
    completions = [{'reward': r, 'content_pass': bool(r), 'normalization': 'answer_object',
                    'format_only_failure': False, 'subtype': subtype if not r else None, 'unknown_answer': unknown,
                    'stop_reason': stop, 'generated_tokens': tokens, 'body_special_ids': [], 'body_unknown_ids': [],
                    'lenient_response': lenient, 'lenient_reward': None if lenient is None else 1}
                   for r in rewards]
    return {'skill': skill, 'completions': completions, 'rewards': list(rewards), 'decode_steps': tokens,
            'prompt_tokens': 100, 'wall_s': 0.5}


def test_summarise_pass_at_k_groups_and_diagnostics():
    rows = [_synthetic_row('R', [1, 0, 0, 0]), _synthetic_row('R', [0, 0, 0, 0], stop='length', subtype='LENGTH_CAP',
                                                               tokens=128),
            _synthetic_row('R', [1, 1, 1, 1]), _synthetic_row('S', [0, 1, 0, 1], unknown=True, lenient='x')]
    summary = comp.summarise(rows, 4)
    r = summary['R']
    assert (r['prompts'], r['samples'], r['successes']) == (3, 12, 5)
    assert r['pass_at_1'] == 5 / 12 and r['pass_at_G'] == 2 / 3
    assert r['groups'] == {'all_fail': 1, 'mixed': 1, 'all_success': 1}
    assert r['success_histogram'] == {'0': 1, '1': 1, '2': 0, '3': 0, '4': 1}
    assert r['diagnostics']['stop_reason'] == {'eos': 8, 'length': 4} and r['diagnostics']['cutoff_length_cap'] == 4
    assert r['tokens'] == {'generated': 4 * 5 + 4 * 128 + 4 * 5, 'generated_mean': (40 + 512) / 12,
                           'generated_max': 128, 'decode_steps': 138, 'prompt_tokens': 300, 'prefill_tokens': 1200}
    s = summary['S']
    assert s['diagnostics']['unknown_answer'] == 4 and s['diagnostics']['lenient_reward_differs'] == 2
    assert s['pass_at_G'] == maxrl.pass_at_k(4, 2, 4) == 1.0 and 'P' not in summary
    with pytest.raises(ValueError):
        comp.summarise([_synthetic_row('R', [1, 0, 1])], 4)
    assert comp.group_outcome([0, 0]) == 'all_fail' and comp.group_outcome([1, 0]) == 'mixed'
    assert comp.group_outcome([1, 1]) == 'all_success'


def test_distinct_decoders_ignore_only_max_length():
    record = {'mode': 'sample', 'effective_config': {'max_new_tokens': 128, 'max_length': 228, 'top_k': 0}}
    rows = [{'prompt_tokens': 100, 'sampler': record},
            {'prompt_tokens': 50, 'sampler': dict(record, effective_config=dict(record['effective_config'],
                                                                                max_length=178))}]
    result = comp.distinct_decoders(rows, 'sampler', 128)
    assert result['rows'] == 2 and 'max_length' not in result['record']['effective_config']
    with pytest.raises(ValueError, match='max_length'):
        comp.distinct_decoders([dict(rows[0], prompt_tokens=99)], 'sampler', 128)
    other = dict(rows[1], sampler=dict(rows[1]['sampler'], effective_config=dict(rows[1]['sampler']['effective_config'],
                                                                                  top_k=50)))
    with pytest.raises(ValueError, match='more than one decoder'):
        comp.distinct_decoders([rows[0], other], 'sampler', 128)


# ------------------------------------------------------------------------------------------------ freeze
def _cpu_freeze(tmp_path, name='freeze.json', **overrides):
    s0.configure('cpu')
    preimage = s0.decoding_preimage('cpu')
    value = {'stage': 'S0_RUNTIME_FREEZE', 'decoding_preimage': preimage,
             'decoding_sha256': s0.decoding_sha256(preimage), 'host_floor_gib': 0.75,
             'scoring_code_bundle_sha256': comp.live_bundle(), 'inventory': {'outcome': 'ADOPT_A'}}
    value.update(overrides)
    return _write_json(tmp_path / name, value), value


def test_read_freeze_schema(tmp_path):
    path, value = _cpu_freeze(tmp_path)
    freeze = comp.read_freeze(path, registered=False)
    assert freeze['decoding_sha256'] == value['decoding_sha256'] and freeze['host_floor_gib'] == 0.75
    assert freeze['sha256'] == sha(path) and freeze['scoring_code_bundle_sha256'] == comp.live_bundle()
    with pytest.raises(ValueError, match='not for cuda'):
        comp.read_freeze(path, registered=True)
    cases = {'stage': ({'stage': 'S0_PILOT_A'}, 'not an S0_RUNTIME_FREEZE'),
             'sha': ({'decoding_sha256': '0' * 64}, 'not the digest'),
             'floor': ({'host_floor_gib': 0}, 'host floor'), 'bool': ({'host_floor_gib': True}, 'host floor'),
             'bundle': ({'scoring_code_bundle_sha256': 'ABC'}, 'bundle'),
             'test': ({'test_mode': True, 'decoding_preimage': dict(value['decoding_preimage'], device='cuda:0')},
                      None)}
    for name, (change, message) in cases.items():
        edited = dict(value, **change)
        if 'decoding_preimage' in change:
            edited['decoding_sha256'] = s0.decoding_sha256(edited['decoding_preimage'])
        target = _write_json(tmp_path / (name + '.json'), edited)
        if message is None:
            with pytest.raises(ValueError, match='not a registered run'):
                comp.read_freeze(target, registered=True)
            comp.read_freeze(target, registered=False)
        else:
            with pytest.raises(ValueError, match=message):
                comp.read_freeze(target, registered=False)
    missing = {k: v for k, v in value.items() if k != 'inventory'}
    with pytest.raises(ValueError, match='lacks: inventory'):
        comp.read_freeze(_write_json(tmp_path / 'missing.json', missing), registered=False)


def test_freeze_sets_the_floor_and_bundle_and_must_match_the_live_runtime(tmp_path, tiny_dir, monkeypatch):
    path, _ = _cpu_freeze(tmp_path)
    args = comp._parser().parse_args(common_argv(tiny_dir, tmp_path / 'a' / 'S0_COMPETENCE.json', '--runtime-freeze',
                                                 str(path), floor=False))
    host = FakeHost()
    cfg = comp._competence_config(args, None, host)
    prepared = comp.prepare_launch(args, cfg, host)
    assert host.calls == [('require', 0.75, comp.LABEL)]
    assert prepared['launch']['floor']['source'].startswith('S0_RUNTIME')
    assert prepared['bundle'] == {'expected': comp.live_bundle(), 'source': 'S0_RUNTIME_FREEZE',
                                  'live': comp.live_bundle()}
    stale, _ = _cpu_freeze(tmp_path, 'stale.json', scoring_code_bundle_sha256='0' * 64)
    args = comp._parser().parse_args(common_argv(tiny_dir, tmp_path / 'b' / 'S0_COMPETENCE.json', '--runtime-freeze',
                                                 str(stale)))
    with pytest.raises(ValueError, match='bundle differs from the runtime freeze'):
        comp.prepare_launch(args, comp._competence_config(args, None, FakeHost()), FakeHost())
    # a freeze made on another runtime: refused after configure(), before the model is ever loaded
    other_preimage = dict(json.loads(path.read_bytes())['decoding_preimage'], num_threads=8)
    other, _ = _cpu_freeze(tmp_path, 'other.json', decoding_preimage=other_preimage,
                           decoding_sha256=s0.decoding_sha256(other_preimage))

    def never(*a, **k):
        raise AssertionError('model loaded')
    monkeypatch.setattr(s0, 'load_model_fp32', never)
    monkeypatch.setattr(comp, 'calibration_plan', cal_subset)
    with pytest.raises(ValueError, match='decoding_sha256 differs from the runtime freeze'):
        comp.main(common_argv(tiny_dir, tmp_path / 'c' / 'S0_COMPETENCE.json', '--runtime-freeze', str(other),
                              '--group-size', '2'), lease_module=FakeLease(), host_module=FakeHost())
    assert not (tmp_path / 'c' / 'S0_COMPETENCE.jsonl').exists()


# ------------------------------------------------------------------------------------------------ scoring
def _journal_like(tokenizer, oracles, rows, group_size):
    """Complete competence rows whose sample 0 is the oracle reference and the rest are wrong."""
    out = []
    for index, row in enumerate(rows):
        ids = ref_ids(tokenizer, oracles, row) + [IM_END]
        wrong = tokenizer('{"answer":[]}', add_special_tokens=False)['input_ids'] + [IM_END]
        completions = []
        for k in range(group_size):
            chosen = ids if k == 0 else wrong
            text = comp.response_text(tokenizer, chosen, 'eos')
            completions.append(dict(completion(chosen), k=k, response=text['response'], lenient_response=None,
                                    body_special_ids=[], body_unknown_ids=[],
                                    **comp.score_completion(row, completion(chosen), text)))
        out.append({'id': row['id'], 'index': index, 'skill': row['skills'][0], 'prompt_hash': row['prompt_hash'],
                    'completions': completions, 'rewards': [c['reward'] for c in completions],
                    'decode_steps': len(ids), 'prompt_tokens': 10, 'wall_s': 0.1})
    return out


def test_score_competence_rescores_every_sample_through_the_canary(comp_rs_data, cal, sections, tokenizer, oracles):
    rows = cal_subset(cal['rows_by_skill'])
    journal = _journal_like(tokenizer, oracles, rows, 2)
    meta = {'model_sha256': 'a' * 64, 'tokenizer_sha256': 'b' * 64, 'decoding_sha256': 'c' * 64}
    prompts = [{'id': row['id']} for row in rows]
    result = comp.score_competence(comp_rs_data, cal, prompts, journal, meta, comp.live_bundle(), 2, sections=sections)
    assert [b['strict_correct'] for b in result['batches']] == [3, 0] and result['canary_receipts_identical']
    assert result['canary_receipt']['overall'] == 'PASS' and result['code_bundles'] == [comp.live_bundle()]
    assert {b['scope_binding']['signatures']['R']['binding'] for b in result['batches']} == {'subset'}
    tampered = json.loads(json.dumps(journal))
    tampered[1]['completions'][1]['reward'] = 1
    with pytest.raises(ValueError, match='differs from the journal: %s sample 1' % rows[1]['id']):
        comp.score_competence(comp_rs_data, cal, prompts, tampered, meta, comp.live_bundle(), 2, sections=sections)
    tampered = json.loads(json.dumps(journal))
    tampered[0]['completions'][0]['subtype'] = 'OTHER'
    with pytest.raises(ValueError, match='differs from the journal'):
        comp.score_competence(comp_rs_data, cal, prompts, tampered, meta, comp.live_bundle(), 2, sections=sections)
    with pytest.raises(canary.CanaryFailed):
        comp.score_competence(comp_rs_data, cal, prompts, journal, meta, '0' * 64, 2, sections=sections)
    with pytest.raises(ValueError, match='differ from the plan'):
        comp.score_competence(comp_rs_data, cal, prompts[::-1], journal, meta, comp.live_bundle(), 2, sections=sections)


def _enroll_rows(tokenizer, oracles, rows, bank, wrong_ids=()):
    out = []
    for row in rows:
        reference = oracles[row['role']][row['id']]['references'][0]
        response = '{"answer":"no"}' if row['id'] in wrong_ids else reference
        out.append({'id': row['id'], 'bank': bank, 'group': enroll._row_group(row), 'prompt_hash': row['prompt_hash'],
                    'response': response, 'generated_tokens': 7, 'stop_reason': 'eos', 'lenient_response': None,
                    'wall_s': 0.01, 'prompt_tokens': 100})
    return out


def test_full_guard_dev_enrollment_is_behaviour_enroll_and_matches_the_scoped_scorer(
        comp_rs_data, cal, guard_bank, sections, tokenizer, oracles, monkeypatch):
    cal_rows = cal_subset(cal['rows_by_skill'])
    wrong = guard_bank[5]['id']
    rows = (_enroll_rows(tokenizer, oracles, list(guard_bank), 'guard_dev', {wrong})
            + _enroll_rows(tokenizer, oracles, cal_rows, 'calibration'))
    meta = {'model_sha256': 'a' * 64, 'tokenizer_sha256': 'b' * 64, 'decoding_sha256': 'c' * 64}
    seen = []
    real_enroll = behaviour.enroll

    def spy(data, role, responses, metadata):
        seen.append((role, responses, metadata))
        return real_enroll(data, role, responses, metadata)
    monkeypatch.setattr(behaviour, 'enroll', spy)
    result = enroll.score_enrollment(comp_rs_data, cal, guard_bank, [{'id': r['id']} for r in rows], rows, meta,
                                     comp.live_bundle(), full_guard=True, sections=sections)
    ((role, responses, metadata),) = seen
    assert role == 'guard_dev' and metadata == meta and len(responses) == 1200
    assert [r['id'] for r in responses] == [r['id'] for r in guard_bank]
    assert all(set(r) == {'id', 'prompt_hash', 'response', 'generated_tokens'} for r in responses)
    enrollment = result['guard_dev_enrollment']
    assert enrollment['kind'] == 'source_correct_enrollment' and enrollment['source_correct_count'] == 1199
    assert wrong not in enrollment['source_correct_ids'] and enrollment['run'] == meta
    assert enrollment['bank_sha256'] == guard_bank.bank_sha256
    assert result['guard_dev_enrollment_status'] == 'ENROLLED_FULL_BANK'
    summary = result['summary']
    assert summary['guard_dev']['source_correct'] == 1199 and summary['calibration']['strict_correct'] == 3
    assert sum(f['rows'] for f in summary['guard_dev']['by_family'].values()) == 1200
    assert len(summary['guard_dev']['by_family']) == 8
    assert summary['calibration']['by_skill'] == {s: {'rows': 1, 'strict_correct': 1, 'unknown_answer': 0}
                                                  for s in comp.SKILLS}
    assert result['calibration_scoring']['scoring']['scope_binding']['kind'] == 'scope_receipt'
    assert result['guard_dev_scoring']['scoring']['scope_binding']['kind'] == 'guard_bank_sha256'
    # the section 8.0 cross-check: a scorer that disagrees with behaviour.enroll is refused
    monkeypatch.setattr(behaviour, 'enroll', lambda *a: dict(real_enroll(*a), row_scores={}))
    with pytest.raises(ValueError, match='disagree'):
        enroll.score_enrollment(comp_rs_data, cal, guard_bank, [{'id': r['id']} for r in rows], rows, meta,
                                comp.live_bundle(), full_guard=True, sections=sections)
    with pytest.raises(ValueError, match='full guard_dev bank'):
        enroll.enrollment_responses(guard_bank, rows[:1199])


# ------------------------------------------------------------------------------------------------ S0-C CLI
def _competence_plan(tokenizer, oracles, rows):
    r, s, p = rows
    return {r['id']: {0: ref_ids(tokenizer, oracles, r) + [IM_END]},
            s['id']: {0: ref_ids(tokenizer, oracles, s) + [IM_END],
                      1: ref_ids(tokenizer, oracles, s) + [IM_START, IM_END]},
            p['id']: {0: ref_ids(tokenizer, oracles, p) + [ENDOFTEXT],
                      1: tokenizer('{"answer":"UNKNOWN"}', add_special_tokens=False)['input_ids'] + [IM_END]}}


def _run_competence(root, tiny_dir, tokenizer, cal, oracles, injector=None, host=None, lease=None, extra=()):
    rows = cal_subset(cal['rows_by_skill'])
    injector = injector or Injector(tokenizer, rows, _competence_plan(tokenizer, oracles, rows))
    host, lease = host or FakeHost(), lease or FakeLease()
    out = root / 'S0_COMPETENCE.json'
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(comp, 'calibration_plan', cal_subset)
        mp.setattr(g, 'sample_group', injector.sample_group)
        recording_writes(mp, lease)
        code = comp.main(common_argv(tiny_dir, out, '--group-size', '2', *extra), lease_module=lease, host_module=host)
    return SimpleNamespace(code=code, out=out, journal=out.with_suffix('.jsonl'), host=host, lease=lease, rows=rows,
                           injector=injector)


@pytest.fixture(scope='module')
def competence_run(tmp_path_factory, tiny_dir, tokenizer, cal, oracles):
    run = _run_competence(tmp_path_factory.mktemp('competence') / 'records', tiny_dir, tokenizer, cal, oracles)
    run.receipt = _receipt(run.out)
    return run


def test_competence_cli_end_to_end(competence_run, tiny_dir, cal):
    run, receipt = competence_run, competence_run.receipt
    assert run.code == 0 and receipt['outcome'] == 'SIGNAL' and receipt['stage'] == 'S0_COMPETENCE'
    assert receipt['test_mode'] is True and receipt['registered_run'] is False and receipt['group_size'] == 2
    lines = _lines(run.journal)
    assert [line['id'] for line in lines] == [row['id'] for row in run.rows]
    for line, row in zip(lines, run.rows):
        assert line['seed'] == comp.comp_seed(row['id']) and line['skill'] == row['skills'][0]
        assert line['rewards'] == [1, 0] and line['outcome'] == 'mixed' and len(line['completions']) == 2
        assert all(line[k] == v for k, v in receipt['journal']['bind'].items())
        assert line['row_sha256'] == digest({k: v for k, v in line.items() if k != 'row_sha256'})
        sampled = line['completions'][1] if line['skill'] == 'R' else None
        if sampled is not None:                          # a genuinely sampled completion of the tiny model
            assert sampled['stop_reason'] == 'length' and sampled['generated_tokens'] == 128
            assert len(sampled['completion_ids']) == len(sampled['raw_step_logprobs']) == 128
            assert sampled['subtype'] == 'LENGTH_CAP' and sampled['reward'] == 0
    bind = receipt['journal']['bind']
    assert bind['weights_sha256'] == tiny_dir.weights_sha256 and bind['decoding_sha256'] == receipt['decoding_sha256']
    assert bind['plan_sha256'] == digest([row['id'] for row in run.rows]) and bind['group_size'] == 2
    assert bind['code_sha256'] == digest(comp._IMPORT_CODE) and bind['test_mode'] is True
    assert receipt['decoding_sha256'] == s0.decoding_sha256(receipt['decoding_preimage'])
    assert receipt['run_meta'] == {'model_sha256': tiny_dir.weights_sha256,
                                   'tokenizer_sha256': s0.TOKENIZER_RECEIPT_SHA,
                                   'decoding_sha256': receipt['decoding_sha256']}
    summary = receipt['summary']
    for skill in comp.SKILLS:
        entry = summary[skill]
        assert (entry['prompts'], entry['samples'], entry['pass_at_1'], entry['pass_at_G']) == (1, 2, 0.5, 1.0)
        assert entry['groups'] == {'all_fail': 0, 'mixed': 1, 'all_success': 0}
    assert summary['S']['diagnostics']['lenient_reward_differs'] == 1
    assert summary['S']['diagnostics']['body_special'] == 1
    assert summary['P']['diagnostics']['unknown_answer'] == 1 and summary['R']['diagnostics']['cutoff_length_cap'] == 1
    assert summary['P']['diagnostics']['stop_reason'] == {'eos': 2, 'length': 0}
    rule = receipt['rule']
    assert rule['skills']['R']['mixed_fraction'] == 1.0 and rule['below_m_min'] == []
    assert receipt['delta_acq_rows'] == 13
    scoring = receipt['scoring']
    assert [b['strict_correct'] for b in scoring['batches']] == [3, 0] and scoring['agrees_with_journal']
    assert scoring['canary_receipt']['overall'] == 'PASS' and scoring['code_bundles'] == [comp.live_bundle()]
    assert receipt['scorer_bundle']['source'].startswith('live')
    assert receipt['inputs']['full_plan'] is False and receipt['inputs']['plan_by_skill'] == {'R': 1, 'S': 1, 'P': 1}
    assert receipt['inputs']['scope_receipt_file_sha256'] == REGISTERED_SCOPE_SHA
    assert receipt['sampler']['record']['effective_config']['top_k'] == 0 and receipt['sampler']['rows'] == 3
    # scoring and the receipt happen while the lease is still held (section 16: the whole run)
    assert run.lease.events == [('acquire', comp.LABEL), ('write', 'S0_COMPETENCE.json'), ('release', 'token-1')]
    assert run.host.calls == [('require', 0.5, comp.LABEL)] + [('guard', 1.0, comp.LABEL + ' ' + row['id'])
                                                             for row in run.rows]
    assert receipt['attempts']['launches'] == 1 and receipt['attempts']['interrupted_sessions'] == []
    assert receipt['fingerprint_unchanged'] is True and receipt['weights_saved'] is False
    assert receipt['journal']['sha256'] == sha(run.journal)
    with pytest.raises(SystemExit, match='create-only'):
        comp.main(common_argv(tiny_dir, run.out, '--group-size', '2'), lease_module=FakeLease(), host_module=FakeHost())


def test_competence_resumes_after_a_crash_and_a_torn_line(tmp_path, competence_run, tiny_dir, tokenizer, cal, oracles):
    root = tmp_path / 'records'
    rows = cal_subset(cal['rows_by_skill'])
    plan = _competence_plan(tokenizer, oracles, rows)
    first = Injector(tokenizer, rows, plan, fail_at=3)   # C13: 2 sample calls per group, so call 3 = group 2
    with pytest.raises(RuntimeError, match='simulated crash'):
        _run_competence(root, tiny_dir, tokenizer, cal, oracles, injector=first)
    journal = root / 'S0_COMPETENCE.jsonl'
    kept = journal.read_bytes()
    assert len(_lines(journal)) == 1 and not (root / 'S0_COMPETENCE.json').exists()
    with open(journal, 'ab') as stream:
        stream.write(b'{"id":"torn-by-a-crash')
    second = _run_competence(root, tiny_dir, tokenizer, cal, oracles, injector=Injector(tokenizer, rows, plan))
    assert second.code == 0 and second.injector.calls == 4   # C13: 2 groups x 2 single-sample calls
    assert journal.read_bytes().startswith(kept)
    receipt = _receipt(second.out)
    assert receipt['this_launch'] == {'resume_index': 1, 'appended': 2}
    attempts = receipt['attempts']
    assert attempts['launches'] == 2 and attempts['resumed'] and len(attempts['interrupted_sessions']) == 1
    crashed = attempts['sessions'][attempts['interrupted_sessions'][0]]
    assert crashed['end']['outcome'] == 'error' and crashed['end']['error']['message'] == 'simulated crash'
    assert crashed['end']['rows_done'] == 1
    resumed = attempts['sessions'][receipt['session']]
    assert resumed['start']['recovered']['event'] == 'TRUNCATED_PARTIAL_LAST_LINE'
    assert (root / 'S0_COMPETENCE.jsonl.recovery.jsonl').is_file()
    # the resumed read equals the uninterrupted one, token for token
    assert [[c['completion_ids'] for c in line['completions']] for line in _lines(journal)] == [
        [c['completion_ids'] for c in line['completions']] for line in _lines(competence_run.journal)]

    def no_walls(summary):
        return {skill: {k: v for k, v in entry.items() if k != 'wall_s'} for skill, entry in summary.items()}
    assert no_walls(receipt['summary']) == no_walls(competence_run.receipt['summary'])
    assert receipt['rule'] == competence_run.receipt['rule']


def test_a_model_that_changed_during_the_read_is_refused(tmp_path, tiny_dir, tokenizer, cal, oracles):
    rows = cal_subset(cal['rows_by_skill'])
    injector = Injector(tokenizer, rows, {})
    real = injector.sample_group

    def mutating(model, *args, **kwargs):
        with torch.no_grad():
            model.model.norm.weight.add_(1.0)
        return real(model, *args, **kwargs)
    injector.sample_group = mutating
    with pytest.raises(ValueError, match='the model state changed during the read'):
        _run_competence(tmp_path / 'records', tiny_dir, tokenizer, cal, oracles, injector=injector)
    assert not (tmp_path / 'records' / 'S0_COMPETENCE.json').exists()


def test_competence_refuses_a_journal_with_another_bind(tmp_path, tiny_dir, tokenizer, cal, oracles):
    root = tmp_path / 'records'
    rows = cal_subset(cal['rows_by_skill'])
    with pytest.raises(RuntimeError):
        _run_competence(root, tiny_dir, tokenizer, cal, oracles,
                        injector=Injector(tokenizer, rows, _competence_plan(tokenizer, oracles, rows), fail_at=3))
    with pytest.MonkeyPatch.context() as mp:              # a resume under a different group size is another read
        mp.setattr(comp, 'calibration_plan', cal_subset)
        with pytest.raises(ValueError, match='bind field group_size differs'):
            comp.main(common_argv(tiny_dir, root / 'S0_COMPETENCE.json', '--group-size', '3'),
                      lease_module=FakeLease(), host_module=FakeHost())
        mp.setattr(comp, 'calibration_plan', lambda by: cal_subset(by)[::-1])
        with pytest.raises(ValueError, match='bind field plan_sha256 differs'):
            comp.main(common_argv(tiny_dir, root / 'S0_COMPETENCE.json', '--group-size', '2'),
                      lease_module=FakeLease(), host_module=FakeHost())


# ------------------------------------------------------------------------------------------------ S0-E CLI
def _enroll_plan(tokenizer, oracles, guard_rows, cal_rows):
    return {guard_rows[0]['id']: ref_ids(tokenizer, oracles, guard_rows[0]) + [IM_END],
            cal_rows[0]['id']: ref_ids(tokenizer, oracles, cal_rows[0]) + [ENDOFTEXT]}


def _run_enroll(root, run, tiny_dir, injector, host=None, lease=None, name=None, extra=()):
    host, lease = host or FakeHost(), lease or FakeLease()
    out = root / (name or 'S0_ENROLL_RUN%d.json' % run)
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(comp, 'calibration_plan', cal_subset)
        mp.setattr(enroll, 'guard_dev_plan', guard_subset)
        mp.setattr(g, 'greedy_decode', injector.greedy_decode)
        recording_writes(mp, lease)
        code = enroll.main(common_argv(tiny_dir, out, '--run', str(run), *extra), lease_module=lease,
                           host_module=host)
    return SimpleNamespace(code=code, out=out, journal=out.with_suffix('.jsonl'), host=host, lease=lease,
                           injector=injector)


@pytest.fixture(scope='module')
def enroll_runs(tmp_path_factory, tiny_dir, tokenizer, cal, guard_bank, oracles):
    guard_rows, cal_rows = guard_subset(guard_bank), cal_subset(cal['rows_by_skill'])
    rows = guard_rows + cal_rows
    plan = _enroll_plan(tokenizer, oracles, guard_rows, cal_rows)
    root = tmp_path_factory.mktemp('enroll') / 'records'
    aborted = _run_enroll(root, 1, tiny_dir, Injector(tokenizer, rows, plan), host=FakeHost(abort_at=3))
    after_abort = _lines(aborted.journal)
    with pytest.raises(KeyboardInterrupt):
        _run_enroll(root, 1, tiny_dir, Injector(tokenizer, rows, plan, fail_at=2, error=KeyboardInterrupt))
    after_interrupt = _lines(root / 'S0_ENROLL_RUN1.jsonl')
    first = _run_enroll(root, 1, tiny_dir, Injector(tokenizer, rows, plan))
    second = _run_enroll(root, 2, tiny_dir, Injector(tokenizer, rows, plan))
    return SimpleNamespace(root=root, aborted=aborted, after_abort=after_abort, after_interrupt=after_interrupt,
                           first=first, second=second, rows=rows, guard_rows=guard_rows, cal_rows=cal_rows, plan=plan,
                           receipts={1: _receipt(first.out), 2: _receipt(second.out)})


def test_enroll_run_resumes_after_a_host_abort_and_an_interrupt(enroll_runs, tiny_dir):
    runs = enroll_runs
    assert runs.aborted.code == 3 and len(runs.after_abort) == 2 and len(runs.after_interrupt) == 3
    assert runs.aborted.host.calls[-1] == ('guard', 1.0, 'comp-rs-001-s0-enroll-run1 ' + runs.rows[2]['id'])
    lines = _lines(runs.first.journal)
    assert lines[:3] == runs.after_interrupt and [line['id'] for line in lines] == [row['id'] for row in runs.rows]
    receipt = runs.receipts[1]
    assert runs.first.code == 0 and receipt['stage'] == 'S0_ENROLL_RUN1' and receipt['enrollment_of_record'] is True
    assert receipt['this_launch'] == {'resume_index': 3, 'appended': 3} and receipt['completed'] is True
    attempts = receipt['attempts']
    assert attempts['launches'] == 3 and len(attempts['interrupted_sessions']) == 2
    outcomes = sorted(entry['end']['outcome'] for entry in attempts['sessions'].values())
    assert outcomes == ['aborted', 'complete', 'error']
    aborted = next(e for e in attempts['sessions'].values() if e['end']['outcome'] == 'aborted')
    assert aborted['end']['abort']['type'] == 'LowMemoryAbort' and aborted['end']['rows_done'] == 2
    errored = next(e for e in attempts['sessions'].values() if e['end']['outcome'] == 'error')
    assert errored['end']['error']['type'] == 'KeyboardInterrupt' and errored['start']['resume_index'] == 2
    assert receipt['journal']['bind']['run'] == 1 and receipt['journal']['bind']['stage'] == 'S0_ENROLL_RUN1'


def test_enroll_receipt_scores_and_enrolls(enroll_runs, tiny_dir, guard_bank):
    runs, receipt = enroll_runs, enroll_runs.receipts[1]
    lines = _lines(runs.first.journal)
    for line, row in zip(lines, runs.rows):
        assert line['bank'] == row['role'] and line['prompt_hash'] == row['prompt_hash']
        assert line['generated_tokens'] == len(line['completion_ids'])
        if row['id'] not in runs.plan:                  # genuine greedy output of the tiny model
            assert line['stop_reason'] == 'length' and line['generated_tokens'] == 128
            assert line['decoder']['mode'] == 'greedy' and line['decoder']['argmax_replay_verified'] is True
    assert receipt['run_meta'] == {'model_sha256': tiny_dir.weights_sha256,
                                   'tokenizer_sha256': s0.TOKENIZER_RECEIPT_SHA,
                                   'decoding_sha256': s0.decoding_sha256(receipt['decoding_preimage'])}
    assert receipt['guard_dev_enrollment'] is None and receipt['guard_dev_enrollment_status'].startswith('NOT_ENROLLED')
    summary = receipt['summary']
    assert summary['guard_dev']['source_correct'] == 1 and summary['calibration']['strict_correct'] == 1
    assert summary['guard_dev']['by_family'] == {row['verifier']['spec']['family']: {
        'rows': 1, 'source_correct': int(i == 0)} for i, row in enumerate(runs.guard_rows)}
    assert summary['calibration']['by_skill']['R'] == {'rows': 1, 'strict_correct': 1, 'unknown_answer': 0}
    assert summary['walls']['rows'] == 6 and summary['walls']['stop_reason'] == {'eos': 2, 'length': 4}
    calibration = receipt['calibration_scoring']
    assert calibration['canary_overall'] == 'PASS' and calibration['code_bundle_sha256'] == comp.live_bundle()
    assert calibration['scoring']['run'] == receipt['run_meta']
    assert calibration['scoring']['strict_correct_ids'] == [runs.cal_rows[0]['id']]
    assert receipt['guard_dev_scoring']['scoring']['strict_correct_ids'] == [runs.guard_rows[0]['id']]
    for key in ('calibration_scoring', 'guard_dev_scoring'):     # the scorer saw each row's own stop reason
        assert receipt[key]['scoring']['counts']['overall']['stop_reason'] == {'eos': 1, 'length': 2}
    assert receipt['canary_receipts_identical'] and receipt['decoder']['rows'] == 6
    assert receipt['inputs']['plan_by_bank'] == {'guard_dev': 3, 'calibration': 3}
    assert receipt['inputs']['full_guard_dev'] is False
    assert receipt['inputs']['guard_dev_bank_sha256'] == guard_bank.bank_sha256
    assert runs.first.lease.events == [('acquire', 'comp-rs-001-s0-enroll-run1'), ('write', 'S0_ENROLL_RUN1.json'),
                                       ('release', 'token-1')]
    assert runs.aborted.lease.events == [('acquire', 'comp-rs-001-s0-enroll-run1'), ('release', 'token-1')]
    assert [c[0] for c in runs.first.host.calls] == ['require', 'guard', 'guard', 'guard']


def test_churn_is_zero_between_two_identical_cpu_runs(enroll_runs, tmp_path):
    runs = enroll_runs
    out = tmp_path / 'S0_CHURN.json'
    code = churn.main(['--test-mode', '--run1', str(runs.first.out), '--run2', str(runs.second.out), '--out', str(out)])
    receipt = _receipt(out)
    assert code == 0 and receipt['outcome'] == 'CHURN_ZERO' and receipt['churn'] == 0 and receipt['rows_compared'] == 6
    assert receipt['differing_ids'] == [] and receipt['scores_equal'] == {'guard_dev_scoring': True,
                                                                          'calibration_scoring': True}
    assert receipt['separate_processes'] is False and receipt['separate_processes_required'] is False
    assert not set(receipt['runs']['1']['sessions']) & set(receipt['runs']['2']['sessions'])
    assert len(receipt['runs']['1']['sessions']) == 3 and receipt['decoded'] is False
    with pytest.raises(SystemExit, match='create-only'):
        churn.main(['--test-mode', '--run1', str(runs.first.out), '--run2', str(runs.second.out), '--out', str(out)])


def test_churn_counts_a_perturbed_output(enroll_runs, tmp_path, tiny_dir, tokenizer):
    runs = enroll_runs
    target = runs.guard_rows[1]['id']
    perturbed = _run_enroll(tmp_path / 'records', 2, tiny_dir,
                            Injector(tokenizer, runs.rows, runs.plan, perturb=target))
    assert perturbed.code == 0
    out = tmp_path / 'S0_CHURN.json'
    code = churn.main(['--test-mode', '--run1', str(runs.first.out), '--run2', str(perturbed.out), '--out', str(out)])
    receipt = _receipt(out)
    assert code == 2 and receipt['outcome'] == 'STOP_NONDETERMINISTIC' and receipt['churn'] == 1
    assert receipt['differing_ids'] == [target] and receipt['by_bank'] == {'guard_dev': 1}
    (diff,) = receipt['differing']
    assert diff['first_difference'] == 0 and diff['index'] == 1


# ------------------------------------------------------------------------------------------------ churn units
def _synthetic_run(root, run, ids, *, completions=None, session=None, process=(1, 2.0), registered=False,
                   bind_changes=None, strict=()):
    journal = root / ('run%d.jsonl' % run)
    bind = dict({'stage': 'S0_ENROLL_RUN%d' % run, 'run': run, 'test_mode': not registered, 'plan_sha256': digest(ids),
                 'decoding_sha256': 'd' * 64, 'arm_sha256': 'a' * 64}, **(bind_changes or {}))
    writer = g.ResumableJsonl(journal, key='id', bind=bind)
    for index, rid in enumerate(ids):
        ids_ = (completions or {}).get(rid, [5, 6, IM_END])
        writer.append({'id': rid, 'index': index, 'bank': 'guard_dev', 'group': 'countdown', 'prompt_hash': 'p' + rid,
                       'prompt_ids_sha256': 'q' + rid, 'completion_ids': ids_, 'stop_reason': 'eos',
                       'response': 'r%d' % sum(ids_), 'generated_tokens': len(ids_),
                       'process': {'session': session or ('%032x' % run), 'pid': process[0],
                                   'process_created': process[1]}})
    scoring = {'scoring': {'strict_correct_ids': sorted(strict)}}
    receipt = {'stage': 'S0_ENROLL_RUN%d' % run, 'schema': enroll.SCHEMA, 'run': run, 'completed': True,
               'registered_run': registered, 'test_mode': not registered, 'inputs': {'planned': len(ids)},
               'journal': {'path': str(journal), 'sha256': sha(journal), 'bytes': journal.stat().st_size,
                           'rows': len(ids), 'bind': bind},
               'scorer_bundle': {'expected': 'b' * 64}, 'guard_dev_scoring': scoring, 'calibration_scoring': scoring}
    return _write_json(root / ('run%d.json' % run), receipt)


def test_compare_runs_refuses_incomparable_runs(tmp_path):
    ids = ['a', 'b', 'c']

    def pair(name, one=None, two=None, registered=False):
        root = tmp_path / name
        return (_synthetic_run(root, 1, ids, registered=registered, **(one or {})),
                _synthetic_run(root, 2, ids, registered=registered, **(two or {})))
    result = churn.compare_runs(*pair('same'), registered=False)
    assert result['churn'] == 0 and result['outcome'] == 'CHURN_ZERO' and result['separate_processes'] is False
    one, two = pair('diff', two={'completions': {'b': [5, 7, IM_END]}})
    result = churn.compare_runs(one, two, registered=False)
    assert result['churn'] == 1 and result['differing_ids'] == ['b'] and result['differing'][0]['first_difference'] == 1
    with pytest.raises(ValueError, match='bind fields differ: decoding_sha256'):
        churn.compare_runs(*pair('bind', two={'bind_changes': {'decoding_sha256': 'e' * 64}}), registered=False)
    with pytest.raises(ValueError, match='launch session wrote rows of both runs'):
        churn.compare_runs(*pair('session', one={'session': 'f' * 32}, two={'session': 'f' * 32}), registered=False)
    with pytest.raises(ValueError, match='separate process'):
        churn.compare_runs(*pair('process', registered=True), registered=True)
    result = churn.compare_runs(*pair('process2', two={'process': (1, 3.0)}, registered=True), registered=True)
    assert result['separate_processes'] is True and result['churn'] == 0
    with pytest.raises(ValueError, match='not a registered run'):
        churn.compare_runs(*pair('mode'), registered=True)
    with pytest.raises(ValueError, match='scored differently'):
        churn.compare_runs(*pair('scores', two={'strict': ['a']}), registered=False)
    one, two = pair('order')
    swapped = _synthetic_run(tmp_path / 'order2', 2, ids[::-1], bind_changes={'plan_sha256': digest(ids)})
    with pytest.raises(ValueError, match='same plan'):
        churn.compare_runs(one, swapped, registered=False)
    one, two = pair('bytes')
    journal = tmp_path / 'bytes' / 'run2.jsonl'
    data = journal.read_bytes()
    journal.write_bytes(data + b'{"id":"x')
    with pytest.raises(ValueError, match='bytes differ'):
        churn.compare_runs(one, two, registered=False)
    receipt = json.loads(two.read_bytes())
    receipt['journal'].update(sha256=sha(journal), bytes=journal.stat().st_size)
    two.write_bytes(json.dumps(receipt).encode('ascii'))
    with pytest.raises(ValueError, match='partial last line'):
        churn.compare_runs(one, two, registered=False)
    assert journal.read_bytes() == data + b'{"id":"x'          # nothing was truncated or repaired
    one, two = pair('derived')
    receipt = json.loads(two.read_bytes())
    path = Path(receipt['journal']['path'])
    rows = [json.loads(line) for line in path.read_bytes().splitlines()]
    path.unlink()
    writer = g.ResumableJsonl(path, key='id', bind=receipt['journal']['bind'])
    for row in rows:
        body = {k: v for k, v in row.items() if k != 'row_sha256' and k not in receipt['journal']['bind']}
        writer.append(dict(body, response='other') if body['id'] == 'c' else body)
    receipt['journal'].update(sha256=sha(path), bytes=path.stat().st_size)
    two.write_bytes(json.dumps(receipt).encode('ascii'))
    with pytest.raises(ValueError, match='equal token IDs but different derived fields: c'):
        churn.compare_runs(one, two, registered=False)
    incomplete = json.loads(one.read_bytes())
    incomplete['completed'] = False
    with pytest.raises(ValueError, match='not complete'):
        churn.compare_runs(_write_json(tmp_path / 'incomplete.json', incomplete), two, registered=False)


def test_compare_runs_asserts_the_runtime_freeze(tmp_path):
    one = _synthetic_run(tmp_path, 1, ['a', 'b'])
    two = _synthetic_run(tmp_path, 2, ['a', 'b'])
    freeze = {'decoding_sha256': 'd' * 64, 'scoring_code_bundle_sha256': 'b' * 64}
    assert churn.compare_runs(one, two, registered=False, freeze=freeze)['churn'] == 0
    with pytest.raises(ValueError, match='decoding_sha256 differs from the runtime freeze'):
        churn.compare_runs(one, two, registered=False, freeze=dict(freeze, decoding_sha256='e' * 64))
    with pytest.raises(ValueError, match='another scorer bundle'):
        churn.compare_runs(one, two, registered=False, freeze=dict(freeze, scoring_code_bundle_sha256='c' * 64))


# ------------------------------------------------------------------------------------------------ CLI refusals
@pytest.fixture
def no_pinned(monkeypatch):
    """Registered-mode refusals must happen before the pinned host budget or lease is ever loaded."""
    def never(*a, **k):
        raise AssertionError('pinned module loaded before the refusal')
    for name in ('hostbudget', 'gpu_lease', 'assert_runtime', 'assert_clean_checkout'):
        monkeypatch.setattr(s0, name, never)


REGISTERED_REFUSALS = [
    (['--device', 'cpu'], 'runs on cuda'), (['--host-floor-gib', '2'], 'test-mode only'),
    (['--model-dir', 'X'], 'test-mode only'), (['--asset-receipt', 'X'], 'test-mode only'),
    ([], 'runtime freeze'), (['--repo', '.'], 'binds the checkout'), (['--out', 'OUT'], 'written to')]


@pytest.mark.parametrize('extra,message', REGISTERED_REFUSALS)
@pytest.mark.parametrize('stage', ['competence', 'enroll'])
def test_registered_refusals_before_any_work(stage, extra, message, no_pinned, tmp_path):
    freeze = [] if message == 'runtime freeze' else ['--runtime-freeze', str(tmp_path / 'F.json')]
    extra = [str(tmp_path / 'x.json') if v == 'OUT' else str(tmp_path) if v == '.' else v for v in extra]
    argv = freeze + extra + (['--run', '1'] if stage == 'enroll' else [])
    main = comp.main if stage == 'competence' else enroll.main
    with pytest.raises(SystemExit, match=message):
        main(argv)


def test_registered_refusals_specific_to_each_stage(no_pinned, tmp_path):
    freeze = ['--runtime-freeze', str(tmp_path / 'F.json')]
    with pytest.raises(SystemExit, match='test-mode only option.*--group-size'):
        comp.main(freeze + ['--group-size', '8'])
    with pytest.raises(SystemExit, match='injected'):
        comp.main(freeze, lease_module=FakeLease())
    with pytest.raises(SystemExit, match='injected'):
        enroll.main(freeze + ['--run', '2'], host_module=FakeHost())
    with pytest.raises(SystemExit):
        enroll.main(freeze + ['--run', '3'])
    with pytest.raises(SystemExit, match='runtime freeze'):
        churn.main(['--run1', 'a', '--run2', 'b'])
    with pytest.raises(SystemExit, match='written to'):
        churn.main(freeze + ['--run1', 'a', '--run2', 'b', '--out', str(tmp_path / 'c.json')])
    with pytest.raises(SystemExit, match='binds the checkout'):
        churn.main(freeze + ['--run1', 'a', '--run2', 'b', '--repo', str(tmp_path)])


@pytest.mark.parametrize('device', [['--device', 'cuda'], []])
@pytest.mark.parametrize('stage', ['competence', 'enroll'])
def test_test_mode_is_refused_on_cuda_before_any_work(stage, device, no_pinned, tmp_path):
    """--test-mode never runs on the GPU: refused with --device cuda (given or the default), before any work."""
    argv = device + ['--test-mode', '--out', str(tmp_path / 'x.json'), '--host-floor-gib', '1']
    argv += ['--run', '1'] if stage == 'enroll' else []
    main = comp.main if stage == 'competence' else enroll.main
    with pytest.raises(SystemExit, match='refused with --device cuda'):
        main(argv, lease_module=FakeLease(), host_module=FakeHost())
    assert not (tmp_path / 'x.json').exists() and not (tmp_path / 'x.jsonl').exists()


def test_test_mode_refusals(tmp_path, tiny_dir):
    base = ['--device', 'cpu', '--test-mode']
    with pytest.raises(SystemExit, match='needs --out'):
        comp.main(base + ['--host-floor-gib', '1'])
    with pytest.raises(SystemExit, match='--host-floor-gib or --runtime-freeze'):
        comp.main(base + ['--out', str(tmp_path / 'a.json')])
    with pytest.raises(SystemExit, match='must be > 0'):
        comp.main(base + ['--out', str(tmp_path / 'a.json'), '--host-floor-gib', '0'])
    with pytest.raises(SystemExit, match='group size'):
        comp.main(base + ['--out', str(tmp_path / 'a.json'), '--host-floor-gib', '1', '--group-size', '1'])
    with pytest.raises(SystemExit, match='.json file'):
        comp.main(base + ['--out', str(tmp_path / 'a.txt'), '--host-floor-gib', '1'])
    with pytest.raises(ValueError, match='inside the bound checkout'):
        enroll.main(base + ['--run', '1', '--out', str(comp.REPO / 'S0_ENROLL_RUN1.json'), '--host-floor-gib', '1'])
    with pytest.raises(SystemExit, match='REFUSING TO START'):
        comp.main(common_argv(tiny_dir, tmp_path / 'b.json'), lease_module=FakeLease(), host_module=FakeHost(free=0.1))
    assert not (tmp_path / 'b.jsonl').exists()


def test_registered_guard_dev_plan_is_the_whole_bank_in_order(guard_bank):
    plan = enroll.guard_dev_plan(guard_bank)
    assert [row['id'] for row in plan] == [row['id'] for row in guard_bank] and len(plan) == enroll.GUARD_DEV_ROWS


def test_a_source_change_during_the_run_writes_no_receipt(tmp_path):
    cfg = {'stage': 'S0_COMPETENCE', 'out': tmp_path / 'r.json'}
    with pytest.raises(ValueError, match='source changed during the run'):
        comp.write_stage_receipt(cfg, {'a': 1}, comp.REPO, comp.CODE_FILES, dict(comp._IMPORT_CODE, extra='0'))
    assert not cfg['out'].exists()
    assert comp.write_stage_receipt(cfg, {'a': 1}, comp.REPO, comp.CODE_FILES, comp._IMPORT_CODE) == sha(cfg['out'])


def test_modules_are_torch_free_at_import_and_hash_their_code():
    import subprocess
    import sys
    code = ('import sys; from experiments.maxrl_comp002_pretraining_v1 import s0_competence, s0_enroll, s0_churn; '
            'print(int("torch" in sys.modules))')
    done = subprocess.run([sys.executable, '-c', code], cwd=str(comp.REPO), capture_output=True, text=True,
                          timeout=120, check=True)
    assert done.stdout.strip() == '0'
    for files in (comp.CODE_FILES, enroll.CODE_FILES, churn.CODE_FILES):
        assert comp.PACKAGE + 's0_competence.py' in files
        assert 'experiments/maxrl_composable_skills_v2/corpus.py' in files
        assert set(scoped_scorer.CODE_FILES) <= set(files)
    assert comp.PACKAGE + 's0_enroll.py' in enroll.CODE_FILES and comp.PACKAGE + 's0_churn.py' in churn.CODE_FILES
    assert comp.PACKAGE + 's0_enroll.py' not in comp.CODE_FILES


def test_c13_samples_are_sequential_single_calls_with_registered_seeds(tmp_path, tiny_dir, tokenizer, cal, oracles):
    rows = cal_subset(cal['rows_by_skill'])
    seen = []
    injector = Injector(tokenizer, rows, _competence_plan(tokenizer, oracles, rows))
    real = injector.sample_group

    def spy(model, prompt_ids, G, seed, **kwargs):
        seen.append((G, seed))
        return real(model, prompt_ids, G, seed, **kwargs)
    injector.sample_group = spy
    run = _run_competence(tmp_path / 'records', tiny_dir, tokenizer, cal, oracles, injector=injector)
    assert run.code == 0 and seen and all(G == 1 for G, _ in seen)
    lines = _lines(run.journal)
    expected = [comp.sample_seed(line['id'], k) for line in lines for k in range(line['group_size'])]
    assert [seed for _, seed in seen] == expected
    for line in lines:
        assert line['sample_seeds'] == [comp.sample_seed(line['id'], k) for k in range(line['group_size'])]
        assert line['sampling'] == 'C13 sequential single-sample calls'
    assert comp.sample_seed('MC2-a', 0) != comp.sample_seed('MC2-a', 1) != comp.comp_seed('MC2-a')


# ------------------------------------------------------------------------------------------ theta0' (section 18)
def _anchor_freeze(root, tiny_dir, delta, name='S0_RUNTIME_FREEZE_WARM.json', **over):
    """A test-mode runtime freeze naming theta0' = tiny Supa + delta (the delta written create-only under root)."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'delta_warm.safetensors'
    delta_sha = arm_build.write_delta(path, delta)
    s0.configure('cpu')
    model, _ = s0.load_model_fp32(tiny_dir.root, 'cpu', tiny_dir.asset)
    s0.add_delta(model, delta)
    arm = s0.anchor_digest(model)
    del model
    anchor = dict({'base_weights_sha256': tiny_dir.weights_sha256, 'delta_path': str(path),
                   'delta_sha256': delta_sha, 'arm_sha256': arm, 'warmstart_receipt_sha256': 'f' * 64}, **over)
    freeze, _ = _cpu_freeze(root, name, anchor=anchor)
    return freeze, anchor


def _bound(anchor):
    return {key: anchor[key] for key in s0.ANCHOR_KEYS}


def test_competence_on_an_anchored_tiny_model_binds_the_anchor(tmp_path, tiny_dir, tokenizer, cal, oracles):
    freeze, anchor = _anchor_freeze(tmp_path / 'freeze', tiny_dir, {'model.norm.weight': torch.full((32,), 0.5)})
    run = _run_competence(tmp_path / 'records', tiny_dir, tokenizer, cal, oracles,
                          extra=('--runtime-freeze', str(freeze)))
    receipt = _receipt(run.out)
    assert run.code == 0 and receipt['completed'] is True
    identity = receipt['identity']
    assert identity['anchor'] == receipt['anchor'] == _bound(anchor)
    assert identity['state_fingerprint'] == anchor['arm_sha256'] != identity['base']['state_fingerprint']
    assert identity['anchor_digest_rule'] == s0.ANCHOR_DIGEST_RULE and identity['anchor_delta']['tensors'] == 1
    assert receipt['journal']['bind']['anchor'] == _bound(anchor)
    assert all(line['anchor'] == _bound(anchor) for line in _lines(run.journal))
    assert receipt['runtime_freeze']['anchor'] == anchor and receipt['fingerprint_unchanged'] is True
    # the run metadata names the model read: theta0', not the Supa weights file
    assert receipt['run_meta']['model_sha256'] == anchor['arm_sha256'] != tiny_dir.weights_sha256
    assert receipt['scoring']['batches'][0]['canary_overall'] == 'PASS'


def test_the_supa_read_binds_an_anchor_with_no_delta(competence_run, tiny_dir):
    receipt = competence_run.receipt
    supa = {'base_weights_sha256': tiny_dir.weights_sha256, 'delta_path': None, 'delta_sha256': None,
            'arm_sha256': None}
    assert receipt['anchor'] == receipt['identity']['anchor'] == receipt['journal']['bind']['anchor'] == supa
    assert 'base' not in receipt['identity'] and receipt['runtime_freeze'] is None


def test_a_journal_is_never_resumed_across_anchors(tmp_path, tiny_dir, tokenizer, cal, oracles):
    """A zero delta leaves every weight bit, so the arm and weights digests agree: only the anchor bind separates
    a Supa journal from a theta0' one, in both directions."""
    rows = cal_subset(cal['rows_by_skill'])
    plan = _competence_plan(tokenizer, oracles, rows)
    freeze, anchor = _anchor_freeze(tmp_path / 'freeze', tiny_dir, {'model.norm.weight': torch.zeros(32)})
    s0.configure('cpu')
    _, plain = s0.load_model_fp32(tiny_dir.root, 'cpu', tiny_dir.asset)
    assert anchor['arm_sha256'] == plain['state_fingerprint']
    warm = ('--runtime-freeze', str(freeze))
    for name, first, second in (('supa_first', (), warm), ('warm_first', warm, ())):
        root = tmp_path / name
        with pytest.raises(RuntimeError, match='simulated crash'):
            _run_competence(root, tiny_dir, tokenizer, cal, oracles, extra=first,
                            injector=Injector(tokenizer, rows, plan, fail_at=3))
        journal = root / 'S0_COMPETENCE.jsonl'
        kept = journal.read_bytes()
        assert len(_lines(journal)) == 1
        with pytest.raises(ValueError, match='bind field anchor differs'):
            _run_competence(root, tiny_dir, tokenizer, cal, oracles, extra=second,
                            injector=Injector(tokenizer, rows, plan))
        assert journal.read_bytes() == kept and not (root / 'S0_COMPETENCE.json').exists()


def test_a_freeze_naming_other_base_weights_is_refused_after_the_load(tmp_path, tiny_dir, tokenizer, cal, oracles):
    freeze, _ = _anchor_freeze(tmp_path / 'freeze', tiny_dir, {'model.norm.weight': torch.full((32,), 0.5)},
                               base_weights_sha256='e' * 64)
    with pytest.raises(ValueError, match='the loaded anchor differs from the runtime freeze'):
        _run_competence(tmp_path / 'records', tiny_dir, tokenizer, cal, oracles,
                        extra=('--runtime-freeze', str(freeze)))
    assert not (tmp_path / 'records' / 'S0_COMPETENCE.jsonl').exists()


def test_enroll_on_an_anchored_tiny_model_binds_the_anchor(tmp_path, tiny_dir, tokenizer, cal, guard_bank, oracles):
    freeze, anchor = _anchor_freeze(tmp_path / 'freeze', tiny_dir, {'model.norm.weight': torch.full((32,), 0.5)})
    guard_rows, cal_rows = guard_subset(guard_bank), cal_subset(cal['rows_by_skill'])
    injector = Injector(tokenizer, guard_rows + cal_rows, _enroll_plan(tokenizer, oracles, guard_rows, cal_rows))
    run = _run_enroll(tmp_path / 'records', 1, tiny_dir, injector, name='S0_ENROLL_RUN1_WARM.json',
                      extra=('--runtime-freeze', str(freeze)))
    receipt = _receipt(run.out)
    assert run.code == 0 and receipt['anchor'] == receipt['journal']['bind']['anchor'] == _bound(anchor)
    assert receipt['run_meta']['model_sha256'] == anchor['arm_sha256']
    assert receipt['calibration_scoring']['scoring']['run'] == receipt['run_meta']
    assert run.journal.name == 'S0_ENROLL_RUN1_WARM.jsonl'


def _registered_freeze(root, name, anchor=None):
    preimage = {'schema': 'COMP-RS-001-S0-DECODING-1', 'device': 'cuda:0', 'dtype': 'float32'}
    return _write_json(root / name, {
        'stage': 'S0_RUNTIME_FREEZE', 'test_mode': False, 'registered_run': True, 'decoding_preimage': preimage,
        'decoding_sha256': s0.decoding_sha256(preimage), 'host_floor_gib': 2.5,
        'scoring_code_bundle_sha256': comp.live_bundle(), 'inventory': {'name': 'PENDING'}, 'anchor': anchor})


REGISTERED_ANCHOR = {'base_weights_sha256': 'b' * 64, 'delta_path': r'C:\records\delta_warm.safetensors',
                     'delta_sha256': 'c' * 64, 'arm_sha256': 'd' * 64, 'warmstart_receipt_sha256': 'e' * 64}


def test_registered_names_carry_the_warm_suffix_for_an_anchored_freeze(tmp_path, monkeypatch, no_pinned):
    records = tmp_path / 'records'
    records.mkdir()
    monkeypatch.setattr(s0, 'RECORDS', records)
    warm = _registered_freeze(tmp_path, 'F_WARM.json', REGISTERED_ANCHOR)
    supa = _registered_freeze(tmp_path, 'F.json')

    def competence(freeze, *extra):
        return comp._competence_config(comp._parser().parse_args(['--runtime-freeze', str(freeze), *extra]),
                                       None, None)
    cfg = competence(warm)
    assert (cfg['out'].name, cfg['journal'].name, cfg['attempts'].name) == (
        'S0_COMPETENCE_WARM.json', 'S0_COMPETENCE_WARM.jsonl', 'S0_COMPETENCE_WARM.attempts.jsonl')
    assert cfg['out'].parent == records.resolve() and cfg['anchor'] == REGISTERED_ANCHOR
    assert competence(supa)['out'].name == 'S0_COMPETENCE.json' and competence(supa)['anchor'] is None
    assert competence(warm, '--out', str(records / 'S0_COMPETENCE_R2_WARM.json'))['out'].name == \
        'S0_COMPETENCE_R2_WARM.json'
    for freeze, name in ((warm, 'S0_COMPETENCE.json'), (supa, 'S0_COMPETENCE_WARM.json')):
        with pytest.raises(SystemExit, match='_WARM receipt name'):
            competence(freeze, '--out', str(records / name))
    for run in (1, 2):
        args = enroll._parser().parse_args(['--run', str(run), '--runtime-freeze', str(warm)])
        cfg = enroll._config(args, None, None)
        assert (cfg['out'].name, cfg['journal'].name) == ('S0_ENROLL_RUN%d_WARM.json' % run,
                                                          'S0_ENROLL_RUN%d_WARM.jsonl' % run)
        args = enroll._parser().parse_args(['--run', str(run), '--runtime-freeze', str(supa)])
        assert enroll._config(args, None, None)['out'].name == 'S0_ENROLL_RUN%d.json' % run
    warm_freeze, supa_freeze = comp.read_freeze(warm, registered=True), comp.read_freeze(supa, registered=True)
    assert churn.registered_out(None, warm_freeze) == records / 'S0_CHURN_WARM.json'
    assert churn.registered_out(None, supa_freeze) == records / 'S0_CHURN.json'
    with pytest.raises(SystemExit, match='_WARM receipt name'):
        churn.registered_out(records / 'S0_CHURN.json', warm_freeze)
    assert list(records.iterdir()) == []


def test_compare_runs_asserts_the_frozen_anchor(tmp_path):
    bound = _bound(REGISTERED_ANCHOR)
    supa_bound = dict(bound, delta_path=None, delta_sha256=None, arm_sha256=None)
    warm_freeze = {'decoding_sha256': 'd' * 64, 'scoring_code_bundle_sha256': 'b' * 64, 'anchor': REGISTERED_ANCHOR}
    supa_freeze = dict(warm_freeze, anchor=None)
    warm = [_synthetic_run(tmp_path / 'w', run, ['a', 'b'], bind_changes={'anchor': bound}) for run in (1, 2)]
    supa = [_synthetic_run(tmp_path / 's', run, ['a', 'b'], bind_changes={'anchor': supa_bound}) for run in (1, 2)]
    result = churn.compare_runs(*warm, registered=False, freeze=warm_freeze)
    assert result['churn'] == 0 and result['bind']['anchor'] == bound
    assert churn.compare_runs(*supa, registered=False, freeze=supa_freeze)['churn'] == 0
    for runs, freeze in ((warm, supa_freeze), (supa, warm_freeze),
                         (warm, dict(warm_freeze, anchor=dict(REGISTERED_ANCHOR, arm_sha256='0' * 64)))):
        with pytest.raises(ValueError, match='another anchor than the runtime freeze'):
            churn.compare_runs(*runs, registered=False, freeze=freeze)
    with pytest.raises(ValueError, match='bind fields differ: anchor'):
        churn.compare_runs(supa[0], warm[1], registered=False)


def test_anchor_helpers():
    assert comp.anchor_arguments(None) == {}
    assert comp.anchor_arguments(REGISTERED_ANCHOR) == {'delta_path': REGISTERED_ANCHOR['delta_path'],
                                                        'delta_sha256': 'c' * 64, 'arm_sha256': 'd' * 64}
    bound = _bound(REGISTERED_ANCHOR)
    supa = dict(bound, delta_path=None, delta_sha256=None, arm_sha256=None)
    assert comp.anchor_matches(bound, REGISTERED_ANCHOR) and not comp.anchor_matches(supa, REGISTERED_ANCHOR)
    assert comp.anchor_matches(supa, None) and comp.anchor_matches(None, None)
    assert not comp.anchor_matches(bound, None) and not comp.anchor_matches(None, REGISTERED_ANCHOR)
    assert not comp.anchor_matches(dict(bound, base_weights_sha256='0' * 64), REGISTERED_ANCHOR)
