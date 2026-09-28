"""S0-Q1 FP32 probe (Amendment 1 section 16): probe prompts, checks T1..T7, the CLI receipt, refusals and failures.

CPU only. No test creates a CUDA tensor or initialises CUDA (asserted), takes the real GPU lease or opens the real
Supa weights. The real tokenizer files, config.json and the CALIBRATION files are read; everything else is
synthetic, in tmp_path.

The tiny model (vocab 151936, hidden 32, 2 layers, 4/2 heads, head_dim 8, intermediate 64, tied) is transformers'
seeded random init with every 2-D weight scaled by SCALE (so the context, not the last token, drives the greedy
chain), stored as BF16 safetensors with a config.json in the real format. Its <|endoftext|> row (151643, an EOS ID
that never occurs in a rendered prompt, and the PAD ID) is solved for so that, under greedy decoding with
max_new_tokens=CAP, probe position 0 stops with EOS at step 1, position 3 at step 3, and every other position runs
to the cap (pinned by test_crafted_model_outcomes). The synthetic CALIBRATION set has five parents with R and S rows
plus a boundary row; the probe must take the first four parents only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from experiments.maxrl_composable_skills_v2 import contracts
from experiments.maxrl_comp002_pretraining_v1 import s0_generate as g
from experiments.maxrl_comp002_pretraining_v1 import s0_q1_fp32 as q1
from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as s0
from experiments.maxrl_comp002_pretraining_v1.preflight import digest, read_json, sha

REPO = Path(__file__).resolve().parents[3]
TINY_SIZES = {'hidden_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
              'head_dim': 8, 'intermediate_size': 64, 'max_window_layers': 2}
SCALE = 10.0
SEED = 20260928
EOS2 = 151643
CAP = 4
STOPS = {0: 1, 3: 3}          # probe position -> greedy step that emits EOS; every other position hits CAP
MARGIN = 1.0
BLOCKS = ('1', '1')
REAL_LENGTHS = [835, 231, 523, 219, 822, 228, 535, 222]


# ------------------------------------------------------------------------------------------------ fixtures
@pytest.fixture(scope='module')
def tokenizer():
    if not (s0.MODEL_DIR / 'tokenizer.json').is_file():
        pytest.skip('Supa tokenizer files absent')
    return s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)


def _messages(k: int, skill: str) -> list:
    words = ' '.join('w%d' % j for j in range(4 + 6 * k + (9 if skill == 'R' else 0)))
    return [{'role': 'system', 'content': 'Solve the task. Return exactly one JSON object with the key answer.'},
            {'role': 'user', 'content': 'Parent %d, skill %s: %s' % (k, skill, words)}]


def _jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True).encode('ascii') + b'\n')


def _json_file(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as stream:
        stream.write((json.dumps(value, indent=2, sort_keys=True) + '\n').encode('ascii'))


def _write_data(root: Path) -> list:
    """Synthetic CALIBRATION set: five parents x (R, S) core rows, one boundary row, and its manifest."""
    rows = []
    for k in range(5):
        for skill in ('R', 'S'):
            messages = _messages(k, skill)
            rows.append({'id': 'MC2-test-%d-%s' % (k, skill), 'parent_id': 'world-test-%d' % k, 'role': 'calibration',
                         'tag': 'core', 'skills': [skill], 'kind': 'composition', 'prompt': messages,
                         'prompt_hash': contracts.digest(messages), 'verifier': '{"kind": "synthetic"}'})
    boundary = _messages(9, 'S')
    rows.append({'id': 'MC2-test-boundary', 'parent_id': 'world-test-0', 'role': 'calibration', 'tag': 'boundary',
                 'skills': ['S'], 'kind': 'composition', 'prompt': boundary, 'prompt_hash': contracts.digest(boundary),
                 'verifier': '{}'})
    _jsonl(root / 'canonical' / 'calibration.jsonl', rows)
    _jsonl(root / 'policy' / 'calibration.jsonl', [{'id': r['id'], 'messages': r['prompt']} for r in rows])
    files = {rel: {'bytes': (root / rel).stat().st_size, 'sha256': sha(root / rel)}
             for rel in ('canonical/calibration.jsonl', 'policy/calibration.jsonl')}
    _json_file(root / 'manifest.json', {'files': files})
    return rows


def _selection(root: Path, rows: list, weights_sha: str) -> dict:
    core = [r for r in rows if r['tag'] == 'core']
    parents = list(dict.fromkeys(r['parent_id'] for r in core))
    scoped = [{'id': r['id'], 'parent_id': r['parent_id'], 'prompt_hash': r['prompt_hash'], 'skill': r['skills'][0]}
              for r in core]
    return {'experiment': 'COMP-RS-001', 'stage': 'Q2_SOURCE_SMOKE_SELECTION',
            'dataset_manifest_sha256': sha(root / 'manifest.json'),
            'canonical_calibration_sha256': sha(root / 'canonical' / 'calibration.jsonl'),
            'policy_calibration_sha256': sha(root / 'policy' / 'calibration.jsonl'),
            'source_weight_sha256': weights_sha, 'parent_worlds': parents, 'rows': scoped,
            'selection_digest': digest(scoped), 'model_outputs_seen': False}


def _config() -> dict:
    real = json.loads((s0.MODEL_DIR / 'config.json').read_text(encoding='utf-8'))
    return dict(real, **TINY_SIZES)


def _asset_receipt(root: Path, config: dict, tensors: int) -> dict:
    weights = root / 'model.safetensors'
    return {'stage': 'Q1_ASSET_CANDIDATE', 'model_path': str(root.resolve()),
            'assets': {'model.safetensors': {'bytes': weights.stat().st_size, 'sha256': sha(weights)},
                       'config.json': {'bytes': (root / 'config.json').stat().st_size,
                                       'sha256': sha(root / 'config.json')}},
            'architecture': {key: config[key] for key in s0.ARCHITECTURE_KEYS},
            'model_header': {'tensors': tensors, 'dtypes': ['BF16']}}


def _write_model(root: Path, config: dict, state: dict) -> dict:
    root.mkdir(parents=True)
    _json_file(root / 'config.json', config)
    save_file(state, str(root / 'model.safetensors'), metadata={'format': 'pt'})
    return _asset_receipt(root, config, len(state))


def _craft(root: Path, prompts: list) -> dict:
    """Seeded random init (2-D weights x SCALE, BF16), then the EOS2 row solved for the STOPS outcomes."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    config = _config()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(SEED)
        seed_model = Qwen3ForCausalLM(Qwen3Config(**config)).eval()
    state = {name: (t.detach() * (SCALE if t.ndim == 2 else 1.0)).to(torch.bfloat16).contiguous()
             for name, t in seed_model.state_dict().items() if name != 'lm_head.weight'}
    del seed_model
    base_receipt = _write_model(root / 'base', config, state)
    with torch.random.fork_rng(devices=[]):
        base, _ = s0.load_model_fp32(root / 'base', 'cpu', base_receipt)
    embed = base.model.embed_tokens.weight
    H, T, signs = [], [], []
    with torch.inference_mode():
        for position, prompt in enumerate(prompts):
            ids = list(prompt)
            for step in range(1, CAP + 1):
                hidden = base.model(input_ids=torch.tensor([ids])).last_hidden_state[0, -1]
                logits = hidden @ embed.T
                logits[EOS2] = -math.inf
                best, token = logits.max(0)
                assert int(token) not in s0.EOS_IDS
                ids.append(int(token))
                if position in STOPS and step > STOPS[position]:
                    continue
                sign = 1.0 if STOPS.get(position) == step else -1.0
                H.append(hidden.double())
                T.append(float(best) + sign * MARGIN)
                signs.append(sign)
    del base
    H, T = torch.stack(H), torch.tensor(T, dtype=torch.float64)
    row = (torch.linalg.pinv(H) @ T).to(torch.bfloat16)
    values = (H.float() @ row.float()).tolist()
    assert all((v - (t - s * MARGIN)) * s > MARGIN / 2 for v, t, s in zip(values, T.tolist(), signs))
    state['model.embed_tokens.weight'][EOS2] = row
    return _write_model(root / 'tiny', config, state)


@pytest.fixture(scope='module')
def env(tmp_path_factory, tokenizer):
    root = tmp_path_factory.mktemp('s0_q1')
    data = root / 'corpus'
    rows = _write_data(data)
    prompts = [s0.render(tokenizer, r['prompt']) for r in rows if r['tag'] == 'core'][:8]
    receipt = _craft(root / 'models', prompts)
    asset = root / 'models' / 'tiny_asset_receipt.json'
    _json_file(asset, receipt)
    selection = root / 'selection.json'
    _json_file(selection, _selection(data, rows, receipt['assets']['model.safetensors']['sha256']))
    return SimpleNamespace(root=root, data=data, rows=rows, prompts=prompts, model_dir=root / 'models' / 'tiny',
                           asset=asset, asset_receipt=receipt, selection=selection,
                           weights_sha=receipt['assets']['model.safetensors']['sha256'])


@pytest.fixture(scope='module')
def loaded(env):
    """The tiny model through the CLI's loader, with its greedy completions."""
    with torch.random.fork_rng(devices=[]):
        model, identity = s0.load_model_fp32(env.model_dir, 'cpu', env.asset_receipt)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    greedy = [g.greedy_decode(model, p, max_new_tokens=CAP) for p in env.prompts]
    return SimpleNamespace(model=model, identity=identity, greedy=greedy)


@pytest.fixture
def torch_state():
    """Restore the process-global torch settings that s0_runtime.configure() changes."""
    import os
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


class FakeLease:
    def __init__(self, token='token-q1'):
        self.LOCK = Path('Z:/fake-temp') / s0.LOCK_NAME
        self.token = token
        self.events = []

    def acquire(self, timeout_s, label):
        self.events.append(('acquire', timeout_s, label))
        return self.token

    def release(self, token):
        self.events.append(('release', token))


class FakeHost:
    class LowMemoryAbort(RuntimeError):
        pass

    def __init__(self, abort_on=None, events=None):
        self.abort_on = abort_on
        self.events = [] if events is None else events

    def snapshot(self):
        return {'available': True, 'free_phys_gib': 3.5}

    def require(self, gib, label):
        self.events.append(('require', gib, label))
        return self.snapshot()

    def guard(self, gib, label):
        self.events.append(('guard', gib, label))
        if self.abort_on is not None and label.endswith(self.abort_on):
            raise self.LowMemoryAbort(label + ' standing down')


def _samples(pid):
    return [{'path': r'\\host\gpu process memory(pid_%d_luid_0x0_0x1099f_phys_0)\shared usage' % pid, 'value': 4096.0},
            {'path': r'\\host\gpu process memory(pid_%d_luid_0x0_0x1099f_phys_0)\dedicated usage' % pid,
             'value': 8192.0},
            {'path': r'\\host\gpu process memory(pid_%d_luid_0x0_0x1099f_phys_0)\shared usage' % (pid * 10 + 1),
             'value': 7.0},
            {'path': r'\\host\gpu adapter memory(luid_0x0_0x1099f_phys_0)\shared usage', 'value': 65536.0}]


def fake_runner(stdout='', returncode=0, stderr='', calls=None):
    def run(argv, **kwargs):
        if calls is not None:
            calls.append((argv, kwargs))
        return SimpleNamespace(returncode=returncode, stdout=stdout, stderr=stderr)
    return run


def fake_counter():
    import os
    return q1.gpu_shared_usage(runner=fake_runner(json.dumps(_samples(os.getpid()))))


def _argv(env, out, *extra):
    return ['--test-mode', '--device', 'cpu', '--out', str(out), '--model-dir', str(env.model_dir),
            '--asset-receipt', str(env.asset), '--selection-receipt', str(env.selection), '--data', str(env.data),
            '--max-new-tokens', str(CAP), '--test-blocks', *BLOCKS, *extra]


def run_cli(env, out, *extra, lease=None, host=None, gpu=fake_counter, checkout=None):
    lease = FakeLease() if lease is None else lease
    host = FakeHost() if host is None else host
    with torch.random.fork_rng(devices=[]):
        code = q1.main(_argv(env, out, *extra), lease_module=lease, host_module=host, gpu_counter=gpu,
                       checkout_check=checkout)
    raw = out.read_bytes()
    return code, json.loads(raw), raw, lease, host


# ------------------------------------------------------------------------------------------ probe prompts
def test_selection_pin_and_registered_constants():
    assert sha(q1.Q2_SELECTION) == q1.Q2_SELECTION_SHA
    assert q1.REGISTERED == {'device': 'cuda:0', 'max_new_tokens': 128, 'blocks': [8, 27],
                             'trainable_numel': 314618880, 'stored_tensors': 310, 'num_hidden_layers': 28}
    assert read_json(s0.ASSET_RECEIPT)['model_header']['tensors'] == q1.REGISTERED['stored_tensors']
    assert (q1.LAUNCH_FLOOR_GIB, q1.GUARD_GIB, q1.HOST_MARGIN_GIB, q1.TOL_NATS) == (2.5, 1.0, 1.0, 1e-4)
    assert s0.LEASE_LABEL.fullmatch(q1.LABEL) and q1.T6_POSITIONS == (1, 3) and q1.PROBE_PARENTS == 4


def test_probe_rows_are_the_first_four_parents_in_receipt_order():
    selection = read_json(q1.Q2_SELECTION)
    rows = q1.probe_rows(selection)
    assert [r['id'] for r in rows] == [r['id'] for r in selection['rows'][:8]]
    assert [r['skill'] for r in rows] == ['R', 'S'] * 4 and [r['position'] for r in rows] == list(range(8))
    assert [r['parent_id'] for r in rows[::2]] == selection['parent_worlds'][:4]
    swapped = copy.deepcopy(selection)
    swapped['rows'][0], swapped['rows'][1] = swapped['rows'][1], swapped['rows'][0]
    swapped['selection_digest'] = digest(swapped['rows'])
    with pytest.raises(ValueError, match=r'not \(R, S\) per parent'):
        q1.probe_rows(swapped)
    reordered = copy.deepcopy(selection)
    reordered['parent_worlds'][0], reordered['parent_worlds'][1] = (reordered['parent_worlds'][1],
                                                                    reordered['parent_worlds'][0])
    with pytest.raises(ValueError, match='per parent'):
        q1.probe_rows(reordered)
    tampered = copy.deepcopy(selection)
    tampered['rows'][2]['prompt_hash'] = '0' * 64
    with pytest.raises(ValueError, match='selection digest mismatch'):
        q1.probe_rows(tampered)
    with pytest.raises(ValueError, match='not a Q2 source-smoke selection receipt'):
        q1.probe_rows(dict(selection, stage='Q2_OTHER'))
    with pytest.raises(ValueError, match='probe parent count'):
        q1.probe_rows(selection, 9)


def test_real_probe_prompts_are_bound_and_rendered(tokenizer, comp_rs_data, tmp_path):
    weights = read_json(s0.ASSET_RECEIPT)['assets']['model.safetensors']['sha256']
    prompts, inputs = q1.load_probe_prompts(comp_rs_data, q1.Q2_SELECTION, tokenizer, weights_sha256=weights)
    selection = read_json(q1.Q2_SELECTION)
    assert [p['id'] for p in prompts] == [r['id'] for r in selection['rows'][:8]]
    assert [p['prompt_tokens'] for p in prompts] == REAL_LENGTHS
    assert all(tuple(p['prompt_ids'][-7:]) == s0.THINK_PREFIX_IDS for p in prompts)
    assert inputs['selection_receipt_sha256'] == q1.Q2_SELECTION_SHA and inputs['pinned'] is True
    assert inputs['policy_calibration_sha256'] == selection['policy_calibration_sha256']
    assert inputs['canonical_calibration_sha256'] == selection['canonical_calibration_sha256']
    assert inputs['dataset_manifest_sha256'] == q1.MANIFEST == selection['dataset_manifest_sha256']
    with pytest.raises(ValueError, match='other source weights'):
        q1.load_probe_prompts(comp_rs_data, q1.Q2_SELECTION, tokenizer, weights_sha256='0' * 64)
    copied = tmp_path / 'Q2_SOURCE_SMOKE_SELECTION.json'
    copied.write_bytes(q1.Q2_SELECTION.read_bytes())
    with pytest.raises(ValueError, match='Q2 selection receipt drift'):
        q1.load_probe_prompts(comp_rs_data, copied, tokenizer, weights_sha256=weights)


def test_synthetic_probe_prompts_and_their_refusals(tokenizer, env, tmp_path):
    prompts, inputs = q1.load_probe_prompts(env.data, env.selection, tokenizer, weights_sha256=env.weights_sha,
                                            pinned=False)
    assert [p['id'] for p in prompts] == ['MC2-test-%d-%s' % (k, s) for k in range(4) for s in 'RS']
    assert [p['prompt_ids'] for p in prompts] == env.prompts and inputs['pinned'] is False
    with pytest.raises(ValueError, match='Q2 selection receipt drift'):
        q1.load_probe_prompts(env.data, env.selection, tokenizer, weights_sha256=env.weights_sha)
    real_weights = read_json(s0.ASSET_RECEIPT)['assets']['model.safetensors']['sha256']
    with pytest.raises(ValueError, match='dataset manifest drift'):  # the pinned receipt over another corpus
        q1.load_probe_prompts(env.data, q1.Q2_SELECTION, tokenizer, weights_sha256=real_weights)

    def variant(name, edit_rows, edit_selection=None):
        root = tmp_path / name
        rows = copy.deepcopy(env.rows)
        edit_rows(rows)
        _jsonl(root / 'canonical' / 'calibration.jsonl', [{k: v for k, v in r.items() if k != '_policy'} for r in rows])
        _jsonl(root / 'policy' / 'calibration.jsonl',
               [{'id': r['id'], 'messages': r.get('_policy', r['prompt'])} for r in rows])
        files = {rel: {'bytes': (root / rel).stat().st_size, 'sha256': sha(root / rel)}
                 for rel in ('canonical/calibration.jsonl', 'policy/calibration.jsonl')}
        _json_file(root / 'manifest.json', {'files': files})
        selection = _selection(root, env.rows, env.weights_sha)
        selection.update(dataset_manifest_sha256=sha(root / 'manifest.json'),
                         canonical_calibration_sha256=files['canonical/calibration.jsonl']['sha256'],
                         policy_calibration_sha256=files['policy/calibration.jsonl']['sha256'])
        if edit_selection is not None:
            edit_selection(selection)
        _json_file(root / 'selection.json', selection)
        return q1.load_probe_prompts(root, root / 'selection.json', tokenizer, weights_sha256=env.weights_sha,
                                     pinned=False)

    assert len(variant('same', lambda rows: None)[0]) == 8
    with pytest.raises(ValueError, match='policy messages differ'):
        variant('policy', lambda rows: rows[2].update(_policy=_messages(7, 'R')))
    with pytest.raises(ValueError, match='not the selected core CALIBRATION row'):
        variant('boundary', lambda rows: rows[3].update(tag='boundary'))
    with pytest.raises(ValueError, match='prompt_hash mismatch'):
        variant('hash', lambda rows: rows[1].update(prompt=_messages(8, 'S')))
    with pytest.raises(ValueError, match='canonical/calibration.jsonl drift'):
        variant('drift', lambda rows: None,
                lambda sel: sel.update(canonical_calibration_sha256='0' * 64))


# ----------------------------------------------------------------------------------------- crafted model
def test_crafted_model_outcomes(env, loaded):
    assert len({len(p) for p in env.prompts}) == 8  # every prompt has its own length: T5 has pads
    assert loaded.identity['stored_tensors'] == loaded.identity['tensors_matched_bitwise'] == 24
    outcomes = [(r['stop_reason'], len(r['completion_ids'])) for r in loaded.greedy]
    assert outcomes == [('eos', 1), ('length', CAP), ('length', CAP), ('eos', 3)] + [('length', CAP)] * 4
    assert loaded.greedy[0]['completion_ids'] == [EOS2] and loaded.greedy[3]['completion_ids'][-1] == EOS2
    assert all(not set(r['completion_ids']) & set(s0.EOS_IDS) for i, r in enumerate(loaded.greedy) if i not in STOPS)
    assert len({tuple(r['completion_ids']) for r in loaded.greedy}) >= 6  # the context drives the chain


# ------------------------------------------------------------------------------------------------ the CLI
def test_cli_test_mode_runs_every_check_and_passes(env, tmp_path, torch_state, capsys):
    out = tmp_path / 'records' / 'S0_Q1_FP32.json'
    code, receipt, raw, lease, host = run_cli(env, out)
    assert code == 0 and receipt['passed'] is True and receipt['failed_checks'] == [] and receipt['completed']
    assert receipt['stage'] == 'S0_Q1_FP32' and receipt['schema'] == q1.SCHEMA
    assert receipt['test_mode'] is True and receipt['registered_run'] is False
    assert b'\r' not in raw and raw.endswith(b'}\n')
    printed = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert printed['sha256'] == hashlib.sha256(raw).hexdigest() and printed['passed'] is True
    checks = receipt['checks']
    assert all(checks[name]['pass'] is True for name in q1.CHECKS)
    lengths = [len(p) for p in env.prompts]
    completions = [c['completion_ids'] for c in receipt['completions']]
    C = [len(c) for c in completions]
    assert C == [1, CAP, CAP, 3] + [CAP] * 4 and completions[0] == [EOS2]
    # T1
    assert checks['T1']['n'] == 24 and checks['T1']['checks'] == {
        'tensors_matched_bitwise': True, 'tied_alias': True, 'fp32_on_device': True, 'parameter_inventory': True}
    assert receipt['identity']['arm_sha256'] == checks['T1']['arm_sha256']
    assert receipt['identity']['parameter_inventory']['unique_numel'] == checks['T1']['unique_numel']
    # T2
    assert checks['T2']['bitwise_equal'] is True and checks['T2']['max_abs'] == 0.0
    assert checks['T2']['n'] == sum(lengths) + sum(C) and checks['T2']['n_sequences'] == 8
    # T3 / T4
    assert checks['T3']['n'] == checks['T4']['n'] == sum(C) == 28
    assert checks['T3']['eos_positions'] == 2 and checks['T3']['argmax_equal'] is True
    assert checks['T3']['max_abs'] <= 1e-4 and checks['T4']['max_abs'] <= 1e-4
    # T5
    t5 = checks['T5']
    width = max(n + c for n, c in zip(lengths, C))
    assert t5['labels']['eos_rows'] == 2 and t5['width'] == width
    assert t5['labels']['pad_positions'] == sum(width - n - c for n, c in zip(lengths, C)) > 0
    assert t5['real_positions_selected']['n'] == sum(lengths) + sum(C) - 8
    assert t5['teacher_forced_left_vs_individual']['n'] == 28 and t5['max_abs'] <= 1e-4
    # T6
    t6 = checks['T6']
    assert t6['inventory']['blocks'] == [1, 1] and t6['loss_tokens'] == C[1] + C[3]
    assert t6['fingerprint_before'] == t6['fingerprint_after'] == receipt['identity']['state_fingerprint']
    assert t6['frozen_with_grad'] == [] and t6['nonfinite_grads'] == [] and t6['missing_grads'] == []
    assert t6['optimizers_before'] == t6['optimizers_after'] == 0 and t6['trainable_parameters'] == 11
    assert receipt['inventory'] == t6['inventory']
    # T7 and the host floor
    t7 = checks['T7']
    assert t7['peak_cuda_allocated_gib'] is None and t7['peak_worker_rss_gib'] > 0
    assert t7['gpu_shared_usage']['available'] is True
    assert t7['gpu_shared_usage']['process_shared_usage_bytes'] == 4096.0
    assert receipt['host_floor']['later_stages_floor_gib'] == pytest.approx(t7['peak_worker_rss_gib'] + 1.0)
    # counts: generate runs one forward per generated token, cached_logprobs one per completion token
    assert receipt['counts'] == {'model_loads': 1, 'generations': 8, 'backward_calls': 1, 'optimizer_steps': 0,
                                 'optimizers_at_end': 0, 'lm_forwards': 28 + 16 + 28 + 8,
                                 'decoder_forwards': 28 + 16 + 28 + 8 + 8 + 1 + 1 + 1, 'generated_tokens': 28}
    # launch order, guards before every unit, the lease held for the probe
    assert host.events[:2] == [('require', 2.5, q1.LABEL), ('guard', 1.0, q1.LABEL + ' launch')]
    assert [e[2].split(' ', 1)[1] for e in host.events[2:]] == (
        ['T1'] + ['prompt %d' % i for i in range(8)] + ['T5', 'T6', 'T7'])
    assert all(e[1] == 1.0 for e in host.events[1:])
    assert lease.events == [('acquire', 900.0, q1.LABEL), ('release', 'token-q1')]
    assert receipt['lease']['label'] == q1.LABEL and receipt['checkout']['checked'] is False
    # decoding identity
    assert receipt['decoding_sha256'] == s0.decoding_sha256(receipt['decoding_preimage'])
    assert receipt['settings']['deterministic_algorithms'] is True and receipt['settings']['num_threads'] == 4
    assert [p['id'] for p in receipt['prompts']] == ['MC2-test-%d-%s' % (k, s) for k in range(4) for s in 'RS']
    assert receipt['code_sha256']['s0_q1_fp32.py'] == sha(Path(q1.__file__))
    assert receipt['native_outputs_scored'] is False and receipt['optimizer_steps'] == 0
    assert not torch.cuda.is_initialized()


def test_cli_records_an_injected_checkout_check(env, tmp_path, torch_state):
    seen = []
    code, receipt, _, _, _ = run_cli(env, tmp_path / 'r.json',
                                     checkout=lambda repo: seen.append(repo) or {'repo': str(repo), 'clean': True})
    assert code == 0 and seen == [REPO]
    assert receipt['checkout'] == {'repo': str(REPO), 'clean': True, 'checked': True, 'injected': True}


def test_cli_t1_failure_is_recorded_and_blocks_the_model_checks(env, tmp_path, torch_state, monkeypatch):
    real = s0._upcast
    monkeypatch.setattr(s0, '_upcast', lambda t: real(t) + (0.01 if t.ndim == 1 else 0.0))
    code, receipt, _, lease, _ = run_cli(env, tmp_path / 'r.json')
    assert code == 2 and receipt['passed'] is False and receipt['completed'] is False
    assert receipt['failed_checks'] == ['T1', 'T2', 'T3', 'T4', 'T5', 'T6']
    assert 'BF16 bitwise' in receipt['checks']['T1']['error']['message']
    assert all(receipt['checks'][n]['skipped'] for n in ('T2', 'T3', 'T4', 'T5', 'T6'))
    assert receipt['checks']['T7']['pass'] is True and receipt['counts']['generations'] == 0
    assert lease.events[-1] == ('release', 'token-q1')


def test_cli_t2_nondeterminism_fails_only_t2(env, tmp_path, torch_state, monkeypatch):
    real, calls = q1.full_logits, []

    def flaky(model, ids):
        calls.append(1)
        logits = real(model, ids)
        return logits + 1e-6 if len(calls) == 2 else logits
    monkeypatch.setattr(q1, 'full_logits', flaky)
    code, receipt, _, _, _ = run_cli(env, tmp_path / 'r.json')
    assert code == 2 and receipt['failed_checks'] == ['T2'] and receipt['completed'] is True
    t2 = receipt['checks']['T2']
    assert t2['bitwise_equal'] is False and 0 < t2['max_abs'] < 1e-5
    assert [s['bitwise_equal'] for s in t2['sequences']] == [False] + [True] * 7
    assert [s['pass'] for s in t2['sequences']] == [False] + [True] * 7


def test_cli_guard_abort_stops_the_remaining_units(env, tmp_path, torch_state):
    code, receipt, _, _, host = run_cli(env, tmp_path / 'r.json', host=FakeHost(abort_on=' T5'))
    assert code == 2 and receipt['failed_checks'] == ['T5', 'T6'] and receipt['completed'] is False
    assert receipt['checks']['T5']['error']['type'] == 'LowMemoryAbort'
    assert receipt['checks']['T6']['skipped'] is True and receipt['checks']['T7']['pass'] is True


def test_cli_refusals_happen_before_any_host_or_lease_work(env, tmp_path, monkeypatch):
    lease, host = FakeLease(), FakeHost()
    out = tmp_path / 'r.json'
    cases = [
        (_argv(env, out)[:1] + ['--device', 'cuda'] + _argv(env, out)[3:], {}, 'refused with --device cuda'),
        (['--device', 'cpu', '--out', str(out)], {}, 'never built on the CPU'),
        (['--device', 'cuda', '--model-dir', str(env.model_dir)], {}, 'test-mode only option'),
        (['--device', 'cuda', '--max-new-tokens', '4'], {}, 'test-mode only option'),
        (['--device', 'cuda', '--repo', str(tmp_path)], {}, 'test-mode only option'),
        (['--device', 'cuda'], {'host_module': host}, 'test-mode only'),
        (['--device', 'cuda', '--out', str(tmp_path / 'elsewhere.json')], {}, 'registered receipts are written'),
        (_argv(env, out), {'lease_module': None}, 'needs an injected lease module'),
        (_argv(env, out)[:3] + ['--test-blocks', '1', '1'], {}, 'needs --out'),
    ]
    for argv, injected, match in cases:
        kwargs = {'lease_module': lease, 'host_module': host} if argv[0] == '--test-mode' else {}
        kwargs.update(injected)
        with pytest.raises(SystemExit, match=match):
            q1.main(argv, **kwargs)
    with pytest.raises(ValueError, match='inside the bound checkout'):
        q1.main(_argv(env, REPO / 'S0_Q1_FP32.json'), lease_module=lease, host_module=host)
    out.write_bytes(b'{}\n')
    with pytest.raises(SystemExit, match='create-only'):
        q1.main(_argv(env, out), lease_module=lease, host_module=host)
    assert out.read_bytes() == b'{}\n' and lease.events == [] and host.events == []
    assert not (REPO / 'S0_Q1_FP32.json').exists() and not torch.cuda.is_initialized()


def test_registered_config_binds_the_pins():
    out = s0.RECORDS / 'S0_Q1_FP32.pytest-config-only.json'
    cfg = q1._config(q1._parser().parse_args(['--device', 'cuda', '--out', str(out)]),
                     {'lease_module': None, 'host_module': None, 'gpu_counter': None, 'checkout_check': None})
    assert cfg['device'] == 'cuda:0' and cfg['test_mode'] is False and cfg['out'] == out.resolve()
    assert (cfg['model_dir'], cfg['asset_receipt'], cfg['selection'], cfg['data'], cfg['repo']) == (
        s0.MODEL_DIR, s0.ASSET_RECEIPT, q1.Q2_SELECTION, s0.DATA, q1.REPO)
    assert cfg['max_new_tokens'] == 128 and cfg['blocks'] == [8, 27]
    default = q1._parser().parse_args([])
    assert default.device == 'cuda' and default.test_mode is False and default.lease_timeout_s == 900.0
    assert not out.exists() and not torch.cuda.is_initialized()


def test_registered_launch_order_refuses_a_dirty_checkout_before_the_lease(monkeypatch):
    events = []
    host = FakeHost(events=events)
    monkeypatch.setattr(s0, 'hostbudget', lambda: host)
    monkeypatch.setattr(s0, 'assert_runtime', lambda: events.append(('runtime',)) or {'python': 'x'})

    def dirty(repo):
        events.append(('checkout', Path(repo)))
        raise ValueError('checkout not clean: sentinel')

    def no_lease():
        raise AssertionError('the lease must not be taken')
    monkeypatch.setattr(s0, 'assert_clean_checkout', dirty)
    monkeypatch.setattr(s0, 'gpu_lease', no_lease)
    out = s0.RECORDS / 'S0_Q1_FP32.pytest-never-written.json'
    with pytest.raises(ValueError, match='sentinel'):
        q1.main(['--device', 'cuda', '--out', str(out)])
    assert events == [('require', 2.5, q1.LABEL), ('guard', 1.0, q1.LABEL + ' launch'), ('runtime',),
                      ('checkout', q1.REPO)]
    assert not out.exists() and not torch.cuda.is_initialized()


# ------------------------------------------------------------------------------ the checks, one at a time
def test_t1_record_gates_the_identity(loaded):
    model, identity = loaded.model, loaded.identity
    good = q1.t1_record(model, identity, 'cpu', registered=False)
    assert good['pass'] and good['n'] == 24 and good['unique_numel'] == identity['parameter_inventory']['unique_numel']
    registered = q1.t1_record(model, identity, 'cpu', registered=True)
    assert not registered['pass'] and registered['checks']['registered_tensor_count'] is False
    assert registered['checks']['registered_layers'] is False
    for key, value in (('tensors_matched_bitwise', 23), ('tied', dict(identity['tied'], same_storage=False)),
                       ('parameter_inventory', None)):
        assert not q1.t1_record(model, dict(identity, **{key: value}), 'cpu', registered=False)['pass']
    assert not q1.t1_record(model, identity, 'cuda:0', registered=False)['checks']['fp32_on_device']


def _sequence(env, loaded, i, **changes):
    greedy = loaded.greedy[i]
    args = dict(prompt_ids=env.prompts[i], completion_ids=greedy['completion_ids'],
                stop_reason=greedy['stop_reason'], raw_logprobs=greedy['raw_step_logprobs'])
    args.update(changes)
    with torch.inference_mode():
        return q1.sequence_checks(loaded.model, max_new_tokens=CAP, **args)


def test_sequence_checks_pass_on_greedy_output(env, loaded):
    for i in range(8):
        result = _sequence(env, loaded, i)
        assert result['T2']['pass'] and result['T3']['pass'] and result['T4']['pass']
        assert result['T3']['n'] == result['T4']['n'] == len(loaded.greedy[i]['completion_ids'])
        assert result['T3']['eos_positions'] == int(i in STOPS)
        assert result['T3']['cross_entropy_vs_selected']['max_abs'] <= 1e-6


def test_t3_fails_on_a_logprob_gap_or_a_non_argmax_token(env, loaded):
    raw = list(loaded.greedy[3]['raw_step_logprobs'])
    raw[-1] += 2e-4  # the EOS position
    result = _sequence(env, loaded, 3, raw_logprobs=raw)
    assert not result['T3']['pass'] and result['T3']['eos_max_abs'] > 1e-4 and result['T4']['pass']
    completion = list(loaded.greedy[1]['completion_ids'])
    completion[1] = (completion[1] + 1) % 151000
    result = _sequence(env, loaded, 1, completion_ids=completion)
    assert result['T3']['argmax_equal'] is False and not result['T3']['pass']
    assert result['T4']['pass'] and result['T4']['argmax_equal'] is False  # T4 gates the tolerance only
    # raw logprobs consistent with the non-greedy tokens: every tolerance holds, only the argmax fails T3
    result = _sequence(env, loaded, 1, completion_ids=completion, raw_logprobs=result['selected'])
    assert result['T3']['max_abs'] <= 1e-6 and result['T3']['argmax_equal'] is False and not result['T3']['pass']
    sums = q1._aggregate('T3', [dict(_sequence(env, loaded, i)['T3']) for i in (1, 2)])
    assert sums['eos_positions'] == 0 and sums['eos_exercised'] is False and sums['pass'] is False


def test_t3_fails_when_cross_entropy_disagrees(env, loaded, monkeypatch):
    real = torch.nn.functional.cross_entropy
    monkeypatch.setattr(torch.nn.functional, 'cross_entropy', lambda *a, **k: real(*a, **k) + 1e-3)
    result = _sequence(env, loaded, 2)
    assert result['T3']['full_forward_mechanics']['within_tolerance']
    assert result['T3']['teacher_forced_head_only']['within_tolerance'] and result['T3']['argmax_equal']
    assert not result['T3']['cross_entropy_vs_selected']['within_tolerance'] and not result['T3']['pass']


def test_t4_fails_when_the_cache_path_drifts(env, loaded, monkeypatch):
    real = g.cached_logprobs

    def drift(*args, **kwargs):
        values, argmax = real(*args, **kwargs)
        return values + 2e-4, argmax
    monkeypatch.setattr(g, 'cached_logprobs', drift)
    result = _sequence(env, loaded, 1)
    assert result['T3']['pass'] and not result['T4']['pass'] and result['T4']['max_abs'] > 1e-4


def _t5(env, loaded):
    with torch.inference_mode():
        selected = [_sequence(env, loaded, i)['selected'] for i in range(8)]
        return q1.padded_batch_check(loaded.model, env.prompts, [r['completion_ids'] for r in loaded.greedy],
                                     [r['stop_reason'] for r in loaded.greedy], selected, CAP)


def test_t5_passes_and_fails_when_pads_are_attended_or_positions_are_wrong(env, loaded, monkeypatch):
    good = _t5(env, loaded)
    assert good['pass'] and good['labels']['pass'] and good['real_positions_selected']['max_abs'] <= 1e-4
    real = g.left_pad_batch

    def attended(*args, **kwargs):
        batch = real(*args, **kwargs)
        return dict(batch, attention_mask=torch.ones_like(batch['attention_mask']))
    monkeypatch.setattr(g, 'left_pad_batch', attended)
    bad = _t5(env, loaded)
    assert not bad['pass'] and bad['teacher_forced_left_vs_individual']['max_abs'] > 1e-3
    assert bad['real_positions_selected']['max_abs'] > 1e-3
    assert 'attention span' in bad['labels']['error']['message']  # completion_labels refuses the span too

    def frozen(*args, **kwargs):  # (a constant per-row shift is invisible under RoPE; frozen positions are not)
        batch = real(*args, **kwargs)
        return dict(batch, position_ids=torch.zeros_like(batch['position_ids']))
    monkeypatch.setattr(g, 'left_pad_batch', frozen)
    bad = _t5(env, loaded)
    assert not bad['pass'] and bad['real_positions_selected']['max_abs'] > 1e-4


def test_t5_fails_on_prompt_position_drift_or_a_label_fault_alone(env, loaded, monkeypatch):
    with torch.inference_mode():
        selected = [_sequence(env, loaded, i)['selected'] for i in range(8)]
    args = (loaded.model, env.prompts, [r['completion_ids'] for r in loaded.greedy],
            [r['stop_reason'] for r in loaded.greedy], selected, CAP)
    real_logits = q1.full_logits

    def nudged(model, ids):  # one prompt-position logit of the individual forward moves by 1e-3
        logits = real_logits(model, ids).clone()
        logits[0, ids[1]] += 1e-3
        return logits
    monkeypatch.setattr(q1, 'full_logits', nudged)
    with torch.inference_mode():
        record = q1.padded_batch_check(*args)
    assert not record['pass'] and record['real_positions_selected']['max_abs'] > 1e-4
    assert record['teacher_forced_left_vs_individual']['within_tolerance'] and record['labels']['pass']
    monkeypatch.setattr(q1, 'full_logits', real_logits)
    real_tf = g.teacher_forced_logprobs

    def left_drift(*a, **k):  # only the left-padded training path drifts
        values = real_tf(*a, **k)
        return [v + 2e-4 for v in values] if k.get('padding_side') == 'left' else values
    monkeypatch.setattr(g, 'teacher_forced_logprobs', left_drift)
    with torch.inference_mode():
        record = q1.padded_batch_check(*args)
    assert not record['pass'] and not record['teacher_forced_left_vs_individual']['within_tolerance']
    assert record['real_positions_selected']['within_tolerance'] and record['labels']['pass']
    monkeypatch.setattr(g, 'teacher_forced_logprobs', real_tf)
    real_labels = q1.mechanics.completion_labels

    def eos_dropped(*a, **k):
        labels = real_labels(*a, **k)
        labels[labels == EOS2] = -100
        return labels
    monkeypatch.setattr(q1.mechanics, 'completion_labels', eos_dropped)
    with torch.inference_mode():
        record = q1.padded_batch_check(*args)
    assert not record['pass'] and record['labels']['eos_rows'] == 0 and record['labels']['eos_rows_expected'] == 2
    assert record['teacher_forced_left_vs_individual']['within_tolerance']
    assert record['real_positions_selected']['within_tolerance']


def _batch(prompts, completions):
    return g.left_pad_batch(prompts, completions, s0.PAD_ID)


def test_label_check_requires_masked_pads_and_every_eos():
    prompts = [[5, 6, 7], [8, 9], [3]]
    completions = [[11, 151645], [12, 13, 14], [15, 151643]]
    reasons = ['eos', 'length', 'eos']
    batch = _batch(prompts, completions)

    def labels_for(batch, reasons, cap=3):
        from experiments.maxrl_comp002_pretraining_v1.mechanics import completion_labels
        return completion_labels(batch['input_ids'], batch['attention_mask'], batch['prompt_lengths'],
                                 batch['completion_lengths'], list(s0.EOS_IDS), reasons, cap)
    labels = labels_for(batch, reasons)
    good = q1.label_check(labels, batch, completions, reasons)
    assert good['pass'] and good['eos_rows'] == 2 and good['pad_positions'] == 2 and good['pad_id_is_an_eos_id']
    dropped = labels.clone()
    dropped[2, -1] = -100  # a genuine 151643 EOS treated like the PAD it equals
    assert not q1.label_check(dropped, batch, completions, reasons)['pass']
    leaked = labels.clone()
    assert batch['offsets'] == [0, 0, 2]
    leaked[2, 0] = batch['input_ids'][2, 0]  # a pad (151643, also an EOS ID) labelled
    record = q1.label_check(leaked, batch, completions, reasons)
    assert not record['pass'] and record['pads_masked'] is False
    prompt_labelled = labels.clone()
    prompt_labelled[0, 2] = batch['input_ids'][0, 2]  # the last prompt token
    assert not q1.label_check(prompt_labelled, batch, completions, reasons)['pass']
    no_eos = _batch([[5, 6], [8, 9, 10]], [[1, 2, 3], [4, 5, 6]])
    assert not q1.label_check(labels_for(no_eos, ['length', 'length']), no_eos, [[1, 2, 3], [4, 5, 6]],
                              ['length', 'length'])['pass']
    no_pad = _batch([[5, 6], [8, 9]], [[1, 151645], [4, 151643]])
    record = q1.label_check(labels_for(no_pad, ['eos', 'eos']), no_pad, [[1, 151645], [4, 151643]], ['eos', 'eos'])
    assert record['pad_positions'] == 0 and record['eos_rows'] == 2 and not record['pass']


def _t6(env, loaded, blocks=(1, 1)):
    return q1.backward_check(loaded.model, [env.prompts[k] for k in q1.T6_POSITIONS],
                             [loaded.greedy[k]['completion_ids'] for k in q1.T6_POSITIONS], list(blocks),
                             loaded.identity['state_fingerprint'])


def test_t6_backward_only_on_the_inventory(env, loaded):
    record = _t6(env, loaded)
    assert record['pass'] and record['backward_calls'] == 1 and record['trainable_parameters'] == 11
    assert record['inventory']['trainable_numel'] == q1.block_numel(loaded.model, 1, 1)
    assert record['frozen_with_grad'] == [] and record['all_zero_grads'] == 0 and record['grad_norm'] > 0
    assert record['loss'] < 0 and math.isfinite(record['loss'])
    assert all(p.grad is None and not p.requires_grad for p in loaded.model.parameters())
    assert s0.state_fingerprint(loaded.model) == loaded.identity['state_fingerprint']
    stale = q1.backward_check(loaded.model, [env.prompts[k] for k in q1.T6_POSITIONS],
                              [loaded.greedy[k]['completion_ids'] for k in q1.T6_POSITIONS], [1, 1], '0' * 64)
    assert not stale['pass'] and stale['unchanged_since_load'] is False and stale['fingerprint_unchanged']
    wide = _t6(env, loaded, blocks=(0, 1))
    assert wide['pass'] and wide['trainable_parameters'] == 22
    with pytest.raises(ValueError, match='block range outside the model'):
        _t6(env, loaded, blocks=(1, 2))


def test_t6_fails_on_a_live_optimizer(env, loaded):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.1)
    record = _t6(env, loaded)
    assert not record['pass'] and record['optimizers_before'] >= 1
    del optimizer


def test_t6_fails_when_a_frozen_parameter_gets_a_gradient(env, loaded, monkeypatch):
    real = q1.inventory_scope

    def leaky(model, first, last):
        record = real(model, first, last)
        model.model.embed_tokens.weight.requires_grad_(True)
        return record
    monkeypatch.setattr(q1, 'inventory_scope', leaky)
    record = _t6(env, loaded)
    assert not record['pass'] and 'model.embed_tokens.weight' in record['frozen_with_grad']
    assert 'lm_head.weight' in record['frozen_with_grad']


def test_t6_fails_on_a_stray_frozen_grad_or_missing_trainable_grads(env, loaded, monkeypatch):
    real = q1.t6_loss
    norm = loaded.model.model.norm.weight

    def stray(values):  # a .grad on a frozen parameter with its requires_grad flag untouched
        norm.grad = torch.zeros_like(norm)
        return real(values)
    monkeypatch.setattr(q1, 't6_loss', stray)
    record = _t6(env, loaded)
    assert not record['pass'] and record['frozen_with_grad'] == ['model.norm.weight']
    assert record['requires_grad_drift'] == [] and record['missing_grads'] == []
    monkeypatch.setattr(q1, 't6_loss', lambda values: real(values).detach().requires_grad_())
    record = _t6(env, loaded)
    assert not record['pass'] and len(record['missing_grads']) == 11 and record['frozen_with_grad'] == []
    assert all(p.grad is None for p in loaded.model.parameters())


def test_t6_fails_on_nonfinite_gradients_and_on_a_state_change(env, loaded, monkeypatch):
    real = q1.t6_loss
    monkeypatch.setattr(q1, 't6_loss', lambda values: real(values) * math.inf)
    record = _t6(env, loaded)
    assert not record['pass'] and record['nonfinite_grads'] and record['fingerprint_unchanged']
    frozen = loaded.model.model.layers[0].mlp.down_proj.weight  # below the inventory: nothing saves it for backward
    saved = frozen.detach().clone()

    def mutating(values):
        with torch.no_grad():
            frozen.mul_(2.0)
        return real(values)
    monkeypatch.setattr(q1, 't6_loss', mutating)
    try:
        record = _t6(env, loaded)
        assert not record['pass'] and record['fingerprint_unchanged'] is False and record['frozen_with_grad'] == []
    finally:
        with torch.no_grad():
            frozen.copy_(saved)
    assert s0.state_fingerprint(loaded.model) == loaded.identity['state_fingerprint']


def test_t6_scope_on_the_registered_architecture():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    if not (s0.MODEL_DIR / 'config.json').is_file():
        pytest.skip('Supa config absent')
    config = Qwen3Config(**json.loads((s0.MODEL_DIR / 'config.json').read_text(encoding='utf-8')))
    with torch.device('meta'):
        model = Qwen3ForCausalLM(config)
    assert q1.block_numel(model, *q1.REGISTERED['blocks']) == q1.REGISTERED['trainable_numel'] == 314618880
    assert q1.block_numel(model, 14, 27) == 220233216
    assert sum(1 for name, _ in model.named_parameters() if name.startswith(q1.block_prefixes(8, 27))) == 20 * 11


def test_t7_records_resources_and_tolerates_an_unavailable_counter(monkeypatch):
    host = FakeHost()
    failing = lambda: q1.gpu_shared_usage(runner=fake_runner(returncode=1, stderr='counter missing'))  # noqa: E731
    record = q1.resources_check('cpu', host, failing)
    assert record['pass'] and record['gpu_shared_usage']['available'] is False
    assert 'counter missing' in record['gpu_shared_usage']['error']['message']
    assert record['peak_cuda_reserved_gib'] is None and record['free_phys_gib'] == 3.5
    real = s0.resources
    monkeypatch.setattr(s0, 'resources', lambda device, module=None: dict(real(device, module=module),
                                                                          process_peak_rss_gib=None))
    assert not q1.resources_check('cpu', host, fake_counter)['pass']
    assert not torch.cuda.is_initialized()


def test_gpu_shared_usage_filters_this_pid_and_records_failures():
    calls = []
    record = q1.gpu_shared_usage(123, runner=fake_runner(json.dumps(_samples(123)), calls=calls))
    assert record['available'] is True and record['error'] is None
    assert record['process_shared_usage_bytes'] == 4096.0 and record['process_dedicated_usage_bytes'] == 8192.0
    assert list(record['process_instances']) == ['pid_123_luid_0x0_0x1099f_phys_0']  # not pid_1231_...
    assert record['adapter_shared_usage_bytes'] == {'luid_0x0_0x1099f_phys_0': 65536.0}
    argv = calls[0][0]
    assert argv[:4] == ['powershell.exe', '-NoProfile', '-NonInteractive', '-Command']
    assert all(path in argv[4] for path in q1.GPU_COUNTERS) and '"' not in argv[4]
    single = q1.gpu_shared_usage(123, runner=fake_runner(json.dumps(_samples(123)[3])))
    assert single['available'] is True and single['process_shared_usage_bytes'] is None
    for runner in (fake_runner(returncode=1, stderr='boom'), fake_runner('not json'), fake_runner('[]')):
        failed = q1.gpu_shared_usage(123, runner=runner)
        assert failed['available'] is False and failed['error'] is not None

    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs['timeout'])
    assert q1.gpu_shared_usage(123, runner=timeout)['error']['type'] == 'TimeoutExpired'


@pytest.mark.skipif(sys.platform != 'win32', reason='Windows performance counters')
def test_gpu_shared_usage_real_counter_query_is_best_effort():
    record = q1.gpu_shared_usage()
    assert record['available'] in (True, False)
    if record['available']:
        assert record['adapter_shared_usage_bytes'] and all(v >= 0 for v in record['adapter_shared_usage_bytes'].values())
    json.dumps(record, allow_nan=False)


def test_optimizer_census_counts_live_optimizers():
    before = q1.optimizer_census()
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.zeros(2))], lr=1e-6)
    assert q1.optimizer_census() == before + 1
    del optimizer
    import gc
    gc.collect()
    assert q1.optimizer_census() == before


def test_receipt_values_are_strict_json():
    value = q1._json_safe({'a': float('nan'), 'b': [math.inf, -math.inf, 1.5], 'c': Path('x'), 'd': (1, True)})
    assert value == {'a': 'nan', 'b': ['inf', '-inf', 1.5], 'c': 'x', 'd': [1, True]}
    json.dumps(value, allow_nan=False)
