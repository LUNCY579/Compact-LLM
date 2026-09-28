"""S0-P on-policy throughput pilot (Amendment 1 section 16, clarifications C10 and C11): selection, per-sample
seeds and sequential sampling, the strict-body response rule, the MaxRL objective and its per-row gradient
accumulation, inventory, optimizer, the section 16 decision, and the whole CLI on CPU.

CPU only: no test creates a CUDA tensor or initialises CUDA (asserted after every test), no test takes the real
GPU lease or reads the real host budget (fakes are injected; CUDA-only calls are replaced by fakes), and the real
Supa weights are never opened. The model is a tiny random Qwen3 (vocab 151936, hidden 32, 2 layers, heads 4/2,
head_dim 8, intermediate 64, tied) whose config.json is the real Supa config with the sizes overridden, saved as
BF16 safetensors in tmp_path with a Q1-format asset receipt. The real inputs read are the Supa tokenizer/config
files, manifest.json, manifests/atomic_R.json and canonical/train.jsonl.
"""
from __future__ import annotations

import contextlib
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from experiments.maxrl_comp002_pretraining_v1 import s0_competence as comp
from experiments.maxrl_comp002_pretraining_v1 import s0_generate as g
from experiments.maxrl_comp002_pretraining_v1 import s0_pilot as pilot
from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as s0
from experiments.maxrl_comp002_pretraining_v1.preflight import digest, sha
from experiments.maxrl_composable_skills_v2 import corpus, oracle
from experiments.maxrl_verifiable_data_v1 import maxrl

REPO = Path(__file__).resolve().parents[3]
TINY = {'hidden_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
        'head_dim': 8, 'intermediate_size': 64, 'max_window_layers': 2}
IM_END, ENDOFTEXT, IM_START = 151645, 151643, 151644
PAD = ENDOFTEXT
# The registered selection (sections 16 S0-P): pinned so a rule change shows; recomputed independently below.
PINNED_SELECTION_DIGEST = '776f8932b2a2d7c3c90460d3924bc91a6820b202ad5fb0b98d750a298322a932'
PINNED_FIRST_IDS = ['MC2-c05354fed5d3eb49e16f1daf7826349b5fbd22f7a3a1d29b06c58d1af0f708a4',
                    'MC2-a4fa6cc630c4e68f67ab3c968b75d89e3d84c358b54ed2ef15d35328aeba4b57',
                    'MC2-123dfd2e8d1d5155aedef1e04dd0dbdf7f98ad612541beb1e624a9d54eb5ad0e']
PROMPT = [9707, 1879, 11, 1034, 525, 3555]
COMPLETIONS = [[5, 6, IM_END], [7, 8, 9, 10], [11, IM_END], [12, 13, ENDOFTEXT]]


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
def no_optimizer_outlives_the_module():
    """The pilot and these tests leave no torch.optim.Optimizer alive (the Q1 T6 census counts every live one)."""
    yield
    gc.collect()
    live = sum(1 for obj in gc.get_objects() if issubclass(type(obj), torch.optim.Optimizer))
    assert live == 0, 'optimizer objects outlived their tests: %d' % live


@pytest.fixture(scope='module')
def real_config():
    if not (s0.MODEL_DIR / 'config.json').is_file() or not (s0.MODEL_DIR / 'tokenizer.json').is_file():
        pytest.skip('Supa tokenizer/config files absent: ' + str(s0.MODEL_DIR))
    return json.loads((s0.MODEL_DIR / 'config.json').read_text(encoding='utf-8'))


def _create(path: Path, data: bytes) -> None:
    with open(path, 'xb') as stream:
        stream.write(data)


@pytest.fixture(scope='module')
def tiny_dir(tmp_path_factory, real_config):
    """A tiny random Qwen3 in the real layout (BF16, no lm_head) and its Q1-format asset receipt."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    root = tmp_path_factory.mktemp('pilot_tiny')
    model_dir = root / 'tiny'
    model_dir.mkdir()
    config = dict(real_config, **TINY)
    _create(model_dir / 'config.json', (json.dumps(config, indent=2, sort_keys=True) + '\n').encode('ascii'))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260928)
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
    asset = root / 'asset_receipt.json'
    _create(asset, (json.dumps(receipt, indent=2, sort_keys=True) + '\n').encode('ascii'))
    return SimpleNamespace(root=model_dir, asset=asset, config=config)


@pytest.fixture
def tiny_model(tiny_dir):
    model, _ = s0.load_model_fp32(tiny_dir.root, 'cpu', tiny_dir.asset)
    return model


@pytest.fixture(scope='module')
def tokenizer(real_config):
    return s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)


@pytest.fixture(scope='module')
def selected(comp_rs_data):
    rows, inputs = pilot.load_pilot_rows(comp_rs_data, 25)
    return SimpleNamespace(rows=rows, inputs=inputs)


class FakeLease:
    def __init__(self):
        self.LOCK = Path('Z:/fake-temp') / s0.LOCK_NAME
        self.events = []

    def acquire(self, timeout_s, label):
        self.events.append(('acquire', label))
        return 'token-1'

    def release(self, token):
        self.events.append(('release', token))


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


@pytest.fixture
def mixed_reward(monkeypatch):
    """The first scored completion of each group is rewarded and the rest are not, so every group is mixed."""
    calls, counts = [], {}

    def fake(row, response):
        counts[row['id']] = counts.get(row['id'], 0) + 1
        calls.append((row['id'], response))
        return int(counts[row['id']] == 1)
    monkeypatch.setattr(pilot, 'reward', fake)
    return calls


def run_cli(tmp_path, tiny_dir, data, *extra, inventory='A', lease=None, host=None, groups=2, size=3, tokens=4):
    out = tmp_path / 'records' / ('S0_PILOT_%s.json' % inventory)
    argv = ['--inventory', inventory, '--device', 'cpu', '--test-mode', '--out', str(out), '--data', str(data),
            '--groups', str(groups), '--group-size', str(size), '--max-new-tokens', str(tokens),
            '--test-blocks', '1', '1', '--model-dir', str(tiny_dir.root), '--asset-receipt', str(tiny_dir.asset),
            '--host-floor-gib', '0.5', *extra]
    lease = FakeLease() if lease is None else lease
    host = FakeHost() if host is None else host
    code = pilot.main(argv, lease_module=lease, host_module=host)
    raw = out.read_bytes()
    return SimpleNamespace(code=code, receipt=json.loads(raw.decode('ascii')), raw=raw, out=out, argv=argv,
                           lease=lease, host=host)


def _files(root: Path) -> dict:
    return {str(p.relative_to(root)): sha(p) for p in sorted(root.rglob('*')) if p.is_file()}


# Every audited open with a write/create flag while recording (bytecode caches excluded): the CLI may create
# exactly its receipt. The hook stays for the process but is inert unless a recording is active.
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_APPEND | os.O_TRUNC
_WRITES = None


def _record_writes(event, args):
    if _WRITES is None or event != 'open' or len(args) < 3:
        return
    path, flags = args[0], args[2]
    if isinstance(path, int) or not isinstance(flags, int) or not flags & _WRITE_FLAGS:
        return
    try:
        name = os.path.normcase(os.path.abspath(os.fsdecode(path)))
    except (TypeError, ValueError):
        return
    if '__pycache__' not in Path(name).parts:
        _WRITES.append(name)


sys.addaudithook(_record_writes)


@contextlib.contextmanager
def recording_writes():
    global _WRITES
    _WRITES = []
    try:
        yield _WRITES
    finally:
        _WRITES = None


def _sums(model) -> list:
    """Individual (unpadded, one row per call) teacher-forced logprob sums of COMPLETIONS after PROMPT."""
    with torch.no_grad():
        return [float(g.teacher_forced_logprobs(model, [PROMPT], [c], PAD, padding_side='none')[0].sum())
                for c in COMPLETIONS]


def _independent_sample_seed(row_id: str, k: int) -> int:
    import hashlib
    return int(hashlib.sha256(('COMP-RS-001-S0-PILOT-SAMPLE|%s|%d' % (row_id, k)).encode()).hexdigest()[:16],
               16) % 2 ** 63


class CapturingOptimizer:
    """Stands in for AdamW in update_step: step() snapshots the accumulated gradients and changes nothing."""

    def __init__(self, model):
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.captured, self.events = None, []

    def zero_grad(self, set_to_none=True):
        self.events.append('zero_grad')
        for _, p in self.params:
            p.grad = None

    def step(self):
        self.events.append('step')
        self.captured = {n: p.grad.detach().clone() for n, p in self.params}


# ------------------------------------------------------------------------------------------------ selection
def test_selection_is_the_sorted_salted_hash_prefix_whatever_the_input_order():
    import hashlib
    ids = ['MC2-%04d' % i for i in range(40)]
    expected = sorted(ids, key=lambda i: hashlib.sha256(('COMP-RS-001-S0-PILOT|' + i).encode()).hexdigest())
    assert pilot.select_pilot_ids(ids, 25) == expected[:25]
    assert pilot.select_pilot_ids(list(reversed(ids)), 25) == expected[:25]
    assert pilot.select_pilot_ids(ids[::3] + ids[1::3] + ids[2::3], 25) == expected[:25]
    assert pilot.select_pilot_ids(ids, 2) == expected[:2]   # a test-mode size is a prefix of the registered 25
    assert expected[:25] != ids[:25]
    for bad_ids, n in ((ids + ids[:1], 25), (ids, 0), (ids, 41), (ids + [''], 25), (tuple(ids), 25)):
        with pytest.raises(ValueError):
            pilot.select_pilot_ids(bad_ids, n)


def test_seed_is_the_s0c_integer_form_under_the_pilot_seed_salt():
    import hashlib
    ids = ['MC2-%04d' % i for i in range(50)]
    seeds = [pilot.pilot_seed(i) for i in ids]
    for row_id, seed in zip(ids, seeds):
        full = hashlib.sha256(('COMP-RS-001-S0-PILOT-SEED|' + row_id).encode()).hexdigest()
        assert seed == int(full[:16], 16) % 2 ** 63 and 0 <= seed < 2 ** 63
        assert seed != int(hashlib.sha256(('COMP-RS-001-S0-PILOT|' + row_id).encode()).hexdigest()[:16], 16) % 2 ** 63
    assert len(set(seeds)) == len(seeds)


def test_sample_seeds_are_the_c10_per_sample_form():
    """C10: sample k of a group is seeded with sha256('COMP-RS-001-S0-PILOT-SAMPLE|<id>|<k>')[:16] mod 2^63."""
    import hashlib
    ids = ['MC2-%04d' % i for i in range(20)]
    everything = []
    for row_id in ids:
        seeds = pilot.sample_seeds(row_id, 8)
        assert seeds == [_independent_sample_seed(row_id, k) for k in range(8)]
        assert seeds == [pilot.sample_seed(row_id, k) for k in range(8)] and all(0 <= s < 2 ** 63 for s in seeds)
        assert pilot.sample_seeds(row_id, 3) == seeds[:3]            # a test-mode G is a prefix of the registered 8
        assert pilot.pilot_seed(row_id) not in seeds                 # the group seed is not one of them
        everything += seeds
    assert len(set(everything)) == len(everything)
    assert pilot.sample_seed('MC2-a', 0) == int(hashlib.sha256(
        b'COMP-RS-001-S0-PILOT-SAMPLE|MC2-a|0').hexdigest()[:16], 16) % 2 ** 63
    for bad in (('', 0), ('MC2-a', -1), ('MC2-a', True), ('MC2-a', 1.0), (None, 0)):
        with pytest.raises(ValueError):
            pilot.sample_seed(*bad)
    with pytest.raises(ValueError):
        pilot.sample_seeds('MC2-a', 0)


def test_registered_selection_over_the_atomic_R_manifest(selected, comp_rs_data):
    import hashlib
    plan = json.loads((comp_rs_data / 'manifests' / 'atomic_R.json').read_text(encoding='utf-8'))
    independent = sorted(plan['ids'], key=lambda i: hashlib.sha256(
        ('COMP-RS-001-S0-PILOT|' + i).encode('utf-8')).hexdigest())[:25]
    inputs = selected.inputs
    assert inputs['selected_ids'] == independent and len(independent) == 25 and plan['rows'] == 3000
    assert inputs['selection_digest'] == digest(independent) == PINNED_SELECTION_DIGEST
    assert independent[:3] == PINNED_FIRST_IDS
    assert [row['id'] for row in selected.rows] == independent
    manifest = json.loads((comp_rs_data / 'manifest.json').read_text(encoding='utf-8'))
    assert inputs['train_sha256'] == manifest['files']['canonical/train.jsonl']['sha256']
    assert inputs['atomic_R_manifest_sha256'] == manifest['files']['manifests/atomic_R.json']['sha256']
    for row in selected.rows:
        assert row['role'] == 'train' and row['tag'] == 'core' and row['skills'] == ['R']
        assert [m['role'] for m in row['prompt']] == ['system', 'user']
        pilot._check_row(row)
    row = selected.rows[0]
    for edit in ({'skills': ['S']}, {'role': 'calibration'}, {'tag': 'boundary'}, {'prompt_hash': '0' * 64},
                 {'prompt': row['prompt'][:1]}):
        with pytest.raises(ValueError):
            pilot._check_row(dict(row, **edit))


def test_the_loader_checks_every_selected_row(comp_rs_data, monkeypatch):
    seen, real = [], pilot._check_row
    monkeypatch.setattr(pilot, '_check_row', lambda row: seen.append(row['id']) or real(row))
    rows, inputs = pilot.load_pilot_rows(comp_rs_data, 3)
    assert seen == inputs['selected_ids'] == [row['id'] for row in rows] == PINNED_FIRST_IDS


def test_load_pilot_rows_refuses_drifted_inputs(tmp_path, comp_rs_data, selected):
    """A root holding the real manifests but a train.jsonl of only the selected rows: every row is there, so only
    the byte check refuses it."""
    root = tmp_path / 'fake_root'
    (root / 'manifests').mkdir(parents=True)
    (root / 'canonical').mkdir()
    _create(root / 'manifest.json', (comp_rs_data / 'manifest.json').read_bytes())
    plan = (comp_rs_data / 'manifests' / 'atomic_R.json').read_bytes()
    _create(root / 'manifests' / 'atomic_R.json', plan)
    _create(root / 'canonical' / 'train.jsonl',
            b''.join(json.dumps(row, sort_keys=True).encode('utf-8') + b'\n' for row in selected.rows))
    with pytest.raises(ValueError, match='train.jsonl drift'):
        pilot.load_pilot_rows(root, 25)
    (root / 'manifests' / 'atomic_R.json').write_bytes(plan.replace(b'"atomic_R"', b'"atomic_S"'))
    with pytest.raises(ValueError, match='atomic_R manifest drift'):
        pilot.load_pilot_rows(root, 25)
    (root / 'manifest.json').write_bytes((comp_rs_data / 'manifest.json').read_bytes() + b' ')
    with pytest.raises(ValueError, match='dataset manifest drift'):
        pilot.load_pilot_rows(root, 25)


# ------------------------------------------------------------------------------------------------ responses
def _completion(ids, reason):
    return {'completion_ids': ids, 'stop_reason': reason, 'generated_tokens': len(ids),
            'raw_step_logprobs': [-0.5] * len(ids)}


def test_response_rule_is_the_strict_body_of_s0c_and_s0e(tokenizer, selected):
    """C11: the scorer's response is the strict body (s0_competence.RESPONSE_RULE); the lenient text is scored
    only as the lenient_reward diagnostic."""
    assert pilot.RESPONSE_RULE == comp.RESPONSE_RULE
    row = selected.rows[0]
    answer = oracle.answers(row['verifier']['spec'])[0]
    assert json.loads(answer).keys() == {'answer'} and corpus.reward(row, answer) == 1
    ids = tokenizer(answer, add_special_tokens=False)['input_ids']
    for eos in (IM_END, ENDOFTEXT):
        text = pilot.response_text(tokenizer, ids + [eos], 'eos')
        assert text == {'response': answer, 'body_special_ids': [], 'body_unknown_ids': [], 'lenient_response': None}
        record = pilot._completion_record(row, _completion(ids + [eos], 'eos'), text)
        assert record['reward'] == 1 and record['lenient_response'] is None and record['lenient_reward'] is None
        assert 'strict_body_reward' not in record
    # the verifier sees content only: a length stop keeps its reward, the stop reason is recorded separately
    assert pilot._completion_record(row, _completion(ids, 'length'),
                                    pilot.response_text(tokenizer, ids, 'length'))['reward'] == 1
    # why the terminal EOS is removed (by position): its text left in fails strict_json
    kept = tokenizer.decode(ids + [IM_END], skip_special_tokens=False, clean_up_tokenization_spaces=False)
    assert kept == answer + '<|im_end|>' and corpus.reward(row, kept) == 0
    wrong = tokenizer('{"answer":"UNKNOWN"}', add_special_tokens=False)['input_ids'] + [IM_END]
    assert pilot._completion_record(row, _completion(wrong, 'eos'),
                                    pilot.response_text(tokenizer, wrong, 'eos'))['reward'] == 0
    # a special token spliced inside the body keeps its text: strict 0, and only the lenient diagnostic scores 1
    spliced = ids[:1] + [IM_START] + ids[1:] + [IM_END]
    text = pilot.response_text(tokenizer, spliced, 'eos')
    assert '<|im_start|>' in text['response'] and text['response'] != answer
    assert text['body_special_ids'] == [IM_START] and text['lenient_response'] == answer
    record = pilot._completion_record(row, _completion(spliced, 'eos'), text)
    assert record['reward'] == 0 and record['lenient_reward'] == 1 and record['lenient_response'] == answer
    # an ID the tokenizer cannot render becomes a visible placeholder: strict 0, lenient (dropped by decode) 1
    hole = len(tokenizer) + 31
    unknown = ids + [hole, IM_END]
    text = pilot.response_text(tokenizer, unknown, 'eos')
    assert text['response'] == answer + '<|unrenderable:%d|>' % hole and text['body_unknown_ids'] == [hole]
    record = pilot._completion_record(row, _completion(unknown, 'eos'), text)
    assert record['reward'] == 0 and record['lenient_reward'] == 1
    # the pilot's rule is s0_competence's, case by case
    for case, reason in ((ids + [IM_END], 'eos'), (ids, 'length'), (spliced, 'eos'), (unknown, 'eos'),
                         (wrong, 'eos'), ([IM_END], 'eos')):
        assert pilot.response_text(tokenizer, case, reason) == comp.response_text(tokenizer, case, reason)
    for bad, reason in ((ids, 'eos'), (ids[:1] + [IM_END] + ids[1:] + [IM_END], 'eos'), (ids + [IM_END], 'length'),
                        (ids, 'stop'), ([], 'length')):
        with pytest.raises(ValueError):
            pilot.response_text(tokenizer, bad, reason)


def test_response_text_delegates_to_the_hashed_s0_competence(tokenizer, monkeypatch):
    calls = []

    def spy(tok, completion_ids, stop_reason):
        calls.append((list(completion_ids), stop_reason))
        return {'response': 'x', 'body_special_ids': [], 'body_unknown_ids': [], 'lenient_response': None}
    monkeypatch.setattr(comp, 'response_text', spy)
    assert pilot.response_text(tokenizer, [5, IM_END], 'eos')['response'] == 'x' and calls == [([5, IM_END], 'eos')]
    assert 's0_competence.py' in pilot.code_sha256()
    assert pilot.code_sha256()['s0_competence.py'] == sha(Path(comp.__file__))
    monkeypatch.setattr(pilot, 'RESPONSE_RULE', pilot.RESPONSE_RULE + ' (drifted)')
    with pytest.raises(ValueError, match='response rule differs'):
        pilot.response_text(tokenizer, [5, IM_END], 'eos')
    monkeypatch.undo()
    monkeypatch.setitem(pilot.CODE_FILES, 's0_competence.py', Path(pilot.__file__).resolve())
    with pytest.raises(ValueError, match='not the hashed file'):
        pilot.response_text(tokenizer, [5, IM_END], 'eos')


def test_advantages_are_the_inherited_maxrl_estimator():
    assert pilot.advantages is maxrl.advantages
    values = pilot.advantages([1, 0, 0, 0])
    assert values == [(1 - 0.25) / (0.25 + 1e-6)] + [(0 - 0.25) / (0.25 + 1e-6)] * 3
    assert values[0] > 0 > values[1]
    for uniform in ([0] * 8, [1] * 8):
        assert pilot.advantages(uniform) == [0.0] * 8
    assert [pilot.group_outcome(r) for r in ([0, 0], [1, 0], [1, 1])] == ['all_fail', 'mixed', 'all_success']
    with pytest.raises(ValueError):
        pilot.group_outcome([1, True])


# ------------------------------------------------------------------------------------------------ objective
def test_group_loss_is_the_registered_objective_with_its_gradient(tiny_model):
    pilot.set_inventory(tiny_model, 1, 1)
    advantage = maxrl.advantages([1, 0, 0, 0])
    reference = _sums(tiny_model)
    tiny_model.train()
    with torch.enable_grad():
        loss, logprobs = pilot.group_loss(tiny_model, PROMPT, COMPLETIONS, advantage)
        loss.backward()
    assert [len(v) for v in logprobs] == [len(c) for c in COMPLETIONS]   # EOS included, pads excluded
    assert all(abs(float(v.detach().sum()) - r) <= 1e-4 for v, r in zip(logprobs, reference))
    assert abs(float(loss.detach()) - (-math.fsum(a * s for a, s in zip(advantage, reference)) / 4)) <= 1e-4
    trainable = {n: p for n, p in tiny_model.named_parameters() if p.requires_grad}
    assert all(p.grad is None for n, p in tiny_model.named_parameters() if n not in trainable)
    grads = {n: p.grad.clone() for n, p in trainable.items()}
    tiny_model.zero_grad(set_to_none=True)
    expected = {n: torch.zeros_like(v) for n, v in grads.items()}
    scale = {n: torch.zeros_like(v) for n, v in grads.items()}   # sum of |per-row terms|: the cancellation scale
    for a, completion in zip(advantage, COMPLETIONS):
        with torch.enable_grad():
            g.teacher_forced_logprobs(tiny_model, [PROMPT], [completion], PAD, padding_side='none')[0].sum().backward()
        for n, p in trainable.items():
            expected[n] -= (a / 4) * p.grad
            scale[n] += (a / 4) * p.grad.abs() * (1 if a > 0 else -1)
            p.grad = None
    for n in grads:
        assert bool(((grads[n] - expected[n]).abs() <= 1e-4 * scale[n] + 1e-8).all()), n
        assert float((grads[n] - expected[n]).abs().max()) < 1e-2 * float(scale[n].max())
    with pytest.raises(ValueError, match='group advantages'):
        pilot.group_loss(tiny_model, PROMPT, COMPLETIONS, [1, 0, 0, 0])   # ints, not the estimator's floats


def test_update_step_accumulates_the_group_loss_gradient_one_unpadded_row_at_a_time(tiny_model, monkeypatch):
    """C10: one unpadded row per micro-batch, each backward before the next row's forward, one step after all G;
    the gradient at the step equals the single-batch group_loss gradient."""
    pilot.set_inventory(tiny_model, 1, 1)
    pilot.enable_checkpointing(tiny_model)
    advantage = maxrl.advantages([1, 0, 0, 0])
    reference_sums = _sums(tiny_model)
    tiny_model.train()
    with torch.enable_grad():
        loss, _ = pilot.group_loss(tiny_model, PROMPT, COMPLETIONS, advantage)
        loss.backward()
    trainable = [(n, p) for n, p in tiny_model.named_parameters() if p.requires_grad]
    reference = {n: p.grad.detach().clone() for n, p in trainable}
    reference_loss = float(loss.detach())
    del loss
    tiny_model.zero_grad(set_to_none=True)
    tiny_model.eval()
    forced, real_forced = [], g.teacher_forced_logprobs

    def spy(model, prompts, completions, pad, **kwargs):
        forced.append({'rows': len(prompts), 'completion': list(completions[0]), 'kwargs': kwargs,
                       'grad_mode': torch.is_grad_enabled(), 'training': model.training,
                       'grads_so_far': sum(p.grad is not None for _, p in trainable)})
        return real_forced(model, prompts, completions, pad, **kwargs)
    monkeypatch.setattr(g, 'teacher_forced_logprobs', spy)
    optimizer = CapturingOptimizer(tiny_model)
    step = pilot.update_step(tiny_model, optimizer, PROMPT, COMPLETIONS, advantage)
    # one unpadded row per call, in sample order, under grad and train mode
    assert [f['rows'] for f in forced] == [1] * 4 and [f['completion'] for f in forced] == COMPLETIONS
    assert all(f['kwargs'] == {'padding_side': 'none', 'checkpoint_head': True} for f in forced)
    assert all(f['grad_mode'] and f['training'] for f in forced)
    # each row's backward ran before the next row's forward: gradients exist from the second call on
    assert [f['grads_so_far'] for f in forced] == [0] + [len(trainable)] * 3
    assert optimizer.events == ['zero_grad', 'step', 'zero_grad']   # exactly one step, after all G backwards
    assert step['micro_batches'] == 4 and len(step['sample_losses']) == 4
    assert step['loss'] == math.fsum(step['sample_losses'])
    for value, a, s in zip(step['sample_losses'], advantage, reference_sums):
        assert abs(value - (-(a / 4) * s)) <= 1e-4
    assert abs(step['loss'] - reference_loss) <= 1e-4
    assert all(abs(x - r) <= 1e-4 for x, r in zip(step['logprob_sums'], reference_sums))
    # the accumulated gradient is the single-batch gradient
    gaps = {n: float((optimizer.captured[n] - reference[n]).abs().max()) for n in reference}
    assert max(gaps.values()) <= 1e-5, gaps
    assert max(float(v.abs().max()) for v in reference.values()) > 1e-3   # a real gradient, not two zeros
    assert tiny_model.training is False and all(p.grad is None for p in tiny_model.parameters())


def test_accumulation_needs_grad_mode_and_estimator_advantages(tiny_model):
    pilot.set_inventory(tiny_model, 1, 1)
    with torch.no_grad(), pytest.raises(ValueError, match='no autograd graph'):
        pilot.accumulate_gradients(tiny_model, PROMPT, COMPLETIONS, maxrl.advantages([1, 0, 0, 0]))
    with pytest.raises(ValueError, match='group advantages'):
        pilot.accumulate_gradients(tiny_model, PROMPT, COMPLETIONS, [1, 0, 0, 0])
    with pytest.raises(ValueError, match='group advantages'):
        pilot.update_step(tiny_model, CapturingOptimizer(tiny_model), PROMPT, COMPLETIONS[:1], [1.0])
    assert all(p.grad is None for p in tiny_model.parameters())


def test_one_registered_step_raises_the_advantage_weighted_logprob(tiny_model):
    pilot.set_inventory(tiny_model, 1, 1)
    pilot.enable_checkpointing(tiny_model)
    tiny_model.eval()
    optimizer = pilot.build_optimizer(tiny_model)
    advantage = maxrl.advantages([1, 0, 0, 0])
    before = _sums(tiny_model)
    step = pilot.update_step(tiny_model, optimizer, PROMPT, COMPLETIONS, advantage)
    after = _sums(tiny_model)
    assert abs(step['loss'] - (-math.fsum(a * s for a, s in zip(advantage, before)) / 4)) <= 1e-4
    change = math.fsum(a * (x - y) for a, x, y in zip(advantage, after, before))
    assert change > 1e-4                   # the loss fell by far more than float32 noise (~4e-6 here)
    assert after[0] - before[0] > 1e-5     # the rewarded completion became more likely


def test_update_changes_exactly_the_inventory_with_checkpointing_on(tiny_model, monkeypatch):
    inventory = pilot.set_inventory(tiny_model, 1, 1)
    assert pilot.enable_checkpointing(tiny_model)['enabled'] is True
    tiny_model.eval()
    optimizer = pilot.build_optimizer(tiny_model)
    forced, real_forced = [], g.teacher_forced_logprobs

    def spy(*args, **kwargs):
        forced.append(kwargs)
        return real_forced(*args, **kwargs)
    monkeypatch.setattr(g, 'teacher_forced_logprobs', spy)
    calls = []
    for index, layer in enumerate(tiny_model.model.layers):
        real = layer._gradient_checkpointing_func

        def counted(*args, _real=real, _index=index, **kwargs):
            calls.append(_index)
            return _real(*args, **kwargs)
        layer._gradient_checkpointing_func = counted
    params = {n: p.detach().clone() for n, p in tiny_model.named_parameters(remove_duplicate=False)}
    buffers = {n: b.detach().clone() for n, b in tiny_model.named_buffers()}
    before = pilot.split_fingerprints(tiny_model, inventory['trainable_names'])
    sampled = pilot.sample_sequential(g, tiny_model, PROMPT, [7, 8], 2)
    assert calls == [] and len(sampled['completions']) == 2   # generation: eval mode, no checkpointing
    pilot.update_step(tiny_model, optimizer, PROMPT, COMPLETIONS, maxrl.advantages([1, 0, 0, 1]))
    assert calls == [0, 1] * 4                                  # every row's forward ran every layer checkpointed
    # one unpadded row per micro-batch (C10), the head at completion positions only
    assert forced == [{'padding_side': 'none', 'checkpoint_head': True}] * 4
    assert tiny_model.training is False and all(p.grad is None for p in tiny_model.parameters())
    trainable = set(inventory['trainable_names'])
    assert trainable == {n for n in params if n.startswith('model.layers.1.')}
    for name, value in tiny_model.named_parameters(remove_duplicate=False):
        changed = not torch.equal(value.detach(), params[name])
        assert changed == (name in trainable), name
    assert all(torch.equal(b, buffers[n]) for n, b in tiny_model.named_buffers())
    after = pilot.split_fingerprints(tiny_model, inventory['trainable_names'])
    assert after['frozen'] == before['frozen'] and after['trainable'] != before['trainable']
    assert after['names'] == before['names'] == before['frozen_names'] + before['trainable_names']
    state = pilot.optimizer_state(optimizer, torch.device('cpu'))
    assert state['on_model_device'] and state['devices'] == ['cpu'] and state['tensors'] == 3 * len(trainable)
    frozen_state = pilot.split_fingerprints(tiny_model, inventory['trainable_names'])
    with pytest.raises(ValueError, match='all advantages are zero'):
        pilot.update_step(tiny_model, optimizer, PROMPT, COMPLETIONS, [0.0] * 4)
    assert pilot.split_fingerprints(tiny_model, inventory['trainable_names']) == frozen_state


def test_sequential_sampling_is_g_seeded_single_sample_calls(tiny_model, monkeypatch):
    """C10: sample k is exactly sample_group(model, prompt, 1, seed_k); the draw reproduces from its seed."""
    tiny_model.eval()
    seeds = pilot.sample_seeds('MC2-unit', 3)
    calls, real = [], g.sample_group

    def spy(model, prompt_ids, G, seed, **kwargs):
        calls.append((list(prompt_ids), G, seed, kwargs))
        return real(model, prompt_ids, G, seed, **kwargs)
    monkeypatch.setattr(g, 'sample_group', spy)
    first = pilot.sample_sequential(g, tiny_model, PROMPT, seeds, 5)
    assert calls == [(PROMPT, 1, seed, {'max_new_tokens': 5}) for seed in seeds]
    monkeypatch.undo()
    again = pilot.sample_sequential(g, tiny_model, PROMPT, seeds, 5)
    assert again == dict(first, wall_s_per_sample=again['wall_s_per_sample'])   # reproducible from the seeds
    assert first['seeds'] == seeds and len(first['completions']) == 3 and len(first['wall_s_per_sample']) == 3
    for seed, completion, steps in zip(seeds, first['completions'], first['decode_steps_per_sample']):
        alone = g.sample_group(tiny_model, PROMPT, 1, seed, max_new_tokens=5)
        assert alone['completions'][0] == completion and alone['decode_steps'] == steps
    assert first['decode_steps'] == sum(first['decode_steps_per_sample'])
    assert first['sampler'] == {k: v for k, v in alone['sampler'].items() if k != 'seed_path'}
    assert first['sampler']['replay_verified'] is True and 'seed_path' not in first['sampler']
    # every call's sampler record must equal the first's
    drift = iter([{}, {'replay_verified': False}])

    def drifting(model, prompt_ids, G, seed, **kwargs):
        out = real(model, prompt_ids, G, seed, **kwargs)
        return dict(out, sampler=dict(out['sampler'], **next(drift)))
    monkeypatch.setattr(g, 'sample_group', drifting)
    with pytest.raises(ValueError, match='sampler record differs'):
        pilot.sample_sequential(g, tiny_model, PROMPT, seeds[:2], 2)
    monkeypatch.setattr(g, 'sample_group', lambda model, prompt_ids, G, seed, **kwargs: real(
        model, prompt_ids, G, seed + 1, **kwargs))
    with pytest.raises(ValueError, match='single-sample call shape'):
        pilot.sample_sequential(g, tiny_model, PROMPT, seeds[:1], 2)


def test_set_inventory_scopes_requires_grad_to_the_blocks(tiny_model, monkeypatch):
    inventory = pilot.set_inventory(tiny_model, 1, 1)
    names = {n for n, p in tiny_model.named_parameters(remove_duplicate=False) if p.requires_grad}
    assert names == set(inventory['trainable_names']) and all(n.startswith('model.layers.1.') for n in names)
    for frozen in ('model.embed_tokens.weight', 'lm_head.weight', 'model.norm.weight',
                   'model.layers.0.mlp.down_proj.weight'):
        assert not dict(tiny_model.named_parameters(remove_duplicate=False))[frozen].requires_grad
    assert inventory['trainable_numel'] == pilot.block_numel(tiny_model, 1, 1) == sum(
        p.numel() for n, p in tiny_model.named_parameters() if n.startswith('model.layers.1.'))
    both = pilot.set_inventory(tiny_model, 0, 1)
    assert both['trainable_numel'] > inventory['trainable_numel']
    assert pilot.set_inventory(tiny_model, 1, 1)['inventory_sha256'] == inventory['inventory_sha256']
    for first, last in ((0, 2), (1, 0), (-1, 1)):
        with pytest.raises(ValueError, match='block range'):
            pilot.set_inventory(tiny_model, first, last)
    monkeypatch.setattr(type(tiny_model), 'gradient_checkpointing_enable', lambda self, **kwargs: None)
    with pytest.raises(ValueError, match='gradient checkpointing'):
        pilot.enable_checkpointing(tiny_model)


def test_registered_inventories_count_the_real_architecture(real_config):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    with torch.device('meta'):
        model = Qwen3ForCausalLM(Qwen3Config(**real_config))
    assert model.config.num_hidden_layers == pilot.NUM_LAYERS
    for name, spec in pilot.INVENTORIES.items():
        assert pilot.block_numel(model, *spec['blocks']) == spec['numel'], name
    total = sum(p.numel() for p in model.parameters())   # tied head counted once
    # section 16: inventory A's FP32 parameters + gradients + both Adam moments = 5.74 GiB
    assert round((total + 3 * pilot.INVENTORIES['A']['numel']) * 4 / 1024 ** 3, 2) == 5.74


def test_optimizer_is_adamw_lr_1e6_wd0_foreach_off_with_device_state(tiny_model):
    assert pilot.optimizer_kwargs('cuda') == {'lr': 1e-6, 'weight_decay': 0.0, 'foreach': False, 'capturable': True}
    assert pilot.optimizer_kwargs('cpu') == {'lr': 1e-6, 'weight_decay': 0.0, 'foreach': False, 'capturable': False}
    inventory = pilot.set_inventory(tiny_model, 1, 1)
    optimizer = pilot.build_optimizer(tiny_model)
    assert type(optimizer) is torch.optim.AdamW
    record = pilot.optimizer_record(optimizer)
    assert record['cls'] == 'torch.optim.adamw.AdamW' and record['numel'] == inventory['trainable_numel']
    assert {k: record['defaults'][k] for k in ('lr', 'weight_decay', 'foreach', 'capturable', 'betas')} == {
        'lr': 1e-6, 'weight_decay': 0.0, 'foreach': False, 'capturable': False, 'betas': [0.9, 0.999]}
    ids = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert ids == {id(p) for p in tiny_model.parameters() if p.requires_grad}
    json.dumps(record, allow_nan=False)


# ------------------------------------------------------------------------------------------------ decision
def _decide(inventory='A', **kwargs):
    measured = dict(device='cuda', completed=True, oom=False, updates=3, max_reserved_gib=5.0,
                    cap_gib=6.24, peak_rss_gib=2.0, free_at_launch_gib=3.5)
    measured.update(kwargs)
    return pilot.decide(inventory, **measured)


def test_decide_implements_the_section_16_inventory_rule_and_host_check():
    assert _decide()['outcome'] == 'ADOPT_A' and _decide()['inventory_fits'] is True
    assert _decide('B')['outcome'] == 'ADOPT_B'
    assert _decide(max_reserved_gib=0.95 * 6.24)['outcome'] == 'ADOPT_A'            # boundary: <= fits
    assert _decide(max_reserved_gib=0.95 * 6.24 + 1e-9)['outcome'] == 'RERUN_WITH_B'
    assert _decide('B', max_reserved_gib=6.0)['outcome'] == 'STOP_HOST_INFEASIBLE'
    assert _decide(oom=True, completed=False)['outcome'] == 'RERUN_WITH_B'
    assert _decide('B', oom=True, completed=False)['outcome'] == 'STOP_HOST_INFEASIBLE'
    assert _decide(peak_rss_gib=2.5)['outcome'] == 'ADOPT_A'                          # 2.5 + 1.0 <= 3.5
    assert _decide(peak_rss_gib=2.6)['outcome'] == 'STOP_HOST_INFEASIBLE'             # host check first
    assert _decide(peak_rss_gib=2.6, oom=True)['outcome'] == 'STOP_HOST_INFEASIBLE'
    assert _decide(completed=False)['outcome'] == 'INCOMPLETE'
    assert _decide(updates=0)['outcome'] == 'UNMEASURED_NO_UPDATE'
    assert _decide(device='cpu', max_reserved_gib=None, cap_gib=None)['outcome'] == 'NOT_MEASURED_ON_CPU'
    assert _decide(free_at_launch_gib=None)['outcome'] == 'UNMEASURED_HOST'
    record = _decide()
    assert record['threshold_gib'] == 0.95 * 6.24 and record['host_ok'] is True and record['vram_ok'] is True
    with pytest.raises(ValueError):
        _decide('C')


def test_marks_read_before_resetting_and_summarise_the_whole_run(monkeypatch):
    events, readings = [], iter([(4.0, 5.1), (5.9, 6.0), (3.0, 3.2)])

    def fake_resources(device, *, module=None):
        allocated, reserved = next(readings)
        events.append(('read', device))
        return {'host': {}, 'free_phys_gib': 3.0, 'process_rss_gib': 1.1 + allocated / 100,
                'process_peak_rss_gib': 1.5 + allocated / 100,
                'cuda': {'max_memory_allocated_gib': allocated, 'max_memory_reserved_gib': reserved,
                         'total_gib': 8.0, 'fraction_cap_gib': 0.78 * 8.0}}
    monkeypatch.setattr(s0, 'resources', fake_resources)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a, **k: events.append(('sync',) + a))
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda *a, **k: events.append(('reset',) + a))
    marks = pilot.Marks('cuda:0', FakeHost())
    for label in ('load', 'group 0', 'final'):
        assert marks.mark(label)['label'] == label
    assert events == [('sync', 0), ('read', 'cuda:0'), ('reset', 0)] * 3
    summary = marks.summary()
    assert summary['max_memory_reserved_gib'] == 6.0 and summary['max_memory_allocated_gib'] == 5.9
    assert summary['cap_gib'] == 0.78 * 8.0 and summary['threshold_gib'] == 0.95 * 0.78 * 8.0
    assert summary['peak_worker_rss_gib'] == 1.5 + 5.9 / 100 and summary['peak_worker_rss_source'] == 'peak_wset'


def test_cpu_marks_touch_no_cuda(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('CUDA call on the CPU path')
    for name in ('synchronize', 'reset_peak_memory_stats', 'max_memory_reserved', 'mem_get_info'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    marks = pilot.Marks('cpu', FakeHost(free=12.0))
    reading = marks.mark('load')
    assert reading['cuda'] is None and reading['free_phys_gib'] == 12.0 and reading['process_rss_gib'] > 0
    summary = marks.summary()
    assert summary['max_memory_reserved_gib'] is None and summary['cap_gib'] is None
    assert summary['peak_worker_rss_gib'] >= reading['process_rss_gib']


# ------------------------------------------------------------------------------------------------ launch and CLI
def test_the_module_imports_no_torch_so_the_launch_snapshot_precedes_it():
    code = ('import json, sys, types\n'
            'import experiments.maxrl_comp002_pretraining_v1.s0_pilot as p\n'
            "heavy = sorted(m for m in ('torch', 'transformers', 'numpy', 'safetensors') if m in sys.modules)\n"
            "package = 'experiments.maxrl_comp002_pretraining_v1.'\n"
            "lazy = sorted(m for m in ('s0_generate', 's0_competence', 'scoped_scorer')\n"
            '              if package + m in sys.modules)\n'
            "host = types.SimpleNamespace(snapshot=lambda: {'available': True, 'free_phys_gib': 2.0})\n"
            'launch = p.launch_snapshot(host)\n'
            "print(json.dumps({'heavy': heavy, 'lazy': lazy, 'torch_imported': launch['torch_imported'],\n"
            "                  'free': launch['free_phys_gib'], 'rss': launch['process_rss_gib']}))\n")
    run = subprocess.run([sys.executable, '-c', code], cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    result = json.loads(run.stdout.strip().splitlines()[-1])
    assert result['heavy'] == [] and result['torch_imported'] is False and result['free'] == 2.0
    assert result['lazy'] == []   # s0_generate and s0_competence are imported on first use, not at import
    assert 0 < result['rss'] < 0.3   # GiB: the interpreter and the scorer modules only


def test_q1_floor_and_a_pilot_receipts(tmp_path):
    q1 = tmp_path / 'q1.json'
    _create(q1, json.dumps({'stage': 'S0_Q1_FP32', 'resources': {'peak_worker_rss_gib': 2.25}}).encode('ascii'))
    floor = pilot.q1_floor(q1, 'resources.peak_worker_rss_gib')
    assert floor['floor_gib'] == 3.25 and floor['peak_worker_rss_gib'] == 2.25 and floor['sha256'] == sha(q1)
    with pytest.raises(ValueError, match='Q1 receipt has no'):
        pilot.q1_floor(q1, 'resources.peak')
    for bad in ({'stage': 'S0_PILOT_A', 'x': 1.0}, {'stage': 'S0_Q1', 'x': 0}, {'stage': 'S0_Q1', 'x': True}):
        path = tmp_path / ('bad%d.json' % len(list(tmp_path.iterdir())))
        _create(path, json.dumps(bad).encode('ascii'))
        with pytest.raises(ValueError):
            pilot.q1_floor(path, 'x')
    duplicate = tmp_path / 'duplicate.json'
    _create(duplicate, b'{"stage": "S0_Q1", "x": 1.0, "x": 9.0}')
    with pytest.raises(ValueError, match='duplicate'):
        pilot.q1_floor(duplicate, 'x')
    a = {'stage': 'S0_PILOT_A', 'schema': pilot.SCHEMA, 'decision': {'outcome': 'RERUN_WITH_B'},
         'registered_run': True, 'test_mode': False}
    for name, edit, registered, match in (
            ('ok', {}, True, None), ('test', {'registered_run': False, 'test_mode': True}, True, 'registered'),
            ('test-ok', {'registered_run': False, 'test_mode': True}, False, None),
            ('adopt', {'decision': {'outcome': 'ADOPT_A'}}, True, 'RERUN_WITH_B'),
            ('stage', {'stage': 'S0_PILOT_B'}, True, 'S0_PILOT_A')):
        path = tmp_path / ('a_%s.json' % name)
        _create(path, json.dumps(dict(a, **edit)).encode('ascii'))
        if match is None:
            assert pilot.a_pilot(path, registered)['sha256'] == sha(path)
        else:
            with pytest.raises(ValueError, match=match):
                pilot.a_pilot(path, registered)


def _registered(*extra):
    return ['--inventory', 'A', '--q1-receipt', 'q1.json', '--q1-peak-rss-key', 'peak', *extra]


def test_registered_mode_takes_no_overrides_and_refuses_before_any_work(tmp_path, monkeypatch):
    def no_work(*args, **kwargs):
        raise AssertionError('work started after a refused command line')
    monkeypatch.setattr(pilot, 'launch_snapshot', no_work)
    monkeypatch.setattr(s0, 'hostbudget', no_work)
    records = s0.RECORDS / 'S0_PILOT_A.unit-test-never-written.json'
    a_ok = tmp_path / 'a_ok.json'
    _create(a_ok, json.dumps({'stage': 'S0_PILOT_A', 'schema': pilot.SCHEMA, 'registered_run': True,
                              'test_mode': False, 'decision': {'outcome': 'RERUN_WITH_B'}}).encode('ascii'))
    cases = [
        (_registered('--groups', '2'), None, 'test-mode only option'),
        (_registered('--model-dir', str(tmp_path)), None, 'test-mode only option'),
        (_registered('--device', 'cpu'), None, 'runs on cuda'),
        (_registered(), FakeHost(), 'test-mode only'),
        (['--inventory', 'A'], None, 'needs --q1-receipt'),
        (_registered('--repo', str(tmp_path)), None, 'binds the checkout'),
        (['--inventory', 'B', '--q1-receipt', 'q1.json', '--q1-peak-rss-key', 'k'], None, 'RERUN_WITH_B'),
        (_registered('--out', str(tmp_path / 'S0_PILOT_A.json')), None, 'registered receipts are written to'),
        (_registered('--out', str(records), '--a-receipt', str(a_ok)), None, 'belongs to a B pilot'),
        (['--inventory', 'A', '--device', 'cpu', '--test-mode', '--host-floor-gib', '1'], None, 'needs --out'),
        (['--inventory', 'A', '--device', 'cpu', '--test-mode', '--out', str(tmp_path / 'x.json')], None,
         'host-floor-gib'),
        (['--inventory', 'A', '--device', 'cpu', '--test-mode', '--out', str(tmp_path / 'x.json'),
          '--host-floor-gib', '1', '--groups', '0'], None, 'groups >= 1'),
        (['--inventory', 'A', '--device', 'cpu', '--test-mode', '--out', str(tmp_path / 'x.json'),
          '--host-floor-gib', '1', '--group-size', '1'], None, 'group size >= 2'),
        # --test-mode never runs on the GPU: refused with --device cuda, given or by default
        (['--inventory', 'A', '--device', 'cuda', '--test-mode', '--out', str(tmp_path / 'x.json'),
          '--host-floor-gib', '1'], FakeHost(), 'refused with --device cuda'),
        (['--inventory', 'B', '--test-mode', '--out', str(tmp_path / 'x.json'), '--host-floor-gib', '1'], None,
         'refused with --device cuda'),
        (_registered('--out', str(records), '--lease-timeout-s', 'nan'), None, 'lease timeout')]
    for argv, host, match in cases:
        with pytest.raises(SystemExit, match=match):
            pilot.main(argv, host_module=host)
    with pytest.raises(ValueError, match='inside the bound checkout'):
        pilot.main(['--inventory', 'A', '--device', 'cpu', '--test-mode', '--out', str(REPO / 'S0_PILOT_A.json'),
                    '--host-floor-gib', '1'])
    existing = tmp_path / 'exists.json'
    _create(existing, b'{}\n')
    with pytest.raises(SystemExit, match='create-only'):
        pilot.main(['--inventory', 'A', '--device', 'cpu', '--test-mode', '--out', str(existing),
                    '--host-floor-gib', '1'])
    assert not (tmp_path / 'x.json').exists()
    cfg = pilot._config(pilot._parser().parse_args(_registered('--out', str(records))), None, None)
    assert {k: cfg[k] for k in ('groups', 'group_size', 'max_new_tokens', 'blocks', 'device', 'label', 'stage')} == {
        'groups': 25, 'group_size': 8, 'max_new_tokens': 128, 'blocks': [8, 27], 'device': 'cuda:0',
        'label': 'comp-rs-001-s0-pilot-a', 'stage': 'S0_PILOT_A'}
    assert cfg['model_dir'] == s0.MODEL_DIR and cfg['asset_receipt'] == s0.ASSET_RECEIPT and cfg['after'] is None
    b = pilot._config(pilot._parser().parse_args(
        ['--inventory', 'B', '--q1-receipt', 'q1.json', '--q1-peak-rss-key', 'k', '--a-receipt', str(a_ok),
         '--out', str(records)]), None, None)
    assert b['blocks'] == [14, 27] and b['after']['sha256'] == sha(a_ok) and b['stage'] == 'S0_PILOT_B'
    assert not records.exists()


def test_cli_refuses_to_start_below_the_launch_floor(tmp_path, tiny_dir, comp_rs_data, monkeypatch):
    def no_load(*args, **kwargs):
        raise AssertionError('loaded past a refused launch floor')
    monkeypatch.setattr(pilot, 'load_pilot_rows', no_load)
    monkeypatch.setattr(s0, 'load_model_fp32', no_load)
    lease, host = FakeLease(), FakeHost(free=0.4)
    with pytest.raises(SystemExit, match='REFUSING TO START comp-rs-001-s0-pilot-a'):
        run_cli(tmp_path, tiny_dir, comp_rs_data, lease=lease, host=host)
    assert host.calls == [('require', 0.5, 'comp-rs-001-s0-pilot-a')] and lease.events == []
    assert not (tmp_path / 'records').exists()


def test_cli_runs_the_whole_pilot_on_cpu(tmp_path, tiny_dir, comp_rs_data, tokenizer, selected, mixed_reward,
                                         monkeypatch):
    def no_checkpoint(*args, **kwargs):
        raise AssertionError('a weight/optimizer checkpoint write was attempted')
    import safetensors.torch
    from transformers import PreTrainedModel
    for owner, name in ((torch, 'save'), (safetensors.torch, 'save_file'), (safetensors.torch, 'save_model'),
                        (safetensors.torch, 'save'), (PreTrainedModel, 'save_pretrained')):
        monkeypatch.setattr(owner, name, no_checkpoint)
    tiny_before, tmp_before = _files(tiny_dir.root), _files(tmp_path)
    with recording_writes() as writes:
        result = run_cli(tmp_path, tiny_dir, comp_rs_data)
    receipt = result.receipt
    # outcome: a CPU pilot measures no VRAM, so it decides no inventory
    assert result.code == 2 and receipt['decision']['outcome'] == 'NOT_MEASURED_ON_CPU'
    assert receipt['completed'] is True and receipt['error'] is None and receipt['oom'] is None
    assert receipt['test_mode'] is True and receipt['registered_run'] is False and receipt['stage'] == 'S0_PILOT_A'
    assert receipt['schema'] == pilot.SCHEMA and receipt['argv'] == result.argv
    assert receipt['decision']['host_ok'] is True and receipt['decision']['inventory_fits'] is None
    # selection and seeds
    ids = selected.inputs['selected_ids'][:2]
    assert receipt['inputs']['selected_ids'] == ids and [grp['id'] for grp in receipt['groups']] == ids
    assert [grp['seed'] for grp in receipt['groups']] == [pilot.pilot_seed(i) for i in ids]
    assert [grp['sample_seeds'] for grp in receipt['groups']] == [
        [_independent_sample_seed(i, k) for k in range(3)] for i in ids]
    assert receipt['inputs']['sample_seed_rule'] == receipt['sample_seed_rule'] == pilot.SAMPLE_SEED_RULE
    assert receipt['response_rule'] == comp.RESPONSE_RULE and receipt['objective']['loss'] == pilot.LOSS_RULE
    assert 'one unpadded row per micro-batch; gradients accumulated over the G rows' in pilot.LOSS_RULE
    assert receipt['code_sha256']['s0_competence.py'] == sha(Path(comp.__file__))
    # per group: rewards, advantages, tokens, stop reasons, walls and resources
    assert receipt['group_outcomes'] == {'mixed': 2, 'all_fail': 0, 'all_success': 0}
    assert receipt['updates'] == 2 and receipt['skipped_updates'] == 0
    rows = {row['id']: row for row in selected.rows}
    for grp in receipt['groups']:
        assert grp['rewards'] == [1, 0, 0] and grp['advantages'] == maxrl.advantages([1, 0, 0])
        assert grp['outcome'] == 'mixed' and len(grp['completions']) == 3
        assert grp['generated_tokens'] == [c['generated_tokens'] for c in grp['completions']]
        assert grp['stop_reasons'] == [c['stop_reason'] for c in grp['completions']]
        assert grp['generation_wall_s'] > 0 and grp['group_wall_s'] >= grp['generation_wall_s']
        assert grp['update']['executed'] is True and grp['update_wall_s'] == grp['update']['wall_s'] > 0
        assert math.isfinite(grp['update']['loss']) and grp['update']['grad_norm'] > 0
        assert grp['update']['micro_batches'] == 3 and grp['update']['loss'] == math.fsum(
            grp['update']['sample_losses'])
        assert grp['update']['sampler_vs_teacher_forced_max_abs'] <= 1e-4
        assert grp['resources']['process_rss_gib'] > 0 and grp['resources']['cuda'] is None
        assert grp['sampler']['replay_verified'] is True and grp['sampler']['logits_processors'] == [g.RECORDER]
        assert grp['decode_steps'] == sum(grp['decode_steps_per_sample']) and len(grp['decode_steps_per_sample']) == 3
        assert len(grp['generation_wall_s_per_sample']) == 3
        assert math.fsum(grp['generation_wall_s_per_sample']) <= grp['generation_wall_s']
        assert grp['prompt_tokens'] == len(s0.render_prompt(tokenizer, rows[grp['id']]['prompt'])[1])
        for c in grp['completions']:
            text = comp.response_text(tokenizer, c['completion_ids'], c['stop_reason'])
            assert c['response'] == text['response'] and c['lenient_response'] == text['lenient_response']
            assert (grp['id'], c['response']) in mixed_reward
            assert 1 <= c['generated_tokens'] == len(c['completion_ids']) <= 4
    # state: frozen untouched, trainable moved, weights discarded
    fingerprints = receipt['fingerprints']
    assert fingerprints['frozen_unchanged'] is True and fingerprints['trainable_changed'] is True
    assert fingerprints['before']['state'] == receipt['identity']['state_fingerprint']
    assert fingerprints['before']['frozen'] == fingerprints['after']['frozen']
    assert receipt['weights_saved'] is False and receipt['checkpoints_written'] == 0
    assert receipt['inventory']['blocks'] == [1, 1] and receipt['gradient_checkpointing']['enabled'] is True
    assert receipt['optimizer']['defaults']['lr'] == 1e-6 and receipt['optimizer']['defaults']['foreach'] is False
    assert receipt['optimizer_state_after_first_step']['on_model_device'] is True
    # eval decode: batch-1 greedy, once per prompt
    evaluation = receipt['eval_decode']
    assert [r['id'] for r in evaluation['records']] == ids and all(r['wall_s'] > 0 for r in evaluation['records'])
    assert evaluation['summary']['completions'] == 2 and receipt['eval_decoder']['mode'] == 'greedy'
    # runtime, lease, host
    assert receipt['settings']['deterministic_algorithms'] is True and receipt['settings']['num_threads'] == 4
    assert receipt['decoding_sha256'] == s0.decoding_sha256(receipt['decoding_preimage'])
    assert receipt['generation_contract'] == g.generation_contract(4)
    assert receipt['code_sha256'] == pilot.code_sha256()
    assert result.lease.events == [('acquire', 'comp-rs-001-s0-pilot-a'), ('release', 'token-1')]
    assert result.host.calls[0] == ('require', 0.5, 'comp-rs-001-s0-pilot-a')
    guards = [c for c in result.host.calls if c[0] == 'guard']
    assert [c[1] for c in guards] == [1.0] * 4 and [c[2] for c in guards] == [
        'comp-rs-001-s0-pilot-a group 0', 'comp-rs-001-s0-pilot-a group 1'] + [
        'comp-rs-001-s0-pilot-a eval ' + i for i in ids]
    assert receipt['launch']['free_phys_gib'] == 64.0 and receipt['launch']['torch_imported'] is True
    assert receipt['memory']['peak_worker_rss_gib'] > 0 and receipt['memory']['max_memory_reserved_gib'] is None
    # nothing written but the receipt: no checkpoint anywhere, the tiny model untouched
    assert writes == [os.path.normcase(str(result.out))]
    assert _files(tiny_dir.root) == tiny_before
    assert set(_files(tmp_path)) - set(tmp_before) == {str(result.out.relative_to(tmp_path))}
    assert b'\r' not in result.raw and result.raw.endswith(b'\n') and result.raw.isascii()
    with pytest.raises(SystemExit, match='create-only'):
        run_cli(tmp_path, tiny_dir, comp_rs_data)


def test_cli_scores_with_the_real_reward_and_skips_uniform_groups(tmp_path, tiny_dir, comp_rs_data, tokenizer,
                                                                  selected):
    result = run_cli(tmp_path, tiny_dir, comp_rs_data)
    receipt = result.receipt
    rows = {row['id']: row for row in selected.rows}
    for grp in receipt['groups']:
        for c in grp['completions']:
            row = rows[grp['id']]
            text = comp.response_text(tokenizer, c['completion_ids'], c['stop_reason'])
            assert c['response'] == text['response'] and c['reward'] == corpus.reward(row, text['response'])
            assert c['lenient_response'] == text['lenient_response']
            assert c['lenient_reward'] == (None if text['lenient_response'] is None
                                           else corpus.reward(row, text['lenient_response']))
            assert 'strict_body_reward' not in c
        # a random tiny model never writes a valid answer in 4 tokens: uniform failure, no step
        assert grp['rewards'] == [0, 0, 0] and grp['advantages'] == [0.0, 0.0, 0.0]
        assert grp['outcome'] == 'all_fail' and grp['update']['executed'] is False and grp['update_wall_s'] == 0.0
    assert receipt['updates'] == 0 and receipt['skipped_updates'] == 2
    assert receipt['decision']['outcome'] == 'UNMEASURED_NO_UPDATE' and receipt['completed'] is True
    assert receipt['fingerprints']['trainable_changed'] is False and receipt['fingerprints']['frozen_unchanged']
    assert receipt['optimizer_state_after_first_step'] is None and receipt['optimizer_state_final']['tensors'] == 0
    assert len(receipt['eval_decode']['records']) == 2
    diagnostics = receipt['completion_diagnostics']
    assert 'strict_body_reward_differs' not in diagnostics and diagnostics['lenient_reward_differs'] == 0
    # no update ran, so every group sampled the loaded weights: each completion reproduces from its recorded seed
    s0.configure('cpu')
    model, _ = s0.load_model_fp32(tiny_dir.root, 'cpu', tiny_dir.asset)
    for grp in receipt['groups']:
        prompt_ids = s0.render_prompt(tokenizer, rows[grp['id']]['prompt'])[1]
        for seed, c in zip(grp['sample_seeds'], grp['completions']):
            alone = g.sample_group(model, prompt_ids, 1, seed, max_new_tokens=4)['completions'][0]
            assert alone['completion_ids'] == c['completion_ids'] and alone['stop_reason'] == c['stop_reason']


def test_cli_a_special_token_in_the_body_scores_zero_strict_and_one_lenient(tmp_path, tiny_dir, comp_rs_data,
                                                                           tokenizer, selected, monkeypatch):
    """C11 end to end: every sample is the row's correct answer with <|im_start|> spliced into the body. The strict
    rewards are all 0, so no group updates; only the diagnostic scores the lenient text 1."""
    by_prompt = {}
    for row in selected.rows[:2]:
        ids = tokenizer(oracle.answers(row['verifier']['spec'])[0], add_special_tokens=False)['input_ids']
        by_prompt[tuple(s0.render_prompt(tokenizer, row['prompt'])[1])] = ids[:1] + [IM_START] + ids[1:] + [IM_END]
    seen, real = [], g.sample_group

    def spliced(model, prompt_ids, G, seed, **kwargs):
        seen.append((G, seed))
        out = real(model, prompt_ids, G, seed, **kwargs)   # the real call, for its sampler record
        ids = by_prompt[tuple(prompt_ids)]
        return dict(out, completions=[{'completion_ids': ids, 'stop_reason': 'eos', 'generated_tokens': len(ids),
                                       'raw_step_logprobs': [-1.0] * len(ids)}])
    monkeypatch.setattr(g, 'sample_group', spliced)
    result = run_cli(tmp_path, tiny_dir, comp_rs_data)
    receipt = result.receipt
    assert seen == [(1, seed) for grp in receipt['groups'] for seed in grp['sample_seeds']]
    for grp in receipt['groups']:
        assert grp['rewards'] == [0, 0, 0] and grp['outcome'] == 'all_fail' and grp['update']['executed'] is False
        for c in grp['completions']:
            assert '<|im_start|>' in c['response'] and c['body_special_ids'] == [IM_START]
            assert c['reward'] == 0 and c['lenient_reward'] == 1
    diagnostics = receipt['completion_diagnostics']
    assert diagnostics['body_special'] == 6 and diagnostics['lenient_reward_differs'] == 6
    assert receipt['updates'] == 0 and receipt['decision']['outcome'] == 'UNMEASURED_NO_UPDATE'


@pytest.mark.parametrize('inventory, target, phase, outcome', [
    ('A', 'update', 'update', 'RERUN_WITH_B'),
    ('B', 'sample', 'generate', 'STOP_HOST_INFEASIBLE'),
    ('A', 'greedy', 'eval', 'RERUN_WITH_B')])
def test_cli_classifies_an_oom(tmp_path, tiny_dir, comp_rs_data, mixed_reward, monkeypatch, inventory, target,
                               phase, outcome):
    def out_of_memory(*args, **kwargs):
        raise torch.OutOfMemoryError('CUDA out of memory. Tried to allocate 2.00 GiB (simulated)')
    owner, name = {'update': (pilot, 'update_step'), 'sample': (g, 'sample_group'),
                   'greedy': (g, 'greedy_decode')}[target]
    monkeypatch.setattr(owner, name, out_of_memory)
    result = run_cli(tmp_path, tiny_dir, comp_rs_data, inventory=inventory)
    receipt = result.receipt
    assert result.code == 2 and receipt['stage'] == 'S0_PILOT_' + inventory
    assert receipt['oom']['phase'] == phase and receipt['oom']['type'] == 'OutOfMemoryError'
    assert 'simulated' in receipt['oom']['message'] and receipt['error'] is None
    decision = receipt['decision']
    assert decision['outcome'] == outcome and decision['inventory_fits'] is False and decision['oom'] is True
    assert receipt['completed'] is False and receipt['fingerprints']['frozen_unchanged'] is True
    if phase == 'eval':
        assert receipt['oom']['group'] is None and receipt['updates'] == 2 and len(receipt['groups']) == 2
        assert receipt['eval_decode']['records'] == []
    else:
        assert receipt['oom']['group'] == 0 and len(receipt['groups']) == 1 and receipt['updates'] == 0
        assert receipt['groups'][0]['oom'] is True and 'resources' in receipt['groups'][0]
        assert receipt['eval_decode']['records'] == [] and receipt['fingerprints']['trainable_changed'] is False
    assert result.lease.events[-1] == ('release', 'token-1')


def test_cli_stands_down_on_the_host_guard(tmp_path, tiny_dir, comp_rs_data, mixed_reward):
    result = run_cli(tmp_path, tiny_dir, comp_rs_data, host=FakeHost(abort_at=2))
    receipt = result.receipt
    assert receipt['abort']['before'] == 'group 1' and receipt['abort']['type'] == 'LowMemoryAbort'
    assert len(receipt['groups']) == 1 and receipt['updates'] == 1 and receipt['eval_decode']['records'] == []
    assert receipt['completed'] is False and receipt['decision']['outcome'] == 'INCOMPLETE' and result.code == 2


def test_cli_host_check_uses_the_free_ram_measured_at_launch(tmp_path, tiny_dir, comp_rs_data, mixed_reward):
    result = run_cli(tmp_path, tiny_dir, comp_rs_data, host=FakeHost(free=1.2))
    decision = result.receipt['decision']
    assert decision['free_phys_gib_at_launch'] == 1.2 and result.receipt['completed'] is True
    assert decision['peak_worker_rss_gib'] + 1.0 > 1.2 and decision['host_ok'] is False
    assert decision['outcome'] == 'STOP_HOST_INFEASIBLE' and result.code == 2


def test_cli_writes_no_receipt_if_its_code_changed_during_the_run(tmp_path, tiny_dir, comp_rs_data, monkeypatch):
    monkeypatch.setattr(pilot, 'code_sha256', lambda: dict(pilot._IMPORT_CODE, **{'s0_generate.py': '0' * 64}))
    with pytest.raises(ValueError, match='pilot source changed during the run'):
        run_cli(tmp_path, tiny_dir, comp_rs_data, groups=1, size=2, tokens=2)
    assert not (tmp_path / 'records' / 'S0_PILOT_A.json').exists()


def test_cli_failure_writes_a_receipt_releases_the_lease_and_reraises(tmp_path, tiny_dir, comp_rs_data,
                                                                      mixed_reward, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError('update failed (simulated)')
    monkeypatch.setattr(pilot, 'update_step', broken)
    lease = FakeLease()
    with pytest.raises(ValueError, match='simulated'):
        run_cli(tmp_path, tiny_dir, comp_rs_data, lease=lease)
    receipt = json.loads((tmp_path / 'records' / 'S0_PILOT_A.json').read_bytes().decode('ascii'))
    assert receipt['error'] == {'type': 'ValueError', 'message': 'update failed (simulated)'}
    assert receipt['completed'] is False and receipt['decision']['outcome'] == 'INCOMPLETE'
    assert lease.events == [('acquire', 'comp-rs-001-s0-pilot-a'), ('release', 'token-1')]
