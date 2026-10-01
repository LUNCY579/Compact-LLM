"""S0 warm-start of theta0' (Amendment 1 section 18, Amendment 2): data binding, order and plan, rendering and
labels, the loss reduction, clipping and the optimizer, the trainable scope, the delta, the anchor under the shared
digest convention and its load_anchor round trip, the envelope check, the receipt as s0_freeze reads it, and the
whole CLI on CPU.

CPU only: no test creates a CUDA tensor or initialises CUDA (asserted after every test), no test takes the real GPU
lease or reads the real host budget (fakes are injected), and the real Supa weights are never opened. The model is a
tiny random Qwen3 (vocab 151936, hidden 32, 2 layers, heads 4/2, head_dim 8, intermediate 64, tied) whose
config.json is the real Supa config with the sizes overridden, saved as BF16 safetensors in a temporary directory
with a Q1-format asset receipt. The real inputs read are the Supa tokenizer/config files, manifest.json, the three
warm-start files, the CALIBRATION canonical and policy files, and (read-only) the S0 runtime freeze.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from experiments.maxrl_comp002_pretraining_v1 import arm_build, mechanics
from experiments.maxrl_comp002_pretraining_v1 import s0_competence as comp
from experiments.maxrl_comp002_pretraining_v1 import s0_freeze
from experiments.maxrl_comp002_pretraining_v1 import s0_generate as g
from experiments.maxrl_comp002_pretraining_v1 import s0_pilot as pilot
from experiments.maxrl_comp002_pretraining_v1 import s0_q1_fp32 as q1
from experiments.maxrl_comp002_pretraining_v1 import s0_runtime as s0
from experiments.maxrl_comp002_pretraining_v1 import s0_warmstart as ws
from experiments.maxrl_comp002_pretraining_v1.preflight import digest, read_json, sha
from experiments.maxrl_composable_skills_v2 import corpus, oracle

REPO = Path(__file__).resolve().parents[3]
TINY = {'hidden_size': 32, 'num_hidden_layers': 2, 'num_attention_heads': 4, 'num_key_value_heads': 2,
        'head_dim': 8, 'intermediate_size': 64, 'max_window_layers': 2}
IM_END, ENDOFTEXT, IM_START = 151645, 151643, 151644
# The registered order (section 18): pinned so a rule change shows; recomputed independently below.
PINNED_ORDER_DIGEST = '985faae24ad5dd1b82edde6302cbbc6a8787f74a55d4983a03633a2d6d43b49a'
PINNED_FIRST_IDS = ['MC2-b5e7f32558de95d8a1ba807ae71c730d255a624cbf2079215f42a52022da57bc',
                    'MC2-68c669f4b14695d6e6aca29d21b29c65f8eb10ccdaf481bc73fa635a349be044',
                    'MC2-07f3024d25fe594d8b8689ea6894f1832fbbd8d2be6e85939d7036e84f237caf']
PINNED_LINEAGE_DIGEST = '4873e4df3ce1495bfca50a222f1d4989c1fd5422915491e8910256be4dee9149'
ENVELOPE_LENGTHS = [835, 231, 523, 219, 822, 228, 535, 222]   # the S0-Q1 probe prompts (test_s0_q1_fp32)
PROMPT = [9707, 1879, 11, 1034, 525, 3555]


# ------------------------------------------------------------------------------------------------ fixtures
@contextlib.contextmanager
def torch_globals():
    """Restore the process-global torch settings configure() changes."""
    saved = (torch.are_deterministic_algorithms_enabled(), torch.is_deterministic_algorithms_warn_only_enabled(),
             torch.get_float32_matmul_precision(), torch.backends.cuda.matmul.allow_tf32,
             torch.backends.cudnn.allow_tf32, torch.get_num_threads(), os.environ.get('CUBLAS_WORKSPACE_CONFIG'))
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(saved[0], warn_only=saved[1])
        torch.set_float32_matmul_precision(saved[2])
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved[3], saved[4]
        torch.set_num_threads(saved[5])
        if saved[6] is None:
            os.environ.pop('CUBLAS_WORKSPACE_CONFIG', None)
        else:
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = saved[6]


@pytest.fixture(autouse=True)
def torch_state():
    """CUDA must never initialise; the global torch settings are restored after every test."""
    with torch_globals():
        yield
    assert torch.cuda.is_initialized() is False


@pytest.fixture(scope='module', autouse=True)
def synthetic_disk_headroom():
    """CPU mechanics tests do not rely on the user's current drive headroom."""
    old = ws.checkpoints.shutil.disk_usage
    ws.checkpoints.shutil.disk_usage = lambda path: SimpleNamespace(free=64 * s0.GIB)
    try:
        yield
    finally:
        ws.checkpoints.shutil.disk_usage = old


@pytest.fixture(scope='module', autouse=True)
def no_optimizer_outlives_the_module():
    """The warm-start and these tests leave no torch.optim.Optimizer alive (the Q1 T6 census counts every one)."""
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'xb') as stream:
        stream.write(data)


@pytest.fixture(scope='module')
def tiny_dir(tmp_path_factory, real_config):
    """A tiny random Qwen3 in the real layout (BF16, no lm_head) and its Q1-format asset receipt."""
    from transformers import Qwen3Config, Qwen3ForCausalLM
    root = tmp_path_factory.mktemp('warmstart_tiny')
    model_dir = root / 'tiny'
    config = dict(real_config, **TINY)
    _create(model_dir / 'config.json', (json.dumps(config, indent=2, sort_keys=True) + '\n').encode('ascii'))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20260930)
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
    return SimpleNamespace(root=model_dir, asset=asset, weights=model_dir / 'model.safetensors',
                           weights_sha256=receipt['assets']['model.safetensors']['sha256'], stored=sorted(state))


@pytest.fixture
def tiny_model(tiny_dir):
    model, _ = s0.load_model_fp32(tiny_dir.root, 'cpu', tiny_dir.asset)
    return model


@pytest.fixture(scope='module')
def tokenizer(real_config):
    return s0.load_tokenizer(s0.MODEL_DIR, s0.TOKENIZER_RECEIPT, asset_receipt=s0.ASSET_RECEIPT)


@pytest.fixture(scope='module')
def warm(comp_rs_data):
    rows, inputs = ws.load_warmstart(comp_rs_data)
    return SimpleNamespace(rows=rows, inputs=inputs)


@pytest.fixture(scope='module')
def rendered(tokenizer, warm):
    items, record = ws.render_rows(tokenizer, warm.rows)
    return SimpleNamespace(items=items, record=record, by_id={item['id']: item for item in items})


class FakeLease:
    def __init__(self):
        self.LOCK = Path('Z:/fake-temp') / s0.LOCK_NAME
        self.events = []

    def acquire(self, timeout_s, label):
        self.events.append(('acquire', label))
        return 'token-ws'

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


class CapturingOptimizer:
    """Stands in for AdamW in train_step: step() snapshots the (clipped) gradients and changes nothing."""

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


def run_cli(records, tiny_dir, data, *extra, lease=None, host=None, rows=16, tokens=3):
    argv = ['--device', 'cpu', '--test-mode', '--records', str(records), '--data', str(data), '--rows', str(rows),
            '--test-blocks', '1', '1', '--model-dir', str(tiny_dir.root), '--asset-receipt', str(tiny_dir.asset),
            '--max-new-tokens', str(tokens), '--host-floor-gib', '0.5', *extra]
    lease = FakeLease() if lease is None else lease
    host = FakeHost() if host is None else host
    code = ws.main(argv, lease_module=lease, host_module=host)
    path = records / ws.RECEIPT_NAME
    if not path.exists():
        path = max(records.glob('S0_WARMSTART_INTERRUPTION_*.json'), key=lambda p: p.stat().st_mtime_ns)
    raw = path.read_bytes()
    return SimpleNamespace(code=code, receipt=json.loads(raw.decode('ascii')), raw=raw, records=records, argv=argv,
                           lease=lease, host=host, delta=records / ws.DELTA_NAME, path=path)


@pytest.fixture(scope='module')
def short_run(tmp_path_factory, tiny_dir, comp_rs_data, tokenizer):
    """One short --test-mode warm-start (16 rows, 2 steps, layer 1 of the tiny model), shared by several tests."""
    records = tmp_path_factory.mktemp('warmstart_cli') / 'records'
    with torch_globals():
        return run_cli(records, tiny_dir, comp_rs_data)


def _item(row_id, completion):
    return {'id': row_id, 'prompt_ids': list(PROMPT), 'completion_ids': list(completion)}


def _reference_rows(model, items):
    """Per row, from ONE full-sequence forward: mechanics labels, selected logprobs and F.cross_entropy (mean)."""
    out = []
    for item in items:
        ids = torch.tensor([item['prompt_ids'] + item['completion_ids']], dtype=torch.long)
        labels = mechanics.completion_labels(ids, torch.ones_like(ids, dtype=torch.bool), [len(item['prompt_ids'])],
                                             [len(item['completion_ids'])], [IM_END, ENDOFTEXT], ['eos'], 128)
        logits = model(input_ids=ids, use_cache=False).logits
        selected = mechanics.selected_logprobs(logits.float(), ids, labels)
        ce = torch.nn.functional.cross_entropy(logits[0, :-1].float(), labels[0, 1:], ignore_index=-100,
                                               reduction='mean')
        out.append(SimpleNamespace(ids=ids, labels=labels, selected=selected, ce=ce))
    return out


def cpu_q1_receipt(receipt):
    """A synthetic passed S0-Q1 receipt under the warm-start's own (CPU) decoding preimage."""
    preimage = receipt['decoding_preimage']
    return {'stage': 'S0_Q1_FP32', 'test_mode': True, 'registered_run': False, 'passed': True, 'failed_checks': [],
            'decoding_preimage': preimage, 'decoding_sha256': s0.decoding_sha256(preimage),
            'host_floor': {'peak_worker_rss_gib': 2.0}}


def _dump(path: Path, value) -> Path:
    _create(path, (json.dumps(value) + '\n').encode('ascii'))
    return path


# ------------------------------------------------------------------------------------------------ constants
def test_registered_constants_are_section_18():
    assert ws.REGISTERED == {'device': 'cuda:0', 'rows': 600, 'accumulation': 8, 'steps': 75, 'epochs': 1,
                             'blocks': [14, 27], 'numel': 220233216, 'num_hidden_layers': 28, 'max_new_tokens': 128}
    assert ws.REGISTERED['blocks'] == pilot.INVENTORIES['B']['blocks'] == list(s0_freeze.INVENTORY_BLOCKS['B'])
    assert ws.REGISTERED['numel'] == pilot.INVENTORIES['B']['numel'] == s0_freeze.INVENTORY_NUMEL['B']
    assert ws.ADAMW == {'lr': 1e-5, 'betas': (0.9, 0.999), 'eps': 1e-8, 'weight_decay': 0.0, 'amsgrad': False,
                        'maximize': False, 'foreach': False}
    assert (ws.MAX_GRAD_NORM, ws.GRAD_NORM_TYPE, ws.GUARD_GIB, ws.ACCUMULATION, ws.EPOCHS) == (1.0, 2.0, 1.0, 8, 1)
    assert ws.LABEL == 'comp-rs-001-s0-warmstart' and s0.LEASE_LABEL.fullmatch(ws.LABEL)
    assert ws.EOS_ID == IM_END == s0.EOS_IDS[0] and ws.MAX_TARGET_TOKENS == 128
    assert ws.ORDER_SALT == 'COMP-RS-001-WARMSTART|' and ws.MANIFEST.startswith('86ac8f0d')
    assert ws.FILES == {'messages': 'sft/warmstart.messages.jsonl', 'lineage': 'sft/warmstart.lineage.jsonl',
                        'core': 'manifests/warmstart_core.json'}
    assert ws.RECEIPT_NAME == 'S0_WARMSTART.json' and ws.DELTA_NAME == 'S0_WARMSTART_DELTA.safetensors'
    # the receipt is the one s0_freeze's theta0' mode reads: its stage name and anchor keys
    assert ws.STAGE == s0_freeze.WARMSTART_STAGE == 'S0_WARMSTART'
    assert ws.ANCHOR_KEYS == s0_freeze.WARMSTART_ANCHOR_KEYS == s0.ANCHOR_KEYS
    assert all(p.is_file() for p in ws.CODE_FILES.values())
    for name in ('s0_warmstart.py', 's0_pilot.py', 's0_q1_fp32.py', 'arm_build.py', 's0_generate.py',
                 's0_runtime.py', 's0_competence.py', 'mechanics.py', 'content_scoring.py'):
        assert ws.PACKAGE + name in ws.CODE_FILES


# ------------------------------------------------------------------------------------------------ order and plan
def test_order_is_the_salted_hash_sort_whatever_the_input_order():
    ids = ['MC2-%04d' % i for i in range(40)]
    expected = sorted(ids, key=lambda i: hashlib.sha256(('COMP-RS-001-WARMSTART|' + i).encode()).hexdigest())
    assert ws.order_ids(ids) == expected != ids
    assert ws.order_ids(list(reversed(ids))) == expected == ws.order_ids(ids[::3] + ids[1::3] + ids[2::3])
    assert ws.order_ids(list(ids)) == ws.order_ids(list(ids))   # deterministic: same input, same order
    assert ws.order_key('MC2-a') == hashlib.sha256(b'COMP-RS-001-WARMSTART|MC2-a').hexdigest()
    unsalted = sorted(ids, key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    assert ws.order_ids(ids) != unsalted                          # the salt is part of the rule
    for bad in (ids + ids[:1], [], ids + [''], tuple(ids)):
        with pytest.raises(ValueError):
            ws.order_ids(bad)


def test_the_registered_order_over_the_real_lineage(warm):
    ids = [row['id'] for row in warm.rows]
    independent = sorted(ids, key=lambda i: hashlib.sha256(('COMP-RS-001-WARMSTART|' + i).encode()).hexdigest())
    order = ws.order_ids(ids)
    assert order == independent and len(order) == 600 and digest(order) == PINNED_ORDER_DIGEST
    assert order[:3] == PINNED_FIRST_IDS
    assert digest(ids) == PINNED_LINEAGE_DIGEST == warm.inputs['lineage_ids_sha256']


def test_exactly_75_steps_of_8_are_planned_for_600_rows(warm):
    order = ws.order_ids([row['id'] for row in warm.rows])
    plan = ws.plan_steps(order)
    assert len(plan) == 75 == ws.REGISTERED['steps'] and all(len(step) == 8 for step in plan)
    assert [i for step in plan for i in step] == order
    assert ws.plan_steps(order[:16]) == [order[:8], order[8:16]]   # a test-mode subset is a prefix of the order
    for n in (601, 599, 596, 4):
        with pytest.raises(ValueError, match='partial last accumulation'):
            ws.plan_steps((order * 2)[:n])
    with pytest.raises(ValueError, match='plan rows'):
        ws.plan_steps([])


# ------------------------------------------------------------------------------------------------ data
def test_real_inputs_are_bound_to_the_manifest(warm, comp_rs_data):
    manifest = json.loads((comp_rs_data / 'manifest.json').read_bytes().decode('utf-8'))
    assert sha(comp_rs_data / 'manifest.json') == ws.MANIFEST == warm.inputs['dataset_manifest_sha256']
    files = warm.inputs['files']
    assert set(files) == set(ws.FILES.values())
    for relative, record in files.items():
        assert record == {'sha256': manifest['files'][relative]['sha256'], 'bytes': manifest['files'][relative]['bytes']}
        assert record['sha256'] == sha(comp_rs_data / relative)
    core = json.loads((comp_rs_data / 'manifests' / 'warmstart_core.json').read_bytes().decode('utf-8'))
    assert warm.inputs['rows'] == 600 == len(warm.rows) and set(r['id'] for r in warm.rows) == set(core['ids'])
    assert warm.inputs['core_ids_sha256'] == digest(core['ids']) and warm.inputs['targets_checked'] == 600
    lineage = [json.loads(line) for line in (comp_rs_data / 'sft' / 'warmstart.lineage.jsonl').read_bytes().splitlines()]
    messages = [json.loads(line) for line in
                (comp_rs_data / 'sft' / 'warmstart.messages.jsonl').read_bytes().splitlines()]
    for row, entry, message in zip(warm.rows, lineage, messages):   # ids in the same line order
        assert row['id'] == entry['id'] and row['prompt'] == message['messages'][:2]
        assert row['target'] == message['messages'][2]['content']
        assert hashlib.sha256(row['target'].encode('utf-8')).hexdigest() == entry['target_sha256']


def _synthetic(n=3):
    messages, lineage = [], []
    for k in range(n):
        target = '{"answer":[%d]}' % k
        messages.append({'messages': [{'role': 'system', 'content': 'Solve.'},
                                      {'role': 'user', 'content': 'Task %d' % k},
                                      {'role': 'assistant', 'content': target}]})
        lineage.append({'id': 'MC2-w%d' % k, 'parent_id': 'world-%d' % k,
                        'target_sha256': hashlib.sha256(target.encode('utf-8')).hexdigest()})
    core = {'name': 'warmstart_core', 'ids': [e['id'] for e in reversed(lineage)], 'rows': n}
    return messages, lineage, core


def test_validate_warmstart_binds_ids_and_targets():
    messages, lineage, core = _synthetic()
    rows = ws.validate_warmstart(messages, lineage, core)
    assert [r['id'] for r in rows] == ['MC2-w0', 'MC2-w1', 'MC2-w2']   # lineage line order, not the core order
    assert rows[1]['target'] == '{"answer":[1]}' and rows[1]['prompt'] == messages[1]['messages'][:2]
    assert rows[2]['line'] == 3 and rows[0]['parent_id'] == 'world-0'

    def refuses(match, m=messages, l=lineage, c=core):
        with pytest.raises(ValueError, match=match):
            ws.validate_warmstart(m, l, c)
    wrong_target = [dict(messages[0], messages=messages[0]['messages'][:2] + [
        {'role': 'assistant', 'content': '{"answer":[9]}'}])] + messages[1:]
    refuses('target sha256', m=wrong_target)
    refuses('differ from manifests/warmstart_core.json', c=dict(core, ids=core['ids'][:2] + ['MC2-other']))
    refuses('warmstart_core manifest schema', c=dict(core, rows=2))
    refuses('warmstart_core manifest schema', c=dict(core, name='warmstart_boundary'))
    refuses('duplicate lineage id', l=lineage[:2] + [dict(lineage[2], id='MC2-w0')],
            c=dict(core, ids=['MC2-w0', 'MC2-w1'], rows=2))
    refuses('pair line by line', m=messages[:2])
    refuses('lineage fields', l=lineage[:2] + [dict(lineage[2], extra=1)])
    refuses(r'not \[system, user, assistant\]', m=messages[:2] + [{'messages': messages[2]['messages'][:2]}])
    swapped = [{'messages': [messages[0]['messages'][1], messages[0]['messages'][0], messages[0]['messages'][2]]}]
    refuses(r'not \[system, user, assistant\]', m=swapped + messages[1:])
    refuses('message fields', m=[{'messages': [dict(messages[0]['messages'][0], name='x')]
                                  + messages[0]['messages'][1:]}] + messages[1:])


def test_load_warmstart_refuses_drifted_files(tmp_path, comp_rs_data):
    root = tmp_path / 'data'
    for relative in ('manifest.json',) + tuple(ws.FILES.values()):
        _create(root / relative, (comp_rs_data / relative).read_bytes())
    rows, _ = ws.load_warmstart(root)
    assert len(rows) == 600
    messages = root / ws.FILES['messages']
    real = messages.read_bytes()
    target = b'{\\"answer\\":[0,1,3,4,6,7]}'    # the first target as it is escaped inside its JSON line
    assert real.count(target) >= 1
    messages.write_bytes(real.replace(target, b'{\\"answer\\":[0,1,3,4,6,8]}', 1))
    with pytest.raises(ValueError, match=r'warmstart.messages.jsonl drift \(manifest\)'):
        ws.load_warmstart(root)
    messages.write_bytes(real)
    lineage = root / ws.FILES['lineage']
    lineage_real = lineage.read_bytes()
    lineage.write_bytes(lineage_real + b'\n')
    with pytest.raises(ValueError, match=r'warmstart.lineage.jsonl drift \(manifest\)'):
        ws.load_warmstart(root)
    lineage.write_bytes(lineage_real)
    core = root / ws.FILES['core']
    core.unlink()
    with pytest.raises(FileNotFoundError):
        ws.load_warmstart(root)
    _create(core, (comp_rs_data / ws.FILES['core']).read_bytes())
    assert len(ws.load_warmstart(root)[0]) == 600
    (root / 'manifest.json').write_bytes((comp_rs_data / 'manifest.json').read_bytes() + b' ')
    with pytest.raises(ValueError, match='dataset manifest drift'):
        ws.load_warmstart(root)


# ------------------------------------------------------------------------------------------------ rendering and labels
def test_every_real_row_renders_with_labels_equal_to_mechanics(tokenizer, warm, rendered):
    record = rendered.record
    assert record['rows_rendered'] == 600 and record['eos_id'] == IM_END
    assert record['prompt_tokens']['max'] == 852 and record['labelled_tokens'] == {'min': 8, 'max': 30,
                                                                                    'total': 11005}
    assert record['sequence_ids_sha256'] == digest([i['prompt_ids'] + i['completion_ids'] for i in rendered.items])
    for row, item in zip(warm.rows, rendered.items):
        text, prompt_ids = s0.render_prompt(tokenizer, row['prompt'])
        assert item['prompt_ids'] == prompt_ids and tuple(prompt_ids[-7:]) == s0.THINK_PREFIX_IDS
        target_ids = tokenizer(row['target'], add_special_tokens=False)['input_ids']
        assert item['completion_ids'] == target_ids + [IM_END] and item['target_tokens'] == len(target_ids)
        assert item['label_positions'] == [len(prompt_ids), len(prompt_ids) + len(target_ids) + 1]
    # the labels are mechanics.completion_labels', built independently here: prompt masked, target + EOS labelled
    for item in rendered.items[:25]:
        ids, labels = ws.row_labels(item['prompt_ids'], item['completion_ids'])
        expected_ids = torch.tensor([item['prompt_ids'] + item['completion_ids']], dtype=torch.long)
        expected = mechanics.completion_labels(expected_ids, torch.ones_like(expected_ids, dtype=torch.bool),
                                               [len(item['prompt_ids'])], [len(item['completion_ids'])],
                                               [IM_END, ENDOFTEXT], ['eos'], 128)
        assert torch.equal(ids, expected_ids) and torch.equal(labels, expected)
        p = len(item['prompt_ids'])
        assert bool((labels[0, :p] == -100).all()) and labels[0, p:].tolist() == item['completion_ids']
        assert labels[0, -1].item() == IM_END


def test_render_row_refuses_targets_outside_the_contract(tokenizer, warm):
    row = warm.rows[0]
    for target, match in (('{"answer":[1]}<|im_end|>', 'special token inside the target'),
                          ('{"answer":[1]}<|endoftext|>', 'special token inside the target'),
                          ('{"answer":[%s]}' % ','.join(str(i) for i in range(200)), 'exceeds the decode cap')):
        messages = row['messages'][:2] + [{'role': 'assistant', 'content': target}]
        with pytest.raises(ValueError, match=match):
            ws.render_row(tokenizer, dict(row, target=target, messages=messages))
    with pytest.raises(ValueError, match='assistant turn elsewhere'):   # a target the template would not place there
        ws.render_row(tokenizer, dict(row, target='{"answer":[1]}'))
    with pytest.raises(ValueError, match='first EOS must terminate'):  # mechanics refuses an EOS inside the target
        ws.row_labels(PROMPT, [5, IM_END, 6, IM_END])


def test_the_loss_path_reads_exactly_the_mechanics_labels(tiny_model, rendered):
    """teacher_forced_logprobs (head at completion positions only) == mechanics.selected_logprobs of the full
    logits at the completion_labels positions (one shift), and the row loss == F.cross_entropy(mean) there."""
    items = rendered.items[:2]
    with torch.no_grad():
        reference = _reference_rows(tiny_model, items)
        ours = [g.teacher_forced_logprobs(tiny_model, [i['prompt_ids']], [i['completion_ids']], ENDOFTEXT,
                                          padding_side='none')[0] for i in items]
        unshifted = [tiny_model(input_ids=r.ids, use_cache=False).logits[0].log_softmax(-1).gather(
            -1, r.ids[0, :, None]).squeeze(-1)[r.labels[0] != -100] for r in reference]
    for ref, values, wrong, item in zip(reference, ours, unshifted, items):
        assert values.shape == ref.selected.shape == (len(item['completion_ids']),)
        assert float((values - ref.selected).abs().max()) <= 1e-5
        assert abs(float(-ref.selected.mean()) - float(ref.ce)) <= 1e-5
        assert float((wrong - ref.selected).abs().max()) > 1e-3     # a zero-shift reading would differ
    pilot.set_inventory(tiny_model, 1, 1)
    with torch.enable_grad():
        result = ws.accumulate_step(tiny_model, items)
    tiny_model.zero_grad(set_to_none=True)
    assert result['row_ids'] == [i['id'] for i in items]
    assert result['row_labelled_tokens'] == [len(i['completion_ids']) for i in items]
    assert all(abs(loss - float(ref.ce)) <= 1e-5 for loss, ref in zip(result['row_losses'], reference))


def test_loss_reduction_is_row_mean_then_step_mean_on_two_constructed_rows(tiny_model):
    items = [_item('a', [5, 6, IM_END]), _item('b', [7, 8, 9, 10, 11, 12, 13, IM_END])]
    pilot.set_inventory(tiny_model, 1, 1)
    trainable = {n: p for n, p in tiny_model.named_parameters() if p.requires_grad}
    with torch.enable_grad():
        reference = _reference_rows(tiny_model, items)
        expected = (reference[0].ce + reference[1].ce) / 2   # the stated rule, in one graph
        expected.backward()
    grads = {n: p.grad.detach().clone() for n, p in trainable.items()}
    tiny_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        pooled_ref = _reference_rows(tiny_model, items)
        pooled = torch.cat([-r.selected for r in pooled_ref]).mean()   # token-weighted: NOT the rule
        pooled.backward()
    pooled_grads = {n: p.grad.detach().clone() for n, p in trainable.items()}
    tiny_model.zero_grad(set_to_none=True)
    with torch.enable_grad():
        result = ws.accumulate_step(tiny_model, items)
    assert result['rows'] == 2 and result['row_labelled_tokens'] == [3, 8]
    assert result['loss'] == math.fsum(result['row_losses']) / 2
    assert abs(result['loss'] - float(expected)) <= 1e-5
    assert abs(result['loss'] - float(pooled)) > 1e-3                   # the two reductions differ here
    scale = max(float(v.abs().max()) for v in grads.values())
    assert scale > 1e-4                                                 # a real gradient
    for n, p in trainable.items():
        assert float((p.grad - grads[n]).abs().max()) <= 1e-5 + 1e-4 * float(grads[n].abs().max()), n
    assert max(float((pooled_grads[n] - grads[n]).abs().max()) for n in trainable) > 1e-3 * scale
    assert all(p.grad is None for n, p in tiny_model.named_parameters() if n not in trainable)
    tiny_model.zero_grad(set_to_none=True)
    with torch.no_grad(), pytest.raises(ValueError, match='no autograd graph'):
        ws.accumulate_step(tiny_model, items)
    with pytest.raises(ValueError, match='step rows'):
        ws.accumulate_step(tiny_model, [])


# ------------------------------------------------------------------------------------------------ step
def test_train_step_clips_to_the_registered_norm_before_the_step(tiny_model, monkeypatch):
    items = [_item('a', [5, 6, IM_END]), _item('b', [7, 8, 9, IM_END])]
    pilot.set_inventory(tiny_model, 1, 1)
    with torch.enable_grad():
        ws.accumulate_step(tiny_model, items)
    trainable = [(n, p) for n, p in tiny_model.named_parameters() if p.requires_grad]
    raw = {n: p.grad.detach().clone() for n, p in trainable}
    norm = math.sqrt(math.fsum(float((v.double() ** 2).sum()) for v in raw.values()))
    tiny_model.zero_grad(set_to_none=True)
    calls, real = [], torch.nn.utils.clip_grad_norm_

    def spy(parameters, max_norm, **kwargs):
        calls.append((len(parameters), max_norm, kwargs))
        return real(parameters, max_norm, **kwargs)
    monkeypatch.setattr(torch.nn.utils, 'clip_grad_norm_', spy)
    optimizer = CapturingOptimizer(tiny_model)
    step = ws.train_step(tiny_model, optimizer, items)
    assert calls == [(len(trainable), 1.0, {'norm_type': 2.0, 'error_if_nonfinite': True, 'foreach': False})]
    assert abs(step['pre_clip_grad_norm'] - norm) <= 1e-5 * norm
    assert optimizer.events == ['zero_grad', 'step', 'zero_grad']
    coefficient = min(1.0, 1.0 / (step['pre_clip_grad_norm'] + 1e-6))
    assert step['clip_coefficient'] == coefficient and step['clipped'] == (coefficient < 1.0)
    for n, _ in trainable:   # the step saw the clipped gradient
        assert torch.allclose(optimizer.captured[n], raw[n] * coefficient, rtol=1e-4, atol=1e-8), n
    # a norm far below the gradient: clipping must act, the step sees exactly max_norm, the record is pre-clip
    monkeypatch.setattr(ws, 'MAX_GRAD_NORM', norm / 50)
    optimizer = CapturingOptimizer(tiny_model)
    step = ws.train_step(tiny_model, optimizer, items)
    assert step['clipped'] is True and step['clip_coefficient'] < 0.03
    assert abs(step['pre_clip_grad_norm'] - norm) <= 1e-5 * norm
    clipped = math.sqrt(math.fsum(float((v.double() ** 2).sum()) for v in optimizer.captured.values()))
    assert abs(clipped - norm / 50) <= 1e-4 * norm
    assert tiny_model.training is False and all(p.grad is None for p in tiny_model.parameters())


def test_train_step_changes_exactly_the_inventory_with_checkpointing_on(tiny_model, rendered):
    inventory = pilot.set_inventory(tiny_model, 1, 1)
    assert pilot.enable_checkpointing(tiny_model)['enabled'] is True
    tiny_model.eval()
    calls = []
    for index, layer in enumerate(tiny_model.model.layers):
        real = layer._gradient_checkpointing_func

        def counted(*args, _real=real, _index=index, **kwargs):
            calls.append(_index)
            return _real(*args, **kwargs)
        layer._gradient_checkpointing_func = counted
    params = {n: p.detach().clone() for n, p in tiny_model.named_parameters(remove_duplicate=False)}
    buffers = {n: b.detach().clone() for n, b in tiny_model.named_buffers()}
    optimizer = ws.build_optimizer(tiny_model)
    try:
        record = pilot.optimizer_record(optimizer)
        assert record['cls'] == 'torch.optim.adamw.AdamW' and record['numel'] == inventory['trainable_numel']
        assert {k: record['defaults'][k] for k in ('lr', 'betas', 'eps', 'weight_decay', 'foreach', 'capturable',
                                                    'amsgrad', 'maximize')} == {
            'lr': 1e-5, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0, 'foreach': False,
            'capturable': False, 'amsgrad': False, 'maximize': False}
        step = ws.train_step(tiny_model, optimizer, rendered.items[:2])
        assert calls == [0, 1] * 2                     # both rows ran every layer checkpointed (train mode)
        assert math.isfinite(step['loss']) and step['pre_clip_grad_norm'] > 0
        trainable = set(inventory['trainable_names'])
        assert trainable == {n for n in params if n.startswith('model.layers.1.')}
        for name, value in tiny_model.named_parameters(remove_duplicate=False):
            assert (not torch.equal(value.detach(), params[name])) == (name in trainable), name
        assert all(torch.equal(b, buffers[n]) for n, b in tiny_model.named_buffers())
        state = pilot.optimizer_state(optimizer, torch.device('cpu'))
        assert state['on_model_device'] and state['tensors'] == 3 * len(trainable)
        assert tiny_model.training is False and all(p.grad is None for p in tiny_model.parameters())
    finally:
        del optimizer
        gc.collect()


def test_optimizer_kwargs_are_the_registered_adamw():
    assert ws.optimizer_kwargs('cuda') == dict(ws.ADAMW, capturable=True)
    assert ws.optimizer_kwargs('cpu') == dict(ws.ADAMW, capturable=False)
    with pytest.raises(ValueError):
        ws.optimizer_kwargs('mps')


# ------------------------------------------------------------------------------------------------ delta and anchor
def _perturbed(tiny_dir, seed=7, scale=0.05):
    """The tiny model with its layer-1 inventory moved as a training run would move it (random, sign-crossing)."""
    model, _ = s0.load_model_fp32(tiny_dir.root, 'cpu', tiny_dir.asset)
    names = pilot.set_inventory(model, 1, 1)['trainable_names']
    params = dict(model.named_parameters())
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n in names:
            params[n].add_(torch.randn(params[n].shape, generator=generator) * scale)
    return model, names


def _stored_upcast(tiny_dir, names):
    with safe_open(str(tiny_dir.weights), framework='pt', device='cpu') as handle:
        return {n: handle.get_tensor(n).to(torch.float32) for n in names}


def test_delta_is_trained_minus_supa_and_create_only(tiny_dir, tmp_path):
    model, names = _perturbed(tiny_dir)
    params = dict(model.named_parameters())
    base = _stored_upcast(tiny_dir, names)
    trained = {n: params[n].detach().clone() for n in names}
    path = tmp_path / 'records' / ws.DELTA_NAME
    record = ws.write_warm_delta(model, tiny_dir.weights, names, path, tiny_dir.weights_sha256)
    assert path.is_file() and record['verified_on_disk'] is True and record['bytes'] == path.stat().st_size
    assert record['file_sha256'] == sha(path) and record['data_section_sha256'] == arm_build.delta_data_digest(path)
    delta = arm_build.read_delta(path, record['data_section_sha256'], record['state_sha256'])
    assert sorted(delta) == sorted(names) and record['tensors'] == len(names) and record['names_sha256'] == digest(
        sorted(names))
    for n in names:   # delta = trained - upcast(stored Supa), bitwise, over the inventory only
        assert delta[n].dtype == torch.float32 and torch.equal(delta[n], trained[n] - base[n]), n
    assert record['numel'] == sum(t.numel() for t in delta.values()) and record['nonzero'] == record['numel']
    assert abs(record['global_norm'] - math.sqrt(math.fsum(float((t.double() ** 2).sum()) for t in delta.values()))) \
        <= 1e-9 * record['global_norm']
    before = path.read_bytes()
    with pytest.raises(ValueError, match='create-only'):
        ws.write_warm_delta(model, tiny_dir.weights, names, path, tiny_dir.weights_sha256)
    assert path.read_bytes() == before and sorted(p.name for p in path.parent.iterdir()) == [ws.DELTA_NAME]
    with pytest.raises(ValueError, match='changed since the load'):
        ws.write_warm_delta(model, tiny_dir.weights, names, tmp_path / 'other.safetensors', '0' * 64)
    frozen = sorted(set(params) - set(names))[:1]
    with pytest.raises(ValueError, match='not trainable'):
        ws.write_warm_delta(model, tiny_dir.weights, frozen, tmp_path / 'frozen.safetensors', tiny_dir.weights_sha256)


def test_the_anchor_is_the_shared_digest_and_round_trips_through_load_anchor(tiny_dir, tmp_path):
    model, names = _perturbed(tiny_dir)
    # Explicit cancellation witness; random sign-crossing need not lose a bit.
    with torch.no_grad():
        param = dict(model.named_parameters())[names[0]]
        base = _stored_upcast(tiny_dir, [names[0]])[names[0]].reshape(-1)
        index = int(torch.nonzero(base, as_tuple=False)[0])
        param.reshape(-1)[index] = base[index] * (2.0 ** -26)
    trained_fingerprint = s0.state_fingerprint(model)
    trained = {n: p.detach().clone() for n, p in model.named_parameters() if n in names}
    path = tmp_path / ws.DELTA_NAME
    delta_sha = ws.write_warm_delta(model, tiny_dir.weights, names, path, tiny_dir.weights_sha256)[
        'data_section_sha256']
    arm, build = ws.materialize_anchor(model, tiny_dir.weights, path, delta_sha, names)
    assert arm == s0.anchor_digest(model) == s0.state_fingerprint(model) and build['arm_sha256'] == arm
    assert build['anchor_digest_rule'] == s0.ANCHOR_DIGEST_RULE and build['add_delta']['tensors'] == len(names)
    # the float32 round trip is not exact here, so the trained weights are not theta0' (why ANCHOR_DIGEST_RULE)
    assert trained_fingerprint != arm
    residual = build['residual']
    assert residual['elements'] > 0 and residual['tensors'] > 0 and residual['within_rounding_bound'] is True
    assert 0 < residual['max_bound_ratio'] <= 1.0
    # an independent rebuild, written out: the S0 FP32 loader, read_delta, param.add_ in FP32
    rebuilt, _ = s0.load_model_fp32(tiny_dir.root, 'cpu', tiny_dir.asset)
    delta = arm_build.read_delta(path, delta_sha)
    with torch.no_grad():
        slots = dict(rebuilt.named_parameters())
        for n, t in delta.items():
            slots[n].add_(t)
    assert s0.anchor_digest(rebuilt) == arm
    assert max(float((slots[n] - trained[n]).abs().max()) for n in names) == residual['max_abs']
    del rebuilt
    # the round trip every later stage makes: load_anchor accepts exactly this anchor
    anchor = {'base_weights_sha256': tiny_dir.weights_sha256, 'delta_path': str(path), 'delta_sha256': delta_sha,
              'arm_sha256': arm}
    loaded, record = ws.round_trip(tiny_dir.root, 'cpu', tiny_dir.asset, anchor)
    assert record['accepted'] is True and record['identity_anchor'] == anchor and record['state_fingerprint'] == arm
    assert record['in_memory_model_reused'] is False and s0.anchor_digest(loaded) == arm
    assert comp.anchor_matches(record['identity_anchor'], dict(anchor, warmstart_receipt_sha256='f' * 64))
    # a digest under any other convention is refused: the trained weights, or section 2's stored-name arm digest
    stored_names = record['stored_names_arm_sha256']
    _, streamed = arm_build.build_arm_streaming(tiny_dir.weights, [('WARM', path, 1.0)], order=('WARM',))
    assert streamed['arm_sha256'] == stored_names != arm
    for wrong in (trained_fingerprint, stored_names):
        with pytest.raises(arm_build.ArmRebuildStop):
            ws.round_trip(tiny_dir.root, 'cpu', tiny_dir.asset, dict(anchor, arm_sha256=wrong))
    with pytest.raises(ValueError, match='digest mismatch'):
        ws.round_trip(tiny_dir.root, 'cpu', tiny_dir.asset, dict(anchor, delta_sha256='0' * 64))


def test_the_anchor_refuses_a_delta_that_is_not_the_inventory(tiny_dir, tmp_path):
    model, names = _perturbed(tiny_dir, seed=11, scale=1e-3)
    path = tmp_path / ws.DELTA_NAME
    delta_sha = ws.write_warm_delta(model, tiny_dir.weights, names, path, tiny_dir.weights_sha256)[
        'data_section_sha256']
    state = s0.state_fingerprint(model)
    with pytest.raises(ValueError, match='not the trainable inventory'):
        ws.materialize_anchor(model, tiny_dir.weights, path, delta_sha, names[:-1])
    with pytest.raises(ValueError, match='digest mismatch'):
        ws.materialize_anchor(model, tiny_dir.weights, path, '0' * 64, names)
    assert s0.state_fingerprint(model) == state        # a refused anchor leaves the trained weights untouched


# ------------------------------------------------------------------------------------------------ envelope
def test_envelope_rule():
    cases = {'{"answer":[1,2]}': (True, True, ['answer_object']),
             '{"answer": "UNKNOWN"}': (True, True, ['answer_object']),
             '```json\n{"answer":[1]}\n```': (True, False, ['code_fence', 'answer_object']),
             '{"answer":[1],"why":"x"}': (True, False, ['extra_object_fields', 'answer_object']),
             ' {"answer":[1]}': (True, False, ['answer_object']),
             '[1,2]': (False, False, ['bare_list']), 'UNKNOWN': (False, False, ['bare_unknown']),
             'the answer is 3': (False, False, None), '': (False, False, None)}
    for response, (envelope, exact, categories) in cases.items():
        assert ws.envelope_of(response) == {'envelope': envelope, 'envelope_exact': exact,
                                            'envelope_categories': categories}, response


def test_envelope_prompts_are_the_s0_q1_probe_with_calibration_rows(tokenizer, comp_rs_data):
    prompts, inputs = ws.load_envelope_prompts(comp_rs_data, tokenizer)
    selection = read_json(q1.Q2_SELECTION)
    parents = selection['parent_worlds'][:4]
    assert [p['id'] for p in prompts] == [r['id'] for r in selection['rows'][:8]]
    assert [p['parent_id'] for p in prompts] == [x for parent in parents for x in (parent, parent)]
    assert [p['skill'] for p in prompts] == ['R', 'S'] * 4 and [p['prompt_tokens'] for p in prompts] == ENVELOPE_LENGTHS
    assert inputs['pinned'] is True and inputs['selection_receipt_sha256'] == q1.Q2_SELECTION_SHA
    assert inputs['calibration_bank_sha256'] == inputs['canonical_calibration_sha256']
    for p in prompts:
        assert p['row']['id'] == p['id'] and p['row']['role'] == 'calibration' and 'verifier' in p['row']
        assert p['prompt_ids'] == s0.render_prompt(tokenizer, p['row']['prompt'])[1]


def test_envelope_check_records_strict_content_and_envelope(tokenizer, comp_rs_data):
    prompts, _ = ws.load_envelope_prompts(comp_rs_data, tokenizer)
    answers = {p['id']: oracle.answers(p['row']['verifier']['spec'])[0] for p in prompts}
    bare = {i: json.dumps(json.loads(a)['answer'], separators=(',', ':')) for i, a in answers.items()}
    texts = {0: answers[prompts[0]['id']], 1: bare[prompts[1]['id']],
             2: '```json\n' + answers[prompts[2]['id']] + '\n```'}
    by_prompt = {tuple(p['prompt_ids']): p['position'] for p in prompts}
    seen = []

    class FakeG:
        @staticmethod
        def greedy_decode(model, prompt_ids, max_new_tokens):
            position = by_prompt[tuple(prompt_ids)]
            seen.append((position, max_new_tokens))
            ids = tokenizer(texts.get(position, 'no idea'), add_special_tokens=False)['input_ids'] + [IM_END]
            return {'completion_ids': ids, 'stop_reason': 'eos', 'generated_tokens': len(ids),
                    'raw_step_logprobs': [0.0] * len(ids), 'decoder': {'mode': 'greedy'}}
    host, records = FakeHost(), []
    ws.envelope_check(FakeG, torch.nn.Linear(1, 1), tokenizer, prompts, 7, host, ws.LABEL, records)
    assert seen == [(k, 7) for k in range(8)]
    assert host.calls == [('guard', 1.0, ws.LABEL + ' envelope ' + p['id']) for p in prompts]
    assert [r['id'] for r in records] == [p['id'] for p in prompts]
    assert [r['envelope'] for r in records] == [True, False, True] + [False] * 5
    assert [r['envelope_exact'] for r in records] == [True] + [False] * 7
    assert [r['reward'] for r in records] == [1] + [0] * 7
    assert [r['content_pass'] for r in records] == [True, True, True] + [False] * 5
    for r, p in zip(records, prompts):
        assert r['reward'] == corpus.reward(p['row'], r['response'])
        assert r['response'] == comp.response_text(tokenizer, r['completion_ids'], 'eos')['response']
        assert r['wall_s'] >= 0 and r['prompt_tokens'] == p['prompt_tokens'] and r['decoder'] == {'mode': 'greedy'}
    summary = ws.envelope_summary(records)
    assert summary['envelope'] == 2 and summary['envelope_fraction'] == 0.25 and summary['envelope_exact'] == 1
    assert summary['strict_pass'] == 1 and summary['content_pass'] == 3 and summary['gated'] is False
    assert summary['by_skill']['R']['strict_pass'] == 1 and summary['by_skill']['S']['content_pass'] == 1
    assert summary['stop_reasons'] == {'eos': 8, 'length': 0}


def durable_envelope_fixture(tokenizer, comp_rs_data):
    """A CPU decoder with real streamer/forward calls and the fixed eight scoring rows."""
    prompts, _ = ws.load_envelope_prompts(comp_rs_data, tokenizer)
    outputs = {tuple(p['prompt_ids']): tokenizer(oracle.answers(p['row']['verifier']['spec'])[0],
              add_special_tokens=False)['input_ids'] + [IM_END] for p in prompts}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.zeros(1))
            self.calls, self.fail_after = [], None
        def forward(self, input_ids):
            return input_ids.float()
        def generate(self, input_ids, max_new_tokens, streamer):
            key = tuple(input_ids)
            self.calls.append(key)
            ids = outputs[key]
            assert len(ids) <= max_new_tokens
            streamer.put(torch.tensor([input_ids]))
            for index, token in enumerate(ids):
                self(torch.tensor([[token]]))
                streamer.put(torch.tensor([token]))
                if self.fail_after == index + 1:
                    raise RuntimeError('synthetic mid-generation interruption')
            streamer.end()
            return {'completion_ids': ids, 'stop_reason': 'eos', 'generated_tokens': len(ids),
                    'raw_step_logprobs': [0.0] * len(ids), 'decoder': {'mode': 'greedy'}}
    class G:
        @staticmethod
        def greedy_decode(model, prompt_ids, max_new_tokens):
            return model.generate(input_ids=prompt_ids, max_new_tokens=max_new_tokens)
    binding = ws.envelope_binding({'arm_sha256': 'a' * 64}, 'b' * 64, prompts, 128)
    return prompts, Model(), G, binding


def test_envelope_resume_skips_committed_rows_without_rescoring(tmp_path, tokenizer, comp_rs_data, monkeypatch):
    prompts, model, decoder, binding = durable_envelope_fixture(tokenizer, comp_rs_data)
    store = ws.checkpoints.Store(tmp_path, 1, {'source': 'bound'}, usage=lambda p: SimpleNamespace(free=64 * s0.GIB))
    records = []
    with pytest.raises(FakeHost.LowMemoryAbort):
        ws.envelope_check(decoder, model, tokenizer, prompts, 128, FakeHost(abort_at=2), ws.LABEL, records,
                          store=store, binding=binding)
    assert len(records) == 1
    first = json.loads(ws.checkpoints.canonical(records[0]))
    store.close()
    scorer = comp.score_completion
    def check(row, *args, **kwargs):
        assert row['id'] != prompts[0]['id'], 'a completed read was rescored'
        return scorer(row, *args, **kwargs)
    monkeypatch.setattr(comp, 'score_completion', check)
    resumed = ws.checkpoints.Store(tmp_path, 1, {'source': 'bound'}, resume=True, reason='host stand-down', usage=lambda p: SimpleNamespace(free=64 * s0.GIB))
    try:
        restored = []
        ws.envelope_check(decoder, model, tokenizer, prompts, 128, FakeHost(), ws.LABEL, restored,
                          store=resumed, binding=binding)
        assert len(restored) == 8 and restored[0] == first
        assert model.calls == [tuple(p['prompt_ids']) for p in prompts]
        assert resumed.cost()['generation_attempted'] == resumed.cost()['generation_returned'] == 8
        assert resumed.cost()['envelope_score_returned'] == 8
        assert all(e['row_sha256'] == digest(e['row']) for e in resumed.events if e['kind'] == 'envelope_row')
        with pytest.raises(ValueError, match='scope drift'):
            resumed.envelope_rows(dict(binding, arm_sha256='c' * 64))
    finally:
        resumed.close()


def test_mid_generation_attempt_emissions_remain_charged_on_resume(tmp_path, tokenizer, comp_rs_data):
    prompts, model, decoder, binding = durable_envelope_fixture(tokenizer, comp_rs_data)
    model.fail_after = 2
    store = ws.checkpoints.Store(tmp_path, 1, {'source': 'bound'}, usage=lambda p: SimpleNamespace(free=64 * s0.GIB))
    with pytest.raises(RuntimeError, match='mid-generation'):
        ws.envelope_check(decoder, model, tokenizer, prompts, 128, FakeHost(), ws.LABEL, [], store=store, binding=binding)
    partial = store.envelope_cost()['attempts'][0]
    assert partial['known_emitted_tokens'] == 2 and partial['emitted_tokens_upper_bound'] == 128
    assert partial['decoder_forward_attempted'] == partial['decoder_forward_returned'] == 2
    assert not store.envelope_rows(binding)
    assert 'generate' not in vars(model) and not model._forward_hooks and not model._forward_pre_hooks
    store.close()
    model.fail_after = None
    resumed = ws.checkpoints.Store(tmp_path, 1, {'source': 'bound'}, resume=True, reason='mid-generation interruption', usage=lambda p: SimpleNamespace(free=64 * s0.GIB))
    try:
        records = []
        ws.envelope_check(decoder, model, tokenizer, prompts, 128, FakeHost(), ws.LABEL, records, store=resumed, binding=binding)
        assert len(records) == 8 and resumed.cost()['generation_attempted'] == 9
        assert resumed.cost()['generation_returned'] == 8
        assert resumed.envelope_cost()['known_emitted_tokens'] == 2 + sum(r['generated_tokens'] for r in records)
        assert resumed.envelope_cost()['emitted_tokens_upper_bound'] == 128 + sum(r['generated_tokens'] for r in records)
        assert resumed.cost()['decoder_forward_returned'] == 2 + sum(r['generated_tokens'] for r in records)
    finally:
        resumed.close()


def test_durable_return_before_row_publication_recovers_without_redecoding(tmp_path, tokenizer, comp_rs_data, monkeypatch):
    prompts, model, decoder, binding = durable_envelope_fixture(tokenizer, comp_rs_data)
    store = ws.checkpoints.Store(tmp_path, 1, {'source': 'bound'}, usage=lambda p: SimpleNamespace(free=64 * s0.GIB))
    append = store.append
    def interrupted(kind, **fields):
        if kind == 'envelope_row':
            raise RuntimeError('interruption before row publication')
        return append(kind, **fields)
    monkeypatch.setattr(store, 'append', interrupted)
    with pytest.raises(RuntimeError, match='before row publication'):
        ws.envelope_check(decoder, model, tokenizer, prompts, 128, FakeHost(), ws.LABEL, [], store=store, binding=binding)
    store.close()
    resumed = ws.checkpoints.Store(tmp_path, 1, {'source': 'bound'}, resume=True, reason='returned result before row publish', usage=lambda p: SimpleNamespace(free=64 * s0.GIB))
    try:
        records = []
        ws.envelope_check(decoder, model, tokenizer, prompts, 128, FakeHost(), ws.LABEL, records, store=resumed, binding=binding)
        assert len(records) == 8 and len(model.calls) == 8
        assert resumed.cost()['generation_attempted'] == 8 and resumed.cost()['envelope_score_returned'] == 9
        assert any(e['kind'] == 'envelope_return_reused' for e in resumed.events)
    finally:
        resumed.close()


# ------------------------------------------------------------------------------------------------ launch and CLI
def test_the_module_imports_no_torch_so_the_launch_snapshot_precedes_it():
    code = ('import json, sys\n'
            'import experiments.maxrl_comp002_pretraining_v1.s0_warmstart as w\n'
            "print(json.dumps(sorted(m for m in ('torch', 'transformers', 'numpy', 'safetensors') if m in sys.modules)))\n")
    run = subprocess.run([sys.executable, '-c', code], cwd=str(REPO), capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    assert json.loads(run.stdout.strip().splitlines()[-1]) == []


def test_registered_mode_takes_no_overrides_and_refuses_before_any_work(tmp_path, monkeypatch):
    def no_work(*args, **kwargs):
        raise AssertionError('work started after a refused command line')
    monkeypatch.setattr(comp, 'prepare_launch', no_work)
    monkeypatch.setattr(s0, 'hostbudget', no_work)
    test = ['--device', 'cpu', '--test-mode', '--host-floor-gib', '1']
    existing = tmp_path / 'existing'
    _create(existing / ws.RECEIPT_NAME, b'{}\n')
    delta_only = tmp_path / 'delta_only'
    _create(delta_only / ws.DELTA_NAME, b'x')
    cases = [
        (['--rows', '16'], None, 'test-mode only option'), (['--test-blocks', '1', '1'], None, 'test-mode only'),
        (['--model-dir', str(tmp_path)], None, 'test-mode only'), (['--host-floor-gib', '3'], None, 'test-mode only'),
        (['--max-new-tokens', '4'], None, 'test-mode only'), (['--device', 'cpu'], None, 'runs on cuda'),
        ([], FakeHost(), 'test-mode only'), (['--repo', str(tmp_path)], None, 'binds the checkout'),
        (['--records', str(tmp_path)], None, 'registered records are written to'),
        (['--lease-timeout-s', 'nan'], None, 'lease timeout'),
        # --test-mode never runs on the GPU: refused with --device cuda, given or by default
        (['--test-mode', '--device', 'cuda', '--records', str(tmp_path / 'r'), '--host-floor-gib', '1'], FakeHost(),
         'refused with --device cuda'),
        (['--test-mode', '--records', str(tmp_path / 'r'), '--host-floor-gib', '1'], None, 'refused with --device cuda'),
        (['--device', 'cpu', '--test-mode', '--host-floor-gib', '1'], None, 'needs --records'),
        (['--device', 'cpu', '--test-mode', '--records', str(tmp_path / 'r')], None, 'host-floor-gib'),
        (test + ['--records', str(tmp_path / 'r'), '--rows', '12'], None, 'multiple of 8'),
        (test + ['--records', str(tmp_path / 'r'), '--rows', '608'], None, 'multiple of 8'),
        (test + ['--records', str(tmp_path / 'r'), '--rows', '0'], None, 'multiple of 8'),
        (test + ['--records', str(tmp_path / 'r'), '--max-new-tokens', '0'], None, 'max_new_tokens'),
        (test + ['--records', str(existing)], None, 'create-only'),
        (test + ['--records', str(delta_only)], None, 'create-only')]
    for argv, host, match in cases:
        with pytest.raises(SystemExit, match=match):
            ws.main(argv, host_module=host)
    with pytest.raises(ValueError, match='inside the bound checkout'):
        ws.main(test + ['--records', str(REPO / 'records')])
    assert not (tmp_path / 'r').exists() and not (REPO / 'records').exists()


def _supa_freeze(tmp_path, anchor=None):
    preimage = {'schema': 'COMP-RS-001-S0-DECODING-1', 'device': 'cpu', 'dtype': 'float32'}
    value = {'stage': 'S0_RUNTIME_FREEZE', 'test_mode': True, 'registered_run': False, 'decoding_preimage': preimage,
             'decoding_sha256': s0.decoding_sha256(preimage), 'host_floor_gib': 0.5,
             'scoring_code_bundle_sha256': comp.live_bundle(), 'inventory': {'name': 'PENDING'}, 'anchor': anchor}
    return _dump(tmp_path / ('freeze_warm.json' if anchor else 'freeze.json'), value)


def test_the_freeze_must_be_plain_supa(tmp_path, monkeypatch):
    monkeypatch.setattr(comp, 'prepare_launch', lambda *a, **k: (_ for _ in ()).throw(AssertionError('launched')))
    anchor = {'base_weights_sha256': 'a' * 64, 'delta_path': str(tmp_path / 'd.safetensors'), 'delta_sha256': 'b' * 64,
              'arm_sha256': 'c' * 64, 'warmstart_receipt_sha256': 'd' * 64}
    base = ['--device', 'cpu', '--test-mode', '--records', str(tmp_path / 'r')]
    with pytest.raises(SystemExit, match='must carry no anchor'):
        ws.main(base + ['--runtime-freeze', str(_supa_freeze(tmp_path, anchor))], host_module=FakeHost())
    args = ws._parser().parse_args(base + ['--runtime-freeze', str(_supa_freeze(tmp_path))])
    cfg = ws._config(args, None, FakeHost())
    assert cfg['freeze']['host_floor_gib'] == 0.5 and cfg['freeze']['anchor'] is None


def test_registered_defaults_read_the_real_supa_freeze():
    freeze_path = s0.RECORDS / 'S0_RUNTIME_FREEZE.json'
    if not freeze_path.is_file() or (s0.RECORDS / ws.RECEIPT_NAME).exists() or (s0.RECORDS / ws.DELTA_NAME).exists():
        pytest.skip('the real S0 runtime freeze is absent or the warm-start already ran')
    args = ws._parser().parse_args([])
    cfg = ws._config(args, None, None)
    assert {k: cfg[k] for k in ('rows', 'blocks', 'max_new_tokens', 'device', 'label', 'stage', 'test_mode')} == {
        'rows': 600, 'blocks': [14, 27], 'max_new_tokens': 128, 'device': 'cuda:0', 'label': 'comp-rs-001-s0-warmstart',
        'stage': 'S0_WARMSTART', 'test_mode': False}
    assert cfg['out'] == s0.RECORDS / ws.RECEIPT_NAME and cfg['delta'] == s0.RECORDS / ws.DELTA_NAME
    assert args.runtime_freeze == freeze_path   # the registered default floor source
    assert cfg['model_dir'] == s0.MODEL_DIR and cfg['asset_receipt'] == s0.ASSET_RECEIPT
    assert cfg['freeze']['sha256'] == sha(freeze_path) and cfg['freeze']['anchor'] is None


def test_registered_launch_reads_the_freeze_floor_and_asserts_a_clean_checkout(monkeypatch):
    freeze_path = s0.RECORDS / 'S0_RUNTIME_FREEZE.json'
    if not freeze_path.is_file() or (s0.RECORDS / ws.RECEIPT_NAME).exists() or (s0.RECORDS / ws.DELTA_NAME).exists():
        pytest.skip('the real S0 runtime freeze is absent or the warm-start already ran')
    calls = []
    monkeypatch.setattr(s0, 'assert_runtime', lambda: calls.append('runtime') or {'python': 'x'})
    monkeypatch.setattr(s0, 'assert_clean_checkout', lambda repo: calls.append(('checkout', Path(repo))) or {
        'repo': str(repo), 'head': 'h', 'clean': True})
    args = ws._parser().parse_args([])
    cfg = ws._config(args, None, None)
    host = FakeHost()
    prepared = comp.prepare_launch(args, cfg, host)
    floor = read_json(freeze_path)['host_floor_gib']
    assert host.calls == [('require', floor, ws.LABEL)] and prepared['launch']['floor']['floor_gib'] == floor
    assert prepared['launch']['floor']['source'] == 'S0_RUNTIME_FREEZE host_floor_gib'
    assert calls == ['runtime', ('checkout', ws.REPO)] and prepared['checkout']['checked'] is True
    assert prepared['freeze']['sha256'] == sha(freeze_path)


def test_cli_refuses_to_start_below_the_launch_floor(tmp_path, tiny_dir, comp_rs_data, monkeypatch):
    def no_load(*args, **kwargs):
        raise AssertionError('loaded past a refused launch floor')
    monkeypatch.setattr(ws, 'load_warmstart', no_load)
    monkeypatch.setattr(s0, 'load_model_fp32', no_load)
    lease, host = FakeLease(), FakeHost(free=0.4)
    with pytest.raises(SystemExit, match='REFUSING TO START comp-rs-001-s0-warmstart'):
        run_cli(tmp_path / 'records', tiny_dir, comp_rs_data, lease=lease, host=host)
    assert host.calls == [('require', 0.5, ws.LABEL)] and lease.events == []
    assert not (tmp_path / 'records').exists()


def test_cli_runs_a_short_warm_start_on_cpu_and_trains_only_the_inventory(short_run, tiny_dir, comp_rs_data,
                                                                          tokenizer, warm, rendered):
    result = short_run
    receipt = result.receipt
    assert result.code == 0 and receipt['outcome'] == 'WARMSTART_COMPLETE' and receipt['completed'] is True
    assert receipt['error'] is None and receipt['abort'] is None and receipt['stage'] == 'S0_WARMSTART'
    assert receipt['test_mode'] is True and receipt['registered_run'] is False and receipt['argv'] == result.argv
    assert receipt['schema'] == ws.SCHEMA and receipt['checkout'] == {'checked': False, 'reason': 'test mode'}
    # plan: the first 16 rows of the registered order, 2 steps of 8
    order = ws.order_ids([row['id'] for row in warm.rows])
    plan = receipt['plan']
    assert plan['rows'] == 16 and plan['steps'] == 2 and plan['subset'] is True and plan['accumulation'] == 8
    assert plan['row_order_sha256'] == PINNED_ORDER_DIGEST and plan['selected_ids_sha256'] == digest(order[:16])
    steps = receipt['steps']
    assert [s['index'] for s in steps] == [0, 1] and [s['row_ids'] for s in steps] == [order[:8], order[8:16]]
    for s in steps:
        assert s['rows'] == 8 and s['loss'] == math.fsum(s['row_losses']) / 8 and len(s['row_losses']) == 8
        assert s['row_labelled_tokens'] == [len(rendered.by_id[i]['completion_ids']) for i in s['row_ids']]
        assert s['labelled_tokens'] == sum(s['row_labelled_tokens']) and s['sequence_tokens'] == sum(
            s['row_sequence_tokens'])
        assert s['pre_clip_grad_norm'] > 0 and s['wall_s'] > 0 and s['resources']['cuda'] is None
        assert s['clip_coefficient'] == min(1.0, 1.0 / (s['pre_clip_grad_norm'] + 1e-6))
    assert receipt['training']['steps'] == 2 and receipt['training']['labelled_tokens'] == sum(
        s['labelled_tokens'] for s in steps)
    # recipe and inputs
    recipe = receipt['recipe']
    assert recipe['loss_rule'] == ws.LOSS_RULE and recipe['gradient_clipping']['max_norm'] == 1.0
    assert recipe['optimizer']['kwargs'] == {'lr': 1e-5, 'betas': [0.9, 0.999], 'eps': 1e-8, 'weight_decay': 0.0,
                                             'amsgrad': False, 'maximize': False, 'foreach': False,
                                             'capturable': False}
    assert recipe['registered_rows'] == 600 and recipe['registered_steps'] == 75 and recipe['epochs'] == 1
    assert receipt['inputs']['files'] == warm.inputs['files'] and receipt['inputs']['rendering'] == rendered.record
    assert receipt['optimizer']['defaults']['lr'] == 1e-5 and receipt['optimizer_state_after_first_step'][
        'on_model_device'] is True
    # scope: frozen bitwise unchanged, trainable changed, the inventory is layer 1 only
    fingerprints = receipt['fingerprints']
    assert fingerprints['frozen_unchanged'] is True and fingerprints['trainable_changed'] is True
    assert fingerprints['before']['state'] == receipt['identity']['state_fingerprint']
    assert fingerprints['final_equals_theta0'] is True and receipt['inventory']['blocks'] == [1, 1]
    assert receipt['gradient_checkpointing']['enabled'] is True
    layer1 = sorted(n for n in tiny_dir.stored if n.startswith('model.layers.1.'))
    assert receipt['inventory']['trainable_names_sha256'] == digest(layer1)
    # the delta: create-only file in the records dir, layer 1 only, nonzero
    record = receipt['delta']
    assert Path(record['path']) == result.delta and result.delta.is_file() and record['file_sha256'] == sha(result.delta)
    delta = arm_build.read_delta(result.delta, record['data_section_sha256'], record['state_sha256'])
    assert sorted(delta) == layer1 and record['nonzero'] > 0 and record['names_sha256'] == digest(layer1)
    assert all(bool(t.abs().max() > 0) for t in delta.values())
    # theta0' (the round-trip rebuild): Supa everywhere outside the inventory, Supa + delta on it
    anchor = receipt['anchor']
    assert set(anchor) == set(s0.ANCHOR_KEYS) and anchor['delta_path'] == str(result.delta.resolve())
    assert anchor['delta_sha256'] == record['data_section_sha256'] and anchor['base_weights_sha256'] == \
        tiny_dir.weights_sha256
    assert receipt['anchor_round_trip']['accepted'] is True and receipt['anchor_round_trip']['identity_anchor'] == anchor
    assert receipt['anchor_build']['arm_sha256'] == anchor['arm_sha256'] and receipt['anchor_digest_rule'] == \
        s0.ANCHOR_DIGEST_RULE
    assert receipt['anchor_build']['trained_equals_theta0'] is (
        receipt['anchor_build']['trained_state_fingerprint'] == anchor['arm_sha256'])
    rebuilt, identity = s0.load_anchor(tiny_dir.root, 'cpu', tiny_dir.asset, **comp.anchor_arguments(anchor))
    assert identity['state_fingerprint'] == anchor['arm_sha256'] == s0.anchor_digest(rebuilt)
    base = _stored_upcast(tiny_dir, tiny_dir.stored)
    state = rebuilt.state_dict()
    for n in tiny_dir.stored:
        assert torch.equal(state[n], base[n] + delta[n] if n in delta else base[n]), n
    # the envelope check: 8 greedy decodes of the S0-Q1 prompts on theta0'
    envelope = receipt['envelope']
    selection = read_json(q1.Q2_SELECTION)
    assert [r['id'] for r in envelope['records']] == [r['id'] for r in selection['rows'][:8]]
    assert envelope['max_new_tokens'] == 3 and envelope['summary']['prompts'] == 8 and envelope['recorded_not_gated']
    assert envelope['decoder']['record']['mode'] == 'greedy' and envelope['decoder']['rows'] == 8
    prompts, _ = ws.load_envelope_prompts(comp_rs_data, tokenizer)
    for r, p in zip(envelope['records'], prompts):
        again = g.greedy_decode(rebuilt, p['prompt_ids'], max_new_tokens=3)
        assert again['completion_ids'] == r['completion_ids'] and r['generated_tokens'] <= 3
        assert r['envelope'] == ws.envelope_of(r['response'])['envelope'] and 'decoder' not in r
        assert r['reward'] == corpus.reward(p['row'], r['response']) and 'content_pass' in r
    # memory, runtime, lease and host
    assert receipt['memory']['peak_worker_rss_gib'] > 0 and receipt['memory']['max_memory_reserved_gib'] is None
    assert receipt['settings']['deterministic_algorithms'] is True and receipt['settings']['num_threads'] == 4
    assert receipt['decoding_sha256'] == s0.decoding_sha256(receipt['decoding_preimage'])
    assert receipt['code_sha256'] == comp.code_sha256(ws.CODE_FILES)
    assert result.lease.events == [('acquire', 'comp-rs-001-s0-warmstart'), ('release', 'token-ws')]
    numel = sum(t.numel() for t in delta.values())
    guards = [c for c in result.host.calls if c[0] == 'guard']
    assert result.host.calls[0] == ('require', 0.5, ws.LABEL)
    assert guards == ([('guard', 1.0, ws.LABEL + ' step 0'), ('guard', 1.0, ws.LABEL + ' step 1'),
                       ('guard', ws.delta_write_gib(numel), ws.LABEL + ' delta write'),
                       ('guard', 1.0 + 4 * numel / s0.GIB, ws.LABEL + ' anchor'),
                       ('guard', 1.0 + 4 * numel / s0.GIB, ws.LABEL + ' round trip')]
                      + [('guard', 1.0, ws.LABEL + ' envelope ' + r['id']) for r in selection['rows'][:8]])
    assert b'\r' not in result.raw and result.raw.endswith(b'\n') and result.raw.isascii()
    assert sorted(p.name for p in result.records.iterdir()) == sorted([ws.RECEIPT_NAME, ws.DELTA_NAME,
           ws.checkpoints.CLAIM, ws.checkpoints.JOURNAL, ws.checkpoints.DIRECTORY])
    assert receipt['optimizer_state_saved'] and receipt['durability']['cost']['optimizer_step_returned'] == 2
    assert not list((result.records / ws.checkpoints.DIRECTORY).glob('*.json'))


def test_the_receipt_is_what_the_theta0_freeze_reads(short_run, tiny_dir, tmp_path):
    """s0_freeze's theta0' mode builds a freeze from this receipt; S0-C's reader returns its anchor, and the anchor
    arguments it hands load_anchor rebuild theta0'."""
    receipt = short_run.receipt
    q1_path = _dump(tmp_path / 'S0_Q1_FP32.json', cpu_q1_receipt(receipt))
    value = s0_freeze.build_warm(q1_path, short_run.path, test_mode=True, asset_receipt=tiny_dir.asset, repo=REPO)
    assert value['anchor'] == dict(receipt['anchor'], warmstart_receipt_sha256=sha(short_run.path))
    assert value['anchor_digest_rule'] == receipt['anchor_digest_rule'] == s0.ANCHOR_DIGEST_RULE
    out = _dump(tmp_path / 'S0_RUNTIME_FREEZE_WARM.json', value)
    read = comp.read_freeze(out, registered=False)
    assert read['anchor'] == value['anchor'] and read['inventory']['name'] == 'PENDING'
    model, identity = s0.load_anchor(tiny_dir.root, 'cpu', tiny_dir.asset, **comp.anchor_arguments(read['anchor']))
    assert comp.anchor_matches(identity['anchor'], read['anchor'])
    assert identity['state_fingerprint'] == receipt['anchor']['arm_sha256']
    del model
    # a registered freeze refuses this test-mode receipt: it reads the registered-run fields too
    with pytest.raises(ValueError, match='not a registered run'):
        s0_freeze.check_warmstart(receipt, test_mode=False, base_weights_sha256=tiny_dir.weights_sha256, repo=REPO)
    # a drifted receipt field name would be refused (the reader keys on these exact names)
    for key in s0.ANCHOR_KEYS:
        renamed = dict(receipt, anchor={(k + '_x' if k == key else k): v for k, v in receipt['anchor'].items()})
        with pytest.raises(ValueError, match='lacks: ' + key):
            s0_freeze.check_warmstart(renamed, test_mode=True, base_weights_sha256=tiny_dir.weights_sha256, repo=REPO)


def test_cli_is_create_only(short_run, tiny_dir, comp_rs_data):
    before = {str(p.relative_to(short_run.records)): p.read_bytes() for p in short_run.records.rglob('*') if p.is_file()}
    with pytest.raises(SystemExit, match='create-only'):
        run_cli(short_run.records, tiny_dir, comp_rs_data)
    assert {str(p.relative_to(short_run.records)): p.read_bytes() for p in short_run.records.rglob('*') if p.is_file()} == before


def test_cli_stands_down_on_the_host_guard_without_a_delta(tmp_path, tiny_dir, comp_rs_data):
    result = run_cli(tmp_path / 'records', tiny_dir, comp_rs_data, host=FakeHost(abort_at=2))
    receipt = result.receipt
    assert result.code == 2 and receipt['outcome'] == 'ABORTED_HOST_GUARD' and receipt['completed'] is False
    assert receipt['abort']['before'] == 'step 1' and receipt['abort']['type'] == 'LowMemoryAbort'
    assert len(receipt['steps']) == 1 and receipt['delta'] is None and receipt['anchor'] is None
    assert receipt['anchor_build'] is None and receipt['anchor_round_trip'] is None
    assert not result.delta.exists() and receipt['envelope']['records'] == []
    assert receipt['fingerprints']['frozen_unchanged'] is True
    assert result.lease.events[-1] == ('release', 'token-ws')


@pytest.mark.parametrize('abort_at,cursor', [(2, 1), (3, 2)])
def test_cli_resumes_boundary_or_final_save_without_repeating_committed_updates(tmp_path, tiny_dir, comp_rs_data,
                                                                              short_run, abort_at, cursor):
    records = tmp_path / 'resumable'
    interrupted = run_cli(records, tiny_dir, comp_rs_data, host=FakeHost(abort_at=abort_at))
    assert interrupted.code == 2 and interrupted.receipt['training']['steps'] == cursor
    claim_bytes = (records / ws.checkpoints.CLAIM).read_bytes()
    old_receipt = interrupted.path.read_bytes()
    resumed = run_cli(records, tiny_dir, comp_rs_data, '--resume', '--resume-reason', 'documented synthetic host stand-down')
    assert resumed.code == 0 and resumed.receipt['durability']['resume_cursor'] == cursor
    assert resumed.receipt['training']['steps'] == 2
    assert resumed.receipt['durability']['cost']['optimizer_step_returned'] == 2
    assert resumed.receipt['delta']['state_sha256'] == short_run.receipt['delta']['state_sha256']
    assert resumed.receipt['anchor']['arm_sha256'] == short_run.receipt['anchor']['arm_sha256']
    assert (records / ws.checkpoints.CLAIM).read_bytes() == claim_bytes and interrupted.path.read_bytes() == old_receipt


def test_an_incomplete_run_offers_no_anchor_to_freeze(tmp_path, tiny_dir, comp_rs_data):
    """A stand-down at the round trip: the delta is written and theta0' digested, but the receipt's anchor is None,
    so s0_freeze refuses the incomplete receipt."""
    result = run_cli(tmp_path / 'records', tiny_dir, comp_rs_data, host=FakeHost(abort_at=4), rows=8)
    receipt = result.receipt
    assert result.code == 2 and receipt['outcome'] == 'ABORTED_HOST_GUARD'
    assert receipt['abort']['before'] == 'round trip' and result.delta.is_file()
    assert receipt['anchor'] is None and receipt['anchor_round_trip'] is None
    assert receipt['anchor_build']['arm_sha256'] and receipt['envelope']['records'] == []
    with pytest.raises(ValueError, match='did not complete successfully'):
        s0_freeze.check_warmstart(receipt, test_mode=True, base_weights_sha256=tiny_dir.weights_sha256, repo=REPO)


def test_cli_failure_writes_a_receipt_releases_the_lease_and_reraises(tmp_path, tiny_dir, comp_rs_data, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError('anchor failed (simulated)')
    monkeypatch.setattr(ws, 'materialize_anchor', broken)
    lease = FakeLease()
    records = tmp_path / 'records'
    with pytest.raises(ValueError, match='simulated'):
        run_cli(records, tiny_dir, comp_rs_data, lease=lease, rows=8)
    receipt = json.loads(next(records.glob('S0_WARMSTART_INTERRUPTION_*.json')).read_bytes().decode('ascii'))
    assert receipt['error'] == {'type': 'ValueError', 'message': 'anchor failed (simulated)'}
    assert receipt['outcome'] == 'FAILED' and receipt['completed'] is False
    assert receipt['delta']['path'] == str((records / ws.DELTA_NAME).resolve())   # the written delta is recorded
    assert (records / ws.DELTA_NAME).is_file() and receipt['anchor'] is None and receipt['anchor_build'] is None
    assert lease.events == [('acquire', 'comp-rs-001-s0-warmstart'), ('release', 'token-ws')]


def test_cli_writes_no_receipt_if_its_code_changed_during_the_run(tmp_path, tiny_dir, comp_rs_data, monkeypatch):
    real = comp.code_sha256
    monkeypatch.setattr(comp, 'code_sha256', lambda files: dict(real(files), **{ws.PACKAGE + 's0_warmstart.py': '0'}))
    with pytest.raises(ValueError, match='warm-start source changed before claim'):
        run_cli(tmp_path / 'records', tiny_dir, comp_rs_data, host=FakeHost(abort_at=1), rows=8)
    assert not (tmp_path / 'records' / ws.RECEIPT_NAME).exists()
