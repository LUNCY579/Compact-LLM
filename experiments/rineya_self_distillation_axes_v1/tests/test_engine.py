import copy
import json
from types import SimpleNamespace
import pytest
import torch
from torch import nn
from torch.nn import functional as F
from experiments.rineya_self_distillation_axes_v1.engine import Engine
from experiments.rineya_self_distillation_axes_v1.native import execute, fixed_precision, verify_qualification
from experiments.rineya_self_distillation_axes_v1.plan import Arm, arms, checkpoint_manifest, validate_config
from experiments.rineya_self_distillation_axes_v1.teacher import tensor_digest
from experiments.rineya_self_distillation_axes_v1.__main__ import main


def config():
    return {'checkpoint': '/not_loaded', 'checkpoint_sha256': 'a'*64, 'dataset': '/not_read', 'dataset_sha256': 'b'*64,
            'task': 'new', 'seed': 17, 'steps': 3, 'learning_rate': .001,
            'trainable_names': ['q_proj.weight'], 'max_trainable_parameters': 1000,
            'max_new_tokens': 3, 'max_sequence_tokens': 512, 'device': 'cpu',
            'min_free_cuda_before_gib': 4., 'min_free_cuda_during_gib': .25}


def rows():
    return [{'id': str(i), 'parent_id': str(i), 'task': 'new' if i < 3 else 'old',
             'split': 'train' if i < 2 else 'dev', 'purpose': 'acquisition' if i < 3 else 'retention',
             'source': 'tiny fixture only', 'admitted_for': 'train' if i < 2 else 'dev',
             'prompt': 'Question ' + str(i), 'reference': 'OK', 'answer': 'OK', 'scorer': 'exact'}
            for i in range(4)]


class TinyTokenizer:
    eos_token_id = 2
    chat_template = 'fixture'
    def encode(self, text, **kwargs):
        return [3 + ord(c) % 61 for c in text]
    def apply_chat_template(self, messages, **kwargs):
        return [1] + self.encode(messages[0]['content']) + [1]
    def decode(self, tokens, **kwargs):
        return ''.join(chr(t + 30) for t in tokens if t != self.eos_token_id)


class TinyLM(nn.Module):
    """BF16 attention fixture. It cannot pass the native 0.6B loader guard."""
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(vocab_size=64)
        self.embed = nn.Embedding(64, 16).bfloat16()
        self.q_proj = nn.Linear(16, 16, bias=False)
        self.k_proj = nn.Linear(16, 8, bias=False).bfloat16()
        self.v_proj = nn.Linear(16, 8, bias=False).bfloat16()
        self.head = nn.Linear(16, 64, bias=False).bfloat16()
        self.requires_grad_(False)
        self.q_proj.weight.requires_grad_(True)
    def forward(self, input_ids, **kwargs):
        x = self.embed(input_ids)
        b, n, _ = x.shape
        q = self.q_proj(x).view(b,n,2,8).transpose(1,2)
        k = self.k_proj(x).view(b,n,1,8).transpose(1,2)
        v = self.v_proj(x).view(b,n,1,8).transpose(1,2)
        y = F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=True)
        return SimpleNamespace(logits=self.head(y.transpose(1,2).reshape(b,n,16)))


def make_engine():
    torch.manual_seed(17)
    return Engine(TinyLM(), TinyTokenizer(), config())


@pytest.mark.parametrize('arm', arms((0., .02), True), ids=lambda a: a.name)
def test_native_mechanics_all_arms_on_tiny_fixture(arm):
    engine = make_engine()
    initial = tensor_digest(engine.model.named_parameters())
    with fixed_precision():
        result = engine.train(rows(), arm)
    assert len(result['checkpoints']) == 3 and len(result['steps']) == 3
    assert result['frozen_parameters_unchanged'] and not result['promotion_allowed']
    assert initial != tensor_digest(engine.model.named_parameters())
    assert all(s['response_tokens'] > 0 for s in result['steps'])
    if arm.ema_rate == 0:
        assert tensor_digest(engine.bank.teacher.items()) == tensor_digest(engine.bank.initial.items())
    assert not any(t.requires_grad for t in engine.bank.teacher.values())


def test_no_update_qualification_and_missing_gradients():
    engine = make_engine()
    before = tensor_digest(engine.model.named_parameters())
    with fixed_precision():
        result = engine.qualify(rows(), arms((0.,)))
    assert result['parameters_unchanged'] and result['teacher_initial_identity']
    assert len(result['diagnostics']) == 5
    assert tensor_digest(engine.model.named_parameters()) == before


@pytest.mark.parametrize('source', ['student', 'teacher'])
def test_rollout_source_prefix_and_graph_swap_guard(source, monkeypatch):
    engine = make_engine()
    called, original = [], engine.generate
    def recording(prefix, seed, greedy=False):
        called.append(prefix)
        return original(prefix, seed, greedy)
    monkeypatch.setattr(engine, 'generate', recording)
    row = rows()[0]
    with fixed_precision(), engine.objective(row, Arm('sd', source, 'forward', .1), 0) as (loss, details):
        with pytest.raises(RuntimeError):
            with engine.bank.use(): pass
        loss.backward()
    assert called[0] == engine.ids(row, source == 'teacher')
    assert details['student_prefix_tokens'] == len(engine.ids(row))
    assert details['teacher_prefix_tokens'] > details['student_prefix_tokens']


def test_eos_included_and_length_limit_recorded(monkeypatch):
    engine = make_engine()
    def forced(tokens, eos=True):
        logits = torch.full((1, len(tokens), 64), -100.)
        logits[..., 2 if eos else 3] = 100.
        return logits
    monkeypatch.setattr(engine, 'forward', forced)
    ids, stop = engine.generate([1], 17)
    assert ids == [2] and stop == 'eos'
    monkeypatch.setattr(engine, 'forward', lambda t: forced(t, False))
    ids, stop = engine.generate([1], 17)
    assert ids == [3,3,3] and stop == 'length_limit'


def test_reference_no_truncation_and_no_task_mixing():
    engine = make_engine()
    row = rows()[0]
    row['reference'] = 'too long'
    with pytest.raises(ValueError): engine.reference_ids(row)
    data = rows()
    data[0]['task'] = 'other'
    with pytest.raises(ValueError): engine.check_rows(data)


def test_pairing_is_reproducible():
    with fixed_precision():
        a = make_engine().train(rows(), Arm())
        b = make_engine().train(rows(), Arm())
    assert a['endpoints'] == b['endpoints']
    assert [s['response_ids'] for s in a['steps']] == [s['response_ids'] for s in b['steps']]
    assert [s['update_l2'] for s in a['steps']] == [s['update_l2'] for s in b['steps']]


def test_native_launch_requires_explicit_opt_in(tmp_path):
    with pytest.raises(PermissionError): execute({}, tmp_path/'no-write', 'train')
    assert not (tmp_path/'no-write').exists()


@pytest.mark.parametrize('mutation', ['status','identity','events','teacher'])
def test_stale_qualification_is_rejected(mutation):
    report = {'status':'EXACT_CHECKPOINT_MECHANICS_QUALIFIED','identity':{'version':1},
              'backend_events':['math'],'checks':{'parameters_unchanged':True,'teacher_initial_identity':True}}
    if mutation == 'status': report['status'] = 'CONTROLLED_MODEL_ONLY'
    if mutation == 'identity': report['identity'] = {'version':2}
    if mutation == 'events': report['backend_events'] = []
    if mutation == 'teacher': report['checks']['teacher_initial_identity'] = False
    with pytest.raises(ValueError): verify_qualification(report, {'version':1})


@pytest.mark.parametrize('key, value', [('checkpoint_sha256','TODO'),('steps',2),('steps',100000),
                                      ('device','auto'),('max_new_tokens',0),('trainable_names',[]),
                                      ('learning_rate',float('nan')),('seed',True)])
def test_config_is_bounded_and_pinned(key,value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError): validate_config(cfg)


def test_valid_config_and_manifest_hash(tmp_path):
    validate_config(config())
    (tmp_path/'config.json').write_text('{}')
    (tmp_path/'model.safetensors').write_bytes(b'not_loadable_fixture')
    first = checkpoint_manifest(tmp_path)
    (tmp_path/'model.safetensors').write_bytes(b'changed')
    assert first['sha256'] != checkpoint_manifest(tmp_path)['sha256']


def test_precision_restored_on_exception():
    before = torch.get_float32_matmul_precision()
    with pytest.raises(RuntimeError), fixed_precision():
        raise RuntimeError('deliberate')
    assert before == torch.get_float32_matmul_precision()


def test_cli_plan_and_dataset(tmp_path):
    plan, data = tmp_path/'plan.json', tmp_path/'data.jsonl'
    assert main(['plan','--output',str(plan)]) == 0
    assert json.loads(plan.read_text())['status'] == 'PLAN_ONLY'
    assert main(['synthetic-data','--parents-per-split','2','--output',str(data)]) == 0
    assert len(data.read_text().splitlines()) == 10



def test_native_orchestration_using_explicit_tiny_fixture_only(tmp_path, monkeypatch):
    from experiments.rineya_self_distillation_axes_v1 import native
    cfg = config()
    ident = {"evidence": "FIXTURE_NOT_NATIVE", "version": 1}
    monkeypatch.setattr(native, "preflight", lambda c: (rows(), ident))
    def bind(c):
        e = make_engine()
        return e.model, e.tokenizer
    monkeypatch.setattr(native, "bind_model", bind)
    report = native.execute(cfg, tmp_path / "qualify")
    assert report["training_updates"] == 0 and report["backend_events"]
    trained = native.execute(cfg, tmp_path / "train", "train", Arm(), report, True)
    assert trained["endpoints"]["promotion_allowed"] is False
    assert (tmp_path / "train" / "selected_delta.pt").exists()
    assert len(list((tmp_path / "train" / "steps").glob("*.json"))) == 3
    with pytest.raises(FileExistsError): native.execute(cfg, tmp_path / "qualify")


def test_native_failure_receipt_survives_loader_error(tmp_path, monkeypatch):
    from experiments.rineya_self_distillation_axes_v1 import native
    monkeypatch.setattr(native, "preflight", lambda c: (rows(), {"fixture": True}))
    def fail(c):
        raise RuntimeError("fixture loader failure")
    monkeypatch.setattr(native, "bind_model", fail)
    with pytest.raises(RuntimeError, match="fixture loader failure"):
        native.execute(config(), tmp_path / "failed")
    assert json.loads((tmp_path / "failed" / "FAILED.json").read_text())["status"] == "FAILED"
