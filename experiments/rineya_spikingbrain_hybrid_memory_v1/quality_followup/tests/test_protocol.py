"""Offline engineering fixtures; no pretrained model or GPU work occurs."""
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys

import pytest

from quality_followup import CONDITIONS, RECIPE, REGISTRATION, SCHEMA
from quality_followup import protocol, runner
from quality_followup.utils import Blocked, digest


def dump(path, value):
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def config(tmp_path, monkeypatch):
    # Test-only paths are allowed to live in pytest's external fixture directory.
    monkeypatch.setattr(protocol, 'external_path', lambda value: Path(value).resolve())
    monkeypatch.setattr(runner, 'external_path', lambda value: Path(value).resolve())
    packet = tmp_path / 'corpus'
    packet.mkdir()
    dump(packet / 'manifest.json', {'fixture': True})
    dump(packet / 'exposure_ledger.json', [])
    value = {'schema': SCHEMA, 'study_id': 'update-order-v1', 'recipe': RECIPE,
             'registration': REGISTRATION, 'source': {'fixture': True},
             'environment': {'python': 'fixture', 'platform': 'fixture', 'versions': protocol.NATIVE_VERSIONS,
                             'cuda': '12.4', 'device': 'NVIDIA GeForce RTX 4070 Laptop GPU'},
             'owner_thread_id': 'fixture-owner',
             'corpus_packet': str(packet), 'review': str(tmp_path / 'review.json'),
             'native_run_root': str(tmp_path / 'native_run'),
             'grant': str(tmp_path / 'grant.json'), 'ownership': str(tmp_path / 'owner.json'),
             'native_python': sys.executable}
    monkeypatch.setattr(protocol, 'source_lock', lambda _: ('s' * 64, 't' * 64))
    path = tmp_path / 'study.json'
    dump(path, value)
    return value, path


def granted(config, monkeypatch, condition='control', stage='train'):
    value, path = config
    proposal = runner.build_plan(path)
    binding = protocol.bind(value, condition)
    timestamp = datetime.now(timezone.utc)
    grant = {key: binding[key] for key in protocol.GRANT_BINDINGS}
    grant.update(schema=SCHEMA + '/native-grant', study_id='update-order-v1',
                 proposal_sha256=proposal['proposal_sha256'], approved_by='Fixture Operator',
                 issued_utc=(timestamp - timedelta(minutes=1)).isoformat(),
                 expires_utc=(timestamp + timedelta(hours=1)).isoformat(),
                 native_run_root=str(Path(value['native_run_root']).resolve()),
                 actions=proposal['grant_requirements']['actions'])
    # The grant schema is distinct; stable study schema remains in the binding.
    dump(Path(value['grant']), grant)
    monkeypatch.setattr(protocol, 'require_review', lambda *_: {'fixture': True})
    return value, proposal, binding, grant


def test_plan_exact_counts_and_no_allocation(config):
    value, path = config
    plan = runner.build_plan(path)
    commands = plan['commands']
    assert len(commands) == 11
    assert [x['stage'] for x in commands].count('qualify') == 5
    assert [x['stage'] for x in commands].count('train') == 2
    assert [x['stage'] for x in commands].count('evaluate') == 3
    assert commands[-1]['stage'] == 'compare'
    assert commands[1]['name'] == 'anchor-evaluation'
    assert all(x['stage'] != 'train' or x['condition'] != 'anchor' for x in commands)
    assert not Path(value['native_run_root']).exists()
    assert not Path(value['grant']).exists()
    assert plan['status'] == 'PLAN_ONLY' and not plan['model_loaded']
    assert plan['proposal_sha256'] == digest({k: v for k, v in plan.items() if k != 'proposal_sha256'})


def test_flags_stop_before_config_or_heavy_import(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, 'load_config', lambda _: pytest.fail('configuration should not be read'))
    monkeypatch.setattr(runner, 'native_admission', lambda _: pytest.fail('GPU must not be queried'))
    with pytest.raises(Blocked, match='allow-model-load'):
        runner.execute_stage('missing', 'qualify', 'anchor', tmp_path / 'output')
    with pytest.raises(Blocked, match='both explicit flags'):
        runner.execute_plan('missing')
    assert not (tmp_path / 'output').exists()


def test_missing_grant_stops_native_stage_before_import_or_output(config, monkeypatch):
    value, path = config
    proposal = runner.build_plan(path)
    op = proposal['commands'][0]
    monkeypatch.setattr(runner, 'native_admission', lambda _: pytest.fail('GPU/model import must stay unreachable'))
    with pytest.raises(Blocked, match='grant is missing'):
        runner.execute_stage(path, 'qualify', 'anchor', op['output'], allow_model_load=True)
    assert not Path(value['native_run_root']).exists()


def test_valid_separate_grant(config, monkeypatch):
    value, proposal, binding, _ = granted(config, monkeypatch)
    # stable schema and permission schema have different meanings
    protocol.authorize(value, 'train', binding, proposal, True, True)


@pytest.mark.parametrize('field', ['code_sha256', 'source_sha256', 'config_sha256',
                                  'corpus_sha256', 'ledger_sha256', 'tokenizer_sha256',
                                  'environment', 'proposal_sha256', 'native_run_root'])
def test_changed_or_reused_grant_blocks(config, monkeypatch, field):
    value, proposal, binding, grant = granted(config, monkeypatch)
    grant[field] = 'changed'
    dump(Path(value['grant']), grant)
    with pytest.raises(Blocked):
        protocol.authorize(value, 'train', binding, proposal, True, True)


@pytest.mark.parametrize('change', ['old_schema', 'expired', 'future', 'naive', 'too_long', 'no_operator', 'actions'])
def test_invalid_grants(config, monkeypatch, change):
    value, proposal, binding, grant = granted(config, monkeypatch)
    stamp = datetime.now(timezone.utc)
    if change == 'old_schema':
        grant['schema'] = 'SBHM-001/v1'
    elif change == 'expired':
        grant['expires_utc'] = (stamp - timedelta(seconds=1)).isoformat()
    elif change == 'future':
        grant['issued_utc'] = (stamp + timedelta(seconds=30)).isoformat()
    elif change == 'naive':
        grant['issued_utc'] = stamp.replace(tzinfo=None).isoformat()
    elif change == 'too_long':
        grant['expires_utc'] = (stamp + timedelta(days=8)).isoformat()
    elif change == 'no_operator':
        grant['approved_by'] = ''
    else:
        grant['actions'].pop()
    dump(Path(value['grant']), grant)
    with pytest.raises(Blocked):
        protocol.authorize(value, 'train', binding, proposal, True, True)


def test_wrong_recipe_and_anchor_training(config, monkeypatch):
    value, _, binding, _ = granted(config, monkeypatch)
    bad = {**value, 'recipe': {**RECIPE, 'steps': 513}}
    with pytest.raises(Blocked, match='recipe'):
        protocol.validate_config(bad)
    bad = {**value, 'environment': {**value['environment'],
           'versions': {**protocol.NATIVE_VERSIONS, 'transformers': '4.57.0'}}}
    with pytest.raises(Blocked, match='pins'):
        protocol.validate_config(bad)
    with pytest.raises(Blocked, match='anchor'):
        protocol.authorize(value, 'train', {**binding, 'condition': 'anchor'}, {}, True, True)


def test_fresh_and_exact_qualification(config, tmp_path):
    value, _ = config
    binding = protocol.bind(value, 'control')
    path = tmp_path / 'qualification.json'
    receipt = {'status': 'PASS_NATIVE_MECHANICS_ONLY', 'created_utc': datetime.now(timezone.utc).isoformat(),
               'binding': binding, 'max_prompt_tokens_checked': 2048}
    dump(path, receipt)
    assert protocol.require_qualification(path, binding) == receipt
    with pytest.raises(Blocked, match='mismatch'):
        protocol.require_qualification(path, {**binding, 'condition': 'candidate'})
    receipt['created_utc'] = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    dump(path, receipt)
    with pytest.raises(Blocked, match='expired'):
        protocol.require_qualification(path, binding)


def test_ownership_stale_and_competing(config):
    value, _ = config
    record = {'schema': SCHEMA + '/ownership', 'study_id': 'update-order-v1',
              'owner_thread_id': value['owner_thread_id'],
              'checked_utc': datetime.now(timezone.utc).isoformat()}
    dump(Path(value['ownership']), record)
    assert protocol.require_ownership(value, processes=[])['other_native_worker_pids'] == []
    with pytest.raises(Blocked, match='competing'):
        protocol.require_ownership(value, processes=[{'pid': 9876543, 'name': 'python.exe',
                                    'cmdline': ['python', '-m', 'sbhm.matrix'], 'cwd': ''}])
    record['checked_utc'] = (datetime.now(timezone.utc) - timedelta(minutes=6)).isoformat()
    dump(Path(value['ownership']), record)
    with pytest.raises(Blocked, match='five minutes'):
        protocol.require_ownership(value, processes=[])


def test_review_bound_to_full_inventory(config):
    value, _ = config
    binding = protocol.bind(value, 'anchor')
    audit_path = Path(value['review']).with_name('audit.json')
    dump(audit_path, {'schema': SCHEMA + '/corpus-audit', 'status': 'PASS_OFFLINE_AUDIT',
                     'manifest_sha256': binding['corpus_sha256'], 'full_tokenizer_replay': True,
                     'counts': {'control': 512, 'candidate': 512, 'dev': 672}, 'ledger_steps': 512})
    review = {'schema': SCHEMA + '/corpus-review', 'study_id': 'update-order-v1',
              'reviewed': True, 'status': 'PASS_OFFLINE_AUDIT', 'reviewed_by': ['Independent Fixture'],
              'audit_path': str(audit_path), 'audit_sha256': protocol.sha(audit_path),
              **{k: binding[k] for k in ('corpus_sha256', 'ledger_sha256', 'code_sha256')}}
    dump(Path(value['review']), review)
    protocol.require_review(value, binding)
    review['code_sha256'] = 'changed'
    dump(Path(value['review']), review)
    with pytest.raises(Blocked, match='code_sha256'):
        protocol.require_review(value, binding)
    review['code_sha256'] = binding['code_sha256']
    dump(Path(value['review']), review)
    dump(audit_path, {'status': 'changed'})
    with pytest.raises(Blocked, match='audit bytes'):
        protocol.require_review(value, binding)


def test_delta_condition_and_hash_binding(config, monkeypatch):
    value, _ = config
    binding = protocol.bind(value, 'control')
    directory = Path(value['native_run_root']) / 'control-train'
    directory.mkdir(parents=True)
    (directory / 'delta.safetensors').write_bytes(b'fixture binary, not a model')
    record = {'status': 'TRAINING_COMPLETED_NOT_EFFICACY', 'stage': 'train',
              'binding': binding, 'records': [{} for _ in range(512)]}
    manifest = {'binding': binding, 'sha256': protocol.sha(directory / 'delta.safetensors'), 'record': record}
    dump(directory / 'delta.json', manifest)
    dump(directory / 'receipt.json', record)
    monkeypatch.setattr(protocol, 'require_training_record', lambda *_: None)
    adapted = protocol.bind(value, 'control', directory)
    assert adapted['delta_sha256'] == manifest['sha256']
    with pytest.raises(Blocked, match='condition'):
        protocol.bind(value, 'candidate', directory)
    (directory / 'delta.safetensors').write_bytes(b'changed')
    with pytest.raises(Blocked, match='bytes'):
        protocol.bind(value, 'control', directory)


def test_completed_training_runtime_records_checked_before_delta_load(config, monkeypatch):
    import quality_followup.corpus as corpus
    value, _ = config
    binding = protocol.bind(value, 'control')
    rows, ledger, runtime = [], [], []
    for step in range(512):
        row = {'id': f'fixture-{step}', 'family_id': 'fixture', 'prompt': 'fixture', 'target': '{"value":1}'}
        rows.append(row)
        ledger.append({'step': step, 'slot_id': row['id'], 'control_row_sha256': digest(row),
                       'input_tokens': 4, 'supervised_positions': 2,
                       'input_sha256': {'control': 'i' * 64}, 'labels_sha256': {'control': 'l' * 64}})
        runtime.append({'step': step, 'slot_id': row['id'], 'row_id': row['id'],
                        'input_tokens': 4, 'supervised_positions': 2,
                        'input_token_sha256': 'i' * 64, 'label_token_sha256': 'l' * 64,
                        'loss': 0.1, 'gradient_norm': 0.2})
    monkeypatch.setattr(corpus, 'load_packet', lambda _: {'control': rows, 'ledger': ledger})
    record = {'status': 'TRAINING_COMPLETED_NOT_EFFICACY', 'stage': 'train',
              'binding': binding, 'records': runtime}
    protocol.require_training_record(value, record, binding)
    runtime[20]['label_token_sha256'] = 'changed'
    with pytest.raises(Blocked, match='runtime token/label'):
        protocol.require_training_record(value, record, binding)


def test_exact_operation_no_resume(config):
    value, path = config
    proposal = runner.build_plan(path)
    op = proposal['commands'][0]
    runner._require_exact_operation(proposal, 'qualify', 'anchor', op['output'], None, None)
    with pytest.raises(Blocked, match='exact reviewed'):
        runner._require_exact_operation(proposal, 'qualify', 'anchor', op['output'] + '-other', None, None)
    Path(op['output']).mkdir(parents=True)
    with pytest.raises(Blocked, match='overwrite'):
        runner._require_exact_operation(proposal, 'qualify', 'anchor', op['output'], None, None)


def test_actual_runtime_token_hashes(monkeypatch):
    import quality_followup.corpus as corpus
    row = {'id': 'fixture', 'family_id': 'fixture', 'prompt': 'fixture', 'target': '{"value":7}'}
    ids, labels = [1, 2, 3, 4], [-100, -100, 3, 4]
    monkeypatch.setattr(corpus, 'encode_lists', lambda *_: (ids, labels))
    ledger = {'slot_id': row['id'], 'control_row_sha256': digest(row), 'input_tokens': 4,
              'supervised_positions': 2, 'input_sha256': {'control': digest(ids)},
              'labels_sha256': {'control': digest(labels)}}
    _, _, record = runner.runtime_tokens(None, row, True, ledger, 'control')
    assert record['input_token_sha256'] == digest(ids)
    ledger['labels_sha256']['control'] = 'changed'
    with pytest.raises(Blocked, match='runtime input/label'):
        runner.runtime_tokens(None, row, True, ledger, 'control')


def test_original_cache_formula():
    assert runner._cache_expected(2048) == 448 * 2 ** 20
    assert runner._cache_expected(17) == 28 * 2 * 8 * 128 * 17 * 4


def test_cpu_random_adaptation_save_reload(config, monkeypatch, tmp_path):
    """Exercise the real private loop/loss/delta on tiny random CPU parameters.

    Only this fixture helper substitutes CPU devices/measurement functions. No
    command-line or permission-bound native path accepts these substitutions.
    """
    import torch
    from types import SimpleNamespace
    import quality_followup.corpus as corpus
    from sbhm import native
    from sbhm.core import LoRALinear
    import sbhm.protocol as old_protocol
    value, _ = config
    binding = protocol.bind(value, 'control')
    monkeypatch.setattr(runner, 'RECIPE', {**RECIPE, 'device': 'cpu'})
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    monkeypatch.setattr(old_protocol, 'resources', lambda *_args, **_kwargs: {'rss_bytes': 1})
    monkeypatch.setattr(corpus, 'encode_lists', lambda *_: ([1, 2, 3, 4], [-100, -100, 3, 4]))

    class TinyRandomLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(17)
            self.embedding = torch.nn.Embedding(16, 8)
            self.head = torch.nn.Linear(8, 16, bias=False)
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(15)])
            attention = torch.nn.Module()
            attention.q_proj = torch.nn.Linear(8, 8, bias=False)
            self.model.layers[14].self_attn = attention
            self.requires_grad_(False)
            attention.q_proj = LoRALinear(attention.q_proj, 8)

        def forward(self, input_ids, use_cache=False, logits_to_keep=None):
            hidden = self.model.layers[14].self_attn.q_proj(self.embedding(input_ids))
            if torch.is_tensor(logits_to_keep):
                hidden = hidden.index_select(1, logits_to_keep)
            return SimpleNamespace(logits=self.head(hidden))

    model = TinyRandomLM()
    frozen_before = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    rows, ledger = [], []
    for step in range(512):
        row = {'id': f'fixture-{step}', 'family_id': 'fixture', 'prompt': 'fixture', 'target': '{"value":7}'}
        rows.append(row)
        ledger.append({'step': step, 'slot_id': row['id'], 'control_row_sha256': digest(row),
                       'input_tokens': 4, 'supervised_positions': 2,
                       'input_sha256': {'control': digest([1, 2, 3, 4])},
                       'labels_sha256': {'control': digest([-100, -100, 3, 4])}})
    qualification_path = tmp_path / 'qualification.json'
    qualification = {'status': 'PASS_NATIVE_MECHANICS_ONLY', 'binding': binding,
                     'created_utc': datetime.now(timezone.utc).isoformat()}
    dump(qualification_path, qualification)
    out = tmp_path / 'fixture-training'
    out.mkdir()
    receipt = runner._training(model, None, value, binding, {'control': rows, 'ledger': ledger},
                               out, qualification, qualification_path)
    assert len(receipt['records']) == 512
    assert receipt['records'][-1]['loss'] < receipt['records'][0]['loss']
    assert all(torch.equal(p, frozen_before[name]) for name, p in model.named_parameters() if not p.requires_grad)
    reloaded = TinyRandomLM()
    native.load_delta(reloaded, out, binding)
    probe = torch.tensor([[1, 2, 3, 4]])
    torch.testing.assert_close(model(probe).logits, reloaded(probe).logits, atol=0, rtol=0)
