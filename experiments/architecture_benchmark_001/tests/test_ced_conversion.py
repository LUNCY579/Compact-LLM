"""Tiny/random CPU fixtures, never official-source receipts or native parity evidence.

Only the official source boundary/runtime and registered sizes are substituted in
pipeline fixtures. Tensor conversion, receipt validation, parent chaining, mechanics,
serialization and fail-closed stage enforcement execute the production code.
"""
import copy
from dataclasses import asdict, replace
import json
from pathlib import Path
import pytest
import torch
from safetensors import SafetensorError
from experiments.architecture_benchmark_001 import ced_conversion as ced, gates, run
from experiments.architecture_benchmark_001.budget import Config, budget, presets
from experiments.architecture_benchmark_001.model import DenseReference
from experiments.architecture_benchmark_001.source import canonical_hash, sha256_file, source_lock
from experiments.architecture_benchmark_001.transfer import identity_expand, narrow, state_digest, to_ced


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def lab(tmp_path, monkeypatch):
    torch.manual_seed(17)
    b = Config(vocab_size=37, hidden_size=16, intermediate_size=32, num_attention_heads=4,
               num_key_value_heads=2, head_dim=4, encoder_layers=4, decoder_layers=0,
               architecture='plain', max_position_embeddings=64)
    base = DenseReference(b).eval()
    w = replace(b, hidden_size=12, intermediate_size=24, num_attention_heads=2, num_key_value_heads=1)
    selection = {'residual': list(range(12)), 'layers': [
        {'kv_groups': [0], 'query_heads': [0, 1], 'ffn': list(range(24))} for _ in range(4)]}
    slim, width_map = narrow(base, w, selection)
    d40, depth_map = identity_expand(slim, 6)
    registry = presets()
    registry.update(B28_1024_official_shape=b, D40_896_near_baseline_budget=d40.config,
        C28_1024_near_baseline_budget=replace(b, architecture='ced', encoder_layers=2,
                                             decoder_layers=2, intermediate_size=36),
        C40_896_near_baseline_budget=replace(d40.config, architecture='ced', encoder_layers=3,
                                             decoder_layers=3, intermediate_size=28))
    monkeypatch.setattr(ced, 'presets', lambda: registry)
    # Native loading itself is NOT exercised by this fixture; see test_native_loader.
    monkeypatch.setattr(run, '_load', lambda *a, **k: (copy.deepcopy(base), None, None))
    rt = {**gates.runtime(), 'transformers': gates.protocol()['native_transformers_version']}
    monkeypatch.setattr(gates, 'runtime', lambda: rt)
    monkeypatch.setattr(ced, 'runtime', lambda: rt)
    audit = {'audit_digest': 'fixture-audit-NOT-OFFICIAL', 'assets': source_lock()['assets'],
             'core_id': 'TINY-UNIT-FIXTURE', 'parameters': budget(b)['parameters']}
    monkeypatch.setattr(ced, 'audit_source', lambda p: copy.deepcopy(audit))
    source = tmp_path/'source'; source.mkdir()
    (source/'model.safetensors').write_bytes(b'FIXTURE ONLY; not official weights')
    pilot = tmp_path/'pilot.jsonl'
    pilot.write_text(json.dumps({'id': 'p', 'group_id': 'p-group', 'split': 'pilot',
                                  'prompt': 'unit-only', 'answer': '0'}) + '\n')
    calibration = tmp_path/'calibration.jsonl'
    calibration.write_text(json.dumps({'id': 'c', 'group_id': 'c-group', 'split': 'calibration',
                                       'text': 'disjoint fixture'}) + '\n')
    folders = {stage: tmp_path/stage for stage in ced.REQUIRED}
    receipts = {}
    previous = None
    for stage in ('G0', 'G1_IDENTITY', 'G1_WIDTH'):
        folder = folders[stage]; folder.mkdir()
        _, required = ced.REQUIRED[stage]
        if stage == 'G1_WIDTH':
            run._save(d40, folder)
            gates.write_once(folder/'lineage.json', {'width': width_map, 'depth': depth_map})
            gates.write_once(folder/'calibration_scores.json', {'fixture_only': True})
        else:
            for name in required:
                gates.write_once(folder/name, {'fixture_only': True})
        ctx = gates.context(audit, calibration if stage == 'G1_WIDTH' else pilot, 'cpu')
        receipts[stage] = run._record(folder, stage, ctx, {k: True for k in ced.REQUIRED[stage][0]},
                                      {'fixture_only': True}, sorted(required), previous)
        previous = receipts[stage]['receipt_sha256']
    def args(arm='C28', name='out'):
        kw = dict(source=source, arm=arm, pilot=pilot, g0=folders['G0']/'G0.json', out=tmp_path/name)
        if arm == 'C40':
            kw.update(identity=folders['G1_IDENTITY']/'G1_IDENTITY.json',
                      width=folders['G1_WIDTH']/'G1_WIDTH.json', calibration=calibration)
        return kw
    return dict(base=base, d40=d40, registry=registry, source=source, audit=audit,
                pilot=pilot, calibration=calibration, folders=folders, receipts=receipts, args=args)


def rewrite_receipt(path, change):
    body = json.loads(path.read_text()); body.pop('receipt_sha256')
    change(body)
    body['receipt_sha256'] = canonical_hash(body)
    path.write_text(json.dumps(body))


@pytest.mark.parametrize('arm,parameters,banks', [('C28', 596311424, 15), ('C40', 594279808, 21)])
def test_registered_geometry_without_real_weights(arm, parameters, banks):
    donor, target = ced._configs(arm)
    assert target.head_dim == donor.head_dim == 128
    with torch.device('meta'):
        model = DenseReference(target)
    assert sum(p.numel() for p in model.parameters()) == budget(target)['parameters'] == parameters
    assert budget(target)['kv_banks'] == banks


@pytest.mark.parametrize('arm', ['C28', 'C40'])
def test_end_to_end_tiny_pipeline(lab, monkeypatch, arm):
    def forbidden(*a, **k):
        raise AssertionError('conversion attempted training')
    monkeypatch.setattr(torch.optim, 'SGD', forbidden)
    monkeypatch.setattr(torch.optim, 'Adam', forbidden)
    monkeypatch.setattr(torch.optim, 'AdamW', forbidden)
    monkeypatch.setattr(torch.Tensor, 'backward', forbidden)
    kw = lab['args'](arm)
    before = state_digest(lab['base'])
    receipt = ced.qualify_ced(**kw)
    assert receipt['stage'] == f'G1_CED_{arm}' and receipt['status'] == 'PASS'
    assert all(v is True for v in receipt['checks'].values())
    for key in ('licenses_CED_G2', 'licenses_registered_training', 'licenses_quality_evaluation', 'licenses_efficacy_claim'):
        assert receipt[key] is False
    assert receipt['metrics']['optimizer_steps'] == 0
    assert receipt['metrics']['function_preserving'] is False
    assert state_digest(lab['base']) == before
    assert gates.validate_receipt(kw['out']/f'G1_CED_{arm}.json', f'G1_CED_{arm}',
                                   gates.context(lab['audit'], lab['pilot'], 'cpu')) == receipt
    model = run._restore(kw['out'], 'cpu')
    lineage = json.loads((kw['out']/'lineage.json').read_text())
    assert set(lineage['tensor_map']) == set(model.state_dict())
    assert lineage['target_state_sha256'] == state_digest(model)
    assert lineage['recovery_required'] is True
    provenance = json.loads((kw['out']/'provenance.json').read_text())
    assert provenance['quality_evaluation_executed'] is False
    assert provenance['C40_recovered_donor_requirement_waived'] is False
    assert len(provenance['validated_parent_receipts']) == (1 if arm == 'C28' else 3)
    if arm == 'C40':
        assert (kw['out']/'ordinary_lineage.json').read_bytes() == (kw['width'].parent/'lineage.json').read_bytes()
    monkeypatch.setattr(run, 'audit_source', lambda p: copy.deepcopy(lab['audit']))
    # Ordinary smoke cannot accept this receipt as G1_WIDTH; registered stages remain absent.
    with pytest.raises(gates.GateClosed):
        run.smoke(kw['source'], kw['pilot'], lab['calibration'], lab['calibration'],
                  kw['out']/f'G1_CED_{arm}.json', kw['out']/'forbidden-smoke', 'cpu')


@pytest.mark.parametrize('arm', ['C28', 'C40'])
def test_ffn_growth_preserves_each_ffn_not_whole_topology(lab, arm):
    donor = lab['base'] if arm == 'C28' else lab['d40']
    _, target = ced._configs(arm)
    converted, primitive = to_ced(donor, target, donor_layer=target.encoder_layers)
    lineage = ced.verify_transfer(donor, converted, primitive)
    assert lineage['ffn_extra_row_indices'] == list(range(4))
    x = torch.randn(2, 3, target.hidden_size)
    with torch.inference_mode():
        for old, new in zip(donor.encoder, list(converted.encoder) + list(converted.decoder)):
            torch.testing.assert_close(old.mlp(x), new.mlp(x), atol=1e-6, rtol=1e-5)
    assert state_digest(donor) != state_digest(converted)


@pytest.mark.parametrize('key', ['memory.norm.weight', 'memory.k_norm.weight', 'memory.k_proj.weight',
    'memory.v_proj.weight', 'encoder.0.q_proj.weight', 'decoder.0.q_proj.weight',
    'decoder.0.mlp.gate_proj.weight', 'decoder.0.mlp.up_proj.weight',
    'decoder.0.mlp.down_proj.weight', 'embed_tokens.weight', 'final_norm.weight'])
def test_changed_target_tensor_is_rejected(lab, key):
    _, target = ced._configs('C28')
    converted, primitive = to_ced(lab['base'], target, donor_layer=target.encoder_layers)
    with torch.no_grad():
        converted.state_dict()[key].view(-1)[-1] += 0.01
    with pytest.raises(gates.GateClosed, match='tensor copy'):
        ced.verify_transfer(lab['base'], converted, primitive)


def test_aliasing_and_undeclared_donor_are_rejected(lab):
    _, target = ced._configs('C28')
    converted, primitive = to_ced(lab['base'], target, donor_layer=target.encoder_layers)
    converted.embed_tokens.weight = lab['base'].embed_tokens.weight
    with pytest.raises(gates.GateClosed, match='alias'):
        ced.verify_transfer(lab['base'], converted, primitive)
    converted, primitive = to_ced(lab['base'], target, donor_layer=target.encoder_layers + 1)
    with pytest.raises(gates.GateClosed, match='donor'):
        ced.verify_transfer(lab['base'], converted, primitive)


@pytest.mark.parametrize('mutation', ['FAIL', 'synthetic', 'old_code', 'runtime', 'parent', 'checks', 'artifacts', 'permission'])
def test_invalid_g0_fails_before_loading(lab, monkeypatch, mutation):
    kw = lab['args']()
    def mutate(b):
        if mutation == 'FAIL': b['status'] = 'FAIL'
        elif mutation == 'synthetic': b['evidence_kind'] = 'mechanical_tests'
        elif mutation == 'old_code': b['context']['implementation_sha256'] = 'historical'
        elif mutation == 'runtime': b['context']['runtime']['python'] = 'changed'
        elif mutation == 'parent': b['parent_receipt_sha256'] = 'unexpected'
        elif mutation == 'checks': b['checks'].pop('native_reference_logits')
        elif mutation == 'artifacts': b['artifacts'].clear()
        else: b['licenses_registered_training'] = True
    rewrite_receipt(kw['g0'], mutate)
    monkeypatch.setattr(run, '_load', lambda *a, **k: pytest.fail('loaded before prerequisites'))
    with pytest.raises(gates.GateClosed):
        ced.qualify_ced(**kw)
    assert not kw['out'].exists()


@pytest.mark.parametrize('stage', ['G1_IDENTITY', 'G1_WIDTH'])
def test_broken_parent_chain_is_rejected(lab, stage):
    kw = lab['args']('C40')
    path = lab['folders'][stage]/f'{stage}.json'
    rewrite_receipt(path, lambda b: b.update(parent_receipt_sha256='different-parent'))
    with pytest.raises(gates.GateClosed, match='chain'):
        ced.qualify_ced(**kw)
    assert not kw['out'].exists()


@pytest.mark.parametrize('artifact', ['model.safetensors', 'config.json', 'lineage.json', 'calibration_scores.json'])
@pytest.mark.parametrize('operation', ['delete', 'mutate'])
def test_width_artifacts_required_before_restore(lab, monkeypatch, artifact, operation):
    kw = lab['args']('C40'); path = kw['width'].parent/artifact
    if operation == 'delete': path.unlink()
    else: path.write_bytes(path.read_bytes() + b'changed')
    monkeypatch.setattr(run, '_restore', lambda *a, **k: pytest.fail('restored before artifact validation'))
    with pytest.raises(gates.GateClosed):
        ced.qualify_ced(**kw)
    assert not kw['out'].exists()


def test_g2_receipt_not_admitted_as_width(lab):
    kw = lab['args']('C40')
    rewrite_receipt(kw['width'], lambda b: b.update(stage='G2'))
    with pytest.raises(gates.GateClosed): ced.qualify_ced(**kw)


@pytest.mark.parametrize('change', ['pilot', 'source', 'save'])
def test_midrun_drift_never_leaves_pass(lab, monkeypatch, change):
    kw = lab['args']()
    real = ced.mechanical_checks
    def drift(model):
        result = real(model)
        if change == 'pilot': lab['pilot'].write_text(lab['pilot'].read_text() + '\n')
        elif change == 'source': monkeypatch.setattr(ced, 'audit_source', lambda p: {**lab['audit'], 'parameters': -1})
        return result
    monkeypatch.setattr(ced, 'mechanical_checks', drift)
    if change == 'save':
        original = run._save
        def corrupt(model, folder):
            original(model, folder)
            (folder/'model.safetensors').write_bytes(b'corrupted')
        monkeypatch.setattr(run, '_save', corrupt)
    with pytest.raises((ValueError, RuntimeError, SafetensorError)):
        ced.qualify_ced(**kw)
    assert (kw['out']/'failure.json').exists()
    receipt = kw['out']/'G1_CED_C28.json'
    assert not receipt.exists() or json.loads(receipt.read_text())['status'] == 'FAIL'


def test_mechanical_failure_is_preserved(lab, monkeypatch):
    kw = lab['args']()
    original = ced.mechanical_checks
    def bad(model):
        checks, metrics = original(model)
        checks['future_token_isolation'] = False
        return checks, metrics
    monkeypatch.setattr(ced, 'mechanical_checks', bad)
    with pytest.raises(gates.GateClosed, match='mechanical checks failed'):
        ced.qualify_ced(**kw)
    assert json.loads((kw['out']/'G1_CED_C28.json').read_text())['status'] == 'FAIL'
    assert (kw['out']/'failure.json').exists()


@pytest.mark.parametrize('protected', ['source', 'G0', 'G1_IDENTITY', 'G1_WIDTH', 'published', 'existing'])
def test_output_protection(lab, protected, tmp_path):
    kw = lab['args']('C40')
    if protected == 'source': kw['out'] = lab['source']/'new'
    elif protected in lab['folders']: kw['out'] = lab['folders'][protected]/'new'
    elif protected == 'published': kw['out'] = tmp_path/'Rineya_Published_Cores'/'new'
    else:
        kw['out'].mkdir(); (kw['out']/'sentinel').write_text('unchanged')
    with pytest.raises((ValueError, OSError)): ced.qualify_ced(**kw)
    assert not (kw['out']/'failure.json').exists()
    if protected == 'existing': assert (kw['out']/'sentinel').read_text() == 'unchanged'


@pytest.mark.parametrize('missing', ['identity', 'width', 'calibration'])
def test_c40_requires_full_chain(lab, missing):
    kw = lab['args']('C40'); kw[missing] = None
    with pytest.raises(gates.GateClosed, match='C40 needs'): ced.qualify_ced(**kw)


def test_c28_rejects_extraneous_donor_and_no_cuda_fallback(lab, monkeypatch):
    kw = lab['args'](); kw['width'] = lab['folders']['G1_WIDTH']/'G1_WIDTH.json'
    with pytest.raises(gates.GateClosed): ced.qualify_ced(**kw)
    kw = lab['args'](); kw['device'] = 'cuda'
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    with pytest.raises(gates.GateClosed, match='no fallback'): ced.qualify_ced(**kw)


def test_wrong_donor_shape_fails_and_records_failure(lab, monkeypatch):
    kw = lab['args']('C40')
    monkeypatch.setattr(run, '_restore', lambda *a: copy.deepcopy(lab['base']))
    with pytest.raises(gates.GateClosed, match='donor differs'): ced.qualify_ced(**kw)
    assert (kw['out']/'failure.json').exists()


@pytest.mark.parametrize('stage', ['G3', 'G4', 'G5'])
def test_registered_stages_unconditionally_closed(stage, monkeypatch):
    monkeypatch.setattr(gates, 'protocol', lambda: {'G3_G5': {'execution_enabled': True}})
    with pytest.raises(gates.GateClosed): gates.require_registered_stage(stage)
    assert run.main(['registered', stage]) == 2


def test_conversion_rejects_protocol_opening(monkeypatch):
    p = gates.protocol(); p['G3_G5']['execution_enabled'] = True
    monkeypatch.setattr(ced, 'protocol', lambda: p)
    with pytest.raises(gates.GateClosed, match='contract drift'): ced._configs('C40')


def test_production_source_missing_is_not_mocked(tmp_path, monkeypatch, capsys):
    rt = {**ced.runtime(), 'transformers': gates.protocol()['native_transformers_version']}
    monkeypatch.setattr(ced, 'runtime', lambda: rt)
    result = ced.main(['--arm', 'C28', '--source', str(tmp_path/'missing'),
                       '--g0', str(tmp_path/'G0.json'), '--out', str(tmp_path/'out')])
    assert result == 2
    assert 'official source directory missing' in capsys.readouterr().out
    assert not (tmp_path/'out').exists()
