"""Synthetic mechanics only. Temporary fixtures never establish official-model evidence."""
from dataclasses import asdict, replace
import copy
import json
import pytest
import torch
from experiments.architecture_benchmark_001.budget import Config, budget, presets
from experiments.architecture_benchmark_001.model import Cache, DenseReference
from experiments.architecture_benchmark_001.transfer import (
    ActivationCalibration, expansion_map, identity_expand, import_qwen_state, native_key_map,
    narrow, select_width, state_digest, to_ced)
from experiments.architecture_benchmark_001.losses import recovery_loss
from experiments.architecture_benchmark_001.measure import paired_group_interval, profile_reference
from experiments.architecture_benchmark_001 import source, gates, run


def tiny():
    return Config(vocab_size=37, hidden_size=24, intermediate_size=48,
                  num_attention_heads=6, num_key_value_heads=3, head_dim=8,
                  encoder_layers=4, decoder_layers=0, architecture='plain')


def make():
    torch.manual_seed(17)
    return DenseReference(tiny()).eval()


def batch():
    return {'input_ids': torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])}


@pytest.mark.parametrize('kw', [{'hidden_size': 2.5}, {'hidden_size': True}, {'head_dim': -2},
    {'decoder_layers': -1}, {'decoder_layers': 2.5}, {'rope_theta': float('nan')},
    {'rms_norm_eps': float('inf')}, {'max_position_embeddings': 0}])
def test_config_rejects_ambiguous_shapes(kw):
    with pytest.raises(ValueError): Config(**kw)


def test_registered_factorial_is_budget_matched_but_diagnostics_are_not():
    p = gates.protocol(); ps = presets(); base = budget(ps[p['primary_arms']['B28']])['parameters']
    for name in p['primary_arms'].values():
        assert abs(budget(ps[name])['parameters']/base-1) <= p['parameter_relative_tolerance']
    assert budget(ps['D40_1024_expansion_diagnostic'])['parameters']/base > 1.3
    assert budget(ps['W28_896_width_only_diagnostic'])['parameters'] < base
    assert (budget(ps['C40_896_near_baseline_budget'])['kv_bytes_per_token_batch1'] /
            budget(ps['B28_1024_official_shape'])['kv_bytes_per_token_batch1']) == 0.65625


def test_qwen_inventory_roundtrip_and_untied_head_rejected():
    model = make(); mapping = native_key_map(model.config)
    native = {src: model.state_dict()[dst].clone() for src, dst in mapping.items()}
    imported = import_qwen_state(native, model.config)
    assert state_digest(model) == state_digest(imported)
    native['lm_head.weight'] = native['model.embed_tokens.weight'].clone()
    import_qwen_state(native, model.config)
    native['lm_head.weight'][0, 0] += 1
    with pytest.raises(ValueError): import_qwen_state(native, model.config)


@pytest.mark.parametrize('damage', ['missing', 'extra', 'shape', 'nonfinite'])
def test_qwen_import_fails_closed(damage):
    m = make(); native = {s: m.state_dict()[d].clone() for s, d in native_key_map(m.config).items()}
    if damage == 'missing': native.pop('model.norm.weight')
    elif damage == 'extra': native['abstain.bias'] = torch.zeros(1)
    elif damage == 'shape': native['model.norm.weight'] = torch.ones(23)
    else: native['model.norm.weight'][0] = float('nan')
    with pytest.raises(ValueError): import_qwen_state(native, m.config)


def test_identity_expansion_preserves_every_original_and_is_not_aliasing():
    m = make(); before = state_digest(m)
    deeper, line = identity_expand(m, 6)
    assert len(line['layers']) == 6
    assert [r['source_layer'] for r in line['layers'] if not r['identity_inserted']] == [0,1,2,3]
    with torch.inference_mode():
        assert torch.equal(m(**batch()).logits, deeper(**batch()).logits)
    with torch.no_grad(): deeper.encoder[0].q_proj.weight.add_(0.1)
    assert state_digest(m) == before
    official = expansion_map(28, 40)
    assert sum(r['identity_inserted'] for r in official) == 12


@pytest.mark.parametrize('depths', [(0, 1), (4, 3), (4, 9)])
def test_invalid_expansion(depths):
    with pytest.raises(ValueError): expansion_map(*depths)


def test_calibrated_slicing_complete_groups_and_recovery_gradient():
    m = make(); before = state_digest(m)
    cal = ActivationCalibration(m).run([batch()], split='calibration')
    c = replace(m.config, hidden_size=16, intermediate_size=32, num_attention_heads=4, num_key_value_heads=2)
    selection = select_width(m.config, c, cal)
    n, line = narrow(m, c, selection)
    assert state_digest(m) == before
    assert set(line['accounted_target_tensors']) == set(n.state_dict())
    S = torch.tensor(selection['residual']); F = torch.tensor(selection['layers'][0]['ffn'])
    assert torch.equal(n.encoder[0].mlp.gate_proj.weight,
                       m.encoder[0].mlp.gate_proj.weight.index_select(0,F).index_select(1,S))
    n.train(); loss = n(**batch()).logits.square().mean(); loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in n.parameters())
    assert sum(p.numel() for p in n.parameters()) == budget(c)['parameters']


@pytest.mark.parametrize('damage', ['duplicate', 'partial_group', 'bad_ffn', 'wrong_width'])
def test_invalid_slicing_never_silently_initializes(damage):
    m = make(); c = replace(m.config, hidden_size=16, intermediate_size=32, num_attention_heads=4, num_key_value_heads=2)
    sel = select_width(m.config, c, ActivationCalibration(m).run([batch()], split='calibration'))
    if damage == 'duplicate': sel['residual'][1] = sel['residual'][0]
    elif damage == 'partial_group': sel['layers'][0]['query_heads'] = [0,1,2,5]
    elif damage == 'bad_ffn': sel['layers'][0]['ffn'][0] = 999
    else: sel['residual'].pop()
    with pytest.raises(ValueError): narrow(m, c, sel)


@pytest.mark.parametrize('split', ['train', 'dev', 'test', 'pilot', 'sealed'])
def test_calibration_split_guard(split):
    with pytest.raises(ValueError): ActivationCalibration(make()).run([batch()], split=split)


def test_calibration_masks_padding_and_removes_hooks_after_failure():
    m = make()
    a = ActivationCalibration(m).run([{'input_ids': torch.tensor([[1,2,3]])}], split='calibration')
    b = ActivationCalibration(m).run([{'input_ids': torch.tensor([[0,0,1,2,3]]),
         'attention_mask': torch.tensor([[0,0,1,1,1]])}], split='calibration')
    for key in a['scores']:
        torch.testing.assert_close(torch.tensor(a['scores'][key]), torch.tensor(b['scores'][key]), rtol=1e-5, atol=1e-6)
    with pytest.raises(ValueError):
        ActivationCalibration(m).run([{'input_ids': torch.tensor([[1,2]]), 'attention_mask': torch.zeros(1,2)}], split='calibration')
    assert not any(layer._forward_hooks or layer._forward_pre_hooks for layer in m.modules())


def test_ced_donor_ffn_growth_and_shared_memory_gradients():
    m = make(); before = state_digest(m)
    c = replace(m.config, architecture='ced', encoder_layers=2, decoder_layers=2, intermediate_size=56)
    n, lineage = to_ced(m, c, donor_layer=2)
    assert lineage['donor_layer'] == 2 and not lineage['function_preserving']
    assert torch.equal(n.memory.k_proj.weight, m.encoder[2].k_proj.weight)
    assert not n.encoder[0].mlp.down_proj.weight[:,48:].count_nonzero()
    assert state_digest(m) == before
    assert set(lineage['accounted_target_tensors']) == set(n.state_dict())
    n.train(); n(**batch()).logits.square().mean().backward()
    assert n.memory.k_proj.weight.grad.abs().sum() > 0
    assert n.memory.v_proj.weight.grad.abs().sum() > 0
    assert n.encoder[0].mlp.down_proj.weight.grad[:,48:].abs().sum() > 0


@pytest.mark.parametrize('donor', [-1,0,4,99])
def test_invalid_memory_donor(donor):
    c = replace(tiny(), architecture='ced', encoder_layers=2, decoder_layers=2)
    with pytest.raises(ValueError): to_ced(make(), c, donor_layer=donor)


def test_cache_stale_after_weight_reload_and_metadata_owned():
    m = make(); ids = batch()['input_ids']; pos = torch.arange(5).repeat(2,1)
    with torch.inference_mode():
        out = m(ids, position_ids=pos, use_cache=True)
        pos.zero_()
        assert out.cache.positions[0,-1] == 4
        m.load_state_dict(m.state_dict())
        with pytest.raises(ValueError): m(ids[:,:1], past=out.cache, use_cache=True)


@pytest.mark.parametrize('damage', ['shape', 'bank', 'dtype', 'metadata'])
def test_malformed_cache_rejected(damage):
    m=make(); ids=batch()['input_ids']
    with torch.inference_mode():
        cache=m(ids,use_cache=True).cache
        if damage=='shape': cache=replace(cache,encoder=((cache.encoder[0][0][:,:,:1],cache.encoder[0][1]),)+cache.encoder[1:])
        elif damage=='bank': cache=replace(cache,memory=cache.encoder[0])
        elif damage=='dtype': cache=replace(cache,encoder=((cache.encoder[0][0].double(),cache.encoder[0][1]),)+cache.encoder[1:])
        else: cache=replace(cache,valid=cache.valid[:,:1])
        with pytest.raises(ValueError): m(ids[:,:1],past=cache,use_cache=True)


@pytest.mark.parametrize('kw', [{'attention_mask':torch.tensor([[1.,0.5]])},
    {'position_ids':torch.tensor([[-1,0]])},{'position_ids':torch.tensor([[40960,40961]])}])
def test_invalid_positions_and_masks(kw):
    with pytest.raises(ValueError): make()(torch.tensor([[1,2]]),**kw)


def test_shifted_chunked_forward_kl_and_packed_mask_matches_manual():
    torch.manual_seed(19)
    s=torch.randn(1,5,37,requires_grad=True); t=torch.randn(1,5,37,requires_grad=True)
    ids=torch.tensor([[1,2,3,4,5]]); seg=torch.tensor([[0,0,1,1,1]])
    got=recovery_loss(s,t,ids,segments=seg,alpha=0.6,temperature=2,chunk_tokens=1)
    keep=torch.tensor([0,2,3]); ss=s[0,keep]; tt=t[0,keep]; yy=ids[0,keep+1]
    manual=0.4*torch.nn.functional.cross_entropy(ss,yy)+0.6*4*torch.nn.functional.kl_div(
        (ss/2).log_softmax(-1),(tt.detach()/2).softmax(-1),reduction='batchmean')
    torch.testing.assert_close(got,manual)
    got.backward(); assert t.grad is None and s.grad[0,4].abs().sum()==0 and s.grad[0,1].abs().sum()==0


@pytest.mark.parametrize('kw', [{'alpha':-0.1},{'alpha':float('nan')},{'temperature':0},{'chunk_tokens':0}])
def test_loss_invalid_spec(kw):
    with pytest.raises(ValueError): recovery_loss(torch.randn(1,3,37),torch.randn(1,3,37),torch.tensor([[1,2,3]]),**kw)


def test_25_step_tiny_engineering_smoke_and_safe_checkpoint(tmp_path):
    student=make().train(); teacher=copy.deepcopy(student).eval(); ids=batch()['input_ids']
    opt=torch.optim.SGD(student.parameters(),lr=0.001)
    before=state_digest(student)
    for _ in range(25):
        opt.zero_grad()
        with torch.no_grad(): target=teacher(ids).logits
        loss=recovery_loss(student(ids).logits,target,ids); loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters())
        opt.step()
    assert state_digest(student)!=before
    run._save(student,tmp_path); restored=run._restore(tmp_path,'cpu')
    with torch.inference_mode(): assert torch.equal(student(ids).logits,restored(ids).logits)


def test_source_missing_weights_and_registered_stages_are_closed(tmp_path):
    with pytest.raises(ValueError): source.audit_source(tmp_path)
    assert run.main(['audit','--source',str(tmp_path)])==2
    for stage in ['G3','G4','G5']:
        with pytest.raises(gates.GateClosed): gates.require_registered_stage(stage)


def receipt(tmp_path, kind='official_model_qualification'):
    artifact=tmp_path/'predictions.json'; artifact.write_text('{}')
    ctx=gates.context({'audit_digest':'synthetic_validator_fixture', 'assets':source.source_lock()['assets']}, artifact, 'cpu')
    p=tmp_path/'G0.json'
    gates.write_once(p,{'schema':2,'stage':'G0','status':'PASS','evidence_kind':kind,'context':ctx,
                       'checks':dict.fromkeys(gates.STAGES['G0']['checks'],True),
                       'artifacts':{'predictions.json':source.sha256_file(artifact)},
                       'parent_receipts':[], 'parent_receipt_sha256':None,
                       'licenses_registered_training':False, 'licenses_efficacy_claim':False})
    return p,ctx


def test_receipt_validation_artifact_drift_and_no_overwrite(tmp_path):
    p,ctx=receipt(tmp_path); gates.validate_receipt(p,'G0',ctx)
    with pytest.raises(FileExistsError): gates.write_once(p,{})
    with pytest.raises(gates.GateClosed): gates.validate_receipt(p,'G1_IDENTITY',ctx)
    with pytest.raises(gates.GateClosed): gates.validate_receipt(p,'G0',{**ctx,'code':'changed'})
    (tmp_path/'predictions.json').write_text('changed')
    with pytest.raises(gates.GateClosed): gates.validate_receipt(p,'G0',ctx)


@pytest.mark.parametrize('kind',['mechanical','random_model','source_audit_only'])
def test_mechanical_receipt_cannot_open_g0(tmp_path,kind):
    p,ctx=receipt(tmp_path,kind)
    with pytest.raises(gates.GateClosed): gates.validate_receipt(p,'G0',ctx)


def test_data_split_overlap_and_output_guard(tmp_path):
    a=[{'id':'a','group_id':'same','text':'a'}]; b=[{'id':'b','group_id':'same','text':'b'}]
    with pytest.raises(gates.GateClosed): run.ensure_disjoint(a,b)
    b[0]['group_id']='other'; b[0]['text']='a'
    with pytest.raises(gates.GateClosed): run.ensure_disjoint(a,b)
    with pytest.raises(gates.GateClosed): run._directory(tmp_path/'out',tmp_path)
    assert len(run.load_rows(source.ROOT/'data/pilot.jsonl','pilot'))==24


def test_paired_group_statistics_and_actual_tiny_profile():
    a=[{'id':str(i),'group_id':str(i//2),'correct':False} for i in range(8)]
    b=[{**r,'correct':True} for r in a]
    effect=paired_group_interval(a,b,resamples=100)
    assert effect['effect']==effect['lower']==effect['upper']==1.0
    assert effect==paired_group_interval(a,b,resamples=100)
    with pytest.raises(ValueError): paired_group_interval(a,b[:-1])
    p=profile_reference(make(),batch()['input_ids'],new_tokens=2,repeats=2)
    assert p['persistent_kv_tensor_bytes']==budget(tiny(),4)['kv_bytes_per_token_batch1']*7*2
    assert p['prefill_plus_first_argmax']['p50_seconds']>0
    assert p['peak_cuda_allocated_bytes'] is None and not p['licenses_quality_or_deployment_claim']


def test_failed_command_preserves_existing_output(tmp_path):
    from experiments.architecture_benchmark_001.run import main
    output = tmp_path/'existing'; output.mkdir()
    (output/'sentinel').write_text('unchanged')
    assert main(['qualify-source', '--source', str(tmp_path/'missing'), '--out', str(output)]) == 2
    assert sorted(p.name for p in output.iterdir()) == ['sentinel']
    assert (output/'sentinel').read_text() == 'unchanged'

def test_save_rejects_existing_weight_file(tmp_path):
    from experiments.architecture_benchmark_001.run import _save
    (tmp_path/'model.safetensors').write_bytes(b'original')
    with pytest.raises(ValueError, match='overwrite'):
        _save(DenseReference(tiny()), tmp_path)
    assert (tmp_path/'model.safetensors').read_bytes() == b'original'
    assert not (tmp_path/'config.json').exists()


def test_loss_rejects_nonbinary_mask_and_float_segments():
    ids = batch()['input_ids']; logits = torch.randn(2, 5, 37)
    with pytest.raises(ValueError, match='binary'):
        recovery_loss(logits, logits, ids, valid=torch.full_like(ids, 2))
    with pytest.raises(ValueError, match='segment'):
        recovery_loss(logits, logits, ids, segments=ids.float())


def test_mechanical_receipt_never_licenses_models(tmp_path):
    from experiments.architecture_benchmark_001.receipt import summarize
    p = tmp_path/'junit.xml'
    p.write_text('<testsuites><testsuite><testcase classname="test_native" name="test_native_random_qwen3_logits[False]"/><testcase classname="test_native" name="test_native_random_qwen3_logits[True]"/></testsuite></testsuites>')
    r = summarize(p, require_native=True)
    assert r['passed'] == 2 and r['native_random_model_cases_passed'] == 2
    assert r['evidence_kind'] == 'mechanical_tests_only'
    assert r['official_weights_loaded'] is False and r['licenses_training'] is False


@pytest.mark.parametrize('body', ['<testcase><failure/></testcase>', '', '<testcase><skipped message="native absent"/></testcase>'])
def test_mechanical_receipt_rejects_failure_empty_or_skip(tmp_path, body):
    from experiments.architecture_benchmark_001.receipt import summarize
    p = tmp_path/'junit.xml'; p.write_text('<testsuites><testsuite>'+body+'</testsuite></testsuites>')
    with pytest.raises(ValueError): summarize(p, require_native=True)
