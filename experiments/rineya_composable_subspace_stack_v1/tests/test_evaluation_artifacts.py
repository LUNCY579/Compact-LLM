from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
import torch
from safetensors.torch import save_file, load_file
from experiments.rineya_composable_subspace_stack_v1 import evaluation as e
from experiments.rineya_composable_subspace_stack_v1.protocol import digest, default_plan, freeze, file_hash, read_json, registered_names
from experiments.rineya_composable_subspace_stack_v1.artifacts import stream_matrices, validate_scope
from experiments.rineya_composable_subspace_stack_v1.__main__ import main


def frozen():
    p = default_plan()
    return p, freeze(p, {name:digest(name) for name in registered_names(p)},
                    cal_sha256=digest('CAL'), check_sha256=digest('CHECK'), audit_sha256=digest('audit'))


def run(name='base', correctness=None, split='CAL'):
    p, f = frozen()
    correct = [False, False, True, True] if correctness is None else correctness
    m = {key:digest(key) for key in e.BOUND}
    m.update(experiment='CSS-001', arm=name, split=split, plan_sha256=digest(p),
             dataset_sha256=f['cal_sha256' if split=='CAL' else 'check_sha256'],
             freeze_sha256=digest(f), artifact_sha256=f['artifacts'][name], evaluation_dtype='float32')
    rows = [{'id':str(i), 'parent_id':'p'+str(i), 'domain':'acquisition' if i<2 else 'retention',
             'role':'capability' if i<2 else 'retention', 'correct':c, 'loss':float(not c)} for i,c in enumerate(correct)]
    return {'schema':1,'metadata':m,'rows':rows}


def test_gain_damage_parent_intervals():
    a = run(); b = run('P1S1C1', [True,False,True,False])
    result = e.compare(a,b)
    assert result['domains']['acquisition']['gains'] == 1
    assert result['domains']['retention']['damages'] == 1
    assert result['domains']['retention']['retained_correct'] == 1
    assert result['release_permitted'] is False


@pytest.mark.parametrize('field', e.BOUND + ('evaluation_dtype', 'split'))
def test_pair_metadata_mismatch(field):
    a, b = run(), run('P1S1C1')
    b['metadata'][field] = digest('changed') if field.endswith('sha256') else 'different'
    with pytest.raises(ValueError):
        e.compare(a,b)


@pytest.mark.parametrize('mutation', ['missing','extra','duplicate','correct_type','parent','domain','role','loss'])
def test_malformed_or_unpaired_rows(mutation):
    a, b = run(), run('P1S1C1')
    if mutation == 'missing': b['rows'].pop()
    if mutation == 'extra':
        row=deepcopy(b['rows'][0]); row['id']='extra'; b['rows'].append(row)
    if mutation == 'duplicate': b['rows'].append(deepcopy(b['rows'][0]))
    if mutation == 'correct_type': b['rows'][0]['correct']=1
    if mutation == 'parent': b['rows'][0]['parent_id']='changed'
    if mutation == 'domain': b['rows'][0]['domain']='changed'
    if mutation == 'role': b['rows'][0]['role']='retention'
    if mutation == 'loss': b['rows'][0]['loss']=float('nan')
    with pytest.raises(ValueError):
        e.compare(a,b)


def test_selection_is_cal_only_and_worst_domain_sensitive():
    _, f=frozen()
    a=run(); safe=run('P1S1C1',[True,True,True,True]); unsafe=run('P0S1C1',[True,True,True,False])
    selected=e.select_on_cal(a,[unsafe,safe],f)
    assert selected['selected']=='P1S1C1'
    assert not selected['candidates'][0]['eligible']
    accepted=e.confirm_check(run('P1S1C1',[True,True,True,True],split='CHECK'),f,selected)
    assert accepted['release_permitted'] is False
    with pytest.raises(ValueError,match='CHECK'):
        e.select_on_cal(run(split='CHECK'),[safe],f)
    with pytest.raises(ValueError,match='differs'):
        e.confirm_check(run('P0S1C1',split='CHECK'),f,selected)


def test_no_false_positive_selection_when_nothing_improves():
    _, f=frozen()
    result=e.select_on_cal(run(),[run('P1S1C1')],f)
    assert result['selected'] is None


def test_artifact_hash_change_refuses_frozen_result():
    _, f=frozen(); r=run(); r['metadata']['artifact_sha256']=digest('other')
    with pytest.raises(ValueError,match='artifact'):
        e.validate_frozen_run(r,f)


def test_functional_loss_interaction_and_missing_loss_refusal():
    a=run(); b=run('P1S0C0'); c=run('P0S1C0'); d=run('P1S1C0')
    for i in range(4):
        a['rows'][i]['loss']=1.; b['rows'][i]['loss']=.8
        c['rows'][i]['loss']=.7; d['rows'][i]['loss']=.6
    result=e.interaction(a,b,c,d)
    assert result['mean']==pytest.approx(.1)
    d['rows'][0]['loss']=None
    with pytest.raises(ValueError,match='losses'):
        e.interaction(a,b,c,d)


def test_factorial_main_effect_and_zero_interactions():
    runs={f'P{p}S{s}C{c}':run(f'P{p}S{s}C{c}',[bool(p)]*4) for p in (0,1) for s in (0,1) for c in (0,1)}
    result=e.factorial_effects(runs)
    assert result['effects']['P']['estimate']==1
    for label in ('S','C','PxS','PxC','SxC','PxSxC'):
        assert result['effects'][label]['estimate']==0


def test_parent_interval_uses_parent_not_row_weighting():
    report=e.parent_interval([1,1,1,-1],['a','a','a','b'],draws=100)
    assert report['estimate']==0
    assert e.parent_interval([1],['a'])['ci95'] is None


def fixture_files(tmp_path, dtype=torch.float32):
    inventory={'layer.weight':[4,4], 'norm.weight':[4]}
    paths=[]
    for i in range(3):
        path=tmp_path/f'source{i}.safetensors'
        save_file({'layer.weight':(torch.eye(4)+i*.1).to(dtype), 'norm.weight':torch.ones(4,dtype=dtype)*(i+1)},str(path))
        paths.append(path)
    return paths,inventory


@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_streamed_artifact_keeps_unselected_base_and_subtracts_once(tmp_path, dtype):
    paths, inv=fixture_files(tmp_path,dtype)
    hashes=[file_hash(p) for p in paths]
    calls=[]
    def transform(name,w,ds):
        calls.append(name)
        return .5*ds[0]+.5*ds[1], {}
    output=tmp_path/'out.safetensors'
    report=stream_matrices(paths,inv,output,['layer.weight'],transform)
    result=load_file(str(output))
    a,b=[load_file(str(p)) for p in paths[1:]]
    assert torch.allclose(result['layer.weight'],.5*a['layer.weight'].float()+.5*b['layer.weight'].float())
    assert torch.equal(result['norm.weight'],torch.ones(4))
    assert all(t.dtype==torch.float32 for t in result.values())
    assert hashes==[file_hash(p) for p in paths] and calls==['layer.weight']
    assert report['weight_sha256']==file_hash(output) and report['tensor_roundtrip_valid']
    with pytest.raises(ValueError,match='exists'):
        stream_matrices(paths,inv,output,['layer.weight'],transform)


def test_stream_rejects_missing_tensor_and_nonfinite_delta(tmp_path):
    paths,inv=fixture_files(tmp_path)
    with pytest.raises(ValueError,match='missing'):
        stream_matrices(paths,inv,tmp_path/'bad1', ['unknown'], lambda *x:None)
    with pytest.raises(ValueError,match='invalid transformed'):
        stream_matrices(paths,inv,tmp_path/'bad2',['layer.weight'],lambda n,w,d:(np.full(w.shape,np.nan),{}))
    assert (tmp_path/'bad2').exists()  # failure artifact is not silently reused


def test_native_scope_budget_and_missing_source_refuse():
    p=default_plan(); key=p['tensors'][0]
    receipt={'roots':{'titan_sparse':'x','opallo':'y'},'inventory':{key:[1024,1024]}}
    validate_scope(p,receipt)
    receipt['inventory'][key]=[99999,99999]
    with pytest.raises(ValueError,match='budget'):
        validate_scope(p,receipt)


def test_cli_plan_is_exclusive_and_refusals_nonzero(tmp_path,capsys):
    plan=tmp_path/'plan.json'
    assert main(['init-plan','--output',str(plan)])==0
    assert main(['validate','--plan',str(plan)])==0
    assert main(['init-plan','--output',str(plan)])==2
    assert 'REFUSED' in capsys.readouterr().err


def test_end_to_end_demo(tmp_path):
    from experiments.rineya_composable_subspace_stack_v1.demo import run_demo
    out=tmp_path/'demo'; report=run_demo(out)
    assert report['status']=='SYNTHETIC_MECHANICS_ONLY'
    assert len(report['arms'])==11
    assert report['synthetic_optimizer_steps']==8
    assert report['native_model_loads']==report['native_optimizer_steps']==0
    assert read_json(out/'report.json')['release_permitted'] is False
    assert not (out/'INCOMPLETE.json').exists()
    with pytest.raises(FileExistsError):
        run_demo(out)


def test_selection_cannot_substitute_soup_for_base_retention_reference():
    _, f = frozen()
    with pytest.raises(ValueError, match='frozen base'):
        e.select_on_cal(run('P0S0C0'), [run('P1S1C1')], f)


def test_frozen_subset_refuses_unregistered_screen_candidate():
    p, _ = frozen()
    f = freeze(p, {'base':digest('base'),'P1S1C1':digest('P1S1C1')},cal_sha256=digest('CAL'),check_sha256=digest('CHECK'),audit_sha256=digest('audit'))
    r = run('P0S1C1'); r['metadata']['freeze_sha256']=digest(f)
    with pytest.raises(ValueError, match='artifact'):
        e.validate_frozen_run(r,f)


def test_stream_rejects_integer_weight_source(tmp_path):
    paths, inv = fixture_files(tmp_path,torch.int64)
    with pytest.raises(ValueError,match='dtype'):
        stream_matrices(paths,inv,tmp_path/'out',['layer.weight'],lambda n,w,d:(np.zeros_like(w),{}))
