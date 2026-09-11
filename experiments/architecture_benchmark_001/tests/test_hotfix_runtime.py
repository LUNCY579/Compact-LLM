"""Synthetic fixtures in temporary directories, never official-model execution evidence."""
from dataclasses import replace
import copy
import json
from pathlib import Path
import pytest
import torch
from safetensors.torch import save_file
from experiments.architecture_benchmark_001 import gates, run, source, engineering, data_tools, ladder, review
from experiments.architecture_benchmark_001.budget import Config
from experiments.architecture_benchmark_001.model import DenseReference


def audit_fixture():
    return {'audit_digest':'SYNTHETIC_TEST_FIXTURE_NOT_OFFICIAL', 'assets':source.source_lock()['assets']}


def make_receipt(folder, stage, ctx, parents=(), model=None):
    folder.mkdir(parents=True,exist_ok=True)
    if model is not None: run._save(model,folder)
    for name in gates.STAGES[stage]['artifacts']:
        if not (folder/name).exists(): (folder/name).write_text('{}')
    links=gates.parent_links(folder,list(parents))
    body={'schema':2,'stage':stage,'status':'PASS','evidence_kind':'official_model_qualification',
          'context':ctx,'checks':dict.fromkeys(gates.STAGES[stage]['checks'],True), 'metrics':{},
          'artifacts':{n:source.sha256_file(folder/n) for n in gates.STAGES[stage]['artifacts']},
          'parent_receipts':links,'parent_receipt_sha256':links[0]['receipt_sha256'] if links else None,
          'licenses_registered_training':False,'licenses_efficacy_claim':False,
          'test_fixture_only':True}
    path=folder/f'{stage}.json'; gates.write_once(path,body); return path


def rewrite(path, edit):
    body=gates.read_hashed_json(path); body.pop('receipt_sha256'); edit(body)
    path.unlink(); gates.write_once(path,body)


def chain(tmp):
    pilot=tmp/'pilot.jsonl'; pilot.write_text('pilot')
    ctx=gates.context(audit_fixture(),pilot,'cpu')
    g0=make_receipt(tmp/'g0','G0',ctx)
    g1=make_receipt(tmp/'g1','G1_IDENTITY',ctx,[g0])
    return ctx,g0,g1,pilot


@pytest.mark.parametrize('damage',['parent_artifact','parent_receipt','missing_parent','missing_check','extra_false','schema1','licenses','wrong_stage'])
def test_strict_recursive_receipt_validation(tmp_path,damage):
    ctx,g0,g1,pilot=chain(tmp_path)
    gates.validate_receipt(g1,'G1_IDENTITY',ctx,data_by_stage=gates.data_bindings(pilot))
    if damage=='parent_artifact': (g0.parent/'predictions.json').write_text('changed')
    elif damage=='parent_receipt': rewrite(g0,lambda b:b.update(metrics={'changed':True}))
    elif damage=='missing_parent': g0.unlink()
    elif damage=='missing_check': rewrite(g1,lambda b:b['checks'].pop('identity_logits_exact'))
    elif damage=='extra_false': rewrite(g1,lambda b:b['checks'].update(additional=False))
    elif damage=='schema1': rewrite(g1,lambda b:b.update(schema=1))
    elif damage=='licenses': rewrite(g1,lambda b:b.update(licenses_registered_training=True))
    else: rewrite(g1,lambda b:b['parent_receipts'][0].update(stage='G1_WIDTH'))
    with pytest.raises(gates.GateClosed): gates.validate_receipt(g1,'G1_IDENTITY',ctx)


def test_ancestor_data_and_global_context_cannot_drift(tmp_path):
    ctx,g0,g1,pilot=chain(tmp_path)
    with pytest.raises(gates.GateClosed):
        gates.validate_receipt(g1,'G1_IDENTITY',ctx,data_by_stage={'G0':'new_pilot_hash'})
    with pytest.raises(gates.GateClosed): gates.validate_receipt(g1,'G1_IDENTITY',{**ctx,'device':'cuda'})
    with pytest.raises(gates.GateClosed): gates.validate_receipt(g1,'G1_IDENTITY',{**ctx,'implementation_sha256':'old'})


def test_manifest_posix_paths_and_required_runtime_inputs():
    manifest=gates.implementation_manifest()
    assert all('\\' not in p for p in manifest)
    for p in ('requirements.txt','requirements-native.txt','protocol.json','model.py','engineering.py'):
        assert p in manifest
    assert gates.implementation_digest()==source.canonical_hash(manifest)


def test_invalid_json_never_creates_receipt_and_existing_is_immutable(tmp_path):
    p=tmp_path/'bad.json'
    with pytest.raises(ValueError): gates.write_once(p,{'value':float('nan')})
    assert not p.exists()
    gates.write_once(p,{'value':1}); original=p.read_bytes()
    with pytest.raises(FileExistsError): gates.write_once(p,{'value':2})
    assert p.read_bytes()==original


def test_artifact_manifest_path_traversal_rejected(tmp_path):
    ctx,g0,g1,pilot=chain(tmp_path)
    rewrite(g0,lambda b:b['artifacts'].update({'../pilot.jsonl':source.sha256_file(pilot)}))
    with pytest.raises(gates.GateClosed): gates.validate_receipt(g0,'G0',ctx)


@pytest.mark.parametrize('length',[0,1,2,7,8,9,15,16,17,101])
def test_windowing_preserves_every_adjacent_target_once(length):
    ids=list(range(length)); windows=data_tools.token_windows(ids,8)
    assert all(2<=len(w)<=8 for w in windows)
    observed=[pair for window in windows for pair in zip(window,window[1:])]
    assert observed==list(zip(ids,ids[1:]))


class Tokenizer:
    eos_token_id=36
    def __len__(self): return 37
    def __call__(self,text,add_special_tokens=False,return_tensors=None):
        ids=[1+ord(c)%34 for c in text]
        return {'input_ids':torch.tensor([ids]) if return_tensors else ids}
    def decode(self,ids,skip_special_tokens=False): return '|'.join(str(int(i)) for i in ids)


def test_prepared_token_ids_are_authoritative_and_locked():
    row={'input_ids':[1,2,3],'text':'not re-tokenized','tokenization':'official_pretokenized_window_v1',
         'tokenizer_sha256':source.source_lock()['assets']['tokenizer.json']}
    assert run._text(Tokenizer(),row,'cpu')['input_ids'].tolist()==[[1,2,3]]
    for update in ({'tokenizer_sha256':'wrong'},{'input_ids':[True,2]},{'input_ids':[1,37]},{'input_ids':[]}):
        with pytest.raises(ValueError): run._text(Tokenizer(),{**row,**update},'cpu')


def test_group_first_data_preparation_and_provenance(monkeypatch,tmp_path):
    monkeypatch.setattr(data_tools,'audit_source',lambda _:audit_fixture())
    monkeypatch.setattr(run,'_load_tokenizer',lambda _:Tokenizer())
    corpus=tmp_path/'raw.jsonl'
    rows=[{'id':f'doc-{i}','group_id':f'family-{i//2}','text':f'{i}: document with unique text {i*i}',
           'source':'synthetic_fixture','source_revision':'immutable-test-v1','license':'test-only','review_status':'approved',
           'source_url':f'https://test.invalid/doc/{i}','source_metadata':{'fixture':True}}
          for i in range(200)]
    corpus.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    result=data_tools.prepare_data(tmp_path/'source',corpus,tmp_path/'data',max_tokens=32)
    assert result['all_splits_nonempty'] and result['pending_review_documents']==0
    seen={}
    for name in ('train','calibration','dev'):
        data=run.load_rows(tmp_path/'data'/f'{name}.jsonl',name)
        for row in data:
            assert row['split']==name and row['source_revision']=='immutable-test-v1'
            assert seen.setdefault(row['group_id'],name)==name
            assert row['tokenizer_sha256']==source.source_lock()['assets']['tokenizer.json']
            assert row['source_url'].startswith('https://test.invalid/') and row['source_metadata']=={'fixture':True}
    assert not result['licenses_registered_training']


@pytest.mark.parametrize('damage',['eval','missing_license','duplicate_cross_group'])
def test_data_prep_refuses_provenance_or_split_laundering(monkeypatch,tmp_path,damage):
    monkeypatch.setattr(data_tools,'audit_source',lambda _:audit_fixture())
    monkeypatch.setattr(run,'_load_tokenizer',lambda _:Tokenizer())
    row={'id':'a','group_id':'family','text':'unique complete document','source':'fixture','source_revision':'v1','license':'fixture'}
    rows=[row]
    if damage=='eval': row['split']='test'
    elif damage=='missing_license': del row['license']
    else: rows.append({**row,'id':'b','group_id':'another'})
    corpus=tmp_path/'raw.jsonl'; corpus.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError): data_tools.prepare_data(tmp_path/'source',corpus,tmp_path/'out')


def test_recovery_remains_closed_even_complete_declarations(tmp_path):
    plan={'training_seeds':[17,29,43],'arms':['B28','D40','C28','C40'],
          'source_weights_sha256':source.source_lock()['assets']['model.safetensors']}
    for k in ('recovery_tokens_per_arm','search_trials_per_arm','max_device_memory_bytes'):plan[k]=100
    for k in ('minimum_effect','noninferiority_margin','regression_band','max_decode_latency_ratio'):plan[k]=.1
    for k in ('recovery_data_manifest_sha256','dev_data_manifest_sha256','regression_data_manifest_sha256','sealed_test_manifest_sha256'):plan[k]='a'*64
    for k in ('optimizer','schedule','primary_endpoint','reviewed_data_provenance','recovered_ordinary_donor_plan','matched_continuation_plan','cost_accounting_plan'):plan[k]='declaration only'
    p=tmp_path/'plan.json';p.write_text(json.dumps(plan))
    result=data_tools.recovery_readiness(p)
    assert result['configuration_complete'] and result['status']=='BLOCKED' and not result['licenses_registered_training']
    for stage in ('G3','G4','G5'):
        assert run.main(['registered',stage])==2


def tiny():
    return Config(vocab_size=37,hidden_size=24,intermediate_size=48,num_attention_heads=6,
                  num_key_value_heads=3,head_dim=8,encoder_layers=4,decoder_layers=0,architecture='plain')


def training_fixture(monkeypatch,tmp):
    c=tiny();torch.manual_seed(17); teacher=DenseReference(c).eval()
    source_dir=tmp/'source';source_dir.mkdir()
    save_file({'model.embed_tokens.weight':teacher.embed_tokens.weight.detach().contiguous()},str(source_dir/'model.safetensors'))
    monkeypatch.setattr(engineering,'audit_source',lambda _:audit_fixture())
    monkeypatch.setattr(run,'_load',lambda *a,**k:(copy.deepcopy(teacher),Tokenizer(),None))
    monkeypatch.setattr(run,'_load_tokenizer',lambda _:Tokenizer())
    monkeypatch.setattr(engineering,'presets',lambda:{gates.protocol()['primary_arms']['B28']:c})
    files={}
    for split,text in [('pilot','small pilot'),('calibration','distinct calibration'),('train','recovery example')]:
        row={'id':split,'group_id':split+'family','split':split,'text':text}
        if split=='pilot': row.update(prompt=text,answer='4')
        files[split]=tmp/f'{split}.jsonl';files[split].write_text(json.dumps(row)+'\n')
    ctx=gates.context(audit_fixture(),files['pilot'],'cpu')
    g0=make_receipt(tmp/'g0','G0',ctx)
    return c,source_dir,files,g0


def test_teacher_cache_and_25_step_b28_engineering_end_to_end(monkeypatch,tmp_path):
    c,s,files,g0=training_fixture(monkeypatch,tmp_path)
    target=engineering.prepare_teacher(s,files['pilot'],files['calibration'],files['train'],g0,tmp_path/'cache','cpu')
    cache_path=tmp_path/'cache/TEACHER_CACHE.json'
    ctx=gates.context(audit_fixture(),files['train'],'cpu');bindings=gates.data_bindings(files['pilot'],files['calibration'],files['train'])
    engineering.validate_teacher_cache(cache_path,ctx,bindings)
    result=engineering.smoke_cached(s,files['pilot'],files['calibration'],files['train'],g0,cache_path,None,tmp_path/'smoke','cpu','B28')
    assert result['status']=='PASS' and len(result['metrics']['losses'])==25
    assert result['metrics']['teacher_backbone_tokens_in_student_loop']==0
    assert result['metrics']['head_projection_work_including_recomputation']['teacher_head_positions']==2*result['metrics']['supervised_shifted_targets']
    assert not result['licenses_registered_training']
    assert (tmp_path/'smoke/model.safetensors').is_file()
    assert result['metrics']['initial_state_sha256']!=result['metrics']['final_state_sha256']
    row=run.load_rows(files['train'],'train')[0];ids=run._text(Tokenizer(),row,'cpu')['input_ids']
    with pytest.raises(ValueError,match='row/order'):
        engineering.load_teacher_row(cache_path,target,0,{**row,'group_id':'wrong'},ids)
    with pytest.raises(ValueError,match='aligned'):
        engineering.load_teacher_row(cache_path,target,0,row,(ids+1)%37)
    shard=cache_path.parent/target['rows'][0]['path'];shard.write_bytes(b'corrupt')
    with pytest.raises(ValueError):engineering.validate_teacher_cache(cache_path,ctx,bindings)


def test_missing_checkpoint_diagnostic_and_no_clean_empty_audit(tmp_path):
    result=review.audit_evidence(tmp_path)
    assert result['status']=='NO_RECEIPTS_FOUND' and not result['licenses_registered_training']
    assert run.main(['diagnose-numerics','--source',str(tmp_path/'missing'),'--out',str(tmp_path/'out'),'--arm','C40'])==2
    assert not (tmp_path/'out').exists()


def test_ladder_zero_exit_without_evidence_is_failure(monkeypatch,tmp_path):
    monkeypatch.setattr(ladder,'audit_source',lambda _:audit_fixture())
    pilot=tmp_path/'p';pilot.write_text('pilot');cal=tmp_path/'c';cal.write_text('cal')
    result=ladder.execute(tmp_path/'source',pilot,cal,None,tmp_path/'ladder','cpu',_launcher=lambda a,l:0)
    assert result['status']=='FAIL' and len(result['results'])==5
    assert result['results'][0]['status']=='FAIL'
    assert all(r['status']=='BLOCKED' for r in result['results'][1:])
    assert not (tmp_path/'ladder/ACTIVE.lock').exists()


def test_ladder_sibling_continuation_and_validated_resume(monkeypatch,tmp_path):
    monkeypatch.setattr(ladder,'audit_source',lambda _:audit_fixture())
    pilot=tmp_path/'p';pilot.write_text('pilot');cal=tmp_path/'c';cal.write_text('cal')
    calls=[];fail_c28=[True]
    def launch(argv,log):
        command=argv[3]; args=argv[4:];output=Path(args[args.index('--out')+1]);stage=output.parent.name
        calls.append(stage);log.write_text('SYNTHETIC TEST ONLY')
        if stage=='G1_CED_C28' and fail_c28[0]:
            output.mkdir();return 2
        parent=[]
        for flag in ('--g0','--g1'):
            if flag in args:parent.append(Path(args[args.index(flag)+1]))
        ctx=gates.context(audit_fixture(),cal if stage=='G1_WIDTH' else pilot,'cpu')
        make_receipt(output,stage,ctx,parent)
        return 0
    out=tmp_path/'ladder'
    first=ladder.execute(tmp_path/'source',pilot,cal,None,out,'cpu',_launcher=launch)
    assert first['status']=='FAIL' and first['results'][-1]['stage']=='G1_CED_C40' and first['results'][-1]['status']=='PASS'
    fail_c28[0]=False;calls.clear()
    second=ladder.execute(tmp_path/'source',pilot,cal,None,out,'cpu',resume=True,_launcher=launch)
    assert second['status']=='PASS' and calls==['G1_CED_C28']
    assert (out/'G1_CED_C28/attempt-0001').is_dir() and (out/'G1_CED_C28/attempt-0002').is_dir()
    cal.write_text('changed')
    with pytest.raises(ValueError,match='session'):ladder.execute(tmp_path/'source',pilot,cal,None,out,'cpu',resume=True,_launcher=launch)


def test_ladder_optimizer_is_explicit(tmp_path):
    with pytest.raises(ValueError,match='include-smoke'):
        ladder.execute(tmp_path/'s',tmp_path/'p',tmp_path/'c',tmp_path/'t',tmp_path/'o','cpu')
    with pytest.raises(ValueError,match='needs --train'):
        ladder.execute(tmp_path/'s',tmp_path/'p',tmp_path/'c',None,tmp_path/'o','cpu',include_smoke=True)


@pytest.mark.parametrize('arm',['C28','C40'])
def test_ced_cached_smoke_requires_and_uses_actual_matched_control(monkeypatch,tmp_path,arm):
    from experiments.architecture_benchmark_001.transfer import identity_expand,to_ced
    c,s,files,g0=training_fixture(monkeypatch,tmp_path)
    base=run._load(s,'cpu')[0]
    ordinary=base if arm=='C28' else identity_expand(base,6)[0]
    total=ordinary.config.encoder_layers
    cc=replace(ordinary.config,architecture='ced',encoder_layers=total//2,decoder_layers=total//2)
    ced,_=to_ced(ordinary,cc,donor_layer=total//2)
    control='B28' if arm=='C28' else 'D40'
    mapping={gates.protocol()['primary_arms']['B28']:c,
             gates.protocol()['primary_arms'][control]:ordinary.config,
             gates.protocol()['primary_arms'][arm]:cc}
    monkeypatch.setattr(engineering,'presets',lambda:mapping)
    ctx=gates.context(audit_fixture(),files['pilot'],'cpu')
    if arm=='C40':
        identity=make_receipt(tmp_path/'identity','G1_IDENTITY',ctx,[g0])
        width=make_receipt(tmp_path/'width','G1_WIDTH',gates.context(audit_fixture(),files['calibration'],'cpu'),[identity],model=ordinary)
        donor_parent=width
    else: donor_parent=g0
    ced_parent=make_receipt(tmp_path/'ced',f'G1_CED_{arm}',ctx,[donor_parent],model=ced)
    engineering.prepare_teacher(s,files['pilot'],files['calibration'],files['train'],g0,tmp_path/'cache','cpu')
    cache=tmp_path/'cache/TEACHER_CACHE.json'
    with pytest.raises(ValueError,match='matched'):
        engineering.smoke_cached(s,files['pilot'],files['calibration'],files['train'],ced_parent,cache,None,tmp_path/'blocked','cpu',arm)
    assert not (tmp_path/'blocked').exists()
    control_out=tmp_path/'control'
    engineering.smoke_cached(s,files['pilot'],files['calibration'],files['train'],donor_parent,cache,None,control_out,'cpu',control)
    result=engineering.smoke_cached(s,files['pilot'],files['calibration'],files['train'],ced_parent,cache,control_out/f'G2_CACHED_{control}.json',tmp_path/'trained','cpu',arm)
    assert result['status']=='PASS' and len(result['parent_receipts'])==2
    assert result['checks']['matched_control_verified'] and not result['licenses_registered_training']
    assert result['metrics']['recipe_sha256']==gates.read_hashed_json(control_out/f'G2_CACHED_{control}.json')['metrics']['recipe_sha256']
    bad_control=control_out/f'G2_CACHED_{control}.json'
    rewrite(bad_control,lambda b:b['metrics'].update(recipe_sha256='wrong'))
    with pytest.raises(ValueError,match='differ'):
        engineering.smoke_cached(s,files['pilot'],files['calibration'],files['train'],ced_parent,cache,bad_control,tmp_path/'wrong-control','cpu',arm)


def test_native_loader_tests_alone_do_not_license_native_random_parity(tmp_path):
    from experiments.architecture_benchmark_001.receipt import summarize
    p=tmp_path/'junit.xml'
    p.write_text('<testsuites><testsuite><testcase classname="test_native_loader" name="a"/><testcase classname="test_native_loader" name="b"/></testsuite></testsuites>')
    assert summarize(p)['native_random_model_cases_passed']==0
    with pytest.raises(ValueError): summarize(p,require_native=True)


def _hashed_receipt(path: Path, body: dict) -> None:
    payload={k:v for k,v in body.items() if k!='receipt_sha256'}
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps({**payload,'receipt_sha256':source.canonical_hash(payload)},indent=2),encoding='utf-8')


def test_audit_evidence_surfaces_strict_failures_in_both_receipt_generations(tmp_path):
    """Regression: reading only metrics.cache_order_parity reported zero strict failures
    for every schema-2 CED FAIL receipt, hiding the experiment's most important result."""
    _hashed_receipt(tmp_path/'legacy/G1_CED_C28.json',{
        'schema':1,'stage':'G1_CED_C28','status':'PASS','artifacts':{},'checks':{},
        'metrics':{'cache_order_parity':{'tail_only':{'elementwise_g0_violations':19},
                                         'chunked_cache':{'elementwise_g0_violations':4114}}}})
    numerics=tmp_path/'current/numerics.json'
    numerics.parent.mkdir(parents=True,exist_ok=True)
    numerics.write_text(json.dumps({'rows':[{'paths':{
        'tail_only':{'elementwise_violations':8},'prefix':{'elementwise_violations':1},
        'chunks':{'size_1':{'elementwise_violations':39267},'size_16':{'elementwise_violations':0}}}},
        {'paths':{'tail_only':{'elementwise_violations':2},'prefix':{'elementwise_violations':0},
                  'chunks':{'size_1':{'elementwise_violations':10},'size_16':{'elementwise_violations':0}}}}]}),
        encoding='utf-8')
    current={'schema':2,'stage':'G1_CED_C28','status':'FAIL',
             'artifacts':{'numerics.json':source.sha256_file(numerics)},
             'checks':{'strict_tail_only_parity':False,'strict_chunked_cache_parity':False,
                       'causal_prefix_parity':False,'decode_choices_exact':True},'metrics':{}}
    _hashed_receipt(tmp_path/'current/G1_CED_C28.json',current)
    _hashed_receipt(tmp_path/'clean/G0.json',{'schema':2,'stage':'G0','status':'PASS','artifacts':{},
        'checks':{'parser_usable':True},'metrics':{}})

    by_path={r['path']:r for r in review.audit_evidence(tmp_path)['records']}
    legacy=by_path['legacy/G1_CED_C28.json']
    assert legacy['recorded_strict_parity_failures']=={'tail_only':19,'chunked_cache':4114}
    assert legacy['any_recorded_strict_parity_failure'] and legacy['declared_status']=='PASS'

    now=by_path['current/G1_CED_C28.json']
    assert now['recorded_strict_parity_failures']=={}
    assert now['failed_strict_parity_checks']==['causal_prefix_parity','strict_chunked_cache_parity','strict_tail_only_parity']
    assert now['strict_parity_violation_totals']=={'tail_only':10,'prefix':1,'chunk_size_1':39277}
    assert now['any_recorded_strict_parity_failure']

    clean=by_path['clean/G0.json']
    assert not clean['any_recorded_strict_parity_failure'] and clean['failed_strict_parity_checks']==[]

    # An unbound or edited sidecar must never contribute counts to the audit.
    numerics.write_text(json.dumps({'rows':[{'paths':{'tail_only':{'elementwise_violations':999999}}}]}),encoding='utf-8')
    tampered={r['path']:r for r in review.audit_evidence(tmp_path)['records']}['current/G1_CED_C28.json']
    assert tampered['strict_parity_violation_totals']=={}
    assert tampered['failed_strict_parity_checks'] and tampered['any_recorded_strict_parity_failure']
