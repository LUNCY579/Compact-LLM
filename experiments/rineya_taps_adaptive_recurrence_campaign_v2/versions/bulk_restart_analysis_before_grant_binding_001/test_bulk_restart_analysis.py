"""CPU metadata/journal fixtures; no model, GPU, tokenizer or grant creation."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import ast
import hashlib
import json
from pathlib import Path

import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_restart_analysis as analysis
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.tests.test_bulk_analysis import fixture as old_fixture


def encoded(value):
    return (json.dumps(value, sort_keys=True, allow_nan=False)+'\n').encode()


def fixture():
    args = old_fixture()
    original_rows, original_traces = args[0], args[1]
    physical = [r for r in original_rows if r['physical_run']]
    chunks = []
    for e in original_traces:
        if e['forward_index']==1 and e['iteration']==0: chunks.append([])
        chunks[-1].append(e)
    traced_rows = [r for r in physical if r['arm']!='qualification_legacy']
    assert len(chunks)==len(traced_rows)
    trace_groups = {id(row):chunk for row,chunk in zip(traced_rows,chunks)}
    stages = [
        [r for r in physical if r['split']=='SMOKE' and r['arm']!='A5'],
        [r for r in physical if r['split']=='DEVELOPMENT' and r['arm']!='A5'],
        [r for r in physical if r['split']=='SMOKE' and r['arm']=='A5'],
        [r for r in physical if r['split']=='DEVELOPMENT' and r['arm']=='A5'],
        [r for r in physical if r['split']=='OPEN_VALIDATION' and r['arm']!='A5'],
        [r for r in physical if r['split']=='OPEN_VALIDATION' and r['arm']=='A5'],
    ]
    rows, traces, forwards, callbacks = [], [], [], []
    elapsed = 0.
    def stamp(event, step=1.):
        nonlocal elapsed
        elapsed += step
        return {**event, 'parent_elapsed_seconds':elapsed,
                'timestamp_utc':(datetime(2026,10,5,tzinfo=timezone.utc)+timedelta(seconds=elapsed)).isoformat()}
    resource = {'rss_bytes':analysis.GIB,'host_available_bytes':2*analysis.GIB,
                'gpu_allocated_bytes':3*analysis.GIB,'gpu_free_bytes':2*analysis.GIB}
    def add(group):
        for row in group:
            rows.append(row); traces.extend(trace_groups.get(id(row), []))
            for event in ('FORWARD_STARTED','FORWARD_COMPLETED'):
                forwards.append(stamp({'event':event,**{k:row[k] for k in ('case_id','split','seed','arm','extra_loops')},
                    'full_model_call':1,'prefix_tokens':3,'forward_seconds':.1 if event=='FORWARD_COMPLETED' else None,
                    **({'resources':resource} if event=='FORWARD_COMPLETED' else {})}))
    def maximum(index):
        forwards.append(stamp({'event':'MAXIMUM_FORWARD_STARTED','attempt':index,'prefix_tokens':357},.1))
        failed = index==7
        forwards.append(stamp({'event':'MAXIMUM_FORWARD_FAILED' if failed else 'MAXIMUM_FORWARD_COMPLETED',
            'attempt':index,'prefix_tokens':357,'completed':not failed,
            'partial_compute':'UNKNOWN' if failed else None,'error':'EXPECTED_BULK_FORCED_EXCEPTION' if failed else None},.1))
    assets={f'asset-{i}':'a'*64 for i in range(9)};assets['modeling_supa_a11.py']=analysis.CUSTOM
    prefix='experiments/rineya_taps_adaptive_recurrence_campaign_v2/'
    admission={'status':'READY_FOR_BULK_RESTART_FP32_DISPATCH','lane':analysis.RESTART_LANE,
        'hard_deadline_seconds':16000,'native_finalization_reserve_seconds':1800,'checkpoint_files':assets,
        'dispatch_hashes':{prefix+'bulk_analysis.py':analysis.ORIGINAL_ANALYZER_SHA,prefix+'bulk_restart_native.py':'b'*64}}
    context={'admission':admission,'registration':{'lane':analysis.RESTART_LANE},'proof_bytes':{},'journal_bytes':{},'integrity_journal':callbacks}
    args[5].update(runner_sha256='b'*64,OPEN_VALIDATION_native_exposure_scope='THIS_RUN_ONLY',
        historical_validation_exposure={'prior_run':'native-bulk-001','attempted_or_exposed_cases':59,
            'completed_deterministic_cases':58,'independent_confirmation':False})
    for name in ('before_state_sha256','after_state_sha256','loaded_state_sha256'):args[6][name]=analysis.STATE
    args[6].update(lane=analysis.RESTART_LANE,loaded_custom_code_sha256=analysis.CUSTOM,
        parameter_gradients='NONE',requires_grad_false=True)
    args[7].update(lane=analysis.RESTART_LANE,native_worker_seconds=7000.)
    args[8].update(before_state_sha256=analysis.STATE,after_state_sha256=analysis.STATE)
    def journal_blobs():
        return {'outcomes.jsonl':b''.join(encoded(r) for r in rows),
                'trajectories.jsonl':b''.join(encoded(r) for r in traces),
                'forwards.jsonl':b''.join(encoded(r) for r in forwards),
                'readout_pairs.jsonl':b''.join(encoded({'case_id':r['case_id']}) for r in rows if r['split']!='SMOKE' and r['physical_run'])}
    def checkpoint(index):
        name,phase,science,total,case_count=analysis.INTEGRITY_STAGES[index-1]
        callbacks.append(stamp({'event':'INTEGRITY_CHECK_STARTED','phase':phase,'sequence':2*index-1,'checkpoint_index':index},.1))
        blobs=journal_blobs()
        if index>1:blobs.update({'frozen_development.json':encoded(args[5]),'A5_qualification.json':encoded(args[9])})
        proof={'status':'PASS_MODEL_ASSETS_SOURCE_INTEGRITY','phase':phase,'checkpoint_index':index,
            'start_event_sequence':2*index-1,'terminal_event_sequence':2*index,
            'before_state_sha256':analysis.STATE,'after_state_sha256':analysis.STATE,
            'loaded_custom_code_sha256':analysis.CUSTOM,'checkpoint_files':assets,'source_sha256':admission['dispatch_hashes'],
            'eval_mode':True,'requires_grad_false':True,'parameter_gradients_none':True,'parameter_gradients':'NONE',
            'trainable_parameters':0,'model_forward_calls_added':0,'resources':resource,'elapsed_seconds':.05,
            'counts':{'scientific_physical_rows':science,'case_count':case_count,'recorded_physical_evaluations':total,
                'model_forward_calls_added_by_checkpoint':0},
            'parent_elapsed_seconds':elapsed+.05,'ended_utc':(datetime(2026,10,5,tzinfo=timezone.utc)+timedelta(seconds=elapsed+.05)).isoformat(),
            'journal_commitments':{n:{'bytes':len(raw),'lines':len(raw.splitlines()),'sha256':hashlib.sha256(raw).hexdigest()} for n,raw in blobs.items()}}
        context['proof_bytes'][name]=encoded(proof)
        digest=hashlib.sha256(context['proof_bytes'][name]).hexdigest()
        callbacks.append(stamp({'event':'INTEGRITY_CHECK_PASSED','phase':phase,'sequence':2*index,'checkpoint_index':index,
            'proof_file':name,'proof_sha256':digest},.1))
        args[6][('DEV_deterministic_integrity_sha256','DEV_complete_integrity_sha256','FINAL_integrity_sha256')[index-1]]=digest
        return digest
    add(stages[0])
    for i in range(1,18):maximum(i)
    for cache,object_ in ((True,False),(False,True)):
        forwards.append(stamp({'event':'CACHE_REJECTION_STARTED','use_cache':cache,'cache_object_present':object_},.1))
        forwards.append(stamp({'event':'CACHE_REJECTED','completed':False,'model_compute_started':False,'wrapper_forward_count':0},.1))
    add(stages[1]);first=checkpoint(1)
    args[5].update(DEV_deterministic_integrity_sha256=first,created_parent_elapsed_seconds=elapsed+.02,
        created_timestamp_utc=(datetime(2026,10,5,tzinfo=timezone.utc)+timedelta(seconds=elapsed+.02)).isoformat())
    args[5]['freeze_sha256']=analysis.payload_digest({k:v for k,v in args[5].items() if k!='freeze_sha256'})
    callbacks.append(stamp({'event':'FREEZE_COMMITTED','freeze_file':'frozen_development.json',
        'freeze_sha256':hashlib.sha256(encoded(args[5])).hexdigest(),'freeze_payload_sha256':args[5]['freeze_sha256']},.05))
    add(stages[2]);maximum(18);add(stages[3]);checkpoint(2)
    add(stages[4]);add(stages[5]);rows.extend(r for r in original_rows if not r['physical_run']);checkpoint(3)
    context['journal_bytes']=journal_blobs()|{'frozen_development.json':encoded(args[5]),'A5_qualification.json':encoded(args[9])}
    args[0],args[1],args[10]=rows,traces,forwards
    return args,context


def test_complete_fixture_keeps_science_and_requires_all_three_boundaries():
    args,context=fixture();before=analysis._aggregate_scientific(*args);after=analysis.aggregate(*args,restart=context)
    assert before['complete'] and after['complete'],after['errors']
    for key in ('reports','costs','physical_rows','expected_physical_rows','oracle_nonphysical_rows','practical_margin'):
        assert before[key]==after[key]
    assert after['restart_integrity']['status']=='PASS_RECORDED_RESTART_INTEGRITY'
    assert after['primary_validation_parent_units']==48 and not after['independent_source_confirmation']


@pytest.mark.parametrize('defect',['missing','state','assets','source','gradient','counts','prefix','freeze','callback','clock_afterVAL','final','old_epoch'])
def test_missing_corrupted_or_late_proofs_never_release_primary(defect):
    args,context=fixture();name=analysis.INTEGRITY_STAGES[1][0]
    proof=json.loads(context['proof_bytes'][name])
    if defect=='missing':context['proof_bytes'].pop(name)
    elif defect=='state':proof['after_state_sha256']='a'*64
    elif defect=='assets':proof['checkpoint_files']={}
    elif defect=='source':proof['source_sha256']={}
    elif defect=='gradient':proof['parameter_gradients']='PRESENT'
    elif defect=='counts':proof['counts']['scientific_physical_rows']=2111
    elif defect=='prefix':proof['journal_commitments']['outcomes.jsonl']['bytes']-=1
    elif defect=='freeze':args[5]['DEV_deterministic_integrity_sha256']='a'*64
    elif defect=='callback':context['integrity_journal'][4]['proof_sha256']='a'*64
    elif defect=='clock_afterVAL':
        first_val=next(e['parent_elapsed_seconds'] for e in args[10] if e.get('event')=='FORWARD_STARTED' and e.get('split')=='OPEN_VALIDATION')
        proof['parent_elapsed_seconds']=first_val+.1
        for e in context['integrity_journal']:
            if e.get('checkpoint_index')==2:e['parent_elapsed_seconds']=first_val+.2
    elif defect=='final':context['proof_bytes'].pop('FINAL_INTEGRITY.json')
    else:context['admission']['status']='READY_FOR_BULK_FP32_DISPATCH'
    if defect in ('state','assets','source','gradient','counts','prefix','clock_afterVAL'):
        context['proof_bytes'][name]=encoded(proof);digest=hashlib.sha256(context['proof_bytes'][name]).hexdigest()
        args[6]['DEV_complete_integrity_sha256']=digest
        next(e for e in context['integrity_journal'] if e.get('checkpoint_index')==2 and e['event']=='INTEGRITY_CHECK_PASSED')['proof_sha256']=digest
    report=analysis.aggregate(*args,restart=context)
    assert not report['complete'] and report['errors']
    assert report['reports']['OPEN_VALIDATION']['frozen_fixed_primary']['accuracy'] is None
    assert not report['reports']['OPEN_VALIDATION']['contrasts']


def test_registered_caps_scoring_and_failed_original_results_remain_closed():
    args,context=fixture()
    args[0].pop(next(i for i,r in enumerate(args[0]) if r['physical_run'] and r['split']=='OPEN_VALIDATION'))
    assert not analysis.aggregate(*args,restart=context)['complete']
    args,context=fixture();args[6]['status']='FAIL'
    assert not analysis.aggregate(*args,restart=context)['complete']
    args,context=fixture();args[6]['checkpoint_files_unchanged']=False
    assert not analysis.aggregate(*args,restart=context)['complete']


def test_original_scientific_body_and_all_unchanged_helpers_are_AST_preserved():
    package=Path(analysis.__file__).parent;old_path=package/'bulk_analysis.py'
    assert hashlib.sha256(old_path.read_bytes()).hexdigest()==analysis.ORIGINAL_ANALYZER_SHA
    old={n.name:n for n in ast.parse(old_path.read_text()).body if isinstance(n,ast.FunctionDef)}
    new={n.name:n for n in ast.parse(Path(analysis.__file__).read_text()).body if isinstance(n,ast.FunctionDef)}
    for name in old:
        if name=='run':continue
        counterpart=new['_aggregate_scientific'] if name=='aggregate' else new[name]
        normalized=deepcopy(counterpart);normalized.name=name
        assert ast.dump(old[name],include_attributes=False)==ast.dump(normalized,include_attributes=False),name
