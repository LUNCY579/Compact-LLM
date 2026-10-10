"""Model-free tests of real scope, consumed-study refusal and bootstrap guards."""
import json
from pathlib import Path
import pytest
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import authority, probe_entry


def admission():
    scope=json.loads((authority.PACKAGE/'ACTIVE_GOAL_SCOPE_001.json').read_text(encoding='utf-8'))
    return {**scope['probe'], 'status':'READY_FOR_PASSIVE_PROBE_FP32_DISPATCH',
        'lane':'GPU_FP32_PASSIVE_INPUT_FEATURE_PROBE',
        'authority_path':str(authority.PACKAGE/'ACTIVE_GOAL_SCOPE_001.json'),
        'authority_sha256':authority.sha(authority.PACKAGE/'ACTIVE_GOAL_SCOPE_001.json'),
        'goal_snapshot_path':str(authority.PACKAGE/'ACTIVE_GOAL_SNAPSHOT_001.json'),
        'goal_snapshot_sha256':authority.sha(authority.PACKAGE/'ACTIVE_GOAL_SNAPSHOT_001.json'),
        'parent_host_available_floor_bytes':7*authority.GIB,'worker_host_available_floor_bytes':6*authority.GIB,
        'worker_rss_ceiling_bytes':6*authority.GIB,'gpu_initial_free_floor_bytes':6*authority.GIB,
        'gpu_allocated_ceiling_bytes':5*authority.GIB,'gpu_global_free_reserve_bytes':authority.GIB,
        'global_finalization_reserve_seconds':1800,'training':False,'native_test':False,
        'device':'cuda:0','dtype':'float32','attn_implementation':'eager','use_cache':False}


def test_actual_new_scope_and_unused_study():
    assert authority.validate_probe_unused(admission())['probe']['cases']==192


@pytest.mark.parametrize('key,value', [('model_API_cap',452),('hard_deadline_seconds',901),
    ('state_adjoints',1),('generation',True),('training',True),('native_test',True),
    ('maximum_prefix_tokens',358),('gpu_initial_free_floor_bytes',5*authority.GIB),
    ('internal_reserve_seconds',0),('global_finalization_reserve_seconds',0),('use_cache',True)])
def test_scope_rejects_expansion_or_lowered_guards(key,value):
    data=admission();data[key]=value
    with pytest.raises(PermissionError):authority.validate_probe_scope(data)


def test_old_allowance_is_not_new_goal_authority():
    data=admission();data['authority_path']=str(authority.CAMPAIGN/'HUMAN_FULL_GENERATION_RESTART_ALLOWANCE_001.json')
    with pytest.raises(PermissionError):authority.validate_probe_scope(data)


def test_started_study_is_consumed_even_without_terminal(monkeypatch):
    original=Path.read_text
    def read(path,*args,**kwargs):
        if path==authority.CAMPAIGN/'BUDGET_LEDGER.jsonl':
            return json.dumps({'event':'native_started','study_id':'V3-PASSIVE-PROBE-001','worker_pid':123})+'\n'
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',read)
    with pytest.raises(PermissionError,match='already attempted'):authority.validate_probe_unused(admission())


@pytest.mark.parametrize('clock', ['nan','inf','-inf','1e99'])
def test_bad_parent_clock_stops_before_worker_import(monkeypatch,tmp_path,clock):
    data=tmp_path/'admission.json';data.write_text(json.dumps(admission()),encoding='utf-8')
    monkeypatch.setenv('PASSIVE_PROBE_PARENT_STARTED_PERF_COUNTER',clock)
    invoked=[];monkeypatch.setattr(probe_entry.runpy,'run_module',lambda *a,**k:invoked.append(a))
    with pytest.raises(PermissionError):probe_entry.main(tmp_path/'release.json',tmp_path,tmp_path/'out',data)
    assert invoked==[]


def test_invalid_release_stops_before_worker_import(monkeypatch,tmp_path):
    data=tmp_path/'admission.json';data.write_text(json.dumps(admission()),encoding='utf-8')
    release=tmp_path/'release.json';release.write_text(json.dumps({'contained':False,'token':'x'}),encoding='utf-8')
    monkeypatch.setenv('PASSIVE_PROBE_PARENT_STARTED_PERF_COUNTER',str(probe_entry.time.perf_counter()))
    invoked=[];monkeypatch.setattr(probe_entry.runpy,'run_module',lambda *a,**k:invoked.append(a))
    with pytest.raises(PermissionError):probe_entry.main(release,tmp_path,tmp_path/'out',data)
    assert invoked==[]
