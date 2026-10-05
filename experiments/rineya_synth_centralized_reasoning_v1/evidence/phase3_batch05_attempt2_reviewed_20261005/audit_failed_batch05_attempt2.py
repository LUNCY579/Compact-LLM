"""Bounded preserved failure byte/identity/cost/chronology audit. No success auditor or probes."""
import hashlib,importlib.util,json,math,time
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
SUP='752e2097ab3484bc46ce33b5d854e03a485e2999829935c7a87f1b7b52dfd30b'
PRES_FILE='98e33e478dabf0360eb743c9915a9fca8b7e29563b521abd0beefaacae509214'
PRES_BODY='f9d171665abbbe3140aabd78c4b3e56b2fd2cb0eed18981fc0581ae8b1495ac7'
DEATH_FILE='a81eecb57fe5aa396b5feb45437b3326ef1c5228d6caced1fd04861e2291d15f'
DEATH_BODY='72c368c6538812a639aa274b7c74a86dee2bd591fe2441cf44c31d0a8a22f3f8'
WRAPPER='1941f5901ea12e87096c767a6341f69d728063b0c66f6d8d8272532701c02b99'
def main():
    path=HERE/'run_head1dc_terminal.py';assert hashlib.sha256(path.read_bytes()).hexdigest()==WRAPPER
    spec=importlib.util.spec_from_file_location('readonly_failed_head1dc_wrapper',path);w=importlib.util.module_from_spec(spec);spec.loader.exec_module(w)
    item=w.scope(5);m,_=w.original_adapter.adapted(5);p=m.pure
    cfg=m.exact_json(item['config'],item['config_sha']);inv=m.exact_json(item['inventory'],item['inventory_sha'])
    leaf,saved=item['raw'],item['saved'];sup=m.exact_json(leaf/'SUPERVISOR_RECEIPT.json',SUP)
    pres=m.exact_json(saved/'preservation_receipt.json',PRES_FILE);assert pres['sha256']==PRES_BODY==p.digest({k:v for k,v in pres.items() if k!='sha256'})
    death_path=HERE/'ROOT_CURRENT_OWNED_DEATH_BATCH05_A2_001.json';death=m.exact_json(death_path,DEATH_FILE)
    assert death['sha256']==DEATH_BODY and pres['status']=='FAILED_OR_NO_WORKER_TERMINAL_BYTES_PRESERVED_NOT_NATIVE_PASS'
    with (patch.object(m.registration.subprocess,'run',side_effect=AssertionError('live OS/Git probe forbidden')) as probe,
            patch.object(m.registration.auditor,'audit_terminal',side_effect=AssertionError('success audit forbidden')) as success):
        gate=m.terminal_gate(w.PACKAGE,leaf,cfg,sup,SUP,w.HEAD,item['config_sha'],death_path,DEATH_FILE)
        assert gate==pres['terminal_ownership'] and gate['Root_current_death_observation']['all_observed_owned_births']==3
        raw=m.raw_inventory(leaf,sup);assert len(raw)==288 and raw==pres['raw_files']=={n:p.file_info(saved/n) for n in raw}
        assert pres['source_commit']==inv['source_commit']==w.HEAD
        assert all(pres[k] is False for k in ('full_native_audit_eligible','native_PASS_obtained','scoring_authorized','training_authorized','study_completed'))
        assert sup['status']=='OWNED_WORKER_ABORTED' and sup['failure']=='stale_external_CPU_watchdog' and sup['returncode']==137
        worker=p.read_json(leaf/'worker_result.json');final=p.read_json(leaf/'PHASE_ACCOUNTING_FINAL.json')
        err=worker['error'];assert err['type']=='RuntimeError' and 'illegal memory access' in err['message']
        assert worker['status']=='SHORT_M0_RAW_COLLECTION_FAILED' and final['status']=='FAILED_PHASE_OBSERVATION'
        assert final['primary_error']==err and final['terminal_error'] is final['close_error'] is None
        assert final['identity']['source_commit']==w.HEAD and final['identity']['config_sha256']==item['config_sha']
        assert worker.get('collection',{}).get('rows',[])==[]
        costs=worker['collection_costs'];assert costs['attempted_forwards']==253 and costs['input_positions']=={'attempted':89757,'returned':89757,'failed':0,'pending':0}
        assert costs['operations']['inference_forward']=={'attempted':253,'returned':253,'failed':0,'pending':0}
        dose=worker['forward_dose'];assert dose['attempted_calls']==255 and dose['input_positions']==90325
        assert 89757+2*cfg['rows'][0]['prompt_tokens']==90325 and all(worker.get(k)==0 for k in ('backwards','optimizer_steps'))
        token_names=[n for n in raw if n.startswith('stage_raw_row')];row_names=[n for n in raw if n.startswith('stage_inference_row_')]
        assert len(token_names)==253 and len(row_names)==19
        failure=p.read_json(leaf/'stage_inference_failure_0020.json')
        assert failure['status']=='COLLECTION_FAILED' and failure['finish_reason']=='error' and failure['decoded_response'] is None and failure['error']==err
        assert failure['prompt_tokens']==cfg['rows'][19]['prompt_tokens']==414 and failure['source_packet_sha256']==cfg['source_packet_sha256']
        markers=p.read_lines(leaf/'PHASE_OBSERVATIONS.jsonl',canonical_lines=True)
        times=[v['phase']['monotonic'] for v in markers];assert all(math.isfinite(t) for t in times) and times==sorted(times)
        error_markers=[v['phase'] for v in markers if v['phase'].get('error')==err]
        assert error_markers
        first_error=min(v['monotonic'] for v in error_markers)
        diag=p.read_json(leaf/'SHORT_RAM_ABORT_DIAGNOSTIC.json')
        assert diag['failure']==sup['failure'] and diag['CPU_predicate_branch']=='over_200ms_age'
        assert diag['CPU_freshness_seconds']==.2 and math.isclose(diag['selected_CPU_age_seconds'],.641,abs_tol=1e-9)
        assert first_error<diag['check_monotonic'] and first_error<times[-1]<=diag['check_monotonic']
        assert diag['causal_OS_or_batch04_explanation_established'] is False
        resources=p.read_json(leaf/'RESOURCE_OBSERVATIONS.json');gpu=[r['gpu_free_bytes'] for r in resources if r.get('kind')=='GPU']
        assert min(gpu)==4599869440 and min(gpu)>=.5*1024**3
        assert worker['wall_seconds']<sup['elapsed_seconds']<600 and math.isclose(worker['wall_seconds'],84.516,abs_tol=.001)
        sources={path:p.file_info(path) for path in cfg['source_pins']};snapshots={n:p.file_info(saved/'source_snapshots'/n) for n in pres['source_snapshots']}
        assert len(sources)==len(snapshots)==131 and snapshots==pres['source_snapshots']
        assert all(sources[path]['sha256']==pin and sources[path]==snapshots[Path(path).relative_to(w.PACKAGE.parents[1]).as_posix()] for path,pin in cfg['source_pins'].items())
        assert p.file_info(item['config'])==p.file_info(saved/'config_snapshot.json')==pres['config_snapshot']
        assert p.file_info(item['inventory'])==p.file_info(saved/'audit_inventory_snapshot.json')==pres['audit_inventory_snapshot']
        assert inv['auditor_dependency_pins']==m.registration.dependency_pins()
        assert all(p.file_info(w.PACKAGE.parents[1]/n)==p.file_info(saved/'auditor_dependency_snapshots'/n)==info for n,info in pres['auditor_dependency_snapshots'].items())
        assert probe.call_count==success.call_count==0
    assert p.file_info(saved/'preservation_receipt.json')['sha256']==PRES_FILE and p.file_info(leaf/'SUPERVISOR_RECEIPT.json')['sha256']==SUP
    result={'schema_version':1,'kind':'PASS_BOUNDED_PRESERVED_FAILED_BATCH05_ATTEMPT2_METADATA_BYTE_CHRONOLOGY_AUDIT',
        'SUP_file_sha256':SUP,'preservation_file_sha256':PRES_FILE,'preservation_body_sha256':PRES_BODY,
        'death_file_sha256':DEATH_FILE,'death_body_sha256':DEATH_BODY,'source_HEAD':w.HEAD,'owned_births_currently_dead_Root_proof':3,
        'raw_pairs_verified':288,'source_snapshot_pairs_verified':131,'auditor_dependency_pairs_verified':6,
        'worker_error':{'type':err['type'],'message':err['message']},'supervisor_abort':'stale_external_CPU_watchdog',
        'first_worker_error_marker_monotonic':first_error,'later_CPU_abort_check_monotonic':diag['check_monotonic'],
        'worker_error_recorded_before_later_CPU_abort':True,'capture_attempted_returned_forwards':253,'capture_input_positions':89757,
        'physical_forwards':255,'physical_input_positions':90325,'durable_partial_final_rows':19,'durable_token_fragments':253,
        'accepted_complete64_rows':0,'recorded_CPU_age_seconds':diag['selected_CPU_age_seconds'],'unchanged_CPU_age_limit_seconds':.2,
        'recorded_min_GPU_free_bytes':min(gpu),'causal_or_numerical_or_quality_conclusion':False,
        'real_OS_Git_GPU_model_assets_decoder_probes_success_audit_grade_retry':False,'scope_accepted_for_complete_M0_population':False}
    result['sha256']=p.digest(result);target=HERE/('FAILED_BATCH05_A2_PRESERVED_AUDIT_%d.json'%time.time_ns())
    target.write_text(p.canonical(result)+'\n',encoding='ascii')
    print(json.dumps({'receipt':str(target),'receipt_sha256':result['sha256'],'raw_pairs':288,'source_pairs':131,'both_errors_retained':True,
        'worker_error_before_later_CPU_abort':True,'accepted64_population':False,'live_probe_success_audit_retry_calls':0}))
if __name__=='__main__':main()
