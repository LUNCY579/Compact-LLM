"""Bounded metadata/byte audit of an admission refusal. No successful-native auditor."""
import importlib.util, json, math, time
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
PRES_FILE='7b5b2ae62ec6b1d26e7f2f5d27f6225b92be8ca111c88fb4a0ca3a5e56ecc26d'
PRES_BODY='ff4689049bb518bd08edbc1aa2f3eb3437fab77aef0a289a1a007cb4c66b49b8'
SUP_SHA='d8b1edf4ddec266ed4200dee197454af8221b19173a541011d1daa3dd9b8f680'
def main():
    spec=importlib.util.spec_from_file_location('preserved_denial_fixed_adapter',HERE/'preserve_fixed_remaining.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    m,_=a.adapted(5);p=m.pure
    leaf=a.PACKAGE/'artifacts/short_m0_atomic_batch05_phase3_attempt1_20261005';saved=a.PACKAGE/'evidence'/leaf.name
    cfg_path=a.PACKAGE/'configs/SYNTH_SHORT_M0_ATOMIC_BATCH05_PHASE3_A1.json'
    inv_path=a.PACKAGE/'artifacts/short_m0_audit_registrations/SYNTH_SHORT_PHASE3_BATCH05_A1_AUDIT_INVENTORY_001.json'
    cfg=m.exact_json(cfg_path,a.ANCHORS[5][0]);inv=m.exact_json(inv_path,a.ANCHORS[5][1])
    sup=m.exact_json(leaf/'SUPERVISOR_RECEIPT.json',SUP_SHA);pres=m.exact_json(saved/'preservation_receipt.json',PRES_FILE)
    assert pres['sha256']==PRES_BODY==p.digest({k:v for k,v in pres.items() if k!='sha256'})
    assert pres['status']=='FAILED_OR_NO_WORKER_TERMINAL_BYTES_PRESERVED_NOT_NATIVE_PASS'
    assert pres['source_commit']==inv['source_commit']==a.HEAD and pres['raw_directory']==str(leaf) and pres['preserved_directory']==str(saved)
    assert all(pres[k] is False for k in ('full_native_audit_eligible','native_PASS_obtained','scoring_authorized','training_authorized','study_completed','preserver_performed_live_identity_or_OS_probe'))
    with (patch.object(m.registration.subprocess,'run',side_effect=AssertionError('live Git/probe forbidden')) as probe,
            patch.object(m.registration.auditor,'audit_terminal',side_effect=AssertionError('successful native auditor forbidden')) as native):
        gate=m.terminal_gate(a.PACKAGE,leaf,cfg,sup,SUP_SHA,a.HEAD,a.ANCHORS[5][0],None,None)
        raw=m.raw_inventory(leaf,sup)
        assert set(raw)=={'SUPERVISOR_RECEIPT.json'} and raw==pres['raw_files']
        assert {name:p.file_info(saved/name) for name in raw}==raw
        assert gate==pres['terminal_ownership'] and gate['no_worker_process_created'] is True and gate['Root_current_death_observation'] is None
        assert sup['status']=='ADMISSION_DENIED' and all(sup[k] is False for k in ('worker_attempted','worker_launched','worker_resumed','production_training_authorized','protected_bank_access','mechanical_qualification_established'))
        assert sup['partial_outputs']==[] and sup['returncode'] is None and sup.get('dispatch_observation') is None
        lease=Path(gate['lease_path']);assert not (lease/'worker.json').exists() and not (lease/'resume.json').exists()
        assert set(gate['lease_files'])=={'reservation.json','terminal.json'} and p.file_info(lease/'terminal.json')==raw['SUPERVISOR_RECEIPT.json']
        observations=sup['admission_observations'];assert len(observations)==1
        obs=observations[0]
        assert obs['gpu_free_bytes']==1428140032 < 4*1024**3 and sup['failure']=='RuntimeError: admission_floor_gpu_free_bytes'
        assert obs['owned_process_identities']==obs['owned_process_samples']==[] and obs['worker_tree_rss_bytes']==0
        assert m.aware_utc(sup['started_utc'])<=m.aware_utc(sup['finished_utc']) and math.isfinite(sup['elapsed_seconds']) and 0<=sup['elapsed_seconds']<600
        assert p.file_info(cfg_path)==p.file_info(saved/'config_snapshot.json')==pres['config_snapshot']
        assert p.file_info(inv_path)==p.file_info(saved/'audit_inventory_snapshot.json')==pres['audit_inventory_snapshot']
        sources={path:p.file_info(path) for path in cfg['source_pins']}
        snapshots={name:p.file_info(saved/'source_snapshots'/name) for name in pres['source_snapshots']}
        assert len(sources)==len(snapshots)==131 and snapshots==pres['source_snapshots']
        assert all(sources[path]['sha256']==pin and sources[path]==snapshots[Path(path).relative_to(a.PACKAGE.parents[1]).as_posix()] for path,pin in cfg['source_pins'].items())
        assert inv['inventories']['source']==cfg['source_pins'] and inv['auditor_dependency_pins']==m.registration.dependency_pins()
        assert all(p.file_info(a.PACKAGE.parents[1]/name)==p.file_info(saved/'auditor_dependency_snapshots'/name)==info for name,info in pres['auditor_dependency_snapshots'].items())
        assert probe.call_count==native.call_count==0
    assert p.file_info(saved/'preservation_receipt.json')['sha256']==PRES_FILE and p.file_info(leaf/'SUPERVISOR_RECEIPT.json')['sha256']==SUP_SHA
    result={'schema_version':1,'kind':'PASS_BOUNDED_BATCH05_ADMISSION_DENIED_PRESERVATION_METADATA_BYTE_AUDIT',
        'config_file_sha256':a.ANCHORS[5][0],'inventory_file_sha256':a.ANCHORS[5][1],'SUP_file_sha256':SUP_SHA,
        'preservation_file_sha256':PRES_FILE,'preservation_body_sha256':PRES_BODY,'launch_HEAD':a.HEAD,
        'one_raw_byte_pair_verified':True,'source131_snapshot_pairs_verified':True,'auditor6_original_snapshot_pins_verified':True,
        'recorded_GPU_free_bytes':obs['gpu_free_bytes'],'unchanged_admission_floor_bytes':4*1024**3,
        'worker_attempted_launched_resumed_all_false':True,'no_worker_resume_files':True,'Root_death_proof_required':False,
        'recorded_admission_and_terminal_times_consistent':True,'no_score_training_or_success_authority':True,
        'live_Git_resource_GPU_model_decoder_probes_assets_or_success_audit':False,'retry_or_actual_model_work':False}
    result['sha256']=p.digest(result);target=HERE/('BATCH05_DENIAL_PRESERVED_AUDIT_%d.json'%time.time_ns())
    target.write_text(p.canonical(result)+'\n',encoding='ascii')
    print(json.dumps({'receipt':str(target),'receipt_sha256':result['sha256'],'denial_verified':True,'raw_pairs':1,'source_pairs':131,'worker_or_success_audit_calls':0}))
if __name__=='__main__':main()
