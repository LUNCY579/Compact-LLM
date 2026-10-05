"""Actual batch05 admission-denied metadata join; no copy, success audit or probe."""
import importlib.util, json, time
from pathlib import Path
from unittest.mock import patch
HERE=Path(__file__).resolve().parent
def main():
    spec=importlib.util.spec_from_file_location('batch05_denial_fixed_adapter',HERE/'preserve_fixed_remaining.py')
    a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
    m,_=a.adapted(5);p=m.pure
    leaf=a.PACKAGE/'artifacts/short_m0_atomic_batch05_phase3_attempt1_20261005'
    config=a.PACKAGE/'configs/SYNTH_SHORT_M0_ATOMIC_BATCH05_PHASE3_A1.json'
    inv=a.PACKAGE/'artifacts/short_m0_audit_registrations/SYNTH_SHORT_PHASE3_BATCH05_A1_AUDIT_INVENTORY_001.json'
    sup_sha='d8b1edf4ddec266ed4200dee197454af8221b19173a541011d1daa3dd9b8f680'
    cfg=m.exact_json(config,a.ANCHORS[5][0]);frozen=m.exact_json(inv,a.ANCHORS[5][1])
    sup=m.exact_json(leaf/'SUPERVISOR_RECEIPT.json',sup_sha)
    with (patch.object(m.registration.subprocess,'run',side_effect=AssertionError('live probe/Git forbidden')) as probe,
            patch.object(m,'copy_exact',side_effect=AssertionError('real copy forbidden')) as copy):
        gate=m.terminal_gate(a.PACKAGE,leaf,cfg,sup,sup_sha,a.HEAD,a.ANCHORS[5][0],None,None)
        raw=m.raw_inventory(leaf,sup)
        assert sup['status']=='ADMISSION_DENIED' and all(sup[k] is False for k in ('worker_attempted','worker_launched','worker_resumed'))
        assert gate['no_worker_process_created'] is True and gate['Root_current_death_observation'] is None
        lease=Path(gate['lease_path']);assert not (lease/'worker.json').exists() and not (lease/'resume.json').exists()
        assert set(raw)=={'SUPERVISOR_RECEIPT.json'} and sup['partial_outputs']==[]
        assert probe.call_count==copy.call_count==0
    assert frozen['source_commit']==a.HEAD and frozen['inventories']['source']==cfg['source_pins']
    assert p.file_info(leaf/'SUPERVISOR_RECEIPT.json')['sha256']==sup_sha
    receipt={'schema_version':1,'kind':'FIXED_BATCH05_ACTUAL_ADMISSION_DENIED_READONLY_METADATA_JOIN',
        'SUP_file_sha256':sup_sha,'config_file_sha256':a.ANCHORS[5][0],'inventory_file_sha256':a.ANCHORS[5][1],
        'launch_HEAD':a.HEAD,'terminal_gate':gate,'raw_files':raw,'worker_attempted_launched_resumed_all_false':True,
        'death_proof_not_required_without_worker':True,'actual_probe_copy_success_audit_grade_or_retry_calls':0}
    receipt['sha256']=p.digest(receipt);target=HERE/('BATCH05_DENIAL_GATE_REVIEW_%d.json'%time.time_ns())
    target.write_text(p.canonical(receipt)+'\n',encoding='ascii')
    print(json.dumps({'receipt':str(target),'receipt_sha256':receipt['sha256'],'raw_files':1,'no_worker_no_death_proof':True,'probe_copy_calls':0}))
if __name__=='__main__':main()
