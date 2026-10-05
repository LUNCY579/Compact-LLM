"""Fixed refreshed05/06 owned inventories: source/JSON only, no resource checks."""
import importlib.util,json,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
HEAD='1dc787533d4e57257c5ae20b8e5324fb49894b70'
ANCHORS={5:('f2fad910e7d3dd9f82793db0864d633f4bbf1cf656af7e082fcd7ae83a6aab96','e6c49948f28a53ded9a029f4dab569b234c9a8e37fc2f0d28d841165970bcaeb'),
    6:('9dba239f5ccce2ab604f2f80e1e8c17e14610382a36eb0c9c9d5ad1a6a01f847','1bb00e7900819e7cd3a9c4a531c728b423765791444a5bb3d35891fad7dfeba8')}
def main():
    path=PACKAGE/'artifacts/phase3_launch_registration_work/prepare_registration.py'
    spec=importlib.util.spec_from_file_location('head1dc_metadata_prepare',path);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)
    p=r.pure;record=p.read_json(HERE/'ROOT_HEAD1DC_INVENTORY_FREEZE_001.json')
    assert record['source_commit']==HEAD and record['sha256']==p.digest({k:v for k,v in record.items() if k!='sha256'})
    results=[]
    for entry in record['inventories']:
        ordinal=entry['ordinal'];inventory=Path(entry['path']);file_sha,body_sha=ANCHORS[ordinal]
        assert (entry['file_sha256'],entry['body_sha256'])==(file_sha,body_sha)
        value=p.read_json(inventory)
        assert p.file_info(inventory)['sha256']==file_sha and value['sha256']==body_sha==p.digest({k:v for k,v in value.items() if k!='sha256'})
        old=p.read_json(PACKAGE/('artifacts/short_m0_audit_registrations/SYNTH_SHORT_PHASE3_BATCH%02d_A1_AUDIT_INVENTORY_001.json'%ordinal))
        assert set(value)==r.FIELDS and {k for k in old if old[k]!=value[k]}=={'source_commit','sha256'} and value['source_commit']==HEAD
        cfg_path=PACKAGE/('configs/SYNTH_SHORT_M0_ATOMIC_BATCH%02d_PHASE3_A1.json'%ordinal);cfg=p.read_json(cfg_path)
        candidate=HERE/('SYNTH_SHORT_M0_ATOMIC_BATCH%02d_PHASE3_A1_CANDIDATE.json'%ordinal)
        assert p.read_bytes(cfg_path)==p.read_bytes(candidate) and p.file_info(cfg_path)['sha256']==value['config_file_sha256']
        assert value['inventories']=={k:cfg[v] for k,v in r.SECTIONS.items()} and value['counts']==r.COUNTS
        assert value['profile_body_sha256']==r.PROFILE_SHA==p.digest(cfg['short_phase_observation_profile'])
        assert value['auditor_dependency_pins']==r.dependency_pins() and len(value['auditor_dependency_pins'])==6
        before={path:p.file_info(path)['sha256'] for path in cfg['source_pins']};assert before==cfg['source_pins'] and len(before)==131
        r.auditor._source_admission(value)
        sup={'preflight':{'source_commit':HEAD},'short_RAM_monitor_diagnostic':{'path':'SHORT_RAM_ABORT_DIAGNOSTIC.json',
            'attempted':False,'persisted':False,'write_error':None,'original_abort_reason_preserved':None}}
        assert r.auditor._phase3_admission(cfg,sup,value,r.PROFILE_SHA,cfg['short_phase_observation_profile'],{},p.require)==r.COUNTS
        assert before=={path:p.file_info(path)['sha256'] for path in cfg['source_pins']} and p.file_info(inventory)['sha256']==file_sha
        results.append({'ordinal':ordinal,'inventory_path':str(inventory),'inventory_file_sha256':file_sha,'inventory_body_sha256':body_sha,
            'only_HEAD_and_checksum_changed':True,'config_candidate_bytes_exact':True,'source131_and_auditor6_current_exact_stable':True})
    assert {x['ordinal'] for x in results}=={5,6}
    receipt={'schema_version':1,'kind':'HEAD1DC_FIXED05_06_FINAL_SOURCE_JSON_INVENTORY_REVIEW','launch_HEAD_root_supplied':HEAD,
        'batches':results,'all14_fields_maps_profile_runtime64_AST_exact':True,'actual_pure_source_and_phase_admission_executed':True,
        'live_Git_resource_model_decoder_asset_probe_or_worker_calls':0,'qualification_training_or_study_result':False,
        'tracked_source_inventory_config_or_HEAD_mutated':False}
    receipt['sha256']=p.digest(receipt);target=HERE/('HEAD1DC_INVENTORY_REVIEW_%d.json'%time.time_ns())
    target.write_text(p.canonical(receipt)+'\n',encoding='ascii')
    print(json.dumps({'receipt':str(target),'receipt_sha256':receipt['sha256'],'batches':[5,6],'source131_auditor6_stable':True,'native_probe_worker_calls':0}))
if __name__=='__main__':main()
