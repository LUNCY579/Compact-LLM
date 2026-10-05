"""Remaining fixed config/inventory source-JSON review. No Git or native probes."""
import hashlib, importlib.util, json, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
HEAD = '9cabf4891221592be7946bb6ded7315bd2962185'
ANCHORS = {
    5: ('add9d169f88a6d62a8f77c521d0b7da57083264238f5b0f8a3ded290bc10624f',
        '14ff8b3942b95ed8a60a0ac02dcc2bb9ec4d6c1a4927625d50f1a440c64a6eaa',
        'b8e2efe1e832ca865bc8a5de8f48b9d601246cd14335a2e6e09ecc4a1362ba00'),
    6: ('a66a23c69209bc43eab58075e705ee1a2bd4793252e8f205f1926b603337da96',
        '2cfca96e208a4f0071e9430a28471f1fa13b441389226995967afc3eacf79c16',
        '06677a414195afb137daba2f434edd8f4ba0a4f82b029c52e17582d8fb6e095c')}

def main():
    stage = PACKAGE / 'artifacts/phase3_launch_registration_work'
    spec = importlib.util.spec_from_file_location('remaining_final_inventory_metadata_prepare', stage / 'prepare_registration.py')
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    p, auditor = prepare.pure, prepare.auditor
    old_path = PACKAGE / 'artifacts/short_m0_audit_registrations/SYNTH_SHORT_PHASE3_BATCH04_A1_AUDIT_INVENTORY_001.json'
    old = p.read_json(old_path)
    assert p.file_info(old_path)['sha256'] == 'cc28dd3d02a8f550699892428022e0ebf61c2cdcbf2dd9286f948cd1aba4a6ea'
    results = []
    for ordinal, (file_sha, body_sha, cfg_sha) in ANCHORS.items():
        config = PACKAGE / ('configs/SYNTH_SHORT_M0_ATOMIC_BATCH%02d_PHASE3_A1.json'%ordinal)
        candidate = HERE / ('SYNTH_SHORT_M0_ATOMIC_BATCH%02d_PHASE3_A1_CANDIDATE.json'%ordinal)
        inventory = PACKAGE / ('artifacts/short_m0_audit_registrations/SYNTH_SHORT_PHASE3_BATCH%02d_A1_AUDIT_INVENTORY_001.json'%ordinal)
        assert p.read_bytes(config) == p.read_bytes(candidate)
        assert p.file_info(config)['sha256'] == cfg_sha
        cfg = p.read_json(config)
        value = p.read_json(inventory)
        assert p.file_info(inventory)['sha256'] == file_sha
        assert value['sha256'] == body_sha == p.digest({k:v for k,v in value.items() if k!='sha256'})
        assert cfg['sha256'] == p.digest({k:v for k,v in cfg.items() if k!='sha256'})
        assert set(value) == prepare.FIELDS
        assert {k for k in old if old[k] != value[k]} == {'config_path','config_file_sha256','source_commit','sha256'}
        assert value['config_path'] == str(config.resolve()) and value['config_file_sha256'] == cfg_sha
        assert value['source_commit'] == HEAD
        assert value['inventories'] == {section:cfg[key] for section,key in prepare.SECTIONS.items()}
        assert value['counts'] == {'source':131,'runtime':65,'decoder':17,'data':5,'model':7}
        assert cfg['batch_selector'] == {'source_kind':'atomic_RS_DEV','ordinal':ordinal}
        assert len(cfg['rows']) == 64 and cfg['dose']['maximum_forward_calls'] == 2050
        assert value['profile_body_sha256'] == prepare.PROFILE_SHA == p.digest(cfg['short_phase_observation_profile'])
        assert value['auditor_dependency_pins'] == prepare.dependency_pins() and len(value['auditor_dependency_pins']) == 6
        source_before = {path:p.file_info(path)['sha256'] for path in cfg['source_pins']}
        assert source_before == value['inventories']['source']
        auditor._source_admission(value)
        sup = {'preflight':{'source_commit':HEAD},'short_RAM_monitor_diagnostic':{
            'path':'SHORT_RAM_ABORT_DIAGNOSTIC.json','attempted':False,'persisted':False,'write_error':None,'original_abort_reason_preserved':None}}
        auditor._phase3_admission(cfg,sup,value,prepare.PROFILE_SHA,cfg['short_phase_observation_profile'],{},p.require)
        source_after = {path:p.file_info(path)['sha256'] for path in cfg['source_pins']}
        assert source_before == source_after and p.file_info(config)['sha256'] == cfg_sha and p.file_info(inventory)['sha256'] == file_sha
        results.append({'ordinal':ordinal,'config_path':str(config),'config_file_sha256':cfg_sha,
            'inventory_path':str(inventory),'inventory_file_sha256':file_sha,'inventory_body_sha256':body_sha,
            'candidate_config_exact_bytes_equal':True,'all14_fields_and_complete_maps_verified':True,
            'all131_current_source_and6_auditor_bytes_exact_stable':True,'only_expected4_inventory_fields_changed':True,
            'source_packet_sha256':cfg['source_packet_sha256'],'dose':cfg['dose']})
    forbidden = [n for n in ('torch','transformers','tokenizers','psutil') if n in sys.modules]
    assert not forbidden
    receipt = {'schema_version':1,'kind':'REMAINING_BATCH05_06_FINAL_OWNED_CONFIG_INVENTORY_SOURCE_JSON_REVIEW',
        'root_supplied_clean_launch_HEAD':HEAD,'actual_Git_or_HEAD_probe_called':False,'batches':results,
        'actual_pure_source_and_phase_admission_validators_executed':True,'forbidden_runtime_imports':forbidden,
        'runtime_decoder_model_assets_binary_read_or_probes':False,'root_config_source_inventory_or_HEAD_modified':False,
        'worker_launch_or_grade_or_native_PASS_issued':False,'native_readiness_or_study_result_established':False}
    receipt['sha256'] = p.digest(receipt)
    target = HERE / ('FINAL_INVENTORY_REVIEW_%d.json'%time.time_ns())
    target.write_text(p.canonical(receipt)+'\n',encoding='ascii')
    print(json.dumps({'receipt':str(target),'receipt_sha256':receipt['sha256'],'batches_reviewed':[5,6],
        'configs_exact_candidates':True,'source131_and_auditor6_exact_stable':True,'native_probe_and_worker_calls':0}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
