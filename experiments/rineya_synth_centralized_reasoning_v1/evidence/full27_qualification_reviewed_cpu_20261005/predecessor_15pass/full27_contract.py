"""Staged three fresh-process9-update qualification scopes; never study authority."""
from copy import deepcopy
import math
import json
import os
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parents[1] if HERE.name=='full27_qualification_work' and HERE.parent.name=='artifacts' else HERE
sys.path.insert(0,str(PACKAGE));sys.path.insert(0,str(HERE))
import backend_qual_contract as prerequisite
import native_contract as native
import study_schedule as schedule
from rs_contract import canonical,digest,require

PURPOSE='full27_three_fresh_process_nine_update_qualification_only'
ID='SYNTH-FULL27Q-001'
VERSION='synth-full27-three-owned-fresh-nine-update-qualification-1'
PROFILE='BF16_FP32_QV8_FULLRECOMPUTE_WORKER_RAM_FULL27_QUALIFICATION_V1'
AUTHORIZATION={'scope':PURPOSE,'production_training':False,'study_training':False,'study_warmstart':False,
               'DEV_collection':False,'scoring':False,'protected_bank_access':False,'paid_services':False}
FILES=('full27_contract.py','full27_chunk.py','full27_worker.py','full27_supervisor.py')
B_AUDIT=PACKAGE/'evidence/backendq003_normalized_witness_B_attempt1_independent_audit_20261005.json'
B_AUDIT_SHA='611ee56ff7da60f9a0e051e8b73e35d7e31813a080bc685b3ed06709181fb5c5'
B_AUDIT_BODY_SHA='c832da0ba2eed596110f4c0618bba50a887511d8802309d718deba1d56232717'
AUDIT_KIND='SYNTH_FULL27_QUALIFICATION_CHUNK_INDEPENDENT_AUDIT_V1'


def source_pins():
    # Retain the complete already qualified A/B source closure, add this scope.
    paths=[Path(name) for name in prerequisite.source_pins()]
    paths += [HERE/n for n in FILES]
    tests=HERE if HERE!=PACKAGE else PACKAGE/'tests'
    paths += sorted(tests.glob('test_full27*.py'))
    require(all(p.is_file() and not p.is_symlink() for p in paths),'complete qualification source closure')
    return {str(p.resolve()):native.filehash(p) for p in paths}


def packet(registration,arm):
    schedule.validate_registration(registration)
    require(arm in ('M1','M2'),'exact adapted arm')
    _,_,pools,records=schedule._inputs()
    rows=[]
    for update in registration['arms'][arm]['updates']:
        for identity in update['occurrence_row_ids']:
            row=deepcopy(pools[arm][identity]);schedule.validate_token_row(row)
            require(records[identity]['arm']==arm and records[identity]['split']=='train' and records[identity]['skill'] in ('R','S'),'atomic TRAIN only')
            rows.append(row)
    require(len(rows)==540,'all27updates540occurrences frozen beforeoutputs')
    return rows


def expected_work(registration,arm,chunk,rows):
    require(type(chunk) is int and chunk in (1,2,3),'one fixed qualification chunk')
    start=(chunk-1)*180;training=rows[start:start+180];reference=training[:20] if chunk>1 else[]
    count=len(training)+len(reference)
    return {'train_forwards':180,'reference_forwards':len(reference),'zero_parity_forwards':2,
        'maximum_forward_calls':count+2,'expected_forward_input_positions':sum(r['total_tokens'] for r in training+reference)+2*training[0]['total_tokens'],
        'maximum_forward_input_positions':102400,'context_cap':512,'native_backwards':180,'native_optimizer_steps':9,
        'reference_autograd_calls':len(reference),'decoder_counts':{'no_grad':56,'forward':count*28,'recompute':count*28},
        'completed_ordinal':chunk*9,'optimizer_final_LR':schedule.learning_rates()[chunk*9-1],
        'native_base_content_checks':15 if chunk==1 else 19,
        'training_input_positions':sum(r['total_tokens'] for r in training),
        'training_supervised_positions':sum(r['supervised_tokens'] for r in training),
        'whole_source_occurrences_prevalidated':540}


def verify_completed_B(receipt_path=B_AUDIT,expected_sha256=B_AUDIT_SHA):
    """Independently selected audit bytes are a prerequisite, never reused state."""
    path=Path(receipt_path)
    require(path.resolve()==B_AUDIT.resolve() and expected_sha256==B_AUDIT_SHA and path.is_file()
            and not path.is_symlink() and native.filehash(path)==expected_sha256,'trusted completed B audit bytes')
    value=native.read_json(path)
    require(value.get('sha256')==B_AUDIT_BODY_SHA==digest({k:v for k,v in value.items() if k!='sha256'})
            and value.get('status')=='PASS_SCOPED_FRESH_PROCESS_C9_RESTORE_REFERENCE20_UPDATE10_BITWISE_CONTINUATION_MECHANICS_ONLY'
            and value.get('one_fresh_process_update10_continuation_mechanics_verified') is True
            and value.get('reference_gradient_control_verified') is True
            and all(value.get(name) is False for name in ('study_warmstart_allowed','study_training_authorized',
                'full_resumed9_qualified','full27_qualified','scoring_authorized','DEV_collection_authorized')),
            'completed independently audited fresh one-update B only; no inherited full27 authority')
    config=Path(value['config_path'])
    require(config.resolve().parent==(PACKAGE/'configs').resolve() and not config.is_symlink()
            and native.filehash(config)==value['config_file_sha256'],'audited B config bytes')
    body=native.read_json(config)
    require(body['qualification_recipe']['sha256']==value['qualification_recipe_sha256'],'B recipe binding')
    for name,pin in body['source_pins'].items():require(native.filehash(Path(name))==pin,'qualified A/B source drift')
    return {'path':str(path.resolve()),'sha256':expected_sha256,'body_sha256':B_AUDIT_BODY_SHA,
            'config_path':str(config.resolve()),'config_sha256':value['config_file_sha256'],'state_reuse_allowed':False,
            'only_one_restored_update10_previously_qualified':True}


def checkpoint_binding(chunk):
    value=deepcopy(prerequisite.checkpoint_binding('A'));groups=180 if chunk==1 else 200
    value['counts']={'no_grad':56,'forward':groups*28,'recompute':groups*28,
                    'per_layer_no_grad':2,'per_layer_forward':groups,'per_layer_recompute':groups}
    value['authority_transfer']='Native7/INFQ and audited A/B helpers only; this distinct full27 scope remains unqualified'
    return value


def verify_decoder_file(summary,descriptor,amendment,leaf):
    """Same bounded paired-region validation as A/B, with this chunk's counts."""
    require(type(descriptor) is dict and set(descriptor)=={'path','rows','bytes','sha256'},'complete decoder file descriptor')
    path=Path(descriptor['path']);expected=sum(amendment['counts'][name] for name in ('forward','recompute','no_grad'))
    require(path.is_absolute() and path.is_file() and not path.is_symlink() and path.resolve().parent==Path(leaf).resolve()
            and descriptor['rows']==expected and type(descriptor['bytes']) is int and 0<descriptor['bytes']<=16*1024**2
            and path.stat().st_size==descriptor['bytes'] and native.filehash(path)==descriptor['sha256'],'exact bounded owned decoder bytes/rows')
    records=[]
    with path.open('rb') as stream:
        for _ in range(expected+1):
            raw=stream.readline(4097)
            if not raw:break
            require(len(raw)<=4096 and raw.endswith(b'\n'),'bounded complete decoder record')
            records.append(json.loads(raw.decode('utf-8'),object_pairs_hook=native._pairs,parse_constant=native._bad_constant))
    require(len(records)==expected and summary.get('removed') is True,'complete decoder work with exact attribute cleanup')
    value=deepcopy(summary);value['regions']=records
    prerequisite.validate_decoder_work(value,amendment)
    return True


def verify_previous(authority,registration,arm,chunk,recipe):
    """Pins supplied by the trusted scope creator, then re-read on every use."""
    import full27_chunk as runner
    fields={'result_path','result_sha256','supervisor_path','supervisor_sha256','audit_path','audit_sha256'}
    require(type(authority) is dict and set(authority)==fields and chunk in (2,3),'one exact audited previous qualification chunk')
    values={}
    for label in ('result','supervisor','audit'):
        path=Path(authority[label+'_path'])
        require(path.is_absolute() and path.is_file() and not path.is_symlink()
                and native.filehash(path)==authority[label+'_sha256'],'previous '+label+' bytes changed')
        values[label]=native.read_json(path)
    result,sup,audit=(values[name] for name in ('result','supervisor','audit'))
    require(Path(authority['result_path']).resolve().parent==Path(authority['supervisor_path']).resolve().parent
            and Path(authority['result_path']).resolve().parent.parent==(PACKAGE/'artifacts').resolve(),'previous owned output leaf')
    require(result.get('status')=='COMPLETE_QUALIFICATION_CHUNK_STUDY_UNAUTHORIZED'
            and result.get('scope')==runner.NATIVE_SCOPE and result.get('qualification_purpose')==PURPOSE
            and result.get('arm')==arm and result.get('chunk_ordinal')==chunk-1
            and result.get('sha256')==digest({k:v for k,v in result.items() if k!='sha256'}),'successful native same-arm previous chunk')
    require(sup.get('status')=='SUPERVISED_WORKER_EXITED_ZERO' and sup.get('returncode')==0
            and sup.get('job_closed_kill_on_close') is True and sup.get('owned_worker_identity_currently_alive') is False
            and sup.get('sampler_threads_terminal_verified') is True and sup.get('gpu_telemetry_cleanup_verified') is True
            and sup.get('config_sha256')==result.get('config_sha256'),'previous terminal containment/config closure')
    require(audit.get('kind')==AUDIT_KIND and audit.get('status')=='PASS_FULL27_QUALIFICATION_CHUNK_MECHANICS_ONLY'
            and audit.get('purpose')==PURPOSE and audit.get('arm')==arm and audit.get('chunk_ordinal')==chunk-1
            and audit.get('result_sha256')==authority['result_sha256'] and audit.get('supervisor_sha256')==authority['supervisor_sha256']
            and audit.get('sha256')==digest({k:v for k,v in audit.items() if k!='sha256'})
            and audit.get('study_warmstart_allowed') is False,'independently selected previous audit binding')
    prior=runner._validate_resume(result['resume_descriptor'],registration,arm,chunk,recipe,result['costs'])
    evidence=prior['qualification_evidence']
    require(result.get('qualification_witness')==evidence['witness'] and result.get('physical_dose')==evidence['physical_dose']
            and evidence.get('CPU_fake_control') is False and evidence.get('cleanup_completed') is True,
            'previous detached boundary and all cleanup costs are in the durable publication lineage')
    require(Path(prior['backend_descriptor']['path']).resolve().parent==Path(authority['result_path']).resolve().parent,'checkpoint belongs to prior owned leaf')
    dose=expected_work(registration,arm,chunk-1,packet(registration,arm))
    require(evidence['physical_dose']=={'attempted_calls':dose['maximum_forward_calls'],
        'input_positions':dose['expected_forward_input_positions'],'operations':{key:dose[key] for key in ('train_forwards','reference_forwards','zero_parity_forwards')}},'complete previous physical scope')
    summary=deepcopy(evidence['decoder_work']);descriptor=summary.pop('region_file')
    verify_decoder_file(summary,descriptor,checkpoint_binding(chunk-1),Path(authority['result_path']).parent)
    require(sup.get('preflight',{}).get('recipe')==recipe and sup['preflight'].get('dose')==dose
            and sup['preflight'].get('purpose')==PURPOSE and sup['preflight'].get('profile')==PROFILE
            and sup['preflight'].get('owned_qualification_authorized') is True,'previous native frozen purpose/source/profile/dose')
    return result


def build_scope(arm,chunk,*,previous_authority=None,authorize_owned_qualification=False):
    import full27_chunk as runner
    registration=schedule.read_json(PACKAGE/'STUDY_DEV_REGISTRATION_DRAFT.json');rows=packet(registration,arm)
    require(type(authorize_owned_qualification) is bool and ((chunk==1 and previous_authority is None)
            or (chunk in (2,3) and type(previous_authority) is dict)),'freshzero or exact prior qualification lineage')
    recipe=runner.recipe(registration,arm,runner.NATIVE_SCOPE);dose=expected_work(registration,arm,chunk,rows)
    previous=None if chunk==1 else verify_previous(previous_authority,registration,arm,chunk,recipe)
    require(dose['expected_forward_input_positions']<=102400,'unchanged qualification position cap')
    prior=native.read_json(PACKAGE/'configs/SYNTH_NQ001_PROC7.json')
    return {'schema_version':1,'id':ID,'contract':VERSION,'purpose':PURPOSE,'authorization':deepcopy(AUTHORIZATION),'arm':arm,'seed':101,'chunk':chunk,
        'registration':registration,'rows':rows,'recipe':recipe,'dose':dose,'resources':deepcopy(native.RESOURCE_BOUNDS),
        'tolerances':deepcopy(native.TOLERANCES),'profile':PROFILE,'base_compute_profile':prerequisite.PROFILE,
        'checkpoint_amendment':checkpoint_binding(chunk),'worker_ram_amendment':prerequisite.worker_ram_binding(),
        'source_pins':source_pins(),'runtime_pins':{**prior['runtime_pins'],
            **prerequisite.checkpoint_runtime_files(),**prerequisite.worker_ram_runtime_files()},
        'checkpoint_runtime_pins':prerequisite.checkpoint_runtime_files(),'worker_ram_runtime_pins':prerequisite.worker_ram_runtime_files(),
        'runtime_versions':deepcopy(native.RUNTIME),'offline_environment':deepcopy(native.OFFLINE_ENV),'precision':deepcopy(prior['precision']),
        'data_pins':deepcopy(prior['data_pins']),'model_dir':str(prerequisite.MODEL_DIR),'model_pins':dict(prerequisite.ASSET_PINS),
        'inference_qualification':prerequisite.inference_binding(),'global_learning_rates':schedule.learning_rates(),
        'source_packet_sha256':digest(rows),'owned_qualification_authorized':authorize_owned_qualification,
        'previous_authority':deepcopy(previous_authority),'previous':deepcopy(previous),'latest_costs':None if previous is None else deepcopy(previous['costs']),
        'completed_B_prerequisite':verify_completed_B(),
        'staged_native_dispatch_forbidden':HERE!=PACKAGE,'native_launch_authorized':False,'study_training_authorized':False,
        'study_warmstart_allowed':False,'continuous27_bitwise_parity_established':False,'qualification_checkpoints_never_final_study':True}


def preflight(path,*,require_clean=True):
    require(HERE==PACKAGE,'staged full27 native dispatch forbidden before independent integration')
    filename=Path(path).absolute()
    require(filename.parent.resolve()==(PACKAGE/'configs').resolve() and filename.is_file() and not filename.is_symlink(),'root frozen full27 config')
    body=native.read_json(filename)
    require(body.get('sha256')==digest({k:v for k,v in body.items() if k!='sha256'})
            and body.get('owned_qualification_authorized') is True,'explicit write-once owned qualification checksum')
    expected=build_scope(body['arm'],body['chunk'],previous_authority=body['previous_authority'],authorize_owned_qualification=True)
    require(canonical({k:v for k,v in body.items() if k!='sha256'})==canonical(expected),'frozen full27 authority/source/order/dose changed')
    require({**native.runtime_files(),**prerequisite.checkpoint_runtime_files(),**prerequisite.worker_ram_runtime_files()}==body['runtime_pins'],
            'all83 current qualified runtime pins retained')
    for name,pin in body['runtime_pins'].items():require(native.filehash(Path(name))==pin,'runtime byte drift')
    for name,pin in body['model_pins'].items():require(native.filehash(prerequisite.MODEL_DIR/name)==pin,'native model/tokenizer bytes changed')
    return {**body,'config_path':str(filename.resolve()),'config_sha256':native.filehash(filename),
            'source_commit':prerequisite._head(require_clean),'worker_path':str(PACKAGE/'full27_worker.py')}


def write_scope(path,arm,chunk,*,previous_authority=None,authorize_owned_qualification=False):
    require(HERE==PACKAGE,'staged CPU full27 preparation cannot create a native dispatch manifest')
    require(authorize_owned_qualification is True,'explicit owned full27 qualification authorization required')
    target=Path(path).absolute()
    require(target.parent.resolve()==(PACKAGE/'configs').resolve() and not target.exists() and not target.is_symlink(),'new write-once root full27 config')
    prerequisite._head(True)
    body=build_scope(arm,chunk,previous_authority=previous_authority,authorize_owned_qualification=True);body['sha256']=digest(body)
    with target.open('xb') as stream:
        raw=(canonical(body)+'\n').encode();require(stream.write(raw)==len(raw),'complete scope write');stream.flush();os.fsync(stream.fileno())
    return body
