"""Separate phase-observed M0 profile; stage dispatch closes before any IO."""
from copy import deepcopy
from pathlib import Path
import os,sys
HERE=Path(__file__).resolve().parent
STAGED=HERE.name=='short_phase_collection_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1] if STAGED else HERE
ROOT=PACKAGE.parents[1]
sys.path.insert(0,str(PACKAGE))
import short_collection_contract as base
from rs_contract import canonical,digest,require
native=base.native
VERSION='synth-short-M0-phase-observed-staged-3'
PURPOSE='short_M0_phase_observed_raw_collection_one_fixed_batch_only'
PROFILE_ID='SHORT_M0_PHASE_OBSERVED_PHYSICAL_RAM_V3'
REGISTRATION_ID='SYNTH_SHORT_PHASE3_INDEPENDENT_AUDIT_INVENTORY_001'
SOURCE_NAMES=('short_phase_contract.py','short_phase_observation.py','short_phase_worker.py','short_phase_supervisor.py')
AUTHORIZATION={**base.AUTHORIZATION,'scope':PURPOSE}
D001_AUDIT=PACKAGE/'evidence/context_phase_no_model_D001_attempt1_independent_audit_20261005.json'
D001_FILE_SHA='f09ad290260eb6b1721a5b34ff1fddb4379b4688d149ca185d3b7b755afd83a0'
D001_BODY_SHA='6f573b9df28c70e06d600c69aadca26f86c7501c89fc966d3083c1aaaeb73acb'
HELPER_PINS={
 'context_phase_contract.py':'021f5b3afa037fcaf4e07db1c5596e5bb7c0f15aba13abf9d77d8942344c5d53',
 'context_phase_worker.py':'a4c69bb350a8255c638881c634f732a6b44cf1f0a5da583ba7d7bd0f901d4278',
 'inference_component.py':'48c57278339f4e42ef7487e296d6051edb55650e61d0397b1c73305720651b0b',
 'inference_qual_worker.py':'1172838c31c1b9440337707826573012e06f9a6a26b606a62a43976d19e99f1e',
 'native_supervisor.py':'edf2de5b0917150c2c79e6a555dc0399995d12859028e62a7f7ece86b767e691',
 'native_worker.py':'156c56c6b9980f30d950fd5245403cc58ec80cb5131fde69451156db9edc22d4',
 'short_collection_contract.py':'1d66d5823d7485864a4f3d887870e62ed68978f08df90f5f7698c4766d2d365e',
 'short_collection_supervisor.py':'f0d178cda0b20a804791ea9c95e89040d9d8dff38437f12ce0cbdddbaff9be09',
 'short_collection_worker.py':'f0ee2c5fd92397ee1ff8cbbd55596e50c2251e9c313542f493fa3609162b7e5b',
 'short_ram_monitor.py':'7c810bce7c3ee46a8b6f40ebd9f07c12a130ddd9fbd366719a88ef7ae82a13ee',
 'worker_ram_probe.py':'e4ec5e19241b877b99cc1ed933514c6fb09721658d91666df900a9a1e00f75a4'}
BASE_FIELDS=set(('schema_version id contract authorization profile resources tolerances runtime_versions runtime_pins '
 'decoder_runtime_pins offline_environment precision source_pins data_pins native7 inference1_audit_sha256 '
 'inference1_scope_transfer model_dir model_pins seed model_role cpu_plan_sha256 batch_selector batch short_RAM_monitor_profile '
 'rows source_packet_sha256 dose collection_is_not_grader_authority correctness_certified retention_certified training_authorized '
 'scoring_authorized language_qualification_established long_context_qualification_established prior_correct_enrollment_performed '
 'protected_bank_access paid_services confirmation_eligible quality_gain_established').split())

def integrated():require(not STAGED and HERE==PACKAGE,'staged phase-observed collector cannot authorize dispatch')

def phase_profile():
    for name,pin in HELPER_PINS.items():require(native.filehash(PACKAGE/name)==pin,'exact unchanged collector/observer helper '+name)
    require(native.filehash(D001_AUDIT)==D001_FILE_SHA,'exact independently reviewed D001 observation bytes')
    audit=native.read_json(D001_AUDIT)
    require(audit['sha256']==D001_BODY_SHA==digest({k:v for k,v in audit.items() if k!='sha256'})
        and audit['status']=='PASS_SCOPED_CURRENT_NO_MODEL_CONTEXT_PHASE_AND_GUARD_OBSERVATION_ONLY'
        and audit['model_loaded'] is audit['native_fit_established'] is audit['historical_cause_established'] is False,
        'D001 is observation provenance only')
    return {'id':PROFILE_ID,'helper_pins':deepcopy(HELPER_PINS),
        'evidence_pins':{str(D001_AUDIT.resolve()):D001_FILE_SHA},'D001_body_sha256':D001_BODY_SHA,
        'guard_queries':'exact original calls/result/error; false init bypass; nonrecursive Recorder',
        'additional_guards':0,'additional_CUDA_queries':0,'collector_math_changed':False,
        'phase_IO':'exclusive append-only per-marker fsync; counted final receipt after original preserve_terminal',
        'limits':{'maximum_markers':200000,'maximum_marker_bytes':65536,'maximum_ledger_bytes':64*1024**2},
        'duration_scope':'Includes marker IO and scheduling; not isolated OS service duration',
        'historical_cause_established':False,'native_fit_established':False,'native_launch_authorized':False}

def source_roster():
    paths=set(PACKAGE.glob('*.py'))|set((PACKAGE/'tests').glob('*.py'))
    paths.update(HERE/name for name in SOURCE_NAMES);paths.update(HERE.glob('test_*.py'))
    paths.update(PACKAGE/name for name in HELPER_PINS)
    paths.update([PACKAGE/'PILOT_DECISIONS.json',PACKAGE/'pilot_trainable_inventory.json',PACKAGE/'NATIVE_QUALIFICATION_PROPOSAL.json',
        PACKAGE/'AGENTS.md',ROOT/'AGENTS.md',ROOT/'RINEYA_SYNTH_CENTRALIZED_REASONING.md'])
    return {str(path.resolve()) for path in paths}

def source_pins():
    paths=source_roster();require(all(Path(p).is_file() and not Path(p).is_symlink() for p in paths),'complete own/transitive/source roster')
    return {p:native.filehash(Path(p)) for p in sorted(paths)}

def validate_scope(value):
    require(type(value) is dict and set(value)==BASE_FIELDS|{'short_phase_observation_profile'},'exact phase collector field roster')
    require(value['contract']==VERSION and value['authorization']==AUTHORIZATION,'new-only phase purpose/authority')
    require(value['short_phase_observation_profile']==phase_profile(),'exact independent observer profile/pins')
    require(value['source_pins']==source_pins(),'complete current executing source path/hash roster; omissions forbidden')
    require(value['resources']==native.RESOURCE_BOUNDS and value['tolerances']==native.TOLERANCES
        and value['runtime_versions']=={**native.RUNTIME,'tokenizers':'0.22.2'} and value['offline_environment']==native.OFFLINE_ENV,
        'unchanged resource/numeric/runtime/ENV policy')
    choice=value['batch_selector'];require(type(choice) is dict and set(choice)=={'source_kind','ordinal'}
        and choice['source_kind'] in ('atomic_RS_DEV','task_guard_DEV') and type(choice['ordinal']) is int
        and 1<=choice['ordinal']<=(6 if choice['source_kind']=='atomic_RS_DEV' else 12),'one existing fixed selector')
    rows=value['rows'];require(type(rows) is list and len(rows)==64,'whole64 denominator')
    checked=[base.validate_row(row) for row in rows]
    require(len({r['row_id'] for r in checked})==64 and value['source_packet_sha256']==digest(checked)
        and value['batch']['rows']==checked and value['batch']['source_packet_sha256']==value['source_packet_sha256'],'whole exact ordered packet')
    p=sum(32*r['prompt_tokens']+496 for r in checked);parity=2*checked[0]['prompt_tokens']
    require(value['dose']=={'context_cap':512,'maximum_generated_tokens':32,'rows':64,'maximum_capture_forwards':2048,
        'maximum_capture_input_positions':p,'maximum_parity_forwards':2,'maximum_parity_input_positions':parity,
        'maximum_forward_calls':2050,'maximum_forward_input_positions':p+parity,'optimizer_steps':0,'backwards':0},'unchanged full role dose')
    ram=base.successor_profile();require(value['short_RAM_monitor_profile']==ram,'exact unchanged RAM2 callback/monitor')
    prior=native.read_json(PACKAGE/'configs/SYNTH_NQ001_PROC7.json')['runtime_pins']
    require(value['runtime_pins']==base.merge_runtime_inventory(prior,ram),'exact inherited64 plus AST65')
    require(len(value['decoder_runtime_pins'])==17 and len(value['data_pins'])==5 and value['model_pins']==base.ASSET_PINS
        and Path(value['model_dir']).resolve()==base.MODEL_DIR.resolve()
        and value['seed']==101 and value['profile']==base.qualified.PROFILE
        and value['model_role']=='fresh_dense_M0_zero_B_only_no_STATE1_or_study_checkpoint','unchanged decoder17/data5/model7/M0 seed profile')
    require(all(value[k] is False for k in ('correctness_certified','retention_certified','training_authorized','scoring_authorized',
        'language_qualification_established','long_context_qualification_established','prior_correct_enrollment_performed',
        'protected_bank_access','paid_services','confirmation_eligible','quality_gain_established')),'all grading/training/retention authority closed')
    return value

def amend_scope(original):
    require(type(original) is dict and set(original)==BASE_FIELDS and original['contract']==base.VERSION
        and original['authorization']==base.AUTHORIZATION,'exact original collector scope before amendment')
    body=deepcopy(original);body['contract']=VERSION;body['authorization']=deepcopy(AUTHORIZATION)
    choice=body['batch_selector'];body['id']='SYNTH-SHORT-M0-PHASE3-%s-%02d'%(choice['source_kind'],choice['ordinal'])
    body['source_pins']=source_pins();body['short_phase_observation_profile']=phase_profile()
    return validate_scope(body)

def build_scope(kind,ordinal):return amend_scope(base.build_scope(kind,ordinal))

def write_scope(path,kind,ordinal):
    integrated();base._committed();target=Path(path).absolute()
    require(target.parent.resolve()==(PACKAGE/'configs').resolve() and target.suffix=='.json' and not target.exists()
        and not target.is_symlink(),'new write-once owned phase collector config')
    body=build_scope(kind,ordinal);body['sha256']=digest(body);raw=(canonical(body)+'\n').encode('ascii')
    require(len(raw)<=native.MAX_JSON,'bounded phase config')
    with target.open('xb',buffering=0) as stream:require(stream.write(raw)==len(raw),'complete phase config');stream.flush();os.fsync(stream.fileno())
    return body

def preflight(path):
    integrated();target=Path(path).absolute()
    require(target.parent.resolve()==(PACKAGE/'configs').resolve() and target.is_file() and not target.is_symlink(),'owned phase config')
    body=native.read_json(target);require(body['sha256']==digest({k:v for k,v in body.items() if k!='sha256'}),'phase config checksum')
    choice=body['batch_selector'];expected=build_scope(choice['source_kind'],choice['ordinal'])
    require(canonical({k:v for k,v in body.items() if k!='sha256'})==canonical(expected),'full current phase scope/source/row/dose/authority closure')
    return {**body,'config_path':str(target.resolve()),'config_sha256':native.filehash(target),'source_commit':base._committed(),
        'worker_path':str(PACKAGE/'short_phase_worker.py'),'bindings':{section:body[key] for section,key in
            (('source','source_pins'),('runtime','runtime_pins'),('decoder','decoder_runtime_pins'),('data','data_pins'),('model','model_pins'))}}

def validate_inventory(value,scope,*,expected_profile_sha,expected_config_sha,expected_commit):
    """Pure admission; caller anchors are external trust, not self-signature authority."""
    validate_scope(scope)
    require(value['id']==REGISTRATION_ID and value['sha256']==digest({k:v for k,v in value.items() if k!='sha256'}),'new-only frozen inventory body')
    require(value['contract']==VERSION and value['source_commit']==expected_commit and value['config_file_sha256']==expected_config_sha
        and value['profile_body_sha256']==expected_profile_sha==digest(scope['short_phase_observation_profile']),
        'externally selected commit/config/profile anchors')
    sections={'source':'source_pins','runtime':'runtime_pins','decoder':'decoder_runtime_pins','data':'data_pins','model':'model_pins'}
    require(set(value['inventories'])==set(value['counts'])==set(sections),'exact inventory sections')
    for section,key in sections.items():require(value['inventories'][section]==scope[key] and value['counts'][section]==len(scope[key]),'every frozen path/hash/count '+section)
    require(value['counts']['runtime']==65 and value['counts']['decoder']==17 and value['counts']['data']==5 and value['counts']['model']==7,
        'required runtime65/decoder17/data5/model7')
    require(set(scope['source_pins'])==source_roster(),'source inventory cannot omit executing dependency')
    return True
