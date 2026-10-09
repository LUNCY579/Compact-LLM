"""Independent original-byte post-Job STUDY chunk consumer; no scorer.

Staging rejects native audit before receipts/APIs. Authored fixtures exercise
the same consumer with explicit schedule/host suppliers and cannot mint PASS.
"""
from copy import deepcopy
from datetime import datetime,timezone
from pathlib import Path
from types import ModuleType,SimpleNamespace
import ast,hashlib,json,math,sys,time

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
EXPECTED=PACKAGE/'study_terminal_audit_v1'
SCOPE='REGISTERED_STUDY_ONE_FRESH_NINE_UPDATE_CHUNK_V1'
AUDITOR_SOURCE_SHA=''
def require(ok,message):
    if not ok:raise ValueError(message)
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(value):return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def seal(value):return {**value,'sha256':digest(value)}
def pairs(items):
    out={}
    for k,v in items:require(k not in out,'duplicate audit JSON field');out[k]=v
    return out
def bad(value):raise ValueError('nonfinite audit JSON')
def info(path):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'original regular audited file');return {'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
def read_json(path,limit=16*1024**2):
    path=Path(path);require(path.is_file() and not path.is_symlink(),'original selected JSON')
    with path.open('rb') as stream:raw=stream.read(limit+1)
    require(len(raw)<=limit,'bounded original JSON');return json.loads(raw,object_pairs_hook=pairs,parse_constant=bad)
def read_ref(ref):
    require(type(ref) is dict and set(ref)=={'path','bytes','sha256'},'exact external original byte descriptor')
    path=Path(ref['path']);require(path.is_absolute() and info(path)=={k:ref[k] for k in ('bytes','sha256')},'external selected original bytes');return read_json(path)
def checksum(value):require(type(value) is dict and value.get('sha256')==digest({k:v for k,v in value.items() if k!='sha256'}),'canonical whole worker/audit body checksum')
def config_fields_from_engine(ref):
    """Read the selected engine's literal schema; do not import data or helpers."""
    path=Path(ref['path']);require(path.suffix=='.py' and info(path)=={k:ref[k] for k in ('bytes','sha256')},'exact selected scientific engine source')
    tree=ast.parse(path.read_bytes());nodes=[n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='FIELDS' for t in n.targets)]
    require(len(nodes)==1,'one original selected engine FIELDS declaration');fields=ast.literal_eval(nodes[0].value)
    require(type(fields) is set and all(type(x) is str for x in fields),'literal exact engine config fields');return fields
def validate_config_shape(config,engine_ref):
    require(type(config) is dict and set(config)==config_fields_from_engine(engine_ref) and config.get('kind')=='REGISTERED_STUDY_TRAINING_CHUNK_V1' and config.get('scope')==SCOPE,'exact selected engine config FIELDS/kind/scope')
    require(config.get('arm') in ('M1','M2') and type(config.get('chunk')) is int and config['chunk'] in (1,2,3),'exact adapted arm and typed three-chunk ordinal');checksum(config)
def safe_child(path,leaf):
    path=Path(path);require(path.is_absolute() and path.parent==Path(leaf) and path.resolve()==path and path.is_file() and not path.is_symlink(),'exact original owned evidence leaf');return path
def _cap_factory():
    token=object()
    def get():return token
    def check(value):require(value is token,'private authored terminal audit capability')
    return get,check
_CPU_capability,_check_cap=_cap_factory()

def original_validators(chunks,lease_root,source_sha256):
    """Retain original resource/event logic, with three purpose/path literals."""
    path=PACKAGE/'auditor.py';raw=path.read_bytes();require(hashlib.sha256(raw).hexdigest()==source_sha256,'externally selected original auditor source')
    tree=ast.parse(raw);events=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='validate_events'));resources=deepcopy(next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='verify_resources'));changes={'event':0,'ownership':0,'lease':0}
    for node in ast.walk(events):
        if isinstance(node,ast.Constant) and node.value=='qualification_measurement_preserved':node.value='study_publication_preserved';changes['event']+=1
    for node in ast.walk(resources):
        if isinstance(node,ast.Constant) and node.value=='stage_worker_ownership_verified.json':node.value='stage_study_worker_ownership_verified.json';changes['ownership']+=1
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Div) and isinstance(node.left,ast.Name) and node.left.id=='PACKAGE' and isinstance(node.right,ast.Constant) and node.right.value=='artifacts/_native_gpu_leases':
            node.left.id='_external_lease_root_parent';node.right.value='_native_gpu_leases';changes['lease']+=1
    require(changes=={'event':1,'ownership':1,'lease':1},'exact original purpose/location adaptation only')
    native=SimpleNamespace(_pairs=pairs,_bad_constant=bad);namespace={'Path':Path,'PACKAGE':PACKAGE,'_external_lease_root_parent':Path(lease_root).parent,'datetime':datetime,'json':json,'math':math,'deepcopy':deepcopy,'require':require,'read_json':read_json,'file_info':info,'safe_child':safe_child,'MAX_EVENTS_BYTES':32*1024**2,'chunks':chunks,'contract':SimpleNamespace(native=native)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[events,resources],type_ignores=[])),str(path),'exec'),namespace)
    return namespace,changes

def publication_chain(protocol,resume,events,config,leaf):
    """Join repeated retained events by exact ordinal/order, never name collapse."""
    expected=[];archives=[]
    if config.get('retained_update_policy') is not None:
        retained=resume['envelope'].get('retained_updates')
        require(type(retained) is list and len(retained)==config['chunk']*9 and
                protocol.get('retained_updates')==retained,'same cumulative producer/envelope all-update list')
        require(len({entry['record']['backend_descriptor']['path'] for entry in retained})==len(retained),
                'distinct original retained archive for every completed update')
        start=(config['chunk']-1)*9
        for ordinal,entry in enumerate(retained[start:],start+1):
            record=entry['record'];require(record['binding']['completed_updates']==ordinal,'current retained publication exact update ordinal')
            archive=record['backend_descriptor'];safe_child(archive['path'],leaf)
            require(info(archive['path'])=={k:archive[k] for k in ('bytes','sha256')},'current retained archive original bytes')
            archives.append(archive)
            expected.append(('retained_update_publish',entry['publication'],record))
            if ordinal in (9,27):expected.append(('GD_export_parity_request',entry['GD_export_parity_request_publication'],record))
    expected.extend((event,resume[key],payload) for event,key,payload in (
        ('checkpoint_publish','publication',resume['envelope']),
        ('terminal_costs_publish','terminal_publication',resume['durable_terminal']),
        ('checkpoint_promote','promotion_publication',resume['promotion'])))
    actual=events['publications']
    require(type(actual) is list and len(actual)==len(expected),'exact retained/GD/scientific publication cardinality')
    for actual,(name,publication,payload) in zip(actual,expected):
        require(actual['name']==name and actual['publication']==publication and actual['payload']==payload,
                'ordered original retained/GD/scientific publication exact byte/value joins')
    return archives

def birth_union(sup,resources):
    rows=[sup['worker_identity']]
    for sample in resources:
        if sample['kind']=='CPU':rows.extend(sample['owned_process_identities'])
    result={};pids={}
    for row in rows:
        birth={k:row[k] for k in ('pid','creation_filetime_100ns')};require(all(type(v) is int and 0<v<2**64 for v in birth.values()),'strict actual launcher/interpreter birth pairs')
        require(birth['pid'] not in pids or pids[birth['pid']]==birth['creation_filetime_100ns'],'conflicting exact birth for owned PID');pids[birth['pid']]=birth['creation_filetime_100ns'];result[tuple(birth.values())]=birth
    require(0<len(result)<=4096,'bounded complete observed owned births');return list(result.values())

def consume(packet,chunks,host,*,fixture_only):
    require(type(packet) is dict and packet.get('fixture_only') is fixture_only,'explicit fixture/native terminal packet separation')
    leaf=Path(packet['output_directory']);require(leaf.is_absolute() and leaf.is_dir() and not leaf.is_symlink(),'exact selected terminal output leaf')
    config=read_ref(packet['config']);approval=read_ref(packet['approval']);review=read_ref(packet['review'])
    require(config.get('arm') in ('M1','M2') and type(config.get('chunk')) is int and config['chunk'] in (1,2,3),'exact adapted arm and typed three-chunk ordinal')
    require(config['scope']==SCOPE and config['fixture_only'] is fixture_only and approval.get('approved') is True and approval.get('scope')==SCOPE and approval.get('fixture_only') is fixture_only and approval.get('source_pins')==config['source_pins'] and review.get('accepted') is True and review.get('source_pins')==config['source_pins'] and review.get('approval_file_sha256')==packet['approval']['sha256'],'original config/approval/independent review/source binding')
    for path,ref in packet['sources'].items():require(ref['path']==path and info(path)=={k:ref[k] for k in ('bytes','sha256')} and config['source_pins'].get(path)==ref['sha256'],'current selected source bytes still original')
    if not fixture_only:
        require(packet['engine_source']['path']==str(PACKAGE/'study_training_v1/study_training.py') and packet['engine_source']['path'] in packet['sources'] and packet['sources'][packet['engine_source']['path']]==packet['engine_source'],'selected original installed engine belongs to approved source closure')
        validate_config_shape(config,packet['engine_source'])
        approval_fields={'kind','scope','fixture_only','approved','permissions','arm','chunk','source_pins','training_registration_value_sha256','baseline_id','owned_boundary_source','owned_boundary_name','model_directory','output_directory','previous_resume_value_sha256','human_evidence_reference','retained_update_policy'}
        review_fields={'kind','scope','fixture_only','accepted','source_pins','approval_file_sha256','training_registration_value_sha256','training_qualification_check','m0_scoring_check','guard_enrollment_check','baseline_id','retained_update_policy'}
        require(set(approval)==approval_fields and set(review)==review_fields and config.get('kind')=='REGISTERED_STUDY_TRAINING_CHUNK_V1','exact original native config/approval/review shapes')
        for key in ('approval','review','training_registration'):
            require(type(config[key]) is dict and {k:config[key][k] for k in ('path','bytes','sha256')}==packet[key] and config[key]['value_sha256']==digest({'approval':approval,'review':review}.get(key,read_ref(packet['training_registration']))),'original configured physical descriptor and value joins')
        required={str(PACKAGE/name) for name in ('native_process.py','study_native_chunk.py','study_schedule.py','auditor.py')}|{str(Path(__file__).resolve())}
        require(required<=set(packet['sources']) and packet['sources'][str(PACKAGE/'native_process.py')]==packet['host_source'],'complete exact original host/scientific/auditor source closure before APIs')
        require(approval.get('kind')=='OWNER_STUDY_TRAINING_CHUNK_AUTHORIZATION_V1' and approval.get('permissions')=={'training':True,'model':True,'optimizer':True,'scoring':False,'protected_bank_access':False,'paid_services':False,'promotion':False} and all(approval.get(k)==config[k] for k in ('arm','chunk','baseline_id','owned_boundary_source','owned_boundary_name','model_directory','output_directory','retained_update_policy')) and review.get('kind')=='INDEPENDENT_STUDY_TRAINING_CHUNK_REVIEW_V1' and review.get('scope')==SCOPE and review.get('fixture_only') is False,'exact selected original native STUDY permission/review')
        require(approval['training_registration_value_sha256']==review['training_registration_value_sha256']==config['training_registration']['value_sha256'] and review['baseline_id']==config['baseline_id'] and all(review[k]==config[k] for k in ('training_qualification_check','m0_scoring_check','guard_enrollment_check','retained_update_policy')) and approval['previous_resume_value_sha256']==(None if config['previous_resume'] is None else config['previous_resume']['value_sha256']) and type(approval['human_evidence_reference']) is str and bool(approval['human_evidence_reference']),'same baseline/source-qualified native review and original prior checkpoint authority')
    if not fixture_only:chunks.validate_producer_policy(config['retained_update_policy'])
    selected={packet[k]['path']:info(packet[k]['path']) for k in ('config','approval','review','training_registration','latest_terminal_costs')}
    selected.update({path:info(path) for path in packet['sources']})
    for key in ('previous_resume','previous_latest_terminal_costs'):
        if key in packet:selected[packet[key]['path']]=info(packet[key]['path'])
    worker=read_ref(packet['worker']);sup=read_ref(packet['supervisor']);checksum(worker)
    require(Path(packet['worker']['path'])==leaf/'worker_result.json' and Path(packet['supervisor']['path'])==leaf/'SUPERVISOR_RECEIPT.json','selected producer/supervisor filenames')
    require(sup.get('status')=='SUPERVISED_WORKER_EXITED_ZERO' and type(sup.get('returncode')) is int and sup['returncode']==0 and sup.get('partial_outputs_frozen') is True,'successful original complete frozen supervisor; no partial/exit-only evidence')
    require(worker.get('fixture_only') is fixture_only and worker.get('worker_scope')==SCOPE and worker.get('config_sha256')==sup.get('config_sha256')==packet['config']['sha256'] and worker.get('native_PASS') is worker.get('scoring_authorized') is worker.get('quality_gain_established') is False,'exact same scoped worker/config closed claims')
    protocol=worker.get('protocol_result');require(type(protocol) is dict and protocol.get('status')=='COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED' and protocol.get('arm')==config['arm'] and protocol.get('chunk_ordinal')==config['chunk'] and protocol.get('prevalidated_whole_occurrences')==180 and protocol.get('completed_updates')==config['chunk']*9,'successful complete original nine/180 scientific producer')
    require(not any(protocol.get(k) for k in ('error','error_type','cleanup_error')),'original scientific/cleanup error cannot gain terminal acceptance')
    registration=read_ref(packet['training_registration']);recipe=chunks.recipe(registration,config['arm'],chunks.NATIVE_SCOPE,retained_update_policy=config.get('retained_update_policy'))
    require(protocol['recipe']==recipe,'same original scientific source/recipe/arm')
    resume=protocol['resume_descriptor'];latest=read_ref(packet['latest_terminal_costs']);require(latest==protocol['costs']==resume['terminal_costs'],'authoritative unchanged final terminal costs')
    chunks._validate_resume(resume,registration,config['arm'],config['chunk']+1,recipe,latest)
    require(worker['terminal_audit'].get('kind')=='WORKER_STUDY_PENDING_POST_JOB_TERMINAL_AUDIT' and worker['terminal_audit'].get('owned_cleanup_returned') is True and worker['terminal_audit'].get('native_PASS') is False,'worker pending evidence cannot claim post-Job death')
    validators,adaptation=original_validators(chunks,packet['lease_root'],packet['original_auditor_sha256']);events=validators['validate_events'](leaf/'events.jsonl',leaf)
    journal=worker['terminal_audit']['operation_accounting'];require(journal['operations']==events['counts'] and journal['last_call']==events['API_calls'],'independent exact complete original journal accounting')
    scientific=deepcopy(latest['primitive_operations']);prior={}
    if config['chunk']>1:
        previous=read_ref(packet['previous_resume']);prior_latest=read_ref(packet['previous_latest_terminal_costs'])
        require({k:config['previous_resume'][k] for k in ('path','bytes','sha256')}==packet['previous_resume'] and {k:config['latest_terminal_costs'][k] for k in ('path','bytes','sha256')}==packet['previous_latest_terminal_costs'],'same original configured prior STUDY references')
        chunks._validate_resume(previous,registration,config['arm'],config['chunk'],recipe,prior_latest);prior=prior_latest['primitive_operations']
    else:require(config.get('previous_resume') is config.get('latest_terminal_costs') is None,'fresh STUDYchunk1 has no prior cost supplier')
    current={}
    for name,total in scientific.items():
        before=prior.get(name,{k:0 for k in total});current[name]={k:total[k]-before[k] for k in total};require(all(v>=0 for v in current[name].values()),'scientific cumulative cost rollback')
    current={name:counts for name,counts in current.items() if any(counts.values())}
    cleanup_delta={}
    for name,counts in events['counts'].items():
        base=current.get(name,{k:0 for k in counts});delta={k:counts[k]-base[k] for k in counts};require(all(v>=0 for v in delta.values()),'journal cannot omit scientific primitive returns')
        if any(delta.values()):
            require(name=='checkpoint_remove_attributes' and delta=={'attempted':1,'returned':1,'failed':0,'pending':0},'only original exact single post-chunk checkpoint attribute removal');cleanup_delta[name]=delta
    require(set(current)<=set(events['counts']),'every scientific primitive occurs in current original journal')
    retained_archives=publication_chain(protocol,resume,events,config,leaf)
    resources=validators['verify_resources'](config,sup,leaf)
    raw=packet['raw_files'];require(type(raw) is dict and raw and len(raw)<=4096,'externally selected complete raw original inventory')
    listed=sup['partial_outputs'];require(len(listed)==len({r['path'] for r in listed}) and all(r.get('frozen') is not False for r in listed),'original frozen raw inventory cardinality')
    for item in listed:
        rel=item['path'];path=safe_child(str(leaf/rel),leaf);require(info(path)=={k:item[k] for k in ('bytes','sha256')} and raw.get(rel)==info(path),'every original SUP-listed raw file byte anchor')
    require(set(raw)=={r['path'] for r in listed}|{'SUPERVISOR_RECEIPT.json'},'raw roster contains exact original inventory plus terminal SUP')
    require(raw['SUPERVISOR_RECEIPT.json']==info(leaf/'SUPERVISOR_RECEIPT.json'),'raw terminal supervisor bytes')
    for archive in retained_archives:require(raw.get(Path(archive['path']).name)=={k:archive[k] for k in ('bytes','sha256')},'every current retained archive independently selected in raw inventory')
    require({p.name for p in leaf.iterdir()}==set(raw) and all(p.is_file() and not p.is_symlink() for p in leaf.iterdir()),'complete actual original leaf roster; no omitted files')
    lease=Path(sup['lease_path']);require({p.name for p in lease.iterdir()}=={'reservation.json','worker.json','resume.json','terminal.json'},'exact original four-file lease roster')
    lease_before={p.name:info(p) for p in lease.iterdir()}
    births=birth_union(sup,read_json(leaf/'RESOURCE_OBSERVATIONS.json'));ownership=read_json(leaf/'stage_study_worker_ownership_verified.json');require(ownership['actual_worker_pid'] in {b['pid'] for b in births},'actual interpreter birth independently sampled')
    require(type(host) is not dict and callable(getattr(host,'is_alive',None)),'actual externally source-bound host interface, not Boolean map')
    observed=[];started=time.monotonic()
    for birth in births:
        alive=host.is_alive(deepcopy(birth));require(type(alive) is bool and alive is False,'fresh actual original birth still live/unknown; no terminal acceptance');observed.append({'birth':birth,'alive':alive})
    finished=time.monotonic();require(finished-started<=30,'bounded fresh post-Job observations')
    for rel,expected in raw.items():require(info(leaf/rel)==expected,'original raw bytes stable through final death observations')
    for path,expected in selected.items():require(info(path)==expected,'selected original config/approval/science/source bytes stable through host observations')
    require({p.name:info(p) for p in lease.iterdir()}==lease_before,'original lease bytes and roster stable through host observations')
    require({p.name for p in leaf.iterdir()}==set(raw) and all(p.is_file() and not p.is_symlink() for p in leaf.iterdir()),'original raw roster stable through host observations')
    return seal({'kind':'INDEPENDENT_POST_JOB_STUDY_CHUNK_BYTE_AND_PROTOCOL_AUDIT','arm':config['arm'],'chunk':config['chunk'],'config_file_sha256':packet['config']['sha256'],'worker_file_sha256':packet['worker']['sha256'],'supervisor_file_sha256':packet['supervisor']['sha256'],'resume_descriptor_sha256':resume['sha256'],'terminal_costs_sha256':digest(latest),'resource_summary':resources,'original_validator_adaptations':adaptation,'post_chunk_cleanup_delta':cleanup_delta,'observed_birth_deaths':observed,'fixture_only':fixture_only,'native_PASS':False,'qualified_resume_authority_issued':False,'scoring_authorized':False,'quality_gain_established':False})

def audit_registered(packet_ref):
    require(HERE==EXPECTED,'staged post-Job STUDY auditor has no native authority')
    packet=read_ref(packet_ref);require(packet.get('fixture_only') is False,'authored terminal packet cannot enter native audit')
    require(packet['host_source']['path']==str(PACKAGE/'native_process.py') and info(packet['host_source']['path'])=={k:packet['host_source'][k] for k in ('bytes','sha256')},'exact externally selected native host sampler source')
    sys.path.insert(0,str(PACKAGE));import study_native_chunk as chunks;import native_process
    require(Path(native_process.__file__).resolve()==PACKAGE/'native_process.py' and Path(chunks.__file__).resolve()==PACKAGE/'study_native_chunk.py','original scientific/host source origins')
    class DeferredSampler:
        def __init__(self):self.sampler=None
        def is_alive(self,birth):
            if self.sampler is None:self.sampler=native_process.Sampler()
            return self.sampler.is_alive(birth)
    return consume(packet,chunks,DeferredSampler(),fixture_only=False)
def _CPU_consume(packet,chunks,host,*,capability):
    _check_cap(capability);return consume(packet,chunks,host,fixture_only=True)
