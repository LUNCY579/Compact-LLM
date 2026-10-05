"""New-only Phase3 raw audit adaptation of exact preserved RAM2 source.

No native model/framework/OS work. Expensive decoder and byte verification are
inside the inherited terminal audit, unavailable until all root prerequisites.
"""
from pathlib import Path
from copy import deepcopy
import ast,hashlib,importlib.util,json,sys,math,os,subprocess,types
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
sys.path.insert(0,str(HERE))
import phase_replay as phase
SIGNAL='ROOT_TERMINAL_AND_PRESERVED'
PIN='d69810dccd11a32595dace73a788c8868efabb2a09f593fd33316e92dcb47442'
DEPENDENCY=HERE/'pinned_dependencies/RAM2_auditor_d69810.py'
VERSION='synth-short-M0-phase-observed-staged-3'
PROFILE_ID='SHORT_M0_PHASE_OBSERVED_PHYSICAL_RAM_V3'
REGISTRATION_ID='SYNTH_SHORT_PHASE3_INDEPENDENT_AUDIT_INVENTORY_001'
AUTHORIZATION={'scope':'short_M0_phase_observed_raw_collection_one_fixed_batch_only',
    'raw_DEV_collection':True,'owner_authority':'explicit_owner_goal_and_frozen_run_manifest','production_training':False,
    'scoring':False,'prior_correct_enrollment':False,'language':False,'long_context':False,'protected_bank_access':False,'paid_services':False}
# Every modification is an exact source-site adaptation. No general purpose,
# arbitrary count or permission substitution is permitted.
REPLACEMENTS=(
 ('package=HERE.parents[2]','package=HERE.parents[1]'),
 ('import short_collection_contract as contract','import short_phase_contract as contract'),
 ('CONTRACT_VERSION="synth-short-M0-physical-RAM-coherent-monitor-staged-2"','CONTRACT_VERSION="'+VERSION+'"'),
 ('require(HERE.name=="audit_work" and HERE.parent.name=="short_ram_successor_work" and HERE.parent.parent.name=="artifacts","stage-only new-profile auditor")',
  'require(HERE.name=="short_phase_audit_work" and HERE.parent.name=="artifacts","stage-only Phase3 adapter")'),
 ('"short_RAM2_audit_integrated_contract",package/\'short_collection_contract.py\'','"short_phase3_audit_integrated_contract",package/\'short_phase_contract.py\''),
 ('_early_terminal_checks(scope,worker,sup,pres,expected_selector,check)','_phase3_early(scope,worker,sup,pres,expected_selector,check,leaf)'),
 ('current_profile=contract.successor_profile()','current_profile=contract.phase_profile()'),
 ('counts=_registered_admission(scope,sup,registration,expected_profile_body_sha256,current_profile,current_sources,check)',
  'counts=_phase3_admission(scope,sup,registration,expected_profile_body_sha256,current_profile,current_sources,check)'),
 ('len(raw_names)==capture_n+77','len(raw_names)==capture_n+79'),
 ('len(sup[\'partial_outputs\'])==capture_n+76','len(sup[\'partial_outputs\'])==capture_n+78'),
 ('contract.merge_runtime_inventory(contract.native.runtime_files(),current_profile)',
  'contract.base.merge_runtime_inventory(contract.native.runtime_files(),scope["short_RAM_monitor_profile"])'),
 ('contract._tokenizer_runtime()','contract.base._tokenizer_runtime()'),
 ('package/\'short_collection_worker.py\'','package/\'short_phase_worker.py\''),
 ('contract.INFERENCE_AUDIT','contract.base.INFERENCE_AUDIT'))

def sha(raw):return hashlib.sha256(raw).hexdigest()
def prepared_source(raw=None):
    raw=DEPENDENCY.read_bytes() if raw is None else raw
    phase.require(sha(raw)==PIN,'exact preserved RAM2 auditor source dependency')
    text=raw.decode('utf-8');changes=[]
    for old,new in REPLACEMENTS:
        phase.require(text.count(old)==1,'exact single registered adaptation site: '+old)
        text=text.replace(old,new,1);changes.append({'original':old,'replacement':new})
    ast.parse(text)
    return text,changes

def _phase3_admission(scope,sup,registration,expected_profile,current_profile,current_sources,check):
    check(registration.get('id')==REGISTRATION_ID and registration.get('schema_version')==1 and registration.get('contract')==VERSION,'new-only Phase3 inventory')
    check(registration.get('sha256')==phase.digest({k:v for k,v in registration.items() if k!='sha256'}),'inventory checksum')
    profile=scope.get('short_phase_observation_profile')
    check(profile==current_profile and profile.get('id')==PROFILE_ID and phase.digest(profile)==expected_profile==registration.get('profile_body_sha256'),
        'externally pinned exact Phase3 profile')
    check(scope.get('contract')==VERSION and registration.get('source_commit')==sup['preflight']['source_commit'],'exact source contract/commit')
    sections={'source':'source_pins','runtime':'runtime_pins','decoder':'decoder_runtime_pins','data':'data_pins','model':'model_pins'}
    check(type(registration.get('inventories')) is dict and type(registration.get('counts')) is dict
        and set(registration['inventories'])==set(registration['counts'])==set(sections)
        and all(type(n) is int and n>0 for n in registration['counts'].values()),'exact typed frozen inventory sections')
    for section,key in sections.items():
        check(type(registration['inventories'][section]) is dict and all(type(p) is str and hash_pin(h) for p,h in registration['inventories'][section].items())
            and registration['inventories'][section]==scope[key] and len(scope[key])==registration['counts'][section],
            'every externally frozen path/hash/count '+section)
    check({k:registration['counts'][k] for k in ('runtime','decoder','data','model')}=={'runtime':65,'decoder':17,'data':5,'model':7},'unchanged mandatory inventories')
    for path,pin in current_sources.items():check(scope['source_pins'].get(path)==pin,'actual integrated/helper source '+path)
    check(scope['short_RAM_monitor_profile']['id']=='SHORT_M0_PHYSICAL_RAM_COHERENT_EXTERNAL_CHECK_V2'
        and scope['short_RAM_monitor_profile']['all_floors_and_numerics_unchanged'] is True,'unchanged RAM2 foundation')
    extension=scope['short_RAM_monitor_profile']['runtime_extension_pins']
    check(type(extension) is dict and len(extension)==1 and all(Path(p).name=='ast.py' and hash_pin(h)
        and scope['runtime_pins'].get(p)==h for p,h in extension.items()),'exact inherited runtime AST extension')
    core={p:h for p,h in scope['runtime_pins'].items() if p not in extension}
    check(len(core)==64 and phase.digest(core)==registration.get('qualified_runtime64_sha256'),'unchanged externally frozen qualified runtime64 core')
    check(sup.get('short_RAM_monitor_diagnostic')=={'path':'SHORT_RAM_ABORT_DIAGNOSTIC.json','attempted':False,'persisted':False,
        'write_error':None,'original_abort_reason_preserved':None},'successful terminal cannot carry abort contradiction')
    return registration['counts']

def _phase3_early(scope,worker,sup,pres,selector,check,leaf,*,fixture=False):
    check(scope.get('contract')==VERSION and scope['authorization']==AUTHORIZATION
        and scope.get('profile')=='BF16_RESIDENT_FP32_ACTIVATIONS_QV8_V1',
        'exact new-only raw collection purpose')
    check(scope['batch_selector']=={'source_kind':selector[0],'ordinal':selector[1]},'independent fixed selector')
    check(sup['status']=='SUPERVISED_WORKER_EXITED_ZERO' and sup['returncode']==0 and sup['partial_outputs_frozen'] is True,'successful frozen owned terminal')
    check(worker['status']=='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED' and worker['error'] is None,'complete raw worker')
    rows=scope['rows'];outputs=worker['collection']['rows']
    check(len(rows)==len(outputs)==64 and len({r['row_id'] for r in rows})==64
        and [r['row_id'] for r in rows]==[r['row_id'] for r in outputs],'whole unique ordered64 before expensive reads')
    check(all(type(r) is dict and set(r)=={'row_id','prompt_sha256','prompt_ids','prompt_tokens','attention_mask'} for r in rows),'prompt-only exact early row fields')
    check(phase.digest(rows)==scope['source_packet_sha256']==worker['collection']['source_packet_sha256'],'exact whole packet')
    check(worker['source_commit']==pres['source_commit']==sup['preflight']['source_commit'],'source barrier')
    markers=phase.read_lines(leaf/'PHASE_OBSERVATIONS.jsonl',canonical_lines=True)
    summary=phase.read_json(leaf/'PHASE_ACCOUNTING_FINAL.json')
    events=phase.read_lines(leaf/'events.jsonl')
    identity=summary['identity']
    check(identity=={'execution_scope':'CPU_FIXTURE_PHASE_COLLECTION_PROTOCOL' if fixture else 'OWNED_SHORT_M0_PHASE_OBSERVED_RAW_COLLECTION','purpose':scope['authorization']['scope'],
        'config_sha256':worker['config_sha256'],'source_commit':worker['source_commit'],'batch_selector':scope['batch_selector']},'exact external raw/final identity')
    proof=phase.replay(markers,summary,worker,events,fixture=fixture,scope=scope,terminal=phase.read_json(leaf/'worker_GPU_final.json'))
    check(proof['native_raw_audit_complete'] is False,'phase proof is prerequisite, not full native authority')
    return proof

def hash_pin(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)

def current_head():
    root=PACKAGE.parents[1]
    clean=subprocess.run(['git','status','--porcelain'],cwd=root,capture_output=True,text=True,check=True).stdout
    phase.require(not clean.strip(),'clean current committed source required')
    return subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True).stdout.strip()

def _source_admission(registration):
    sources=registration['inventories']['source']
    required=set(PACKAGE.glob('*.py'))|set((PACKAGE/'tests').glob('*.py'))
    required|={PACKAGE/n for n in ('PILOT_DECISIONS.json','pilot_trainable_inventory.json','NATIVE_QUALIFICATION_PROPOSAL.json','AGENTS.md')}
    required|={PACKAGE.parents[1]/n for n in ('AGENTS.md','RINEYA_SYNTH_CENTRALIZED_REASONING.md')}
    phase.require(set(sources)=={str(p.resolve()) for p in required},'complete exact executing source roster before imports')
    for path,pin in sources.items():phase.require(hash_pin(pin) and phase.file_info(path)['sha256']==pin,'current externally pinned source before imports '+path)
    dependencies={str(p.resolve()):phase.file_info(p)['sha256'] for p in
        (HERE/'audit_adapter.py',HERE/'phase_replay.py',HERE/'test_phase_audit.py',DEPENDENCY,HERE/'pinned_dependencies/PRESERVATION.json')}
    phase.require(registration.get('auditor_dependency_pins')==dependencies,'exact independently registered auditor dependency closure')
    for name,module in tuple(sys.modules.items()):
        expected=PACKAGE/(name+'.py')
        if expected.is_file() and name in sys.modules:
            phase.require(getattr(module,'__file__',None) is not None and Path(module.__file__).resolve()==expected.resolve(),
                'no conflicting cached package module '+name)
    return sources

def prerequisites(*,terminal_signal,inventory_path,inventory_sha,profile_sha,config_sha,preservation_sha,source_commit):
    phase.require(terminal_signal==SIGNAL,'root terminal+preserved signal before any heavy reads')
    for value in (inventory_sha,profile_sha,config_sha,preservation_sha):
        phase.require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'independently supplied exact anchors')
    phase.require(type(source_commit) is str and len(source_commit)==40 and all(c in '0123456789abcdef' for c in source_commit),'independent exact launch HEAD')
    integrated=PACKAGE/'short_phase_contract.py'
    phase.require(integrated.is_file() and not integrated.is_symlink(),'integrated Phase3 dependency absent; native audit closed')
    path=Path(inventory_path).absolute();phase.require(path.is_file() and not path.is_symlink()
        and path.resolve().parent==(PACKAGE/'artifacts/short_m0_audit_registrations').resolve(),'owned independent inventory file')
    raw=phase.read_bytes(path);phase.require(sha(raw)==inventory_sha,'external inventory file SHA')
    registration=phase.loads(raw)
    phase.require(registration.get('sha256')==phase.digest({k:v for k,v in registration.items() if k!='sha256'}),'inventory canonical body anchor')
    phase.require(registration['source_commit']==source_commit and registration['config_file_sha256']==config_sha
        and registration['profile_body_sha256']==profile_sha,'independent launch/profile/config anchors')
    phase.require(registration.get('root_path')==str(PACKAGE.parents[1].resolve())
        and registration.get('package_path')==str(PACKAGE.resolve()),'exact external root/package registration')
    _source_admission(registration)
    phase.require(current_head()==source_commit,'actual clean current HEAD equals independent launch HEAD')
    return registration

def audit_terminal(config_path,raw_directory,preserved_directory,*,expected_selector,expected_config_file_sha256,
    expected_preservation_body_sha256,inventory_registration_path,expected_inventory_registration_file_sha256,
    expected_profile_body_sha256,expected_source_commit,terminal_preservation_signal=None,output_path=None):
    registration=prerequisites(terminal_signal=terminal_preservation_signal,inventory_path=inventory_registration_path,
        inventory_sha=expected_inventory_registration_file_sha256,profile_sha=expected_profile_body_sha256,
        config_sha=expected_config_file_sha256,preservation_sha=expected_preservation_body_sha256,source_commit=expected_source_commit)
    context=_terminal_context(config_path,raw_directory,preserved_directory,registration,expected_selector,
        expected_config_file_sha256,expected_preservation_body_sha256,expected_profile_body_sha256,expected_source_commit)
    source,changes=prepared_source();inherited=_inherited_module(source)
    result=inherited.audit_terminal(config_path,raw_directory,preserved_directory,expected_selector=expected_selector,
        expected_config_file_sha256=expected_config_file_sha256,expected_preservation_body_sha256=expected_preservation_body_sha256,
        inventory_registration_path=inventory_registration_path,expected_inventory_registration_file_sha256=expected_inventory_registration_file_sha256,
        expected_profile_body_sha256=expected_profile_body_sha256,terminal_preservation_signal=SIGNAL,output_path=None)
    # The inherited closure verifies all actual model byte hashes twice. Only
    # now inspect bounded header metadata to replay the exact first-copy order.
    profile,header,assets=_verified_loader_evidence(context)
    return _finish_composition(context,result,source,changes,profile,header,assets,registration,expected_source_commit,
        output_path=output_path,raw_directory=raw_directory)

def _verified_loader_evidence(context):
    scope=context['scope'];assets={name:phase.file_info(Path(scope['model_dir'])/name) for name in scope['model_pins']}
    profile=phase.loader_source(phase.read_bytes(PACKAGE/'native_loader.py',128*1024))
    return profile,phase.read_header(Path(scope['model_dir'])/'model.safetensors'),assets

def _finish_composition(context,result,source,changes,profile,header,assets,registration,expected_source_commit,
    *,output_path=None,raw_directory=None,fixture=False):
    proof=phase.loader_tree(context['markers'],context['worker']['loader'],profile,header,assets)
    phase.require(current_head()==expected_source_commit,'same clean actual HEAD after full closure')
    for path,pin in registration['inventories']['source'].items():phase.require(phase.file_info(path)['sha256']==pin,'source stable after full closure')
    result.update(phase3_independent_phase_replay=context['phase_proof'],phase3_independent_loader_replay=proof,
        original_RAM2_dependency_sha256=PIN,transparent_exact_source_adaptations=changes,
        adapted_auditor_source_sha256=sha(source.encode('utf-8')),full_phase_raw_model_resource_audit_complete=True)
    if fixture:
        phase.require(output_path is None,'CPU fixture cannot write native audit receipt')
        result.update(status='PASS_CPU_FIXTURE_TERMINAL_COMPOSITION_ONLY',fixture_only=True,
            full_phase_raw_model_resource_audit_complete=False,native_raw_audit_complete=False)
    result['sha256']=phase.digest({k:v for k,v in result.items() if k!='sha256'})
    if output_path is not None:_write_output(output_path,Path(raw_directory).absolute(),result)
    return result

def _owned(path,parent,*,directory=False):
    path=Path(path).absolute()
    phase.require(path.resolve().parent==Path(parent).resolve() and (path.is_dir() if directory else path.is_file())
        and not path.is_symlink(),'exact owned terminal evidence path')
    if not directory:phase.regular(path)
    return path

def _resource_types(worker,sup,resources,heartbeats):
    def number(value):return type(value) in (int,float) and math.isfinite(value) and value>=0
    phase.require(number(worker['wall_seconds']) and number(sup['elapsed_seconds'])
        and worker['wall_seconds']<600 and sup['elapsed_seconds']<600,'finite unchanged terminal deadlines')
    phase.require(type(resources) is list and resources and all(type(r) is dict and r.get('kind') in ('CPU','GPU') for r in resources),
        'closed recorded resource observation kinds')
    for kind,fields in (('CPU',('sample_started_monotonic','sample_completed_monotonic','host_available_bytes','worker_tree_rss_bytes','disk_free_bytes')),
        ('GPU',('gpu_sample_started_monotonic','gpu_sample_completed_monotonic','gpu_free_bytes'))):
        rows=[r for r in resources if r['kind']==kind]
        phase.require(len(rows)>=2 and all(all(number(r[k]) for k in fields) for r in rows),'finite complete '+kind+' observations')
        phase.require(all(r[fields[0]]<=r[fields[1]] for r in rows)
            and all(a[fields[0]]<=b[fields[0]] and a[fields[1]]<=b[fields[1]] for a,b in zip(rows,rows[1:])),
            'chronological recorded '+kind+' sample envelopes')
    phase.require(len(heartbeats)>=2 and all(number(r['unix_seconds']) for r in heartbeats)
        and all(a['unix_seconds']<=b['unix_seconds'] for a,b in zip(heartbeats,heartbeats[1:])),'finite ordered recorded heartbeat times')
    for row in sup['admission_observations']+[sup['dispatch_observation']]:
        phase.require(all(type(row[k]) is int and row[k]>=0 for k in ('gpu_free_bytes','host_available_bytes','disk_free_bytes')),
            'typed recorded admission amounts')

def _terminal_context(config_path,raw_directory,preserved_directory,registration,selector,config_sha,preservation_sha,profile_sha,commit,*,fixture=False):
    phase.require(type(selector) is tuple and len(selector)==2 and selector[0] in ('atomic_RS_DEV','task_guard_DEV')
        and type(selector[1]) is int and 1<=selector[1]<=(6 if selector[0]=='atomic_RS_DEV' else 12),'exact independent fixed selector')
    config=_owned(config_path,PACKAGE/'configs');leaf=_owned(raw_directory,PACKAGE/'artifacts',directory=True)
    saved=_owned(preserved_directory,PACKAGE/'evidence',directory=True)
    phase.require(registration.get('config_path')==str(config.resolve()),'exact registration config path')
    raw=phase.read_bytes(config,8*1024**2);phase.require(sha(raw)==config_sha,'external exact config file')
    scope=phase.loads(raw);worker=phase.read_json(leaf/'worker_result.json');sup=phase.read_json(leaf/'SUPERVISOR_RECEIPT.json')
    pres=phase.read_json(saved/'preservation_receipt.json')
    phase.require(scope['sha256']==phase.digest({k:v for k,v in scope.items() if k!='sha256'}),'canonical frozen config body')
    phase.require(pres['sha256']==preservation_sha==phase.digest({k:v for k,v in pres.items() if k!='sha256'}),'external exact preservation body')
    phase.require(scope['short_phase_observation_profile']['id']==PROFILE_ID
        and phase.digest(scope['short_phase_observation_profile'])==profile_sha,'external exact phase profile')
    phase.require(worker['source_commit']==sup['preflight']['source_commit']==pres['source_commit']==commit,'external exact source HEAD joins')
    phase.require(worker['config_sha256']==sup['config_sha256']==config_sha,'external worker/supervisor config joins')
    phase.require(pres['raw_directory']==str(leaf) and pres['preserved_directory']==str(saved) and sup['output_dir']==str(leaf),
        'exact terminal/raw/preserved crosslinks')
    lease=Path(sup['lease_path']).absolute()
    phase.require(lease.resolve().parent==(PACKAGE/'artifacts/_native_gpu_leases').resolve() and lease.is_dir() and not lease.is_symlink(),
        'exact owned frozen native lease evidence')
    for path in lease.glob('*.json'):phase.read_json(path)
    names={p.name for p in leaf.iterdir() if p.is_file()}
    phase.require(names==set(pres['raw_files']) and len(names)<=2127,'complete bounded raw file denominator')
    for name in names:
        phase.require(Path(name).name==name,'owned leaf raw filename')
        info=phase.file_info(leaf/name)
        phase.require(info==phase.file_info(saved/name)==pres['raw_files'][name],'externally preserved raw byte pair '+name)
        if name.endswith('.jsonl'):phase.read_lines(leaf/name)
        elif name.endswith('.json'):phase.read_json(leaf/name)
        else:phase.require(name in ('worker.stdout.log','worker.stderr.log') and info['bytes']<=16*1024**2,'bounded exact terminal worker logs')
    phase.require(phase.file_info(config)==phase.file_info(saved/'config_snapshot.json')==pres['config_snapshot'],'externally preserved config bytes')
    _resource_types(worker,sup,phase.read_json(leaf/'RESOURCE_OBSERVATIONS.json'),phase.read_lines(leaf/'supervisor_heartbeat.jsonl'))
    proof=_phase3_early(scope,worker,sup,pres,selector,phase.require,leaf,fixture=fixture)
    return {'scope':scope,'worker':worker,'supervisor':sup,'preservation':pres,'phase_proof':proof,
        'markers':phase.read_lines(leaf/'PHASE_OBSERVATIONS.jsonl',canonical_lines=True)}

class _NativeReads:
    def __init__(self,actual):self.actual=actual
    def __getattr__(self,name):return getattr(self.actual,name)
    def read_json(self,path,limit=8*1024**2):return phase.read_json(path,limit)
    def filehash(self,path):return phase.file_info(path)['sha256']

def _inherited_module(source):
    """Execute the exact 14-site adaptation with bounded IO-only facades."""
    module=types.ModuleType('short_phase3_preserved_RAM2_adapter');module.__file__=str(HERE/'audit_adapter.py')
    exec(compile(source,str(DEPENDENCY),'exec'),module.__dict__)
    module._phase3_early=_phase3_early;module._phase3_admission=_phase3_admission
    actual_util=importlib.util
    def specification(name,path):
        spec=actual_util.spec_from_file_location(name,path);actual_loader=spec.loader
        class Loader:
            def create_module(self,spec):return actual_loader.create_module(spec)
            def exec_module(self,loaded):
                actual_loader.exec_module(loaded);loaded.native=_NativeReads(loaded.native)
        spec.loader=Loader();return spec
    util=types.SimpleNamespace(spec_from_file_location=specification,module_from_spec=actual_util.module_from_spec)
    module.importlib=types.SimpleNamespace(util=util,metadata=importlib.metadata)
    # The original JSONL reader uses json.loads. Its file/line/count bounds have
    # already been independently verified; force finite numeric decoding too.
    module.json=types.SimpleNamespace(loads=lambda raw,**kwargs:phase.loads(raw.encode('utf-8') if isinstance(raw,str) else raw),
        dump=json.dump)
    return module

def _write_output(output,leaf,result):
    target=Path(output).absolute()
    phase.require(target.resolve().parent==(PACKAGE/'evidence').resolve() and target.name.startswith(leaf.name+'_independent_audit_')
        and target.suffix=='.json' and not target.exists() and not target.is_symlink(),'new owned exclusive completed audit output')
    raw=(phase.canonical(result)+'\n').encode('ascii');phase.require(len(raw)<=phase.MAX_JSON,'bounded completed audit output')
    with target.open('xb',buffering=0) as stream:
        phase.require(stream.write(raw)==len(raw),'complete audit receipt write');stream.flush();os.fsync(stream.fileno())
    phase.require(phase.read_json(target)==result,'independent completed audit byte reread')
