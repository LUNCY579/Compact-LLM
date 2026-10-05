"""New-only Phase3 raw audit adaptation of exact preserved RAM2 source.

No native model/framework/OS work. Expensive decoder and byte verification are
inside the inherited terminal audit, unavailable until all root prerequisites.
"""
from pathlib import Path
from copy import deepcopy
import ast,hashlib,importlib.util,json,sys
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
sys.path.insert(0,str(HERE))
import phase_replay as phase
SIGNAL='ROOT_TERMINAL_AND_PRESERVED'
PIN='d69810dccd11a32595dace73a788c8868efabb2a09f593fd33316e92dcb47442'
DEPENDENCY=HERE/'pinned_dependencies/RAM2_auditor_d69810.py'
VERSION='synth-short-M0-phase-observed-staged-3'
PROFILE_ID='SHORT_M0_PHASE_OBSERVED_PHYSICAL_RAM_V3'
REGISTRATION_ID='SYNTH_SHORT_PHASE3_INDEPENDENT_AUDIT_INVENTORY_001'
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
    check(registration.get('id')==REGISTRATION_ID and registration.get('contract')==VERSION,'new-only Phase3 inventory')
    check(registration.get('sha256')==phase.digest({k:v for k,v in registration.items() if k!='sha256'}),'inventory checksum')
    profile=scope.get('short_phase_observation_profile')
    check(profile==current_profile and profile.get('id')==PROFILE_ID and phase.digest(profile)==expected_profile==registration.get('profile_body_sha256'),
        'externally pinned exact Phase3 profile')
    check(scope.get('contract')==VERSION and registration.get('source_commit')==sup['preflight']['source_commit'],'exact source contract/commit')
    sections={'source':'source_pins','runtime':'runtime_pins','decoder':'decoder_runtime_pins','data':'data_pins','model':'model_pins'}
    check(set(registration['inventories'])==set(registration['counts'])==set(sections),'exact frozen inventory sections')
    for section,key in sections.items():
        check(registration['inventories'][section]==scope[key] and len(scope[key])==registration['counts'][section],
            'every externally frozen path/hash/count '+section)
    check({k:registration['counts'][k] for k in ('runtime','decoder','data','model')}=={'runtime':65,'decoder':17,'data':5,'model':7},'unchanged mandatory inventories')
    for path,pin in current_sources.items():check(scope['source_pins'].get(path)==pin,'actual integrated/helper source '+path)
    check(scope['short_RAM_monitor_profile']['id']=='SHORT_M0_PHYSICAL_RAM_COHERENT_EXTERNAL_CHECK_V2'
        and scope['short_RAM_monitor_profile']['all_floors_and_numerics_unchanged'] is True,'unchanged RAM2 foundation')
    check(sup.get('short_RAM_monitor_diagnostic')=={'path':'SHORT_RAM_ABORT_DIAGNOSTIC.json','attempted':False,'persisted':False,
        'write_error':None,'original_abort_reason_preserved':None},'successful terminal cannot carry abort contradiction')
    return registration['counts']

def _phase3_early(scope,worker,sup,pres,selector,check,leaf):
    check(scope.get('contract')==VERSION and scope['authorization']['scope']=='short_M0_phase_observed_raw_collection_one_fixed_batch_only',
        'exact new-only raw collection purpose')
    check(scope['batch_selector']=={'source_kind':selector[0],'ordinal':selector[1]},'independent fixed selector')
    check(sup['status']=='SUPERVISED_WORKER_EXITED_ZERO' and sup['returncode']==0 and sup['partial_outputs_frozen'] is True,'successful frozen owned terminal')
    check(worker['status']=='COMPLETE_SCOPED_SHORT_M0_RAW_COLLECTION_UNSCORED' and worker['error'] is None,'complete raw worker')
    rows=scope['rows'];outputs=worker['collection']['rows']
    check(len(rows)==len(outputs)==64 and len({r['row_id'] for r in rows})==64
        and [r['row_id'] for r in rows]==[r['row_id'] for r in outputs],'whole unique ordered64 before expensive reads')
    check(phase.digest(rows)==scope['source_packet_sha256']==worker['collection']['source_packet_sha256'],'exact whole packet')
    check(worker['source_commit']==pres['source_commit']==sup['preflight']['source_commit'],'source barrier')
    markers=phase.parse((leaf/'PHASE_OBSERVATIONS.jsonl').read_bytes())
    summary=json.loads((leaf/'PHASE_ACCOUNTING_FINAL.json').read_bytes(),object_pairs_hook=phase.pairs)
    events=[]
    with (leaf/'events.jsonl').open('rb') as stream:
        for line in stream:
            check(line.endswith(b'\n') and len(line)<=65536,'bounded actual Journal event')
            events.append(json.loads(line,object_pairs_hook=phase.pairs))
    identity=summary['identity']
    check(identity=={'execution_scope':'OWNED_SHORT_M0_PHASE_OBSERVED_RAW_COLLECTION','purpose':scope['authorization']['scope'],
        'config_sha256':worker['config_sha256'],'source_commit':worker['source_commit'],'batch_selector':scope['batch_selector']},'exact external raw/final identity')
    proof=phase.replay(markers,summary,worker,events,fixture=False)
    check(proof['native_raw_audit_complete'] is False,'phase proof is prerequisite, not full native authority')

def prerequisites(*,terminal_signal,inventory_path,inventory_sha,profile_sha,config_sha,preservation_sha,source_commit):
    phase.require(terminal_signal==SIGNAL,'root terminal+preserved signal before any heavy reads')
    for value in (inventory_sha,profile_sha,config_sha,preservation_sha):
        phase.require(type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value),'independently supplied exact anchors')
    phase.require(type(source_commit) is str and len(source_commit)==40,'independent exact launch HEAD')
    integrated=PACKAGE/'short_phase_contract.py'
    phase.require(integrated.is_file() and not integrated.is_symlink(),'integrated Phase3 dependency absent; native audit closed')
    path=Path(inventory_path).absolute();phase.require(path.is_file() and not path.is_symlink()
        and path.resolve().parent==(PACKAGE/'artifacts/short_m0_audit_registrations').resolve(),'owned independent inventory file')
    phase.require(sha(path.read_bytes())==inventory_sha,'external inventory file SHA')
    registration=json.loads(path.read_bytes(),object_pairs_hook=phase.pairs)
    phase.require(registration['source_commit']==source_commit and registration['config_file_sha256']==config_sha
        and registration['profile_body_sha256']==profile_sha,'independent launch/profile/config anchors')
    return registration

def audit_terminal(config_path,raw_directory,preserved_directory,*,expected_selector,expected_config_file_sha256,
    expected_preservation_body_sha256,inventory_registration_path,expected_inventory_registration_file_sha256,
    expected_profile_body_sha256,expected_source_commit,terminal_preservation_signal=None,output_path=None):
    prerequisites(terminal_signal=terminal_preservation_signal,inventory_path=inventory_registration_path,
        inventory_sha=expected_inventory_registration_file_sha256,profile_sha=expected_profile_body_sha256,
        config_sha=expected_config_file_sha256,preservation_sha=expected_preservation_body_sha256,source_commit=expected_source_commit)
    # A complete independent loader-guard replay profile must be registered
    # before this native driver can run inherited decoder/model byte checks.
    phase.require(False,'full independent loader phase replay registration not yet implemented; native audit remains closed')
