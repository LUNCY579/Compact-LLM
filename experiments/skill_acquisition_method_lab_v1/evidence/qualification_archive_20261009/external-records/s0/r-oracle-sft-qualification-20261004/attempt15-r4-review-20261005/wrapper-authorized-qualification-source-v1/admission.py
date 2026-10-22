"""Fresh source-issued CPU probe authority; target qualification is an output.

Metadata preparation is the explicit pre-root boundary, as in the previously
observed owner fixture. Target CLR startup is inside the issued target root.
No existing grant, target-native receipt or caller status can issue this permit.
"""
import hashlib,json,math,os,time,types,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OWNER='01a0dfa3-9cdf-7703-a362-32f559ec15a7'
HEAD='eee2e0bd31854ea207b231d65eedd4c1c94e1011'
FIELDS=('operation_attempts','read_bytes','write_bytes','identity_hash_attempts','identity_hash_bytes','metadata_bytes','wall_seconds')
BOUNDS=dict(zip(FIELDS,(150000,268435456,1048576,2048,33554432,1048576,60)))
TAIL=dict(zip(FIELDS,(256,65536,65536,16,65536,65536,5)))
STANDING_AUTH_HISTORY={'path':r'C:\Users\clclc\BusinessProject\comp-rs-001-records\s0\r-oracle-sft-qualification-20261004\attempt14-failure-correction-20261005\VERIFIED_STANDING_AUTHORIZATION_20261005.json','bytes':1581,'sha256':'4458c60aaa4afc1e6ee1ad1d31b5e22cd1da507df2777d7e66e04d362767d994'}
AUTH={'path': 'C:\\Users\\clclc\\BusinessProject\\comp-rs-001-records\\s0\\r-oracle-sft-qualification-20261004\\attempt15-r4-review-20261005\\OWNER_NATIVE_PUBLICATION_AUTHORIZATION_20261009_62.json', 'bytes': 2126, 'sha256': '0ed66ea842a5126071a751eb3ca4407b3aab76c54c8c9f75c2e32e5401fcdbd0'}
PROVIDER={'path':r'C:\Users\clclc\BusinessProject\comp-rs-001-records\s0\r-oracle-sft-qualification-20261004\attempt15-review-20261005\wrapper-deadline-correction-v1\owned_job_provider.py','bytes':24794,'sha256':'c1ab087e0c9cbfc0967e3f8ac21ee643728bc883ff9432a104fd74c98adcb3c4'}
JOB_CENSUS={'path': 'C:\\Users\\clclc\\BusinessProject\\comp-rs-001-records\\s0\\r-oracle-sft-qualification-20261004\\attempt15-r4-review-20261005\\wrapper-authorized-qualification-source-v1\\reusable_census.py', 'bytes': 10032, 'sha256': '198190b57f2ec3eccd604be9b46d3f46328eb1f9adf39de9208fba5e57fcac4d'}
OLD_CENSUS={'path':r'C:\Users\clclc\BusinessProject\comp-rs-001-records\s0\r-oracle-sft-qualification-20261004\attempt15-review-20261005\owned-mechanism-terminal-time-review1\job_census.py','bytes':4022,'sha256':'3d05bc47b9b88bb255332aa1fbf0d29b716ab1a5a80d36899b78f551d282a220'}
OUTPUT=ROOT/'native-one-attempt'
ACTOR_OUTPUT=OUTPUT/'actor-probe'
SOURCE_NAMES=('admission.py','issue_grant.py','harness.py','backend.py','materialize.py','runtime_custody.py','run_qualification.py','accounting.py','reusable_census.py','v2_consumer_bindings.py','source_envelopes.py','failure_envelope.py','bootstrap_entry.py','metadata_entry.py')
_SEAL=object()
CASE_NAMES=('normal_exit','timeout')
CASE_SELECTED=dict(zip(FIELDS,(8192,4194304,131072,128,4194304,327680,60)))
def partitions():
    values={name:dict(CASE_SELECTED) for name in CASE_NAMES}
    values['native_host']=dict(zip(FIELDS,(60000,0,0,0,0,0,60)))
    values['logs']=dict(zip(FIELDS,(4096,16777216,16384,0,0,16384,60)))
    values['final_tail']=dict(TAIL)
    values['control']={key:BOUNDS[key]-sum(v[key] for v in values.values()) for key in FIELDS if key!='wall_seconds'}
    values['control']['wall_seconds']=60
    require(all(v>0 for v in values['control'].values()),'positive exact original remainder')
    require(all(sum(v[key] for v in values.values())==BOUNDS[key] for key in FIELDS if key!='wall_seconds'),'complete disjoint unchanged six-counter reservation')
    return values
def require(v,m):
    if v is not True:raise RuntimeError(m)
import accounting as cost
try:
    _PREP_OWNER=cost.entry_preparation_owner()
    _ENTRY_OWNER_BOUND=True
except RuntimeError:
    # Cold imports exist for mechanical source controls only. Native entry
    # must arrive through the exact same-process bootstrap registry.
    _PREP_OWNER=cost.PreparationOwner()
    _ENTRY_OWNER_BOUND=False
_ROOT_OWNER=None
_MODULES={}
_PIN_RELEASE_ATTEMPTED=False
def release_original_pins(custody,before_close):
    global _PIN_RELEASE_ATTEMPTED
    require(_PIN_RELEASE_ATTEMPTED is False,'original pin close plan already attempted; preserve unresolved handles')
    # Mark BEFORE counter/native callbacks. Even a failing before callback may
    # be followed by an actual original close inside RuntimeCustody.
    _PIN_RELEASE_ATTEMPTED=True
    return custody.close_all(before_close=before_close)

def preparation_owner():
    require(_ROOT_OWNER is None,'preparation cannot be reopened after original root');return _PREP_OWNER
def activate_root(owner):
    global _ROOT_OWNER
    require(type(owner) is cost.Budget and _ROOT_OWNER is None,'one exact original owner activation')
    _ROOT_OWNER=owner;return owner
def checked_owner(owner=None):
    if owner is None:
        require(_ROOT_OWNER is None,'explicit original owner required after origin; no raw/free helper path')
        return _PREP_OWNER
    require(type(owner) in (cost.Budget,cost.PreparationOwner),'exact concrete accounting owner')
    require((_ROOT_OWNER is None and owner is _PREP_OWNER) or owner is _ROOT_OWNER,'same source owner, no second allocation')
    return owner
def canonical(value,*,owner=None,maximum=131072):return checked_owner(owner).canonical(value,maximum)
def strict(blob,*,owner=None,maximum=131072):return checked_owner(owner).strict(blob,maximum)
def ref(path,blob,*,owner=None):
    return {'path':str(path),'bytes':len(blob),'sha256':checked_owner(owner).sha(blob)}
def read_ref(item,maximum=33554432,*,owner=None):
    o=checked_owner(owner)
    require(type(item) in (dict,cost.FrozenMap) and set(item)=={'path','bytes','sha256'} and type(item['path']) is str
      and type(item['bytes']) is int and 0<item['bytes']<=maximum and type(item['sha256']) is str and len(item['sha256'])==64
      and not set(item['sha256'])-set('0123456789abcdef'),'exact bounded source reference')
    def read():
        path=Path(item['path']);require(path.is_absolute() and '..' not in path.parts and path.resolve()==path
          and all(not q.is_symlink() and not(getattr(q.stat(),'st_file_attributes',0)&1024) for q in (path,*path.parents))
          and path.stat().st_size==item['bytes'],'regular exact source path')
        with path.open('rb') as stream:blob=stream.read(item['bytes']+1)
        require(len(blob)==item['bytes'],'exact source extent');return blob
    blob=o.run('control',{'operation_attempts':1,'read_bytes':item['bytes']+1},read)
    require(o.sha(blob)==item['sha256'],'exact source SHA changed');return blob
def fixed(name,maximum=131072,*,owner=None):
    o=checked_owner(owner)
    def read():
        path=ROOT/name;require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=maximum,'fixed metadata missing')
        with path.open('rb') as stream:blob=stream.read(maximum+1)
        require(len(blob)<=maximum,'fixed metadata grew');return blob
    blob=o.run('control',{'operation_attempts':1,'read_bytes':maximum+1},read)
    return strict(blob,owner=o,maximum=maximum),ref(ROOT/name,blob,owner=o)
def module(item,name,*,owner=None):
    o=checked_owner(owner);blob=read_ref(item,owner=o)
    def load():
        require(name not in _MODULES,'source module installed once')
        value=types.ModuleType(name);value.__file__=item['path'];value.__dict__['__CE8_EXECUTED_SOURCE_SHA256']=item['sha256']
        exec(compile(blob,item['path'],'exec'),value.__dict__);_MODULES[name]=value;return value
    return o.run('control',{'operation_attempts':32,'metadata_bytes':4096},load)
def validate_authorization(auth):
    # This exact create-only human record is an authorization input. It cannot
    # override automatic tool policy, issue a grant, or waive downstream gates.
    require(type(auth) is dict and auth.get('schema')=='CE8-DIRECT-QUALIFICATION-PUBLICATION-AUTHORIZATION-62'
      and auth.get('owner_thread_id')==OWNER and auth.get('direct_human_authorization') is True
      and auth.get('provenance')=='Direct human user message in this owner chat; not delegated task text'
      and auth.get('publication_scope')=='QUALIFICATION_ONLY'
      and auth.get('normal_automatic_review_required') is True
      and auth.get('normal_resource_admission_required') is True
      and auth.get('existing_caps_and_scope_preserved') is True
      and auth.get('native_grant_created_by_this_record') is False
      and auth.get('execution_grant_created_by_this_record') is False
      and auth.get('this_record_overrides_tool_policy') is False,
      'exact direct human qualification authorization with ordinary gates preserved')
    cpu=auth.get('CPU_prerequisite')
    require(type(cpu) is dict and set(cpu)=={'physical_root_seconds','single_nested_tail_seconds','cases',
      'attempts_per_fresh_reviewed_grant','models_allowed','optimizer_updates','gpu_use_allowed',
      'launch_ram_bytes','ongoing_ram_bytes'}
      and type(cpu['physical_root_seconds']) is int and cpu['physical_root_seconds']==60
      and type(cpu['single_nested_tail_seconds']) is int and cpu['single_nested_tail_seconds']==5
      and cpu['cases']==list(CASE_NAMES)
      and type(cpu['attempts_per_fresh_reviewed_grant']) is int and cpu['attempts_per_fresh_reviewed_grant']==1
      and type(cpu['models_allowed']) is int and cpu['models_allowed']==0
      and type(cpu['optimizer_updates']) is int and cpu['optimizer_updates']==0
      and cpu['gpu_use_allowed'] is False and cpu['launch_ram_bytes']==8589934592
      and cpu['ongoing_ram_bytes']==2147483648,'exact unchanged one-attempt CPU prerequisite scope')
    model=auth.get('registered_model_qualification_scope')
    require(type(model) is dict and model.get('physical_optimizer_updates')==6
      and model.get('data')=='TRAIN_ONLY' and model.get('weights')=='DISCARD'
      and model.get('hard_physical_deadline_seconds')==19200
      and model.get('requires_separate_registered_model_source_and_native_checks') is True,
      'model scope remains separate from this zero-update CPU component')
    return True

def validate_packet(p):
    require(type(p) is dict and p['schema']=='CE8-ACTOR-QUALIFICATION-PACKET-1' and p['owner_thread_id']==OWNER
      and p['source_head']==HEAD and p['phase']=='CE8_OWNED_NON_ML_OS_FIXTURE' and p['bounds']==BOUNDS and p['final_tail']==TAIL
      and p['provider_ref']==PROVIDER and p['job_census_ref']==JOB_CENSUS and p['authorization_ref']==AUTH and p['attempts']==1
      and p['models_allowed']==0 and p['optimizer_updates']==0 and p['gpu_use_allowed'] is False
      and p['frameworks_allowed']==[] and p['scientific_inputs_allowed']==[] and p['launch_ram_bytes']==8589934592
      and p['ongoing_ram_bytes']==2147483648 and p['max_rss_bytes']==536870912
      and p['output_root']==str(ACTOR_OUTPUT) and p['target_case']=='FIXED_RETAINED_BARE_PYTHON_NORMAL_AND_TIMEOUT'
      and p['case_names']==list(CASE_NAMES) and p['aggregate_partitions']==partitions(),
      'one exact CPU60 actor qualification scope; no model or prior actor qualification')
    require(set(p['source_refs'])==set(SOURCE_NAMES),'closed current source census')
    entryref=p['entry_identities_ref']
    require(type(entryref) is dict and entryref['path']==str(ROOT/'ENTRY_IDENTITIES_V2.json'),'fixed entry identity manifest path')
    identities=strict(read_ref(entryref,16384))
    require(type(identities) is dict and set(identities)=={'schema','accounting_ref','admission_ref','entry_ref'}
      and identities['schema']=='CE8-FIXED-LOCAL-CALLER-ENTRY-IDENTITIES-1'
      and identities['accounting_ref']==p['source_refs']['accounting.py']
      and identities['admission_ref']==p['source_refs']['admission.py']
      and identities['entry_ref']==p['source_refs']['bootstrap_entry.py'],'exact acyclic entry identities match execution census')
    if _ENTRY_OWNER_BOUND:
        require(globals().get('_ENTRY_IDENTITIES')==identities and globals().get('_ENTRY_IDENTITIES_REF')==entryref,
          'same source-owned entry provenance; no caller status authority')

    for name,item in p['source_refs'].items():
        require(Path(item['path'])==ROOT/name,'exact current source path');read_ref(item,1048576)
    read_ref(PROVIDER,1048576);read_ref(JOB_CENSUS,1048576);auth=strict(read_ref(AUTH,65536))
    validate_authorization(auth)
    m=strict(read_ref(p['target_manifest_ref'],131072))
    require(type(m) is dict and set(m)=={'schema','actor_source_ref','actor_source_freeze_ref','artifact_ref','runtimeconfig_ref','runtime_ref','framework_refs','runtime_dependency_refs','runtime_package_manifest_ref','runtime_namespace_roots','platform_boundary','launch_kind','case'}
      and m['schema']=='CE8-EXACT-ACTOR-TARGET-2' and m['launch_kind']=='DOTNET_EXEC_FIXED_DLL'
      and m['case']==p['target_case'] and type(m['framework_refs']) is list and bool(m['framework_refs'])
      and type(m['runtime_dependency_refs']) is dict and bool(m['runtime_dependency_refs']),
      'compiled exact actor/runtime/case manifest absent; source metadata is not qualification')
    # Raw runtime identity verification is performed once by PRE-GRANT original
    # deny-write/delete custody. This metadata predicate grants no permission.
    for item in (m['actor_source_ref'],m['actor_source_freeze_ref'],m['artifact_ref'],m['runtimeconfig_ref'],m['runtime_ref'],*m['runtime_dependency_refs'].values()):
        require(type(item) is dict and set(item)=={'path','bytes','sha256'} and type(item['bytes']) is int and 0<item['bytes']<=33554432,'bounded future held raw identity')
    config=strict(read_ref(m['runtimeconfig_ref']))
    ro=config['runtimeOptions'];require(ro['tfm']=='net8.0' and ro['framework']=={'name':'Microsoft.NETCore.App','version':'8.0.21'} and ro['rollForward']=='Disable','exact .NET8.0.21 runtime, no roll-forward')
    require(p['target_source_ref']==m['actor_source_ref'] and p['runtime_ref']==m['runtime_ref'],'same exact actor and actual process image')
    binding=preparation_owner().before('control',operation_attempts=128,metadata_bytes=4096)
    require('v2_consumer_bindings' not in sys.modules and 'ce8_prepared_v2_bindings' not in sys.modules,
      'no unpaid ambient v2 helper alias')
    helper=_MODULES.get('ce8_prepared_v2_bindings')
    if helper is None:helper=module(p['source_refs']['v2_consumer_bindings.py'],'ce8_prepared_v2_bindings',owner=preparation_owner())
    else:require(helper.__file__==p['source_refs']['v2_consumer_bindings.py']['path']
      and helper.__dict__.get('__CE8_EXECUTED_SOURCE_SHA256')==p['source_refs']['v2_consumer_bindings.py']['sha256'],
      'same prepared exact v2 helper, no alias/reload credit')
    helper.validate_v2_target_bindings(m);binding.finish()
    return m
def review_binding(p,pref,freeze,fref,review):
    require(freeze['packet_ref']==pref and freeze['source_refs']==p['source_refs'] and freeze['target_manifest_ref']==p['target_manifest_ref'], 'acyclic exact source freeze')
    require(review['packet_ref']==pref and review['source_freeze_ref']==fref and review['owner_thread_id']==OWNER
      and review['source_review_status']=='PASS_FOR_THIS_BOUNDED_OS_FIXTURE' and review['native_evidence_status']=='NOT_RUN'
      and review['reviewed_source_refs']==p['source_refs'] and review['target_manifest_ref']==p['target_manifest_ref']
      and review['complete_all7_accounting_review']=='PASS_EXACT_UNCHANGED_CAPS'
      and review['preparation_bound']==dict(zip(FIELDS,cost.PREPARATION_BOUNDS))
      and review['preparation_phase_status']=='PASS_PROSPECTIVE_BOUND'
      and review['authorization_ref']==AUTH and review['automatic_tool_policy_overridden'] is False
      and review['scope']=={'attempts':1,'case_names':list(CASE_NAMES),'frameworks_allowed':[],
        'gpu_use_allowed':False,'models_allowed':0,'optimizer_updates':0,
        'phase':'CE8_OWNED_NON_ML_OS_FIXTURE','physical_root_seconds':60,
        'scientific_inputs_allowed':[],'single_nested_tail_seconds':5},
      'independent exact scope/cost/source/authorization review; native proof is an output')
    return True
_STATIC_SEAL=object()
_STATIC_CONTEXTS={}
class PreparedStaticViews:
    __slots__=('_seal','_prep_owner','_pid','_custody','_modules','_root_owner','packet','packet_ref','target','envelopes','envelopes_module','failure_module','creation_plan','tail_authority')
    def __setattr__(self,*args):raise RuntimeError('immutable original preparation-owned static context')
    def __delattr__(self,*args):raise RuntimeError('immutable original preparation-owned static context')
    @classmethod
    def create(cls,p,pref,target,custody,modules,owner):
        require(type(owner) is cost.PreparationOwner and owner is preparation_owner(),'same bounded original preparation owner')
        custody_module=_MODULES.get('ce8_prepared_runtime_custody')
        require(type(custody_module) is types.ModuleType and type(custody) is custody_module.RuntimeCustody
          and custody.require_source_owner() is custody,'same paid source-owned custody class; no ambient import')
        ticket=owner.before('control',operation_attempts=1,metadata_bytes=2048)
        packet_view={key:p[key] for key in ('target_source_ref','target_manifest_ref','source_refs')}
        target_view={key:target[key] for key in ('runtime_ref','artifact_ref')}
        ticket.finish()
        # Only static validated data is frozen here. No fresh root clock/grant
        # exists, so no dynamic root work migrates into preparation.
        values=owner.freeze({'packet':packet_view,'packet_ref':pref,'target':target_view},65536)
        alias_ticket=owner.before('control',operation_attempts=1,metadata_bytes=128)
        require('source_envelopes' not in sys.modules and 'ce8_prepared_source_envelopes' not in sys.modules,
          'source envelope code must not execute through an unpaid ambient alias')
        alias_ticket.finish()
        envelope_module=module(p['source_refs']['source_envelopes.py'],'ce8_prepared_source_envelopes',owner=owner)
        plan=owner.run('control',{'operation_attempts':128,'metadata_bytes':65536},lambda:envelope_module.immutable_plan(envelope_module.derive_envelopes(
          {'path':str(ROOT/'FIXTURE_GRANT.json'),'bytes':envelope_module.UInt(15),'sha256':envelope_module.HexText(64)},
          p['target_manifest_ref'],p['target_source_ref'],target['runtime_ref'],ACTOR_OUTPUT,BOUNDS,partitions(),
          target['artifact_ref']['path'],CASE_SELECTED)))
        require(plan['prepayment_upper']<=4096 and max(plan['birth_uppers'].values())<=4096 and plan['final_upper']<=24576,
          'complete actual-path source envelopes exceed proposed smaller bounds; no fitted discount')
        # Exact prepared backend object owns a once-paid immutable creation plan.
        creation_plan=modules[1].PreparedCreationPlan.prepare(owner,modules[0],target,custody=custody)
        fields_ticket=owner.before('control',operation_attempts=1,metadata_bytes=512)
        # Select the new complete registry/code/reference construction before
        # allocation, under unchanged separately bounded PREP caps.
        tail_authority=owner.run('control',{'operation_attempts':32,'metadata_bytes':4096},
          lambda:cost.PreparedTailAuthority.prepare(owner,p,custody))
        obj=object.__new__(cls)
        fields={'_seal':_STATIC_SEAL,'_prep_owner':owner,'_pid':os.getpid(),'_custody':custody,
          '_modules':tuple(modules),'_root_owner':None,'creation_plan':creation_plan,'tail_authority':tail_authority,'envelopes':plan,'envelopes_module':envelope_module,'failure_module':_MODULES['ce8_prepared_failure_envelope'],**dict(values.items())}
        for key,value in fields.items():object.__setattr__(obj,key,value)
        _STATIC_CONTEXTS[id(obj)]=obj;fields_ticket.finish()
        owner.run('control',{'operation_attempts':256,'metadata_bytes':65536},lambda:tail_authority.prepare_catalogue(obj))
        return obj
    def bind_original_root(self,owner,custody):
        ticket=owner.before('control',operation_attempts=1,metadata_bytes=256)
        require(type(self) is PreparedStaticViews and self._seal is _STATIC_SEAL and _STATIC_CONTEXTS.get(id(self)) is self
          and self._pid==os.getpid() and self._prep_owner is _PREP_OWNER and self._custody is custody
          and self._root_owner is None and checked_owner(owner) is owner,'one same-context original-root custody handoff')
        require(not custody._closed and not custody._failed and custody._pid==os.getpid(),'same live read pins span original root')
        self.creation_plan.bind_original_root(owner,self)
        object.__setattr__(self,'_root_owner',owner);ticket.finish();return self
    def require(self,owner,custody):
        require(self._seal is _STATIC_SEAL and _STATIC_CONTEXTS.get(id(self)) is self and self._root_owner is owner
          and self._custody is custody and self._pid==os.getpid() and self._prep_owner is _PREP_OWNER,
          'same source-owned immutable static views; no decoded caller object or second owner')
        return None
class SourceIssuedFixturePermit:
    __slots__=('_seal','_pid','_owner','_static','packet','packet_ref','grant','grant_ref','target','custody','prepared_modules')
    def __setattr__(self,*args):raise RuntimeError('immutable source-issued permit')
    def __delattr__(self,*args):raise RuntimeError('immutable source-issued permit')
    def __init__(self,seal,static,g,gref,custody,owner):
        require(seal is _SEAL and checked_owner(owner) is owner and type(static) is PreparedStaticViews,'actual original issuer/root/static source types')
        static.bind_original_root(owner,custody)
        # Root owns only newly issued dynamic data. Static immutable objects are
        # borrowed BY IDENTITY from the same paid preparation/custody context.
        dynamic=owner.freeze({'grant':g,'grant_ref':gref},8192)
        fields_ticket=owner.before('control',operation_attempts=1,metadata_bytes=512)
        fields={'_seal':seal,'_pid':os.getpid(),'_owner':owner,'_static':static,'custody':custody,
          'packet':static.packet,'packet_ref':static.packet_ref,'target':static.target,'prepared_modules':static._modules,
          'grant':dynamic['grant'],'grant_ref':dynamic['grant_ref']}
        for key,value in fields.items():object.__setattr__(self,key,value)
        fields_ticket.finish()
    @property
    def owner(self):return self._owner
    def require(self):
        ticket=self._owner.before('control',operation_attempts=1,metadata_bytes=128)
        require(self._seal is _SEAL and self._pid==os.getpid() and checked_owner(self._owner) is self._owner
          and self.custody._pid==os.getpid() and not self.custody._closed and not self.custody._failed,
          'same immutable source permit/owner and live original pins')
        self._static.require(self._owner,self.custody)
        require(self.packet is self._static.packet and self.packet_ref is self._static.packet_ref
          and self.target is self._static.target and self.prepared_modules is self._static._modules,
          'static preparation views are borrowed, never recopied or replaced in root')
        ticket.finish();return None
def admission():
    require(False,'NATIVE_NO_ADMIT: cold metadata cannot recreate original owner/custody; direct handoff required')
