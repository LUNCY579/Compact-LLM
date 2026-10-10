"""AST-extracted mechanical authorization controls and create-only source seal."""
import ast, copy, difflib, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
BASE=ROOT.parent/'wrapper-complete-close-ledger-source-v1'
NAMES=('admission.py','issue_grant.py','harness.py','backend.py','materialize.py','runtime_custody.py','run_qualification.py','accounting.py','reusable_census.py','source_envelopes.py','failure_envelope.py','bootstrap_entry.py','metadata_entry.py','v2_consumer_bindings.py')
def ref(path):
 raw=path.read_bytes();return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
def publish(name,value):
 with (ROOT/name).open('x') as stream:json.dump(value,stream,sort_keys=True,indent=2);stream.write('\n')
declared=[]
for name in NAMES:
 tree=ast.parse((ROOT/name).read_text(),filename=str(ROOT/name))
 for node in ast.walk(tree):
  if not isinstance(node,ast.Dict):continue
  try:item=ast.literal_eval(node)
  except (ValueError,TypeError):continue
  if type(item) is dict and set(item)=={'path','bytes','sha256'} and type(item['path']) is str:
   assert ref(Path(item['path']))==item,(name,item)
   declared.append({'source':str(ROOT/name),'ref':item})
assert len(declared)==5
tree=ast.parse((ROOT/'admission.py').read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('require','validate_authorization','review_binding')]
namespace={'OWNER':'01a0dfa3-9cdf-7703-a362-32f559ec15a7','CASE_NAMES':('normal_exit','timeout')}
for node in tree.body:
 if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='AUTH' for t in node.targets):namespace['AUTH']=ast.literal_eval(node.value)
exec(compile(ast.Module(body=functions,type_ignores=[]),'mechanical_authorization_extract','exec'),namespace)
auth=json.loads(Path(namespace['AUTH']['path']).read_text())
checks=[]
def test(name,action,denied=False):
 try:result=action()
 except (RuntimeError,KeyError,TypeError,AttributeError):
  assert denied,name;result='DENIED'
 else:assert not denied and result is True,name
 checks.append({'name':name,'pass':True,'outcome':result})
test('exact direct authorization accepted mechanically',lambda:namespace['validate_authorization'](auth))
for key,value in [('schema','stale'),('owner_thread_id','other'),('publication_scope','ACQUISITION'),('normal_automatic_review_required',False),('normal_resource_admission_required',False),('existing_caps_and_scope_preserved',False),('native_grant_created_by_this_record',True),('this_record_overrides_tool_policy',True)]:
 bad=copy.deepcopy(auth);bad[key]=value
 test('authorization refuses '+key,lambda bad=bad:namespace['validate_authorization'](bad),True)
for key,value in [('physical_root_seconds',61),('single_nested_tail_seconds',6),('models_allowed',1),('optimizer_updates',6),('gpu_use_allowed',True),('attempts_per_fresh_reviewed_grant',2),('launch_ram_bytes',1),('cases',['normal_exit'])]:
 bad=copy.deepcopy(auth);bad['CPU_prerequisite'][key]=value
 test('CPU scope refuses '+key,lambda bad=bad:namespace['validate_authorization'](bad),True)
test('missing authorization refuses',lambda:namespace['validate_authorization']({}),True)

# No candidate modules or native libraries imported. Build only synthetic data.
class FakeCost:
 PREPARATION_BOUNDS=(50000,134217728,4194304,2048,134217728,8388608,1800)
namespace['cost']=FakeCost
namespace['FIELDS']=('operation_attempts','read_bytes','write_bytes','identity_hash_attempts','identity_hash_bytes','metadata_bytes','wall_seconds')
pref={'path':'synthetic packet','bytes':1,'sha256':'1'*64};fref={'path':'synthetic freeze','bytes':1,'sha256':'2'*64}
source={'sample':'same'};target={'path':'synthetic target','bytes':1,'sha256':'3'*64}
packet={'source_refs':source,'target_manifest_ref':target}
freeze={'packet_ref':pref,'source_refs':source,'target_manifest_ref':target}
scope={'attempts':1,'case_names':['normal_exit','timeout'],'frameworks_allowed':[], 'gpu_use_allowed':False,'models_allowed':0,'optimizer_updates':0,'phase':'CE8_OWNED_NON_ML_OS_FIXTURE','physical_root_seconds':60,'scientific_inputs_allowed':[],'single_nested_tail_seconds':5}
review={'packet_ref':pref,'source_freeze_ref':fref,'owner_thread_id':namespace['OWNER'],'source_review_status':'PASS_FOR_THIS_BOUNDED_OS_FIXTURE','native_evidence_status':'NOT_RUN','reviewed_source_refs':source,'target_manifest_ref':target,'complete_all7_accounting_review':'PASS_EXACT_UNCHANGED_CAPS','preparation_bound':dict(zip(namespace['FIELDS'],FakeCost.PREPARATION_BOUNDS)),'preparation_phase_status':'PASS_PROSPECTIVE_BOUND','authorization_ref':namespace['AUTH'],'automatic_tool_policy_overridden':False,'scope':scope}
test('exact synthetic review accepted with native proof as output',lambda:namespace['review_binding'](packet,pref,freeze,fref,review))
for key,value in [('packet_ref',{}),('source_freeze_ref',{}),('reviewed_source_refs',{}),('authorization_ref',{}),('scope',{}),('native_evidence_status','PASS'),('automatic_tool_policy_overridden',True),('complete_all7_accounting_review','UNPROVED')]:
 bad=copy.deepcopy(review);bad[key]=value
 test('review refuses '+key,lambda bad=bad:namespace['review_binding'](packet,pref,freeze,fref,bad),True)
test('missing review refuses',lambda:namespace['review_binding'](packet,pref,freeze,fref,{}),True)

issuer=ast.parse((ROOT/'issue_grant.py').read_text())
entry=next(n for n in issuer.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_and_issue')
assert isinstance(entry.body[0],ast.Expr) and isinstance(entry.body[0].value,ast.Call)
first=entry.body[0]
class FakeOwner:
 _closed=False;_root_handoff=False;fatal=False
class FakeA:
 _ENTRY_OWNER_BOUND=True
 _owner=FakeOwner()
 require=staticmethod(namespace['require'])
 @classmethod
 def preparation_owner(cls):return cls._owner
class FakeEntryCost:
 @staticmethod
 def entry_preparation_owner():return FakeA._owner
guard=compile(ast.Module(body=[first],type_ignores=[]),'mechanical_first_entry_guard','exec')
def entry_check():exec(guard,{'a':FakeA,'cost':FakeEntryCost});return True
test('exact live original ENTRY owner accepted mechanically',entry_check)
FakeA._ENTRY_OWNER_BOUND=False;test('cold imported ENTRY denied',entry_check,True);FakeA._ENTRY_OWNER_BOUND=True
for field in ('_closed','_root_handoff','fatal'):
 setattr(FakeA._owner,field,True);test('ENTRY refuses '+field,entry_check,True);setattr(FakeA._owner,field,False)
original_entry_method=FakeEntryCost.entry_preparation_owner
FakeEntryCost.entry_preparation_owner=staticmethod(lambda:object())
test('replacement PREP owner denied',entry_check,True)
FakeEntryCost.entry_preparation_owner=original_entry_method

diff=''.join(line for name in NAMES for line in difflib.unified_diff((BASE/name).read_text().splitlines(True),(ROOT/name).read_text().splitlines(True),fromfile=str(BASE/name),tofile=str(ROOT/name)))
with (ROOT/'EXACT_SOURCE_DIFF.patch').open('x') as stream:stream.write(diff)
source_delta=sum(ref(ROOT/n)['bytes']-ref(BASE/n)['bytes'] for n in NAMES)
publish('MECHANICAL_RELEASE_EVIDENCE.json',{'schema':'CE8-AUTHORIZED-RELEASE-MECHANICAL-62','classification':'AST_EXTRACTED_SYNTHETIC_GATES_ONLY_NOT_NATIVE_EVIDENCE','checks':checks,'all_pass':True,'candidate_module_imports':0,'native_calls':0,'grants_issued':0,'eligibility_written':False,'optimizer_updates':0})
publish('RELEASE_COST_DELTA.json',{'schema':'CE8-AUTHORIZED-RELEASE-SOURCE-DELTA-62','source_delta_bytes':source_delta,'new_authorization_bytes':ref(Path(namespace['AUTH']['path']))['bytes'],'historical_authorization_bytes':1581,'authorization_input_delta_bytes':545,'authorization_new_reads':0,'authorization_new_hash_attempts':0,'authorization_replaces_existing_paid_read_hash_parse':True,'fixed_AUTH_read_ceiling_bytes':65537,'new_source_SHA_attempts':0,'new_source_read_groups':0,'source_raw_PREP_custody_increase_bytes':source_delta,'fixed_metadata_materializer_source_reads_unchanged':True,'new_authorization_predicates_and_review_scope_comparison':'Complete finite deterministic recipe inside existing validate_packet/module/review PREP source groups; independent exact metadata and code/literal catalogue composition still required. No zero-cost Native/library claim.','registered_caps_changed':False,'prospective_delta_is_full_accounting_pass':False})
publish('GENERATOR_FAILURE_PRESERVATION.json',{'schema':'CE8-BUILDER-FAILURE-PRESERVED-62','builder_ref':ref(ROOT.parent/'build_authorized_qualification_source_62.py'),'error':'TypeError: sequence item 0: expected str instance, generator found','location':'final diff aggregation after source edits and all14 syntax checks','failed_process_exit_code':1,'correction':'Separate create-only finalizer flattens diff iterators; original builder and source files preserved','native_execution':False,'runtime_metadata_created':False})
for forbidden in ('PACKET.json','NATIVE_SOURCE_FREEZE.json','MATERIALIZATION_ACCOUNTING.json','ENTRY_IDENTITIES_V2.json','TARGET_ARTIFACT_INPUT.json','INDEPENDENT_REVIEW_DECISION.json','OWNER_DISPATCH_READINESS.json','FIXTURE_GRANT.json','native-one-attempt'):
 assert not (ROOT/forbidden).exists(),forbidden
publish('OFFLINE_SOURCE_FREEZE.json',{'schema':'CE8-AUTHORIZED-QUALIFICATION-RELEASE-OFFLINE-FREEZE-62','classification':'SOURCE_PROPOSAL_PENDING_INDEPENDENT_REVIEW_NO_ADMIT','source_head':'eee2e0bd31854ea207b231d65eedd4c1c94e1011','source_refs':{n:ref(ROOT/n) for n in NAMES},'support_refs':{n:ref(ROOT/n) for n in ('finalize_authorized_source.py','EXACT_SOURCE_DIFF.patch','MECHANICAL_RELEASE_EVIDENCE.json','RELEASE_COST_DELTA.json','GENERATOR_FAILURE_PRESERVATION.json')},'verified_static_declarations':declared,'previous_source_freeze_ref':ref(BASE/'OFFLINE_SOURCE_FREEZE.json'),'human_authorization_ref':namespace['AUTH'],'old_unconditional_quarantine_replaced_by_exact_live_entry_guard':True,'all_downstream_review_custody_resources_timer_process_and_lease_guards_retained':True,'caps_changed':False,'source_branch_changed':False,'runtime_metadata_created':False,'eligibility_written':False,'grants_issued':0,'native_calls':0,'actor_runs':0,'models_loaded':0,'optimizer_updates':0,'dispatch_admitted':False,'qualification_pass':False,'complete_all7_fit_proven':False})
print(json.dumps({'freeze_ref':ref(ROOT/'OFFLINE_SOURCE_FREEZE.json'),'mechanical_check_count':len(checks),'source_delta_bytes':source_delta},sort_keys=True))
