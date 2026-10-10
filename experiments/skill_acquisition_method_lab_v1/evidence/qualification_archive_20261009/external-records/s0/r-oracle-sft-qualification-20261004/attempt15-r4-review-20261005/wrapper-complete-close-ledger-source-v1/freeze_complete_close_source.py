"""Create-only freezer; exact final static source references verified."""
import ast,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
names=('admission.py','issue_grant.py','harness.py','backend.py','materialize.py','runtime_custody.py',
 'run_qualification.py','accounting.py','reusable_census.py','v2_consumer_bindings.py',
 'source_envelopes.py','failure_envelope.py','bootstrap_entry.py','metadata_entry.py')
def ref(p):
 b=p.read_bytes();return {'path':str(p),'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
declared=[]
for name in names:
 tree=ast.parse((ROOT/name).read_text())
 for node in ast.walk(tree):
  if not isinstance(node,ast.Dict):continue
  try:item=ast.literal_eval(node)
  except (ValueError,TypeError):continue
  if type(item) is dict and set(item)=={'path','bytes','sha256'} and type(item['path']) is str:
   assert ref(Path(item['path']))==item,(name,item)
   declared.append({'source':str(ROOT/name),'ref':item,'verified':True})
assert len(declared)==4
value={'schema':'CE8-COMPLETE-CLOSE-LEDGER-OFFLINE-FREEZE-1','classification':'OFFLINE_SOURCE_NO_ADMIT',
 'source_head':'eee2e0bd31854ea207b231d65eedd4c1c94e1011','source_refs':{n:ref(ROOT/n) for n in names},
 'support_refs':{n:ref(ROOT/n) for n in ('apply_complete_close.py','complete_close_mechanics.py',
  'COMPLETE_CLOSE_MECHANICAL_EVIDENCE.json','COMPLETE_CLOSE_LEDGER_PLAN.md','freeze_complete_close_source.py')},
 'correction_ref':ref(ROOT.parent/'CLOSE_GUARD_AND_LOG_STATE_CORRECTION_FOR_REVIEW_20261009_60.json'),
 'verified_static_declarations':declared,'all_static_executable_declared_refs_verified':True,
 'previous_source_freeze_ref':ref(ROOT.parent/'wrapper-bounded-failure-ledger-source-v1'/'OFFLINE_SOURCE_FREEZE.json'),
 'previous_target_ref':ref(ROOT/'TARGET_ARTIFACT_INPUT_PREVIOUS.json'),'previous_target_is_current_binding':False,
 'native_calls':0,'actor_runs':0,'models_loaded':0,'optimizer_updates':0,'dispatchable':False,
 'eligibility_written':False,'grant_issued':False,'qualification_pass':False,'all7_fit_proven':False,
 'complete_workspace_proved':False,'original_caps_changed':False,'issuer_unconditional_refusal_preserved':True,
 'internal_log_block_bytes':1536,'log_state_bytes':5120,
 'finite_correction':'Complete original close operation error ledger and actual1536x2/5120 storage replacement',
 'remaining_gaps':['Exact graph/probe/ledger/literal and complete5120 state lifetime source composition',
  'Whole original PREP publisher/metadata source/compile/callback/all-seven logical contract',
  'Complete native ABI/image8192/cleanup16 and fresh source-domain metadata',
  'Actual Native qualification and terminal closure']}
p=ROOT/'OFFLINE_SOURCE_FREEZE.json'
with p.open('x') as f:json.dump(value,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps(ref(p),sort_keys=True))
