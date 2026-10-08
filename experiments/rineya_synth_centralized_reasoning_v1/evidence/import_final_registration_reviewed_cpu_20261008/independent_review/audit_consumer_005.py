"""Independent absolute consumer/preview joins; metadata only, no rerun."""
from pathlib import Path
from copy import deepcopy
import hashlib,json
HERE=Path(__file__).resolve().parent;P=HERE.parents[1];I=P/'artifacts/import_exact_registration_work';C=P/'artifacts/import_final_review_consumer_work_005_absolute'
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode()).hexdigest()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
    v=json.loads(p.read_bytes());assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});return v
paths=[I/'join_actual_review.py',I/'IMPORT_OBSERVATION_CONFIG_005.json',I/'IMPORT_OBSERVATION_RUNTIME_005.json',I/'IMPORT_OBSERVATION_REGISTRATION_005.json',I/'EXACT_IMPORT_RUN_RECIPE005.json',HERE/'FRESH_IMPORT_CURRENT_BYTE_REVIEW005.json',C/'ROOT_REVIEWED_IMPORT_RUNTIME_BYTE_ROSTER_SOURCE_CACHE_CLOSURE.json',C/'ACTUAL_BYTE_REVIEW_PROVENANCE.json',C/'CPU_PROSPECTIVE_APPROVED_SCHEMA_FIXTURE.json',C/'CPU_FULL_QUALIFIER_ACTUAL_WINDOW_REPLAY.json']
before={str(p):sha(p) for p in paths}
config,runtime,registration,recipe,window,consumer,provenance,preview,replay=[read(p) for p in paths[1:]]
assert before[str(paths[0])]=='5fdb6499724fa47cb11b8ce238b646b1a2ad1d50268bd4944292230f6d7d933b'
assert before[str(paths[5])]=='2db32a4e7e9677f66bacf34ad5ae35deec7b2700b579befa5dac78406579ca6f'
assert before[str(paths[6])]=='6998734c076989c0802ac586824700db6516149c7f14ce3e3e1a1618affc0cda'
def ref(reference):
    p=Path(reference['path']);assert p.is_absolute() and str(p)==reference['path'] and sha(p)==reference['file_sha256'] and read(p)['sha256']==reference['body_sha256']
for reference in (replay['actual_window_reference'],replay['consumer_reference'],replay['preview_reference'],replay['provenance_reference'],provenance['actual_window_reference'],provenance['exact_consumer_reference'],preview['exact_review_consumer_reference'],preview['actual_byte_window_provenance_reference']):ref(reference)
expected={'kind':'ROOT_REVIEWED_IMPORT_RUNTIME_BYTE_ROSTER_SOURCE_CACHE_CLOSURE','source_commit':config['source_commit'],'source_inventory_sha256':digest(config['source_files']),'runtime_closure_sha256':digest(runtime['files']),'rosters_sha256':digest(runtime['rosters']),'cache_policy_sha256':digest(runtime['cache_policy']),'library_search_policy_sha256':digest(runtime['library_search_policy']),'measurement_policy_sha256':digest(runtime['measurement_policy']),'measurement_grammar_sha256':digest(runtime['measurement_grammar']),'complete_current_byte_review':True}
assert consumer=={**expected,'sha256':digest(expected)}
assert window['source_files']==config['source_files'] and window['runtime_files']==runtime['files']
assert window['rosters_before']==window['rosters_after']==runtime['rosters']
assert provenance['actual_window_complete_maps_rosters_and67_cache_pairs_joined'] is True and provenance['independent_review_reference'] is None
assert replay['actual_window_reference']==provenance['actual_window_reference'] and replay['consumer_reference']==provenance['exact_consumer_reference']==preview['exact_review_consumer_reference']
assert replay['descriptor_calls']==8921 and replay['roster_calls']==18 and replay['source_count']==196 and replay['runtime_count']==8725
assert replay['result']=={'kind':'IMPORT_RUNTIME_BYTE_ROSTER_QUALIFICATION_ONLY','source_inventory_sha256':digest(config['source_files']),'runtime_inventory_sha256':digest(runtime['files']),'current_module_execution_proven':False,'native_effects_or_authority_issued':False}
assert replay['selected_public_qualifier']=={'path':str(P/'import_runtime_qualification.py'),'file_sha256':sha(P/'import_runtime_qualification.py')}
prospective_config=deepcopy(config);prospective_config.pop('sha256');prospective_config['native_launch_authorized']=True;prospective_config['sha256']=digest(prospective_config)
assert prospective_config==preview['proposed_config'] and preview['proposed_runtime']==runtime
prospective_registration=deepcopy(registration);prospective_registration.pop('sha256');prospective_registration.update(native_launch_authorized=True,config_body_sha256=prospective_config['sha256'],qualification_review_path=replay['consumer_reference']['path'],qualification_review_file_sha256=replay['consumer_reference']['file_sha256'],qualification_review_body_sha256=replay['consumer_reference']['body_sha256']);prospective_registration['sha256']=digest(prospective_registration)
assert prospective_registration==preview['proposed_registration']
assert preview['prospective_expected_head_CPU_placeholder']=='b'*40 and preview['real_owner_permission_required'] is preview['real_dispatch_commit_required'] is True
assert preview['actual_closed_recipe_reference']==recipe['packet_references']
assert config['native_launch_authorized'] is registration['native_launch_authorized'] is False and config['owner_approval'] is registration['owner_approval'] is None
for v in (preview,replay,provenance):assert v['native_launch_authorized'] is v['native_authority_issued'] is False
assert replay['native_context_or_gate_called'] is replay['real_dispatch_Git_join_performed'] is preview['native_context_or_gate_called'] is False
after={str(p):sha(p) for p in paths};assert before==after
result={'kind':'INDEPENDENT_FINAL005_ABSOLUTE_CONSUMER_PREVIEW_METADATA_REVIEW001','consumer_file_sha256':before[str(paths[6])],'consumer_body_sha256':consumer['sha256'],'actual_window_file_sha256':before[str(paths[5])],'actual_window_body_sha256':window['sha256'],'absolute_all_reference_joins_valid':True,'consumer_exact_original_schema_valid':True,'prospective_CPU_schema_changes_only_authorized_fields':True,'reported_complete_descriptor_calls':8921,'reported_complete_roster_calls':18,'closed005_unchanged_and_no_owner_permission':True,'pins_before':before,'pins_after':after,'pins_stable':True,'native_gate_context_framework_or_runtime_hash_called':False,'full_qualifier_replay_not_repeated':True,'review_source_sha256':sha(Path(__file__))}
result['sha256']=digest(result)
with (HERE/'INDEPENDENT_FINAL005_CONSUMER001.json').open('x',encoding='utf-8',newline='\n') as out:json.dump(result,out,sort_keys=True,indent=2);out.write('\n')
print(canonical({'body_sha256':result['sha256'],'consumer_absolute_reference_valid':True,'native_authority_issued':False}))
