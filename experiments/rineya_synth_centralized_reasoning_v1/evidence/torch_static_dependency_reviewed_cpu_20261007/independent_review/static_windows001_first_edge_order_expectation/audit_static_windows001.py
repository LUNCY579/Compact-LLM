"""Static PE obligations and two selected Windows identities, no DLL loads."""
from pathlib import Path
import ast, hashlib, json, os, stat, struct, sys, types
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;PACKAGE=next(p for p in WORK.parents if p.name=='rineya_synth_centralized_reasoning_v1');PARSER=PACKAGE/'evidence/observer_startup_dispatch_reviewed_20261006/passive_pe_candidates/passive_pe_startup_candidates.py';READER=PACKAGE/'evidence/passive_runtime_inventory_reviewed_cpu_20261006/current_reader_002/inventory.py';BASIS=PACKAGE/'artifacts/runtime_refresh_work/TENSOR_RUNTIME_BYTE_CANDIDATE001.json'
FILES=[WORK/n for n in ('TORCH_STATIC_EFFECT_OBLIGATIONS001.json','TWO_WINDOWS_DLL_CANDIDATES001.json','ADDITIVE_WINDOWS_RUNTIME_CANDIDATE001.json','LIBRARY_EXPECTATION_INPUTS001.json','torch_static_effects.py','SOURCE_PRESERVATION004.json')]+[PARSER,READER,BASIS]
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
before=pins()
def selected(path):v=json.loads(path.read_text(encoding='utf-8'));assert v['sha256']==digest({k:x for k,x in v.items() if k!='sha256'});return v
s=selected(WORK/'TORCH_STATIC_EFFECT_OBLIGATIONS001.json');w=selected(WORK/'TWO_WINDOWS_DLL_CANDIDATES001.json');a=selected(WORK/'ADDITIVE_WINDOWS_RUNTIME_CANDIDATE001.json');i=selected(WORK/'LIBRARY_EXPECTATION_INPUTS001.json');b=selected(BASIS)
assert before[str(WORK/'TORCH_STATIC_EFFECT_OBLIGATIONS001.json')]['sha256']=='1b41e376976ab66f6e9a75030acb01faecef417d40d560cc94cd04ff033d2d3f';assert before[str(WORK/'TWO_WINDOWS_DLL_CANDIDATES001.json')]['sha256']=='9f08a476bd5668a3d9b43e93aa9cc5abfe401fccebd50f8aa371f27e3aa1b789';assert before[str(WORK/'ADDITIVE_WINDOWS_RUNTIME_CANDIDATE001.json')]['sha256']=='58ba3e85d864d2aea24042486742966c5155939b902e971dd3f2f6a9742e31de'
assert s['parser_source']['sha256']==before[str(PARSER)]['sha256'];assert s['input_reference']['file_sha256']==before[str(WORK/'LIBRARY_EXPECTATION_INPUTS001.json')]['sha256']
records=s['selected_pe_records'];assert len(records)==38 and sum(x['descriptor']['bytes'] for x in records.values())==s['bytes_read']==3763222184;assert len(s['literal_dependency_edges'])==244
expected_seeds=i['explicit_torch_lib_dll_leaves']+[p for p in i['potential_unpreloaded_extension_leaves'] if Path(p).parent.name=='torch'];assert set(expected_seeds)==set(records)
byname={}
for p in i['library_files']:byname.setdefault(Path(p).name.casefold(),[]).append(p)
edges=[];initial=set(i['expected_initial_loaded_libraries'])
for p,row in records.items():
    assert row['descriptor']==i['library_files'][p] and row['machine']==34404 and row['optional_magic']==523
    for imported in row['imports']:
        candidates=byname.get(imported['name'].casefold(),[]);kind='API_SET_UNRESOLVED' if imported['name'].casefold().startswith(('api-ms-','ext-ms-')) else 'SELECTED_UNIQUE_LITERAL_NAME' if len(candidates)==1 else 'SELECTED_AMBIGUOUS_LITERAL_NAME' if candidates else 'NOT_IN_SELECTED_BINARY_MAP'
        edges.append({'from':p,**imported,'classification':kind,'selected_candidates':candidates,'candidate_already_in_observer009_baseline':bool(candidates) and all(x in initial for x in candidates)})
assert edges==s['literal_dependency_edges'];required=sorted({p for e in edges if e['kind']=='import' and e['classification']=='SELECTED_UNIQUE_LITERAL_NAME' for p in e['selected_candidates']});assert required==s['ordinary_import_unique_selected_candidate_paths'] and len(required)==34
unresolved=[e for e in edges if e['classification']!='SELECTED_UNIQUE_LITERAL_NAME' or e['kind']=='delay_import'];assert unresolved==s['unresolved_or_delay_edges'] and len(unresolved)==117
assert len(i['explicit_torch_lib_dll_leaves'])==37;assert len(s['new_vs_observer009_candidate_paths'])==24
assert set(w['files'])=={r'C:\Windows\System32\msvcp140.dll',r'C:\Windows\System32\nvcuda.dll'};assert sorted(d['bytes'] for d in w['files'].values())==[643512,3807352];assert sum(d['bytes'] for d in w['files'].values())==w['actual_hashed_bytes']==4450864
assert a['files']=={**b['files'],**w['files']} and not set(b['files'])&set(w['files']);assert len(a['files'])==a['file_count']==8716 and a['new_windows_files']==w['files'];assert all(d['role']=='windows_system_binary' for d in w['files'].values());assert a['observed_cache_pins_unchanged']==59
# Compile only the bounded passive parser definition and validate exact small
# selected Torch bytes plus both Windows candidates. No native module import.
def require(v,msg):
    if not v:raise ValueError(msg)
tree=ast.parse(PARSER.read_bytes());fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='pe_imports');env={'struct':struct,'require':require};exec(compile(ast.Module(body=[fn],type_ignores=[]),str(PARSER),'exec'),env)
reader=types.ModuleType('independent_static_reader');reader.__file__=str(READER);exec(compile(READER.read_bytes(),str(READER),'exec'),reader.__dict__)
sample=dict(w['files'])
for p,row in sorted(records.items(),key=lambda x:x[1]['descriptor']['bytes'])[:3]:sample[p]=row['descriptor']
probe=reader._collect(sorted(sample),sample);assert {p:{k:d[k] for k in ('bytes','sha256','role')} for p,d in probe['files'].items()}==sample
parsed_samples={}
for text,d in sample.items():
    p=Path(text);raw=p.read_bytes();assert len(raw)==d['bytes'] and hashlib.sha256(raw).hexdigest()==d['sha256'];parsed=env['pe_imports'](raw);parsed_samples[text]={'imports':len(parsed['imports']),'machine':parsed['machine'],'magic':parsed['optional_magic']}
    if text in records:assert parsed=={k:records[text][k] for k in ('machine','optional_magic','imports')}
for bad in (b'',b'MZ'+b'\0'*62):
    try:env['pe_imports'](bad)
    except ValueError:pass
    else:raise AssertionError('malformed PE admitted')
for receipt in (s,w,a):
    for key in ('native_runtime_qualified','native_launch_authorized','native_authority_issued','native_PASS'):assert receipt[key] is False
assert s['actual_loader_resolution_observed'] is s['actual_per_boundary_effects_observed'] is False;assert a['API_set_resolution_observed'] is a['exact_per_boundary_effects_observed'] is False
after=pins();assert before==after
receipt={'kind':'INDEPENDENT_STATIC_PE_TWO_WINDOWS_CANDIDATES_AUDIT001','source_pins_before':before,'source_pins_after':after,'source_pins_unchanged':True,'selected_PE_receipt_records':38,'receipt_bytes_previously_hashed':3763222184,'independent_edge_classifications':len(edges),'ordinary_unique_candidates':34,'unresolved_or_delay_edges':117,'new_candidates_vs_observer_baseline':24,'explicit_torch_DLL_source_leaves':37,'actual_probe_files':len(sample),'actual_probe_hashed_bytes':probe['actual_hashed_bytes'],'actual_two_Windows_bytes':4450864,'actual_parsed_samples':parsed_samples,'malformed_PE_refusals':2,'tensor_runtime_additive_map_count':8716,'tensor8714_basis_descriptors_retained':True,'historical_observer8716_map_is_distinct':True,'bounded_parser_source_inspected':{'sections':96,'import_descriptors_per_directory':4096,'ASCII_name_bytes':260,'field_spans_checked':True},'full_3_76GB_hash_not_repeated':True,'DLL_load_or_driver_change_performed':False,'actual_loader_resolution_observed':False,'actual_per_boundary_effects_observed':False,'native_runtime_qualified':False,'native_authority_issued':False};receipt['sha256']=digest(receipt)
with (HERE/'INDEPENDENT_STATIC_WINDOWS001.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
print(canonical({'PE_records':38,'edges':len(edges),'probe_files':len(sample),'probe_bytes':probe['actual_hashed_bytes'],'body_sha256':receipt['sha256']}))
