from pathlib import Path
import json,hashlib,os
P=Path(__file__).resolve().parents[2];S=P/'artifacts/prefix_input_owned_execution_work';D=P/'evidence/library_boundary_reviewed_cpu_20261005'
D.mkdir(exist_ok=False)
files={f.name:f for f in S.glob('*.py')}
for n in ['LIBRARY_BOUNDARY_CPU_CONTROLS_001.json','LIBRARY_BOUNDARY_CPU_CONTROLS_001.txt','LIBRARY_BOUNDARY_CPU_CONTROLS_003.json','LIBRARY_BOUNDARY_CPU_CONTROLS_003.txt','ROOT_LIBRARY_SELECTOR_CPU_007.json','ROOT_LIBRARY_SELECTOR_CPU_007.txt','ROOT_LIBRARY_REGISTRY_REGRESSION_008.json','ROOT_LIBRARY_REGISTRY_REGRESSION_008.txt']:
 files['history/root_and_builder/'+n]=S/n
for n in ['pre_library_boundary_001','LIBRARY_BOUNDARY_SOURCE_FROZEN_003','LIBRARY_EVENT_DIAGNOSTIC_002','LIBRARY_MANIFEST_DIAGNOSTIC_001','LIBRARY_MANIFEST_SELECTOR_DIAGNOSTIC_002','LIBRARY_MANIFEST_INACTIVE_GUARD_DIAGNOSTIC_003','LIBRARY_MANIFEST_GUARD_IDENTITY_DIAGNOSTIC_004','LIBRARY_MANIFEST_GUARD_SELECTOR_REGRESSION_DIAGNOSTIC_005','LIBRARY_MANIFEST_PRE_REGISTRY_006','pre_library_framework_integration_004','pre_framework_binding_identity_fix_005','pre_selector_method_repair_006','pre_installer_registry_008','pre_method_code_binding_010']:
 for f in (S/n).rglob('*'):
  if f.is_file() and '__pycache__' not in f.parts:files['history/'+n+'/'+f.relative_to(S/n).as_posix()]=f
rows={}
for name,source in files.items():
 raw=source.read_bytes();target=D/name;target.parent.mkdir(parents=True,exist_ok=True)
 with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 assert target.read_bytes()==raw
 rows[name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'source_path':str(source)}
v={'kind':'ROOT_SOURCE_DLL_BOUNDARY_CPU_PRESERVATION','files':rows,'fixture_only':True,'native_PASS':False,'native_authority_issued':False,'native_qualification_proven':False,'current_host_observer_bootstrap_DLL_bytes_resources_proven':False,'combined_replay_pending':True}
v['sha256']=hashlib.sha256(json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False).encode('ascii')).hexdigest()
with (D/'ROOT_PRESERVATION.json').open('x',encoding='utf-8') as f:f.write(json.dumps(v,sort_keys=True,indent=2)+'\n')
print(json.dumps({'preserved':len(rows),'sha256':v['sha256'],'path':str(D)}))
