"""Preserved source CPU replay; never accepts native qualification."""
from pathlib import Path
import builtins,hashlib,json,sys,time,unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
NAMES=['test_outer_native_workflow','test_contract_loader','test_worker_bindings',
 'test_full_workflow_fixture_support','test_full_native_cpu_workflow',
 'test_frozen_bootstrap_loader','test_registered_source_consumption',
 'test_cpu_loader_integration','test_registered_worker_orchestration',
 'test_library_policy','test_framework_library_enclosure','test_library_boundary_manifest']
FORBIDDEN={'torch','transformers','tensorflow','jax','flax','numpy','psutil','native_worker','native_process','native_nvml','ctypes','safetensors'}
def canonical(v):return json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 original=builtins.__import__;attempts=[];inert_imports=[]
 def imports(name,*args,**kwargs):
  if name.split('.',1)[0] in FORBIDDEN:
   module=sys.modules.get(name)
   if name in {'native_worker','native_process','psutil'} and getattr(module,'__CPU_inert_import_fixture__',False) is True and getattr(module,'__file__',None) is None:
    inert_imports.append(name);return module
   attempts.append(name);raise AssertionError('actual native/framework import prohibited: '+name)
  return original(name,*args,**kwargs)
 sources={p.name:sha(p) for p in HERE.glob('*.py')}
 builtins.__import__=imports;start=time.monotonic()
 try:
  with (HERE/'ROOT_COMBINED_DLL_CPU_REPLAY_012.txt').open('x',encoding='utf-8') as log:
   result=unittest.TextTestRunner(stream=log,verbosity=2).run(unittest.defaultTestLoader.loadTestsFromNames(NAMES))
 finally:builtins.__import__=original
 after={n:sha(HERE/n) for n in sources}
 value={'kind':'ROOT_PRESERVED_DLL_BOUNDARY_CPU_REPLAY_012','selected_test_modules':NAMES,
  'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'skips':len(result.skipped),
  'wall_seconds':time.monotonic()-start,'source_before':sources,'source_after':after,
  'source_bytes_unchanged':sources==after,'forbidden_actual_import_attempts':attempts,
  'injected_inert_module_imports':inert_imports,'fixture_only':True,'native_PASS':False,'native_authority_issued':False,
  'current_OS_bootstrap_DLL_observer_runtime_GPU_qualified':False,
  'three_arm_training_transfer_retention_measured':False}
 value['sha256']=hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
 with (HERE/'ROOT_COMBINED_DLL_CPU_REPLAY_012.json').open('x',encoding='utf-8') as f:f.write(json.dumps(value,sort_keys=True,indent=2)+'\n')
 print(json.dumps({'tests':result.testsRun,'errors':len(result.errors),'failures':len(result.failures),'seconds':value['wall_seconds'],'sha256':value['sha256'],'sources_unchanged':sources==after,'forbidden_imports':attempts}))
 return not(result.wasSuccessful() and sources==after and not attempts and not result.skipped)
if __name__=='__main__':raise SystemExit(main())
