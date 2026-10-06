from pathlib import Path
import builtins,hashlib,json,sys,time,unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 sources={p.name:sha(p) for p in HERE.glob('*.py')};original=builtins.__import__;attempts=[]
 def imports(name,*args,**kwargs):
  if name.split('.',1)[0] in {'ctypes','torch','numpy','psutil','native_worker','native_process','native_nvml','transformers','safetensors'}:
   attempts.append(name);raise AssertionError('actual native/framework import prohibited: '+name)
  return original(name,*args,**kwargs)
 builtins.__import__=imports;start=time.monotonic()
 try:
  with (HERE/'STARTUP_MEASUREMENT_CPU_REPLAY_002.txt').open('x',encoding='utf-8') as out:
   result=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py'))
 finally:builtins.__import__=original
 after={n:sha(HERE/n) for n in sources}
 v={'kind':'STAGED_OBSERVER_STARTUP_MEASUREMENT_SOURCE_CPU_REPLAY','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'seconds':time.monotonic()-start,'source_before':sources,'source_after':after,'sources_unchanged':sources==after,'forbidden_actual_import_attempts':attempts,'native_observer_DLL_loaded_or_registered':False,'native_PASS':False,'native_authority_issued':False,'current_host_inventory_qualified':False,'complete_native_dispatch_registration':False}
 v['sha256']=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 with (HERE/'STARTUP_MEASUREMENT_CPU_REPLAY_002.json').open('x',encoding='utf-8') as out:out.write(json.dumps(v,sort_keys=True,indent=2)+'\n')
 print(json.dumps(v));return not(result.wasSuccessful() and sources==after and not attempts and not result.skipped)
if __name__=='__main__':raise SystemExit(main())
