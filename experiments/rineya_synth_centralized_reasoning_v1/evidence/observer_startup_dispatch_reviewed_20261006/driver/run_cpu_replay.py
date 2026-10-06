from pathlib import Path
import builtins,hashlib,json,sys,time,unittest
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 before={p.name:sha(p) for p in HERE.glob('*.py')};attempts=[];original=builtins.__import__;started=time.monotonic()
 def imports(name,*args,**kwargs):
  if name.split('.',1)[0] in {'ctypes','torch','psutil','native_process','native_nvml','native_supervisor','numpy','transformers','safetensors'}:
   attempts.append(name);raise AssertionError('actual native/framework import prohibited: '+name)
  return original(name,*args,**kwargs)
 builtins.__import__=imports
 try:
  with (HERE/'DRIVER_CPU_REPLAY_001.txt').open('x',encoding='utf-8') as out:result=unittest.TextTestRunner(stream=out,verbosity=2).run(unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py'))
 finally:builtins.__import__=original
 after={n:sha(HERE/n) for n in before}
 v={'kind':'OBSERVER_LAUNCH_DRIVER_SOURCE_CPU_REPLAY_V1','tests':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'skips':len(result.skipped),'seconds':time.monotonic()-started,'source_before':before,'source_after':after,'sources_unchanged':before==after,'forbidden_actual_import_attempts':attempts,'native_API_executed':False,'observer_DLL_loaded_or_registered':False,'native_PASS':False,'native_authority_issued':False}
 v['sha256']=hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 with (HERE/'DRIVER_CPU_REPLAY_001.json').open('x',encoding='utf-8') as out:out.write(json.dumps(v,indent=2,sort_keys=True)+'\n')
 print(json.dumps(v));return not(result.wasSuccessful() and before==after and not attempts and not result.skipped)
if __name__=='__main__':raise SystemExit(main())
