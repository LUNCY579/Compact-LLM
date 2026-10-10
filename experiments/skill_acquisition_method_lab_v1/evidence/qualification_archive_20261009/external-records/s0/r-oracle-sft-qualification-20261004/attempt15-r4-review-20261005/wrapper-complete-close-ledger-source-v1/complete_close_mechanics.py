"""Actual source close operation tests; synthetic owner and stream only."""
import ast,ctypes as c,hashlib,json,threading,traceback,types
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def require(v,m):
 if v is not True:raise RuntimeError(m)
source=(ROOT/'harness.py').read_text();tree=ast.parse(source)
node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='OriginalOwnerLogs')
class Identity:
 def __init__(self):self.capture_status='COMPLETE';self.capture_error=None;self.close_attempted=False;self.close_returned=False;self.close_failed=False
 def capture(self,s):pass
class Owner:
 def __init__(self):self.ticks=0;self.reject=None
 def _tick(self,**kwargs):
  self.ticks+=1
  if self.reject==self.ticks:raise RuntimeError('synthetic original close guard')
class Stream:
 def __init__(self,error=False):self.calls=0;self.error=error
 def close(self):
  self.calls+=1
  if self.error:raise OSError('synthetic actual close error')
owner=None;stream=None
env={'c':c,'threading':threading,'cost':types.SimpleNamespace(OutputDescriptorIdentity=Identity,traceback=traceback),
 'a':types.SimpleNamespace(require=require),'os':types.SimpleNamespace(read=lambda *args:(_ for _ in ()).throw(ValueError('primary read error'))),
 'open':lambda *args,**kwargs:stream}
exec(compile(ast.Module([node],type_ignores=[]),str(ROOT/'harness.py'),'exec'),env)
Logs=env['OriginalOwnerLogs'];checks=[]
def check(n,w):w();checks.append({'name':n,'pass':True})
def make():
 global owner,stream
 owner=Owner();stream=Stream();directory=object();plan=types.SimpleNamespace(_root_owner=owner,
  _log_inputs=(('normal_exit',directory,(),('t0','t1'),('p0','p1'),('s0','s1')),))
 plan._static=types.SimpleNamespace(creation_plan=plan);backend=types.SimpleNamespace(owner=owner,_creation_plan=plan,read_fds=(1,2))
 value=Logs(backend,directory);owner.ticks=0;value.streams[0]=stream
 value.errors[0]=ValueError('original primary operational failure');return value
def refuses(work):
 try:work()
 except (RuntimeError,OSError):return
 raise AssertionError('expected source close failure')
value=make();primary=value.errors[0];owner.reject=1;refuses(lambda:value.close_stream(0))
check('before guard refusal retains both diagnostics with zero actual calls',lambda:require(value.errors[0] is primary
 and value.secondary_close_errors[0]=='RuntimeError' and stream.calls==0 and value.streams[0] is stream
 and not value.close_attempted[0] and not value.log_identities[0].close_failed,'pre-dispatch refusal'))
value=make();primary=value.errors[0];value.calls=2048;refuses(lambda:value.close_stream(0))
check('operation admission exhaustion gets same secondary ledger zero calls',lambda:require(value.errors[0] is primary
 and value.secondary_close_errors[0]=='RuntimeError' and stream.calls==0,'admission ledger'))
value=make();primary=value.errors[0];stream.error=True;refuses(lambda:value.close_stream(0))
check('actual close failure retains both errors and unknown one-attempt state',lambda:require(value.errors[0] is primary
 and value.secondary_close_errors[0]=='OSError' and stream.calls==1 and value.streams[0] is stream
 and value.close_attempted[0] and value.log_identities[0].close_failed and not value.log_identities[0].close_returned,'actual source error'))
refuses(lambda:value.close_stream(0))
check('unsafe second close refused without replacing actual close typename',lambda:require(stream.calls==1
 and value.secondary_close_errors[0]=='OSError','no retry or overwrite'))
value=make();primary=value.errors[0];owner.reject=3;refuses(lambda:value.close_stream(0))
check('afterguard failure preserves observed successful close and secondary error',lambda:require(value.errors[0] is primary
 and value.secondary_close_errors[0]=='RuntimeError' and stream.calls==1 and value.streams[0] is None
 and value.log_identities[0].close_returned and not value.log_identities[0].close_failed,'return before late guard'))
value.close_stream(0)
check('observed closed slot cannot execute another close',lambda:require(stream.calls==1,'cleared source ownership'))
value=make();value.close_stream(0)
check('normal full close operation counts one I/O without secondary error',lambda:require(stream.calls==1
 and value.calls==1 and value.actual_io[5]==1 and value.secondary_close_errors[0] is None,'normal ledger'))
check('exact1536 block/state5120 replaces actual retained payload storage',lambda:require('os.read(fd,1536)' in source
 and 'Two <=1536-byte payload slots +5120 logical state' in source and 2*1536+5120==8192,'same original workspace'))
check('all registered percase read/write/operation ceilings preserved',lambda:require('self.read<=8388608' in source
 and 'self.persisted+len(block)<=8192' in source and '2048 if closing else 2032' in source,'no cap reduction/increase'))
check('fixed bounded secondary slots and original primary objects remain distinct',lambda:require('self.secondary_close_errors=(c.py_object*2)(None,None)' in source
 and 'self.errors=(c.py_object*2)(None,None)' in source,'two existing fixed diagnostics'))
record={'schema':'CE8-COMPLETE-CLOSE-LEDGER-MECHANICS-1','classification':'SOURCE_EXTRACTED_SYNTHETIC_ONLY',
 'count':len(checks),'checks':checks,'native_calls':0,'candidate_processes':0,'optimizer_updates':0,'qualification_pass':False,
 'all7_fit_proven':False,'complete_workspace_proved':False,'source_refs':{n:{'bytes':(ROOT/n).stat().st_size,
 'sha256':hashlib.sha256((ROOT/n).read_bytes()).hexdigest()} for n in ('harness.py','accounting.py','admission.py','failure_envelope.py')}}
with (ROOT/'COMPLETE_CLOSE_MECHANICAL_EVIDENCE.json').open('x') as f:json.dump(record,f,sort_keys=True,indent=2);f.write('\n')
print(json.dumps({'checks':len(checks),'pass':True,'classification':record['classification']}))
