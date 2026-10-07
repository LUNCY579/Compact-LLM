"""Pinned original heartbeat/RAM/Job guard with Torch view permanently absent."""
from pathlib import Path
import ast,hashlib,time
PACKAGE=next(p for p in Path(__file__).resolve().parents if p.name=='rineya_synth_centralized_reasoning_v1')
SOURCE=PACKAGE/'inference_qual_worker.py'
PIN='1172838c31c1b9440337707826573012e06f9a6a26b606a62a43976d19e99f1e'
def require(value,message):
    if not value:raise ValueError(message)
def bind(output,owner,started,*,owned_resources,heartbeat_reader,available,monotonic=time.monotonic,wall=time.time):
    raw=SOURCE.read_bytes();require(len(raw)<=16*1024**2 and hashlib.sha256(raw).hexdigest()==PIN,'unchanged original worker resource guard source')
    functions=[n for n in ast.parse(raw,str(SOURCE)).body if isinstance(n,ast.FunctionDef) and n.name=='make_guard'];require(len(functions)==1,'one original guard definition')
    ns={'Path':Path,'require':require,'GIB':1024**3,'time':time,'owned_job_resources':owned_resources,'read_heartbeat':heartbeat_reader}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),ns)
    require(all(callable(v) for v in (owned_resources,heartbeat_reader,available,monotonic,wall)),'explicit import-stage resource suppliers')
    return ns['make_guard'](output,owner,started,lambda:None,owned_resources=owned_resources,heartbeat_reader=heartbeat_reader,available=available,monotonic=monotonic,wall=wall)
