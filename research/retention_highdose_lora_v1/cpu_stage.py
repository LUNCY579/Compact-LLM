"""Bound a CPU materialization with the existing foreground watchdog and lease recovery."""
from pathlib import Path
import argparse
import importlib.util
import os
import runpy
import sys
import time

PILOT=Path('C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917')
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(PILOT))
from experiments.model_quality_gpu_v1.integrity import file_hash,read_json,write_new
from experiments.model_quality_gpu_v1.supervisor import bounded_child,claim_worker

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--_worker',action='store_true')
    ap.add_argument('--expected-script-sha')
    ap.add_argument('arguments',nargs=argparse.REMAINDER)
    a=ap.parse_args()
    arguments=a.arguments[1:] if a.arguments[:1]==['--'] else a.arguments
    if not arguments or arguments[0] not in ('lora','scalar'):
        raise ValueError('Only the two registered materializations are supported')
    script=HERE/'materialize.py'
    if a._worker:
        claim_worker(a.out)
        started=time.monotonic()
        code=1
        failure=None
        try:
            if file_hash(script)!=a.expected_script_sha:
                raise ValueError('Materializer changed after dispatch')
            sys.argv=[str(script),*arguments]
            try:
                runpy.run_path(str(script),run_name='__main__')
                code=0
            except SystemExit as exc:
                code=0 if exc.code is None else exc.code if type(exc.code) is int else 1
            if file_hash(script)!=a.expected_script_sha:
                raise ValueError('Materializer changed during execution')
        except Exception as exc:
            failure={'type':type(exc).__name__,'message':str(exc)}
            raise
        finally:
            write_new(a.out/'CPU_TERMINAL.json',{'status':'COMPLETE' if code==0 and failure is None else 'FAILED',
                'returncode':code,'failure':failure,'script_sha256':a.expected_script_sha,
                'wall_seconds':time.monotonic()-started,'model_execution':False})
        return code
    if a.expected_script_sha is not None:
        raise ValueError('private digest override refused')
    command=[sys.executable,str(Path(__file__).resolve()),'--out',str(a.out.absolute()),
             '--expected-script-sha',file_hash(script),'--_worker','--',*arguments]
    code=bounded_child(command,a.out.absolute(),600)
    supervisor=read_json(a.out/'supervisor.json')
    if supervisor['status']=='TIMEOUT_OR_INTERRUPTED':
        wrapper=PILOT/'docs/model-quality/audit-2026-09-17/supervised_probe.py'
        spec=importlib.util.spec_from_file_location('doselora_owned_lease_recovery',wrapper)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        module.recover_own_timedout_lease(a.out.absolute(),module.gpu_lease_path())
    if code==0:
        terminal=read_json(a.out/'CPU_TERMINAL.json')
        if terminal['status']!='COMPLETE' or terminal['returncode']!=0:
            raise ValueError('CPU worker has no successful terminal')
    return code

if __name__=='__main__':
    raise SystemExit(main())
