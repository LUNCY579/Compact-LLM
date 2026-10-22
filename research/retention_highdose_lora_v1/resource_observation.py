"""Supervised runtime/resource observation; loads no model and runs no forward."""
from pathlib import Path
import argparse
import json
import sys
import tempfile

PILOT=Path('C:/Users/clclc/BusinessProject/ChatBubble-retention-pilot-20260917')
sys.path.insert(0,str(PILOT))
from experiments.model_quality_gpu_v1.integrity import read_json,write_new
from experiments.model_quality_gpu_v1.resources import lease
from experiments.model_quality_gpu_v1.supervisor import bounded_child,claim_worker

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--_worker',action='store_true')
    args=ap.parse_args()
    if args._worker:
        claim_worker(args.out)
        with lease(Path(tempfile.gettempdir())/'rineya-gpu-0.lock'):
            from experiments.model_quality_gpu_v1.native import configure_runtime,runtime
            from experiments.model_quality_gpu_v1.resources import guard,snapshot
            configure_runtime(17)
            actual=runtime(require=True)
            try:
                resources=guard(before_load=True,phase='train')
                status,error='TRAIN_PRELOAD_GUARD_PASSES',None
            except RuntimeError as exc:
                resources=snapshot()
                status,error='TRAIN_PRELOAD_GUARD_REFUSED',str(exc)
            write_new(args.out/'RESOURCE_OBSERVATION.json',{'status':status,'resources':resources,'error':error,
                'runtime':actual,'model_loaded':False,'model_forward_calls':0,'optimizer_steps':0,
                'source_weight_files_read':False,'quality_claim':False})
        return 0
    command=[sys.executable,str(Path(__file__).resolve()),'--out',str(args.out.absolute()),'--_worker']
    rc=bounded_child(command,args.out.absolute(),60)
    if rc==0:
        record=read_json(args.out/'RESOURCE_OBSERVATION.json')
        print(json.dumps({'status':record['status'],'available_host_GiB':record['resources']['available_host_bytes']/2**30,
                          'error':record['error'],'model_loaded':False}))
    return rc

if __name__=='__main__':
    raise SystemExit(main())
