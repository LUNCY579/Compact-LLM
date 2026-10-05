"""Framework-free parent for bounded full-recompute tiny-Qwen CPU successor proof."""
from datetime import datetime,timezone
import ast
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

WORK=Path(__file__).resolve().parent
PACKAGE=WORK.parents[1]
OLD=PACKAGE/'artifacts/checkpoint_profile_work'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def child_source():
    tree=ast.parse((OLD/'prove_cpu_checkpoint.py').read_text())
    value=next(node.value.value for node in tree.body if isinstance(node,ast.Assign)
               and any(isinstance(target,ast.Name) and target.id=='CHILD' for target in node.targets))
    value=value.replace('from torch.utils.checkpoint import checkpoint',
                        'from torch.utils.checkpoint import checkpoint,set_checkpoint_early_stop')
    value=value.replace('import checkpoint_profile as profile','import checkpoint_successor_profile as profile')
    value=value.replace('profile.install(model,torch,checkpoint)',
                        'profile.install(model,torch,checkpoint,early_stop_context=set_checkpoint_early_stop)')
    value=value.replace('CPU_RANDOM_TINY_QWEN_NONREENTRANT_28_LAYER_112_FACTOR_CONTROL',
                        'CPU_RANDOM_TINY_QWEN_FULL_NONREENTRANT_RECOMPUTE_CONTROL')
    original="installed['verify']();fi['verify']();installed['remove']();fi['remove']()"
    checks=r'''
def check_regions(installed,expected):
 snapshot=installed['snapshot']()
 assert snapshot['failed_or_pending'] is False and snapshot['checkpoint_early_stop'] is False
 for phases in snapshot['layers'].values():
  for name in ('forward','recompute'):
   counts=phases[name]
   assert counts['failed']==counts['pending']==0 and counts['attempted']==counts['returned']==expected//28
   assert counts['elapsed_seconds']>=0
 regions={}
 for region in snapshot['regions']:
  assert region['status']=='returned' and region['elapsed_seconds']>=0
  regions.setdefault(region['region_id'],set()).add(region['phase'])
 assert len(regions)==expected and all(phases=={'forward','recompute'} for phases in regions.values())
 assert snapshot['counters']['forward_regions']==snapshot['counters']['recompute_regions']==expected
 return snapshot
original_regions=check_regions(installed,1120)
restored_regions=check_regions(fi,560)
aggregate_regions={'forward_regions':original_regions['counters']['forward_regions']+restored_regions['counters']['forward_regions'],
 'recompute_regions':original_regions['counters']['recompute_regions']+restored_regions['counters']['recompute_regions'],
 'failed':0,'pending':0,'all_recomputations_normal_exit':True}
assert aggregate_regions['forward_regions']==aggregate_regions['recompute_regions']==1680
installed['verify']();fi['verify']();installed['remove']();fi['remove']()
'''
    assert original in value
    value=value.replace(original,checks)
    value=value.replace("'checkpoint_recompute_regions':installed['counters'],",
        "'checkpoint_recompute_regions':installed['counters'],'aggregate_full_regions':aggregate_regions,'original40_region_detail':original_regions,'restored20_region_detail':restored_regions,")
    value=value.replace("'output_CE_all112_grad_Adam_moments_RNG_bitwise_equal':True,",
        "'output_CE_exact_numerical_equal':True,'all112_grad_Adam_moments_RNG_byte_equal':True,")
    value=value.replace('PASS_CPU_RANDOM_TINY_QWEN_CHECKPOINT_CONTROL_NATIVE_UNQUALIFIED',
                        'PASS_CPU_RANDOM_TINY_QWEN_FULL_RECOMPUTE_NATIVE_UNQUALIFIED')
    return value


def main():
    assert not any(name in sys.modules for name in ('torch','transformers'))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--receipt-name',default='cpu_full_recompute_proof_20261005.json')
    args=parser.parse_args()
    name=Path(args.receipt_name)
    assert name.name==args.receipt_name and name.suffix=='.json' and not (WORK/name).exists(), 'new owned write-once receipt leaf'
    candidate=WORK/'checkpoint_successor_profile.py'
    before={'candidate':sha(candidate),'proof':sha(__file__),'old_proof_source':sha(OLD/'prove_cpu_checkpoint.py'),
            'old_CPU_proof_receipt':sha(OLD/'cpu_checkpoint_proof_20261005.json')}
    child=child_source()
    environment=dict(os.environ,CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',
                     USE_TF='0',USE_FLAX='0',USE_TORCH='1',TOKENIZERS_PARALLELISM='false')
    started=datetime.now(timezone.utc)
    result=subprocess.run([sys.executable,'-I','-B','-c',child,str(PACKAGE),str(WORK)],
                          env=environment,text=True,capture_output=True,timeout=180)
    payload=json.loads(result.stdout) if result.returncode==0 else {
        'status':'FAILED_CPU_RANDOM_TINY_QWEN_FULL_RECOMPUTE','stderr':result.stderr,'stdout':result.stdout}
    after={'candidate':sha(candidate),'proof':sha(__file__),'old_proof_source':sha(OLD/'prove_cpu_checkpoint.py'),
           'old_CPU_proof_receipt':sha(OLD/'cpu_checkpoint_proof_20261005.json')}
    assert before==after,'source/proof/old-artifact byte drift during CPU control'
    payload.update({'child_returncode':result.returncode,'parent_framework_free':True,
        'started_at_utc':started.isoformat(),'finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_hashes_before':before,'source_hashes_after':after,'generated_child_source_sha256':hashlib.sha256(child.encode()).hexdigest(),
        'saved_storage_measurement_scope':'unique storage seen by saved_tensors_hooks for first-row forward plus completion loss, including preexisting/cast weights; not live/RSS/CUDA peak',
        'root_source_edits':False,'GPU_calls':0,'published_weights_opened':False,'native_launch_authorized':False})
    path=WORK/name
    with path.open('x',encoding='utf-8') as stream:
        json.dump(payload,stream,sort_keys=True,indent=2,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    print(json.dumps({'status':payload['status'],'child_returncode':result.returncode,'receipt':str(path),'before_after_hashes_equal':before==after}))
    return result.returncode


if __name__=='__main__':raise SystemExit(main())
