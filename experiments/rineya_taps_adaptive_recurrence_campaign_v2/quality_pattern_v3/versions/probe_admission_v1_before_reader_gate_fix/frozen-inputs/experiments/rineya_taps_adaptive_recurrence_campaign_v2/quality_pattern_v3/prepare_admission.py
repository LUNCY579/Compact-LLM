"""Freeze the actual passive-probe dependency map without importing Torch."""
import ast
import json
from pathlib import Path
import subprocess
import sys
from .authority import PACKAGE, CAMPAIGN, ROOT, GIB, sha
from ..binding import admitted_source


def main():
    target=PACKAGE/'probe_admission_v1.json'
    if target.exists():raise FileExistsError('preserve existing freeze')
    old=json.loads((CAMPAIGN/'full_generation_restart_admission_v1.json').read_text(encoding='utf-8'))
    source=Path(old['source']);history,inherited=admitted_source(source)
    files={ROOT/relative for relative in inherited}
    files.update((ROOT/'experiments/rineya_taps_adaptive_recurrence_v1').glob('*.py'))
    for n in ('binding.py','control.py','gpu_native.py','native_supervisor.py','native_entry.py',
              'bulk_restart_supervisor.py','supervision.py','run_apparatus.py','full_generation_restart_authority.py'):
        files.add(CAMPAIGN/n)
    files.update(PACKAGE.glob('*.py'))
    files.update((PACKAGE/'tests').glob('*.py'))
    for n in ('ACTIVE_GOAL_SCOPE_001.json','ACTIVE_GOAL_SNAPSHOT_001.json',
              'PASSIVE_PROBE_REGISTRATION_001.md','PROBE_NATIVE_DRAFT_001.md','PROBE_ANALYSIS_PLAN_001.md'):
        files.add(PACKAGE/n)
    data=PACKAGE/'data/dev_prefix_probe_001'
    files.update(data.glob('*.json'))
    model=CAMPAIGN/'runs/binding-001/MODEL_DATA_BINDING.json'
    smoke=CAMPAIGN/'runs/binding-001/PROMPT_AUDIT.json'
    files.update((model,smoke))
    files.add(ROOT/'Rineya_Cross_Scale_Capability_Transfer_Lab_v1/data/evaluation/supa_layer_ablation_smoke.jsonl')
    # Pure analysis source is part of the intended run contract, not an invented fixture member.
    files.add(PACKAGE/'probe_analysis.py')
    missing=[str(p) for p in files if not p.is_file()]
    if missing:raise FileNotFoundError('actual dependency absent:'+repr(missing))
    node=next(n for n in ast.parse((PACKAGE/'probe_native.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef) and n.name=='count_contract')
    counts=ast.literal_eval(node.body[0].value)
    scope=json.loads((PACKAGE/'ACTIVE_GOAL_SCOPE_001.json').read_text(encoding='utf-8'))
    admission={**scope['probe'],'status':'READY_FOR_PASSIVE_PROBE_FP32_DISPATCH',
        'lane':'GPU_FP32_PASSIVE_INPUT_FEATURE_PROBE','source':str(source),'interpreter':sys.executable,
        'head':subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        'authority_path':str(PACKAGE/'ACTIVE_GOAL_SCOPE_001.json'),'authority_sha256':sha(PACKAGE/'ACTIVE_GOAL_SCOPE_001.json'),
        'goal_snapshot_path':str(PACKAGE/'ACTIVE_GOAL_SNAPSHOT_001.json'),'goal_snapshot_sha256':sha(PACKAGE/'ACTIVE_GOAL_SNAPSHOT_001.json'),
        'device':'cuda:0','dtype':'float32','attn_implementation':'eager','use_cache':False,'training':False,'native_test':False,
        'count_contract':counts,'checkpoint_files':old['checkpoint_files'],'gpu_uuid':old['gpu_uuid'],'gpu_driver_version':old['gpu_driver_version'],
        'parent_host_available_floor_bytes':7*GIB,'worker_host_available_floor_bytes':6*GIB,'worker_rss_ceiling_bytes':6*GIB,
        'gpu_initial_free_floor_bytes':6*GIB,'gpu_allocated_ceiling_bytes':5*GIB,'gpu_global_free_reserve_bytes':GIB,
        'global_finalization_reserve_seconds':1800,'model_binding_path':str(model),'model_binding_sha256':sha(model),
        'smoke_prompt_audit_path':str(smoke),'smoke_prompt_audit_sha256':sha(smoke),
        'review_receipt':'quality_pattern_v3/reviews/probe_dispatch_receipt_001.json',
        'dispatch_hashes':{p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)},
        'analysis_source_path':str(PACKAGE/'probe_analysis.py'),'analysis_source_sha256':sha(PACKAGE/'probe_analysis.py'),
        'analysis_plan_path':str(PACKAGE/'PROBE_ANALYSIS_PLAN_001.md'),'analysis_plan_sha256':sha(PACKAGE/'PROBE_ANALYSIS_PLAN_001.md'),
        'no_retry':True,'models_loaded_by_preparation':0}
    for key,name in (('dev_bank','DEVELOPMENT_BANK.json'),('dev_token_binding','DEVELOPMENT_TOKEN_BINDING.json'),('dev_export_manifest','MANIFEST.json')):
        admission[key+'_path']=str(data/name);admission[key+'_sha256']=sha(data/name)
    with target.open('x',encoding='utf-8') as handle:json.dump(admission,handle,indent=2);handle.write('\n')
    assert 'torch' not in sys.modules
    print(json.dumps({'admission':str(target),'sha256':sha(target),'sources':len(files),'models':0}))


if __name__=='__main__':main()
