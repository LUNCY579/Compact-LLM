"""Bounded stage/fresh-package test harness, no native/runtime-binary operations."""
from pathlib import Path
import hashlib,json,shutil,subprocess,sys,tempfile,time
HERE=Path(__file__).resolve().parent
assert HERE.name=='encoder_runtime_v2_work' and HERE.parent.name=='artifacts'
PACKAGE=HERE.parents[1]
QUARTET=('long_native_contract.py','long_native_core.py','long_native_worker.py','long_native_supervisor.py')
NEW=('encoder_registration_v2.py','admitted_runtime_observer.py','test_long_native_runtime_v2.py')
ROOT_NAMES=('native_contract.py','native_loader.py','native_precision.py','native_mechanics.py','native_process.py','native_nvml.py',
    'native_supervisor.py','native_worker.py','inference_contract.py','inference_component.py','inference_qual_worker.py','long_inference_inputs.py',
    'cpu_tokenizer.py','long_inference_component.py','backend_qual_contract.py','study_schedule.py','encoder_inventory.py','stdlib_inventory.py','rs_contract.py')
def pins(paths):return {str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
def run():
    observed=[PACKAGE/name for name in ROOT_NAMES]+[PACKAGE/name for name in QUARTET]+[HERE/name for name in (*QUARTET,*NEW,'portable_cpu_controls.py')]
    before=pins(observed);started=time.monotonic()
    staged=subprocess.run([sys.executable,'-I','-B',str(HERE/'test_long_native_runtime_v2.py')],capture_output=True,text=True,timeout=30)
    with tempfile.TemporaryDirectory() as leaf:
        package=Path(leaf)/'package';tests=package/'tests';tests.mkdir(parents=True)
        for name in ROOT_NAMES:shutil.copyfile(PACKAGE/name,package/name)
        for name in (*QUARTET,*NEW[:-1]):shutil.copyfile(HERE/name,package/name)
        shutil.copyfile(HERE/NEW[-1],tests/NEW[-1])
        for rel in ('configs/SYNTH_NQ001_PROC7.json','evidence/long_train_prompts_cpu_20261005/TRAIN_PROMPTS_CPU.json'):
            target=package/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PACKAGE/rel,target)
        absent_before=not (package/'artifacts/encoder_runtime_v2_work').exists()
        fresh=subprocess.run([sys.executable,'-I','-B',str(tests/NEW[-1])],capture_output=True,text=True,timeout=30)
        absent_after=not (package/'artifacts/encoder_runtime_v2_work').exists()
    after=pins(observed)
    return {'status':'PASS_CPU_ONLY_V2_CONTROLS' if staged.returncode==fresh.returncode==0 and before==after and absent_before and absent_after else 'FAILED_CPU_V2_CONTROLS',
        'stage':{'returncode':staged.returncode,'stdout':staged.stdout,'stderr':staged.stderr},
        'fresh_root':{'returncode':fresh.returncode,'stdout':fresh.stdout,'stderr':fresh.stderr,
            'ignored_stage_absent_before':absent_before,'ignored_stage_absent_after':absent_after},
        'source_before':before,'source_after':after,'source_unchanged':before==after,'wall_seconds':time.monotonic()-started,
        'encoder_framework_model_GPU_execution':False,'runtime_binary_hashes_recomputed':False,
        'native_authority_issued':False,'native_registration_or_config_written':False}
if __name__=='__main__':
    result=run();print(json.dumps(result,sort_keys=True))
    raise SystemExit(0 if result['status']=='PASS_CPU_ONLY_V2_CONTROLS' else 1)
