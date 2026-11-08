"""Create the fresh LoRA contract/plan/grant from verified identical data bytes."""
from pathlib import Path
from argparse import Namespace
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import shutil
import sys
import time

BASE=Path('C:/Users/clclc/BusinessProject')
RECORDS=BASE/'rineya-dataset-retention-records'
OLD=RECORDS/'doseint001-20260919'
OUT=RECORDS/'highdose-lora001-resume-20260920'
HERE=Path(__file__).resolve().parent
PILOT=BASE/'ChatBubble-retention-doselora-resume-20260920'

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p,value):
    with Path(p).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def copy_new(src,dst):
    with Path(src).open('rb') as source,Path(dst).open('xb') as target:
        shutil.copyfileobj(source,target)
    if sha(src)!=sha(dst):
        raise ValueError('Copy hash differs')

def main():
    sys.path.insert(0,str(PILOT))
    spec=importlib.util.spec_from_file_location('doselora_train_prepare',HERE/'train.py')
    train=importlib.util.module_from_spec(spec);sys.modules[spec.name]=train;spec.loader.exec_module(train)
    from experiments.model_quality_gpu_v1.integrity import digest
    old_gate=read(OLD/'BUILD_GATE_DUAL899.json')
    old_contract=read(OLD/'prepared-dual899/DATA_CONTRACT.json')
    old_plan=read(OLD/'PLAN_DUAL899.json')
    for name,expected in old_contract['files'].items():
        assert sha(OLD/'prepared-dual899'/name)==expected
    assert read(OLD/'RESULTS.json')['status']=='COMPLETE_COMPARABLE'
    OUT.mkdir(exist_ok=True)
    prepared=OUT/'prepared';prepared.mkdir(exist_ok=False)
    for name in old_contract['files']:
        copy_new(OLD/'prepared-dual899'/name,prepared/name)
    contract={**old_contract,'experiment':train.EXPERIMENT,
              'parent_data_contract_sha256':sha(OLD/'prepared-dual899/DATA_CONTRACT.json'),
              'parent_prepared':str(OLD/'prepared-dual899'),
              'what_this_is':'New experimental contract; exact original training artifacts retained with their original build provenance.'}
    write(prepared/'DATA_CONTRACT.json',contract)
    plan={**old_plan,'experiment':train.EXPERIMENT,
          'optimizer':{**old_plan['optimizer'],'peak_lr':train.PEAK_LR},
          'trainable':'blocks8-27_lora16_projections_and_norms',
          'parameterization':train.PARAMETERIZATION,
          'parent_data_plan_sha256':digest(old_plan),
          'declared_change':'LoRA rank16 plus higher peakLR; FP32/native optimizer/loss/token schedule/scope/norms/data matched.'}
    write(OUT/'PLAN.json',plan)
    copy_new(HERE/'PREREG.md',OUT/'PREREG.md')
    grant={'experiment':train.EXPERIMENT,'decision':'APPROVED','training_permission':True,
        'approved_by':'current user goal priority3 plus explicit delegated best-step discretion; agent-instantiated numeric envelope',
        'authorization_reference':str(OLD/'USER_OBJECTIVE.md'),'authorization_sha256':sha(OLD/'USER_OBJECTIVE.md'),
        'authorization_secondary_reference':str(OLD/'USER_STEERING_02.md'),
        'authorization_secondary_sha256':sha(OLD/'USER_STEERING_02.md'),
        'registration_sha256':sha(OUT/'PREREG.md'),
        'human_data_review':'NOT_PERFORMED','limits':dict(train.LIMITS),'excludes':list(train.EXCLUDES),
        'plan_sha256':digest(plan),'data_contract_sha256':sha(prepared/'DATA_CONTRACT.json'),
        'quality_claim_permitted':False,'release_permitted':False,
        'what_training_permission_means':'One fixed high-dose LoRA training trajectory with independent qualification; no adaptive search or release.'}
    write(OUT/'GRANT.json',grant)
    args=Namespace(prepared=str(prepared),plan=str(OUT/'PLAN.json'),grant=str(OUT/'GRANT.json'))
    train.inputs(args,retain=False)
    identity=train.execution_identity()
    gate={**old_gate,'status':'PASS_FRESH_LORA_DATA_AND_EXECUTABLE_BINDING',
          'created_utc':datetime.now(timezone.utc).isoformat(),'prepared':str(prepared),
          'plan_path':str(OUT/'PLAN.json'),'plan_file_sha256':sha(OUT/'PLAN.json'),
          'registration_sha256':sha(OUT/'PREREG.md'),'pilot_checkout':identity['pilot_checkout'],
          'training_execution_identity':identity,'parent_data_contract_sha256':contract['parent_data_contract_sha256']}
    gate.pop('amendment_sha256',None)
    write(OUT/'BUILD_GATE_LORA.json',gate)
    write(OUT/'BUILD_GATE_NATIVE_NORM.json',old_gate)
    # The new phase begins with its first persisted storage action; preparation is included.
    storage=OUT/'storage/PLAN.json'
    start=read(OUT/'SESSION_START.json')['session_start_unix']
    write(OUT/'SESSION_ENVELOPE.json',{'schema':1,'session_start_unix':start,
        'session_start_provenance':'first resume resource observation; SESSION_START.json',
        'deadline_unix':start+12600,'total_owned_work_wall_seconds':12600,
        'probe_worker_seconds_all_attempts':8100,'probe_rows_reserved_all_attempts':3000,
        'probe_generation_tokens_reserved_all_attempts':270000,'new_records_disk_bytes':7*2**30,
        'free_disk_floor_bytes':14*2**30,'pretrain_free_disk_bytes':int(19.5*2**30),
        'pretrain_host_bytes':7*2**30,'native_limits':train.LIMITS,'quality_claim':False})
    print(json.dumps({'status':gate['status'],'rows':plan['presentations'],'processed_tokens':plan['processed_tokens'],
                      'first_training_row_tokens':len(json.loads((prepared/'microfit-train.jsonl').open().readline())['ids']),
                      'limits':train.LIMITS,'pilot_checkout':identity['pilot_checkout']},indent=2))

if __name__=='__main__':
    main()
