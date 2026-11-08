"""Freeze the prospective registration and validate D4 inputs without model imports."""
from pathlib import Path
import collections
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone

BASE = Path('C:/Users/clclc/BusinessProject')
RECORDS = BASE / 'rineya-dataset-retention-records'
OUT = RECORDS / 'doseint001-20260919'
PILOT = BASE / 'ChatBubble-retention-pilot-20260917'
SOURCE = BASE / 'Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze'
OBJECTIVE = Path('C:/Users/clclc/.codex/attachments/9253cba1-b908-4858-98f1-34c710d720e8/goal-objective.md')

def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()

def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))

def write(p, data):
    with Path(p).open('x', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, indent=2, allow_nan=False)
        f.write('\n')

def main():
    sys.path.insert(0, str(PILOT))
    from experiments.model_quality_gpu_v1.integrity import digest
    from experiments.reasoning_posttraining_v1 import budget, process_dose
    from experiments.reasoning_posttraining_v1.controller import presentations, design_for, checkout
    from experiments.reasoning_posttraining_v1.process_arm import validate_execution_exclusions
    prepared = RECORDS / 'dose-D4-prepared'
    contract = read(prepared / 'DATA_CONTRACT.json')
    for name, expected in contract['files'].items():
        assert sha(prepared / name) == expected, f'prepared bytes changed: {name}'
    plan = read(RECORDS / 'dose-D4-PLAN.json')
    design_for(plan)
    presentations(prepared, plan, retain=False)
    train = [json.loads(x) for x in (prepared / 'microfit-train.jsonl').read_text(encoding='utf-8').splitlines()]
    validate_execution_exclusions(plan, contract, train)
    parents = {x['parent_id'] for x in train}
    assert len(train) == len(parents) == len({x['id'] for x in train}) == 3072
    banks = {}
    for name, relative in [('dose', 'process-transition-probe-001/CANDIDATE_VIEWS.jsonl'), ('select', 'select-views-001/SELECT_VIEWS.jsonl')]:
        p = RECORDS / relative
        rows = [json.loads(x) for x in p.read_text(encoding='utf-8').splitlines()]
        evalparents = {x['parent_id'] for x in rows}
        assert len(rows) == len(evalparents) == 384
        banks[name] = {'path': str(p), 'sha256': sha(p), 'parents': len(evalparents),
                       'training_parent_overlap': len(parents & evalparents)}
    assert banks['dose']['training_parent_overlap'] == 0
    assert banks['dose']['sha256'].startswith('45674251')
    source_lock = read(PILOT / 'experiments/model_quality_gpu_v1/source_lock.json')
    # Native controller also independently audits all source assets at launch.
    assets = {name: sha(SOURCE / name) for name in ('model.safetensors', 'config.json', 'generation_config.json', 'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt')}
    assert assets['model.safetensors'] == '962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b'
    checkout_sha = checkout()
    free = shutil.disk_usage(OUT.parent).free
    assert free >= 30 * 2**30, '16 GiB planned artifacts plus unchanged 14 GiB floor'
    OUT.mkdir(exist_ok=True)
    for src, name in [(Path(__file__).with_name('PREREG.md'), 'PREREG.md'), (OBJECTIVE, 'USER_OBJECTIVE.md')]:
        with (OUT / name).open('xb') as dst:
            dst.write(src.read_bytes())
    limits = dict(process_dose.LIMITS)
    limits.update(model_worker_launches=1, retries=0, processed_training_tokens_all_attempts=433813)
    grant = {
        'approved_by': 'current user explicit goal instruction: one D4 run; numeric ceiling instantiated by agent',
        'authorization_reference': str(OBJECTIVE), 'authorization_sha256': sha(OBJECTIVE),
        'registration_sha256': sha(OUT / 'PREREG.md'),
        'data_contract_sha256': sha(prepared / 'DATA_CONTRACT.json'),
        'decision': 'APPROVED', 'excludes': list(process_dose.EXCLUDES), 'experiment': process_dose.EXPERIMENT,
        'human_data_review': 'NOT_PERFORMED', 'limits': limits, 'plan_sha256': digest(plan),
        'quality_claim_permitted': False, 'release_permitted': False, 'training_permission': True,
        'what_training_permission_means': 'One fixed D4 run for DOSEINT-001, new grant/journal; no adaptive search or release.'}
    budget.validate_grant(grant, sha(prepared / 'DATA_CONTRACT.json'), digest(plan), process_dose.EXPERIMENT)
    write(OUT / 'GRANT.json', grant)
    scripts = {name: sha(PILOT / 'experiments/rab001_exploration' / (name + '_probe.py')) for name in ('process', 'retention', 'long_context')}
    write(OUT / 'BUILD_GATE.json', {
        'status': 'PASS_DOSE_ONLY', 'created_utc': datetime.now(timezone.utc).isoformat(),
        'banks': banks, 'select_bridge_status': 'BLOCKED_TRAIN_EVAL_PARENT_OVERLAP',
        'training_rows': len(train), 'training_parent_count': len(parents),
        'prepared': str(prepared), 'plan_path': str(RECORDS / 'dose-D4-PLAN.json'),
        'plan_file_sha256': sha(RECORDS / 'dose-D4-PLAN.json'), 'source': str(SOURCE),
        'source_assets_sha256': assets, 'pilot_checkout': checkout_sha,
        'probe_script_sha256': scripts,
        'wrapper_sha256': sha(PILOT / 'docs/model-quality/audit-2026-09-17/supervised_probe.py'),
        'registration_sha256': sha(OUT / 'PREREG.md'), 'free_disk_bytes': free,
        'quality_claim': False, 'release_permitted': False})
    print(json.dumps({'status': 'PASS_DOSE_ONLY', 'banks': banks, 'pilot_checkout': checkout_sha, 'out': str(OUT)}, indent=2))

if __name__ == '__main__':
    main()
