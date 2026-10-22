"""Freeze the revised disjoint corpus and exercise the complete native input gate."""
from argparse import Namespace
from pathlib import Path
import json
import shutil
import sys
from datetime import datetime, timezone
from prepare import BASE, RECORDS, OUT, PILOT, SOURCE, OBJECTIVE, sha, read, write

def main():
    sys.path.insert(0, str(PILOT))
    from experiments.model_quality_gpu_v1.integrity import digest
    from experiments.reasoning_posttraining_v1 import budget, process_dose
    from experiments.reasoning_posttraining_v1.controller import inputs, design_for, checkout
    prepared = OUT / 'prepared-dual899'
    plan_path = OUT / 'PLAN_DUAL899.json'
    plan = read(plan_path)
    contract = read(prepared / 'DATA_CONTRACT.json')
    rows = [json.loads(x) for x in (prepared / 'microfit-train.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len({r['parent_id'] for r in rows}) == 2697
    assert plan['processed_tokens'] == 380921 and len(plan['update_groups']) == 193
    build = read(prepared / 'ARM_BUILD.json')
    assert build['per_family_taken'] == {f:899 for f in ('ledger_replay','prerequisite_plan','register_machine')}
    assert build['rejected_overlength'] == 0
    assert sha(OBJECTIVE) == read(OUT / 'GRANT.json')['authorization_sha256']
    design_for(plan)
    pilot_head = checkout()
    assert shutil.disk_usage(OUT).free >= 31 * 2**30
    amendment = Path(__file__).with_name('AMENDMENT_01.md')
    with (OUT / amendment.name).open('xb') as f:
        f.write(amendment.read_bytes())
    grant = read(OUT / 'GRANT.json')
    grant.update(registration_sha256=sha(OUT / 'PREREG.md'), amendment_sha256=sha(amendment),
                 plan_sha256=digest(plan), data_contract_sha256=sha(prepared / 'DATA_CONTRACT.json'),
                 what_training_permission_means='One fixed maximum balanced dual-bank-disjoint dose (2697 rows), not exact historical D4; new unused grant/journal; no adaptive search or release.')
    grant['limits']['processed_training_tokens_all_attempts'] = 380921
    grant_path = OUT / 'GRANT_DUAL899.json'
    write(grant_path, grant)
    args = Namespace(prepared=str(prepared), plan=str(plan_path), grant=str(grant_path),
                     experiment=process_dose.EXPERIMENT, command='train', bank=None, heldout_bank=[])
    inputs(args, retain_presentations=False)
    gate = read(OUT / 'BUILD_GATE.json')
    for info in gate['banks'].values():
        views = [json.loads(x) for x in Path(info['path']).read_text(encoding='utf-8').splitlines()]
        assert sha(info['path']) == info['sha256']
        overlap = {r['parent_id'] for r in rows} & {r['parent_id'] for r in views}
        assert not overlap
        info['training_parent_overlap'] = len(overlap)
    gate.update(status='PASS_NATIVE_INPUTS_AND_DUAL_BANK_DISJOINTNESS',
                created_utc=datetime.now(timezone.utc).isoformat(),
                select_bridge_status='PROSPECTIVE_MODEL_CUSTODY_REQUIRED',
                prepared=str(prepared), plan_path=str(plan_path), plan_file_sha256=sha(plan_path),
                training_rows=2697, training_parent_count=2697, pilot_checkout=pilot_head,
                amendment_sha256=sha(amendment), free_disk_bytes=shutil.disk_usage(OUT).free)
    for name, expected in gate['source_assets_sha256'].items():
        assert sha(SOURCE / name) == expected
    write(OUT / 'BUILD_GATE_DUAL899.json', gate)
    write(OUT / 'LEGACY_D4_REFUSAL.json', {
        'status':'BLOCKED_AT_BUILD', 'returncode':1, 'worker_reserved':False,
        'model_calls':0, 'optimizer_steps':0,
        'error':'current process training requires matching plan/contract selection exclusions; legacy or historical inventories are not current-gate eligible',
        'refusal_site':'experiments/reasoning_posttraining_v1/process_arm.py:155',
        'superseded_gate_sha256':sha(OUT / 'BUILD_GATE.json'),
        'resolution':'new dual899 plan and grant; guards and historical artifacts unchanged'})
    print(json.dumps({'status':gate['status'],'rows':2697,'tokens':380921,'banks':gate['banks']},indent=2))

if __name__ == '__main__':
    main()
