"""Create a local scientific closeout from completed, independently audited evidence."""
from datetime import datetime, timezone
from pathlib import Path
import time

from paths import OUT, HERE, sha, read, write
from launch import verify_frozen


def main():
    frozen = verify_frozen()
    results = read(OUT / 'RESULTS.json')
    costs = read(OUT / 'COST_REPORT.json')
    training = read(OUT / 'TRAINING_AUDIT.json')
    standard = read(OUT / 'STANDARD_NUMERICAL_AUDIT.json')
    binding = read(OUT / 'BINDING_NUMERICAL_AUDIT.json')
    envelope = read(OUT / 'SESSION_ENVELOPE.json')
    terminal = read(OUT / 'posttrain/TERMINAL.json')
    assert results['status'] == 'COMPLETE_COMPARABLE'
    assert costs['status'] == 'COMPLETE_VERIFIED_COST_REPORT'
    assert training['status'] == 'PASS_TRAINING_CUSTODY_ACCOUNTING_AND_CHECKPOINTS'
    assert standard['status'] == 'PASS_INDEPENDENT_STANDARD_RECOUNT'
    assert binding['status'] == 'PASS_RECORDED_CUSTODY_AND_INDEPENDENT_NUMERICS'
    assert standard['results_sha256'] == binding['result_sha256'] == sha(OUT / 'RESULTS.json')
    assert standard['audit_script_sha256'] == sha(HERE / 'audit_standard.py')
    assert binding['audit_script_sha256'] == sha(HERE / 'audit_binding.py')
    assert terminal['status'] == 'COMPLETE_FIVE_PROBES'
    assert len(terminal['attempts']) == 5 and all(attempt['returncode'] == 0 for attempt in terminal['attempts'])
    assert results['predictions']['primary_engineering_conjunction'] == 'FAILED_PREDICTION'
    assert results['predictions']['binding_efficacy'] == 'NOT_ESTABLISHED'
    assert time.time() <= envelope['deadline_unix']
    weight = OUT / 'arms/mixed/model.safetensors'
    arm = read(OUT / 'arms/mixed/ARM.json')
    weight_sha = sha(weight)
    assert weight_sha == arm['arm_weight_sha256'] == '894190b696498aa29b19285c86a7335070b051385c9135dc46d937cb0ba55f54'
    evidence = ('RESULTS.json', 'RESULTS.md', 'COST_REPORT.json', 'COST_REPORT.md',
        'TRAINING_AUDIT.json', 'STANDARD_NUMERICAL_AUDIT.json', 'BINDING_NUMERICAL_AUDIT.json',
        'INTERPRETATION.md', 'BINDING_ERROR_ANALYSIS.md', 'RETENTION_ERROR_ANALYSIS.md',
        'EXPOSURE_DESCRIPTION.json', 'MIXTURE_COMPARISON_v2.png', 'MIXTURE_COMPARISON_v2.svg',
        'MIXTURE_COMPARISON_v2_FIGURE.json', 'posttrain/TERMINAL.json',
        'code_after_run/AUDIT_SCRIPT_ARCHIVE.json')
    logical_bytes = sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file())
    assert logical_bytes + 2**20 < envelope['new_records_disk_bytes']
    write(OUT / 'CLOSEOUT.json', {'schema': 1, 'experiment': 'BINDING-MIXTURE-001',
        'status': 'COMPLETE_AUDITED_EXPERIMENT_FAILED_PRIMARY_CRITERIA',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'all_five_probes_succeeded': True, 'training_retries': 0, 'inference_retries': 0,
        'native_grant_consumed': True, 'frozen_execution_code_files': len(frozen['code_files_sha256']),
        'goal_acquisition_without_forgetting_achieved': False,
        'predictions': results['predictions'], 'evidence_sha256': {name: sha(OUT / name) for name in evidence},
        'candidate_path': str(weight), 'candidate_sha256': weight_sha,
        'checkpoint_manifest_sha256': read(OUT / 'journal/runs/train-001/result.json')['checkpoint_manifest_sha256'],
        'logical_records_bytes_before_closeout': logical_bytes,
        'session_elapsed_at_closeout_seconds': time.time() - envelope['session_start_unix'],
        'next_recommendation': 'Separately preregister a fresh controlled diagnostic on frozen source, LoRA and mixture: counterfactual binding selectivity plus source-conditioned arithmetic preservation. Diagnose both failures before another training run.',
        'next_experiment_started': False, 'next_experiment_permissions_granted_by_this_record': False,
        'quality_claim': False, 'selection_permitted': False, 'release_permitted': False,
        'closeout_script_sha256': sha(Path(__file__))})
    print('COMPLETE_AUDITED_EXPERIMENT_FAILED_PRIMARY_CRITERIA')


if __name__ == '__main__':
    main()
