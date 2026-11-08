"""Close the audited diagnostic without claiming the original model goal is met."""
from datetime import datetime, timezone
import time
from pathlib import Path
from paths import ROOT, HERE
import common as c
from prepare_runtime import verify_frozen


def main():
    freeze = verify_frozen()
    result = c.read(ROOT / 'RESULTS.json')
    audit = c.read(ROOT / 'NUMERICAL_AUDIT.json')
    costs = c.read(ROOT / 'COST_REPORT.json')
    terminal = c.read(ROOT / 'battery/TERMINAL.json')
    grant = c.read(ROOT / 'GRANT.json')
    c.require(result['status'] == 'COMPLETE_COMPARABLE' and
        audit['status'] == 'PASS_INDEPENDENT_RAW_SEMANTIC_AND_NUMERICAL_RECOMPUTATION' and
        audit['result_sha256'] == c.sha(ROOT / 'RESULTS.json') and
        c.sha(audit['auditor_script_path']) == audit['auditor_script_sha256'] and
        costs['status'] == 'COMPLETE_VERIFIED_COST_REPORT', 'independent completed evidence is required')
    c.require(terminal['status'] == 'COMPLETE_THREE_MODELS' and
        [row['model_role'] for row in terminal['attempts']] == ['source', 'lora', 'mixed'] and
        all(row['returncode'] == 0 for row in terminal['attempts']), 'successful complete model battery required')
    c.require(result['source_conditioned_retention_descriptive']['n'] == 4 and
        result['primary']['arithmetic']['reference_cohort_status'] == 'INSUFFICIENT_REFERENCE_COHORT' and
        result['primary']['binding']['net'] == result['primary']['binding']['nd'] == 0,
        'closeout must preserve the observed inconclusive primary results')
    plan = c.read(ROOT / 'PLAN.json')
    current = {role: c.snapshot(plan, role) for role in c.MODEL_SHAS}
    c.require(current == freeze['model_runtime_input_snapshots'], 'a frozen model/runtime/input changed')
    c.phase_guard(plan, grant)
    files = ('RESULTS.json', 'RESULTS.md', 'NUMERICAL_AUDIT.json', 'AUDIT_FINDINGS.md',
        'COST_REPORT.json', 'COST_REPORT.md', 'INTERPRETATION.md', 'ARITHMETIC_DIAGNOSIS.md',
        'BINDING_ERROR_ANALYSIS.json', 'BINDING_ERROR_ANALYSIS.md',
        'NATIVE_COEFFICIENT_AUDIT.json', 'NATIVE_COEFFICIENT_AUDIT.md',
        'DIAGNOSTIC_COMPARISON.png', 'DIAGNOSTIC_COMPARISON.svg', 'DIAGNOSTIC_COMPARISON_FIGURE.json',
        'DATA_AUDIT.json', 'code_after_preparation/DATA_AUDIT_CODE_ARCHIVE.json',
        'battery/TERMINAL.json')
    c.write_new(ROOT / 'CLOSEOUT.json', {'schema': 1, 'experiment': c.EXPERIMENT,
        'status': 'COMPLETE_AUDITED_DIAGNOSTIC_PRIMARY_FLOOR_AND_INSUFFICIENT_REFERENCE_COHORT',
        'created_utc': datetime.now(timezone.utc).isoformat(), 'models_changed': False,
        'new_training_performed': False, 'successful_workers': 3, 'retries': 0,
        'all_original_model_slots_completed': True, 'further_model_work_in_this_phase': False,
        'primary': result['primary'], 'predictions': result['predictions'],
        'evidence_sha256': {name: c.sha(ROOT / name) for name in files},
        'prelaunch_binding_sha256': c.sha(ROOT / 'PRELAUNCH_BINDING.json'),
        'scorer_sha256': result['scorer_sha256'], 'closeout_script_sha256': c.sha(Path(__file__).resolve()),
        'session_elapsed_at_closeout_seconds': time.time() - grant['session_start_unix'],
        'original_goal_achieved': False, 'prior_failed_criteria_changed': False,
        'next_step': 'Qualify a native-exposure-preserving integration using the existing data: original193 native groups/denominators/LR clock, binding token-loss sums appended with the same native denominator, exact zero-addend control recovery. Fresh preregistration and numeric training grant required.',
        'next_step_model_execution_granted': False, 'quality_claim': False,
        'promotion_permitted': False, 'independent_confirmation': False})
    print('DIAGNOSTIC_CLOSED_ORIGINAL_GOAL_UNACHIEVED')


if __name__ == '__main__':
    main()
