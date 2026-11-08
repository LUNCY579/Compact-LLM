"""Bind admitted data, three existing models and fixed inference code before execution."""
import shutil
import sys
from pathlib import Path

if __package__:
    from . import common as c
    from .paths import ROOT, HERE, BASELINE, MIXTURE, PILOT, PYTHON
else:
    import common as c
    from paths import ROOT, HERE, BASELINE, MIXTURE, PILOT, PYTHON


CODE_FILES = ('__init__.py', 'paths.py', 'common.py', 'probe.py', 'binding.py', 'arithmetic.py',
              'prepare_data.py', 'prepare_runtime.py', 'run_all.py', 'score.py')


def verify_frozen(root=ROOT):
    root = Path(root)
    value = c.read(root / 'PRELAUNCH_BINDING.json')
    c.require(value['status'] == 'PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION', 'diagnostic freeze is not passed')
    for name, expected in value['code_files_sha256'].items():
        c.require(c.sha(HERE / name) == expected, 'frozen diagnostic code changed: ' + name)
    for name, expected in value['artifact_files_sha256'].items():
        c.require(c.sha(root / name) == expected, 'frozen diagnostic artifact changed: ' + name)
    return value


def main():
    c.require(Path(sys.executable).resolve() == PYTHON.resolve(), 'freeze requires the pinned RAB interpreter')
    data = ROOT / 'candidate-v1'
    admission = c.read(data / 'DATA_ADMISSION.json')
    c.require(admission['status'] == 'ADMITTED_FOR_NAMED_EXPLORATORY_DIAGNOSTIC' and
              admission['training_permission'] is False and admission['model_execution_permission'] is False,
              'new diagnostic data is not independently admitted')
    baseline = c.read(BASELINE / 'BASELINE_PLAN.json')
    envelope = c.read(ROOT / 'SESSION_ENVELOPE.json')
    criteria = c.read(ROOT / 'CRITERIA.json')
    audit = c.read(ROOT / 'DATA_AUDIT.json')
    c.require(audit['status'] == 'PASS_FROZEN_DATA_SEMANTICS_AND_CONTACTS' and
              audit['data_admission_sha256'] == c.sha(data / 'DATA_ADMISSION.json') and
              audit['input_sha256'] == c.sha(data / 'DIAGNOSTIC_INPUTS.jsonl'), 'independent data audit differs')
    closeout = c.read(MIXTURE / 'CLOSEOUT.json')
    c.require(closeout['status'] == 'COMPLETE_AUDITED_EXPERIMENT_FAILED_PRIMARY_CRITERIA' and
              closeout['goal_acquisition_without_forgetting_achieved'] is False,
              'prior failed experiment must remain preserved')
    c.require(c.runtime_versions() == baseline['runtime_versions'], 'pinned probe runtime changed')
    code = {name: c.sha(HERE / name) for name in CODE_FILES}
    pilot = c.pilot_code_files(PILOT)
    c.require(pilot == baseline['pilot_code_sha256'], 'qualified inference/supervisor source changed')
    models = dict(baseline['models'])
    folder = MIXTURE / 'arms/mixed'
    models['mixed'] = {'path': str(folder), 'weight_sha256': c.MODEL_SHAS['mixed'],
        'receipt_sha256': {name: c.sha(folder / name) for name in ('ARM.json', 'MATERIALIZATION.json')}}
    plan = {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_PLAN', 'experiment': c.EXPERIMENT,
        'pilot_root': str(PILOT), 'pilot_head': baseline['pilot_head'], 'python_path': str(PYTHON),
        'output_root': str(ROOT / 'eval'), 'input_path': str(data / 'DIAGNOSTIC_INPUTS.jsonl'),
        'input_sha256': c.sha(data / 'DIAGNOSTIC_INPUTS.jsonl'), 'source_dir': baseline['source_dir'],
        'sidecars_sha256': baseline['sidecars_sha256'], 'models': models,
        'code_sha256': {name: code[name] for name in ('common.py', 'probe.py')},
        'pilot_code_sha256': pilot, 'runtime_versions': c.runtime_versions()}
    c.require(admission.get('schema') == 1 and admission.get('record') == 'BINDING_DIAGNOSTIC_SCOPED_DATA_ADMISSION' and
              admission.get('mandatory_checks_passed') is True and admission.get('rows_per_model') == c.NROWS and
              admission.get('binding_parent_worlds') == 32 and admission.get('arithmetic_parent_worlds') == 96 and
              admission.get('frozen_models_weight_sha256') == c.MODEL_SHAS and
              c.absolute(admission['policy_input_path']) == c.absolute(plan['input_path']) and
              admission['policy_input_sha256'] == admission['evidence_sha256']['DIAGNOSTIC_INPUTS.jsonl'] == plan['input_sha256'],
              'scoped admission does not bind the exact current bank')
    for name, expected in admission['evidence_sha256'].items():
        c.require(Path(name).name == name and c.sha(data / name) == expected, 'admitted data evidence changed: ' + name)
    c.validate_plan(plan)
    rows = c.validate_inputs(c.read_jsonl(plan['input_path']))
    prefills = sum(len(row['prompt_ids']) for row in rows)
    c.require(4 * (prefills + c.NROWS * 24) <= c.LIMITS['processed_input_tokens'], 'all-attempt input reserve exceeds cap')
    snapshots = {role: c.snapshot(plan, role) for role in c.MODEL_SHAS}
    c.require(envelope['deadline_unix'] - envelope['session_start_unix'] == c.LIMITS['total_owned_work_seconds'] and
              envelope['model_worker_launches_all_attempts'] == c.LIMITS['workers'] and
              envelope['rows_reserved_all_attempts'] == c.LIMITS['rows'] and
              envelope['generation_tokens_reserved_all_attempts'] == c.LIMITS['reserved_generated_tokens'],
              'fresh numeric envelope differs from execution limits')
    c.require(criteria['primary']['holm_family_size'] == 2, 'unregistered primary multiplicity')
    for name in ('PLAN.json', 'GRANT.json', 'PRELAUNCH_BINDING.json'):
        c.require(not (ROOT / name).exists(), 'prelaunch outputs are create-only')
    c.write_new(ROOT / 'PLAN.json', plan)
    grant = {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_GRANT', 'experiment': c.EXPERIMENT,
        'decision': 'APPROVED',
        'authorization_reference': 'User goal plus explicit delegation to choose and perform the best recommended next step; separately bounded evaluation-only diagnostic, no training or reused permit.',
        'plan_path': str(ROOT / 'PLAN.json'), 'plan_sha256': c.sha(ROOT / 'PLAN.json'),
        'limits': c.LIMITS, 'registration_path': str(ROOT / 'PREREG.md'),
        'registration_sha256': c.sha(ROOT / 'PREREG.md'), 'data_admission_path': str(data / 'DATA_ADMISSION.json'),
        'data_admission_sha256': c.sha(data / 'DATA_ADMISSION.json'),
        'session_start_unix': envelope['session_start_unix'], 'deadline_unix': envelope['deadline_unix']}
    c.write_new(ROOT / 'GRANT.json', grant)
    c.validate_grant(ROOT / 'PLAN.json', grant)
    c.phase_guard(plan, grant, seconds_needed=2700)
    artifacts = ['PLAN.json', 'GRANT.json', 'PREREG.md', 'CRITERIA.json', 'SESSION_ENVELOPE.json',
                 'DATA_POLICY_REVIEW.md', 'CONTACT_RULES.json', 'DATA_AUDIT.json']
    artifacts += [p.relative_to(ROOT).as_posix() for p in sorted(data.iterdir()) if p.is_file()]
    archive = ROOT / 'code_before_inference'
    archive.mkdir(exist_ok=False)
    for name, expected in code.items():
        shutil.copyfile(HERE / name, archive / name)
        c.require(c.sha(archive / name) == expected, 'code archive did not preserve bytes')
    c.write_new(archive / 'FILES.json', code)
    c.write_new(ROOT / 'PRELAUNCH_BINDING.json', {'schema': 1, 'experiment': c.EXPERIMENT,
        'status': 'PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION', 'code_files_sha256': code,
        'artifact_files_sha256': {name: c.sha(ROOT / name) for name in artifacts},
        'model_runtime_input_snapshots': snapshots, 'pilot_code_sha256': pilot,
        'rows_per_model': c.NROWS, 'parent_counts': {'binding': 32, 'arithmetic': 96},
        'prefill_tokens_per_model': prefills, 'generation_tokens_reserved_per_attempt': c.NROWS * 24,
        'prior_closeout_sha256': c.sha(MIXTURE / 'CLOSEOUT.json'),
        'model_workers_launched': 0, 'training_permission': False, 'quality_claim': False})
    verify_frozen()
    print('DIAGNOSTIC_PRELAUNCH_FROZEN_NO_MODEL_LAUNCH')


if __name__ == '__main__':
    main()
