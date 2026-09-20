"""Freeze the new integration grant, data, dependencies and measured qualification scope."""
from argparse import Namespace
from pathlib import Path
import shutil
import sys
import time

from paths import BASE, OUT, HERE, PILOT, CONTROL, MIXTURE, DIAGNOSTIC, BASELINE, sha, read, write


def main():
    import train
    train.inputs(Namespace(prepared=OUT / 'prepared', plan=OUT / 'PLAN.json', grant=OUT / 'GRANT.json'), retain=False)
    identity = train.execution_identity()
    sys.path.insert(0, str(BASE / 'ChatBubble-dose-param'))
    from research.retention_highdose_lora_resume_v1.paired import SourceArchive
    import run_probe
    import report
    prior = read(CONTROL / 'BUILD_GATE_LORA_FINAL.json')
    build = read(OUT / 'prepared/DATA_BUILD.json')
    admission, audit = read(OUT / 'DATA_USE_ADMISSION.json'), read(OUT / 'PREPARATION_AUDIT.json')
    if (admission['status'] != 'ADMITTED_FOR_NAMED_TRAIN_DATA_AND_EXPLORATORY_EVALUATION_SCOPE' or
        admission['experiment'] != train.EXPERIMENT or admission['model_execution_permission'] is not False):
        raise ValueError('new scoped data-use admission required')
    if not str(audit['status']).startswith('PASS'):
        raise ValueError('independent preparation audit required')
    archive = SourceArchive.load(OUT / 'SOURCE_RELOCATION.json', sha(OUT / 'SOURCE_RELOCATION.json'))
    archive.revalidate()
    for name, expected in prior['source_assets_sha256'].items():
        if sha(Path(prior['source']) / name) != expected:
            raise ValueError('source asset changed: ' + name)
    for root in (CONTROL, MIXTURE, DIAGNOSTIC):
        if read(root / 'RESULTS.json')['status'] != 'COMPLETE_COMPARABLE':
            raise ValueError('frozen comparator must be complete and comparable')
    if sha(OUT / 'PREREG.md') != read(OUT / 'GRANT.json')['registration_sha256']:
        raise ValueError('registration/grant binding changed')
    banks = {item['bank']: {'path': item['path'], 'sha256': item['sha256'], 'parents': item['eval_parents'],
                           'training_parent_overlap': item['mixture_parent_overlap']}
             for item in build['acquisition_parent_exclusions']}
    gate = {'status': 'PASS_INTEGRATION_DATA_AND_EXECUTABLE_BINDING', 'created_unix': time.time(),
        'experiment': train.EXPERIMENT, 'banks': banks, 'training_rows': 3465, 'training_parent_count': 2761,
        'prepared': str(OUT / 'prepared'), 'plan_path': str(OUT / 'PLAN.json'), 'plan_file_sha256': sha(OUT / 'PLAN.json'),
        'source': prior['source'], 'source_assets_sha256': prior['source_assets_sha256'],
        'pilot_checkout': identity['pilot_checkout'], 'training_execution_identity': identity,
        'probe_script_sha256': prior['probe_script_sha256'], 'wrapper_sha256': prior['wrapper_sha256'],
        'implementation_digest': run_probe.implementation_digest(), 'supervisor_sha256': sha(run_probe.SUPERVISOR_CODE),
        'data_build_sha256': sha(OUT / 'prepared/DATA_BUILD.json'),
        'binding_data_admission_sha256': sha(Path(build['binding_data_admission_path'])),
        'data_use_admission_path': str(OUT / 'DATA_USE_ADMISSION.json'), 'data_use_admission_sha256': sha(OUT / 'DATA_USE_ADMISSION.json'),
        'registration_sha256': sha(OUT / 'PREREG.md'), 'criteria_sha256': sha(OUT / 'CRITERIA.json'),
        'session_envelope_sha256': sha(OUT / 'SESSION_ENVELOPE.json'),
        'source_relocation_path': str(OUT / 'SOURCE_RELOCATION.json'), 'source_relocation_sha256': sha(OUT / 'SOURCE_RELOCATION.json'),
        'frozen_results_sha256': {'mixed': sha(MIXTURE / 'RESULTS.json'), 'lora': sha(CONTROL / 'RESULTS.json'),
                                  'diagnostic': sha(DIAGNOSTIC / 'RESULTS.json')},
        'binding_baseline_scores_sha256': sha(BASELINE / 'BASELINE_SCORES.json'),
        'binding_score_plan_sha256': sha(BASELINE / 'SCORE_PLAN.json'),
        'binding_baseline_plan_sha256': sha(BASELINE / 'BASELINE_PLAN.json'),
        'zero_addend_qualification_scope': 'EXACT_TWO_UPDATE_PREFIX_ONLY',
        'quality_claim': False, 'release_permitted': False}
    if any((OUT / name).exists() for name in ('BUILD_GATE.json', 'PRELAUNCH_BINDING.json', 'code_before_training')):
        raise ValueError('prelaunch freeze is create-only')
    names = ('__init__.py', 'train.py', 'adapters.py', 'prepare_data.py', 'paths.py', 'launch.py', 'cpu_stage.py',
             'posttrain.py', 'materialize.py', 'run_probe.py', 'eval_common.py', 'eval_probe.py', 'report.py', 'finalize_prelaunch.py')
    code = {name: sha(HERE / name) for name in names}
    external = report.external_code_snapshot()
    write(OUT / 'BUILD_GATE.json', gate)
    # Validate the freshly registered gate before publishing a PASS freeze.
    run_probe.source_relocation_snapshot(gate)
    artifacts = ('PLAN.json', 'GRANT.json', 'PREREG.md', 'CRITERIA.json', 'SESSION_ENVELOPE.json', 'BUILD_GATE.json',
        'ISOLATION_AUDIT.json', 'SOURCE_RELOCATION.json', 'DATA_USE_ADMISSION.json', 'PREPARATION_AUDIT.json',
        'prepared/DATA_CONTRACT.json', 'prepared/DATA_BUILD.json', 'prepared/INTEGRATION_GROUPS.json',
        'prepared/SOURCE_MEMBERSHIP.json', 'prepared/microfit-train.jsonl',
        'STORAGE_QUALIFICATION.json', 'SIDECAR_STORAGE_QUALIFICATION.json')
    record = {'schema': 1, 'status': 'PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION',
        'code_files_sha256': code, 'external_code_files_sha256': external,
        'artifact_files_sha256': {name: sha(OUT / name) for name in artifacts},
        'training_execution_identity': identity, 'source_assets_sha256': prior['source_assets_sha256'],
        'plan_sha256': sha(OUT / 'PLAN.json'), 'grant_sha256': sha(OUT / 'GRANT.json'),
        'registration_sha256': sha(OUT / 'PREREG.md'), 'session_envelope_sha256': sha(OUT / 'SESSION_ENVELOPE.json'),
        'build_gate_sha256': sha(OUT / 'BUILD_GATE.json'), 'expected_active_optimizer_updates': 193,
        'qualification_optimizer_updates': 2, 'measured_zero_addend_scope': 'EXACT_TWO_UPDATE_PREFIX_ONLY',
        'expected_charged_training_and_qualification_tokens': 590245, 'expected_qualification_calls': 76,
        'model_execution': False, 'quality_claim': False}
    destination = OUT / 'code_before_training'; destination.mkdir()
    for name, expected in code.items():
        shutil.copyfile(HERE / name, destination / name)
        if sha(destination / name) != expected:
            raise ValueError('archived execution bytes differ')
    write(destination / 'FILES.json', code)
    write(OUT / 'PRELAUNCH_BINDING.json', record)
    from launch import verify_frozen
    verify_frozen()
    report.verify_prelaunch(OUT)
    print('INTEGRATION_PRELAUNCH_FROZEN; NO MODEL LAUNCHED')


if __name__ == '__main__':
    main()
