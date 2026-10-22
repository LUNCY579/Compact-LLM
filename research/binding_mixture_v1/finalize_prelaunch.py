"""Freeze the exact mixed-data execution and comparison identities before launch."""
from argparse import Namespace
from pathlib import Path
import sys
import time

from paths import BASE, OUT, HERE, PILOT, sha, read, write


def main():
    import train
    train.inputs(Namespace(prepared=str(OUT/'prepared'), plan=str(OUT/'PLAN.json'),
                           grant=str(OUT/'GRANT.json')), retain=False)
    identity = train.execution_identity()
    control = BASE/'rineya-dataset-retention-records/highdose-lora001-resume-20260920'
    baseline = BASE/'rineya-dataset-retention-records/binding-dev-baseline001-20260920'
    prior = read(control/'BUILD_GATE_LORA_FINAL.json')
    build = read(OUT/'prepared/DATA_BUILD.json')
    sys.path.insert(0, str(BASE/'ChatBubble-dose-param'))
    from research.retention_highdose_lora_resume_v1.paired import SourceArchive
    SourceArchive.load(OUT/'SOURCE_RELOCATION.json', sha(OUT/'SOURCE_RELOCATION.json'))
    for name, expected in prior['source_assets_sha256'].items():
        if sha(Path(prior['source'])/name) != expected:
            raise ValueError('Source asset differs: '+name)
    if read(control/'RESULTS.json')['status'] != 'COMPLETE_COMPARABLE':
        raise ValueError('Qualified control report required')
    if read(baseline/'BASELINE_SCORES.json')['new_skill_headroom'] != 'NEW_SKILL_HEADROOM_PRESENT':
        raise ValueError('The registered new-skill headroom route did not qualify')
    if sha(OUT/'PREREG.md') != read(OUT/'GRANT.json')['registration_sha256']:
        raise ValueError('Prospective registration changed')
    banks = {item['bank']: {'path': item['path'], 'sha256': item['sha256'],
                           'parents': item['eval_parents'], 'training_parent_overlap': item['mixture_parent_overlap']}
             for item in build['acquisition_parent_exclusions']}
    gate = {'status': 'PASS_BINDING_MIXTURE_DATA_AND_EXECUTABLE_BINDING', 'created_unix': time.time(),
        'banks': banks, 'training_rows': 3465, 'training_parent_count': 2761,
        'prepared': str(OUT/'prepared'), 'plan_path': str(OUT/'PLAN.json'),
        'plan_file_sha256': sha(OUT/'PLAN.json'), 'source': prior['source'],
        'source_assets_sha256': prior['source_assets_sha256'], 'pilot_checkout': identity['pilot_checkout'],
        'training_execution_identity': identity, 'probe_script_sha256': prior['probe_script_sha256'],
        'wrapper_sha256': prior['wrapper_sha256'], 'data_build_sha256': sha(OUT/'prepared/DATA_BUILD.json'),
        'binding_data_admission_sha256': sha(Path(build['binding_data_admission_path'])),
        'registration_sha256': sha(OUT/'PREREG.md'), 'criteria_sha256': sha(OUT/'CRITERIA.json'),
        'session_envelope_sha256': sha(OUT/'SESSION_ENVELOPE.json'),
        'source_relocation_sha256': sha(OUT/'SOURCE_RELOCATION.json'),
        'control_results_sha256': sha(control/'RESULTS.json'),
        'binding_baseline_scores_sha256': sha(baseline/'BASELINE_SCORES.json'),
        'binding_score_plan_sha256': sha(baseline/'SCORE_PLAN.json'),
        'binding_baseline_plan_sha256': sha(baseline/'BASELINE_PLAN.json'),
        'quality_claim': False, 'release_permitted': False}
    write(OUT/'BUILD_GATE.json', gate)
    names = ('train.py', 'adapters.py', 'materialize.py', 'prepare_data.py', 'paths.py', 'launch.py',
             'cpu_stage.py', 'posttrain.py', 'run_probe.py', 'binding_common.py', 'binding_probe.py',
             'report.py', 'finalize_prelaunch.py')
    code = {name: sha(HERE/name) for name in names}
    artifact_names = ('CRITERIA.json', 'ISOLATION_AUDIT.json', 'SOURCE_RELOCATION.json',
                      'prepared/DATA_CONTRACT.json', 'prepared/DATA_BUILD.json')
    write(OUT/'PRELAUNCH_BINDING.json', {'schema': 1, 'status': 'PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION',
        'code_files_sha256': code, 'artifact_files_sha256': {name: sha(OUT/name) for name in artifact_names},
        'training_execution_identity': identity, 'source_assets_sha256': prior['source_assets_sha256'],
        'plan_sha256': sha(OUT/'PLAN.json'), 'grant_sha256': sha(OUT/'GRANT.json'),
        'registration_sha256': sha(OUT/'PREREG.md'), 'session_envelope_sha256': sha(OUT/'SESSION_ENVELOPE.json'),
        'build_gate_sha256': sha(OUT/'BUILD_GATE.json'), 'expected_optimizer_updates': 296,
        'expected_charged_training_and_qualification_tokens': 586213, 'expected_qualification_calls': 48,
        'model_execution': False, 'quality_claim': False})
    archive = OUT/'code_before_training'
    archive.mkdir(exist_ok=False)
    for name in names:
        with (archive/name).open('xb') as stream:
            stream.write((HERE/name).read_bytes())
        if sha(archive/name) != code[name]:
            raise ValueError('Execution archive differs')
    write(archive/'FILES.json', code)
    print('MIXTURE_PRELAUNCH_FROZEN; no model was run')


if __name__ == '__main__':
    main()
