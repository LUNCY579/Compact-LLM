"""Freeze and verify a descriptive join; never execute or select an operator."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
PROGRAMME, REPO = HERE.parents[2], HERE.parents[4]
STAGE = PROGRAMME / 'cross_scale/generation_stage/v1'
STAGE_ID = 'dcfc67e0828a3b47da41980a88ce47813b4f8b63c184502f6898bbeaf7222fd3'
CONFIG = STAGE / 'configs/generation_screen_gpu_v1.json'
STAGE_REG = STAGE / 'evidence/00_preregistration' / STAGE_ID / 'registration.json'
GENERATION = STAGE / 'evidence/04_small_generation' / STAGE_ID
SCHEMA = 'cross_scale_descriptive_fidelity_relation_v1'


def check(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    def pairs(items):
        value = {}
        for key, item in items:
            check(key not in value, 'duplicate JSON key: ' + key)
            value[key] = item
        return value
    def bad(value):
        raise ValueError('nonfinite JSON value: ' + value)
    return json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=pairs, parse_constant=bad)


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def within(path, root=REPO):
    path = Path(path)
    path = (REPO / path if not path.is_absolute() else path).resolve()
    check(path.is_relative_to(Path(root).resolve()), 'path outside authorized subtree')
    return path


def relative(path):
    return within(path).relative_to(REPO).as_posix()


def git(*args):
    return subprocess.check_output(['git', '-C', str(REPO), *args], stderr=subprocess.DEVNULL)


def committed(path, ref='HEAD'):
    check(ref == 'HEAD' or bool(re.fullmatch('[0-9a-f]{40}', ref)), 'invalid commit identity')
    check(hashlib.sha256(git('show', ref + ':' + relative(path))).hexdigest() == sha(path),
          'file bytes differ from committed artifact: ' + relative(path))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
        stream.write('\n')


def source_files():
    paths = sorted([*HERE.glob('*.py'), HERE / 'protocol.md'])
    check({'run_report.py', 'report_core.py', 'qualification_bytes.py', 'protocol.md'} <= {p.name for p in paths},
          'diagnostic runtime source is incomplete')
    return {relative(path): sha(path) for path in paths}


def stage_inputs():
    config, registration = read(CONFIG), read(STAGE_REG)
    contract = {k: v for k, v in registration.items() if k not in
                ('status', 'registration_sha256', 'registered_against', 'registered_utc')}
    check(registration['status'] == 'FROZEN' and registration['registration_sha256'] == STAGE_ID
          and canonical(contract) == STAGE_ID, 'invalid frozen Stage 2 registration')
    check(config == registration['config'] and sha(CONFIG) == registration['config_sha256'],
          'Stage 2 configuration changed')
    check(config['role'] == 'architecture_dev' and config['track'] == 'cross_scale'
          and config['confirmatory_generation_licensed'] is False, 'wrong data role')
    dataset_path = within(config['dataset_manifest'], PROGRAMME / 'data/generated/v2/cross_scale')
    check(sha(dataset_path) == config['dataset_manifest_sha256'] == registration['dataset_manifest_sha256'],
          'frozen B dataset manifest changed')
    dataset = read(dataset_path)
    info = dataset['datasets']['architecture_dev']
    check(dataset['track'] == 'cross_scale' and info['path'] == 'architecture_dev.jsonl'
          and info['examples'] == info['scenarios'] == 128, 'incorrect architecture data role')
    architecture = within(dataset_path.parent / info['path'], dataset_path.parent)
    check(sha(architecture) == info['sha256'], 'frozen architecture file changed')
    export_path = within(config['upstream_export']['manifest_path'], STAGE)
    check(sha(export_path) == config['upstream_export']['manifest_sha256'], 'export changed')
    export = read(export_path)
    check(export['status'] == 'COMPLETE' and export['source_objectives'] == {'XG':'B3','XS':'BS','XR':'BR'},
          'export objectives differ')
    upstream = export['upstream']
    check(upstream['status'] == 'VERIFIED_MECHANISM_GO', 'upstream mechanism was not admitted')
    aggregate = within(upstream['aggregate_directory'], PROGRAMME / 'cross_scale/evidence/01_alignment_matrix')
    audit = within(upstream['audit_directory'], PROGRAMME / 'cross_scale/evidence/07_independent_audit')
    paths = {'screen_config': CONFIG, 'screen_registration': STAGE_REG, 'bridge_export': export_path,
             'matrix': aggregate / 'matrix.json', 'selection': aggregate / 'selection.json',
             'alignment_audit': audit / 'audit_receipt.json', 'dataset_manifest': dataset_path,
             'architecture_data': architecture}
    for key, digest_key in [('matrix', 'matrix_sha256'), ('selection', 'selection_sha256'),
                            ('alignment_audit', 'audit_receipt_sha256')]:
        check(sha(paths[key]) == upstream[digest_key], 'alignment provenance changed: ' + key)
    selection = read(paths['selection'])
    check(read(paths['alignment_audit'])['status'] == 'PASS', 'independent alignment audit did not pass')
    matrix = read(paths['matrix'])
    pairs = matrix['pairs']
    check(len(pairs) == 16 and {(p['supa_layer'], p['donor_layer']) for p in pairs}
          == {(s, d) for s in (8, 12, 16, 19) for d in (12, 16, 20, 24)}
          and all(set(p['objectives']) == {'B0', 'B1', 'B2', 'B3', 'BS', 'BR'} for p in pairs),
          'complete 96-cell alignment matrix required')
    check(selection['status'] == 'MECHANISM_GO' and selection['selected'] == [upstream['selected']],
          'complete frozen mechanism GO required')
    check((export['context']['supa_layer'], export['context']['donor_layer'],
           export['context']['selected_objective']) == (19, 24, 'B3'), 'wrong selected transformation')
    for name, expected in registration['sources'].items():
        path = within(name)
        check(sha(path) == expected, 'registered Stage 2 source changed: ' + name)
        if path.is_relative_to(PROGRAMME):
            committed(path)
    for path in paths.values():
        committed(path)
    return config, registration, paths


def registration_contract():
    _, _, paths = stage_inputs()
    sources = source_files()
    for path in sources:
        committed(REPO / path)
    return {'schema': SCHEMA, 'track': 'cross_scale', 'analysis_role': 'descriptive_only',
            'stage_registration': STAGE_ID, 'sources': sources,
            'inputs': {key: {'path': relative(path), 'sha256': sha(path)} for key, path in paths.items()},
            'selection_modified': False, 'confirmation_access': False,
            'model_execution': False, 'inference_claim': False}


def freeze():
    check(not list(GENERATION.rglob('started.json')), 'freeze must precede every B generation attempt')
    contract = registration_contract()
    digest = canonical(contract)
    path = HERE / 'evidence/preregistration' / digest / 'registration.json'
    write(path, {'status': 'FROZEN', 'registration_sha256': digest, 'contract': contract,
                 'frozen_utc': datetime.now(timezone.utc).isoformat(),
                 'commit_sha': git('rev-parse', 'HEAD').decode().strip()})
    return path


def verify(path):
    path = within(path, HERE / 'evidence/preregistration')
    value = read(path)
    check(value['status'] == 'FROZEN' and canonical(value['contract']) == value['registration_sha256'],
          'diagnostic registration is not frozen')
    check(value['contract'] == registration_contract(), 'diagnostic source/input contract changed')
    committed(path)
    return value


def prospective_start(start_path, registration_path, registration):
    start = read(start_path)
    check(start['registration_sha256'] == STAGE_ID and start['role'] == 'architecture_dev',
          'foreign generation start')
    frozen = datetime.fromisoformat(registration['frozen_utc'].replace('Z', '+00:00'))
    began = datetime.fromisoformat(start['start_utc'].replace('Z', '+00:00'))
    check(frozen.tzinfo is not None and began.tzinfo is not None and frozen <= began,
          'diagnostic was frozen after generation began')
    committed(registration_path, start['commit_sha'])
    for source in registration['contract']['sources']:
        committed(REPO / source, start['commit_sha'])
    return start


def artifact_hashes(directory):
    result = {}
    for path in sorted(directory.rglob('*')):
        if not path.is_file():
            continue
        resolved = within(path, directory)
        check(resolved == path.absolute(), 'aliased raw artifact')
        name = path.relative_to(directory).as_posix()
        if name not in ('run_manifest.json', 'abandonment.json'):
            result[name] = sha(path)
    return result


def generation_terminals(starts):
    terminal_paths = set()
    for start_path in starts:
        terminals = [start_path.parent / name for name in ('run_manifest.json', 'abandonment.json')
                     if (start_path.parent / name).is_file()]
        check(len(terminals) == 1, 'unfinished or ambiguous generation attempt')
        terminal_paths.add(str(terminals[0].resolve()))
    observed = {str(p.resolve()) for name in ('run_manifest.json', 'abandonment.json')
                for p in GENERATION.rglob(name)}
    check(terminal_paths == observed, 'generation start/terminal inventory differs')
    return terminal_paths


def verified_screen(directory, registration_path, diagnostic):
    from qualification_bytes import verify_qualification
    directory = within(directory, STAGE / 'evidence/05_screen_audit' / STAGE_ID)
    check(directory.parent.name == STAGE_ID and bool(re.fullmatch('attempt_[0-9]{3,}', directory.name)),
          'explicit completed screen audit directory required')
    names = {'audit_receipt.json', 'report.json', 'selection.json', 'raw_manifest_hashes.json'}
    check({p.name for p in directory.iterdir()} == names, 'screen audit artifact inventory differs')
    for name in names:
        committed(directory / name)
    receipt = read(directory / 'audit_receipt.json')
    config, stage_registration, _ = stage_inputs()
    expected = {'schema_version': 'cross_scale_stage2_generation_shard_v1', 'track': 'cross_scale',
                'role': 'architecture_dev', 'registration_sha256': STAGE_ID, 'status': 'COMPLETE',
                'config_sha256': canonical(config), 'config_file_sha256': sha(CONFIG),
                'sources': stage_registration['sources'], 'source_sha256': canonical(stage_registration['sources']),
                'analysis_role': 'candidate_selection', 'n_per_complete_cell': 128,
                'independent_rescoring': True, 'first_logit_metrics_recomputed': True, 'confirmation_accessed': False}
    check(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items()),
          'screen audit contract does not match frozen inputs')
    check(receipt['artifacts_sha256'] == {name: sha(directory / name) for name in names - {'audit_receipt.json'}},
          'screen audit artifact bytes changed')
    report, selection = read(directory / 'report.json'), read(directory / 'selection.json')
    check(selection == {'status': report['status'], 'selected': report['selected'],
                       'registration_sha256': STAGE_ID, 'config_sha256': canonical(config),
                       'confirmatory_generation_licensed': False}, 'screen selection/report disagree')
    complete = {name for name, cell in report['cells'].items() if cell['status'] == 'COMPLETE'}
    failed = {name for name, cell in report['cells'].items() if cell['status'] == 'FAILED'}
    check(set(receipt['complete_cells']) == complete and set(receipt['failed_cells']) == failed
          and complete | failed == set(config['cells']) and 'X0' in complete, 'screen cell coverage differs')
    for name in failed:
        check(report['cells'][name]['failure'] == receipt['failed_cells'][name], 'failed-cell receipt changed')
    manifests = read(directory / 'raw_manifest_hashes.json')
    starts = list(GENERATION.rglob('started.json'))
    check(starts, 'no generation starts for claimed complete X0')
    verified_hashes = {}
    qualification_results = {}
    for name in failed:
        failure = receipt['failed_cells'][name]
        if failure['phase'] == 'qualification':
            qualification_results[name] = failure
    for start_path in starts:
        prospective_start(start_path, registration_path, diagnostic)
    check(generation_terminals(starts) == set(manifests), 'generation start/terminal inventory differs')
    for name, expected_hash in manifests.items():
        path = within(name, GENERATION)
        check(path.name in ('run_manifest.json', 'abandonment.json') and sha(path) == expected_hash,
              'raw terminal manifest changed')
        committed(path)
        raw = read(path)
        check(raw['registration_sha256'] == STAGE_ID and raw['config_sha256'] == canonical(config),
              'raw manifest belongs to another contract')
        cell = raw['cell_id']
        check(cell in config['cells'] and raw['cell'] == config['cells'][cell]
              and raw['status'] in ('COMPLETE', 'FAILED', 'ABANDONED'), 'unknown raw cell/status')
        start_path = path.parent / 'started.json'
        start = read(start_path)
        committed(start_path)
        check(raw['started_sha256'] == sha(start_path)
              and all(raw.get(k) == v for k, v in start.items() if k != 'status'), 'raw started contract changed')
        artifacts = artifact_hashes(path.parent)
        check(artifacts == raw['artifacts_sha256'], 'raw artifact inventory/bytes changed after screen audit')
        for relative_name, digest in artifacts.items():
            verified_hashes[relative(path.parent / relative_name)] = digest
        verified_hashes[relative(path)] = expected_hash
        qualification = raw['runtime_qualification']
        check(cell not in qualification_results or qualification_results[cell] == qualification,
              'qualification changed between cell attempts')
        qualification_results[cell] = qualification
    check(set(qualification_results) == set(config['cells']), 'missing cell qualification provenance')
    for cell, qualification in qualification_results.items():
        hashes = verify_qualification(qualification, cell, STAGE, STAGE_ID, stage_registration['sources'],
                                      canonical(config), read, sha, committed)
        verified_hashes.update({relative(path): digest for path, digest in hashes.items()})
    return report, config, {relative(directory / name): sha(directory / name) for name in names}, verified_hashes


def build_report(registration_path, audit_directory):
    from report_core import analyze
    registration_path = within(registration_path, HERE / 'evidence/preregistration')
    diagnostic = verify(registration_path)
    report, config, audited_files, raw_files = verified_screen(audit_directory, registration_path, diagnostic)
    matrix_path = REPO / diagnostic['contract']['inputs']['matrix']['path']
    result = analyze(read(matrix_path), report, config)
    again = verified_screen(audit_directory, registration_path, diagnostic)
    check(again == (report, config, audited_files, raw_files), 'screen inputs changed during reporting')
    check(verify(registration_path) == diagnostic, 'diagnostic inputs changed during reporting')
    output = HERE / 'evidence/reports' / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ'))
    output.mkdir(parents=True, exist_ok=False)
    write(output / 'report.json', result)
    write(output / 'receipt.json', {'status': 'COMPLETE', 'schema': SCHEMA,
          'created_utc': datetime.now(timezone.utc).isoformat(), 'registration_path': relative(registration_path),
          'registration_sha256': sha(registration_path), 'audit_files': audited_files, 'raw_manifests': raw_files,
          'report_sha256': sha(output / 'report.json'), 'selection_modified': False,
          'model_execution': False, 'confirmation_accessed': False,
          'limitations': 'Reconciles committed audit records and artifact bytes; does not rerun inference, scoring or tensor metrics.'})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('freeze', 'verify', 'report'))
    parser.add_argument('--registration', type=Path)
    parser.add_argument('--screen-audit', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        print(freeze())
    elif args.action == 'verify':
        check(args.registration is not None, 'registration required')
        print(json.dumps({'status': 'VERIFIED', 'registration': verify(args.registration)['registration_sha256']}))
    else:
        check(args.registration is not None and args.screen_audit is not None, 'registration and screen audit required')
        print(build_report(args.registration, args.screen_audit))


if __name__ == '__main__':
    main()
