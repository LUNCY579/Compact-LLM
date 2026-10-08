"""Refresh a closed import recipe from an explicitly selected implementation.

This CPU metadata builder reads package sources and Git blobs, never framework
modules or selected runtime payloads. The later complete-current-byte review,
distinct dispatch commit and human grant stay explicit pending witnesses.
"""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import argparse, hashlib, json, subprocess

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
ROOT = PACKAGE.parents[1]
PASSIVE = PACKAGE / 'evidence/import_launch_corrections_reviewed_cpu_20261007/frozen_ram_source_fix002/PASSIVE_IMPORT_BOOTSTRAP_ASSUMPTIONS002.json'
PASSIVE_PIN = '0fe6d869bbf61a3125841164a0e11d2eb2a8ef46743f89bd7423259b02d62d91'
RAW_LEAVES = ('OWNED_EXTERNAL_ANCHORS.json', 'SUPERVISOR_RECEIPT.json',
    'RESOURCE_OBSERVATIONS.json', 'supervisor_ownership.json',
    'PHASE_OBSERVATIONS.jsonl', 'PHASE_ACCOUNTING_FINAL.json',
    'IMPORT_LIBRARY_MEASUREMENT.json', 'worker_result.json', 'worker.stdout.log',
    'worker.stderr.log', 'supervisor_heartbeat.jsonl',
    'SHORT_RAM_ABORT_DIAGNOSTIC.json', 'WORKER_LAUNCH_FAILURE.json')
ANCHORS = ('expected_config_file', 'expected_config_body', 'registration_path',
    'expected_registration_file', 'expected_registration_body', 'runtime_path',
    'expected_runtime_file', 'expected_runtime_body', 'expected_head')
REVIEW_SOURCE = PACKAGE / 'artifacts/import_current_byte_review_work/review_current_bytes.py'
REVIEW_SOURCE_PIN = '7286041069771edb48f5a801155c0c8903648279b3043138bddd697e7ab7f8c0'

def require(value, message):
    if not value:
        raise ValueError(message)

def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()

def seal(value):
    require('sha256' not in value, 'unsealed builder value')
    return {**value, 'sha256': digest(value)}

def read_sealed(path, expected_file=None):
    raw = path.read_bytes()
    if expected_file is not None:
        require(hashlib.sha256(raw).hexdigest() == expected_file, 'selected passive receipt file bytes')
    value = json.loads(raw)
    require(value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}), 'exact input packet seal')
    return value

def write(path, value):
    raw = (canonical(value) + '\n').encode()
    with path.open('xb') as handle:
        handle.write(raw)
    return {'path': str(path), 'file_sha256': hashlib.sha256(raw).hexdigest(), 'body_sha256': value['sha256']}

def descriptor(path):
    raw = path.read_bytes()
    role = {'.json': 'source_json', '.md': 'source_markdown', '.c': 'native_observer_source', '.h': 'native_observer_source'}.get(path.suffix, 'python_source')
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'role': role}

def git(*args, data=None):
    return subprocess.run(['git', '-c', 'maintenance.auto=false', '-c', 'gc.auto=0', *args], cwd=ROOT, input=data, capture_output=True, check=True, timeout=120).stdout

def source_inventory(implementation):
    require(type(implementation) is str and len(implementation) == 40 and all(c in '0123456789abcdef' for c in implementation), 'explicit full implementation commit')
    require(git('rev-parse', 'HEAD').decode().strip() == implementation, 'selected implementation is actual HEAD before recipe construction')
    status = git('status', '--porcelain=v1', '-z', '--untracked-files=all')
    require(not status, 'clean implementation checkout before source selection')
    path = PACKAGE / 'import_runtime_qualification.py'
    qualifier = ModuleType('CPU_selected_public_import_qualifier')
    qualifier.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), qualifier.__dict__)
    plan = read_sealed(PACKAGE / 'evidence/import_public_graph_reviewed_cpu_20261007/frozen_public_graph002/PUBLIC_IMPORT_SOURCE_PLAN002.json')
    sources = {p: descriptor(Path(p)) for p in qualifier.B.package_source_roster()}
    sources.update(plan['durable_dependencies'])
    for row in plan['sources'].values():
        sources[row['target_path']] = descriptor(Path(row['target_path']))
    for name in ('import_dispatch_postrun.py', 'import_terminal_audit.py'):
        sources[str(PACKAGE / name)] = descriptor(PACKAGE / name)
    observer = read_sealed(PACKAGE / 'configs/SYNTH_OBSERVER_STARTUP_RUNTIME_009.json')['library_event_observer']
    sources.update(observer['native_source_files'])
    sources[observer['build_receipt']['path']] = descriptor(Path(observer['build_receipt']['path']))
    ordered = sorted(sources)
    specs = [implementation + ':' + Path(p).relative_to(ROOT).as_posix() for p in ordered]
    blobs = git('cat-file', '--batch', data=('\n'.join(specs) + '\n').encode())
    cursor = 0
    for path in ordered:
        end = blobs.index(b'\n', cursor)
        header = blobs[cursor:end].decode('ascii').split()
        require(len(header) == 3 and header[1] == 'blob' and header[2].isdecimal(), 'selected source actual Git blob')
        size = int(header[2]); cursor = end + 1
        raw = blobs[cursor:cursor + size]; cursor += size
        require(blobs[cursor:cursor + 1] == b'\n', 'exact Git batch blob boundary'); cursor += 1
        require(raw == Path(path).read_bytes() and descriptor(Path(path)) == sources[path], 'source disk descriptor equals immutable implementation blob')
    require(cursor == len(blobs), 'no unconsumed Git source blobs')
    require(git('rev-parse', 'HEAD').decode().strip() == implementation and git('status', '--porcelain=v1', '-z', '--untracked-files=all') == status, 'source selection HEAD and checkout stable')
    return sources, qualifier

def prepare_reader(implementation, selection_reference, suffix):
    """Preserve the reviewed algorithm, substitute only selected metadata pins."""
    raw = REVIEW_SOURCE.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == REVIEW_SOURCE_PIN, 'exact predecessor passive full-window reader')
    text = raw.decode('utf-8')
    substitutions = {
        "SELECTION=INPUTS/'ACTUAL_SOURCE_RUNTIME_SELECTION002.json'": "SELECTION=INPUTS/'ACTUAL_SOURCE_RUNTIME_SELECTION" + suffix + ".json'",
        "SELECTION_PIN='96007671eed07e5ba062ebd6f429b57ca1d06283481ab8ba26f3914389763466'": "SELECTION_PIN='" + selection_reference['file_sha256'] + "'",
        "EXPECTED_HEAD='d39d64b1447c8f0baffdebcc4a0a1092f76bbd88'": "EXPECTED_HEAD='" + implementation + "'",
        "IMPORT_OBSERVATION_RUNTIME_003.json": 'IMPORT_OBSERVATION_RUNTIME_' + suffix + '.json',
        "IMPORT_OBSERVATION_CONFIG_003.json": 'IMPORT_OBSERVATION_CONFIG_' + suffix + '.json',
        "IMPORT_OBSERVATION_REGISTRATION_003.json": 'IMPORT_OBSERVATION_REGISTRATION_' + suffix + '.json',
        'selection[\'runtime_count\']==8724': 'selection[\'runtime_count\']==8725',
        "len(pairs)==66,'exact59 plus7 unique passive cache pairs'": "len(pairs)==67,'exact59 plus8 unique passive cache pairs'",
        "'additional_passive_cache_pairs':7": "'additional_passive_cache_pairs':8",
        'FRESH_IMPORT_CURRENT_BYTE_REVIEW001.json': 'FRESH_IMPORT_CURRENT_BYTE_REVIEW' + suffix + '.json',
        'FRESH_IMPORT_CURRENT_BYTE_REVIEW_FAILURE001.json': 'FRESH_IMPORT_CURRENT_BYTE_REVIEW_FAILURE' + suffix + '.json',
    }
    counts = {}
    for old, new in substitutions.items():
        count = text.count(old)
        require(count > 0, 'reviewed reader substitution exists: ' + old)
        if old.startswith(('SELECTION=', 'SELECTION_PIN=', 'EXPECTED_HEAD=', 'selection[', 'len(pairs)', "'additional_passive_cache_pairs'")):
            require(count == 1, 'single exact reader constant substitution')
        counts[old] = count
        text = text.replace(old, new)
    compile(text, str(REVIEW_SOURCE), 'exec')
    directory = PACKAGE / 'artifacts' / ('import_final_byte_review_work_' + suffix)
    require(not directory.exists(), 'new final byte-review workdir absent')
    directory.mkdir()
    target = directory / 'review_current_bytes.py'
    target.write_bytes(text.encode('utf-8'))
    reference = {'path': str(target), 'file_sha256': hashlib.sha256(target.read_bytes()).hexdigest()}
    write(directory / 'READER_SUCCESSOR_SELECTION.json', seal({'kind': 'STRICT_PASSIVE_IMPORT_BYTE_READER_SUCCESSOR_SELECTION', 'implementation_commit': implementation, 'predecessor_reader': {'path': str(REVIEW_SOURCE), 'file_sha256': REVIEW_SOURCE_PIN}, 'selected_reader': reference, 'substitution_counts': counts, 'selection_reference': selection_reference, 'runtime_count': 8725, 'cache_pairs': 67, 'native_runtime_qualified': False, 'full_byte_reader_executed': False, 'native_launch_authorized': False}))
    return reference

def build(implementation, suffix='005'):
    require(len(suffix) == 3 and suffix.isdecimal(), 'exact recipe successor suffix')
    targets = [HERE / (stem + suffix + '.json') for stem in ('IMPORT_SOURCE_RECIPE', 'ACTUAL_SOURCE_RUNTIME_SELECTION', 'EXACT_IMPORT_RUN_RECIPE', 'IMPORT_OBSERVATION_CONFIG_', 'IMPORT_OBSERVATION_RUNTIME_', 'IMPORT_OBSERVATION_REGISTRATION_')]
    require(not any(p.exists() for p in targets), 'preserve prior recipes without overwrite')
    sources, qualifier = source_inventory(implementation)
    config = deepcopy(read_sealed(HERE / 'IMPORT_OBSERVATION_CONFIG_004.json'))
    runtime = deepcopy(read_sealed(HERE / 'IMPORT_OBSERVATION_RUNTIME_004.json'))
    registration = deepcopy(read_sealed(HERE / 'IMPORT_OBSERVATION_REGISTRATION_004.json'))
    prior_recipe = read_sealed(HERE / 'EXACT_IMPORT_RUN_RECIPE004.json')
    passive = read_sealed(PASSIVE, PASSIVE_PIN)
    # The selected public successor must explicitly admit only the exact V2 copy
    # extension. This call validates metadata, not current runtime bytes/history.
    overlay = qualifier.passive_preload_overlay(passive)
    require(len(overlay) == 11 and len(passive['additional_cache_candidates']) == 8, 'exact extended copy preload population')
    require(runtime['cache_policy']['additional_preload_sources'] == overlay and len(runtime['files']) == 8725, 'retained registered runtime descriptor map')
    require(runtime['measurement_policy']['initial_loaded_libraries'] == passive['source_derived_baseline_expectation'] and len(passive['source_derived_baseline_expectation']) == 39, 'unchanged falsifiable baseline39')
    for value in (config, runtime, registration):
        value.pop('sha256'); value['source_commit'] = implementation
    config['source_files'] = registration['source_files'] = sources
    config['runtime_files'] = registration['runtime_files'] = runtime['files']
    main = config['main_source']
    runtime['cache_policy']['separate_main_source'] = {'path': main, 'descriptor': sources[main]}
    transitive = {p: d for p, d in sources.items() if Path(p).is_relative_to(PACKAGE / 'evidence')}
    source_ref = write(HERE / ('IMPORT_SOURCE_RECIPE' + suffix + '.json'), seal({'kind': 'REGISTERED_IMPORT_TRANSITIVE_SOURCE_ROSTER', 'source_files': transitive}))
    runtime['metadata_references']['source_roster'] = source_ref
    runtime['metadata_references']['additional_preload'] = {'path': str(PASSIVE), 'file_sha256': PASSIVE_PIN, 'body_sha256': passive['sha256']}
    output = Path(prior_recipe['output_directory'])
    preserved = Path(prior_recipe['preserved_directory'])
    require(not output.exists() and not preserved.exists(), 'new registered output and preservation directories absent')
    exceptions = {str(preserved / leaf): {'role': 'evidence_output'} for leaf in (*RAW_LEAVES, 'PRESERVATION.json', *('lease_' + name for name in ('reservation.json', 'worker.json', 'resume.json', 'terminal.json')))}
    for path in sources:
        exceptions[str(preserved / 'source_snapshots' / (hashlib.sha256(path.encode()).hexdigest() + Path(path).suffix))] = {'role': 'evidence_source_snapshot', 'source_path': path}
    registration['git_output_exceptions'] = exceptions
    paths = {name: HERE / ('IMPORT_OBSERVATION_' + name.upper() + '_' + suffix + '.json') for name in ('config', 'runtime', 'registration')}
    registration['config_path'] = str(paths['config'])
    qualifier.import_git_exception_policy(exceptions, sources, runtime['files'], runtime['library_event_observer'], qualifier.B)
    qualifier._source_roster({'config': config, 'runtime': runtime}, qualifier.B, qualifier.QualificationBudget())
    require(config['native_launch_authorized'] is registration['native_launch_authorized'] is config['native_authority_issued'] is False and config['owner_approval'] is registration['owner_approval'] is None, 'closed recipe grants no authority')
    runtime = seal(runtime); config = seal(config)
    registration.update(config_body_sha256=config['sha256'], runtime_body_sha256=runtime['sha256'])
    registration = seal(registration)
    refs = {name: write(paths[name], value) for name, value in (('config', config), ('runtime', runtime), ('registration', registration))}
    selection = seal({'kind': 'ACTUAL_IMPLEMENTATION_IMPORT_SOURCE_RUNTIME_SELECTION', 'implementation_commit': implementation, 'source_files': sources, 'runtime_files': runtime['files'], 'rosters': runtime['rosters'], 'source_count': len(sources), 'runtime_count': len(runtime['files']), 'source_inventory_sha256': digest(sources), 'runtime_inventory_sha256': digest(runtime['files']), 'actual_current_bytes_hashed_in_one_window': False, 'source_disk_bytes_equal_actual_implementation_blobs': True, 'runtime_payloads_read_by_builder': False, 'native_authority_issued': False})
    selection_ref = write(HERE / ('ACTUAL_SOURCE_RUNTIME_SELECTION' + suffix + '.json'), selection)
    anchors = {'expected_config_file': refs['config']['file_sha256'], 'expected_config_body': refs['config']['body_sha256'], 'registration_path': refs['registration']['path'], 'expected_registration_file': refs['registration']['file_sha256'], 'expected_registration_body': refs['registration']['body_sha256'], 'runtime_path': refs['runtime']['path'], 'expected_runtime_file': refs['runtime']['file_sha256'], 'expected_runtime_body': refs['runtime']['body_sha256'], 'expected_head': None}
    command = {'executable': runtime['cache_policy']['interpreter_profile']['executable'], 'argv_prefix': ['-I', '-S', '-B', str(PACKAGE / 'import_dispatch_postrun.py'), '--config', refs['config']['path'], '--output-dir', str(output), '--preserved-dir', str(preserved)], 'nine_anchors': anchors, 'separate_owner_references_required': True, 'required_owner_reference_arguments': ['--' + prefix + '-' + part for prefix in ('grant', 'owner-receipt') for part in ('path', 'file-sha256', 'body-sha256')]}
    recipe = seal({'kind': 'CLOSED_EXACT_IMPORT_ONLY_RUN_RECIPE', 'implementation_commit': implementation, 'external_dispatch_head': None, 'run_id': prior_recipe['run_id'], 'output_directory': str(output), 'preserved_directory': str(preserved), 'packet_references': refs, 'source_recipe': source_ref, 'source_runtime_selection': selection_ref, 'source_count': len(sources), 'runtime_count': len(runtime['files']), 'raw_future_leaf_allowlist': list(RAW_LEAVES), 'exact_preservation_exception_count': len(exceptions), 'owner_grant_reference': None, 'owner_receipt_reference': None, 'complete_current_byte_review': None, 'future_command': command, 'scope': qualifier.SCOPE, 'permission': qualifier.PERMISSION, 'resources': config['resources'], 'passive_extended_reference': runtime['metadata_references']['additional_preload'], 'preload_candidate_count': 11, 'passive_cache_candidate_total': 67, 'native_launch_authorized': False, 'native_runtime_qualified': False, 'native_authority_issued': False})
    recipe_ref = write(HERE / ('EXACT_IMPORT_RUN_RECIPE' + suffix + '.json'), recipe)
    reader_ref = prepare_reader(implementation, selection_ref, suffix)
    return {'implementation': implementation, 'sources': len(sources), 'runtime': len(runtime['files']), 'exceptions': len(exceptions), 'overlay': 11, 'cache_total': 67, 'selection_reference': selection_ref, 'recipe_reference': recipe_ref, 'reader_reference': reader_ref, 'native_launch_authorized': False}

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--implementation', required=True)
    parser.add_argument('--suffix', default='005')
    args = parser.parse_args()
    print(canonical(build(args.implementation, args.suffix)))
