"""Join a selected fresh byte window to the existing exact qualifier schema.

The exact consumer body remains unchanged. A separate external provenance
envelope anchors the observed window and the consumer; the closed packet set is
never mutated. Prospective schema values exist only inside a named CPU fixture.
"""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import argparse, hashlib, json

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]

def module(path, name):
    value = ModuleType(name); value.__file__ = str(path)
    exec(compile(path.read_bytes(), str(path), 'exec'), value.__dict__)
    return value

R = module(HERE / 'refresh_final.py', 'CPU_actual_review_helpers')
Q = module(PACKAGE / 'import_runtime_qualification.py', 'CPU_actual_review_qualifier')

def external(path, file_pin, body_pin):
    raw = path.read_bytes()
    R.require(hashlib.sha256(raw).hexdigest() == file_pin, 'externally selected actual-window file pin')
    value = json.loads(raw)
    R.require(value['sha256'] == body_pin == R.digest({k: v for k, v in value.items() if k != 'sha256'}), 'externally selected actual-window body pin')
    return value, {'path': str(path.absolute()), 'file_sha256': file_pin, 'body_sha256': body_pin}

def review_body(config, runtime):
    return R.seal({'kind': 'ROOT_REVIEWED_IMPORT_RUNTIME_BYTE_ROSTER_SOURCE_CACHE_CLOSURE', 'source_commit': config['source_commit'], 'source_inventory_sha256': R.digest(config['source_files']), 'runtime_closure_sha256': R.digest(runtime['files']), 'rosters_sha256': R.digest(runtime['rosters']), 'cache_policy_sha256': R.digest(runtime['cache_policy']), 'library_search_policy_sha256': R.digest(runtime['library_search_policy']), 'measurement_policy_sha256': R.digest(runtime['measurement_policy']), 'measurement_grammar_sha256': R.digest(runtime['measurement_grammar']), 'complete_current_byte_review': True})

def join(window_path, file_pin, body_pin, output, suffix='005'):
    R.require(not output.exists(), 'new review consumer directory; preserve predecessors')
    window, window_ref = external(window_path, file_pin, body_pin)
    config = R.read_sealed(HERE / ('IMPORT_OBSERVATION_CONFIG_' + suffix + '.json'))
    runtime = R.read_sealed(HERE / ('IMPORT_OBSERVATION_RUNTIME_' + suffix + '.json'))
    registration = R.read_sealed(HERE / ('IMPORT_OBSERVATION_REGISTRATION_' + suffix + '.json'))
    recipe = R.read_sealed(HERE / ('EXACT_IMPORT_RUN_RECIPE' + suffix + '.json'))
    selection = R.read_sealed(HERE / ('ACTUAL_SOURCE_RUNTIME_SELECTION' + suffix + '.json'))
    selected = {**config['source_files'], **runtime['files']}
    R.require(window['kind'] == 'FRESH_PASSIVE_IMPORT_SOURCE_RUNTIME_BYTE_REVIEW_V1' and window['implementation_commit'] == recipe['implementation_commit'] == config['source_commit'], 'actual-window exact selected implementation')
    R.require(window['source_files'] == config['source_files'] == selection['source_files'] and window['runtime_files'] == runtime['files'] == selection['runtime_files'], 'actual-window exact selected complete maps')
    observed = {p: {k: d[k] for k in ('bytes', 'sha256', 'role')} for p, d in window['observed_files'].items()}
    R.require(observed == selected and window['actual_hashed_bytes'] == sum(d['bytes'] for d in selected.values()) and window['physical_read_counters'] == {'opens': len(selected), 'bytes': sum(d['bytes'] for d in selected.values()), 'opened_metadata_checks': 2 * len(selected)}, 'actual-window complete original physical read denominator')
    R.require(window['rosters_before'] == window['rosters_after'] == runtime['rosters'] and len(runtime['rosters']) == 18, 'actual-window exact18 roster boundaries')
    R.require(window['whole_selected_byte_review_completed'] is True and window['native_runtime_qualified'] is window['native_launch_authorized'] is window['native_authority_issued'] is window['native_context_minted'] is window['native_API_called'] is window['Torch_psutil_ctypes_imported'] is False, 'actual-window no native or launch authority')
    R.require(window['historical_cache_pairs'] == 59 and window['additional_passive_cache_pairs'] == 8 and len(window['passive_cache_code_equivalence']) == 67 and all(r['complete_code_source_equivalence'] is True and r['cache_executed'] is r['source_execution_history_proven'] is False for r in window['passive_cache_code_equivalence'].values()), 'actual-window complete67 passive code equivalences')
    R.require(window['git_before'] == window['git_after'] == {'HEAD': config['source_commit'], 'dirty_porcelain': ''} and window['single_window_seconds'] <= 600 and window['byte_budget'] == 16 * 1024**3 and window['deadline_seconds'] == 600, 'actual-window unchanged original budgets and Git state')
    for name in ('config', 'runtime', 'registration'):
        expected = recipe['packet_references'][name]
        actual = window['metadata_anchors'][name]
        R.require(all(actual[k] == expected[k] for k in ('path', 'file_sha256', 'body_sha256')), 'window exact selected packet metadata anchors')
    output.mkdir(parents=True)
    consumer = review_body(config, runtime)
    consumer_ref = R.write(output / 'ROOT_REVIEWED_IMPORT_RUNTIME_BYTE_ROSTER_SOURCE_CACHE_CLOSURE.json', consumer)
    provenance = R.seal({'kind': 'ACTUAL_IMPORT_BYTE_WINDOW_TO_EXACT_QUALIFIER_REVIEW_PROVENANCE', 'actual_window_reference': window_ref, 'exact_consumer_reference': consumer_ref, 'implementation_commit': config['source_commit'], 'source_inventory_sha256': consumer['source_inventory_sha256'], 'runtime_closure_sha256': consumer['runtime_closure_sha256'], 'actual_window_complete_maps_rosters_and67_cache_pairs_joined': True, 'independent_review_reference': None, 'native_runtime_qualified': False, 'native_launch_authorized': False, 'native_authority_issued': False})
    provenance_ref = R.write(output / 'ACTUAL_BYTE_REVIEW_PROVENANCE.json', provenance)
    # Reviewed reader seams consume the already anchored complete window. All
    # descriptor, role, roster, source/cache, observer and exact policy logic runs.
    # No native reader/gate/context is called; this is a prospective CPU shape.
    helpers = Q.original_helpers(); descriptor_calls = []; roster_calls = []
    def observed_descriptor(path, descriptor, **kwargs):
        R.require(observed.get(str(path)) == descriptor, 'fresh-window selected descriptor and role join')
        descriptor_calls.append(str(path))
    def observed_roster(root, suffixes, excluded, *, recursive=True, budget=None):
        matches = [row for row in window['rosters_after'].values() if (row['root'], row['suffixes'], row['excluded_directory_names'], row['recursive']) == (str(root), suffixes, excluded, recursive)]
        R.require(len(matches) == 1, 'fresh-window exact physical roster reader join')
        roster_calls.append(str(root))
        return matches[0]['files']
    helpers.file_descriptor = observed_descriptor
    helpers.physical_roster = observed_roster
    prospective_config = deepcopy(config); prospective_config.pop('sha256'); prospective_config['native_launch_authorized'] = True; prospective_config = R.seal(prospective_config)
    prospective_registration = deepcopy(registration); prospective_registration.pop('sha256'); prospective_registration.update(native_launch_authorized=True, config_body_sha256=prospective_config['sha256'], qualification_review_path=consumer_ref['path'], qualification_review_file_sha256=consumer_ref['file_sha256'], qualification_review_body_sha256=consumer_ref['body_sha256']); prospective_registration = R.seal(prospective_registration)
    fixture = {'config': prospective_config, 'runtime': runtime, 'registration': prospective_registration, 'expected_head': 'b' * 40}
    R.require(config['source_commit'] != fixture['expected_head'], 'explicit CPU dispatch placeholder distinct from actual implementation')
    result = Q._qualify(fixture, Q.QualificationBudget(), helpers)
    R.require(result['current_module_execution_proven'] is result['native_effects_or_authority_issued'] is False and len(descriptor_calls) == len(selected) and len(roster_calls) == 18, 'complete qualifier CPU reader replay with no authority')
    preview = R.seal({'kind': 'CPU_PROSPECTIVE_APPROVED_SCHEMA_REVIEW_FIXTURE_ONLY', 'actual_closed_recipe_reference': recipe['packet_references'], 'proposed_config': prospective_config, 'proposed_runtime': runtime, 'proposed_registration': prospective_registration, 'prospective_expected_head_CPU_placeholder': fixture['expected_head'], 'exact_review_consumer_reference': consumer_ref, 'actual_byte_window_provenance_reference': provenance_ref, 'instantiation_rule_after_explicit_permission': 'Create separate approved config and registration with only native_launch_authorized=True and the reviewed qualifier reference; preserve every selected source/runtime/resource/profile field. Bind actual separately committed dispatch HEAD and actual packet file/body anchors through a new out-of-band direct-human owner grant/receipt. Closed005 stays historical and unchanged.', 'real_owner_permission_required': True, 'real_dispatch_commit_required': True, 'native_context_or_gate_called': False, 'native_launch_authorized': False, 'native_authority_issued': False})
    preview_ref = R.write(output / 'CPU_PROSPECTIVE_APPROVED_SCHEMA_FIXTURE.json', preview)
    replay = R.seal({'kind': 'CPU_FULL_QUALIFIER_REPLAY_FROM_ACTUAL_WINDOW', 'actual_window_reference': window_ref, 'consumer_reference': consumer_ref, 'provenance_reference': provenance_ref, 'preview_reference': preview_ref, 'selected_public_qualifier': {'path': str(PACKAGE / 'import_runtime_qualification.py'), 'file_sha256': hashlib.sha256((PACKAGE / 'import_runtime_qualification.py').read_bytes()).hexdigest()}, 'source_count': len(config['source_files']), 'runtime_count': len(runtime['files']), 'descriptor_calls': len(descriptor_calls), 'roster_calls': len(roster_calls), 'result': result, 'file_descriptor_and_physical_roster_from_selected_actual_window': True, 'real_dispatch_Git_join_performed': False, 'native_context_or_gate_called': False, 'native_launch_authorized': False, 'native_authority_issued': False})
    replay_ref = R.write(output / 'CPU_FULL_QUALIFIER_ACTUAL_WINDOW_REPLAY.json', replay)
    return {'consumer_reference': consumer_ref, 'provenance_reference': provenance_ref, 'preview_reference': preview_ref, 'replay_reference': replay_ref}

if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--window', type=Path, required=True); parser.add_argument('--window-file', required=True); parser.add_argument('--window-body', required=True); parser.add_argument('--output', type=Path, required=True); parser.add_argument('--suffix', default='005')
    args = parser.parse_args()
    print(R.canonical(join(args.window, args.window_file, args.window_body, args.output, args.suffix)))
