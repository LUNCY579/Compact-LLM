"""Read-only source/JSON config review. Never invokes launch/runtime/model helpers."""
import ast, hashlib, json, math, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
ROOT = PACKAGE.parents[1]
CONFIG = PACKAGE / 'configs/SYNTH_SHORT_M0_ATOMIC_BATCH04_PHASE3_A1.json'
FILE_SHA = 'e6d56e4a592b68bc7e288022b98bd73e94a468f6243625579bfb3bcb7d3c0b4d'
BODY_SHA = 'f514a097645647184191a42acda0311d0f338484e9027011d5cd5598c212dd38'
PROFILE_SHA = '2d5547910aa57b0509a532972d899ceb5c913e41274bfa773d25536e6d8c9676'
AUDITOR = PACKAGE / 'artifacts/short_phase_audit_work'
AUDITOR_PINS = {'audit_adapter.py': '12a6bb2e1807f281d22b437dc086b6dc7fa9cc53a6f331f698165ab14c8a3533',
    'phase_replay.py': '270ec43ee8168f7032562095f6c07e49b20fa815e70cb7a648d6c2fb8dd23cbd',
    'test_phase_audit.py': '407a853721d6da3a62861dd27bf75cbf793659ed057f053b97a958f347fd1ba0',
    'fixture_support.py': '0645ea43da2febb0ea958e61fad86cfb07838cf1278869497ed3181a5c50988b',
    'pinned_dependencies/RAM2_auditor_d69810.py': 'd69810dccd11a32595dace73a788c8868efabb2a09f593fd33316e92dcb47442',
    'pinned_dependencies/PRESERVATION.json': 'ad2f4896cd6ceecaa0d8cd99dff055463e88cfd7748abf0d69919d5ae71262a3'}

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)
def digest(value):
    return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()
def filehash(path):
    path = Path(path)
    if path.suffix.lower() not in ('.py', '.json', '.md'):
        raise ValueError('review refuses binary file reads: ' + str(path))
    if path.is_symlink() or not path.is_file():
        raise ValueError('regular source/metadata required')
    return hashlib.sha256(path.read_bytes()).hexdigest()
def pairs(items):
    value = {}
    for key, item in items:
        if key in value:
            raise ValueError('duplicate JSON key')
        value[key] = item
    return value
def read(path):
    path = Path(path)
    if path.suffix != '.json' or path.stat().st_size > 8 * 1024**2:
        raise ValueError('bounded JSON metadata only')
    def finite(text):
        value = float(text)
        if not math.isfinite(value):
            raise ValueError('nonfinite JSON value')
        return value
    return json.loads(path.read_bytes(), object_pairs_hook=pairs, parse_float=finite,
        parse_constant=lambda text: (_ for _ in ()).throw(ValueError('nonfinite JSON literal')))
def declarations(path):
    tree = ast.parse(Path(path).read_text(encoding='utf-8'))
    values = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            try:
                values[node.targets[0].id] = ast.literal_eval(node.value)
            except (ValueError, TypeError):
                pass
    return tree, values

def main():
    checks = []
    def check(value, name):
        checks.append({'id': name, 'passed': bool(value)})
        if not value:
            raise ValueError(name)
    scope = read(CONFIG)
    check(filehash(CONFIG) == FILE_SHA, 'independent exact config file SHA')
    check(scope['sha256'] == BODY_SHA == digest({k: v for k, v in scope.items() if k != 'sha256'}), 'independent exact canonical body SHA')
    profile = scope['short_phase_observation_profile']
    check(digest(profile) == PROFILE_SHA, 'independent exact Phase3 profile SHA')
    check(scope['contract'] == 'synth-short-M0-phase-observed-staged-3' and scope['id'] == 'SYNTH-SHORT-M0-PHASE3-atomic_RS_DEV-04', 'new-only exact contract and batch id')
    check(scope['batch_selector'] == {'source_kind': 'atomic_RS_DEV', 'ordinal': 4}, 'whole original ordinal4 selector')
    rows = scope['rows']
    fields = {'row_id', 'prompt_sha256', 'prompt_ids', 'prompt_tokens', 'attention_mask'}
    check(type(rows) is list and len(rows) == len({r['row_id'] for r in rows}) == 64, 'whole unique64 rows')
    check(all(type(r) is dict and set(r) == fields and type(r['prompt_tokens']) is int
        and 0 < r['prompt_tokens'] <= 480 and len(r['prompt_ids']) == r['prompt_tokens']
        and r['attention_mask'] == [1] * r['prompt_tokens']
        and all(type(t) is int and 0 <= t < 151936 for t in r['prompt_ids']) for r in rows), 'prompt-only exact token/mask rows without oracle fields')
    check(scope['source_packet_sha256'] == digest(rows) == scope['batch']['source_packet_sha256']
        and scope['batch']['rows'] == rows, 'whole packet and batch byte identity')
    capture = sum(32 * r['prompt_tokens'] + sum(range(32)) for r in rows)
    parity = 2 * rows[0]['prompt_tokens']
    expected_dose = {'context_cap': 512, 'maximum_generated_tokens': 32, 'rows': 64,
        'maximum_capture_forwards': 2048, 'maximum_capture_input_positions': capture,
        'maximum_parity_forwards': 2, 'maximum_parity_input_positions': parity,
        'maximum_forward_calls': 2050, 'maximum_forward_input_positions': capture + parity,
        'optimizer_steps': 0, 'backwards': 0}
    check(scope['dose'] == expected_dose and (capture, parity, capture + parity) == (510336, 384, 510720), 'independent complete physical input-position and2050-forward arithmetic')
    old = read(PACKAGE / 'configs/SYNTH_SHORT_M0_ATOMIC_BATCH04_RAM2_A1.json')
    check(rows == old['rows'] and scope['dose'] == old['dose'], 'same complete original batch and dose; no failed/partial outputs imported')
    for key in ('resources', 'tolerances', 'runtime_versions', 'precision', 'offline_environment', 'runtime_pins', 'decoder_runtime_pins', 'data_pins', 'model_pins', 'model_dir', 'seed', 'model_role', 'native7', 'inference1_audit_sha256', 'short_RAM_monitor_profile'):
        check(scope[key] == old[key], 'unchanged original RAM2 binding ' + key)
    _, native = declarations(PACKAGE / 'native_contract.py')
    check(scope['resources'] == native['RESOURCE_BOUNDS'] and scope['tolerances'] == native['TOLERANCES'], 'all original live floors and numerics retained as policy')
    counts = {name: len(scope[key]) for name, key in (('source', 'source_pins'), ('runtime', 'runtime_pins'), ('decoder', 'decoder_runtime_pins'), ('data', 'data_pins'), ('model', 'model_pins'))}
    check(counts == {'source': 131, 'runtime': 65, 'decoder': 17, 'data': 5, 'model': 7}, 'exact131/65/17/5/7 inventory counts')
    needed = set(PACKAGE.glob('*.py')) | set((PACKAGE / 'tests').glob('*.py'))
    needed |= {PACKAGE / n for n in ('PILOT_DECISIONS.json', 'pilot_trainable_inventory.json', 'NATIVE_QUALIFICATION_PROPOSAL.json', 'AGENTS.md')}
    needed |= {ROOT / n for n in ('AGENTS.md', 'RINEYA_SYNTH_CENTRALIZED_REASONING.md')}
    source_before = {str(p.resolve()): filehash(p) for p in needed}
    check(scope['source_pins'] == source_before, 'complete exact current executing source roster and131 current bytes')
    for path, pin in scope['data_pins'].items():
        check(filehash(path) == pin, 'actual current declared data bytes ' + Path(path).name)
    plan_path = PACKAGE / 'evidence/short_collection_plan_cpu_20261004/short_M0_collection_plan_cpu.json'
    plan = read(plan_path)
    check(scope['cpu_plan_sha256'] == plan['sha256'] == digest({k: v for k, v in plan.items() if k != 'sha256'}), 'frozen CPU collection plan body')
    check(scope['batch'] == plan['batches']['atomic_RS_DEV'][3] and scope['batch']['start'] == 192 and scope['batch']['stop'] == 256, 'exact frozen192:256 whole-row slice')
    inputs = read(PACKAGE / plan['source_pins']['atomic_RS_DEV'][0])
    check(len(inputs['rows']) == 384 and rows == inputs['rows'][192:256], 'current original384 public prompt rows; no row reduction')
    prior = read(PACKAGE / 'configs/SYNTH_NQ001_PROC7.json')
    extension = scope['short_RAM_monitor_profile']['runtime_extension_pins']
    core = {p: h for p, h in scope['runtime_pins'].items() if p not in extension}
    check(len(extension) == 1 and all(Path(p).name == 'ast.py' for p in extension)
        and all(scope['runtime_pins'].get(p) == h for p, h in extension.items())
        and core == prior['runtime_pins'] and len(core) == 64, 'exact inherited qualified runtime64 plus singleton AST65 metadata')
    phase_tree, phase_decl = declarations(PACKAGE / 'short_phase_contract.py')
    check(profile['helper_pins'] == phase_decl['HELPER_PINS'] and all(filehash(PACKAGE / n) == pin for n, pin in profile['helper_pins'].items()), 'exact11 unchanged phase/math/guard helper bytes')
    for path, pin in profile['evidence_pins'].items():
        check(filehash(path) == pin, 'exact phase observation/size metadata dependency ' + Path(path).name)
    d001 = read(PACKAGE / 'evidence/context_phase_no_model_D001_attempt1_independent_audit_20261005.json')
    check(d001['sha256'] == profile['D001_body_sha256'] == phase_decl['D001_BODY_SHA'] == digest({k: v for k, v in d001.items() if k != 'sha256'})
        and d001['status'] == 'PASS_SCOPED_CURRENT_NO_MODEL_CONTEXT_PHASE_AND_GUARD_OBSERVATION_ONLY'
        and all(d001[k] is False for k in ('model_loaded', 'native_fit_established', 'historical_cause_established')), 'D001 provenance is no-model observation only')
    check(profile['additional_guards'] == profile['additional_CUDA_queries'] == 0 and profile['collector_math_changed'] is False
        and all(profile[k] is False for k in ('historical_cause_established', 'native_fit_established', 'native_launch_authorized')), 'phase-only observation does not change math/queries or establish fit/cause')
    for path, pin in scope['short_RAM_monitor_profile']['CPU_evidence_pins'].items():
        check(filehash(path) == pin, 'exact prior RAM2 CPU dependency ' + Path(path).name)
    loader_tree, loader_decl = declarations(PACKAGE / 'native_loader.py')
    check(filehash(PACKAGE / 'native_loader.py') == '43229ee3ab59719ddc076ce252fdab029a9b5f43c11c66e9ccc84d092f378174', 'actual loader bytes match independently reviewed auditor source-order policy')
    check(scope['model_pins'] == loader_decl['ASSET_PINS'], 'exact seven source-defined model/asset pins; binary bytes not reread')
    constructor = next(n for n in loader_tree.body if isinstance(n, ast.FunctionDef) and n.name == '_source_shapes')
    namespace = {'__builtins__': {'range': range, 'list': list}}
    exec(compile(ast.Module(body=[constructor], type_ignores=[]), '<reviewed-pure-shape-metadata>', 'exec'), namespace)
    shapes = namespace['_source_shapes']()
    sizes_path = PACKAGE.parent / 'post_training_mimo_centralized_v1/receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json'
    sizes = read(sizes_path)['files']
    chunk = 2 * 1024**2
    k = sum(math.ceil(2 * math.prod(shape) / chunk) for shape in shapes.values())
    a = sum(math.ceil(sizes[n]['bytes'] / chunk) + 1 for n in loader_decl['ASSET_PINS'])
    api = 2048 + 322
    guards = 9 + 2 * a + 6 * k + 2 * api + 2 * 2048 + 2
    phases = 3 * guards - 2 + 2 * api + 2048 + 68 + 18 + 4
    budget = profile['prospective_observation_budget']
    check((len(shapes), k, a, api, guards, phases, 2 * phases) == (310, 682, 588, 2370, 14115, 49221, 98442)
        and all(budget[key] == value for key, value in {'tensor_chunks': k, 'asset_hash_iterations_with_EOF': a,
            'maximum_API_calls': api, 'maximum_original_guards': guards, 'maximum_phases': phases, 'maximum_markers': 2 * phases}.items()), 'independent source/size-only prospective phase budget')
    check(budget['conservative_success_marker_bytes'] == 381 and budget['conservative_success_ledger_bytes'] == 381 * 2 * phases == 37506402
        and budget['conservative_success_ledger_bytes'] <= profile['limits']['maximum_ledger_bytes']
        and budget['maximum_markers'] <= profile['limits']['maximum_markers'], 'unchanged bounded observation ledger metadata; no fsync/model fit measurement')
    worker_source = (PACKAGE / 'short_phase_worker.py').read_text(encoding='utf-8')
    check(all(name + '=original.' + name in worker_source for name in ('ForwardDose', 'LocalGeneratedDecoder', 'DurableRawCollection', 'collect_loaded')), 'actual producer imports exact original collector/decoder/math objects')
    actual_auditor = {str((AUDITOR / name).resolve()): filehash(AUDITOR / name) for name in AUDITOR_PINS}
    check(all(actual_auditor[str((AUDITOR / name).resolve())] == pin for name, pin in AUDITOR_PINS.items()), 'all six exact independently reviewed auditor dependencies')
    check(all(scope[k] is False for k in ('correctness_certified', 'retention_certified', 'training_authorized', 'scoring_authorized',
        'language_qualification_established', 'long_context_qualification_established', 'prior_correct_enrollment_performed',
        'protected_bank_access', 'paid_services', 'confirmation_eligible', 'quality_gain_established')), 'all study/score/enrollment/protected/paid authority remains closed')
    after = {str(p.resolve()): filehash(p) for p in needed}
    check(source_before == after and filehash(CONFIG) == FILE_SHA, 'current source and selected config bytes stable throughout review')
    forbidden = [n for n in ('torch', 'transformers', 'tokenizers', 'psutil', 'subprocess') if n in sys.modules]
    check(not forbidden, 'no native/framework/decoder/OS probe module imports')
    receipt = {'kind': 'PHASE3_BATCH04_READ_ONLY_PRELAUNCH_SOURCE_METADATA_REVIEW', 'schema_version': 1,
        'config_path': str(CONFIG), 'config_file_sha256': FILE_SHA, 'config_body_sha256': BODY_SHA,
        'profile_body_sha256': PROFILE_SHA, 'checks': checks, 'counts': counts, 'dose': expected_dose,
        'prompt_token_range': [min(r['prompt_tokens'] for r in rows), max(r['prompt_tokens'] for r in rows)],
        'auditor_dependency_pins_required': actual_auditor, 'qualified_runtime64_sha256_required': digest(core),
        'source_map_sha256': digest(source_before), 'producer_math_changed': False,
        'root_supplied_preconfig_source_commit_hint': '82d4c173', 'actual_Git_HEAD_queried': False,
        'final_config_commit_launch_HEAD_or_owned_auditor_inventory_verified': False,
        'model_runtime_decoder_binary_pins_are_metadata_only_in_this_review': True,
        'native_OS_GPU_model_decoder_encoder_probes_or_execution': False,
        'native_ready': False, 'native_PASS_or_study_result': False,
        'remaining_launch_gates': ['final committed launch HEAD', 'independently frozen owned auditor inventory/file SHA',
            'current production preflight and binary-byte inventory refresh by authorized root', 'fresh unchanged resource admission'],
        'failed_or_partial_old_M0_batches_included': False}
    receipt['sha256'] = digest(receipt)
    target = HERE / ('phase3_batch04_config_review_%d.json' % time.time_ns())
    target.write_text(canonical(receipt) + '\n', encoding='ascii')
    print(json.dumps({'receipt': str(target), 'checks_passed': len(checks), 'counts': counts, 'dose': expected_dose,
        'native_ready': False, 'forbidden_runtime_imports': forbidden}))
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
