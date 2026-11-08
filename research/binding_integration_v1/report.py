"""Create-only integration report; all six candidate legs gate every endpoint.

Legacy definitions and thresholds remain frozen. The previously opened controlled
diagnostic is report-only, with its inadequate source-anchor cohort unchanged.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import json
import math
from pathlib import Path
import sys
import time

WORKTREE = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(WORKTREE))
from research.binding_integration_v1 import materialize, run_probe
from research.binding_mixture_v1 import report as mixed_report
from research.binding_signal_eval_v1 import score as bind_score
from research.binding_diagnostic_v1 import score as diagnostic_score
from research.retention_highdose_lora_resume_v1 import paired, report as control_report
from research.retention_dose_parameterization_v1 import report_battery as prior

ROOT = run_probe.BASE / 'rineya-dataset-retention-records/binding-integration001-20260920'
MIXTURE = ROOT.parent / 'binding-mixture001-20260920'
CONTROL = ROOT.parent / 'highdose-lora001-resume-20260920'
DIAGNOSTIC = ROOT.parent / 'binding-diagnostic001-20260920'
BASELINE = ROOT.parent / 'binding-dev-baseline001-20260920'
EXPERIMENT = 'BINDING-INTEGRATION-001'
LEGS = {**prior.LEGS, 'binding': 'binding', 'diagnostic': 'diagnostic'}
RESERVATIONS = {'dose': (384, 128), 'select': (384, 128), 'retention': (150, 24),
                'long_context': (256, 24), 'binding': (384, 24), 'diagnostic': (896, 24)}
require = run_probe.require


def read(path):
    return run_probe.read(Path(path))


def sha(path):
    return run_probe.sha(Path(path))


def no_models():
    require(not any(name in sys.modules for name in ('torch', 'transformers', 'safetensors', 'tokenizers')),
            'offline reporting must not import model or tokenizer runtimes')


def external_code_snapshot():
    """Pin inherited scientific readers; no historical module is edited."""
    modules = (mixed_report, mixed_report.bc, mixed_report.bc.baseline, mixed_report.materialize, mixed_report.run_probe,
               control_report, control_report.materialize, control_report.run_probe, paired,
               prior, prior.paired, prior.run_probe, bind_score, bind_score.c,
               diagnostic_score, diagnostic_score.c, diagnostic_score.arithmetic, diagnostic_score.binding,
               sys.modules['research.retention_dose_parameterization_v1.build_arm'])
    paths = {Path(module.__file__).resolve() for module in modules}
    directories = {path.parent for path in paths}
    directories.add(WORKTREE / 'research/binding_signal_v2')
    # Broad code-only pins include the original generation core, independent
    # policy decoder, and transitive helpers without opening data or weights.
    for directory in directories:
        paths.update(path.resolve() for path in directory.rglob('*.py'))
    paths.update(path for path in paired._qualification_paths(run_probe.PILOT).values() if path.suffix == '.py')
    return {str(path): sha(path) for path in sorted(paths)}


def verify_prelaunch(root):
    frozen = read(root / 'PRELAUNCH_BINDING.json')
    require(frozen.get('status') == 'PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION', 'prelaunch freeze did not pass')
    required_code = {'report.py', 'run_probe.py', 'materialize.py', 'eval_common.py', 'eval_probe.py'}
    require(required_code <= set(frozen['code_files_sha256']), 'prelaunch omitted report or execution code')
    required_artifacts = {'PLAN.json', 'GRANT.json', 'PREREG.md', 'CRITERIA.json', 'SESSION_ENVELOPE.json',
                          'BUILD_GATE.json', 'DATA_USE_ADMISSION.json', 'SOURCE_RELOCATION.json'}
    require(required_artifacts <= set(frozen['artifact_files_sha256']), 'prelaunch omitted prospective artifacts')
    for base, pins in ((HERE, frozen['code_files_sha256']), (root, frozen['artifact_files_sha256'])):
        for name, expected in pins.items():
            relative = Path(name); target = (base / relative).resolve()
            require(not relative.is_absolute() and '..' not in relative.parts and target.is_relative_to(base.resolve()) and
                    sha(target) == expected, 'prelaunch code/artifact identity changed: ' + name)
    require(frozen.get('external_code_files_sha256') == external_code_snapshot(),
            'inherited statistical/custody helpers differ from the prospective code freeze')
    return sha(root / 'PRELAUNCH_BINDING.json')


def validate_criteria(criteria):
    require(criteria['experiment'] == EXPERIMENT and criteria['route'] == 'NEW_SKILL_ACQUISITION_NOT_RESTORATION',
            'unregistered integration claim route')
    expected_bands = {'dose': [260, 330], 'select': [260, 330], 'references_retained': [37, 45],
        'arithmetic_panel': [55, 85], 'long_context_report_only': [145, 180], 'literal_unknown_report_only': [12, 32],
        'person_missing': [72, 96], 'person_vacant': [80, 96], 'person_answerable': [64, 96],
        'role_answerable': [72, 96], 'false_unknown_on_positive': [0, 10]}
    require(criteria['prediction_bands'] == expected_bands and criteria['primary_engineering'] == {
        'each_acquisition_at_least': 284, 'fixed47_retained_at_least': 40, 'arithmetic150_at_least': 62,
        'all_binding_row_bands_required': True}, 'prior engineering thresholds or prediction bands changed')
    rule = criteria['binding_efficacy']; acquisition = criteria['acquisition_comparison']
    require(rule['comparator'] == 'lora' and rule['primary_categories'] == list(bind_score.UNKNOWN_CATEGORIES) and
            rule['positive_net_required'] is True and rule['holm_family_size'] == 2 and
            rule['adjusted_exact_p_at_most'] == .05 and rule['positive_control_and_false_unknown_bands_required'] is True,
            'binding efficacy no longer uses the registered LoRA/world/control rule')
    require(acquisition['comparators'] == ['mixed', 'lora'] and acquisition['relationship'] == 'cross_training_draw' and
            acquisition['absolute_net_floor'] == 20 and acquisition['exact_p_required'] is True and
            acquisition['cost_engineering_net_minimum'] == -20 and acquisition['cost_bound_does_not_establish_preservation'] is True,
            'cross-draw acquisition floor or cost rule changed')
    require(criteria['controlled_diagnostic_role'] == 'SPENT_REPORT_ONLY_NOT_SELECTION_OR_CONFIRMATION' and
            criteria['long_context_role'] == 'SPENT_REPORT_ONLY_NOT_SELECTION_OR_CONFIRMATION' and
            criteria['quality_claim'] is False and criteria['release_permitted'] is False,
            'report-only evidence cannot be converted into admission or confirmation')
    require(criteria['controlled_diagnostic_prediction_bands'] == {'all16_binding_parents': [0, 8],
            'false_unknown_on_positive256': [16, 64], 'strict_arithmetic_anchors96': [0, 12]},
            'controlled diagnostic prospective report-only bands changed')


def verify_admission(root, gate):
    path = Path(gate['data_use_admission_path']).resolve()
    require(path == root / 'DATA_USE_ADMISSION.json' and sha(path) == gate['data_use_admission_sha256'],
            'new fourth-model data-use admission identity differs')
    record = read(path)
    require(record.get('record') == 'BINDING_INTEGRATION_SCOPED_DATA_USE_ADMISSION' and record.get('experiment') == EXPERIMENT and
            record.get('status') == 'ADMITTED_FOR_NAMED_TRAIN_DATA_AND_EXPLORATORY_EVALUATION_SCOPE' and
            record.get('independent_confirmation') is False and record.get('release_permitted') is False,
            'new evaluation scope is not admitted')
    controlled = record['evaluation_scope']['controlled_diagnostic896']
    require(controlled['status'] == 'OPENED_DIAGNOSTIC_REUSE_REPORT_ONLY' and controlled['rows_per_model'] == 896 and
            controlled['fourth_model_use_is_new_scope_not_old_three_model_permission'] is True and
            controlled['selection_or_promotion_permission'] is False and controlled['independent_confirmation'] is False,
            'opened diagnostic scope was silently generalized')
    pins = {str(path): sha(path)}
    def walk(value):
        if isinstance(value, dict):
            if isinstance(value.get('path'), str) and isinstance(value.get('sha256'), str):
                target = Path(value['path']).resolve()
                require(sha(target) == value['sha256'], 'admitted data/control evidence changed')
                pins[str(target)] = value['sha256']
            for child in value.values():
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
    walk(record)
    code_pins = record['evaluation_scope']['frozen_reference_and_reporter_code_sha256']
    require(isinstance(code_pins, dict) and
            {Path(module.__file__).resolve() for module in (paired, bind_score, diagnostic_score)} <=
            {Path(name).resolve() for name in code_pins}, 'admission omitted frozen statistical/scoring helpers')
    for name, expected in code_pins.items():
        target = Path(name).resolve()
        require(sha(target) == expected, 'admitted frozen reference/scoring helper changed')
        pins[str(target)] = expected
    return record, pins


def probe_cost_accounting(root, envelope):
    """Recount the normal six-success battery; recovery needs its own admission."""
    root = Path(root).resolve()
    require(not (root / 'ATTEMPT_MAP.json').exists() and not (root / 'posttrain/RECOVERY_COMPLETE.json').exists(),
            'this reporter does not accept a replacement or unqualified recovery')
    required = {'integrated-' + leg for leg in LEGS}
    ignored = {'PLAN.json', 'TERMINAL.json'}
    paths = [path for path in sorted((root / 'posttrain').glob('*.json')) if path.name not in ignored]
    require({path.stem for path in paths} == required, 'six-probe attempt receipt inventory differs')
    attempts, pins = [], {}
    for path in paths:
        record = read(path); key = path.stem; leg = key.removeprefix('integrated-')
        out = root / 'eval' / key
        require(record['leg'] == key and Path(record['out']).resolve() == out and
                type(record['returncode']) is int and record['returncode'] == 0,
                'candidate attempt is unsuccessful or points to a different output')
        n, maximum = RESERVATIONS[leg]
        require(record['reserved_rows'] == n and record['reserved_generation_tokens'] == n * maximum,
                'per-leg inference reservation changed')
        supervisor_path = out / ('supervisor.json' if leg in ('binding', 'diagnostic') else 'worker/supervisor.json')
        supervisor = read(supervisor_path); seconds = supervisor['wall_seconds']
        require(supervisor['status'] == 'EXITED' and type(supervisor['returncode']) is int and supervisor['returncode'] == 0 and
                type(seconds) in (int, float) and math.isfinite(seconds) and 0 <= seconds <= 900 and
                type(record['worker_seconds_charged']) in (int, float) and
                math.isclose(record['worker_seconds_charged'], seconds, rel_tol=0, abs_tol=1e-6),
                'observed worker time/return code differs from the attempt charge')
        attempts.append({'leg': key, 'out': str(out), 'returncode': 0, 'worker_seconds_charged': float(seconds),
                         'reserved_rows': n, 'reserved_generation_tokens': n * maximum})
        pins[path.relative_to(root).as_posix()] = sha(path)
        pins[supervisor_path.relative_to(root).as_posix()] = sha(supervisor_path)
    totals = {'probe_worker_seconds': sum(item['worker_seconds_charged'] for item in attempts),
              'probe_rows_reserved': sum(item['reserved_rows'] for item in attempts),
              'probe_generation_tokens_reserved': sum(item['reserved_generation_tokens'] for item in attempts)}
    for key, ceiling in (('probe_worker_seconds', 'probe_worker_seconds_all_attempts'),
                         ('probe_rows_reserved', 'probe_rows_reserved_all_attempts'),
                         ('probe_generation_tokens_reserved', 'probe_generation_tokens_reserved_all_attempts')):
        require(totals[key] <= envelope[ceiling], 'global probe envelope exceeded')
    require(len(attempts) <= envelope['probe_workers_all_attempts'] and envelope['probe_retries_total'] == 1,
            'registered global worker/retry cap changed')
    return {**totals, 'attempt_count': 6, 'successful_attempts': 6, 'failed_attempts': 0,
            'individual_attempts': attempts, 'cost_source_sha256': pins}


def completion_receipt(root, envelope):
    path = root / 'posttrain/TERMINAL.json'; terminal = read(path)
    require(terminal.get('status') == 'COMPLETE_SIX_PROBES', 'six-probe battery is incomplete')
    costs = probe_cost_accounting(root, envelope)
    actual_attempts = [read(root / 'posttrain' / (item['leg'] + '.json')) for item in costs['individual_attempts']]
    require(isinstance(terminal.get('attempts'), list) and len(terminal['attempts']) == 6 and
            {item['leg']: item for item in terminal['attempts']} == {item['leg']: item for item in actual_attempts},
            'battery terminal does not bind exactly the six observed attempt receipts')
    bindings = {}
    for leg in LEGS:
        out = root / 'eval' / ('integrated-' + leg)
        filename = 'AUX_BINDING.json' if leg in ('binding', 'diagnostic') else 'PROBE_WEIGHT_BINDING.json'
        value = read(out / filename)
        require(value.get('status') == 'COMPLETE' and value.get('complete') is True, 'candidate leg is incomplete')
        bindings['integrated-' + leg] = sha(out / filename)
    require(terminal.get('final_binding_sha256') == bindings and terminal.get('cost_accounting') == costs,
            'battery final bindings or all-attempt cost accounting differ')
    return {'path': str(path), 'sha256': sha(path), 'final_binding_sha256': bindings, 'cost_accounting': costs}


class ArchiveChain:
    """Compose only qualified A/B/C code, cwd and three auxiliary-input aliases."""
    def __init__(self, old, new):
        require(old.archive == new.original and new.archive == run_probe.PILOT.resolve(), 'source relocation chain is disconnected')
        self.old, self.new = old, new

    def normalize_additional_inputs(self, values):
        return self.old.normalize_additional_inputs(self.new.normalize_additional_inputs(values))

    def canonical_cwd(self, value, python):
        path = Path(value).resolve()
        if path == self.new.archive:
            path = Path(self.new.canonical_cwd(path, python))
        return self.old.canonical_cwd(path, python)

    def revalidate(self):
        self.old.revalidate(); self.new.revalidate()


def verify_generation_tokens(generations, inputs, decoder, eos):
    """Independent text/EOS/accounting check for both fixed auxiliary sizes."""
    require(generations['rows'] == len(inputs) and generations['eos_ids'] == eos and
            generations['numerics'] == mixed_report.bc.baseline.NUMERICS and generations['gold_file_read'] is False,
            'auxiliary numerical/input/EOS scope differs')
    predictions = generations['predictions']
    require(len(predictions) == len(inputs) and [row['id'] for row in predictions] == [row['id'] for row in inputs],
            'auxiliary generation IDs or order differ from the full admitted bank')
    for original, row in zip(inputs, predictions):
        tokens = row['generated_ids_including_eos']
        require(row['parent_id'] == original['parent_id'] and type(row['terminated']) is bool and
                isinstance(row['prediction'], str) and type(row['input_tokens']) is int and row['input_tokens'] == len(original['prompt_ids']),
                'auxiliary input tokens or parent identity differ')
        require(isinstance(tokens, list) and 1 <= len(tokens) <= 24 and all(type(t) is int and 0 <= t < 151936 for t in tokens) and
                all(type(row[key]) is int for key in ('generated_tokens', 'forward_calls', 'processed_input_tokens')) and
                row['generated_tokens'] == row['forward_calls'] == len(tokens) and
                row['processed_input_tokens'] == len(original['prompt_ids']) + len(tokens) - 1 and
                row['terminated'] == (tokens[-1] in eos) and not any(t in eos for t in tokens[:-1]) and
                (row['terminated'] or len(tokens) == 24) and decoder.decode(tokens) == row['prediction'],
                'auxiliary generation text, EOS, or actual token charges differ')
    actual = {'rows': len(inputs), 'generated_tokens_including_eos': sum(row['generated_tokens'] for row in predictions),
              'forward_calls': sum(row['forward_calls'] for row in predictions),
              'processed_input_tokens': sum(row['processed_input_tokens'] for row in predictions)}
    require(generations['actual_budget_charges'] == actual, 'auxiliary actual charge totals do not reproduce generations')
    return actual


def acquisition_maps(loaded):
    """Recompute frozen canonical JSON correctness, preserving its EOS policy."""
    output = {}
    canonical = lambda value: json.dumps(value, sort_keys=True, separators=(',', ':'))
    for role, legs in loaded.items():
        output[role] = {}
        for bank in ('dose', 'select'):
            raw, binding, ids = legs[bank]
            gold = prior.jsonl(binding['views_path']); predictions = prior.unique_rows(raw['predictions'], role + '/' + bank)
            require(gold.keys() == predictions.keys() == ids and len(ids) == 384, 'acquisition coverage changed')
            values = {}
            for key, row in predictions.items():
                require(isinstance(row['prediction'], str) and row['reference'] == gold[key]['reference'],
                        'acquisition stored prediction/reference differs from its frozen text contract')
                try:
                    correct = canonical(json.loads(row['prediction'])) == gold[key]['reference']
                except (ValueError, TypeError):
                    correct = False
                require(type(row['canonical']) is bool and row['canonical'] == correct, 'canonical acquisition score does not reproduce raw output')
                values[key] = correct
            output[role][bank] = values
    return output


def acquisition_contrasts(maps, criteria):
    result = {}
    floor = criteria['acquisition_comparison']['absolute_net_floor']
    for comparator in ('mixed', 'lora'):
        family = {bank: paired.paired_boolean_summary(maps[comparator][bank], maps['integrated'][bank], relationship='cross_training_draw')
                  for bank in ('dose', 'select')}
        control_report.holm(family)
        for item in family.values():
            item.update(cross_training_draw_abs_net_floor=floor, cross_training_draw_floor_cleared=abs(item['net']) >= floor,
                        structural_claim_eligible=abs(item['net']) >= floor and item['holm_adjusted_p'] <= .05,
                        direction='integrated_higher' if item['net'] > 0 else 'integrated_lower' if item['net'] < 0 else 'no_net_change')
        result[comparator] = family
    return result


def score_binding(inputs, gold, bundles, frozen_baseline, frozen_mixed, decode):
    endpoints, maps, roster = {}, {}, {}
    for role, bundle in bundles.items():
        endpoints[role], maps[role], roster[role] = bind_score.score_rows(inputs, gold, bundle['predictions'], decode)
    for role in ('source', 'lora'):
        require(endpoints[role] == frozen_baseline['endpoints'][role] and maps[role] == frozen_baseline['world_maps'][role] and
                roster[role] == frozen_baseline['roster_person_world_any_maps'][role], 'original BIND baseline recount differs')
    require(endpoints['mixed'] == frozen_mixed['binding']['endpoints']['mixed'] and
            maps['mixed'] == frozen_mixed['binding']['world_maps']['mixed'], 'old mixture BIND recount differs')
    contrasts = {}
    for comparator in ('mixed', 'lora'):
        contrasts[comparator] = {policy: {name: paired.paired_boolean_summary(maps[comparator][policy][name], values,
            relationship='cross_training_draw') for name, values in groups.items()} for policy, groups in maps['integrated'].items()}
        if comparator == 'lora':
            for policy in contrasts[comparator]:
                control_report.holm({category: contrasts[comparator][policy][category] for category in bind_score.UNKNOWN_CATEGORIES})
    return {'endpoints': endpoints, 'world_maps': maps, 'roster_person_world_any_maps': roster,
            'contrasts_integrated_minus': contrasts,
            'roster_world_contrasts_integrated_minus': {role: paired.paired_boolean_summary(roster[role], roster['integrated'],
                relationship='cross_training_draw') for role in ('mixed', 'lora')},
            'paired_unit': '32 independent parent worlds; all three category views must be correct',
            'primary_comparator': 'lora', 'mixed_comparator_role': 'nominal descriptive',
            'normalization': frozen_baseline['normalization']}


def prediction_verdicts(candidate, binding, acquisition, criteria):
    retention = candidate['retention']['policies']['permissive']; fresh = binding['endpoints']['integrated']
    observed = {'dose': candidate['dose']['correct'], 'select': candidate['select']['correct'],
                'references_retained': retention['references_retained'], 'arithmetic_panel': retention['panel_correct'],
                'long_context_report_only': candidate['long_context']['correct'],
                'literal_unknown_report_only': candidate['long_context']['literal_unknown'],
                'false_unknown_on_positive': fresh['totals']['false_unknown_on_positive'],
                **{name: fresh['categories'][name]['canonical_complete_exact'] for name in bind_score.CATEGORIES}}
    bands = {name: prior.band(observed[name], bounds) for name, bounds in criteria['prediction_bands'].items()}
    rule = criteria['primary_engineering']
    engineering = (all(observed[bank] >= rule['each_acquisition_at_least'] for bank in ('dose', 'select')) and
                   observed['references_retained'] >= rule['fixed47_retained_at_least'] and
                   observed['arithmetic_panel'] >= rule['arithmetic150_at_least'] and
                   all(bands[name]['status'] == 'PASS' for name in (*bind_score.CATEGORIES, 'false_unknown_on_positive')))
    controls = all(bands[name]['status'] == 'PASS' for name in ('person_answerable', 'role_answerable', 'false_unknown_on_positive'))
    efficacy = binding['contrasts_integrated_minus']['lora']['canonical_complete_exact']
    established = controls and any(efficacy[name]['net'] > 0 and efficacy[name]['holm_adjusted_p'] <=
        criteria['binding_efficacy']['adjusted_exact_p_at_most'] for name in bind_score.UNKNOWN_CATEGORIES)
    cost = {role: {bank: {'net': item['net'], 'minimum': criteria['acquisition_comparison']['cost_engineering_net_minimum'],
                         'status': 'PASS' if item['net'] >= criteria['acquisition_comparison']['cost_engineering_net_minimum'] else 'FAILED_PREDICTION',
                         'establishes_preservation': False} for bank, item in family.items()} for role, family in acquisition.items()}
    return {'bands': bands, 'primary_engineering_conjunction': 'PASS' if engineering else 'FAILED_PREDICTION',
            'binding_efficacy': 'ESTABLISHED_EXPLORATORY_DEV_GAIN' if established else 'NOT_ESTABLISHED',
            'binding_positive_controls_pass': controls, 'acquisition_cost_engineering': cost,
            'long_context_used_for_selection_or_confirmation': False,
            'controlled_diagnostic_used_for_selection_or_confirmation': False}


def score_controlled(inputs, gold, bundles, frozen):
    typed = diagnostic_score.load_typed_scorer()
    endpoints, maps = {}, {}
    for role, bundle in bundles.items():
        endpoints[role], maps[role] = diagnostic_score.score_rows(inputs, gold, bundle['predictions'], typed)
        if role != 'integrated':
            require(endpoints[role] == frozen['endpoints'][role] and maps[role] == frozen['parent_maps'][role],
                    'opened controlled-diagnostic baseline recount differs')
    cohort = sorted(parent for parent, value in maps['source']['arithmetic']['strict_anchor'].items() if value)
    previous = frozen['source_conditioned_retention_descriptive']
    require(cohort == previous['parent_ids'] and len(cohort) == previous['n'] == 4,
            'opened source-anchor cohort changed; no backfilling or new eligibility is permitted')
    anchors = {value['parent_id']: value['id'] for value in gold.values() if value['section'] == 'arithmetic' and
               value['numeric_variant'] == 0 and value['response_type'] == 'unit_explicit'}
    retention = {'n': 4, 'parent_ids': cohort, 'anchor_ids': [anchors[parent] for parent in cohort],
                 'status': 'INSUFFICIENT_REFERENCE_COHORT', 'minimum_reference_n': 24,
                 'source_only_eligibility': True, 'inferential_tests_performed': False, 'models': {}}
    for role in bundles:
        lost = [parent for parent in cohort if not maps[role]['arithmetic']['strict_anchor'][parent]]
        retention['models'][role] = {'retained': len(cohort) - len(lost), 'lost_parent_ids': lost,
                                     'lost_anchor_ids': [anchors[parent] for parent in lost]}
    contrasts = {role: {section: {name: paired.paired_boolean_summary(maps[role][section][name], values,
                    relationship='cross_training_draw') for name, values in groups.items()}
                    for section, groups in maps['integrated'].items()} for role in ('mixed', 'lora')}
    return {'role': 'SPENT_REPORT_ONLY_NOT_SELECTION_OR_CONFIRMATION', 'endpoints': endpoints, 'parent_maps': maps,
            'nominal_descriptive_contrasts_integrated_minus': contrasts,
            'source_conditioned_retention_descriptive': retention, 'primary_endpoint_substitution': False,
            'independent_confirmation': False, 'acquisition_floor_applied_to_32_or_96_parents': False}


def controlled_predictions(controlled, criteria):
    endpoint = controlled['endpoints']['integrated']
    actual = {'all16_binding_parents': endpoint['parent_counts']['binding']['canonical_complete_exact/all16'],
              'false_unknown_on_positive256': endpoint['totals']['binding']['false_exact_unknown'],
              'strict_arithmetic_anchors96': endpoint['parent_counts']['arithmetic']['strict_anchor']}
    return {'role': 'REPORT_ONLY_NOT_ACCEPTANCE', 'bands': {name: prior.band(actual[name], bounds)
            for name, bounds in criteria['controlled_diagnostic_prediction_bands'].items()},
            'primary_endpoint_substitution': False}


def remember_binding_files(binding, pins):
    for key, value in binding.items():
        if key.endswith('_path') and isinstance(value, str):
            digest_key = key[:-5] + '_sha256'
            if isinstance(binding.get(digest_key), str):
                path = Path(value).resolve()
                require(sha(path) == binding[digest_key], 'bound evidence changed before analysis')
                pins[str(path)] = binding[digest_key]
    for name, expected in binding['additional_input_sha256_before'].items():
        pins[str(Path(name).resolve())] = expected


def load_standard(root, gate, frozen, archives, pins):
    loaded, custody, snapshots, receipts = {}, {}, {}, {}
    implementation = run_probe.implementation_digest(); context = None
    configurations = {
        'integrated': (root, gate, root / 'BUILD_GATE.json', run_probe.arm_snapshot, archives.new),
        'mixed': (MIXTURE, read(MIXTURE / 'BUILD_GATE.json'), MIXTURE / 'BUILD_GATE.json', mixed_report.run_probe.arm_snapshot, archives.old),
        'lora': (CONTROL, read(CONTROL / 'BUILD_GATE_LORA_FINAL.json'), CONTROL / 'BUILD_GATE_LORA_FINAL.json', control_report.run_probe.arm_snapshot, archives.old)}
    for role, (base, own_gate, own_gate_path, snapshot_fn, archive) in configurations.items():
        loaded[role], custody[role] = {}, {}
        for leg, kind in prior.LEGS.items():
            directory = root / 'eval' / ('integrated-' + leg) if role == 'integrated' else Path(frozen[role]['custody'][role][leg]['directory']).resolve()
            require(directory.is_relative_to(base / 'eval'), 'authoritative standard attempt escapes its record')
            raw, binding, ids = paired.validate_probe_binding(directory / 'result.json', directory / 'PROBE_WEIGHT_BINDING.json', kind, source_archive=archive)
            precheck = read(directory / 'PRECHECK.json'); command = precheck['command']
            require(precheck['status'] == 'PRECHECK_PASSED' and precheck['kind'] == kind and isinstance(command, list) and len(command) >= 2,
                    'standard probe precheck is missing')
            observed = (str(Path(command[0]).resolve()), archives.canonical_cwd(precheck['cwd'], command[0]))
            context = observed if context is None else context
            require(observed == context and binding['implementation_digest'] == implementation, 'standard scientific runtime/scorer context differs')
            require(Path(binding['build_gate_path']).resolve() == own_gate_path and binding['build_gate_sha256'] == sha(own_gate_path) and
                    binding['script_sha256'] == own_gate['probe_script_sha256'][kind] and binding['wrapper_sha256'] == own_gate['wrapper_sha256'],
                    'standard probe did not use its registered gate/code')
            if leg in ('dose', 'select'):
                require(raw['instrument'] == ('process-transition-probe-001' if leg == 'dose' else 'select-views-001') and
                        raw['max_new_tokens'] == 128 and binding['views_sha256'] == own_gate['banks'][leg]['sha256'],
                        'acquisition bank or decoding limit changed')
            declared = {'directory': str(directory), 'binding_sha256': sha(directory / 'PROBE_WEIGHT_BINDING.json'),
                        'result_sha256': sha(directory / 'result.json'), 'weight_sha256': binding['weight_sha256']}
            if role != 'integrated':
                for field in ('binding_sha256', 'result_sha256', 'weight_sha256'):
                    require(declared[field] == frozen[role]['custody'][role][leg][field], 'comparator is not its authoritative successful attempt')
            loaded[role][leg] = raw, binding, ids; custody[role][leg] = declared
            pins[str(directory / 'PROBE_WEIGHT_BINDING.json')] = declared['binding_sha256']
            pins[str(directory / 'PRECHECK.json')] = sha(directory / 'PRECHECK.json')
            remember_binding_files(binding, pins)
        first = loaded[role]['dose'][1]
        arm = base / 'arms' / role
        require(Path(first['arm_dir']).resolve() == arm, 'standard probe points to another arm')
        live = snapshot_fn(arm, own_gate); snapshots[role] = live['training']; receipts[role] = read(arm / 'ARM.json')
        for _, current, _ in loaded[role].values():
            require(current['weight_sha256'] == live['weight_sha256'] and current['training_identity'] == first['training_identity'] and
                    current['training_snapshot_before'] == current['training_snapshot_after'] == live['training'],
                    'within-arm model or training identity differs across standard legs')
        require(receipts[role]['stored_dtype'] == 'float32' and receipts[role]['checkpoint_binding']['checkout'] == own_gate['pilot_checkout'],
                'arm precision or qualified training checkout differs')
    require(receipts['integrated']['source_assets_sha256'] == receipts['mixed']['source_assets_sha256'] == receipts['lora']['source_assets_sha256'],
            'registered models do not share the original source assets')
    require(snapshots['integrated']['updates'] == snapshots['lora']['updates'] == 193 and snapshots['mixed']['updates'] == 296 and
            snapshots['integrated']['native_clock_tokens'] == 380921 and snapshots['integrated']['processed_tokens'] == 577417,
            'integration/native/old-mixture trajectory structures were collapsed')
    for role in ('mixed', 'lora'):
        mixed_report.check_standard_pair(loaded['integrated'], loaded[role], archives)
    return loaded, custody, snapshots, receipts


def load_original_binding(gate, frozen, candidate, pins):
    old_gate = read(MIXTURE / 'BUILD_GATE.json')
    for key in ('binding_baseline_scores_sha256', 'binding_score_plan_sha256'):
        require(gate[key] == old_gate[key], 'original BIND instrument registration changed')
    path = mixed_report.resolve_attempt(MIXTURE, 'mixed-binding', mixed_report.attempt_map(MIXTURE))
    control_weight = frozen['lora']['custody']['lora']['dose']['weight_sha256']
    inputs, gold, bundles, custody, baseline, decode = mixed_report.binding_inputs(MIXTURE, old_gate, path, control_weight, CONTROL)
    require(custody == frozen['mixed']['binding_custody'], 'original BIND control custody differs from the frozen mixture report')
    bundles['integrated'] = candidate
    require(candidate['before']['input_sha256'] == baseline['input_sha256'], 'new BIND candidate uses another input bank')
    for role in ('source', 'lora', 'mixed'):
        directory = BASELINE / role if role != 'mixed' else path
        pins[str(directory / 'GENERATIONS.json')] = sha(directory / 'GENERATIONS.json')
    return inputs, gold, bundles, custody, baseline, decode


def load_controlled(candidate, frozen, decoder, eos):
    diagnostic_score.verify_frozen(DIAGNOSTIC)
    plan_path, grant_path = DIAGNOSTIC / 'PLAN.json', DIAGNOSTIC / 'GRANT.json'
    plan = diagnostic_score.c.validate_plan(read(plan_path))
    diagnostic_score.c.validate_grant(plan_path, read(grant_path))
    inputs = diagnostic_score.c.validate_inputs(diagnostic_score.c.read_jsonl(plan['input_path']))
    gold = diagnostic_score.validate_data(inputs, diagnostic_score.c.read_jsonl(DIAGNOSTIC / 'candidate-v1/GOLD.jsonl'),
                                           read(DIAGNOSTIC / 'candidate-v1/METADATA.json'))
    bundles, custody = {}, {}
    for role in ('source', 'lora', 'mixed'):
        bundles[role], custody[role] = diagnostic_score.verify_worker(plan, plan_path, grant_path, role, inputs, decoder, eos)
        require(custody[role] == frozen['custody'][role], 'opened diagnostic baseline successful worker changed')
    require(candidate['before']['input_sha256'] == plan['input_sha256'], 'new controlled diagnostic uses another bank')
    bundles['integrated'] = candidate
    return inputs, gold, bundles, custody


def require_six_legs(standard, auxiliary):
    require(set(standard) == set(prior.LEGS) and set(auxiliary) == {'binding', 'diagnostic'},
            'all six candidate comparability legs must pass before any endpoint')


def build_report(root=ROOT):
    root = Path(root).resolve()
    try:
        no_models()
        gate_path = root / 'BUILD_GATE.json'; gate = read(gate_path)
        require(str(gate.get('status', '')).startswith('PASS'), 'integration build gate is not PASS')
        prelaunch = verify_prelaunch(root)
        for name, field in (('PREREG.md', 'registration_sha256'), ('CRITERIA.json', 'criteria_sha256'),
                            ('SESSION_ENVELOPE.json', 'session_envelope_sha256')):
            require(sha(root / name) == gate[field], 'prospective registration changed')
        criteria, envelope = read(root / 'CRITERIA.json'), read(root / 'SESSION_ENVELOPE.json')
        validate_criteria(criteria)
        require(time.time() <= envelope['deadline_unix'], 'report exceeds the registered phase deadline')
        admission, pins = verify_admission(root, gate)
        frozen = {role: read(base / 'RESULTS.json') for role, base in (('mixed', MIXTURE), ('lora', CONTROL), ('diagnostic', DIAGNOSTIC))}
        for role, base in (('mixed', MIXTURE), ('lora', CONTROL), ('diagnostic', DIAGNOSTIC)):
            require(frozen[role]['status'] == 'COMPLETE_COMPARABLE' and sha(base / 'RESULTS.json') == gate['frozen_results_sha256'][role],
                    'prospectively frozen comparator results changed')
            pins[str(base / 'RESULTS.json')] = gate['frozen_results_sha256'][role]
        require(sha(mixed_report.__file__) == frozen['mixed']['reporter_sha256'] and
                sha(control_report.__file__) == frozen['lora']['reporter_sha256'] and
                sha(diagnostic_score.__file__) == frozen['diagnostic']['scorer_sha256'], 'frozen scoring helper changed')
        old_archive = paired.SourceArchive.load(MIXTURE / 'SOURCE_RELOCATION.json', frozen['mixed']['source_relocation_sha256'])
        new_archive = paired.SourceArchive.load(root / 'SOURCE_RELOCATION.json', gate['source_relocation_sha256'])
        archives = ArchiveChain(old_archive, new_archive)
        completion = completion_receipt(root, envelope)
        pins[completion['path']] = completion['sha256']
        pins.update({str(root / name): value for name, value in completion['cost_accounting']['cost_source_sha256'].items()})
        loaded, custody, snapshots, receipts = load_standard(root, gate, frozen, archives, pins)
        # The new generic worker owns its exact grant/launch schema; its frozen
        # model-free verifier performs no scoring or endpoint extraction.
        sys.path.insert(0, str(HERE))
        auxiliary_module = importlib.import_module('research.binding_integration_v1.eval_common')
        auxiliary, auxiliary_custody = {}, {}
        for kind in ('binding', 'diagnostic'):
            auxiliary[kind], auxiliary_custody[kind] = auxiliary_module.verify_completed(root, kind)
            pins.update(auxiliary_custody[kind]['files_sha256'])
            value = auxiliary[kind]
            require(value['record'] == 'BINDING_INTEGRATION_AUX_GENERATIONS' and value['kind'] == kind and
                    value['before']['weight_sha256'] == custody['integrated']['dose']['weight_sha256'] and
                    value['before']['sidecars_sha256'] == {name: receipts['integrated']['source_assets_sha256'][name] for name in run_probe.SIDECARS},
                    'auxiliary leg uses a different model, sidecar inventory or kind')
            directory = root / 'eval' / ('integrated-' + kind)
            for name in ('AUX_BINDING.json', 'GENERATIONS.json', 'worker_terminal.json', 'supervisor.json',
                         'launch.json', 'spawn.json', 'worker_claim.json'):
                pins[str(directory / name)] = sha(directory / name)
        require_six_legs(loaded['integrated'], auxiliary)
        source_dir = Path(receipts['integrated']['source'])
        tokenizer = diagnostic_score.c.read(source_dir / 'tokenizer.json'); tokenizer_config = diagnostic_score.c.read(source_dir / 'tokenizer_config.json')
        decoder = diagnostic_score.ByteLevelDecoder(tokenizer, tokenizer_config)
        eos = diagnostic_score.expected_eos(source_dir, tokenizer, tokenizer_config)
        bind_data = load_original_binding(gate, frozen, auxiliary['binding'], pins)
        diag_data = load_controlled(auxiliary['diagnostic'], frozen['diagnostic'], decoder, eos)
        for kind, inputs, bundles in (('binding', bind_data[0], bind_data[2]), ('diagnostic', diag_data[0], diag_data[2])):
            for bundle in bundles.values():
                verify_generation_tokens(bundle, inputs, decoder, eos)
        arm = receipts['integrated']
        checkpoint = Path(arm['checkpoint'])
        manifest = materialize.unseal(read(checkpoint / 'manifest.json'))
        qualification = materialize.training_qualification({'checkpoint': str(checkpoint), 'manifest': manifest,
            'checkpoint_manifest_sha256': sha(checkpoint / 'manifest.json')}, Path(arm['training_result_path']))
        for name in ('ARM.json', 'MATERIALIZATION.json'):
            receipt = read(root / 'arms/integrated' / name)
            require(all(receipt.get(key) == value for key, value in qualification.items()), 'export/training/zero-addend qualification custody differs')
        archives.revalidate()
        # Every candidate/control leg, grant, data scope and model identity has
        # now passed. Scientific endpoint computation begins only here.
        endpoints, maps = control_report.extract(loaded, control_report.load_pure_retention())
        for role in ('mixed', 'lora'):
            require(endpoints[role] == frozen[role]['endpoints'][role], 'legacy comparator recount differs from its frozen report')
        acquisitions = acquisition_maps(loaded)
        acquisition = acquisition_contrasts(acquisitions, criteria)
        binding = score_binding(bind_data[0], bind_data[1], bind_data[2], bind_data[4], frozen['mixed'], bind_data[5])
        controlled = score_controlled(diag_data[0], diag_data[1], diag_data[2], frozen['diagnostic'])
        controlled['prospective_predictions'] = controlled_predictions(controlled, criteria)
        secondary = {role: {name: paired.paired_boolean_summary(maps[role][name], values, relationship='cross_training_draw')
                           for name, values in maps['integrated'].items()} for role in ('mixed', 'lora')}
        verdicts = prediction_verdicts(endpoints['integrated'], binding, acquisition, criteria)
        require(verify_prelaunch(root) == prelaunch and all(sha(path) == expected for path, expected in pins.items()),
                'bound data/control/candidate evidence changed during reporting')
        archives.revalidate(); no_models()
        require(time.time() <= envelope['deadline_unix'], 'completed report exceeds registered phase deadline')
        return {'schema': 1, 'record': 'BINDING_INTEGRATION_RESULTS', 'experiment': EXPERIMENT,
            'status': 'COMPLETE_COMPARABLE', 'created_utc': datetime.now(timezone.utc).isoformat(), 'root': str(root),
            'build_gate_sha256': sha(gate_path), 'criteria_sha256': sha(root / 'CRITERIA.json'), 'reporter_sha256': sha(Path(__file__)),
            'prelaunch_binding_sha256': prelaunch, 'data_use_admission_sha256': gate['data_use_admission_sha256'],
            'frozen_results_sha256': gate['frozen_results_sha256'], 'verified_completion': completion,
            'source_relocation_chain_sha256': [old_archive.sha256, new_archive.sha256], 'custody': custody,
            'auxiliary_custody': auxiliary_custody, 'binding_baseline_custody': bind_data[3], 'controlled_baseline_custody': diag_data[3],
            'training_snapshots': snapshots, 'training_qualification': qualification, 'endpoints': endpoints,
            'legacy_maps': maps, 'acquisition_maps': acquisitions, 'acquisition_contrasts_integrated_minus': acquisition,
            'secondary_contrasts_integrated_minus': secondary, 'binding': binding, 'controlled_diagnostic': controlled,
            'predictions': verdicts, 'model_calls': 0, 'training_permission': False, 'quality_claim': False,
            'release_permitted': False, 'promotion_permitted': False, 'independent_confirmation': False,
            'multiplicity': {'acquisition': 'Separate Holm(2) family for dose/SELECT against EACH comparator; absolute net20 also required.',
                'binding': 'Separate LoRA-reference Holm(2) UNKNOWN-world family; mixed comparisons nominal descriptive.',
                'controlled_diagnostic': 'Opened report-only parent-level descriptions; N4 source-conditioned cohort has no inferential tests.'},
            'limitations': ['Previous failed predictions remain failed; no endpoint substitutes for another criterion.',
                'Control/native group coefficients are preserved by construction; clipping and Adam still couple gradients.',
                'Measured zero-addend state equality covers exactly two updates, not the full193-update trajectory.',
                'Binding LR-weighted coefficient exposure is about0.67717 of the prior mixture despite identical training rows.',
                'The control already failed retention at38/47; zero-addend recovery does not establish preservation.',
                'The fixed47 source cohort and both typed scoring policies remain unchanged.',
                'Binding worlds, not correlated paraphrases, are the statistical units.',
                'Nonsignificance and engineering acquisition-cost allowances do not establish equivalence or preservation.',
                'The controlled diagnostic is spent; its source-reference N4 is inadequate and cannot replace the fixed47 gate.',
                'Reserved coverage and legacy LC fingerprints remain unverified; human review NOT_PERFORMED.']}
    except (Exception, KeyboardInterrupt) as exc:
        return {'schema': 1, 'experiment': EXPERIMENT, 'status': 'BLOCKED_ON_COMPARABILITY',
                'error_type': type(exc).__name__, 'reason': str(exc), 'scientific_endpoints_emitted': False,
                'quality_claim': False, 'release_permitted': False, 'model_calls': 0}


def markdown(result):
    if result['status'] != 'COMPLETE_COMPARABLE':
        return '# BINDING-INTEGRATION-001\n\n**BLOCKED_ON_COMPARABILITY**\n\n' + result['reason'] + '\n\nNo endpoints emitted.\n'
    predictions = result['predictions']
    lines = ['# BINDING-INTEGRATION-001', '', 'Status: **COMPLETE_COMPARABLE**.', '',
        f"Primary engineering conjunction: **{predictions['primary_engineering_conjunction']}**. Binding efficacy: **{predictions['binding_efficacy']}**.", '',
        '| Model | Dose /384 | SELECT /384 | Permissive retained /47 | Permissive panel /150 | Strict fixed47 | Strict panel /150 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for role in ('integrated', 'mixed', 'lora'):
        row = result['endpoints'][role]; policies = row['retention']['policies']
        lines.append(f"| {role} | {row['dose']['correct']} | {row['select']['correct']} | {policies['permissive']['references_retained']} | "
            f"{policies['permissive']['panel_correct']} | {policies['strict']['references_retained']} | {policies['strict']['panel_correct']} |")
    lines += ['', '| Integrated minus comparator | Bank | net | nd | Exact p | Holm(2) p | Exact threshold at nd | 20-view floor |',
              '|---|---|---:|---:|---:|---:|---:|---|']
    for role, family in result['acquisition_contrasts_integrated_minus'].items():
        for bank, item in family.items():
            lines.append(f"| {role} | {bank} | {item['net']:+d} | {item['nd']} | {item['exact_p']:.9g} | {item['holm_adjusted_p']:.9g} | "
                         f"{item['minimum_significant_abs_net_at_observed_nd']} | {item['cross_training_draw_floor_cleared']} |")
    lines += ['', '| Model | Missing rows /96 | Vacant rows /96 | Answerable person /96 | Answerable role /96 | False UNKNOWN /192 | Missing worlds /32 | Vacant worlds /32 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for role, endpoint in result['binding']['endpoints'].items():
        counts = endpoint['categories']; worlds = result['binding']['world_maps'][role]['canonical_complete_exact']
        lines.append('| ' + role + ' | ' + ' | '.join(str(counts[name]['canonical_complete_exact']) for name in
            ('person_missing', 'person_vacant', 'person_answerable', 'role_answerable')) +
            f" | {endpoint['totals']['false_unknown_on_positive']} | {sum(worlds['person_missing'].values())} | {sum(worlds['person_vacant'].values())} |")
    lines += ['', '| Original BIND: integrated minus LoRA | net worlds | nd | Exact p | Holm(2) p | Exact threshold at nd |',
              '|---|---:|---:|---:|---:|---:|']
    for name in bind_score.UNKNOWN_CATEGORIES:
        item = result['binding']['contrasts_integrated_minus']['lora']['canonical_complete_exact'][name]
        lines.append(f"| {name} | {item['net']:+d} | {item['nd']} | {item['exact_p']:.9g} | {item['holm_adjusted_p']:.9g} | {item['minimum_significant_abs_net_at_observed_nd']} |")
    lines += ['', '| Prediction | Observed | Registered band | Verdict |', '|---|---:|---|---|']
    for name, value in predictions['bands'].items():
        lines.append(f"| {name} | {value['observed']} | {value['lower']}–{value['upper']} | {value['status']} |")
    diag = result['controlled_diagnostic']; cohort = diag['source_conditioned_retention_descriptive']
    lines += ['', 'Opened controlled diagnostic: **report-only**, with unchanged **N=4 / INSUFFICIENT_REFERENCE_COHORT**.', '',
              '| Model | All16 binding /32 parents | False UNKNOWN /256 | Strict arithmetic anchors /96 | Strict source-anchor retained /4 |',
              '|---|---:|---:|---:|---:|']
    for role, endpoint in diag['endpoints'].items():
        lines.append(f"| {role} | {endpoint['parent_counts']['binding']['canonical_complete_exact/all16']} | "
            f"{endpoint['totals']['binding']['false_exact_unknown']} | {endpoint['parent_counts']['arithmetic']['strict_anchor']} | {cohort['models'][role]['retained']} |")
    lines += ['', '| Controlled report-only prediction | Observed | Registered band | Verdict |', '|---|---:|---|---|']
    for name, value in diag['prospective_predictions']['bands'].items():
        lines.append(f"| {name} | {value['observed']} | {value['lower']}–{value['upper']} | {value['status']} |")
    lines += ['', 'Every integrated fixed47 source-known loss remains reported, irrespective of significance:', '',
              ', '.join(result['endpoints']['integrated']['retention']['policies']['permissive']['registered_source_lost_ids']) or 'None', '',
              *['- ' + value for value in result['limitations']], '',
              'RESULTS.json contains all parent/row maps, strict sensitivities, nominal secondary contrasts, exact thresholds, loss IDs and custody evidence.', '']
    return '\n'.join(lines)


def run(root=ROOT):
    root = Path(root).resolve()
    require(not (root / 'RESULTS.json').exists() and not (root / 'RESULTS.md').exists(), 'results already exist; replacement forbidden')
    result = build_report(root)
    run_probe.write(root / 'RESULTS.json', result)
    with (root / 'RESULTS.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(); result = run(args.root)
    print({'status': result['status'], 'results': str(args.root / 'RESULTS.json')})
    return 0 if result['status'] == 'COMPLETE_COMPARABLE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
