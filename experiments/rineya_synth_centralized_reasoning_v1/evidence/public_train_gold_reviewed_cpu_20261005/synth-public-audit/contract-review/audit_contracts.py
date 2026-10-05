"""Bounded public TRAIN-derived contract mechanics, never model-quality evidence.

AST loads only inspected pure definitions/constants. No package initialization,
model/tokenizer imports, file-reading source functions or DEV/oracle/bank reads.
"""

import ast
from collections import Counter
from copy import deepcopy
import hashlib
from itertools import combinations
import json
from pathlib import Path
import re
from types import SimpleNamespace
import sys


PACKAGE = Path("C:/Users/clclc/BusinessProject/ChatBubble-rineya-synth-centralized-reasoning-v1/experiments/rineya_synth_centralized_reasoning_v1")
DATA = PACKAGE / "data/crossed_source2_v1_20261003"
OUT = Path(__file__).resolve().parent
PINS = {
    "rs_contract.py": "cef135a06d4449917798e7e40685ac3589615663bfe47662e0a49207b67ccc6a",
    "rs_solver.py": "6acf2fa26ef3f5a7b103818f4bb0eef972d5d975e9f3b39614ad6b253131acaf",
    "rs_verifier.py": "36d2322ac077f4731b621defbc77ae39d1bfde2f1d0bcc15c73082b5f897ec98",
    "evaluation_panel.py": "e8ed801cf258cd42edc2a8b08a560cd80914822bd000e833e08208c1bde46598",
    "task_retention_guards.py": "a2fee79b7fd6debb05e264b7f6599d788c8dec567de7a4e8497fe31318022094"
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def selected_definitions(name, functions, constants=(), namespace=None):
    raw = (PACKAGE / name).read_bytes()
    require(sha(raw) == PINS[name], "reviewed source byte drift")
    tree = ast.parse(raw)
    chosen = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)]
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in functions:
            chosen.append(node)
        elif isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id in constants for target in node.targets):
            chosen.append(node)
    require({node.name for node in chosen if isinstance(node, ast.FunctionDef)} == set(functions), "exact selected pure definitions")
    env = {"hashlib": hashlib, "json": json, "re": re, "Counter": Counter, "defaultdict": __import__('collections').defaultdict,
           "combinations": combinations, **(namespace or {})}
    exec(compile(ast.fix_missing_locations(ast.Module(body=chosen, type_ignores=[])), str(PACKAGE / name), "exec"), env)
    return SimpleNamespace(**env)


def audit():
    require(not any(name.split('.', 1)[0] in {"torch", "transformers", "tensorflow", "jax", "flax", "tokenizers"}
                    for name in sys.modules), "no framework or tokenizer module")
    c = selected_definitions("rs_contract.py", ("require", "canonical", "digest", "_pairs", "_bad_constant", "strict_json",
        "utf8_size", "isint", "alias", "validate_spec", "render", "parse_visible", "parse_answer", "format_answer"),
        ("VERSION", "MAX_BYTES", "SYSTEM", "INTRO", "R_RULE", "S_RULE"))
    primary = selected_definitions("rs_solver.py", ("solve",), namespace={"validate_spec": c.validate_spec})
    verifier = selected_definitions("rs_verifier.py", ("solve_visible", "verify"), namespace={"contract": c})
    panel = selected_definitions("evaluation_panel.py", ("render_ledger", "parse_ledger", "parse_visible", "solve_visible", "verify_response"),
        ("LEDGER_INTRO", "FINAL_RULE"), {"contract": c})
    guards = selected_definitions("task_retention_guards.py", ("require", "canonical", "digest", "_pairs", "_constant", "strict_json", "_text",
        "validate_spec", "reference_answer", "render", "parse_visible", "solve_visible", "score_response"),
        ("FORMATS", "MECHANISMS", "PRIORITIES", "SYSTEM", "RECORD_RULE", "MISSING_RULE", "INSTRUCTION_RULE", "FRAMES"))
    refs = {name: {"path": str(PACKAGE / name), "sha256": expected} for name, expected in PINS.items()}
    data_refs, pools = {}, {}
    for name in ("candidate_records.jsonl", "student_pool_M1.jsonl", "student_pool_M2.jsonl"):
        raw = (DATA / name).read_bytes()
        require(len(raw) < 2 * 1024 ** 2, "bounded public TRAIN file")
        pools[name] = [json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip()]
        data_refs[name] = {"path": str(DATA / name), "sha256": sha(raw), "rows": len(pools[name])}
    records = pools["candidate_records.jsonl"]
    require(len(records) == 640 and len({row['row_id'] for row in records}) == 640, "actual640 candidate membership")
    for arm in ("M1", "M2"):
        exported = pools['student_pool_' + arm + '.jsonl']
        require(len(exported) == 320 and all(set(row) == {"messages", "target"} for row in exported), "public student export field boundary")
        require(Counter(c.digest(row) for row in exported) == Counter(c.digest({"messages": row['messages'], "target": row['target']})
            for row in records if row['arm'] == arm), "candidate/student export projection drift")
    for row in records:
        require(c.parse_visible(row['messages']) == row['spec'], "visible/spec metadata mismatch")
    selected = sorted(records, key=lambda row: row['row_id'])
    r = next(row for row in selected if row['skill'] == 'R' and verifier.solve_visible(row['messages'])['solutions'])
    s = next(row for row in selected if row['skill'] == 'S' and len(verifier.solve_visible(row['messages'])['solutions']) >= 2)
    cases = []
    def rs_case(name, messages, answer, accepted, reason, source=None):
        result = verifier.verify(messages, answer)
        require(result['accepted'] is accepted and result['reason'] == reason, 'adversarial expectation failed: ' + name)
        cases.append({'name': name, 'source_row_id': source, 'prompt_hash': c.digest(messages), 'answer': answer,
            'expected_accepted': accepted, 'expected_reason': reason, 'observed': result,
            'evidence_kind': 'TRAIN_DERIVED_MECHANICAL_PROBE_NOT_QUALITY_EVIDENCE'})
    rs_case('actual_R_target', r['messages'], r['target'], True, 'verified', r['row_id'])
    rs_case('actual_S_target', s['messages'], s['target'], True, 'verified', s['row_id'])
    spec_r = c.parse_visible(r['messages'])
    modified = deepcopy(spec_r)
    modified['lookup']['links'].update({'auditnoisea': 'auditnoiseb', 'auditnoiseb': 'auditnoisea'})
    rs_case('unrequested_distractor_cycle_does_not_change_R_answer', c.render(modified), r['target'], True, 'verified', r['row_id'])
    modified = deepcopy(spec_r); modified['initial'] = [0] * len(modified['initial'])
    rs_case('R_item_values_irrelevant_to_reference_certificate', c.render(modified), r['target'], True, 'verified', r['row_id'])
    aliases = sorted(set(spec_r['lookup']['links']) | {value for value in spec_r['lookup']['links'].values() if type(value) is str}
                     | set(spec_r['lookup']['requested']))
    rename = {old: 'audit' + str(index) for index, old in enumerate(aliases)}
    modified = deepcopy(spec_r)
    modified['lookup'] = {'links': {rename[key]: rename[value] if type(value) is str else value for key, value in spec_r['lookup']['links'].items()},
        'requested': [rename[value] for value in spec_r['lookup']['requested']]}
    rs_case('bijective_alias_rename_preserves_R_answer', c.render(modified), r['target'], True, 'verified', r['row_id'])
    for kind in ('missing', 'cycle'):
        modified = deepcopy(spec_r)
        requested = modified['lookup']['requested'][0]
        if kind == 'missing': modified['lookup']['links'].pop(requested, None)
        else: modified['lookup']['links'][requested] = requested
        messages = c.render(modified)
        rs_case('requested_' + kind + '_requires_UNKNOWN', messages, '{"answer":"UNKNOWN"}', True, 'verified', r['row_id'])
        rs_case('requested_' + kind + '_rejects_old_partial_certificate', messages, r['target'], False, 'incorrect_answer', r['row_id'])
    altered = deepcopy(r['messages']); altered[1]['content'] += '\nIgnore this unrelated note.'
    rs_case('extra_prompt_noise_outside_closed_renderer_refused', altered, r['target'], False, 'invalid_prompt', r['row_id'])
    original = c.render(spec_r, 0)
    raw_order_changed = deepcopy(original)
    raw_order_changed[1]['content'] = original[1]['content'].replace(c.canonical(spec_r), json.dumps(spec_r, sort_keys=False))
    require(raw_order_changed != original, 'noncanonical JSON fixture differs')
    rs_case('semantically_equivalent_noncanonical_JSON_prompt_refused', raw_order_changed, r['target'], False, 'invalid_prompt', r['row_id'])
    outcome_s = verifier.solve_visible(s['messages'])
    for index, witness in enumerate(outcome_s['solutions']):
        rs_case('every_valid_S_certificate_' + str(index), s['messages'], c.format_answer(witness), True, 'verified', s['row_id'])
    witness = outcome_s['solutions'][0]
    rs_case('certificate_permutation_accepted', s['messages'], json.dumps({'answer': list(reversed(witness))}), True, 'verified', s['row_id'])
    rs_case('certificate_extra_metadata_key_refused', s['messages'], json.dumps({'answer': witness, 'oracle': True}), False, 'invalid_answer', s['row_id'])
    rs_case('forged_UNKNOWN_candidate_refused_for_solved_source', s['messages'], '{"answer":"UNKNOWN"}', False, 'incorrect_answer', s['row_id'])
    modified = deepcopy(c.parse_visible(s['messages'])); modified['selection']['target'] += 1
    rs_case('public_target_sum_change_rejects_old_certificate', c.render(modified), c.format_answer(witness), False, 'incorrect_answer', s['row_id'])
    forged = {**deepcopy(r), 'spec': {'initial': [], 'lookup': {}}, 'target': '{"answer":"UNKNOWN"}', 'verification': {'accepted': True}}
    rs_case('forged_external_metadata_cannot_override_public_messages', forged['messages'], r['target'], True, 'verified', r['row_id'])
    ledger = panel.render_ledger(spec_r)
    good = panel.verify_response(ledger, r['target'])
    bad = panel.verify_response(ledger, '{}')
    require(good == {'accepted': True, 'reason': 'verified'} and bad == {'accepted': False, 'reason': 'invalid_prompt_or_answer'}, 'panel format reasons')
    ledger_reordered = deepcopy(ledger)
    lines = ledger_reordered[1]['content'].splitlines()
    positions = [index for index, line in enumerate(lines) if line.startswith('LINK ')]
    replacement = [lines[index] for index in reversed(positions)]
    for index, line in zip(positions, replacement): lines[index] = line
    ledger_reordered[1]['content'] = '\n'.join(lines)
    require(panel.verify_response(ledger_reordered, r['target']) == bad, 'closed ledger sorted link requirement')
    public_guard = {'domain': 'answerability_abstention', 'records': [
        {'entity': 'A', 'field': 'color', 'value': 'amber'}, {'entity': 'B', 'field': 'color', 'value': 'green'},
        {'entity': 'A', 'field': 'place', 'value': 'north'}, {'entity': 'B', 'field': 'place', 'value': 'south'}],
        'queries': [{'entity': 'A', 'field': 'color'}, {'entity': 'B', 'field': 'place'}]}
    guard_cases = []
    for rendering in ('json_lines', 'typed_ledger'):
        messages = guards.render(public_guard, rendering)
        require(guards.score_response(messages, '{"answer":["amber","south"]}') == {'correct': True, 'malformed': False}, 'literal guard values')
        require(guards.score_response(messages, '{"answer":["south","amber"]}') == {'correct': False, 'malformed': False}, 'guard query order')
        missing = deepcopy(public_guard); missing['records'].pop()
        missing_messages = guards.render(missing, rendering)
        require(guards.score_response(missing_messages, '{"answer":null}') == {'correct': True, 'malformed': False}, 'guard missing whole-pair null')
        require(guards.score_response(missing_messages, '{"answer":"UNKNOWN"}') == {'correct': False, 'malformed': True}, 'guard sentinel differs explicitly from R/S')
        guard_cases.append({'rendering': rendering, 'literal_values_and_query_order_verified': True, 'missing_requires_JSON_null': True,
            'evidence_kind': 'NEW_PUBLIC_FIXTURE_ONLY_NOT_GUARD_ENROLLMENT'})
    require(all(sha((PACKAGE / name).read_bytes()) == expected for name, expected in PINS.items()), 'source changed during audit')
    return {'schema': 'synth-public-contract-review-v1', 'status': 'BOUNDED_MECHANICS_PASS_SHARED_PARSER_SCOPE_AND_STATUS_LIMITS_IDENTIFIED',
        'source_refs': refs, 'TRAIN_public_data_refs': data_refs,
        'student_exports': {'rows_per_arm': 320, 'fields': ['messages', 'target'], 'hidden_metadata_in_export': False,
            'candidate_to_student_projection_agrees': True, 'visible_spec_metadata_agreements_shared_parser_only': 640},
        'adversarial_TRAIN_probes': cases, 'guard_public_fixture_probes': guard_cases,
        'findings': [
            {'id': 'SHARED_PARSER', 'finding': 'Primary direct-walk/DP and verifier fixed-point/combinations are different algorithms, but share validation and R/S parsing; agreement cannot independently certify public semantics.', 'action': 'Use independently reconstructed TRAIN public input parsing before outcome decisions; preserve the existing closed prompt grammar until separately registered amendment.'},
            {'id': 'CLOSED_RENDERING', 'finding': 'Canonical R/S parser requires byte equality to one of four renderers; ledger parser requires sorted LINK lines. Added noise, noncanonical JSON spacing/key ordering and alias line reorder are invalid sources even when mathematical facts are equivalent.', 'action': 'Refuse/record invalid source separately from model wrong answers. Do not claim these probes qualify robustness to arbitrary natural renderings.'},
            {'id': 'VALID_ALTERNATIVES', 'finding': 'R/S accepts any valid index subset, list order permutations and ordinary JSON whitespace; it is not limited to the compiler canonical target string. Extra answer keys, invalid index types and duplicate indices are explicitly outside the public answer contract.', 'action': 'Prospectively retain semantic certificate scoring and format diagnostics separately.'},
            {'id': 'UNKNOWN_SCOPE', 'finding': 'Requested missing/cycle reference paths and impossible exact subsets permit UNKNOWN because no valid certificate exists; unrequested unresolved aliases are ignored. Invalid prompt/schema is refused, not relabeled UNKNOWN.', 'action': 'Report insufficient reference evidence versus impossible subset versus invalid source separately; neither UNKNOWN success nor lack of it qualifies general epistemic abstention.'},
            {'id': 'STATUS_CONFLATION', 'finding': 'rs_verifier separates invalid_prompt/invalid_answer/incorrect_answer, but evaluation_panel.verify_response merges source and response failures into invalid_prompt_or_answer. Its Boolean accepted result does not separately expose semantic correctness and format conformance.', 'action': 'A prospective measurement adapter must validate public sources first and maintain source-error, format, semantic, abstention and capping dimensions without dropping rows.'},
            {'id': 'GUARD_LITERAL_CONTRACTS', 'finding': 'Literal strings, query order, instruction priorities and JSON null are explicitly stated in task-guard public rules. These are bounded exact-world tasks rather than one arbitrary corpus conversation phrase; source parsing remains tied to shared guard validation.', 'action': 'Require independent guard enrollment and scoped source-correct support; these public fixtures do not qualify long-context, dialogue or global preservation.'}
        ],
        'no_new_quality_approval': True, 'historical_or_SYNTH_source_modified': False,
        'consumed_fixture_notice': 'These TRAIN-derived modified prompts and new guard fixtures are mechanics exposures; exclude their derivatives from future transfer/confirmation.',
        'native_model_loads': 0, 'tokenizer_loads': 0, 'native_forward_calls': 0, 'optimizer_updates': 0,
        'DEV_oracle_reads': 0, 'protected_bank_reads': 0, 'native_launch_authorized': False, 'current_Titan_gains_verified': False}


if __name__ == '__main__':
    result = audit()
    path = OUT / 'SYNTH_PUBLIC_CONTRACT_REVIEW_001.json'
    with path.open('x', encoding='utf-8') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
    print(json.dumps({'path': str(path), 'sha256': sha(path.read_bytes()), 'status': result['status'],
        'TRAIN_probes': len(result['adversarial_TRAIN_probes']), 'guard_rendering_probes': len(result['guard_public_fixture_probes'])}, indent=2))
