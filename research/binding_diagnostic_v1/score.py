"""Model-free, fail-closed scoring of the registered frozen-model diagnostic.

Only RESULTS.json and RESULTS.md are created. No endpoint is published unless
all data, code, model-byte, runtime and supervised-generation custody gates pass.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
import importlib.util
import hashlib
import itertools
import math
from pathlib import Path
import sys
import unicodedata

if __package__:
    from . import arithmetic, binding, common as c
else:
    import arithmetic
    import binding
    import common as c

ROLES = ('source', 'lora', 'mixed')
POLICIES = ('permissive', 'strict')
HERE = Path(__file__).resolve().parent
REQUIRED_ARTIFACTS = {'PLAN.json', 'GRANT.json', 'PREREG.md', 'CRITERIA.json', 'SESSION_ENVELOPE.json',
                      *{'candidate-v1/' + name for name in ('DIAGNOSTIC_INPUTS.jsonl', 'GOLD.jsonl',
                                                           'METADATA.json', 'DATA_ADMISSION.json')}}


def no_model_imports():
    c.require(not any(name in sys.modules for name in ('torch', 'tensorflow', 'jax', 'flax', 'transformers', 'tokenizers')),
              'offline scoring must remain standard-library-only and model-free')


def normalize(text, *, casefold=True):
    c.require(isinstance(text, str), 'exact text scorer requires strings')
    value = ' '.join(unicodedata.normalize('NFKC', text).split())
    return value.casefold() if casefold else value


def probability(wins, losses):
    n = wins + losses
    return min(Fraction(1), Fraction(2 * sum(math.comb(n, k) for k in range(min(wins, losses) + 1)), 1 << n))


def exact_pair(reference, candidate):
    """Candidate minus reference; maps contain one Boolean per independent parent."""
    c.require(reference and reference.keys() == candidate.keys() and
              all(type(v) is bool for values in (reference, candidate) for v in values.values()),
              'nonempty, aligned Boolean parent maps required')
    wins = sorted(k for k in reference if candidate[k] and not reference[k])
    losses = sorted(k for k in reference if reference[k] and not candidate[k])
    nd = len(wins) + len(losses)
    def threshold(alpha):
        return next((net for net in range(nd % 2, nd + 1, 2)
                     if probability((nd + net) // 2, (nd - net) // 2) <= alpha), None)
    p = probability(len(wins), len(losses))
    return {'n': len(reference), 'reference_correct': sum(reference.values()), 'candidate_correct': sum(candidate.values()),
            'wins': len(wins), 'losses': len(losses), 'win_parent_ids': wins, 'loss_parent_ids': losses,
            'net': len(wins) - len(losses), 'nd': nd, 'exact_p': float(p), 'exact_p_fraction': str(p),
            'minimum_significant_abs_net_at_observed_nd': threshold(Fraction(1, 20)),
            'minimum_initial_Holm2_abs_net_at_observed_nd': threshold(Fraction(1, 40)),
            'thresholds_are_power_calculations': False}


def holm_two(primary):
    c.require(set(primary) == {'binding', 'arithmetic'}, 'exactly the registered Holm(2) family is required')
    running = Fraction(0)
    for rank, (_, item) in enumerate(sorted(primary.items(), key=lambda pair: Fraction(pair[1]['multiplicity_p_fraction']))):
        running = max(running, min(Fraction(1), (2 - rank) * Fraction(item['multiplicity_p_fraction'])))
        item.update(holm_adjusted_p=float(running), holm_adjusted_p_fraction=str(running), holm_family_size=2)


def verify_frozen(root):
    root = c.absolute(root, directory=True)
    freeze = c.read(root / 'PRELAUNCH_BINDING.json')
    c.require(freeze.get('status') == 'PASS_INPUTS_SOURCE_CODE_BEFORE_EXECUTION', 'prelaunch freeze did not pass')
    code = freeze.get('code_files_sha256', {})
    artifacts = freeze.get('artifact_files_sha256', {})
    c.require({'score.py', 'common.py', 'probe.py', 'arithmetic.py', 'binding.py'} <= set(code), 'incomplete scoring code freeze')
    c.require(REQUIRED_ARTIFACTS <= set(artifacts), 'incomplete prospective artifact freeze')
    for base, pins in ((HERE, code), (root, artifacts)):
        for name, expected in pins.items():
            relative = Path(name)
            c.require(not relative.is_absolute() and '..' not in relative.parts and
                      (base / relative).resolve().is_relative_to(base.resolve()), 'freeze path escapes its named root')
            c.require(c.sha(base / relative) == expected, 'frozen bytes changed: ' + name)
    return freeze


def load_typed_scorer():
    path = Path(arithmetic.FROZEN_SCORER['path'])
    c.require(c.sha(path) == arithmetic.FROZEN_SCORER['sha256'], 'qualified typed arithmetic grammar changed')
    spec = importlib.util.spec_from_file_location('diagnostic_frozen_typed_arithmetic', path)
    c.require(spec is not None and spec.loader is not None, 'typed grammar cannot be loaded')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    no_model_imports()
    return module


class ByteLevelDecoder:
    """Reproduce the frozen ByteLevel decoder with skip_special_tokens=True."""
    def __init__(self, tokenizer, config):
        c.require(tokenizer.get('decoder') == {'type': 'ByteLevel', 'add_prefix_space': False,
                  'trim_offsets': False, 'use_regex': False}, 'unsupported tokenizer decoder; fail closed')
        c.require(config.get('clean_up_tokenization_spaces', False) is False,
                  'tokenizer cleanup requires a separately qualified decoder')
        vocab = tokenizer['model']['vocab']
        self.vocab = {value: key for key, value in vocab.items()}
        c.require(len(self.vocab) == len(vocab), 'duplicate tokenizer vocabulary IDs')
        self.special = set()
        for token in tokenizer['added_tokens']:
            self.vocab[token['id']] = token['content']
            if token['special']:
                self.special.add(token['id'])
        # Explicit construction avoids dependence on any tokenizer package.
        standard = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
        extra = [value for value in range(256) if value not in standard]
        self.bytes = {chr(value): value for value in standard}
        self.bytes.update({chr(256 + index): value for index, value in enumerate(extra)})

    def decode(self, ids):
        data = bytearray()
        for token in ids:
            c.require(token in self.vocab, 'generation contains an unmapped vocabulary ID')
            if token in self.special:
                continue
            for char in self.vocab[token]:
                if char in self.bytes:
                    data.append(self.bytes[char])
                else:
                    data.extend(char.encode('utf-8'))
        return data.decode('utf-8', errors='replace')


def expected_eos(source, tokenizer, config):
    # Match the frozen run._eos_ids: generation_config overrides tokenizer EOS.
    token = config.get('eos_token')
    if isinstance(token, dict):
        token = token['content']
    matches = [value['id'] for value in tokenizer['added_tokens'] if value['content'] == token]
    c.require(len(matches) == 1, 'tokenizer EOS identity is ambiguous')
    value = c.read(source / 'generation_config.json').get('eos_token_id', matches[0])
    ids = value if isinstance(value, list) else [value]
    c.require(ids and all(type(v) is int and 0 <= v < 151936 for v in ids), 'cannot recover the original tokenizer EOS IDs')
    return sorted(set(ids))


def validate_data(inputs, gold, metadata):
    c.require(len(gold) == 896 and len({r['id'] for r in gold}) == 896 and
              [r['id'] for r in gold] == [r['id'] for r in inputs], 'gold/input IDs or order differ')
    by_id = {r['id']: r for r in inputs}
    c.require(set(metadata) == {'binding', 'arithmetic'} and
              Counter(row.get('section') for row in gold) == {'binding': 512, 'arithmetic': 384},
              'registered gold section inventory differs')
    for section, module, record in (('binding', binding, 'BINDING_DIAGNOSTIC_FACT_PAIRS'),
                                    ('arithmetic', arithmetic, arithmetic.RECORD)):
        targets = [row.copy() for row in gold if row['section'] == section]
        if section == 'arithmetic':
            for row in targets:
                row.pop('section')
        policy = [{key: value for key, value in by_id[row['id']].items() if key != 'prompt_ids'} for row in targets]
        module.validate_rows({'schema': 1, 'record': record, 'seed': 31, 'policy_inputs': policy,
                              'gold': targets, 'metadata': metadata[section]})
    expected = set(itertools.product(binding.DOMAINS, (0, 1), ('missing', 'vacant'), (False, True)))
    crossed = defaultdict(set)
    for target in gold:
        if target['section'] == 'binding':
            c.require(type(target['answerable']) is bool and type(target['template']) is int,
                      'binding factors have invalid types')
            crossed[target['parent_id']].add((target['domain'], target['template'], target['pair'], target['answerable']))
    c.require(len(crossed) == 32 and all(values == expected for values in crossed.values()) and
              len(metadata['binding']) == 32 and {r['parent_id'] for r in metadata['binding']} == set(crossed),
              'binding factors/metadata do not describe the full registered crossing')
    return {row['id']: row for row in gold}


def validate_criteria(criteria):
    primary = criteria['primary']; ar = primary['arithmetic']
    c.require(criteria['paired_units'] == {'binding': 32, 'arithmetic': 96} and primary['holm_family_size'] == 2 and
              primary['adjusted_exact_p_at_most'] == .05 and
              primary['binding'] == {'endpoint': 'all16_canonical_complete_exact', 'contrast': 'mixed_minus_lora',
                                      'positive_net_at_least': 8} and
              ar == {'endpoint': 'strict_correct_on_source_successful_explicit_anchor', 'anchor_numeric_variant': 0,
                     'anchor_response_type': 'unit_explicit', 'eligibility_model': 'source', 'contrast': 'mixed_minus_lora',
                     'minimum_reference_n': 24, 'minimum_abs_net': 7, 'minimum_abs_net_fraction': .2,
                     'insufficient_label': 'INSUFFICIENT_REFERENCE_COHORT', 'source_conditioned_source_vs_candidate_tests': 'FORBIDDEN',
                     'empty_cohort_multiplicity_placeholder': 1}, 'registered primary rules changed or scorer structure differs')
    c.require(all(criteria[key] is False for key in ('legacy_acquisition_comparisons_performed', 'legacy_acquisition_floor_changed',
              'goal_success_redefined', 'quality_claim', 'promotion_permitted')), 'diagnostic cannot redefine or promote the goal')


def verify_predictions(generations, inputs, reservation, decoder, eos):
    predictions = generations['predictions']
    c.require(len(predictions) == 896 and [r['id'] for r in predictions] == [r['id'] for r in inputs],
              'generation IDs/order differ from the complete admitted input')
    c.require(generations['eos_ids'] == eos, 'original EOS policy changed')
    for row, original in zip(predictions, inputs):
        tokens = row['generated_ids_including_eos']
        c.require(row['parent_id'] == original['parent_id'] and type(row['terminated']) is bool and
                  isinstance(row['prediction'], str) and type(row['input_tokens']) is int and
                  row['input_tokens'] == len(original['prompt_ids']), 'generation input/parent custody differs')
        c.require(isinstance(tokens, list) and 1 <= len(tokens) == row['generated_tokens'] <= 24 and
                  all(type(row[key]) is int for key in ('generated_tokens', 'forward_calls', 'processed_input_tokens')) and
                  all(type(t) is int and 0 <= t < 151936 for t in tokens) and
                  row['terminated'] == (tokens[-1] in eos) and not any(t in eos for t in tokens[:-1]) and
                  (row['terminated'] or len(tokens) == 24), 'generation token bounds or EOS custody differ')
        c.require(decoder.decode(tokens) == row['prediction'], 'saved prediction differs from exact generated-token decoding')
        c.require(row['forward_calls'] == len(tokens) and
                  row['processed_input_tokens'] == row['input_tokens'] + len(tokens) - 1,
                  'actual generation token/call accounting differs')
    actual = {'rows': len(predictions),
              'generated_tokens_including_eos': sum(row['generated_tokens'] for row in predictions),
              'forward_calls': sum(row['forward_calls'] for row in predictions),
              'processed_input_tokens': sum(row['processed_input_tokens'] for row in predictions)}
    c.require(generations['actual_budget_charges'] == actual and generations['extra_warmup_or_qualification_calls'] == 0 and
              actual['generated_tokens_including_eos'] <= reservation['reserved_generated_tokens'] and
              actual['forward_calls'] <= reservation['forward_calls'] and
              actual['processed_input_tokens'] <= reservation['processed_input_tokens'], 'actual charges exceed or disagree with reservations')
    return actual


def verify_worker(plan, plan_path, grant_path, role, inputs, decoder, eos):
    root = Path(plan['output_root'])
    out = root / role
    receipt = c.read(out / 'DIAGNOSTIC_BINDING.json')
    attempt = 1
    if receipt.get('status') != 'COMPLETE':
        prior_supervisor = c.read(out / 'supervisor.json')
        c.require(receipt.get('status') == 'FAILED' and receipt.get('complete') is False and
                  receipt.get('attempt') == 1 and receipt.get('model_role') == role and
                  receipt.get('plan_sha256') == c.sha(plan_path) and receipt.get('grant_sha256') == c.sha(grant_path) and
                  receipt.get('supervisor_sha256') == c.sha(out / 'supervisor.json') and
                  not (out / 'GENERATIONS.json').exists() and
                  prior_supervisor.get('status') in ('EXITED', 'TIMEOUT_OR_INTERRUPTED') and
                  type(prior_supervisor.get('returncode')) is int and prior_supervisor['returncode'] != 0,
                  'retry cannot replace a successful or incompletely verified original attempt')
        out = root / (role + '-retry1'); attempt = 2
        receipt = c.read(out / 'DIAGNOSTIC_BINDING.json')
    else:
        c.require(not (root / (role + '-retry1')).exists(), 'successful output replacement is forbidden')
    c.require(not any((out / name).exists() for name in ('FAILURE.json', 'supervisor_failure.json')),
              'selected worker has a failure receipt')
    c.require(receipt.get('status') == 'COMPLETE' and receipt.get('complete') is True and
              receipt.get('model_role') == role and receipt.get('attempt') == attempt and
              type(receipt.get('controller_returncode')) is int and receipt['controller_returncode'] == 0 and
              receipt.get('failure') is None, 'controller did not establish successful completion')
    for field, filename in (('supervisor', 'supervisor.json'), ('worker_terminal', 'worker_terminal.json'),
                            ('generations', 'GENERATIONS.json')):
        c.require(c.sha(out / filename) == receipt[field + '_sha256'], 'worker artifact binding changed')
    supervisor, terminal, generations = (c.read(out / name) for name in ('supervisor.json', 'worker_terminal.json', 'GENERATIONS.json'))
    c.verify_terminal(supervisor, terminal)
    identities = c.verify_launch(out, plan_path, grant_path, plan, role, supervisor, attempt=attempt)
    c.require(all(receipt.get(key) == value for key, value in identities.items()), 'launch identities changed')
    for key, path in (('plan_sha256', plan_path), ('grant_sha256', grant_path)):
        c.require(receipt[key] == terminal[key] == generations[key] == c.sha(path), 'worker belongs to another plan/grant')
    slot = c.slot_name(role, attempt)
    reservation_path = root / ('RESERVATION_' + slot + '.json')
    reservation = c.read(reservation_path)
    c.require(c.sha(reservation_path) == receipt['reservation_sha256'] and
              reservation['model_role'] == role and reservation['attempt'] == attempt and
              reservation['plan_sha256'] == c.sha(plan_path) and reservation['grant_sha256'] == c.sha(grant_path) and
              reservation['reserved_worker_seconds'] == 900 and reservation['rows'] == 896 and
              reservation['forward_calls'] == reservation['reserved_generated_tokens'] == 896 * 24 and
              reservation['processed_input_tokens'] == sum(len(row['prompt_ids']) for row in inputs) + 896 * 24,
              'registered reservation geometry differs')
    owner = reservation['controller_owner']
    launch = c.read(out / 'launch.json')
    # write_new stores a trailing newline, which is part of the lease file hash.
    c.require(hashlib.sha256(c.canonical(owner) + b'\n').hexdigest() == reservation['controller_lease_sha256'] and
              owner['owner_pid'] == launch['supervisor_pid'] and owner['plan_sha256'] == c.sha(plan_path) and
              owner['grant_sha256'] == c.sha(grant_path), 'reservation controller ownership differs')
    before = generations['before']
    c.require(before == generations['after'] == terminal['before'] == terminal['after'] == c.snapshot(plan, role),
              'model/input/code/runtime custody differs before, after, or now')
    c.require(terminal['model_role'] == generations['model_role'] == role and
              terminal['generations_sha256'] == c.sha(out / 'GENERATIONS.json') and
              generations['record'] == 'BINDING_DIAGNOSTIC_GENERATIONS' and generations['rows'] == 896 and
              generations['parent_worlds'] == 128 and generations['gold_file_read'] is False and
              generations['training_permission'] is False and generations['numerics'] == c.NUMERICS,
              'generation scope/numerics differ')
    precheck = c.read(out / 'PRECHECK.json')
    c.require(precheck['status'] == 'PRECHECK_PASSED' and precheck['before'] == before and
              precheck['model_role'] == role and precheck['numerics'] == c.NUMERICS and
              precheck['model_input_fields'] == ['prompt_ids'] and precheck['worker_input_fields'] == sorted(c.INPUT_KEYS) and
              precheck['gold_file_read'] is False and precheck['plan_sha256'] == c.sha(plan_path) and
              precheck['grant_sha256'] == c.sha(grant_path), 'precheck differs from the admitted worker')
    charges = verify_predictions(generations, inputs, reservation, decoder, eos)
    return generations, {'directory': str(out), 'attempt': attempt,
                         'files_sha256': {name: c.sha(out / name) for name in ('DIAGNOSTIC_BINDING.json', 'GENERATIONS.json',
                            'supervisor.json', 'worker_terminal.json', 'PRECHECK.json', 'launch.json', 'spawn.json', 'worker_claim.json')},
                         'reservation_sha256': c.sha(reservation_path), 'worker_seconds': supervisor['wall_seconds'],
                         'actual_budget_charges': charges}


def score_rows(inputs, gold, predictions, typed):
    details = {}; groups = {'binding': defaultdict(list), 'arithmetic': defaultdict(list)}
    for original, pred in zip(inputs, predictions):
        target = gold[original['id']]
        c.require(pred['id'] == target['id'] and pred['parent_id'] == target['parent_id'], 'score join differs')
        text, terminated = pred['prediction'], pred['terminated']
        row = {'id': target['id'], 'parent_id': target['parent_id'], 'section': target['section'],
               'prediction': text, 'reference': target['reference'], 'terminated': terminated}
        if target['section'] == 'binding':
            parsed = binding.parse_policy_input(original['prompt'])
            canonical = normalize(text) == normalize(target['reference'])
            strict = normalize(text, casefold=False) == normalize(target['reference'], casefold=False)
            row.update({key: target[key] for key in ('domain', 'template', 'pair', 'state', 'answerable')})
            row.update(canonical_text_exact=canonical, strict_text_exact=strict,
                       canonical_complete_exact=canonical and terminated, strict_complete_exact=strict and terminated,
                       false_exact_unknown=target['answerable'] and normalize(text) == 'unknown',
                       exact_roster_person_on_unknown=not target['answerable'] and normalize(text) in {normalize(name) for name in parsed['roster']})
        else:
            row.update({key: target[key] for key in ('family', 'subtype', 'numeric_variant', 'response_type')})
            for policy in POLICIES:
                score = typed.score_prediction(text, terminated, target['targets'][policy])
                c.require(type(score['correct']) is bool, 'typed scorer returned non-Boolean correctness')
                row[policy + '_correct'] = score['correct']
                row[policy + '_score'] = score
        details[row['id']] = row
        groups[target['section']][target['parent_id']].append(row)
    maps = {'binding': {}, 'arithmetic': {}}
    for parent, rows in groups['binding'].items():
        c.require(len(rows) == 16, 'binding statistical unit changed')
        subsets = {'all16': rows, 'all_positive8': [r for r in rows if r['answerable']],
                   'all_unknown8': [r for r in rows if not r['answerable']]}
        for field in ('domain', 'template', 'pair', 'state'):
            for value in sorted({str(row[field]) for row in rows}):
                subsets[field + '/' + value] = [row for row in rows if str(row[field]) == value]
        for domain in binding.DOMAINS:
            for template in (0, 1):
                for pair in ('missing', 'vacant'):
                    key = f'counterfactual_pair/{domain}/{template}/{pair}'
                    subsets[key] = [r for r in rows if (r['domain'], r['template'], r['pair']) == (domain, template, pair)]
                    c.require(len(subsets[key]) == 2, 'counterfactual pair is incomplete')
        for metric in ('canonical_complete_exact', 'strict_complete_exact'):
            for key, selected in subsets.items():
                maps['binding'].setdefault(metric + '/' + key, {})[parent] = all(row[metric] for row in selected)
        maps['binding'].setdefault('no_false_unknown_all_positive8', {})[parent] = not any(r['false_exact_unknown'] for r in rows)
        maps['binding'].setdefault('any_exact_roster_person_on_unknown', {})[parent] = any(r['exact_roster_person_on_unknown'] for r in rows)
    for parent, rows in groups['arithmetic'].items():
        anchors = [r for r in rows if r['numeric_variant'] == 0 and r['response_type'] == 'unit_explicit']
        c.require(len(rows) == 4 and len(anchors) == 1, 'arithmetic parent/anchor structure changed')
        for policy in POLICIES:
            key = policy + '_correct'
            maps['arithmetic'].setdefault(policy + '_anchor', {})[parent] = anchors[0][key]
            maps['arithmetic'].setdefault(policy + '_all4', {})[parent] = all(row[key] for row in rows)
            for response in arithmetic.RESPONSE_TYPES:
                maps['arithmetic'].setdefault(policy + '/' + response + '_both_variants', {})[parent] = all(
                    row[key] for row in rows if row['response_type'] == response)
    binding_rows = [row for row in details.values() if row['section'] == 'binding']
    arithmetic_rows = [row for row in details.values() if row['section'] == 'arithmetic']
    totals = {'binding': {'rows': len(binding_rows), 'parents': len(groups['binding']),
              **{key: sum(row[key] for row in binding_rows) for key in ('terminated', 'canonical_text_exact', 'strict_text_exact',
                  'canonical_complete_exact', 'strict_complete_exact', 'false_exact_unknown', 'exact_roster_person_on_unknown')}},
              'arithmetic': {'rows': len(arithmetic_rows), 'parents': len(groups['arithmetic']),
              **{key: sum(row[key] for row in arithmetic_rows) for key in ('terminated', 'permissive_correct', 'strict_correct')},
              'by_response_type': {response: {'rows': sum(r['response_type'] == response for r in arithmetic_rows),
                 **{policy: sum(r[policy + '_correct'] for r in arithmetic_rows if r['response_type'] == response) for policy in POLICIES}}
                 for response in arithmetic.RESPONSE_TYPES}}}
    return {'totals': totals, 'parent_counts': {section: {name: sum(values.values()) for name, values in endpoints.items()}
                for section, endpoints in maps.items()}, 'rows': details}, maps


def analyze(endpoints, maps, gold, criteria):
    source_anchors = maps['source']['arithmetic']['strict_anchor']
    cohort = sorted(parent for parent, correct in source_anchors.items() if correct)
    anchors = {row['parent_id']: row for row in gold.values() if row['section'] == 'arithmetic' and
               row['numeric_variant'] == 0 and row['response_type'] == 'unit_explicit'}
    c.require(len(anchors) == 96 and source_anchors.keys() == anchors.keys(), 'source anchor identity inventory differs')
    primary = {'binding': exact_pair(maps['lora']['binding']['canonical_complete_exact/all16'],
                                     maps['mixed']['binding']['canonical_complete_exact/all16'])}
    primary['binding'].update(contrast='mixed_minus_lora', statistical_unit='parent_world',
                              multiplicity_p_fraction=primary['binding']['exact_p_fraction'], observed_test=True)
    if cohort:
        primary['arithmetic'] = exact_pair({p: maps['lora']['arithmetic']['strict_anchor'][p] for p in cohort},
                                            {p: maps['mixed']['arithmetic']['strict_anchor'][p] for p in cohort})
        primary['arithmetic'].update(observed_test=True, multiplicity_p_fraction=primary['arithmetic']['exact_p_fraction'])
    else:
        primary['arithmetic'] = {'n': 0, 'net': None, 'nd': None, 'exact_p': None, 'observed_test': False,
                                 'multiplicity_p_fraction': '1', 'placeholder_reason': 'EMPTY_SOURCE_REFERENCE_COHORT'}
    primary['arithmetic'].update(contrast='mixed_minus_lora', statistical_unit='source_successful_anchor_parent',
                                 source_only_eligibility=True, candidate_conditioned_eligibility=False)
    holm_two(primary)
    alpha = criteria['primary']['adjusted_exact_p_at_most']
    binding_rule = criteria['primary']['binding']
    primary['binding']['large_positive_diagnostic_difference'] = (
        primary['binding']['net'] >= binding_rule['positive_net_at_least'] and primary['binding']['holm_adjusted_p'] <= alpha)
    ar = criteria['primary']['arithmetic']
    floor = max(ar['minimum_abs_net'], math.ceil(Fraction(str(ar['minimum_abs_net_fraction'])) * len(cohort)))
    primary['arithmetic']['minimum_abs_net_for_practical_diagnostic_effect'] = floor
    sufficient = len(cohort) >= ar['minimum_reference_n']
    meaningful = sufficient and primary['arithmetic']['holm_adjusted_p'] <= alpha and abs(primary['arithmetic']['net']) >= floor
    primary['arithmetic'].update(reference_cohort_status='SUFFICIENT_REGISTERED_REFERENCE_COUNT' if sufficient else ar['insufficient_label'],
        large_relative_diagnostic_difference=bool(meaningful), direction=('mixed_higher' if primary['arithmetic']['net'] > 0 else 'mixed_lower')
        if meaningful else 'NOT_ESTABLISHED', preservation_established=False)
    retention = {'selection_rule': 'source strict correctness on numeric_variant=0,response_type=unit_explicit',
                 'source_only_eligibility': True, 'candidate_conditioned_eligibility': False,
                 'source_vs_candidate_p_values_permitted': False, 'n': len(cohort), 'parent_ids': cohort,
                 'anchor_ids': [anchors[p]['id'] for p in cohort],
                 'family_reference_counts': dict(Counter(anchors[p]['family'] for p in cohort)), 'models': {}}
    for role in ROLES:
        values = maps[role]['arithmetic']['strict_anchor']
        lost = [parent for parent in cohort if not values[parent]]
        retention['models'][role] = {'retained': len(cohort) - len(lost), 'n': len(cohort),
            'retained_fraction': (len(cohort) - len(lost)) / len(cohort) if cohort else None,
            'lost_parent_ids': lost, 'lost_anchor_ids': [anchors[p]['id'] for p in lost], 'inferential_test_performed': False}
    secondary = {'policy': 'nominal descriptive; no endpoint substitution', 'binding_mixed_minus_lora': {
        name: exact_pair(maps['lora']['binding'][name], maps['mixed']['binding'][name]) for name in maps['source']['binding']},
        'unselected_arithmetic_anchors': {role + '_minus_source': exact_pair(source_anchors, maps[role]['arithmetic']['strict_anchor'])
            for role in ('lora', 'mixed')},
        'arithmetic_mixed_minus_lora': {name: exact_pair(maps['lora']['arithmetic'][name], maps['mixed']['arithmetic'][name])
                                      for name in maps['source']['arithmetic']}}
    actual = {**{'binding_all16_' + role: sum(maps[role]['binding']['canonical_complete_exact/all16'].values()) for role in ROLES},
              'source_strict_anchor_count': len(cohort),
              **{'source_conditioned_strict_retention_fraction_' + role: retention['models'][role]['retained_fraction'] for role in ('lora', 'mixed')}}
    predictions = {key: {'observed': actual[key], 'registered_band': band,
                         'verdict': 'UNAVAILABLE_EMPTY_REFERENCE_COHORT' if actual[key] is None else
                         'PASS' if band[0] <= actual[key] <= band[1] else 'FAILED_PREDICTION'}
                   for key, band in criteria['prediction_bands'].items()}
    return {'primary': primary, 'source_conditioned_retention_descriptive': retention, 'secondary': secondary,
            'predictions': predictions, 'endpoints': endpoints, 'parent_maps': maps}


def build_report(root):
    no_model_imports()
    root = c.absolute(root, directory=True)
    freeze = verify_frozen(root)
    plan_path, grant_path = root / 'PLAN.json', root / 'GRANT.json'
    plan = c.validate_plan(c.read(plan_path)); grant = c.read(grant_path)
    c.validate_grant(plan_path, grant)
    c.phase_guard(plan, grant)
    c.require(Path(plan['output_root']).resolve() == root / 'eval' and
              Path(plan['input_path']).resolve() == root / 'candidate-v1/DIAGNOSTIC_INPUTS.jsonl', 'plan roots differ from diagnostic record')
    battery = c.read(root / 'battery/TERMINAL.json')
    c.require(battery.get('status') == 'COMPLETE_THREE_MODELS' and
              type(battery.get('session_elapsed_seconds')) in (int, float) and
              math.isfinite(battery['session_elapsed_seconds']) and 0 <= battery['session_elapsed_seconds'] <= 7200,
              'three-model battery is incomplete or exceeds its registered phase envelope')
    c.require(not (Path(plan['output_root']) / 'CONTROLLER_LEASE.json').exists(), 'controller is still active or ownership is unsettled')
    inputs = c.validate_inputs(c.read_jsonl(plan['input_path']))
    gold = validate_data(inputs, c.read_jsonl(root / 'candidate-v1/GOLD.jsonl'), c.read(root / 'candidate-v1/METADATA.json'))
    criteria = c.read(root / 'CRITERIA.json')
    validate_criteria(criteria)
    source = Path(plan['source_dir']); tokenizer = c.read(source / 'tokenizer.json'); config = c.read(source / 'tokenizer_config.json')
    decoder = ByteLevelDecoder(tokenizer, config); eos = expected_eos(source, tokenizer, config)
    generations, custody = {}, {}
    for role in ROLES:
        generations[role], custody[role] = verify_worker(plan, plan_path, grant_path, role, inputs, decoder, eos)
    costs = c.attempt_costs(plan, grant_path)
    expected_slots = {c.slot_name(role, attempt) for role in ROLES for attempt in range(1, custody[role]['attempt'] + 1)}
    c.require({v['slot'] for v in costs['attempts']} == expected_slots and costs['totals']['retries'] <= 1,
              'unexpected attempt/reservation inventory')
    c.require([item['model_role'] for item in battery['attempts']] == list(ROLES) and
              all(item['attempt'] == custody[item['model_role']]['attempt'] == 1 and
                  type(item['returncode']) is int and item['returncode'] == 0 for item in battery['attempts']) and
              battery['final_binding_sha256'] == {role: custody[role]['files_sha256']['DIAGNOSTIC_BINDING.json'] for role in ROLES} and
              battery['cost_accounting'] == costs, 'battery does not bind these exact three successful attempts and charges')
    c.require(all(0 <= item['worker_seconds_charged'] <= 900 for item in costs['attempts']), 'per-attempt worker budget exceeded')
    before_scoring = {str(path): c.sha(path) for path in (root / 'PRELAUNCH_BINDING.json', root / 'battery/TERMINAL.json')}
    before_scoring.update({str(Path(plan['output_root']) / ('RESERVATION_' + item['slot'] + '.json')):
                          item['reservation_sha256'] for item in costs['attempts']})
    for evidence in custody.values():
        before_scoring.update({str(Path(evidence['directory']) / name): pin for name, pin in evidence['files_sha256'].items()})
    # All three completed workers and every comparability gate precede endpoints.
    typed = load_typed_scorer()
    endpoints, maps = {}, {}
    for role in ROLES:
        endpoints[role], maps[role] = score_rows(inputs, gold, generations[role]['predictions'], typed)
    scientific = analyze(endpoints, maps, gold, criteria)
    c.require(verify_frozen(root) == freeze and all(c.sha(path) == pin for path, pin in before_scoring.items()) and
              c.sha(Path(arithmetic.FROZEN_SCORER['path'])) == arithmetic.FROZEN_SCORER['sha256'], 'evidence changed during offline scoring')
    c.phase_guard(plan, grant)
    no_model_imports()
    return {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_RESULTS', 'experiment': c.EXPERIMENT,
            'status': 'COMPLETE_COMPARABLE', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'prelaunch_binding_sha256': c.sha(root / 'PRELAUNCH_BINDING.json'), 'criteria_sha256': c.sha(root / 'CRITERIA.json'),
            'battery_terminal_sha256': c.sha(root / 'battery/TERMINAL.json'), 'scorer_sha256': c.sha(HERE / 'score.py'),
            'typed_scorer': arithmetic.FROZEN_SCORER, 'custody': custody, 'all_attempt_costs': costs, **scientific,
            'goal_success_redefined': False, 'legacy_acquisition_comparisons_performed': False,
            'legacy_acquisition_floor_changed': False, 'quality_claim': False, 'promotion_permitted': False,
            'independent_confirmation': False, 'training_permission': False, 'model_calls': 0,
            'limitations': ['Exploratory diagnostic; previous failed criteria remain failed.',
                'Parent worlds are independent units; related views and fixed domains are not independent samples.',
                'Relative retention is not preservation; nonsignificance is not equivalence.',
                'Controlled prompt sensitivity does not isolate training data, schedule, update count, or norm causes.',
                'Reserved coverage remains unverified; human review NOT_PERFORMED.']}


def markdown(result):
    if result['status'] != 'COMPLETE_COMPARABLE':
        return '# BINDING-DIAGNOSTIC-001\n\n**BLOCKED_ON_COMPARABILITY**\n\n' + result['reason'] + '\n\nNo scientific endpoints emitted.\n'
    lines = ['# BINDING-DIAGNOSTIC-001', '', 'Status: **COMPLETE_COMPARABLE**. Exploratory diagnostic only; no promotion or goal-completion claim.', '',
             '| Model | All16 binding /32 | Strict anchors /96 | Source-anchor retained | False UNKNOWN /256 |',
             '|---|---:|---:|---:|---:|']
    cohort = result['source_conditioned_retention_descriptive']
    for role in ROLES:
        endpoint = result['endpoints'][role]
        lines.append(f"| {role} | {endpoint['parent_counts']['binding']['canonical_complete_exact/all16']} | "
                     f"{endpoint['parent_counts']['arithmetic']['strict_anchor']} | {cohort['models'][role]['retained']}/{cohort['n']} | "
                     f"{endpoint['totals']['binding']['false_exact_unknown']} |")
    lines += ['', '| Primary: mixture minus LoRA | N parents | net | nd | Exact p | Holm(2) p | Nominal exact net threshold at observed nd |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, item in result['primary'].items():
        lines.append(f"| {name} | {item['n']} | {item['net']} | {item['nd']} | {item['exact_p']} | {item['holm_adjusted_p']} | "
                     f"{item.get('minimum_significant_abs_net_at_observed_nd')} |")
    lines += ['', 'Binding large positive diagnostic difference: **' + str(result['primary']['binding']['large_positive_diagnostic_difference']) + '**.',
              'Arithmetic reference status: **' + result['primary']['arithmetic']['reference_cohort_status'] + '**; large relative difference: **' +
              str(result['primary']['arithmetic']['large_relative_diagnostic_difference']) + '**; direction: **' + result['primary']['arithmetic']['direction'] + '**.', '',
              'Source-conditioned retention and loss IDs are descriptive; no source-versus-candidate p-values are attached to that selected cohort.', '',
              'Exact thresholds condition on observed discordance; they are not prospective power calculations. '
              'An empty arithmetic cohort contributes only a p=1 multiplicity placeholder and has no observed test.', '',
              '| Prediction | Observed | Registered band | Verdict |', '|---|---:|---|---|']
    for name, item in result['predictions'].items():
        lines.append(f"| {name} | {item['observed']} | {item['registered_band']} | {item['verdict']} |")
    lines += ['', 'Every source-known strict anchor loss (irrespective of significance):', '']
    for role in ('lora', 'mixed'):
        lines.append('- ' + role + ': ' + (', '.join(cohort['models'][role]['lost_anchor_ids']) or 'None'))
    lines += ['', *['- ' + text for text in result['limitations']], '',
              'RESULTS.json retains all parent maps, row scores, exact thresholds, descriptive secondary comparisons, loss IDs, and custody hashes.', '']
    return '\n'.join(lines)


def run(root):
    root = c.absolute(root, directory=True)
    c.require(not (root / 'RESULTS.json').exists() and not (root / 'RESULTS.md').exists(), 'results already exist; no replacement')
    try:
        result = build_report(root)
    except Exception as exc:
        result = {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_RESULTS', 'status': 'BLOCKED_ON_COMPARABILITY',
                  'error_type': type(exc).__name__, 'reason': str(exc), 'scientific_endpoints_emitted': False,
                  'quality_claim': False, 'promotion_permitted': False, 'training_permission': False, 'model_calls': 0}
    c.write_new(root / 'RESULTS.json', result)
    with (root / 'RESULTS.md').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(markdown(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    result = run(args.root)
    print({'status': result['status'], 'results': str(args.root / 'RESULTS.json')})
    return 0 if result['status'] == 'COMPLETE_COMPARABLE' else 2


if __name__ == '__main__':
    raise SystemExit(main())
