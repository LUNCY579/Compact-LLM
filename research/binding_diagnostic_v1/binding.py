"""Deterministic matched binding facts with an independent policy-text decoder."""
from collections import Counter
import hashlib
import itertools
import json
import random
import re

DOMAINS = {
    'glider_hangar': {
        'roles': ('wing inspector', 'tether steward', 'launch marshal', 'weather scribe', 'hangar runner', 'radio keeper'),
        'duties': ('checking the sailcloth', 'coiling the spare line', 'counting departing gliders', 'logging wind readings', 'carrying wheel chocks', 'testing the headset'),
    },
    'orchard_press': {
        'roles': ('pomace attendant', 'vat measurer', 'crate steward', 'press watcher', 'bottle sorter', 'barrel keeper'),
        'duties': ('removing apple pulp', 'measuring the juice level', 'stacking empty crates', 'watching the press gauge', 'sorting clean bottles', 'checking barrel seals'),
    },
}
SYSTEM = ('Resolve the requested person strictly from this record. The people list alone cannot fill an unstated post. '
          'A declared empty post has nobody. Emit one full roster name when its assignment follows from the text; '
          'emit UNKNOWN otherwise. Append no explanation.')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def parse_policy_input(prompt):
    """Parse stated assignments and duties only; never inspect row metadata or gold."""
    if not isinstance(prompt, list) or len(prompt) != 2 or prompt[0] != {'role': 'system', 'content': SYSTEM}:
        raise ValueError('unexpected binding policy')
    if prompt[1].get('role') != 'user':
        raise ValueError('missing user record')
    lines = prompt[1]['content'].splitlines()
    roster_line = next(line for line in lines if line.startswith('People present: '))
    roster = roster_line.removeprefix('People present: ').removesuffix('.').split('; ')
    if len(roster) != 6 or len(set(roster)) != 6:
        raise ValueError('invalid roster')
    assignments, duties = {}, {}
    query = None
    template = None
    for line in lines:
        match = re.fullmatch(r'The (.+) is (.+)\.', line)
        if match:
            role, person = match.groups()
            if person not in roster or role in assignments:
                raise ValueError('invalid or duplicate assignment')
            assignments[role] = person
            template = 0
            continue
        match = re.fullmatch(r'The list places (.+) in the (.+) post\.', line)
        if match:
            person, role = match.groups()
            if person not in roster or role in assignments:
                raise ValueError('invalid or duplicate assignment')
            assignments[role] = person
            template = 1
            continue
        match = re.fullmatch(r'The (.+) has no occupant\.', line)
        if not match:
            match = re.fullmatch(r'The list marks the (.+) post as vacant\.', line)
        if match:
            role = match[1]
            if role in assignments:
                raise ValueError('conflicting vacancy')
            assignments[role] = None
            continue
        match = re.fullmatch(r'The task of (.+) belongs to the (.+)\.', line)
        if match:
            duty, role = match.groups()
            if duty in duties:
                raise ValueError('duplicate duty')
            duties[duty] = role
            continue
        match = re.fullmatch(r'The (.+) post handles (.+)\.', line)
        if match:
            role, duty = match.groups()
            if duty in duties:
                raise ValueError('duplicate duty')
            duties[duty] = role
            continue
        for pattern in (r'Who performs (.+)\?', r'Who is responsible for (.+)\?'):
            match = re.fullmatch(pattern, line)
            if match:
                if query is not None:
                    raise ValueError('duplicate query')
                query = match[1]
        if not (line.startswith('Record for ') or line.startswith('People present: ') or line.startswith('Who ')):
            raise ValueError('unparsed policy fact: ' + line)
    if query not in duties or len(duties) != 6 or len(set(duties.values())) != 6:
        raise ValueError('invalid task mapping')
    role = duties[query]
    state = 'missing' if role not in assignments else 'vacant' if assignments[role] is None else 'occupied'
    return {'reference': assignments.get(role) or 'UNKNOWN', 'state': state,
            'queried_role': role, 'queried_duty': query, 'roster': roster, 'template': template,
            'assignments': assignments, 'duties': duties}


def build_rows(seed=31):
    rng = random.Random(seed)
    starts = ('Eln', 'Zav', 'Tir', 'Ves', 'Nal', 'Orv', 'Ked', 'Ruv', 'Daz', 'Fal', 'Yen', 'Pol', 'Siv', 'Mir', 'Ulv', 'Bex')
    ends = ('aro', 'elin', 'avo', 'urin', 'essa', 'orin', 'iva', 'alen', 'umar', 'enia', 'evan', 'ovia', 'erin', 'avan', 'ira', 'olan')
    first = [a + b for a, b in itertools.product(starts, ends)]
    last = [a + b for a, b in itertools.product(('Vos', 'Ker', 'Del', 'Zan', 'Til', 'Mav', 'Bor', 'Fel'),
                                               ('kel', 'var', 'din', 'met', 'rin', 'vek', 'lan', 'dor'))]
    people = rng.sample([a + ' ' + b for a, b in itertools.product(first, last)], 32 * 6)
    inputs, gold, metadata = [], [], []
    for parent_index in range(32):
        names = people[parent_index * 6:(parent_index + 1) * 6]
        order = rng.sample(range(6), 6)
        assignment_order = rng.sample(range(4), 4)
        duty_first = parent_index % 2 == 0
        parent = 'BDIAG-B-' + digest([seed, parent_index, names])[:24]
        metadata.append({'parent_id': parent, 'section': 'binding', 'parent_index': parent_index,
                         'names': names, 'duty_order': order, 'assignment_order': assignment_order,
                         'duty_first': duty_first, 'related_views': 16})
        for domain, values in DOMAINS.items():
            roles, duties = values['roles'], values['duties']
            for template in (0, 1):
                for pair, target in (('missing', 0), ('vacant', 1)):
                    for occupied in (True, False):
                        state = 'occupied' if occupied else pair
                        facts = []
                        for index in assignment_order:
                            if index == target and not occupied:
                                if pair == 'vacant':
                                    facts.append(f'The {roles[index]} has no occupant.' if template == 0 else
                                                 f'The list marks the {roles[index]} post as vacant.')
                            else:
                                facts.append(f'The {roles[index]} is {names[index]}.' if template == 0 else
                                             f'The list places {names[index]} in the {roles[index]} post.')
                        tasks = [(f'The task of {duties[index]} belongs to the {roles[index]}.' if template == 0 else
                                  f'The {roles[index]} post handles {duties[index]}.') for index in order]
                        body = tasks + facts if duty_first else facts + tasks
                        question = f'Who performs {duties[target]}?' if template == 0 else f'Who is responsible for {duties[target]}?'
                        user = '\n'.join([f'Record for {domain.replace("_", " ")}.',
                            'People present: ' + '; '.join(names) + '.', *body, question])
                        prompt = [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': user}]
                        identifier = parent + ':' + digest([domain, template, pair, state])[:16]
                        reference = names[target] if occupied else 'UNKNOWN'
                        inputs.append({'id': identifier, 'parent_id': parent, 'prompt': prompt})
                        gold.append({'id': identifier, 'parent_id': parent, 'section': 'binding', 'domain': domain,
                                     'template': template, 'pair': pair, 'state': state, 'answerable': occupied,
                                     'reference': reference, 'duty_first': duty_first})
    bundle = {'schema': 1, 'record': 'BINDING_DIAGNOSTIC_FACT_PAIRS', 'seed': seed,
              'policy_inputs': inputs, 'gold': gold, 'metadata': metadata}
    validate_rows(bundle)
    return bundle


def validate_rows(bundle):
    inputs, gold = bundle['policy_inputs'], bundle['gold']
    if len(inputs) != 512 or len(gold) != 512 or len({row['id'] for row in inputs}) != 512:
        raise ValueError('requires 512 unique binding views')
    paired = {}
    counts = Counter()
    for row, target in zip(inputs, gold):
        if row['id'] != target['id'] or row['parent_id'] != target['parent_id']:
            raise ValueError('gold joins differ')
        actual = parse_policy_input(row['prompt'])
        if actual['reference'] != target['reference'] or actual['state'] != target['state']:
            raise ValueError('independent text oracle differs')
        key = (row['parent_id'], target['domain'], target['template'], target['pair'])
        paired.setdefault(key, {})[target['answerable']] = row['prompt'][1]['content'].splitlines()
        counts[row['parent_id']] += 1
    if len(counts) != 32 or set(counts.values()) != {16} or len(paired) != 256:
        raise ValueError('incorrect parent or matched-pair structure')
    for key, states in paired.items():
        positive, negative = states[True], states[False]
        if positive[-1] != negative[-1]:
            raise ValueError('counterfactual query changed')
        removed, added = set(positive) - set(negative), set(negative) - set(positive)
        if len(removed) != 1 or len(added) != (0 if key[-1] == 'missing' else 1):
            raise ValueError('more than the decisive assignment changed')
    return bundle
