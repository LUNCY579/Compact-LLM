"""Fresh visible anchor/dev worlds for multi-hop binding and numerical retention.

No model execution. All eight binding derivatives remain in one world/split.
Gold comes from relational and integer oracles, not template answer extraction.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

from .rule_bank import canon, digest

PEOPLE = {
    'anchor': ['Nora Vale', 'Theo Moss', 'Mira Lane', 'Omar Reed', 'Lena Ford', 'Ivan Pike', 'Ada Brook', 'Noah Wells', 'Iris Shaw', 'Leon Hart', 'Eva Cole', 'Finn Lake'],
    'dev': ['Rosa Glen', 'Hugo West', 'Cora Bell', 'Amir Snow', 'Sana Wood', 'Luis Hill', 'Maya Stone', 'Owen Field', 'Elin Park', 'Arlo Heath', 'Vera Dale', 'Nico Frost'],
}
ROLES = {
    'anchor': ['night porter', 'map keeper', 'camp guide', 'supply clerk', 'garden steward', 'stage manager', 'store warden', 'book restorer'],
    'dev': ['evening custodian', 'chart archivist', 'trail leader', 'stock recorder', 'grounds supervisor', 'hall coordinator', 'depot attendant', 'paper conservator'],
}
OBJECTS = ['blue folder', 'brass key', 'linen bundle', 'wooden tray', 'silver token', 'green notebook', 'copper box', 'canvas bag']
PLACES = ['north office', 'east gate', 'west room', 'south shed', 'upper hall', 'lower store', 'main desk', 'side door']
ACTIVITIES = ['{person} checked the visitor list.', '{person} attended the morning briefing.',
              '{person} inspected the posted schedule.', '{person} signed the attendance book.']
FILLER = ['The report covers a single day, and every assignment mentioned was in force throughout that day.',
          'A separate maintenance notice describes a damaged bench and lists a repair appointment for the following week.',
          'The filing instructions require original receipts to be kept with their matching copies. No receipt changes an assignment or delivery stated here.']
SYSTEM = ('Use only the supplied report. The report is partial: people may hold multiple roles, '
          'so missing assignments cannot be inferred by elimination. Return only the person\'s '
          'full name. If the report does not uniquely identify the person, return UNKNOWN.')


def binding_answer(payload):
    """Relational oracle: select delivery role, then resolve its named occupant."""
    query = payload['query']
    if query['task'] == 'direct':
        roles = {query['role']}
    else:
        roles = {role for role, obj, place in payload['actions']
                 if obj == query['object'] and place == query['place']}
    if not roles:
        return 'UNKNOWN'
    people = set()
    for role in roles:
        occupants = {person for r, person in payload['bindings'] if r == role}
        if len(occupants) != 1:
            return 'UNKNOWN'
        people.update(occupants)
    return next(iter(people)) if len(people) == 1 else 'UNKNOWN'


def render_binding(payload, split, order):
    if split == 'anchor':
        bindings = [f'{person} served as the {role}.' for role, person in payload['bindings']]
        actions = [f'The {role} delivered the {obj} to the {place}.' for role, obj, place in payload['actions']]
    else:
        bindings = [f'The {role} duties were assigned to {person}.' for role, person in payload['bindings']]
        actions = [f'The {obj} was taken to the {place} by the {role}.' for role, obj, place in payload['actions']]
    lines = bindings + actions + payload['neutral_lines']
    context = ' '.join([FILLER[0], *[lines[i] for i in order], *FILLER[1:]])
    q = payload['query']
    if q['task'] == 'direct':
        question = f"Who served as the {q['role']}?" if split == 'anchor' else f"Name the person assigned the {q['role']} duties."
    else:
        question = f"Who delivered the {q['object']} to the {q['place']}?" if split == 'anchor' else f"Which person took the {q['object']} to the {q['place']}?"
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': 'REPORT\n' + context + '\nQUESTION\n' + question}], context


def binding_world(rng, split, salience):
    people = rng.sample(PEOPLE[split], 4)
    roles = rng.sample(ROLES[split], 5)
    objects, places = rng.sample(OBJECTS, 4), rng.sample(PLACES, 4)
    counts = {'low': [1, 1, 3, 3], 'equal': [2, 2, 2, 2], 'high': [3, 3, 1, 1]}[salience]
    neutral = [ACTIVITIES[j].format(person=person) for person, count in zip(people, counts) for j in range(count - 1)]
    actions = list(zip(roles[:4], objects, places))
    order = list(range(8 + len(neutral)))
    rng.shuffle(order)
    world = 'FBW-' + digest([people, roles, objects, places, counts, order])[:24]
    result = []
    for swap in [False, True]:
        occupants = people.copy()
        if swap:
            occupants[0], occupants[1] = occupants[1], occupants[0]
        for missing in [False, True]:
            bindings = list(zip(roles[:4], occupants))
            if missing:
                bindings[0] = (roles[4], occupants[0])
            for task in ['direct', 'two_hop']:
                query = {'task': task, 'role': roles[0]} if task == 'direct' else {'task': task, 'object': objects[0], 'place': places[0]}
                payload = {'bindings': bindings, 'actions': actions, 'neutral_lines': neutral, 'query': query}
                reference = binding_answer(payload)
                prompt, context = render_binding(payload, split, order)
                measured = [context.count(person) for person in people]
                if measured != counts:
                    raise AssertionError('identity/missing variant changed mention counts')
                if (reference == 'UNKNOWN') != missing:
                    raise AssertionError('constructed answerability disagrees with relational oracle')
                result.append({'id': f'{world}-{task}-s{int(swap)}-m{int(missing)}',
                               'world_id': world, 'family': 'binding', 'task': task, 'split': split,
                               'template': f'{split}-{task}-parallel-four-bindings', 'salience': salience,
                               'answerable': not missing, 'identity_swap': swap,
                               'reference': reference, 'prompt': prompt, 'semantic_key': digest(payload),
                               'oracle_payload': payload, 'people': people, 'mention_counts': measured,
                               'context': context, 'gold_verified_by': 'binding_answer'})
    return result


def arithmetic_key(a, op, b):
    if op == '+':
        a, b = sorted([a, b])
    return f'arithmetic:{a}{op}{b}'


def numeric_world(rng, split, family):
    result = []
    if family == 'time':
        hour, delay = rng.randrange(24), rng.randrange(15, 40)
        minutes = [rng.randrange(0, 60 - delay), rng.randrange(60 - delay, 60)]
        parameters = [hour, delay, minutes]
        cases = [(hour, minute, delay) for minute in minutes]
    else:
        a, b = rng.randrange(20, 90), rng.randrange(2, 18)
        parameters = [a, b]
        cases = [(a, '+', b), (a, '-', b)]
    world = 'FNW-' + digest([family, parameters])[:24]
    for i, case in enumerate(cases):
        if family == 'time':
            hour, minute, delay = case
            end = (hour * 60 + minute + delay) % 1440
            reference = f'{end // 60:02}:{end % 60:02}'
            question = (f'A clock shows {hour:02}:{minute:02}. What time is it {delay} minutes later?'
                        if split == 'anchor' else f'Advance {hour:02}:{minute:02} by {delay} minutes and give the new clock reading.')
            system = 'Use 24-hour HH:MM only. After midnight wrap around to the next day.'
            semantic = f'time:{hour}:{minute}:{delay}'
            payload = {'hour': hour, 'minute': minute, 'delay': delay, 'carry': i == 1}
        else:
            a, op, b = case
            reference = str(a + b if op == '+' else a - b)
            question = f'Calculate {a} {op} {b}.' if split == 'anchor' else f'Give the integer result of {a} {op} {b}.'
            system = 'Return only the integer answer.'
            semantic = arithmetic_key(a, op, b)
            payload = {'a': a, 'op': op, 'b': b}
        result.append({'id': f'{world}-{i}', 'world_id': world, 'family': family, 'task': family,
                       'split': split, 'template': f'{split}-{family}', 'answerable': True,
                       'reference': reference, 'prompt': [{'role': 'system', 'content': system},
                                                         {'role': 'user', 'content': question}],
                       'semantic_key': semantic, 'oracle_payload': payload,
                       'gold_verified_by': 'integer_arithmetic'})
    return result


def build(seed=20260920, binding_worlds_per_condition=8, clock_worlds=32, arithmetic_worlds=16, excluded=None):
    rng = random.Random(seed)
    seen, output = set(excluded or []), []
    for split in ['anchor', 'dev']:
        generators = [(lambda condition=condition: binding_world(rng, split, condition), binding_worlds_per_condition)
                      for condition in ['low', 'equal', 'high']]
        generators += [(lambda: numeric_world(rng, split, 'time'), clock_worlds),
                       (lambda: numeric_world(rng, split, 'arithmetic'), arithmetic_worlds)]
        for generate, count in generators:
            accepted, attempts = 0, 0
            while accepted < count:
                attempts += 1
                if attempts > 10000:
                    raise ValueError('bounded disjoint generation exhausted')
                rows = generate()
                keys = {row['semantic_key'] for row in rows}
                if keys & seen or len(keys) != len(rows):
                    continue
                seen.update(keys)
                output.extend(rows)
                accepted += 1
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--exclude-bank', type=Path, action='append', default=[])
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('refusing to overwrite binding bank')
    exclusions = set()
    for path in args.exclude_bank:
        for row in json.loads(path.read_text(encoding='utf-8')):
            key = row['semantic_key']
            if key.startswith('arithmetic:') and '+' in key:
                a, b = map(int, key.removeprefix('arithmetic:').split('+'))
                key = arithmetic_key(a, '+', b)
            exclusions.add(key)
    rows = build(excluded=exclusions)
    args.output_dir.mkdir(parents=True)
    hashes = {}
    for split in ['anchor', 'dev']:
        path = args.output_dir / f'{split}.json'
        path.write_text(json.dumps([r for r in rows if r['split'] == split], indent=2) + '\n', encoding='utf-8')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    report = {'version': 1, 'role': 'VISIBLE_PROSPECTIVE_DEVELOPMENT', 'rows': len(rows),
              'worlds': len({r['world_id'] for r in rows}),
              'counts': dict(Counter(f"{r['split']}/{r['family']}/{r['task']}" for r in rows)),
              'excluded_semantic_keys': len(exclusions), 'semantic_overlap': 0, 'model_calls': 0,
              'source_qualified': False, 'confirmation': False,
              'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'input_sha256': {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in args.exclude_bank},
              'output_sha256': hashes,
              'limitations': ['Different templates and entity pools are still one synthetic generator.',
                              'Missing bindings are omissions, not a complete taxonomy of uncertainty.',
                              'Clock wraparound and two-hop distractors need source qualification.',
                              'Disjointness covers declared banks; it is not an exhaustive historical-contamination audit.',
                              'Source-margin selection and token-matched anchor sampling are not yet implemented.']}
    (args.output_dir / 'BUILD.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
