"""Prospective visible development worlds with decisive rules and joint tasks.

No model calls; no sealed or confirmatory split. Joint train rows are available
only to a separately declared supervised positive-control arm, not component-only
composition training. Gold and audit labels never enter prompts.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

FAMILIES = ('ledger_replay', 'prerequisite_plan', 'register_machine')
RULES = {
    'ledger_replay': 'If status is not posted, return before unchanged. Otherwise add increases account by amount; scale multiplies account by amount; transfer simultaneously subtracts amount from account and adds amount to other. Preserve every other balance.',
    'prerequisite_plan': 'The state is an integer mask from 0 to 255. An action is applicable exactly when every pre_on bit is set and every pre_off bit is clear in before. If not applicable, return before unchanged. If applicable, clear all delete bits first, then set all add bits. Restrict the result to 8 bits.',
    'register_machine': 'Registers are integers modulo 97. add increases register by value. mix increases register by twice other plus value. swap exchanges register and other simultaneously. conditional increases register by value if other is greater than threshold, otherwise decreases register by value. Reduce updated registers modulo 97 and preserve others.',
}


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canon(value).encode()).hexdigest()


def transition(family, before, event):
    """Exact integer reference semantics declared in RULES, with no model scorer."""
    if family == 'prerequisite_plan':
        if type(before) is not int or not 0 <= before <= 255:
            raise ValueError('before must be an 8-bit integer')
        if any(type(event[k]) is not int or not 0 <= event[k] <= 255 for k in ['pre_on', 'pre_off', 'add', 'delete']):
            raise ValueError('action masks must be 8-bit integers')
        applicable = before & event['pre_on'] == event['pre_on'] and before & event['pre_off'] == 0
        return ((before & ~event['delete']) | event['add']) & 255 if applicable else before
    state = dict(before)
    if any(type(v) is not int for v in state.values()):
        raise ValueError('states must contain integers')
    if family == 'ledger_replay':
        account, other, amount = event['account'], event['other'], event['amount']
        if account == other or account not in state or other not in state or type(amount) is not int:
            raise ValueError('invalid ledger operands')
        if event['status'] != 'posted':
            return state
        if event['kind'] == 'add':
            state[account] += amount
        elif event['kind'] == 'scale':
            state[account] *= amount
        elif event['kind'] == 'transfer':
            state[account], state[other] = before[account] - amount, before[other] + amount
        else:
            raise ValueError('unknown ledger operation')
    elif family == 'register_machine':
        register, other, value, threshold = event['register'], event['other'], event['value'], event['threshold']
        if register == other or register not in state or other not in state or type(value) is not int or type(threshold) is not int:
            raise ValueError('invalid register operands')
        if event['kind'] == 'swap':
            state[register], state[other] = before[other] % 97, before[register] % 97
        elif event['kind'] == 'add':
            state[register] = (before[register] + value) % 97
        elif event['kind'] == 'mix':
            state[register] = (before[register] + 2 * before[other] + value) % 97
        elif event['kind'] == 'conditional':
            state[register] = (before[register] + (value if before[other] > threshold else -value)) % 97
        else:
            raise ValueError('unknown register operation')
    else:
        raise ValueError('unknown family')
    return state


def joint_answer(family, before, events):
    if len(events) != 2:
        raise ValueError('exactly two possible events are required')
    states = [transition(family, before, event) for event in events]
    return canon(states[0]) if states[0] == states[1] else 'UNKNOWN'


def typed_shape(value):
    if isinstance(value, dict):
        return {key: typed_shape(item) for key, item in sorted(value.items())}
    if isinstance(value, list):
        return [typed_shape(item) for item in value]
    return type(value).__name__


def make_world(rng, family):
    if family not in FAMILIES:
        raise ValueError('unknown family')
    if family == 'ledger_replay':
        v = rng.randrange(3, 14)
        multipliers = rng.sample(list(range(2 + rng.randrange(2), 13, 2)), 2)
        multiplier, other_multiplier = multipliers
        before = dict(zip(rng.sample(['A', 'B', 'C', 'D'], 4), [v, rng.randrange(-20, 20), rng.randrange(-20, 20), rng.randrange(-20, 20)]))
        account, other = list(before)[:2]
        first = dict(kind='add', status='posted', account=account, other=other, amount=v * (multiplier - 1))
        same = {**first, 'kind': 'scale', 'amount': multiplier}
        different = {**same, 'amount': other_multiplier}
        direct = [(before, first, 'add'), (before, same, 'scale'), (before, different, 'scale_counterfactual'),
                  (before, {**first, 'status': 'void'}, 'status_false'),
                  (before, {**first, 'kind': 'scale'}, 'dispatch_counterfactual'),
                  (before, {**first, 'kind': 'transfer'}, 'simultaneous_transfer')]
    elif family == 'prerequisite_plan':
        target, required, forbidden, neutral_set, neutral_clear = [1 << bit for bit in rng.sample(range(8), 5)]
        before = (rng.randrange(256) | required | neutral_set) & ~target & ~forbidden & ~neutral_clear
        first = dict(kind='action', pre_on=required, pre_off=forbidden, add=target, delete=target)
        # Both alternate additions have exactly two set bits. Which extra bit is
        # already present determines sufficiency; no array/type/cardinality tell.
        same = {**first, 'add': target | neutral_set}
        different = {**first, 'add': target | neutral_clear}
        direct = [(before, first, 'overlapping_delete_add'), (before, same, 'idempotent_extra_bit'),
                  (before, different, 'new_extra_bit'), (before & ~required, first, 'pre_on_false'),
                  (before | forbidden, first, 'pre_off_false')]
    else:
        register, other, spare = rng.sample(['X', 'Y', 'Z'], 3)
        before = {register: rng.randrange(80, 97), other: rng.randrange(2, 15), spare: rng.randrange(97)}
        value = rng.randrange(20, 35)
        first = dict(kind='mix', register=register, other=other, value=value, threshold=before[other])
        equivalent_shift = 97 * rng.choice([-2, -1, 0, 1, 2, 3])
        same = {**first, 'kind': 'add', 'value': 2 * before[other] + value + equivalent_shift}
        different = {**same, 'value': same['value'] + rng.choice([-12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12])}
        direct = [(before, first, 'mix_with_modulus'), (before, same, 'modulo_equivalent_add'),
                  (before, different, 'add_counterfactual'),
                  (before, {**first, 'kind': 'conditional', 'threshold': before[other] - 1}, 'conditional_true'),
                  (before, {**first, 'kind': 'conditional'}, 'conditional_false'),
                  (before, {**first, 'kind': 'swap'}, 'swap')]
    equal_pair, unequal_pair = [first, same], [first, different]
    if rng.randrange(2):
        equal_pair.reverse()
        unequal_pair.reverse()
    if joint_answer(family, before, equal_pair) == 'UNKNOWN' or joint_answer(family, before, unequal_pair) != 'UNKNOWN':
        raise AssertionError('joint construction does not determine intended sufficiency')
    if typed_shape(equal_pair) != typed_shape(unequal_pair):
        raise AssertionError('sufficiency type/shape cue')
    return before, direct, equal_pair, unequal_pair


def render_row(family, split, world, task, payload, reference, audit_label, ordinal):
    instruction = RULES[family]
    if task == 'transition':
        ending = 'Apply the event. Return only the resulting state as JSON.'
    elif task == 'boundary':
        instruction = 'Two candidate resulting states are supplied. Their order does not matter.'
        ending = 'If both candidates are exactly equal, return that state as JSON; otherwise return UNKNOWN.'
    else:
        ending = 'Exactly one of the two events occurred, but which one is unknown. Apply each possible event separately to before. If both resulting states are equal, return that state as JSON; otherwise return UNKNOWN.'
    if split == 'dev':
        ending = 'Follow the rule exactly. ' + ending
    return {'id': f'{world}-{task}-{ordinal}', 'world_id': world, 'family': family,
            'split': split, 'task': task, 'template': f'{split}-{task}',
            'reference': reference, 'audit_label': audit_label,
            'semantic_key': digest([family, task, payload]),
            'prompt': [{'role': 'system', 'content': instruction + ' ' + ending},
                       {'role': 'user', 'content': canon(payload)}]}


def build(seed=20260919, train_worlds=16, dev_worlds=8):
    if train_worlds < 1 or dev_worlds < 1:
        raise ValueError('both splits require worlds')
    rng = random.Random(seed)
    output, seen, worlds = [], set(), set()
    for split, count in [('train', train_worlds), ('dev', dev_worlds)]:
        for family in FAMILIES:
            accepted, tries = 0, 0
            while accepted < count:
                tries += 1
                if tries > 10000:
                    raise ValueError('could not generate disjoint worlds in bounded attempts')
                before, direct, equal, unequal = make_world(rng, family)
                world = 'RCW-' + digest([family, before, direct, equal, unequal])[:24]
                if world in worlds:
                    continue
                rows = []
                for i, (state, event, label) in enumerate(direct):
                    rows.append(render_row(family, split, world, 'transition', {'before': state, 'event': event},
                                           canon(transition(family, state, event)), label, i))
                for i, events in enumerate([equal, unequal]):
                    answer = joint_answer(family, before, events)
                    states = [transition(family, before, e) for e in events]
                    rows.append(render_row(family, split, world, 'boundary', {'candidate_states': states}, answer, 'sufficient' if i == 0 else 'insufficient', i))
                    rows.append(render_row(family, split, world, 'joint', {'before': before, 'events': events}, answer, 'sufficient' if i == 0 else 'insufficient', i))
                keys = {r['semantic_key'] for r in rows}
                if len(keys) != len(rows) or keys & seen:
                    continue
                output.extend(rows)
                seen.update(keys)
                worlds.add(world)
                accepted += 1
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20260919)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit('refusing to overwrite a prospective bank')
    rows = build(args.seed)
    args.output_dir.mkdir(parents=True)
    paths = []
    for split in ['train', 'dev']:
        path = args.output_dir / f'{split}.jsonl'
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                if row['split'] == split:
                    stream.write(json.dumps(row, sort_keys=True) + '\n')
        paths.append(path)
    report = {'version': 2, 'seed': args.seed, 'role': 'PROSPECTIVE_VISIBLE_DEVELOPMENT', 'model_calls': 0,
              'rows': len(rows), 'worlds': len({r['world_id'] for r in rows}),
              'counts': dict(Counter(f"{r['split']}/{r['family']}/{r['task']}" for r in rows)),
              'world_and_exact_semantic_overlap': 0,
              'joint_train_policy': 'Only a separately declared joint-supervised positive control may use joint training rows.',
              'independently_sealed': False, 'qualified_by_model': False,
              'limitations': ['Rules share a generator and narrow synthetic schemas.',
                              'Dev template variation is only a prefix change, not a substantive paraphrase split.',
                              'Equal candidate-state comparison is a deliberately simple boundary component.',
                              'Further shortcut and rendered-prompt review required before model execution.',
                              'No scale, quality, continual-learning or composition success established.'],
              'builder_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}}
    (args.output_dir / 'BUILD.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
