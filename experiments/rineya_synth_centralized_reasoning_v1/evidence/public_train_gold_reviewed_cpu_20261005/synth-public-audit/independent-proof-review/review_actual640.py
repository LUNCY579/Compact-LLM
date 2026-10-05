"""Third implementation: public parsing, memoized references and subset DP.

Reads only explicitly approved TRAIN files and external CPU receipts. It never
imports SYNTH/RDD, opens held-out banks, constructs tokenizers or loads models.
"""
from collections import Counter
from fractions import Fraction
import hashlib
import json
from math import comb
from pathlib import Path
import subprocess

SOURCE = Path('C:/Users/clclc/BusinessProject/ChatBubble-rineya-synth-centralized-reasoning-v1')
BUNDLE = SOURCE / 'experiments/rineya_synth_centralized_reasoning_v1/data/crossed_source2_v1_20261003'
AUDIT = Path(__file__).resolve().parent.parent
EXPECTED_HEAD = '3cc7f5e69f05b5cee07b52d4eb30c668b2c2808e'
PINS = {
    'candidate_records.jsonl': '90c151edb668416935608912a0d21c0425dbbd9f37a81a1796d4132dcee9f350',
    'student_pool_M1.jsonl': '5e3ab59b30ea07ae29b89c712b9eb39de1392cd9cb62c92418858c1aa003e1b9',
    'student_pool_M2.jsonl': '8fde7abe77b5479288a913034f110c5197c6a5764281b294aea6576fd7123752',
    'enrollment.json': '2730f96e1e7b9b23bab6f5bfc869400d16fd42eca2071d004c68aacd2bb1defc',
    'plan.json': 'a7de5c99f772ece147b08249c332c8d3448e9747a8c2df64dcd0375ff36534d7',
    'schedule_result.json': '2664eabfa4cfb9c4e28742cb58f4c6649b24eb80a16df3d5f2cf1a195a058744',
    'training_occurrences.json': '9fb20d227e4b72d30dfe44be47e0764f9939544f7de34992c6ebce7a0b5d39dc',
}

def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(',', ':'), allow_nan=False)

def digest(value):
    return hashlib.sha256(canonical(value).encode('ascii')).hexdigest()

def unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        assert key not in result, 'duplicate JSON key'
        result[key] = value
    return result

def parse(raw):
    return json.loads(raw, object_pairs_hook=unique_pairs,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))

def public_problem(messages):
    assert len(messages) == 2 and messages[0]['role'] == 'system' and messages[1]['role'] == 'user'
    text = messages[1]['content']
    lines = text.splitlines()
    if lines[-1] == 'END_SPEC':
        assert lines[4] == 'SPEC' and len(lines) == 7
        problem = parse(lines[5])
    else:
        assert lines[4] == 'DATA_FIELDS' and lines[-1] == 'END_DATA_FIELDS'
        problem = {}
        for line in lines[5:-1]:
            key, value = line.split(' = ', 1)
            assert key not in problem
            problem[key] = parse(value)
    assert set(problem) in ({'initial', 'lookup'}, {'initial', 'selection'})
    assert all(type(value) is int for value in problem['initial'])
    assert 'There are no updates, filtering, or precedence rules.' in text
    if 'lookup' in problem:
        assert 'Resolve EVERY requested alias through links until an integer item index is reached.' in text
        assert 'Return ALL resolved items, each exactly once.' in text
    else:
        assert 'All items are eligible; no reference lookup is needed.' in text
        assert 'Choose exactly k distinct eligible item indices whose INITIAL values sum to target.' in text
    return problem

def solve(problem):
    if 'lookup' in problem:
        memo, active = {}, set()
        links = problem['lookup']['links']
        def resolve(alias):
            if type(alias) is int:
                assert 0 <= alias < len(problem['initial'])
                return alias
            if alias in memo:
                return memo[alias]
            if alias in active or alias not in links:
                return None
            active.add(alias)
            endpoint = resolve(links[alias])
            active.remove(alias)
            memo[alias] = endpoint
            return endpoint
        endpoints = [resolve(alias) for alias in problem['lookup']['requested']]
        return () if None in endpoints else (tuple(sorted(set(endpoints))),)
    # Indexed dynamic program retaining all exact-cardinality certificates.
    k, target = problem['selection']['k'], problem['selection']['target']
    states = {(0, 0): {()}}
    for index, value in enumerate(problem['initial']):
        updated = {state: set(paths) for state, paths in states.items()}
        for (size, total), paths in states.items():
            if size < k:
                updated.setdefault((size + 1, total + value), set()).update(path + (index,) for path in paths)
        states = updated
    return tuple(sorted(states.get((k, target), ())))

def accept(target, certificates, n):
    response = parse(target)
    if set(response) != {'answer'}:
        return False
    answer = response['answer']
    if type(answer) is str:
        return answer == 'UNKNOWN' and not certificates
    return (type(answer) is list and bool(answer) and all(type(v) is int and 0 <= v < n for v in answer)
            and len(set(answer)) == len(answer) and tuple(sorted(answer)) in certificates)

def identity():
    def git(*args):
        return subprocess.check_output(['git', '-C', str(SOURCE), *args], text=True, timeout=15).strip()
    value = {'head': git('rev-parse', 'HEAD'), 'branch': git('branch', '--show-current'), 'status': git('status', '--porcelain')}
    assert value['head'] == EXPECTED_HEAD and value['branch'] == 'research/rineya-synth-centralized-reasoning-v1'
    return value

def main():
    before = identity()
    files = {}
    for name, pin in PINS.items():
        raw = (BUNDLE / name).read_bytes()
        assert len(raw) <= 32 * 1024**2 and hashlib.sha256(raw).hexdigest() == pin
        files[name] = [parse(line) for line in raw.splitlines() if line.strip()] if name.endswith('.jsonl') else parse(raw)
    proof_raw = (AUDIT / 'SYNTH_ACTUAL640_PUBLIC_GOLD_AUDIT_001.json').read_bytes()
    assert hashlib.sha256(proof_raw).hexdigest() == '9fec2144b09e9dc71d1a0cec5c44a40d38e20dcc1a3671401c941227bca57657'
    proof = parse(proof_raw)
    assert proof['sha256'] == digest({k: v for k, v in proof.items() if k != 'sha256'})
    assert proof['oracle_implementation']['sha256'] == hashlib.sha256((AUDIT / 'audit_student_visible.py').read_bytes()).hexdigest()
    reported = {row['row_id']: row for row in proof['rows']}
    assert len(reported) == len(files['candidate_records.jsonl']) == 640
    assert files['enrollment.json']['plan_sha256'] == files['plan.json']['sha256']
    solved, chance, counts = {}, {}, Counter()
    for arm in ('M1', 'M2'):
        private = [row for row in files['candidate_records.jsonl'] if row['arm'] == arm]
        public = files['student_pool_' + arm + '.jsonl']
        assert len(private) == len(public) == 320
        assert sorted(row['row_id'] for row in private) == files['enrollment.json']['record_ids'][arm]
        for raw, sidecar in zip(public, private):
            # Fix public gold before consulting private labels/spec/verifier data.
            problem = public_problem(raw['messages'])
            certificates = solve(problem)
            skill = 'R' if 'lookup' in problem else 'S'
            assert certificates
            assert set(raw) == {'messages', 'target'} and raw['messages'] == sidecar['messages'] and raw['target'] == sidecar['target']
            assert problem == sidecar['spec'] and skill == sidecar['skill'] and sidecar['split'] == 'train'
            assert accept(raw['target'], certificates, len(problem['initial']))
            row_id = sidecar['row_id']; row = reported[row_id]
            assert row['certificate_set_sha256'] == digest(certificates) and row['all_certificate_count'] == len(certificates)
            assert row['public_messages_sha256'] == digest(raw['messages']) and row['provided_target_sha256'] == digest(raw['target'])
            assert row['arm'] == arm and row['parent_id'] == sidecar['parent_id'] and row['status'] == 'PASS_ACTUAL_PUBLIC_TARGET'
            solved[row_id] = skill
            counts['rows'] += 1; counts[arm + '/' + skill] += 1; counts['certificates'] += len(certificates)
            counts['nonunique_rows'] += len(certificates) > 1
            for answer in certificates:
                assert accept(json.dumps({'answer': list(reversed(answer))}), certificates, len(problem['initial']))
            if skill == 'S':
                chance[row_id] = Fraction(len(certificates), comb(len(problem['initial']), problem['selection']['k']))
    assert counts['rows'] == 640 and counts['certificates'] == 824 and counts['nonunique_rows'] == 68
    assert set(row['parent_id'] for row in files['candidate_records.jsonl']) == set(files['enrollment.json']['parents'])
    assert len(files['enrollment.json']['parents']) == 20
    schedule, occurrences = files['schedule_result.json'], files['training_occurrences.json']
    assert schedule['schedules'] == occurrences['row_ids'] and schedule['sha256'] == occurrences['schedule_sha256']
    expected = {'M1': (Fraction(4387, 138600), Fraction(16399, 602910)), 'M2': (Fraction(13331, 369600), Fraction(75629, 2411640))}
    probabilities = {}
    for arm in ('M1', 'M2'):
        rows = [row for row in files['candidate_records.jsonl'] if row['arm'] == arm]
        ids = schedule['schedules'][arm]
        assert len(ids) == 540 and all(row_id in {r['row_id'] for r in rows} for row_id in ids)
        selected = [r['row_id'] for r in rows if solved[r['row_id']] == 'S']
        occurrences_s = [row_id for row_id in ids if solved[row_id] == 'S']
        assert len(selected) == 160 and len(occurrences_s) == 348
        pool_rate = sum((chance[row_id] for row_id in selected), Fraction()) / 160
        scheduled_rate = sum((chance[row_id] for row_id in occurrences_s), Fraction()) / 348
        assert (pool_rate, scheduled_rate) == expected[arm]
        probabilities[arm] = {'pool_fraction': str(pool_rate), 'pool_percent': float(pool_rate * 100),
                              'schedule_fraction': str(scheduled_rate), 'schedule_percent': float(scheduled_rate * 100)}
    for name, pin in PINS.items():
        assert hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == pin
    after = identity()
    assert after['head'] == before['head'] and after['branch'] == before['branch']
    result = {'schema': 'synth-third-implementation-public-train-proof-review-v1', 'status': 'PASS_ACTUAL640_DP_AND_MEMOIZED_REFERENCE_RECHECK',
              'source_before': before, 'source_after': after,
              'owner_local_changes_are_recorded_not_altered_or_admitted': True,
              'approved_train_file_sha256': PINS, 'reviewed_audit_sha256': hashlib.sha256(proof_raw).hexdigest(),
              'implementation_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'counts': dict(counts),
              'exact_uniform_k_subset_baselines': probabilities, 'SYNTH_or_RDD_imports': [],
              'no_native_models_tokenizers_or_optimizer': True, 'DEV_protected_confirmation_bank_reads': 0,
              'native_quality_retention_or_readiness_established': False,
              'interpretation': 'Confirms these approved TRAIN targets and certificate density only; no neural improvement or isolated diversity-causality claim.'}
    result['sha256'] = digest(result)
    output = Path(__file__).with_name('SYNTH_PUBLIC640_INDEPENDENT_PROOF_REVIEW_001.json')
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(result, stream, sort_keys=True, indent=2); stream.write('\n')
    print(json.dumps({'receipt': str(output), 'file_sha256': hashlib.sha256(output.read_bytes()).hexdigest(), 'counts': dict(counts), 'probabilities': probabilities}, indent=2))

if __name__ == '__main__':
    main()
