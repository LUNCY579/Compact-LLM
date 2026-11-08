"""Prepare unapproved TRAIN-only R oracle targets; never load/train a model."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys

from experiments.skill_acquisition_native_v1 import data
from experiments.maxrl_composable_skills_v2 import oracle, corpus
from experiments.maxrl_comp002_pretraining_v1 import canary

ORDER = 'COMP-RS-R-ORACLE-SFT-ORDER-1|'
PERMUTE = 'COMP-RS-R-ORACLE-PERMUTE-1'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def write_json(path, value):
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n')


def prepare(root, out, tokenize=False):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    rows, manifests, binding = data.load_atomic_train(root, ('R',))
    ids = sorted(rows, key=lambda i: (hashlib.sha256((ORDER + i).encode()).hexdigest(), i))
    targets, sizes, difficulties = {'canonical': [], 'permuted': []}, Counter(), Counter()
    for exposure, row_id in enumerate(ids):
        row = rows[row_id]
        reference = oracle.answers(row['verifier']['spec'], limit=1)[0]
        answer = json.loads(reference)['answer']
        if type(answer) is not list or len(set(answer)) != len(answer):
            raise ValueError('R proposal requires a nonempty distinct resolved set: ' + row_id)
        # One logical exposure per row; retry does not create a new target.
        permuted = sorted(answer, key=lambda i: (digest([PERMUTE, 1301, row_id, 0, i]), i))
        options = {'canonical': reference, 'permuted': canonical({'answer': permuted}).decode()}
        sizes[len(answer)] += 1
        difficulties[row['difficulty']] += 1
        for arm, target in options.items():
            if canary.direct_strict(row, target) != 1 or corpus.reward(row, target) != 1:
                raise ValueError('Independent oracle target verification failed: ' + row_id)
            targets[arm].append({'id': row_id, 'parent_id': row['parent_id'], 'line': exposure + 1,
                'logical_exposure': 0, 'prompt': row['prompt'],
                'messages': row['prompt'] + [{'role': 'assistant', 'content': target}],
                'target': target, 'target_sha256': hashlib.sha256(target.encode()).hexdigest()})
    records = {}
    for arm, values in targets.items():
        path = out / (arm + '.messages.jsonl')
        h = hashlib.sha256()
        with path.open('xb') as stream:
            for row in values:
                raw = canonical(row) + b'\n'
                stream.write(raw)
                h.update(raw)
        records[arm] = {'path': str(path.resolve()), 'sha256': h.hexdigest(), 'bytes': path.stat().st_size,
            'rows': len(values), 'ordered_target_digest': digest([[r['id'], r['target_sha256']] for r in values])}
    census = None
    if tokenize:
        # Local pinned tokenizer only. Importing Torch for label masks does not
        # instantiate a causal model or initialize CUDA.
        from experiments.maxrl_comp002_pretraining_v1 import s0_runtime, s0_warmstart
        import torch
        if torch.cuda.is_initialized():
            raise RuntimeError('Target planning must not initialize CUDA')
        tokenizer = s0_runtime.load_tokenizer(s0_runtime.MODEL_DIR, s0_runtime.TOKENIZER_RECEIPT,
                                            asset_receipt=s0_runtime.ASSET_RECEIPT)
        census = {}
        token_lengths = {}
        for arm, values in targets.items():
            rendered, receipt = s0_warmstart.render_rows(tokenizer, values)
            token_lengths[arm] = [len(r['completion_ids']) for r in rendered]
            census[arm] = dict(receipt,
                total_prompt_tokens=sum(r['prompt_tokens'] for r in rendered),
                total_supervised_tokens=sum(len(r['completion_ids']) for r in rendered),
                total_sequence_positions=sum(r['prompt_tokens'] + len(r['completion_ids']) for r in rendered),
                max_sequence_positions=max(r['prompt_tokens'] + len(r['completion_ids']) for r in rendered),
                tokenizer_receipt_sha256=s0_runtime.TOKENIZER_RECEIPT_SHA)
        census['same_completion_length_rows'] = sum(a == b for a, b in zip(token_lengths['canonical'], token_lengths['permuted']))
        if torch.cuda.is_initialized():
            raise RuntimeError('Target planning initialized CUDA')
    result = {'schema': 'R-ORACLE-SUPERVISION-PREPARATION-1', 'approved': False,
        'training_authorized': False, 'classification': 'TRAIN-only verified labels and optional tokenizer census; not model efficacy',
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'data': binding, 'atomic_R_sha256': digest(manifests['R']), 'rows': len(ids),
        'parent_ids': len({r['parent_id'] for r in rows.values()}), 'order_rule': ORDER,
        'permutation_rule': [PERMUTE, 1301, 'row_id', 0, 'index'], 'size_counts': dict(sizes),
        'difficulty_counts': dict(difficulties),
        'valid_orderings_per_size': {n: math.factorial(n) for n in sizes},
        'targets': records, 'independent_target_checks': 2 * len(ids),
        'changed_orders': sum(a['target'] != b['target'] for a, b in zip(targets['canonical'], targets['permuted'])),
        'census': census, 'model_loads': 0, 'optimizer_steps': 0,
        'heldout_model_outputs_read': False, 'original_study_continued': False,
        'scope_note': 'Canonical oracle references are capped, not the full valid-answer population. Fixed hash-permutation targets are not exact permutation marginalization.'}
    write_json(out / 'PREPARATION.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tokenize', action='store_true')
    args = parser.parse_args()
    result = prepare(args.data, args.out, args.tokenize)
    print(json.dumps({'rows': result['rows'], 'changed_orders': result['changed_orders'],
                      'model_loads': 0, 'optimizer_steps': 0, 'receipt': str(args.out / 'PREPARATION.json')}))


if __name__ == '__main__':
    main()
