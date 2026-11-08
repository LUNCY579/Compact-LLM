"""Prepare fresh unprotected synthetic TRAIN/DEV for review, never approve it.

No tokenizer/model is loaded. Token distances and shape admission must be
checked with the selected locked tokenizer before native use. This is a small
controlled development probe, not a general capability or confirmation bank.
"""
import argparse
import hashlib
import json
from pathlib import Path


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write('\n')


def distant(split, category, index):
    tag = 'Cedar' if split == 'train' else 'Quartz'
    number = (210 if split == 'train' else 740) + index * 7
    entity, other = f'{tag}-{category}-{index}', f'{tag}-decoy-{index}'
    if category == 'lookup':
        evidence = f'The archive records {entity} with access code {number}.'
        query, target = f'Return only the access code of {entity}.', str(number)
    elif category == 'multiple_facts':
        evidence = f'{entity} has north value {number} and south value {number+3}.'
        query = f'Give {entity}\'s north and south values as north/south, with no spaces.'
        target = f'{number}/{number+3}'
    elif category == 'join':
        bridge = f'{tag}-relay-{index}'
        evidence = f'{entity} is assigned to {bridge}. The access code of {bridge} is {number}.'
        query, target = f'What access code follows the assignment of {entity}? Return only the code.', str(number)
    elif category == 'update':
        evidence = f'{entity} originally had code {number-5}. Correction: {entity}\'s current code is {number}.'
        query, target = f'Return only the current corrected code for {entity}.', str(number)
    elif category == 'identifier':
        target = f'{tag.upper()}-{number}-Z{index+1}'
        evidence = f'The exact shipment identifier of {entity} is {target}.'
        query = f'Copy the exact shipment identifier of {entity}; preserve case and hyphens.'
    elif category == 'number':
        target = f'{number}.{index+10:02d}'
        evidence = f'The recorded measurement for {entity} is {target}.'
        query = f'Copy only the measurement for {entity}, preserving the two decimal places.'
    else:
        evidence = f'Only {other} has a recorded access code: {number}. No code is recorded for {entity}.'
        query, target = f'Return {entity}\'s access code if supplied, otherwise exactly UNKNOWN.', 'UNKNOWN'
    count = 24 if split == 'train' else (24, 48, 96)[index % 3]
    filler = '\n'.join(f'Unrelated log N{i}: value {(i*13+number)%997}; closed.' for i in range(count))
    intro = ('Read the register and answer the last request.\n' if split == 'train'
             else 'Use only the following archived notes to fulfill the final request.\n')
    prompt = intro + evidence + '\n' + filler + '\nFinal request: ' + query
    return prompt, target, {
        'role': 'distant_evidence', 'category': category,
        'evidence_start_char': len(intro), 'evidence_end_char': len(intro)+len(evidence),
        'query_start_char': len(prompt)-len(query), 'distractor_entries': count,
        'required_min_evidence_to_query_tokens_exclusive': 128,
    }


def retention(split, category, index):
    n = (17 if split == 'train' else 63) + index
    label = ('Birch' if split == 'train' else 'Opal') + str(index)
    if category == 'arithmetic':
        prompt, target = f'Calculate {n} + {n+4}. Return only the integer.', str(2*n+4)
    elif category == 'short_lookup':
        prompt, target = f'Inventory: {label}={n}; Other={n+11}. Return only {label}\'s value.', str(n)
    elif category == 'missing':
        prompt = f'Record: {label} has a weight of {n}. What is its color? Return exactly UNKNOWN if absent.'
        target = 'UNKNOWN'
    elif category == 'formatting':
        prompt, target = f'Return exactly this identifier, with no additional text: {label.upper()}-{n}', f'{label.upper()}-{n}'
    else:
        prompt, target = f'Numbers: {n+9}, {n}, {n+2}. Return only the smallest number.', str(n)
    return prompt, target, {'role': 'short_retention_probe', 'category': category}


def prepare(directory):
    directory.mkdir(parents=True, exist_ok=False)
    splits, metadata = {}, {}
    for split, count in (('train', 4), ('dev', 6)):
        rows = []
        for producer, categories in (
            (distant, ('lookup', 'multiple_facts', 'join', 'update', 'identifier', 'number', 'missing')),
            (retention, ('arithmetic', 'short_lookup', 'missing', 'formatting', 'ordering')),
        ):
            for category in categories:
                for index in range(count):
                    prompt, target, design = producer(split, category, index)
                    family = f'{split}-{design["role"]}-{category}'
                    row_id = f'{family}-{index:02d}'
                    rows.append({'id': row_id, 'family_id': family, 'prompt': prompt, 'target': target})
                    metadata[row_id] = {'split': split, 'family_id': family, **design}
        path = directory / f'{split}.jsonl'
        with path.open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True)+'\n')
        splits[split] = {'path': path.name, 'sha256': file_hash(path), 'role': split, 'protected': False}
    write_json(directory/'design.json', {
        'status': 'UNREVIEWED_PROPOSAL_NOT_NATIVE_ADMISSION',
        'generator_revision': 'distant_evidence_v1', 'seed': None, 'randomness': 'none',
        'scope': '48 TRAIN and 72 DEV synthetic controlled probes; no protected/repository data',
        'review_required': ['Verify targets and missing-evidence semantics',
                            'Review semantic/template separation across splits',
                            'Measure evidence distance and token limits with exact source tokenizer',
                            'Register exposure, primary distant subset and separate retention subset'],
        'limitations': ['Short retention probes do not measure broad original-model capability',
                        'One model/configuration/seed cannot establish general recurrent learnability',
                        'Shared task templates require human leakage review'],
        'rows': metadata,
    })
    write_json(directory/'manifest.json', {
        'reviewed_for_sbhm': False,
        'purpose': 'Fresh synthetic controlled development proposal; review pending',
        'design': {'path': 'design.json', 'sha256': file_hash(directory/'design.json')},
        'evaluation_groups': {
            role: sorted({item['family_id'] for item in metadata.values()
                          if item['split'] == 'dev' and item['role'] == role})
            for role in ('distant_evidence', 'short_retention_probe')
        },
        'splits': splits,
    })


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    prepare(args.output)
    print(json.dumps({'status': 'UNREVIEWED_PROPOSAL_NOT_NATIVE_ADMISSION',
                      'output': str(args.output.resolve())}))


if __name__ == '__main__':
    main()
