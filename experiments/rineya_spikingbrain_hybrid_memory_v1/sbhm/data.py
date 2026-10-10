"""Explicit train/dev only. No repository-wide discovery or protected-bank access."""
from pathlib import Path
import json
import random
from .protocol import Blocked, read_json, safe_file, sha, write_json


def validate_evaluation_groups(groups, dev_rows):
    """Optional reviewed grouping of DEV families; labels do not prove distance."""
    if groups is None:
        return None
    if not isinstance(groups, dict) or not groups:
        raise Blocked('evaluation_groups must be a nonempty group-to-family-list map')
    covered = set()
    result = {}
    for name, families in groups.items():
        if (not isinstance(name, str) or not name.strip() or not isinstance(families, list) or
                not families or any(not isinstance(family, str) or not family.strip() for family in families)):
            raise Blocked('evaluation_groups require nonempty group names and family string lists')
        if len(set(families)) != len(families) or covered.intersection(families):
            raise Blocked('evaluation_groups contain duplicate or overlapping DEV families')
        covered.update(families)
        result[name] = sorted(families)
    if covered != {row['family_id'] for row in dev_rows}:
        raise Blocked('evaluation_groups must cover every DEV family exactly once')
    return {name: result[name] for name in sorted(result)}


def load_corpus(path):
    if not path:
        raise Blocked('reviewed train/dev manifest required')
    root, manifest = Path(path).parent, read_json(path)
    if set(manifest.get('splits', {})) != {'train', 'dev'}:
        raise Blocked('only explicit train and dev splits are accepted; no final/check/test banks')
    if manifest.get('reviewed_for_sbhm') is not True:
        raise Blocked('corpus has not been marked reviewed for this experiment')
    for name in ('design', 'review'):
        if name in manifest:
            item = manifest[name]
            if not isinstance(item, dict) or set(item) != {'path', 'sha256'}:
                raise Blocked(f'{name} must lock an exact path and SHA256')
            p = safe_file(root, item['path'])
            if p.stat().st_size > 16*1024*1024 or sha(p) != item['sha256']:
                raise Blocked(f'{name} metadata exceeds bound or has changed')
    result = {}
    for split, item in manifest['splits'].items():
        if item.get('role') != split or item.get('protected') is not False:
            raise Blocked('protected/ambiguous data role')
        p = safe_file(root, item['path'])
        if p.stat().st_size > 16*1024*1024 or sha(p) != item['sha256']:
            raise Blocked('dataset exceeds bound or has changed')
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
        if not rows:
            raise Blocked('empty split')
        for row in rows:
            if set(row) != {'id', 'family_id', 'prompt', 'target'} or any(not isinstance(v, str) or not v for v in row.values()):
                raise Blocked('rows require exactly four nonempty strings: id/family_id/prompt/target')
        if len({r['id'] for r in rows}) != len(rows):
            raise Blocked('duplicate row IDs')
        result[split] = rows
    for field in ('id', 'family_id', 'prompt'):
        norm = lambda s: ' '.join(s.casefold().split())
        if {norm(r[field]) for r in result['train']} & {norm(r[field]) for r in result['dev']}:
            raise Blocked(f'train/dev overlap in {field}')
    if 'evaluation_groups' in manifest and manifest['evaluation_groups'] is None:
        raise Blocked('present evaluation_groups must be a nonempty map')
    validate_evaluation_groups(manifest.get('evaluation_groups'), result['dev'])
    return result, sha(path)


def ordered(rows, seed, steps):
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    return [rows[indices[i % len(indices)]] for i in range(steps)]


def encode(tokenizer, row, training=False):
    import torch
    text = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
    prompt = tokenizer.encode(text, add_special_tokens=False)
    if not prompt:
        raise Blocked('empty tokenized prompt')
    ids = list(prompt)
    labels = [-100]*len(prompt)
    if training:
        target = tokenizer.encode(row['target'], add_special_tokens=False)
        if tokenizer.eos_token_id is None:
            raise Blocked('source tokenizer has no EOS')
        target += [tokenizer.eos_token_id]
        ids += target
        labels += target
    return torch.tensor([ids], dtype=torch.long), torch.tensor([labels], dtype=torch.long)


def make_fixture_corpus(directory):
    """Inspectable engineering data, NOT a reviewed native efficacy corpus."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split in ('train', 'dev'):
        rows = []
        for i in range(8):
            if split == 'train':
                prompt = f'Register A{i} is {i+11}; register B{i} is {i+27}. Return only the value of A{i}.'
                target, family = str(i+11), 'single_lookup'
            else:
                prompt = f'An earlier entry says X{i}=31. A later correction says X{i}={i+51}. Return the corrected value only.'
                target, family = str(i+51), 'updated_lookup'
            rows.append({'id': f'{split}-{i}', 'family_id': family, 'prompt': prompt, 'target': target})
        p = directory/f'{split}.jsonl'
        p.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        splits[split] = {'path': p.name, 'sha256': sha(p), 'role': split, 'protected': False}
    write_json(directory/'manifest.json', {'reviewed_for_sbhm': False, 'purpose': 'engineering fixtures only', 'splits': splits})
