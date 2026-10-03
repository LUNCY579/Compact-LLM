"""Verify manifest integrity and all delivered row/projection identities.
No model, tokenizer, fitting, rewriting, or acceptance upgrade is performed.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path

SPLITS = {'train': 4800, 'dev': 600, 'holdout': 600}


def read_jsonl(path: Path) -> list[dict]:
    payload = path.read_bytes()
    if b'\r' in payload or not payload.endswith(b'\n'):
        raise ValueError(f'Expected UTF-8 LF-delimited file: {path.name}')
    return [json.loads(line) for line in payload.decode('utf-8').splitlines()]


def verify(root: Path) -> dict:
    root = root.resolve()
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    for name, expected in manifest['files'].items():
        p = root / name
        if p.is_symlink() or not p.is_file() or not p.resolve().is_relative_to(root):
            raise ValueError(f'Invalid manifest path: {name}')
        raw = p.read_bytes()
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError(f'Integrity mismatch: {name}')
    all_rows = []
    for split, count in SPLITS.items():
        rows = read_jsonl(root / 'data' / f'{split}.jsonl')
        worlds = read_jsonl(root / 'worlds' / f'{split}.jsonl')
        sft = read_jsonl(root / 'sft' / f'{split}.jsonl')
        if len(rows) != count or len(worlds) != count or len(sft) != count:
            raise ValueError(f'Count mismatch: {split}')
        for row, world, view in zip(rows, worlds, sft, strict=True):
            if row['id'] != world['id'] or row['group_id'] != world['group_id'] or row['split'] != split:
                raise ValueError(f'Identity mismatch: {split}')
            if view != {'messages': row['messages']} or row['messages'][-1]['content'] != world['gold']:
                raise ValueError(f'Projection or world mismatch: {row["id"]}')
            if [x['role'] for x in row['messages']] != ['system', 'user', 'assistant']:
                raise ValueError('Message role mismatch')
            if any(not isinstance(x['content'], str) or not x['content'].strip() for x in row['messages']):
                raise ValueError('Empty message')
            if row['messages'][-1]['content'] == 'UNKNOWN' and row['metadata']['needle_char_spans']:
                raise ValueError('UNKNOWN with invented positive evidence')
        all_rows.extend(rows)
    if read_jsonl(root / 'data/all.jsonl') != all_rows:
        raise ValueError('Combined export differs from source splits')
    if len({r['id'] for r in all_rows}) != 6000 or len({r['group_id'] for r in all_rows}) != 6000:
        raise ValueError('Duplicate row or scenario ids')
    receipts = read_jsonl(root / 'reports/verification_rows.jsonl')
    if [r['id'] for r in receipts] != [r['id'] for r in all_rows]:
        raise ValueError('Verification receipt alignment mismatch')
    if any(not x['agree'] or x['derived_answer'] != r['messages'][-1]['content']
           for x, r in zip(receipts, all_rows, strict=True)):
        raise ValueError('Verification answer mismatch')
    acceptance = json.loads((root / 'reports/acceptance.json').read_text())
    if acceptance['corpus_accepted'] is not False:
        raise ValueError('Unqualified acceptance claim')
    return {'status': 'INTEGRITY_AND_EXPORT_IDENTITY_PASS', 'manifest_files_checked': len(manifest['files']),
            'unique_rows': 6000, 'split_counts': SPLITS,
            'unknown_rows': sum(r['messages'][-1]['content'] == 'UNKNOWN' for r in all_rows),
            'combined_export_matches': True, 'sft_projections_match': True,
            'world_and_verification_labels_match': True,
            'native_token_qualification': 'UNCHANGED_UNVERIFIED', 'corpus_accepted': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    print(json.dumps(verify(args.root), indent=2))
