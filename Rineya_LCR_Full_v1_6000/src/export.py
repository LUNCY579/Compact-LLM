"""Create derived candidate exports; this does not authorize model training.
The three source splits remain the authoritative 6,000 unique rows.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

SPLITS = {'train': 4800, 'dev': 600, 'holdout': 600}


def export(root: Path, replace: bool = False) -> dict:
    targets = [root / 'data/all.jsonl', *(root / 'sft' / f'{s}.jsonl' for s in SPLITS)]
    if not replace and any(p.exists() for p in targets):
        raise FileExistsError('Refusing existing exports; use --replace only in a working copy')
    rows = []
    for split, count in SPLITS.items():
        part = [json.loads(x) for x in (root / 'data' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
        if len(part) != count or any(x['split'] != split for x in part):
            raise ValueError(f'Invalid split/count: {split}')
        for row in part:
            if [m['role'] for m in row['messages']] != ['system', 'user', 'assistant']:
                raise ValueError(f'Invalid message roles: {row["id"]}')
        rows.extend(part)
    if len({r['id'] for r in rows}) != 6000 or len({r['group_id'] for r in rows}) != 6000:
        raise ValueError('Nonunique ids or scenario groups')
    (root / 'sft').mkdir(exist_ok=True)
    with (root / 'data/all.jsonl').open('w', encoding='utf-8', newline='\n') as out:
        for row in rows:
            out.write(json.dumps(row, ensure_ascii=False, separators=(',', ':')) + '\n')
    for split in SPLITS:
        with (root / 'sft' / f'{split}.jsonl').open('w', encoding='utf-8', newline='\n') as out:
            for row in rows:
                if row['split'] == split:
                    out.write(json.dumps({'messages': row['messages']}, ensure_ascii=False, separators=(',', ':')) + '\n')
    return {'unique_rows': 6000, 'split_counts': SPLITS,
            'combined_export_is_duplicate_view': True, 'training_authorized': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--replace', action='store_true')
    args = parser.parse_args()
    print(json.dumps(export(args.root, args.replace), indent=2))
