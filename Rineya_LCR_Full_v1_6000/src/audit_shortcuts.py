"""Reproduce the finite shortcut audit without rerunning semantic validation.
Run in a working COPY: this overwrites the two shortcut report files, never data.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    root = parser.parse_args().root
    rows = audit.load(root)
    worlds = [json.loads(line) for split in audit.SPLITS
              for line in (root / 'worlds' / f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    audit.worlds_by_id = {world['id']: world for world in worlds}
    if len(rows) != len(audit.worlds_by_id) or {row['id'] for row in rows} != set(audit.worlds_by_id):
        raise ValueError('Row/world identity mismatch')
    report, features = audit.shortcuts(rows)
    audit.save(root / 'reports/shortcut_floors.json', report)
    with (root / 'reports/surface_features.jsonl').open('w', encoding='utf-8', newline='\n') as out:
        for row, feature in zip(rows, features, strict=True):
            item = {'id': row['id'], 'split': row['split'],
                    'unanswerable': row['messages'][-1]['content'] == 'UNKNOWN', 'features': feature}
            out.write(json.dumps(item, separators=(',', ':')) + '\n')
    print(json.dumps({'rows': len(rows),
        'named_answer_baselines_le_25pct': report['all_named_answer_baselines_le_25pct'],
        'question_only_answerability': report['question_only_answerability'],
        'selector_plus_status_answerability': report['selector_plus_status_answerability']}, indent=2))


if __name__ == '__main__':
    main()
