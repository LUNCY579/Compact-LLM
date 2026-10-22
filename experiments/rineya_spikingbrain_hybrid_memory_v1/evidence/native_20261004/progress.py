"""Compact status from actual stage receipts; no model/library imports."""
import json
from pathlib import Path
import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run-root', type=Path,
                    default=Path(__file__).resolve().parents[2]/'runs/titan_transition_v2_20261004')
root = parser.parse_args().run_root.resolve()
result = []
for directory in sorted(root.iterdir()):
    if not directory.is_dir():
        continue
    item = {'stage': directory.name, 'state': 'REQUESTED_NO_COMPLETED_RECEIPT'}
    receipt = directory / 'receipt.json'
    if receipt.exists():
        value = json.loads(receipt.read_text(encoding='utf-8'))
        item.update(state=value['status'], created_utc=value.get('created_utc'))
        if 'records' in value:
            item.update(steps=len(value['records']), first_loss=value['records'][0]['loss'],
                        last_loss=value['records'][-1]['loss'])
        if 'entries' in value:
            item.update(teacher_rows=len(value['entries']), file_bytes=value['teacher_cache_file_bytes'])
        if 'predictions' in value:
            item.update(dev_rows=len(value['predictions']), strict_exact_match=value['strict_exact_match'])
    result.append(item)
print(json.dumps({'stages': result, 'comparison_complete': (root / 'comparison.json').exists()}, indent=2))
