"""Where does the typed grammar disagree with the legacy scorer, and is the disagreement fair?

A retention gate is only as good as its scorer. HOTFIX-005 replaced number-sequence matching
with a typed grammar because the old one accepted `10:25 PM` for `10:25 AM`. That argument is
one-directional: it says the old scorer was too loose. It does not establish that the new one
is not too tight.

This cross-tabulates the two scorers on the frozen source's own answers, taken from a
G1_RETENTION receipt artifact. Four cells:

    typed+ legacy+   both agree the answer is right
    typed+ legacy-   the typed grammar is FAIRER (equivalent units the old scorer rejected)
    typed- legacy+   the typed grammar is STRICTER -- the cell that needs justifying
    typed- legacy-   both agree the answer is wrong

Every row in the third cell is printed with its raw text and parse reason, because a
systematic pattern there is a defect in the frozen grammar, not a property of the model.

    python experiments/rab001_exploration/scorer_bias_scan.py <retention.json> [label]

Exploratory. Outside the hashed root: no receipt, no gate, no efficacy claim. It reads a
receipt artifact and does string comparisons; no model is loaded.
"""
from __future__ import annotations

import collections
import json
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.rab001_exploration.scoring import answer_matches       # noqa: E402
from experiments.architecture_benchmark_001.source import ROOT          # noqa: E402

PROBE = ROOT / 'data' / 'monitor_probe.jsonl'
OUT = ROOT / 'evidence' / 'retention-design-v1'

#: Leading prose the grammar does not strip. Grouped so a systematic family is visible as one
#: pattern rather than as N unrelated rows.
FAMILIES = (
    ('sentence_prefix_not_in_contract', re.compile(
        r'^(it (ends|starts|begins|runs|arrives|finishes)\b|end time\s*:|start time\s*:|'
        r'the [a-z ]+ (ends|starts|begins|is)\b|the (percent|total|answer|result)\b)')),
    ('shows_the_arithmetic', re.compile(r'[=+]|\bto\b.*\bto\b')),
)


def family(text: str) -> str:
    low = ' '.join((text or '').strip().casefold().split())
    for name, pattern in FAMILIES:
        if pattern.search(low):
            return name
    return 'other'


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    artifact = Path(sys.argv[1])
    label = sys.argv[2] if len(sys.argv) > 2 else artifact.parent.name
    body = json.loads(artifact.read_text(encoding='utf-8'))
    probe = {json.loads(l)['id']: json.loads(l)
             for l in PROBE.read_text(encoding='utf-8-sig').splitlines() if l.strip()}

    cells = collections.Counter()
    by_class = collections.defaultdict(collections.Counter)
    stricter, fairer = [], []
    for row in body['rows']:
        got = row['source']
        want = probe[row['id']]['answer']
        typed = bool(got['correct'])
        legacy = bool(got['terminated'] and answer_matches(got['raw'], want))
        cell = ('typed+' if typed else 'typed-') + ('legacy+' if legacy else 'legacy-')
        cells[cell] += 1
        by_class[row['reply_class']][cell] += 1
        record = {'id': row['id'], 'reply_class': row['reply_class'], 'reference_answer': want,
                  'raw': got['raw'], 'target': row['target'], 'parse_reason': got['reason'],
                  'disagreement_family': family(got['raw'])}
        if legacy and not typed:
            stricter.append(record)
        elif typed and not legacy:
            fairer.append(record)

    typed_total = cells['typed+legacy+'] + cells['typed+legacy-']
    legacy_total = cells['typed+legacy+'] + cells['typed-legacy+']
    families = collections.Counter(r['disagreement_family'] for r in stricter)
    classes = collections.Counter(r['reply_class'] for r in stricter)

    result = {
        'schema': 1, 'evidence_kind': 'exploratory_scorer_disagreement_scan_only',
        'note': 'Cross-tabulation of the typed grammar against the legacy number-sequence '
                'scorer on the frozen source. Reads a receipt artifact; no model was loaded. '
                'Outside the hashed root: no receipt, no gate, no efficacy claim.',
        'label': label, 'artifact': str(artifact), 'rows': len(body['rows']),
        'cells': dict(cells),
        'typed_total': typed_total, 'legacy_total': legacy_total,
        'typed_minus_legacy': typed_total - legacy_total,
        'by_class': {k: dict(v) for k, v in by_class.items()},
        'stricter_than_legacy_count': len(stricter),
        'stricter_by_family': dict(families), 'stricter_by_class': dict(classes),
        'stricter_rows': stricter,
        'fairer_than_legacy_count': len(fairer), 'fairer_rows': fairer,
        'interpretation': (
            'Rows where the typed grammar is stricter are only defensible if they are genuinely '
            'wrong answers. A concentrated family there is a defect in the frozen prefix list, '
            'and it shrinks the reference set -- which makes the retention gate WEAKER, since '
            'there are fewer items a candidate can be caught losing.'),
        'licenses_registered_training': False, 'licenses_efficacy_claim': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'SCORER_BIAS_{label}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    print(f'{label}: {len(body["rows"])} rows')
    for cell in ('typed+legacy+', 'typed+legacy-', 'typed-legacy+', 'typed-legacy-'):
        print(f'  {cell}: {cells[cell]:3d}')
    print(f'  typed {typed_total}/{len(body["rows"])}  legacy {legacy_total}/{len(body["rows"])}')
    print(f'\n  stricter-than-legacy rows: {len(stricter)}')
    print(f'    by class : {dict(classes)}')
    print(f'    by family: {dict(families)}')
    for r in stricter:
        print(f"      [{r['reply_class'][:4]}] want={r['reference_answer']!r:16s} "
              f"got={r['raw']!r:44s} [{r['disagreement_family']}]")
    print(f'\n  fairer-than-legacy rows: {len(fairer)}')
    for r in fairer:
        print(f"      [{r['reply_class'][:4]}] want={r['reference_answer']!r:16s} got={r['raw']!r}")
    print('\nwrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
