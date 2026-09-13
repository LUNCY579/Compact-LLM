"""Of the 103 rows the source gets wrong, how many does it actually get RIGHT?

The retention gate scores an arm against the 47 rows the source answers correctly. That leaves
103 rows of headroom, and a finetune at lr 1e-5 converts 32 of them. Before crediting that to
reasoning, it has to be split from the cheaper explanation: the model already knew the answer
and simply wrote it in a form the grammar rejects.

So for every failed row this asks one question -- **does the source's own text already assert
the right answer?** -- in the units the target is actually encoded in:

  * clock targets carry `minute_of_day`, not a numeral. `It ends at 6:55 PM.` has to be read as
    415, not as the numbers 6 and 55. A naive numeric scan misses every clock row and silently
    reports them all as reasoning failures; that error inflated an earlier version of this
    analysis to "93% genuine".
  * duration targets carry SECONDS (`'2700'`) while the model writes `45 minutes`. Compound
    forms (`1 hour 45 minutes`) have to be summed over adjacent pairs.
  * everything else is a plain rational, read with commas stripped.

Bare-meridiem clocks are scored generously: `10:10` with no am/pm matches both interpretations.
That biases the count TOWARD "format only", which is the conservative direction here -- it
makes the genuine-reasoning residual a lower bound rather than an optimistic one.

Writes `evidence/retention-design-v1/HEADROOM_AUDIT.json`. Pure scoring arithmetic, no model,
no torch, no GPU -- safe to run beside a training job.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.architecture_benchmark_001 import retention, run        # noqa: E402
from experiments.architecture_benchmark_001.gates import implementation_digest  # noqa: E402
from experiments.architecture_benchmark_001.source import ROOT           # noqa: E402

OUT = ROOT / 'evidence' / 'retention-design-v1'

NUM_RE = re.compile(r'-?\d[\d,]*(?:\.\d+)?')
CLOCK_RE = re.compile(r'\b(\d{1,2}):(\d{2})\s*(am|pm|a\.m\.|p\.m\.)?', re.I)
DUR_RE = re.compile(r'(\d[\d,]*(?:\.\d+)?)\s*'
                    r'(seconds?|secs?|s|minutes?|mins?|min|hours?|hrs?|hr|h)\b', re.I)


def clock_minutes(text: str) -> set[int]:
    found = set()
    for hour, minute, meridiem in CLOCK_RE.findall(text):
        hour, minute = int(hour), int(minute)
        if minute > 59:
            continue
        if meridiem:
            pm = meridiem.lower().replace('.', '').startswith('p')
            found.add((hour % 12 + (12 if pm else 0)) * 60 + minute)
        else:
            if hour <= 23:
                found.add(hour * 60 + minute)
            if 1 <= hour <= 12:                      # ambiguous: credit both readings
                found.add((hour % 12) * 60 + minute)
                found.add((hour % 12 + 12) * 60 + minute)
    return found


def duration_seconds(text: str) -> set[Fraction]:
    parts = [(Fraction(n.replace(',', '')), retention.DURATION_UNITS[u.lower()])
             for n, u in DUR_RE.findall(text) if u.lower() in retention.DURATION_UNITS]
    found = {value * scale for value, scale in parts}
    for i in range(len(parts) - 1):                  # `1 hour 45 minutes`
        found.add(parts[i][0] * parts[i][1] + parts[i + 1][0] * parts[i + 1][1])
    return found


def rationals(text: str) -> set[Fraction]:
    found = set()
    for match in NUM_RE.finditer(text):
        try:
            found.add(Fraction(match.group(0).replace(',', '')))
        except (ValueError, ZeroDivisionError):
            pass
    return found


def asserts_target(text: str, target: dict) -> bool:
    kind = target['kind']
    if kind == 'clock':
        return target['minute_of_day'] in clock_minutes(text)
    try:
        wanted = Fraction(str(target['value']))
    except (ValueError, ZeroDivisionError, KeyError):
        return False
    return wanted in (duration_seconds(text) if kind == 'duration' else rationals(text))


def main() -> int:
    digest = implementation_digest()
    receipt = ROOT / 'outputs' / f'retention-{digest[:8]}' / 'B28' / 'retention.json'
    if not receipt.is_file():
        raise SystemExit(f'no G1_RETENTION_B28 artifact at digest {digest[:8]}: {receipt}')
    frozen = json.loads(receipt.read_text(encoding='utf-8'))
    raw = {r['id']: r['source'] for r in frozen['rows']}
    rows = run.load_probe_rows(ROOT / 'data' / 'monitor_probe.jsonl')
    audit = json.loads((OUT / 'PROBE_LABEL_AUDIT.json').read_text(encoding='utf-8'))

    report = {'schema': 1, 'evidence_kind': 'exploratory_scoring_analysis_only',
              'note': 'Why the source fails the rows outside the retention reference, split into '
                      'answers the grammar rejected and answers the model got wrong. No model is '
                      'run: the source text is read from the frozen G1_RETENTION_B28 receipt.',
              'reference_digest': digest, 'reference_receipt': str(receipt), 'policies': {}}

    for policy in retention.BARE_POLICIES:
        overlay = retention.build_overlay(audit, policy)
        reference = set(frozen['reference_ids_by_policy'][policy])
        buckets, by_class, ids = Counter(), defaultdict(Counter), defaultdict(list)
        for row in rows:
            rid, target, src = row['id'], overlay['targets'][row['id']], raw[row['id']]
            if retention.score_prediction(src['raw'], src['terminated'], target)['correct']:
                continue
            if not src['terminated']:
                bucket = 'runaway'
            elif asserts_target(src['raw'], target):
                bucket = 'format_only'
            else:
                bucket = 'wrong_answer'
            buckets[bucket] += 1
            by_class[row['reply_class']][bucket] += 1
            ids[bucket].append(rid)
        scored_correct = {r['id'] for r in rows
                          if retention.score_prediction(raw[r['id']]['raw'],
                                                        raw[r['id']]['terminated'],
                                                        overlay['targets'][r['id']])['correct']}
        if scored_correct != reference:
            raise SystemExit(f're-scored source under {policy} does not reproduce the reference')
        total = sum(buckets.values())
        report['policies'][policy] = {
            'rows': len(rows), 'source_correct': len(reference), 'headroom': total,
            'buckets': dict(buckets), 'by_reply_class': {k: dict(v) for k, v in by_class.items()},
            'ids': {k: sorted(v) for k, v in ids.items()},
            'format_only_ceiling': buckets['format_only'],
            'grammar_limited_ceiling': len(reference) + buckets['format_only'],
        }

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'HEADROOM_AUDIT.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    for policy, body in report['policies'].items():
        print(f"[{policy}] source {body['source_correct']}/{body['rows']} correct, "
              f"{body['headroom']} headroom")
        for bucket, n in sorted(body['buckets'].items(), key=lambda kv: -kv[1]):
            print(f"    {bucket:<14} {n:3d}  ({100*n/body['headroom']:.0f}%)")
        for cls, counts in sorted(body['by_reply_class'].items()):
            print(f"    {cls:<22} {dict(sorted(counts.items()))}")
        print(f"    grammar-limited ceiling: {body['grammar_limited_ceiling']}/{body['rows']} "
              f"reachable without changing a single weight")
    print('wrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
