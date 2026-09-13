"""Measure typed-grammar retention for any checkpoint, including ones the gate refuses.

`G1_RETENTION_<arm>` declares `G1_QUALITY_<arm>` as its parent, so an arm whose quality receipt
FAILS cannot hold a retention receipt. That is the correct wiring -- a degenerate model has
nothing to retain -- but it means the registered gate can never be *demonstrated* on the arms it
exists to reject. This script does that demonstration outside the receipt system.

It imports the package's own scorer and retention arithmetic, so the numbers are produced by the
same code the gate runs. What it does not produce is a receipt, and it cannot open any stage.

    python experiments/rab001_exploration/retention_probe.py <label> <checkpoint|source> [device]

`<checkpoint>` is a directory holding config.json + model.safetensors, or the literal `source`
to score the frozen source against itself. Writes
`evidence/retention-design-v1/RETENTION_PROBE_<label>.json`.

One model job at a time: the host has 15.2 GiB and this loads the source AND the candidate.
`hostbudget.require` refuses to start if the machine cannot take it.

Exploratory. Outside the hashed root: no receipt, no gate, open splits only, no efficacy claim.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hostbudget                                                       # noqa: E402
from experiments.architecture_benchmark_001 import retention, run       # noqa: E402
from experiments.architecture_benchmark_001.gates import protocol       # noqa: E402
from experiments.architecture_benchmark_001.source import ROOT          # noqa: E402

SOURCE = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/'
              'Rineya-Supa-1.7.0-devfreeze')
OUT = ROOT / 'evidence' / 'retention-design-v1'


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    label, target = sys.argv[1], sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
    checkpoint = None if target == 'source' else Path(target)
    hostbudget.require(4.0, f'retention probe {label}')
    print(hostbudget.describe(), flush=True)

    policy = protocol()['G1_RETENTION']
    probe_path = ROOT / 'data' / 'monitor_probe.jsonl'
    rows = run.load_probe_rows(probe_path)
    audit = json.loads((ROOT / 'evidence/retention-design-v1/PROBE_LABEL_AUDIT.json')
                       .read_text(encoding='utf-8'))
    overlays = {name: retention.build_overlay(audit, name) for name in retention.BARE_POLICIES}

    started = time.perf_counter()
    reference_model, tokenizer, _ = run._load(SOURCE, device)
    reference_model.eval()
    eos = run._eos_ids(SOURCE, tokenizer)
    print(f'scoring source on {len(rows)} probe rows...', flush=True)
    source_outputs = run._greedy_probe(reference_model, tokenizer, rows, eos, device)
    del reference_model
    hostbudget.guard(2.0, f'retention probe {label}')

    if checkpoint is None:
        arm_outputs = source_outputs
    else:
        print(f'scoring {label} from {checkpoint}...', flush=True)
        model = run._restore(checkpoint, device)
        arm_outputs = run._greedy_probe(model, tokenizer, rows, eos, device)
        del model

    result = {'schema': 1, 'evidence_kind': 'exploratory_typed_retention_probe_only',
              'note': 'Produced by the package scorer but OUTSIDE the receipt system: no '
                      'receipt, no gate, no stage opened, no efficacy claim. Exists because '
                      'G1_RETENTION refuses an arm whose G1_QUALITY receipt failed, so the '
                      'gate cannot otherwise be demonstrated on the arms it rejects.',
              'label': label, 'checkpoint': None if checkpoint is None else str(checkpoint),
              'device': device, 'probe_rows': len(rows),
              'probe_sha256': run.sha256_file(probe_path),
              'registered_bare_policy': policy['bare_policy'],
              'wall_seconds': None, 'by_bare_policy': {}}

    for name, overlay in overlays.items():
        targets = overlay['targets']
        source_scored = run._score_probe(source_outputs, targets)
        arm_scored = run._score_probe(arm_outputs, targets)
        source_correct = {i for i, r in source_scored.items() if r['correct']}
        arm_correct = {i for i, r in arm_scored.items() if r['correct']}
        reference = source_correct | arm_correct
        counts = run._retention_classes(rows, source_scored, arm_scored, reference)
        verdict = retention.monitor_verdict(counts, policy)
        result['by_bare_policy'][name] = {
            'status': verdict['status'], 'failed_classes': verdict['failed_classes'],
            'uninformative_classes': verdict['uninformative_classes'],
            'source_correct_total': len(source_correct),
            'arm_correct_total': len(arm_correct),
            'reference_size': len(reference),
            'lost_total': sum(c['lost'] for c in counts.values()),
            'gained_total': sum(c['gained'] for c in counts.values()),
            'per_class': counts,
            'source_rate_by_class': {
                cls: f"{c['source_correct_terminated']}/{c['rows']}" for cls, c in counts.items()},
        }

    registered = result['by_bare_policy'][policy['bare_policy']]
    other = 'strict' if policy['bare_policy'] == 'permissive' else 'permissive'
    result['bare_policy_verdicts_agree'] = (
        registered['status'] == result['by_bare_policy'][other]['status'])
    result['samples'] = [
        {'id': r['id'], 'reply_class': r['reply_class'],
         'target': overlays[policy['bare_policy']]['targets'][r['id']],
         'source': s['text'], 'arm': a['text'],
         'source_terminated': s['terminated'], 'arm_terminated': a['terminated']}
        for r, s, a in list(zip(rows, source_outputs, arm_outputs))[:24]]
    result['wall_seconds'] = time.perf_counter() - started
    result['licenses_registered_training'] = False
    result['licenses_efficacy_claim'] = False

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'RETENTION_PROBE_{label}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    print(f"\n{label}: registered ({policy['bare_policy']}) status "
          f"{registered['status']}  source {registered['source_correct_total']}/{len(rows)}  "
          f"arm {registered['arm_correct_total']}/{len(rows)}  "
          f"lost {registered['lost_total']}  gained {registered['gained_total']}")
    for cls, c in registered['per_class'].items():
        print(f"   {cls:22s} source {c['source_correct_terminated']:2d}/{c['rows']}  "
              f"arm {c['arm_correct']:2d}/{c['rows']}  lost {c['lost']:2d}  "
              f"term {c['terminated']:3d}/{c['rows']}")
    print(f"   other policy ({other}): {result['by_bare_policy'][other]['status']}, "
          f"verdicts agree: {result['bare_policy_verdicts_agree']}")
    print(f"   {result['wall_seconds']:.0f}s -> {path}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
