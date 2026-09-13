"""Measure typed-grammar retention for any checkpoint, including ones the gate refuses.

`G1_RETENTION_<arm>` declares `G1_QUALITY_<arm>` as its parent, so an arm whose quality receipt
FAILS cannot hold a retention receipt. That is the correct wiring -- a degenerate model has
nothing to retain -- but it means the registered gate can never be *demonstrated* on the arms it
exists to reject. This script does that demonstration outside the receipt system.

It imports the package's own scorer and retention arithmetic, so the numbers are produced by the
same code the gate runs. What it does not produce is a receipt, and it cannot open any stage.

    python experiments/rab001_exploration/retention_probe.py <label> <target> [device]

`<target>` is one of:
  * a directory holding config.json + model.safetensors
  * `source` -- the frozen source scored against itself (tautological, a control)
  * `ced_lastN` -- built in memory by `to_ced` with N trailing layers displaced, the same
    construction `rigorous_pilot.py` uses, so the displaced-layer family can be measured on
    CAPABILITY rather than on cross-entropy

Writes `evidence/retention-design-v1/RETENTION_PROBE_<label>.json`.

The source side is NOT re-generated. It is re-scored from the raw answers stored in the
G1_RETENTION_B28 receipt at the current digest, and the run aborts unless that reproduces both
frozen references exactly. That halves the runtime and the memory, and it makes the reference
provably the same one the registered gate uses.

One model job at a time. `hostbudget.require` refuses to start if the machine cannot take it.

Exploratory. Outside the hashed root: no receipt, no gate, open splits only, no efficacy claim.
"""
from __future__ import annotations

import gc
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
from experiments.architecture_benchmark_001.gates import (implementation_digest,       # noqa: E402
                                                          protocol)
from experiments.architecture_benchmark_001.source import ROOT          # noqa: E402

SOURCE = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/'
              'Rineya-Supa-1.7.0-devfreeze')
OUT = ROOT / 'evidence' / 'retention-design-v1'


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    label, target = sys.argv[1], sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else 'cpu'
    hostbudget.require(5.0, f'retention probe {label}')
    print(hostbudget.describe(), flush=True)

    policy = protocol()['G1_RETENTION']
    probe_path = ROOT / 'data' / 'monitor_probe.jsonl'
    rows = run.load_probe_rows(probe_path)
    audit = json.loads((ROOT / 'evidence/retention-design-v1/PROBE_LABEL_AUDIT.json')
                       .read_text(encoding='utf-8'))
    overlays = {name: retention.build_overlay(audit, name) for name in retention.BARE_POLICIES}

    # The source side comes from the frozen G1_RETENTION_B28 receipt at the CURRENT digest,
    # re-scored from its stored raw answers. No second model pass, and the reference is provably
    # the one the registered gate uses. The equality check below is the control: if re-scoring
    # does not reproduce both frozen references exactly, something has drifted and no verdict
    # here would mean anything.
    short = implementation_digest()[:8]
    receipt_path = ROOT / 'outputs' / f'retention-{short}' / 'B28' / 'retention.json'
    if not receipt_path.is_file():
        raise SystemExit(f'no G1_RETENTION_B28 artifact at the current digest {short}: '
                         f'{receipt_path}. Run the retention stage at this digest first.')
    frozen = json.loads(receipt_path.read_text(encoding='utf-8'))
    references = {k: set(v) for k, v in frozen['reference_ids_by_policy'].items()}
    source_raw = {r['id']: r['source'] for r in frozen['rows']}
    if set(source_raw) != {r['id'] for r in rows}:
        raise SystemExit('frozen receipt does not cover exactly the probe rows')
    source_scored_by_policy = {}
    for name, overlay in overlays.items():
        scored = {i: {**retention.score_prediction(rec['raw'], rec['terminated'],
                                                   overlay['targets'][i]),
                      'reply_class': next(r['reply_class'] for r in rows if r['id'] == i)}
                  for i, rec in source_raw.items()}
        got = {i for i, r in scored.items() if r['correct']}
        if got != references[name]:
            raise SystemExit(f're-scored source under {name} does not reproduce the frozen '
                             f'reference: {len(got)} vs {len(references[name])}')
        source_scored_by_policy[name] = scored
    print(f'frozen reference reproduced exactly: '
          f'{ {k: len(v) for k, v in references.items()} }', flush=True)

    started = time.perf_counter()
    tokenizer = run._load_tokenizer(SOURCE)
    eos = run._eos_ids(SOURCE, tokenizer)
    built_from = None
    if target == 'source':
        model = run._load(SOURCE, device)[0]
        built_from = 'frozen source (tautological control)'
    elif target.startswith('ced_last'):
        # Exactly rigorous_pilot.py's construction: N trailing layers displaced onto a shared
        # encoder-final bank. Built in memory; no weights are written.
        from dataclasses import replace as _replace
        from experiments.architecture_benchmark_001.transfer import to_ced
        n = int(target[len('ced_last'):])
        src = run._load(SOURCE, device)[0]
        enc = src.config.total_layers - n
        if not 0 < n < src.config.total_layers:
            raise SystemExit(f'ced_last{n} is outside 1..{src.config.total_layers - 1}')
        model, _ = to_ced(src, _replace(src.config, architecture='ced',
                                        encoder_layers=enc, decoder_layers=n), donor_layer=enc)
        del src
        gc.collect()
        built_from = f'to_ced(encoder_layers={enc}, decoder_layers={n}, donor_layer={enc})'
    else:
        model = run._restore(Path(target), device)
        built_from = f'checkpoint {target}'
    model.eval()
    hostbudget.guard(1.5, f'retention probe {label}')
    print(f'scoring {label} ({built_from}) on {len(rows)} probe rows...', flush=True)
    arm_outputs = run._greedy_probe(model, tokenizer, rows, eos, device)
    del model
    gc.collect()

    result = {'schema': 1, 'evidence_kind': 'exploratory_typed_retention_probe_only',
              'note': 'Produced by the package scorer but OUTSIDE the receipt system: no '
                      'receipt, no gate, no stage opened, no efficacy claim. Exists because '
                      'G1_RETENTION refuses an arm whose G1_QUALITY receipt failed, so the '
                      'gate cannot otherwise be demonstrated on the arms it rejects.',
              'label': label, 'target': target, 'built_from': built_from,
              'reference_receipt': str(receipt_path), 'reference_digest': implementation_digest(),
              'frozen_reference_sizes': {k: len(v) for k, v in references.items()},
              'device': device, 'probe_rows': len(rows),
              'probe_sha256': run.sha256_file(probe_path),
              'registered_bare_policy': policy['bare_policy'],
              'wall_seconds': None, 'by_bare_policy': {}}

    for name, overlay in overlays.items():
        targets = overlay['targets']
        source_scored = source_scored_by_policy[name]
        arm_scored = run._score_probe(arm_outputs, targets)
        source_correct = {i for i, r in source_scored.items() if r['correct']}
        arm_correct = {i for i, r in arm_scored.items() if r['correct']}
        # Union with the arm's own correct set, matching G1_RETENTION's reference rule.
        reference = source_correct | arm_correct
        counts = run._retention_classes(rows, source_scored, arm_scored, reference)
        verdict = retention.monitor_verdict(counts, policy, policy['probe_classes'])
        result['by_bare_policy'][name] = {
            'status': verdict['status'], 'failed_classes': verdict['failed_classes'],
            'uninformative_classes': verdict['uninformative_classes'],
            'source_correct_total': len(source_correct),
            'arm_correct_total': len(arm_correct),
            'reference_size': len(reference),
            'lost_total': sum(c['lost'] for c in counts.values()),
            # `gained` is identically 0 wherever the reference is the source/arm union, because
            # every arm-correct id is in that union by construction. The meaningful count is how
            # many of the arm's correct answers the SOURCE got wrong, which is reference - source.
            'gained_total_is_vacuous_here': True,
            'arm_correct_outside_source': len(reference) - len(source_correct),
            'reference_minus_source_note': 'answers the arm gets right that the frozen source does not',
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
         'source': source_raw[r['id']]['raw'], 'arm': a['text'],
         'source_terminated': source_raw[r['id']]['terminated'],
         'arm_terminated': a['terminated']}
        for r, a in list(zip(rows, arm_outputs))[:24]]
    result['wall_seconds'] = time.perf_counter() - started
    result['licenses_registered_training'] = False
    result['licenses_efficacy_claim'] = False

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f'RETENTION_PROBE_{label}.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')

    print(f"\n{label}: registered ({policy['bare_policy']}) status "
          f"{registered['status']}  source {registered['source_correct_total']}/{len(rows)}  "
          f"arm {registered['arm_correct_total']}/{len(rows)}  ref {registered['reference_size']}  "
          f"lost {registered['lost_total']}  new-vs-source {registered['arm_correct_outside_source']}")
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
