"""Measure a published dense core on the frozen 150-row panel, and decide whether it ships.

`materialize_opallo.py` proves the published file *contains* the trained weights. It
cannot prove the published file *behaves* like them, because the vault is BF16 and this
programme trained and measured in float32. Casting 314M parameters from fp32 to bf16
moves essentially every one of them; 8 mantissa bits is a 4e-3 relative perturbation, and
whether that is worth zero panel rows or five is an empirical question.

So this runs the published core through the identical path that produced the checkpoint's
number: same 150 rows, same frozen reference receipt, same greedy decode, same CPU device,
same scorer. A score measured any other way is not comparable and must not be quoted.

    python experiments/rab001_exploration/validate_published_core.py \
        --core Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze \
        --against B28last20_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95 \
        --min-correct 70 --min-retained 38

`--min-correct` and `--min-retained` are required and have no defaults on purpose. They
belong to a pre-registration written before the number exists (`OPALLO_PREREG.md`), not to
this file, and passing them explicitly at the call site makes the commit that ran the
check also record what it was checked against.

Reported alongside the score, because a raw count on a 150-row instrument whose measured
seed-to-seed noise floor is 15-24 discordant rows is not by itself informative:

  * per-row agreement with the float32 checkpoint -- the number that actually isolates the
    cast, since both models saw identical prompts under identical decoding and any
    disagreement is the cast and nothing else;
  * retained / lost / gained against the 47-row frozen reference set;
  * termination, which is the failure mode this programme has repeatedly found to move
    when accuracy does not.

Writes `PANEL.json` beside the core and patches `CORE.json`'s `panel_validation` block in
place. The verdict is written whether it passes or fails -- a failed validation is
evidence, and deleting it would leave a published core carrying an unmeasured claim.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '-1')
sys.stdout.reconfigure(encoding='utf-8')

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.architecture_benchmark_001 import retention, run                # noqa: E402
from experiments.architecture_benchmark_001.gates import implementation_digest   # noqa: E402
from experiments.architecture_benchmark_001.source import ROOT                   # noqa: E402
from experiments.rab001_exploration import hostbudget                            # noqa: E402

EVIDENCE = ROOT / 'evidence' / 'retention-design-v1'


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--core', required=True, help='published core directory')
    parser.add_argument('--against', help='arm-checkpoint tag whose float32 outputs to compare to')
    parser.add_argument('--min-correct', type=int, required=True, help='pre-registered threshold')
    parser.add_argument('--min-retained', type=int, required=True, help='pre-registered threshold')
    parser.add_argument('--policy', default='permissive')
    args = parser.parse_args()

    core_dir = (REPO / args.core) if not Path(args.core).is_absolute() else Path(args.core)
    if not (core_dir / 'model.safetensors').is_file():
        raise SystemExit(f'no dense weights at {core_dir}')
    # One fp32 copy of a 596M model on the CPU, plus the bf16 file being read.
    hostbudget.require(3.6, f'panel validation of {core_dir.name}')

    digest = implementation_digest()
    receipt = ROOT / 'outputs' / f'retention-{digest[:8]}' / 'B28' / 'retention.json'
    if not receipt.is_file():
        raise SystemExit(f'no frozen reference receipt at digest {digest[:8]}: {receipt}')
    frozen = json.loads(receipt.read_text(encoding='utf-8'))
    references = {k: set(v) for k, v in frozen['reference_ids_by_policy'].items()}
    source_raw = {r['id']: r['source'] for r in frozen['rows']}

    rows = run.load_probe_rows(ROOT / 'data' / 'monitor_probe.jsonl')
    audit = json.loads((EVIDENCE / 'PROBE_LABEL_AUDIT.json').read_text(encoding='utf-8'))
    overlays = {n: retention.build_overlay(audit, n) for n in retention.BARE_POLICIES}
    if set(source_raw) != {r['id'] for r in rows}:
        raise SystemExit('frozen receipt does not cover exactly the probe rows')

    # Re-score the frozen source predictions and require them to reproduce the reference
    # exactly. If that fails the instrument has moved and no score measured here means
    # anything, so it is checked before the core is even loaded.
    source_scored = {}
    for name, overlay in overlays.items():
        scored = {i: {**retention.score_prediction(rec['raw'], rec['terminated'],
                                                   overlay['targets'][i]),
                      'reply_class': next(r['reply_class'] for r in rows if r['id'] == i)}
                  for i, rec in source_raw.items()}
        if {i for i, r in scored.items() if r['correct']} != references[name]:
            raise SystemExit(f're-scored source under {name} does not reproduce the reference')
        source_scored[name] = scored
    print(f'frozen reference reproduced: {({k: len(v) for k, v in references.items()})}',
          flush=True)

    model, tokenizer, _ = run._load(core_dir, 'cpu')
    eos = run._eos_ids(core_dir, tokenizer)
    print(f'loaded {core_dir.name} on cpu | {sum(p.numel() for p in model.parameters()):,} '
          f'parameters | {hostbudget.describe()}', flush=True)

    outputs = run._greedy_probe(model, tokenizer, rows, eos, 'cpu')
    predictions = {o['id']: {'text': o['text'], 'terminated': o['terminated']} for o in outputs}
    del model

    results, per_policy = {}, {}
    for name, overlay in overlays.items():
        scored = run._score_probe(outputs, overlay['targets'])
        counts = run._retention_classes(rows, source_scored[name], scored, references[name])
        correct = sorted(i for i, r in scored.items() if r['correct'])
        results[name] = scored
        per_policy[name] = {
            'correct_total': len(correct),
            'rows': len(rows),
            'reference_n': len(references[name]),
            'retained_total': len(set(correct) & references[name]),
            'lost_total': len(references[name] - set(correct)),
            'gained_total': len(set(correct) - references[name]),
            'terminated_total': sum(1 for o in outputs if o['terminated']),
            'correct_ids': correct,
            'per_class': counts,
        }
        print(f'  {name:11s} correct {len(correct):3d}/{len(rows)} | '
              f'kept {len(set(correct) & references[name]):2d}/{len(references[name])} '
              f'lost {len(references[name] - set(correct)):2d} '
              f'gained {len(set(correct) - references[name]):2d} | '
              f"terminated {per_policy[name]['terminated_total']:3d}", flush=True)

    # Per-row agreement with the float32 checkpoint. This is the measurement that isolates
    # the cast: identical prompts, identical decoding, one difference.
    parity = None
    if args.against:
        artifact = EVIDENCE / f'ARM_DOSE_{args.against}.json'
        if not artifact.is_file():
            raise SystemExit(f'no float32 artifact to compare against: {artifact}')
        prior = json.loads(artifact.read_text(encoding='utf-8'))
        if not prior.get('completed'):
            raise SystemExit(f'{args.against} did not complete; its recorded score is its '
                             f'step-0 probe, which is the untrained source')
        final = prior['checkpoints'][-1]
        reference_outputs = final['outputs']
        if set(reference_outputs) != set(predictions):
            raise SystemExit('float32 artifact does not cover exactly the same probe rows')
        same_text = [i for i in predictions
                     if predictions[i]['text'] == reference_outputs[i]['text']]
        same_term = [i for i in predictions
                     if predictions[i]['terminated'] == reference_outputs[i]['terminated']]
        # `per_row` is a flat {row_id: bool} under the run's *registered* policy, so it is
        # only comparable when this validation scores under that same policy.
        registered = prior.get('registered_bare_policy')
        per_row = final['retention'].get('per_row')
        prior_correct = None
        if isinstance(per_row, dict) and registered == args.policy:
            prior_correct = {i for i, ok in per_row.items() if ok}
            if len(prior_correct) != final['retention'][args.policy]['correct_total']:
                raise SystemExit('float32 per-row record disagrees with its own total')
        elif isinstance(per_row, dict):
            print(f'  NOTE per-row correctness was recorded under {registered!r}, not '
                  f'{args.policy!r}; skipping the correctness-level parity breakdown')
        parity = {
            'float32_tag': args.against,
            'float32_artifact': str(artifact.relative_to(REPO)),
            'float32_correct_total': final['retention'][args.policy]['correct_total'],
            'float32_retained_total': final['retention'][args.policy]['retained_total'],
            'bf16_correct_total': per_policy[args.policy]['correct_total'],
            'bf16_retained_total': per_policy[args.policy]['retained_total'],
            'identical_text_rows': len(same_text),
            'identical_termination_rows': len(same_term),
            'rows': len(predictions),
            'differing_ids': sorted(set(predictions) - set(same_text)),
        }
        if prior_correct is not None:
            now = set(per_policy[args.policy]['correct_ids'])
            parity['discordant_correctness_rows'] = len(prior_correct ^ now)
            parity['lost_vs_float32'] = sorted(prior_correct - now)
            parity['gained_vs_float32'] = sorted(now - prior_correct)
        print(f"\n  cast parity vs float32 {args.against}:")
        print(f"    identical generated text  {len(same_text):3d}/{len(predictions)}")
        print(f"    identical termination     {len(same_term):3d}/{len(predictions)}")
        print(f"    panel {parity['float32_correct_total']} (fp32) -> "
              f"{parity['bf16_correct_total']} (bf16), kept "
              f"{parity['float32_retained_total']} -> {parity['bf16_retained_total']}")

    headline = per_policy[args.policy]
    passed = (headline['correct_total'] >= args.min_correct
              and headline['retained_total'] >= args.min_retained)
    verdict = {
        'schema': 1,
        'evidence_kind': 'published_core_panel_validation',
        'status': 'PASS' if passed else 'FAIL',
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'core_directory': core_dir.name,
        'implementation_digest': digest,
        'reference_receipt': str(receipt.relative_to(REPO)),
        'probe': {
            'rows': len(rows),
            'path': 'experiments/architecture_benchmark_001/data/monitor_probe.jsonl',
            'device': 'cpu', 'decoding': 'greedy', 'dtype_at_inference': 'float32',
            'note': 'BF16 weights upcast to float32 for inference, which is the path '
                    'run._load takes for every core including the source.',
        },
        'registered_policy': args.policy,
        'thresholds': {'min_correct': args.min_correct, 'min_retained': args.min_retained,
                       'preregistration': 'OPALLO_PREREG.md'},
        'headline': {k: v for k, v in headline.items() if k not in ('correct_ids', 'per_class')},
        'by_policy': per_policy,
        'cast_parity': parity,
        'predictions': predictions,
        'licenses_registered_training': False,
        'licenses_efficacy_claim': False,
    }
    (core_dir / 'PANEL.json').write_text(json.dumps(verdict, indent=2, sort_keys=True) + '\n',
                                         encoding='utf-8')

    core_path = core_dir / 'CORE.json'
    core = json.loads(core_path.read_text(encoding='utf-8'))
    core['panel_validation'] = {
        'status': verdict['status'],
        'measured_at_utc': verdict['measured_at_utc'],
        'instrument': '150-row arithmetic word-problem panel (monitor_probe.jsonl), classes '
                      'percentage_reasoning / ratio_proportion / time_duration',
        'policy': args.policy,
        'correct_total': headline['correct_total'],
        'rows': headline['rows'],
        'source_correct_total': len(references[args.policy]),
        'retained_total': headline['retained_total'],
        'lost_total': headline['lost_total'],
        'gained_total': headline['gained_total'],
        'terminated_total': headline['terminated_total'],
        'thresholds': verdict['thresholds'],
        'cast_parity': None if parity is None else
            {k: v for k, v in parity.items() if k != 'differing_ids'},
        'noise_floor_note': 'Same-config different-seed panel pairs on this instrument run '
                            '15-24 discordant rows. A single-seed panel count is not a '
                            'significance claim; see NOISE_FLOOR.md.',
        'detail': 'PANEL.json',
    }
    core_path.write_text(json.dumps(core, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    print(f"\n{verdict['status']}: panel {headline['correct_total']}/{headline['rows']} "
          f"(threshold >= {args.min_correct}), kept {headline['retained_total']}/"
          f"{headline['reference_n']} (threshold >= {args.min_retained})")
    print(f'  wrote {core_dir / "PANEL.json"} and patched CORE.json')
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
