"""Greedy model soup over the saved arm checkpoints.

Averaging the weights of several fine-tunes of one pretrained model often beats every
ingredient, because fine-tunes from a shared initialisation stay in one loss basin. That is the
claim this tests, on checkpoints written by `arm_dose.py`.

Two rules make the difference between a real result and a self-deception:

**Ingredients must be compatible.** Each checkpoint carries a `soup_key` -- mode, read_tail,
source digest, and a hash of its tensor-name set. Two checkpoints are averaged only if those
match exactly. A 596M B28 and a 1.156B H56 are different shapes; an `Hlast4` and a full `H`
train different tensors. Mixing them is meaningless, so it is refused rather than coerced.

**Selection runs on a held-out set, never on the panel.** Greedy soup chooses which ingredients
to keep by measuring them, and measuring on the 150-row panel would fit the soup to the test set
and make the reported panel number worthless. Selection therefore uses the IN-DOMAIN dev built by
`build_splits_onpanel.py` -- same three reply classes as the panel, drawn from families the panel
never took. The panel is scored once, at the end, for reporting only.

    python experiments/rab001_exploration/model_soup.py [--eval-rows N] [tag ...]

With no tags it soups every compatible group it finds. Selection metric is normalised
exact-match against the reference answer: strip, casefold, collapse whitespace, drop a trailing
period. That is deliberately cruder than the typed scorer, and it is only ever used to RANK
candidates -- every candidate is scored the same way, so the ranking is fair. The final panel
number uses the full typed scorer and the wrong_answer/format_only strata, exactly as every other
result in this programme.

One model at a time, probed on the CPU, guarded by hostbudget. Averaging itself needs no model:
it is done on the saved tensors directly.

Exploratory. Outside the hashed root: no receipt, no gate, no efficacy claim, no release.
"""
from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hostbudget                                                       # noqa: E402
from experiments.architecture_benchmark_001 import retention, run        # noqa: E402
from experiments.architecture_benchmark_001.gates import (implementation_digest,  # noqa: E402
                                                          protocol)
from experiments.architecture_benchmark_001.source import ROOT           # noqa: E402

CKPT = ROOT / 'outputs' / 'arm-checkpoints'
OUT = ROOT / 'evidence' / 'retention-design-v1'
ONPANEL = ROOT / 'outputs' / 'splits_onpanel'

SRC = REPO / 'Supa'
if not (SRC / 'model.safetensors').is_file():
    SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
               r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')

_WS = re.compile(r'\s+')


def normalise(text: str) -> str:
    return _WS.sub(' ', text.strip().casefold()).rstrip('.').strip()


def load_manifest(directory: Path) -> dict | None:
    path = directory / 'MANIFEST.json'
    if not path.is_file() or not (directory / 'trainable.safetensors').is_file():
        return None
    body = json.loads(path.read_text(encoding='utf-8'))
    body['_dir'] = directory
    return body


def build_arm(mode: str, read_tail, device: str):
    """Rebuild the architecture the checkpoint was trained in, weights still at source values."""
    if mode.startswith('B28'):
        return run._load(SRC, device)[0]
    from experiments.dual28_ced_v1.model import DualStack
    source = run._load(SRC, device)[0]
    model = DualStack.from_source(source, mode)
    del source
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()
    return model


def apply_tensors(model, tensors: dict) -> int:
    by_name = dict(model.named_parameters())
    missing = [n for n in tensors if n not in by_name]
    if missing:
        raise SystemExit(f'checkpoint tensor absent from the rebuilt model: {missing[:3]}')
    with torch.no_grad():
        for name, value in tensors.items():
            target = by_name[name]
            if tuple(target.shape) != tuple(value.shape):
                raise SystemExit(f'shape mismatch for {name}: {tuple(target.shape)} vs '
                                 f'{tuple(value.shape)}')
            target.copy_(value.to(target.device, target.dtype))
    return len(tensors)


def average(paths: list[Path]) -> dict:
    """Uniform average. Streams one checkpoint at a time so N ingredients cost one copy."""
    from safetensors.torch import load_file
    total = None
    for path in paths:
        tensors = load_file(str(path / 'trainable.safetensors'), device='cpu')
        if total is None:
            total = {k: v.to(torch.float64) for k, v in tensors.items()}
        else:
            if set(tensors) != set(total):
                raise SystemExit(f'tensor-name mismatch in {path.name}')
            for k, v in tensors.items():
                total[k] += v.to(torch.float64)
        del tensors
        gc.collect()
    return {k: (v / len(paths)).to(torch.float32) for k, v in total.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags', nargs='*')
    parser.add_argument('--eval-rows', type=int, default=60,
                        help='held-out rows used for SELECTION (default 60)')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()

    if not ONPANEL.is_dir():
        raise SystemExit(
            'no in-domain held-out set. Run build_splits_onpanel.py first -- selecting soup '
            'ingredients on the 150-row panel would fit the soup to the test set.')

    manifests = [m for m in (load_manifest(d) for d in sorted(CKPT.iterdir()) if d.is_dir()) if m]
    if args.tags:
        manifests = [m for m in manifests if m['tag'] in args.tags]
    if not manifests:
        raise SystemExit(f'no usable checkpoints under {CKPT}')

    groups: dict[str, list[dict]] = {}
    for m in manifests:
        groups.setdefault(json.dumps(m['soup_key'], sort_keys=True), []).append(m)
    print(f'{len(manifests)} checkpoint(s) in {len(groups)} compatible group(s)')
    for key, members in groups.items():
        print(f"  [{json.loads(key)['mode']}] {', '.join(m['tag'] for m in members)}")

    dev = [json.loads(l) for l in (ONPANEL / 'dev.jsonl').read_text(encoding='utf-8').splitlines()
           if l.strip()][:args.eval_rows]
    rows = run.load_probe_rows(ROOT / 'data' / 'monitor_probe.jsonl')
    audit = json.loads((OUT / 'PROBE_LABEL_AUDIT.json').read_text(encoding='utf-8'))
    policy = protocol()['G1_RETENTION']
    overlay = retention.build_overlay(audit, policy['bare_policy'])
    frozen = json.loads((ROOT / 'outputs' / f'retention-{implementation_digest()[:8]}' / 'B28' /
                         'retention.json').read_text(encoding='utf-8'))
    references = set(frozen['reference_ids_by_policy'][policy['bare_policy']])
    source_scored = {i: retention.score_prediction(r['raw'], r['terminated'],
                                                   overlay['targets'][i])
                     for i, r in ((x['id'], x['source']) for x in frozen['rows'])}
    strata = json.loads((OUT / 'HEADROOM_AUDIT.json').read_text(encoding='utf-8'))
    of_stratum = {rid: name for name, ids
                  in strata['policies'][policy['bare_policy']]['ids'].items() for rid in ids}
    tok = run._load_tokenizer(SRC)
    eos = run._eos_ids(SRC, tok)
    print(f'selection set: {len(dev)} in-domain held-out rows | panel: {len(rows)} rows')

    def evaluate(mode, read_tail, tensors, panel=False):
        """Selection score on held-out dev, and optionally the full typed panel score."""
        hostbudget.guard(1.2, 'model soup evaluate')
        model = build_arm(mode, read_tail, args.device)
        apply_tensors(model, tensors)
        model.eval()
        outputs = run._greedy_probe(model, tok, dev, eos, args.device)
        hits = sum(1 for o in outputs
                   if normalise(o['text']) == normalise(
                       next(d['answer'] for d in dev if d['id'] == o['id'])))
        result = {'dev_exact': hits, 'dev_rows': len(dev)}
        if panel:
            probe = run._greedy_probe(model, tok, rows, eos, args.device)
            scored = run._score_probe(probe, overlay['targets'])
            gained = [i for i, r in scored.items() if r['correct'] and i in of_stratum]
            result.update({
                'panel_correct': sum(1 for r in scored.values() if r['correct']),
                'panel_kept': sum(1 for i in references if scored[i]['correct']),
                'panel_lost': sum(1 for i in references if not scored[i]['correct']),
                'gained_wrong_answer': sum(1 for i in gained
                                           if of_stratum[i] == 'wrong_answer'),
                'gained_format_only': sum(1 for i in gained
                                          if of_stratum[i] == 'format_only'),
                'reference_n': len(references)})
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    report = {'schema': 1, 'evidence_kind': 'exploratory_weight_averaging_only',
              'note': 'Greedy soup. Ingredients selected on an IN-DOMAIN held-out set, never on '
                      'the panel; the panel is scored once at the end for reporting. Outside '
                      'the hashed root: no receipt, no gate, no efficacy claim, no release.',
              'selection_metric': 'normalised exact match on held-out reference answers',
              'selection_rows': len(dev), 'groups': {}}

    for key, members in groups.items():
        soup_key = json.loads(key)
        mode, read_tail = soup_key['mode'], soup_key['read_tail']
        if len(members) < 2:
            print(f"\n[{mode}] only {len(members)} ingredient; averaging needs at least 2. "
                  'Skipping.')
            report['groups'][mode] = {'status': 'SKIPPED_SINGLE_INGREDIENT',
                                      'tags': [m['tag'] for m in members]}
            continue
        print(f'\n[{mode}] ranking {len(members)} ingredients on held-out dev')
        from safetensors.torch import load_file
        ranked = []
        for m in members:
            tensors = load_file(str(m['_dir'] / 'trainable.safetensors'), device='cpu')
            score = evaluate(mode, read_tail, tensors)
            del tensors
            gc.collect()
            ranked.append((score['dev_exact'], m))
            print(f"   {m['tag']:<52} dev {score['dev_exact']:>3}/{len(dev)}")
        ranked.sort(key=lambda kv: -kv[0])

        kept = [ranked[0][1]]
        best = ranked[0][0]
        print(f"   start: {kept[0]['tag']} at {best}/{len(dev)}")
        trail = []
        for score, candidate in ranked[1:]:
            trial = kept + [candidate]
            tensors = average([m['_dir'] for m in trial])
            got = evaluate(mode, read_tail, tensors)['dev_exact']
            del tensors
            gc.collect()
            improved = got > best
            trail.append({'candidate': candidate['tag'], 'soup_dev': got, 'previous': best,
                          'accepted': improved})
            print(f"   + {candidate['tag']:<50} dev {got:>3}/{len(dev)} "
                  f"{'ACCEPT' if improved else 'reject'}")
            if improved:
                kept, best = trial, got

        final_tensors = average([m['_dir'] for m in kept]) if len(kept) > 1 else load_file(
            str(kept[0]['_dir'] / 'trainable.safetensors'), device='cpu')
        final = evaluate(mode, read_tail, final_tensors, panel=True)
        del final_tensors
        gc.collect()
        best_single = evaluate(mode, read_tail,
                               load_file(str(ranked[0][1]['_dir'] / 'trainable.safetensors'),
                                         device='cpu'), panel=True)
        report['groups'][mode] = {
            'status': 'OK', 'ingredients': [m['tag'] for m in members],
            'ranked': [{'tag': m['tag'], 'dev_exact': s} for s, m in ranked],
            'greedy_trail': trail, 'soup': [m['tag'] for m in kept],
            'soup_result': final, 'best_single_result': best_single,
            'soup_beats_best_single_on_panel':
                final.get('panel_correct', 0) > best_single.get('panel_correct', 0)}
        print(f"\n   SOUP  ({len(kept)} of {len(members)}): dev {final['dev_exact']}/{len(dev)} "
              f"| panel {final['panel_correct']}/150 kept {final['panel_kept']}/"
              f"{final['reference_n']} lost {final['panel_lost']} "
              f"(+{final['gained_wrong_answer']} real, +{final['gained_format_only']} fmt)")
        print(f"   BEST SINGLE {ranked[0][1]['tag']}: dev {best_single['dev_exact']}/{len(dev)} "
              f"| panel {best_single['panel_correct']}/150")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'MODEL_SOUP.json'
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print('\nwrote', path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
