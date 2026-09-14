"""Uniform weight average over checkpoints that share a tensor set. No training, no selection.

`model_soup.py` refuses to run because its selection set is quarantined, and that refusal is
correct: greedy soup CHOOSES which ingredients to keep, so it needs a held-out set, and
selecting on a contaminated one would fit the soup to the test set.

A uniform average over a FIXED set chooses nothing. There are no ingredients to rank, so no
held-out set is required and the quarantine is irrelevant. That is the whole reason this exists
separately rather than as a flag on the other script -- the distinction is the guard.

    python experiments/rab001_exploration/uniform_soup.py <tag> <tag> [tag ...] [--label NAME]

Every ingredient must share an identical tensor-name set and the same soup_key mode, read_tail
and source lineage; anything else is refused rather than coerced. Accumulates in float64 and
streams one checkpoint at a time, so N ingredients cost one resident copy rather than N.

Writes outputs/arm-checkpoints/SOUP_<label>/ (create-only). Evaluating the result is a separate
step -- this never loads a model.

Exploratory. Outside the hashed root: no receipt, no gate, no efficacy claim, no release.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.architecture_benchmark_001.source import ROOT            # noqa: E402

CKPT = ROOT / 'outputs' / 'arm-checkpoints'


def load(tag: str) -> dict:
    directory = CKPT / tag
    manifest, weights = directory / 'MANIFEST.json', directory / 'trainable.safetensors'
    if not manifest.is_file() or not weights.is_file():
        raise SystemExit(f'no usable checkpoint at {directory}')
    body = json.loads(manifest.read_text(encoding='utf-8'))
    if body.get('completed') is not True:
        raise SystemExit(f'ingredient did not complete: {tag}')
    if hashlib.sha256(weights.read_bytes()).hexdigest() != body['sha256']:
        raise SystemExit(f'checkpoint hash mismatch, refusing to average: {tag}')
    body['_dir'] = directory
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags', nargs='+')
    parser.add_argument('--label', default=None)
    args = parser.parse_args()
    if len(args.tags) < 2:
        raise SystemExit('a soup needs at least two ingredients')

    import torch
    from safetensors.torch import load_file, save_file

    parts = [load(t) for t in args.tags]
    first = parts[0]['soup_key']
    for p in parts[1:]:
        for key in ('mode', 'read_tail', 'source_digest', 'tensor_names_sha256'):
            if p['soup_key'].get(key) != first.get(key):
                raise SystemExit(
                    f'incompatible ingredients: soup_key.{key} differs between '
                    f'{parts[0]["tag"]} and {p["tag"]}. Averaging is undefined across different '
                    'tensor sets, architectures or source lineages.')

    total, n = None, len(parts)
    for p in parts:
        tensors = load_file(str(p['_dir'] / 'trainable.safetensors'), device='cpu')
        if total is None:
            total = {k: v.to(torch.float64) for k, v in tensors.items()}
        else:
            if set(tensors) != set(total):
                raise SystemExit(f'tensor-name mismatch in {p["tag"]}')
            for k, v in tensors.items():
                total[k] += v.to(torch.float64)
        del tensors
        print(f'  accumulated {p["tag"]}', flush=True)
    merged = {k: (v / n).to(torch.float32) for k, v in total.items()}
    del total

    label = args.label or f'{first["mode"]}_x{n}'
    destination = CKPT / f'SOUP_{label}'
    if destination.exists():
        raise SystemExit(f'soup directory already exists (create-only): {destination}')
    destination.mkdir(parents=True, exist_ok=False)
    out = destination / 'trainable.safetensors'
    save_file(merged, str(out))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    (destination / 'MANIFEST.json').write_text(json.dumps({
        'schema': 1, 'evidence_kind': 'exploratory_uniform_soup_only',
        'note': 'Uniform average over a FIXED ingredient set. No selection was performed, so no '
                'held-out set was consulted and the splits_onpanel quarantine does not apply. '
                'Exploratory: not a released model, opens no gate, carries no receipt.',
        'ingredients': [{'tag': p['tag'], 'sha256': p['sha256'], 'seed': p.get('seed'),
                         'headline': p.get('headline')} for p in parts],
        'ingredient_count': n,
        'tensors': len(merged), 'parameters': int(sum(t.numel() for t in merged.values())),
        'bytes': out.stat().st_size, 'sha256': sha, 'dtype': 'float32',
        'soup_key': dict(first),
        'completed': True,
        'arm': parts[0].get('arm'), 'lr': parts[0].get('lr'),
        'schedule': parts[0].get('schedule'), 'token_budget': parts[0].get('token_budget'),
        'reconstruct': parts[0].get('reconstruct'),
        'licenses_release': False, 'licenses_efficacy_claim': False,
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    print(f'\nwrote {destination}')
    print(f'  {n} ingredients, {len(merged)} tensors, '
          f'{sum(t.numel() for t in merged.values()):,} parameters, '
          f'{out.stat().st_size / 2**30:.2f} GiB')
    print(f'  sha256 {sha[:32]}...')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
