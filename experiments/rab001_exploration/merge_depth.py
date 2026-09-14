"""Cheap weight merges over saved depth-arm checkpoints. No training, no GPU.

Two merges are defined here and they are NOT the same operation. Keeping them apart matters,
because one of them is already predicted to fail.

  union      For blocks both arms trained, average. For blocks only the deeper arm trained,
             take the deeper arm's value unchanged. This preserves B28last20's early-block
             learning in full and averages only where the two actually disagree.

  delta      Treat the shallower arm as having a zero delta on blocks it never trained, then
             average. On those blocks that HALVES the deeper arm's delta, which is exactly
             alpha=0.5 interpolation toward the source. ALPHA_B28_PARTIAL_20260913.json already
             measured that path for full B28: retention does not come back (kept 47, 42, 40, 40,
             41 across alpha 0, .25, .5, .75, 1 while gains climb 0, 14, 23, 27, 32). So `delta`
             is expected to land between the two arms, and it is included as the control that
             makes `union` interpretable rather than as a candidate.

Both arms must be deltas from the SAME source, which the soup_key's source lineage pins. Only
trainable tensors are stored in a checkpoint, so "blocks the arm never trained" simply means
tensor names absent from its file.

    python experiments/rab001_exploration/merge_depth.py <tagA> <tagB> [--mode union|delta]

Writes outputs/arm-checkpoints/MERGE_<mode>_<tagA>+<tagB>/ (create-only) holding the merged
trainable tensors plus a manifest. Evaluating the result is a separate step -- use
retention_probe.py or arm_compare.py -- because a probe needs a model and this script never
loads one.

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
    manifest = directory / 'MANIFEST.json'
    weights = directory / 'trainable.safetensors'
    if not manifest.is_file() or not weights.is_file():
        raise SystemExit(f'no usable checkpoint at {directory}')
    body = json.loads(manifest.read_text(encoding='utf-8'))
    if body.get('completed') is not True:
        raise SystemExit(f'ingredient did not complete: {tag}')
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != body['sha256']:
        raise SystemExit(f'checkpoint hash mismatch, refusing to merge: {tag}')
    body['_dir'] = directory
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('tags', nargs=2)
    parser.add_argument('--mode', choices=('union', 'delta'), default='union')
    args = parser.parse_args()

    import torch
    from safetensors.torch import load_file, save_file

    a, b = (load(t) for t in args.tags)
    for key in ('mode', 'source_digest'):
        if a['soup_key'].get(key) != b['soup_key'].get(key):
            raise SystemExit(
                f'incompatible ingredients: soup_key.{key} differs '
                f'({a["soup_key"].get(key)!r} vs {b["soup_key"].get(key)!r}). These are not '
                'deltas from the same source in the same architecture and averaging them is '
                'undefined.')

    ta = load_file(str(a['_dir'] / 'trainable.safetensors'), device='cpu')
    tb = load_file(str(b['_dir'] / 'trainable.safetensors'), device='cpu')
    shared = sorted(set(ta) & set(tb))
    only_a = sorted(set(ta) - set(tb))
    only_b = sorted(set(tb) - set(ta))
    if not shared:
        raise SystemExit('ingredients share no trainable tensors; nothing to merge')
    print(f'{args.tags[0]}: {len(ta)} tensors | {args.tags[1]}: {len(tb)} tensors')
    print(f'  shared {len(shared)} | only-{args.tags[0]} {len(only_a)} | '
          f'only-{args.tags[1]} {len(only_b)}')

    merged, provenance = {}, {}
    for name in shared:
        if ta[name].shape != tb[name].shape:
            raise SystemExit(f'shape mismatch on shared tensor {name}')
        merged[name] = ((ta[name].to(torch.float64) + tb[name].to(torch.float64)) / 2
                        ).to(torch.float32)
        provenance[name] = 'averaged'
    for name, source, owner in [(n, ta, args.tags[0]) for n in only_a] + \
                               [(n, tb, args.tags[1]) for n in only_b]:
        if args.mode == 'union':
            merged[name] = source[name].clone()
            provenance[name] = f'taken from {owner} (other arm never trained it)'
        else:
            # The absent arm's value IS the source value, so averaging halves this delta.
            merged[name] = (source[name].to(torch.float64) / 2).to(torch.float32)
            provenance[name] = f'halved toward source (alpha=0.5 on {owner} only)'

    destination = CKPT / f'MERGE_{args.mode}_{args.tags[0]}+{args.tags[1]}'
    if destination.exists():
        raise SystemExit(f'merge directory already exists (create-only): {destination}')
    destination.mkdir(parents=True, exist_ok=False)
    out = destination / 'trainable.safetensors'
    save_file(merged, str(out))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()

    names_hash = hashlib.sha256('\n'.join(sorted(merged)).encode()).hexdigest()
    (destination / 'MANIFEST.json').write_text(json.dumps({
        'schema': 1, 'evidence_kind': 'exploratory_merged_weights_only',
        'note': 'Weight merge of two depth arms. NOT a trained model and NOT a soup: the two '
                'ingredients train different tensor sets, so this is a union/task-arithmetic '
                'merge rather than uniform averaging of like-for-like fits. Exploratory only: '
                'no receipt, no gate, no release. Evaluate with retention_probe.py.',
        'merge_mode': args.mode,
        'ingredients': [
            {'tag': a['tag'], 'arm': a.get('arm'), 'sha256': a['sha256'],
             'tensors': len(ta), 'parameters': a['parameters']},
            {'tag': b['tag'], 'arm': b.get('arm'), 'sha256': b['sha256'],
             'tensors': len(tb), 'parameters': b['parameters']}],
        'tensors': len(merged),
        'parameters': int(sum(t.numel() for t in merged.values())),
        'shared_tensors': len(shared), 'only_first': len(only_a), 'only_second': len(only_b),
        'bytes': out.stat().st_size, 'sha256': sha, 'dtype': 'float32',
        'soup_key': {'mode': a['soup_key']['mode'], 'read_tail': a['soup_key'].get('read_tail'),
                     'source_digest': a['soup_key']['source_digest'],
                     'tensor_names_sha256': names_hash},
        'tensor_provenance': provenance,
        'delta_mode_caveat':
            'In delta mode, blocks only one arm trained are halved toward the source, which is '
            'alpha=0.5 interpolation. ALPHA_B28_PARTIAL_20260913.json measured that path for '
            'full B28 and retention did not recover, so delta mode is a control, not a '
            'candidate.',
        'licenses_release': False, 'licenses_efficacy_claim': False,
    }, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    print(f'\nwrote {destination}')
    print(f'  {len(merged)} tensors, {sum(t.numel() for t in merged.values()):,} parameters, '
          f'{out.stat().st_size / 2**30:.2f} GiB')
    print(f'  sha256 {sha[:32]}...')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
