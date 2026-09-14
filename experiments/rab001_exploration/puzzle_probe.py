"""Score a dense core on Rineya Puzzle Reasoning 51k, to separate capacity from task distance.

RAB-001 concluded that this 596M model has a **capacity** ceiling on reasoning, because
nothing moved the Reasoning Forge -- not schedule, depth, dose, architecture, weight
averaging, or an 11.5x dose of purpose-built reasoning data. That conclusion has a confound
nobody controlled. Forge prompts run about 1,400 tokens against training rows of about 128,
so "cannot reason" and "cannot reason *over a long record*" were never separated.

This instrument separates them. Puzzle prompts are ~90 tokens, every answer is a short
string, and every row is solver-verified to have exactly one solution. If the model scores
meaningfully here, the ceiling was distance, not capacity, and the capacity conclusion has
to be withdrawn.

    python experiments/rab001_exploration/puzzle_probe.py --build-sample
    python experiments/rab001_exploration/puzzle_probe.py --model Supa --label supa
    python experiments/rab001_exploration/puzzle_probe.py \
        --model Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze --label opallo

`--build-sample` freezes 450 rows -- 30 per family across all 15 families -- to a
create-only file with a sha256, and every scoring run checks that hash. Sampling after
seeing a score is how an instrument stops being one.

**Chance is not zero here and the artifact says so.** `binary_class` and `boolean` families
are two-way and `three_way_logic` is three-way, so a model that guesses scores well above
nothing. Two floors are computed from the frozen sample before any model runs:

  * `uniform_chance` -- 1/k over the distinct accepted answers in that family;
  * `majority_baseline` -- what always answering that family's single most common answer
    would score, which is the stronger and more honest floor.

A family's result is only interesting above its own floor, and the per-family table is
published so that reading is available rather than implied by an aggregate.

Generation goes through the harness's own `DenseReference` forward and greedy loop with
TF32 forced off, and renders with the same chat-template call the panel probe uses, so
these numbers sit on the same measurement path as every other number in this programme.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

import torch

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.architecture_benchmark_001 import run                            # noqa: E402
from experiments.architecture_benchmark_001.gates import implementation_digest    # noqa: E402
from experiments.architecture_benchmark_001.source import ROOT                    # noqa: E402
from experiments.rab001_exploration import hostbudget                             # noqa: E402

_saved_argv, sys.argv = sys.argv, sys.argv[:1]
try:
    from experiments.rab001_exploration.retention_dose_gpu import force_tf32_off  # noqa: E402
finally:
    sys.argv = _saved_argv

CORPUS = REPO / 'Rineya-Puzzle-Reasoning-51k' / 'data'
OUT = ROOT / 'evidence' / 'retention-design-v1'
SAMPLE = OUT / 'PUZZLE_SAMPLE_test450_seed17.jsonl'

_PUNCT = re.compile(r'[^\w\s]')


def normalize(value: str) -> str:
    """The corpus's own `casefold_whitespace_punctuation` rule."""
    return ' '.join(_PUNCT.sub(' ', str(value)).casefold().split())


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_sample(per_family: int, seed: int) -> None:
    if SAMPLE.exists():
        raise SystemExit(f'sample already exists (create-only): {SAMPLE}\n'
                         f'Resampling after seeing a score is how an instrument stops '
                         f'being one.')
    rows = [json.loads(l) for l in (CORPUS / 'test.jsonl').read_text(encoding='utf-8')
            .splitlines() if l.strip()]
    train_sig = {r['structural_signature'] for r in
                 (json.loads(l) for l in (CORPUS / 'train.jsonl').read_text(encoding='utf-8')
                  .splitlines() if l.strip())}
    by_family = collections.defaultdict(list)
    for row in rows:
        by_family[row['family']].append(row)
    rng = random.Random(seed)
    chosen = []
    for family in sorted(by_family):
        pool = sorted(by_family[family], key=lambda r: r['id'])
        if len(pool) < per_family:
            raise SystemExit(f'family {family} has only {len(pool)} test rows')
        chosen.extend(rng.sample(pool, per_family))
    chosen.sort(key=lambda r: r['id'])
    leaked = [r['id'] for r in chosen if r['structural_signature'] in train_sig]
    if leaked:
        raise SystemExit(f'CONTAMINATION: {len(leaked)} sampled test rows share a structural '
                         f'signature with TRAIN, e.g. {leaked[:3]}')
    SAMPLE.parent.mkdir(parents=True, exist_ok=True)
    SAMPLE.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in chosen),
                      encoding='utf-8')
    print(f'wrote {SAMPLE}')
    print(f'  {len(chosen)} rows, {len(by_family)} families x {per_family}, seed {seed}')
    print(f'  sha256 {sha256_file(SAMPLE)}')
    print(f'  no sampled row shares a structural signature with TRAIN')
    print('\n' + floors_table(chosen))


def floors(rows: list[dict]) -> dict:
    """Per-family chance floors, computed from the frozen sample before any model runs."""
    out = {}
    by_family = collections.defaultdict(list)
    for row in rows:
        by_family[row['family']].append(row)
    for family, group in sorted(by_family.items()):
        answers = collections.Counter(normalize(r['final_answer']) for r in group)
        distinct = len(answers)
        top = answers.most_common(1)[0]
        out[family] = {
            'items': len(group),
            'answer_type': collections.Counter(r['answer_type'] for r in group)
                           .most_common(1)[0][0],
            'distinct_answers': distinct,
            'uniform_chance': 1 / distinct,
            'majority_baseline': top[1] / len(group),
            'majority_answer': top[0],
        }
    return out


def floors_table(rows: list[dict]) -> str:
    data = floors(rows)
    lines = [f"{'family':26s} {'type':16s} {'n':>3s} {'distinct':>8s} "
             f"{'uniform':>8s} {'majority':>9s}"]
    for family, info in data.items():
        lines.append(f"{family:26s} {info['answer_type']:16s} {info['items']:3d} "
                     f"{info['distinct_answers']:8d} {info['uniform_chance']:8.3f} "
                     f"{info['majority_baseline']:9.3f}")
    overall = sum(i['majority_baseline'] * i['items'] for i in data.values()) / \
        sum(i['items'] for i in data.values())
    lines.append(f"{'ALL':26s} {'':16s} {sum(i['items'] for i in data.values()):3d} "
                 f"{'':8s} {'':8s} {overall:9.3f}")
    return '\n'.join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-sample', action='store_true')
    parser.add_argument('--per-family', type=int, default=30)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--model')
    parser.add_argument('--label')
    parser.add_argument('--checkpoint', help='arm-checkpoints tag to apply over the model')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=24)
    args = parser.parse_args()

    if args.build_sample:
        build_sample(args.per_family, args.seed)
        return 0
    if not args.model or not args.label:
        raise SystemExit('--model and --label are required unless --build-sample')
    if not SAMPLE.is_file():
        raise SystemExit(f'no frozen sample; run --build-sample first: {SAMPLE}')

    rows = [json.loads(l) for l in SAMPLE.read_text(encoding='utf-8').splitlines() if l.strip()]
    sample_sha = sha256_file(SAMPLE)
    model_dir = (REPO / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    if not (model_dir / 'model.safetensors').is_file():
        raise SystemExit(f'no dense weights at {model_dir}')
    result_path = OUT / f'PUZZLE_{args.label}.json'
    if result_path.exists():
        raise SystemExit(f'create-only result already exists: {result_path}')

    hostbudget.require(3.0, f'puzzle probe of {args.label}')
    numerics = force_tf32_off()
    print(hostbudget.describe())
    print(f'TF32 forced off: {numerics}', flush=True)

    model, tokenizer, _ = run._load(model_dir, args.device)
    eos = run._eos_ids(model_dir, tokenizer)
    applied = None
    if args.checkpoint:
        from safetensors.torch import load_file
        ckpt = ROOT / 'outputs' / 'arm-checkpoints' / args.checkpoint
        manifest = json.loads((ckpt / 'MANIFEST.json').read_text(encoding='utf-8'))
        if not manifest.get('completed'):
            raise SystemExit(f'{args.checkpoint} did not complete and must not be scored')
        state = model.state_dict()
        trained = load_file(str(ckpt / 'trainable.safetensors'), device='cpu')
        with torch.no_grad():
            for name, value in trained.items():
                state[name].copy_(value.to(state[name].dtype).to(state[name].device))
        applied = {'tag': manifest['canonical_tag'], 'tensors': len(trained)}
        print(f'  applied checkpoint {applied["tag"]} ({applied["tensors"]} tensors)')
    print(f'loaded {model_dir.name} on {args.device} | {len(rows)} rows | '
          f'sample sha {sample_sha[:12]}', flush=True)

    predictions, started = {}, time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            ids = tokenizer.apply_chat_template(
                [{'role': 'user', 'content': row['prompt']}], tokenize=True,
                add_generation_prompt=True, enable_thinking=False,
                return_tensors='pt').to(args.device)
            generated, past, current, terminated = [], None, ids, False
            for _ in range(args.max_new_tokens):
                step = model(current, past=past, use_cache=True, last_position_only=True)
                past = step.cache
                token = int(step.logits[0, -1].argmax())
                if token in eos:
                    terminated = True
                    break
                generated.append(token)
                current = torch.tensor([[token]], dtype=torch.long, device=args.device)
            del past
            predictions[row['id']] = {
                'prediction': tokenizer.decode(generated, skip_special_tokens=True).strip(),
                'terminated': terminated, 'generated_tokens': len(generated),
                'input_tokens': int(ids.shape[-1]),
            }
            if index % 60 == 0:
                print(f'  {index}/{len(rows)} | {time.perf_counter() - started:.0f}s',
                      flush=True)
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    scored, by_family = {}, collections.defaultdict(lambda: {'items': 0, 'hits': 0})
    for row in rows:
        record = predictions[row['id']]
        accepted = {normalize(a) for a in (row.get('accepted_answers')
                                           or [row['final_answer']])}
        first = record['prediction'].splitlines()[0] if record['prediction'] else ''
        hit = normalize(first) in accepted
        scored[row['id']] = hit
        by_family[row['family']]['items'] += 1
        by_family[row['family']]['hits'] += int(hit)

    floor = floors(rows)
    total = sum(scored.values())
    print(f"\n{'family':26s} {'n':>3s} {'hits':>5s} {'rate':>6s} {'majority':>9s} {'delta':>7s}")
    for family in sorted(by_family):
        info, base = by_family[family], floor[family]['majority_baseline']
        rate = info['hits'] / info['items']
        print(f"{family:26s} {info['items']:3d} {info['hits']:5d} {rate:6.3f} "
              f"{base:9.3f} {rate - base:+7.3f}")
    overall_floor = sum(f['majority_baseline'] * f['items'] for f in floor.values()) / len(rows)
    print(f"{'ALL':26s} {len(rows):3d} {total:5d} {total/len(rows):6.3f} "
          f"{overall_floor:9.3f} {total/len(rows) - overall_floor:+7.3f}")

    record = {
        'schema': 1,
        'evidence_kind': 'exploratory_puzzle_reasoning_probe_only',
        'note': 'Rineya Puzzle Reasoning 51k test sample, frozen before any model ran. '
                'Separates reasoning capacity from task distance: prompts are ~90 tokens '
                'against the Reasoning Forge\'s ~1,400, so a score here is not confounded '
                'with long-record handling. Chance is not zero and the per-family floors '
                'below were computed from the frozen sample before scoring.',
        'label': args.label,
        'model_directory': str(model_dir.relative_to(REPO)) if model_dir.is_relative_to(REPO)
                           else str(model_dir),
        'checkpoint_applied': applied,
        'sample_path': str(SAMPLE.relative_to(REPO)),
        'sample_sha256': sample_sha,
        'rows': len(rows),
        'exact': total,
        'exact_rate': total / len(rows),
        'majority_baseline_rate': overall_floor,
        'delta_over_majority_baseline': total / len(rows) - overall_floor,
        'terminated': sum(1 for p in predictions.values() if p['terminated']),
        'by_family': {k: {**v, 'rate': v['hits'] / v['items'], **floor[k]}
                      for k, v in sorted(by_family.items())},
        'per_row_correct': scored,
        'predictions': predictions,
        'max_new_tokens': args.max_new_tokens,
        'device': args.device, 'numerics': numerics,
        'implementation_digest': implementation_digest(),
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - started,
        'licenses_efficacy_claim': False,
        'licenses_release': False,
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n',
                           encoding='utf-8')
    print(f'\n  wrote {result_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
