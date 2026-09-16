"""Score a dense core on single-step state transitions from Forge RL v2.

The puzzle probe measured `hanoi_planning` and `code_breaking` at **literal zero** and read
that as the shape of the ceiling: not uniform mediocrity, but an absent multi-step search and
state-tracking capability. This instrument asks the obvious next question, which no probe in
this programme has asked: can the model do **one** step?

Each view supplies a `before` state and one `event`, and asks for the complete resulting state
as JSON. Prompts are ~150 tokens, so unlike every reasoning null in this programme this is not
confounded with prompt length. All 36,752 source labels were independently re-derived by
`oracle.replay_transition` and agreed (rrp005/PROCESS_TRANSITION_AUDIT.json).

    python experiments/rab001_exploration/process_probe.py --model Supa --label supa
    python experiments/rab001_exploration/process_probe.py \
        --model Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze --label opallo

**The floor is not zero and it is not uniform.** `copy_before` -- emit the supplied state
unchanged -- is answer-agnostic, reads only the prompt, and scores 31/128 on `ledger_replay`
because a non-posted status, a `scale` by 1 and an `add` of 0 are all no-ops. It scores 0/128
on `prerequisite_plan`. An aggregate quoted without the per-family floors repeats CL-040.
Floors are recomputed from the frozen views on every run and checked against the values
published in `forge-v2-prereg-001/PROCESS_PROBE_FLOORS.md` before any model is loaded.

Scoring separates presentation from arithmetic, because every instrument in this programme
scores by exact match and LONG_CONTEXT_RETENTION.md showed that costs real rows:

  * `strict`    -- the decoded text equals the canonical reference exactly;
  * `canonical` -- the decoded text parses as JSON and canonicalises to the reference.

`canonical` is the arithmetic measure. The gap between them is a formatting story, reported
as one rather than pooled into the headline.

Generation goes through the harness's own `DenseReference` forward and greedy loop with TF32
forced off, and renders with the same chat-template call the panel and puzzle probes use, so
these numbers sit on the same measurement path as every other number in this programme.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
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
from experiments.architecture_benchmark_001.source import ROOT, source_config     # noqa: E402
from experiments.architecture_benchmark_001.transfer import import_qwen_state     # noqa: E402
from experiments.rab001_exploration import hostbudget                             # noqa: E402

_saved_argv, sys.argv = sys.argv, sys.argv[:1]
try:
    from experiments.rab001_exploration.retention_dose_gpu import force_tf32_off  # noqa: E402
finally:
    sys.argv = _saved_argv

RECORDS = REPO.parent / 'rineya-dataset-retention-records'
VIEWS = RECORDS / 'process-transition-probe-001' / 'CANDIDATE_VIEWS.jsonl'
VIEWS_AUDIT = RECORDS / 'process-transition-probe-001' / 'AUDIT.json'
OUT = ROOT / 'evidence' / 'retention-design-v1'

# Published in forge-v2-prereg-001/PROCESS_PROBE_FLOORS.md BEFORE any model call.
PUBLISHED_FLOORS = {
    'ledger_replay': {'copy_before': 31, 'constant_majority': 1, 'n': 128},
    'register_machine': {'copy_before': 7, 'constant_majority': 1, 'n': 128},
    'prerequisite_plan': {'copy_before': 0, 'constant_majority': 9, 'n': 128},
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(',', ':'))


def binom_at_least(k: int, n: int, p: float) -> float:
    """Exact P(X >= k | n, p). Used one-sided against the family's own floor."""
    if k <= 0:
        return 1.0
    if p <= 0.0:
        return 0.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k, n + 1))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Two-sided exact interval by bisection on the exact binomial tail. No scipy."""
    lower = 0.0
    if k > 0:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            if binom_at_least(k, n, mid) > alpha / 2:
                hi = mid
            else:
                lo = mid
        lower = lo
    upper = 1.0
    if k < n:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = (lo + hi) / 2
            # P(X <= k) = 1 - P(X >= k+1)
            if 1 - binom_at_least(k + 1, n, mid) > alpha / 2:
                lo = mid
            else:
                hi = mid
        upper = hi
    return lower, upper


def load_trained(source: Path, device: str, checkpoint: Path):
    """Apply an MQ/RRP checkpoint over the frozen source, on the SAME path as the baseline.

    The baseline was measured through run._load, which converts the HF state dict into a
    DenseReference via import_qwen_state. Scoring a trained model through any other path
    would make the pre/post comparison a comparison of two implementations. So the trained
    tensors are overlaid onto the source HF state and the identical conversion is run.

    Every entry is verified against the sealed manifest before it is applied: an unsealed
    manifest, a status other than COMPLETE, an unknown parameter name, a shape mismatch or a
    tensor whose bytes do not hash to the recorded sha256 all refuse the load.
    """
    from safetensors.torch import load_file
    from experiments.model_quality_gpu_v1.integrity import unseal

    body = unseal(json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8")))
    if body.get("status") != "COMPLETE":
        raise SystemExit('checkpoint is not COMPLETE and must not be scored')
    state = load_file(str(source / "model.safetensors"), device="cpu")
    applied = 0
    for entry in body["entries"]:
        name, path = entry["name"], checkpoint / entry["file"]
        if name not in state:
            raise SystemExit('checkpoint names a parameter the source does not have: {}'.format(name))
        if sha256_file(path) != entry["sha256"]:
            raise SystemExit('checkpoint tensor bytes differ from the manifest: {}'.format(entry["file"]))
        trained = load_file(str(path), device="cpu")["parameter"]
        if list(trained.shape) != entry["shape"] or trained.shape != state[name].shape:
            raise SystemExit('checkpoint shape differs from the source: {}'.format(name))
        state[name] = trained.to(state[name].dtype)
        applied += 1
    raw = json.loads((source / "config.json").read_text(encoding="utf-8"))
    model = import_qwen_state(state, source_config(raw)).to(device=device,
                                                            dtype=torch.float32).eval()
    tokenizer = run._load_tokenizer(source)
    return model, tokenizer, {"path": str(checkpoint), "applied_tensors": applied,
                              "experiment": body.get("experiment"),
                              "progress": body.get("progress"),
                              "manifest_sha256": sha256_file(checkpoint / "manifest.json")}


def load_views() -> list[dict]:
    if not VIEWS.is_file():
        raise SystemExit('no frozen views: {}'.format(VIEWS))
    declared = json.loads(VIEWS_AUDIT.read_text(encoding='utf-8'))['views_sha256']
    actual = sha256_file(VIEWS)
    if actual != declared:
        raise SystemExit('views sha256 differs from its audit: {} != {}'.format(actual, declared))
    rows = [json.loads(l) for l in VIEWS.read_text(encoding='utf-8').splitlines() if l.strip()]
    for row in rows:
        if row['view_role'] != 'CANDIDATE_NOT_ADMITTED_OR_TRAINED':
            raise SystemExit('a view claims a role this probe does not grant')
        blob = json.dumps(row['prompt'])
        # Structural, not substring. A substring test is VACUOUS on prerequisite_plan, whose
        # reference is a small integer that coincides with digits of the event ("4" appears
        # inside {"add":4,...}) -- it fires on clean rows and would be trusted as a detector.
        # What actually matters is that the user content carries the two input keys and
        # nothing else, and that the label-bearing source id never appears.
        if row['source_id'] in blob:
            raise SystemExit('label-bearing source id leaked into the prompt: {}'.format(row['id']))
        if 'proposed_after' in blob or '"valid"' in blob:
            raise SystemExit('answer or label key leaked into the prompt: {}'.format(row['id']))
        if sorted(json.loads(row['prompt'][1]['content']).keys()) != ['before', 'event']:
            raise SystemExit('user content is not exactly before+event: {}'.format(row['id']))
    return rows


def floors_from(rows: list[dict]) -> dict:
    """Recomputed every run; a drift from the published values stops the probe."""
    out = {}
    for family in sorted({r['family'] for r in rows}):
        fam = [r for r in rows if r['family'] == family]
        copy_before = sum(
            1 for r in fam
            if canonical(json.loads(r['prompt'][1]['content'])['before']) == r['reference'])
        counts = collections.Counter(r['reference'] for r in fam)
        out[family] = {'copy_before': copy_before,
                       'constant_majority': counts.most_common(1)[0][1],
                       'n': len(fam)}
    if out != PUBLISHED_FLOORS:
        raise SystemExit('floors drifted from the published pre-registration:\n'
                         '  recomputed {}\n  published  {}'.format(out, PUBLISHED_FLOORS))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--checkpoint', type=Path,
                        help='an MQ/RRP checkpoint directory to apply over the frozen source')
    args = parser.parse_args(argv)

    rows = load_views()
    floors = floors_from(rows)
    views_sha = sha256_file(VIEWS)

    model_dir = (REPO / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    if not (model_dir / 'model.safetensors').is_file():
        raise SystemExit('no dense weights at {}'.format(model_dir))
    OUT.mkdir(parents=True, exist_ok=True)
    result_path = OUT / 'PROCESS_{}.json'.format(args.label)
    if result_path.exists():
        raise SystemExit('create-only result already exists: {}'.format(result_path))

    hostbudget.require(3.0, 'process probe of {}'.format(args.label))
    numerics = force_tf32_off()
    print(hostbudget.describe())
    print('TF32 forced off: {}'.format(numerics), flush=True)
    print('floors reproduce the pre-registration exactly: {}'.format(floors), flush=True)

    if args.checkpoint:
        model, tokenizer, checkpoint_identity = load_trained(model_dir, args.device,
                                                            args.checkpoint)
        print('applied checkpoint {} ({} tensors, {} updates)'.format(
            checkpoint_identity['experiment'], checkpoint_identity['applied_tensors'],
            (checkpoint_identity['progress'] or {}).get('updates')), flush=True)
    else:
        model, tokenizer, _ = run._load(model_dir, args.device)
        checkpoint_identity = None
    eos = run._eos_ids(model_dir, tokenizer)
    print('loaded {} on {} | {} views | views sha {}'.format(
        model_dir.name, args.device, len(rows), views_sha[:12]), flush=True)

    predictions, started = {}, time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            ids = tokenizer.apply_chat_template(
                row['prompt'], tokenize=True, add_generation_prompt=True,
                enable_thinking=False, return_tensors='pt').to(args.device)
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
            if index % 48 == 0:
                print('  {}/{} | {:.0f}s'.format(index, len(rows),
                                                 time.perf_counter() - started), flush=True)
    elapsed = time.perf_counter() - started
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    detail = []
    for row in rows:
        got = predictions[row['id']]
        text = got['prediction']
        strict = text == row['reference']
        canonical_ok, parsed_ok = False, False
        try:
            parsed = json.loads(text)
            parsed_ok = True
            canonical_ok = canonical(parsed) == row['reference']
        except (ValueError, TypeError):
            parsed_ok = False
        detail.append({'id': row['id'], 'family': row['family'], 'reference': row['reference'],
                       'prediction': text, 'strict': strict, 'canonical': canonical_ok,
                       'parsed': parsed_ok, 'terminated': got['terminated'],
                       'generated_tokens': got['generated_tokens']})

    per_family = {}
    for family in sorted({r['family'] for r in rows}):
        fam = [d for d in detail if d['family'] == family]
        n = len(fam)
        strict = sum(d['strict'] for d in fam)
        canon = sum(d['canonical'] for d in fam)
        # The floor that binds is the STRONGEST honest shortcut, not copy_before by default.
        # On prerequisite_plan the state is an 8-bit integer with 58 distinct values in 128,
        # so constant_majority (9) beats copy_before (0). Omitting the stronger predictor is
        # exactly the error RRP002_DEV_V2_FLOOR_CORRECTION had to withdraw.
        copy_k = floors[family]['copy_before']
        const_k = floors[family]['constant_majority']
        floor_k = max(copy_k, const_k)
        floor_p = floor_k / n
        lo, hi = clopper_pearson(canon, n)
        per_family[family] = {
            'n': n, 'strict_correct': strict, 'canonical_correct': canon,
            'parsed': sum(d['parsed'] for d in fam),
            'terminated': sum(d['terminated'] for d in fam),
            'copy_before_count': copy_k, 'constant_majority_count': const_k,
            'binding_floor_predictor': ('copy_before' if copy_k >= const_k
                                        else 'constant_majority'),
            'binding_floor_count': floor_k, 'binding_floor_rate': round(floor_p, 4),
            'canonical_rate': round(canon / n, 4),
            'presentation_only_gain': canon - strict,
            'exact_binomial_p_vs_floor_one_sided': binom_at_least(canon, n, floor_p),
            'clopper_pearson_95': [round(lo, 4), round(hi, 4)],
            'clears_floor': lo > floor_p,
        }

    total_n = len(rows)
    total_canon = sum(d['canonical'] for d in detail)
    total_strict = sum(d['strict'] for d in detail)
    total_floor = sum(max(floors[f]['copy_before'], floors[f]['constant_majority'])
                      for f in floors)

    payload = {
        'record': 'PROCESS_TRANSITION_PROBE',
        'schema': 1,
        'what_this_is': ('One-step state-transition generation from Forge RL v2 process labels. '
                         'Bounds single-step arithmetic/state precision. NOT process '
                         'verification, NOT multi-step reasoning, NOT a preservation or '
                         'release result.'),
        'label': args.label,
        'model_dir': str(model_dir),
        'checkpoint': checkpoint_identity,
        'device': args.device,
        'views_sha256': views_sha,
        'views': total_n,
        'distinct_parents': len({r['parent_id'] for r in rows}),
        'max_new_tokens': args.max_new_tokens,
        'implementation_digest': implementation_digest(),
        'numerics': numerics,
        'runtime': {'python': sys.version.split()[0], 'torch': torch.__version__},
        'wall_seconds': round(elapsed, 2),
        'aggregate': {
            'n': total_n, 'strict_correct': total_strict, 'canonical_correct': total_canon,
            'canonical_rate': round(total_canon / total_n, 4),
            'binding_floor_count': total_floor,
            'binding_floor_rate': round(total_floor / total_n, 4),
            'presentation_only_gain': total_canon - total_strict,
            'note': ('the aggregate is REPORTED, never the gate: the three families have '
                     'different binding floors (ledger_replay 0.2422 copy_before, '
                     'register_machine 0.0547 copy_before, prerequisite_plan 0.0703 '
                     'constant_majority)'),
        },
        'per_family': per_family,
        'decision_rule': ('pre-registered in forge-v2-prereg-001/PROCESS_PROBE_FLOORS.md: a core '
                          'clears a family only if the two-sided exact binomial 95% lower bound '
                          'on canonical_correct exceeds that family binding floor, where the '
                          'binding floor is the STRONGER of copy_before and constant_majority'),
        'floors_published_before_any_model_call': floors,
        'limitations': [
            'TRAIN-only derivative views; CANDIDATE_NOT_ADMITTED_OR_TRAINED, never for selection',
            'single-step competence does not establish multi-step search or planning',
            'prerequisite_plan positives cover applicable actions only',
            'exact-match scoring; strict vs canonical separates presentation from arithmetic',
        ],
        'optimizer_steps': 0,
        'quality_claim': False,
        'measured_utc': datetime.now(timezone.utc).isoformat(),
        'predictions': detail,
    }
    result_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n',
                           encoding='utf-8', newline='\n')

    print('\n{}: canonical {}/{} = {:.4f} (strict {}, binding floor {})'.format(
        args.label, total_canon, total_n, total_canon / total_n, total_strict, total_floor))
    for family, value in per_family.items():
        print('  {:<20} {:>3}/{} = {:.4f}  floor {:.4f} ({})  CI {}  {}'.format(
            family, value['canonical_correct'], value['n'], value['canonical_rate'],
            value['binding_floor_rate'], value['binding_floor_predictor'],
            value['clopper_pearson_95'],
            'CLEARS' if value['clears_floor'] else 'does not clear'))
    print('\nwrote {}'.format(result_path))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
