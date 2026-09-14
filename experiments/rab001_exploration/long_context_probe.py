"""Score a dense core on the Rineya Long-Context Demonstration Suite, on this host's runtime.

Supa is a long-context core; 161/256 on this suite is the number the family roster keeps it
for. RAB-001 retrained 20 of its 28 blocks on 5,233 tokens of ~128-token conversational rows
and never once checked what that did to a 1,000-word context. This closes that gap.

    python experiments/rab001_exploration/long_context_probe.py --model Supa --label supa
    python experiments/rab001_exploration/long_context_probe.py \
        --model Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze --label opallo

The suite ships its own runner, `Rineya_Long_Context_Demo_v1/scripts/run_local_hf.py`, and
this is deliberately not it. That runner passes `device_map` unconditionally, which this
Transformers build refuses without `accelerate`, and installing a package into the pinned
cu126 environment mid-programme is a worse trade than writing forty lines: every frozen
receipt in this experiment was produced under the current pin.

So generation goes through the harness path instead -- the same `DenseReference` forward and
the same greedy loop that produced every other number here, with TF32 forced off. Rendering
is copied from the suite's `render_chat` (chat template, `add_generation_prompt=True`,
`enable_thinking=False`, `add_special_tokens=False`) so the prompts are byte-identical to
what the suite's own runner would have built, and the output file carries the suite's field
names so its own `evaluate_predictions.py` scores it unmodified.

Two deliberate differences from the published 2026-08-26 evidence, both recorded in the
artifact rather than absorbed:

  * **Transformers 4.51.3, not 5.14.1.** The published 161/256 is therefore *not* the
    comparator. Run the base model through this script too and compare against that.
  * **EOS is the full official list** from `generation_config.json`, not the tokenizer's
    single `eos_token_id`. This is what the rest of RAB-001 uses. Both models get it.

Prompts are never truncated. A row that will not fit is recorded as skipped, with its token
count, because silently shortening a long-context prompt deletes the evidence the question
is about -- and a skipped row that is quietly scored as wrong is a score for the harness,
not the model.
"""
from __future__ import annotations

import argparse
import json
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

# retention_dose_gpu parses sys.argv at import time, so it is imported the way arm_dose
# imports it. force_tf32_off lives there and is the proven one; forking it would risk two
# definitions of "TF32 is off" in a programme whose numbers depend on exactly one.
_saved_argv, sys.argv = sys.argv, sys.argv[:1]
try:
    from experiments.rab001_exploration.retention_dose_gpu import force_tf32_off  # noqa: E402
finally:
    sys.argv = _saved_argv

SUITE = REPO / 'Rineya_Long_Context_Demo_v1'
OUT = ROOT / 'evidence' / 'retention-design-v1'


def render(tokenizer, row: dict) -> str:
    """Byte-identical to the suite's `render_chat` for system+user rows with a template."""
    messages = [{'role': str(m['role']), 'content': str(m['content'])}
                for m in row['messages']
                if isinstance(m, dict) and m.get('role') in {'system', 'user'}
                and 'content' in m]
    if not messages:
        raise SystemExit(f'row {row.get("id")} has no system or user turn')
    return tokenizer.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='dense core directory')
    parser.add_argument('--label', required=True, help='short name for the artifact')
    parser.add_argument('--checkpoint', help='arm-checkpoints tag to apply over the model, so '
                                             'an arm can be scored without materializing it')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-new-tokens', type=int, default=24)
    parser.add_argument('--max-context', type=int, default=4096)
    parser.add_argument('--rows', type=int, help='limit for smoke tests')
    args = parser.parse_args()

    model_dir = (REPO / args.model) if not Path(args.model).is_absolute() else Path(args.model)
    if not (model_dir / 'model.safetensors').is_file():
        raise SystemExit(f'no dense weights at {model_dir}')
    result_path = OUT / f'LONGCTX_{args.label}.json'
    if result_path.exists():
        raise SystemExit(f'create-only result already exists: {result_path}')

    rows = [json.loads(l) for l in
            (SUITE / 'data' / 'all_prompts.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip()]
    key = {json.loads(l)['id']: json.loads(l) for l in
           (SUITE / 'data' / 'all_answer_key.jsonl').read_text(encoding='utf-8').splitlines()
           if l.strip()}
    if set(key) != {r['id'] for r in rows}:
        raise SystemExit('answer key does not cover exactly the prompt rows')
    if args.rows:
        rows = rows[:args.rows]

    hostbudget.require(3.0, f'long-context probe of {model_dir.name}')
    numerics = force_tf32_off()
    print(f'{hostbudget.describe()}')
    print(f'TF32 forced off: {numerics}', flush=True)

    model, tokenizer, _ = run._load(model_dir, args.device)
    eos = run._eos_ids(model_dir, tokenizer)
    applied = None
    if args.checkpoint:
        from safetensors.torch import load_file
        ckpt = ROOT / 'outputs' / 'arm-checkpoints' / args.checkpoint
        manifest = json.loads((ckpt / 'MANIFEST.json').read_text(encoding='utf-8'))
        if not manifest.get('completed'):
            raise SystemExit(f'{args.checkpoint} did not complete; its recorded score is its '
                             f'step-0 probe, which is the untrained source, and it must not '
                             f'be scored')
        state = model.state_dict()
        trained = load_file(str(ckpt / 'trainable.safetensors'), device='cpu')
        with torch.no_grad():
            for name, value in trained.items():
                state[name].copy_(value.to(state[name].dtype).to(state[name].device))
        applied = {'tag': manifest['canonical_tag'], 'tensors': len(trained),
                   'sha256': manifest['sha256']}
        print(f'  applied checkpoint {applied["tag"]} ({applied["tensors"]} tensors)')
    print(f'loaded {model_dir.name} on {args.device} | eos {sorted(eos)} | '
          f'{len(rows)} rows', flush=True)

    predictions, skipped, started = [], 0, time.perf_counter()
    with torch.inference_mode():
        for index, row in enumerate(rows, 1):
            text = render(tokenizer, row)
            ids = tokenizer(text, return_tensors='pt',
                            add_special_tokens=False)['input_ids'].to(args.device)
            n = int(ids.shape[-1])
            if n + args.max_new_tokens > args.max_context:
                # Recorded, never truncated: the needle may be in the part that would go.
                skipped += 1
                predictions.append({'id': row['id'], 'status': 'skipped_context_overflow',
                                    'prediction': '', 'input_tokens': n,
                                    'context_limit': args.max_context})
                continue
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
            predictions.append({
                'id': row['id'], 'status': 'ok',
                'prediction': tokenizer.decode(generated, skip_special_tokens=True).strip(),
                'input_tokens': n, 'generated_tokens': len(generated),
                'terminated': terminated, 'decoding': 'greedy', 'seed': 17,
            })
            if index % 32 == 0:
                print(f'  {index}/{len(rows)} | {time.perf_counter() - started:.0f}s',
                      flush=True)
    del model
    if args.device == 'cuda':
        torch.cuda.empty_cache()

    # Score with the suite's own acceptance rule: strict exact against `accepted_answers`.
    def normalize(value: str) -> str:
        return ' '.join(str(value).strip().lower().replace('.', ' ').split())

    scored, by = {}, {}
    for record in predictions:
        target = key[record['id']]
        accepted = target.get('accepted_answers') or [target['answer']]
        first = record['prediction'].splitlines()[0] if record['prediction'] else ''
        hit = normalize(first) in {normalize(a) for a in accepted}
        scored[record['id']] = hit
        for stratum in ('split', 'variant', 'task_type', 'needle_profile', 'difficulty'):
            bucket = by.setdefault(stratum, {}).setdefault(target[stratum],
                                                           {'items': 0, 'hits': 0})
            bucket['items'] += 1
            bucket['hits'] += int(hit)

    # Abstention is measured separately from accuracy. A model can stop emitting the UNKNOWN
    # token while still recognising the answer is absent -- Opallo answers "The current
    # evidence does not identify the cause." -- and "stopped abstaining" and "stopped saying
    # the word the scorer wants" are different failures with different fixes.
    by_id = {p['id']: p for p in predictions}
    unanswerable = [i for i in by_id if key[i]['variant'] == 'unanswerable']
    says_unknown = sum(1 for i in unanswerable
                       if 'unknown' in normalize(by_id[i]['prediction']))

    total = sum(scored.values())
    showcase = sum(1 for i, v in scored.items() if v and key[i]['split'] == 'showcase')
    stability = sum(1 for i, v in scored.items() if v and key[i]['split'] == 'stability_test')
    record = {
        'schema': 1,
        'evidence_kind': 'exploratory_long_context_retention_probe_only',
        'note': 'Rineya Long-Context Demonstration Suite scored through the RAB-001 harness '
                'forward path, not the suite runner, because that runner needs accelerate and '
                'the cu126 environment is pinned. Rendering, decoding and acceptance match '
                'the suite. Transformers 4.51.3 here vs 5.14.1 for the published 161/256, so '
                'the published figure is NOT the comparator -- compare against a base-model '
                'run from this same script.',
        'label': args.label,
        'model_directory': str(model_dir.relative_to(REPO)) if model_dir.is_relative_to(REPO)
                           else str(model_dir),
        'checkpoint_applied': applied,
        'rows_total': len(rows),
        'rows_skipped_context_overflow': skipped,
        'max_context': args.max_context, 'max_new_tokens': args.max_new_tokens,
        'device': args.device, 'numerics': numerics,
        'transformers': __import__('importlib.metadata', fromlist=['version'])
                        .version('transformers'),
        'implementation_digest': implementation_digest(),
        'measured_at_utc': datetime.now(timezone.utc).isoformat(),
        'wall_seconds': time.perf_counter() - started,
        'unanswerable_rows': len(unanswerable),
        'unanswerable_correct': sum(scored[i] for i in unanswerable),
        'says_unknown_on_unanswerable': says_unknown,
        'all256_hits': total,
        'showcase64_hits': showcase,
        'stability192_hits': stability,
        'by_stratum': by,
        'per_row_correct': scored,
        'predictions': {p['id']: p for p in predictions},
        'licenses_efficacy_claim': False,
        'licenses_release': False,
    }
    result_path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n',
                           encoding='utf-8')
    # Denominators come from the rows actually scored. Hard-coding 64/192 printed a
    # confident "showcase 0/64" for an 8-row smoke that never touched 64 showcase rows.
    n_showcase = sum(1 for i in scored if key[i]['split'] == 'showcase')
    n_stability = sum(1 for i in scored if key[i]['split'] == 'stability_test')
    print(f'\n{args.label}: all {total}/{len(rows)} | showcase {showcase}/{n_showcase} | '
          f'stability {stability}/{n_stability} | skipped {skipped}')
    print(f'  abstention: says UNKNOWN on {says_unknown}/{len(unanswerable)} unanswerable rows (Supa 13, Opallo 2)')
    print(f'  wrote {result_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
