"""Is the source model even off the floor on Rineya Reasoning Forge?

Before investing in a new benchmark, two things have to be true: it must not overlap the
existing panel, and the model must score somewhere a difference could show. The first is
settled -- 0/150 verbatim and 0/150 digit-blinded against the panel, on both train and
test_iid. This measures the second.

    python experiments/rab001_exploration/forge_probe.py [--split test_iid] [--rows 200]
                                                         [--max-prompt 2048] [--device cuda]
                                                         [--checkpoint TAG] [--label NAME]

Why GPU by default, when every panel probe runs on the CPU: the panel's reference is a frozen
CPU measurement, so a panel probe must match the device that produced it or a 1e-4 FP32
difference can flip a greedy argmax and manufacture a retention loss. This benchmark has no
frozen reference -- whatever device is used here establishes it -- so the constraint does not
apply, and GPU prefill on 2k-token prompts is roughly an order of magnitude faster.

Scoring is exact match on the normalised final line: strip, casefold, collapse whitespace, drop
a trailing period. Forge answers are short and categorical or numeric (median 8 characters, 481
distinct over 800 rows), so this is a fair reading rather than the typed-grammar scorer the
arithmetic panel needs. It is reported alongside a looser containment score so the gap between
"knows it" and "formats it" stays visible, which is the confound that cost this programme a
correct headline once already.

Long rows are skipped, not truncated: truncating a record-collection prompt can silently remove
the record the question depends on, turning a reasoning test into a retrieval-failure test.
The skipped count is reported.

Exploratory. Outside the hashed root: no receipt, no gate, no efficacy claim, no release.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import hostbudget                                                         # noqa: E402
from experiments.architecture_benchmark_001 import run                     # noqa: E402
from experiments.architecture_benchmark_001.gates import implementation_digest  # noqa: E402
from experiments.architecture_benchmark_001.source import ROOT             # noqa: E402

FORGE = REPO / 'rineya_reasoning_forge_v1' / 'data'
OUT = ROOT / 'evidence' / 'retention-design-v1'
SRC = REPO / 'Supa'
if not (SRC / 'model.safetensors').is_file():
    SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
               r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')

_WS = re.compile(r'\s+')


def normalise(text: str) -> str:
    return _WS.sub(' ', text.strip().casefold()).rstrip('.').strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', default='test_iid')
    parser.add_argument('--rows', type=int, default=200)
    parser.add_argument('--max-prompt', type=int, default=2048)
    parser.add_argument('--max-new', type=int, default=24)
    parser.add_argument('--seed', type=int, default=17)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint', default=None,
                        help='arm-checkpoints tag to overlay onto the source (default: source)')
    parser.add_argument('--label', default=None)
    args = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    hostbudget.require(3.0, 'forge probe')

    path = FORGE / f'{args.split}.jsonl.gz'
    if not path.is_file():
        raise SystemExit(f'no such split: {path}')
    rows = [json.loads(line) for line in gzip.open(path, 'rt', encoding='utf-8')]
    random.Random(args.seed).shuffle(rows)

    tok = run._load_tokenizer(SRC)
    eos = run._eos_ids(SRC, tok)

    selected, skipped_long, skipped_noanswer = [], 0, 0
    for row in rows:
        answer = next((m['content'] for m in row['messages'] if m['role'] == 'assistant'), None)
        if answer is None:
            skipped_noanswer += 1
            continue
        prompt = [m for m in row['messages'] if m['role'] != 'assistant']
        ids = tok.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True,
                                      enable_thinking=False)
        if len(ids) > args.max_prompt:
            skipped_long += 1
            continue
        selected.append({'id': row['id'], 'ids': ids, 'answer': answer})
        if len(selected) >= args.rows:
            break
    if not selected:
        raise SystemExit('no rows fit the prompt budget')
    print(f'{args.split}: {len(selected)} rows selected, {skipped_long} skipped as too long '
          f'(> {args.max_prompt} tokens), {skipped_noanswer} without an assistant turn')

    model = run._load(SRC, args.device)[0]
    applied = None
    if args.checkpoint:
        from safetensors.torch import load_file
        directory = ROOT / 'outputs' / 'arm-checkpoints' / args.checkpoint
        tensors = load_file(str(directory / 'trainable.safetensors'), device='cpu')
        by_name = dict(model.named_parameters())
        with torch.no_grad():
            for name, value in tensors.items():
                if name not in by_name:
                    raise SystemExit(f'checkpoint tensor missing from model: {name}')
                by_name[name].copy_(value.to(by_name[name].device, by_name[name].dtype))
        applied = {'tag': args.checkpoint, 'tensors': len(tensors)}
        print(f'overlaid {len(tensors)} trained tensors from {args.checkpoint}')
        del tensors
    model.eval()

    exact = contains = terminated = 0
    records, started = [], time.perf_counter()
    with torch.inference_mode():
        for i, row in enumerate(selected, start=1):
            ids = torch.tensor([row['ids']], dtype=torch.long, device=args.device)
            generated, past, current, done = [], None, ids, False
            for _ in range(args.max_new):
                step = model(current, past=past, use_cache=True, last_position_only=True)
                past = step.cache
                token = int(step.logits[0, -1].argmax())
                if token in eos:
                    done = True
                    break
                generated.append(token)
                current = torch.tensor([[token]], dtype=torch.long, device=args.device)
            text = tok.decode(generated, skip_special_tokens=True).strip()
            del past
            want, got = normalise(row['answer']), normalise(text)
            is_exact = got == want
            is_contains = bool(want) and want in got
            exact += is_exact
            contains += is_contains
            terminated += done
            records.append({'id': row['id'], 'answer': row['answer'], 'text': text,
                            'exact': is_exact, 'contains': is_contains, 'terminated': done,
                            'prompt_tokens': len(row['ids'])})
            if i % 25 == 0:
                print(f'  {i}/{len(selected)} | exact {exact} | contains {contains} | '
                      f'{time.perf_counter()-started:.0f}s', flush=True)

    n = len(selected)
    body = {
        'schema': 1, 'evidence_kind': 'exploratory_forge_floor_probe_only',
        'note': 'Floor check on Rineya Reasoning Forge. Scored on GPU deliberately: this '
                'benchmark has no frozen CPU reference, so the panel device-integrity '
                'constraint does not apply. Long rows are skipped rather than truncated. '
                'Outside the hashed root: no receipt, no gate, no efficacy claim.',
        'label': args.label or (args.checkpoint or 'source'),
        'split': args.split, 'rows_scored': n, 'rows_skipped_too_long': skipped_long,
        'max_prompt_tokens': args.max_prompt, 'max_new_tokens': args.max_new,
        'seed': args.seed, 'device': args.device,
        'checkpoint_applied': applied,
        'reference_digest': implementation_digest(),
        'exact': exact, 'exact_rate': exact / n,
        'contains': contains, 'contains_rate': contains / n,
        'terminated': terminated,
        'wall_seconds': time.perf_counter() - started,
        'records': records,
        'licenses_efficacy_claim': False, 'licenses_release': False,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    tag = args.label or (args.checkpoint or 'source')
    out = OUT / f'FORGE_{args.split}_{tag}.json'
    out.write_text(json.dumps(body, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(f'\n{tag} on {args.split}: exact {exact}/{n} ({100*exact/n:.1f}%) | '
          f'contains {contains}/{n} ({100*contains/n:.1f}%) | terminated {terminated}/{n}')
    print('wrote', out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
