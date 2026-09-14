"""Mixed-length TRAIN: does seeing long sequences preserve off-distribution stopping?

Every finetuning row this programme has used is about 128 tokens long. Probed on Rineya
Reasoning Forge, whose rows run about 1,400, three independently trained models each lost 13-16
rows of termination and recovered zero (FORGE_TRANSFER.md, and A1 in
AUTONOMOUS_PLAN_20260914.md). Correcting the learning-rate schedule moved that by three rows, so
it is not an optimizer artifact. The remaining candidate is the data: a model finetuned only on
short sequences drifts in where it places a stop token.

This builds a TRAIN split mixing splits_large rows with forge TRAIN rows rendered into the same
`text` shape, so the only thing that changes against the banked runs is the length distribution
of the training data.

    python experiments/rab001_exploration/build_splits_mixed.py [--forge-token-share 0.30]
                                                                [--max-forge-tokens 2048]
                                                                [--budget 26000] [--seed 17]

Contamination guards, all failing closed. The last split this programme built passed eight
id/group_id disjointness assertions and still leaked 47 of 150 panel questions verbatim, because
different template families render identical stems. So this one checks rendered TEXT:

  * no panel stem, verbatim or digit-blinded, may appear in any training row;
  * no forge TRAIN row may share an id with any forge evaluation split;
  * no forge evaluation QUESTION tail may appear verbatim in a training row.

The last of those checks questions rather than answers on purpose. Forge answers are short
categorical labels -- room names and colours like `Workshop`, `Studio`, `silver`, `amber` -- so
an answer-verbatim check flags eleven of them against a 10,000-row general-conversation corpus
purely because they are ordinary English words. A guard that cries wolf is a guard someone
eventually switches off, and a leaked eval item means its PROMPT appears in training.

Writes to a NEW create-only directory. Existing splits are never edited: a training corpus is
bound into every completed run's `data_sha256`, and rewriting one would silently break
comparability with every banked result.

Exploratory. Outside the hashed root: no receipt, no gate, no efficacy claim.
"""
from __future__ import annotations

import argparse
import gzip
import json
import random
import re
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.architecture_benchmark_001.run import _load_tokenizer     # noqa: E402
from experiments.architecture_benchmark_001.source import ROOT             # noqa: E402

FORGE = REPO / 'rineya_reasoning_forge_v1' / 'data'
LARGE = ROOT / 'outputs' / 'splits_large'
SRC = REPO / 'Supa'
if not (SRC / 'model.safetensors').is_file():
    SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
               r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')

_WS = re.compile(r'\s+')
blind = lambda s: re.sub(r'\d', '#', s)


def panel_stems() -> list[str]:
    rows = [json.loads(l) for l in (ROOT / 'data' / 'monitor_probe.jsonl')
            .read_text(encoding='utf-8').splitlines() if l.strip()]
    out = []
    for row in rows:
        user = [m for m in row['messages'] if m['role'] == 'user'][-1]['content']
        out.append(user.split(': ')[-1].strip() if ': ' in user else user.strip())
    return [s for s in out if s]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--forge-token-share', type=float, default=0.30)
    parser.add_argument('--max-forge-tokens', type=int, default=2048)
    parser.add_argument('--budget', type=int, default=26000)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()
    if not 0 < args.forge_token_share < 1:
        raise SystemExit('forge token share must be strictly between 0 and 1')

    tok = _load_tokenizer(SRC)
    render = lambda messages: tok.apply_chat_template(
        messages, tokenize=False, enable_thinking=False, add_generation_prompt=False)
    ntok = lambda text: len(tok(text, add_special_tokens=False)['input_ids'])

    large = [json.loads(l) for l in (LARGE / 'train.jsonl').read_text(encoding='utf-8')
             .splitlines() if l.strip()]
    forge_train = [json.loads(l) for l in
                   gzip.open(FORGE / 'train.jsonl.gz', 'rt', encoding='utf-8')]
    # Guard on eval QUESTIONS, not eval answers. Forge answers are short categorical labels --
    # room names and colours like 'Workshop', 'Studio', 'silver', 'amber' -- and an
    # answer-verbatim check flags 11 of them in a 10,000-row general-conversation corpus purely
    # because those are ordinary English words. That is a false alarm, and a guard that cries
    # wolf is a guard someone eventually switches off. A leaked eval item means its PROMPT
    # appears in training, so that is what gets checked: a long distinctive slice of each eval
    # question's record block.
    eval_ids, eval_prompts = set(), set()
    for name in ('test_iid', 'test_transfer', 'test_length', 'validation'):
        for line in gzip.open(FORGE / f'{name}.jsonl.gz', 'rt', encoding='utf-8'):
            row = json.loads(line)
            eval_ids.add(row['id'])
            user = next((m['content'] for m in row['messages'] if m['role'] == 'user'), '')
            body = _WS.sub(' ', user).strip()
            if len(body) >= 300:
                eval_prompts.add(body[-260:])      # the tail carries the question itself
    print(f'splits_large TRAIN {len(large)} | forge TRAIN {len(forge_train)} | '
          f'forge eval ids {len(eval_ids)}')

    rng = random.Random(args.seed)
    rng.shuffle(forge_train)
    want_forge = int(args.budget * args.forge_token_share)
    chosen, forge_tokens, skipped_long, skipped_id = [], 0, 0, 0
    for row in forge_train:
        if forge_tokens >= want_forge:
            break
        if row['id'] in eval_ids:
            skipped_id += 1
            continue
        text = render(row['messages'])
        n = ntok(text)
        if n > args.max_forge_tokens:
            skipped_long += 1
            continue
        chosen.append({'id': row['id'], 'group_id': row['id'], 'split': 'train',
                       'reply_class': 'forge_reasoning', 'text': text,
                       'source_row_id': row['id'], 'origin': 'forge', 'tokens': n})
        forge_tokens += n
    if not chosen:
        raise SystemExit('no forge rows fit the budget')

    # Interleave deterministically so long rows are spread through the run rather than clustered.
    want_large = args.budget - forge_tokens
    picked_large, large_tokens = [], 0
    for row in large:
        if large_tokens >= want_large:
            break
        n = ntok(row['text'])
        picked_large.append({**row, 'origin': 'large', 'tokens': n})
        large_tokens += n
    merged = []
    stride = max(1, len(picked_large) // max(1, len(chosen)))
    f = iter(chosen)
    for i, row in enumerate(picked_large):
        merged.append(row)
        if i % stride == stride - 1:
            nxt = next(f, None)
            if nxt is not None:
                merged.append(nxt)
    merged.extend(f)

    stems = panel_stems()
    blob = ' \n '.join(r['text'] for r in merged)
    bblob = blind(blob)
    flat = _WS.sub(' ', blob)
    leaks = {
        'panel_stem_verbatim': sum(1 for s in stems if s in blob),
        'panel_stem_blinded': sum(1 for s in stems if blind(s) in bblob),
        'forge_eval_id_in_train': sum(1 for r in merged if r['id'] in eval_ids),
        'forge_eval_prompt_tail_verbatim': sum(1 for q in eval_prompts if q in flat),
    }
    for name, count in leaks.items():
        if count:
            raise SystemExit(f'CONTAMINATION: {name} = {count}; refusing to write the split')
    print('contamination guards clear:', leaks)

    # max_forge_tokens belongs in the name: it changes the length distribution, which is the
    # whole variable under test, and two settings must not collide on one create-only path.
    out = ROOT / 'outputs' / (f'splits_mixed_f{int(args.forge_token_share*100)}'
                              f'_b{args.budget}_L{args.max_forge_tokens}_seed{args.seed}')
    if out.exists():
        raise SystemExit(f'split directory already exists (create-only): {out}')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'train.jsonl').write_text(
        ''.join(json.dumps({k: v for k, v in r.items() if k not in ('origin', 'tokens')},
                           ensure_ascii=False) + '\n' for r in merged), encoding='utf-8')
    for name in ('dev.jsonl', 'task_probe.jsonl'):
        (out / name).write_text((LARGE / name).read_text(encoding='utf-8'), encoding='utf-8')

    total = forge_tokens + large_tokens
    manifest = {
        'schema': 1, 'evidence_kind': 'exploratory_mixed_length_train_split',
        'note': 'TRAIN mixes splits_large with forge TRAIN rows to test whether exposure to long '
                'sequences preserves off-distribution termination. DEV and task_probe are copied '
                'byte-for-byte from splits_large so the only change is TRAIN length composition. '
                'Forge rows are labelled reply_class=forge_reasoning and are NOT arithmetic, so '
                'the panel remains a transfer benchmark.',
        'seed': args.seed, 'budget_tokens': args.budget,
        'forge_token_share_requested': args.forge_token_share,
        'forge_token_share_actual': forge_tokens / total,
        'rows': {'total': len(merged), 'forge': len(chosen), 'large': len(picked_large)},
        'tokens': {'total': total, 'forge': forge_tokens, 'large': large_tokens},
        'forge_rows_skipped_too_long': skipped_long, 'forge_rows_skipped_eval_id': skipped_id,
        'max_forge_tokens': args.max_forge_tokens,
        'contamination_guards': leaks,
        'dev_and_task_probe_copied_from': str(LARGE),
        'licenses_registered_training': False, 'licenses_efficacy_claim': False,
    }
    (out / 'SPLIT_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'\nwrote {out}')
    print(f'  {len(merged)} rows, {total:,} tokens '
          f'({forge_tokens:,} forge = {100*forge_tokens/total:.1f}%, {large_tokens:,} large)')
    print(f'  forge rows {len(chosen)} (skipped {skipped_long} too long, {skipped_id} eval ids)')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
