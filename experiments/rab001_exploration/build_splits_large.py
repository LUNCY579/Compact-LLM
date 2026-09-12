"""Grow TRAIN for a large-budget run WITHOUT moving DEV or TASK.

Every result in this programme is scored on `outputs/splits/dev.jsonl` and
`outputs/splits/task_probe.jsonl`. A large-budget run is only comparable to the 40-step runs
if those two files are bit-identical, so this script's real job is not "make a bigger TRAIN" —
it is "make a bigger TRAIN and *prove* the evaluation sets did not move".

`build_splits.py` cannot be used for this by changing `N_TRAIN`, for two independent reasons,
both of which silently relocate DEV:

  1. `dev_fams = other_fams[N_TRAIN : N_TRAIN + N_DEV]` is indexed FROM the train count, so a
     larger TRAIN slides DEV onto an entirely different set of families.
  2. One `random.Random(SEED)` is consumed in order — 800 `rng.choice` calls for TRAIN, then
     150 for DEV, then 150 for TASK. Drawing more TRAIN rows advances the stream, so even at a
     fixed family list DEV would pick a *different row* from each of its families.

Either alone would have quietly invalidated the comparison while every disjointness assertion
in `build_splits.py` still passed, because the splits would remain perfectly family-disjoint —
just not the same splits.

So this replays the original draw exactly, then continues it:

    Random(17) -> shuffle(arith) -> shuffle(other)
      -> 800 TRAIN choices   (the original TRAIN, reproduced)
      -> 150 DEV choices     (the original DEV,   reproduced)
      -> 150 TASK choices    (the original TASK,  reproduced)
      -> EXTRA TRAIN choices (new, drawn from families no split has used)

and then refuses to write unless the regenerated DEV and TASK match the existing files
byte-for-byte and the new TRAIN begins with the existing TRAIN byte-for-byte.

Output goes to `outputs/splits_large/`. `outputs/splits/` is never touched, so a run already
reading it cannot be disturbed.

Exploratory. Outside the hashed experiment root; no receipt, no gate. Open splits only.

usage: build_splits_large.py [n_train_total]
"""
import json, random, sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load_tokenizer

CORPUS = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
              r'\Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0\data\sft\by_class')
E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification'
         r'\experiments\architecture_benchmark_001')
OLD = E / 'outputs/splits'
OUT = E / 'outputs/splits_large'

# Must match build_splits.py exactly, or the replay is not a replay.
SEED = 17
N_TRAIN_ORIG, N_DEV, N_TASK = 800, 150, 150
PROBE = ('percentage_reasoning', 'ratio_proportion', 'time_duration')
UNUSABLE = ('arithmetic_chain',)

N_TRAIN = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
if N_TRAIN < N_TRAIN_ORIG:
    raise SystemExit(f'TRAIN may only grow; {N_TRAIN} < {N_TRAIN_ORIG}')

tok = _load_tokenizer(Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
                           r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze'))


def render(messages):
    return tok.apply_chat_template(messages, tokenize=False, enable_thinking=False,
                                   add_generation_prompt=False)


# ---- replay the original family ordering -----------------------------------
rng = random.Random(SEED)
probe, other = {}, {}
for cls_file in sorted(CORPUS.glob('*.jsonl')):
    cls = cls_file.stem
    if cls in UNUSABLE:
        continue
    bucket = probe if cls in PROBE else other
    for line in cls_file.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        bucket.setdefault(r['metadata']['template_family'], []).append(r)

arith_fams = sorted(probe); rng.shuffle(arith_fams)
other_fams = sorted(other); rng.shuffle(other_fams)
assert not (set(arith_fams) & set(other_fams)), 'a family spans probe and held-in classes'
print(f'families: probe {len(arith_fams)}, held-in {len(other_fams)}')

# DEV and TASK keep their ORIGINAL slices. Extra TRAIN comes from after DEV, i.e. from
# families that no split has ever used -- never from DEV's or TASK's.
train_fams = other_fams[:N_TRAIN_ORIG]
dev_fams = other_fams[N_TRAIN_ORIG:N_TRAIN_ORIG + N_DEV]
task_fams = arith_fams[:N_TASK]
spare_fams = other_fams[N_TRAIN_ORIG + N_DEV:]
extra_n = N_TRAIN - N_TRAIN_ORIG
if extra_n > len(spare_fams):
    raise SystemExit(f'need {extra_n} more families, only {len(spare_fams)} unused remain')
extra_fams = spare_fams[:extra_n]
assert not (set(extra_fams) & set(dev_fams)), 'extra TRAIN would take a DEV family'
assert not (set(extra_fams) & set(task_fams)), 'extra TRAIN would take a TASK family'
assert not (set(extra_fams) & set(train_fams)), 'extra TRAIN would duplicate a TRAIN family'


def prepared(rows, split):
    return [{'id': r['id'], 'group_id': r['metadata']['template_family'], 'split': split,
             'reply_class': r['metadata']['reply_class'], 'text': render(r['messages']),
             'source_row_id': r['id']} for r in rows]


def task_rows(rows):
    out = []
    for r in rows:
        msgs = r['messages']
        if msgs[-1]['role'] != 'assistant':
            continue
        out.append({'id': r['id'], 'group_id': r['metadata']['template_family'], 'split': 'task',
                    'reply_class': r['metadata']['reply_class'],
                    'messages': msgs[:-1], 'answer': msgs[-1]['content'].strip(),
                    'difficulty': r['metadata'].get('difficulty'),
                    'multi_turn': r['metadata'].get('multi_turn')})
    return out


# Draw in the ORIGINAL order so the stream lands identically, then continue it.
train = prepared([rng.choice(other[f]) for f in train_fams], 'train')
dev = prepared([rng.choice(other[f]) for f in dev_fams], 'dev')
task = task_rows([rng.choice(probe[f]) for f in task_fams])   # `arith` is `probe` upstream
extra = prepared([rng.choice(other[f]) for f in extra_fams], 'train')

# ---- the control: prove the evaluation sets did not move --------------------
line = lambda r: json.dumps(r, ensure_ascii=False)
readl = lambda p: [l for l in Path(p).read_text(encoding='utf-8').splitlines() if l.strip()]
checks = {}
for name, rows in (('dev', dev), ('task_probe', task)):
    old = readl(OLD / f'{name}.jsonl')
    new = [line(r) for r in rows]
    checks[name] = {'old_rows': len(old), 'new_rows': len(new), 'identical': old == new}
old_train = readl(OLD / 'train.jsonl')
new_train = [line(r) for r in train]
checks['train_prefix'] = {'old_rows': len(old_train), 'new_rows': len(new_train),
                          'identical': old_train == new_train}
for name, c in checks.items():
    print(f"  {name:14} {c['old_rows']:>6} -> {c['new_rows']:>6} identical={c['identical']}")
if not all(c['identical'] for c in checks.values()):
    raise SystemExit('REFUSING TO WRITE: the replay did not reproduce the existing splits, so a '
                     'larger TRAIN would not be comparable to the runs already recorded.')

full_train = train + extra
print(f'\nTRAIN {len(old_train)} -> {len(full_train)} (prefix preserved, {len(extra)} appended)')

# ---- disjointness, on the FULL set -----------------------------------------
fams = {'train': {r['group_id'] for r in full_train}, 'dev': {r['group_id'] for r in dev},
        'task': {r['group_id'] for r in task}}
ids = {'train': {r['id'] for r in full_train}, 'dev': {r['id'] for r in dev},
       'task': {r['id'] for r in task}}
overlaps = {f'{a}|{b}': len(fams[a] & fams[b]) for a in fams for b in fams if a < b}
id_overlaps = {f'{a}|{b}': len(ids[a] & ids[b]) for a in ids for b in ids if a < b}
assert all(v == 0 for v in overlaps.values()), overlaps
assert all(v == 0 for v in id_overlaps.values()), id_overlaps
assert len(fams['train']) == len(full_train), 'a TRAIN family appears twice'
train_classes = sorted({r['reply_class'] for r in full_train})
for bad in (*PROBE, *UNUSABLE):
    assert bad not in train_classes, f'{bad} leaked into TRAIN'
    assert bad not in {r['reply_class'] for r in dev}, f'{bad} leaked into DEV'
assert {r['reply_class'] for r in task} <= set(PROBE), 'TASK contains a non-probe class'

# ---- budget arithmetic, so the run cannot be planned on a guess ------------
tokens = [len(tok(r['text'], add_special_tokens=False)['input_ids']) for r in full_train]
target = sum(t - 1 for t in tokens if t >= 2)   # next-token targets, matching the trainer
print(f'one epoch = {target} supervised target tokens over {len(full_train)} rows '
      f'(mean {target / len(full_train):.1f}/row)')

OUT.mkdir(parents=True, exist_ok=True)
for name, rows in (('train', full_train), ('dev', dev), ('task_probe', task)):
    (OUT / f'{name}.jsonl').write_text('\n'.join(line(r) for r in rows) + '\n', encoding='utf-8')
    print(f'{name:16} {len(rows):>6} rows -> {OUT.name}/{name}.jsonl')

(OUT / 'SPLIT_MANIFEST.json').write_text(json.dumps({
    'schema': 1, 'evidence_kind': 'exploratory_split_manifest_only',
    'note': 'TRAIN extended for a large-budget run. DEV and TASK are REPRODUCED BIT-FOR-BIT '
            'from outputs/splits/ and the new TRAIN begins with the old TRAIN bit-for-bit, so '
            'results here are comparable to the 40-step runs. Extra TRAIN families are drawn '
            'only from families no split had used. Open splits only; no receipt, no gate.',
    'seed': SEED, 'grouping_key': 'metadata.template_family',
    'derived_from': str(OLD), 'comparability_controls': checks,
    'rows': {'train': len(full_train), 'dev': len(dev), 'task_probe': len(task)},
    'train_rows_original': len(old_train), 'train_rows_appended': len(extra),
    'family_overlaps': overlaps, 'id_overlaps': id_overlaps,
    'held_in_families_available': len(other_fams), 'held_in_families_unused': len(spare_fams) - extra_n,
    'one_epoch_supervised_target_tokens': target,
    'train_classes': train_classes,
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}, indent=2) + '\n', encoding='utf-8')
print('wrote', OUT / 'SPLIT_MANIFEST.json')
