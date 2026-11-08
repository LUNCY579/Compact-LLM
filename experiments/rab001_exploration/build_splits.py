"""Build rigorous, family-disjoint splits from the v3 corpus.

The pilot so far used 25 training rows, an 8-row held-out CE set and a 6-prompt task probe.
Those are too small to support the conclusions being drawn from them. This builds:

  TRAIN  held-in classes, one row per template family
  DEV    held-in classes, families disjoint from TRAIN, for held-out CE/KL
  TASK   the three PROBE classes, families disjoint from everything, verifiable integers

The probe classes are chosen from `baseline_scan.py`, which measured what the FROZEN SOURCE
can actually do. `arithmetic_chain` was the obvious candidate and is **unusable**: the source
scores 0/20 on it, so every arm would read 0/N before and after and the probe would measure
nothing. Measured source baselines:

    percentage_reasoning  14/20  70%      counting_sets      2/20  10%
    ratio_proportion      10/20  50%      sequence_pattern   2/20  10%
    time_duration          7/20  35%      unit_conversion    1/20   5%
    arithmetic_chain       0/20   0%      comparison_ranking 0/20   not numeric

PROBE = the three classes above 30%. They are excluded from TRAIN and DEV, so the task set
measures **retention** of a capability the source demonstrably has, and both degradation and
improvement are visible against a ~52% baseline rather than against a floor of zero.

Choosing the probe on the frozen teacher's ability is legitimate: it is measured before any
arm is trained and is not tuned on any student outcome.

Grouping is by `metadata.template_family`, and no family may appear in more than one split.
The corpus ships 16,504 families with no family exceeding 8 rows, so one row per family gives
independent draws.

Rendering is verified against the already-prepared `train-smoke.jsonl` before anything is
written: if my chat-template call does not reproduce the stored text byte-for-byte, the new
splits would not be comparable to the existing runs and the script refuses to continue.

Exploratory. Outside the hashed experiment root; produces no receipt and opens no gate.
"""
import json, random, sys
from pathlib import Path

# This host's console is cp1252. A model emitting non-ASCII token salad would otherwise
# kill a run inside a print AFTER the expensive compute was done -- which is exactly how
# three C28 runs were lost. Output formatting must never destroy a measurement.
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass

sys.path.insert(0, r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification')
from experiments.architecture_benchmark_001.run import _load_tokenizer

CORPUS = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/by_class')
E = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\experiments\architecture_benchmark_001')
OUT = E / 'outputs/splits'
SEED = 17
N_TRAIN, N_DEV, N_TASK = 800, 150, 150
# Classes the frozen source can actually do (baseline_scan.py). Held out of TRAIN/DEV.
PROBE = ('percentage_reasoning', 'ratio_proportion', 'time_duration')
# Measured at 0/20 for the source: unusable as a retention probe, and excluded from training
# so it cannot quietly become an in-domain class either.
UNUSABLE = ('arithmetic_chain',)

tok = _load_tokenizer(Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze'))


def render(messages):
    """Exactly the rendering used for the existing prepared splits."""
    return tok.apply_chat_template(messages, tokenize=False, enable_thinking=False,
                                   add_generation_prompt=False)


# ---- verify the rendering reproduces the existing prepared data -------------
smoke = [json.loads(l) for l in (E / 'data/train-smoke.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
by_id = {}
for cls_file in CORPUS.glob('*.jsonl'):
    for line in cls_file.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_id[r['id']] = r

checked = matched = 0
for row in smoke:
    src = by_id.get(row.get('source_row_id') or row['id'])
    if src is None:
        continue
    checked += 1
    matched += (render(src['messages']) == row['text'])
if not checked:
    raise SystemExit('could not locate any train-smoke source row in the corpus; aborting')
print(f'rendering check: {matched}/{checked} prepared rows reproduced byte-for-byte')
if matched != checked:
    raise SystemExit('rendering does not match the existing prepared splits; refusing to write '
                     'splits that would not be comparable to prior runs')

# ---- collect one row per family -------------------------------------------
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
        fam = r['metadata']['template_family']
        bucket.setdefault(fam, []).append(r)

print(f'families: probe classes {len(probe)}, held-in classes {len(other)}')
arith_fams = sorted(probe); rng.shuffle(arith_fams)
other_fams = sorted(other); rng.shuffle(other_fams)
assert not (set(arith_fams) & set(other_fams)), 'a family spans probe and held-in classes'
arith = probe

train_fams = other_fams[:N_TRAIN]
dev_fams = other_fams[N_TRAIN:N_TRAIN + N_DEV]
task_fams = arith_fams[:N_TASK]
assert not (set(train_fams) & set(dev_fams))


def prepared(rows, split):
    out = []
    for r in rows:
        text = render(r['messages'])
        out.append({'id': r['id'], 'group_id': r['metadata']['template_family'], 'split': split,
                    'reply_class': r['metadata']['reply_class'], 'text': text,
                    'source_row_id': r['id']})
    return out


def task_rows(rows):
    out = []
    for r in rows:
        msgs = r['messages']
        if msgs[-1]['role'] != 'assistant':
            continue
        answer = msgs[-1]['content'].strip()
        out.append({'id': r['id'], 'group_id': r['metadata']['template_family'], 'split': 'task',
                    'reply_class': r['metadata']['reply_class'],
                    'messages': msgs[:-1], 'answer': answer,
                    'difficulty': r['metadata'].get('difficulty'),
                    'multi_turn': r['metadata'].get('multi_turn')})
    return out


train = prepared([rng.choice(other[f]) for f in train_fams], 'train')
dev = prepared([rng.choice(other[f]) for f in dev_fams], 'dev')
task = task_rows([rng.choice(arith[f]) for f in task_fams])

OUT.mkdir(parents=True, exist_ok=True)
for name, rows in (('train', train), ('dev', dev), ('task_probe', task)):
    path = OUT / f'{name}.jsonl'
    path.write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows) + '\n', encoding='utf-8')
    print(f'{name:16} {len(rows):>5} rows -> {path.name}')

# ---- disjointness and leakage proof ---------------------------------------
fams = {'train': {r['group_id'] for r in train}, 'dev': {r['group_id'] for r in dev},
        'task': {r['group_id'] for r in task}}
ids = {'train': {r['id'] for r in train}, 'dev': {r['id'] for r in dev}, 'task': {r['id'] for r in task}}
overlaps = {f'{a}|{b}': len(fams[a] & fams[b]) for a in fams for b in fams if a < b}
id_overlaps = {f'{a}|{b}': len(ids[a] & ids[b]) for a in ids for b in ids if a < b}
classes = sorted({r['reply_class'] for r in train})
assert all(v == 0 for v in overlaps.values()), overlaps
assert all(v == 0 for v in id_overlaps.values()), id_overlaps
for bad in (*PROBE, *UNUSABLE):
    assert bad not in classes, f'{bad} leaked into TRAIN'
    assert bad not in {r['reply_class'] for r in dev}, f'{bad} leaked into DEV'
assert {r['reply_class'] for r in task} <= set(PROBE), 'TASK contains a non-probe class'

manifest = {
    'schema': 1, 'evidence_kind': 'exploratory_split_manifest_only',
    'note': 'Family-disjoint splits from the v3 corpus. arithmetic_chain is held out of TRAIN '
            'and DEV so the task set measures RETENTION of a capability the frozen source '
            'already has. Outside the experiment package; no receipt, no gate.',
    'seed': SEED, 'grouping_key': 'metadata.template_family',
    'rendering': 'official chat template, enable_thinking=False, add_generation_prompt=False; '
                 f'verified to reproduce {matched}/{checked} existing prepared rows byte-for-byte',
    'rows': {'train': len(train), 'dev': len(dev), 'task_probe': len(task)},
    'families': {k: len(v) for k, v in fams.items()},
    'family_overlaps': overlaps, 'id_overlaps': id_overlaps,
    'train_classes': classes, 'probe_classes': list(PROBE), 'unusable_excluded': list(UNUSABLE),
    'probe_selection_basis': 'frozen-source baseline numeric-match rate from baseline_scan.py; '
                             'classes above 30% retained. arithmetic_chain excluded at 0/20.',
    'task_class_counts': {c: sum(1 for r in task if r['reply_class'] == c) for c in PROBE},
    'licenses_registered_training': False, 'licenses_efficacy_claim': False,
}
(OUT / 'SPLIT_MANIFEST.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
print('\nfamily overlaps:', overlaps, '| id overlaps:', id_overlaps)
print('train classes:', len(classes), '| probe classes held out:', PROBE)
print('task class counts:', {c: sum(1 for r in task if r['reply_class']==c) for c in PROBE})
print('wrote', OUT / 'SPLIT_MANIFEST.json')
