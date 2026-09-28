"""The 6,492 on-panel training rows nobody has used, and an in-domain DEV that actually tracks.

`build_splits_large.py` partitions the corpus into a `probe` bucket -- the three classes the
150-row capability panel is drawn from -- and an `other` bucket, then draws TRAIN and DEV
*exclusively from `other`*. That is deliberate: it makes the panel a TRANSFER benchmark, and
the measured 47/150 -> 73/150 was achieved without the model ever seeing an on-panel row.

It also leaves 975 of the corpus's 1,125 probe-class families completely unused. This builds a
split from those, family-disjoint from the panel by construction.

Two things follow, and they should not be confused with each other:

1. TRAINING on this converts the panel from a transfer benchmark into an IN-DOMAIN
   generalization benchmark. The families stay disjoint so it is still a real held-out test,
   but it is a narrower claim and must be labelled as such. It must never be quoted as
   evidence of general capability gain. For a shipped product, in-domain data on the task you
   care about is the right data; for a claim about intelligence it is the weaker one.

2. The DEV slice here fixes a measurement defect. `splits_large/dev.jsonl` is drawn from the
   same 36 off-panel classes as TRAIN, so `dev_ce` measures fit to the distribution being
   trained and is disjoint from what the panel scores. That is most of the celebrated
   "likelihood/capability decoupling": CE was never a capability proxy for this panel. An
   on-panel DEV gives a CE that can actually serve as an early-stopping signal.

Replays `build_splits_large.py`'s family ordering exactly (same SEED, same sorted-then-shuffle),
so the 150 families it excludes are provably the panel's own. That replay is verified against
the real `task_probe.jsonl` before anything is written: 150 of 150 group_ids must match.

Inherits the calibration/pilot guard: families appearing in the registered CALIBRATION or PILOT
splits are excluded, so a later stage that scores on them is not reading its own training data.

    python experiments/rab001_exploration/build_splits_onpanel.py [dev_families]

Writes outputs/splits_onpanel/{train,dev}.jsonl and SPLIT_MANIFEST.json. Open splits only; no
receipt, no gate, and nothing here licenses registered training.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from experiments.architecture_benchmark_001.run import _load_tokenizer          # noqa: E402
from experiments.architecture_benchmark_001.source import ROOT                  # noqa: E402

CORPUS = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
              r'\Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0\data\sft\by_class')
SRC = REPO / 'Supa'
if not (SRC / 'model.safetensors').is_file():
    SRC = Path(r'C:\Users\clclc\BusinessProject\Chat-Bubble'
               r'\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze')

#: Must match build_splits_large.py exactly, or the replay does not reproduce the panel.
SEED, N_TASK = 17, 150
PROBE = ('percentage_reasoning', 'ratio_proportion', 'time_duration')
N_DEV_FAMS = int(sys.argv[1]) if len(sys.argv) > 1 else 120
OUT = ROOT / 'outputs' / 'splits_onpanel'


def main() -> int:
    tok = _load_tokenizer(SRC)
    render = lambda messages: tok.apply_chat_template(
        messages, tokenize=False, enable_thinking=False, add_generation_prompt=False)

    probe = {}
    for cls in PROBE:
        for line in (CORPUS / f'{cls}.jsonl').read_text(encoding='utf-8').splitlines():
            if line.strip():
                row = json.loads(line)
                probe.setdefault(row['metadata']['template_family'], []).append(row)
    print(f'probe-class families {len(probe)}, rows {sum(len(v) for v in probe.values())}')

    # Replay build_splits_large.py's ordering. `rng` is consumed in the same sequence there,
    # but only the family ORDER matters for identifying the panel's slice.
    rng = random.Random(SEED)
    fams = sorted(probe)
    rng.shuffle(fams)
    task_fams, spare = set(fams[:N_TASK]), fams[N_TASK:]

    panel = [json.loads(l) for l in
             (ROOT / 'outputs/splits_large/task_probe.jsonl').read_text(encoding='utf-8')
             .splitlines() if l.strip()]
    panel_fams = {r['group_id'] for r in panel}
    if panel_fams != task_fams:
        raise SystemExit(f'replay does not reproduce the panel: {len(panel_fams & task_fams)} '
                         f'of {len(panel_fams)} families match; refusing to build')
    print(f'replay verified: {len(panel_fams)}/{len(panel_fams)} panel families reproduced')

    E = ROOT
    reserved = set()
    for name in ('data/calibration.jsonl', 'data/pilot.jsonl'):
        path = E / name
        if path.is_file():
            reserved |= {json.loads(l)['group_id'] for l in
                         path.read_text(encoding='utf-8').splitlines() if l.strip()}
    clean = [f for f in spare if f not in reserved]
    print(f'reserved (calibration+pilot) {len(reserved)} | excluded '
          f'{len(spare)-len(clean)} of {len(spare)} spare probe families')
    if N_DEV_FAMS >= len(clean):
        raise SystemExit(f'{N_DEV_FAMS} dev families requested, only {len(clean)} available')

    dev_fams, train_fams = clean[:N_DEV_FAMS], clean[N_DEV_FAMS:]

    def prepared(names, split):
        return [{'id': r['id'], 'group_id': r['metadata']['template_family'], 'split': split,
                 'reply_class': r['metadata']['reply_class'], 'text': render(r['messages']),
                 'source_row_id': r['id']}
                for r in (rng.choice(probe[f]) for f in names)]

    def scorable(names):
        """DEV must be SCORABLE, not just trainable.

        `prepared()` renders the full chat text including the assistant's reply, which is what
        a training row needs and exactly what an evaluation row must not contain. Greedy model
        soup has to select ingredients on a held-out set -- selecting on the 150-row panel
        would fit the soup to the test set -- so DEV is emitted in the same shape as the panel:
        the messages with the final assistant turn removed, plus that turn as the reference
        answer. Mirrors `build_splits_large.py::task_rows`.
        """
        out = []
        for r in (rng.choice(probe[f]) for f in names):
            messages = r['messages']
            if messages[-1]['role'] != 'assistant':
                continue
            out.append({'id': r['id'], 'group_id': r['metadata']['template_family'],
                        'split': 'task', 'reply_class': r['metadata']['reply_class'],
                        'messages': messages[:-1], 'answer': messages[-1]['content'].strip(),
                        'difficulty': r['metadata'].get('difficulty'),
                        'multi_turn': r['metadata'].get('multi_turn')})
        return out

    train, dev = prepared(train_fams, 'train'), scorable(dev_fams)
    if len(dev) != len(dev_fams):
        print(f'  note: {len(dev_fams)-len(dev)} dev families dropped (no assistant turn)')

    large = ROOT / 'outputs' / 'splits_large'
    other = {}
    for name in ('train', 'dev'):
        other[name] = {json.loads(l)['group_id'] for l in
                       (large / f'{name}.jsonl').read_text(encoding='utf-8').splitlines()
                       if l.strip()}
    checks = {
        'train|panel': len({r['group_id'] for r in train} & panel_fams),
        'dev|panel': len({r['group_id'] for r in dev} & panel_fams),
        'train|dev': len({r['group_id'] for r in train} & {r['group_id'] for r in dev}),
        'train|calibration_pilot': len({r['group_id'] for r in train} & reserved),
        'dev|calibration_pilot': len({r['group_id'] for r in dev} & reserved),
        'train|splits_large_train': len({r['group_id'] for r in train} & other['train']),
        'train|splits_large_dev': len({r['group_id'] for r in train} & other['dev']),
        'id|panel': len({r['id'] for r in train + dev} & {r['id'] for r in panel}),
    }
    for name, count in checks.items():
        if count:
            raise SystemExit(f'disjointness violated: {name} shares {count}')
    print('disjointness verified:', checks)

    tokens = lambda rows: sum(len(tok(r['text'], add_special_tokens=False)['input_ids']) - 1
                              for r in rows)
    train_tokens = tokens(train)
    # DEV rows are scorable (messages + answer), so there is no `text` to count. Report the
    # reference-answer length instead, which is what a generation probe actually has to produce.
    dev_tokens = sum(len(tok(r['answer'], add_special_tokens=False)['input_ids']) for r in dev)

    OUT.mkdir(parents=True, exist_ok=True)
    for name, rows in (('train', train), ('dev', dev)):
        (OUT / f'{name}.jsonl').write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    from collections import Counter
    manifest = {
        'schema': 1, 'evidence_kind': 'exploratory_open_split_only',
        'note': 'IN-DOMAIN split: drawn from the 975 probe-class families the panel did not '
                'take. Training on this converts the 150-row panel from a TRANSFER benchmark '
                'into an IN-DOMAIN generalization benchmark. Family-disjoint from the panel, '
                'from calibration/pilot, and from splits_large. A result measured after '
                'training on this must never be quoted as a general capability gain.',
        'seed': SEED, 'grouping_key': 'metadata.template_family',
        'corpus': str(CORPUS), 'derived_from': str(large),
        'probe_classes': list(PROBE),
        'families': {'corpus_probe_total': len(probe), 'panel': len(task_fams),
                     'spare': len(spare), 'clean_spare': len(clean),
                     'train': len(train_fams), 'dev': len(dev_fams)},
        'rows': {'train': len(train), 'dev': len(dev)},
        'supervised_target_tokens': {'train': train_tokens},
        'dev_reference_answer_tokens': dev_tokens,
        'dev_format': 'scorable: messages with the final assistant turn removed, plus that turn '
                      'as `answer`. Same shape as the 150-row panel, so the same probe and '
                      'scorer run on it unchanged.',
        'train_classes': dict(Counter(r['reply_class'] for r in train)),
        'dev_classes': dict(Counter(r['reply_class'] for r in dev)),
        'panel_replay_verified': True, 'disjointness': checks,
        'licenses_registered_training': False, 'licenses_efficacy_claim': False,
    }
    (OUT / 'SPLIT_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    print(f'TRAIN {len(train)} rows / {train_tokens:,} supervised tokens')
    print(f'DEV   {len(dev)} rows / {dev_tokens:,} supervised tokens')
    print('classes:', manifest['train_classes'])
    print('wrote', OUT)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
