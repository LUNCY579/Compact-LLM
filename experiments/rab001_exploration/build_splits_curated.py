"""Select a training split from the existing pool by a named rule, at a fixed token budget.

The deliverable trained on **41 of `splits_large`'s 10,000 rows** -- 0.41% of the pool --
and the four seeds drew *completely disjoint* row sets scoring 74, 62, 73 and 68. At this
dose the dominant lever is not how much data there is but which forty rows you get, and
adding more has already been measured to hurt: 15,600 tokens scores 64, a 25,000-token
mixture 48, and 11.5x reasoning data moved reasoning accuracy by zero.

So this selects rather than accumulates.

    python experiments/rab001_exploration/build_splits_curated.py --rule balanced
    python experiments/rab001_exploration/build_splits_curated.py --rule compositional
    python experiments/rab001_exploration/build_splits_curated.py --rule random --seed 17

**Selection is by corpus-intrinsic metadata only.** `difficulty`, `capability_group`,
`length_mode` and `reply_class` come from v3.0's own curation record, rejoined onto the pool
through `source_row_id` (which covers 10,000 of 10,000 rows). Nothing here may consult the
panel, and no rule may be chosen by what scores well -- picking training rows by test score
is how a 47-row leak gets built, and this programme has already built one.

**Why this inherits its contamination guarantees instead of asserting new ones.** Every row
written comes verbatim from `splits_large/train.jsonl`, which was built with family- and
id-disjointness against DEV and the task probe. A subset of a clean pool is clean. DEV and
`task_probe.jsonl` are copied byte-for-byte, so the only thing that differs from the banked
runs is which TRAIN rows exist. The panel's three classes are absent from the pool entirely,
which is checked here rather than remembered.

**Sized to the budget on purpose.** `arm_dose` shuffles TRAIN by seed and consumes from the
front, so a split holding *exactly* the budget's worth of rows is fully consumed at every
seed and the seed changes only the order. That turns replication into a clean test of
selection: the four deliverable seeds confounded "different rows" with "different order",
and this separates them.

Create-only. A training corpus is bound into every completed run's `data_sha256` and
rewriting one silently breaks comparability with every banked result.
"""
from __future__ import annotations

import argparse
import collections
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

POOL = ROOT / 'outputs' / 'splits_large'
V3 = REPO / 'Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0' / 'data' / 'sft'
V3_FILES = ('rineya_v3_reasoning_train_50000.jsonl',
            'rineya_v3_natural_communication_train_30000.jsonl',
            'rineya_v3_integrated_reasoning_dialogue_train_30000.jsonl',
            'rineya_v1_v2_preservation_slice_14000.jsonl')
SRC = REPO / 'Supa'

_WS = re.compile(r'\s+')
ABSTAIN_SHARE = 0.33      # set from --abstain-share; see rule_repair
blind = lambda s: re.sub(r'\d', '#', s)


def load_pool() -> list[dict]:
    return [json.loads(l) for l in (POOL / 'train.jsonl').read_text(encoding='utf-8')
            .splitlines() if l.strip()]


def load_metadata(rows: list[dict]) -> dict[str, dict]:
    """Rejoin v3.0's curation record through source_row_id. Must cover every row."""
    want = {r['source_row_id'] for r in rows}
    found = {}
    for name in V3_FILES:
        for line in (V3 / name).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] in want and row['id'] not in found:
                found[row['id']] = row['metadata']
    missing = want - set(found)
    if missing:
        raise SystemExit(f'{len(missing)} pool rows have no v3.0 metadata, e.g. '
                         f'{sorted(missing)[:3]}; selection by metadata would silently '
                         f'drop them')
    return found


def panel_stems() -> list[str]:
    rows = [json.loads(l) for l in (ROOT / 'data' / 'monitor_probe.jsonl')
            .read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    out = []
    for row in rows:
        user = [m for m in row['messages'] if m['role'] == 'user'][-1]['content']
        out.append(user.split(': ')[-1].strip() if ': ' in user else user.strip())
    return [s for s in out if s]


# ---------------------------------------------------------------------------
# Selection rules. Each takes (rows, meta, rng) and returns rows in priority order;
# the caller then takes from the front until the token budget is met. A rule expresses
# a hypothesis about what makes a training row useful -- never about what scores well.
# ---------------------------------------------------------------------------

def rule_random(rows, meta, rng):
    """Control. What a seeded draw from the whole pool already does."""
    out = list(rows)
    rng.shuffle(out)
    return out


def rule_balanced(rows, meta, rng):
    """Round-robin over reply_class, so ~40 rows cover as many of the 36 classes as
    possible. A random draw of 41 leaves a dozen classes unseen purely by chance."""
    buckets = collections.defaultdict(list)
    for row in rows:
        buckets[row['reply_class']].append(row)
    for group in buckets.values():
        rng.shuffle(group)
    order, names = [], sorted(buckets)
    rng.shuffle(names)
    while any(buckets[n] for n in names):
        for name in names:
            if buckets[name]:
                order.append(buckets[name].pop())
    return order


def _by_difficulty(rows, meta, rng, wanted):
    first = [r for r in rows if meta[r['source_row_id']].get('difficulty') == wanted]
    rest = [r for r in rows if meta[r['source_row_id']].get('difficulty') != wanted]
    rng.shuffle(first)
    rng.shuffle(rest)
    return first + rest


def rule_compositional(rows, meta, rng):
    """Hardest tier first. Hypothesis: the most informative rows are the ones that
    compose several steps, because the panel's failures are compositional."""
    return _by_difficulty(rows, meta, rng, 'compositional')


def rule_basic(rows, meta, rng):
    """The deliberate opposite of `compositional`, so the difficulty axis has a control
    arm. Without it, a compositional win could be any difference from random at all."""
    return _by_difficulty(rows, meta, rng, 'basic')


def rule_reasoning(rows, meta, rng):
    """Only the `reasoning` capability group -- 4,009 of the 10,000. Hypothesis: the
    conversational half of the pool contributes nothing to an arithmetic panel."""
    first = [r for r in rows if meta[r['source_row_id']].get('capability_group') == 'reasoning']
    rest = [r for r in rows if meta[r['source_row_id']].get('capability_group') != 'reasoning']
    rng.shuffle(first)
    rng.shuffle(rest)
    return first + rest


def rule_diverse(rows, meta, rng):
    """Greedy coverage: never repeat a template_family, and spread across reply_class and
    domain. Hypothesis: at 40 rows, redundancy is the expensive mistake."""
    order = list(rows)
    rng.shuffle(order)
    seen_family, seen_class, seen_domain, picked, spare = set(), set(), set(), [], []
    for row in order:
        info = meta[row['source_row_id']]
        family = info.get('template_family')
        if family in seen_family:
            spare.append(row)
            continue
        key = (info.get('reply_class'), info.get('domain'))
        if key in {(c, d) for c in seen_class for d in seen_domain} and len(picked) > 12:
            spare.append(row)
            continue
        seen_family.add(family)
        seen_class.add(info.get('reply_class'))
        seen_domain.add(info.get('domain'))
        picked.append(row)
    return picked + spare


_ABSTAIN = re.compile(
    r"(do(es)?n[’']?t know|can[’']?t (tell|say|determine)|"
    r"not (stated|given|supported|determined|enough|specified|clear)|unknown|"
    r"no(t)? enough information|doesn[’']?t say|unclear|insufficient)", re.I)


def final_reply(text: str) -> str:
    """The last assistant turn with any silent-reasoning block removed."""
    tail = text.split('<|im_start|>assistant')[-1]
    return re.sub(r'<think>.*?</think>', '', tail, flags=re.S).replace('<|im_end|>', '').strip()


def is_abstention(row: dict) -> bool:
    return bool(_ABSTAIN.search(final_reply(row['text'])))


def rule_repair(rows, meta, rng):
    """Repair what training measurably broke, using only rows already in the pool.

    Opallo lost 11 rows of abstention on the long-context suite -- Supa emits UNKNOWN on 13
    of 64 unanswerable rows, Opallo on 2 -- and that loss is real under both a strict and a
    containment-tolerant scorer, so it is not a formatting artifact.

    248 of the pool's 10,000 rows already end in an explicit knowledge-limit reply, spread
    over five classes (`insufficient_information`, `evidence_grounded_dialogue`,
    `evidence_inference`, `boundary_setting`, `tactful_disagreement`). Using those costs no
    new contamination surface, because they are the same rows the pool was already cleared
    for.

    `ABSTAIN_SHARE` of the selection is abstention rows, **interleaved** rather than
    front-loaded: taking abstention rows first would fill the entire 5,233-token budget with
    them (248 rows at ~130 tokens is 32k) and leave the arithmetic panel nothing to learn
    from. Interleaving at a count ratio means the budget cut-off lands on a mixture. The
    remainder comes from the `diverse` rule, and the share appears in the split's directory
    name so two settings cannot collide on one create-only path.
    """
    abstain = [r for r in rows if is_abstention(r)]
    other = rule_diverse([r for r in rows if not is_abstention(r)], meta, rng)
    rng.shuffle(abstain)
    if not abstain or ABSTAIN_SHARE <= 0:
        return other + abstain
    # One abstention row every `step` positions gives it ~ABSTAIN_SHARE of any prefix.
    step = max(1, round(1 / ABSTAIN_SHARE))
    merged, a, o = [], iter(abstain), iter(other)
    while True:
        nxt = next(a, None) if len(merged) % step == 0 else next(o, None)
        if nxt is None:
            nxt = next(o, None) or next(a, None)
            if nxt is None:
                break
        merged.append(nxt)
    seen = {r['id'] for r in merged}
    return merged + [r for r in rows if r['id'] not in seen]


RULES = {'random': rule_random, 'balanced': rule_balanced,
         'compositional': rule_compositional, 'basic': rule_basic,
         'reasoning': rule_reasoning, 'diverse': rule_diverse, 'repair': rule_repair}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rule', required=True, choices=sorted(RULES))
    parser.add_argument('--budget', type=int, default=5233,
                        help='supervised target tokens, len-1 per row')
    parser.add_argument('--seed', type=int, default=17,
                        help='selection seed; NOT the training seed')
    parser.add_argument('--abstain-share', type=float, default=0.33,
                        help='fraction of the repair rule that is abstention rows')
    args = parser.parse_args()
    global ABSTAIN_SHARE
    ABSTAIN_SHARE = args.abstain_share

    rows = load_pool()
    meta = load_metadata(rows)
    print(f'pool {len(rows)} rows | metadata joined for {len(meta)} source ids')

    tok = _load_tokenizer(SRC)
    ntok = {}
    for row in rows:
        ntok[row['id']] = len(tok(row['text'], add_special_tokens=False)['input_ids'])

    order = RULES[args.rule](rows, meta, random.Random(args.seed))
    if len(order) != len(rows):
        raise SystemExit(f'rule {args.rule} returned {len(order)} of {len(rows)} rows; a '
                         f'rule may reorder the pool but never drop from it')
    chosen, supervised = [], 0
    for row in order:
        if supervised >= args.budget:
            break
        chosen.append(row)
        supervised += ntok[row['id']] - 1        # next-token objective: len-1 targets
    if not chosen:
        raise SystemExit('no rows selected')

    # Guards. The pool is clean, so these confirm the subset did not become unclean --
    # and confirm the panel-class absence this programme relies on, rather than recall it.
    stems = panel_stems()
    blob = ' \n '.join(r['text'] for r in chosen)
    bblob = blind(blob)
    panel_classes = {'percentage_reasoning', 'ratio_proportion', 'time_duration'}
    leaks = {
        'panel_stem_verbatim': sum(1 for s in stems if s in blob),
        'panel_stem_blinded': sum(1 for s in stems if blind(s) in bblob),
        'panel_class_rows': sum(1 for r in chosen if r['reply_class'] in panel_classes),
        'duplicate_ids': len(chosen) - len({r['id'] for r in chosen}),
        'rows_not_from_pool': sum(1 for r in chosen if r['id'] not in {x['id'] for x in rows}),
    }
    for name, count in leaks.items():
        if count:
            raise SystemExit(f'CONTAMINATION: {name} = {count}; refusing to write the split')
    print('guards clear:', leaks)

    suffix = f'_a{int(round(args.abstain_share * 100))}' if args.rule == 'repair' else ''
    out = (ROOT / 'outputs' /
           f'splits_curated_{args.rule}{suffix}_b{args.budget}_s{args.seed}')
    if out.exists():
        raise SystemExit(f'split directory already exists (create-only): {out}')
    out.mkdir(parents=True, exist_ok=False)
    (out / 'train.jsonl').write_text(
        ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in chosen), encoding='utf-8')
    for name in ('dev.jsonl', 'task_probe.jsonl'):
        (out / name).write_text((POOL / name).read_text(encoding='utf-8'), encoding='utf-8')

    info = [meta[r['source_row_id']] for r in chosen]
    tally = lambda key: dict(collections.Counter(str(m.get(key)) for m in info).most_common())
    manifest = {
        'schema': 1, 'evidence_kind': 'exploratory_curated_selection_split',
        'note': 'A SUBSET of splits_large TRAIN chosen by a named rule at a fixed supervised-'
                'token budget. Rows are verbatim from the pool, so family/id disjointness '
                'against DEV and the task probe is inherited. DEV and task_probe are copied '
                'byte-for-byte, so the only difference from the banked runs is which TRAIN '
                'rows exist. Sized to the budget so arm_dose consumes ALL of it at every '
                'seed -- the seed then changes only the order, which separates selection '
                'from the draw.',
        'selection_rule': args.rule,
        'selection_seed': args.seed,
        'selection_inputs': 'v3.0 curation metadata rejoined via source_row_id; the panel '
                            'is never consulted',
        'derived_from': str(POOL.relative_to(REPO)),
        'rows': len(chosen),
        'supervised_target_tokens_available': supervised,
        'max_usable_budget': supervised,
        'requested_budget': args.budget,
        'raw_tokens': sum(ntok[r['id']] for r in chosen),
        'composition': {k: tally(k) for k in
                        ('difficulty', 'capability_group', 'length_mode', 'domain',
                         'multi_turn', 'speech_act')},
        'reply_classes': dict(collections.Counter(r['reply_class'] for r in chosen)
                              .most_common()),
        'distinct_reply_classes': len({r['reply_class'] for r in chosen}),
        'distinct_template_families': len({meta[r['source_row_id']].get('template_family')
                                           for r in chosen}),
        'abstention_rows': sum(1 for r in chosen if is_abstention(r)),
        'abstention_row_share': sum(1 for r in chosen if is_abstention(r)) / len(chosen),
        'abstain_share_requested': args.abstain_share if args.rule == 'repair' else None,
        'contamination_guards': leaks,
        'dev_and_task_probe_copied_from': str(POOL),
        'licenses_registered_training': False, 'licenses_efficacy_claim': False,
    }
    (out / 'SPLIT_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'\nwrote {out}')
    print(f"  {len(chosen)} rows, {supervised:,} supervised tokens "
          f"({manifest['raw_tokens']:,} raw)")
    print(f"  {manifest['distinct_reply_classes']}/36 reply classes, "
          f"{manifest['distinct_template_families']} template families")
    print(f"  difficulty {manifest['composition']['difficulty']}")
    print(f"  capability {manifest['composition']['capability_group']}")
    print(f"  abstention rows {manifest['abstention_rows']} "
          f"({100 * manifest['abstention_row_share']:.1f}%)")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
