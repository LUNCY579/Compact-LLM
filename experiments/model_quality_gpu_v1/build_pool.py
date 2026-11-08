"""Build the MQ-001 candidate corpus: TRAIN, DEV and the open historical exclusion bank.

MQ-001 ships the training machinery and a documented 2,000-example quality pool that was
never constructed. This constructs it, and stops at the review gate: it emits candidate
JSONL only. `prepare` then compiles a frozen plan and writes `DATA_REVIEW_PENDING.json`,
which a **named human reviewer** must approve. Nothing here approves anything.

    python -m experiments.model_quality_gpu_v1.build_pool --out /absolute/mq-data-001

Source is `Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0`, the corpus this
programme has already characterised. Only its **single-turn** rows are used, because
MQ-001 v1 accepts exactly one user/assistant exchange with an optional leading system turn;
multi-turn rows are excluded rather than flattened, since flattening would silently change
the task.

**Evaluation contracts come from the corpus's own answers, never invented, and are
punctuation-tolerant.**

  * answer of <= 6 words (ignoring trailing `.!?`) -> `scorer: exact` over the variant set
    `{bare, bare + '.', literal}`, so a full stop cannot decide correctness
  * anything longer                                -> no `evaluation`; response CE only

**DEV polar answers are balanced YES/NO**, globally to within one row and per class. Both
rules exist because the first pool lacked them: 47,648 of 50,964 corpus answers end in a
full stop, so a literal gold made punctuation the metric and the DEV baseline scored 1/41;
and the unbalanced draw put a YES gold on all three `contradiction_detection` rows, so a
model that simply stopped saying "No" scored 5/7 on the polar slice and was indistinguishable
from one that had learned the task. See `../../MQ001_FIRST_PAIR.md`.

No `answerable: false` / `["UNKNOWN"]` rows are produced. MQ-001 requires that such a task
*actually request the UNKNOWN contract in its prompt*, and v3's `insufficient_information`
rows ask for a natural knowledge-limit sentence instead. Relabelling them as epistemic
abstention would be exactly the mislabelling MQ-001's runbook warns against, so those rows
are kept and scored on their own literal answer. If an UNKNOWN-contract screen is wanted it
needs purpose-built prompts and its own review.

**The exclusion bank is the point of contact with RAB-001.** It carries the 150-row frozen
arithmetic panel and the `splits_large` training pool, so `overlap_audit` can prove MQ's
TRAIN and DEV touch neither. Panel rows are multi-turn; each is represented by its **final
user turn**, which is the question stem and therefore the thing that would actually leak.
That representation is for leakage detection only -- no exclusion-bank row is ever trained
or scored on.

DEV is held out by `template_family`, not by row, because different rows of one family
render the same stem with different fillers. Holding out rows would put paraphrases of a
DEV question into TRAIN and the id-level guard would not notice.
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import unicodedata
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

V3 = REPO / 'Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0' / 'data' / 'sft'
V3_FILES = ('rineya_v3_reasoning_train_50000.jsonl',
            'rineya_v3_natural_communication_train_30000.jsonl',
            'rineya_v3_integrated_reasoning_dialogue_train_30000.jsonl')
RAB = REPO / 'experiments' / 'architecture_benchmark_001'
NUMERIC = re.compile(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)')
BLOCKED = ('<|im_start|>', '<|im_end|>', '<think>', '</think>', '<|endoftext|>')

normal = lambda t: ' '.join(unicodedata.normalize('NFKC', t).casefold().split())

# Longest answer, in words after trailing punctuation is stripped, that still gets a
# scoreable `exact` contract.
#
# MEASURED, not chosen for neatness. The first pool used 4, which silently made the DEV
# instrument polarity-biased: contradiction_detection has exactly two answers in the whole
# corpus -- 'Yes, they conflict.' (3 words) and 'No, they can both be true.' (6 words) -- so
# a 4-word cap admitted every YES and excluded every NO. That class could only ever
# contribute YES rows, and a model that stopped saying "No" swept it.
#
# Coverage and both-polarity class count against the threshold:
#     4 words -> 50.0% scoreable, 5 classes with BOTH polarities
#     5        -> 55.9%,          5
#     6        -> 63.4%,          8   <- contradiction_detection becomes balanced here
#     8        -> 82.7%,          8
#    10        -> 95.0%,          8
# Six is the smallest value that removes the bias. Beyond it, coverage rises but no further
# class gains its missing polarity, and longer literal golds only get more brittle.
MAX_ANSWER_WORDS = 6


def core(answer: str) -> str:
    """The answer with trailing sentence punctuation removed."""
    return answer.strip().rstrip('.!?').strip()


def polarity(answer: str) -> str | None:
    """YES / NO / None, from the first word. Drives DEV polarity balancing."""
    if re.match(r'^\s*yes\b', answer, re.I):
        return 'YES'
    if re.match(r'^\s*no\b', answer, re.I):
        return 'NO'
    return None


def contract(answer: str) -> dict | None:
    """The evaluation specification implied by the corpus's own answer, or None.

    **Punctuation-tolerant by construction.** 47,648 of the corpus's 50,964 single-turn
    answers end in a full stop and only 100 end in nothing, so a gold of the literal answer
    makes a trailing period decide correctness. It measurably did: the first pool's DEV
    baseline scored 1/41, and of the whole-sequence arm's six gains, five were the model
    learning to add punctuation rather than learning anything. `classify()` accepts a LIST of
    answers and tests `text.strip() in [a.strip() for a in answers]`, so both surface forms
    are offered and the row scores content instead of style.

    **`number` is deliberately not used.** It would seem the natural fit for numeric answers,
    but `classify()` requires the *prediction* to `re.fullmatch` a bare-decimal pattern, which
    a trailing period breaks -- reintroducing exactly the defect being fixed, on the one side
    a variant list cannot reach. `exact` over {bare, bare + '.'} accepts both and is
    predictable. The cost is that '18.0' no longer matches a gold of '18'; this model emits
    neither, and a silent punctuation trap is worse than a narrow accepted set.
    """
    text = core(answer)
    if not text or len(text.split()) > MAX_ANSWER_WORDS:
        return None
    variants = {text, f'{text}.', answer.strip()}
    return {'scorer': 'exact', 'answers': sorted(variants), 'answerable': True}


def usable(messages: list[dict]) -> bool:
    if [m['role'] for m in messages] not in (['user', 'assistant'],
                                             ['system', 'user', 'assistant']):
        return False
    for message in messages:
        text = message.get('content')
        if not isinstance(text, str) or not text.strip() or any(b in text for b in BLOCKED):
            return False
    return not messages[-1]['content'].startswith('\n')


def load_v3() -> list[dict]:
    rows = []
    for name in V3_FILES:
        for line in (V3 / name).read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['metadata'].get('multi_turn'):
                continue
            messages = [{'role': m['role'], 'content': m['content']} for m in row['messages']]
            if not usable(messages):
                continue
            rows.append({'id': row['id'],
                         'template_family': row['metadata']['template_family'],
                         'messages': messages,
                         'reply_class': row['metadata'].get('reply_class'),
                         'difficulty': row['metadata'].get('difficulty')})
    rows.sort(key=lambda r: r['id'])
    return rows


def load_exclusions() -> list[dict]:
    """RAB-001's frozen panel and training pool, as single-turn leakage probes."""
    out, seen = [], set()
    panel = [json.loads(l) for l in (RAB / 'data' / 'monitor_probe.jsonl')
             .read_text(encoding='utf-8-sig').splitlines() if l.strip()]
    for row in panel:
        user = [m for m in row['messages'] if m['role'] == 'user'][-1]['content']
        answer = row['answer']
        if not user.strip() or not answer.strip():
            continue
        if any(b in user or b in answer for b in BLOCKED):
            continue
        rid = f'rab-panel::{row["id"]}'
        out.append({'id': rid, 'template_family': f'rab-panel::{row["group_id"]}',
                    'messages': [{'role': 'user', 'content': user},
                                 {'role': 'assistant', 'content': answer.strip()}]})
        seen.add(rid)

    pool = [json.loads(l) for l in (RAB / 'outputs' / 'splits_large' / 'train.jsonl')
            .read_text(encoding='utf-8').splitlines() if l.strip()]
    wanted = {r['source_row_id'] for r in pool}
    families = {r['source_row_id']: r['group_id'] for r in pool}
    for name in V3_FILES + ('rineya_v1_v2_preservation_slice_14000.jsonl',):
        path = V3 / name
        if not path.is_file():
            continue
        for line in path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row['id'] not in wanted:
                continue
            messages = [{'role': m['role'], 'content': m['content']} for m in row['messages']]
            user = [m for m in messages if m['role'] == 'user']
            if not user or not messages[-1]['role'] == 'assistant':
                continue
            probe = [{'role': 'user', 'content': user[-1]['content']},
                     {'role': 'assistant', 'content': messages[-1]['content']}]
            if not usable(probe):
                continue
            rid = f'rab-pool::{row["id"]}'
            if rid in seen:
                continue
            seen.add(rid)
            out.append({'id': rid,
                        'template_family': f'rab-pool::{families.get(row["id"], row["id"])}',
                        'messages': probe})
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, help='new directory for the candidate JSONL')
    parser.add_argument('--train-rows', type=int, default=2000)
    parser.add_argument('--dev-rows', type=int, default=64)
    parser.add_argument('--seed', type=int, default=17)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'output directory already exists (create-only): {out}')

    rows = load_v3()
    print(f'v3 single-turn usable rows: {len(rows):,}')
    excluded = load_exclusions()
    print(f'historical exclusion bank : {len(excluded):,} '
          f'({sum(1 for r in excluded if r["id"].startswith("rab-panel"))} panel, '
          f'{sum(1 for r in excluded if r["id"].startswith("rab-pool"))} pool)')

    # Anything a history row would leak is removed from the candidate pool BEFORE selection,
    # on id, family, normalised prompt and digit-blinded prompt -- the four keys the
    # compiler audits. Doing it here turns a late hard failure into a reported count.
    banned_family = {r['template_family'] for r in excluded}
    banned_prompt, banned_blind = set(), set()
    for r in excluded:
        p = normal(next(m['content'] for m in r['messages'] if m['role'] == 'user'))
        banned_prompt.add(p)
        banned_blind.add(re.sub(r'\d', '#', p))
    kept, dropped = [], collections.Counter()
    for row in rows:
        prompt = normal(next(m['content'] for m in row['messages'] if m['role'] == 'user'))
        if row['template_family'] in banned_family:
            dropped['family_in_history'] += 1
        elif prompt in banned_prompt:
            dropped['prompt_in_history'] += 1
        elif re.sub(r'\d', '#', prompt) in banned_blind:
            dropped['digit_blind_prompt_in_history'] += 1
        else:
            kept.append(row)
    print(f'candidates after removing history overlap: {len(kept):,}  dropped {dict(dropped)}')
    if not kept:
        raise SystemExit('no candidate rows survive the historical exclusion')

    # DEV is held out by FAMILY. Different rows of one family render the same stem with
    # different fillers, so holding out rows alone would leave paraphrases in TRAIN.
    by_family = collections.defaultdict(list)
    for row in kept:
        by_family[row['template_family']].append(row)
    families = sorted(by_family)
    rng = random.Random(args.seed)
    rng.shuffle(families)

    # One row per family, preferring a row whose own answer yields a scoreable contract.
    # A first pass took min-id and left 31 of 64 DEV rows with no `evaluation` at all,
    # halving the primary metric's sample on an instrument whose seed-to-seed noise this
    # programme has repeatedly measured as large. Selection is on answer SHAPE -- a corpus
    # property, fixed before any model runs -- never on a model outcome.
    #
    # This makes DEV a short-answer instrument rather than a representative sample of the
    # corpus, and that is recorded in the manifest. It does not bias the comparison: every
    # arm is scored on the same rows.
    # One row per family, preferring a scoreable contract, and BALANCING POLARITY.
    #
    # The first pool did not balance and drew a DEV whose polar golds ran 5 YES / 2 NO with
    # all three contradiction_detection rows YES. A model that simply stopped saying "No"
    # then scored 5/7 on that slice and looked like it had learned contradiction detection --
    # which is exactly what the first trained pair did, and the instrument could not tell the
    # difference, because it contained no negative case to catch it.
    #
    # The corpus supports balance easily: contradiction_detection is 566 YES / 566 NO,
    # logical_implication 493/513, evidence_inference 330/318. The imbalance was a sampling
    # accident, not a limitation. Since a template family usually contains rows of both
    # polarities, polarity is chosen by picking WITHIN the family, greedily against the
    # running count.
    dev, dev_families = [], set()
    tally = collections.Counter()
    for family in families:
        if len(dev) >= args.dev_rows:
            break
        group = sorted(by_family[family], key=lambda r: r['id'])
        scoreable = [r for r in group if contract(r['messages'][-1]['content']) is not None]
        pool_rows = scoreable or group
        want = 'NO' if tally['YES'] > tally['NO'] else 'YES'
        pick = (next((r for r in pool_rows
                      if polarity(r['messages'][-1]['content']) == want), None)
                or next((r for r in pool_rows
                         if polarity(r['messages'][-1]['content']) is None), None)
                or pool_rows[0])
        tally[polarity(pick['messages'][-1]['content']) or 'non-polar'] += 1
        dev.append(pick)
        dev_families.add(family)

    # Repair pass. The greedy assignment above cannot always hit balance, because a family
    # may contain only one polarity, and by then the counts are already committed. So after
    # selection, swap within families that can supply the under-represented polarity: the
    # family stays in DEV, only which of its rows represents it changes. That preserves the
    # family-level holdout exactly while moving the polarity counts.
    # Two constraints must hold together, and satisfying only the global one is not enough:
    # a DEV can be 4 YES / 4 NO overall while contradiction_detection is all-YES and
    # estimation_sanity is all-NO, which leaves the per-class blind spot intact. So repair
    # per-class one-sidedness FIRST, then fix any residual global skew using classes that
    # contribute a single polar row, where a swap cannot create one-sidedness.
    pol_of = lambda r: polarity(r['messages'][-1]['content'])

    def alternatives(index: int, target: str):
        group = sorted(by_family[dev[index]['template_family']], key=lambda r: r['id'])
        scoreable = [r for r in group if contract(r['messages'][-1]['content']) is not None]
        return next((r for r in (scoreable or group) if pol_of(r) == target), None)

    def apply_swap(index: int, replacement) -> None:
        tally[pol_of(dev[index]) or 'non-polar'] -= 1
        tally[pol_of(replacement) or 'non-polar'] += 1
        dev[index] = replacement

    def polar_by_class():
        out = collections.defaultdict(list)
        for index, row in enumerate(dev):
            if pol_of(row):
                out[row['reply_class']].append(index)
        return out

    for _ in range(500):
        classes = polar_by_class()
        one_sided = [(cls, idxs) for cls, idxs in sorted(classes.items())
                     if len(idxs) >= 2 and len({pol_of(dev[i]) for i in idxs}) == 1]
        if one_sided:
            cls, idxs = one_sided[0]
            want = 'NO' if pol_of(dev[idxs[0]]) == 'YES' else 'YES'
            moved = False
            for i in idxs:
                alt = alternatives(i, want)
                if alt is not None:
                    apply_swap(i, alt)
                    moved = True
                    break
            if not moved:
                raise SystemExit(
                    f'DEV class {cls!r} is one-sided ({pol_of(dev[idxs[0]])}) and none of its '
                    f'held-out families can supply the opposite polarity. A model that merely '
                    f'flips polarity would score as though it had learned this task.')
            continue
        if abs(tally['YES'] - tally['NO']) <= 1:
            break
        short, long = ('NO', 'YES') if tally['YES'] > tally['NO'] else ('YES', 'NO')
        # Only swap rows whose class contributes exactly one polar row; swapping those can
        # never make a class one-sided.
        singles = [idxs[0] for cls, idxs in sorted(classes.items())
                   if len(idxs) == 1 and pol_of(dev[idxs[0]]) == long]
        alt = next(((i, a) for i in singles for a in [alternatives(i, short)] if a is not None),
                   None)
        if alt is None:
            raise SystemExit(
                f'DEV polarity stuck at {tally["YES"]} YES vs {tally["NO"]} NO with no safe '
                f'swap available. An unbalanced polar slice lets a model that merely flips '
                f'polarity score as though it had learned the task.')
        apply_swap(*alt)
    else:
        raise SystemExit('polarity repair did not converge')
    print(f'DEV polarity balanced: {tally["YES"]} YES / {tally["NO"]} NO / '
          f'{tally["non-polar"]} non-polar')
    train_pool = [r for r in kept if r['template_family'] not in dev_families]
    rng.shuffle(train_pool)
    train = sorted(train_pool[:args.train_rows], key=lambda r: r['id'])
    print(f'DEV {len(dev)} rows from {len(dev_families)} held-out families | '
          f'TRAIN {len(train)} rows from {len({r["template_family"] for r in train})} families')

    def emit(row, with_contract):
        record = {'id': row['id'], 'template_family': row['template_family'],
                  'messages': row['messages']}
        if with_contract:
            spec = contract(row['messages'][-1]['content'])
            if spec is not None:
                record['evaluation'] = spec
        return record

    dev_records = [emit(r, True) for r in dev]

    # A class contributing two or more polar DEV rows must not be one-sided. The global
    # balance above can be satisfied while one class stays all-YES and another all-NO, which
    # would leave exactly the per-class blind spot that made "learned contradiction
    # detection" indistinguishable from "learned to say Yes" the first time.
    per_class = collections.defaultdict(collections.Counter)
    for row in dev:
        pol = polarity(row['messages'][-1]['content'])
        if pol:
            per_class[row['reply_class']][pol] += 1
    one_sided = {c: dict(v) for c, v in per_class.items()
                 if sum(v.values()) >= 2 and len(v) == 1}
    if one_sided:
        raise SystemExit(f'one-sided polar classes in DEV: {one_sided}. A model that flips '
                         f'polarity would score as though it had learned these tasks.')
    train_records = [emit(r, False) for r in train]
    scored = collections.Counter(r.get('evaluation', {}).get('scorer', 'not_scored')
                                 for r in dev_records)
    print(f'DEV contracts: {dict(scored)}')

    out.mkdir(parents=True, exist_ok=False)
    for name, records in (('train.jsonl', train_records), ('dev.jsonl', dev_records),
                          ('open-history-exclusions.jsonl', excluded)):
        (out / name).write_text(
            ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records), encoding='utf-8')
        size = (out / name).stat().st_size
        print(f'  wrote {name:32s} {len(records):6,} rows  {size/1024/1024:.2f} MiB')
        if size > 64 * 1024 * 1024:
            raise SystemExit(f'{name} exceeds the compiler\'s 64 MiB v1 limit')
        if len(records) > 20000:
            raise SystemExit(f'{name} exceeds the compiler\'s 20,000-row v1 limit')

    manifest = {
        'schema': 1, 'experiment': 'MQ-001', 'kind': 'candidate_corpus_pending_review',
        'source_corpus': 'Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0',
        'single_turn_only': True, 'multi_turn_rows_excluded_not_flattened': True,
        'seed': args.seed,
        'rows': {'train': len(train_records), 'dev': len(dev_records),
                 'excluded': len(excluded)},
        'dev_holdout': 'by template_family',
        'dev_contracts': dict(scored),
        'dev_polarity': {'YES': tally['YES'], 'NO': tally['NO'],
                         'non_polar': tally['non-polar']},
        'dev_polarity_by_class': {c: dict(v) for c, v in sorted(per_class.items())},
        'punctuation_tolerant_golds': True,
        'contract_note': 'exact scorer over {bare, bare+".", literal}. `number` is not used: '
                         'classify() requires the PREDICTION to fullmatch a bare-decimal '
                         'pattern, which a trailing period breaks -- the very defect being '
                         'fixed, on the side a variant list cannot reach.',
        'dev_selection': 'one row per held-out family, preferring a row whose own answer '
                         'yields a scoreable contract. DEV is therefore a SHORT-ANSWER '
                         'instrument, not a representative sample of the corpus. Selection '
                         'is on answer shape, a corpus property fixed before any model runs, '
                         'never on a model outcome; and it cannot bias the arm comparison '
                         'because every arm is scored on the same rows.',
        'contracts_derived_from': "the corpus's own assistant answer; none invented",
        'unknown_contract_rows': 0,
        'unknown_contract_note': 'MQ-001 requires an UNKNOWN task to request that contract in '
                                 'its prompt. v3 insufficient_information rows ask for a '
                                 'natural knowledge-limit sentence instead, so relabelling '
                                 'them as epistemic abstention was refused.',
        'history_overlap_removed': dict(dropped),
        'exclusion_bank': 'RAB-001 frozen panel (final user turn) and splits_large pool',
        'review_status': 'PENDING -- no approval is expressed or implied by this file',
    }
    (out / 'POOL_MANIFEST.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(f'\nwrote {out}')
    print('  next: python -m experiments.model_quality_gpu_v1 prepare --source Supa \\')
    print(f'          --train {out}/train.jsonl --dev {out}/dev.jsonl \\')
    print(f'          --excluded {out}/open-history-exclusions.jsonl --out <new dir>')
    print('  then a NAMED HUMAN REVIEWER must approve DATA_REVIEW_PENDING.json.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
