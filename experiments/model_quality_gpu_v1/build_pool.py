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

**Evaluation contracts come from the corpus's own answers, never invented.**

  * answer is a bare decimal (`9.`, `-15`, `.5`)  -> `scorer: number`
  * answer is short prose (<= 4 words)            -> `scorer: exact`, the literal answer
  * anything longer                               -> no `evaluation`; response CE only

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


def contract(answer: str) -> dict | None:
    """The evaluation specification implied by the corpus's own answer, or None."""
    text = answer.strip()
    if NUMERIC.fullmatch(text):
        return {'scorer': 'number', 'answers': [text], 'answerable': True}
    if len(text.split()) <= 4:
        return {'scorer': 'exact', 'answers': [text], 'answerable': True}
    return None


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
    dev, dev_families = [], set()
    for family in families:
        if len(dev) >= args.dev_rows:
            break
        group = sorted(by_family[family], key=lambda r: r['id'])
        pick = next((r for r in group if contract(r['messages'][-1]['content']) is not None),
                    group[0])
        dev.append(pick)
        dev_families.add(family)
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
