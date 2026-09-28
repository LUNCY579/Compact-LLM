"""Prospective TRAIN-only local-state SFT view/token survey, not a training approval."""
import collections
import gzip
import json
from pathlib import Path
import statistics
import sys

ROOT = Path('C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification')
sys.path.insert(0, str(ROOT))
from experiments.reasoning_posttraining_v1.repair_preservation import tokenizer_only
from experiments.reasoning_posttraining_v1.serialization import compile_row
from experiments.reasoning_posttraining_v1.sources import import_forge_v2, stable_hash
from experiments.model_quality_gpu_v1.integrity import digest, write_new

records_root = ROOT.parent / 'rineya-dataset-retention-records'
out = records_root / 'process-transition-probe-001'
if out.exists():
    raise FileExistsError(out)
binding = json.loads((records_root / 'a1-prepared-003/SOURCE_BINDING.json').read_text())
source = Path(binding['root']) / 'process/train_transitions.jsonl.gz'
if stable_hash(source) != binding['files']['process/train_transitions.jsonl.gz']['sha256']:
    raise ValueError('process release bytes changed')
modules = import_forge_v2(binding)
counts = collections.Counter()
per_parent = collections.defaultdict(dict)
for line in gzip.open(source, 'rt', encoding='utf8'):
    row = json.loads(line)
    if row['source_split'] != 'train' or row['parent_id'] != row['group_id']:
        raise ValueError('non-TRAIN or broken parent identity')
    expected = modules['oracle'].replay_transition(row['family'], row['before'], row['event'])
    if (expected == row['proposed_after']) != row['valid']:
        raise ValueError('process label fails the independent transition checker')
    counts[(row['family'], row['valid'])] += 1
    if row['valid']:
        old = per_parent[row['family']].get(row['parent_id'])
        if old is None or digest(['step-choice', row['id']]) < digest(['step-choice', old['id']]):
            per_parent[row['family']][row['parent_id']] = row
rules = {
    'ledger_replay': ('Apply one event to the supplied account balances. If status is not posted, '
        'leave all balances unchanged. add: add amount to account. scale: multiply account by amount. '
        'transfer: subtract amount from account and add it to other, simultaneously. Return only '
        'the complete resulting balances as a JSON object.'),
    'register_machine': ('Apply one instruction to the supplied registers. Registers are integers modulo 97. '
        'swap exchanges register and other simultaneously. add adds value to register. mix adds '
        '(2 times other plus value) to register. conditional adds value if other is greater than threshold, '
        'otherwise subtracts value. Reduce updated numeric registers modulo 97; preserve other registers. '
        'Return only the complete resulting registers as a JSON object.'),
    'prerequisite_plan': ('The state is an 8-bit flag mask. A supplied applicable action requires all pre_on '
        'bits set and all pre_off bits clear. Apply delete bits first, then add bits. Return only the '
        'resulting state as a JSON integer. Names and costs do not change the state.'),
}
tok = tokenizer_only(ROOT / 'Supa')
views, metrics = [], []
for family in sorted(per_parent):
    chosen = sorted(per_parent[family].values(), key=lambda r: digest(['process-probe-v1', 17, r['parent_id']]))[:128]
    for row in chosen:
        # Id, parent, valid label, proposed_after and source metadata NEVER enter prompt.
        prompt = [{'role': 'system', 'content': rules[family]},
                  {'role': 'user', 'content': json.dumps({'before': row['before'], 'event': row['event']},
                                                       sort_keys=True, separators=(',', ':'))}]
        final = json.dumps(row['proposed_after'], sort_keys=True, separators=(',', ':'))
        compiled = compile_row(prompt, final, tok, max_sequence=2048, completion_reserve=128)
        views.append({'id': 'RRP-process:' + row['id'], 'parent_id': row['parent_id'],
                      'source_id': row['id'], 'source_split': 'train', 'family': family,
                      'prompt': prompt, 'reference': final, 'gold_verified_by': 'oracle.replay_transition',
                      'view_role': 'CANDIDATE_NOT_ADMITTED_OR_TRAINED'})
        metrics.append({'id': views[-1]['id'], 'family': family,
                        'processed_tokens': compiled['processed_tokens'],
                        'supervised_tokens': compiled['supervised_tokens']})
out.mkdir()
with (out / 'CANDIDATE_VIEWS.jsonl').open('x', encoding='utf8', newline='\n') as f:
    for row in views:
        f.write(json.dumps(row, sort_keys=True) + '\n')
summary = {family: {'rows': sum(r['family'] == family for r in metrics),
                   'mean_processed': statistics.mean(r['processed_tokens'] for r in metrics if r['family'] == family),
                   'mean_supervised': statistics.mean(r['supervised_tokens'] for r in metrics if r['family'] == family),
                   'max_processed': max(r['processed_tokens'] for r in metrics if r['family'] == family)} for family in rules}
write_new(out / 'AUDIT.json', {'source_sha256': stable_hash(source),
    'all_process_labels_checked': sum(counts.values()), 'labels_by_family': {str(k): v for k, v in counts.items()},
    'unique_source_parents': {k: len(v) for k, v in per_parent.items()},
    'selected_unique_parents': len({r['parent_id'] for r in views}), 'selection_seed': 17,
    'selection': 'one lowest-hash positive step per parent, then 128 lowest-hash parents per family',
    'views_sha256': stable_hash(out / 'CANDIDATE_VIEWS.jsonl'), 'summary': summary, 'metrics': metrics,
    'model_calls': 0, 'optimizer_steps': 0, 'training_admitted': False,
    'limitations': ['TRAIN-only derivative; full replay/DEV overlap and mixture admission remain pending',
                   'local transition competence does not prove full-history reasoning transfer',
                   'prerequisite positives cover applicable actions only',
                   'related views are not independent worlds; parent grouping must survive every split',
                   'supervised-token mixture weights may differ greatly from processed-token quotas']})
print(json.dumps({'checked_process_labels': sum(counts.values()), 'selected_parents': len(views),
                  'summary': summary, 'model_calls': 0, 'training_admitted': False}, indent=2))
