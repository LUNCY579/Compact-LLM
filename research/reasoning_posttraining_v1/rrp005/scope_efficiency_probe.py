"""Tokenization-only TRAIN study while the independent source baseline runs.

No pretrained model is loaded. Public-scope filtering is checked against the parsed
canonical world. This does not demonstrate long-context generalization or model efficacy.
"""
import collections
import gzip
import json
from pathlib import Path
import sys
import statistics

ROOT = Path('C:/Users/clclc/BusinessProject/ChatBubble-ced-native-loader-qualification')
sys.path.insert(0, str(ROOT))
from experiments.reasoning_posttraining_v1.repair_preservation import tokenizer_only
from experiments.reasoning_posttraining_v1.sources import import_forge_v2, stable_hash
from experiments.reasoning_posttraining_v1.views import scope_only
from experiments.reasoning_posttraining_v1.serialization import compile_row
from experiments.model_quality_gpu_v1.integrity import digest, write_new

OUT = ROOT.parent / 'rineya-dataset-retention-records/scope-efficiency-001'
if OUT.exists():
    raise FileExistsError(OUT)
binding = json.loads((ROOT.parent / 'rineya-dataset-retention-records/a1-prepared-003/SOURCE_BINDING.json').read_text())
release = Path(binding['root'])
for rel in ('metadata/train.jsonl.gz', 'data/rl/train.jsonl.gz'):
    if stable_hash(release / rel) != binding['files'][rel]['sha256']:
        raise ValueError('TRAIN source changed')
modules = import_forge_v2(binding)
tok = tokenizer_only(ROOT / 'Supa')
buckets = collections.defaultdict(list)
with gzip.open(release / 'metadata/train.jsonl.gz', 'rt', encoding='utf8') as stream:
    for line in stream:
        row = json.loads(line)
        if row['split'] != 'train':
            raise ValueError('non-TRAIN source')
        key = (row['family'], row['reference'].get('status', 'ANSWER'))
        buckets[key].append((digest(['RRP-scope-efficiency-v1', 17, row['id']]), row))
        buckets[key].sort(key=lambda item: item[0])
        del buckets[key][32:]
selected = {row['id']: row for bucket in buckets.values() for _, row in bucket}
records = []
with gzip.open(release / 'data/rl/train.jsonl.gz', 'rt', encoding='utf8') as stream:
    for line in stream:
        policy = json.loads(line)
        if policy['id'] not in selected:
            continue
        meta = selected[policy['id']]
        world = modules['text'].parse_prompt(policy['prompt'])
        if not modules['oracle'].verify_trace(world, meta['reference'], meta['trace_steps']):
            raise ValueError('source structured trace is not valid')
        if not modules['rewards'].score_response(world, meta['reference_text'], certificate=meta['certificate'])['correct']:
            raise ValueError('source final target is not valid')
        short, proof = scope_only(policy['prompt'], modules['text'].parse_prompt, modules['text'].canonical_case)
        row = {'id': policy['id'], 'parent_id': meta['group_id'], 'family': meta['family'],
               'stratum': meta['reference'].get('status', 'ANSWER'), 'scope_proof': proof,
               'source_split': 'train', 'views': {}}
        for view, prompt in (('full', policy['prompt']), ('scope_only', short)):
            for mode in ('direct', 'reasoning'):
                messages = [dict(m) for m in prompt]
                if mode == 'reasoning':
                    messages[0]['content'] = modules['common'].SYSTEM_REASON
                compiled = compile_row(messages, meta['reference_text'], tok, mode=mode,
                    trace=meta['teacher_trace'].strip() if mode == 'reasoning' else None,
                    max_sequence=262144, completion_reserve=128)
                row['views'][view + '_' + mode] = {
                    'processed_tokens': len(compiled['ids']),
                    'prompt_tokens': len(compiled['prompt_ids']),
                    'supervised_tokens': compiled['supervised_tokens'],
                    'fits_2048': len(compiled['ids']) <= 2048 and len(compiled['prompt_ids']) + 128 <= 2048,
                    'complete_generation_tokens': compiled['answer_end'] - compiled['answer_start'] + 1,
                }
        records.append(row)
names = list(records[0]['views'])
summary = {}
for name in names:
    rows = [r['views'][name] for r in records]
    summary[name] = {'rows': len(rows), 'fits_2048': sum(r['fits_2048'] for r in rows),
                     'mean_processed_tokens': statistics.mean(r['processed_tokens'] for r in rows),
                     'mean_supervised_tokens': statistics.mean(r['supervised_tokens'] for r in rows),
                     'max_complete_generation_tokens': max(r['complete_generation_tokens'] for r in rows),
                     'fit_by_family': {family: sum(r['views'][name]['fits_2048'] for r in records if r['family'] == family)
                                       for family in sorted({r['family'] for r in records})}}
OUT.mkdir()
write_new(OUT / 'TOKEN_SURVEY.json', {'source_tree_sha256': binding['tree_sha256'],
    'selection': '32 smallest sha256 IDs per family/status, seed17; TRAIN only; selected before model outcomes',
    'worlds': len(records), 'summary': summary, 'records': records,
    'model_calls': 0, 'optimizer_steps': 0,
    'inference_generation_budget_qualified': False, 'quality_claim': False,
    'claim_limit': 'scope-only is an explicitly easier same-world input; reasoning targets require their own native training/decoding qualification'})
print(json.dumps({'worlds': len(records), 'summary': summary, 'model_calls': 0}, indent=2))
