"""Independent exact-tokenizer geometry, pair invariance and teacher budget audit.

Reads an explicitly supplied generated proposal; never loads model weights or
marks the corpus reviewed. CPU zeros are used only to measure safetensors header
overhead at the actual per-row shapes, and no synthetic cache file is written.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import torch
from safetensors.torch import save
from transformers import AutoTokenizer

experiment = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(experiment))
from sbhm.data import encode, ordered

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--corpus', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--steps', type=int)
parser.add_argument('--seed', type=int, default=17)
args = parser.parse_args()
corpus = args.corpus.resolve()
design = json.loads((corpus/'design.json').read_text(encoding='utf-8'))
manifest = json.loads((corpus/'manifest.json').read_text(encoding='utf-8'))
source = Path(design['tokenizer']['path'])
config = json.loads((source/'config.json').read_text(encoding='utf-8'))
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
assert tokenizer.is_fast and version('transformers') == '4.51.3'
assert config['hidden_size'] == 1024 and config['head_dim'] == 128
all_rows, records, pairs, counterfactuals = {}, [], defaultdict(list), defaultdict(list)


def token_span(offsets, prompt_offset, chars):
    start, end = [prompt_offset+value for value in chars]
    overlap = [i for i, (left, right) in enumerate(offsets) if right > start and left < end]
    assert overlap
    return [overlap[0], overlap[-1]+1]


for split in ('train', 'dev'):
    rows = [json.loads(line) for line in (corpus/f'{split}.jsonl').read_text(encoding='utf-8').splitlines()]
    all_rows[split] = rows
    for row in rows:
        item = design['rows'][row['id']]
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert rendered.count(row['prompt']) == 1
        prompt_offset = rendered.index(row['prompt'])
        encoded = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        ids, labels = encode(tokenizer, row, training=True)
        prompt_length = len(encoded['input_ids'])
        answer_eos_length = ids.shape[1]-prompt_length
        assert ids[0, :prompt_length].tolist() == encoded['input_ids']
        assert (labels[0, :prompt_length] == -100).all()
        assert labels[0, prompt_length:].tolist() == ids[0, prompt_length:].tolist()
        assert labels[0, -1].item() == tokenizer.eos_token_id
        assert answer_eos_length <= 32
        assert prompt_length <= 2048 and (split != 'train' or ids.shape[1] <= 512)
        stored = item['token_geometry']
        assert stored['prompt_tokens'] == prompt_length
        assert stored['target_tokens']+1 == answer_eos_length
        assert stored['train_tokens'] == ids.shape[1]
        record = {'id': row['id'], 'split': split, 'role': item['role'],
                  'category': item['category'], 'prompt_tokens': prompt_length,
                  'answer_plus_eos_tokens': answer_eos_length, 'training_input_tokens': ids.shape[1]}
        if item['role'] != 'retention':
            blocks = item['block_spans']
            for label, key in (('evidence', 'evidence_block_text'), ('filler', 'filler_block_text'), ('query', 'query_text')):
                start, end = blocks[label]
                assert row['prompt'][start:end] == item[key]
            block_tokens = {key: token_span(encoded['offset_mapping'], prompt_offset, chars)
                            for key, chars in blocks.items()}
            facts = []
            for fact in item['evidence_spans']:
                assert row['prompt'][fact['start_char']:fact['end_char']] == fact['record_text']
                facts.append(token_span(encoded['offset_mapping'], prompt_offset,
                                        [fact['start_char'], fact['end_char']]))
            required = [span for span, fact in zip(facts, item['evidence_spans']) if fact['required']]
            assert required
            gap = block_tokens['query'][0]-block_tokens['evidence'][1]
            earliest = prompt_length-min(span[0] for span in required)
            latest = prompt_length-max(span[1] for span in required)
            assert block_tokens == stored['block_token_spans'] and facts == stored['evidence_token_spans']
            assert stored['evidence_to_query_gap_tokens'] == gap
            assert stored['earliest_required_to_answer_tokens'] == earliest
            assert stored['latest_required_to_answer_tokens'] == latest
            assert stored['rendered_prompt_sha256'] == sha256(rendered.encode()).hexdigest()
            if item['role'] == 'near_control':
                assert earliest <= 128
            else:
                assert item['role'] == 'distant_heldout'
                low, high = stored['registered_gap_band']
                assert low <= gap <= high and gap > 128
            record.update(evidence_to_query_gap_tokens=gap,
                          earliest_required_to_answer_tokens=earliest,
                          latest_required_to_answer_tokens=latest)
            pairs[(split, item['position_pair_id'])].append((row, item, record))
            counterfactuals[(split, item['world_id'], item['role'])].append((row, item))
        records.append(record)

for key, pair in pairs.items():
    assert len(pair) == 2 and {info['role'] for _, info, _ in pair} == {'near_control', 'distant_heldout'}
    first, second = pair
    assert first[0]['target'] == second[0]['target']
    assert first[2]['prompt_tokens'] == second[2]['prompt_tokens']
    for field in ('evidence_block_text', 'filler_block_text', 'query_text', 'query', 'events'):
        assert first[1][field] == second[1][field]

for key, pair in counterfactuals.items():
    assert len(pair) == 2 and {item['counterfactual_id'] for _, item in pair} == {0, 1}
    first, second = pair
    assert first[0]['target'] != second[0]['target']
    assert first[1]['filler_block_text'] == second[1]['filler_block_text']
    assert first[1]['query_text'] == second[1]['query_text'] and first[1]['query'] == second[1]['query']
    first_shell = first[0]['prompt'].replace(first[1]['evidence_block_text'], '<EVIDENCE_BLOCK>')
    second_shell = second[0]['prompt'].replace(second[1]['evidence_block_text'], '<EVIDENCE_BLOCK>')
    assert first_shell == second_shell

for field in ('id', 'family_id', 'prompt'):
    normalize = lambda value: ' '.join(value.casefold().split())
    assert not ({normalize(row[field]) for row in all_rows['train']} &
                {normalize(row[field]) for row in all_rows['dev']})
for field in ('world_id', 'template_id'):
    assert not ({design['rows'][row['id']][field] for row in all_rows['train']} &
                {design['rows'][row['id']][field] for row in all_rows['dev']})

steps = args.steps or 2*len(all_rows['train'])
schedule = ordered(all_rows['train'], args.seed, steps)
exposures = Counter(row['id'] for row in schedule)
assert set(exposures) == {row['id'] for row in all_rows['train']}
train_geometry = {record['id']: record for record in records if record['split'] == 'train'}
headers = {}
for length in sorted({train_geometry[row_id]['training_input_tokens'] for row_id in exposures}):
    tensors = {name: torch.zeros((1, length, width), dtype=torch.float32, device='cpu')
               for name, width in (('hidden', config['hidden_size']), ('target', config['hidden_size']),
                                   ('cos', config['head_dim']), ('sin', config['head_dim']))}
    payload = sum(tensor.numel()*tensor.element_size() for tensor in tensors.values())
    serialized = save(tensors)
    headers[length] = len(serialized)-payload
    del tensors, serialized
unique_tokens = sum(train_geometry[row_id]['training_input_tokens'] for row_id in exposures)
payload = unique_tokens * (2*config['hidden_size']+2*config['head_dim'])*4
header_total = sum(headers[train_geometry[row_id]['training_input_tokens']] for row_id in exposures)
assert payload+header_total <= 512*2**20
summary = {}
for split in ('train', 'dev'):
    subset = [record for record in records if record['split'] == split]
    distant = [record for record in subset if record['role'] == 'distant_heldout']
    near = [record for record in subset if record['role'] == 'near_control']
    summary[split] = {'rows': len(subset), 'roles': dict(Counter(record['role'] for record in subset)),
                     'prompt_tokens_max': max(record['prompt_tokens'] for record in subset),
                     'training_input_tokens_max': max(record['training_input_tokens'] for record in subset),
                     'answer_plus_eos_tokens_max': max(record['answer_plus_eos_tokens'] for record in subset),
                     'distant_gap_tokens_min': min(record['evidence_to_query_gap_tokens'] for record in distant),
                     'distant_gap_tokens_max': max(record['evidence_to_query_gap_tokens'] for record in distant),
                     'near_earliest_required_to_answer_tokens_max': max(record['earliest_required_to_answer_tokens'] for record in near)}
receipt = {
    'created_utc': datetime.now(timezone.utc).isoformat(),
    'status': 'PASS_TOKEN_GEOMETRY_PAIR_INVARIANCE_TEACHER_BUDGET_ONLY',
    'corpus_path': str(corpus), 'source_path': str(source),
    'transformers_version': version('transformers'), 'tokenizers_version': version('tokenizers'),
    'source_tokenizer_files_sha256': {name: sha256((source/name).read_bytes()).hexdigest()
                                    for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json')},
    'corpus_files_sha256': {name: sha256((corpus/name).read_bytes()).hexdigest()
                            for name in ('manifest.json', 'design.json', 'train.jsonl', 'dev.jsonl')},
    'summary': summary, 'position_pairs_checked': len(pairs),
    'counterfactual_pairs_checked': len(counterfactuals),
    'training_exposure': {'seed': args.seed, 'steps_per_stage': steps,
                         'unique_train_rows': len(exposures), 'all_train_rows_covered': True,
                         'minimum_exposures_per_row': min(exposures.values()),
                         'maximum_exposures_per_row': max(exposures.values()),
                         'ordered_row_ids': [row['id'] for row in schedule]},
    'teacher_cache_estimate': {'dtype': 'float32', 'payload_bytes_per_training_token': 9216,
                              'unique_train_tokens': unique_tokens, 'tensor_payload_bytes': payload,
                              'safetensors_header_bytes': header_total,
                              'all_target_files_serialized_bytes': payload+header_total,
                              'cap_bytes': 512*2**20,
                              'header_overhead_by_length': headers,
                              'teacher_receipt_json_excluded': True,
                              'actual_teacher_activations_captured': False},
    'corpus_reviewed_flag_observed': manifest['reviewed_for_sbhm'],
    'corpus_manifest_modified': False, 'weights_loaded': False, 'gpu_tensors_allocated': False,
    'training_fit_or_behavior_result': False, 'records': records,
}
with args.output.open('x', encoding='utf-8') as stream:
    json.dump(receipt, stream, indent=2)
    stream.write('\n')
print(json.dumps({key: receipt[key] for key in ('status', 'summary', 'position_pairs_checked',
                 'counterfactual_pairs_checked', 'teacher_cache_estimate')}, indent=2))
