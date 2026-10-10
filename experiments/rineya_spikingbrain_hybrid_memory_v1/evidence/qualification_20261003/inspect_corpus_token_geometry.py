"""Tokenize the explicit unreviewed synthetic proposal; no model or training."""
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

from transformers import AutoTokenizer

experiment = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(experiment))
from sbhm.data import encode

corpus = experiment/'corpora/distant_evidence_v1'
source = Path('C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/'
              'Rineya-Titan-Transition-1.0.0-devfreeze')
tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True, trust_remote_code=False)
manifest = json.loads((corpus/'manifest.json').read_text(encoding='utf-8'))
design = json.loads((corpus/'design.json').read_text(encoding='utf-8'))
assert manifest['reviewed_for_sbhm'] is False
assert tokenizer.is_fast
assert version('transformers') == '4.51.3'
limits = {'train_input_tokens': 512, 'dev_prompt_tokens': 2048, 'local_window_tokens': 128}
records = []
for split in ('train', 'dev'):
    for line in (corpus/f'{split}.jsonl').read_text(encoding='utf-8').splitlines():
        row = json.loads(line)
        item = design['rows'][row['id']]
        prompt_ids, _ = encode(tokenizer, row, training=False)
        train_ids, labels = encode(tokenizer, row, training=True)
        rendered = tokenizer.apply_chat_template([{'role': 'user', 'content': row['prompt']}],
                     tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert rendered.count(row['prompt']) == 1
        char_offset = rendered.index(row['prompt'])
        offsets = tokenizer(rendered, add_special_tokens=False, return_offsets_mapping=True)
        assert offsets['input_ids'] == prompt_ids[0].tolist()
        record = {'id': row['id'], 'split': split, 'family_id': row['family_id'],
                  'role': item['role'], 'category': item['category'],
                  'prompt_tokens': prompt_ids.shape[1],
                  'answer_plus_eos_tokens': train_ids.shape[1]-prompt_ids.shape[1],
                  'training_input_tokens': train_ids.shape[1],
                  'prompt_labels_all_ignored': bool((labels[0, :prompt_ids.shape[1]] == -100).all()),
                  'target_eos_label_correct': labels[0, -1].item() == tokenizer.eos_token_id,
                  'within_split_limit': (train_ids.shape[1] <= limits['train_input_tokens'] if split == 'train'
                                        else prompt_ids.shape[1] <= limits['dev_prompt_tokens'])}
        if item['role'] == 'distant_evidence':
            evidence_start = char_offset + item['evidence_start_char']
            evidence_end = char_offset + item['evidence_end_char']
            query_start = char_offset + item['query_start_char']
            query_end = char_offset + len(row['prompt'])
            evidence_tokens = [i for i, (start, end) in enumerate(offsets['offset_mapping'])
                               if end > evidence_start and start < evidence_end]
            query_tokens = [i for i, (start, end) in enumerate(offsets['offset_mapping'])
                            if end > query_start and start < query_end]
            assert evidence_tokens and query_tokens
            gap = query_tokens[0] - evidence_tokens[-1] - 1
            record.update(evidence_start_token_inclusive=evidence_tokens[0],
                          evidence_end_token_exclusive=evidence_tokens[-1]+1,
                          query_start_token_inclusive=query_tokens[0],
                          query_end_token_exclusive=query_tokens[-1]+1,
                          evidence_to_query_gap_tokens=gap,
                          evidence_beyond_window=gap > limits['local_window_tokens'])
        records.append(record)
summary = {}
for split in ('train', 'dev'):
    subset = [r for r in records if r['split'] == split]
    distant = [r for r in subset if r['role'] == 'distant_evidence']
    summary[split] = {'rows': len(subset), 'distant_rows': len(distant),
                     'short_retention_rows': len(subset)-len(distant),
                     'prompt_tokens_min': min(r['prompt_tokens'] for r in subset),
                     'prompt_tokens_max': max(r['prompt_tokens'] for r in subset),
                     'training_input_tokens_max': max(r['training_input_tokens'] for r in subset),
                     'distant_gap_tokens_min': min(r['evidence_to_query_gap_tokens'] for r in distant),
                     'distant_gap_tokens_max': max(r['evidence_to_query_gap_tokens'] for r in distant),
                     'all_rows_within_limit': all(r['within_split_limit'] for r in subset),
                     'all_distant_evidence_beyond_window': all(r['evidence_beyond_window'] for r in distant)}
receipt = {'created_utc': datetime.now(timezone.utc).isoformat(),
           'status': 'TOKEN_GEOMETRY_ONLY_CORPUS_REMAINS_UNREVIEWED',
           'source_path': str(source), 'corpus_path': str(corpus),
           'transformers_version': version('transformers'), 'tokenizers_version': version('tokenizers'),
           'source_files_sha256': {name: sha256((source/name).read_bytes()).hexdigest() for name in
                                   ('config.json', 'tokenizer.json', 'tokenizer_config.json')},
           'chat_template_sha256': sha256(tokenizer.chat_template.encode()).hexdigest(),
           'corpus_files_sha256': {name: sha256((corpus/name).read_bytes()).hexdigest() for name in
                                   ('manifest.json', 'design.json', 'train.jsonl', 'dev.jsonl')},
           'distance_definition': 'Tokens strictly between last token overlapping evidence and first token overlapping final query, in the actual rendered source chat prompt',
           'limits': limits, 'summary': summary, 'records': records,
           'reviewed_for_sbhm': False, 'weights_loaded': False, 'gpu_execution': False,
           'training_or_behavior_evidence': False}
result = Path(__file__).resolve().parent/'corpus_token_geometry.json'
result.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'status': receipt['status'], 'summary': summary, 'receipt': str(result)}, indent=2))
assert all(s['all_rows_within_limit'] and s['all_distant_evidence_beyond_window'] for s in summary.values())
