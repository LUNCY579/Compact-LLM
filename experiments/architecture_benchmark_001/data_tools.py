"""Offline, group-first recovery data preparation. No download or training launch.

Raw input records must declare source, immutable revision, license and group_id.
Review assertions are recorded as user declarations, not automatically verified facts.
Windows retain exact official token IDs, avoiding decode/re-tokenize boundary drift.
"""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import unicodedata
from .gates import GateClosed, protocol, write_once
from .source import audit_source, canonical_hash, sha256_file, source_lock


def group_split(group_id: str, seed: int = 17) -> str:
    if not isinstance(group_id, str) or not group_id.strip() or type(seed) is not int:
        raise ValueError('nonempty independent group and integer seed required')
    bucket = int(hashlib.sha256(f'{seed}\0{group_id}'.encode()).hexdigest()[:16], 16) % 1000
    return 'calibration' if bucket < 50 else 'dev' if bucket < 100 else 'train'


def token_windows(ids: list[int], max_tokens: int) -> list[list[int]]:
    if type(max_tokens) is not int or max_tokens < 2 or any(type(t) is not int or t < 0 for t in ids):
        raise ValueError('positive integer token IDs and context >= 2 required')
    # One-token overlap preserves every adjacent LM target exactly once. It does not
    # preserve all earlier context of a long document; this is a short-context recipe.
    return [ids[start:start+max_tokens] for start in range(0, len(ids)-1, max_tokens-1)]


def prepare_data(source: Path, input_path: Path, out: Path, *, seed: int = 17, max_tokens: int = 512) -> dict:
    from .run import _directory, _load_tokenizer, ensure_disjoint
    if type(max_tokens) is not int or not 2 <= max_tokens <= protocol()['decoding']['max_context']:
        raise GateClosed('data preparation may not expand the registered context')
    audit = audit_source(source)
    original_hash = sha256_file(input_path)
    tokenizer = _load_tokenizer(source)
    _directory(out, source)
    splits = {k: [] for k in ('calibration', 'train', 'dev')}
    seen_ids, owners = set(), {}
    sources, duplicate_documents, pending = {}, 0, 0
    with input_path.open(encoding='utf-8-sig') as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise GateClosed(f'raw row {number} is not an object')
            required = ('id', 'group_id', 'source', 'source_revision', 'license', 'text')
            if any(not isinstance(row.get(k), str) or not row[k].strip() for k in required):
                raise GateClosed(f'raw row {number} needs {required}; do not fabricate provenance')
            if row.get('split', 'train') != 'train':
                raise GateClosed('evaluation/calibration examples cannot be relabelled as recovery training')
            if row['id'] in seen_ids:
                raise GateClosed(f'duplicate document ID: {row["id"]}')
            seen_ids.add(row['id'])
            normalized = ' '.join(unicodedata.normalize('NFKC', row['text']).casefold().split())
            key = hashlib.sha256(normalized.encode()).hexdigest()
            if key in owners:
                if owners[key] != row['group_id']:
                    raise GateClosed('duplicate document crosses declared independent groups; resolve before splitting')
                duplicate_documents += 1
                continue
            owners[key] = row['group_id']
            split = group_split(row['group_id'], seed)
            review = row.get('review_status', 'pending')
            if review not in ('approved', 'pending'):
                raise GateClosed('review_status must be approved or pending, never silently reclassified')
            pending += review != 'approved'
            ids = tokenizer(row['text'], add_special_tokens=False)['input_ids']
            eos = tokenizer.eos_token_id
            if type(eos) is not int or not 0 <= eos < len(tokenizer):
                raise GateClosed('official tokenizer needs a valid document EOS token')
            if not ids or ids[-1] != eos:
                ids = [*ids, eos]
            provenance = {k: row[k] for k in ('source', 'source_revision', 'license')}
            metadata = {k: row[k] for k in ('source_url', 'source_metadata', 'generation_prompt', 'seed_document_id') if k in row}
            sources[canonical_hash(provenance)] = provenance
            for i, window in enumerate(token_windows(ids, max_tokens)):
                text = tokenizer.decode(window, skip_special_tokens=False)
                if not text.strip():
                    raise GateClosed('decoded inspection text for a token window is empty')
                splits[split].append({'id': f'{row["id"]}::window-{i:06d}', 'document_id': row['id'],
                    'group_id': row['group_id'], 'split': split, 'text': text, 'input_ids': window,
                    'tokenization': 'official_pretokenized_window_v1',
                    'tokenizer_sha256': source_lock()['assets']['tokenizer.json'],
                    'review_status': review, **provenance, **metadata})
    if not seen_ids:
        raise GateClosed('no raw documents supplied')
    # A common short tail (e.g. punctuation+EOS) can recur across independent
    # documents. Do not silently contaminate splits or discard a whole corpus:
    # retain one exact window deterministically, favoring open DEV over calibration
    # over train. Record every dropped window and token target; no semantic dedup claim.
    window_owners, dropped_windows = {}, {k: {'windows': 0, 'shifted_targets': 0} for k in splits}
    for split in ('dev', 'calibration', 'train'):
        kept = []
        for row in sorted(splits[split], key=lambda r: (r['group_id'], r['id'])):
            key = ' '.join(unicodedata.normalize('NFKC', row['text']).casefold().split())
            if key in window_owners:
                dropped_windows[split]['windows'] += 1
                dropped_windows[split]['shifted_targets'] += len(row['input_ids'])-1
                continue
            window_owners[key] = split
            kept.append(row)
        splits[split] = kept
    ensure_disjoint(*splits.values())
    artifacts, statistics = {}, {}
    for split, rows in splits.items():
        # Stable order independent of source-file order; group assignment happened first.
        rows.sort(key=lambda row: (row['group_id'], row['id']))
        name = f'{split}.jsonl'
        with (out/name).open('x', encoding='utf-8', newline='\n') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
        artifacts[name] = sha256_file(out/name)
        statistics[split] = {'rows': len(rows), 'groups': len({r['group_id'] for r in rows}),
                             'input_tokens': sum(len(r['input_ids']) for r in rows),
                             'supervised_shifted_targets': sum(len(r['input_ids'])-1 for r in rows)}
    if sha256_file(input_path) != original_hash or audit_source(source)['audit_digest'] != audit['audit_digest']:
        raise GateClosed('raw corpus or official tokenizer/source changed during preparation')
    return write_once(out/'DATA_MANIFEST.json', {'schema': 2, 'evidence_kind': 'data_preparation_only',
        'raw_sha256': original_hash, 'source_audit_digest': audit['audit_digest'],
        'tokenizer_sha256': source_lock()['assets']['tokenizer.json'], 'seed': seed, 'max_tokens': max_tokens,
        'group_policy': 'group SHA256 buckets before tokenization/windowing: 90% train, 5% calibration, 5% open DEV in expectation',
        'window_policy': 'one-token overlap, document-local EOS; exact token IDs authoritative; no cross-document packing',
        'documents': len(seen_ids), 'within_group_duplicates_removed': duplicate_documents,
        'exact_window_deduplication': {'priority': ['dev', 'calibration', 'train'], 'dropped': dropped_windows},
        'pending_review_documents': pending, 'sources': list(sources.values()), 'splits': statistics,
        'artifacts': artifacts, 'status': 'PREPARED_NOT_TRAINING_AUTHORIZED',
        'all_splits_nonempty': all(v['rows'] for v in statistics.values()),
        'licenses_registered_training': False, 'licenses_efficacy_claim': False,
        'limitations': ['License/review fields are declarations, not legal clearance or a quality audit.',
                       'Exact duplicate detection does not detect all near-duplicates or benchmark contamination.',
                       'No sealed TEST is read or generated; small inputs can yield empty splits.']})


def recovery_readiness(plan_path: Path) -> dict:
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    missing = []
    if not isinstance(plan, dict):
        raise GateClosed('recovery plan must be a JSON object')
    if plan.get('training_seeds') != [17, 29, 43]:
        missing.append('training_seeds must be [17, 29, 43]')
    if plan.get('arms') != ['B28', 'D40', 'C28', 'C40']:
        missing.append('all four registered arms and matched continuation controls')
    if plan.get('source_weights_sha256') != source_lock()['assets']['model.safetensors']:
        missing.append('exact official source hash')
    for key in ('recovery_tokens_per_arm', 'search_trials_per_arm', 'max_device_memory_bytes'):
        value = plan.get(key)
        if type(value) is not int or value <= 0:
            missing.append(f'positive integer {key}')
    for key in ('minimum_effect', 'noninferiority_margin', 'regression_band', 'max_decode_latency_ratio'):
        value = plan.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            missing.append(f'finite positive {key}')
    for key in ('recovery_data_manifest_sha256', 'dev_data_manifest_sha256',
                'regression_data_manifest_sha256', 'sealed_test_manifest_sha256'):
        value = plan.get(key)
        if not isinstance(value, str) or len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            missing.append(f'actual frozen {key}')
    for key in ('optimizer', 'schedule', 'primary_endpoint', 'reviewed_data_provenance',
                'recovered_ordinary_donor_plan', 'matched_continuation_plan', 'cost_accounting_plan'):
        if not isinstance(plan.get(key), str) or not plan[key].strip():
            missing.append(key)
    return {'schema': 2, 'evidence_kind': 'readiness_check_only', 'plan_sha256': sha256_file(plan_path),
            'status': 'BLOCKED', 'configuration_complete': not missing, 'missing_items': missing,
            'recovery_runner_implemented': False, 'licenses_registered_training': False,
            'licenses_efficacy_claim': False,
            'reason': 'G3-G5 remain closed even when these declarations are filled. '
                      'This lint check neither verifies scientific adequacy nor opens an optimizer campaign.'}
