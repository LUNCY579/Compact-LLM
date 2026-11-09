"""Bounded, create-only CC0 train/validation audit. Never invokes release scripts."""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time

ROOT = Path(r'C:\Users\clclc\BusinessProject\ChatBubble-ced-native-loader-qualification\rineya_long_context_cc0_v1_main')
TOKENIZER_ROOT = Path(r'C:\Users\clclc\BusinessProject\SupaExperiment\Rineya-Supa-A1.1')
SPLITS = ('train', 'validation')
SHARDS = {s: tuple(f'data/sft/{s}/shard-{i:03d}-of-{n:03d}.jsonl' for i in range(n)) for s, n in [('train', 22), ('validation', 5)]}
LINEAGES = {s: f'reports/main_lineage/{s}.jsonl' for s in SPLITS}
SOURCE_FILES = ('README.md', 'DATA_CARD.md', 'SPLIT_AND_CONTAMINATION_POLICY.md', 'reports/release_manifest.json',
                'reports/validation_report.json', 'reports/sha256sums.txt', 'scripts/corpus_core.py',
                'scripts/build_main.py', 'scripts/main_templates.py', 'scripts/validate_release.py', 'schemas/sft_record.schema.json')
ALLOWED = frozenset(SOURCE_FILES + tuple(LINEAGES.values()) + sum(SHARDS.values(), ()))
TOKENIZER_ASSETS = {
    'merges.txt': '8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5',
    'tokenizer.json': 'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
    'tokenizer_config.json': 'd5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101',
    'vocab.json': 'ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910',
}
CUE = 'certified archival shelf date'
STRATA = ('2k', '4k', '8k', '16k', '32k')


def approved_path(root: Path, relative: str) -> Path:
    if relative not in ALLOWED:
        raise ValueError('Input not explicitly allowlisted: ' + relative)
    path = root / relative
    if root.is_symlink() or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Symlink input refused')
    path.resolve().relative_to(root.resolve())
    return path


def dump_new(path: Path, value) -> None:
    with path.open('x', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=2, ensure_ascii=False)
        handle.write('\n')


def add_confusion(counts, predicted_unknown: bool, actual_unknown: bool):
    counts['TP' if predicted_unknown and actual_unknown else 'FP' if predicted_unknown else 'FN' if actual_unknown else 'TN'] += 1


def confusion_report(counts):
    c = {key: counts.get(key, 0) for key in ('TP', 'FP', 'FN', 'TN')}
    positive, negative = c['TP'] + c['FN'], c['TN'] + c['FP']
    c['rows'] = positive + negative
    c['balanced_accuracy'] = (c['TP'] / positive + c['TN'] / negative) / 2 if positive and negative else None
    c['accuracy'] = (c['TP'] + c['TN']) / c['rows'] if c['rows'] else None
    return c


def last_user(messages):
    return next(message['content'] for message in reversed(messages[:-1]) if message['role'] == 'user')


def registry_first(text, split):
    pattern = r'^R01 :: exact_name=(.*?) ; card=' if split == 'train' else r'^Audit member 01 is exactly “(.*?)” \('
    found = re.search(pattern, text, flags=re.MULTILINE)
    return found.group(1) if found else None


def keep_smallest(bucket, key, sample, n):
    bucket.append((key, sample))
    bucket.sort(key=lambda item: item[0])
    del bucket[n:]


def worker(out: Path):
    started = time.monotonic()
    import psutil
    process = psutil.Process()
    peak_rss = 0
    inventory = []
    expected_hashes = {}

    def guard():
        nonlocal peak_rss
        peak_rss = max(peak_rss, process.memory_info().rss)
        if time.monotonic() - started > 175:
            raise RuntimeError('Audit wall-time guard exceeded 175 seconds')
        if peak_rss > 1536 * 1024 * 1024:
            raise RuntimeError('Audit RSS guard exceeded 1536 MiB')

    def stream(relative, json_lines=False):
        path = approved_path(ROOT, relative)
        before = path.stat()
        h = hashlib.sha256()
        count = 0
        with path.open('rb') as handle:
            if json_lines:
                for raw in handle:
                    h.update(raw)
                    count += 1
                    if count % 256 == 0:
                        guard()
                    yield json.loads(raw)
            else:
                for raw in iter(lambda: handle.read(1024 * 1024), b''):
                    h.update(raw)
                    guard()
                    yield raw
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError('Input changed during audit: ' + relative)
        digest = h.hexdigest()
        if relative in expected_hashes and digest != expected_hashes[relative]:
            raise RuntimeError('Producer checksum mismatch: ' + relative)
        inventory.append({'path': relative, 'bytes': after.st_size, 'sha256': digest,
                          'rows': count if json_lines else None,
                          'producer_checksum_match': digest == expected_hashes[relative] if relative in expected_hashes else None})

    checksum = b''.join(stream('reports/sha256sums.txt')).decode('utf-8')
    for line in checksum.splitlines():
        digest, rel = line.split('  ', 1)
        if rel in ALLOWED:
            expected_hashes[rel] = digest
    for relative in SOURCE_FILES:
        if relative != 'reports/sha256sums.txt':
            for _ in stream(relative):
                pass

    lineage = {}
    families = {s: set() for s in SPLITS}
    origins = {s: set() for s in SPLITS}
    lineage_counts = {}
    for split in SPLITS:
        count = 0
        for item in stream(LINEAGES[split], True):
            if item['split'] != split or item['stratum'] not in STRATA or item['sft_id'] in lineage:
                raise RuntimeError('Invalid or duplicate lineage identity')
            # Retain no task_spec, stored answer, source fact, or hidden solver result.
            lineage[item['sft_id']] = (split, item['stratum'], item['source_family_id'], item['task_key'])
            families[split].add(item['source_family_id'])
            origins[split].update(item['origin_group_ids'])
            count += 1
        lineage_counts[split] = count

    seen = {s: set() for s in SPLITS}
    confusion = {s: collections.Counter() for s in SPLITS}
    by_stratum = {s: {k: collections.Counter() for k in STRATA} for s in SPLITS}
    tasks = {s: collections.Counter() for s in SPLITS}
    multiturn = {s: collections.Counter() for s in SPLITS}
    exceptions = {s: collections.Counter() for s in SPLITS}
    constraints = {s: collections.Counter() for s in SPLITS}
    shortest = {}
    stratum_sample = {s: [] for s in STRATA}
    errors = []
    row_count = 0
    for split in SPLITS:
        for relative in SHARDS[split]:
            shortest[relative] = []
            for row in stream(relative, True):
                sid = row['id']
                info = lineage.get(sid)
                if info is None or info[0] != split or row['metadata']['split'] != split or sid in seen[split]:
                    raise RuntimeError('Invalid or duplicate SFT identity: ' + sid)
                seen[split].add(sid)
                stratum, family, task = info[1:]
                messages = row['messages']
                if messages[-1]['role'] != 'assistant':
                    raise RuntimeError('Missing final assistant target')
                actual_multi = len(messages) > 3
                if actual_multi != row['metadata']['multi_turn']:
                    raise RuntimeError('Multi-turn declaration mismatch')
                multiturn[split][str(actual_multi).lower()] += 1
                tasks[split][task] += 1
                target = messages[-1]['content']
                predicted = CUE in last_user(messages).lower()
                actual = target == 'UNKNOWN'
                add_confusion(confusion[split], predicted, actual)
                add_confusion(by_stratum[split][stratum], predicted, actual)
                if predicted != actual:
                    errors.append({'id': sid, 'question_cue': predicted, 'actual_unknown': actual})
                if task == 'exception_policy':
                    exceptions[split][target] += 1
                if task == 'constraint_intersection':
                    first = registry_first(messages[1]['content'], split)
                    constraints[split]['rows'] += 1
                    constraints[split]['parsed'] += first is not None
                    constraints[split]['first_registry_name_correct'] += first == target
                sample = {'id': sid, 'split': split, 'shard': relative, 'stratum': stratum,
                          'prompt_characters': sum(len(m['content']) for m in messages[:-1]), 'messages': messages}
                keep_smallest(shortest[relative], (sample['prompt_characters'], sid), sample, 4)
                rank = hashlib.sha256(('cc0-native-sample-v1:' + sid).encode()).hexdigest()
                keep_smallest(stratum_sample[stratum], (rank, sid), sample, 2)
                row_count += 1
            print('Audited ' + relative, flush=True)
    if any(len(seen[s]) != lineage_counts[s] for s in SPLITS) or row_count != 47016:
        raise RuntimeError('Allowed population/lineage count mismatch')
    isolation = {'id_overlap': len(seen['train'] & seen['validation']),
                 'source_family_overlap': len(families['train'] & families['validation']),
                 'semantic_origin_overlap': len(origins['train'] & origins['validation']),
                 'family_counts': {s: len(families[s]) for s in SPLITS},
                 'origin_counts': {s: len(origins[s]) for s in SPLITS}}
    selected = {}
    for bucket in shortest.values():
        for _, sample in bucket:
            selected.setdefault(sample['id'], {**sample, 'selection_reasons': []})['selection_reasons'].append('four_shortest_characters_in_shard')
    for bucket in stratum_sample.values():
        for _, sample in bucket:
            selected.setdefault(sample['id'], {**sample, 'selection_reasons': []})['selection_reasons'].append('two_smallest_seeded_id_hashes_in_declared_stratum')
    print(f'All {row_count} permitted rows audited; tokenizing {len(selected)} frozen samples.', flush=True)

    # Tokenizer-only runtime: no Torch/model import or model directory scan.
    os.environ['USE_TORCH'] = '0'
    os.environ['USE_TF'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    if importlib.metadata.version('transformers') != '4.57.1' or importlib.metadata.version('tokenizers') != '0.22.2':
        raise RuntimeError('Exact tokenizer runtime mismatch')
    from transformers import Qwen2TokenizerFast
    token_rows = []
    assets = []
    with tempfile.TemporaryDirectory(prefix='tokenizer-only-', dir=out) as temporary:
        for basename, expected in TOKENIZER_ASSETS.items():
            path = TOKENIZER_ROOT / basename
            if any(p.is_symlink() for p in (path, *path.parents)):
                raise RuntimeError('Tokenizer symlink refused')
            raw = path.read_bytes()
            digest = hashlib.sha256(raw).hexdigest()
            if digest != expected:
                raise RuntimeError('Tokenizer asset changed: ' + basename)
            assets.append({'path': str(path), 'bytes': len(raw), 'sha256': digest})
            (Path(temporary) / basename).write_bytes(raw)
        tokenizer = Qwen2TokenizerFast.from_pretrained(temporary, local_files_only=True, trust_remote_code=False)
        for sid, sample in sorted(selected.items()):
            guard()
            messages = sample['messages']
            prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True, enable_thinking=False)
            full = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False, enable_thinking=False)
            prefix_match = full[:len(prompt)] == prompt
            if not prefix_match or full[len(prompt):].count(tokenizer.eos_token_id) != 1:
                raise RuntimeError('Chat template prefix/EOS mismatch')
            token_rows.append({k: v for k, v in sample.items() if k != 'messages'} | {
                'first_user_tokens': len(tokenizer.encode(messages[1]['content'], add_special_tokens=False)),
                'rendered_prompt_tokens': len(prompt), 'rendered_total_tokens': len(full),
                'completion_tokens': len(full) - len(prompt), 'prompt_le_1920': len(prompt) <= 1920,
                'total_le_2048': len(full) <= 2048, 'prompt_prefix_match': prefix_match, 'completion_eos_count': 1})
    if 'torch' in sys.modules:
        raise RuntimeError('Unexpected Torch import in tokenizer-only worker')
    guard()
    report = {
        'schema': 'mrs_phase2.cc0_audit/1', 'status': 'P0_STOP_FATAL_SHORTCUT_VERIFIED',
        'root': str(ROOT), 'audit_seconds': time.monotonic() - started, 'peak_rss_mib': peak_rss / 1048576,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'python': sys.version, 'runtime': {p: importlib.metadata.version(p) for p in ('transformers', 'tokenizers', 'psutil')},
        'input_inventory': inventory, 'row_counts': {s: len(seen[s]) for s in SPLITS},
        'question_only_rule': 'Predict UNKNOWN iff final user message contains certified archival shelf date (case insensitive).',
        'question_only_confusion': {s: confusion_report(confusion[s]) for s in SPLITS},
        'question_only_by_stratum': {s: {k: confusion_report(v) for k, v in by_stratum[s].items()} for s in SPLITS},
        'question_cue_errors': errors, 'task_counts': tasks, 'multi_turn_counts': multiturn,
        'exception_policy_answer_counts': exceptions, 'first_registry_constraint_rule': constraints,
        'train_validation_isolation': isolation,
        'lineage_read_reason': 'Declared strata and semantic-origin/family IDs are absent from SFT; only train and validation lineage read. Hidden task_spec values discarded, never supplied to tokenizer or any model.',
        'tokenizer_assets': assets,
        'native_sample': {'selection': 'Four shortest-character prompts per shard (tie ID), plus two smallest SHA256(cc0-native-sample-v1:ID) per declared stratum; union deduplicated.',
                          'rows': len(token_rows), 'is_census': False, 'not_representative': True,
                          'scope': 'Native tokenizer counts only. No full token census because fatal shortcut blocks acceptance.',
                          'prompt_min': min(r['rendered_prompt_tokens'] for r in token_rows),
                          'prompt_max': max(r['rendered_prompt_tokens'] for r in token_rows),
                          'prompt_over_1920': sum(not r['prompt_le_1920'] for r in token_rows),
                          'total_over_2048': sum(not r['total_le_2048'] for r in token_rows),
                          'enable_thinking': False, 'trust_remote_code': False, 'truncations': 0},
        'limits': {'supervisor_seconds': 180, 'worker_seconds': 175, 'worker_rss_mib': 1536},
        'protected_access': {'test_payload_reads_or_hashes': 0, 'sealed_payload_reads_or_hashes': 0,
                             'archives_read_or_hashed': 0, 'raw_source_payloads_read': 0},
        'claims': {'corpus_accepted': False, 'model_calls': 0, 'training_steps': 0, 'torch_imported': False,
                   'independent_text_answer_verification': False, 'external_quarantine_exclusion': False,
                   'native_model_execution': False, 'confirmatory': False, 'release': False}}
    dump_new(out / 'NATIVE_TOKEN_SAMPLE.json', token_rows)
    dump_new(out / 'CC0_AUDIT.json', report)
    markdown = (
        '# CC0 P0: stop, fatal shortcut verified\n\n'
        f'Streamed and hashed all {row_count:,} permitted SFT rows: 44,016 train and 3,000 validation. '
        'The exact question-only cue identifies every UNKNOWN row in both splits. '
        'Exception-policy answers and first-registry constraint answers are also recorded in the JSON receipt. '
        'This dataset is not accepted for Phase II training.\n\n'
        f'Native tokenizer evidence covers {len(token_rows)} deterministic samples, not a census or representative estimate. '
        f"{report['native_sample']['prompt_over_1920']} sampled prompts exceed 1,920 tokens; "
        f"{report['native_sample']['total_over_2048']} sampled full chats exceed 2,048. "
        'No truncation, model loading or training occurred.\n\n'
        'Only named train/validation shards and their two lineage files were read. No test/sealed payload, archive, '
        'raw anchor snapshot or release validator was opened/run. Source and input hashes, exact confusion matrices, '
        'task/multi-turn counts, isolation counts, resource measurements and token sample IDs are preserved alongside this report.\n')
    with (out / 'CC0_AUDIT.md').open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(markdown)
    print(json.dumps({'status': report['status'], 'question_only': report['question_only_confusion'],
                      'exceptions': exceptions, 'constraints': constraints, 'native_sample': report['native_sample'],
                      'isolation': isolation, 'peak_rss_mib': report['peak_rss_mib']}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.output.resolve()
    if args.worker:
        worker(out)
        return
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    command = [sys.executable, '-B', str(Path(__file__).resolve()), '--worker', '--output', str(out)]
    status = 'WORKER_FAILED'
    with (out / 'worker_stdout.log').open('x', encoding='utf-8') as stdout:
        process = subprocess.Popen(command, stdout=stdout, stderr=subprocess.STDOUT)
        try:
            return_code = process.wait(timeout=180)
            status = 'COMPLETED' if return_code == 0 else 'WORKER_FAILED'
        except subprocess.TimeoutExpired:
            process.kill()
            return_code = process.wait(timeout=10)
            status = 'TIMEOUT_TERMINATED'
    dump_new(out / 'SUPERVISOR.json', {'status': status, 'return_code': return_code,
              'elapsed_seconds': time.monotonic() - started, 'timeout_seconds': 180,
              'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'command': command,
              'model_calls': 0, 'training_steps': 0})
    print(status)
    if return_code:
        raise SystemExit(return_code)


if __name__ == '__main__':
    main()
