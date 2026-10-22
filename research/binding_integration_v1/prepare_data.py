"""Create-only BINDING-INTEGRATION-001 execution layout from admitted TRAIN bytes.

No model, tokenizer, source weights, new corpus, PLAN or GRANT is read/created.
The original 193 native groups, native denominators and native token LR clock
are preserved. Binding rows are appended within each update and share its native
denominator; their coefficient budget is deliberately not held at the old mix.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys

BASE = Path('C:/Users/clclc/BusinessProject')
RECORDS = BASE / 'rineya-dataset-retention-records'
ROOT = RECORDS / 'binding-integration001-20260920'
OUT = ROOT / 'prepared'
CONTROL = RECORDS / 'highdose-lora001-resume-20260920'
MIXTURE = RECORDS / 'binding-mixture001-20260920'
BINDING = RECORDS / 'binding-signal-prep-20260920/candidate-v2'
PILOT = BASE / 'ChatBubble-binding-integration-pilot-20260920'
NATIVE = RECORDS / 'doseint001-20260919/prepared-dual899/microfit-train.jsonl'
BINDING_TRAIN = BINDING / 'compiled-v1/TRAIN_COMPILED.jsonl'
LEDGER = RECORDS / 'binding-diagnostic001-20260920/NATIVE_COEFFICIENT_AUDIT.json'
CHECKPOINT2 = CONTROL / 'journal/runs/train-001/checkpoint-00002'
EXPERIMENT = 'BINDING-INTEGRATION-001'
DATA_FILES = ('microfit-train.jsonl', 'INTEGRATION_GROUPS.json', 'SOURCE_MEMBERSHIP.json', 'DATA_BUILD.json')
PINS = {
    NATIVE: 'df64db3cfbfb97fe9f9a690862a3f18d5967f6e031007805ed7e3d89bd3170d3',
    BINDING_TRAIN: 'd2f0c35ce9a18f0c81e627d84a648dd1f9a26ecf257958c1f0f1664a3cba4574',
    BINDING / 'DATA_ADMISSION.json': '854ff8124cdc2e9411a0cb867633b96ec3a08b1bfceb28ee0d8be90706bf9935',
    MIXTURE / 'prepared/DATA_BUILD.json': '41987b43706c970521f8af03eb99c37d0473096415df3ce32faa080ad41b5964',
    MIXTURE / 'prepared/SOURCE_MEMBERSHIP.json': '6f1a1d3bd3596bcc21b6ee19950352d8cf386054d294dee5b9ffc9d555a41cb4',
    MIXTURE / 'prepared/microfit-train.jsonl': 'ceeb85e0acea810aa4e4512f9fd46057f41a2af892988e087267dbac6f1e7322',
    MIXTURE / 'CLOSEOUT.json': '23e6732a25205a6d88813e23ca706ec5418e437cf5e2d71a873bc82c6372001e',
    MIXTURE / 'RESULTS.json': '3ea8b2278a4efe57848df1c0fb5a6af94c6e9e156e599410a9b13015c1c7afde',
    CONTROL / 'PLAN.json': 'c0af45e4e28bc425839f1ebf0b544e5326d2b43feafb78c3cb4dee218a317b20',
    CONTROL / 'prepared/DATA_CONTRACT.json': 'd959f58f8a64d006f8a4da6a00ca79562a5fb1076585ed39c6018173f6df5abb',
    CONTROL / 'journal/runs/train-001/steps.jsonl': '95d0d3bfedb9f2ff1d8d7e0f469bddfe89ec9b0ac83cb946b1b26732be707352',
    CHECKPOINT2 / 'manifest.json': 'afc09301260af14ea033f42e4ccb006982f8a36f7b31db4807d521e7399c5376',
    LEDGER: 'a467e836f0bc6ca87f6b485a802ec82ab4b0c4fbf9bc58da2dcab1b55b6f98ce',
    PILOT / 'experiments/model_quality_gpu_v1/data.py': '148204b01b9e17e9979f3ef1a92b3effae6c72186cb9d07ce72e0afb611f65ab',
    PILOT / 'experiments/model_quality_gpu_v1/config.py': 'de8c268886978050634e0e8f18691439c024c5d8c31a6132cdc4913b3a367834',
}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def regular(path):
    path = Path(path).absolute()
    require('..' not in path.parts and not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe input path')
    require(path.is_file() and path.suffix not in ('.safetensors', '.pt', '.bin'), 'missing input or weight payload read prohibited')
    require('Rineya_Long_Context_Demo_v1' not in path.parts, 'new raw LC reads prohibited')
    return path


def sha(path):
    with regular(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'duplicate JSON field')
            result[key] = value
        return result
    def invalid(value):
        raise ValueError('nonfinite JSON constant: ' + value)
    return json.loads(regular(path).read_text(encoding='utf-8-sig'), object_pairs_hook=pairs, parse_constant=invalid)


def write_new(path, value):
    path = Path(path).absolute()
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'unsafe output path')
    with path.open('xb') as stream:
        stream.write(canonical(value) + b'\n'); stream.flush(); os.fsync(stream.fileno())


def no_model_runtime():
    require(not any(name in sys.modules for name in ('torch', 'transformers', 'tokenizers', 'tensorflow', 'jax', 'safetensors')), 'data preparation imported model/tokenizer runtime')


def compiled_entries(path, expected_sha, label):
    raw = regular(path).read_bytes()
    require(hashlib.sha256(raw).hexdigest() == expected_sha, 'compiled source bytes differ')
    entries = []
    for index, line in enumerate(raw.splitlines(keepends=True)):
        require(line.strip() and line.endswith(b'\n'), 'compiled source rows must be newline terminated')
        row = json.loads(line)
        ids, labels, prompt = row['ids'], row['labels'], row['prompt_ids']
        end = row['answer_end']
        require(row['source_split'] == 'train' and row['mode'] == 'direct' and row['truncated'] is False, 'only unchanged direct TRAIN views allowed')
        require(isinstance(row['id'], str) and row['id'] and isinstance(row['parent_id'], str) and row['parent_id'], 'source identities missing')
        require(len(ids) == len(labels) == row['processed_tokens'] and 0 < len(ids) <= 2048 and ids[:len(prompt)] == prompt, 'compiled source sequence geometry differs')
        require(all(type(t) is int and 0 <= t < 151936 for t in ids), 'source token IDs invalid')
        require(labels[0] == -100 and row['answer_start'] == len(prompt) and len(prompt) <= end < len(ids) and ids[end] == 151645,
                'response-plus-EOS boundaries differ')
        require(labels == [token if len(prompt) <= i <= end else -100 for i, token in enumerate(ids)] and
                sum(t != -100 for t in labels[1:]) == row['supervised_tokens'], 'shifted supervised mask differs')
        entries.append({'row': row, 'raw': line, 'source': label, 'source_index': index,
                        'source_row_sha256_including_newline': hashlib.sha256(line).hexdigest()})
    require(len({e['row']['id'] for e in entries}) == len(entries), 'duplicate source view IDs')
    return entries


def original_binding_order(native, binding, old_lines, membership):
    require(len(old_lines) == len(membership) == len(native) + len(binding), 'old mixture inventory differs')
    sources = {'native_dual899': native, 'binding_v2': binding}
    seen = {key: [] for key in sources}
    previous_indices = {}
    for index, (line, record) in enumerate(zip(old_lines, membership)):
        label, source_index = record['source'], record['source_index']
        require(label in sources and type(source_index) is int and 0 <= source_index < len(sources[label]), 'old mixture source mapping invalid')
        entry = sources[label][source_index]
        expected = {'mixture_index': index, 'id': entry['row']['id'], 'parent_id': entry['row']['parent_id'],
                    'source': label, 'source_index': source_index,
                    'source_row_sha256_including_newline': entry['source_row_sha256_including_newline']}
        require(record == expected and line == entry['raw'], 'old mixture membership or source line bytes differ')
        seen[label].append(source_index)
        require((label, source_index) not in previous_indices, 'old source row repeated')
        previous_indices[label, source_index] = index
    require(seen['native_dual899'] == list(range(len(native))) and sorted(seen['binding_v2']) == list(range(len(binding))), 'old mixture drops/reorders source rows')
    return seen['binding_v2'], previous_indices


def layout_indices(native_groups, binding_order):
    """Pure floor partition; indices in returned groups address new execution rows."""
    native_flat = [i for group in native_groups for i in group]
    require(native_groups and all(native_groups) and native_flat == list(range(len(native_flat))), 'original native groups must partition ordered rows')
    require(binding_order and sorted(binding_order) == list(range(len(binding_order))), 'binding subsequence must be an exact permutation')
    execution, ngroups, bgroups, combined = [], [], [], []
    for group_index, native in enumerate(native_groups):
        start, stop = group_index * len(binding_order) // len(native_groups), (group_index + 1) * len(binding_order) // len(native_groups)
        n = list(range(len(execution), len(execution) + len(native)))
        execution.extend(('native_dual899', source_index) for source_index in native)
        b = list(range(len(execution), len(execution) + stop - start))
        execution.extend(('binding_v2', source_index) for source_index in binding_order[start:stop])
        ngroups.append(n); bgroups.append(b); combined.append(n + b)
    require([index for group in combined for index in group] == list(range(len(execution))), 'combined groups are not contiguous')
    return execution, ngroups, bgroups, combined


def assemble(native, binding, native_groups, binding_order, previous_indices, applied_rates):
    execution, ngroups, bgroups, combined = layout_indices(native_groups, binding_order)
    sources = {'native_dual899': native, 'binding_v2': binding}
    ordered = [sources[label][source_index] for label, source_index in execution]
    rows = [entry['row'] for entry in ordered]
    group_for_index = {i: g for g, group in enumerate(combined) for i in group}
    membership = [{'execution_index': i, 'id': e['row']['id'], 'parent_id': e['row']['parent_id'],
                   'source': e['source'], 'source_index': e['source_index'],
                   'previous_mixture_index': previous_indices[e['source'], e['source_index']],
                   'optimizer_group_index': group_for_index[i],
                   'source_row_sha256_including_newline': e['source_row_sha256_including_newline']} for i, e in enumerate(ordered)]
    def tokens(groups):
        return [sum(len(rows[i]['ids']) for i in group) for group in groups]
    def supervised(groups):
        return [sum(sum(t != -100 for t in rows[i]['labels'][1:]) for i in group) for group in groups]
    ntokens, btokens = tokens(ngroups), tokens(bgroups)
    nd, bd = supervised(ngroups), supervised(bgroups)
    require(all(d > 0 for d in nd), 'native update lacks supervised targets')
    cumulative = []; current = 0
    for value in ntokens:
        current += value; cumulative.append(current)
    require(len(applied_rates) == len(combined), 'LR schedule length differs')
    record = {'schema': 1, 'record': 'BINDING_INTEGRATION_GROUPS', 'experiment': EXPERIMENT,
              'native_update_groups': ngroups, 'binding_update_groups': bgroups, 'combined_update_groups': combined,
              'original_native_update_groups': native_groups, 'binding_source_order': binding_order,
              'native_denominators': nd, 'binding_denominators': bd,
              'native_processed_per_update': ntokens, 'binding_processed_per_update': btokens,
              'combined_processed_per_update': [a + b for a, b in zip(ntokens, btokens)],
              'native_processed_cumulative': cumulative, 'applied_native_learning_rates': applied_rates,
              'native_processed_tokens': sum(ntokens), 'binding_processed_tokens': sum(btokens),
              'processed_tokens': sum(ntokens) + sum(btokens), 'native_supervised_tokens': sum(nd),
              'binding_supervised_tokens': sum(bd), 'supervised_tokens': sum(nd) + sum(bd),
              'update_groups_count': len(combined), 'row_order_per_exposure': [r['id'] for r in rows],
              'row_order_sha256': digest([r['id'] for r in rows]),
              'max_sequence_tokens': max(len(r['ids']) for r in rows), 'sequence_token_limit': 2048,
              'native_group_token_limit': 2048, 'maximum_combined_processed_tokens_per_update': max(a + b for a, b in zip(ntokens, btokens)),
              'combined_token_count_is_accumulation_not_sequence_length': True,
              'binding_partition_rule': 'binding_order[floor(g*B/G):floor((g+1)*B/G)] appended after native group g',
              'loss_definition': '(native summed shifted token loss + binding summed shifted token loss) / ORIGINAL native supervised denominator',
              'learning_rate_clock': 'Only cumulative original native processed tokens, total380921; binding tokens do not advance the schedule.',
              'effective_binding_mean_coefficients': [b / n for b, n in zip(bd, nd)],
              'training_permission': False, 'model_execution_permission': False}
    return ordered, membership, record


def prepare():
    no_model_runtime()
    require(ROOT.is_dir() and not OUT.exists() and not OUT.is_symlink(), 'prepared directory must be fresh under the new record root')
    bound = {}
    def bind(path, expected):
        path = Path(path).absolute()
        require(sha(path) == expected, 'frozen dependency changed: ' + str(path))
        bound[str(path)] = expected
    for path, pin in PINS.items():
        bind(path, pin)
    old_build = read(MIXTURE / 'prepared/DATA_BUILD.json')
    require(old_build['status'] == 'PASS_DATA_PREPARATION_ONLY' and old_build['rows'] == 3465, 'old admitted data build differs')
    for path, pin in old_build['input_files_sha256'].items():
        bind(path, pin)
    for name, pin in old_build['output_files_sha256'].items():
        bind(MIXTURE / 'prepared' / name, pin)
    admission = read(BINDING / 'DATA_ADMISSION.json')
    require(admission['status'] == 'ADMITTED_FOR_NAMED_EXPLORATORY_SCOPE' and admission['training_permission'] is False and admission['external_exclusions_complete'] is False, 'old binding admission scope differs')
    for name, pin in admission['evidence_sha256'].items():
        path = (BINDING / name).resolve()
        require(path.is_relative_to(BINDING.resolve()), 'admission evidence escapes its root')
        bind(path, pin)
    for name, pin in admission['compiled_files_sha256'].items():
        bind(BINDING / 'compiled-v1' / name, pin)
    native_contract = read(CONTROL / 'prepared/DATA_CONTRACT.json')
    for name, pin in native_contract['files'].items():
        bind(CONTROL / 'prepared' / name, pin)
    require(read(MIXTURE / 'CLOSEOUT.json')['status'] == 'COMPLETE_AUDITED_EXPERIMENT_FAILED_PRIMARY_CRITERIA', 'previous failed result must remain failed')
    ledger = read(LEDGER)
    require(ledger['status'] == 'COMPLETE_DATA_ONLY_COEFFICIENT_AUDIT' and ledger['original_updates'] == 193, 'coefficient design ledger differs')
    for path, pin in ledger['source_files_sha256'].items():
        bind(path, pin)
    native = compiled_entries(NATIVE, PINS[NATIVE], 'native_dual899')
    binding = compiled_entries(BINDING_TRAIN, PINS[BINDING_TRAIN], 'binding_v2')
    require(len(native) == 2697 and len(binding) == 768, 'source TRAIN view inventory differs')
    old_lines = (MIXTURE / 'prepared/microfit-train.jsonl').read_bytes().splitlines(keepends=True)
    binding_order, previous = original_binding_order(native, binding, old_lines, read(MIXTURE / 'prepared/SOURCE_MEMBERSHIP.json'))
    native_plan = read(CONTROL / 'PLAN.json')
    native_steps = [json.loads(line) for line in (CONTROL / 'journal/runs/train-001/steps.jsonl').read_text(encoding='utf-8').splitlines()]
    sys.path.insert(0, str(PILOT))
    from experiments.model_quality_gpu_v1.data import group_updates
    from experiments.model_quality_gpu_v1.config import Design, rate_at
    require(Path(group_updates.__code__.co_filename).resolve() == (PILOT / 'experiments/model_quality_gpu_v1/data.py').resolve() and
            Path(rate_at.__code__.co_filename).resolve() == (PILOT / 'experiments/model_quality_gpu_v1/config.py').resolve(), 'native helpers imported from another checkout')
    native_rows = [e['row'] for e in native]
    native_groups = group_updates(native_rows, 2048)
    require(native_groups == native_plan['update_groups'] and len(native_groups) == len(native_steps) == 193, 'original native partition differs')
    require([r['id'] for r in native_rows] == native_plan['row_order_per_exposure'], 'native original order differs')
    design = Design(processed_budget=380921)
    processed = supervised = 0; rates = []
    for count, (group, step) in enumerate(zip(native_groups, native_steps), 1):
        processed += sum(len(native_rows[i]['ids']) for i in group)
        supervised += sum(sum(t != -100 for t in native_rows[i]['labels'][1:]) for i in group)
        rate = rate_at(1e-4, processed, 380921, design)
        require(step['updates'] == step['next_group'] == count and step['processed_tokens'] == processed and step['supervised_tokens'] == supervised and
                step['row_ids'] == [native_rows[i]['id'] for i in group] and step['lr'] == rate, 'native original steps/LR do not reproduce')
        rates.append(rate)
    require(processed == 380921 and supervised == 37090, 'native token clock differs')
    ordered, membership, groups = assemble(native, binding, native_groups, binding_order, previous, rates)
    rows = [e['row'] for e in ordered]
    require(len(rows) == len({r['id'] for r in rows}) == 3465 and len({r['parent_id'] for r in rows}) == 2761 and
            Counter(Counter(r['parent_id'] for r in rows).values()) == {1: 2697, 12: 64}, 'actual view/parent inventory differs')
    require(groups['processed_tokens'] == 577417 and groups['supervised_tokens'] == 39502 and groups['binding_supervised_tokens'] == 2412 and
            groups['max_sequence_tokens'] == 277 and groups['maximum_combined_processed_tokens_per_update'] == 3114 and len(rows[0]['ids']) == 116,
            'integration token geometry differs')
    require(Counter(map(len, groups['binding_update_groups'])) == {3: 4, 4: 189}, 'floor partition did not assign all768 once')
    for index, proposed in enumerate(ledger['proposed_integration']['groups']):
        ng, bg = groups['native_update_groups'][index], groups['binding_update_groups'][index]
        require(proposed['group'] == index + 1 and proposed['native_row_indices'] == native_groups[index] and
                proposed['native_ids'] == [rows[i]['id'] for i in ng] and proposed['binding_ids'] == [rows[i]['id'] for i in bg] and
                proposed['binding_mixture_row_indices'] == [membership[i]['previous_mixture_index'] for i in bg] and
                proposed['native_supervised_denominator'] == groups['native_denominators'][index] and
                proposed['native_processed_tokens'] == groups['native_processed_per_update'][index] and
                proposed['binding_processed_tokens'] == groups['binding_processed_per_update'][index] and
                proposed['binding_supervised_tokens'] == groups['binding_denominators'][index] and proposed['applied_native_lr'] == rates[index],
                'prepared layout differs from reviewed coefficient ledger')
    checkpoint = read(CHECKPOINT2 / 'manifest.json')
    require(checkpoint['sha256'] == digest({k: v for k, v in checkpoint.items() if k != 'sha256'}) and checkpoint['status'] == 'COMPLETE', 'zero-addend reference manifest seal differs')
    require(checkpoint['progress'] == {'updates': 2, 'next_group': 2, 'processed_tokens': 4032, 'supervised_tokens': 388} and
            checkpoint['binding']['plan_sha256'] == digest(native_plan) and len(checkpoint['entries']) == 360, 'zero-addend reference provenance differs')
    zero_reference = {'checkpoint_path': str(CHECKPOINT2), 'manifest_sha256': PINS[CHECKPOINT2 / 'manifest.json'],
                      'tensor_files_sha256': {entry['file']: entry['sha256'] for entry in checkpoint['entries']}}
    new_binding_budget = sum(lr * d / n for lr, d, n in zip(rates, groups['binding_denominators'], groups['native_denominators']))
    old_binding_budget = ledger['coefficient_budgets']['mixture']['sum_applied_lr_times_group_loss_coefficient_fraction']['binding_v2']
    scope = {'status': 'PASS_REUSE_OF_EXACT_PREVIOUSLY_ADMITTED_TRAIN_BYTES_ONLY', 'new_experiment': EXPERIMENT,
             'new_corpus': False, 'dev_rows_used_for_training': False, 'retention_or_unit_replay_added': False,
             'binding_admission_path': str(BINDING / 'DATA_ADMISSION.json'), 'binding_admission_sha256': PINS[BINDING / 'DATA_ADMISSION.json'],
             'external_exclusions_complete': False, 'legacy_LC_stopword_fingerprints': admission['legacy_LC_stopword_fingerprints'],
             'reserved_bank_coverage': admission['reserved_bank_coverage'], 'unknown_overlap_counts': None,
             'no_old_grant_reused': True, 'required_before_model_work': 'Fresh registration/PLAN/GRANT, source/runtime qualification and exact zero-addend recovery; data preparation grants no execution permission.'}
    build = {'schema': 1, 'record': 'BINDING_INTEGRATION_DATA_BUILD', 'experiment': EXPERIMENT, 'status': 'PASS_DATA_PREPARATION_ONLY',
             'rows': 3465, 'presentations': 3465, 'parent_worlds': 2761, 'unique_worlds': 2761, 'exposures_per_view': 1,
             'processed_tokens': 577417, 'supervised_tokens': 39502, 'native_processed_tokens': 380921, 'native_supervised_tokens': 37090,
             'binding_processed_tokens': 196496, 'binding_supervised_tokens': 2412, 'update_groups_count': 193,
             'first_row_tokens': 116, 'max_sequence_tokens': 277, 'sequence_token_limit': 2048,
             'maximum_combined_processed_tokens_per_update': 3114, 'combined_token_count_is_accumulation_not_sequence_length': True,
             'source_geometry': old_build['source_geometry'], 'native_relative_order_preserved': True, 'prior_binding_subsequence_preserved': True,
             'source_rows_copied_byte_identically': True, 'source_rows_repeated_or_dropped': False, 'labels_modified': False,
             'rows_copied_byte_identically': True, 'binding_data_admission_path': str(BINDING / 'DATA_ADMISSION.json'),
             'binding_train_dev_separation': old_build['binding_train_dev_separation'],
             'native_grouping_denominators_and_lr_clock_preserved': True,
             'row_order_sha256': groups['row_order_sha256'], 'input_files_sha256': bound, 'admission_scope': scope,
             'acquisition_parent_exclusions': old_build['acquisition_parent_exclusions'],
             'exclusion_verification_scope': 'Existing scoped admission/exclusion corpus and evidence identities verified by hashes; no new raw LC/ANY_READ reads or universal coverage claim.',
             'original_native_plan_path': str(CONTROL / 'PLAN.json'), 'original_native_plan_file_sha256': PINS[CONTROL / 'PLAN.json'],
             'original_native_steps_path': str(CONTROL / 'journal/runs/train-001/steps.jsonl'), 'original_native_steps_sha256': PINS[CONTROL / 'journal/runs/train-001/steps.jsonl'],
             'reviewed_coefficient_ledger_path': str(LEDGER), 'reviewed_coefficient_ledger_sha256': PINS[LEDGER],
             'zero_addend_reference': zero_reference,
             'zero_addend_reference_manifest_only_not_tensor_payloads_rehashed': True,
             'zero_addend_first2_geometry': {'native_rows': 28, 'native_processed_tokens': 4032, 'native_supervised_tokens': 388,
                                            'row_counts_per_group': [14, 14], 'processed_tokens_per_group': [2036, 1996], 'supervised_tokens_per_group': [213, 175]},
             'coefficient_accounting': {'native_applied_lr_sum': sum(rates), 'binding_applied_lr_times_native_normalized_mass': new_binding_budget,
                 'previous_mixture_binding_applied_lr_mass': old_binding_budget, 'binding_coefficient_budget_ratio_to_previous_mixture': new_binding_budget / old_binding_budget,
                 'binding_exposure_count_unchanged': True, 'binding_coefficient_budget_held_constant': False,
                 'interpretation': 'Same768 binding views once; approximately0.677 of previous mixture binding scalar coefficient budget. Not constant binding influence: clipping, Adam and changed states couple both sources.'},
             'builder_sha256': sha(Path(__file__)), 'qualification_tokens_included': False, 'model_calls': 0, 'source_weights_read': False,
             'tokenizer_used': False, 'raw_LC_or_ANY_READ_sources_opened': False, 'human_data_review': 'NOT_PERFORMED',
             'training_permission': False, 'model_execution_permission': False, 'quality_claim': False, 'release_permitted': False}
    require(all(sha(path) == pin for path, pin in bound.items()), 'source/evidence changed during preparation')
    no_model_runtime()
    OUT.mkdir(exist_ok=False)
    try:
        with (OUT / 'microfit-train.jsonl').open('xb') as stream:
            for entry in ordered:
                stream.write(entry['raw'])
            stream.flush(); os.fsync(stream.fileno())
        require((OUT / 'microfit-train.jsonl').read_bytes().splitlines(keepends=True) == [e['raw'] for e in ordered], 'written row bytes differ')
        write_new(OUT / 'INTEGRATION_GROUPS.json', groups)
        write_new(OUT / 'SOURCE_MEMBERSHIP.json', membership)
        build['output_files_sha256'] = {name: sha(OUT / name) for name in DATA_FILES if name != 'DATA_BUILD.json'}
        write_new(OUT / 'DATA_BUILD.json', build)
        require(all(sha(path) == pin for path, pin in bound.items()), 'source/evidence changed while writing outputs')
        write_new(OUT / 'DATA_CONTRACT.json', {'schema': 1, 'record': 'BINDING_INTEGRATION_DATA_CONTRACT', 'experiment': EXPERIMENT,
            'rows': 3465, 'parent_worlds': 2761, 'exposures_per_view': 1, 'source_split': 'train',
            'files': {name: sha(OUT / name) for name in DATA_FILES}, 'source_geometry': old_build['source_geometry'],
            'admission_scope': scope, 'native_processed_tokens': 380921, 'processed_tokens': 577417, 'supervised_tokens': 39502,
            'optimizer_updates': 193, 'human_data_review': 'NOT_PERFORMED', 'training_permission': False,
            'model_execution_permission': False, 'quality_claim': False, 'release_permitted': False})
    except Exception as exc:
        write_new(OUT / 'FAILURE.json', {'status': 'FAILED_DATA_PREPARATION', 'type': type(exc).__name__, 'message': str(exc), 'partial_outputs_preserved': True})
        raise
    return {'status': build['status'], 'prepared': str(OUT), 'contract_sha256': sha(OUT / 'DATA_CONTRACT.json'),
            'files_sha256': {name: sha(OUT / name) for name in DATA_FILES}, 'optimizer_updates': 193,
            'native_processed_tokens': 380921, 'processed_tokens': 577417, 'supervised_tokens': 39502,
            'zero_addend_first2_geometry': build['zero_addend_first2_geometry'], 'coefficient_accounting': build['coefficient_accounting']}


if __name__ == '__main__':
    print(json.dumps(prepare(), indent=2))
