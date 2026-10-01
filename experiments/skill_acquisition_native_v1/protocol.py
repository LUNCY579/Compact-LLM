"""Pre-framework validation for prospective native skill-acquisition packets.

This module grants nothing. A packet must record a separate authorization,
native qualification, measured ceilings and completed S0 evidence. Rejected
packets never reach a framework loader. Synthetic CPU fixtures are explicit.
"""
from __future__ import annotations

import hashlib
import contextlib
import contextvars
import json
import math
from pathlib import Path
import struct
import time
import uuid

SCHEMA = 'SKILL-ACQUISITION-NATIVE-PACKET-1'
UPDATE_SCHEMA = 'SKILL-ACQUISITION-UPDATE-RECIPE-1'
MANIFEST = '86ac8f0d851f2178752357f62249eab5a23fd3de8fa032e7373614891df4cd44'
INVENTORIES = {'A': ([8, 27], 314618880), 'B': ([14, 27], 220233216)}
ARMS = {'R', 'S', 'P', 'MT', 'R_prime', 'S_prime', 'R_to_S', 'S_to_R'}
ANCHOR_KEYS = ('base_weights_sha256', 'delta_path', 'delta_sha256', 'arm_sha256')
LIMITS = ('generation_attempts', 'generated_token_charge', 'prefill_token_charge',
          'forward_attempts', 'backward_attempts', 'optimizer_attempts',
          'checkpoint_attempts', 'invocations', 'wall_seconds', 'host_floor_gib')
IO_LIMITS = ('identity_hash_attempts', 'identity_hash_bytes', 'chunk_io_read_bytes',
             'chunk_io_write_bytes', 'checkpoint_restore_attempts', 'model_load_attempts')
QUALIFICATION_SCHEMA = 'SKILL-ACQUISITION-NATIVE-QUALIFICATION-1'
PACKAGES = {'torch': '2.10.0+cu126', 'transformers': '4.57.1',
            'tokenizers': '0.22.2', 'safetensors': '0.7.0'}
SAMPLER = {'temperature': 1, 'top_p': 1, 'top_k': 0, 'repetition_penalty': 1,
           'max_new_tokens': 128, 'enable_thinking': False, 'use_cache': True,
           'batch_size': 1}
QUAL_COUNTS = {'native_sampled_trajectories': 16, 'native_task_scored_trajectories': 16,
               'native_parity_positions': 16, 'mechanics_token_fixtures': 2,
               'mechanics_reward_fixtures': 2,
               'loss_forwards': 16, 'backwards': 16, 'reference_backwards': 1,
               'optimizer_steps': 2, 'checkpoint_saves': 2, 'checkpoint_restores': 1}
QUAL_CASES = ('unequal_lengths', 'eos_included_once', 'length_cap_without_eos',
              'one_uniform_one_mixed', 'all_uniform_skips_adam',
              'empty_adam_state', 'C_greater_than_U', 'rollback_returned_update',
              'partial_step_failure_accounted')
_IO_OBSERVER = contextvars.ContextVar('skill_acquisition_admission_io', default=None)


@contextlib.contextmanager
def _io_scope(path, *, hashing=False, upper=None):
    callback = _IO_OBSERVER.get()
    tally = {'bytes': 0}
    if callback is None:
        yield tally
        return
    path = Path(path)
    ceiling = path.stat().st_size if upper is None else min(path.stat().st_size, upper)
    fields = {'operation_id': uuid.uuid4().hex, 'path': str(path.resolve()),
              'read_bytes_upper': ceiling, 'write_bytes_upper': 0,
              'identity_hash_attempts_upper': int(hashing),
              'identity_hash_bytes_upper': ceiling if hashing else 0}
    callback('input_io', 'started', **fields)
    start = time.perf_counter()
    try:
        yield tally
    except BaseException:
        callback('input_io', 'failed', **fields, known_bytes_read=tally['bytes'],
                 wall_s=time.perf_counter() - start)
        raise
    else:
        callback('input_io', 'returned', **fields, bytes_read=tally['bytes'],
                 wall_s=time.perf_counter() - start)


class AdmissionRejected(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise AdmissionRejected(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False).encode('ascii')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def hex64(value):
    return type(value) is str and len(value) == 64 and not set(value) - set('0123456789abcdef')


def integer(value, low=0):
    return type(value) is int and value >= low


def positive(value):
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def file_record(path):
    path = Path(path)
    require(path.is_file(), 'regular payload file required')
    h = hashlib.sha256()
    with _io_scope(path, hashing=True) as tally:
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                tally['bytes'] += len(block)
                h.update(block)
    return {'path': str(path.resolve()), 'bytes': path.stat().st_size, 'sha256': h.hexdigest()}


def read_json(path, maximum=32 * 1024 * 1024):
    def unique(pairs):
        out = {}
        for key, value in pairs:
            require(key not in out, 'duplicate JSON key: ' + key)
            out[key] = value
        return out
    path = Path(path)
    require(path.is_file(), 'regular JSON file required')
    with _io_scope(path, hashing=True, upper=maximum + 1) as tally:
        with path.open('rb') as stream:
            raw = stream.read(maximum + 1)
            tally['bytes'] += len(raw)
        observed_sha = hashlib.sha256(raw).hexdigest()
    require(len(raw) <= maximum, 'JSON size limit')
    value = json.loads(raw.decode('utf8'), object_pairs_hook=unique,
                       parse_constant=lambda _: (_ for _ in ()).throw(AdmissionRejected('nonfinite JSON')))
    require(type(value) is dict, 'JSON object required')
    return value, observed_sha


def bound_json(record, what):
    require(type(record) is dict and set(record) >= {'path', 'sha256'}, what + ' receipt binding')
    require(type(record['path']) is str and Path(record['path']).is_absolute()
            and hex64(record['sha256']), what + ' absolute path/digest')
    value, sha = read_json(record['path'])
    require(sha == record['sha256'], what + ' changed')
    return value


def registered(value, what):
    require(value.get('test_mode') is False and value.get('registered_run') is True,
            what + ' is not registered native evidence')


def anchor(value):
    require(type(value) is dict and set(value) >= set(ANCHOR_KEYS), 'anchor fields')
    require(all(hex64(value[k]) for k in ANCHOR_KEYS if k != 'delta_path'), 'anchor digests')
    require(type(value['delta_path']) is str and Path(value['delta_path']).is_absolute(),
            'anchor delta path')
    return {key: value[key] for key in ANCHOR_KEYS}


def validate_update_recipe(recipe, *, synthetic_cpu=False):
    require(type(recipe) is dict and recipe.get('schema') == UPDATE_SCHEMA, 'update recipe schema')
    require(recipe.get('synthetic_cpu') is synthetic_cpu, 'synthetic CPU declaration')
    require(integer(recipe.get('B'), 2) and recipe['B'] % 2 == 0 and integer(recipe.get('G'), 2),
            'even prompt batch and group size')
    objective = recipe.get('objective')
    require(type(objective) is dict and objective.get('method') in {'maxrl', 'grpo'}, 'objective')
    require(positive(objective.get('epsilon')), 'explicit finite epsilon')
    if objective['method'] == 'grpo':
        require(objective.get('ddof') == 1, 'GRPO sample standard deviation must be explicit')
    optimizer = recipe.get('optimizer')
    fields = {'class', 'lr', 'betas', 'eps', 'weight_decay', 'foreach', 'capturable'}
    require(type(optimizer) is dict and set(optimizer) >= fields, 'explicit optimizer fields')
    require(optimizer['class'] in {'AdamW', 'torch.optim.AdamW'}
            and positive(optimizer['lr']) and positive(optimizer['eps']),
            'AdamW learning rate/epsilon')
    require(type(optimizer['betas']) in (list, tuple) and len(optimizer['betas']) == 2
            and all(type(x) in (int, float) and math.isfinite(x) and 0 <= x < 1
                    for x in optimizer['betas']), 'AdamW betas')
    require(type(optimizer['weight_decay']) in (int, float) and optimizer['weight_decay'] == 0,
            'unsupported weight decay')
    require(optimizer['foreach'] is False and optimizer['capturable'] is (not synthetic_cpu),
            'optimizer execution path')
    require('ratio_clipping' in recipe and recipe['ratio_clipping'] is None
            and 'gradient_clipping' in recipe and recipe['gradient_clipping'] is None,
            'this implementation has no ratio/gradient clipping')
    kl = recipe.get('anchor_KL')
    require(type(kl) is dict and kl.get('enabled') is False
            and type(kl.get('coefficient')) in (int, float) and kl['coefficient'] == 0,
            'this implementation has no reference KL')
    require(integer(recipe.get('max_new_tokens'), 1) and integer(recipe.get('pad_id')),
            'decode cap/padding ID')
    require(type(recipe.get('eos_ids')) is list and recipe['eos_ids']
            and all(integer(x) for x in recipe['eos_ids'])
            and len(set(recipe['eos_ids'])) == len(recipe['eos_ids']), 'EOS IDs')
    if 'gradient_checkpointing' in recipe:
        require(recipe['gradient_checkpointing'] == {'use_reentrant': False}, 'checkpointing execution path')
    if not synthetic_cpu:
        require(recipe.get('approved') is True and recipe.get('training_authorized') is True,
                'update recipe not authorized')
        require(recipe['B'] == 2 and recipe['G'] == 8 and objective['epsilon'] == 1e-6,
                'registered B2/G8/epsilon drift')
        require(objective['method'] == 'maxrl', 'native GRPO requires a separately implemented registration')
        require(optimizer['lr'] == 1e-6 and list(optimizer['betas']) == [0.9, 0.999]
                and optimizer['eps'] == 1e-8, 'selected native optimizer drift')
        require(recipe['max_new_tokens'] == 128 and recipe['pad_id'] == 151643
                and recipe['eos_ids'] == [151645, 151643], 'registered decoding drift')
    return recipe


def _schema(value, schema, stage, what):
    require(value.get('schema') == schema and value.get('stage') == stage, what + ' schema/stage')
    registered(value, what)


def _preimage(value):
    require(type(value) is dict and value.get('schema') == 'COMP-RS-001-S0-DECODING-1', 'runtime preimage schema')
    require(value.get('device') == 'cuda:0' and value.get('dtype') == 'float32'
            and value.get('num_threads') == 4 and value.get('cublas_workspace_config') == ':4096:8'
            and value.get('deterministic_algorithms') is True
            and value.get('deterministic_warn_only') is False
            and value.get('tf32_cuda_matmul') is False and value.get('tf32_cudnn') is False
            and value.get('float32_matmul_precision') == 'highest', 'fixed FP32 runtime drift')
    require(value.get('packages') == PACKAGES and value.get('python') == '3.11.15', 'runtime package drift')
    decode = value.get('decoding') or {}
    require(decode.get('greedy') is True and decode.get('do_sample') is False
            and decode.get('batch_size') == 1 and decode.get('max_new_tokens') == 128
            and decode.get('enable_thinking') is False
            and decode.get('pad_token_id') == 151643
            and decode.get('eos_token_id') == [151645, 151643], 'frozen scored decoding drift')
    require(type(value.get('cuda')) is dict and value['cuda'].get('device_name')
            and value['cuda'].get('driver_version'), 'CUDA/driver identity missing')
    return digest(value)


def _freeze(value, schema):
    _schema(value, schema, 'S0_RUNTIME_FREEZE', 'warm freeze')
    require(_preimage(value.get('decoding_preimage')) == value.get('decoding_sha256'), 'freeze preimage digest')
    require(positive(value.get('host_floor_gib')) and hex64(value.get('scoring_code_bundle_sha256')),
            'freeze host/scorer identity')
    require(type(value.get('anchor')) is dict
            and set(value['anchor']) == set(ANCHOR_KEYS) | {'warmstart_receipt_sha256'}
            and hex64(value['anchor']['warmstart_receipt_sha256']), 'warm freeze receipt anchor')
    anchor(value['anchor'])


def _same_freeze(receipt, reference, value, what):
    bound = receipt.get('runtime_freeze') or {}
    require(bound.get('sha256') == reference['sha256'] and bound.get('anchor') == value['anchor']
            and bound.get('decoding_sha256') == value['decoding_sha256'], what + ' freeze provenance')


def _payload(record, what):
    require(type(record) is dict and type(record.get('path')) is str
            and Path(record['path']).is_absolute() and hex64(record.get('sha256')), what + ' payload binding')
    observed = file_record(record['path'])
    require(observed['sha256'] == record['sha256'], what + ' payload changed')
    return observed


def _delta_digest(path):
    with _io_scope(path, hashing=True) as tally:
        return _delta_digest_stream(path, tally)


def _delta_digest_stream(path, tally):
    """Stream the safetensors data section without a framework or tensor load."""
    with Path(path).open('rb') as stream:
        first = stream.read(8)
        tally['bytes'] += len(first)
        require(len(first) == 8, 'delta header truncated')
        size = struct.unpack('<Q', first)[0]
        require(0 < size <= 1024 * 1024, 'delta header size')
        raw = stream.read(size)
        tally['bytes'] += len(raw)
        require(len(raw) == size, 'delta header truncated')
        header = json.loads(raw)
        header.pop('__metadata__', None)
        require(header, 'delta tensors missing')
        end = 0
        for info in sorted(header.values(), key=lambda x: x['data_offsets'][0]):
            a, b = info['data_offsets']
            require(info['dtype'] == 'F32' and a == end
                    and b - a == 4 * math.prod(info['shape']), 'delta layout/dtype')
            end = b
        h = hashlib.sha256()
        remaining = end
        while remaining:
            block = stream.read(min(1024 * 1024, remaining))
            tally['bytes'] += len(block)
            require(block, 'delta data truncated')
            h.update(block)
            remaining -= len(block)
        trailing = stream.read(1)
        tally['bytes'] += len(trailing)
        require(not trailing, 'delta trailing bytes')
        return h.hexdigest()


@contextlib.contextmanager
def _journal_stream(path):
    with _io_scope(path) as tally:
        with Path(path).open('rb') as raw:
            class Reader:
                def readline(self, maximum):
                    line = raw.readline(maximum)
                    tally['bytes'] += len(line)
                    return line
            yield Reader()


def _warm(initial, expected):
    q1 = bound_json((initial.get('inputs') or {}).get('q1'), 'native Q1')
    _schema(q1, 'COMP-RS-001-S0-Q1-FP32-1', 'S0_Q1_FP32', 'Q1')
    require(q1.get('passed') is True and q1.get('failed_checks') == []
            and q1.get('decoding_preimage') == initial['decoding_preimage']
            and q1.get('decoding_sha256') == initial['decoding_sha256'], 'passed matching Q1 runtime')
    peak = (q1.get('host_floor') or {}).get('peak_worker_rss_gib')
    require(positive(peak) and initial['host_floor_gib'] == peak + 1, 'initial Q1 host floor')
    ref = (initial.get('inputs') or {}).get('warmstart')
    warm = bound_json(ref, 'completed warmstart')
    _schema(warm, 'COMP-RS-001-S0-WARMSTART-1', 'S0_WARMSTART', 'warmstart')
    require(ref['sha256'] == initial['anchor']['warmstart_receipt_sha256'], 'warmstart receipt ancestry')
    require(warm.get('completed') is True and warm.get('outcome') == 'WARMSTART_COMPLETE'
            and anchor(warm.get('anchor')) == expected, 'warmstart completion/anchor')
    plan, training, steps = warm.get('plan') or {}, warm.get('training') or {}, warm.get('steps')
    require(plan.get('rows') == 600 and type(plan.get('rows')) is int
            and plan.get('steps') == 75 and type(plan.get('steps')) is int
            and training.get('steps') == 75 and type(training.get('steps')) is int
            and type(steps) is list and len(steps) == 75
            and all(type(step) is dict and type(step.get('index')) is int for step in steps)
            and [step.get('index') for step in steps] == list(range(75)), 'warmstart 600/75 applied counts')
    fingerprints = warm.get('fingerprints') or {}
    require(fingerprints.get('frozen_unchanged') is True
            and fingerprints.get('trainable_changed') is True
            and fingerprints.get('final_equals_theta0') is True, 'warmstart state acceptance')
    inputs, delta = warm.get('inputs') or {}, warm.get('delta') or {}
    require(inputs.get('dataset_manifest_sha256') == MANIFEST, 'warmstart data provenance')
    require(delta.get('path') == expected['delta_path']
            and delta.get('data_section_sha256') == expected['delta_sha256'], 'warm delta receipt')
    _payload({'path': delta['path'], 'sha256': delta.get('file_sha256')}, 'warm delta')
    require(_delta_digest(delta['path']) == expected['delta_sha256'], 'warm delta data digest')
    asset_ref = {'path': inputs.get('asset_receipt'), 'sha256': inputs.get('asset_receipt_sha256')}
    assets = bound_json(asset_ref, 'base assets')
    require(assets.get('stage') == 'Q1_ASSET_CANDIDATE', 'base asset receipt stage')
    root = Path(assets.get('model_path', ''))
    require(root.is_absolute(), 'absolute model asset root')
    require((assets.get('assets') or {}).get('model.safetensors', {}).get('sha256') == expected['base_weights_sha256'],
            'base weight ancestry')
    for name, record in assets['assets'].items():
        require(Path(name).name == name, 'asset filename scope')
        _payload({'path': str(root / name), 'sha256': record.get('sha256')}, 'base asset ' + name)
    require(hex64(inputs.get('tokenizer_receipt_sha256')), 'tokenizer receipt binding')
    return warm, assets


def _competence(value, initial_ref, initial, expected):
    _schema(value, 'COMP-RS-001-S0-COMPETENCE-2', 'S0_COMPETENCE', 'competence')
    require(value.get('completed') is True and value.get('outcome') == 'SIGNAL', 'full warmed competence SIGNAL required')
    _same_freeze(value, initial_ref, initial, 'competence')
    require(anchor(value.get('anchor')) == expected and value.get('decoding_sha256') == initial['decoding_sha256'], 'competence identity')
    inputs = value.get('inputs') or {}
    require(inputs.get('dataset_manifest_sha256') == MANIFEST and inputs.get('full_plan') is True
            and type(inputs.get('planned')) is int and inputs['planned'] == 450
            and inputs.get('plan_by_skill') == {'R': 150, 'S': 150, 'P': 150}, 'competence full 450-row plan')
    require(value.get('group_size') == 8 and type(value.get('group_size')) is int
            and value.get('m_min') == .1 and value.get('delta_acq_rows') == 13,
            'competence G8/margin registration')
    summary = value.get('summary') or {}
    require(set(summary) == {'R', 'S', 'P'}, 'competence skill population')
    for skill, entry in summary.items():
        groups = entry.get('groups') or {}
        require(entry.get('prompts') == 150 and type(entry.get('prompts')) is int
                and entry.get('samples') == 1200 and type(entry.get('samples')) is int
                and entry.get('G') == 8 and type(entry.get('G')) is int, 'competence per-skill counts')
        require(set(groups) == {'mixed', 'all_fail', 'all_success'}
                and all(integer(v) for v in groups.values()) and sum(groups.values()) == 150,
                'competence group counts')
        if skill in {'R', 'S'}:
            require(groups['mixed'] >= 15, 'competence strict mixed-group gate')
    require((value.get('scorer_bundle') or {}).get('expected') == initial['scoring_code_bundle_sha256'], 'competence scorer bundle')


def _pilots(adopted, initial_ref, initial):
    refs = adopted.get('inputs') or {}
    pilot_ref = refs.get('pilot')
    pilot = bound_json(pilot_ref, 'adopted pilot')
    inventory = adopted['inventory']['name']
    require(pilot.get('stage') == 'S0_PILOT_' + inventory, 'adopted pilot stage')
    registered(pilot, 'adopted pilot')
    _same_freeze(pilot, initial_ref, initial, 'adopted pilot')
    require(pilot.get('anchor') == initial['anchor'] and pilot.get('decoding_sha256') == initial['decoding_sha256'], 'pilot identity')
    decision = pilot.get('decision') or {}
    require(pilot.get('completed') is True and all(pilot.get(k) is None for k in ('error', 'abort', 'oom'))
            and decision.get('outcome') == 'ADOPT_' + inventory
            and decision.get('inventory') == inventory
            and integer(pilot.get('updates'), 1), 'adopted pilot actual update/completion')
    peaks = [decision.get('peak_worker_rss_gib')]
    if inventory == 'B':
        a_ref = refs.get('a_pilot')
        a = bound_json(a_ref, 'A fallback pilot')
        registered(a, 'A fallback pilot')
        require(a.get('stage') == 'S0_PILOT_A'
                and (a.get('decision') or {}).get('outcome') == 'RERUN_WITH_B', 'A fallback outcome')
        _same_freeze(a, initial_ref, initial, 'A fallback')
        require(a.get('anchor') == initial['anchor'] and a.get('decoding_sha256') == initial['decoding_sha256'], 'A fallback identity')
        after = pilot.get('after_a_pilot') or {}
        require(after.get('sha256') == a_ref['sha256'] and after.get('anchor') == initial['anchor']
                and after.get('outcome') == 'RERUN_WITH_B' and after.get('registered_run') is True,
                'B pilot exact A ancestry')
        peaks.append(a['decision'].get('peak_worker_rss_gib'))
    else:
        require(refs.get('a_pilot') is None, 'A adoption has no fallback receipt')
    require(all(positive(peak) for peak in peaks), 'measured pilot peak RSS')
    require(adopted['host_floor_gib'] == max([initial['host_floor_gib']] + [peak + 1 for peak in peaks]), 'adopted measured host floor')


def _churn(value, adopted_ref, adopted, expected):
    _schema(value, 'COMP-RS-001-S0-CHURN-1', 'S0_CHURN', 'churn')
    require(value.get('outcome') == 'CHURN_ZERO' and type(value.get('churn')) is int
            and value['churn'] == 0 and type(value.get('rows_compared')) is int
            and value['rows_compared'] == 1650, 'full zero-churn enrollment required')
    _same_freeze(value, adopted_ref, adopted, 'churn')
    require(value.get('separate_processes') is True and value.get('separate_processes_required') is True
            and value.get('shared_processes') == []
            and value.get('scores_equal') == {'guard_dev_scoring': True, 'calibration_scoring': True}, 'separate-process scored churn acceptance')
    runs = value.get('runs') or {}
    require(set(runs) == {'1', '2'}, 'two enrollment receipts required')
    process_sets, session_sets = [], []
    for run in (1, 2):
        reference = runs[str(run)]
        enroll = bound_json({'path': reference.get('receipt'), 'sha256': reference.get('receipt_sha256')}, 'enrollment run')
        _schema(enroll, 'COMP-RS-001-S0-ENROLL-1', 'S0_ENROLL_RUN%d' % run, 'enrollment')
        require(enroll.get('completed') is True and enroll.get('run') == run
                and enroll.get('native_outputs_scored') is True and enroll.get('fingerprint_unchanged') is True,
                'complete unchanged enrollment')
        _same_freeze(enroll, adopted_ref, adopted, 'enrollment')
        require(anchor(enroll.get('anchor')) == expected
                and enroll.get('decoding_sha256') == adopted['decoding_sha256']
                and (enroll.get('scorer_bundle') or {}).get('expected') == adopted['scoring_code_bundle_sha256'], 'enrollment identity/scorer')
        inputs, summary = enroll.get('inputs') or {}, enroll.get('summary') or {}
        require(inputs.get('full_guard_dev') is True and inputs.get('full_calibration') is True
                and inputs.get('dataset_manifest_sha256') == MANIFEST and inputs.get('planned') == 1650
                and inputs.get('plan_by_bank') == {'guard_dev': 1200, 'calibration': 450}
                and (summary.get('guard_dev') or {}).get('rows') == 1200
                and (summary.get('calibration') or {}).get('rows') == 450, 'enrollment bank counts')
        journal = enroll.get('journal') or {}
        require(journal.get('rows') == 1650 and type(journal.get('rows')) is int
                and journal.get('path') == reference.get('journal')
                and journal.get('sha256') == reference.get('journal_sha256'), 'enrollment journal count/binding')
        _payload(journal, 'enrollment journal')
        bind = journal.get('bind') or {}
        require(bind.get('stage') == 'S0_ENROLL_RUN%d' % run and bind.get('run') == run
                and anchor(bind.get('anchor')) == expected
                and bind.get('decoding_sha256') == adopted['decoding_sha256'], 'enrollment journal bind')
        require(journal.get('bind_sha256') == digest(bind), 'enrollment journal bind digest')
        ids, observed_processes, observed_sessions = [], set(), set()
        with _journal_stream(journal['path']) as stream:
            while True:
                line = stream.readline(1024 * 1024 + 1)
                if not line:
                    break
                require(len(line) <= 1024 * 1024 and line.endswith(b'\n'), 'enrollment journal torn/oversized row')
                row = json.loads(line)
                body = {key: item for key, item in row.items() if key != 'row_sha256'}
                require(row.get('row_sha256') == digest(body)
                        and canonical(row) + b'\n' == line, 'enrollment journal row hash/canonical bytes')
                require(all(row.get(key) == item for key, item in bind.items()), 'enrollment journal row bind drift')
                ids.append(row.get('id'))
                process = row.get('process') or {}
                require(integer(process.get('pid'), 1) and positive(process.get('process_created'))
                        and type(process.get('session')) is str and process['session'], 'journal process identity')
                observed_processes.add((process.get('pid'), process.get('process_created')))
                observed_sessions.add(process.get('session'))
        require(len(ids) == 1650 and len(set(ids)) == 1650
                and all(type(row_id) is str and row_id for row_id in ids)
                and digest(ids) == bind.get('plan_sha256'), 'actual full enrollment journal population')
        processes, sessions = reference.get('processes'), reference.get('sessions')
        require(type(processes) is list and processes and type(sessions) is list and sessions, 'enrollment process/session evidence')
        require({tuple(process) for process in processes} == observed_processes
                and set(sessions) == observed_sessions, 'enrollment actual process/session provenance')
        process_sets.append(observed_processes)
        session_sets.append(observed_sessions)
    require(not process_sets[0] & process_sets[1] and not session_sets[0] & session_sets[1], 'enrollment process/session overlap')


def validate_prerequisites(packet):
    prerequisites = packet.get('prerequisites')
    require(type(prerequisites) is dict, 'S0 prerequisites missing')
    initial = bound_json(prerequisites.get('initial_warm_freeze'), 'initial warm freeze')
    competence = bound_json(prerequisites.get('competence'), 'competence')
    adopted = bound_json(prerequisites.get('adopted_warm_freeze'), 'adopted warm freeze')
    churn = bound_json(prerequisites.get('churn'), 'churn')
    for value, name in ((initial, 'initial freeze'), (competence, 'competence'),
                        (adopted, 'adopted freeze'), (churn, 'churn')):
        registered(value, name)
    _freeze(initial, 'COMP-RS-001-S0-FREEZE-WARM-1')
    _freeze(adopted, 'COMP-RS-001-S0-FREEZE-WARM-PILOT-1')
    initial_sha = prerequisites['initial_warm_freeze']['sha256']
    expected = anchor(packet['anchor'])
    require(anchor(initial.get('anchor')) == expected and anchor(adopted.get('anchor')) == expected
            and initial['anchor'] == adopted['anchor'],
            'common warmed anchor changed')
    require(initial.get('stage') == adopted.get('stage') == 'S0_RUNTIME_FREEZE', 'freeze stages')
    require(competence.get('stage') == 'S0_COMPETENCE' and competence.get('completed') is True
            and competence.get('outcome') == 'SIGNAL', 'full warmed competence SIGNAL required')
    bound = competence.get('runtime_freeze') or {}
    require(bound.get('sha256') == initial_sha and anchor(bound.get('anchor')) == expected,
            'competence initial-freeze provenance')
    require(anchor(competence.get('anchor')) == expected, 'competence anchor')
    decoding = initial.get('decoding_sha256')
    require(hex64(decoding) and competence.get('decoding_sha256') == adopted.get('decoding_sha256')
            == decoding, 'decoding identity drift')
    adopted_inputs = adopted.get('inputs') or {}
    require((adopted_inputs.get('initial_warm_freeze') or {}).get('sha256') == initial_sha
            and (adopted_inputs.get('competence') or {}).get('sha256')
            == prerequisites['competence']['sha256'], 'adopted-freeze ancestry')
    inventory = adopted.get('inventory') or {}
    name = inventory.get('name')
    require(name in INVENTORIES and inventory.get('blocks') == INVENTORIES[name][0]
            and inventory.get('registered_numel') == INVENTORIES[name][1], 'adopted inventory')
    require(packet.get('inventory') == {'name': name, 'blocks': INVENTORIES[name][0],
                                       'numel': INVENTORIES[name][1]}, 'packet inventory drift')
    require((initial.get('inventory') or {}).get('name') == 'PENDING'
            and inventory.get('decision', {}).get('outcome') == 'ADOPT_' + name, 'freeze phase/adoption decision')
    require(initial['decoding_preimage'] == adopted['decoding_preimage']
            and initial['scoring_code_bundle_sha256'] == adopted['scoring_code_bundle_sha256'], 'adopted runtime/scorer changed')
    require(churn.get('stage') == 'S0_CHURN' and churn.get('outcome') == 'CHURN_ZERO'
            and churn.get('churn') == 0 and churn.get('rows_compared') == 1650,
            'full zero-churn enrollment required')
    churn_freeze = churn.get('runtime_freeze') or {}
    require(churn_freeze.get('sha256') == prerequisites['adopted_warm_freeze']['sha256']
            and anchor(churn_freeze.get('anchor')) == expected, 'churn adopted-freeze provenance')
    warm, assets = _warm(initial, expected)
    _competence(competence, prerequisites['initial_warm_freeze'], initial, expected)
    _pilots(adopted, prerequisites['initial_warm_freeze'], initial)
    _churn(churn, prerequisites['adopted_warm_freeze'], adopted, expected)
    return {'initial': initial, 'competence': competence, 'adopted': adopted, 'churn': churn,
            'warm': warm, 'assets': assets}


def _planning(packet, proposal, prerequisites):
    """Rebuild the exact native plan/caps with the pure-stdlib planning module."""
    from . import plan as planning
    recommended = proposal.get('recommended_recipe') or {}
    optimizer = recommended.get('optimizer') or {}
    require((recommended.get('objective') or {}).get('name') == 'Practical centered MaxRL'
            and optimizer.get('learning_rate') == 1e-6 and optimizer.get('betas') == [.9, .999]
            and optimizer.get('epsilon') == 1e-8 and optimizer.get('schedule') == 'constant'
            and optimizer.get('warmup_steps') == 0
            and (recommended.get('ratio_clipping') or {}).get('enabled') is False
            and (recommended.get('anchor_KL') or {}).get('enabled') is False
            and all(value is False for value in (recommended.get('other_interventions') or {}).values()),
            'selected recommended proposal recipe drift')
    data_ref = packet.get('data_manifest')
    data = bound_json(data_ref, 'dataset manifest')
    require(data_ref['sha256'] == MANIFEST, 'actual dataset manifest identity')
    for relative in ('canonical/train.jsonl', 'policy/train.jsonl'):
        entry = (data.get('files') or {}).get(relative) or {}
        _payload({'path': str(Path(data_ref['path']).parent / relative),
                  'sha256': entry.get('sha256')}, 'training source ' + relative)
    needed = planning.ARMS[packet['arm']]
    refs = packet.get('atomic_manifests')
    require(type(refs) is dict and set(refs) == set(needed), 'atomic manifest scope')
    manifests = {}
    for skill in needed:
        relative = 'manifests/atomic_%s.json' % skill
        entry = (data.get('files') or {}).get(relative) or {}
        require(entry.get('sha256') == refs[skill].get('sha256'), 'atomic manifest dataset binding')
        manifests[skill] = bound_json(refs[skill], 'atomic ' + skill)
    try:
        planned = planning.build_plan(proposal, manifests, packet['arm'])
    except (ValueError, KeyError, TypeError) as error:
        raise AdmissionRejected('exact native plan: ' + str(error)) from error
    exported = bound_json(packet.get('plan'), 'ordered plan')
    require(exported == planned.export() and exported['synthetic'] is False,
            'exact ordered plan/seed/proposal binding')
    census = bound_json(packet.get('census'), 'tokenizer census')
    require(census.get('schema') == 'S1_PROMPT_LENGTH_CENSUS-1'
            and census.get('plan_sha256') == planned.sha256
            and census.get('tokenizer_receipt_sha256') == prerequisites['warm']['inputs']['tokenizer_receipt_sha256']
            and census.get('measured') is True, 'measured plan/tokenizer census binding')
    timings = bound_json(packet.get('timings'), 'measured timing prices')
    require(timings.get('schema') == 'S1_TIMING_PRICES-1' and timings.get('measured') is True
            and timings.get('inventory') == packet['inventory']
            and timings.get('update_recipe_sha256') == digest(packet['update_recipe'])
            and timings.get('qualification_sha256') == packet['qualification']['sha256'], 'timing qualification/scope binding')
    prices = timings.get('prices')
    require(type(prices) is dict and all(positive(value) for value in prices.values()), 'positive measured timing prices')
    io = bound_json(packet.get('io_budget'), 'measured unit IO budget')
    require(io.get('schema') == 'S1_IO_BUDGET-1' and io.get('measured') is True
            and io.get('plan_sha256') == planned.sha256 and io.get('inventory') == packet['inventory']
            and io.get('qualification_sha256') == packet['qualification']['sha256'],
            'unit IO budget measurement/plan/qualification binding')
    try:
        caps = planning.derive_caps(planned, census.get('prompt_lengths'), prices,
                                    packet['inventory']['numel'], unit_io_budget=io)
    except (ValueError, KeyError, TypeError, ZeroDivisionError) as error:
        raise AdmissionRejected('resolved plan ceilings: ' + str(error)) from error
    require(not caps['unresolved'], 'unresolved plan ceilings')
    return planned, caps, census, timings


def validate_qualification_evidence(qualification, packet, planned):
    """Audit prospective qualification populations; this never grants execution.

    Required bound JSON files are ``qualification_scope`` (schema
    S1_QUALIFICATION_SCOPE-1), ``scope_authorization``, ``evidence`` (schema
    S1_QUALIFICATION_EVIDENCE-1), and ``cost_seal``. Scope names the receipt's
    code/anchor/start/recipe/runtime/data bindings and explicit gradient metrics
    and tolerances. Its separate authorization names the exact scope hash.

    Evidence contains ``native_samples``, ``token_fixtures``, ``reward_fixtures``
    and ``operations``. Every operation has a unique id, scope hash, origin,
    kind, phase, input and output. Native samples reference returned generation,
    differentiable teacher-forced and two distinct strict task-scorer operations.
    Token/reward fixtures are explicitly supplied inputs, never native samples.
    Physical mechanics counts and cases are derived from returned/failed
    operation observations, rather than case booleans. The final cost seal binds
    the evidence hash, operation hash and derived counts.

    Metadata cannot establish physical execution by itself. Acceptance depends
    on a separately authorized, controlled native qualifier producing the sealed
    operation observations; receipt-shaped unit fixtures are only parser tests.
    """
    scope_ref = qualification.get('qualification_scope')
    scope = bound_json(scope_ref, 'qualification scope')
    bindings = ('code_sha256', 'anchor', 'start', 'inventory', 'update_recipe_sha256',
                'sampler_sha256', 'plan_sha256', 'manifest_sha256',
                'tokenizer_receipt_sha256', 'runtime_preimage', 'decoding_sha256',
                'scorer_bundle_sha256')
    require(scope.get('schema') == 'S1_QUALIFICATION_SCOPE-1'
            and all(scope.get(key) == qualification[key] for key in bindings),
            'qualification scope source/recipe/anchor binding')
    authorization = bound_json(qualification.get('scope_authorization'), 'qualification authorization')
    require(authorization.get('approved') is True
            and authorization.get('qualification_authorized') is True
            and authorization.get('scope_sha256') == scope_ref['sha256'],
            'separate qualification scope authorization')
    metric = scope.get('gradient_reference') or {}
    require(metric.get('norm') == 'l2' and metric.get('error_metric') == 'max_abs_and_relative_l2'
            and metric.get('zero_reference_rule') == 'absolute_only'
            and hex64(metric.get('implementation_sha256'))
            and positive(metric.get('absolute_tolerance')) and positive(metric.get('relative_tolerance')),
            'explicit registered gradient reference metrics/tolerances')
    scorers = scope.get('strict_scorers') or {}
    require(set(scorers) == {'native', 'independent'} and all(hex64(value) for value in scorers.values())
            and scorers['native'] != scorers['independent'], 'registered independent strict scorer implementations')
    evidence_ref = qualification.get('evidence')
    evidence = bound_json(evidence_ref, 'qualification operation evidence')
    require(evidence.get('schema') == 'S1_QUALIFICATION_EVIDENCE-1'
            and evidence.get('scope_sha256') == scope_ref['sha256'], 'qualification evidence scope')
    operations = evidence.get('operations')
    require(type(operations) is list and operations, 'qualification operation population')
    by_id = {}
    for operation in operations:
        require(type(operation) is dict and type(operation.get('id')) is str and operation['id']
                and operation['id'] not in by_id and operation.get('scope_sha256') == scope_ref['sha256']
                and operation.get('phase') in {'returned', 'failed', 'unknown'}
                and operation.get('origin') in {'actual_native_sampling', 'actual_task_scoring',
                                               'declared_mechanics_fixture'}
                and type(operation.get('input')) is dict and type(operation.get('output')) is dict,
                'qualification operation identity/scope/phase')
        by_id[operation['id']] = operation
    operation_order = {operation['id']: index for index, operation in enumerate(operations)}

    used = set()
    def observed(identifier, kind, origin, *, phase='returned', cuda=False):
        operation = by_id.get(identifier) or {}
        require(operation.get('kind') == kind and operation.get('origin') == origin
                and operation.get('phase') == phase
                and (not cuda or operation.get('device') == 'cuda:0'),
                'qualification operation provenance: ' + kind)
        if origin != 'declared_mechanics_fixture':
            used.add(identifier)
        return operation['input'], operation['output']

    samples = evidence.get('native_samples')
    tokens = evidence.get('token_fixtures')
    rewards = evidence.get('reward_fixtures')
    require(type(samples) is list and len(samples) >= 16
            and type(tokens) is list and len(tokens) >= 2
            and type(rewards) is list and len(rewards) >= 2, 'separate native/fixture populations')
    ids, native_ids, policy_by_boundary, parity_errors = set(), set(), {}, []
    def unique(record, origin):
        require(type(record) is dict and record.get('origin') == origin
                and type(record.get('id')) is str and record['id'] and record['id'] not in ids,
                'disjoint native/fixture identity and origin')
        ids.add(record['id'])

    def token_sequence(value, reason):
        require(type(value) is list and 1 <= len(value) <= 128
                and all(integer(token) for token in value), 'qualification completion tokens')
        eos = [index for index, token in enumerate(value) if token in packet['update_recipe']['eos_ids']]
        require((reason == 'eos' and eos == [len(value) - 1])
                or (reason == 'length_cap' and len(value) == 128 and not eos),
                'qualification EOS/cap token provenance')
        return eos

    def logprobs(value, length):
        require(type(value) is list and len(value) == length
                and all(type(x) in (int, float) and math.isfinite(x) and x <= 0 for x in value),
                'qualification per-token likelihood population')
        return value

    for sample in samples:
        unique(sample, 'actual_native_sample')
        trajectory = sample.get('trajectory') or {}
        boundary = trajectory.get('logical_boundary')
        require(integer(boundary) and boundary < len(planned.boundaries), 'native qualification boundary')
        prompt = next((index for index, item in enumerate(planned.boundaries[boundary].prompts)
                       if item.row_id == trajectory.get('row_id')), None)
        require(prompt is not None and trajectory == planned.trajectory(boundary, prompt, trajectory.get('sample_k'))
                and trajectory['logical_id'] not in native_ids, 'native qualification exact trajectory/seed')
        native_ids.add(trajectory['logical_id'])
        policy = sample.get('policy_sha256')
        require(hex64(policy) and policy_by_boundary.setdefault(boundary, policy) == policy,
                'native qualification one behavior policy within boundary')
        generated_input, generated = observed(sample.get('generation_operation_id'), 'generation',
                                               'actual_native_sampling', cuda=True)
        require(generated_input == {'trajectory': trajectory, 'policy_sha256': policy,
                                    'sampler_sha256': qualification['sampler_sha256'],
                                    'token_source': 'native_model_generate', 'reward_override': False},
                'native generation source/seed/sampler; fixtures cannot be native samples')
        completion = generated.get('completion_ids')
        token_sequence(completion, generated.get('stop_reason'))
        require(type(generated.get('completion_text')) is str, 'native completion scoring text')
        raw = logprobs(generated.get('raw_step_logprobs'), len(completion))
        completion_sha = digest(generated)
        teacher_input, teacher = observed(sample.get('teacher_operation_id'), 'teacher_forced',
                                         'actual_native_sampling', cuda=True)
        expected_input = {'generation_operation_id': sample['generation_operation_id'],
                          'completion_sha256': completion_sha, 'policy_sha256': policy}
        require(teacher_input == dict(expected_input, differentiable=True,
                                      token_source='returned_native_generation')
                and operation_order[sample['generation_operation_id']] < operation_order[sample['teacher_operation_id']],
                'native differentiable teacher-score provenance')
        forced = logprobs(teacher.get('selected_logprobs'), len(completion))
        errors = [abs(a - b) for a, b in zip(raw, forced)]
        require(all(error <= 1e-4 for error in errors), 'native per-token likelihood parity')
        parity_errors.extend(errors)
        scorer_ids = []
        task_rewards = []
        for field, kind, scorer_key in (('native_scorer_operation_id', 'strict_task_scorer', 'native'),
                                       ('independent_scorer_operation_id', 'independent_strict_task_scorer', 'independent')):
            scoring_input, scoring = observed(sample.get(field), kind, 'actual_task_scoring')
            require(all(scoring_input.get(key) == value for key, value in expected_input.items())
                    and operation_order[sample['generation_operation_id']] < operation_order[sample[field]]
                    and scoring_input.get('trajectory') == trajectory
                    and scoring_input.get('reward_source') == 'strict_task_scoring'
                    and scoring_input.get('reward_override') is False
                    and scoring_input.get('scorer_bundle_sha256') == qualification['scorer_bundle_sha256']
                    and scoring_input.get('implementation_sha256') == scorers[scorer_key],
                    'independent strict task scorer provenance; fixture rewards excluded')
            scorer_ids.append(scoring_input['implementation_sha256'])
            require(type(scoring.get('reward')) is int and scoring['reward'] in (0, 1), 'strict task reward')
            task_rewards.append(scoring['reward'])
        require(scorer_ids[0] != scorer_ids[1] and task_rewards[0] == task_rewards[1],
                'independent strict scorer parity')

    fixture_ids, lengths, eos_count, capped_count = set(), [], 0, 0
    for fixture in tokens:
        unique(fixture, 'declared_mechanics_token_fixture')
        fixture_ids.add(fixture['id'])
        require(fixture.get('token_source') == 'supplied_fixture_tokens'
                and fixture.get('counts_as_native_sample') is False
                and integer(fixture.get('prompt_length'), 1) and integer(fixture.get('padding_length')),
                'explicit supplied token fixture provenance')
        completion = fixture.get('completion_ids')
        eos = token_sequence(completion, fixture.get('stop_reason'))
        inp, output = observed(fixture.get('mask_operation_id'), 'loss_forward',
                               'declared_mechanics_fixture', cuda=True)
        n, start = len(completion), fixture['prompt_length'] - 1
        require(inp.get('token_fixture_id') == fixture['id'] and fixture['id'] in inp.get('fixture_ids', [])
                and output.get('mask') == {'selected_predictor_positions': list(range(start, start + n)),
                                          'prompt_positions_selected': 0, 'pad_positions_selected': 0,
                                          'eos_positions_selected': len(eos)},
                'observed fixture one-shift completion/EOS/prompt/pad mask')
        lengths.append(n)
        eos_count += len(eos)
        capped_count += int(fixture['stop_reason'] == 'length_cap')
    reward_map = {}
    for fixture in rewards:
        unique(fixture, 'declared_mechanics_reward_fixture')
        fixture_ids.add(fixture['id'])
        value = fixture.get('rewards')
        require(fixture.get('reward_source') == 'supplied_fixture_rewards'
                and fixture.get('counts_as_task_reward') is False
                and type(value) is list and len(value) == 2
                and all(type(group) is list and len(group) == 8
                        and all(type(x) is int and x in (0, 1) for x in group) for group in value),
                'explicit B2/G8 mechanics reward fixtures')
        reward_map[fixture['id']] = value

    mechanics = [operation for operation in operations
                 if operation.get('origin') == 'declared_mechanics_fixture']
    require(mechanics and all(operation.get('device') == 'cuda:0'
                              and operation['input'].get('fixture_ids')
                              and set(operation['input']['fixture_ids']) <= fixture_ids for operation in mechanics),
            'mechanics operation fixture/CUDA provenance')
    returned = [operation for operation in mechanics if operation['phase'] == 'returned']
    kinds = {'loss_forwards': 'loss_forward', 'backwards': 'backward',
             'reference_backwards': 'reference_backward', 'optimizer_steps': 'optimizer_step',
             'checkpoint_saves': 'checkpoint_save', 'checkpoint_restores': 'checkpoint_restore'}
    counts = {name: sum(operation.get('kind') == kind for operation in returned) for name, kind in kinds.items()}
    counts.update(native_sampled_trajectories=len(samples), native_task_scored_trajectories=len(samples),
                  native_parity_positions=len(parity_errors), mechanics_token_fixtures=len(tokens),
                  mechanics_reward_fixtures=len(rewards))
    require(all(counts[name] >= minimum for name, minimum in QUAL_COUNTS.items())
            and type(qualification.get('counts')) is dict
            and all(type(value) is int for value in qualification['counts'].values())
            and qualification.get('counts') == counts, 'derived substantive native/fixture qualification counts')
    require({operation['id'] for operation in operations
             if operation.get('origin') in {'actual_native_sampling', 'actual_task_scoring'}
             and operation['phase'] == 'returned'} == used,
            'native operation population must exactly match audited samples')

    mixed_reference, skipped = False, []
    saved, restores = {}, {}
    for operation in returned:
        kind, inp, output = operation.get('kind'), operation['input'], operation['output']
        if kind == 'checkpoint_save':
            require(hex64(output.get('manifest_sha256')), 'qualification checkpoint save proofs')
            saved.setdefault(output['manifest_sha256'], operation_order[operation['id']])
        elif kind == 'reference_backward':
            reference = output.get('gradient_reference') or {}
            require(reference.get('norm') == metric['norm'] and reference.get('error_metric') == metric['error_metric']
                    and reference.get('implementation_sha256') == metric['implementation_sha256']
                    and reference.get('fixed_loss_denominator') == 16
                    and all(type(reference.get(key)) in (int, float) and math.isfinite(reference[key])
                            and reference[key] >= 0 for key in ('reference_norm', 'difference_norm', 'max_abs_error'))
                    and reference['difference_norm'] >= reference['max_abs_error']
                    and reference['max_abs_error'] <= metric['absolute_tolerance']
                    and (reference['reference_norm'] == 0
                         or reference['difference_norm'] / reference['reference_norm'] <= metric['relative_tolerance']),
                    'registered independent gradient-reference numeric proof')
            for fixture_id in inp['fixture_ids']:
                if fixture_id in reward_map:
                    uniform = [len(set(group)) == 1 for group in reward_map[fixture_id]]
                    mixed_reference |= sum(uniform) == 1
        elif kind == 'optimizer_step':
            require(integer(output.get('U_before')) and output.get('U_after') == output['U_before'] + 1
                    and type(output.get('U_after')) is int and hex64(output.get('step_proof_sha256')),
                    'proved actual Adam update count')
        elif kind == 'uniform_skip':
            before, after = output.get('before') or {}, output.get('after') or {}
            require(integer(before.get('C')) and integer(before.get('U'))
                    and type(before.get('adam_initialized')) is bool
                    and all(hex64(before.get(key)) for key in ('parameters_sha256', 'adam_sha256'))
                    and after == dict(before, C=before['C'] + 1)
                    and any(fixture_id in reward_map and all(len(set(group)) == 1 for group in reward_map[fixture_id])
                            for fixture_id in inp['fixture_ids']), 'observed uniform skip C/U/Adam proof')
            skipped.append(after)
        elif kind == 'checkpoint_restore':
            state = output.get('state_sha256') or {}
            require(output.get('manifest_sha256') in saved
                    and set(state) == {'parameters', 'adam', 'python_rng', 'numpy_rng', 'torch_cpu_rng', 'torch_cuda_rng'}
                    and all(type(pair) is list and len(pair) == 2 and hex64(pair[0]) and pair[0] == pair[1]
                            for pair in state.values()), 'observed parameter/Adam/all-RNG bitwise restore')
            restores[output['manifest_sha256']] = operation_order[operation['id']]
    rollback = False
    for operation in returned:
        if operation.get('kind') == 'rollback_reuse_check':
            output = operation['output']
            old = by_id.get(output.get('returned_optimizer_operation_id')) or {}
            require(old.get('kind') == 'optimizer_step' and old.get('phase') == 'returned'
                    and old.get('origin') == 'declared_mechanics_fixture'
                    and output.get('restored_manifest_sha256') in restores
                    and saved[output['restored_manifest_sha256']] < operation_order[old['id']]
                    < restores[output['restored_manifest_sha256']] < operation_order[operation['id']]
                    and integer(output.get('generation_before'))
                    and integer(output.get('generation_after')) and output['generation_after'] > output['generation_before']
                    and output.get('reuse_outcome') == 'STALE_GENERATION_REJECTED'
                    and integer(output.get('charged_before'), 1) and integer(output.get('charged_after'), 1)
                    and output['charged_after'] >= output['charged_before'], 'observed rollback fence/cost proof')
            rollback = True
    failures = [operation for operation in mechanics if operation['phase'] == 'failed']
    partial = any(operation.get('kind') == 'backward'
                  and integer(operation['input'].get('reserved_work_units'), 1)
                  and integer(operation['output'].get('charged_work_units'), 1)
                  and operation['output']['charged_work_units'] >= operation['input']['reserved_work_units']
                  for operation in failures)
    cases = {'unequal_lengths': len(set(lengths)) > 1, 'eos_included_once': eos_count >= 1,
             'length_cap_without_eos': capped_count >= 1, 'one_uniform_one_mixed': mixed_reference,
             'all_uniform_skips_adam': {state['adam_initialized'] for state in skipped} == {False, True},
             'empty_adam_state': any(not state['adam_initialized'] and state['U'] == 0 for state in skipped),
             'C_greater_than_U': any(state['C'] > state['U'] for state in skipped),
             'rollback_returned_update': rollback, 'partial_step_failure_accounted': partial}
    require(all(cases.values()) and qualification.get('mechanics_cases') == cases
            and all(value is True for value in qualification['mechanics_cases'].values()),
            'derived mechanics fixture case coverage')
    parity = {'tolerance_nats': 1e-4, 'positions': len(parity_errors), 'max_abs_nats': max(parity_errors)}
    require(qualification.get('native_score_parity') == parity, 'derived native per-token score parity')
    seal = bound_json(qualification.get('cost_seal'), 'qualification cost seal')
    require(seal.get('schema') == 'S1_QUALIFICATION_COST_SEAL-1'
            and seal.get('kind') == 'INVOCATION_END' and seal.get('completed') is True
            and seal.get('scope_sha256') == scope_ref['sha256']
            and seal.get('evidence_sha256') == evidence_ref['sha256']
            and seal.get('operations_sha256') == digest(operations) and seal.get('counts') == counts,
            'sealed qualification operation/cost evidence')
    ledger = bound_json(seal.get('cost_ledger'), 'qualification cost ledger')
    require(ledger.get('schema') == 'S1_QUALIFICATION_COST_LEDGER-1'
            and ledger.get('scope_sha256') == scope_ref['sha256']
            and type(ledger.get('entries')) is list and ledger['entries'], 'qualification cost ledger scope/entries')
    previous, started, ended, reserved, attempts = '0' * 64, {}, {}, 0, {}
    start_index, end_index = {}, {}
    terminal = None
    for index, entry in enumerate(ledger['entries']):
        require(type(entry) is dict and type(entry.get('seq')) is int and entry['seq'] == index
                and entry.get('previous_sha256') == previous and entry.get('scope_sha256') == scope_ref['sha256']
                and entry.get('row_sha256') == digest({key: value for key, value in entry.items() if key != 'row_sha256'}),
                'qualification ledger hash chain')
        previous = entry['row_sha256']
        event, identifier = entry.get('event'), entry.get('operation_id')
        if event == 'started':
            operation = by_id.get(identifier) or {}
            require(identifier not in started and operation
                    and entry.get('operation_kind') == operation.get('kind')
                    and entry.get('origin') == operation.get('origin')
                    and entry.get('input_sha256') == digest(operation['input'])
                    and integer(entry.get('reserved_work_units'), 1)
                    and integer(operation['input'].get('reserved_work_units', 1), 1)
                    and integer(operation['output'].get('charged_work_units', 0))
                    and entry['reserved_work_units'] >= operation['input'].get('reserved_work_units', 1)
                    and entry['reserved_work_units'] >= operation['output'].get('charged_work_units', 0),
                    'qualification charged start reservation')
            started[identifier] = entry
            start_index[identifier] = index
            reserved += entry['reserved_work_units']
            attempts[operation['kind']] = attempts.get(operation['kind'], 0) + 1
        elif event in {'returned', 'failed'}:
            operation = by_id.get(identifier) or {}
            require(identifier in started and identifier not in ended and operation.get('phase') == event
                    and entry.get('output_sha256') == digest(operation['output'])
                    and integer(entry.get('known_work_units'))
                    and entry['known_work_units'] <= started[identifier]['reserved_work_units'],
                    'qualification ledger returned/failed operation proof')
            ended[identifier] = event
            end_index[identifier] = index
        else:
            require(event == 'qualification_complete' and index == len(ledger['entries']) - 1,
                    'qualification ledger terminal event')
            terminal = entry
    require(list(started) == [operation['id'] for operation in operations]
            and all((identifier not in ended) == (operation['phase'] == 'unknown')
                    for identifier, operation in by_id.items()), 'qualification all attempts retained, including unknown starts')
    for sample in samples:
        generation = sample['generation_operation_id']
        require(all(end_index[generation] < start_index[sample[field]]
                    for field in ('teacher_operation_id', 'native_scorer_operation_id', 'independent_scorer_operation_id')),
                'qualification generation must return before dependent scoring starts')
    for operation in returned:
        if operation.get('kind') == 'checkpoint_restore':
            manifest = operation['output']['manifest_sha256']
            require(any(save.get('kind') == 'checkpoint_save' and save['output']['manifest_sha256'] == manifest
                        and end_index[save['id']] < start_index[operation['id']] for save in returned),
                    'qualification saved checkpoint must return before restore starts')
        elif operation.get('kind') == 'rollback_reuse_check':
            output = operation['output']
            old_id, manifest = output['returned_optimizer_operation_id'], output['restored_manifest_sha256']
            require(any(restore.get('kind') == 'checkpoint_restore'
                        and restore['output']['manifest_sha256'] == manifest
                        and end_index[old_id] < start_index[restore['id']]
                        and end_index[restore['id']] < start_index[operation['id']]
                        and any(save.get('kind') == 'checkpoint_save'
                                and save['output']['manifest_sha256'] == manifest
                                and end_index[save['id']] < end_index[old_id] for save in returned)
                        for restore in returned), 'qualification rollback returned-work dependency order')
    charged = {'attempts_by_kind': attempts, 'reserved_work_units': reserved,
               'failed_attempts': sum(event == 'failed' for event in ended.values()),
               'unknown_attempts': len(started) - len(ended)}
    require(terminal is not None and terminal.get('evidence_sha256') == evidence_ref['sha256']
            and terminal.get('operations_sha256') == digest(operations)
            and terminal.get('operation_ids') == [operation['id'] for operation in operations]
            and terminal.get('counts') == counts and terminal.get('charged_physical_totals') == charged
            and seal.get('terminal_row_sha256') == terminal['row_sha256'],
            'qualification terminal scope/evidence/IDs/charged physical totals')
    return {'counts': counts, 'mechanics_cases': cases, 'native_score_parity': parity,
            'charged_physical_totals': charged}


def _qualification(packet, prerequisites, planned, census, timings, code_sha256):
    qualification = bound_json(packet.get('qualification'), 'native qualification')
    require(qualification.get('schema') == QUALIFICATION_SCHEMA
            and qualification.get('passed') is True and qualification.get('registered_run') is True
            and qualification.get('test_mode') is False and qualification.get('device') == 'cuda:0'
            and qualification.get('completed') is True and qualification.get('error') is None,
            'complete native CUDA qualification missing')
    require(qualification.get('code_sha256') == code_sha256
            and qualification.get('anchor') == anchor(packet['anchor'])
            and qualification.get('start') == packet['start']
            and qualification.get('inventory') == packet['inventory']
            and qualification.get('update_recipe_sha256') == digest(packet['update_recipe'])
            and qualification.get('sampler_sha256') == digest(packet['sampler'])
            and qualification.get('plan_sha256') == planned.sha256
            and qualification.get('manifest_sha256') == MANIFEST
            and qualification.get('tokenizer_receipt_sha256') == census['tokenizer_receipt_sha256'],
            'qualification source/start/recipe/plan/tokenizer drift')
    adopted = prerequisites['adopted']
    require(qualification.get('runtime_preimage') == adopted['decoding_preimage']
            and qualification.get('decoding_sha256') == adopted['decoding_sha256']
            and qualification.get('scorer_bundle_sha256') == adopted['scoring_code_bundle_sha256'],
            'qualification runtime/scorer drift')
    require(qualification.get('score_parity_passed') is True
            and qualification.get('resume_parameters_adam_rng_bitwise') is True
            and qualification.get('gradient_reference_passed') is True
            and qualification.get('finite_confined_gradients') is True
            and qualification.get('frozen_state_unchanged') is True, 'score/gradient/checkpoint acceptance missing')
    validate_qualification_evidence(qualification, packet, planned)
    require(qualification.get('timing_prices') == timings['prices'], 'timing prices differ from measured qualification')
    fit = qualification.get('resources') or {}
    required = ('peak_worker_rss_gib', 'checkpoint_delta_peak_worker_rss_gib', 'gpu_total_gib',
                'peak_reserved_gib', 'min_gpu_free_gib', 'min_disk_free_bytes')
    require(all(positive(fit.get(key)) for key in required), 'measured resource fit missing')
    require(fit.get('memory_fraction') == .78 and fit.get('reserved_fraction_limit') == .95
            and fit['peak_reserved_gib'] <= .95 * .78 * fit['gpu_total_gib']
            and fit['min_gpu_free_gib'] >= .5, 'qualified VRAM envelope')
    floor = max(adopted['host_floor_gib'], fit['peak_worker_rss_gib'] + 1,
                fit['checkpoint_delta_peak_worker_rss_gib'] + 1)
    return qualification, floor


def validate_current_runtime(packet, current_runtime_preimage, *, io_observer=None):
    """Phase two: called after framework configuration, BEFORE any model construction.

    A frozen historical preimage must never be passed off as a fresh observation.
    The runner is responsible for collecting the actual configured preimage.
    """
    token = _IO_OBSERVER.set(io_observer) if io_observer is not None else None
    try:
        adopted = bound_json(packet['prerequisites']['adopted_warm_freeze'], 'adopted warm freeze')
        require(_preimage(current_runtime_preimage) == adopted['decoding_sha256']
                and current_runtime_preimage == adopted['decoding_preimage'], 'current runtime preimage not bound')
    finally:
        if token is not None:
            _IO_OBSERVER.reset(token)
    return current_runtime_preimage


def _admit_metadata(packet, *, code_sha256, current_runtime_preimage=None):
    """Phase-one METADATA admission, before optional framework-only initialization.

    This is not a model-load permit by itself: the production runner must then
    call validate_current_runtime on the actual configured framework/CUDA state,
    check fresh resource admission and charge all work before constructing a model.
    The optional callback is for framework initialization, never model loading.
    """
    require(type(packet) is dict and packet.get('schema') == SCHEMA, 'native packet schema')
    require(packet.get('approved') is True and packet.get('training_authorized') is True,
            'packet is unapproved; no framework import')
    require(packet.get('status') == 'APPROVED_FOR_NATIVE_TRAINING', 'packet status')
    require(packet.get('mode') == 'production' and packet.get('lease_timeout_s') == 900
            and packet.get('checkpoint_cadence') == 10, 'production mode/lease/cadence contract')
    authorization = bound_json(packet.get('authorization'), 'operator authorization')
    require(authorization.get('approved') is True and authorization.get('training_authorized') is True
            and authorization.get('packet_preimage_sha256') == digest({k: v for k, v in packet.items()
                                                                       if k != 'authorization'}),
            'authorization does not name this exact packet')
    require(type(code_sha256) is dict and code_sha256
            and all(type(name) is str and name and hex64(value) for name, value in code_sha256.items())
            and packet.get('code_sha256') == code_sha256, 'source bundle drift')
    require(packet.get('manifest_sha256') == MANIFEST and packet.get('arm') in ARMS, 'data/arm scope')
    require(packet['arm'] not in {'R_to_S', 'S_to_R'}, 'sequential first-donor/H1/start integration not implemented')
    validate_update_recipe(packet.get('update_recipe'), synthetic_cpu=False)
    operation_fields = {'boundary_id', 'generation', 'symbolic_snapshot', 'start_policy_id', 'trainable_names'}
    require(not operation_fields & set(packet['update_recipe']), 'operation fields cannot enter packet recipe preimage')
    proposal = bound_json(packet.get('proposal'), 'recipe proposal')
    require(proposal.get('schema') == 'COMP-RS-001-S1-RECIPE-PROPOSAL-2'
            and proposal.get('version') == 2, 'proposal schema/version')
    prerequisites = validate_prerequisites(packet)
    expected_start = {'kind': 'warm_anchor', 'arm_sha256': anchor(packet['anchor'])['arm_sha256']}
    require(packet.get('start') == expected_start, 'exact independent warm start required')
    require(packet.get('sampler') == SAMPLER, 'fixed raw-policy sampler contract')
    if current_runtime_preimage is not None:
        validate_current_runtime(packet, current_runtime_preimage)
    output, checkout = packet.get('output_dir'), packet.get('source_checkout')
    require(type(output) is str and Path(output).is_absolute()
            and type(checkout) is str and Path(checkout).is_absolute(), 'absolute owned output/checkout paths')
    target = Path(output).resolve()
    protected = [Path(checkout).resolve(), Path(packet['data_manifest']['path']).resolve().parent,
                 Path(prerequisites['assets']['model_path']).resolve()]
    require(all(target != root and not target.is_relative_to(root) and not root.is_relative_to(target)
                for root in protected), 'output overlaps checkout/data/model source')
    planned, caps, census, timings = _planning(packet, proposal, prerequisites)
    limits = packet.get('limits')
    require(type(limits) is dict and set(limits) >= set(LIMITS), 'hard ceilings missing')
    require(all(positive(limits[key]) for key in LIMITS), 'unresolved/nonfinite ceilings')
    require(all(integer(limits[key], 1) for key in LIMITS if key not in {'wall_seconds', 'host_floor_gib'}),
            'integer compute ceilings')
    mapping = {'generation_attempts': 'rollouts_started', 'generated_token_charge': 'generated_tokens_charged',
               'prefill_token_charge': 'prefill_tokens_charged', 'forward_attempts': 'loss_forward_started',
               'backward_attempts': 'backward_started', 'optimizer_attempts': 'optimizer_started',
               'checkpoint_attempts': 'checkpoint_started', 'invocations': 'invocations_started', 'wall_seconds': 'wall_charged_s'}
    require(all(limits[name] == caps[key] for name, key in mapping.items()), 'hard ceilings differ from exact plan/census/prices')
    qualification, floor = _qualification(packet, prerequisites, planned, census, timings, code_sha256)
    require(limits['host_floor_gib'] >= floor, 'host floor weakened')
    require(limits.get('resumes') == caps['resumes'] and type(limits.get('resumes')) is int,
            'resume ceiling drift')
    require(type(limits.get('active_metadata_reserve_bytes')) is int
            and limits['active_metadata_reserve_bytes'] == caps['active_metadata_reserve_bytes']
            and limits.get('active_storage_reserve_bytes') == caps['active_storage_reserve_bytes']
            and limits.get('disk_floor_bytes') == 14 * 1024 ** 3, 'disk floor/reservation drift')
    require(qualification['resources']['min_disk_free_bytes'] >= limits['disk_floor_bytes'] + caps['active_storage_reserve_bytes'], 'qualified disk envelope')
    require(all(integer(limits.get(name), 1) for name in IO_LIMITS), 'explicit integer hash/IO/load/restore ceilings')
    io = bound_json(packet.get('io_budget'), 'measured IO budget')
    require(io.get('schema') == 'S1_IO_BUDGET-1' and io.get('measured') is True
            and io.get('plan_sha256') == planned.sha256 and io.get('inventory') == packet['inventory']
            and io.get('qualification_sha256') == packet['qualification']['sha256'], 'IO budget measurement/plan binding')
    require(io.get('caps') == {name: limits[name] for name in IO_LIMITS}, 'IO ceilings differ from approved measured budget')
    return packet


def admit_packet(packet, *, code_sha256, framework_loader=None, current_runtime_preimage=None,
                 io_observer=None):
    """Validate metadata before a framework-only callback; runtime gate is separate.

    Malformed/read-failed packet evidence fails closed as AdmissionRejected.
    Callback exceptions retain their original class: they are execution failures,
    not mislabeled malformed-packet errors.
    """
    token = _IO_OBSERVER.set(io_observer)
    try:
        try:
            admitted = _admit_metadata(packet, code_sha256=code_sha256,
                                       current_runtime_preimage=current_runtime_preimage)
        except AdmissionRejected:
            raise
        except (KeyError, TypeError, IndexError, OSError, ValueError) as error:
            raise AdmissionRejected('malformed or unreadable packet evidence: ' + str(error)) from error
    finally:
        _IO_OBSERVER.reset(token)
    return framework_loader() if framework_loader is not None else admitted


def admitted_context(packet, *, io_observer=None):
    """Materialize bound metadata after separate admission; never grants execution.

    Every repeated file read/hash is observed. Runtime/resource/authorization
    checks remain the runner's two-phase admission responsibility.
    """
    token = _IO_OBSERVER.set(io_observer)
    try:
        require(type(packet) is dict and packet.get('schema') == SCHEMA, 'context packet schema')
        proposal = bound_json(packet.get('proposal'), 'recipe proposal')
        require(proposal.get('schema') == 'COMP-RS-001-S1-RECIPE-PROPOSAL-2'
                and proposal.get('version') == 2, 'context proposal schema/version')
        prerequisites = validate_prerequisites(packet)
        planned, caps, census, timings = _planning(packet, proposal, prerequisites)
        io = bound_json(packet.get('io_budget'), 'measured IO budget')
        require(io.get('schema') == 'S1_IO_BUDGET-1' and io.get('measured') is True
                and io.get('plan_sha256') == planned.sha256
                and io.get('qualification_sha256') == packet['qualification']['sha256']
                and io.get('inventory') == packet['inventory'], 'context IO budget binding')
        return {'prerequisites': prerequisites, 'proposal': proposal, 'planned': planned,
                'caps': caps, 'census': census, 'timings': timings, 'io': io,
                'execution_authorized_by_context': False}
    except AdmissionRejected:
        raise
    except (KeyError, TypeError, IndexError, OSError, ValueError) as error:
        raise AdmissionRejected('malformed or unreadable context evidence: ' + str(error)) from error
    finally:
        _IO_OBSERVER.reset(token)
