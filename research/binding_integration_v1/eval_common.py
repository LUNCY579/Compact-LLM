"""Gold-free auxiliary inference contracts; the original decoder is reused verbatim."""
from collections import Counter
from contextlib import contextmanager
import importlib.metadata
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import uuid

if __package__:
    from .paths import BASE, HERE, OUT, PILOT, RABPY, BASELINE, sha, read, write
else:
    from paths import BASE, HERE, OUT, PILOT, RABPY, BASELINE, sha, read, write

CODE_ROOT = BASE / 'ChatBubble-dose-param'
sys.path.insert(0, str(CODE_ROOT))
from research.binding_signal_eval_v1 import common as original

NUMERICS = dict(original.NUMERICS)
KINDS = {'binding': (384, {12: 32}), 'diagnostic': (896, {16: 32, 4: 96})}
LIMITS = {'workers': 3, 'worker_seconds': 900, 'total_worker_seconds': 2700,
          'rows': 2176, 'reserved_generated_tokens': 52224, 'forward_calls': 52224,
          'processed_input_tokens': 500000, 'retries': 1}
WORKER_FILES = ('eval_common.py', 'eval_probe.py', 'paths.py', 'run_probe.py', 'train.py', 'adapters.py')
CORE_FILES = {name: CODE_ROOT / 'research/binding_signal_eval_v1' / name for name in ('common.py', 'probe.py')}
GRANT_KEYS = {'schema', 'record', 'experiment', 'decision', 'authorization_reference',
    'arm_path', 'arm_weight_sha256', 'arm_receipts_sha256', 'source_dir', 'sidecars_sha256',
    'build_gate_path', 'build_gate_sha256', 'data_use_admission_sha256', 'datasets',
    'code_sha256', 'core_code_sha256', 'pilot_code_sha256', 'runtime_versions',
    'python_path', 'output_root', 'limits', 'deadline_unix'}


def require(ok, message):
    if not ok:
        raise ValueError(message)


def context(grant_path, gate_path):
    grant, gate = read(grant_path), read(gate_path)
    require(set(grant) == GRANT_KEYS and type(grant['schema']) is int and grant['schema'] == 1 and
        all(type(value) is int for value in grant['limits'].values()) and str(gate.get('status', '')).startswith('PASS') and
        grant['record'] == 'BINDING_INTEGRATION_AUX_GRANT' and grant['decision'] == 'APPROVED' and
        grant['experiment'] == 'BINDING-INTEGRATION-001' and grant['limits'] == LIMITS and
        grant['build_gate_sha256'] == sha(gate_path) and Path(grant['build_gate_path']).resolve() == Path(gate_path).resolve(),
        'explicit auxiliary grant/gate mismatch')
    require(grant['deadline_unix'] == read(OUT / 'SESSION_ENVELOPE.json')['deadline_unix'] and
            Path(grant['python_path']).resolve() == RABPY.resolve(), 'auxiliary deadline or interpreter changed')
    baseline = read(BASELINE / 'BASELINE_PLAN.json')
    require(grant['runtime_versions'] == baseline['runtime_versions'] and
            Path(grant['source_dir']).resolve() == Path(gate['source']).resolve() == Path(baseline['source_dir']).resolve() and
            grant['sidecars_sha256'] == baseline['sidecars_sha256'] and
            set(grant['arm_receipts_sha256']) == {'ARM.json', 'MATERIALIZATION.json'} and
            grant['core_code_sha256'] == baseline['code_sha256'] and
            {name: sha(path) for name, path in CORE_FILES.items()} == grant['core_code_sha256'],
            'qualified original decoding core changed')
    require(set(grant['code_sha256']) == set(WORKER_FILES) and
            {name: sha(HERE / name) for name in WORKER_FILES} == grant['code_sha256'], 'auxiliary execution code changed')
    require(original.pilot_code_files(PILOT) == grant['pilot_code_sha256'] == baseline['pilot_code_sha256'],
            'qualified backend/supervisor code changed')
    require(set(grant['datasets']) == set(KINDS), 'auxiliary datasets changed')
    for kind, (n, parents) in KINDS.items():
        value = grant['datasets'][kind]
        require(set(value) == {'input_path', 'input_sha256', 'rows', 'parent_counts'} and
                value['rows'] == n and value['parent_counts'] == {str(k): v for k, v in parents.items()},
                'dataset geometry or gold-free grant schema differs')
    require(Path(grant['arm_path']).resolve() == (OUT / 'arms/integrated').resolve() and
            Path(grant['output_root']).resolve() == (OUT / 'eval').resolve(), 'auxiliary model/output roots differ')
    admission = read(OUT / 'DATA_USE_ADMISSION.json')
    require(grant['data_use_admission_sha256'] == sha(OUT / 'DATA_USE_ADMISSION.json') == gate['data_use_admission_sha256'] and
            admission['status'] == 'ADMITTED_FOR_NAMED_TRAIN_DATA_AND_EXPLORATORY_EVALUATION_SCOPE' and
            admission['experiment'] == 'BINDING-INTEGRATION-001', 'fresh data-use admission differs')
    references = {
        'binding': admission['evaluation_scope']['legacy_binding384']['input_gold_and_baseline_report_pins']['DEV_INPUTS.jsonl'],
        'diagnostic': admission['evaluation_scope']['controlled_diagnostic896']['input_gold_and_report_pins']['candidate-v1/DIAGNOSTIC_INPUTS.jsonl'],
    }
    for kind, spec in references.items():
        require(Path(grant['datasets'][kind]['input_path']).resolve() == Path(spec['path']).resolve() and
                grant['datasets'][kind]['input_sha256'] == spec['sha256'], 'auxiliary grant names an unadmitted dataset')
    return grant, gate


def load_inputs(grant, kind):
    spec = grant['datasets'][kind]
    path = Path(spec['input_path'])
    require(sha(path) == spec['input_sha256'], 'registered input bytes changed')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    require(len(rows) == spec['rows'] and len({row['id'] for row in rows}) == len(rows), 'input identity inventory differs')
    parents = Counter()
    for row in rows:
        require(set(row) == {'id', 'parent_id', 'prompt', 'prompt_ids'}, 'gold/extra fields in worker input')
        require(isinstance(row['id'], str) and isinstance(row['parent_id'], str), 'invalid join identity')
        prompt = row['prompt']
        require(isinstance(prompt, list) and len(prompt) == 2 and
                [value['role'] for value in prompt] == ['system', 'user'] and
                all(set(value) == {'role', 'content'} and isinstance(value['content'], str) for value in prompt),
                'only system/user policy text is allowed')
        ids = row['prompt_ids']
        require(isinstance(ids, list) and ids and len(ids) + 24 <= 2048 and
                all(type(token) is int and 0 <= token < 151936 for token in ids), 'invalid or truncating input tokens')
        parents[row['parent_id']] += 1
    require(Counter(parents.values()) == KINDS[kind][1], 'correlated parent geometry changed')
    return rows


def slot(kind, attempt):
    require(kind in KINDS and type(attempt) is int and attempt in (1, 2), 'unregistered auxiliary slot')
    return 'integrated-' + kind + ('' if attempt == 1 else '-retry1')


def phase_guard(grant, reserve=0):
    require(time.time() + reserve <= grant['deadline_unix'], 'auxiliary stage exceeds registered deadline')
    envelope = read(OUT / 'SESSION_ENVELOPE.json')
    require(shutil.disk_usage(OUT).free >= envelope['free_disk_floor_bytes'], 'unchanged disk floor refused')
    require(sum(path.stat().st_size for path in OUT.rglob('*') if path.is_file()) <= envelope['new_records_disk_bytes'],
            'owned logical disk budget exceeded')


def global_costs():
    """Read persistent reservations made by the parent six-leg controller."""
    envelope = read(OUT / 'SESSION_ENVELOPE.json')
    specifications = {'dose': (384, 128), 'select': (384, 128), 'retention': (150, 24),
                      'long_context': (256, 24), 'binding': (384, 24), 'diagnostic': (896, 24)}
    total = {'attempts': 0, 'worker_seconds': 0.0, 'rows': 0, 'generation_tokens': 0, 'retries': 0}
    for path in sorted((OUT / 'posttrain/reservations').glob('*.json')):
        value = read(path); suffix = value['suffix']; attempt = value['attempt']
        require(suffix in specifications and attempt in (1, 2), 'unregistered global probe reservation')
        name = 'integrated-' + suffix + ('' if attempt == 1 else '-retry1')
        n, max_new = specifications[suffix]
        directory = OUT / 'eval' / name
        require(path.name == name + '.json' and Path(value['out']).resolve() == directory.resolve() and
                value['reserved_rows'] == n and value['reserved_generation_tokens'] == n * max_new and
                value['reserved_worker_seconds'] == 900 and value['session_envelope_sha256'] == sha(OUT / 'SESSION_ENVELOPE.json'),
                'global reservation geometry differs')
        supervisor_path = directory / ('supervisor.json' if suffix in KINDS else 'worker/supervisor.json')
        charged = 900.0
        if supervisor_path.exists():
            supervisor = read(supervisor_path); wall = supervisor.get('wall_seconds')
            if (supervisor.get('status') in ('EXITED', 'TIMEOUT_OR_INTERRUPTED') and type(supervisor.get('returncode')) is int and
                    type(wall) in (int, float) and math.isfinite(wall) and wall >= 0):
                charged = float(wall)
        total['attempts'] += 1; total['retries'] += attempt - 1; total['worker_seconds'] += charged
        total['rows'] += n; total['generation_tokens'] += n * max_new
    require(total['attempts'] <= envelope['probe_workers_all_attempts'] and
            total['worker_seconds'] <= envelope['probe_worker_seconds_all_attempts'] and
            total['rows'] <= envelope['probe_rows_reserved_all_attempts'] and
            total['generation_tokens'] <= envelope['probe_generation_tokens_reserved_all_attempts'] and
            total['retries'] <= envelope['probe_retries_total'], 'global all-attempt probe budget exceeded')
    return total


def snapshot(grant, kind):
    arm = Path(grant['arm_path'])
    source = Path(grant['source_dir'])
    require(original.runtime_versions() == grant['runtime_versions'], 'pinned auxiliary runtime changed')
    model_sha = sha(arm / 'model.safetensors')
    require(model_sha == grant['arm_weight_sha256'], 'auxiliary model bytes changed')
    sidecars = {name: sha(arm / name) for name in original.SIDECARS}
    require(sidecars == grant['sidecars_sha256'] == {name: sha(source / name) for name in original.SIDECARS},
            'model/source tokenizer or configuration changed')
    receipts = {name: sha(arm / name) for name in ('ARM.json', 'MATERIALIZATION.json')}
    require(receipts == grant['arm_receipts_sha256'], 'qualified exported model provenance changed')
    spec = grant['datasets'][kind]
    require(sha(spec['input_path']) == spec['input_sha256'], 'auxiliary input bytes changed')
    return {'kind': kind, 'input_sha256': spec['input_sha256'], 'weight_sha256': model_sha,
            'sidecars_sha256': sidecars, 'arm_receipts_sha256': receipts,
            'runtime_versions': original.runtime_versions(),
            'worker_code_sha256': {name: sha(HERE / name) for name in WORKER_FILES},
            'generation_core_sha256': {name: sha(path) for name, path in CORE_FILES.items()},
            'pilot_code_sha256': original.pilot_code_files(PILOT)}


def costs(grant, grant_path):
    totals = {'workers': 0, 'total_worker_seconds': 0.0, 'rows': 0, 'reserved_generated_tokens': 0,
              'forward_calls': 0, 'processed_input_tokens': 0, 'retries': 0}
    attempts = []
    for path in sorted((OUT / 'eval').glob('AUX_RESERVATION_*.json')):
        value = read(path); kind, attempt = value['kind'], value['attempt']
        name = slot(kind, attempt); n = KINDS[kind][0]
        require(path.name == 'AUX_RESERVATION_' + name + '.json' and value['grant_sha256'] == sha(grant_path) and
                value['rows'] == n and value['reserved_generated_tokens'] == value['forward_calls'] == n * 24,
                'auxiliary reservation mismatch')
        charged = 900.0
        supervisor_path = OUT / 'eval' / name / 'supervisor.json'
        if supervisor_path.exists():
            supervisor = read(supervisor_path); wall = supervisor.get('wall_seconds')
            if (supervisor.get('status') in ('EXITED', 'TIMEOUT_OR_INTERRUPTED') and
                    type(supervisor.get('returncode')) is int and type(wall) in (int, float) and math.isfinite(wall) and wall >= 0):
                charged = float(wall)
        totals['workers'] += 1; totals['retries'] += attempt - 1; totals['total_worker_seconds'] += charged
        for key in ('rows', 'reserved_generated_tokens', 'forward_calls', 'processed_input_tokens'):
            require(type(value[key]) is int and value[key] >= 0, 'invalid auxiliary reservation amount')
            totals[key] += value[key]
        attempts.append({'slot': name, 'worker_seconds_charged': charged, 'reservation_sha256': sha(path)})
    require(all(value <= LIMITS[key] for key, value in totals.items()), 'auxiliary subcap exceeded')
    return {'totals': totals, 'attempts': attempts}


@contextmanager
def controller_lease(grant_path):
    root = OUT / 'eval'; root.mkdir(exist_ok=True)
    path = root / 'AUX_CONTROLLER_LEASE.json'
    owner = {'owner_pid': os.getpid(), 'nonce': str(uuid.uuid4()), 'grant_sha256': sha(grant_path)}
    write(path, owner)
    try:
        yield owner
    finally:
        require(read(path) == owner, 'auxiliary lease owner changed; preserve the lease')
        path.unlink()


def verify_launch(out, grant_path, gate_path, kind, attempt, supervisor):
    launch, spawn, claim = (read(out / name) for name in ('launch.json', 'spawn.json', 'worker_claim.json'))
    argv = launch['argv']
    def option(name):
        require(argv.count(name) == 1, 'duplicate/missing worker option')
        return argv[argv.index(name) + 1]
    require(Path(launch['cwd']).resolve() == PILOT.resolve() and Path(launch['out']).resolve() == out.resolve() and
            launch['timeout_seconds'] == 900 and Path(argv[0]).resolve() == RABPY.resolve() and
            Path(argv[1]).resolve() == (HERE / 'eval_probe.py').resolve() and
            Path(option('--grant')).resolve() == Path(grant_path).resolve() and
            Path(option('--build-gate')).resolve() == Path(gate_path).resolve() and
            Path(option('--out')).resolve() == out.resolve() and option('--kind') == kind and
            option('--attempt') == str(attempt) and argv.count('--_worker') == argv.count('--confirm-model-execution') == 1,
            'auxiliary supervised launch mismatch')
    require(spawn['launch_sha256'] == claim['launch_sha256'] == original.digest(launch) and
            launch['supervisor_pid'] == spawn['supervisor_pid'] == claim['supervisor_pid'] and
            spawn['child_pid'] == supervisor['child_pid'] and
            (claim['pid'] == spawn['child_pid'] or (claim.get('launcher_redirected') is True and claim['parent_pid'] == spawn['child_pid'])),
            'auxiliary worker ownership mismatch')
    return {name + '_sha256': sha(out / (name + '.json')) for name in ('launch', 'spawn', 'worker_claim')}


def verify_completed(root, kind):
    """Return generations/custody only after every completed-leg identity passes."""
    require(Path(root).resolve() == OUT.resolve() and kind in KINDS, 'unregistered auxiliary report root/kind')
    grant_path, gate_path = OUT / 'AUX_PROBE_GRANT.json', OUT / 'BUILD_GATE.json'
    out = OUT / 'eval' / slot(kind, 1)
    reservation_path = OUT / 'eval' / ('AUX_RESERVATION_' + out.name + '.json')
    parent_path = OUT / 'posttrain/reservations' / (out.name + '.json')
    readset = [grant_path, gate_path, reservation_path, parent_path, *[out / name for name in
        ('AUX_BINDING.json', 'GENERATIONS.json', 'PRECHECK.json', 'supervisor.json', 'worker_terminal.json',
         'launch.json', 'spawn.json', 'worker_claim.json')]]
    initial_pins = {str(path): sha(path) for path in readset}
    grant, gate = context(grant_path, gate_path)
    require(not (OUT / 'eval/AUX_CONTROLLER_LEASE.json').exists(), 'auxiliary controller ownership is unsettled')
    require(not (OUT / 'eval' / slot(kind, 2)).exists(), 'retry reporting requires a separately verified recovery')
    receipt = read(out / 'AUX_BINDING.json')
    require(receipt['status'] == 'COMPLETE' and receipt['complete'] is True and receipt['kind'] == kind and
            receipt['attempt'] == 1 and receipt['controller_returncode'] == receipt['worker_returncode'] == 0 and
            receipt['grant_sha256'] == sha(grant_path) and receipt['build_gate_sha256'] == sha(gate_path) and
            receipt['failure'] is None and not (out / 'FAILURE.json').exists(), 'successful auxiliary receipt absent')
    for field, name in (('supervisor', 'supervisor.json'), ('worker_terminal', 'worker_terminal.json'), ('generations', 'GENERATIONS.json')):
        require(receipt[field + '_sha256'] == sha(out / name), 'auxiliary bound file changed')
    supervisor, terminal, bundle = (read(out / name) for name in ('supervisor.json', 'worker_terminal.json', 'GENERATIONS.json'))
    original.verify_terminal(supervisor, terminal)
    identities = verify_launch(out, grant_path, gate_path, kind, 1, supervisor)
    require(all(receipt[key] == value for key, value in identities.items()), 'auxiliary launch identity changed')
    require(terminal['kind'] == bundle['kind'] == kind and terminal['grant_sha256'] == bundle['grant_sha256'] == sha(grant_path) and
            terminal['build_gate_sha256'] == bundle['build_gate_sha256'] == sha(gate_path) and
            terminal['generations_sha256'] == sha(out / 'GENERATIONS.json'), 'auxiliary terminal/grant mismatch')
    current = snapshot(grant, kind)
    require(receipt['before'] == receipt['after'] == terminal['before'] == terminal['after'] ==
            bundle['before'] == bundle['after'] == current, 'auxiliary input/code/runtime/model custody differs')
    if __package__:
        from . import run_probe
    else:
        import run_probe
    training = run_probe.arm_snapshot(Path(grant['arm_path']), gate)
    require(receipt['training_before'] == receipt['training_after'] == training and
            receipt['arm_receipt_sha256'] == sha(Path(grant['arm_path']) / 'ARM.json'), 'auxiliary training qualification/provenance changed')
    rows = load_inputs(grant, kind)
    reservation_path = OUT / 'eval' / ('AUX_RESERVATION_' + out.name + '.json')
    reservation = read(reservation_path)
    require(receipt['reservation_sha256'] == sha(reservation_path) and reservation['kind'] == kind and reservation['attempt'] == 1 and
            reservation['grant_sha256'] == sha(grant_path) and reservation['rows'] == len(rows) and
            reservation['forward_calls'] == reservation['reserved_generated_tokens'] == len(rows) * 24 and
            reservation['processed_input_tokens'] == sum(len(row['prompt_ids']) for row in rows) + len(rows) * 24,
            'auxiliary reservation differs')
    parent_path = OUT / 'posttrain/reservations' / (out.name + '.json')
    require(reservation['parent_reservation_sha256'] == sha(parent_path), 'auxiliary parent reservation changed')
    owner = reservation['controller_owner']
    owner_bytes = (json.dumps(owner, indent=2, allow_nan=False) + '\n').encode()
    require(hashlib.sha256(owner_bytes).hexdigest() == reservation['controller_lease_sha256'] and
            owner['owner_pid'] == read(out / 'launch.json')['supervisor_pid'] and owner['grant_sha256'] == sha(grant_path),
            'auxiliary controller lease identity differs')
    require(bundle['record'] == 'BINDING_INTEGRATION_AUX_GENERATIONS' and bundle['rows'] == len(rows) and
            bundle['numerics'] == NUMERICS and bundle['gold_file_read'] is False and bundle['training_permission'] is False and
            bundle['extra_warmup_or_qualification_calls'] == 0, 'auxiliary numerical/scope contract differs')
    precheck = read(out / 'PRECHECK.json')
    require(precheck['status'] == 'PRECHECK_PASSED' and precheck['kind'] == kind and precheck['before'] == current and
            precheck['numerics'] == NUMERICS and precheck['gold_file_read'] is False and precheck['model_input_fields'] == ['prompt_ids'],
            'auxiliary precheck differs')
    # The qualified diagnostic decoder is stdlib-only; no model or tokenizer
    # runtime is imported to verify saved token strings and accounting.
    from research.binding_diagnostic_v1.score import ByteLevelDecoder, expected_eos
    source = Path(grant['source_dir']); tokenizer = read(source / 'tokenizer.json'); config = read(source / 'tokenizer_config.json')
    decoder = ByteLevelDecoder(tokenizer, config); eos = expected_eos(source, tokenizer, config)
    require(bundle['eos_ids'] == eos and len(bundle['predictions']) == len(rows), 'auxiliary EOS/row inventory changed')
    for row, predicted in zip(rows, bundle['predictions']):
        ids = predicted['generated_ids_including_eos']
        require(predicted['id'] == row['id'] and predicted['parent_id'] == row['parent_id'] and
                type(predicted['terminated']) is bool and type(predicted['input_tokens']) is int and
                predicted['input_tokens'] == len(row['prompt_ids']) and isinstance(ids, list) and 1 <= len(ids) <= 24 and
                all(type(token) is int and 0 <= token < 151936 for token in ids) and
                predicted['terminated'] == (ids[-1] in eos) and not any(token in eos for token in ids[:-1]) and
                (predicted['terminated'] or len(ids) == 24), 'auxiliary generation identity/bounds changed')
        require(predicted['prediction'] == decoder.decode(ids) and
                predicted['generated_tokens'] == predicted['forward_calls'] == len(ids) and
                predicted['processed_input_tokens'] == len(row['prompt_ids']) + len(ids) - 1,
                'auxiliary decoding or token ledger differs')
    actual = {'rows': len(rows), 'generated_tokens_including_eos': sum(p['generated_tokens'] for p in bundle['predictions']),
              'forward_calls': sum(p['forward_calls'] for p in bundle['predictions']),
              'processed_input_tokens': sum(p['processed_input_tokens'] for p in bundle['predictions'])}
    require(actual == bundle['actual_budget_charges'] and actual['generated_tokens_including_eos'] <= reservation['reserved_generated_tokens'] and
            actual['forward_calls'] <= reservation['forward_calls'] and actual['processed_input_tokens'] <= reservation['processed_input_tokens'],
            'auxiliary actual charges differ from their reservation')
    phase_guard(grant); local = costs(grant, grant_path); global_ = global_costs()
    require(all(sha(path) == expected for path, expected in initial_pins.items()) and
            snapshot(grant, kind) == current and run_probe.arm_snapshot(Path(grant['arm_path']), gate) == training,
            'validated auxiliary evidence changed during verification')
    return bundle, {'directory': str(out), 'kind': kind, 'binding_sha256': sha(out / 'AUX_BINDING.json'),
        'generations_sha256': sha(out / 'GENERATIONS.json'), 'grant_sha256': sha(grant_path),
        'files_sha256': initial_pins, 'training_snapshot': training, 'actual_budget_charges': actual,
        'worker_wall_seconds': supervisor['wall_seconds'], 'auxiliary_subcap_accounting': local,
        'global_reservation_accounting': global_, 'gold_file_read_by_worker': False}
