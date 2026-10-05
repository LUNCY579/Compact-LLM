"""Explicit CPU native-schema filesystem fixture; never a native receipt."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import hashlib, importlib.util, json
HERE=Path(__file__).resolve().parent

def source_module(name):
    path=HERE/name
    spec=importlib.util.spec_from_file_location('native_schema_fixture_'+path.stem,path)
    value=importlib.util.module_from_spec(spec)
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),value.__dict__)
    return value

def canonical(v):return json.dumps(v,ensure_ascii=True,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(v):return hashlib.sha256(canonical(v).encode('ascii')).hexdigest()
def seal(v):
    body={k:x for k,x in v.items() if k!='sha256'}
    return {**body,'sha256':digest(body)}

def native_frame(base):
    self=SimpleNamespace(root=Path(base),fixtures=source_module('fixture_support.py'),auditor=source_module('prefix_input_owned_auditor.py'))
    root = self.root / 'CPU_NATIVE_SCHEMA_ORCHESTRATION_TEST'
    root.mkdir()
    frame = self.fixtures.terminal_frame(root)
    original, preserved = (frame['raw'], frame['preserved'])
    cpu_terminal = frame['terminal']
    owner = dict(cpu_terminal['worker_identity'])
    times = {row['event']: row['monotonic'] for row in cpu_terminal['timeline']}
    native_scope = self.auditor.NATIVE_SCOPE
    native_status = 'NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT'
    manifest = {'scope': native_scope, 'permission': self.auditor.PERMISSION, 'resources': self.auditor.RESOURCES, 'source_files': frame['manifest']['source_pins'], 'runtime_files': {}, 'input_file_sha256': self.auditor.INPUT_FILE, 'input_body_sha256': self.auditor.INPUT_BODY, 'prefix_ids_sha256': self.auditor.PREFIX_SHA}
    manifest = seal(manifest)
    anchors = {'config_path': str(root / 'CPU_METADATA_NATIVE_CONFIG.json'), 'anchors': {'expected_config_file': 'b' * 64}}
    
    def pair(name, value):
        raw = (canonical(value) + '\n').encode('ascii')
        for destination in (original, preserved):
            (destination / name).write_bytes(raw)
    worker = json.loads((original / 'worker_result.json').read_text())
    worker.update(scope=native_scope, status=native_status)
    pair('worker_result.json', worker)
    pair('ownership_verified.json', {'scope': native_scope})
    final = json.loads((original / 'PHASE_ACCOUNTING_FINAL.json').read_text())
    final.update(scope=native_scope, status=native_status)
    pair('PHASE_ACCOUNTING_FINAL.json', final)
    report = deepcopy(cpu_terminal['worker_report'])
    report.update(scope=native_scope, status=native_status, final_phase_summary=final)
    modules = {'CPU_METADATA_MODULE': {'path': next(iter(manifest['source_files'])), 'sha256': next(iter(manifest['source_files'].values()))['sha256']}}
    pair('OWNED_WORKER_RETURN.json', {'report': report, 'runtime_execution_modules': modules, 'owned_membership_verified': {'actual_worker_pid': owner['pid'], 'actual_parent_pid': 12344, 'launcher_root_pid': owner['pid'], 'supervisor_pid': 12344, 'job_name': 'CPU_METADATA_JOB', 'owned_job_membership_verified': True, 'query_job_handle_closed_before_framework_import': True}, 'IO': {'owned_return_write': {'attempted': 1, 'returned': 1, 'failed': 0, 'pending': 0}}, 'native_PASS': False})
    pair('OWNED_EXTERNAL_ANCHORS.json', anchors)
    
    def observation(value):
        return {'sample_started_monotonic': value['monotonic'], 'sample_completed_monotonic': value['monotonic'], 'host_available_bytes': value['available_bytes'], 'worker_tree_rss_bytes': value['worker_RSS_bytes'], 'disk_free_bytes': value['disk_free_bytes'], 'gpu_sample_started_monotonic': value['GPU_monotonic'], 'gpu_sample_completed_monotonic': value['GPU_monotonic'], 'gpu_free_bytes': value['free_bytes'], 'gpu_total_bytes': value['total_bytes'], 'GPU_index': 0, 'gpu_uuid': value['GPU_identity'], 'gpu_observation_id': str(value['GPU_monotonic'])}
    admissions = [observation(value) for value in cpu_terminal['admission_observations']]
    full_owner = {**owner, 'assigned_before_resume': True, 'job_kill_on_close': True, 'breakaway_allowed': False, 'job_name': 'CPU_METADATA_JOB'}
    lease_root = root / 'CPU_METADATA_LEASE'
    lease_root.mkdir()
    supervisor = {'config_sha256': 'b' * 64, 'preflight': {'source_commit': 'a' * 40, 'resources': self.auditor.RESOURCES}, 'worker_attempted': True, 'worker_launched': True, 'worker_resumed': True, 'preflight_revalidated_before_current_observations': True, 'job_closed_kill_on_close': True, 'sampler_threads_terminal_verified': True, 'gpu_telemetry_cleanup_verified': True, 'partial_outputs_frozen': True, 'owned_worker_identity_currently_alive': False, 'unjoined_sampler_threads': [], 'elapsed_seconds': times['lease_terminal'] - times['admission_first'], 'worker_identity': full_owner, 'worker_resume_monotonic': times['worker_resumed'], 'deadline_monotonic': times['worker_resumed'] + 600, 'admission_observations': admissions[:2], 'dispatch_observation': admissions[2], 'lease_path': str(lease_root), 'status': 'SUPERVISED_WORKER_EXITED_ZERO', 'returncode': 0}
    pair('SUPERVISOR_RECEIPT.json', supervisor)
    pair('supervisor_ownership.json', {'worker_pid': owner['pid'], 'supervisor_pid': 12344, 'job_name': 'CPU_METADATA_JOB', 'config_sha256': 'b' * 64})
    samples = []
    gpu_id = admissions[0]['gpu_uuid']
    for row in cpu_terminal['resource_samples']['CPU']:
        samples.append({'kind': 'CPU', 'sample_started_monotonic': row['monotonic'], 'host_available_bytes': row['available_bytes'], 'worker_tree_rss_bytes': row['worker_RSS_bytes'], 'owned_process_identities': [owner]})
    for row in cpu_terminal['resource_samples']['GPU']:
        samples.append({'kind': 'GPU', 'gpu_sample_started_monotonic': row['monotonic'], 'gpu_free_bytes': row['free_bytes'], 'gpu_total_bytes': row['total_bytes'], 'gpu_uuid': gpu_id})
    samples.sort(key=lambda row: row.get('sample_started_monotonic', row.get('gpu_sample_started_monotonic')))
    pair('RESOURCE_OBSERVATIONS.json', samples)
    leases = {'reservation.json': {'config_sha256': 'b' * 64}, 'worker.json': full_owner, 'resume.json': {'worker_identity': full_owner, 'external_monitor_ready_before_resume': True, 'resumed_monotonic': times['worker_resumed']}, 'terminal.json': supervisor}
    lease_files = {}
    for name, value in leases.items():
        raw = (canonical(value) + '\n').encode('ascii')
        (lease_root / name).write_bytes(raw)
        (preserved / ('lease_' + name)).write_bytes(raw)
        lease_files[name] = {'source_path': str(lease_root / name), 'path': 'lease_' + name, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    preservation = deepcopy(frame['preservation'])
    preservation.update(kind='PRESERVED_REGISTERED_ONE_PREFIX_NATIVE_TENSOR_DIAGNOSTIC', scope=native_scope, manifest_sha256=manifest['sha256'], source_files=manifest['source_files'], fixture_only=False, native_authority_issued=False, lease_files=lease_files)
    preservation['raw_files'] = {path.name: {'bytes': path.stat().st_size, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()} for path in original.iterdir()}
    preservation = seal(preservation)
    (preserved / 'PRESERVATION.json').write_text(canonical(preservation) + '\n', encoding='ascii')
    source_pins = {path: {'bytes': value['bytes'], 'sha256': value['sha256']} for path, value in manifest['source_files'].items()}
    return {'raw':original,'preserved':preserved,'manifest':manifest,'source_pins':source_pins,'preservation':preservation,'anchors':anchors,'owner':owner,'explicit_CPU_metadata_fixture':True}
