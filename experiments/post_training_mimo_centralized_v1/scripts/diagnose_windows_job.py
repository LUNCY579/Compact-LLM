"""One registered CPU-only two-case Windows Job diagnosis; never native capture."""
from __future__ import annotations

import argparse
import ctypes as c
from ctypes import wintypes as w
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .windows_job_cpu import Kernel, TracedJob, QUERY, SYNC, TERMINATE, WAIT_TIMEOUT, require, validate_owned_identity, cleanup_owned

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'experiments/post_training_mimo_centralized_v1'
CONFIG = PACKAGE / 'configs/JOB_CONTAINMENT_DIAGNOSTIC_V1.json'
OUTPUT = ROOT / 'artifacts/mimo_central/job_containment_diagnostic_v1_20261001'
MAX_BYTES = 1048576
CONFIG_SHA256 = '8102866aa48da32d4d236d42dec171b32dec2425c6f2a5f7b72682cd9aa80d20'


def write(path, value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
    require(len(raw) <= MAX_BYTES, 'CPU receipt exceeds registered byte budget')
    pending = path.with_name(path.name + '.pending.' + uuid.uuid4().hex)
    with pending.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.rename(pending, path)  # Windows rename never replaces an existing destination.


def read(path):
    require(path.stat().st_size <= MAX_BYTES, 'CPU metadata exceeds registered byte budget')
    return json.loads(path.read_bytes())


def logger(path):
    def record(event):
        raw = json.dumps(event, sort_keys=True, separators=(',', ':'), allow_nan=False).encode() + b'\n'
        current = path.stat().st_size if path.exists() else 0
        require(current + len(raw) <= MAX_BYTES, 'CPU API log budget exhausted')
        with path.open('ab') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    return record


def forbidden_modules():
    return sorted(name for name in sys.modules if name.split('.')[0] in {'torch', 'transformers', 'tensorflow', 'jax', 'cupy'})


def sources():
    paths = [CONFIG, Path(__file__).resolve(), PACKAGE / 'scripts/windows_job_cpu.py',
             PACKAGE / 'scripts/run_cpu_job_diagnostic.py']
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def wait_file(path, deadline, guard=lambda: None):
    while not path.exists():
        guard()
        require(time.monotonic() < deadline, 'CPU-only metadata wait timeout')
        time.sleep(.05)
    value = read(path)
    guard()
    require(time.monotonic() < deadline, 'CPU-only metadata returned after phase deadline')
    return value


def worker(args):
    directory = args.case_path.resolve()
    require(directory.parent == OUTPUT.resolve() and directory.name in ('IMMEDIATE_ASSIGN', 'READY_BEFORE_ASSIGN'), 'owned CPU case path required')
    require(not forbidden_modules(), 'framework imported into CPU diagnostic')
    events = logger(directory / 'WORKER_API.jsonl')
    kernel = Kernel(events)
    me = kernel.current()
    identity = kernel.identity(me, os.getpid())
    lineage = kernel.lineage(os.getpid(), args.supervisor_pid)
    own_sources = sources()
    write(directory / 'READY.json', {'token': args.token, 'identity': identity, 'lineage': lineage,
          'ready_utc_ns': time.time_ns(), 'ready_monotonic_ns': time.monotonic_ns(), 'sources': own_sources, 'sys_executable': sys.executable,
          'forbidden_modules': forbidden_modules()})
    go = wait_file(directory / 'GO.json', time.monotonic() + 5)
    require(go['token'] == args.token and go['sources'] == own_sources and go['supervisor_pid'] == args.supervisor_pid, 'CPU GO source/owner mismatch')
    parent = kernel.open(args.supervisor_pid, 0x0040 | QUERY)
    duplicate = w.HANDLE()
    try:
        require(kernel.identity(parent, args.supervisor_pid) == go['supervisor_identity'], 'CPU supervisor PID reused')
        result, error = kernel.call('DuplicateHandle', parent, w.HANDLE(go['job_handle']), me,
                                   c.byref(duplicate), 0, False, 0x2)
        require(result, f'specific Job duplication failed: {error}')
        witness = kernel.membership(me, duplicate, 'actual_interpreter_specific_owned_job')
        require(witness['api_success'], 'primary worker membership API failure')
        negative = kernel.membership(me, w.HANDLE(-2), 'invalid_job_handle_negative_control')
        require(not negative['api_success'] and negative['member'] is None, 'invalid Job handle negative control did not fail')
        write(directory / 'WORKER_RESULT.json', {'status': 'RECORDED_CPU_API_AND_LINEAGE_ONLY',
              'identity': identity, 'lineage': lineage, 'sources': own_sources, 'membership': witness,
              'negative_control': negative, 'forbidden_modules': forbidden_modules(),
              'model_operations': 0, 'gpu_operations': 0, 'native_capture_authorized': False})
    finally:
        kernel.close(duplicate)
        kernel.close(parent)
    return 0


def git(*args):
    p = subprocess.run(['git', '-c', 'safe.directory=' + ROOT.as_posix(), '-C', str(ROOT), *args], capture_output=True, text=True, timeout=10)
    require(p.returncode == 0, p.stderr)
    return p.stdout.strip()


def verify_preservation():
    summary = read(PACKAGE / 'receipts/EXACT_PREFIX_V1_ATTEMPT_SUMMARY.json')
    for entry in summary['evidence_files']:
        for path in (ROOT / entry['tracked_path'], Path(entry['original_path'])):
            require(hashlib.sha256(path.read_bytes()).hexdigest() == entry['sha256'], 'V1 artifact changed: ' + str(path))
    return {'files': 13, 'byte_identical': True}


def supervise():
    started = time.monotonic()
    require(not OUTPUT.exists(), 'single CPU diagnostic directory already exists; no retry')
    require(not forbidden_modules(), 'framework imported into CPU supervisor')
    cfg = read(CONFIG)
    require(hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256, 'frozen CPU diagnostic configuration changed')
    require(cfg['cases'] == ['IMMEDIATE_ASSIGN', 'READY_BEFORE_ASSIGN'] and cfg['limits']['suite_seconds'] == 30
            and cfg['model_operations'] == cfg['gpu_operations'] == 0 and not cfg['native_capture_authorized'], 'registered CPU-only scope changed')
    require(hashlib.sha256(Path(sys.executable).read_bytes()).hexdigest() == cfg['executable_sha256']
            and Path(sys.executable).resolve() == Path(cfg['executable']).resolve(), 'pinned venv launcher identity changed')
    require(hashlib.sha256((ROOT / 'experiments/opallo_native_collector_v1/supervisor.py').read_bytes()).hexdigest()
            == cfg['inherited_supervisor_sha256'], 'original supervisor changed')
    require(git('status', '--porcelain') == '', 'commit clean CPU source before launch')
    source_sha, source_files, before = git('rev-parse', 'HEAD'), sources(), verify_preservation()
    OUTPUT.mkdir(parents=True)
    events = logger(OUTPUT / 'SUPERVISOR_API.jsonl')
    kernel = Kernel(events)
    admission = []
    for index in range(2):
        sample = {'at_utc_ns': time.time_ns(), 'host_available_gib': kernel.available(),
                  'disk_free_gib': shutil.disk_usage(OUTPUT).free / 2**30}
        admission.append(sample)
        require(sample['host_available_gib'] >= 3.3125 and sample['disk_free_gib'] >= 4, 'CPU preworker reserve blocks launch')
        if index == 0:
            time.sleep(2)
    write(OUTPUT / 'LAUNCH.json', {'config': cfg, 'source_sha': source_sha, 'sources': source_files,
          'admission': admission, 'preservation_before': before, 'parent_reported_agent_settings':
          {'model': 'gpt-6.1-sol', 'thinking': 'max', 'service_tier': 'default', 'global_settings_changed': False}})
    results, peak = [], 0
    supervisor_identity = kernel.identity(kernel.current(), os.getpid())
    try:
        for name in cfg['cases']:
            directory = OUTPUT / name
            directory.mkdir()
            process = job = held = expected = None
            owned_children = {}
            cleanup = []
            token = uuid.uuid4().hex

            def guard():
                nonlocal peak
                require(not kernel.logging_errors, 'CPU API evidence logging failed')
                require(time.monotonic() - started < 30, 'CPU suite wall ceiling')
                require(kernel.available() >= 1.25, 'CPU host reserve ceiling')
                rss = 0
                if process is not None and process.poll() is None:
                    rss += kernel.rss(int(process._handle))
                    table = kernel.parent_table()
                    children = [pid for pid, parent in table.items() if parent == process.pid]
                    require(len(children) <= 1, 'unexpected extra CPU worker descendants')
                    for pid in children:
                        if pid not in owned_children:
                            handle = kernel.open(pid, QUERY | SYNC | TERMINATE)
                            identity = kernel.identity(handle, pid)
                            lineage = kernel.lineage(pid, os.getpid())
                            validate_owned_identity(identity, identity, lineage, os.getpid(), process.pid)
                            require(identity['created_filetime'] >= root_identity['created_filetime'], 'CPU descendant predates owned launcher')
                            owned_children[pid] = {'handle': handle, 'identity': identity, 'lineage': lineage}
                        require(not any(parent == pid for parent in table.values()), 'CPU worker created an unauthorized grandchild')
                for item in owned_children.values():
                    handle = item['handle']
                    if kernel.call('WaitForSingleObject', handle, 0)[0] == WAIT_TIMEOUT:
                        rss += kernel.rss(handle)
                peak = max(peak, rss)
                require(rss <= .5 * 2**30, 'CPU worker-tree RSS ceiling')
                for path in directory.glob('*'):
                    require(path.stat().st_size <= MAX_BYTES, 'CPU file ceiling')

            try:
                guard()
                command = [cfg['executable'], '-B', '-m', 'experiments.post_training_mimo_centralized_v1.scripts.diagnose_windows_job',
                           '--case-path', str(directory), '--supervisor-pid', str(os.getpid()), '--token', token]
                with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
                    process = subprocess.Popen(command, cwd=str(ROOT), stdout=stdout, stderr=stderr,
                                               creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                                               env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1'))
                    if name == 'IMMEDIATE_ASSIGN':
                        job = TracedJob(kernel, process)
                    root_identity = kernel.identity(int(process._handle), process.pid)
                    ready = wait_file(directory / 'READY.json', min(started + 30, time.monotonic() + 5), guard)
                    require(ready['token'] == token and ready['sources'] == source_files, 'CPU READY token/source mismatch')
                    expected = ready['identity']
                    if expected['pid'] == process.pid:
                        held = int(process._handle)
                    else:
                        guard()
                        require(expected['pid'] in owned_children, 'CPU READY process was not an observed owned descendant')
                        held = owned_children[expected['pid']]['handle']
                    actual = kernel.identity(held, expected['pid'])
                    validate_owned_identity(expected, actual, kernel.lineage(expected['pid'], os.getpid()), os.getpid(), process.pid)
                    if name == 'READY_BEFORE_ASSIGN':
                        job = TracedJob(kernel, process)
                    root_member = kernel.membership(int(process._handle), job.handle, 'launcher_root')
                    actual_member = kernel.membership(held, job.handle, 'actual_interpreter_parent_query')
                    require(root_member['api_success'] and root_member['member'] and actual_member['api_success'], 'primary parent membership API failure/root not assigned')
                    write(directory / 'GO.json', {'token': token, 'sources': source_files, 'job_handle': int(job.handle),
                          'supervisor_pid': os.getpid(), 'supervisor_identity': supervisor_identity,
                          'root_identity': root_identity})
                    deadline = min(started + 30, time.monotonic() + 5)
                    while process.poll() is None:
                        guard()
                        require(time.monotonic() < deadline, 'CPU worker exit timeout')
                        time.sleep(.1)
                    require(process.returncode == 0, 'CPU metadata worker failed')
                    measured = read(directory / 'WORKER_RESULT.json')
                    guard()
                    require(measured['membership'] == actual_member and not measured['forbidden_modules'], 'CPU parent/worker membership witnesses disagree')
                    results.append({'case': name, 'root_identity': root_identity, 'actual_identity': actual,
                                    'ready_before_assignment': name == 'READY_BEFORE_ASSIGN',
                                    'launcher_membership': root_member, 'actual_membership': actual_member,
                                    'worker_exit_code': process.returncode})
            finally:
                cleanup = cleanup_owned(kernel, job, owned_children, process, os.getpid())
                write(directory / 'CLEANUP.json', cleanup)
                require(cleanup['status'] == 'OWNED_CPU_PROCESSES_EXITED' and not kernel.logging_errors, 'CPU cleanup/evidence failure')
        require(git('rev-parse', 'HEAD') == source_sha and git('status', '--porcelain') == '' and sources() == source_files, 'CPU source drift')
        after = verify_preservation()
        require(time.monotonic() - started < 30, 'CPU suite late final evidence; block success')
        final = {'status': 'MEASURED_CPU_CONTAINMENT_DIAGNOSTIC_ONLY', 'source_sha': source_sha,
                 'cases': results, 'elapsed_seconds': time.monotonic() - started,
                 'peak_observed_process_tree_rss_bytes': peak, 'preservation_after': after,
                 'model_operations': 0, 'gpu_operations': 0, 'native_capture_authorized': False,
                 'forbidden_modules': forbidden_modules(), 'E1_authorized': False}
        write(OUTPUT / 'DIAGNOSTIC_RESULT.json', final)
        print(json.dumps(final, sort_keys=True), flush=True)
        return 0
    except BaseException as error:
        write(OUTPUT / 'FAILURE_RECEIPT.json', {'status': 'BLOCKED_CPU_DIAGNOSTIC', 'error': str(error), 'cases': results,
              'elapsed_seconds': time.monotonic() - started, 'model_operations': 0, 'gpu_operations': 0})
        raise


def guarded_supervise():
    existed = OUTPUT.exists()
    try:
        return supervise()
    except BaseException as error:
        if not existed:
            OUTPUT.mkdir(parents=True, exist_ok=True)
            failure = OUTPUT / 'FAILURE_RECEIPT.json'
            if not failure.exists():
                write(failure, {'status': 'BLOCKED_CPU_DIAGNOSTIC_STARTUP_OR_ADMISSION',
                      'error': type(error).__name__ + ': ' + str(error), 'model_operations': 0, 'gpu_operations': 0})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--case-path', type=Path)
    parser.add_argument('--supervisor-pid', type=int)
    parser.add_argument('--token')
    parser.add_argument('--outer-pid', type=int)
    parser.add_argument('--outer-token')
    args = parser.parse_args()
    require(os.name == 'nt', 'Windows-only diagnostic')
    if args.supervise:
        require(args.case_path is None and args.supervisor_pid is None and args.token is None, 'supervisor has no worker overrides')
        require(args.outer_pid and args.outer_token, 'bounded CPU outer guard required')
        outer = OUTPUT.with_name(OUTPUT.name + '_outer')
        boot_kernel = Kernel(logger(outer / 'BOOT_API.jsonl'))
        identity = boot_kernel.identity(boot_kernel.current(), os.getpid())
        write(outer / 'BOOT_READY.json', {'token': args.outer_token, 'identity': identity,
              'lineage': boot_kernel.lineage(os.getpid(), args.outer_pid), 'sources': sources(),
              'ready_utc_ns': time.time_ns(), 'ready_monotonic_ns': time.monotonic_ns()})
        go = wait_file(outer / 'BOOT_GO.json', time.monotonic() + 5)
        require(go['token'] == args.outer_token and go['identity'] == identity and go['sources'] == sources(), 'CPU outer guard binding mismatch')
        parent = boot_kernel.open(args.outer_pid, 0x0040)
        duplicate = w.HANDLE()
        try:
            result, error = boot_kernel.call('DuplicateHandle', parent, w.HANDLE(go['job_handle']), boot_kernel.current(), c.byref(duplicate), 0, False, 0x2)
            require(result, f'CPU outer Job duplicate failed: {error}')
            membership = boot_kernel.membership(boot_kernel.current(), duplicate, 'CPU_supervisor_outer_job')
            require(membership['api_success'] and membership['member'] and not boot_kernel.logging_errors, 'CPU supervisor is not externally contained')
        finally:
            boot_kernel.close(duplicate)
            boot_kernel.close(parent)
        return guarded_supervise()
    require(args.case_path is not None and args.supervisor_pid and args.token and len(args.token) == 32, 'owned CPU worker arguments required')
    return worker(args)


if __name__ == '__main__':
    sys.exit(main())
