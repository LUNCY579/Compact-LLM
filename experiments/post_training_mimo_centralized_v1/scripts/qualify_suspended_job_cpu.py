"""One prospective stdlib-only suspended Job containment/cleanup qualification."""
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

from . import diagnose_windows_job as metadata
from .suspended_job_launch import launch_suspended
from .windows_job_cpu import Kernel, TracedJob, QUERY, SYNC, TERMINATE, WAIT_TIMEOUT, require, validate_owned_identity, cleanup_owned

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'experiments/post_training_mimo_centralized_v1'
CONFIG = PACKAGE / 'configs/SUSPENDED_JOB_CPU_QUAL_V1.json'
OUTPUT = ROOT / 'artifacts/mimo_central/suspended_job_cpu_qual_v1_20261001'
CONFIG_SHA256 = '55a92ccfd0836775793cf3212d7f08a6a5df879e455f20efb97861047a3217cd'
MODULE = 'experiments.post_training_mimo_centralized_v1.scripts.qualify_suspended_job_cpu'
write, read, wait_file = metadata.write, metadata.read, metadata.wait_file


def config():
    require(hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256, 'frozen suspended CPU config changed')
    return read(CONFIG)


def sources():
    paths = [CONFIG, PACKAGE / 'AMENDMENT_024.md', Path(__file__).resolve(),
             PACKAGE / 'scripts/suspended_job_launch.py', PACKAGE / 'scripts/windows_job_cpu.py',
             PACKAGE / 'scripts/diagnose_windows_job.py']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def zero_scope():
    return {'model_operations': 0, 'gpu_operations': 0, 'native_capture_authorized': False,
            'capture_authorized': False, 'capability_authorized': False, 'E1_authorized': False,
            'forbidden_modules': metadata.forbidden_modules()}


def require_pinned_actual_image(identity, cfg):
    expected = Path(cfg['cpu_interpreter']).resolve()
    require(os.path.normcase(str(Path(identity['image_path']).resolve())) == os.path.normcase(str(expected)),
            'actual CPU interpreter redirected from frozen image')
    require(hashlib.sha256(expected.read_bytes()).hexdigest() == cfg['cpu_interpreter_sha256'],
            'actual CPU interpreter image bytes changed')


def preserved_evidence():
    counts = {}
    for name in ('EXACT_PREFIX_V1_ATTEMPT_SUMMARY.json', 'JOB_CONTAINMENT_DIAGNOSTIC_V1_OUTCOME.json'):
        summary = read(PACKAGE / 'receipts' / name)
        entries = summary.get('evidence_files', summary.get('artifacts'))
        require(entries, 'prior attempt evidence manifest missing')
        for entry in entries:
            original = Path(entry['original_path'].replace('\\\\', '\\'))
            tracked = ROOT / entry['tracked_path']
            require(hashlib.sha256(original.read_bytes()).hexdigest() == entry['sha256']
                    and hashlib.sha256(tracked.read_bytes()).hexdigest() == entry['sha256'],
                    'prior attempt evidence changed')
        counts[name] = len(entries)
    require(list(counts.values()) == [13, 9], 'prior evidence count changed')
    return counts


class Tree:
    """Only held creation-bound descendants of this Popen root may be retained."""
    def __init__(self, kernel, directory, deadline, maximum_total_processes=9):
        self.kernel, self.directory, self.deadline = kernel, directory, deadline
        self.process = self.job = self.root_identity = None
        self.children, self.peak = {}, 0
        self.min_ram = self.min_disk = None
        self.frozen_sources = sources()
        self.maximum_total_processes = maximum_total_processes
        self.peak_job_lifetime_processes = self.peak_job_active_processes = 0

    def retain(self, pid):
        if pid == self.process.pid:
            return {'handle': int(self.process._handle), 'identity': self.root_identity,
                    'lineage': self.kernel.lineage(pid, os.getpid())}
        if pid not in self.children:
            handle = self.kernel.open(pid, QUERY | SYNC | TERMINATE)
            try:
                identity = self.kernel.identity(handle, pid)
                lineage = self.kernel.lineage(pid, os.getpid())
                validate_owned_identity(identity, identity, lineage, os.getpid(), self.process.pid)
                require(identity['created_filetime'] >= self.root_identity['created_filetime'],
                        'owned CPU child predates root')
                self.children[pid] = {'handle': handle, 'identity': identity, 'lineage': lineage}
            except BaseException:
                self.kernel.close(handle)
                raise
        return self.children[pid]

    def rss_if_active(self, handle):
        waited, _ = self.kernel.call('WaitForSingleObject', handle, 0)
        require(waited in (0, WAIT_TIMEOUT), 'owned CPU RSS wait failed')
        if waited == 0:
            return 0
        try:
            return self.kernel.rss(handle)
        except Exception:
            require(self.kernel.call('WaitForSingleObject', handle, 0)[0] == 0,
                    'unverified live CPU RSS query failure')
            return 0

    def guard(self):
        require(time.monotonic() < self.deadline and not self.kernel.logging_errors
                and not metadata.forbidden_modules() and sources() == self.frozen_sources,
                'CPU tree deadline/evidence/scope/source guard')
        ram, disk = self.kernel.available(), shutil.disk_usage(self.directory).free / 2**30
        self.min_ram = ram if self.min_ram is None else min(self.min_ram, ram)
        self.min_disk = disk if self.min_disk is None else min(self.min_disk, disk)
        require(ram >= 1.25 and disk >= 4, 'CPU ongoing RAM/disk reserve floor')
        rss = 0
        if self.process is not None:
            rss += self.rss_if_active(int(self.process._handle))
        if self.job is not None and self.job.handle:
            accounting = self.kernel.job_accounting(self.job.handle)
            self.peak_job_lifetime_processes = max(self.peak_job_lifetime_processes, accounting['total_processes'])
            self.peak_job_active_processes = max(self.peak_job_active_processes, accounting['active_processes'])
            require(accounting['total_processes'] <= self.maximum_total_processes
                    and accounting['active_processes'] <= 4, 'CPU owned Job lifetime/active process ceiling')
            pids = self.kernel.job_pids(self.job.handle)
            require(len(pids) <= 4, 'CPU unchanged four-process simultaneous ceiling')
            for pid in pids:
                if pid != self.process.pid:
                    self.retain(pid)
        for item in self.children.values():
            rss += self.rss_if_active(item['handle'])
        require(len(self.observed_created_pids()) <= 9, 'CPU cumulative observed process ceiling')
        self.peak = max(self.peak, rss)
        require(rss <= .5 * 2**30, 'CPU unchanged full-tree RSS ceiling')
        for path in self.directory.rglob('*'):
            if path.is_file():
                require(path.stat().st_size <= 1048576, 'CPU per-file byte ceiling')

    def launch(self, command, stdout, stderr):
        def popen(*args, **kwargs):
            self.process = subprocess.Popen(*args, **kwargs)
            self.kernel.record({'event': 'owned_cpu_root_created', 'pid': self.process.pid,
                                'command': command})
            return self.process
        def create_job(kernel, process):
            self.job = TracedJob(kernel, process)
            return self.job
        self.process, self.job, proof = launch_suspended(
            self.kernel, command, ROOT, stdout, stderr,
            min(self.deadline, time.monotonic() + 5), self.guard, popen, create_job)
        self.root_identity = proof['root_identity']
        return proof

    def cleanup(self, manual_fallback=True):
        final_accounting, accounting_error = None, None
        if self.job is not None and self.job.handle:
            try:
                final_accounting = self.kernel.job_accounting(self.job.handle)
                self.peak_job_lifetime_processes = max(self.peak_job_lifetime_processes, final_accounting['total_processes'])
                self.peak_job_active_processes = max(self.peak_job_active_processes, final_accounting['active_processes'])
                require(final_accounting['total_processes'] <= self.maximum_total_processes
                        and final_accounting['active_processes'] <= 4, 'final CPU Job accounting ceiling')
            except Exception as error:
                accounting_error = str(error)
        result = cleanup_owned(self.kernel, self.job, self.children, self.process,
                               os.getpid(), manual_fallback=manual_fallback)
        result['final_job_accounting_before_close'] = final_accounting
        if accounting_error:
            result['status'] = 'BLOCKED_OWNED_CLEANUP'
            result['errors'].append('final Job accounting: ' + accounting_error)
        return result

    def resources(self):
        return {'peak_observed_process_tree_rss_bytes': self.peak,
                'minimum_observed_available_ram_gib': self.min_ram,
                'minimum_observed_disk_free_gib': self.min_disk,
                'peak_job_lifetime_process_count': self.peak_job_lifetime_processes,
                'peak_job_active_process_count': self.peak_job_active_processes}

    def observed_created_pids(self):
        return sorted(set(self.children) | ({self.process.pid} if self.process is not None else set()))


def actual_ready(tree, ready, token, frozen, cfg):
    require(ready['token'] == token and ready['sources'] == frozen, 'CPU READY token/source mismatch')
    tree.guard()
    item = tree.retain(ready['identity']['pid'])
    validate_owned_identity(ready['identity'], item['identity'], ready['lineage'], os.getpid(), tree.process.pid)
    require_pinned_actual_image(item['identity'], cfg)
    member = tree.kernel.membership(item['handle'], tree.job.handle, 'actual_interpreter_before_GO')
    require(member['api_success'] and member['member'], 'actual interpreter is outside suspended owned Job')
    return item, member


def publish_go(directory, name, token, identity, frozen, kernel, job_handle, source_sha):
    parent_identity = kernel.identity(kernel.current(), os.getpid())
    require(not kernel.logging_errors, 'CPU GO evidence incomplete')
    write(directory / name, {'token': token, 'identity': identity, 'sources': frozen,
          'supervisor_pid': os.getpid(), 'supervisor_identity': parent_identity,
          'job_handle': int(job_handle), 'source_sha': source_sha})
    require(not kernel.logging_errors, 'CPU GO identity evidence incomplete')


def handshake(kernel, directory, prefix, parent_pid, token):
    frozen = sources()
    me = kernel.current()
    identity = kernel.identity(me, os.getpid())
    require_pinned_actual_image(identity, config())
    lineage = kernel.lineage(os.getpid(), parent_pid)
    write(directory / (prefix + 'READY.json'), {'token': token, 'identity': identity, 'lineage': lineage,
          'sources': frozen, 'ready_monotonic_ns': time.monotonic_ns(), **zero_scope()})
    go = wait_file(directory / (prefix + 'GO.json'), time.monotonic() + 5)
    require(go['token'] == token and go['sources'] == frozen and go['identity'] == identity
            and go['supervisor_pid'] == parent_pid, 'CPU GO owner/source/identity mismatch')
    parent, duplicate = kernel.open(parent_pid, 0x0040 | QUERY), w.HANDLE()
    try:
        try:
            require(kernel.identity(parent, parent_pid) == go['supervisor_identity'], 'CPU GO supervisor identity changed')
            result, error = kernel.call('DuplicateHandle', parent, w.HANDLE(go['job_handle']), me,
                                        c.byref(duplicate), 0, False, 2)
            require(result, f'CPU specific Job duplication failed: {error}')
            member = kernel.membership(me, duplicate, prefix + 'actual_interpreter_specific_job')
            require(member['api_success'] and not kernel.logging_errors, 'CPU worker membership API/evidence failed')
        finally:
            kernel.close(parent)
        require(not kernel.logging_errors, 'CPU handshake close evidence failed')
    except BaseException:
        if duplicate.value:
            kernel.close(duplicate)
        raise
    return identity, member, duplicate, go


def dummy(args):
    directory = args.case_path.resolve()
    require(directory.parent == OUTPUT.resolve() and directory.name in config()['cases'][:2],
            'registered CPU dummy path required')
    kernel = Kernel(metadata.logger(directory / 'DUMMY_API.jsonl'))
    identity, member, job, go = handshake(kernel, directory, 'DUMMY_', args.parent_pid, args.token)
    try:
        require(member['member'], 'CPU dummy did not inherit specific Job')
    finally:
        kernel.close(job)
    require(not kernel.logging_errors, 'CPU dummy close evidence failed')
    # Autonomous exit cannot occur within the entire owned 45-second window.
    # The outer Job/guard still enforces that unchanged window.
    deadline = time.monotonic() + config()['limits']['external_ceiling_seconds'] + config()['limits']['phase_seconds']
    write(directory / 'DUMMY_ARMED.json', {'identity': identity, 'member': member, 'token': args.token,
          'autonomous_exit_not_before_monotonic_ns': int(deadline * 1e9), **zero_scope()})
    while time.monotonic() < deadline:
        time.sleep(.1)
    raise RuntimeError('CPU dummy survived its registered parent cleanup phase')


def worker(args):
    cfg, directory = config(), args.case_path.resolve()
    require(directory.parent == OUTPUT.resolve() and directory.name in cfg['cases'], 'registered CPU worker path required')
    kernel = Kernel(metadata.logger(directory / 'WORKER_API.jsonl'))
    identity, member, job, go = handshake(kernel, directory, '', args.parent_pid, args.token)
    if directory.name == 'WRONG_JOB_GO':
        try:
            require(member['member'] is False, 'wrong specific Job unexpectedly matched')
        finally:
            kernel.close(job)
        require(not kernel.logging_errors, 'wrong Job refusal evidence failed')
        write(directory / 'WORKER_RESULT.json', {'status': 'REFUSED_WRONG_SPECIFIC_JOB_BEFORE_CPU_WORK',
              'identity': identity, 'membership': member, 'token': args.token, **zero_scope()})
        return 2
    try:
        require(member['member'], 'CPU worker did not inherit suspended specific Job')
        command = [cfg['cpu_interpreter'], '-B', '-m', MODULE, '--dummy', '--case-path', str(directory),
                   '--parent-pid', str(os.getpid()), '--token', args.token]
        with (directory / 'dummy.stdout.log').open('xb') as stdout, (directory / 'dummy.stderr.log').open('xb') as stderr:
            child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                     env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
        write(directory / 'DUMMY_LAUNCH.json', {'pid': child.pid, 'command': command, 'token': args.token, **zero_scope()})
        ready = wait_file(directory / 'DUMMY_READY.json', time.monotonic() + 5)
        require(ready['identity']['pid'] == child.pid and ready['token'] == args.token and ready['sources'] == sources(),
                'CPU dummy READY identity/source mismatch')
        handle = kernel.open(child.pid, QUERY | SYNC)
        try:
            observed = kernel.identity(handle, child.pid)
            validate_owned_identity(ready['identity'], observed, ready['lineage'], os.getpid(), child.pid)
            require_pinned_actual_image(observed, cfg)
            inherited = kernel.membership(handle, job, 'dummy_specific_job_before_GO')
            require(inherited['api_success'] and inherited['member'], 'CPU dummy inheritance unavailable')
        finally:
            kernel.close(handle)
        publish_go(directory, 'DUMMY_GO.json', args.token, ready['identity'], sources(), kernel, job, go['source_sha'])
        armed = wait_file(directory / 'DUMMY_ARMED.json', time.monotonic() + 5)
        require(armed['identity'] == ready['identity'] and armed['token'] == args.token, 'CPU dummy arm identity changed')
    finally:
        kernel.close(job)
    require(not kernel.logging_errors, 'CPU worker cleanup evidence lost before arm')
    write(directory / 'WORKER_ARMED.json', {'identity': identity, 'dummy_identity': ready['identity'],
          'membership': member, 'token': args.token, **zero_scope()})
    release = wait_file(directory / 'MAIN_RELEASE.json', time.monotonic() + 5)
    require(release['token'] == args.token and release['identity'] == identity, 'CPU main release mismatch')
    write(directory / 'WORKER_RESULT.json', {'status': 'NORMAL_CPU_WORKER_EXIT', 'identity': identity,
          'token': args.token, **zero_scope()})
    return 0


def wait_exit(tree, expected):
    deadline = min(tree.deadline, time.monotonic() + 5)
    while tree.process.poll() is None:
        tree.guard()
        require(time.monotonic() < deadline, 'CPU owned worker exit phase expired')
        time.sleep(.1)
    require(tree.process.returncode == expected and time.monotonic() < deadline, 'CPU worker wrong/late exit')


def confirm_job_termination_before_close(tree):
    require(tree.job is not None and tree.job.handle, 'termination proof requires the inner Job still open')
    owned = [{'handle': int(tree.process._handle), 'identity': tree.root_identity},
             *tree.children.values()]
    witnesses = []
    for item in owned:
        tree.guard()
        waited, error = tree.kernel.call('WaitForSingleObject', item['handle'], 2000,
                                         context={'termination_before_job_close': item['identity']})
        require(waited == 0 and time.monotonic() < tree.deadline and not tree.kernel.logging_errors,
                f'owned Job termination unproved before close: {waited}/{error}')
        code = tree.kernel.exit_code(item['handle'], {'termination_before_job_close': item['identity']})
        require(code == 2 and not tree.kernel.logging_errors, 'owned Job termination expected exit code two')
        witnesses.append({'identity': item['identity'], 'wait_return': waited,
                          'exit_code': code,
                          'inner_job_handle_still_open': int(tree.job.handle)})
    return witnesses


def run_case(cfg, name, deadline, frozen, source_sha):
    directory = OUTPUT / name
    directory.mkdir()
    kernel = Kernel(metadata.logger(directory / 'CONTROLLER_API.jsonl'))
    tree = Tree(kernel, directory, deadline, maximum_total_processes=2 if name == 'WRONG_JOB_GO' else 3)
    token = uuid.uuid4().hex
    wrong_job = None
    error = None
    witnesses = {}
    try:
        command = [cfg['executable'], '-B', '-m', MODULE, '--worker', '--case-path', str(directory),
                   '--parent-pid', str(os.getpid()), '--token', token]
        with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
            witnesses['suspended_launch'] = tree.launch(command, stdout, stderr)
            ready = wait_file(directory / 'READY.json', min(deadline, time.monotonic() + 5), tree.guard)
            actual, member = actual_ready(tree, ready, token, frozen, cfg)
            require(ready['identity']['pid'] != tree.process.pid, 'pinned venv pair shape changed')
            witnesses['actual_identity'], witnesses['actual_membership'] = actual['identity'], member
            go_job = tree.job.handle
            if name == 'WRONG_JOB_GO':
                wrong_job, code = kernel.call('CreateJobObjectW', None, None)
                require(wrong_job, f'CPU wrong-Job control creation failed: {code}')
                go_job = wrong_job
            publish_go(directory, 'GO.json', token, actual['identity'], frozen, kernel, go_job, source_sha)
            if name == 'WRONG_JOB_GO':
                wait_exit(tree, 2)
                refused = read(directory / 'WORKER_RESULT.json')
                require(refused['status'] == 'REFUSED_WRONG_SPECIFIC_JOB_BEFORE_CPU_WORK'
                        and refused['membership']['api_success'] and refused['membership']['member'] is False
                        and not (directory / 'WORKER_ARMED.json').exists(), 'wrong Job refusal control failed')
                witnesses['wrong_job_refusal'] = refused
            else:
                armed = wait_file(directory / 'WORKER_ARMED.json', min(deadline, time.monotonic() + 5), tree.guard)
                require(armed['identity'] == actual['identity'] and armed['token'] == token, 'CPU worker arm mismatch')
                descendant = tree.retain(armed['dummy_identity']['pid'])
                require(descendant['identity'] == armed['dummy_identity'], 'CPU dummy held identity changed')
                require_pinned_actual_image(descendant['identity'], cfg)
                inherited = kernel.membership(descendant['handle'], tree.job.handle, 'held_dummy_inner_job')
                require(inherited['api_success'] and inherited['member'], 'held dummy Job mismatch')
                witnesses['dummy_identity'], witnesses['dummy_membership'] = descendant['identity'], inherited
                if name == 'NORMAL_EXIT_CLEANUP':
                    write(directory / 'MAIN_RELEASE.json', {'token': token, 'identity': actual['identity']})
                    wait_exit(tree, 0)
                    alive = kernel.call('WaitForSingleObject', descendant['handle'], 0)[0]
                    require(alive == WAIT_TIMEOUT, 'normal cleanup dummy was not active before final Job close')
                    witnesses['dummy_wait_before_final_job_close'] = alive
                else:
                    require(tree.process.poll() is None
                            and kernel.call('WaitForSingleObject', actual['handle'], 0)[0] == WAIT_TIMEOUT
                            and kernel.call('WaitForSingleObject', descendant['handle'], 0)[0] == WAIT_TIMEOUT,
                            'termination control requires active verified tree')
                    tree.job.terminate()
                    witnesses['explicit_owned_job_termination'] = True
                    witnesses['termination_exit_before_job_close'] = confirm_job_termination_before_close(tree)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        if wrong_job:
            try:
                kernel.close(wrong_job)
            except Exception as exc:
                error = (error or '') + '; wrong Job handle close: ' + str(exc)
        cleanup = tree.cleanup(manual_fallback=False)
        write(directory / 'CLEANUP.json', cleanup)
    passed = (error is None and cleanup['status'] == 'OWNED_CPU_PROCESSES_EXITED'
              and not cleanup['owned_process_fallback_pids'] and not kernel.logging_errors
              and sources() == frozen and time.monotonic() < deadline)
    result = {'status': 'PASS_CPU_CONTAINMENT_AND_OWNED_CLEANUP' if passed else 'BLOCKED_CPU_CASE',
              'case': name, 'error': error, 'witnesses': witnesses, 'cleanup': cleanup,
              'main_worker_root_created': int(tree.process is not None),
              'observed_created_pids': tree.observed_created_pids(),
              'os_processes_created_observed': len(tree.observed_created_pids()), **tree.resources(), **zero_scope()}
    write(directory / 'CASE_RESULT.json', result)
    require(passed, 'CPU case blocked: ' + json.dumps(result, sort_keys=True))
    return result


def case_creation_accounting(attempted_cases):
    """Missing attempted-case receipts make aggregate counts unknown, not zero."""
    complete, missing, errors = [], [], {}
    for name in attempted_cases:
        path = OUTPUT / name / 'CASE_RESULT.json'
        if not path.exists():
            missing.append(name)
            continue
        try:
            value = read(path)
            require(value['main_worker_root_created'] in (0, 1)
                    and isinstance(value['os_processes_created_observed'], int)
                    and 0 <= value['os_processes_created_observed'] <= 3,
                    'CPU case accounting fields invalid')
            complete.append(value)
        except Exception as error:
            missing.append(name)
            errors[name] = type(error).__name__ + ': ' + str(error)
    roots = sum(value['main_worker_root_created'] for value in complete)
    processes = sum(value['os_processes_created_observed'] for value in complete)
    return {'attempted_cases': list(attempted_cases), 'case_receipts_missing': missing,
            'case_receipt_errors': errors,
            'process_creation_accounting_status': 'UNCERTAIN_MISSING_CASE_RECEIPT' if missing else 'COMPLETE_CASE_RECEIPTS',
            'worker_roots_created': None if missing else roots,
            'experimental_os_processes_created_observed': None if missing else processes,
            'complete_receipt_worker_root_lower_bound': roots,
            'complete_receipt_os_process_lower_bound': processes}


def controller(args):
    cfg = config()
    directory = OUTPUT.with_name(OUTPUT.name + '_outer')
    kernel = Kernel(metadata.logger(directory / 'BOOT_API.jsonl'))
    identity, member, job, go = handshake(kernel, directory, 'BOOT_', args.parent_pid, args.token)
    try:
        require(member['member'], 'CPU controller is outside suspended external Job')
    finally:
        kernel.close(job)
    require(not kernel.logging_errors, 'CPU bootstrap evidence failed before suite')
    require(not OUTPUT.exists(), 'CPU suite already consumed')
    OUTPUT.mkdir()
    started, deadline = time.monotonic(), time.monotonic() + 30
    results, attempted, error = [], [], None
    try:
        for name in cfg['cases']:
            require(sources() == go['sources'], 'CPU controller source changed')
            attempted.append(name)
            results.append(run_case(cfg, name, deadline, go['sources'], go['source_sha']))
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    accounting = case_creation_accounting(attempted)
    passed = (error is None and len(results) == 3 and not accounting['case_receipts_missing']
              and time.monotonic() < deadline)
    result = {'status': 'PASS_CPU_SUSPENDED_CONTAINMENT_AND_CLEANUP_ONLY' if passed else 'BLOCKED_CPU_SUITE',
              'source_sha': go['source_sha'], 'sources': sources(), 'elapsed_seconds': time.monotonic() - started,
              'cases': results, 'error': error, **accounting, **zero_scope()}
    write(OUTPUT / 'QUALIFICATION_RESULT.json', result)
    return 0 if passed else 2


def outer():
    started = time.monotonic()
    require(len(sys.argv) == 1 and os.name == 'nt' and not metadata.forbidden_modules(), 'CPU outer has no overrides/frameworks')
    cfg, frozen = config(), sources()
    require(Path(sys.executable).resolve() == Path(cfg['executable']).resolve(), 'CPU outer pinned venv required')
    for path_key, hash_key in [('executable', 'executable_sha256'), ('cpu_interpreter', 'cpu_interpreter_sha256')]:
        require(hashlib.sha256(Path(cfg[path_key]).read_bytes()).hexdigest() == cfg[hash_key], 'pinned CPU executable changed')
    require(metadata.git('status', '--porcelain') == '', 'commit clean CPU source before launch')
    prior = preserved_evidence()
    directory = OUTPUT.with_name(OUTPUT.name + '_outer')
    require(not directory.exists() and not OUTPUT.exists(), 'CPU qualification attempt consumed')
    directory.mkdir(parents=True)
    kernel = Kernel(metadata.logger(directory / 'OUTER_API.jsonl'))
    tree = Tree(kernel, directory, started + 45)
    token, error = uuid.uuid4().hex, None
    source_sha = metadata.git('rev-parse', 'HEAD')
    try:
        samples = []
        for index in range(2):
            tree.guard()
            available = kernel.available()
            samples.append({'available_ram_gib': available, 'monotonic_ns': time.monotonic_ns()})
            require(available >= 3.3125, 'CPU preworker reserve floor')
            if index == 0:
                time.sleep(2)
        require(samples[1]['monotonic_ns'] - samples[0]['monotonic_ns'] >= 2_000_000_000,
                'CPU preworker sample separation shortened')
        write(directory / 'ADMISSION.json', {'samples': samples, 'prior_evidence_verified': prior, **zero_scope()})
        command = [cfg['cpu_interpreter'], '-B', '-m', MODULE, '--controller',
                   '--parent-pid', str(os.getpid()), '--token', token]
        write(directory / 'OUTER_LAUNCH.json', {'source_sha': source_sha, 'sources': frozen,
              'command': command, 'limits': cfg['limits'], 'token': token, **zero_scope()})
        with (directory / 'stdout.log').open('xb') as stdout, (directory / 'stderr.log').open('xb') as stderr:
            proof = tree.launch(command, stdout, stderr)
            write(directory / 'OUTER_SUSPENDED_PROOF.json', proof)
            ready = wait_file(directory / 'BOOT_READY.json', min(tree.deadline, time.monotonic() + 5), tree.guard)
            actual, _ = actual_ready(tree, ready, token, frozen, cfg)
            require(actual['identity']['pid'] == tree.process.pid, 'CPU controller single-process shape changed')
            publish_go(directory, 'BOOT_GO.json', token, actual['identity'], frozen, kernel, tree.job.handle, source_sha)
            while tree.process.poll() is None:
                tree.guard()
                time.sleep(.1)
            require(tree.process.returncode == 0 and time.monotonic() < tree.deadline, 'CPU controller nonzero/late')
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        cleanup = tree.cleanup()
        write(directory / 'OUTER_CLEANUP.json', cleanup)
    passed = (error is None and cleanup['status'] == 'OWNED_CPU_PROCESSES_EXITED'
              and not cleanup['owned_process_fallback_pids'] and not kernel.logging_errors
              and sources() == frozen and time.monotonic() < tree.deadline)
    result = {'status': 'PASS_CPU_ONLY_EXTERNAL_CONTAINMENT' if passed else 'BLOCKED_CPU_OUTER_GUARD',
              'error': error, 'elapsed_seconds': time.monotonic() - started, 'source_sha': source_sha,
              'sources': frozen, 'cleanup': cleanup, 'controller_os_processes_created': int(tree.process is not None),
              **tree.resources(), **zero_scope()}
    write(directory / 'OUTER_RESULT.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 2


def guarded_outer():
    """Consume and preserve admission/startup exceptions without overwriting evidence."""
    try:
        return outer()
    except BaseException as exc:
        directory = OUTPUT.with_name(OUTPUT.name + '_outer')
        result = {'status': 'BLOCKED_CPU_OUTER_EXCEPTION',
                  'error': type(exc).__name__ + ': ' + str(exc),
                  'process_creation_accounting': 'Use retained API/launch receipts; not assumed zero.',
                  **zero_scope()}
        try:
            directory.mkdir(parents=True, exist_ok=True)
            receipt = directory / 'OUTER_EXCEPTION.json'
            if not receipt.exists():
                write(receipt, result)
        except Exception as receipt_error:
            result['receipt_error'] = type(receipt_error).__name__ + ': ' + str(receipt_error)
        print(json.dumps(result, sort_keys=True), flush=True)
        return 2


def main():
    require(not metadata.forbidden_modules(), 'CPU scope already imported framework')
    if len(sys.argv) == 1:
        return guarded_outer()
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--controller', action='store_true')
    mode.add_argument('--worker', action='store_true')
    mode.add_argument('--dummy', action='store_true')
    parser.add_argument('--case-path', type=Path)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--token', required=True)
    args = parser.parse_args()
    require(len(args.token) == 32 and all(ch in '0123456789abcdef' for ch in args.token), 'CPU attempt token required')
    require(args.controller or args.case_path is not None, 'registered CPU case path required')
    return controller(args) if args.controller else (worker(args) if args.worker else dummy(args))


if __name__ == '__main__':
    sys.exit(main())
