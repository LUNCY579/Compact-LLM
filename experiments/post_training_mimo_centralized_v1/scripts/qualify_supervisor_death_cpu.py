"""Prospective CPU controls for last-Job-handle lifetime, not native qualification."""
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
import threading
import time
import uuid

from . import diagnose_windows_job as metadata
from . import qualify_suspended_job_cpu as prior
from .suspended_job_launch import launch_suspended
from .windows_job_cpu import Kernel, TracedJob, QUERY, SYNC, TERMINATE, WAIT_TIMEOUT, require, validate_owned_identity

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / 'experiments/post_training_mimo_centralized_v1'
CONFIG = PACKAGE / 'configs/SUPERVISOR_DEATH_CPU_V1.json'
CONFIG_SHA256 = '1d4903116e1c23a6f41abe6c6a43b72adca06a885f85745be94a2786a3e5ae18'
OUTPUT = ROOT / 'artifacts/mimo_central/supervisor_death_cpu_v1_20261001'
AUTHORIZATION = ROOT / 'artifacts/mimo_central/authorizations/supervisor_death_cpu_v1_20261001.json'
CONSUMED = PACKAGE / 'receipts/SUPERVISOR_DEATH_CPU_V1_OUTCOME.json'
MODULE = 'experiments.post_training_mimo_centralized_v1.scripts.qualify_supervisor_death_cpu'
DUP_PROCESS_RIGHTS = QUERY | SYNC | TERMINATE | 0x0040
write, read, wait_file = metadata.write, metadata.read, metadata.wait_file


def config():
    require(hashlib.sha256(CONFIG.read_bytes()).hexdigest() == CONFIG_SHA256,
            'frozen supervisor-death CPU config changed')
    cfg = read(CONFIG)
    require(cfg['limits']['retainer_process_rights'] == DUP_PROCESS_RIGHTS
            and cfg['limits']['remote_duplicate_close_options'] == 1,
            'registered public remote-close rights/options changed')
    return cfg


def sources():
    paths = [CONFIG, PACKAGE / 'AMENDMENT_025.md', Path(__file__).resolve(),
             PACKAGE / 'scripts/windows_job_cpu.py', PACKAGE / 'scripts/suspended_job_launch.py',
             PACKAGE / 'scripts/qualify_suspended_job_cpu.py', PACKAGE / 'scripts/diagnose_windows_job.py']
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def scope():
    return prior.zero_scope()


def check_scope():
    require(not metadata.forbidden_modules(), 'framework imported into CPU-only control')


def preserved_evidence():
    counts = {}
    for name in ('EXACT_PREFIX_V1_ATTEMPT_SUMMARY.json', 'JOB_CONTAINMENT_DIAGNOSTIC_V1_OUTCOME.json',
                 'SUSPENDED_JOB_CPU_V1_OUTCOME.json'):
        summary = read(PACKAGE / 'receipts' / name)
        entries = summary.get('evidence_files', summary.get('artifacts'))
        require(entries, 'consumed evidence manifest missing')
        for entry in entries:
            original = Path(entry['original_path'].replace('\\\\', '\\'))
            tracked = ROOT / entry['tracked_path']
            require(hashlib.sha256(original.read_bytes()).hexdigest() == entry['sha256']
                    and hashlib.sha256(tracked.read_bytes()).hexdigest() == entry['sha256'],
                    'consumed raw evidence changed')
        counts[name] = len(entries)
    require(list(counts.values()) == [13, 9, 26], '48-artifact preservation count changed')
    return counts


def validate_authorization(auth, cfg, source_sha, frozen):
    require(type(auth.get('parent_approved')) is bool and auth['parent_approved']
            and type(auth.get('physical_cpu_run_authorized')) is bool and auth['physical_cpu_run_authorized'],
            'parent-reviewed exact-source physical CPU authorization is required')
    require(auth.get('experiment_id') == cfg['experiment_id']
            and auth.get('config_sha256') == CONFIG_SHA256 and auth.get('source_sha') == source_sha
            and auth.get('sources') == frozen and auth.get('limits') == cfg['limits'],
            'physical CPU authorization experiment/config/source/budget mismatch')
    review = auth.get('independent_review', {})
    require(review.get('status') == 'PASS_CPU_FIXTURES_ONLY' and review.get('source_sha') == source_sha,
            'exact-source independent fixture review is required')
    require(auth.get('native_capture_authorized') is False
            and auth.get('capture_authorized') is False and auth.get('capability_authorized') is False
            and auth.get('E1_authorized') is False, 'CPU authorization cannot grant native/capability/E1 scope')
    return auth


def authorization(cfg, source_sha, frozen):
    require(AUTHORIZATION.is_file(), 'physical CPU proposal has no parent-reviewed launch authorization')
    return validate_authorization(read(AUTHORIZATION), cfg, source_sha, frozen)


def refuse_consumed():
    require(not CONSUMED.exists() and not OUTPUT.exists(),
            'supervisor-death CPU registration consumed; separately register a successor')


class DeadlineSafety:
    """Deadline action uses the owned outer Job or original suspended startup root."""
    def __init__(self, deadline):
        self.deadline, self.job, self.pending_root = deadline, None, None
        self.intervened, self.errors, self.stopped = False, [], False
        self.lock, self.cancelled = threading.RLock(), threading.Event()
        self.thread = threading.Thread(target=self._watch, daemon=True, name='CPU-outer-deadline')

    def start(self):
        self.thread.start()

    def track_root(self, process, kernel):
        """Own only the original new Popen handle, before identity/Job logging."""
        with self.lock:
            require(self.job is None and self.pending_root is None,
                    'outer startup root ownership overlap')
            self.pending_root = (process, kernel)
            if self.intervened or time.monotonic() >= self.deadline:
                self.fire()  # A late Popen return still receives owned safety cleanup.
                raise RuntimeError('outer startup root returned after deadline/intervention')

    def install(self, job):
        with self.lock:
            require(self.job is None and not self.intervened and time.monotonic() < self.deadline,
                    'outer safety Job installation after deadline/intervention')
            self.job = job
            self.pending_root = None

    def detach(self, job):
        with self.lock:
            require(self.job is None or self.job is job, 'outer safety Job ownership changed')
            self.job = None

    def fire(self):
        with self.lock:
            self.intervened = True
            if self.job is not None:
                try:
                    self.job.terminate()
                    self.job.kernel.record({'event': 'uncredited_outer_deadline_safety_termination'})
                except BaseException as error:
                    self.errors.append(type(error).__name__ + ': ' + str(error))
            elif self.pending_root is not None:
                process, kernel = self.pending_root
                try:
                    result, error = kernel.call('TerminateProcess', int(process._handle), 2,
                        context={'uncredited_suspended_startup_root': process.pid,
                                 'original_popen_handle': True})
                    require(result, f'owned suspended startup safety termination failed: {error}')
                except BaseException as error:
                    self.errors.append(type(error).__name__ + ': ' + str(error))

    def _watch(self):
        if not self.cancelled.wait(max(0, self.deadline - time.monotonic())):
            self.fire()

    def stop(self):
        self.cancelled.set()
        self.thread.join(timeout=.2)
        self.stopped = not self.thread.is_alive()
        if not self.stopped:
            self.errors.append('outer safety thread did not stop within bounded join')


class WatchedOuterJob(TracedJob):
    """Keep deadline ownership until the actual outer Job close succeeds."""
    def __init__(self, kernel, process):
        self.watchdog = None
        super().__init__(kernel, process)

    def attach(self, watchdog):
        watchdog.install(self)
        self.watchdog = watchdog

    def close(self):
        # No watchdog lock spans native accounting, logging or process waits.
        # A failed close retains the installed safety action.
        super().close()
        if self.watchdog is not None:
            self.watchdog.detach(self)
            self.watchdog = None


class SafetyTree(prior.Tree):
    """Outer Job/process handles only; never retain an inner Job reference."""
    def __init__(self, kernel, directory, deadline, cfg, watchdog, prior_total=0):
        super().__init__(kernel, directory, deadline, cfg['limits']['maximum_total_os_processes_created'] - prior_total)
        self.cfg, self.watchdog, self.frozen_sources = cfg, watchdog, sources()
        self.safety_cleanup_used = False
        self.remote_duplicate_close_attempts = set()

    def retain(self, pid):
        if pid == self.process.pid:
            require(self.root_identity is not None, 'supervisor root identity not yet verified')
            return {'handle': int(self.process._handle), 'identity': self.root_identity,
                    'lineage': self.root_lineage}
        if pid not in self.children:
            handle = self.kernel.open(pid, DUP_PROCESS_RIGHTS)
            try:
                identity = self.kernel.identity(handle, pid)
                lineage = self.kernel.lineage(pid, os.getpid())
                validate_owned_identity(identity, identity, lineage, os.getpid(), self.process.pid)
                prior.require_pinned_actual_image(identity, self.cfg)
                require(identity['created_filetime'] >= self.root_identity['created_filetime'],
                        'retainer predates held supervisor root')
                member = self.kernel.membership(handle, self.job.handle, 'held_retainer_outer_safety_job')
                require(member['api_success'] and member['member'], 'retainer is outside outer safety Job')
                self.children[pid] = {'handle': handle, 'identity': identity, 'lineage': lineage}
                self.kernel.record({'event': 'owned_retainer_process_held', 'identity': identity})
            except BaseException:
                self.kernel.close(handle)
                raise
        return self.children[pid]

    def guard(self):
        check_scope()
        limits = self.cfg['limits']
        require(time.monotonic() < self.deadline and not self.watchdog.intervened
                and not self.watchdog.errors and not self.kernel.logging_errors
                and sources() == self.frozen_sources, 'CPU deadline/evidence/source/safety guard')
        ram, disk = self.kernel.available(), shutil.disk_usage(self.directory).free / 2**30
        self.min_ram = ram if self.min_ram is None else min(self.min_ram, ram)
        self.min_disk = disk if self.min_disk is None else min(self.min_disk, disk)
        require(ram >= limits['host_reserve_gib'] and disk >= limits['disk_free_gib'],
                'CPU ongoing RAM/disk reserve floor')
        rss = self.rss_if_active(int(self.process._handle)) if self.process is not None else 0
        if self.job is not None and self.job.handle:
            accounting = self.kernel.job_accounting(self.job.handle)
            self.peak_job_lifetime_processes = max(self.peak_job_lifetime_processes, accounting['total_processes'])
            self.peak_job_active_processes = max(self.peak_job_active_processes, accounting['active_processes'])
            require(accounting['total_processes'] <= self.maximum_total_processes
                    and accounting['active_processes'] <= limits['maximum_simultaneous_job_processes'],
                    'outer CPU Job cumulative/active process ceiling')
            pids = self.kernel.job_pids(self.job.handle)
            require(len(pids) <= limits['maximum_simultaneous_job_processes'], 'outer CPU active PID ceiling')
            for pid in pids:
                if pid != self.process.pid:
                    self.retain(pid)
        for item in self.children.values():
            rss += self.rss_if_active(item['handle'])
        require(len(self.observed_created_pids()) <= self.maximum_total_processes,
                'cumulative observed CPU process ceiling')
        self.peak = max(self.peak, rss)
        require(rss <= limits['tree_rss_gib'] * 2**30, 'CPU full-tree RSS ceiling')
        for path in OUTPUT.rglob('*'):
            if path.is_file():
                require(path.stat().st_size <= limits['max_file_bytes'], 'CPU per-file byte ceiling')

    def launch(self, command, stdout, stderr):
        def popen(*args, **kwargs):
            self.process = subprocess.Popen(*args, **kwargs)
            self.watchdog.track_root(self.process, self.kernel)
            self.kernel.record({'event': 'owned_supervisor_root_created', 'pid': self.process.pid})
            return self.process
        def outer_job(kernel, process):
            self.job = WatchedOuterJob(kernel, process)
            self.job.attach(self.watchdog)
            return self.job
        self.process, self.job, proof = launch_suspended(
            self.kernel, command, ROOT, stdout, stderr, min(self.deadline, time.monotonic() + 5),
            self.guard, popen, outer_job)
        self.root_identity, self.root_lineage = proof['root_identity'], proof['root_lineage']
        prior.require_pinned_actual_image(self.root_identity, self.cfg)
        return proof

    def cleanup(self):
        owned = ([int(self.process._handle)] if self.process is not None else [])
        owned.extend(item['handle'] for item in self.children.values())
        try:
            # This snapshot classifies later outer close as safety, never as proof.
            self.safety_cleanup_used = any(self.kernel.call('WaitForSingleObject', handle, 0)[0] != 0
                                           for handle in owned)
        except BaseException:
            self.safety_cleanup_used = True
        result = super().cleanup(manual_fallback=True)
        accounting = result.get('final_job_accounting_before_close')
        self.safety_cleanup_used = (self.safety_cleanup_used or accounting is None
                                    or accounting['active_processes'] != 0)
        result['uncredited_outer_safety_cleanup_used'] = self.safety_cleanup_used
        return result


def verify_live(tree, item):
    observed = tree.kernel.identity(item['handle'], item['identity']['pid'])
    validate_owned_identity(item['identity'], observed, item['lineage'], os.getpid(), tree.process.pid)
    prior.require_pinned_actual_image(observed, tree.cfg)
    require(tree.kernel.call('WaitForSingleObject', item['handle'], 0)[0] == WAIT_TIMEOUT,
            'registered held process is not active')


def observe_alive(tree, items):
    start = time.monotonic()
    duration = tree.cfg['limits']['minimum_alive_observation_seconds']
    require(start + duration < min(tree.deadline, start + tree.cfg['limits']['phase_seconds']),
            'alive observation has insufficient registered budget')
    samples = 0
    while True:
        tree.guard()
        for item in items:
            waited, error = tree.kernel.call('WaitForSingleObject', item['handle'], 0,
                                             context={'bounded_alive_identity': item['identity']})
            require(waited == WAIT_TIMEOUT, f'held process ended/failed in alive interval: {waited}/{error}')
        samples += 1
        elapsed = time.monotonic() - start
        if elapsed >= duration:
            require(not tree.kernel.logging_errors, 'alive evidence logging failed')
            return {'elapsed_seconds': elapsed, 'samples': samples,
                    'held_identities': [item['identity'] for item in items], 'wait_return': WAIT_TIMEOUT}
        time.sleep(min(tree.cfg['limits']['poll_seconds'], duration - elapsed))


def exit_witness(tree, item, expected_code=None):
    deadline = min(tree.deadline, time.monotonic() + tree.cfg['limits']['phase_seconds'])
    while True:
        tree.guard()
        waited, error = tree.kernel.call('WaitForSingleObject', item['handle'], 0,
                                         context={'experimental_exit_identity': item['identity']})
        require(waited in (0, WAIT_TIMEOUT), f'held exit wait API failed: {waited}/{error}')
        require(time.monotonic() < deadline, 'held experimental exit phase expired')
        if waited == 0:
            code = tree.kernel.exit_code(item['handle'], item['identity'])
            require(code != 259 and (expected_code is None or code == expected_code)
                    and not tree.kernel.logging_errors and not tree.watchdog.intervened,
                    'held experimental exit code/evidence/safety mismatch')
            return {'identity': item['identity'], 'wait_return': waited, 'exit_code': code,
                    'observed_before_outer_safety_cleanup': True}
        time.sleep(tree.cfg['limits']['poll_seconds'])


def close_remote_duplicate(tree, item, armed, token, frozen):
    require(armed['token'] == token and armed['sources'] == frozen and armed['identity'] == item['identity']
            and armed['membership']['api_success'] and armed['membership']['member'] is True,
            'retained duplicate descriptor owner/source/membership mismatch')
    value = prior.serialize_job_handle(armed['duplicate_value'])
    key = (item['identity']['pid'], item['identity']['created_filetime'], value)
    require(key not in tree.remote_duplicate_close_attempts, 'remote duplicate closure is single-attempt only')
    tree.guard()
    verify_live(tree, item)
    tree.remote_duplicate_close_attempts.add(key)  # CLOSE_SOURCE can consume on API failure too.
    result, error = tree.kernel.call('DuplicateHandle', item['handle'], w.HANDLE(value),
                                    None, None, 0, False, 1,
                                    context={'remote_close_registered_child_duplicate': item['identity'],
                                             'duplicate_value': value, 'target_process': None,
                                             'output_handle': None, 'options': 1})
    # CLOSE_SOURCE can close even on failure: subsequent exit cannot rescue it.
    require(result and not tree.kernel.logging_errors,
            f'remote duplicate close failed; closure outcome not credited: {error}')
    return {'api_return': int(result), 'last_error': int(error), 'options': 1,
            'target_process': None, 'output_handle': None, 'observer_inner_job_reference_acquired': False,
            'identity': item['identity'], 'duplicate_value': value}


def terminate_supervisor(tree, supervisor):
    tree.guard()
    verify_live(tree, supervisor)
    code = tree.cfg['limits']['supervisor_termination_exit_code']
    result, error = tree.kernel.call('TerminateProcess', supervisor['handle'], code,
                                    context={'registered_stalled_supervisor_identity': supervisor['identity']})
    require(result and not tree.kernel.logging_errors, f'owned supervisor termination failed: {error}')
    return exit_witness(tree, supervisor, code)


def experiment_sequence(tree, supervisor, retainer, armed, token, frozen, name):
    witnesses = {}
    if name == tree.cfg['cases'][0]:
        witnesses['supervisor_exit'] = terminate_supervisor(tree, supervisor)
        witnesses['retainer_alive_after_supervisor_death'] = observe_alive(tree, [retainer])
        witnesses['remote_duplicate_close'] = close_remote_duplicate(tree, retainer, armed, token, frozen)
    else:
        require(name == tree.cfg['cases'][1], 'unregistered supervisor-death CPU case')
        witnesses['remote_duplicate_close'] = close_remote_duplicate(tree, retainer, armed, token, frozen)
        witnesses['both_alive_after_duplicate_close'] = observe_alive(tree, [supervisor, retainer])
        witnesses['supervisor_exit'] = terminate_supervisor(tree, supervisor)
    witnesses['retainer_exit'] = exit_witness(tree, retainer)
    return witnesses


def public_ready(kernel, directory, name, parent_pid, token, cfg, frozen):
    identity = kernel.identity(kernel.current(), os.getpid())
    prior.require_pinned_actual_image(identity, cfg)
    lineage = kernel.lineage(os.getpid(), parent_pid)
    value = {'identity': identity, 'lineage': lineage, 'token': token, 'sources': frozen, **scope()}
    write(directory / name, value)
    return value


def check_go(go, ready, parent_pid, source_sha, kernel):
    require(go['identity'] == ready['identity'] and go['token'] == ready['token']
            and go['sources'] == ready['sources'] and go['supervisor_pid'] == parent_pid
            and go['source_sha'] == source_sha, 'CPU GO owner/token/source mismatch')
    parent = kernel.open(parent_pid, QUERY | 0x0040)
    try:
        require(kernel.identity(parent, parent_pid) == go['supervisor_identity'], 'CPU GO parent identity changed')
    finally:
        kernel.close(parent)
    require(not kernel.logging_errors, 'CPU GO parent evidence incomplete')


def duplicate_inner_job(kernel, go):
    parent, duplicate = None, w.HANDLE()
    try:
        try:
            parent = kernel.open(go['supervisor_pid'], QUERY | 0x0040)
            require(kernel.identity(parent, go['supervisor_pid']) == go['supervisor_identity'],
                    'retainer duplicate supervisor identity changed')
            value = prior.serialize_job_handle(go['job_handle'])
            result, error = kernel.call('DuplicateHandle', parent, w.HANDLE(value), kernel.current(),
                                        c.byref(duplicate), 0, False, 2)
            require(result, f'registered noninheritable inner Job duplication failed: {error}')
            member = kernel.membership(kernel.current(), duplicate, 'retainer_registered_specific_inner_job')
            require(member['api_success'] and member['member'] and not kernel.logging_errors,
                    'retainer specific inner Job membership/evidence failed')
        finally:
            if parent:
                kernel.close(parent)
        require(not kernel.logging_errors, 'retainer duplicate parent close evidence failed')
    except BaseException:
        if duplicate.value:
            kernel.close(duplicate)
        raise
    return duplicate, member


def role_admission(args):
    cfg, frozen = config(), sources()
    check_scope()
    require(args.case_path is not None and args.case_path.resolve().parent == OUTPUT.resolve()
            and args.case_path.name in cfg['cases'] and args.case_path.is_dir(), 'registered CPU role path required')
    authorization(cfg, args.source_sha, frozen)  # No Git/subprocess query inside experimental Jobs.
    require(not CONSUMED.exists(), 'CPU role cannot reuse a consumed registration')
    return cfg, frozen, args.case_path.resolve()


def retainer(args):
    cfg, frozen, directory = role_admission(args)
    kernel = Kernel(metadata.logger(directory / 'RETAINER_API.jsonl'))
    ready = public_ready(kernel, directory, 'RETAINER_READY.json', args.parent_pid, args.token, cfg, frozen)
    go = wait_file(directory / 'RETAINER_GO.json', time.monotonic() + 5)
    check_go(go, ready, args.parent_pid, args.source_sha, kernel)
    duplicate, member = duplicate_inner_job(kernel, go)
    deadline = time.monotonic() + cfg['limits']['external_ceiling_seconds'] + cfg['limits']['phase_seconds']
    require(not kernel.logging_errors, 'retainer evidence incomplete before arm')
    write(directory / 'RETAINER_ARMED.json', {**ready, 'membership': member,
          'duplicate_value': prior.serialize_job_handle(duplicate),
          'duplicate_inheritable_argument': False,
          'autonomous_exit_not_before_monotonic_ns': int(deadline * 1e9),
          'stalled_state': 'SLEEP_ONLY_NO_FURTHER_HANDLE_OPERATIONS'})
    # The observer remotely closes this exact handle. Never touch the stale wrapper.
    while time.monotonic() < deadline:
        time.sleep(.1)
    raise RuntimeError('retainer survived the registered external guard window')


def supervisor(args):
    cfg, frozen, directory = role_admission(args)
    kernel = Kernel(metadata.logger(directory / 'SUPERVISOR_API.jsonl'))
    ready = public_ready(kernel, directory, 'SUPERVISOR_READY.json', args.parent_pid, args.token, cfg, frozen)
    go = wait_file(directory / 'SUPERVISOR_GO.json', time.monotonic() + 5)
    check_go(go, ready, args.parent_pid, args.source_sha, kernel)
    child = job = None
    try:
        command = [cfg['cpu_interpreter'], '-B', '-m', MODULE, '--retainer', '--case-path', str(directory),
                   '--parent-pid', str(os.getpid()), '--token', args.token, '--source-sha', args.source_sha]
        with (directory / 'retainer.stdout.log').open('xb') as stdout, (directory / 'retainer.stderr.log').open('xb') as stderr:
            child, job, proof = launch_suspended(kernel, command, ROOT, stdout, stderr, time.monotonic() + 5)
        child_ready = wait_file(directory / 'RETAINER_READY.json', time.monotonic() + 5)
        require(child_ready['token'] == args.token and child_ready['sources'] == frozen
                and child_ready['identity'] == proof['root_identity'] and child.pid == child_ready['identity']['pid'],
                'retainer direct-interpreter READY shape/source changed')
        prior.require_pinned_actual_image(child_ready['identity'], cfg)
        observed = kernel.identity(int(child._handle), child.pid)
        validate_owned_identity(child_ready['identity'], observed, child_ready['lineage'], os.getpid(), child.pid)
        member = kernel.membership(int(child._handle), job.handle, 'supervisor_held_retainer_specific_inner_job')
        require(member['api_success'] and member['member'], 'supervisor-held retainer inner membership failed')
        write(directory / 'RETAINER_GO.json', {'identity': child_ready['identity'], 'token': args.token,
              'sources': frozen, 'source_sha': args.source_sha, 'supervisor_pid': os.getpid(),
              'supervisor_identity': ready['identity'], 'job_handle': prior.serialize_job_handle(job.handle)})
        child_armed = wait_file(directory / 'RETAINER_ARMED.json', time.monotonic() + 5)
        require(child_armed['identity'] == child_ready['identity'] and child_armed['sources'] == frozen
                and child_armed['token'] == args.token and child_armed['membership']['api_success']
                and child_armed['membership']['member'] and not kernel.logging_errors,
                'retainer arm source/identity/membership/evidence changed')
        deadline = time.monotonic() + cfg['limits']['external_ceiling_seconds'] + cfg['limits']['phase_seconds']
        write(directory / 'SUPERVISOR_ARMED.json', {**ready, 'retainer': child_armed,
              'retainer_suspended_proof': proof, 'retainer_inner_membership': member,
              'autonomous_exit_not_before_monotonic_ns': int(deadline * 1e9),
              'stalled_state': 'SLEEP_ONLY_RETAINING_ORIGINAL_INNER_JOB'})
        while time.monotonic() < deadline:
            time.sleep(.1)
        raise RuntimeError('supervisor survived registered external guard window')
    finally:
        if job is not None:
            job.close()


def case_passed(error, tree, cleanup, witnesses):
    return (error is None and 'retainer_exit' in witnesses and 'supervisor_exit' in witnesses
            and cleanup['status'] == 'OWNED_CPU_PROCESSES_EXITED'
            and not cleanup['owned_process_fallback_pids']
            and not cleanup['uncredited_outer_safety_cleanup_used'] and not cleanup['logging_errors']
            and not tree.watchdog.intervened and not tree.watchdog.errors and not tree.kernel.logging_errors
            and time.monotonic() < tree.deadline and sources() == tree.frozen_sources)


def run_case(cfg, name, deadline, watchdog, source_sha, frozen, prior_total):
    directory = OUTPUT / name
    directory.mkdir()
    token = uuid.uuid4().hex
    kernel = Kernel(metadata.logger(directory / 'OBSERVER_API.jsonl'))
    tree = SafetyTree(kernel, directory, deadline, cfg, watchdog, prior_total)
    witnesses, error = {}, None
    write(directory / 'CASE_STARTED.json', {'case': name, 'token': token, 'source_sha': source_sha, **scope()})
    try:
        command = [cfg['cpu_interpreter'], '-B', '-m', MODULE, '--supervisor', '--case-path', str(directory),
                   '--parent-pid', str(os.getpid()), '--token', token, '--source-sha', source_sha]
        with (directory / 'supervisor.stdout.log').open('xb') as stdout, (directory / 'supervisor.stderr.log').open('xb') as stderr:
            witnesses['outer_suspended_launch'] = tree.launch(command, stdout, stderr)
        ready = wait_file(directory / 'SUPERVISOR_READY.json', min(deadline, time.monotonic() + 5), tree.guard)
        supervisor_item = tree.retain(tree.process.pid)
        require(ready['identity'] == supervisor_item['identity'] and ready['token'] == token
                and ready['sources'] == frozen, 'supervisor READY owner/source changed')
        parent_identity = kernel.identity(kernel.current(), os.getpid())
        write(directory / 'SUPERVISOR_GO.json', {'identity': ready['identity'], 'token': token,
              'sources': frozen, 'source_sha': source_sha, 'supervisor_pid': os.getpid(),
              'supervisor_identity': parent_identity})  # Deliberately no outer Job handle.
        armed = wait_file(directory / 'SUPERVISOR_ARMED.json', min(deadline, time.monotonic() + 5), tree.guard)
        require(armed['identity'] == ready['identity'] and armed['token'] == token and armed['sources'] == frozen
                and armed['stalled_state'] == 'SLEEP_ONLY_RETAINING_ORIGINAL_INNER_JOB'
                and armed['autonomous_exit_not_before_monotonic_ns'] > int(watchdog.deadline * 1e9),
                'supervisor stall identity/source/deadline changed')
        child_arm = armed['retainer']
        retainer_item = tree.retain(child_arm['identity']['pid'])
        require(child_arm['identity'] == retainer_item['identity'] and child_arm['token'] == token
                and child_arm['sources'] == frozen and child_arm['membership']['api_success']
                and child_arm['membership']['member'] is True and child_arm['duplicate_inheritable_argument'] is False
                and child_arm['stalled_state'] == 'SLEEP_ONLY_NO_FURTHER_HANDLE_OPERATIONS'
                and child_arm['autonomous_exit_not_before_monotonic_ns'] > int(watchdog.deadline * 1e9),
                'retainer stalled duplicate descriptor invalid')
        require(armed['retainer_suspended_proof']['root_identity'] == retainer_item['identity']
                and armed['retainer_suspended_proof']['previous_suspend_count'] == 1
                and armed['retainer_inner_membership']['api_success']
                and armed['retainer_inner_membership']['member'] is True,
                'supervisor retainer enrollment proof missing')
        tree.guard()
        require(kernel.job_accounting(tree.job.handle)['total_processes'] == 2
                and set(kernel.job_pids(tree.job.handle)) == {supervisor_item['identity']['pid'], retainer_item['identity']['pid']},
                'registered direct two-process shape changed')
        witnesses['authenticated_retainer_arm'] = child_arm
        witnesses.update(experiment_sequence(tree, supervisor_item, retainer_item, child_arm, token, frozen, name))
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        cleanup = tree.cleanup()
        write(directory / 'CLEANUP.json', cleanup)
    accounting = cleanup.get('final_job_accounting_before_close')
    passed = case_passed(error, tree, cleanup, witnesses) and accounting is not None and accounting['total_processes'] == 2
    result = {'status': 'PASS_CPU_LAST_HANDLE_CONTROL_ONLY' if passed else 'BLOCKED_CPU_SUPERVISOR_DEATH_CASE',
              'case': name, 'token': token, 'source_sha': source_sha, 'sources': frozen, 'error': error,
              'witnesses': witnesses, 'cleanup': cleanup, 'observed_created_pids': tree.observed_created_pids(),
              'observed_process_creation_lower_bound': len(tree.observed_created_pids()),
              'job_lifetime_processes_created': accounting['total_processes'] if accounting is not None else None,
              'process_creation_accounting_status': 'NATIVE_OUTER_JOB_ACCOUNTING' if accounting is not None else 'UNKNOWN_NATIVE_JOB_ACCOUNTING',
              'physical_claim': 'bounded CPU last-handle control only; no native containment grant',
              **tree.resources(), **scope()}
    write(directory / 'CASE_RESULT.json', result)
    require(passed, 'supervisor-death CPU control blocked: ' + (error or 'cleanup/evidence/deadline/accounting'))
    return result


def observed_log_lower_bound(directory):
    """Keep valid logged observations even if the final case receipt is missing."""
    path = directory / 'OBSERVER_API.jsonl'
    if not path.exists():
        return 0, []
    identities, lifetime, errors = set(), 0, []
    try:
        require(path.stat().st_size <= 1048576, 'observer API log byte ceiling')
        for number, raw in enumerate(path.read_bytes().splitlines(), 1):
            try:
                event = json.loads(raw)
                if event.get('event') == 'owned_supervisor_root_created':
                    require(type(event.get('pid')) is int and event['pid'] > 0, 'invalid root PID')
                    identities.add(event['pid'])
                elif event.get('event') == 'windows_api' and event.get('api') == 'TerminateProcess':
                    context = event.get('context') or {}
                    pid = context.get('uncredited_suspended_startup_root')
                    if context.get('original_popen_handle') is True and type(pid) is int and pid > 0:
                        identities.add(pid)
                elif event.get('event') == 'owned_retainer_process_held':
                    identities.add(event['identity']['pid'])
                elif event.get('event') == 'owned_job_lifetime_accounting':
                    count = event['total_processes']
                    require(type(count) is int and 0 <= count < 2**32, 'invalid observed native count')
                    lifetime = max(lifetime, count)
            except Exception as error:
                errors.append(f'line {number}: {type(error).__name__}: {error}')
    except Exception as error:
        errors.append(type(error).__name__ + ': ' + str(error))
    return max(len(identities), lifetime), errors


def case_accounting(attempted, frozen, source_sha):
    lower, total, missing, errors = 0, 0, [], {}
    for name in attempted:
        path = OUTPUT / name / 'CASE_RESULT.json'
        observed_lower, log_errors = observed_log_lower_bound(path.parent)
        try:
            value = read(path)
            count, observed = value['job_lifetime_processes_created'], value['observed_process_creation_lower_bound']
            require(value['case'] == name and value['source_sha'] == source_sha and value['sources'] == frozen
                    and type(observed) is int and 0 <= observed < 2**32,
                    'case receipt identity/observed lower bound invalid')
            observed_lower = max(observed_lower, observed)
            require(type(count) is int and 0 <= count < 2**32, 'native creation count missing/invalid')
            observed_lower = max(observed_lower, count)
            require(observed_lower <= count <= 9, 'native count conflicts with observations or budget')
            require(value['status'] == 'PASS_CPU_LAST_HANDLE_CONTROL_ONLY' and not log_errors,
                    'failed case or incomplete API evidence keeps aggregate outcome unknown')
            total += count
        except BaseException as error:
            missing.append(name)
            errors[name] = type(error).__name__ + ': ' + str(error)
        if log_errors:
            errors[name] = errors.get(name, '') + '; log: ' + '; '.join(log_errors)
        lower += observed_lower
    return {'attempted_cases': list(attempted), 'missing_or_invalid_case_receipts': missing,
            'case_receipt_errors': errors, 'experimental_os_processes_created': None if missing else total,
            'confirmed_observed_process_lower_bound': lower,
            'process_creation_accounting_status': 'UNKNOWN_INCOMPLETE_CASE_RECEIPTS' if missing else 'COMPLETE_NATIVE_JOB_ACCOUNTING'}


def observer(cfg, source_sha, frozen, approved):
    started = time.monotonic()
    refuse_consumed()
    validate_authorization(approved, cfg, source_sha, frozen)
    require(os.name == 'nt' and Path(sys.executable).resolve() == Path(cfg['executable']).resolve(),
            'CPU observer requires the registered pinned Windows venv')
    require(metadata.git('status', '--porcelain') == '', 'CPU observer requires committed clean reviewed source')
    for path_key, hash_key in [('executable', 'executable_sha256'), ('cpu_interpreter', 'cpu_interpreter_sha256')]:
        require(hashlib.sha256(Path(cfg[path_key]).read_bytes()).hexdigest() == cfg[hash_key],
                'pinned CPU interpreter bytes changed')
    prior_evidence = preserved_evidence()
    OUTPUT.mkdir(parents=True)
    kernel = Kernel(metadata.logger(OUTPUT / 'ADMISSION_API.jsonl'))
    watchdog = DeadlineSafety(started + cfg['limits']['external_ceiling_seconds'])
    watchdog.start()
    deadline = started + cfg['limits']['suite_seconds']
    results, attempted, error = [], [], None
    try:
        samples = []
        for index in range(2):
            check_scope()
            available = kernel.available()
            samples.append({'available_ram_gib': available, 'monotonic_ns': time.monotonic_ns()})
            require(available >= cfg['limits']['preworker_host_gib']
                    and shutil.disk_usage(OUTPUT).free / 2**30 >= cfg['limits']['disk_free_gib'],
                    'CPU preworker RAM/disk floor')
            if index == 0:
                time.sleep(cfg['limits']['preworker_sample_separation_seconds'])
        require(samples[1]['monotonic_ns'] - samples[0]['monotonic_ns'] >= 2_000_000_000,
                'CPU preworker sample separation shortened')
        write(OUTPUT / 'ADMISSION.json', {'samples': samples, 'preserved_evidence': prior_evidence,
              'source_sha': source_sha, 'sources': frozen, 'limits': cfg['limits'],
              'authorization_sha256': hashlib.sha256(AUTHORIZATION.read_bytes()).hexdigest(), **scope()})
        total = 0
        for name in cfg['cases']:
            require(sources() == frozen and time.monotonic() < deadline and not watchdog.intervened,
                    'CPU suite source/deadline/safety changed')
            attempted.append(name)
            result = run_case(cfg, name, deadline, watchdog, source_sha, frozen, total)
            total += result['job_lifetime_processes_created']
            require(total <= cfg['limits']['maximum_total_os_processes_created'], 'CPU suite cumulative process ceiling')
            results.append(result)
    except BaseException as exc:
        error = type(exc).__name__ + ': ' + str(exc)
    finally:
        watchdog.stop()
    accounting = case_accounting(attempted, frozen, source_sha)
    passed = (error is None and len(results) == 2 and not accounting['missing_or_invalid_case_receipts']
              and accounting['experimental_os_processes_created'] == 4 and not watchdog.intervened
              and not watchdog.errors and watchdog.stopped and not kernel.logging_errors
              and time.monotonic() < deadline and sources() == frozen)
    result = {'status': 'PASS_CPU_SUPERVISOR_DEATH_LAST_HANDLE_ONLY' if passed else 'BLOCKED_CPU_SUPERVISOR_DEATH_SUITE',
              'source_sha': source_sha, 'sources': frozen, 'elapsed_seconds': time.monotonic() - started,
              'cases': results, 'error': error, 'outer_safety_intervened': watchdog.intervened,
              'outer_safety_errors': watchdog.errors, **accounting, **scope()}
    write(OUTPUT / 'QUALIFICATION_RESULT.json', result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0 if passed else 2


def main():
    check_scope()
    cfg, frozen = config(), sources()
    if len(sys.argv) == 1:
        refuse_consumed()
        source_sha = metadata.git('rev-parse', 'HEAD')
        approved = authorization(cfg, source_sha, frozen)  # Before any Kernel/process/output creation.
        try:
            return observer(cfg, source_sha, frozen, approved)
        except BaseException as error:
            # An authorized failed/missing outcome stays unknown, never zero/PASS.
            result = {'status': 'BLOCKED_AUTHORIZED_CPU_EXCEPTION', 'source_sha': source_sha,
                      'sources': frozen, 'error': type(error).__name__ + ': ' + str(error),
                      'experimental_os_processes_created': None,
                      'process_creation_accounting_status': 'UNKNOWN_USE_PRESERVED_API_LOWER_BOUNDS', **scope()}
            try:
                OUTPUT.mkdir(parents=True, exist_ok=True)
                write(OUTPUT / 'OUTER_EXCEPTION.json', result)
            except BaseException as receipt_error:
                result['receipt_error'] = type(receipt_error).__name__ + ': ' + str(receipt_error)
            print(json.dumps(result, sort_keys=True), flush=True)
            return 2
    parser = argparse.ArgumentParser()
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument('--supervisor', action='store_true')
    modes.add_argument('--retainer', action='store_true')
    parser.add_argument('--case-path', type=Path, required=True)
    parser.add_argument('--parent-pid', type=int, required=True)
    parser.add_argument('--token', required=True)
    parser.add_argument('--source-sha', required=True)
    args = parser.parse_args()
    require(len(args.token) == 32 and all(ch in '0123456789abcdef' for ch in args.token)
            and len(args.source_sha) == 40 and all(ch in '0123456789abcdef' for ch in args.source_sha),
            'registered CPU token/source required')
    return supervisor(args) if args.supervisor else retainer(args)


if __name__ == '__main__':
    sys.exit(main())
