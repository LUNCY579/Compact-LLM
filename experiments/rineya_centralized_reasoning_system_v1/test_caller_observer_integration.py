"""One bounded final source6 invocation: same K/O/C roles/proxies/library.

No actual child, Win32 API, UV execution, host DB, network, model or installer.
Temporary real OperationsStore drives actual canonical transactions and receipts.
"""
import ctypes as ct
import importlib.util
import json
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

P = Path(__file__).resolve().parent

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

m = load('integration_source5', P / 'caller_observer_integration.py')
lib = m.library()
ops = m.operations()


class Clock:
    def __init__(self):
        self.origin = time.monotonic()
        self.wall_origin = time.time()
        self.offset = 0.0
    def __call__(self):
        return time.monotonic() - self.origin + self.offset
    def wall(self):
        return self.wall_origin + self()
    def advance(self, seconds):
        self.offset += seconds


class NoTimer:
    def __init__(self, seconds, fn):
        self.fn, self.daemon = fn, False
    def start(self): pass
    def cancel(self): pass
    def is_alive(self): return False
    def join(self, timeout=None): pass


class QueueChannel:
    def __init__(self, incoming, outgoing):
        self.incoming, self.outgoing = incoming, outgoing
    def send(self, raw): self.outgoing.put(raw)
    def receive(self):
        raw = self.incoming.get(timeout=1)
        m.need(raw is not None, 'fixture channel role lost')
        return raw


class FixtureAdmission:
    def __init__(self, gate, fail=None):
        self.gate, self.fail = gate, fail
    def verify(self, kernel):
        m.need(not kernel.actual, 'fixture cannot qualify actual kernel')
        if self.fail == 'admission':
            raise m.Refused('controlled absent actual admission')
        return dict(self.gate)
    def role_ticket(self):
        return {'controlled_capsule': m.digest(self.gate), 'actual_qualification': False}
    def verify_role(self, kernel, bootstrap):
        m.need(not kernel.actual and bootstrap['authority'] == self.role_ticket() and
               bootstrap['gate'] == self.gate, 'controlled exact role authority')
        if self.fail == 'role_authority':
            raise m.Refused('controlled role authority absent before O store')
        return self.gate


class Fn:
    def __init__(self, fn): self.fn = fn
    def __call__(self, *args): return self.fn(*args)


class FileKernel:
    """Controlled typed I/O drives the real Win32Publication implementation."""
    def __init__(self, kernel):
        self.kernel, self.rows, self.next = kernel, {}, 800
        self.pending_length, self.error = 0, 0
        self.CreateFileW = Fn(self.create)
        self.CreateEventW = Fn(lambda *a: 999)
        self.CloseHandle = Fn(lambda *a: True)
        self.WriteFile, self.ReadFile = Fn(self.write), Fn(self.read)
        self.GetOverlappedResult, self.CancelIoEx = Fn(self.result), Fn(self.cancel)
    def create(self, path, access, share, security, disposition, flags, template):
        assert disposition == 1 and flags == 0xC0000000 and share == 0
        if Path(path).exists(): return ct.c_void_p(-1).value
        self.next += 1
        self.rows[self.next] = Path(path)
        Path(path).touch(exist_ok=False)
        return self.next
    def write(self, handle, buffer, length, count, overlap):
        raw = ct.string_at(buffer, length)
        with self.rows[handle].open('r+b') as stream:
            stream.write(raw); stream.flush()
        self.pending_length = length
        if self.kernel.fault == 'whole_late': self.kernel.clock.advance(2.1)
        if self.kernel.fault == 'pending_cancel':
            self.error = 997
            self.kernel.clock.advance(2.1)
            return False
        return True
    def read(self, handle, buffer, length, count, overlap):
        raw = self.rows[handle].read_bytes()
        if self.kernel.fault == 'whole_readback': raw = b'!' + raw[1:]
        ct.memmove(buffer, raw, length)
        self.pending_length = length
        return True
    def result(self, handle, overlap, count, wait):
        assert not wait
        if self.kernel.fault == 'pending_cancel':
            self.error = 996
            return False
        ct.cast(count, ct.POINTER(m.wt.DWORD)).contents.value = self.pending_length
        return True
    def cancel(self, handle, overlap):
        self.kernel.cancel_requested = True
        return False


class Kernel:
    actual = False
    def __init__(self, clock, fault=None):
        self.clock, self.fault = clock, fault
        self.held, self.states = {}, {}
        self.birth_attempts, self.termination_attempts = set(), set()
        self.pending_io, self.pending_attributes = [], []
        self.outer = self.inner = None
        self.outer_stop_requested = False
        self.incoming, self.outgoing, self.buffers, self.threads = {}, {}, {}, {}
        self.errors, self.resumes = [], []
        self.copies = {'process': 333, 'thread': 334, 'inner': 335}
        self.Udata = {'stdout': bytearray(b'uv 0.0.1 (controlled)\n'), 'stderr': bytearray()}
        self.cancel_requested, self.gate = False, None
        self.k = FileKernel(self)
    def wall(self): return self.clock.wall()
    def new_job(self, count):
        if count == 3: self.outer = 33
        else: self.inner = 11
        return self.outer if count == 3 else self.inner
    def job_limits(self, count):
        peak = 2 * m.GIB if self.fault == 'commit' else 3 * 1048576
        return {'limit_flags': m.JOB_FLAGS, 'ActiveProcessLimit': count,
                'JobMemoryLimit': 2 * m.GIB, 'peak_commit': peak}
    def create(self, role, gate):
        self.gate = gate
        assert role not in self.birth_attempts
        self.birth_attempts.add(role)
        info = SimpleNamespace(hProcess=100 + len(self.held), hThread=200 + len(self.held),
                               dwProcessId=500 + len(self.held), dwThreadId=600 + len(self.held))
        self.held[role] = info
        self.states[role] = {'signal': False, 'code': None}
        if self.fault == 'partial_birth' and role == 'U':
            info.hThread = None
            raise m.Refused('controlled partial birth with stable alive handle')
        if self.fault == 'ambiguous_birth' and role == 'U':
            del self.held[role]; del self.states[role]
            raise m.Refused('controlled attempted birth no returned identity')
        if role != 'U':
            self.incoming[role], self.outgoing[role] = queue.Queue(), queue.Queue()
            self.buffers[role] = bytearray()
        return self.identity(role)
    def identity(self, role):
        if self.fault == 'first_identity' and role == 'U':
            raise m.Refused('controlled first identity unavailable')
        info = self.held[role]
        argv = [m.UV, '--version'] if role == 'U' else [m.PYTHON, '-I', '-S', '-B', str(P / 'caller_observer_integration.py'), '--role', role]
        result = {'pid': info.dwProcessId, 'creation_time': '100000' + role,
                  'image': m.UV if role == 'U' else m.PYTHON,
                  'command_hash': m.digest(argv), 'command_hash_kind': 'REQUESTED_ARGV_NOT_OBSERVED_OS_COMMAND_LINE'}
        if self.fault == 'U_identity_drift' and role == 'U' and 'U' in self.resumes:
            result['creation_time'] = 'changed'
        return result
    def in_job(self, role, count):
        if self.fault == 'membership' and role == 'U':
            raise m.Refused('controlled missing birth membership')
        assert role in self.held
    def members(self, count):
        values = [self.held[r].dwProcessId for r in self.held if not self.states[r]['signal'] and (count == 3 or r == 'U')]
        if self.fault == 'missing_empty' and count == 1 and 'U' in self.held:
            values = [self.held['U'].dwProcessId]
        return values
    def copies_for_c(self): return dict(self.copies)
    def parent_identity(self):
        return {'pid': 1, 'creation_time': 'controlled-K', 'image': m.PYTHON}
    def verify_copies(self, returned):
        m.need(returned['copies'] == self.copies and returned['identity'] == self.identity('U') and
               returned['thread_id'] == self.held['U'].dwThreadId, 'controlled received handle identity')
    def resume(self, role):
        self.in_job(role, 3); self.resumes.append(role)
        if role == 'U':
            if self.fault == 'K_loss':
                raise m.Refused('controlled K loss: no completed return/crash recovery')
            if self.fault not in {'alive_termination', 'missing_signal'}:
                self.states['U'] = {'signal': True, 'code': 0}
            return
        if self.fault == role + '_loss':
            self.states[role] = {'signal': True, 'code': 2}
            return
        thread = threading.Thread(target=self.role, args=(role,), daemon=True)
        self.threads[role] = thread; thread.start()
    def role(self, role):
        channel = QueueChannel(self.incoming[role], self.outgoing[role])
        try:
            raw = channel.receive(); boot = json.loads(raw[4:])
            provider = FixtureAdmission(boot['gate'], self.fault)
            if self.fault in {'nonce', 'sequence', 'actual_role'} and role == 'C':
                if self.fault == 'nonce': boot['nonce'] = '0' * 64
                elif self.fault == 'sequence':
                    original_send = channel.send
                    def bad_sequence(raw):
                        frame = json.loads(raw[4:]); frame['sequence'] = 1
                        encoded = m.canonical(frame)
                        original_send(len(encoded).to_bytes(4, 'little') + encoded)
                    channel.send = bad_sequence
                else: boot['identity'] = dict(boot['identity'], creation_time='wrong')
            if role == 'C':
                guard = lambda: lib.Deadline(self.clock, NoTimer)
                m.role_entry(role, self, channel, boot, fixture_admission=provider, lib=lib,
                    clock=self.clock, sleep=time.sleep, guard_factory=guard, wall_clock=self.wall)
            else:
                def store_factory():
                    store = ops.OperationsStore(boot['gate']['db_path'], boot['gate']['receipt_dir'], clock=self.wall)
                    original = store.mark_running
                    def mark(*args):
                        if self.fault == 'blocked_before':
                            self.clock.advance(10.1); time.sleep(0.01)
                            raise m.Refused('controlled blocked-before-COMMIT response')
                        if self.fault == 'before_commit': raise m.Refused('controlled before-COMMIT failure')
                        value = original(*args)
                        if self.fault == 'blocked_after':
                            self.clock.advance(10.1); time.sleep(0.01)
                            raise m.Refused('controlled blocked-after-COMMIT response')
                        if self.fault == 'after_commit': raise m.Refused('controlled after-COMMIT response loss')
                        if self.fault == 'stale_return': self.clock.advance(120)
                        return value
                    store.mark_running = mark
                    return store
                m.role_entry(role, self, channel, boot, fixture_admission=provider,
                             fixture_store_factory=store_factory, clock=self.clock, wall_clock=self.wall)
            self.states[role] = {'signal': True, 'code': 0}
        except BaseException as exc:
            self.errors.append(role + ': ' + type(exc).__name__ + ': ' + str(exc))
            self.states[role] = {'signal': True, 'code': 2}
    def snapshot(self, role):
        if self.fault == 'missing_peak' and role == 'U':
            raise m.Refused('controlled exact peak missing after exit')
        state = {'signal': False, 'code': None} if role == 'K' else self.states[role]
        peak = m.GIB if self.fault == 'RSS' else 1048576
        cpu = 11 if self.fault == 'CPU' and role == 'U' else 0.002 + self.clock() * 0.001
        return {'rss_bytes': peak // 2, 'peak_rss_bytes': peak,
                'private_commit_bytes': 1048576, 'peak_private_commit_bytes': 1048576,
                'CPU_seconds': cpu, 'signal': state['signal'], 'exit_code': state['code'], 'sampled_at': self.clock()}
    def read(self, role, name, maximum):
        if role == 'U':
            data = self.Udata[name]; raw = bytes(data[:maximum]); del data[:maximum]
            eof = not raw and not data and self.states['U']['signal']
            if self.fault == 'missing_EOF':
                eof = False
                if not raw: self.clock.advance(10.1)
            return raw, eof
        if name == 'stderr': return b'', self.states[role]['signal']
        data = self.buffers[role]
        while not self.outgoing[role].empty(): data.extend(self.outgoing[role].get_nowait())
        raw = bytes(data[:maximum]); del data[:maximum]
        return raw, not raw and not data and self.states[role]['signal']
    def write(self, role, raw):
        if self.fault == 'queued_reply_late' and role == 'C':
            value = json.loads(raw[4:])
            if value.get('operation') == 'RESERVE' and 'value' in value['payload']:
                self.clock.advance(10.1)
        if self.fault == 'queued_reply_stale_lease' and role == 'C':
            value = json.loads(raw[4:])
            if value.get('operation') == 'RESERVE' and 'value' in value['payload']:
                db = ops.OperationsStore(self.gate['db_path'], self.gate['receipt_dir'], clock=self.wall)
                db.db.execute("UPDATE leases SET expires=? WHERE kind='model'", (self.wall() - 1,))
                db.close()
        if self.fault == 'ack' and role == 'C':
            value = json.loads(raw[4:])
            if value.get('operation') == 'BIRTH' and 'value' in value['payload']:
                value['payload']['value']['token_sha256'] = '0' * 64
                raw = m.canonical(value); raw = len(raw).to_bytes(4, 'little') + raw
        self.incoming[role].put(raw)
    def kill_u(self):
        if 'U' in self.states and self.fault != 'alive_termination':
            self.states['U'] = {'signal': True, 'code': self.states['U']['code'] if self.states['U']['signal'] else 2}
    def cleanup_u(self):
        self.kill_u()
        if 'U' in self.states and not self.states['U']['signal']:
            self.termination_attempts.add('U')
            return ['genuinely alive failed exact termination; original remains held']
        return []
    def stop_outer_once(self):
        if self.outer_stop_requested: return
        self.outer_stop_requested = True
        for role in self.states:
            if self.fault == 'alive_termination' and role == 'U': continue
            self.states[role] = {'signal': True, 'code': 2}
        for values in self.incoming.values(): values.put(None)
    def telemetry(self):
        return {'age_seconds': 6 if self.fault == 'stale_host' else 0, 'sampled_at': self.clock(),
                'host_available_bytes': 8 * m.GIB, 'disk_free_bytes': 20 * m.GIB,
                'source': 'controlled host data, not actual qualification'}
    def release(self):
        assert not self.pending_io and not self.members(3) and not self.members(1)
        for role in self.held: assert self.states[role]['signal']
        self.outer = self.inner = None
    def publish_whole(self, directory, name, payload, phase):
        with patch.object(ct, 'get_last_error', lambda: self.k.error):
            return m.Win32Publication(self, directory, name, payload, phase).publish()
    def join(self):
        for t in self.threads.values(): t.join(timeout=1.1)


class WholePath(unittest.TestCase):
    def scenario(self, fault=None):
        clock = Clock()
        with tempfile.TemporaryDirectory(prefix='crs014-controlled-') as folder:
            root = Path(folder); leaf = root / 'leaf'; leaf.mkdir()
            db, receipts = root / 'fixture.sqlite3', root / 'receipts'
            store = ops.OperationsStore(db, receipts, clock=clock.wall)
            lease = store.acquire_lease('controlled-O', 60)
            row = store.register_job('CONTROLLED014', {'fixture': 'same immutable library'}, 'FIXTURE014')
            store.ready(row['key']); store.close()
            gate = {'key': row['key'], 'request_hash': row['request_hash'], 'owner': lease['owner'], 'fence': lease['fence'],
                    'epoch': row['epoch'], 'db_path': str(db), 'receipt_dir': str(receipts), 'leaf_dir': str(leaf),
                    'cwd': str(P / '.venv/tool-version-007/controlled'), 'temp': str(P / '.venv/tool-version-007/controlled-temp'),
                    'source_commit': 'CONTROLLED_NOT_ACTUAL', 'contract_sha256': 'CONTROLLED', 'runtime_pin': 'CONTROLLED',
                    'admission_pin': 'CONTROLLED', 'separately_registered': True, 'source_roles_commit_accepted': True,
                    'loader_documentary_accepted': True, 'trusted_metadata_boundary_accepted': True,
                    'physical_paths_verified': True, 'empty_scoped_paths_verified': True, 'telemetry_age_seconds': 0,
                    'preworker_available_bytes': 8 * m.GIB, 'disk_free_bytes': 20 * m.GIB, 'host_reserve_bytes': 1.25 * m.GIB,
                    'gpu_status': 'UNMEASURED_NOT_QUALIFIED', 'argv': ['--version'], 'tool_path': m.UV,
                    'tool_sha256': m.UV_SHA, 'canonical_operations_sha256': m.OPS_SHA}
            kernel = Kernel(clock, fault)
            engine = m.Integration(kernel, FixtureAdmission(gate, fault), clock=clock, sleep=time.sleep)
            original_leaf = m.durable_leaf
            def write(directory, name, payload):
                ref = original_leaf(directory, name, payload)
                if fault == 'caller_late' and name == 'caller.json': clock.advance(10.1)
                if fault == 'observer_late' and name == 'observer.json': clock.advance(2.1)
                if fault == 'native_cost' and name == 'caller.json':
                    payload['result']['native_elapsed_seconds'] += 1
                    original_leaf(directory, name, payload)
                return ref
            with patch.object(m, 'durable_leaf', write): result = engine.run()
            kernel.stop_outer_once(); kernel.join()
            for writer in engine.writers.values():
                # Fixture transport has returned and controlled roles stopped;
                # end idle ordinary tool threads before the next case.
                if writer.thread.is_alive() and not writer.pending.is_set() and writer.queue.empty():
                    writer.close()
            store = ops.OperationsStore(db, receipts, clock=clock.wall)
            row = store.get_job(gate['key']); store.close()
            result.update(row=row, births=set(kernel.birth_attempts), resumes=list(kernel.resumes),
                live_U='U' in kernel.states and not kernel.states['U']['signal'],
                pending_io=list(kernel.pending_io), cancel_requested=kernel.cancel_requested, role_errors=kernel.errors)
            return result

    def rejected(self, fault):
        result = self.scenario(fault)
        self.assertEqual(result['status'], 'LATE_OR_UNKNOWN_UNACCEPTED', result)
        self.assertFalse(result['retry_allowed'])
        cost = result['failed_phase_cost_evidence']
        self.assertIn(cost['cost_disposition'], {'MEASURED_AT_FAILURE_NOT_FINAL_EXIT', 'PARTIAL_OR_UNKNOWN'})
        self.assertIn('prior_resource_samples', cost)
        self.assertFalse(cost['final_phase_CPU_endpoint_qualified'])
        self.assertTrue(cost['unknown_is_not_zero'])
        return result

    def test_normal_completed_role_library_path(self):
        r = self.scenario()
        self.assertEqual(r['status'], 'ACCEPTED_SOURCE_PATH_SYNTHETIC', r)
        self.assertEqual(r['row']['state'], 'SUCCEEDED')
        self.assertEqual(r['births'], {'O', 'C', 'U'})
        self.assertEqual(r['resumes'].count('U'), 1)
        self.assertGreater(r['native_seconds'], 0)
        self.assertLessEqual(r['caller_wall_seconds'], 10)
        self.assertLessEqual(r['observer_wall_seconds'], 2)
        self.assertFalse(r['actual_qualification_from_fixture'])
    def test_admission_refuses_before_any_birth(self):
        self.assertEqual(self.rejected('admission')['births'], set())
    def test_partial_birth_retained_full(self):
        self.assertEqual(self.rejected('partial_birth')['row']['reserved_seconds'], 10)
    def test_ambiguous_birth_never_zero(self):
        self.assertIsNone(self.rejected('ambiguous_birth')['row']['actual_seconds'])
    def test_first_identity_retains_exposure(self):
        self.assertEqual(self.rejected('first_identity')['row']['reserved_seconds'], 10)
    def test_nonce_mismatch(self): self.rejected('nonce')
    def test_sequence_mismatch(self): self.rejected('sequence')
    def test_actual_C_identity_mismatch(self): self.rejected('actual_role')
    def test_ack_exact_frame(self): self.rejected('ack')
    def test_C_loss(self): self.rejected('C_loss')
    def test_O_loss(self): self.rejected('O_loss')
    def test_genuinely_alive_failed_termination(self):
        r = self.rejected('alive_termination'); self.assertTrue(r['live_U'])
        self.assertEqual(r['row']['reserved_seconds'], 10)
    def test_before_commit_unknown_stop(self):
        self.assertIsNone(self.rejected('before_commit')['row']['actual_seconds'])
    def test_after_commit_same_identity_return(self):
        r = self.scenario('after_commit')
        self.assertEqual(r['status'], 'ACCEPTED_SOURCE_PATH_SYNTHETIC', r)
        self.assertEqual(r['births'], {'O', 'C', 'U'})
    def test_blocked_before_commit_K_stop(self):
        self.assertIsNone(self.rejected('blocked_before')['row']['actual_seconds'])
    def test_blocked_after_commit_K_stop(self):
        self.assertIsNone(self.rejected('blocked_after')['row']['actual_seconds'])
    def test_stale_commit_response_return(self): self.rejected('stale_return')
    def test_missing_EOF(self): self.rejected('missing_EOF')
    def test_missing_signal(self): self.rejected('missing_signal')
    def test_post_exit_peak_unavailable(self): self.rejected('missing_peak')
    def test_membership_absent(self): self.rejected('membership')
    def test_empty_membership_absent(self): self.rejected('missing_empty')
    def test_RSS_combined_excess(self): self.rejected('RSS')
    def test_commit_aggregate_excess(self): self.rejected('commit')
    def test_CPU_excess(self): self.rejected('CPU')
    def test_current_host_stale(self): self.rejected('stale_host')
    def test_caller_durable_leaf_crosses10(self): self.rejected('caller_late')
    def test_observer_durable_leaf_crosses2(self):
        self.assertEqual(self.rejected('observer_late')['row']['state'], 'SUCCEEDED')
    def test_whole_publication_crosses2_native_success_immutable(self):
        r = self.rejected('whole_late'); self.assertEqual(r['row']['state'], 'SUCCEEDED')
        self.assertGreater(r['row']['actual_seconds'], 0)
    def test_unknown_cancel_retains_IO_owner(self):
        r = self.rejected('pending_cancel')
        self.assertTrue(r['pending_io'] and r['cancel_requested'])
        self.assertTrue(r['pending_io'][0].buffer and r['pending_io'][0].overlapped)
    def test_whole_readback_corruption(self): self.rejected('whole_readback')
    def test_native_cost_immutable_conflict(self): self.rejected('native_cost')
    def test_K_loss_containment_not_durable_recovery(self):
        r = self.rejected('K_loss')
        self.assertFalse(r['K_crash_durable_recovery_claimed'])
        self.assertIsNone(r['row']['actual_seconds'])
    def test_actual_admission_cannot_take_fixture_boolean(self):
        with self.assertRaises(m.Refused):
            m.ActualAdmission('absent', 'absent', '0' * 64).verify(Kernel(Clock()))
    def test_role_authority_precedes_store_mutation(self):
        r = self.rejected('role_authority')
        self.assertEqual(r['row']['state'], 'READY')
        self.assertNotIn('U', r['births'])
    def test_final_K_CLI_refuses_before_constructing_owner(self):
        with patch('sys.argv', ['caller_observer_integration.py', '--capsule', 'bogus',
                                '--grant', 'bogus', '--grant-sha256', '0' * 64]), \
             patch.object(m, 'actual_dispatch', side_effect=AssertionError('must not construct K')):
            self.assertEqual(m.main(), 2)
    def test_unknown_API_failure_retains_exact_strong_owner(self):
        r = self.rejected('alive_termination')
        self.assertIsInstance(r['recovery_owner'], m.Integration)
        self.assertTrue(r['recovery_owner'].kernel.held['U'].hProcess)
    def test_absolute_observer_timer_includes_transition_delay(self):
        clock = Clock(); TC = clock(); clock.advance(0.3)
        calls = []
        class Timer(NoTimer):
            def __init__(self, seconds, fn):
                calls.append(seconds); super().__init__(seconds, fn)
        guard = m.PhaseGuard(Kernel(clock), clock, 2, 'observer', started=TC, timer_factory=Timer)
        self.assertLess(calls[0], 1.71)
        self.assertEqual(guard.until, TC + 2)
        guard.close()
    def test_required_blob_coverage_empty_refused(self):
        with self.assertRaises(m.Refused): m.validate_blob_coverage([], [])
    def test_required_blob_coverage_incomplete_refused(self):
        path = str(P / 'caller_observer_integration.py')
        pin = {'path': path, 'bytes': 1, 'sha256': 'a' * 64}
        blob = {'repository_path': (Path('experiments') / P.name / 'caller_observer_integration.py').as_posix(),
                'physical_path': path, 'sha256': 'a' * 64, 'blob_oid': 'a' * 40}
        with self.assertRaises(m.Refused): m.validate_blob_coverage([blob], [pin])
    def test_required_blob_duplicate_refused(self):
        path = str(P / 'caller_observer_integration.py')
        pin = {'path': path, 'bytes': 1, 'sha256': 'a' * 64}
        blob = {'repository_path': (Path('experiments') / P.name / 'caller_observer_integration.py').as_posix(),
                'physical_path': path, 'sha256': 'a' * 64, 'blob_oid': 'a' * 40}
        with self.assertRaises(m.Refused): m.validate_blob_coverage([blob, blob], [pin])
    def test_queued_reply_actual_completion_late_refused(self): self.rejected('queued_reply_late')
    def test_queued_reply_actual_consumption_stale_lease_refused(self): self.rejected('queued_reply_stale_lease')
    def test_failed_CPU_excess_preserved_not_clipped(self):
        r = self.rejected('CPU')
        costs = r['failed_phase_cost_evidence']
        self.assertGreater(costs['known_role_CPU_seconds']['U'], 10)
        self.assertTrue(costs['known_overrun_not_clipped'])
    def test_failed_post_stop_peak_unknown_retains_prior_measurements(self):
        r = self.rejected('missing_peak')
        costs = r['failed_phase_cost_evidence']
        self.assertEqual(costs['cost_disposition'], 'PARTIAL_OR_UNKNOWN')
        self.assertIn('U', costs['post_stop_unavailable'])
        self.assertTrue(costs['prior_resource_samples'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
