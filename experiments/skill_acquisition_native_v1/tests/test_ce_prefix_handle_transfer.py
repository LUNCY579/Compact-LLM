"""Fake Windows ABI and pure binding checks; no OS/child/native execution."""
import copy
import ctypes
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_prefix_handle_transfer as t
from . import test_ce_checkpoint_local_issuer as base


def fixture(root):
    common, request, _, current, delegation, _ = base.fixture(root)
    request['branch'] = 'resumed'
    prefix = copy.deepcopy(current); prefix['os_identity'].update(pid=42, creation_filetime_100ns=110)
    prefix['sources']['request_ref'] = {'path': str(root / 'prefix-request.json'), 'sha256': '8' * 64, 'bytes': 12}
    sources, identity = prefix['sources'], prefix['os_identity']
    identity['argv'] = [sources['process_image_ref']['path'], '-m', 'fixture', '--request', sources['request_ref']['path']]
    prefix['worker_identity_sha256'] = t.digest(t.worker.worker_identity_preimage(identity, sources,
        common['request_sha256'], common['scope_sha256']))
    terminal = {'schema': t.worker.TERMINAL, 'os_identity': identity, 'worker_identity_sha256': prefix['worker_identity_sha256'],
        'request_sha256': common['request_sha256'], 'scope_sha256': common['scope_sha256'], 'exit_code': 0,
        'exit_filetime_100ns': 130, 'status_file_used': False, 'native_qualified': False}
    spec = {'schema': t.worker.SCOPE, 'mode': 'registered_metadata', 'request_sha256': common['request_sha256'],
        'scope_sha256': common['scope_sha256'], 'expected_pid': 42, 'expected_creation_filetime_100ns': 110,
        'launch_not_before_filetime_100ns': 110, 'launch_not_after_filetime_100ns': 110,
        'expected_boot_identity': identity['boot_identity'], 'source_checkout': str(root), 'argv': identity['argv'],
        'source_files': sources['source_files'], 'approved_code_sha256': common['bindings']['source']['code_sha256'],
        'python_ref': sources['python_ref'], 'process_image_ref': sources['process_image_ref'], 'request_ref': sources['request_ref'],
        'entry_module': 'fixture', 'bounds': {'operation_attempts_upper': 100, 'io_read_bytes_upper': 10000,
            'identity_hash_attempts_upper': 100, 'identity_hash_bytes_upper': 10000, 'wall_seconds_upper': 100}}
    manifest_ref = {'path': str(root / 'prefix.json'), 'sha256': '9' * 64, 'bytes': 100}
    request['lineage_declaration'].update(prefix_manifest_ref=manifest_ref,
        prefix_manifest={'bindings': {'qualification_evidence_binding': {'worker_identity_sha256': prefix['worker_identity_sha256']}}})
    record = {'schema': t.SCHEMA, 'branch': 'resumed', 'remote_handle': 501, 'desired_access': t.READ_ONLY,
        'inheritable': False, 'request_sha256': common['request_sha256'], 'scope_sha256': common['scope_sha256'],
        'target_proof': current, 'target_request_ref': current['sources']['request_ref'],
        'parent_os_identity': delegation['parent_os_identity'], 'parent_worker_identity_sha256': delegation['parent_worker_identity_sha256'],
        'prefix_terminal_proof': terminal, 'prefix_live_proof': prefix, 'prefix_observation_spec': spec,
        'prefix_manifest_ref': manifest_ref, 'native_qualified': False}
    return common, request, current, delegation, record


class Tests(unittest.TestCase):
    def test_native_sender_and_receiver_refuse_before_objects_os_or_file_work(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('unapproved input/OS')
        with patch.object(Path, 'open', side_effect=AssertionError('unapproved file')):
            with self.assertRaises(ValueError): t.transfer_prefix(Trap(), None, Trap(), Trap(), Trap(), Trap(), Trap(), Trap(), Trap())
            with self.assertRaisesRegex(ValueError, 'verifier remains unregistered'):
                t.prefix_terminal_handle(Trap(), Trap(), True, Trap(), controller_request=Trap(), service_spec=Trap(),
                    current_handle=Trap(), observation_permit=Trap(), delegation=Trap())
        self.assertFalse(t.worker._REGISTERED_METADATA_VERIFIERS)

    def test_actual_duplicate_abi_requests_only_readonly_noninheritable_access(self):
        calls = []
        def get_current(): return -1
        def duplicate(source, handle, target, result, access, inherit, flags):
            calls.append((source, handle, target, access, inherit, flags)); result._obj.value = 501; return 1
        api = SimpleNamespace(ctypes=ctypes, kernel=SimpleNamespace(GetCurrentProcess=get_current, DuplicateHandle=duplicate))
        self.assertEqual(t.duplicate_readonly(api, 101, 202), 501)
        self.assertEqual(calls, [(-1, 101, 202, t.READ_ONLY, False, 0)])

    def test_real_access_query_rejects_writable_inherited_protected_or_unqueryable_handle(self):
        for access, flags, success in ((t.READ_ONLY, 0, True), (t.READ_ONLY | 1, 0, False),
                (t.READ_ONLY, 1, False), (t.READ_ONLY, 2, False)):
            def information(handle, pointer): pointer._obj.value = flags; return 1
            def query(handle, kind, pointer, size, length):
                pointer._obj.access = access; length._obj.value = size; return 0
            api = SimpleNamespace(ctypes=ctypes, kernel=SimpleNamespace(GetHandleInformation=information),
                                  ntdll=SimpleNamespace(NtQueryObject=query))
            with self.subTest(access=access, flags=flags):
                if success: self.assertEqual(t.handle_access(api, 501), {'granted_access': t.READ_ONLY, 'handle_flags': 0})
                else:
                    with self.assertRaisesRegex(ValueError, 'read-only/noninheritable/closable'): t.handle_access(api, 501)

    def test_binding_accepts_exact_prefix_target_views_and_rejects_reuse_foreign_refs_or_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            common, request, current, delegation, record = fixture(Path(directory))
            self.assertIs(t.validate_record(record, common, request, current, delegation), record)
            observed = copy.deepcopy(current); observed['sources']['launcher_provenance'] = 'child-current view'
            t.validate_record(record, common, request, observed, delegation)
            for change in ('target_create', 'target_request', 'prefix_create', 'prefix_source', 'exit', 'manifest', 'rights', 'boolean_handle'):
                altered = copy.deepcopy(record)
                if change == 'target_create': altered['target_proof']['os_identity']['creation_filetime_100ns'] += 1
                elif change == 'target_request': altered['target_request_ref']['sha256'] = '0' * 64
                elif change == 'prefix_create': altered['prefix_observation_spec']['expected_creation_filetime_100ns'] += 1
                elif change == 'prefix_source': altered['prefix_live_proof']['sources']['source_files']['synthetic-fixture']['sha256'] = '0' * 64
                elif change == 'exit': altered['prefix_terminal_proof']['exit_code'] = 1
                elif change == 'manifest': altered['prefix_manifest_ref']['sha256'] = '0' * 64
                elif change == 'rights': altered['desired_access'] |= 1
                else: altered['remote_handle'] = True
                with self.subTest(change=change), self.assertRaises(ValueError):
                    t.validate_record(altered, common, request, current, delegation)

    def test_parent_original_clone_close_is_independent_of_target_numeric_values(self):
        from experiments.skill_acquisition_native_v1 import ce_checkpoint_processes as process
        current = SimpleNamespace(close=lambda: calls.append('original clone closed'))
        calls = []; owner = object.__new__(process.CheckpointProcesses); owner.prefix_terminal_handle = current
        owner.close_prefix_transfer(); owner.close_prefix_transfer()
        self.assertEqual(calls, ['original clone closed'])
        self.assertIsNone(owner.prefix_terminal_handle)


if __name__ == '__main__': unittest.main()
