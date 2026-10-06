"""V2 managed-only scope with complete inert execution evidence.

All selected .dll/.pyd/.exe bytes are inert text. Observer/module/directory/WinAPI
facts are explicit CPU suppliers. No real native/framework/OS APIs or Python DLL
notification callbacks are called. Finalized event logs must survive close().
"""
from copy import deepcopy
import importlib.util
import types
import unittest
from unittest.mock import Mock, patch
import test_library_policy as legacy


class ManagedSupplier(legacy.Supplier):
    def scope(self):
        self.calls.append(('scope',))
        result = {'kind': 'OBSERVED_MANAGED_DLL_DIRECTORY_SCOPE', 'managed_setup_complete': True,
                  'directories': list(self.policy['search_directories']),
                  'system_directory': self.policy['system_directory'], 'default_flags': 0xC00,
                  'legacy_directory': '', 'foreign_directory_registration': 'UNOBSERVED',
                  'evidence_basis': 'explicit_injected_CPU_managed_observation', 'closed': False}
        result.update(self.scope_changes)
        return result
    def close(self):
        self.calls.append(('close',)); self.closed = True
        if self.close_error: raise ValueError('injected managed cookie removal refusal')
        result = {'kind': 'OBSERVED_MANAGED_DLL_DIRECTORY_CLEANUP', 'managed_cleanup_complete': True,
                  'remaining_managed_directories': [], 'legacy_directory': '',
                  'all_managed_cookies_removed': True,
                  'foreign_directory_registration': 'UNOBSERVED', 'closed': True}
        result.update(self.cleanup_changes)
        return result


class ManagedScopeTests(unittest.TestCase):
    def setUp(self):
        self.f = legacy.LibraryBoundaryTests('runTest'); self.f.setUp()
        self.addCleanup(self.f.doCleanups)
        self.B = self.f.B; self.f.policy['kind'] = self.B.MANAGED_KIND
        self.f.supplier = ManagedSupplier(self.f.policy)
    def guard(self):return self.f.guard()
    def failed_close(self, guard):
        with self.assertRaisesRegex(ValueError, 'cleanup failed|cannot publish completion'): guard.close()
        self.assertTrue(guard.closed); self.assertTrue(self.f.supplier.closed)
    def test_V2_actual_registered_extension_create_exec_and_finalized_drain(self):
        guard = self.guard(); loader = self.f.loader(guard)
        spec = importlib.util.spec_from_loader('CPU_inert_extension', loader)
        with patch.object(loader.actual, 'create_module', side_effect=self.f.delegate_create), \
                patch.object(loader.actual, 'exec_module', return_value=None):
            module = loader.create_module(spec); loader.exec_module(module)
        result = guard.close()
        self.assertEqual(result['policy_kind'], self.B.MANAGED_KIND)
        self.assertEqual(result['foreign_directory_registration'], 'UNOBSERVED')
        self.assertTrue(result['fixture_only']); self.assertFalse(result['native_PASS'])
        self.assertFalse(result['native_authority_issued']); self.assertFalse(result['native_qualification_proven'])
        close_index = self.f.supplier.calls.index(('close',))
        self.assertIn(('events',), self.f.supplier.calls[close_index+1:])
        self.assertIn(('libraries',), self.f.supplier.calls[close_index+1:])
    def test_foreign_scope_false_or_absent_claims_and_universal_complete_refused(self):
        for changes in ({'foreign_directory_registration': False},
                        {'foreign_directory_registration': 'ABSENT'}, {'complete': True},
                        {'unmanaged_directory_registration': False}):
            with self.subTest(changes=changes):
                self.f.supplier = ManagedSupplier(self.f.policy); self.f.supplier.scope_changes = changes
                with self.assertRaisesRegex(ValueError, 'managed DLL search scope'): self.guard()
                self.assertTrue(self.f.supplier.closed)
    def test_managed_scope_boolean_integer_impostors_refused(self):
        for field, value in (('managed_setup_complete', 1), ('closed', 0)):
            with self.subTest(field=field):
                self.f.supplier = ManagedSupplier(self.f.policy); self.f.supplier.scope_changes[field] = value
                with self.assertRaisesRegex(ValueError, 'managed DLL search scope'): self.guard()
    def test_V1_scope_does_not_silently_satisfy_V2(self):
        self.f.supplier = legacy.Supplier(self.f.policy)
        with self.assertRaisesRegex(ValueError, 'cleanup failed'): self.guard()
    def test_wrong_directories_flags_or_legacy_directory_refused(self):
        for changes in ({'directories': list(reversed(self.f.policy['search_directories']))},
                        {'default_flags': 0}, {'legacy_directory': 'CPU_unmanaged'},
                        {'evidence_basis': 'Win32_managed_directory_operations'}):
            with self.subTest(changes=changes):
                self.f.supplier = ManagedSupplier(self.f.policy); self.f.supplier.scope_changes = changes
                with self.assertRaisesRegex(ValueError, 'managed DLL search scope'): self.guard()
    def test_incomplete_events_or_snapshot_still_refuse_initialization(self):
        for name in ('event_changes', 'snapshot_changes'):
            with self.subTest(name=name):
                self.f.supplier = ManagedSupplier(self.f.policy)
                getattr(self.f.supplier, name)['complete'] = False
                with self.assertRaises(ValueError): self.guard()
                self.assertTrue(self.f.supplier.closed)
    def test_unregistered_transient_load_unload_still_poisoned(self):
        guard = self.guard()
        with self.assertRaises(ValueError):
            with guard.extension(self.f.ext, 'create'):
                self.f.delegate_create(); self.f.supplier.add(self.f.extra); self.f.supplier.unload(self.f.extra)
        self.assertTrue(guard.failed); self.failed_close(guard)
    def test_snapshot_without_complete_event_identity_still_refused(self):
        guard = self.guard()
        with self.assertRaises(ValueError):
            with guard.extension(self.f.ext, 'create'):
                self.f.delegate_create(); self.f.supplier.event_rows.pop()
        self.assertTrue(guard.failed); self.failed_close(guard)
    def test_final_event_log_unavailable_after_close_prevents_completion(self):
        guard = self.guard(); original = self.f.supplier.events
        def events():
            result = original()
            if self.f.supplier.closed: result['complete'] = False
            return result
        self.f.supplier.events = events
        with self.assertRaisesRegex(ValueError, 'cleanup failed'): guard.close()
        self.assertTrue(guard.failed); self.assertTrue(guard.closed)
        self.assertTrue(guard.failed)
    def test_cleanup_foreign_overclaims_and_boolean_integer_impostors_refused(self):
        changes = ({'foreign_directory_registration': False}, {'complete': True},
                   {'all_cookies_removed': True}, {'managed_cleanup_complete': 1},
                   {'all_managed_cookies_removed': 1}, {'closed': 1},
                   {'remaining_managed_directories': ['CPU_residual']})
        for change in changes:
            with self.subTest(change=change):
                self.f.supplier = ManagedSupplier(self.f.policy); guard = self.guard()
                self.f.supplier.cleanup_changes = change
                with self.assertRaisesRegex(ValueError, 'cleanup failed'): guard.close()
                self.assertTrue(guard.failed)
    def test_provider_bytes_drift_still_prevents_completion(self):
        guard = self.guard()
        from pathlib import Path
        Path(self.f.policy['provider_source']['path']).write_bytes(b'# changed provider after guard start\n')
        with self.assertRaisesRegex(ValueError, 'selected library/source bytes changed'): guard.close()
        self.assertTrue(guard.failed); self.assertTrue(guard.closed)
    def test_native_V1_guard_and_direct_provider_refuse_before_OS_import(self):
        C = types.SimpleNamespace(stage_native_gate=lambda: None,
            verify_native_library_event_observer=lambda observer: None)
        old = deepcopy(self.f.policy); old['kind'] = self.B.KIND
        with patch.object(self.B, 'ManagedWin32LibraryProvider', side_effect=AssertionError('must not construct')):
            with self.assertRaisesRegex(ValueError, 'legacy V1 native provider refused'):
                self.B.make_native_guard(old, C)
        def old_trace(stage):
            return {'kind': 'VERIFIED_CURRENT_DLL_DIRECTORY_BOOTSTRAP_TRACE', 'stage': stage,
                    'fresh_process_identity_verified': True, 'complete_registration_trace': True,
                    'unmanaged_directory_registration': False, 'provider_source_verified': True}
        with self.assertRaisesRegex(ValueError, 'legacy V1 native provider refused'):
            self.B.Win32LibraryProvider(C, old_trace, object())
    def test_native_V2_stage_gate_refuses_before_policy_paths_or_observer(self):
        C = types.SimpleNamespace(stage_native_gate=lambda: (_ for _ in ()).throw(ValueError('native closed')))
        with self.assertRaisesRegex(ValueError, 'native closed'): self.B.make_native_guard(None, C)
        with self.assertRaisesRegex(ValueError, 'native closed'): self.B.ManagedWin32LibraryProvider(C)
    @staticmethod
    def provenance(stage):
        return {'kind': 'VERIFIED_CURRENT_LIBRARY_BOOTSTRAP_PROVENANCE', 'stage': stage,
                'fresh_process_identity_verified': True, 'provider_source_verified': True,
                'preloaded_sources_verified': True, 'module_event_scope_provenance_verified': True}
    def test_V2_provenance_has_no_foreign_trace_and_all_flags_are_exact_booleans(self):
        provider = object.__new__(self.B.ManagedWin32LibraryProvider)
        provider.bootstrap_verifier = self.provenance
        provider._bootstrap('CPU_source_only')
        for field in ('fresh_process_identity_verified', 'provider_source_verified',
                      'preloaded_sources_verified', 'module_event_scope_provenance_verified'):
            for value in (False, 1):
                with self.subTest(field=field, value=value):
                    provider.bootstrap_verifier = lambda stage: {**self.provenance(stage), field: value}
                    with self.assertRaisesRegex(ValueError, 'bootstrap provenance'): provider._bootstrap('CPU_source_only')
        provider.bootstrap_verifier = lambda stage: {**self.provenance(stage), 'complete_registration_trace': True}
        with self.assertRaisesRegex(ValueError, 'bootstrap provenance'): provider._bootstrap('CPU_source_only')
    def test_actual_V2_scope_and_cleanup_source_use_only_inert_API_suppliers(self):
        provider = object.__new__(self.B.ManagedWin32LibraryProvider)
        provider.directories = list(self.f.policy['search_directories']); provider.system_directory = self.f.policy['system_directory']
        provider.closed = False; provider.cookies = [('CPU_cookie_a', 101), ('CPU_cookie_b', 102)]
        removed = []; finalized = []
        provider.ctypes = types.SimpleNamespace(set_last_error=lambda number: None)
        provider.kernel = types.SimpleNamespace(GetDllDirectoryW=object(),
            RemoveDllDirectory=lambda cookie: removed.append(cookie) or True)
        provider._string = lambda function: ''; provider.bootstrap_verifier = self.provenance
        provider.library_event_observer = types.SimpleNamespace(close=lambda: finalized.append(True) or
            {'kind': 'OBSERVED_PROCESS_LIBRARY_EVENT_CLEANUP', 'complete': True, 'closed': True})
        scope = provider.scope()
        self.assertEqual(scope['foreign_directory_registration'], 'UNOBSERVED')
        self.assertNotIn('complete', scope); self.assertNotIn('unmanaged_directory_registration', scope)
        cleanup = provider.close()
        self.assertEqual(cleanup['foreign_directory_registration'], 'UNOBSERVED')
        self.assertNotIn('complete', cleanup); self.assertEqual(removed, [102, 101]); self.assertEqual(finalized, [True])
        self.assertEqual(provider.cookies, []); self.assertTrue(cleanup['all_managed_cookies_removed'])


if __name__ == '__main__': unittest.main()
