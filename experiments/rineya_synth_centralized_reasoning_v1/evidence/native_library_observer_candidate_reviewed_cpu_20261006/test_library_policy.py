"""Actual shared source guard + actual registered-loader CPU controls.

Selected DLLs are inert text files. All directory/module observations are
explicit suppliers. No native library, framework, model or host API is called.
"""
import builtins
from copy import deepcopy
import hashlib
import importlib.machinery
import importlib.util
import json
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent


def descriptor(raw, role='python_source'):
    return {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'role': role}


def source(path):
    raw = path.read_bytes()
    module = types.ModuleType('CPU_library_' + path.stem)
    module.__file__ = str(path)
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    module.__compiled_source_descriptor__ = descriptor(raw)
    return module


class Contract:
    @staticmethod
    def require(value, message):
        if not value:
            raise ValueError(message)
    @staticmethod
    def canonical(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    parse = staticmethod(json.loads)
    @staticmethod
    def file_descriptor(path, expected, *, budget=None):
        path = Path(path)
        before = path.stat()
        raw = path.read_bytes()
        after = path.stat()
        Contract.require((before.st_size, before.st_mtime_ns, before.st_ino) ==
                         (after.st_size, after.st_mtime_ns, after.st_ino) and
                         descriptor(raw, expected['role']) == expected,
                         'actual selected library/source bytes changed')
    @staticmethod
    def read_bytes(path, *, budget=None):
        return Path(path).read_bytes()


class Supplier:
    """Injected observations; these rows never describe host DLL state."""
    def __init__(self, policy):
        self.policy = deepcopy(policy)
        self.rows = {p: index + 101 for index, p in enumerate(policy['initial_loaded_libraries'])}
        self.calls = []
        self.closed = False
        self.scope_changes = {}
        self.snapshot_changes = {}
        self.event_rows = []
        self.event_changes = {}
        self.cleanup_changes = {}
        self.close_error = False
    def begin(self, directories, system_directory):
        self.calls.append(('begin', directories, system_directory))
        return self.scope()
    def scope(self):
        self.calls.append(('scope',))
        result = {'kind': 'OBSERVED_DLL_DIRECTORY_SCOPE', 'complete': True,
                  'directories': list(self.policy['search_directories']),
                  'system_directory': self.policy['system_directory'],
                  'default_flags': 0xC00, 'legacy_directory': '',
                  'unmanaged_directory_registration': False,
                  'completeness_basis': 'explicit_injected_CPU_observation', 'closed': False}
        result.update(self.scope_changes)
        return result
    def libraries(self):
        self.calls.append(('libraries',))
        result = {'kind': 'OBSERVED_PROCESS_LIBRARY_SNAPSHOT', 'complete': True,
                  'libraries': [{'path': p, 'module_handle': h} for p, h in sorted(self.rows.items())]}
        result.update(self.snapshot_changes)
        return result
    def add(self, path, handle=None):
        self.rows[path] = max(self.rows.values()) + 1 if handle is None else handle
        self.event_rows.append({'sequence': len(self.event_rows) + 1, 'event': 'load',
                                'path': path, 'module_handle': self.rows[path]})
    def unload(self, path):
        self.event_rows.append({'sequence': len(self.event_rows) + 1, 'event': 'unload',
                                'path': path, 'module_handle': self.rows[path]})
        del self.rows[path]
    def events(self):
        self.calls.append(('events',))
        result = {'kind': 'OBSERVED_PROCESS_LIBRARY_EVENTS', 'complete': True,
                  'events': deepcopy(self.event_rows)}
        result.update(self.event_changes)
        return result
    def close(self):
        self.calls.append(('close',)); self.closed = True
        if self.close_error:
            raise ValueError('injected RemoveDllDirectory refusal')
        result = {'kind': 'OBSERVED_DLL_DIRECTORY_CLEANUP', 'complete': True,
                  'remaining_managed_directories': [], 'legacy_directory': '',
                  'all_cookies_removed': True, 'closed': True}
        result.update(self.cleanup_changes)
        return result


class LibraryBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='CPU_inert_DLL_boundary_')
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.runtime = self.root / 'runtime'; self.runtime.mkdir()
        self.system = self.root / 'system'; self.system.mkdir()
        self.sources = self.root / 'sources'; self.sources.mkdir()
        for name in ('prefix_library_policy.py', 'prefix_registered_loader.py'):
            (self.sources / name).write_bytes((HERE / name).read_bytes())
        self.B = source(self.sources / 'prefix_library_policy.py')
        self.L = source(self.sources / 'prefix_registered_loader.py')
        self.C = Contract()
        self.base = str(self.system / 'inert_interpreter.exe')
        self.ext = str(self.runtime / 'inert_extension.pyd')
        self.dep = str(self.runtime / 'inert_dependency.dll')
        self.extra = str(self.runtime / 'inert_extra.dll')
        self.files = {}
        for path, role in ((self.base, 'interpreter_executable'), (self.ext, 'extension_binary'),
                           (self.dep, 'native_library'), (self.extra, 'native_library')):
            raw = ('INERT CPU SELECTED FILE: ' + Path(path).name + '\n').encode()
            Path(path).write_bytes(raw); self.files[path] = descriptor(raw, role)
        self.policy = {'kind': self.B.KIND,
                       'search_directories': [str(self.system), str(self.runtime)],
                       'system_directory': str(self.system), 'library_files': self.files,
                       'initial_loaded_libraries': [self.base],
                       'extension_transitions': {self.ext: {'create': [self.ext, self.dep], 'exec': []}},
                       'framework_transition': [],
                       'provider_source': {'path': str(self.sources / 'prefix_library_policy.py'),
                                           'descriptor': descriptor((self.sources / 'prefix_library_policy.py').read_bytes())}}
        self.supplier = Supplier(self.policy)
    def guard(self, supplier=None, policy=None):
        return self.B._CPU_make_guard(self.policy if policy is None else policy, self.C,
                                     self.supplier if supplier is None else supplier,
                                     _capability=self.B._CPU_library_capability())
    def loader(self, guard):
        actual = importlib.machinery.ExtensionFileLoader('CPU_inert_extension', self.ext)
        return self.L.RegisteredLoader(actual, self.ext, self.files[self.ext], self.C, {}, library_guard=guard)
    def delegate_create(self, *args):
        self.supplier.add(self.ext); self.supplier.add(self.dep)
        return types.ModuleType('CPU_inert_extension')
    def test_guard_runs_on_real_registered_extension_create_and_exec(self):
        guard = self.guard(); loader = self.loader(guard)
        spec = importlib.util.spec_from_loader('CPU_inert_extension', loader)
        with (patch.object(importlib.machinery.ExtensionFileLoader, 'create_module', side_effect=self.delegate_create) as create,
              patch.object(importlib.machinery.ExtensionFileLoader, 'exec_module', return_value=None) as execute):
            module = loader.create_module(spec); loader.exec_module(module)
        create.assert_called_once_with(spec); execute.assert_called_once_with(module)
        self.assertTrue(loader.observed['CPU_inert_extension']['current_module_execution_observed'])
        self.assertEqual([row['boundary'].rsplit(':', 1)[-1] for row in guard.observed], ['create', 'exec'])
        receipt = guard.close()
        self.assertTrue(receipt['fixture_only']); self.assertFalse(receipt['native_PASS'])
        self.assertFalse(receipt['native_authority_issued']); self.assertFalse(receipt['native_qualification_proven'])
        self.assertTrue(self.supplier.closed)
    def test_missing_guard_refuses_before_real_extension_delegate(self):
        loader = self.loader(None)
        with (patch.object(importlib.machinery.ExtensionFileLoader, 'create_module') as create,
              patch.object(importlib.machinery.ExtensionFileLoader, 'exec_module') as execute):
            with self.assertRaisesRegex(ValueError, 'observed DLL boundary'): loader.create_module(None)
            with self.assertRaisesRegex(ValueError, 'observed DLL boundary'): loader.exec_module(types.ModuleType('CPU_inert_extension'))
        create.assert_not_called(); execute.assert_not_called()
    def test_unregistered_dependency_poisoned_after_actual_delegate(self):
        guard = self.guard(); loader = self.loader(guard)
        unregistered = str(self.runtime / 'unregistered.dll'); Path(unregistered).write_bytes(b'inert')
        def delegate(*args):
            self.delegate_create(); self.supplier.add(unregistered)
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module', side_effect=delegate) as create:
            with self.assertRaisesRegex(ValueError, 'unregistered dependent'): loader.create_module(None)
        create.assert_called_once()
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
        self.assertTrue(self.supplier.closed)
    def test_registered_but_unexpected_extra_dependency_rejected(self):
        guard = self.guard()
        with self.assertRaisesRegex(ValueError, 'unplanned actual library load event'):
            with guard.extension(self.ext, 'create'):
                self.delegate_create(); self.supplier.add(self.extra)
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_missing_dependency_rejected_after_delegate(self):
        guard = self.guard()
        with self.assertRaisesRegex(ValueError, 'exact observed dependent-library transition'):
            with guard.extension(self.ext, 'create'): self.supplier.add(self.ext)
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_mismatched_dependent_bytes_refused_after_delegate(self):
        guard = self.guard()
        with self.assertRaisesRegex(ValueError, 'selected library/source bytes changed'):
            with guard.extension(self.ext, 'create'):
                self.delegate_create(); Path(self.dep).write_bytes(b'inert altered dependency')
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_provider_source_drift_refuses_before_delegate(self):
        guard = self.guard(); loader = self.loader(guard)
        Path(self.policy['provider_source']['path']).write_bytes(b'# changed selected guard\n')
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module') as create:
            with self.assertRaisesRegex(ValueError, 'selected library/source bytes changed'): loader.create_module(None)
        create.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_unselected_compiled_guard_refused_before_supplier(self):
        self.B.__compiled_source_descriptor__ = descriptor(b'counterfeit')
        with self.assertRaisesRegex(ValueError, 'actual compiled library guard'): self.guard()
        self.assertEqual(self.supplier.calls, [])
    def test_changed_search_dirs_refuse_before_native_delegate(self):
        guard = self.guard(); loader = self.loader(guard)
        self.supplier.scope_changes['directories'] = [str(self.system)]
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module') as create:
            with self.assertRaisesRegex(ValueError, 'scope changed/incomplete'): loader.create_module(None)
        create.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_incomplete_snapshot_refuses_before_delegate(self):
        guard = self.guard(); loader = self.loader(guard)
        self.supplier.snapshot_changes['complete'] = False
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module') as create:
            with self.assertRaisesRegex(ValueError, 'complete actual loaded-library snapshot'): loader.create_module(None)
        create.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_incomplete_search_scope_refuses_and_cleans_partial_start(self):
        self.supplier.scope_changes['complete'] = False
        with self.assertRaisesRegex(ValueError, 'scope changed/incomplete'): self.guard()
        self.assertTrue(self.supplier.closed)
    def test_initial_extra_library_refused_and_cleans_partial_start(self):
        self.supplier.add(self.extra)
        with self.assertRaisesRegex(ValueError, 'exact complete initial'): self.guard()
        self.assertTrue(self.supplier.closed)
    def test_declared_native_scope_cannot_relabel_CPU_observation(self):
        self.supplier.scope_changes['completeness_basis'] = 'Win32_calls_and_verified_fresh_bootstrap_trace'
        with self.assertRaisesRegex(ValueError, 'scope changed/incomplete'): self.guard()
    def test_unmanaged_directory_registration_refused(self):
        self.supplier.scope_changes['unmanaged_directory_registration'] = True
        with self.assertRaisesRegex(ValueError, 'scope changed/incomplete'): self.guard()
    def test_framework_import_observes_nested_extension_and_direct_dependency(self):
        policy = deepcopy(self.policy); policy['framework_transition'] = [self.extra]
        guard = self.guard(policy=policy)
        finder = self.L.RegisteredFinder({}, self.C, {}, library_guard=guard)
        loader = self.loader(guard)
        with (patch.object(importlib.machinery.ExtensionFileLoader, 'create_module', side_effect=self.delegate_create),
              patch.object(importlib.machinery.ExtensionFileLoader, 'exec_module', return_value=None)):
            with finder.framework_import():
                module = loader.create_module(None); loader.exec_module(module); self.supplier.add(self.extra)
        self.assertEqual(len(guard.observed), 3)
        receipt = guard.close(); self.assertFalse(receipt['native_qualification_proven'])
    def test_framework_guard_refuses_unobserved_dependency(self):
        guard = self.guard()
        with self.assertRaisesRegex(ValueError, 'unplanned actual library load event'):
            with guard.framework_import(): self.supplier.add(self.extra)
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_framework_direct_preload_is_observed_before_nested_extension(self):
        policy = deepcopy(self.policy); policy['framework_transition'] = [self.extra]
        guard = self.guard(policy=policy); loader = self.loader(guard)
        with (patch.object(importlib.machinery.ExtensionFileLoader, 'create_module', side_effect=self.delegate_create),
              patch.object(importlib.machinery.ExtensionFileLoader, 'exec_module', return_value=None)):
            with guard.framework_import():
                self.supplier.add(self.extra)
                module = loader.create_module(None); loader.exec_module(module)
        self.assertEqual(len(guard.observed), 3)
        self.assertFalse(guard.close()['native_PASS'])
    def test_unexpected_preload_refused_before_nested_extension_delegate(self):
        guard = self.guard(); loader = self.loader(guard)
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module') as create:
            with self.assertRaisesRegex(ValueError, 'conservation before nested native delegation'):
                with guard.framework_import():
                    self.supplier.add(self.extra); loader.create_module(None)
        create.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_failing_delegate_retains_actual_post_failure_snapshot(self):
        guard = self.guard(); loader = self.loader(guard)
        def failing_delegate(*args):
            self.delegate_create(); raise RuntimeError('inert native delegate failure')
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module', side_effect=failing_delegate):
            with self.assertRaisesRegex(RuntimeError, 'inert native delegate failure'): loader.create_module(None)
        failures = [row for row in guard.observed if row.get('delegated_boundary_failed')]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0]['new_libraries'], sorted([self.ext, self.dep]))
        self.assertTrue(any(row.get('loader_refusal') for row in guard.observed))
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_source_start_failure_and_cleanup_failure_block_completion(self):
        self.supplier.scope_changes['complete'] = False; self.supplier.close_error = True
        with self.assertRaisesRegex(ValueError, 'cleanup failed'): self.guard()
        self.assertTrue(self.supplier.closed)
    def test_nested_verified_library_loss_refused_at_framework_completion(self):
        guard = self.guard(); loader = self.loader(guard)
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module', side_effect=self.delegate_create):
            with self.assertRaisesRegex(ValueError, 'actual library event/snapshot identity join'):
                with guard.framework_import():
                    loader.create_module(None); del self.supplier.rows[self.dep]
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_source_only_finder_refuses_framework_import(self):
        finder = self.L.RegisteredFinder({}, self.C, {})
        with self.assertRaisesRegex(ValueError, 'before framework import'): finder.framework_import()
    def test_existing_loaded_library_handle_change_refused(self):
        guard = self.guard()
        with self.assertRaisesRegex(ValueError, 'actual library event/snapshot identity join'):
            with guard.framework_import(): self.supplier.rows[self.base] += 999
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_external_library_change_before_delegate_refused(self):
        guard = self.guard(); self.supplier.add(self.extra)
        with self.assertRaisesRegex(ValueError, 'unplanned actual library load event'):
            with guard.framework_import(): self.fail('delegate must not run')
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_duplicate_library_handle_snapshot_refused(self):
        self.supplier.rows[self.extra] = self.supplier.rows[self.base]
        with self.assertRaisesRegex(ValueError, 'unique actual loaded-library'): self.guard()
        self.assertTrue(self.supplier.closed)
    def test_selected_policy_mutation_cannot_change_private_guard(self):
        guard = self.guard()
        self.policy['extension_transitions'][self.ext]['create'] = []
        with guard.extension(self.ext, 'create'): self.delegate_create()
        self.assertFalse(guard.close()['native_PASS'])
    def test_cleanup_failure_blocks_completion_and_removes_finder(self):
        guard = self.guard(); finder = self.L.RegisteredFinder({}, self.C, {}, library_guard=guard)
        self.supplier.close_error = True
        with patch.object(self.L.sys, 'meta_path', [finder]):
            with self.assertRaisesRegex(ValueError, 'cleanup failed'): self.L.remove(finder)
            self.assertNotIn(finder, self.L.sys.meta_path)
        self.assertIsNone(finder.library_cleanup); self.assertTrue(guard.failed)
    def test_incomplete_cleanup_blocks_completion(self):
        guard = self.guard(); self.supplier.cleanup_changes['all_cookies_removed'] = False
        with self.assertRaisesRegex(ValueError, 'cleanup failed'): guard.close()
        self.assertTrue(guard.failed); self.assertFalse(guard.native_PASS)
    def test_cleanup_source_drift_blocks_completion(self):
        guard = self.guard()
        Path(self.policy['provider_source']['path']).write_bytes(b'# changed before cleanup\n')
        with self.assertRaisesRegex(ValueError, 'selected library/source bytes changed'): guard.close()
        self.assertTrue(self.supplier.closed); self.assertTrue(guard.failed)
    def test_native_provider_stage_refuses_before_OS_import(self):
        contract = types.SimpleNamespace(stage_native_gate=lambda: (_ for _ in ()).throw(ValueError('native stage closed')))
        original = builtins.__import__
        attempted = []
        def imports(name, *args, **kwargs):
            if name.split('.', 1)[0] in {'ctypes', 'torch'}:
                attempted.append(name); raise AssertionError('forbidden actual native import')
            return original(name, *args, **kwargs)
        with patch.object(builtins, '__import__', side_effect=imports):
            with self.assertRaisesRegex(ValueError, 'native stage closed'): self.B.Win32LibraryProvider(contract)
        self.assertEqual(attempted, [])
    def test_native_provider_requires_verified_trace_before_OS_import(self):
        contract = types.SimpleNamespace(stage_native_gate=lambda: None)
        with self.assertRaisesRegex(ValueError, 'current fresh-bootstrap'): self.B.Win32LibraryProvider(contract)
        contract.verify_native_library_event_observer = lambda observer: None
        with self.assertRaisesRegex(ValueError, 'bootstrap trace incomplete'):
            self.B.Win32LibraryProvider(contract, lambda stage: True, object())
    def test_CPU_supplier_capability_cannot_be_forged(self):
        with self.assertRaisesRegex(ValueError, 'private CPU library supplier capability'):
            self.B._CPU_make_guard(self.policy, self.C, self.supplier, _capability=object())
        self.assertEqual(self.supplier.calls, [])
    def test_selected_library_outside_finite_directory_refused(self):
        policy = deepcopy(self.policy); policy['search_directories'] = [str(self.system)]
        with self.assertRaisesRegex(ValueError, 'exact search directory'): self.guard(policy=policy)
        self.assertEqual(self.supplier.calls, [])
    def test_native_install_missing_boundary_refuses_before_insertion(self):
        value = {'runtime': {'library_search_policy': {}, 'files': {}, 'stdlib_layout': {},
                            'cache_policy': {'namespace_roots': {}}}, 'config': {'source_files': {}}}
        contract = Contract(); contract.QualificationBudget = lambda: None
        with patch.object(self.L.sys, 'modules', {}), patch.object(self.L.sys, 'meta_path', [importlib.machinery.PathFinder]):
            with self.assertRaisesRegex(ValueError, 'before native finder installation'):
                self.L._install_body(value, contract)
            self.assertEqual(self.L.sys.meta_path, [importlib.machinery.PathFinder])
    def test_actual_selected_helper_source_consumption_before_supplier(self):
        source_row = self.policy['provider_source']
        value = {'runtime': {'library_search_policy': {'enforced_boundary': self.policy}},
                 'config': {'source_files': {source_row['path']: source_row['descriptor']}}}
        guard = self.L._selected_library_guard(value, self.C, None, fixture_only=True,
                    library_supplier=self.supplier, library_bootstrap_verifier=None)
        self.assertEqual(guard.policy, self.policy)
        self.assertFalse(guard.close()['native_authority_issued'])
    def test_actual_finder_install_extension_guard_and_cleanup_share_supplier(self):
        source_row = self.policy['provider_source']
        value = {'runtime': {'library_search_policy': {'enforced_boundary': self.policy},
                            'files': self.files, 'stdlib_layout': {},
                            'cache_policy': {'namespace_roots': {}}},
                 'config': {'source_files': {source_row['path']: source_row['descriptor']}}}
        class Budget:
            def check(self, *args): pass
        contract = Contract(); contract.QualificationBudget = Budget
        importer_state = types.SimpleNamespace(modules={}, meta_path=[importlib.machinery.PathFinder])
        with patch.object(self.L, 'sys', importer_state):
            finder = self.L._install_body(value, contract, fixture_only=True,
                    library_supplier=self.supplier, resolver=lambda path, layout: str(Path(path).absolute()))
            self.assertIs(importer_state.meta_path[0], finder)
            actual_spec = importlib.util.spec_from_file_location('CPU_inert_extension', self.ext,
                loader=importlib.machinery.ExtensionFileLoader('CPU_inert_extension', self.ext))
            with patch.object(importlib.machinery.PathFinder, 'find_spec', return_value=actual_spec):
                spec = finder.find_spec('CPU_inert_extension')
            self.assertIs(spec.loader.library_guard, finder.library_guard)
            with (patch.object(importlib.machinery.ExtensionFileLoader, 'create_module', side_effect=self.delegate_create),
                  patch.object(importlib.machinery.ExtensionFileLoader, 'exec_module', return_value=None)):
                with finder.framework_import():
                    module = spec.loader.create_module(spec); spec.loader.exec_module(module)
            self.L.remove(finder)
            self.assertNotIn(finder, importer_state.meta_path)
            self.assertTrue(finder.library_cleanup['fixture_only'])
            self.assertEqual(len(finder.library_cleanup['observed_boundaries']), 3)
            self.assertFalse(finder.library_cleanup['native_authority_issued'])
    def test_selected_helper_source_drift_blocks_before_supplier(self):
        source_row = self.policy['provider_source']
        value = {'runtime': {'library_search_policy': {'enforced_boundary': self.policy}},
                 'config': {'source_files': {source_row['path']: source_row['descriptor']}}}
        Path(source_row['path']).write_bytes(b'# source drift before actual consume\n')
        with self.assertRaisesRegex(ValueError, 'selected library/source bytes changed'):
            self.L._selected_library_guard(value, self.C, None, fixture_only=True,
                        library_supplier=self.supplier, library_bootstrap_verifier=None)
        self.assertEqual(self.supplier.calls, [])
    def test_caught_loader_descriptor_refusal_poisoned_before_completion(self):
        guard = self.guard(); loader = self.loader(guard)
        Path(self.ext).write_bytes(b'inert selected extension changed')
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module') as create:
            with self.assertRaisesRegex(ValueError, 'selected library/source bytes changed'): loader.create_module(None)
        create.assert_not_called(); self.assertTrue(guard.failed)
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
        self.assertTrue(self.supplier.closed)
    def test_caught_loader_exec_refusal_poisoned_before_completion(self):
        guard = self.guard(); loader = self.loader(guard)
        Path(self.ext).write_bytes(b'inert selected extension changed')
        with patch.object(importlib.machinery.ExtensionFileLoader, 'exec_module') as execute:
            with self.assertRaisesRegex(ValueError, 'selected library/source bytes changed'):
                loader.exec_module(types.ModuleType('CPU_inert_extension'))
        execute.assert_not_called(); self.assertTrue(guard.failed)
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_prospective_dependency_drift_refused_before_extension_delegate(self):
        guard = self.guard(); loader = self.loader(guard)
        Path(self.dep).write_bytes(b'inert dependency changed before its first load')
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module') as create:
            with self.assertRaisesRegex(ValueError, 'selected library/source bytes changed'): loader.create_module(None)
        create.assert_not_called(); self.assertTrue(guard.failed)
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_transient_unknown_library_event_refused_despite_final_snapshot(self):
        guard = self.guard(); loader = self.loader(guard)
        unknown = str(self.root / 'transient_outside_selected_directories.dll')
        Path(unknown).write_bytes(b'inert transient supplier file')
        def delegate(*args):
            self.supplier.add(unknown); self.supplier.unload(unknown)
            return self.delegate_create()
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module', side_effect=delegate) as create:
            with self.assertRaisesRegex(ValueError, 'unregistered transient dependent library event'): loader.create_module(None)
        create.assert_called_once(); self.assertNotIn(unknown, self.supplier.rows)
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_registered_library_unload_event_refused(self):
        guard = self.guard()
        with self.assertRaisesRegex(ValueError, 'library unload violates qualified lifetime conservation'):
            with guard.extension(self.ext, 'create'):
                self.delegate_create(); self.supplier.unload(self.dep)
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_incomplete_library_event_trace_refused_before_delegate(self):
        guard = self.guard(); loader = self.loader(guard)
        self.supplier.event_changes['complete'] = False
        with patch.object(importlib.machinery.ExtensionFileLoader, 'create_module') as create:
            with self.assertRaisesRegex(ValueError, 'complete actual library load/unload event observation'): loader.create_module(None)
        create.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_loaded_snapshot_without_events_cannot_claim_observed_closure(self):
        guard = self.guard()
        with self.assertRaisesRegex(ValueError, 'actual library event/snapshot identity join'):
            with guard.extension(self.ext, 'create'):
                self.supplier.rows[self.ext] = 900; self.supplier.rows[self.dep] = 901
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_prior_library_event_log_rewrite_refused(self):
        guard = self.guard()
        with guard.extension(self.ext, 'create'): self.delegate_create()
        self.supplier.event_rows[0]['path'] = self.extra
        with self.assertRaisesRegex(ValueError, 'append-only actual library event trace conservation'):
            with guard.extension(self.ext, 'exec'): self.fail('delegate must not run')
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_native_event_observer_requires_separate_source_qualification(self):
        contract = types.SimpleNamespace(stage_native_gate=lambda: None)
        with self.assertRaisesRegex(ValueError, 'separately source-qualified complete native library event observer'):
            self.B.Win32LibraryProvider(contract, lambda stage: True, object())
    def win32_cleanup_source_fixture(self, *, remove_failure=False, bootstrap_failure=False):
        # Execute only the native provider cleanup source body with inert API
        # suppliers. Constructor, native imports and actual OS APIs are excluded.
        provider = object.__new__(self.B.Win32LibraryProvider)
        provider.closed = False; provider.cookies = [('CPU_dir_a', 101), ('CPU_dir_b', 102)]
        removed = []; observer_calls = []
        def remove(cookie):
            removed.append(cookie); return not (remove_failure and cookie == 102)
        provider.kernel = types.SimpleNamespace(RemoveDllDirectory=remove, GetDllDirectoryW=object())
        provider.ctypes = types.SimpleNamespace(set_last_error=lambda number: None)
        def bootstrap(stage):
            if bootstrap_failure: raise ValueError('injected bootstrap final observation failure')
        provider._bootstrap = bootstrap; provider._string = lambda function: ''
        def observer_close():
            observer_calls.append('closed')
            return {'kind': 'OBSERVED_PROCESS_LIBRARY_EVENT_CLEANUP', 'complete': True, 'closed': True}
        provider.library_event_observer = types.SimpleNamespace(close=observer_close)
        return provider, removed, observer_calls
    def test_native_cleanup_source_always_closes_observer_after_cookie_failure(self):
        provider, removed, observer = self.win32_cleanup_source_fixture(remove_failure=True)
        with self.assertRaisesRegex(ValueError, 'actual Win32 library/event cleanup failed'): provider.close()
        self.assertEqual(removed, [102, 101]); self.assertEqual(observer, ['closed'])
        self.assertEqual(provider.cookies, [('CPU_dir_b', 102)])
    def test_native_cleanup_source_always_closes_observer_after_trace_failure(self):
        provider, removed, observer = self.win32_cleanup_source_fixture(bootstrap_failure=True)
        with self.assertRaisesRegex(ValueError, 'actual Win32 library/event cleanup failed'): provider.close()
        self.assertEqual(removed, [102, 101]); self.assertEqual(observer, ['closed'])
    def test_native_cleanup_source_positive_uses_only_injected_APIs(self):
        provider, removed, observer = self.win32_cleanup_source_fixture()
        result = provider.close()
        self.assertTrue(result['all_cookies_removed']); self.assertEqual(provider.cookies, [])
        self.assertEqual(removed, [102, 101]); self.assertEqual(observer, ['closed'])
    def test_caught_finder_extension_descriptor_refusal_poisoned(self):
        guard = self.guard()
        finder = self.L.RegisteredFinder(self.files, self.C, {},
                    resolver=lambda path, layout: str(Path(path).absolute()), library_guard=guard)
        spec = importlib.util.spec_from_file_location('CPU_inert_extension', self.ext,
                loader=importlib.machinery.ExtensionFileLoader('CPU_inert_extension', self.ext))
        with patch.object(importlib.machinery.PathFinder, 'find_spec', return_value=spec):
            with self.assertRaisesRegex(ValueError, 'poisoned native library boundary cannot return success'):
                with finder.framework_import():
                    Path(self.ext).write_bytes(b'inert extension drift before finder checks')
                    with self.assertRaisesRegex(ValueError, 'selected library/source bytes changed'):
                        finder.find_spec('CPU_inert_extension')
        self.assertTrue(guard.failed)
        with self.assertRaisesRegex(ValueError, 'cannot publish completion'): guard.close()
    def test_absent_optional_module_query_has_no_execution_and_does_not_poison(self):
        guard = self.guard(); finder = self.L.RegisteredFinder({}, self.C, {}, library_guard=guard)
        with patch.object(importlib.machinery.PathFinder, 'find_spec', return_value=None):
            with finder.framework_import():
                with self.assertRaises(ModuleNotFoundError): finder.find_spec('CPU_optional_missing_module')
        self.assertFalse(guard.failed); self.assertFalse(guard.close()['native_PASS'])


if __name__ == '__main__':
    unittest.main()
