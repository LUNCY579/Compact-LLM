"""Actual outer-reader/library-manifest controls with inert selected bytes.

All historical65/779 hashes, Git, layout and DLL observations are injected.
Already-loaded stdlib dictionaries supply helper imports, including contextlib;
their fixture origin never proves the origin of the executing namespace. Selector
controls use the private real CPU capability and original context/registry mode.
The native wrapper ordering control has an explicit injected selector seam and
an inert body. Native/model/runtime authority stays false.
"""
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
import ast, builtins, contextlib, copy, datetime, hashlib
import importlib, importlib.abc, importlib.machinery, importlib.util
import json, marshal, math, os, re, stat, subprocess, sys, tempfile, time, types, unittest
from unittest.mock import patch
from outer_context_fixture import OuterContextFixture, HERE
from test_full_native_cpu_workflow import compiled
from test_library_policy import Supplier


BOOTSTRAP_NAMES = ('pathlib', 'ast', 'hashlib', 'importlib', 'importlib.util',
    'importlib.abc', 'importlib.machinery', 'marshal', 'types', 'stat', 'time',
    'json', 'math', 're', 'sys', 'subprocess', 'copy', 'datetime', 'os', 'contextlib')
FORBIDDEN_IMPORTS = {'torch', 'ctypes', 'psutil', 'transformers', 'safetensors', 'numpy'}


class LibraryBoundaryManifestControls(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='CPU_library_manifest_', dir=HERE)
        self.addCleanup(self.temp.cleanup)
        self.fixture = OuterContextFixture(self.temp.name); self.C = self.fixture.C
        self.interpreter = str(self.fixture.base/'python.exe')
        self.extension = str(self.fixture.base/'DLLs/CPU_inert.pyd')
        provider = str(self.fixture.package/'prefix_library_policy.py')
        self.policy = {'kind': 'FINITE_WINDOWS_DLL_BOUNDARY_V1',
            'search_directories': [str(self.fixture.base), str(self.fixture.base/'DLLs')],
            'system_directory': str(self.fixture.base),
            'library_files': {p: copy.deepcopy(self.fixture.inert_files[p])
                for p in (self.interpreter, self.extension)},
            'initial_loaded_libraries': [self.interpreter],
            'extension_transitions': {self.extension: {'create': [self.extension], 'exec': []}},
            'framework_transition': [self.extension],
            'provider_source': {'path': provider, 'descriptor': copy.deepcopy(self.fixture.sources[provider])}}
        self.fixture.runtime = copy.deepcopy(self.fixture.runtime)
        self.fixture.runtime['library_search_policy']['enforced_boundary'] = self.policy
        self.assumptions = {'kind': 'CPU_ALREADY_LOADED_STDLIB_EXPORT_SUPPLIER_ASSUMPTIONS',
            'fixture_only': True, 'current_execution_proven': False,
            'actual_fresh_bootstrap_qualified': False,
            'inert_origin_represents_executing_namespace': False,
            'modules': [{'name': name, 'actual_export_module_name': sys.modules[name].__name__,
                'actual_module_origin': getattr(sys.modules[name], '__file__', None),
                'actual_spec_origin': getattr(getattr(sys.modules[name], '__spec__', None), 'origin', None),
                'export_supplier': 'already_loaded_standard_library_module_dictionary',
                'current_execution_proven': False} for name in BOOTSTRAP_NAMES]}
        self.reseal()
        # Compile before proxy bootstrap scope so the selected loader keeps
        # the actual importer state object whose lists are isolated below.
        self.loader = compiled(self.fixture.package/'prefix_registered_loader.py')
        self.worker = compiled(self.fixture.package/'prefix_input_owned_worker.py'); self.worker.C = self.C
        self.selector_capability = self.C._mint_CPU_native_context_fixture()
        self.attempted_forbidden = []
        original = builtins.__import__
        def imports(name, *args, **kwargs):
            if name.split('.', 1)[0] in FORBIDDEN_IMPORTS:
                self.attempted_forbidden.append(name)
                raise AssertionError('forbidden native/framework import: '+name)
            return original(name, *args, **kwargs)
        self.import_patch = patch.object(builtins, '__import__', side_effect=imports)
        self.import_patch.start(); self.addCleanup(self.import_patch.stop)
        self.addCleanup(lambda: self.assertEqual(self.attempted_forbidden, []))

    def reseal(self, *, bootstrap_exports=False):
        """Rejoin all documents so refusals test semantics beyond old hashes."""
        f = self.fixture; C = self.C
        if bootstrap_exports:
            path = str(f.site/'typing_extensions.py'); pin = f.inert_files[path]['sha256']
            modules = {name: {'path': path, 'source_sha256': pin, 'cache_path': None,
                'cache_sha256': None} for name in BOOTSTRAP_NAMES}
            f.bootstrap = copy.deepcopy(f.bootstrap)
            f.bootstrap['preloaded_bootstrap_modules_sha256'] = C.digest(modules)
            f.bootstrap = f._seal(f.bootstrap)
            f.runtime['cache_policy']['preloaded_bootstrap_modules'] = modules
            f.runtime['cache_policy']['fresh_bootstrap_evidence'] = f.anchor(f.bootstrap_path, f.bootstrap)
        f.runtime = f._seal(f.runtime)
        f.config = copy.deepcopy(f.config); f.config['runtime_files'] = copy.deepcopy(f.runtime['files'])
        f.config = f._seal(f.config)
        f.review = copy.deepcopy(f.review)
        f.review.update(source_inventory_sha256=C.digest(f.config['source_files']),
            runtime_closure_sha256=C.digest(f.runtime['files']),
            rosters_sha256=C.digest(f.runtime['rosters']),
            cache_policy_sha256=C.digest(f.runtime['cache_policy']),
            library_search_policy_sha256=C.digest(f.runtime['library_search_policy']))
        f.review = f._seal(f.review)
        f.registration = copy.deepcopy(f.registration)
        f.registration.update(config_body_sha256=f.config['sha256'], runtime_body_sha256=f.runtime['sha256'],
            source_files=copy.deepcopy(f.config['source_files']), runtime_files=copy.deepcopy(f.runtime['files']),
            qualification_review_file_sha256=f.anchor(f.review_path, f.review)['file_sha256'],
            qualification_review_body_sha256=f.review['sha256'])
        f.registration = f._seal(f.registration)
        for path, value in ((f.config_path, f.config), (f.runtime_path, f.runtime),
                (f.registration_path, f.registration), (f.review_path, f.review), (f.bootstrap_path, f.bootstrap)):
            path.write_bytes(f._raw(value)); f.documents[str(path)] = value
        f.anchors.update(expected_config_file=f.anchor(f.config_path, f.config)['file_sha256'],
            expected_config_body=f.config['sha256'],
            expected_runtime_file=f.anchor(f.runtime_path, f.runtime)['file_sha256'],
            expected_runtime_body=f.runtime['sha256'],
            expected_registration_file=f.anchor(f.registration_path, f.registration)['file_sha256'],
            expected_registration_body=f.registration['sha256'])

    @contextmanager
    def isolated(self):
        # These proxy dictionaries are explicit already-loaded exports. Their
        # inert metadata is only a bootstrap supplier assumption.
        self.reseal(bootstrap_exports=True)
        with self.fixture.activate():
            modules = {}
            for name in BOOTSTRAP_NAMES:
                proxy = ModuleType(name); proxy.__dict__.update(sys.modules[name].__dict__)
                proxy.__file__ = str(self.fixture.site/'typing_extensions.py')
                proxy.__cached__ = None; proxy.__spec__ = None; modules[name] = proxy
            allowed = [importlib.machinery.BuiltinImporter, importlib.machinery.FrozenImporter,
                importlib.machinery.PathFinder]
            with patch.dict(sys.modules, modules, clear=True), patch.object(sys, 'meta_path', allowed):
                # The already-loaded sys export observes the same isolated
                # list used by the actual installed finder and selector.
                modules['sys'].meta_path = sys.meta_path
                yield

    def selected(self):
        return self.fixture.runtime['library_search_policy']['enforced_boundary']

    def install(self):
        context = self.fixture.context(); loader = self.loader
        supplier = Supplier(self.selected())
        finder = loader._CPU_install_fixture(context, self.C,
            _capability=self.C._mint_CPU_native_context_fixture(), library_supplier=supplier)
        return context, loader, finder, supplier

    @contextmanager
    def CPU_selector_context(self, context, finder):
        value = self.C.context_value(context)
        assumptions = {'kind': 'CPU_SOURCE_OWNED_SELECTOR_FIXTURE',
            'fixture_only': True, 'native_PASS': False, 'native_authority_issued': False,
            'selector_boundary': 'actual private CPU capability; original context and recorded mode',
            'native_environment_or_runtime_qualified': False}
        self.assertTrue(value['fixture_only']); self.assertTrue(finder.fixture_only)
        self.assertEqual(finder.selected_context_projection, self.C.canonical(value))
        yield value, assumptions

    def CPU_selector(self, value, requested=None):
        return self.worker._CPU_installed_framework_boundary_fixture(value, requested,
            _capability=self.selector_capability)

    def assert_selected_method(self, value, finder):
        actual = self.CPU_selector(value, finder.framework_import)
        self.assertIs(actual.__self__, finder)
        self.assertIs(actual.__func__, type(finder).framework_import)
        self.assertEqual(actual.__func__.__code__.co_filename,
            str(self.fixture.package/'prefix_registered_loader.py'))
        self.assertEqual(finder.selected_loader_source,
            self.fixture.config['source_files'][str(self.fixture.package/'prefix_registered_loader.py')])
        self.assertIsNotNone(finder.library_guard)
        return actual

    @staticmethod
    def noop_bound_method(owner, documentary_filename):
        # Deliberately counterfeit source-origin metadata. A co_filename
        # string alone must not identify the selected class method.
        namespace = {'contextlib': contextlib}
        exec(compile('def framework_import(self):\n    return contextlib.nullcontext()\n',
            documentary_filename, 'exec'), namespace)
        return types.MethodType(namespace['framework_import'], owner)

    def test_actual_metadata_rosters_and_selected_bytes_accept_inert_library_subset(self):
        with self.fixture.activate():
            value = self.C.native_metadata(self.fixture.config, self.fixture.registration,
                self.fixture.runtime, self.fixture.head)
            self.C.qualification_rosters(value)
        context = self.fixture.context(); value = self.C.context_value(context)
        self.assertEqual(value['runtime']['library_search_policy']['enforced_boundary'], self.policy)
        coverage = self.fixture.coverage()
        self.assertTrue({self.interpreter, self.extension, self.policy['provider_source']['path']} <=
            set(coverage['real_descriptor_paths']))
        self.assertEqual(set(coverage['injected_historical_runtime_paths']), set(self.fixture.historical_runtime))
        self.assertTrue(value['fixture_only']); self.assertFalse(value['native_PASS'])
        self.assertFalse(value['native_authority_issued']); self.assertFalse(coverage['native_runtime_byte_closure_proven'])
        self.assertFalse(coverage['current_module_execution_proven'])

    def test_resealed_provider_path_drift_refused_by_actual_source_join(self):
        self.selected()['provider_source']['path'] = str(self.fixture.package/'prefix_input_owned_worker.py')
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'provider exact source join'): self.fixture.context()

    def test_resealed_provider_descriptor_drift_refused_by_actual_source_join(self):
        self.selected()['provider_source']['descriptor']['sha256'] = '0'*64
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'provider exact source join'): self.fixture.context()

    def test_resealed_library_descriptor_drift_refused_by_actual_runtime_subset_join(self):
        self.selected()['library_files'][self.extension]['sha256'] = '0'*64
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'exact qualified runtime subset'): self.fixture.context()

    def test_resealed_library_outside_runtime_map_refused(self):
        extra = str(self.fixture.base/'DLLs/not_registered.dll')
        self.selected()['library_files'][extra] = copy.deepcopy(self.selected()['library_files'][self.extension])
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'exact qualified runtime subset'): self.fixture.context()

    def test_resealed_python_source_as_library_refused(self):
        path = str(self.fixture.site/'torch/__init__.py')
        self.selected()['library_files'][path] = copy.deepcopy(self.fixture.runtime['files'][path])
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'exact qualified runtime subset'): self.fixture.context()

    def test_resealed_library_only_role_drift_refused(self):
        self.selected()['library_files'][self.extension]['role'] = 'torch_binary'
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'exact qualified runtime subset'): self.fixture.context()

    def test_resealed_equal_library_runtime_role_drift_refused_by_native_metadata(self):
        self.selected()['library_files'][self.extension]['role'] = 'torch_binary'
        self.fixture.runtime['files'][self.extension]['role'] = 'torch_binary'
        self.reseal()
        with self.assertRaisesRegex(ValueError, 'runtime root/role/suffix assignment'): self.fixture.context()

    def test_actual_provider_bytes_drift_refused_before_supplier_observation(self):
        context = self.fixture.context(); loader = compiled(self.fixture.package/'prefix_registered_loader.py')
        supplier = Supplier(self.selected()); path = Path(self.selected()['provider_source']['path'])
        raw = path.read_bytes(); path.write_bytes(b'#'+raw[1:])
        with self.fixture.activate():
            with self.assertRaisesRegex(ValueError, 'registered runtime exact stable bytes'):
                loader._selected_library_guard(self.C.context_value(context), self.C, None,
                    fixture_only=True, library_supplier=supplier, library_bootstrap_verifier=None)
        self.assertEqual(supplier.calls, [])

    def test_actual_installed_finder_encloses_framework_supplier_and_cleans_up(self):
        # Native context/body prerequisites are deliberately injected solely
        # for the source-wrapper control; no native capability is minted.
        worker = self.worker
        with self.isolated():
            context, loader, finder, supplier = self.install(); baseline = list(sys.meta_path)[1:]
            def framework():
                self.assertEqual(finder.library_guard._depth, 1)
                supplier.add(self.extension); return 'INERT_FRAMEWORK_EXPORT'
            def body(ctx, owner, output, value, suppliers, revalidate):
                return {'api': {'invoke_framework': suppliers['framework']}}
            try:
                with self.CPU_selector_context(context, finder) as (value, selector_assumptions):
                    actual = self.assert_selected_method(value, finder)
                    wrapper_value = copy.deepcopy(value); wrapper_value['fixture_only'] = False
                    # This native wrapper's context refusal/selector are explicit
                    # ordering seams. The actual selector was verified above via
                    # the original CPU record, never via a promoted CPU record.
                    with patch.object(self.C, 'stage_native_gate'), \
                            patch.object(self.C, 'context_value', return_value=wrapper_value), \
                            patch.object(worker, '_installed_framework_boundary', return_value=actual), \
                            patch.object(worker, '_native_binding_suppliers', return_value={'framework': framework}), \
                            patch.object(worker, '_tensor_binding_body', side_effect=body):
                        api = worker.real_torch_bindings(context, {}, 'CPU_unused', framework_boundary=finder.framework_import)
                        self.assertEqual(api['invoke_framework'](), 'INERT_FRAMEWORK_EXPORT')
                    self.assertFalse(selector_assumptions['native_PASS'])
                    self.assertFalse(selector_assumptions['native_authority_issued'])
                    self.assertFalse(selector_assumptions['native_environment_or_runtime_qualified'])
                self.assertEqual(finder.library_guard.observed[-1]['boundary'], 'framework_import')
                self.assertEqual(finder.library_guard.observed[-1]['new_libraries'], [self.extension])
            finally:
                loader.remove(finder)
            self.assertEqual(sys.meta_path, baseline)
            self.assertTrue(supplier.closed); self.assertTrue(finder.library_cleanup['fixture_only'])
            self.assertFalse(finder.library_cleanup['native_PASS']); self.assertFalse(finder.library_cleanup['native_authority_issued'])
            observed = finder.observed['contextlib']
            self.assertEqual(observed['loader_kind'], 'externally_qualified_bootstrap_assumption')
            self.assertFalse(observed['current_module_execution_observed'])
            self.assertFalse(observed['bootstrap_execution_inferred'])
        self.assertFalse(self.assumptions['current_execution_proven'])
        self.assertFalse(self.assumptions['actual_fresh_bootstrap_qualified'])
        self.assertFalse(self.assumptions['inert_origin_represents_executing_namespace'])

    def test_actual_installed_CPU_finder_cannot_satisfy_native_selector(self):
        with self.isolated():
            context, loader, finder, supplier = self.install()
            try:
                with self.assertRaisesRegex(ValueError, 'installed selected context-bound library finder'):
                    self.worker._installed_framework_boundary(self.C.context_value(context), finder.framework_import)
            finally: loader.remove(finder)
            self.assertTrue(supplier.closed); self.assertFalse(finder.library_cleanup['native_authority_issued'])

    def test_recorded_CPU_installation_cannot_be_promoted_by_mutable_flags_and_projection(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                value = self.C.context_value(context); self.assert_selected_method(value, finder)
                promoted = copy.deepcopy(value); promoted['fixture_only'] = False
                for supplied in (value, promoted):
                    with self.subTest(context_fixture_flag=supplied['fixture_only']), \
                            patch.object(finder, 'fixture_only', False), \
                            patch.object(finder, 'selected_context_projection', self.C.canonical(supplied)):
                        with self.assertRaisesRegex(ValueError, 'CPU installation record has no native selector authority'):
                            self.worker._installed_framework_boundary(supplied, finder.framework_import)
                self.assertEqual(finder.library_guard.observed, [])
            finally: loader.remove(finder)

    def test_private_selector_refuses_forged_capability_and_non_CPU_projection(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                value = self.C.context_value(context); self.assert_selected_method(value, finder)
                with self.assertRaisesRegex(ValueError, 'private CPU native-context source fixture capability'):
                    self.worker._CPU_installed_framework_boundary_fixture(value, finder.framework_import,
                        _capability=object())
                changed = copy.deepcopy(value); changed['fixture_only'] = False
                with self.assertRaisesRegex(ValueError, 'CPU installed-selector fixture only'):
                    self.CPU_selector(changed, finder.framework_import)
                self.assertEqual(finder.library_guard.observed, [])
            finally: loader.remove(finder)

    def test_counterfeit_finder_and_guard_selected_filenames_pins_state_cannot_create_registry_record(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            counterfeit = None
            try:
                value = self.C.context_value(context); self.assert_selected_method(value, finder)
                helper = str(self.fixture.package/'prefix_library_policy.py')
                installer = str(self.fixture.package/'prefix_registered_loader.py')
                # Supplied no-op classes deliberately copy every documentary
                # filename/pin/state predicate. Only the real source-owned
                # installer can retain the private object/class/method record.
                guard_namespace = {'contextlib': contextlib,
                    '__compiled_source_descriptor__': copy.deepcopy(value['config']['source_files'][helper])}
                exec(compile('class CounterfeitGuard:\n'
                    '    active=True; failed=False; closed=False\n'
                    '    def framework_import(self): return contextlib.nullcontext()\n',
                    helper, 'exec'), guard_namespace)
                fake_guard = guard_namespace['CounterfeitGuard']()
                finder_namespace = {'_check_installed_record': loader._check_installed_record}
                exec(compile('class CounterfeitFinder:\n'
                    '    def framework_import(self): return self.library_guard.framework_import()\n',
                    installer, 'exec'), finder_namespace)
                counterfeit = finder_namespace['CounterfeitFinder']()
                counterfeit.contract = self.C; counterfeit.fixture_only = True
                counterfeit.library_guard = counterfeit.selected_library_guard = fake_guard
                counterfeit.selected_loader_source = copy.deepcopy(value['config']['source_files'][installer])
                counterfeit.selected_context_projection = self.C.canonical(value)
                sys.meta_path.append(counterfeit)
                with self.assertRaisesRegex(ValueError, 'exact selected finder executable code and class globals'):
                    self.CPU_selector(value, counterfeit.framework_import)
                with self.assertRaisesRegex(ValueError, 'actual source-owned installation record'):
                    loader._check_installed_record(counterfeit, value,
                        _CPU_capability=self.selector_capability)
                self.assertEqual(finder.library_guard.observed, [])
            finally:
                if counterfeit in sys.meta_path: sys.meta_path.remove(counterfeit)
                loader.remove(finder)

    def test_removed_finder_reinsertion_does_not_restore_source_owned_installation_record(self):
        with self.isolated():
            context, loader, finder, supplier = self.install()
            value = self.C.context_value(context); self.assert_selected_method(value, finder)
            loader.remove(finder)
            self.assertTrue(supplier.closed); self.assertFalse(finder.library_cleanup['native_authority_issued'])
            sys.meta_path.insert(0, finder)
            try:
                with self.assertRaisesRegex(ValueError, 'actual source-owned installation record'):
                    loader._check_installed_record(finder, value,
                        _CPU_capability=self.selector_capability)
                with self.assertRaisesRegex(ValueError, 'active unfailed unclosed library guard before native import'):
                    self.CPU_selector(value, finder.framework_import)
            finally: sys.meta_path.remove(finder)

    def test_actual_installed_finder_refuses_nullcontext_and_arbitrary_callbacks(self):
        with self.isolated():
            context, loader, finder, supplier = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    for callback in (contextlib.nullcontext, lambda: contextlib.nullcontext()):
                        with self.subTest(callback_kind=type(callback).__name__):
                            with self.assertRaisesRegex(ValueError, 'callback cannot replace installed library boundary'):
                                self.CPU_selector(value, callback)
                            wrapper_value = copy.deepcopy(value); wrapper_value['fixture_only'] = False
                            with patch.object(self.C, 'stage_native_gate'), \
                                    patch.object(self.C, 'context_value', return_value=wrapper_value), \
                                    patch.object(self.worker, '_installed_framework_boundary',
                                        side_effect=lambda ignored, requested: self.CPU_selector(value, requested)), \
                                    patch.object(self.worker, '_native_binding_suppliers',
                                        side_effect=AssertionError('must not prepare suppliers')) as suppliers:
                                with self.assertRaisesRegex(ValueError, 'callback cannot replace installed library boundary'):
                                    self.worker.real_torch_bindings(context, {}, 'CPU_unused', framework_boundary=callback)
                                suppliers.assert_not_called()
                    self.assertEqual(finder.library_guard.observed, [])
            finally: loader.remove(finder)
            self.assertTrue(supplier.closed); self.assertFalse(finder.library_cleanup['native_authority_issued'])

    def test_actual_installed_finder_refuses_wrong_context_projection(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    changed = copy.deepcopy(value); changed['expected_head'] = 'b'*40
                    with self.assertRaisesRegex(ValueError, 'installed selected context-bound library finder'):
                        self.CPU_selector(changed, finder.framework_import)
            finally: loader.remove(finder)

    def test_actual_installed_finder_refuses_wrong_compile_descriptor(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    for key, changed in (('sha256', '0'*64), ('bytes', finder.selected_loader_source['bytes']+1)):
                        descriptor = {**finder.selected_loader_source, key: changed}
                        with self.subTest(field=key), patch.object(finder, 'selected_loader_source', descriptor):
                            with self.assertRaisesRegex(ValueError, 'installed selected context-bound library finder'):
                                self.CPU_selector(value, finder.framework_import)
            finally: loader.remove(finder)

    def test_actual_finder_removed_from_importer_cannot_satisfy_native_selector(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    sys.meta_path.remove(finder)
                    with self.assertRaisesRegex(ValueError, 'installed selected context-bound library finder'):
                        self.CPU_selector(value, finder.framework_import)
            finally: loader.remove(finder)
            self.assertFalse(finder.library_cleanup['native_authority_issued'])

    def test_actual_installed_finder_refuses_method_from_another_finder(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    other = loader.RegisteredFinder(finder.files, self.C, finder.layout,
                        library_guard=finder.library_guard)
                    self.assert_selected_method(value, finder)
                    with self.assertRaisesRegex(ValueError, 'callback cannot replace installed library boundary'):
                        self.CPU_selector(value, other.framework_import)
            finally: loader.remove(finder)

    def test_actual_installed_finder_refuses_missing_library_guard(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    with patch.object(finder, 'library_guard', None):
                        with self.assertRaisesRegex(ValueError, 'installed selected context-bound library finder'):
                            self.CPU_selector(value, finder.framework_import)
            finally: loader.remove(finder)

    def test_actual_installed_finder_refuses_inactive_library_guard(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    for field, state in (('active', False), ('failed', True), ('closed', True)):
                        with self.subTest(guard_state=field), patch.object(finder.library_guard, field, state):
                            with self.assertRaisesRegex(ValueError, 'active unfailed unclosed library guard before native import'):
                                self.CPU_selector(value, finder.framework_import)
            finally: loader.remove(finder)

    def test_actual_installed_finder_refuses_arbitrary_guard_object(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    counterfeit = types.SimpleNamespace(active=True, failed=False, closed=False,
                        framework_import=contextlib.nullcontext)
                    for guard in (object(), counterfeit):
                        with self.subTest(guard_kind=type(guard).__name__), patch.object(finder, 'library_guard', guard):
                            with self.assertRaisesRegex(ValueError, 'actual installed selected library guard identity'):
                                self.CPU_selector(value, finder.framework_import)
            finally: loader.remove(finder)

    def test_actual_installed_finder_refuses_noop_bound_method_replacement(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    for filename, error in ((str(Path(__file__)), 'installed selected context-bound library finder'),
                            (str(self.fixture.package/'prefix_registered_loader.py'), 'actual installed selected finder method')):
                        method = self.noop_bound_method(finder, filename)
                        with self.subTest(documentary_filename=Path(filename).name), \
                                patch.object(finder, 'framework_import', method):
                            with self.assertRaisesRegex(ValueError, error):
                                self.CPU_selector(value, method)
                    self.assertEqual(finder.library_guard.observed, [])
            finally: loader.remove(finder)

    def test_actual_installed_guard_refuses_noop_bound_method_replacement(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    guard = finder.library_guard
                    for filename in (str(Path(__file__)), str(self.fixture.package/'prefix_library_policy.py')):
                        method = self.noop_bound_method(guard, filename)
                        with self.subTest(documentary_filename=Path(filename).name), \
                                patch.object(guard, 'framework_import', method):
                            with self.assertRaisesRegex(ValueError, 'selected library guard method/source identity'):
                                self.CPU_selector(value, finder.framework_import)
                    self.assertEqual(guard.observed, [])
            finally: loader.remove(finder)

    def test_actual_installed_guard_refuses_selected_identity_marker_swap(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    with patch.object(finder, 'selected_library_guard', object()):
                        with self.assertRaisesRegex(ValueError, 'actual installed selected library guard identity'):
                            self.CPU_selector(value, finder.framework_import)
                    self.assertEqual(finder.library_guard.observed, [])
            finally: loader.remove(finder)

    def test_actual_installed_guard_refuses_joint_counterfeit_identity_swap(self):
        class CounterfeitGuard:
            active = True; failed = False; closed = False
            def framework_import(self): return contextlib.nullcontext()
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder); counterfeit = CounterfeitGuard()
                    with patch.object(finder, 'library_guard', counterfeit), \
                            patch.object(finder, 'selected_library_guard', counterfeit):
                        with self.assertRaisesRegex(ValueError, 'selected library guard method/source identity'):
                            self.CPU_selector(value, finder.framework_import)
                    self.assertEqual(finder.library_guard.observed, [])
            finally: loader.remove(finder)

    def test_actual_installed_guard_refuses_compiled_provider_descriptor_drift(self):
        with self.isolated():
            context, loader, finder, _ = self.install()
            try:
                with self.CPU_selector_context(context, finder) as (value, _):
                    self.assert_selected_method(value, finder)
                    namespace = finder.library_guard.framework_import.__func__.__globals__
                    original = namespace['__compiled_source_descriptor__']
                    for key, changed in (('sha256', '0'*64), ('bytes', original['bytes']+1)):
                        with self.subTest(field=key), patch.dict(namespace,
                                {'__compiled_source_descriptor__': {**original, key: changed}}):
                            with self.assertRaisesRegex(ValueError, 'selected library guard compiled descriptor'):
                                self.CPU_selector(value, finder.framework_import)
                    self.assertEqual(finder.library_guard.observed, [])
            finally: loader.remove(finder)

    def test_installed_finder_refuses_library_drift_before_framework_delegate(self):
        with self.isolated():
            _, loader, finder, supplier = self.install()
            path = Path(self.extension); raw = path.read_bytes(); path.write_bytes(b'X'+raw[1:])
            delegate = unittest.mock.Mock(side_effect=AssertionError('must not import framework'))
            with self.assertRaisesRegex(ValueError, 'registered runtime exact stable bytes'):
                with finder.framework_import(): delegate()
            delegate.assert_not_called(); self.assertTrue(finder.library_guard.failed)
            with self.assertRaisesRegex(ValueError, 'cannot publish completion'): loader.remove(finder)
            self.assertNotIn(finder, sys.meta_path); self.assertTrue(supplier.closed)
            self.assertIsNone(finder.library_cleanup)

    def test_CPU_selected_boundary_cannot_enter_public_native_loader_or_bindings(self):
        context = self.fixture.context(); loader = compiled(self.fixture.package/'prefix_registered_loader.py')
        worker = compiled(self.fixture.package/'prefix_input_owned_worker.py'); worker.C = self.C
        with patch.object(self.C, 'stage_native_gate'), \
                patch.object(loader, '_install_body', side_effect=AssertionError('must not install native finder')):
            with self.assertRaisesRegex(ValueError, 'no native loader authority'): loader.install(context, self.C)
            with self.assertRaisesRegex(ValueError, 'cannot create real native bindings'):
                worker.real_torch_bindings(context, {}, 'CPU_unused', framework_boundary=lambda: None)


if __name__ == '__main__':
    unittest.main()
