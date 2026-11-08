"""Pure capability refusals and native source composition; no model execution."""
import ast
import copy
import inspect
import json
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest

from experiments.skill_acquisition_native_v1 import ce_native_authority as a
from experiments.skill_acquisition_native_v1 import ce_native_backend as b


class Trap:
    def __getattribute__(self, name): raise AssertionError('native work before admission: ' + name)
    def __call__(self, *args, **kwargs): raise AssertionError('native callback before admission')
    def __fspath__(self): raise AssertionError('native path before admission')


def context():
    root = Path(tempfile.gettempdir()).resolve()
    asset_path = Path(b.__file__).resolve().parents[1] / 'maxrl_comp002_pretraining_v1/evidence/Q1_ASSET_CANDIDATE.json'
    asset_raw = asset_path.read_bytes()
    asset = json.loads(asset_raw)
    recipe = {'synthetic_cpu': False, 'trainable_names': ['model.layers.14.fixture.weight'],
        'optimizer': {'capturable': True}, 'accumulation_rows': 8, 'microbatch_rows': 1}
    anchors = {key: a.digest(key) for key in ('common_anchor_sha256', 'base_weights_sha256',
        'warm_delta_sha256', 'tokenizer_receipt_sha256')}
    anchors['own_start_sha256'] = anchors['common_anchor_sha256']
    anchors['base_weights_sha256'] = asset['assets']['model.safetensors']['sha256']
    spec = {'schema': b.SCHEMA, 'run_seed': 123, 'inventory': {'blocks': [14, 27], 'numel': b.B_NUMEL},
        'parameter_inventory': [{'name': recipe['trainable_names'][0], 'shape': [b.B_NUMEL], 'dtype': 'F32'}],
        'source_checkout': str(root / 'ce-source'), 'output_directory': str(root / 'ce-output'),
        'model_dir': asset['model_path'], 'asset_receipt_utf8': asset_raw.decode('utf8'),
        'asset_receipt_ref': {'path': str(asset_path), 'sha256': b.PINNED_ASSET_RECEIPT_SHA256, 'bytes': len(asset_raw)},
        'tokenizer_receipt_ref': {'path': str(root / 'tokenizer.json'), 'sha256': anchors['tokenizer_receipt_sha256'], 'bytes': 100},
        'warm_delta_ref': {'path': str(root / 'warm.safetensors'), 'sha256': 'b' * 64,
            'data_sha256': anchors['warm_delta_sha256'], 'state_sha256': 'c' * 64, 'bytes': 4 * b.B_NUMEL + 100},
        'operation_bounds': {kind: {'read_bytes_upper': 10, 'write_bytes_upper': 10,
            'identity_hash_attempts_upper': 1, 'identity_hash_bytes_upper': 10} for kind in b.OPERATION_KINDS},
        'operation_bounds_status': 'AUTHORIZED_QUALIFICATION_UPPER'}
    runtime = {'device': 'cuda:0', 'dtype': 'float32'}
    recipe_hash = a.digest(recipe)
    result = {'schema': a.CONTEXT, 'mode': 'registered_scope', 'phase': 'qualification', 'arm': 'canonical',
        'request_sha256': 'd' * 64, 'scope_sha256': 'e' * 64, 'static_recipe_sha256': recipe_hash,
        'admission_evidence_verified': True, 'production_training_authorized': False,
        'bindings': {'recipe': {'recipe': recipe}, 'anchors': anchors,
            'runtime': {'decoding_preimage': runtime, 'decoding_sha256': a.digest(runtime)},
            'source': {'code_sha256': {'implementation.py': 'f' * 64}},
            'scope': {'role': 'qualification_only', 'qualification_authorized': True,
                'production_training_authorized': False, 'static_recipe_sha256': recipe_hash,
                'native_execution_stages': sorted(a.STAGES), 'native_backend': spec}}}
    for kind, minimum in b.loader_work_lower_bounds(result).items():
        spec['operation_bounds'][kind] = {key: max(spec['operation_bounds'][kind][key], value) for key, value in minimum.items()}
    return result


class ClosedRouteTests(unittest.TestCase):
    def test_import_is_framework_free_and_all_issuers_stay_unregistered(self):
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)
        self.assertFalse(a._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)
        from experiments.skill_acquisition_native_v1 import ce_controller as c
        self.assertFalse(c._REGISTERED_NATIVE_FACTORIES)
        self.assertFalse(c._REGISTERED_OUTER_AUTHORITY_VERIFIERS)

    def test_opaque_object_and_true_callback_cannot_issue_native_authority(self):
        with self.assertRaisesRegex(a.NativeAdmissionError, 'no registered'):
            a.NativeExecutionPermit(Trap(), object(), lambda *args: True, operation=Trap(), gate=Trap())
        value = context(); value['approved'] = True; value['training_authorized'] = True
        with self.assertRaisesRegex(a.NativeAdmissionError, 'no registered'):
            a.NativeExecutionPermit(value, object(), lambda *args: True, operation=Trap(), gate=Trap())

    def test_fixture_flags_and_counterfeit_permit_refuse_before_callbacks(self):
        class Forged:
            def require(self, *args, **kwargs): return {'approved': True}
        for permit in (None, True, {'approved': True}, Forged(), object.__new__(a.NativeExecutionPermit)):
            with self.subTest(permit_type=type(permit).__name__):
                with self.assertRaises(a.NativeAdmissionError):
                    a.require_execution_permit(permit, Trap(), 'runtime_anchor', operation=Trap(), gate=Trap())

    def test_capability_callback_fields_cannot_be_reassigned(self):
        permit = object.__new__(a.NativeExecutionPermit)
        for key in ('_operation', '_gate', '_context', '_issuer', '_verifier', '_seal'):
            with self.assertRaisesRegex(a.NativeAdmissionError, 'immutable'): setattr(permit, key, Trap())

    def test_backend_and_native_checkpoint_constructors_refuse_before_path_model_or_import(self):
        with self.assertRaises(a.NativeAdmissionError): b.NativeCEBackend(Trap(), None)
        with self.assertRaises(a.NativeAdmissionError): b.create_backend({'native_execution_permit': {'approved': True}})
        with self.assertRaises(a.NativeAdmissionError):
            b.NativeCECheckpoints(Trap(), Trap(), Trap(), b.B_NUMEL, Trap(),
                                 metadata_reserve=Trap(), operation=Trap(), native_permit=None)
        self.assertNotIn('torch', sys.modules)

    def test_every_public_lifecycle_entry_requires_permit_before_native_objects(self):
        backend = object.__new__(b.NativeCEBackend)
        backend.permit, backend.context, backend.recipe = None, {}, {}
        backend.operation = backend.gate = backend.model = backend.optimizer = Trap()
        methods = (lambda: backend.load({}, Trap(), Trap()), lambda: backend.update(Trap(), Trap()),
            lambda: backend.open_journal({}, Trap(), Trap(), False, None),
            lambda: backend.fingerprint('unapproved'), lambda: backend.checkpoints({}, Trap()),
            lambda: backend.save(Trap(), Trap()), lambda: backend.restore(Trap(), Trap()),
            lambda: backend.refresh_policy(Trap()), lambda: backend.donor({}, Trap()),
            lambda: backend.qualification_delta({}, Trap()), lambda: backend.cleanup())
        for invoke in methods:
            with self.assertRaises(a.NativeAdmissionError): invoke()

    def test_checkpoint_entries_deny_before_underlying_storage_work(self):
        storage = object.__new__(b.NativeCECheckpoints)
        storage.permit, storage.context = None, {}
        storage.operation = storage.gate = storage.storage = Trap()
        for invoke in (lambda: storage.attach_ledger(Trap()),
                       lambda: storage.save(Trap(), Trap(), Trap(), Trap()),
                       lambda: storage.restore(Trap(), Trap())):
            with self.assertRaises(a.NativeAdmissionError): invoke()


class NativeBindingTests(unittest.TestCase):
    def test_complete_pure_spec_is_not_authority_and_does_not_touch_assets(self):
        value = context(); spec = b.validate_backend_spec(value)
        self.assertEqual(spec['run_seed'], 123)
        with self.assertRaises(a.NativeAdmissionError): b.create_backend(value)
        self.assertNotIn('torch', sys.modules)

    def test_unknown_seed_prices_and_wrong_inventory_fail_without_defaults(self):
        mutations = (lambda s: s.pop('run_seed'), lambda s: s.update(run_seed=True),
            lambda s: s.update(operation_bounds_status='PROPOSED_UNMEASURED'),
            lambda s: s['operation_bounds'].pop('model_load'),
            lambda s: s['operation_bounds']['model_load'].update(read_bytes_upper=True),
            lambda s: s['parameter_inventory'][0].update(shape=[b.B_NUMEL - 1]),
            lambda s: s.update(inventory={'blocks': [8, 27], 'numel': b.B_NUMEL}),
            lambda s: s['warm_delta_ref'].update(data_sha256='0' * 64))
        for mutate in mutations:
            value = context(); mutate(value['bindings']['scope']['native_backend'])
            with self.assertRaises(a.NativeAdmissionError): b.validate_backend_spec(value)

    def test_zero_or_undersized_loader_prices_refuse_before_asset_or_framework_work(self):
        value = context()
        minimum = b.loader_work_lower_bounds(value)
        self.assertGreater(minimum['model_load']['identity_hash_bytes_upper'],
                           2 * json.loads(value['bindings']['scope']['native_backend']['asset_receipt_utf8'])['model_header']['data_bytes'])
        for kind in ('model_load', 'tokenizer_load', 'input_io', 'fingerprint'):
            for field, lower in minimum[kind].items():
                if not lower: continue
                with self.subTest(kind=kind, field=field):
                    changed = context()
                    changed['bindings']['scope']['native_backend']['operation_bounds'][kind][field] = lower - 1
                    with self.assertRaises(a.NativeAdmissionError): b.validate_backend_spec(changed)
            changed = context()
            changed['bindings']['scope']['native_backend']['operation_bounds'][kind] = dict.fromkeys(b.PROFILE_FIELDS, 0)
            with self.assertRaises(a.NativeAdmissionError): b.validate_backend_spec(changed)
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('transformers', sys.modules)

    def test_summary_or_changed_raw_asset_bytes_cannot_fake_analytical_loader_facts(self):
        for mutate in (lambda s: s.pop('asset_receipt_utf8'),
                       lambda s: s.update(asset_receipt_utf8=s['asset_receipt_utf8'] + ' '),
                       lambda s: s['asset_receipt_ref'].update(bytes=True),
                       lambda s: s['warm_delta_ref'].update(bytes=4 * b.B_NUMEL)):
            value = context(); mutate(value['bindings']['scope']['native_backend'])
            with self.assertRaises(a.NativeAdmissionError): b.validate_backend_spec(value)

    def test_production_cannot_reuse_unmeasured_qualification_profiles(self):
        value = context(); value['phase'] = 'training'
        with self.assertRaisesRegex(a.NativeAdmissionError, 'measured'): b.validate_backend_spec(value)

    def test_exact_full_plan_rejects_replayed_parent_boundary_and_partial_batch(self):
        plan = [{'boundary': i, 'logical_exposure': 0, 'row_ids': ['r%d' % (8 * i + j) for j in range(8)]}
                for i in range(375)]
        self.assertEqual(len(b._ce_plan_rows(plan)), 375)
        for mutate in (lambda p: p[1]['row_ids'].__setitem__(0, 'r0'),
                       lambda p: p[1]['row_ids'].pop(), lambda p: p[1].update(logical_exposure=1),
                       lambda p: p.pop()):
            changed = copy.deepcopy(plan); mutate(changed)
            with self.assertRaises(a.NativeAdmissionError): b._ce_plan_rows(changed)

    def test_native_lifecycle_source_uses_real_loader_core_storage_and_delta_primitives(self):
        # This is a source-composition audit only. It cannot certify CUDA work,
        # gradient parity, restore determinism, measured costs or learning.
        source = ast.parse(textwrap.dedent(inspect.getsource(b.NativeCEBackend)))
        calls = {node.func.attr for node in ast.walk(source) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Attribute)}
        self.assertTrue({'load_anchor', 'load_tokenizer', 'set_inventory', 'enable_checkpointing',
            'AdamW', 'state_digests', 'boundary_update', 'extract_donor', 'save', 'restore',
            'synchronize', 'empty_cache'} <= calls)
        self.assertFalse(hasattr(b, 'main'))

    def test_fingerprint_profile_does_not_redeclare_total_as_extra_passes(self):
        backend = object.__new__(b.NativeCEBackend)
        backend.spec = {'operation_bounds': {'fingerprint': {'read_bytes_upper': 10,
            'write_bytes_upper': 10, 'identity_hash_attempts_upper': 3, 'identity_hash_bytes_upper': 1000}}}
        events = []
        backend.operation = lambda kind, phase, **fields: events.append((kind, phase, fields))
        with backend._span('fingerprint', reason='pure_profile_fixture', state_bytes=100): pass
        start = events[0][2]
        self.assertNotIn('identity_hash_attempts_upper', start)
        self.assertNotIn('identity_hash_bytes_upper', start)
        self.assertEqual(start['state_bytes'], 100)

    def test_prepaid_release_spies_clear_references_even_when_guard_or_receipt_fails(self):
        # Isolate cleanup wiring with Python spies and a test-only admission
        # seam. No native permit/registry/framework/model is created or used.
        for failure in ('guard', 'receipt', 'admission', 'admission_guard'):
            events = []
            backend = object.__new__(b.NativeCEBackend)
            backend._admit = lambda *args, **kwargs: None
            backend.spec = {'operation_bounds': {'cleanup': dict.fromkeys(b.PROFILE_FIELDS, 0)}}
            backend._cleanup_operation_id = backend._cleanup_reserved_at = None
            backend._cleanup_reserved = backend._cleanup_finished = False
            backend.closed = False
            backend.model = backend.optimizer = backend.tokenizer = backend.torch = None
            def operation(kind, phase, **fields):
                events.append((kind, phase))
                if failure == 'receipt' and phase == 'returned': raise RuntimeError('retained receipt refusal')
            backend.operation = operation
            backend._reserve_cleanup()
            self.assertEqual(events, [('cleanup', 'started')])
            if failure.startswith('admission'):
                def revoke(*args, **kwargs): raise a.NativeAdmissionError('authority refusal after prepaid reservation')
                backend._admit = revoke
            def guard(stage):
                events.append(('guard', stage))
                if failure in ('guard', 'admission_guard'): raise RuntimeError('resource guard refusal')
            backend.gate = guard
            backend.model, backend.tokenizer = object(), object()
            backend.optimizer = SimpleNamespace(zero_grad=lambda **kwargs: events.append(('optimizer', 'cleared')))
            backend.torch = SimpleNamespace(cuda=SimpleNamespace(
                is_initialized=lambda: True,
                synchronize=lambda index: events.append(('cuda_spy', 'synchronized')),
                empty_cache=lambda: events.append(('cuda_spy', 'released'))))
            expected_error = a.NativeAdmissionError if failure.startswith('admission') else RuntimeError
            with self.assertRaisesRegex(expected_error, 'refusal') as caught: backend.cleanup()
            if failure == 'admission_guard':
                self.assertIn('authority refusal', str(caught.exception))
                self.assertEqual(caught.exception.cleanup_secondary_failures[0]['message'], 'resource guard refusal')
            self.assertIsNone(backend.model); self.assertIsNone(backend.optimizer); self.assertIsNone(backend.tokenizer)
            self.assertTrue(backend.closed); self.assertTrue(backend._cleanup_finished)
            self.assertIn(('cuda_spy', 'released'), events)
            self.assertEqual(sum(kind == 'cleanup' and phase == 'started' for kind, phase in events), 1)
        self.assertNotIn('torch', sys.modules)

    def test_cleanup_after_import_failure_does_not_initialize_cuda(self):
        backend = object.__new__(b.NativeCEBackend)
        backend._admit = lambda *args, **kwargs: None
        backend.spec = {'operation_bounds': {'cleanup': dict.fromkeys(b.PROFILE_FIELDS, 0)}}
        backend._cleanup_operation_id = backend._cleanup_reserved_at = None
        backend._cleanup_reserved = backend._cleanup_finished = False
        backend.model = backend.optimizer = backend.tokenizer = backend.torch = None
        events = []; backend.operation = lambda kind, phase, **fields: events.append((kind, phase))
        backend.gate = lambda stage: None; backend.closed = False
        backend._reserve_cleanup()
        def forbidden(*args): raise AssertionError('cleanup initialized CUDA after failed preparation')
        backend.torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False,
            synchronize=forbidden, empty_cache=forbidden))
        backend.cleanup()
        self.assertTrue(backend.closed); self.assertTrue(backend._cleanup_finished)
        self.assertEqual(events, [('cleanup', 'started'), ('cleanup', 'returned')])
        self.assertNotIn('torch', sys.modules)


if __name__ == '__main__': unittest.main()
