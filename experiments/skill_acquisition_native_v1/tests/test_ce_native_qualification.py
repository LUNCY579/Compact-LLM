"""Pure qualification contracts/source checks; no framework or model execution."""
import ast
import copy
import inspect
from pathlib import Path
import sys
import tempfile
import textwrap
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_native_authority as authority
from experiments.skill_acquisition_native_v1 import ce_native_qualification as q


class Trap:
    def __getattribute__(self, name): raise AssertionError('work before qualification admission: ' + name)
    def __call__(self, *args, **kwargs): raise AssertionError('callback before qualification admission')


def declaration():
    root = Path(tempfile.gettempdir()).resolve() / 'native-qualification-fixture'
    ordered = ['r%d' % index for index in range(100)]
    receipt = {'prepared_row_bindings': [{'id': rid} for rid in ordered],
        'witnesses': {'max_prompt_tokens': {'row_ids': ['r0']},
                     'max_labeled_tokens': {'row_ids': ordered[1:90]},
                     'max_sequence_positions': {'row_ids': ['r0']}}}
    spec = {'schema': q.SCHEMA, 'all_witness_ids': ordered[:90], 'gradient_batch_ids': ordered[:8],
        'adam_batches': [ordered[:8], ordered[:8]], 'owned_artifact_directory': str(root / 'raw'),
        'json_artifact_bytes_upper': 4 * q.CHUNK, 'tensor_artifact_bytes_upper': 1024 * q.CHUNK,
        'artifact_chunk_bytes': q.CHUNK, 'reader_bounds': dict.fromkeys(('operation_attempts_upper',
            'io_read_bytes_upper', 'identity_hash_attempts_upper', 'identity_hash_bytes_upper', 'reducer_work_bytes_upper'), 10 ** 10)}
    context = {'bindings': {'scope': {'qualifier_execution': spec, 'native_backend': {'output_directory': str(root)}}}}
    return context, receipt


class ClosedQualificationTests(unittest.TestCase):
    def test_framework_free_source_and_all_service_registries_closed(self):
        self.assertNotIn('torch', sys.modules); self.assertNotIn('transformers', sys.modules)
        self.assertFalse(q._REGISTERED_ACCOUNTING_FACTORIES)
        self.assertFalse(q._REGISTERED_CHECKPOINT_ADAPTERS)
        self.assertFalse(q._REGISTERED_RESOURCE_ADAPTERS)
        self.assertFalse(authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)

    def test_constructor_denies_json_or_counterfeit_authority_before_backend_or_services(self):
        for permit in (None, {'approved': True}, True, object.__new__(authority.NativeExecutionPermit)):
            with self.assertRaises(authority.NativeAdmissionError):
                q.NativeQualificationExecutor(Trap(), permit, Trap(), Trap())

    def test_begin_stage_and_cleanup_refuse_before_unconstructed_native_objects(self):
        executor = object.__new__(q.NativeQualificationExecutor)
        executor.context, executor.permit = {}, None
        executor.operation = executor.gate = executor.backend = executor.services = Trap()
        for invoke in (lambda: executor.begin_qualification({}, Trap(), Trap(), Trap(), Trap()),
                       lambda: executor.qualify_stage('runtime_anchor', Trap(), Trap(), {}),
                       lambda: executor.cleanup()):
            with self.assertRaises(authority.NativeAdmissionError): invoke()

    def test_missing_accounting_checkpoint_and_resource_adapters_cannot_return_flag_observations(self):
        forged = lambda *args: {'measured': True, 'parameters_bitwise': True}
        for registry in (q._REGISTERED_ACCOUNTING_FACTORIES, q._REGISTERED_CHECKPOINT_ADAPTERS, q._REGISTERED_RESOURCE_ADAPTERS):
            with self.assertRaises(q.MissingQualificationService): q._registered(forged, registry, 'fixture')

    def test_constructor_rejects_foreign_backend_permit_or_arm_before_callbacks(self):
        # Isolate the post-admission binding checks; no real capability or
        # registry entry is created and no framework/backend body executes.
        from experiments.skill_acquisition_native_v1.ce_native_backend import NativeCEBackend
        token = object()
        context = {'phase': 'qualification', 'arm': 'canonical'}
        backend = object.__new__(NativeCEBackend)
        backend.permit, backend.context = object(), context
        backend.operation = backend.gate = Trap()
        with patch.object(authority, 'require_execution_permit', return_value=None) as admission:
            with self.assertRaisesRegex(authority.NativeAdmissionError, 'exact native permit'):
                q.NativeQualificationExecutor(context, token, backend, {})
            self.assertEqual(admission.call_count, 1)
        backend.permit, backend.context = token, {'phase': 'qualification', 'arm': 'permuted'}
        with patch.object(authority, 'require_execution_permit', return_value=None) as admission:
            with self.assertRaisesRegex(authority.NativeAdmissionError, 'context drift'):
                q.NativeQualificationExecutor(context, token, backend, {})
            self.assertEqual(admission.call_count, 1)
        self.assertFalse(authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS)

    def test_span_forwards_cuda_device_only_after_backend_configuration(self):
        executor = object.__new__(q.NativeQualificationExecutor)
        executor.stage, executor.attempt_id = 'likelihood_gradient_parity', 'pure-spy'
        executor.operation, executor.backend = Trap(), Trap()
        executor._cuda_ready = False
        returned = object()
        with patch.object(q.update, '_span', return_value=returned) as span:
            self.assertIs(executor._span('metadata_setup'), returned)
            self.assertIsNone(span.call_args.args[3])
        device, calls = object(), []
        executor.backend = SimpleNamespace(torch=SimpleNamespace(device=lambda name: calls.append(name) or device))
        executor._cuda_ready = True
        with patch.object(q.update, '_span', return_value=returned) as span:
            self.assertIs(executor._span('qualification_backward', row_id='r0'), returned)
            self.assertIs(span.call_args.args[3], device)
            self.assertEqual(span.call_args.args[2]['qualification_stage'], 'likelihood_gradient_parity')
        self.assertEqual(calls, ['cuda:0'])
        self.assertNotIn('torch', sys.modules)


class QualificationDeclarationTests(unittest.TestCase):
    def test_full_ninety_witnesses_and_separate_eight_gradient_rows_are_bound(self):
        context, receipt = declaration()
        spec = q.validate_execution_spec(context, receipt)
        self.assertEqual(len(spec['all_witness_ids']), 90)
        self.assertEqual(len(spec['gradient_batch_ids']), 8)
        self.assertEqual(len(spec['adam_batches']), 2)
        spec['gradient_batch_ids'].pop()
        self.assertEqual(len(context['bindings']['scope']['qualifier_execution']['gradient_batch_ids']), 8)

    def test_missing_ties_wrong_order_or_unregistered_gradient_and_adam_rows_refuse(self):
        mutations = (lambda s: s['all_witness_ids'].pop(),
            lambda s: s['all_witness_ids'].reverse(), lambda s: s['gradient_batch_ids'].__setitem__(0, 'r99'),
            lambda s: s['gradient_batch_ids'].__setitem__(0, 'r1'),
            lambda s: s['adam_batches'][0].__setitem__(0, 'r99'),
            lambda s: s.update(adam_batches=[s['gradient_batch_ids']]),
            lambda s: s.update(adam_batches=[s['gradient_batch_ids']] * 9))
        for mutate in mutations:
            context, receipt = declaration(); mutate(context['bindings']['scope']['qualifier_execution'])
            with self.assertRaises(authority.NativeAdmissionError): q.validate_execution_spec(context, receipt)

    def test_artifact_owner_and_finite_limits_have_no_defaults_or_parent_escape(self):
        for mutate in (lambda s: s.update(owned_artifact_directory=str(Path(tempfile.gettempdir()).resolve() / 'outside')),
                       lambda s: s.update(owned_artifact_directory=str(Path(s['owned_artifact_directory']) / '..' / '..' / 'outside')),
                       lambda s: s.update(artifact_chunk_bytes=True), lambda s: s.update(artifact_chunk_bytes=q.CHUNK + 4),
                       lambda s: s.update(artifact_chunk_bytes=1025), lambda s: s.pop('reader_bounds')):
            context, receipt = declaration(); mutate(context['bindings']['scope']['qualifier_execution'])
            with self.assertRaises(authority.NativeAdmissionError): q.validate_execution_spec(context, receipt)

    def test_gradient_layout_is_exact_contiguous_f32(self):
        layout, count = q._inventory_offsets([{'name': 'a', 'shape': [2, 3], 'dtype': 'F32'},
                                             {'name': 'b', 'shape': [4], 'dtype': 'F32'}])
        self.assertEqual(count, 10)
        self.assertEqual([item['offset_elements'] for item in layout], [0, 6])
        for bad in ([{'name': 'a', 'shape': [True], 'dtype': 'F32'}],
                    [{'name': 'a', 'shape': [2], 'dtype': 'BF16'}],
                    [{'name': 'a', 'shape': [2], 'dtype': 'F32'}] * 2):
            with self.assertRaises(authority.NativeAdmissionError): q._inventory_offsets(bad)

    def test_actual_stage_source_retains_raw_vectors_and_uses_independent_gradient_formula(self):
        tree = ast.parse(textwrap.dedent(inspect.getsource(q.NativeQualificationExecutor)))
        calls = {node.func.attr for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
        self.assertTrue({'load', 'cached_logprobs', 'teacher_forced_logprobs', 'cross_entropy', 'backward',
            'gradient_checkpointing_disable', 'enable_checkpointing', 'rng_state', 'restore_rng',
            'read_masks', 'read_likelihood', 'read_gradient', 'commit_boundary', 'qualification_delta', 'fsync'} <= calls)
        # Source composition proves neither native execution nor numerical fit.
        self.assertFalse(hasattr(q, 'main'))


if __name__ == '__main__': unittest.main()
