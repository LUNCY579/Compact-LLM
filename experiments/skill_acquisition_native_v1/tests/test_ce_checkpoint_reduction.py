"""Focused pure raw-file/state/negative-control reductions; never native PASS."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.skill_acquisition_native_v1 import ce_checkpoint_reduction as r
from . import test_ce_checkpoint_qualification as cp


def controls_fixture():
    descriptor = {'tensor_hashes': {'param::w': 'a' * 64, 'exp_avg::w': 'b' * 64,
        'exp_avg_sq::w': 'c' * 64, 'step::w': 'd' * 64, 'rng::cpu': 'e' * 64, 'rng::cuda::0': 'f' * 64},
        'specifications': {'param::w': ['F32', [2]], 'exp_avg::w': ['F32', [2]],
            'exp_avg_sq::w': ['F32', [2]], 'step::w': ['F32', []], 'rng::cpu': ['U8', [3]], 'rng::cuda::0': ['U8', [4]]},
        'Adam_initialized': ['w'], 'rng': {'cuda_count': 1, 'python': [3, [1, 2], None]},
        'optimizer_groups': [{'lr': 1e-5}], 'module_training': {'model': False}}
    digest = r.digest(descriptor)
    ops = {}
    for index, (start, end) in enumerate(((1, 2), (5, 6), (8, 9), (12, 13))):
        ops['state%d' % index] = {'kind': 'qualification_state_fingerprint', 'phase': 'returned',
            'started_seq': start, 'returned_seq': end,
            'result': {'state_descriptor': copy.deepcopy(descriptor), 'state_sha256': digest}}
    ops['corrupt'] = {'kind': 'checkpoint_restore', 'phase': 'failed', 'started_seq': 3, 'returned_seq': 4,
                      'result': {'error_type': 'ValueError', 'error': 'latest checkpoint payload bytes corrupt'}}
    ops['restore'] = {'kind': 'checkpoint_restore', 'phase': 'returned', 'started_seq': 10, 'returned_seq': 11}
    controls = dict.fromkeys(('before_state_sha256', 'after_corruption_state_sha256',
        'after_stale_state_sha256', 'recovered_state_sha256'), digest)
    stale_progress = {'C': 2, 'U': 2, 'generation': 0, 'history': [], 'snapshot': 'a' * 64, 'symbolic_snapshot': 'a' * 64}
    stale = dict(stale_progress, schema=r.checkpoint.RESTORE_SCHEMA, progress=stale_progress,
                 install_fence_required=True, manifest_sha256='b' * 64)
    controls.update(corruption_error={'type': 'ValueError', 'message': 'latest checkpoint payload bytes corrupt'},
                    stale_fence_error={'type': 'ValueError', 'message': 'exact ancestor nested restore fence'},
                    corruption_restore_operation={'operation_id': 'corrupt', 'operation': copy.deepcopy(ops['corrupt'])},
                    stale_attempt_record=stale)
    return SimpleNamespace(operations=ops, progress_record=lambda: {'C': 2, 'U': 2, 'generation': 1, 'history': []}), controls


class Tests(unittest.TestCase):
    def test_four_actual_descriptors_and_restore_order_derive_negative_controls(self):
        state, controls = controls_fixture()
        observed = r.reduce_negative_controls(state, controls)
        self.assertTrue(observed['negative_controls_independently_recomputed'])
        self.assertEqual(observed['failed_restore_operations'], 1)
        self.assertEqual(observed['returned_restore_operations'], 1)
    def test_changed_descriptor_or_unrelated_hash_cannot_pass_equal_boolean_claim(self):
        for mutate in (lambda state, values: state.operations['state1']['result']['state_descriptor']['rng'].update(cuda_count=0),
                       lambda state, values: values.update(after_corruption_state_sha256='0' * 64),
                       lambda state, values: state.operations['restore'].update(returned_seq=3)):
            state, values = controls_fixture(); mutate(state, values)
            with self.assertRaises(ValueError): r.reduce_negative_controls(state, values)
    def test_missing_state_descriptor_or_failed_restore_is_not_a_proof(self):
        for mutate in (lambda state: state.operations['state0']['result'].pop('state_descriptor'),
                       lambda state: state.operations.pop('corrupt')):
            state, values = controls_fixture(); mutate(state)
            with self.assertRaises(ValueError): r.reduce_negative_controls(state, values)
    def test_unrelated_exception_or_future_stale_fence_cannot_certify_control(self):
        for mutate in (lambda values: values.update(corruption_error={'type': 'OSError', 'message': 'disk full'}),
                       lambda values: values['stale_attempt_record']['progress'].update(generation=2),
                       lambda values: values['corruption_restore_operation'].update(operation_id='foreign')):
            state, values = controls_fixture(); mutate(values)
            with self.assertRaises(ValueError): r.reduce_negative_controls(state, values)
    def test_actual_saved_parameter_adam_and_rng_files_compare_with_paid_reader(self):
        helper = cp.Contracts(); helper.setUp()
        self.addCleanup(helper.doCleanups)
        left, right = helper.state(helper.left), helper.state(helper.right)
        observed = r.checkpoint.compare_checkpoint_states(helper.reader(), left, helper.left, right, helper.right,
            parameter_inventory=helper.inventory)
        self.assertTrue(observed['parameters_bitwise']); self.assertTrue(observed['torch_cuda_rng_bitwise'])
        self.assertFalse(observed['native_qualified'])
        changed = helper.state(helper.right, lambda tensors: tensors.update({'rng::cuda::0': ('U8', [4], b'xxxx')}))
        with self.assertRaisesRegex(ValueError, 'bytes differ'):
            r.checkpoint.compare_checkpoint_states(helper.reader(), left, helper.left, changed, helper.right,
                parameter_inventory=helper.inventory)
    def test_outer_preflight_denial_precedes_raw_branch_file_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reader = r.raw.RawEvidenceReader(lambda *args, **kw: None,
                {key: 10 ** 10 for key in r.raw.BUDGET_FIELDS}, root)
            binding = {'stage': 'checkpoint_resume', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64,
                'static_recipe_sha256': 'c' * 64, 'code_sha256': {'fixture': 'd' * 64},
                'worker_identity_sha256': 'e' * 64, 'attempt_id': 'parent'}
            refs = {name: {'path': str(root / (name + '.json')), 'sha256': 'a' * 64, 'bytes': 1} for name in r.BRANCHES}
            children = {name: dict(binding, worker_identity_sha256=str(index + 1) * 64) for index, name in enumerate(r.BRANCHES)}
            def denied(*args): raise ValueError('prepaid verification denied')
            with self.assertRaisesRegex(ValueError, 'prepaid verification denied'):
                r.reduce_checkpoint_branches({}, refs, children, binding, reader=reader, outer_preflight=denied)


if __name__ == '__main__': unittest.main()
