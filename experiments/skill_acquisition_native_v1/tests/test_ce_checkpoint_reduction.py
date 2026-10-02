"""Focused pure raw-file/state/negative-control reductions; never native PASS."""
import copy
import hashlib
from pathlib import Path
import tempfile
import struct
from types import SimpleNamespace
import unittest

from experiments.skill_acquisition_native_v1 import ce_checkpoint_reduction as r
from . import test_ce_checkpoint_qualification as cp
from . import test_ce_qualification_accounting as qa


class ThreeBranchFixture:
    """Real files/durable events with explicit simulated computation, no graphs."""
    def __init__(self, root):
        self.root, self.serial = root, 0
        self.context, self.parent_proof, self.receipt = qa.fixture(root / 'parent')
        self.context['prepared_receipt'] = self.receipt
        scope = self.context['bindings']['scope']
        scope['checkpoint_qualification'] = {'owned_directory': str(root / 'checkpoint')}
        scope['native_backend']['output_directory'] = str(root)
        scope['native_backend']['parameter_inventory'] = [{'name': 'w', 'shape': [2], 'dtype': 'F32'}]
        self.parent_proof.update(worker_identity_sha256='f' * 64)
        self.parent_binding = self.binding(self.context, self.parent_proof, 'parent')
        self.branch_refs, self.branch_bindings, self.branches = {}, {}, {}
        self.prefix = None
        for branch, identity in zip(r.BRANCHES, ('1' * 64, '2' * 64, '3' * 64)):
            self.build_branch(branch, identity)
    def write(self, path, value, binding=None, role=None):
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = r.raw.canonical(value) + b'\n'; path.write_bytes(blob)
        ref = {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
        if binding is not None: ref.update(role=role, binding_sha256=r.digest(binding))
        return ref
    def file_ref(self, path):
        blob = path.read_bytes()
        return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    def binding(self, context, proof, attempt):
        return {'stage': 'checkpoint_resume', 'request_sha256': context['request_sha256'],
            'scope_sha256': context['scope_sha256'], 'static_recipe_sha256': context['static_recipe_sha256'],
            'code_sha256': context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': proof['worker_identity_sha256'], 'attempt_id': attempt}
    def commit(self, ledger):
        progress = ledger.progress_record(); C, U = progress['C'], progress['U']
        policy = ledger._ce._expected_policy(); final = ('b' if U == 0 else 'c') * 64
        meta = {'boundary_id': C, 'generation': ledger.generation, 'symbolic_snapshot': ledger.symbolic_snapshot,
            'start_policy_id': policy, 'static_recipe_sha256': ledger.context['static_recipe_sha256'],
            'objective': 'SUPERVISED_EQUAL_ROW_TOKEN_MEAN_CE'}
        def span(kind, fields, output=None):
            self.serial += 1; oid = 'compute-%d' % self.serial
            ledger.operation(kind, 'started', oid, **fields)
            record = dict(fields, **(output or {}), wall_s=0.0)
            ledger.operation(kind, 'returned', oid, **record)
            return dict(record, kind=kind, operation_id=oid)
        before = span('fingerprint', dict(meta, reason='ce_pre_boundary_identity', state_bytes=100),
            {'policy_sha256': policy, 'U': U, 'frozen_state_sha256': 'f' * 64})
        span('checkpoint_configuration', meta); span('gradient_reset', meta)
        rows = []
        for rid in ledger.adam_batches[C]:
            row = ledger._ce.census[rid]
            fields = dict(meta, prompt_id=rid, selected_tokens=row['labeled_tokens'], full_sequence_tokens=row['full_sequence_tokens'],
                **{key: row[key] for key in ('prompt_ids_sha256', 'completion_ids_sha256', 'prepared_row_sha256')})
            forward, backward = span('loss_forward', fields), span('backward', fields)
            rows.append(dict(row, forward_operation=forward, backward_operation=backward,
                selected_predictor_positions=list(range(row['prompt_tokens'] - 1, row['full_sequence_tokens'] - 1)),
                target_positions=list(range(row['prompt_tokens'], row['full_sequence_tokens'])),
                eos_positions_selected=1, prompt_positions_selected=0, pad_positions_selected=0))
        validation = span('gradient_validation', meta, {'finite_confined': True, 'raw_gradient_norm_float64': .5})
        clip = span('gradient_clipping', dict(meta, **ledger.recipe['gradient_clipping']), {'clipped_gradient_norm_float64': .5})
        step = span('optimizer', dict(meta, U_before=U, expected_U_after=U + 1))
        after = span('fingerprint', dict(meta, reason='ce_post_boundary_identity', state_bytes=100),
            {'policy_sha256': final, 'U': U + 1, 'frozen_state_sha256': 'f' * 64})
        span('gradient_cleanup', dict(meta, optimizer_attempted=True))
        result = dict(meta, schema='SKILL-ACQUISITION-CE-BOUNDARY-RESULT-1', synthetic_cpu=True,
            loss_reduction=ledger.recipe['loss_reduction'], accumulation_rows=8, U_before=U, U_after=U + 1,
            updated=True, requires_restore=False, rows=rows, identity_operation=before, final_identity_operation=after,
            final_policy_id=final, frozen_state_sha256='f' * 64, optimizer_operation=step,
            gradient_validation_operation=validation, clipping_operation=clip, raw_gradient_norm_float64=.5,
            clipped_gradient_norm_float64=.5, optimizer_recipe=ledger.recipe['optimizer'], gradient_clipping_recipe=ledger.recipe['gradient_clipping'])
        ledger.commit_boundary(result)
    def restore(self, ledger):
        self.serial += 1; oid = 'restore-%d' % self.serial
        before = ledger.progress_record(); progress = copy.deepcopy(before)
        progress.update(generation=before['generation'] + 1, snapshot=str((before['generation'] + 4) % 10) * 64)
        progress['symbolic_snapshot'] = progress['snapshot']
        record = dict(progress, progress=progress, symbolic_snapshot=progress['snapshot'],
            manifest_sha256='a' * 64, install_fence_required=True)
        ledger.operation('checkpoint_restore', 'started', oid)
        ledger.operation('checkpoint_restore', 'returned', oid, C=progress['C'], U=progress['U'],
            generation=progress['generation'], symbolic_snapshot=progress['snapshot'], manifest_sha256='a' * 64)
        ledger.install_restore(record)
        return record
    def state_file(self, directory, ledger, binding):
        U = ledger.progress_record()['U']
        tensors = {'param::w': ('F32', [2], struct.pack('<2f', 3, 4)),
            'exp_avg::w': ('F32', [2], struct.pack('<2f', .5, .25)),
            'exp_avg_sq::w': ('F32', [2], struct.pack('<2f', .2, .1)),
            'step::w': ('F32', [], struct.pack('<f', U)), 'rng::cpu': ('U8', [3], b'abc'),
            'rng::cuda::0': ('U8', [4], b'defg')}
        header, body = {}, b''
        for name, (dtype, shape, data) in sorted(tensors.items()):
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(body), len(body) + len(data)]}; body += data
        encoded = r.raw.canonical(header); blob = struct.pack('<Q', len(encoded)) + encoded + body
        path = directory / 'state.safetensors'; path.write_bytes(blob)
        original = {'qualification_evidence_binding': binding, 'qualification_batches': ledger.adam_batches,
            'recipe': ledger.recipe, 'code': ledger.context['bindings']['source']['code_sha256'],
            'own_start_sha256': 'a' * 64, 'common_anchor_sha256': 'a' * 64}
        descriptor = {'tensor_hashes': {name: hashlib.sha256(data).hexdigest() for name, (_, _, data) in tensors.items()},
            'specifications': {name: [dtype, shape] for name, (dtype, shape, _) in tensors.items()},
            'Adam_initialized': ['w'], 'rng': {'python': [3, [1, 2], None], 'numpy': ['MT19937', [3, 4], 0, 0, 0.0],
                'cuda_count': 1, 'numpy_version': 'fixture'}, 'optimizer_groups': [{'lr': 1e-5}], 'module_training': {'model': False}}
        manifest = {'schema': r.checkpoint.CHECKPOINT_SCHEMA, 'bindings': original, 'bindings_sha256': r.digest(original),
            'progress': ledger.progress_record(), **{key: value for key, value in descriptor.items() if key != 'tensor_hashes'},
            'Adam_scalar_steps': U, 'frozen_state_sha256': 'f' * 64,
            'payload': {'path': str(path), 'file_sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}}
        manifest['manifest_sha256'] = r.digest(manifest)
        return manifest, self.write(directory / 'manifest.json', manifest, binding, 'checkpoint_manifest'), descriptor
    def negative(self, ledger, descriptor):
        fingerprint = r.digest(descriptor)
        def state():
            self.serial += 1; oid = 'state-%d' % self.serial
            ledger.operation('qualification_state_fingerprint', 'started', oid)
            ledger.operation('qualification_state_fingerprint', 'returned', oid, state_sha256=fingerprint, state_descriptor=descriptor)
        state(); self.serial += 1; bad = 'corruption-%d' % self.serial
        ledger.operation('checkpoint_restore', 'started', bad)
        ledger.operation('checkpoint_restore', 'failed', bad, error={'type': 'ValueError', 'message': 'latest checkpoint payload bytes corrupt'})
        rejected = {'operation_id': bad, 'operation': copy.deepcopy(ledger.operations[bad])}
        state()
        old = ledger.progress_record(); stale = dict(old, schema=r.checkpoint.RESTORE_SCHEMA,
            progress=dict(old, symbolic_snapshot=old['snapshot']), symbolic_snapshot=old['snapshot'],
            manifest_sha256='a' * 64, install_fence_required=True)
        try: ledger.install_restore(stale)
        except ValueError as error: stale_error = {'type': type(error).__name__, 'message': str(error)}
        else: raise AssertionError('stale fence accepted')
        state(); recovery = self.restore(ledger); state()
        return dict.fromkeys(('before_state_sha256', 'after_corruption_state_sha256', 'after_stale_state_sha256', 'recovered_state_sha256'), fingerprint) | {
            'corruption_error': {'type': 'ValueError', 'message': 'latest checkpoint payload bytes corrupt'},
            'stale_fence_error': stale_error, 'corruption_restore_operation': rejected,
            'stale_attempt_record': stale, 'recovery_restore_record': recovery}
    def build_branch(self, branch, identity):
        context = copy.deepcopy(self.context); proof = dict(self.parent_proof, worker_identity_sha256=identity)
        directory = self.root / 'checkpoint' / 'children' / branch; directory.mkdir(parents=True)
        context['bindings']['scope']['qualifier_bootstrap']['owned_directory'] = str(directory / 'bootstrap')
        helper = qa.Tests(); helper.root, helper.context, helper.proof, helper.receipt = directory, context, proof, self.receipt
        lineage = helper.child_payload(branch)
        for key in ('parent_request_sha256',): lineage[key] = self.parent_binding['request_sha256']
        lineage['parent_proof'].update(request_sha256=self.parent_binding['request_sha256'])
        lineage['parent_stage_seal'].update(request_sha256=self.parent_binding['request_sha256'])
        if branch == 'resumed':
            lineage.update(prefix_manifest=self.prefix['manifest'], prefix_manifest_ref=self.prefix['manifest_ref'],
                prefix_checkpoint_bindings_sha256=self.prefix['manifest']['bindings_sha256'],
                inherited_history=self.prefix['manifest']['progress']['history'])
        lineage['declaration_sha256'] = r.digest(r.accounting.checkpoint_child_declaration(lineage))
        context['checkpoint_child_declaration'] = r.accounting.checkpoint_child_declaration(lineage)
        context['checkpoint_child_observed_lineage'] = lineage
        common = context['bindings']['scope']['qualifier_accounting']
        spec = dict(common, owned_directory=str(directory / 'ledger'))
        context['checkpoint_child_accounting'] = {'specification': spec,
            'owner': {'accounting_spec_sha256': r.digest(spec), 'prepared_receipt_sha256': self.receipt['receipt_sha256']}}
        ledger = r.accounting.QualificationAccounting(context, None, proof, self.receipt)
        cap = r.accounting.CheckpointChildAdmission(context, None, proof, lineage)
        ledger.admit_checkpoint_child_prefix(lineage, admission_handle=cap)
        ledger.begin_stage('checkpoint_resume', branch)
        binding = self.binding(context, proof, branch)
        if branch == 'resumed': self.restore(ledger)
        stop = 1 if branch == 'prefix' else 2
        while ledger.progress_record()['U'] < stop: self.commit(ledger)
        _, _, descriptor = self.state_file(directory, ledger, binding)
        controls = self.negative(ledger, descriptor)
        manifest, manifest_ref, _ = self.state_file(directory, ledger, binding)
        controls_ref = self.write(directory / 'controls.json', {'schema': r.raw.SCHEMA, **binding,
            'evidence_kind': 'checkpoint_negative_controls', 'payload': controls}, binding, 'checkpoint_negative_controls')
        ledger.finish_stage([manifest_ref, controls_ref]); ledger.end()
        seal_ref = self.write(directory / 'seal.json', ledger.snapshot())
        context['prepared_receipt'] = self.receipt
        payload = {'schema': r.RESULT, 'branch': branch, 'observed_lineage': lineage, 'child_context': context,
            'ledger_refs': {'header': self.file_ref(ledger.header_path), 'journal': self.file_ref(ledger.path), 'seal': seal_ref},
            'final_manifest_ref': manifest_ref, 'negative_controls_ref': controls_ref}
        result_ref = self.write(directory / 'result.json', {'schema': r.raw.SCHEMA, **binding,
            'evidence_kind': 'checkpoint_branch', 'payload': payload}, binding, 'checkpoint_branch')
        self.branch_refs[branch], self.branch_bindings[branch] = result_ref, binding
        self.branches[branch] = {'payload': payload, 'ledger': ledger, 'directory': directory, 'manifest': manifest, 'manifest_ref': manifest_ref}
        if branch == 'prefix': self.prefix = self.branches[branch]
    def reduce(self):
        reader = r.raw.RawEvidenceReader(lambda *args, **kw: None, {key: 10 ** 12 for key in r.raw.BUDGET_FIELDS}, self.root)
        return r.reduce_checkpoint_branches(self.context, self.branch_refs, self.branch_bindings, self.parent_binding,
            reader=reader, outer_preflight=lambda *args: None)
    def refresh_result(self, branch):
        value = self.branches[branch]
        self.branch_refs[branch] = self.write(value['directory'] / 'result.json', {'schema': r.raw.SCHEMA,
            **self.branch_bindings[branch], 'evidence_kind': 'checkpoint_branch', 'payload': value['payload']},
            self.branch_bindings[branch], 'checkpoint_branch')


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
                      'result': {'error': {'type': 'ValueError', 'message': 'latest checkpoint payload bytes corrupt'}}}
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
    def test_three_real_branch_envelopes_ledgers_restores_and_state_files_reduce(self):
        with tempfile.TemporaryDirectory(prefix='three-branch-reduction-') as directory:
            fixture = ThreeBranchFixture(Path(directory))
            result = fixture.reduce()
            self.assertEqual([result['branches'][name]['physical_optimizer_updates'] for name in r.BRANCHES], [2, 1, 1])
            self.assertEqual([result['branches'][name]['logical_inherited_U'] for name in r.BRANCHES], [0, 0, 1])
            self.assertTrue(result['U2_state_comparison']['parameters_bitwise'])
            self.assertTrue(result['U2_state_comparison']['adam_bitwise'])
            self.assertTrue(result['U2_state_comparison']['torch_cuda_rng_bitwise'])
            self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified'])
            self.assertFalse(result['parent_costs_imported'])
            self.assertEqual(result['child_derived_reservations'], {
                key: sum(fixture.branches[name]['ledger'].cumulative[key] for name in r.BRANCHES)
                for key in r.accounting.COST_FIELDS})
    def test_rehashed_wrong_child_history_cannot_pass_the_original_ledger_semantics(self):
        with tempfile.TemporaryDirectory(prefix='rehashed-child-history-') as directory:
            fixture = ThreeBranchFixture(Path(directory)); child = fixture.branches['resumed']
            path = child['ledger'].path
            entries = [r.raw._json(line) for line in path.read_bytes().splitlines()]
            prefix = next(entry for entry in entries if entry['kind'] == 'checkpoint_child_prefix')
            rows = prefix['payload']['inherited_history'][0]['row_ids']; rows[0], rows[1] = rows[1], rows[0]
            previous, output = None, []
            for entry in entries:
                body = {key: value for key, value in entry.items() if key != 'sha256'}
                body['previous'] = previous
                changed = dict(body, sha256=r.digest(body)); previous = changed['sha256']
                output.append(r.raw.canonical(changed) + b'\n')
            path.write_bytes(b''.join(output))
            child['payload']['ledger_refs']['journal'] = fixture.file_ref(path); fixture.refresh_result('resumed')
            with self.assertRaises(ValueError): fixture.reduce()
    def test_rehashed_refund_and_foreign_parent_are_rejected_end_to_end(self):
        for mutation in ('refund', 'parent'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory(prefix='child-cost-lineage-') as directory:
                fixture = ThreeBranchFixture(Path(directory)); child = fixture.branches['resumed']
                if mutation == 'refund':
                    path = Path(child['payload']['ledger_refs']['seal']['path'])
                    value = r.raw._json(path.read_bytes()); value['cumulative']['io_read_bytes_upper'] -= 1
                    child['payload']['ledger_refs']['seal'] = fixture.write(path, value)
                else:
                    child['payload']['observed_lineage']['parent_worker_identity_sha256'] = '0' * 64
                fixture.refresh_result('resumed')
                with self.assertRaises(ValueError): fixture.reduce()
    def test_rehashed_foreign_endpoint_with_valid_file_and_manifest_hashes_fails_byte_comparison(self):
        with tempfile.TemporaryDirectory(prefix='foreign-child-endpoint-') as directory:
            fixture = ThreeBranchFixture(Path(directory)); child = fixture.branches['resumed']
            manifest = copy.deepcopy(child['manifest']); path = Path(manifest['payload']['path'])
            blob = bytearray(path.read_bytes()); header_bytes = struct.unpack('<Q', blob[:8])[0]
            table = r.raw._json(bytes(blob[8:8 + header_bytes]))
            blob[8 + header_bytes + table['param::w']['data_offsets'][0]] ^= 1
            path.write_bytes(blob); manifest['payload']['file_sha256'] = hashlib.sha256(blob).hexdigest()
            manifest['manifest_sha256'] = r.digest({key: value for key, value in manifest.items() if key != 'manifest_sha256'})
            ref = fixture.write(child['directory'] / 'manifest.json', manifest, fixture.branch_bindings['resumed'], 'checkpoint_manifest')
            child['payload']['final_manifest_ref'] = ref
            fixture.refresh_result('resumed')
            with self.assertRaisesRegex(ValueError, 'bytes differ'): fixture.reduce()
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
