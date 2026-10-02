"""Retained real temporary-file accounting fixtures, without models/frameworks."""
import copy
import hashlib
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from experiments.skill_acquisition_native_v1 import ce_execution_verifier as v
from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as a
from experiments.skill_acquisition_native_v1 import ce_qualification_artifacts as raw
from .test_ce_qualification_accounting import fixture
from . import test_ce_checkpoint_reduction as branches
from . import test_ce_resource_qualification as resources
from . import test_ce_delta_artifacts as deltas
from . import test_ce_delta_rebuild as rebuilds


class ReplayControls(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.context, self.proof, self.receipt = fixture(self.root)
        self.ledger = a.QualificationAccounting(self.context, None, self.proof, self.receipt)
        self.context['prepared_receipt'] = self.receipt
        self.context['allowed_worker_identities'] = [self.proof['worker_identity_sha256']]
        self.context['verification_binding'] = {'stage': 'qualification_verifier',
            'request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256'],
            'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': 'f' * 64, 'attempt_id': 'metadata-verifier-fixture'}
        self.preflights, self.operations = [], []

    def ref(self, path):
        content = Path(path).read_bytes()
        return {'path': str(path), 'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content)}

    def seal(self, change=None):
        value = self.ledger.snapshot()
        if change: change(value)
        path = self.root / 'seal.json'; path.write_bytes(raw.canonical(value) + b'\n')
        return self.ref(path)

    def verify(self, seal_change=None, handles=None, preflight=None):
        reader = raw.RawEvidenceReader(lambda kind, phase, **fields: self.operations.append((kind, phase, fields)),
            {'operation_attempts_upper': 10000, 'io_read_bytes_upper': 1024 ** 3,
             'identity_hash_attempts_upper': 1000000, 'identity_hash_bytes_upper': 8 * 1024 ** 3,
             'reducer_work_bytes_upper': 8 * 1024 ** 3}, self.root)
        return v.verify_qualification(self.context, self.ref(self.ledger.header_path), self.ref(self.ledger.path),
            self.seal(seal_change), reader=reader, worker_handles=handles,
            outer_preflight=preflight or (lambda kind, fields: self.preflights.append((kind, fields))))

    def rewrite(self, mutate):
        entries = [raw._json(line) for line in self.ledger.path.read_bytes().splitlines()]
        mutate(entries)
        previous = None; lines = []
        for seq, entry in enumerate(entries):
            body = {key: value for key, value in entry.items() if key != 'sha256'}
            body['seq'], body['previous'] = seq, previous
            changed = dict(body, sha256=raw.digest(body)); previous = changed['sha256']
            lines.append(raw.canonical(changed) + b'\n')
        self.ledger.path.write_bytes(b''.join(lines))

    def test_setup_worker_replay_derives_totals_and_never_native_pass(self):
        result = self.verify()
        self.assertEqual(result['derived_totals'], self.ledger.cumulative)
        self.assertEqual(result['events'], len(self.ledger.events))
        self.assertEqual(result['status'], 'UNQUALIFIED')
        self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified'])
        self.assertFalse(result['all_stages_complete_by_replay'])
        self.assertEqual(self.preflights[0][0], 'qualification_execution_verifier')

    def test_failed_operation_reservations_are_not_refunded(self):
        self.ledger.begin_stage('runtime_anchor', 'failed-stage')
        self.ledger.operation('model_load', 'started', 'load')
        self.ledger.operation('model_load', 'failed', 'load', error_type='fixture')
        self.ledger.fail_stage(RuntimeError('fixture failed'))
        result = self.verify()
        self.assertEqual(result['derived_totals'], self.ledger.cumulative)
        self.assertEqual(result['derived_attempts'][0]['status'], 'FAILED')
        self.assertGreaterEqual(result['derived_totals']['io_read_bytes_upper'], 10 ** 9)
        with self.assertRaisesRegex(raw.EvidenceRejected, 'totals/refunds'):
            self.verify(lambda seal: seal['cumulative'].update(io_read_bytes_upper=0))

    def test_pending_lifetime_cleanup_is_charged_and_cannot_be_sealed_complete(self):
        self.ledger.begin_stage('runtime_anchor', 'pending-cleanup')
        self.ledger.operation('cleanup', 'started', 'lifetime', deferred_model_cleanup=True)
        result = self.verify()
        self.assertEqual(len(result['pending_operations']), 1)
        self.assertTrue(result['pending_operations'][0]['lifetime_cleanup'])
        self.assertGreaterEqual(result['derived_totals']['wall_seconds_upper'], 1000)
        with self.assertRaisesRegex(raw.EvidenceRejected, 'claims completion'):
            self.verify(lambda seal: seal.update(completed=True))

    def test_tampered_hash_and_partial_tail_fail_without_repair(self):
        content = self.ledger.path.read_bytes()
        changed = content.replace(b'"seq":0', b'"seq":9', 1); self.ledger.path.write_bytes(changed)
        with self.assertRaisesRegex(raw.EvidenceRejected, 'hash/body/order'):
            self.verify()
        self.ledger.path.write_bytes(content[:-1])
        with self.assertRaisesRegex(raw.EvidenceRejected, 'partial ledger tail'):
            self.verify()
        self.assertEqual(self.ledger.path.read_bytes(), content[:-1])

    def test_rehashed_setup_repricing_and_foreign_worker_still_fail(self):
        original = self.ledger.path.read_bytes()
        self.rewrite(lambda entries: entries[0]['payload']['profile'].update(read_bytes_upper=0))
        with self.assertRaisesRegex(raw.EvidenceRejected, 'refund/repricing'):
            self.verify()
        self.ledger.path.write_bytes(original)
        self.rewrite(lambda entries: entries[1]['payload']['proof'].update(worker_identity_sha256='7' * 64))
        with self.assertRaisesRegex(raw.EvidenceRejected, 'foreign worker'):
            self.verify()

    def test_rehashed_worker_source_and_header_recipe_mismatch_rejected(self):
        original = self.ledger.path.read_bytes()
        self.rewrite(lambda entries: entries[1]['payload']['proof'].update(code_sha256={'foreign.py': '8' * 64}))
        with self.assertRaisesRegex(raw.EvidenceRejected, 'worker source'):
            self.verify()
        self.ledger.path.write_bytes(original)
        header = raw._json(self.ledger.header_path.read_bytes()); header['recipe']['optimizer']['lr'] *= 2
        self.ledger.header_path.write_bytes(raw.canonical(header) + b'\n')
        with self.assertRaisesRegex(raw.EvidenceRejected, 'original header'):
            self.verify()

    def test_stage_predecessor_and_attempt_reuse_are_replayed_not_trusted(self):
        self.ledger.begin_stage('runtime_anchor', 'attempt')
        self.ledger.fail_stage(RuntimeError('fixture'))
        self.ledger.begin_stage('runtime_anchor', 'second')
        original = self.ledger.path.read_bytes()
        self.rewrite(lambda entries: entries[-1]['payload'].update(stage='longest_masks'))
        with self.assertRaisesRegex(ValueError, 'predecessor'):
            self.verify()
        self.ledger.path.write_bytes(original)
        self.rewrite(lambda entries: entries[-1]['payload'].update(attempt_id='attempt'))
        with self.assertRaisesRegex(ValueError, 'reused'):
            self.verify()

    def test_preflight_denial_precedes_all_reader_work(self):
        def deny(*args): raise RuntimeError('verification scope denied')
        with self.assertRaisesRegex(RuntimeError, 'scope denied'):
            self.verify(preflight=deny)
        self.assertEqual(self.operations, [])

    def test_adam_artifact_is_recomputed_from_original_commits_and_order(self):
        path = self.root / 'prior.json'; path.write_bytes(b'{"fixture":true}\n')
        for stage in a.q.STAGES[1:a.q.STAGES.index('adam_clipping')]:
            self.ledger.begin_stage(stage, 'prior/' + stage); self.ledger.finish_stage([self.ref(path)])
        self.ledger.begin_stage('adam_clipping', 'adam-two')
        helper = object.__new__(branches.ThreeBranchFixture); helper.serial = 0
        helper.commit(self.ledger); helper.commit(self.ledger)
        commits = [entry['payload']['result'] for entry in self.ledger.events if entry['kind'] == 'commit']
        payload = {'boundaries': commits, 'progress': self.ledger.progress_record(),
            'operation_order': [{'clipping': self.ledger.operation_evidence(result['clipping_operation']['operation_id']),
                'optimizer': self.ledger.operation_evidence(result['optimizer_operation']['operation_id'])} for result in commits]}
        binding = v._binding_for(self.context, {'stage': 'adam_clipping', 'attempt_id': 'adam-two'}, self.proof['worker_identity_sha256'])
        observed = v._adam_readback(self.context, self.ledger, payload, binding)
        self.assertEqual(observed['actual_optimizer_steps'], 2); self.assertTrue(observed['clip_before_adam'])
        wrong = copy.deepcopy(payload); wrong['operation_order'][0]['clipping']['returned_seq'] = 0
        with self.assertRaisesRegex(ValueError, 'operation order/history'):
            v._adam_readback(self.context, self.ledger, wrong, binding)

    def delta_fixture(self):
        helper = rebuilds.Tests(); helper.setUp(); self.addCleanup(helper.doCleanups)
        contract, differences = helper.fixture()
        refs = contract['refs']; output = helper.root / 'output'; output.mkdir()
        for role, ref in refs.items():
            directory = helper.root / (role + '_external') if role in ('base', 'warm') else output
            directory.mkdir(exist_ok=True)
            target = directory / ('model.safetensors' if role == 'base' else role + '.safetensors')
            Path(ref['path']).rename(target); ref['path'] = str(target)
        reference = dict(refs['delta'], role='qualification_delta', binding_sha256=raw.digest(helper.binding))
        context = copy.deepcopy(self.context)
        inventory = contract['selected_inventory']; backend = context['bindings']['scope']['native_backend']
        backend.update(parameter_inventory=inventory, output_directory=str(output), model_dir=str(Path(refs['base']['path']).parent),
            warm_delta_ref=dict(refs['warm'], data_sha256=contract['warm_data_sha256'], state_sha256=contract['warm_state_sha256']))
        context['bindings']['anchors'].update(common_anchor_sha256=contract['common_anchor_sha256'],
            own_start_sha256=contract['common_anchor_sha256'], base_weights_sha256=refs['base']['sha256'])
        bounds = {key: 10 ** 12 for key in raw.BUDGET_FIELDS}
        context['bindings']['scope']['qualifier_execution']['delta_rebuild_reader_bounds'] = {
            role: dict(bounds) for role in ('base', 'warm', 'output')}
        context['bindings']['scope']['publication'] = {'delta_rebuild_wall_seconds_upper': 3600,
            'bounds': {'wall_seconds_upper': 19200}}
        reader = raw.RawEvidenceReader(lambda *args, **fields: self.operations.append((args, fields)), bounds, output)
        saved = deltas.d.read_delta(reader, reference, helper.binding, inventory)
        donor = {'schema': 'SKILL-ACQUISITION-DONOR-DELTA-1', **{key: reference[key] for key in ('path', 'bytes')},
            'file_sha256': reference['sha256'], 'common_anchor_sha256': context['bindings']['anchors']['common_anchor_sha256'],
            'own_start_sha256': context['bindings']['anchors']['own_start_sha256'], 'reference': 'EXACT_REBUILT_COMMON_WARM_ANCHOR',
            **{key: saved[key] for key in ('data_section_sha256', 'state_sha256', 'numel', 'tensors', 'delta_norm')},
            'trainable_names_sha256': raw.digest(sorted(entry['name'] for entry in inventory)),
            'final_master_sha256': contract['final_master_sha256'], 'deployed_arm_sha256': contract['deployed_arm_sha256'],
            'roundtrip': {'residual_nonzero_elements': sum(value != 0 for value in differences), 'residual_nonzero_tensors': 1}}
        observed = {'own_start_sha256': donor['own_start_sha256'], 'common_reference_sha256': donor['common_anchor_sha256'],
            'delta_dtype': 'float32', 'delta_rebuilt_verified': True, 'delta_file_sha256': reference['sha256'],
            'delta_data_sha256': saved['data_section_sha256'], 'final_master_sha256': donor['final_master_sha256'],
            'rebuilt_endpoint_sha256': donor['deployed_arm_sha256'], 'rounding_residual_max_abs': max(abs(value) for value in differences),
            'rounding_residual_l2': sum(value * value for value in differences) ** .5}
        cp_binding = dict(helper.binding, stage='checkpoint_resume', attempt_id='checkpoint')
        manifest = {'schema': 'SKILL-ACQUISITION-NATIVE-CHECKPOINT-1',
            'bindings': {'qualification_evidence_binding': cp_binding, 'inventory': inventory}, 'progress': {'C': 2, 'U': 2},
            'payload': {'path': refs['checkpoint']['path'], 'file_sha256': refs['checkpoint']['sha256'], 'bytes': refs['checkpoint']['bytes']},
            'frozen_state_sha256': contract['checkpoint_frozen_state_sha256']}
        manifest['manifest_sha256'] = raw.digest(manifest)
        manifest_path = output / 'parent.json'; manifest_path.write_bytes(raw.canonical(manifest) + b'\n')
        frozen_runtime, _ = helper.digest({'frozen': [.5], 'rope': [1., .125]})
        source = {'schema': rebuilds.d.SCHEMA, 'selected_inventory': inventory, 'full_state_inventory': contract['full_state_inventory'],
            'nonbase_ref': refs['buffers'], 'checkpoint_binding': cp_binding, 'parent_manifest_ref': self.ref(manifest_path),
            'parent_restore_record': {'manifest_sha256': manifest['manifest_sha256']}, 'base_ref': refs['base'], 'warm_ref': refs['warm'],
            'warm_data_sha256': contract['warm_data_sha256'], 'warm_state_sha256': contract['warm_state_sha256'],
            'common_anchor_sha256': contract['common_anchor_sha256'], 'frozen_runtime_sha256': frozen_runtime}
        source_path = output / 'source.json'
        source_path.write_bytes(raw.canonical(dict(helper.binding, schema=raw.SCHEMA, evidence_kind='delta_full_state', payload=source)) + b'\n')
        source_ref = dict(self.ref(source_path), role='delta_full_state', binding_sha256=raw.digest(helper.binding))
        buffers_ref = dict(refs['buffers'], role='delta_buffers', binding_sha256=raw.digest(helper.binding))
        self.operations.clear()
        return context, {'donor': donor, 'observations': observed, 'full_state_source_ref': source_ref}, reference, helper.binding, reader, source_ref, buffers_ref

    def delta_readback(self, values, *, preflight=None):
        context, payload, reference, binding, reader, source_ref, buffers_ref = values
        return v._delta_readback(context, payload, reference, binding, reader,
            outer_preflight=preflight or (lambda kind, fields: self.preflights.append((kind, fields))),
            source_ref=source_ref, buffers_ref=buffers_ref)

    def test_saved_delta_readback_rebuilds_external_anchor_and_parent_master_bytes(self):
        values = self.delta_fixture(); result = self.delta_readback(values)
        self.assertTrue(result['roundtrip_independently_rebuilt']); self.assertFalse(result['execution_verified'])
        self.assertFalse(result['native_qualified']); self.assertEqual(result['byte_derived_rebuild']['residual_nonzero_elements'], 1)
        self.assertEqual(self.preflights[0][0], 'saved_delta_source_rebuild')
        self.assertEqual(self.preflights[0][1]['wall_seconds_upper'], 3600)
        paths = {fields.get('path') for args, fields in self.operations if fields.get('path')}
        self.assertIn(str(Path(values[0]['bindings']['scope']['native_backend']['model_dir']) / 'model.safetensors'), paths)
        self.assertIn(values[0]['bindings']['scope']['native_backend']['warm_delta_ref']['path'], paths)
        context, payload, reference, binding, reader, source_ref, buffers_ref = values
        changed = copy.deepcopy(payload); changed['donor']['delta_norm'] = 4
        with self.assertRaisesRegex(ValueError, 'independent stream'):
            self.delta_readback((context, changed, reference, binding, reader, source_ref, buffers_ref))
        changed = copy.deepcopy(payload); changed['observations']['rounding_residual_l2'] = 0.
        with self.assertRaisesRegex(ValueError, 'independent saved-byte rebuild'):
            self.delta_readback((context, changed, reference, binding, reader, source_ref, buffers_ref))

    def test_delta_role_linkage_and_source_budget_refuse_before_external_reads(self):
        values = self.delta_fixture()
        def deny(*args): raise RuntimeError('delta scope denied')
        with self.assertRaisesRegex(RuntimeError, 'scope denied'): self.delta_readback(values, preflight=deny)
        self.assertEqual(self.operations, [])
        context, payload, reference, binding, reader, source_ref, buffers_ref = values
        context['bindings']['scope']['publication']['delta_rebuild_wall_seconds_upper'] = 20000
        with self.assertRaisesRegex(ValueError, 'prepaid delta rebuild wall ceiling'): self.delta_readback(values)
        self.assertEqual(self.operations, [])
        context['bindings']['scope']['publication']['delta_rebuild_wall_seconds_upper'] = 3600
        changed = copy.deepcopy(payload); changed['full_state_source_ref']['sha256'] = 'f' * 64
        with self.assertRaisesRegex(ValueError, 'stage-retained full-state'): self.delta_readback((context, changed, reference, binding, reader, source_ref, buffers_ref))
        self.assertFalse(any(fields.get('path') for args, fields in self.operations))
        context['bindings']['scope']['qualifier_execution']['delta_rebuild_reader_bounds']['base']['io_read_bytes_upper'] = 0
        with self.assertRaisesRegex(ValueError, 'read/hash budget exhausted'): self.delta_readback(values)
        self.assertFalse(any(fields.get('path', '').endswith('model.safetensors') for args, fields in self.operations))

    def test_delta_external_path_is_pinned_and_buffers_must_be_retained_by_stage(self):
        values = self.delta_fixture()
        context, payload, reference, binding, reader, source_ref, buffers_ref = values
        with self.assertRaisesRegex(ValueError, 'non-base state artifact linkage'):
            self.delta_readback((context, payload, reference, binding, reader, source_ref, None))
        context['bindings']['scope']['native_backend']['model_dir'] = str(self.root / 'unrelated-model')
        with self.assertRaisesRegex(ValueError, 'pinned external base/warm/common'):
            self.delta_readback(values)

    def test_rehashed_delta_source_cannot_name_a_foreign_checkpoint_worker(self):
        values = self.delta_fixture()
        context, payload, reference, binding, reader, source_ref, buffers_ref = values
        path = Path(source_ref['path']); envelope = raw._json(path.read_bytes())
        envelope['payload']['checkpoint_binding']['worker_identity_sha256'] = 'f' * 64
        path.write_bytes(raw.canonical(envelope) + b'\n')
        changed_ref = dict(self.ref(path), role=source_ref['role'], binding_sha256=source_ref['binding_sha256'])
        payload['full_state_source_ref'] = changed_ref
        with self.assertRaisesRegex(ValueError, 'same admitted parent worker/source'):
            self.delta_readback((context, payload, reference, binding, reader, changed_ref, buffers_ref))
        self.assertFalse(any(fields.get('path', '').endswith('model.safetensors') for args, fields in self.operations))

    def test_runtime_readback_requires_actual_loader_identity_and_original_operation_rows(self):
        context = self.context; anchors = context['bindings']['anchors']
        context['bindings']['runtime']['decoding_preimage'] = {'device': 'cpu', 'dtype': 'float32', 'fixture': True}
        anchors.update(base_weights_sha256='1' * 64, tokenizer_receipt_sha256='2' * 64)
        backend = context['bindings']['scope']['native_backend']
        backend.update(parameter_inventory=[{'name': 'w', 'shape': [2], 'dtype': 'F32'}],
            warm_delta_ref={'data_sha256': '3' * 64})
        self.ledger.begin_stage('runtime_anchor', 'runtime')
        for kind in ('configured_runtime', 'process_rng_seed', 'model_load', 'inventory_configuration', 'fingerprint'):
            self.ledger.operation(kind, 'started', kind)
            self.ledger.operation(kind, 'returned', kind)
        binding = v._binding_for(context, {'stage': 'runtime_anchor', 'attempt_id': 'runtime'}, self.proof['worker_identity_sha256'])
        observed = {'actual_anchor_sha256': anchors['own_start_sha256'], 'tokenizer_receipt_sha256': anchors['tokenizer_receipt_sha256'],
            'actual_runtime_preimage': context['bindings']['runtime']['decoding_preimage'],
            'decoder_checkpointing': {'use_reentrant': False}, 'head_checkpointing': True}
        payload = {'observations': observed, 'inventory': backend['parameter_inventory'], 'worker_proof': self.proof,
            'identity': {'state_fingerprint': anchors['own_start_sha256'], 'weights_sha256': anchors['base_weights_sha256'],
                'anchor': {'base_weights_sha256': anchors['base_weights_sha256'], 'delta_sha256': backend['warm_delta_ref']['data_sha256'],
                    'arm_sha256': anchors['common_anchor_sha256']}}}
        result = v._runtime_readback(context, self.ledger, payload, binding)
        self.assertTrue(result['loader_identity_and_operation_linkage_consistent']); self.assertFalse(result['execution_verified'])
        wrong = copy.deepcopy(payload); wrong['identity']['anchor']['delta_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'loader/anchor identity'):
            v._runtime_readback(context, self.ledger, wrong, binding)

    def test_resource_readback_requires_original_probe_rows_and_unmodified_prefix_costs(self):
        context, proof, receipt = fixture(self.root / 'resources')
        context['prepared_receipt'] = receipt; proof['os_identity'] = {'pid': 1234}
        scope = context['bindings']['scope']; scope['qualifier_resources'] = {'gpu_uuid': resources.GPU}
        limits = resources.limits(); spec = {'required_barriers': ['launch', 'resource_io_measurement'],
            'measured_operation_kinds': ['resource_probe'], 'limits': limits}
        scope['resource_measurement'] = spec
        ledger = a.QualificationAccounting(context, None, proof, receipt)
        dummy = self.root / 'resources' / 'prior.json'; dummy.write_bytes(b'{"fixture":true}\n')
        for stage in a.q.STAGES[1:a.q.STAGES.index('resource_io_measurement')]:
            ledger.begin_stage(stage, 'prior/' + stage); ledger.finish_stage([self.ref(dummy)])
        ledger.begin_stage('resource_io_measurement', 'resource-readback')
        samples = resources.samples()
        for index, sample in enumerate(samples):
            oid = 'probe-%d' % index
            ledger.operation('resource_probe', 'started', oid)
            ledger.operation('resource_probe', 'returned', oid, bytes_read=100, bytes_written=0,
                identity_hash_attempts_actual=1, identity_hash_bytes_actual=100, wall_s=.25,
                resource_observation={key: value for key, value in sample.items() if key != 'stage'})
        snapshot = ledger.snapshot()
        observed = {}
        observed.update(resources.q.reduce_samples(samples, spec['required_barriers'], worker_pid=1234,
            gpu_uuid=resources.GPU, limits=limits))
        observed.update(resources.q.reduce_operations(ledger.operations, spec['measured_operation_kinds'],
            worker_identity_sha256=proof['worker_identity_sha256']))
        payload = {'required_barriers': spec['required_barriers'], 'measured_operation_kinds': spec['measured_operation_kinds'],
            'limits': limits, 'resource_samples': samples, 'raw_operations': copy.deepcopy(ledger.operations),
            'accounting_snapshot': snapshot, 'worker_proof': copy.deepcopy(proof), 'observations': observed}
        binding = v._binding_for(context, {'stage': 'resource_io_measurement', 'attempt_id': 'resource-readback'}, proof['worker_identity_sha256'])
        reader = raw.RawEvidenceReader(lambda *args, **fields: None, {key: 10 ** 12 for key in raw.BUDGET_FIELDS}, self.root)
        result = v._resource_readback(context, ledger, payload, binding, reader=reader, reference_bytes=len(raw.canonical(payload)))
        self.assertEqual(result['peak_reserved_gib'], 2); self.assertEqual(result['operation_measurements'][0]['io_read_bytes'], 100)
        changed = copy.deepcopy(payload); changed['resource_samples'][1]['cuda_peak_reserved_bytes'] -= 1
        with self.assertRaisesRegex(ValueError, 'original returned probe'):
            v._resource_readback(context, ledger, changed, binding, reader=reader, reference_bytes=len(raw.canonical(changed)))
        changed = copy.deepcopy(payload); changed['accounting_snapshot']['cumulative']['io_read_bytes_upper'] -= 1
        with self.assertRaisesRegex(ValueError, 'totals/refunds'):
            v._resource_readback(context, ledger, changed, binding, reader=reader, reference_bytes=len(raw.canonical(changed)))

    def test_checkpoint_bundle_replays_retained_parent_prefixes_and_all_three_children(self):
        tree = object.__new__(branches.ThreeBranchFixture)
        tree.root, tree.serial = self.root / 'bundle', 0
        tree.root.mkdir()
        tree.context, _, tree.receipt = fixture(tree.root / 'parent')
        tree.context['prepared_receipt'] = tree.receipt
        scope = tree.context['bindings']['scope']
        scope['checkpoint_qualification'] = {'owned_directory': str(tree.root / 'checkpoint')}
        scope['native_backend'].update(output_directory=str(tree.root), parameter_inventory=[{'name': 'w', 'shape': [2], 'dtype': 'F32'}])
        tree.parent_proof = {'schema': a.PROOF, 'fresh_process': True, 'boot_verified': True,
            'worker_identity_sha256': 'f' * 64, 'boot_identity_sha256': 'd' * 64, 'pid': 11, 'creation_filetime_100ns': 100,
            'request_sha256': tree.context['request_sha256'], 'scope_sha256': tree.context['scope_sha256'],
            'code_sha256': scope.get('code_sha256', tree.context['bindings']['source']['code_sha256'])}
        parent = a.QualificationAccounting(tree.context, None, tree.parent_proof, tree.receipt)
        dummy = tree.root / 'prior.json'; dummy.write_bytes(b'{"fixture":true}\n')
        for stage in a.q.STAGES[1:a.q.STAGES.index('adam_clipping')]:
            parent.begin_stage(stage, 'prior/' + stage); parent.finish_stage([tree.file_ref(dummy)])
        parent.begin_stage('adam_clipping', 'parent-adam'); tree.commit(parent); tree.commit(parent)
        parent.finish_stage([tree.file_ref(dummy)]); parent.begin_stage('checkpoint_resume', 'parent')
        tree.parent_binding = tree.binding(tree.context, tree.parent_proof, 'parent')
        parent.operation('checkpoint_save', 'started', 'parent-save')
        manifest, manifest_ref, _ = tree.state_file(tree.root / 'parent', parent, tree.parent_binding, 'parent-save')
        parent.operation('checkpoint_save', 'returned', 'parent-save', manifest_sha256=manifest['manifest_sha256'])
        def retained(prefix):
            directory = tree.root / prefix; directory.mkdir()
            header = directory / 'header.json'; header.write_bytes(parent.header_path.read_bytes())
            journal = directory / 'journal.jsonl'; journal.write_bytes(parent.path.read_bytes())
            return {'header': tree.file_ref(header), 'journal': tree.file_ref(journal),
                'seal': tree.write(directory / 'snapshot.json', parent.snapshot())}
        original = retained('initial')
        progress = parent.progress_record()
        restored_progress = dict(copy.deepcopy(progress), generation=progress['generation'] + 1, snapshot='4' * 64, symbolic_snapshot='4' * 64)
        restored = dict(restored_progress, schema=branches.r.checkpoint.RESTORE_SCHEMA, progress=restored_progress,
            manifest_sha256=manifest['manifest_sha256'], install_fence_required=True, source_worker_binding=tree.parent_binding)
        parent.operation('checkpoint_restore', 'started', 'parent-restore')
        parent.operation('checkpoint_restore', 'returned', 'parent-restore', C=2, U=2, generation=restored_progress['generation'],
            symbolic_snapshot='4' * 64, manifest_sha256=manifest['manifest_sha256'])
        parent.install_restore(restored)
        post = retained('post')
        tree.branch_refs, tree.branch_bindings, tree.branches, tree.prefix = {}, {}, {}, None
        for branch, identity in zip(branches.r.BRANCHES, ('1' * 64, '2' * 64, '3' * 64)):
            tree.build_branch(branch, identity)
        terminals = {name: {'worker_identity_sha256': binding['worker_identity_sha256'],
            'request_sha256': tree.context['request_sha256'], 'scope_sha256': tree.context['scope_sha256'],
            'status_file_used': False, 'exit_code': 0} for name, binding in tree.branch_bindings.items()}
        bundle = {'schema': 'CE8-CHECKPOINT-BRANCHES-BUNDLE-1', 'parent_binding': tree.parent_binding,
            'parent_ledger_refs': original, 'parent_post_restore_ledger_refs': post,
            'parent_post_restore_snapshot_ref': post['seal'], 'parent_manifest_ref': manifest_ref,
            'parent_restore_record': restored, 'branch_refs': tree.branch_refs, 'branch_bindings': tree.branch_bindings,
            'child_terminal_proofs': terminals, 'native_qualified': False, 'execution_verified': False}
        reader = raw.RawEvidenceReader(lambda *args, **fields: None, {key: 10 ** 12 for key in raw.BUDGET_FIELDS}, tree.root)
        result = v._checkpoint_readback(tree.context, parent, bundle, tree.parent_binding,
            reader=reader, outer_preflight=lambda *args: None)
        self.assertTrue(result['observations']['parameters_bitwise']); self.assertFalse(result['execution_verified'])
        wrong = copy.deepcopy(bundle); wrong['parent_restore_record']['manifest_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'returned current-parent restore'):
            v._checkpoint_readback(tree.context, parent, wrong, tree.parent_binding,
                reader=reader, outer_preflight=lambda *args: None)

    def test_checkpoint_child_artifact_binds_separate_raw_request_to_same_study(self):
        attempt = {'stage': 'checkpoint_resume', 'attempt_id': 'parent-cp', 'status': 'RETURNED'}
        parent = v._binding_for(self.context, attempt, self.proof['worker_identity_sha256'])
        child = dict(parent, worker_identity_sha256='6' * 64, attempt_id='child-cp')
        request_path = self.root / 'child-request.json'; request_path.write_bytes(b'{"branch":"uninterrupted"}\n')
        request_ref = self.ref(request_path)
        lineage = {'schema': a.CHILD_PREFIX, 'worker_identity_sha256': child['worker_identity_sha256'],
            'study_request_sha256': child['request_sha256'], 'child_request_sha256': request_ref['sha256'],
            'child_request_ref': request_ref, 'parent_request_sha256': parent['request_sha256'],
            'parent_worker_identity_sha256': parent['worker_identity_sha256'],
            **{key: parent[key] for key in ('scope_sha256', 'static_recipe_sha256', 'code_sha256')},
            'parent_proof': {key: parent[key] for key in ('request_sha256', 'scope_sha256', 'worker_identity_sha256')},
            'child_proof': {**{key: child[key] for key in ('request_sha256', 'scope_sha256', 'worker_identity_sha256')},
                'sources': {'request_ref': request_ref}}}
        artifact = dict(request_ref, role='checkpoint_branch', binding_sha256=raw.digest(child))
        attempt['evidence'] = {'raw_refs': [artifact]}
        state = SimpleNamespace(attempts=[attempt])
        reductions, gaps = v._reductions(self.context, state, {'parent-cp': parent['worker_identity_sha256']},
            None, {artifact['path']: child}, {child['worker_identity_sha256']: lineage})
        self.assertEqual(reductions, {})
        self.assertTrue(any('OS handle' in gap for gap in gaps))
        wrong = copy.deepcopy(lineage); wrong['child_request_sha256'] = child['request_sha256']
        with self.assertRaisesRegex(ValueError, 'raw child request hash'):
            v._reductions(self.context, state, {'parent-cp': parent['worker_identity_sha256']}, None,
                {artifact['path']: child}, {child['worker_identity_sha256']: wrong})

    def test_supplied_worker_flags_cannot_replace_actual_execution_handles(self):
        with self.assertRaisesRegex(raw.EvidenceRejected, 'source-issued handles'):
            self.verify(handles={self.proof['worker_identity_sha256']: {'native_actual': True, 'terminal': True}})


if __name__ == '__main__': unittest.main()
