"""Actual stdlib files/replay/copy fixtures, explicitly no native computation."""
import copy
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_checkpoint_initializer as i
from experiments.skill_acquisition_native_v1 import checkpoint as base
from . import test_ce_qualification_accounting as qa
from . import test_ce_checkpoint_reduction as branches


class Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='checkpoint-initializer-pure-')
        self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        self.events = []
        self.operation = lambda kind, phase, **fields: self.events.append((kind, phase, fields))
        self.bounds = {key: 10 ** 12 for key in i.raw.BUDGET_FIELDS}
        self.binding = {'stage': 'checkpoint_resume', 'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64,
            'static_recipe_sha256': 'c' * 64, 'code_sha256': {'fixture': 'd' * 64},
            'worker_identity_sha256': 'e' * 64, 'attempt_id': 'prefix'}
        self.profile = {'read_bytes_upper': 10 ** 7, 'write_bytes_upper': 10 ** 7,
            'identity_hash_attempts_upper': 1000, 'identity_hash_bytes_upper': 10 ** 7, 'wall_s_upper': 1000}
    def reader(self): return i.raw.RawEvidenceReader(self.operation, self.bounds, self.root)
    def file_ref(self, path):
        blob = path.read_bytes(); return {'path': str(path), 'sha256': hashlib.sha256(blob).hexdigest(), 'bytes': len(blob)}
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(i.raw.canonical(value) + b'\n')
        return self.file_ref(path)
    def store(self):
        directory = self.root / 'original'; directory.mkdir()
        inventory = [{'name': 'w', 'shape': [2], 'dtype': 'F32'}]
        tensors = {'param::w': ('F32', [2], struct.pack('<2f', 3, 4)),
            'exp_avg::w': ('F32', [2], struct.pack('<2f', .5, .25)),
            'exp_avg_sq::w': ('F32', [2], struct.pack('<2f', .2, .1)),
            'step::w': ('F32', [], struct.pack('<f', 1)), 'rng::cpu': ('U8', [3], b'abc'),
            'rng::cuda::0': ('U8', [4], b'defg')}
        header, body = {}, b''
        for name, (dtype, shape, data) in sorted(tensors.items()):
            header[name] = {'dtype': dtype, 'shape': shape, 'data_offsets': [len(body), len(body) + len(data)]}; body += data
        encoded = i.raw.canonical(header); payload = directory / 'checkpoint-000000001.safetensors'
        payload.write_bytes(struct.pack('<Q', len(encoded)) + encoded + body)
        payload_ref = self.file_ref(payload)
        bindings = {'qualification_evidence_binding': self.binding, 'qualification_batches': [['r%d' % n for n in range(8)], ['r%d' % n for n in range(8, 16)]], 'inventory': inventory}
        history = [{'boundary': 0, 'row_ids': bindings['qualification_batches'][0], 'updated': True}]
        manifest = {'schema': i.cp.CHECKPOINT_SCHEMA, 'sequence': 1, 'save_operation_id': 'saved',
            'bindings': bindings, 'bindings_sha256': i.raw.digest(bindings), 'inventory': inventory,
            'inventory_sha256': i.raw.digest(inventory), 'Adam_initialized': {'w': True}, 'Adam_scalar_steps': 1,
            'rng': {'cuda_count': 1}, 'progress': {'C': 1, 'U': 1, 'history': history, 'generation': 0, 'snapshot': 'f' * 64},
            'specifications': {name: [dtype, shape] for name, (dtype, shape, data) in tensors.items()},
            'payload': {'path': payload_ref['path'], 'bytes': payload_ref['bytes'], 'file_sha256': payload_ref['sha256']}}
        manifest['manifest_sha256'] = i.raw.digest(manifest)
        refs = {'bindings': self.write(directory / base.BINDING_FILE, bindings),
            'manifest': self.write(directory / 'checkpoint-000000001.json', manifest), 'payload': payload_ref}
        rows, previous = [], None
        for seq, fields in enumerate(({'kind': 'manifest_intent', 'sequence': 1, 'operation_id': 'saved', 'manifest_sha256': manifest['manifest_sha256']},
                {'kind': 'operation', 'operation': 'checkpoint_save', 'phase': 'returned', 'operation_id': 'saved', 'manifest_sha256': manifest['manifest_sha256']})):
            row = dict(fields, seq=seq, previous=previous, binding_sha256=manifest['bindings_sha256'])
            row['sha256'] = i.raw.digest(row); previous = row['sha256']; rows.append(row)
        journal = directory / base.JOURNAL_FILE; journal.write_bytes(b''.join(i.raw.canonical(row) + b'\n' for row in rows))
        refs['journal'] = self.file_ref(journal)
        return refs, manifest
    def test_native_initializer_denies_before_any_object_or_path_traversal(self):
        class Trap:
            def __getattribute__(self, name): raise AssertionError('unadmitted traversal')
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        with patch.object(Path, 'open', side_effect=AssertionError('unadmitted file')):
            with self.assertRaises(ValueError):
                i.CheckpointChildInitializer(Trap(), None, Trap(), operation=Trap(), gate=Trap(), reader=Trap(), bootstrap_handle=Trap())
        self.assertEqual(before, set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'})
        self.assertFalse(i.accounting._REGISTERED_CHILD_PREFIX_VERIFIERS)
    def test_original_four_file_publication_chain_and_streamed_payload_are_verified(self):
        refs, manifest = self.store()
        self.assertEqual(i.read_checkpoint_store(self.reader(), refs, self.binding, maximum_bytes=100000), manifest)
        self.assertTrue(any(kind == 'qualification_artifact_readback' and phase == 'returned' for kind, phase, _ in self.events))
        forged = copy.deepcopy(manifest); forged['manifest_sha256'] = 'a' * 64
        with self.assertRaisesRegex(ValueError, 'observed lineage'):
            i.read_checkpoint_store(self.reader(), refs, self.binding, expected_manifest=forged, maximum_bytes=100000)
    def test_missing_publication_chain_partial_tail_and_payload_change_refuse(self):
        refs, _ = self.store(); path = Path(refs['journal']['path']); original = path.read_bytes()
        for blob in (original + b'{partial', original.splitlines(keepends=True)[0]):
            path.write_bytes(blob); changed = dict(refs, journal=self.file_ref(path))
            with self.assertRaises(ValueError): i.read_checkpoint_store(self.reader(), changed, self.binding, maximum_bytes=100000)
        path.write_bytes(original)
        payload = Path(refs['payload']['path']); payload.write_bytes(payload.read_bytes()[:-1])
        with self.assertRaises(ValueError): i.read_checkpoint_store(self.reader(), refs, self.binding, maximum_bytes=100000)
    def test_copy_preserves_original_four_files_and_isolates_real_byte_corruption(self):
        refs, _ = self.store(); original = {key: Path(ref['path']).read_bytes() for key, ref in refs.items()}
        target = self.root / 'children' / 'prefix' / 'corruption'
        reserved = []
        result = i.copy_checkpoint_store(self.reader(), refs, self.binding, target, owned_root=self.root,
            operation=self.operation, gate=lambda stage: None, profile=self.profile,
            metadata_reserve=lambda *args, **kw: reserved.append((args, kw)), corrupt_last_byte=True)
        for key, ref in refs.items(): self.assertEqual(Path(ref['path']).read_bytes(), original[key])
        for key in ('bindings', 'journal', 'manifest'):
            self.assertEqual(Path(result['references'][key]['path']).read_bytes(), original[key])
        mutated = Path(result['references']['payload']['path']).read_bytes()
        self.assertEqual(mutated[:-1], original['payload'][:-1]); self.assertNotEqual(mutated[-1:], original['payload'][-1:])
        self.assertEqual(result['mutation']['mutated_sha256'], hashlib.sha256(mutated).hexdigest())
        self.assertFalse(result['native_qualified']); self.assertEqual(len(reserved), 3)
    def test_copy_admission_rejection_precedes_destination_and_failed_hash_keeps_charge(self):
        refs, _ = self.store(); target = self.root / 'denied'
        def deny(kind, phase, **fields): raise ValueError('prepaid copy denied')
        with self.assertRaisesRegex(ValueError, 'prepaid copy denied'):
            i.copy_checkpoint_store(self.reader(), refs, self.binding, target, owned_root=self.root,
                operation=deny, gate=lambda stage: None, profile=self.profile, metadata_reserve=lambda *a, **kw: None)
        self.assertFalse(target.exists())
        changed = copy.deepcopy(refs); changed['bindings']['sha256'] = '0' * 64
        reader = self.reader(); before = dict(reader.remaining)
        with self.assertRaises(ValueError):
            i.copy_checkpoint_store(reader, changed, self.binding, target, owned_root=self.root,
                operation=self.operation, gate=lambda stage: None, profile=self.profile, metadata_reserve=lambda *a, **kw: None)
        self.assertLess(reader.remaining['io_read_bytes_upper'], before['io_read_bytes_upper'])
        self.assertTrue(any(k == 'checkpoint_adapter_io' and p == 'failed' for k, p, _ in self.events))
        self.assertFalse(any(k == 'checkpoint_adapter_io' and p == 'returned' for k, p, _ in self.events))
    def test_returned_restore_binding_requires_actual_current_child_operation(self):
        progress = {'C': 1, 'U': 1, 'history': [{'updated': True}], 'generation': 0, 'snapshot': 'a' * 64}
        restored = dict(progress, generation=1, snapshot='b' * 64, symbolic_snapshot='b' * 64)
        restored = dict(restored, progress=copy.deepcopy(restored), manifest_sha256='c' * 64, install_fence_required=True)
        op = {'kind': 'checkpoint_restore', 'phase': 'returned', 'worker_id': self.binding['worker_identity_sha256'],
            'attempt_id': self.binding['attempt_id'], 'stage': 'checkpoint_resume',
            'result': {key: restored[key] for key in ('manifest_sha256', 'C', 'U', 'generation', 'symbolic_snapshot')}}
        owner = SimpleNamespace(operations={'restored': op}, progress_record=lambda: progress)
        record, bound = i.bind_returned_restore(owner, restored, self.binding)
        self.assertEqual(record['source_worker_binding'], self.binding); self.assertNotIn('source_worker_binding', op)
        self.assertEqual(bound['source_worker_binding'], self.binding)
        op['worker_id'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'current-child'): i.bind_returned_restore(owner, restored, self.binding)

    def test_parent_observer_views_share_only_exact_stable_os_and_raw_source_identity(self):
        from experiments.skill_acquisition_native_v1 import ce_worker_evidence as evidence
        ref = {'path': str(self.root / 'pinned.bin'), 'sha256': 'a' * 64, 'bytes': 10}
        identity = {'schema': evidence.IDENTITY, 'pid': 11, 'creation_filetime_100ns': 100,
            'boot_identity': 'fixture-boot', 'image_path': ref['path'], 'argv': ['python', '-m', 'fixture']}
        sources = {'python_ref': ref, 'process_image_ref': ref, 'request_ref': ref,
            'source_files': {'fixture.py': ref}, 'entry': 'fixture.py', 'entry_script_ref': None,
            'runtime_launcher': 'parent-current', 'source_observation': 'current observer'}
        retained = {'schema': evidence.PROOF, 'os_identity': identity, 'sources': sources,
            'request_sha256': 'b' * 64, 'scope_sha256': 'c' * 64}
        retained['worker_identity_sha256'] = evidence.digest(evidence.worker_identity_preimage(identity, sources, 'b' * 64, 'c' * 64))
        remote = copy.deepcopy(retained); remote['sources'].update(runtime_launcher='remote-child', source_observation='remote observer')
        self.assertEqual(i.validate_parent_view(remote, retained), remote)
        for change in (lambda view: view['os_identity'].update(creation_filetime_100ns=101),
                       lambda view: view['sources']['request_ref'].update(sha256='d' * 64),
                       lambda view: view['sources']['source_files']['fixture.py'].update(bytes=11)):
            bad = copy.deepcopy(remote); change(bad)
            with self.assertRaisesRegex(ValueError, 'OS/source identity drift'): i.validate_parent_view(bad, retained)
    def parent_fixture(self):
        context, proof, receipt = qa.fixture(self.root)
        context['prepared_receipt'] = receipt
        context['bindings']['scope']['checkpoint_qualification'] = {'schema': i.cp.SPEC, 'owned_directory': str(self.root / 'checkpoint'),
            'child_initializer': {'schema': i.SPEC, 'reader_bounds': self.bounds, 'max_store_bytes': 10 ** 7,
                'branch_attempt_ids': {name: 'child/' + name for name in i.cp.BRANCHES}}}
        proof.update(request_sha256=context['request_sha256'], scope_sha256=context['scope_sha256'],
            code_sha256=context['bindings']['source']['code_sha256'], pid=11, creation_filetime_100ns=10)
        parent = i.accounting.QualificationAccounting(context, None, proof, receipt)
        for stage in i.accounting.q.STAGES[1:i.accounting.q.STAGES.index('adam_clipping')]:
            parent.begin_stage(stage, stage); parent.finish_stage([self.write(self.root / (stage + '.json'), {'fixture': True})])
        parent.begin_stage('adam_clipping', 'parent-adam')
        helper = object.__new__(branches.ThreeBranchFixture); helper.serial = 0
        helper.commit(parent); helper.commit(parent)
        parent.finish_stage([self.write(self.root / 'adam.json', {'fixture': True})]); parent.begin_stage('checkpoint_resume', 'parent-cp')
        binding = helper.binding(context, proof, 'parent-cp')
        refs = {'header': self.file_ref(parent.header_path), 'journal': self.file_ref(parent.path),
            'seal': self.write(self.root / 'parent-stage.json', parent.snapshot())}
        stage = {'schema': 'CE8-CHECKPOINT-PARENT-STAGE-SEAL-1', 'stage': 'checkpoint_resume', 'completed': False,
            'worker_open': True, 'C': 2, 'U': 2, 'worker_identity_sha256': proof['worker_identity_sha256'],
            'request_sha256': context['request_sha256'], 'scope_sha256': context['scope_sha256'],
            'ledger_head': parent._head(), 'binding_sha256': parent.binding_sha256}
        return context, proof, receipt, binding, refs, stage
    def test_live_parent_raw_replay_derives_costs_and_refuses_forged_stage_or_refund(self):
        context, proof, receipt, binding, refs, stage = self.parent_fixture()
        lineage = {'parent_proof': proof, 'parent_stage_seal': stage}
        observed = i.read_parent_evidence(self.reader(), context, refs, binding, lineage)
        self.assertEqual(observed['ledger_head'], stage['ledger_head']); self.assertFalse(observed['execution_verified'])
        bad = copy.deepcopy(lineage); bad['parent_stage_seal']['ledger_head'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'raw replay'): i.read_parent_evidence(self.reader(), context, refs, binding, bad)
        snapshot = i.raw._json(Path(refs['seal']['path']).read_bytes()); snapshot['cumulative']['io_read_bytes_upper'] -= 1
        refs['seal'] = self.write(Path(refs['seal']['path']), snapshot)
        with self.assertRaisesRegex(ValueError, 'totals/refunds'): i.read_parent_evidence(self.reader(), context, refs, binding, lineage)
    def test_concrete_initializer_opens_independent_child_ledger_under_common_scope(self):
        parent, parent_proof, receipt, binding, refs, stage = self.parent_fixture()
        child = copy.deepcopy(parent); child_proof = dict(parent_proof, worker_identity_sha256='1' * 64, pid=12, creation_filetime_100ns=11)
        request_ref = self.write(self.root / 'immutable-child-request.json', {'fixture': True})
        child_proof['sources'] = {'request_ref': request_ref}
        payload = {'schema': i.accounting.CHILD_PREFIX, 'branch': 'uninterrupted', 'inherited_U': 0, 'inherited_history': [],
            'study_request_sha256': child['request_sha256'], 'parent_request_sha256': parent['request_sha256'],
            'scope_sha256': child['scope_sha256'], 'static_recipe_sha256': child['static_recipe_sha256'],
            'code_sha256': child['bindings']['source']['code_sha256'], **child['bindings']['anchors'],
            'qualification_batches': child['bindings']['scope']['qualifier_execution']['adam_batches'],
            'ingestion_receipt_sha256': receipt['receipt_sha256'], 'worker_identity_sha256': child_proof['worker_identity_sha256'],
            'parent_worker_identity_sha256': parent_proof['worker_identity_sha256'], 'parent_proof': parent_proof,
            'parent_stage_seal': stage, 'parent_ledger_refs': refs, 'child_request_ref': request_ref,
            'child_request_sha256': request_ref['sha256'], 'child_proof': child_proof}
        child['checkpoint_child_declaration'] = i.accounting.checkpoint_child_declaration(payload)
        payload['declaration_sha256'] = i.raw.digest(child['checkpoint_child_declaration'])
        child['checkpoint_child_observed_lineage'] = payload
        child['checkpoint_child_inputs'] = {'parent_context': parent, 'parent_binding': binding}
        common = child['bindings']['scope']['qualifier_accounting']
        spec = dict(common, owned_directory=str(i.child_directory(child, 'uninterrupted') / 'ledger'))
        owner = {'accounting_spec_sha256': i.raw.digest(spec), 'prepared_receipt_sha256': receipt['receipt_sha256']}
        child['checkpoint_child_accounting'] = {'specification': spec, 'owner': owner}
        child['checkpoint_child_bootstrap'] = dict(child['bindings']['scope']['qualifier_bootstrap'],
            owned_directory=str(i.child_directory(child, 'uninterrupted') / 'bootstrap'))
        outer = i.bootstrap.BootstrapAccounting(child, None, child_proof)
        bounds = i.accounting._price(spec['setup_bounds']); bounds['metadata_bytes_upper'] = 2 * i.raw.MAX_JSON
        ticket = outer.reserve_setup(owner, bounds)
        cap = i.accounting.CheckpointChildAdmission(child, None, child_proof, payload)
        init = i.CheckpointChildInitializer(child, None, child_proof, operation=self.operation, gate=lambda stage: None,
            reader=self.reader(), bootstrap_handle=ticket, admission_handle=cap)
        request = {'schema': i.worker.REQUEST, 'branch': 'uninterrupted', 'study_request_sha256': child['request_sha256'],
            'scope_sha256': child['scope_sha256'], 'static_recipe_sha256': child['static_recipe_sha256'],
            'code_sha256': child['bindings']['source']['code_sha256'], 'common_anchor_sha256': child['bindings']['anchors']['common_anchor_sha256'],
            'ingestion_receipt_sha256': receipt['receipt_sha256'], 'lineage_declaration': child['checkpoint_child_declaration'],
            'plan_sha256': i.raw.digest({'qualification_batches': payload['qualification_batches'], 'branch': 'uninterrupted'})}
        child_ledger = init.construct_accounting(child, None, child_proof, receipt, request)
        self.assertEqual(child_ledger.directory, Path(spec['owned_directory']))
        self.assertEqual(child['bindings']['scope'], parent['bindings']['scope'])
        child_ledger.admit_checkpoint_child_prefix(payload, admission_handle=init.predecessor_capability(child, None, child_proof, payload))
        self.assertEqual(init.begin_child_stage(child_ledger, request), 'child/uninterrupted')
        helper = object.__new__(branches.ThreeBranchFixture); helper.serial = 0
        helper.commit(child_ledger); helper.commit(child_ledger)
        marker = self.write(i.child_directory(child, 'uninterrupted') / 'retained.json', {'fixture': True})
        init.finish_child_accounting(child_ledger, request, None,
            retained={'branch': 'uninterrupted', 'raw_refs': [marker], 'native_qualified': False})
        self.assertIsNone(child_ledger.active); self.assertTrue(child_ledger.worker_open)
        child_ledger.end(); self.assertFalse(child_ledger.worker_open)
        self.assertEqual(init.parent_evidence['derived_totals'], i.raw._json(Path(refs['seal']['path']).read_bytes())['cumulative'])


if __name__ == '__main__': unittest.main()
