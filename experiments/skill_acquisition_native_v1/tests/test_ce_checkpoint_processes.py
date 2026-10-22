"""Pure immutable-request, fixed-ledger and prepaid-file tests; no launches."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_checkpoint_processes as p
from . import test_ce_qualification_accounting as qa


class Tests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory(prefix='checkpoint-processes-pure-')
        self.addCleanup(tmp.cleanup); self.root = Path(tmp.name)
        context, proof, receipt = qa.fixture(self.root)
        scope = context['bindings']['scope']; cp_root = self.root / 'checkpoint'
        scope['checkpoint_qualification'] = {'schema': p.cp.SPEC, 'owned_directory': str(cp_root)}
        scope['native_backend']['source_checkout'] = str(self.root / 'checkout')
        module = 'experiments.skill_acquisition_native_v1.ce_checkpoint_worker'; source_name = module.replace('.', '/') + '.py'
        context['bindings']['source']['code_sha256'] = {source_name: 'c' * 64}
        ref = lambda path: {'path': str(path), 'sha256': 'd' * 64, 'bytes': 100}
        template = {'cwd': str(self.root / 'checkout'), 'source_checkout': str(self.root / 'checkout'),
            'worker_module': module, 'source_files': {source_name: ref(self.root / 'checkout' / source_name)},
            'approved_code_sha256': {source_name: 'c' * 64}, 'python_ref': ref(self.root / 'python.exe'),
            'process_image_ref': ref(self.root / 'python.exe'), 'worker_topology': 'direct',
            'environment': {'PATH': 'synthetic fixture'}, 'deadline_seconds': 10, 'handshake_seconds': 2,
            'rpc_seconds': 2, 'terminate_grace_seconds': 1, 'max_message_bytes': 10000,
            'max_messages': 10, 'max_stderr_bytes': 10000,
            'bounds': {'operation_attempts_upper': 100, 'io_read_bytes_upper': 100000,
                'identity_hash_attempts_upper': 100, 'identity_hash_bytes_upper': 100000, 'wall_seconds_upper': 10}}
        template['source_files'][source_name]['sha256'] = 'c' * 64
        caps = {key: 10 ** 12 for key in p.accounting.COST_FIELDS}; caps['wall_seconds_upper'] = 10000
        spec = {'schema': p.SPEC, 'owned_directory': str(cp_root / 'processes'), 'bounds': caps,
            'max_json_bytes': p.raw.MAX_JSON, 'reader_bounds': {key: 10 ** 12 for key in p.raw.BUDGET_FIELDS},
            'launch_templates': {branch: copy.deepcopy(template) for branch in p.cp.BRANCHES}}
        scope['checkpoint_processes'] = spec
        scope['checkpoint_processes_owner'] = {'accounting_spec_sha256': p.raw.digest(spec), 'prepared_receipt_sha256': receipt['receipt_sha256']}
        proof.update(request_sha256=context['request_sha256'], scope_sha256=context['scope_sha256'],
            code_sha256=context['bindings']['source']['code_sha256'],
            os_identity={'pid': 11, 'creation_filetime_100ns': 10, 'boot_identity': 'fixture-boot',
                'image_path': str(self.root / 'python.exe'), 'argv': ['fixture']})
        self.context, self.proof, self.receipt = context, proof, receipt
        self.owner = p.accounting.QualificationAccounting(context, None, proof, receipt)
        marker = self.root / 'marker.json'; marker.write_bytes(b'{"fixture":true}\n')
        self.marker = {'path': str(marker), 'sha256': hashlib.sha256(marker.read_bytes()).hexdigest(), 'bytes': marker.stat().st_size}
        for stage in p.accounting.q.STAGES[1:p.accounting.q.STAGES.index('checkpoint_resume')]:
            self.owner.begin_stage(stage, stage); self.owner.finish_stage([self.marker])
        self.owner.begin_stage('checkpoint_resume', 'parent')
        outer = self.owner.bootstrap_handle._ledger
        ticket = outer.reserve_setup(scope['checkpoint_processes_owner'], caps)
        self.processes = p.CheckpointProcesses(context, None, receipt, proof, backend=None, owner=self.owner,
            handoff=None, parent_handle=None, bootstrap_handle=ticket)
        self.binding = {'stage': 'checkpoint_resume', 'request_sha256': context['request_sha256'],
            'scope_sha256': context['scope_sha256'], 'static_recipe_sha256': context['static_recipe_sha256'],
            'code_sha256': context['bindings']['source']['code_sha256'],
            'worker_identity_sha256': proof['worker_identity_sha256'], 'attempt_id': 'parent'}
    def parent(self):
        return {'binding': self.binding, 'proof': self.proof, 'ledger_refs': {key: self.marker for key in ('header', 'journal', 'seal')},
            'stage_seal': {'schema': 'CE8-CHECKPOINT-PARENT-STAGE-SEAL-1', 'stage': 'checkpoint_resume',
                'C': 2, 'U': 2, 'worker_open': True, 'completed': False,
                'worker_identity_sha256': self.proof['worker_identity_sha256'], 'request_sha256': self.context['request_sha256'],
                'scope_sha256': self.context['scope_sha256'], 'ledger_head': 'a' * 64, 'binding_sha256': 'b' * 64},
            'common_context_ref': self.marker}
    def test_native_constructor_refuses_before_object_paths_models_or_issuers(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('unadmitted traversal')
        before = set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'}
        with patch.object(Path, 'open', side_effect=AssertionError('unadmitted file')):
            with self.assertRaises(ValueError):
                p.CheckpointProcesses(Trap(), None, Trap(), Trap(), backend=Trap(), owner=Trap(), handoff=Trap(),
                    parent_handle=Trap(), bootstrap_handle=Trap(), launch_issuer=Trap(), launch_verifier=Trap())
        self.assertEqual(before, set(sys.modules) & {'torch', 'numpy', 'transformers', 'safetensors'})
        self.assertFalse(p.launcher._REGISTERED_LAUNCH_VERIFIERS)
    def test_immutable_requests_exclude_future_identity_and_keep_scope_unmodified(self):
        frozen = copy.deepcopy(self.context['bindings']['scope'])
        parent = self.parent()
        for branch in ('uninterrupted', 'prefix'):
            request = p.child_request(self.context, parent, branch)
            self.assertFalse({'worker_identity_sha256', 'child_request_ref', 'child_request_sha256'} & set(request))
            self.assertFalse({'worker_identity_sha256', 'child_request_ref', 'child_request_sha256'} & set(request['lineage_declaration']))
            reference = self.processes.write_json(p.initializer.child_directory(self.context, branch) / 'request.json', request)
            self.assertEqual(reference['sha256'], hashlib.sha256(p.raw.canonical(request) + b'\n').hexdigest())
            self.assertEqual(self.context['bindings']['scope'], frozen)
            self.assertEqual(request['lease_delegation_path'], str(p.initializer.child_directory(self.context, branch) / 'LEASE_DELEGATION.json'))
    def test_resumed_request_uses_exact_original_prefix_manifest_and_history(self):
        batches = p.cp.qualification_batches(self.context)
        manifest = {'progress': {'C': 1, 'U': 1, 'history': [{'boundary': 0, 'row_ids': batches[0], 'updated': True}]},
            'bindings': {'qualification_batches': batches}, 'bindings_sha256': 'f' * 64}
        prefix = {'branch': 'prefix', 'manifest': manifest, 'manifest_ref': self.marker,
            'store_refs': {key: self.marker for key in ('bindings', 'journal', 'manifest', 'payload')}}
        request = p.child_request(self.context, self.parent(), 'resumed', prefix=prefix)
        self.assertEqual(request['lineage_declaration']['prefix_manifest'], manifest)
        self.assertEqual(request['lineage_declaration']['inherited_history'], manifest['progress']['history'])
        self.assertEqual(request['initializer_inputs']['prefix_store'], prefix['store_refs'])
        wrong = copy.deepcopy(prefix); wrong['manifest']['progress']['U'] = 2
        with self.assertRaisesRegex(ValueError, 'prefix U1'): p.child_request(self.context, self.parent(), 'resumed', prefix=wrong)
    def test_launch_spec_binds_raw_request_parent_os_and_full_source_without_pid_defaults(self):
        spec = p.child_launch_spec(self.context, 'prefix', self.marker, self.proof)
        self.assertEqual(spec['request_ref'], self.marker)
        self.assertEqual(spec['argv'][-1], self.marker['path'])
        self.assertEqual(spec['parent_identity']['creation_filetime_100ns'], 10)
        self.assertEqual(spec['approved_code_sha256'], self.context['bindings']['source']['code_sha256'])
        bad = copy.deepcopy(self.context)
        bad['bindings']['scope']['checkpoint_processes']['launch_templates']['prefix']['request_ref'] = self.marker
        with self.assertRaisesRegex(ValueError, 'excludes derived'): p.child_launch_spec(bad, 'prefix', self.marker, self.proof)
    def test_frozen_parent_journal_prefix_survives_later_main_ledger_append_and_replays(self):
        refs, snapshot = self.processes.capture_ledger(self.owner, 'PARENT_PARTIAL')
        saved = Path(refs['journal']['path']).read_bytes()
        self.owner.operation('resource_probe', 'started', 'later'); self.owner.operation('resource_probe', 'returned', 'later')
        self.assertNotEqual(self.owner._head(), snapshot['ledger_head'])
        self.assertEqual(Path(refs['journal']['path']).read_bytes(), saved)
        context = dict(self.context, prepared_receipt=self.receipt)
        header = p.execution._header(context, p.execution._read_json(self.processes.reader, refs['header'], self.binding, 'qualification_header'))
        state, _, _ = p.execution._replay(self.processes.reader, refs['journal'], self.binding, context, header, {self.proof['worker_identity_sha256']})
        p.execution._seal(snapshot, state)
        self.assertEqual(state._head(), snapshot['ledger_head'])
        self.assertNotEqual(refs['journal']['path'], str(self.owner.path))
    def test_create_only_file_publication_and_partial_failure_keep_permanent_outer_reservations(self):
        target = self.processes.directory / 'record.json'
        before = dict(self.processes.counts)
        reference = self.processes.write_json(target, {'native_qualified': False})
        blob = target.read_bytes(); self.assertEqual(reference['sha256'], hashlib.sha256(blob).hexdigest())
        with self.assertRaisesRegex(ValueError, 'create-only'): self.processes.write_json(target, {'replace': True})
        self.assertEqual(target.read_bytes(), blob)
        self.assertGreater(self.processes.counts['io_write_bytes_upper'], before['io_write_bytes_upper'])
        self.assertTrue(any(op['phase'] == 'failed' for op in self.processes.operations.values()))
        self.assertEqual(self.processes.handle.evidence['status'], 'STARTED')
        self.assertIs(self.processes.operation, self.processes.operation)
    def test_metadata_fixture_cannot_call_native_launcher_or_save_model(self):
        with patch.object(p.launcher, 'launch_native', side_effect=AssertionError('process creation')):
            with self.assertRaisesRegex(ValueError, 'cannot execute'):
                self.processes.launch(self.context, None, {'branch': 'uninterrupted'}, self.parent(), [])
            with self.assertRaisesRegex(ValueError, 'cannot execute'):
                self.processes.save_parent(self.context, None, None, self.owner, self.binding)

    def observation_fixture(self):
        from . import test_ce_checkpoint_reduction as branches
        from . import test_ce_checkpoint_qualification as checkpoint_files
        # Durable semantic CE/CP records represent explicit simulated work;
        # bytes below are actual tiny files, never native model observations.
        helper = object.__new__(branches.ThreeBranchFixture); helper.serial = 0
        helper.commit(self.owner); helper.commit(self.owner)
        files = checkpoint_files.Contracts(); files.setUp(); self.addCleanup(files.doCleanups)
        files.left = self.binding; files.right = dict(self.binding, worker_identity_sha256='1' * 64)
        left, right = files.state(files.left), files.state(files.right)
        comparison = p.cp.compare_checkpoint_states(files.reader(), left, files.left, right, files.right,
            parameter_inventory=files.inventory)
        manifest = p.raw._json(Path(left['path']).read_bytes())
        self.owner.operation('checkpoint_save', 'started', 'actual-metadata-save')
        self.owner.operation('checkpoint_save', 'returned', 'actual-metadata-save', manifest_sha256=manifest['manifest_sha256'],
            C=2, U=2, generation=self.owner.generation, symbolic_snapshot=self.owner.symbolic_snapshot)
        before = self.owner.snapshot()
        progress = copy.deepcopy(self.owner.progress_record()); progress.update(generation=progress['generation'] + 1,
            snapshot='9' * 64, symbolic_snapshot='9' * 64)
        restored = dict(progress, progress=copy.deepcopy(progress), manifest_sha256=manifest['manifest_sha256'],
            install_fence_required=True, schema=p.cp.RESTORE_SCHEMA, source_worker_binding=self.binding)
        self.owner.operation('checkpoint_restore', 'started', 'actual-metadata-restore')
        self.owner.operation('checkpoint_restore', 'returned', 'actual-metadata-restore',
            **{key: restored[key] for key in ('C', 'U', 'generation', 'symbolic_snapshot', 'manifest_sha256')})
        self.owner.install_restore(restored)
        branches_reduced, terminals = {}, {}
        for index, branch in enumerate(p.cp.BRANCHES):
            identity = str(index + 1) * 64
            branches_reduced[branch] = {'binding': dict(self.binding, worker_identity_sha256=identity),
                'checkpoint_save_operations': 1, 'failed_restore_operations': 1,
                'returned_restore_operations': 2 if branch == 'resumed' else 1,
                'negative_controls_independently_recomputed': True,
                'derived_costs': {key: 10 for key in p.accounting.COST_FIELDS}}
            terminals[branch] = {'schema': 'CE8-WINDOWS-PROCESS-TERMINAL-1', 'status_file_used': False,
                'exit_code': 0, 'exit_filetime_100ns': 200, 'worker_identity_sha256': identity,
                'request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256'],
                'os_identity': {'pid': 20 + index, 'creation_filetime_100ns': 100}}
        reduced = {'branches': branches_reduced, 'parent_costs_imported': False,
            'U2_state_comparison': comparison, 'parent_state_comparison': copy.deepcopy(comparison)}
        return before, reduced, terminals, restored

    def test_exact_external_interface_uses_actual_parent_records_and_raw_byte_comparison(self):
        before, reduced, terminals, restored = self.observation_fixture()
        observed = p.checkpoint_observations(self.owner, before, reduced, terminals, restored, self.binding)
        result = p.external_result(self.marker, observed, reduced, self.processes.counts)
        self.assertEqual(result['raw_artifacts'], [self.marker]); self.assertIsInstance(result['observations'], dict)
        self.assertEqual(observed['checkpoint_saves'], 4); self.assertEqual(observed['checkpoint_restores'], 5)
        self.assertEqual(observed['checkpoint_restore_attempts'], 8); self.assertEqual(observed['actual_processes'], 4)
        p.accounting.q.validate_stage_observations(self.context, 'checkpoint_resume', observed)
        self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified'])

    def test_foreign_terminal_current_fence_or_raw_inequality_cannot_be_stage_observations(self):
        before, reduced, terminals, restored = self.observation_fixture()
        for change in (lambda d, t, r: t['prefix'].update(worker_identity_sha256='0' * 64),
                       lambda d, t, r: r.update(generation=0),
                       lambda d, t, r: d['U2_state_comparison'].update(adam_bitwise=False),
                       lambda d, t, r: d['parent_state_comparison'].update(torch_cuda_rng_bitwise=False),
                       lambda d, t, r: d['branches']['resumed'].update(checkpoint_save_operations=0)):
            data, proofs, fence = copy.deepcopy(reduced), copy.deepcopy(terminals), copy.deepcopy(restored); change(data, proofs, fence)
            with self.assertRaises(ValueError): p.checkpoint_observations(self.owner, before, data, proofs, fence, self.binding)

    def test_three_owned_terminal_clones_survive_original_close_and_release_only_after_final_replay(self):
        from . import test_ce_worker_evidence as handles
        terminal = handles.WorkerContractTests()
        records, closed = {}, []
        for branch in p.cp.BRANCHES:
            original, releases = terminal.terminal_fixture()
            clone = original.retain_terminal(); proof = clone.terminal_evidence(); original.close()
            self.assertEqual(releases, [111]); self.assertEqual(clone.terminal_evidence(), proof)
            records[branch] = {'handle': clone, 'binding': self.binding, 'terminal': proof}
            closed.append(releases)
        self.processes.terminal_handle_records = records
        self.assertEqual(self.processes.handle.evidence['status'], 'STARTED')
        self.processes.close_terminal_handles()
        self.assertEqual(self.processes.terminal_handle_records, {})
        self.assertTrue(self.processes.final_handle_lifetime_closed)
        self.assertTrue(all(values == [111, 222] for values in closed))
        self.assertEqual(self.processes.handle.evidence['status'], 'RETURNED')
        self.processes.close_terminal_handles()  # owned cleanup is idempotent

    def test_terminal_handle_cleanup_preserves_primary_and_closes_other_owned_handles(self):
        from . import test_ce_worker_evidence as handles
        terminal = handles.WorkerContractTests()
        first, closed_first = terminal.terminal_fixture(); second, closed_second = terminal.terminal_fixture()
        self.processes.terminal_handle_records = {'uninterrupted': {'handle': first}, 'prefix': {'handle': second}}
        original = ValueError('raw final evidence failed')
        from experiments.skill_acquisition_native_v1 import ce_worker_evidence as evidence
        implementation = evidence.WinWorkerHandle.close
        def failing_close(current):
            implementation(current)
            if current is first: raise RuntimeError('cleanup receipt failed')
        with patch.object(evidence.WinWorkerHandle, 'close', failing_close):
            with self.assertRaises(ValueError) as caught: self.processes.close_terminal_handles(original)
        self.assertIs(caught.exception, original)
        self.assertEqual(closed_first, [111]); self.assertEqual(closed_second, [111])
        self.assertEqual(self.processes.terminal_handle_records, {})
        self.assertEqual(self.processes.handle.evidence['status'], 'FAILED')
        self.assertEqual(original.checkpoint_terminal_handle_cleanup[0]['message'], 'cleanup receipt failed')

    def test_raw_terminal_dict_cannot_replace_a_retained_handle_for_cleanup_or_replay(self):
        self.processes.terminal_handle_records = {'prefix': {'handle': {'terminal': True}}}
        with self.assertRaisesRegex(ValueError, 'actual retained terminal'): self.processes.close_terminal_handles()
        self.assertEqual(set(self.processes.terminal_handle_records), {'prefix'})
        with self.assertRaisesRegex(ValueError, 'cannot execute'): self.processes.retained_terminal_handles()


if __name__ == '__main__': unittest.main()
