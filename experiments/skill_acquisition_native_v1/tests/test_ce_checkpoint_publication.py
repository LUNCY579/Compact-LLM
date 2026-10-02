"""Actual small publication files and durable pure Q metadata; no models."""
import copy
import hashlib
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_checkpoint_publication as p
from experiments.skill_acquisition_native_v1 import ce_qualification_accounting as a
from .test_ce_qualification_accounting import fixture
from . import test_ce_journal as cpu


class Tests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.context, self.proof, self.receipt = fixture(self.root)
        scope = self.context['bindings']['scope']
        scope['checkpoint_qualification'] = {'owned_directory': str(self.root / 'checkpoints')}
        bounds = {key: 10 ** 11 for key in p.publication.q.COST_FIELDS}; bounds['wall_seconds_upper'] = 1000
        scope['checkpoint_branch_publication'] = {'schema': p.SPEC, 'max_receipt_bytes': p.raw.MAX_JSON,
            'bounds': bounds, 'reader_bounds': {key: 10 ** 12 for key in p.raw.BUDGET_FIELDS}}
        self.context['checkpoint_child_declaration'] = {'branch': 'prefix'}
        self.context['checkpoint_child_observed_lineage'] = {'schema': a.CHILD_PREFIX, 'branch': 'prefix'}
        self.ledger = a.QualificationAccounting(self.context, None, self.proof, self.receipt)
        dummy = self.root / 'prior.json'; dummy.write_bytes(b'{"fixture":true}\n')
        reference = {'path': str(dummy), 'sha256': hashlib.sha256(dummy.read_bytes()).hexdigest(), 'bytes': dummy.stat().st_size}
        for stage in a.q.STAGES[1:a.q.STAGES.index('checkpoint_resume')]:
            self.ledger.begin_stage(stage, 'prior/' + stage); self.ledger.finish_stage([reference])
        self.ledger.begin_stage('checkpoint_resume', 'child/prefix')
        helper = cpu.Controls(); helper.bindings = {'recipe': self.ledger.recipe}
        proxy = SimpleNamespace(progress=self.ledger._ce.progress, generation=self.ledger.generation,
            symbolic_snapshot=self.ledger.symbolic_snapshot, static_recipe_sha256=self.context['static_recipe_sha256'],
            plan_rows=self.ledger.adam_batches, census=self.ledger._ce.census, operation=self.ledger.operation)
        self.ledger.commit_boundary(helper.successful_result(proxy))
        self.binding = {'stage': 'checkpoint_resume', 'worker_identity_sha256': self.proof['worker_identity_sha256'],
            'request_sha256': self.context['request_sha256'], 'scope_sha256': self.context['scope_sha256'],
            'static_recipe_sha256': self.context['static_recipe_sha256'],
            'code_sha256': self.context['bindings']['source']['code_sha256'], 'attempt_id': 'child/prefix'}
        path = self.root / 'saved.safetensors'
        self.manifest = {'schema': 'SKILL-ACQUISITION-NATIVE-CHECKPOINT-1',
            'bindings': {'qualification_evidence_binding': self.binding}, 'payload': {'path': str(path)},
            'progress': self.ledger.progress_record()}
        path.with_suffix('.json').write_bytes(p.canonical(self.manifest) + b'\n')
        self.publisher = p.CheckpointBranchPublisher(self.context, None, self.proof, self.receipt)
        self.request = {'branch': 'prefix'}

    def link(self, source, target, root):
        self.assertEqual(source.parent.resolve(), root.resolve())
        self.assertEqual(target.parent.resolve(), root.resolve())
        os.link(source, target)
        return {'method': 'pure_fixture_exclusive_link'}

    def retain(self):
        with patch.object(p.publication, '_atomic_no_replace', self.link):
            return self.publisher.retain_branch(self.context, None, self.request, self.binding,
                self.manifest, self.ledger, None, negative_control_evidence={'fixture': True})

    def finish(self, retained):
        self.ledger.finish_stage(retained['raw_refs']); self.ledger.end()

    def publish(self, retained):
        with patch.object(p.publication, '_atomic_no_replace', self.link):
            return self.publisher.publish_after_release(self.context, None, self.request, self.binding,
                retained, self.ledger, result_schema=p.RESULT, native_qualified=False)

    def test_native_constructor_refuses_before_input_or_file_access(self):
        class Trap:
            def __getattribute__(self, key): raise AssertionError('before native authority')
        with patch.object(Path, 'open', side_effect=AssertionError('unapproved IO')):
            with self.assertRaisesRegex(ValueError, 'issuer is unregistered'):
                p.CheckpointBranchPublisher(Trap(), Trap(), Trap(), Trap())

    def test_two_phase_publication_retains_ended_journal_and_never_claims_os_terminal(self):
        retained = self.retain(); self.finish(retained)
        original_journal = self.ledger.path.read_bytes()
        result = self.publish(retained)
        self.assertEqual(self.ledger.path.read_bytes(), original_journal)
        self.assertTrue(result['body_completed']); self.assertFalse(result['os_terminal_verified'])
        self.assertFalse(result['native_qualified']); self.assertFalse(result['execution_verified'])
        reference = result['raw_artifact']; content = Path(reference['path']).read_bytes()
        self.assertEqual(hashlib.sha256(content).hexdigest(), reference['sha256'])
        body = p.raw._json(content)['payload']
        snapshot = p.raw._json(Path(body['ledger_refs']['seal']['path']).read_bytes())
        self.assertEqual(snapshot['cumulative'], self.ledger.cumulative)
        self.assertFalse(snapshot['worker_open'])
        self.assertEqual(body['ledger_refs']['journal']['sha256'], hashlib.sha256(original_journal).hexdigest())
        self.assertEqual(result['publication_prepayment']['status'], 'RETURNED')
        with self.assertRaisesRegex(ValueError, 'live child publication'):
            self.publish(retained)

    def test_open_clock_or_unfinished_stage_cannot_publish(self):
        retained = self.retain()
        with self.assertRaisesRegex(ValueError, 'cleanup must end'):
            self.publish(retained)
        self.assertEqual(self.publisher.handle.evidence['status'], 'FAILED')
        self.assertFalse((self.publisher.directory / 'CHECKPOINT_BRANCH.json').exists())

    def test_other_ended_ledger_is_refused_before_snapshot_or_path_reads(self):
        retained = self.retain(); self.finish(retained)
        class Trap:
            def __getattribute__(self, key): raise AssertionError('foreign accounting owner was read')
        with self.assertRaisesRegex(ValueError, 'original child accounting owner'):
            self.publisher.publish_after_release(self.context, None, self.request, self.binding,
                retained, Trap(), result_schema=p.RESULT, native_qualified=False)

    def test_returned_manifest_tamper_is_retained_failure(self):
        path = Path(self.manifest['payload']['path']).with_suffix('.json')
        changed = copy.deepcopy(self.manifest); changed['progress']['U'] = 0
        path.write_bytes(p.canonical(changed) + b'\n')
        with self.assertRaisesRegex(ValueError, 'manifest bytes differ'):
            self.retain()
        self.assertEqual(self.publisher.handle.evidence['status'], 'FAILED')
        self.assertEqual(p.raw._json(path.read_bytes())['progress']['U'], 0)

    def test_publication_derivation_cannot_change_common_prices_or_directory(self):
        common = copy.deepcopy(self.context['bindings']['scope'])
        bad = p.publication_specification(self.context); bad['bounds']['io_read_bytes_upper'] -= 1
        self.context['checkpoint_child_publication'] = bad
        with self.assertRaisesRegex(ValueError, 'only its branch directory'):
            p.publication_specification(self.context)
        self.assertEqual(self.context['bindings']['scope'], common)

    def final_entry(self):
        from experiments.skill_acquisition_native_v1 import ce_bootstrap_accounting as boot, ce_native_services as services
        from .test_ce_native_services import bootstrap_spec
        serial = getattr(self, '_footer_case', 0) + 1; self._footer_case = serial
        self.context['bindings']['scope']['qualifier_bootstrap']['owned_directory'] = str(self.root / ('final-bootstrap-' + str(serial)))
        outer = boot.BootstrapAccounting(self.context, None, self.proof)
        owner = {'accounting_spec_sha256': 'a' * 64, 'prepared_receipt_sha256': self.receipt['receipt_sha256']}
        ticket = outer.reserve_setup(owner, dict.fromkeys(boot.COST_FIELDS, 1000)); ticket.begin(); ticket.complete()
        source_spec = bootstrap_spec(self.root / ('service-final-' + str(serial)))
        source_spec['caps'].update(read_bytes=10**9, write_bytes=10**9, hash_attempts=100000,
            hash_bytes=10**9, metadata_bytes=10**9, operation_attempts=10000, wall_seconds=10000)
        source_spec['finalization_bounds'] = {'read_bytes_upper': 128 * 1024 ** 2, 'write_bytes_upper': 16 * 1024 ** 2,
            'identity_hash_attempts_upper': 32, 'identity_hash_bytes_upper': 256 * 1024 ** 2, 'wall_s_upper': 120}
        service = services.BootstrapLedger(source_spec, self.context['request_sha256'], synthetic_fixture=True)
        runtime = object.__new__(services.NativeControllerServices)
        runtime.bootstrap, runtime.spec = service, {'bootstrap': source_spec}
        runtime.constructor_prepayment = SimpleNamespace(evidence={'native_qualified': False, 'reservation_ref': {'fixture': True}})
        payment = SimpleNamespace(status='RETURNED', evidence={'status': 'RETURNED', 'native': False})
        path = self.root / 'checkpoints' / 'children' / 'prefix'; path.mkdir(parents=True, exist_ok=True)
        return {'source_service_runtime': runtime, 'child_scope_bootstrap': outer, 'service_spec': runtime.spec,
            'entry_prepayment': payment, 'worker_handle': SimpleNamespace(closed=True), 'parent_handle': SimpleNamespace(closed=True),
            'child_final_context': self.context, 'child_final_proof': self.proof, 'child_final_request': {'branch': 'prefix'},
            'child_result': {'raw_artifact': {'path': str(path / 'branch.json'), 'sha256': 'b' * 64, 'bytes': 100}}}

    def test_actual_child_footer_ends_both_original_clocks_after_cleanup_and_retains_refs(self):
        from experiments.skill_acquisition_native_v1 import ce_checkpoint_configuration as configuration
        entry = self.final_entry()
        with patch.object(configuration, '_EVIDENCE', {'configuration_ref': {'path': str(self.root / 'config.json'), 'sha256': 'c' * 64, 'bytes': 100}}), \
                patch.object(p.publication, '_atomic_no_replace', self.link):
            ref = p.retain_child_final_cost_clock(entry)
        body = p.raw._json(Path(ref['path']).read_bytes())
        self.assertEqual(body['schema'], p.FINAL_COST_CLOCK)
        self.assertTrue(body['scope_bootstrap_evidence']['snapshot']['clock_recorded_end'])
        self.assertTrue(body['service_bootstrap_evidence']['snapshot']['clock_ended'])
        self.assertEqual(body['branch_ref'], entry['child_result']['raw_artifact'])
        self.assertFalse(body['os_terminal_verified']); self.assertFalse(body['native_qualified'])
        self.assertEqual(body['child_context_sha256'], p.digest(p.authority.context_preimage(self.context)))
        self.assertLessEqual(ref['bytes'], p.MAX_FINAL_COST_CLOCK)

    def test_unclosed_handle_or_short_finalization_budget_denies_before_clock_end(self):
        for condition in ('handle', 'budget'):
            entry = self.final_entry()
            if condition == 'handle': entry['worker_handle'].closed = False
            else: entry['source_service_runtime'].bootstrap.spec['finalization_bounds']['write_bytes_upper'] = 1
            with self.subTest(condition=condition), self.assertRaises(ValueError): p.retain_child_final_cost_clock(entry)
            self.assertFalse(entry['child_scope_bootstrap'].closed)
            self.assertFalse(entry['source_service_runtime'].bootstrap.sealed)


if __name__ == '__main__': unittest.main()
