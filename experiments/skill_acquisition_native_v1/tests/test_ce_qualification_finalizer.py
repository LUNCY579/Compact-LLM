"""Real raw-cost files and closed native entry controls; no OS/model work."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_qualification_finalizer as f
from . import test_ce_qualification_accounting as qa, test_ce_bootstrap_accounting as ba
from .test_ce_native_services import Trap
from . import test_ce_checkpoint_reduction as branches
from . import test_ce_service_bootstrap_verifier as service_fixture
from . import test_ce_finalizer_accounting as cost_fixture
from experiments.skill_acquisition_native_v1 import ce_finalizer_accounting as costs


class Tests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(prefix='qualification-finalizer-pure-'); self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)

    def scope_fixture(self):
        context, proof, receipt = qa.fixture(self.root)
        context['prepared_receipt'] = receipt
        context['verification_binding'] = {'stage': 'qualification_finalizer', 'request_sha256': context['request_sha256'],
            'scope_sha256': context['scope_sha256'], 'static_recipe_sha256': context['static_recipe_sha256'],
            'code_sha256': context['bindings']['source']['code_sha256'], 'worker_identity_sha256': proof['worker_identity_sha256'],
            'attempt_id': 'pure-finalizer'}
        scope = context['bindings']['scope']; costs = {key: 1000 for key in f.q.COST_FIELDS}
        scope['checkpoint_processes'] = {'bounds': dict(costs)}
        scope['checkpoint_processes_owner'] = {'accounting_spec_sha256': '1' * 64, 'prepared_receipt_sha256': receipt['receipt_sha256']}
        scope['publication'] = {'bounds': dict(costs)}
        clock = ba.Clock(); ledger = f.boot.BootstrapAccounting(context, None, proof, clock=clock)
        contracts = f._expected_bootstrap_owners(context)
        for contract in contracts:
            handle = ledger.reserve_setup(contract['owner'], contract['bounds']); handle.begin(); handle.complete()
        clock.advance(2); ledger.end(); capsule = ledger.inspect_evidence()
        reader = f.raw.RawEvidenceReader(lambda *args, **fields: None, {key: 10 ** 12 for key in f.raw.BUDGET_FIELDS}, self.root)
        return context, proof, ledger, capsule, reader

    def test_actual_scope_files_replay_and_capsule_refund_is_refused(self):
        context, proof, ledger, capsule, reader = self.scope_fixture()
        result = f.replay_scope_bootstrap(context, capsule, proof, reader=reader, preflight=lambda *args: None)
        self.assertEqual(result['derived_totals'], ledger.cumulative)
        self.assertTrue(result['clock_coverage_complete']); self.assertFalse(result['native_qualified'])
        wrong = copy.deepcopy(capsule); wrong['snapshot']['cumulative']['io_read_bytes_upper'] -= 1
        with self.assertRaisesRegex(ValueError, 'capsule refund/clock/owner'):
            f.replay_scope_bootstrap(context, wrong, proof, reader=reader, preflight=lambda *args: None)

    def test_scope_external_preflight_denies_before_any_file_or_hash(self):
        context, proof, ledger, capsule, reader = self.scope_fixture()
        def deny(*args): raise RuntimeError('paid finalizer scope denied')
        with patch.object(Path, 'open', side_effect=AssertionError('unpaid read')):
            with self.assertRaisesRegex(RuntimeError, 'scope denied'):
                f.replay_scope_bootstrap(context, capsule, proof, reader=reader, preflight=deny)

    def test_native_entry_and_counterfeit_opaque_result_deny_before_io(self):
        with patch.object(Path, 'open', side_effect=AssertionError('native IO before authority')):
            with self.assertRaises(ValueError): f.verify_supervision(Trap(), Trap(), Trap())
            with self.assertRaises(ValueError): f.consume_verified({'native_qualified': True}, Trap())
            with self.assertRaises(ValueError): f.consume_verified(object.__new__(f.VerifiedQualificationEvidence), Trap())
            with self.assertRaises(ValueError): f.VerifiedQualificationEvidence(native_qualified=True)

    def test_terminal_suffix_requires_actual_finite_order_and_original_deadline(self):
        result = f.clock_suffix(103, 100, 105, 4, 10)
        self.assertEqual(result['suffix_seconds_upper_observed'], 2)
        for args in ((103, 100, 102, 4, 10), (103, 100, 108, 4, 10),
                     (103, 100, 105, 4, 4), (103, 100, True, 4, 10),
                     (103, 100, float('inf'), 4, 10)):
            with self.assertRaises(ValueError): f.clock_suffix(*args)

    def test_known_gap_requires_explicit_evidence_and_unknown_gap_refuses(self):
        gap = next(iter(f.EXECUTION_GAPS)); key = f.EXECUTION_GAPS[gap]
        observed = {'raw_ref': {'path': str(self.root / 'actual.json'), 'sha256': 'a' * 64, 'bytes': 10}}
        self.assertEqual(f.discharge_known_gaps([gap], f.EXECUTION_GAPS, {key: observed}), {gap: observed})
        for gaps, evidence in (([gap], {}), ([gap], {key: True}), (['new unsupported proof gap'], {key: observed})):
            with self.assertRaises(ValueError): f.discharge_known_gaps(gaps, f.EXECUTION_GAPS, evidence)

    def terminal_fixture(self):
        parent = {'request_sha256': 'a' * 64, 'scope_sha256': 'b' * 64}
        branches, terminals, refs = {}, {}, {}
        for index, branch in enumerate(f.checkpoint.BRANCHES):
            binding = dict(parent, worker_identity_sha256=str(index + 1) * 64)
            os_identity = {'pid': 10 + index, 'creation_filetime_100ns': 100, 'boot_identity': 'fixture'}
            proof = dict(parent, os_identity=os_identity)
            costs = {'io_read_bytes_upper': 20 + index}
            branches[branch] = {'binding': binding, 'lineage': {'child_proof': proof}, 'derived_costs': costs}
            ref = {'path': str(self.root / (branch + '.json')), 'sha256': 'c' * 64, 'bytes': 10}; refs[branch] = ref
            terminal = dict(binding, schema=f.worker.TERMINAL, status_file_used=False, exit_code=0, os_identity=os_identity)
            terminals[branch] = {'binding': binding, 'terminal_ref': ref, 'child_derived_reservations': costs,
                'terminal_proof': terminal, 'terminal_observed_monotonic': 102.}
        independent = {'checkpoint_resume': {'raw_branches': {'branches': branches}}}
        reference = {'path': str(self.root / 'bundle.json'), 'sha256': 'd' * 64, 'bytes': 10}
        retained = {'schema': 'CE8-PARENT-CHILD-TERMINAL-REPLAY-1', 'raw_numeric_reductions': copy.deepcopy(independent),
                    'checkpoint_bundle_ref': reference, 'child_terminal_results': terminals}
        return retained, independent, {'reference': reference, 'payload': {'child_terminal_refs': refs}}, parent

    def test_child_replay_flags_cannot_replace_exact_refs_costs_os_and_time(self):
        retained, independent, bundle, parent = self.terminal_fixture()
        self.assertEqual(set(f.compare_child_replay(retained, independent, bundle, parent_binding=parent)), set(f.checkpoint.BRANCHES))
        branch = f.checkpoint.BRANCHES[0]
        mutations = (lambda record: record['terminal_ref'].update(sha256='f' * 64),
                     lambda record: record['child_derived_reservations'].update(io_read_bytes_upper=0),
                     lambda record: record['terminal_proof'].update(exit_code=False),
                     lambda record: record['terminal_proof']['os_identity'].update(pid=999),
                     lambda record: record.pop('terminal_observed_monotonic'))
        for mutation in mutations:
            wrong = copy.deepcopy(retained); wrong['child_os_terminal_verified'] = True
            mutation(wrong['child_terminal_results'][branch])
            with self.assertRaises(ValueError): f.compare_child_replay(wrong, independent, bundle, parent_binding=parent)

    def test_three_child_ended_raw_costs_replay_and_refund_or_late_terminal_refuse(self):
        tree = branches.ThreeBranchFixture(self.root / 'tree')
        owner = costs.fixture_owner(cost_fixture.spec(self.root), self.root)
        bundle = {'branch_refs': tree.branch_refs, 'branch_bindings': tree.branch_bindings, 'child_final_cost_clock_refs': {}}
        stamps = {}; footers = {}
        for branch in f.checkpoint.BRANCHES:
            raw_branch = tree.branches[branch]; child = raw_branch['payload']['child_context']
            proof = raw_branch['payload']['observed_lineage']['child_proof']; clock = ba.Clock()
            scope = child['bindings']['scope']
            scope['qualifier_bootstrap']['owned_directory'] = str(raw_branch['directory'] / 'closed-bootstrap')
            scope['checkpoint_branch_publication'] = {'schema': 'CE8-CHECKPOINT-BRANCH-PUBLICATION-SPEC-1',
                'bounds': {key: 1000 for key in f.q.COST_FIELDS}}
            outer = f.boot.BootstrapAccounting(child, None, proof, clock=clock)
            publication = None
            for contract in f._expected_bootstrap_owners(child, child=True):
                handle = outer.reserve_setup(contract['owner'], contract['bounds']); handle.begin(); handle.complete()
                publication = handle
            clock.advance(2); outer.end(); outer_evidence = outer.inspect_evidence()
            service_spec = service_fixture.specification(raw_branch['directory'] / 'closed-service')
            service_clock = service_fixture.Clock()
            service = f.services.BootstrapLedger(service_spec, child['request_sha256'], synthetic_fixture=True, clock=service_clock)
            service.operation('fixture', 'started', 'paid-inner'); service_clock.advance(1)
            service.operation('fixture', 'returned', 'paid-inner', wall_s=1); service_clock.advance(1)
            service.end(); service_evidence = service.inspect_evidence()
            specification = {'bootstrap': service_spec}
            config_ref = tree.write(raw_branch['directory'] / 'configuration.json', {'service_spec': specification})
            raw_branch['payload']['outer_publication_prepayment'] = publication.evidence
            tree.refresh_result(branch)
            footer = {'schema': 'CE8-CHECKPOINT-CHILD-FINAL-COST-CLOCK-1', 'branch': branch,
                'request_sha256': child['request_sha256'], 'scope_sha256': child['scope_sha256'], 'worker_proof': proof,
                'branch_ref': tree.branch_refs[branch], 'scope_bootstrap_evidence': outer_evidence,
                'service_bootstrap_evidence': service_evidence, 'service_specification': specification, 'configuration_ref': config_ref}
            path = raw_branch['directory'] / 'CHILD_FINAL_COST_CLOCK.json'
            bundle['child_final_cost_clock_refs'][branch] = tree.write(path, footer)
            footers[branch] = footer; stamps[branch] = {'terminal_observed_monotonic': 103.}
        result, gaps = f._child_costs(tree.context, bundle, stamps, tree.parent_binding, owner)
        self.assertEqual(gaps, []); self.assertEqual(set(result), set(f.checkpoint.BRANCHES))
        for value in result.values():
            self.assertTrue(value['scope_bootstrap']['clock_coverage_complete'])
            self.assertTrue(value['service_bootstrap']['recorded_clock_ended'])
            self.assertEqual(value['service_terminal_suffix']['suffix_seconds_upper_observed'], 1)
            self.assertFalse(value['service_bootstrap']['native_qualified'])
        branch = f.checkpoint.BRANCHES[0]; footer = footers[branch]
        footer['scope_bootstrap_evidence']['snapshot']['cumulative']['io_read_bytes_upper'] -= 1
        path = Path(bundle['child_final_cost_clock_refs'][branch]['path'])
        bundle['child_final_cost_clock_refs'][branch] = tree.write(path, footer)
        with self.assertRaisesRegex(ValueError, 'capsule refund/clock/owner'):
            f._child_costs(tree.context, bundle, stamps, tree.parent_binding, owner)
        footer['scope_bootstrap_evidence']['snapshot']['cumulative']['io_read_bytes_upper'] += 1
        bundle['child_final_cost_clock_refs'][branch] = tree.write(path, footer)
        stamps[branch]['terminal_observed_monotonic'] = 1000.
        with self.assertRaisesRegex(ValueError, 'original clock reservation/deadline'):
            f._child_costs(tree.context, bundle, stamps, tree.parent_binding, owner)


if __name__ == '__main__': unittest.main()
