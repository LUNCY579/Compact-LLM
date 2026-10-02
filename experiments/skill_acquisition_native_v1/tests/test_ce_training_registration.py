"""Actual canonical TRAIN metadata draft fixtures; no models or source grants."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import ce_training_registration as registration
from . import test_ce_registration as qualification_tests


class Trap:
    def __getattribute__(self, name): raise AssertionError('native work from draft')


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        basis = qualification_tests.RegistrationTests
        basis.setUpClass()
        base = basis(methodName='test_real_recipe_witness_contract_and_unapproved_backend_scope')
        cls.qualification = base.documents()
        cls.plan = json.loads(Path(basis.plan['path']).read_bytes())
        cls.preparation = json.loads(Path(basis.proposal['input_refs']['preparation']['path']).read_bytes())
        cls.source = basis.source
        cls.refs = {'decision': basis.decision_ref, 'preparation': basis.proposal['input_refs']['preparation'], 'plan': basis.plan}

    def budget(self):
        return registration.propose_budget(self.qualification, self.plan,
            output_root=Path(tempfile.gettempdir()) / 'ce8-production-packet-fixture')

    def documents(self, budget=None, source=None):
        return registration.build_documents(self.qualification, self.plan, self.preparation, source or self.source,
            budget or self.budget(), output_root=Path(tempfile.gettempdir()) / 'ce8-production-packet-fixture')

    def test_full3000_row375_ledger_recipe_census_and_native_roles_are_bound_in_draft(self):
        documents = self.documents(); ledger = documents['NATIVE_LEDGER_SPEC.json']; scope = documents['SCOPE.json']
        receipt = documents['EXPECTED_PREPARED_RECEIPT.json']['expected_receipt']
        self.assertEqual(ledger['plan'], self.plan); self.assertEqual(len(ledger['bindings']['prepared_rows']), 3000)
        self.assertEqual(ledger['ingestion_receipt_sha256'], receipt['receipt_sha256'])
        self.assertEqual(ledger['bindings']['runtime'], documents['RUNTIME.json']['decoding_preimage'])
        self.assertEqual(ledger['bindings']['source'], self.source['code_sha256'])
        self.assertEqual(ledger['bindings']['metadata_roots'], [scope['native_backend']['output_directory']])
        self.assertEqual(scope['execution']['checkpoint_boundaries'], [0, 1, 75, 150, 225, 300, 375])
        self.assertFalse(scope['approved']); self.assertFalse(scope['production_training_authorized'])
        self.assertFalse(documents['QUALIFICATION_DEPENDENCY.json']['completed_pair_verified'])
        self.assertIsNone(documents['QUALIFICATION_DEPENDENCY.json']['selection'])
        self.assertEqual(len(documents['SERVICE_SPEC.json']['role_descriptors']), 11)
        policy=documents['SERVICE_SPEC.json']['supervision']
        self.assertEqual(set(policy),{'schema','owned_directory','max_json_bytes','finalizer'})
        self.assertEqual(len(policy['finalizer']),7)
        self.assertEqual(documents['RESOURCES.json']['sample_attempts_upper'],32768)
        self.assertFalse(documents['PRODUCTION_PRICE_JUSTIFICATION.json']['measured_prices'])
        self.assertNotIn('torch', sys.modules); self.assertNotIn('transformers', sys.modules)

    def test_incomplete_full_dose_caps_discounted_totals_or_measurement_labels_refuse(self):
        edits = (lambda b: b['native_caps']['operation_starts'].update(checkpoint_save=6),
            lambda b: b['native_caps']['operation_starts'].update(loss_forward=2999),
            lambda b: b['native_caps'].update(io_read_bytes=1), lambda b: b['native_caps'].update(wall_seconds=1),
            lambda b: b.update(measured_prices=True), lambda b: b['journal_operation_bounds'].pop('optimizer'),
            lambda b: b['service_bootstrap']['operation_profiles'].pop('qualification_artifact_readback'),
            lambda b: b['native_caps']['operation_starts'].update(fingerprint=811),
            lambda b: b['supervision'].pop('finalizer'),lambda b:b['resource'].update(sample_attempts_upper=True))
        for edit in edits:
            budget = self.budget(); edit(budget)
            with self.subTest(edit=edit), self.assertRaises(ValueError): self.documents(budget=budget)

    def test_changed_qualified_core_or_unselected_boolean_receipt_cannot_build(self):
        source = copy.deepcopy(self.source); source['code_sha256'][registration.admission.CORE_SOURCE_PATHS[1]] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'sources changed'): self.documents(source=source)
        with self.assertRaisesRegex(ValueError, 'source-selected qualification'):
            registration.build_documents(self.qualification, self.plan, self.preparation, self.source, self.budget(),
                output_root=Path(tempfile.gettempdir()) / 'ce8-production-fixture', qualification_selection={'native_qualified': True})

    def test_source_census_spares_and_full_checkpoint_hashes_remain_unmeasured(self):
        budget=self.budget();census=budget['operation_census']
        self.assertEqual(census['fingerprint_arithmetic']['total'],812)
        self.assertEqual(census['finite_start_spares']['fingerprint'],212)
        self.assertEqual(census['resource_guard_source_upper']+census['resource_spare_attempts'],32768)
        self.assertGreaterEqual(budget['journal_operation_bounds']['fingerprint']['identity_hash_bytes_upper'],
                                census['checkpoint_streamed_hash_bytes_upper'])
        self.assertFalse(census['unit_prices_measured']);self.assertFalse(census['model_or_os_execution'])
        self.assertFalse(budget['approved']);self.assertEqual(budget['native_caps']['resumes'],0)

    def test_create_only_actual_ref_chain_remains_denied_and_foreign_plan_ref_precedes_writes(self):
        documents = self.documents()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            bad = copy.deepcopy(self.refs); bad['plan']['sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'canonical TRAIN plan'): registration.write_draft(documents, root, bad)
            self.assertEqual(list(root.iterdir()), [])
            refs = registration.write_draft(documents, root, self.refs)
            for reference in refs.values():
                blob = Path(reference['path']).read_bytes()
                self.assertEqual(hashlib.sha256(blob).hexdigest(), reference['sha256']); self.assertEqual(len(blob), reference['bytes'])
            request = json.loads(Path(refs['REQUEST.json']['path']).read_bytes())
            scope = json.loads(Path(refs['SCOPE.json']['path']).read_bytes())
            justification = json.loads(Path(refs['PRODUCTION_PRICE_JUSTIFICATION.json']['path']).read_bytes())
            authorization = json.loads(Path(refs['SCOPE_AUTHORIZATION.json']['path']).read_bytes())
            self.assertEqual(scope['native_ledger_ref'], refs['NATIVE_LEDGER_SPEC.json'])
            self.assertEqual(scope['production_price_policy']['justification_ref'], refs['PRODUCTION_PRICE_JUSTIFICATION.json'])
            self.assertEqual(justification['production_budget']['resources_sha256'], refs['RESOURCES.json']['sha256'])
            self.assertEqual(authorization['request_preimage_sha256'], registration.digest(registration.controller.request_preimage(request)))
            self.assertIsNone(request['qualifiedreceipt_ref']); self.assertFalse(request['approved']); self.assertFalse(request['training_authorized'])
            with self.assertRaisesRegex(ValueError, 'unapproved'): registration.controller.run_controller(request, Trap())
            with self.assertRaisesRegex(ValueError, 'create-only'): registration.write_draft(documents, root, self.refs)


if __name__ == '__main__': unittest.main()
