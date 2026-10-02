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
        qscope = self.qualification['SCOPE.json']; receipt = self.qualification['EXPECTED_PREPARED_RECEIPT.json']['expected_receipt']
        profiles = {kind: {key: qscope['qualifier_accounting']['operation_profiles'][kind][key] for key in registration.journal.PROFILE}
                    for kind in registration.journal.NATIVE_KINDS}
        starts = dict.fromkeys(profiles, 1)
        starts.update(loss_forward=3000, backward=3000, tokenizer_render=3000, fingerprint=753,
            checkpoint_configuration=375, gradient_reset=375, gradient_validation=375, gradient_clipping=375,
            optimizer=375, gradient_cleanup=375, checkpoint_save=7, checkpoint_adapter_io=7)
        setup = registration.qualification_draft.profile(64 * registration.raw.CHUNK, 64 * registration.raw.CHUNK,
                                                       10000, 64 * registration.raw.CHUNK, 60)
        census = receipt['census']; caps = dict.fromkeys(registration.journal.COUNTERS, 1)
        caps.update(schema=registration.journal.NATIVE_CAP_SCHEMA, synthetic_cpu=False, operation_starts=starts,
                    invocations=1, resumes=0, metadata_reserved_bytes=4 * 1024 ** 3)
        for direction, kind in (('forward', 'loss_forward'), ('backward', 'backward')):
            caps[direction + '_selected_tokens'] = starts[kind] * census['max_labeled_tokens']
            caps[direction + '_sequence_tokens'] = starts[kind] * census['max_sequence_positions']
        for counter, field in {'io_read_bytes': 'read_bytes_upper', 'io_write_bytes': 'write_bytes_upper',
                'identity_hash_attempts': 'identity_hash_attempts_upper', 'identity_hash_bytes': 'identity_hash_bytes_upper'}.items():
            caps[counter] = setup[field] + sum(starts[kind] * profiles[kind][field] for kind in profiles) + 8 * 1024 ** 3
        deadline = 19200
        caps['wall_seconds'] = deadline + setup['wall_s_upper'] + sum(starts[kind] * profiles[kind]['wall_s_upper'] for kind in profiles)
        backend_profiles = {kind: {key: (profiles[kind] if kind in profiles else qscope['native_backend']['operation_bounds'][kind])[key]
                                  for key in registration.backend.PROFILE_FIELDS} for kind in registration.backend.OPERATION_KINDS}
        service = copy.deepcopy(self.qualification['SERVICE_SPEC.json']['bootstrap'])
        for kind in ('qualification_reducer_overhead', 'qualification_artifact_readback'):
            service['operation_profiles'][kind] = registration.qualification_draft.profile(256 * 1024 ** 2, 0, 400000, 4 * 1024 ** 3, 120)
        service['operation_profiles']['publication'] = registration.qualification_draft.profile(8 * 1024 ** 3 + 8 * registration.raw.CHUNK + 2,
            registration.raw.MAX_JSON, 8, 8 * 1024 ** 3 + 16 * registration.raw.CHUNK, 120)
        service['caps']['hash_attempts'] = 20000000
        resource = copy.deepcopy(self.qualification['SERVICE_SPEC.json']['resource'])
        stages = set(registration.journal.NATIVE_KINDS) | registration.backend.OPERATION_KINDS | {'ce_boundary', 'runtime_anchor',
            'backend_construct', 'controller_cleanup', 'donor_publish_verify', 'checkpoint_resume'}
        for stage in stages: resource['host_transient_bytes'].setdefault(stage, 64 * registration.raw.CHUNK)
        bounds = dict(self.qualification['SCOPE.json']['qualifier_bootstrap']['self_bounds'])
        return {'schema': registration.BUDGET, 'approved': False, 'production_training_authorized': False,
            'measured_prices': False, 'reservation_basis': 'Explicit synthetic metadata test ceilings; no unit-price or resource measurement.',
            'physical_deadline_seconds': deadline, 'journal_operation_bounds': profiles, 'backend_operation_bounds': backend_profiles,
            'journal_setup_bounds': setup, 'native_caps': caps, 'bootstrap_bounds': bounds, 'service_bootstrap': service,
            'training_reader_bounds': {key: 16 * 1024 ** 3 for key in registration.raw.BUDGET_FIELDS},
            'resource': resource,
            'publication': {'max_json_bytes': registration.raw.MAX_JSON, 'artifact_read_bytes_upper': 8 * 1024 ** 3}}

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
        self.assertFalse(documents['PRODUCTION_PRICE_JUSTIFICATION.json']['measured_prices'])
        self.assertNotIn('torch', sys.modules); self.assertNotIn('transformers', sys.modules)

    def test_incomplete_full_dose_caps_discounted_totals_or_measurement_labels_refuse(self):
        edits = (lambda b: b['native_caps']['operation_starts'].update(checkpoint_save=6),
            lambda b: b['native_caps']['operation_starts'].update(loss_forward=2999),
            lambda b: b['native_caps'].update(io_read_bytes=1), lambda b: b['native_caps'].update(wall_seconds=1),
            lambda b: b.update(measured_prices=True), lambda b: b['journal_operation_bounds'].pop('optimizer'),
            lambda b: b['service_bootstrap']['operation_profiles'].pop('qualification_artifact_readback'))
        for edit in edits:
            budget = self.budget(); edit(budget)
            with self.subTest(edit=edit), self.assertRaises(ValueError): self.documents(budget=budget)

    def test_changed_qualified_core_or_unselected_boolean_receipt_cannot_build(self):
        source = copy.deepcopy(self.source); source['code_sha256'][registration.admission.CORE_SOURCE_PATHS[1]] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'sources changed'): self.documents(source=source)
        with self.assertRaisesRegex(ValueError, 'source-selected qualification'):
            registration.build_documents(self.qualification, self.plan, self.preparation, self.source, self.budget(),
                output_root=Path(tempfile.gettempdir()) / 'ce8-production-fixture', qualification_selection={'native_qualified': True})

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
