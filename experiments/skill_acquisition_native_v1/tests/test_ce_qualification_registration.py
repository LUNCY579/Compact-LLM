"""Actual canonical metadata materialization; no activation/models/processes."""
import copy
from pathlib import Path
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import ce_qualification_registration as registration
from . import test_ce_registration as fixtures


class ConfigurationRegistrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): fixtures.RegistrationTests.setUpClass()

    def test_full_config_is_acyclic_and_authorizes_only_discarded_qualification(self):
        t = fixtures.RegistrationTests()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'registration'; docs = t.documents()
            head = docs['REGISTRATION_DRAFT.json']['source_head_basis']
            refs, config = registration.write_qualification_configuration(docs, folder,
                {'decision': t.decision_ref, 'preparation': t.proposal['input_refs']['preparation'], 'plan': t.plan},
                launch_window={'not_before': 1, 'not_after': 2}, source_head=head)
            self.assertTrue(config['request']['qualification_authorized'])
            self.assertFalse(config['request']['training_authorized'])
            self.assertEqual(config['service_spec']['scope_adapters']['worker']['request_ref'], refs['REQUEST.json'])
            self.assertNotIn('service_spec_ref', config['request'])
            self.assertEqual(registration.digest({k: v for k, v in config.items() if k != 'configuration_sha256'}),
                             config['configuration_sha256'])
            self.assertFalse(docs['SCOPE.json']['qualification_authorized'])
            with self.assertRaisesRegex(ValueError, 'create-only'):
                registration.write_qualification_configuration(docs, folder,
                    {'decision': t.decision_ref, 'preparation': t.proposal['input_refs']['preparation'], 'plan': t.plan},
                    launch_window={'not_before': 1, 'not_after': 2}, source_head=head)

    def test_changed_choice_or_training_scope_cannot_be_authorized(self):
        t = fixtures.RegistrationTests(); docs = t.documents(); head = docs['REGISTRATION_DRAFT.json']['source_head_basis']
        for mutate in (lambda d: d['REGISTRATION_DRAFT.json'].update(choice='paired_canonical_permuted'),
                       lambda d: d['SCOPE.json'].update(production_training_authorized=True)):
            altered = copy.deepcopy(docs); mutate(altered)
            with self.assertRaises(ValueError): registration.authorize_documents(altered, source_head=head)
