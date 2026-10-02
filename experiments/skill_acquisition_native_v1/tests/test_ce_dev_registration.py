"""Real DEV metadata budget drafts; no outputs, models or source activation."""
import copy
import hashlib
import json
import re
from pathlib import Path
import sys
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import ce_dev_scope as scope,ce_dev_registration as registration


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[3]
        proposal=root/'experiments/skill_acquisition_method_lab_v1/CE8_REGISTRATION_PROPOSAL_20261001.json'
        ref={'path':str(proposal),'sha256':scope.registration.PROPOSAL_SHA,'bytes':proposal.stat().st_size}
        cls.proposal,banks,policy=scope.load_metadata(ref)
        cls.scope_documents=scope.build_documents(cls.proposal,banks,policy_ref=policy)
        qdir=Path(r'C:\Users\clclc\BusinessProject\comp-rs-001-records\s0\r-oracle-sft-qualification-20261002\canonical-attempt10\registration')
        cls.q={name:json.loads((qdir/name).read_bytes()) for name in ('SCOPE.json','SERVICE_SPEC.json','SOURCE_SNAPSHOT.json')}

    def budget(self):
        return registration.propose_budget(self.proposal,self.scope_documents,self.q,
            output_root=Path(tempfile.gettempdir())/'ce8-dev-budget-fixture')

    def documents(self,budget=None):
        return registration.build_documents(self.scope_documents,budget or self.budget(),source=self.q['SOURCE_SNAPSHOT.json'])

    def test_five_fixed_read_census_coldload_cleanup_and_caps_compose(self):
        budget=self.budget();self.assertEqual(budget['total_completions'],9750)
        for name,model,_ in registration.runtime.RUNS:
            read=budget['reads'][name];census=read['census'];entry=read['entry_spec']
            self.assertEqual(census['runtime_starts']['dev_decode'],1950)
            self.assertEqual(census['runtime_starts']['dev_journal'],1953)
            self.assertEqual(census['runtime_starts']['dev_fingerprint'],2)
            self.assertEqual(census['runtime_starts']['dev_delta_rebuild'],int(model=='CE375_donor'))
            self.assertEqual(census['live_refresh_contract']['cached_worker_identity_os_queries'],0)
            self.assertEqual(read['declared_callback_counts']['dev_entry_lease_assert'],census['source_verifier_calls_upper'])
            self.assertEqual(read['declared_callback_counts']['dev_entry_verify'],census['source_verifier_calls_upper'])
            self.assertEqual(read['declared_callback_counts']['worker_os_binding'],64)
            self.assertEqual(entry['resource']['sample_attempts_upper'],4096)
            self.assertEqual(set(entry['constructor_bounds']),set(registration.services.COUNTERS))
            self.assertRegex(entry['lease']['label'],r'^comp-rs-001-[a-z0-9][a-z0-9.-]{2,80}$')
            self.assertEqual(read['declared_callback_counts']['worker_runtime_binding'],8)
            self.assertEqual(read['declared_callback_counts']['worker_source_verification'],
                             2 * len(self.q['SOURCE_SNAPSHOT.json']['source_files']) + 32)
            self.assertGreaterEqual(entry['training_reader_bounds']['identity_hash_attempts_upper'],1600000)
            self.assertGreater(entry['training_reader_bounds']['io_read_bytes_upper'],entry['constructor_bounds']['read_bytes'])
            self.assertIn('qualification_artifact_readback',entry['bootstrap']['operation_profiles'])
            self.assertIn('qualification_reducer_overhead',entry['bootstrap']['operation_profiles'])
            self.assertEqual(set(read['policy']['operation_profiles']),registration.runtime.KINDS)
            self.assertGreaterEqual(read['policy']['operation_profiles']['dev_cleanup']['wall_s_upper'],read['policy']['physical_wall_seconds_upper'])
            for key,total in read['composition']['permanent_profile_totals_with_actual_wall_and_metadata'].items():
                self.assertGreaterEqual(entry['bootstrap']['caps'][key],total)
            parent=read['supervisor_spec'];parent_census=read['supervisor_census']
            self.assertEqual(set(parent),{'schema','bootstrap','constructor_bounds','resource','current_worker','launch','reader_bounds'})
            self.assertNotEqual(parent['bootstrap']['owned_directory'],entry['bootstrap']['owned_directory'])
            self.assertNotEqual(parent_census['metadata_intake_slot'],parent_census['worker_metadata_intake_slot'])
            self.assertEqual(parent_census['declared_callback_counts']['process_launch'],1)
            self.assertEqual(parent_census['declared_callback_counts']['process_terminal'],4)
            self.assertTrue(parent_census['same_handle_terminal_required'])
            for key,total in parent_census['composition']['permanent_profile_totals_with_actual_wall_and_metadata'].items():
                self.assertGreaterEqual(parent['bootstrap']['caps'][key],total)
        self.assertFalse(budget['unit_prices_measured']);self.assertFalse(budget['evaluation_authorized'])
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

    def test_discounted_counts_caps_or_cleanup_cannot_materialize(self):
        edits=(lambda b:b['reads']['Supa-read-1']['census']['runtime_starts'].update(dev_decode=1949),
            lambda b:b['reads']['Supa-read-1']['declared_callback_counts'].update(dev_decode=1949),
            lambda b:b['reads']['Supa-read-1']['entry_spec']['bootstrap']['caps'].update(read_bytes=1),
            lambda b:b['reads']['Supa-read-1']['supervisor_spec']['bootstrap']['caps'].update(read_bytes=1),
            lambda b:b['reads']['Supa-read-1']['entry_spec']['training_reader_bounds'].update(io_read_bytes_upper=8388608),
            lambda b:b['reads']['Supa-read-1']['policy']['operation_profiles']['dev_cleanup'].update(wall_s_upper=1),
            lambda b:b.update(unit_prices_measured=True),lambda b:b['reads'].pop('CE375_donor-read-1'))
        for edit in edits:
            budget=self.budget();edit(budget)
            with self.subTest(edit=edit),self.assertRaises(ValueError):self.documents(budget)

    def test_real_create_only_draft_keeps_complete_scope_and_unresolved_grants(self):
        docs=self.documents();config=docs['DEV_ENTRY_CONFIGURATION_DRAFT.json']
        self.assertEqual(set(config),{'schema','context_refs','entry_specs','training_selection','training_selection_sha256','configuration_sha256'})
        self.assertIsNone(config['training_selection'])
        parent=docs['DEV_SUPERVISOR_CONFIGURATION_DRAFT.json']
        self.assertEqual(set(parent),{'schema','worker_configuration_ref','scope_receipt_ref','supervisor_specs','configuration_sha256'})
        self.assertIsNone(parent['worker_configuration_ref'])
        for context in docs['DEV_CONTEXT_DRAFTS.json'].values():
            self.assertFalse(context['evaluation_authorized']);self.assertIsNone(context['model']);self.assertIsNone(context['runtime'])
            self.assertEqual(context['plan']['total_greedy_completions'],9750)
        with tempfile.TemporaryDirectory() as temp:
            refs=registration.write_draft(docs,Path(temp))
            for reference in refs.values():
                blob=Path(reference['path']).read_bytes();self.assertEqual(hashlib.sha256(blob).hexdigest(),reference['sha256'])
            parent_file=json.loads(Path(refs['DEV_SUPERVISOR_CONFIGURATION_DRAFT.json']['path']).read_bytes())
            self.assertEqual(parent_file['scope_receipt_ref'],refs['DEV_RSP_SCOPE_RECEIPT.json'])
            with self.assertRaisesRegex(ValueError,'create-only'):registration.write_draft(docs,Path(temp))


if __name__=='__main__':unittest.main()
