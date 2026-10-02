"""DEV identity/scope controls only; no model outputs, models or sealed banks."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import ce_dev_scope as scope


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[3]/'experiments/skill_acquisition_method_lab_v1/CE8_REGISTRATION_PROPOSAL_20261001.json'
        cls.ref={'path':str(path),'bytes':path.stat().st_size,'sha256':scope.registration.PROPOSAL_SHA}
        cls.proposal,cls.banks,cls.policy=scope.load_metadata(cls.ref)

    def documents(self):return scope.build_documents(self.proposal,self.banks,policy_ref=self.policy)

    def test_actual_fixed_RSP_guard_scope_and_five_reads_are_bound_without_outputs(self):
        docs=self.documents();value=docs['DEV_EVALUATION_SCOPE.json'];plan=docs['DEV_RUN_PLAN.json']
        self.assertEqual([len(value['banks'][key]['ids']) for key in scope.BANKS],[250,250,250,1200])
        self.assertEqual(sum(read['rows'] for read in plan['reads']),9750)
        self.assertEqual([read['model_name'] for read in plan['reads']],['Supa','Supa','warm_anchor','warm_anchor','CE375_donor'])
        self.assertEqual(plan['generation_token_ceiling'],1248000)
        self.assertFalse(value['evaluation_authorized']);self.assertFalse(value['model_outputs_read'])
        self.assertFalse(plan['production_dispatch_authorized']);self.assertFalse(plan['refill_allowed'])
        for name in scope.BANKS[:3]:
            self.assertEqual(value['scope_bindings'][name]['signatures'][self.proposal['banks'][name]['signature']]['binding'],'full')
            self.assertEqual(value['banks'][name]['ids_sha256'],self.proposal['banks'][name]['ids_sha256'])
        self.assertEqual(value['scope_bindings']['guard_dev_legacy']['kind'],'guard_bank_sha256')
        self.assertIsNone(value['banks']['guard_dev_legacy']['scope_receipt_sha256'])
        self.assertEqual(len(value['banks']['guard_dev_legacy']['family_by_id']),1200)
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)

    def test_changed_ID_parent_or_guard_family_cannot_freeze(self):
        proposal=copy.deepcopy(self.proposal);proposal['banks']['dev_core_P']['ids'][0]='foreign'
        with self.assertRaises(ValueError):scope.build_documents(proposal,self.banks,policy_ref=self.policy)
        proposal=copy.deepcopy(self.proposal);proposal['banks']['dev_core_R']['parent_ids'][0]='foreign'
        with self.assertRaisesRegex(ValueError,'parent IDs'):scope.build_documents(proposal,self.banks,policy_ref=self.policy)
        proposal=copy.deepcopy(self.proposal);proposal['banks']['guard_dev_legacy']['families']['countdown']=149
        with self.assertRaisesRegex(ValueError,'family counts'):scope.build_documents(proposal,self.banks,policy_ref=self.policy)

    def test_Q1_R_S_RS_receipt_does_not_bind_newP_scope(self):
        docs=self.documents();old=copy.deepcopy(docs['DEV_RSP_SCOPE_RECEIPT.json']);old['signatures'].pop('P')
        declared=docs['DEV_EVALUATION_SCOPE.json']['banks']['dev_core_P']['scoring_scope']
        with self.assertRaisesRegex(ValueError,'lacks signature P'):
            scope.scorer.bind_scope_receipt(self.banks['dev'],declared,old,expected_sha256=scope.digest(old))

    def test_actual_scope_matches_analyzer_before_any_model_or_output_binding(self):
        from experiments.skill_acquisition_native_v1 import ce_dev_analysis as analysis
        actual=self.documents()['DEV_EVALUATION_SCOPE.json']
        self.assertIs(analysis._scope(actual),actual)
        self.assertEqual(actual['greedy'],analysis.GREEDY)
        report=analysis.analyze_dev(actual)
        self.assertEqual(report['status'],'NOT_RUN');self.assertFalse(report['evaluation_authorized'])

    def test_real_create_only_documents_and_sealed_or_foreign_bank_refuse(self):
        with tempfile.TemporaryDirectory() as temp:
            refs=scope.write_draft(self.documents(),Path(temp))
            for ref in refs.values():
                blob=Path(ref['path']).read_bytes();self.assertEqual(hashlib.sha256(blob).hexdigest(),ref['sha256'])
            with self.assertRaisesRegex(ValueError,'create-only'):scope.write_draft(self.documents(),Path(temp))
        foreign=scope.scorer.Bank(self.banks['dev'],role='test',bank_sha256=scope.DEV_SHA)
        with self.assertRaisesRegex(ValueError,'literal canonical DEV'):
            scope.build_documents(self.proposal,{'dev':foreign,'guard_dev':self.banks['guard_dev']},policy_ref=self.policy)


if __name__=='__main__':unittest.main()
