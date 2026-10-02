"""Pure projected receipts on actual frozen DEV metadata; no model outputs."""
import copy
from pathlib import Path
import sys
import unittest

from experiments.skill_acquisition_native_v1 import ce_dev_analysis as a,ce_dev_scope as manifest


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path=Path(__file__).resolve().parents[3]/'experiments/skill_acquisition_method_lab_v1/CE8_REGISTRATION_PROPOSAL_20261001.json'
        ref={'path':str(path),'sha256':manifest.registration.PROPOSAL_SHA,'bytes':path.stat().st_size}
        proposal,banks,policy=manifest.load_metadata(ref)
        cls.actual_scope=manifest.build_documents(proposal,banks,policy_ref=policy)['DEV_EVALUATION_SCOPE.json']
    def fixtures(self):
        scope=copy.deepcopy(self.actual_scope)
        for index,model in enumerate(a.MODELS,1):
            scope['model_bindings'][model].update(model_sha256=str(index)*64,tokenizer_sha256='d'*64,decoding_sha256='e'*64)
        def read(name,model,process,pass_ids,content_ids=None):
            bank=scope['banks'][name];rows=[];scores=[];content_ids=pass_ids if content_ids is None else content_ids
            for rid in bank['ids']:
                rows.append({'id':rid,'parent_id':bank['parent_by_id'][rid],'prompt_hash':bank['prompt_hash_by_id'][rid],
                    'prompt_ids_sha256':a.digest(['projected-prompt',rid]),'completion_ids':[7,151645],
                    'response':'projected metadata fixture, not actual model output','stop_reason':'eos','generated_tokens':2})
                scores.append({'id':rid,'parent_id':bank['parent_by_id'][rid],'strict':int(rid in pass_ids),
                    'content_pass':rid in content_ids,'stop_reason':'eos','generated_tokens':2})
            scoring_scope=copy.deepcopy(bank['scoring_scope']);scoring_scope.pop('ids')
            binding=scope['model_bindings'][model]
            receipt={'schema_version':a.scorer.SCHEMA_VERSION,'kind':'scoped_scoring_receipt','role':bank['role'],
                'bank_sha256':bank['bank_sha256'],'manifest_sha256':scope['manifest_sha256'],
                'scope':scoring_scope,'scope_receipt_sha256':bank['scope_receipt_sha256'],'diagnostic':False,
                'run':{key:binding[key] for key in ('model_sha256','tokenizer_sha256','decoding_sha256')},
                'code_sha256':scope['source']['scorer_code_sha256'],'code_bundle_sha256':binding['scorer_bundle_sha256'],
                'strict_endpoint':'experiments.maxrl_composable_skills_v2.corpus.reward (unchanged)',
                'responses_sha256':a.digest([{key:row[key] for key in a.scorer.RESPONSE_FIELDS} for row in rows]),
                'strict_correct_ids':[rid for rid in bank['ids'] if rid in pass_ids],'rows':scores}
            return {'schema':a.READ,'bank':name,'model_name':model,'process_identity_sha256':process,
                'journal_rows':rows,'scoring_receipt':receipt}
        baseline={}
        for index,model in enumerate(a.BASELINES,1):
            first={};second={};members={}
            for name in a.BANKS:
                ids=scope['banks'][name]['ids'];n=0 if name=='dev_core_R' else (59 if name=='guard_dev_legacy' else 29)
                first[name]=read(name,model,str(index+3)*64,set(ids[:n]));second[name]=read(name,model,str(index+5)*64,set(ids[:n]))
                members[name]=a.scorer.enrollment_from_receipt(first[name]['scoring_receipt'])
            baseline[model]={'first_reads':first,'second_reads':second,'memberships':members,'zero_token_churn':True}
        donor={}
        for name in a.BANKS:
            ids=scope['banks'][name]['ids'];n=13 if name=='dev_core_R' else (59 if name=='guard_dev_legacy' else 29)
            donor[name]=read(name,'CE375_donor','9'*64,set(ids[:n]),set(ids[:n+3]))
        return scope,baseline,donor
    def test_actual_scope_draft_has_explicit_missing_receipts_and_no_fake_zero_metrics(self):
        report=a.analyze_dev(self.actual_scope)
        self.assertEqual(report['status'],'NOT_RUN');self.assertTrue(report['missing_dependencies'])
        self.assertNotIn('acquisition_H1',report);self.assertFalse(report['native_qualified'])
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
    def test_actual_full1950_scope_projected_join_H1_and_conditional_zero_loss_attainability(self):
        scope,baseline,donor=self.fixtures();report=a.analyze_dev(scope,baseline,donor)
        self.assertEqual(report['acquisition_H1']['state'],'PASS');self.assertEqual(report['acquisition_H1']['strict_net'],13)
        self.assertEqual(report['acquisition_H1']['content_net'],16)
        retention=report['source_correct_retention']['Supa']
        self.assertEqual(retention['dev_core_S']['n'],29);self.assertEqual(retention['guard_dev_legacy']['n'],59)
        self.assertEqual(retention['dev_core_S']['loss_limb']['state'],'PASS')
        self.assertTrue(report['strict_content_separate']);self.assertIn('not750 independent',report['assumptions'][0])
    def test_membership_and_zero_churn_cannot_be_claimed_from_flags(self):
        scope,baseline,donor=self.fixtures()
        baseline['warm_anchor']['second_reads']['dev_core_R']['journal_rows'][0]['completion_ids'][0]=8
        with self.assertRaisesRegex(ValueError,'STOP_NONDETERMINISTIC'):a.analyze_dev(scope,baseline,donor)
        scope,baseline,donor=self.fixtures();baseline['Supa']['memberships']['dev_core_S']['source_correct_ids'].pop()
        with self.assertRaisesRegex(ValueError,'frozen membership'):a.analyze_dev(scope,baseline,donor)
    def test_partial_parent_runtime_deployed_arm_and_process_alias_refuse(self):
        scope,baseline,donor=self.fixtures()
        for edit in (lambda d:d['dev_core_R']['journal_rows'].pop(),
                     lambda d:d['dev_core_S']['journal_rows'][0].update(parent_id='foreign'),
                     lambda d:d['dev_core_P']['scoring_receipt']['run'].update(decoding_sha256='f'*64),
                     lambda d:d['guard_dev_legacy']['scoring_receipt']['run'].update(model_sha256='f'*64)):
            changed=copy.deepcopy(donor);edit(changed)
            with self.assertRaises(ValueError):a.analyze_dev(scope,baseline,changed)
        baseline['Supa']['second_reads']['dev_core_R']['process_identity_sha256']=baseline['Supa']['first_reads']['dev_core_R']['process_identity_sha256']
        with self.assertRaisesRegex(ValueError,'distinct independent'):a.analyze_dev(scope,baseline,donor)
    def test_retention_losses_are_not_offset_by_many_repairs_and_empty_sources_uncertifiable(self):
        scope,baseline,donor=self.fixtures();name='dev_core_S';ids=scope['banks'][name]['ids']
        score=donor[name]['scoring_receipt'];score['rows'][0]['strict']=0
        for row in score['rows'][29:40]:row['strict']=1
        score['strict_correct_ids']=[row['id'] for row in score['rows'] if row['strict']]
        report=a.analyze_dev(scope,baseline,donor);s=report['source_correct_retention']['warm_anchor'][name]
        self.assertEqual(s['losses'],1);self.assertEqual(s['repairs'],11);self.assertEqual(s['loss_limb']['losses'],1)
        for model in a.BASELINES:
            for reads in ('first_reads','second_reads'):
                receipt=baseline[model][reads]['dev_core_P']['scoring_receipt']
                for row in receipt['rows']:row['strict']=0
                receipt['strict_correct_ids']=[]
            baseline[model]['memberships']['dev_core_P']=a.scorer.enrollment_from_receipt(baseline[model]['first_reads']['dev_core_P']['scoring_receipt'])
        report=a.analyze_dev(scope,baseline,donor)
        self.assertEqual(report['source_correct_retention']['Supa']['dev_core_P']['loss_limb']['state'],'UNCERTIFIABLE')

    def test_significant_twelve_gain_rows_still_fail_net_floor_and_zero_discordance_uncertifiable(self):
        scope,baseline,donor=self.fixtures();receipt=donor['dev_core_R']['scoring_receipt']
        receipt['rows'][12]['strict']=0;receipt['strict_correct_ids']=[row['id'] for row in receipt['rows'] if row['strict']]
        report=a.analyze_dev(scope,baseline,donor)
        self.assertEqual(report['acquisition_H1']['exact_mcnemar']['state'],'PASS')
        self.assertEqual(report['acquisition_H1']['strict_net'],12);self.assertEqual(report['acquisition_H1']['state'],'FAIL')
        for row in receipt['rows']:row['strict']=0
        receipt['strict_correct_ids']=[]
        report=a.analyze_dev(scope,baseline,donor)
        self.assertEqual(report['acquisition_H1']['state'],'UNCERTIFIABLE')


if __name__=='__main__':unittest.main()
