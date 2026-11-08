"""Small synthetic checks; never read the real partial alignment matrix."""
import copy
import importlib.util
import math
import pathlib
import sys
import unittest
sys.dont_write_bytecode=True
path=pathlib.Path(__file__).with_name('audit_alignment_matrix_v2.py')
spec=importlib.util.spec_from_file_location('independent_matrix_audit',path)
audit=importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)

def scalar(tokens, cosine=.7, mse=1., ratio=1.):
    return {'tokens':tokens,'mean_token_cosine':cosine,'mean_token_relative_mse':mse,
        'global_relative_mse':mse,'predicted_rms':ratio,'target_rms':1.,'residual_magnitude_ratio':ratio}

def records():
    out=[]
    for i,j in audit.PAIRS:
        objectives={}
        for name in audit.OBJECTIVES:
            cosine,mse=(.4,2.) if name=='B0' else (.3,2.) if name in ('BS','BR') else (.7,1.)
            groups={group:{'all_tokens':scalar(64,cosine,mse),'ordinary_tokens':scalar(48,cosine,mse),
                'special_tokens':scalar(16,cosine,mse),'final_prompt_token':scalar(32,cosine,mse)} for group in ('up','down','delta')}
            groups['mapped_delta_to_supa_rms']=1.
            steps=96 if name in ('B2','B3') else 0
            objectives[name]={'heldout':groups,'optimizer_steps':steps,
                'training_trace':[{'step':k,'training_prompt_index':k,'loss':1.,'delta_loss':1.} for k in range(steps)],
                'bridge_coefficients':5246464}
        out.append({'supa_layer':i,'donor_layer':j,'objectives':objectives,
            'config_sha256':audit.EXPECTED_CONFIG_SHA256,'donor_state_before':'a'*64,
            'donor_state_after':'a'*64,'native_replay_bit_identical':True,
            'trainable_backbone_parameters':0,'backbone_gradient_tensors':0})
    return out

class RankingTests(unittest.TestCase):
    def test_missing_matrix_rejected_before_metrics_access(self):
        with self.assertRaises(audit.AuditError) as caught:
            audit.reconstruct_selection([{'supa_layer':8,'donor_layer':12}])
        self.assertEqual(caught.exception.code,'INCOMPLETE_MATRIX')

    def test_duplicate_pair_rejected(self):
        rows=records();rows[-1]=rows[0]
        with self.assertRaises(audit.AuditError):audit.reconstruct_selection(rows)

    def test_all_six_objectives_required(self):
        rows=records();del rows[0]['objectives']['BR']
        with self.assertRaises(audit.AuditError):audit.reconstruct_selection(rows)

    def test_rank_prefers_cosine_then_mse_then_depths_then_objective(self):
        rows=records()
        decision=audit.reconstruct_selection(rows)
        self.assertEqual((decision['selected'][0]['supa_layer'],decision['selected'][0]['donor_layer'],decision['selected'][0]['objective']),(8,12,'B1'))
        target=rows[-1]['objectives']['B3']['heldout']['delta']['ordinary_tokens']
        target['mean_token_cosine']=.8
        self.assertEqual(audit.reconstruct_selection(rows)['selected'][0]['objective'],'B3')
        rows[0]['objectives']['B2']['heldout']['delta']['ordinary_tokens']['mean_token_cosine']=.8
        rows[0]['objectives']['B2']['heldout']['delta']['ordinary_tokens']['mean_token_relative_mse']=.9
        selected=audit.reconstruct_selection(rows)['selected'][0]
        self.assertEqual((selected['supa_layer'],selected['donor_layer'],selected['objective']),(8,12,'B2'))

    def test_exact_boundaries_inclusive_and_one_ulp_below_rejected(self):
        rows=records();obj=rows[0]['objectives']
        obj['B0']['heldout']['delta']['ordinary_tokens']['mean_token_cosine']=.5
        for name in ('BS','BR'):obj[name]['heldout']['delta']['ordinary_tokens']['mean_token_cosine']=.5
        candidate=obj['B1']['heldout']['delta']
        candidate['ordinary_tokens'].update(mean_token_cosine=.6,mean_token_relative_mse=1.6,residual_magnitude_ratio=.25)
        candidate['all_tokens']['mean_token_cosine']=.5
        decision=audit.reconstruct_selection(rows)['comparisons'][0]
        self.assertTrue(decision['go'])
        candidate['ordinary_tokens']['mean_token_cosine']=math.nextafter(.6,0.)
        self.assertFalse(audit.reconstruct_selection(rows)['comparisons'][0]['go'])

    def test_all_six_registered_gates_are_binding(self):
        mutations=[('ordinary_tokens','mean_token_cosine',.59),
                   ('ordinary_tokens','mean_token_relative_mse',1.61),
                   ('all_tokens','mean_token_cosine',.49),
                   ('ordinary_tokens','residual_magnitude_ratio',4.01)]
        for stratum,key,value in mutations:
            rows=records();rows[0]['objectives']['B1']['heldout']['delta'][stratum][key]=value
            with self.subTest(key=key,value=value):self.assertFalse(audit.reconstruct_selection(rows)['comparisons'][0]['go'])
        for control in ('B0','BS','BR'):
            rows=records();rows[0]['objectives'][control]['heldout']['delta']['ordinary_tokens']['mean_token_cosine']=.65
            with self.subTest(control=control):self.assertFalse(audit.reconstruct_selection(rows)['comparisons'][0]['go'])

    def test_no_eligible_candidate_keeps_generation_closed(self):
        rows=records()
        for row in rows:
            for name in ('B1','B2','B3'):row['objectives'][name]['heldout']['delta']['ordinary_tokens']['mean_token_cosine']=.2
        decision=audit.reconstruct_selection(rows)
        self.assertEqual(decision['status'],'MECHANISM_NO_GO')
        self.assertEqual(decision['selected'],[])
        self.assertFalse(decision['large_generation_licensed'])
        self.assertFalse(decision['small_generation_eligible_for_separate_freeze'])

class MetricAndPathTests(unittest.TestCase):
    def checker(self):
        checker=audit.MatrixAuditor('synthetic-fixture')
        # Read only the already-frozen tiny config to establish its exact digest;
        # no real capture, pair metrics or activation artifacts are opened.
        import json
        checker.config=json.loads((audit.TRACK/'configs/alignment_v2.json').read_text())
        checker.held_tokens=64;checker.training_order=None;checker.stratum_counts=None
        return checker

    def test_complete_synthetic_metric_schema(self):
        self.checker().validate_metrics(records()[0],(8,12))

    def test_optimizer_budget_rejects_early_stop(self):
        row=records()[0];row['objectives']['B2']['optimizer_steps']=95
        with self.assertRaises(audit.AuditError):self.checker().validate_metrics(row,(8,12))

    def test_nonoptimized_control_cannot_have_steps(self):
        row=records()[0];row['objectives']['BR']['optimizer_steps']=1
        with self.assertRaises(audit.AuditError):self.checker().validate_metrics(row,(8,12))

    def test_nonfinite_or_missing_stratum_rejected(self):
        for mode in ('nan','missing','wrong_count'):
            row=records()[0];strata=row['objectives']['B0']['heldout']['delta']
            if mode=='nan':strata['ordinary_tokens']['mean_token_cosine']=float('nan')
            elif mode=='missing':del strata['special_tokens']
            else:strata['final_prompt_token']['tokens']=31
            with self.subTest(mode=mode),self.assertRaises(audit.AuditError):self.checker().validate_metrics(row,(8,12))

    def test_scope_rejects_parent_escape_and_tools_output(self):
        for path in (audit.AUDIT_ROOT.parent/'forbidden',audit.TOOLS_DIR/'forbidden',audit.AUDIT_ROOT):
            with self.subTest(path=str(path)),self.assertRaises(audit.AuditError):audit.exclusive_output(path)

if __name__=='__main__':unittest.main(verbosity=2)
