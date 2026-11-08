"""Stdlib controls for the registered fixed-shape causal diagnostic."""
import copy
import hashlib
import unittest
from experiments.post_training_mimo_centralized_v1.scripts import diagnose_capture_causality as d


class CausalControls(unittest.TestCase):
    def setUp(self):
        self.cfg = d.load_config(d.PACKAGE / d.CONFIG_RELATIVE)
        self.plan = d.causal_plan(self.cfg['prompt_ids'], self.cfg['generated_ids'])

    def test_registration_byte_pin(self):
        self.assertEqual(hashlib.sha256((d.PACKAGE / d.CONFIG_RELATIVE).read_bytes()).hexdigest(), d.CONFIG_SHA256)

    def test_changed_registration_rejected(self):
        changed = copy.deepcopy(self.cfg); changed['ceilings']['model_or_base_forwards'] = 5
        with self.assertRaises(RuntimeError): d.validate_config(changed)

    def test_foreign_config_path_rejected(self):
        with self.assertRaises(RuntimeError): d.load_config(d.PACKAGE / 'configs/E0_TITAN_BF16.json')

    def test_four_fixed_shapes(self):
        self.assertEqual([len(t['ids']) for t in self.plan], [131]*4)

    def test_suffix_boundaries_and_protected_rows(self):
        baseline = self.plan[0]['ids']
        for trial in self.plan[1:]:
            start = trial['mutation_start']
            self.assertEqual(trial['ids'][:start], baseline[:start])
            self.assertTrue(all(trial['ids'][i] != baseline[i] for i in range(start,131)))
            self.assertEqual(trial['protected_positions'], list(range(127,start)))
            self.assertEqual(trial['ids'][start:], [17,23,151646][start-128:])

    def test_invalid_trace_rejected(self):
        for prompt, generated in [(self.cfg['prompt_ids'][:-1],[16,22,151645]),(self.cfg['prompt_ids'],[16,22,0]),([-1]*128,[16,22,151645])]:
            with self.assertRaises(RuntimeError): d.causal_plan(prompt, generated)

    def test_actual_schedule_counts(self):
        counts=d.expected_counts()
        self.assertEqual(sum(len(t['ids']) for t in self.plan),counts['forward_input_positions'])
        self.assertEqual(1+1+310+len(self.plan),counts['logical_model_calls'])
        self.assertEqual(counts['logical_model_calls'],316)
        self.assertEqual(counts['direct_head_projection_calls'],0)
        self.assertEqual(counts['new_generated_tokens']+counts['optimizer_steps']+counts['teacher_calls'],0)

    def test_actual_triangular_mask_entries(self):
        minimum=-3.3895313892515355e38
        values=[[0.0 if k<=q else minimum for k in range(131)] for q in range(131)]
        measured=d.mask_measurement(values,minimum)
        self.assertEqual(measured['allowed_entries']+measured['forbidden_entries'],17161)
        self.assertTrue(measured['exact_triangular_match'])
        values[0][130]=0.0
        with self.assertRaises(RuntimeError): d.mask_measurement(values,minimum)

    def test_wrong_mask_shape_and_sentinel_rejected(self):
        with self.assertRaises(RuntimeError): d.mask_measurement([[0.0]],-3.3895313892515355e38)
        with self.assertRaises(RuntimeError): d.mask_measurement([[0.0]*131 for _ in range(131)],float('-inf'))

    def test_actual_rotary_positions(self):
        self.assertTrue(d.position_measurement([list(range(131))])['exact_reference_match'])
        with self.assertRaises(RuntimeError): d.position_measurement([list(range(1,132))])

    def test_exact_hook_invocations(self):
        actual={'first_decoder_pre':1,'rotary_pre':1,'final_norm_post':1}
        d.require_hook_counts(actual)
        for key in actual:
            for count in (0,2):
                changed=actual.copy();changed[key]=count
                with self.assertRaises(RuntimeError): d.require_hook_counts(changed)

    def test_bitwise_and_logprob_equality_required(self):
        h='a'*64
        self.assertTrue(d.equality_measurement(h,h,0.0,0.0,[0.0])['passed'])
        for other,maximum,mean,lp in [('b'*64,0.,0.,[0.]),(h,1e-9,0.,[0.]),(h,0.,0.,[1e-9])]:
            self.assertFalse(d.equality_measurement(h,other,maximum,mean,lp)['passed'])

    def test_nonfinite_differences_rejected(self):
        for value in (float('nan'),float('inf'),-1.0):
            with self.assertRaises(RuntimeError): d.equality_measurement('a'*64,'a'*64,value,0.,[])

    def test_diagnostic_grants_closed(self):
        for key in ('qualification_authorized','DEV_collection_authorized','language_grading_authorized','E1_authorized','CHECK_FINAL_access'):
            self.assertIs(self.cfg[key],False)


if __name__=='__main__': unittest.main()
