"""Independent optional grammar audit through unchanged inert byte replay."""
from pathlib import Path
from copy import deepcopy
import ast, hashlib, json, sys, types, unittest
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;GFILE=WORK/'bounded_boundary_grammar.py';TEST=WORK/'test_boundary_grammar.py';PACKAGE=next(p for p in WORK.parents if p.name=='rineya_synth_centralized_reasoning_v1');ENGINE=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006/prefix_library_evidence.py';BASETEST=ENGINE.with_name('test_library_evidence.py');POLICY=ENGINE.with_name('prefix_library_policy.py');FILES=[GFILE,TEST,ENGINE,BASETEST,POLICY,WORK/'LIBRARY_EXPECTATION_INPUTS001.json',PACKAGE/'prefix_tensor_contract.py']
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(GFILE)]['sha256']=='d8e3d06bd61e57af51a547f6a72323b5f4d9ebe2802e3bba1b2c0f5182afe5be';assert BEFORE[str(TEST)]['sha256']=='f9b0a4ba232ea368df54a9f0d6926251336fc298e6a3dc698368185aa85c6f58';assert BEFORE[str(ENGINE)]['sha256']=='aebab23452fa53ad71edbd4762620d929b4b8a936d41eb278f3b1f15ff1a57f2'
def load(p,name):m=types.ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);old=sys.modules.get('prefix_library_evidence');T=load(TEST,'independent_grammar_fixture');Base=T.Controls;DETAILS={}
class Controls(Base):
    def test_independent_sibling_permutations_exact_effects_and_source_identity(self):
        for reverse in (False,True):
            result=self.run_packet(self.packet(reverse));self.assertTrue(result['conservation_complete']);self.assertTrue(result['exact_library_effect_replay_unchanged']);self.assertFalse(result['native_PASS']);self.assertFalse(result['native_authority_issued'])
        grammar=self.grammar;self.assertEqual(set(grammar['required_extensions'])|set(grammar['optional_extensions']),set(self.policy['extension_transitions']));self.assertEqual(grammar['maximum_boundaries'],1+2*len(self.policy['extension_transitions']));self.assertEqual(grammar['maximum_depth'],64);self.assertEqual(grammar['maximum_observations'],4096);DETAILS['two_sibling_orders_actual_original_replay']=True
    def test_independent_unknown_repeat_missing_parent_depth_and_fixed_paths_refuse(self):
        self.test_unknown_repeat_unmatched_parent_and_missing_required_refuse()
        for change in ('depth','required','optional'):
            original=deepcopy(self.grammar)
            try:
                if change=='depth':self.grammar['maximum_depth']=65
                elif change=='required':self.grammar['required_extensions'].append('unregistered')
                else:self.grammar['optional_extensions'].append(self.ext)
                with self.assertRaises(ValueError):self.run_packet(self.packet())
            finally:self.grammar=original
    def test_independent_missing_events_effect_changes_snapshot_and_unload_refuse(self):
        self.test_wrong_effect_or_missing_event_still_rejected_by_original_consumer()
        # Restore expected effect after inherited test's deliberate mutation.
        self.policy['extension_transitions'][self.ext]['create']=[self.ext]
        value=self.packet();value['observations'][3]['snapshot']['libraries'][-1]['module_handle']+=100
        with self.assertRaises(ValueError):self.run_packet(value)
        value=self.packet();value['observations'][3]['event_trace']['events'][0]['event']='unload'
        with self.assertRaises(ValueError):self.run_packet(value)
        DETAILS['original_snapshot_event_effect_rejections_actual']=True
    def test_independent_exec_before_create_unclosed_sequence_and_model_refuse(self):
        self.test_exec_before_create_and_unclosed_stack_cannot_complete()
        value=self.packet();value['observations'][2]['boundary']['id']=9;value['observations'][2]['boundary_id']=9
        with self.assertRaisesRegex(ValueError,'sequential'):self.run_packet(value)
        self.grammar['model']=True
        with self.assertRaises(ValueError):self.run_packet(self.packet())
if __name__=='__main__':
    names=('test_independent_sibling_permutations_exact_effects_and_source_identity','test_independent_unknown_repeat_missing_parent_depth_and_fixed_paths_refuse','test_independent_missing_events_effect_changes_snapshot_and_unload_refuse','test_independent_exec_before_create_unclosed_sequence_and_model_refuse');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in names));sys.meta_path.remove(trap)
    if old is None:sys.modules.pop('prefix_library_evidence',None)
    else:sys.modules['prefix_library_evidence']=old
    after=pins();receipt={'kind':'INDEPENDENT_OPTIONAL_BOUNDARY_GRAMMAR_DRAFT_REVIEW001','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'source_pins_unchanged':BEFORE==after,'details':DETAILS,'actual_original_raw_event_snapshot_effect_replay_executed':True,'observations_explicit_inert_CPU':True,'fixed_policy_library_effects_preserved':True,'integrated_policy_changed':False,'native_API_or_framework_called':False,'actual_native_effect_expectations_known':False,'native_authority_issued':False,'native_qualification':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_BOUNDARY_GRAMMAR001.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
