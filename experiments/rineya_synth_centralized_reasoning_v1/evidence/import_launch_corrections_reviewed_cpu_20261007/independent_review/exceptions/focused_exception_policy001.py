"""Independent exact future-leaf Git exception registration, real CPU repos."""
from pathlib import Path
from types import ModuleType
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_exception_policy001';TEST=FROZEN/'test_exception_policy.py';FILES=[p for p in FROZEN.rglob('*') if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'registration_git_join.py')]['sha256']=='ce61740e153be1be370eba4d64984d9b6244cafa50b27ebff680a7c4204f32e2';assert BEFORE[str(FROZEN/'import_runtime_qualification.py')]['sha256']=='fa2598114c8834b959253ec0ec2de5e2af7a18f15299304981125e9869afeeb3';assert BEFORE[str(TEST)]['sha256']=='99f912f6b58a513616590c24c1462e6f8c75ebf350b848d6c69f6a836b24a221'
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);T=ModuleType('independent_exact_exception_fixture');T.__file__=str(TEST);exec(compile(TEST.read_bytes(),str(TEST),'exec'),T.__dict__);Base=T.Controls;DETAILS={}
class Independent(Base):
    def test_actual_two_commit214_absent_future_leaves_and_native2_policy(self):
        self.test_actual214future_absent_leaves_and_native2_Q_policy();self.assertNotEqual(self.impl,self.head);self.assertEqual(len(self.sources),196);self.assertEqual(len(self.native),2);DETAILS['actual_two_commit_sources']=196;DETAILS['actual_absent_future_leaves']=214
    def test_source_packet_escape_and_registered_native_snapshot_rebind_refuse(self):self.test_unknown_native_originalsource_packet_and_escape_refuse()
    def test_actual_untracked_registered_leaf_allowed_foreign_tracked_dirty_refuse(self):self.test_exact_untrackedoutput_allowed_but_foreign_or_tracked_dirty_refuse();DETAILS['actual_Git_dirty_refusals']=True
    def test_wildcard_unknown_native_snapshot_and_boolean_CPU_capability_refuse(self):
        self.test_wildcard_alias_parent_or_unknown_snapshot_role_refuse();self.test_Q_rejects_unregistered_native_C_H_even_correct_hashleaf()
        with self.assertRaises(ValueError):T.J._CPU_join(self.root,packets=self.packets,source_files=self.sources,implementation_commit=self.impl,expected_head=self.head,git_output_exceptions={},_capability=True)
        self.assertEqual(T.Q.B.RESOURCES['whole_scope_seconds'],600)
if __name__=='__main__':
    names=('test_actual_two_commit214_absent_future_leaves_and_native2_policy','test_source_packet_escape_and_registered_native_snapshot_rebind_refuse','test_actual_untracked_registered_leaf_allowed_foreign_tracked_dirty_refuse','test_wildcard_unknown_native_snapshot_and_boolean_CPU_capability_refuse');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);after=pins();receipt={'kind':'INDEPENDENT_EXACT_IMPORT_GIT_EXCEPTION_POLICY001_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_source_pins_unchanged':BEFORE==after,'details':DETAILS,'actual_real_temporary_Git_implementation_and_dispatch_commits':True,'actual_qualifier_native2_exception_function_executed':True,'full_generic_qualifier_overlay_positive_not_repeated':True,'actual_native_process_or_framework_called':False,'native_context_or_authority_issued':False,'current_repository_registration_or_runtime_qualified':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_EXCEPTION_POLICY001.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
