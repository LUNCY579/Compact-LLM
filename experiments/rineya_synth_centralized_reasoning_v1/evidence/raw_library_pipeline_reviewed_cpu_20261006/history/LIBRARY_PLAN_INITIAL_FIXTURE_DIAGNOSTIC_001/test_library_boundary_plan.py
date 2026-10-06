"""External boundary plan schema and actual resealed runtime/review joins."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import sys,tempfile,unittest
from outer_context_fixture import OuterContextFixture,HERE
import test_library_boundary_manifest as M

class LibraryBoundaryPlanControls(unittest.TestCase):
    def setUp(self):
        self.m=M.LibraryBoundaryManifestControls();self.m.setUp();self.addCleanup(self.m.doCleanups)
        self.f=self.m.fixture;self.C=self.m.C;self.ext=self.m.extension
        self.plan=[{'id':1,'parent_id':None,'name':'framework_import'},
                   {'id':2,'parent_id':1,'name':'extension:'+self.ext+':create'},
                   {'id':3,'parent_id':1,'name':'extension:'+self.ext+':exec'}]
    def install_plan(self,plan=None,*,review=True):
        self.f.runtime['library_boundary_plan']=deepcopy(self.plan if plan is None else plan)
        if review:self.f.review['library_boundary_plan_sha256']=self.C.digest(self.f.runtime['library_boundary_plan'])
        self.m.reseal()
    def context(self):return self.f.context()
    def test_actual_runtime_review_anchor_plan_join_keeps_CPU_identity(self):
        self.install_plan();value=self.C.context_value(self.context())
        self.assertEqual(value['runtime']['library_boundary_plan'],self.plan)
        self.assertTrue(value['fixture_only']);self.assertFalse(value['native_authority_issued'])
        self.assertEqual(self.C.library_boundary_plan(value['runtime']),self.plan)
    def test_plan_absence_preserves_existing_schema_without_authority(self):
        self.assertIsNone(self.C.library_boundary_plan(self.f.runtime))
        value=self.C.context_value(self.context());self.assertNotIn('library_boundary_plan',value['runtime'])
        self.assertTrue(value['fixture_only'])
    def test_resealed_plan_change_without_independent_review_digest_refused(self):
        self.install_plan();self.f.runtime['library_boundary_plan'][2]['id']=4;self.m.reseal()
        with self.assertRaisesRegex(ValueError,'boundary plan review join'):self.context()
    def test_resealed_missing_review_field_refused(self):
        self.install_plan(review=False)
        with self.assertRaisesRegex(ValueError,'independent qualified review schema'):self.context()
    def reject_plan(self,plan,message):
        runtime=deepcopy(self.f.runtime);runtime['library_boundary_plan']=plan
        with self.assertRaisesRegex(ValueError,message):self.C.library_boundary_plan(runtime)
    def test_untyped_duplicate_and_nonpositive_ids_refused(self):
        for changed in (True,0,-1,2**64):
            plan=deepcopy(self.plan);plan[0]['id']=changed
            self.reject_plan(plan,'typed selected boundary identity')
        plan=deepcopy(self.plan);plan[2]['id']=2;self.reject_plan(plan,'typed selected boundary identity')
    def test_unknown_policy_operation_and_name_refused(self):
        for name in ('extension:'+self.ext+':load','extension:C:/UNKNOWN.dll:create','nope'):
            plan=deepcopy(self.plan);plan[1]['name']=name;self.reject_plan(plan,'selected extension operation|framework or extension')
    def test_bool_missing_and_future_parent_refused(self):
        for parent in (True,99,None):
            plan=deepcopy(self.plan);plan[1]['parent_id']=parent
            self.reject_plan(plan,'typed selected boundary parent|open earlier parent')
    def test_closed_parent_cannot_be_reopened_by_later_subtree(self):
        plan=deepcopy(self.plan)+[{'id':4,'parent_id':2,'name':'extension:'+self.ext+':exec'}]
        self.reject_plan(plan,'open earlier parent')
    def test_one_first_framework_and_exact_fieldset_required(self):
        self.reject_plan([self.plan[0],{'id':2,'parent_id':None,'name':'framework_import'}],'one first framework')
        plan=deepcopy(self.plan);plan[1]['declared_complete']=True;self.reject_plan(plan,'typed selected boundary identity')
    def test_nested_extension_plan_is_valid_and_detached(self):
        plan=deepcopy(self.plan);plan[2]['parent_id']=2
        runtime=deepcopy(self.f.runtime);runtime['library_boundary_plan']=plan
        result=self.C.library_boundary_plan(runtime);self.assertEqual(result,plan)
        result[0]['id']=99;self.assertEqual(runtime['library_boundary_plan'][0]['id'],1)
    def test_missing_enforced_policy_refused(self):
        runtime=deepcopy(self.f.runtime);runtime['library_boundary_plan']=deepcopy(self.plan)
        del runtime['library_search_policy']['enforced_boundary']
        with self.assertRaisesRegex(ValueError,'requires enforced library policy'):self.C.library_boundary_plan(runtime)
    def test_empty_and_oversized_plan_refused_before_rows(self):
        self.reject_plan([],'finite nonempty');self.reject_plan([object()]*2049,'finite nonempty')

if __name__=='__main__':unittest.main()
