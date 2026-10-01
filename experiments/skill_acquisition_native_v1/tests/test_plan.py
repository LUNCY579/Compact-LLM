"""Synthetic whole-unit resource budgets only; no dataset/model/framework work."""
import copy
import unittest

from experiments.skill_acquisition_native_v1 import plan


def synthetic_plan(arm='R'):
    count=3000 if arm=='MT' else 1500
    boundaries=tuple(plan.Boundary(i,tuple(plan.Prompt('R' if p==0 or arm!='MT' else 'S',
                     'synthetic-row-%d-%d'%(i,p),tuple(range(8))) for p in range(2))) for i in range(count))
    return plan.Plan(arm,1301,True,boundaries,plan.digest({'synthetic_arm':arm}),{},'b'*64)


class StorageBudgetTests(unittest.TestCase):
    def setUp(self):
        self.planned=synthetic_plan()
        self.numel=220233216

    def io(self,W,planned=None):
        return {'schema':'S1_IO_BUDGET-1','measured':True,'plan_sha256':(planned or self.planned).sha256,
                'inventory':{'numel':self.numel},'caps':{'chunk_io_write_bytes':W},
                'synthetic_accounting_fixture':True}

    def test_no_unproved_64MiB_default_when_unit_IO_census_missing(self):
        caps=plan.derive_caps(self.planned,inventory_numel=self.numel)
        self.assertIsNone(caps['active_metadata_reserve_bytes'])
        self.assertIsNone(caps['active_storage_reserve_bytes'])
        self.assertIn('bound cumulative unit IO write ceiling for active metadata',caps['unresolved'])

    def test_owned_metadata_is_bounded_by_all_cumulative_writes(self):
        W=512*1024**2
        caps=plan.derive_caps(self.planned,inventory_numel=self.numel,unit_io_budget=self.io(W))
        self.assertEqual(caps['active_metadata_reserve_bytes'],W)
        self.assertEqual(caps['active_storage_reserve_bytes'],28*self.numel+W)
        basis=caps['active_metadata_reserve_basis']
        self.assertEqual(basis['unit_io_budget_preimage_sha256'],plan.digest(self.io(W)))
        self.assertIn('retired tensor-payload',basis['scope'])

    def test_64MiB_remains_minimum_but_never_replaces_larger_write_ceiling(self):
        caps=plan.derive_caps(self.planned,inventory_numel=self.numel,unit_io_budget=self.io(1))
        self.assertEqual(caps['active_metadata_reserve_bytes'],64*1024**2)
        self.assertEqual(caps['active_storage_reserve_bytes'],28*self.numel+64*1024**2)

    def test_wrong_plan_inventory_measurement_and_counter_type_refused(self):
        for change in ('plan','inventory','measured','counter'):
            value=self.io(100000000)
            if change=='plan':value['plan_sha256']='0'*64
            if change=='inventory':value['inventory']['numel']+=1
            if change=='measured':value['measured']=False
            if change=='counter':value['caps']['chunk_io_write_bytes']=True
            with self.subTest(change=change),self.assertRaises(ValueError):
                plan.derive_caps(self.planned,inventory_numel=self.numel,unit_io_budget=value)

    def test_atomic_and_MT_keep_dose_compute_and_resume_currency(self):
        atomic=plan.derive_caps(self.planned,inventory_numel=self.numel,unit_io_budget=self.io(100000000))
        mt_plan=synthetic_plan('MT')
        mt=plan.derive_caps(mt_plan,inventory_numel=self.numel,unit_io_budget=self.io(200000000,mt_plan))
        self.assertEqual((atomic['final_C'],mt['final_C']),(1500,3000))
        self.assertEqual((atomic['nominal_rollouts'],mt['nominal_rollouts']),(24000,48000))
        self.assertEqual((atomic['optimizer_started'],mt['optimizer_started']),(1875,3750))
        self.assertEqual((atomic['invocations_started'],mt['invocations_started']),(9,9))
        self.assertEqual(mt['active_metadata_reserve_bytes'],200000000)
        self.assertEqual(mt['active_storage_reserve_bytes'],28*self.numel+200000000)

    def test_claimed_smaller_metadata_number_cannot_launder_safe_bound(self):
        io=self.io(512*1024**2)
        io['claimed_metadata_only_upper']=64*1024**2
        caps=plan.derive_caps(self.planned,inventory_numel=self.numel,unit_io_budget=io)
        self.assertEqual(caps['active_metadata_reserve_bytes'],512*1024**2)
        self.assertEqual(caps['active_metadata_reserve_basis']['smaller_metadata_only_reserve_status'],
                         'UNQUALIFIED_WRITER_CENSUS_AND_ENFORCEMENT_REQUIRED')

    def test_bound_inline_metadata_census_tightens_W_and_carries_writer_envelope(self):
        io=self.io(512*1024**2)
        census={'schema':'S1_METADATA_CENSUS-1','plan_sha256':self.planned.sha256,
                'io_preimage_sha256':plan.digest(io),'fixed_bytes_upper':64*1024**2,
                'classes':{kind:{'writes_upper':10,'serialized_bytes_upper':4096} for kind in plan.METADATA_CLASSES}}
        io['metadata_census']=census
        caps=plan.derive_caps(self.planned,inventory_numel=self.numel,unit_io_budget=io,metadata_census=census)
        expected=64*1024**2+4*10*4096
        self.assertEqual(caps['active_metadata_reserve_bytes'],expected)
        self.assertEqual(caps['active_storage_reserve_bytes'],28*self.numel+expected)
        self.assertEqual(caps['metadata_write_envelope']['classes'],census['classes'])
        self.assertEqual(caps['active_metadata_reserve_basis']['smaller_metadata_only_reserve_status'],'ENFORCED_METADATA_CENSUS')
        wrong=copy.deepcopy(census);wrong['io_preimage_sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'preimage'):
            plan.derive_caps(self.planned,inventory_numel=self.numel,unit_io_budget=io,metadata_census=wrong)

    def test_metadata_census_class_type_population_not_unbounded_free_number(self):
        io=self.io(512*1024**2)
        census={'schema':'S1_METADATA_CENSUS-1','plan_sha256':self.planned.sha256,
                'io_preimage_sha256':plan.digest(io),'fixed_bytes_upper':64*1024**2,
                'classes':{kind:{'writes_upper':10,'serialized_bytes_upper':4096} for kind in plan.METADATA_CLASSES}}
        census['classes']['runner_receipt']['writes_upper']=False
        with self.assertRaisesRegex(ValueError,'finite count'):
            plan.derive_caps(self.planned,inventory_numel=self.numel,unit_io_budget=io,metadata_census=census)


if __name__=='__main__':unittest.main()
