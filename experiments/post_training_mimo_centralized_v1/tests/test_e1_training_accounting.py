"""Stdlib tests of independent E1 call budgets and interruption accounting."""
from copy import deepcopy
import unittest
from experiments.post_training_mimo_centralized_v1.scripts import e1_training_accounting as a


class AccountingControls(unittest.TestCase):
    def setUp(self):
        self.events=[]
        self.ledger=a.E1CallAccounting(lambda kind,**fields:self.events.append((kind,fields)))

    def test_separate_full_e1_optimizer_budget(self):
        for update in range(256):
            self.ledger.call('optimizer_step',lambda:None,stage=f'update{update+1}')
        with self.assertRaises(ValueError): self.ledger.call('optimizer_step',lambda:None,stage='over')
        self.assertEqual(self.ledger.snapshot()['operations']['optimizer_step'],dict(attempted=256,returned=256,failed=0,pending=0))

    def test_attempt_before_actual_function_and_return_after(self):
        def op():
            self.assertEqual(self.events[-1][0],'operation_attempt')
            self.assertEqual(self.ledger.snapshot()['operations']['forward']['pending'],1)
            return 42
        self.assertEqual(self.ledger.call('forward',op,stage='row',input_positions=10,supervised_positions=2),42)
        self.assertEqual([kind for kind,_ in self.events],['operation_attempt','operation_returned'])
        self.assertEqual(self.ledger.snapshot()['positions'],dict(input_positions_attempted=10,input_positions_returned=10,supervised_positions_attempted=2,supervised_positions_returned=2))

    def test_failed_forward_keeps_attempted_positions(self):
        def fail(): raise RuntimeError('synthetic forward failure')
        with self.assertRaises(RuntimeError): self.ledger.call('forward',fail,stage='row',input_positions=7,supervised_positions=3)
        snap=self.ledger.snapshot()
        self.assertEqual(snap['operations']['forward'],dict(attempted=1,returned=0,failed=1,pending=0))
        self.assertEqual(snap['positions']['input_positions_attempted'],7)
        self.assertEqual(snap['positions']['input_positions_returned'],0)
        self.assertEqual(self.events[-1][0],'operation_failed')
        restored=a.E1CallAccounting.from_snapshot(lambda *args,**kw:None,snap)
        self.assertEqual(restored.snapshot(),snap)

    def test_nested_loader_copies_and_lifo(self):
        outer=self.ledger.begin('checkpoint_load',stage='load')
        inner=self.ledger.begin('tensor_copy',stage='copy')
        with self.assertRaises(ValueError): self.ledger.returned(outer)
        self.assertEqual(len(self.ledger.snapshot()['pending_calls']),2)
        self.ledger.returned(inner);self.ledger.returned(outer)
        self.assertEqual(self.ledger.snapshot()['pending_calls'],[])

    def test_pending_intent_cannot_resume(self):
        self.ledger.begin('backward',stage='partial')
        with self.assertRaises(ValueError): a.E1CallAccounting.from_snapshot(lambda *args,**kw:None,self.ledger.snapshot())

    def test_resume_preserves_used_budget(self):
        for update in range(255): self.ledger.call('optimizer_step',lambda:None,stage=f'u{update+1}')
        restored=a.E1CallAccounting.from_snapshot(lambda *args,**kw:None,self.ledger.snapshot())
        restored.call('optimizer_step',lambda:None,stage='u256')
        with self.assertRaises(ValueError):restored.call('optimizer_step',lambda:None,stage='u257')
        self.assertEqual(restored.snapshot()['last_call_id'],256)

    def test_independent_tensor_and_checkpoint_caps(self):
        for name in ('tensor_copy','checkpoint_save','checkpoint_reload'):
            for index in range(a.CEILINGS[name]):self.ledger.call(name,lambda:None,stage=f'{name}{index}')
            with self.assertRaises(ValueError):self.ledger.call(name,lambda:None,stage='over')

    def test_illegal_operation_metadata(self):
        for op,stage,inputs,supervised in [('other','x',0,0),('forward','x',True,1),('forward','x',513,1),('forward','x',10,0),('forward','x',10,10),('backward','x',3,1),('optimizer_step','bad stage',0,0)]:
            with self.assertRaises(ValueError):self.ledger.begin(op,stage=stage,input_positions=inputs,supervised_positions=supervised)
        self.assertEqual(self.ledger.snapshot()['last_call_id'],0)

    def test_snapshot_copies_are_isolated(self):
        snap=self.ledger.snapshot();snap['ceilings']['optimizer_step']=999;snap['operations']['forward']['attempted']=9
        self.assertEqual(self.ledger.snapshot()['ceilings']['optimizer_step'],256)
        self.assertEqual(self.ledger.snapshot()['operations']['forward']['attempted'],0)

    def test_restore_refuses_scope_or_cap_changes(self):
        for key,value in [('scope','other'),('sequence_ceiling',1024),('E1_authorized',True),('native_launch_authorized',True)]:
            snap=self.ledger.snapshot();snap[key]=value
            with self.assertRaises(ValueError):a.E1CallAccounting.from_snapshot(lambda *args,**kw:None,snap)
        snap=self.ledger.snapshot();snap['ceilings']['model_construct']=True
        with self.assertRaises(ValueError):a.E1CallAccounting.from_snapshot(lambda *args,**kw:None,snap)

    def test_restore_refuses_counter_and_id_drift(self):
        self.ledger.call('optimizer_step',lambda:None,stage='u1')
        for change in ('sum','id','dtype'):
            snap=self.ledger.snapshot()
            if change=='sum':snap['operations']['optimizer_step']['returned']=0
            if change=='id':snap['last_call_id']=0
            if change=='dtype':snap['operations']['optimizer_step']['attempted']=True
            with self.assertRaises(ValueError):a.E1CallAccounting.from_snapshot(lambda *args,**kw:None,snap)

    def test_failed_forward_position_delta_cannot_exceed_shape(self):
        self.ledger.call('forward',lambda:None,stage='ok',input_positions=2,supervised_positions=1)
        def fail():raise ValueError('failure')
        with self.assertRaises(ValueError):self.ledger.call('forward',fail,stage='bad',input_positions=512,supervised_positions=1)
        snap=self.ledger.snapshot();snap['positions']['input_positions_attempted']=600
        with self.assertRaises(ValueError):a.E1CallAccounting.from_snapshot(lambda *args,**kw:None,snap)

    def test_return_publication_failure_does_not_reclassify_work(self):
        def sink(kind,**fields):
            if kind=='operation_returned':raise OSError('disk unavailable')
        ledger=a.E1CallAccounting(sink)
        with self.assertRaises(OSError):ledger.call('optimizer_step',lambda:None,stage='returned')
        self.assertEqual(ledger.snapshot()['operations']['optimizer_step'],dict(attempted=1,returned=1,failed=0,pending=0))

    def test_attempt_publication_failure_leaves_pending_without_execution(self):
        calls=[]
        def sink(kind,**fields):raise OSError('disk unavailable')
        ledger=a.E1CallAccounting(sink)
        with self.assertRaises(OSError):ledger.call('optimizer_step',lambda:calls.append(1),stage='intent')
        self.assertEqual(calls,[])
        self.assertEqual(ledger.snapshot()['operations']['optimizer_step'],dict(attempted=1,returned=0,failed=0,pending=1))

    def test_non_callable_never_attempted(self):
        with self.assertRaises(ValueError):self.ledger.call('backward',None,stage='bad')
        self.assertEqual(self.ledger.snapshot()['last_call_id'],0)

    def test_accounting_never_grants_native_launch(self):
        self.assertIs(self.ledger.snapshot()['E1_authorized'],False)
        self.assertIs(self.ledger.snapshot()['native_launch_authorized'],False)


if __name__=='__main__':unittest.main()
