"""Synthetic engine publication controls; no framework or native authority."""
from copy import deepcopy
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import e1_training_engine_v2 as e
from experiments.post_training_mimo_centralized_v1.scripts.e1_training_accounting import E1CallAccounting
from experiments.post_training_mimo_centralized_v1.tests import test_e1_training_engine as legacy


class Publisher(legacy.AnalyticBackend):
    def __init__(self, fail_step=None, stochastic=False):
        super().__init__(stochastic=stochastic);self.fail_step=fail_step;self.published=[]
    def publish_checkpoint(self,envelope):
        step=envelope['cursor']['completed_optimizer_steps']
        if self.fail_step==step:raise OSError('synthetic metadata publication failure')
        self.published.append(deepcopy(envelope))
        return {'synthetic_publication':True,'completed_step':step,'save_call_id':envelope['save_call_id']}


class PublicationControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy.EngineControls.setUpClass();cls.plan=legacy.EngineControls.plan;cls.rows=legacy.EngineControls.rows
    def get(self,key):return legacy.array(self.rows[key])
    def ledger(self,events=None):
        return E1CallAccounting(lambda kind,**fields:events.append((kind,fields)) if events is not None else None)
    def test_publisher_required_before_any_calls(self):
        backend=legacy.AnalyticBackend()
        with self.assertRaises(ValueError):e.run(self.plan,self.get,backend,self.ledger(),stop_after_update=0)
        self.assertEqual(backend.saves,[])
    def test_post_save_envelope_published_before_exposure(self):
        backend=Publisher();events=[]
        result=e.run(self.plan,self.get,backend,self.ledger(events),stop_after_update=0)
        self.assertEqual(len(backend.published),1)
        envelope=result['checkpoint_envelopes'][0]
        self.assertEqual(backend.published[0],envelope)
        self.assertEqual(envelope['cost_snapshot']['operations']['checkpoint_save']['returned'],1)
        self.assertEqual(result['checkpoint_publications'][0]['save_call_id'],envelope['save_call_id'])
        self.assertEqual(events[-1][0],'checkpoint_envelope_published')
    def test_failed_initial_publication_retains_returned_save(self):
        result=e.run(self.plan,self.get,Publisher(fail_step=0),self.ledger(),stop_after_update=0)
        self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertEqual(result['checkpoint_envelopes'],[])
        self.assertEqual(result['call_accounting']['operations']['checkpoint_save'],dict(attempted=1,returned=1,failed=0,pending=0))
    def test_later_publication_failure_keeps_completed_updates(self):
        result=e.run(self.plan,self.get,Publisher(fail_step=64),self.ledger(),stop_after_update=64)
        self.assertEqual(result['completed_optimizer_steps'],64)
        self.assertEqual(result['call_accounting']['operations']['checkpoint_save']['returned'],2)
        self.assertEqual(result['call_accounting']['operations']['optimizer_step']['returned'],64)
        self.assertEqual([x['cursor']['completed_optimizer_steps'] for x in result['checkpoint_envelopes']],[0])
        self.assertIs(result['automatic_resume_authorized'],False)
    def test_publication_event_failure_refuses_boundary(self):
        def sink(kind,**fields):
            if kind=='checkpoint_envelope_published':raise OSError('journal failure')
        backend=Publisher();result=e.run(self.plan,self.get,backend,E1CallAccounting(sink),stop_after_update=0)
        self.assertEqual(len(backend.published),1)
        self.assertEqual(result['checkpoint_envelopes'],[])
        self.assertEqual(result['call_accounting']['operations']['checkpoint_save']['returned'],1)
    def test_invalid_publication_receipt_refuses_boundary(self):
        class Bad(Publisher):
            def publish_checkpoint(self,envelope):return None
        result=e.run(self.plan,self.get,Bad(),self.ledger(),stop_after_update=0)
        self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertEqual(result['checkpoint_envelopes'],[])
    def test_versioned_checkpoint_not_original_schema(self):
        result=e.run(self.plan,self.get,Publisher(),self.ledger(),stop_after_update=0)
        self.assertEqual(result['checkpoint_envelopes'][0]['kind'],'E1_CONTROLLED_BOUNDARY_COMPONENT_CHECKPOINT_V2')
        self.assertIs(result['E1_authorized'],False)
        self.assertIs(result['native_resume_qualified'],False)
    def test_full_schedule_publishes_all_five_completed_boundaries(self):
        backend=Publisher();result=e.run(self.plan,self.get,backend,self.ledger())
        self.assertEqual(result['status'],'COMPLETE_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertEqual([x['cursor']['completed_optimizer_steps'] for x in backend.published],[0,64,128,192,256])
        self.assertEqual(len(result['checkpoint_publications']),5)
        self.assertEqual(result['call_accounting']['operations']['forward']['returned'],5120)
        self.assertEqual(result['call_accounting']['operations']['optimizer_step']['returned'],256)
    def test_published_controlled_resume_preserves_state_rng_and_costs(self):
        uninterrupted=Publisher(stochastic=True);whole=e.run(self.plan,self.get,uninterrupted,self.ledger())
        first=Publisher(stochastic=True);part=e.run(self.plan,self.get,first,self.ledger(),stop_after_update=64)
        saved=part['checkpoint_envelopes'][-1]
        restored=Publisher(stochastic=False)
        ledger=E1CallAccounting.from_snapshot(lambda *args,**kwargs:None,saved['cost_snapshot'])
        result=e.run(self.plan,self.get,restored,ledger,resume=saved,latest_terminal_costs=saved['cost_snapshot'])
        self.assertEqual(result['status'],'COMPLETE_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertEqual((restored.w,restored.m,restored.v,restored.t,restored.rng.getstate()),
                         (uninterrupted.w,uninterrupted.m,uninterrupted.v,uninterrupted.t,uninterrupted.rng.getstate()))
        self.assertEqual(result['call_accounting']['operations']['optimizer_step']['returned'],256)
    def test_backend_receives_explicit_frozen_component(self):
        seen=[]
        class Observe(Publisher):
            def forward_loss(self,row):
                seen.append((row['row_id'],row['component']))
                return super().forward_loss(row)
        result=e.run(self.plan,self.get,Observe(),self.ledger(),stop_after_update=64)
        self.assertEqual(result['status'],'STOPPED_AT_SAVED_COMPONENT_BOUNDARY')
        expected=[(key.split(':',1)[1],key.split(':',1)[0]) for step in self.plan['steps'][:64] for key in step['row_keys']]
        self.assertEqual(seen,expected)
    def test_wrong_component_refused_before_model_callback(self):
        def wrong(key):return dict(self.get(key),component='language_replay')
        result=e.run(self.plan,wrong,Publisher(),self.ledger())
        self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertEqual(result['call_accounting']['operations']['forward']['attempted'],0)
    def test_native_scope_requires_primitive_snapshot_before_calls(self):
        class Incomplete(Publisher):execution_scope='UNQUALIFIED_NATIVE_COMPONENT'
        backend=Incomplete()
        with self.assertRaises(ValueError):e.run(self.plan,self.get,backend,self.ledger(),stop_after_update=0)
        self.assertEqual(backend.saves,[])
    def test_native_primitive_failure_metadata_is_retained(self):
        class Observed(Publisher):
            execution_scope='UNQUALIFIED_NATIVE_COMPONENT'
            def primitive_operation_snapshot(self):return {'synthetic_witness_only':True,'optimizer_returned':1}
        result=e.run(self.plan,self.get,Observed(fail_step=0),self.ledger(),stop_after_update=0)
        self.assertEqual(result['backend_primitive_accounting']['optimizer_returned'],1)
        self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
    def test_failed_native_snapshot_refuses_success(self):
        class Broken(Publisher):
            execution_scope='UNQUALIFIED_NATIVE_COMPONENT'
            def primitive_operation_snapshot(self):raise RuntimeError('missing final witness')
        result=e.run(self.plan,self.get,Broken(),self.ledger(),stop_after_update=0)
        self.assertEqual(result['status'],'FAILED_ENGINE_COMPONENT_UNQUALIFIED')
        self.assertIn('missing final witness',result['backend_primitive_snapshot_error'])


if __name__=='__main__':unittest.main()
