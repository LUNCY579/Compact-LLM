"""Pure paid-clock/trap controls; no Timer/OS/framework/tokenizer/model work."""
import copy
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_entry_metadata_prepayment as p
from experiments.skill_acquisition_native_v1 import ce_checkpoint_entry_provider as entry
from .test_ce_checkpoint_entry_provider import fixture


class Tests(unittest.TestCase):
    def setUp(self):
        tmp=tempfile.TemporaryDirectory(prefix='entry-preparation-pure-');self.addCleanup(tmp.cleanup)
        self.configuration, _, _, _, old=fixture(Path(tmp.name))
        spec=copy.deepcopy(old.spec)
        spec['owned_directory']=str(Path(tmp.name)/'preparation-ledger')
        spec['operation_profiles']['entry_preparation']=dict(spec['setup_bounds'],wall_s_upper=1800)
        self.configuration['service_spec']['bootstrap']=spec
        self.ledger=p.services.BootstrapLedger(spec,old.request_sha256,synthetic_fixture=True,clock=lambda:0.0)
        self.before=set(sys.modules)&{'torch','numpy','transformers','safetensors'}
        self.addCleanup(lambda:self.assertEqual(self.before,set(sys.modules)&{'torch','numpy','transformers','safetensors'}))

    def run_fixture(self,body,values):
        times=iter(values)
        with (patch.object(p.services.threading,'Timer',side_effect=AssertionError('fixture Timer')),
              patch.object(p.services.os,'_exit',side_effect=AssertionError('fixture OS exit'))):
            return p.fixture_preparation(self.ledger,self.configuration,body,clock=lambda:next(times))

    def test_native_denies_before_configuration_timer_or_body(self):
        class Trap:
            def __getattribute__(self,key):raise AssertionError('pre-admission traversal')
        with patch.object(p.services.threading,'Timer',side_effect=AssertionError('Timer')):
            with self.assertRaisesRegex(ValueError,'services owner'):p.run_preparation(Trap(),Trap(),Trap())

    def test_reservation_precedes_body_and_success_retains_full_profile(self):
        def body():
            op=list(self.ledger.operations.values())[-1]
            self.assertEqual((op['kind'],op['phase']),('entry_preparation','started'))
            self.assertEqual(op['profile']['wall_s_upper'],1800)
            return 'metadata only'
        self.assertEqual(self.run_fixture(body,[0.0,1.0,2.0,3.0]),'metadata only')
        op=list(self.ledger.operations.values())[-1]
        self.assertEqual(op['phase'],'returned')
        self.assertGreaterEqual(self.ledger.counters['wall_seconds'],1800)

    def test_deadline_includes_returned_append_and_closes_owned_results(self):
        provider=entry.CheckpointEntryProvider();closed=[];failed=[];now=[0.0]
        handles=[object.__new__(entry.worker.WinWorkerHandle) for _ in range(3)]
        def prepared(*args):
            now[0]=1799.0
            return dict(zip(('prefix_handle','parent_handle','worker_handle'),handles))
        provider._open_after_payment=prepared
        actual=self.ledger._event
        def append(kind,payload,charges):
            row=actual(kind,payload,charges)
            if kind=='operation' and payload['phase']=='returned':now[0]=1801.0
            return row
        self.ledger._event=append
        def payment_guard(runtime,configuration,body):
            return p.fixture_preparation(self.ledger,self.configuration,body,clock=lambda:now[0])
        payment=type('FixturePayment',(),{'status':'RESERVED','fail':lambda self,error:failed.append(error)})()
        with (patch.object(p,'run_preparation',side_effect=payment_guard),
              patch.object(p.services.threading,'Timer',side_effect=AssertionError('fixture Timer')),
              patch.object(p.services.os,'_exit',side_effect=AssertionError('fixture OS exit')),
              patch.object(entry.worker.WinWorkerHandle,'close',lambda handle:closed.append(handle))):
            with self.assertRaisesRegex(ValueError,'deadline after returned append') as caught:
                provider._guarded_preparation({},None,payment,'no path access')
        self.assertEqual(closed,handles);self.assertEqual(failed,[caught.exception])
        self.assertTrue(self.ledger.fatal)
        self.assertEqual(list(self.ledger.operations.values())[-1]['phase'],'returned')
        phases=[row['payload']['phase'] for row in self.ledger.events if row['kind']=='operation']
        self.assertEqual(phases,['started','returned'])
        self.assertFalse(hasattr(caught.exception,'preparation_accounting_failure'))
        self.assertGreaterEqual(self.ledger.counters['wall_seconds'],1800)
        with self.assertRaisesRegex(ValueError,'terminal/fatal'):
            self.ledger.operation('entry_preparation','started','late-retry',
                **self.ledger.spec['operation_profiles']['entry_preparation'])

    def assert_post_append_clock_denied(self,appended):
        with self.assertRaisesRegex(ValueError,'deadline after returned append'):
            self.run_fixture(lambda:'never accepted',[0.0,1.0,2.0,appended])
        self.assertTrue(self.ledger.fatal)
        phases=[row['payload']['phase'] for row in self.ledger.events if row['kind']=='operation']
        self.assertEqual(phases,['started','returned'])

    def test_nonfinite_post_append_clock_is_sticky_without_duplicate_terminal_phase(self):
        self.assert_post_append_clock_denied(float('nan'))

    def test_regressed_post_append_clock_is_sticky_without_duplicate_terminal_phase(self):
        self.assert_post_append_clock_denied(1.0)

    def test_post_append_clock_at_deadline_can_return(self):
        self.assertEqual(self.run_fixture(lambda:'bounded metadata',[0.0,1.0,1799.0,1800.0]),'bounded metadata')
        self.assertFalse(self.ledger.fatal)
        self.assertEqual(list(self.ledger.operations.values())[-1]['phase'],'returned')

    def test_expired_after_prepaid_start_denies_before_body_retains_cost(self):
        called=[]
        with self.assertRaisesRegex(ValueError,'deadline before body'):
            self.run_fixture(lambda:called.append(True),[0.0,1800.0,1800.0])
        self.assertEqual(called,[])
        self.assertEqual(list(self.ledger.operations.values())[-1]['phase'],'failed')
        self.assertGreaterEqual(self.ledger.counters['wall_seconds'],1800)

    def test_late_body_cannot_return_and_keeps_pending_failed_accounting(self):
        with self.assertRaisesRegex(ValueError,'physical deadline exceeded') as caught:
            self.run_fixture(lambda:'never accepted',[0.0,1.0,1801.0,1801.0])
        self.assertTrue(self.ledger.fatal)
        self.assertEqual(list(self.ledger.operations.values())[-1]['phase'],'started')
        self.assertTrue(hasattr(caught.exception,'preparation_accounting_failure'))

    def test_body_error_survives_failed_observer_and_full_cost_remains(self):
        original=RuntimeError('metadata body failed')
        def body():raise original
        actual=self.ledger.operation
        def observe(kind,phase,*args,**kw):
            if phase=='failed':raise OSError('failure append denied')
            return actual(kind,phase,*args,**kw)
        self.ledger.operation=observe
        with self.assertRaises(RuntimeError) as caught:self.run_fixture(body,[0.0,1.0,2.0])
        self.assertIs(caught.exception,original)
        self.assertEqual(caught.exception.preparation_accounting_failure['type'],'OSError')
        self.assertEqual(list(self.ledger.operations.values())[-1]['phase'],'started')

    def test_wrong_frozen_profile_denies_before_body(self):
        self.ledger.spec['operation_profiles']['entry_preparation']['wall_s_upper']=1801
        self.configuration['service_spec']['bootstrap']['operation_profiles']['entry_preparation']['wall_s_upper']=1801
        with self.assertRaisesRegex(ValueError,'fixed1800'):
            self.run_fixture(lambda:(_ for _ in ()).throw(AssertionError('body')),[0.0])

    def test_provider_closes_owned_results_if_post_body_guard_rejects(self):
        provider=entry.CheckpointEntryProvider();closed=[];failed=[]
        parent=object.__new__(entry.worker.WinWorkerHandle)
        own=object.__new__(entry.worker.WinWorkerHandle)
        provider._open_after_payment=lambda *args:{'parent_handle':parent,'worker_handle':own,'prefix_handle':None}
        original=ValueError('post-body preparation deadline')
        def reject(runtime,configuration,body):body();raise original
        payment=type('FixturePayment',(),{'status':'RESERVED','fail':lambda self,error:failed.append(error)})()
        with patch.object(p,'run_preparation',side_effect=reject), \
             patch.object(entry.worker.WinWorkerHandle,'close',lambda handle:closed.append(handle)):
            with self.assertRaises(ValueError) as caught:
                provider._guarded_preparation({},None,payment,'no path access')
        self.assertIs(caught.exception,original)
        self.assertEqual(closed,[parent,own]);self.assertEqual(failed,[original])
