"""Pure refusal/ownership integration checks; no native processes or ML."""
import copy
from contextlib import nullcontext
from pathlib import Path
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_rejected_return as rejected
from experiments.skill_acquisition_native_v1 import ce_journal_limits as bounds
from experiments.skill_acquisition_native_v1 import ce_native_services as services
from experiments.skill_acquisition_native_v1 import ce_process_launch as launch
from experiments.skill_acquisition_native_v1 import ce_checkpoint_lease as lease
from experiments.skill_acquisition_native_v1 import ce_entry_factories as factories
from experiments.skill_acquisition_native_v1 import ce_operation_contracts as contracts
from . import test_ce_rejected_return as rejection_controls
from .test_ce_service_bootstrap_verifier import Clock


class Revision2Guards(unittest.TestCase):
    def test_entry_outer_metadata_stays_on_original_bootstrap_after_q_attachment(self):
        events=[]
        bootstrap=SimpleNamespace(operation=lambda kind,phase,**fields:events.append((kind,phase,fields)))
        dispatcher=services.StableDispatcher(bootstrap)
        dispatcher.ledger=SimpleNamespace(operation=lambda *args,**kwargs:
            self.fail('entry metadata incorrectly borrowed Q profile'))
        value=object.__new__(factories._RowsMetadataAuthority)
        object.__setattr__(value,'operation',dispatcher)
        object.__setattr__(value,'gate',lambda stage:events.append(('gate',stage,{})))
        object.__setattr__(value,'profile',dict(read_bytes_upper=512*1024**2,write_bytes_upper=64*1024**2,
            identity_hash_attempts_upper=20000,identity_hash_bytes_upper=512*1024**2,wall_s_upper=360))
        factories._reserve_entry_outer(value,'ingestion_qualification_rows',{})
        physical=[item for item in events if item[0]=='input_io']
        self.assertEqual([item[1] for item in physical],['started','returned'])
        self.assertTrue(all(item[2]['wall_contract'] is contracts.OPAQUE_INPUT_FRAME for item in physical))

    def test_untrusted_string_subclass_conversion_is_never_called(self):
        class Derived(str):
            def __str__(self): raise AssertionError('untrusted string conversion ran')
        with self.assertRaisesRegex(ValueError, 'unsupported'):
            rejected.encode_fields({'value':Derived('original')}, rejected.default_policy())

    def test_del_escaping_refuses_before_typed_tree_exceeds_ticket(self):
        with self.assertRaisesRegex(ValueError, 'encoded byte'):
            rejected.encode_fields({'value':'\x7f'*11000}, rejected.default_policy())

    def test_bootstrap_byte_limit_halts_before_append_without_refunding_ticket(self):
        owner=rejection_controls.PrepaidRejectedReturn();owner.setUp();self.addCleanup(owner.doCleanups)
        spec,_=owner.bootstrap('basis');spec=copy.deepcopy(spec)
        spec['owned_directory']=str(owner.root/'bounded')
        spec['journal_limits']=bounds.default_limits('bootstrap')
        ledger=services.BootstrapLedger(spec,'a'*64,synthetic_fixture=True,clock=Clock())
        before=copy.deepcopy(ledger.counters);count=len(ledger.events)
        ledger._known_size=spec['journal_limits']['bytes_upper']-65536
        with patch.object(Path, 'open', side_effect=AssertionError('append before size guard')):
            with self.assertRaisesRegex(ValueError, 'producer'):
                ledger._event('control',{'reason':'fixture'}, {})
        self.assertTrue(ledger.fatal);self.assertEqual(ledger.counters,before)
        self.assertEqual(len(ledger.events),count);self.assertIsNotNone(ledger.rejection_ticket)
        with self.assertRaisesRegex(ValueError, 'fatal'):
            ledger.operation('input_io','started','new')

    def test_reserved_failure_tail_and_absolute_terminal_bound(self):
        value=bounds.default_limits('bootstrap');size=value['bytes_upper']-65536
        with self.assertRaisesRegex(ValueError,'producer'):
            bounds.check_append(value,rows=0,size=size,line_bytes=1)
        bounds.check_append(value,rows=0,size=size,line_bytes=65536,terminal=True)
        with self.assertRaisesRegex(ValueError,'producer'):
            bounds.check_append(value,rows=0,size=size+1,line_bytes=65536,terminal=True)
        self.assertIsNone(bounds.limits_from_spec({},'bootstrap'))

    def test_multiple_redirector_candidates_refuse_before_os_handle_work(self):
        current=object.__new__(launch.OwnedProcess);calls=[]
        class Budget:
            def run(self,kind,work): calls.append(kind);return work()
        current.os=SimpleNamespace();current.budget=Budget()
        current.spec={'handshake_seconds':1};current.deadline=time.perf_counter()+2
        with patch.object(launch,'_child_pids',return_value=[41,42]):
            with self.assertRaisesRegex(ValueError,'one direct worker candidate'):
                current._find_worker({'pid':40})
        self.assertEqual(calls,['process_launch_child_census'])

    def test_delegated_metadata_does_not_grant_parent_model_use(self):
        current=object.__new__(lease.SupervisorGpuLease);owned=object();events=[]
        proof={'os_identity':{'pid':41}}
        declaration={'child_os_identity':{'pid':41},'branch':'resumed'}
        current.state='CHILD';current.child=(owned,proof,declaration)
        current.backend=SimpleNamespace(model=None,optimizer=None)
        current._admit=lambda:None;current._span=lambda action:nullcontext({})
        current._parent=lambda:events.append('parent')
        current._child_proof=lambda target:copy.deepcopy(proof)
        current._lock=lambda workers:events.append(('lock',workers))
        current.assert_delegated_metadata(owned,copy.deepcopy(declaration))
        self.assertEqual(events,['parent',('lock',[41])])
        with self.assertRaisesRegex(ValueError,'parent GPU use'):
            current.assert_parent_use()
        current.backend.model=object()
        with self.assertRaisesRegex(ValueError,'released parent'):
            current.assert_delegated_metadata(owned,declaration)
        current.backend.model=None
        with self.assertRaisesRegex(ValueError,'exact child'):
            current.assert_delegated_metadata(object(),declaration)

if __name__=='__main__':unittest.main()
