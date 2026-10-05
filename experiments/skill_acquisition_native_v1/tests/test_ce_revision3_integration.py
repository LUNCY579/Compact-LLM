"""Actual ledger-backed source callbacks and new-policy seal consistency only."""
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import runner
from experiments.skill_acquisition_native_v1 import ce_execution_verifier as verifier
from . import test_ce_rejected_return as controls


class Revision3Integration(unittest.TestCase):
    def setUp(self):
        self.owner=controls.PrepaidRejectedReturn();self.owner.setUp()
        self.addCleanup(self.owner.doCleanups)

    def test_source_hash_returns_through_real_contract_ledger(self):
        _,ledger=self.owner.bootstrap('file-ledger')
        path=self.owner.root/'code.bin';path.write_bytes(b'actual source fixture')
        result=runner.observed_file(path,ledger.operation)
        self.assertEqual(result['bytes'],21)
        self.assertFalse(ledger.fatal)
        self.assertEqual(list(ledger.operations.values())[-1]['phase'],'returned')

    def test_source_hash_read_failure_retains_failed_same_role(self):
        _,ledger=self.owner.bootstrap('failed-file-ledger')
        path=self.owner.root/'code.bin';path.write_bytes(b'metadata')
        original_open=Path.open
        def open_control(current,*args,**kwargs):
            if current==path and args and args[0]=='rb':raise OSError('fixture read failed')
            return original_open(current,*args,**kwargs)
        with patch.object(Path,'open',open_control):
            with self.assertRaises(OSError):runner.observed_file(path,ledger.operation)
        self.assertFalse(ledger.fatal)
        self.assertEqual(list(ledger.operations.values())[-1]['phase'],'failed')

    def test_payload_header_failure_retains_failed_original_role(self):
        _,ledger=self.owner.bootstrap('failed-payload-ledger')
        weights=self.owner.root/'base.bin';warm=self.owner.root/'warm.bin'
        weights.write_bytes(b'x'*32);warm.write_bytes(b'x'*32)
        class Reader:
            start=8
            header={'slot':{'dtype':'INVALID','data_offsets':[0,24]}}
            def __init__(self,path):pass
            def close(self):pass
        with patch.object(runner.update,'_TensorFile',Reader):
            with self.assertRaisesRegex(ValueError,'dtype'):
                runner.model_identity_lower_bound(weights,warm,ledger.operation)
        self.assertFalse(ledger.fatal)
        self.assertEqual(list(ledger.operations.values())[-1]['phase'],'failed')

    def test_pinned_helper_returns_and_fails_with_same_source_role(self):
        for failed in (False,True):
            _,ledger=self.owner.bootstrap('helper-'+str(failed))
            path=self.owner.root/('helper-'+str(failed)+'.py');path.write_bytes(b'# source')
            def load():
                if failed:raise LookupError('fixture helper failure')
                return SimpleNamespace(__file__=str(path))
            if failed:
                with self.assertRaises(LookupError):runner.pinned_helper(path,load,ledger.operation)
            else:runner.pinned_helper(path,load,ledger.operation)
            self.assertFalse(ledger.fatal)
            self.assertEqual(list(ledger.operations.values())[-1]['phase'],'failed' if failed else 'returned')

    def test_payload_header_callbacks_return_through_real_ledger(self):
        _,ledger=self.owner.bootstrap('payload-ledger')
        weights=self.owner.root/'base.bin';warm=self.owner.root/'warm.bin'
        weights.write_bytes(b'x'*32);warm.write_bytes(b'x'*32)
        class Reader:
            def __init__(self,path):
                self.start=8
                self.header={'slot':{'dtype':'BF16' if Path(path)==weights else 'F32','data_offsets':[0,24]}}
            def close(self):pass
        with patch.object(runner.update,'_TensorFile',Reader):
            value=runner.model_identity_lower_bound(weights,warm,ledger.operation)
        self.assertEqual(value['identity_hash_attempts_lower'],5)
        self.assertFalse(ledger.fatal)
        self.assertTrue(all(op['phase']=='returned' for key,op in ledger.operations.items() if key!='leaf'))

    def test_new_policy_snapshot_seal_uses_published_fields_and_frozen_policy(self):
        context,receipt,ledger=self.owner.qualification('seal')
        state=self.owner.shadow(context,receipt,ledger)
        # The independent file parser installs the first retained worker proof
        # before applying its row. The narrow shadow helper only applies rows.
        state.proof=copy.deepcopy(next(row['payload']['proof'] for row in ledger.events if row['kind']=='worker'))
        observed=ledger.snapshot()
        self.assertNotIn('rejection_policy',observed)
        complete,pending=verifier._seal(observed,state)
        self.assertFalse(complete);self.assertTrue(pending)
        changed=copy.deepcopy(observed);changed['rejection_used']=True
        with self.assertRaisesRegex(ValueError,'rejection evidence'):
            verifier._seal(changed,state)

if __name__=='__main__':unittest.main()
