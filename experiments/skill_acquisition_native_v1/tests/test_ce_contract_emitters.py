"""Candidate source-emitter mechanics only; no permits, graphs or native proof."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_operation_contracts as c
from experiments.skill_acquisition_native_v1 import runner, ce_ingestion as ingestion
from experiments.skill_acquisition_native_v1 import ce_native_backend as backend
from experiments.skill_acquisition_native_v1 import checkpoint, ce_checkpoint_lease as lease
from experiments.skill_acquisition_native_v1 import ce_checkpoint_local_issuer as local


class Tests(unittest.TestCase):
    def test_file_leaf_precedes_read_and_denied_start_does_no_payload_IO(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'small.bin';path.write_bytes(b'bounded metadata fixture')
            calls=[]
            def observe(kind,phase,**fields):calls.append((kind,phase,fields))
            result=runner.observed_file(path,observe)
            self.assertIs(calls[0][2]['wall_contract'],c.FILE_HASH_LEAF)
            self.assertEqual(result['sha256'],hashlib.sha256(path.read_bytes()).hexdigest())
            with patch.object(Path,'open',side_effect=AssertionError('payload read before admission')):
                with self.assertRaisesRegex(ValueError,'denied start'):
                    runner.observed_file(path,lambda *a,**k:(_ for _ in ()).throw(ValueError('denied start')))

    def test_same_kind_backend_frame_and_child_file_leaf_are_distinct(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'small.bin';path.write_bytes(b'metadata');calls=[]
            obj=object.__new__(backend.NativeCEBackend)
            obj.spec={'operation_bounds':{'input_io':{'read_bytes_upper':64,'write_bytes_upper':0,
                'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':64}}}
            obj.operation=lambda kind,phase,**fields:calls.append((kind,phase,fields))
            obj.gate=lambda stage:calls.append(('guard',stage,{}))
            result=obj._read_ref({'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            starts=[entry for entry in calls if entry[1]=='started']
            self.assertEqual(len(starts),2)
            self.assertIs(starts[0][2]['wall_contract'],c.BACKEND_REF_FRAME)
            self.assertIs(starts[1][2]['wall_contract'],c.FILE_HASH_LEAF)
            self.assertLess(calls.index(starts[0]),calls.index(starts[1]))
            self.assertEqual(result['bytes'],8)

    def test_teacher_and_prepared_row_emit_guarded_roles_without_native_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'teacher.jsonl';blob=b'{"id":"row","target_sha256":"'+b'b'*64+b'"}\n';path.write_bytes(blob)
            source=object.__new__(ingestion.VerifiedSources)
            source._index={'row':{'offset':0,'bytes':len(blob),'line_sha256':hashlib.sha256(blob).hexdigest()}}
            source._target_ref={'path':str(path),'bytes':len(blob)};events=[]
            operation=lambda kind,phase,**fields:events.append((kind,phase,fields))
            source._target('row',operation,lambda stage:events.append(('guard',stage,{})))
            self.assertIs(events[0][2]['wall_contract'],c.TEACHER_ROW)
            prepared=object.__new__(ingestion.PreparedRows)
            prepared.operation=operation;prepared.gate=lambda stage:None;prepared.tokenizer=None
            prepared.verified=SimpleNamespace(_target=lambda *a,**k:{'id':'row','target_sha256':'b'*64},
                _expected_census={'labelled_tokens':{'max':10},'prompt_tokens':{'max':10},'max_sequence_positions':20})
            prepared.renderer=lambda *a:{'id':'row','target_sha256':'b'*64,'render_sha256':'a'*64,
                'prompt_ids':[42],'completion_ids':[7,151645]}
            value=prepared._render('row')
            self.assertIs([e for e in events if e[0]=='tokenizer_render' and e[1]=='started'][0][2]['wall_contract'],c.PREPARED_ROW)
            self.assertEqual(value['completion_ids'],[7,151645]);self.assertNotIn('native_qualified',value)

    def test_checkpoint_hash_roles_preserve_builtin_extra_and_inclusive_wall_fields(self):
        obj=object.__new__(checkpoint.NativeCheckpoints);events=[]
        obj._emit=lambda kind,phase,oid,**fields:events.append((kind,phase,fields))
        cases=(('checkpoint_frozen_state',c.CP_FROZEN_HASH,1),('checkpoint_payload_file',c.CP_FILE_HASH,1),
            ('checkpoint_payload_data',c.CP_DATA_HASH,1),('checkpoint_streamed_publication',c.CP_PUBLICATION_HASH,3),
            ('checkpoint_restored_saved_and_live_tensor',c.CP_TENSOR_PAIR_HASH,2))
        for reason,token,attempts in cases:
            events.clear()
            with obj._hash_span(123,reason,attempts):pass
            self.assertIs(events[0][2]['wall_contract'],token);self.assertIs(events[1][2]['wall_contract'],token)
            self.assertEqual(events[0][2]['identity_hash_attempts_upper'],attempts-1)
            self.assertEqual(events[0][2]['identity_hash_bytes_upper'],0)
            self.assertEqual(events[1][2]['identity_hash_attempts_actual'],attempts)
            self.assertEqual(events[1][2]['identity_hash_bytes_actual'],123)
            self.assertIn('wall_seconds',events[1][2]);self.assertNotIn('wall_s',events[1][2])
        events.clear()
        with self.assertRaisesRegex(ValueError,'consumer failed'):
            with obj._hash_span(123,'checkpoint_streamed_publication',3):raise ValueError('consumer failed')
        self.assertEqual(events[-1][1],'failed');self.assertIs(events[-1][2]['wall_contract'],c.CP_PUBLICATION_HASH)
        self.assertFalse(events[-1][2]['identity_hash_actual_known'])

    def test_checkpoint_journal_and_observer_order_preserve_live_token_identity(self):
        obj=object.__new__(checkpoint.NativeCheckpoints);events=[]
        obj.ledger=SimpleNamespace(operation=lambda kind,phase,**fields:events.append(('observer',phase,fields)))
        def journal(kind,**fields):
            events.append(('journal',fields['phase'],json.loads(checkpoint.canonical(fields))))
            return {'sha256':'a'*64}
        obj._journal=journal
        obj._emit('fingerprint','started','op',wall_contract=c.CP_TENSOR_PAIR_HASH)
        obj._emit('fingerprint','returned','op',wall_contract=c.CP_TENSOR_PAIR_HASH,wall_seconds=.1)
        self.assertEqual([(e[0],e[1]) for e in events],[('observer','started'),('journal','started'),('journal','returned'),('observer','returned')])
        self.assertIs(events[0][2]['wall_contract'],c.CP_TENSOR_PAIR_HASH)
        self.assertEqual(events[1][2]['wall_contract'],str(c.CP_TENSOR_PAIR_HASH))

    def test_qualification_fresh_storage_cannot_price_rolling_as_fresh_and_legacy_remains(self):
        obj=object.__new__(checkpoint.NativeCheckpoints)
        obj.bindings={'role':'qualification_only','checkpoint_mode':'CE8_NATIVE_QUALIFICATION_CHECKPOINT_1'}
        obj.ledger=SimpleNamespace(spec={'operation_contracts':{'fixture_metadata_only':True}})
        role=obj._frame_contract('save');self.assertIs(role['wall_contract'],c.CP_SAVE_FRESH)
        obj._require_fresh_history([],role)
        with self.assertRaisesRegex(ValueError,'empty owned storage'):obj._require_fresh_history(['ancestor'],role)
        obj.ledger.spec={};obj._require_fresh_history(['ancestor'],role)
        obj.bindings={'role':'production'}
        self.assertIs(obj._frame_contract('save')['wall_contract'],c.OPAQUE_CHECKPOINT_SAVE)
        self.assertIs(obj._frame_contract('restore')['wall_contract'],c.OPAQUE_CHECKPOINT_RESTORE)

    def test_lease_parent_check_and_other_owned_actions_have_explicit_distinct_roles(self):
        obj=object.__new__(lease.SupervisorGpuLease);events=[]
        obj.spec={'operation_bounds':{'transfer':{'read_bytes_upper':64,'write_bytes_upper':0,
            'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':64,'wall_s_upper':5}}}
        obj.operation=lambda kind,phase,**fields:events.append((kind,phase,fields));obj.fatal=False
        with obj._span('check_parent_use'):pass
        self.assertIs(events[0][2]['wall_contract'],c.LEASE_PARENT_CHECK)
        with obj._span('suspend_parent'):pass
        self.assertIs([e for e in events if e[1]=='started'][-1][2]['wall_contract'],c.OPAQUE_LEASE_FRAME)
        with self.assertRaisesRegex(ValueError,'fixture failure'):
            with obj._span('check_parent_use'):raise ValueError('fixture failure')
        self.assertTrue(obj.fatal);self.assertEqual(events[-1][1],'failed')

    def test_fixed_metadata_helpers_keep_opaque_role_and_no_model_libraries(self):
        events=[];operation=lambda kind,phase,**fields:events.append((kind,phase,fields))
        result=local._paid(operation,{'read_bytes_upper':32,'write_bytes_upper':0,
            'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':32,'wall_s_upper':5},'input_io',lambda:7)
        self.assertEqual(result,7);self.assertIs(events[0][2]['wall_contract'],c.OPAQUE_INPUT_FRAME)
        self.assertIs(copy.deepcopy(c.CP_RESTORE),c.CP_RESTORE)
        self.assertEqual(set(sys.modules)&{'torch','numpy','transformers','safetensors'},set())

    def test_checkpoint_adapter_same_kind_source_roles_are_distinct(self):
        calls=[];obj=object.__new__(backend.NativeCEBackend)
        obj._admit=lambda *a:None;obj._ready=lambda:None;obj._journal=lambda *a:None
        obj.U=2;obj.model=obj.optimizer=None
        obj.spec={'operation_bounds':{'checkpoint_adapter_io':{'read_bytes_upper':64,'write_bytes_upper':64,
            'identity_hash_attempts_upper':1,'identity_hash_bytes_upper':64}}}
        obj.operation=lambda kind,phase,**fields:calls.append((kind,phase,fields));obj.gate=lambda stage:None
        progress={'C':2,'U':2};journal=SimpleNamespace(progress_record=lambda:progress)
        checkpoints=SimpleNamespace(attach_ledger=lambda *a:None,save=lambda *a:{'progress':progress},
            restore=lambda *a:{'progress':progress})
        obj.save(checkpoints,journal);obj.restore(checkpoints,journal)
        starts=[entry for entry in calls if entry[1]=='started']
        self.assertIs(starts[0][2]['wall_contract'],c.CP_SAVE_ADAPTER)
        self.assertIs(starts[1][2]['wall_contract'],c.CP_RESTORE_ADAPTER)
        self.assertEqual([entry[0] for entry in starts],['checkpoint_adapter_io','checkpoint_adapter_io'])


if __name__=='__main__':unittest.main()
