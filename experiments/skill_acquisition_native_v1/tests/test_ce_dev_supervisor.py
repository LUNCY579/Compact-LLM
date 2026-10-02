"""Metadata-only source/lifecycle controls; no OS/process/framework/model work."""
import copy
import hashlib
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch,MagicMock

from experiments.skill_acquisition_native_v1 import ce_dev_supervisor as s
from experiments.skill_acquisition_native_v1.tests.test_ce_dev_runtime import context_fixture


def price():
    return dict(read_bytes_upper=512*s.raw.CHUNK,write_bytes_upper=16*s.raw.CHUNK,
        identity_hash_attempts_upper=10000,identity_hash_bytes_upper=1024*s.raw.CHUNK,wall_s_upper=19200)


def proof_fixture():
    identity={'schema':s.worker.IDENTITY,'pid':42,'creation_filetime_100ns':100,'boot_identity':'windows-boot-guid-test',
        'image_path':str(Path(tempfile.gettempdir())/'python.exe'),'argv':['python','-m','fixed_module']}
    ref={'path':str(Path(tempfile.gettempdir())/'file'),'sha256':'a'*64,'bytes':1}
    sources=dict.fromkeys(('python_ref','process_image_ref','request_ref'),ref)
    sources.update(source_files={'fixed_module.py':ref},entry='fixed_module.py',entry_script_ref=None,
        runtime_launcher=None,launcher_provenance='supervisor view')
    value={'schema':s.worker.PROOF,'fresh_process':True,'boot_verified':True,'source_verified':True,
        'request_sha256':'b'*64,'scope_sha256':'c'*64,'os_identity':identity,'sources':sources}
    value['worker_identity_sha256']=s.digest(s.worker.worker_identity_preimage(identity,sources,value['request_sha256'],value['scope_sha256']))
    return value


def owner_fixture(events):
    owner=object.__new__(s.DevSupervisor);owner.role=s.entry.RUN_IDS[0];owner.context=context_fixture()
    owner.context.update(request_sha256='b'*64,scope_sha256='c'*64)
    ref={'path':str(Path(tempfile.gettempdir())/'fixed.json'),'sha256':'a'*64,'bytes':1}
    owner.child={'context_refs':{owner.role:ref},'entry_specs':{owner.role:{'bootstrap':{'owned_directory':tempfile.gettempdir()}}}}
    owner.configuration={'supervisor':{'scope_receipt_ref':ref}}
    owner.spec={'bootstrap':{'owned_directory':tempfile.gettempdir(),'operation_profiles':dict.fromkeys(
        ('dev_supervisor_lifetime','dev_supervisor_metadata','dev_supervisor_verify','dev_supervisor_publication'),price())}}
    owner.operation=lambda kind,phase,**fields:events.append((kind,phase))
    owner.resource=SimpleNamespace(launch=lambda:events.append('resource launch'))
    owner.current_handle=None
    owner.observe_current=lambda:events.append('current OS proof') or {}
    owner.build_launch=lambda identity:{'request_sha256':'b'*64,'scope_sha256':'c'*64,'deadline_seconds':100}
    owner.reference=lambda *args:ref
    owner.publish=lambda value:events.append('completion published') or ref
    owner.bootstrap=SimpleNamespace(header_path=Path(tempfile.gettempdir())/'header',seal_path=Path(tempfile.gettempdir())/'seal',
        end=lambda:events.append('cost owner end'),inspect_evidence=lambda:events.append('cost seal readback') or {'seal_ref':ref})
    return owner


class Tests(unittest.TestCase):
    def test_exact_publication_reference_must_be_original_paid_return_before_end(self):
        binding={'stage':'dev_scoring','request_sha256':'b'*64,'scope_sha256':'c'*64,'static_recipe_sha256':'d'*64,
            'code_sha256':{'fixture.py':'e'*64},'worker_identity_sha256':'f'*64,'attempt_id':s.entry.RUN_IDS[0]}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);spec={'owned_directory':str(root),'operation_profiles':{'dev_supervisor_publication':price()},
                'setup_bounds':price(),'finalization_bounds':price(),'caps':dict.fromkeys(s.services.COUNTERS,10**12)}
            # Explicit fixture accounting only; no authority or execution proof.
            ledger=s.services.BootstrapLedger(spec,binding['request_sha256'],synthetic_fixture=True)
            completion={'path':str(root/'DEV_COMPLETION.json'),'sha256':'a'*64,'bytes':100}
            ledger.operation('dev_supervisor_publication','started','publication',**price())
            ledger.operation('dev_supervisor_publication','returned','publication',reference=completion)
            ledger.end();refs=ledger.inspect_evidence()
            def reader():return s.raw.RawEvidenceReader(lambda *args,**fields:None,
                dict.fromkeys(s.raw.BUDGET_FIELDS,10**12),str(root))
            paid=reader()
            original=s.replay.verify_service_bootstrap({'mode':'synthetic_fixture','request_sha256':binding['request_sha256'],
                'verification_binding':binding},spec,refs['header_ref'],refs['journal_ref'],refs['seal_ref'],
                reader=paid,outer_preflight=lambda *args:None)
            result=s._publication_link(paid,refs['journal_ref'],binding,completion,original)
            self.assertLess(result['publication_seq'],result['end_reservation_seq']);self.assertLess(result['end_reservation_seq'],result['end_seq'])
            wrong=dict(completion,sha256='0'*64)
            with self.assertRaisesRegex(ValueError,'original paid publication'):s._publication_link(reader(),refs['journal_ref'],binding,wrong,original)

    def test_source_absence_refuses_before_paths_launch_or_metadata_objects(self):
        before=set(sys.modules)&{'torch','numpy','transformers','safetensors'}
        with patch.object(Path,'open',side_effect=AssertionError('unpaid IO')),patch.object(s.launch,'launch_native',side_effect=AssertionError('process')):
            with self.assertRaisesRegex(ValueError,'closed before IO'):s._configuration()
            with self.assertRaisesRegex(ValueError,'before IO'):s.verify_completed_dev({},reader=object(),outer_preflight=lambda *args:None)
            with self.assertRaisesRegex(ValueError,'original DEV launch'):s.launch_verifier({},'a'*64,'b'*64,'c'*64)
        self.assertEqual(before,set(sys.modules)&{'torch','numpy','transformers','safetensors'})

    def test_ready_source_identity_ignores_view_text_but_rejects_actual_changes(self):
        observed=proof_fixture();reported=copy.deepcopy(observed)
        reported['sources'].update(runtime_launcher={'path':'launcher'},launcher_provenance='current child view',source_observation='different observer text')
        spec={key:observed[key] for key in ('request_sha256','scope_sha256')}
        s.same_worker_proof(observed,reported,spec)
        for field in ('pid','creation_filetime_100ns'):
            wrong=copy.deepcopy(reported);wrong['os_identity'][field]+=1
            with self.assertRaisesRegex(ValueError,'identity digest'):s.same_worker_proof(observed,wrong,spec)
        wrong=copy.deepcopy(reported);wrong['sources']['source_files']['fixed_module.py']['sha256']='f'*64
        with self.assertRaisesRegex(ValueError,'identity digest'):s.same_worker_proof(observed,wrong,spec)

    def test_actual_protocol_terminal_readback_close_publish_then_end(self):
        events=[];owner=owner_fixture(events);proof=proof_fixture()
        terminal={'schema':s.worker.TERMINAL,'worker_identity_sha256':proof['worker_identity_sha256'],
            'os_identity':proof['os_identity'],'exit_filetime_100ns':200,'exit_code':0}
        clone=SimpleNamespace(close=lambda:events.append('terminal clone closed'))
        monitor=SimpleNamespace(identity_sha256=proof['worker_identity_sha256'],retain_terminal=lambda:events.append('terminal retained') or clone)
        class Owned:
            def __init__(self):self.monitor=monitor
            def handshake(self):events.append('ready');return {'worker_proof':proof}
            def request(self,command,data):
                events.append(command)
                return {'body_completed':True,'native_qualified':False,'os_terminal_verified':False,'raw_artifact':{}} if command=='read' else {'finished':True,'native_qualified':False}
            def terminal(self,seconds):events.append('same OS exit');return terminal
            def close(self):events.append('owned process closed')
        def verify(handle,*args,**kwargs):return terminal if handle is clone else proof
        def readback(*args):
            events.append('original raw worker readback')
            return {'token_journal_ref':{},'attempts_journal_ref':{},'binding':{},'worker_service_refs':{},'worker_service_snapshot':{}}
        with patch.object(s.launch,'NativeLaunchPermit'),patch.object(s.launch,'launch_native',return_value=Owned()), \
                patch.object(s.worker,'verify_worker_handle',side_effect=verify),patch.object(s,'read_worker',side_effect=readback):
            result=s.supervise(owner)
        self.assertEqual([x for x in events if type(x) is str],['resource launch','current OS proof','ready','read','finish','same OS exit',
            'terminal retained','original raw worker readback','owned process closed','terminal clone closed','completion published','cost owner end','cost seal readback'])
        self.assertFalse(result['native_qualified']);self.assertFalse(result['learning_claim'])
        self.assertIn(('dev_supervisor_lifetime','returned'),events)

    def test_failed_read_preserves_original_and_stops_only_owned_process(self):
        events=[];owner=owner_fixture(events);proof=proof_fixture();original=RuntimeError('child read failed')
        class Owned:
            monitor=SimpleNamespace(identity_sha256=proof['worker_identity_sha256'])
            def handshake(self):return {'worker_proof':proof}
            def request(self,*args):raise original
            def _stop_owned(self,reason,primary):events.append(('owned stop',reason,primary))
            def close(self):events.append('owned close');raise ValueError('owned close receipt failed')
        with patch.object(s.launch,'NativeLaunchPermit'),patch.object(s.launch,'launch_native',return_value=Owned()), \
                patch.object(s.worker,'verify_worker_handle',return_value=proof):
            with self.assertRaises(RuntimeError) as raised:s.supervise(owner)
        self.assertIs(raised.exception,original);self.assertIs(events[-2][2],original)
        self.assertNotIn('cost owner end',events);self.assertIn('owned close',events)
        self.assertEqual(original.accounting_failure['message'],'owned close receipt failed')

    def test_denied_close_preflight_still_physically_closes_original_handle(self):
        events=[];owner=owner_fixture(events);owner.current_handle=SimpleNamespace(closed=False)
        def close():owner.current_handle.closed=True;events.append('physical close')
        owner.current_handle.close=close
        owner.operation=lambda *args,**fields:(_ for _ in ()).throw(RuntimeError('outer costs exhausted'))
        with self.assertRaisesRegex(RuntimeError,'costs exhausted'):owner.close_current()
        self.assertEqual(events,['physical close']);self.assertIsNone(owner.current_handle)

    def test_source_verifier_reserves_before_configuration_or_spec_digest(self):
        events=[];owner=owner_fixture(events)
        owner.context.update(request_sha256='b'*64,scope_sha256='c'*64)
        bound={'request_sha256':'b'*64,'scope_sha256':'c'*64};hashed=s.digest(bound);owner._observation_specs={hashed:bound}
        owner._admit=lambda:events.append('source configuration verified')
        with patch.object(s,'digest',side_effect=lambda value:events.append('digest') or hashed):
            self.assertTrue(s.metadata_verifier(owner,hashed,'b'*64,'c'*64))
        self.assertEqual(events[:3],[('dev_supervisor_verify','started'),'source configuration verified','digest'])
        self.assertEqual(events[-1],('dev_supervisor_verify','returned'))

    def test_short_completion_write_is_retained_failure_without_seal(self):
        events=[];owner=owner_fixture(events);stream=MagicMock();stream.write.return_value=0
        with patch.object(Path,'open',return_value=stream),self.assertRaisesRegex(ValueError,'short DEV completion'):
            owner.publish=type(owner).publish.__get__(owner);owner.publish({'schema':s.COMPLETE})
        self.assertIn(('dev_supervisor_publication','started'),events);self.assertIn(('dev_supervisor_publication','failed'),events)
        self.assertNotIn('cost owner end',events)


if __name__=='__main__':unittest.main()
