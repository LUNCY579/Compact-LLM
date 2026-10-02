"""Original cost files and full375 metadata record fixtures; no model/OS work."""
import copy
import hashlib
from pathlib import Path
import struct
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_completed_training as c
from . import test_ce_training_supervisor as transcript
from . import test_ce_journal as original
from .test_ce_finalizer_accounting import Clock, spec
from . import test_ce_service_bootstrap_verifier as service_fixture


class Trap:
    def __getattribute__(self,name):raise AssertionError('before source completed pin')
    def __call__(self,*args,**fields):raise AssertionError('before source completed pin')


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='completed-training-pure-');cls.addClassCleanup(cls.temp.cleanup)
        cls.root=Path(cls.temp.name); helper=object.__new__(transcript.Tests)
        helper.root=cls.root;_,helper.bindings,_=original.fixture()
        cls.ce_state,cls.artifact,cls.ce_header,cls.ce_journal,cls.schedule=helper.native_metadata_transcript()
    def ref(self,path):
        blob=path.read_bytes();return {'path':str(path),'sha256':hashlib.sha256(blob).hexdigest(),'bytes':len(blob)}
    def write(self,path,value):path.write_bytes(c.canonical(value)+b'\n');return self.ref(path)
    def setUp(self):
        self.directory=self.root/self._testMethodName;self.directory.mkdir()
        self.selection=self.pair()
    def pair(self):
        artifact=copy.deepcopy(self.artifact);recipe_sha=c.digest(artifact['bindings']['recipe'])
        runtime={'decoding_preimage':artifact['bindings']['runtime']};anchors={'common_anchor_sha256':'a'*64,'own_start_sha256':'a'*64}
        rtref=self.write(self.directory/'RUNTIME.json',runtime);aref=self.write(self.directory/'ANCHORS.json',anchors)
        src=self.write(self.directory/'SOURCE.json',{'code_sha256':artifact['bindings']['source']})
        ledger=self.write(self.directory/'LEDGER_SPEC.json',artifact)
        scope={'native_ledger_ref':ledger,'native_backend':{'parameter_inventory':[{'name':'w','shape':[2],'dtype':'F32'}]},
               'execution':{'boundaries':375,'checkpoint_boundaries':self.schedule}}
        sref=self.write(self.directory/'SCOPE.json',scope)
        self.clock=Clock();self.owner=object.__new__(c.costs.FinalizerCostOwner)
        policy=spec(self.directory)
        self.owner._initialize(policy,self.root,'a'*64,sref['sha256'],None,native=False,clock=self.clock)
        binding={'stage':'training_supervisor','request_sha256':'a'*64,'scope_sha256':sref['sha256'],
            'static_recipe_sha256':recipe_sha,'code_sha256':artifact['bindings']['source'],'worker_identity_sha256':'9'*64,
            'attempt_id':'independent-production-completion'}
        service_spec={'bootstrap':service_fixture.specification(self.directory/'service-bootstrap')}
        service_ref=self.write(self.directory/'SERVICE_SPEC.json',service_spec)
        service_refs={};service_results={}
        for name,dirname in (('worker','service-bootstrap'),('supervisor','supervisor-service-bootstrap')):
            svspec=dict(copy.deepcopy(service_spec['bootstrap']),owned_directory=str(self.directory/dirname))
            svclock=service_fixture.Clock();sv=c.supervisor.services.BootstrapLedger(svspec,'a'*64,synthetic_fixture=True,clock=svclock)
            svclock.advance(1);sv.end();service_refs[name]=sv.inspect_evidence()
            reader=c.raw.RawEvidenceReader(lambda *args,**fields:None,{key:10**12 for key in c.raw.BUDGET_FIELDS},self.root)
            service_results[name]=c.service_replay.verify_service_bootstrap({'mode':'synthetic_fixture','request_sha256':'a'*64,'verification_binding':binding},
                svspec,service_refs[name]['header_ref'],service_refs[name]['journal_ref'],service_refs[name]['seal_ref'],reader=reader,outer_preflight=lambda *args:None)
        header=c.canonical({'w':{'dtype':'F32','shape':[2],'data_offsets':[0,8]}})
        payload=struct.pack('<Q',len(header))+header+struct.pack('<2f',3,4)
        dp=self.directory/'donor.safetensors';dp.write_bytes(payload);dref=self.ref(dp)
        reader=c.raw.RawEvidenceReader(lambda *args,**fields:None,{key:10**12 for key in c.raw.BUDGET_FIELDS},self.root)
        saved=c.delta.read_delta(reader,{**dref,'role':'qualification_delta','binding_sha256':c.digest(binding)},binding,scope['native_backend']['parameter_inventory'])
        donor={'schema':'SKILL-ACQUISITION-DONOR-DELTA-1','path':dref['path'],'file_sha256':dref['sha256'],
            'common_anchor_sha256':'a'*64,'own_start_sha256':'a'*64,'reference':'EXACT_REBUILT_COMMON_WARM_ANCHOR',
            **saved,'final_master_sha256':self.ce_state.progress['history'][-1]['final_policy_id'],'deployed_arm_sha256':'b'*64}
        body={'schema':c.runtime.PUBLICATION,'status':'TRAINING_BODY_COMPLETE',
            **{key:binding[key] for key in ('request_sha256','scope_sha256','static_recipe_sha256')},
            'raw_refs':{'donor':dref,'header':self.ce_header,'journal':self.ce_journal},'journal_snapshot':self.ce_state.snapshot(),
            'invocation_id':'metadata-invocation','donor':donor}
        bref=self.write(self.directory/'BODY.json',body)
        proof={'schema':c.worker.TERMINAL,'request_sha256':'a'*64,'scope_sha256':sref['sha256'],'worker_identity_sha256':'9'*64,
               'exit_code':0,'status_file_used':False,'fixture_only':True}
        self.clock.advance(1);self.owner.operation('finalizer_terminal_observation','started','terminal')
        self.owner.operation('finalizer_terminal_observation','returned','terminal',terminal_proof=proof,observed_before=101.,observed_after=101.)
        pre={'binding':binding,'body_ref':bref,'original_journal_refs':body['raw_refs'],'actual_worker_terminal':proof,
             'derived_journal':self.ce_state.snapshot(),'checkpoint_saves':c.supervisor.validate_checkpoint_schedule(self.ce_state,scope['execution']),
             'saved_donor':saved,'worker_service_refs':service_refs['worker'],'worker_service_replay':service_results['worker']}
        pref=self.owner.write_json('TRAINING_PRE_CLOSE_REPLAY.json',pre)
        snap=service_results['supervisor']['derived_snapshot']
        self.owner.operation('finalizer_service_end','started','end-service')
        self.owner.operation('finalizer_service_end','returned','end-service',observed_before=101.,observed_after=101.,evidence=service_refs['supervisor'])
        receipt={'schema':c.supervisor.SCHEMA,'status':'BODY_EXIT_VERIFIED_REQUIRES_COST_SEAL','updates':375,'body_ref':bref,
            'independent_replay_ref':pref,'finalizer_header_ref':self.owner.header_ref,'expected_finalizer_seal_path':str(self.owner.seal_path),
            'cost_owner_binding_sha256':c.digest(self.owner.header),'request_sha256':'a'*64,'scope_sha256':sref['sha256'],
            'native_qualified':False,'learning_claim':False,'supervisor_service_replay':service_results['supervisor'],
            'supervisor_terminal_suffix':{'recorded_end':101.,'observed_after':101.,'suffix_upper_seconds':120,'observed_suffix_seconds_upper':0.}}
        receiptref=self.owner.write_json('TRAINING_COMPLETION.json',receipt);evidence=self.owner.close_and_seal()
        joint={'schema':c.supervisor.SCHEMA,'status':'TRAINING_BODY_EXIT_COSTS_VERIFIED','completed':True,'updates':375,
               'receipt_ref':receiptref,'cost_evidence':evidence,'native_qualified':False,'learning_claim':False}
        completionref=self.write(self.directory/'JOINT.json',joint)
        return {'schema':c.SCHEMA,'owned_root':str(self.root),'binding':binding,'finalizer_specification':policy,
            'runtime_sha256':c.digest(runtime),'anchors_sha256':c.digest(anchors),'references':{
                'completion':completionref,'receipt':receiptref,'preclose':pref,'header':evidence['header_ref'],
                'journal':evidence['journal_ref'],'seal':evidence['seal_ref'],'body':bref,'CE_header':self.ce_header,
                'CE_journal':self.ce_journal,'scope':sref,'ledger_spec':ledger,'runtime':rtref,'anchors':aref,'source':src,'donor':dref,
                'service_spec':service_ref}}
    def verify(self,selection=None):
        reader=c.raw.RawEvidenceReader(lambda *args,**fields:None,{key:10**12 for key in c.raw.BUDGET_FIELDS},self.root)
        return c.verify_fixture_pair(selection or self.selection,reader=reader,outer_preflight=lambda *args:None)
    def test_full375_seven_CP_and_cost_pair_fixture_never_admits_DEV(self):
        result=self.verify();self.assertFalse(result['completed_pair_verified']);self.assertTrue(result['synthetic_pair_consistent'])
        self.assertEqual(result['updates'],375);self.assertEqual(result['checkpoint_boundaries'],self.schedule)
        self.assertFalse(result['evaluation_authorized']);self.assertFalse(result['execution_reverified']);self.assertFalse(result['learning_claim'])
        self.assertEqual(result['derived_finalizer_costs'],self.owner.counts)
    def test_default_closed_header_only_and_flag_only_before_read(self):
        with patch.object(Path,'open',side_effect=AssertionError('unselected read')):
            with self.assertRaisesRegex(ValueError,'consumer closed'):c.selected_training()
            for value in ({'completed':True},MappingProxyType(self.selection),True):
                with self.assertRaisesRegex(ValueError,'consumer closed'):c.verify_completed_training(value,reader=Trap(),outer_preflight=Trap())
    def test_missing_seal_refund_and_partial_finalizer_tail_reject(self):
        wrong=copy.deepcopy(self.selection);wrong['references'].pop('seal')
        with self.assertRaisesRegex(ValueError,'all completed'):self.verify(wrong)
        sealpath=Path(self.selection['references']['seal']['path']);seal=c.raw._json(sealpath.read_bytes())
        seal['consumed_declared_bounds']['io_read_bytes_upper']-=1;self.write(sealpath,seal)
        wrong=copy.deepcopy(self.selection);wrong['references']['seal']=self.ref(sealpath)
        with self.assertRaisesRegex(ValueError,'cost seal refund'):self.verify(wrong)
    def test_saved_donor_actual_bytes_mismatch_and_preclose_CP_omission_reject(self):
        donor=Path(self.selection['references']['donor']['path']);data=donor.read_bytes();donor.write_bytes(data[:-1]+b'x')
        with self.assertRaisesRegex(ValueError,'bytes/hash mismatch'):self.verify()
        donor.write_bytes(data)
        path=Path(self.selection['references']['preclose']['path']);pre=c.raw._json(path.read_bytes());pre['checkpoint_saves'].pop()
        self.write(path,pre)
        # Original finalizer write refs cannot be replaced with an unrelated
        # rehashed metadata record, even when the claimed count remains375.
        with self.assertRaises(ValueError):self.verify()
    def test_original_service_bytes_cannot_be_replaced_by_unchanged_replay_DTO(self):
        pre=c.raw._json(Path(self.selection['references']['preclose']['path']).read_bytes())
        path=Path(pre['worker_service_refs']['journal_ref']['path']);original=path.read_bytes()
        path.write_bytes(original.replace(b'"seq":0',b'"seq":9',1))
        with self.assertRaises(ValueError):self.verify()
    def test_fixture_header_cannot_be_promoted_by_source_pin_and_completion_flags(self):
        pinned=MappingProxyType(self.selection)
        with patch.object(c,'_SOURCE_SELECTION',pinned),patch.object(c,'_SOURCE_SELECTION_SHA256',c.digest(self.selection)):
            reader=c.raw.RawEvidenceReader(lambda *args,**fields:None,{key:10**12 for key in c.raw.BUDGET_FIELDS},self.root)
            with self.assertRaisesRegex(ValueError,'original native finalizer header'):
                c.verify_completed_training(pinned,reader=reader,outer_preflight=lambda *args:None)


if __name__=='__main__':unittest.main()
