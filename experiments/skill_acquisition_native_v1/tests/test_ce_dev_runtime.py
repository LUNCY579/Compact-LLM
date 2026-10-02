"""Closed DEV read schema/order/churn controls, with no framework or outputs."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_dev_runtime as d


def context_fixture():
    banks = {}
    for name,n in zip(d.BANKS,(250,250,250,1200)):
        ids = ['%s-%04d' % (name,i) for i in range(n)]
        banks[name] = {'ids':ids,'ids_sha256':d.digest(ids),'parent_by_id':{rid:'world-%04d'%i for i,rid in enumerate(ids)},
            'prompt_hash_by_id':dict.fromkeys(ids,'a'*64)}
    def ref(name):return {'path':str(Path(tempfile.gettempdir())/name),'sha256':'b'*64,'bytes':100}
    model = {'model_dir':str(Path(tempfile.gettempdir())/'base'),'base_weights_sha256':'c'*64,'Supa_sha256':'c'*64,
        'Supa_state_fingerprint_sha256':'d'*64,'warm_anchor_sha256':'e'*64,'CE375_donor_sha256':'f'*64,
        'asset_receipt_ref':ref('asset.json'),'tokenizer_receipt_ref':ref('tokenizer.json'),
        'warm_delta_ref':dict(ref('warm.safetensors'),data_sha256='1'*64),
        'donor_ref':dict(ref('donor.safetensors'),data_sha256='2'*64,deployed_arm_sha256='f'*64)}
    runtime = {'decoding_preimage':{'device':'cuda:0','dtype':'float32'}}
    runtime['decoding_sha256']=d.digest(runtime['decoding_preimage'])
    scope = {'schema':'CE8-DEV-EVALUATION-SCOPE-1','evaluation_authorized':True,'production_training_authorized':False,
        'greedy':d.GREEDY,'banks':banks,'model_bindings':{name:{'model_sha256':model[name+'_sha256'],
            'tokenizer_sha256':model['tokenizer_receipt_ref']['sha256'],'decoding_sha256':runtime['decoding_sha256']}
            for name in ('Supa','warm_anchor','CE375_donor')}}
    plan = {'schema':'CE8-DEV-FIXED-READ-PLAN-1','evaluation_authorized':True,'scope_sha256':d.digest(scope),
        'rows_per_read':1950,'total_greedy_completions':9750,'max_new_tokens':128,'refill_allowed':False,
        'checkpoint_selection_allowed':False,'allowed_roles':['dev','guard_dev'],
        'reads':[{'read_id':read,'model_name':name,'read_index':index,'fresh_process_required':True,
                  'bank_names':list(d.BANKS),'rows':1950} for read,name,index in d.RUNS]}
    profile = {'read_bytes_upper':10**9,'write_bytes_upper':10**6,'identity_hash_attempts_upper':10000,
               'identity_hash_bytes_upper':10**9,'wall_s_upper':1000}
    return {'schema':d.CONTEXT,'request_sha256':'3'*64,'scope_sha256':d.digest(scope),'evaluation_authorized':True,
        'production_training_authorized':False,'scope':scope,'plan':plan,'runtime':runtime,'model':model,
        'source':{'fixture.py':'4'*64},'read_id':'Supa-read-1','policy':{'schema':d.POLICY,
        'output_directory':str(Path(tempfile.gettempdir())/'dev-output'),'physical_wall_seconds_upper':1000,
        'journal_bytes_upper':16*d.raw.CHUNK,'receipt_bytes_upper':d.raw.MAX_JSON,
        'operation_profiles':{kind:dict(profile) for kind in d.KINDS}}}


class Tests(unittest.TestCase):
    def test_permit_recheck_reservation_precedes_first_digest_and_verifier(self):
        events=[];context={'scope_sha256':'a'*64,'request_sha256':'b'*64}
        operation=lambda kind,phase,**fields:events.append((kind,phase))
        gate=lambda stage:None;worker=object();lease=object()
        price=dict(read_bytes_upper=0,write_bytes_upper=0,identity_hash_attempts_upper=64,
            identity_hash_bytes_upper=256*d.raw.CHUNK,wall_s_upper=5)
        issuer=SimpleNamespace(dev_verify_profile=price)
        def verifier(*args):events.append('verify');return True
        permit=object.__new__(d.DevReadPermit)
        values=dict(_seal=d._SEAL,_context=d.digest(context),_issuer=issuer,_verifier=verifier,
            _operation=operation,_gate=gate,_worker=worker,_lease=lease,_verify_profile=copy.deepcopy(price))
        for key,value in values.items():object.__setattr__(permit,key,value)
        original=d.digest
        def digest(value):events.append('digest');return original(value)
        with patch.object(d,'_REGISTERED_DEV_VERIFIERS',frozenset((verifier,))),patch.object(d,'digest',side_effect=digest):
            permit.require(context,operation,gate,worker,lease)
            self.assertEqual(events[0],('dev_entry_verify','started'));self.assertEqual(events[1:3],['digest','verify'])
            events.clear();issuer.dev_verify_profile=dict(price,wall_s_upper=4)
            with self.assertRaisesRegex(ValueError,'immutable verification'):permit.require(context,operation,gate,worker,lease)
            self.assertNotIn('digest',events);self.assertIn(('dev_entry_verify','failed'),events)
            events.clear()
            with self.assertRaisesRegex(ValueError,'owned lease'):permit.require(context,operation,gate,worker,object())
            self.assertEqual(events,[])

    def test_default_closed_before_context_hash_io_framework_or_model(self):
        class Trap:
            def __getattribute__(self,name):raise AssertionError('unadmitted source access')
        before=set(sys.modules)&{'torch','numpy','transformers','safetensors'}
        with patch.object(Path,'open',side_effect=AssertionError('unpaid DEV IO')):
            with self.assertRaisesRegex(ValueError,'remains closed'):d.DevReadPermit(Trap(),Trap(),lambda *args:True,Trap(),Trap())
            with self.assertRaisesRegex(ValueError,'permit before IO'):d.run_dev_read(Trap(),'Supa-read-1',operation=Trap(),gate=Trap(),worker_handle=Trap(),lease_owner=Trap())
        self.assertEqual(before,set(sys.modules)&{'torch','numpy','transformers','safetensors'})
    def test_fixed1950_five_read_plan_and_supa_hash_namespace(self):
        context=context_fixture();self.assertIs(d.validate_context(context),context)
        self.assertNotEqual(context['model']['Supa_sha256'],context['model']['Supa_state_fingerprint_sha256'])
        changes=(lambda c:c['plan']['reads'].pop(),lambda c:c['plan']['reads'][0].update(rows=1949),
            lambda c:c['plan'].update(refill_allowed=True),lambda c:c['model'].update(Supa_sha256='d'*64),
            lambda c:c['model']['donor_ref'].update(deployed_arm_sha256='e'*64),
            lambda c:c['policy']['operation_profiles']['dev_cleanup'].update(wall_s_upper=1))
        for change in changes:
            wrong=copy.deepcopy(context);change(wrong)
            with self.assertRaises(ValueError):d.validate_context(wrong)
    def test_fake_source_callback_or_historical_worker_dictionary_cannot_grant(self):
        context=context_fixture();called=[]
        def verifier(*args):called.append(args);return True
        operation=lambda *args,**fields:None;gate=lambda stage:None
        issuer=SimpleNamespace(operation=operation,gate=gate,dev_worker_handle={'native':True},dev_lease_owner=object())
        with patch.object(d,'_REGISTERED_DEV_VERIFIERS',frozenset((verifier,))):
            with self.assertRaisesRegex(ValueError,'worker/lease/observer'):d.DevReadPermit(context,issuer,verifier,operation,gate)
        self.assertEqual(called,[])
    def test_donor_rebuild_order_never_collapses_warm_and_R(self):
        self.assertEqual(d.model_load_steps('CE375_donor'),['base_FP32','add_warm_FP32','add_saved_R_FP32'])
        self.assertEqual(d.model_load_steps('Supa'),['base_FP32'])
        with self.assertRaises(ValueError):d.model_load_steps('best_checkpoint')
    def test_one_process_read_retains_and_refuses_partial_existing_journals(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp);paths=d.fresh_journal_paths(directory)
            paths[0].write_bytes(b'partial-source-read')
            with self.assertRaisesRegex(ValueError,'no implicit resume'):d.fresh_journal_paths(directory)
            self.assertEqual(paths[0].read_bytes(),b'partial-source-read')
    def test_churn_is_exact_generated_token_identity_not_same_decoded_text(self):
        first=[{'id':'row-%04d'%i,'completion_ids':[1,151645],'stop_reason':'eos'} for i in range(1950)]
        second=copy.deepcopy(first);self.assertTrue(d.validate_churn(first,second)['zero_generated_token_churn'])
        second[42]['completion_ids']=[2,151645]
        self.assertEqual(d.validate_churn(first,second)['changed_ids'],['row-0042'])
        second.pop()
        with self.assertRaises(ValueError):d.validate_churn(first,second)


if __name__=='__main__':unittest.main()
