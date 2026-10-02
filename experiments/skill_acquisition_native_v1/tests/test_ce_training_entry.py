"""Closed source entry and paid metadata intake; no OS/framework/dispatch."""
import copy
from pathlib import Path
import sys
import tempfile
from types import MappingProxyType
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_training_entry as e,ce_training_configuration as c
from .test_ce_qualified_receipt import Tests as PairFixture


def configuration(root,selection):
    ref=lambda name:{'path':str(root/(name+'.json')),'sha256':e.digest(name),'bytes':10}
    refs={key:ref(key) for key in e.controller.REFS};refs['qualifiedreceipt_ref']=selection['references']['receipt']
    request=e.controller.build_request('training','canonical',refs,{key:100000 for key in e.controller.BOUND_FIELDS})
    request.update(mode='registered_scope',approved=True,training_authorized=True,qualification_authorized=False,status='AUTHORIZED_TRAINING')
    code={fn.__module__.replace('.','/')+'.py':'a'*64 for fn in e.role_functions().values()}
    worker={'entry_module':e.ENTRY_MODULE,'request_ref':ref('request'),'approved_code_sha256':code,'source_checkout':str(root/'source')}
    spec={'schema':e.services.SCHEMA,'approved_code_sha256':code,'scope_adapters':{'static_recipe_sha256':selection['binding']['static_recipe_sha256'],'worker':worker},
        'resource':{'resource_probe':{'output_root':str(root/'output')}},'bootstrap':{'owned_directory':str(root/'output'/'services')},
        'constructor_prepayment':{'owned_directory':str(root/'output'/'services')},'role_descriptors':{
            name:{'module':fn.__module__,'symbol':fn.__name__,'source_sha256':'a'*64} for name,fn in e.role_functions().items()}}
    rpc={'schema':e.rpc.SPEC,'deadline_seconds':60,'rpc_seconds':5,'max_message_bytes':4096,'max_messages':8,
        'max_request_bytes':e.raw.MAX_JSON,'operation_bounds':dict(read_bytes_upper=e.raw.MAX_JSON+1,write_bytes_upper=4096,
            identity_hash_attempts_upper=64,identity_hash_bytes_upper=64*e.raw.MAX_JSON,wall_s_upper=5)}
    result={'schema':e.SCHEMA,'request':request,'service_spec':spec,'request_ref':worker['request_ref'],'service_spec_ref':ref('service'),
        'rpc_spec':rpc,'qualification_selection':selection,'qualification_selection_sha256':e.digest(selection)}
    return dict(result,configuration_sha256=e.digest(result))


class Tests(unittest.TestCase):
    def setUp(self):
        temp=tempfile.TemporaryDirectory();self.addCleanup(temp.cleanup);self.root=Path(temp.name)
        pair=PairFixture();pair.setUp();self.addCleanup(pair.doCleanups)
        self.config=configuration(self.root,copy.deepcopy(pair.selection))
    def test_closed_source_entry_refuses_before_argv_io_model_or_registry_mutation(self):
        before=set(sys.modules)&{'torch','transformers','numpy','safetensors'}
        with patch.object(Path,'open',side_effect=AssertionError('closed source before IO')):
            with self.assertRaisesRegex(ValueError,'remains closed'):e.main(['--request','unapproved'])
        self.assertIsNone(e._INSTALLED_SHA256);self.assertFalse(e.services._REGISTERED_SERVICE_VERIFIERS)
        self.assertEqual(before,set(sys.modules)&{'torch','transformers','numpy','safetensors'})
    def test_configuration_keeps_exact_selection_and_fixed_entry(self):
        self.assertIs(e.validate_configuration(self.config),self.config)
        for edit in (lambda cfg:cfg['request'].update(qualification_authorized=True),
            lambda cfg:cfg['service_spec']['scope_adapters']['worker'].update(entry_module='caller.module'),
            lambda cfg:cfg.update(qualification_selection_sha256='b'*64)):
            changed=copy.deepcopy(self.config);edit(changed)
            changed['configuration_sha256']=e.digest({key:value for key,value in changed.items() if key!='configuration_sha256'})
            with self.assertRaises(ValueError):e.validate_configuration(changed)
    def test_fixed_source_intake_permanent_reservation_precedes_read_and_literal_pin_cannot_change(self):
        policy={'schema':c.DECLARATION,'configuration_path':str(self.root/'configuration.json'),
            'owned_directory':str(self.root/'output'/'intake'),'source_checkout':str(self.root/'source'),
            'output_root':str(self.root/'output'),'choice_sha256':self.config['request']['selectedoperatorchoice_ref']['sha256'],
            'static_recipe_sha256':self.config['service_spec']['scope_adapters']['static_recipe_sha256'],
            'qualification_selection_sha256':self.config['qualification_selection_sha256'],
            'bounds':{'read_bytes':64*1024**2,'write_bytes':1024**2,'hash_attempts':10000,'hash_bytes':256*1024**2,
                'metadata_bytes':1024**2,'operation_attempts':10000,'wall_seconds':120}}
        Path(policy['configuration_path']).write_bytes(e.canonical(self.config)+b'\n')
        original=Path.open;starts=[]
        def opened(path,mode='r',*args,**kwargs):
            if str(path)==policy['configuration_path'] and mode=='rb':
                first=e.raw._json((Path(policy['owned_directory'])/'parent.jsonl').read_bytes().splitlines()[0]);starts.append(first['kind'])
            return original(path,mode,*args,**kwargs)
        with (patch.object(c,'_SOURCE_DECLARATION',MappingProxyType(policy)),patch.object(c,'_SOURCE_DECLARATION_SHA256',e.digest(policy)),
              patch.object(c,'_LOADED',None),patch.object(c,'_EVIDENCE',None),patch.object(Path,'open',opened)):
            configured,evidence=c.load_source_configuration()
            self.assertEqual(configured,self.config);self.assertTrue(evidence['metadata_only']);self.assertFalse(evidence['native_qualified'])
            self.assertEqual(starts,['PERMANENT_RESERVATION'])
            policy['qualification_selection_sha256']='b'*64
            with self.assertRaisesRegex(ValueError,'literal source declaration pin drift'):c.load_source_configuration()
    def test_rpc_train_body_finish_order_and_no_early_finish(self):
        events=[];commands=iter(({'command':'train','data':{}},{'command':'finish','data':{}}))
        class Pipe:
            def emit(self,*args):events.append(args)
            def receive(self,sequence):return next(commands)
        ref={'path':str(self.root/'body.json'),'sha256':'a'*64,'bytes':10}
        def train():events.append('actual-body');return {'seal':{'status':'UNQUALIFIED_TRAINING_BODY_SEAL','body_ref':ref}}
        self.assertEqual(e.serve_training(Pipe(),{'proof':'fixture'},train),0)
        self.assertEqual(events[0][0],'ready');self.assertEqual(events[1],'actual-body')
        self.assertFalse(events[2][2]['os_terminal_verified']);self.assertFalse(events[2][2]['native_qualified'])
        commands=iter(({'command':'finish','data':{}},))
        with self.assertRaisesRegex(ValueError,'fixed train command'):e.serve_training(Pipe(),{},lambda:(_ for _ in ()).throw(AssertionError('early body')))


if __name__=='__main__':unittest.main()
