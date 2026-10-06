"""Observer-only permission/schema/context tests; no native APIs or model."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import unittest
import contract as C
class ContractControls(unittest.TestCase):
    def setUp(self):
        self.head='a'*40;self.implementation='e'*40;root=C.PACKAGE;self.worker=str(C.EXPECTED/'worker.py')
        def d(role,bytes=1):return {'bytes':bytes,'sha256':'b'*64,'role':role}
        sources={str(C.EXPECTED/n):d('python_source') for n in ('contract.py','worker.py','supervisor.py','audit.py')}
        native={str(root/('prefix_library_observer.'+n)):d('native_observer_source') for n in ('c','h')};sources.update(native)
        adapter={'path':str(root/'prefix_library_observer.py'),'descriptor':d('python_source')};sources[adapter['path']]=adapter['descriptor']
        binary={'path':str(root/'native_library_observer.dll'),'descriptor':d('observer_binary')};fixture={'path':str(root/'observer_only_fixture.dll'),'descriptor':d('observer_fixture_binary')}
        observer={'kind':'ROOT_REGISTERED_NATIVE_LIBRARY_OBSERVER_V1','adapter_source':adapter,'native_source_files':native,'binary':binary,'abi':dict(C.ABI),'build_receipt':{'path':str(root/'native_library_observer_build.json'),'file_sha256':'c'*64,'body_sha256':'d'*64}}
        files={binary['path']:binary['descriptor'],fixture['path']:fixture['descriptor']}
        self.runtime=C.seal({'kind':'PROSPECTIVE_OBSERVER_STARTUP_BYTE_ADMISSION_V1','implementation_commit':self.implementation,'files':files,'inherited65_map':{},'stdlib779_map':{},'windows_library_identities':{},'library_event_observer':observer,'stdlib_layout':{},'cache_policy':{},'allowed_initial_loaded_libraries':[binary['path']],'qualification_review':{}})
        self.config=C.seal({'kind':'REGISTERED_OBSERVER_STARTUP_MEASUREMENT_V1','scope':C.SCOPE,'permission':deepcopy(C.PERMISSION),'resources':deepcopy(C.RESOURCES),'implementation_commit':self.implementation,'source_files':sources,'runtime_files':files,'observer_registration':observer,'binary_fixture':fixture,'worker_path':self.worker,'offline_environment':{'HF_HUB_OFFLINE':'1'},'native_launch_authorized':False})
        self.registration=C.seal({'kind':'ROOT_REGISTERED_OBSERVER_STARTUP_LAUNCH_V1','implementation_commit':self.implementation,'config_body_sha256':self.config['sha256'],'runtime_body_sha256':self.runtime['sha256'],'source_files':sources,'runtime_files':files,'profile_sha256':C.digest({'scope':C.SCOPE,'permission':C.PERMISSION,'resources':C.RESOURCES,'ABI':C.ABI}),'worker_path':self.worker,'owner_approval':{'approved':False,'scope':C.SCOPE,'evidence_reference':''},'native_launch_authorized':False})
    def schema(self):return C._schema(self.config,self.runtime,self.registration,self.head,fixture_only=True)
    def test_separate_scope_metadata_keeps_zero_model_tensor_permission(self):
        value=self.schema();self.assertTrue(value['fixture_only']);self.assertFalse(value['native_authority_issued'])
        self.assertEqual(value['config']['permission']['allocations'],0)
        self.assertEqual(value['config']['implementation_commit'],self.implementation)
        self.assertNotEqual(value['config']['implementation_commit'],value['expected_head'])
    def test_embedded_dispatch_HEAD_and_mixed_implementation_are_refused(self):
        self.runtime['implementation_commit']='f'*40
        with self.assertRaisesRegex(ValueError,'implementation ancestor'):self.schema()
        for value in (self.config,self.runtime,self.registration):value['implementation_commit']=self.head
        with self.assertRaisesRegex(ValueError,'implementation ancestor'):self.schema()
    def test_ancestor_and_exact_committed_sources_join_distinct_later_JSON_bytes(self):
        import tempfile,hashlib,subprocess
        from types import SimpleNamespace
        with tempfile.TemporaryDirectory(prefix='CPU_dispatch_source_') as td:
            repository=Path(td).resolve();package=repository/'experiments/rineya_synth_centralized_reasoning_v1';package.mkdir(parents=True)
            paths={name:package/name for name in ('config.json','runtime.json','registration.json','worker.py')}
            for name,path in paths.items():path.write_bytes(('INERT original '+name).encode('ascii'))
            original={name:path.read_bytes() for name,path in paths.items()};raw=original['worker.py']
            value={'expected_head':self.head,'config':{'implementation_commit':self.implementation,'source_files':{str(paths['worker.py']):{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}}},'anchors':{'runtime_path':str(paths['runtime.json']),'registration_path':str(paths['registration.json'])}}
            calls=[]
            def run(args,**kwargs):
                calls.append(args)
                if args[1]=='rev-parse':return SimpleNamespace(stdout=self.head+'\n')
                if args[1]=='status':return SimpleNamespace(stdout='')
                if args[1]=='merge-base':return SimpleNamespace(stdout=b'')
                commit,path=args[2].split(':',1);name=Path(path).name
                self.assertEqual(commit,self.implementation if name=='worker.py' else self.head)
                return SimpleNamespace(stdout=original[name])
            with patch.object(C,'PACKAGE',package),patch.object(C.subprocess,'run',side_effect=run):
                C._committed_dispatch_checks(value,paths['config.json'])
            self.assertIn(['git','merge-base','--is-ancestor',self.implementation,self.head],calls)
            def invalid(args,**kwargs):
                if args[1]=='merge-base':raise subprocess.CalledProcessError(1,args)
                return run(args,**kwargs)
            with patch.object(C,'PACKAGE',package),patch.object(C.subprocess,'run',side_effect=invalid):
                with self.assertRaises(subprocess.CalledProcessError):C._committed_dispatch_checks(value,paths['config.json'])
            original['worker.py']=b'INERT foreign committed source'
            with patch.object(C,'PACKAGE',package),patch.object(C.subprocess,'run',side_effect=run):
                with self.assertRaisesRegex(ValueError,'implementation-ancestor source bytes'):C._committed_dispatch_checks(value,paths['config.json'])
            original['worker.py']=raw;paths['runtime.json'].write_bytes(b'changed JSON')
            with patch.object(C,'PACKAGE',package),patch.object(C.subprocess,'run',side_effect=run):
                with self.assertRaisesRegex(ValueError,'JSON bytes committed'):C._committed_dispatch_checks(value,paths['config.json'])
    def test_each_model_tensor_or_CUDA_flag_is_rejected(self):
        for name in ('model','torch_import','CUDA_context','tensor_allocation','forward','backward','optimizer','scoring'):
            config=deepcopy(self.config);config['permission'][name]=True
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,'permission/resource'):C._schema(config,self.runtime,self.registration,self.head,fixture_only=True)
    def test_two_tensor_scope_cannot_be_reused(self):
        self.config['scope']='ONE_EXPOSED_PREFIX_CUDA_INT64_INPUT_MASK_ONLY'
        with self.assertRaisesRegex(ValueError,'observer-only config scope'):self.schema()
    def test_native_launch_requires_explicit_separate_approval(self):
        with self.assertRaisesRegex(ValueError,'launch approval'):C._schema(self.config,self.runtime,self.registration,self.head,fixture_only=False)
    def test_CPU_flags_cannot_carry_native_approval(self):
        self.config['native_launch_authorized']=True
        with self.assertRaisesRegex(ValueError,'CPU fixture'):self.schema()
    def test_resource_floor_change_is_rejected(self):
        self.config['resources']['GPU_free_at_admission_GiB']=1
        with self.assertRaisesRegex(ValueError,'permission/resource'):self.schema()
    def test_required_worker_auditor_source_and_fixed_path(self):
        self.config['source_files'].pop(str(C.EXPECTED/'audit.py'))
        with self.assertRaisesRegex(ValueError,'source roster'):self.schema()
    def test_own_fixture_binary_cannot_point_to_other_data(self):
        self.config['binary_fixture']['path']=str(C.PACKAGE/'model.safetensors')
        with self.assertRaisesRegex(ValueError,'inert fixture binary'):self.schema()
    def test_native_stage_refuses_before_selected_bytes(self):
        with patch.object(C,'_read_context',side_effect=AssertionError('must not read')):
            with self.assertRaisesRegex(ValueError,'no native authority'):C.validate_context('missing')
    def test_actual_external_selected_json_reader_and_changed_anchor_refusal(self):
        import tempfile,hashlib
        with tempfile.TemporaryDirectory(prefix='CPU_observer_contract_') as td:
            root=Path(td)/'rineya_synth_centralized_reasoning_v1';(root/'configs').mkdir(parents=True)
            config_path=root/'configs/CPU.json';runtime_path=root/'runtime.json';registration_path=root/'registration.json'
            def write(path,value):path.write_bytes((C.canonical(value)+'\n').encode('ascii'));return hashlib.sha256(path.read_bytes()).hexdigest()
            old_root=str(C.PACKAGE)
            def rebase(value):
                if type(value) is str:return value.replace(old_root,str(root))
                if type(value) is list:return [rebase(v) for v in value]
                if type(value) is dict:return {rebase(k):rebase(v) for k,v in value.items()}
                return value
            config=C.seal(rebase(self.config));runtime=C.seal(rebase(self.runtime));registration=rebase(self.registration)
            registration.update(config_body_sha256=config['sha256'],runtime_body_sha256=runtime['sha256'])
            registration=C.seal(registration)
            cf=write(config_path,config);rf=write(runtime_path,runtime);sf=write(registration_path,registration)
            anchors={'expected_config_file':cf,'expected_config_body':config['sha256'],'runtime_path':str(runtime_path),'expected_runtime_file':rf,'expected_runtime_body':runtime['sha256'],'registration_path':str(registration_path),'expected_registration_file':sf,'expected_registration_body':registration['sha256'],'expected_head':self.head}
            # Pure schema plus actual JSON/anchor readers; physical historical
            # map bytes and shared inventory are explicit fixture seams only.
            with patch.object(C,'PACKAGE',root),patch.object(C,'EXPECTED',root/'observer_startup_measurement_v1'),patch.object(C,'_shared_inventory_checks'),patch.object(C,'file_descriptor'):
                context=C._CPU_context_fixture(config_path,_capability=C._CPU_capability(),**anchors)
                self.assertTrue(C.context_value(context)['fixture_only'])
                config_path.write_bytes(config_path.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'external file bytes'):
                    C._CPU_context_fixture(config_path,_capability=C._CPU_capability(),**anchors)
    def test_stream_reads_only_selected_size_plus_one_and_rejects_growth(self):
        import tempfile,hashlib
        with tempfile.TemporaryDirectory(prefix='CPU_selected_stream_') as td:
            path=Path(td)/'selected.py';path.write_bytes(b'abc')
            descriptor={'bytes':3,'sha256':hashlib.sha256(b'abc').hexdigest(),'role':'python_source'}
            reads=[]
            class Growing:
                def __enter__(self):return self
                def __exit__(self,*args):pass
                def read(self,n):reads.append(n);return b'abc' if len(reads)==1 else b'x'
            actual=Path.open
            def opened(p,*args,**kwargs):return Growing() if p==path else actual(p,*args,**kwargs)
            with patch.object(Path,'open',opened):
                with self.assertRaisesRegex(ValueError,'growing selected'):C.file_descriptor(path,descriptor)
            self.assertEqual(reads,[3,1])
    def test_original_opaque_CPU_context_cannot_be_forged_or_rebased(self):
        value=self.schema();value.update(config_path='CPU_ONLY',anchors={})
        context=C._CPU_context_value_fixture(value,_capability=C._CPU_capability());value['config']['scope']='changed'
        self.assertEqual(C.context_value(context)['config']['scope'],C.SCOPE)
        with self.assertRaisesRegex(ValueError,'factory context'):C.context_value(object())
        with self.assertRaisesRegex(ValueError,'private'):C._CPU_context_value_fixture(value,_capability=object())

if __name__=='__main__':unittest.main()
