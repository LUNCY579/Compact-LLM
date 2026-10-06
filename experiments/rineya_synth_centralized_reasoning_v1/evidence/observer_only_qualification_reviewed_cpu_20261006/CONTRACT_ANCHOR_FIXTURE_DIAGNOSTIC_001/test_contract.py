"""Observer-only permission/schema/context tests; no native APIs or model."""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import unittest
import contract as C
class ContractControls(unittest.TestCase):
    def setUp(self):
        self.head='a'*40;root=C.PACKAGE;self.worker=str(C.EXPECTED/'worker.py')
        def d(role,bytes=1):return {'bytes':bytes,'sha256':'b'*64,'role':role}
        sources={str(C.EXPECTED/n):d('python_source') for n in ('contract.py','worker.py','supervisor.py','audit.py')}
        native={str(root/('prefix_library_observer.'+n)):d('native_observer_source') for n in ('c','h')};sources.update(native)
        adapter={'path':str(root/'prefix_library_observer.py'),'descriptor':d('python_source')};sources[adapter['path']]=adapter['descriptor']
        binary={'path':str(root/'native_library_observer.dll'),'descriptor':d('observer_binary')};fixture={'path':str(root/'observer_only_fixture.dll'),'descriptor':d('observer_fixture_binary')}
        observer={'kind':'ROOT_REGISTERED_NATIVE_LIBRARY_OBSERVER_V1','adapter_source':adapter,'native_source_files':native,'binary':binary,'abi':dict(C.ABI),'build_receipt':{'path':str(root/'native_library_observer_build.json'),'file_sha256':'c'*64,'body_sha256':'d'*64}}
        files={binary['path']:binary['descriptor'],fixture['path']:fixture['descriptor']}
        self.runtime=C.seal({'kind':'OBSERVER_ONLY_COMPLETE_CURRENT_RUNTIME_V1','source_commit':self.head,'files':files,'inherited65_map':{},'stdlib779_map':{},'windows_library_identities':{},'library_event_observer':observer,'stdlib_layout':{},'cache_policy':{},'initial_loaded_libraries':[binary['path']],'qualification_review':{}})
        self.config=C.seal({'kind':'REGISTERED_OBSERVER_ONLY_QUALIFICATION_V1','scope':C.SCOPE,'permission':deepcopy(C.PERMISSION),'resources':deepcopy(C.RESOURCES),'source_commit':self.head,'source_files':sources,'runtime_files':files,'observer_registration':observer,'binary_fixture':fixture,'worker_path':self.worker,'offline_environment':{'HF_HUB_OFFLINE':'1'},'native_launch_authorized':False})
        self.registration=C.seal({'kind':'ROOT_REGISTERED_OBSERVER_ONLY_LAUNCH_V1','source_commit':self.head,'config_body_sha256':self.config['sha256'],'runtime_body_sha256':self.runtime['sha256'],'source_files':sources,'runtime_files':files,'profile_sha256':C.digest({'scope':C.SCOPE,'permission':C.PERMISSION,'resources':C.RESOURCES,'ABI':C.ABI}),'worker_path':self.worker,'owner_approval':{'approved':False,'scope':C.SCOPE,'evidence_reference':''},'native_launch_authorized':False})
    def schema(self):return C._schema(self.config,self.runtime,self.registration,self.head,fixture_only=True)
    def test_separate_scope_metadata_keeps_zero_model_tensor_permission(self):
        value=self.schema();self.assertTrue(value['fixture_only']);self.assertFalse(value['native_authority_issued'])
        self.assertEqual(value['config']['permission']['allocations'],0)
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
            cf=write(config_path,self.config);rf=write(runtime_path,self.runtime);sf=write(registration_path,self.registration)
            anchors={'expected_config_file':cf,'expected_config_body':self.config['sha256'],'runtime_path':str(runtime_path),'expected_runtime_file':rf,'expected_runtime_body':self.runtime['sha256'],'registration_path':str(registration_path),'expected_registration_file':sf,'expected_registration_body':self.registration['sha256'],'expected_head':self.head}
            # Pure schema plus actual JSON/anchor readers; physical historical
            # map bytes and shared inventory are explicit fixture seams only.
            with patch.object(C,'PACKAGE',root),patch.object(C,'_shared_inventory_checks'),patch.object(C,'file_descriptor'):
                context=C._CPU_context_fixture(config_path,_capability=C._CPU_capability(),**anchors)
                self.assertTrue(C.context_value(context)['fixture_only'])
                config_path.write_bytes(config_path.read_bytes()+b' ')
                with self.assertRaisesRegex(ValueError,'external file bytes'):
                    C._CPU_context_fixture(config_path,_capability=C._CPU_capability(),**anchors)
    def test_original_opaque_CPU_context_cannot_be_forged_or_rebased(self):
        value=self.schema();value.update(config_path='CPU_ONLY',anchors={})
        context=C._CPU_context_value_fixture(value,_capability=C._CPU_capability());value['config']['scope']='changed'
        self.assertEqual(C.context_value(context)['config']['scope'],C.SCOPE)
        with self.assertRaisesRegex(ValueError,'factory context'):C.context_value(object())
        with self.assertRaisesRegex(ValueError,'private'):C._CPU_context_value_fixture(value,_capability=object())

if __name__=='__main__':unittest.main()
