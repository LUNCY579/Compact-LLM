"""Exact inert observer registration and captured source/context binding.

No DLL is loaded. No compiler, ABI export, native observer/WinAPI or framework
is invoked. Historical65/779/Git/layout remain explicit OuterContextFixture seams.
"""
from copy import deepcopy
from pathlib import Path
from types import ModuleType
import builtins,hashlib,tempfile,types,unittest
from unittest.mock import patch
from outer_context_fixture import OuterContextFixture,HERE,descriptor


class ObserverRegistrationControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_observer_registration_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.f=OuterContextFixture(self.temp.name);self.C=self.f.C
        self.old65=deepcopy(self.f.runtime['inherited65_map']);self.old779=deepcopy(self.f.runtime['stdlib779_map'])
        self.attempts=[];original=builtins.__import__
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'ctypes','torch','psutil','native_worker','native_process','native_nvml','transformers','safetensors','numpy'}:
                self.attempts.append(name);raise AssertionError('native/framework import forbidden')
            return original(name,*args,**kwargs)
        trap=patch.object(builtins,'__import__',side_effect=imports);trap.start();self.addCleanup(trap.stop)
        self.addCleanup(lambda:self.assertEqual(self.attempts,[]))
        self.adapter=self.f.package/'prefix_library_observer.py'
        raw=b'class NativeLibraryObserver:\n    def __init__(self,binding): self.binding=binding\n    def begin(self): return {}\n    def events(self): return {}\n    def close(self): return {}\n    def abi_binding(self): return self.binding\n    def bootstrap(self,stage): return {}\n    def normalize_library_path(self,path): return path\n'
        self.adapter.write_bytes(raw);self.f.sources[str(self.adapter)]=descriptor(raw,'python_source')
        native={}
        for suffix in ('c','h'):
            path=self.f.package/('prefix_library_observer.'+suffix);raw=('/* inert selected source '+suffix+' */\n').encode();path.write_bytes(raw)
            native[str(path)]=descriptor(raw,'native_observer_source');self.f.sources[str(path)]=deepcopy(native[str(path)])
        self.dll=self.f.package/'native_library_observer.dll';raw=b'INERT CPU DLL TEXT; NEVER LOAD\n';self.dll.write_bytes(raw)
        self.binary={'path':str(self.dll),'descriptor':descriptor(raw,'observer_binary')}
        self.f.runtime=deepcopy(self.f.runtime);self.f.runtime['files'][str(self.dll)]=deepcopy(self.binary['descriptor'])
        base=str(self.f.base/'python.exe');provider=str(self.f.package/'prefix_library_policy.py')
        self.f.runtime['library_search_policy']['enforced_boundary']={'kind':'FINITE_WINDOWS_DLL_BOUNDARY_V2',
            'search_directories':[str(self.f.base),str(self.f.package)],'system_directory':str(self.f.base),
            'library_files':{base:deepcopy(self.f.inert_files[base]),str(self.dll):deepcopy(self.binary['descriptor'])},
            'initial_loaded_libraries':[base,str(self.dll)],'extension_transitions':{},'framework_transition':[],
            'provider_source':{'path':provider,'descriptor':deepcopy(self.f.sources[provider])}}
        self.receipt_path=self.f.package/'native_library_observer_build.json'
        self.body={'kind':'NATIVE_LIBRARY_OBSERVER_BUILD_V1','source_commit':'a'*40,'source_inputs_committed':False,
            'native_source_files':deepcopy(native),'binary':deepcopy(self.binary),'abi':deepcopy(self.C.LIBRARY_OBSERVER_ABI),
            'toolchain':{'compiler_path':str(self.f.root/'MSVC/cl.exe'),'compiler_version':'14.44.35207','compiler_file_sha256':'c'*64,
                         'linker_path':str(self.f.root/'MSVC/link.exe'),'linker_version':'14.44.35207','linker_file_sha256':'d'*64},
            'options':{'compile':['/TC','/O2','/W4','/WX','/GS-','/Zl'],
                       'link':['/DLL','/NOENTRY','/NODEFAULTLIB','/MACHINE:X64']+['/EXPORT:'+x for x in ('plobs_abi_version','plobs_start','plobs_snapshot','plobs_copy_rows','plobs_stop')]},
            'architecture':'x64','metadata_only':True,'native_PASS':False,'native_authority_issued':False}
        self.registration={'kind':'ROOT_REGISTERED_NATIVE_LIBRARY_OBSERVER_V1','adapter_source':{'path':str(self.adapter),'descriptor':deepcopy(self.f.sources[str(self.adapter)])},
            'native_source_files':native,'binary':deepcopy(self.binary),'abi':deepcopy(self.C.LIBRARY_OBSERVER_ABI),'build_receipt':None}
        self.f.runtime['library_event_observer']=self.registration;self.reseal()
    def reseal(self):
        f=self.f;C=self.C;self.body=f._seal({k:v for k,v in self.body.items() if k!='sha256'})
        raw=f._raw(self.body);self.receipt_path.write_bytes(raw);f.sources[str(self.receipt_path)]=descriptor(raw,'source_json')
        self.registration['build_receipt']={'path':str(self.receipt_path),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':self.body['sha256']}
        f.runtime=f._seal({k:v for k,v in f.runtime.items() if k!='sha256'})
        self.registration=f.runtime['library_event_observer']
        f.config={**f.config,'source_files':deepcopy(f.sources),'runtime_files':deepcopy(f.runtime['files'])};f.config=f._seal({k:v for k,v in f.config.items() if k!='sha256'})
        f.review.update(source_inventory_sha256=C.digest(f.sources),runtime_closure_sha256=C.digest(f.runtime['files']),
            library_search_policy_sha256=C.digest(f.runtime['library_search_policy']));f.review=f._seal({k:v for k,v in f.review.items() if k!='sha256'})
        f.registration.update(config_body_sha256=f.config['sha256'],runtime_body_sha256=f.runtime['sha256'],source_files=deepcopy(f.sources),runtime_files=deepcopy(f.runtime['files']),
            qualification_review_file_sha256=f.anchor(f.review_path,f.review)['file_sha256'],qualification_review_body_sha256=f.review['sha256']);f.registration=f._seal({k:v for k,v in f.registration.items() if k!='sha256'})
        for path,value in ((f.config_path,f.config),(f.runtime_path,f.runtime),(f.review_path,f.review),(f.registration_path,f.registration)):
            path.write_bytes(f._raw(value));f.documents[str(path)]=value
        f.anchors.update(expected_config_file=f.anchor(f.config_path,f.config)['file_sha256'],expected_config_body=f.config['sha256'],
            expected_runtime_file=f.anchor(f.runtime_path,f.runtime)['file_sha256'],expected_runtime_body=f.runtime['sha256'],
            expected_registration_file=f.anchor(f.registration_path,f.registration)['file_sha256'],expected_registration_body=f.registration['sha256'])
    def context(self):
        with self.f.activate(),patch.object(self.C,'package_source_roster',lambda selected=None:set(self.f.sources)):
            return self.f.context()
    def schema(self):
        with self.f.activate():return self.C.library_event_observer_registration(self.f.config,self.f.runtime)
    def reject(self,pattern):
        self.reseal()
        with self.assertRaisesRegex(ValueError,pattern):self.context()
    def binding(self):
        context=self.context();module=ModuleType('CPU_selected_observer');module.__file__=str(self.adapter)
        exec(compile(self.adapter.read_bytes(),str(self.adapter),'exec'),module.__dict__)
        module.__compiled_source_descriptor__=deepcopy(self.registration['adapter_source']['descriptor'])
        binding={**deepcopy(self.binary),'abi':deepcopy(self.C.LIBRARY_OBSERVER_ABI),'dll_handle':101}
        observer=module.NativeLibraryObserver(binding);cap=self.C._mint_CPU_native_context_fixture()
        with self.f.activate():self.C._CPU_register_library_event_observer_fixture(context,module,observer,binding,_capability=cap)
        return context,module,observer,binding,cap
    def test_registration_real_inert_reader_maps_and_original65_779_preserved(self):
        value=self.C.context_value(self.context())
        self.assertEqual(value['runtime']['inherited65_map'],self.old65);self.assertEqual(value['runtime']['stdlib779_map'],self.old779)
        self.assertEqual(value['runtime']['library_event_observer']['abi'],self.C.LIBRARY_OBSERVER_ABI)
        self.assertIn(str(self.dll),self.f.actual_file_checks);self.assertTrue(value['fixture_only']);self.assertFalse(value['native_PASS'])
    def test_optional_absence_keeps_legacy_metadata_compatible(self):
        del self.f.runtime['library_event_observer'];del self.f.runtime['files'][str(self.dll)]
        del self.f.runtime['library_search_policy']['enforced_boundary']['library_files'][str(self.dll)]
        self.f.runtime['library_search_policy']['enforced_boundary']['initial_loaded_libraries'].remove(str(self.dll))
        for path in list(self.f.sources):
            if path.endswith(('.c','.h','native_library_observer_build.json')):del self.f.sources[path]
        self.assertIsNone(self.C.library_event_observer_registration(self.f.config,self.f.runtime))
    def test_same_count_binary_descriptor_substitution_refused(self):
        self.registration['binary']['descriptor']['sha256']='0'*64;self.reject('disjoint exact runtime join|source/runtime')
    def test_unknown_owned_DLL_path_and_general_package_role_refused(self):
        self.registration['binary']['path']=str(self.f.package/'other.dll');self.reject('owned observer DLL singleton')
    def test_native_source_omission_or_role_substitution_refused(self):
        del self.registration['native_source_files'][str(self.f.package/'prefix_library_observer.h')];self.reject('two native observer source leaves')
    def test_resealed_ABI_boolean_or_architecture_change_refused(self):
        self.registration['abi']['version']=True;self.reject('reviewed native observer ABI')
    def test_build_binary_metadata_mismatch_refused_beyond_hashes(self):
        self.body['binary']['descriptor']['sha256']='0'*64;self.reject('build receipt exact source/binary/ABI joins')
    def test_unreviewed_compile_link_export_options_refused(self):
        self.body['options']['link'].append('/EXPORT:arbitrary_hook');self.reject('linker exports/options')
    def test_sourcecommit_is_build_ancestor_not_future_launch_identity(self):
        self.body['source_commit']='b'*40;self.reseal();self.assertEqual(self.schema()['kind'],'ROOT_REGISTERED_NATIVE_LIBRARY_OBSERVER_V1')
    def test_CPU_captured_module_object_methods_ABI_context_binding(self):
        context,module,observer,binding,cap=self.binding()
        result=self.C._CPU_verify_library_event_observer_fixture(observer,_capability=cap)
        self.assertTrue(result['fixture_only']);self.assertFalse(result['native_PASS']);self.assertFalse(result['native_authority_issued'])
        observer.binding={**binding,'dll_handle':102}
        with self.assertRaisesRegex(ValueError,'ABI handle retained'):self.C._CPU_verify_library_event_observer_fixture(observer,_capability=cap)
    def test_foreign_instance_or_forged_capability_cannot_supply_binding(self):
        context,module,observer,binding,cap=self.binding()
        with self.assertRaisesRegex(ValueError,'binding required'):
            self.C._CPU_verify_library_event_observer_fixture(module.NativeLibraryObserver(binding),_capability=cap)
        with self.assertRaises(ValueError):self.C._CPU_verify_library_event_observer_fixture(observer,_capability=object())
    def test_method_swap_and_class_marker_drift_refused(self):
        context,module,observer,binding,cap=self.binding()
        observer.normalize_library_path=types.MethodType(lambda self,path:path,observer)
        with self.assertRaisesRegex(ValueError,'observer methods retained'):self.C._CPU_verify_library_event_observer_fixture(observer,_capability=cap)
    def test_native_gate_precedes_metadata_and_CPUbinding_cannot_promote(self):
        context,module,observer,binding,cap=self.binding()
        with patch.object(self.C,'read_bytes',side_effect=AssertionError('no read before stage')):
            with self.assertRaisesRegex(ValueError,'stage has no native authority'):self.C.verify_native_library_event_observer(observer)
        with patch.object(self.C,'stage_native_gate'):
            with self.assertRaisesRegex(ValueError,'CPU observer binding has no native provider authority'):self.C.verify_native_library_event_observer(observer)
            with self.assertRaisesRegex(ValueError,'CPU context cannot register native observer'):
                self.C._register_library_event_observer(context,module,observer,binding)


if __name__=='__main__':unittest.main()
