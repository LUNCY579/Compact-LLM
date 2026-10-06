"""Actual selected adapter + C-owned binding + provider guard CPU composition."""
from pathlib import Path
from copy import deepcopy
from types import ModuleType
from unittest.mock import patch
import unittest
import test_library_observer_registration as R
import test_library_observer_adapter as A
HERE=Path(__file__).resolve().parent
class NativeObserverBindingControls(unittest.TestCase):
    def setUp(self):
        self.case=R.ObserverRegistrationControls();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.f=self.case.f;self.C=self.case.C
        raw=(HERE/'prefix_library_observer.py').read_bytes();self.case.adapter.write_bytes(raw)
        from outer_context_fixture import descriptor
        pin=descriptor(raw,'python_source');self.f.sources[str(self.case.adapter)]=pin
        self.f.runtime['library_event_observer']['adapter_source']['descriptor']=deepcopy(pin)
        self.case.reseal();self.context=self.case.context()
        self.module=ModuleType('CPU_actual_selected_adapter');self.module.__file__=str(self.case.adapter)
        exec(compile(raw,str(self.case.adapter),'exec'),self.module.__dict__)
        self.module.__compiled_source_descriptor__=deepcopy(pin)
        self.t=A.Transport();self.t.binding={'path':str(self.case.dll),'descriptor':deepcopy(self.case.binary['descriptor']),'abi':dict(self.C.LIBRARY_OBSERVER_ABI),'dll_handle':101}
        self.observer=self.module._CPU_observer_fixture(self.context,self.C,self.t,_capability=self.module._CPU_capability())
        self.cap=self.C._mint_CPU_native_context_fixture()
        with self.f.activate():self.C._CPU_register_library_event_observer_fixture(self.context,self.module,self.observer,self.observer.abi_binding(),_capability=self.cap)
        self.observer.check=lambda:self.C._CPU_verify_library_event_observer_fixture(self.observer,_capability=self.cap)
    def test_actual_selected_adapter_original_context_binding_begin_final_drain(self):
        self.observer.begin();self.assertEqual(self.observer.events()['events'],[])
        result=self.observer.close();self.assertTrue(result['complete'])
        binding=self.C._CPU_verify_library_event_observer_fixture(self.observer,_capability=self.cap)
        self.assertTrue(binding['fixture_only']);self.assertFalse(binding['native_PASS'])
    def test_actual_adapter_normalization_method_swap_is_rejected_by_C_capture(self):
        from types import MethodType
        self.observer.normalize_library_path=MethodType(lambda self,raw:raw,self.observer)
        with self.assertRaisesRegex(ValueError,'original observer methods retained'):
            self.C._CPU_verify_library_event_observer_fixture(self.observer,_capability=self.cap)
    def test_actual_main_factory_captures_original_observer_hooks_without_public_suppliers(self):
        from test_full_native_cpu_workflow import compiled
        W=compiled(self.f.package/'prefix_input_owned_worker.py');W.C=self.C
        called=[]
        class Loader:
            def install(_,context,contract,**hooks):called.append((context,contract,hooks));return 'CPU_FINDER'
        with self.f.activate():
            install,observer,adapter=W._observer_install_factory(self.context,{},Loader(),_CPU_capability=self.cap,transport=self.t)
            self.assertEqual(install(self.context,self.C),'CPU_FINDER')
        ctx,contract,hooks=called[0]
        self.assertIs(ctx,self.context);self.assertIs(contract,self.C)
        self.assertIs(hooks['library_event_observer'],observer)
        self.assertIs(hooks['library_bootstrap_verifier'].__self__,observer)
        self.assertIs(type(observer),adapter.NativeLibraryObserver)
        self.assertFalse(self.C._CPU_verify_library_event_observer_fixture(observer,_capability=self.cap)['native_PASS'])
        with self.assertRaisesRegex(ValueError,'same original main'):install(object(),self.C)
    def test_actual_compiled_candidate_and_toolchain_receipt_pass_metadata_joins_without_load(self):
        from outer_context_fixture import descriptor
        import json
        directory=HERE/'observer_source_work'
        receipt=json.loads((directory/'native_library_observer_build.json').read_text(encoding='utf-8'))
        raw=(directory/'native_build_001/native_library_observer.dll').read_bytes()
        self.case.dll.write_bytes(raw);binary={'path':str(self.case.dll),'descriptor':descriptor(raw,'observer_binary')}
        self.f.runtime['files'][str(self.case.dll)]=deepcopy(binary['descriptor'])
        self.f.runtime['library_search_policy']['enforced_boundary']['library_files'][str(self.case.dll)]=deepcopy(binary['descriptor'])
        native={}
        for name in ('prefix_library_observer.c','prefix_library_observer.h'):
            target=self.f.package/name;data=(directory/name).read_bytes();target.write_bytes(data)
            native[str(target)]=descriptor(data,'native_observer_source');self.f.sources[str(target)]=native[str(target)]
        receipt['native_source_files']=native;receipt['binary']=binary;receipt['sha256']=self.C.digest({k:v for k,v in receipt.items() if k!='sha256'})
        path=self.f.package/'native_library_observer_build.json';path.write_bytes((self.C.canonical(receipt)+'\n').encode('ascii'))
        pin=descriptor(path.read_bytes(),'source_json');self.f.sources[str(path)]=pin
        self.f.runtime['library_event_observer'].update(native_source_files=native,binary=binary,
            build_receipt={'path':str(path),'file_sha256':pin['sha256'],'body_sha256':receipt['sha256']})
        self.case.body=receipt;self.case.registration=self.f.runtime['library_event_observer']
        self.case.reseal();value=self.C.context_value(self.case.context())
        self.assertTrue(value['fixture_only']);self.assertFalse(value['native_PASS'])
        self.assertEqual(value['runtime']['library_event_observer']['binary']['descriptor']['bytes'],5632)
        self.assertEqual(value['runtime']['library_event_observer']['abi'],self.C.LIBRARY_OBSERVER_ABI)
    def test_actual_adapter_public_native_mode_and_foreign_instance_refused(self):
        with patch.object(self.C,'stage_native_gate'):
            with self.assertRaisesRegex(ValueError,'CPU observer binding has no native provider authority'):
                self.C.verify_native_library_event_observer(self.observer)
        other=self.module._CPU_observer_fixture(self.context,self.C,self.t,_capability=self.module._CPU_capability())
        with self.assertRaisesRegex(ValueError,'binding required'):
            self.C._CPU_verify_library_event_observer_fixture(other,_capability=self.cap)

if __name__=='__main__':unittest.main()
