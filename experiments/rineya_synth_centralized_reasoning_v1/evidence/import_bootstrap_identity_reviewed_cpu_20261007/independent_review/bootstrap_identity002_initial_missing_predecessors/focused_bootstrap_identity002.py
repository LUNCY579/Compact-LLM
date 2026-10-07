"""Independent current module/cache/frame metadata checks on frozen source."""
from pathlib import Path
from types import ModuleType
from unittest.mock import patch
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_bootstrap_identity002';TEST=FROZEN/'test_import_bootstrap.py';FILES=[p for p in FROZEN.rglob('*') if p.is_file()]
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();assert BEFORE[str(FROZEN/'import_bootstrap.py')]['sha256']=='749b093fb05fd58f37fdc3109196e60494ac70d46fc97aaf89c9e0e366b43c5b';assert BEFORE[str(TEST)]['sha256']=='fb83fd13f0e1f3cb3157ba225041dbdb59616baef4c4123d29124dd6a8bf1601'
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
trap=Trap();sys.meta_path.insert(0,trap);T=ModuleType('independent_bootstrap_fixture');T.__file__=str(TEST);exec(compile(TEST.read_bytes(),str(TEST),'exec'),T.__dict__);Base=T.Controls;H=T.H;C=T.C;DETAILS={}
class Independent(Base):
    def test_actual_cache_code_source_shape_and_historical_conservation(self):
        self.test_actual_additive_source_cache_code_equivalence_historical_map_unchanged();self.assertEqual(C.code_fingerprint.__code__.co_name,'code_fingerprint');DETAILS['actual_inert_cache_code_source_equivalence']=True
    def test_wrong_code_history_missing_population_spec_and_cache_field_refuse(self):
        self.test_wrong_cache_source_code_unknown_and_history_rejected();original=self.module.__spec__.origin;(self.root/'different.py').write_bytes(b'VALUE=7\n');self.module.__spec__.origin=str(self.root/'different.py')
        with self.assertRaises(ValueError):H.verify_additional_preloads(self.value,C,{'CPU_additional':self.module})
        self.module.__spec__.origin=original;self.module.__cached__=None
        with self.assertRaisesRegex(ValueError,'cache field shape'):H.verify_additional_preloads(self.value,C,{'CPU_additional':self.module})
    def test_actual_authored_main_frame_wrong_namespace_and_rebound_code_refuse(self):
        self.test_own_current_main_frame_source_equivalence_and_wrong_globals();p=self.root/'different_main.py';source=b"def probe(helper,contract,descriptor,path):\n import sys\n return helper.verify_current_main(globals(),sys.modules['__main__'],sys._getframe(),descriptor,contract,path)\n";p.write_bytes(source);m=ModuleType('__main__');m.__file__=str(p);m.__spec__=None;m.__cached__=None;exec(compile(source,str(p),'exec'),m.__dict__);descriptor=T.desc(p,'python_source');p.write_bytes(source.replace(b'return helper',b'return 1 if False else helper'));descriptor=T.desc(p,'python_source')
        with patch.dict(sys.modules,{'__main__':m}),self.assertRaisesRegex(ValueError,'matches selected source'):m.probe(H,C,descriptor,p)
        DETAILS['authored_current_frame_probe_not_actual_worker']=True
    def test_module_spec_loader_and_dual_package_root_identity_refuse(self):
        self.test_wrong_module_spec_name_or_foreign_loader_refused();Base.setUp(self)
        foreign=T.importlib.machinery.SourceFileLoader('CPU_additional',str(self.path));self.module.__loader__=foreign
        with self.assertRaisesRegex(ValueError,'loader identity'):H.verify_additional_preloads(self.value,C,{'CPU_additional':self.module})
        self.module.__spec__.loader=foreign;self.module.__path__=[str(self.root)]
        with self.assertRaisesRegex(ValueError,'unregistered package path'):H.verify_additional_preloads(self.value,C,{'CPU_additional':self.module})
        self.module.__spec__.submodule_search_locations=[str(self.root)];self.module.__path__=[str(self.root)];self.assertTrue(H.verify_additional_preloads(self.value,C,{'CPU_additional':self.module})['CPU_additional']['current_state_source_verified'])
        self.module.__path__=[]
        with self.assertRaisesRegex(ValueError,'dual roots'):H.verify_additional_preloads(self.value,C,{'CPU_additional':self.module})
    def test_closed_native_preload_unknown_finder_and_source_qualifier_order(self):
        self.test_no_native_preload_from_CPU_context_and_same_source_reader_fullqualifier_seam();finder=H._FinitePreludeFinder(self.value,C)
        with self.assertRaisesRegex(ValueError,'unknown additional'):finder.find_spec('CPU_undeclared_bootstrap')
        self.assertIs(finder.find_spec('sys'),None);DETAILS['native_preload_actual_refusal']=True
if __name__=='__main__':
    names=('test_actual_cache_code_source_shape_and_historical_conservation','test_wrong_code_history_missing_population_spec_and_cache_field_refuse','test_actual_authored_main_frame_wrong_namespace_and_rebound_code_refuse','test_closed_native_preload_unknown_finder_and_source_qualifier_order','test_module_spec_loader_and_dual_package_root_identity_refuse');r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Independent(n) for n in names));sys.meta_path.remove(trap);after=pins();receipt={'kind':'INDEPENDENT_FROZEN_IMPORT_BOOTSTRAP_IDENTITY002_REVIEW','tests_run':r.testsRun,'failures':len(r.failures),'errors':len(r.errors),'all_controls_completed':r.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'frozen_sources_unchanged':BEFORE==after,'details':DETAILS,'actual_inert_source_cache_complete_code_metadata_checked':True,'additional_module_namespace_explicit_fake_ModuleType':True,'own_main_authored_tiny_function_frame_only':True,'actual_worker_main_or_psutil_preload_executed':False,'full_generic_qualifier_or_runtime_hash_executed':False,'original63_59_maps_not_mutated':True,'native_API_or_framework_called':False,'native_authority_issued':False};receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_BOOTSTRAP_IDENTITY002.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if r.wasSuccessful() and BEFORE==after else 1)
