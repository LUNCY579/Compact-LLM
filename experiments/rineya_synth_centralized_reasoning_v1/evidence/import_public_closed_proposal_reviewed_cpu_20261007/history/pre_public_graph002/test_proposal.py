"""Closed public proposal joins and adversarial reseals; no approved context."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent
def load(p,name):
    m=ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
M=load(HERE/'construct_proposal.py','CPU_closed_public_constructor');I=load(HERE/'build_inputs.py','CPU_public_constructor_inputs')
class Controls(unittest.TestCase):
    def setUp(self):
        self.inputs,recipe=I.inputs();self.proposal=M.build(**self.inputs);self.expected_sources=M.digest(self.inputs['source_files']);self.expected_runtime=M.digest(self.proposal['runtime']['files'])
    def check(self):return M.validate(self.proposal,source_plan=self.inputs['source_plan'],expected_source_map_sha256=self.expected_sources,expected_runtime_map_sha256=self.expected_runtime)
    def reseal(self):
        for name in ('config','runtime'):
            v=self.proposal[name];self.proposal[name]=M.seal({k:x for k,x in v.items() if k!='sha256'})
        g=self.proposal['registration'];g.update(config_body_sha256=self.proposal['config']['sha256'],runtime_body_sha256=self.proposal['runtime']['sha256']);self.proposal['registration']=M.seal({k:x for k,x in g.items() if k!='sha256'});self.proposal=M.seal({k:x for k,x in self.proposal.items() if k!='sha256'})
    def test_complete_closed_schema_with_full_passive_map_joins(self):
        result=self.check();self.assertFalse(result['native_launch_authorized']);self.assertEqual(len(self.proposal['runtime']['files']),8724);self.assertEqual(len(self.proposal['runtime']['measurement_policy']['initial_loaded_libraries']),39);self.assertEqual(len(self.proposal['runtime']['cache_policy']['preloaded_module_cache_pins']),59);self.assertIsNone(self.proposal['implementation_commit']);self.assertIsNone(self.proposal['all_current_byte_qualification_review'])
        Q=I.module(I.PACKAGE/'artifacts/torch_import_observation_work/native_candidate/import_runtime_qualification.py');self.assertEqual(set(self.proposal['config']),Q.CONFIG_FIELDS);self.assertEqual(set(self.proposal['runtime']),Q.RUNTIME_FIELDS);self.assertEqual(set(self.proposal['registration']),Q.REGISTRATION_FIELDS)
    def test_resealed_target_source_descriptor_rebinding_refused(self):
        p=next(row['target_path'] for row in self.inputs['source_plan']['sources'].values());self.proposal['config']['source_files'][p]['sha256']='0'*64;self.proposal['registration']['source_files']=self.proposal['config']['source_files'];self.reseal()
        with self.assertRaisesRegex(ValueError,'externalexpectedsource'):self.check()
    def test_resealed_owner_grant_or_complete_review_forgery_refused(self):
        original=deepcopy(self.proposal);self.proposal['owner_grant_reference']={'path':'fake','body_sha256':'a'*64};self.reseal()
        with self.assertRaisesRegex(ValueError,'closedgrant'):self.check()
        self.proposal=original;self.proposal['all_current_byte_qualification_review']={'complete_current_byte_review':True};self.reseal()
        with self.assertRaisesRegex(ValueError,'cannotbeforged'):self.check()
    def test_resealed_runtime_and_cache_reference_change_refused(self):
        p=next(iter(self.proposal['runtime']['files']));self.proposal['runtime']['files'][p]['sha256']='0'*64;self.proposal['config']['runtime_files']=self.proposal['registration']['runtime_files']=self.proposal['runtime']['files'];self.reseal()
        with self.assertRaisesRegex(ValueError,'externalexpectedsource/runtime'):self.check()
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_CLOSED_PUBLIC_IMPORT_NATIVE_SCHEMA_CONSTRUCTION_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_Q_fieldsets_checked':True,'native_qualifier_called_on_false_grant':False,'native_context_minted':False,'native_or_framework_called':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'construct_proposal.py',HERE/'build_inputs.py',Path(__file__).resolve())}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
