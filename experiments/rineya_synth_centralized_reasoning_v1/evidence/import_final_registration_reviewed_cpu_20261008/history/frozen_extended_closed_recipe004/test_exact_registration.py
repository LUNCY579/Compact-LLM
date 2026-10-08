"""Exact d39 closed recipe joins, no grant or approved native-reader call."""
from pathlib import Path
from copy import deepcopy
from types import ModuleType
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
def module(path):m=ModuleType('CPU_source_actual_Q');m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
Q=module(PACKAGE/'import_runtime_qualification.py');D=module(PACKAGE/'import_dispatch_postrun.py')
R=module(HERE/'revise_extended.py')
def seal(v):return Q.B.parse(Q.B.canonical({**v,'sha256':Q.B.digest(v)}).encode())
def validate(recipe,config,runtime,registration,expected_selection):
    for v in (recipe,config,runtime,registration):Q.B.require(v['sha256']==Q.B.digest({k:x for k,x in v.items() if k!='sha256'}),'exactpacketbody')
    old=Q.B.sealed_json(PACKAGE/'configs/SYNTH_CONTEXT_PHASE_NO_MODEL_D001.json','53cfe48bef01d531c8c16555edc97eeebd02b76c9e5b995d20513e6b37c1b030','944db028d5c7fcc7a0210590ac551200a184e143af4973ef5c08a13faa84cc62')
    Q.B.require(config['resources']==old['resources'] and registration['profile_sha256']==Q.B.digest({'scope':Q.SCOPE,'permission':Q.PERMISSION,'resources':old['resources']}),'exactscope/resourceprofile')
    Q.B.require(config['source_commit'] is runtime['source_commit'] is registration['source_commit'] is recipe['implementation_commit'] is None,'pendingnewactualimplementationjoin')
    Q.B.require(config['source_files']==registration['source_files'] and config['runtime_files']==registration['runtime_files']==runtime['files'] and Q.B.digest(config['source_files'])==expected_selection['source_inventory_sha256'] and Q.B.digest(runtime['files'])==expected_selection['runtime_inventory_sha256'],'independentactualsource/runtimejoins')
    Q.B.require(config['native_launch_authorized'] is registration['native_launch_authorized'] is recipe['native_launch_authorized'] is False and recipe['owner_grant_reference'] is recipe['owner_receipt_reference'] is config['owner_approval'] is registration['owner_approval'] is None,'closedgrantfalse')
    Q.B.require(recipe['external_dispatch_head'] is recipe['complete_current_byte_review'] is registration['qualification_review_path'] is None,'futuredispatch/reviewnotfabricated')
    reference=runtime['metadata_references']['additional_preload'];Q.B.require(reference==recipe['passive_extended_reference'] and reference['file_sha256']==R.PASSIVE_PIN,'exactextendedpassiveref');receipt=Q.B.sealed_json(reference['path'],reference['file_sha256'],reference['body_sha256']);Q.B.require(runtime['cache_policy']['additional_preload_sources']==R.overlay(receipt),'exactsealedextendedoverlay')
    Q.B.require(set(config)==Q.CONFIG_FIELDS and set(runtime)==Q.RUNTIME_FIELDS and set(registration)==Q.REGISTRATION_FIELDS,'actualapprovedschemafieldsetclosedvalues')
    Q.import_git_exception_policy(registration['git_output_exceptions'],config['source_files'],runtime['files'],runtime['library_event_observer'],Q.B);Q._source_roster({'config':config,'runtime':runtime},Q.B,Q.QualificationBudget());return True
class Controls(unittest.TestCase):
    def setUp(self):
        self.recipe=json.loads((HERE/'EXACT_IMPORT_RUN_RECIPE004.json').read_bytes());self.config=json.loads((HERE/'IMPORT_OBSERVATION_CONFIG_004.json').read_bytes());self.runtime=json.loads((HERE/'IMPORT_OBSERVATION_RUNTIME_004.json').read_bytes());self.reg=json.loads((HERE/'IMPORT_OBSERVATION_REGISTRATION_004.json').read_bytes());self.selection=json.loads((HERE/'ACTUAL_SOURCE_RUNTIME_SELECTION003.json').read_bytes())
    def check(self):return validate(self.recipe,self.config,self.runtime,self.reg,self.selection)
    def reseal(self):
        for key in ('config','runtime','reg','recipe'):
            value=getattr(self,key);setattr(self,key,seal({k:v for k,v in value.items() if k!='sha256'}))
    def test_real_implementation_source_recipe_and_closed_future_command(self):
        self.assertTrue(self.check());self.assertEqual(len(self.config['source_files']),196);self.assertEqual(len(self.runtime['files']),8725);self.assertEqual(len(self.runtime['cache_policy']['additional_preload_sources']),11);self.assertIsNone(self.recipe['future_command']['nine_anchors']['expected_head']);self.assertIn('import_dispatch_postrun.py',self.recipe['future_command']['argv_prefix'][3]);self.assertEqual(len(self.reg['git_output_exceptions']),214)
    def test_resealed_resource_profile_or_implementation_changes_refuse(self):
        self.config['resources']['GPU_free_at_admission_GiB']=1;self.reseal()
        with self.assertRaisesRegex(ValueError,'resourceprofile'):self.check()
    def test_source_runtime_or_snapshotleaf_rebinding_refuse(self):
        p=str(PACKAGE/'import_dispatch_postrun.py');self.config['source_files'][p]['sha256']='0'*64;self.reg['source_files']=self.config['source_files'];self.reseal()
        with self.assertRaisesRegex(ValueError,'source/runtimejoins'):self.check()
    def test_no_ownergrant_or_complete_current_review_forgery(self):
        self.recipe['owner_grant_reference']={'path':'notapproved'};self.reseal()
        with self.assertRaisesRegex(ValueError,'closedgrant'):self.check()
        with self.assertRaises(ValueError):D.dispatch('notread',preserved_directory='notcreated',anchors={})
    def test_extended_preload_reference_or_cache_kind_reseal_refuse(self):
        self.runtime['cache_policy']['additional_preload_sources']['copy']['cached_value_kind']='string';self.reseal()
        with self.assertRaisesRegex(ValueError,'extendedoverlay'):self.check()
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_CLOSED_EXTENDED_PENDING_IMPLEMENTATION_IMPORT_RECIPE_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_package_source_recipe_and_Git_exception_helpers':True,'full_approved_qualifier_extended_overlay_called':False,'implementation_commit_pending':True,'native_reader_or_context_called':False,'native_grant_issued':False,'native_API_framework_called':False}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
