"""Versioned passive overlay and actual full qualifier with declared CPU readers."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1]
def module(p,name):m=ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
Q=module(HERE/'import_runtime_qualification.py','CPU_versioned_overlay_Q');F=module(PACKAGE/'artifacts/torch_import_observation_work/native_candidate/test_import_runtime_qualification.py','CPU_full_overlay_fixture')
PASSIVE=PACKAGE/'evidence/import_launch_corrections_reviewed_cpu_20261007/frozen_ram_source_fix002/PASSIVE_IMPORT_BOOTSTRAP_ASSUMPTIONS002.json'
class Controls(F.Controls):
    def extended(self):
        self.add_passive_overlay();raw=PASSIVE.read_bytes();self.assertEqual(hashlib.sha256(raw).hexdigest(),'0fe6d869bbf61a3125841164a0e11d2eb2a8ef46743f89bd7423259b02d62d91');receipt=json.loads(raw);runtime=self.value['runtime'];overlay=Q.passive_preload_overlay(receipt,helpers=self.B,budget=Q.QualificationBudget());runtime['cache_policy']['additional_preload_sources']=overlay;runtime['metadata_references']['additional_preload']={'path':str(PASSIVE),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':receipt['sha256']}
        selection=json.loads((PACKAGE/'artifacts/import_exact_registration_work/ACTUAL_SOURCE_RUNTIME_SELECTION003.json').read_bytes());self.assertEqual(selection['source_count'],196);self.value['config']['source_files']=self.value['registration']['source_files']=deepcopy(selection['source_files']);self.assertEqual(len(self.value['config']['source_files']),196)
        runtime['cache_policy']['separate_main_source']['descriptor']=self.value['config']['source_files'][self.value['config']['main_source']]
        source_recipe={'kind':'REGISTERED_IMPORT_TRANSITIVE_SOURCE_ROSTER','source_files':{p:d for p,d in self.value['config']['source_files'].items() if Path(p).is_relative_to(PACKAGE/'evidence')}};source_recipe['sha256']=self.B.digest(source_recipe);runtime['metadata_references']['source_roster']=self.reference('actual196-roster.json',source_recipe)
        def selected_descriptor(path,d,**kwargs):
            selected={**self.value['config']['source_files'],**runtime['files']};self.assertEqual(selected[str(path)],d)
        self.B.file_descriptor=selected_descriptor
        for row in overlay.values():
            self.assertEqual(runtime['files'][row['path']],row['descriptor'])
            if row['cache_path'] is not None:runtime['files'][row['cache_path']]=row['cache_descriptor']
        native=runtime['library_event_observer']['native_source_files'];target=PACKAGE/'evidence/prefix_input_owned_CPU_EXTENDED';self.value['registration']['git_output_exceptions']={str(target/'source_snapshots'/(hashlib.sha256(p.encode()).hexdigest()+Path(p).suffix)):{'role':'evidence_source_snapshot','source_path':p} for p in native};self.reseal();review={'kind':'ROOT_REVIEWED_IMPORT_RUNTIME_BYTE_ROSTER_SOURCE_CACHE_CLOSURE','source_commit':'a'*40,'source_inventory_sha256':self.B.digest(self.value['config']['source_files']),'runtime_closure_sha256':self.B.digest(runtime['files']),'rosters_sha256':self.B.digest(runtime['rosters']),'cache_policy_sha256':self.B.digest(runtime['cache_policy']),'library_search_policy_sha256':self.B.digest(runtime['library_search_policy']),'measurement_policy_sha256':self.B.digest(runtime['measurement_policy']),'measurement_grammar_sha256':self.B.digest(runtime['measurement_grammar']),'complete_current_byte_review':True};review['sha256']=self.B.digest(review);ref=self.reference('extended-review.json',review);self.value['registration'].update(qualification_review_path=ref['path'],qualification_review_file_sha256=ref['file_sha256'],qualification_review_body_sha256=ref['body_sha256']);self.reseal();return receipt
    def test_original_V1_and_full_V2_8725_native2_qualifier(self):
        prior=json.loads((PACKAGE/'evidence/psutil_bootstrap_assumptions_reviewed_cpu_20261007/PASSIVE_PSUTIL_BOOTSTRAP_ASSUMPTIONS001.json').read_bytes());self.assertEqual(len(Q.passive_preload_overlay(prior,helpers=self.B)),10);self.extended();result=Q._qualify(self.value,Q.QualificationBudget(),self.B);self.assertFalse(result['current_module_execution_proven']);self.assertEqual(len(self.value['runtime']['files']),8725);self.assertEqual(len(self.value['runtime']['cache_policy']['additional_preload_sources']),11);self.assertEqual(len(self.value['config']['source_files']),len(self.value['registration']['source_files']))
    def test_wrong_kind_version_or_predecessor_reference_refuse(self):
        receipt=json.loads(PASSIVE.read_bytes())
        for key,value in [('kind','unknown'),('schema_version',True),('predecessor_reference',{'path':'rebound'})]:
            bad=deepcopy(receipt);bad[key]=value
            with self.subTest(key=key),self.assertRaises(ValueError):Q.passive_preload_overlay(bad,helpers=self.B)
    def test_arbitrary_added_module_or_copy_byte_rebinding_refuse(self):
        receipt=json.loads(PASSIVE.read_bytes());bad=deepcopy(receipt);bad['source_modules']['foreign']=bad['source_modules'].pop('copy')
        with self.assertRaises(ValueError):Q.passive_preload_overlay(bad,helpers=self.B)
        bad=deepcopy(receipt);bad['source_modules']['copy']['descriptor']['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'copy source bytes'):Q.passive_preload_overlay(bad,helpers=self.B)
    def test_changed_original_baseline_or_cache_execution_refuse(self):
        receipt=json.loads(PASSIVE.read_bytes());bad=deepcopy(receipt);bad['source_derived_baseline_expectation'].pop()
        with self.assertRaisesRegex(ValueError,'expectations unchanged'):Q.passive_preload_overlay(bad,helpers=self.B)
        bad=deepcopy(receipt);copy=next(p for p,d in bad['additional_cache_candidates'].items() if d['source_path'].endswith('copy.py'));bad['additional_cache_candidates'][copy]['cache_executed']=True
        with self.assertRaisesRegex(ValueError,'copy cache bytes'):Q.passive_preload_overlay(bad,helpers=self.B)
    def test_resealed_unknown_native_snapshot_stays_closed(self):
        self.extended();sources=self.value['config']['source_files'];source=str(PACKAGE/'foreign.c');sources[source]={'bytes':1,'sha256':'a'*64,'role':'native_observer_source'};target=PACKAGE/'evidence/prefix_input_owned_CPU_EXTENDED/source_snapshots'/(hashlib.sha256(source.encode()).hexdigest()+'.c');self.value['registration']['git_output_exceptions'][str(target)]={'role':'evidence_source_snapshot','source_path':source};self.reseal()
        with self.assertRaises(ValueError):Q._qualify(self.value,Q.QualificationBudget(),self.B)
if __name__=='__main__':
    names=('test_original_V1_and_full_V2_8725_native2_qualifier','test_wrong_kind_version_or_predecessor_reference_refuse','test_arbitrary_added_module_or_copy_byte_rebinding_refuse','test_changed_original_baseline_or_cache_execution_refuse','test_resealed_unknown_native_snapshot_stays_closed');before=hashlib.sha256((HERE/'import_runtime_qualification.py').read_bytes()).hexdigest();result=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(Controls(n) for n in names));assert hashlib.sha256((HERE/'import_runtime_qualification.py').read_bytes()).hexdigest()==before;v={'kind':'CPU_EXACT_VERSION2_COPY_OVERLAY_QUALIFIER_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'full_actual_qualifier_V2_positive':True,'sourcehash_and_roster_readers_CPU_injected':True,'native_context_or_authority_issued':False,'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'import_runtime_qualification.py',Path(__file__).resolve())}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
