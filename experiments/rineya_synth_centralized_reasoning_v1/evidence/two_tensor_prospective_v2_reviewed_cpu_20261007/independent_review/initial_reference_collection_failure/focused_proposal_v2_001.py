"""Independent byte/source/map/dose audit of one closed proposal packet."""
from pathlib import Path
import ast, copy, hashlib, json, sys, types, unittest
HERE=Path(__file__).resolve().parent;WORK=HERE.parent;PACKET=WORK/'TWO_TENSOR_PROSPECTIVE_PROPOSAL_V2_001.json';TEST=WORK/'test_two_tensor_proposal.py';PACKAGE=next(p for p in WORK.parents if p.name=='rineya_synth_centralized_reasoning_v1')
if hashlib.sha256(PACKET.read_bytes()).hexdigest()!='16025748dd613931ca4ef8c6a03f831cd7395627e3053d6eca90a469fc2efb51':raise ValueError('exact packet bytes')
if hashlib.sha256(TEST.read_bytes()).hexdigest()!='f83f676bf577c4d186bbbc8ae81df05a6a01a3a062fdfc04d7aa1a2ea6c92c92':raise ValueError('exact wrapper bytes')
def load(path,name):
    m=types.ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker'):raise AssertionError('native import forbidden '+fullname)
TRAP=Trap();sys.meta_path.insert(0,TRAP);V=load(TEST,'independent_proposal_wrapper');C=V.C;ORIGINAL=C.parse(PACKET.read_bytes());CLOSED=ORIGINAL['closed_source_packet'];CORE=PACKAGE/'evidence/prefix_input_core_reviewed_cpu_20261005/contract.py';INPUT=PACKAGE/'evidence/exception_frame_helper_reviewed_cpu_20261005/root/ROOT_FAILURE_PREFIX_INPUT_001.json'
FILES={PACKET,TEST,CORE,INPUT,Path(C.__file__)}
for row in CLOSED['source_candidates'].values():FILES.add(Path(row['path']))
FILES.update(Path(p) for p in ORIGINAL['selected_current_helpers'])
for row in ORIGINAL['observed009_selection'].values():
    if type(row) is dict and 'path' in row:FILES.add(Path(row['path']))
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(FILES)}
BEFORE=pins();DETAILS={}
EXPECTED={'prefix_input_owned_worker.py':'6cd0a9fd598de3fc8bacc494eb67189f7bfa9963c280100d0f396ee315cc1828','prefix_native_pipeline.py':'9efb6f15444a6449734fe85d5eddd72d3cefb2a76c0b7af46264b44ed68cbb2b','prefix_tensor_contract.py':'0d12fd01bb1f06c29054943a3a114efc6eac9619c33989e5ce030d97a3cf4263','prefix_registered_loader.py':'62c25f6cf5f843a08463dbb7c3c573f6a038f036917fc4130da0aab5d4b7e0c5','prefix_input_owned_auditor.py':'985fed33d442b83c69db9bd919865537dd5427f0e690dec934f017ed8e181d4e','prefix_input_owned_supervisor.py':'5980cafad63ce1d9c4cd3c7e598a3855276d28343c3e67841c7a881b9f3b3c22'}
def literal_assignment(tree,name):
    node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets));return ast.literal_eval(node.value)
def literal_dict_fields(function,name):
    node=next(n for n in function.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets));return [ast.literal_eval(k) for k in node.value.keys]
def reseal(p):return V.seal(p)
class Controls(unittest.TestCase):
    def test_independent_current_source_input_maps_and_closed_flags(self):
        p=ORIGINAL;self.assertEqual(set(CLOSED['source_candidates']),set(EXPECTED))
        for name,row in CLOSED['source_candidates'].items():self.assertEqual(row['sha256'],EXPECTED[name]);self.assertEqual(BEFORE[row['path']],{'bytes':row['bytes'],'sha256':row['sha256']})
        for path,descriptor in p['selected_current_helpers'].items():self.assertEqual(BEFORE[path],{k:descriptor[k] for k in ('bytes','sha256')});self.assertEqual(descriptor['role'],'python_source')
        self.assertEqual(hashlib.sha256(INPUT.read_bytes()).hexdigest(),C.INPUT_FILE);iv=C.parse(INPUT.read_bytes());self.assertEqual(iv['sha256'],C.INPUT_BODY);self.assertEqual(len(iv['prefix_ids']),422);self.assertEqual(iv['attention_mask'],[1]*422);self.assertEqual(C.digest(iv['prefix_ids']),C.PREFIX_SHA)
        ref=p['observed009_selection']['observation_reference'];observation=C.sealed_json(ref['path'],ref['file_sha256'],ref['body_sha256']);rows=observation['bootstrap']['tensor_prelude']['modules_at_guard_boundary'];modules={};caches={};aliases={};built=[]
        for row in rows:
            if row['name']=='__main__':continue
            if row.get('classification')=='stdlib_class_alias':aliases[row['name']]=row['alias_metadata'];continue
            if row['origin'] in ('built-in','frozen'):built.append(row['name']);continue
            cache=row['cached'];modules[row['name']]={'path':row['origin'],'source_sha256':row['source_sha256'],'cache_path':None if cache is None else cache['path'],'cache_sha256':None if cache is None else cache['sha256'],'cached_attribute_present':row['cached_attribute_present'],'cached_value_kind':row['cached_value_kind']}
            if cache and cache['path'] is not None:self.assertTrue(cache['source_equivalence']);caches[cache['path']]={'bytes':cache['bytes'],'sha256':cache['sha256'],'role':'qualified_bytecode_cache'}
        self.assertEqual(p['observed009_selection']['maps'],{'preloaded_bootstrap_modules':modules,'preloaded_module_cache_pins':caches,'preloaded_class_aliases':aliases});self.assertEqual((len(modules),len(caches),len(aliases),len(built)),(63,59,2,51));self.assertNotIn('__main__',modules)
        for k in ('native_launch_authorized','native_runtime_qualified','native_authority_issued'):self.assertIs(p[k],False)
        self.assertTrue(all(p[k] is None for k in ('owner_approval','implementation_commit','dispatch_head')));self.assertTrue(all(x is None for x in CLOSED['pending'].values()));self.assertTrue(all(v['status']=='PENDING' and v['witness'] is None for v in p['current_witness_index'].values()));self.assertIsNone(p['separate_tensor_main']['actual_observation']);self.assertFalse(p['separate_tensor_main']['native_context_qualified']);self.assertEqual(V.validate_proposal_v2(p)['maps'],[63,59,2]);DETAILS['independent_row_projection']=[63,59,2,51]
    def test_independent_source_derived_scope_and_early_writer_fields(self):
        core=load(CORE,'independent_original_core_contract');self.assertEqual(hashlib.sha256(CORE.read_bytes()).hexdigest(),CLOSED['operation_registration']['accepted_core_source_pins']['contract.py']);dose=core.dose();pipeline=Path(CLOSED['source_candidates']['prefix_native_pipeline.py']['path']);pt=ast.parse(pipeline.read_bytes());phases=literal_assignment(pt,'STARTUP_PHASES');self.assertEqual(len(core.OPS),35);self.assertEqual(len(core.GUARDS),8);self.assertEqual(dose['maximum_phases'],58);self.assertEqual(core.LIMITS['maximum_markers'],128)
        registered=ORIGINAL['whole_program_registration'];self.assertEqual(registered['startup_order'],list(phases));self.assertEqual(registered['nonguard_success_operations'],len(core.OPS)+len(phases));self.assertEqual(registered['combined_maximum_phases'],dose['maximum_phases']+len(phases));self.assertEqual(registered['combined_maximum_markers'],2*(dose['maximum_phases']+len(phases)));self.assertEqual(registered['dose']['logical_tensor_bytes'],2*len(C.parse(INPUT.read_bytes())['prefix_ids'])*8);self.assertEqual(CLOSED['resources'],C.RESOURCES)
        worker=Path(CLOSED['source_candidates']['prefix_input_owned_worker.py']['path']);wt=ast.parse(worker.read_bytes());write=next(n for n in wt.body if isinstance(n,ast.FunctionDef) and n.name=='_write_precore_failure_return');self.assertEqual(ORIGINAL['early_failure_return']['report_fields'],literal_dict_fields(write,'provisional'));self.assertEqual(ORIGINAL['early_failure_return']['envelope_fields'],literal_dict_fields(write,'envelope'));self.assertEqual((len(literal_dict_fields(write,'provisional')),len(literal_dict_fields(write,'envelope'))),(11,7));DETAILS['source_derived_counts']={'core':len(core.OPS),'startup':len(phases),'phases':dose['maximum_phases']+len(phases),'markers':2*(dose['maximum_phases']+len(phases)),'bytes':registered['dose']['logical_tensor_bytes'],'early_fields':[7,11]}
    def test_resealed_source_helper_omission_and_map_main_forgery_refused(self):
        for kind in ('source','helper_omission','map_omission','future_main'):
            p=copy.deepcopy(ORIGINAL)
            if kind=='source':p['closed_source_packet']['source_candidates']['prefix_input_owned_worker.py']['sha256']='0'*64;V.seal(p['closed_source_packet'])
            elif kind=='helper_omission':p['selected_current_helpers'].pop(next(iter(p['selected_current_helpers'])))
            elif kind=='map_omission':p['observed009_selection']['maps']['preloaded_bootstrap_modules'].pop(next(iter(p['observed009_selection']['maps']['preloaded_bootstrap_modules'])))
            else:p['separate_tensor_main']['actual_observation']={'renamed_observer':True}
            with self.subTest(kind=kind),self.assertRaises(ValueError):V.validate_proposal_v2(reseal(p))
    def test_resealed_dose_authority_pending_resource_refused(self):
        for kind in ('dose','authority','witness','resource'):
            p=copy.deepcopy(ORIGINAL)
            if kind=='dose':p['whole_program_registration']['combined_maximum_markers']=130
            elif kind=='authority':p['native_launch_authorized']=True
            elif kind=='witness':next(iter(p['current_witness_index'].values()))['witness']={'qualified':True}
            else:p['closed_source_packet']['resources']['GPU_free_at_admission_GiB']=1;V.seal(p['closed_source_packet'])
            with self.subTest(kind=kind),self.assertRaises(ValueError):V.validate_proposal_v2(reseal(p))
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));sys.meta_path.remove(TRAP);after=pins();receipt={'kind':'INDEPENDENT_CLOSED_PROPOSAL_V2_SOURCE_MAP_SCOPE_REVIEW','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'source_pins_before':BEFORE,'source_pins_after':after,'selected_bytes_unchanged':BEFORE==after,'details':DETAILS,'actual_existing_closed_packet_and_V2_readers_exercised':True,'observer009_rows_independently_projected':True,'scope_counts_from_pinned_original_core_and_pipeline_source':True,'early_fields_from_actual_writer_AST':True,'pending_requirement_text_is_descriptive_not_exact_value_validated':True,'proposal_exact_file_hash_pinned':True,'native_import_trap_active':True,'native_API_or_framework_called':False,'native_runtime_qualified':False,'native_authority_issued':False,'training_transfer_quality_result':False}
    receipt['sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
    with (HERE/'INDEPENDENT_PROPOSAL_V2_001_REVIEW.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() and BEFORE==after else 1)
