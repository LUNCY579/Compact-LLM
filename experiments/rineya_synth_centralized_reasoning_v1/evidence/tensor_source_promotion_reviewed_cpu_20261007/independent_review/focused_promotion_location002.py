"""Frozen closed promotion graph: exact deltas, mirrors and inert interfaces."""
from pathlib import Path
from unittest.mock import patch
import ast, hashlib, json, shutil, sys, tempfile, types, unittest
HERE=Path(__file__).resolve().parent;FROZEN=HERE.parent/'frozen_location002';LOCATION=FROZEN/'location_candidate';PLAN=FROZEN/'PROMOTION_SOURCE_PLAN001.json';PACKAGE=next(p for p in FROZEN.parents if p.name=='rineya_synth_centralized_reasoning_v1')
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode('ascii')).hexdigest()
plan=json.loads(PLAN.read_text(encoding='utf-8'));assert plan['sha256']==digest({k:v for k,v in plan.items() if k!='sha256'});assert hashlib.sha256(PLAN.read_bytes()).hexdigest()=='8ad0242bd26d17dd9f68a9360ae645774b58aafaeba0959866432d42086468bd'
FILES=[p for p in FROZEN.rglob('*') if p.is_file()]
for row in plan['sources'].values():
    target=Path(row['promotion_target'])
    if target.exists():FILES.append(target)
def pins():return {str(p):{'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in FILES}
BEFORE=pins();EXPECTED={'prefix_tensor_contract.py':'de99dbcf40f3ba094c3932ace795523a7995f9708973aa09b5810154cb93ecca','prefix_native_pipeline.py':'1fc35de5925a7b4b2d7b91ab2a7a80f7eab0ace58ffec9648b75a41d2eb29420','prefix_input_owned_worker.py':'3da27e8cf117ca078583739a2bdc8501188c4a37e1cd6999a16f2dc0cde9c387'}
for n,h in EXPECTED.items():assert BEFORE[str(LOCATION/n)]['sha256']==h
def load(path,name):
    raw=path.read_bytes();m=types.ModuleType(name);m.__file__=str(path);m.__compiled_source_descriptor__={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'};exec(compile(raw,str(path),'exec'),m.__dict__);return m
class Trap:
    def find_spec(self,fullname,path=None,target=None):
        if fullname.split('.')[0] in ('ctypes','torch','psutil','native_worker','native_process','pynvml'):raise AssertionError('native import forbidden '+fullname)
TRAP=Trap();sys.meta_path.insert(0,TRAP);DETAILS={}
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='independent_closed_promotion_');self.pkg=Path(self.tmp.name)/'experiments/rineya_synth_centralized_reasoning_v1';self.pkg.mkdir(parents=True)
        for p in LOCATION.glob('*.py'):shutil.copy2(p,self.pkg/p.name)
    def tearDown(self):self.tmp.cleanup()
    def test_exact_three_file_delta_and_eight_unchanged_sources(self):
        old=(FROZEN/'predecessors/prefix_tensor_contract.py').read_text(encoding='utf-8');new=(LOCATION/'prefix_tensor_contract.py').read_text(encoding='utf-8');self.assertEqual(new,old.replace("predecessor=read_bytes(HERE/'predecessor_sources'/name)","predecessor_root=HERE/'predecessor_sources' if HERE.parent.name=='artifacts' else PACKAGE/'evidence/site_disabled_tensor_packet_reviewed_cpu_20261007/predecessor_sources';predecessor=read_bytes(predecessor_root/name)"))
        for name in ('prefix_native_pipeline.py','prefix_input_owned_worker.py'):
            old=(FROZEN/'predecessors'/name).read_text(encoding='utf-8');new=(LOCATION/name).read_text(encoding='utf-8');beforeC=hashlib.sha256((FROZEN/'predecessors/prefix_tensor_contract.py').read_bytes()).hexdigest();old=old.replace(beforeC,EXPECTED['prefix_tensor_contract.py'])
            if name=='prefix_native_pipeline.py':old=old.replace("'bytes': 74855","'bytes': 75016")
            if name=='prefix_input_owned_worker.py':old=old.replace(hashlib.sha256((FROZEN/'predecessors/prefix_native_pipeline.py').read_bytes()).hexdigest(),EXPECTED['prefix_native_pipeline.py'])
            self.assertEqual(new,old)
        unchanged=set(plan['sources'])-set(EXPECTED);self.assertEqual(len(unchanged),8)
        for n in unchanged:self.assertEqual((LOCATION/n).read_bytes(),(FROZEN/'predecessors'/n).read_bytes())
        DETAILS['exact_changed_sources']=list(EXPECTED);DETAILS['unchanged_sources']=8
    def test_actual_target_predecessors_durable_six_and_compile_all(self):
        self.assertEqual(len(plan['sources']),11)
        for n,row in plan['sources'].items():
            target=Path(row['promotion_target']);self.assertEqual(target,PACKAGE/n);expected=plan['current_public_predecessors'][n];self.assertEqual(target.exists(),expected['present'])
            if target.exists():self.assertEqual(BEFORE[str(target)],{k:expected['descriptor'][k] for k in ('bytes','sha256')})
            raw=(LOCATION/n).read_bytes();self.assertEqual({'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'},row['descriptor']);compile(raw,str(LOCATION/n),'exec')
        actual=PACKAGE/'evidence/site_disabled_tensor_packet_reviewed_cpu_20261007/predecessor_sources';self.assertEqual(len(list(actual.glob('*.py'))),6)
        for p in (FROZEN/'durable_prospective_predecessor_sources').glob('*.py'):self.assertEqual(p.read_bytes(),(actual/p.name).read_bytes())
    def test_actual_public_mirror_links_closed_gates_and_bad_pin(self):
        C=load(self.pkg/'prefix_tensor_contract.py','independent_public_C');W=load(self.pkg/'prefix_input_owned_worker.py','independent_public_W');P=load(self.pkg/'prefix_native_pipeline.py','independent_public_P');S=load(self.pkg/'prefix_input_owned_supervisor.py','independent_public_S');A=load(self.pkg/'prefix_input_owned_auditor.py','independent_public_A')
        for c in (C,W.C,P.C,S.C):
            self.assertEqual(c.PACKAGE,self.pkg)
            with self.assertRaises(ValueError):c.stage_native_gate()
        self.assertEqual(A.PACKAGE,self.pkg);self.assertEqual(W.CONTRACT_PIN,EXPECTED['prefix_tensor_contract.py']);self.assertEqual(W.ENTRY_PIPELINE_PIN,EXPECTED['prefix_native_pipeline.py']);self.assertEqual(P.CONTRACT_BOOTSTRAP['sha256'],W.CONTRACT_PIN)
        f=self.pkg/'prefix_tensor_contract.py';f.write_bytes(f.read_bytes()+b'\n# drift\n')
        with self.assertRaisesRegex(ValueError,'exact stable source'):load(self.pkg/'prefix_input_owned_worker.py','independent_bad_public_W')
    def test_actual_source_validator_helper_omission_and_closed_plan(self):
        # Read author plan checker definitions against the original frozen plan;
        # staged paths retain original provenance and candidate bytes match.
        T=load(FROZEN/'test_source_promotion.py','independent_plan_checker');original_LOCATION=T.LOCATION;T.LOCATION=HERE.parent/'location_candidate'
        try:
            self.assertEqual(T.validate_plan(plan)['source_count'],11);bad=json.loads(json.dumps(plan));bad['sources'].pop('typing_alias_checks.py');bad['sha256']=digest({k:v for k,v in bad.items() if k!='sha256'})
            with self.assertRaisesRegex(ValueError,'eleven-source'):T.validate_plan(bad)
        finally:T.LOCATION=original_LOCATION
        for k in ('promotion_performed','native_authority_issued','native_launch_authorized','future_gate_successor_included'):self.assertIs(plan[k],False)
    def test_actual_direct_main_metadata_helper_on_public_shape(self):
        selected=self.pkg/'prefix_input_owned_worker.py';raw=selected.read_bytes();tree=ast.parse(raw,str(selected));self.assertEqual(ast.unparse(tree.body[-1].test),"__name__ == '__main__'");tree.body.pop();m=types.ModuleType('__main__');m.__file__=str(selected);m.__spec__=None;m.__cached__=None;m.__loader__=__import__('importlib').machinery.SourceFileLoader('__main__',str(selected));exec(compile(tree,str(selected),'exec'),m.__dict__);C=load(self.pkg/'prefix_tensor_contract.py','independent_public_direct_C');m.C=C;cap=C._mint_CPU_native_context_fixture();value={'fixture_only':True,'config':{'source_files':{str(selected):{'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}}}}
        with patch.dict(sys.modules,{'__main__':m}):observed=m._CPU_main_provenance_probe(value,sys,_capability=cap);prepared=m._binding_compiled_source(value)
        self.assertEqual(prepared['compiled_source'],observed);self.assertFalse(observed['source_execution_history_proven']);self.assertFalse(observed['current_module_execution_observed']);DETAILS['public_shape_main_metadata_actual']=True
    def test_actual_private_writer_and_consumer_replay_public_shape(self):
        for relative in ('evidence/prefix_input_core_reviewed_cpu_20261005','evidence/exception_frame_helper_reviewed_cpu_20261005'):
            shutil.copytree(PACKAGE/relative,self.pkg/relative)
        W=load(self.pkg/'prefix_input_owned_worker.py','independent_public_core_W');P=load(self.pkg/'prefix_native_pipeline.py','independent_public_core_P');A=load(self.pkg/'prefix_input_owned_auditor.py','independent_public_core_A');F=load(W.C.CORE/'fixture_support.py','independent_public_core_F');output=self.pkg/'artifacts/inert';output.mkdir(parents=True);ledger,phases=W._entry_journal_bootstrap(output)
        actions={n:lambda *a:None for n in phases}
        def execute(state,owned):run,core,proof=P._bind_shared_core_executor(W,owned);api=F.InjectedAPI();return run(api,api.guard,output)
        report=W._entry_phase_body(ledger,phases,actions,execute);self.assertTrue(report['whole_scope_success_validated']);(output/'CPU_ENTRY_RETURN.json').write_text(A.canonical(report),encoding='utf-8');preserved=output.parent/'preserved';shutil.copytree(output,preserved)
        def desc(p):b=p.read_bytes();return {'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest()}
        selected={r['path']:{'bytes':r['bytes'],'sha256':r['sha256']} for r in ledger.entry_bootstrap_sources};raw={n:desc(output/n) for n in A.RAW_CORE};prefix=json.loads(W.source_core()['C'].INPUT.read_bytes())['prefix_ids'];value=A._CPU_shared_entry_bytes(preserved,original_directory=output,raw_files=raw,report_file='CPU_ENTRY_RETURN.json',report_descriptor=desc(output/'CPU_ENTRY_RETURN.json'),prefix=prefix,selected_source_pins=selected);self.assertTrue(value['source_outcome_completed']);self.assertFalse(value['native_authority_issued']);DETAILS['public_shape_private_writer_consumer_actual']=True
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));sys.meta_path.remove(TRAP);after=pins();receipt={'kind':'INDEPENDENT_CLOSED_PUBLIC_LOCATION_PROMOTION_REVIEW002','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'all_controls_completed':result.wasSuccessful(),'source_public_and_frozen_pins_before':BEFORE,'source_public_and_frozen_pins_after':after,'pins_unchanged':BEFORE==after,'details':DETAILS,'public_shaped_temporary_mirrors_only':True,'main_fixture_one_EOF_dispatch_removed_functions_frames_unchanged':True,'actual_direct_script_or_native_process_executed':False,'public_source_promotion_performed':False,'gate_successor_included':False,'native_API_or_framework_called':False,'native_authority_issued':False,'handoff_unchanged_nine_wording_count_is_eight':True,'first_review_fixture_expectation_failures_preserved':'first_location002_fixture_expectation_failure'}
    receipt['sha256']=digest(receipt)
    with (HERE/'INDEPENDENT_PROMOTION_LOCATION002_SUCCESSOR.json').open('x',encoding='utf-8') as stream:json.dump(receipt,stream,sort_keys=True,indent=2);stream.write('\n')
    raise SystemExit(0 if result.wasSuccessful() and BEFORE==after else 1)
