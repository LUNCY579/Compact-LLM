"""Focused source-qualified wire fixtures; no real captures, populations or models."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType,SimpleNamespace
from unittest.mock import patch
import hashlib,json,os,sys,tempfile,time,unittest

HERE=Path(__file__).resolve().parent
PACKAGE=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1')
READS=[];IMPORTS=[]
def audit(event,args):
    if event=='import' and args and args[0].split('.',1)[0] in ('torch','transformers','ctypes','psutil','tokenizers','safetensors'):
        IMPORTS.append(args[0]);raise AssertionError('framework/API import forbidden')
    if event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike)):
        value=str(args[0]).replace('\\','/').lower()
        if any(s in value for s in ('/data/','/weights','/model','/check/','/final/','arithmetic_guard_population','token_pool')):raise AssertionError('real data/model/captures forbidden')
        READS.append(str(args[0]))
sys.addaudithook(audit);sys.path.insert(0,str(HERE))
import arithmetic_measurement as M
import arithmetic_evidence as E

class WireControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(dir=HERE);self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
        f=self.root/'authored_proof.py';f.write_bytes(b'def observe(**request):\n    return request["expected"]\n')
        self.proof_source={'path':str(f),'bytes':f.stat().st_size,'sha256':hashlib.sha256(f.read_bytes()).hexdigest()}
        self.pins={str(PACKAGE/'arithmetic_guard.py'):M.SCORER_SOURCE_SHA256,f.as_posix():self.proof_source['sha256']}
        self.pins[self.proof_source['path']]=self.proof_source['sha256']
    def ref(self,name,value):
        path=self.root/name;raw=(M.canonical(value)+'\n').encode();path.write_bytes(raw)
        return {'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'value_sha256':M.digest(value)}
    def spec(self,name,purpose,extra=None):
        value={'purpose':purpose,'baseline_id':'AUTHORED_BASELINE','fixture_only':True,'qualified':True,**(extra or {})}
        return {'source':self.proof_source,'function':'observe','request':self.ref(name+'_request.json',{'expected':value}),
            'result':self.ref(name+'_result.json',value),'purpose':purpose}
    def fixture(self):
        rows=[];labels=[];batches=[]
        for i in range(512):
            rid=M.digest(['ROW',i]);parent=M.digest(['FAMILY',i//4]);labels.append({'row_id':rid,'parent_id':parent,
                'parent':{'expression_sha256':parent},'answer':0})
            rows.append({'row_id':rid,'raw':{'finish_reason':'generation_cap' if i%2 else 'eos','decoded_response':'{"answer":0}'}})
        for i in range(8):batches.append({'rows':[{'row_id':r['row_id'],'prompt_ids':[1]} for r in rows[i*64:(i+1)*64]]})
        domain={'domain_id':'arithmetic','class':'task','row_ids':[r['row_id'] for r in rows],'component_ids':[r['parent_id'] for r in labels]}
        imported={'fixture_only':True,'rows':rows,'domain':domain};plan={'rows':512,'parent_components':128,'batches':{'arithmetic_guard_DEV':batches},'row_order_sha256':M.digest(domain['row_ids'])}
        policy={'sha256':'a'*64};refs={'plan':self.ref('plan.json',plan),'labels':self.ref('labels.json',{'rows':labels}),
            'policy':self.ref('policy.json',policy),'roster':self.ref('roster.json',{'domains':[domain],'candidate_outputs_used':False})}
        line={'scope':'AVAILABLE_SOURCES_AND_OBSERVED_CONSUMED_OCCURRENCES','namespace_only_proof':False,
            'plan_file_sha256':refs['plan']['sha256'],'guard_family_hashes':list(set(domain['component_ids'])),
            'guard_input_hashes':[M.digest([1])]}
        checks={'disjointness_check':self.spec('lineage','ARITHMETIC_AVAILABLE_TRAIN_DEV_LINEAGE_DISJOINT',line),
            'chronology_check':self.spec('chronology','ROSTER_FROZEN_BEFORE_CAPTURE',{'roster_file_sha256':refs['roster']['sha256']}),
            'scorer_check':self.spec('scorer','ARITHMETIC_SCORER_SOURCE_LABEL_POLICY_QUALIFIED',{
                'labels_file_sha256':refs['labels']['sha256'],'scorer_source_sha256':M.SCORER_SOURCE_SHA256,'policy_value_sha256':refs['policy']['value_sha256']}),
            'capture_checks':[self.spec('capture%d'%i,'ARITHMETIC_FIXED64_RAW_CAPTURE_QUALIFIED',{'ordinal':i,'rows':64,'source_kind':'arithmetic_guard_DEV'}) for i in range(1,9)]}
        rawref=self.ref('capture_registration.json',{'AUTHORED_ONLY':True})
        c={'kind':'REGISTERED_ARITHMETIC_M0_MEASUREMENT001','fixture_only':True,'baseline_id':'AUTHORED_BASELINE',
            'source_pins':self.pins,**refs,'raw_capture_registration':rawref,**checks}
        binding={'baseline_id':c['baseline_id'],'source_pins':c['source_pins'],**refs,'raw_capture_registration':rawref,
            'proofs_sha256':M.digest(checks)}
        approval={'kind':'OWNER_ARITHMETIC_M0_SCORING_AUTHORIZATION001','fixture_only':True,'approved':True,
            'binding':binding,'permissions':{'arithmetic_M0_scoring':True,'arithmetic_enrollment_contribution':True,'training':False,
                'protected_bank_access':False,'language_ratings':False},'human_evidence_reference':'AUTHORED_NO_NATIVE_AUTHORITY'}
        c['approval']=self.ref('approval.json',approval);c['review']=self.ref('review.json',{
            'kind':'INDEPENDENT_ARITHMETIC_M0_MEASUREMENT_REVIEW001','fixture_only':True,'accepted':True,'binding':binding,
            'approval_file_sha256':c['approval']['sha256']});c=M.seal(c)
        return self.ref('registration.json',c),c,imported,labels
    def scorer(self):
        def score(row,response,policy):
            try:obj=M.parse(response.encode());ok=type(obj) is dict and set(obj)=={'answer'} and type(obj['answer']) is int
            except ValueError:return {'correct':False,'malformed':True}
            return {'correct':ok and obj['answer']==row['answer'],'malformed':not ok}
        return SimpleNamespace(score=score)
    def test_actual_conditional_CPU_wire_full512_whole_response_and_cap(self):
        ref,c,imported,labels=self.fixture();calls=[]
        prepared=M._CPU_prepare(ref,lambda r:(calls.append(r) or imported),capability=M._CPU_capability())
        self.assertEqual([c['raw_capture_registration']],calls)
        value=M._CPU_score(prepared,self.scorer(),capability=M._CPU_capability())
        self.assertEqual(512,sum(value['scores'].values()));self.assertEqual(256,sum(x['at_generation_cap'] for x in value['details']))
        context=M._value(prepared);context['imported']['rows'][0]['raw']['decoded_response']=' {"answer": null} '
        changed=M.Prepared(context,capability=M._CPU_capability());score=M._CPU_score(changed,self.scorer(),capability=M._CPU_capability())
        self.assertTrue(score['details'][0]['abstained']);self.assertFalse(score['details'][0]['correct'])
        context['imported']['rows'][0]['raw']['decoded_response']='text {"answer":0}'
        score=M._CPU_score(M.Prepared(context,capability=M._CPU_capability()),self.scorer(),capability=M._CPU_capability())
        self.assertTrue(score['details'][0]['malformed'])
    def test_approval_proof_and_eight_capture_order_fail_before_loader(self):
        ref,c,imported,_=self.fixture();altered=deepcopy(c);altered['capture_checks'].reverse();altered=M.seal({k:v for k,v in altered.items() if k!='sha256'})
        bad=self.ref('wrong_registration.json',altered)
        with self.assertRaises(ValueError):M._CPU_prepare(bad,lambda _:(_ for _ in ()).throw(AssertionError('loader before gate')),capability=M._CPU_capability())
        with self.assertRaisesRegex(ValueError,'fixed own evidence'):M.proof(c['scorer_check'],c['source_pins'],purpose=c['scorer_check']['purpose'],fixture_only=False,baseline_id=c['baseline_id'])
    def test_CPU_cannot_mint_native_and_native_gates_deny_before_reads(self):
        ref,c,imported,_=self.fixture();prepared=M._CPU_prepare(ref,lambda _:imported,capability=M._CPU_capability())
        value=M._value(prepared);value['config']['fixture_only']=False
        with self.assertRaises(ValueError):M.Prepared(value,capability=M._CPU_capability())
        with self.assertRaises(ValueError):M._native_value(prepared)
        for function,args in ((M.prepare_registered,({},)),(M.pre_capture_admission,({},)),(M.score_prepared,(prepared,)),
            (M.enroll_registered,(prepared,{},None,{}))):
            with self.subTest(function=function.__name__),self.assertRaisesRegex(ValueError,'staged arithmetic'):function(*args)
    def test_lineage_missing_comparability_partial_exposure_and_dialogue_authority_refused(self):
        base={'purpose':'ARITHMETIC_AVAILABLE_TRAIN_DEV_LINEAGE_DISJOINT','baseline_id':'AUTHORED','fixture_only':False,
            'scope':'AVAILABLE_SOURCES_AND_OBSERVED_CONSUMED_OCCURRENCES','namespace_only_proof':False,
            'categories':{name:{} for name in E.CATEGORIES},'shared_commitment_profile':{'family':'different_hash_namespace'}}
        base= M.seal(base);request={'baseline_id':'AUTHORED','evidence':self.ref('incompatible_lineage.json',base)}
        with self.assertRaisesRegex(ValueError,'shared typed'):E.lineage(request)
        with self.assertRaises(ValueError):E.validate('arbitrary_callback',{})
        v=M.seal({'purpose':'ALL_DOMAIN_M0_SCORING_QUALIFIED','baseline_id':'AUTHORED','fixture_only':False,'domain_id':'dialogue'})
        with self.assertRaisesRegex(ValueError,'actual domain source'):E.domain_scores({'baseline_id':'AUTHORED','evidence':self.ref('missing_dialogue.json',v)})
    def test_scored_context_mutation_cannot_enroll_and_missing_domains_remain_inconclusive(self):
        ref,c,imported,_=self.fixture();prepared=M._CPU_prepare(ref,lambda _:imported,capability=M._CPU_capability())
        scored=M._CPU_score(prepared,self.scorer(),capability=M._CPU_capability());score_ref=self.ref('scores.json',scored['scores'])
        evidence=self.spec('domain','ALL_DOMAIN_M0_SCORING_QUALIFIED',{'domain_id':'arithmetic','roster_file_sha256':c['roster']['sha256'],
            'score_file_sha256':score_ref['sha256'],'scorer_sha256':'1'*64,'labels_sha256':'2'*64,'qualification_receipt_sha256':'3'*64})
        path=PACKAGE/'dev_statistics.py';stats=ModuleType('authored_original_statistics');stats.__file__=str(path)
        exec(compile(path.read_bytes(),str(path),'exec'),stats.__dict__);stats.policy=lambda:{'retention':{'required_domain_classes':['task','language','long_context']}}
        value=M._CPU_enroll(prepared,scored,score_ref,{'arithmetic':evidence},stats,capability=M._CPU_capability())
        self.assertEqual('INCONCLUSIVE',value['enrollment']['status'])
        forged=deepcopy(scored);forged['raw_import_sha256']='0'*64;forged=M.seal({k:v for k,v in forged.items() if k!='sha256'})
        with self.assertRaises(ValueError):M._CPU_enroll(prepared,forged,score_ref,{'arithmetic':evidence},stats,capability=M._CPU_capability())
    def test_full_original_membership_rejects_proper_subset_duplicate_order_and_changed_refs(self):
        input_rows=[{'row_id':'AUTHORED_%d'%i,'prompt_ids':[i+1]} for i in range(3)]
        semantics=[{'row_id':row['row_id'],'parent':{'canonical_expression':{'op':'add','args':[i,1]}}} for i,row in enumerate(input_rows)]
        a=self.ref('original_input.json',{'rows':input_rows});b=self.ref('original_semantic.json',{'rows':semantics})
        records=[{'original_input_reference':a,'original_semantic_reference':b,'original_row_index':i,'row_id':row['row_id']} for i,row in enumerate(input_rows)]
        self.assertTrue(E.original_membership(records))
        for changed in (records[:2],[records[0],records[0],records[2]],list(reversed(records))):
            with self.subTest(changed=changed),self.assertRaises(ValueError):E.original_membership(changed)
        changed=deepcopy(records);changed[0]['original_semantic_reference']['sha256']='0'*64
        with self.assertRaises(ValueError):E.original_membership(changed)
    def test_raw_proof_A_cannot_return_declared_B_capture_anchors_or_model_source(self):
        config={'sha256':'1'*64,'source_packet_sha256':'2'*64,'model_pins':{'base':'3'*64},'source_pins':{'source':'4'*64}}
        request={'expected_selector':['arithmetic_guard_DEV',1],'expected_config_file_sha256':'5'*64,
            'config_path':str(self.root/'config_A.json'),'preserved_directory':str(self.root/'preserved_A'),
            'expected_source_commit':'a'*40,'expected_plan_file_sha256':'6'*64}
        audit={'batch_selector':{'source_kind':'arithmetic_guard_DEV','ordinal':1},'config_file_sha256':'5'*64,
            'config_path':request['config_path'],'preserved_path':request['preserved_directory'],'source_commit':'a'*40,
            'config_body_sha256':'1'*64,'source_packet_sha256':'2'*64,'rows':64,'hash_closure':{'all_equal':True}}
        claim={'actual_audit_receipt':{'sha256':'7'*64},'audit_file_sha256':'7'*64,'config_file_sha256':'5'*64,
            'preserved_directory':request['preserved_directory'],'ordinal':1,'source_kind':'arithmetic_guard_DEV','rows':64,
            'plan_file_sha256':'6'*64,'source_commit':'a'*40,'source_packet_sha256':'2'*64,
            'model_pins':config['model_pins'],'source_pins':config['source_pins']}
        self.assertEqual(claim,E.bind_raw_capture(claim,audit,request,config))
        for field,other in (('audit_file_sha256','8'*64),('config_file_sha256','9'*64),('ordinal',2),
            ('preserved_directory',str(self.root/'preserved_B')),('model_pins',{'base':'0'*64}),('source_pins',{'other':'0'*64})):
            changed=deepcopy(claim);changed[field]=other
            with self.subTest(field=field),self.assertRaisesRegex(ValueError,'another audit/config'):E.bind_raw_capture(changed,audit,request,config)

if __name__=='__main__':
    files=('arithmetic_measurement.py','arithmetic_evidence.py','test_arithmetic_measurement.py')
    before={n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files};start=time.monotonic()
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WireControls))
    value={'kind':'ARITHMETIC_MEASUREMENT_NEW_WIRE_CPU_CONTROLS001','tests':result.testsRun,'success':result.wasSuccessful(),
        'failures':len(result.failures),'errors':len(result.errors),'seconds':time.monotonic()-start,'source_before':before,
        'source_after':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in files},'read_paths':sorted(set(READS)),
        'forbidden_import_attempts':IMPORTS,'real_population_capture_model_framework_or_ratings_performed':False,
        'native_scoring_admitted':False,'actual_available_lineage_disjointness_established':False}
    value['sha256']=M.digest(value);(HERE/'CPU_CONTROLS001.json').write_bytes((M.canonical(value)+'\n').encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)
