"""New paired wire controls using authored metadata only; no genuine reads."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
from unittest.mock import patch
import hashlib,json,sys,tempfile,time,unittest
HERE=Path(__file__).resolve().parent
def module(name):
    value=ModuleType(name);value.__file__=str(HERE/(name+'.py'))
    exec(compile(Path(value.__file__).read_bytes(),value.__file__,'exec'),value.__dict__);return value
E=module('arithmetic_evidence');M=module('arithmetic_measurement')
def seal(value):value.pop('sha256',None);value['sha256']=E.digest(value);return value
def authored_audit():
    return seal({'schema_version':1,'status':'PASS_SHORT_SCOPED_NATIVE_INFERENCE_MECHANICS',
        'hash_closure':{'config_file_sha256':E.CONFIG_FILE,'config_body_sha256':E.CONFIG_BODY,
            'preservation_receipt_sha256':'fe49928806b6edd62b5c004afa5dde94e9d9d217f1026d6ecbee739db7644f67',
            'raw_file_count':95,'source_and_artifact_bytes_unchanged_during_audit':True},
        'accounting':{'actual_forward_calls':84,'actual_input_positions':28627,'returned_API_calls':403,
            'failed_API_calls':0,'pending_API_calls':0,'backwards':0,'optimizer_steps':0,'parity_forward_calls':2,
            'parity_input_positions':838,'qualification_forward_calls':82,'qualification_input_positions':27789,
            'individual_durable_token_stages':78,'generated_only_bulk_forwards':4,'registered_operation_counts':dict(E.OPERATIONS)}})
def audit_ref():return {'path':str(E.LEGACY_INFQ_AUDIT_PATH),'bytes':26435,'sha256':E.LEGACY_INFQ_AUDIT_SHA,'value_sha256':'AUTHORED_TEST_ONLY'}
def config_ref():return {'path':str(E.LEGACY_INFQ_CONFIG_PATH),'bytes':45120,'sha256':E.CONFIG_FILE,'value_sha256':'AUTHORED_TEST_ONLY'}
def metadata_request():return {'baseline_id':'AUTHORED_BASELINE','audit_reference':audit_ref()}

class WireControls(unittest.TestCase):
    def test_fixed_validate_normalizes_then_genuine_manifest_remains_unknown(self):
        read=[]
        def authored_read(ref):read.append(ref);return authored_audit()
        with patch.object(E,'read',authored_read):
            value=E.validate('INFQ_HISTORICAL_SOURCE_RECIPE_METADATA',metadata_request())
        self.assertEqual([audit_ref()],read);self.assertEqual(84,value['original_schema_accounting']['observed_forward_calls'])
        self.assertFalse(value['qualified']);self.assertFalse(value['native_admission_authorized'])
        self.assertNotIn('observed_model_input_calls',value);self.assertFalse(value['modern_all_equal_synthesized'])
        with self.assertRaisesRegex(ValueError,'UNKNOWN observed causal prefix'):
            E._original_call_manifest({'observed_forward_calls':84},authored_audit())
        request={'baseline_id':'AUTHORED_BASELINE','evidence':'AUTHORED_REFERENCE'}
        attempts=[{'attempt_id':name,'original_audit_reference':audit_ref(),'original_config_reference':config_ref(),
            'includes_failed_and_partial_prefixes':True,'observed_complete':True,'original_journal_references':['AUTHORED_ONLY'],
            'observed_forward_calls':84} for name in ['INFQ001_attempt1',*sorted(E.KNOWN_PARTIAL_ATTEMPTS-{'INFQ001_attempt1'})]]
        evidence={'scope':'AVAILABLE_SOURCES_AND_OBSERVED_CONSUMED_OCCURRENCES','namespace_only_proof':False,
            'categories':{k:{} for k in E.CATEGORIES},'attempt_commitments':attempts,
            'shared_commitment_profile':{'family':'arithmetic_full_canonical_ast_json_v1','input':'ordered_model_input_token_ids_json_v1',
                'tokenizer_sha256':'aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4',
                'chat_template_sha256':'a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8'}}
        with tempfile.TemporaryDirectory(dir=HERE,prefix='authored_parser_') as directory:
            package=Path(directory);parser=b'def normalize(value):return value\n';(package/'arithmetic_guard.py').write_bytes(parser)
            def authored_inputs(ref):
                if ref==audit_ref():return authored_audit()
                if ref==config_ref():return {'sha256':E.CONFIG_BODY}
                raise AssertionError('unexpected genuine/raw source read')
            with patch.object(E,'PACKAGE',package),patch.object(E,'SCORER',hashlib.sha256(parser).hexdigest()),patch.object(E,'evidence',return_value=evidence),patch.object(E,'read',authored_inputs):
                with self.assertRaisesRegex(ValueError,'UNKNOWN observed causal prefix'):
                    E.validate('ARITHMETIC_AVAILABLE_TRAIN_DEV_LINEAGE_DISJOINT',request)
                evidence['attempt_commitments'][0]['original_config_reference']['sha256']='0'*64
                with self.assertRaisesRegex(ValueError,'auditA/configB forbidden'):
                    E.validate('ARITHMETIC_AVAILABLE_TRAIN_DEV_LINEAGE_DISJOINT',request)
    def test_fixed_original_reference_schema_and_family_empty_mutations_refuse(self):
        for field,value in (('path',str(HERE/'raw_provider.json')),('sha256','0'*64),('bytes',True)):
            request=metadata_request();request['audit_reference'][field]=value
            with patch.object(E,'read',side_effect=AssertionError('read before exact role gate')):
                with self.subTest(field=field),self.assertRaises(ValueError):E.validate('INFQ_HISTORICAL_SOURCE_RECIPE_METADATA',request)
        for mutate in (lambda v:v['hash_closure'].update(config_body_sha256='0'*64),lambda v:v['accounting'].update(actual_input_positions=28626)):
            value=authored_audit();mutate(value);seal(value)
            with patch.object(E,'read',return_value=value),self.assertRaises(ValueError):E.validate('INFQ_HISTORICAL_SOURCE_RECIPE_METADATA',metadata_request())
        with self.assertRaisesRegex(ValueError,'nonempty original'):E.original_membership([])
    def test_paired_proof_rejects_qualified_escalation_and_foreign_callback_before_read(self):
        spec={'source':{'path':str(HERE/'foreign_callback.py')},'function':'validate','request':'REQUEST','result':'RESULT',
            'purpose':'INFQ_HISTORICAL_SOURCE_RECIPE_METADATA'}
        with patch.object(M,'read_ref',side_effect=AssertionError('foreign callback caused read')):
            with self.assertRaisesRegex(ValueError,'fixed own evidence'):M.proof(spec,{},purpose=spec['purpose'],fixture_only=False,baseline_id='AUTHORED_BASELINE')
        spec['source']['path']=str(HERE/'arithmetic_evidence.py')
        with patch.object(E,'read',return_value=authored_audit()):normalized=E.validate(spec['purpose'],metadata_request())
        def read_ref(ref):return metadata_request() if ref=='REQUEST' else normalized
        with patch.object(M,'read_ref',read_ref),patch.object(M,'source',return_value=E),patch.object(E,'read',return_value=authored_audit()):
            with self.assertRaisesRegex(ValueError,'replay/source/baseline differs'):M.proof(spec,{},purpose=spec['purpose'],fixture_only=False,baseline_id='AUTHORED_BASELINE')
        escalated={**normalized,'qualified':True}
        with patch.object(M,'read_ref',lambda ref:metadata_request() if ref=='REQUEST' else escalated),patch.object(M,'source',return_value=E),patch.object(E,'read',return_value=authored_audit()):
            with self.assertRaisesRegex(ValueError,'replay/source/baseline differs'):M.proof(spec,{},purpose=spec['purpose'],fixture_only=False,baseline_id='AUTHORED_BASELINE')
    def test_staged_dispatch_native_context_and_raw_audit_witness_stay_closed(self):
        self.assertEqual('rineya_synth_centralized_reasoning_v1',M.PACKAGE.name);self.assertNotEqual(M.HERE,M.EXPECTED)
        with patch.object(M,'read_ref',side_effect=AssertionError('staged dispatch read input')):
            with self.assertRaisesRegex(ValueError,'staged arithmetic'):M.prepare_registered({'path':'AUTHORED_UNUSED'})
        fixture=M.Prepared({'config':{'fixture_only':True}},capability=M._CPU_capability())
        with self.assertRaisesRegex(ValueError,'native-qualified context'):M._native_value(fixture)
        with self.assertRaisesRegex(ValueError,'CPU token cannot mint'):M.Prepared({'config':{'fixture_only':False}},capability=M._CPU_capability())
        with patch.object(E,'evidence',return_value={'auditor_source_sha256':'WRONG'}),patch.object(E.Path,'is_file',return_value=False):
            with self.assertRaisesRegex(ValueError,'reviewed integrated'):E.raw_capture({'baseline_id':'AUTHORED','evidence':'UNUSED'})

if __name__=='__main__':
    def no_genuine_inputs(event,args):
        if event=='open' and type(args[0]) in (str,bytes):
            name=str(args[0]).replace('\\','/').lower()
            if '/chatbubble-rineya-synth-centralized-reasoning-v1/' in name or '/rineya-infq-' in name:
                raise RuntimeError('paired wire fixtures cannot read genuine metadata/raw/provider inputs')
    sys.addaudithook(no_genuine_inputs)
    started=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(WireControls))
    receipt={'kind':'AUTHORED_PAIRED_INFQ_SOURCE_CONSUMER_WIRE_CONTROLS003','tests':result.testsRun,'errors':len(result.errors),
        'failures':len(result.failures),'success':result.wasSuccessful(),'seconds':time.monotonic()-started,'fixture_only':True,
        'genuine_metadata_raw_solver_model_tokenizer_GPU_scoring_projection_enrollment_reads_or_actions':False,
        'source_pins':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('arithmetic_evidence.py','arithmetic_measurement.py','test_paired_infq_interface.py')}}
    receipt['sha256']=E.digest(receipt)
    with (HERE/'CPU_WIRE_CONTROLS003.json').open('xb') as f:f.write((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)
