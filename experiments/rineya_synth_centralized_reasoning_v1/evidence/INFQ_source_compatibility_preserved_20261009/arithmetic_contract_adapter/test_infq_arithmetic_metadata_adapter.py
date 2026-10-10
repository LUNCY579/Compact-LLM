"""Authored metadata-shape controls only; no genuine source/raw fixture reads."""
from pathlib import Path
import hashlib,json,sys,time,unittest
from copy import deepcopy
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import infq_arithmetic_metadata_adapter as A

def seal(value):
    value.pop('sha256',None);value['sha256']=A.digest(value);return value
def fixture():
    ops={**A.ROLE_CALLS,'tensor_copy':310,'checkpoint_load':1,'cuda_context_init':1,'model_construct':1,
        'install_activation_hooks':1,'install_fp32_compute':1,'install_qv_lora':1,'verify_frozen_base':1,
        'zero_lora_query_after':1,'zero_lora_query_before':1}
    audit=seal({'schema_version':1,'status':'PASS_SHORT_SCOPED_NATIVE_INFERENCE_MECHANICS',
        'hash_closure':{'config_file_sha256':A.CONFIG_FILE,'config_body_sha256':A.CONFIG_BODY,
            'preservation_receipt_sha256':'fe49928806b6edd62b5c004afa5dde94e9d9d217f1026d6ecbee739db7644f67',
            'raw_file_count':95,'source_and_artifact_bytes_unchanged_during_audit':True},
        'accounting':{'actual_forward_calls':84,'actual_input_positions':28627,'returned_API_calls':403,
            'failed_API_calls':0,'pending_API_calls':0,'backwards':0,'optimizer_steps':0,'parity_forward_calls':2,
            'parity_input_positions':838,'qualification_forward_calls':82,'qualification_input_positions':27789,
            'individual_durable_token_stages':78,'generated_only_bulk_forwards':4,'registered_operation_counts':ops}})
    report={'kind':'INDEPENDENT_INFQ_EXPOSURE_PROJECTION_SAFE_METADATA_REVIEW001',
        'attributed_single_scope_closure':{'physical_forward_calls':84,'input_positions':28627,'selected_source_rows':4,
            'shared_source_parents':2,'source_parent_ids_from_committed_note':list(A.PARENTS),'accepted_scoped_projection':True,
            'partition':[{'role':role,'calls':n,'input_positions':p} for role,n,p in zip(A.REPORT_ROLES,(1,1,78,4),(419,419,26544,1245))],
            'global_exposure_coverage_established':False,'independent_per_call_prehook_token_telemetry_established':False,
            'complete_retention_disjointness_or_membership_qualified':False,'arithmetic_family_mapping':'UNDEFINED'}}
    scope='c6cabdae832bb844a7ee3bf66e30fb6e7515b8eef87d645571b563fad608c0ed'
    execution=seal({'kind':'ROOT_ACTUAL_INFQ_EXISTING_RECORD_CPU_PROJECTION001','exit_code':0,'calls':84,
        'input_positions':28627,'source_rows':4,'source_parents':list(A.PARENTS),'global_exposure_coverage':False,
        'source_recipe_reconstruction_not_independent_token_telemetry':True,'arithmetic_family_mapping':'UNDEFINED',
        'scope_file_sha256':scope,'output_files':{'PROJECTION_RECEIPT001.json':{'bytes':A.ROLES['receipt'][1],
            'sha256':A.ROLES['receipt'][2]}}})
    receipt=seal({'all95_original_byte_links_verified':True,'original_file_count':105,
        'source_recipe_reconstruction_verified':True,'global_exposure_coverage':False,
        'result_body_sha256':A.RESULT_BODY,'read_scope_sha256':scope})
    return {'audit':audit,'report':report,'pin_index':{'pin_count':13,'all13_named_pins_matched_before_report_sealing':True},
        'execution':execution,'receipt':receipt}

class Controls(unittest.TestCase):
    def test_original_nested_schema_compatibility_preserves_unknown_authority(self):
        docs=fixture();value=A._sanitized_contract(docs)
        self.assertEqual(84,value['original_schema_accounting']['observed_forward_calls'])
        self.assertEqual(28627,value['original_schema_accounting']['observed_input_positions'])
        self.assertEqual(2,len(value['sanitized_source_recipe']['source_parents']))
        self.assertFalse(value['qualified']);self.assertFalse(value['native_admission_authorized'])
        self.assertFalse(value['consumer_boundary']['modern_hash_closure_all_equal_field_synthesized'])
        self.assertNotIn('observed_model_input_calls',value);self.assertNotIn('input_ids',A.canonical(value))
        with self.assertRaisesRegex(ValueError,'UNKNOWN'):A.require_complete_lineage(value)
    def test_resealed_original_schema_and_accounting_mutations_refuse(self):
        for mode in ('top_level_only','wrong_config','count','bool','version','operation','renamed_operation'):
            docs=fixture();v=docs['audit']
            if mode=='top_level_only':v['config_body_sha256']=v['hash_closure'].pop('config_body_sha256')
            elif mode=='wrong_config':v['hash_closure']['config_file_sha256']='0'*64
            elif mode=='count':v['accounting']['actual_input_positions']-=1
            elif mode=='bool':v['accounting']['failed_API_calls']=False
            elif mode=='version':v['schema_version']=True
            elif mode=='operation':v['accounting']['registered_operation_counts']['tensor_copy']-=1
            else:v['accounting']['registered_operation_counts']['renamed_model_construct']=v['accounting']['registered_operation_counts'].pop('model_construct')
            seal(v)
            with self.subTest(mode=mode),self.assertRaises(ValueError):A._sanitized_contract(docs)
    def test_projection_metadata_joins_and_no_false_qualification(self):
        for mode in ('parents','result','receipt','scope','observed','global','partition','renamed_role'):
            docs=fixture()
            if mode=='parents':docs['execution']['source_parents'][0]='invented_parent'
            elif mode=='result':docs['receipt']['result_body_sha256']='0'*64
            elif mode=='receipt':docs['execution']['output_files']['PROJECTION_RECEIPT001.json']['sha256']='0'*64
            elif mode=='scope':docs['execution']['scope_file_sha256']='0'*64
            elif mode=='observed':docs['report']['attributed_single_scope_closure']['independent_per_call_prehook_token_telemetry_established']=True
            elif mode=='global':docs['execution']['global_exposure_coverage']=True
            elif mode=='partition':docs['report']['attributed_single_scope_closure']['partition'][2]['calls']=True
            else:docs['report']['attributed_single_scope_closure']['partition'][2]['role']='declared_observed_inputs'
            for k in ('execution','receipt'):seal(docs[k])
            with self.subTest(mode=mode),self.assertRaises(ValueError):A._sanitized_contract(docs)
        for raw in (b'{"x":1,"x":2}',b'{"x":1e999}',b'{"x":NaN}'):
            with self.assertRaises(ValueError):A.strict_json(raw)

if __name__=='__main__':
    def audit_hook(event,args):
        if event=='open' and type(args[0]) in (str,bytes):
            target=str(args[0]).replace('\\','/').lower()
            if '/experiments/rineya_synth_centralized_reasoning_v1/' in target:
                raise RuntimeError('authored tests cannot read package metadata/raw/source fixtures')
    sys.addaudithook(audit_hook)
    start=time.monotonic();result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    receipt={'kind':'AUTHORED_INFQ_METADATA_COMPATIBILITY_CONTROLS001','tests':result.testsRun,
        'errors':len(result.errors),'failures':len(result.failures),'success':result.wasSuccessful(),
        'seconds':time.monotonic()-start,'fixture_only':True,'genuine_metadata_or_raw_fixture_reads':False,
        'model_tokenizer_GPU_scoring_or_projection_actions':False,
        'source_pins':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('infq_arithmetic_metadata_adapter.py','test_infq_arithmetic_metadata_adapter.py')}}
    receipt['sha256']=A.digest(receipt)
    with (HERE/'CPU_CONTROLS001.json').open('xb') as f:f.write((json.dumps(receipt,sort_keys=True,indent=2)+'\n').encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)
