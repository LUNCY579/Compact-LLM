"""Production metadata/coherence controls; no model, OS or dispatch authority."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from experiments.skill_acquisition_native_v1 import ce_training_admission as t,ce_training_journal as j
from .test_ce_native_backend import context as backend_context


class Trap:
    def __getattribute__(self,name):raise AssertionError('work before admission')
    def __call__(self,*args,**kwargs):raise AssertionError('work before admission')


def metadata():
    root=Path(__file__).resolve().parents[3]
    proposal=json.loads((root/'experiments/skill_acquisition_method_lab_v1/CE8_REGISTRATION_PROPOSAL_20261001.json').read_bytes())
    plan_ref=proposal['native_backend']['native_ledger_artifact_interface']['per_phase_arm']['qualification']['canonical']['plan_ref']
    plan=json.loads(Path(plan_ref['path']).read_bytes());prep=json.loads(Path(proposal['input_refs']['preparation']['path']).read_bytes())
    ref=lambda name:{'path':str(Path(tempfile.gettempdir())/(name+'.json')),'sha256':t.digest(name)}
    refs={key:ref(key) for key in t.controller.REFS};refs['preparation_ref']=proposal['input_refs']['preparation']
    refs['qualifiedreceipt_ref']=ref('qualified-receipt')
    bounds={key:1000000 for key in t.controller.BOUND_FIELDS};request=t.controller.build_request('training','canonical',refs,bounds)
    request.update(mode='registered_scope',approved=True,training_authorized=True,qualification_authorized=False,status='AUTHORIZED_TRAINING')
    code={key:'a'*64 for key in t.CORE_SOURCE_PATHS};recipe=proposal['static_recipe'];recipe_sha=t.digest(recipe)
    runtime={'decoding_preimage':proposal['runtime_preimage'],'decoding_sha256':proposal['runtime_preimage_sha256']}
    declared=t.q.declared_census(prep,'canonical');census=dict(declared,rows=declared['rows_rendered']);census.pop('rows_rendered')
    expected={'prepared_row_bindings':[{'id':rid} for batch in plan['ordered_plan'] for rid in batch['row_ids']],'census':census}
    expected['receipt_sha256']=t.digest(expected)
    native={'output_directory':str(Path(tempfile.gettempdir())/'production'),'operation_bounds':{'fixture':{}},
            'operation_bounds_status':'AUTHORIZED_BOUNDED_PRODUCTION_UPPER'}
    policy={'schema':t.PRICE_POLICY,'approved':True,'basis':native['operation_bounds_status'],'measured_prices':False,
            'operation_bounds_sha256':t.digest(native['operation_bounds'])}
    scope={'schema':t.SCOPE,'role':'production_training','approved':True,'production_training_authorized':True,
        'qualification_authorized':False,'evaluation_authorized':False,'arm':'canonical','allowed_data_roles':['train'],
        'choice_sha256':refs['selectedoperatorchoice_ref']['sha256'],'static_recipe_sha256':recipe_sha,'native_backend':native,
        'execution':{'rows':3000,'epochs':1,'boundaries':375,'checkpoint_boundaries':t.SCHEDULE,'run_seed':20261001},
        'resources_sha256':refs['resources_ref']['sha256'],'attempt_caps_sha256':refs['attempt_caps_ref']['sha256'],
        'production_price_policy':policy,'native_ledger_ref':ref('ledger')}
    qualified={'completed_pair_verified':True,'arm':'canonical','receipt_ref':refs['qualifiedreceipt_ref'],
        'static_recipe_sha256':recipe_sha,'source_code_sha256':code,'runtime':runtime,'anchors':proposal['anchors'],
        'census':census,'prepared_receipt_sha256':expected['receipt_sha256']}
    justify={'schema':t.PRICE_JUSTIFICATION,'approved':True,'production_training_authorized':True,'basis':policy['basis'],
        'measured_prices':False,'prices_are_measurements':False,'reservation_basis':'explicit fixture upper scope, not a measurement',
        'static_recipe_sha256':recipe_sha,'qualified_receipt_sha256':refs['qualifiedreceipt_ref']['sha256'],
        'source_code_sha256':code,'operation_bounds_sha256':policy['operation_bounds_sha256'],
        'production_budget':{'rows':3000,'epochs':1,'updates':375,'checkpoint_boundaries':t.SCHEDULE,
            'resources_sha256':refs['resources_ref']['sha256'],'attempt_caps_sha256':refs['attempt_caps_ref']['sha256']}}
    bindings={'scope':scope,'selectedoperatorchoice':{'schema':'CE8-OPERATOR-CHOICE-1','approved':True,'choice':'canonical_only',
        'new_experiment':True,'preserved_stop_receipt_sha256':t.q.WARM_STOP},'recipe':{'recipe':recipe},'source':{'code_sha256':code},
        'runtime':runtime,'anchors':proposal['anchors'],'plan':plan,'preparation':prep,
        'ingestion':{'schema':'CE8-INGESTION-SPEC-1','arm':'canonical','plan_sha256':plan['plan_sha256'],
            'preparation_ref':refs['preparation_ref'],'expected_census':declared},
        'evaluation':{'evaluation_authorized':False,'allowed_data_roles':['train']},
        'scope_authorization':{'schema':t.AUTHORIZATION,'approved':True,'training_authorized':True,'qualification_authorized':False,
            'request_preimage_sha256':t.digest(t.controller.request_preimage(request))},
        'production_price_justification':justify,'native_ledger_ref':scope['native_ledger_ref'],
        'expected_prepared_receipt':{'expected_receipt':expected}}
    bindings['native_ledger']={'schema':'CE8-NATIVE-LEDGER-SPEC-1','owned_directory':str(Path(native['output_directory'])/'ledger'),
        'plan':plan,'ingestion_receipt_sha256':expected['receipt_sha256'],'bindings':{'recipe':recipe,'source':code,
            'runtime':runtime['decoding_preimage'],'anchors':{key:proposal['anchors'][key] for key in ('own_start_sha256','common_anchor_sha256')},
            'prepared_rows':expected['prepared_row_bindings']}}
    return request,bindings,qualified,code


class Tests(unittest.TestCase):
    def test_closed_native_admission_and_journal_refuse_before_paths_framework_or_callbacks(self):
        with self.assertRaises(ValueError):t.admit_training(Trap(),code_sha256=Trap(),outer_admission=Trap(),
            io_observer=Trap(),qualified_reader=Trap(),qualifier_preflight=Trap())
        with self.assertRaises(ValueError):j.native_journal(Trap(),Trap(),Trap(),False,None,Trap(),Trap())
        self.assertNotIn('torch',sys.modules);self.assertNotIn('transformers',sys.modules)
    def test_full375_shape_is_coherent_and_bounded_prices_remain_unmeasured(self):
        request,bindings,qualified,code=metadata();shape=t.validate_bindings(request,bindings,qualified,code)
        self.assertEqual(shape['dose'],{'rows':3000,'epochs':1,'updates':375,'states':376,'microbatch_rows':1,'equal_row_denominator':8})
        self.assertFalse(shape['admission_evidence_verified']);self.assertFalse(shape['native_qualified']);self.assertFalse(shape['unit_prices_measured'])
    def test_pair_scope_source_census_and_price_discounts_refuse(self):
        request,bindings,qualified,code=metadata()
        edits=(lambda b,q:b['scope']['execution'].update(boundaries=376),lambda b,q:q.update(completed_pair_verified=False),
            lambda b,q:b['scope'].update(evaluation_authorized=True),lambda b,q:b['anchors'].update(own_start_sha256='b'*64),
            lambda b,q:q['census'].update(total_prompt_tokens=q['census']['total_prompt_tokens']+1),
            lambda b,q:b['production_price_justification']['production_budget'].update(updates=374),
            lambda b,q:b['scope']['production_price_policy'].update(measured_prices=True),
            lambda b,q:q['source_code_sha256'].update(**{t.CORE_SOURCE_PATHS[1]:'b'*64}))
        for edit in edits:
            changed,proof=copy.deepcopy(bindings),copy.deepcopy(qualified);edit(changed,proof)
            with self.assertRaises(ValueError):t.validate_bindings(request,changed,proof,code)
    def test_fixed_backend_ast_bridge_accepts_guard_only_and_rejects_numerical_change(self):
        old=b'def validate_backend_spec(context):\n return "old"\ndef update(x):\n return x+1\n'
        new=b'def validate_backend_spec(context):\n return "new"\ndef update(x):\n return x+1\n'
        contract={'schema':t.BACKEND_COMPATIBILITY,'approved':True,'module':t.CORE_SOURCE_PATHS[0],
            'changed_symbols':['validate_backend_spec'],'qualified_backend_ref':{'sha256':hashlib.sha256(old).hexdigest()},
            'production_backend_ref':{'sha256':hashlib.sha256(new).hexdigest()}}
        result=t.backend_gate_compatibility(old,new,contract,contract['qualified_backend_ref']['sha256'],contract['production_backend_ref']['sha256'])
        self.assertTrue(result['numerical_ast_unchanged'])
        numerical=new.replace(b'x+1',b'x+2');bad=copy.deepcopy(contract);bad['production_backend_ref']['sha256']=hashlib.sha256(numerical).hexdigest()
        with self.assertRaisesRegex(ValueError,'outside admission'):t.backend_gate_compatibility(old,numerical,bad,
            bad['qualified_backend_ref']['sha256'],bad['production_backend_ref']['sha256'])
    def test_native_journal_adapter_calls_existing_backend_before_model_and_rejects_foreign_owner(self):
        context={'phase':'training','qualifiedreceipt_verified':True,'production_training_authorized':True,'native_services':{}}
        operation=object.__new__(j.StableDispatcher);operation.ledger=None;gate=object();permit=object()
        context['native_execution_permit']=permit
        backend=object.__new__(j.NativeCEBackend);backend.context=context;backend.permit=permit;backend.operation=operation;backend.gate=gate
        backend.model=backend.optimizer=None;calls=[];backend.open_journal=lambda *args:calls.append(args) or 'actual-delegate'
        context['native_services']['training_backend']=backend
        with patch.object(j.authority,'require_execution_permit',return_value=None):
            self.assertEqual(j.native_journal(context,'prepared','proof',False,None,operation,gate),'actual-delegate')
            self.assertEqual(calls,[(context,'prepared','proof',False,None)])
            backend.model=object()
            with self.assertRaises(ValueError):j.native_journal(context,'prepared','proof',False,None,operation,gate)
    def test_backend_bounded_status_needs_verified_pair_and_raw_full_budget(self):
        from experiments.skill_acquisition_native_v1 import ce_native_backend as b
        value=backend_context();value['phase']='training';value['qualifiedreceipt_verified']=value['production_training_authorized']=True
        scope=value['bindings']['scope'];scope.update(role='production_training',production_training_authorized=True,
            execution={'rows':3000,'epochs':1,'boundaries':375,'run_seed':123})
        native=scope['native_backend'];native['operation_bounds_status']='AUTHORIZED_BOUNDED_PRODUCTION_UPPER'
        value['completed_qualification']={'completed_pair_verified':True,'receipt_ref':{'sha256':'b'*64}}
        policy={'schema':t.PRICE_POLICY,'approved':True,'basis':native['operation_bounds_status'],'measured_prices':False,
            'operation_bounds_sha256':t.digest(native['operation_bounds'])};scope['production_price_policy']=policy
        value['bindings']['production_price_justification']={'schema':t.PRICE_JUSTIFICATION,'approved':True,'production_training_authorized':True,
            'basis':policy['basis'],'measured_prices':False,'prices_are_measurements':False,'operation_bounds_sha256':policy['operation_bounds_sha256'],
            'static_recipe_sha256':value['static_recipe_sha256'],'source_code_sha256':value['bindings']['source']['code_sha256'],
            'qualified_receipt_sha256':'b'*64,'production_budget':{'rows':3000,'epochs':1,'updates':375,'checkpoint_boundaries':t.SCHEDULE}}
        b.validate_backend_spec(value)
        changed=copy.deepcopy(value);changed['completed_qualification']['completed_pair_verified']=False
        with self.assertRaisesRegex(ValueError,'actual verified'):b.validate_backend_spec(changed)
        changed=copy.deepcopy(value);changed['bindings']['production_price_justification']['measured_prices']=True
        with self.assertRaisesRegex(ValueError,'not measured'):b.validate_backend_spec(changed)


if __name__=='__main__':unittest.main()
