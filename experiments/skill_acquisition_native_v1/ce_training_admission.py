"""Source-admitted canonical production metadata; no permit or dispatch grant.

The raw completed qualification receipt and finalizer costs are one dependency.
Separate production scope/price authorization is required. Synthetic callbacks
and receipt flags cannot substitute for the existing opaque outer authority or
the source-pinned completed-pair consumer. No framework is imported here.
"""
from __future__ import annotations

import copy
import ast
import hashlib
from pathlib import Path
import time
import uuid

from . import ce_controller as controller, ce_qualification as q
from . import ce_qualification_artifacts as raw, ce_plan

SCOPE='CE8-NATIVE-PRODUCTION-SCOPE-1'
AUTHORIZATION='CE8-PRODUCTION-AUTHORIZATION-1'
PRICE_POLICY='CE8-PRODUCTION-PRICE-POLICY-1'
PRICE_JUSTIFICATION='CE8-PRODUCTION-PRICE-JUSTIFICATION-1'
BACKEND_COMPATIBILITY='CE8-PRODUCTION-BACKEND-GATE-COMPATIBILITY-1'
SCHEDULE=[0,1,75,150,225,300,375]
CORE_SOURCE_PATHS=('experiments/skill_acquisition_native_v1/ce_native_backend.py',
    'experiments/skill_acquisition_native_v1/ce_update.py','experiments/skill_acquisition_native_v1/ce_journal.py',
    'experiments/skill_acquisition_native_v1/ce_ingestion.py','experiments/skill_acquisition_native_v1/ce_plan.py',
    'experiments/skill_acquisition_native_v1/checkpoint.py','experiments/skill_acquisition_native_v1/update.py',
    'experiments/maxrl_comp002_pretraining_v1/s0_runtime.py','experiments/maxrl_comp002_pretraining_v1/s0_warmstart.py')
require,canonical,digest=raw.require,raw.canonical,raw.digest


def _ref(value):
    require(type(value) is dict and type(value.get('path')) is str and Path(value['path']).is_absolute()
            and raw._hex(value.get('sha256')),'explicit source-bound production reference')
    return value


def _read(reference,operation,remaining,*,json_object=True):
    _ref(reference)
    # A finite upper is reserved before path inspection; stat cannot silently
    # enlarge an already admitted scope. All actual bytes must match the ref.
    upper=reference.get('bytes',raw.MAX_JSON)
    require(type(upper) is int and 0<upper<=raw.MAX_JSON and remaining['io_read_bytes_upper']>=upper+1
            and remaining['identity_hash_attempts_upper']>=1 and remaining['identity_hash_bytes_upper']>=upper+1,
            'production metadata permanent read/hash upper before paths')
    identifier=uuid.uuid4().hex;started=time.perf_counter()
    operation('qualification_metadata','started',operation_id=identifier,read_bytes_upper=upper+1,
        write_bytes_upper=0,identity_hash_attempts_upper=1,identity_hash_bytes_upper=upper+1)
    remaining['io_read_bytes_upper']-=upper+1;remaining['identity_hash_attempts_upper']-=1
    remaining['identity_hash_bytes_upper']-=upper+1
    try:
        path=Path(reference['path']);require(path.is_file() and not path.is_symlink() and 0<path.stat().st_size<=upper,
                                             'regular bounded production metadata')
        with path.open('rb') as stream:blob=stream.read(upper+1)
        require(0<len(blob)<=upper and hashlib.sha256(blob).hexdigest()==reference['sha256']
            and ('bytes' not in reference or len(blob)==reference['bytes']),'production raw metadata bytes/hash changed')
        value=raw._json(blob) if json_object else blob
        require(not json_object or type(value) is dict,'production raw object required')
    except BaseException as primary:
        try:operation('qualification_metadata','failed',operation_id=identifier,wall_s=time.perf_counter()-started,
            error={'type':type(primary).__name__,'message':str(primary)[:1000]})
        except BaseException as failure:raw._accounting(primary,failure);raise primary from failure
        raise
    operation('qualification_metadata','returned',operation_id=identifier,bytes_read=len(blob),bytes_written=0,
        identity_hash_attempts_actual=1,identity_hash_bytes_actual=len(blob),wall_s=time.perf_counter()-started)
    return value


def backend_gate_compatibility(old_bytes,new_bytes,contract,qualified_sha,production_sha):
    """One fixed metadata symbol may change; all numerical bodies stay exact."""
    require(type(contract) is dict and contract.get('schema')==BACKEND_COMPATIBILITY and contract.get('approved') is True
            and contract.get('changed_symbols')==['validate_backend_spec']
            and contract.get('module')==CORE_SOURCE_PATHS[0]
            and contract['qualified_backend_ref']['sha256']==qualified_sha
            and contract['production_backend_ref']['sha256']==production_sha,
            'separately approved exact backend admission-only source bridge')
    require(type(old_bytes) is bytes and type(new_bytes) is bytes and len(old_bytes)<=raw.MAX_JSON and len(new_bytes)<=raw.MAX_JSON
            and hashlib.sha256(old_bytes).hexdigest()==qualified_sha and hashlib.sha256(new_bytes).hexdigest()==production_sha,
            'pinned actual old/new backend source bytes')
    def unchanged(blob):
        tree=ast.parse(blob)
        matches=[node for node in tree.body if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name=='validate_backend_spec']
        require(len(matches)==1,'one exact top-level backend admission symbol')
        tree.body=[node for node in tree.body if node is not matches[0]]
        return ast.dump(tree,include_attributes=False)
    require(unchanged(old_bytes)==unchanged(new_bytes),'backend numerical/runtime/CE/checkpoint bodies changed outside admission guard')
    return {'schema':BACKEND_COMPATIBILITY,'qualified_backend_sha256':qualified_sha,
            'production_backend_sha256':production_sha,'changed_symbols':['validate_backend_spec'],'numerical_ast_unchanged':True}


def validate_bindings(request,bindings,qualified,code_sha256):
    """Pure shape/coherence validation, never source authority or execution."""
    scope=bindings['scope'];choice=bindings['selectedoperatorchoice'];recipe=bindings['recipe']['recipe']
    require(choice.get('schema')=='CE8-OPERATOR-CHOICE-1' and choice.get('approved') is True
            and choice.get('choice')=='canonical_only' and choice.get('new_experiment') is True
            and choice.get('preserved_stop_receipt_sha256')==q.WARM_STOP,'canonical new study preserves original STOP')
    require(scope.get('schema')==SCOPE and scope.get('role')=='production_training'
            and scope.get('approved') is True and scope.get('production_training_authorized') is True
            and scope.get('qualification_authorized') is False and scope.get('evaluation_authorized') is False
            and scope.get('arm')=='canonical' and scope.get('allowed_data_roles')==['train']
            and scope.get('choice_sha256')==request['selectedoperatorchoice_ref']['sha256'],'separate TRAIN-only production scope')
    require(type(qualified) is dict and qualified.get('completed_pair_verified') is True
            and qualified.get('arm')=='canonical' and _ref(qualified['receipt_ref'])==request['qualifiedreceipt_ref'],
            'source-completed qualification receipt and original finalizer cost pair required')
    require(digest(recipe)==scope.get('static_recipe_sha256')==qualified.get('static_recipe_sha256')
            and recipe.get('synthetic_cpu') is False and recipe.get('accumulation_rows')==8
            and type(recipe.get('accumulation_rows')) is int and recipe.get('microbatch_rows')==1
            and recipe.get('loss_reduction')=='mean_labeled_tokens_per_row_then_equal_rows'
            and recipe.get('optimizer_origin')=='FRESH_PER_ARM','same qualified fixed CE8 recipe')
    source=bindings['source']
    require(type(code_sha256) is dict and code_sha256 and source.get('code_sha256')==code_sha256,
            'production full current source bundle')
    qualified_code=qualified.get('source_code_sha256')
    require(type(qualified_code) is dict and all(raw._hex(code_sha256.get(path))
        and code_sha256[path]==qualified_code.get(path) for path in CORE_SOURCE_PATHS[1:]),
        'qualified numerical/ingestion/CE/checkpoint/runtime kernel source changed')
    if code_sha256.get(CORE_SOURCE_PATHS[0])!=qualified_code.get(CORE_SOURCE_PATHS[0]):
        bridge=bindings.get('backend_gate_compatibility_verified')
        require(type(bridge) is dict and bridge.get('schema')==BACKEND_COMPATIBILITY
            and bridge.get('qualified_backend_sha256')==qualified_code.get(CORE_SOURCE_PATHS[0])
            and bridge.get('production_backend_sha256')==code_sha256.get(CORE_SOURCE_PATHS[0])
            and bridge.get('changed_symbols')==['validate_backend_spec'] and bridge.get('numerical_ast_unchanged') is True,
            'source-derived exact backend admission bridge required')
    runtime=bindings['runtime'];anchors=bindings['anchors']
    require(runtime==qualified.get('runtime')
            and runtime.get('decoding_sha256')==digest(runtime['decoding_preimage'])
            and anchors==qualified.get('anchors') and anchors.get('own_start_sha256')==anchors.get('common_anchor_sha256'),
            'same qualified runtime and independent common warm anchor')
    execution=scope.get('execution')
    require(controller.checkpoint_boundaries(execution)==tuple(SCHEDULE),'seven fixed production checkpoint publications')
    require(execution.get('rows')==3000 and type(execution['rows']) is int
            and execution.get('epochs')==1 and type(execution['epochs']) is int
            and execution.get('run_seed')==20261001 and type(execution['run_seed']) is int,
            'explicit one seed/one pass/3000-row production dose')
    batches=ce_plan.validate_plan(bindings['plan'])
    require(len(batches)==375 and sum(map(len,batches))==3000 and bindings['plan']['label_binding']['arm']=='canonical'
            and bindings['plan']['preparation']['sha256']==request['preparation_ref']['sha256'],
            'same complete deterministic canonical CE8 target plan')
    preparation=bindings['preparation'];ingestion=bindings['ingestion']
    qualified_census=qualified.get('census');require(type(qualified_census) is dict,'actual qualified token census')
    mapped_census=dict(qualified_census,rows_rendered=qualified_census.get('rows'))
    declared=q.declared_census(preparation,'canonical')
    require(preparation.get('approved') is False and preparation.get('training_authorized') is False
            and ingestion.get('schema')=='CE8-INGESTION-SPEC-1' and ingestion.get('arm')=='canonical'
            and ingestion.get('plan_sha256')==bindings['plan']['plan_sha256']
            and ingestion.get('preparation_ref')==request['preparation_ref']
            and ingestion.get('expected_census')==declared
            and all(mapped_census.get(key)==value for key,value in declared.items()),
            'qualified source census and actual fresh ingestion expectation')
    evaluation=bindings['evaluation']
    require(evaluation.get('evaluation_authorized') is False and evaluation.get('allowed_data_roles')==['train'],
            'production scope cannot consume sealed evaluation data')
    require(scope.get('resources_sha256')==request['resources_ref']['sha256']
            and scope.get('attempt_caps_sha256')==request['attempt_caps_ref']['sha256'],'frozen resource and total attempt caps')
    authorization=bindings['scope_authorization']
    require(authorization.get('schema')==AUTHORIZATION and authorization.get('approved') is True
            and authorization.get('training_authorized') is True and authorization.get('qualification_authorized') is False
            and authorization.get('request_preimage_sha256')==digest(controller.request_preimage(request)),
            'exact separate production authorization preimage')
    native=scope['native_backend'];policy=scope.get('production_price_policy');justification=bindings['production_price_justification']
    require(type(policy) is dict and policy.get('schema')==PRICE_POLICY and policy.get('approved') is True
            and policy.get('basis') in ('MEASURED_PRODUCTION','AUTHORIZED_BOUNDED_PRODUCTION_UPPER')
            and policy.get('measured_prices') is (policy['basis']=='MEASURED_PRODUCTION')
            and native.get('operation_bounds_status')==policy['basis']
            and policy.get('operation_bounds_sha256')==digest(native['operation_bounds']),
            'explicit measured or separately authorized bounded production prices')
    require(justification.get('schema')==PRICE_JUSTIFICATION and justification.get('approved') is True
            and justification.get('production_training_authorized') is True and justification.get('basis')==policy['basis']
            and justification.get('measured_prices') is policy['measured_prices']
            and justification.get('static_recipe_sha256')==scope['static_recipe_sha256']
            and justification.get('qualified_receipt_sha256')==request['qualifiedreceipt_ref']['sha256']
            and justification.get('source_code_sha256')==code_sha256
            and justification.get('operation_bounds_sha256')==policy['operation_bounds_sha256'],
            'raw production price justification bound to this qualified recipe/source/operation envelope')
    require(justification.get('production_budget')=={'rows':3000,'epochs':1,'updates':375,
        'checkpoint_boundaries':SCHEDULE,'resources_sha256':request['resources_ref']['sha256'],
        'attempt_caps_sha256':request['attempt_caps_ref']['sha256']},'explicit frozen full375 production budget justification')
    if policy['measured_prices']:
        require(qualified.get('unit_prices_measured') is True and justification.get('complete_model_checkpoint_os_unit_prices') is True
                and type(justification.get('measurement_refs')) is list and justification['measurement_refs'],
                'measured label requires complete actual unit measurement provenance')
        for ref in justification['measurement_refs']:_ref(ref)
    else:
        require(justification.get('prices_are_measurements') is False
                and type(justification.get('reservation_basis')) is str and justification['reservation_basis'],
                'bounded production reservations remain explicitly unmeasured')
    ledger=bindings['native_ledger']
    expectation=bindings.get('expected_prepared_receipt',{}).get('expected_receipt')
    require(type(expectation) is dict and expectation.get('receipt_sha256')==qualified.get('prepared_receipt_sha256')
            and digest({key:value for key,value in expectation.items() if key!='receipt_sha256'})==expectation['receipt_sha256'],
            'exact full qualified prepared receipt expectation, never counts alone')
    require(scope.get('native_ledger_ref')==bindings['native_ledger_ref'] and ledger.get('schema')=='CE8-NATIVE-LEDGER-SPEC-1'
            and ledger.get('plan')==bindings['plan'] and ledger.get('bindings',{}).get('recipe')==recipe
            and ledger['bindings'].get('source')==code_sha256 and ledger['bindings'].get('runtime')==runtime['decoding_preimage']
            and ledger['bindings'].get('anchors')=={key:anchors[key] for key in ('own_start_sha256','common_anchor_sha256')}
            and ledger.get('ingestion_receipt_sha256')==expectation['receipt_sha256']
            and ledger['bindings'].get('prepared_rows')==expectation.get('prepared_row_bindings')
            and ledger.get('owned_directory')==str(Path(native['output_directory'])/'ledger'),
            'exact source-bound native durable CE ledger artifact')
    return {'dose':{'rows':3000,'epochs':1,'updates':375,'states':376,'microbatch_rows':1,'equal_row_denominator':8},
            'execution':copy.deepcopy(execution),'static_recipe_sha256':scope['static_recipe_sha256'],
            'price_basis':policy['basis'],'unit_prices_measured':policy['measured_prices'],
            'admission_evidence_verified':False,'native_qualified':False}


def admit_training(request,*,code_sha256,outer_admission,io_observer,qualified_selection=None,
                   qualified_reader,qualifier_preflight):
    controller.validate_request(request)
    require(request['phase']=='training' and request['arm']=='canonical','production canonical training only')
    require(type(outer_admission) is controller.OuterAdmission and outer_admission.context.get('mode')=='registered_scope',
            'original source-issued production metadata outer capability')
    outer_admission._verify_issuer() # empty source registry refuses before any file/import work
    require(callable(io_observer) and type(qualified_reader) is raw.RawEvidenceReader and callable(qualifier_preflight),
            'same independently prepaid qualified-pair reader and production observer')
    refs={key:request[key] for key in controller.REFS}
    outer=outer_admission('qualification_metadata',refs)
    require(outer['mode']=='registered_scope' and outer['phase']=='training','real production outer prepayment')
    from . import ce_qualified_receipt as completed
    if qualified_selection is None:qualified_selection=completed.selected_qualification()
    qualified=completed.verify_completed_qualification(qualified_selection,reader=qualified_reader,outer_preflight=qualifier_preflight)
    remaining=copy.deepcopy(request['bootstrap_bounds']);controller._bounds(remaining)
    bindings={key[:-4]:_read(ref,io_observer,remaining) for key,ref in refs.items()}
    scope=bindings['scope'];ledger_ref=_ref(scope.get('native_ledger_ref'))
    policy=scope.get('production_price_policy');require(type(policy) is dict,'explicit production price policy before bound read')
    justification_ref=_ref(policy.get('justification_ref'))
    bindings['native_ledger_ref']=copy.deepcopy(ledger_ref)
    bindings['native_ledger']=_read(ledger_ref,io_observer,remaining)
    bindings['production_price_justification']=_read(justification_ref,io_observer,remaining)
    bindings['expected_prepared_receipt']=_read(_ref(scope.get('expected_prepared_receipt_ref')),io_observer,remaining)
    qualified_code=qualified.get('source_code_sha256',{});backend_path=CORE_SOURCE_PATHS[0]
    if qualified_code.get(backend_path)!=code_sha256.get(backend_path):
        contract=_read(_ref(scope.get('backend_gate_compatibility_ref')),io_observer,remaining)
        old_ref,new_ref=_ref(contract.get('qualified_backend_ref')),_ref(contract.get('production_backend_ref'))
        require(old_ref['sha256']==qualified_code.get(backend_path) and new_ref==bindings['source']['source_files'][backend_path],
                'qualified old and actual production source file references')
        old_bytes=_read(old_ref,io_observer,remaining,json_object=False);new_bytes=_read(new_ref,io_observer,remaining,json_object=False)
        identifier=uuid.uuid4().hex
        io_observer('scope_admission','started',operation_id=identifier,read_bytes_upper=0,write_bytes_upper=0,
                    identity_hash_attempts_upper=2,identity_hash_bytes_upper=len(old_bytes)+len(new_bytes))
        try:bindings['backend_gate_compatibility_verified']=backend_gate_compatibility(old_bytes,new_bytes,contract,old_ref['sha256'],new_ref['sha256'])
        except BaseException as error:
            io_observer('scope_admission','failed',operation_id=identifier,error={'type':type(error).__name__,'message':str(error)[:1000]});raise
        io_observer('scope_admission','returned',operation_id=identifier,identity_hash_attempts_actual=2,
                    identity_hash_bytes_actual=len(old_bytes)+len(new_bytes))
    shape=validate_bindings(request,bindings,qualified,code_sha256)
    return {'schema':controller.CONTEXT,'mode':'registered_scope','phase':'training','arm':'canonical',
        'request_sha256':digest(controller.request_preimage(request)),'scope_sha256':request['scope_ref']['sha256'],
        'static_recipe_sha256':shape['static_recipe_sha256'],'bindings':bindings,
        'refs':dict(copy.deepcopy(refs),qualifiedreceipt_ref=copy.deepcopy(request['qualifiedreceipt_ref']),native_ledger_ref=copy.deepcopy(ledger_ref)),
        'operator_choice':'canonical_only','execution':shape['execution'],'admission_evidence_verified':True,
        'qualifiedreceipt_verified':True,'production_training_authorized':True,'qualification_authorized':False,
        'completed_qualification':copy.deepcopy(qualified),'bootstrap_remaining':remaining,
        'price_basis':shape['price_basis'],'unit_prices_measured':shape['unit_prices_measured']}
