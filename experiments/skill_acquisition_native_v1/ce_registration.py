"""Create-only canonical qualification registration drafts; no execution grants.

Counts/bytes are analytical envelopes and explicit abort ceilings. Wall limits
are reservations for discarded qualification, not measured production prices.
Historical observed token receipts remain unchanged; current-source expectations
are explicitly separate and must be recomputed by real preparation before load.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
from pathlib import Path

from . import ce_controller as controller, ce_qualification as qualification
from . import ce_native_authority as authority, ce_native_backend as backend
from . import ce_native_qualification as executor, ce_qualification_accounting as accounting
from . import ce_checkpoint_qualification as checkpoint, ce_checkpoint_lease as lease
from . import ce_checkpoint_local_issuer as local, ce_resource_evidence as resource
from . import ce_resource_qualification as measurement, ce_bootstrap_accounting as bootstrap
from . import ce_publication as publication, ce_checkpoint_publication as branch_publication
from . import ce_checkpoint_processes as processes, ce_checkpoint_initializer as initializer
from . import ce_ingestion as ingestion, runner

SCHEMA = 'CE8-CANONICAL-QUALIFICATION-DRAFT-1'
STATUS = 'DRAFT_UNAUTHORIZED_DISCARDED_QUALIFICATION'
DECISION_SHA = '9e39a9d443f4f4343bb6ad6c21192b97e1a7964d6195dd360ac0a3b5e5377fe5'
PROPOSAL_SHA = 'f7ba366640be2e932b08bb41e6a056f88deb6c6295509a7701e269923ebd448e'
MIB, GIB = 1024**2, 1024**3
require, canonical, digest = authority.require, authority.canonical, authority.digest


def _clone(value): return json.loads(canonical(value))
def _ceil(value): return 1 << max(0,(max(1,value)-1).bit_length())
def profile(read=0,write=0,hashes=0,hashed=0,wall=1):
    return dict(read_bytes_upper=read,write_bytes_upper=write,identity_hash_attempts_upper=hashes,
                identity_hash_bytes_upper=hashed,wall_s_upper=wall)
def costs(read=0,write=0,hashes=0,hashed=0,metadata=0,attempts=1,wall=1):
    return dict(io_read_bytes_upper=read,io_write_bytes_upper=write,identity_hash_attempts_upper=hashes,
                identity_hash_bytes_upper=hashed,metadata_bytes_upper=metadata,operation_attempts_upper=attempts,
                wall_seconds_upper=wall)


def raw_reference(path,*,expected_sha=None,maximum=4*MIB):
    path=Path(path); require(path.is_absolute() and path.is_file() and not path.is_symlink(), 'exact existing metadata reference')
    require(0<path.stat().st_size<=maximum, 'registration metadata byte ceiling; no model payload read')
    raw=path.read_bytes(); hashed=hashlib.sha256(raw).hexdigest()
    require(expected_sha is None or hashed==expected_sha, 'pinned registration input changed')
    return {'path':str(path),'sha256':hashed,'bytes':len(raw)},raw


def source_snapshot(checkout):
    """Same complete package inventory as the actual controller, including tests."""
    fields={}; counts=dict(read_bytes=0,hashes=0)
    def observe(kind,phase,operation_id,**metadata):
        if phase=='started': counts['read_bytes']+=metadata['read_bytes_upper']; counts['hashes']+=metadata['identity_hash_attempts_upper']
    code=runner.code_bundle(observe,root=Path(checkout))
    for name,hashed in code.items(): fields[name]={'path':str(Path(checkout)/name),'sha256':hashed,'bytes':(Path(checkout)/name).stat().st_size}
    return {'schema':'CE8-SOURCE-SNAPSHOT-1','code_sha256':code,'source_files':fields,
            'source_checkout':str(checkout),'source_freeze':False,'metadata_reads':counts}


def metadata_source_preview(preparation_ref,plan_ref,recipe_sha):
    """Real full source revalidation using an explicit non-native metadata route."""
    starts=[]
    def outer(stage,refs):
        return {'schema':controller.OUTER,'mode':'synthetic_fixture','admitted':True,'phase':'qualification',
            'arm':'canonical','request_sha256':digest({'preview':refs}), 'scope_sha256':digest('metadata-preview'),
            'source_sha256':digest('not-native-authority'),'runtime_sha256':digest('no-frameworks'),
            'static_recipe_sha256':recipe_sha}
    def observe(kind,phase,**fields):
        if phase=='started': starts.append({'kind':kind,**fields})
    refs=lambda ref:{key:ref[key] for key in ('path','sha256')}
    verified=ingestion.verify_sources(refs(preparation_ref),refs(plan_ref),'canonical',outer,operation=observe,gate=lambda stage:None)
    return {'schema':'CE8-NONNATIVE-SOURCE-REVALIDATION-1','native_qualified':False,'native_execution_authorized':False,
            'tokenizer_recomputed':False,'model_loads':0,'receipt':verified.receipt,
            'declared_operation_reservations':starts,
            'classification':'actual TRAIN/teacher source revalidation; synthetic admission receipt carries no native grant'}


def expected_receipt(prior,source_receipt,recipe_sha):
    require(prior.get('schema')==ingestion.SCHEMA and prior.get('arm')=='canonical'
            and prior.get('model_loads')==prior.get('optimizer_steps')==0
            and prior.get('tokenizer_rendered') is True and prior.get('training_authorized') is False,
            'prior actual canonical tokenizer-only receipt required')
    body={key:_clone(value) for key,value in prior.items() if key!='receipt_sha256'}
    body['static_recipe_sha256']=recipe_sha; body['source_receipt_sha256']=source_receipt['receipt_sha256']
    return dict(body,receipt_sha256=digest(body))


def qualification_profiles(proposal,receipt,source):
    """No fixture prices: known payload/inventory work plus declared abort ceilings."""
    spec=proposal['native_backend']; asset=json.loads(spec['asset_receipt_utf8'])
    n=spec['inventory']['numel']; selected=4*n; full=2*asset['model_header']['data_bytes']
    tensors=asset['model_header']['tensors']; names=len(spec['parameter_inventory'])
    metadata=64*MIB; source_bytes=sum(ref['bytes'] for ref in source['source_files'].values())
    source_count=len(source['source_files'])
    max_hash=max(row['prepared_hash_bytes_upper'] for row in receipt['prepared_row_bindings'])
    ceiling=proposal['native_backend']['operation_bounds']
    profiles={kind:profile(wall=5) for kind in accounting.KINDS}
    for kind,declared in ceiling.items():
        if kind in profiles: profiles[kind]=dict(declared,wall_s_upper=5)
    model_hashes=min(ceiling['model_load']['identity_hash_attempts_upper'],12*(tensors+names+16))
    profiles['model_load']=dict(ceiling['model_load'],identity_hash_attempts_upper=model_hashes,wall_s_upper=600)
    profiles['tokenizer_load']=dict(ceiling['tokenizer_load'],identity_hash_attempts_upper=64,wall_s_upper=60)
    profiles['module_import']=profile(_ceil(source_bytes+metadata),0,source_count+32,_ceil(source_bytes+metadata),30)
    state_bytes=full+2*selected+metadata
    profiles['fingerprint']=profile(0,0,3*names+66,_ceil(state_bytes),30)
    profiles['inventory_configuration']=profile(0,0,tensors+32,_ceil(full+metadata),20)
    profiles['input_io']=dict(ceiling['input_io'],identity_hash_attempts_upper=64,wall_s_upper=5)
    profiles['cleanup']=profile(wall=proposal['qualification']['deferred_cleanup_lifetime_reservation']['wall_s_upper'])
    checkpoint_write=12*n+64*MIB
    profiles['checkpoint_save']=profile(8*checkpoint_write,checkpoint_write,10000,32*GIB,60)
    profiles['checkpoint_restore']=profile(8*checkpoint_write,64*MIB,10000,64*GIB,60)
    profiles['checkpoint_adapter_io']=dict(ceiling['checkpoint_adapter_io'],wall_s_upper=60)
    for kind in ('checkpoint_construct','journal_construct'):
        profiles[kind]=dict(ceiling[kind],wall_s_upper=30)
    for kind in ('loss_forward','qualification_reference_forward','qualification_candidate_forward'):
        profiles[kind]=profile(0,0,3, max_hash,10)
    for kind in ('backward','qualification_backward'): profiles[kind]=profile(wall=20)
    profiles['qualification_likelihood']=profile(0,0,3,max_hash,20)
    for kind in ('qualification_mask','tokenizer_render'):
        profiles[kind]=profile(0,0,16,4*max_hash,1)
    for kind in ('gradient_validation','gradient_clipping','qualification_gradient_validation'):
        profiles[kind]=profile(0,0,names+16,_ceil(selected+metadata),5)
    profiles['optimizer']=profile(wall=10)
    profiles['qualification_tensor_write']=profile(selected+1,selected,names+2,2*selected+metadata,60)
    profiles['qualification_artifact']=dict(profile(2*4*MIB+2,4*MIB,16,64*MIB,5),reducer_work_bytes_upper=4*MIB)
    profiles['qualification_artifact_readback']=profile(3*selected+64*MIB,0,8,3*selected+64*MIB,60)
    profiles['qualification_reducer_overhead']=dict(profile(0,0,10000,128*GIB,60),reducer_work_bytes_upper=512*GIB)
    profiles['qualification_state_fingerprint']=profile(0,0,3*names+66,8*GIB,30)
    profiles['qualification_lease']=profile(16388,4096,4,16384,5)
    profiles['donor_chunk']=dict(ceiling['donor_chunk'],wall_s_upper=1)
    for kind,wall in (('donor_reference',60),('donor_write',60),('donor_roundtrip',120)):
        profiles[kind]=dict(ceiling[kind],wall_s_upper=wall)
    profiles['resource_probe']=profile(64*MIB+1,0,1,64*MIB+1,10)
    profiles['resource_cuda_probe']=profile(wall=1)
    for value in profiles.values(): accounting._profile(value)
    return profiles,{'selected_fp32_bytes':selected,'full_fp32_source_bytes':full,'checkpoint_write_bytes':checkpoint_write,
        'source_file_bytes':source_bytes,'source_files':source_count,'max_prepared_hash_bytes':max_hash,
        'rule':'Known sizes/counts are analytical; remaining allowances are explicit fail-closed qualification ceilings, not measured prices.'}


def required_interfaces():
    return [
        {'name':'parent_qualification_factory','api':'context -> NativeQualificationExecutor(NativeCEBackend(context, opaque permit), fixed services)',
         'existing':'ce_native_backend.create_backend / ce_native_qualification.NativeQualificationExecutor','status':'composition factory not source-registered'},
        {'name':'accounting_factory','api':'context,permit,proof,prepared_receipt -> create_accounting(... actual bootstrap_handle, bootstrap_verifier)',
         'existing':'ce_qualification_accounting.create_accounting','status':'actual parent prepayment bridge required'},
        {'name':'bootstrap_constructor_prepayment','api':'request,service_spec,ControllerServicePermit -> source-owned object.require_constructor(spec_sha,request_sha)',
         'existing':'ce_service_prepayment.bootstrap_constructor_prepayment','status':'concrete permanent reservation; exact source registration remains closed'},
        {'name':'outer_issuer/outer_verifier/outer_reservation','api':'actual metadata outer scope and permanent bounded stage reservation',
         'existing':'ce_controller.OuterAdmission','status':'reviewed source adapters/registration required'},
        {'name':'worker_observation','api':'context,stable operation -> exact paid current OS WorkerObservationPermit/spec',
         'existing':'ce_worker_evidence / ce_checkpoint_entry_provider.current_observation_spec','status':'actual parent pre-model bridge required'},
        {'name':'prepared_rows','api':'common,ServicePermit,operation,gate,proof + exact request/service_spec/worker_handle/observation_permit',
         'existing':'ce_entry_factories.prepare_checkpoint_entry_rows','status':'source supplied separately; existing outer/service factory registration remains closed'},
        {'name':'prefix_terminal_handle','api':'actual retained/duplicated terminated prefix WinWorkerHandle transported to resumed child',
         'existing':'WinWorkerHandle.retain_terminal','status':'cross-process transport/adoption required; path/status cannot substitute'},
        {'name':'resource_recorder','api':'same context/permit/dispatcher NativeResourceRecorder shared across launch/runtime/final barriers',
         'existing':'ce_resource_evidence.NativeResourceRecorder / ce_resource_qualification.resource_adapter',
         'status':'barrier collection/service composition must be source-wired before resource qualification'},
        {'name':'native_journal/publication/completion_seal','api':'qualification leaf owner; actual prepayment/raw/OS/source/clock replay before final receipt',
         'existing':'ce_publication / ce_execution_verifier / ce_bootstrap_verifier','status':'registered complete parent service composition required'}]


def child_environment():
    """Explicit minimal future child environment; no credential variables copied."""
    allowed=('SYSTEMROOT','WINDIR','COMSPEC','TEMP','TMP','PATH','USERPROFILE','LOCALAPPDATA','APPDATA')
    result={key:value for key in allowed if (value:=os.environ.get(key)) is not None}
    require(all(result.get(key) for key in ('SYSTEMROOT','TEMP','PATH')), 'explicit Windows child environment metadata missing')
    result.update(PYTHONHASHSEED='20261001',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    return result


def build_documents(proposal,decision,decision_ref,proposal_ref,prior_receipt,source_preview,source,*,output_root,artifact_directory,source_head,resource_probe,worker_probe):
    require(decision_ref['sha256']==DECISION_SHA and proposal_ref['sha256']==PROPOSAL_SHA
            and decision.get('schema')=='CE8-OPERATOR-CHOICE-1' and decision.get('approved') is True
            and decision.get('choice')=='canonical_only' and decision.get('new_experiment') is True
            and decision.get('preserved_stop_receipt_sha256')==qualification.WARM_STOP,
            'exact selected delegated canonical decision and original proposal')
    require(proposal['approved'] is False and proposal['training_authorized'] is False
            and digest(proposal['static_recipe'])==proposal['static_recipe_sha256'], 'unchanged unapproved recipe basis')
    output_root,artifact_directory=Path(output_root),Path(artifact_directory)
    require(output_root.is_absolute() and artifact_directory.is_absolute()
            and not output_root.is_relative_to(Path(source['source_checkout'])), 'separate absolute future qualification output owner')
    receipt=expected_receipt(prior_receipt,source_preview['receipt'],proposal['static_recipe_sha256'])
    profiles,analytical=qualification_profiles(proposal,receipt,source)
    metadata=64*MIB; selected=analytical['selected_fp32_bytes']; model=copy.deepcopy(proposal['native_backend'])
    for key in ('native_ledger_artifact_interface','nested_operation_profiles','qualification_output_directory_template',
                'production_output_directory_template','resolved_output_directories','weight_header_metadata','asset_metadata_purpose'):
        model.pop(key,None)
    model.update(output_directory=str(output_root),source_freeze={'head':source_head,'frozen':False},
                 operation_bounds_status='PROPOSED_QUALIFICATION_UPPER_NOT_AUTHORIZED',
                 operation_bounds={kind:{key:profiles[kind][key] for key in backend.PROFILE_FIELDS} for kind in backend.OPERATION_KINDS})
    limits={'launch_host_gib':5.5,'ongoing_host_gib':1.0,'max_worker_gib':4.5,'gpu_free_gib':.5,
        'allocator_fraction':.78,'reserved_fraction_limit':.95,'disk_floor_bytes':14*GIB,'launch_disk_bytes':32*GIB,
        'active_storage_reserve_bytes':16*GIB}
    probe=copy.deepcopy(resource_probe['spec']); probe['output_root']=str(output_root); probe['cuda_probe_bounds']={'wall_s_upper':1}
    resource.validate_spec(probe)
    profiles['resource_probe']=profile(probe['nvidia_smi_ref']['bytes']+1,0,1,probe['nvidia_smi_ref']['bytes']+1,probe['probe_bounds']['wall_s_upper'])
    union={rid for item in receipt['witnesses'].values() for rid in item['row_ids']}
    witness=[row['id'] for row in receipt['prepared_row_bindings'] if row['id'] in union]
    gradient=proposal['qualification']['witnesses']['arms']['canonical']['gradient_reference_boundary_row_ids']
    batches=[gradient,[rid for rid in witness if rid not in gradient][:8]]
    require(len(witness)==90 and all(len(batch)==8 for batch in batches), 'exact witness and two independent qualification batches')
    reader={'operation_attempts_upper':20000,'io_read_bytes_upper':128*GIB,'identity_hash_attempts_upper':200000,
            'identity_hash_bytes_upper':512*GIB,'reducer_work_bytes_upper':1024*GIB}
    envelope=copy.deepcopy(proposal['qualification']['global_caps'])
    stage_caps=copy.deepcopy(proposal['qualification']['stage_caps'])
    # Native model/CP reservations are the unchanged proposal hard caps. Outer
    # bootstrap/process/publication allowances are separately proposed below.
    scope={'schema':qualification.SCOPE,'role':'qualification_only','approved':False,'qualification_authorized':False,
        'production_training_authorized':False,'evaluation_authorized':False,'arm':'canonical','choice_sha256':decision_ref['sha256'],
        'static_recipe_sha256':proposal['static_recipe_sha256'],'native_execution_stages':sorted(authority.STAGES),
        'allowed_data_roles':['train'],'likelihood_tolerance_nats':.0001,
        'gradient_reference':copy.deepcopy(proposal['qualification']['criteria']['likelihood_gradient_parity']['gradient_reference']),
        'native_backend':model,'qualifier_resources':probe,
        'qualifier_execution':{'schema':executor.SCHEMA,'all_witness_ids':witness,'gradient_batch_ids':gradient,'adam_batches':batches,
            'owned_artifact_directory':str(output_root/'raw'),'json_artifact_bytes_upper':4*MIB,
            'tensor_artifact_bytes_upper':selected,'artifact_chunk_bytes':MIB,'reader_bounds':reader},
        'qualifier_accounting':{'schema':accounting.SPEC,'owned_directory':str(output_root/'ledger'),
            'ingestion_receipt_sha256':receipt['receipt_sha256'],'operation_profiles':profiles,
            'setup_bounds':profile(64*MIB,64*MIB,16,64*MIB,30),'stage_caps':stage_caps,'global_caps':envelope,
            'physical_wall_seconds_upper':19200},
        'resource_measurement':{'schema':measurement.SCHEMA,'required_barriers':['launch','before_model','after_runtime',
            'after_update','after_checkpoint','after_delta','final','resource_io_measurement'],
            'measured_operation_kinds':['resource_probe','resource_cuda_probe'],'limits':limits,
            'owned_directory':str(output_root/'resource'),'json_bytes_upper':4*MIB,'reducer_work_bytes_upper':16*MIB},
        'checkpoint_qualification':{'schema':checkpoint.SPEC,'owned_directory':str(output_root/'checkpoints'),
            'state_proof_bounds':{key:profiles['qualification_state_fingerprint'][key] for key in backend.PROFILE_FIELDS},
            'child_initializer':{'schema':initializer.SPEC,'reader_bounds':reader,'max_store_bytes':4*selected+metadata,
                'branch_attempt_ids':{branch:'canonical-'+branch for branch in checkpoint.BRANCHES}},
            'local_issuer':{'schema':local.SPEC,'deadline_seconds':3600,'rpc_seconds':60,
                'max_message_bytes':MIB,'max_messages':8,'max_request_bytes':4*MIB,
                'operation_bounds':profile(4*MIB+1,MIB,64,256*MIB,60)}}}
    lease_profile=profile(4*4097,4096,4,4*4096,5)
    scope['checkpoint_lease']={'schema':lease.SPEC,'lock_path':str(Path(os.environ.get('TEMP',str(output_root.parent)))/'rineya-gpu-0.lock'),
        'label':'comp-rs-001-ce8-canonical-qualification','timeout_seconds':60,'terminal_verify_timeout_seconds':60,
        'lock_bytes_upper':4096,'helper_ref':{'path':proposal['resource_envelope']['gpu_lease_path'],
            'sha256':proposal['resource_envelope']['gpu_lease_sha256'],'bytes':Path(proposal['resource_envelope']['gpu_lease_path']).stat().st_size},
        'parent_released_cuda_bytes_upper':0,'child_released_cuda_bytes_upper':0,
        'operation_bounds':{'acquire':profile(metadata,4096,64,metadata,180),'transfer':lease_profile,
                            'release':profile(4*4097,4096,4,4*4096,19200)}}
    lease.validate_spec(scope['checkpoint_lease'])
    outer_bounds=costs(256*GIB,64*GIB,200000,1024*GIB,2*GIB,50000,76800)
    scope['qualifier_bootstrap']={'schema':bootstrap.SPEC,'owned_directory':str(output_root/'bootstrap'),
        'caps':outer_bounds,'self_bounds':costs(metadata,metadata,10000,metadata,metadata,10000,60),
        'physical_wall_seconds_upper':19200}
    pub_bounds=costs(256*GIB,64*MIB,200000,1024*GIB,64*MIB,20000,19200)
    pub={'schema':branch_publication.SPEC,'max_receipt_bytes':4*MIB,'bounds':pub_bounds,'reader_bounds':reader}
    scope['checkpoint_branch_publication']=pub
    scope['publication']={**pub,'schema':publication.SPEC,'owned_directory':str(output_root/'publication')}
    scope['publication_owner']={'accounting_spec_sha256':digest(scope['publication']),
                                'prepared_receipt_sha256':receipt['receipt_sha256']}
    templates={}
    observer_module='experiments.skill_acquisition_native_v1.ce_child_scope_adapters'
    observer_source=source['code_sha256'][observer_module.replace('.', '/')+'.py']
    for branch in checkpoint.BRANCHES:
        templates[branch]={'cwd':source['source_checkout'],'source_checkout':source['source_checkout'],
            'worker_module':'experiments.skill_acquisition_native_v1.ce_checkpoint_worker','worker_topology':'venv_redirector',
            'python_ref':worker_probe['spec']['python_ref'],'process_image_ref':worker_probe['spec']['process_image_ref'],
            'source_files':source['source_files'],'approved_code_sha256':source['code_sha256'],
            'observation_factory_descriptor':{'module':observer_module,'symbol':'child_launch_observation','source_sha256':observer_source},
            'environment':child_environment(),
            'deadline_seconds':3600,'handshake_seconds':180,'rpc_seconds':3600,'terminate_grace_seconds':30,
            'max_message_bytes':MIB,'max_messages':8,'max_stderr_bytes':MIB,
            'bounds':{'operation_attempts_upper':10000,'io_read_bytes_upper':4*GIB,'identity_hash_attempts_upper':10000,
                'identity_hash_bytes_upper':4*GIB,'wall_seconds_upper':3600},
            'observation_bounds':{'operation_attempts_upper':100000,'io_read_bytes_upper':4*GIB,
                'identity_hash_attempts_upper':100000,'identity_hash_bytes_upper':4*GIB,'wall_seconds_upper':3600}}
    scope['checkpoint_processes']={'schema':processes.SPEC,'owned_directory':str(output_root/'checkpoints/processes'),
        'bounds':outer_bounds,'max_json_bytes':4*MIB,'reader_bounds':reader,'launch_templates':templates,
        'prefix_transfer_bounds':profile(metadata,metadata,64,256*MIB,60)}
    scope['checkpoint_processes_owner']={'accounting_spec_sha256':digest(scope['checkpoint_processes']),
                                       'prepared_receipt_sha256':receipt['receipt_sha256']}
    service_profiles={kind:profile(metadata,metadata,64,256*MIB,60) for kind in
        ('controller_stage','source_verification','scope_admission','framework_preparation','publication','interruption',
         'input_io','tokenizer_load','tokenizer_render','entry_preparation','entry_worker_binding','worker_runtime_binding',
         'worker_source_verification','worker_os_binding','worker_handle_close','worker_handle_duplicate',
         'scope_capsule','scope_metadata','outer_reservation','terminal_publication')}
    service_profiles['supervisor_lease']=scope['checkpoint_lease']['operation_bounds']['transfer']
    service_profiles['supervisor_lease_acquire']=scope['checkpoint_lease']['operation_bounds']['acquire']
    service_profiles['supervisor_lease_release']=scope['checkpoint_lease']['operation_bounds']['release']
    service_profiles['prefix_handle_transfer']=profile(metadata,metadata,64,256*MIB,60)
    service_profiles['resource_probe']=profiles['resource_probe']; service_profiles['resource_cuda_probe']=profile(wall=1)
    service_profiles['entry_lifetime']=profile(4*GIB,metadata,10000,4*GIB,3600)
    service_profiles['entry_preparation']=profile(2*GIB,metadata,100000,4*GIB,1800)
    service={'schema':'CE8-NATIVE-CONTROLLER-SERVICES-1','approved_code_sha256':source['code_sha256'],
        'role_descriptors':{},'physical_deadline_seconds':19200,
        'supervision':{'schema':'CE8-QUALIFICATION-SUPERVISION-SPEC-1',
            'owned_directory':str(output_root/'supervision'),'max_json_bytes':4*MIB},
        'constructor_prepayment':{'schema':'CE8-SERVICE-CONSTRUCTOR-PREPAYMENT-1',
            'owned_directory':str(output_root/'service-bootstrap'),
            'source_verifier_bounds':{'hash_attempts':5,'hash_bytes':32*MIB},
            'bounds':{'read_bytes':2*metadata,'write_bytes':2*metadata,'hash_attempts':20000,
                'hash_bytes':4*metadata,'metadata_bytes':2*metadata,'operation_attempts':4,'wall_seconds':120}},
        'bootstrap':{'owned_directory':str(output_root/'service-bootstrap'),
            'setup_bounds':profile(metadata,metadata,10000,metadata,60),
            'operation_profiles':service_profiles,'caps':{'read_bytes':64*GIB,'write_bytes':4*GIB,'hash_attempts':2000000,
                'hash_bytes':256*GIB,'metadata_bytes':4*GIB,'operation_attempts':100000,'wall_seconds':256000}},
        'resource':{'limits':limits,'resource_probe':probe,
            'host_transient_bytes':{kind:64*MIB for kind in runner.HOST_STAGES | set(qualification.STAGES) | accounting.KINDS |
                {'before_framework','before_model','ingestion_sources','ingestion_tokenizer','ingestion_qualification_rows',
                 'qualification_artifact','qualification_parent_release','qualification_state_fingerprint',
                 'after_runtime','after_update','after_checkpoint','after_delta','final'}}}}
    expectation={'schema':'CE8-PREPARED-RECEIPT-EXPECTATION-1','observed_current_scope':False,
        'prior_observed_tokenizer_receipt_sha256':prior_receipt['receipt_sha256'], 'current_source_metadata_revalidated':True,
        'tokenizer_recomputed':False,'expected_receipt':receipt,
        'condition':'Actual fresh parent PreparedRows and every child must reproduce this exact receipt before model allocation.'}
    documents={'RECIPE.json':{'recipe':proposal['static_recipe']},'RUNTIME.json':{'decoding_preimage':proposal['runtime_preimage'],
        'decoding_sha256':proposal['runtime_preimage_sha256']},'ANCHORS.json':proposal['anchors'],
        'SOURCE_SNAPSHOT.json':source,'EXPECTED_PREPARED_RECEIPT.json':expectation,'SOURCE_REVALIDATION.json':source_preview,
        'INGESTION_SPEC.json':{'schema':'CE8-INGESTION-SPEC-1','arm':'canonical','plan_sha256':prior_receipt['plan_sha256'],
            'preparation_ref':prior_receipt['preparation_ref'],'tokenizer_receipt_sha256':prior_receipt['tokenizer_receipt_sha256'],
            'expected_census':qualification.declared_census(json.loads(Path(prior_receipt['preparation_ref']['path']).read_bytes()),'canonical')},
        'ATTEMPT_CAPS.json':{'schema':'CE8-QUALIFICATION-CAPS-1','stage_caps':stage_caps,'global_caps':envelope},
        'RESOURCES.json':{'schema':'CE8-QUALIFICATION-RESOURCES-1','limits':limits,'probe':probe,
            'fit_measured':False,'io_scope':'tracked Python file IO/hashes; external utility and whole-unit OS IO remain unmeasured'},
        'SERVICE_SPEC.json':service,'SCOPE.json':scope,
        'EVALUATION_SCOPE.json':{'schema':'CE8-QUALIFICATION-EVALUATION-1','evaluation_authorized':False,
            'allowed_data_roles':['train'],'greedy_heldout_outputs':0,'sealed_roles':['DEV','CALIBRATION','TEST','CHALLENGE','guard_dev','guard_test','guard_challenge']}}
    documents['REGISTRATION_DRAFT.json']={'schema':SCHEMA,'status':STATUS,'choice':'canonical_only',
        'selected_decision_ref':decision_ref,'unapproved_proposal_ref':proposal_ref,'source_head_basis':source_head,
        'source_freeze':False,'approved':False,'qualification_authorized':False,'training_authorized':False,
        'evaluation_authorized':False,'dispatch_authorized':False,'native_qualified':False,
        'qualification_weights':'discarded; never production donor or reset anchor','analytical_work':analytical,
        'profile_status':'UNMEASURED_FINITE_UPPER_RESERVATIONS_FOR_DISCARDED_QUALIFICATION_ONLY',
        'root_physical_deadline_seconds':19200,'per_worker_leaf_caps':envelope,
        'additional_outer_reservations':{'scope_bootstrap':outer_bounds,'service_bootstrap':service['bootstrap']['caps'],
            'checkpoint_processes':outer_bounds,'branch_publication_each':pub_bounds,
            'rule':'Separate retained owners; these additional ceilings require explicit review and are not part of the old57600 leaf-wall allowance.'},
        'required_source_interfaces':required_interfaces(),
        'runtime_artifact_interfaces':{'ingestion_bootstrap_ref':'actual paid service capsule retained in top-level qualifier_setup_refs before permit freeze',
            'outer_setup_reservation_ref':'actual paid constructor reservation capsule in top-level qualifier_setup_refs before Q files and permit freeze',
            'parent_child_observed_proofs':'actual paid OS identities after entry/launch, never static fakePID/hash',
            'source_launch_environment':'explicit minimal credential-free child environment; actual child must equal this frozen environment'},
        'unknowns_abort_policy':'Any profile/cap/deadline/identity/disk/host excess stops. No measured-price claim, smaller floor, fallback inventory, repeated stage or production/evaluation work.',
        'unmeasured_work':['external NVIDIA utility OS IO','whole-unit OS/driver/import IO','fresh CUDA latency/peak fit',
                           'current full source origin and terminal-handle transport'],'no_grants_or_registry_activation':True}
    documents['REGISTRATION_DRAFT.json']['profile_call_contracts']={
        'cleanup':{'max_parent_starts':1,'max_child_starts_each':1,'wall_s_upper':19200,'deferred':True},
        'checkpoint_save':{'total_starts_upper':6,'single_write_bytes_upper':12*proposal['inventory']['numel']+64*MIB},
        'checkpoint_restore':{'total_starts_upper':8,'corruption_failures_upper':3,'all_charged':True},
        'qualification_state_fingerprint':{'total_starts_upper':12,'hash_bytes_per_start_upper':8*GIB},
        'supervisor_lease':{'owner':'original service bootstrap','single_wall_s_upper':5},
        'supervisor_lease_acquire':{'max_starts':1,'single_wall_s_upper':180},
        'supervisor_lease_release':{'max_starts':1,'single_wall_s_upper':19200,'deferred':True}}
    return documents


def write_draft(documents,directory,external_refs):
    """Create-only metadata publication, then actual raw ref/request linkage."""
    directory=Path(directory); require(directory.is_absolute(), 'absolute registration draft owner')
    names=set(documents)|{'REQUEST.json','SCOPE_AUTHORIZATION.json','PACKET_INDEX.json'}
    require(not any(os.path.lexists(directory/name) for name in names), 'create-only draft already exists')
    directory.mkdir(parents=True,exist_ok=True)
    references={}
    def emit(name,value):
        raw=json.dumps(value,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False).encode()+b'\n'
        require(len(raw)<=16*MIB, 'bounded registration artifact')
        with (directory/name).open('xb') as stream:
            require(stream.write(raw)==len(raw), 'short create-only registration write'); stream.flush(); os.fsync(stream.fileno())
        references[name]={'path':str(directory/name),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
    for name,value in documents.items():
        if name=='SCOPE.json': continue
        emit(name,value)
    scope=_clone(documents['SCOPE.json']); scope['attempt_caps_sha256']=references['ATTEMPT_CAPS.json']['sha256']; scope['resources_sha256']=references['RESOURCES.json']['sha256']
    scope['expected_prepared_receipt_ref']=references['EXPECTED_PREPARED_RECEIPT.json']
    emit('SCOPE.json',scope)
    refs={'selectedoperatorchoice_ref':external_refs['decision'],'recipe_ref':references['RECIPE.json'],
        'scope_ref':references['SCOPE.json'],'source_ref':references['SOURCE_SNAPSHOT.json'],'runtime_ref':references['RUNTIME.json'],
        'anchors_ref':references['ANCHORS.json'],'evaluation_ref':references['EVALUATION_SCOPE.json'],
        'preparation_ref':{key:external_refs['preparation'][key] for key in ('path','sha256')},
        'plan_ref':{key:external_refs['plan'][key] for key in ('path','sha256')},'ingestion_ref':references['INGESTION_SPEC.json'],
        'resources_ref':references['RESOURCES.json'],'attempt_caps_ref':references['ATTEMPT_CAPS.json']}
    bounds=copy.deepcopy(scope['qualifier_bootstrap']['self_bounds'])
    request=controller.build_request('qualification','canonical',refs,bounds)
    request['scope_authorization_ref']=None
    auth={'schema':'CE8-QUALIFICATION-AUTHORIZATION-1','approved':False,'qualification_authorized':False,'training_authorized':False,
          'request_preimage_sha256':controller.digest(controller.request_preimage(request)),
          'status':'DRAFT_SOURCE_FREEZE_AND_INDEPENDENT_GATE_REVIEW_REQUIRED'}
    emit('SCOPE_AUTHORIZATION.json',auth); request['scope_authorization_ref']=references['SCOPE_AUTHORIZATION.json']
    emit('REQUEST.json',request)
    index={'schema':SCHEMA,'status':STATUS,'references':references,'approved':False,'qualification_authorized':False,
           'training_authorized':False,'evaluation_authorized':False,'dispatch_authorized':False,'native_qualified':False}
    emit('PACKET_INDEX.json',index)
    return references
