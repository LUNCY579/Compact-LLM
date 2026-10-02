"""Fixed production CE8 worker entry; source adoption/dispatch default CLOSED.

Only reviewed source configuration and its literal completed qualification pin
can install the existing concrete implementations. RPC carries no authority.
Executed body output remains separate from OS-terminal/cost and learning proof.
"""
from __future__ import annotations

import argparse
import copy
import importlib
import os
from pathlib import Path
import sys
from types import MappingProxyType

from . import ce_native_services as services, ce_controller as controller, ce_native_authority as authority
from . import ce_training_runtime as runtime, ce_scope_adapters as scope, ce_service_prepayment as prepayment
from . import ce_qualified_receipt as completed, ce_worker_evidence as worker, ce_checkpoint_local_issuer as rpc
from . import ce_qualification_artifacts as raw

SCHEMA='CE8-TRAINING-SOURCE-CONFIGURATION-1'
ENTRY_MODULE=__package__+'.ce_training_entry'
_SOURCE_CONFIGURATION=_INTAKE_EVIDENCE=_INSTALLED_SHA256=None
require,canonical,digest=raw.require,raw.canonical,raw.digest


def role_functions():
    return {'bootstrap_outer':runtime.bootstrap_outer,'outer_issuer':runtime.outer_issuer,'outer_verifier':runtime.outer_verifier,
        'outer_reservation':runtime.outer_reservation,'production_admission':runtime.admit_context,
        'worker_observation':scope.worker_observation,'bootstrap_constructor_prepayment':prepayment.bootstrap_constructor_prepayment,
        'native_journal':runtime.native_journal,'publication':runtime.publication_role,
        'interruption':runtime.interruption_role,'completion_seal':runtime.completion_seal_role}


def validate_configuration(value):
    require(type(value) is dict and set(value)=={'schema','request','service_spec','request_ref','service_spec_ref',
        'rpc_spec','qualification_selection','qualification_selection_sha256','configuration_sha256'} and value['schema']==SCHEMA,
        'exact fixed production source configuration')
    request,spec=value['request'],value['service_spec'];controller.validate_request(request)
    require(request['phase']=='training' and request['arm']=='canonical' and request['training_authorized'] is True
            and request['qualification_authorized'] is False and spec.get('schema')==services.SCHEMA,'production grant required; no qualification/evaluation selector')
    require(value['configuration_sha256']==digest({key:item for key,item in value.items() if key!='configuration_sha256'}),
            'production source configuration digest drift')
    for key in ('request_ref','service_spec_ref'):
        ref=value[key];require(type(ref) is dict and set(ref)=={'path','sha256','bytes'} and Path(ref['path']).is_absolute()
            and raw._hex(ref['sha256']) and raw._int(ref['bytes'],1) and ref['bytes']<=raw.MAX_JSON,'exact bounded production '+key)
    require(type(value['qualification_selection']) is dict and raw._hex(value['qualification_selection_sha256'])
            and digest(value['qualification_selection'])==value['qualification_selection_sha256'],
            'immutable full completed qualification selection pin')
    completed._shape(value['qualification_selection'])
    require(request['qualifiedreceipt_ref']==value['qualification_selection']['references']['receipt'],
            'production request differs from source-selected completed receipt')
    policy=spec['scope_adapters']['worker']
    require(policy['entry_module']==ENTRY_MODULE and policy['request_ref']==value['request_ref']
            and policy['approved_code_sha256']==spec['approved_code_sha256'],'fixed production entry/raw request/full source')
    rpc.validate_spec(value['rpc_spec'])
    canonical(value)
    return value


def _configuration():
    global _SOURCE_CONFIGURATION,_INTAKE_EVIDENCE
    if _SOURCE_CONFIGURATION is None:
        from . import ce_training_configuration as intake
        value,evidence=intake.load_source_configuration()
        _SOURCE_CONFIGURATION=MappingProxyType(value);_INTAKE_EVIDENCE=evidence
    require(type(_SOURCE_CONFIGURATION) is MappingProxyType,'fixed production source configuration remains closed')
    return validate_configuration(copy.deepcopy(dict(_SOURCE_CONFIGURATION)))


def service_metadata_verifier(issuer,request_sha256,spec_sha256):
    require(issuer is sys.modules[__name__] and __name__==ENTRY_MODULE,'canonical fixed production metadata issuer')
    config=_configuration()
    require(request_sha256==digest(controller.request_preimage(config['request']))
            and spec_sha256==digest(config['service_spec']),'exact fixed production metadata request/spec')
    return True


def install_source_registries():
    global _INSTALLED_SHA256
    config=_configuration()
    require(_INSTALLED_SHA256 in (None,config['configuration_sha256']),'production source cannot switch packet')
    if _INSTALLED_SHA256 is not None:return config
    code=config['service_spec']['approved_code_sha256'];functions=role_functions();descriptors=config['service_spec']['role_descriptors']
    require(set(descriptors)==set(functions),'exact finite production roles')
    for name,fn in functions.items():
        hashed=code.get(fn.__module__.replace('.','/')+'.py')
        require(raw._hex(hashed) and descriptors[name]=={'module':fn.__module__,'symbol':fn.__name__,'source_sha256':hashed},
                'fixed reviewed production role descriptor: '+name)
    selected=config['qualification_selection'];pin=config['qualification_selection_sha256']
    require(completed._SOURCE_SELECTION is None or dict(completed._SOURCE_SELECTION)==selected,
            'completed qualification source selection cannot switch')
    require(completed._SOURCE_SELECTION_SHA256 in (None,pin),'completed qualification literal pin cannot switch')
    # All fixed dependencies are checked before registry/source mutation.
    completed._SOURCE_SELECTION=MappingProxyType(copy.deepcopy(selected));completed._SOURCE_SELECTION_SHA256=pin
    services._REGISTERED_SERVICE_VERIFIERS=frozenset((service_metadata_verifier,))
    services._REGISTERED_ROLE_FACTORIES=frozenset(functions.values())
    services._REGISTERED_BOOTSTRAP_PREPAYMENT_TYPES=frozenset((prepayment.ServiceConstructorPrepayment,))
    controller._REGISTERED_OUTER_AUTHORITY_VERIFIERS=frozenset((runtime.outer_verifier,))
    controller._REGISTERED_NATIVE_FACTORIES=frozenset((runtime.create_training_backend,))
    authority._REGISTERED_NATIVE_AUTHORITY_VERIFIERS=frozenset((runtime.native_authority_verifier,))
    worker._REGISTERED_METADATA_VERIFIERS=frozenset((scope.worker_metadata_verifier,))
    _INSTALLED_SHA256=config['configuration_sha256']
    return config


def serve_training(pipe,proof,train):
    pipe.emit('ready',0,{'worker_proof':proof})
    require(pipe.receive(1)=={'command':'train','data':{}},'one fixed train command, no scope/authority payload')
    result=train();seal=result.get('seal') if type(result) is dict else None
    require(type(seal) is dict and seal.get('status')=='UNQUALIFIED_TRAINING_BODY_SEAL'
            and type(seal.get('body_ref')) is dict,'actual retained production body before RPC response')
    ref=seal['body_ref']
    require(Path(ref['path']).is_absolute() and raw._hex(ref['sha256']) and raw._int(ref['bytes'],1)
            and ref['bytes']<=raw.MAX_JSON,'bounded actual body reference')
    pipe.emit('response',1,{'raw_artifact':{key:ref[key] for key in ('path','sha256','bytes')},
        'body_completed':True,'os_terminal_verified':False,'native_qualified':False})
    require(pipe.receive(2)=={'command':'finish','data':{}},'finish after production body only')
    pipe.emit('response',2,{'finished':True,'native_qualified':False})
    return 0


def main(argv=None):
    config=install_source_registries()
    parser=argparse.ArgumentParser();parser.add_argument('--request',required=True);args=parser.parse_args(argv)
    require(Path(args.request)==Path(config['request_ref']['path']),'argv cannot select another production request')
    request,spec=config['request'],config['service_spec']
    permit=services.ControllerServicePermit(request,spec,sys.modules[__name__],service_metadata_verifier)
    service=services.NativeControllerServices(request,permit,spec);service.metadata_intake_evidence=copy.deepcopy(_INTAKE_EVIDENCE)
    primary=None
    try:
        profile=services._profile(spec['bootstrap']['operation_profiles']['scope_metadata'])
        for ref,expected in ((config['request_ref'],request),(config['service_spec_ref'],spec)):
            require(rpc.read_json(ref,Path(ref['path']).parent,service.bootstrap.operation,profile,raw.MAX_JSON)==expected,
                    'raw fixed production request/service bytes changed')
        context=service.admit_request(request);proof=service.verify_worker(context)
        binding=raw._json(os.environ['CE8_RPC_BINDING'])
        require(binding['request_sha256']==context['request_sha256'] and binding['scope_sha256']==context['scope_sha256'],
                'RPC binding cannot replace production scope')
        pipe=rpc.BoundedRPC(config['rpc_spec'],binding,os.getpid(),sys.stdin.buffer,sys.stdout.buffer,service.bootstrap.operation)
        return serve_training(pipe,proof,lambda:controller.run_controller(request,service,runtime.create_training_backend))
    except BaseException as error:primary=error;raise
    finally:
        if service.worker_handle is not None:
            try:service.worker_handle.close()
            except BaseException as closing:
                if primary is not None:raw._accounting(primary,closing);raise primary from closing
                raise
        if primary is None:service.bootstrap.end();service.bootstrap.inspect_evidence()


if __name__=='__main__':
    raise SystemExit(importlib.import_module(__package__+'.ce_training_entry').main())
