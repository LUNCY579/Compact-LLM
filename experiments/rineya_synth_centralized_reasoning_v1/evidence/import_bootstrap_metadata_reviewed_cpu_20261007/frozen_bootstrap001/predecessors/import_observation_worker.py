"""One owned import/version/effect observation; explicit user CUDA/tensors absent."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
from copy import deepcopy
from datetime import datetime,timezone
import ast,hashlib,importlib,os,time
HERE=Path(__file__).resolve().parent
CONTRACT_DESCRIPTOR={'bytes': 6205, 'sha256': 'e8e5fdde7a112567faf2beee7bbbf1524445a39de4efd2628db4ca136d6cf43e', 'role': 'python_source'}
MEASUREMENT_DESCRIPTOR={'bytes': 11124, 'sha256': '55bcdff206a8574b6246e5f2c0c047ded4cac3735e1898dc1ec85c32fd01d8d1', 'role': 'python_source'}

def selected_module(path,descriptor):
    path=Path(path).absolute();before=path.stat()
    if path.resolve()!=path or path.is_symlink() or not 0<=before.st_size<=16*1024**2:raise ValueError('bounded physical import-stage source')
    with path.open('rb') as f:raw=f.read(before.st_size+1)
    after=path.stat()
    if len(raw)!=before.st_size or (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino) or descriptor!={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':'python_source'}:raise ValueError('externally selected import-stage source descriptor')
    module=ModuleType('owned_import_'+path.stem);module.__file__=str(path);exec(compile(raw,str(path),'exec'),module.__dict__);module.__compiled_source_descriptor__=dict(descriptor);return module
C=selected_module(HERE/'import_observation_contract.py',CONTRACT_DESCRIPTOR)

def original_ledger(output):
    source=C.B.CORE/'pipeline.py';raw=C.read_bytes(source);C.require(hashlib.sha256(raw).hexdigest()==C.B.CORE_PINS['pipeline.py'],'original bounded durable writer source')
    nodes=[n for n in ast.parse(raw,str(source)).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ('counts','projected','error_value','Ledger')];C.require(len(nodes)==4,'exact original source writer definitions')
    facade=SimpleNamespace(CPU_SCOPE=C.SCOPE,LIMITS={'maximum_markers':128,'maximum_marker_bytes':65536,'maximum_ledger_bytes':8388608,'maximum_worker_json_bytes':16*1024**2},require=C.require,canonical=C.canonical)
    namespace={'C':facade,'Path':Path,'deepcopy':deepcopy,'datetime':datetime,'timezone':timezone,'time':time,'os':os};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),namespace);return namespace['Ledger'](output)

def _run_import_body(api,output,measurement_factory,identity):
    C.require(type(api) is dict and set(api)=={'source_qualification','ownership_verify','resource_guard','framework'} and all(callable(v) for v in api.values()),'exact private import-only API boundary')
    ledger=original_ledger(output);measurement=None;primary=None;cleanup=None;version=None;raw=None;report={'scope':C.SCOPE,'status':'MEASUREMENT_FAILURE_PRESERVED','native_PASS':False,'native_authority_issued':False,'model_forwards':0,'tensor_allocations':0,'user_CUDA_context_requested':False}
    try:
        ledger.call('source_qualification',api['source_qualification']);ledger.call('ownership_verify',api['ownership_verify']);ledger.call('resource_before_import',lambda:api['resource_guard']('before_import'));measurement=ledger.call('measurement_begin',measurement_factory)
        def imported():
            with measurement.framework_import():return api['framework']()
        module=ledger.call('framework_import',imported)
        version=ledger.call('torch_version_read',lambda:getattr(module,'__version__'))
        C.require(type(version) is str and 0<len(version)<=128,'bounded actual import version metadata')
        ledger.call('resource_after_import',lambda:api['resource_guard']('after_import'))
        ledger.call('measurement_observations',lambda:{'scope':measurement.provider.scope(),'libraries':measurement.provider.libraries(),'events':measurement.provider.events()})
    except BaseException as error:primary=error
    try:ledger.call('measurement_cleanup',measurement.close if measurement is not None else lambda:{'measurement_not_acquired':True})
    except BaseException as error:cleanup=error
    try:
        raw=ledger.call('raw_measurement_publication',lambda:_publish_measurement(ledger,measurement,identity))
        report.update(torch_version=version,measurement_file_sha256=raw['file_sha256'],measurement_body_sha256=raw['body_sha256'],import_only_operations=deepcopy(ledger.ops),errors=[{'type':type(e).__name__,'message':str(e)[:4096]} for e in (primary,cleanup) if e is not None])
        if primary is None and cleanup is None:
            C.require(list(ledger.ops)==list(C.OPERATIONS[:-1]) and all(row=={'attempted':1,'returned':1,'failed':0,'pending':0} for row in ledger.ops.values()),'exact import-only reached-operation order/counts');report['status']='MEASUREMENT_RECORDED_UNQUALIFIED'
        ledger.call('worker_result_publication',lambda:ledger.durable('worker_result.json',report))
        report['final_phase_summary']=ledger.finish(report);report['operation_accounting']=deepcopy(ledger.ops);report['IO_accounting']=deepcopy(ledger.io)
    except BaseException as publication:
        if primary is not None:raise primary from publication
        if cleanup is not None:raise cleanup from publication
        raise
    if primary is not None:
        if cleanup is not None:raise primary from cleanup
        raise primary
    if cleanup is not None:raise cleanup
    return report

def _publish_measurement(ledger,measurement,identity):
    if measurement is None:raise ValueError('measurement construction failed before raw packet acquisition')
    packet=measurement.export_raw_packet(identity);raw=(C.canonical(packet)+'\n').encode('ascii');ledger.durable('IMPORT_LIBRARY_MEASUREMENT.json',packet);return {'body_sha256':packet['sha256'],'file_sha256':hashlib.sha256(raw).hexdigest()}

def _CPU_run_import_fixture(context,api,output,measurement_factory,identity,*,_capability):
    C._check_CPU(_capability);value=C.context_value(context);C.require(value['scope']==C.SCOPE and value.get('fixture_only') is True,'independent CPU import context only');return _run_import_body(api,output,measurement_factory,identity)

def _native_framework():return importlib.import_module('torch')
def main():
    C.stage_native_gate()
    raise ValueError('closed import-only dispatcher requires separately registered owner/context/Job bindings')
if __name__=='__main__':raise SystemExit(main())
