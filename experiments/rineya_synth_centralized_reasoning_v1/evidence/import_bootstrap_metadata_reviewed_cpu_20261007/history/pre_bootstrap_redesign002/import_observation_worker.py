"""One owned import/version/effect observation; explicit user CUDA/tensors absent."""
from pathlib import Path
from types import ModuleType,SimpleNamespace
from copy import deepcopy
from datetime import datetime,timezone
import ast,hashlib,importlib,os,time
HERE=Path(__file__).resolve().parent
CONTRACT_DESCRIPTOR={'bytes': 13302, 'sha256': '1bda856263df1f3fc6105788559c2a69bdb2a9c916aab7375ac57cca4f216fc0', 'role': 'python_source'}
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

def _run_import_body(api,output,measurement_factory,identity,*,owned_cleanup=None):
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
    partial_raw=getattr(primary,'__measurement_partial_raw__',None) if primary is not None else None
    try:ledger.call('measurement_cleanup',owned_cleanup if owned_cleanup is not None else measurement.close if measurement is not None else lambda:{'measurement_not_acquired':True})
    except BaseException as error:cleanup=error
    try:
        raw=ledger.call('raw_measurement_publication',lambda:_publish_measurement(ledger,measurement,identity,partial_raw=partial_raw))
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

def _publish_measurement(ledger,measurement,identity,*,partial_raw=None):
    if measurement is None and partial_raw is None:raise ValueError('measurement construction failed before raw packet acquisition')
    packet=partial_raw if measurement is None else measurement.export_raw_packet(identity);raw=(C.canonical(packet)+'\n').encode('ascii');ledger.durable('IMPORT_LIBRARY_MEASUREMENT.json',packet);return {'body_sha256':packet['sha256'],'file_sha256':hashlib.sha256(raw).hexdigest()}

def _CPU_run_import_fixture(context,api,output,measurement_factory,identity,*,_capability):
    C._check_CPU(_capability);value=C.context_value(context);C.require(value['scope']==C.SCOPE and value.get('fixture_only') is True,'independent CPU import context only');return _run_import_body(api,output,measurement_factory,identity)

def _native_framework():return importlib.import_module('torch')

def _selected_shared_worker(contract,value):
    path=HERE/'predecessors/prefix_input_owned_worker.py';descriptor=value['config']['source_files'][str(path)];contract.file_descriptor(path,descriptor);raw=contract.read_bytes(path);tree=ast.parse(raw,str(path));removed=0;kept=[]
    for node in tree.body:
        if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='C':removed+=1
        elif isinstance(node,ast.If) and isinstance(node.test,ast.Compare) and isinstance(node.test.left,ast.Name) and node.test.left.id=='__name__':removed+=1
        else:kept.append(node)
    contract.require(removed==2,'exact shared worker C-init and final dispatch source adaptations');module=ModuleType('import_only_shared_worker_functions');module.__file__=str(path);module.__dict__['C']=contract;exec(compile(ast.Module(body=kept,type_ignores=[]),str(path),'exec'),module.__dict__);return module

def _measurement_loader(value,measurement,module):
    path=HERE/'predecessors/prefix_registered_loader.py';descriptor=value['config']['source_files'][str(path)];C.file_descriptor(path,descriptor);raw=C.read_bytes(path);tree=ast.parse(raw,str(path));changes=0
    def selected_guard(actual,contract,*args,**kwargs):
        C.require(actual==value and contract is C and type(measurement) is module.MeasurementGuard and type(measurement).__module__==module.__name__,'source-owned exact import measurement object/context')
        C.require(module.__compiled_source_descriptor__==value['config']['source_files'][str(HERE/'measurement_guard.py')],'externally selected measurement source descriptor')
        C.require(type(measurement).framework_import.__globals__ is module.__dict__ and measurement.active and not measurement.closed and measurement.native_authority_issued is False,'original selected measurement methods/state; no permission callback')
        return measurement
    for node in tree.body:
        if isinstance(node,ast.FunctionDef) and node.name=='_selected_library_guard':
            node.body=ast.parse('return __import_measurement_selector__(value,contract,budget,fixture_only=fixture_only,library_supplier=library_supplier,library_bootstrap_verifier=library_bootstrap_verifier,library_event_observer=library_event_observer)').body;changes+=1
    C.require(changes==1,'exact one import-only source loader selector adaptation');ast.fix_missing_locations(tree);loader=ModuleType('selected_import_measurement_loader');loader.__file__=str(path);loader.__dict__['__import_measurement_selector__']=selected_guard;exec(compile(tree,str(path),'exec'),loader.__dict__);loader.__compiled_source_descriptor__=dict(descriptor);return loader

class _OwnedMeasurement:
    def __init__(self,state,measurement):self.state=state;self.actual=measurement;self.provider=measurement.provider
    def framework_import(self):return self.actual.framework_import()
    def close(self):
        primary=None;result=None
        try:
            if self.state.get('finder') is not None:
                self.state['loader'].remove(self.state['finder']);result=self.state['finder'].library_cleanup;self.state['finder']=None
            elif not self.actual.closed:result=self.actual.close()
        except BaseException as failure:primary=failure
        try:
            if self.state.get('transition') is not None:self.state['utilities']._restore_registered_roots(self.state['transition']);self.state['transition']=None
        except BaseException as cleanup:
            if primary is not None:raise primary from cleanup
            raise
        if primary is not None:raise primary
        return result
    def export_raw_packet(self,identity):return self.actual.export_raw_packet(identity)


def _ownership_source_slice(value):
    path=C.PACKAGE/'native_worker.py';descriptor=value['config']['source_files'][str(path)];C.file_descriptor(path,descriptor);raw=C.read_bytes(path);names={'_job_query_error','_job_query_api','owned_job_pids','owned_job_resources','in_owned_job','verify_worker_ownership','read_heartbeat'}
    nodes=[node for node in ast.parse(raw,str(path)).body if isinstance(node,ast.FunctionDef) and node.name in names];C.require({node.name for node in nodes}==names,'exact seven pinned ownership/resource/heartbeat source definitions')
    import sys,re,json
    C.require('ctypes' in sys.modules,'selected byte-verified preloaded ctypes before original Job helper calls')
    namespace={'__file__':str(path),'__name__':'import_owned_helper_source','ctypes':sys.modules['ctypes'],'os':os,'re':re,'json':json,'Path':Path,'MAX_OWNED_JOB_PIDS':256,'require':C.require,'contract':SimpleNamespace(filehash=lambda p:hashlib.sha256(C.read_bytes(p)).hexdigest(),_pairs=C.B.pairs,_bad_constant=C.B.constant)}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace);return SimpleNamespace(**{name:namespace[name] for name in names})

def _cleanup_acquired_state(state):
    measurement=state.get('measurement');primary=None;result=None
    try:
        if measurement is not None:result=_OwnedMeasurement(state,measurement).close()
        elif state.get('finder') is not None:state['loader'].remove(state['finder']);state['finder']=None
    except BaseException as error:primary=error
    try:
        if state.get('transition') is not None:state['utilities']._restore_registered_roots(state['transition']);state['transition']=None
    except BaseException as cleanup:
        if primary is not None:raise primary from cleanup
        raise
    if primary is not None:raise primary
    return result if measurement is not None else {'measurement_not_acquired':True,'owned_roots_restored':True}


def _native_factory(context,owner,output):
    C.stage_native_gate();value=C.context_value(context);C.require(value['scope']==C.SCOPE and value['fixture_only'] is False,'actual separate import observation context only')
    utilities=_selected_shared_worker(C,value);state={'utilities':utilities,'context':context,'owner':owner,'output':Path(output),'measurement':None,'finder':None}
    def source():
        C.revalidate_context(context);utilities._site_disabled_bootstrap_setup(context);state['transition']=utilities._registered_search_roots(context);state['resource_module']=selected_module(HERE/'import_resource_guard.py',value['config']['source_files'][str(HERE/'import_resource_guard.py')])
        return {'scope':C.SCOPE,'source_qualified':True,'native_authority_issued':False}
    def ownership():
        state['native_worker']=_ownership_source_slice(value);return state['native_worker'].verify_worker_ownership(owner,value['config_path'])
    def resource(stage):
        native=state['native_worker'];ram_path=C.PACKAGE/'worker_ram_probe.py';ram=selected_module(ram_path,value['config']['source_files'][str(ram_path)])
        if 'resource_guard' not in state:state['resource_guard']=state['resource_module'].bind(output,owner,time.monotonic(),owned_resources=native.owned_job_resources,heartbeat_reader=native.read_heartbeat,available=ram.available_bytes)
        return state['resource_guard'](stage)
    def measurement_factory():
        G=selected_module(HERE/'measurement_guard.py',value['config']['source_files'][str(HERE/'measurement_guard.py')]);adapter_path=C.PACKAGE/'prefix_library_observer.py';adapter=selected_module(adapter_path,value['config']['source_files'][str(adapter_path)]);observer=adapter.make_native_observer(context,C,owner);C._register_library_event_observer(context,adapter,observer,observer.abi_binding());state['observer']=observer
        provider=G._O.ManagedWin32LibraryProvider(C,observer.bootstrap,observer);state['identity']={'source_commit':value['expected_head'],'original_scope':C.SCOPE,'owned_birth':observer.birth,'owned_birth_status':'VALIDATED','context_sha256':C.digest(value),'owner_receipt_sha256':C.digest(owner),'producer_source':{'path':str(HERE/'measurement_guard.py'),'descriptor':value['config']['source_files'][str(HERE/'measurement_guard.py')]},'fixture_only':False,'native_PASS':False,'native_authority_issued':False};policy=value['runtime']['measurement_policy'];grammar=value['runtime']['measurement_grammar'];measurement=object.__new__(G.MeasurementGuard);state['measurement']=measurement
        try:measurement.__init__(policy,grammar,C,provider)
        except BaseException as primary:
            try:primary.__measurement_partial_raw__=measurement.export_raw_packet(state.get('identity'))
            except BaseException:pass
            raise
        projection=C.loader_context_value(context);loader=_measurement_loader(projection,measurement,G)
        finder=loader._install_observed_source_body(projection,C,fixture_only=False,resolver=loader.physical,library_bootstrap_verifier=observer.bootstrap,library_event_observer=observer);state.update(loader=loader,finder=finder)
        state['identity']={'source_commit':value['expected_head'],'original_scope':C.SCOPE,'owned_birth':observer.birth,'owned_birth_status':'VALIDATED','context_sha256':C.digest(value),'owner_receipt_sha256':C.digest(owner),'producer_source':{'path':str(HERE/'measurement_guard.py'),'descriptor':value['config']['source_files'][str(HERE/'measurement_guard.py')]},'fixture_only':False,'native_PASS':False,'native_authority_issued':False}
        return _OwnedMeasurement(state,measurement)
    api={'source_qualification':source,'ownership_verify':ownership,'resource_guard':resource,'framework':_native_framework}
    return api,measurement_factory,state

def _run_native_owned(context,owner,output):
    api,factory,state=_native_factory(context,owner,output);identity={}
    def acquire():
        try:measurement=factory()
        finally:identity.update(state.get('identity',{}))
        return measurement
    return _run_import_body(api,output,acquire,identity,owned_cleanup=lambda:_cleanup_acquired_state(state))

def main():
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    C.enroll_native_gate_from_owned_output(args.output_dir);external=C.parse(C.read_bytes(args.output_dir/'OWNED_EXTERNAL_ANCHORS.json'));context=C.validate_native_manifest(args.config,**external['anchors']);owner=C.parse(C.read_bytes(args.output_dir/'supervisor_ownership.json'));report=_run_native_owned(context,owner,args.output_dir);return 0 if report['status']=='MEASUREMENT_RECORDED_UNQUALIFIED' else 2
if __name__=='__main__':raise SystemExit(main())
