"""Actual original SUP body -> worker -> preservation -> independent audit.

Every host/Job/resource/death and observer supplier is explicitly inert CPU.
Only selected fixture bytes are read; no real native helper/API is imported.
"""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
from unittest.mock import patch
import ast,builtins,hashlib,sys,tempfile,unittest
import launch_driver as D

PACKAGE=D.PACKAGE
PACKET=PACKAGE/'evidence/observer_startup_measurement_reviewed_cpu_20261006'
TASK_TEMP=Path(r'C:\Users\clclc\BusinessProject\observer-launch-driver-CPU-20261006')

def source(path,name):
    m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
A=source(PACKET/'audit.py','driver_CPU_A')
saved=sys.modules.get('audit');sys.modules['audit']=A
try:F=source(PACKET/'test_audit.py','driver_CPU_fixture')
finally:
    if saved is None:sys.modules.pop('audit',None)
    else:sys.modules['audit']=saved
S=source(PACKET/'supervisor.py','driver_CPU_S')
P=source(PACKET/'preserve.py','driver_CPU_P')
saved=sys.modules.get('supervisor');sys.modules['supervisor']=S
try:SF=source(PACKET/'test_supervisor.py','driver_CPU_SUPfixture')
finally:
    if saved is None:sys.modules.pop('supervisor',None)
    else:sys.modules['supervisor']=saved

class DriverControls(unittest.TestCase):
    def setUp(self):
        TASK_TEMP.mkdir(exist_ok=True);real=tempfile.TemporaryDirectory
        def temporary(*args,**kwargs):kwargs['dir']=TASK_TEMP;return real(*args,**kwargs)
        self.case=F.ObserverAuditControls(methodName='runTest')
        with patch.object(F.tempfile,'TemporaryDirectory',side_effect=temporary):self.case.setUp()
        self.addCleanup(self.case.doCleanups);self.c=self.case.C;self.value=deepcopy(self.case.value)
        self.value['config']['kind']='REGISTERED_OBSERVER_STARTUP_MEASUREMENT_V1'
        self.value['config']=self.c.seal(self.value['config']);self.value['registration']['config_body_sha256']=self.value['config']['sha256']
        self.value['registration']=self.c.seal(self.value['registration'])
        self.value['anchors'].update(expected_config_body=self.value['config']['sha256'],expected_registration_body=self.value['registration']['sha256'])
        self.context=self.c._CPU_context_value_fixture(self.value,_capability=self.c._CPU_capability())
        self.root_patch=patch.object(self.c,'PACKAGE',self.case.package);self.root_patch.start();self.addCleanup(self.root_patch.stop)
        self.deaths=[]
        def death(birth):self.deaths.append(birth);return False
        self.death=death;self.receipt=self.case.root/'driver.json'

    def host(self):
        case=self.case;c=self.c;context=self.context;outer=self
        class Host(SF.Host):
            def __init__(self):
                super().__init__(case.root);self.identity={**case.birth,'job_name':'Local\\SYNTH_NQ_JOB_'+'1'*32,'assigned_before_resume':True,'job_kill_on_close':True,'breakaway_allowed':False};self.executed=False;self.first=None;self.last=None
            def write_new(self,path,value):
                super().write_new(path,value);path.parent.mkdir(parents=True,exist_ok=True)
                with path.open('xb') as stream:stream.write((c.canonical(value)+'\n').encode('ascii'))
            def acquire_lease(self,output,pin):
                self.calls.append('lease');lease=output.parent/'_native_gpu_leases/CPU_owned';lease.mkdir(parents=True)
                self.write_new(lease/'reservation.json',{'config_sha256':pin,'supervisor':{'pid':12344,'create_time':1000.}});return lease
            def new_output(self,output):return Path(output)
            def worker(self,argv,output,**kwargs):
                worker=super().worker(argv,output,**kwargs)
                owner={'worker_pid':case.birth['pid'],'root_worker_pid':case.birth['pid'],'supervisor_pid':12344,'supervisor_create_time':1000.,'job_name':self.identity['job_name'],'config_sha256':'c'*64}
                self.write_new(output/'supervisor_ownership.json',owner)
                for n in ('worker.stdout.log','worker.stderr.log'):(output/n).write_bytes(b'')
                (output/'supervisor_heartbeat.jsonl').write_bytes(b'{}\n')
                def poll():
                    if not self.executed:
                        self.executed=True
                        def clock():self.now+=.001;return self.now
                        case.result=case.W._CPU_run_fixture(context,c,case.transport,case.provider,output,_capability=c._CPU_capability(),clock=clock)
                        rows=[c.parse(line) for line in (output/'OBSERVER_PHASES.jsonl').read_bytes().splitlines()]
                        self.first=rows[0]['monotonic'];self.last=rows[-1]['monotonic']
                        self.returncode=0 if case.result['status']=='OBSERVER_OPERATIONS_RETURNED_PENDING_TERMINAL_AUDIT' else 2
                    return self.returncode
                worker.poll=poll;return worker
            def monitor(self,worker,output,deadline,observation):
                monitor=super().monitor(worker,output,deadline,observation)
                def close():
                    self.calls.append('samplers_joined');self.joined=True
                    rows=[]
                    for t in (self.first or self.now,self.last or self.now):
                        rows.extend([{'kind':'CPU','sample_started_monotonic':t,'host_available_bytes':8*1024**3,'worker_tree_rss_bytes':512*1024**2,'owned_process_identities':[case.birth]}, {'kind':'GPU','gpu_sample_started_monotonic':t,'gpu_free_bytes':8*1024**3,'gpu_total_bytes':8*1024**3,'gpu_uuid':'CPU_INERT_GPU'}])
                    return rows
                monitor.close=close;return monitor
            def inventory(self,output):return [{'path':p.name,'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in sorted(output.iterdir())]
        return Host()

    def run_driver(self,host=None,supervise=None):
        self.current_host=self.host() if host is None else host
        if supervise is None:
            def supervise():return S._CPU_supervise_fixture(self.context,self.current_host,self.case.raw,contract=self.c,_capability=S._CPU_capability())
        return D._CPU_run_fixture(self.context,self.c,self.case.raw,self.case.preserved,self.receipt,_capability=self.c._CPU_capability(),supervise=supervise,preservation_module=P,audit_module=A,death=self.death)

    def test_actual_original_SUP_worker_preserve_audit_with_same_original_context(self):
        result=self.run_driver();self.assertEqual(result['status'],'OBSERVER_DRIVER_RETURNED_PENDING_CURRENT_QUALIFICATION')
        self.assertTrue(result['terminal_audit']['observer_mechanics_recorded']);self.assertFalse(result['native_PASS']);self.assertTrue(result['fixture_only'])
        self.assertNotIn('owned_process_identities',self.c.parse((self.case.raw/'SUPERVISOR_RECEIPT.json').read_bytes()))
        self.assertTrue(self.current_host.closed);self.assertTrue(self.current_host.joined);self.assertTrue(self.current_host.nvml_closed)
        self.assertLess(self.current_host.calls.index('Job_assigned'),self.current_host.calls.index('resume'))
        self.assertEqual((self.case.raw/'SUPERVISOR_RECEIPT.json').read_bytes(),(self.case.raw.parent/'_native_gpu_leases/CPU_owned/terminal.json').read_bytes())
        self.assertTrue(self.deaths);self.assertTrue(all(b==self.case.birth for b in self.deaths))
        self.assertEqual(result['source_adaptations'],{'preservation_path_changes':2,'preservation_birth_projection_changes':1})
    def test_failed_worker_is_preserved_but_actual_terminal_audit_refuses(self):
        self.case.transport.rows=[{'invalid':'inert poison'}]
        with self.assertRaises(ValueError):self.run_driver()
        record=self.c.parse(self.receipt.read_bytes());self.assertEqual(record['status'],'OBSERVER_DRIVER_FAILED')
        self.assertEqual(record['supervisor']['receipt']['status'],'SUPERVISED_WORKER_FAILED')
        self.assertIsNotNone(record['preservation']);self.assertIsNone(record['terminal_audit']);self.assertFalse(record['native_PASS'])
    def test_primary_supervisor_error_identity_survives_preservation_refusal(self):
        primary=RuntimeError('inert original supervisor fault')
        def supervise():raise primary
        with self.assertRaises(RuntimeError) as caught:self.run_driver(supervise=supervise)
        self.assertIs(caught.exception,primary);self.assertIsNotNone(primary.__cause__)
        record=self.c.parse(self.receipt.read_bytes());self.assertEqual(record['errors'][0]['stage'],'supervisor')
        self.assertEqual(record['incomplete_preservation']['owned_birth_status'],'UNAVAILABLE');self.assertFalse(record['native_authority_issued'])
    def test_launch_stage_refuses_before_helper_or_parser_imports(self):
        with patch.object(D,'SelectedNamespaces',side_effect=AssertionError('native helper construction forbidden')):
            with self.assertRaisesRegex(ValueError,'no native authority'):D.main()
    def test_selected_inert_helper_origin_ignores_shadow_and_refuses_byte_drift(self):
        value=deepcopy(self.value);sources=value['config']['source_files'];files=value['runtime']['files']
        def selected(path,text,role='python_source'):
            path.parent.mkdir(parents=True,exist_ok=True);raw=text.encode('ascii');path.write_bytes(raw)
            return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
        for name in D.HELPERS:
            path=self.case.package/(name+'.py');sources[str(path)]=selected(path,"origin='selected inert source'\n")
        for family in ('psutil','ctypes'):
            path=self.case.package/'CPU_runtime'/family/'__init__.py';files[str(path)]=selected(path,"origin='inert package'\n")
        extension=self.case.package/'CPU_runtime/_ctypes.pyd';files[str(extension)]=selected(extension,'INERT NEVER LOADED','extension_binary')
        value['runtime']['cache_policy']={'allowed_bootstrap_modules':{'ctypes':{'path':str(self.case.package/'CPU_runtime/ctypes/__init__.py')},'_ctypes':{'path':str(extension)}}}
        namespace=D.SelectedNamespaces(self.c,value);shadow=self.case.root/'shadow';shadow.mkdir()
        (shadow/'native_process.py').write_text("raise AssertionError('generic shadow must never execute')\n")
        original=sys.modules.get('native_process')
        try:
            with patch.object(sys,'path',[str(shadow),*sys.path]):module=namespace.load('native_process')
            self.assertEqual(module.origin,'selected inert source');self.assertEqual(module.__file__,str(self.case.package/'native_process.py'))
            with self.assertRaisesRegex(ValueError,'cannot fall back'):namespace.find_spec('psutil._unselected')
            (self.case.package/'native_nvml.py').write_bytes(b'changed inert selected source')
            with self.assertRaises(ValueError):namespace.load('native_nvml')
        finally:
            if original is None:sys.modules.pop('native_process',None)
            else:sys.modules['native_process']=original
            namespace.close()
    def test_counted_output_adapters_preserve_original_PACKAGE_and_mutex_identity(self):
        raw=(PACKAGE/'native_supervisor.py').read_bytes();ns={'Path':Path,'PACKAGE':PACKAGE,'__file__':str(PACKAGE/'native_supervisor.py')}
        fn,count=D._external_function(raw,'_new_output',ns,self.case.root)
        target=self.case.root/'fresh_external';self.assertEqual(fn(target),target)
        self.assertEqual(count,1);self.assertEqual(fn.__globals__['PACKAGE'],PACKAGE)
        _,proof=D.supervisor_adapter(S,self.case.root)
        self.assertEqual(proof,{'supervisor_namespace_insertions':1,'new_output_path_changes':1,'lease_path_changes':1,'original_lifecycle_changes':3,'mutex_identity_changed':False})
        tree=ast.parse(raw.decode('utf-8'));mutex=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='_LeaseMutex')
        self.assertIn('str(PACKAGE)',ast.unparse(mutex));self.assertNotIn('_driver_root',ast.unparse(mutex))
    def test_birth_projection_preserves_executing_child_and_refuses_PID_birth_conflict(self):
        child={'pid':12346,'creation_filetime_100ns':134000000000000002}
        path=self.case.raw/'RESOURCE_OBSERVATIONS.json';self.case.write(path,[{'kind':'CPU','owned_process_identities':[child]}])
        self.assertEqual(D.observed_births(self.c,{'worker_identity':self.case.birth},self.case.raw),[self.case.birth,child])
        self.case.write(path,[{'kind':'CPU','owned_process_identities':[dict(self.case.birth,creation_filetime_100ns=1)]}])
        with self.assertRaisesRegex(ValueError,'conflicting observed birth'):D.observed_births(self.c,{'worker_identity':self.case.birth},self.case.raw)

if __name__=='__main__':unittest.main()
