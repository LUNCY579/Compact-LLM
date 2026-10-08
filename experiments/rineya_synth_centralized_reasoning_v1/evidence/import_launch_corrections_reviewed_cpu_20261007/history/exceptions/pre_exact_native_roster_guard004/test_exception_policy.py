"""Real two-commit Git joins with exact future outputs, no native work."""
from pathlib import Path
from types import ModuleType
from copy import deepcopy
import hashlib,json,subprocess,tempfile,unittest,sys
HERE=Path(__file__).resolve().parent
def module(p,name):m=ModuleType(name);m.__file__=str(p);exec(compile(p.read_bytes(),str(p),'exec'),m.__dict__);return m
J=module(HERE/'registration_git_join.py','CPU_future_leaf_git');Q=module(HERE/'import_runtime_qualification.py','CPU_native_source_exception_Q')
class Controls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='CPU_future_exception_');self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name).resolve();self.package=self.root/'experiments/rineya_synth_centralized_reasoning_v1';self.package.mkdir(parents=True);self.git('init','-q');self.git('config','user.email','cpu@example.invalid');self.git('config','user.name','CPU');self.sources={}
        for name,role in [('source'+str(i)+'.py','python_source') for i in range(194)]+[('prefix_library_observer.c','native_observer_source'),('prefix_library_observer.h','native_observer_source')]:
            p=self.package/name;raw=b'# inert selected source\n';p.write_bytes(raw);self.sources[str(p)]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}
        self.git('add','.');self.git('commit','-qm','implementation');self.impl=self.git('rev-parse','HEAD').decode().strip();self.native={p:d for p,d in self.sources.items() if d['role']=='native_observer_source'};self.runtime=self.seal({'source_commit':self.impl,'files':{},'library_event_observer':{'native_source_files':self.native}});self.config=self.seal({'source_commit':self.impl,'source_files':self.sources,'runtime_files':{}});self.reg=self.seal({'source_commit':self.impl,'source_files':self.sources,'runtime_files':{},'config_body_sha256':self.config['sha256'],'runtime_body_sha256':self.runtime['sha256']});self.packets={}
        for name,v in [('config',self.config),('runtime',self.runtime),('registration',self.reg)]:p=self.package/(name+'.json');raw=(J.canonical(v)+'\n').encode();p.write_bytes(raw);self.packets[name]={'path':str(p),'file_sha256':hashlib.sha256(raw).hexdigest(),'body_sha256':v['sha256']}
        self.git('add','.');self.git('commit','-qm','dispatch');self.head=self.git('rev-parse','HEAD').decode().strip();target=self.package/'evidence/prefix_input_owned_fixture';self.exceptions={}
        for i in range(18):self.exceptions[str(target/('raw'+str(i)+'.json'))]={'role':'evidence_output'}
        for p in self.sources:self.exceptions[str(target/'source_snapshots'/(hashlib.sha256(p.encode()).hexdigest()+Path(p).suffix))]={'role':'evidence_source_snapshot','source_path':p}
    def git(self,*args):return subprocess.run(['git','-c','maintenance.auto=false','-c','gc.auto=0',*args],cwd=self.root,capture_output=True,check=True,timeout=30).stdout
    def seal(self,v):return {**v,'sha256':J.digest(v)}
    def join(self):return J._CPU_join(self.root,packets=self.packets,source_files=self.sources,implementation_commit=self.impl,expected_head=self.head,git_output_exceptions=self.exceptions,_capability=J._CPU_capability())
    def test_actual214future_absent_leaves_and_native2_Q_policy(self):
        self.assertEqual(len(self.exceptions),214);result=self.join();self.assertFalse(result['native_authority_issued']);prior,priorB=Q.PACKAGE,Q.B.PACKAGE;Q.PACKAGE=Q.B.PACKAGE=self.package
        try:Q.import_git_exception_policy(self.exceptions,self.sources,{},self.runtime['library_event_observer'],Q.B)
        finally:Q.PACKAGE=prior;Q.B.PACKAGE=priorB
    def test_unknown_native_originalsource_packet_and_escape_refuse(self):
        original=deepcopy(self.exceptions)
        for p,entry in [(str(self.root/'escape.json'),{'role':'evidence_output'}),(next(iter(self.sources)),{'role':'run_output'}),(self.packets['config']['path'],{'role':'run_output'})]:
            self.exceptions=deepcopy(original);self.exceptions[p]=entry
            with self.subTest(p=p),self.assertRaises(ValueError):self.join()
        self.exceptions=deepcopy(original);source=next(iter(self.native));leaf=next(p for p,e in self.exceptions.items() if e.get('source_path')==source);self.exceptions[leaf]['source_path']=next(p for p,d in self.sources.items() if d['role']=='python_source')
        with self.assertRaises(ValueError):self.join()
    def test_exact_untrackedoutput_allowed_but_foreign_or_tracked_dirty_refuse(self):
        leaf=Path(next(p for p,e in self.exceptions.items() if e['role']=='evidence_output'));leaf.parent.mkdir(parents=True);leaf.write_bytes(b'fixtureoutput');self.join();foreign=leaf.with_name('foreign.json');foreign.write_bytes(b'foreign')
        with self.assertRaises(ValueError):self.join()
        foreign.unlink();source=Path(next(iter(self.sources)));source.write_bytes(b'# modified tracked\n')
        with self.assertRaisesRegex(ValueError,'tracked changes'):self.join()
    def test_wildcard_alias_parent_or_unknown_snapshot_role_refuse(self):
        self.exceptions[str(self.package/'evidence/prefix_input_owned_fixture/*.json')]={'role':'evidence_output'}
        with self.assertRaises(ValueError):self.join()
    def test_Q_rejects_unregistered_native_C_H_even_correct_hashleaf(self):
        prior=Q.PACKAGE;Q.PACKAGE=self.package
        try:
            source=str(self.package/'foreign.c');sources={**self.sources,source:{'bytes':1,'sha256':'a'*64,'role':'native_observer_source'}};leaf=self.package/'evidence/prefix_input_owned_fixture/source_snapshots'/(hashlib.sha256(source.encode()).hexdigest()+'.c');exceptions={str(leaf):{'role':'evidence_source_snapshot','source_path':source}}
            with self.assertRaisesRegex(ValueError,'exactselectednativeobserver'):Q.import_git_exception_policy(exceptions,sources,{},self.runtime['library_event_observer'],Q.B)
        finally:Q.PACKAGE=prior
    def test_full_actual_qualifier_metadata_with_exact_native2_snapshots(self):
        package=next(p for p in HERE.parents if p.name=='rineya_synth_centralized_reasoning_v1');path=package/'artifacts/torch_import_observation_work/native_candidate/test_import_runtime_qualification.py';F=module(path,'CPU_full_Q_native_exception_fixture');base=F.Controls();base.setUp()
        try:
            base.add_passive_overlay();value=base.value;native=value['runtime']['library_event_observer']['native_source_files'];target=package/'evidence/prefix_input_owned_CPU_NATIVE_SNAPSHOT';exceptions={str(target/'source_snapshots'/(hashlib.sha256(p.encode()).hexdigest()+Path(p).suffix)):{'role':'evidence_source_snapshot','source_path':p} for p in native};value['registration']['git_output_exceptions']=exceptions;base.reseal();result=Q._qualify(value,Q.QualificationBudget(),base.B);self.assertFalse(result['current_module_execution_proven'])
        finally:base.doCleanups()
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_REAL_GIT_EXACT_FUTURE_LEAF_AND_NATIVE_OBSERVER_EXCEPTION_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'real_temporary_Git_two_commits':True,'actual214absentleafregistration':True,'native_context_or_authority_issued':False,'source_hashes':{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest() for n in ('registration_git_join.py','import_runtime_qualification.py','test_exception_policy.py')}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
