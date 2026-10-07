"""Real original raw event/snapshot replay, inert files and CPU observations."""
from pathlib import Path
from copy import deepcopy
from types import ModuleType
import hashlib,json,sys,unittest
HERE=Path(__file__).resolve().parent;PACKAGE=HERE.parents[1];SOURCE=PACKAGE/'evidence/raw_library_pipeline_reviewed_cpu_20261006'
def load(path,name):
    m=ModuleType(name);m.__file__=str(path);exec(compile(path.read_bytes(),str(path),'exec'),m.__dict__);return m
G=load(HERE/'bounded_boundary_grammar.py','CPU_bounded_library_grammar');E=G.engine()
sys.modules['prefix_library_evidence']=E
B=load(SOURCE/'test_library_evidence.py','CPU_original_raw_library_fixture')
class Controls(unittest.TestCase):
    def setUp(self):
        self.base=B.LibraryEvidenceControls();self.base.setUp();self.addCleanup(self.base.doCleanups);self.policy=deepcopy(self.base.policy);self.files=self.base.files;self.sources=self.base.sources;self.identity={**self.base.identity,'owned_birth_status':'VALIDATED','owner_receipt_sha256':'c'*64};self.ext=self.base.names['ext.pyd'];self.second=str(self.base.runtime/'second.pyd');raw=b'inert second extension\n';Path(self.second).write_bytes(raw);self.files[self.second]=B.descriptor(raw,'extension_binary');self.policy['library_files']=deepcopy(self.files);self.policy['extension_transitions'][self.second]={'create':[self.second],'exec':[]};self.grammar={'schema_version':2,'kind':'SOURCE_BOUND_FINITE_LIBRARY_BOUNDARY_GRAMMAR_V2','required_extensions':[self.ext,self.second],'optional_extensions':[],'maximum_boundaries':5,'maximum_depth':64,'maximum_observations':4096,'model':False}
    def packet(self,reverse=False):
        state={self.base.names['base.exe']:101};events=[];points=[];plan=[{'id':1,'parent_id':None,'name':'framework_import'}];bid=2
        for selected in ([self.second,self.ext] if reverse else [self.ext,self.second]):
            for operation in ('create','exec'):plan.append({'id':bid,'parent_id':1,'name':'extension:'+selected+':'+operation});bid+=1
        dirs={str(p):{'device':p.stat().st_dev,'inode':p.stat().st_ino} for p in (self.base.system,self.base.runtime)}
        def add(event,boundary=None,new=()):
            for path in new:
                handle=102+len(events);state[path]=handle;events.append({'sequence':len(events)+1,'event':'load','path':path,'module_handle':handle})
            points.append({'ordinal':len(points)+1,'time_ns':(len(points)+1)*10**6,'event':event,'boundary_id':None if boundary is None else boundary['id'],'boundary':deepcopy(boundary),'scope':None if event=='cleanup' else self.base.scope(),'snapshot':{'kind':'OBSERVED_PROCESS_LIBRARY_SNAPSHOT','complete':True,'libraries':[{'path':p,'module_handle':h} for p,h in state.items()]},'event_trace':{'kind':'OBSERVED_PROCESS_LIBRARY_EVENTS','complete':True,'events':deepcopy(events)},'directory_identities':deepcopy(dirs),'cleanup':self.base.cleanup() if event=='cleanup' else None,'observation_errors':{}})
        add('initial');add('enter',plan[0])
        for boundary in plan[1:]:
            add('enter',boundary);add('exit',boundary,E.expected_new(self.policy,boundary['name']))
        add('exit',plan[0],self.policy['framework_transition']);add('precleanup');add('cleanup')
        return {'schema_version':2,'kind':E.OBSERVED_PACKET_KIND,'identity':deepcopy(self.identity),'policy_sha256':E.digest(self.policy),'library_files':deepcopy(self.files),'observations':points,'failure_records':[],'terminal':{'status':'complete','recording_complete':True,'guard_failed':False,'guard_closed':True},'producer_claims':{'completion_claimed':True,'observation_count':len(points),'event_count':len(events),'boundary_count':5,'complete_search_directory_coverage':True},'fixture_only':True,'native_PASS':False,'native_authority_issued':False}
    def run_packet(self,value):
        value={k:v for k,v in value.items() if k!='sha256'};value['sha256']=E.digest(value);raw=(E.canonical(value)+'\n').encode();return G.validate_observed(raw,policy=self.policy,grammar=self.grammar,expected_identity=self.identity,selected_file_map=self.files,selected_source_map=self.sources,expected_file_sha256=hashlib.sha256(raw).hexdigest())
    def test_sibling_order_permutations_with_exact_original_effect_replay(self):
        for reverse in (False,True):
            result=self.run_packet(self.packet(reverse));self.assertTrue(result['conservation_complete']);self.assertTrue(result['exact_library_effect_replay_unchanged']);self.assertFalse(result['native_PASS'])
    def test_unknown_repeat_unmatched_parent_and_missing_required_refuse(self):
        for change in ('unknown','repeat','parent','missing'):
            value=self.packet();rows=value['observations']
            if change=='unknown':rows[2]['boundary']['name']='extension:unregistered:create'
            elif change=='repeat':rows[4]['boundary']['name']=rows[2]['boundary']['name']
            elif change=='parent':rows[2]['boundary']['parent_id']=None
            else:
                del rows[6:10]
                for i,row in enumerate(rows):row['ordinal']=i+1
            with self.subTest(change=change),self.assertRaises(ValueError):self.run_packet(value)
    def test_wrong_effect_or_missing_event_still_rejected_by_original_consumer(self):
        value=self.packet();value['observations'][3]['event_trace']['events']=[]
        with self.assertRaises(ValueError):self.run_packet(value)
        value=self.packet();self.policy['extension_transitions'][self.ext]['create']=[];value['policy_sha256']=E.digest(self.policy)
        with self.assertRaises(ValueError):self.run_packet(value)
    def test_exec_before_create_and_unclosed_stack_cannot_complete(self):
        value=self.packet();value['observations'][2]['boundary']['name']='extension:'+self.ext+':exec'
        with self.assertRaisesRegex(ValueError,'exec follows'):self.run_packet(value)
        value=self.packet();del value['observations'][-3]
        with self.assertRaises(ValueError):self.run_packet(value)
if __name__=='__main__':
    result=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls));v={'kind':'CPU_BOUNDARY_GRAMMAR_V2_OPTIONAL_DRAFT_CONTROLS','tests_run':result.testsRun,'failures':len(result.failures),'errors':len(result.errors),'actual_original_event_snapshot_replay_executed':True,'observations_explicit_inert_CPU':True,'native_API_or_framework_called':False,'integrated_policy_changed':False,'native_authority_issued':False,'source_hashes':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (HERE/'bounded_boundary_grammar.py',Path(__file__).resolve(),G.SOURCE)}}
    if len(sys.argv)>1:
        with Path(sys.argv[1]).open('x',encoding='utf-8') as f:json.dump(v,f,sort_keys=True,indent=2);f.write('\n')
    raise SystemExit(0 if result.wasSuccessful() else 1)
