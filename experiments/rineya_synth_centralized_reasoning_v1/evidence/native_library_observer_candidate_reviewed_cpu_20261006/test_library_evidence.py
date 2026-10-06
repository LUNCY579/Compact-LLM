"""Source-shaped offline packets; current guard does not emit this packet.

Inert actual files provide external descriptors/directory identities. Raw event
and module observations are explicit CPU fixtures, never host DLL observations.
"""
from pathlib import Path
from copy import deepcopy
from unittest.mock import patch
import builtins,hashlib,tempfile,unittest
import prefix_library_evidence as E

HERE=Path(__file__).resolve().parent

def descriptor(raw,role):return {'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'role':role}

class LibraryEvidenceControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_library_evidence_',dir=HERE);self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name).resolve();self.system=self.root/'System32';self.system.mkdir();self.runtime=self.root/'runtime';self.runtime.mkdir()
        self.attempts=[];original=builtins.__import__
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'ctypes','torch','psutil','transformers','numpy','safetensors','native_worker','native_process','native_nvml'}:
                self.attempts.append(name);raise AssertionError('actual native import prohibited: '+name)
            return original(name,*args,**kwargs)
        trap=patch.object(builtins,'__import__',side_effect=imports);trap.start();self.addCleanup(trap.stop)
        self.addCleanup(lambda:self.assertEqual(self.attempts,[]))
        self.files={};self.names={}
        for name,root,role in (('base.exe',self.system,'interpreter_binary'),('ext.pyd',self.runtime,'extension_binary'),('dep.dll',self.runtime,'native_library'),('direct.dll',self.runtime,'native_library'),('unused.dll',self.runtime,'native_library')):
            p=root/name;raw=('inert CPU selected bytes '+name+'\n').encode();p.write_bytes(raw)
            self.names[name]=str(p);self.files[str(p)]=descriptor(raw,role)
        guard=self.root/'prefix_library_policy.py';guard.write_bytes((HERE/'prefix_library_policy.py').read_bytes())
        producer=self.root/'CPU_future_packet_producer.py';producer.write_bytes(b'# explicit CPU fixture; no native packet producer exists\n')
        self.sources={str(p):descriptor(p.read_bytes(),'python_source') for p in (guard,producer)}
        self.policy={'kind':'FINITE_WINDOWS_DLL_BOUNDARY_V1','search_directories':[str(self.system),str(self.runtime)],'system_directory':str(self.system),
            'library_files':deepcopy(self.files),'initial_loaded_libraries':[self.names['base.exe']],
            'extension_transitions':{self.names['ext.pyd']:{'create':[self.names['ext.pyd']],'exec':[self.names['dep.dll']]}},
            'framework_transition':[self.names['direct.dll']],
            'provider_source':{'path':str(guard),'descriptor':self.sources[str(guard)]}}
        self.identity={'source_commit':'a'*40,'original_scope':E.ORIGINAL_SCOPE,'owned_birth':{'pid':101,'creation_filetime_100ns':1234567},
            'context_sha256':'b'*64,'producer_source':{'path':str(producer),'descriptor':deepcopy(self.sources[str(producer)])},
            'fixture_only':True,'native_PASS':False,'native_authority_issued':False}
        self.plan=[{'id':1,'parent_id':None,'name':'framework_import'},
            {'id':2,'parent_id':1,'name':'extension:'+self.names['ext.pyd']+':create'},
            {'id':3,'parent_id':1,'name':'extension:'+self.names['ext.pyd']+':exec'}]

    def scope(self):
        common={'directories':self.policy['search_directories'],'system_directory':self.policy['system_directory'],'default_flags':0xC00,'legacy_directory':'','closed':False}
        return ({**common,'kind':'OBSERVED_DLL_DIRECTORY_SCOPE','complete':True,'unmanaged_directory_registration':False,'completeness_basis':'explicit_injected_CPU_observation'}
            if self.policy['kind'].endswith('V1') else {**common,'kind':'OBSERVED_MANAGED_DLL_DIRECTORY_SCOPE','managed_setup_complete':True,'foreign_directory_registration':'UNOBSERVED','evidence_basis':'explicit_injected_CPU_managed_observation'})

    def cleanup(self):
        common={'remaining_managed_directories':[],'legacy_directory':'','closed':True}
        return ({**common,'kind':'OBSERVED_DLL_DIRECTORY_CLEANUP','complete':True,'all_cookies_removed':True}
            if self.policy['kind'].endswith('V1') else {**common,'kind':'OBSERVED_MANAGED_DLL_DIRECTORY_CLEANUP','managed_cleanup_complete':True,'all_managed_cookies_removed':True,'foreign_directory_registration':'UNOBSERVED'})

    def packet(self):
        state={self.names['base.exe']:101};events=[];points=[]
        dirs={str(p):{'device':p.stat().st_dev,'inode':p.stat().st_ino} for p in (self.system,self.runtime)}
        def add(kind,bid=None,new=None):
            if new:
                p=self.names[new];handle=101+len(events)+1;state[p]=handle
                events.append({'sequence':len(events)+1,'event':'load','path':p,'module_handle':handle})
            points.append({'ordinal':len(points)+1,'time_ns':(len(points)+1)*10**6,'event':kind,'boundary_id':bid,
                'scope':None if kind=='cleanup' else self.scope(),
                'snapshot':{'kind':'OBSERVED_PROCESS_LIBRARY_SNAPSHOT','complete':True,'libraries':[{'path':p,'module_handle':h} for p,h in state.items()]},
                'event_trace':{'kind':'OBSERVED_PROCESS_LIBRARY_EVENTS','complete':True,'events':deepcopy(events)},
                'directory_identities':deepcopy(dirs),'cleanup':self.cleanup() if kind=='cleanup' else None})
        add('initial');add('enter',1);add('enter',2);add('exit',2,'ext.pyd');add('enter',3);add('exit',3,'dep.dll');add('exit',1,'direct.dll');add('precleanup');add('cleanup')
        return {'schema_version':1,'kind':'CPU_LIBRARY_BOUNDARY_RAW_PACKET_V1','identity':deepcopy(self.identity),'policy_sha256':E.digest(self.policy),
            'library_files':deepcopy(self.files),'observations':points,
            'producer_claims':{'completion_claimed':True,'observation_count':len(points),'event_count':len(events),'boundary_count':3,'complete_search_directory_coverage':self.policy['kind'].endswith('V1')},
            'fixture_only':True,'native_PASS':False,'native_authority_issued':False}

    def raw(self,value):
        value={k:v for k,v in value.items() if k!='sha256'};value['sha256']=E.digest(value)
        return (E.canonical(value)+'\n').encode('ascii')

    def run_raw(self,raw):
        return E.validate_packet(raw,policy=self.policy,expected_identity=self.identity,selected_file_map=self.files,
            selected_source_map=self.sources,expected_boundaries=self.plan,expected_file_sha256=hashlib.sha256(raw).hexdigest())

    def accept(self,value):return self.run_raw(self.raw(value))
    def reject(self,value,pattern):
        with self.assertRaisesRegex(ValueError,pattern):self.accept(value)

    def test_nested_source_transitions_recomputed_without_native_authority(self):
        result=self.accept(self.packet())
        self.assertEqual(result['independently_recomputed'],{'observation_count':9,'event_count':3,'boundary_count':3})
        self.assertEqual(set(result['final_libraries']),{self.names[n] for n in ('base.exe','ext.pyd','dep.dll','direct.dll')})
        self.assertTrue(result['fixture_only'])
        for key in ('native_PASS','native_authority_issued','native_producer_integration_verified','current_runtime_bytes_qualified','lossless_native_observer_qualified','actual_search_directory_coverage_qualified'):self.assertFalse(result[key])

    def test_V2_unknown_foreign_state_is_honest_and_no_coverage_claim(self):
        self.policy['kind']='FINITE_WINDOWS_DLL_BOUNDARY_V2';good=self.packet();self.accept(good)
        bad=deepcopy(good);bad['observations'][0]['scope']['foreign_directory_registration']=False
        self.reject(bad,'UNOBSERVED')
        bad=deepcopy(good);bad['producer_claims']['complete_search_directory_coverage']=True
        self.reject(bad,'V2 foreign scope is UNOBSERVED')

    def test_event_prefix_drop_reorder_rewrite_after_reseal(self):
        for mode in ('drop','reorder','rewrite'):
            with self.subTest(mode=mode):
                bad=self.packet();rows=bad['observations'][-1]['event_trace']['events']
                if mode=='drop':rows.pop()
                elif mode=='reorder':rows.reverse()
                else:rows[0]['module_handle']=999
                self.reject(bad,'append-only event trace')

    def test_missing_load_cannot_be_hidden_by_complete_snapshot(self):
        bad=self.packet();bad['observations'][3]['event_trace']['events']=[]
        self.reject(bad,'replayed events equal complete snapshot')

    def test_transient_unload_and_unexpected_load_refused(self):
        bad=self.packet();point=bad['observations'][3];point['event_trace']['events'].append({'sequence':2,'event':'unload','path':self.names['ext.pyd'],'module_handle':102})
        self.reject(bad,'unload/transient lifetime')
        bad=self.packet();point=bad['observations'][3];point['event_trace']['events'][0]['path']=self.names['unused.dll']
        self.reject(bad,'exact prospective load path/handle')
        bad=self.packet();bad['observations'][3]['event_trace']['events'][0]['path']=str(self.runtime/'unselected.dll')
        self.reject(bad,'ordered registered raw library event')

    def test_missing_reordered_and_bool_boundary_identity_refused(self):
        bad=self.packet();del bad['observations'][2:4]
        for i,row in enumerate(bad['observations']):row['ordinal']=i+1;row['time_ns']=(i+1)*10**6
        self.reject(bad,'exact source boundary order/parent')
        bad=self.packet();bad['observations'][2]['boundary_id']=3
        self.reject(bad,'exact source boundary order/parent')
        bad=self.packet();bad['observations'][1]['boundary_id']=True
        self.reject(bad,'typed positive boundary identity')

    def test_handle_alias_and_existing_handle_replacement_refused(self):
        bad=self.packet();bad['observations'][3]['snapshot']['libraries'][1]['module_handle']=101
        self.reject(bad,'unique registered snapshot path/handle')
        bad=self.packet();bad['observations'][3]['snapshot']['libraries'][0]['module_handle']=999
        self.reject(bad,'replayed events equal complete snapshot')

    def test_external_byte_source_and_original_identity_joins(self):
        bad=self.packet();bad['library_files'][self.names['ext.pyd']]['sha256']='0'*64
        self.reject(bad,'external original identity/policy/byte-map')
        bad=self.packet();bad['identity']['owned_birth']['creation_filetime_100ns']+=1
        self.reject(bad,'external original identity/policy/byte-map')
        old=self.sources[self.identity['producer_source']['path']]['sha256'];self.sources[self.identity['producer_source']['path']]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'external producer source byte join'):self.accept(self.packet())
        self.sources[self.identity['producer_source']['path']]['sha256']=old

    def test_cleanup_and_numeric_scope_bool_masquerades_refused(self):
        for kind,stage,key,number in [('V1',0,'complete',1),('V1',0,'unmanaged_directory_registration',0),('V1',0,'closed',0),('V2',0,'managed_setup_complete',1),('V2',8,'all_managed_cookies_removed',1),('V2',8,'closed',1)]:
            with self.subTest(kind=kind,key=key):
                self.policy['kind']='FINITE_WINDOWS_DLL_BOUNDARY_'+kind;bad=self.packet()
                target=bad['observations'][stage]['cleanup' if stage==8 else 'scope'];target[key]=number
                self.reject(bad,'exact scope observation|exact successful terminal cleanup')
        self.policy['kind']='FINITE_WINDOWS_DLL_BOUNDARY_V2';bad=self.packet();bad['observations'][-1]['cleanup']['managed_cleanup_complete']=False
        self.reject(bad,'exact successful terminal cleanup')

    def test_CPU_flags_and_producer_counts_are_not_authority(self):
        bad=self.packet();bad['identity']['fixture_only']=1
        self.reject(bad,'incoming identity retains typed CPU flags')
        bad=self.packet();bad['native_PASS']=True
        self.reject(bad,'raw CPU evidence cannot issue native authority')
        bad=self.packet();bad['producer_claims']['event_count']=0
        self.reject(bad,'independently recomputed conservation')

    def test_directory_and_clock_chronology_conservation(self):
        bad=self.packet();bad['observations'][4]['directory_identities'][str(self.system)]['inode']+=1
        self.reject(bad,'directory physical identity conservation')
        bad=self.packet();bad['observations'][4]['time_ns']=0
        self.reject(bad,'chronological checkpoint clock')
        bad=self.packet();bad['observations'][-1]['time_ns']=601*10**9
        self.reject(bad,'bounded recorded scope duration')

    def test_complete_checkpoint_and_snapshot_population_is_required(self):
        bad=self.packet();bad['observations'][-1]['event_trace']['complete']=False
        self.reject(bad,'complete bounded raw event batch')
        bad=self.packet();bad['observations'][0]['snapshot']['libraries'].append({'path':self.names['unused.dll'],'module_handle':987})
        self.reject(bad,'exact initial snapshot population')
        bad=self.packet();del bad['observations'][-2]
        self.reject(bad,'typed ordered raw checkpoint')

    def test_actual_strict_byte_parser_duplicate_overflow_and_bounds(self):
        for raw,pattern in ((b'{"x":1,"x":2}','duplicate JSON field'),(b'{"x":NaN}','nonfinite JSON constant'),(b'{"x":1e999}','nonfinite/overflow'),(b'{"x":18446744073709551616}','bounded integer')):
            with self.subTest(raw=raw):
                with self.assertRaisesRegex(ValueError,pattern):self.run_raw(raw)
        raw=b' '*(E.MAX_PACKET_BYTES+1)
        with self.assertRaisesRegex(ValueError,'packet file bytes'):self.run_raw(raw)
        with self.assertRaisesRegex(ValueError,'packet file bytes'):
            E.validate_packet(self.raw(self.packet()),policy=self.policy,expected_identity=self.identity,selected_file_map=self.files,selected_source_map=self.sources,expected_boundaries=self.plan,expected_file_sha256='0'*64)


if __name__=='__main__':unittest.main()
