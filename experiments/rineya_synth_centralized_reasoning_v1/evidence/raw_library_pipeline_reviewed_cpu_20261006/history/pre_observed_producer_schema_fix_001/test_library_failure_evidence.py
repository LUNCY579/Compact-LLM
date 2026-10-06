"""Observed V2 complete/failure packets; explicit CPU source-shaped fixtures.

External plans/identity anchors remain independent from incoming producer rows.
No native observer, current runtime or terminal integration is qualified here.
"""
from copy import deepcopy
import hashlib,unittest
import prefix_library_evidence as E
import test_library_evidence as historical
import test_library_policy as producer_fixture

class ObservedLibraryFailureControls(unittest.TestCase):
    def setUp(self):
        self.base=historical.LibraryEvidenceControls();self.base.setUp();self.addCleanup(self.base.doCleanups)
        self.policy=self.base.policy;self.files=self.base.files;self.sources=self.base.sources;self.plan=self.base.plan
        self.identity={**deepcopy(self.base.identity),'owned_birth_status':'VALIDATED','owner_receipt_sha256':'c'*64}

    def packet(self):
        value=self.base.packet();value.update(schema_version=2,kind=E.OBSERVED_PACKET_KIND,identity=deepcopy(self.identity),failure_records=[],
            terminal={'status':'complete','recording_complete':True,'guard_failed':False,'guard_closed':True})
        for row in value['observations']:
            row['boundary']=deepcopy(next((p for p in self.plan if p['id']==row['boundary_id']),None))
            row['observation_errors']={}
        return value

    def error(self,message='inert delegate original failure'):
        return {'type':'ValueError','message':message,'message_exact':True}

    def failure(self):
        value=self.packet();rows=value['observations'][:4];rows[-1]['event']='boundary_error'
        parent=deepcopy(rows[-1]);parent.update(ordinal=5,time_ns=5*10**6,boundary_id=1,boundary=deepcopy(self.plan[0]))
        before=deepcopy(parent);before.update(ordinal=6,time_ns=6*10**6,event='precleanup',boundary_id=None,boundary=None)
        closed=deepcopy(before);closed.update(ordinal=7,time_ns=7*10**6,event='cleanup',scope=None,cleanup=self.base.cleanup())
        value['observations']=rows+[parent,before,closed]
        value['failure_records']=[{'ordinal':1,'stage':'boundary_error','boundary_id':2,'observation_ordinal':4,'error':self.error()},
            {'ordinal':2,'stage':'boundary_error','boundary_id':1,'observation_ordinal':5,'error':self.error()}]
        value['terminal']={'status':'failed','recording_complete':True,'guard_failed':True,'guard_closed':True}
        value['producer_claims'].update(completion_claimed=False,observation_count=7,event_count=1,boundary_count=2)
        return value

    def accept(self,value):
        body={k:v for k,v in value.items() if k!='sha256'};body['sha256']=E.digest(body)
        raw=(E.canonical(body)+'\n').encode('ascii')
        return E.validate_observed_packet(raw,policy=self.policy,expected_identity=self.identity,selected_file_map=self.files,
            selected_source_map=self.sources,expected_boundaries=self.plan,expected_file_sha256=hashlib.sha256(raw).hexdigest())

    def reject(self,value,pattern):
        with self.assertRaisesRegex(ValueError,pattern):self.accept(value)

    def component_error(self,value,ordinal,component,raw=None):
        point=value['observations'][ordinal-1];point[component]=raw;error=self.error(component+' observation unavailable')
        point['observation_errors'][component]=error
        value['failure_records'].append({'ordinal':len(value['failure_records'])+1,'stage':'observation:'+component,
            'boundary_id':point['boundary_id'],'observation_ordinal':ordinal,'error':deepcopy(error)})

    def test_complete_packet_independent_reached_counts_remain_unqualified(self):
        result=self.accept(self.packet());self.assertTrue(result['conservation_complete'])
        self.assertEqual(result['reached_counts'],{'observations':9,'entered':3,'returned':3,'failed':0,'unreached':0,'verified_events':3})
        self.assertTrue(result['fixture_only']);self.assertFalse(result['native_PASS']);self.assertFalse(result['lossless_native_observer_qualified'])

    def test_delegate_failure_retains_partial_load_and_unreached_plan(self):
        result=self.accept(self.failure());self.assertEqual(result['status'],'OBSERVED_LIBRARY_FAILURE_PRESERVED')
        self.assertEqual(result['reached_counts'],{'observations':7,'entered':2,'returned':0,'failed':2,'unreached':1,'verified_events':1})
        self.assertEqual(result['failure_records'][0]['error']['message'],'inert delegate original failure')
        self.assertFalse(result['conservation_complete']);self.assertFalse(result['native_PASS'])

    def test_missing_post_failure_observation_is_explicit_unknown(self):
        value=self.failure();self.component_error(value,4,'snapshot')
        result=self.accept(value);self.assertFalse(result['verified_prefix_available']);self.assertIsNone(result['final_verified_libraries'])
        self.assertIsNone(result['producer_event_count_consistent'])
        self.assertIn({'ordinal':4,'component':'snapshot'},result['unknown_components'])
        value['observations'][3]['observation_errors']={}
        self.reject(value,'missing observed component must have explicit error')

    def test_cleanup_failure_retains_original_and_ordered_secondary(self):
        value=self.failure();self.component_error(value,7,'cleanup');value['terminal'].update(status='incomplete',recording_complete=False)
        result=self.accept(value);self.assertEqual(result['status'],'OBSERVED_LIBRARY_INCOMPLETE_PRESERVED')
        self.assertEqual(result['failure_records'][0]['error']['message'],'inert delegate original failure')
        self.assertEqual(result['failure_records'][-1]['error']['message'],'cleanup observation unavailable')
        self.assertFalse(result['conservation_complete'])
        del value['failure_records'][-1];self.reject(value,'component error joins retained failure record')

    def test_unavailable_birth_and_no_observations_never_fabricate_identity(self):
        self.identity.update(owned_birth=None,owned_birth_status='UNAVAILABLE');value=self.packet()
        value.update(observations=[],failure_records=[{'ordinal':1,'stage':'provider_begin','boundary_id':None,'observation_ordinal':None,'error':self.error()}],
            terminal={'status':'incomplete','recording_complete':False,'guard_failed':True,'guard_closed':False})
        value['producer_claims'].update(completion_claimed=False,observation_count=0,event_count=0,boundary_count=0)
        result=self.accept(value);self.assertIsNone(result['original_identity']['owned_birth']);self.assertEqual(result['reached_counts']['entered'],0)
        self.assertFalse(result['verified_prefix_available']);self.assertEqual(result['unobserved_stages'],['initial','precleanup','cleanup'])
        bad=self.packet();self.reject(bad,'unavailable birth cannot complete')

    def test_missing_cleanup_open_boundary_and_unexact_message_cannot_complete(self):
        value=self.failure();value['observations']=value['observations'][:4];value['failure_records']=value['failure_records'][:1]
        value['terminal'].update(status='incomplete',recording_complete=False,guard_closed=False);value['producer_claims']['observation_count']=4
        result=self.accept(value);self.assertEqual(result['status'],'OBSERVED_LIBRARY_INCOMPLETE_PRESERVED');self.assertFalse(result['conservation_complete'])
        self.assertIn('cleanup',result['unobserved_stages']);self.assertFalse(result['cleanup_observation_verified'])
        value=self.failure();value['failure_records'][0]['error']['message_exact']=False
        result=self.accept(value);self.assertEqual(result['status'],'OBSERVED_LIBRARY_INCOMPLETE_PRESERVED')

    def test_partial_packet_resealed_success_claim_rejected(self):
        value=self.failure();value['terminal'].update(status='complete',guard_failed=False);value['producer_claims']['completion_claimed']=True
        self.reject(value,'success has no failures')
        value=self.failure();value['producer_claims']['completion_claimed']=True
        self.reject(value,'typed complete/failure producer claims')

    def test_emitted_plan_names_parent_and_attempted_counts_are_external(self):
        value=self.failure();value['observations'][2]['boundary']['name']=self.plan[2]['name']
        self.reject(value,'emitted name/parent/order equals external plan')
        value=self.failure();value['observations'][2]['boundary']['parent_id']=None
        self.reject(value,'emitted name/parent/order equals external plan')
        value=self.failure();value['producer_claims']['boundary_count']=0
        self.reject(value,'independent reached checkpoint/entry counts')

    def test_failure_prefix_drop_rewrite_handle_and_unload_refused(self):
        value=self.failure();value['observations'][-1]['event_trace']['events']=[]
        self.reject(value,'invalid replay must retain explicit component failure')
        value=self.failure();value['observations'][-1]['snapshot']['libraries'][0]['module_handle']=999
        self.reject(value,'invalid replay must retain explicit component failure')
        value=self.failure();value['observations'][3]['event_trace']['events'].append({'sequence':2,'event':'unload','path':self.base.names['ext.pyd'],'module_handle':102})
        self.reject(value,'invalid replay must retain explicit component failure')

    def test_explicit_bad_raw_component_preserved_as_unknown_not_success(self):
        value=self.failure();bad={'kind':'OBSERVED_PROCESS_LIBRARY_SNAPSHOT','complete':True,'libraries':[{'path':str(self.base.runtime/'unknown.dll'),'module_handle':777}]}
        self.component_error(value,4,'snapshot',bad)
        result=self.accept(value);self.assertTrue(result['independent_violations']);self.assertFalse(result['conservation_complete'])
        self.assertFalse(result['verified_prefix_available'])

    def test_V2_foreign_honesty_numeric_bool_and_error_links(self):
        self.policy['kind']='FINITE_WINDOWS_DLL_BOUNDARY_V2';value=self.failure();self.accept(value)
        value['observations'][0]['scope']['foreign_directory_registration']=False
        self.reject(value,'invalid raw component requires its explicit failure')
        value=self.failure();value['terminal']['guard_failed']=1
        self.reject(value,'exact typed observed terminal state')
        value=self.failure();value['failure_records'][0]['ordinal']=2
        self.reject(value,'ordered bounded original failure records')

    def test_recorded_nonfixture_provenance_is_not_CPU_relabel_or_qualification(self):
        self.identity['fixture_only']=False;self.policy['kind']='FINITE_WINDOWS_DLL_BOUNDARY_V2';value=self.packet();value['fixture_only']=False
        for row in value['observations']:
            if row['scope'] is not None:row['scope']['evidence_basis']='Win32_managed_directory_operations'
        result=self.accept(value);self.assertFalse(result['fixture_only']);self.assertFalse(result['native_PASS']);self.assertFalse(result['native_producer_integration_verified'])
        value['identity']['fixture_only']=True;self.reject(value,'external observed source/context/owner identity')

    def producer(self):
        case=producer_fixture.LibraryBoundaryTests();case.setUp();self.addCleanup(case.doCleanups)
        selected=case.policy['provider_source']
        identity={**deepcopy(self.identity),'producer_source':deepcopy(selected)}
        plan=[{'id':1,'parent_id':None,'name':'framework_import'},
            {'id':2,'parent_id':1,'name':'extension:'+case.ext+':create'},
            {'id':3,'parent_id':1,'name':'extension:'+case.ext+':exec'}]
        return case,identity,plan

    def replay_produced(self,case,identity,plan,guard):
        packet=guard.export_raw_packet(identity)
        raw=(E.canonical(packet)+'\n').encode('ascii')
        return E.validate_observed_packet(raw,policy=case.policy,expected_identity=identity,selected_file_map=case.files,
            selected_source_map={identity['producer_source']['path']:identity['producer_source']['descriptor']},
            expected_boundaries=plan,expected_file_sha256=hashlib.sha256(raw).hexdigest())

    def test_actual_current_guard_emits_complete_packet_consumed_independently(self):
        case,identity,plan=self.producer();guard=case.guard()
        with guard.framework_import():
            with guard.extension(case.ext,'create'):case.supplier.add(case.ext);case.supplier.add(case.dep)
            with guard.extension(case.ext,'exec'):pass
        guard.close();result=self.replay_produced(case,identity,plan,guard)
        self.assertTrue(result['conservation_complete']);self.assertFalse(result['native_PASS'])
        self.assertEqual(result['reached_counts']['entered'],3)

    def test_actual_current_guard_original_delegate_failure_packet_preserved(self):
        case,identity,plan=self.producer();guard=case.guard()
        with self.assertRaisesRegex(ValueError,'actual inert original delegate failure'):
            with guard.framework_import():
                with guard.extension(case.ext,'create'):
                    case.supplier.add(case.ext);raise ValueError('actual inert original delegate failure')
        with self.assertRaisesRegex(ValueError,'failed DLL boundary cannot publish completion'):guard.close()
        result=self.replay_produced(case,identity,plan,guard)
        self.assertFalse(result['conservation_complete']);self.assertFalse(result['native_PASS'])
        self.assertTrue(any(row['error']['message']=='actual inert original delegate failure' for row in result['failure_records']))


if __name__=='__main__':unittest.main()
