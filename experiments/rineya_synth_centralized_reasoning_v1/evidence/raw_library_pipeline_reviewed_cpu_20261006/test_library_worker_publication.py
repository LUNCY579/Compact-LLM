"""Actual owned main publishes retained library evidence after cleanup on CPU.

All host/runtime/ownership observations are explicit inert suppliers. External
plan/source/context selection comes from actual sealed fixture documents, never
from the producer's observed boundary list. No native authority is established.
"""
from pathlib import Path
import copy, hashlib, sys, unittest
from unittest.mock import patch
import test_owned_library_entry as owned
import prefix_library_evidence as E


class LibraryWorkerPublicationControls(unittest.TestCase):
    def setUp(self):
        self.case=owned.OwnedLibraryEntryControls();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.W=self.case.W;self.L=self.case.L;self.C=self.case.C;self.fixture=self.case.fixture
        self.m=self.case.fixture_case;self.output=self.case.output
        self.plan=[{'id':1,'parent_id':None,'name':'framework_import'}]
        self.fixture.runtime['library_boundary_plan']=copy.deepcopy(self.plan)
        self.fixture.review['library_boundary_plan_sha256']=self.C.digest(self.plan)
        self.m.reseal()

    def packet(self):
        raw=(self.output/'LIBRARY_BOUNDARY_EVIDENCE.json').read_bytes();value=self.C.parse(raw)
        self.assertEqual(set(value),{'schema_version','kind','source_commit','original_scope','context_sha256',
            'external_anchors','policy_sha256','provider_source','owner_receipt_sha256','raw_packet',
            'raw_packet_sha256','main_errors','fixture_only','native_PASS','native_authority_issued','sha256'})
        self.assertEqual(value['sha256'],self.C.digest({k:v for k,v in value.items() if k!='sha256'}))
        self.assertFalse(value['native_PASS']);self.assertFalse(value['native_authority_issued'])
        self.assertTrue(value['fixture_only']);self.assertEqual(value['kind'],'OWNED_LIBRARY_BOUNDARY_EVIDENCE_V1')
        self.assertEqual(value['raw_packet_sha256'],None if value['raw_packet'] is None else self.C.digest(value['raw_packet']))
        return value

    def replay(self,envelope,context,*,child=False):
        selected=self.C.context_value(context);raw_packet=envelope['raw_packet']
        actual_pid=12346 if child else self.case.owner['worker_pid']
        identity={'source_commit':selected['expected_head'],'original_scope':self.C.NATIVE_SCOPE,
            'owned_birth':{'pid':actual_pid,'creation_filetime_100ns':134000000000000002 if child else 134000000000000001},
            'owned_birth_status':'VALIDATED','context_sha256':self.C.digest(selected),
            'producer_source':{'path':str(self.fixture.package/'prefix_input_owned_worker.py'),
                'descriptor':selected['config']['source_files'][str(self.fixture.package/'prefix_input_owned_worker.py')]},
            'owner_receipt_sha256':self.C.digest(self.case.owner),'fixture_only':True,
            'native_PASS':False,'native_authority_issued':False}
        raw=(self.C.canonical(raw_packet)+'\n').encode('ascii')
        policy=selected['runtime']['library_search_policy']['enforced_boundary']
        return E.validate_observed_packet(raw,policy=policy,expected_identity=identity,
            selected_file_map=policy['library_files'],selected_source_map=selected['config']['source_files'],
            expected_boundaries=selected['runtime']['library_boundary_plan'],expected_file_sha256=hashlib.sha256(raw).hexdigest())

    def test_actual_main_raw_file_written_only_after_close_drain_and_export(self):
        order=[];write=self.W._write_library_boundary_evidence;remove=self.L.remove
        def removing(finder):
            order.append('remove');result=remove(finder);order.append('drained_exported');return result
        def writing(output,value,owner,raw,errors):
            order.append('write');self.assertTrue(self.case.installed.library_guard.closed)
            self.assertTrue(self.case.library_supplier.closed);self.assertIsNotNone(raw)
            self.assertNotIn(self.case.installed,sys.meta_path)
            self.assertEqual(raw['observations'][-1]['event'],'cleanup')
            return write(output,value,owner,raw,errors)
        with self.m.isolated():
            self.case.prepare()
            with patch.object(self.L,'remove',side_effect=removing),patch.object(self.W,'_write_library_boundary_evidence',side_effect=writing):
                proof=self.case.run_main()
            envelope=self.packet();result=self.replay(envelope,proof['original_context'])
        self.assertEqual(order,['remove','drained_exported','write'])
        self.assertEqual(envelope['main_errors'],[]);self.assertTrue(result['conservation_complete'])
        self.assertFalse(result['native_PASS']);self.assertFalse(result['lossless_native_observer_qualified'])
        self.assertEqual(envelope['context_sha256'],self.C.digest(self.C.context_value(proof['original_context'])))
        self.assertEqual(envelope['provider_source'],self.m.selected()['provider_source'])
        self.assertEqual(envelope['owner_receipt_sha256'],self.C.digest(self.case.owner))
        self.assertEqual(envelope['external_anchors'],self.fixture.anchors)
        self.assertFalse(proof['library_terminal_evidence_join_complete'])

    def test_executing_child_birth_selected_instead_of_launcher_birth(self):
        self.case.inert.interpreter_child=True
        with self.m.isolated():
            self.case.prepare();proof=self.case.run_main();envelope=self.packet()
            result=self.replay(envelope,proof['original_context'],child=True)
        birth=envelope['raw_packet']['identity']['owned_birth']
        self.assertEqual(birth['pid'],12346);self.assertNotEqual(birth['pid'],self.case.owner['worker_pid'])
        self.assertEqual(birth['creation_filetime_100ns'],134000000000000002)
        self.assertTrue(result['conservation_complete'])

    def test_ownership_failure_keeps_original_exception_and_unavailable_birth_without_guard_failure(self):
        primary=RuntimeError('inert ownership failure')
        def fail(owner,config):raise primary
        self.case.inert.verify=fail
        with self.m.isolated():
            self.case.prepare()
            with self.assertRaises(RuntimeError) as caught:self.case.run_main()
            envelope=self.packet()
        self.assertIs(caught.exception,primary);self.assertEqual(envelope['main_errors'][0]['stage'],'ownership')
        packet=envelope['raw_packet'];self.assertIsNone(packet['identity']['owned_birth'])
        self.assertEqual(packet['identity']['owned_birth_status'],'UNAVAILABLE')
        self.assertFalse(packet['terminal']['guard_failed']);self.assertTrue(packet['terminal']['guard_closed'])
        self.assertEqual(packet['terminal']['status'],'incomplete');self.assertEqual(packet['failure_records'],[])
        self.assertEqual([row['event'] for row in packet['observations']],['initial','precleanup','cleanup'])
        self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists())

    def test_constructor_failure_retains_actual_partial_packet_and_missing_birth(self):
        primary=ValueError('inert begin failure')
        def begin(*args):raise primary
        with self.m.isolated():
            self.case.prepare();self.case.library_supplier.begin=begin
            with self.assertRaises(ValueError) as caught:self.case.run_main()
            envelope=self.packet()
        self.assertIs(caught.exception,primary);self.assertEqual(envelope['main_errors'][0]['stage'],'install')
        packet=envelope['raw_packet'];self.assertIsNotNone(packet)
        self.assertEqual(packet['identity']['owned_birth_status'],'UNAVAILABLE')
        self.assertIsNone(packet['identity']['owned_birth']);self.assertNotEqual(packet['terminal']['status'],'complete')
        self.assertFalse(packet['producer_claims']['completion_claimed'])
        self.assertGreater(len(packet['failure_records']),0);self.assertTrue(self.case.library_supplier.closed)
        self.assertFalse((self.output/'OWNED_WORKER_RETURN.json').exists())

    def test_owned_return_write_failure_keeps_prior_bytes_and_emits_completed_guard_packet(self):
        prior=b'prior immutable owned return\n'
        with self.m.isolated():
            self.case.prepare();(self.output/'OWNED_WORKER_RETURN.json').write_bytes(prior)
            with self.assertRaises(FileExistsError):self.case.run_main()
            envelope=self.packet()
        self.assertEqual((self.output/'OWNED_WORKER_RETURN.json').read_bytes(),prior)
        self.assertEqual(envelope['main_errors'][0]['stage'],'owned_return')
        self.assertEqual(envelope['raw_packet']['terminal']['status'],'complete')
        self.assertTrue(self.case.library_supplier.closed)

    def test_framework_failure_keeps_failed_raw_checkpoint_after_cleanup(self):
        def fail():raise ValueError('inert framework failure')
        self.case.inert.framework=fail
        with self.m.isolated():
            self.case.prepare()
            with self.assertRaisesRegex(ValueError,'cannot publish completion'):self.case.run_main()
            envelope=self.packet()
        packet=envelope['raw_packet'];self.assertTrue(packet['terminal']['guard_failed'])
        self.assertTrue(packet['terminal']['guard_closed']);self.assertNotEqual(packet['terminal']['status'],'complete')
        self.assertIn('boundary_error',[row['event'] for row in packet['observations']])
        self.assertIn('cleanup',[row['stage'] for row in envelope['main_errors']])

    def test_cleanup_failure_exports_partial_packet_and_never_returns_success(self):
        with self.m.isolated():
            self.case.prepare();self.case.library_supplier.close_error=True
            with self.assertRaisesRegex(ValueError,'DLL search-scope cleanup failed'):self.case.run_main()
            envelope=self.packet()
        self.assertEqual(envelope['main_errors'][0]['stage'],'cleanup')
        packet=envelope['raw_packet'];self.assertNotEqual(packet['terminal']['status'],'complete')
        self.assertTrue(packet['terminal']['guard_closed']);self.assertTrue(packet['terminal']['guard_failed'])
        self.assertFalse(packet['producer_claims']['completion_claimed'])

    def test_export_failure_occurs_after_cleanup_and_publishes_honest_missing_raw(self):
        primary=ValueError('inert retained journal export failure');install=self.L._CPU_install_fixture
        def installing(*args,**kwargs):
            finder=install(*args,**kwargs)
            def fail(identity=None):
                self.assertTrue(finder.library_guard.closed);self.assertTrue(self.case.library_supplier.closed)
                raise primary
            finder.library_guard.export_raw_packet=fail
            return finder
        with self.m.isolated():
            self.case.prepare()
            with patch.object(self.L,'_CPU_install_fixture',side_effect=installing):
                with self.assertRaises(ValueError) as caught:self.case.run_main()
            envelope=self.packet()
        self.assertIs(caught.exception,primary);self.assertIsNone(envelope['raw_packet'])
        self.assertIsNone(envelope['raw_packet_sha256']);self.assertEqual(envelope['main_errors'][0]['stage'],'export')
        self.assertTrue(self.case.library_supplier.closed)

    def test_raw_publication_collision_does_not_replace_existing_bytes(self):
        prior=b'prior immutable library packet\n'
        with self.m.isolated():
            self.case.prepare();(self.output/'LIBRARY_BOUNDARY_EVIDENCE.json').write_bytes(prior)
            with self.assertRaises(FileExistsError):self.case.run_main()
        self.assertEqual((self.output/'LIBRARY_BOUNDARY_EVIDENCE.json').read_bytes(),prior)
        self.assertTrue(self.case.library_supplier.closed)

    def test_dual_primary_cleanup_and_publication_failure_keep_primary_identity_and_cause(self):
        primary=RuntimeError('inert original failure')
        self.case.inert.verify=lambda *args:(_ for _ in ()).throw(primary)
        prior=b'prior library evidence\n'
        with self.m.isolated():
            self.case.prepare();self.case.library_supplier.close_error=True
            (self.output/'LIBRARY_BOUNDARY_EVIDENCE.json').write_bytes(prior)
            with self.assertRaises(RuntimeError) as caught:self.case.run_main()
        self.assertIs(caught.exception,primary);self.assertIsInstance(primary.__cause__,ValueError)
        self.assertIn('cleanup failed',str(primary.__cause__))
        self.assertIsInstance(primary.__library_publication_error__,FileExistsError)
        self.assertEqual(primary.__library_publication_IO__,{'attempted':1,'returned':0,'failed':1,'pending':0})
        self.assertEqual((self.output/'LIBRARY_BOUNDARY_EVIDENCE.json').read_bytes(),prior)

    def test_error_stringifier_failure_never_hides_original_object_or_publication_error(self):
        invoked=[]
        class BadMessage(RuntimeError):
            def __str__(self):invoked.append('str');raise ValueError('message rendering failed')
        primary=BadMessage('unavailable ordinary message')
        self.case.inert.verify=lambda *args:(_ for _ in ()).throw(primary)
        with self.m.isolated():
            self.case.prepare()
            with self.assertRaises(BadMessage) as caught:self.case.run_main()
            envelope=self.packet()
        self.assertIs(caught.exception,primary)
        error=envelope['main_errors'][0]['error'];self.assertEqual(error['type'],'BadMessage')
        self.assertFalse(error['message_exact']);self.assertEqual(error['message'],'<message unavailable>')
        self.assertEqual(invoked,[])

    def test_error_metadata_never_stringifies_or_represents_non_string_arguments(self):
        invoked=[]
        class Representation:
            def __str__(self):invoked.append('str');raise AssertionError('argument str prohibited')
            def __repr__(self):invoked.append('repr');raise AssertionError('argument repr prohibited')
        error=RuntimeError(Representation());record=self.W._library_error_record(error)
        self.assertEqual(record,{'type':'RuntimeError','message':'<message unavailable>','message_exact':False})
        self.assertEqual(invoked,[])

    def test_identity_join_failure_still_closes_scope_and_publishes_unavailable_birth(self):
        original=self.W._library_evidence_identity;failure=ValueError('inert identity join failure')
        def identity(value,owner,**kwargs):
            if kwargs.get('use_topology',True):raise failure
            return original(value,owner,**kwargs)
        with self.m.isolated():
            self.case.prepare()
            with patch.object(self.W,'_library_evidence_identity',side_effect=identity):
                with self.assertRaises(ValueError) as caught:self.case.run_main()
            envelope=self.packet()
        self.assertIs(caught.exception,failure);self.assertTrue(self.case.library_supplier.closed)
        self.assertEqual(envelope['main_errors'][0]['stage'],'export')
        self.assertEqual(envelope['raw_packet']['identity']['owned_birth_status'],'UNAVAILABLE')
        self.assertNotEqual(envelope['raw_packet']['terminal']['status'],'complete')

    def test_external_plan_is_selected_from_context_and_not_producer_observed_boundaries(self):
        with self.m.isolated():
            self.case.prepare();proof=self.case.run_main();envelope=self.packet()
            self.assertTrue(self.replay(envelope,proof['original_context'])['conservation_complete'])
        selected=self.C.context_value(proof['original_context']);policy=selected['runtime']['library_search_policy']['enforced_boundary']
        raw=(self.C.canonical(envelope['raw_packet'])+'\n').encode('ascii')
        wrong=[{'id':1,'parent_id':None,'name':'framework_import'},
            {'id':2,'parent_id':1,'name':'extension:'+self.m.extension+':create'}]
        with self.assertRaisesRegex(ValueError,'complete external boundary population|complete'):
            E.validate_observed_packet(raw,policy=policy,expected_identity=envelope['raw_packet']['identity'],
                selected_file_map=policy['library_files'],selected_source_map=selected['config']['source_files'],
                expected_boundaries=wrong,expected_file_sha256=hashlib.sha256(raw).hexdigest())


if __name__=='__main__':unittest.main()
