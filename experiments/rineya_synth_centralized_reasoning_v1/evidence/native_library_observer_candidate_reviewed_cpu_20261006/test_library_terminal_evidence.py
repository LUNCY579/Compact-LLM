"""Actual owned library packet -> original/preserved bytes -> independent terminal replay.

All native process, resource, bootstrap and library observations are explicit
CPU suppliers. No returned PASS or library replay confers native authority.
"""
from pathlib import Path
from copy import deepcopy
import hashlib,unittest
from unittest.mock import patch
import test_owned_library_entry as O
from test_full_native_cpu_workflow import compiled

HERE=Path(__file__).resolve().parent
class LibraryTerminalEvidenceControls(unittest.TestCase):
    def setUp(self):
        self.case=O.OwnedLibraryEntryControls();self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.c=self.case;self.f=self.c.fixture;self.C=self.c.C
        plan=[{'id':1,'parent_id':None,'name':'framework_import'}]
        self.f.runtime['library_boundary_plan']=plan
        self.f.review['library_boundary_plan_sha256']=self.C.digest(plan)
        self.c.fixture_case.reseal()
    def produce(self,fail_framework=False):
        c=self.c
        with c.fixture_case.isolated():
            c.prepare()
            if fail_framework:
                def fail():raise RuntimeError('inert original framework import failure')
                c.inert.framework=fail;execution=[]
                execute=c.W._owned_execution_factory(c.L,_CPU_capability=c.cap,suppliers=c.inert.mapping(),host=c.host,execution=execution)
                def installed(ctx,contract):return c.install()
                with self.assertRaisesRegex(ValueError,'cannot publish completion'):
                    c.W._registered_main_body(c.context,c.owner,c.output,loader_install=installed,loader_remove=c.L.remove,
                        ownership_verify=lambda owner,config:c.inert.verify(owner,config),execute=execute,environment=c.environment)
                self.assertEqual(len(execution),1)
                proof={'original_context':c.context,'pipeline_proof':execution[0]}
            else:proof=c.run_main()
        self.assertTrue((c.output/'LIBRARY_BOUNDARY_EVIDENCE.json').is_file())
        return proof
    def audit(self,mutate=None,fail_framework=False):
        c=self.c;proof=self.produce(fail_framework);context=proof['original_context']
        S=compiled(self.f.package/'prefix_input_owned_supervisor.py')
        A=compiled(self.f.package/'prefix_input_owned_auditor.py')
        F=compiled(HERE/'full_native_cpu_fixture_support.py')
        frame=F.materialize_full_native_CPU_fixture(self.f,c.output,context=context,
            topology=proof['pipeline_proof']['binding']['topology_observations'][0],ownership=c.owner)
        target=self.f.package/'evidence/CPU_LIBRARY_TERMINAL_PACKET'
        def revalidate():
            actual=self.C._native_manifest_value(self.f.config_path,**self.f.anchors)
            value=self.C.context_value(context)
            for key in ('fixture_only','native_PASS','native_authority_issued','status'):value.pop(key,None)
            self.assertEqual(actual,value);return actual
        with self.f.activate():
            receipt=S._CPU_preserve_native_fixture(frame['raw'],target,context,contract=self.C,
                _capability=self.C._mint_CPU_native_context_fixture(),death_verifier=lambda _:False,revalidate=revalidate)
            if mutate is not None:mutate(frame,target)
            raw=(target/'PRESERVATION.json').read_bytes()
            result=A._CPU_integrated_native_audit_fixture(target,original_directory=frame['raw'],contract=self.C,
                context=context,_capability=self.C._mint_CPU_native_context_fixture(),death_verifier=lambda _:False,
                expected_preservation_file_sha256=hashlib.sha256(raw).hexdigest(),expected_preservation_body_sha256=self.C.parse(raw)['sha256'])
        return result
    def test_actual_worker_packet_is_preserved_and_independently_consumed(self):
        result=self.audit()
        self.assertEqual(result['status'],'CPU_INTEGRATED_SOURCE_WORKFLOW_COMPLETE')
        self.assertTrue(result['library_terminal_evidence_join_complete'])
        evidence=result['library_evidence'];self.assertTrue(evidence['actual_packet_bytes_consumed'])
        self.assertTrue(evidence['conservation_complete'])
        self.assertEqual(evidence['replay']['reached_counts']['entered'],1)
        self.assertFalse(result['native_PASS']);self.assertFalse(result['native_authority_issued'])
        self.assertFalse(evidence['native_observer_qualified'])
        self.assertTrue(evidence['original_owned_clock_window_joined'])
    def test_actual_interpreter_child_birth_is_joined_in_terminal_population(self):
        from test_worker_bindings import InertSuppliers
        child=InertSuppliers(self.f,self.c.owner,interpreter_child=True);child.torch=self.c.inert.torch
        original=child.framework
        def framework():
            self.assertEqual(self.c.installed.library_guard._depth,1)
            self.c.library_supplier.add(self.c.fixture_case.extension);return original()
        child.framework=framework;self.c.inert=child
        result=self.audit();born=result['library_evidence']['replay']['original_identity']['owned_birth']
        self.assertEqual(born['pid'],12346)
        self.assertNotEqual(born['pid'],self.c.owner['worker_pid'])
        self.assertIn(born,result['owned_process_identities'])
    def test_actual_failed_framework_packet_and_core_failure_are_preserved(self):
        result=self.audit(fail_framework=True)
        self.assertEqual(result['status'],'CPU_INTEGRATED_SOURCE_FAILURE_PRESERVED')
        self.assertTrue(result['library_terminal_evidence_join_complete'])
        evidence=result['library_evidence'];self.assertFalse(evidence['conservation_complete'])
        self.assertFalse(evidence['replay']['conservation_complete'])
        self.assertTrue(evidence['replay']['failure_records'])
        self.assertFalse(result['native_PASS'])
    def test_resealed_entire_journal_shift_outside_owned_window_refused(self):
        # Every relative/event/identity check still agrees after this shift.
        # Original supervised clocks independently reject the false chronology.
        def mutate(frame,target):
            for root in (frame['raw'],target):
                path=root/'LIBRARY_BOUNDARY_EVIDENCE.json';env=self.C.parse(path.read_bytes())
                packet=env['raw_packet']
                for row in packet['observations']:
                    if type(row['time_ns']) is int:row['time_ns']+=10**12
                packet['sha256']=self.C.digest({k:v for k,v in packet.items() if k!='sha256'})
                env['raw_packet_sha256']=self.C.digest(packet);env['sha256']=self.C.digest({k:v for k,v in env.items() if k!='sha256'})
                path.write_bytes((self.C.canonical(env)+'\n').encode('ascii'))
            # Re-sign the preservation and SUP packet denominator, preserving
            # all original supervised clocks and resource/body evidence.
            name='LIBRARY_BOUNDARY_EVIDENCE.json';raw=(frame['raw']/name).read_bytes()
            descriptor={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            for root in (frame['raw'],target):
                path=root/'SUPERVISOR_RECEIPT.json';sup=self.C.parse(path.read_bytes())
                for row in sup['partial_outputs']:
                    if row['path']==name:row.update(descriptor)
                path.write_bytes((self.C.canonical(sup)+'\n').encode('ascii'))
            sup_raw=(frame['raw']/'SUPERVISOR_RECEIPT.json').read_bytes()
            (frame['lease_root']/'terminal.json').write_bytes(sup_raw);(target/'lease_terminal.json').write_bytes(sup_raw)
            path=target/'PRESERVATION.json';pres=self.C.parse(path.read_bytes())
            pres['raw_files'][name]=descriptor;pres['raw_files']['SUPERVISOR_RECEIPT.json']={'bytes':len(sup_raw),'sha256':hashlib.sha256(sup_raw).hexdigest()}
            pres['lease_files']['terminal.json'].update(bytes=len(sup_raw),sha256=hashlib.sha256(sup_raw).hexdigest())
            pres['current_death_observation']['supervisor_file_sha256']=hashlib.sha256(sup_raw).hexdigest()
            pres['sha256']=self.C.digest({k:v for k,v in pres.items() if k!='sha256'})
            path.write_bytes((self.C.canonical(pres)+'\n').encode('ascii'))
        with self.assertRaisesRegex(ValueError,'inside original owned resume/deadline'):
            self.audit(mutate)
    def test_packet_original_copy_drift_refused(self):
        def mutate(frame,target):
            f=frame['raw']/'LIBRARY_BOUNDARY_EVIDENCE.json';f.write_bytes(f.read_bytes()+b' ')
        with self.assertRaisesRegex(ValueError,'exact preserved bytes|source|byte|size|SHA|sha|hash'):self.audit(mutate)
    def test_selected_consumer_source_drift_refused(self):
        def mutate(frame,target):
            f=self.f.package/'prefix_library_evidence.py';raw=f.read_bytes();f.write_bytes(b'#'+raw[1:])
        with self.assertRaises(ValueError):self.audit(mutate)
    def test_same_count_event_rewrite_after_all_packet_reseals_refused(self):
        # Re-signing copied packets cannot change independent event/plan joins.
        proof=self.produce();context=proof['original_context'];A=compiled(self.f.package/'prefix_input_owned_auditor.py')
        raw=(self.c.output/'LIBRARY_BOUNDARY_EVIDENCE.json').read_bytes();envelope=self.C.parse(raw)
        for row in envelope['raw_packet']['observations']:
            trace=row['event_trace']
            if trace and trace['events']:trace['events'][0]['module_handle']+=1000
        packet=envelope['raw_packet'];packet['sha256']=self.C.digest({k:v for k,v in packet.items() if k!='sha256'})
        envelope['raw_packet_sha256']=self.C.digest(packet);envelope['sha256']=self.C.digest({k:v for k,v in envelope.items() if k!='sha256'})
        value=self.C.context_value(context)
        reader=A.Reader(self.c.output)
        birth=packet['identity']['owned_birth']
        with self.assertRaises(ValueError):A._audit_library_evidence_bytes((self.C.canonical(envelope)+'\n').encode('ascii'),
            selected_context=value,owner_receipt=self.c.owner,executing_birth=birth,reader=reader)

if __name__=='__main__':unittest.main()
