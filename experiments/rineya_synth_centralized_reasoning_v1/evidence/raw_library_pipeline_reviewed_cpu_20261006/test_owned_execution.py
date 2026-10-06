"""Whole-path CPU controls of separate permission, owned lifecycle and cleanup."""
from pathlib import Path
import copy,importlib.util,tempfile,unittest
HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('owned_execution_fixture',HERE/'fixture_support.py');F=importlib.util.module_from_spec(spec);exec(compile((HERE/'fixture_support.py').read_text(encoding='utf-8'),str(HERE/'fixture_support.py'),'exec'),F.__dict__)
class OwnedExecutionControls(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='owned_fixture_',dir=HERE);self.root=Path(self.tmp.name).resolve();self.assertEqual(self.root.parent,HERE.resolve());self.addCleanup(self.tmp.cleanup)
    def test_complete_whole_path_original_checks_owned_order_and_flat_copy(self):
        frame=F.terminal_frame(self.root);terminal=frame['terminal'];self.assertEqual(terminal['status'],'CPU_OWNED_TERMINAL_COMPLETE')
        self.assertEqual([x['event'] for x in terminal['timeline']],list(F.C.TIMELINE));self.assertEqual(terminal['worker_report']['input_summary']['logical_tensor_bytes'],6752)
        self.assertEqual(F.F.check_reads(frame['api'].recorder),F.F.EXPECTED_CHECK_READS);self.assertFalse(terminal['native_authority_issued'])
        self.assertEqual(set(frame['preservation']['raw_files']),set(F.C.RAW_CORE)|{'OWNED_TERMINAL.json'})
    def test_permission_mutation_and_unknown_manifest_before_output(self):
        for edit in ('permission','unknown'):
            manifest=F.C.fixture_manifest(F.source_pins())
            if edit=='permission':manifest['permission']['allocations']=3
            else:manifest['unregistered_permission']=True
            manifest['sha256']=F.C.digest({k:v for k,v in manifest.items() if k!='sha256'});host=F.FakeHost(self.root/edit)
            with self.assertRaises(ValueError):F.S.supervise_injected(manifest,F.F.InjectedAPI(),host,self.root/edit)
            self.assertFalse(host.calls);self.assertFalse((self.root/edit).exists())
    def test_wrong_ownership_and_forged_factory_before_torch(self):
        frame=F.terminal_frame(self.root,host_faults={'membership':False});self.assertEqual(frame['terminal']['status'],'CPU_OWNED_TERMINAL_FAILED');self.assertNotIn('api.torch_import',frame['api'].recorder.events)
        manifest=frame['manifest'];host=F.FakeHost(self.root/'other');api=F.F.InjectedAPI()
        with self.assertRaises(ValueError):F.S.W.run_injected(manifest,object(),host.owner,api,host,self.root/'other')
        self.assertFalse(api.recorder.events)
    def test_actual_preflight_none_and_changed_return_fail(self):
        frame=F.terminal_frame(self.root,api_faults={'api.config_preflight:1':None});self.assertEqual(frame['terminal']['status'],'CPU_OWNED_TERMINAL_FAILED');self.assertNotIn('api.torch_import',frame['api'].recorder.events)
        other=self.root/'changed';other.mkdir();frame=F.terminal_frame(other,api_faults={'api.config_preflight:2':{'changed':True}})
        self.assertEqual(frame['terminal']['status'],'CPU_OWNED_TERMINAL_FAILED');self.assertNotIn('terminal_cuda_is_initialized',frame['terminal']['worker_report']['operation_accounting'])
    def test_predicate_shortcircuit_retains_primary_and_cleanup(self):
        frame=F.terminal_frame(self.root,api_faults={'ids.shape':(1,421)});self.assertEqual(frame['terminal']['status'],'CPU_OWNED_TERMINAL_FAILED');self.assertEqual(F.F.check_reads(frame['api'].recorder),['ids.is_tensor','ids.shape'])
        self.assertTrue(all(frame['terminal']['terminal_flags'].values()));self.assertEqual(frame['terminal']['worker_report']['error']['type'],'ValueError')
    def test_live_birth_and_cleanup_failures_never_complete(self):
        frame=F.terminal_frame(self.root,host_faults={'current_death':[{'identity':{'pid':12345,'creation_filetime_100ns':134000000000000001},'currently_alive':True}],'close_NVML':False})
        self.assertEqual(frame['terminal']['status'],'CPU_OWNED_TERMINAL_FAILED');self.assertFalse(frame['terminal']['terminal_flags']['NVML_closed']);self.assertTrue(frame['terminal']['secondary_errors'])
    def test_hostile_cleanup_error_does_not_replace_primary_or_skip_cleanup(self):
        class HostileError(RuntimeError):
            def __str__(self):raise RuntimeError('hostile message')
        frame=F.terminal_frame(self.root,api_faults={'torch.tensor':RuntimeError('primary')},host_faults={'join_samplers':HostileError(),'current_death':HostileError(),'resource_samples':HostileError()})
        self.assertEqual(frame['terminal']['worker_report']['error']['message'],'primary');self.assertEqual(frame['terminal']['status'],'CPU_OWNED_TERMINAL_FAILED')
        self.assertIn('close_lease',frame['host'].calls);self.assertTrue(any(x.get('error',{}).get('message_unavailable') for x in frame['terminal']['secondary_errors']))
    def test_native_stage_gate_before_hostile_arguments(self):
        class Hostile:
            def __getattribute__(self,name):raise AssertionError('native argument inspected')
        for fn in (F.C.native_entry,F.S.W.run_native,F.S.supervise_native):
            with self.assertRaises(ValueError):fn(Hostile())
if __name__=='__main__':unittest.main()
