"""CPU outer context controls: actual inert bytes, hashes and finite rosters."""
from pathlib import Path
import tempfile, unittest
from unittest.mock import patch
from outer_context_fixture import OuterContextFixture, HERE, source

class OuterNativeWorkflowControls(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix='CPU_outer_',dir=HERE)
        self.root=Path(self.temp.name).resolve(); self.fixture=OuterContextFixture(self.root); self.C=self.fixture.C
    def tearDown(self):
        self.assertEqual(self.root.parent,HERE); self.temp.cleanup()
    def test_actual_selected_documents_source_rosters_and_context(self):
        context=self.fixture.context(); v=self.C.context_value(context)
        self.assertTrue(v['fixture_only']); self.assertFalse(v['native_PASS']); self.assertFalse(v['native_authority_issued'])
        c=self.fixture.coverage()
        self.assertEqual(set(c['real_descriptor_paths']),set(self.fixture.sources)|set(self.fixture.inert_files))
        self.assertEqual(set(c['injected_historical_runtime_paths']),set(self.fixture.historical_runtime))
        self.assertEqual(len(self.fixture.git_calls),3); self.assertFalse(c['native_runtime_byte_closure_proven'])
        v['config']['scope']='changed'; self.assertEqual(self.C.context_value(context)['config']['scope'],self.C.NATIVE_SCOPE)
        with self.assertRaises(ValueError): source('prefix_tensor_contract.py').context_value(context)
    def test_corrupted_selected_config_rejected_by_actual_file_anchor(self):
        with self.fixture.config_path.open('ab') as f: f.write(b' ')
        with self.assertRaisesRegex(ValueError,'external file anchor'): self.fixture.context()
    def test_corrupted_bootstrap_and_review_rejected_by_actual_reader(self):
        for n in ('bootstrap_path','review_path'):
            p=getattr(self.fixture,n); raw=p.read_bytes(); p.write_bytes(raw+b' ')
            with self.assertRaisesRegex(ValueError,'external file anchor'): self.fixture.context()
            p.write_bytes(raw)
    def test_source_same_size_mutation_rejected_by_actual_streamed_hash(self):
        p=self.fixture.package/'prefix_input_owned_worker.py'; raw=p.read_bytes(); p.write_bytes(b'#'+raw[1:])
        with self.assertRaisesRegex(ValueError,'registered runtime exact stable bytes'): self.fixture.context()
    def test_unregistered_source_and_runtime_roster_additions_rejected(self):
        p=self.fixture.package/'unregistered.py'; p.write_bytes(b'# added source\n')
        with self.assertRaisesRegex(ValueError,'source roster'): self.fixture.context()
        p.unlink(); p=self.fixture.site/'torch/unregistered.py'; p.write_bytes(b'# added runtime\n')
        with self.assertRaisesRegex(ValueError,'physical runtime roster'): self.fixture.context()
    def test_inert_binary_bytes_hashed_without_loading(self):
        p=self.fixture.base/'DLLs/CPU_inert.pyd'; raw=p.read_bytes(); p.write_bytes(b'X'+raw[1:])
        with self.assertRaisesRegex(ValueError,'registered runtime exact stable bytes'): self.fixture.context()
    def test_capability_and_public_native_consumers_refuse_fixture(self):
        with self.fixture.activate():
            with self.assertRaisesRegex(ValueError,'private CPU'):
                self.C._CPU_native_context_fixture(self.fixture.config_path,_capability=object(),**self.fixture.anchors)
        context=self.fixture.context(); loader=source('prefix_registered_loader.py')
        with patch.object(self.C,'stage_native_gate'),patch.object(loader,'_install_body',side_effect=AssertionError('must not install')):
            with self.assertRaisesRegex(ValueError,'no native loader authority'): loader.install(context,self.C)
        with patch.object(self.C,'stage_native_gate'):
            with self.assertRaisesRegex(ValueError,'no native consumer authority'): self.C.revalidate_context(context)
        worker=source('prefix_input_owned_worker.py')
        with patch.object(worker,'C',self.C),patch.object(self.C,'stage_native_gate'):
            with self.assertRaisesRegex(ValueError,'cannot create real native bindings'):
                worker.real_torch_bindings(context,{'pid':1,'creation_filetime_100ns':1},self.root)

if __name__=='__main__': unittest.main()
