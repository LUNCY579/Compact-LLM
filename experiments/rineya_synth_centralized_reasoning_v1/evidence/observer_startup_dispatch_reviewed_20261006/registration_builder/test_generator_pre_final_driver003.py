"""Closed JSON construction from exact metadata/source; no runtime binary IO."""
from copy import deepcopy
from pathlib import Path
import ast,builtins,tempfile,unittest
from unittest.mock import patch
import generator as G

class DispatchGeneratorControls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.data=G.inputs()
    def setUp(self):
        self.attempts=[];original=builtins.__import__
        def imports(name,*args,**kwargs):
            if name.split('.',1)[0] in {'torch','ctypes','psutil','numpy','native_worker','native_process','native_nvml','transformers','safetensors'}:
                self.attempts.append(name);raise AssertionError('native/framework import forbidden: '+name)
            return original(name,*args,**kwargs)
        trap=patch.object(builtins,'__import__',side_effect=imports);trap.start();self.addCleanup(trap.stop);self.addCleanup(lambda:self.assertEqual(self.attempts,[]))
    def packet(self,**kwargs):return G.construct(**deepcopy(self.data),**kwargs)
    def test_actual_closed_packet_exact_maps_no_bootstrap_baseline_or_approval(self):
        p=self.packet();self.assertEqual(len(p['config']['source_files']),17);self.assertEqual(len(p['runtime']['inherited65_map']),65);self.assertEqual(len(p['runtime']['stdlib779_map']),779)
        self.assertFalse(p['config']['native_launch_authorized']);self.assertFalse(p['registration']['owner_approval']['approved']);self.assertFalse(p['review']['complete_current_byte_review'])
        for key in ('source_promotion_complete','dispatch_blobs_committed','bootstrap_observed','loaded_baseline_observed','namespace_execution_proven','resource_qualification_complete','native_PASS'):self.assertFalse(p[key])
        self.assertIsNone(p['config']['implementation_commit']);self.assertIsNone(p['dispatch_head_external'])
    def test_two_commits_and_actual_startup_schema_CPU_construction(self):
        p=self.packet(implementation_commit='a'*40,dispatch_head='b'*40)
        root=G.PACKAGE;namespace={'Path':Path,'PACKAGE':root,'EXPECTED':root/'observer_startup_measurement_v1',**self.data['constants_value'],'require':G.require,'canonical':G.canonical,'digest':G.digest,'sha':G.sha}
        tree=ast.parse(G.read_source(G.STARTUP/'contract.py'));fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_schema')
        exec(compile(ast.Module(body=[fn],type_ignores=[]),str(G.STARTUP/'contract.py'),'exec'),namespace)
        value=namespace['_schema'](p['config'],p['runtime'],p['registration'],'b'*40,fixture_only=True)
        self.assertTrue(value['fixture_only']);self.assertFalse(value['native_authority_issued'])
        with self.assertRaisesRegex(ValueError,'distinct external'):self.packet(implementation_commit='a'*40,dispatch_head='a'*40)
    def test_source_same_count_substitution_or_omitted_source_refused(self):
        d=deepcopy(self.data);key=str(G.PACKAGE/'observer_startup_measurement_v1/worker.py');d['promoted_sources'][str(G.PACKAGE/'observer_startup_measurement_v1/other.py')]=d['promoted_sources'].pop(key)
        with self.assertRaisesRegex(ValueError,'five exact'):G.construct(**d)
        d=deepcopy(self.data);key=next(iter(d['build']['native_source_files']));d['promoted_sources'][key]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'build C/H'):G.construct(**d)
    def test_historical_or_measured_binary_substitution_refused(self):
        d=deepcopy(self.data);key=next(iter(d['passive']['inherited65_map']));d['passive']['files'][key]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'inherited pins'):G.construct(**d)
        d=deepcopy(self.data);key=next(k for k,v in d['passive']['files'].items() if v['role']=='observer_binary');d['passive']['files'][key]['sha256']='0'*64
        with self.assertRaisesRegex(ValueError,'existing build binary'):G.construct(**d)
        d=deepcopy(self.data);known=next(p for p in d['passive']['files'] if Path(p).parent==Path(r'C:\Windows\System32'));entry={k:d['passive']['files'][known][k] for k in ('bytes','sha256','role')};d['additional_windows'][known.upper()]={**entry,'sha256':'0'*64}
        with self.assertRaisesRegex(ValueError,'Windows descriptor conflict'):G.construct(**d)
    def test_authorized_mode_requires_exact_human_evidence_and_keeps_qualification_false(self):
        ap=G.pinned_json(G.HERE/'OWNER_APPROVAL_VERIFIED_001.json',G.APPROVAL_FILE,G.APPROVAL_BODY)
        p=self.packet(implementation_commit='a'*40,dispatch_head='b'*40,approval=ap,approved_evidence_reference='verified approval fixture reference')
        self.assertTrue(p['config']['native_launch_authorized']);self.assertTrue(p['registration']['owner_approval']['approved'])
        self.assertFalse(p['review']['complete_current_byte_review']);self.assertFalse(p['native_PASS']);self.assertFalse(p['bootstrap_observed'])
        for changed in ({'approved':True},{**ap,'sha256':'0'*64}):
            with self.assertRaisesRegex(ValueError,'verified human'):self.packet(implementation_commit='a'*40,dispatch_head='b'*40,approval=changed,approved_evidence_reference='fixture')
        with self.assertRaisesRegex(ValueError,'implementation commit'):self.packet(approval=ap,approved_evidence_reference='fixture')
        p=self.packet(implementation_commit='a'*40,approval=ap,approved_evidence_reference='verifiedfixture')
        self.assertTrue(p['config']['native_launch_authorized']);self.assertIsNone(p['dispatch_head_external']);self.assertFalse(p['dispatch_blobs_committed'])
    def test_byte_exact_build_receipt_and_prospective_policy_scope_retained(self):
        p=self.packet();b=p['runtime']['library_event_observer'];self.assertEqual(b['build_receipt']['file_sha256'],G.BUILD_FILE);self.assertEqual(b['build_receipt']['body_sha256'],G.BUILD_BODY)
        self.assertFalse(self.data['build']['source_inputs_committed']);self.assertEqual(p['runtime']['cache_policy']['imports'],['ctypes','ctypes.wintypes','_ctypes'])
        self.assertNotIn('initial_loaded_libraries',p['runtime']);self.assertNotIn('fresh_process_bootstrap_qualified',p['review']);self.assertNotIn('environment_and_loader_assumptions_qualified',p['review'])
    def test_exact_source_roles_and_actual_shared_metadata_checks_stop_at_false_review(self):
        p=self.packet(implementation_commit='a'*40)
        changed=[key for key,d in self.data['passive']['files'].items() if Path(key).parent.name=='DLLs' and Path(key).suffix.lower() in {'.pyd','.dll'}]
        self.assertTrue(changed)
        for key in changed:
            self.assertEqual(p['runtime']['files'][key]['role'],'extension_binary')
            self.assertEqual(p['runtime']['files'][key]['sha256'],self.data['passive']['files'][key]['sha256'])
        namespace={'__file__':str(G.PACKAGE/'observer_startup_measurement_v1/contract.py'),'__name__':'closed_generator_CPU_shared_checks'}
        exec(compile(G.read_source(G.PACKAGE/'observer_startup_measurement_v1/contract.py'),namespace['__file__'],'exec'),namespace)
        # Only the unpublished review JSON is supplied. Real role, original
        # metadata, observer C/H/build and bootstrap joins execute unchanged.
        namespace['_selected_json']=lambda path,file_sha,body_sha: p['review']
        with self.assertRaisesRegex(ValueError,'independent prospective byte/import/library admission review'):
            namespace['_shared_inventory_checks']({'config':p['config'],'runtime':p['runtime']})
        key=changed[0];p['runtime']['files'][key]['role']='interpreter_binary'
        with self.assertRaisesRegex(ValueError,'runtime role/root assignment'):
            namespace['_shared_inventory_checks']({'config':p['config'],'runtime':p['runtime']})
    def test_metadata_anchor_duplicate_nonfinite_and_binary_read_refusal(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'candidate.dll';p.write_bytes(b'inert fixture')
            with self.assertRaisesRegex(ValueError,'no binary reads'):G.read_source(p)
        for raw in (b'{"x":1,"x":2}',b'{"x":NaN}',b'{"x":1e999}'):
            with self.assertRaises(ValueError):G.parse(raw)
        with self.assertRaisesRegex(ValueError,'file pin'):G.pinned_json(G.PASSIVE,'0'*64,G.PASSIVE_BODY)

if __name__=='__main__':unittest.main()
