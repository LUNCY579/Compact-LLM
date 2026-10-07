"""Independent closed source packet checks; no bootstrap/native operation runs."""
from pathlib import Path
import ast, copy, hashlib, json, runpy, sys, unittest
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
PARENT = WORK.parents[1]
FILES = ('prefix_tensor_contract.py', 'prefix_input_owned_worker.py', 'prefix_native_pipeline.py', 'prefix_registered_loader.py', 'prefix_input_owned_supervisor.py', 'prefix_input_owned_auditor.py', 'registration_git_join.py', 'test_closed_packet.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before['prefix_tensor_contract.py'] == '2eb80f8efb8a11da790f6b9e7ec0f4bad8088a9b63571eed4383fa050fe03f0a'
assert before['prefix_input_owned_worker.py'] == '0fb8844d656252205a4297f36cb1e67738763ba3034e2be5b1b18c4c1640059b'
assert before['prefix_native_pipeline.py'] == '6ddc06ee6c3db26ad33b5ff7bfc6b334413cd14cce9f9eff3915753503e522a1'
assert before['test_closed_packet.py'] == '004284cae04f9d8ebf1ec192c662fb3d64664dfcee696ae2b8dafc8ae182c645'
packet_raw = (WORK / 'PROSPECTIVE_PACKET_003.json').read_bytes()
assert hashlib.sha256(packet_raw).hexdigest() == '6c385020c03b5814505c96039a316edbe70057f749f2290dd4e8be7675a05348'
ns = runpy.run_path(str(WORK / 'test_closed_packet.py'), run_name='independent_closed_packet_fixture')
C, W, PIPE, Base = ns['C'], ns['W'], ns['P'], ns['Controls']
class IndependentControls(Base):
    def setUp(self): self.packet = json.loads(packet_raw)
    def test_current_source_graph_and_pending_packet_return_no_context_or_permit(self):
        result = C.validate_prospective_packet(self.packet)
        self.assertFalse(result['native_launch_authorized']); self.assertFalse(result['native_authority_issued'])
        self.assertEqual(result['pending_observation_count'], 11)
        self.assertTrue(all(value is None for value in self.packet['pending'].values()))
        self.assertEqual(self.packet['source_candidates']['prefix_tensor_contract.py']['sha256'], before['prefix_tensor_contract.py'])
        self.assertEqual(set(self.packet['helper_sources']), {str(WORK / 'registration_git_join.py')})
        self.assertEqual(PIPE.CONTRACT_BOOTSTRAP, {'bytes': (WORK / 'prefix_tensor_contract.py').stat().st_size, 'sha256': before['prefix_tensor_contract.py'], 'role': 'python_source'})
        self.assertIs(C._native_manifest_value, C._amended_manifest_value)
        with self.assertRaises(ValueError): C.context_value(result)
    def test_missing_helper_and_resealed_descriptor_authority_or_pending_overrides_refused(self):
        bad = copy.deepcopy(self.packet); bad['helper_sources'] = {}; self.seal(bad)
        with self.assertRaises(ValueError): C.validate_prospective_packet(bad)
        bad = copy.deepcopy(self.packet); bad['helper_sources'][next(iter(bad['helper_sources']))]['sha256'] = '0' * 64; self.seal(bad)
        with self.assertRaises(ValueError): C.validate_prospective_packet(bad)
        for group, key, value in [('source_candidates', 'prefix_input_owned_worker.py', 'source'), ('runtime_candidate', None, 'runtime'), ('pending', next(iter(self.packet['pending'])), 'pending')]:
            bad = copy.deepcopy(self.packet)
            if value == 'source': bad[group][key]['sha256'] = '0' * 64
            elif value == 'runtime': bad[group]['files'] = {}
            else: bad[group][key] = True
            self.seal(bad)
            with self.assertRaises(ValueError): C.validate_prospective_packet(bad)
        for field in ('native_launch_authorized', 'native_authority_issued', 'model_fit_established', 'training_or_transfer_result'):
            bad = copy.deepcopy(self.packet); bad[field] = True; self.seal(bad)
            with self.assertRaises(ValueError): C.validate_prospective_packet(bad)
    def test_actual_selected_argv_and_root_cleanup_refuse_foreign_mutation(self):
        self.test_actual_source_argv_adaptation_and_owned_root_restoration()
        original = sys.path; before_path = tuple(original)
        try:
            original.append('CPU_OWNED_REGISTERED_ROOT')
            transition = {'original_container': original, 'before': before_path, 'path_after': list(original)}
            original.append('CPU_FOREIGN_ADDITION')
            with self.assertRaises(ValueError): W._restore_registered_roots(transition)
            self.assertEqual(original[-1], 'CPU_FOREIGN_ADDITION')
        finally: original[:] = before_path
    def test_native_setup_refuses_and_single_EOF_dispatch_routes_amended_reader(self):
        for operation in (lambda: C.native_entry(None), lambda: W.main(), lambda: W._site_disabled_bootstrap_setup(object()), lambda: W._registered_search_roots(object()), lambda: PIPE.supervise_registered(None, None)):
            with self.assertRaises(ValueError): operation()
        for name in ('prefix_tensor_contract.py', 'prefix_input_owned_worker.py'):
            tree = ast.parse((WORK / name).read_bytes())
            dispatches = [node for node in tree.body if isinstance(node, ast.If) and '__name__' in ast.unparse(node.test)]
            self.assertEqual(len(dispatches), 1); self.assertIs(tree.body[-1], dispatches[0])
        tree = ast.parse((WORK / 'prefix_tensor_contract.py').read_bytes())
        alias = next(node for node in tree.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '_native_manifest_value' for target in node.targets))
        self.assertEqual(ast.unparse(alias.value), '_amended_manifest_value')
        self.assertLess(alias.lineno, tree.body[-1].lineno)
        source = (WORK / 'prefix_native_pipeline.py').read_text()
        self.assertIn("'source_commit':value['expected_head']", source)
        self.assertFalse({'ctypes', '_ctypes', 'torch', 'psutil', 'native_worker', 'native_supervisor'} & set(sys.modules))

METHODS = ('test_current_source_graph_and_pending_packet_return_no_context_or_permit', 'test_missing_helper_and_resealed_descriptor_authority_or_pending_overrides_refused', 'test_actual_selected_argv_and_root_cleanup_refuse_foreign_mutation', 'test_native_setup_refuses_and_single_EOF_dispatch_routes_amended_reader')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}; assert before == after
assert (WORK / 'PROSPECTIVE_PACKET_003.json').read_bytes() == packet_raw
public_sha = hashlib.sha256((PARENT / 'prefix_tensor_contract.py').read_bytes()).hexdigest()
assert public_sha == '93df1a1c1c7eb78a0e3bb0b5f728e140fcc19021bd2980a9680ee78c3760c3d2'
receipt = {'kind': 'INDEPENDENT_CLOSED_SITE_DISABLED_TENSOR_SOURCE_PACKET_REVIEW_003', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'packet_file_sha256': hashlib.sha256(packet_raw).hexdigest(), 'packet_body_sha256': json.loads(packet_raw)['sha256'], 'public_tensor_contract_sha256': public_sha, 'exact_helper_singleton_omission_and_rebinding_refused': True, 'pending_fields_retained_unobserved': 11, 'actual_selected_original_argv_statements_executed_inert': True, 'owned_root_restoration_and_foreign_mutation_refusal_exercised': True, 'amended_reader_alias_and_single_EOF_dispatch_source_checked': True, 'prospective_result_not_context_or_permit': True, 'preloader_or_native_setup_executed': False, 'ctypes_DLL_Torch_native_API_or_site_hook_called': False, 'site_disabled': sys.flags.no_site == 1, 'current_tensor_bootstrap_cache_baseline_qualified': False, 'native_launch_authorized': False, 'native_PASS': False, 'native_authority_issued': False, 'prior_missing_helper_finding': {'source_sha256': 'd8f4387a5c3775fe09c95bbf72d4ddf63bcb770a6fe5f98912ffe5af10a832d3', 'tool_chunk_id': '877761', 'resealed_empty_helper_map_accepted': True, 'native_authority_bypass_demonstrated': False, 'predecessor_retained_by_builder': True}}
receipt['sha256'] = hashlib.sha256(json.dumps(receipt, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False).encode('ascii')).hexdigest()
out = HERE / 'INDEPENDENT_REVIEW_003.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
