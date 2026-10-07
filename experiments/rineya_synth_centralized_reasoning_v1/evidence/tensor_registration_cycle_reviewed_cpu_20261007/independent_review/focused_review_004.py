"""Independent real-Git semantic checks; native and full metadata stay untested."""
from pathlib import Path
import hashlib, json, runpy, sys, unittest
HERE = Path(__file__).resolve().parent
WORK = HERE.parent
P = WORK.parents[1]
FILES = ('registration_git_join.py', 'prefix_tensor_contract.py', 'test_registration_git_join.py')
before = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before == {'registration_git_join.py': '2aa39c31be453c9abea135f0486cb61dcaa71c630a417cfbb02f6315e9ed3b30', 'prefix_tensor_contract.py': '35b4fccdbc253767c31a2039ef1f17dcc04f19015c9489c820a1fdb482dfc2b2', 'test_registration_git_join.py': 'de268ff442286981d3b9d80d4c17b5ae79d3a8c745ec2a9cd94caf0d586077f2'}
ns = runpy.run_path(str(WORK / 'test_registration_git_join.py'), run_name='independent_real_git_fixture')
J, Base = ns['J'], ns['Controls']
class IndependentControls(Base):
    def rewrite(self, name, value):
        q = self.root / (name + '.json')
        raw = (json.dumps(value, sort_keys=True) + '\n').encode('utf-8')
        q.write_bytes(raw)
        self.packets[name] = {'path': str(q), 'file_sha256': hashlib.sha256(raw).hexdigest(), 'body_sha256': value['sha256']}
    def test_actual_two_commit_positive_and_distinct_external_head(self):
        result = self.join()
        self.assertEqual(result['implementation_commit'], self.implementation)
        self.assertEqual(result['dispatch_head'], self.head)
        self.assertNotEqual(self.head, self.implementation)
        self.assertEqual(result['packet_bodies'], {name: self.packets[name]['body_sha256'] for name in self.packets})
        self.assertFalse(result['native_runtime_qualified'])
        with self.assertRaises(ValueError): self.join(expected_head=self.implementation)
    def test_duplicate_and_nonfinite_numeric_parse_refused(self):
        for raw in (b'{"nested":{"x":1,"x":2}}', b'{"x":1e999}', b'{"x":-1e999}', b'{"x":Infinity}', b'{"x":NaN}'):
            with self.assertRaises(ValueError): J.parse(raw)
    def test_executable_output_exception_and_runtime_map_rebound_refused(self):
        output = self.root / 'script.py'; output.write_bytes(b'# authored never executed\n')
        self.exceptions[str(output)] = {'role': 'run_output'}
        with self.assertRaises(ValueError): self.join()
        output.unlink(); self.exceptions = {}
        config = self.seal({**{k: v for k, v in self.config.items() if k != 'sha256'}, 'runtime_files': {'unmatched': {}}})
        registration = self.seal({**{k: v for k, v in self.registration.items() if k != 'sha256'}, 'config_body_sha256': config['sha256']})
        self.rewrite('config', config); self.rewrite('registration', registration)
        self.git('add', 'config.json', 'registration.json'); self.git('commit', '-qm', 'independent mismatched runtime map')
        self.head = self.git('rev-parse', 'HEAD').decode().strip()
        with self.assertRaisesRegex(ValueError, 'same selected runtime map'): self.join()
    def test_current_source_at_later_commit_refused_by_ancestor_blob_size(self):
        self.source.write_bytes(b'# authored changed current source of a different length\n')
        raw = self.source.read_bytes(); self.sources[str(self.source)] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest(), 'role': 'python_source'}
        config = self.seal({**{k: v for k, v in self.config.items() if k != 'sha256'}, 'source_files': self.sources})
        registration = self.seal({**{k: v for k, v in self.registration.items() if k != 'sha256'}, 'source_files': self.sources, 'config_body_sha256': config['sha256']})
        self.rewrite('config', config); self.rewrite('registration', registration)
        self.git('add', 'implementation.py', 'config.json', 'registration.json'); self.git('commit', '-qm', 'independent implementation bytes changed after ancestor')
        self.head = self.git('rev-parse', 'HEAD').decode().strip()
        with self.assertRaisesRegex(ValueError, 'source immutable Git blob exact bounded size before capture'): self.join()

METHODS = ('test_actual_two_commit_positive_and_distinct_external_head', 'test_duplicate_and_nonfinite_numeric_parse_refused', 'test_executable_output_exception_and_runtime_map_rebound_refused', 'test_current_source_at_later_commit_refused_by_ancestor_blob_size')
result = unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite(IndependentControls(name) for name in METHODS))
after = {name: hashlib.sha256((WORK / name).read_bytes()).hexdigest() for name in FILES}
assert before == after
public_sha = hashlib.sha256((P / 'prefix_tensor_contract.py').read_bytes()).hexdigest()
assert public_sha == '93df1a1c1c7eb78a0e3bb0b5f728e140fcc19021bd2980a9680ee78c3760c3d2'
receipt = {'kind': 'INDEPENDENT_CPU_REGISTRATION_GIT_JOIN_REVIEW_004', 'tests_run': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'source_before': before, 'source_after': after, 'public_tensor_contract_sha256': public_sha, 'real_temporary_git_commits_exercised': True, 'fixed_point_removed_in_pure_Git_fixture': True, 'strict_duplicate_and_nonfinite_parser_controls': True, 'source_ancestor_blob_size_gate_exercised': True, 'executable_output_exception_refused': True, 'full_amended_metadata_executed': False, 'full_amended_metadata_source_inspected_only': True, 'native_integration_implemented': False, 'site_disabled': sys.flags.no_site == 1, 'model_runtime_native_read': False, 'native_PASS': False, 'native_runtime_qualified': False, 'native_authority_issued': False, 'prior_parser_finding': {'helper_sha256': '2e31d7758d0ed1336786d31f6d867f9fc9def95b5dd970785bab625b7184f2f1', 'tool_chunk_id': 'd868dc', 'overflow_float_parse_is_nonfinite': True, 'later_digest_refused': True, 'whole_join_bypass_demonstrated': False, 'exact_predecessor_preserved_by_builder': 'pre_float_review003'}}
receipt['sha256'] = J.digest(receipt)
out = HERE / 'INDEPENDENT_REVIEW_004.json'
with out.open('x', encoding='utf-8') as f: json.dump(receipt, f, sort_keys=True, indent=2); f.write('\n')
print(json.dumps({'file_sha256': hashlib.sha256(out.read_bytes()).hexdigest(), 'body_sha256': receipt['sha256'], 'tests': result.testsRun, 'failures': len(result.failures), 'errors': len(result.errors), 'path': str(out)}, sort_keys=True))
raise SystemExit(0 if result.wasSuccessful() else 1)
