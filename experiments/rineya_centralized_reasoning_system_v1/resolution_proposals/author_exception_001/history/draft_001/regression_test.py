"""Bounded proposal regression only. Frozen candidates are never modified."""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

PROPOSAL = Path(__file__).resolve().parent
PACKAGE = PROPOSAL.parents[1]
REPOSITORY = PACKAGE.parents[1]
sys.path.insert(0, str(REPOSITORY))

from experiments.rineya_centralized_reasoning_system_v1 import reader
from experiments.rineya_centralized_reasoning_system_v1 import reader_execution as execution
from experiments.rineya_centralized_reasoning_system_v1.tests.test_reader import artifact_fixture

OBSERVATIONS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proposed_texts(patch_path):
    """Apply a reviewed unified diff in memory, checking every old context byte."""
    lines = Path(patch_path).read_bytes().decode().splitlines(keepends=True)
    result, index = {}, 0
    while index < len(lines):
        assert lines[index].startswith('--- a/'), 'unexpected diff header'
        relative = lines[index].strip()[6:]
        assert lines[index + 1].strip() == '+++ b/' + relative
        source = (REPOSITORY / relative).resolve()
        assert source.is_relative_to(PACKAGE)
        original = source.read_bytes().decode().splitlines(keepends=True)
        index += 2; cursor = 0; output = []
        while index < len(lines) and not lines[index].startswith('--- a/'):
            header = re.fullmatch(r'@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@.*\n', lines[index])
            assert header, 'unexpected hunk header'
            start, old_count = int(header[1]) - 1, int(header[2] or 1)
            output.extend(original[cursor:start]); cursor = start; consumed = 0
            index += 1
            while index < len(lines) and not lines[index].startswith(('@@ ', '--- a/')):
                line = lines[index]; prefix, content = line[0], line[1:]
                assert prefix in ' +-'
                if prefix in ' -':
                    assert original[cursor] == content, 'frozen context mismatch'
                    cursor += 1; consumed += 1
                if prefix in ' +': output.append(content)
                index += 1
            assert consumed == old_count
        output.extend(original[cursor:]); result[relative] = ''.join(output)
    return result


class ProposalRegression(unittest.TestCase):
    def test_actual_pinned_psutil_path_refusal_and_string_success(self):
        import psutil
        self.assertEqual(os.name, 'nt')
        self.assertEqual(psutil.__version__, '7.2.2')
        with tempfile.TemporaryDirectory(dir=PROPOSAL / 'outputs') as temporary:
            path = Path(temporary)
            with self.assertRaises(TypeError) as caught:
                psutil.disk_usage(path)
            actual = psutil.disk_usage(str(path))
            self.assertGreater(actual.total, 0)
            self.assertTrue(0 <= actual.free <= actual.total)
            self.assertTrue(math.isfinite(actual.percent))
            source = proposed_texts(PROPOSAL / 'probe_disk_usage.patch')
            updated = source['experiments/rineya_centralized_reasoning_system_v1/owned_runtime_identity_probe.py']
            self.assertIn('psutil.disk_usage(str(self.output))', updated)
            self.assertIn('psutil.disk_usage(str(output))', updated)
            self.assertNotIn('psutil.disk_usage(self.output)', updated)
            self.assertNotIn('psutil.disk_usage(output)', updated)
            OBSERVATIONS['pinned_api'] = {
                'psutil_version': psutil.__version__, 'module_path': str(Path(psutil.__file__).resolve()),
                'module_sha256': sha(psutil.__file__), 'path_negative_error': str(caught.exception),
                'actual_string_disk_usage_succeeded': True,
                'disk_result_used_for_host_admission': False,
            }

    def test_actual_writer_ref_and_proposed_validator_byte_binding(self):
        cfg = reader.read_json(reader.CONTRACT)
        texts = proposed_texts(PROPOSAL / 'reader_supervisor_bytes.patch')
        updated = texts['experiments/rineya_centralized_reasoning_system_v1/reader_execution.py']
        tree = ast.parse(updated)
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'validate_native_artifacts')
        with tempfile.TemporaryDirectory(dir=PROPOSAL / 'outputs') as temporary:
            package = Path(temporary); (package / 'results').mkdir()
            with patch.object(reader, 'PACKAGE', package), patch.object(execution, 'PACKAGE', package), \
                    patch.object(execution, 'host_state_path', return_value=package / 'NEVER_OPENED.sqlite3'):
                binding, contents = artifact_fixture(cfg)
                self.assertTrue(execution.validate_native_artifacts(contents, cfg, binding))
                output = Path(contents['go']['run_identity']['registration_payload']['output']); output.mkdir()
                path = output / 'SUPERVISOR_RESULT.json'
                execution.write_new(path, contents['supervisor_result'])  # Actual unchanged native writer.
                actual_bytes = path.read_bytes(); object_sha = reader.digest(contents['supervisor_result'])
                actual_sha = sha(path)
                self.assertTrue(actual_bytes.endswith(b'\n'))
                self.assertNotEqual(actual_sha, object_sha)
                contents['job_outcome']['job']['result']['evidence']['supervisor_receipt_sha256'] = actual_sha
                with self.assertRaises(reader.ReaderBlocked) as original_error:
                    execution.validate_native_artifacts(contents, cfg, binding)
                namespace = dict(vars(execution)); namespace['PACKAGE'] = package
                exec(compile(ast.Module(body=[function], type_ignores=[]), '<proposed-diff-in-memory-only>', 'exec'), namespace)
                proposed = namespace['validate_native_artifacts']
                refs = {'supervisor_result': {'path': str(path.resolve()), 'sha256': actual_sha}}
                self.assertTrue(proposed(contents, cfg, binding, refs))
                path.write_bytes(actual_bytes + b' ')  # Same parsed object, different immutable bytes.
                self.assertEqual(reader.read_json(path), contents['supervisor_result'])
                with self.assertRaises(reader.ReaderBlocked): proposed(contents, cfg, binding, refs)
                refs['supervisor_result']['sha256'] = sha(path)
                with self.assertRaises(reader.ReaderBlocked): proposed(contents, cfg, binding, refs)
                OBSERVATIONS['actual_writer'] = {
                    'producer': 'unchanged reader_execution.write_new/native.write_new',
                    'actual_trailing_newline': True, 'actual_file_sha256': actual_sha,
                    'parsed_object_digest': object_sha, 'frozen_validator_error': str(original_error.exception),
                    'proposed_in_memory_validator_accepts_actual_byte_reference': True,
                    'same_object_byte_drift_rejected': True, 'reference_update_without_terminal_evidence_update_rejected': True,
                    'protocol_fixture_class': 'CPU_SYNTHETIC_INTERFACE_NOT_NATIVE',
                    'native_qualification_published': False,
                }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, default=PROPOSAL / 'outputs/CPU_REGRESSION_RESULT_001.json')
    args = parser.parse_args()
    outputs = PROPOSAL / 'outputs'; outputs.mkdir(exist_ok=True)
    report = args.report.resolve()
    assert report.is_relative_to(outputs.resolve()) and not report.exists()
    manifest = reader.read_json(PROPOSAL / 'PATCH_MANIFEST.json')
    for name, expected in manifest['frozen_sources'].items(): assert sha(PACKAGE / name) == expected
    for name, expected in manifest['patch_sha256'].items(): assert sha(PROPOSAL / name) == expected
    proposed = {name: value for patch_name in manifest['patch_sha256'] for name, value in proposed_texts(PROPOSAL / patch_name).items()}
    for name, expected in manifest['proposed_source_sha256'].items():
        value = proposed['experiments/rineya_centralized_reasoning_system_v1/' + name]
        assert hashlib.sha256(value.encode()).hexdigest() == expected
    stream = io.StringIO(); started = time.perf_counter()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(ProposalRegression))
    seconds = time.perf_counter() - started
    for name, expected in manifest['frozen_sources'].items(): assert sha(PACKAGE / name) == expected
    assert not ({'torch', 'transformers', 'tokenizers', 'safetensors'} & set(sys.modules))
    payload = {'status': 'CPU_PROPOSED_DIFF_REGRESSION_ONLY_NOT_INTEGRATED', 'tests': result.testsRun,
               'passed': result.wasSuccessful(), 'seconds': seconds, 'output': stream.getvalue(), 'observations': OBSERVATIONS,
               'source_pins': manifest['frozen_sources'], 'patch_pins': manifest['patch_sha256'],
               'frozen_candidates_unchanged': True, 'framework_imports': 0, 'model_calls': 0, 'actual_probe_runs': 0,
               'canonical_host_database_access': False, 'owner_exception_granted': False,
               'native_or_reader_qualification': False, 'builder_independent_approval': False}
    with report.open('x', encoding='utf-8', newline='\n') as out: out.write(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'tests': result.testsRun, 'passed': result.wasSuccessful(), 'seconds': seconds,
                      'report': str(report), 'report_sha256': sha(report)}))
    if not result.wasSuccessful(): print(stream.getvalue())
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    raise SystemExit(main())
