"""Small file-only qualification provenance regressions; no model dependencies."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from qualification_bytes import verify_qualification


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    data = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False)
    return hashlib.sha256(data.encode()).hexdigest()


class QualificationBytesTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.stage = Path(self.temporary.name).resolve() / 'stage'
        self.stage_id = 'a' * 64
        self.cell = 'XG_g0.03'
        self.sources = {'src/example.py': 'b' * 64}
        self.config_hash = 'c' * 64
        self.root = self.stage / 'evidence/01_runtime_qualification' / self.stage_id / self.cell
        self.committed_bytes = {}
        self.commit_checks = []

    def save(self, path, value, *, committed=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, sort_keys=True), encoding='utf-8')
        if committed:
            self.committed_bytes[path] = path.read_bytes()

    def commit_check(self, path):
        path = Path(path)
        self.commit_checks.append(path)
        if self.committed_bytes.get(path) != path.read_bytes():
            raise ValueError('synthetic artifact is not committed')

    def phase(self, phase='cpu', index=0, status='COMPLETE', failure_class=None):
        directory = self.root / phase / f'attempt_{index:03d}'
        started = {
            'schema_version': 'cross_scale_generation_cell_qualification_phase_v1',
            'track': 'cross_scale', 'purpose': 'runtime_qualification_only',
            'registration_sha256': self.stage_id, 'config_sha256': self.config_hash,
            'source_sha256': canonical(self.sources), 'sources': self.sources,
            'cell_id': self.cell, 'phase': phase, 'device': 'cpu' if phase == 'cpu' else 'cuda:0',
            'generation_role': 'calibration', 'attempt': index, 'status': 'STARTED',
            'start_utc': '2026-09-13T01:00:00+00:00', 'pid': 1234,
        }
        self.save(directory / 'started.json', started)
        # These arbitrary bytes deliberately are not a serialized tensor.
        binary = directory / 'raw/example.bin'
        binary.parent.mkdir()
        binary.write_bytes(b'\x00tiny raw artifact\xff')
        terminal = {**started, 'status': status}
        if status == 'COMPLETE':
            self.save(directory / 'report.json', {'cell_id': self.cell, 'fixture_count': 8})
            terminal['report_sha256'] = digest(directory / 'report.json')
        elif status == 'FAILED':
            self.save(directory / 'failure.json', {'message': 'synthetic failure'})
            terminal['failure_class'] = failure_class or 'cell_rejection'
        else:
            terminal['original_process_absent'] = True
        terminal['artifacts_sha256'] = {
            p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file()
        }
        filename = 'abandonment.json' if status == 'ABANDONED' else 'run_manifest.json'
        self.save(directory / filename, terminal)
        return directory, terminal

    def phase_result(self, directory, receipt):
        return {'status': 'FAILED', 'phase': 'qualification', 'directory': str(directory),
                'receipt_sha256': digest(directory / 'run_manifest.json'),
                'receipt': deepcopy(receipt), 'reason': 'synthetic failure'}

    def comparison(self, *, passed=True, prior_failures=False):
        if prior_failures:
            self.phase('cpu', 0, 'FAILED', 'resource_admission')
            self.phase('gpu', 0, 'FAILED', 'runtime_resource')
        index = 1 if prior_failures else 0
        cpu, _ = self.phase('cpu', index)
        gpu, _ = self.phase('gpu', index)
        receipt = {
            'status': 'PASS' if passed else 'FAILED', 'phase': 'qualification',
            'cell_id': self.cell, 'registration_sha256': self.stage_id,
            'config_sha256': self.config_hash,
            'cpu_receipt_sha256': digest(cpu / 'run_manifest.json'),
            'gpu_receipt_sha256': digest(gpu / 'run_manifest.json'),
            'result': {'pass': passed, 'comparisons': []},
        }
        path = self.root / 'comparison.json'
        self.save(path, receipt)
        return {'status': receipt['status'], 'phase': 'qualification', 'directory': str(self.root),
                'receipt_sha256': digest(path), 'receipt': deepcopy(receipt)}

    def verify(self, result, **overrides):
        arguments = {'expected_cell': self.cell, 'stage_root': self.stage,
                     'stage_id': self.stage_id, 'expected_sources': self.sources,
                     'expected_config_hash': self.config_hash,
                     'reader': lambda p: json.loads(Path(p).read_text(encoding='utf-8')),
                     'hasher': digest, 'committed_checker': self.commit_check}
        arguments.update(overrides)
        return verify_qualification(result, **arguments)

    def test_valid_hard_cpu_failure_binds_receipts_and_untracked_binary(self):
        directory, receipt = self.phase(status='FAILED')
        hashes = self.verify(self.phase_result(directory, receipt))
        expected = {str(p): digest(p) for p in directory.rglob('*') if p.is_file()}
        self.assertEqual(hashes, expected)
        self.assertTrue(all(Path(path).is_absolute() for path in hashes))
        self.assertIn(directory / 'failure.json', self.commit_checks)
        self.assertNotIn(directory / 'raw/example.bin', self.commit_checks)

    def test_valid_comparison_binds_both_phases(self):
        result = self.comparison()
        hashes = self.verify(result)
        self.assertEqual(hashes, {str(p): digest(p) for p in self.root.rglob('*') if p.is_file()})
        self.assertIn(self.root / 'comparison.json', self.commit_checks)
        for phase in ('cpu', 'gpu'):
            self.assertIn(self.root / phase / 'attempt_000/report.json', self.commit_checks)

    def test_valid_failed_comparison_is_preserved_without_gate_replay(self):
        result = self.comparison(passed=False)
        self.assertIn(str(self.root / 'comparison.json'), self.verify(result))

    def test_prior_operational_failure_artifacts_are_preserved(self):
        result = self.comparison(prior_failures=True)
        hashes = self.verify(result)
        for phase in ('cpu', 'gpu'):
            prior = self.root / phase / 'attempt_000'
            self.assertIn(str(prior / 'run_manifest.json'), hashes)
            self.assertIn(str(prior / 'raw/example.bin'), hashes)
            self.assertIn(prior / 'failure.json', self.commit_checks)

    def test_missing_comparison_receipt_is_rejected(self):
        result = self.comparison()
        (self.root / 'comparison.json').unlink()
        with self.assertRaisesRegex(ValueError, 'comparison receipt missing'):
            self.verify(result)

    def test_changed_saved_comparison_is_rejected_even_if_committed(self):
        result = self.comparison()
        changed = {**result['receipt'], 'created_utc': 'changed'}
        self.save(self.root / 'comparison.json', changed)
        with self.assertRaisesRegex(ValueError, 'comparison hash changed'):
            self.verify(result)

    def test_changed_embedded_receipt_is_rejected(self):
        result = self.comparison()
        result['receipt']['extra'] = 'not in the saved receipt'
        with self.assertRaisesRegex(ValueError, 'embedded comparison differs'):
            self.verify(result)

    def test_missing_phase_terminal_is_rejected(self):
        result = self.comparison()
        (self.root / 'cpu/attempt_000/run_manifest.json').unlink()
        with self.assertRaisesRegex(ValueError, 'unfinished or ambiguous phase terminal'):
            self.verify(result)

    def test_missing_artifact_is_rejected(self):
        result = self.comparison()
        (self.root / 'gpu/attempt_000/raw/example.bin').unlink()
        with self.assertRaisesRegex(ValueError, 'artifact inventory/hash changed'):
            self.verify(result)

    def test_changed_artifact_is_rejected(self):
        result = self.comparison()
        (self.root / 'gpu/attempt_000/raw/example.bin').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'artifact inventory/hash changed'):
            self.verify(result)

    def test_added_unlisted_file_is_rejected(self):
        result = self.comparison()
        (self.root / 'cpu/attempt_000/unlisted.bin').write_bytes(b'extra')
        with self.assertRaisesRegex(ValueError, 'artifact inventory/hash changed'):
            self.verify(result)

    def test_ambiguous_terminal_is_rejected(self):
        result = self.comparison()
        self.save(self.root / 'cpu/attempt_000/abandonment.json', {'status': 'ABANDONED'})
        with self.assertRaisesRegex(ValueError, 'unfinished or ambiguous phase terminal'):
            self.verify(result)

    def test_wrong_config_binding_is_rejected(self):
        result = self.comparison()
        with self.assertRaisesRegex(ValueError, 'phase contract changed'):
            self.verify(result, expected_config_hash='d' * 64)

    def test_wrong_source_binding_is_rejected(self):
        result = self.comparison()
        with self.assertRaisesRegex(ValueError, 'phase contract changed'):
            self.verify(result, expected_sources={'src/foreign.py': 'e' * 64})

    def test_uncommitted_receipt_is_rejected(self):
        result = self.comparison()
        del self.committed_bytes[self.root / 'cpu/attempt_000/report.json']
        with self.assertRaisesRegex(ValueError, 'not committed'):
            self.verify(result)


if __name__ == '__main__':
    unittest.main()
