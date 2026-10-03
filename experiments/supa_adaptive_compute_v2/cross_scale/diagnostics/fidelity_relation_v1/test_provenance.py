"""Small boundary tests for the separate descriptive report CLI."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import run_report as cli


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def put(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding='utf-8')
        return path

    def test_json_duplicate_and_nonfinite_rejected(self):
        path = self.root / 'bad.json'
        for value in ('{"x":1,"x":2}', '{"x":NaN}', '{"x":Infinity}'):
            with self.subTest(value=value):
                path.write_text(value)
                with self.assertRaises(ValueError):
                    cli.read(path)

    def test_exclusive_output(self):
        path = self.root / 'receipt.json'
        cli.write(path, {'status': 'COMPLETE'})
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            cli.write(path, {'status': 'REPLACED'})
        self.assertEqual(path.read_bytes(), original)

    def test_resolved_path_escape_rejected(self):
        with self.assertRaises(ValueError):
            cli.within(self.root / '..' / 'outside.json', self.root)
        self.assertEqual(cli.within(self.root / 'allowed.json', self.root), self.root / 'allowed.json')

    def test_commit_check_requires_exact_bytes(self):
        path = self.put('source.json', {'value': 1})
        with patch.object(cli, 'relative', return_value='source.json'):
            with patch.object(cli, 'git', return_value=path.read_bytes()):
                cli.committed(path)
            with patch.object(cli, 'git', return_value=b'{"value":2}'):
                with self.assertRaises(ValueError):
                    cli.committed(path)
            with self.assertRaises(ValueError):
                cli.committed(path, 'unknown-ref')

    def test_freeze_refuses_any_existing_generation_start(self):
        generation = self.root / 'generation'
        self.put('generation/X0/shard_000/attempt_000/started.json', {})
        with patch.object(cli, 'GENERATION', generation), patch.object(cli, 'registration_contract') as contract:
            with self.assertRaises(ValueError):
                cli.freeze()
            contract.assert_not_called()

    def test_start_must_contain_committed_diagnostic_and_postdate_freeze(self):
        diagnostic = {'frozen_utc': '2026-09-14T05:00:00+00:00',
                      'contract': {'sources': {'diagnostic/report_core.py': 'unused'}}}
        start = {'registration_sha256': cli.STAGE_ID, 'role': 'architecture_dev',
                 'start_utc': '2026-09-14T05:01:00+00:00', 'commit_sha': 'a' * 40}
        path = self.put('started.json', start)
        registration_path = self.root / 'registration.json'
        with patch.object(cli, 'committed') as committed:
            cli.prospective_start(path, registration_path, diagnostic)
            self.assertEqual(committed.call_count, 2)
            self.assertTrue(all(call.args[1] == 'a' * 40 for call in committed.call_args_list))
        start['start_utc'] = '2026-09-14T04:59:00+00:00'
        self.put('started.json', start)
        with patch.object(cli, 'committed') as committed:
            with self.assertRaises(ValueError):
                cli.prospective_start(path, registration_path, diagnostic)
            committed.assert_not_called()
        start['start_utc'] = '2026-09-14T05:01:00+00:00'
        self.put('started.json', start)
        with patch.object(cli, 'committed', side_effect=ValueError('not in generation commit')):
            with self.assertRaises(ValueError):
                cli.prospective_start(path, registration_path, diagnostic)

    def test_raw_inventory_detects_added_file_and_ambiguous_terminal(self):
        start = self.put('generation/one/started.json', {})
        self.put('generation/one/run_manifest.json', {})
        with patch.object(cli, 'GENERATION', self.root / 'generation'):
            expected = {str((start.parent / 'run_manifest.json').resolve())}
            self.assertEqual(cli.generation_terminals([start]), expected)
            before = cli.artifact_hashes(start.parent)
            self.put('generation/one/unlisted.json', {})
            self.assertNotEqual(before, cli.artifact_hashes(start.parent))
            self.put('generation/one/abandonment.json', {})
            with self.assertRaises(ValueError):
                cli.generation_terminals([start])

    def test_generation_terminal_without_start_is_rejected(self):
        self.put('generation/orphan/run_manifest.json', {})
        with patch.object(cli, 'GENERATION', self.root / 'generation'):
            with self.assertRaises(ValueError):
                cli.generation_terminals([])

    def test_changed_dynamic_inputs_abort_before_report_write(self):
        import report_core
        diagnostic = {'contract': {'inputs': {'matrix': {'path': 'unused.json'}}}}
        first = ({'status': 'SCREEN_NO_GO'}, {}, {'audit': 'a'}, {'raw': 'b'})
        changed = ({'status': 'SCREEN_GO'}, {}, {'audit': 'c'}, {'raw': 'b'})
        with patch.object(cli, 'within', side_effect=lambda path, *args: Path(path)), \
             patch.object(cli, 'verify', return_value=diagnostic), \
             patch.object(cli, 'verified_screen', side_effect=[first, changed]), \
             patch.object(cli, 'read', return_value={}), \
             patch.object(report_core, 'analyze', return_value={}), \
             patch.object(cli, 'write') as write:
            with self.assertRaisesRegex(ValueError, 'screen inputs changed'):
                cli.build_report(self.root / 'reg.json', self.root / 'audit')
            write.assert_not_called()


if __name__ == '__main__':
    unittest.main()
