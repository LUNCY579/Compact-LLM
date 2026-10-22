"""Cross-platform source binding and aggregate-storage fail-closed controls."""
from pathlib import Path
import tempfile
import unittest

from experiments.rineya_verified_resampling_v1 import contracts as C
from experiments.rineya_verified_resampling_v1.runner import Journal


class HardeningTests(unittest.TestCase):
    def test_windows_checkout_newlines_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'source.py'
            p.write_bytes(b'x = 1\ny = 2\n')
            expected = C.blob_sha(p)
            self.assertTrue(C.matches_git_blob(p, expected))
            p.write_bytes(b'x = 1\r\ny = 2\r\n')
            self.assertTrue(C.matches_git_blob(p, expected))
            p.write_bytes(b'x = 2\r\ny = 2\r\n')
            self.assertFalse(C.matches_git_blob(p, expected))

    def test_active_byte_hash_preserves_newline_difference(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d)/'a', Path(d)/'b'
            a.write_bytes(b'x\n'); b.write_bytes(b'x\r\n')
            self.assertNotEqual(C.file_sha(a), C.file_sha(b))

    def test_role_must_be_string(self):
        with self.assertRaises(C.Blocked):
            C.validate_public([{'id': 'a'*64, 'messages': [{'role': [], 'content': 'x'}]}])

    def test_all_output_files_count(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'a').write_bytes(b'a'*20)
            (root/'nested').mkdir()
            (root/'nested'/'b').write_bytes(b'b'*30)
            self.assertEqual(C.directory_bytes(root), 50)

    def test_storage_blocks_new_attempt_before_begin(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root/'other.log').write_bytes(b'x'*100)
            journal = Journal(root/'events.jsonl', 100)
            with self.assertRaises(C.Blocked):
                journal.append('begin', {'slot':0, 'task':'a'*64, 'index':0, 'arm':'greedy', 'seed':1})
            self.assertFalse((root/'events.jsonl').exists())

    def test_qualification_trace_limit_not_full_cap_claim(self):
        source = (C.PACKAGE/'torch_sampler.py').read_text()
        self.assertNotIn('"qualified_max_new_tokens"', source)
        self.assertIn('"qualification_trace_cap"', source)


if __name__ == '__main__':
    unittest.main()
