import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from experiments.skill_acquisition_native_v1 import data, protocol


class DataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        (self.root / 'manifests').mkdir()
        (self.root / 'canonical').mkdir()
        (self.root / 'policy').mkdir()

    def fixture(self, *, duplicate=False, missing=False, wrong_role=False, wrong_policy=False, missing_policy=False):
        rows, files = [], {}
        for skill in ('R', 'S'):
            ids = [skill + str(i) for i in range(2)]
            raw = json.dumps({'name': 'atomic_' + skill, 'rows': 2, 'ids': ids}).encode()
            path = 'manifests/atomic_' + skill + '.json'
            (self.root / path).write_bytes(raw)
            files[path] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            for row_id in ids:
                prompt = [{'role': 'user', 'content': row_id}]
                rows.append({'id': row_id, 'prompt': prompt, 'prompt_hash': protocol.digest(prompt),
                             'role': 'test' if wrong_role else 'train', 'tag': 'core', 'kind': 'composition',
                             'skills': [skill], 'data_source': 'rineya_maxrl_comp_v2'})
        if duplicate:
            rows.append(rows[0])
        if missing:
            rows.pop()
        raw = b''.join(json.dumps(row).encode() + b'\n' for row in rows)
        (self.root / data.TRAIN_FILE).write_bytes(raw)
        files[data.TRAIN_FILE] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        policy = [{'id': r['id'], 'messages': r['prompt']} for r in rows]
        if wrong_policy:
            policy[0]['messages'] = [{'role': 'user', 'content': 'changed'}]
        if missing_policy:
            policy.pop()
        raw = b''.join(json.dumps(r).encode() + b'\n' for r in policy)
        (self.root / data.POLICY_FILE).write_bytes(raw)
        files[data.POLICY_FILE] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
        raw = json.dumps({'files': files}).encode()
        (self.root / 'manifest.json').write_bytes(raw)
        return hashlib.sha256(raw).hexdigest()

    def load(self, sha):
        return data.load_atomic_train(self.root, ['R', 'S'], expected_count=2, expected_manifest=sha)

    def test_exact_atomic_sets_and_source_receipts(self):
        rows, manifests, receipt = self.load(self.fixture())
        self.assertEqual(set(rows), {'R0', 'R1', 'S0', 'S1'})
        self.assertEqual(set(manifests), {'R', 'S'})
        self.assertEqual(receipt['rows'], 4)

    def test_duplicate_missing_or_evaluation_rows_rejected(self):
        for kwargs in ({'duplicate': True}, {'missing': True}, {'wrong_role': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(protocol.AdmissionRejected):
                self.load(self.fixture(**kwargs))

    def test_payload_tampering_rejected(self):
        sha = self.fixture()
        with (self.root / data.TRAIN_FILE).open('ab') as stream:
            stream.write(b'{}\n')
        with self.assertRaises(protocol.AdmissionRejected):
            self.load(sha)

    def test_policy_prompt_or_membership_drift_rejected_even_with_rehashed_payload(self):
        for kwargs in ({'wrong_policy': True}, {'missing_policy': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(protocol.AdmissionRejected):
                self.load(self.fixture(**kwargs))

    def test_compound_skill_and_bridge_requests_rejected(self):
        sha = self.fixture()
        for skills in (['RS'], ['R', 'R'], ['U'], []):
            with self.subTest(skills=skills), self.assertRaises(protocol.AdmissionRejected):
                data.load_atomic_train(self.root, skills, expected_count=2, expected_manifest=sha)

    def test_strict_json_rejects_duplicate_keys_and_nonfinite(self):
        for raw in (b'{"id":"x","id":"y"}', b'{"x":NaN}'):
            with self.assertRaises(protocol.AdmissionRejected):
                data._json(raw)

    def test_renderer_is_explicit_and_ids_are_bound(self):
        rows, _, _ = self.load(self.fixture())
        calls = []
        def renderer(tokenizer, prompt):
            calls.append((tokenizer, prompt))
            return 'rendered', [1, 2, 3]
        rendered = data.render_training_prompts('fixture', rows, renderer)
        self.assertEqual(len(calls), 4)
        self.assertEqual(rendered['R0']['prompt_ids_sha256'], protocol.digest([1, 2, 3]))
        with self.assertRaises(protocol.AdmissionRejected):
            data.render_training_prompts('fixture', rows, None)


if __name__ == '__main__':
    unittest.main()
