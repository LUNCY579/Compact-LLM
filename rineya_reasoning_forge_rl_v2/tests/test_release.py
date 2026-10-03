"""Additional integrity and evaluator edge cases for the completed release."""
from __future__ import annotations
import contextlib
import copy
import io
import tempfile
import unittest
from pathlib import Path
from forge_rl.common import Writer, read_jsonl, dumps, sha_file, write_json
from forge_rl.build import build_data
from forge_rl.probes import build_probes, evaluate_probes
from forge_rl.audits import verify_probes, audit_release
from forge_rl.evaluate import evaluate
from forge_rl.adapters import VerifiableReward
from forge_rl.process_eval import evaluate_process


class ReleaseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.base = Path(cls.temp.name)
        cls.root = cls.base / 'release'
        with contextlib.redirect_stdout(io.StringIO()):
            build_data(cls.root, scale=.002)
            build_probes(cls.root)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_probe_readback_independent(self):
        report = verify_probes(self.root)
        self.assertEqual(report['views_checked'], 48)
        self.assertEqual(report['parent_worlds'], 8)
        self.assertTrue(report['all_four_long_layouts_record_multiset_matched'])

    def test_probe_refuses_overwrite(self):
        with self.assertRaises(FileExistsError):
            build_probes(self.root)

    def test_probe_invalid_count(self):
        with self.assertRaises(ValueError):
            build_probes(self.base/'unused', parents_per_family=0)

    def test_probe_gold_scoring(self):
        p = self.base/'probe_gold.jsonl'
        with Writer(p) as out:
            for row in read_jsonl(self.root/'probes/metadata.jsonl.gz'):
                out.write({'id':row['id'], 'prediction':dumps(row['reference'])})
        report = evaluate_probes(self.root, p)
        self.assertEqual(report['views_received'], 48)
        self.assertTrue(all(v == 1 for v in report['accuracy_by_variant_missing_as_wrong'].values()))
        self.assertTrue(all(v['both_correct'] == 8 for v in report['paired'].values()))

    def test_probe_missing_scoring(self):
        p = self.base/'probe_empty.jsonl'; p.write_text('')
        report = evaluate_probes(self.root,p)
        self.assertEqual(report['views_received'],0)
        self.assertTrue(all(v==0 for v in report['accuracy_by_variant_missing_as_wrong'].values()))
        self.assertTrue(all(v['complete_pairs']==0 for v in report['paired'].values()))

    def test_probe_duplicate_fail(self):
        row = next(read_jsonl(self.root/'probes/metadata.jsonl.gz'))
        p = self.base/'probe_dup.jsonl'
        p.write_text((dumps({'id':row['id'], 'prediction':dumps(row['reference'])})+'\n')*2)
        with self.assertRaises(ValueError): evaluate_probes(self.root,p)

    def test_probe_unhashable_id(self):
        p = self.base/'probe_bad_id.jsonl';p.write_text(dumps({'id':[], 'prediction':'{}'})+'\n')
        with self.assertRaises(ValueError): evaluate_probes(self.root,p)

    def test_probe_checksum(self):
        p = self.root/'probes/views.jsonl.gz';original=p.read_bytes()
        try:
            p.write_bytes(original+b'altered')
            with self.assertRaises(ValueError):verify_probes(self.root)
        finally:p.write_bytes(original)

    def test_entire_smoke_release_audit(self):
        report=audit_release(self.root)
        self.assertEqual(report['canonical_reason_mode_references_scored'],64)
        self.assertEqual(report['normalized_unique_user_prompts'],64)
        self.assertFalse(report['source_overlap']['performed'])
        self.assertEqual(report['process_labels']['true'],report['process_labels']['false'])

    def test_reason_adapter_matches_own_prompt(self):
        adapter=VerifiableReward(self.root,mode='reason_then_final')
        row=next(read_jsonl(self.root/'data/rl_reason/train.jsonl.gz'))
        ref=next(read_jsonl(self.root/'metadata/train.jsonl.gz'))
        completion='<think>'+ref['teacher_trace']+'</think><final>'+ref['reference_text']+'</final>'
        self.assertEqual(adapter([completion],[row['id']],[row['prompt']]),[1])
        self.assertIsInstance(adapter.__name__,str)

    def test_reason_adapter_wrong_mode_prompt(self):
        adapter=VerifiableReward(self.root,mode='reason_then_final')
        row=next(read_jsonl(self.root/'data/rl/train.jsonl.gz'))
        with self.assertRaises(ValueError):adapter(['{}'],[row['id']],[row['prompt']])


    def test_adapter_rejects_reference_tampering(self):
        path=self.root/'metadata/train.jsonl.gz';original=path.read_bytes()
        try:
            path.write_bytes(original+b'tampered')
            with self.assertRaises(ValueError):VerifiableReward(self.root)
        finally:path.write_bytes(original)

    def test_adapter_does_not_retain_large_teacher_traces(self):
        adapter=VerifiableReward(self.root)
        self.assertTrue(all(set(row)=={'world','certificate','prompt_sha256','reason_prompt_sha256'} for row in adapter.rows.values()))

    def test_adapter_unhashable_id_is_rejected(self):
        adapter=VerifiableReward(self.root)
        with self.assertRaises(ValueError):adapter(['{}'],[[]],[[]])

    def test_adapter_nontext_completion_is_rejected(self):
        adapter=VerifiableReward(self.root)
        row=next(read_jsonl(self.root/'data/rl/train.jsonl.gz'))
        with self.assertRaises(ValueError):adapter([False],[row['id']],[row['prompt']])

    def test_evaluation_reason_tokens_reported(self):
        ref=next(read_jsonl(self.root/'metadata/test_iid.jsonl.gz'))
        p=self.base/'measured.jsonl'
        p.write_text(dumps({'id':ref['id'],'prediction':ref['reference_text'],'completion_tokens':10,'reasoning_tokens':4})+'\n')
        report=evaluate(self.root,p,'test_iid')
        self.assertEqual(report['overall']['mean_measured_completion_tokens'],10)
        self.assertEqual(report['overall']['mean_measured_reasoning_tokens'],4)
        self.assertEqual(report['overall']['measured_reasoning_token_coverage'],1/8)

    def test_evaluation_bool_group_size_rejected(self):
        with self.assertRaises(ValueError):evaluate(self.root,self.base/'unused','test_iid',samples_per_prompt=True)

    def test_evaluation_bad_finish_reason(self):
        ref=next(read_jsonl(self.root/'metadata/test_iid.jsonl.gz'))
        p=self.base/'finish_bad.jsonl'
        p.write_text(dumps({'id':ref['id'],'prediction':'{}','finish_reason':[]})+'\n')
        with self.assertRaises(ValueError):evaluate(self.root,p,'test_iid')

    def test_evaluation_unhashable_id(self):
        p=self.base/'badid.jsonl';p.write_text(dumps({'id':[],'prediction':'{}'})+'\n')
        with self.assertRaises(ValueError):evaluate(self.root,p,'test_iid')

    def test_process_unhashable_prediction_id(self):
        p=self.base/'processbad.jsonl';p.write_text(dumps({'id':[],'prediction':'{}'})+'\n')
        with self.assertRaises(ValueError):evaluate_process(self.root/'process/train_transitions.jsonl.gz',p)


if __name__=='__main__':
    unittest.main()
