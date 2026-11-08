import json
from pathlib import Path
import tempfile
import unittest
from experiments.post_training_mimo_centralized_v1.scripts import audit_exclusions as a


class ExclusionControls(unittest.TestCase):
    def setUp(self):
        self.q={"entries":[{"bars":["ANY_READ"],"path":"sealed/x.jsonl","ids":["old-id"]}]}
        self.row={"parent_id":"p1","role":"TRAIN","prompt":"A fact"}
        self.old={"prompt_hashes":[],"problem_hashes":[],"provenance":{"canonical_instances":0}}

    def test_exact_historical_chat_caught(self):
        self.old["prompt_hashes"]=[a.digest(a.policy_messages(self.row))]
        r=a.scan_rows([("new","TRAIN",[self.row])],self.q,self.old)
        self.assertFalse(r["scoped_exact_checks_pass"])
        self.assertEqual(r["errors"][0]["type"],"historical_exact_chat")

    def test_parent_and_normalized_text_collisions_caught(self):
        dev={**self.row,"role":"DEV","prompt":" A  fact\n"}
        r=a.scan_rows([("a","TRAIN",[self.row]),("b","DEV",[dev])],self.q,self.old)
        self.assertEqual({e["type"] for e in r["errors"]},{"cross_role_parent","current_normalized_prompt_collision"})

    def test_known_barred_id_rejected(self):
        row={**self.row,"parent_id":"old-id"}
        self.assertFalse(a.scan_rows([("a","TRAIN",[row])],self.q,self.old)["scoped_exact_checks_pass"])

    def test_closed_role_never_scanned(self):
        with self.assertRaises(ValueError):a.scan_rows([("a","CHECK",[])],self.q,self.old)

    def test_protected_and_escaped_paths_rejected_before_open(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            for path in [root/"sealed/x.jsonl",root/"a/fresh_sealed.jsonl",root/"test2.jsonl",root/"../outside"]:
                with self.assertRaises(ValueError):a.readable(root,path,self.q)

    def test_bad_hash_inventory_and_assistant_leak_fail(self):
        with self.assertRaises(ValueError):a.inventory(json.dumps({**self.old,"prompt_hashes":["invalid"]}))
        with self.assertRaises(ValueError):a.policy_messages({"messages":[{"role":"assistant","content":"gold"}]})

    def test_clean_scoped_case_has_no_broader_inference(self):
        r=a.scan_rows([("a","TRAIN",[self.row])],self.q,self.old)
        self.assertTrue(r["scoped_exact_checks_pass"])
        self.assertEqual(r["rows_compared"],1)


if __name__=="__main__":unittest.main()
