import unittest
from worker import Budget, MAX_FORWARDS, MAX_GENERATIONS, MAX_TOKEN_POSITIONS, prompt_system, validate_containment, validate_freeze
from benchmark import build_screen
import json
import tempfile
from pathlib import Path
from geometry import sha256


class BudgetTests(unittest.TestCase):
    def test_prospective_limits_can_fail(self):
        clock=[0.]
        b=Budget(now=lambda:clock[0])
        b.forward(256)
        self.assertEqual(b.receipt()["model_forwards"],1)
        for attr,value,action in (
            ("forwards",MAX_FORWARDS,lambda:b.forward(1)),
            ("token_positions",MAX_TOKEN_POSITIONS,lambda:b.forward(1)),
            ("generations",MAX_GENERATIONS,b.generation),
        ):
            before=getattr(b,attr)
            setattr(b,attr,value)
            with self.assertRaises(RuntimeError):action()
            setattr(b,attr,before)
        clock[0]=600
        with self.assertRaises(RuntimeError):b.check()

    def test_full_fixed_slot_screen_fits_forward_envelope(self):
        # 64*24 screen steps plus 2 dev rows*2 padding variants*4 steps.
        self.assertEqual(64*24+2*2*4,1552)
        self.assertLessEqual(1552,MAX_FORWARDS)
        self.assertLessEqual(64*(256+23)+4*(256+3),MAX_TOKEN_POSITIONS)

    def test_prompt_control_changes_exactly_transfer_systems(self):
        rows=build_screen()
        self.assertEqual(sum(prompt_system(row,"PROMPT") != row.system for row in rows),4)
        for row in rows:
            self.assertEqual(prompt_system(row,"TH"),row.system)
            self.assertEqual(prompt_system(row,"PROMPT") != row.system,"rules" in row.metadata)

    def test_unqualified_containment_cannot_start_native(self):
        with self.assertRaisesRegex(RuntimeError,"CONTAINMENT_INTEGRATION_UNQUALIFIED"):
            validate_containment(Path("nonexistent"),"0"*64)

    def test_empty_freeze_cannot_be_accepted(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"freeze.json"
            path.write_text(json.dumps({"files":{}}),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"incomplete"):
                validate_freeze(path,sha256(path))

    def test_false_owner_verifier_cannot_qualify(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"handoff.json"
            path.write_text("{}",encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"owner verifier"):
                validate_containment(path,sha256(path),lambda *args:False)


if __name__=="__main__":unittest.main()
