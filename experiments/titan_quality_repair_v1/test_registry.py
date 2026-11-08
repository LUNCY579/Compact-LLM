"""Model-free registry and irreversible-accounting falsifiers.
Run only under the root's exclusive hardware-policy offline supervisor.
"""
from contextlib import contextmanager
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import budget, registry, schema


class RegistryTests(unittest.TestCase):
    def test_eight_recipes_sixteen_exact_trials_and_fixed_one_epoch(self):
        value = registry.build_registry()
        self.assertEqual(len(value["recipes"]), 8)
        self.assertEqual(len(value["trials"]), 16)
        self.assertEqual({trial["trial_id"] for trial in value["trials"]},
                         {f"TQR-R{number:02d}-S{seed}" for number in range(1, 9) for seed in (17, 29)})
        self.assertEqual({(r["prompt_coverage"], r["difficulty"], r["replay_fraction"]) for r in value["recipes"]},
                         {(p, d, f) for p in ("canonical", "diverse") for d in ("regular", "counterfactual") for f in (0, 0.2)})
        for recipe in value["recipes"]:
            self.assertEqual(recipe["total_examples"], 2000)
            self.assertEqual(recipe["repair_examples"] + recipe["replay_examples"], 2000)
            self.assertEqual(recipe["replay_examples"], 400 if recipe["replay_fraction"] else 0)
            train = recipe["training"]
            self.assertEqual(train["optimizer_steps"] * train["gradient_accumulation"] * train["microbatch"], 2000)
            self.assertEqual((train["rank"], train["alpha"], train["dropout"]), (16, 32, 0.05))
            self.assertEqual((train["learning_rate_schedule"], train["warmup_steps"], train["weight_decay"]), ("constant", 0, 0))
            self.assertEqual((train["adamw_betas"],train["adamw_eps"],train["gradient_clip_norm"]),([0.9,0.999],1e-8,1.0))
            self.assertTrue(train["gradient_checkpointing"] and train["preserve_rng_state"] and train["deterministic_algorithms"])
            self.assertTrue(train["inference_use_cache"])
            self.assertFalse(train["training_use_cache"] or train["tf32"] or train["amp"] or train["dtype_fallback"] or train["qualification_search"])
            self.assertEqual(train["attention_implementation"],"eager")
            self.assertEqual(train["loss_reduction"],"mean_over_positive_assistant_and_final_EOS_tokens")
            self.assertEqual(train["data_order"],"fixed_registered_seed_shuffle_single_2000_example_epoch")
        self.assertEqual(value["source"]["architecture"], "Qwen3ForCausalLM")
        self.assertEqual(value["source"]["weight_sha256"], registry.SOURCE_WEIGHTS_SHA256)
        self.assertFalse(value["source"]["activation_offsets"] or value["source"]["activation_routers"])
        membership=value["evaluation_membership"]
        self.assertEqual(7*64+4*16,membership["development"]["prompts_per_arm"])
        self.assertEqual(7*512+4*128,membership["final"]["prompts_per_arm"])
        self.assertFalse(membership["selection_uses_model_outcomes"])

    def test_registry_mutation_and_wrong_source_refused(self):
        original = registry.build_registry()
        changed = registry.build_registry()
        changed["recipes"][0]["training"]["learning_rate"] = 1e-4
        with self.assertRaisesRegex(ValueError, "REGISTRY_DRIFT"):
            registry.validate_registry(changed)
        self.assertEqual(registry.build_registry(), original)
        changed = registry.build_registry()
        changed["source"]["weight_sha256"] = "a" * 64
        with self.assertRaisesRegex(ValueError, "REGISTRY_DRIFT"):
            registry.make_plan(changed)

    def test_complete_budget_counts_qualification_exports_and_failure_charge(self):
        plan = registry.make_plan()
        self.assertEqual(plan["budget_seconds"], {"qualification": 3600, "training": 28800, "evaluation": 54000})
        self.assertEqual(plan["total_budget_seconds"], 86400)
        self.assertEqual(plan["base_evaluation_prompts"], 512 + 8192 + 12288)
        self.assertEqual(plan["evaluation_prompts_with_export_checks"], 20992 + 1024)
        self.assertEqual(plan["total_generation_calls"], 22016 + 256)
        self.assertEqual(plan["max_decode_token_slots"], 1425408)
        self.assertEqual((plan["training_microbatches_total"], plan["training_optimizer_updates_total"]), (32000,4000))
        self.assertEqual(plan["max_teacher_forced_token_positions"], 32000*2048)
        self.assertEqual(plan["feasibility"]["status"], "BLOCKED_GENUINE_MEASURED_THROUGHPUT_REQUIRED")
        self.assertFalse(plan["native_execution_authorized"])


class BudgetTests(unittest.TestCase):
    def setUp(self):
        budget.ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(prefix="budget-fixture-", dir=budget.ARTIFACT_ROOT)
        self.base = Path(self.temp.name).resolve()
        self.assertTrue(self.base.is_relative_to(budget.ARTIFACT_ROOT.resolve()))
        self.addCleanup(self.temp.cleanup)
        self.root = self.base / "ledger"
        patcher = patch.object(budget, "CANONICAL_LEDGER_ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.ledger = budget.BudgetLedger()
        self.trial = registry.build_registry()["trials"][0]["trial_id"]

    def test_crashed_unknown_failed_attempts_always_remain_charged(self):
        self.ledger.charge_attempt("training", "trial-first", 1800, trial_id=self.trial)
        reopened = budget.BudgetLedger()
        self.assertEqual(reopened.snapshot()["charged_seconds"]["training"], 1800)
        reopened.record_result("trial-first", "FAILED", observed_seconds=2)
        self.assertEqual(reopened.snapshot()["charged_seconds"]["training"], 1800)
        with self.assertRaisesRegex(RuntimeError, "ALREADY_USED"):
            reopened.charge_attempt("training", "retry-new-id", 1800, trial_id=self.trial)
        with self.assertRaisesRegex(RuntimeError, "ALREADY_RECORDED"):
            reopened.record_result("trial-first", "SUCCEEDED", observed_seconds=1)

    def test_explicit_unknown_receipt_cannot_refund_or_reset(self):
        self.ledger.charge_attempt("qualification", "qual-first", 600)
        self.ledger.record_result("qual-first", "OUTCOME_UNKNOWN", receipt="fixture-only-not-native")
        self.assertEqual(budget.BudgetLedger().snapshot()["charged_seconds"]["qualification"], 600)
        with self.assertRaisesRegex(ValueError, "ROOT_ROTATION"):
            budget.BudgetLedger(self.base / "rotated")

    def test_training_attempt_cost_fixed_and_full_exhaustion_skips(self):
        with self.assertRaisesRegex(ValueError, "FIXED_ATTEMPT_COST"):
            self.ledger.charge_attempt("training", "cheap-attempt", 1, trial_id=self.trial)
        self.ledger.charge_attempt("evaluation", "eval-full", 54000)
        skip = self.ledger.charge_attempt("evaluation", "eval-after-full", 1)
        self.assertEqual(skip["kind"], "SKIPPED_BUDGET_EXHAUSTED")
        self.assertEqual(skip["declared_seconds"], 0)
        self.assertEqual(self.ledger.snapshot()["charged_seconds"]["evaluation"], 54000)
        with self.assertRaisesRegex(RuntimeError, "ALREADY_USED"):
            self.ledger.charge_attempt("evaluation", "eval-after-full", 1)

    def test_observed_overrun_is_retained_and_blocks_more_spending(self):
        self.ledger.charge_attempt("qualification", "qual-one", 3600)
        result = self.ledger.record_result("qual-one", "ABORTED", observed_seconds=3602)
        self.assertEqual(result["additional_overrun_seconds"], 2)
        state = self.ledger.snapshot()
        self.assertEqual(state["charged_seconds"]["qualification"], 3602)
        self.assertTrue(state["budget_overrun"])
        self.assertEqual(self.ledger.charge_attempt("qualification", "qual-two", 1)["kind"], "SKIPPED_BUDGET_EXHAUSTED")
        self.assertEqual(self.ledger.charge_attempt("evaluation", "eval-after-overrun", 1)["kind"], "SKIPPED_BUDGET_EXHAUSTED")

    def test_foreign_writer_lock_never_reclaimed(self):
        path = self.root / budget.LOCK
        path.write_text('{"pid":1,"token":"foreign"}', encoding="utf-8")
        with self.assertRaises(FileExistsError):
            self.ledger.charge_attempt("qualification", "qual-busy", 1)
        self.assertEqual(json.loads(path.read_bytes())["token"], "foreign")

    def test_changed_event_or_missing_header_refuses_preserving_evidence(self):
        self.ledger.charge_attempt("qualification", "qual-one", 1)
        event = next(self.root.glob("EVENT_*.json"))
        original = event.read_bytes()
        value = json.loads(original)
        value["declared_seconds"] = 0
        event.write_text(json.dumps(value), encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "EVENT_UNREADABLE_OR_CHANGED"):
            self.ledger.snapshot()
        event.write_bytes(original)
        (self.root / budget.HEADER).unlink()
        with self.assertRaisesRegex(RuntimeError, "HEADER_MISSING"):
            budget.BudgetLedger()
        self.assertTrue(event.exists())

    def test_invalid_scalar_costs_ids_and_stages_refuse(self):
        for cost in (True, -1, float("nan"), float("inf"), None):
            with self.assertRaises(ValueError):
                self.ledger.charge_attempt("qualification", "qual-bad", cost)
        with self.assertRaisesRegex(ValueError, "ATTEMPT_ID"):
            self.ledger.charge_attempt("qualification", "../another-owner", 1)
        with self.assertRaisesRegex(ValueError, "UNKNOWN_BUDGET_STAGE"):
            self.ledger.charge_attempt("unregistered", "qual-bad", 1)


if __name__ == "__main__":
    unittest.main()
