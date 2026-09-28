"""Synthetic CPU checks for the fixed prospective E1 schedule; no data-file access."""
import ast
from collections import Counter
from copy import deepcopy
import inspect
import json
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import e1_schedule as schedule


def synthetic_rows():
    rows = []
    for component, count in schedule.COMPONENT_COUNTS.items():
        for index in range(count):
            # Replay variants may share a latent parent; acquisition parents may not.
            parent_index = index // 2 if component == "deterministic_replay" else index
            rows.append({
                "key": f"{component}:{index}", "component": component, "role": "TRAIN",
                "parent_id": f"{component}:parent:{parent_index}",
                "nonpadding_tokens": 80 + index % 23,
                "completion_tokens": 2 + index % 11,
            })
    return rows


def synthetic_identities():
    return {field: schedule.sha256_json(field) for field in schedule.IDENTITY_FIELDS}


class E1ScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = synthetic_rows()
        cls.identities = synthetic_identities()
        cls.plan = schedule.build_plan(cls.rows, identities=cls.identities)

    def test_exact_one_epoch_dose_and_interleave_without_family_quotas(self):
        by_key = {row["key"]: row for row in self.rows}
        used = []
        self.assertEqual(len(self.plan["steps"]), 256)
        for index, step in enumerate(self.plan["steps"], 1):
            self.assertEqual(step["optimizer_step"], index)
            self.assertEqual(len(step["row_keys"]), 20)
            components = [by_key[key]["component"] for key in step["row_keys"]]
            self.assertEqual(components.count("acquisition"), 16)
            self.assertEqual(sum(value != "acquisition" for value in components), 4)
            self.assertEqual([value == "acquisition" for value in components],
                             [True, True, True, True, False] * 4)
            used.extend(step["row_keys"])
        self.assertEqual(Counter(used), Counter(by_key.keys()))
        self.assertEqual(self.plan["totals"]["rows"], 5120)
        self.assertEqual({name: entry["rows"] for name, entry in self.plan["totals"]["components"].items()},
                         {"acquisition": 4096, "deterministic_replay": 768, "language_replay": 256})
        self.assertIsNone(self.plan["schedule_config"]["family_quotas_per_step"])

    def test_loss_is_completion_mean_then_twenty_sequence_mean(self):
        config = self.plan["schedule_config"]
        self.assertEqual(config["microbatch_sequences"], 1)
        self.assertEqual(config["sequences_per_optimizer_step"], 20)
        self.assertEqual(config["loss"]["token_reduction"],
                         "mean CE over completion tokens including EOS in each sequence")
        self.assertEqual(config["loss"]["sequence_reduction"],
                         "arithmetic mean of 20 sequence losses per optimizer step")
        self.assertEqual(config["loss"]["implementation_status"], "PROPOSED_NO_TRAINING_LOOP")

    def test_lr_boundaries_exact_prospective_update_semantics(self):
        self.assertEqual(schedule.WARMUP_STEPS, 13)
        self.assertEqual(schedule.learning_rate(1), 1e-4 / 13)
        self.assertEqual(schedule.learning_rate(13), 1e-4)
        self.assertEqual(schedule.learning_rate(14), 1e-4 * 242 / 243)
        self.assertEqual(schedule.learning_rate(255), 1e-4 / 243)
        self.assertEqual(schedule.learning_rate(256), 0.0)
        rates = [step["learning_rate"] for step in self.plan["steps"]]
        self.assertTrue(all(first < second for first, second in zip(rates[:12], rates[1:13])))
        self.assertTrue(all(first > second for first, second in zip(rates[12:-1], rates[13:])))
        for bad in (0, 257, -1, True, 1.0, None):
            with self.subTest(update=bad), self.assertRaises(ValueError):
                schedule.learning_rate(bad)

    def test_checkpoints_dev_and_final_only_are_frozen(self):
        self.assertEqual(self.plan["checkpoint_steps"], [0, 64, 128, 192, 256])
        self.assertEqual(self.plan["dev_evaluation_steps"], [0, 64, 128, 192, 256])
        self.assertEqual([step["optimizer_step"] for step in self.plan["steps"] if step["checkpoint_after"]],
                         [64, 128, 192, 256])
        self.assertEqual([step["optimizer_step"] for step in self.plan["steps"] if step["dev_evaluation_after"]],
                         [64, 128, 192, 256])
        self.assertEqual(self.plan["final_candidate_optimizer_step"], 256)
        self.assertEqual(self.plan["schedule_config"]["held_out_access"], {"CHECK": False, "FINAL": False})
        self.assertEqual(self.plan["status"], "PROPOSED_PLAN_ONLY_NO_NATIVE_EXECUTION")

    def test_actual_tokens_and_prefix_ledger_are_exact(self):
        by_key = {row["key"]: row for row in self.rows}
        nonpadding = completion = 0
        components = Counter()
        for step in self.plan["steps"]:
            batch = [by_key[key] for key in step["row_keys"]]
            tokens = sum(row["nonpadding_tokens"] for row in batch)
            target_tokens = sum(row["completion_tokens"] for row in batch)
            self.assertEqual(step["nonpadding_tokens"], tokens)
            self.assertEqual(step["completion_tokens"], target_tokens)
            nonpadding += tokens
            completion += target_tokens
            components.update(row["component"] for row in batch)
            self.assertEqual(step["cumulative_nonpadding_tokens"], nonpadding)
            self.assertEqual(step["cumulative_completion_tokens"], completion)
            self.assertEqual(step["cumulative_component_rows"], dict(components))
        self.assertEqual(self.plan["totals"]["nonpadding_tokens"], sum(row["nonpadding_tokens"] for row in self.rows))
        self.assertEqual(self.plan["totals"]["completion_tokens"], sum(row["completion_tokens"] for row in self.rows))
        self.assertLess(self.plan["totals"]["nonpadding_tokens"], schedule.MAX_NONPADDING_TOKENS)

    def test_stable_json_hash_and_repeatability_without_input_mutation(self):
        rows, identities = deepcopy(self.rows), deepcopy(self.identities)
        before = deepcopy(rows)
        first = schedule.build_plan(rows, identities=identities)
        second = schedule.build_plan(rows, identities=identities)
        self.assertEqual(first, second)
        self.assertEqual(rows, before)
        body = deepcopy(first)
        digest = body.pop("plan_sha256")
        self.assertEqual(digest, schedule.sha256_json(body))
        self.assertEqual(json.loads(json.dumps(first)), first)
        rows[0]["nonpadding_tokens"] += 1
        identities["source_sha256"] = "a" * 64
        self.assertEqual(first, second)

    def test_bound_input_order_changes_schedule_and_hash(self):
        rows = deepcopy(self.rows)
        rows[0], rows[1] = rows[1], rows[0]
        changed = schedule.build_plan(rows, identities=self.identities)
        self.assertNotEqual(changed["plan_sha256"], self.plan["plan_sha256"])
        self.assertNotEqual(changed["ordered_rows_sha256"], self.plan["ordered_rows_sha256"])
        self.assertNotEqual(changed["steps"], self.plan["steps"])
        with self.assertRaises(ValueError):
            schedule.validate_plan(self.plan, rows, identities=self.identities)

    def test_seed101_golden_shuffle_order_and_combined_replay(self):
        # These boundary rows distinguish one RNG shuffled in acquisition/replay
        # order from independent RNGs, a reversed call order, or separate replay shuffles.
        self.assertEqual(self.plan["steps"][0]["row_keys"], [
            "acquisition:1644", "acquisition:543", "acquisition:2732", "acquisition:2144",
            "deterministic_replay:124", "acquisition:3753", "acquisition:397", "acquisition:3905",
            "acquisition:856", "language_replay:61", "acquisition:1071", "acquisition:163",
            "acquisition:78", "acquisition:1283", "language_replay:90", "acquisition:3149",
            "acquisition:263", "acquisition:4060", "acquisition:1078", "deterministic_replay:612",
        ])
        self.assertEqual(self.plan["steps"][-1]["row_keys"][-5:], [
            "acquisition:2208", "acquisition:3953", "acquisition:3720", "acquisition:1595",
            "deterministic_replay:381",
        ])

    def test_seed_gate_rejects_confirmation_and_unregistered_seeds(self):
        for seed in (211, 307, 0, 100, 102, True, 101.0, "101", None):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                schedule.build_plan(self.rows, identities=self.identities, seed=seed)

    def test_omitted_or_extra_rows_and_wrong_component_counts_refused(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.assertRaises(ValueError):
                schedule.build_plan(rows, identities=self.identities)
        rows = deepcopy(self.rows)
        rows[-1]["component"] = "deterministic_replay"
        with self.assertRaisesRegex(ValueError, "4096/768/256"):
            schedule.build_plan(rows, identities=self.identities)

    def test_duplicate_rows_and_acquisition_parents_refused(self):
        for field in ("key", "parent_id"):
            rows = deepcopy(self.rows)
            rows[1][field] = rows[0][field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                schedule.build_plan(rows, identities=self.identities)
        rows = deepcopy(self.rows)
        rows[-1] = dict(rows[-2])
        with self.assertRaises(ValueError):
            schedule.build_plan(rows, identities=self.identities)

    def test_replay_variants_share_parent_but_cross_component_parent_is_refused(self):
        self.assertEqual(self.rows[4096]["parent_id"], self.rows[4097]["parent_id"])
        schedule.validate_plan(self.plan, self.rows, identities=self.identities)
        rows = deepcopy(self.rows)
        rows[-1]["parent_id"] = rows[0]["parent_id"]
        with self.assertRaisesRegex(ValueError, "component boundaries"):
            schedule.build_plan(rows, identities=self.identities)

    def test_every_nontrain_role_is_refused_before_scheduling(self):
        for role in ("TRAIN_SELECTION", "DEV", "CHECK", "FINAL", "train", None):
            rows = deepcopy(self.rows)
            rows[0]["role"] = role
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "TRAIN only"):
                schedule.build_plan(rows, identities=self.identities)

    def test_row_schema_tokens_and_missing_fields_fail_closed(self):
        mutations = [
            ("component", "other"), ("key", ""), ("parent_id", " "),
            ("nonpadding_tokens", 1025), ("nonpadding_tokens", 0),
            ("nonpadding_tokens", True), ("nonpadding_tokens", 80.0),
            ("completion_tokens", 129), ("completion_tokens", 0),
            ("completion_tokens", -1), ("completion_tokens", float("nan")),
            ("completion_tokens", 80),
        ]
        for field, value in mutations:
            rows = deepcopy(self.rows)
            rows[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                schedule.build_plan(rows, identities=self.identities)
        rows = deepcopy(self.rows)
        del rows[0]["role"]
        with self.assertRaises(ValueError):
            schedule.build_plan(rows, identities=self.identities)

    def test_exact_identity_schema_and_digest_format_required(self):
        for field in schedule.IDENTITY_FIELDS:
            identities = deepcopy(self.identities)
            del identities[field]
            with self.subTest(field=field), self.assertRaises(ValueError):
                schedule.build_plan(self.rows, identities=identities)
        for digest in ("a" * 40, "G" * 64, "A" * 64, None, 123):
            identities = deepcopy(self.identities)
            identities["source_sha256"] = digest
            with self.subTest(digest=digest), self.assertRaises(ValueError):
                schedule.build_plan(self.rows, identities=identities)

    def test_schedule_ledger_and_rng_mutations_fail_even_if_plan_rehashed(self):
        for change in ("duplicate", "reorder", "tokens", "learning_rate", "rng", "config"):
            plan = deepcopy(self.plan)
            if change == "duplicate":
                plan["steps"][0]["row_keys"][1] = plan["steps"][0]["row_keys"][0]
            elif change == "reorder":
                plan["steps"][0]["row_keys"].reverse()
            elif change == "tokens":
                plan["steps"][0]["nonpadding_tokens"] += 1
            elif change == "learning_rate":
                plan["steps"][0]["learning_rate"] = 1e-4
            elif change == "rng":
                plan["rng_contract"]["calls"].reverse()
                plan["rng_contract_sha256"] = schedule.sha256_json(plan["rng_contract"])
            else:
                plan["schedule_config"]["learning_rate"]["warmup_steps"] = 12
                plan["schedule_config_sha256"] = schedule.sha256_json(plan["schedule_config"])
            del plan["plan_sha256"]
            plan["plan_sha256"] = schedule.sha256_json(plan)
            with self.subTest(change=change), self.assertRaises(ValueError):
                schedule.validate_plan(plan, self.rows, identities=self.identities)

    def test_resume_at_zero_middle_and_final_boundaries(self):
        for completed in (0, 1, 13, 64, 128, 192, 255, 256):
            cursor = schedule.make_resume_cursor(self.plan, completed)
            next_update = schedule.validate_resume_cursor(cursor, self.plan, rows=self.rows,
                                                          identities=self.identities)
            self.assertEqual(next_update, completed + 1 if completed < 256 else None)
            prefix = self.plan["steps"][:completed]
            self.assertEqual(cursor["consumed_rows"], 20 * completed)
            self.assertEqual(cursor["consumed_component_rows"]["acquisition"], 16 * completed)
            self.assertEqual(cursor["consumed_nonpadding_tokens"], sum(step["nonpadding_tokens"] for step in prefix))
            self.assertEqual(cursor["consumed_completion_tokens"], sum(step["completion_tokens"] for step in prefix))
            self.assertEqual(cursor["native_checkpoint_integrity"], "UNVERIFIED_BY_SCHEDULE_CONTRACT")

    def test_resume_rejects_partial_skipped_and_inconsistent_prefixes(self):
        original = schedule.make_resume_cursor(self.plan, 64)
        fields = (
            "accumulated_sequences", "completed_optimizer_steps", "next_optimizer_step",
            "next_sequence_index", "consumed_rows", "consumed_nonpadding_tokens", "consumed_completion_tokens",
        )
        for field in fields:
            cursor = deepcopy(original)
            cursor[field] += 1
            with self.subTest(field=field), self.assertRaises(ValueError):
                schedule.validate_resume_cursor(cursor, self.plan)
        cursor = deepcopy(original)
        cursor["consumed_component_rows"]["deterministic_replay"] += 1
        with self.assertRaises(ValueError):
            schedule.validate_resume_cursor(cursor, self.plan)
        for completed in (True, 64.5, -1, 257):
            with self.subTest(completed=completed), self.assertRaises(ValueError):
                schedule.make_resume_cursor(self.plan, completed)

    def test_resume_rejects_changed_source_config_model_data_parent_or_plan(self):
        cursor = schedule.make_resume_cursor(self.plan, 64)
        for field in schedule.IDENTITY_FIELDS:
            changed = deepcopy(self.identities)
            changed[field] = "f" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                schedule.validate_resume_cursor(cursor, self.plan, rows=self.rows, identities=changed)
            changed_plan = schedule.build_plan(self.rows, identities=changed)
            with self.subTest(field=field, fresh_plan=True), self.assertRaises(ValueError):
                schedule.validate_resume_cursor(cursor, changed_plan)
        rows = deepcopy(self.rows)
        rows[0]["completion_tokens"] += 1
        with self.assertRaises(ValueError):
            schedule.validate_resume_cursor(cursor, self.plan, rows=rows, identities=self.identities)

    def test_resume_rejects_rewritten_rng_contract_and_extra_claims(self):
        cursor = schedule.make_resume_cursor(self.plan, 64)
        cursor["rng_contract_sha256"] = "f" * 64
        with self.assertRaises(ValueError):
            schedule.validate_resume_cursor(cursor, self.plan)
        cursor = schedule.make_resume_cursor(self.plan, 64)
        cursor["native_checkpoint_integrity"] = True
        with self.assertRaises(ValueError):
            schedule.validate_resume_cursor(cursor, self.plan)

    def test_module_has_only_stdlib_imports_and_no_file_or_native_launch_api(self):
        tree = ast.parse(inspect.getsource(schedule))
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                modules.add(node.module.split(".")[0])
        self.assertLessEqual(modules, {
            "__future__", "collections", "copy", "hashlib", "json", "platform", "random", "re", "sys",
        })
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                self.assertNotIn(node.func.id, {"open", "exec", "eval", "__import__"})


if __name__ == "__main__":
    unittest.main()
