"""Lightweight fixtures, not Rineya capability evidence. Run only in owned CPU job."""
from __future__ import annotations

import collections
import copy
import random
import unittest
from unittest.mock import patch

from . import generator as g, oracles, schema


def fixture_replay(family):
    p = g._payload(family, 0, 8, random.Random(91))
    row = g._row(family, p, "fixture", "canonical")
    row.update(split="replay", is_replay=True, replay_target=schema.canonical(row["expected"]),
               provenance={"source_path": "fixture-only://not-a-real-training-source", "source_sha256": "a" * 64,
                           "source_line": 1, "source_id": "unit-" + family,
                           "target_type": "HUMAN_OR_DETERMINISTIC_EXISTING_TRAIN_GOLD", "gold_verified": True})
    return row


class DataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dev = g.build_split("dev")

    def test_development_membership_and_variants(self):
        self.assertEqual(len(self.dev), 512)
        self.assertEqual(len({r["world_id"] for r in self.dev}), 448)
        self.assertEqual(len({r["row_id"] for r in self.dev}), 512)
        for family in schema.FAMILIES:
            rows = [r for r in self.dev if r["family"] == family]
            self.assertEqual(len([r for r in rows if r["variant_id"] == "canonical"]), 64)
            self.assertEqual(len(rows), 80 if family in schema.REPAIR_FAMILIES else 64)
        worlds = collections.defaultdict(list)
        for row in self.dev:
            worlds[row["world_id"]].append(row)
        for variants in worlds.values():
            self.assertEqual(len({r["operative_key"] for r in variants}), 1)
            self.assertTrue(all(r["expected"] == variants[0]["expected"] for r in variants))
        for family in ("answerability", "grounded_qa"):
            attached = [r for r in self.dev if r["family"] == family and r["variant_id"] != "canonical"]
            self.assertEqual(collections.Counter(r["strata"]["scenario"] for r in attached), dict.fromkeys(g.SCENARIOS, 2))

    def test_every_gold_and_contract_verified(self):
        for row in self.dev:
            self.assertTrue(oracles.verify_gold(row))
            self.assertEqual(g.render_prompt(row), row["user"])
            target = g.target_reply(row)
            self.assertTrue(target)
            if not row["answerable"]:
                self.assertEqual(target, row["contract"]["abstention"])
                if row["contract"]["kind"] == "text":
                    self.assertEqual(target, schema.ABSTENTION)

    def test_prospective_final_membership_gold_and_leakage(self):
        # Fresh prospective data only, no old bank I/O or model outcomes. The
        # production compiler seals payloads externally after these audits.
        rows = g.build_split("final")
        self.assertEqual(len(rows), 4096)
        self.assertEqual(len({r["world_id"] for r in rows}), 3584)
        self.assertEqual(len({r["row_id"] for r in rows}), 4096)
        self.assertFalse({r["world_id"] for r in rows} & {r["world_id"] for r in self.dev})
        self.assertFalse({r["operative_key"] for r in rows} & {r["operative_key"] for r in self.dev})
        self.assertTrue(all(g.partition_of(r["operative_key"]) == "final" and oracles.verify_gold(r) for r in rows))
        for family in schema.FAMILIES:
            group = [r for r in rows if r["family"] == family]
            self.assertEqual(sum(r["variant_id"] == "canonical" for r in group), 512)
            self.assertEqual(len(group), 640 if family in schema.REPAIR_FAMILIES else 512)
        self.assertEqual(g.design_metadata()["final"]["prompts"], len(rows))

    def test_typed_gold_bool_and_float_coercion_rejected(self):
        p = {"a": 1, "b": 0, "operator": "add"}
        base = g._row("arithmetic", p, "fixture", "canonical")
        for mutation in (True, 1.0):
            row = copy.deepcopy(base)
            row["expected"] = mutation
            with self.assertRaisesRegex(ValueError, "STOP_GOLD"):
                oracles.verify_gold(row)
        self.assertFalse(schema.typed_equal({"a": 1}, {"a": True}))
        self.assertFalse(schema.typed_equal([1], [1.0]))
        with patch.dict(oracles.ORACLES, {"arithmetic": (lambda p: 1, lambda p: True)}):
            with self.assertRaisesRegex(ValueError, "STOP_INDEPENDENT_ORACLE_DISAGREEMENT"):
                oracles.solve("arithmetic", p)

    def test_modulo_wrap_marks_support_expansion_inside_old_operand_range(self):
        payload = {"before": {"X": 95, "Y": 10, "Z": 1},
                   "event": {"kind": "add", "register": "X", "other": "Y", "value": 5, "threshold": 10, "tick": 1}}
        row = g._row("register", payload, "fixture", "canonical")
        self.assertEqual(row["expected"]["X"], 3)
        self.assertTrue(row["support_expansion"])
        self.assertTrue(row["strata"]["wrap"])
        self.assertFalse(row["strata"]["operand_support_expansion"])
        payload["before"]["X"] = 5
        plain = g._row("register", payload, "fixture", "canonical")
        self.assertFalse(plain["support_expansion"])
        self.assertFalse(plain["strata"]["wrap"])

    def test_gold_and_answerability_mutations_rejected(self):
        row = copy.deepcopy(next(r for r in self.dev if r["family"] == "ledger"))
        row["expected"]["A"] += 1
        with self.assertRaisesRegex(ValueError, "STOP_GOLD"):
            g.target_reply(row)
        row = copy.deepcopy(next(r for r in self.dev if not r["answerable"]))
        row["answerable"] = True
        with self.assertRaisesRegex(ValueError, "STOP_GOLD"):
            g.target_reply(row)

    def test_operative_key_ignores_names_untouched_values_and_tick(self):
        p = {"before": {"A": 19, "B": 7}, "event": {"kind": "scale", "account": "A", "other": "B", "amount": 3, "status": "posted", "tick": 1}}
        q = {"before": {"L": 19, "M": -123}, "event": {"kind": "scale", "account": "L", "other": "M", "amount": 3, "status": "posted", "tick": 90}}
        self.assertEqual(g.operative_key("ledger", p), g.operative_key("ledger", q))
        self.assertEqual(g.operative_number_key("ledger", p), ("ledger_replay", "scale", "posted", 19, 3, 57))

    def test_exposed_collision_is_replaced_with_same_stratum(self):
        parent = next(r for r in self.dev if r["family"] == "register")
        fresh = g.build_split("dev", [parent["operative_key"]])
        self.assertEqual(len(fresh), 512)
        self.assertNotIn(parent["operative_key"], {r["operative_key"] for r in fresh})
        self.assertEqual(collections.Counter(r["answerable"] for r in fresh), collections.Counter(r["answerable"] for r in self.dev))

    def test_no_silent_drop_when_generation_exhausts(self):
        with patch.object(g, "MAX_ATTEMPTS_PER_WORLD", 1), patch.object(g, "partition_of", return_value="final"):
            with self.assertRaisesRegex(ValueError, "STOP_GENERATION_EXHAUSTED"):
                g._family_rows("ledger", 4, "dev")

    def test_training_exact_dose_signs_branches_and_contract_coverage(self):
        recipe = {"prompt_coverage": "diverse", "difficulty": "regular", "replay_fraction": 0.0}
        rows = g.training_rows(recipe, 17, [])
        self.assertEqual(len(rows), 2000)
        self.assertEqual(len({r["row_id"] for r in rows}), 2000)
        self.assertEqual(collections.Counter(r["variant_id"] for r in rows), dict.fromkeys(schema.STYLES, 500))
        for family in schema.REPAIR_FAMILIES:
            group = [r for r in rows if r["family"] == family]
            self.assertEqual(len(group), 500)
            self.assertEqual(collections.Counter(r["strata"]["operand_sign"] for r in group), {"positive": 250, "negative": 250})
        branches = collections.Counter(r["strata"]["conditional_branch"] for r in rows if "conditional_branch" in r["strata"])
        self.assertEqual(branches[True], branches[False])
        self.assertTrue(all(g.partition_of(r["operative_key"]) == "train" for r in rows))
        self.assertFalse({r["operative_key"] for r in rows} & {r["operative_key"] for r in self.dev})
        self.assertFalse({r["world_id"] for r in rows} & {r["world_id"] for r in self.dev})

    def test_counterfactual_pairs_keep_world_and_partition(self):
        rows = g.training_rows({"prompt_coverage": "diverse", "difficulty": "counterfactual", "replay_fraction": 0.0}, 29, [])
        pairs = collections.defaultdict(list)
        for row in rows:
            pairs[row["pair_id"]].append(row)
        self.assertEqual(len(pairs), 1000)
        self.assertEqual(len({r["row_id"] for r in rows}), 2000)
        for pair in pairs.values():
            self.assertEqual(len(pair), 2)
            self.assertEqual(len({r["world_id"] for r in pair}), 1)
            self.assertEqual({g.partition_of(r["operative_key"]) for r in pair}, {"train"})
            if pair[0]["family"] in {"answerability", "grounded_qa"}:
                self.assertEqual({r["answerable"] for r in pair}, {True, False})
            elif pair[0]["family"] == "register" and pair[0]["payload"]["event"]["kind"] == "conditional":
                self.assertEqual({r["strata"]["conditional_branch"] for r in pair}, {True, False})

    def test_replay_exact_count_repeats_and_original_targets(self):
        source = [fixture_replay("ledger"), fixture_replay("register")]
        rows = g.training_rows({"prompt_coverage": "canonical", "difficulty": "regular", "replay_fraction": 0.2}, 17, source)
        replay = [r for r in rows if r.get("is_replay")]
        self.assertEqual(len(rows), 2000)
        self.assertEqual(len(replay), 400)
        self.assertEqual({r["split"] for r in replay}, {"replay"})
        self.assertEqual(collections.Counter(r["family"] for r in replay), {"ledger": 200, "register": 200})
        self.assertEqual(sum(r["strata"]["repeated_replay_source"] for r in replay), 398)
        for row in replay:
            self.assertEqual(g.target_reply(row), row["replay_target"])
            original = next(s for s in source if s["family"] == row["family"])
            self.assertEqual(row["user"], original["user"])
            self.assertEqual(row["system"], original["system"])

    def test_replay_missing_source_or_unverified_gold_is_blocked(self):
        recipe = {"prompt_coverage": "canonical", "difficulty": "regular", "replay_fraction": 0.2}
        with self.assertRaisesRegex(ValueError, "STOP_REPLAY_SOURCE_POOL_EMPTY"):
            g.training_rows(recipe, 17, [])
        source = [fixture_replay("ledger"), fixture_replay("register")]
        source[0]["provenance"]["gold_verified"] = False
        with self.assertRaisesRegex(ValueError, "STOP_REPLAY_GOLD_NOT_INDEPENDENTLY_VERIFIED"):
            g.training_rows(recipe, 17, source)
        source[0]["provenance"]["gold_verified"] = True
        source[0]["replay_target"] = '{"A":0}'
        with self.assertRaisesRegex(ValueError, "STOP_REPLAY_GOLD"):
            g.training_rows(recipe, 17, source)

    def test_supported_and_missing_cross_every_prompt_style(self):
        rows = g.training_rows({"prompt_coverage": "diverse", "difficulty": "regular", "replay_fraction": 0.0}, 17, [])
        for family in ("answerability", "grounded_qa"):
            for style in schema.STYLES:
                self.assertEqual({r["answerable"] for r in rows if r["family"] == family and r["variant_id"] == style}, {True, False})
            self.assertEqual({r["strata"]["scenario"] for r in rows if r["family"] == family}, set(g.SCENARIOS))


if __name__ == "__main__":
    unittest.main()
