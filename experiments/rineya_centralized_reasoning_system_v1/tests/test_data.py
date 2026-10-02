"""Independent hand controls for latent gold, public boundary and scoring."""
import copy
import inspect
import unittest
from experiments.rineya_centralized_reasoning_system_v1 import data, scoring


def e(eid, time, kind, **fields):
    return {"id": eid, "time": time, "kind": kind, **fields}


def world(events, time=5, entity="parcel", complete=True):
    return {"events": events, "query": {"entity": entity, "time": time}, "complete": complete}


class OracleControls(unittest.TestCase):
    def setUp(self):
        self.initial = e("i", 0, "INIT", entity="parcel", owner="Alice")
        self.transfer = e("t", 2, "TRANSFER", entity="parcel", owner="Bob", mode="CONFIRMED")

    def test_confirmed_time_and_order(self):
        self.assertEqual(data.oracle(world([self.transfer, self.initial])), {"status": "SUPPORTED", "value": "Bob"})
        self.assertEqual(data.oracle(world([self.transfer, self.initial], time=1)), {"status": "SUPPORTED", "value": "Alice"})

    def test_proposals_negation_and_cancel_never_bind(self):
        for mode in ("PROPOSED", "NEGATED"):
            t = dict(self.transfer, mode=mode)
            self.assertEqual(data.oracle(world([self.initial, t]))["value"], "Alice")
        events = [self.initial, self.transfer, e("c", 3, "CANCEL", target="t")]
        self.assertEqual(data.oracle(world(events))["value"], "Alice")
        self.assertEqual(data.oracle(world(events, time=2))["value"], "Bob")

    def test_cancel_invalid_reference_fails(self):
        for target in ("missing", "i"):
            with self.assertRaises(ValueError):
                data.oracle(world([self.initial, e("c", 3, "CANCEL", target=target)]))

    def test_unknown_contradiction_and_same_value_duplicate(self):
        alt = dict(self.transfer, id="u", owner="Carol")
        self.assertEqual(data.oracle(world([self.initial, self.transfer, alt])), {"status": "CONTRADICTORY", "value": None})
        self.assertEqual(data.oracle(world([self.initial, self.transfer, dict(self.transfer, id="u")]))["value"], "Bob")
        self.assertEqual(data.oracle(world([self.initial], complete=False))["status"], "UNKNOWN")
        self.assertEqual(data.oracle(world([]))["status"], "UNKNOWN")

    def test_aliases_ambiguity_cycles(self):
        aliases = [e("a", 0, "ALIAS", alias="P", canonical="parcel"), e("b", 0, "ALIAS", alias="B", canonical="Bob")]
        self.assertEqual(data.oracle(world([self.initial, dict(self.transfer, owner="B")] + aliases, entity="P"))["value"], "Bob")
        self.assertEqual(data.oracle(world([self.initial] + aliases + [e("x", 0, "ALIAS", alias="P", canonical="other")], entity="P"))["status"], "UNKNOWN")
        cyc = [e("a", 0, "ALIAS", alias="P", canonical="Q"), e("b", 0, "ALIAS", alias="Q", canonical="P")]
        self.assertEqual(data.oracle(world([self.initial] + cyc, entity="P"))["status"], "UNKNOWN")

    def test_prerequisite_and_exception_polarity(self):
        for kind in ("REQUIRE", "EXCEPT"):
            constraint = e("r", 0, kind, target="t", predicate="permit")
            self.assertEqual(data.oracle(world([self.initial, self.transfer, constraint]))["status"], "UNKNOWN")
            for state in ("SUPPORTED", "REFUTED"):
                fact = e("f", 1, "FACT", predicate="permit", status=state)
                blocked = (kind, state) in (("REQUIRE", "REFUTED"), ("EXCEPT", "SUPPORTED"))
                self.assertEqual(data.oracle(world([self.initial, self.transfer, constraint, fact]))["value"], "Alice" if blocked else "Bob")

    def test_fact_sample_time_and_conflicting_latest(self):
        constraint = e("r", 0, "REQUIRE", target="t", predicate="permit")
        positive = e("f", 1, "FACT", predicate="permit", status="SUPPORTED")
        later = e("g", 3, "FACT", predicate="permit", status="REFUTED")
        self.assertEqual(data.oracle(world([self.initial, self.transfer, constraint, positive, later]))["value"], "Bob")
        negative = dict(later, time=1)
        self.assertEqual(data.oracle(world([self.initial, self.transfer, constraint, positive, negative]))["status"], "CONTRADICTORY")

    def test_source_parser_or_runtime_not_oracle_dependency(self):
        source = inspect.getsource(data.oracle)
        self.assertNotIn("core", source)
        self.assertNotIn("splitlines", source)
        self.assertNotIn("render_event", source)

    def test_empty_is_refuted_and_owner_same_time_contradictory(self):
        empty = e("x", 2, "EMPTY", entity="parcel")
        self.assertEqual(data.oracle(world([self.initial, empty])), {"status": "REFUTED", "value": None})
        self.assertEqual(data.oracle(world([self.initial, empty, self.transfer])), {"status": "CONTRADICTORY", "value": None})


class DatasetControls(unittest.TestCase):
    def test_determinism_exact_counts_and_reader(self):
        bundle = data.build_dataset("DEV", 64, 99)
        self.assertEqual(bundle, data.build_dataset("DEV", 64, 99))
        self.assertEqual(len(bundle["cases"]), 64)
        self.assertEqual(set(bundle["gold"]), {c["id"] for c in bundle["cases"]})
        self.assertEqual(len(data.reader_cases()["cases"]), 16)
        self.assertTrue(all(set(c) == data.PUBLIC_FIELDS for c in bundle["cases"]))

    def test_closed_roles_and_type_confusion(self):
        for split in ("CHECK", "FINAL", "CONFIRM", "test"):
            with self.assertRaises(ValueError):
                data.build_dataset(split, 16, 1)
        for count in (True, 0, 4097):
            with self.assertRaises(ValueError):
                data.build_dataset("DEV", count, 1)

    def test_split_audit_and_known_bad_controls(self):
        train, dev = data.build_dataset("TRAIN", 32, 1), data.build_dataset("DEV", 64, 2)
        self.assertTrue(data.audit_splits(train, dev)["pass"])
        bad = copy.deepcopy(dev)
        bad["cases"][0]["parent_id"] = train["cases"][0]["parent_id"]
        with self.assertRaises(ValueError):
            data.audit_splits(train, bad)
        leaked = copy.deepcopy(dev["cases"][0])
        leaked["scoring_payload"] = {"answer": "Alice"}
        with self.assertRaises(ValueError):
            data.validate_public(leaked)

    def test_baselines_are_executable_and_nonperfect(self):
        bundle = data.build_dataset("DEV", 64, 7)
        results = {name: scoring.score(pred, bundle["gold"]) for name, pred in data.shortcut_baselines(bundle).items()}
        self.assertTrue(all(r["semantic_accuracy"] < 1 for r in results.values()))
        self.assertGreater(results["always_unknown"]["counts"]["erroneous_abstention"], 0)


class ScorerControls(unittest.TestCase):
    def test_four_semantic_states_and_format(self):
        gold = {"p": {"status": "SUPPORTED", "value": "Alice"}, "n": {"status": "REFUTED", "value": None},
                "u": {"status": "UNKNOWN", "value": None}, "c": {"status": "CONTRADICTORY", "value": None}}
        self.assertEqual(scoring.score(gold, gold)["semantic_accuracy"], 1)
        self.assertTrue(scoring.score_one(dict(gold["p"], evidence_ids=["e0"]), gold["p"])["semantic_exact"])
        self.assertFalse(scoring.score_one(dict(gold["p"], evidence_ids=["e0", "e0"]), gold["p"])["format_valid"])
        for bad in ({"status": "SUPPORTED", "value": True}, {"status": "UNKNOWN", "value": "Alice"},
                    {"status": "SUPPORTED", "value": "Alice", "gold": 1}, "Alice"):
            self.assertFalse(scoring.score_one(bad, gold["p"])["format_valid"])

    def test_missing_failure_extra_keep_denominators(self):
        gold = {"a": {"status": "SUPPORTED", "value": "A"}, "b": {"status": "UNKNOWN", "value": None}}
        result = scoring.score({"a": {"error": "OOM"}, "extra": gold["b"]}, gold)
        self.assertEqual(result["scheduled_cases"], 2)
        self.assertEqual(result["semantic_accuracy"], 0)
        self.assertEqual(result["counts"]["operational_error"], 1)
        self.assertEqual(result["counts"]["missing"], 1)
        self.assertEqual(result["extra_prediction_ids"], ["extra"])

    def test_corrupt_gold_raises_and_zero_denominators_undefined(self):
        with self.assertRaises(ValueError):
            scoring.score({}, {"a": {"status": "SUPPORTED", "value": None}})
        r = scoring.score({}, {"u": {"status": "UNKNOWN", "value": None}})
        self.assertIsNone(r["answerable_accuracy"])


if __name__ == "__main__":
    unittest.main()
