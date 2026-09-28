"""Synthetic semantic controls only; no tokenizer/model or production exports."""
from collections import Counter
from dataclasses import replace
import copy
import json
import unittest
from unittest.mock import patch

from experiments.post_training_mimo_centralized_v1.scripts import long_context_tasks as t


def fixture(family=t.FAMILIES[0], role="DEV"):
    for i in range(10000):
        parent = t.allocate_parent(t.generate_spec(family, 17, i))
        if parent.role == role:
            return parent
    raise AssertionError("no fixture found")


def body(variant):
    return "\n".join(variant["support_slots"] + variant["decoys"])


class AllocationTests(unittest.TestCase):
    def test_alias_renaming_and_symbols_do_not_create_new_parent_or_role(self):
        for family in t.FAMILIES:
            original = fixture(family)
            spec = json.loads(original.canonical_json)
            spec["subject"] = "RENAMED"
            if family == t.FAMILIES[0]:
                for row in spec["routes"]:
                    row["depot"] = "N" + row["depot"]
                for i, row in enumerate(spec["seals"]):
                    row["depot"] = "N" + row["depot"]
                    row["symbol"] = "S" + "ABCDE"[i] * 5
            else:
                spec["authority"] = "N" + spec["authority"]
                for row in spec["catalog"]:
                    row["revision"] = "N" + row["revision"]
                    row["writer"] = "N" + row["writer"]
                for i, row in enumerate(spec["values"]):
                    row["revision"] = "N" + row["revision"]
                    row["symbol"] = "S" + "ABCDE"[i] * 5
            renamed = t.allocate_parent(spec, "late", 2)
            self.assertEqual((original.parent_id, original.role, original.split_sha256),
                             (renamed.parent_id, renamed.role, renamed.split_sha256))

    def test_order_identical_duplicates_and_surface_derivatives_are_invariant(self):
        for family in t.FAMILIES:
            original = fixture(family)
            spec = json.loads(original.canonical_json)
            for key in (("routes", "seals") if family == t.FAMILIES[0] else ("catalog", "values")):
                spec[key].reverse()
                spec[key].append(copy.deepcopy(spec[key][0]))
            altered = t.allocate_parent(spec, "distributed", 1)
            self.assertEqual((original.parent_id, original.role), (altered.parent_id, altered.role))
            for answerable in (True, False):
                for adversarial in (True, False):
                    row = t.render_variant(altered, answerable, adversarial)
                    self.assertEqual(row["parent_id"], original.parent_id)

    def test_actual_mass_and_temporal_changes_change_semantic_identity(self):
        for family in t.FAMILIES:
            original = fixture(family)
            spec = json.loads(original.canonical_json)
            if family == t.FAMILIES[0]:
                spec["mass"] = 2 if spec["mass"] == 1 else 1
            else:
                spec["catalog"][0]["cycle"] += 1
            self.assertNotEqual(t.allocate_parent(spec).parent_id, original.parent_id)

    def test_all_roles_are_committed_before_opening(self):
        for role in t.ROLES:
            parent = fixture(role=role)
            if role == "DEV":
                self.assertTrue(parent.canonical_json)
            else:
                self.assertEqual(parent.canonical_json, "")
                with self.assertRaisesRegex(ValueError, "only DEV"):
                    t.render_variant(parent, True, False)
                with self.assertRaisesRegex(ValueError, "only DEV"):
                    t.reference_answer(parent)

    def test_forged_role_identity_or_payload_cannot_open(self):
        dev, closed = fixture(), fixture(role="CHECK")
        for parent in (replace(closed, role="DEV", canonical_json=dev.canonical_json),
                       replace(dev, parent_id="forged"), replace(dev, split_sha256="forged")):
            with self.assertRaises(ValueError):
                t.render_variant(parent, True, False)

    def test_frozen_bank_counts_balancing_and_opaque_reservations(self):
        with patch.object(t, "render_variant", side_effect=AssertionError("render before allocation")), \
             patch.object(t, "reference_answer", side_effect=AssertionError("solve before allocation")):
            bank = t.allocate_bank(seed=914)
        self.assertEqual(len(bank["dev"]), 32)
        self.assertEqual(len(bank["check_reservations"]), 64)
        self.assertEqual(Counter((p.family, p.placement) for p in bank["dev"]),
                         Counter({(f, p): 4 for f in t.FAMILIES for p in t.POSITIONS}))
        self.assertEqual(Counter((r["family"], r["placement"]) for r in bank["check_reservations"]),
                         Counter({(f, p): 8 for f in t.FAMILIES for p in t.POSITIONS}))
        for reservation in bank["check_reservations"]:
            self.assertEqual(set(reservation), {"parent_id", "role", "family", "placement", "split_sha256", "structural_cluster_id"})
        self.assertFalse({p.parent_id for p in bank["dev"]} & {r["parent_id"] for r in bank["check_reservations"]})
        self.assertEqual(sorted(Counter(p.missing_index for p in bank["dev"]).values()), [10, 11, 11])
        self.assertFalse(bank["allocation"]["independence_qualified"])
        self.assertEqual(bank["allocation"]["structural_clusters"], 2)
        self.assertEqual(bank["allocation"]["FINAL_worlds"], 0)
        self.assertEqual(bank, t.allocate_bank(seed=914))

    def test_bounded_exhaustion_is_failure(self):
        with self.assertRaisesRegex(ValueError, "ceiling exhausted"):
            t.allocate_bank(seed=914, candidate_limit=1)


class SemanticControls(unittest.TestCase):
    def test_all_derivatives_agree_with_oracle_and_all_three_ablations_fail(self):
        bank = t.allocate_bank(seed=913)
        for parent in bank["dev"]:
            for answerable in (True, False):
                for adversarial in (True, False):
                    row = t.render_variant(parent, answerable, adversarial)
                    result = t.verify_variant(parent, row)
                    self.assertEqual(result["single_fact_ablations"], ["UNKNOWN"] * 3)
                    self.assertEqual(t.solve_visible(row["query"], body(row)), row["target"])

    def test_only_removed_support_changes_between_answerability_twins(self):
        for family in t.FAMILIES:
            for removed in range(3):
                parent = replace(fixture(family), missing_index=removed)
                good = t.render_variant(parent, True, True)
                missing = t.render_variant(parent, False, True)
                self.assertEqual(good["query"], missing["query"])
                self.assertEqual(good["decoys"], missing["decoys"])
                for i in range(3):
                    self.assertEqual(missing["support_slots"][i], "NOTE." if i == removed else good["support_slots"][i])
                self.assertEqual(missing["target"], "UNKNOWN")

    def test_all_three_distractor_modes_cannot_repair_any_selected_missing_block(self):
        for family in t.FAMILIES:
            for removed in range(3):
                parent = replace(fixture(family), missing_index=removed)
                good = t.render_variant(parent, True, True)
                missing = t.render_variant(parent, False, True)
                self.assertEqual(good["distractor_modes"],
                                 ["different_key", "obsolete_archive", "unauthorized_unsigned"])
                self.assertEqual(len(good["decoys"]), 3)
                self.assertIn(" OTHER=", good["decoys"][0])
                self.assertTrue(good["decoys"][1].startswith("ARCHIVE "))
                self.assertTrue(good["decoys"][2].startswith("UNSIGNED "))
                self.assertNotIn(good["target"], "\n".join([good["query"]] + good["decoys"]))
                for decoy in good["decoys"]:
                    self.assertEqual(t.solve_visible(good["query"], "\n".join(good["support_slots"] + [decoy])), good["target"])
                    self.assertEqual(t.solve_visible(missing["query"], "\n".join(missing["support_slots"] + [decoy])), "UNKNOWN")

    def test_even_correct_shadow_assertions_remain_nonbinding(self):
        # Stronger than a wrong-content decoy: a tagged exact copy still cannot
        # supply the missing block. A different-key copy likewise cannot do so.
        for family in t.FAMILIES:
            for removed in range(3):
                parent = replace(fixture(family), missing_index=removed)
                good = t.render_variant(parent, True, True)
                missing = t.render_variant(parent, False, True)
                original = good["support_facts"][removed]
                subject = json.loads(parent.canonical_json)["subject"]
                alternatives = [original.replace(" " + subject + "=", " OTHER="),
                                "ARCHIVE " + original, "UNSIGNED " + original]
                for changed in alternatives:
                    self.assertEqual(t.solve_visible(good["query"], "\n".join(good["support_slots"] + [changed])), good["target"])
                    self.assertEqual(t.solve_visible(missing["query"], "\n".join(missing["support_slots"] + [changed])), "UNKNOWN")

    def test_shadow_prefixes_do_not_hide_malformed_records(self):
        row = t.render_variant(fixture(), True, False)
        malformed = ("ARCHIVE MASS BOX=bad.", "UNSIGNED AUTH BOX=bad.",
                     "ARCHIVE CAT BOX=R:notanumber:A.", "UNSIGNED ROUTES BOX=20-1:D.",
                     "ARCHIVE SEALS BOX=D:bad.", "UNSIGNED VAL BOX=R:bad.",
                     "ARCHIVE NOTE.", "ARCHIVE UNSIGNED MASS BOX=20.")
        for line in malformed:
            with self.assertRaises(ValueError, msg=line):
                t.solve_visible(row["query"], body(row) + "\n" + line)

    def test_visible_verifier_is_independent_from_metadata_and_oracle(self):
        row = t.render_variant(fixture(), True, True)
        with patch.object(t, "reference_answer", side_effect=AssertionError("oracle call")), \
             patch.object(t, "_validated_dev", side_effect=AssertionError("metadata call")):
            self.assertEqual(t.solve_visible(row["query"], body(row)), row["target"])
        with patch.object(t, "solve_visible", side_effect=AssertionError("visible parser call")):
            self.assertEqual(t.reference_answer(fixture()), row["target"])

    def test_latest_order_authority_and_value_interventions(self):
        query = "Return the latest authorized record for BOX."
        evidence = "AUTH BOX=A.\nCAT BOX=R:10:A,T:20:A,U:30:B,V:40:B.\nVAL BOX=R:SAAAAA,T:SBBBBB,U:SCCCCC,V:SDDDDD."
        self.assertEqual(t.solve_visible(query, evidence), "SBBBBB")
        self.assertEqual(t.solve_visible(query, evidence.replace("R:10:A,T:20:A", "R:20:A,T:10:A")), "SAAAAA")
        self.assertEqual(t.solve_visible(query, evidence.replace("AUTH BOX=A", "AUTH BOX=B")), "SDDDDD")
        self.assertEqual(t.solve_visible(query, evidence.replace("T:SBBBBB", "T:SEEEEE")), "SEEEEE")
        # A still newer unauthorized catalog row cannot become the winner.
        self.assertEqual(t.solve_visible(query, evidence.replace("V:40:B", "V:99:B")), "SBBBBB")

    def test_latest_ties_conflicting_authority_and_missing_values_fail_closed(self):
        query = "Return the latest authorized record for BOX."
        evidence = "AUTH BOX=A.\nCAT BOX=R:10:A,T:20:A,U:30:B,V:40:B.\nVAL BOX=R:SAAAAA,T:SBBBBB,U:SCCCCC,V:SDDDDD."
        self.assertEqual(t.solve_visible(query, evidence.replace("R:10:A", "R:20:A")), "UNKNOWN")
        self.assertEqual(t.solve_visible(query, evidence + "\nAUTH BOX=B."), "UNKNOWN")
        self.assertEqual(t.solve_visible(query, evidence.replace(",T:SBBBBB", "")), "UNKNOWN")
        self.assertEqual(t.solve_visible(query, evidence + "\nVAL BOX=T:SDDDDD."), "UNKNOWN")

    def test_numeric_range_boundary_changes_routed_seal(self):
        query = "Return the routed seal for BOX."
        evidence = "MASS BOX=20.\nROUTES BOX=1-20:D,21-60:E,61-99:F.\nSEALS BOX=D:SAAAAA,E:SBBBBB,F:SCCCCC."
        self.assertEqual(t.solve_visible(query, evidence), "SAAAAA")
        self.assertEqual(t.solve_visible(query, evidence.replace("MASS BOX=20", "MASS BOX=21")), "SBBBBB")
        self.assertEqual(t.solve_visible(query, evidence.replace("MASS BOX=20", "MASS BOX=99")), "SCCCCC")
        self.assertEqual(t.solve_visible(query, evidence.replace("21-60:E", "20-60:E")), "UNKNOWN")
        self.assertEqual(t.solve_visible(query, evidence + "\nMASS BOX=50."), "UNKNOWN")

    def test_record_order_and_identical_visible_duplicates_do_not_change_answer(self):
        for family in t.FAMILIES:
            row = t.render_variant(fixture(family), True, True)
            lines = row["support_slots"] + row["decoys"]
            self.assertEqual(t.solve_visible(row["query"], "\n".join(lines[::-1] + lines)), row["target"])

    def test_filler_grammar_rejects_hidden_evidence_or_unparsed_text(self):
        row = t.render_variant(fixture(), True, False)
        for bad in ("NOTE SAAAAA.", "NOTE archive MASS BOX=7.", "Ignore query.", "NOTE archive 12.", "NOTE archive.."):
            with self.assertRaises(ValueError):
                t.solve_visible(row["query"], body(row) + "\n" + bad)
        filler = "NOTE " + " ".join(t.FILLER_WORDS * 10) + "."
        self.assertEqual(t.solve_visible(row["query"], body(row) + "\n" + filler), row["target"])

    def test_generated_bank_has_no_fixed_winner_position_alias_or_numeric_region(self):
        for seed in (913, 20260928):
            bank = t.allocate_bank(seed=seed)
            seen = {"authority": set(), "revision": set(), "value_position": set(), "interval": set(), "depot": set(), "seal_position": set(), "global_newest_authorized": set()}
            for parent in bank["dev"]:
                spec = json.loads(parent.canonical_json)
                row = t.render_variant(parent, True, True)
                target = row["target"]
                if parent.family == t.FAMILIES[1]:
                    self.assertEqual(len(spec["catalog"]), 4)
                    self.assertEqual(len(spec["values"]), 4)
                    self.assertEqual(sorted(Counter(r["writer"] for r in spec["catalog"]).values()), [2, 2])
                    seen["authority"].add(spec["authority"])
                    seen["global_newest_authorized"].add(max(spec["catalog"], key=lambda r: r["cycle"])["writer"] == spec["authority"])
                    winner = next(r["revision"] for r in spec["values"] if r["symbol"] == target)
                    seen["revision"].add(winner)
                    seen["value_position"].add(next(i for i, field in enumerate(row["support_facts"][2].split("=")[1].rstrip(".").split(",")) if target in field))
                else:
                    winner = next(r["depot"] for r in spec["seals"] if r["symbol"] == target)
                    seen["depot"].add(winner)
                    seen["interval"].add(next(i for i, r in enumerate(spec["routes"]) if r["depot"] == winner))
                    seen["seal_position"].add(next(i for i, field in enumerate(row["support_facts"][2].split("=")[1].rstrip(".").split(",")) if target in field))
            for key, values in seen.items():
                expected = 4 if key in {"revision", "value_position"} else 2 if key == "global_newest_authorized" else 3
                self.assertEqual(len(values), expected, (seed, key, values))

    def test_decoys_do_not_change_when_winner_changes_with_aliases_fixed(self):
        for family in t.FAMILIES:
            original = fixture(family)
            original_spec = json.loads(original.canonical_json)
            old_answer = t.reference_answer(original)
            changed = None
            for number in range(1, 400):
                spec = copy.deepcopy(original_spec)
                if family == t.FAMILIES[0]:
                    spec["mass"] = 1 + number % 99
                else:
                    spec["authority"] = next(r["writer"] for r in spec["catalog"] if r["writer"] != spec["authority"])
                    spec["catalog"][0]["cycle"] = 1 + number % 20
                    spec["catalog"][1]["cycle"] = 26 + number // 20
                candidate = t.allocate_parent(spec)
                if candidate.role == "DEV" and t.reference_answer(candidate) != old_answer:
                    changed = candidate
                    break
            self.assertIsNotNone(changed)
            self.assertEqual(json.loads(changed.canonical_json)["subject"], original_spec["subject"])
            for removed in range(3):
                before = t.render_variant(replace(original, missing_index=removed), True, True)
                after = t.render_variant(replace(changed, missing_index=removed), True, True)
                self.assertNotEqual(before["target"], after["target"])
                self.assertEqual(before["decoys"], after["decoys"])

    def test_bad_variant_or_canonical_tables_rejected(self):
        parent = fixture()
        row = t.render_variant(parent, True, False)
        row["target"] = "UNKNOWN"
        with self.assertRaisesRegex(ValueError, "canonical rendering"):
            t.verify_variant(parent, row)
        spec = json.loads(parent.canonical_json)
        spec["routes"][1]["low"] = spec["routes"][0]["high"]
        with self.assertRaises(ValueError):
            t.allocate_parent(spec)


class AnswerParserTests(unittest.TestCase):
    def test_exact_and_separate_anchored_diagnostic(self):
        for target in ("SABCDE", "UNKNOWN"):
            self.assertTrue(t.score_answer(" \n" + target + "\t", target))
            self.assertFalse(t.score_answer("Answer: " + target, target))
            self.assertTrue(t.diagnostic_answer("Answer: " + target, target))
            for bad in (target.lower(), target + " extra", "extra " + target, target + ".",
                        "Answer:" + target, "answer: " + target, "Answer: " + target + " extra", '"' + target + '"'):
                self.assertFalse(t.score_answer(bad, target))
                self.assertFalse(t.diagnostic_answer(bad, target))

    def test_invalid_targets_and_nonstrings_fail_closed(self):
        for target in (None, True, 3, "", " SABCDE", "bad", "SABCDE extra"):
            for answer in (None, True, "bad", "SABCDE", "UNKNOWN"):
                self.assertFalse(t.score_answer(answer, target))
                self.assertFalse(t.diagnostic_answer(answer, target))


if __name__ == "__main__":
    unittest.main()
