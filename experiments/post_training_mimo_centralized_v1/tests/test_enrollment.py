"""Synthetic contract fixtures only: no collector, model outcomes or human ratings."""
import base64
import contextlib
from copy import deepcopy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from experiments.post_training_mimo_centralized_v1.scripts import enrollment as e


CALLS = []


def toy_scorer(row, raw_response):
    """Fixture-only deterministic scorer; not native or language qualification."""
    CALLS.append(row["row_id"])
    if row["panel"] == "long_context":
        well_formed = raw_response in {"SABCDE", "UNKNOWN"}
        return {"correct": well_formed and raw_response == row["scoring_payload"]["answer"],
                "malformed": not well_formed}
    try:
        parsed = json.loads(raw_response, object_pairs_hook=e._unique_object)
        well_formed = type(parsed) is dict and set(parsed) == {"answer"}
        return {"correct": well_formed and type(parsed["answer"]) is type(row["scoring_payload"]["answer"])
                and parsed["answer"] == row["scoring_payload"]["answer"], "malformed": not well_formed}
    except (ValueError, TypeError):
        return {"correct": False, "malformed": True}


def failing_scorer(row, raw_response):
    raise RuntimeError("synthetic scorer failure")


def invalid_scorer(row, raw_response):
    return {"correct": 1, "malformed": False}


def identity(candidate=False):
    return {"model_id": "SYNTHETIC_CONTRACT_FIXTURE" + ("_CANDIDATE" if candidate else "_SOURCE"),
            "weight_sha256": "b" * 64 if candidate else "a" * 64,
            "tokenizer_sha256": "c" * 64, "chat_template_sha256": "d" * 64, "runtime_sha256": "e" * 64}


def row(parent="fixture-parent", variant="standard", panel="acquisition", family="fixture-family"):
    answer = None if variant == "missing_evidence" else "yes"
    if panel == "long_context":
        answer = "UNKNOWN" if "-unanswerable_" in variant else "SABCDE"
    row_id = parent + "-" + variant
    return {"row_id": row_id, "parent_id": parent, "family": family, "stratum": "early",
            "variant": variant, "role": "DEV", "panel": panel, "prompt_token_hash": e.digest(row_id),
            "scoring_payload": {"answer": answer}}


def manifest(rows=None, seeds=None, scorer=toy_scorer):
    return {"schema_version": 1, "status": "FROZEN", "role": "DEV", "frozen_at_utc": "2026-01-01T00:00:00Z",
            "source_identity": identity(), "bank_sha256": "f" * 64, "seeds": seeds or [101],
            "eos_token_ids": [99], "decoding": {"max_new_tokens": 24, "do_sample": False},
            "scorer": e.scorer_binding(scorer, sorted(e.PANELS)), "rows": rows or [row()]}


def responses(m, source=None):
    ident = identity(source is not None)
    records = []
    for r in m["rows"]:
        for seed in m["seeds"]:
            raw = r["scoring_payload"]["answer"] if r["panel"] == "long_context" else json.dumps({"answer": r["scoring_payload"]["answer"]})
            records.append({"row_id": r["row_id"], "seed": seed, "prompt_token_hash": r["prompt_token_hash"],
                            "identity": deepcopy(ident), "raw_response": raw, "generated_token_ids": [42, 99],
                            "prompt_tokens": int(r["variant"].split("-", 1)[0]) if r["panel"] == "long_context" else 20,
                            "completion_tokens": 2, "eos_reached": True, "stop_reason": "eos", "failure": None,
                            "cost": {"wall_seconds": 1.0, "gpu_seconds": 0.5, "usd": 0}})
    return {"schema_version": 1, "role": "DEV", "endpoint": "candidate" if source else "source",
            "manifest_sha256": e.digest(m), "bank_sha256": m["bank_sha256"], "decoding_sha256": e.digest(m["decoding"]),
            "identity": ident, "started_at_utc": "2099-01-01T00:00:00Z" if source else "2026-01-02T00:00:00Z",
            "finished_at_utc": "2099-01-01T00:00:01Z" if source else "2026-01-02T00:00:01Z",
            "source_enrollment_sha256": source["receipt_sha256"] if source else None, "records": records}


def twins(parent="fixture-twins"):
    return [row(parent, variant, "deterministic_retention", "answerability_abstention")
            for variant in ("answerable", "missing_evidence")]


def bundle(m, r, kind="native_response_collection"):
    return {"schema_version": 1, "kind": kind, "status": "PASS", "context_sha256": e.digest(e.qualification_context(m, r)),
            "evidence_sha256": "1" * 64, "collector_code_sha256": "2" * 64, "qualification_receipt_sha256": "3" * 64}


def fixture_evidence_verifier(evidence, context):
    # Only exercises the callback interface. No production bundle is created.
    return evidence["context_sha256"] == e.digest(context)


class EnrollmentTests(unittest.TestCase):
    def setUp(self):
        CALLS.clear()

    def test_import_preserves_raw_response_and_grants_no_native_permission(self):
        m = manifest()
        r = responses(m)
        r["records"][0]["raw_response"] = '  {"answer":"yes"} \n'
        original = deepcopy(r)
        result = e.enroll(m, r)
        self.assertEqual(result["status"], e.STRUCTURE_STATUS)
        self.assertEqual(result["response_document"], original)
        self.assertEqual(result["outcomes"][0]["status"], "UNMEASURED")
        self.assertEqual(CALLS, [])
        self.assertFalse(result["native_execution_permission"])
        self.assertFalse(result["training_permission"])
        self.assertEqual(result["receipt_sha256"], e.digest({k: v for k, v in result.items() if k != "receipt_sha256"}))
        result["response_document"]["records"][0]["raw_response"] = "mutated"
        self.assertEqual(r, original)

    def test_repeated_parents_seeds_and_twins_stay_inside_parent(self):
        m = manifest(twins() + twins("fixture-twins-two"), [101, 211])
        r = responses(m)
        r["records"][0]["raw_response"] = '{"answer":null}'
        result = e.enroll(m, r, scorer=toy_scorer)
        self.assertEqual(result["parent_cluster_count"], 2)
        self.assertEqual(result["accounting"]["attempts"], 8)
        self.assertEqual([len(p["derivatives"]) for p in result["parent_groups"]], [2, 2])
        first = result["parent_groups"][0]["parent_observations"]
        self.assertEqual(first[0]["mean_correct"], .5)
        self.assertFalse(first[0]["both_twins_correct"])
        self.assertEqual(first[0]["false_abstentions"], 1)
        self.assertTrue(first[1]["both_twins_correct"])
        self.assertEqual(len(result["source_membership"]), 8)
        self.assertEqual(result["status"], e.STRUCTURE_STATUS)

    def test_long_context_keeps_eight_variants_and_extension_scope(self):
        m = manifest([row("fixture-lc", variant, "long_context", "three_fact_keyed_join")
                      for variant in sorted(e.LONG_CONTEXT_VARIANTS)])
        r = responses(m)
        result = e.enroll(m, r, scorer=toy_scorer)
        self.assertEqual(result["parent_cluster_count"], 1)
        self.assertEqual(len(result["parent_groups"][0]["derivatives"]), 8)
        context = e.qualification_context(m, r)
        self.assertEqual(context["required_contexts"], [2048, 4096])
        self.assertEqual(context["maximum_required_context_tokens"], 4096)
        self.assertTrue(context["long_context_zero_update_extension_required"])
        bad = deepcopy(r)
        bad["records"][0]["prompt_tokens"] = 1024
        with self.assertRaisesRegex(ValueError, "length drift"):
            e.enroll(m, bad)
        m["rows"].pop()
        with self.assertRaisesRegex(ValueError, "eight variants"):
            e.validate_manifest(m)

    def test_malformed_eos_and_runtime_failures_are_preserved_with_all_costs(self):
        m = manifest([row("fixture-a"), row("fixture-b"), row("fixture-c")])
        r = responses(m)
        r["records"][0]["raw_response"] = '{"answer":"yes","answer":"yes"}'
        r["records"][1].update(generated_token_ids=[42], completion_tokens=1, eos_reached=False, stop_reason="length")
        r["records"][2].update(raw_response="partial", generated_token_ids=[], completion_tokens=0,
                                eos_reached=False, stop_reason="error", failure="synthetic memory failure")
        result = e.enroll(m, r, scorer=toy_scorer)
        self.assertEqual([o["correct"] for o in result["outcomes"]], [False, False, False])
        self.assertTrue(result["outcomes"][0]["malformed"])
        self.assertTrue(result["outcomes"][1]["semantic_correct"])
        self.assertEqual(result["accounting"], {"attempts": 3, "failed_attempts": 1, "eos_failures": 2,
                                               "prompt_tokens": 60, "completion_tokens": 3,
                                               "cost": {"wall_seconds": 3.0, "gpu_seconds": 1.5, "usd": 0}})
        self.assertEqual(result["response_document"], r)

    def test_missing_duplicate_and_unexpected_responses_rejected_before_scoring(self):
        m = manifest(twins())
        original = responses(m)
        for kind in ("missing", "duplicate", "unexpected"):
            r = deepcopy(original)
            if kind == "missing":
                r["records"].pop()
            elif kind == "duplicate":
                r["records"].append(deepcopy(r["records"][0]))
            else:
                r["records"][0]["row_id"] = "invented"
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                e.enroll(m, r, scorer=toy_scorer)
        self.assertEqual(CALLS, [])
        m["rows"].pop()
        with self.assertRaisesRegex(ValueError, "missing twin"):
            e.validate_manifest(m)

    def test_duplicate_expected_rows_and_parent_family_drift_rejected(self):
        m = manifest(twins())
        m["rows"].append(deepcopy(m["rows"][0]))
        with self.assertRaisesRegex(ValueError, "duplicate expected"):
            e.validate_manifest(m)
        m = manifest(twins())
        m["rows"][1]["family"] = "different"
        with self.assertRaisesRegex(ValueError, "parent family"):
            e.validate_manifest(m)

    def test_identity_hash_schema_and_decoding_drift_rejected(self):
        m = manifest()
        for field in ("model_id", "weight_sha256", "tokenizer_sha256", "chat_template_sha256", "runtime_sha256"):
            r = responses(m)
            r["identity"][field] = "wrong" if field == "model_id" else "0" * 64
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "source model identity drift"):
                e.enroll(m, r)
        for field in ("manifest_sha256", "bank_sha256", "decoding_sha256"):
            r = responses(m)
            r[field] = "0" * 64
            with self.subTest(field=field), self.assertRaises(ValueError):
                e.enroll(m, r)
        r = responses(m)
        r["records"][0]["identity"]["weight_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "per-record"):
            e.enroll(m, r)
        r = responses(m)
        r["records"][0]["prompt_token_hash"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "prompt token identity"):
            e.enroll(m, r)
        r = responses(m)
        r["native_qualified"] = True
        with self.assertRaisesRegex(ValueError, "unexpected fields"):
            e.enroll(m, r)

    def test_strict_token_eos_and_cost_accounting(self):
        m = manifest()
        mutations = [{"completion_tokens": 3}, {"generated_token_ids": [True, 99]},
                     {"eos_reached": 1}, {"stop_reason": "length"},
                     {"generated_token_ids": [99, 42]}, {"generated_token_ids": [99, 99]},
                     {"seed": True}, {"prompt_tokens": 0}, {"failure": ""},
                     {"cost": {"wall_seconds": float("nan"), "gpu_seconds": 0, "usd": 0}},
                     {"cost": {"wall_seconds": 0, "gpu_seconds": True, "usd": 0}},
                     {"generated_token_ids": [42] * 25 + [99], "completion_tokens": 26}]
        for mutation in mutations:
            r = responses(m)
            r["records"][0].update(mutation)
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                e.enroll(m, r)

    def test_check_final_and_other_roles_cannot_be_imported(self):
        for role in ("CHECK", "FINAL", "TRAIN", "TRAIN_SELECTION", "DEV/CAL"):
            m = manifest()
            m["role"] = role
            with self.subTest(role=role), self.assertRaisesRegex(ValueError, "only frozen DEV"):
                e.enroll(m, responses(m))
            m = manifest()
            m["rows"][0]["role"] = role
            with self.assertRaises(ValueError):
                e.enroll(m, responses(m))

    def test_scorer_binding_and_failures_cannot_disappear(self):
        m = manifest()
        m["scorer"]["code_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "scorer code"):
            e.enroll(m, responses(m), scorer=toy_scorer)
        for callback in (failing_scorer, invalid_scorer):
            m = manifest(scorer=callback)
            r = responses(m)
            result = e.enroll(m, r, scorer=callback, qualification_bundle=bundle(m, r),
                              qualification_verifier=fixture_evidence_verifier)
            self.assertEqual(result["outcomes"][0]["status"], "SCORER_FAILED")
            self.assertEqual(result["status"], e.STRUCTURE_STATUS)
            self.assertIsNone(result["source_membership"][0]["correct"])
            self.assertEqual(len(result["response_document"]["records"]), 1)

    def test_language_remains_blocked_even_when_callbacks_claim_qualification(self):
        m = manifest([row(panel="language")])
        r = responses(m)
        result = e.enroll(m, r, scorer=toy_scorer)
        self.assertEqual(CALLS, [])
        self.assertEqual(result["outcomes"][0]["status"], "BLOCKED")
        self.assertIsNone(result["outcomes"][0]["correct"])
        for callback in (None, fixture_evidence_verifier, lambda evidence, context: True,
                         lambda evidence, context: False):
            result = e.enroll(m, r, scorer=toy_scorer,
                              qualification_bundle=bundle(m, r), qualification_verifier=fixture_evidence_verifier,
                              language_qualification_bundle=bundle(m, r, "language_grader"),
                              language_qualification_verifier=callback)
            self.assertEqual(result["status"], e.STRUCTURE_STATUS)
            self.assertEqual(result["outcomes"][0]["status"], "BLOCKED")
            self.assertIsNone(result["outcomes"][0]["correct"])
            self.assertIsNone(result["outcomes"][0]["semantic_correct"])
            self.assertIsNone(result["outcomes"][0]["malformed"])
            self.assertFalse(result["language_grader_qualified"])
            self.assertFalse(result["native_qualification_verified_externally"])
            self.assertIn("four-dimension calibrated", result["outcomes"][0]["reason"])
        self.assertEqual(CALLS, [])

    def test_declared_bundle_and_arbitrary_callback_never_promote_qualification(self):
        m = manifest()
        r = responses(m)
        evidence = bundle(m, r)
        result = e.enroll(m, r, scorer=toy_scorer, qualification_bundle=evidence)
        self.assertIsNone(result["qualification_claims"][0]["callback_assertion"])
        self.assertEqual(result["status"], e.STRUCTURE_STATUS)
        bad = deepcopy(evidence)
        bad["context_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "not bound"):
            e.enroll(m, r, scorer=toy_scorer, qualification_bundle=bad,
                     qualification_verifier=fixture_evidence_verifier)
        for callback in (fixture_evidence_verifier, lambda evidence, context: True,
                         lambda evidence, context: False, lambda evidence, context: 1):
            result = e.enroll(m, r, scorer=toy_scorer, qualification_bundle=evidence,
                              qualification_verifier=callback)
            self.assertEqual(result["status"], e.STRUCTURE_STATUS)
            self.assertEqual(result["qualification_claims"][0]["status"], "UNVERIFIED_DIAGNOSTIC_CLAIM")
            self.assertEqual(result["qualification_claims"][0]["bundle"], evidence)
            self.assertFalse(result["native_qualification_verified_externally"])
            self.assertFalse(result["native_capture_qualified"])
            self.assertFalse(result["language_grader_qualified"])
            self.assertTrue(result["diagnostic_scoring_only"])
            self.assertFalse(result["native_execution_permission"])
            self.assertFalse(result["training_permission"])
            self.assertFalse(result["automatic_promotion"])
            self.assertTrue(result["outcomes"][0]["correct"])
        self.assertIsNotNone(result["qualification_claims"][0]["callback_error"])

    def test_candidate_requires_scored_prior_source_and_expected_identity(self):
        m = manifest()
        unscored = e.enroll(m, responses(m))
        scored = e.enroll(m, responses(m), scorer=toy_scorer)
        r = responses(m, scored)
        for source, expected_identity in ((None, identity(True)), (unscored, identity(True)), (scored, None)):
            with self.subTest(source=source is None), self.assertRaises(ValueError):
                e.enroll(m, r, scorer=toy_scorer, source_enrollment=source, candidate_identity=expected_identity)
        r["started_at_utc"] = "2026-01-01T00:00:00Z"
        with self.assertRaisesRegex(ValueError, "predates source"):
            e.enroll(m, r, scorer=toy_scorer, source_enrollment=scored, candidate_identity=identity(True))
        r = responses(m, scored)
        r["identity"]["weight_sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "candidate model identity"):
            e.enroll(m, r, scorer=toy_scorer, source_enrollment=scored, candidate_identity=identity(True))

    def test_source_and_candidate_stay_diagnostic_even_with_claimed_qualification(self):
        m = manifest()
        base = responses(m)
        source = e.enroll(m, base, scorer=toy_scorer, qualification_bundle=bundle(m, base),
                          qualification_verifier=lambda evidence, context: True)
        candidate = responses(m, source)
        result = e.enroll(m, candidate, scorer=toy_scorer, source_enrollment=source, candidate_identity=identity(True),
                          qualification_bundle=bundle(m, candidate), qualification_verifier=lambda evidence, context: True)
        self.assertEqual(result["status"], e.STRUCTURE_STATUS)
        self.assertFalse(result["native_capture_qualified"])
        self.assertFalse(result["native_qualification_verified_externally"])
        self.assertEqual(result["paired_summary"]["source_correct_denominator"], 1)
        self.assertEqual(result["paired_summary"]["source_correct_losses"], 0)
        self.assertEqual(result["source_membership_sha256"], source["source_membership_sha256"])
        source["status"] = "BASE_ENROLLMENT"
        source["native_qualification_verified_externally"] = True
        source["receipt_sha256"] = e.digest({k: v for k, v in source.items() if k != "receipt_sha256"})
        with self.assertRaisesRegex(ValueError, "unqualified structural import"):
            e.enroll(m, responses(m, source), scorer=toy_scorer, source_enrollment=source, candidate_identity=identity(True))

    def test_source_membership_is_frozen_zero_denominator_is_undefined(self):
        m = manifest()
        base = responses(m)
        base["records"][0]["raw_response"] = '{"answer":"no"}'
        source = e.enroll(m, base, scorer=toy_scorer)
        original = deepcopy(source)
        result = e.enroll(m, responses(m, source), scorer=toy_scorer, source_enrollment=source,
                          candidate_identity=identity(True))
        self.assertEqual(result["source_membership"], source["source_membership"])
        self.assertEqual(source, original)
        self.assertEqual(result["paired_summary"]["source_correct_denominator"], 0)
        self.assertIsNone(result["paired_summary"]["source_correct_loss_rate"])
        self.assertEqual(result["paired_summary"]["source_incorrect_repair_rate"], 1)
        paired_observation = result["paired_parent_groups"][0]["derivatives"][0]["paired_observations"][0]
        self.assertFalse(paired_observation["base_correct"])
        self.assertTrue(paired_observation["candidate_correct"])
        self.assertFalse(paired_observation["base_eos_failure"])
        self.assertFalse(paired_observation["candidate_malformed"])
        source["source_membership"][0]["correct"] = True
        with self.assertRaisesRegex(ValueError, "source enrollment hash drift"):
            e.enroll(m, responses(m, source), source_enrollment=source, candidate_identity=identity(True))

    def test_rehashed_membership_cannot_drift_from_preserved_source_outcomes(self):
        m = manifest()
        source = e.enroll(m, responses(m), scorer=toy_scorer)
        source["source_membership"][0]["correct"] = False
        source["source_membership_sha256"] = e.digest(source["source_membership"])
        source["receipt_sha256"] = e.digest({k: v for k, v in source.items() if k != "receipt_sha256"})
        with self.assertRaisesRegex(ValueError, "outcomes/membership drift"):
            e.enroll(m, responses(m, source), source_enrollment=source, candidate_identity=identity(True))

    def test_losses_repairs_and_family_regressions_remain_separate(self):
        m = manifest([row("fixture-loss", family="loss"), row("fixture-repair", family="repair")])
        base = responses(m)
        base["records"][1]["raw_response"] = '{"answer":"no"}'
        source = e.enroll(m, base, scorer=toy_scorer)
        candidate = responses(m, source)
        candidate["records"][0]["raw_response"] = '{"answer":"no"}'
        result = e.enroll(m, candidate, scorer=toy_scorer, source_enrollment=source, candidate_identity=identity(True))
        summary = result["paired_summary"]
        self.assertEqual(summary["source_correct_losses"], 1)
        self.assertEqual(summary["source_incorrect_repairs"], 1)
        self.assertEqual(summary["per_family"][0]["source_correct_loss_rate"], 1)
        self.assertIsNone(summary["per_family"][0]["source_incorrect_repair_rate"])

    def test_cli_structure_only_write_once_and_rejections_preserve_exact_inputs(self):
        m = manifest()
        r = responses(m)
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            root = Path(directory)
            mp, rp, out = root / "manifest.json", root / "responses.json", root / "receipt.json"
            mp.write_text(json.dumps(m), encoding="utf-8")
            rp.write_text(json.dumps(r), encoding="utf-8")
            args = ["--manifest", str(mp), "--responses", str(rp), "--output", str(out)]
            self.assertEqual(e.main(args), 0)
            result = json.loads(out.read_text())
            self.assertEqual(result["status"], e.STRUCTURE_STATUS)
            saved = out.read_bytes()
            self.assertEqual(e.main(args), 2)
            self.assertEqual(out.read_bytes(), saved)
            rejected = root / "rejection.json"
            raw = b'{"schema_version":1,"schema_version":1,"records":["raw failure"]}'
            rp.write_bytes(raw)
            args[-1] = str(rejected)
            self.assertEqual(e.main(args), 2)
            failure = json.loads(rejected.read_text())
            self.assertEqual(failure["status"], "REJECTED_RESPONSE_IMPORT")
            self.assertEqual(base64.b64decode(failure["raw_inputs_base64"]["responses"]), raw)
            self.assertEqual(base64.b64decode(failure["raw_inputs_base64"]["manifest"]), mp.read_bytes())

    def test_import_has_no_torch_tokenizer_or_environment_side_effects(self):
        root = Path(__file__).resolve().parents[3]
        code = ("import os, sys; "
                "from experiments.post_training_mimo_centralized_v1.scripts import enrollment; "
                "assert os.environ.get('USE_TORCH') == 'fixture-sentinel'; "
                "assert not any(k.startswith(('torch', 'transformers', 'tokenizers')) "
                "or 'tokenize_bridge' in k or 'build_long_context' in k for k in sys.modules)")
        env = dict(os.environ, USE_TORCH="fixture-sentinel")
        completed = subprocess.run([sys.executable, "-c", code], cwd=root, env=env, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
