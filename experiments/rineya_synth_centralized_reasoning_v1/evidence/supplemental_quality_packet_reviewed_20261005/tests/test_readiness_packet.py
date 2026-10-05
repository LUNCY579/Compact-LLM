"""Independent malformed-input and drift controls; no native framework imports."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import readiness_packet as rp


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row(row_id, total, completion):
    prompt = [17] * (total - completion)
    target = [23] * (completion - 1) + [151645]
    return {"row_id": row_id, "prompt_ids": prompt, "completion_ids": target,
            "input_ids": prompt + target, "labels": [-100] * len(prompt) + target,
            "attention_mask": [1] * total, "prompt_tokens": len(prompt),
            "supervised_tokens": completion, "total_tokens": total}


def corpus():
    candidates, pools, schedules = [], {}, {}
    for arm, extra_total, extra_target in (("M1", 422, 214), ("M2", 434, 215)):
        pools[arm], schedules[arm] = [], []
        for ordinal in range(540):
            row_id = hashlib.sha256((arm + str(ordinal)).encode()).hexdigest()
            candidates.append({"row_id": row_id, "parent_id": "parent%02d" % (ordinal % 20),
                               "skill": "R" if ordinal < 192 else "S", "split": "train", "arm": arm,
                               "verification": {"accepted": True, "reason": "verified"}})
            pools[arm].append(row(row_id, 241 + (ordinal < extra_total), 10 + (ordinal < extra_target)))
            schedules[arm].append(row_id)
    schedule = {"status": "MATCHED_CPU_SCHEDULE", "optimizer_updates_per_arm": 27,
                "schedules": schedules, "sha256": hashlib.sha256(b"independent synthetic schedule").hexdigest()}
    occurrences = {"row_ids": deepcopy(schedules), "schedule_sha256": schedule["sha256"]}
    return schedule, occurrences, candidates, pools


def recipe():
    adaptation = {"method": "qv_lora", "rank": 8, "alpha": 16, "base_frozen": True,
                  "layers": list(range(28)), "projections": ["q_proj", "v_proj"], "dropout": 0.0,
                  "target_format": "final_answer_only", "common_warmstart": False,
                  "context_cap_tokens": 512, "microbatch_sequences": 1, "gradient_accumulation_sequences": 20,
                  "packing": False, "padding": False, "trainable_parameters": 1146880, "trainable_tensors": 112,
                  "loss": "completion_plus_EOS_mean_per_sequence_then_mean_over_accumulation",
                  "replay": "none_in_first_atomic_dose_screen; separate prior-correct guards mandatory"}
    optimizer = {"name": "AdamW", "betas": [0.9, 0.999], "eps": 1e-8, "weight_decay": 0.0,
                 "clip_norm": 1.0, "optimizer_state_dtype": "float32", "peak_lr": 1e-4,
                 "schedule": "5_percent_warmup_then_linear_decay_indexed_by_frozen_matched_optimizer_updates"}
    student = {"name": "Rineya-Titan-Experimental-1.0.0-devfreeze", "base_parameters": 596049920,
               **rp.SOURCE_ASSET_PINS}
    decisions = {"student": student, "adaptation": adaptation, "optimizer": optimizer,
                 "seeds": {"first_development_seed": 101}, "selection": {"checkpoint": "final_only"},
                 "permissions": {"training": False, "native_inference": False}}
    peak = Fraction(1, 10000)
    rates = [float(peak / 2), float(peak)] + [float(peak * Fraction(n, 25)) for n in range(24, -1, -1)]
    draft = {"student": deepcopy(student), "adaptation": deepcopy(adaptation), "seed": 101,
             "optimizer": {**optimizer, "learning_rates_by_1based_update": rates, "warmup_updates": 2},
             "dose": {"full_updates_per_adapted_arm": 27, "sequences_per_update": 20, "whole_sequences_per_arm": 540,
                      "M1_actual_input_tokens": 130562, "M2_actual_input_tokens": 130574,
                      **{key: False for key in ("padding", "packing", "truncation", "dropping", "replay_added", "warmstart", "RS_training")}},
             "resume_contract": {"chunks_per_arm": 3, "full_updates_per_chunk": 9,
                                 "qualification_STATE1_warmstart_allowed": False},
             "native_launch_authorized": False, "source_and_decision_pins": {}}
    return decisions, draft


class ParsingAndPathsTests(unittest.TestCase):
    def test_duplicate_nested_keys_refused(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            rp.parse_json(b'{"outer":{"same":1,"same":2}}')

    def test_nonfinite_and_overflow_numbers_refused(self):
        for raw in (b'{"x":NaN}', b'{"x":Infinity}', b'{"x":1e1000}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                rp.parse_json(raw)

    def test_bounded_json_and_nesting(self):
        with self.assertRaisesRegex(ValueError, "bounded"):
            rp.parse_json(b" " * (rp.MAX_BYTES + 1))
        with self.assertRaisesRegex(ValueError, "nesting"):
            rp.parse_json(b"[" * 66 + b"1" + b"]" * 66)

    def test_empty_file_and_empty_digest_refused(self):
        self.assertFalse(rp.is_hash(hashlib.sha256(b"").hexdigest()))
        self.assertFalse(rp.is_hash("0" * 64))
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "empty.json").write_bytes(b"")
            with self.assertRaisesRegex(ValueError, "nonempty"):
                rp.read_selected(temporary, "empty.json")

    def test_traversal_absolute_and_windows_paths_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            for relative in ("../a.json", "/a.json", "a/../b.json", "a//b.json", "C:/a.json", "a\\b.json"):
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    rp.safe_path(temporary, relative)

    def test_symlink_ancestor_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            target = base / "real"
            target.mkdir()
            (target / "file.json").write_text("{}", encoding="utf-8")
            try:
                (base / "link").symlink_to(target, target_is_directory=True)
            except OSError:
                # Test the same Windows refusal without granting symlink privilege.
                original = Path.lstat
                def reparse(path, *args, **kwargs):
                    info = original(path, *args, **kwargs)
                    if path == target:
                        return SimpleNamespace(st_mode=info.st_mode, st_file_attributes=0x400)
                    return info
                with patch.object(Path, "lstat", autospec=True, side_effect=reparse):
                    with self.assertRaisesRegex(ValueError, "reparse"):
                        rp.read_selected(base, "real/file.json")
                return
            with self.assertRaisesRegex(ValueError, "symlink"):
                rp.read_selected(base, "link/file.json")

    def test_source_byte_drift_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary, "file.json")
            path.write_bytes(b"{}")
            _, ref = rp.read_selected(temporary, "file.json")
            path.write_bytes(b'{"changed":true}')
            with self.assertRaisesRegex(ValueError, "byte drift"):
                rp.read_selected(temporary, "file.json", expected_sha256=ref["sha256"])

    def test_model_file_read_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "model.safetensors").write_bytes(b"never read")
            with self.assertRaisesRegex(ValueError, "text/metadata"):
                rp.read_selected(temporary, "model.safetensors")


class RecipeAndCorpusTests(unittest.TestCase):
    def test_exact_lr_vector_from_independent_formula(self):
        values = rp.learning_rates_exact()
        actual = [Fraction(item["numerator"], item["denominator"]) for item in values]
        expected = [Fraction(5, 100000), Fraction(1, 10000)] + [Fraction(i, 250000) for i in range(24, -1, -1)]
        self.assertEqual(actual, expected)
        self.assertEqual([item["update"] for item in values], list(range(1, 28)))
        self.assertEqual(actual[-1], 0)

    def test_valid_frozen_recipe(self):
        decisions, draft = recipe()
        self.assertEqual(rp.validate_recipe(decisions, draft)["seed"], 101)

    def test_rank_source_final_selection_and_lr_drift_refused(self):
        for mutate in (lambda d, r: d["adaptation"].update(rank=16),
                       lambda d, r: d["student"].update(weight_sha256=hashlib.sha256(b"replacement model").hexdigest()),
                       lambda d, r: d["selection"].update(checkpoint="best"),
                       lambda d, r: r["optimizer"]["learning_rates_by_1based_update"].__setitem__(26, 1e-4)):
            decisions, draft = recipe()
            mutate(decisions, draft)
            with self.assertRaises(ValueError):
                rp.validate_recipe(decisions, draft)

    def test_token_masks_eos_and_supervision_counts(self):
        original = row("id", 242, 11)
        rp.validate_token_row(original)
        for mutate in (lambda r: r["labels"].__setitem__(0, 17),
                       lambda r: r["completion_ids"].__setitem__(-1, 23),
                       lambda r: r["attention_mask"].__setitem__(0, 0),
                       lambda r: r.update(supervised_tokens=10),
                       lambda r: r["input_ids"].__setitem__(0, True)):
            invalid = deepcopy(original)
            mutate(invalid)
            with self.assertRaises(ValueError):
                rp.validate_token_row(invalid)

    def test_whole540_exact_token_totals(self):
        result = rp.audit_dose(*corpus())
        self.assertEqual((result["M1"]["input_tokens"], result["M2"]["input_tokens"]), (130562, 130574))
        self.assertEqual((result["M1"]["supervised_tokens"], result["M2"]["supervised_tokens"]), (5614, 5615))
        self.assertEqual(result["M1"]["chunks"][-1]["last_occurrence"], 540)

    def test_shortened_duplicate_and_rs_training_refused(self):
        for mutate in (lambda s, o, c, p: s["schedules"]["M1"].pop(),
                       lambda s, o, c, p: p["M1"].append(deepcopy(p["M1"][0])),
                       lambda s, o, c, p: c[0].update(skill="RS")):
            inputs = corpus()
            mutate(*inputs)
            with self.assertRaises(ValueError):
                rp.audit_dose(*inputs)


class PacketTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name)
        self.source, self.external = self.base / "source", self.base / "external"
        self.source.mkdir()
        self.external.mkdir()
        for name in rp.ALL_FILES:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("public source fixture\n", encoding="utf-8")
        decisions, draft = recipe()
        dump(self.source / (rp.PACKAGE + "/PILOT_DECISIONS.json"), decisions)
        dump(self.source / (rp.PACKAGE + "/STUDY_DEV_REGISTRATION_DRAFT.json"), draft)
        dump(self.source / (rp.BUNDLE + "/pilot_decisions.json"), decisions)
        dump(self.source / (rp.PACKAGE + "/DEV_STATISTICS_POLICY.json"), {
            "id": "HISTORICAL-FIVE-POINT-CI", "advancement": {
                "atomic_M2_minus_M1_minimum": 0.05, "atomic_lower_CI_strictly_positive": True}})
        schedule, occurrences, candidates, pools = corpus()
        dump(self.source / (rp.BUNDLE + "/schedule_result.json"), schedule)
        dump(self.source / (rp.BUNDLE + "/training_occurrences.json"), occurrences)
        for name, values in (("candidate_records.jsonl", candidates),
                             ("token_pool_M1.jsonl", pools["M1"]), ("token_pool_M2.jsonl", pools["M2"])):
            (self.source / rp.BUNDLE / name).write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
        self.identity = patch.object(rp, "git_identity", return_value={"head": "a" * 40, "branch": "synthetic-fixture", "clean": True})
        self.identity.start()
        self.addCleanup(self.identity.stop)
        fixture_dev_pins = {name: hashlib.sha256((self.source / name).read_bytes()).hexdigest() for name in rp.DEV_FILES}
        self.dev_pins = patch.object(rp, "DEV_FILE_HASHES", fixture_dev_pins)
        self.dev_pins.start()
        self.addCleanup(self.dev_pins.stop)

    def test_missing_binders_remain_blocked_no_partial_m0_admission(self):
        packet = rp.build_packet(self.source, self.external)
        self.assertIn("BLOCKED", packet["status"])
        self.assertFalse(packet["measurement_freeze"]["baseline_complete"])
        self.assertEqual(packet["measurement_freeze"]["M0_audited_raw_rows_at_owner_handoff"], 256)
        self.assertEqual(packet["measurement_freeze"]["M0_scored_rows"], 0)
        self.assertEqual(packet["measurement_freeze"]["complete_baseline_required"]["audits"], 6)
        self.assertTrue(all(slot["status"] == "UNRESOLVED" for slot in packet["unresolved_slots"].values()))
        self.assertTrue(all(value is False for value in packet["permissions"].values()))
        self.assertFalse(packet["execution_permit"])
        self.assertFalse(packet["native_ready"])

    def test_retains_all_checkpoints_gd_original9_27_final_primary(self):
        packet = rp.build_packet(self.source, self.external)
        self.assertEqual(packet["producer_contract"]["retain_all_original_checkpoint_steps"], list(range(1, 28)))
        self.assertEqual(packet["producer_contract"]["GD_admission_original_steps"], [9, 27])
        self.assertTrue(packet["producer_contract"]["actual_native_export_parity_required"])
        self.assertFalse(packet["producer_contract"]["qualification_full27_weights_are_study_weights"])
        self.assertEqual(packet["measurement_freeze"]["primary_checkpoint"], "final_only")

    def test_untrusted_future_metadata_never_changes_gate(self):
        metadata = {slot: [{"path": "preserved/future.json", "sha256": hashlib.sha256(slot.encode()).hexdigest()}] for slot in rp.SLOTS}
        packet = rp.build_packet(self.source, self.external, supplementary_refs=metadata)
        self.assertTrue(all(slot["candidate_metadata_refs"] and slot["status"] == "UNRESOLVED"
                            and slot["candidate_refs_qualify_gate"] is False for slot in packet["unresolved_slots"].values()))
        self.assertFalse(packet["native_ready"])
        with self.assertRaises(ValueError):
            rp.build_packet(self.source, self.external, supplementary_refs={"synth_owner_review": [{"path": "a", "sha256": "bad"}]})

    def test_original_history_and_new_supplement_separate(self):
        packet = rp.build_packet(self.source, self.external)
        self.assertEqual(packet["historical_policy"]["id"], "HISTORICAL-FIVE-POINT-CI")
        self.assertEqual(packet["prospective_quality_policy"]["supplement_id"], "SYNTH-QUALITY-3PP-ZERO-DAMAGE-20261005")
        self.assertFalse(packet["prospective_quality_policy"]["replaces_historical_policy"])
        self.assertFalse(packet["prospective_quality_policy"]["claims_statistical_power"])

    def test_exclusive_write_and_external_hash_check(self):
        path = self.external / "packet.json"
        result = rp.write_packet(path, self.source, self.external)
        audit = rp.check_packet(path, result["file"]["sha256"], self.source, self.external)
        self.assertEqual(audit["status"], "PASS_BLOCKED_METADATA_PACKET_BYTE_BINDINGS")
        with self.assertRaises(ValueError):
            rp.write_packet(path, self.source, self.external)
        with self.assertRaises(ValueError):
            rp.check_packet(path, "0" * 64, self.source, self.external)

    def test_changed_source_refuses_even_when_packet_itself_is_unchanged(self):
        path = self.external / "packet.json"
        result = rp.write_packet(path, self.source, self.external)
        (self.source / (rp.PACKAGE + "/rs_solver.py")).write_text("source drift\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "snapshot drift"):
                rp.check_packet(path, result["file"]["sha256"], self.source, self.external)

    def test_open_dev_membership_drift_refused_before_new_packet_build(self):
        (self.source / rp.DEV_FILES[0]).write_text("changed family grouping\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "byte drift"):
            rp.build_packet(self.source, self.external)

    def test_source_head_changes_during_build_refused(self):
        with patch.object(rp, "git_identity", side_effect=[
                {"head": "a" * 40, "branch": "fixture", "clean": True},
                {"head": "b" * 40, "branch": "fixture", "clean": True}]):
            with self.assertRaisesRegex(ValueError, "source changed"):
                rp.build_packet(self.source, self.external)

    def test_resealed_execution_or_ready_flags_refused(self):
        packet = rp.build_packet(self.source, self.external)
        for flag in ("execution_permit", "native_ready", "owner_reviewed"):
            invalid = deepcopy(packet)
            invalid[flag] = True
            invalid["sha256"] = rp.digest({key: value for key, value in invalid.items() if key != "sha256"})
            path = self.external / (flag + ".json")
            dump(path, invalid)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):
                rp.check_packet(path, sha, self.source, self.external)

    def test_bad_prospective_policy_does_not_bind_as3point(self):
        path = self.external / "bad_policy.json"
        dump(path, {"minimum_macro_atomic_gain": 0.03})
        with self.assertRaisesRegex(ValueError, "distinct prospective"):
            rp.build_packet(self.source, self.external, {"relative_path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})


if __name__ == "__main__":
    unittest.main()
