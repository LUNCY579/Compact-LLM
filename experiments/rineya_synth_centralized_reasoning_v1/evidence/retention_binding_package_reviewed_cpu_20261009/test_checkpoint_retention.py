"""Inert metadata controls only; synthetic hashes never identify actual payloads."""
from copy import deepcopy
import ast
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import checkpoint_retention as binding


def _file(path):
    return {"path": path, "sha256": binding.digest({"inert_descriptor": path}), "bytes": 123}


def _counts(number):
    return {"attempted": number, "returned": number, "failed": 0, "pending": 0}


def _costs(counter, step, scope):
    chunks = (step - 1) // 9
    operations = {
        "gradient_reset": step, "forward": step * 20, "completion_loss": step * 20,
        "backward": step * 20, "finite_gradient_clip": step, "optimizer_step": step,
        "checkpoint_save": chunks, "checkpoint_reload": chunks,
        "checkpoint_publish": chunks, "terminal_costs_publish": chunks, "checkpoint_promote": chunks,
    }
    primitives = {name: _counts(step - int(name.endswith(("publish_write", "publish_verify"))))
                  for name in ("retained_update_path", "retained_update_save", "retained_update_verify",
                               "retained_update_publish_write", "retained_update_publish_verify")}
    for name in ("GD_export_parity_request_write", "GD_export_parity_request_verify"):
        primitives[name] = _counts(sum(s < step for s in binding.GD_STEPS))
    return {"schema_version": 1, "scope": scope, "protocol": {k: _counts(v) for k, v in operations.items()},
            "primitive_operations": primitives,
            "positions": {"input_attempted": counter["input_tokens"], "input_returned": counter["input_tokens"],
                          "supervised_attempted": counter["supervised_tokens"],
                          "supervised_returned": counter["supervised_tokens"]}}


def _publication(path, record):
    return dict(_file(path), payload_sha256=binding.producer_digest(record), durable=True)


def make_fixture(non_ascii=False, owner_metadata_shape=False):
    """Fictional metadata, including optional owner-shaped assertions; no payloads.

    owner_metadata_shape tests native-format refusals and immutable false outputs.
    It supplies no genuine producer identity, native receipt, state, or parity.
    """
    evidence = "producer_metadata" if owner_metadata_shape else "mechanical_fixture"
    schedule = {
        "schema": binding.SCHEDULE_SCHEMA,
        "evidence_kind": "frozen_owner_schedule_metadata" if owner_metadata_shape else "mechanical_fixture",
        "experiment": "SYNTH-CRS-001", "seed": 101, "occurrences_per_arm": 540,
        "updates_per_arm": 27, "occurrences_per_update": 20,
        "retained_steps": list(binding.STEPS), "scientific_resume_boundaries": [9, 18, 27],
        "gd_original_steps": [9, 27],
        "producer_checkpoint": {"kind": binding.OWNER_RETAINED_KIND if owner_metadata_shape else "INERT_FIXTURE_RETAINED",
                                "schema_version": 1},
        "producer_record": {"kind": binding.OWNER_RECORD_KIND if owner_metadata_shape else "INERT_FIXTURE_RETAINED_RECORD",
                            "schema_version": 1},
        "scientific_checkpoint": {"kind": binding.OWNER_SCIENTIFIC_KIND if owner_metadata_shape else "INERT_FIXTURE_SCIENTIFIC_BOUNDARY",
                                 "schema_version": 1},
        "producer_policy_sha256": binding.digest("INERT_FIXTURE_POLICY"),
        "cost_scope": binding.OWNER_COST_SCOPE if owner_metadata_shape else "INERT_FIXTURE_COST_SCOPE",
        # Explicit test table: this is not the accepted scientific LR schedule.
        "learning_rates": [{"update": step, "numerator": 1, "denominator": 10000,
                            "float_value": 0.0001} for step in binding.STEPS],
        "arms": {},
    }
    manifest = {"schema": binding.MANIFEST_SCHEMA, "evidence_kind": evidence,
                "schedule_sha256": None, "arms": {}}
    for arm in binding.ARMS:
        identity = {key: binding.digest({"inert_identity": key, "arm": arm}) for key in binding.HASH_FIELDS}
        identity["trajectory_id"] = ("OWNER_ASSERTED_" if owner_metadata_shape else "INERT_FIXTURE_") + arm + ("_β" if non_ascii else "")
        metadata = {
            "schema_version": 1, "kind": schedule["producer_checkpoint"]["kind"],
            "experiment": "SYNTH-CRS-001",
            "scope": "UNQUALIFIED_NATIVE_IMPLEMENTATION" if owner_metadata_shape else "CPU_RANDOM_TOY",
            "arm": arm, "seed": 101,
            "recipe_sha256": identity["recipe_sha256"], "cohort_sha256": identity["cohort_sha256"],
            "backend_source_sha256": identity["backend_source_sha256"],
            "fresh_start_attestation_sha256": binding.digest({"inert_start": arm}),
            "adapter_parameter_order": [("declared" if owner_metadata_shape else "fixture") + ".adapter." + suffix
                                        for suffix in ("A", "β" if non_ascii else "B")],
            "adapter_inventory_sha256": binding.digest({"inert_adapter_inventory": arm}),
            "native_launch_authorized": False, "training_authorized": False,
            "native_resume_qualified": False, "qualification_STATE1_used": False,
            "producer_policy_sha256": schedule["producer_policy_sha256"], "resumable": False,
        }
        counters = [{"completed_updates": step, "consumed_occurrences": step * 20,
                     "input_tokens": step * 4200 + (1 if arm == "M2" else 0),
                     "supervised_tokens": step * 200, "learning_rate": 0.0001,
                     "consumed_order_sha256": binding.digest({"inert_order": arm, "step": step}),
                     "next_update_ordinal": step + 1} for step in binding.STEPS]
        parity_pins = {"policy_sha256": binding.digest({"inert_parity_policy": arm}),
                       "probe_registration_sha256": binding.digest({"inert_probes": arm})}
        scientific_metadata = {k: deepcopy(v) for k, v in metadata.items()
                               if k not in ("producer_policy_sha256", "resumable")}
        scientific_metadata.update(schedule["scientific_checkpoint"])
        schedule["arms"][arm] = {"identity": identity, "metadata_pins": metadata,
                                 "scientific_metadata_pins": scientific_metadata,
                                 "gd_parity": parity_pins, "counters": counters}
        prefix = "C:/Users/clclc/BusinessProject/" + ("OWNER_ASSERTED_METADATA" if owner_metadata_shape else "INERT_FIXTURE_METADATA")
        prefix += ("_β" if non_ascii else "") + "/" + arm
        checkpoints = []
        for step, counter in enumerate(counters, 1):
            descriptor = dict(_file(prefix + "/retained/step-" + str(step) + ".inert"),
                              schema_version=1, kind=schedule["producer_checkpoint"]["kind"],
                              metadata=dict(deepcopy(metadata), binding=deepcopy(counter)))
            descriptor["metadata_sha256"] = binding.producer_digest(descriptor["metadata"])
            native_record = {
                "schema_version": 1, "kind": schedule["producer_record"]["kind"], "arm": arm, "seed": 101,
                "recipe_sha256": identity["recipe_sha256"],
                "producer_policy_sha256": schedule["producer_policy_sha256"],
                "binding": deepcopy(counter), "backend_descriptor": deepcopy(descriptor),
                "posted_save_costs": _costs(counter, step, schedule["cost_scope"]),
                "GD_export_parity_required": step in binding.GD_STEPS, "GD_export_parity_qualified": False,
            }
            native_record["sha256"] = binding.producer_digest(native_record)
            checkpoints.append({
                "schema": binding.CHECKPOINT_SCHEMA, "evidence_kind": evidence,
                "artifact_role": "original_study_checkpoint" if owner_metadata_shape else "mechanical_fixture_checkpoint",
                "arm": arm, "seed": 101, "step": step,
                "producer_checkpoint": deepcopy(schedule["producer_checkpoint"]),
                "identity": deepcopy(identity), "roles": binding.roles_for_step(step),
                "counters": deepcopy(counter), "save_status": "SAVE_RETURNED", "lineage_status": "COMPLETE",
                "native_descriptor": descriptor, "producer_record": native_record,
                "publication": _publication(prefix + "/records/step-" + str(step) + ".json", native_record),
                "GD_export_parity_request_publication":
                    _publication(prefix + "/requests/step-" + str(step) + ".json", native_record)
                    if step in binding.GD_STEPS else None,
            })
        scientific = []
        for step in binding.RESUME_STEPS:
            descriptor = dict(_file(prefix + "/scientific/step-" + str(step) + ".inert"),
                              **schedule["scientific_checkpoint"],
                              metadata=dict(deepcopy(scientific_metadata), binding=deepcopy(counters[step - 1])))
            descriptor["metadata_sha256"] = binding.producer_digest(descriptor["metadata"])
            scientific.append({
                "schema": binding.SCIENTIFIC_SCHEMA, "evidence_kind": evidence,
                "arm": arm, "step": step, "identity": deepcopy(identity),
                "counters": deepcopy(counters[step - 1]), "role": "registered_scientific_resume_boundary",
                "retained_step_descriptor_sha256": binding.digest(checkpoints[step - 1]),
                "native_descriptor": descriptor,
            })
        exports = []
        for step in binding.GD_STEPS:
            directory = prefix + "/gd-export/step-" + str(step)
            files = [{"role": "adapter_config", "file": _file(directory + "/adapter_config.json")},
                     {"role": "adapter_weights", "file": _file(directory + "/adapter_model.safetensors")}]
            exports.append({
                "schema": binding.EXPORT_SCHEMA, "evidence_kind": evidence,
                "artifact_role": "standard_peft_payload_metadata" if owner_metadata_shape else "mechanical_fixture_peft_payload",
                "arm": arm, "step": step,
                "identity": deepcopy(identity), "source_checkpoint_sha256": binding.digest(checkpoints[step - 1]),
                "format": "standard_peft_lora" if owner_metadata_shape else "inert_fixture",
                "directory": directory, "files": files,
                "inventory_sha256": binding.digest(files), "status": "EXPORT_REPORTED_COMPLETE",
                "snapshot_receipt": _file(directory + "/snapshot-receipt.json"),
                "export_receipt": _file(directory + "/export-receipt.json"),
            })
        parity = {
            "schema": binding.PARITY_SCHEMA, "evidence_kind": evidence,
            "arm": arm, "identity": deepcopy(identity),
            "parity_policy_sha256": parity_pins["policy_sha256"],
            "probe_registration_sha256": parity_pins["probe_registration_sha256"],
            "receipt_schema": "gd001_export_parity_result_v1",
            "evidence_level": "native_pilot" if owner_metadata_shape else "mechanical_fixture",
            "joins": [{"step": step, "source_checkpoint_sha256": binding.digest(checkpoints[step - 1]),
                       "peft_export_sha256": binding.digest(export)}
                      for step, export in zip(binding.GD_STEPS, exports)],
            "receipts": {name: _file(prefix + "/parity/" + name + ".json")
                         for name in ("parity_result", "producer_trace", "gd_trace", "strict_load")},
        }
        manifest["arms"][arm] = {
            "identity": deepcopy(identity), "lineage_status": "COMPLETED", "completed_updates": 27,
            "failed_operations": 0, "pending_operations": 0, "incomplete_saves": 0,
            "completion_receipt": _file(prefix + "/completion.json"),
            "checkpoints": checkpoints, "scientific_boundaries": scientific,
            "gd_exports": exports, "parity_join": parity,
        }
    manifest["schedule_sha256"] = binding.digest(schedule)
    return schedule, manifest, manifest["schedule_sha256"]


class RetentionBindingControls(unittest.TestCase):
    def setUp(self):
        self.schedule, self.manifest, self.pin = make_fixture()

    def validate(self):
        return binding.validate_retention(self.manifest, self.schedule, self.pin)

    def refused(self, mutation, text=None):
        mutation()
        with self.assertRaisesRegex(binding.ValidationError, text or "."):
            self.validate()

    def checkpoint(self, step=1, arm="M1"):
        return self.manifest["arms"][arm]["checkpoints"][step - 1]

    def reseal_record(self, checkpoint):
        record = checkpoint["producer_record"]
        record["sha256"] = binding.producer_digest({k: v for k, v in record.items() if k != "sha256"})
        checkpoint["publication"]["payload_sha256"] = binding.producer_digest(record)
        if checkpoint["GD_export_parity_request_publication"] is not None:
            checkpoint["GD_export_parity_request_publication"]["payload_sha256"] = binding.producer_digest(record)

    def test_complete_fixture_is_explicitly_structural_and_closed(self):
        result = self.validate()
        self.assertEqual(result["checkpoint_descriptors"], 54)
        self.assertEqual(result["scientific_checkpoint_descriptors"], 6)
        self.assertEqual(result["peft_export_descriptors"], 4)
        self.assertEqual(result["parity_receipt_joins"], 2)
        self.assertEqual(result["evidence_kind"], "mechanical_fixture")
        self.assertTrue(result["structural_only"])
        for field in ("native_admissible", "native_execution_verified", "native_export_parity_verified",
                      "native_resume_qualified", "training_authorized", "capability_claims_admissible",
                      "retained_archives_resumable", "gd_request_publications_are_parity_evidence",
                      "payloads_read", "receipt_contents_verified"):
            self.assertIs(result[field], False)
        self.assertTrue(result["required_external_owner_verification"])

    def test_owner_shaped_metadata_also_never_admits_native_or_capability_claims(self):
        schedule, manifest, pin = make_fixture(owner_metadata_shape=True)
        with patch("builtins.open", side_effect=AssertionError("original payload I/O forbidden")):
            result = binding.validate_retention(manifest, schedule, pin)
        self.assertEqual(result["evidence_kind"], "producer_metadata")
        self.assertTrue(result["structural_only"])
        for field in ("native_admissible", "native_execution_verified", "native_export_parity_verified",
                      "native_resume_qualified", "training_authorized", "capability_claims_admissible"):
            self.assertIs(result[field], False)
        self.assertIs(result["receipt_contents_verified"], False)

    def test_native_schedule_requires_exact_frozen_owner_kinds_and_version1(self):
        for name, field, value in (("producer_checkpoint", "kind", "ARBITRARY_RETAINED_KIND"),
                                   ("producer_record", "kind", "ARBITRARY_RECORD_KIND"),
                                   ("scientific_checkpoint", "kind", "ARBITRARY_SCIENTIFIC_KIND"),
                                   ("producer_checkpoint", "schema_version", 2),
                                   ("producer_record", "schema_version", 2),
                                   ("scientific_checkpoint", "schema_version", 2)):
            with self.subTest(name=name, field=field):
                schedule, manifest, _ = make_fixture(owner_metadata_shape=True)
                schedule[name][field] = value
                pin = binding.digest(schedule)
                manifest["schedule_sha256"] = pin
                with self.assertRaisesRegex(binding.ValidationError, "exact frozen owner kind"):
                    binding.validate_retention(manifest, schedule, pin)

    def test_native_retained_and_scientific32mib_byte_ceiling(self):
        for group in ("checkpoints", "scientific_boundaries"):
            with self.subTest(group=group):
                schedule, manifest, pin = make_fixture(owner_metadata_shape=True)
                manifest["arms"]["M1"][group][0]["native_descriptor"]["bytes"] = binding.MAX_STUDY_STATE_BYTES + 1
                with self.assertRaisesRegex(binding.ValidationError, "32MiB"):
                    binding.validate_retention(manifest, schedule, pin)

    def test_native_schedule_cannot_bind_toy_metadata_scope(self):
        schedule, manifest, _ = make_fixture(owner_metadata_shape=True)
        schedule["arms"]["M1"]["metadata_pins"]["scope"] = "CPU_RANDOM_TOY"
        schedule["arms"]["M1"]["scientific_metadata_pins"]["scope"] = "CPU_RANDOM_TOY"
        pin = binding.digest(schedule)
        manifest["schedule_sha256"] = pin
        with self.assertRaisesRegex(binding.ValidationError, "fixture/unknown metadata scope"):
            binding.validate_retention(manifest, schedule, pin)

    def test_original_metadata_path_cannot_relabel_fixture_artifact(self):
        schedule, manifest, pin = make_fixture(owner_metadata_shape=True)
        descriptor = manifest["arms"]["M1"]["checkpoints"][0]["native_descriptor"]
        descriptor["path"] = "C:/Users/clclc/BusinessProject/mechanical_fixture/step1.inert"
        with self.assertRaisesRegex(binding.ValidationError, "fixture artifact"):
            binding.validate_retention(manifest, schedule, pin)

    def test_native_schedule_source_recipe_runtime_data_hashes_must_resolve(self):
        for field in ("source_sha256", "recipe_sha256", "runtime_sha256", "data_sha256"):
            with self.subTest(field=field):
                schedule, manifest, _ = make_fixture(owner_metadata_shape=True)
                schedule["arms"]["M1"]["identity"][field] = None
                pin = binding.digest(schedule)
                manifest["schedule_sha256"] = pin
                with self.assertRaisesRegex(binding.ValidationError, "pending is refused"):
                    binding.validate_retention(manifest, schedule, pin)

    def test_no_descriptor_payload_or_receipt_is_opened(self):
        with patch("builtins.open", side_effect=AssertionError("payload I/O forbidden")):
            self.validate()

    def test_validator_imports_only_stdlib_and_no_producer_code(self):
        source = Path(binding.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        imports = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        imports |= {alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertEqual(imports, {"__future__", "hashlib", "json", "math", "re"})

    def test_three_boundary_checkpoints_cannot_substitute_for27(self):
        self.refused(lambda: self.manifest["arms"]["M1"].update(
            checkpoints=[self.checkpoint(s) for s in (9, 18, 27)]), "retain every original")

    def test_missing_arm_refused(self):
        self.refused(lambda: self.manifest["arms"].pop("M2"))

    def test_missing_original_step_refused(self):
        self.refused(lambda: self.manifest["arms"]["M2"]["checkpoints"].pop(12))

    def test_duplicate_or_reordered_step_refused(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["checkpoints"].__setitem__(1, deepcopy(self.checkpoint())))

    def test_extra_retained_step_refused(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["checkpoints"].append(deepcopy(self.checkpoint())))

    def test_duplicate_case_insensitive_payload_path_refused(self):
        self.refused(lambda: self.checkpoint(2)["native_descriptor"].update(
            path=self.checkpoint()["native_descriptor"]["path"].upper()), "duplicate retained file")

    def test_duplicate_original_payload_hash_refused(self):
        self.refused(lambda: self.checkpoint(2)["native_descriptor"].update(
            sha256=self.checkpoint()["native_descriptor"]["sha256"]), "duplicate original")

    def test_relative_or_aliased_paths_refused(self):
        for path in ("step1.inert", "C:/owned/../step1.inert", "C:/owned/step1.inert.", "C:/owned//step1.inert"):
            with self.subTest(path=path):
                self.setUp()
                self.refused(lambda: self.checkpoint()["native_descriptor"].update(path=path))

    def test_reserved_windows_device_paths_refused(self):
        for name in ("CON", "nul.json", "PrN", "AUX.txt", "COM1.json", "Lpt9", "CONIN$", "CONOUT$",
                     "COM¹.json", "COM²", "COM³.txt", "LPT¹", "LPT².json", "LPT³"):
            with self.subTest(name=name):
                self.setUp()
                self.refused(lambda: self.checkpoint()["native_descriptor"].update(
                    path="C:/Users/clclc/BusinessProject/" + name + "/step1.inert"), "device")

    def test_state1_and_full27_qualification_refused(self):
        for token in ("STATE1", "FULL27", "qualification"):
            with self.subTest(token=token):
                self.setUp()
                self.refused(lambda: self.checkpoint()["native_descriptor"].update(
                    path="C:/Users/clclc/BusinessProject/" + token + "/step1.inert"), "qualification")

    def test_stale_schedule_pin_refused(self):
        self.refused(lambda: self.schedule.update(seed=102), "external SHA256")

    def test_rehashed_wrong_seed_or_boundaries_refused(self):
        for key, value in (("seed", 102), ("scientific_resume_boundaries", list(range(1, 28))),
                           ("gd_original_steps", [18, 27]), ("occurrences_per_arm", 541)):
            with self.subTest(key=key):
                self.setUp()
                self.schedule[key] = value
                self.pin = binding.digest(self.schedule)
                self.manifest["schedule_sha256"] = self.pin
                with self.assertRaisesRegex(binding.ValidationError, "accepted plan changed"):
                    self.validate()

    def test_pending_recipe_runtime_source_data_pins_refused(self):
        for field in ("recipe_sha256", "runtime_sha256", "source_sha256", "data_sha256"):
            with self.subTest(field=field):
                self.setUp()
                self.schedule["arms"]["M1"]["identity"][field] = None
                self.pin = binding.digest(self.schedule)
                with self.assertRaisesRegex(binding.ValidationError, "pending is refused"):
                    self.validate()

    def test_pending_format_refused(self):
        self.schedule["producer_checkpoint"]["kind"] = None
        self.pin = binding.digest(self.schedule)
        with self.assertRaises(binding.ValidationError):
            self.validate()

    def test_textual_pending_format_pins_refused(self):
        for value in ("PENDING", "TBD", "unresolved"):
            with self.subTest(value=value):
                self.setUp()
                self.schedule["producer_checkpoint"]["kind"] = value
                self.pin = binding.digest(self.schedule)
                with self.assertRaisesRegex(binding.ValidationError, "pending/unresolved"):
                    self.validate()

    def test_missing_per_step_counters_not_inferred_from_totals(self):
        self.schedule["arms"]["M1"]["counters"] = [self.schedule["arms"]["M1"]["counters"][-1]]
        self.pin = binding.digest(self.schedule)
        with self.assertRaisesRegex(binding.ValidationError, "exact externally supplied27|27-step"):
            self.validate()

    def test_altered_occurrence_token_order_and_lr_counters_refused(self):
        for field, value in (("consumed_occurrences", 19), ("input_tokens", 4201),
                             ("supervised_tokens", 201), ("consumed_order_sha256", "0" * 64),
                             ("learning_rate", 0.0002), ("next_update_ordinal", 3)):
            with self.subTest(field=field):
                self.setUp()
                self.refused(lambda: self.checkpoint()["counters"].update({field: value}))

    def test_frozen_rate_float_fraction_mismatch_refused(self):
        self.schedule["learning_rates"][0]["numerator"] = 2
        self.pin = binding.digest(self.schedule)
        with self.assertRaisesRegex(binding.ValidationError, "fraction/float mismatch"):
            self.validate()

    def test_nonadjacent_reused_occurrence_prefix_refused(self):
        counters = self.schedule["arms"]["M1"]["counters"]
        counters[2]["consumed_order_sha256"] = counters[0]["consumed_order_sha256"]
        self.pin = binding.digest(self.schedule)
        with self.assertRaisesRegex(binding.ValidationError, "duplicate/stale cumulative"):
            self.validate()

    def test_native_binding_lr_mismatch_refused(self):
        self.refused(lambda: self.checkpoint()["native_descriptor"]["metadata"]["binding"].update(
            learning_rate=0.0002), "native metadata")

    def test_native_metadata_digest_mismatch_refused(self):
        self.refused(lambda: self.checkpoint()["native_descriptor"].update(metadata_sha256="0" * 64))

    def test_native_kind_and_schema_version_mismatch_refused(self):
        for field, value in (("kind", "UNBOUND_KIND"), ("schema_version", 2)):
            with self.subTest(field=field):
                self.setUp()
                self.refused(lambda: self.checkpoint()["native_descriptor"].update({field: value}))

    def test_native_permission_or_retained_resumability_claim_refused(self):
        for field in ("native_launch_authorized", "training_authorized", "native_resume_qualified", "resumable"):
            with self.subTest(field=field):
                self.setUp()
                self.refused(lambda: self.checkpoint()["native_descriptor"]["metadata"].update({field: True}))

    def test_native_record_checksum_and_descriptor_join_refused(self):
        self.refused(lambda: self.checkpoint()["producer_record"].update(sha256="0" * 64))

    def test_failed_incomplete_and_pending_lineage_refused(self):
        for field, value in (("lineage_status", "FAILED"), ("completed_updates", 26),
                             ("failed_operations", 1), ("pending_operations", 1), ("incomplete_saves", 1)):
            with self.subTest(field=field):
                self.setUp()
                self.refused(lambda: self.manifest["arms"]["M1"].update({field: value}))

    def test_save_status_and_noncomplete_record_refused(self):
        self.refused(lambda: self.checkpoint().update(save_status="SAVE_PENDING"))

    def test_failed_native_cost_cannot_hide_behind_complete_sidecar(self):
        record = self.checkpoint()["producer_record"]
        record["posted_save_costs"]["protocol"]["optimizer_step"]["failed"] = 1
        record["sha256"] = binding.producer_digest({k: v for k, v in record.items() if k != "sha256"})
        with self.assertRaisesRegex(binding.ValidationError, "failed/pending"):
            self.validate()

    def test_pending_primitive_cannot_hide_behind_resealed_complete_record(self):
        checkpoint = self.checkpoint()
        checkpoint["producer_record"]["posted_save_costs"]["primitive_operations"]["retained_update_save"]["pending"] = 1
        self.reseal_record(checkpoint)
        with self.assertRaisesRegex(binding.ValidationError, "failed/pending"):
            self.validate()

    def test_resealed_primitive_cost_prefix_cannot_disappear(self):
        checkpoint = self.checkpoint()
        checkpoint["producer_record"]["posted_save_costs"]["primitive_operations"]["owner_extra_operation"] = _counts(1)
        self.reseal_record(checkpoint)
        with self.assertRaisesRegex(binding.ValidationError, "regress/disappear"):
            self.validate()

    def test_posted_save_cost_position_counter_mismatch_refused(self):
        record = self.checkpoint()["producer_record"]
        record["posted_save_costs"]["positions"]["input_returned"] -= 1
        record["sha256"] = binding.producer_digest({k: v for k, v in record.items() if k != "sha256"})
        with self.assertRaisesRegex(binding.ValidationError, "token counter/cost"):
            self.validate()

    def test_publication_must_join_exact_record(self):
        self.refused(lambda: self.checkpoint()["publication"].update(payload_sha256="0" * 64))

    def test_gd_request_is_required_but_cannot_assert_qualification(self):
        self.refused(lambda: self.checkpoint(9)["producer_record"].update(GD_export_parity_qualified=True))

    def test_missing_actual_peft_inventory_not_replaced_by_request(self):
        self.refused(lambda: self.manifest["arms"]["M1"].update(gd_exports=[]), "both original")

    def test_scientific_roles_are_step_markers_not_resume_authority(self):
        record = self.checkpoint(9)
        self.assertIn("scientific_boundary_step", record["roles"])
        self.assertNotIn("scientific_resume_boundary", record["roles"])
        self.refused(lambda: self.checkpoint(1)["roles"].append("scientific_resume_boundary"))

    def test_missing_distinct_scientific_descriptors_refused(self):
        self.refused(lambda: self.manifest["arms"]["M1"].update(scientific_boundaries=[]),
                     "three distinct scientific")

    def test_scientific_schedule_step_relabeling_refused(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["scientific_boundaries"][0].update(step=1))

    def test_retained_archive_cannot_substitute_for_scientific_snapshot(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["scientific_boundaries"][0].update(
            native_descriptor=deepcopy(self.checkpoint(9)["native_descriptor"])), "retained archives cannot")

    def test_scientific_payload_cannot_reuse_retained_path(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["scientific_boundaries"][0]["native_descriptor"].update(
            path=self.checkpoint(9)["native_descriptor"]["path"]), "duplicate retained file")

    def test_scientific_recipe_start_cursor_metadata_and_hash_join_refused(self):
        for location, field, value in (("metadata", "recipe_sha256", "0" * 64),
                                       ("metadata", "fresh_start_attestation_sha256", "0" * 64),
                                       ("descriptor", "metadata_sha256", "0" * 64)):
            with self.subTest(field=field):
                self.setUp()
                descriptor = self.manifest["arms"]["M1"]["scientific_boundaries"][0]["native_descriptor"]
                target = descriptor["metadata"] if location == "metadata" else descriptor
                self.refused(lambda: target.update({field: value}))

    def test_scientific_retained_sidecar_join_must_not_be_stale(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["scientific_boundaries"][0].update(
            retained_step_descriptor_sha256="0" * 64))

    def test_missing_gd27_or_wrong_gd_source_refused(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["gd_exports"][1].update(step=18))

    def test_stale_source_checkpoint_export_join_refused(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["gd_exports"][0].update(
            source_checkpoint_sha256="0" * 64))

    def test_peft_file_inventory_requires_actual_payload_descriptors(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["gd_exports"][0]["files"].pop())

    def test_peft_inventory_hash_stale_refused(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["gd_exports"][0].update(inventory_sha256="0" * 64))

    def test_peft_config_or_weights_cannot_escape_directory(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["gd_exports"][0]["files"][1]["file"].update(
            path="C:/Users/clclc/BusinessProject/INERT_FIXTURE_METADATA/other/adapter_model.safetensors"))

    def test_reused_peft_directory_refused(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["gd_exports"][1].update(
            directory=self.manifest["arms"]["M1"]["gd_exports"][0]["directory"]), "reused export")

    def test_separate_parity_result_and_strict_load_descriptors_required(self):
        for field in ("parity_result", "strict_load"):
            with self.subTest(field=field):
                self.setUp()
                self.refused(lambda: self.manifest["arms"]["M1"]["parity_join"]["receipts"].pop(field))

    def test_stale_parity_source_and_export_joins_refused(self):
        for field in ("source_checkpoint_sha256", "peft_export_sha256"):
            with self.subTest(field=field):
                self.setUp()
                self.refused(lambda: self.manifest["arms"]["M1"]["parity_join"]["joins"][0].update({field: "0" * 64}))

    def test_stale_parity_policy_and_probe_registration_refused(self):
        for field in ("parity_policy_sha256", "probe_registration_sha256"):
            with self.subTest(field=field):
                self.setUp()
                self.refused(lambda: self.manifest["arms"]["M1"]["parity_join"].update({field: "0" * 64}))

    def test_boolean_parity_metadata_is_not_admitted(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["parity_join"].update(parity_matched=True),
                     "exact fields")

    def test_fixture_relabeling_cannot_cross_frozen_lineage(self):
        self.refused(lambda: self.manifest.update(evidence_kind="producer_metadata"))

    def test_fixture_kind_cannot_bind_original_schedule(self):
        self.schedule["evidence_kind"] = "frozen_owner_schedule_metadata"
        self.pin = binding.digest(self.schedule)
        with self.assertRaisesRegex(binding.ValidationError, "fixture"):
            self.validate()

    def test_fixture_descendant_cannot_claim_native_pilot(self):
        self.refused(lambda: self.manifest["arms"]["M1"]["parity_join"].update(evidence_level="native_pilot"))

    def test_foreign_arm_recipe_source_runtime_or_data_refused(self):
        for field in ("recipe_sha256", "source_sha256", "runtime_sha256", "data_sha256"):
            with self.subTest(field=field):
                self.setUp()
                self.refused(lambda: self.checkpoint()["identity"].update({field: "0" * 64}))

    def test_unknown_authority_field_and_bool_integer_refused(self):
        self.refused(lambda: self.manifest.update(native_admissible=True), "exact fields")

    def test_bool_cannot_replace_completed_updates_integer(self):
        self.refused(lambda: self.checkpoint()["counters"].update(completed_updates=True))

    def test_nonjson_tuple_cannot_alias_list_fields(self):
        self.refused(lambda: self.checkpoint().update(roles=tuple(self.checkpoint()["roles"])))

    def test_nonascii_ids_and_paths_match_both_actual_canonical_conventions(self):
        schedule, manifest, pin = make_fixture(non_ascii=True)
        result = binding.validate_retention(manifest, schedule, pin)
        self.assertIs(result["native_admissible"], False)
        record = manifest["arms"]["M1"]["checkpoints"][0]["producer_record"]
        body = {k: v for k, v in record.items() if k != "sha256"}
        actual = hashlib.sha256(json.dumps(body, ensure_ascii=False, sort_keys=True,
                                          separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(record["sha256"], actual)
        self.assertNotEqual(binding.digest(body), actual)
        expected_sidecar = hashlib.sha256(json.dumps(schedule, ensure_ascii=True, sort_keys=True,
                                                   separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()
        self.assertEqual(pin, expected_sidecar)
        descriptor = record["backend_descriptor"]
        self.assertEqual(descriptor["metadata_sha256"], binding.producer_digest(descriptor["metadata"]))
        self.assertNotEqual(descriptor["metadata_sha256"], binding.digest(descriptor["metadata"]))

    def test_validation_never_mutates_inputs_and_returns_independent_lists(self):
        before = (binding.canonical(self.schedule), binding.canonical(self.manifest))
        result = self.validate()
        result["retained_steps_per_arm"].append(28)
        self.assertEqual(before, (binding.canonical(self.schedule), binding.canonical(self.manifest)))
        self.assertEqual(self.validate()["retained_steps_per_arm"], list(range(1, 28)))

    def test_nonfinite_schedule_metadata_refused(self):
        self.refused(lambda: self.schedule["learning_rates"][0].update(float_value=float("nan")), "finite JSON")


if __name__ == "__main__":
    unittest.main()
