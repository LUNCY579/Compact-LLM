"""Independent adversarial replay controls using the actual injected CPU core."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def source_module(name):
    path = HERE / name
    spec = importlib.util.spec_from_file_location("independent_test_" + path.stem, path)
    value = importlib.util.module_from_spec(spec)
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), value.__dict__)
    return value


def canonical(value):
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode("ascii")).hexdigest()


def seal(value):
    body = {key: item for key, item in value.items() if key != "sha256"}
    return {**body, "sha256": digest(body)}


class OwnedAuditorControls(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="auditor_fixture_", dir=HERE)
        self.root = Path(self.directory.name).resolve()
        self.auditor = source_module("prefix_input_owned_auditor.py")
        self.fixtures = source_module("fixture_support.py")
        self.frame = self.fixtures.terminal_frame(self.root)

    def tearDown(self):
        self.assertEqual(self.root.parent, HERE.resolve())
        self.directory.cleanup()

    def audit(self, frame=None):
        frame = self.frame if frame is None else frame
        preservation = json.loads((frame["preserved"] / "PRESERVATION.json").read_text())
        return self.auditor.audit_cpu(frame["preserved"], original_directory=frame["raw"],
            manifest=frame["manifest"], expected_manifest_sha256=frame["manifest"]["sha256"],
            source_pins=frame["manifest"]["source_pins"],
            expected_preservation_file_sha256=hashlib.sha256((frame["preserved"] / "PRESERVATION.json").read_bytes()).hexdigest(),
            expected_preservation_body_sha256=preservation["sha256"])

    def read(self, name):
        return json.loads((self.frame["raw"] / name).read_text())

    def write_pair(self, name, value, *, ledger=False):
        raw = (("".join(canonical(row) + "\n" for row in value)) if ledger else canonical(value) + "\n").encode("ascii")
        for root in (self.frame["raw"], self.frame["preserved"]):
            (root / name).write_bytes(raw)
        return raw

    def resign_preservation(self):
        path = self.frame["preserved"] / "PRESERVATION.json"
        value = json.loads(path.read_text())
        for name in value["raw_files"]:
            raw = (self.frame["raw"] / name).read_bytes()
            value["raw_files"][name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        path.write_text(canonical(seal(value)) + "\n", encoding="ascii")

    def mutate_terminal(self, callback):
        terminal = self.read("OWNED_TERMINAL.json")
        callback(terminal)
        self.write_pair("OWNED_TERMINAL.json", seal(terminal))
        self.resign_preservation()

    def test_complete_original_owned_CPU_proof_has_no_native_authority(self):
        value = self.audit()
        self.assertEqual(value["status"], "CPU_AUDIT_COMPLETE")
        self.assertTrue(value["independent_replay_complete"])
        self.assertTrue(value["fixture_only"])
        for name in ("native_authority_issued", "native_PASS", "native_fit_established", "scoring_authorized", "M0_membership", "cause_established"):
            self.assertIs(value[name], False)
        self.assertEqual(value["sha256"], digest({key: item for key, item in value.items() if key != "sha256"}))

    def test_actual_false_predicate_is_preserved_as_failure_not_complete(self):
        other_root = self.root / "failed_case"
        other_root.mkdir()
        frame = self.fixtures.terminal_frame(other_root, api_faults={"ids.is_tensor": False})
        value = self.audit(frame)
        self.assertEqual(value["status"], "CPU_AUDIT_FAILURE_PRESERVED")
        self.assertFalse(value["native_PASS"])

    def test_pre_core_admission_failure_is_partial_metadata_never_complete(self):
        other_root = self.root / "admission_failure"
        other_root.mkdir()
        frame = self.fixtures.terminal_frame(other_root, host_faults={"observation": {"invalid": True}})
        value = self.audit(frame)
        self.assertEqual(value["status"], "CPU_AUDIT_FAILURE_PRESERVED")
        self.assertFalse(value["independent_replay_complete"])
        self.assertTrue(value["missing_core_files"])
        self.assertFalse(value["native_PASS"])

    def test_resealed_missing_or_reordered_marker_is_rejected(self):
        original = [json.loads(line) for line in (self.frame["raw"] / "PHASE_OBSERVATIONS.jsonl").read_text().splitlines()]
        for records in (original[:-1], [original[1], original[0], *original[2:]]):
            self.write_pair("PHASE_OBSERVATIONS.jsonl", records, ledger=True)
            self.resign_preservation()
            with self.subTest(rows=len(records)), self.assertRaises(ValueError):
                self.audit()

    def test_resealed_nonboolean_predicate_cannot_pass_as_truthy(self):
        records = [json.loads(line) for line in (self.frame["raw"] / "PHASE_OBSERVATIONS.jsonl").read_text().splitlines()]
        next(row for row in records if row["operation"] == "input_shape_matches" and row["event"] == "returned")["result"] = 1
        self.write_pair("PHASE_OBSERVATIONS.jsonl", records, ledger=True)
        self.resign_preservation()
        with self.assertRaises(ValueError):
            self.audit()

    def test_fully_resealed_guard_child_omission_cannot_pass_on_counter_equality(self):
        records = [json.loads(line) for line in (self.frame["raw"] / "PHASE_OBSERVATIONS.jsonl").read_text().splitlines()]
        records = [row for row in records if row["operation"] not in ("guard_cuda_is_initialized", "guard_cuda_mem_get_info")]
        ordinal_map = {}
        for row in records:
            original = row["ordinal"]
            if row["event"] == "attempt":
                ordinal_map[original] = len(ordinal_map) + 1
            row["ordinal"] = ordinal_map[original]
        raw = self.write_pair("PHASE_OBSERVATIONS.jsonl", records, ledger=True)
        counts = {}
        for row in records:
            item = counts.setdefault(row["operation"], {"attempted": 0, "returned": 0, "failed": 0, "pending": 0})
            item["attempted" if row["event"] == "attempt" else row["event"]] += 1
        final = self.read("PHASE_ACCOUNTING_FINAL.json")
        final["operations"] = counts
        final["markers"] = len(records)
        final["marker_bytes"] = len(raw)
        final["IO"]["marker_write"] = {"attempted": len(records), "returned": len(records), "failed": 0, "pending": 0}
        self.write_pair("PHASE_ACCOUNTING_FINAL.json", final)
        terminal = self.read("OWNED_TERMINAL.json")
        returned_report = terminal["worker_report"]
        returned_report.update(final_phase_summary=final, operation_accounting=counts, IO_accounting=final["IO"])
        terminal["worker_report_sha256"] = digest(returned_report)
        self.write_pair("OWNED_TERMINAL.json", seal(terminal))
        self.resign_preservation()
        with self.assertRaises(ValueError):
            self.audit()

    def test_resealed_loaded_guard_memory_none_or_negative_is_rejected(self):
        original = [json.loads(line) for line in (self.frame["raw"] / "PHASE_OBSERVATIONS.jsonl").read_text().splitlines()]
        for result in (None, [-1, 24 * 1024**3], [True, 24 * 1024**3]):
            records = deepcopy(original)
            next(row for row in records if row["operation"] == "guard_cuda_mem_get_info" and row["event"] == "returned")["result"] = result
            self.write_pair("PHASE_OBSERVATIONS.jsonl", records, ledger=True)
            self.resign_preservation()
            with self.subTest(result=result), self.assertRaises(ValueError):
                self.audit()

    def test_resealed_admission_only_samples_do_not_cover_actual_work(self):
        def admission_only(terminal):
            first = terminal["admission_observations"][0]
            terminal["resource_samples"] = {"CPU": [{"monotonic": first["monotonic"], "available_bytes": first["available_bytes"], "worker_RSS_bytes": 0}],
                "GPU": [{"monotonic": first["GPU_monotonic"], "free_bytes": first["free_bytes"], "total_bytes": first["total_bytes"]}]}
        self.mutate_terminal(admission_only)
        with self.assertRaises(ValueError):
            self.audit()

    def test_resealed_bad_owned_flags_death_birth_and_timeline_are_rejected(self):
        original = self.read("OWNED_TERMINAL.json")
        mutations = (lambda value: value["terminal_flags"].update(job_closed=False),
                     lambda value: value["current_death"][0]["identity"].update(creation_filetime_100ns=2),
                     lambda value: value["current_death"][0].update(currently_alive=True),
                     lambda value: value["timeline"].reverse())
        for mutate in mutations:
            terminal = deepcopy(original)
            mutate(terminal)
            self.write_pair("OWNED_TERMINAL.json", seal(terminal))
            self.resign_preservation()
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.audit()

    def test_resealed_stale_admission_or_lowered_resource_floor_is_rejected(self):
        original = self.read("OWNED_TERMINAL.json")
        for change in (lambda terminal: terminal["admission_observations"][2].update(monotonic=0),
                       lambda terminal: terminal["resource_samples"]["CPU"][0].update(available_bytes=0),
                       lambda terminal: terminal["resource_samples"]["GPU"][0].update(free_bytes=0)):
            terminal = deepcopy(original)
            change(terminal)
            self.write_pair("OWNED_TERMINAL.json", seal(terminal))
            self.resign_preservation()
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.audit()

    def test_original_and_preserved_raw_byte_pair_cannot_diverge(self):
        (self.frame["raw"] / "ownership_verified.json").write_bytes(b'{"scope":"changed"}\n')
        with self.assertRaises(ValueError):
            self.audit()

    def test_source_snapshot_or_extra_source_roster_is_rejected(self):
        snapshots = self.frame["preserved"] / "source_snapshots"
        (snapshots / "unexpected.py").write_bytes(b"# CPU extra source\n")
        with self.assertRaises(ValueError):
            self.audit()

    def test_duplicate_and_overflowing_JSON_reject_before_semantics(self):
        for raw in (b'{"a":1,"a":2}', b'{"ignored":1e999}', b'{"ignored":NaN}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                self.auditor.parse(raw)

    def test_cpu_to_native_conversion_and_public_native_probe_path_are_closed(self):
        manifest = deepcopy(self.frame["manifest"])
        manifest["native_launch_authorized"] = True
        manifest = seal(manifest)
        with self.assertRaises(ValueError):
            self.auditor.audit_cpu(self.frame["preserved"], original_directory=self.frame["raw"], manifest=manifest,
                expected_manifest_sha256=manifest["sha256"], source_pins=manifest["source_pins"],
                expected_preservation_file_sha256="a" * 64, expected_preservation_body_sha256="b" * 64)
        with patch.object(self.auditor, "Reader", side_effect=AssertionError("native paths read")):
            with self.assertRaises(ValueError):
                self.auditor.native_audit("not_read", native_launch_authorized=True, native_runtime_closure_complete=True)

    def test_explicit_CPU_native_schema_fixture_runs_actual_native_adapter_without_pass(self):
        # These are labelled CPU metadata conversions, never actual native
        # outputs. The real native adapter runs every file/schema/replay join;
        # only its direct-package gate is patched inside this isolated control.
        root = self.root / "CPU_NATIVE_SCHEMA_ORCHESTRATION_TEST"
        root.mkdir()
        frame = self.fixtures.terminal_frame(root)
        original, preserved = frame["raw"], frame["preserved"]
        cpu_terminal = frame["terminal"]
        owner = dict(cpu_terminal["worker_identity"])
        times = {row["event"]: row["monotonic"] for row in cpu_terminal["timeline"]}
        native_scope = self.auditor.NATIVE_SCOPE
        native_status = "NATIVE_INPUT_CHECKS_RETURNED_PENDING_TERMINAL_AUDIT"
        manifest = {"scope": native_scope, "permission": self.auditor.PERMISSION,
                    "resources": self.auditor.RESOURCES, "source_files": frame["manifest"]["source_pins"],
                    "runtime_files": {}, "input_file_sha256": self.auditor.INPUT_FILE,
                    "input_body_sha256": self.auditor.INPUT_BODY, "prefix_ids_sha256": self.auditor.PREFIX_SHA}
        manifest = seal(manifest)
        anchors = {"config_path": str(root / "CPU_METADATA_NATIVE_CONFIG.json"),
                   "anchors": {"expected_config_file": "b" * 64}}
        def pair(name, value):
            raw = (canonical(value) + "\n").encode("ascii")
            for destination in (original, preserved):
                (destination / name).write_bytes(raw)
        worker = json.loads((original / "worker_result.json").read_text())
        worker.update(scope=native_scope, status=native_status)
        pair("worker_result.json", worker)
        pair("ownership_verified.json", {"scope": native_scope})
        final = json.loads((original / "PHASE_ACCOUNTING_FINAL.json").read_text())
        final.update(scope=native_scope, status=native_status)
        pair("PHASE_ACCOUNTING_FINAL.json", final)
        report = deepcopy(cpu_terminal["worker_report"])
        report.update(scope=native_scope, status=native_status, final_phase_summary=final)
        modules = {"CPU_METADATA_MODULE": {"path": next(iter(manifest["source_files"])),
                   "sha256": next(iter(manifest["source_files"].values()))["sha256"]}}
        pair("OWNED_WORKER_RETURN.json", {"report": report, "runtime_execution_modules": modules,
            "owned_membership_verified": {"actual_worker_pid": owner["pid"], "actual_parent_pid": 12344,
                "launcher_root_pid": owner["pid"], "supervisor_pid": 12344, "job_name": "CPU_METADATA_JOB",
                "owned_job_membership_verified": True, "query_job_handle_closed_before_framework_import": True},
            "IO": {"owned_return_write": {"attempted": 1, "returned": 1, "failed": 0, "pending": 0}}, "native_PASS": False})
        pair("OWNED_EXTERNAL_ANCHORS.json", anchors)
        def observation(value):
            return {"sample_started_monotonic": value["monotonic"], "sample_completed_monotonic": value["monotonic"],
                "host_available_bytes": value["available_bytes"], "worker_tree_rss_bytes": value["worker_RSS_bytes"],
                "disk_free_bytes": value["disk_free_bytes"], "gpu_sample_started_monotonic": value["GPU_monotonic"],
                "gpu_sample_completed_monotonic": value["GPU_monotonic"], "gpu_free_bytes": value["free_bytes"],
                "gpu_total_bytes": value["total_bytes"], "GPU_index": 0, "gpu_uuid": value["GPU_identity"],
                "gpu_observation_id": str(value["GPU_monotonic"])}
        admissions = [observation(value) for value in cpu_terminal["admission_observations"]]
        full_owner = {**owner, "assigned_before_resume": True, "job_kill_on_close": True,
                      "breakaway_allowed": False, "job_name": "CPU_METADATA_JOB"}
        lease_root = root / "CPU_METADATA_LEASE"
        lease_root.mkdir()
        supervisor = {"config_sha256": "b" * 64, "preflight": {"source_commit": "a" * 40, "resources": self.auditor.RESOURCES},
            "worker_attempted": True, "worker_launched": True, "worker_resumed": True,
            "preflight_revalidated_before_current_observations": True, "job_closed_kill_on_close": True,
            "sampler_threads_terminal_verified": True, "gpu_telemetry_cleanup_verified": True, "partial_outputs_frozen": True,
            "owned_worker_identity_currently_alive": False, "unjoined_sampler_threads": [],
            "elapsed_seconds": times["lease_terminal"] - times["admission_first"], "worker_identity": full_owner,
            "worker_resume_monotonic": times["worker_resumed"], "deadline_monotonic": times["worker_resumed"] + 600,
            "admission_observations": admissions[:2], "dispatch_observation": admissions[2], "lease_path": str(lease_root),
            "status": "SUPERVISED_WORKER_EXITED_ZERO", "returncode": 0}
        pair("SUPERVISOR_RECEIPT.json", supervisor)
        pair("supervisor_ownership.json", {"worker_pid": owner["pid"], "supervisor_pid": 12344, "job_name": "CPU_METADATA_JOB", "config_sha256": "b" * 64})
        samples = []
        gpu_id = admissions[0]["gpu_uuid"]
        for row in cpu_terminal["resource_samples"]["CPU"]:
            samples.append({"kind": "CPU", "sample_started_monotonic": row["monotonic"],
                "host_available_bytes": row["available_bytes"], "worker_tree_rss_bytes": row["worker_RSS_bytes"], "owned_process_identities": [owner]})
        for row in cpu_terminal["resource_samples"]["GPU"]:
            samples.append({"kind": "GPU", "gpu_sample_started_monotonic": row["monotonic"],
                "gpu_free_bytes": row["free_bytes"], "gpu_total_bytes": row["total_bytes"], "gpu_uuid": gpu_id})
        samples.sort(key=lambda row: row.get("sample_started_monotonic", row.get("gpu_sample_started_monotonic")))
        pair("RESOURCE_OBSERVATIONS.json", samples)
        leases = {"reservation.json": {"config_sha256": "b" * 64}, "worker.json": full_owner,
            "resume.json": {"worker_identity": full_owner, "external_monitor_ready_before_resume": True,
                            "resumed_monotonic": times["worker_resumed"]}, "terminal.json": supervisor}
        lease_files = {}
        for name, value in leases.items():
            raw = (canonical(value) + "\n").encode("ascii")
            (lease_root / name).write_bytes(raw)
            (preserved / ("lease_" + name)).write_bytes(raw)
            lease_files[name] = {"source_path": str(lease_root / name), "path": "lease_" + name,
                                 "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        preservation = deepcopy(frame["preservation"])
        preservation.update(kind="PRESERVED_REGISTERED_ONE_PREFIX_NATIVE_TENSOR_DIAGNOSTIC", scope=native_scope,
                            manifest_sha256=manifest["sha256"], source_files=manifest["source_files"],
                            fixture_only=False, native_authority_issued=False, lease_files=lease_files)
        preservation["raw_files"] = {path.name: {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                     for path in original.iterdir()}
        preservation = seal(preservation)
        (preserved / "PRESERVATION.json").write_text(canonical(preservation) + "\n", encoding="ascii")
        source_pins = {path: {"bytes": value["bytes"], "sha256": value["sha256"]} for path, value in manifest["source_files"].items()}
        with patch.object(self.auditor, "_direct_package", return_value=True):
            value = self.auditor._native_preserved_body(preserved, original_directory=original, manifest=manifest,
                source_pins=source_pins, expected_preservation_file_sha256=hashlib.sha256((preserved / "PRESERVATION.json").read_bytes()).hexdigest(),
                expected_preservation_body_sha256=preservation["sha256"], expected_head="a" * 40, anchors=anchors,
                _CPU_fixture_capability=self.auditor._mint_CPU_native_schema_fixture())
        self.assertEqual(value["status"], "CPU_NATIVE_SCHEMA_COMPLETE")
        self.assertIs(value["fixture_only"], True)
        self.assertFalse(value["native_PASS"])
        self.assertFalse(value["native_authority_issued"])
        self.assertFalse(value["native_fit_established"])


if __name__ == "__main__":
    unittest.main()
