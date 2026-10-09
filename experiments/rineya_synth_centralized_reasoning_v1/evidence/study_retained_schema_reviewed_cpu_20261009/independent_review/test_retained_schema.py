"""Targeted authored metadata/schema refusal controls; no framework or real data."""
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace
import ast
import hashlib
import json
import sys
import time
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
core = ModuleType("exact_schema_authored_controls"); core.__file__ = str(HERE / "test_retained_updates.py")
exec(compile((HERE / "test_retained_updates.py").read_bytes(), core.__file__, "exec"), core.__dict__)
chunk, schedule = core.chunk, core.schedule
VERSIONS = (True, 1.0, "1", None, 0, 2, "MISSING")


class SchemaControls(core.RetentionControls):
    def baseline(self):
        obj, preserve, policy = self.fixture()
        result = self.run_part(obj, preserve, policy)
        self.assertEqual("COMPLETE_PROTOCOL_CHUNK_STUDY_UNAUTHORIZED", result["status"], result.get("error"))
        descriptor = result["resume_descriptor"]["envelope"]["backend_descriptor"]
        self.assertEqual(chunk.METADATA_FIELDS, set(descriptor["metadata"]))
        return result

    def deny_reads(self):
        reads = []
        def forbidden(*args): reads.append(args); raise AssertionError("malformed schema reached byte read")
        patch.object(chunk, "_read_bytes", forbidden).start()
        return reads

    def version(self, value, version):
        if version == "MISSING": value.pop("schema_version")
        else: value["schema_version"] = version

    def test_scientific_inner_kind_and_retained_only_fields_refuse_before_bytes(self):
        result = self.baseline(); scientific = result["resume_descriptor"]["envelope"]["backend_descriptor"]
        reads = self.deny_reads()
        for transform in (
                lambda d: d["metadata"].update(kind=chunk.RETAINED_BACKEND_KIND),
                lambda d: d["metadata"].update(resumable=False),
                lambda d: d["metadata"].update(producer_policy_sha256="0" * 64)):
            malformed = deepcopy(scientific); transform(malformed)
            malformed["metadata_sha256"] = schedule.digest(malformed["metadata"])
            with self.subTest(transform=transform), self.assertRaisesRegex(ValueError, "scientific metadata kind/schema"):
                chunk._descriptor(malformed, malformed["metadata"]["binding"], result["recipe"])
        malformed = deepcopy(result["retained_updates"][-1]["record"]["backend_descriptor"])
        malformed["kind"] = chunk.BACKEND_KIND
        with self.assertRaisesRegex(ValueError, "scientific metadata kind/schema"):
            chunk._descriptor(malformed, malformed["metadata"]["binding"], result["recipe"])
        self.assertEqual([], reads)

    def test_scientific_outer_and_inner_versions_are_exact_integer_one(self):
        result = self.baseline(); original = result["resume_descriptor"]["envelope"]["backend_descriptor"]
        reads = self.deny_reads()
        for level in ("outer", "inner"):
            for version in VERSIONS:
                malformed = deepcopy(original)
                self.version(malformed if level == "outer" else malformed["metadata"], version)
                malformed["metadata_sha256"] = schedule.digest(malformed["metadata"])
                with self.subTest(level=level, version=version), self.assertRaises(ValueError):
                    chunk._descriptor(malformed, original["metadata"]["binding"], result["recipe"])
        self.assertEqual([], reads)

    def test_retained_inner_species_versions_and_positive_production_shape(self):
        result = self.baseline(); original = result["retained_updates"][-1]["record"]["backend_descriptor"]
        chunk._retained_descriptor(original, original["metadata"]["binding"], result["recipe"])
        self.assertEqual(chunk.METADATA_FIELDS | {"resumable", "producer_policy_sha256"}, set(original["metadata"]))
        reads = self.deny_reads()
        malformed = deepcopy(original); malformed["metadata"]["kind"] = chunk.BACKEND_KIND
        malformed["metadata_sha256"] = schedule.digest(malformed["metadata"])
        with self.assertRaises(ValueError): chunk._retained_descriptor(malformed, original["metadata"]["binding"], result["recipe"])
        for level in ("outer", "inner"):
            for version in VERSIONS:
                malformed = deepcopy(original)
                self.version(malformed if level == "outer" else malformed["metadata"], version)
                malformed["metadata_sha256"] = schedule.digest(malformed["metadata"])
                with self.subTest(level=level, version=version), self.assertRaises(ValueError):
                    chunk._retained_descriptor(malformed, original["metadata"]["binding"], result["recipe"])
        self.assertEqual([], reads)

    def test_resealed_resume_and_envelope_versions_refuse_before_archive_or_publication_reads(self):
        result = self.baseline(); original = result["resume_descriptor"]
        reads = self.deny_reads()
        for level in ("resume", "envelope"):
            for version in VERSIONS:
                malformed = deepcopy(original)
                self.version(malformed if level == "resume" else malformed["envelope"], version)
                malformed["envelope"]["sha256"] = schedule.digest({k: v for k, v in malformed["envelope"].items() if k != "sha256"})
                malformed["sha256"] = schedule.digest({k: v for k, v in malformed.items() if k != "sha256"})
                with self.subTest(level=level, version=version), self.assertRaises(ValueError):
                    chunk._validate_resume(malformed, core.REGISTRATION, "M1", 2, result["recipe"], original["terminal_costs"])
        self.assertEqual([], reads)

    def test_resealed_retained_record_versions_refuse_before_archive_or_publication_reads(self):
        result = self.baseline(); original = result["retained_updates"][0]
        reads = self.deny_reads()
        for version in VERSIONS:
            malformed = deepcopy(original); self.version(malformed["record"], version)
            malformed["record"]["sha256"] = schedule.digest({k: v for k, v in malformed["record"].items() if k != "sha256"})
            with self.subTest(version=version), self.assertRaises(ValueError):
                chunk._retained_entry(malformed, core.REGISTRATION, "M1", 1, result["recipe"])
        self.assertEqual([], reads)

    def test_final_restore_and_archive_reject_retyped_retained_metadata_before_reads_deserialization(self):
        result = self.baseline()
        source = ast.parse((HERE / "final_adapter_inference.py").read_bytes())
        definitions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name in ("restore_final", "_validate_archive")]
        calls = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Calls")
        ns = {"chunks": chunk, "backend": core.backend, "deepcopy": deepcopy, "require": chunk.require,
              "FIXTURE_SCOPE": "AUTHORED", "NATIVE_SCOPE": "AUTHORED_NATIVE_TAG"}
        exec(compile(ast.Module(body=[calls, *definitions], type_ignores=[]), str(HERE / "final_adapter_inference.py"), "exec"), ns)
        original = deepcopy(result["retained_updates"][-1]["record"]["backend_descriptor"])
        original["kind"] = chunk.BACKEND_KIND
        binding = deepcopy(original["metadata"]["binding"])
        binding.update(completed_updates=27, consumed_occurrences=540, next_update_ordinal=28, learning_rate=0.0)
        original["metadata"].update(scope="CPU_RANDOM_TOY", binding=binding)
        original["metadata_sha256"] = schedule.digest(original["metadata"])
        ns["loader"] = SimpleNamespace(_adapter_shapes=lambda: {})
        reads = []; loads = []
        torch = SimpleNamespace(cuda=SimpleNamespace(is_initialized=lambda: False),
                                load=lambda *args, **kwargs: loads.append(args))
        fixture = {"kind": "CPU_RANDOM_TINY_FINAL27_RECIPE_FIXTURE_ONLY", "descriptor": original,
                   "binding": binding, "profile": {"expected_adapter_shapes": {}, "expected_layers": 1,
                       "expected_linear_sites": 1}, "expected_recipe_sha256": original["metadata"]["recipe_sha256"],
                   "cohort_sha256": original["metadata"]["cohort_sha256"],
                   "expected_backend_source_sha256": original["metadata"]["backend_source_sha256"]}
        value = ns["restore_final"](torch, SimpleNamespace(), lambda d: reads.append(d),
            self.journal, self.guard, arm="M1", fixture=fixture)
        self.assertFalse(value["inference_allowed"])
        self.assertIn("scientific metadata kind/schema", value["error"]["message"])
        with self.assertRaisesRegex(ValueError, "scientific metadata kind/schema"):
            ns["_validate_archive"](torch, {"metadata": original["metadata"], "state": {}, "occurrence_order": []},
                original, binding, {}, [], set())
        scientific = deepcopy(result["resume_descriptor"]["envelope"]["backend_descriptor"])
        payload = {"metadata": deepcopy(scientific["metadata"]), "state": {}, "occurrence_order": []}
        payload["metadata"]["kind"] = chunk.RETAINED_BACKEND_KIND
        with self.assertRaisesRegex(ValueError, "scientific archive metadata kind/schema"):
            ns["_validate_archive"](torch, payload, scientific, binding, {}, [], set())
        self.assertEqual([], reads); self.assertEqual([], loads)


if __name__ == "__main__":
    names = [name for name in SchemaControls.__dict__ if name.startswith("test_")]
    sources = ("study_native_chunk.py", "final_adapter_inference.py", "study_torch_backend.py",
               "test_retained_updates.py", "test_retained_schema.py")
    before = {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in sources}
    start = time.monotonic(); suite = unittest.TestSuite(SchemaControls(name) for name in names)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    receipt = {"kind": "AUTHORED_RETAINED_SCHEMA_CPU_CONTROLS001", "tests": result.testsRun,
               "failures": len(result.failures), "errors": len(result.errors), "success": result.wasSuccessful(),
               "seconds": time.monotonic() - start, "source_before": before,
               "source_after": {name: hashlib.sha256((HERE / name).read_bytes()).hexdigest() for name in sources},
               "read_paths": sorted(set(core.READS)), "forbidden_import_attempts": core.FORBIDDEN_IMPORTS,
               "native_framework_model_optimizer_performed": False, "actual_export_parity_established": False,
               "fixtures": "Authored full-shape descriptor metadata, scalar archives and exact resealed documents only."}
    receipt["sha256"] = schedule.digest(receipt)
    (HERE / "SCHEMA_CONTROLS001.json").write_bytes((schedule.canonical(receipt) + "\n").encode())
    raise SystemExit(0 if result.wasSuccessful() else 1)
