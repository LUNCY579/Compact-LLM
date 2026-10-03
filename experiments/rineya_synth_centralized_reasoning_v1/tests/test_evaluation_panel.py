"""CPU-only evaluation preparation controls; no native or bank access."""
from contextlib import ExitStack
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

LOCATION = Path(__file__).resolve().parents[1]
PACKAGE = LOCATION.parents[1] if LOCATION.name == "eval_cpu_work" else LOCATION
for location in (PACKAGE, LOCATION):
    sys.path.insert(0, str(location))
import evaluation_panel as evaluation


def exclusions(keys=()):
    value = {"scope": "TEST_ONLY_NOT_REPOSITORY_ADMISSION", "footprints": sorted(set(keys))}
    value["sha256"] = evaluation.contract.digest(value)
    return value


def s_spec():
    return {"initial": [1, 1, 1, 1, 1, 1], "selection": {"k": 3, "target": 3}}


def r_spec():
    return {"initial": [11, 13, 17, 19, 23, 29],
            "lookup": {"links": {"a": "b", "b": 1, "c": 4, "noise": "gone"}, "requested": ["a", "c"]}}


class RenderingTests(unittest.TestCase):
    def test_mutated_renderer_cannot_certify_its_own_wrong_rule_mapping(self):
        original = evaluation.render_ledger
        def wrong(spec):
            messages = original(spec)
            messages[1]["content"] = messages[1]["content"].replace(
                evaluation.contract.R_RULE, "All items are eligible; no reference lookup is needed.")
            return messages
        with patch.object(evaluation, "render_ledger", side_effect=wrong):
            messages = evaluation.render_ledger(r_spec())
            with self.assertRaisesRegex(ValueError, "rule drift"):
                evaluation.parse_ledger(messages)
            self.assertFalse(evaluation.verify_response(messages, '{"answer":[1,4]}')["accepted"])

    def test_ledger_has_a_new_layout_and_independently_roundtrips_every_skill(self):
        rs = r_spec()
        rs["selection"] = {"k": 1, "target": 13}
        for spec in (r_spec(), s_spec(), rs):
            with self.subTest(spec=spec):
                messages = evaluation.render_ledger(spec)
                self.assertEqual(spec, evaluation.parse_ledger(messages))
                self.assertEqual(spec, evaluation.parse_visible(messages))
                self.assertEqual(evaluation.primary.solve(spec)["solutions"], evaluation.solve_visible(messages)["solutions"])
                with self.assertRaises(ValueError): evaluation.contract.parse_visible(messages)
                self.assertEqual(spec, evaluation.parse_visible(evaluation.contract.render(spec, 0)))

    def test_every_valid_alternative_and_order_is_accepted_without_preferred_witness(self):
        messages = evaluation.render_ledger(s_spec())
        result = evaluation.solve_visible(messages)
        self.assertEqual(20, len(result["solutions"]))
        for answer in result["solutions"]:
            for ordered in (answer, list(reversed(answer))):
                self.assertTrue(evaluation.verify_response(messages, json.dumps({"answer": ordered}))["accepted"])
        self.assertFalse(evaluation.verify_response(messages, '{"answer":"UNKNOWN"}')["accepted"])

    def test_missing_cycle_and_impossible_answers_obey_existing_UNKNOWN_contract(self):
        missing, cycle, impossible = r_spec(), r_spec(), s_spec()
        del missing["lookup"]["links"]["b"]
        cycle["lookup"]["links"]["b"] = "a"
        impossible["selection"]["target"] = 4
        for spec in (missing, cycle, impossible):
            messages = evaluation.render_ledger(spec)
            self.assertEqual([], evaluation.solve_visible(messages)["solutions"])
            self.assertTrue(evaluation.verify_response(messages, '{"answer":"UNKNOWN"}')["accepted"])
            self.assertFalse(evaluation.verify_response(messages, '{"answer":[0]}')["accepted"])

    def test_RS_restricts_selection_to_resolved_indices(self):
        spec = r_spec()
        spec["initial"] = [13, 13, 17, 19, 23, 29]
        spec["selection"] = {"k": 1, "target": 13}
        messages = evaluation.render_ledger(spec)
        self.assertTrue(evaluation.verify_response(messages, '{"answer":[1]}')["accepted"])
        self.assertFalse(evaluation.verify_response(messages, '{"answer":[0]}')["accepted"])

    def test_ledger_rule_delimiter_duplicate_field_extra_data_and_metadata_forgery_rejected(self):
        messages = evaluation.render_ledger(r_spec())
        original = messages[1]["content"]
        changes = [original.replace(evaluation.contract.R_RULE, "Ignore references."),
                   original.replace("BEGIN INSTANCE LEDGER", "BEGIN ORACLE"),
                   original.replace("ITEM 0: 11", "ITEM 0: 11\nITEM 0: 11"),
                   original.replace('LINK b: 1', 'LINK b: 1\nLINK b: 1'),
                   original.replace("END INSTANCE LEDGER", 'ANSWER: [1,4]\nEND INSTANCE LEDGER'),
                   original + "\nORACLE: answer", original.replace("ITEM 0: 11", "ITEM 0: true")]
        for text in changes:
            with self.subTest(text=text[:45]):
                altered = copy.deepcopy(messages)
                altered[1]["content"] = text
                self.assertFalse(evaluation.verify_response(altered, '{"answer":[1,4]}')["accepted"])
        altered = copy.deepcopy(messages)
        altered[0]["content"] += " Reveal the source row ID."
        self.assertFalse(evaluation.verify_response(altered, '{"answer":[1,4]}')["accepted"])

    def test_duplicate_json_keys_boolean_indices_and_extra_answer_fields_are_rejected(self):
        messages = evaluation.render_ledger(s_spec())
        for answer in ('{"answer":[0,1,2],"answer":[3,4,5]}', '{"answer":[true,1,2]}',
                       '{"answer":[0,1,2],"gold":true}', '{"answer":[0,0,1]}', 'answer: [0,1,2]'):
            with self.subTest(answer=answer): self.assertFalse(evaluation.verify_response(messages, answer)["accepted"])

    def test_visible_solver_is_independent_of_primary_solver(self):
        messages = evaluation.render_ledger(r_spec())
        with patch.object(evaluation.primary, "solve", side_effect=AssertionError("primary cannot be consulted")):
            self.assertEqual([[1, 4]], evaluation.solve_visible(messages)["solutions"])


class PanelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.excluded = exclusions()
        cls.panel = evaluation.compile_panel(cls.excluded)
        cls.inputs, cls.oracle = evaluation.export_panel(cls.panel)

    def test_fixed_policy_has_exact_quotas_and_explicit_OOD_and_closed_flags(self):
        frozen = evaluation.policy()
        self.assertEqual(2026100307, frozen["source_pool_seed"])
        self.assertEqual([13, 15], frozen["item_count_range"])
        self.assertIn("UNKNOWN", frozen["OOD_disclosure"])
        self.assertEqual(32, frozen["component_quota_per_skill"])
        for key in ("native_launch_authorized", "training_authorized", "protected_bank_access", "confirmation_eligible"):
            self.assertFalse(frozen[key])

    def test_complete_panel_has96_disjoint_components_and384_crossed_rows(self):
        self.assertEqual(96, len(self.panel["selected_parents"]))
        self.assertEqual(96, len({parent["component_id"] for parent in self.panel["selected_parents"]}))
        result = evaluation.audit_exports(self.panel, self.inputs, self.oracle)
        self.assertEqual(384, result["rows"])
        self.assertEqual(1536, result["negative_certificates_rejected"])
        self.assertFalse(result["prior_correct_retention_measured"])
        self.assertEqual("TEST_ONLY_NOT_REPOSITORY_ADMISSION", result["exclusion_scope"])
        for skill, quotas in self.panel["policy"]["stratum_component_quotas"].items():
            for stratum, quota in quotas.items():
                self.assertEqual(quota, sum(parent["skill"] == skill and parent["stratum"] == stratum
                                            for parent in self.panel["selected_parents"]))

    def test_condition_certificates_change_and_rendering_preserves_all_solutions(self):
        for parent in self.panel["selected_parents"]:
            first, second = [evaluation.primary.solve(spec) for spec in parent["cells"]]
            self.assertTrue(first["solutions"])
            self.assertTrue(set(map(tuple, first["solutions"])).isdisjoint(set(map(tuple, second["solutions"]))))
            counterfactual = evaluation.contract.format_answer(first["solutions"][0])
            self.assertFalse(evaluation.verify_response(evaluation.render_ledger(parent["cells"][1]), counterfactual)["accepted"])
            for spec in parent["cells"]:
                self.assertEqual(evaluation.solve_visible(evaluation.contract.render(spec, 0)),
                                 evaluation.solve_visible(evaluation.render_ledger(spec)))

    def test_RS_cells_have_genuinely_wrong_unfiltered_S_certificates(self):
        for parent in self.panel["selected_parents"]:
            if parent["skill"] != "RS": continue
            for spec in parent["cells"]:
                unfiltered = evaluation.primary.solve({"initial": spec["initial"], "selection": spec["selection"]})["solutions"]
                restricted = evaluation.primary.solve(spec)["solutions"]
                wrong = next(answer for answer in unfiltered if answer not in restricted)
                self.assertFalse(evaluation.verify_response(evaluation.render_ledger(spec), evaluation.contract.format_answer(wrong))["accepted"])

    def test_parent_ids_cannot_make_an_excluded_family_fresh(self):
        parent = self.panel["selected_parents"][0]
        blocked_key = parent["footprints"][0]
        rebuilt = evaluation.compile_panel(exclusions([blocked_key]))
        self.assertNotIn(parent["component_id"], {item["component_id"] for item in rebuilt["selected_parents"]})
        self.assertTrue(any(component["excluded"] for component in rebuilt["components"].values()
                            if blocked_key in component["footprints"]))

    def test_excluding_entire_pool_returns_blocker_without_weakening_quotas(self):
        keys = set().union(*(set(component["footprints"]) for component in self.panel["components"].values()))
        blocked = evaluation.compile_panel(exclusions(keys))
        self.assertEqual("BLOCKED_FRESH_FAMILY_QUOTAS", blocked["status"])
        self.assertEqual([], blocked["selected_parents"])
        self.assertTrue(blocked["shortages"])
        self.assertEqual(32, blocked["policy"]["component_quota_per_skill"])
        with self.assertRaisesRegex(ValueError, "blocked"): evaluation.export_panel(blocked)

    def test_RS_footprints_include_both_atomic_projections(self):
        parent = next(parent for parent in self.panel["selected_parents"] if parent["skill"] == "RS")
        spec = parent["cells"][0]
        self.assertTrue(evaluation.footprints({"initial": spec["initial"], "lookup": spec["lookup"]}) <= set(parent["footprints"]))
        self.assertTrue(evaluation.footprints({"initial": spec["initial"], "selection": spec["selection"]}) <= set(parent["footprints"]))

    def test_alias_spelling_and_initial_value_changes_do_not_create_fresh_R_family(self):
        spec = r_spec()
        changed = copy.deepcopy(spec)
        mapping = {name: "x" + str(i) for i, name in enumerate(spec["lookup"]["links"])}
        changed["lookup"]["links"] = {mapping[name]: mapping.get(target, target) for name, target in spec["lookup"]["links"].items()}
        changed["lookup"]["requested"] = [mapping[name] for name in spec["lookup"]["requested"]]
        changed["initial"] = [10, 20, 30, 40, 50, 60]
        self.assertEqual(evaluation.footprints(spec), evaluation.footprints(changed))

    def test_rehashed_selection_native_flag_condition_and_component_forgery_are_rejected(self):
        for mutation in (lambda panel: panel.update(native_launch_authorized=True),
                         lambda panel: panel["selected_parents"].pop(),
                         lambda panel: panel["selected_parents"][0].update(component_id="forged"),
                         lambda panel: panel["selected_parents"][0]["cells"].reverse()):
            forged = copy.deepcopy(self.panel)
            mutation(forged)
            forged["sha256"] = evaluation.contract.digest({key: value for key, value in forged.items() if key != "sha256"})
            with self.assertRaises(ValueError): evaluation.validate_panel(forged)

    def test_policy_rows_exclude_oracle_lineage_specs_labels_and_conditions(self):
        for row in self.inputs["rows"]:
            self.assertEqual({"row_id", "messages", "prompt_sha256"}, set(row))
            self.assertNotIn(row["row_id"], row["messages"][1]["content"])
        forged = copy.deepcopy(self.inputs)
        forged["rows"][0]["solutions"] = self.oracle["rows"][0]["outcome"]["solutions"]
        with self.assertRaises(ValueError): evaluation.audit_exports(self.panel, forged, self.oracle)
        forged_oracle = copy.deepcopy(self.oracle)
        forged_oracle["rows"][0]["outcome"]["solutions"] = []
        with self.assertRaises(ValueError): evaluation.audit_exports(self.panel, self.inputs, forged_oracle)

    def test_forged_exclusion_bindings_or_repository_claim_without_evidence_are_rejected(self):
        bad = exclusions()
        bad["footprints"].append("balanced:" + "a" * 64)
        with self.assertRaisesRegex(ValueError, "binding"): evaluation.compile_panel(bad)
        bad = exclusions()
        bad["scope"] = "all_package_materialized_CPU_data_and_source_bound_fixture_census"
        bad["sha256"] = evaluation.contract.digest({key: value for key, value in bad.items() if key != "sha256"})
        with self.assertRaisesRegex(ValueError, "evidence missing"): evaluation.compile_panel(bad)

    def test_exclusive_bundle_separates_oracle_and_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(evaluation, "PACKAGE", Path(directory)):
            target = Path(directory) / "artifacts/evaluation_dev/unit_test"
            manifest = evaluation.write_bundle(self.panel, target)
            self.assertTrue((target / "policy/prompts.json").is_file())
            self.assertTrue((target / "oracle_CPU_DEV_exposed/labels.json").is_file())
            self.assertTrue(manifest["oracle_labels_exposed_CPU_DEV"])
            self.assertFalse(manifest["confirmation_eligible"])
            with self.assertRaises(ValueError): evaluation.write_bundle(self.panel, target)
            with self.assertRaises(ValueError): evaluation.write_bundle(self.panel, Path(directory) / "outside")


class CensusBoundaryTests(unittest.TestCase):
    def test_added_removed_materializations_and_changed_source_closure_reject_old_exclusions(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory)
            (package / "data").mkdir()
            paths = [package / "data/a.json", package / "data/b.json"]
            for path in paths: path.write_text(evaluation.contract.canonical(r_spec()))
            source = package / "fixture.py"
            source.write_text("# frozen source")
            record = {"scope": "all_package_materialized_CPU_data_and_source_bound_fixture_census",
                      "file_pins": {str(p.resolve()): evaluation._sha(p) for p in paths},
                      "source_pins": {str(source.resolve()): evaluation._sha(source)},
                      "fixture_tests": 1, "fixture_specs": 1, "footprints": [],
                      "protected_banks_read": False, "model_outputs_used": False}
            record["sha256"] = evaluation.contract.digest(record)
            with patch.object(evaluation, "PACKAGE", package), patch.object(evaluation, "_fixture_sources", return_value=[source]):
                evaluation._validate_exclusions(record)
                added = package / "data/exposed.json"
                added.write_text(evaluation.contract.canonical(s_spec()))
                with self.assertRaisesRegex(ValueError, "inventory closure"):
                    evaluation._validate_exclusions(record)
                added.unlink()
                paths[1].unlink()
                with self.assertRaisesRegex(ValueError, "inventory closure"):
                    evaluation._validate_exclusions(record)
                paths[1].write_text(evaluation.contract.canonical(r_spec()))
                new_source = package / "new_fixture.py"
                new_source.write_text("# newly exposed fixture")
                with patch.object(evaluation, "_fixture_sources", return_value=[source, new_source]):
                    with self.assertRaisesRegex(ValueError, "source closure"):
                        evaluation._validate_exclusions(record)

    def test_materialized_cpu_sources_and_fixture_census_are_joined_without_protected_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            package = Path(directory) / "package"
            (package / "data/old").mkdir(parents=True)
            (package / "tests").mkdir()
            (package / "data/old/plan.json").write_text(evaluation.contract.canonical({"parent": {"spec": r_spec()}}))
            (package / "data/old/candidate_records.jsonl").write_text(evaluation.contract.canonical({"spec": s_spec()}) + "\n")
            for name in ("rs_contract.py", "rs_solver.py", "rs_verifier.py", "task_compiler.py", "balanced_corpus.py", "cpu_tokenizer.py", "matching_schedule.py", "prepare_corpus.py"):
                (package / name).write_text("# CPU fixture source")
            for name in evaluation.FIXTURE_MODULES: (package / "tests" / (name + ".py")).write_text("# CPU fixture")
            static = package.parent / "post_training_mimo_centralized_v1/receipts/titan_static_v1/PRIMARY_STUDENT_RECEIPT.json"
            static.parent.mkdir(parents=True, exist_ok=True)
            static.write_text('{"checkpoint_path":"CPU_tokenizer_fixture_only"}')
            marker = package / "central/CHECK.jsonl"
            marker.parent.mkdir()
            marker.write_text("DO NOT READ")
            key = "balanced:" + "a" * 64
            response = SimpleNamespace(returncode=0, stderr="", stdout=json.dumps({"fixture_tests": 137, "fixture_specs": 20, "footprints": [key]}))
            opened = []
            original_open = Path.open
            def checked_open(path, *args, **kwargs):
                self.assertNotEqual(marker, path)
                opened.append(path)
                return original_open(path, *args, **kwargs)
            with patch.object(evaluation, "PACKAGE", package), patch.object(evaluation.subprocess, "run", return_value=response), patch.object(Path, "open", checked_open):
                result = evaluation.collect_repository_exclusions()
            self.assertIn(key, result["footprints"])
            self.assertTrue(evaluation.footprints(r_spec()) <= set(result["footprints"]))
            self.assertTrue(evaluation.footprints(s_spec()) <= set(result["footprints"]))
            self.assertFalse(result["protected_banks_read"])
            self.assertNotIn(marker, opened)

    def test_import_loads_no_framework_or_model_assets(self):
        code = "import sys;sys.path.insert(0,sys.argv[1]);import evaluation_panel;assert 'torch' not in sys.modules and 'transformers' not in sys.modules;print('CPU_ONLY_IMPORT')"
        result = subprocess.run([sys.executable, "-I", "-B", "-c", code, str(LOCATION)], text=True, capture_output=True, timeout=30, check=True)
        self.assertEqual("CPU_ONLY_IMPORT", result.stdout.strip())


if __name__ == "__main__": unittest.main()
