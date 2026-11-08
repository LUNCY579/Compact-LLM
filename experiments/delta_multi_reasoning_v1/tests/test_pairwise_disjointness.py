"""Synthetic CPU mechanics only: never load model assets or historical evaluation rows."""
import copy
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "audit" / "pairwise_disjointness.py"
spec = importlib.util.spec_from_file_location("dmrf_pairwise_disjointness", MODULE_PATH)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


def rid(n, prefix="", step=1):
    return f"{prefix}RRFR2-{n:024x}-step{step}-valid"


def words(label):
    return " ".join(f"{label}_{i}" for i in range(25))


def rows(offset=0, prefix=""):
    return [{"id": rid(offset + i, prefix), "prompt": words(str(offset + i)),
             "messages": [{"role": "user", "content": words(str(offset + i))},
                          {"role": "assistant", "content": "SAME ASSISTANT TEXT"}]}
            for i in range(1, 4)]


def side(source, field="prompt", prefix=""):
    return {"source": source, "field": field, "prefix": prefix}


def write_source(root, name, data):
    content = "".join(json.dumps(row) + "\n" for row in data).encode()
    (root / (name + ".jsonl")).write_bytes(content)
    return {"path": name + ".jsonl", "sha256": p.sha(content), "rows": len(data), "row_id_field": "id"}


def plan(root, method="text_v1", left=None, right=None, ls=None, rs=None):
    left = rows() if left is None else left
    right = rows(10) if right is None else right
    return {"schema": p.SCHEMA, "source_commit": p.BASE_COMMIT,
            "sources": {"L": write_source(root, "L", left), "R": write_source(root, "R", right)},
            "checks": [{"id": "L_R_method", "pair": "L_R", "method": method,
                        "left": ls or side("L"), "right": rs or side("R")}]}


def only(root, config):
    return p.audit_manifest(config, root)["checks"][0]


@pytest.mark.parametrize("method", sorted(p.METHODS))
def test_all_methods_have_fresh_bidirectional_canaries(tmp_path, method):
    left, right = rows(prefix="RRP-select:"), rows(10)
    ls, rs = side("L"), side("R")
    if method.startswith("rrfr2_"):
        ls, rs = side("L", "id", "RRP-select:"), side("R", "id")
    elif method == "srb_row_v1":
        for data, offset, prefix in ((left, 0, "rehearsal:"), (right, 10, "")):
            for i, row in enumerate(data):
                row["id"] = f"{prefix}rineya_srb_v1_clarification_{offset + i:05d}"
        ls, rs = side("L", "id", "rehearsal:"), side("R", "id")
    result = only(tmp_path, plan(tmp_path, method, left, right, ls, rs))
    assert result["status"] == "SUPPORTED_ZERO"
    assert result["zero_admissible"]
    assert len(result["controls"]["cross_injections"]) == 2
    assert all(c["passed"] for c in result["controls"]["cross_injections"])


def test_real_delta_h_namespace_regression_is_unobservable(tmp_path):
    left, right = rows(prefix="RRP-select:"), rows(10)
    for i, row in enumerate(right):
        row["id"] = f"rehearsal:rineya_srb_v1_safety_boundary_{i:05d}"
    config = plan(tmp_path, "rrfr2_stem_v1", left, right,
                  side("L", "id", "RRP-select:"), side("R", "id", "rehearsal:"))
    result = only(tmp_path, config)
    assert result["status"] == "UNOBSERVABLE"
    assert result["detail"]["right"]["covered_rows"] == 0
    assert result["overlap"] is None and not result["zero_admissible"]
    config["checks"][0].update(method="text_v1", left=side("L", "messages.user"), right=side("R", "prompt"))
    assert only(tmp_path, config)["status"] == "SUPPORTED_ZERO"


def test_asymmetric_extractor_passes_self_but_fails_cross(tmp_path, monkeypatch):
    original = p._extract
    def wrong(row, spec, method):
        keys, namespace = original(row, spec, method)
        return frozenset((spec["source"], key) for key in keys), namespace
    monkeypatch.setattr(p, "_extract", wrong)
    result = only(tmp_path, plan(tmp_path))
    assert all(c["passed"] for c in result["controls"]["self"])
    assert not result["controls"]["passed"]
    assert result["status"] == "UNOBSERVABLE" and result["overlap"] is None


def test_original_unstripped_wrapper_bug_is_caught(tmp_path, monkeypatch):
    original = p._extract
    def wrong(row, spec, method):
        keys, namespace = original(row, spec, method)
        return frozenset(spec["prefix"] + key for key in keys), namespace
    monkeypatch.setattr(p, "_extract", wrong)
    config = plan(tmp_path, "rrfr2_stem_v1", rows(prefix="RRP-select:"), rows(10),
                  side("L", "id", "RRP-select:"), side("R", "id"))
    result = only(tmp_path, config)
    assert all(c["passed"] for c in result["controls"]["self"])
    assert result["status"] == "UNOBSERVABLE" and result["overlap"] is None


def test_stuck_zero_comparator_is_not_a_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(p, "_compare", lambda a, b: {"keys": set(), "left_rows": set(), "right_rows": set()})
    result = only(tmp_path, plan(tmp_path))
    assert result["status"] == "UNOBSERVABLE" and not result["zero_admissible"]


@pytest.mark.parametrize("method", ["text_v1", "ngram20_v1", "rrfr2_row_v1", "rrfr2_stem_v1"])
def test_actual_overlap_is_contradicted_not_edited(tmp_path, method):
    left, right = rows(), rows(10)
    if method in p.TEXT_METHODS:
        right[1]["prompt"] = left[2]["prompt"]
        ls, rs = side("L"), side("R")
    else:
        right[1]["id"] = left[2]["id"]
        ls, rs = side("L", "id"), side("R", "id")
    config = plan(tmp_path, method, left, right, ls, rs)
    before = {name: (tmp_path / f"{name}.jsonl").read_bytes() for name in ("L", "R")}
    result = only(tmp_path, config)
    assert result["status"] == "CONTRADICTED" and result["overlap"]["left_rows"] == 1
    assert not result["zero_admissible"]
    assert before == {name: (tmp_path / f"{name}.jsonl").read_bytes() for name in ("L", "R")}


def test_normalization_detects_unicode_case_whitespace_without_answer_leakage(tmp_path):
    left, right = rows(), rows(10)
    left[0]["prompt"] = "Ｆｏｏ\tBAR  23"
    right[0]["prompt"] = "foo bar 23"
    result = only(tmp_path, plan(tmp_path, left=left, right=right))
    assert result["overlap"]["unique_keys"] == 1
    assert result["controls"]["passed"]
    # Different inputs with identical assistant responses are still input-text-disjoint.
    config = plan(tmp_path, ls=side("L", "messages.user"), rs=side("R", "messages.user"))
    assert only(tmp_path, config)["status"] == "SUPPORTED_ZERO"


def test_digits_are_not_erased(tmp_path):
    left, right = rows(), rows(10)
    left[0]["prompt"], right[0]["prompt"] = "compute 12", "compute 13"
    assert only(tmp_path, plan(tmp_path, left=left, right=right))["status"] == "SUPPORTED_ZERO"


def test_ngram_overlap_without_exact_text_equality(tmp_path):
    left, right = rows(), rows(10)
    right[0]["prompt"] = "extra " + left[0]["prompt"] + " suffix"
    assert only(tmp_path, plan(tmp_path, left=left, right=right))["status"] == "SUPPORTED_ZERO"
    assert only(tmp_path, plan(tmp_path, "ngram20_v1", left, right))["status"] == "CONTRADICTED"


@pytest.mark.parametrize("short_count", [1, 3])
def test_ngram_ineligible_rows_are_not_silently_dropped(tmp_path, short_count):
    right = rows(10)
    for row in right[:short_count]:
        row["prompt"] = "too short"
    result = only(tmp_path, plan(tmp_path, "ngram20_v1", right=right))
    assert result["status"] == "UNOBSERVABLE" and result["overlap"] is None
    assert result["detail"]["right"]["uncovered_rows"] == short_count


def test_duplicate_canonical_keys_do_not_fake_self_cardinality(tmp_path):
    left = rows()
    left[1]["prompt"] = left[0]["prompt"]
    result = only(tmp_path, plan(tmp_path, left=left))
    self_control = result["controls"]["self"][0]
    assert self_control["rows"] == 3 and self_control["expected_unique_keys"] == 2
    assert result["status"] == "SUPPORTED_ZERO"


def test_parent_stem_fields_use_cross_schema_controls(tmp_path):
    left, right = rows(prefix="RRP-select:"), rows(10)
    for i, row in enumerate(left):
        row["parent_id"] = f"RRFR2-{i + 100:024x}"
    config = plan(tmp_path, "rrfr2_stem_v1", left, right, side("L", "parent_id"), side("R", "id"))
    assert only(tmp_path, config)["status"] == "SUPPORTED_ZERO"


@pytest.mark.parametrize("defect", ["missing", "empty", "partial_grammar", "duplicate_id", "wrong_hash", "wrong_count",
                                   "malformed_json", "duplicate_json_key", "blank_line", "missing_path", "unbound"])
def test_invalid_or_unobservable_sources_cannot_produce_zero(tmp_path, defect):
    config = plan(tmp_path)
    path = tmp_path / "R.jsonl"
    right = rows(10)
    if defect == "missing":
        del right[0]["prompt"]
    elif defect == "empty":
        right[0]["prompt"] = " "
    elif defect == "partial_grammar":
        right[0]["id"] = "different-corpus:1"
        config["checks"][0].update(method="rrfr2_row_v1", left=side("L", "id"), right=side("R", "id"))
    elif defect == "duplicate_id":
        right[1]["id"] = right[0]["id"]
    if defect in {"missing", "empty", "partial_grammar", "duplicate_id"}:
        config["sources"]["R"] = write_source(tmp_path, "R", right)
    elif defect == "wrong_hash":
        config["sources"]["R"]["sha256"] = "0" * 64
    elif defect == "wrong_count":
        config["sources"]["R"]["rows"] = 4
    elif defect in {"malformed_json", "duplicate_json_key", "blank_line"}:
        payload = {"malformed_json": "{oops}\n", "duplicate_json_key": '{"id":"a","id":"b"}\n', "blank_line": "\n"}[defect]
        path.write_text(payload)
        config["sources"]["R"].update(sha256=p.sha(path.read_bytes()), rows=1)
    elif defect == "missing_path":
        path.unlink()
    else:
        config["sources"]["R"]["sha256"] = None
    result = only(tmp_path, config)
    assert result["status"] in {"UNOBSERVABLE", "INVALID", "UNVERIFIED"}
    assert not result["zero_admissible"] and result["overlap"] is None


def test_actual_namespace_sets_must_match(tmp_path):
    left, right = rows(), rows(10)
    for row in left:
        row["id"] = "rineya_srb_v1_safety_boundary_" + str(left.index(row))
    for row in right:
        row["id"] = "rineya_srb_v1_clarification_" + str(right.index(row))
    result = only(tmp_path, plan(tmp_path, "srb_row_v1", left, right, side("L", "id"), side("R", "id")))
    assert result["status"] == "UNOBSERVABLE"


def test_controls_run_per_pair_and_method_not_once_per_batch(tmp_path, monkeypatch):
    config = plan(tmp_path)
    config["sources"]["H"] = write_source(tmp_path, "H", rows(20))
    config["checks"].append({"id": "L_H", "pair": "L_H", "method": "text_v1", "left": side("L"), "right": side("H")})
    config["checks"].append({"id": "L_R_ngram", "pair": "L_R", "method": "ngram20_v1", "left": side("L"), "right": side("R")})
    original, calls = p._controls, []
    def control(*args):
        calls.append((args[2]["source"], args[3]["source"], args[4]))
        return original(*args)
    monkeypatch.setattr(p, "_controls", control)
    report = p.audit_manifest(config, tmp_path)
    assert len(calls) == 3 and len(set(calls)) == 3
    assert len({r["binding_sha256"] for r in report["checks"]}) == 3


def test_one_passing_pair_cannot_cover_a_broken_second_pair(tmp_path, monkeypatch):
    config = plan(tmp_path)
    config["sources"]["H"] = write_source(tmp_path, "H", rows(20))
    config["checks"].append({"id": "L_H", "pair": "L_H", "method": "text_v1", "left": side("L"), "right": side("H")})
    original = p._extract
    def broken_h(row, spec, method):
        keys, namespace = original(row, spec, method)
        return (frozenset("H:" + k for k in keys) if spec["source"] == "H" else keys), namespace
    monkeypatch.setattr(p, "_extract", broken_h)
    result = p.audit_manifest(config, tmp_path)
    assert result["checks"][0]["status"] == "SUPPORTED_ZERO"
    assert result["checks"][1]["status"] == "UNOBSERVABLE"
    with pytest.raises(p.Refusal):
        p.require_supported(result)


def test_multiple_namespaces_receive_individual_canaries(tmp_path):
    left, right = [], []
    for prefix in ("safety_boundary", "clarification", "knowledge_boundary", "uncertainty"):
        left.append({"id": f"rineya_srb_v1_{prefix}_00001"})
        right.append({"id": f"rineya_srb_v1_{prefix}_00002"})
    result = only(tmp_path, plan(tmp_path, "srb_row_v1", left, right, side("L", "id"), side("R", "id")))
    assert result["status"] == "SUPPORTED_ZERO"
    assert len(result["controls"]["cross_injections"]) == 8


@pytest.mark.parametrize("defect", ["empty_checks", "duplicate_check", "unknown_method", "unregistered_source",
                                   "unknown_option", "wrong_base", "self_claim", "bool_count", "empty_sources"])
def test_invalid_registry_cannot_be_vacuously_admitted(tmp_path, defect):
    config = plan(tmp_path)
    if defect == "empty_checks": config["checks"] = []
    elif defect == "duplicate_check": config["checks"].append(copy.deepcopy(config["checks"][0]))
    elif defect == "unknown_method": config["checks"][0]["method"] = "magic"
    elif defect == "unregistered_source": config["checks"][0]["right"]["source"] = "absent"
    elif defect == "unknown_option": config["checks"][0]["skip_positive_control"] = True
    elif defect == "wrong_base": config["source_commit"] = "0" * 40
    elif defect == "self_claim": config["checks"][0]["right"]["source"] = "L"
    elif defect == "bool_count": config["sources"]["R"]["rows"] = True
    elif defect == "empty_sources": config["sources"] = {}
    with pytest.raises(p.Refusal): p.audit_manifest(config, tmp_path)


def test_multi_user_messages_are_not_silently_concatenated(tmp_path):
    right = rows(10)
    right[0]["messages"].append({"role": "user", "content": "other question"})
    result = only(tmp_path, plan(tmp_path, right=right, rs=side("R", "messages.user")))
    assert result["status"] == "INVALID"


def test_deterministic_repeat_and_source_binding_change(tmp_path):
    config = plan(tmp_path)
    first = p.audit_manifest(config, tmp_path)
    assert first == p.audit_manifest(config, tmp_path)
    config["sources"]["R"] = write_source(tmp_path, "R", rows(20))
    assert first["checks"][0]["binding_sha256"] != only(tmp_path, config)["binding_sha256"]


@pytest.mark.parametrize("collision", [False, True])
def test_cli_enforces_verdict_and_never_overwrites(tmp_path, collision):
    left, right = rows(), rows(10)
    if collision: right[0]["prompt"] = left[0]["prompt"]
    config = plan(tmp_path, left=left, right=right)
    manifest, output = tmp_path / "plan.json", tmp_path / "receipt.json"
    manifest.write_text(json.dumps(config))
    before = {f.name: f.read_bytes() for f in tmp_path.iterdir()}
    assert p.main(["--manifest", str(manifest), "--out", str(output)]) == (1 if collision else 0)
    written = output.read_bytes()
    assert p.main(["--manifest", str(manifest), "--out", str(output)]) == 2
    assert output.read_bytes() == written
    for name, data in before.items(): assert (tmp_path / name).read_bytes() == data
    assert p.main(["--manifest", str(manifest), "--out", str(tmp_path / "L.jsonl")]) == 2
    assert (tmp_path / "L.jsonl").read_bytes() == before["L.jsonl"]


def test_cli_refusal_and_optimized_python_keep_guards(tmp_path):
    config = plan(tmp_path)
    config["sources"]["R"]["sha256"] = None
    manifest, output = tmp_path / "plan.json", tmp_path / "receipt.json"
    manifest.write_text(json.dumps(config))
    result = subprocess.run([sys.executable, "-O", str(MODULE_PATH), "--manifest", str(manifest), "--out", str(output)],
                            capture_output=True, text=True, check=False, timeout=10)
    assert result.returncode == 2
    report = json.loads(output.read_text())
    assert report["status"] == "REFUSED" and report["checks"][0]["overlap"] is None


def test_module_is_cpu_stdlib_only():
    import ast
    tree = ast.parse(MODULE_PATH.read_text())
    imports = {node.names[0].name.split(".")[0] if isinstance(node, ast.Import) else node.module.split(".")[0]
               for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom))}
    assert imports <= {"__future__", "argparse", "copy", "hashlib", "json", "re", "sys", "unicodedata", "pathlib", "typing"}


def test_nonempty_but_unrelated_baseline_hit_cannot_satisfy_injection(tmp_path, monkeypatch):
    left, right = rows(), rows(10)
    right[2]["prompt"] = left[2]["prompt"]
    monkeypatch.setattr(p, "_injected", lambda receiver, *args: copy.deepcopy(receiver))
    result = only(tmp_path, plan(tmp_path, left=left, right=right))
    assert result["status"] == "UNOBSERVABLE"
    assert result["overlap"] is None
    assert not result["controls"]["passed"]


def test_control_payloads_do_not_mutate_any_original_row(tmp_path):
    config = plan(tmp_path)
    source = {name: p.load_source(spec, tmp_path) for name, spec in config["sources"].items()}
    before = copy.deepcopy(source)
    result = p.audit_check(config["checks"][0], source, "0" * 64)
    assert result["status"] == "SUPPORTED_ZERO" and source == before


def test_report_has_no_prompt_content_or_generic_clean_claim(tmp_path):
    result = p.audit_manifest(plan(tmp_path), tmp_path)
    text = json.dumps(result)
    assert words("1") not in text and "SAME ASSISTANT TEXT" not in text
    assert result["authority"] == {"model_calls": 0, "training_steps": 0, "rescoring": False,
                                    "historical_outcome_updates": False, "new_execution_authorization": False}
    assert "not semantic leakage" in result["scope"]


@pytest.mark.parametrize("mutation", ["unknown_prefix", "changed_selector", "source_missing", "nonfinite", "invalid_utf8", "unhashable_method"])
def test_additional_fail_closed_paths(tmp_path, mutation):
    config = plan(tmp_path)
    if mutation == "unknown_prefix":
        config["checks"][0].update(method="rrfr2_row_v1", left=side("L", "id", "wrong:"), right=side("R", "id"))
    elif mutation == "changed_selector":
        config["checks"][0]["right"]["field"] = "not_present"
    elif mutation == "source_missing":
        config["sources"]["R"]["path"] = "unavailable_remote_mount.jsonl"
    elif mutation in {"nonfinite", "invalid_utf8"}:
        content = b'{"id":"a","x":NaN}\n' if mutation == "nonfinite" else b'\xff\n'
        (tmp_path / "R.jsonl").write_bytes(content)
        config["sources"]["R"].update(sha256=p.sha(content), rows=1)
    else:
        config["checks"][0]["method"] = []
        with pytest.raises(p.Refusal): p.audit_manifest(config, tmp_path)
        return
    result = only(tmp_path, config)
    assert not result["zero_admissible"] and result["overlap"] is None


def test_empty_source_is_invalid_even_when_digest_matches(tmp_path):
    config = plan(tmp_path)
    (tmp_path / "R.jsonl").write_bytes(b"")
    config["sources"]["R"]["sha256"] = p.sha(b"")
    assert only(tmp_path, config)["status"] == "INVALID"


def test_shipped_host_template_cannot_claim_historical_measurements():
    template = MODULE_PATH.with_name("pairwise_disjointness_manifest.template.json")
    config = json.loads(template.read_text())
    result = p.audit_manifest(config, template.parent)
    assert len(result["checks"]) == 8 and result["status"] == "REFUSED"
    assert all(c["status"] == "UNVERIFIED" and c["overlap"] is None for c in result["checks"])
    assert {c["pair"] for c in result["checks"]} == {"views__delta_T_s23", "views__delta_H", "hskill_confirm002__delta_H"}
