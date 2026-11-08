"""Offline custody verification and Stage C transport analysis; no model imports.

The caller supplies a separately frozen custody registry and its byte hash.
This module never manufactures trusted expected hashes from the files it is asked
to analyze. Custody authenticates captured evidence; it cannot prove an untrusted
producer performed native execution. Synthetic fixtures are confined to tests.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import numpy as np

from . import mechanics as m
from . import receipts as r
from . import workflow as w

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = "experiments/cross_scale_composable_001/"
ROLES = ("student_base", "donor_base", "donor_skilled")
PAIRS = ((0, 0), (13, 17), (27, 35))
LAYERS = {"student_base": (0, 13, 27), "donor_base": (0, 17, 35), "donor_skilled": (0, 17, 35)}
WIDTHS = {"student_base": 1024, "donor_base": 2560, "donor_skilled": 2560}
ANALYSIS_PROTOCOL = {"version": "C-mechanism-002", "require_beats_shuffle": True, "rcond": 1e-12, "seed": 1729}


def read_json(path):
    def pairs(items):
        result = {}
        for key, value in items:
            r.require(key not in result, "duplicate JSON key")
            result[key] = value
        return result
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError("nonfinite JSON: " + value)))


def git_bytes(*args):
    done = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, check=False)
    r.require(done.returncode == 0, "historical source revision/path unavailable")
    return done.stdout


def historical_file(commit, name):
    r.require(isinstance(commit, str) and re.fullmatch(r"[0-9a-f]{40}", commit), "invalid historical commit")
    r.require(isinstance(name, str) and not Path(name).is_absolute() and ".." not in Path(name).parts,
              "unsafe historical source path")
    resolved = git_bytes("rev-parse", "--verify", commit + "^{commit}").decode().strip()
    r.require(resolved == commit, "source does not resolve to exact commit")
    return git_bytes("show", commit + ":" + name)


def verify_historical_source(spec):
    r.require(spec.get("schema") == 1 and spec.get("experiment") == "CSC-001" and spec.get("stage") == "C",
              "not a Stage C native registration")
    r.require(spec.get("native_capture_authorized") is True and spec.get("max_total_forwards") == 96
              and spec.get("max_role_forwards") == 32 and spec.get("max_optimizer_steps") == 0
              and spec.get("max_generated_tokens") == 0, "wrong registered execution envelope")
    files = spec.get("source_files")
    r.require(isinstance(files, dict) and bool(files) and r.digest(files) == spec.get("implementation_digest"),
              "invalid historical implementation inventory")
    for required in ("native_capture.py", "native_session.py", "receipts.py"):
        r.require(PACKAGE + required in files, "native capture implementation not bound")
    amendment = spec.get("stage_c_mechanism_amendment", {})
    r.require(amendment.get("path") == PACKAGE + "STAGE_C_MECHANISM_AMENDMENT.md"
              and files.get(amendment["path"]) == amendment.get("sha256"), "mechanism amendment not source-bound")
    r.require(spec.get("analysis_protocol") == ANALYSIS_PROTOCOL, "analysis decision protocol differs")
    for name, expected in files.items():
        r.valid_hash(expected, "historical source byte hash")
        r.require(hashlib.sha256(historical_file(spec["source_commit"], name)).hexdigest() == expected,
                  "capture implementation differs from its committed source: " + name)
    calibration = spec["calibration"]
    original_bytes = historical_file(spec["source_commit"], PACKAGE + "evidence/CALIBRATION_BINDING.v2.json")
    architecture_bytes = historical_file(spec["source_commit"], PACKAGE + "evidence/ARCHITECTURE_MAP.json")
    r.require(hashlib.sha256(original_bytes).hexdigest() == calibration["source_binding_file_sha256"],
              "original calibration bytes differ")
    r.require(hashlib.sha256(architecture_bytes).hexdigest() == calibration["architecture_file_sha256"],
              "original architecture bytes differ")
    original, architecture = json.loads(original_bytes), json.loads(architecture_bytes)
    r.require(original["rows"] == calibration["source_binding_rows"], "original calibration membership differs")
    for key in ("calibration_sha256", "dataset_sha256"):
        r.require(original[key] == calibration[key], "original " + key + " differs")
    r.require(calibration["layer_map_sha256"] == r.digest(architecture["layer_map"]), "registered layer map differs")
    r.require(architecture["layer_map"]["capture_pairs_student_donor"] == [list(pair) for pair in PAIRS],
              "registered capture pairs differ")
    r.require(architecture["layer_map"]["donor_to_student"] == list(m.normalized_depth_map(36, 28)),
              "invalid registered normalized depth map")
    rows = calibration["rows"]
    r.require(len(rows) == len(original["rows"]) == 32 and r.digest(rows) == calibration["rows_sha256"],
              "token calibration rows differ")
    r.validate_calibration(original)
    r.require(sum(row["split"] == "fit" for row in rows) == 24, "wrong FIT/CHECK allocation")
    for row, registered in zip(rows, original["rows"]):
        for key in ("id", "parent_id", "split", "open"):
            r.require(row.get(key) == registered.get(key), "calibration identity/partition differs")
        r.require(row.get("messages_sha256") == registered["prompt_sha256"], "original prompt identity differs")
        ids = row["input_ids"]
        r.require(isinstance(ids, list) and 0 < len(ids) <= 256 and all(type(i) is int and i >= 0 for i in ids),
                  "invalid registered prompt token count")
        r.require(r.digest(ids) == row["input_ids_sha256"] and r.digest(row["prompt"]) == row["prompt_sha256"],
                  "registered token/rendered prompt digest differs")
    coordinates = [{"id": row["id"], "input_ids_sha256": row["input_ids_sha256"],
                    "position": len(row["input_ids"]) - 1, "token_id": row["input_ids"][-1]} for row in rows]
    r.require(r.digest(coordinates) == calibration["prompt_positions_sha256"], "registered token positions differ")
    r.require(r.digest(spec["tokenizer"]["files"]) == calibration["tokenizer_sha256"], "registered tokenizer differs")
    for key in ("student", "donor", "adapter"):
        r.require(r.digest(spec["assets"][key]["files"]) == spec["assets"][key]["model_sha256"],
                  "registered asset identity differs")
    return original


def analysis_identity():
    commit = git_bytes("rev-parse", "HEAD").decode().strip()
    r.require(not git_bytes("status", "--porcelain=v1", "--untracked-files=all").strip(), "analysis source is dirty")
    files = {}
    for name in ("analyze_captures.py", "mechanics.py", "receipts.py", "workflow.py"):
        relative = PACKAGE + name
        actual = r.file_digest(ROOT / relative)
        r.require(hashlib.sha256(historical_file(commit, relative)).hexdigest() == actual,
                  "analysis implementation not committed")
        files[relative] = actual
    return {"source_commit": commit, "files": files, "implementation_digest": r.digest(files)}


def expected_metadata(spec, role, layer, tensor_sha256):
    cal = spec["calibration"]
    anchor = spec["assets"]["student" if role == "student_base" else "donor"]["model_sha256"]
    identity = r.digest({"base": anchor, "adapter": spec["assets"]["adapter"]["model_sha256"]}) if role == "donor_skilled" else anchor
    result = {key: cal[key] for key in ("dataset_sha256", "calibration_sha256", "tokenizer_sha256",
              "layer_map_sha256", "prompt_positions_sha256")}
    result.update(role=role, model_sha256=identity, anchor_checkpoint_sha256=anchor,
                  implementation_digest=spec["implementation_digest"],
                  runtime_binding_sha256=r.digest(spec["runtime"]["student_base" if role == "student_base" else "donor"]),
                  tensor_sha256=tensor_sha256, layer_ids=[layer], pooling="last_nonpad_prompt_token", eval_mode=True,
                  optimization_steps=0, dtype="float32", padding_excluded=True,
                  row_ids=[row["id"] for row in cal["rows"]], shape=[32, WIDTHS[role]], capture_site="block_output_raw")
    return result


def required_receipt_paths(spec, role):
    attempt = Path(spec["attempt_root"]).resolve() / role
    supervision = attempt / "supervision"
    journal = Path(spec["reservation_dir"]).resolve() / role
    files = [attempt / "WORKER_SPEC.json", attempt / "SESSION_TERMINAL.json",
             *(supervision / name for name in ("launch.json", "spawn.json", "worker_claim.json", "supervisor.json")),
             journal.parent / "SESSION_BINDING.json", journal / "CLAIM.json", journal / "TERMINAL.json"]
    files += [journal / f"{i:03d}-{event}.json" for i in range(1, 33) for event in ("RESERVED", "COMPLETED")]
    return files


def verify_role(spec, spec_sha256, role, custody):
    attempt = Path(spec["attempt_root"]).resolve() / role
    supervision, directory = attempt / "supervision", attempt / "supervision" / "capture"
    result_path = directory / "RESULT.json"
    r.require(Path(custody["result_path"]).resolve() == result_path, "result is outside registered role attempt")
    r.verify_file(result_path, custody["result_sha256"])
    required = {str(path): path for path in required_receipt_paths(spec, role)}
    r.require(set(custody["receipt_files"]) == set(required), "incomplete or extra custody receipt inventory")
    for name, path in required.items():
        r.verify_file(path, custody["receipt_files"][name])
    for failure in (attempt / "FAILURE.json", directory / "FAILURE.json", supervision / "supervisor_failure.json"):
        r.require(not failure.exists(), "failed child cannot supply qualified captures")
    result = read_json(result_path)
    r.require(result.get("evidence_kind") not in ("SYNTHETIC_CPU_FIXTURE_ONLY", "SYNTHETIC_TEST_ONLY")
              and result.get("fixture") is not True, "fixture result is not native evidence")
    r.require(result.get("schema") == 1 and result.get("status") == "COMPLETE" and result.get("completed") is True,
              "incomplete native capture result")
    r.require(result.get("role") == role and result.get("session_id") == spec["session_id"]
              and result.get("source_commit") == spec["source_commit"], "capture source/session/role differs")
    r.require(result.get("native_model_forwards") == 32 and result.get("native_tokenizer_encodings") == 32
              and result.get("generation_calls") == 0 and result.get("optimizer_steps") == 0,
              "native capture call counts differ")
    worker_path = attempt / "WORKER_SPEC.json"
    worker = read_json(worker_path)
    r.require(worker.get("frozen_registration_sha256") == spec_sha256 and r.file_digest(worker_path) == result["spec_sha256"],
              "worker does not bind frozen registration")
    body = {key: value for key, value in worker.items() if key not in ("lease", "frozen_registration_sha256")}
    r.require(body == spec, "role changed frozen registration fields")
    lease = worker["lease"]
    r.require(lease["path"] == spec["lease_path"] and isinstance(lease.get("token"), str) and bool(lease["token"]),
              "worker lease binding differs")
    terminal, supervisor = read_json(attempt / "SESSION_TERMINAL.json"), read_json(supervision / "supervisor.json")
    r.require(terminal.get("status") == "COMPLETE" and terminal.get("completed") is True
              and terminal.get("role") == role and terminal.get("child_returncode") == 0
              and terminal.get("native_forward_count") == 32 and terminal.get("optimizer_steps") == 0,
              "unsuccessful controller terminal")
    r.require(supervisor.get("status") == "EXITED" and supervisor.get("returncode") == 0
              and supervisor.get("cleanup") is None, "supervisor reports failed/interrupted child")
    launch, spawn, claim = (read_json(supervision / name) for name in ("launch.json", "spawn.json", "worker_claim.json"))
    runtime = spec["runtime"]["student_base" if role == "student_base" else "donor"]
    expected_argv = [runtime["python_executable"], "-B", "-m", PACKAGE.rstrip("/").replace("/", ".") + ".native_capture",
                     "--spec", str(worker_path), "--spec-sha256", result["spec_sha256"], "--role", role,
                     "--out", str(directory), "--supervision", str(supervision)]
    r.require(launch.get("argv") == expected_argv and Path(launch.get("cwd", "")).resolve() == ROOT.resolve()
              and launch.get("out") == str(supervision) and launch.get("supervisor_pid") == lease["owner_pid"],
              "supervisor launch differs from registered capture")
    r.require(type(launch.get("timeout_seconds")) is int and 0 < launch["timeout_seconds"] <= 600,
              "invalid supervised duration")
    r.valid_hash(launch.get("context_sha256"), "supervisor context")
    for receipt in (spawn, claim):
        r.require(receipt.get("launch_sha256") == r.digest(launch)
                  and receipt.get("supervisor_pid") == lease["owner_pid"], "supervisor custody chain differs")
    r.require(supervisor.get("child_pid") == spawn.get("child_pid"), "supervisor child PID differs")
    direct = claim.get("pid") == spawn.get("child_pid") and claim.get("parent_pid") == lease["owner_pid"]
    redirected = claim.get("launcher_redirected") is True and claim.get("parent_pid") == spawn.get("child_pid")
    r.require(direct or redirected, "worker not recorded supervisor child")
    journal = Path(spec["reservation_dir"]).resolve()
    r.require(not (journal / "ACTIVE.json").exists(), "capture session still has active worker")
    binding = {"session_id": spec["session_id"], "frozen_registration_sha256": spec_sha256}
    r.require(read_json(journal / "SESSION_BINDING.json") == binding, "reservation session differs")
    active = {**binding, "worker_spec_sha256": result["spec_sha256"], "role": role, "pid": claim["pid"]}
    role_journal = journal / role
    r.require(read_json(role_journal / "CLAIM.json") == active, "role reservation claim differs")
    ended = read_json(role_journal / "TERMINAL.json")
    r.require(ended == {**active, "status": "COMPLETE", "reserved_forwards": 32, "error": None, "retry_authorized": False},
              "incomplete/failed reservation terminal")
    r.require(len(list(role_journal.glob("*-RESERVED.json"))) == len(list(role_journal.glob("*-COMPLETED.json"))) == 32,
              "durable reservation/completion count differs")
    rows = spec["calibration"]["rows"]
    r.require(len(result.get("rows", [])) == 32 and set(result["captures"]) == {str(layer) for layer in LAYERS[role]},
              "capture rows/layers differ")
    for ordinal, (registered, row) in enumerate(zip(rows, result["rows"]), 1):
        for key in ("id", "parent_id", "split", "prompt_sha256", "input_ids_sha256"):
            r.require(row.get(key) == registered[key], "capture row identity differs")
        r.require(row.get("position") == len(registered["input_ids"]) - 1
                  and row.get("token_id") == registered["input_ids"][-1] and row.get("forward_ordinal") == ordinal,
                  "capture token position/forward ordinal differs")
        reserved = read_json(role_journal / f"{ordinal:03d}-RESERVED.json")
        wanted = {**active, "ordinal": ordinal, "row_id": registered["id"],
                  "input_ids_sha256": registered["input_ids_sha256"], "forwards_charged": 1}
        r.require({key: reserved.get(key) for key in wanted} == wanted, "forward reservation identity differs")
        r.require(read_json(role_journal / f"{ordinal:03d}-COMPLETED.json") == row
                  and read_json(directory / f"row-{ordinal - 1:03d}.json") == row, "row completion custody differs")
        names = {f"row-{ordinal - 1:03d}-layer-{layer:02d}.npy" for layer in LAYERS[role]}
        r.require(set(row["artifacts"]) == names, "per-prompt artifact set differs")
        for name, expected in row["artifacts"].items():
            r.verify_file(directory / name, expected)
    captures, expected = {}, {}
    for layer in LAYERS[role]:
        captured = result["captures"][str(layer)]
        r.require(captured.get("file") == f"layer-{layer:02d}.npy", "unsafe/unexpected aggregate file")
        path = directory / captured["file"]
        r.verify_file(path, captured["file_sha256"])
        r.require(path.stat().st_size <= 32 * WIDTHS[role] * 4 + 65536, "capture array exceeds registered shape allocation bound")
        matrix = np.load(path, allow_pickle=False)
        r.require(matrix.dtype == np.dtype("float32") and matrix.shape == (32, WIDTHS[role])
                  and bool(np.isfinite(matrix).all()), "capture matrix dtype/shape/finiteness differs")
        for ordinal in range(32):
            row_path = directory / f"row-{ordinal:03d}-layer-{layer:02d}.npy"
            r.require(row_path.stat().st_size <= WIDTHS[role] * 4 + 65536, "row array exceeds registered shape allocation bound")
            row_array = np.load(row_path, allow_pickle=False)
            r.require(row_array.dtype == np.dtype("float32") and row_array.shape == (WIDTHS[role],)
                      and np.array_equal(row_array, matrix[ordinal]), "per-row/aggregate capture mismatch")
        meta = expected_metadata(spec, role, layer, m.tensor_digest({"residual": matrix}))
        r.require(captured["metadata"] == meta, "capture metadata differs from frozen registration/verified tensors")
        expected[layer] = meta
        captures[layer] = {"metadata": captured["metadata"], "tensors": {"residual": matrix}}
    return captures, expected


def save_array(directory, name, value):
    path = directory / name
    with path.open("xb") as stream:
        np.save(stream, value, allow_pickle=False)
        stream.flush()
        os.fsync(stream.fileno())
    return {"file": name, "sha256": r.file_digest(path), "shape": list(value.shape), "dtype": str(value.dtype)}


def residual_support(x_fit, actual, random, actual_target, random_target, fit_mask, rows):
    """Numerical support in identified donor directions; never a skill criterion."""
    centered = x_fit.astype("float64") - x_fit.astype("float64").mean(0)
    basis = m.orthonormal_basis(centered.T)

    def metrics(delta, target, mask):
        inputs, outputs = delta[mask], target[mask]
        norm = m.delta_report({"v": inputs})["l2"]
        output_norm = m.delta_report({"v": outputs})["l2"]
        supported = inputs @ basis
        supported_norm = m.delta_report({"v": supported})["l2"] if basis.shape[1] else 0.
        energy = min(1., (supported_norm / norm) ** 2) if norm else None
        return {"input_l2": norm, "output_l2": output_norm, "output_input_l2_gain": output_norm / norm if norm else None,
                "identified_support_energy_fraction": energy,
                "zero_or_numerically_tiny_support": energy is None or energy <= m.RCOND ** 2,
                "numeric_tiny_energy_threshold": m.RCOND ** 2}

    groups = {"pooled": np.ones(len(rows), dtype=bool), "fit": fit_mask, "heldout": ~fit_mask}
    result = {"fit_design_rank": basis.shape[1], "rcond": m.RCOND,
              "support_definition": "Squared energy in the rowspace of centered FIT donor base activations; residual changes themselves are uncentered.",
              "partitions": {name: {"actual": metrics(actual, actual_target, mask),
                                     "random": metrics(random, random_target, mask)} for name, mask in groups.items()},
              "family_breakdown": {"available": False, "reason": "Family labels are absent from frozen token rows; none inferred from identifiers."},
              "skill_transfer_established": False}
    if all(isinstance(row.get("family"), str) and bool(row["family"]) for row in rows):
        result["family_breakdown"] = {"available": True, "families": {}}
        for family in sorted({row["family"] for row in rows}):
            members = np.array([row["family"] == family for row in rows])
            result["family_breakdown"]["families"][family] = {
                partition: {"rows": int((mask & members).sum()),
                            "actual": metrics(actual, actual_target, mask & members),
                            "random": metrics(random, random_target, mask & members)}
                for partition, mask in groups.items() if bool((mask & members).any())}
    return result


def run(spec_path, spec_sha256, custody_path, custody_sha256, output):
    r.verify_file(spec_path, spec_sha256)
    r.verify_file(custody_path, custody_sha256)
    spec, custody = read_json(spec_path), read_json(custody_path)
    r.require(custody.get("schema") == 1 and custody.get("evidence_kind") == "NATIVE_STAGE_C_CAPTURE"
              and custody.get("frozen_registration_sha256") == spec_sha256 and set(custody.get("roles", {})) == set(ROLES),
              "native custody registry scope differs")
    original = verify_historical_source(spec)
    r.require(not (Path(spec["attempt_root"]).parent / "SESSION_CLOSED.json").exists(), "session closed after failed/refused worker")
    source = analysis_identity()
    all_captures, all_expected = {}, {}
    for role in ROLES:
        all_captures[role], all_expected[role] = verify_role(spec, spec_sha256, role, custody["roles"][role])
    output = Path(output).resolve()
    r.require(not output.exists() and not output.is_relative_to(ROOT), "new external analysis directory required")
    output.mkdir(parents=True)
    reports = {}
    try:
        fit_mask = np.array([row["split"] == "fit" for row in original["rows"]])
        for student_layer, donor_layer in PAIRS:
            key = f"student-{student_layer:02d}-donor-{donor_layer:02d}"
            layers = {"student_base": student_layer, "donor_base": donor_layer, "donor_skilled": donor_layer}
            captures = {role: all_captures[role][layers[role]] for role in ROLES}
            expected = {role: all_expected[role][layers[role]] for role in ROLES}
            fit, target, report = w.fit_captured_residuals(original, captures, expected,
                                                         forbidden_row_ids=set(), forbidden_parent_ids=set())
            x, y = captures["donor_base"]["tensors"]["residual"], captures["student_base"]["tensors"]["residual"]
            change = captures["donor_skilled"]["tensors"]["residual"] - x
            permutation = np.random.default_rng(1729).permutation(int(fit_mask.sum()))
            shuffled = m.fit_transport(x[fit_mask][permutation], y[fit_mask])
            donor_norm = m.delta_report({"residual": change})["l2"]
            random_residual = m.norm_match({"residual": np.random.default_rng(1729).normal(size=change.shape)}, donor_norm)["residual"]
            random_target = fit.residual(random_residual)
            shuffle_diagnostics = m.transport_diagnostics(shuffled, x[~fit_mask], y[~fit_mask])
            artifacts = {}
            for name, array in (("weight", fit.weight), ("bias", fit.bias),
                                ("donor_fit_mean", x[fit_mask].astype("float64").mean(0)),
                                ("student_fit_mean", y[fit_mask].astype("float64").mean(0)),
                                ("target_residual", target), ("random_donor_residual", random_residual),
                                ("random_target_residual", random_target),
                                ("shuffle_weight", shuffled.weight), ("shuffle_bias", shuffled.bias)):
                artifacts[name] = save_array(output, key + "-" + name + ".npy", array)
            report.update(student_layer=student_layer, donor_layer=donor_layer, fit_singular_values=fit.singular_values.tolist(),
                          effective_condition_number=float(fit.singular_values[0] / fit.singular_values[fit.rank - 1]),
                          unidentified_donor_dimensions=fit.weight.shape[0] - fit.rank, artifacts=artifacts,
                          shuffle_control={"seed": 1729, "fit_row_permutation": permutation.tolist(),
                              "heldout": shuffle_diagnostics},
                          random_donor_residual_control={"seed": 1729, "matching": "global donor residual Frobenius norm",
                              "donor_l2": donor_norm, "random_donor_l2": m.delta_report({"residual": random_residual})["l2"],
                              "transported_l2": m.delta_report({"residual": target})["l2"],
                              "random_transported_l2": m.delta_report({"residual": random_target})["l2"],
                              "heldout_overlap": m.activation_overlap(target[~fit_mask], random_target[~fit_mask])})
            report["mean_baseline_pass"] = report["heldout"]["mse"] < report["frozen_fit_mean_baseline_mse"]
            report["shuffle_baseline_pass"] = report["heldout"]["mse"] < shuffle_diagnostics["mse"]
            report["stage_c_continuation_pass"] = report["mean_baseline_pass"] and report["shuffle_baseline_pass"]
            report["residual_support"] = residual_support(x[fit_mask], change, random_residual, target, random_target,
                                                         fit_mask, spec["calibration"]["rows"])
            r.canonical_bytes(report)
            r.write_once(output / (key + ".json"), report)
            reports[key] = report
        final = reports["student-27-donor-35"]
        result = {"schema": 1, "status": "COMPLETE", "completed": True,
                  "evidence_kind": "VERIFIED_NATIVE_CAPTURE_TRANSPORT_ANALYSIS", "native_model_calls_this_analysis": 0,
                  "source_capture_commit": spec["source_commit"], "analysis_source": source,
                  "analysis_protocol": ANALYSIS_PROTOCOL, "stage_c_mechanism_amendment": spec["stage_c_mechanism_amendment"],
                  "frozen_registration_sha256": spec_sha256, "custody_registry_sha256": custody_sha256,
                  "capture_result_sha256": {role: custody["roles"][role]["result_sha256"] for role in ROLES},
                  "pairs": reports, "final_pair_continuation_pass": bool(final["stage_c_continuation_pass"]),
                  "continuation_criterion": "Final registered pair heldout MSE below BOTH frozen FIT-mean and seed1729 shuffled-FIT baselines; all provenance and nondegeneracy gates passed.",
                  "scientific_improvement_claim": False, "student_skill_acquisition_established": False}
        r.write_once(output / "RESULT.json", result)
        return result
    except BaseException as exc:
        r.write_once(output / "FAILURE.json", {"status": "FAILED", "completed": False,
                     "error_type": type(exc).__name__, "error": str(exc), "scientific_improvement_claim": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--spec-sha256", required=True)
    parser.add_argument("--custody", required=True, type=Path)
    parser.add_argument("--custody-sha256", required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    result = run(args.spec, args.spec_sha256, args.custody, args.custody_sha256, args.out)
    print(json.dumps({"status": result["status"], "final_pair_continuation_pass": result["final_pair_continuation_pass"],
                      "scientific_improvement_claim": False}))


if __name__ == "__main__":
    main()
