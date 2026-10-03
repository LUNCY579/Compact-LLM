"""Analysis-only cost adapter for one verified bookkeeping-only failure.

Original files remain byte-identical. A verified ACCOUNTING_COMPLETE receipt
supplies an explicitly labelled virtual terminal only to the old cost reader.
No scientific endpoint, model, weight or generation body is consumed.
"""
from __future__ import annotations

import argparse
import ast
import copy
from contextlib import contextmanager
import importlib
import inspect
import json
import math
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parents[1]
if str(WORKTREE) not in sys.path:
    sys.path.insert(0, str(WORKTREE))
from research.binding_integration_v1 import cost_report as original

ROOT = original.ROOT
ORIGINAL_COST_SHA256 = "b475d3cdc3eaefdcdcd672b033a1982d37930e93bac9001ca455e59f93d24474"
EXPECTED_RECOVERY_STATUS = "COMPLETE_SIX_PROBES_ACCOUNTING_CORRECTION"
ELAPSED_BASIS = "ACCOUNTING_CORRECTION_TIME_NOT_MODEL_COMPLETION"
require, sha, seconds = original.require, original.sha, original.seconds
observed_worker_charge, launch_identity, LEGS = original.observed_worker_charge, original.launch_identity, original.LEGS


def read_json(path):
    path = Path(path)
    sha(path)  # Refuse weight payloads, generation bodies, symlinks and oversized files.
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    require(isinstance(value, dict), "accounting correction must be a JSON object")
    return value


def assert_original_checks_preserved():
    """The copied probe reader differs by one explicit metadata filename only."""
    require(sha(Path(original.__file__)) == ORIGINAL_COST_SHA256, "original cost reporter changed; new review required")
    old_node = ast.parse(inspect.getsource(original.probe_costs)).body[0]
    new_node = ast.parse(inspect.getsource(_probe_costs_allow_accounting_receipt)).body[0]
    old_node.name = new_node.name
    matches = [node for node in ast.walk(old_node) if isinstance(node, ast.Assign) and
               any(isinstance(target, ast.Name) and target.id == "excluded" for target in node.targets)]
    require(len(matches) == 1 and isinstance(matches[0].value, ast.Set), "old cost inventory guard structure changed")
    matches[0].value.elts.append(ast.Constant(value="ACCOUNTING_COMPLETE.json"))
    require(ast.dump(old_node, include_attributes=False) == ast.dump(new_node, include_attributes=False),
            "recovered probe costs changed checks beyond the single metadata exclusion")


def load_recovery_api():
    return importlib.import_module("research.binding_integration_v1.accounting_recovery")


def recovery_context(root, api=None):
    root = Path(root).resolve()
    module = load_recovery_api() if api is None else api
    receipt = module.verify_accounting_complete(root)
    path = root / "posttrain/ACCOUNTING_COMPLETE.json"
    require(read_json(path) == receipt and receipt.get("status") == EXPECTED_RECOVERY_STATUS,
            "stored canonical accounting correction differs from its revalidated receipt")
    require(type(receipt.get("original_wrapper_returncode")) is int and receipt["original_wrapper_returncode"] == 1 and
            receipt.get("elapsed_time_basis") == ELAPSED_BASIS,
            "original wrapper failure or correction-time provenance absent")
    expected_prior = root / "posttrain/TERMINAL.json"
    require(Path(receipt["prior_terminal_path"]).resolve() == expected_prior,
            "correction names a different original terminal")
    evidence = {str(path): sha(path), str(Path(original.__file__).resolve()): ORIGINAL_COST_SHA256}
    for stem in ("prior_terminal", "dispatch", "correction_freeze", "original_report", "correction_module", "controller_exit_observation"):
        target = Path(receipt[stem + "_path"]).resolve()
        require(sha(target) == receipt[stem + "_sha256"], "correction provenance bytes changed: " + stem)
        evidence[str(target)] = receipt[stem + "_sha256"]
    require(Path(receipt["dispatch_path"]).resolve() == root / "posttrain/DISPATCH_RESERVATION.json" and
            Path(receipt["correction_freeze_path"]).resolve() == root / "ACCOUNTING_CORRECTION_FREEZE.json" and
            Path(receipt["original_report_path"]).resolve() == HERE / "report.py" and
            Path(receipt["correction_module_path"]).resolve() == HERE / "accounting_recovery.py",
            "recovery may only adapt the registered dispatch bookkeeping")
    frozen = read_json(root / "PRELAUNCH_BINDING.json")
    require(receipt["original_report_sha256"] == frozen["code_files_sha256"]["report.py"],
            "original scientific reporter differs from its immutable execution freeze")
    prior = read_json(expected_prior)
    require(prior.get("status") == "FAILED_OR_BLOCKED" and prior.get("failure") == {
                "type": "ValueError", "message": "six-probe attempt receipt inventory differs"} and
            isinstance(prior.get("attempts"), list) and len(prior["attempts"]) == 6 and
            all(type(item.get("returncode")) is int and item["returncode"] == 0 for item in prior["attempts"]),
            "only the exact post-six-success bookkeeping failure can be adapted")
    proof = receipt["proof_files_sha256"]
    require(isinstance(proof, dict) and proof, "accounting correction proof inventory absent")
    for name, pin in proof.items():
        target = Path(name)
        require(".." not in target.parts, "unsafe accounting proof path")
        if not target.is_absolute():
            target = root / target
        target = target.resolve()
        require(sha(target) == pin, "accounting proof changed")
        evidence[str(target)] = pin
    view = module.completed_terminal_view(receipt)
    require(view["status"] == "COMPLETE_SIX_PROBES" and view["attempts"] == prior["attempts"] and
            view["final_binding_sha256"] == receipt["final_binding_sha256"] and
            view["cost_accounting"] == receipt["cost_accounting"] and
            view["session_elapsed_seconds"] == receipt["session_elapsed_seconds"] and
            isinstance(view.get("accounting_correction"), dict),
            "virtual accounting view changed attempt content or omitted explicit correction identity")
    require(not (root / "ATTEMPT_MAP.json").exists() and not (root / "posttrain/RECOVERY_COMPLETE.json").exists(),
            "bookkeeping recovery may not hide a model retry or result replacement")
    return module, receipt, prior, view, evidence


def recovered_probe_costs(root, envelope, read, partial=None, *, context=None):
    root = Path(root).resolve()
    module, receipt, prior, view, evidence = recovery_context(root) if context is None else context
    expected_prior = root / "posttrain/TERMINAL.json"
    # Real reads still enter the old reporter's hash inventory; only the returned
    # interpretation of that one terminal is virtualized, never its file bytes.
    for path, pin in evidence.items():
        require(sha(path) == pin, "recovery proof changed before cost replay")
    read(root / "posttrain/ACCOUNTING_COMPLETE.json")
    def corrected_read(path):
        value = read(path)
        if Path(path).resolve() == expected_prior:
            require(value == prior, "original failure terminal changed during accounting")
            return copy.deepcopy(view)
        return value
    costs = _probe_costs_allow_accounting_receipt(root, envelope, corrected_read, partial)
    require(costs["totals"]["attempts"] == costs["totals"]["successful_attempts"] == 6 and
            costs["totals"]["failed_attempts"] == costs["totals"]["unobserved_parent_results"] == 0,
            "bookkeeping correction cannot change the six-success model inventory")
    costs["session_elapsed_at_accounting_correction_seconds"] = costs.pop("session_elapsed_at_probe_completion_seconds")
    costs["elapsed_time_basis"] = ELAPSED_BASIS
    costs["original_posttrain_terminal_status"] = prior["status"]
    costs["original_posttrain_wrapper_returncode"] = 1
    costs["bookkeeping_recovery_receipt"] = {"path": str(root / "posttrain/ACCOUNTING_COMPLETE.json"),
                                            "sha256": sha(root / "posttrain/ACCOUNTING_COMPLETE.json")}
    return costs


@contextmanager
def cost_reader_adapter(context):
    saved = original.probe_costs
    def adapted(root, envelope, read, partial=None):
        return recovered_probe_costs(root, envelope, read, partial, context=context)
    original.probe_costs = adapted
    try:
        yield
    finally:
        original.probe_costs = saved


def build_report(root=ROOT, *, now_unix=None):
    root = Path(root).resolve()
    try:
        assert_original_checks_preserved()
        context = recovery_context(root)
        module, receipt, prior, view, proof = context
        with cost_reader_adapter(context):
            result = original.build_report(root, now_unix=now_unix)
        require(module.verify_accounting_complete(root) == receipt, "accounting correction changed during cost audit")
        require(all(sha(path) == pin for path, pin in proof.items()), "accounting proof changed during cost audit")
        result["record"] = "BINDING_INTEGRATION_RECOVERED_COST_REPORT"
        result["original_cost_reporter_sha256"] = ORIGINAL_COST_SHA256
        result["reporter_sha256"] = sha(Path(__file__))
        result["original_scientific_report_sha256"] = receipt["original_report_sha256"]
        result["accounting_correction_module_sha256"] = receipt["correction_module_sha256"]
        result["canonical_accounting_receipt_sha256"] = sha(root / "posttrain/ACCOUNTING_COMPLETE.json")
        result["formal_correction_freeze_sha256"] = receipt["correction_freeze_sha256"]
        result["bookkeeping_recovery"] = {
            "original_terminal_path": str(root / "posttrain/TERMINAL.json"),
            "original_terminal_sha256": receipt["prior_terminal_sha256"],
            "original_terminal_status": "FAILED_OR_BLOCKED", "original_failure": prior["failure"],
            "original_wrapper_returncode": 1,
            "controller_exit_observation_path": receipt["controller_exit_observation_path"],
            "controller_exit_observation_sha256": receipt["controller_exit_observation_sha256"],
            "controller_exit_observation_source": "Parent-observed exec wrapper return; separate from six native supervisor return codes.",
            "successful_model_workers": 6, "model_retries_for_correction": 0,
            "original_terminal_rewritten": False, "virtual_terminal_used_only_for_offline_cost_interpretation": True,
            "elapsed_time_basis": ELAPSED_BASIS}
        result["source_receipt_sha256"].update(proof)
        result["scope_notes"] = [note for note in result["scope_notes"] if not note.startswith("The receipt records elapsed wall through probe completion")]
        result["scope_notes"].extend([
            "Original posttrain wrapper return code 1 and FAILED_OR_BLOCKED terminal remain preserved. Six successful model workers are established by the separately frozen, revalidated accounting correction.",
            "Only ACCOUNTING_COMPLETE.json is added to the old cost reader's ignored metadata inventory; native, export, storage, probe and failure-charge checks are otherwise unchanged.",
            "Canonical receipt elapsed time is measured at accounting correction creation, not inferred as model-completion time. Component intervals are not summed into session elapsed time."])
        if result["status"] == "COMPLETE_VERIFIED_COST_REPORT":
            result["status"] = "COMPLETE_VERIFIED_COST_REPORT_WITH_ACCOUNTING_RECOVERY"
        return result
    except Exception as exc:
        return {"schema": 1, "record": "BINDING_INTEGRATION_RECOVERED_COST_REPORT",
                "status": "BLOCKED_COST_ACCOUNTING", "root": str(root),
                "blockers": [str(exc)], "costs": {}, "source_receipt_sha256": {},
                "original_cost_reporter_sha256": ORIGINAL_COST_SHA256,
                "reporter_sha256": sha(Path(__file__)), "model_execution": False,
                "source_weights_read": False, "quality_claim": False, "release_permitted": False}


def markdown(value):
    body = original.markdown(value)
    note = ("Original posttrain wrapper: return code 1, FAILED_OR_BLOCKED. "
            "Six model workers succeeded. A separately frozen accounting correction resolves only the dispatch-metadata inventory error; "
            "the original terminal and execution code were not rewritten.\n\n")
    return body.replace("Status: **", note + "Status: **", 1) if "bookkeeping_recovery" in value else body


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    paths = args.root / "COST_REPORT_RECOVERED.json", args.root / "COST_REPORT_RECOVERED.md"
    require(not any(path.exists() for path in paths), "recovered cost outputs are create-only")
    value = build_report(args.root)
    if value["status"] != "COMPLETE_VERIFIED_COST_REPORT_WITH_ACCOUNTING_RECOVERY":
        print(json.dumps(value, indent=2, allow_nan=False)); return 2
    texts = json.dumps(value, indent=2, allow_nan=False) + "\n", markdown(value)
    require(sum(len(text.encode("utf-8")) for text in texts) <= original.OUTPUT_ALLOWANCE, "cost output exceeds reserved bytes")
    for path, text in zip(paths, texts):
        with path.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(text)
    print(json.dumps({"status": value["status"], "outputs": [str(path) for path in paths]}))
    return 0


def _probe_costs_allow_accounting_receipt(root, envelope, read, partial=None):
    """Charge every durable reservation, including failure/no-terminal attempts."""
    root = Path(root).resolve(); post = root / "posttrain"
    required = {"integrated-" + name for name in LEGS}
    excluded = {"PLAN.json", "TERMINAL.json", "DISPATCH_RESERVATION.json", "RECOVERY_PLAN.json", "RECOVERY_COMPLETE.json", "ACCOUNTING_COMPLETE.json"}
    replacements = {}
    if (root / "ATTEMPT_MAP.json").exists():
        mapping = read(root / "ATTEMPT_MAP.json")
        require(mapping.get("schema") == 1 and isinstance(mapping.get("replacements"), dict), "invalid explicit retry map")
        replacements = mapping["replacements"]
    require(len(replacements) <= 1, "more than one global inference retry")
    for key, relative in replacements.items():
        require(key in required and Path(relative).parts == ("eval", key + "-retry1"), "unregistered retry path")
    allowed = required | {key + "-retry1" for key in replacements}
    receipt_paths = {p.stem: p for p in post.glob("*.json") if p.name not in excluded}
    reservation_paths = {p.stem: p for p in (post / "reservations").glob("*.json")}
    require(set(receipt_paths) <= allowed and set(reservation_paths) <= allowed, "unexpected six-leg attempt/reservation")
    require(set(receipt_paths) <= set(reservation_paths), "attempt has no prospective parent reservation")
    attempts, receipts, selected, cost_pins = [], {}, {}, {}
    if partial is not None:
        partial.update(status="PARTIAL_NOT_COMPLETION_VERIFIED", attempts=attempts)
    for slot, reservation_path in sorted(reservation_paths.items()):
        reserve = read(reservation_path)
        retry = slot.endswith("-retry1"); key = slot[:-7] if retry else slot
        suffix = key.removeprefix("integrated-"); n, maximum = LEGS[suffix]
        out = root / "eval" / slot
        require(reserve["suffix"] == suffix and reserve["attempt"] == (2 if retry else 1) and
                Path(reserve["out"]).resolve() == out and reserve["reserved_rows"] == n and
                reserve["reserved_generation_tokens"] == n * maximum and reserve["reserved_worker_seconds"] == 900 and
                reserve["session_envelope_sha256"] == sha(root / "SESSION_ENVELOPE.json"),
                "parent reservation geometry/envelope differs")
        value = read(receipt_paths[slot]) if slot in receipt_paths else None
        auxiliary = suffix in ("binding", "diagnostic")
        supervisor_path = out / ("supervisor.json" if auxiliary else "worker/supervisor.json")
        supervisor = read(supervisor_path) if supervisor_path.exists() else None
        charge, basis = observed_worker_charge(supervisor)
        code = value.get("returncode") if value else None
        item = {"leg": key, "attempt_slot": slot, "out": str(out), "returncode": code,
                "foreground_wall_seconds": seconds(value["wall_seconds"]) if value else None,
                "worker_seconds_charged": charge, "accounting": basis, "reserved_rows": n,
                "reserved_generation_tokens": n * maximum, "parent_reservation_sha256": sha(reservation_path)}
        attempts.append(item)
        totals = {"attempts": len(attempts), "successful_attempts": sum(x["returncode"] == 0 for x in attempts),
                  "failed_attempts": sum(type(x["returncode"]) is int and x["returncode"] != 0 for x in attempts),
                  "unobserved_parent_results": sum(x["returncode"] is None for x in attempts),
                  "worker_seconds_charged": sum(x["worker_seconds_charged"] for x in attempts),
                  "rows_reserved": sum(x["reserved_rows"] for x in attempts),
                  "generation_tokens_reserved": sum(x["reserved_generation_tokens"] for x in attempts)}
        if partial is not None:
            partial["observed_or_reserved_totals"] = dict(totals)
        if value is None:
            continue  # Reservation and any observed failed/successful worker remain charged.
        require(value["leg"] == key and Path(value["out"]).resolve() == out and type(code) is int and
                value["reserved_rows"] == n and value["reserved_generation_tokens"] == n * maximum and
                math.isclose(seconds(value["worker_seconds_charged"]), charge, rel_tol=0, abs_tol=1e-6),
                "attempt result omitted or changed failure/unobserved charges")
        receipts[slot] = value
        cost_pins[(post / (slot + ".json")).relative_to(root).as_posix()] = sha(receipt_paths[slot])
        if supervisor is not None:
            cost_pins[supervisor_path.relative_to(root).as_posix()] = sha(supervisor_path)
            require((supervisor.get("returncode") == 0) == (code == 0), "foreground/supervisor success states disagree")
        if code != 0:
            continue
        require(supervisor is not None and supervisor["status"] == "EXITED" and supervisor["returncode"] == 0, "successful leg has no observed successful supervisor")
        launch_identity(out if auxiliary else out / "worker", supervisor, 900, read)
        terminal_path = out / ("worker_terminal.json" if auxiliary else "worker/probe_terminal.json")
        terminal = read(terminal_path)
        require(terminal["status"] == ("COMPLETE" if auxiliary else "COMPLETED") and terminal["complete"] is True and terminal["failure"] is None,
                "successful leg has no complete worker terminal")
        require(terminal["returncode" if "returncode" in terminal else "system_exit_code"] == 0 if not auxiliary else True,
                "standard worker exit differs")
        filename = "AUX_BINDING.json" if auxiliary else "PROBE_WEIGHT_BINDING.json"; binding_path = out / filename
        binding = read(binding_path)
        require(binding["status"] == "COMPLETE" and binding["complete"] is True and
                binding["controller_returncode" if auxiliary else "returncode"] == 0 and
                binding["supervisor_sha256"] == sha(supervisor_path) and
                binding["worker_terminal_sha256" if auxiliary else "terminal_sha256"] == sha(terminal_path),
                "successful custody does not bind the cost terminals")
        if auxiliary:
            require(binding["worker_returncode"] == 0, "auxiliary worker did not exit successfully")
            local_path = root / "eval" / ("AUX_RESERVATION_" + slot + ".json")
            local = read(local_path)
            require(binding["reservation_sha256"] == sha(local_path) and local["parent_reservation_sha256"] == sha(reservation_path) and
                    local["rows"] == n and local["reserved_generated_tokens"] == n * maximum, "auxiliary local/global reservation differs")
        expected = root / replacements[key] if key in replacements else root / "eval" / key
        require(out == expected and key not in selected, "successful leg replaced without an explicit failed-attempt map")
        selected[key] = sha(binding_path)
    require(reservation_paths, "no probe reservations")
    expected_directories = set(reservation_paths)
    require({p.name for p in (root / "eval").iterdir() if p.is_dir()} == expected_directories, "probe output directory has no parent reservation")
    require(not (root / "POSTTRAIN_CONTROLLER_LEASE.json").exists() and not (root / "eval/AUX_CONTROLLER_LEASE.json").exists(),
            "posttraining/auxiliary admission lease is unsettled")
    for key, ceiling in (("attempts", "probe_workers_all_attempts"), ("failed_attempts", "probe_retries_total"),
                         ("worker_seconds_charged", "probe_worker_seconds_all_attempts"), ("rows_reserved", "probe_rows_reserved_all_attempts"),
                         ("generation_tokens_reserved", "probe_generation_tokens_reserved_all_attempts")):
        require(totals[key] <= envelope[ceiling], "global six-leg attempt budget exceeded")
    require(totals["unobserved_parent_results"] == 0 and len(receipts) == len(reservation_paths),
            "parent result missing; durable reservations retained in partial ledger")
    failures = [x for x in attempts if x["returncode"] != 0]
    require(set(selected) == required and len(attempts) == 6 + len(replacements) and
            len(failures) == len(replacements) and {x["leg"] for x in failures} == set(replacements),
            "six unique successful legs and all original failures required")
    original_path = post / "TERMINAL.json"; original = read(original_path); terminal = original
    require(isinstance(original.get("attempts"), list) and original["attempts"], "battery terminal omitted attempts")
    original_slots = [Path(x["out"]).name for x in original["attempts"]]
    require(len(original_slots) == len(set(original_slots)) and
            all(receipts.get(Path(x["out"]).name) == x for x in original["attempts"]), "battery changed/duplicated attempt records")
    if replacements:
        require(original["status"] == "FAILED_OR_BLOCKED" and set(replacements) <= set(original_slots), "retry lacks preserved original failure")
        terminal = read(post / "RECOVERY_COMPLETE.json")
        require(terminal["prior_terminal_sha256"] == sha(original_path) and terminal["attempt_map_sha256"] == sha(root / "ATTEMPT_MAP.json"),
                "recovery is not linked to its preserved terminal/map")
    require(terminal["status"] == "COMPLETE_SIX_PROBES" and terminal["final_binding_sha256"] == selected,
            "six-probe complete terminal/final bindings absent")
    require(isinstance(terminal.get("attempts"), list) and
            {Path(x["out"]).name: x for x in terminal["attempts"]} == receipts and len(terminal["attempts"]) == len(receipts),
            "final terminal must retain every exact attempt, including failures")
    # Exact schema produced by frozen report.probe_cost_accounting for normal completion.
    cost_accounting = {"probe_worker_seconds": totals["worker_seconds_charged"], "probe_rows_reserved": totals["rows_reserved"],
        "probe_generation_tokens_reserved": totals["generation_tokens_reserved"], "attempt_count": len(attempts),
        "successful_attempts": 6, "failed_attempts": len(failures),
        "individual_attempts": [{"leg": x["leg"], "out": x["out"], "returncode": x["returncode"],
            "worker_seconds_charged": x["worker_seconds_charged"], "reserved_rows": x["reserved_rows"],
            "reserved_generation_tokens": x["reserved_generation_tokens"]} for x in attempts],
        "cost_source_sha256": cost_pins}
    require(terminal["cost_accounting"] == cost_accounting, "terminal cost_accounting differs from independent attempt/supervisor hashes")
    for field in ("probe_worker_seconds", "probe_rows_reserved", "probe_generation_tokens_reserved"):
        require(math.isclose(seconds(terminal[field]), cost_accounting[field], rel_tol=0, abs_tol=1e-6), "terminal totals omitted or doubled costs")
    elapsed = seconds(terminal["session_elapsed_seconds"])
    require(elapsed <= envelope["total_owned_work_wall_seconds"], "completed probe phase exceeded overall envelope")
    return {"attempts": attempts, "totals": totals, "terminal_cost_accounting": cost_accounting,
            "session_elapsed_at_probe_completion_seconds": elapsed, "measured_gpu_active_seconds": None,
            "actual_generated_tokens": None, "generation_accounting": "Full per-attempt reservations, including failure/unobserved attempts; no generation body or scientific endpoint read."}


if __name__ == "__main__":
    raise SystemExit(main())
