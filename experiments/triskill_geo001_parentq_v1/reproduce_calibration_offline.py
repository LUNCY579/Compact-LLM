"""Offline, no-model reproduction of the reviewed PARENTQ draft's design findings.

Run only through the programme's supervised small-numerical-fixture slot. This is
not a native worker, qualification, freeze, model test, or grant. The legacy rule
and generator are read from commit 3e07fd87; GEO benchmark bytes are checked against
the original section-H freeze. Only its eight T/H parents are constructed. No J/K
oracle, native output, sealed bank, tokenizer, checkpoint, torch or transformers is
read/imported. All receipts must be outside tracked source, under BusinessProject.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

LEGACY_COMMIT = "3e07fd8740e8ff4debd94f9c17ca3e06d056a0ed"
GEO_FREEZE_SHA256 = "f18d48670e8928afb72b7182756f8f7bb6c8e241fb5988ae1daa91ebc534b687"
LANE = "experiments/triskill_geo001_parentq_v1"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def load_bytes(name, data, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(data, str(path), "exec"), module.__dict__)
    return module


def committed_bytes(root, filename):
    result = subprocess.run(["git", "show", f"{LEGACY_COMMIT}:{LANE}/{filename}"],
                            cwd=root, check=True, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE)
    return result.stdout


def no_model_modules():
    forbidden = ("torch", "transformers", "safetensors")
    found = [name for name in sys.modules
             if any(name == package or name.startswith(package + ".") for package in forbidden)]
    if found:
        raise RuntimeError(f"model modules imported: {found}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=20000)
    args = parser.parse_args()
    root, output = args.source_root.resolve(), args.output_dir.resolve()
    business = root.parent.resolve()
    if output == root or root in output.parents or business not in output.parents:
        raise RuntimeError("output must be under BusinessProject, outside tracked source")
    if ".codex" in (part.lower() for part in output.parts):
        raise RuntimeError("outputs under .codex are forbidden")
    if any((parent / ".git").exists() for parent in (output, *output.parents)):
        raise RuntimeError("outputs inside a Git checkout are forbidden")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1" or os.environ.get("PYTHONDONTWRITEBYTECODE") != "1":
        raise RuntimeError("require CUDA_VISIBLE_DEVICES=-1 and PYTHONDONTWRITEBYTECODE=1")
    if not 1 <= args.simulations <= 20000:
        raise RuntimeError("fixture simulation cap is 20000")
    no_model_modules()
    import numpy as np

    freeze_path = root / "experiments/triskill_geometric_pilot_v1/FREEZE_SECTIONH.json"
    freeze_bytes = freeze_path.read_bytes()
    if digest(freeze_bytes) != GEO_FREEZE_SHA256:
        raise RuntimeError("original GEO freeze changed")
    freeze = json.loads(freeze_bytes)
    benchmark_path = freeze_path.parent / "benchmark.py"
    benchmark_bytes = benchmark_path.read_bytes()
    if digest(benchmark_bytes) != freeze["files"]["benchmark.py"]:
        raise RuntimeError("frozen GEO benchmark changed")
    benchmark = load_bytes("_calibration_geo_benchmark", benchmark_bytes, benchmark_path)
    decide_bytes = committed_bytes(root, "parentq_decide.py")
    rows_bytes = committed_bytes(root, "parentq_rows.py")
    d = load_bytes("_calibration_legacy_decide", decide_bytes, root / LANE / "parentq_decide.py")
    rowmod = load_bytes("_calibration_legacy_rows", rows_bytes, root / LANE / "parentq_rows.py")

    # Build exactly the T/H screen parents; never call GEO's full build_screen,
    # development or any chain constructor (those also generate J/K golds).
    spec = benchmark._SECTION_H["screen"]
    screen = tuple(benchmark._transition_h("screen", i + 1, family, payload)
                   for i, (family, payload) in enumerate(spec["t"])) + tuple(
        benchmark._answerability_h("screen", world, kind, number)
        for world, kind, number in spec["h"])
    benchmark.build_screen = lambda: screen
    tables = d.gate_tables(benchmark)
    bank = rowmod.build_bank(benchmark)
    # Canary strings here are explicitly fabricated mechanics fixtures, not the
    # exposed CONFIRM-002 stored predictions or a native identity measurement.
    canary = [{"row_id": f"fixture-canary-{i}",
               "stored": {"BASE": f"base-{i}", "SPARSE": f"t-{i}", "H25": f"h-{i}"}}
              for i in range(8)]
    sources = {"BASE": "BASE", "T": "SPARSE", "H": "H25"}

    def h_reply(row, mode):
        if mode == "abstain":
            return benchmark.ABSTAIN_PHRASE
        if mode == "right":
            return row.expected if row.expected != benchmark.UNKNOWN else benchmark.ABSTAIN_PHRASE
        return row.expected if row.expected != benchmark.UNKNOWN else "41"

    def outputs(t_right=lambda row, i: True, base_t_right=lambda row, i: False,
                h_on=lambda row, i: "right", h_off=lambda row, i: "right",
                base_on=lambda row, i: "answer", base_off=lambda row, i: "answer"):
        out = {arm: {row["row_id"]: row["stored"][sources[arm]] for row in canary} for arm in sources}
        for i, row in enumerate(bank["T"]):
            out["T"][row.row_id] = row.expected if t_right(row, i) else "{}"
            out["BASE"][row.row_id] = row.expected if base_t_right(row, i) else "{}"
        for block, parent, base in (("H_ON", h_on, base_on), ("H_OFF", h_off, base_off)):
            for i, row in enumerate(bank[block]):
                out["H"][row.row_id] = h_reply(row, parent(row, i))
                out["BASE"][row.row_id] = h_reply(row, base(row, i))
        return out

    def verdict(out):
        return d.decide(benchmark, bank, canary, out, {"padding_parity_ok": True}, tables=tables)

    scenarios = {
        "perfect_parent_fixture": {},
        "legacy_middling_shuffled_index_mod3": {
            "t_right": lambda row, i: i % 3 != 0,
            "h_on": lambda row, i: "right" if i % 3 else "answer"},
        "legacy_half_shuffled_index_mod2": {
            "t_right": lambda row, i: i % 2 != 0,
            "h_on": lambda row, i: "right" if i % 2 else "answer"},
        "H_never_abstains": {"h_on": lambda row, i: "answer"},
        "H_always_abstains_BASE_correct": {
            "h_on": lambda row, i: "abstain", "base_on": lambda row, i: "right"},
        "T_BASE_both_wrong": {"t_right": lambda row, i: False},
        "BASE_ceiling_both_parents_correct": {
            "base_t_right": lambda row, i: True, "base_on": lambda row, i: "right"},
    }
    illustrations = {}
    for name, policies in scenarios.items():
        v = verdict(outputs(**policies))
        illustrations[name] = {"status": v["status"], "gate": v["screen_parent_gate"],
                               "cells_T": v["cells_T"], "cells_H_on": v["cells_H_on"]}

    # Equal BASE and H stochastic accuracy independently on all four H rows is
    # an explicit no-advantage toy null. Its rate is not a universal false-pass
    # bound: correlation, type-specific rates and deterministic decoding matter.
    null = []
    for p in (0.08, 0.15, 0.25, 0.5, 0.6, 0.75, 0.9):
        cells = [(1 - p) ** 2, (1 - p) * p, p * (1 - p), p ** 2]
        null.append({"BASE_and_H_accuracy": p,
                     "H_gate_pass_probability": d.plug_in(tables["H"], {kind: cells for kind in d.H_KINDS})})

    def batch_predictive(table, alphas):
        n = args.simulations
        total = np.zeros(n)
        for combo, passed in zip(table["combos"], table["passed"]):
            if not passed:
                continue
            term = np.ones(n)
            by_subtype = {}
            for kind, cell in zip(table["subtypes"], combo):
                by_subtype.setdefault(kind, []).append(cell)
            for kind, cells in by_subtype.items():
                alpha = alphas[kind]
                numerator, denominator = np.ones(n), np.ones(n)
                for cell, k in Counter(cells).items():
                    for j in range(k):
                        numerator *= alpha[:, cell] + j
                for j in range(len(cells)):
                    denominator *= alpha.sum(axis=1) + j
                term *= numerator / denominator
            total += term
        return total

    rng = np.random.default_rng(20261005)
    alphas = {}
    # Prospective toy with true pass .817152: T row success .8, H answers
    # correct, H missing abstention .6, BASE T/missing wrong and answers correct.
    for kind in d.T_SUBTYPES:
        success = rng.binomial(6, 0.8, args.simulations)
        alphas[kind] = np.column_stack((6 - success, success, np.zeros_like(success), np.zeros_like(success))) + d.PRIOR
    alphas["answerable"] = np.tile([0, 0, 0, 12], (args.simulations, 1)) + d.PRIOR
    for kind in ("distractor", "cue"):
        success = rng.binomial(10, 0.6, args.simulations)
        alphas[kind] = np.column_stack((10 - success, success, np.zeros_like(success), np.zeros_like(success))) + d.PRIOR
    simulated_p = batch_predictive(tables["T"], alphas) * batch_predictive(tables["H"], alphas)
    truth = {kind: [0.2, 0.8, 0, 0] for kind in d.T_SUBTYPES}
    truth.update({"answerable": [0, 0, 0, 1], "distractor": [0.4, 0.6, 0, 0], "cue": [0.4, 0.6, 0, 0]})
    calibration = {"description": "independent T .8; H answers 1, missing abstention .6; BASE T/missing 0, answers 1",
                   "true_gate_probability": d.plug_in(tables["T"], truth) * d.plug_in(tables["H"], truth),
                   "simulations": args.simulations, "seed": 20261005,
                   "qualified_fraction": float(np.mean(simulated_p >= d.QUALIFIED_AT)),
                   "marginal_fraction": float(np.mean((simulated_p >= d.NOT_QUALIFIED_BELOW) & (simulated_p < d.QUALIFIED_AT))),
                   "not_qualified_fraction": float(np.mean(simulated_p < d.NOT_QUALIFIED_BELOW)),
                   "predictive_quantiles_10_50_90": [float(np.quantile(simulated_p, q)) for q in (.1, .5, .9)]}

    conditional = []
    for row in bank["T"]:
        if rowmod.subtype(row) != "register_machine:conditional":
            continue
        payload = json.loads(row.user)
        before, event = payload["before"], payload["event"]
        branch = before[event["other"]] > event["threshold"]
        conditional.append({"row_id": row.row_id, "value": event["value"], "true_branch": branch,
                            "applied_delta": event["value"] if branch else -event["value"]})
    screen_conditional = next(row for row in screen if row.task_kind == "transition_parent"
                              and rowmod.subtype(row) == "register_machine:conditional")
    payload = json.loads(screen_conditional.user)
    e, b = payload["event"], payload["before"]

    def scale_key(row):
        payload, answer = json.loads(row.user), json.loads(row.expected)
        before, event = payload["before"], payload["event"]
        return ("ledger_replay", "scale", before[event["account"]], event["amount"], answer[event["account"]])

    bank_t14 = next(row for row in bank["T"] if row.row_id == "h-parentq-T14-ledger_replay")
    screen_t2 = next(row for row in screen if row.row_id == "h-screen-T2-ledger_replay")
    representativeness = {"conditional_rows": conditional,
                          "conditional_positive_value_count": sum(r["value"] > 0 for r in conditional),
                          "conditional_negative_value_count": sum(r["value"] < 0 for r in conditional),
                          "screen_conditional": {"row_id": screen_conditional.row_id, "value": e["value"],
                                                 "true_branch": b[e["other"]] > e["threshold"]},
                          "operative_scale_collision": {"bank_row": bank_t14.row_id, "screen_row": screen_t2.row_id,
                                                        "bank_key": scale_key(bank_t14), "screen_key": scale_key(screen_t2),
                                                        "collides": scale_key(bank_t14) == scale_key(screen_t2)}}

    blanket = verdict(outputs(h_on=lambda row, i: "answer", h_off=lambda row, i: "abstain"))["h_contract"]
    nonspecific = verdict(outputs(h_on=lambda row, i: "answer", base_on=lambda row, i: "right",
                                 base_off=lambda row, i: "right"))["h_contract"]
    missing_row = next(row for row in bank["H_ON"] if row.expected == benchmark.UNKNOWN)
    label_holes = {
        "blanket_H_off_abstention": blanket,
        "NOT_SPECIFIC_without_BASE_off_gain": nonspecific,
        "natural_phrases_uncredited": {text: benchmark.score_response(missing_row, text)
                                       for text in ("The seal's code was not recorded.", "The seal's code was not provided.")},
        "H_ON_lenient_available_but_not_reported_by_legacy_decide":
            sum(d.lenient_answer(benchmark, row, f"The code is {row.expected}.")
                for row in bank["H_ON"] if row.expected != benchmark.UNKNOWN),
    }
    no_model_modules()
    receipt = {"schema": 1, "experiment": "TRISKILL-GEO-001-PARENTQ-OFFLINE-CALIBRATION",
               "created_utc": datetime.now(timezone.utc).isoformat(), "legacy_commit": LEGACY_COMMIT,
               "purpose": "synthetic design reproduction; not parent qualification or Rineya capability evidence",
               "source_identity": {"GEO_FREEZE_SECTIONH.json": digest(freeze_bytes),
                                   "benchmark.py": digest(benchmark_bytes), "legacy_parentq_decide.py": digest(decide_bytes),
                                   "legacy_parentq_rows.py": digest(rows_bytes),
                                   "reproduce_calibration_offline.py": digest(Path(__file__).read_bytes())},
               "scope": {"constructed_screen_T_H_parents": len(screen), "J_K_oracles_constructed": 0,
                         "native_model_loads": 0, "native_forwards": 0, "native_generations": 0,
                         "training_updates": 0, "checkpoint_reads": 0, "protected_bank_reads": 0,
                         "canary": "fabricated mechanics strings, no native canary result"},
               "gate_truth_table_pass_counts": {kind: sum(table["passed"]) for kind, table in tables.items()},
               "legacy_illustrations": illustrations, "null_H_toy_models": null,
               "calibration_toy": calibration, "bank_representativeness": representativeness,
               "legacy_label_holes": label_holes,
               "limitations": ["Equal independent Bernoulli nulls are illustrative, not a universal error bound.",
                               "Toy calibration assumes exchangeability that the bank metadata fails to establish.",
                               "No native determinism, padding, canary or parent capability was tested.",
                               "No design alternative, new bank, GEO amendment or freeze is selected."]}
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / "CPU_CALIBRATION_REPRODUCTION.json"
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    summary = ["# Offline synthetic PARENTQ calibration reproduction", "",
               "No model, native outcome or capability evidence. Legacy draft remains interpretable.", "",
               "| Legacy scenario | Predictive P | Status |", "|---|---:|---|"]
    summary += [f"| {name} | {item['gate']['predictive']:.9f} | {item['status']} |"
                for name, item in illustrations.items()]
    summary += ["", "| Equal BASE/H toy accuracy | H gate pass |", "|---:|---:|"]
    summary += [f"| {item['BASE_and_H_accuracy']:.2f} | {item['H_gate_pass_probability']:.9f} |" for item in null]
    summary += ["", f"Declared toy true combined pass: {calibration['true_gate_probability']:.9f}.",
                f"Across {args.simulations} synthetic banks: qualified {calibration['qualified_fraction']:.4f}, "
                f"marginal {calibration['marginal_fraction']:.4f}, not qualified {calibration['not_qualified_fraction']:.4f}.",
                "", f"Conditional bank signs: {representativeness['conditional_positive_value_count']} positive, "
                f"{representativeness['conditional_negative_value_count']} negative; screen value {e['value']}.",
                f"T14/T2 operative collision: {scale_key(bank_t14)} == {scale_key(screen_t2)}.",
                "", f"Legacy blanket-abstention label: {blanket['label']}; H_OFF answerable abstained "
                f"{blanket['rates']['H_off']['answerable_abstained']}/12.",
                "", "All choices remain prospective and unfrozen; see PROSPECTIVE_DESIGN_OPTIONS.md."]
    with (output / "CPU_CALIBRATION_REPRODUCTION.md").open("x", encoding="utf-8") as handle:
        handle.write("\n".join(summary) + "\n")
    print(json.dumps({"receipt": str(receipt_path), "sha256": digest(receipt_path.read_bytes()),
                      "perfect_predictive": illustrations['perfect_parent_fixture']['gate']['predictive'],
                      "calibration_toy": calibration, "native_model_loads": 0}, indent=2))


if __name__ == "__main__":
    main()
