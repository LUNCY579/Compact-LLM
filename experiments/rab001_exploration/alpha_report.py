"""Report the fixed twelve alpha comparisons without loading model code or weights."""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from experiments.rab001_exploration import frontier_report as common

GRID = (0.0, .25, .5, .75, 1.0)
SPECS = {
    "B28": ("B28_lr1e-05_b5200_seed17", "b28-wise"),
    "P": ("P_lr1e-05_b5200_seed17_wsd0.2-0.2_b20.95", "p-output-gate"),
}
PAIRS = tuple((arm, str(alpha), str(endpoint)) for arm in SPECS for alpha in GRID[1:-1] for endpoint in (0.0, 1.0))


def output_identity(outputs):
    return {rid: {"text": out["text"], "terminated": out["terminated"]} for rid, out in outputs.items()}


def validate_sweep(body, arm, context, dose, selection=None):
    require = common.require
    canonical_tag, kind = SPECS[arm]
    selection = ({"canonical_tag": canonical_tag, "active_tag": canonical_tag, "preserved_failure": None}
                 if selection is None else selection)
    require(selection["canonical_tag"] == canonical_tag, "sweep canonical tag mismatch")
    tag = selection["active_tag"]
    common.validate_run_identity(dose, selection)
    for key, value in {"completed": True, "tag": tag, "kind": kind, "arm": arm, "mode": arm,
                       "endpoint_controls_passed": True, "registered_bare_policy": "permissive",
                       "reference_n": 47, "panel_rows": 150}.items():
        require(body.get(key) == value, f"{key} must equal {value!r}")
    require(body["grid"] == list(GRID), "alpha grid differs from fixed registration")
    require(set(body["results"]) == {str(alpha) for alpha in GRID}, "missing or extra alpha results")
    for key, value in {"cuda_matmul_allow_tf32": False, "cudnn_allow_tf32": False,
                       "float32_matmul_precision": "highest", "probe_device": "cpu", "build_device": "cuda"}.items():
        require(body["numerics"].get(key) == value, f"numerical control mismatch: {key}")
    provenance = body["provenance"]
    require(provenance["implementation_digest"] == common.DIGEST, "implementation digest mismatch")
    require(dose["completed"] and dose["reference_digest"] == common.DIGEST, "trained endpoint is incomplete or has wrong digest")
    require(dose["arm"] == arm and dose["lr"] == 1e-5 and dose["seed"] == 17 and dose["token_budget"] == 5200,
            "saved dose does not identify the registered checkpoint")
    require(dose["schedule_spec"] == ("cosine:0.03" if arm == "B28" else "wsd:0.2:0.2")
            and dose["beta2"] == (.999 if arm == "B28" else .95), "saved dose recipe mismatch")
    require(provenance["seed"] == 17 and provenance["saved_after_tokens"] == dose["supervised_target_tokens"]
            and provenance["saved_after_steps"] == dose["optimizer_steps"], "checkpoint dose binding mismatch")
    require(provenance["checkpoint_sha256"] == dose["checkpoint"]["sha256"], "checkpoint hash disagrees with saved endpoint")
    for key in ("0.0", "1.0"):
        require(body["endpoint_controls"][key]["passed"] is True, "declared endpoint control failed")
    points, rowmaps = {}, {}
    for alpha in GRID:
        key = str(alpha)
        result = body["results"][key]
        require(result["alpha"] == alpha and result["kind"] == kind, "point identity mismatch")
        actual = output_identity(result["outputs"])
        if alpha == 0.0:
            require(actual == context["source"], "alpha zero source text/termination mismatch")
        if alpha == 1.0:
            require(actual == output_identity(dose["checkpoints"][-1]["outputs"]), "alpha one saved endpoint text/termination mismatch")
        require(set(actual) == context["ids"], "alpha output ID mismatch")
        checkpoint = {"retention": {"permissive": result, "per_row": result["per_row"]},
                      "outputs": result["outputs"],
                      "answers_changed_vs_source": sum(actual[rid]["text"] != context["source"][rid]["text"] for rid in actual)}
        summary = common.summarize_checkpoint(checkpoint, context, require_outputs=True)
        require(result["gain_strata"] == summary["gained_by_stratum"], "stored alpha gain strata mismatch")
        for name in ("wrong_answer", "format_only", "runaway"):
            expected = sorted(rid for rid in summary["gained_ids"] if context["strata"][rid] == name)
            require(sorted(result["gain_ids_by_stratum"][name]) == expected, "stored gain-stratum IDs mismatch")
        points[key] = {"alpha": alpha, **summary}
        rowmaps[key] = dict(result["per_row"])
    return {"status": "VALID", "tag": tag, "canonical_tag": canonical_tag, "kind": kind, "points": points,
            "run_selection": selection,
            "endpoint_controls_independently_verified": True, "_per_row": rowmaps}


def fixed_pair_report(sweeps):
    pairs, pvalues = {}, {}
    for arm, alpha, endpoint in PAIRS:
        key = f"{arm}:alpha{alpha}~alpha{endpoint}"
        try:
            common.require(sweeps[arm]["status"] == "VALID", "sweep unavailable or invalid")
            left, right = (sweeps[arm]["_per_row"][value] for value in (alpha, endpoint))
            common.require(len(left) == 150 and set(left) == set(right), "paired IDs mismatch; intersections forbidden")
            b = sum(left[rid] and not right[rid] for rid in left)
            c = sum(right[rid] and not left[rid] for rid in left)
            pairs[key] = {"status": "AVAILABLE", "arm": arm, "alpha": float(alpha), "compared_with_alpha": float(endpoint),
                          "interior_only_correct": b, "endpoint_only_correct": c, "discordant": b + c, "rows_compared": 150}
            pvalues[key] = common.mcnemar_exact(b, c)
        except (common.InvalidArtifact, KeyError) as exc:
            pairs[key] = {"status": "UNAVAILABLE", "reason": str(exc)}
            pvalues[key] = Fraction(1)
    adjusted = common.holm(pvalues)
    complete = all(pair["status"] == "AVAILABLE" for pair in pairs.values())
    for key, pair in pairs.items():
        p, q = pvalues[key], adjusted[key]
        pair.update(p_raw_exact=str(p), p_raw=float(p), p_holm_exact=str(q), p_holm=float(q),
                    p_is_missing_placeholder=pair["status"] != "AVAILABLE",
                    significance_status="COMPLETE_FAMILY" if complete else "WITHHELD_INCOMPLETE_FAMILY",
                    significant=complete and pair["status"] == "AVAILABLE" and q < common.ALPHA)
        pair["favours"] = (("interior" if pair["interior_only_correct"] > pair["endpoint_only_correct"] else "endpoint")
                           if pair["significant"] else None)
    return pairs


def build_report(paths, context, registry=None):
    registry = common.load_run_registry() if registry is None else registry
    common.validate_run_registry(registry)
    sweeps = {}
    for arm, (tag, _) in SPECS.items():
        selection = common.resolve_run(tag, registry=registry)
        path = paths.get(arm)
        if path is None or not path.is_file():
            sweeps[arm] = {"status": "MISSING", "path": str(path) if path else None, "run_selection": selection}
            continue
        try:
            body = common.read_json(path)
            dose_path = Path(selection["artifact_path"])
            dose = common.read_json(dose_path)
            common.require(body["provenance"]["saved_endpoint_sha256"] == common.sha256(dose_path), "saved endpoint file hash mismatch")
            data_hashes = {(REPO / relative).resolve(): value for relative, value in body["provenance"]["data_hashes"].items()}
            expected_inputs = {(common.BASE / "data/monitor_probe.jsonl").resolve(),
                               (common.EVIDENCE / "PROBE_LABEL_AUDIT.json").resolve(),
                               (common.EVIDENCE / "HEADROOM_AUDIT.json").resolve(),
                               (common.EVIDENCE / "DISTINGUISHABILITY_PREREG.md").resolve()}
            common.require(set(data_hashes) == expected_inputs, "alpha input hash inventory mismatch")
            for candidate, expected in data_hashes.items():
                common.require(common.sha256(candidate) == expected, f"recorded alpha input hash mismatch: {candidate}")
            sweeps[arm] = validate_sweep(body, arm, context, dose, selection)
        except (common.InvalidArtifact, KeyError, TypeError, ValueError, OSError) as exc:
            sweeps[arm] = {"status": "INVALID", "reason": str(exc)}
        sweeps[arm].update(path=str(path), sha256=common.sha256(path), run_selection=selection)
    pairs = fixed_pair_report(sweeps)
    for sweep in sweeps.values():
        sweep.pop("_per_row", None)
    return {"schema": 1, "evidence_kind": "exploratory_fixed_alpha_family_report",
            "family_status": "COMPLETE" if all(pair["status"] == "AVAILABLE" for pair in pairs.values()) else "INCOMPLETE",
            "registration": "DISTINGUISHABILITY_PREREG.md continuation 2026-09-13 22:31 UTC",
            "fixed_grid": list(GRID), "fixed_test_count": 12, "alpha": str(common.ALPHA),
            "correction": "Holm over all twelve fixed tests; missing/invalid pairs use p=1 placeholders",
            "selection_policy": "none; the full grid is descriptive and cannot select a promoted model or soup ingredient",
            "sweeps": sweeps, "pairs": pairs, "run_registry": registry,
            "loss_classifier": {"function": "headroom_audit.failure_stratum", "script_sha256": common.sha256(Path(common.headroom_audit.__file__))},
            "licenses_efficacy_claim": False, "licenses_release": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b28", type=Path, help="B28 alpha sweep REPORT.json")
    parser.add_argument("--p", type=Path, help="P alpha sweep REPORT.json")
    parser.add_argument("--out", type=Path, required=True, help="New create-only JSON report path")
    args = parser.parse_args(argv)
    common.require(not args.out.exists(), f"output already exists: {args.out}")
    panel = [json.loads(line) for line in (common.BASE / "data/monitor_probe.jsonl").read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    context = common.make_context(panel,
        common.read_json(common.BASE / f"outputs/retention-{common.DIGEST[:8]}/B28/retention.json"),
        common.read_json(common.EVIDENCE / "HEADROOM_AUDIT.json"),
        common.read_json(common.EVIDENCE / "PROBE_LABEL_AUDIT.json"),
        common.read_json(common.BASE / "protocol.json")["G1_RETENTION"])
    result = build_report({"B28": args.b28, "P": args.p}, context)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"{result['family_status']}: {sum(p['status'] == 'AVAILABLE' for p in result['pairs'].values())}/12 comparisons available; {args.out}")
    return 2 if any(sweep["status"] == "INVALID" for sweep in result["sweeps"].values()) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except common.InvalidArtifact as exc:
        raise SystemExit(str(exc))
