from __future__ import annotations

"""v2.3.1 diagnostic-completion: rematerialize rejected doses, research matrix, oracle, overlap."""
import argparse, json, shutil, subprocess, sys
from pathlib import Path
from typing import Any


def resolve_python() -> str:
    """Prefer a torch-capable interpreter when PATH python is a lightweight env."""
    import os
    import shutil

    candidates: list[Path] = []
    here = Path(__file__).resolve()
    # short-response-model-phase-0-1-2/scripts/... -> repo root Chat-Bubble
    phase_root = here.parents[2]
    repo_root = phase_root.parent
    candidates += [
        phase_root / ".venv" / "Scripts" / "python.exe",
        phase_root / ".venv" / "bin" / "python",
        repo_root / ".venv" / "Scripts" / "python.exe",
        repo_root / ".venv" / "bin" / "python",
    ]
    env_py = os.environ.get("V23_PYTHON") or os.environ.get("PYTHON")
    if env_py:
        candidates.insert(0, Path(env_py))
    for c in candidates:
        if c.exists():
            return str(c)
    which = shutil.which("python")
    return which or sys.executable


def run(args: list[str]) -> None:
    import os

    print("+", " ".join(args), flush=True)
    env = os.environ.copy()
    root = str(Path.cwd().resolve())
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(args, check=True, env=env)


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def write_json(p: Path, obj: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def tag(w: float) -> str:
    return f"{w:.2f}".replace(".", "p")


def best_nonzero_weight(dose_root: Path, policy: str = "preservation_guarded") -> float:
    """Least-negative / best opportunity_margin among weight>0 doses."""
    result = dose_root / "DOSE_SWEEP_RESULT.json"
    if not result.exists():
        return 1.0
    data = load(result)
    cands = (data.get("final_selection") or data.get("coarse_selection") or {}).get("candidates") or []
    nonzero = []
    for c in cands:
        name = str(c.get("name") or "")
        if name == "greedy_anchor":
            continue
        # recover weight from name suffix or policies path
        w = None
        if "_dose_" in name:
            try:
                w = float(name.rsplit("_", 1)[-1].replace("p", "."))
            except ValueError:
                w = None
        if w is None or w <= 0:
            continue
        pol = (c.get("policies") or {}).get(policy) or {}
        nonzero.append((float(pol.get("opportunity_margin") or -1e9), float(pol.get("benefit") or 0), w, name))
    if not nonzero:
        return 1.0
    nonzero.sort(reverse=True)
    return float(nonzero[0][2])


def compose(py: str, base: str, ingredients: list[dict[str, Any]], model_dir: Path, force: bool) -> Path:
    model_dir.parent.mkdir(parents=True, exist_ok=True)
    manifest = model_dir / "v23_composition_manifest.json"
    if manifest.exists() and not force:
        return model_dir
    spec = model_dir.parent / "composition.json"
    write_json(spec, {"schema_version": "2.3.1", "ingredients": ingredients})
    args = [
        py,
        "scripts/phase_9_student_distillation/compose_residual_adapters_v23.py",
        "--base-model",
        base,
        "--spec",
        str(spec),
        "--output-dir",
        str(model_dir),
        "--force",
    ]
    run(args)
    return model_dir


def eval_full(
    py: str,
    model: str,
    dest: Path,
    paths: dict[str, Path],
    require_cuda: bool,
    reuse: bool,
) -> Path:
    manifest = dest / "candidate_evaluation_manifest.json"
    if reuse and manifest.exists():
        return manifest
    args = [
        py,
        "scripts/phase_9_student_distillation/evaluate_candidate_v23.py",
        "--model",
        model,
        "--output-root",
        str(dest),
        "--original-benchmark",
        str(paths["original_dev"]),
        "--original-db",
        str(paths["original_db"]),
        "--original-calibration",
        str(paths["original_cal"]),
        "--historical-procedural-benchmark",
        str(paths["v211_dev"]),
        "--historical-procedural-db",
        str(paths["v211_db"]),
        "--historical-procedural-calibration",
        str(paths["v211_cal"]),
        "--v23-rag-benchmark",
        str(paths["v23_rag_dev"]),
        "--v23-rag-ood-benchmark",
        str(paths["v23_rag_ood"]),
        "--v23-rag-db",
        str(paths["v23_rag_db"]),
        "--v23-rag-calibration",
        str(paths["v23_rag_cal"]),
        "--v23-direct-benchmark",
        str(paths["direct_dev"]),
        "--v23-direct-ood-benchmark",
        str(paths["direct_ood"]),
        "--v212-benchmark",
        str(paths["v212_dev"]),
        "--v22-benchmark",
        str(paths["v22_dev"]),
        "--maximum-evidence-tokens",
        "384",
        "--research-ready-rag",
        "--resume",
    ]
    if require_cuda:
        args.append("--require-cuda")
    run(args)
    return manifest


def eval_discovery_direct(py: str, model: str, dest: Path, bench: Path, require_cuda: bool, reuse: bool) -> Path:
    out = dest / "results.jsonl"
    if reuse and out.exists():
        return out
    args = [
        py,
        "scripts/phase_9_student_distillation/evaluate_residual_reasoning_v23.py",
        "--model",
        model,
        "--benchmark",
        str(bench),
        "--output-dir",
        str(dest),
        "--minimum-cases",
        "500",
        "--resume",
    ]
    if require_cuda:
        args.append("--require-cuda")
    run(args)
    return out


def eval_discovery_rag(
    py: str, model: str, dest: Path, bench: Path, db: Path, cal: Path, require_cuda: bool, reuse: bool
) -> Path:
    out = dest / "results.jsonl"
    if reuse and out.exists():
        return out
    args = [
        py,
        "scripts/phase_9_student_distillation/evaluate_rag.py",
        "--model",
        model,
        "--benchmark",
        str(bench),
        "--output-dir",
        str(dest),
        "--knowledge-db",
        str(db),
        "--gate-calibration",
        str(cal),
        "--conditions",
        "real_rag",
        "--retrieval-selection-policy",
        "query_aware_v23",
        "--maximum-evidence-tokens",
        "384",
        "--max-new-tokens",
        "32",
        "--minimum-cases-per-condition",
        "1",
        "--minimum-supported-cases",
        "1",
        "--minimum-unsupported-cases",
        "1",
        "--resume",
    ]
    if require_cuda:
        args.append("--require-cuda")
    run(args)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v23-root", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--greedy-model", type=Path, required=True)
    ap.add_argument("--v22-run-root", type=Path, required=True)
    ap.add_argument("--original-corpus-root", type=Path, required=True)
    ap.add_argument("--v211-corpus-root", type=Path, required=True)
    ap.add_argument("--v212-corpus-root", type=Path, required=True)
    ap.add_argument("--v22-corpus-root", type=Path, required=True)
    ap.add_argument("--v23-direct-root", type=Path, required=True)
    ap.add_argument("--v23-rag-root", type=Path, required=True)
    ap.add_argument("--require-cuda", action="store_true")
    ap.add_argument("--reuse-existing", action="store_true")
    ap.add_argument("--force-compose", action="store_true")
    ap.add_argument("--skip-gpu", action="store_true", help="CPU-only reporting from existing artifacts")
    a = ap.parse_args()
    py = resolve_python()
    print(f"Using python: {py}", flush=True)
    root = a.v23_root
    diag = root / "diagnostics_v231"
    diag.mkdir(parents=True, exist_ok=True)
    models = diag / "models"
    evals = diag / "eval"
    disc = diag / "discovery_eval"
    reasoning_adapter = root / "residual_reasoning_training" / "adapter"
    procedure_adapter = root / "residual_procedure_training" / "adapter"
    cfg = load(a.config)
    policy = str(cfg["selection"]["official_policy"])
    rw_best = best_nonzero_weight(root / "dose_reasoning", policy)
    pw_best = best_nonzero_weight(root / "dose_procedure", policy)

    named = [
        ("reasoning_best_nonzero", [("reasoning_residual", reasoning_adapter, rw_best)], "residual_reasoning"),
        ("reasoning_full", [("reasoning_residual", reasoning_adapter, 1.0)], "residual_reasoning"),
        ("procedure_best_nonzero", [("procedure_residual", procedure_adapter, pw_best)], "residual_procedure"),
        ("procedure_full", [("procedure_residual", procedure_adapter, 1.0)], "residual_procedure"),
        (
            "best_rp_composition",
            [
                ("reasoning_residual", reasoning_adapter, rw_best),
                ("procedure_residual", procedure_adapter, pw_best),
            ],
            "residual_joint_composition",
        ),
    ]

    paths = {
        "original_dev": a.original_corpus_root / "reviewed_splits" / "dev.reviewed.jsonl",
        "original_db": a.v22_run_root / "original_phase9.sqlite3"
        if (a.v22_run_root / "original_phase9.sqlite3").exists()
        else root / "original_phase9.sqlite3",
        "original_cal": a.v22_run_root / "original_calibration_query_aware_v21.json"
        if (a.v22_run_root / "original_calibration_query_aware_v21.json").exists()
        else root / "original_calibration_query_aware_v21.json",
        "v211_dev": a.v211_corpus_root / "reviewed_splits" / "dev.reviewed.jsonl",
        "v211_db": a.v22_run_root / "procedural_v211.sqlite3"
        if (a.v22_run_root / "procedural_v211.sqlite3").exists()
        else root / "procedural_v211.sqlite3",
        "v211_cal": a.v22_run_root / "v211_calibration_query_aware_v21.json"
        if (a.v22_run_root / "v211_calibration_query_aware_v21.json").exists()
        else root / "v211_calibration_query_aware_v21.json",
        "v23_rag_dev": a.v23_rag_root / "splits" / "dev.reviewed.jsonl",
        "v23_rag_ood": a.v23_rag_root / "splits" / "ood_dev.reviewed.jsonl",
        "v23_rag_db": root / "v23_procedural.sqlite3",
        "v23_rag_cal": root / "v23_procedural_discovery_calibration.json",
        "direct_dev": a.v23_direct_root / "splits" / "dev.reviewed.jsonl",
        "direct_ood": a.v23_direct_root / "splits" / "ood_dev.reviewed.jsonl",
        "direct_discovery": a.v23_direct_root / "splits" / "discovery.reviewed.jsonl",
        "rag_discovery": a.v23_rag_root / "splits" / "discovery.reviewed.jsonl",
        "v212_dev": a.v212_corpus_root / "reviewed_splits" / "dev.reviewed.jsonl",
        "v22_dev": a.v22_corpus_root / "reviewed_splits" / "dev.reviewed.jsonl",
    }

    model_paths: dict[str, Path] = {}
    if not a.skip_gpu:
        for name, ings, _track in named:
            for _n, adapter, _w in ings:
                if not Path(adapter).exists():
                    raise FileNotFoundError(adapter)
            ingredients = [{"name": n, "adapter": str(ad), "weight": float(w)} for n, ad, w in ings]
            model_dir = models / name / "model"
            model_paths[name] = compose(py, str(a.greedy_model), ingredients, model_dir, a.force_compose or not a.reuse_existing)
            eval_full(py, str(model_paths[name]), evals / name, paths, a.require_cuda, a.reuse_existing)
            # discovery alignment for oracle (skip joint for discovery to keep roster lean: full + best nonzero)
            if name != "best_rp_composition":
                eval_discovery_direct(
                    py, str(model_paths[name]), disc / name / "direct", paths["direct_discovery"], a.require_cuda, a.reuse_existing
                )
                eval_discovery_rag(
                    py,
                    str(model_paths[name]),
                    disc / name / "rag",
                    paths["rag_discovery"],
                    paths["v23_rag_db"],
                    paths["v23_rag_cal"],
                    a.require_cuda,
                    a.reuse_existing,
                )
    else:
        for name, _ings, _track in named:
            m = models / name / "model"
            if m.exists():
                model_paths[name] = m

    # Research candidate manifest (greedy + named). Prefer full evals; fall back to dose probes.
    greedy_eval = root / "eval_greedy_anchor"
    candidates = [
        {
            "name": "greedy_anchor",
            "track": "baseline_v22_greedy",
            "model": str(a.greedy_model),
            "adapter": None,
            "evaluation_manifest": str(greedy_eval / "candidate_evaluation_manifest.json"),
            "evidence_scope": "full_dev_ood",
        }
    ]

    def probe_manifest(label: str, weight: float) -> Path | None:
        dose_root = root / f"dose_{label}"
        name = f"{label}_dose_{tag(weight)}"
        p = dose_root / name / "eval" / "candidate_evaluation_manifest.json"
        return p if p.exists() else None

    probe_fallback = {
        "reasoning_best_nonzero": ("reasoning", rw_best),
        "reasoning_full": ("reasoning", 1.0),
        "procedure_best_nonzero": ("procedure", pw_best),
        "procedure_full": ("procedure", 1.0),
    }
    for name, _ings, track in named:
        man = evals / name / "candidate_evaluation_manifest.json"
        scope = "full_dev_ood"
        if not man.exists():
            if name == "best_rp_composition":
                print(f"WARNING: missing full eval for {name}; skipping from matrix until composed", flush=True)
                continue
            label, w = probe_fallback[name]
            alt = probe_manifest(label, w)
            if alt is None:
                print(f"WARNING: missing probe/full eval for {name}; skipping", flush=True)
                continue
            man = alt
            scope = "dose_probe"
            print(f"NOTE: using dose-probe evidence for {name} -> {man}", flush=True)
        candidates.append(
            {
                "name": name,
                "track": track,
                "model": str(model_paths.get(name) or ""),
                "adapter": None,
                "evidence_scope": scope,
                "weight_note": {
                    "reasoning_best_nonzero": rw_best,
                    "procedure_best_nonzero": pw_best,
                    "best_rp_composition": {"reasoning": rw_best, "procedure": pw_best},
                }.get(name),
                "evaluation_manifest": str(man),
            }
        )
    cand_manifest = diag / "candidates_research_v231.json"
    write_json(cand_manifest, {"schema_version": "2.3.1", "candidates": candidates})

    # Research matrix
    run(
        [
            py,
            "scripts/phase_9_student_distillation/select_residual_capability_v231.py",
            "--config",
            str(a.config),
            "--candidates-manifest",
            str(cand_manifest),
            "--output-json",
            str(diag / "research_matrix_v231.json"),
            "--output-md",
            str(diag / "RESEARCH_MATRIX_V231.md"),
            "--also-copy-md",
            str(root / "RESIDUAL_CAPABILITY_V23_RESEARCH_MATRIX.md"),
        ]
    )

    # Dose ladders (CPU)
    run(
        [
            py,
            "scripts/phase_9_student_distillation/aggregate_dose_ladders_v231.py",
            "--v23-root",
            str(root),
            "--output-json",
            str(diag / "dose_ladders_v231.json"),
            "--output-md",
            str(diag / "DOSE_LADDERS_V231.md"),
        ]
    )

    # Teacher overlap (CPU, mining)
    mining = root / "residual_mining"
    run(
        [
            py,
            "scripts/phase_9_student_distillation/analyze_teacher_overlap_v231.py",
            "--teacher",
            f"track_a={mining / 'direct_track_a' / 'results.jsonl'}",
            "--teacher",
            f"track_b={mining / 'direct_track_b' / 'results.jsonl'}",
            "--teacher",
            f"procedure_teacher={mining / 'direct_procedure_teacher' / 'results.jsonl'}",
            "--greedy",
            str(mining / "direct_greedy" / "results.jsonl"),
            "--output-json",
            str(diag / "teacher_overlap.json"),
            "--output-md",
            str(diag / "TEACHER_OVERLAP.md"),
        ]
    )

    # Discovery oracle (teachers + residuals)
    oracle_direct = [
        f"greedy={mining / 'direct_greedy' / 'results.jsonl'}",
        f"track_a={mining / 'direct_track_a' / 'results.jsonl'}",
        f"track_b={mining / 'direct_track_b' / 'results.jsonl'}",
        f"procedure_teacher={mining / 'direct_procedure_teacher' / 'results.jsonl'}",
    ]
    oracle_rag = [
        f"greedy={mining / 'rag_greedy' / 'results.jsonl'}",
        f"track_b={mining / 'rag_track_b' / 'results.jsonl'}",
        f"procedure_teacher={mining / 'rag_procedure_teacher' / 'results.jsonl'}",
    ]
    for name in ("reasoning_best_nonzero", "reasoning_full", "procedure_best_nonzero", "procedure_full"):
        d = disc / name / "direct" / "results.jsonl"
        r = disc / name / "rag" / "results.jsonl"
        if d.exists():
            oracle_direct.append(f"{name}={d}")
        if r.exists():
            oracle_rag.append(f"{name}={r}")
    disc_oracle_args = [
        py,
        "scripts/phase_9_student_distillation/oracle_routing_upper_bound_v23.py",
        "--section-label",
        "discovery_teachers_and_residuals",
        "--output-json",
        str(diag / "oracle_routing_upper_bound.json"),
        "--output-md",
        str(diag / "ORACLE_ROUTING_UPPER_BOUND.md"),
    ]
    for s in oracle_direct:
        disc_oracle_args += ["--direct", s]
    for s in oracle_rag:
        disc_oracle_args += ["--rag", s]
    run(disc_oracle_args)

    # DEV residual-only oracle
    dev_oracle_args = [
        py,
        "scripts/phase_9_student_distillation/oracle_routing_upper_bound_v23.py",
        "--section-label",
        "dev_greedy_and_residuals",
        "--output-json",
        str(diag / "oracle_routing_dev_residuals.json"),
        "--output-md",
        str(diag / "ORACLE_ROUTING_DEV_RESIDUALS.md"),
        "--direct",
        f"greedy={greedy_eval / 'v23_direct' / 'results.jsonl'}",
        "--rag",
        f"greedy={greedy_eval / 'v23_procedural_rag' / 'results.jsonl'}",
    ]
    for name in ("reasoning_best_nonzero", "reasoning_full", "procedure_best_nonzero", "procedure_full", "best_rp_composition"):
        d = evals / name / "v23_direct" / "results.jsonl"
        r = evals / name / "v23_procedural_rag" / "results.jsonl"
        if d.exists():
            dev_oracle_args += ["--direct", f"{name}={d}"]
        if r.exists():
            dev_oracle_args += ["--rag", f"{name}={r}"]
    run(dev_oracle_args)

    # Fix misleading single-candidate oracle under diagnostics/
    old_diag = root / "diagnostics"
    old_md = old_diag / "ORACLE_ROUTING_UPPER_BOUND.md"
    if old_md.exists():
        broken = old_diag / "ORACLE_ROUTING_UPPER_BOUND.v23_broken_single_candidate.md"
        if not broken.exists():
            shutil.copy2(old_md, broken)
        old_md.write_text(
            "# v2.3 Oracle Routing Upper Bound (superseded)\n\n"
            "The original v2.3 oracle only received `greedy` because residual dose selection "
            "returned weight 0 and the runner gated oracle candidates on selected weight > 0.\n\n"
            "See `../diagnostics_v231/ORACLE_ROUTING_UPPER_BOUND.md` for the corrected multi-specialist "
            "oracle (teachers + residuals + best nonzero doses, including answerable-only).\n\n"
            "Archived original: `ORACLE_ROUTING_UPPER_BOUND.v23_broken_single_candidate.md`.\n",
            encoding="utf-8",
        )

    # Point production selection at research diagnostics without changing winner
    sel_path = root / "residual_capability_v23_selection.json"
    if sel_path.exists():
        sel = load(sel_path)
        sel["research_diagnostics_ref"] = {
            "schema_version": "2.3.1",
            "research_matrix": str(diag / "research_matrix_v231.json"),
            "research_matrix_md": str(root / "RESIDUAL_CAPABILITY_V23_RESEARCH_MATRIX.md"),
            "oracle": str(diag / "ORACLE_ROUTING_UPPER_BOUND.md"),
            "dose_ladders": str(diag / "DOSE_LADDERS_V231.md"),
            "teacher_overlap": str(diag / "TEACHER_OVERLAP.md"),
            "named_weights": {
                "reasoning_best_nonzero": rw_best,
                "procedure_best_nonzero": pw_best,
                "reasoning_full": 1.0,
                "procedure_full": 1.0,
                "best_rp_composition": {"reasoning": rw_best, "procedure": pw_best},
            },
        }
        write_json(sel_path, sel)

    summary = {
        "schema_version": "2.3.1",
        "v23_root": str(root),
        "diagnostics_root": str(diag),
        "named_weights": {"reasoning_best_nonzero": rw_best, "procedure_best_nonzero": pw_best},
        "candidates": [c["name"] for c in candidates],
        "skip_gpu": a.skip_gpu,
    }
    write_json(diag / "DIAGNOSTIC_COMPLETION_SUMMARY.json", summary)
    (diag / "DIAGNOSTIC_COMPLETION_SUMMARY.md").write_text(
        "# v2.3.1 Diagnostic Completion\n\n"
        f"- Named weights: reasoning_best_nonzero=`{rw_best}`, procedure_best_nonzero=`{pw_best}`\n"
        f"- Research matrix: `{diag / 'RESEARCH_MATRIX_V231.md'}`\n"
        f"- Dose ladders: `{diag / 'DOSE_LADDERS_V231.md'}`\n"
        f"- Oracle: `{diag / 'ORACLE_ROUTING_UPPER_BOUND.md'}`\n"
        f"- Teacher overlap: `{diag / 'TEACHER_OVERLAP.md'}`\n"
        "- Production selected candidate remains `greedy_anchor`.\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
