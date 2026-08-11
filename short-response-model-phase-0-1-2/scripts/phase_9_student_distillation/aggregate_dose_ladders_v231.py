from __future__ import annotations

"""CPU-only v2.3.1 dose ladders from retained probe eval summaries."""
import argparse, json
from pathlib import Path
from typing import Any


def load(p: Path) -> dict[str, Any]:
    return json.loads(p.read_text(encoding="utf-8-sig"))


def num(d: dict[str, Any], k: str) -> float | None:
    try:
        v = d.get(k)
        return None if v is None else float(v)
    except (TypeError, ValueError):
        return None


def cond(summary: dict[str, Any], name: str) -> dict[str, Any]:
    c = summary.get("conditions")
    if isinstance(c, dict):
        return c.get(name) or {}
    return {}


def weight_from_name(name: str) -> float | None:
    if name == "greedy_anchor":
        return 0.0
    # reasoning_dose_0p10 / procedure_dose_1p00
    parts = name.rsplit("_", 1)
    if len(parts) != 2:
        return None
    tag = parts[-1]
    try:
        return float(tag.replace("p", "."))
    except ValueError:
        return None


def ladder_for(dose_root: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not dose_root.exists():
        return rows
    for child in sorted(dose_root.iterdir()):
        if not child.is_dir():
            continue
        manifest = child / "eval" / "candidate_evaluation_manifest.json"
        if not manifest.exists():
            continue
        m = load(manifest)
        name = child.name
        direct = load(Path(m["v23_direct_summary"])) if m.get("v23_direct_summary") else {}
        ood = load(Path(m["v23_direct_ood_summary"])) if m.get("v23_direct_ood_summary") else {}
        proc = load(Path(m["v23_procedural_summary"])) if m.get("v23_procedural_summary") else {}
        proc_ood = load(Path(m["v23_procedural_ood_summary"])) if m.get("v23_procedural_ood_summary") else {}
        real = cond(proc, "real_rag")
        oracle = cond(proc, "oracle_rag")
        real_ood = cond(proc_ood, "real_rag")
        oracle_ood = cond(proc_ood, "oracle_rag")
        rows.append(
            {
                "name": name,
                "label": label,
                "weight": weight_from_name(name),
                "evidence_scope": "dose_probe",
                "direct_case_count": direct.get("case_count"),
                "direct": {
                    "answerable": num(direct, "answerable_correctness"),
                    "answer_coverage": num(direct, "answer_coverage"),
                    "false_abstention": num(direct, "false_abstention_rate"),
                    "abstention": num(direct, "abstention_correctness"),
                },
                "direct_ood": {
                    "answerable": num(ood, "answerable_correctness"),
                    "answer_coverage": num(ood, "answer_coverage"),
                    "false_abstention": num(ood, "false_abstention_rate"),
                    "abstention": num(ood, "abstention_correctness"),
                },
                "v23_procedure": {
                    "dev_real_rag": num(real, "procedure_correctness"),
                    "dev_oracle_rag": num(oracle, "procedure_correctness"),
                    "ood_real_rag": num(real_ood, "procedure_correctness"),
                    "ood_oracle_rag": num(oracle_ood, "procedure_correctness"),
                    "dev_case_count": proc.get("case_count"),
                    "ood_case_count": proc_ood.get("case_count"),
                },
            }
        )
    rows.sort(key=lambda r: (r.get("weight") is None, r.get("weight") or 0.0, r["name"]))
    return rows


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# v2.3.1 Dose Ladders (probe reuse)",
        "",
        "Probe directs are smoke-sized (~720). Procedural columns use retained dose-eval summaries.",
        "Named full-matrix candidates are evaluated separately under `diagnostics_v231/eval_*`.",
        "",
    ]
    for key, title in (("reasoning", "Reasoning residual doses"), ("procedure", "Procedure residual doses")):
        rows = payload.get(key) or []
        lines += [f"## {title}", ""]
        if key == "procedure":
            lines += [
                "| Dose | Weight | Ans DEV | Cov DEV | FA DEV | Ans OOD | Cov OOD | FA OOD | Proc DEV real | Proc DEV oracle | Proc OOD real | Proc OOD oracle |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in rows:
                d, o, p = r["direct"], r["direct_ood"], r["v23_procedure"]
                lines.append(
                    f"| {r['name']} | {r.get('weight')} | {d['answerable']} | {d['answer_coverage']} | {d['false_abstention']} | "
                    f"{o['answerable']} | {o['answer_coverage']} | {o['false_abstention']} | "
                    f"{p['dev_real_rag']} | {p['dev_oracle_rag']} | {p['ood_real_rag']} | {p['ood_oracle_rag']} |"
                )
        else:
            lines += [
                "| Dose | Weight | Ans DEV | Cov DEV | FA DEV | Ans OOD | Cov OOD | FA OOD | Proc DEV real | Proc OOD real |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
            for r in rows:
                d, o, p = r["direct"], r["direct_ood"], r["v23_procedure"]
                lines.append(
                    f"| {r['name']} | {r.get('weight')} | {d['answerable']} | {d['answer_coverage']} | {d['false_abstention']} | "
                    f"{o['answerable']} | {o['answer_coverage']} | {o['false_abstention']} | "
                    f"{p['dev_real_rag']} | {p['ood_real_rag']} |"
                )
        lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v23-root", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    a = ap.parse_args()
    payload = {
        "schema_version": "2.3.1",
        "reasoning": ladder_for(a.v23_root / "dose_reasoning", "reasoning"),
        "procedure": ladder_for(a.v23_root / "dose_procedure", "procedure"),
    }
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    a.output_md.write_text(markdown(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
