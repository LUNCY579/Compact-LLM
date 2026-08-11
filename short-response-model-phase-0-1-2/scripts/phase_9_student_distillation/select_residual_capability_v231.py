from __future__ import annotations

"""v2.3.1 research matrix: keep rejected candidates with status + cost decomposition."""
import argparse, json
from pathlib import Path
from typing import Any

from scripts.phase_9_student_distillation.select_residual_capability_v23 import load, select


def annotate_research(out: dict[str, Any], manifest: dict[str, Any] | None = None) -> dict[str, Any]:
    official = out["official_policy"]
    selected = out.get("selected_candidate")
    by_name = {c["name"]: c for c in (manifest or {}).get("candidates", [])}
    for c in out["candidates"]:
        if c.get("technical_invalidity"):
            status = "TECHNICALLY_INVALID"
        elif c["name"] == selected:
            status = "SELECTED"
        else:
            status = "REJECTED_BY_POLICY"
        c["research_status"] = status
        src = by_name.get(c["name"]) or {}
        if src.get("evidence_scope"):
            c["evidence_scope"] = src["evidence_scope"]
        if src.get("weight_note") is not None:
            c["weight_note"] = src["weight_note"]
        pol = c.get("policies", {}).get(official, {})
        c["research_policy_snapshot"] = {
            "benefit": pol.get("benefit"),
            "preservation_cost": pol.get("preservation_cost"),
            "opportunity_margin": pol.get("opportunity_margin"),
            "preservation_cost_decomposition": pol.get("preservation_cost_decomposition"),
        }
    out["schema_version"] = "2.3.1"
    out["research_note"] = (
        "Production selection is unchanged. REJECTED_BY_POLICY candidates remain in the "
        "matrix for research reporting (capability unlocked vs preservation cost). "
        "evidence_scope=dose_probe means smoke/probe evals; full_dev_ood means the full suite."
    )
    return out


def markdown(out: dict[str, Any]) -> str:
    official = out["official_policy"]
    lines = [
        "# v2.3.1 Residual Capability Research Matrix",
        "",
        f"**Production status:** `{out['status']}`",
        f"**Selected (production):** `{out['selected_candidate']}`",
        f"**Official policy:** `{official}`",
        "",
        out.get("research_note", ""),
        "",
        "| Candidate | Status | Scope | Track | Ans DEV | Cov DEV | FA DEV | Ans OOD | Cov OOD | FA OOD | V23 proc DEV | V23 proc OOD | Benefit | Pres. cost | Margin |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in out["candidates"]:
        p = c["policies"].get(official, {})
        lines.append(
            f"| {c['name']} | {c.get('research_status')} | {c.get('evidence_scope', '')} | {c.get('track')} | "
            f"{c['direct']['answerable']} | {c['direct'].get('answer_coverage')} | {c['direct']['false_abstention']} | "
            f"{c['direct_ood']['answerable']} | {c['direct_ood'].get('answer_coverage')} | {c['direct_ood']['false_abstention']} | "
            f"{c['v23_procedure']['dev']} | {c['v23_procedure']['ood']} | "
            f"{p.get('benefit')} | {p.get('preservation_cost')} | {p.get('opportunity_margin')} |"
        )
    lines += ["", "## Preservation-cost decomposition", ""]
    for c in out["candidates"]:
        snap = c.get("research_policy_snapshot") or {}
        decomp = snap.get("preservation_cost_decomposition") or {}
        weighted = decomp.get("weighted") or {}
        unweighted = decomp.get("unweighted") or {}
        lines += [
            f"### `{c['name']}` (`{c.get('research_status')}`)",
            "",
            f"- Benefit: `{snap.get('benefit')}`",
            f"- Preservation cost (total): `{snap.get('preservation_cost')}`",
            f"- Opportunity margin: `{snap.get('opportunity_margin')}`",
            f"- Unweighted: protected_loss_sum=`{unweighted.get('original_protected_relative_loss_sum')}`, "
            f"false_abstention=`{unweighted.get('false_abstention_cost')}`, "
            f"abstention_loss=`{unweighted.get('abstention_correctness_loss')}`, "
            f"shortness=`{unweighted.get('shortness_loss')}`",
            f"- Weighted: original=`{weighted.get('original_preservation')}`, "
            f"false_abstention=`{weighted.get('false_abstention')}`, "
            f"abstention_loss=`{weighted.get('abstention_correctness_loss')}`, "
            f"shortness=`{weighted.get('shortness_loss')}`",
            "",
        ]
    lines += [
        f"Pareto frontier: `{', '.join(out.get('pareto_frontier') or [])}`",
        f"Policy selections: `{out.get('policy_selections')}`",
        "",
        out.get("sealed_policy", ""),
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--candidates-manifest", type=Path, required=True)
    ap.add_argument("--output-json", type=Path, required=True)
    ap.add_argument("--output-md", type=Path, required=True)
    ap.add_argument("--also-copy-md", type=Path, action="append", default=[])
    a = ap.parse_args()
    manifest = load(a.candidates_manifest)
    out = annotate_research(select(load(a.config), manifest), manifest)
    a.output_json.parent.mkdir(parents=True, exist_ok=True)
    a.output_json.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    md = markdown(out)
    a.output_md.write_text(md, encoding="utf-8")
    for p in a.also_copy_md:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(md, encoding="utf-8")
    print(json.dumps(out, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
