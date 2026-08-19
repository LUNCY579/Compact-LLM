from __future__ import annotations

import collections
import html
import statistics
from pathlib import Path
from typing import Any, Iterable

from ..utils import atomic_write_json, atomic_write_text, html_page, iter_jsonl, markdown_table, stable_int_hash, utc_now_iso


def _rate(numerator: int | float, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _group_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    strict = sum(bool(row.get("score", {}).get("correct_strict")) for row in rows)
    lenient = sum(bool(row.get("score", {}).get("correct_lenient")) for row in rows)
    latencies = []
    for row in rows:
        response = row.get("response")
        if isinstance(response, dict) and response.get("latency_ms") is not None:
            latencies.append(float(response["latency_ms"]))
    summary: dict[str, Any] = {
        "n": n,
        "strict_correct": strict,
        "strict_accuracy": _rate(strict, n),
        "lenient_correct": lenient,
        "lenient_accuracy": _rate(lenient, n),
        "strict_lenient_gap": _rate(lenient - strict, n),
        "mean_latency_ms": statistics.fmean(latencies) if latencies else None,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
    }
    parse_status = collections.Counter(
        str(row.get("score", {}).get("parse_status", "UNKNOWN")) for row in rows
    )
    summary["parse_status_counts"] = dict(sorted(parse_status.items()))
    if any("contract_valid" in row.get("score", {}) for row in rows):
        summary["contract_valid_rate"] = _rate(
            sum(bool(row.get("score", {}).get("contract_valid")) for row in rows), n
        )
        summary["done_match_rate"] = _rate(
            sum(bool(row.get("score", {}).get("done_match")) for row in rows), n
        )
        summary["state_exact_rate"] = _rate(
            sum(bool(row.get("score", {}).get("state", {}).get("exact")) for row in rows), n
        )
        summary["state_lenient_rate"] = _rate(
            sum(bool(row.get("score", {}).get("state", {}).get("lenient_equivalent")) for row in rows), n
        )
        summary["mean_state_token_f1"] = statistics.fmean(
            float(row.get("score", {}).get("state", {}).get("token_f1", 0.0)) for row in rows
        )
        summary["step_exact_rate"] = _rate(
            sum(bool(row.get("score", {}).get("step", {}).get("exact")) for row in rows), n
        )
        summary["step_lenient_rate"] = _rate(
            sum(bool(row.get("score", {}).get("step", {}).get("lenient_equivalent")) for row in rows), n
        )
    if any("termination_reason" in row.get("score", {}) for row in rows):
        termination = collections.Counter(
            str(row.get("score", {}).get("termination_reason", "UNKNOWN")) for row in rows
        )
        summary["termination_counts"] = dict(sorted(termination.items()))
        summary["mean_turns_used"] = statistics.fmean(
            float(row.get("score", {}).get("turns_used", 0.0)) for row in rows
        )
        exact_depths = [
            int(row["score"]["first_exact_state_divergence"])
            for row in rows
            if row.get("score", {}).get("first_exact_state_divergence") is not None
        ]
        lenient_depths = [
            int(row["score"]["first_lenient_state_divergence"])
            for row in rows
            if row.get("score", {}).get("first_lenient_state_divergence") is not None
        ]
        summary["mean_first_exact_divergence"] = statistics.fmean(exact_depths) if exact_depths else None
        summary["mean_first_lenient_divergence"] = statistics.fmean(lenient_depths) if lenient_depths else None
        summary["all_turns_contract_valid_rate"] = _rate(
            sum(bool(row.get("score", {}).get("all_turns_contract_valid")) for row in rows), n
        )
    return summary


def _stratify(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        value = row.get(field)
        if value is None and field == "variant":
            value = row.get("metadata", {}).get("variant")
        groups[str(value)] .append(row)
    return {key: _group_summary(group) for key, group in sorted(groups.items())}


def _macro_accuracy(groups: dict[str, Any], key: str) -> float | None:
    values = [float(value[key]) for value in groups.values() if value.get("n", 0) > 0]
    return statistics.fmean(values) if values else None


def _audit_rows(rows: list[dict[str, Any]], limit: int = 100) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if (not row.get("score", {}).get("correct_strict"))
        or row.get("score", {}).get("correct_strict") != row.get("score", {}).get("correct_lenient")
    ]
    candidates.sort(key=lambda row: stable_int_hash(str(row.get("request_id"))))
    return candidates[:limit]


def build_metrics(scored_path: str | Path) -> dict[str, Any]:
    rows = list(iter_jsonl(scored_path, tolerate_truncated_tail=True))
    by_protocol: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    for row in rows:
        by_protocol[str(row.get("protocol", "UNKNOWN"))].append(row)
    protocols: dict[str, Any] = {}
    for protocol, protocol_rows in sorted(by_protocol.items()):
        category = _stratify(protocol_rows, "category")
        domain = _stratify(protocol_rows, "domain")
        difficulty = _stratify(protocol_rows, "difficulty")
        variants = _stratify(protocol_rows, "variant") if any(row.get("metadata", {}).get("variant") for row in protocol_rows) else None
        protocols[protocol] = {
            "overall": _group_summary(protocol_rows),
            "by_category": category,
            "by_domain": domain,
            "by_difficulty": difficulty,
            "by_variant": variants,
            "macro_category_strict_accuracy": _macro_accuracy(category, "strict_accuracy"),
            "macro_category_lenient_accuracy": _macro_accuracy(category, "lenient_accuracy"),
            "worst_category_strict_accuracy": min(
                (group["strict_accuracy"] for group in category.values()), default=None
            ),
            "worst_category_lenient_accuracy": min(
                (group["lenient_accuracy"] for group in category.values()), default=None
            ),
        }
    return {
        "schema_version": "1.0",
        "generated_at": utc_now_iso(),
        "source": str(Path(scored_path).resolve()),
        "total_records": len(rows),
        "protocols": protocols,
    }


def _report_markdown(metrics: dict[str, Any]) -> str:
    lines = [
        "# Evaluation Report",
        "",
        f"Generated: `{metrics['generated_at']}`",
        "",
        "The strict track requires unambiguous contract-compliant output. The lenient track is diagnostic and credits conservatively recoverable answers or equivalent state wording. Raw generations remain available for audit.",
        "",
        "## Protocol summary",
        "",
    ]
    rows = []
    for name, payload in metrics["protocols"].items():
        overall = payload["overall"]
        rows.append(
            [
                name,
                overall["n"],
                f"{100 * overall['strict_accuracy']:.2f}%",
                f"{100 * overall['lenient_accuracy']:.2f}%",
                f"{100 * overall['strict_lenient_gap']:.2f}%",
                f"{overall['mean_latency_ms']:.2f}" if overall["mean_latency_ms"] is not None else "n/a",
            ]
        )
    lines.append(markdown_table(["Protocol", "N", "Strict", "Lenient", "Gap", "Mean latency ms"], rows))
    for name, payload in metrics["protocols"].items():
        overall = payload["overall"]
        lines.extend(["", f"## {name}", ""])
        if "contract_valid_rate" in overall:
            lines.append(
                markdown_table(
                    ["Transition metric", "Value"],
                    [
                        ["Structured contract valid", f"{100 * overall['contract_valid_rate']:.2f}%"],
                        ["Done flag match", f"{100 * overall['done_match_rate']:.2f}%"],
                        ["Next state exact", f"{100 * overall['state_exact_rate']:.2f}%"],
                        ["Next state lenient", f"{100 * overall['state_lenient_rate']:.2f}%"],
                        ["Mean next-state token F1", f"{overall['mean_state_token_f1']:.4f}"],
                        ["Next step exact", f"{100 * overall['step_exact_rate']:.2f}%"],
                        ["Next step lenient", f"{100 * overall['step_lenient_rate']:.2f}%"],
                    ],
                )
            )
        if "termination_counts" in overall:
            lines.extend(["", "### Rollout termination", ""])
            lines.append(markdown_table(["Reason", "Count"], sorted(overall["termination_counts"].items())))
        lines.extend(["", "### Category results", ""])
        category_rows = [
            [
                category,
                group["n"],
                f"{100 * group['strict_accuracy']:.2f}%",
                f"{100 * group['lenient_accuracy']:.2f}%",
            ]
            for category, group in payload["by_category"].items()
        ]
        lines.append(markdown_table(["Category", "N", "Strict", "Lenient"], category_rows))
        if payload.get("by_variant"):
            lines.extend(["", "### State-ablation variants", ""])
            variant_rows = [
                [variant, group["n"], f"{100 * group['strict_accuracy']:.2f}%", f"{100 * group['lenient_accuracy']:.2f}%"]
                for variant, group in payload["by_variant"].items()
            ]
            lines.append(markdown_table(["Variant", "N", "Strict", "Lenient"], variant_rows))
    return "\n".join(lines) + "\n"


def _report_html(metrics: dict[str, Any]) -> str:
    summary_rows = []
    for name, payload in metrics["protocols"].items():
        overall = payload["overall"]
        summary_rows.append(
            f"<tr><td>{html.escape(name)}</td><td>{overall['n']}</td><td>{100*overall['strict_accuracy']:.2f}%</td><td>{100*overall['lenient_accuracy']:.2f}%</td><td>{100*overall['strict_lenient_gap']:.2f}%</td></tr>"
        )
    body = [
        "<h1>Evaluation Report</h1>",
        f"<p class='muted'>Generated {html.escape(metrics['generated_at'])}</p>",
        "<p>The strict track requires unambiguous contract-compliant output. The lenient track is diagnostic.</p>",
        "<h2>Protocol summary</h2>",
        "<table><thead><tr><th>Protocol</th><th>N</th><th>Strict</th><th>Lenient</th><th>Gap</th></tr></thead><tbody>",
        *summary_rows,
        "</tbody></table>",
    ]
    for name, payload in metrics["protocols"].items():
        body.append(f"<h2>{html.escape(name)}</h2>")
        body.append("<table><thead><tr><th>Category</th><th>N</th><th>Strict</th><th>Lenient</th></tr></thead><tbody>")
        for category, group in payload["by_category"].items():
            body.append(
                f"<tr><td>{html.escape(category)}</td><td>{group['n']}</td><td>{100*group['strict_accuracy']:.2f}%</td><td>{100*group['lenient_accuracy']:.2f}%</td></tr>"
            )
        body.append("</tbody></table>")
    return html_page("Rineya Evaluation Report", "\n".join(body))


def build_evaluation_reports(scored_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    metrics = build_metrics(scored_path)
    atomic_write_json(output / "metrics.json", metrics)
    atomic_write_text(output / "report.md", _report_markdown(metrics))
    atomic_write_text(output / "report.html", _report_html(metrics))
    rows = list(iter_jsonl(scored_path, tolerate_truncated_tail=True))
    audit = _audit_rows(rows)
    audit_dir = output / "manual_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(audit_dir / "audit_sample.json", audit)
    audit_md = ["# Deterministic Manual-Audit Sample", "", f"Cases: {len(audit)}", ""]
    for row in audit:
        audit_md.extend(
            [
                f"## {row.get('request_id')}",
                "",
                f"- Protocol: `{row.get('protocol')}`",
                f"- Category: `{row.get('category')}`",
                f"- Strict: `{row.get('score', {}).get('correct_strict')}`",
                f"- Lenient: `{row.get('score', {}).get('correct_lenient')}`",
                "",
                "```text",
                str(row.get("raw_output", ""))[:4000],
                "```",
                "",
            ]
        )
    atomic_write_text(audit_dir / "audit_sample.md", "\n".join(audit_md))
    return metrics
