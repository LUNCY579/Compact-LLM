from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from .run_context import RunContext
from .utils import read_json, read_jsonl, utc_timestamp, write_json


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def _candidate_table(context: RunContext) -> str:
    rows: list[str] = []
    protocol = context.config.get("evaluation.selection_protocol")
    aliases = context.state.payload.get("aliases") or {}
    seen: set[str] = set()
    for alias in (
        "anchor",
        "greedy",
        "uniform",
        "global_best",
        "layerwise_seed",
        "layerwise_best",
        "selected",
    ):
        candidate_id = aliases.get(alias)
        if not candidate_id or candidate_id in seen:
            continue
        seen.add(candidate_id)
        evaluation_dir = context.path("evaluations", "dev", str(protocol), str(candidate_id))
        result_path = evaluation_dir / "candidate_result.json"
        if not result_path.exists():
            continue
        result = read_json(result_path)
        summary = result.get("summary") or {}
        recipe = result.get("recipe") or {}
        rows.append(
            "| {alias} | `{cid}` | {alpha} | {layers} | {objective} | {quality} | {tail} | {tokens} |".format(
                alias=alias,
                cid=candidate_id,
                alpha=_fmt(recipe.get("global_alpha"), 3),
                layers=len(recipe.get("layer_alphas") or {}),
                objective=_fmt((result.get("objective") or {}).get("score")),
                quality=_fmt(summary.get("mean_quality_score")),
                tail=_fmt(summary.get("worst_decile_mean_quality_score")),
                tokens=_fmt(summary.get("median_reply_tokens"), 2),
            )
        )
    return "\n".join(rows) or "| No completed candidates | - | - | - | - | - | - | - |"


def make_plots(context: RunContext) -> list[str]:
    if not context.config.get("reporting.make_plots", True):
        return []
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    generated: list[str] = []
    plot_dir = context.path("reports", "plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    global_rows = _read_csv(context.path("analysis", "global", "interpolation_curve.csv"))
    if global_rows:
        x = [float(row["alpha"]) for row in global_rows]
        y = [float(row["objective"]) for row in global_rows]
        fig, ax = plt.subplots()
        ax.plot(x, y, marker="o")
        ax.set_xlabel("Greedy coefficient α")
        ax.set_ylabel("Development objective")
        ax.set_title("Global anchor→greedy interpolation")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "global_interpolation.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path.relative_to(context.root)))

    layer_rows = _read_csv(
        context.path("analysis", "layers", "layer_influence_summary.csv")
    )
    if layer_rows:
        layers = [int(row["layer"]) for row in layer_rows]
        means = [float(row["combined_mean"]) for row in layer_rows]
        low = [float(row["ci_low"]) for row in layer_rows]
        high = [float(row["ci_high"]) for row in layer_rows]
        error = [
            [max(0.0, mean - lo) for mean, lo in zip(means, low)],
            [max(0.0, hi - mean) for hi, mean in zip(high, means)],
        ]
        fig, ax = plt.subplots()
        ax.errorbar(layers, means, yerr=error, marker="o", capsize=3)
        ax.axhline(0.0, linewidth=1)
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Greedy-layer utility advantage")
        ax.set_title("Bidirectional behavioral layer influence")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "layer_influence.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path.relative_to(context.root)))

        target = [float(row["centered_best_alpha"]) for row in layer_rows]
        smooth = [float(row["smoothed_seed_alpha"]) for row in layer_rows]
        fig, ax = plt.subplots()
        ax.plot(layers, target, marker="o", label="Centered-grid target")
        ax.plot(layers, smooth, marker="x", label="Reliability-shrunk seed")
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlabel("Transformer layer")
        ax.set_ylabel("Greedy coefficient αₗ")
        ax.set_title("Layer-wise coefficients")
        ax.legend()
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "layer_coefficients.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path.relative_to(context.root)))

    delta_rows = _read_csv(
        context.path("analysis", "layers", "delta_influence_join.csv")
    )
    if delta_rows:
        x = [float(row["relative_delta_norm"]) for row in delta_rows]
        y = [float(row["absolute_behavioral_influence"]) for row in delta_rows]
        labels = [str(row["layer"]) for row in delta_rows]
        fig, ax = plt.subplots()
        ax.scatter(x, y)
        for x_value, y_value, label in zip(x, y, labels):
            ax.annotate(label, (x_value, y_value), xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Relative parameter delta norm")
        ax.set_ylabel("Absolute behavioral influence")
        ax.set_title("Weight movement vs. intervention influence (diagnostic)")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        path = plot_dir / "delta_vs_behavioral_influence.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path.relative_to(context.root)))

    probability_path = context.path("analysis", "probability", "token_profile.jsonl")
    if probability_path.exists():
        probability_rows = read_jsonl(probability_path)
        if probability_rows:
            x = [float(row["anchor_target_probability"]) for row in probability_rows]
            y = [float(row["greedy_target_probability"]) for row in probability_rows]
            upper = max([1e-9, *x, *y])
            fig, ax = plt.subplots()
            ax.scatter(x, y, alpha=0.35)
            ax.plot([0.0, upper], [0.0, upper], linewidth=1)
            ax.set_xlim(0.0, upper)
            ax.set_ylim(0.0, upper)
            ax.set_xlabel("Anchor probability of target token")
            ax.set_ylabel("Greedy probability of target token")
            ax.set_title("Endpoint target-token probabilities")
            ax.grid(True, alpha=0.25)
            fig.tight_layout()
            path = plot_dir / "endpoint_target_token_probabilities.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            generated.append(str(path.relative_to(context.root)))
    return generated


def write_research_report(context: RunContext) -> Path:
    plots = make_plots(context)
    review_packet = write_human_review_packet(context)
    compatibility = (
        read_json(context.path("manifests", "compatibility.json"))
        if context.path("manifests", "compatibility.json").exists()
        else {}
    )
    split = (
        read_json(context.path("manifests", "dataset_split_audit.json"))
        if context.path("manifests", "dataset_split_audit.json").exists()
        else {}
    )
    global_selection = (
        read_json(context.path("analysis", "global", "global_selection.json"))
        if context.path("analysis", "global", "global_selection.json").exists()
        else None
    )
    endpoint_comparison = (
        read_json(context.path("analysis", "global", "endpoint_comparison.json"))
        if context.path("analysis", "global", "endpoint_comparison.json").exists()
        else None
    )
    delta_influence = (
        read_json(context.path("analysis", "layers", "delta_influence_correlation.json"))
        if context.path("analysis", "layers", "delta_influence_correlation.json").exists()
        else None
    )
    layer_seed = (
        read_json(context.path("analysis", "layers", "layerwise_seed_selection.json"))
        if context.path("analysis", "layers", "layerwise_seed_selection.json").exists()
        else None
    )
    layer_best = (
        read_json(context.path("analysis", "layers", "layerwise_best_selection.json"))
        if context.path("analysis", "layers", "layerwise_best_selection.json").exists()
        else None
    )
    freeze = (
        read_json(context.path("selection", "selection_freeze.json"))
        if context.path("selection", "selection_freeze.json").exists()
        else None
    )
    sealed = (
        read_json(context.path("sealed", "sealed_test_result.json"))
        if context.path("sealed", "sealed_test_result.json").exists()
        else None
    )
    probability = (
        read_json(context.path("analysis", "probability", "summary.json"))
        if context.path("analysis", "probability", "summary.json").exists()
        else None
    )
    secondary_protocols = (
        read_json(context.path("analysis", "protocols", "secondary_protocols.json"))
        if context.path("analysis", "protocols", "secondary_protocols.json").exists()
        else None
    )
    plot_lines = "\n".join(f"- `{path}`" for path in plots) or "- None generated"
    global_text = "Not completed."
    if global_selection:
        selected = global_selection["selected"]
        global_text = (
            f"Selected α = **{_fmt(selected['recipe']['global_alpha'], 4)}**, "
            f"development objective = **{_fmt(selected['objective']['score'])}**. "
            f"Paired probability of improvement over the best endpoint = "
            f"**{_fmt(global_selection['selected_minus_best_endpoint']['probability_improvement'], 3)}**."
        )
    endpoint_text = "Endpoint complementarity was not computed."
    if endpoint_comparison:
        complementarity = endpoint_comparison.get("complementarity") or {}
        endpoint_text = (
            f"Anchor wins: **{complementarity.get('anchor_wins', 'N/A')}**; "
            f"Greedy wins: **{complementarity.get('greedy_wins', 'N/A')}**; "
            f"ties: **{complementarity.get('ties', 'N/A')}**. "
            f"The descriptive per-case oracle margin over the better endpoint is "
            f"**{_fmt(complementarity.get('per_case_oracle_margin_over_best_endpoint'))}**."
        )

    layer_text = "Not completed."
    if layer_seed:
        layer_text = (
            f"The intervention scan covered **{len(layer_seed['layers'])} layers**. "
            f"The seed objective was **{_fmt(layer_seed['seed_result']['objective']['score'])}**."
        )
        if layer_best:
            layer_text += (
                f" Coordinate refinement evaluated **{layer_best['candidate_evaluations']}** candidates "
                f"and selected objective **{_fmt(layer_best['selected']['objective']['score'])}**."
            )
    delta_text = "Parameter-delta association was not computed."
    if delta_influence:
        estimate = (
            ((delta_influence.get("correlations") or {}).get("relative_delta_norm") or {})
            .get("absolute_behavioral_influence", {})
            .get("spearman", {})
        )
        delta_text = (
            f"Across **{delta_influence.get('matched_layer_count', 'N/A')} matched layers**, "
            f"the diagnostic Spearman association between relative parameter-delta norm and "
            f"absolute behavioral influence was **{_fmt(estimate.get('coefficient'))}** "
            f"(exploratory permutation p = **{_fmt(estimate.get('permutation_p_value'))}**). "
            "This statistic is descriptive only and never enters coefficient selection."
        )

    probability_text = "The optional endpoint probability profile was not run."
    if probability:
        probability_text = (
            f"Teacher-forced profiling covered **{probability.get('case_count', 'N/A')} cases** "
            f"and **{probability.get('token_position_count', 'N/A')} target-token positions**. "
            f"Mean coarse-grained Jensen–Shannon lower bound: "
            f"**{_fmt(probability.get('mean_coarse_js_lower_bound'))}**; mean top-k Jaccard: "
            f"**{_fmt(probability.get('mean_top_k_jaccard'))}**. Mean target-token probability "
            f"was **{_fmt(probability.get('mean_anchor_target_probability'))}** for Anchor and "
            f"**{_fmt(probability.get('mean_greedy_target_probability'))}** for Greedy. "
            f"Mean EOS probability was **{_fmt(probability.get('mean_anchor_eos_probability'))}** "
            f"for Anchor and **{_fmt(probability.get('mean_greedy_eos_probability'))}** for Greedy."
        )

    secondary_text = "No secondary generation protocol was run."
    if secondary_protocols:
        if secondary_protocols.get("enabled") is False:
            secondary_text = "Secondary protocol evaluation was disabled by configuration."
        else:
            protocol_lines: list[str] = []
            for protocol_name, values in (secondary_protocols.get("protocols") or {}).items():
                records = list(values or [])
                if not records:
                    protocol_lines.append(f"- `{protocol_name}`: no finalist evidence")
                    continue
                best = max(
                    records,
                    key=lambda item: float(((item.get("objective") or {}).get("score", float("-inf")))),
                )
                protocol_lines.append(
                    f"- `{protocol_name}`: {len(records)} unique finalists; best candidate "
                    f"`{best.get('candidate_id')}` with objective "
                    f"**{_fmt((best.get('objective') or {}).get('score'))}**"
                )
            secondary_text = "\n".join(protocol_lines) or "No enabled secondary protocol produced evidence."

    selection_text = "Selection has not been frozen."
    if freeze:
        selection_text = (
            f"Candidate `{freeze['selected_candidate_id']}` was frozen using development evidence only, "
            f"with objective **{_fmt(freeze['selected_development_objective']['score'])}**."
        )
    sealed_text = "The sealed test has not been run."
    if sealed:
        result = sealed["result"]
        sealed_text = (
            f"The frozen candidate was evaluated once on the sealed split. Test objective: "
            f"**{_fmt(result['objective']['score'])}**; mean quality: "
            f"**{_fmt(result['summary'].get('mean_quality_score'))}**."
        )

    text = f"""# Rineya Anchor–Greedy Layer-wise Soup Research Report

Generated: {utc_timestamp()}  
Run: `{context.run_id}`  
Selection protocol: `{context.config.get('evaluation.selection_protocol')}`  
Quality evaluator: **v4.5**

## Research question

Can two compatible 12-layer Rineya checkpoints—the preserved anchor and the improved greedy-soup endpoint—be combined more effectively than selecting either endpoint, first with a uniform/global coefficient and then with independently estimated layer coefficients?

The pipeline tests three nested hypotheses:

1. **Uniform soup:** a simple 50/50 average may lie in a useful low-error region.
2. **Global interpolation:** one scalar α may outperform both endpoints.
3. **Layer-adaptive behavioral soup:** different layers may prefer different αₗ values, provided those values are selected by intervention evidence rather than activation magnitude alone.

## Experimental controls and integrity

- Compatible checkpoint shapes: **{compatibility.get('compatible', False)}**
- Parameter count: **{int(compatibility.get('parameter_count', 0)):,}**
- Discovered layers: `{compatibility.get('layer_indices', [])}`
- Development cases: **{(split.get('dev') or {}).get('case_count', 'N/A')}**
- Sealed-test cases: **{(split.get('test') or {}).get('case_count', 'N/A')}**
- Source checkpoints are never modified; every candidate is materialized in this run directory.
- All coefficient choices use the development split. The sealed split is accessed only after `selection_freeze.json` exists.
- Floating-point tensors are merged with FP32 accumulation; quantized endpoints are rejected by default.

## Candidate overview

| Alias | Candidate | Global α | Explicit layer α count | Objective | Mean quality | Worst-decile quality | Median tokens |
|---|---|---:|---:|---:|---:|---:|---:|
{_candidate_table(context)}

## Global interpolation result

{global_text}

The objective uses predeclared transformations and weights. Noise is not removed by altering endpoint scores after seeing results; uncertainty is estimated with paired bootstrap resampling over the same stable case IDs.

## Endpoint complementarity

{endpoint_text}

The per-case oracle is an upper-bound diagnostic. It is not a deployable router, not a soup score, and not used to tune coefficients.

## Layer-wise influence and adaptive merge

{layer_text}

For each layer, the lab performs an Anchor→Greedy single-layer replacement and, when enabled, the reverse Greedy→Anchor replacement. It records the change in per-response utility, combines the two directions, bootstraps the mean effect, and separately tests a centered α grid around the globally selected model. Weak coefficients are shrunk toward the global α and neighboring coefficients are softly regularized before coordinate refinement.

**Interpretation boundary:** this is an intervention-based estimate of behavioral influence under the configured evaluation distribution. It is not a unique causal decomposition of every token, and it does not claim that a high-influence layer is universally important for all tasks.

## Parameter movement versus behavioral influence

{delta_text}

A large norm difference does not reveal whether Anchor or Greedy is better, and a small movement may still cross an output decision boundary. The pipeline therefore selects coefficients from behavioral interventions, not weight magnitude.

## Endpoint probability-distribution diagnostic

{probability_text}

The distribution profile is teacher-forced on a common target trajectory. Its divergence value is computed after coarse-graining the vocabulary and is therefore reported only as a conservative lower bound. It is diagnostic and does not select any merge coefficient.

## Secondary protocol robustness

{secondary_text}

Secondary protocols are evaluated only after the development selection is frozen. They are descriptive robustness checks and cannot replace or alter the predeclared selection protocol within this run. A thinking-mode coefficient search requires a separate run with that protocol declared as the selection protocol from the start.

## Frozen selection

{selection_text}

## Sealed evaluation

{sealed_text}

## Generated plots

{plot_lines}

## Primary evidence files

- `manifests/compatibility.json`
- `manifests/dataset_split_audit.json`
- `analysis/global/interpolation_curve.csv`
- `analysis/global/global_selection.json`
- `analysis/global/endpoint_case_complementarity.csv`
- `analysis/global/endpoint_category_complementarity.csv`
- `analysis/layers/per_response_layer_influence.jsonl`
- `analysis/layers/layer_influence_summary.csv`
- `analysis/layers/delta_influence_join.csv`
- `analysis/layers/delta_influence_correlation.json`
- `analysis/layers/centered_layer_grid.csv`
- `analysis/layers/coordinate_search_trace.jsonl`
- `analysis/probability/summary.json` (when enabled)
- `analysis/probability/token_profile.jsonl` (when enabled)
- `analysis/protocols/secondary_protocols.json` (when enabled)
- `analysis/protocols/secondary_protocol_scores.csv` (when enabled)
- `selection/selection_freeze.json`
- `sealed/access_ledger.json`

## Limitations

- The Model Soups results motivate compatible weight averaging, but do not guarantee that these two language-model endpoints share a sufficiently flat interpolation path.
- The “one layer” finding concerns where RL adaptation can be learned when other layers are frozen; this lab tests post-hoc checkpoint interventions and merging, which is related but not identical.
- Repeated development evaluation can overfit a small benchmark. Use a sufficiently large, diverse development set and reserve the sealed split for one frozen candidate.
- A layer coefficient can compensate for interactions elsewhere. Coordinate descent is therefore a practical search method, not proof of an independently optimal layer.
- Quality Evaluator v4.5 remains an automated instrument. Inspect raw replies and obtain human review before a production decision.
"""
    path = context.path("reports", "RESEARCH_REPORT.md")
    path.write_text(text, encoding="utf-8")
    write_json(
        context.path("reports", "report_manifest.json"),
        {
            "generated_at": utc_timestamp(),
            "report": str(path),
            "plots": plots,
            "human_review_packet": [str(value) for value in review_packet] if review_packet else None,
        },
    )
    return path


def write_human_review_packet(context: RunContext) -> tuple[Path, Path] | None:
    from .evaluation.objective import align_rows, case_utility
    from .utils import read_jsonl, write_jsonl

    aliases = context.state.payload.get("aliases") or {}
    selected_id = aliases.get("selected") or aliases.get("layerwise_best") or aliases.get("global_best")
    anchor_id = aliases.get("anchor")
    greedy_id = aliases.get("greedy")
    if not selected_id or not anchor_id or not greedy_id:
        return None
    protocol = str(context.config.get("evaluation.selection_protocol"))

    def rows(candidate_id: str) -> list[dict[str, Any]]:
        return read_jsonl(
            context.path("evaluations", "dev", protocol, candidate_id, "results.jsonl")
        )

    selected_rows = rows(str(selected_id))
    anchor_rows = rows(str(anchor_id))
    greedy_rows = rows(str(greedy_id))
    selected_anchor, anchor_aligned = align_rows(selected_rows, anchor_rows)
    selected_greedy, greedy_aligned = align_rows(selected_rows, greedy_rows)
    objective = context.config.get("objective")
    packet: list[dict[str, Any]] = []
    for selected, anchor, greedy in zip(
        selected_anchor, anchor_aligned, greedy_aligned
    ):
        selected_utility = case_utility(selected, objective)
        anchor_utility = case_utility(anchor, objective)
        greedy_utility = case_utility(greedy, objective)
        packet.append(
            {
                "id": selected["id"],
                "category": selected.get("category"),
                "messages": selected.get("messages"),
                "reference": selected.get("reference"),
                "anchor_reply": anchor.get("reply"),
                "greedy_reply": greedy.get("reply"),
                "selected_reply": selected.get("reply"),
                "anchor_quality": anchor.get("quality_score"),
                "greedy_quality": greedy.get("quality_score"),
                "selected_quality": selected.get("quality_score"),
                "selected_minus_anchor_utility": selected_utility - anchor_utility,
                "selected_minus_greedy_utility": selected_utility - greedy_utility,
                "endpoint_reply_disagreement": str(anchor.get("reply", "")).strip()
                != str(greedy.get("reply", "")).strip(),
                "review_priority": max(
                    abs(selected_utility - anchor_utility),
                    abs(selected_utility - greedy_utility),
                ),
            }
        )
    packet.sort(key=lambda row: float(row["review_priority"]), reverse=True)
    limit = max(1, int(context.config.get("reporting.top_case_examples", 20)))
    selected_packet = packet[:limit]
    jsonl_path = context.path("reports", "HUMAN_REVIEW_PACKET.jsonl")
    write_jsonl(jsonl_path, selected_packet)
    lines = [
        "# Human Review Packet",
        "",
        "These development cases changed most between the frozen/leading candidate and the two endpoints. Automated scores are evidence, not a substitute for review.",
        "",
    ]
    for index, row in enumerate(selected_packet, 1):
        user_message = ""
        for message in reversed(row.get("messages") or []):
            if message.get("role") == "user":
                user_message = str(message.get("content", ""))
                break
        lines.extend(
            [
                f"## {index}. `{row['id']}` — {row.get('category', 'uncategorized')}",
                "",
                f"**User:** {user_message}",
                "",
                f"**Anchor:** {row.get('anchor_reply')}",
                "",
                f"**Greedy:** {row.get('greedy_reply')}",
                "",
                f"**Selected:** {row.get('selected_reply')}",
                "",
                f"Selected − Anchor utility: `{_fmt(row.get('selected_minus_anchor_utility'))}`  ",
                f"Selected − Greedy utility: `{_fmt(row.get('selected_minus_greedy_utility'))}`",
                "",
            ]
        )
    markdown_path = context.path("reports", "HUMAN_REVIEW_PACKET.md")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return jsonl_path, markdown_path
