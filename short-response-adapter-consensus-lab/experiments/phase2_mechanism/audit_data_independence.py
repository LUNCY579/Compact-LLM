from __future__ import annotations

import hashlib
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors


LAB_ROOT = Path(__file__).resolve().parents[2]
OFFICIAL_ROOT = LAB_ROOT.parent / "short-response-model-phase-0-1-2"
SPLIT_ROOT = OFFICIAL_ROOT / "data/phase_9_student_distillation/multidomain_reasoning_v22/reviewed_splits"
RAW_RESULT = LAB_ROOT / "results/sealed/v22_frozen_finalists.jsonl"
CORNERSTONE = LAB_ROOT / "reports/phase2_cornerstone_verification.json"
OUTPUT = LAB_ROOT / "results/phase2_mechanism/data_independence_audit.json"
REPORT = LAB_ROOT / "reports/PHASE2_DATA_INDEPENDENCE.md"

CONDITIONS = {
    ("track_a_k1", "hard_plurality"): "track_a",
    ("full_k7", "hard_plurality"): "hard_k7",
    ("full_k7", "probability_mean"): "probability_k7",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def user_prompt(row: dict[str, Any]) -> str:
    messages = row.get("messages") or []
    return "\n".join(str(message.get("content") or "") for message in messages if message.get("role") == "user")


def exact_normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def template_normalized(text: str) -> str:
    value = exact_normalized(text)
    value = re.sub(r"\b[0-9a-f]{8,}\b", "<id>", value)
    value = re.sub(r"(?<![a-z])[-+]?\d+(?:\.\d+)?(?:%|[a-z]+)?", "<num>", value)
    value = re.sub(r"\b(?:mon|tue|wed|thu|fri|sat|sun)(?:day)?\b", "<day>", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def structural_shape(value: Any, *, key: str | None = None) -> Any:
    if isinstance(value, dict):
        if key in {"values", "mapping", "distances", "weights", "scores"}:
            child_shapes = sorted(json.dumps(structural_shape(item), sort_keys=True) for item in value.values())
            return {"dynamic_map_size": len(value), "value_shapes": child_shapes}
        return {name: structural_shape(item, key=str(name)) for name, item in sorted(value.items())}
    if isinstance(value, list):
        shapes = sorted({json.dumps(structural_shape(item), sort_keys=True) for item in value})
        return {"list_length": len(value), "item_shapes": shapes}
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def template_family(row: dict[str, Any]) -> str:
    packet = {
        "reasoning_family": row.get("reasoning_family"),
        "decision": row.get("decision"),
        "difficulty": row.get("difficulty"),
        "verification_shape": structural_shape(row.get("verification") or {}),
    }
    return hashlib.sha256(
        json.dumps(packet, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:16]


def duplicate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    exact_prompts = Counter(exact_normalized(user_prompt(row)) for row in rows)
    template_prompts = Counter(template_normalized(user_prompt(row)) for row in rows)
    ids = Counter(str(row["id"]) for row in rows)
    return {
        "rows": len(rows),
        "unique_ids": len(ids),
        "duplicate_id_rows": sum(count - 1 for count in ids.values()),
        "unique_exact_normalized_prompts": len(exact_prompts),
        "duplicate_exact_prompt_rows": sum(count - 1 for count in exact_prompts.values()),
        "unique_value_normalized_prompts": len(template_prompts),
        "duplicate_value_normalized_prompt_rows": sum(count - 1 for count in template_prompts.values()),
        "largest_value_normalized_prompt_cluster": max(template_prompts.values()),
    }


def nearest_similarity(
    train_dev: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> dict[str, Any]:
    train_dev_text = [template_normalized(user_prompt(row)) for row in train_dev]
    test_text = [template_normalized(user_prompt(row)) for row in test]
    vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    all_matrix = vectorizer.fit_transform([*train_dev_text, *test_text])
    train_dev_matrix = all_matrix[: len(train_dev_text)]
    test_matrix = all_matrix[len(train_dev_text) :]
    cross_model = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute", n_jobs=-1)
    cross_model.fit(train_dev_matrix)
    cross_distances, cross_indices = cross_model.kneighbors(test_matrix)
    within_model = NearestNeighbors(n_neighbors=2, metric="cosine", algorithm="brute", n_jobs=-1)
    within_model.fit(test_matrix)
    within_distances, within_indices = within_model.kneighbors(test_matrix)
    cross_similarities = 1.0 - cross_distances[:, 0]
    within_similarities = 1.0 - within_distances[:, 1]
    thresholds = (0.80, 0.90, 0.95, 0.99, 0.999999)
    cross_examples = []
    for test_index in np.argsort(-cross_similarities)[:20]:
        neighbor_index = int(cross_indices[test_index, 0])
        cross_examples.append(
            {
                "test_id": str(test[test_index]["id"]),
                "neighbor_id": str(train_dev[neighbor_index]["id"]),
                "neighbor_split": str(train_dev[neighbor_index]["intended_split"]),
                "similarity": float(cross_similarities[test_index]),
            }
        )
    within_examples = []
    for test_index in np.argsort(-within_similarities)[:20]:
        neighbor_index = int(within_indices[test_index, 1])
        within_examples.append(
            {
                "test_id": str(test[test_index]["id"]),
                "neighbor_id": str(test[neighbor_index]["id"]),
                "similarity": float(within_similarities[test_index]),
            }
        )
    return {
        "vectorizer": "word TF-IDF, 1-2 grams, min_df=2, value-normalized prompts",
        "feature_count": len(vectorizer.vocabulary_),
        "cross_split_test_to_train_dev": {
            "mean": float(cross_similarities.mean()),
            "median": float(np.median(cross_similarities)),
            "p90": float(np.quantile(cross_similarities, 0.90)),
            "p95": float(np.quantile(cross_similarities, 0.95)),
            "p99": float(np.quantile(cross_similarities, 0.99)),
            "counts_at_threshold": {
                str(value): int(np.sum(cross_similarities >= value)) for value in thresholds
            },
            "top_neighbors": cross_examples,
        },
        "within_test_nearest_other": {
            "mean": float(within_similarities.mean()),
            "median": float(np.median(within_similarities)),
            "p90": float(np.quantile(within_similarities, 0.90)),
            "p95": float(np.quantile(within_similarities, 0.95)),
            "p99": float(np.quantile(within_similarities, 0.99)),
            "counts_at_threshold": {
                str(value): int(np.sum(within_similarities >= value)) for value in thresholds
            },
            "top_neighbors": within_examples,
        },
    }


def load_correctness() -> dict[str, dict[str, bool]]:
    conditions: dict[str, dict[str, bool]] = defaultdict(dict)
    for row in iter_jsonl(RAW_RESULT):
        alias = CONDITIONS.get((str(row.get("committee")), str(row.get("method"))))
        if alias:
            conditions[alias][str(row["example_id"])] = bool(row["correct"])
    return conditions


def cluster_bootstrap(
    test_rows: list[dict[str, Any]],
    correctness: dict[str, dict[str, bool]],
    *,
    cluster_name: str,
    cluster_fn: Any,
    candidate: str,
    baseline: str,
    samples: int = 100_000,
) -> dict[str, Any]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in test_rows:
        grouped[str(cluster_fn(row))].append(str(row["id"]))
    clusters = sorted(grouped)
    sums = np.asarray(
        [
            sum(
                int(correctness[candidate][example_id]) - int(correctness[baseline][example_id])
                for example_id in grouped[cluster]
            )
            for cluster in clusters
        ],
        dtype=np.float64,
    )
    sizes = np.asarray([len(grouped[cluster]) for cluster in clusters], dtype=np.float64)
    rng = np.random.default_rng(
        int.from_bytes(hashlib.sha256(f"{cluster_name}:{candidate}:{baseline}".encode()).digest()[:8], "big")
    )
    batch_size = 10_000
    boot = []
    for start in range(0, samples, batch_size):
        count = min(batch_size, samples - start)
        indices = rng.integers(0, len(clusters), size=(count, len(clusters)))
        boot.append(sums[indices].sum(axis=1) / sizes[indices].sum(axis=1))
    values = np.concatenate(boot)
    low, high = np.quantile(values, [0.025, 0.975])
    family_effects = [
        {
            "cluster": cluster,
            "n": len(grouped[cluster]),
            "difference": float(sums[index] / sizes[index]),
        }
        for index, cluster in enumerate(clusters)
    ]
    return {
        "cluster_definition": cluster_name,
        "candidate": candidate,
        "baseline": baseline,
        "cluster_count": len(clusters),
        "min_cluster_size": int(sizes.min()),
        "max_cluster_size": int(sizes.max()),
        "micro_difference": float(sums.sum() / sizes.sum()),
        "macro_cluster_difference": statistics.mean(item["difference"] for item in family_effects),
        "cluster_bootstrap_ci_low": float(low),
        "cluster_bootstrap_ci_high": float(high),
        "bootstrap_samples": samples,
        "cluster_effects": family_effects,
    }


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def pp(value: float) -> str:
    return f"{100 * value:+.2f} pp"


def make_report(packet: dict[str, Any]) -> None:
    lines = [
        "# Phase 2 Data Independence and Template-Family Audit",
        "",
        "**Status: complete descriptive sensitivity audit.**",
        "",
        "The dataset contains unique IDs and no duplicate result keys, but deterministic generation creates repeated task structures. This report distinguishes row uniqueness from statistical independence and adds cluster-bootstrap sensitivity analyses. It does not estimate a single definitive effective sample size.",
        "",
        "## Exact and normalized duplication",
        "",
        "| Split | Rows | Unique IDs | Exact normalized prompts | Duplicate exact-prompt rows | Value-normalized prompt forms | Duplicate value-normalized rows | Largest normalized cluster |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("train", "dev", "test"):
        row = packet["duplicates"][split]
        lines.append(
            f"| {split.upper()} | {row['rows']:,} | {row['unique_ids']:,} | {row['unique_exact_normalized_prompts']:,} | "
            f"{row['duplicate_exact_prompt_rows']:,} | {row['unique_value_normalized_prompts']:,} | "
            f"{row['duplicate_value_normalized_prompt_rows']:,} | {row['largest_value_normalized_prompt_cluster']:,} |"
        )
    lines.extend(
        [
            "",
            f"Exact normalized prompt overlap across splits: **{packet['split_overlap']['exact_normalized_prompt_overlap_count']}** forms. Value-normalized template overlap across all three splits: **{packet['split_overlap']['value_normalized_prompt_overlap_count']}** forms.",
            "",
            "## Lexical nearest-neighbor similarity",
            "",
            "Prompts are lowercased and numeric/scenario values are replaced before word 1–2 gram TF-IDF similarity. This intentionally detects structural lexical reuse and is not semantic equivalence.",
            "",
            "| Search | Mean | Median | P90 | P95 | P99 | ≥0.95 | ≥0.99 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for label, key in (
        ("TEST → TRAIN/DEV", "cross_split_test_to_train_dev"),
        ("TEST → other TEST", "within_test_nearest_other"),
    ):
        row = packet["nearest_similarity"][key]
        lines.append(
            f"| {label} | {row['mean']:.3f} | {row['median']:.3f} | {row['p90']:.3f} | {row['p95']:.3f} | "
            f"{row['p99']:.3f} | {row['counts_at_threshold']['0.95']:,} | {row['counts_at_threshold']['0.99']:,} |"
        )
    lines.extend(
        [
            "",
            "## Paired effect sensitivity to clustered resampling",
            "",
            "The original example bootstrap samples rows. The sensitivity intervals below resample entire clusters with replacement and retain all examples within a sampled cluster.",
            "",
            "| Comparison | Cluster unit | Clusters | Size range | Micro difference | Macro-cluster difference | 95% cluster-bootstrap CI |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in packet["cluster_bootstrap"]:
        lines.append(
            f"| {row['candidate']} − {row['baseline']} | {row['cluster_definition']} | {row['cluster_count']} | "
            f"{row['min_cluster_size']}–{row['max_cluster_size']} | {pp(row['micro_difference'])} | "
            f"{pp(row['macro_cluster_difference'])} | [{pp(row['cluster_bootstrap_ci_low'])}, {pp(row['cluster_bootstrap_ci_high'])}] |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Row-level uniqueness is verified; independence is not.",
            "- Numeric/value-normalized prompts reveal repeated lexical structures by design.",
            "- Reasoning-family and structural-template cluster intervals quantify how sensitive the paired headline is to treating task families, rather than examples, as the resampling unit.",
            "- A cluster interval containing zero does not negate the observed finite-corpus difference; it limits generalization to a population of unseen task/template families.",
            "- Structural signatures are audit heuristics, not ground-truth latent-task labels. The appendix should report results under multiple plausible cluster definitions.",
            "",
            "## Provenance",
            "",
            f"- Raw result SHA256: `{packet['provenance']['raw_sha256']}`",
            f"- TEST dataset SHA256: `{packet['provenance']['test_sha256']}`",
            f"- Analysis script SHA256: `{packet['provenance']['script_sha256']}`",
        ]
    )
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    if read_json(CORNERSTONE).get("status") != "PASS":
        raise RuntimeError("Cornerstone verification must pass first")
    splits = {
        split: list(iter_jsonl(SPLIT_ROOT / f"{split}.reviewed.jsonl"))
        for split in ("train", "dev", "test")
    }
    duplicates = {split: duplicate_summary(rows) for split, rows in splits.items()}
    exact_sets = {
        split: {exact_normalized(user_prompt(row)) for row in rows} for split, rows in splits.items()
    }
    template_sets = {
        split: {template_normalized(user_prompt(row)) for row in rows} for split, rows in splits.items()
    }
    split_overlap = {
        "exact_normalized_prompt_overlap_count": len(
            exact_sets["train"] & exact_sets["dev"]
            | exact_sets["train"] & exact_sets["test"]
            | exact_sets["dev"] & exact_sets["test"]
        ),
        "value_normalized_prompt_overlap_count": len(
            template_sets["train"] & template_sets["dev"] & template_sets["test"]
        ),
        "train_test_value_normalized_overlap": len(template_sets["train"] & template_sets["test"]),
        "dev_test_value_normalized_overlap": len(template_sets["dev"] & template_sets["test"]),
    }
    nearest = nearest_similarity([*splits["train"], *splits["dev"]], splits["test"])
    correctness = load_correctness()
    cluster_specs = [
        ("reasoning_family", lambda row: row.get("reasoning_family")),
        ("domain", lambda row: row.get("domain")),
        ("reasoning×difficulty×decision×verification-shape", template_family),
    ]
    cluster_results = []
    for cluster_name, cluster_fn in cluster_specs:
        for candidate, baseline in (("hard_k7", "track_a"), ("hard_k7", "probability_k7")):
            cluster_results.append(
                cluster_bootstrap(
                    splits["test"],
                    correctness,
                    cluster_name=cluster_name,
                    cluster_fn=cluster_fn,
                    candidate=candidate,
                    baseline=baseline,
                )
            )
    packet = {
        "schema_version": "1.0",
        "status": "DATA_INDEPENDENCE_SENSITIVITY_COMPLETE",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "analysis_role": "TEST DESCRIPTIVE / SENSITIVITY",
        "duplicates": duplicates,
        "split_overlap": split_overlap,
        "nearest_similarity": nearest,
        "cluster_bootstrap": cluster_results,
        "provenance": {
            "raw_sha256": sha256_file(RAW_RESULT),
            "train_sha256": sha256_file(SPLIT_ROOT / "train.reviewed.jsonl"),
            "dev_sha256": sha256_file(SPLIT_ROOT / "dev.reviewed.jsonl"),
            "test_sha256": sha256_file(SPLIT_ROOT / "test.reviewed.jsonl"),
            "script_sha256": sha256_file(Path(__file__).resolve()),
        },
        "warnings": [
            "Template normalization and structural signatures are heuristic.",
            "TF-IDF similarity measures lexical structure, not semantic task identity.",
            "Cluster bootstrap targets generalization across the chosen clusters, not the finite-corpus paired difference.",
        ],
    }
    OUTPUT.write_text(json.dumps(packet, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    make_report(packet)
    print(
        json.dumps(
            {
                "status": packet["status"],
                "duplicates": duplicates,
                "split_overlap": split_overlap,
                "cluster_bootstrap": [
                    {
                        key: row[key]
                        for key in (
                            "candidate",
                            "baseline",
                            "cluster_definition",
                            "cluster_count",
                            "micro_difference",
                            "cluster_bootstrap_ci_low",
                            "cluster_bootstrap_ci_high",
                        )
                    }
                    for row in cluster_results
                ],
                "output": str(OUTPUT),
                "report": str(REPORT),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
