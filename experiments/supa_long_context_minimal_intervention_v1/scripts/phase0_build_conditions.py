#!/usr/bin/env python3
"""Phase 0, step 1: build and cache the four context conditions. CPU only.

Reads the frozen Rineya Long-Context Demonstration Suite and emits one cached
row per (example, condition). No model is loaded here, so the expensive GPU
stage never re-does this work and the selector can be audited on its own.

    python scripts/phase0_build_conditions.py [--smoke]

Outputs ``results/phase0_conditions/conditions.jsonl`` plus a selector report.
"""

from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from transformers import AutoTokenizer  # noqa: E402

from slcmi import context_selection as cs  # noqa: E402
from slcmi.evaluation import USER_TEMPLATE, count_tokens_factory  # noqa: E402
from slcmi.provenance import (  # noqa: E402
    CANONICAL_CORE,
    DEMO_SUITE,
    Run,
    read_jsonl,
    sha256,
    write_json,
    write_jsonl,
)
from slcmi.suites import SUITES  # noqa: E402

RANDOM_SEED_BASE = 20260905
SMOKE_SIZE = 32


def build(rows: list[dict], count_tokens, oracle_mode: str = "lexical") -> tuple[list[dict], list[dict]]:
    """Return (condition rows, per-example selector diagnostics)."""
    conditions: list[dict] = []
    diagnostics: list[dict] = []

    for index, row in enumerate(rows):
        context = row["context"]
        question = row["question"]
        chunks = cs.chunk_context(context, count_tokens)
        anchor = cs.anchor_text(context)

        full_tokens = count_tokens(context)
        query = cs.select_query(chunks, question, anchor, count_tokens=count_tokens)
        random_sel = cs.select_random(
            chunks, query.n_chunks, RANDOM_SEED_BASE + index, anchor, count_tokens=count_tokens
        )
        oracle = cs.select_oracle(
            chunks,
            row["supporting_facts"],
            row["answerable"],
            anchor,
            count_tokens=count_tokens,
            context=context,
            mode=oracle_mode,
        )

        shared = {
            "example_id": row["id"],
            "scenario_id": row["scenario_id"],
            "split": row["split"],
            "variant": row["variant"],
            "task_type": row["task_type"],
            "task_family": f"{row['task_type']}/{row['variant']}",
            "difficulty": row.get("difficulty", "unknown"),
            "needle_profile": row["needle_profile"],
            "length_bucket": row["length_bucket"],
            "answerable": row["answerable"],
            "question": question,
            "accepted_answers": row["accepted_answers"],
            "gold_answer": row["answer"],
            "original_context_tokens": full_tokens,
            "n_chunks_available": len(chunks),
        }

        emitted = {
            "A_full": (context, list(range(len(chunks))), full_tokens, {}),
            "B_query": (query.text, query.chunk_ids, query.n_tokens, query.diagnostics),
            "C_random": (
                random_sel.text,
                random_sel.chunk_ids,
                random_sel.n_tokens,
                random_sel.diagnostics,
            ),
        }
        if oracle is not None:
            emitted["D_oracle"] = (
                oracle.text,
                oracle.chunk_ids,
                oracle.n_tokens,
                oracle.diagnostics,
            )

        for condition, (text, ids, n_tokens, diag) in emitted.items():
            conditions.append(
                {
                    **shared,
                    "condition": condition,
                    "context": text,
                    "selected_chunk_ids": ids,
                    "selected_context_tokens": n_tokens,
                    "retention_ratio": round(n_tokens / full_tokens, 4),
                    "selector_diagnostics": diag,
                }
            )

        diagnostics.append(
            {
                "example_id": row["id"],
                "answerable": row["answerable"],
                "n_chunks": len(chunks),
                "chunk_tokens": [c.n_tokens for c in chunks],
                "anchor_tokens": count_tokens(anchor),
                "full_tokens": full_tokens,
                "query_chunk_ids": query.chunk_ids,
                "query_tokens": query.n_tokens,
                "random_chunk_ids": random_sel.chunk_ids,
                "random_tokens": random_sel.n_tokens,
                "oracle_available": oracle is not None,
                "oracle_chunk_ids": oracle.chunk_ids if oracle else None,
                "oracle_tokens": oracle.n_tokens if oracle else None,
                "oracle_localisation_confidence": (
                    oracle.diagnostics["localisation_confidence"] if oracle else None
                ),
                "oracle_localisation_mode": (
                    oracle.diagnostics["localisation_mode"] if oracle else None
                ),
                "query_oracle_jaccard": (
                    round(
                        len(set(query.chunk_ids) & set(oracle.chunk_ids))
                        / len(set(query.chunk_ids) | set(oracle.chunk_ids)),
                        4,
                    )
                    if oracle
                    else None
                ),
                "bm25_scores": query.diagnostics.get("bm25_scores"),
                "degenerate_query": query.diagnostics.get("degenerate_query", False),
            }
        )
    return conditions, diagnostics


def report(diagnostics: list[dict], rows: list[dict]) -> dict:
    def stats(values):
        values = [v for v in values if v is not None]
        if not values:
            return None
        return {
            "min": min(values),
            "median": round(statistics.median(values), 2),
            "mean": round(statistics.mean(values), 2),
            "max": max(values),
        }

    answerable = [d for d in diagnostics if d["answerable"]]
    with_oracle = [d for d in diagnostics if d["oracle_available"]]
    full_mean = statistics.mean(d["full_tokens"] for d in diagnostics)

    return {
        "n_examples": len(diagnostics),
        "selector_config": {
            "chunk_target_tokens": cs.CHUNK_TARGET_TOKENS,
            "select_budget_tokens": cs.SELECT_BUDGET_TOKENS,
            "neighbour_radius": cs.NEIGHBOUR_RADIUS,
            "hard_ceiling_tokens": cs.HARD_CEILING_TOKENS,
            "anchor": f"first {cs.ANCHOR_SENTENCES} sentence(s), applied to B/C/D alike",
            "scorer": "Okapi BM25, k1=%.1f b=%.2f, IDF over this document's chunks"
            % (cs.BM25_K1, cs.BM25_B),
            "oracle_min_confidence": cs.ORACLE_MIN_CONFIDENCE,
            "random_seed_base": RANDOM_SEED_BASE,
        },
        "chunks_per_document": stats([d["n_chunks"] for d in diagnostics]),
        "anchor_tokens": stats([d["anchor_tokens"] for d in diagnostics]),
        "tokens": {
            "A_full": stats([d["full_tokens"] for d in diagnostics]),
            "B_query": stats([d["query_tokens"] for d in diagnostics]),
            "C_random": stats([d["random_tokens"] for d in diagnostics]),
            "D_oracle": stats([d["oracle_tokens"] for d in with_oracle]),
        },
        "mean_retention_vs_full": {
            "B_query": round(
                statistics.mean(d["query_tokens"] for d in diagnostics) / full_mean, 4
            ),
            "C_random": round(
                statistics.mean(d["random_tokens"] for d in diagnostics) / full_mean, 4
            ),
            "D_oracle": (
                round(statistics.mean(d["oracle_tokens"] for d in with_oracle) / full_mean, 4)
                if with_oracle
                else None
            ),
        },
        "budget_match_B_over_C": round(
            statistics.mean(d["query_tokens"] for d in diagnostics)
            / statistics.mean(d["random_tokens"] for d in diagnostics),
            4,
        ),
        "oracle_availability": {
            "eligible": len(with_oracle),
            "answerable_rows": len(answerable),
            "all_rows": len(diagnostics),
            "share_of_answerable": round(len(with_oracle) / len(answerable), 4)
            if answerable
            else None,
            "unavailable_reason_unanswerable": len(diagnostics) - len(answerable),
            "unavailable_reason_localisation": len(answerable) - len(with_oracle),
            "note": (
                "Oracle is excluded for unanswerable rows by construction: a "
                "question with no determined answer has no supporting evidence "
                "to select. Rows are also excluded when a supporting fact fails "
                "to localise, rather than being approximated."
            ),
        },
        "query_vs_oracle_overlap": {
            "mean_jaccard": round(
                statistics.mean(d["query_oracle_jaccard"] for d in with_oracle), 4
            )
            if with_oracle
            else None,
            "identical_selections": sum(
                1
                for d in with_oracle
                if set(d["query_chunk_ids"]) == set(d["oracle_chunk_ids"])
            ),
            "note": (
                "Low overlap means D selects different chunks from B. Whether D "
                "is an upper bound is measured in the eval stage, not assumed "
                "here."
            ),
        },
        "degenerate_queries": sum(1 for d in diagnostics if d["degenerate_query"]),
        "source": {"rows": len(rows)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help=f"first {SMOKE_SIZE} rows only")
    parser.add_argument("--suite", default="demo", choices=sorted(SUITES))
    args = parser.parse_args()

    suite = SUITES[args.suite]
    stage = f"phase0_conditions_{suite.key}" if suite.key != "demo" else "phase0_conditions"
    if args.smoke:
        stage += "_smoke"

    rows = suite.loader(args.smoke)

    config = {
        "stage": stage,
        "suite": suite.key,
        "suite_role": suite.role,
        "suite_note": suite.note,
        "dataset": str(suite.path),
        "dataset_sha256": sha256(suite.path),
        "oracle_localisation_mode": suite.oracle_mode,
        "n_rows": len(rows),
        "prompt_template": USER_TEMPLATE,
        "selector": "frozen; see slcmi.context_selection module constants",
    }

    with Run(stage, config) as run:
        tokenizer = AutoTokenizer.from_pretrained(
            CANONICAL_CORE, trust_remote_code=True, local_files_only=True
        )
        count_tokens = count_tokens_factory(tokenizer)
        conditions, diagnostics = build(rows, count_tokens, suite.oracle_mode)
        summary = report(diagnostics, rows)
        summary["suite"] = {"key": suite.key, "role": suite.role, "note": suite.note,
                           "oracle_localisation_mode": suite.oracle_mode}

        write_jsonl(run.out / "conditions.jsonl", conditions, overwrite=True)
        write_jsonl(run.out / "selector_diagnostics.jsonl", diagnostics, overwrite=True)
        write_json(run.out / "selector_report.json", summary, overwrite=True)
        run.finish("COMPLETE", {"summary": summary})

    print(f"suite={suite.key} ({suite.role})  rows={len(rows)}  condition_rows={len(conditions)}")
    print(f"tokens: {summary['tokens']}")
    print(f"retention: {summary['mean_retention_vs_full']}")
    print(f"B/C budget match: {summary['budget_match_B_over_C']}")
    print(f"oracle: {summary['oracle_availability']}")
    print(f"query-vs-oracle overlap: {summary['query_vs_oracle_overlap']}")
    print(f"wrote {run.out}")


if __name__ == "__main__":
    main()
