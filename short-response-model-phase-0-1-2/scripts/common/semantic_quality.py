from __future__ import annotations

"""Deprecated compatibility shim for Evaluator v4.

The former semantic-overlap evaluator was removed because it produced
subjective catch-all verdicts. New code must use ``observable_quality`` and
explicit response contracts. This module retains only a minimal import target
for external extensions; it never computes relevance, intent, or word-salad
judgments.
"""

from typing import Any

from scripts.common.observable_quality import detect_observable_defects, words
from scripts.common.quality_evaluation import QUALITY_EVALUATION_VERSION


def content_concepts(text: str) -> list[str]:
    """Return visible word tokens for legacy callers; no semantic judgment."""

    return [token for token in words(text) if len(token) > 1]


def semantic_integrity_analysis(
    record: dict[str, Any],
    reply: str,
    *,
    reference: str | None,
    corpus_profile: dict[str, Any] | None = None,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return observable mechanical evidence under a deprecated function name."""

    detected = detect_observable_defects(reply)
    critical = sorted({item["label"] for item in detected["defects"] if item["severity"] == "critical"})
    major = sorted({item["label"] for item in detected["defects"] if item["severity"] == "major"})
    minor = sorted({item["label"] for item in detected["defects"] if item["severity"] == "minor"})
    return {
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "deprecated": True,
        "replacement_api": "scripts.common.observable_quality.detect_observable_defects",
        "mechanical_integrity_score": float(detected["mechanical_integrity_score"]),
        "critical": critical,
        "major": major,
        "minor": minor,
        "review_diagnostics": detected["review_diagnostics"],
        "metrics": detected["metrics"],
    }
