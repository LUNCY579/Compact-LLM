#!/usr/bin/env python3
"""Emit an audit snapshot of every frozen parameter in this lab.

This is a SNAPSHOT, not an input. The scripts read their constants from the
``slcmi`` modules, so a config file that fed them would be a second source of
truth. This dumps the live values, with the module and attribute each came from,
so a reader can diff the report against the code that produced it.

    python scripts/emit_frozen_config.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from slcmi import adapters, context_selection as cs, evaluation, preservation, stats  # noqa: E402
from slcmi import training_data  # noqa: E402
from slcmi.provenance import CANONICAL_CORE, DEMO_SUITE, checkpoint_files, write_json  # noqa: E402
from slcmi.suites import LCR_DEV_SAMPLE, LCR_DEV_SEED, LCR_SFT  # noqa: E402


def main() -> None:
    snapshot = {
        "_note": (
            "Generated snapshot of live module constants. Not read by any script; "
            "the modules are the single source of truth."
        ),
        "selector (slcmi.context_selection)": {
            "CHUNK_TARGET_TOKENS": cs.CHUNK_TARGET_TOKENS,
            "CHUNK_OVERLAP_SENTENCES": cs.CHUNK_OVERLAP_SENTENCES,
            "SELECT_BUDGET_TOKENS": cs.SELECT_BUDGET_TOKENS,
            "NEIGHBOUR_RADIUS": cs.NEIGHBOUR_RADIUS,
            "HARD_CEILING_TOKENS": cs.HARD_CEILING_TOKENS,
            "USE_SENTENCE_ANCHOR": cs.USE_SENTENCE_ANCHOR,
            "ANCHOR_SENTENCES": cs.ANCHOR_SENTENCES,
            "BM25_K1": cs.BM25_K1,
            "BM25_B": cs.BM25_B,
            "ORACLE_MIN_CONFIDENCE": cs.ORACLE_MIN_CONFIDENCE,
            "n_stopwords": len(cs.STOPWORDS),
        },
        "decoding (slcmi.evaluation)": {
            "MAX_NEW_TOKENS": evaluation.MAX_NEW_TOKENS,
            "SEED": evaluation.SEED,
            "DTYPE": str(evaluation.DTYPE),
            "SYSTEM_PROMPT": evaluation.SYSTEM_PROMPT,
            "USER_TEMPLATE": evaluation.USER_TEMPLATE,
            "do_sample": False,
            "enable_thinking": False,
            "scorer": "Rineya_Long_Context_Demo_v1/scripts/evaluate_predictions.py::normalize",
        },
        "adapters (slcmi.adapters)": {
            "PROTECTED_LAYERS": list(adapters.PROTECTED_LAYERS),
            "DEFAULT_LAYERS": list(adapters.DEFAULT_LAYERS),
            "EXPANDED_LAYERS": list(adapters.EXPANDED_LAYERS),
            "ATTENTION_TARGETS": list(adapters.ATTENTION_TARGETS),
            "DEFAULT_RANK": adapters.DEFAULT_RANK,
            "DEFAULT_ALPHA": adapters.DEFAULT_ALPHA,
            "DEFAULT_DROPOUT": adapters.DEFAULT_DROPOUT,
        },
        "curriculum (slcmi.training_data)": {
            "LCR_SFT": str(LCR_SFT),
            "CURRICULUM_SEED": training_data.CURRICULUM_SEED,
            "UNKNOWN_SURFACE": training_data.UNKNOWN_SURFACE,
        },
        "preservation (slcmi.preservation)": {
            "PRESERVATION_CLASSES": list(preservation.PRESERVATION_CLASSES),
            "PRESERVATION_PER_CLASS": preservation.PRESERVATION_PER_CLASS,
            "PRESERVATION_SEED": preservation.PRESERVATION_SEED,
        },
        "statistics (slcmi.stats)": {
            "BOOTSTRAP_RESAMPLES": stats.BOOTSTRAP_RESAMPLES,
            "BOOTSTRAP_SEED": stats.BOOTSTRAP_SEED,
            "significance_claimed": False,
        },
        "suites": {
            "demo": str(DEMO_SUITE),
            "lcr_dev_source": str(LCR_SFT),
            "LCR_DEV_SAMPLE": LCR_DEV_SAMPLE,
            "LCR_DEV_SEED": LCR_DEV_SEED,
        },
        "canonical_checkpoint": {
            "path": str(CANONICAL_CORE),
            "sha256": checkpoint_files(),
        },
    }
    out = ROOT / "configs/frozen_parameters.json"
    write_json(out, snapshot, overwrite=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
