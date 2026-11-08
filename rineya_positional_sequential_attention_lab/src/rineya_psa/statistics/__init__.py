"""Clustered paired inference and artifact provenance (SPEC sections 7 and 9)."""

from rineya_psa.statistics.bootstrap import (
    DEFAULT_N_RESAMPLES,
    DEFAULT_SEED,
    BootstrapResult,
    InteractionContrastResult,
    McNemarResult,
    PairedBootstrapResult,
    bootstrap_ci_of_statistic,
    interaction_contrast,
    mcnemar_exact,
    paired_bootstrap,
    summarize_paired_binary,
)
from rineya_psa.statistics.provenance import (
    canonical_json,
    config_hash,
    git_state,
    package_versions,
    stamp,
    write_result,
)

__all__ = [
    "DEFAULT_N_RESAMPLES",
    "DEFAULT_SEED",
    "BootstrapResult",
    "InteractionContrastResult",
    "McNemarResult",
    "PairedBootstrapResult",
    "bootstrap_ci_of_statistic",
    "canonical_json",
    "config_hash",
    "git_state",
    "interaction_contrast",
    "mcnemar_exact",
    "package_versions",
    "paired_bootstrap",
    "stamp",
    "summarize_paired_binary",
    "write_result",
]
