"""Scientific scope identity, and the within-study qualification gate.

`TGSA_MTP_CANONICAL_PATH.md` names five identifiers as hard evidence-integrity
fields — ``task_scope_sha256``, ``model_scope_sha256``, ``objective_scope_sha256``,
``competence_scope_sha256``, ``environment_lock_sha256`` — and says a missing or
mismatched one "is an evidence-integrity failure, not a documentation warning".
It also draws a line the code did not draw:

    An external Phase-1 report may never replace within-study qualification when
    Phase 2 changes the generator, families, difficulty band, depth, objective,
    tokenizer, split, or metric.

Before this module those identifiers appeared in that markdown file and nowhere
in any Python. The only competence gate was ``require_go(external_report)``,
which checks that a report exists, parses, is not development-only, and says GO.
It cannot check *what the report is about*. A Phase-1 `GO` earned on one band
would therefore certify a Phase-2 study on a different generator, a different
depth, or a different objective, and nothing downstream could tell — because the
gate reads the verdict, not the scope.

That is precisely how Phase 1 v4 failed. Its regime was rejected because
``pointer_chase`` and ``direct_lookup`` were inert and ``state_update``
saturated: the families did not move on the scope actually being measured. A
scope-blind gate cannot detect that condition in a *later* study, however
rigorous the earlier one was.

So scope is hashed, carried, and compared. A study whose scope differs from the
report certifying it is refused.

Nothing here evaluates competence. It answers only: *is this report about this
study?*
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCOPE_SCHEMA_VERSION = "tgsa.scientific-scope.v1"

#: Every identifier the canonical path requires. Recorded together so a caller
#: cannot quietly emit a subset.
SCOPE_FIELDS: tuple[str, ...] = (
    "task_scope_sha256",
    "model_scope_sha256",
    "objective_scope_sha256",
    "competence_scope_sha256",
    "environment_lock_sha256",
)


class ScopeMismatchError(ValueError):
    """A report and a study describe different science."""


def canonical_json_bytes(value: object) -> bytes:
    """Byte-stable JSON. Matches tgsa.phase2_integrity so digests are comparable."""

    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _require(mapping: Mapping[str, object], key: str, where: str) -> object:
    if key not in mapping:
        raise ScopeMismatchError(
            f"{where} is missing {key!r}; scope identity cannot be computed from an "
            f"incomplete description, and guessing a default would defeat the point"
        )
    return mapping[key]


# ---------------------------------------------------------------------------
# The four scientific scopes
# ---------------------------------------------------------------------------


def task_scope(dataset: Mapping[str, object]) -> dict[str, object]:
    """What is being asked. Changing any of it changes what competence means."""

    families = _require(dataset, "families", "dataset")
    if not isinstance(families, Mapping):
        raise ScopeMismatchError("dataset.families must be an object")
    vocabulary = _require(dataset, "vocabulary", "dataset")
    splits = dataset.get("splits")
    return {
        "schema_version": SCOPE_SCHEMA_VERSION,
        # Family NAMES and their generator parameters both matter: same names
        # with different hop ranges is a different task.
        "families": {str(k): families[k] for k in sorted(families)},
        "vocabulary": dict(sorted(dict(vocabulary).items())),  # type: ignore[arg-type]
        # Which split is the unbiased draw. v4 shipped a band whose test split
        # was whatever training had not exhausted; that is a different task from
        # one whose test split is representative.
        "split_generation_order": dataset.get(
            "split_generation_order", ["train", "dev", "test"]
        ),
        "split_sizes": (
            {k: dict(sorted(dict(v).items())) for k, v in sorted(dict(splits).items())}
            if isinstance(splits, Mapping) else None
        ),
    }


def model_scope(model: Mapping[str, object]) -> dict[str, object]:
    """Architecture and capacity.

    Depth is here because v4 never ran its wider rung and could not tell whether
    the inert families were a task problem or a capacity problem. A competence
    report earned at one depth says nothing about another.
    """

    keys = (
        "embed_dim", "num_heads", "num_layers", "ffn_multiplier",
        "trajectory_rank", "tie_embeddings", "bias", "dropout",
        "max_sequence_length",
    )
    return {
        "schema_version": SCOPE_SCHEMA_VERSION,
        **{key: model.get(key) for key in keys},
    }


def compared_conditions(study: Mapping[str, object]) -> list[str]:
    """The conditions a study contrasts, however it happens to declare them.

    Phase 1 declares ``arms``; Phase 2 declares ``cells`` of (tgsa_arm,
    mtp_mode). Both are "what is being compared", and both belong in the
    objective scope, so this normalizes them to one sorted list of labels rather
    than imposing one phase's shape on the other.
    """

    arms = study.get("arms")
    if isinstance(arms, Sequence) and not isinstance(arms, (str, bytes)):
        return sorted(str(arm) for arm in arms)

    cells = study.get("cells")
    if isinstance(cells, Sequence) and not isinstance(cells, (str, bytes)):
        labels: list[str] = []
        for cell in cells:
            if isinstance(cell, Mapping):
                labels.append(
                    "/".join(f"{k}={cell[k]}" for k in sorted(cell))
                )
            else:
                labels.append(str(cell))
        return sorted(labels)

    raise ScopeMismatchError(
        "study declares neither `arms` nor `cells`, so there is nothing to say "
        "about what it compares; an objective scope cannot be computed"
    )


def objective_scope(
    training: Mapping[str, object],
    *,
    conditions: Sequence[str],
    auxiliary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """What is optimized, and which conditions are compared.

    The auxiliary block is where an MTP head enters. Adding a future-token loss
    changes the objective, so a competence report earned without it does not
    transfer to a study with it — which is the specific substitution the
    canonical path forbids.
    """

    keys = ("learning_rate", "batch_size", "beta1", "beta2", "weight_decay",
            "gradient_clip_norm", "epsilon")
    return {
        "schema_version": SCOPE_SCHEMA_VERSION,
        "conditions": sorted(str(c) for c in conditions),
        "optimizer": {key: training.get(key) for key in keys},
        "auxiliary_objective": dict(sorted(dict(auxiliary).items())) if auxiliary else None,
    }


def competence_scope(analysis: Mapping[str, object]) -> dict[str, object]:
    """The metric and the decision rule the verdict was reached under.

    A GO under one estimator, gate, or set of primary families is not a GO under
    another.
    """

    keys = (
        "primary_families", "guardrail_family", "interval_estimator",
        "primary_gate", "confidence",
        "minimum_history_delta", "minimum_cascade_delta", "minimum_alignment_drop",
        "direct_lookup_noninferiority_margin",
    )
    return {
        "schema_version": SCOPE_SCHEMA_VERSION,
        **{key: analysis.get(key) for key in keys},
    }


def environment_lock(device: Mapping[str, object] | None = None) -> dict[str, object]:
    """The execution environment, to the granularity that changes arithmetic.

    Recorded rather than pinned: the point is that a mismatch is *visible*, not
    that one machine is blessed.
    """

    payload: dict[str, object] = {
        "schema_version": SCOPE_SCHEMA_VERSION,
        "python": ".".join(str(part) for part in sys.version_info[:3]),
        "platform": platform.system(),
    }
    try:  # torch is a runtime dependency, but scope must never be fatal to import
        import torch

        payload["torch"] = torch.__version__
    except Exception as error:  # noqa: BLE001
        payload["torch"] = f"unavailable: {type(error).__name__}"
    if device:
        payload["device"] = dict(sorted(dict(device).items()))
    return payload


# ---------------------------------------------------------------------------
# Assembly and comparison
# ---------------------------------------------------------------------------


def compute_scope(
    study: Mapping[str, object],
    *,
    auxiliary: Mapping[str, object] | None = None,
    device: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Every scope digest for one study configuration, plus the payloads."""

    dataset = _require(study, "dataset", "study")
    model = _require(study, "model", "study")
    training = _require(study, "training", "study")
    analysis = study.get("analysis") or {}
    conditions = compared_conditions(study)

    payloads = {
        "task": task_scope(dataset),  # type: ignore[arg-type]
        "model": model_scope(model),  # type: ignore[arg-type]
        "objective": objective_scope(training, conditions=conditions, auxiliary=auxiliary),  # type: ignore[arg-type]
        "competence": competence_scope(analysis),  # type: ignore[arg-type]
        "environment": environment_lock(device),
    }
    return {
        "schema_version": SCOPE_SCHEMA_VERSION,
        "task_scope_sha256": sha256_json(payloads["task"]),
        "model_scope_sha256": sha256_json(payloads["model"]),
        "objective_scope_sha256": sha256_json(payloads["objective"]),
        "competence_scope_sha256": sha256_json(payloads["competence"]),
        "environment_lock_sha256": sha256_json(payloads["environment"]),
        "payloads": payloads,
    }


def _digests(scope: Mapping[str, object], where: str) -> dict[str, str]:
    missing = [field for field in SCOPE_FIELDS if not scope.get(field)]
    if missing:
        raise ScopeMismatchError(
            f"{where} does not carry {missing}. The canonical path treats a missing "
            f"identifier as an evidence-integrity failure, not a documentation "
            f"warning, so it is refused rather than defaulted"
        )
    return {field: str(scope[field]) for field in SCOPE_FIELDS}


def require_scope_match(
    report: Mapping[str, object],
    study_scope: Mapping[str, object],
    *,
    compare: Sequence[str] = SCOPE_FIELDS,
    ignore_environment: bool = False,
) -> dict[str, object]:
    """Refuse a competence report that is not about this study.

    ``ignore_environment`` exists for the honest case where a report was earned
    on another machine and only the environment differs. It never relaxes the
    four scientific scopes, and the mismatch is still reported.
    """

    recorded = report.get("scope")
    if not isinstance(recorded, Mapping):
        raise ScopeMismatchError(
            "the competence report carries no `scope` block, so it cannot be shown "
            "to be about this study. Re-measure with a scope-recording runner; an "
            "external verdict alone is not a within-study qualification"
        )

    theirs = _digests(recorded, "the competence report's scope")
    ours = _digests(study_scope, "this study's scope")

    fields = [f for f in compare
              if not (ignore_environment and f == "environment_lock_sha256")]
    differing = [f for f in fields if theirs[f] != ours[f]]
    environment_differs = theirs["environment_lock_sha256"] != ours["environment_lock_sha256"]

    if differing:
        detail = ", ".join(
            f"{f}: report {theirs[f][:12]} != study {ours[f][:12]}" for f in differing
        )
        raise ScopeMismatchError(
            f"the competence report describes different science: {detail}. An external "
            f"Phase-1 report may not replace within-study qualification when the "
            f"generator, families, band, depth, objective, split, or metric change. "
            f"Re-measure competence on this study's scope"
        )

    return {
        "schema_version": SCOPE_SCHEMA_VERSION,
        "matched_fields": list(fields),
        "environment_differs": environment_differs,
        "environment_ignored": bool(ignore_environment),
        "scope": ours,
    }
