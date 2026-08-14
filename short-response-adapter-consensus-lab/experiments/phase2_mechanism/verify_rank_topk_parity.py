from __future__ import annotations

import hashlib
import itertools
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from consensus_lab.voting.rules import _stable_topk_indices  # noqa: E402


RULES = ROOT / "src/consensus_lab/voting/rules.py"
OUTPUT = ROOT / "results/phase2_mechanism/rank_topk_parity_verification.json"
REPORT = ROOT / "reports/PHASE2_RANK_TOPK_PARITY.md"
SEED = 20260812


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def full_stable_prefix(scores: torch.Tensor, k: int) -> torch.Tensor:
    return torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :k]


def require_equal(scores: torch.Tensor, k: int) -> None:
    expected = full_stable_prefix(scores, k)
    actual = _stable_topk_indices(scores, k)
    if not torch.equal(actual, expected):
        raise RuntimeError(
            f"Stable partial top-k mismatch for shape={tuple(scores.shape)}, k={k}: "
            f"expected={expected.tolist()} actual={actual.tolist()}"
        )


def main() -> None:
    exhaustive_vectors = list(itertools.product((-1.0, 0.0, 1.0), repeat=7))
    exhaustive = torch.tensor(exhaustive_vectors, dtype=torch.float32)
    exhaustive_checks = 0
    for k in range(1, 8):
        require_equal(exhaustive, k)
        exhaustive_checks += len(exhaustive_vectors)

    generator = torch.Generator().manual_seed(SEED)
    randomized_checks = 0
    for vocabulary_size in (8, 31, 257, 4096):
        scores = torch.randn(256, vocabulary_size, generator=generator)
        # Force score and boundary ties, including rows with every score equal.
        scores[:32, : min(12, vocabulary_size)] = 2.0
        scores[0].zero_()
        for k in sorted({1, min(2, vocabulary_size), min(5, vocabulary_size), vocabulary_size}):
            require_equal(scores, k)
            randomized_checks += int(scores.shape[0])

    realistic_vocabulary = 151_936
    realistic = torch.randn(7, realistic_vocabulary, generator=generator)
    realistic[:, 100:110] = 2.0
    require_equal(realistic, 5)
    full_sort_index_bytes = realistic.numel() * torch.tensor([], dtype=torch.int64).element_size()
    partial_result_index_bytes = 7 * 5 * torch.tensor([], dtype=torch.int64).element_size()

    report = "\n".join(
        [
            "# Rank top-k implementation parity",
            "",
            "Status: **RANK_TOPK_IMPLEMENTATION_PARITY_VERIFIED**",
            "",
            "The memory-safe stable partial top-k returns the exact prefix of the original "
            "stable full-vocabulary descending sort. The check covers every 7-token vector "
            "over a three-value grid at every rank depth, randomized/tie-stressed matrices, "
            "and the model's realistic 151,936-token vocabulary at K=7 experts and depth 5.",
            "Because rank points are a deterministic function of these identical per-expert "
            "rankings and the unchanged secondary score uses the same raw log probabilities, "
            "each accepted token is identical. Induction on the shared accepted prefix then "
            "gives an identical full trajectory.",
            "",
            f"- Exhaustive vector/rank checks: **{exhaustive_checks:,}**",
            f"- Randomized row/rank checks: **{randomized_checks:,}**",
            "- Realistic-vocabulary expert rows: **7**",
            f"- Original full-sort index allocation proxy: **{full_sort_index_bytes:,} bytes**",
            f"- Partial result index allocation: **{partial_result_index_bytes:,} bytes**",
            "",
            "The post-example CUDA cache release occurs outside generation timing and only "
            "returns unused allocator blocks; it cannot change logits or decoded outputs.",
            "",
        ]
    )
    REPORT.write_text(report, encoding="utf-8")
    packet = {
        "schema_version": "1.0",
        "status": "RANK_TOPK_IMPLEMENTATION_PARITY_VERIFIED",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "exhaustive": {
            "vocabulary_size": 7,
            "score_alphabet": [-1.0, 0.0, 1.0],
            "vectors": len(exhaustive_vectors),
            "rank_depths": list(range(1, 8)),
            "vector_rank_checks": exhaustive_checks,
        },
        "randomized": {
            "rows_per_vocabulary": 256,
            "vocabulary_sizes": [8, 31, 257, 4096],
            "row_rank_checks": randomized_checks,
            "forced_ties": True,
        },
        "realistic_vocabulary": {
            "expert_rows": 7,
            "vocabulary_size": realistic_vocabulary,
            "rank_depth": 5,
            "exact": True,
            "full_sort_index_bytes": full_sort_index_bytes,
            "partial_result_index_bytes": partial_result_index_bytes,
        },
        "equivalence": {
            "reference": "torch.argsort(scores, descending=True, stable=True)[:, :k]",
            "candidate": "consensus_lab.voting.rules._stable_topk_indices(scores, k)",
            "all_checks_exact": True,
            "downstream_consequence": (
                "identical per-expert stable rankings imply identical rank score totals, "
                "secondary tie-breaks, accepted tokens, and autoregressive trajectories"
            ),
        },
        "provenance": {
            "verification_script": str(Path(__file__).resolve()),
            "verification_script_sha256": sha256(Path(__file__).resolve()),
            "voting_rules": str(RULES),
            "voting_rules_sha256": sha256(RULES),
            "report": str(REPORT),
            "report_sha256": sha256(REPORT),
        },
    }
    OUTPUT.write_text(json.dumps(packet, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": packet["status"], "output": str(OUTPUT), "report": str(REPORT)}, indent=2))


if __name__ == "__main__":
    main()
