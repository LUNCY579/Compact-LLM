#!/usr/bin/env python3
"""Compare local primitives with immutable RineyaBeta main-branch source blobs."""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
import tempfile
from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parents[1]
SRC = LAB_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

from rineya_significance_lab.runtime import (  # noqa: E402
    SignificancePolicy,
    bottom_right_causal_mask,
    post_rope_gqa_significance,
    select_keep_indices,
)

SOURCE_COMMIT = "9e00be74098878290dc2b0d54cbffee4652c6088"
SIGNIFICANCE_BLOB = "30ee29724dce52de5a288f6421123fb718240541"
CHUNKED_BLOB = "499b3ccc8d4e6954d4dac89aacea590ef25fc46f"


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        text=True,
        encoding="utf-8",
        capture_output=True,
    ).stdout


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference file: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def verify(source_repo: Path) -> dict[str, object]:
    repo = source_repo.resolve(strict=True)
    sig_blob = _git(repo, "rev-parse", f"{SOURCE_COMMIT}:scripts/kv/significance.py").strip()
    chunk_blob = _git(repo, "rev-parse", f"{SOURCE_COMMIT}:scripts/kv/chunked_prefill.py").strip()
    if sig_blob != SIGNIFICANCE_BLOB or chunk_blob != CHUNKED_BLOB:
        raise RuntimeError("RineyaBeta source blob identity mismatch")
    with tempfile.TemporaryDirectory(prefix="rineya-source-parity-") as temporary:
        root = Path(temporary)
        sig_path = root / "significance.py"
        chunk_path = root / "chunked_prefill.py"
        sig_path.write_text(_git(repo, "show", f"{SOURCE_COMMIT}:scripts/kv/significance.py"), encoding="utf-8")
        chunk_path.write_text(_git(repo, "show", f"{SOURCE_COMMIT}:scripts/kv/chunked_prefill.py"), encoding="utf-8")
        source = _load("rineya_source_significance", sig_path)
        chunked = _load("rineya_source_chunked", chunk_path)

        generator = torch.Generator().manual_seed(17)
        query = torch.randn((1, 16, 4, 8), generator=generator)
        key = torch.randn((1, 8, 4, 8), generator=generator)
        expected_score = source._score_rows(type("M", (), {})(), query, key, 2, None)
        local_score = post_rope_gqa_significance(query, key, window=2)
        torch.testing.assert_close(local_score, expected_score, rtol=0, atol=0)

        ranking = torch.rand((1, 8, 12), generator=generator)
        source_policy = source.Policy(budget=6, sinks=1, recent=2, count_exponent=1.0, pool=1)
        expected_keep = source.select_keep(ranking, 12, source_policy, ranking.device)
        local_keep = select_keep_indices(
            ranking,
            SignificancePolicy(budget=6, sinks=1, recent=2, count_exponent=1.0),
        )
        if not torch.equal(local_keep, expected_keep):
            raise AssertionError("selection indices differ from source")

        source._COUNTS.clear()
        source._COUNTS[0] = torch.tensor([[[3.0, 2.0]]])
        gather_index = torch.tensor([[[0, 2]]])
        source._gather_counts(0, gather_index, 3)
        if not torch.equal(source._COUNTS[0], torch.tensor([[[3.0, 0.0]]])):
            raise AssertionError("count-lag gather differs from audited source")

        source_mask = chunked.causal_lower_right(3, 7, torch.device("cpu"), torch.float32)
        local_mask = bottom_right_causal_mask(3, 7, device="cpu", dtype=torch.float32)
        if not torch.equal(source_mask, local_mask):
            raise AssertionError("bottom-right causal mask differs from source")

    return {
        "schema_version": "1.0",
        "status": "PASS",
        "source": {
            "remote": "https://github.com/LUNCY579/RineyaBeta.git",
            "commit": SOURCE_COMMIT,
            "significance_blob": SIGNIFICANCE_BLOB,
            "chunked_prefill_blob": CHUNKED_BLOB,
        },
        "checks": {
            "post_rope_gqa_score": "exact_tensor_equal",
            "sink_heavy_recent_selection": "exact_indices_equal_no_ties",
            "count_lag_gather": "exact_tensor_equal",
            "bottom_right_causal_mask": "exact_tensor_equal",
        },
        "claim_limit": "Primitive parity only; this is not end-to-end source benchmark reproduction.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=LAB_ROOT / "results" / "source_parity.json",
    )
    args = parser.parse_args()
    result = verify(args.source_repo)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
