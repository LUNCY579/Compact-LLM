from __future__ import annotations

"""Unit tests for Phase 4B's dependency-light structural and selection logic."""

import json
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from scripts.common.check_human_gate import has_human_pass, normalize_phase_label
from scripts.common.model_introspection import linear_from_slices
from scripts.common.reference_selection import resolve_selected_model
from scripts.phase_4b_structural_reduction.recommend_reference_candidate import (
    recommend,
)
from scripts.phase_4b_structural_reduction.reduce_depth import choose_layers


class Phase4BStructuralReductionTests(unittest.TestCase):
    def test_linear_from_slices_matches_manual_selection(self) -> None:
        """Physical linear slicing must exactly match direct tensor indexing."""

        layer = nn.Linear(5, 4, bias=True)
        with torch.no_grad():
            layer.weight.copy_(torch.arange(20, dtype=torch.float32).reshape(4, 5))
            layer.bias.copy_(torch.arange(4, dtype=torch.float32))
        rows = torch.tensor([1, 3])
        cols = torch.tensor([0, 2, 4])
        sliced = linear_from_slices(layer, row_indices=rows, column_indices=cols)
        self.assertEqual(tuple(sliced.weight.shape), (2, 3))
        torch.testing.assert_close(sliced.weight, layer.weight[rows][:, cols])
        torch.testing.assert_close(sliced.bias, layer.bias[rows])

    def test_depth_candidate_selection(self) -> None:
        """Suffix and uniform controls must retain the requested layer counts."""

        self.assertEqual(choose_layers(8, "drop-last", 2, ""), [0, 1, 2, 3, 4, 5])
        self.assertEqual(choose_layers(8, "keep-first", 3, ""), [0, 1, 2])
        self.assertEqual(choose_layers(8, "uniform", 4, ""), [0, 2, 5, 7])
        self.assertEqual(choose_layers(8, "uniform-drop", 2, ""), [0, 1, 3, 4, 6, 7])
        self.assertEqual(choose_layers(8, "explicit", 0, "0,3,7"), [0, 3, 7])

    def test_recommend_reduced_candidate_or_dense_fallback(self) -> None:
        """Recommendation prefers eligible reduction and otherwise keeps dense."""

        dense = {
            "name": "dense_reference",
            "eligible": True,
            "parameter_reduction_fraction": 0.0,
            "peak_vram_gb": 2.0,
            "mean_generation_seconds": 1.0,
        }
        reduced = {
            "name": "mlp_10pct",
            "eligible": True,
            "parameter_reduction_fraction": 0.08,
            "peak_vram_gb": 1.8,
            "mean_generation_seconds": 0.9,
        }
        rejected = {
            "name": "drop_last_4",
            "eligible": False,
            "parameter_reduction_fraction": 0.20,
            "peak_vram_gb": 1.5,
            "mean_generation_seconds": 0.7,
        }
        self.assertEqual(recommend([dense, reduced, rejected])["name"], "mlp_10pct")
        reduced["eligible"] = False
        self.assertEqual(recommend([dense, reduced, rejected])["name"], "dense_reference")

    def test_phase_4b_gate_and_reference_manifest(self) -> None:
        """Downstream resolution requires both a valid label and human-gated manifest."""

        self.assertEqual(normalize_phase_label("4b"), "4B")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            report = root / "PHASE_4B_STOP_GATE.md"
            report.write_text("- Human decision: `PASS`\n", encoding="utf-8")
            self.assertTrue(has_human_pass(report))

            manifest = root / "selected.json"
            manifest.write_text(
                json.dumps(
                    {
                        "human_gate": "PASS",
                        "selected_model": "artifacts/candidate",
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                resolve_selected_model(manifest_path=manifest),
                "artifacts/candidate",
            )


if __name__ == "__main__":
    unittest.main()
