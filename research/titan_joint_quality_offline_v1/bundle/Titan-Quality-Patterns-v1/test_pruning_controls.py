"""Stdlib tiny-vector fixtures; no checkpoint, model or task data is read."""
from copy import deepcopy
import math
import unittest

from pruning_controls import (ARM_NAMES, MAX_ELEMENTS_PER_TENSOR, MAX_TOTAL_ELEMENTS,
                              SPARSE_PARENT_SHA256, build_controls, prospective_recipe)


class PruningControlFixtures(unittest.TestCase):
    def test_exact_three_controls_source_immutability_and_norm_equality(self):
        vectors = {"first": [4.0, 3.0, 2.0, 1.0], "second": [1, -2, 3, -4, 5, -6, 7, -8]}
        original = deepcopy(vectors)
        result = build_controls(vectors, frame="toy")
        self.assertEqual(result["status"], "MECHANICAL_NO_MODEL_GAIN")
        self.assertEqual(tuple(result["arms"]), ARM_NAMES)
        self.assertEqual(vectors, original)
        self.assertEqual(result["source_before_sha256"], result["source_after_sha256"])
        self.assertTrue(result["source_unchanged"])
        self.assertEqual(result["arms"][ARM_NAMES[0]], original)
        for name in vectors:
            pruned = result["arms"][ARM_NAMES[1]][name]
            scaled = result["arms"][ARM_NAMES[2]][name]
            self.assertTrue(math.isclose(math.hypot(*pruned), math.hypot(*scaled), rel_tol=1e-12, abs_tol=1e-12))
            self.assertTrue(result["tensor_receipts"][name]["matched_norm_verified_on_fixture"])
        self.assertFalse(result["native_authorized"])
        self.assertFalse(result["model_quality_gain"])
        self.assertEqual(result["model_calls"], 0)

    def test_absolute_magnitude_ties_use_ascending_flat_indices(self):
        result = build_controls({"ties": [-7, 7, -7, 7, 7, -7, 7, -7]}, frame="toy")
        self.assertEqual(result["tensor_receipts"]["ties"]["selected_flat_indices"], [0, 1])
        self.assertEqual(result["arms"][ARM_NAMES[1]]["ties"], [-7, 7, 0, 0, 0, 0, 0, 0])

    def test_integer_ranking_does_not_round_away_distinct_magnitudes(self):
        result = build_controls({"integer": [2**53, 2**53 + 1, 1, 0]}, frame="toy")
        self.assertEqual(result["tensor_receipts"]["integer"]["selected_flat_indices"], [1])
        self.assertEqual(result["arms"][ARM_NAMES[1]]["integer"], [0, 2**53 + 1, 0, 0])

    def test_zero_tensor_is_safe_and_tie_rule_is_explicit(self):
        result = build_controls({"zero": [0, 0.0, -0.0, 0]}, frame="toy")
        receipt = result["tensor_receipts"]["zero"]
        self.assertEqual(receipt["selected_flat_indices"], [0])
        self.assertEqual(receipt["alpha_t"], 0.0)
        self.assertTrue(receipt["zero_tensor"])
        self.assertEqual(receipt["dense_norm"], receipt["top25_norm"])
        self.assertEqual(result["arms"][ARM_NAMES[2]]["zero"], [0, 0, 0, 0])

    def test_vector_boundaries_reject_bool_nonfinite_missing_and_nonquarter_lengths(self):
        invalid = ({}, {"x": []}, {"x": [1, 2, 3]}, {"x": [1, 2, 3, True]},
                   {"x": [1, 2, 3, float("nan")]}, {"x": [1, 2, 3, float("inf")]},
                   {"x": [1, 2, 3, "4"]}, {"x": [[1], [2], [3], [4]]},
                   {"": [1, 2, 3, 4]}, {"x": [0] * (MAX_ELEMENTS_PER_TENSOR + 4)})
        for vectors in invalid:
            with self.subTest(vectors_type=type(vectors).__name__):
                with self.assertRaises(ValueError):
                    build_controls(vectors, frame="toy")

    def test_total_element_limit_rejects_oversized_fixture_collection(self):
        count = MAX_TOTAL_ELEMENTS // MAX_ELEMENTS_PER_TENSOR + 1
        with self.assertRaisesRegex(ValueError, "total fixture"):
            build_controls({f"x{index}": [0] * MAX_ELEMENTS_PER_TENSOR for index in range(count)}, frame="toy")

    def test_absolute_checkpoint_and_raw_lora_frames_and_names_are_rejected(self):
        for frame in ("absolute_checkpoint", "source_whole_weights", "raw_lora", "fp32_model"):
            with self.assertRaises(ValueError):
                build_controls({"x": [1, 2, 3, 4]}, frame=frame, parent_sha256=SPARSE_PARENT_SHA256)
        with self.assertRaisesRegex(ValueError, "LoRA"):
            build_controls({"layer.lora_A.weight": [1, 2, 3, 4]}, frame="new_relative_delta",
                           parent_sha256=SPARSE_PARENT_SHA256)

    def test_prospective_relative_frame_requires_exact_immediate_parent(self):
        for parent in (None, "0" * 64, "a28629c9", True):
            with self.assertRaises(ValueError):
                build_controls({"x": [1, 2, 3, 4]}, frame="new_relative_delta", parent_sha256=parent)
        result = build_controls({"x": [1, 2, 3, 4]}, frame="new_relative_delta",
                                parent_sha256=SPARSE_PARENT_SHA256)
        self.assertEqual(result["status"], "MECHANICAL_NO_MODEL_GAIN")
        self.assertFalse(result["source_frame_independently_verified"])
        self.assertFalse(result["native_authorized"])

    def test_recipe_is_one_new_delta_three_transforms_not_native_permission(self):
        recipe = prospective_recipe()
        self.assertIn("exactly once", recipe["conversion"])
        self.assertIn("never add", recipe["inherited_delta_rule"])
        self.assertEqual(recipe["operator_envelope"]["arm_count"], 3)
        self.assertFalse(recipe["operator_envelope"]["free_parameter_sweep"])
        self.assertIsNone(recipe["native_command"])
        self.assertFalse(recipe["native_authorized"])
        self.assertFalse(recipe["protected_data_authorized"])
        self.assertFalse(recipe["model_quality_gain_established"])


if __name__ == "__main__":
    unittest.main()
