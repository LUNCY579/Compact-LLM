from __future__ import annotations

import unittest

from scripts.common.response_control import (
    finalize_response,
    first_sentence_boundary,
    sentence_boundaries,
    validate_response_control_mode,
)


class ResponseControlTests(unittest.TestCase):
    def assertBoundary(self, text: str) -> None:  # noqa: N802 - unittest style helper
        self.assertTrue(first_sentence_boundary(text).found, text)

    def assertNoBoundary(self, text: str) -> None:  # noqa: N802 - unittest style helper
        self.assertFalse(first_sentence_boundary(text).found, text)

    def test_simple_and_multilingual_sentence_boundaries(self) -> None:
        self.assertBoundary("Sounds good.")
        self.assertBoundary("Really?")
        self.assertBoundary("好呀！")
        self.assertBoundary("可以。")

    def test_decimal_version_and_abbreviation_are_not_premature_boundaries(self) -> None:
        self.assertNoBoundary("The value is 3.14")
        self.assertNoBoundary("Use v2.1")
        self.assertNoBoundary("Ask Dr. Smith")
        self.assertNoBoundary("The U.S. rule")
        self.assertBoundary("The value is 3.14.")
        self.assertBoundary("Ask Dr. Smith.")

    def test_url_domain_and_markdown_code_are_protected(self) -> None:
        self.assertNoBoundary("Visit example.com")
        self.assertBoundary("Visit example.com.")
        self.assertNoBoundary("Run `print('ok.')`")
        self.assertNoBoundary("```python\nprint('ok.')\n```")

    def test_unclosed_quote_or_bracket_does_not_stop(self) -> None:
        self.assertNoBoundary('He said "Stop.')
        self.assertBoundary('He said "Stop."')
        self.assertNoBoundary("Use the option (recommended.")
        self.assertBoundary("Use the option (recommended).")

    def test_strict_finalization_trims_after_first_validated_sentence(self) -> None:
        result = finalize_response("First sentence. Second sentence.", "one_sentence")
        self.assertEqual(result.text, "First sentence.")
        self.assertTrue(result.trimmed)
        self.assertEqual(result.sentence_count, 1)

    def test_eos_only_preserves_text_and_reports_multiple_sentences(self) -> None:
        result = finalize_response("First sentence. Second sentence.", "eos_only")
        self.assertEqual(result.text, "First sentence. Second sentence.")
        self.assertFalse(result.trimmed)
        self.assertEqual(result.sentence_count, 2)

    def test_internal_ellipsis_is_protected_and_terminal_ellipsis_is_one_boundary(self) -> None:
        self.assertNoBoundary("Well... okay")
        self.assertNoBoundary("Well… okay")
        boundaries = sentence_boundaries("Well...")
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0].reason, "validated_ellipsis_boundary")

    def test_cjk_attached_next_sentence_needs_no_ascii_space(self) -> None:
        result = finalize_response("可以。再见。", "one_sentence")
        self.assertEqual(result.text, "可以。")

    def test_ordered_list_marker_and_line_boundary(self) -> None:
        self.assertNoBoundary("1. Do this")
        result = finalize_response("Sure\nSecond line", "one_sentence")
        self.assertEqual(result.text, "Sure")
        self.assertEqual(result.first_boundary_reason, "validated_line_boundary")

    def test_invalid_mode_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            validate_response_control_mode("punctuation_only")


if __name__ == "__main__":
    unittest.main()
