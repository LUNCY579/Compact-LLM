import unittest

from audit_metadata_exclusions import contains_term, identifier_audit, term_audit


class MetadataAuditTests(unittest.TestCase):
    def test_phrase_boundary_and_unicode(self):
        self.assertTrue(contains_term("At the ＭＵＳＥＵＭ, now.", "museum"))
        self.assertFalse(contains_term("museumlike", "museum"))
        self.assertFalse(contains_term("alpha", ""))

    def test_report_contacts_without_filtering_or_faking_8grams(self):
        rows = [{"id": "one", "world": {"people": ["Ada Reed"]}, "messages": [
            {"role": "system", "content": "hidden forbidden title"},
            {"role": "user", "content": "According to Ada Reed, the museum is open."}]}]
        manifest = {"forbidden_proper_nouns": ["According", "Ada Reed"],
                    "forbidden_titles": ["hidden forbidden title"],
                    "forbidden_domains": ["museum"], "forbidden_task_type_names": []}
        audit = term_audit(rows, manifest)
        self.assertEqual(audit["status"], "BLOCKED_LITERAL_CONTACT")
        self.assertEqual(audit["exact_full_person_name_overlap"], ["ada reed"])
        self.assertEqual(audit["literal_contacts"]["forbidden_titles"], [])
        self.assertIsNone(audit["eight_gram_collision_count"])
        self.assertEqual(len(rows), 1)

    def test_missing_index_is_unknown_not_zero(self):
        reports = identifier_audit([{"id": "new", "parent_id": "old"}], {"entries": [
            {"key": "known", "bars": ["TRAIN"], "ids": ["old"]},
            {"key": "sealed", "bars": ["ANY_READ"], "sha256": "whole-file-only"}]})
        self.assertEqual(reports[0]["overlap"], ["old"])
        self.assertEqual(reports[0]["status"], "BLOCKED_ID_CONTACT")
        self.assertIsNone(reports[1]["overlap"])
        self.assertEqual(reports[1]["status"], "UNVERIFIED_NO_APPROVED_ROW_INDEX")


if __name__ == "__main__":
    unittest.main()
