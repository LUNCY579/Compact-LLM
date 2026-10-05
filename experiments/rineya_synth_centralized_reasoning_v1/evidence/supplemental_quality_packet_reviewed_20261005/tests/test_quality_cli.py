import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import quality_cli
from test_measurement_enrollment import fixture


class CliTests(unittest.TestCase):
    def test_enroll_and_no_overwrite(self):
        cohort, baseline, sources = fixture()
        policy = json.loads((quality_cli.ROOT / "QUALITY_SCREEN_POLICY_001.json").read_text(encoding="utf-8"))
        with tempfile.TemporaryDirectory(dir=quality_cli.ROOT) as name:
            root = Path(name)
            for filename, content in (("cohort.json", cohort), ("baseline.json", baseline),
                                      ("policy.json", policy)):
                (root / filename).write_text(json.dumps(content), encoding="utf-8")
            args = ["enroll", "--cohort", "cohort.json", "--baseline", "baseline.json", "--policy", "policy.json",
                    "--baseline-id", "fixture:M0", "--output", "enrollment.json"]
            with patch.object(quality_cli, "ROOT", root), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(quality_cli.main(args), 0)
                original = (root / "enrollment.json").read_bytes()
                self.assertEqual(quality_cli.main(args), 2)
                self.assertEqual((root / "enrollment.json").read_bytes(), original)

    def test_native_permission_option_absent(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                quality_cli.main(["decide", "--allow-model-load"])

    def test_path_escape_refused_before_read(self):
        with contextlib.redirect_stdout(io.StringIO()):
            result = quality_cli.main(["enroll", "--cohort", "../private.json", "--baseline", "missing.json",
                                      "--baseline-id", "fixture:M0", "--output", "never-created.json"])
        self.assertEqual(result, 2)
        self.assertFalse((quality_cli.ROOT / "never-created.json").exists())


if __name__ == "__main__":
    unittest.main()
