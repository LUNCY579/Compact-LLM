import json
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from experiments.rineya_centralized_reasoning_system_v1 import procedure_development as dev
from experiments.rineya_centralized_reasoning_system_v1.operations import OperationsStore


class ProcedureFailureRecoveryTests(unittest.TestCase):
    def test_distinct_interruptions_preserve_both_original_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = OperationsStore(root/"state.sqlite3", root/"receipts")
            try:
                for label in ("first interruption", "second interruption"):
                    with patch.object(dev, "execute_case", side_effect=RuntimeError(label)):
                        with self.assertRaisesRegex(RuntimeError, label):
                            dev.run_development(store, count=16, seed=202610014)
                files = list((root/"receipts").glob("procedure-development-failure-*.json"))
                self.assertEqual(len(files), 2)
                records = [json.loads(path.read_bytes())["payload"] for path in files]
                self.assertEqual({r["exception"] for r in records}, {"RuntimeError: first interruption", "RuntimeError: second interruption"})
                self.assertEqual(len({r["attempt_id"] for r in records}), 2)
                self.assertEqual(len({r["study_key"] for r in records}), 1)
            finally:
                store.close()


if __name__ == "__main__":
    unittest.main()
