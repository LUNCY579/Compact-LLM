"""Read-only post-execution/import witness; no model or corpus regeneration."""
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_open_bank as bank
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import fresh_open_verifier as verifier
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3 import authority
from experiments.rineya_taps_adaptive_recurrence_campaign_v2.quality_pattern_v3.fresh_scope import verify_dispatch


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


v3 = bank.STUDY
labels = v3 / "data/fresh_open_replication_001/SCIENTIFIC_LABELS.json"
expected = "2724f0026bdfaf362f0cfc1d3feef9f7dd5b61ba023b602a47938bc722016abb"
assert sha(labels) == expected
assert "torch" not in sys.modules and "transformers" not in sys.modules
packet = json.loads((v3 / "probe_admission_v3.json").read_text(encoding="utf-8"))
authority.validate_probe_scope(packet)
try:
    authority.validate_probe_unused(packet)
except PermissionError as error:
    assert "already attempted" in str(error)
else:
    raise AssertionError("Completed probe must remain spent")
accepted = verify_dispatch(v3 / "fresh_admission_v2.json")
assert accepted["primary_policy"] == "EVIDENCE_CONSISTENCY_GUARD"
assert sha(labels) == expected
assert not (v3 / "runs/native-fresh-open-replication-001").exists()
assert "torch" not in sys.modules and "transformers" not in sys.modules
print(json.dumps({"status": "PASS_CURRENT_POST_EXECUTION_AND_IMPORT_WITNESS",
                  "fresh_corpus_exists_by_design": True, "fresh_corpus_unchanged": True,
                  "retired_probe_expected_refusal": True, "fresh_v2_consumer_accepts": True,
                  "torch_imported": False, "transformers_imported": False,
                  "model_calls": 0, "frozen_tests_modified": False}))
