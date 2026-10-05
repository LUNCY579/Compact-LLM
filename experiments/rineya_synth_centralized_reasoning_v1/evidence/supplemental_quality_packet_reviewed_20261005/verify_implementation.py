"""Stdlib fixture/syntax verification with framework import tripwires."""
from __future__ import annotations

import hashlib
import importlib.abc
import json
import os
from pathlib import Path
import sys
import time
import unittest

ROOT = Path(__file__).resolve().parent
VERSION = sys.argv[1] if len(sys.argv) > 1 else "001"
TEMP = ROOT / ("verification-temp-" + VERSION)
TEMP.mkdir(exist_ok=False)
os.environ["TEMP"] = str(TEMP)
os.environ["TMP"] = str(TEMP)
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
forbidden = {"torch", "transformers", "peft", "tensorflow", "jax", "flax", "tokenizers", "safetensors"}
attempts = []


class NoModelFrameworks(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".", 1)[0] in forbidden:
            attempts.append(fullname)
            raise AssertionError("Offline verification forbids model/tokenizer framework import: " + fullname)
        return None


sys.meta_path.insert(0, NoModelFrameworks())
sys.path.insert(0, str(ROOT))
suite = unittest.defaultTestLoader.discover(str(ROOT / "tests"))
started = time.monotonic()
log_path = ROOT / ("VERIFICATION_" + VERSION + ".log")
with log_path.open("x", encoding="utf-8") as log:
    result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
syntax = []
hashes = {}
for path in sorted([*ROOT.glob("*.py"), *(ROOT / "tests").glob("*.py")]):
    compile(path.read_text(encoding="utf-8"), str(path), "exec")
    hashes[path.relative_to(ROOT).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    syntax.append(path.relative_to(ROOT).as_posix())
payload = {"schema": 1, "status": "PASS_MECHANICAL_ONLY" if result.wasSuccessful() and not attempts else "FAILED",
    "tests_run": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
    "skipped": len(result.skipped), "framework_import_attempts": attempts,
    "frameworks_loaded": sorted(name for name in sys.modules if name.split(".", 1)[0] in forbidden),
    "native_model_loads": 0, "native_forward_calls": 0, "optimizer_updates": 0,
    "native_tokenizer_execution": 0, "protected_or_confirmation_bank_reads": 0,
    "elapsed_seconds": time.monotonic() - started, "runtime": sys.executable,
    "syntax_compiled_in_memory": syntax, "implementation_hashes": hashes,
    "log": {"path": str(log_path), "sha256": hashlib.sha256(log_path.read_bytes()).hexdigest()},
    "fixture_results_establish_model_quality": False,
    "new_test_temp_and_caches_under_businessproject": True}
output = ROOT / ("VERIFICATION_" + VERSION + ".json")
with output.open("x", encoding="utf-8") as stream:
    json.dump(payload, stream, indent=2)
print(json.dumps({k: v for k, v in payload.items() if k not in {"implementation_hashes", "syntax_compiled_in_memory"}}, indent=2))
raise SystemExit(0 if payload["status"] == "PASS_MECHANICAL_ONLY" and not payload["frameworks_loaded"] else 1)
