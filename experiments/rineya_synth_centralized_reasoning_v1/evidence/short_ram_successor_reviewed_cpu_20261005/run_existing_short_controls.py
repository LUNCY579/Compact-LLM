"""Run existing short CPU controls explicitly against staged adapters.

Two decoder controls use injected tiny decoder/model-asset hash fixtures, never
published model bytes or installed decoder binary hashing. The prior exact-copy
supervisor assertion is replaced by the declared-delta successor control.
"""
from contextlib import ExitStack
import hashlib
import importlib.util
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
sys.path.insert(0, str(HERE)); sys.path.append(str(PACKAGE))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module); return module


contract = load("short_collection_contract", HERE/"short_collection_contract.py")
worker = load("short_collection_worker", HERE/"short_collection_worker.py")
supervisor = load("short_collection_supervisor", HERE/"short_collection_supervisor.py")
legacy = load("existing_short_controls_on_staged_sources", HERE/"root_test_short_collection.py")
legacy.WORK = HERE
for module in (contract, worker, supervisor):
    if Path(module.__file__).resolve().parent != HERE:
        raise ValueError("existing controls must target staged sources")


def tiny_decoder_control(case):
    with tempfile.TemporaryDirectory(dir=HERE) as directory, ExitStack() as stack:
        path = Path(directory)/"CPU_FIXTURE_DECODER.py"; path.write_bytes(b"CPU_SYNTHETIC_DECODER_SOURCE")
        pin = hashlib.sha256(path.read_bytes()).hexdigest()
        real_hash = contract.native.filehash
        def filehash(filename):
            filename = Path(filename)
            if filename.parent.resolve() == contract.MODEL_DIR.resolve():
                return contract.ASSET_PINS[filename.name]  # Declared tiny binding fixture only.
            return real_hash(filename)
        class Engine:
            def token_to_id(self, value): return {"<|im_end|>": 151645, "<|endoftext|>": 151643}[value]
            def decode(self, ids, *, skip_special_tokens): return "" if ids == [151645, 151643] else "CPU_SYNTHETIC"
        class Tokenizer:
            @staticmethod
            def from_file(path): return Engine()
        stack.enter_context(patch.object(contract.native, "filehash", side_effect=filehash))
        stack.enter_context(patch.dict(sys.modules, {"tokenizers": SimpleNamespace(Tokenizer=Tokenizer)}))
        if case._testMethodName == "test_local_generated_only_decoder_uses_pinned_json_CPU_without_AutoTokenizer":
            stack.enter_context(patch.object(contract, "_tokenizer_runtime", return_value={str(path.resolve()): pin}))
        getattr(case, case._testMethodName)()


def suite():
    result = unittest.TestSuite()
    tiny = {"test_local_generated_only_decoder_uses_pinned_json_CPU_without_AutoTokenizer",
            "test_executed_decoder_python_closure_and_implementation_drift_rejected"}
    for cls in (legacy.ContractTests, legacy.WorkerTests):
        for name in unittest.defaultTestLoader.getTestCaseNames(cls):
            case = cls(name)
            if name in tiny:
                result.addTest(unittest.FunctionTestCase(lambda case=case: tiny_decoder_control(case),
                    description=case.id()+" [CPU_TINY_DECODER_BINDING_FIXTURE]"))
            else:
                result.addTest(case)
    return result


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(suite())
    print("STAGED_TARGETS="+str(HERE))
    print("DECODER_CONTROLS=CPU_TINY_FIXTURES_NO_PUBLISHED_MODEL_OR_DECODER_BINARY_READS")
    raise SystemExit(0 if result.wasSuccessful() else 1)
