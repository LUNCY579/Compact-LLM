"""Bind the already measured exposed CPU arithmetic prompts; no native launch."""
from pathlib import Path
from types import ModuleType
from collections import Counter
import argparse
import hashlib
import json
import os
import sys

P = Path(__file__).resolve().parent.parent
W = P / "artifacts/arithmetic_capture_preparation_work"
OUT = P / "artifacts/arithmetic_capture_actual_plan_work"


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def read(path, pin):
    raw = path.read_bytes()
    assert len(raw) <= 16 * 1024**2 and hashlib.sha256(raw).hexdigest() == pin
    value = json.loads(raw)
    return value, {"path": str(path.resolve()), "bytes": len(raw), "sha256": pin}


def write(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False,
                      allow_nan=False) + "\n").encode()
    with path.open("xb", buffering=0) as stream:
        assert stream.write(raw) == len(raw)
        os.fsync(stream.fileno())
    assert path.read_bytes() == raw
    return {"path": str(path.resolve()), "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(), "value_sha256": digest(value)}


parser = argparse.ArgumentParser()
parser.add_argument("--contract-sha256", required=True)
args = parser.parse_args()
source_path = W / "arithmetic_capture_contract.py"
source_raw = source_path.read_bytes()
assert hashlib.sha256(source_raw).hexdigest() == args.contract_sha256
prompt, prompt_ref = read(Path("C:/Users/clclc/BusinessProject/synth-arithmetic-tokenmeasurement-20261009-attempt001/policy_prompt_tokens.json"),
    "7e2ec66fd08c0a7c5fd56444e0d3fa86cbd0b93b40c3cf8ffad75512ff6d7d66")
measurement, measurement_ref = read(Path("C:/Users/clclc/BusinessProject/synth-arithmetic-tokenmeasurement-20261009-attempt001_projection_corrected/measurement_projected.json"),
    "e9a17b5fbca93c62684c5ddaa0380f721bd0e718158766c01f88ff55f626521e")
population, population_ref = read(P / "evidence/arithmetic_guard_reviewed_cpu_20261009/current/ARITHMETIC_GUARD_POPULATION001.json",
    "2992454925eecfabda4a6046d1d59ef78e718c1639d58714aa1404f90888f5c3")
policy, policy_ref = read(P / "evidence/arithmetic_guard_reviewed_cpu_20261009/current/SYNTH_ARITHMETIC_GUARD_POLICY001.json",
    "42ee9c26f0fe1ab42c368618a5c25385fe09dac0f5777bab92df5c34c5fa05b7")
ledger, ledger_ref = read(P / "evidence/arithmetic_source_ledger_reviewed_cpu_20261009/current/ARITHMETIC_SOURCE_LEDGER001.json",
    "c46f9ca13871e9bd9ab7695f0b79e729a9300d208b4d2bc9fce83ba92e56a252")
for value in (measurement, population, policy, ledger):
    assert value["sha256"] == digest({k: v for k, v in value.items() if k != "sha256"})
assert measurement["measured_rows"] == measurement["expected_rows"] == 512
assert measurement["overflows"] == [] and measurement["failure"] is None
assert measurement["population_sha256"] == population["sha256"]
assert measurement["policy_sha256"] == policy["sha256"] == population["policy_sha256"]
assert len(prompt["rows"]) == len(population["rows"]) == 512
assert Counter(r["parent_id"] for r in prompt["rows"]) == Counter(r["parent_id"] for r in population["rows"])
metadata = []
for tokens, row in zip(prompt["rows"], population["rows"]):
    assert tokens["row_id"] == row["row_id"] and tokens["parent_id"] == row["parent_id"]
    assert tokens["messages"][1] == {"role": "user", "content": row["prompt"]}
    metadata.append({"row_id": row["row_id"], "component_id": row["parent_id"],
        "mechanism": "bounded_integer_arithmetic", "variant": row["condition"], "rendering": row["rendering"]})
module = ModuleType("root_selected_closed_arithmetic_plan")
module.__file__ = str(source_path)
exec(compile(source_raw, str(source_path), "exec"), module.__dict__)
try:
    module.integrated()
except ValueError:
    pass
else:
    raise AssertionError("this builder must remain staged and closed")
plan = module.build_plan(prompt["rows"], metadata, measurement_reference=measurement_ref,
    prompt_reference=prompt_ref, policy_reference=policy_ref, ledger_reference=ledger_ref, fixture_only=False)
assert plan["sha256"] == digest({k: v for k, v in plan.items() if k != "sha256"})
assert plan["row_order_sha256"] == measurement["row_order_sha256"] == population["row_order_sha256"]
batches = plan["batches"]["arithmetic_guard_DEV"]
assert [b["maximum_total_input_positions"] for b in batches] == [208184,208310,237054,237212,294186,293930,407552,409404]
assert len(batches) == 8 and sum(b["maximum_total_forwards"] for b in batches) == 16400
assert sum(b["maximum_total_input_positions"] for b in batches) == 2295832
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == args.contract_sha256
assert not {"torch", "transformers", "tokenizers", "safetensors", "ctypes", "psutil"}.intersection(sys.modules)
OUT.mkdir(exist_ok=False)
plan_ref = write(OUT / "ARITHMETIC_CAPTURE_PLAN001.json", plan)
module.PLAN_PATH = OUT / "ARITHMETIC_CAPTURE_PLAN001.json"
module.PLAN_REF = plan_ref
assert module.plan() == plan  # CPU reconstruction only; no preflight/native call.
receipt = {"kind": "ROOT_ACTUAL_ARITHMETIC_CAPTURE_PLAN_CPU001", "plan_reference": plan_ref,
    "builder_source": {"path": str(source_path), "sha256": args.contract_sha256},
    "population_reference": population_ref, "prompt_reference": prompt_ref,
    "measurement_reference": measurement_ref, "policy_reference": policy_ref, "ledger_reference": ledger_ref,
    "rows": 512, "parents": 128, "batches": 8, "maximum_forwards": 16400,
    "maximum_input_positions": 2295832, "native_launch_performed": False,
    "scoring_performed": False, "M0_enrollment_performed": False, "native_qualification_established": False}
receipt["sha256"] = digest(receipt)
write(OUT / "ROOT_ACTUAL_PLAN001.json", receipt)
print(json.dumps({"plan": plan_ref, "receipt_sha256": receipt["sha256"]}))
