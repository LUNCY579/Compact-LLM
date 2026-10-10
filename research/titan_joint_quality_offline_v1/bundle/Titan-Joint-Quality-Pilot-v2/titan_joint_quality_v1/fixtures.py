"""Synthetic pipeline demonstrations. These outputs are never model-quality evidence."""
from copy import deepcopy
from .common import DOMAIN_ROLE, PARENT_SHA, SUPA_SHA, digest, new_directory, write_json

class TinyTokenizer:
    def encode_case(self, messages, target):
        value = int(target) % 7
        return {"ids": [1, 2, 3, 4, value + 5, 13, 14], "response_start": 4,
            "response_end": 6, "eos_position": 5, "eos_token_id": 13}

def tiny_rows():
    rows = []
    allocation = {"reasoning": 100, "transition": 20, "arithmetic": 20, "long_context": 10,
                  "grounding": 30, "format": 20}
    for domain, parents in allocation.items():
        for parent in range(parents):
            for variant in range(2):
                rows.append({"case_id": f"toy.{domain}.{parent}.v{variant}", "parent_id": f"toy.{domain}.{parent}",
                    "family_id": "CPU_FIXTURE_" + domain, "domain": domain, "role": DOMAIN_ROLE[domain],
                    "messages": [{"role": "user", "content": f"Synthetic fixture classify {parent}"}],
                    "canonical_answer": str((parent + variant) % 7), "target": str((parent + variant) % 7),
                    "requested_format": "plain", "variant_id": variant, "pair_kind": "fixture_only"})
    return rows

def run_fixture_pair(out):
    from .exposure import encode_rows, build_exposure_plan
    from .training import run_cpu_fixture
    from .analysis import TRANSITION_SHA, freeze_baselines, analyze_fixture
    from .data import generate_bank
    plan = build_exposure_plan(encode_rows(tiny_rows(), TinyTokenizer()),
        {"parent_sha256": PARENT_SHA, "data_sha256": digest({"fixture": "tiny_rows"}),
         "tokenizer_sha256": digest({"fixture": "TinyTokenizer"})})
    if plan["status"] != "PROSPECTIVE_OFFLINE_EXPOSURE_PLAN":
        raise ValueError(str(plan["rejections"]))
    arms = {"control": run_cpu_fixture(plan, arm="reasoning_only", checkpoint_after=1),
            "joint": run_cpu_fixture(plan, arm="joint", checkpoint_after=1)}
    refs = [{k: v for k, v in row.items() if k != "messages"} for row in generate_bank("pilot")]
    pred = {m: [{"case_id": r["case_id"], "response": r["target"], "completed": True, "truncated": False}
               for r in refs] for m in ("supa", "sparse", "transition", "control", "joint")}
    for i, row in enumerate(refs):
        if (row["domain"] == "reasoning" and row["variant_id"] == 0) or (row["domain"] == "arithmetic" and row["parent_id"].endswith("0000")):
            pred["sparse"][i]["response"] = "999999999"
        if row["domain"] == "arithmetic" and row["parent_id"].endswith("0000"):
            pred["control"][i]["response"] = "999999999"
    freeze = freeze_baselines(refs, {m: pred[m] for m in ("supa", "sparse", "transition")},
        {"supa": SUPA_SHA, "sparse": PARENT_SHA, "transition": TRANSITION_SHA}, evidence_kind="fixture")
    proof = {"evidence_kind": "fixture", **arms}
    report = analyze_fixture(refs, pred, freeze, proof, reasoning_shortcut_floor=.1)
    path = new_directory(out)
    write_json(path / "FIXTURE_EXPOSURE_PLAN.json", plan)
    write_json(path / "FIXTURE_PAIR_RECEIPT.json", proof)
    write_json(path / "FIXTURE_BASELINE_FREEZE.json", freeze)
    write_json(path / "FIXTURE_GATE_REPORT.json", report)
    return {"status": "CPU_FIXTURE_PIPELINE_COMPLETE_NOT_MODEL_EVIDENCE",
        "gate_status": report["status"], "output": str(path),
        "resume_bitwise_equal": {a: r["full_checkpoint_resume_bitwise_equal"] for a, r in arms.items()},
        "native_model_calls": 0, "native_model_loads": 0, "quality_improvement_established": False}
