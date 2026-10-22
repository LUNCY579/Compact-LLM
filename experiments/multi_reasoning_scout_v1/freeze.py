"""Create the final MRS-1 registration once, after offline validation and before inference."""
from __future__ import annotations
from collections import Counter
from . import bank, budget, core, prompts


def verify_bank():
    manifest = core.read_json(core.RECORDS / "BANK_MANIFEST.json")
    core.verify_files(core.RECORDS / "bank", manifest["files"])
    generated = bank.generate()
    for name, rows in zip(("public", "gold", "worlds"), generated):
        expected = b"".join(core.canonical(row) + b"\n" for row in rows)
        if (core.RECORDS / f"bank/{name}.jsonl").read_bytes() != expected:
            raise ValueError("bank does not deterministically regenerate: " + name)
    public, gold, worlds = generated
    access = core.read_json(core.RECORDS / "ACCESS_MANIFEST.json")
    ids = {row["id"] for row in public}
    if ids & set(access["quarantined_ids"]):
        raise ValueError("quarantined ID present")
    # Names cannot manufacture an independent world. Verify the underlying
    # complete generative state, with lexical names removed, is also distinct.
    structural = [core.digest({k: v for k, v in row["world"].items() if k != "names"}) for row in worlds]
    if len(set(structural)) != len(worlds):
        raise ValueError("structurally duplicated world")
    if set(manifest["case_ids"]) != ids or len(public) != manifest["n"]:
        raise ValueError("bank manifest IDs differ")
    return {"status": "PASS", "cases": len(public), "structural_worlds_without_names": len(set(structural)),
            "splits": dict(Counter(r["split"] for r in public)), "quarantined_id_overlap": 0,
            "deterministic_byte_regeneration": True, "protected_payload_reads": 0,
            "limitation": "No protected payload is opened. Semantic overlap with protected populations is unverified."}


def main():
    if (core.RECORDS / "FREEZE.json").exists():
        raise FileExistsError("registration already frozen")
    receipt = verify_bank()
    core.source_identity()
    core.write_new(core.RECORDS / "PROMPT_MANIFEST.json", prompts.manifest())
    core.write_new(core.RECORDS / "protocol.json", {
        "experiment": "MRS-1", "version": 1, "classification": core.CLASSIFICATION,
        "primary_old_candidate": "d1-seed17-bf16-export", "secondary_source": "supa-1.7-frozen",
        "bank": "MRS-bank-1", "prompts": "MRS-prompts-1", "primary_mode": "M1",
        "M2": "implemented and fixture-tested; not dispatched in this experiment",
        "arm_order": ["R0", "A1", "D1", "D2", "D3", "A2_selector", "R1", "R2", "R3", "A3_selector", "S1_diagnostic"],
        "partition_order": ["SMOKE", "CAL", "CHECK"], "within_partition_order": "frozen public.jsonl order",
        "budgets": budget.estimate(), "smoke": {"cases": 8, "minimum_valid_terminated_per_role": 6,
            "roles": ["R0", "R1", "R2", "R3", "A2_selector", "A3_selector"]},
        "primary_contrast": "A3-A2 on D1", "practical_contrast": "A3-A0",
        "selectors": {"S0": "valid-answer majority; first valid role breaks ties", "S1": "answer-only model selector",
                      "S2": "checker-informed model selector; primary A3 output"},
        "parser": "one complete JSON object; no duplicate/extra keys; no repair; EOS required",
        "oracle": "analysis only after generation; never a selector input",
        "bootstrap": {"draws": 2000, "seed": 1978, "unit": "paired world", "coverage": .95},
        "confirmatory_claim": False, "release_claim": False, "no_training": True,
        "future": "same frozen protocol with distinct registered checkpoint; P0/P1/P2 receipts required",
    })
    core.make_freeze()
    core.write_new(core.RECORDS / "evidence/FREEZE_RECEIPT.json", {
        **receipt, "freeze_sha256": core.file_hash(core.RECORDS / "FREEZE.json"),
        "model_calls": 0, "model_results_observed": False,
        "source": core.source_identity(), "fixture_efficacy": False,
    })
    print({"freeze_sha256": core.file_hash(core.RECORDS / "FREEZE.json"), **receipt})


if __name__ == "__main__":
    main()
