"""Create the MRS-2 registration once, after offline validation and before inference."""
from __future__ import annotations
from collections import Counter

from experiments.multi_reasoning_scout_v1 import budget
from . import core, prompts, runner

FREEZE_FILES = ("protocol.json", "PROMPT_MANIFEST.json", "MRS2_PROTOCOL.md", "MRS2_PREREG.md",
                "CONTRACT_PROVENANCE.json")


def verify_bank():
    """MRS-2 reuses the MRS-1 frozen bank byte-for-byte; it never regenerates it."""
    binding = core.mrs1_binding()
    public = core.public_bank()
    gold = core.gold_labels()
    if {i.id for i in public} != set(gold):
        raise ValueError("public and gold populations differ")
    access = core.read_json(core.MRS1_RECORDS / "ACCESS_MANIFEST.json")
    if {i.id for i in public} & set(access["quarantined_ids"]):
        raise ValueError("quarantined ID present")
    # Label blindness is structural, not lexical. The answer vocabulary (including the
    # literal "UNKNOWN") legitimately appears in the system prompt, and the digits 0-3
    # appear throughout problem text, so scanning for the gold string proves nothing.
    # What must hold is that an independent role's prompt is EXACTLY the fixed system
    # text, the fixed role text and the public problem -- leaving no channel a label
    # could travel through.
    leak_fields = ("answerable", "difficulty", "verification")
    for item in public:
        for role in prompts.INDEPENDENT:
            messages = prompts.messages(item, role)
            if [m["role"] for m in messages] != ["system", "user"]:
                raise ValueError("unexpected prompt structure")
            if messages[0]["content"] != prompts.SYSTEM:
                raise ValueError("system prompt is not the frozen text")
            if messages[1]["content"] != prompts.ROLES[role] + "\n\nPROBLEM\n" + item.problem:
                raise ValueError("role prompt carries text beyond the role and the problem")
            if any(f in messages[1]["content"] for f in leak_fields):
                raise ValueError("gold metadata field name reachable in prompt")
    return {"status": "PASS", "cases": len(public), "bank_reused_from": "MRS-1",
            "bank_regenerated": False, "splits": dict(Counter(i.split for i in public)),
            "quarantined_id_overlap": 0, "protected_payload_reads": 0,
            "mrs1_binding": binding,
            "limitation": "No protected payload is opened. Semantic overlap with protected "
                          "populations is unverified."}


def main():
    if (core.RECORDS / "FREEZE.json").exists():
        raise FileExistsError("registration already frozen")
    receipt = verify_bank()
    core.source_identity()
    core.write_new(core.RECORDS / "PROMPT_MANIFEST.json", prompts.manifest())
    core.write_new(core.RECORDS / "CONTRACT_PROVENANCE.json", {
        "output_contract": "C1_BARE",
        "selected_by": core.PROBE,
        "selection_rule": "format compliance only; >= 0.75 on every gated role for both "
                          "candidates; answer correctness forbidden as an input",
        "measured_compliance": {"supa-1.7-frozen": {"R0": "8/8", "R2": "8/8", "R3": "8/8"},
                                "d1-seed17-bf16-export": {"R0": "8/8", "R2": "8/8", "R3": "8/8"}},
        "rejected": {"C0_JSON_MRS1": "0/8 every role, both models (the MRS-1 contract)",
                     "C2_LINE": "0/8 every role, both models",
                     "C3_JSON_PREFILL": "checker roles 0-5/8; second field never emitted",
                     "C4_LINE_PREFILL": "checker roles 0/8; second field never emitted"},
        "measured_limitation": "This family emits exactly one field and then stops. Checker "
                               "booleans are unobtainable; R2 abstention and R3 recomputation "
                               "carry those verdicts instead.",
        "unmeasured_by_the_probe": ["R1", "D1", "D2", "D3", "S1", "S2"],
        "unmeasured_handling": "MRS-2's own SCHEMA_SMOKE gate covers all nine consumed roles, "
                               "including both selectors, and fails closed before CAL/CHECK.",
    })
    core.write_new(core.RECORDS / "protocol.json", {
        "experiment": "MRS-2", "version": 1, "classification": core.CLASSIFICATION,
        "supersedes": "MRS-1 (unrunnable: frozen JSON contract, 0/8 native smoke on both models)",
        "changed_from_MRS1": ["output contract C1_BARE", "selector projection is answer-only",
                              "R2/R3 lose their unsatisfiable boolean-field sentences",
                              "smoke gate widened to all nine consumed roles",
                              "constant-answer baseline reported alongside every arm"],
        "unchanged_from_MRS1": ["bank bytes", "item/world population and order", "arm structure",
                                "call budget and token caps", "decoding policy", "resource guards",
                                "statistics", "oracle-is-analysis-only rule"],
        "primary_old_candidate": "d1-seed17-bf16-export", "secondary_source": "supa-1.7-frozen",
        "bank": "MRS-bank-1 (reused by hash)", "prompts": "MRS2-prompts-1", "primary_mode": "M1",
        "M2": "not dispatched; enabling it requires a new prospective budget",
        "arm_order": ["R0", "A1", "D1", "D2", "D3", "A2_selector", "R1", "R2", "R3",
                      "A3_selector", "S1_diagnostic"],
        "partition_order": ["SMOKE", "CAL", "CHECK"],
        "within_partition_order": "frozen public.jsonl order",
        "budgets": runner.estimate(), "limits": budget.LIMITS,
        "smoke": {"cases": 8, "threshold": runner.SMOKE_THRESHOLD, "roles": list(runner.SMOKE_GATED)},
        "primary_contrast": "A3-A2 on d1-seed17-bf16-export", "practical_contrast": "A3-A0",
        "selectors": {"S0": "valid-answer majority; first valid role breaks ties",
                      "S1": "answer-only model selector over repeated-direct candidates",
                      "S2": "model selector over the diverse roles; primary A3 output"},
        "selector_projection_caveat": "Under a single-field contract S1 and S2 receive the same "
                                      "shape of information and differ only in candidate source. "
                                      "This is weaker than MRS-1 intended and is registered.",
        "parser": "exactly one in-vocabulary answer token; EOS required; no repair of any kind",
        "oracle": "analysis only after generation; never a selector input",
        "constant_baseline": "reported for every arm, family and difficulty stratum",
        "bootstrap": {"draws": 2000, "seed": 1978, "unit": "paired world", "coverage": .95},
        "confirmatory_claim": False, "release_claim": False, "no_training": True,
        "future": "same frozen protocol with a distinct registered checkpoint; hash-bound "
                  "P0/P1/P2 receipts required for cells C and D",
    })
    paths = [core.RECORDS / n for n in FREEZE_FILES]
    core.write_new(core.RECORDS / "FREEZE.json", {
        "version": "MRS-2", "classification": core.CLASSIFICATION,
        "source_files": core.source_files(), "source": core.source_identity(),
        "files": {str(p.relative_to(core.ROOT)).replace("\\", "/"): core.file_hash(p) for p in paths},
        "mrs1": core.mrs1_binding(),
        "model_results_observed": False, "future_protocol_changes": "NEW EXPERIMENT REQUIRED",
    })
    core.write_new(core.RECORDS / "evidence/FREEZE_RECEIPT.json", {
        **receipt, "freeze_sha256": core.file_hash(core.RECORDS / "FREEZE.json"),
        "model_calls": 0, "model_results_observed": False,
        "source": core.source_identity(), "fixture_efficacy": False,
    })
    print({"freeze_sha256": core.file_hash(core.RECORDS / "FREEZE.json"), **{
        k: receipt[k] for k in ("status", "cases", "splits", "bank_regenerated")}})


if __name__ == "__main__":
    main()
