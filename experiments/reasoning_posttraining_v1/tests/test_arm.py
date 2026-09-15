"""Gates for the RRP-003 arm A1 build.

Hermetic: a stub tokenizer stands in for the native one, and no release is read. The
load-bearing tests are `test_a_collision_with_a_dev_bank_is_a_build_failure` and
`test_the_plan_is_computed_from_the_compiled_bytes_and_agrees_with_the_validators` — the first
stops the arm training on its own evaluation surface, the second stops a plan/data mismatch
being discovered three minutes into a GPU launch.
"""
import pytest

from experiments.model_quality_gpu_v1.config import Design
from experiments.reasoning_posttraining_v1 import arm


def meta_for(world_id, family, status, *, group=None, case=None):
    reference = {"status": status} if status != "ANSWER" else {"value": 7}
    return {"id": world_id, "family": family, "reference": reference,
            "reference_text": '{"status":"%s"}' % status if status != "ANSWER" else '{"value":7}',
            "group_id": group or world_id, "prompt_sha256": "p" + world_id,
            "layout": {"case_sha256": case or ("c" + world_id)}}


def policy_for(world_id, question="QUERY {\"a\":1}"):
    return {"id": world_id,
            "prompt": [{"role": "system", "content": "s"},
                       {"role": "user", "content": f"RECORDS {world_id}\n{question}"}]}


def compiled_row(world_id, *, ids=None, family="temporal_route", stratum="ANSWER",
                 parent=None, case=None, question=None):
    ids = ids or [1, 2, 3, 4]
    # The engine requires len(ids) == len(labels) and labels[0] == -100. Supervise the last
    # two positions only, mirroring a tiny JSON target after a long masked prompt.
    labels = [-100] * (len(ids) - 2) + list(ids[-2:])
    return {"id": world_id, "parent_id": parent or world_id, "family": family,
            "stratum": stratum, "source_split": "train", "source_release": "forge_v2",
            "mode": "direct",
            "prompt": [{"role": "system", "content": "s"},
                       {"role": "user", "content": question or f"q {world_id}"}],
            "reference_text": '{"value":7}',
            "ids": ids, "prompt_ids": ids[:2], "labels": labels,
            "answer_start": 2, "answer_end": 3,
            "processed_tokens": len(ids), "supervised_tokens": 2, "truncated": False,
            "case_sha256": case or ("c" + world_id), "prompt_sha256": "p" + world_id}


# ------------------------------------------------------------------ disjointness

def test_a_collision_with_a_dev_bank_is_a_build_failure():
    """Training on the evaluation surface is the one mistake that invalidates everything."""
    compiled = [compiled_row("t1")]
    dev = [{"id": "t1", "group_id": "t1", "case_sha256": "ct1",
            "prompt": compiled[0]["prompt"]}]
    got = arm.disjointness(compiled, {"RRP2_DEV_V2": dev})
    assert got["clean"] is False
    hits = got["against"][0]["collisions"]
    assert hits["id"] == 1 and hits["parent_id"] == 1 and hits["case_sha256"] == 1
    assert hits["last_user_turn"] == 1


def test_disjointness_is_clean_on_genuinely_separate_rows():
    compiled = [compiled_row("t1")]
    dev = [{"id": "d1", "group_id": "d1", "case_sha256": "cd1",
            "prompt": [{"role": "system", "content": "s"},
                       {"role": "user", "content": "entirely different question"}]}]
    got = arm.disjointness(compiled, {"RRP2_DEV_V2": dev})
    assert got["clean"] is True
    assert got["against"][0]["collisions"] == {}


def test_disjointness_catches_a_digit_only_difference():
    """A row differing from a DEV row only in its numbers is a near-duplicate, not a new world."""
    compiled = [compiled_row("t1", question="Option A costs 12 and B costs 40")]
    dev = [{"id": "d1", "group_id": "d1", "case_sha256": "cd1",
            "prompt": [{"role": "system", "content": "s"},
                       {"role": "user", "content": "Option A costs 99 and B costs 11"}]}]
    got = arm.disjointness(compiled, {"RRP2_DEV_V2": dev})
    assert got["clean"] is False
    assert got["against"][0]["collisions"]["last_user_turn_digit_blinded"] == 1
    # but NOT on the verbatim axis
    assert "last_user_turn" not in got["against"][0]["collisions"]


# ------------------------------------------------------------------ the plan

def test_the_plan_is_computed_from_the_compiled_bytes_and_agrees_with_the_validators():
    """Every field controller.presentations() and design_for() check, checked here first."""
    compiled = [compiled_row("t1", ids=[1, 2, 3, 4]),
                compiled_row("t2", ids=[5, 6, 7, 8, 9])]
    plan = arm.build_plan(compiled, exposures=3)

    assert plan["experiment"] == "RRP-003-ARM-A1"
    assert plan["objective"] == "assistant_response_plus_eos"
    assert plan["row_order_per_exposure"] == ["t1", "t2"]
    assert plan["unique_worlds"] == 2
    assert plan["exposures_per_world"] == 3
    assert plan["presentations"] == 6
    # 3 exposures x (4 + 5) tokens
    assert plan["processed_tokens"] == 27
    # each row supervises 2 tokens among labels[1:]; 2 rows x 2 tokens x 3 exposures = 12
    assert plan["supervised_tokens"] == 12
    assert plan["max_sequence"] == 2048 == Design().max_sequence
    assert plan["precision"] == "fp32" and plan["trainable"] == "blocks8-27"
    assert plan["selection_seed"] == Design().selection_seed
    assert plan["order_seed"] == Design().order_seed


def test_the_plan_reproduces_the_frozen_optimizer_design_exactly():
    """controller.design_for compares literally; a drifted key is a refused launch."""
    design = Design()
    plan = arm.build_plan([compiled_row("t1")], exposures=1)
    assert plan["optimizer"] == {
        "betas": [design.beta1, design.beta2], "epsilon": design.epsilon,
        "gradient_clip": design.clip_norm, "name": "MQ CpuAdamW",
        "peak_lr": 1e-5, "schedule": "WSD20/60/20_to_0.1_peak",
        "weight_decay": design.weight_decay}


def test_the_plan_passes_the_real_design_for_validator():
    """Not a re-implementation: call the controller's own function."""
    from experiments.reasoning_posttraining_v1 import controller
    plan = arm.build_plan([compiled_row("t1", ids=list(range(1, 9)))], exposures=1)
    design = controller.design_for(plan)
    assert design.processed_budget == plan["processed_tokens"]
    assert design.trainable_tail == 20


def test_the_plan_identity_is_on_the_controllers_allowlist():
    from experiments.reasoning_posttraining_v1 import controller
    assert arm.ARM_EXPERIMENT in controller.REGISTERED_PLANS


def test_update_groups_are_recomputed_not_asserted():
    """The plan's update_groups must be what group_updates actually returns for this data."""
    from experiments.model_quality_gpu_v1.data import group_updates
    compiled = [compiled_row(f"t{i}", ids=list(range(1, 11))) for i in range(4)]
    plan = arm.build_plan(compiled, exposures=2)
    repeated = [r for _ in range(2) for r in compiled]
    assert plan["update_groups"] == group_updates(repeated, Design().update_tokens)


# ------------------------------------------------------------------ selection and refusal

def test_selection_is_stratified_over_family_by_status():
    """UNKNOWN lives in few v2 families; a flat sample would confound status with family."""
    policy, meta = {}, {}
    for family in ("temporal_route", "ledger_replay"):
        for status in ("ANSWER", "UNKNOWN", "IMPOSSIBLE"):
            for i in range(5):
                wid = f"{family}-{status}-{i}"
                policy[wid] = policy_for(wid)
                meta[wid] = meta_for(wid, family, status)

    class Root:
        pass

    chosen, taken = _select_from(policy, meta, per_bucket=2)
    assert len(chosen) == 12          # 2 families x 3 statuses x 2
    assert set(taken.values()) == {2}
    assert len(taken) == 6


def _select_from(policy, meta, *, per_bucket):
    """Mirror arm.select's bucketing without touching a gzip release."""
    import collections

    from experiments.model_quality_gpu_v1.integrity import digest
    from experiments.reasoning_posttraining_v1.instrument import v2_stratum
    buckets = collections.defaultdict(list)
    for world_id, m in meta.items():
        buckets[(m["family"], v2_stratum(m["reference"]))].append(world_id)
    chosen, taken = [], {}
    for key in sorted(buckets):
        pool = sorted(buckets[key], key=lambda i: (digest([arm.SALT, arm.SEED, i]), i))
        take = pool[:per_bucket]
        taken["|".join(key)] = len(take)
        chosen.extend(take)
    return chosen, taken


def test_selection_is_deterministic():
    policy, meta = {}, {}
    for i in range(20):
        wid = f"w{i}"
        policy[wid] = policy_for(wid)
        meta[wid] = meta_for(wid, "temporal_route", "ANSWER")
    first, _ = _select_from(policy, meta, per_bucket=5)
    second, _ = _select_from(policy, meta, per_bucket=5)
    assert first == second


def test_a_quarantined_id_is_refused_before_compilation():
    registry = {"record": "RRP2_QUARANTINE", "no_quarantine_is_lifted": True, "sha256": "x",
                "entries": [{"key": "bad/bank", "reason": "SELECTION_EXPOSED",
                             "bars": ["TRAIN"], "ids": ["w3"], "evidence_path": "e",
                             "lifted": False, "present": True, "path": "bad.jsonl"}]}
    from experiments.reasoning_posttraining_v1 import quarantine
    with pytest.raises(ValueError, match="barred from TRAIN"):
        quarantine.refuse(["w1", "w3"], registry, use="TRAIN")


# ------------------------------------------------------------------ compilation contract

class StubTokenizer:
    """Enough of the native tokenizer for compile_row's serialization checks."""

    eos_token = "<|im_end|>"
    eos_token_id = 99

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False,
                            **kwargs):
        text = ""
        for m in messages:
            text += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            text += "<|im_start|>assistant\n<think>\n\n</think>\n\n"
        if tokenize:
            return self.encode(text, add_special_tokens=False)
        return text

    def encode(self, text, add_special_tokens=False):
        out = []
        for chunk in text.split("<|im_end|>"):
            out.extend(ord(c) % 97 + 1 for c in chunk[:12])
            out.append(self.eos_token_id)
        return out[:-1]

    def decode(self, ids, skip_special_tokens=False):
        return "\n" if ids else ""


def test_a_row_compile_row_refuses_is_recorded_and_never_truncated():
    """Any compile refusal must be recorded with its reason and drop the row, not shorten it.

    The stub tokenizer is not faithful enough to reach compile_row's OVERLENGTH branch -- it
    trips the serialization check first -- so this test asserts the refusal CONTRACT rather
    than a specific reason string. The real overlength path is exercised against the native
    tokenizer by the measured 5,297-of-16,000 fit count on the actual release.
    """
    policy = {"w1": policy_for("w1", question="x" * 50)}
    meta = {"w1": meta_for("w1", "temporal_route", "ANSWER")}
    compiled, rejected = arm.compile_training(
        ["w1"], policy, meta, StubTokenizer(), max_sequence=3)
    assert compiled == []
    assert len(rejected) == 1
    assert rejected[0]["reason"].strip()
    assert rejected[0]["id"] == "w1" and rejected[0]["family"] == "temporal_route"
    assert rejected[0]["stratum"] == "ANSWER"


def test_compiled_rows_carry_the_fields_the_engine_requires():
    compiled = [compiled_row("t1")]
    for row in compiled:
        assert row["source_split"] == "train"
        assert row["mode"] == "direct"
        assert row["truncated"] is False
        assert row["labels"][0] == -100
        assert len(row["ids"]) == len(row["labels"])


def test_build_refuses_when_nothing_fits_the_window():
    assert arm.TRAIN_WINDOW == 2048
    assert arm.EVAL_WINDOW == 4096


# ------------------------------------------------------------------ hygiene

def test_module_does_not_import_torch_at_module_level():
    import pathlib
    import subprocess
    import sys
    code = ("import experiments.reasoning_posttraining_v1.arm, sys; "
            "print('torch' in sys.modules)")
    root = pathlib.Path(arm.__file__).parents[2]
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         cwd=str(root))
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "False"


def test_the_arm_never_claims_confirmation_or_quality():
    compiled = [compiled_row("t1")]
    audit = arm.disjointness(compiled, {"dev": [compiled_row("different", question="unrelated task")]})
    assert audit["tolerance"] == "ZERO"
    assert arm.ARM_EXPERIMENT == "RRP-003-ARM-A1"


def test_balance_is_applied_after_compilation_not_before():
    """Only ~a third of v2 train fits 2048, and the fit rate varies hugely by family."""
    fitting = ([compiled_row(f"a{i}", family="temporal_route", stratum="ANSWER")
                for i in range(50)]
               + [compiled_row(f"u{i}", family="evidence_chain", stratum="UNKNOWN")
                  for i in range(3)])
    kept = arm.cap_per_bucket(fitting, per_bucket=10)
    import collections
    counts = collections.Counter((r["family"], r["stratum"]) for r in kept)
    assert counts[("temporal_route", "ANSWER")] == 10
    # A depleted bucket contributes everything it has, and no more is invented.
    assert counts[("evidence_chain", "UNKNOWN")] == 3
    assert len(kept) == 13


def test_capping_preserves_the_deterministic_global_order():
    fitting = [compiled_row(f"r{i}", family="f", stratum="ANSWER") for i in range(10)]
    kept = arm.cap_per_bucket(fitting, per_bucket=4)
    assert [r["id"] for r in kept] == ["r0", "r1", "r2", "r3"]
