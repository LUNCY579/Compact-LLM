"""Arithmetic-only CPU tokenizer diagnostics; unchanged atomic R/S encoder.

Policy rows contain prompt IDs only. Gold arrays/masks remain explicitly CPU
oracle diagnostics. Every overflow is retained; none authorizes native scoring.
"""
from copy import deepcopy
from pathlib import Path
import hashlib
import importlib
import json
import sys

PACKAGE = next(p for p in Path(__file__).resolve().parents if p.name == "rineya_synth_centralized_reasoning_v1")
SOURCE_PINS = {"cpu_tokenizer.py": "268d47037581e7d022cb4d4d53dd9b7e4454909c74b910572d2ba153e463b70f",
               "arithmetic_guard.py": "02025fecbddd0fa882e67501c19556aad01795b608a33821f14201757fdc25b9"}
SYSTEM = 'Compute the displayed arithmetic expression. Return exactly one JSON object with key "answer" and an integer value. No explanations or extra keys.'


def require(value, message):
    if not value: raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def modules():
    for name, expected in SOURCE_PINS.items():
        require(hashlib.sha256((PACKAGE / name).read_bytes()).hexdigest() == expected, "exact tokenizer/guard source pin")
    if str(PACKAGE) not in sys.path: sys.path.insert(0, str(PACKAGE))
    result = [importlib.import_module(name) for name in ("arithmetic_guard", "cpu_tokenizer")]
    for name, module in zip(("arithmetic_guard", "cpu_tokenizer"), result):
        require(Path(module.__file__).resolve() == PACKAGE / (name + ".py"), "selected pure module origin")
    result[1]._no_frameworks()
    return result


class EncodingFailure(ValueError):
    def __init__(self, message, partial):
        super().__init__(message); self.partial = deepcopy(partial)


def _ids(value):
    require(type(value) is list and bool(value) and all(type(n) is int and 0 <= n < 151936 for n in value), "typed nonempty Titan token IDs")
    return list(value)


def encode_arithmetic(row, checked_tokenizer, policy):
    guard, tokens = modules()
    partial = {"row_id": row.get("row_id"), "stage": "guard_validation"}
    require(policy == guard.policy(), "exact frozen arithmetic policy")
    answer = guard.validate_row(row, policy)
    target = guard.canonical({"answer": answer})
    require(guard.score(row, target, policy) == {"correct": True, "malformed": False}, "independent canonical arithmetic target")
    try:
        binding = checked_tokenizer.verify_binding()
        require(checked_tokenizer._engine.padding is None and checked_tokenizer._engine.truncation is None,
                "tokenizer padding/truncation forbidden")
        messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": row["prompt"]}]
        partial.update(stage="template", messages=messages, target_CPU_only=target)
        prompt = checked_tokenizer._template.render(messages=deepcopy(messages), tools=None, **tokens.TEMPLATE_KWARGS)
        require(type(prompt) is str and prompt.endswith(tokens.ASSISTANT_PREFIX), "exact nonthinking assistant prefix")
        prompt_ids = _ids(checked_tokenizer._engine.encode(prompt, add_special_tokens=False).ids)
        target_ids = _ids(checked_tokenizer._engine.encode(target, add_special_tokens=False).ids)
        partial.update(stage="roundtrip", prompt_ids=prompt_ids, target_ids_CPU_only=target_ids)
        require(tokens.PAD_ID not in prompt_ids + target_ids and tokens.EOS_ID not in target_ids, "PAD/target EOS forbidden")
        completion = target_ids + [tokens.EOS_ID]
        require(checked_tokenizer._engine.decode(prompt_ids, skip_special_tokens=False) == prompt
                and checked_tokenizer._engine.decode(target_ids, skip_special_tokens=False) == target
                and checked_tokenizer._engine.decode(completion, skip_special_tokens=False) == target + "<|im_end|>", "prompt/target/EOS decode roundtrip")
        full = checked_tokenizer._template.render(messages=deepcopy(messages) + [{"role": "assistant", "content": target}],
            tools=None, add_generation_prompt=False, enable_thinking=False)
        require(full == prompt + target + "<|im_end|>\n", "exact full template target/EOS/newline")
        joint_ids = _ids(checked_tokenizer._engine.encode(full, add_special_tokens=False).ids)
        newline = _ids(checked_tokenizer._engine.encode("\n", add_special_tokens=False).ids)
        partial.update(stage="joint_boundary", joint_ids_CPU_only=joint_ids)
        end = len(prompt_ids) + len(completion)
        require(joint_ids[:len(prompt_ids)] == prompt_ids and joint_ids[len(prompt_ids):end] == completion
                and joint_ids[end:] == newline, "exact joint prompt/completion/EOS/newline token boundaries")
        require(checked_tokenizer.verify_binding() == binding, "tokenizer binding drift")
        tokens._no_frameworks()
        overflow = len(prompt_ids) > 480 or len(completion) > 32 or end > 512
        policy_row = {"row_id": row["row_id"], "parent_id": row["parent_id"], "messages": messages,
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "input_ids": prompt_ids,
            "attention_mask": [1] * len(prompt_ids), "prompt_tokens": len(prompt_ids), "prompt_token_sha256": digest(prompt_ids)}
        oracle = {"row_id": row["row_id"], "parent_id": row["parent_id"], "target": target,
            "target_ids": target_ids, "completion_ids": completion, "input_ids": prompt_ids + completion,
            "labels": [-100] * len(prompt_ids) + completion, "attention_mask": [1] * end,
            "completion_including_EOS_tokens": len(completion), "total_tokens": end,
            "full_template_sha256": hashlib.sha256(full.encode()).hexdigest()}
        return {"policy_row": policy_row, "oracle_CPU_only": oracle, "overflow": overflow,
                "joint_boundaries_verified": True, "tokenizer_binding": binding}
    except Exception as error:
        raise EncodingFailure(str(error), partial) from error


def measure_population(population, tokenizer, policy, *, check):
    guard, tokens = modules()
    require(callable(check), "explicit cooperative resource check")
    require(population.get("sha256") == digest({k: v for k, v in population.items() if k != "sha256"})
            and population == guard.compile_population(policy), "exact complete original arithmetic population")
    require(len(population["rows"]) == 512, "all original512 arithmetic rows")
    binding = tokenizer.verify_binding(); rows = []; oracle = []; overflows = []; failure = None
    for index, row in enumerate(population["rows"]):
        try:
            check("before_arithmetic_row_%d" % index)
            encoded = encode_arithmetic(row, tokenizer, policy)
            rows.append(encoded["policy_row"]); oracle.append(encoded["oracle_CPU_only"])
            if encoded["overflow"]:
                overflows.append({"row_id": row["row_id"], "prompt_tokens": encoded["policy_row"]["prompt_tokens"],
                    "completion_including_EOS_tokens": encoded["oracle_CPU_only"]["completion_including_EOS_tokens"],
                    "total_tokens": encoded["oracle_CPU_only"]["total_tokens"]})
            check("after_arithmetic_row_%d" % index)
        except Exception as error:
            failure = {"row_id": row["row_id"], "ordinal": index + 1, "error_type": type(error).__name__,
                "error": str(error), "partial": deepcopy(getattr(error, "partial", None))}
            break
    try: require(tokenizer.verify_binding() == binding, "final tokenizer binding drift")
    except Exception as error:
        failure = failure or {"ordinal": None, "error_type": type(error).__name__, "error": str(error), "partial": None}
    value = {"kind": "SYNTH_ARITHMETIC_CPU_TOKEN_MEASUREMENT001", "population_sha256": population["sha256"],
        "policy_sha256": policy["sha256"], "parent_order_sha256": population["parent_order_sha256"],
        "row_order_sha256": population["row_order_sha256"], "binding": binding, "policy_rows": rows,
        "oracle_rows_CPU_only": oracle, "overflows": overflows, "failure": failure,
        "expected_rows": 512, "measured_rows": len(rows), "unmeasured_rows": 512 - len(rows),
        "status": "FAILED_TOKEN_BOUNDARY" if failure else "BLOCKED_CONTEXT_FIT" if overflows else "CPU_CONTEXT_MEASURED_NATIVE_UNADMITTED",
        "context_cap_tokens": 512, "prompt_cap_tokens": 480, "generation_reserve_tokens": 32,
        "native_launch_authorized": False, "native_scoring_authorized": False, "training_admitted": False,
        "M0_enrollment_performed": False, "model_quality_established": False, "population_changed": False}
    value["sha256"] = digest(value)
    return value
