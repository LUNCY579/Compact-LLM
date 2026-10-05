"""CPU prompt-only TRAIN augmentation for unqualified long mechanics."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parents[1]
sys.path.insert(0, str(PACKAGE))
from long_inference_inputs import LocalPromptTokenizer, no_frameworks
from inference_component import validate_row
from rs_contract import digest


def binding(path):
    return {"path": str(path), "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    no_frameworks()
    config_path = PACKAGE / "configs/SYNTH_INFQ001_TRAIN4.json"
    corpus_path = PACKAGE / "data/crossed_source2_v1_20261003/candidate_records.jsonl"
    token_source = PACKAGE / "evidence/long_context_guards_root1_20261004/receipt.json"
    source_before = {str(x): binding(x) for x in (config_path, corpus_path, token_source, Path(__file__))}
    config = json.loads(config_path.read_text())
    assert binding(corpus_path)["sha256"] == config["data_pins"][str(corpus_path)]
    short = config["rows"][0]
    selected = []
    with corpus_path.open() as stream:
        for line in stream:
            value = json.loads(line)
            if value["row_id"] == short["row_id"]:
                selected.append(value)
    assert len(selected) == 1
    source = selected[0]
    assert source["arm"] == "M1" and source["split"] == "train" and source["skill"] in ("R", "S")
    messages = deepcopy(source["messages"])
    assert messages[-1]["role"] == "user"
    tokenizer_binding = json.loads(token_source.read_text())["tokenizer_binding"]
    tokenizer = LocalPromptTokenizer(tokenizer_binding)
    assert tokenizer.encode_prompt(messages) == short["prompt_ids"]
    assert len(short["prompt_ids"]) == 419
    original = messages[-1]["content"]

    def make(count):
        result = deepcopy(messages)
        result[-1]["content"] = original + "\n\nNOTE: These entries are unrelated to the task.\n" + "neutral " * count + "\nEND NOTE."
        return result, tokenizer.encode_prompt(result)

    probes = [make(n)[1] for n in (0, 1, 2)]
    assert len(probes[1]) - len(probes[0]) == len(probes[2]) - len(probes[1]) == 1
    rows = []
    for cap in (1024, 2048, 4096):
        target = cap - 32
        count = target - len(probes[0]); assert 0 <= count <= 4096
        augmented, ids = make(count)
        assert len(ids) == target and augmented[:-1] == messages[:-1]
        assert augmented[-1]["content"].startswith(original)
        assert tokenizer.encode_prompt(augmented) == ids
        row = {"row_id": digest({"purpose": "TRAIN_ONLY_LONG_MECHANICS_NOT_STUDY", "source_row": short["row_id"], "context_cap": cap}),
               "prompt_sha256": digest(augmented), "prompt_ids": ids,
               "prompt_tokens": len(ids), "attention_mask": [1] * len(ids)}
        validate_row(row, context_cap=cap, generation_cap=32)
        rows.append({"messages": augmented, "row": row, "context_cap": cap,
            "neutral_words": count, "source_arm": "M1", "source_split": "train",
            "source_row_id": short["row_id"], "source_parent_id": source["parent_id"],
            "source_skill": source["skill"], "native_execution_authorized": False})
    assert tokenizer.verify_binding() == tokenizer_binding
    source_after = {str(x): binding(x) for x in (config_path, corpus_path, token_source, Path(__file__))}
    assert source_after == source_before
    no_frameworks()
    result = {"schema_version": 1, "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "status": "CPU_TRAIN_DERIVED_LONG_PROMPTS_NATIVE_UNQUALIFIED",
        "source_before": source_before, "source_after": source_after,
        "tokenizer_binding": tokenizer_binding, "short_parity_row": short,
        "short_parity_prompt_tokens": 419, "two_parity_positions": 838,
        "rows": rows, "source_packet_sha256": digest([x["row"] for x in rows]),
        "generation_reservation": 32, "padding_added": False, "prompt_or_target_truncated": False,
        "model_config_or_weights_read": False, "framework_or_GPU_execution": False,
        "guard_prompts_or_oracle_sidecars_read": False, "actual_grading_performed": False,
        "training_corpus_changed": False, "native_launch_authorized": False,
        "semantic_answer_correctness_claimed": False,
        "scope": "Numerical mechanics inputs only; original TRAIN text retained, unrelated note appended. No training or evaluation use."}
    result["sha256"] = digest(result)
    target = HERE / "TRAIN_PROMPTS_CPU.json"
    with target.open("xb", buffering=0) as stream:
        stream.write((json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n").encode())
        os.fsync(stream.fileno())
    print({"rows": len(rows), "prompt_tokens": [x["row"]["prompt_tokens"] for x in rows],
        "two_parity_positions": 838, "body_sha256": result["sha256"],
        "native_launch_authorized": False})


if __name__ == "__main__":
    main()
