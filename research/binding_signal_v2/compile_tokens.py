"""Tokenizer-only v2 readiness audit using unchanged qualified serialization.

Derived from binding_signal_v1/compile_tokens.py. No model runtime imports,
source-weight reads, inference, training permit, or edits to frozen input bytes.
DEV policy inputs and gold are separate, create-only files.
"""
from pathlib import Path
from collections import Counter
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import sys

BASE = Path("C:/Users/clclc/BusinessProject")
DATA = BASE / "rineya-dataset-retention-records/binding-signal-prep-20260920/candidate-v2"
OUT = DATA / "compiled-v1"
PILOT = BASE / "ChatBubble-retention-doselora-resume-20260920"
SOURCE = BASE / "Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze"
BUILD_SHA256 = "f2a4e1986ce7b42b7317f3d888a3471addc358a5aabab6659d49685035c62741"
NATIVE_FILES = {
    "experiments/reasoning_posttraining_v1/serialization.py": "4be59f7d4022f1e0687a47a42be4949d6020516b697155245e6763dcb0aa1b40",
    "experiments/model_quality_gpu_v1/data.py": "148204b01b9e17e9979f3ef1a92b3effae6c72186cb9d07ce72e0afb611f65ab",
    "experiments/model_quality_gpu_v1/source_lock.json": "4945317bfe2152690230bcd3e509a286b6b9bbe6a84c0f1af3bd86c449abaa9a",
}
SIDECARS = ("config.json", "generation_config.json", "tokenizer.json",
            "tokenizer_config.json", "vocab.json", "merges.txt")
EXPECTED_SIDECARS = {
    "config.json": "660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd",
    "generation_config.json": "2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2",
    "tokenizer.json": "aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4",
    "tokenizer_config.json": "d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101",
    "vocab.json": "ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910",
    "merges.txt": "8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def no_model_runtime():
    require(not any(name in sys.modules for name in ("torch", "tensorflow", "jax")),
            "tokenizer-only compiler imported a forbidden model runtime")


def main():
    os.environ.update(CUDA_VISIBLE_DEVICES="", TOKENIZERS_PARALLELISM="false",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", USE_TORCH="0",
                      USE_TF="0", USE_FLAX="0")
    no_model_runtime()
    require(not OUT.exists() and not OUT.is_symlink(), "token compilation outputs must be fresh")
    raw_build = (DATA / "BUILD.json").read_bytes()
    require(hashlib.sha256(raw_build).hexdigest() == BUILD_SHA256, "frozen v2 BUILD identity changed")
    build = json.loads(raw_build)
    require(build["record"] == "BINDING_SIGNAL_V2_DATA_READINESS", "wrong candidate version")
    input_hashes = {"BUILD.json": BUILD_SHA256}
    data = {}
    for split in ("train", "dev"):
        filename = split.upper() + ".jsonl"
        raw = (DATA / filename).read_bytes()
        input_hashes[filename] = hashlib.sha256(raw).hexdigest()
        require(input_hashes[filename] == build["files_sha256"][filename], "frozen candidate rows changed")
        data[split] = [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]
    generator_path = Path(__file__).with_name("generate.py")
    require(sha(generator_path) == build["generator_sha256"], "frozen v2 generator identity changed")
    spec = importlib.util.spec_from_file_location("binding_signal_v2_frozen_generator", generator_path)
    generate = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = generate
    spec.loader.exec_module(generate)
    audit = generate.audit(data)
    require(audit["rows"] == {"train": 768, "dev": 384}, "frozen default row inventory changed")
    native = {name: sha(PILOT / name) for name in NATIVE_FILES}
    require(native == NATIVE_FILES, "qualified native serialization/grouping/source-lock bytes changed")
    compiler_sha = sha(Path(__file__))
    assets = {name: sha(SOURCE / name) for name in SIDECARS}
    require(assets == EXPECTED_SIDECARS, "native tokenizer/source sidecar identity differs")
    versions = {name: importlib.metadata.version(name) for name in ("transformers", "tokenizers")}
    require(versions == {"transformers": "4.57.1", "tokenizers": "0.22.2"}, "pinned tokenizer runtime differs")
    sys.path.insert(0, str(PILOT))
    from experiments.reasoning_posttraining_v1.serialization import compile_row
    from experiments.model_quality_gpu_v1.data import group_updates
    require(Path(compile_row.__code__.co_filename).resolve() == (PILOT / next(iter(NATIVE_FILES))).resolve(),
            "serialization imported from a different checkout")
    from transformers import AutoTokenizer
    no_model_runtime()
    tokenizer = AutoTokenizer.from_pretrained(str(SOURCE), local_files_only=True,
                                              trust_remote_code=False, use_fast=True)
    require(tokenizer.is_fast and tokenizer.eos_token_id == 151645, "native fast tokenizer/EOS differs")
    no_model_runtime()
    OUT.mkdir()
    totals = {}
    for split, rows in data.items():
        lengths, prompt_lengths, supervised, final_lengths, compiled = [], [], [], [], []
        categories = Counter()
        policy_path = OUT / ("TRAIN_COMPILED.jsonl" if split == "train" else "DEV_INPUTS.jsonl")
        with policy_path.open("x", encoding="utf-8", newline="\n") as stream:
            gold = (OUT / "DEV_GOLD.jsonl").open("x", encoding="utf-8", newline="\n") if split == "dev" else None
            try:
                for row in rows:
                    require([m["role"] for m in row["messages"]] == ["system", "user", "assistant"], "invalid native message roles")
                    prompt = row["messages"][:2]
                    require(row["messages"][2]["content"] == row["reference"], "assistant target differs")
                    value = compile_row(prompt, row["reference"], tokenizer, max_sequence=2048, completion_reserve=128)
                    require(value["labels"][0] == -100 and value["truncated"] is False, "wrong shifted mask or truncation")
                    n = sum(token != -100 for token in value["labels"][1:])
                    require(n == value["supervised_tokens"] and value["labels"][value["answer_end"]] == 151645
                            and all(token == -100 for token in value["labels"][value["answer_end"] + 1:]),
                            "shifted response/EOS supervision differs")
                    lengths.append(len(value["ids"])); prompt_lengths.append(len(value["prompt_ids"]))
                    supervised.append(n); final_lengths.append(value["final_content_tokens"])
                    categories[row["category"]] += 1
                    if split == "train":
                        output = {"id": row["id"], "parent_id": row["parent_id"], "source_split": "train",
                                  "family": row["family"], "category": row["category"], "prompt": prompt,
                                  "reference": row["reference"], **value}
                        compiled.append(output)
                    else:
                        output = {"id": row["id"], "parent_id": row["parent_id"], "prompt": prompt,
                                  "prompt_ids": value["prompt_ids"]}
                        require(set(output) == {"id", "parent_id", "prompt", "prompt_ids"}, "DEV gold leaked into policy inputs")
                        gold.write(json.dumps({"id": row["id"], "parent_id": row["parent_id"],
                                              "category": row["category"], "reference": row["reference"]}, sort_keys=True) + "\n")
                    stream.write(json.dumps(output, sort_keys=True, ensure_ascii=False) + "\n")
            finally:
                if gold is not None:
                    gold.close()
        totals[split] = {"rows": len(rows), "worlds": len({row["parent_id"] for row in rows}),
            "categories": dict(categories), "prompt_tokens": sum(prompt_lengths), "teacher_forced_tokens": sum(lengths),
            "supervised_response_plus_eos_tokens": sum(supervised), "max_prompt_tokens": max(prompt_lengths),
            "max_full_tokens": max(lengths), "max_answer_content_tokens": max(final_lengths), "truncated_rows": 0}
        if split == "train":
            groups = group_updates(compiled, 2048)
            totals[split]["native_2048_token_update_groups"] = len(groups)
            write_json(OUT / "TRAIN_UPDATE_GROUPS.json", {"schema": 1, "record": "NATIVE_2048_GROUP_INVENTORY_ONLY",
                "update_tokens": 2048, "row_order": [row["id"] for row in compiled], "update_groups": groups,
                "processed_tokens": sum(lengths), "supervised_tokens": sum(supervised), "training_permission": False})
    require(all(sha(DATA / name) == value for name, value in input_hashes.items()), "frozen data/BUILD changed during compilation")
    require({name: sha(SOURCE / name) for name in SIDECARS} == assets, "tokenizer sidecars changed during compilation")
    require({name: sha(PILOT / name) for name in NATIVE_FILES} == native
            and sha(generator_path) == build["generator_sha256"] and sha(Path(__file__)) == compiler_sha,
            "compiler/generator/native helper bytes changed during compilation")
    no_model_runtime()
    exclusion_receipts = {}
    for name in ("OPEN_EXCLUSION_AUDIT_V2.json", "SPENT_LC_TEXT_CONTACT_AUDIT.json"):
        path = DATA / name
        if path.is_file():
            record = json.loads(path.read_text(encoding="utf-8"))
            exclusion_receipts[name] = {"sha256": sha(path), "status": record.get("status"),
                                        "interpretation": "existing scoped audit only; not a training permit"}
    report = {"schema": 1, "record": "BINDING_SIGNAL_V2_TOKEN_AUDIT",
        "status": "TOKENIZER_COMPILED_NOT_TRAINING_AUTHORIZATION", "source": str(SOURCE),
        "source_sidecars_sha256_before": assets, "source_sidecars_sha256_after": assets,
        "input_sha256_before": input_hashes, "input_sha256_after": input_hashes,
        "output_sha256": {path.name: sha(path) for path in OUT.iterdir() if path.is_file()},
        "native_checkout": str(PILOT), "native_helpers_sha256": native, "tokenizer_runtime": versions,
        "generator_sha256": sha(generator_path), "compiler_sha256": compiler_sha, "totals": totals,
        "build_audit": audit, "scoped_exclusion_receipts": exclusion_receipts,
        "unresolved_external_coverage": ["legacy LC stopword/8-gram implementation", "approved reserved exclusion indices"],
        "model_weights_read": False, "model_weights_loaded": False, "model_forward_calls": 0,
        "model_runtime_imported": False, "gpu_visible": False, "dev_gold_separate_from_policy_inputs": True,
        "dev_policy_input_fields": ["id", "parent_id", "prompt", "prompt_ids"],
        "human_data_review": "NOT_PERFORMED", "training_permission": False,
        "quality_claim": False, "release_permitted": False}
    write_json(OUT / "TOKEN_AUDIT.json", report)
    print(json.dumps({"status": report["status"], "out": str(OUT), "totals": totals}, indent=2))


if __name__ == "__main__":
    main()
