"""Independent stdlib ByteLevel input-ID/render/shape audit; no tokenizer runtime."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

C = Path(__file__).resolve().parent.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""): h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    first_path = C / "runs/bulk-tokenizer-binding-001/BULK_TOKENIZER_BINDING.json"
    path = C / "runs/bulk-tokenizer-binding-002/BULK_TOKENIZER_BINDING.json"
    first, bound = read(first_path), read(path)
    producer = C / "bulk_binding.py"
    assert bound["binding_source_sha256"] == sha(producer)
    archived = C / "versions/bulk_binding_v1/bulk_binding.py"
    assert first["binding_source_sha256"] == sha(archived)
    packet = C / "runs/bulk-bank-apparatus-001"
    bank = {row["id"]: row for row in read(packet / "bank.json")}
    manifest = read(packet / "manifest.json")
    assert bound["bank_sha256"] == sha(packet / "bank.json") == first["bank_sha256"]
    assert bound["manifest_sha256"] == sha(packet / "manifest.json")
    assert bound["normalizer_source_sha256"] == sha(C / "open_bank.py") == manifest["normalizer_source_sha256"]
    assert bound["generator_scorer_sha256"] == sha(C / "bulk_bank.py")
    metadata = read(C / "runs/binding-001/MODEL_DATA_BINDING.json")
    assert bound["canonical_source"] == metadata["source_path"] and bound["checkpoint_files"] == metadata["checkpoint_files"]
    for field in ("python", "torch", "transformers"):
        assert bound["runtime"][field] == metadata["runtime"][field]
    tokenizer_path = Path(bound["canonical_source"]) / "tokenizer.json"
    assert sha(tokenizer_path) == bound["checkpoint_files"]["tokenizer.json"]
    tokenizer = read(tokenizer_path)
    assert tokenizer["model"]["type"] == "BPE" and tokenizer["decoder"]["type"] == "ByteLevel"
    assert tokenizer["normalizer"]["type"] == "NFC"
    ordinary = {identifier: token for token, identifier in tokenizer["model"]["vocab"].items()}
    added = {row["id"]: row["content"] for row in tokenizer["added_tokens"]}
    base = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    codes, extra = base[:], 0
    for value in range(256):
        if value not in base:
            base.append(value); codes.append(256 + extra); extra += 1
    byte_decoder = {chr(char): byte for byte, char in zip(base, codes)}
    def decode(ids):
        parts = []
        for identifier in ids:
            if identifier in added: parts.append(added[identifier].encode("utf-8"))
            else: parts.append(bytes(byte_decoder[char] for char in ordinary[identifier]))
        return b"".join(parts).decode("utf-8")
    rows = bound["rows"]
    assert len(rows) == len(bank) == 288 and {row["case_id"] for row in rows} == set(bank)
    first_rows = {row["case_id"]: row for row in first["rows"]}
    templates, shape_sums = set(), 0
    for row in rows:
        raw = bank[row["case_id"]]
        assert row["raw_prompt"] == raw["prompt"]
        assert row["rendered_prompt"].count(raw["prompt"]) == 1
        templates.add(tuple(row["rendered_prompt"].split(raw["prompt"])))
        assert row["case_id"] not in row["rendered_prompt"] and row["parent_id"] not in row["rendered_prompt"]
        assert decode(row["input_token_ids"]) == unicodedata.normalize("NFC", row["rendered_prompt"])
        count = len(row["input_token_ids"])
        assert all(type(identifier) is int and 0 <= identifier < bound["tokenizer_length"] for identifier in row["input_token_ids"])
        assert count == row["input_tokens"] and 1 <= count <= 350
        assert row["max_new_tokens"] == 8 and row["max_forward_prefix_tokens"] == count + 7
        assert row["max_processed_prefix_positions"] == 8*count + 28
        shape_sums += row["max_processed_prefix_positions"]
        for field in ("parent_id", "split", "generator_family", "category", "difficulty_stratum", "variant_id"):
            assert row[field] == raw[field]
        assert row["input_token_ids"] == first_rows[row["case_id"]]["input_token_ids"]
        assert row["rendered_prompt"] == first_rows[row["case_id"]]["rendered_prompt"]
    assert len(templates) == 1
    assert bound["shapes"] == {"batch": 1, "padding": False, "cache": False, "max_input_tokens": 232,
                                "max_output_tokens": 8, "max_forward_prefix_tokens": 239}
    assert bound["native_model_loads"] == first["native_model_loads"] == 0
    assert bound["world_parents"] == 144 and bound["development_parents"] == 96 and bound["validation_parents"] == 48
    receipt = read(path.parent / "receipt.json")
    assert receipt["binding_sha256"] == sha(path) and receipt["native_model_loads"] == 0
    assert "torch" not in sys.modules and "transformers" not in sys.modules
    result = {"schema_version": "1.0", "created_utc": datetime.now(timezone.utc).isoformat(), "reviewer": "/root/adversarial_review",
        "status": "PASS_BULK_NO_MODEL_BINDING_SOURCE_AND_RAW_ID_AUDIT", "binding_path": str(path), "binding_sha256": sha(path),
        "producer_sha256": sha(producer), "archived_producer_v1_sha256": sha(archived), "first_binding_sha256": sha(first_path),
        "tokenizer_asset_sha256": sha(tokenizer_path), "review_script_sha256": sha(__file__),
        "independent_decoded_input_id_rows": 288, "rendered_input_includes_metadata_ID": False,
        "first_and_second_binding_all_rendered_IDs_equal": True, "shapes": bound["shapes"],
        "bulk_generation_position_cap": 11*shape_sums, "generation_position_sum_per_one_policy": shape_sums,
        "evidence_scope": "independent ByteLevel ID decoding and exact input/render/shape arithmetic; canonical encoding lineage from reviewed root-only offline tokenizer producer",
        "native_model_loads": 0, "native_calls": 0, "real_GPU_inventory_calls": 0,
        "limitations": ["Decoded IDs equal complete rendered user prompts; no truncation, future continuation or metadata IDs added",
            "Model weights were not loaded; this is tokenizer binding, not max357 native mechanical qualification",
            "Current source closes reused normalizer hash before module import; v1 binding001 and producer remain preserved",
            "Fresh parent7GiB RAM/6GiB GPU and native source/state/mechanics/cost gates remain required"], "unresolved_binding_blockers": []}
    with (C / "reviews/bulk_binding_cpu_receipt_001.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True); stream.write("\n")
    print(json.dumps({"status": result["status"], "binding_sha256": sha(path), "decoded_rows": 288, "max_input": 232,
                      "bulk_generation_position_cap": result["bulk_generation_position_cap"]}))


if __name__ == "__main__": main()
