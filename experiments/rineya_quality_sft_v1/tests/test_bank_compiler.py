"""Small authored fixtures only; no real owner bank/admission or model calls."""
import copy
import json
from pathlib import Path
import pytest

from experiments.model_quality_gpu_v1.integrity import digest, file_hash, seal, unseal
from experiments.rineya_quality_sft_v1 import bank as B
from experiments.rineya_quality_sft_v1.compiler import compile_plan, targets, validate_plan
from experiments.rineya_quality_sft_v1.contract import DATASET_OWNER, DEV_COUNTS, TRAIN_COUNTS


class FixtureTokenizer:
    """Lossless character fixture with explicit final stub, EOS and full history."""
    eos_token, eos_token_id, chat_template = "<EOS>", 2, "RQSFT fixture only"
    stub = "assistant:\n<think>\n\n</think>\n\n"
    def encode(self, text, **kwargs):
        ids = []
        while text:
            if text.startswith(self.eos_token):
                ids.append(self.eos_token_id)
                text = text[len(self.eos_token):]
            else:
                ids.append(ord(text[0]) + 10)
                text = text[1:]
        return ids
    def decode(self, ids, skip_special_tokens=False):
        return "".join(("" if skip_special_tokens else self.eos_token) if i == 2 else chr(i - 10) for i in ids)
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert kwargs.get("enable_thinking") is False
        history = messages if add_generation_prompt else messages[:-1]
        text = "".join(m["role"] + ":" + m["content"] + (self.eos_token if m["role"] == "assistant" else "") + "\n"
                       for m in history) + self.stub
        if not add_generation_prompt:
            text += messages[-1]["content"] + self.eos_token + "\n"
        return self.encode(text) if tokenize else text


def fixture_rows():
    result = {}
    ancestry = [{"status": "FIXTURE_ONLY", "role": "open_fixture", "protected": False, "sha256": "d" * 64}]
    for split, counts in (("train", TRAIN_COUNTS), ("dev", DEV_COUNTS)):
        rows = []
        for category, count in counts.items():
            for i in range(count):
                key = f"{split}-{category}-{i}"
                rows.append({"id": key, "parent_id": key, "parent_size": 1, "pair_id": None,
                             "split": split, "category": category, "family": f"{split}-family-{category}",
                             "scenario_id": key, "lexicon_id": f"{split}-lexicon", "admission": "FIXTURE_ONLY",
                             "ancestry": copy.deepcopy(ancestry), "messages": [{"role": "user", "content": "Fixture " + key},
                                                                                  {"role": "assistant", "content": "A"}],
                             "final_target_verified": True, "empty_target_verified": False,
                             "evaluation": {"dimension": "solving", "scorer": "exact", "answer": "A",
                                            "answerable": True, "critical": False}})
                row = rows[-1]
                if (split == "train" and category == "grounded") or (split == "dev" and category == "grounding_contracts" and i < 64):
                    pair = f"{split}-evidence-pair-{i // 2}"
                    row.update(parent_id=pair, parent_size=2, pair_id=pair, scenario_id=pair)
                    row["evaluation"].update(dimension="grounding_answerability", answerable=i % 2 == 0)
                    if i % 2:
                        row["messages"][-1]["content"] = "UNKNOWN"
                        row["evaluation"]["answer"] = "UNKNOWN"
                elif split == "dev" and category == "grounding_contracts":
                    row["evaluation"]["dimension"] = "output_contract"
        result[split] = rows
    return result


def fixture_bundle(tmp_path, banks=None):
    banks = fixture_rows() if banks is None else banks
    files = {}
    for split, rows in banks.items():
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
        files[split] = {"path": path.name, "sha256": file_hash(path), "rows": len(rows)}
    manifest = {"schema": "RQSFT_OWNER_BUNDLE_V1", "experiment": "RQSFT-001", "owner": DATASET_OWNER,
                "export_id": "AUTHORED_FIXTURE_NOT_OWNER_ADMISSION", "fixture_only": True, "admission": "FIXTURE_ONLY",
                "files": files, "metadata": {"ancestry": banks["train"][0]["ancestry"], "reviews": []}}
    manifest["review_manifest_sha256"] = B.export_binding(manifest)
    path = tmp_path / "fixture-manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, manifest


def fixture_plan(tmp_path, banks=None, tokenizer=None):
    path, _ = fixture_bundle(tmp_path, banks)
    bank = B.read_owner_bundle(path, allow_fixture=True)
    return compile_plan(bank, tokenizer or FixtureTokenizer(), {"fixture": True}, {"sha256": "e" * 64}, fixture=True)


def test_all_rows_quotas_seeded_groups_no_subset(tmp_path):
    plan = fixture_plan(tmp_path)
    body = validate_plan(plan)
    assert body["kind"] == "fixture" and body["admission"]["fixture_only"]
    assert len(body["train"]) == 200 and len(body["dev"]) == 512
    assert sorted(i for group in body["groups"] for i in group) == list(range(200))
    assert body["processed_tokens"] == sum(len(r["ids"]) for r in body["train"]) <= 25000
    assert all(sum(len(body["train"][i]["ids"]) for i in group) <= 2048 for group in body["groups"])
    other = tmp_path / "repeat"
    other.mkdir()
    repeated = validate_plan(fixture_plan(other))
    assert [r["id"] for r in body["train"]] == [r["id"] for r in repeated["train"]]


def test_full_history_and_final_eos_masks_and_verified_empty(tmp_path):
    rows = fixture_rows()
    rows["train"][0]["messages"] = [{"role": "system", "content": "Contract."}, {"role": "user", "content": "Earlier question."},
                                   {"role": "assistant", "content": "Earlier answer."}, {"role": "user", "content": "Final question."},
                                   {"role": "assistant", "content": "A"}]
    rows["train"][1]["messages"][-1]["content"] = ""
    rows["train"][1]["empty_target_verified"] = True
    rows["train"][1]["evaluation"]["answer"] = ""
    body = validate_plan(fixture_plan(tmp_path, rows))
    first = next(r for r in body["train"] if r["id"] == rows["train"][0]["id"])
    decoded = FixtureTokenizer().decode(first["prompt_ids"])
    assert "Earlier answer." in decoded and "Final question." in decoded
    response = targets(first, "response")
    assert all(t == -100 for t in response[:first["answer_start"]])
    assert response[first["answer_end"]] == 2 and response[-1] == -100
    whole = targets(first, "whole")
    assert whole == [-100] + first["ids"][1:]
    empty = next(r for r in body["train"] if r["id"] == rows["train"][1]["id"])
    assert empty["answer_start"] == empty["answer_end"]
    assert [t for t in targets(empty, "response") if t != -100] == [2]


def test_pair_linked_parents_and_complete_parent_atomicity(tmp_path):
    rows = fixture_rows()
    for row in rows["train"][:2]:
        row.update(parent_id="fixture-pair-parent", parent_size=2, pair_id="pair-1")
    body = validate_plan(fixture_plan(tmp_path, rows))
    indices = [i for i, r in enumerate(body["train"]) if r["pair_id"] == "pair-1"]
    assert any(set(indices).issubset(group) for group in body["groups"])
    assert max(indices) - min(indices) == 1


@pytest.mark.parametrize("field", ["family", "scenario_id", "lexicon_id", "parent_id"])
def test_cross_split_structural_overlap_rejected(field):
    rows = fixture_rows()
    rows["dev"][0][field] = rows["train"][0][field]
    with pytest.raises(ValueError, match="overlap"):
        B.validate_rows(rows, True)


def test_digit_blind_overlap_and_missing_target_cannot_pass():
    rows = fixture_rows()
    rows["train"][0]["messages"][0]["content"] = "NFKC task 123"
    rows["dev"][0]["messages"][0]["content"] = "ＮＦＫＣ task 999"
    with pytest.raises(ValueError, match="digitblind"):
        B.validate_rows(rows, True)
    rows = fixture_rows()
    rows["train"][0]["messages"][-1]["content"] = ""
    with pytest.raises(ValueError, match="empty target"):
        B.validate_rows(rows, True)


@pytest.mark.parametrize("mutation", ["quota", "missing_row", "incomplete_parent", "incomplete_pair", "test_ancestry", "no_gold"])
def test_admission_preserves_exact_export_and_refuses_incomplete(mutation):
    rows = fixture_rows()
    if mutation == "quota": rows["train"][0]["category"] = "solving"
    if mutation == "missing_row": rows["train"].pop()
    if mutation == "incomplete_parent": rows["train"][0]["parent_size"] = 3
    if mutation == "incomplete_pair": rows["train"][0]["pair_id"] = "singleton-pair"
    if mutation == "test_ancestry": rows["train"][0]["ancestry"][0]["status"] = "TEST_ONLY"
    if mutation == "no_gold": del rows["dev"][0]["evaluation"]["answer"]
    with pytest.raises(ValueError):
        B.validate_rows(rows, True)


@pytest.mark.parametrize("mutation", ["unpaired_train", "same_answerability", "unpaired_dev", "mismatched_dimension"])
def test_required_grounded_and_dev_evidence_pairs(mutation):
    rows = fixture_rows()
    if mutation == "unpaired_train":
        for row in rows["train"][:2]: row["pair_id"] = None
    if mutation == "same_answerability": rows["train"][1]["evaluation"]["answerable"] = True
    if mutation == "unpaired_dev":
        pair = [r for r in rows["dev"] if r["pair_id"] == "dev-evidence-pair-0"]
        for row in pair: row["pair_id"] = None
    if mutation == "mismatched_dimension": rows["train"][1]["evaluation"]["dimension"] = "solving"
    with pytest.raises(ValueError):
        B.validate_rows(rows, True)


def test_fixture_cannot_be_native_and_metadata_blocks_payload_access(tmp_path, monkeypatch):
    path, manifest = fixture_bundle(tmp_path)
    with pytest.raises(ValueError, match="Fixture-only"):
        B.read_owner_bundle(path)
    bundle = B.read_owner_bundle(path, allow_fixture=True)
    with pytest.raises(ValueError, match="Fixture admission"):
        compile_plan(bundle, FixtureTokenizer(), {}, {"sha256": "e" * 64}, fixture=False)
    manifest["metadata"]["ancestry"][0]["status"] = "EXCLUDED"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    monkeypatch.setattr(B, "file_hash", lambda p: pytest.fail("Payload hash/read must not happen"))
    with pytest.raises(ValueError, match="ancestry"):
        B.read_owner_bundle(path, allow_fixture=True)


def test_declared_approval_without_attributable_receipt_rejected_before_payload(tmp_path, monkeypatch):
    path, manifest = fixture_bundle(tmp_path)
    manifest.update(fixture_only=False, admission="ADMITTED")
    manifest["metadata"]["ancestry"][0]["status"] = "ADMITTED"
    binding = B.export_binding(manifest)
    manifest["review_manifest_sha256"] = binding
    reviews = []
    for kind in sorted(B.REVIEW_KINDS):
        # Deliberately invalid fixture receipt: author reviews own export.
        receipt = {"schema": "RQSFT_OWNER_REVIEW_V1", "kind": kind, "owner_export_binding_sha256": binding,
                   "status": "APPROVED", "reviewer_id": "fixture-author", "exporter_id": "fixture-author",
                   "reviewer_role": "independent", "checks": {kind: True},
                   "attribution": {"principal": "fixture-author", "record_reference": "invalid synthetic fixture"}}
        review_path = tmp_path / (kind + ".json")
        review_path.write_text(json.dumps(receipt), encoding="utf-8")
        reviews.append({"kind": kind, "path": review_path.name, "sha256": file_hash(review_path),
                        "reviewer_id": "fixture-author", "reviewer_role": "independent", "status": "APPROVED"})
    manifest["metadata"]["reviews"] = reviews
    path.write_text(json.dumps(manifest), encoding="utf-8")
    original_hash = B.file_hash
    def checked_hash(p):
        assert Path(p).suffix != ".jsonl", "Unreviewed payload read"
        return original_hash(p)
    monkeypatch.setattr(B, "file_hash", checked_hash)
    with pytest.raises(ValueError, match="Receipt contents"):
        B.read_owner_bundle(path)


@pytest.mark.parametrize("failure", ["owner", "duplicate_manifest", "malformed_receipt", "duplicate_receipt"])
def test_malformed_metadata_never_opens_payload(failure, tmp_path, monkeypatch):
    path, manifest = fixture_bundle(tmp_path)
    if failure == "owner":
        manifest["owner"] = dict(DATASET_OWNER, thread_id="competing-fixture-owner")
    elif failure == "duplicate_manifest":
        path.write_text('{"schema":"RQSFT_OWNER_BUNDLE_V1","schema":"duplicate"}', encoding="utf-8")
    else:
        manifest.update(fixture_only=False, admission="ADMITTED")
        manifest["metadata"]["ancestry"][0]["status"] = "ADMITTED"
        manifest["review_manifest_sha256"] = B.export_binding(manifest)
        reviews = []
        for kind in sorted(B.REVIEW_KINDS):
            receipt = tmp_path / (kind + "-invalid.json")
            receipt.write_text("[malformed" if failure == "malformed_receipt" else '{"kind":"gold","kind":"duplicate"}', encoding="utf-8")
            reviews.append({"kind": kind, "path": receipt.name, "sha256": file_hash(receipt),
                            "reviewer_id": "independent-invalid-fixture", "reviewer_role": "independent", "status": "APPROVED"})
        manifest["metadata"]["reviews"] = reviews
    if failure != "duplicate_manifest": path.write_text(json.dumps(manifest), encoding="utf-8")
    original = B.file_hash
    def checked_hash(p):
        assert Path(p).suffix != ".jsonl", "Payload must remain unread"
        return original(p)
    monkeypatch.setattr(B, "file_hash", checked_hash)
    with pytest.raises(ValueError):
        B.read_owner_bundle(path, allow_fixture=failure in {"owner", "duplicate_manifest"})


def test_native_scope_rejects_false_exact_source_fields(tmp_path):
    body = unseal(fixture_plan(tmp_path))
    body["kind"] = "native_plan"
    body["admission"]["fixture_only"] = False
    body["source_identity"] = {"core_id": "wrong-source", "weight_sha256": "f" * 64, "parameters": 596049920}
    with pytest.raises(ValueError, match="exact-source"):
        validate_plan(seal(body))


def test_input_reparse_guard_is_called_before_path_resolution(monkeypatch, tmp_path):
    calls = []
    def refuse(path):
        calls.append(Path(path))
        raise ValueError("Fixture simulated reparse ancestor")
    monkeypatch.setattr(B.I, "check_input_path", refuse)
    with pytest.raises(ValueError, match="reparse"):
        B.read_owner_bundle(tmp_path / "alias" / "manifest.json", allow_fixture=True)
    assert calls == [tmp_path / "alias" / "manifest.json"]


@pytest.mark.parametrize("kind", ["train_sequence", "total_tokens", "dev_prompt", "dev_target"])
def test_no_truncation_padding_subset_or_budget_escape(kind, tmp_path):
    rows = fixture_rows()
    if kind == "train_sequence": rows["train"][0]["messages"][0]["content"] += "x" * 600
    if kind == "total_tokens":
        for row in rows["train"]: row["messages"][0]["content"] += "x" * 120
    if kind == "dev_prompt": rows["dev"][0]["messages"][0]["content"] += "x" * 2048
    if kind == "dev_target": rows["dev"][0]["messages"][-1]["content"] = "x" * 64
    with pytest.raises(ValueError):
        fixture_plan(tmp_path, rows)


def test_native_tokenization_mismatch_rejected(tmp_path):
    class Mismatch(FixtureTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            result = super().apply_chat_template(messages, **kwargs)
            return result + [99] if kwargs.get("tokenize") else result
    with pytest.raises(ValueError, match="token identities"):
        fixture_plan(tmp_path, tokenizer=Mismatch())


def test_plan_scope_design_and_group_tampering_rejected(tmp_path):
    plan = fixture_plan(tmp_path)
    broken = copy.deepcopy(plan)
    broken["processed_tokens"] -= 1
    with pytest.raises(ValueError, match="digest mismatch"):
        validate_plan(broken)
    body = copy.deepcopy(unseal(plan))
    body["groups"][0].pop()
    with pytest.raises(ValueError, match="grouping changed"):
        validate_plan(seal(body))
    body = unseal(plan)
    body["design"]["max_updates"] = 33
    with pytest.raises(ValueError, match="fixed RQSFT design"):
        validate_plan(seal(body))
