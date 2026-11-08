import hashlib
import re
import zipfile

import pytest

from experiments.reasoning_posttraining_v1.prepare import PRIMITIVE_QUOTAS, QUOTAS, select_microfit
from experiments.reasoning_posttraining_v1.serialization import assistant_record, compile_row
from experiments.reasoning_posttraining_v1 import sources
from experiments.reasoning_posttraining_v1.views import scope_only


class Tokenizer:
    eos_token = "<|im_end|>"
    eos_token_id = 151645

    def encode(self, text, add_special_tokens=False):
        return [self.eos_token_id if t == self.eos_token else ord(t) + 1000
                for t in re.findall(r"<\|im_end\|>|[\s\S]", text)]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(self.eos_token if t == self.eos_token_id else chr(t - 1000) for t in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=False):
        prompt = "".join("<|im_start|>" + m["role"] + "\n" + m["content"] + self.eos_token + "\n"
                         for m in messages if m["role"] != "assistant")
        prefix = prompt + "<|im_start|>assistant\n"
        if add_generation_prompt:
            text = prefix + ("" if enable_thinking else "<think>\n\n</think>\n\n")
        else:
            m = messages[-1]
            text = prefix + "<think>\n" + m.get("reasoning_content", "") + "\n</think>\n\n" + m["content"] + self.eos_token + "\n"
        return self.encode(text) if tokenize else text


PROMPT = [{"role": "system", "content": "Use only supplied facts."}, {"role": "user", "content": "What is 2+3?"}]


@pytest.mark.parametrize("mode", ["direct", "reasoning"])
def test_native_contract_mask_and_eos(mode):
    tok = Tokenizer()
    r = compile_row(PROMPT, '{"value":5}', tok, mode=mode,
                    trace="Add 2 and 3 to obtain 5." if mode == "reasoning" else None)
    assert all(x == -100 for x in r["labels"][:r["answer_start"]])
    assert r["labels"][r["answer_end"]] == tok.eos_token_id
    assert all(x == -100 for x in r["labels"][r["answer_end"] + 1:])
    assert sum(x != -100 for x in r["labels"]) == r["supervised_tokens"]
    suffix = tok.decode(r["ids"][r["answer_start"]:r["answer_end"]])
    assert suffix.startswith("<think>") if mode == "reasoning" else suffix == '{"value":5}'
    if mode == "reasoning":
        assert "<think>" not in tok.decode(r["prompt_ids"])
        assert suffix.count("<think>") == suffix.count("</think>") == 1


@pytest.mark.parametrize("final,trace,mode", [("", None, "direct"), ("5", "x", "direct"),
    ("5", None, "reasoning"), ("5", "<think>x</think>", "reasoning"), ("<|im_end|>", None, "direct"),
    ("5", None, "unknown"), (" 5", None, "direct")])
def test_invalid_target_rejected(final, trace, mode):
    with pytest.raises(ValueError): assistant_record(final, trace, mode)


def test_reasoning_never_appended_after_empty_stub():
    class Closed(Tokenizer):
        def apply_chat_template(self, messages, **kwargs):
            if kwargs.get("add_generation_prompt"):
                kwargs["enable_thinking"] = False
            return super().apply_chat_template(messages, **kwargs)
    with pytest.raises(ValueError, match="already closed"):
        compile_row(PROMPT, "5", Closed(), mode="reasoning", trace="2+3=5")


@pytest.mark.parametrize("limit,reserve", [(8, 1), (2048, 3000), (True, 128)])
def test_context_refuses_without_truncation(limit, reserve):
    with pytest.raises(ValueError): compile_row(PROMPT, "5", Tokenizer(), max_sequence=limit, completion_reserve=reserve)


def test_native_tokenization_disagreement():
    class Changed(Tokenizer):
        def apply_chat_template(self, *args, **kwargs):
            r = super().apply_chat_template(*args, **kwargs)
            return [777] + r if kwargs.get("tokenize") else r
    with pytest.raises(ValueError, match="token prefix"):
        compile_row(PROMPT, "5", Changed())


def candidate_bank():
    return [{"id": f"{family}/{label}/{i}", "source_split": "train", "family": family,
             "answer_class": label, "direct_eligible": True, "gold_verified": True}
            for family, classes in QUOTAS.items() for label, count in classes.items() for i in range(count + 2)]


def test_selection_is_deterministic_exact_quotas():
    rows = candidate_bank()
    a = select_microfit(rows)
    assert a == select_microfit(list(reversed(rows)))
    assert len(a) == len({r["id"] for r in a}) == 32
    for family, classes in QUOTAS.items():
        for label, count in classes.items():
            assert sum(r["family"] == family and r["answer_class"] == label for r in a) == count
    assert {r["id"] for r in a} != {r["id"] for r in select_microfit(rows, 23)}


@pytest.mark.parametrize("change", ["split", "duplicate", "length", "gold"])
def test_selection_fails_closed(change):
    rows = candidate_bank()
    if change == "split": rows[0]["source_split"] = "validation"
    if change == "duplicate": rows.append(rows[0])
    if change == "length":
        for row in rows: row["direct_eligible"] = False
    if change == "gold":
        for row in rows: row["gold_verified"] = False
    with pytest.raises(ValueError): select_microfit(rows)


def test_primitive_microfit_explicit_source_policy():
    rows = [{"id": f"v1/{family}/{i}", "source_split": "train", "family": family,
             "answer_class": "EXACT_PRIMITIVE", "direct_eligible": True, "gold_verified": True}
            for family in PRIMITIVE_QUOTAS for i in range(7)]
    selected = select_microfit(rows, quotas=PRIMITIVE_QUOTAS)
    assert len(selected) == 32
    assert all(sum(r["family"] == family for r in selected) == 4 for family in PRIMITIVE_QUOTAS)
    with pytest.raises(ValueError): select_microfit(rows)  # never silently substitute this for the v2 design


@pytest.mark.parametrize("case", ["valid", "changed", "unlisted", "traversal", "hash"])
def test_release_import_boundary(tmp_path, monkeypatch, case):
    source = tmp_path / "release.zip"; root = tmp_path / "tree"; root.mkdir()
    rel = "../escape.py" if case == "traversal" else "bundle/core.py"
    with zipfile.ZipFile(source, "w") as z: z.writestr(rel, b"x = 1\n")
    (root / "core.py").write_bytes(b"changed" if case == "changed" else b"x = 1\n")
    if case == "unlisted": (root / "extra.py").write_bytes(b"x = 2\n")
    expected = hashlib.sha256(source.read_bytes()).hexdigest()
    monkeypatch.setitem(sources.ARCHIVES, "fixture", ("bundle", "wrong" if case == "hash" else expected))
    if case == "valid":
        report = sources.verify_release(source, root, "fixture")
        assert report["archive_sha256"] == expected
    else:
        with pytest.raises(ValueError): sources.verify_release(source, root, "fixture")
    assert source.exists()


def test_short_view_preserves_target_and_never_uses_gold():
    p = [{"role": "system", "content": "Rules"}, {"role": "user", "content":
         'SCOPE "W1"\nRecord R1 | scope="W1" | value=5\nRecord R2 | scope="W2" | value=8\nQUERY value'}]
    def parse(prompt):
        return {"records": [s for s in prompt[1]["content"].splitlines() if s.startswith("Record") and 'scope="W1"' in s], "query": "value"}
    shorter, receipt = scope_only(p, parse, lambda x: x)
    assert parse(shorter) == parse(p)
    assert receipt["target_records_preserved"] == receipt["other_scope_records_removed"] == 1
    assert "Record R2" not in shorter[1]["content"]
    assert p[1]["content"].count("Record") == 2
    assert not receipt["minimal_causal_evidence_claim"]


def test_short_view_refuses_ambiguous_public_scope():
    p = [{"role": "system", "content": "Rules"}, {"role": "user", "content": 'SCOPE "W1"\nSCOPE "W2"'}]
    with pytest.raises(ValueError, match="unambiguous"):
        scope_only(p, lambda _: {"records": []}, lambda x: x)
