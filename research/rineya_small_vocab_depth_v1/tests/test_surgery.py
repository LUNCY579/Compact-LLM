"""Checkpoint-surgery and architecture invariants on BUILT arms (brief Phase 19).

Needs built arms, listed in SMALLVD_ARMS (os.pathsep-separated directories, each containing
SURGERY.json). Skips cleanly when none are given, so the tokenizer suite still runs on a bare
checkout. CPU / fp32 only: equivalence is asserted BITWISE on hidden states, which needs one
device and one dtype on both sides.

Run:  SMALLVD_ARMS=<a1>;<a2>;<a3> .venv-rab001-cu126/Scripts/python.exe -B -m pytest -p no:cacheprovider tests/test_surgery.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
torch = pytest.importorskip("torch")
import param_accounting as pa  # noqa: E402
import surgery  # noqa: E402
from layer_ablation import keep_layers  # noqa: E402

TEACHER = Path(os.environ.get(
    "SMALLVD_TEACHER",
    r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze"))
ARMS = [Path(p) for p in os.environ.get("SMALLVD_ARMS", "").split(os.pathsep) if p]
pytestmark = pytest.mark.skipif(not ARMS, reason="no built arms given in SMALLVD_ARMS")

TEXTS = [
    "<|im_start|>user\nWhat is the capital of France?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nParis",
    "def add(a, b):\n    return a + b\n",
    "The quick brown fox jumps over the lazy dog. It was the best of times.",
]


def manifest(arm):
    return json.loads((arm / "SURGERY.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_verify_built(arm):
    checks = surgery.verify_built(arm, TEACHER)
    assert checks["all_finite"] and checks["data_section_sha256_reproduces"]


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_architecture_and_exact_parameter_count(arm):
    m = manifest(arm)
    cfg = json.loads((arm / "config.json").read_text(encoding="utf-8"))
    tcfg = json.loads((TEACHER / "config.json").read_text(encoding="utf-8"))
    assert cfg["num_hidden_layers"] == len(m["kept_layers_old_index"])
    assert cfg["vocab_size"] == m["vocab_rows"]
    for k in ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim"):
        assert cfg[k] == tcfg[k]
    r = surgery.reload_checks(arm)
    expect = pa.analytic(tcfg, cfg["vocab_size"], cfg["num_hidden_layers"])["unique_physical_parameters"]
    assert r["unique_params_torch"] == r["unique_params_by_storage"] == expect
    assert r["naive_state_dict_sum"] == expect + cfg["vocab_size"] * cfg["hidden_size"], \
        "the alias sum should exceed the unique count by exactly one embedding matrix"


# bf16 by default: host RAM, not correctness, is the constraint (15.2 GB, shared). Bitwise hidden-
# state equality holds in ANY dtype as long as both sides use the same one; logits are then formed
# in fp32 from those identical hidden states, so the 1e-5 tolerance keeps its meaning.
DTYPE = getattr(torch, os.environ.get("SMALLVD_TEST_DTYPE", "bfloat16"))


def _load(path, dtype=None):
    from transformers import Qwen3ForCausalLM
    return Qwen3ForCausalLM.from_pretrained(str(path), torch_dtype=dtype or DTYPE,
                                            attn_implementation="eager", local_files_only=True).eval()


def _logits(model, h):
    return h.float() @ model.lm_head.weight.float().T


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_forward_equivalence_with_teacher_bypass(arm):
    """The built student IS the teacher with the same blocks bypassed and the same rows gathered.
    Final hidden states must be BITWISE equal on inputs whose teacher tokenisation uses only
    retained tokens; logits equal on the retained ids to fp32 GEMM tolerance."""
    import tokenizers
    m = manifest(arm)
    kept = m["kept_layers_old_index"]
    id_map = surgery.load_id_map(arm) if (arm / "token_id_map.json").exists() else None
    old_tok = tokenizers.Tokenizer.from_file(str(TEACHER / "tokenizer.json"))
    new_tok = tokenizers.Tokenizer.from_file(str(arm / "tokenizer.json"))
    inputs = []
    for s in TEXTS:
        oi = old_tok.encode(s, add_special_tokens=False).ids
        if id_map is not None and not all(i in id_map for i in oi):
            continue
        ni = new_tok.encode(s, add_special_tokens=False).ids
        assert ni == ([id_map[i] for i in oi] if id_map else oi)
        inputs.append((oi, ni))
    assert inputs, "no retained-only input available for the equivalence check"
    teacher = _load(TEACHER)
    ref = []
    with torch.no_grad(), keep_layers(teacher, kept):
        for oi, _ in inputs:
            out = teacher.model(input_ids=torch.tensor([oi]), use_cache=False).last_hidden_state
            ref.append((out.clone(), _logits(teacher, out).clone()))
    del teacher
    student = _load(arm)
    with torch.no_grad():
        for (oi, ni), (h_ref, l_ref) in zip(inputs, ref):
            h = student.model(input_ids=torch.tensor([ni]), use_cache=False).last_hidden_state
            assert torch.equal(h, h_ref), "final hidden states are not bitwise identical"
            l = _logits(student, h)
            if id_map is not None:
                cols = torch.tensor(sorted(id_map, key=lambda o: id_map[o]))
                l_ref = l_ref.index_select(-1, cols)
            else:
                l_ref = l_ref[..., : l.shape[-1]]
            assert (l - l_ref).abs().max().item() <= 1e-5


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_save_reload_parity(arm, tmp_path):
    student = _load(arm)
    student.save_pretrained(tmp_path / "re", safe_serialization=True)
    again = _load(tmp_path / "re")
    sa, sb = student.state_dict(), again.state_dict()
    assert sa.keys() == sb.keys()
    assert all(torch.equal(sa[k], sb[k]) for k in sa)
    assert again.lm_head.weight.untyped_storage().data_ptr() == again.model.embed_tokens.weight.untyped_storage().data_ptr()


@pytest.mark.parametrize("arm", ARMS, ids=lambda p: p.name)
def test_generation_references_no_deleted_ids(arm):
    student = _load(arm)
    cfg = json.loads((arm / "config.json").read_text(encoding="utf-8"))
    gen = json.loads((arm / "generation_config.json").read_text(encoding="utf-8"))
    rows = cfg["vocab_size"]
    for k in ("bos_token_id", "eos_token_id", "pad_token_id"):
        for v in ([gen[k]] if not isinstance(gen.get(k), list) else gen[k]):
            if v is not None:
                assert 0 <= v < rows
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(arm), local_files_only=True)
    text = tok.apply_chat_template([{"role": "user", "content": "Say hello."}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    ids = tok(text, return_tensors="pt", add_special_tokens=False)
    out = student.generate(**ids, max_new_tokens=8, do_sample=False,
                           eos_token_id=gen["eos_token_id"], pad_token_id=gen["pad_token_id"])
    assert int(out.max()) < rows


def test_negative_control_wrong_layer_map_is_caught(tmp_path):
    """verify_built must FAIL on an arm whose manifest claims a different layer set than the
    weights contain. Hard link, not copy: no extra disk."""
    arm = next((a for a in ARMS if len(manifest(a)["kept_layers_old_index"]) < 28), None)
    if arm is None:
        pytest.skip("needs a depth-pruned arm")
    fake = tmp_path / "fake"
    fake.mkdir()
    os.link(arm / "model.safetensors", fake / "model.safetensors")
    for p in arm.iterdir():
        if p.name not in ("model.safetensors", "SURGERY.json"):
            shutil.copyfile(p, fake / p.name)
    m = manifest(arm)
    kept = m["kept_layers_old_index"]
    dropped = next(l for l in range(28) if l not in kept)
    m["kept_layers_old_index"] = sorted(kept[1:] + [dropped])
    (fake / "SURGERY.json").write_text(json.dumps(m), encoding="utf-8")
    with pytest.raises(AssertionError):
        surgery.verify_built(fake, TEACHER)


@pytest.mark.parametrize("arm", [a for a in ARMS if (a / "token_id_map.json").exists()], ids=lambda p: p.name)
def test_virtual_student_equals_built_student_on_arbitrary_text(arm):
    """The layer study measures VIRTUAL students (teacher + bypass + candidate tokenisation mapped
    back to old ids + logits restricted to retained ids). On ANY text -- including text whose
    tokenisation changed -- that must equal the built checkpoint: hidden states bitwise, logits on
    the retained ids to fp32 GEMM tolerance. This is what licenses measuring without building."""
    import virtual_student as vs
    m = manifest(arm)
    kept = m["kept_layers_old_index"]
    voc = vs.Vocab(arm, TEACHER / "tokenizer.json")
    texts = ["中文测试 and naïve café, Dvořák; x ≤ y ⇒ ∑ᵢ aᵢ", "photosynthesis antidisestablishmentarianism"]
    teacher = _load(TEACHER)
    ref = []
    with torch.no_grad(), keep_layers(teacher, kept):
        for s in texts:
            old_ids = voc.encode_old_ids(s)
            h = teacher.model(input_ids=torch.tensor([old_ids]), use_cache=False).last_hidden_state
            ref.append((h.clone(), _logits(teacher, h)[..., voc.retained].clone()))
    del teacher
    student = _load(arm)
    with torch.no_grad():
        for s, (h_ref, l_ref) in zip(texts, ref):
            new_ids = voc.tok.encode(s, add_special_tokens=False).ids
            h = student.model(input_ids=torch.tensor([new_ids]), use_cache=False).last_hidden_state
            assert torch.equal(h, h_ref)
            assert (_logits(student, h) - l_ref).abs().max().item() <= 1e-5
