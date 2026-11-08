"""Torch mechanisms are tested locally; optional HF tests never silently pass."""
import copy
from types import SimpleNamespace
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn
from rgi import native, data
from rgi.io import digest, seal
from conftest import make_bank

torch.set_num_threads(1)


class CausalBackbone(nn.Module):
    def __init__(self, vocab=256, width=8):
        super().__init__()
        self.embed_tokens = nn.Embedding(vocab, width)
    def forward(self, input_ids, attention_mask=None, use_cache=False):
        emb = self.embed_tokens(input_ids)
        pos = torch.arange(1, emb.shape[1] + 1, device=emb.device)[None, :, None]
        return SimpleNamespace(last_hidden_state=emb.cumsum(dim=1) / pos)


class LoRAHead(nn.Module):
    def __init__(self, width=8, vocab=256):
        super().__init__()
        self.base = nn.Linear(width, vocab, bias=False)
        self.lora_A = nn.Linear(width, 2, bias=False)
        self.lora_B = nn.Linear(2, vocab, bias=False)
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.base(x) + self.lora_B(self.lora_A(x))


class TorchMechanismFixture(nn.Module):
    """Causal mean-prefix fixture, explicitly NOT a Qwen3/Rineya simulation."""
    def __init__(self):
        super().__init__()
        self.model = CausalBackbone()
        self.lm_head = LoRAHead()
        for name, p in self.named_parameters():
            p.requires_grad_("lora_" in name)


@pytest.fixture
def model():
    torch.manual_seed(17)
    return TorchMechanismFixture()


def full_reference(model, item):
    ids = torch.tensor([item["input_ids"]])
    hidden = model.model(input_ids=ids).last_hidden_state
    logits = model.lm_head(hidden)
    all_losses = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 256), ids[:, 1:].reshape(-1), reduction="none")
    mask = torch.tensor([r != "prompt" for r in item["roles"][1:]])
    return all_losses[mask]


@pytest.mark.parametrize("chunk_size", [1, 2, 16, 100])
def test_chunked_next_token_nll_matches_full_logits(model, chunk_size):
    item = make_bank()["items"][0]
    actual = native.answer_losses(model, item, "cpu", chunk_size)
    expected = full_reference(model, item)
    assert torch.allclose(actual, expected, atol=1e-6)
    assert actual.numel() == 2


def test_first_target_does_not_see_itself(model):
    item = make_bank()["items"][0]
    ids = torch.tensor([item["input_ids"]])
    changed = ids.clone()
    changed[0, 2] = 99
    h1 = model.model(input_ids=ids).last_hidden_state
    h2 = model.model(input_ids=changed).last_hidden_state
    assert torch.equal(h1[:, 1], h2[:, 1])
    assert not torch.equal(h1[:, 2], h2[:, 2])


def test_chunked_gradients_match_full_logits(model):
    other = copy.deepcopy(model)
    item = make_bank()["items"][0]
    native.answer_losses(model, item, "cpu", 1).mean().backward()
    full_reference(other, item).mean().backward()
    for (n, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        if p.requires_grad:
            assert torch.allclose(p.grad, q.grad, atol=1e-7), n
        else:
            assert p.grad is None


@pytest.mark.parametrize("accumulation", [1, 3])
def test_optimizer_frozen_host_and_exact_dose(model, accumulation):
    bank = make_bank(split="train")
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    order = list(range(2 * accumulation))
    settings = {"updates": 2, "gradient_accumulation": accumulation, "learning_rate": .01, "max_grad_norm": 1}
    receipt = native.optimize(model, bank, order, settings, "cpu")
    assert receipt["processed_tokens"] == 8 * accumulation
    assert receipt["supervised_tokens"] == 4 * accumulation
    assert receipt["updates"] == 2
    changed = []
    for n, p in model.named_parameters():
        if not torch.equal(before[n], p):
            changed.append(n)
            assert "lora_" in n
    assert changed
    assert all(np.isfinite(x["mean_nll"]) for x in receipt["trace"])


def test_optimizer_refuses_host_parameters(model):
    model.model.embed_tokens.weight.requires_grad_(True)
    settings = {"updates": 1, "gradient_accumulation": 1, "learning_rate": .01, "max_grad_norm": 1}
    with pytest.raises(ValueError, match="only LoRA"):
        native.optimize(model, make_bank(split="train"), [0], settings, "cpu")


def test_optimizer_refuses_wrong_step_plan(model):
    settings = {"updates": 2, "gradient_accumulation": 1, "learning_rate": .01, "max_grad_norm": 1}
    with pytest.raises(ValueError, match="plan/order"):
        native.optimize(model, make_bank(split="train"), [0], settings, "cpu")


def test_wrapper_get_base_model(model):
    class Wrapper(nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner
        def get_base_model(self):
            return self.inner
    item = make_bank()["items"][0]
    assert torch.equal(native.answer_losses(model, item, "cpu"), native.answer_losses(Wrapper(model), item, "cpu"))


def test_memory_guard_cannot_be_disabled(monkeypatch):
    import psutil
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=20 * 2**30))
    with pytest.raises(ValueError, match="disabled"):
        native.resource_guard({"evidence_class": "native_model", "device": "cpu", "min_free_ram_gib": 0})
    monkeypatch.setattr(psutil, "virtual_memory", lambda: SimpleNamespace(available=1 * 2**30))
    with pytest.raises(ValueError, match="memory gate"):
        native.resource_guard({"evidence_class": "native_model", "device": "cpu", "min_free_ram_gib": 8})


def test_native_guard_refuses_missing_cuda():
    if torch.cuda.is_available():
        pytest.skip("This negative control requires a CPU-only host")
    with pytest.raises(ValueError, match="CUDA unavailable"):
        native.resource_guard({"evidence_class": "tiny_random_model", "device": "cuda:0", "min_free_ram_gib": 0})


def build_hf_fixture(tmp_path):
    pytest.importorskip("transformers", reason="Optional Transformers unavailable; native loader unqualified")
    pytest.importorskip("tokenizers", reason="Optional tokenizer dependency unavailable")
    from transformers import Qwen3Config, Qwen3ForCausalLM, PreTrainedTokenizerFast
    from tokenizers import Tokenizer, models, pre_tokenizers
    tok = Tokenizer(models.WordLevel({"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}, unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]")
    host = tmp_path / "tiny_host"
    host.mkdir()
    config = Qwen3Config(vocab_size=16, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512)
    torch.manual_seed(17)
    Qwen3ForCausalLM(config).save_pretrained(host, safe_serialization=True)
    tokenizer.save_pretrained(host)
    spec = {"evidence_class": "tiny_random_model", "precision": "float32", "device": "cpu", "model_path": str(host),
            "tokenizer_path": str(host), "model_sha256": native.fingerprint(host, "model"),
            "tokenizer_sha256": native.fingerprint(host, "tokenizer"), "max_length": 512,
            "max_scored_tokens": 10000, "min_free_ram_gib": 0, "adapters": []}
    raw = data.mini_bank(2)
    raw["evidence_class"] = "tiny_random_model"
    raw = seal({k: v for k, v in raw.items() if k != "content_sha256"})
    bank = data.encode_bank(raw, tokenizer, spec["tokenizer_sha256"])
    return spec, bank


@pytest.mark.optional_hf
def test_real_tiny_qwen3_local_loader_and_collector(tmp_path):
    spec, bank = build_hf_fixture(tmp_path)
    result = native.collect(bank, spec)
    assert result["evidence_class"] == "tiny_random_model" and result["completed"]
    assert all(np.isfinite(r["nll"]) for r in result["rows"])
    assert native.fingerprint(spec["model_path"], "model") == spec["model_sha256"]


@pytest.mark.optional_hf
def test_real_tiny_qwen3_unmerged_lora_training(tmp_path):
    pytest.importorskip("peft", reason="Optional PEFT unavailable; native training/stack unqualified")
    spec, bank = build_hf_fixture(tmp_path)
    for item in bank["items"]:
        item["split"] = "train"
    bank = seal({k: v for k, v in bank.items() if k != "content_sha256"})
    spec["training"] = {"optimizer": "adamw", "updates": 2, "gradient_accumulation": 1, "rank": 2,
        "alpha": 4, "learning_rate": .001, "max_grad_norm": 1, "target_modules": ["q_proj", "v_proj"], "seed": 17}
    plan = native.training_plan(bank, spec)
    receipt = native.train(bank, spec, plan, tmp_path / "train_run", execute=True)
    assert receipt["updates"] == 2 and not receipt["quality_improvement_established"]
    assert native.fingerprint(spec["model_path"], "model") == spec["model_sha256"]
    spec["adapters"] = [{"path": str(tmp_path / "train_run" / "adapter"),
                         "sha256": receipt["adapter_sha256"], "coefficient": .25}]
    for item in bank["items"]:
        item["split"] = "development"
    bank = seal({k: v for k, v in bank.items() if k != "content_sha256"})
    result = native.collect(bank, spec)
    assert result["components"] == {receipt["adapter_sha256"]: .25}
