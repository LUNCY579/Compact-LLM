# Rineya Supa-A1

Supa-A1 is the first A-series tracked abstention core in the Rineya family. It
promotes the evaluated checkpoint formerly called
`Rineya-Nimbus-Abstain-1.0` without changing its weights or behavior.

## Results

| Split | Role | Strict exact |
|---|---|---:|
| Showcase64 | threshold selection / checkpoint evaluation | **43/64** |
| Stability192 | previously inspected validation; S4 added post-confirmation | **125/192** |
| All256 | combined, not independent | **168/256** |

The offline S4 gate scored 46/64 on Showcase and 125/192 on Stability. The
published checkpoint itself is 43/64 and 125/192; it does not inherit the
offline tuning score. Sealed TEST has not been run.

## What changed

The dense tensor is Nimbus 1.6 plus the Stage-A step-200 LoRA at scale 0.8125,
merged in FP32 and stored in BF16. A custom Qwen3 class adds
`5.156249985098839` to token 51378 (`UNKNOWN`) before greedy argmax. The rule is
exact for `logP(UNKNOWN) - logP(top1)` because the shared softmax normalizer
cancels. The weight SHA-256 is `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b` and is byte-identical to Supa
Core 1.7; the behavioral difference is the model class and config.

## Load

Local standalone-directory load:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "Rineya-Supa-A1-1.0.0-devfreeze"
tokenizer = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(
    path, dtype="bfloat16", trust_remote_code=True
)
```

Private family-vault load (Transformers 5.14 resolves custom code at repository
root, so the root runtime module is intentionally mirrored):

```python
repo = "Luncy1978/rineya-published-cores"
subfolder = "Rineya-Supa-A1-1.0.0-devfreeze"
tokenizer = AutoTokenizer.from_pretrained(repo, subfolder=subfolder)
model = AutoModelForCausalLM.from_pretrained(
    repo,
    subfolder=subfolder,
    dtype="bfloat16",
    trust_remote_code=True,
)
```

Use greedy decoding (`do_sample=False`). The custom code requirement and the
fact that the bias is applied at every decode step are material compatibility
constraints, not optional implementation details.

## No-bias control and naming

The former `Rineya-Nimbus-Dense-1.0-nobias` checkpoint is retained under
`evidence/` as Supa-A1's ablation control. It is not published as `Nimbus-A1`:
its weights are identical to Supa Core 1.7, so that name would collide with the
distinct Nimbus 1.6 lineage. `Nimbus-A1` is instead a tracking label for the
actual Nimbus 1.6 core.

See `CORE.json`, `FILES.json`, `SOURCE_CHECKPOINT.json`, and `evidence/` for
machine-readable provenance and frozen predictions. `EVALUATION_MANIFEST.json`
explicitly records that the exact runner revision used to generate the frozen
predictions was not durably captured; their files and scoring inputs are still
hash-pinned.
