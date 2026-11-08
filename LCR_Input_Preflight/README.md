# LCR input preflight — 16 September 2026

**Status: input preparation only. No corpus was generated.**

The commission requires real target-tokenizer measurements and exclusion checks against existing evaluation material. These assets were not available in the assistant's working container. A metadata read of the private Hugging Face repository succeeded; its file-download capability was not available through that connector. Public download and local package-install attempts failed. This package does not substitute approximate token counts, guessed evaluation rows, or an acceptance certificate.

## Collect the required inputs locally

The collector uses only the Python standard library. It makes no network requests and never loads model weights. It verifies four tokenizer-asset hashes against the inspected Supa 1.7 source lock, then packages those assets and only the exclusion files explicitly selected on the command line. It does not automatically upload anything. Source files remain unchanged. Do not select sealed evaluation files you are not authorized to share, credentials, model weights, or unrelated private files.

PowerShell example (replace paths with the actual local paths):

```powershell
python collect_inputs.py `
  --tokenizer-dir "C:\path\to\Rineya-Supa-1.7.0-devfreeze" `
  --native-probe `
  --exclusion "C:\path\to\permitted_long_context_prompts.jsonl" `
  --exclusion "C:\path\to\permitted_renderer_and_entity_manifest.json" `
  --output "C:\path\to\LCR_Inputs.zip"
```

Add another `--exclusion` for each permitted relevant bank or renderer/entity inventory. Text, JSON/JSONL, gzip, ZIP, Python renderer source, and Markdown manifests are accepted as explicitly chosen audit inputs; their content is not executed or presumed to be a valid evaluation bank. Assets are individually limited to 64 MiB; combined selected bytes are limited to 256 MiB. Exceeding limits refuses collection rather than silently dropping data.

A hash mismatch is a blocker: verify which core is actually targeted. Do not replace the expected hashes just to make the collector pass. The published-core model.safetensors file is **not needed**.

The output is `COLLECTED_INPUTS_NOT_CORPUS_ACCEPTANCE`. Token measurements, leakage checks, corpus generation, and model training have not run merely because collection succeeds.

## Contract decisions still needed

See `CONTRACT_QUESTIONS.md`. Its proposed readings are not silently approved amendments to the user's original brief.

## Local collector mechanics tests

```text
python -m unittest discover -s tests -v
```

Tests use small invented fixtures, not native tokenizers. Their success is not native tokenizer qualification or evidence of corpus quality.

With `--native-probe`, run the collector using the existing qualified local Transformers environment. It records six synthetic tokenizer/rendering test vectors, software version and tokenizer class; no model weights are loaded. This optional native path was NOT executed in the assistant runtime (Transformers was unavailable). The collected vectors are tokenizer sanity checks, not 6,000-row token measurements. Omit this flag only when collecting assets without a native tokenizer receipt.
