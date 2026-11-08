# RDD-001 offline inventory — 2026-10-05

This receipt records file hashes, JSON metadata and safetensors tensor headers only. No tensor payload was deserialized, no native model/tokenizer was instantiated, no optimizer or GPU allocation occurred, and no sealed or confirmation bank was opened. Fixture results are not capability evidence.

The initial RDD checkout was clean on `research/rineya-distillation-dynamics-v1` at `2d7be4b5c2ba8a610b09d17455d519eecaa421e6`. Applicable package AGENTS, root/package README, PROTOCOL, RUNBOOK, VALIDATION, native loader, qualification, integration, registry and host contracts were read. Initial shared Git lock was absent. Source identities and evidence remain unmodified.

## Exact local source and teacher candidates

All paths below have prefix `C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/`. The upstream `PRIMARY_STUDENT_RECEIPT.json` names the exact Titan-Experimental as its 0.6B source. That upstream choice is evidence of identity, not an RDD asset review or execution permit.

| Local core | Weight SHA-256 | RDD inventory snapshot SHA-256 | Stored header elements |
|---|---|---|---:|
| Rineya-Titan-Experimental-1.0.0-devfreeze | `537201676814ade103c4efa63f5ebe758ea309971803c4ad633dbca5b6b531b7` | `4e6e8f7f0ef89662c2ec910280a72501e05f2e89cf88498a62edfa029af0609f` | 596,049,920 |
| Rineya-Supa-1.7.0-devfreeze | `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b` | `a1da3fdd639d368c45db9e3ba8b8a0237e2c59abe81ce88493f514c59768c339` | 596,049,920 |
| Rineya-Baya-1.7.0-devfreeze | `1737cd1ee2bdde714c259adf2012ae44c89ad82a06797653343bad7e3cf88bba` | `4a7bf00973daba359c9ab55a43974edf7815794284e3a0db540c4321394c1a68` | 596,049,920 |
| Rineya-Halo-1.2.0-devfreeze | `38fb96fe286776919032c4418eef0a2534e2c2c2865c1465cc9dd4cfe7e36544` | `f8a2de172d881e341f3c92dbce0cdec446bb52b8f9cf7c13af1b45f26ed58f8d` | 596,049,920 |

Each has 310 stored tensors with standard `model.embed_tokens.weight` / `model.layers.*` names, an explicit `qwen3` / `Qwen3ForCausalLM` config, tied word embeddings and no `auto_map` or quantization config. Titan/Supa/Baya store BF16 (about 1.11 GiB payload each); Halo stores FP32 (about 2.22 GiB). These are **static layout candidates**, not parity-qualified exports or loaded-parameter counts. No custom checkpoint conversion or stock-Qwen substitution was performed. No RDD teacher competence result or reviewed student-prefix feedback was found; candidate identity does not establish teacher suitability.

## Tokenizer and prompt contract

Titan, Supa and Baya have identical `tokenizer.json`, `tokenizer_config.json`, `generation_config.json`, merges and vocabulary file bytes. Their tokenizer backend canonical digest is `91a5d4d92157b8d3cdb2599dca8ff445ecec6fdaeab57904c229ece438493bca`; their chat template text digest is `a55ee1b1660128b7098723e0abcd92caa0788061051c62d51cbe87d9cf1974d8`. Their base BPE vocabulary contains 151,643 entries; the model output vocabulary is 151,936. Added token metadata must be preserved.

Halo shares the vocabulary/model and added token maps and template text, but its backend digest is `a373b7ff0c52176575a72ede185c306613ef1c01a2346f900f3b631ce1dc23f9`. Its ByteLevel pretokenizer and decoder settings differ (`trim_offsets`, `add_prefix_space`, `use_regex`). This violates the current strict backend-identity contract; equal vocabulary size and template text are insufficient. Its config was saved by Transformers 5.14.1 with `rope_parameters`; the registered runtime is 4.57.1, making exact loaded configuration semantics an additional unqualified issue.

Titan/Supa/Baya model config EOS is 151645; their generation config EOS list is [151645, 151643], generation/tokenizer pad is 151643. The historical central E0 uses EOS [151645, 151643] and pad 151645. RDD currently derives EOS from model config, so it uses only 151645. This is an unqualified stopping/masking contract difference, not permission to silently patch assets or inherit a scope.

Native rendered prompts, consumed token IDs and model output parity were not measured. A complete prospective consumed-prompt manifest must record actual rendering, terminal handling, rollouts, independent replay, evaluations, teacher qualification and source baselines as they occur; source recipes alone cannot substitute for it.

## Existing E0 and export evidence

The bounded search examined central `E0_RESULT.json` receipts and filename inventories for export-review receipts in central results, local artifacts and RSD records. No receipt with status `PASS_E0_SCOPED_NO_EFFICACY_CLAIM` satisfying unchanged RDD admission, and no `REVIEWED_NATIVE_EXPORT` attestation tied to these exact student/teacher snapshots, was identified. This is a bounded search finding, not proof that no such file exists elsewhere.

`C:/Users/clclc/BusinessProject/ChatBubble-post-training-Mimo/experiments/post_training_mimo_centralized_v1/results/headless_unpadded_ce512_v2_20260930_0910/E0_RESULT.json` has historical status `PASS_E0_UNPADDED_CE512_SCOPED` and all five required mechanic flags true. It has one historical load and four scoped optimizer steps. Its config digest is `42a32cfa2fe4dc1250a288bacec2a60d150c5498d8dced9bba23b0c5037f5929`. Its config scopes unpadded single-sequence CE through 512 tokens, generation cap zero, and host reserve 1.25 GiB. The unchanged RDD status contract rejects it; it does not qualify sampled-policy KL, student/teacher generation, 1024-token contexts or the stricter RDD resource profile. It must not be relabeled or treated as imported authority.

The central Titan v1–v4 attempts are retained failures. v1 hit a WindowsPath API error; v2–v4 failed host RAM admission before any load/update. Other central evidence includes a load/forward-only PASS, padding-likelihood failure and a streamed CE attempt stopped by GPU headroom. `asset_inventory.json` records all statuses and hashes without weakening any guard.

The GD historical receipt `C:/Users/clclc/BusinessProject/gd001-records/qualification-20261003T051747Z/qualification_summary.json` reports blocked native prerequisites, no complete eligible trajectory with complete consumed-prompt inventory and no GD native forward calls. Halo saved payloads end at 250 versus planned 800; Baya's completed training metadata lacks checkpoint weight payloads. These findings cannot certify a new RDD export. GD is owned by another active chat; no GD files were edited.

## Real host and ownership snapshot

Current read-only snapshot at `2026-10-05T18:56:20Z`: RTX 4070 Laptop GPU, 8188 MiB total / 7018 MiB free; 31.218 GiB host RAM / 12.415 GiB free; 60.907 GiB disk free. These numbers are time-varying observations, not admission. An earlier snapshot had approximately 10.8 GiB host RAM and 6703 MiB GPU free.

The parent recorded active COMP-RS attempt14 Python PIDs 158788/47240/159052/158976/159284/159120 at its initial snapshot. Their later absence does not erase that busy evidence. The current snapshot contains Python PIDs 212380/213040 under `SupaExperiment-nonrecurrent-adaptation-v1/.venv`, and GPU PIDs with insufficient inspection permissions. The shared lock `C:/Users/clclc/BusinessProject/Chat-Bubble/.git/rineya-central-gpu0.lock` is absent, but host availability/ownership and native admission remain unestablished. No worker was started or stopped.

The existing `.venv-mq001` distribution metadata matches Python 3.11, torch 2.10.0+cu126, Transformers 4.57.1, tokenizers 0.22.2 and safetensors 0.7.0; psutil is 7.2.2. PEFT and pytest are absent from that runtime. No environment or package was installed or modified.

Unchanged RDD floors: 7.4 GiB host admission twice; 6 GiB running host headroom; 4 GiB initial GPU free; 0.5 GiB running GPU free; 4 GiB disk free; 4 GiB process-tree RSS; 1800 seconds per invocation. Static serialized bytes do not prove that loading two models, generation, KL, gradients and save/reload fit these limits.

## Next decision

Review one exact frozen source/teacher pair and one prospective learning rate, with the proposal's small selected arm set. Before any native grant, obtain genuine matching E0 evidence for the intended scope, independent exact-native/standard-HF and custom-LoRA/PEFT parity, teacher prefix review, fresh owned catalogs and complete exclusion metadata, and an owned available host meeting unchanged guards. Teacher competence screening and disposable native KL/backward/save-reload remain prepared work, not completed qualification. Confirmation remains unopened.

Machine-readable receipts: `asset_inventory.json` SHA-256 `ab2dfaf1566886ebde1ca3b4088ff5ee335fb006707bef768921c057d2a87e78`; `host_inventory.json` SHA-256 `5da7564534040451859bf48fedc270274cc837358c07cac7e30af777e117de47`. The accompanying stdlib `inventory_metadata.py` documents the offline collection method. All new outputs are additive under BusinessProject outside tracked source and `.codex`.
