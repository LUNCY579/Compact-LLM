# EFE-FUNC-001 exact asset audit

Decision: **HOLD — the two original skill endpoints are absent.** The original anchor and all six sidecars are present and hash-verified. The historical independent composition, its individual controls, and the original training checkpoint directories are absent. No model was loaded, checkpoint arithmetic performed, GPU work launched, or weights downloaded. This audit reads saved reports and metadata; it does not open protected evaluation payloads.

This paragraph describes the initial local audit. The subsequent task-level live vault
check and aborted original T restoration are in [vault_availability.json](vault_availability.json).
The pinned original T/H LFS hashes and sizes match and are accessible. The initial broad
interpretation that the goal banned all original restoration was corrected: it bars
replacement weights. No complete restored endpoint is locally hash-verified, and the
restoration was stopped on RAM pressure. Initial audit counts below are not whole-task totals.

The machine-readable [asset_audit.json](asset_audit.json) binds the inspected Git objects, local record SHA256s, endpoint lineage, the complete source header inventory, and declared changed tensor mapping. Records repository HEAD at audit: `3a7b044f587a09233f5c416456b0e17fdc33d76a`.

## Original anchor and tokenizer

Exact directory: `C:\Users\clclc\BusinessProject\Chat-Bubble\Rineya_Published_Cores\Rineya-Supa-1.7.0-devfreeze`.

| Asset | SHA256 verified from present file bytes |
|---|---|
| model.safetensors | `962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b` |
| config.json | `660db3b73d788119c04535e48cf9be5f55bc3100841a718637ae695b442f27dd` |
| generation_config.json | `2325da0f15bb848e018c5ae071b7943332e9f871d6b60e2ed22ca97d4cb993d2` |
| tokenizer.json | `aeb13307a71acd8fe81861d94ad54ab689df773318809eed3cbe794b4492dae4` |
| tokenizer_config.json | `d5d09f07b48c3086c508b30d1c9114bd1189145b74e982a265350c923acd8101` |
| vocab.json | `ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910` |
| merges.txt | `8831e4f1a044471340f7c0a83d7bd71306a5b867e95fd870f74d0c5308a904d5` |

All seven hashes match both original training `binding.json` records. All six sidecar hashes also match the retained INDEP-001 sidecar copies. The weight file is 1,192,135,096 bytes. A raw safetensors header read, without importing Torch or constructing a model, finds 310 stored BF16 tensors and 596,049,920 stored elements. Config declares tied word embeddings; `model.embed_tokens.weight` is stored and `lm_head.weight` is absent. Native tied parameter identity and weight finiteness remain unqualified.

This is the exact historical BF16 anchor. Upcasting these original anchor values is lossless; rounding the *trained endpoint or resulting update* into a BF16 export loses information. A similarly named Supa/A1 core or published Titan export is not an admissible replacement.

## Missing skill endpoints and custody

The record filenames called “delta” contain **partial trained absolute FP32 parameters**, not signed deltas. The signed update must be extracted by subtracting the exact upcast anchor on the declared trained tensors.

| Component | Original relative path in records repository | Required SHA256 | Bytes | Present |
|---|---|---|---:|---|
| T, seed 17 | `int001-20260917/trained_fp32.safetensors` | `07156fb5aa4a8c510c63e8af2118a08a8b6bf004549532d497c2e89e2deec24d` | 1,258,500,416 | No |
| H, boundary skill | `compose001-20260918/trained_H_fp32.safetensors` | `a1103060591a91ee7d9ceb2ad7450621be352b087cd19188f2c383a0dbbc7b90` | 1,258,500,512 | No |

Both `WEIGHTS_DELETED.json` files record deletion on September 26, 2026 and explicitly classify these as Tier A: **“NOT rebuildable; restorable only from the vault.”** `TRAINED_TENSOR_BACKUP.json` identifies T as the RRP-015 control draw at selection seed 17 (384 transition rows, 54,172 processed tokens, 28 optimizer updates), and H as the COMPOSE-001 H draw (648 boundary rows, 54,217 processed tokens, 28 updates). Their training source bindings agree exactly.

The original T checkpoint, `rrp015-20260917/control-journal/runs/train-001/microfit/checkpoint-00028`, and H checkpoint, `compose001-20260918/H/journal/runs/train-001/microfit/checkpoint-00028`, are absent. The earlier T `control-hf/RECLAIMED_REDUNDANT_EXPORT.json` says the checkpoint was retained *then*; its present absence means that earlier receipt is not current availability. Consolidation from that checkpoint cannot now reconstruct the trained endpoint.

Saved custody receipts point to private repository `Luncy1978/rineya-published-cores`, pinned revision `33b5f9a46f4c49f22f1a305fdb21373f15401eb8`, under:

- `research-records/delta-weight-screens-20260918/_trained_tensors/delta_T_seed17.safetensors`
- `research-records/delta-weight-screens-20260918/_trained_tensors/delta_H_boundary.safetensors`

The deletion receipts record historical verified pull-back hashes and sizes. This audit did not recheck the remote vault or download anything. Current local endpoint custody is missing; a historical backup receipt is not locally available model bytes. The total missing endpoint payload is 2,517,000,928 bytes, before any resulting candidate checkpoint or workspace.

A bounded filename/size scan inspected 1,162 safetensors files in the records tree and the existing HF cache for this vault. No original endpoint object was found. The only same-size records candidate is a different `trained_C_fp32.safetensors`; its saved backup hash is `e42afe6a89231188cf03879647267b2040d2a6fccc1ff1fc3a85448d28c9e447`, so it is excluded. No third-skill payload was opened. The existing vault cache has neither required endpoint SHA256 blob. This is a bounded search, not an assertion about every filesystem drive.

## Exact frozen recipe and reconstruction boundary

The candidate is:

`theta_source + 1.00 * P16(endpoint_T - theta_source) + 0.25 * P16(endpoint_H - theta_source)`.

`P16` truncates each 2D delta to `min(16, min(shape))` singular components; nonmatrix deltas are retained in full. Saved MAG/COMPOSE metadata declares 220 changed tensors: 140 matrices and 80 nonmatrix tensors, covering layers 8–27, with 90 source tensors untouched. The JSON binds every declared changed name and its shape from the exact source header. Because the endpoint files are absent, their actual key sets, shapes, precision and finite values cannot be reverified.

Historical expected independent output hash: `b5a70d700cce689ddab1e80bc449f755b2dc24b165ecfb32aa2e27162c16efec`; expected rebuilt T-only INDEP control hash: `15895ae868f65d01b569c9580f3d6ca7275e7f31ce59b853618d48cf52d8689b`. Their weights are absent. Other saved controls are also absent: original T16, H16, raw TH16, and TH16s. Recipes and receipts alone cannot rebuild any of them without the two Tier A endpoint objects.

`build_indep.py` was inspected directly. It subtracts in FP32, computes SVD and the combined update in FP64, adds to the upcast source in FP64 and stores changed tensors in FP32. It pins 16 Torch threads. **It copies the 90 untouched source tensors in their original BF16 dtype**, although the receipt says `stored_dtype: float32`. That receipt field is incomplete. It also retains a stale SCALED-H description; numeric coefficients and inspected code identify the actual recipe. The candidate receipt's corrected label and output hash are retained rather than silently rewriting historical evidence.

COMPOSE and MAG used FP32 SVD/addition paths, while the later INDEP path uses FP64 SVD after FP32 subtraction. HLOC upcasts all source tensors to FP32. A generic implementation of the same mathematical equation therefore does not automatically reproduce the historical stored tensor object or file hash. Future qualification must bind the exact common arithmetic, thread count, runtime, stored dtypes and tensor identity. No such qualification can run on the absent endpoints.

The historical shared-scale control is **per tensor**, not global coefficient 0.675:

`s_t = ||P16(delta_T_t)|| / ||P16(delta_T_t) + P16(delta_H_t)||`.

Its 220 saved factors range from 0.517485038576387 to 0.7495600167218748, median 0.6749567023937648. The JSON binds the factors and their receipt hash. Substituting a global scalar changes the control.

The historical reproducibility note distinguishes a safetensors header change from a tensor change and reports SVD sensitivity to thread count and memory pressure. A future file-hash mismatch requires checking schema, names, shapes, dtypes and tensor contents; a data-section digest alone is meaningful only after its tensor mapping is bound. No rebuilt object exists here to make that comparison.

## Provenance and decision limits

The requested historical sources were read from these exact Git objects:

- `b8e40d377c9cd461cd30c215a2c713515904197b`: `MAGNITUDE_IS_THE_MASTER_VARIABLE_20260920.md`, blob `4318c8e556f90701a279d55e3274c36a6370c62e`; `REPRODUCIBILITY_NOTE.md` is separately hash-bound in JSON.
- `9abd010e675d2853fe689404c188baecb21af719`: endpoint audit `RESULT.md`, blob `7f77217ea150cd6323ee1a3b269c6cff6260767f`; `HLOC_RESULT.md`, blob `18c211d19d3b5d0a3127021670b982dd8ba21f7f`; and `run_hloc.py`.

INDEP-001's coefficient-selection records are spent development evidence, not an established optimum or fresh preservation confirmation. The endpoint audit identifies an insensitive retention gate; HLOC reports indeterminate localization. Those conclusions do not qualify missing model assets or replace fresh data. HLOC's historical statement that deleted derived arms rebuild from its script plus two checkpoints was conditional on endpoints that were available then and have since been reclaimed.

No narrower AGENTS.md was found in the EFE target, records tree, or source asset ancestors inspected. All source assets, record trees, original checkouts and sibling worktrees were read-only; only these two audit deliverables were written in the current EFE worktree.

The execution blocker is exact: the original T and H trained objects and their local source checkpoints are missing. Restoring identical original vault objects is a potential custody route, unlike substituting replacement weights. Same-seed retraining would create a new execution draw and is not proof of endpoint identity; training or substituting exports is disallowed. Model-free protocol and native integration preparation can proceed. Actual confirmation requires the two exact objects to become locally available with verified file bytes, admissible data, resource admission and native qualification before any candidate outcomes are read. Current decision remains HOLD with zero native model execution; no unchanged restoration retry follows the RAM refusal.
