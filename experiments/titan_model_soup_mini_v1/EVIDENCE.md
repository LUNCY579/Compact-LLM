# Evidence and novelty register — inspected 2026-09-22

This file preserves observations, not endorsement of all historical prose. No source file,
recorded outcome, quarantine role, old threshold, native-read budget, or lease was edited.

## GitHub snapshots

**ChatBubble / Titan publication snapshot: `954a0c2637e691b493c5ac3d488653a8c1adf675`**

- `Rineya_Published_Cores/Rineya-Titan-Experimental-1.0.0-devfreeze/CORE.json`
  documents exact dense Supa 1.7 ancestry, 596,049,920 parameters, 28 blocks, width 1024,
  vocabulary 151936, tied embeddings, 220 trained tensors in blocks 8-27, 54,172 processed
  tokens and 28 optimizer updates. The previous conversational 5,254-token figure is not
  the RRP-015 Titan processed-token count. Published Titan stores BF16; measured through
  FP32 DenseReference. Supa-A1.1 was explicitly not its source.
- `.../AMENDMENT_001.json` distinguishes the unrounded FP32 parent (235 acquisition;
  long-context 157) from the stored BF16 artifact (234; 162). The recorded BF16 census
  erases 60.61% of update elements, including 24/220 entire trained tensors. Do not turn
  BF16 back into FP32 and call it the unrounded checkpoint. Historical mechanism wording
  is not a proof of universal tail damage.
- `Rineya_Published_Cores/Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze/CORE.json`
  binds top-25%-magnitude support of the same RRP-015 delta, no new training, stored FP32.
  It records 232/384 acquisition, 41/47 source arithmetic retained, and 7 source-correct
  long-context rows lost. Neither a net score gain nor nonsignificance means no forgetting.
  Exact promoted sparse hash is pinned; do not silently substitute a rebuild at threshold ties.

**ChatBubble / Opallo snapshot: `171a3014abfd0ad4afb741070a0403cb6c97b609`**

- `Rineya_Published_Cores/Rineya-Opallo-1.0.0-devfreeze/CORE.json` binds the SAME Supa hash
  and tokenizer assets, same 28L/1024/151936/tied architecture, last-20-block fine-tuning,
  full exported BF16 weights. Arithmetic 73/150 and source-retained 41/47 are historical
  selection-panel observations, not general intelligence gains. Long-context strict score
  regressed 161 to 135/256. Old literal UNKNOWN/containment interpretation in that record
  is not used for this experiment: that interpretation has later corrections and native
  review remains a separate line. A new matched primary scorer is required.
- The exact core is `Rineya-Opallo-1.0.0-devfreeze`, not a guessed
  `Rineya-Opallo-Experimental-...` directory.

**rineya-dataset-retention-records snapshot: `90255f33bc78cdfd6872b3ff9cc1982d5b9b5d42`**

- `int001-20260917/INT001_RESULTS.md`: existing FP32 Titan/Supa alpha sweep.
  Alpha 0.50 observed acquisition 219/384 and retention 42/47, versus stored Titan
  234 and 39. This licenses an informative precedent, not a new experiment headline.
  Its long-context/abstention bank is report-only. Gate v2 and the legacy monitor disagree;
  this package neither chooses between them retroactively nor imports them as universal gates.
- `ATTENUATION_CORRECTION_20260918.md`: strength-matched comparison 235 -> 223 -> 219;
  12 rows lost from attenuation and 4 additional rows from the second task, paired p=.45
  for the latter. The old 16-row skill-conflict claim is withdrawn. Same-skill TT16s
  measurements 247/238 remain observations of seed averaging, not orthogonal-skill proof.
- `BANK_SPLIT_IS_CONTAMINATION_20260920.md`: D4 parents overlap SELECT-VIEWS-001 on
  275/384 rows, including 128/128 prerequisite_plan parents. Therefore a locally fine-tuned
  descendant cannot inherit its parent's evaluation-admissibility without an ancestry audit.
- `TRAINED_TENSOR_BACKUP.json`: five preserved training outputs, including T seeds
  17/23/29, H boundary and MIX rehearsal. Full Supa hash and original trained-output hashes
  are recorded, but some backup hash-verification flags are false. Backup `delta_*` names
  and `trained_fp32` paths are NOT interchangeable representations. Check the exporter and
  full tensor inventory before any reconstruction. No local checkpoint was accessed here.
- `README.md`: weights live outside GitHub, sidecars are canonical, line-ending conversion
  must remain disabled for existing evidence, and masked rebuilds may differ at threshold
  ties. This mini package pins exact input bytes rather than weakening identity on a mismatch.
- Latest commit records GEN2-001's spent one-attempt RAM-race envelope. That experiment's
  resource rules and remaining budget are outside SOUP-MINI-001 and untouched.

## Algebraic classification

Let S be exact Supa, T=S+dT, O=S+dO, sparse T=S+M*dT.

1. Soup(S,T) is delta scaling (existing INT-001 family), not an independent skill donor.
2. Soup(T,sparse T) is a soft mask when T is the matching unrounded FP32 parent:
   at half, S+[M+0.5(1-M)]*dT. Stored BF16 Titan makes the equality inapplicable
   to the unrounded delta; label that an export-level interpolation instead.
3. Soup(T,O) at half is S+0.5*dT+0.5*dO. Compare against S+0.5*dT and S+0.5*dO,
   in addition to full donors. This separates component weakening from added-task effects.
4. Local child L=T+dL: average with T to preserve dT, with S to halve both updates.
   Averaging different skill deltas does not preserve their full strength by definition.
5. Small update norm and vector orthogonality are diagnostics, not guarantees of functional
   independence, stable downstream behavior, lossless composition, or broad intelligence.

## Primary external references

- Wortsman et al., *Model soups*, ICML 2022, arXiv:2203.05482:
  https://arxiv.org/abs/2203.05482 . Averaging fine-tuned checkpoints can improve validation
  performance without ensemble inference cost; it is not a guarantee for cross-skill LLMs.
- Ilharco et al., *Editing Models with Task Arithmetic*, ICLR 2023, arXiv:2212.04089:
  https://arxiv.org/abs/2212.04089 . Shared-base weight differences and their composition.
- Safetensors format and official reader:
  https://github.com/safetensors/safetensors#format and https://huggingface.co/docs/safetensors/ .
  The chunked writer follows the little-endian 8-byte header length, padded JSON header,
  tensor shape/dtype/offset descriptors, contiguous little-endian tensor payload format;
  tests round-trip through the official reader, including BF16 inputs and tied aliases.

## Novelty boundary

The proposed primary new comparison is the **exact promoted Titan-Sparse/Opallo midpoint**
with matched attenuation controls and fresh admissible data. Same-base scaling and seed soup
have precedents. This was a targeted record audit, not an exhaustive repository-wide search;
run the existing settled-experiment/receipt check before spending native budget. Hashes and
coefficients, not marketing names, define whether a prior experiment is the same.
