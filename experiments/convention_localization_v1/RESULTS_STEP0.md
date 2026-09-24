# CONVENTION-LOC-001, step 0: weight-only direct-logit capacity (exploratory)

**Verdict: fails its negative control and does not localize the convention.** This is a screening
prior for a forward-pass screen. It is not a mechanism claim.

## Question

ADD-100 (Supa + dSparse + dOpallo) ends 178/192 development answers with a period. SPARSE and AVG
end 0 with one. The period arrives as the merged token `}.` after a JSON answer (114/178), or as
`.` after a bare number (64/178). That gives two decisions:

- **D1** `}.` vs `}`
- **D2** `.` vs `<|im_end|>`

The question: can the **weights alone** show which blocks' residual writers (o_proj, down_proj)
aim dOpallo at those decisions? `weight_dla.py` computes
r = d_model·‖dWᵀv̂‖²/‖dW‖²_F for v = g ⊙ (e_a − e_b), where g is Supa's final RMSNorm gain and the
embeddings are tied. Each r is read against 512 random token-pair contrasts. **dSparse emits no
period, so it goes through the identical pipeline as the negative control.**

## Result (`WEIGHT_DLA.json`)

Mean ratio to the null median, by block group:

| blocks | D1 dOpallo | D1 dSparse | D2 dOpallo | D2 dSparse |
|---|---|---|---|---|
| 8–12 | 1.00 | 0.94 | 1.10 | 0.94 |
| 13–17 | 1.15 | 0.97 | 1.20 | 1.07 |
| 18–22 | 1.48 | 1.08 | **3.23** | **1.56** |
| 23–27 | 2.18 | 2.73 | 7.21 | 6.53 |

- **The control fails the inference.** dSparse, which never emits the period, aligns with both
  contrasts as strongly as dOpallo does in the last five blocks (block 27 down_proj, D2: 22.5 vs
  22.8). Late-block writers aim at the end-of-answer tokens **whatever they learned**, so high
  alignment there is not evidence of the convention.
- **One residual difference survives the control:** in blocks 18–22, dOpallo's D2 alignment is
  about **2×** dSparse's (log2 ratio +1.05). The null medians match between the two deltas
  (0.91–0.98), so this is not a scale artefact. It is recorded as a **hypothesis only**.

## What this licenses

Nothing about mechanism. This agrees with memory `surface-conventions-transfer-with-a-delta`:
no weight-space statistic sees the convention. A forward-pass test is required.

**Pre-stated prediction for the GPU screen (step 1, not yet built).** If the 18–22 hint is real,
removing dOpallo from blocks 18–22 lowers ADD-100's period rate **more** than removing it from
23–27, even though 23–27 has the larger raw alignment. If the period rate instead follows raw late
alignment (23–27 matters most), the weight hint was noise.

## Limits

Direct path only. No activations, so no sign. Magnitude capacity, not use. One pair of deltas.
