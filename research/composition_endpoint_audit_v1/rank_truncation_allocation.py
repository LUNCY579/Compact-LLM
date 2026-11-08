"""Is LoRA's q-heavy allocation a RANK-TRUNCATION artifact, or a property of LoRA's TRAINING?

Both LoRA objects on disk put far more squared norm into q_proj than into o_proj (gap +6.78
depth-matched, +9.03 for DOSELORA), while both dense deltas put q ~ o (gap +0.90, +0.01).
LoRA is rank-constrained, so the obvious deflationary account is that ANY rank-16 object looks
like that -- in which case the signature says nothing about LoRA's training.

Test: compare delta_T's allocation to P16(delta_T)'s. Same tensors, same depth, same delta;
the ONLY difference is the rank-16 truncation. Per tensor, ||P16(d)||^2 is the top-16
singular energy, so this needs one svdvals pass and no reconstruction.

CANARY: ||P16(delta_T)|| must reproduce the recorded 0.674029920773733.
"""
import torch
from safetensors import safe_open

REC = r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records"
SRC = r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/model.safetensors"
T17 = f"{REC}/int001-20260917/trained_fp32.safetensors"
FAMS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main():
    dense = {f: 0.0 for f in FAMS}
    p16 = {f: 0.0 for f in FAMS}
    nrm = nrm16 = 0.0
    with safe_open(SRC, framework="pt") as S, safe_open(T17, framework="pt") as A:
        for k in sorted(set(A.keys())):
            f = next((x for x in FAMS if x in k), None)
            d = A.get_tensor(k).double() - S.get_tensor(k).double()
            if d.ndim != 2:                      # 1D norm deltas pass through P16 unchanged
                nrm += float((d * d).sum()); nrm16 += float((d * d).sum()); continue
            e = torch.linalg.svdvals(d) ** 2
            tot, t16 = float(e.sum()), float(e[:16].sum())
            nrm += tot; nrm16 += t16
            if f:
                dense[f] += tot; p16[f] += t16

    print(f"{'family':11s} {'dense %':>9s} {'P16 %':>9s} {'shift':>8s}")
    for f in FAMS:
        a, b = 100 * dense[f] / nrm, 100 * p16[f] / nrm16
        print(f"{f:11s} {a:9.2f} {b:9.2f} {b-a:+8.2f}")
    qa, oa = 100 * dense["q_proj"] / nrm, 100 * dense["o_proj"] / nrm
    qb, ob = 100 * p16["q_proj"] / nrm16, 100 * p16["o_proj"] / nrm16
    print(f"\nq - o gap:   dense {qa-oa:+.2f}   after rank-16 truncation {qb-ob:+.2f}")
    print(f"             LoRA-596 (depth-matched) +6.78   DOSELORA +9.03")
    print(f"\n||delta_T|| {nrm**0.5:.6f}   ||P16(delta_T)|| {nrm16**0.5:.15f}   "
          f"(top-16 keeps {100*nrm16/nrm:.1f}% of the energy)")
    assert abs(nrm16 ** 0.5 - 0.674029920773733) < 1e-6, "CANARY FAILED -- not the recorded object"
    print("CANARY PASS: ||P16(delta_T)|| reproduces the recorded 0.674029920773733")


if __name__ == "__main__":
    raise SystemExit(main())
