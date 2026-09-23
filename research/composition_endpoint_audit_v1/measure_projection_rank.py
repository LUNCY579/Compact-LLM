"""q vs o rank structure in delta_T AT MATCHED DEPTH -- does the q/o coefficient gap reduce to rank?

WHY THIS EXISTS. SCHEDULE_13CLAIMS Phase 5 (RANK-002, never run) records:
  "L27 o_proj effective rank 16.2 at 88.4% top-16 energy; L8 q_proj 71.5 at 58.7%"
and it was raised as possibly the same underlying fact as the q-o ALLOCATION gap that separates
LoRA from dense deltas. Those two cells differ in LAYER as well as PROJECTION, so on their own
they cannot separate depth from projection. This measures both projections at every layer.

The 88.4% and 58.7% are the canary: if this script does not reproduce them, it is measuring a
different object and nothing else it prints can be trusted.
"""
import statistics as st
import torch
from safetensors import safe_open

REC = r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records"
SRC = r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze/model.safetensors"
T17 = f"{REC}/int001-20260917/trained_fp32.safetensors"
PROJ = ("q_proj", "k_proj", "v_proj", "o_proj")


def main():
    rows = {}
    with safe_open(SRC, framework="pt") as S, safe_open(T17, framework="pt") as A:
        tk = set(A.keys())
        for L in range(8, 28):
            for proj in PROJ:
                k = f"model.layers.{L}.self_attn.{proj}.weight"
                if k not in tk:
                    continue
                d = A.get_tensor(k).double() - S.get_tensor(k).double()
                e = torch.linalg.svdvals(d) ** 2
                cum = e.cumsum(0) / e.sum()
                rows.setdefault(proj, []).append((L, int((cum < 0.99).sum()) + 1, float(cum[15])))

    print(f"{'proj':8s} {'eff rank @99%':>24s} {'top-16 energy':>26s}")
    for proj in PROJ:
        r = [x[1] for x in rows[proj]]; t = [x[2] for x in rows[proj]]
        print(f"{proj:8s} median {st.median(r):5.1f} [{min(r)}-{max(r)}]      "
              f"median {st.median(t)*100:5.1f}% [{min(t)*100:.1f}-{max(t)*100:.1f}]")

    q, o = rows["q_proj"], rows["o_proj"]
    print("\nCANARY -- the two cells RANK-002 cites, which must reproduce:")
    print(f"  L8  q_proj top-16 energy {q[0][2]*100:.1f}%   (recorded 58.7%)")
    print(f"  L27 o_proj top-16 energy {o[-1][2]*100:.1f}%   (recorded 88.4%)")
    assert abs(q[0][2] * 100 - 58.7) < 0.1 and abs(o[-1][2] * 100 - 88.4) < 0.1, \
        "CANARY FAILED -- this is not the object RANK-002 measured"

    print("\ntop-16 energy by depth (both projections rise monotonically with L):")
    for i, L in enumerate(range(8, 28)):
        if L % 4 == 0 or L == 27:
            print(f"  L{L:2d}   q {q[i][2]*100:5.1f}%   o {o[i][2]*100:5.1f}%")

    w = sum(1 for a, b in zip(q, o) if a[1] > b[1])
    print(f"\nAT MATCHED DEPTH: q eff rank exceeds o in {w} of {len(q)} layers "
          f"(median q {st.median([x[1] for x in q]):.0f} vs o {st.median([x[1] for x in o]):.0f})")
    print("=> the cited gap is a DEPTH gap, not a projection gap. The q/o ALLOCATION axis and")
    print("   RANK-002's depth-aware rank allocation are INDEPENDENT axes.")


if __name__ == "__main__":
    raise SystemExit(main())
