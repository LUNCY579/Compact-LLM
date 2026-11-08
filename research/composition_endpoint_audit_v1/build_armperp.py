"""ARMPERP-001 build: theta_supa + P16(dT,s17) + alpha * d_perp, registered in ARMPERP_PREREG.md.

    d_perp = P16(dT,s23) - c * P16(dT,s17)        same-skill content, EXACTLY orthogonal to dT
    c      = <P16(dT,s23), P16(dT,s17)> / ||P16(dT,s17)||^2
    alpha  chosen so the TOTAL norm lands on indep003_s17_h025's 0.6974713381755033

WHAT IS REUSED RATHER THAN REWRITTEN. `p16()` is copied verbatim from
indep005-20260921/build_indep5.py -- rank-16 SVD per 2D tensor in FP64, 1D tensors passed
through WHOLE so the 80 norm deltas are kept exactly as every other arm in this family keeps
them. A second truncation implementation would make the contrast a contrast of implementations.

c AND alpha COME FROM RECORDED QUANTITIES, AND THE BUILD VERIFIES THE REALISATION. Deriving
them from indep005's recorded delta_cosine and norms avoids a second SVD pass over both
checkpoints on a box with 5.7 GiB free. That is a convenience, NOT a gate: gate 1 measures the
realised cosine on the BUILT tensors and refuses if Gram-Schmidt did not actually land.

GATE 5 IS A POSITIVE CONTROL AND IT RUNS BEFORE THE ARM IS WRITTEN. Perturbing c or alpha by 1%
must be REJECTED by gates 1 and 2. A gate that has never rejected anything is indistinguishable
from one that passes. The variants are evaluated analytically from the accumulated inner
products rather than materialised -- 2 x 2.38 GB of scratch does not exist on this disk, and the
quantities the gates test are fully determined before the bytes are written.
"""
from pathlib import Path
import json, hashlib, os, sys, time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

REC = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
SRC_DIR = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze")
SRC = SRC_DIR / "model.safetensors"
T17 = REC / "int001-20260917/trained_fp32.safetensors"          # delta_T seed 17
T23 = REC / "rep001-20260917/s23/trained_fp32.safetensors"      # delta_T seed 23 -- SAME SKILL
OUT_DIR = REC / "armperp001-20260923/arm-perp"
RANK = 16

# --- recorded quantities (indep005 MATERIALIZATION_same_a0554.json, indep003 h025) -----------
REC_COS = 0.5546985992149074      # cos(P16 dT s23, P16 dT s17), flattened
REC_N17 = 0.674029797822056       # ||P16(dT,s17)||
REC_N23 = 0.6847359459102159      # ||P16(dT,s23)||
TARGET  = 0.6974713381755033      # indep003_s17_h025 built_update_norm -- the norm to match
MAG001_N17 = 0.674029920773733    # MAG-001's independently recorded value for the same object
COS_TOL, NORM_TOL, N17_TOL = 1e-9, 5e-7, 1e-6


def p16(delta: torch.Tensor) -> torch.Tensor:
    """Rank-16 truncation of a 2D update; 1D tensors pass through unchanged. VERBATIM from
    build_indep5.py -- 1D norm deltas are kept whole, exactly as COMPOSE-001 kept them."""
    if delta.ndim != 2:
        return delta.double()
    d = delta.double()
    U, S, Vh = torch.linalg.svd(d, full_matrices=False)
    r = min(RANK, S.numel())
    return (U[:, :r] * S[:r]) @ Vh[:r]


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    c = REC_COS * REC_N23 / REC_N17
    n_perp_sq = REC_N23 ** 2 - (c ** 2) * (REC_N17 ** 2)
    alpha = ((TARGET ** 2 - REC_N17 ** 2) / n_perp_sq) ** 0.5
    print(f"c = {c:.15f}   ||d_perp|| = {n_perp_sq**0.5:.15f}   alpha = {alpha:.15f}", flush=True)

    # ---- GATE 5: POSITIVE CONTROL, evaluated BEFORE anything is built -----------------------
    # Perturbing c by 1% leaves a residual along dT: cos = -0.01*c*||dT|| / ||2nd component||.
    # Perturbing alpha by 1% moves the total norm off TARGET. Both must EXCEED tolerance.
    bad_c = 0.01 * c * REC_N17 / (alpha * n_perp_sq ** 0.5)
    bad_norm = ((REC_N17 ** 2 + (1.01 * alpha) ** 2 * n_perp_sq) ** 0.5) - TARGET
    print(f"\nGATE 5 POSITIVE CONTROL (must both REJECT):")
    print(f"  c   +1% -> |realised cos| {abs(bad_c):.6e}  vs tol {COS_TOL:.0e}  "
          f"{'REJECTED ok' if abs(bad_c) > COS_TOL else 'ACCEPTED -- GATE IS BLIND'}")
    print(f"  a   +1% -> |norm error|   {abs(bad_norm):.6e}  vs tol {NORM_TOL:.0e}  "
          f"{'REJECTED ok' if abs(bad_norm) > NORM_TOL else 'ACCEPTED -- GATE IS BLIND'}")
    if not (abs(bad_c) > COS_TOL and abs(bad_norm) > NORM_TOL):
        raise SystemExit("REFUSED: a gate failed to reject its deliberate mutation")

    # ---- build ------------------------------------------------------------------------------
    t0 = time.time()
    out, ip, ss_total, ss_second, ss_17 = {}, 0.0, 0.0, 0.0, 0.0
    layers, applied, rank_report = set(), 0, []
    with safe_open(str(SRC), framework="pt") as S, \
         safe_open(str(T17), framework="pt") as A, \
         safe_open(str(T23), framework="pt") as B:
        skeys = sorted(S.keys())            # materialise ONCE -- .keys() is a consuming view
        tkeys = set(A.keys()) & set(B.keys())
        for k in skeys:
            src = S.get_tensor(k)
            if k not in tkeys:
                out[k] = src.to(torch.float32)
                continue
            s = src.double()
            p17 = p16(A.get_tensor(k).double() - s)
            p23 = p16(B.get_tensor(k).double() - s)
            second = alpha * (p23 - c * p17)          # alpha * d_perp
            upd = p17 + second
            ip += float((second * p17).sum())
            ss_second += float((second * second).sum())
            ss_17 += float((p17 * p17).sum())
            ss_total += float((upd * upd).sum())
            out[k] = (s + upd).to(torch.float32)
            applied += 1
            if (m := k.split("layers.")[-1].split(".")[0]).isdigit():
                layers.add(int(m))
            if p17.ndim == 2 and len(rank_report) < 21:      # GATE 7: effective rank of d_perp
                dp = p23 - c * p17
                sv = torch.linalg.svdvals(dp)
                e = (sv ** 2).cumsum(0) / (sv ** 2).sum()
                rank_report.append(int((e < 0.99).sum()) + 1)
            if applied % 40 == 0:
                print(f"  {applied} tensors  {time.time()-t0:.0f}s", flush=True)

    n17, n_tot = ss_17 ** 0.5, ss_total ** 0.5
    cos_realised = ip / (ss_second ** 0.5 * n17)
    print(f"\n--- GATES ---")
    g1 = abs(cos_realised) < COS_TOL
    g2 = abs(n_tot - TARGET) < NORM_TOL
    g3 = abs(n17 - MAG001_N17) < N17_TOL
    g4 = (applied == 220 and min(layers) == 8 and max(layers) == 27)
    print(f"  1 realised cos(2nd, P16 dT)   {cos_realised:+.3e}  tol {COS_TOL:.0e}   {'PASS' if g1 else 'FAIL'}")
    print(f"  2 total norm {n_tot:.15f} vs {TARGET:.15f}  d={n_tot-TARGET:+.2e}   {'PASS' if g2 else 'FAIL'}")
    print(f"  3 ||P16(dT,s17)|| {n17:.15f} vs MAG-001 {MAG001_N17:.15f}  d={n17-MAG001_N17:+.2e}   {'PASS' if g3 else 'FAIL'}")
    print(f"  4 {applied} tensors, layers {min(layers)}-{max(layers)}   {'PASS' if g4 else 'FAIL'}")
    rr = sorted(rank_report)
    print(f"  7 d_perp effective rank @99% energy over {len(rr)} sampled 2D tensors: "
          f"min {rr[0]} median {rr[len(rr)//2]} max {rr[-1]}  (algebraic bound 32, comparators 16)")
    if not (g1 and g2 and g3 and g4):
        raise SystemExit("REFUSED: a build gate failed; nothing written")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for f in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
              "vocab.json", "merges.txt"):
        if (SRC_DIR / f).exists():
            (OUT_DIR / f).write_bytes((SRC_DIR / f).read_bytes())
    save_file(out, str(OUT_DIR / "model.safetensors"), metadata={"format": "pt"})
    sha = sha256_file(OUT_DIR / "model.safetensors")

    # GATE 6: collision check against the comparators' recorded output hashes
    COMPARATORS = {"indep005_same_a0554": "0ed6b802ea741af1693bc39d3308fc744c9f3ea448f5765f161ac0017fa1fef5"}
    g6 = sha not in COMPARATORS.values()
    print(f"  6 out sha256 {sha[:16]}... differs from comparators: {'PASS' if g6 else 'FAIL'}")
    if not g6:
        raise SystemExit("REFUSED: built arm is byte-identical to a comparator")

    mat = {"record": "ARMPERP_MATERIALIZATION", "schema": 1, "experiment": "ARMPERP-001",
           "prereg": "research/composition_endpoint_audit_v1/ARMPERP_PREREG.md",
           "formula": "theta_supa + P16(dT,s17) + alpha*(P16(dT,s23) - c*P16(dT,s17))",
           "c": c, "alpha": alpha, "rank_r": RANK,
           "norm_P16_delta_T": n17, "built_update_norm": n_tot, "target_norm": TARGET,
           "norm_second_component": ss_second ** 0.5,
           "delta_cosine": cos_realised,
           "second_component_encodes": "SAME SKILL (delta_T seed 23), orthogonalised against seed 17",
           "d_perp_effective_rank_99pct": {"min": rr[0], "median": rr[len(rr)//2], "max": rr[-1],
                                            "n_tensors": len(rr), "algebraic_bound": 2 * RANK},
           "tensors_applied": applied, "layers": [min(layers), max(layers)],
           "stored_dtype": "float32", "layout": "huggingface",
           "source_weight_sha256": sha256_file(SRC), "out_weight_sha256": sha,
           "trained_s17_sha256": sha256_file(T17), "trained_s23_sha256": sha256_file(T23),
           "gates": {"realised_cosine": g1, "total_norm": g2, "norm_P16_T": g3,
                     "coverage": g4, "collision": g6,
                     "positive_control_rejected_both_mutations": True},
           "torch": torch.__version__, "quality_claim": False, "release_permitted": False}
    (OUT_DIR.parent / "MATERIALIZATION_perp.json").write_text(
        json.dumps(mat, indent=1), encoding="utf-8", newline="\n")
    print(f"\nBUILT {OUT_DIR}  sha {sha[:16]}...  {time.time()-t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
