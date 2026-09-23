"""P6: build SPARSE_MLP and its norm-matched scalar control, then score, in ONE leased window.

The lease covers CONSTRUCTION as well as scoring. A streaming build over 220 tensors peaks near
the same host RAM as a forward pass on this box, so exempting it is the mistake that starved a
peer's live run earlier tonight.

Blocks until the card is genuinely free: gpu_lease.acquire() polls the lock AND the process
table, so a non-participating job still holds us off.
"""
import subprocess, sys, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "fresh_dual_skill_instrument_v1"))
from gpu_lease import lease  # noqa: E402

ALPHA_MLP = 0.5365930  # placeholder; recomputed exactly below before use


def main():
    import compose_modules as cm
    # exact norm of dSparse|MLP, measured not assumed
    import numpy as np
    from subspace import ST, ROOTS  # noqa
    with lease(label="fds1:P6", job="build 2 arms + score 2 arms x 192 rows"):
        for arm, out in (("SPARSE_MLP", HERE / "arms/SPARSE_MLP"),
                         ("SCALAR_MATCHED_MLP", HERE / "arms/SCALAR_MATCHED_MLP")):
            if out.exists():
                print(f"  {arm} already built, skipping BUILD (flag, not a verify)", flush=True)
                continue
            r = cm.build(arm, out)
            print(f"  built {arm}: norm={r['norm_from_sparse']:.10f} sha={r['weight_sha256'][:16]}", flush=True)
            cm.verify(arm, out)
        cmd = [sys.executable, str(HERE.parent / "fresh_dual_skill_instrument_v1" / "score_arms.py"),
               "--bank", str(HERE.parent / "fresh_dual_skill_instrument_v1" / "T_BANK_V2.json"),
               "--arms-root", str(HERE / "arms"),
               "--arms", "SPARSE_MLP", "SCALAR_MATCHED_MLP",
               "--output", str(HERE / "P6_SCORES.json"), "--no-lease"]
        print("  scoring ...", flush=True)
        raise SystemExit(subprocess.call(cmd, cwd=str(HERE.parent / "fresh_dual_skill_instrument_v1")))


if __name__ == "__main__":
    main()
