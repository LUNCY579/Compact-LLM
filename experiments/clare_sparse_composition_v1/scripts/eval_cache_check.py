"""Native check of the evaluation-only FP32 weight caches (CALIBRATION rows only, never DEV/TEST).

    python scripts/eval_cache_check.py --qualification qualification/<native dir>/NATIVE_QUALIFICATION.json --out <json>

1. CANARY: with caches ON, regenerate the qualification's recorded base-competence examples and
   require byte-identical responses (the recorded ones came from the uncached path).
2. Time the same rows uncached vs cached and require identical outputs.
Writes one JSON report (create-only). Takes the shared GPU lease; preflight as for a probe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qualification", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    from cli import MAX_NEW_TOKENS, _blocked, _load_stack
    from clare import resources
    from clare.config import StudyConfig
    from clare.data import Corpus
    from clare.io import read_json, write_json

    cfg = StudyConfig.load()
    q = read_json(args.qualification)
    if q.get("status") != "QUALIFIED":
        raise SystemExit("the reference qualification is not QUALIFIED")
    pre = resources.preflight("probe", args.out.parent, artifact_gib=0.01, vram_need_gib=4.0)
    if pre["status"] != "READY":
        return _blocked(args.out, {"preflight": pre, "blockers": pre["blockers"]})
    gl, hb = resources.gpu_lease(), resources.hostbudget()
    rec = {"experiment": "CLARE-RINEYA-001", "check": "eval-weight-cache", "reference": str(args.qualification)}
    with gl.lease(label="CLARE-RINEYA-001", job="eval-cache-check", timeout_seconds=600):
        hb.require(3.0, "clare-eval-cache-check")
        native, runtime, tok, model, fp, patch, host = _load_stack(cfg, "cuda")
        rec["runtime"] = runtime
        model.eval()
        corpus = Corpus(cfg.raw["dataset"]["root"])
        # the same selection as the qualification's base-competence smoke
        cal = []
        for sig in ("R", "U", "S", "P", "RU", "US", "SP"):
            rows = sorted(corpus.rows("calibration", "calibration", [sig]),
                          key=lambda r: hashlib.sha256(r.id.encode()).hexdigest())[:6]
            cal.extend(rows)
        recorded = q["base_competence_calibration"]["examples"]
        probe = cal[:len(recorded)]
        skip = [s.name[: -len(".weight")] for s in fp]

        def run(rows):
            t0 = time.time()
            out = [tok.decode(native.greedy_generate(model, native.prompt_ids(r, tok), "cuda", MAX_NEW_TOKENS)["ids"],
                              skip_special_tokens=False) for r in rows]
            return out, (time.time() - t0) / len(rows)

        native.set_eval_weight_cache(model, False)
        plain, t_plain = run(probe)
        native.set_eval_weight_cache(model, True, skip=skip)
        host.cache_effective(True)
        cached, t_cached = run(probe)
        canary = [c[:160] == e["response"] for c, e in zip(cached, recorded)]
        rec.update({"rows": len(probe), "s_per_row_uncached": round(t_plain, 3), "s_per_row_cached": round(t_cached, 3),
                    "speedup": round(t_plain / t_cached, 2), "identical_outputs": plain == cached,
                    "canary_matches_recorded": sum(canary), "canary_rows": len(canary),
                    "status": "PASS" if (plain == cached and all(canary)) else "FAIL"})
    write_json(args.out, rec)
    print(json.dumps({k: rec[k] for k in ("status", "s_per_row_uncached", "s_per_row_cached", "speedup",
                                          "identical_outputs", "canary_matches_recorded", "canary_rows")}))
    return 0 if rec["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
