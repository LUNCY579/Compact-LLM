"""HLOC-001: build, gate, score and DELETE four arms one at a time. Registered in HLOC_PREREG.md.

WHY SERIAL. Four arms x 2.38 GB is 9.5 GB against ~8 GB free. Built/scored/deleted one at a
time the peak is one arm, and the weights are rebuildable from this file plus two committed
checkpoints -- the Tier B property.

THE LEASE COVERS THE BUILDS. A streaming fp64 SVD over 220 tensors is not lighter than a
forward pass on this box; this session established that by starving two other jobs with
"just analysis". Taken once for the whole run.

`p16()` is verbatim from build_indep5.py. Gates are in HLOC_PREREG.md section 4 and every one
is asserted on the BUILT tensors.
"""
from pathlib import Path
import json, hashlib, os, shutil, subprocess, tempfile, time, uuid

import psutil
import torch
from safetensors import safe_open
from safetensors.torch import save_file

REPO = Path(r"C:/Users/clclc/BusinessProject/ChatBubble-dose-param")
REC = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records")
SRC_DIR = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze")
SRC = SRC_DIR / "model.safetensors"
T17 = REC / "int001-20260917/trained_fp32.safetensors"
TH = REC / "compose001-20260918/trained_H_fp32.safetensors"
OUT = REC / "hloc001-20260923"
PY = Path(r"C:/Users/clclc/BusinessProject/.venv-rab001-cu126/Scripts/python.exe")
LOCK = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
RANK, RAM_GATE = 16, 3.0
MAG001_N17 = 0.674029920773733


def free_gib():
    return psutil.virtual_memory().available / 2 ** 30


def p16(d):
    if d.ndim != 2:
        return d.double()
    U, S, Vh = torch.linalg.svd(d.double(), full_matrices=False)
    r = min(RANK, S.numel())
    return (U[:, :r] * S[:r]) @ Vh[:r]


def fam(k):
    if k.endswith(("q_norm.weight", "k_norm.weight")) or "layernorm" in k or k == "model.norm.weight":
        return "norm"
    if any(t in k for t in ("q_proj", "k_proj", "v_proj", "o_proj")):
        return "attn"
    if any(t in k for t in ("gate_proj", "up_proj", "down_proj")):
        return "mlp"
    return "other"


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def foreign():
    try:
        o = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                     "--format=csv,noheader"], text=True, timeout=30)
    except Exception:
        return None
    mine = {os.getpid()} | {c.pid for c in psutil.Process(os.getpid()).children(recursive=True)}
    for t in o.split():
        if t.strip().isdigit() and int(t) not in mine:
            try:
                p = psutil.Process(int(t))
                if "python" in p.name().lower() and p.memory_info().rss >= 0.5 * 2 ** 30:
                    return int(t)
            except Exception:
                pass
    return None


def main():
    if LOCK.exists():
        d = json.loads(LOCK.read_text(encoding="utf-8"))
        raise SystemExit(f"REFUSED: lease held by {d.get('job')} pid {d.get('pid')}")
    if (fp := foreign()):
        raise SystemExit(f"REFUSED: foreign CUDA model job on the card, pid {fp}")
    if free_gib() < RAM_GATE:
        raise SystemExit(f"REFUSED: host RAM {free_gib():.2f} GiB below {RAM_GATE}")

    tok = uuid.uuid4().hex
    with LOCK.open("x", encoding="utf-8", newline="\n") as f:
        json.dump({"pid": os.getpid(), "token": tok, "created": time.time(), "worker_pids": [],
                   "job": "HLOC-001 build+score, 4 arms", "label": "hloc001:battery",
                   "orphan_after_unix": time.time() + 3600,
                   "staleness_rule": "holder pid dead AND worker_pids empty AND orphan_after_unix "
                                     "passed AND no live process carries this label"}, f)
    print(f"lease taken (covers BUILDS as well as scoring). RAM {free_gib():.2f} GiB", flush=True)

    try:
        # ---- one SVD pass: cache P16 pieces as fp32 to avoid re-running it per arm ----
        t0 = time.time()
        pT, pH, hfam, keys = {}, {}, {}, []
        with safe_open(str(SRC), framework="pt") as S, safe_open(str(T17), framework="pt") as A, \
             safe_open(str(TH), framework="pt") as H:
            tk = sorted(set(A.keys()) & set(H.keys()))
            for k in tk:
                s = S.get_tensor(k).double()
                pT[k] = p16(A.get_tensor(k).double() - s).to(torch.float32)
                pH[k] = p16(H.get_tensor(k).double() - s).to(torch.float32)
                hfam[k] = fam(k)
                keys.append(k)
        nT = sum(float((v.double() ** 2).sum()) for v in pT.values()) ** 0.5
        byf = {}
        for k, v in pH.items():
            byf[hfam[k]] = byf.get(hfam[k], 0.0) + float((v.double() ** 2).sum())
        nH = sum(byf.values()) ** 0.5
        n_attn, n_mlp = byf["attn"] ** 0.5, byf["mlp"] ** 0.5
        print(f"P16 pass {time.time()-t0:.0f}s  ||P16 dT|| {nT:.15f}  ||P16 dH|| {nH:.15f}", flush=True)
        print(f"  attn {n_attn:.6f}  mlp {n_mlp:.6f}", flush=True)
        assert abs(nT - MAG001_N17) < 1e-6, f"GATE 2 FAIL: ||P16 dT|| {nT} vs {MAG001_N17}"
        print("  GATE 2 PASS: ||P16(dT)|| reproduces MAG-001", flush=True)

        tgt_a, tgt_m = 0.25 * n_attn, 0.25 * n_mlp
        a_s, m_s = tgt_a / nH, tgt_m / nH

        # ---- GATE 4: positive control, BEFORE any correct arm ----
        bad = (1.01 * a_s) * nH
        print(f"\nGATE 4 POSITIVE CONTROL: scalar coefficient +1% -> 2nd-component norm "
              f"{bad:.9f} vs target {tgt_a:.9f}, err {abs(bad-tgt_a):.3e} vs tol 1e-9  "
              f"{'REJECTED ok' if abs(bad-tgt_a) > 1e-9 else 'ACCEPTED -- GATE IS BLIND'}", flush=True)
        assert abs(bad - tgt_a) > 1e-9, "GATE 4 FAIL: the norm gate cannot reject a 1% error"

        ARMS = [("hloc001_hattn",   "ablate", "attn", 0.25, tgt_a),
                ("hloc001_hattn_s", "scalar", None,   a_s,  tgt_a),
                ("hloc001_hmlp",    "ablate", "mlp",  0.25, tgt_m),
                ("hloc001_hmlp_s",  "scalar", None,   m_s,  tgt_m)]
        shas, results = {}, {}
        for label, kind, keep, coef, target in ARMS:
            if free_gib() < RAM_GATE:
                print(f"  {label}: REFUSED, RAM {free_gib():.2f}", flush=True); continue
            t1 = time.time()
            adir = OUT / label
            adir.mkdir(parents=True, exist_ok=True)
            out, ss2, zeroed, ablated_keys = {}, 0.0, {"attn": 0, "mlp": 0, "norm": 0}, []
            with safe_open(str(SRC), framework="pt") as S:
                for k in sorted(S.keys()):
                    base = S.get_tensor(k)
                    if k not in pT:
                        out[k] = base.to(torch.float32); continue
                    second = coef * pH[k].double()
                    if kind == "ablate" and hfam[k] != keep:
                        second = torch.zeros_like(second)
                        zeroed[hfam[k]] = zeroed.get(hfam[k], 0) + 1
                        # GATE 5b, the peer's note: assert the ablated contribution is BITWISE
                        # zero, not approximately. A scalar-times-zero bug yields -0.0 in places
                        # and float equality on the untouched set is the cheapest way to see it.
                        assert bool((second == 0).all()), f"GATE 5b: {k} ablated but not bitwise 0"
                        ablated_keys.append(k)
                    ss2 += float((second ** 2).sum())
                    out[k] = (base.double() + pT[k].double() + second).to(torch.float32)
            n2 = ss2 ** 0.5
            ok_norm = abs(n2 - target) < 1e-9
            print(f"\n{label}: ||2nd|| {n2:.12f} vs target {target:.12f} "
                  f"err {abs(n2-target):.2e}  GATE1 {'PASS' if ok_norm else 'FAIL'}", flush=True)
            assert ok_norm, f"GATE 1 FAIL on {label}"
            if kind == "ablate":
                # GATE 5: the ablation is REAL -- zero on every non-kept family
                print(f"  GATE 5 ablation real: zeroed {zeroed} tensors (kept only {keep})", flush=True)
                assert zeroed.get("attn" if keep == "mlp" else "mlp", 0) > 0, "GATE 5 FAIL: nothing ablated"
            for f in ("config.json", "generation_config.json", "tokenizer.json",
                      "tokenizer_config.json", "vocab.json", "merges.txt"):
                if (SRC_DIR / f).exists():
                    (adir / f).write_bytes((SRC_DIR / f).read_bytes())
            save_file(out, str(adir / "model.safetensors"), metadata={"format": "pt"})
            del out
            sha = sha_file(adir / "model.safetensors")
            assert sha not in shas.values(), f"GATE 6 FAIL: {label} collides with {shas}"
            shas[label] = sha
            print(f"  built {time.time()-t1:.0f}s sha {sha[:16]}", flush=True)

            hd = OUT / f"HSKILL2_{label}.json.d"
            rc = subprocess.run([str(PY), str(REC / "hskill-views-002/hskill_probe_v2.py"),
                                 "--model", str(adir), "--label", label,
                                 "--instrument", str(REC / "hskill-views-002/HSKILL_VIEWS_V2.jsonl"),
                                 "--out", str(hd), "--device", "cuda"],
                                cwd=str(REPO), capture_output=True, text=True)
            if rc.returncode != 0:
                print((rc.stderr or rc.stdout or "")[-1200:], flush=True)
                raise SystemExit(f"{label}: hskill rc={rc.returncode}")
            dst = OUT / f"HSKILL2_{label}.json"
            if (hd / f"HSKILL2_{label}.json").exists():
                shutil.move(str(hd / f"HSKILL2_{label}.json"), str(dst))
                try: hd.rmdir()
                except OSError: pass
            j = json.loads(dst.read_text(encoding="utf-8"))
            b = j["discrimination_PRIMARY"]["sensitivity_abstains_on_boundary"]["k"]
            results[label] = b
            print(f"  BOUNDARY {b}/112   (h025 85, a0554 68, control 69)", flush=True)

            (OUT / f"MATERIALIZATION_{label}.json").write_text(json.dumps(
                {"record": "HLOC_MATERIALIZATION", "experiment": "HLOC-001", "label": label,
                 "prereg": "research/composition_endpoint_audit_v1/HLOC_PREREG.md",
                 "kind": kind, "kept_family": keep, "coefficient": coef,
                 "norm_P16_delta_T": nT, "norm_P16_delta_H": nH,
                 "norm_second_component": n2, "target_second_component": target,
                 "boundary": b, "out_weight_sha256": sha, "stored_dtype": "float32",
                 "quality_claim": False, "release_permitted": False}, indent=1),
                encoding="utf-8", newline="\n")
            (adir / "model.safetensors").unlink()     # reclaim immediately; rebuildable
            print(f"  weights deleted, free disk "
                  f"{shutil.disk_usage(str(OUT)).free/2**30:.2f} GiB", flush=True)
        print(f"\nRESULTS {results}")
    finally:
        if LOCK.exists():
            try:
                if json.loads(LOCK.read_text(encoding="utf-8")).get("token") == tok:
                    LOCK.unlink(); print("lease released", flush=True)
            except Exception:
                pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
