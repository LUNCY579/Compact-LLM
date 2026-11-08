"""Finalize the ABST-001 launch hash chain (ABST001_PREREG.md §9; R1). Run ONCE, after the code is final and the
dry battery (design/abst_harness.py) has passed on it:
  0. REFUSES unless design/ABST_HARNESS_PASS.json exists, says all_pass, is revision R1, and every module hash it
     recorded equals the file on disk now -- and those hashes cover every gated module and hostbudget (R1 review:
     'the pass log is a precondition' was procedural only)
  1. launch/abst_hashes.txt : sha256 of every gated module and data file (sha256sum -c format, POSIX paths),
     including the pass log itself
  2. launch/abst_run.sh     : __HASHES_SHA__ -> sha256(abst_hashes.txt)
  3. launch/launch_abst.ps1 : __RUN_SH_SHA__ -> sha256(abst_run.sh); __RUNNER_SHA__ -> sha256(launch/smallvd_runner.py)
Refuses if a placeholder is missing (already finalized), if abst_hashes.txt exists, or if a gated file is absent.
The gated set is every package module run_abst.py imports (directly or transitively), hostbudget, the panel and
every data file it reads, the teacher files, RTG1 dev/train (R1: the structure_id check), and the pinned venv's
interpreter, pyvenv.cfg and the version files of torch, transformers, tokenizers and safetensors (R1, cold review #34)."""
import hashlib
import json
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
RAB = PKG.parents[1] / "experiments" / "rab001_exploration"
BP = Path(r"C:\Users\clclc\BusinessProject")
REC = BP / "rineya-dataset-retention-records"
T = BP / "Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze"
S = REC / "smallvd001-20260924"
RTG1 = BP / "ChatBubble-rtg1/research/rineya_reasoning_corpus_v1/sft"
VENV = BP / ".venv-rab001-cu126"
SP = VENV / "Lib/site-packages"
MODULES = ["abst_verdict.py", "abst_measure.py", "abst_describe.py", "run_abst.py", "hlcc_verdict.py",
           "linear_compensation.py", "virtual_student.py", "layer_ablation.py", "layer_select.py",
           "run_layer_study.py", "param_accounting.py", "gpu_lease.py"]
DATA = [RAB / "hostbudget.py",
        REC / "abst001-20260925/panel/ABST_PANEL.jsonl", REC / "abst001-20260925/panel/ABST_MANIFEST.json",
        S / "calibration/LAYER_PANELS.jsonl", S / "calibration/TOKENIZER_CORPUS.jsonl",
        S / "layer_study/S1_teacher_outputs.json", S / "layer_study/S6_vocab_frontier.json", S / "lc_arms/LC_ARMS.json",
        REC / "hlcc001-20260925/result/HLCC_RESULT.json", REC / "hlcc001-20260925/panel/CONFIRM_PANEL.jsonl",
        S / "vocab/V4_32768/token_id_map.json", S / "vocab/V4_32768/tokenizer.json",
        S / "vocab/V2_65536/token_id_map.json", S / "vocab/V2_65536/tokenizer.json",
        T / "tokenizer.json", T / "config.json", T / "model.safetensors",
        RTG1 / "dev.jsonl", RTG1 / "train.jsonl",
        VENV / "Scripts/python.exe", VENV / "pyvenv.cfg", SP / "torch/version.py", SP / "transformers/__init__.py",
        SP / "tokenizers/__init__.py", SP / "safetensors/__init__.py", PKG / "design/ABST_HARNESS_PASS.json"]
DIRS = {PKG.name: PKG, RAB.name: RAB}


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def posix(p: Path) -> str:
    s = str(p).replace("\\", "/")
    return "/" + s[0].lower() + s[2:] if s[1] == ":" else s


def harness_gate() -> dict:
    log = json.loads((PKG / "design" / "ABST_HARNESS_PASS.json").read_text(encoding="utf-8"))
    if log.get("record") != "ABST_HARNESS_PASS" or log.get("revision") != "R1" or log.get("all_pass") is not True:
        raise SystemExit("the harness pass log is not an R1 all_pass log")
    rec = log["run_abst_code_sha256"]
    need = {f"{PKG.name}/{m}" for m in MODULES} | {f"{RAB.name}/hostbudget.py"}
    if not need <= set(rec):
        raise SystemExit(f"the pass log does not cover {sorted(need - set(rec))}")
    stale = [k for k, v in rec.items() if sha(DIRS[k.split("/")[0]] / k.split("/", 1)[1]) != v]
    if stale:
        raise SystemExit(f"the code changed since the harness passed: {stale}")
    return {"scenarios": len(log["scenarios"]), "utc": log["utc"]}


def main() -> int:
    hp = PKG / "launch" / "abst_hashes.txt"
    rs = PKG / "launch" / "abst_run.sh"
    ps = PKG / "launch" / "launch_abst.ps1"
    t, u = rs.read_bytes().decode("utf-8"), ps.read_bytes().decode("utf-8")
    if hp.exists() or t.count("__HASHES_SHA__") != 1 or u.count("__RUN_SH_SHA__") != 1 or u.count("__RUNNER_SHA__") != 1:
        raise SystemExit("already finalized (abst_hashes.txt exists or a placeholder is missing)")
    hg = harness_gate()
    lines = []
    for m in MODULES:
        p = PKG / m
        if not p.exists():
            raise SystemExit(f"missing {p}")
        lines.append(f"{sha(p)}  {m}")
    for p in DATA:
        if not p.exists():
            raise SystemExit(f"missing {p}")
        lines.append(f"{sha(p)}  {posix(p)}")
    with open(hp, "xb") as f:
        f.write(("\n".join(lines) + "\n").encode("utf-8"))
    hs = sha(hp)
    rs.write_bytes(t.replace("__HASHES_SHA__", hs).encode("utf-8"))
    rsh = sha(rs)
    rn = sha(PKG / "launch" / "smallvd_runner.py")
    ps.write_bytes(u.replace("__RUN_SH_SHA__", rsh).replace("__RUNNER_SHA__", rn).encode("utf-8"))
    print(f"harness pass log: {hg}\nabst_hashes.txt {hs}\nabst_run.sh {rsh}\nsmallvd_runner.py {rn}\n"
          f"launch_abst.ps1 {sha(ps)}\nlines {len(lines)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
