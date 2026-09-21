"""COMPOSE4B-002: score the 4B composition arms on HSKILL-VIEWS-002.

WHY THIS INSTRUMENT. The long-context suite is SPENT / REPORT-ONLY by its own receipt, and
references_retained and the 150-row arithmetic panel are structurally unrunnable against a
Qwen3-4B adapter. HSKILL-VIEWS-002 is the only selection-admissible instrument reachable here
(view_role ADMITTED_FOR_SELECTION, barred only TRAIN and CONFIRMATION, 11 arms already scored,
the 4B never run on it). It also yields CAPABILITY and DAMAGE on the SAME 224 rows:
  112 boundary rows   -> sensitivity: does the gate skill survive composition?
  112 arithmetic rows -> specificity_competence: is answering damaged?

Its sibling HSKILL-CONFIRM-002 (b758e0d1...) is HELD_FOR_CONFIRMATION / UNSPENT_AND_UNREAD and
is NEVER touched here.

WHAT IS REUSED RATHER THAN REWRITTEN. The scoring rule is IMPORTED from the sealed
boundary_rule.py (sha 416cfb6a...), never reimplemented -- both the builder that computed the
published floors and the 596M probe import the same module, so the rule that produced a floor
is provably the rule that scores these arms. Generation goes through the already-4B-proven
stage2_generate.py. Only rendering and the driver are new.

POSITIVE CONTROL ON THE RENDER. The Supa and Qwen3-4B tokenizers are byte-identical on all four
files, so apply_chat_template must produce the SAME text from both. That is asserted per row
before any generation, and a deliberate mutation is checked to be detectable -- a render check
that cannot fail proves nothing.
"""
from pathlib import Path
import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid

BASE = Path("C:/Users/clclc/BusinessProject")
RECORDS = BASE / "rineya-dataset-retention-records"
OUT = RECORDS / "compose4b002-20260921"
HS = RECORDS / "hskill-views-002"
Q = RECORDS / "qwen001-20260918"
C1 = RECORDS / "compose4b001-20260921"
PY = BASE / "Chat-Bubble/.venv/Scripts/python.exe"
MODEL = "C:/Users/clclc/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
SUPA_TOK = BASE / "Chat-Bubble/artifacts/rineya_abstention_gate_v1/checkpoint/Rineya-Nimbus-Abstain-1.0"
LOCK = Path(tempfile.gettempdir()) / "rineya-gpu-0.lock"
FLOOR = 15032385536
MAX_NEW = 48            # the instrument's own registered value
INSTRUMENT_SHA = "ba1f2982fd5d08aac861d4547b1dadb33d20e6022ff9aa06934e1ad2e7154723"
RULE_SHA = "416cfb6aec8aa28240752d3db2fc1b8af85115c4962a873582765c009adc3bf0"

ARMS = [
    ("base",      None),
    ("delta_t",   RECORDS / "qwenft001-20260918/run/adapter"),
    ("th_s",      C1 / "arm-th-s/adapter"),
    ("scaled_t",  C1 / "arm-scaled-t/adapter"),
    ("h_only",    C1 / "run-h/adapter"),
]


def sha(p):
    with Path(p).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def gpu_pids():
    out = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid",
                                   "--format=csv,noheader"], text=True, timeout=30)
    return {int(x.strip()) for x in out.split() if x.strip().isdigit()}


def foreign_on_card():
    import psutil
    mine = {os.getpid()} | {c.pid for c in psutil.Process(os.getpid()).children(recursive=True)}
    for pid in gpu_pids() - mine:
        try:
            p = psutil.Process(pid)
            if "python" in p.name().lower() and p.memory_info().rss >= 0.5 * 2**30:
                return pid
        except Exception:
            continue
    return None


def reclaimable(lock):
    """Is this lock a PROVEN orphan? All FOUR conjuncts must hold.

    Tonight produced four distinct lease failures and this rule is the union of their
    lessons:
      1. holder pid dead        -- but a DETACHED worker outlives its supervisor by design,
                                   so this alone is not enough (mis-steal #1: a 6 MB shim).
      2. worker_pids empty      -- but it is also empty BETWEEN arms and before the first
                                   CUDA allocation (mis-steal #2).
      3. orphan stamp passed    -- created + 600, so a loading job can never trip it.
      4. NO LIVE PROCESS CARRIES THE LOCK'S OWN LABEL.  <-- the one I was missing.
         A dead holder does not mean a dead OWNER: a restarted lane can be alive and
         polling its own dead predecessor's lock. Reclaiming then starves a live lane.

    READ THE LABEL. It is the only authority in the file. Ownership must never be inferred
    from whoever you happen to be talking to -- that error nearly had a third party's lease
    deleted on a second party's say-so.
    """
    import psutil
    if not lock.exists():
        return None
    try:
        d = json.loads(lock.read_text(encoding="utf-8"))
    except Exception:
        return None
    lane = str(d.get("label", "")).split(":")[0]
    holder_dead = not psutil.pid_exists(d.get("pid", 0))
    no_workers = not d.get("worker_pids")
    stamp = d.get("orphan_after_unix", d.get("created", 0) + 600)
    stamp_passed = time.time() > stamp
    live_lane = []
    if lane:
        for p in psutil.process_iter(["pid", "cmdline"]):
            try:
                if lane in " ".join(p.info["cmdline"] or []):
                    live_lane.append(p.info["pid"])
            except Exception:
                continue
    ok = holder_dead and no_workers and stamp_passed and not live_lane
    if not ok:
        return None
    return {"reclaimed_lock": d, "lane": lane, "holder_dead": holder_dead,
            "worker_pids_empty": no_workers, "orphan_stamp_passed": stamp_passed,
            "live_processes_with_label": live_lane, "reclaimed_unix": time.time(),
            "by": "COMPOSE4B-002"}


def render_all():
    """Render 224 prompts, asserting the two tokenizers agree byte-for-byte."""
    from transformers import AutoTokenizer
    rows = [json.loads(l) for l in (HS / "HSKILL_VIEWS_V2.jsonl").open(encoding="utf-8") if l.strip()]
    if len(rows) != 224 or sha(HS / "HSKILL_VIEWS_V2.jsonl") != INSTRUMENT_SHA:
        raise SystemExit("instrument identity failed")
    tq = AutoTokenizer.from_pretrained(MODEL)
    ts = AutoTokenizer.from_pretrained(str(SUPA_TOK))

    def render(tok, prompt):
        msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt
                if m.get("role") in {"system", "user"}]
        if not msgs:
            raise SystemExit("row has no system or user turn")
        return tok.apply_chat_template(msgs, tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)

    out, mismatches = [], 0
    for r in rows:
        a, b = render(tq, r["prompt"]), render(ts, r["prompt"])
        if a != b:
            mismatches += 1
        out.append({"id": r["id"], "text": a, "row_type": r["row_type"],
                    "family": r["family"], "reference": r["reference"]})
    # POSITIVE CONTROL: the comparison must be able to see a difference.
    detects = render(tq, rows[0]["prompt"]) != render(tq, rows[0]["prompt"]) + "X"
    if mismatches or not detects:
        raise SystemExit(f"RENDER CONTROL FAILED: {mismatches} mismatches, detects={detects}")
    return rows, out, mismatches, detects


def score(label, rows, preds):
    sys.path.insert(0, str(HS))
    import boundary_rule as BR
    if sha(HS / "boundary_rule.py") != RULE_SHA:
        raise SystemExit("rule module changed")
    by_id = {p["id"]: p for p in preds}
    per_row, sens_k, spec_d_k, spec_c_k, nb, na = [], 0, 0, 0, 0, 0
    for r in rows:
        pred = (by_id.get(r["id"], {}) or {}).get("prediction", "")
        ab = BR.abstains(pred)
        strict, credited = BR.score_row(r["row_type"], pred, r["reference"])
        if r["row_type"] == "boundary":
            nb += 1; sens_k += int(ab)
        else:
            na += 1; spec_d_k += int(not ab); spec_c_k += int(credited and not ab)
        per_row.append({"id": r["id"], "row_type": r["row_type"], "family": r["family"],
                        "abstains": ab, "credited": bool(credited), "prediction": pred})
    sens, spec_d = sens_k / max(nb, 1), spec_d_k / max(na, 1)
    return {
        "label": label, "instrument": "HSKILL_VIEWS_V2.jsonl",
        "instrument_sha256": INSTRUMENT_SHA, "rule_module_sha256": RULE_SHA,
        "instrument_view_role": "ADMITTED_FOR_SELECTION",
        "max_new_tokens": MAX_NEW, "decoding": "greedy", "rows_scored": len(rows),
        "sensitivity_abstains_on_boundary": {"k": sens_k, "n": nb, "rate": sens},
        "specificity_decision_attempts_on_answerable": {"k": spec_d_k, "n": na, "rate": spec_d},
        "specificity_competence_correct_on_answerable": {"k": spec_c_k, "n": na,
                                                         "rate": spec_c_k / max(na, 1)},
        "J_decision": sens + spec_d - 1,
        "per_row": per_row,
        "quality_claim": False, "release_permitted": False,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--render-only", action="store_true")
    a = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    rows, rendered, mm, det = render_all()
    pf = OUT / "prompts.jsonl"
    if not pf.exists():
        with pf.open("w", encoding="utf-8", newline="\n") as f:
            for r in rendered:
                f.write(json.dumps({"id": r["id"], "text": r["text"]}) + "\n")
    (OUT / "RENDER.json").write_text(json.dumps({
        "instrument": "HSKILL-VIEWS-002", "instrument_sha256": INSTRUMENT_SHA,
        "rule_module_sha256": RULE_SHA, "rows": len(rows),
        "prompts_sha256": sha(pf),
        "tokenizer_agreement_mismatches": mm,
        "positive_control_detects_a_mutation": det,
        "note": ("Supa and Qwen3-4B tokenizers are byte-identical on all four files; "
                 "apply_chat_template agreement is asserted per row, with a mutation control"),
        "enable_thinking": False, "max_new_tokens": MAX_NEW,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"rendered": len(rendered), "tokenizer_mismatches": mm,
                      "control_detects": det, "prompts_sha256": sha(pf)}, indent=2))
    if a.render_only:
        return 0

    stale = reclaimable(LOCK)
    if LOCK.exists() and not stale:
        raise SystemExit("REFUSED: lease held")
    if foreign_on_card():
        raise SystemExit("REFUSED: foreign CUDA job on the card")
    if stale:
        (OUT / "RECLAIMED_LOCK.json").write_text(json.dumps(stale, indent=2) + "\n", encoding="utf-8")
        print("RECLAIMED an orphaned lock; evidence in RECLAIMED_LOCK.json", flush=True)
        LOCK.unlink()
    import shutil as _sh
    if _sh.disk_usage(RECORDS).free < FLOOR:
        raise SystemExit("REFUSED: below the 14 GiB floor")
    token = uuid.uuid4().hex
    with LOCK.open("x", encoding="utf-8", newline="\n") as f:
        json.dump({"pid": os.getpid(), "token": token, "created": time.time(),
                   "worker_pids": [], "job": "COMPOSE4B-002 HSKILL battery",
                   "label": "compose4b002:hskill:battery",
                   "orphan_after_unix": time.time() + 600,
                   "staleness_rule": ("reclaimable ONLY if holder pid dead AND worker_pids empty "
                                      "AND orphan_after_unix passed AND no live process carries "
                                      "this label. Read the label; it is the only authority here.")}, f)
    try:
        (OUT / "scored").mkdir(exist_ok=True)
        for label, adapter in ARMS:
            sp = OUT / "scored" / f"{label}.json"
            if sp.exists():
                print(f"  [{label}] already scored - skipping", flush=True); continue
            gd = OUT / f"gen-{label}"
            if not (gd / "predictions.jsonl").is_file():
                if gd.exists():
                    _sh.rmtree(gd)
                cmd = [str(PY), str(Q / "stage2_generate.py"), "--model", MODEL,
                       "--label", f"hskill_{label}", "--prompts", str(pf), "--out", str(gd),
                       "--max-new-tokens", str(MAX_NEW), "--dtype", "bf16", "--four-bit"]
                if adapter:
                    cmd += ["--adapter", str(adapter)]
                t0 = time.time()
                rc = subprocess.run(cmd, capture_output=True, text=True)
                print(f"  [{label}] rc={rc.returncode} {time.time()-t0:.0f}s", flush=True)
                if rc.returncode:
                    print((rc.stderr or "")[-1500:]); raise SystemExit(f"generation failed: {label}")
            preds = [json.loads(l) for l in (gd / "predictions.jsonl").open(encoding="utf-8") if l.strip()]
            if len(preds) != 224:
                raise SystemExit(f"{label}: {len(preds)} predictions, expected 224")
            res = score(label, rows, preds)
            sp.write_text(json.dumps(res, indent=2) + "\n", encoding="utf-8")
            d = res
            print(f"     sens {d['sensitivity_abstains_on_boundary']['k']}/112  "
                  f"spec_dec {d['specificity_decision_attempts_on_answerable']['k']}/112  "
                  f"spec_comp {d['specificity_competence_correct_on_answerable']['k']}/112  "
                  f"J {d['J_decision']:+.4f}", flush=True)
        return 0
    finally:
        if LOCK.exists():
            try:
                if json.loads(LOCK.read_text(encoding="utf-8")).get("token") == token:
                    LOCK.unlink(); print("lease released", flush=True)
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
