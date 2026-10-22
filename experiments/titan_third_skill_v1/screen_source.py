"""TRISKILL-001 Phase 0: source-competence screen on SIF-001 DEV atomic banks.

Registered in PHASE0_PREREG.md before any model read. Writes only inside this directory.
Scoring is SIF-001's own scoring.score_records, imported from the dataset, never reimplemented.
Launch through WMI (interference_mechanism_v1/wmi_runner.py) with the hermes interpreter.
"""
from __future__ import annotations

import gc, gzip, hashlib, json, platform, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
SIF = Path(r"C:/Users/clclc/BusinessProject/Rineya-Dataset/Rineya_Skill_Integration_Foundry_v1")
CORES = Path(r"C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores")
BANKS = ["A", "C", "P", "R", "S"]
ARMS = {
    "BASE": (CORES / "Rineya-Supa-1.7.0-devfreeze",
             "962abc5bcd5a83e3b7693bd278ae9215b6d83389a847913da9330b34dd5dd24b"),
    "SPARSE": (CORES / "Rineya-Titan-Sparse-Experimental-1.0.0-devfreeze",
               "a28629c9d3454fbf2e884fd7ca442c1b406e7caf4b36d34b2c0614d0add31284"),
    "TT": (CORES / "Rineya-Titan-Transition-1.0.0-devfreeze",
           "a4d73353db9a91cbba7b4cd7c8fe4fe6822670d10fe523b26a6d651bb9509567"),
}
OUT = HERE / "PHASE0_SCORES.json"
PRED_DIR = HERE / "phase0_predictions"
MAX_NEW = 48

sys.path.insert(0, str(EXP / "fresh_dual_skill_instrument_v1"))
sys.path.insert(0, str(SIF / "src"))
from score_arms import render, template_selftest          # noqa: E402  (self-tested soup-lane template)
from sif.scoring import score_records, score_answer       # noqa: E402  (the dataset's own scorer)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 24), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank(bank: str):
    inputs = [json.loads(l) for l in (SIF / "data/eval/dev" / f"atomic_{bank}.inputs.jsonl")
              .read_text(encoding="utf-8").splitlines() if l.strip()]
    keys = [json.loads(l) for l in gzip.open(SIF / "private_oracles/dev" / f"atomic_{bank}.keys.jsonl.gz",
                                              "rt", encoding="utf-8")]
    kid = {k["id"]: k for k in keys}
    if len(inputs) != 128 or set(kid) != {r["id"] for r in inputs}:
        raise SystemExit(f"bank {bank}: unexpected shape")
    for r in inputs:
        ph = hashlib.sha256(r["messages"][1]["content"].encode()).hexdigest()
        if ph != kid[r["id"]]["prompt_sha256"]:
            raise SystemExit(f"bank {bank}: prompt hash mismatch {r['id']}")
    return inputs, keys


def scorer_canary(banks) -> dict:
    n = 0
    for bank, (inputs, keys) in banks.items():
        for k in keys:
            if not score_answer(k["completion"], k["answer"], k["response_format"])["strict_correct"]:
                raise SystemExit(f"SCORER CANARY: gold not strict-correct {k['id']}")
            if score_answer('{"answer":99}', k["answer"], k["response_format"])["strict_correct"]:
                raise SystemExit(f"SCORER CANARY: wrong answer accepted {k['id']}")
            if k["response_format"] == "json":
                if score_answer(k["completion"] + ".", k["answer"], "json")["format_valid"]:
                    raise SystemExit(f"SCORER CANARY: trailing period accepted as JSON {k['id']}")
            n += 1
    # the canary must be able to fail: a shuffled-gold scorer run must NOT be all-correct
    inputs, keys = banks["C"]
    shifted = [{"id": k["id"], "prompt_sha256": k["prompt_sha256"], "completion": keys[(i + 1) % len(keys)]["completion"]}
               for i, k in enumerate(keys)]
    summ, _ = score_records(keys, shifted)
    if summ["groups"]["all_rows_diagnostic"]["strict_accuracy"] >= 0.99:
        raise SystemExit("SCORER CANARY IS VACUOUS: shuffled gold scored as correct")
    return {"rows_checked": n, "shuffled_gold_strict_accuracy_C": summ["groups"]["all_rows_diagnostic"]["strict_accuracy"]}


def unit_path(arm: str, bank: str) -> Path:
    return PRED_DIR / f"{arm}__atomic_{bank}.jsonl"


def unit_complete(arm: str, bank: str, inputs) -> bool:
    """A unit counts as done only if its file holds exactly this bank's ids with matching prompt hashes.

    Files are written atomically (tmp + rename), so a crash mid-bank leaves no file, never a short one.
    """
    p = unit_path(arm, bank)
    if not p.exists():
        return False
    preds = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    want = {r["id"]: hashlib.sha256(r["messages"][1]["content"].encode()).hexdigest() for r in inputs}
    got = {x["id"]: x["prompt_sha256"] for x in preds}
    if len(preds) != len(inputs) or got != want:
        raise SystemExit(f"CORRUPT UNIT {p.name}: refusing to resume over it")
    return True


def main() -> int:
    """Amendment 1 (ATTEMPT_LOG.md): per-(arm, bank) checkpointing, so a GPU driver reset costs one
    bank, not the run. Generation, decoding and scoring are unchanged."""
    print("startup checks", flush=True)
    if OUT.exists():
        raise SystemExit(f"refusing to overwrite {OUT}")
    template_selftest()
    banks = {b: load_bank(b) for b in BANKS}
    canary = scorer_canary(banks)
    print(f"  scorer canary PASS: {canary}", flush=True)
    input_hashes = {f"dev/atomic_{b}.inputs.jsonl": sha256_file(SIF / "data/eval/dev" / f"atomic_{b}.inputs.jsonl") for b in BANKS}
    input_hashes.update({f"dev/atomic_{b}.keys.jsonl.gz": sha256_file(SIF / "private_oracles/dev" / f"atomic_{b}.keys.jsonl.gz") for b in BANKS})
    weight_hashes = {}
    for arm, (d, pin) in ARMS.items():
        got = sha256_file(d / "model.safetensors")
        if got != pin:
            raise SystemExit(f"WEIGHT HASH MISMATCH {arm}: {got}")
        weight_hashes[arm] = got
    print(f"  weight hashes verified for {sorted(weight_hashes)}", flush=True)

    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from gpu_lease import lease, LOCK
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    result = {
        "schema": 1, "experiment": "TRISKILL-001", "phase": "0-source-competence-screen",
        "registered_in": "PHASE0_PREREG.md",
        "instrument": "SIF-001 DEV atomic A/C/P/R/S, 128 rows each",
        "sif_root": str(SIF), "input_hashes": input_hashes,
        "scorer": {"module": "sif.scoring.score_records", "sha256": sha256_file(SIF / "src/sif/scoring.py")},
        "collector_sha256": sha256_file(Path(__file__)),
        "weights": weight_hashes, "scorer_canary": canary,
        "runtime": {"python": platform.python_version(), "torch": torch.__version__,
                    "transformers": transformers.__version__, "device": torch.cuda.get_device_name(0),
                    "dtype": "float32", "tf32": False, "decoding": "greedy", "batch_size": 1,
                    "max_new_tokens": MAX_NEW, "thinking_disabled": True},
        "gpu_lease": str(LOCK), "arms": {},
        "quality_claim": False, "release_permitted": False,
    }
    PRED_DIR.mkdir(exist_ok=True)
    todo = {arm: [b for b, (inputs, _) in banks.items() if not unit_complete(arm, b, inputs)] for arm in ARMS}
    print(f"  units remaining: { {a: v for a, v in todo.items() if v} }", flush=True)
    if any(todo.values()):
        with lease(label="triskill001-phase0", job="source screen: 3 arms x 640 SIF dev rows (resumable)"):
            for arm, (d, _) in ARMS.items():
                if not todo[arm]:
                    continue
                tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
                model = AutoModelForCausalLM.from_pretrained(str(d), torch_dtype=torch.float32, local_files_only=True,
                                                             low_cpu_mem_usage=True, device_map={"": "cuda"})
                model.eval()
                if next(model.parameters()).dtype != torch.float32:
                    raise SystemExit(f"{arm}: not FP32")
                for bank in todo[arm]:
                    inputs, _ = banks[bank]
                    preds = []
                    for r in inputs:
                        text = render(r["messages"][0]["content"], r["messages"][1]["content"])
                        enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
                        with torch.inference_mode():
                            out = model.generate(**enc, do_sample=False, max_new_tokens=MAX_NEW,
                                                 eos_token_id=tok.eos_token_id, pad_token_id=tok.eos_token_id)
                        gen = tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                        preds.append({"id": r["id"], "prompt_sha256": hashlib.sha256(r["messages"][1]["content"].encode()).hexdigest(),
                                      "completion": gen})
                    pp = unit_path(arm, bank)
                    tmp = pp.with_suffix(".jsonl.tmp")
                    tmp.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in preds), encoding="utf-8", newline="\n")
                    tmp.replace(pp)
                    print(f"  unit done {arm}/{bank}", flush=True)
                del model, tok
                gc.collect(); torch.cuda.empty_cache()
    # Aggregate from the unit files only, so a resumed run and a single run score identically.
    for arm in ARMS:
        arm_rec = {}
        for bank, (inputs, keys) in banks.items():
            pp = unit_path(arm, bank)
            preds = [json.loads(l) for l in pp.read_text(encoding="utf-8").splitlines() if l.strip()]
            # SIF scorer strips the completion itself; pass the raw generation.
            summary, rows = score_records(keys, preds)
            arm_rec[bank] = {"summary": summary, "rows": rows, "predictions_sha256": sha256_file(pp)}
            g = summary["groups"]
            js = [x for x in rows if x["variant"] != "alternate_format"]
            print(json.dumps({"arm": arm, "bank": bank,
                              "strict_json96": sum(x["strict_correct"] for x in js),
                              "semantic_json96": sum(x["semantic_correct"] for x in js),
                              "format_json96": sum(x["format_valid"] for x in js),
                              "strict_all128": round(g["all_rows_diagnostic"]["strict_accuracy"] * 128)}), flush=True)
        result["arms"][arm] = {"banks": arm_rec}
    OUT.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
