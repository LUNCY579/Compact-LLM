"""CONFIRM-002 boundary scorer: score arms on an HSKILL instrument with the native probe's contract.

Mirrors rineya-dataset-retention-records/hskill-views-002/hskill_probe_v2.py:
- rendering: tokenizer.apply_chat_template(system+user, add_generation_prompt=True, enable_thinking=False);
- tokenisation: add_special_tokens=False;
- decoding: greedy, max_new_tokens 48, stopping on ANY EOS id listed in the arm's generation_config.json
  (the probe's run._eos_ids);
- decode with skip_special_tokens=True, then strip.
Scoring is imported from boundary_rule.py (sha recorded):
  boundary abstention  = score_row('boundary', reply, ref)[1]  (credited, first line)
  answerable attempt   = not abstains(first_line(reply))
  answerable correct   = score_row('answerable', reply, ref)[1] (canonical exact match, first line)
The harness differs from the probe's model class (HF transformers here), so its equivalence is a GATE:
Stage A's canary must reproduce the recorded artifacts (PREREG §4.1).

REFUSES, by name and before opening anything, the sealed HSKILL_CONFIRM_V2.jsonl. Accepts only banks
whose sha256 is in ALLOWED. Create-only output.
"""
from __future__ import annotations

import argparse, gc, hashlib, json, platform, sys, time
from pathlib import Path

HV = Path(r"C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/hskill-views-002")
sys.path.insert(0, str(HV))
import boundary_rule as BR  # noqa: E402

RULE_SHA_PREFIX = "416cfb6a"
ALLOWED = {
    "ba1f2982fd5d08aac861d4547b1dadb33d20e6022ff9aa06934e1ad2e7154723": "HSKILL_VIEWS_V2 (selection bank, exposed)",
    "686df45b94fa000be8df8d15e1f026a7f74e22deadfa4aacda96f5c6244c8ad5": "HSKILL_SOUP_V3 (CONFIRM-002 fresh bank)",
}
FORBIDDEN_NAMES = {"HSKILL_CONFIRM_V2.jsonl"}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()


def render(tok, prompt):
    msgs = [{"role": str(m["role"]), "content": str(m["content"])} for m in prompt if m.get("role") in {"system", "user"}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)


def score_reply(row, reply):
    if row["row_type"] == "boundary":
        return {"credited": bool(BR.score_row("boundary", reply, row["reference"])[1])}
    return {"credited": bool(BR.score_row("answerable", reply, row["reference"])[1]),
            "attempt": not BR.abstains(BR.first_line(reply))}


def summarise(preds):
    b = [p for p in preds if p["row_type"] == "boundary"]; a = [p for p in preds if p["row_type"] == "answerable"]
    return {"boundary_abstain": sum(p["credited"] for p in b), "n_boundary": len(b),
            "answerable_attempt": sum(p["attempt"] for p in a), "answerable_correct": sum(p["credited"] for p in a), "n_answerable": len(a)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bank", type=Path, required=True)
    ap.add_argument("--arms-root", type=Path, required=True)
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    ap.add_argument("--no-lease", action="store_true")
    a = ap.parse_args()
    if a.bank.name in FORBIDDEN_NAMES:
        raise SystemExit("REFUSING: that is the sealed HSKILL-CONFIRM-002 instrument; CONFIRM-002 never reads it")
    if a.output.exists():
        raise SystemExit(f"refusing to overwrite {a.output}")
    bsha = sha256(a.bank)
    if bsha not in ALLOWED:
        raise SystemExit(f"bank sha {bsha[:12]} is not an allowed CONFIRM-002 instrument")
    rule_sha = sha256(HV / "boundary_rule.py")
    if not rule_sha.startswith(RULE_SHA_PREFIX):
        raise SystemExit("boundary_rule.py has changed")
    rows = [json.loads(l) for l in a.bank.read_text(encoding="utf-8").splitlines() if l.strip()]

    import torch, transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.set_per_process_memory_fraction(0.78, 0)    # past VRAM, Windows pages into host RAM instead of failing

    result = {"schema": 1, "instrument": "CONFIRM-002 boundary scorer", "bank": a.bank.name, "bank_sha256": bsha,
              "bank_label": ALLOWED[bsha], "rule_module_sha256": rule_sha,
              "runtime": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
                          "device": torch.cuda.get_device_name(0), "dtype": "float32", "decoding": "greedy",
                          "max_new_tokens": a.max_new_tokens, "thinking_disabled": True, "tf32": False},
              "arms": [], "quality_claim": False}
    for name in a.arms:
        d = a.arms_root / name
        eos = json.loads((d / "generation_config.json").read_text(encoding="utf-8"))["eos_token_id"]
        eos = eos if isinstance(eos, list) else [eos]
        t0 = time.time()
        tok = AutoTokenizer.from_pretrained(str(d), local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(str(d), torch_dtype=torch.float32, local_files_only=True,
                                                     low_cpu_mem_usage=True, device_map={"": "cuda"}).eval()
        preds = []
        for r in rows:
            enc = tok(render(tok, r["prompt"]), return_tensors="pt", add_special_tokens=False)
            enc = {k: v.to("cuda") for k, v in enc.items()}
            with torch.inference_mode():
                out = model.generate(**enc, do_sample=False, max_new_tokens=a.max_new_tokens, eos_token_id=eos,
                                     pad_token_id=eos[-1])
            new = out[0, enc["input_ids"].shape[1]:].tolist()
            while new and new[-1] in eos:                  # the probe never appends the EOS token it stops on
                new = new[:-1]
            reply = tok.decode(new, skip_special_tokens=True).strip()
            preds.append({"id": r["id"], "row_type": r["row_type"], "family": r["family"], "prediction": reply,
                          "generated_tokens": len(new), **score_reply(r, reply)})
        rec = {"arm": name, "eos": eos, "seconds": round(time.time() - t0, 1), "summary": summarise(preds), "predictions": preds}
        result["arms"].append(rec)
        print(json.dumps({"arm": name, **rec["summary"], "seconds": rec["seconds"]}), flush=True)
        del model, tok
        gc.collect(); torch.cuda.empty_cache()
    a.output.write_bytes((json.dumps(result, indent=1, ensure_ascii=False) + "\n").encode("utf-8"))
    print(f"wrote {a.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
