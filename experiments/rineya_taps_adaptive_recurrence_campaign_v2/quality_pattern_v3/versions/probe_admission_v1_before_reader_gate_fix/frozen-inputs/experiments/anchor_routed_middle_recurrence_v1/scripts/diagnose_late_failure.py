"""Reproduce one stopped L27 forward under unchanged bounds; never generate."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT.parents[1] / "Rineya_Cross_Scale_Capability_Transfer_Lab_v1/src")]
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from armr.evaluation import encode, forward_logits
from armr.hooks import Reapplication, SafetyError, frozen_inference
from armr.provenance import DATA, cases_from, checkpoint_files, new_run, state_hash, write_json
from run_experiment import resolve_model


def main():
    config = json.loads((ROOT / "configs/reapplication_screen.json").read_text())
    model_path = resolve_model(None)
    cases = cases_from(DATA / config["smoke_file"])
    out, manifest = new_run("01b_late_failure_diagnosis", config, model_path, DATA / config["smoke_file"])
    start = time.perf_counter()
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    files = checkpoint_files(model_path)
    assert files["model.safetensors"] == config["checkpoint_sha256"]
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, trust_remote_code=True,
                dtype=torch.float32, attn_implementation="eager", low_cpu_mem_usage=True).eval().requires_grad_(False)
    before = state_hash(model)
    ids = encode(tokenizer, cases[0])
    wrapper = Reapplication(model, 27, 2, emergency_bounds=tuple(config["emergency_rms_ratio_bounds"]))
    stopped = False
    with frozen_inference(model):
        base = forward_logits(model, ids)
        try:
            with wrapper:
                forward_logits(model, ids)
        except SafetyError as exc:
            if str(exc) != "hidden norm emergency bound":
                raise
            stopped = True
        restored = forward_logits(model, ids)
    after = state_hash(model)
    after_files = checkpoint_files(model_path)
    checks = {"same_guard_stopped_forward": stopped, "recorded_two_applications": len(wrapper.records) == 2,
              "restored_logits_bit_identical": torch.equal(base, restored), "state_unchanged": before == after,
              "checkpoint_files_unchanged": files == after_files, "no_gradients": all(p.grad is None and not p.requires_grad for p in model.parameters()),
              "no_wrapper_left": "forward" not in model.model.layers[27].__dict__ and not model._forward_pre_hooks}
    receipt = {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
               "case_id": cases[0]["case_id"], "layer": 27, "applications": 2,
               "hidden": wrapper.records, "generation_performed": False,
               "interpretation": "PASS means expected safety failure reproduced and isolated; L27 recurrence remains FAIL"}
    write_json(out / "diagnosis.json", receipt)
    manifest.update(status=receipt["status"], checkpoint_files_before=files, checkpoint_files_after=after_files,
                    state_hash_before=before, state_hash_after=after, canonical_unchanged=before==after and files==after_files,
                    wall_seconds=time.perf_counter()-start, peak_rss_bytes=psutil.Process().memory_info().peak_wset,
                    decision=receipt)
    write_json(out / "run_manifest.json", manifest)
    ratio = wrapper.records[-1]["rms_ratio_to_normal"]
    (out / "report.md").write_text(f"""# Isolated late-control failure diagnosis

Status: {receipt['status']} for failure isolation; L27 repeat-2 remains FAIL.

Question: what triggered the original screen's L27 stop, and did its wrapper
restore the original model? Setup: same first smoke prompt, CPU FP32/eager,
seed 17, unchanged emergency bounds {config['emergency_rms_ratio_bounds']},
one intact forward, one stopped repeat-2 forward, one restored forward.
No generation, threshold change, screen retry, or training.

Results: second-pass RMS / first-pass RMS = {ratio:.6f}. Full RMS, norms,
cosines, deltas and checks are in diagnosis.json. Canonical state/files and
restored logits were checked independently. Exact provenance is in run_manifest.json.

Interpretation: this is a late readout control failure, not evidence that the
ten completed middle candidates failed. L27 was excluded from candidate
selection before the screen. Its repeat-3 and remaining repeat-2 cases remain
unrun. The original failed screen directory remains immutable and FAIL.

Non-claims: successful failure isolation does not make L27 safe or establish
recurrence efficacy. It does not authorize routing, PEFT, or sealed evaluation.

Next gate: audit the complete middle-candidate receipts against the unchanged
per-candidate gates before deciding whether any fixed arm is licensed.
""", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    print(f"RUN_DIR={out}")
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
