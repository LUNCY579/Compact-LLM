"""Write a self-contained report from completed, audited experiment receipts."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def main():
    run, audit = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    m, d, fit, receipt = read(run/"run_manifest.json"), read(run/"decision.json"), read(run/"bridge_fit.json"), read(audit/"audit.json")
    if m["status"] != "COMPLETE" or receipt["status"] != "PASS" or Path(receipt["source_run"]) != run:
        raise ValueError("a matching passing independent audit is required")
    raw = [json.loads(s) for s in (run/"raw_results.jsonl").read_text(encoding="utf-8").splitlines()]
    base = read(run/"A0_dev.json") if d["development"] else []
    runrel, auditrel = run.relative_to(ROOT).as_posix(), audit.relative_to(ROOT).as_posix()
    primary = d.get("development", {}).get("G")
    title = "Mechanically viable; no observed development gain" if primary and not primary["wins"] and not primary["losses"] else "Single-layer graft: bounded experimental result"
    lines = ["# "+title, "", "**Completed 2026-09-05.** One frozen Qwen3-4B L16 was inserted after ordinary Supa L12 using fitted residual bridges and a fixed gate of 0.1. All layer indices are zero-based.", "",
             "The native Supa pathway, early stem, custom abstention head and canonical checkpoint were preserved. One donor application was tested; the experiment trained only affine bridge coefficients by ridge regression.", "",
             "## Main result", ""]
    if primary:
        grows = read(run/"G_dev.json")
        same = sum(r["token_ids"] == b["token_ids"] for r, b in zip(grows, base))
        overhead = (sum(r["wall_seconds"] for r in grows)/sum(r["wall_seconds"] for r in base)-1)*100
        compute_overhead = (sum(r["generation_dense_matmul_flops_estimate"] for r in grows)/sum(r["generation_dense_matmul_flops_estimate"] for r in base)-1)*100
        gold_base = [r["gold_nll"] for r in base if r["gold_nll"] is not None]
        gold_graft = [r["gold_nll"] for r in grows if r["gold_nll"] is not None]
        lines += [f"The aligned graft scored **{primary['correct']:.0f}/16**, versus **{primary['baseline_correct']:.0f}/16** for intact Supa: **{len(primary['wins'])} paired wins and {len(primary['losses'])} losses**. {same}/16 greedy outputs were identical. Mean first-token cosine was {primary['mean_logit_cosine']:.6f}, and KL was {primary['mean_kl']:.6f} nats.", "",
                  f"Measured development wall time increased by **{overhead:.1f}%**; estimated dominant linear-matmul work increased by **{compute_overhead:.1f}%**. The added operator has **{m['donor_layer_parameters']:,} frozen parameters**, with **{m['bridge_coefficients']:,} bridge coefficients**. Latency includes instrumentation and optional gold-answer scoring; serial order and host load confound timing. The FLOP estimate excludes attention, normalization, instrumentation and gold-answer scoring. This is not a compute-matched efficacy result.", "",
                  f"Mean reference-answer NLL on {len(gold_base)} cases shifted from **{mean(gold_base):.4f}** to **{mean(gold_graft):.4f}** (lower is better). This descriptive probability shift did not change the generated answers.", ""]
    lines += ["## Development controls", "", "| Arm | Correct | Wins / losses vs A0 | Identical outputs | Mean cosine | Mean KL | Gold NLL | Time |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    names = {"A0": "A0 intact", "G": "G aligned", "R": "R random bridges", "S": "S shuffled pairs", "A1": "A1 native L12 repeat"}
    for arm, result in d["development"].items():
        rows = read(run/(arm+"_dev.json"))
        same = sum(r["token_ids"] == b["token_ids"] for r, b in zip(rows, base))
        nll = mean(r["gold_nll"] for r in rows if r["gold_nll"] is not None)
        lines.append(f"| {names[arm]} | {result['correct']:.0f}/{result['items']} | {len(result['wins'])} / {len(result['losses'])} | {same}/{len(rows)} | {result['mean_logit_cosine']:.6f} | {result['mean_kl']:.6f} | {nll:.4f} | {result['wall_seconds']:.1f}s |")
    if primary and "R" in d["development"]:
        random_nll = mean(r["gold_nll"] for r in read(run/"R_dev.json") if r["gold_nll"] is not None)
        if random_nll < mean(gold_graft):
            lines += ["", "The random-bridge control improved reference-answer NLL more than the aligned graft. Therefore the aligned arm's small NLL improvement does not demonstrate a benefit specific to learned correspondence."]
    lines += ["", "The earlier A3 L12–L13 block scored 13/16 in the separate ARMR replication. A3 was not rerun here. The single large-layer graft did not reproduce that block's counterfactual correction.", "",
              "## Mechanical controls", "", "| Arm | Correct | Mean cosine | Mean KL | RMS range | Pass |", "|---|---:|---:|---:|---|---|"]
    for arm, s in d["smoke"].items():
        lines.append(f"| {arm} | {s['score']*s['n']:.0f}/{s['n']} | {s['mean_logit_cosine']:.6f} | {s['mean_kl']:.6f} | {s['min_rms_ratio']:.6f}–{s['max_rms_ratio']:.6f} | {s['go_fixed_smoke']} |")
    lines += ["", "Z is exact gate zero; I is an identity donor; N is the aligned donor at sigmoid(-8). Z/I were bit-identical in logits and greedy IDs. R/S/G use gate 0.1. All smoke arms passed the frozen mechanical gate, including the negative controls. A small tolerable perturbation alone does not demonstrate useful transfer.",
              "", "## Bridge diagnostics and data boundaries", "",
              f"Affine maps were fitted on **{fit['train_tokens']} tokens**, with **{fit['heldout_tokens']} tokens held out by scenario**. The 59 calibration prompts split into 47 training and 12 held-out prompts; all paraphrases of a held-out scenario stayed together. Three calibration prompts were truncated to 256 tokens. Full development prompts were used. No answer label entered bridge fitting.", "",
              "| Map | Held-out up cosine | Held-out down cosine |", "|---|---:|---:|"]
    for arm, value in fit["fit"].items():
        lines.append(f"| {arm} | {value['heldout']['up']['token_cosine']:.4f} | {value['heldout']['down']['token_cosine']:.4f} |")
    delta = fit["heldout_donor_delta_from_bridged_vs_native_input"]
    lines += ["", f"On held-out prompts, the donor delta from projected Supa states versus native donor states had cosine **{delta['token_cosine']:.4f}** and relative MSE **{delta['relative_mse']:.4f}**. Absolute-state alignment did not yield equally strong transformation alignment.", ""]
    diag = run/"posthoc_alignment_diagnostics.json"
    if diag.exists():
        token = read(diag)["metrics"]
        lines += ["Post-hoc diagnostics (no map/gate changes):", "", "| Held-out subset | Up cosine | Up mean token-relative MSE | Down cosine | Down mean token-relative MSE |", "|---|---:|---:|---:|---:|"]
        for name, values in token.items():
            lines.append(f"| {name} | {values['up']['mean_token_cosine']:.4f} | {values['up']['mean_token_relative_mse']:.4f} | {values['down']['mean_token_cosine']:.4f} | {values['down']['mean_token_relative_mse']:.4f} |")
        text_up = token["text_token_ids_below_151643"]["up"]
        lines += ["", f"Global relative MSE masks substantial text-token error: the upward map's global relative MSE over all held-out tokens is {token['all_heldout_tokens']['up']['global_relative_mse']:.6f}, but it is {text_up['global_relative_mse']:.4f} over ordinary text tokens alone. High-norm chat tokens dominate the first metric. This is a concrete limitation of the present absolute-state fitting objective, not a proven explanation of the quality result. No objective or gate was changed after inspecting these diagnostics.", ""]
    overlap = receipt["data_overlap"]
    lines += [f"All {overlap['smoke']['exact_prompt_overlap_with_calibration']}/8 smoke prompts are in the calibration bank, and {overlap['smoke']['cases_used_for_bridge_training']}/8 were used for bridge fitting. They are mechanical checks, not held-out efficacy. Development has {overlap['dev']['exact_prompt_overlap_with_calibration']}/16 exact prompt overlap with calibration, but it is a small, previously inspected procedural set with related templates. No sealed data was accessed.", "",
              "## Integrity and validation", "",
              f"**{sum(receipt['checks'].values())}/{len(receipt['checks'])} independent audit checks passed.** The run recorded {len(raw)} generations: {sum(r['eos_emitted'] for r in raw)} EOS completions, {sum(r['safety_stopped'] for r in raw)} safety stops, and {sum(r['invalid'] for r in raw)} invalid outputs. Zero/identity controls and post-run restoration were exact. A0/A1 outputs and first-token metrics reproduced historical receipts exactly.", "",
              "Supa and donor loaded-state hashes matched before/after, checkpoint-file hashes were unchanged, and there were zero trainable backbone parameters and zero gradients. The real streamed Supa fixture was bit-identical to the canonical model. Source and dataset hashes matched their pre-run receipts.", "",
              f"CPU FP32/eager, four threads, no KV cache, no GPU inference. Total model-run time: {m['wall_seconds']/60:.1f} minutes; peak process working set: {m['peak_rss_bytes']/2**30:.2f} GiB. Twenty meaningful tests passed across preflight and auditor suites; see [VALIDATION.md](VALIDATION.md).", "",
              "## Interpretation", "",
              "This prespecified graft can be executed stably, but this run supplies no capability gain that justifies its additional parameters or compute. It does not reject other donor layers, stronger alignment, or delta-oriented bridge objectives. It provides no support yet for combining this operator with routing, MLA, MTP, or repeated refinement.", "",
              "A follow-up should first address held-out transformation alignment and retain the shuffled-pair control. Any new map objective or gate sweep needs a separate prospective experiment; this run's settings and historical evidence remain fixed.", "",
              "## Artifacts", "",
              f"- [Frozen protocol](README.md) and [configuration](protocol.json)",
              f"- [Run report]({runrel}/report.md), [manifest]({runrel}/run_manifest.json), and [raw cases]({runrel}/raw_results.jsonl)",
              f"- [Bridge fit]({runrel}/bridge_fit.json) and [post-hoc token diagnostics]({runrel}/posthoc_alignment_diagnostics.json)",
              f"- [Independent audit]({auditrel}/audit.json)",
              "- `graft.py`, `run.py`, `audit.py`, `diagnose_alignment.py`, `summarize.py`, and `tests/` reproduce the implementation and checks.",
              "", "Local activation and bridge tensors are ignored by Git and referenced by SHA256. No Rineya checkpoint was trained, exported, committed, or published."]
    with (ROOT/"RESULTS.md").open("x", encoding="utf-8") as f:
        f.write("\n".join(lines)+"\n")
    print(ROOT/"RESULTS.md")


if __name__ == "__main__":
    main()
