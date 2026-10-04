"""Bounded, fail-closed reanalysis of exposed historical arithmetic predictions."""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import io
import itertools
import json
import platform
import sys
import time
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
POLICIES = ("permissive", "strict")
SEEDS = (17, 23, 29, 41)


class InvalidEvidence(ValueError):
    pass


def require(condition, message):
    if not condition:
        raise InvalidEvidence(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def json_no_duplicates(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda token: (_ for _ in ()).throw(
                          InvalidEvidence(f"nonfinite JSON number: {token}")))


def unique_rows(rows, label, expected_count):
    require(isinstance(rows, list), f"{label}: rows must be a list")
    result = {}
    for row in rows:
        require(isinstance(row, dict) and isinstance(row.get("id"), str), f"{label}: malformed row")
        require(row["id"] not in result, f"{label}: duplicate item {row['id']}")
        result[row["id"]] = row
    require(len(result) == expected_count, f"{label}: expected {expected_count} rows, got {len(result)}")
    return result


def validate_bindings(monitor, audit_rows):
    require(set(monitor) == set(audit_rows), "monitor/audit IDs do not match")
    for item_id, row in monitor.items():
        audited = audit_rows[item_id]
        require(sha(canonical(row)) == audited["source_row_sha256"], f"{item_id}: full row binding mismatch")
        require(sha(canonical(row["messages"])) == audited["messages_sha256"], f"{item_id}: thread binding mismatch")
        index = audited["operative_user_message_index_0based"]
        require(type(index) is int and 0 <= index < len(row["messages"]), f"{item_id}: invalid operative index")
        message = row["messages"][index]
        require(message["role"] == "user", f"{item_id}: operative message must be user")
        require(message["content"] == audited["operative_prompt"], f"{item_id}: operative prompt mismatch")
        require(sha(message["content"].encode()) == audited["operative_prompt_sha256"], f"{item_id}: prompt hash mismatch")
        require(row["answer"] == audited["original_answer"], f"{item_id}: reference mismatch")
        require(row["group_id"] == audited["family_id"], f"{item_id}: family mismatch")
        require(row["reply_class"] == audited["reply_class"], f"{item_id}: reply class mismatch")
        require(row["split"] == "task", f"{item_id}: input is not the declared exposed task monitor")


def load_scorer(source):
    """Execute the pinned pure file without package import or inherited bytecode writes."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            modules = [alias.name for alias in node.names] if isinstance(node, ast.Import) else [node.module]
            require(all(name in {"math", "re", "fractions", "__future__"} for name in modules),
                    f"scorer has an undeclared import: {modules}")
    module = types.ModuleType("frozen_stability_retention")
    exec(compile(tree, "pinned_retention.py", "exec"), module.__dict__)
    return module


def score_outputs(outputs, targets, scorer):
    require(isinstance(outputs, dict) and set(outputs) == set(targets), "cached output IDs differ from monitor")
    scored = {}
    for item_id, record in outputs.items():
        require(isinstance(record, dict) and set(record) == {"text", "terminated"}, f"{item_id}: invalid output record")
        require(isinstance(record["text"], str) and type(record["terminated"]) is bool, f"{item_id}: invalid text/EOS types")
        scored[item_id] = scorer.score_prediction(record["text"], record["terminated"], targets[item_id])
    return scored


def transition_counts(source, final, item_ids=None):
    ids = set(source) if item_ids is None else set(item_ids)
    require(ids <= set(source) and ids <= set(final), "transition counts use missing item IDs")
    reference = {item_id for item_id in ids if source[item_id]["correct"]}
    correct = {item_id for item_id in ids if final[item_id]["correct"]}
    losses = reference - correct
    gains = correct - reference
    return {
        "rows": len(ids), "reference_n": len(reference), "correct_total": len(correct),
        "retained_total": len(reference & correct), "lost_total": len(losses), "gained_total": len(gains),
        "net_total": len(gains) - len(losses),
        "terminated_total": sum(final[item_id]["terminated"] for item_id in ids),
        "conditional_loss_fraction": f"{len(losses)}/{len(reference)}" if reference else None,
        "conditional_loss_rate": len(losses) / len(reference) if reference else None,
        "gained_ids": sorted(gains), "lost_ids": sorted(losses),
        "failure_reasons": dict(Counter(final[item_id]["reason"] for item_id in ids if not final[item_id]["correct"])),
    }


def validate_stored(stored, recomputed, label):
    for key in ("reference_n", "correct_total", "retained_total", "lost_total", "gained_total", "terminated_total"):
        require(type(stored.get(key)) is int and stored[key] == recomputed[key],
                f"{label}: stored {key}={stored.get(key)!r}, recomputed {recomputed[key]}")


def overlap(left, right):
    intersection, union = set(left) & set(right), set(left) | set(right)
    return {"left_n": len(set(left)), "right_n": len(set(right)),
            "intersection_n": len(intersection), "union_n": len(union),
            "jaccard": len(intersection) / len(union) if union else None,
            "intersection_ids": sorted(intersection)}


def item_metrics(source, finals):
    correct = [record["correct"] for record in finals]
    gains = [not source["correct"] and outcome for outcome in correct]
    losses = [source["correct"] and not outcome for outcome in correct]
    return {"source_correct": source["correct"], "success_draws": sum(correct),
            "gain_draws": sum(gains), "loss_draws": sum(losses),
            "stable_gain": all(gains), "recurrent_loss": sum(losses) >= 2,
            "universal_loss": all(losses), "all_draws_correct": all(correct),
            "all_draws_incorrect": not any(correct), "mixed_final_outcomes": any(correct) and not all(correct)}


def identities(bank):
    fields = ("canonical_tag", "source_assets_sha256", "source_weights_sha256", "data_sha256",
              "reference_receipt", "reference_digest", "exploration_harness_sha256",
              "optimizer_harness_sha256", "optimizer_storage_harness_sha256", "run_directory",
              "checkpoint", "numerics", "schedule_spec", "optimizer", "train_device", "probe_device",
              "supervised_target_tokens", "optimizer_steps", "train_rows_consumed", "train_row_ids_sha256")
    return {key: bank.get(key) for key in fields}


def analyze(config, blobs, guard):
    monitor = unique_rows([json_no_duplicates(line) for line in blobs["monitor"].decode("utf-8-sig").splitlines()
                           if line.strip()], "monitor", config["expected_rows"])
    audit = json_no_duplicates(blobs["audit"])
    audited = unique_rows(audit["rows"], "audit", config["expected_rows"])
    validate_bindings(monitor, audited)
    scorer = load_scorer(blobs["scorer"].decode("utf-8-sig"))
    targets = {policy: scorer.build_overlay(audit, policy)["targets"] for policy in POLICIES}
    banks = {seed: json_no_duplicates(blobs[f"seed{seed}"]) for seed in SEEDS}
    source_scores, final_scores, draw_counts, family_rows = {}, {}, {}, []
    base_outputs = None
    for seed, bank in banks.items():
        guard()
        require(bank["seed"] == seed, f"seed {seed}: wrong recorded draw")
        for field, expected in config["expected_recipe"].items():
            require(bank.get(field) == expected, f"seed {seed}: recipe field {field} differs")
        require(bank["source_assets_sha256"] == config["expected_source_assets"], f"seed {seed}: source asset identity mismatch")
        require(bank["source_weights_sha256"] == config["expected_source_assets"]["model.safetensors"], f"seed {seed}: source weight identity mismatch")
        require(bank["data_sha256"]["monitor_probe"] == sha(blobs["monitor"]), f"seed {seed}: monitor data identity mismatch")
        require(bank["data_sha256"]["probe_label_audit"] == sha(blobs["audit"]), f"seed {seed}: label audit identity mismatch")
        train_ids = bank["train_row_ids_consumed"]
        require(isinstance(train_ids, list) and all(isinstance(item, str) for item in train_ids), f"seed {seed}: bad consumed training IDs")
        require(len(train_ids) == bank["train_rows_consumed"], f"seed {seed}: consumed training ID count mismatch")
        require(sha("\n".join(train_ids).encode()) == bank["train_row_ids_sha256"], f"seed {seed}: consumed training ID digest mismatch")
        checkpoints = bank["checkpoints"]
        require(len(checkpoints) == 2, f"seed {seed}: unexpected number of historical checkpoints")
        initial, final = checkpoints
        require(initial["milestone"] == 0 and initial["tokens"] == 0 and initial["steps"] == 0,
                f"seed {seed}: initial checkpoint is not before training")
        require(final["milestone"] == 5200 and final["tokens"] >= 5200 and final["steps"] == bank["optimizer_steps"],
                f"seed {seed}: final checkpoint is not the declared completed dose")
        if base_outputs is None:
            base_outputs = initial["outputs"]
        require(initial["outputs"] == base_outputs, f"seed {seed}: raw initial outputs differ across draws")
        source_scores[seed], final_scores[seed], draw_counts[seed] = {}, {}, {}
        for policy in POLICIES:
            source = score_outputs(initial["outputs"], targets[policy], scorer)
            final_scored = score_outputs(final["outputs"], targets[policy], scorer)
            source_scores[seed][policy], final_scores[seed][policy] = source, final_scored
            start_counts, end_counts = transition_counts(source, source), transition_counts(source, final_scored)
            validate_stored(initial["retention"][policy], start_counts, f"seed {seed} initial {policy}")
            validate_stored(final["retention"][policy], end_counts, f"seed {seed} final {policy}")
            require(bank["reference_sizes"][policy] == start_counts["reference_n"], f"seed {seed}: reference size mismatch")
            draw_counts[seed][policy] = {**end_counts, "historical_retention_status": final["retention"][policy]["status"]}
            for checkpoint, scores, label in ((initial, source, "initial"), (final, final_scored, "final")):
                if policy == "permissive":
                    stored_map = checkpoint["retention"]["per_row"]
                    require(stored_map == {item_id: record["correct"] for item_id, record in scores.items()},
                            f"seed {seed} {label}: stored permissive item correctness mismatch")
            for category in ("reply_class", "kind", "template"):
                labels = {item_id: (monitor[item_id]["reply_class"] if category == "reply_class" else
                                   targets[policy][item_id]["kind"] if category == "kind" else
                                   audited[item_id]["independent_recomputation"]["template"]) for item_id in monitor}
                for family in sorted(set(labels.values())):
                    ids = {item_id for item_id, value in labels.items() if value == family}
                    counts = transition_counts(source, final_scored, ids)
                    family_rows.append({"seed": seed, "policy": policy, "category": category, "family": family,
                                        **{key: value for key, value in counts.items() if not key.endswith("_ids") and key != "failure_reasons"}})
                    if category == "reply_class":
                        recorded = final["retention"][policy]["per_class"][family]
                        for key, computed in (("rows", "rows"), ("reference_n", "reference_n"), ("lost", "lost_total"),
                                              ("retained", "retained_total"), ("gained", "gained_total"),
                                              ("terminated", "terminated_total"), ("arm_correct", "correct_total")):
                            require(recorded[key] == counts[computed], f"seed {seed} {policy} {family}: stored family {key} mismatch")
        headline = bank["headline"]
        for key, expected in (("correct_start", draw_counts[seed]["permissive"]["reference_n"]),
                              ("correct_end", draw_counts[seed]["permissive"]["correct_total"]),
                              ("gained_end", draw_counts[seed]["permissive"]["gained_total"]),
                              ("lost_end", draw_counts[seed]["permissive"]["lost_total"]),
                              ("retained_end", draw_counts[seed]["permissive"]["retained_total"]),
                              ("net_end", draw_counts[seed]["permissive"]["net_total"])):
            require(headline[key] == expected, f"seed {seed}: headline {key} mismatch")

    items, stability = [], {}
    for policy in POLICIES:
        source = source_scores[SEEDS[0]][policy]
        policy_items = []
        for item_id in sorted(monitor):
            finals = [final_scores[seed][policy][item_id] for seed in SEEDS]
            metrics = item_metrics(source[item_id], finals)
            row = {"id": item_id, "policy": policy, "reply_class": monitor[item_id]["reply_class"],
                   "group_id": monitor[item_id]["group_id"], "template": audited[item_id]["independent_recomputation"]["template"],
                   "kind": targets[policy][item_id]["kind"], "target": targets[policy][item_id],
                   "operative_prompt": audited[item_id]["operative_prompt"], "original_answer": monitor[item_id]["answer"],
                   "historical_entity_label_issue": audited[item_id]["unit_or_entity_label_mismatch"],
                   **metrics, "source": source[item_id],
                   "draws": {str(seed): final_scores[seed][policy][item_id] for seed in SEEDS}}
            items.append(row)
            policy_items.append(row)
        stability[policy] = {
            "unique_panel_rows": len(monitor), "fixed_source_correct_n": sum(record["correct"] for record in source.values()),
            **{key + "_ids": [row["id"] for row in policy_items if row[key]]
               for key in ("stable_gain", "recurrent_loss", "universal_loss", "all_draws_correct", "all_draws_incorrect", "mixed_final_outcomes")},
            "success_draw_histogram": dict(sorted(Counter(row["success_draws"] for row in policy_items).items())),
            "gain_draw_histogram": dict(sorted(Counter(row["gain_draws"] for row in policy_items if not row["source_correct"]).items())),
            "loss_draw_histogram_conditioned_source_correct": dict(sorted(Counter(row["loss_draws"] for row in policy_items if row["source_correct"]).items())),
        }

    pairs = []
    for left, right in itertools.combinations(SEEDS, 2):
        for policy in POLICIES:
            for metric in ("gained", "lost"):
                pairs.append({"left_seed": left, "right_seed": right, "policy": policy, "metric": metric,
                              **overlap(draw_counts[left][policy][metric + "_ids"], draw_counts[right][policy][metric + "_ids"])})
        pairs.append({"left_seed": left, "right_seed": right, "policy": "recorded_train_ids", "metric": "train_overlap",
                      **overlap(banks[left]["train_row_ids_consumed"], banks[right]["train_row_ids_consumed"])})
    policy_diff = []
    for label, scores in [("source", source_scores[SEEDS[0]])] + [(f"seed{seed}", final_scores[seed]) for seed in SEEDS]:
        changed = [item_id for item_id in sorted(monitor) if scores["permissive"][item_id]["correct"] != scores["strict"][item_id]["correct"]]
        policy_diff.append({"stage": label, "disagreement_n": len(changed), "disagreement_ids": changed,
                            "by_class": dict(Counter(monitor[item_id]["reply_class"] for item_id in changed))})
    summary = {
        "schema": 1, "experiment": config["experiment"], "status": "CACHED_EVIDENCE_REPRODUCED",
        "evidence_kind": "descriptive_historical_monitor_reanalysis", "native_execution": False,
        "unique_evaluation_items": len(monitor), "repeated_draw_count": len(SEEDS),
        "seeds": list(SEEDS), "policies": list(POLICIES), "draws": draw_counts, "stability": stability,
        "pairwise": pairs, "policy_disagreements": policy_diff,
        "audit_human_review_status": audit["human_review_status"], "entity_label_issue_ids": audit["mismatch_ids"],
        "limitations": ["Historical exposed monitor outputs; repeated items are not independent observations.",
                        "No fresh native execution, efficacy, causal schedule effect or model release is established.",
                        "Typed grammar excludes some semantically valid free-form answers; human target review remains pending.",
                        "Source and harness identity claims are preserved from caches; model binaries were not opened.",
                        "Zero recorded training-ID overlap does not establish independent evaluation outcomes."],
    }
    return summary, items, family_rows


def csv_text(rows, fieldnames):
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def render_html(summary, items, family_rows):
    def table(rows, fields):
        header = "".join(f"<th>{html.escape(field.replace('_', ' '))}</th>" for field in fields)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(field, '')))}</td>" for field in fields) + "</tr>" for row in rows)
        return f"<div class='scroll'><table><thead><tr>{header}</tr></thead><tbody>{body}</tbody></table></div>"
    sections = []
    for policy in POLICIES:
        state = summary["stability"][policy]
        counts = [{"seed": seed, **summary["draws"][seed][policy]} for seed in SEEDS]
        sections.append(f"<h2>{policy.title()} policy</h2><p>Fixed source correct: {state['fixed_source_correct_n']}/150. "
                        f"Stable gains in all 4 draws: {len(state['stable_gain_ids'])}. "
                        f"Source successes lost in at least 2 draws: {len(state['recurrent_loss_ids'])}; "
                        f"lost in all 4: {len(state['universal_loss_ids'])}.</p>" +
                        table(counts, ["seed", "correct_total", "retained_total", "gained_total", "lost_total", "conditional_loss_fraction", "terminated_total", "historical_retention_status"]))
        for name, flag in (("Stable gains", "stable_gain"), ("Recurrent source losses", "recurrent_loss")):
            selected = [row for row in items if row["policy"] == policy and row[flag]]
            details = []
            for row in selected:
                draws = [{"seed": seed, "answer": row["draws"][str(seed)]["raw"],
                          "correct": row["draws"][str(seed)]["correct"], "reason": row["draws"][str(seed)]["reason"]} for seed in SEEDS]
                details.append(f"<details><summary>{html.escape(row['id'])} · {row['template']} · "
                               f"{row['success_draws']}/4 final successes</summary><p>{html.escape(row['operative_prompt'])}</p>"
                               f"<p>Reference: {html.escape(row['original_answer'])}; source: {html.escape(row['source']['raw'])}.</p>" +
                               table(draws, ["seed", "answer", "correct", "reason"]) + "</details>")
            sections.append(f"<h3>{name}</h3>" + ("".join(details) or "<p>None.</p>"))
    overview = table(family_rows, ["seed", "policy", "category", "family", "rows", "reference_n", "correct_total", "gained_total", "lost_total"])
    pairs = table(summary["pairwise"], ["left_seed", "right_seed", "policy", "metric", "left_n", "right_n", "intersection_n", "union_n", "jaccard"])
    policies = table(summary["policy_disagreements"], ["stage", "disagreement_n", "by_class"])
    limits = "".join(f"<li>{html.escape(item)}</li>" for item in summary["limitations"])
    return "<!doctype html><html lang='en'><meta charset='utf-8'><title>Rineya cached gain/loss stability</title>" + \
           "<style>body{font:16px system-ui;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#203047}h1,h2{color:#163655}" + \
           "table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:8px;border:1px solid #d6dfe8;text-align:left}" + \
           "th{background:#eaf0f6}details{margin:8px 0;padding:10px;background:#f4f7fa}summary{cursor:pointer;font-size:14px}.scroll{overflow:auto}" + \
           ".boundary{border-left:5px solid #bd741f;background:#fff4e6;padding:14px}</style><body><h1>Rineya cached gain/loss stability</h1>" + \
           "<p class='boundary'>150 exposed historical monitor items, repeated across four training draws. Descriptive reanalysis; " + \
           "no fresh native run or causal improvement claim. The integrity status reproduces cached totals and does not qualify a model.</p>" + \
           "".join(sections) + "<h2>Strict/permissive differences</h2>" + policies + "<h2>Draw overlap</h2>" + pairs + \
           "<h2>Class, typed kind and arithmetic template breakdown</h2>" + overview + "<h2>Evidence limits</h2><ul>" + limits + \
           "</ul><p>Measured hardware conditions are in the enclosing supervisor's receipts. Exact input/runtime bindings are in freeze.json.</p></body></html>"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to((HERE / "runs").resolve()) and output != (HERE / "runs").resolve(),
            "output must be a new child directory of this experiment's runs/")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        config = json_no_duplicates((HERE / "config.json").read_bytes())
        require(config["policies"] == list(POLICIES) and config["seeds"] == list(SEEDS), "configuration changed the fixed study grid")
        budget = config["resource_budget"]
        def guard():
            require(time.monotonic() - started <= budget["wall_seconds"], "local wall-clock limit exceeded")
            require(sum(path.stat().st_size for path in output.iterdir() if path.is_file()) <= budget["output_mib"] * 1024 ** 2,
                    "local output-size limit exceeded")
        blobs, manifest = {}, {}
        for name, record in config["inputs"].items():
            path = (REPO / record["path"]).resolve()
            require(path.is_relative_to(REPO) and path.is_file(), f"input missing or outside worktree: {name}")
            require(path.stat().st_size <= budget["max_input_mib"] * 1024 ** 2, f"input is unexpectedly large: {name}")
            blobs[name] = path.read_bytes()
            manifest[name] = {"path": record["path"], "sha256": sha(blobs[name]), "bytes": len(blobs[name]),
                              "expected_sha256": record["sha256"]}
        implementation = {path.relative_to(REPO).as_posix(): sha(path.read_bytes())
                          for path in sorted(HERE.rglob("*")) if path.is_file() and "runs" not in path.relative_to(HERE).parts
                          and "__pycache__" not in path.parts}
        freeze = {"schema": 1, "phase": "before_scoring", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "inputs": manifest, "implementation": implementation, "config": config,
                  "runtime": {"executable": sys.executable, "python": sys.version, "platform": platform.platform(),
                              "stdlib_only": True, "argv": sys.argv, "working_directory": str(Path.cwd())},
                  "historical_identities": {str(seed): identities(json_no_duplicates(blobs[f"seed{seed}"])) for seed in SEEDS},
                  "resource_measurement": "external hardware supervisor required; local guards cover elapsed time and output size"}
        (output / "freeze.json").write_text(json.dumps(freeze, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        for name, record in manifest.items():
            require(record["sha256"] == record["expected_sha256"], f"input byte hash mismatch: {name}")
        summary, items, families = analyze(config, blobs, guard)
        guard()
        summary["elapsed_analysis_seconds"] = time.monotonic() - started
        summary["freeze_sha256"] = sha((output / "freeze.json").read_bytes())
        flat = []
        for row in items:
            record = {key: value for key, value in row.items() if key not in {"draws", "source", "target"}}
            record.update({"source_answer": row["source"]["raw"], "source_reason": row["source"]["reason"]})
            for seed in SEEDS:
                final = row["draws"][str(seed)]
                record.update({f"seed{seed}_{key}": final[key] for key in ("correct", "raw", "reason", "terminated")})
            flat.append(record)
        payloads = {"items.json": json.dumps(items, ensure_ascii=False, indent=2) + "\n",
                    "items.csv": csv_text(flat, list(flat[0])),
                    "family.csv": csv_text(families, list(families[0])),
                    "pairwise.csv": csv_text([{**row, "intersection_ids": " | ".join(row["intersection_ids"])} for row in summary["pairwise"]], list(summary["pairwise"][0])),
                    "report.html": render_html(summary, items, families)}
        require(sum(len(value.encode()) for value in payloads.values()) < budget["output_mib"] * 1024 ** 2,
                "prospective report output exceeds budget")
        for name, payload in payloads.items():
            (output / name).write_text(payload, encoding="utf-8", newline="")
            guard()
        summary["report_hashes"] = {name: sha((output / name).read_bytes()) for name in payloads}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        guard()
        require(not ({"torch", "numpy", "transformers", "scipy"} & set(sys.modules)), "undeclared numerical/model dependency imported")
        print(json.dumps({"status": summary["status"], "output": str(output),
                          "rows": summary["unique_evaluation_items"], "elapsed_seconds": summary["elapsed_analysis_seconds"]}))
        return 0
    except Exception as exc:
        failure = {"status": "INVALID", "error_type": type(exc).__name__, "reason": str(exc),
                   "elapsed_seconds": time.monotonic() - started, "success_summary_valid": False}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        if (output / "summary.json").exists():
            (output / "summary.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
