"""Tokenizer-only native direct serialization and deterministic TRAIN fit audit."""
from __future__ import annotations

import os

# Disable model frameworks and all remote lookup before any third-party import.
BACKENDS = {"USE_TORCH": "0", "USE_TF": "0", "USE_FLAX": "0", "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"}
os.environ.update(BACKENDS)

import argparse
import ast
import csv
import hashlib
import html
import importlib.metadata
import io
import json
import math
import platform
import sys
import time
import types
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ASSETS = {"tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"}
FORBIDDEN_FRAMEWORKS = {"torch", "tensorflow", "flax", "jax"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(value):
    return hashlib.sha256(value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()


def read_json(data):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, f"duplicate JSON key: {key}")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=pairs,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def framework_guard():
    require(not (FORBIDDEN_FRAMEWORKS & set(sys.modules)), "model framework imported in tokenizer-only process")
    require(not any(name.startswith("transformers.models.") and ".modeling_" in name for name in sys.modules),
            "model implementation imported in tokenizer-only process")


def pure_module(name, source, path, allowed):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module]
            require(all(module in allowed for module in names), f"undeclared import in {name}: {names}")
    module = types.ModuleType(name)
    module.__file__ = str(path)
    # Dataclass validation resolves annotations through sys.modules.
    sys.modules[name] = module
    exec(compile(tree, str(path), "exec"), module.__dict__)
    framework_guard()
    return module


def train_row(bridge, parent, revision, variant):
    # This guard occurs before renderer, reference or verifier calls.
    require(parent.role == "TRAIN", "only TRAIN parents may be rendered, solved or verified")
    row = bridge.make_row(parent, revision, variant)
    require(row["role"] == "TRAIN" and row["variant_id"] == variant and row["parent_id"] == parent.parent_id,
            "generated identity/role/variant mismatch")
    require(row["verifier_result"] and row["mutation_control_rejected"], "semantic verifier controls failed")
    return row


def prepare_rows(bridge, config):
    require(config["counts"] == {"TRAIN": 128, "TRAIN_SELECTION": 0, "DEV": 0, "CHECK": 0, "FINAL": 0},
            "fixed TRAIN-only quotas changed")
    require(config["variants"] == [0, 1], "both unchanged public variants required")
    parents, selection = bridge.prepare_parents(config["counts"], config["selection_seed"])
    require(len(parents) == 128 and all(parent.role == "TRAIN" for parent in parents), "allocator returned non-TRAIN or wrong quota")
    require(len({parent.parent_id for parent in parents}) == 128, "duplicate canonical parents")
    counts = Counter(read_json(parent.canonical_json)["family"] for parent in parents)
    require(counts == {family: 32 for family in bridge.FAMILIES}, "family balance differs from fixed quota")
    rows = []
    for parent in parents:
        paired = [train_row(bridge, parent, config["source_revision"], variant) for variant in config["variants"]]
        require(paired[0]["target"] == paired[1]["target"] and paired[0]["target_sha256"] == paired[1]["target_sha256"],
                "paired renderers changed the verified JSON target")
        rows.extend(paired)
    parent_records = [{"parent_id": parent.parent_id, "role": parent.role, "canonical_parent_sha256": parent.canonical_parent_sha256,
                       "split_sha256": parent.split_sha256, "structure_id": parent.structure_id,
                       "generator_seed": parent.generator_seed, "candidate_index": parent.candidate_index} for parent in parents]
    return parent_records, rows, selection


def native_measure(messages, target, tokenizer):
    """Independent native transcript/boundary/mask derivation, without a fit skip."""
    kwargs = {"enable_thinking": False}
    prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
    prefix_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, **kwargs)
    require(prefix.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "nonthinking generation prefix differs")
    require(prefix_ids == tokenizer.encode(prefix, add_special_tokens=False), "native generation tokens differ from explicit encoding")
    full_messages = messages + [{"role": "assistant", "content": target}]
    full = tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False, **kwargs)
    require(full == prefix + target + tokenizer.eos_token + "\n", "full native transcript differs from prefix+target+EOS+newline")
    ids = tokenizer.apply_chat_template(full_messages, tokenize=True, add_generation_prompt=False, **kwargs)
    require(ids == tokenizer.encode(full, add_special_tokens=False), "native full tokens differ from explicit encoding")
    require(type(ids) is list and all(type(token) is int and token >= 0 for token in ids), "invalid native token ID list")
    require(ids[:len(prefix_ids)] == prefix_ids, "full native conversation changes generation token prefix")
    target_ids = tokenizer.encode(target, add_special_tokens=False)
    require(bool(target_ids) and tokenizer.eos_token_id not in target_ids, "empty JSON target or embedded EOS")
    start, end = len(prefix_ids), len(prefix_ids) + len(target_ids)
    require(ids[start:end] == target_ids, "separate JSON target encoding changes full-token boundary")
    require([index for index in range(start, len(ids)) if ids[index] == tokenizer.eos_token_id] == [end],
            "response must contain exactly one EOS immediately after target")
    require(tokenizer.decode(ids[end + 1:], skip_special_tokens=False) == "\n", "post-EOS tail must be exactly newline")
    require(tokenizer.decode(ids[start:end + 1], skip_special_tokens=False) == target + tokenizer.eos_token,
            "target/EOS token roundtrip differs")
    labels = [-100] * start + ids[start:end + 1] + [-100] * (len(ids) - end - 1)
    return {"ids": ids, "prompt_ids": prefix_ids, "labels": labels,
            "completion_mask": [int(start <= index <= end) for index in range(len(ids))],
            "answer_start": start, "answer_end": end, "processed_tokens": len(ids), "prompt_tokens": start,
            "supervised_tokens": end - start + 1, "final_content_tokens": len(target_ids),
            "post_eos_tokens": len(ids) - end - 1, "post_eos_tokens_supervised": False,
            "response_eos_count": 1, "generation_full_prefix_parity": True,
            "token_hash": sha(canonical(ids)), "labels_hash": sha(canonical(labels)),
            "supervised_fraction": (end - start + 1) / len(ids), "truncated": False}


def profile_decision(measured, profile, reserve, max_completion):
    reasons = []
    if measured["processed_tokens"] > profile:
        reasons.append("PROCESSED_SEQUENCE_EXCEEDS_PROFILE")
    if measured["prompt_tokens"] + reserve > profile:
        reasons.append("PROMPT_PLUS_COMPLETION_RESERVE_EXCEEDS_PROFILE")
    if measured["supervised_tokens"] > max_completion:
        reasons.append("SUPERVISED_COMPLETION_EXCEEDS_128")
    return {"profile": profile, "completion_reserve": reserve, "retained": not reasons,
            "rejection_reasons": reasons, "prompt_reserve_headroom": profile - measured["prompt_tokens"] - reserve,
            "processed_headroom": profile - measured["processed_tokens"]}


def compare_serializer(serialization, messages, target, tokenizer, measured, profile, reserve):
    expected_fit = measured["processed_tokens"] <= profile and measured["prompt_tokens"] + reserve <= profile
    try:
        compiled = serialization.compile_row(messages, target, tokenizer, mode="direct", max_sequence=profile,
                                             completion_reserve=reserve)
    except ValueError as exc:
        require(not expected_fit and str(exc).startswith("OVERLENGTH:"), f"unexpected serializer rejection: {exc}")
        return "EXPLICIT_OVERLENGTH_REPRODUCED"
    require(expected_fit, "serializer accepted an independently overlength row")
    for key in ("ids", "prompt_ids", "labels", "answer_start", "answer_end", "processed_tokens",
                "supervised_tokens", "final_content_tokens", "post_eos_tokens_supervised", "truncated"):
        require(compiled[key] == measured[key], f"independent serializer mismatch: {key}")
    require(compiled["reasoning_content_tokens"] == 0, "direct contract supervises reasoning content")
    return "EXACT_NATIVE_IDS_PREFIX_EOS_MASK_PARITY"


def distribution(values):
    values = sorted(values)
    require(bool(values), "empty distribution cannot represent completed audit")
    return {"count": len(values), "sum": sum(values), "min": values[0], "max": values[-1], "mean": sum(values) / len(values),
            **{f"p{q}": values[math.ceil(q / 100 * len(values)) - 1] for q in (50, 90, 95)}}


def summarize(rows, decisions, families):
    result = []
    for family in families:
        for variant in (0, 1):
            for profile in (512, 1024, 2048):
                group = [record for record in decisions if record["family"] == family and record["variant"] == variant and record["profile"] == profile]
                require(len(group) == 32, "missing family/variant/profile decisions")
                retained = [record for record in group if record["retained"]]
                result.append({"family": family, "variant": variant, "profile": profile, "rows": len(group),
                               "retained": len(retained), "rejected": len(group) - len(retained),
                               "rejection_reasons": dict(Counter(reason for record in group for reason in record["rejection_reasons"])),
                               **{field: distribution([record[field] for record in group])
                                  for field in ("processed_tokens", "prompt_tokens", "supervised_tokens", "final_content_tokens", "supervised_fraction")},
                               "retained_processed_tokens_per_pass": sum(record["processed_tokens"] for record in retained),
                               "retained_supervised_tokens_per_pass": sum(record["supervised_tokens"] for record in retained)})
    paired = []
    for parent_id in sorted({row["parent_id"] for row in rows}):
        zero, one = sorted([row for row in rows if row["parent_id"] == parent_id], key=lambda row: row["variant"])
        require(zero["target_sha256"] == one["target_sha256"], "token paired target identity changed")
        require(zero["supervised_tokens"] == one["supervised_tokens"] and zero["final_content_tokens"] == one["final_content_tokens"],
                "renderer altered completion token budget")
        paired.append({"parent_id": parent_id, "family": zero["family"], "target_sha256": zero["target_sha256"],
                       "variant0_processed": zero["processed_tokens"], "variant1_processed": one["processed_tokens"],
                       "variant1_minus_variant0_processed": one["processed_tokens"] - zero["processed_tokens"],
                       "variant1_minus_variant0_prompt": one["prompt_tokens"] - zero["prompt_tokens"],
                       "supervised_tokens": zero["supervised_tokens"],
                       "shorter_variant": 0 if zero["processed_tokens"] < one["processed_tokens"] else 1 if one["processed_tokens"] < zero["processed_tokens"] else "tie"})
    efficient = []
    for family in families:
        for profile in (512, 1024, 2048):
            candidates = [record for record in decisions if record["family"] == family and record["profile"] == profile and record["retained"]]
            candidates.sort(key=lambda record: (record["processed_tokens"] / record["supervised_tokens"], record["processed_tokens"], record["row_id"]))
            unique = set()
            chosen = []
            for candidate in candidates:
                if candidate["parent_id"] in unique:
                    continue
                unique.add(candidate["parent_id"])
                chosen.append({key: candidate[key] for key in ("row_id", "parent_id", "variant", "processed_tokens", "supervised_tokens", "supervised_fraction")})
                if len(chosen) == 8:
                    break
            efficient.append({"family": family, "profile": profile, "ranking": "fewer processed tokens per supervised token, then processed tokens, then row_id",
                              "candidate_rows": chosen, "training_eligibility": False})
    return result, paired, efficient


def csv_text(rows):
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def html_report(summary, family_rows, pairs):
    def table(rows, fields):
        head = "".join(f"<th>{html.escape(field.replace('_',' '))}</th>" for field in fields)
        body = "".join("<tr>" + "".join(f"<td>{html.escape(str(row.get(field,'')))}</td>" for field in fields) + "</tr>" for row in rows)
        return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
    families = [{key: (round(value["mean"], 2) if isinstance(value, dict) and "mean" in value else value)
                 for key, value in row.items()} for row in family_rows]
    counts = summary["profile_totals"]
    return "<!doctype html><html lang='en'><meta charset='utf-8'><title>Rineya token fit audit</title>" + \
           "<style>body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#24354a}table{border-collapse:collapse;width:100%;font-size:14px}" + \
           "th,td{padding:8px;border:1px solid #d6dfe8;text-align:left}th{background:#eaf0f6}.scroll{overflow:auto}.boundary{background:#fff3de;padding:14px}" + \
           "details{margin:10px 0;padding:10px;background:#f4f7fa}</style><body><h1>Rineya token fit and training signal</h1>" + \
           "<p class='boundary'>128 TRAIN parents · two unchanged public renderers · three context profiles · 768 fit decisions. " + \
           "Tokenizer suitability only: no model loaded, no training, no learnability or intelligence claim.</p><p>Exact pinned nonthinking template; " + \
           "supervise JSON answer plus one EOS. Prompt, closed thinking stub and native trailing newline are masked.</p>" + \
           "<h2>Context retention</h2>" + table(counts, ["profile", "completion_reserve", "rows", "retained", "rejected"]) + \
           "<h2>Family means and suitability</h2>" + table(families, ["family", "variant", "profile", "rows", "retained", "rejected", "processed_tokens", "prompt_tokens", "supervised_tokens", "supervised_fraction"]) + \
           "<h2>Paired renderer cost</h2><p>Positive delta means variant 1 uses more tokens for the same verified target. " + \
           "A shorter renderer is a cost candidate; model behavior was not measured.</p>" + \
           table(pairs, ["parent_id", "family", "variant0_processed", "variant1_processed", "variant1_minus_variant0_processed", "supervised_tokens", "shorter_variant"]) + \
           "<h2>Efficient fitting candidates within each family</h2>" + "".join(
               f"<details><summary>{html.escape(group['family'])} · profile {group['profile']}</summary>" +
               table(group["candidate_rows"], ["row_id", "variant", "processed_tokens", "supervised_tokens", "supervised_fraction"]) + "</details>"
               for group in summary["efficient_candidates"] if group["candidate_rows"]) + \
           "<h2>Evidence boundary</h2><p>All CHECK/FINAL quotas were zero; neither was rendered or solved. " + \
           "Existing training exclusions and admission gates remain unqualified. Source/core files are read-only. " + \
           "Processed-token counts include the native trailing newline; it receives no supervision. " + \
           "Exact source, parent, runtime and asset hashes are in freeze.json; measured hardware conditions are in the outer supervisor receipt.</p></body></html>"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    output = args.output.resolve()
    require(output.is_relative_to((HERE / "runs").resolve()) and output != (HERE / "runs").resolve(), "output must be a new study runs/ child")
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        framework_guard()
        config = read_json((HERE / "config.json").read_bytes())
        require(config["context_profiles"] == [512, 1024, 2048] and config["completion_reserve"] == 128
                and config["max_supervised_completion"] == 128 and config["mode"] == "direct", "fixed audit profiles or contract changed")
        budget = config["resource_budget"]
        def guard():
            framework_guard()
            require(time.monotonic() - started <= budget["wall_seconds"], "local wall-clock budget exceeded")
            require(sum(path.stat().st_size for path in output.rglob("*") if path.is_file()) <= budget["output_mib"] * 1024 ** 2,
                    "local study output budget exceeded")
        runtime = {name: importlib.metadata.version(name) for name in config["runtime"]}
        require(runtime == config["runtime"], f"wrong tokenizer runtime: {runtime}")
        sources, source_manifest = {}, {}
        for name, record in config["source_files"].items():
            path = (REPO / record["path"]).resolve()
            require(path.is_relative_to(REPO) and path.is_file(), "source missing or outside worktree")
            sources[name] = path.read_bytes()
            source_manifest[name] = {"path": record["path"], "sha256": sha(sources[name]), "bytes": len(sources[name])}
            require(source_manifest[name]["sha256"] == record["sha256"], f"source hash mismatch: {name}")
        require(set(config["tokenizer_assets"]) == ASSETS, "undeclared tokenizer asset set")
        asset_raw, asset_manifest = {}, {}
        asset_directory = Path(config["tokenizer_directory"])
        for name, expected in config["tokenizer_assets"].items():
            path = asset_directory / name
            require(path.is_file() and path.stat().st_size <= 16 * 1024 ** 2, f"tokenizer asset missing/oversized: {name}")
            asset_raw[name] = path.read_bytes()
            require(sha(asset_raw[name]) == expected, f"tokenizer asset hash mismatch: {name}")
            asset_manifest[name] = {"path": str(path), "sha256": expected, "bytes": len(asset_raw[name])}
        tokenizer_config = read_json(asset_raw["tokenizer_config.json"])
        require(tokenizer_config["tokenizer_class"] == config["tokenizer_class"], "tokenizer class mismatch")
        require(sha(tokenizer_config["chat_template"].encode()) == config["chat_template_sha256"], "chat template hash mismatch")
        bridge = pure_module("token_fit_frozen_bridge", sources["bridge"], REPO / config["source_files"]["bridge"]["path"],
                             {"__future__", "argparse", "collections", "dataclasses", "datetime", "hashlib", "json", "pathlib", "random", "re", "sys", "time", "typing"})
        serialization = pure_module("token_fit_frozen_serialization", sources["serialization"], REPO / config["source_files"]["serialization"]["path"], {"__future__", "re"})
        parents, generated, selection = prepare_rows(bridge, config)
        parent_payload = canonical(parents) + b"\n"
        generated_payload = b"".join(canonical(row) + b"\n" for row in generated)
        (output / "parents.json").write_bytes(parent_payload)
        (output / "generated_train.jsonl").write_bytes(generated_payload)
        implementation = {path.relative_to(REPO).as_posix(): sha(path.read_bytes()) for path in sorted(HERE.rglob("*"))
                          if path.is_file() and "runs" not in path.relative_to(HERE).parts and "__pycache__" not in path.parts}
        freeze = {"schema": 1, "phase": "before_token_measurement", "created_utc": datetime.now(timezone.utc).isoformat(),
                  "config": config, "source_files": source_manifest, "tokenizer_assets": asset_manifest,
                  "chat_template_sha256": config["chat_template_sha256"], "implementation": implementation,
                  "runtime": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                              "dependencies": runtime, "disabled_backends_and_offline": BACKENDS},
                  "parents_sha256": sha(parent_payload), "generated_train_sha256": sha(generated_payload),
                  "parent_ids": [record["parent_id"] for record in parents], "selection": selection,
                  "counts": config["counts"], "rendered_rows": len(generated), "training_eligibility": False}
        (output / "freeze.json").write_bytes(json.dumps(freeze, indent=2, ensure_ascii=False).encode() + b"\n")
        guard()
        bundle = output / "tokenizer_assets"
        bundle.mkdir()
        for name, raw in asset_raw.items():
            (bundle / name).write_bytes(raw)
        # Model/config/weights are absent from this four-file-only directory.
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(str(bundle), local_files_only=True, trust_remote_code=False, use_fast=True)
        framework_guard()
        require(tokenizer.is_fast and tokenizer.eos_token == config["eos_token"] and tokenizer.eos_token_id == config["eos_id"],
                "tokenizer type/EOS mismatch")
        require(sha(tokenizer.chat_template.encode()) == config["chat_template_sha256"], "loaded native template differs from frozen asset")
        rows, decisions = [], []
        with (output / "tokens.jsonl").open("xb") as handle:
            for generated_row in generated:
                guard()
                require(generated_row["role"] == "TRAIN", "sealed role entered token measurement")
                messages = [{"role": "system", "content": config["system_message"]}, {"role": "user", "content": generated_row["prompt"]}]
                measured = native_measure(messages, generated_row["target"], tokenizer)
                row = {"row_id": generated_row["row_id"], "parent_id": generated_row["parent_id"], "family": generated_row["family"],
                       "variant": generated_row["variant_id"], "target_sha256": generated_row["target_sha256"],
                       "prompt_sha256": generated_row["prompt_sha256"], "generated_row_sha256": sha(canonical(generated_row)), **measured}
                handle.write(canonical(row) + b"\n")
                rows.append(row)
                for profile in config["context_profiles"]:
                    decision = profile_decision(measured, profile, config["completion_reserve"], config["max_supervised_completion"])
                    parity = compare_serializer(serialization, messages, generated_row["target"], tokenizer, measured, profile, config["completion_reserve"])
                    decisions.append({"row_id": row["row_id"], "parent_id": row["parent_id"], "family": row["family"], "variant": row["variant"],
                                      **decision, **{key: measured[key] for key in ("processed_tokens", "prompt_tokens", "supervised_tokens", "final_content_tokens", "supervised_fraction", "post_eos_tokens")},
                                      "serializer_validation": parity, "truncated": False, "training_eligibility": False})
        require(len(rows) == 256 and len(decisions) == 768, "incomplete audit grid")
        family_rows, pairs, efficient = summarize(rows, decisions, bridge.FAMILIES)
        profile_totals = [{"profile": profile, "completion_reserve": 128, "rows": 256,
                           "retained": sum(record["retained"] for record in decisions if record["profile"] == profile),
                           "rejected": sum(not record["retained"] for record in decisions if record["profile"] == profile)} for profile in config["context_profiles"]]
        summary = {"schema": 1, "status": "MEASURED_TOKENIZER_FIT_ONLY", "experiment": config["experiment"],
                   "model_loaded": False, "frameworks_imported": False, "training_eligibility": False,
                   "parents": len(parents), "rendered_rows": len(rows), "fit_decisions": len(decisions),
                   "counts": config["counts"], "profile_totals": profile_totals, "families": family_rows,
                   "paired_variant_delta": distribution([record["variant1_minus_variant0_processed"] for record in pairs]),
                   "efficient_candidates": efficient, "token_distributions": {field: distribution([row[field] for row in rows])
                   for field in ("processed_tokens", "prompt_tokens", "supervised_tokens", "final_content_tokens", "supervised_fraction")},
                   "eos_checks": {"response_eos": 1, "post_eos_supervision": False, "generation_full_prefix_parity": True},
                   "freeze_sha256": sha((output / "freeze.json").read_bytes()), "elapsed_analysis_seconds": time.monotonic() - started,
                   "limitations": ["Tokenizer fit and verified deterministic targets do not establish learnability or native model qualification.",
                                   "Paired renderers share parents/targets; repeated fit profiles are not independent experiments.",
                                   "Efficient candidates are prospective cost guidance, not training admission or a token-budget change."]}
        flat_decisions = [{**record, "rejection_reasons": " | ".join(record["rejection_reasons"])} for record in decisions]
        flat_families = [{key: (value["mean"] if isinstance(value, dict) and "mean" in value else json.dumps(value) if isinstance(value, dict) else value)
                          for key, value in record.items()} for record in family_rows]
        reports = {"fit.csv": csv_text(flat_decisions), "pairs.csv": csv_text(pairs), "family.csv": csv_text(flat_families),
                   "report.html": html_report(summary, family_rows, pairs)}
        for name, payload in reports.items():
            (output / name).write_text(payload, encoding="utf-8", newline="")
            guard()
        summary["report_hashes"] = {path.relative_to(output).as_posix(): sha(path.read_bytes()) for path in output.rglob("*")
                                    if path.is_file() and path.name != "summary.json"}
        (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        guard()
        print(json.dumps({"status": summary["status"], "output": str(output), "parents": len(parents), "fit_decisions": len(decisions),
                          "profile_totals": profile_totals, "elapsed_seconds": summary["elapsed_analysis_seconds"]}))
        return 0
    except Exception as exc:
        failure = {"status": "INVALID", "error_type": type(exc).__name__, "reason": str(exc), "success_summary_valid": False,
                   "elapsed_seconds": time.monotonic() - started}
        (output / "failure.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        if (output / "summary.json").exists():
            (output / "summary.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(failure), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
