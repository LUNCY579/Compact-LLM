"""SMALLVD-001 Phases 3-5 -- token-utility profile, protected set, candidate tokenizers V1-V4, and
their HELD-OUT validation against the registered REJECT / FLAG criteria (SMALLVD001_PREREG.md §3).

Outputs (create-only) in --out:
  TOKEN_UTILITY_PROFILE.jsonl.gz   one row per regular vocabulary entry (151,643)
  PROTECTED_SET.json               every protected token, by category, with the reason
  V1_98304/ V2_65536/ V3_49152/ V4_32768/   complete tokenizer + config sidecars + token_id_map.json
  VOCAB_CANDIDATES.json            per-candidate measurements, verdicts and file hashes

NOTE ON BACKUP: the records repo's .gitignore excludes tokenizer.json / vocab.json / merges.txt by
bare filename (they used to be byte-identical Supa copies). These candidates are NOT Supa copies.
Their sha256 is recorded here, and they rebuild deterministically from this script, the committed
corpus and the teacher files.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "rab001_exploration"))
import hostbudget  # noqa: E402
import param_accounting as pa  # noqa: E402
import protected_tokens as pt  # noqa: E402
import token_utility as tu  # noqa: E402
import tokenizer_validation as tv  # noqa: E402
import vocab_prune as vp  # noqa: E402

TARGETS = {"V1": 98304, "V2": 65536, "V3": 49152, "V4": 32768}
FLAG_OVERALL, FLAG_CATEGORY, FLAG_P99 = 1.10, 1.25, 1.50


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            h.update(b)
    return h.hexdigest()


def chat_template_tokens(teacher: Path, tok) -> set[str]:
    """Every regular token the real template emits around content: roles, newlines, the injected
    empty think block, and the tool-calling scaffold."""
    from transformers import AutoTokenizer
    t = AutoTokenizer.from_pretrained(str(teacher), local_files_only=True)
    tools = [{"type": "function", "function": {"name": "get_weather", "description": "Get weather",
              "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}]
    renders = []
    for msgs in ([{"role": "system", "content": "S"}, {"role": "user", "content": "U"},
                  {"role": "assistant", "content": "A"}, {"role": "user", "content": "V"}],
                 [{"role": "user", "content": "U"}]):
        for tl in (None, tools):
            for gen in (True, False):
                renders.append(t.apply_chat_template(msgs, tools=tl, tokenize=False,
                                                     add_generation_prompt=gen, enable_thinking=False))
    renders.append(t.apply_chat_template(
        [{"role": "user", "content": "U"},
         {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "f", "arguments": {"a": 1}}}]},
         {"role": "tool", "content": "R"}], tokenize=False, add_generation_prompt=True, enable_thinking=False))
    inv = {i: s for s, i in tok.get_vocab(with_added_tokens=False).items()}
    out = set()
    for r in renders:
        for i in tok.encode(r, add_special_tokens=False).ids:
            if i in inv:
                out.add(inv[i])
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--corpus", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    hostbudget.require(1.2, "SMALLVD-001 vocab candidates")
    a.out.mkdir(parents=True, exist_ok=False)
    import tokenizers
    files = vp.load_teacher_tokenizer_files(a.teacher)
    tj = files["tokenizer.json"]
    vocab, merges = tj["model"]["vocab"], vp.merges_of(tj)
    inv = {i: t for t, i in vocab.items()}
    old = tokenizers.Tokenizer.from_file(str(a.teacher / "tokenizer.json"))

    fit = tu.load_corpus(a.corpus, "fit")
    held = tu.load_corpus(a.corpus, "heldout")
    counts = tu.count_tokens(old, fit)
    util_id = tu.mixture_utility(counts)
    util = {inv[i]: u for i, u in util_id.items() if i in inv}
    by_cat_tok = {c: {inv[i]: n for i, n in cnt.items() if i in inv} for c, cnt in counts["counts"].items()}
    hostbudget.guard(0.6, "vocab candidates")

    chat_toks = chat_template_tokens(a.teacher, old)
    prot = pt.build_protected(vocab, chat_toks, by_cat_tok)
    protected = set().union(*prot["sets"].values())
    parents, rank, bases = vp.defining_parents(vocab, merges)
    closure = set()
    stack = list(protected)
    while stack:
        t = stack.pop()
        if t in closure:
            continue
        closure.add(t)
        if t in parents:
            stack += list(parents[t])
    with open(a.out / "PROTECTED_SET.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump({"schema": 1, "by_category": {k: sorted(v) for k, v in prot["sets"].items()},
                   "counts": {k: len(v) for k, v in prot["sets"].items()},
                   "union": len(protected), "closure_including_ancestors": len(closure | set(bases)),
                   "absent_from_vocab": prot["absent_from_vocab"],
                   "reasons": pt.__doc__}, f, indent=1, ensure_ascii=False)

    results, retained_sets, removal_rank = {}, {}, {}
    for name, rows in TARGETS.items():
        n_added = len(tj["added_tokens"])
        retained, log = vp.greedy_prune(vocab, merges, util, protected, vp.rows_for(rows, n_added))
        retained_sets[name] = retained
        for k, (t, _) in enumerate(log):
            removal_rank.setdefault(t, {})[name] = k
        new_tj, id_map = vp.prune_tokenizer_json(tj, retained)
        d = a.out / f"{name}_{rows}"
        hashes = vp.write_tokenizer_dir(d, files, new_tj, id_map, rows)
        new = tokenizers.Tokenizer.from_file(str(d / "tokenizer.json"))
        cmp_ = tv.compare_on_rows(old, new, held)
        frag = tv.word_fragmentation(old, new, held)
        winv = tv.word_invariance(old, new, id_map, held)          # Amendment 1, REJECT 5
        probes = tv.probe_fragmentation(old, new)
        special = tv.special_behaviour(old, new, id_map)
        contract_ok = all(vocab[t] in id_map for t in prot["sets"]["contract"])
        # chat-template check: every template token retained and the render maps elementwise
        tmpl_ok = chat_toks <= retained
        reject = []
        if cmp_["all"]["roundtrip_exact_rate"] < 1.0:
            reject.append(f"round-trip failures: {len(cmp_['roundtrip_failures'])}")
        if special["violations"]:
            reject.append(f"special-token violations: {special['violations']}")
        if not tmpl_ok:
            reject.append("chat-template token pruned")
        if not contract_ok:
            reject.append("output-contract token pruned")
        if winv["violations"]:
            reject.append(f"invariance violations on HELD-OUT words: {winv['violations']}")
        flags = []
        if cmp_["all"]["token_ratio"] > FLAG_OVERALL:
            flags.append(f"overall token ratio {cmp_['all']['token_ratio']:.4f} > {FLAG_OVERALL}")
        for c, v in cmp_["by_category"].items():
            if v["token_ratio"] > FLAG_CATEGORY:
                flags.append(f"{c} token ratio {v['token_ratio']:.4f} > {FLAG_CATEGORY}")
        if cmp_["all"]["ratio_p99"] > FLAG_P99:
            flags.append(f"P99 per-row ratio {cmp_['all']['ratio_p99']:.4f} > {FLAG_P99}")
        cfg = files["config.json"]
        acct = {L: pa.analytic(cfg, rows, L)["unique_physical_parameters"] for L in pa.DEPTH_GRID}
        results[name] = {
            "rows": rows, "regular_tokens": len(retained), "added_tokens": n_added,
            "merges_kept": len(new_tj["model"]["merges"]),
            "embedding_parameters": rows * cfg["hidden_size"],
            "unique_parameters_by_depth": acct,
            "heldout": {k: v for k, v in cmp_.items() if k != "worst_p99"},
            "worst_p99_examples": [{**w, "text_head": next(r["text"][:300] for r in held if r["id"] == w["id"])}
                                   for w in cmp_["worst_p99"][:20]],
            "word_fragmentation": frag, "word_invariance": winv,
            "probe_fragmentation": probes, "special_behaviour": special,
            "contract_retained": contract_ok, "chat_template_retained": tmpl_ok,
            "REJECT": reject, "FLAG": flags, "verdict": "REJECT" if reject else ("FLAG" if flags else "PASS"),
            "file_sha256": hashes,
        }
        print(f"{name} {rows}: ratio {cmp_['all']['token_ratio']:.4f} p99 {cmp_['all']['ratio_p99']:.3f} "
              f"roundtrip {cmp_['all']['roundtrip_exact_rate']:.4f} invariance {winv['violations']} violations over "
              f"{winv['qualifying_distinct']} words ({winv['qualifying_fraction_distinct']:.3f} of distinct) "
              f"verdict {results[name]['verdict']} {flags}", flush=True)
        hostbudget.guard(0.6, "vocab candidates")

    # per-token profile
    memo = {}
    docf = {}
    for c, dfc in counts["doc_freq"].items():
        for i, n in dfc.items():
            docf[i] = docf.get(i, 0) + n
    tot = {i: sum(counts["counts"][c].get(i, 0) for c in counts["counts"]) for i in inv}
    cand_toks = {name: tokenizers.Tokenizer.from_file(str(a.out / f"{name}_{rows}" / "tokenizer.json"))
                 for name, rows in TARGETS.items()}
    prot_cat = {}
    for k, v in prot["sets"].items():
        for t in v:
            prot_cat.setdefault(t, []).append(k)
    with gzip.open(a.out / "TOKEN_UTILITY_PROFILE.jsonl.gz", "wt", encoding="utf-8", newline="\n") as f:
        for t, i in sorted(vocab.items(), key=lambda kv: kv[1]):
            text = pt.token_bytes(t).decode("utf-8", "replace")
            row = {"id": i, "token": t, "text": text, "byte_class": pt.byte_class(t),
                   "count_fit": tot.get(i, 0), "doc_freq_fit": docf.get(i, 0),
                   "by_category": {c: counts["counts"][c][i] for c in counts["counts"] if counts["counts"][c].get(i)},
                   "mixture_utility_per_M": round(util.get(t, 0.0), 4),
                   "protected": prot_cat.get(t, []), "base_byte": t not in parents,
                   "merge_depth": tu.merge_depth(t, parents, memo), "merge_rank": rank.get(t),
                   "retained": {n: t in s for n, s in retained_sets.items()},
                   "removal_order": removal_rank.get(t, {})}
            if not all(row["retained"].values()):
                row["reencoded_len"] = {n: len(cand_toks[n].encode(text, add_special_tokens=False).ids)
                                        for n, s in retained_sets.items() if t not in s}
                row["modelled_marginal_cost_per_M"] = {n: round(util.get(t, 0.0) * (L - 1), 4)
                                                       for n, L in row["reencoded_len"].items()}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    man = {"schema": 1, "record": "SMALLVD_VOCAB_CANDIDATES", "experiment": "SMALLVD-001",
           "builder_sha256": sha256_file(Path(__file__)),
           "code_sha256": {p.name: sha256_file(p) for p in sorted(HERE.glob("*.py"))},
           "teacher_tokenizer_sha256": sha256_file(a.teacher / "tokenizer.json"),
           "corpus": str(a.corpus), "corpus_sha256": sha256_file(a.corpus),
           "fit_rows": len(fit), "heldout_rows": len(held),
           "fit_tokens_by_category": dict(counts["tokens"]), "category_weights": tu.CATEGORY_WEIGHTS,
           "protected_union": len(protected), "flag_thresholds": [FLAG_OVERALL, FLAG_CATEGORY, FLAG_P99],
           "candidates": results,
           "profile_sha256": sha256_file(a.out / "TOKEN_UTILITY_PROFILE.jsonl.gz"),
           "quality_claim": False, "release_permitted": False}
    with open(a.out / "VOCAB_CANDIDATES.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(man, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
