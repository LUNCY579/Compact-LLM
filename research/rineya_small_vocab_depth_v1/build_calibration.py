"""SMALLVD-001 Phase 2 + layer panels -- build the tokenizer calibration corpus and the layer-ablation
panels from sources DISJOINT from every evaluation instrument, and PROVE the disjointness with a
positive control.

Exclusions (from the eval-instrument and corpus scouts, 2026-09-24; see SMALLVD001_PREREG.md §4a):
  * v3 classes that carry instrument rows or their template-family siblings, excluded WHOLESALE:
    percentage_reasoning, ratio_proportion, time_duration (the 150-row panel = retention probe),
    boundary_setting, clarification_request, insufficient_information (HSKILL1/2),
    counting_sets, unit_conversion (HSKILL2 content-key exposure), arithmetic_chain (the
    retention design's excluded fourth class). HSKILL-CONFIRM-002 is sealed and UNREAD; it is
    excluded by its source classes and is never opened here.
  * the v1/v2 corpora, the v3 preservation slice, every DPO file, Forge RL v2, the records repo,
    the long-context Demo and Supa-A1 evidence copies, the Invisible-Limit lab, cc0 sealed_eval.
  * any code/doc file containing an RCI/Demo generator marker string.

Disjointness check: a candidate row is FLAGGED if its normalised last user turn equals an
instrument row's, or if it shares a normalised >= 80-char sentence with any instrument's user
content (system prompts are excluded from the sentence check -- the v3 system prompts are shared
by design and are instructions, not items). A POSITIVE CONTROL injects real instrument rows and
the build REFUSES unless every one is flagged. A 0 from a check that has never fired is not
evidence of disjointness.

Deterministic: every sampling decision is a sha256 ordering of ids; the same inputs give the same
bytes. Runs under .venv-rab001-cu126 (transformers 4.51.3, the probes' pinned renderer), CPU only.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "experiments" / "rab001_exploration"))
import hostbudget  # noqa: E402
import synthetic_text  # noqa: E402

# A small CPU job (~0.5 GB peak), not a model job: its own floor, below the 3.6 GiB model gate,
# and it stands down mid-run rather than push a paging host further.
RAM_REQUIRE, RAM_GUARD = 1.2, 0.6
LC_MAX_CHARS = 11_000
BUILD_HISTORY = ["v1 (same day): long_context panel had 0 rows because the context filter was "
                 "6,000 chars and every cc0 validation packet is >= ~10K; the sentence-overlap path had "
                 "no positive control of its own. v1 was never committed or read; it was moved out of "
                 "the records tree to the session scratchpad, not deleted."]

BP = Path(r"C:/Users/clclc/BusinessProject")
V3 = BP / "Chat-Bubble/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0/data/sft/by_class"
CC0 = BP / "Chat-Bubble/rineya_long_context_cc0_v1_main/data/sft"
RTG1 = BP / "ChatBubble-rtg1/research/rineya_reasoning_corpus_v1/sft"
STORY = BP / "StoryForge_Foundation_Corpus"
SITE = BP / "Chat-Bubble/.venv/Lib/site-packages"
STDLIB = Path(r"C:/Users/clclc/AppData/Local/Programs/Python/Python313/Lib")
NPM = Path(r"C:/Users/clclc/AppData/Roaming/npm/node_modules")
REC = BP / "rineya-dataset-retention-records"
DOSE = BP / "ChatBubble-dose-param/experiments/architecture_benchmark_001/data"

EXCLUDED_V3 = {"percentage_reasoning", "ratio_proportion", "time_duration", "boundary_setting",
               "clarification_request", "insufficient_information", "counting_sets",
               "unit_conversion", "arithmetic_chain"}
PANEL_CLASSES = {
    "concise_reasoning": ["causal_reasoning", "conditional_rules", "logical_implication", "sequence_pattern",
                          "spatial_relations", "schedule_ordering", "estimation_sanity", "comparison_ranking",
                          "constraint_selection", "physical_commonsense", "contradiction_detection"],
    "grounded_extraction": ["evidence_grounded_dialogue", "evidence_inference", "multi_turn_context_update",
                            "acknowledgment_update"],
    "instruction_writing": ["communication_rewrite", "conversation_closure", "natural_follow_up",
                            "practical_next_step", "plan_revision", "informal_texting"],
    "conversation": ["empathy_validation", "deescalation", "tactful_disagreement", "apology_repair",
                     "negotiation_compromise", "preference_opinion", "comparative_recommendation",
                     "decision_explanation", "roleplay_reasoning", "troubleshooting_diagnosis"],
}
MARKERS = ("consolidated account of the", "asterhall", "closed-context reading test", "pellingham",
           "bellweather", "talia frost", "thalia lark", "record of the hidden key")
INSTRUMENT_FILES = {
    "monitor_probe_150": DOSE / "monitor_probe.jsonl",
    "hskill_views_002": REC / "hskill-views-002/HSKILL_VIEWS_V2.jsonl",
    "select_views_001": REC / "select-views-001/SELECT_VIEWS.jsonl",
    "process_probe_001": REC / "process-transition-probe-001/CANDIDATE_VIEWS.jsonl",
    "longctx_demo_256": BP / "Chat-Bubble/Rineya_Long_Context_Demo_v1/data/all_prompts.jsonl",
    "rci001_calibration": REC / "rci001-20260918/RCI_CALIBRATION.jsonl",
    "rci002_calibration": REC / "rci002-20260918/RCI_CALIBRATION.jsonl",
    "alias_001": REC / "alias001-20260918/ALIAS_PROBE.jsonl",
}


def h(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(p: Path) -> str:
    x = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 23), b""):
            x.update(b)
    return x.hexdigest()


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s).casefold()).strip()


def sentences(s: str, min_len: int = 80) -> set[str]:
    out = set()
    for part in re.split(r"(?<=[.!?])\s+|\n+", s):
        n = norm(part)
        if len(n) >= min_len:
            out.add(h(n))
    return out


def ordered(ids, salt: str):
    return sorted(ids, key=lambda i: h(f"{salt}:{i}"))


# ---------------------------------------------------------------- instrument fingerprints
def strings_of(obj):
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k not in ("system", "system_prompt"):
                yield from strings_of(v)
    elif isinstance(obj, list):
        for v in obj:
            if isinstance(v, dict) and v.get("role") == "system":
                continue
            yield from strings_of(v)


def last_user(obj) -> str | None:
    msgs = obj.get("messages") if isinstance(obj, dict) else None
    if isinstance(msgs, list):
        users = [m.get("content", "") for m in msgs if isinstance(m, dict) and m.get("role") == "user"]
        if users:
            return users[-1]
    for k in ("prompt", "question", "user", "input", "text"):
        if isinstance(obj, dict) and isinstance(obj.get(k), str):
            return obj[k]
    return None


def instrument_index() -> dict:
    sent, users, per, samples = set(), set(), {}, []
    for name, p in INSTRUMENT_FILES.items():
        if not p.exists():
            raise SystemExit(f"instrument file missing, refusing to claim disjointness: {p}")
        n = 0
        for line in p.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            u = last_user(r)
            if u:
                users.add(h(norm(u)))
            for s in strings_of(r):
                if len(s) >= 80:
                    sent |= sentences(s)
            if n < 2 and u and len(u) > 100:
                samples.append({"instrument": name, "last_user": u})
            n += 1
        per[name] = {"rows": n, "sha256": sha256_file(p)}
    return {"sentences": sent, "users": users, "per_instrument": per, "controls": samples}


def flagged(row_user: str | None, row_text: str, idx: dict) -> bool:
    if row_user and h(norm(row_user)) in idx["users"]:
        return True
    return bool(sentences(row_text) & idx["sentences"])


# ---------------------------------------------------------------- sources
def v3_rows(classes, max_per_class=None):
    for c in classes:
        if c in EXCLUDED_V3:
            raise AssertionError(f"excluded class requested: {c}")
        with open(V3 / f"{c}.jsonl", encoding="utf-8") as f:
            for line in f:
                yield c, json.loads(line)


def one_per_family(rows):
    seen, out = set(), []
    for c, r in rows:
        fam = r.get("metadata", {}).get("template_family") or r["id"]
        if fam not in seen:
            seen.add(fam)
            out.append((c, r))
    return out


def render(tok, messages, add_generation_prompt):
    return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt,
                                   enable_thinking=False)


def read_text_file(p: Path, limit: int) -> str:
    try:
        s = p.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return ""
    s = s.replace("\r\n", "\n")          # StoryForge is CRLF-only; normalised and recorded
    if any(m in s.casefold() for m in MARKERS):
        return ""
    return s[:limit]


def chunks(s: str, size: int, n: int):
    s = s.strip()
    step = max(size, len(s) // max(1, n))
    return [s[i:i + size] for i in range(0, len(s), step)][:n]


_WALK = {}


def walk(root: Path) -> list[str]:
    """One directory walk per root, shared by every extension (this disk is slow: a full
    site-packages walk measured 6 min 40 s)."""
    key = str(root)
    if key not in _WALK:
        out = []
        for d, dirs, files in os.walk(root):
            dirs[:] = [x for x in dirs if x != "__pycache__"]
            out += [os.path.join(d, f) for f in files]
        _WALK[key] = out
    return _WALK[key]


def files_sampled(pattern_roots, exts, n, salt, max_bytes=400_000, minified_line=300):
    paths = []
    for root in pattern_roots:
        paths += [p for p in walk(root) if p.endswith(tuple(exts))]
    paths = [p for p in paths if os.path.getsize(p) <= max_bytes]
    out = []
    for p in ordered(paths, salt):
        s = read_text_file(Path(p), 6000)
        if not s.strip():
            continue
        lines = s.splitlines() or [""]
        if sum(len(l) for l in lines) / len(lines) > minified_line:
            continue                      # minified bundle, not representative text
        out.append((p, s))
        if len(out) >= n:
            break
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seed", type=int, default=17)
    a = ap.parse_args(argv)
    hostbudget.require(RAM_REQUIRE, "SMALLVD-001 calibration build")
    a.out.mkdir(parents=True, exist_ok=False)
    from transformers import AutoTokenizer
    import transformers
    if transformers.__version__ != "4.51.3":
        raise SystemExit("render under the probes' pinned transformers 4.51.3")
    tok = AutoTokenizer.from_pretrained(str(a.teacher), local_files_only=True)
    idx = instrument_index()

    # POSITIVE CONTROL first: real instrument rows must be flagged, or no 0 below means anything
    ctrl = [flagged(c["last_user"], c["last_user"], idx) for c in idx["controls"]]
    if not ctrl or not all(ctrl):
        raise SystemExit(f"POSITIVE CONTROL FAILED: {sum(ctrl)}/{len(ctrl)} instrument rows flagged")
    # SENTENCE PATH, controlled separately: an instrument's text wrapped in new words so the exact
    # user-turn match cannot fire. If only the exact path works, a paraphrased or re-templated copy
    # would pass unseen.
    sent_ctrl = [flagged(None, "An unrelated preface sentence. " + c["last_user"] + " A closing remark.",
                         {"users": set(), "sentences": idx["sentences"]})
                 for c in idx["controls"] if sentences(c["last_user"])]
    if not sent_ctrl or not all(sent_ctrl):
        raise SystemExit(f"SENTENCE-PATH CONTROL FAILED: {sum(sent_ctrl)}/{len(sent_ctrl)}")
    # NEGATIVE control: fresh text must NOT be flagged, or the check flags everything
    if flagged("A brand-new question about lighthouses and tides.",
               "A brand-new question about lighthouses and tides that no instrument has ever asked anyone.", idx):
        raise SystemExit("NEGATIVE CONTROL FAILED: fresh text was flagged")

    sources, corpus, panels, flags = {}, [], [], {}

    def note_source(name, paths):
        sources[name] = {"files": len(paths), "sha256_of_sorted_file_hashes":
                         h("".join(sorted(sha256_file(Path(p)) for p in paths)))}

    def add_corpus(cat, src, rid, text, user=None):
        if len(corpus) % 2000 == 0:
            hostbudget.guard(RAM_GUARD, "SMALLVD-001 calibration build")
        if flagged(user, text, idx):
            flags.setdefault(cat, []).append(rid)
            return
        split = "fit" if int(h(rid), 16) % 10 < 7 else "heldout"
        corpus.append({"id": rid, "category": cat, "split": split, "source": src, "text": text})

    # ---- Rineya chat sources (untouched v3 classes; cc0 train; RTG-1 train)
    all_panel_classes = [c for v in PANEL_CLASSES.values() for c in v]
    v3 = one_per_family(v3_rows(all_panel_classes))
    by_id = {r["id"]: (c, r) for c, r in v3}
    cls_group = {c: g for g, cs in PANEL_CLASSES.items() for c in cs}
    corpus_cat = {"concise_reasoning": "concise_reasoning", "grounded_extraction": "grounded_answering",
                  "instruction_writing": "instruction", "conversation": "conversation"}
    for rid in ordered(by_id, "v3-corpus")[:6000]:
        c, r = by_id[rid]
        msgs = r["messages"]
        g = cls_group[c]
        if g == "instruction_writing":
            add_corpus("instruction", f"v3:{c}", rid, msgs[0]["content"] + "\n\n" + msgs[-2]["content"], msgs[-2]["content"])
        else:
            add_corpus(corpus_cat[g], f"v3:{c}", rid, render(tok, msgs, False), msgs[-2]["content"])
    note_source("v3_by_class_untouched", [str(V3 / f"{c}.jsonl") for c in all_panel_classes])

    cc0_train = sorted(glob.glob(str(CC0 / "train" / "*.jsonl")))[:2]
    n = 0
    for p in cc0_train:
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                u = r["messages"][1]["content"]
                add_corpus("retrieval_context", "cc0:train", r["id"], u[:4000], None)
                n += 1
                if n >= 1500:
                    break
        if n >= 1500:
            break
    note_source("cc0_train_first_shards", cc0_train)
    rtg_train = RTG1 / "train.jsonl"
    with open(rtg_train, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f]
    rt = {x["id"]: x for x in rows}
    for rid in ordered(rt, "rtg-corpus")[:1500]:
        x = rt[rid]
        add_corpus("concise_reasoning", "rtg1:train", rid,
                   render(tok, x["messages"] + [{"role": "assistant", "content": x["completion"]}], False),
                   x["messages"][-1]["content"])
    note_source("rtg1_train", [str(rtg_train)])

    # ---- non-Rineya text
    story = sorted(glob.glob(str(STORY / "**" / "*.txt"), recursive=True))
    for p in story:
        s = read_text_file(Path(p), 10_000_000)
        for i, ch in enumerate(chunks(s, 2000, 90)):
            add_corpus("english_prose", "storyforge", f"story:{Path(p).stem}:{i}", ch)
    note_source("storyforge", story)
    rst = files_sampled([SITE], [".rst"], 400, "rst")
    for p, s in rst:
        add_corpus("english_prose", "site-packages:rst", f"rst:{h(p)[:16]}", s[:2500])
    note_source("site_packages_rst", [p for p, _ in rst])
    py = files_sampled([SITE / d for d in ("requests", "urllib3", "jinja2", "click", "rich", "httpx",
                                            "packaging", "pydantic", "yaml", "networkx")] + [STDLIB],
                       [".py"], 1600, "py")
    for p, s in py:
        add_corpus("python", "python-source", f"py:{h(p)[:16]}", s[:4000])
    note_source("python_sources", [p for p, _ in py])
    js = files_sampled([NPM / "npm" / "lib", NPM / "@nestjs"], [".js", ".ts"], 1400, "js")
    for p, s in js:
        add_corpus("javascript", "npm-js-ts", f"js:{h(p)[:16]}", s[:4000])
    note_source("npm_js_ts", [p for p, _ in js])
    jsn = files_sampled([NPM, SITE], [".json"], 1200, "json", max_bytes=200_000, minified_line=2000)
    for p, s in jsn:
        add_corpus("json", "json-files", f"json:{h(p)[:16]}", s[:3000])
    note_source("json_files", [p for p, _ in jsn])
    meta = sorted(glob.glob(str(SITE / "*.dist-info" / "METADATA")))
    for p in meta:
        s = read_text_file(Path(p), 20000)
        body = s.split("\n\n", 1)[1] if "\n\n" in s else s
        for i, ch in enumerate(chunks(body, 2500, 4)):
            add_corpus("markdown", "metadata-md", f"md:{h(p)[:16]}:{i}", ch)
        urls = "\n".join(l for l in s.splitlines() if "http" in l)
        if urls:
            add_corpus("numbers_dates_urls_paths", "metadata-urls", f"url:{h(p)[:16]}", urls[:2000])
    note_source("dist_info_metadata", meta)
    for cat in synthetic_text.GENERATORS:
        for i, t in enumerate(synthetic_text.generate(cat, 4000, a.seed)):
            add_corpus(cat, "synthetic", f"syn:{cat}:{i}", t)
    sources["synthetic"] = {"generator_sha256": sha256_file(HERE / "synthetic_text.py"), "seed": a.seed}

    # ---- LAYER PANELS (chat rows with references; text rows) -- rows are removed from the corpus
    used = set()

    def add_panel(cat, rid, kind, src, prompt=None, reference=None, scorer=None, text=None, user=None):
        if flagged(user, (prompt or "") + (reference or "") + (text or ""), idx):
            flags.setdefault("panel:" + cat, []).append(rid)
            return False
        panels.append({"id": rid, "category": cat, "kind": kind, "source": src,
                       "split": "select" if int(h(rid), 16) % 2 == 0 else "holdout",
                       "prompt_text": prompt, "reference": reference, "scorer": scorer, "text": text})
        used.add(rid)
        return True

    corpus_ids = {r["id"] for r in corpus}
    for g, classes in PANEL_CLASSES.items():
        pool = [rid for rid in ordered(by_id, f"panel:{g}") if by_id[rid][0] in classes and rid not in corpus_ids]
        k = 0
        for rid in pool:
            c, r = by_id[rid]
            msgs = r["messages"]
            k += add_panel(g, rid, "chat", f"v3:{c}", render(tok, msgs[:-1], True), msgs[-1]["content"],
                           "exact" if g in ("concise_reasoning", "grounded_extraction") else None,
                           user=msgs[-2]["content"])
            if k >= 64:
                break
    with open(RTG1 / "dev.jsonl", encoding="utf-8") as f:
        dev = {x["id"]: x for x in map(json.loads, f)}
    unk = [i for i in ordered(dev, "rtg-panel") if dev[i]["completion"] == "UNKNOWN"][:32]
    det = [i for i in ordered(dev, "rtg-panel") if dev[i]["completion"] != "UNKNOWN"][:32]
    for rid in unk + det:
        x = dev[rid]
        add_panel("abstention_unknown", rid, "chat", "rtg1:dev", render(tok, x["messages"], True),
                  x["completion"], "exact", user=x["messages"][-1]["content"])
    # cc0 validation packets start at ~10K characters; <= 11,000 keeps 733 single-question rows
    # (~2.7K tokens) spread over every reply class. The first build used 6,000 and got ZERO rows --
    # a whole protected category silently absent (see manifest "build_history").
    lc = {}
    for p in sorted(glob.glob(str(CC0 / "validation" / "*.jsonl"))):
        with open(p, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                if len(r["messages"]) == 3 and len(r["messages"][1]["content"]) <= LC_MAX_CHARS:
                    lc[r["id"]] = r
    k = 0
    for rid in ordered(lc, "lc-panel"):
        r = lc[rid]
        k += add_panel("long_context", rid, "chat", "cc0:validation",
                       render(tok, r["messages"][:-1], True), r["messages"][-1]["content"], "exact",
                       user=r["messages"][1]["content"])
        if k >= 64:
            break
    if k < 64:
        raise SystemExit(f"long_context panel short: {k}/64")
    text_src = {"english_prose": [r for r in corpus if r["source"] == "storyforge"],
                "python": [r for r in corpus if r["category"] == "python"],
                "javascript": [r for r in corpus if r["category"] == "javascript"],
                "markdown": [r for r in corpus if r["category"] == "markdown"]}
    for cat, rows in text_src.items():
        for r in [x for x in rows if x["split"] == "heldout"][:32]:
            add_panel("text_" + cat, "panel:" + r["id"], "text", r["source"], text=r["text"])
    # panel text rows came from the HELD-OUT corpus split; drop them from the corpus entirely so the
    # tokenizer never measures on a row the layer study also uses
    panel_src_ids = {p["id"][len("panel:"):] for p in panels if p["kind"] == "text"}
    corpus = [r for r in corpus if r["id"] not in panel_src_ids]

    counts = {}
    for p in panels:
        counts.setdefault(p["category"], {"select": 0, "holdout": 0})[p["split"]] += 1
    ccounts = {}
    for r in corpus:
        ccounts.setdefault(r["category"], {"fit": 0, "heldout": 0})[r["split"]] += 1

    def write_jsonl(path, rows):
        with open(path, "x", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    write_jsonl(a.out / "TOKENIZER_CORPUS.jsonl", corpus)
    write_jsonl(a.out / "LAYER_PANELS.jsonl", panels)
    manifest = {
        "schema": 1, "record": "SMALLVD_CALIBRATION_BUILD", "experiment": "SMALLVD-001",
        "builder_sha256": sha256_file(Path(__file__)), "seed": a.seed,
        "renderer": {"transformers": transformers.__version__, "enable_thinking": False},
        "excluded_v3_classes": sorted(EXCLUDED_V3), "panel_classes": PANEL_CLASSES,
        "instruments_checked": idx["per_instrument"],
        "positive_control": {"exact_path": {"injected": len(ctrl), "flagged": sum(ctrl)},
                             "sentence_path": {"injected": len(sent_ctrl), "flagged": sum(sent_ctrl)},
                             "negative_control_fresh_text_flagged": False},
        "build_history": BUILD_HISTORY, "long_context_max_chars": LC_MAX_CHARS,
        "flagged_and_dropped": {k: len(v) for k, v in flags.items()},
        "flagged_ids_sample": {k: v[:10] for k, v in flags.items()},
        "sources": sources,
        "tokenizer_corpus": {"rows": len(corpus), "by_category": ccounts,
                             "sha256": sha256_file(a.out / "TOKENIZER_CORPUS.jsonl")},
        "layer_panels": {"rows": len(panels), "by_category": counts,
                         "sha256": sha256_file(a.out / "LAYER_PANELS.jsonl")},
        "notes": ["StoryForge is CRLF-only; normalised to LF before chunking",
                  "HSKILL-CONFIRM-002 was not opened; excluded by its source classes",
                  "panel text rows are removed from the tokenizer corpus"],
    }
    with open(a.out / "CALIBRATION_MANIFEST.json", "x", encoding="utf-8", newline="\n") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)
    print(json.dumps({k: manifest[k] for k in ("positive_control", "flagged_and_dropped")}, indent=1))
    print(json.dumps(manifest["tokenizer_corpus"]["by_category"], indent=0))
    print(json.dumps(manifest["layer_panels"]["by_category"], indent=0))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
