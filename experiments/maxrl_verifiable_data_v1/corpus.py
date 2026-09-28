"""Reproducible corpus builder and independent persisted-data audit.

python -m experiments.maxrl_verifiable_data_v1.corpus build --out /new/directory
python -m experiments.maxrl_verifiable_data_v1.corpus audit --data /existing/directory
"""
from __future__ import annotations
import argparse
from collections import Counter
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import random
import re
from .generators import make
from .oracle import solutions
from .specs import VERSION, FAMILIES, canonical, digest, fingerprint, partition, parse_visible, render, require
from .verifier import score, compute_score

SPLITS = ("train", "dev", "test", "challenge", "warmstart")
BAD_FORMAT = ("", "{}", '{"answer":null}', '{"answer":true}', '{"answer":0.0}',
              '{"answer":[],"answer":[]}', '{"answer":[],"extra":1}', '{"answer":NaN}',
              '```json\n{"answer":[]}\n```', '{"answer":[]} trailing')

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as src:
        for block in iter(lambda: src.read(1048576), b""): h.update(block)
    return h.hexdigest()

def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as src:
        for number, line in enumerate(src, 1):
            try: yield json.loads(line)
            except ValueError as exc: raise ValueError(f"{path}:{number}: malformed JSON") from exc

def verl_row(row: dict) -> dict:
    # Only `prompt` is fed to the policy. No oracle references are included anywhere.
    return {"data_source": "rineya_maxrl_v1/"+row["family"], "prompt": row["prompt"],
            "ability": row["family"], "reward_model": {"style": "rule", "ground_truth": canonical(row["verifier"])},
            "extra_info": {k: row[k] for k in ("id", "parent_id", "split", "difficulty", "template_id")}}

def wrong_answer(f: str, s: dict, good: str) -> str:
    """A syntactically valid, same-envelope negative; check it really fails."""
    a = json.loads(good)["answer"]
    candidates = []
    if isinstance(a, str):
        for i, ch in enumerate(a):
            if ch in "+-*/":
                for op in "+-*/":
                    if op != ch: candidates.append(a[:i]+op+a[i+1:])
    else:
        for i in range(len(a)):
            for x in range(0, 12):
                if x != a[i]: candidates.append(a[:i]+[x]+a[i+1:])
        candidates.extend((list(reversed(a)), [x+1 for x in a]))
    for value in candidates:
        text = canonical({"answer": value})
        if score(f, s, text) == 0: return text
    raise ValueError("could not construct a live negative")

def source_manifest() -> dict:
    root = Path(__file__).parent
    return {str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))}

def quotas(scale: str) -> dict[str, dict[int, int]]:
    if scale == "full":
        return {"train": {0: 1000, 1: 1000, 2: 1000}, "dev": {0: 100, 1: 100, 2: 100},
                "test": {0: 100, 1: 100, 2: 100}, "challenge": {3: 300}, "warmstart": {0: 120, 1: 80}}
    if scale == "pilot":
        return {"train": {0: 12, 1: 12, 2: 12}, "dev": {0: 3, 1: 3, 2: 3},
                "test": {0: 3, 1: 3, 2: 3}, "challenge": {3: 6}, "warmstart": {0: 4, 1: 2}}
    raise ValueError("unknown scale")

def build(out: Path, scale: str = "full", seed: int = 260202710) -> dict:
    require(type(seed) is int and seed >= 0, "bad root seed")
    require(not out.exists(), "output must be a NEW directory; never overwrite a dataset")
    out.mkdir(parents=True)
    stats = Counter(); per_cell = {}; multi = Counter(); attempts = {}; proxy_lengths = []
    seen = set(); seen_prompts = set()
    with ExitStack() as stack:
        streams = {}
        for directory in ("canonical", "policy", "verl_jsonl", "oracles"):
            (out/directory).mkdir()
            for split in SPLITS:
                streams[directory, split] = stack.enter_context((out/directory/(split+".jsonl")).open("w", encoding="utf-8", newline="\n"))
        (out/"sft").mkdir()
        sft = stack.enter_context((out/"sft/warmstart.messages.jsonl").open("w", encoding="utf-8", newline="\n"))
        sft_meta = stack.enter_context((out/"sft/warmstart.lineage.jsonl").open("w", encoding="utf-8", newline="\n"))
        for split, levels in quotas(scale).items():
            for family in FAMILIES:
                for level, count in levels.items():
                    cell = f"{split}/{family}/{level}"
                    rng = random.Random(int(digest([seed, VERSION, split, family, level])[:16], 16))
                    written = 0; tried = 0
                    while written < count:
                        tried += 1
                        require(tried <= count*2000+10000, "bounded generation exhausted: "+cell)
                        spec = make(family, level, rng)
                        if spec is None: stats["construction_rejections"] += 1; continue
                        fp = fingerprint(family, spec)
                        if partition(fp) != split: continue
                        if fp in seen: stats["duplicate_rejections"] += 1; continue
                        references = solutions(family, spec)
                        require(bool(references), "construction not independently solvable: "+cell)
                        require(all(score(family, spec, ref) == 1 for ref in references), "oracle/verifier disagreement")
                        template = rng.randrange(4, 6) if split == "challenge" else rng.randrange(4)
                        prompt = render(family, spec, template)
                        require(parse_visible(prompt) == (family, spec), "visible specification mismatch")
                        ph = digest(prompt)
                        require(ph not in seen_prompts, "duplicate prompt")
                        seen.add(fp); seen_prompts.add(ph)
                        negative = wrong_answer(family, spec, references[0])
                        require(all(score(family, spec, bad) == 0 for bad in BAD_FORMAT), "format exploit")
                        identifier = "MRL1-"+fp
                        row = {"schema_version": VERSION, "id": identifier, "parent_id": identifier,
                               "problem_hash": fp, "prompt_hash": ph, "family": family, "split": split,
                               "difficulty": level, "template_id": template, "prompt": prompt,
                               "verifier": {"version": VERSION, "family": family, "spec": spec},
                               "provenance": {"origin": "original_bounded_generator", "root_seed": seed,
                                              "cell_attempt": tried, "model_generated": False}}
                        oracle = {"id": identifier, "origin": "independent_public_spec_search",
                                  "references": references, "distinct_references_lower_bound": len(references),
                                  "negative_control": negative, "negative_reward": 0,
                                  "is_policy_rollout": False}
                        exported = verl_row(row)
                        require(compute_score(exported["data_source"], references[0], exported["reward_model"]["ground_truth"]) == 1, "exported reward failed")
                        for directory, value in (("canonical", row), ("policy", {"id": identifier, "messages": prompt}), ("verl_jsonl", exported), ("oracles", oracle)):
                            streams[directory, split].write(canonical(value)+"\n")
                        if split == "warmstart":
                            target = references[rng.randrange(len(references))]
                            sft.write(canonical({"messages": prompt+[{"role": "assistant", "content": target}]})+"\n")
                            sft_meta.write(canonical({"id": identifier, "source_split": split, "target_origin": "symbolic_oracle_not_teacher_or_policy", "target_sha256": hashlib.sha256(target.encode()).hexdigest()})+"\n")
                        stats["rows"] += 1; stats[split] += 1; stats["reference_checks"] += len(references)
                        stats["negative_checks"] += len(BAD_FORMAT)+1
                        if len(references) > 1: multi[family] += 1
                        proxy_lengths.append(len(canonical(prompt)))
                        written += 1
                    attempts[cell] = tried; per_cell[cell] = written
                    print(f"built {cell}: {written}", flush=True)
    files = {str(p.relative_to(out)): {"sha256": file_hash(p), "bytes": p.stat().st_size} for p in sorted(out.rglob("*.jsonl"))}
    lengths = sorted(proxy_lengths)
    manifest = {"dataset": "Rineya-MaxRL-Verifiable-v1", "version": VERSION, "experiment": "MAXRL-DATA-001",
                "scale": scale, "root_seed": seed, "status": "SYNTHETIC_DATA_VERIFIED_NATIVE_UNQUALIFIED",
                "counts": dict(stats), "per_cell": per_cell, "multiple_reference_rows_by_family": dict(multi),
                "generation_attempts": attempts, "files": files, "generator_sources": source_manifest(),
                "serialized_prompt_characters_NOT_TOKENS": {"min": lengths[0], "median": lengths[len(lengths)//2], "p95": lengths[int(.95*(len(lengths)-1))], "max": lengths[-1]},
                "evaluation_visibility": "ALL INCLUDED SPLITS EXPOSED; NOT SEALED",
                "unqualified": ["exact Rineya tokenizer/length/EOS", "pretrained solvability and model difficulty", "external historical overlap", "semantic/isomorphism leakage", "human prompt review", "preservation bank", "native verl integration", "Parquet engine roundtrip", "native training and efficacy"]}
    (out/"manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest

def audit(data: Path, resolve: bool = True) -> dict:
    m = json.loads((data/"manifest.json").read_text())
    require(m["generator_sources"] == source_manifest(), "generator source drift")
    for path, expected in m["files"].items():
        p = data/path
        require(p.is_file() and p.stat().st_size == expected["bytes"] and file_hash(p) == expected["sha256"], "artifact hash/size mismatch: "+path)
    ids=set(); fps=set(); prompts=set(); stats=Counter(); observed_cells=Counter()
    for split in SPLITS:
        rows=iter(read_jsonl(data/"canonical"/(split+".jsonl")))
        ors=iter(read_jsonl(data/"oracles"/(split+".jsonl")))
        pols=iter(read_jsonl(data/"policy"/(split+".jsonl")))
        vrs=iter(read_jsonl(data/"verl_jsonl"/(split+".jsonl")))
        for row in rows:
            oracle=next(ors, None); pol=next(pols, None); vr=next(vrs, None)
            require(oracle is not None and pol is not None and vr is not None, "short sidecar")
            f, s = parse_visible(row["prompt"])
            require(row["schema_version"] == VERSION and f == row["family"] and row["verifier"] == {"version": VERSION, "family": f, "spec": s}, "unbound reward spec")
            fp=fingerprint(f,s); ph=digest(row["prompt"])
            require(fp == row["problem_hash"] and ph == row["prompt_hash"] and row["id"] == row["parent_id"] == "MRL1-"+fp and partition(fp) == row["split"] == split, "identity/split mismatch")
            require(row["id"] not in ids and fp not in fps and ph not in prompts, "cross-split or within-split duplicate")
            ids.add(row["id"]); fps.add(fp); prompts.add(ph)
            require(row["template_id"] in (4,5) if split == "challenge" else row["template_id"] in (0,1,2,3), "template leakage")
            require(row["prompt"] == render(f,s,row["template_id"]), "template identity mismatch")
            require(oracle["id"] == row["id"] and oracle["is_policy_rollout"] is False, "oracle identity/origin")
            require(pol == {"id": row["id"], "messages": row["prompt"]} and vr == verl_row(row), "policy/export field leakage or drift")
            require(oracle["references"] and all(score(f,s,a) == 1 for a in oracle["references"]), "bad positive certificate")
            require(score(f,s,oracle["negative_control"]) == 0 and all(score(f,s,a) == 0 for a in BAD_FORMAT), "false positive")
            if resolve: require(solutions(f,s) == oracle["references"], "visible-only independent re-solve mismatch")
            stats["rows"] += 1; stats[split] += 1; stats["positive_checks"] += len(oracle["references"]); stats["negative_checks"] += len(BAD_FORMAT)+1
            observed_cells[f"{split}/{f}/{row['difficulty']}"] += 1
        require(all(next(it, None) is None for it in (ors,pols,vrs)), "extra sidecar rows")
    require(dict(observed_cells) == m["per_cell"] and stats["rows"] == m["counts"]["rows"], "count/quota mismatch")
    warm = list(read_jsonl(data/"canonical/warmstart.jsonl"))
    sfts = list(read_jsonl(data/"sft/warmstart.messages.jsonl"))
    metas = list(read_jsonl(data/"sft/warmstart.lineage.jsonl"))
    require(len(warm) == len(sfts) == len(metas), "SFT line alignment")
    for row, view, meta in zip(warm,sfts,metas):
        require(set(view) == {"messages"} and view["messages"][:-1] == row["prompt"] and view["messages"][-1]["role"] == "assistant", "SFT boundary")
        target=view["messages"][-1]["content"]
        require(meta["id"] == row["id"] and meta["source_split"] == "warmstart" and meta["target_sha256"] == hashlib.sha256(target.encode()).hexdigest(), "SFT metadata")
        require(score(row["family"],row["verifier"]["spec"],target) == 1, "SFT target invalid")
    return {"status": "PASS", "manifest_sha256": file_hash(data/"manifest.json"), "independent_visible_resolve": resolve,
            "checks": dict(stats), "duplicates": 0, "split_violations": 0, "policy_oracle_leaks": 0,
            "scope": "included instances, exact canonical identities, public rendering and reference certificates only; NOT native model qualification"}

def main() -> None:
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest="command",required=True)
    b=sub.add_parser("build"); b.add_argument("--out",type=Path,required=True); b.add_argument("--scale",choices=("full","pilot"),default="full"); b.add_argument("--seed",type=int,default=260202710)
    a=sub.add_parser("audit"); a.add_argument("--data",type=Path,required=True); a.add_argument("--receipt",type=Path)
    args=p.parse_args()
    if args.command == "build": result=build(args.out,args.scale,args.seed)
    else:
        result=audit(args.data)
        if args.receipt:
            with args.receipt.open("x") as out: json.dump(result,out,indent=2,sort_keys=True); out.write("\n")
    print(json.dumps(result,indent=2,sort_keys=True))

if __name__ == "__main__": main()
