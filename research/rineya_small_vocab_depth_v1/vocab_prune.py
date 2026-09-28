"""SMALLVD-001 Phases 5-6 -- closure-respecting BPE vocabulary pruning and explicit id remapping.

WHY NOT "DELETE NON-ENGLISH TOKENS". The teacher tokenizer is byte-level BPE (Qwen2): 256 base
byte symbols, 151,387 merges, each non-base token created by exactly ONE defining merge (a, b).
A token can only ever be emitted if its whole merge ancestry is present. Deleting a token that
LOOKS foreign but is an intermediate of an English token (many are) silently makes that English
token unreachable. Deleting by Unicode range is both too crude and structurally unsafe.

THE CONSTRUCTION.
  * A retained set R is VALID iff it contains all 256 base symbols and is closed under
    defining-merge ancestry. Then the tokenizer built from R's merges can only emit tokens in R,
    and because the bases are kept EVERY byte string stays encodable. Byte-level decoding is
    concatenation, so round-trip exactness is inherited, not hoped for (and still tested).
  * Removal is GREEDY BY EFFECTIVE UTILITY: repeatedly remove the removable token (no retained
    children, not protected) whose mixture-weighted count is smallest, then credit that count to
    both parents. Each occurrence of a removed token is modelled as becoming one occurrence of each
    parent, so the effective count is the marginal extra tokens paid for removing it. That is
    an approximation of BPE re-segmentation, and the REAL expansion is measured afterwards by
    re-tokenising held-out text (tokenizer_validation.py). The approximation is never reported
    as a result.
  * Protected tokens (protected_tokens.py) are never removable, and neither are their ancestors,
    because a token with a retained child is not removable.

REMAPPING. old_id -> new_id is MONOTONE: retained regular tokens keep their relative order and
are renumbered 0..K-1, and the 26 added tokens follow in their original order. The 267 padding rows
(151,669..151,935) are never produced by the tokenizer and are dropped. Everything that names a
token id -- tokenizer.json added_tokens, tokenizer_config added_tokens_decoder,
config/generation_config bos/eos/pad -- is rewritten from the SAME map.
"""
from __future__ import annotations

import copy
import hashlib
import heapq
import json
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def canonical_json_bytes(obj) -> bytes:
    """What gets written to disk AND hashed -- one serialisation, so the hash is of the file."""
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def merges_of(tj: dict) -> list[tuple[str, str]]:
    out = []
    for m in tj["model"]["merges"]:
        a, b = (m.split(" ", 1) if isinstance(m, str) else m)
        out.append((a, b))
    return out


def defining_parents(vocab: dict, merges: list) -> tuple[dict, dict, list]:
    """token -> (a, b) of its lowest-rank merge; token -> that rank; and the base symbols."""
    parents, rank = {}, {}
    for r, (a, b) in enumerate(merges):
        t = a + b
        if t in vocab and t not in parents:
            parents[t] = (a, b)
            rank[t] = r
    bases = sorted((t for t in vocab if t not in parents), key=lambda t: vocab[t])
    return parents, rank, bases


def is_closed(retained: set, parents: dict) -> bool:
    return all(parents[t][0] in retained and parents[t][1] in retained
               for t in retained if t in parents)


def greedy_prune(vocab: dict, merges: list, utility: dict, protected: set, target_regular: int):
    """Return (retained set, removal log). `utility[t]` is t's mixture-weighted count.

    Tie-break when utilities are equal (most tokens have utility 0 on any finite corpus): remove
    the LATER merge first. Later merges are longer and rarer in the teacher's own training
    distribution, so the teacher's merge order is used as the prior where the calibration corpus
    is silent. This makes the result fully deterministic."""
    parents, rank, bases = defining_parents(vocab, merges)
    if target_regular < len(bases):
        raise ValueError("target below the 256 base symbols -- byte coverage would be lost")
    retained = set(vocab)
    eff = {t: float(utility.get(t, 0.0)) for t in vocab}
    nchild = dict.fromkeys(vocab, 0)
    for t, (a, b) in parents.items():
        nchild[a] += 1
        nchild[b] += 1
    locked = set(protected) | set(bases)
    unknown = locked - set(vocab)
    if unknown:
        raise ValueError(f"{len(unknown)} protected tokens are not in the vocabulary")
    heap = []

    def push(t):
        heapq.heappush(heap, (eff[t], -rank.get(t, -1), vocab[t], t))

    for t in vocab:
        if t not in locked and nchild[t] == 0:
            push(t)
    log = []
    while len(retained) > target_regular:
        if not heap:
            raise RuntimeError(f"cannot reach {target_regular}: the protected closure alone "
                               f"has {len(retained)} tokens")
        e, _, _, t = heapq.heappop(heap)
        if t not in retained or nchild[t] != 0 or e != eff[t]:
            continue
        retained.discard(t)
        log.append((t, e))
        a, b = parents[t]
        for p in (a, b):          # a == b is visited twice: two occurrences, two child links
            eff[p] += eff[t]
            nchild[p] -= 1
        for p in {a, b}:
            if p in retained and p not in locked and nchild[p] == 0:
                push(p)
    assert is_closed(retained, parents), "closure violated -- this is a bug, not a data issue"
    return retained, log


def build_id_map(vocab: dict, added: list, retained: set) -> dict:
    """Monotone old_id -> new_id over retained regular tokens then every added token."""
    kept = sorted(vocab[t] for t in retained)
    id_map = {old: new for new, old in enumerate(kept)}
    k = len(kept)
    for i, a in enumerate(sorted(added, key=lambda x: x["id"])):
        id_map[a["id"]] = k + i
    return id_map


def prune_tokenizer_json(tj: dict, retained: set) -> tuple[dict, dict]:
    vocab = tj["model"]["vocab"]
    if tj["model"]["type"] != "BPE" or tj["model"].get("byte_fallback") or tj["model"].get("ignore_merges"):
        raise ValueError("only plain byte-level BPE without byte_fallback/ignore_merges is supported")
    for part in ("post_processor", "decoder"):
        if (tj.get(part) or {}).get("type") != "ByteLevel":
            raise ValueError(f"{part} is not ByteLevel -- it may carry token ids that need remapping")
    parents, _, _ = defining_parents(vocab, merges_of(tj))
    if not is_closed(retained, parents):
        raise ValueError("retained set is not closed under merge ancestry")
    # tokenizers 0.21.4 IGNORES the "id" field of added_tokens and assigns ids by LIST POSITION
    # after the model vocab (measured: swapping two id fields changes nothing; swapping two list
    # entries swaps their ids). The remap below is only correct if list order == id order and
    # the regular ids are dense, so both are asserted rather than assumed.
    added_ids = [a["id"] for a in tj["added_tokens"]]
    if added_ids != sorted(added_ids):
        raise ValueError("added_tokens list is not in id order -- the library would renumber it")
    id_map = build_id_map(vocab, tj["added_tokens"], retained)
    new = copy.deepcopy(tj)
    new["model"]["vocab"] = {t: id_map[vocab[t]] for t in sorted(retained, key=lambda t: vocab[t])}
    new["model"]["merges"] = [[a, b] for a, b in merges_of(tj)
                              if a in retained and b in retained and (a + b) in retained]
    for a in new["added_tokens"]:
        a["id"] = id_map[a["id"]]
    k = len(retained)
    assert [a["id"] for a in new["added_tokens"]] == list(range(k, k + len(new["added_tokens"])))
    assert sorted(new["model"]["vocab"].values()) == list(range(k))
    return new, id_map


def remap_tokenizer_config(tc: dict, id_map: dict) -> dict:
    new = copy.deepcopy(tc)
    dec = tc.get("added_tokens_decoder", {})
    new["added_tokens_decoder"] = {str(id_map[int(k)]): v for k, v in
                                   sorted(dec.items(), key=lambda kv: int(kv[0]))}
    return new


def _remap_ids(v, id_map):
    if v is None:
        return None
    if isinstance(v, list):
        return [id_map[int(x)] for x in v]
    return id_map[int(v)]


def remap_model_config(cfg: dict, id_map: dict, vocab_rows: int, n_layers: int | None = None) -> dict:
    new = copy.deepcopy(cfg)
    new["vocab_size"] = vocab_rows
    for key in ("bos_token_id", "eos_token_id", "pad_token_id"):
        if key in new:
            new[key] = _remap_ids(new[key], id_map)
    if n_layers is not None:
        new["num_hidden_layers"] = n_layers
        if new.get("max_window_layers") is not None:
            new["max_window_layers"] = min(new["max_window_layers"], n_layers)
        if isinstance(new.get("layer_types"), list):
            raise ValueError("layer_types present -- depth surgery must subset it explicitly")
    return new


def remap_generation_config(gc: dict, id_map: dict) -> dict:
    new = copy.deepcopy(gc)
    for key in ("bos_token_id", "eos_token_id", "pad_token_id", "decoder_start_token_id"):
        if key in new:
            new[key] = _remap_ids(new[key], id_map)
    return new


def slow_files(new_tj: dict) -> tuple[bytes, bytes]:
    """vocab.json and merges.txt regenerated FROM the new tokenizer.json, so the slow and fast
    tokenizers cannot disagree. Byte format matches the teacher's (version header, trailing \\n)."""
    vocab = json.dumps(new_tj["model"]["vocab"], ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    merges = ("#version: 0.2\n" + "".join(f"{a} {b}\n" for a, b in new_tj["model"]["merges"])).encode("utf-8")
    return vocab, merges


def rows_for(target_rows: int, n_added: int) -> int:
    """Regular tokens to keep so regular + added fills EXACTLY target_rows embedding rows."""
    return target_rows - n_added


def load_teacher_tokenizer_files(teacher_dir: Path) -> dict:
    t = Path(teacher_dir)
    return {
        "tokenizer.json": json.loads((t / "tokenizer.json").read_text(encoding="utf-8")),
        "tokenizer_config.json": json.loads((t / "tokenizer_config.json").read_text(encoding="utf-8")),
        "config.json": json.loads((t / "config.json").read_text(encoding="utf-8")),
        "generation_config.json": json.loads((t / "generation_config.json").read_text(encoding="utf-8")),
    }


def write_tokenizer_dir(out_dir: Path, files: dict, new_tj: dict, id_map: dict, target_rows: int,
                        n_layers: int | None = None) -> dict:
    """Write a complete tokenizer + config sidecar set. Create-only: refuses to overwrite."""
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "tokenizer.json": canonical_json_bytes(new_tj),
        "tokenizer_config.json": canonical_json_bytes(remap_tokenizer_config(files["tokenizer_config.json"], id_map)),
        "config.json": canonical_json_bytes(remap_model_config(files["config.json"], id_map, target_rows, n_layers)),
        "generation_config.json": canonical_json_bytes(remap_generation_config(files["generation_config.json"], id_map)),
    }
    payload["vocab.json"], payload["merges.txt"] = slow_files(new_tj)
    payload["token_id_map.json"] = canonical_json_bytes(
        {"schema": 1, "direction": "old_id -> new_id", "n": len(id_map),
         "map": {str(k): v for k, v in sorted(id_map.items())}})
    hashes = {}
    for name, data in payload.items():
        with open(out_dir / name, "xb") as f:
            f.write(data)
        hashes[name] = sha256_bytes(data)
    return hashes
