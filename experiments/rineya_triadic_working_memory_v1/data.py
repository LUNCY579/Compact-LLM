"""Fresh procedural diagnostic data; never discovers or opens existing holdouts.

Disjoint entity namespaces and surface templates across train/dev/holdout.
Oracle prompts contain supporting FACTS, not the requested final answer.
This is an experiment diagnostic, not a substitute for established retention.
"""
import hashlib
import json
from pathlib import Path
import random

FAMILIES = ("binding", "update", "unrelated_after_update", "scope", "two_hop",
            "three_hop", "missing", "conflict", "role_state", "format_constraint")
SPLITS = ("train", "dev", "holdout", "retention_demo")
TEMPLATES = {
    "train": ("{e}'s {r} is {v}.", "What is {e}'s {r}?"),
    "dev": ("Record: {e}; attribute {r}; value {v}.", "For {e}, retrieve the value of {r}."),
    "holdout": ("{r}({e}) = {v}.", "Resolve {r}({e})."),
    "retention_demo": ("{e} has {v} as its {r}.", "Give the {r} associated with {e}."),
}


def generate_rows(split: str, count: int, seed: int = 20261005, distractors: int = 4):
    if split not in SPLITS or count < 1 or not 0 <= distractors <= 256:
        raise ValueError("invalid split/count/distractor count")
    rng = random.Random(f"twm-001:{split}:{seed}")
    fact_template, query_template = TEMPLATES[split]
    rows = []
    for i in range(count):
        family = FAMILIES[i % len(FAMILIES)]
        group = f"{split}-{seed}-{i}"
        prefix = f"{split[0]}{i}"
        e, other = f"person_{prefix}", f"person_{prefix}_other"
        a, b, c, d = [f"place_{prefix}_{x}" for x in rng.sample(range(100), 4)]
        facts, support, intermediate, probes = [], [], [], []
        def fact(entity, relation, value):
            return fact_template.format(e=entity, r=relation, v=value)
        def add(entity, relation, value, useful=True):
            text = fact(entity, relation, value)
            facts.append(text)
            if useful:
                support.append(text)
        question = query_template.format(e=e, r="home")
        answer = a
        policy = "Use only the records. Reply UNKNOWN for an absent fact."
        if family in ("binding", "update", "unrelated_after_update", "format_constraint"):
            add(e, "home", a)
            add(e, "work", b, useful=False)
            add(other, "home", b, useful=False)
            if family == "update":
                policy += " Records are chronological; a later value replaces an earlier one for the same attribute."
                add(e, "home", c)
                answer = c
            elif family == "unrelated_after_update":
                policy += " Records are chronological; updates change only the named attribute."
                add(e, "work", c, useful=False)
            elif family == "format_constraint":
                policy += " Wrap the final value in square brackets."
                answer = f"[{a}]"
        elif family == "scope":
            add(e, "home_in_simulation", b, useful=False)
            add(e, "home_in_actual_world", a)
            question = query_template.format(e=e, r="home_in_actual_world")
        elif family in ("two_hop", "three_hop"):
            item, box, room = f"item_{prefix}", f"box_{prefix}", f"room_{prefix}"
            add(item, "container", box)
            add(box, "location", room)
            intermediate = [box, room]
            probes = [dict(question=query_template.format(e=item, r="container"), answer=box),
                      dict(question=query_template.format(e=box, r="location"), answer=room)]
            if family == "three_hop":
                add(room, "building", a)
                question = f"Follow container, then location, then building from {item}. Give the final building."
            else:
                question = f"Follow container, then location from {item}. Give the final room."
                answer = room
            add(other, "location", b, useful=False)
        elif family == "missing":
            add(e, "work", b, useful=False)
            answer = "UNKNOWN"
        elif family == "conflict":
            policy = "These records are unordered. Reply CONFLICT when distinct values are asserted for the queried attribute."
            add(e, "home", a)
            add(e, "home", b)
            answer = "CONFLICT"
        elif family == "role_state":
            policy = "Respect each character's knowledge. Other characters' knowledge is not shared. Reply UNKNOWN when needed."
            add(other, "known_password", a, useful=False)
            add(e, "known_city", b, useful=False)
            question = query_template.format(e=e, r="known_password")
            answer = "UNKNOWN"
        for j in range(distractors):
            add(f"noise_{prefix}_{j}", rng.choice(("home", "work", "location")),
                f"place_{prefix}_noise{j}", useful=False)
        # Keep update ordering fixed; prompt/oracle differ only in available facts.
        suffix = f"\n{question}\nReturn only the answer, without explanation."
        prompt = policy + "\n" + "\n".join(facts) + suffix
        oracle = policy + "\n" + "\n".join(support) + suffix
        row = dict(id=group, group_id=group, split=split, family=family, prompt=prompt,
                   retrieval_probes=probes,
                   oracle_prompt=oracle, answer=answer, intermediate_answers=intermediate,
                   generator="twm-001-v1", data_seed=seed, distractors=distractors,
                   template_partition=split, entity_partition=split)
        row["row_sha256"] = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
        rows.append(row)
    return rows


def with_retrieval_probes(rows):
    """Add direct-edge queries sharing a cluster with their parent reasoning task."""
    output = []
    for row in rows:
        output.append(row)
        for i, probe in enumerate(row.get("retrieval_probes", [])):
            extra = dict(row)
            extra.update(id=f"{row['id']}:edge{i+1}", family="hop_edge_retrieval", answer=probe["answer"],
                         retrieval_probes=[], intermediate_answers=[])
            for field in ("prompt", "oracle_prompt"):
                prefix = row[field].rsplit("\n", 2)[0]
                extra[field] = prefix + "\n" + probe["question"] + "\nReturn only the answer, without explanation."
            extra.pop("row_sha256", None)
            extra["row_sha256"] = hashlib.sha256(json.dumps(extra, sort_keys=True).encode()).hexdigest()
            output.append(extra)
    return output


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def load_rows(path, purpose: str, allow_holdout: bool = False):
    if purpose not in ("train", "eval"):
        raise ValueError("purpose must be train or eval")
    path = Path(path)
    # Do not use this utility to access pre-existing sealed project benchmarks.
    if any("sealed" in part.lower() or "test2" in part.lower() for part in path.parts):
        raise ValueError("existing sealed/TEST2 paths are excluded")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    ids = set()
    for row in rows:
        required = ("id", "group_id", "split", "family", "prompt", "answer")
        if any(k not in row for k in required) or row["id"] in ids:
            raise ValueError("malformed row or duplicate ID")
        ids.add(row["id"])
        if purpose == "train" and row["split"] != "train":
            raise ValueError("training may read only declared training rows")
        if purpose == "eval" and row["split"] == "train":
            raise ValueError("training rows cannot be reported as evaluation")
        if row["split"] == "holdout" and not allow_holdout:
            raise ValueError("prospective holdout is closed; explicit allow_holdout required")
        if row["split"] not in SPLITS + ("retention",):
            raise ValueError("unknown split")
        if not all(isinstance(row[k], str) and row[k] for k in required):
            raise ValueError("row fields must be nonempty strings")
    if not rows:
        raise ValueError("empty data file")
    return rows
