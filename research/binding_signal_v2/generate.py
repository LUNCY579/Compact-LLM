"""Build original synthetic binding-signal TRAIN/DEV data; no model or tokenizer.

No existing evaluation prompt is an input. All facts, entities, templates and
answers are generated here. CLI outputs are fresh directories beneath the fixed
records root. This is data readiness only, not a training or evaluation permit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

ROOT_SEED = 23
ARTIFACT_ROOT = Path("C:/Users/clclc/BusinessProject/rineya-dataset-retention-records/binding-signal-prep-20260920")

APPROVED_MANIFEST = ARTIFACT_ROOT.parent / "lcr-exclusion-manifest/LCR_EXCLUSION_MANIFEST.json"
APPROVED_MANIFEST_SHA256 = "54b02b35c37b1cc842dbf413aac56980d3d3ade203f3ea5092bb332416914858"
PARENT_BUILD = ARTIFACT_ROOT / "candidate-v1/BUILD.json"
PARENT_BUILD_SHA256 = "e2302e8f376409f5e4feec24c17126ac28699be7e4b55cff3ccd8a9f84016371"
PARENT_CODE_COMMIT = "b521cbdac08697070dae998054b83b51c2c6467d"
PARENT_GENERATOR_BLOB_SHA256 = "95eda94e7a71cb804db70c03915dc8a4bcfdeb444fbc4115234c9335ebdbd006"
LITERAL_FIELDS = ("forbidden_proper_nouns", "forbidden_titles", "forbidden_domains", "forbidden_task_type_names")

SYSTEM = (
    "Use only the facts in this record. For a person question, return the full name of the unique "
    "person explicitly assigned to the relevant role. If no unique person is determined, return "
    "UNKNOWN. A vacant role has no occupant. For a role question, return the exact role name "
    "specified for the duty, even when its occupant is unspecified or the role is vacant. "
    "A list of people or roles is not an assignment. Return only the name, role, or UNKNOWN."
)
PEOPLE = {
    "train": tuple(f"{first} {last}" for first in
                   ("Anika", "Bennet", "Carla", "Davin", "Estelle", "Florian", "Giselle", "Hector")
                   for last in ("Ashby", "Bexley", "Clifton", "Dawson", "Ellery", "Fenton", "Goodwin", "Hadley")),
    "dev": tuple(f"{first} {last}" for first in
                 ("Ilona", "Jonah", "Katya", "Lukas", "Miriam", "Nessa", "Oren", "Priya")
                 for last in ("Ingram", "Jacobs", "Keaton", "Latham", "Merrick", "Nesbitt", "Osborne", "Phelan")),
}
ROLES = {
    "train": ("equipment steward", "practice coordinator", "court marshal", "fitness guide",
              "fixture organizer", "ball custodian", "jersey keeper", "warmup leader",
              "score recorder", "membership clerk", "training captain", "refreshment helper"),
    "dev": ("fabric custodian", "needle keeper", "pattern organizer", "sewing coordinator",
            "thread steward", "button sorter", "knitting guide", "quilt planner",
            "embroidery mentor", "stitch tutor", "ribbon keeper", "craft convener"),
}
ACTIONS = {
    "train": ("arrange the practice cones", "prepare the spare jerseys", "count the training balls",
              "check the court markings", "organize the warmup drills", "collect the fitness mats"),
    "dev": ("sort the fabric bundles", "prepare the sewing kits", "organize the pattern folders",
            "count the spare buttons", "arrange the knitting needles", "collect the embroidery hoops"),
}
DOMAINS = {"train": "community sports club", "dev": "community craft and sewing circle"}

TEMPLATES = {
    "train_club_register": {
        "split": "train", "opening": "Community sports club roster.",
        "roles": "Club roles: {roles}.", "people": "Participants: {people}.",
        "occupied": "{person} serves as {role}.", "vacant": "The {role} role is vacant.",
        "duty": "The duty to {action} belongs to the {role}.",
        "person": ("Who holds the club role responsible for the duty to {action}?",
                   "Who is explicitly assigned to the club role responsible for the duty to {action}?",
                   "Whose club role carries responsibility for the duty to {action}?"),
        "role": ("Identify the club role responsible for the duty to {action}.",
                 "Provide the club role responsible for the duty to {action}.",
                 "Specify the club role responsible for the duty to {action}."),
    },
    "train_practice_bulletin": {
        "split": "train", "opening": "Community sports club practice bulletin.",
        "roles": "Duty roles in this bulletin: {roles}.", "people": "Participants listed in this bulletin: {people}.",
        "occupied": "The {role} appointment is held by {person}.",
        "vacant": "The {role} appointment is unfilled.",
        "duty": "Responsibility to {action} sits with the {role} appointment.",
        "person": ("Who occupies the appointment responsible to {action}?",
                   "Who is established as the appointment holder responsible to {action}?",
                   "Whose appointment carries the responsibility to {action}?"),
        "role": ("Identify the appointment carrying the responsibility to {action}.",
                 "Provide the role attached to the responsibility to {action}.",
                 "Specify the appointment attached to the responsibility to {action}."),
    },
    "dev_sewing_circle": {
        "split": "dev", "opening": "Community sewing circle appointments.",
        "roles": "Circle appointments: {roles}.", "people": "Circle participants: {people}.",
        "occupied": "The circle records {person} in the {role} appointment.",
        "vacant": "The circle has nobody in the {role} appointment.",
        "duty": "The task to {action} is allocated to the {role} appointment.",
        "person": ("Who holds the circle appointment allocated the task to {action}?",
                   "Who is recorded in the circle appointment allocated the task to {action}?",
                   "Whose circle appointment is allocated the task to {action}?"),
        "role": ("Identify the circle appointment allocated the task to {action}.",
                 "Provide the circle role allocated the task to {action}.",
                 "Specify the circle appointment allocated the task to {action}."),
    },
    "dev_craft_circle": {
        "split": "dev", "opening": "Community craft circle duty sheet.",
        "roles": "Circle roles in this sheet: {roles}.", "people": "Participants in this sheet: {people}.",
        "occupied": "For this circle, {role} is a role held by {person}.",
        "vacant": "For this circle, the {role} role has no occupant.",
        "duty": "The role tasked to {action} is {role}.",
        "person": ("Who occupies the circle role tasked to {action}?",
                   "Who is explicitly recorded as holding the circle role tasked to {action}?",
                   "Whose circle role has the task to {action}?"),
        "role": ("Identify the circle role tasked to {action}.",
                 "Provide the role tasked to {action}.",
                 "Specify the circle role bearing the task to {action}."),
    },
}
CONDITIONS = ("occupied", "missing", "vacant")
CATEGORIES = ("person_answerable", "person_missing", "person_vacant", "role_answerable")
LICENSE_STATUS = "UNSPECIFIED_INTERNAL_RESEARCH"

# Independent input grammars: never derive these from TEMPLATES or render_user.
INPUT_GRAMMARS = {
    "Community sports club roster.": {
        "family": "train_club_register", "roles": r"Club roles: (.+)\.",
        "people": r"Participants: (.+)\.", "occupied": r"(.+) serves as (.+)\.", "role_first": False,
        "vacant": r"The (.+) role is vacant\.", "duty": r"The duty to (.+) belongs to the (.+)\.",
        "questions": (
            ("person", r"Who holds the club role responsible for the duty to (.+)\?"),
            ("person", r"Who is explicitly assigned to the club role responsible for the duty to (.+)\?"),
            ("person", r"Whose club role carries responsibility for the duty to (.+)\?"),
            ("role", r"Identify the club role responsible for the duty to (.+)\."),
            ("role", r"Provide the club role responsible for the duty to (.+)\."),
            ("role", r"Specify the club role responsible for the duty to (.+)\.")),
    },
    "Community sports club practice bulletin.": {
        "family": "train_practice_bulletin", "roles": r"Duty roles in this bulletin: (.+)\.",
        "people": r"Participants listed in this bulletin: (.+)\.",
        "occupied": r"The (.+) appointment is held by (.+)\.", "role_first": True,
        "vacant": r"The (.+) appointment is unfilled\.",
        "duty": r"Responsibility to (.+) sits with the (.+) appointment\.",
        "questions": (
            ("person", r"Who occupies the appointment responsible to (.+)\?"),
            ("person", r"Who is established as the appointment holder responsible to (.+)\?"),
            ("person", r"Whose appointment carries the responsibility to (.+)\?"),
            ("role", r"Identify the appointment carrying the responsibility to (.+)\."),
            ("role", r"Provide the role attached to the responsibility to (.+)\."),
            ("role", r"Specify the appointment attached to the responsibility to (.+)\.")),
    },
    "Community sewing circle appointments.": {
        "family": "dev_sewing_circle", "roles": r"Circle appointments: (.+)\.",
        "people": r"Circle participants: (.+)\.",
        "occupied": r"The circle records (.+) in the (.+) appointment\.", "role_first": False,
        "vacant": r"The circle has nobody in the (.+) appointment\.",
        "duty": r"The task to (.+) is allocated to the (.+) appointment\.",
        "questions": (
            ("person", r"Who holds the circle appointment allocated the task to (.+)\?"),
            ("person", r"Who is recorded in the circle appointment allocated the task to (.+)\?"),
            ("person", r"Whose circle appointment is allocated the task to (.+)\?"),
            ("role", r"Identify the circle appointment allocated the task to (.+)\."),
            ("role", r"Provide the circle role allocated the task to (.+)\."),
            ("role", r"Specify the circle appointment allocated the task to (.+)\.")),
    },
    "Community craft circle duty sheet.": {
        "family": "dev_craft_circle", "roles": r"Circle roles in this sheet: (.+)\.",
        "people": r"Participants in this sheet: (.+)\.",
        "occupied": r"For this circle, (.+) is a role held by (.+)\.", "role_first": True,
        "vacant": r"For this circle, the (.+) role has no occupant\.",
        "duty": r"The role tasked to (.+) is (.+)\.",
        "questions": (
            ("person", r"Who occupies the circle role tasked to (.+)\?"),
            ("person", r"Who is explicitly recorded as holding the circle role tasked to (.+)\?"),
            ("person", r"Whose circle role has the task to (.+)\?"),
            ("role", r"Identify the circle role tasked to (.+)\."),
            ("role", r"Provide the role tasked to (.+)\."),
            ("role", r"Specify the circle role bearing the task to (.+)\.")),
    },
}


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")


def digest(value) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def random_for(*parts) -> random.Random:
    return random.Random(int(digest([ROOT_SEED, *parts]), 16))


def family_digest(name: str) -> str:
    # Split labels are metadata, not evidence that wording is independent.
    return digest({key: value for key, value in TEMPLATES[name].items() if key != "split"})


def validate_world(world: dict) -> dict[str, dict]:
    """Validate explicit facts, without consulting templates or stored targets."""
    require(isinstance(world, dict), "world must be an object")
    roles, people, assignments = world.get("roles"), world.get("people"), world.get("assignments")
    require(isinstance(roles, list) and len(roles) == 6 and len(set(roles)) == 6,
            "world requires six distinct roles")
    require(isinstance(people, list) and len(people) == 6 and len(set(people)) == 6,
            "world requires six distinct people")
    require(all(isinstance(value, str) and value and value != "UNKNOWN" for value in roles + people),
            "invalid entity name")
    require(isinstance(assignments, list), "assignment facts missing")
    by_role = {}
    for fact in assignments:
        require(isinstance(fact, dict) and set(fact) == {"role", "status", "person"}, "invalid assignment fact")
        role = fact["role"]
        require(role in roles and role not in by_role, "ambiguous or duplicate role binding")
        require(fact["status"] in ("occupied", "vacant"), "unknown assignment status")
        require((fact["status"] == "occupied" and fact["person"] in people) or
                (fact["status"] == "vacant" and fact["person"] is None), "invalid occupant/vacancy fact")
        by_role[role] = fact
    require(isinstance(world.get("action"), str) and world["action"], "duty action missing")
    return by_role


def oracle(world: dict, query: dict) -> str:
    """Answer from assignments and the duty's designated role, never template labels.

    Missing and vacant assignments both determine no person. A role question
    requires only the role designation, irrespective of the person's availability.
    """
    assignments = validate_world(world)
    require(query.get("kind") in ("person", "role"), "unknown question kind")
    role = query.get("target_role")
    require(role in world["roles"], "question designates an unknown role")
    if query["kind"] == "role":
        return role
    fact = assignments.get(role)
    return fact["person"] if fact is not None and fact["status"] == "occupied" else "UNKNOWN"


def condition_of(world: dict, query: dict) -> str:
    fact = validate_world(world).get(query["target_role"])
    return "missing" if fact is None else fact["status"]


def decode_policy_input(text: str) -> dict:
    """Reconstruct facts and question from user text alone, with no gold argument."""
    require(isinstance(text, str), "policy input must be text")
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    require(len(lines) >= 5 and lines[0] in INPUT_GRAMMARS, "policy input has no recognized independent grammar")
    grammar = INPUT_GRAMMARS[lines[0]]
    roles = re.fullmatch(grammar["roles"], lines[1])
    people = re.fullmatch(grammar["people"], lines[2])
    require(roles is not None and people is not None, "policy input omitted its role/person domains")
    world = {"roles": roles[1].split("; "), "people": people[1].split("; "), "assignments": []}
    duty = None
    for line in lines[3:-1]:
        occupied = re.fullmatch(grammar["occupied"], line)
        vacant = re.fullmatch(grammar["vacant"], line)
        designation = re.fullmatch(grammar["duty"], line)
        require(sum(match is not None for match in (occupied, vacant, designation)) == 1,
                "policy input has an unrecognized or ambiguous fact sentence")
        if occupied is not None:
            role, person = (occupied[1], occupied[2]) if grammar["role_first"] else (occupied[2], occupied[1])
            world["assignments"].append({"role": role, "status": "occupied", "person": person})
        elif vacant is not None:
            world["assignments"].append({"role": vacant[1], "status": "vacant", "person": None})
        else:
            require(duty is None, "policy input has multiple duty designations")
            duty = (designation[2], designation[1]) if grammar.get("duty_role_first") else (designation[1], designation[2])
    require(duty is not None, "policy input omitted its duty-to-role designation")
    matches = [(kind, match[1]) for kind, pattern in grammar["questions"]
               for match in [re.fullmatch(pattern, lines[-1])] if match is not None]
    require(len(matches) == 1, "policy input question is missing or ambiguous")
    kind, question_action = matches[0]
    action, role = duty
    require(question_action == action, "policy question and duty describe different actions")
    world["action"] = action
    query = {"kind": kind, "target_role": role}
    answer = oracle(world, query)
    return {"template_family": grammar["family"], "world": world, "query": query, "answer": answer}


def verify_policy_evidence(text: str, world: dict, query: dict) -> None:
    parsed = decode_policy_input(text)
    facts = parsed["world"]
    require(parsed["template_family"] == world["template_family"], "policy input template family differs")
    require(set(facts["roles"]) == set(world["roles"]) and set(facts["people"]) == set(world["people"]),
            "policy input domains differ from world facts")
    require(sorted(facts["assignments"], key=lambda f: f["role"]) ==
            sorted(world["assignments"], key=lambda f: f["role"]),
            "policy input omitted or changed assignment/vacancy evidence")
    require(facts["action"] == world["action"] and parsed["query"]["target_role"] == query["target_role"],
            "policy input duty-to-role designation differs")
    require(parsed["query"]["kind"] == query["kind"], "policy input role/person question kind differs")
    require(parsed["answer"] == oracle(world, query), "policy-input oracle answer differs from world oracle")


def make_world(split: str, index: int) -> dict:
    require(split in PEOPLE and type(index) is int and index >= 0, "invalid split/world index")
    rng = random_for("world", split, index)
    roles, people = rng.sample(ROLES[split], 6), rng.sample(PEOPLE[split], 6)
    occupied = roles[:4]
    assignments = [{"role": role, "status": "occupied", "person": person}
                   for role, person in zip(occupied, people[:4])]
    assignments.append({"role": roles[4], "status": "vacant", "person": None})
    rng.shuffle(assignments)
    family = rng.choice(sorted(name for name, template in TEMPLATES.items() if template["split"] == split))
    world = {"schema": 1, "split": split, "index": index,
             "template_family": family, "roles": roles, "people": people,
             "assignments": assignments, "action": rng.choice(ACTIONS[split]),
             "view_targets": {"occupied": rng.choice(occupied), "vacant": roles[4], "missing": roles[5]}}
    world["base_world_id"] = "BINDV2-" + split.upper() + "-" + digest(["world-identity", ROOT_SEED, split, index, world])[:24]
    validate_world(world)
    return world


def queries(world: dict):
    # Three person paraphrases per condition (nine views). One role question for
    # each condition uses a different paraphrase (three views): four equal groups.
    for index, condition in enumerate(CONDITIONS):
        for paraphrase in range(3):
            yield {"kind": "person", "target_role": world["view_targets"][condition],
                   "condition": condition, "paraphrase": paraphrase}
        yield {"kind": "role", "target_role": world["view_targets"][condition],
               "condition": condition, "paraphrase": index}


def render_user(world: dict, query: dict) -> str:
    template = TEMPLATES[world["template_family"]]
    facts = [template[fact["status"]].format(role=fact["role"], person=fact["person"])
             for fact in world["assignments"]]
    rng = random_for("render-order", world["split"], world["base_world_id"], query)
    rng.shuffle(facts)
    roles, people = list(world["roles"]), list(world["people"])
    rng.shuffle(roles); rng.shuffle(people)
    duty = template["duty"].format(action=world["action"], role=query["target_role"])
    facts.insert(rng.randrange(len(facts) + 1), duty)
    return "\n".join([template["opening"], template["roles"].format(roles="; ".join(roles)),
                       template["people"].format(people="; ".join(people)), *facts, "",
                       template[query["kind"]][query["paraphrase"]].format(action=world["action"])])


def make_row(world: dict, query: dict) -> dict:
    answer = oracle(world, query)
    category = "role_answerable" if query["kind"] == "role" else (
        "person_answerable" if condition_of(world, query) == "occupied" else "person_" + condition_of(world, query))
    row_id = world["base_world_id"] + ":" + digest(query)[:16]
    return {"schema": 1, "id": row_id, "parent_id": world["base_world_id"],
            "base_world_id": world["base_world_id"], "split": world["split"], "source_split": world["split"],
            "family": "role_person_binding", "category": category,
            "template_family": world["template_family"], "template_sha256": family_digest(world["template_family"]),
            "world": world, "query": query, "reference": answer,
            "messages": [{"role": "system", "content": SYSTEM},
                         {"role": "user", "content": render_user(world, query)},
                         {"role": "assistant", "content": answer}],
            "license": LICENSE_STATUS, "provenance": "original self-generated synthetic, no external corpus read",
            "human_data_review": "NOT_PERFORMED", "quality_claim": False, "release_permitted": False}


def generate(train_worlds: int = 64, dev_worlds: int = 32) -> dict[str, list[dict]]:
    require(type(train_worlds) is int and 1 <= train_worlds <= 64 and
            type(dev_worlds) is int and 1 <= dev_worlds <= 32, "world counts exceed the bounded build scope")
    return {split: [make_row(world, query) for index in range(count)
                    for world in [make_world(split, index)] for query in queries(world)]
            for split, count in (("train", train_worlds), ("dev", dev_worlds))}


def approved_manifest() -> dict:
    raw = APPROVED_MANIFEST.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == APPROVED_MANIFEST_SHA256, "approved exclusion manifest identity changed")
    manifest = json.loads(raw.decode("utf-8"))
    require(all(manifest[key] is False for key in ("contains_questions", "contains_answers", "contains_answer_key")),
            "exclusion aid unexpectedly contains protected content")
    return manifest


def literal_normalize(text: str) -> str:
    return " ".join(re.sub(r"[\W_]+", " ", unicodedata.normalize("NFKC", text).casefold()).split())


def literal_contacts(dataset: dict[str, list[dict]], manifest: dict) -> dict:
    """Check complete user turns against every approved literal term; never filter."""
    terms = {field: [(term, " " + literal_normalize(term) + " ") for term in manifest[field]
                     if literal_normalize(term)] for field in LITERAL_FIELDS}
    report = {}
    for split, rows in dataset.items():
        padded = [(row["id"], " " + literal_normalize(message["content"]) + " ")
                  for row in rows for message in row["messages"] if message["role"] == "user"]
        fields = {}
        for field, values in terms.items():
            fields[field] = []
            for term, needle in values:
                matches = sorted({row_id for row_id, text in padded if needle in text})
                if matches:
                    fields[field].append({"term": term, "row_ids": matches, "row_count": len(matches)})
        report[split] = fields
    return report


def audit(dataset: dict[str, list[dict]]) -> dict:
    require(set(dataset) == {"train", "dev"}, "exact TRAIN/DEV partitions required")
    catalog = {split: {family_digest(name) for name, template in TEMPLATES.items() if template["split"] == split}
               for split in ("train", "dev")}
    require(not catalog["train"] & catalog["dev"], "TRAIN/DEV template-content overlap")
    ids, worlds, world_content, semantic_worlds, entities, families, categories = set(), {}, {}, {}, {}, {}, {}
    prompt_hashes = {split: set() for split in dataset}
    for split, rows in dataset.items():
        require(isinstance(rows, list) and rows, "empty split")
        entities[split], families[split], categories[split] = set(), set(), Counter()
        by_world = {}
        for row in rows:
            require(row.get("split") == row.get("source_split") == split, "row crossed its declared partition")
            require(isinstance(row.get("id"), str) and row["id"] not in ids, "duplicate row ID")
            ids.add(row["id"])
            world, query = row["world"], row["query"]
            answer = oracle(world, query)
            facts = validate_world(world)
            require(len(facts) == 5 and sum(f["status"] == "occupied" for f in facts.values()) == 4
                    and sum(f["status"] == "vacant" for f in facts.values()) == 1
                    and len({f["person"] for f in facts.values() if f["status"] == "occupied"}) == 4,
                    "build worlds require four distinct occupants, one vacancy and one unspecified role")
            world_id = world["base_world_id"]
            require(world.get("split") == split and row.get("parent_id") == row.get("base_world_id") == world_id,
                    "world derivative crossed partitions or changed parent")
            require(world_id not in worlds or worlds[world_id] == split, "base world appears in multiple partitions")
            worlds[world_id] = split
            fingerprint = digest(world)
            require(world_id not in world_content or world_content[world_id] == fingerprint,
                    "derivatives of a base world have conflicting facts")
            world_content[world_id] = fingerprint
            semantic_fingerprint = digest({"roles": sorted(world["roles"]), "people": sorted(world["people"]),
                "assignments": sorted(world["assignments"], key=lambda fact: fact["role"]),
                "action": world["action"], "view_targets": world["view_targets"]})
            require(semantic_fingerprint not in semantic_worlds or semantic_worlds[semantic_fingerprint] == world_id,
                    "duplicate semantic world under a different world ID")
            semantic_worlds[semantic_fingerprint] = world_id
            family = world["template_family"]
            require(family in TEMPLATES and TEMPLATES[family]["split"] == split and row.get("template_family") == family,
                    "template family leaked across partitions")
            require(row.get("template_sha256") == family_digest(family), "template bytes differ")
            families[split].add(family_digest(family))
            require(set(world["roles"]) <= set(ROLES[split]) and set(world["people"]) <= set(PEOPLE[split]),
                    "entity pool leaked across partitions")
            entities[split].update(world["roles"] + world["people"])
            require(world["action"] in ACTIONS[split], "action pool leaked across partitions")
            require(query.get("condition") == condition_of(world, query), "declared condition differs from facts")
            require(query in list(queries(world)), "unregistered view or paraphrase")
            messages = row.get("messages")
            require(isinstance(messages, list) and len(messages) == 3 and
                    [message.get("role") for message in messages] == ["system", "user", "assistant"],
                    "policy messages have an invalid structure")
            verify_policy_evidence(messages[1].get("content"), world, query)
            expected = make_row(world, query)
            require(row["id"] == expected["id"] and row.get("category") == expected["category"],
                    "view identity/category differs from oracle")
            require(row.get("reference") == answer and row.get("messages") == expected["messages"],
                    "wrong answer or rendered prompt differs from independent world oracle")
            require(row.get("license") == LICENSE_STATUS and row.get("human_data_review") == "NOT_PERFORMED"
                    and row.get("release_permitted") is False,
                    "licensing/review status changed")
            by_world.setdefault(world_id, []).append(digest(query))
            categories[split][row["category"]] += 1
            prompt_hashes[split].add(digest(row["messages"][:2]))
        for world_id, views in by_world.items():
            require(len(views) == len(set(views)) == 12, "base world must retain all twelve unique derivative views")
        require(set(categories[split]) == set(CATEGORIES) and len(set(categories[split].values())) == 1,
                "answerability/control categories are not balanced")
    require(not entities["train"] & entities["dev"], "TRAIN/DEV entity overlap")
    require(not families["train"] & families["dev"], "TRAIN/DEV template-content overlap")
    require(not prompt_hashes["train"] & prompt_hashes["dev"], "TRAIN/DEV rendered-prompt overlap")
    manifest = approved_manifest()
    contacts = literal_contacts(dataset, manifest)
    require(not any(matches for fields in contacts.values() for matches in fields.values()),
            "approved literal exclusion contact: " + json.dumps(contacts, sort_keys=True))
    return {"status": "BUILD_ONLY_BLOCKED_EXTERNAL_COVERAGE", "internal_build_status": "PASS", "root_seed": ROOT_SEED,
            "worlds": dict(Counter(worlds.values())), "rows": {split: len(rows) for split, rows in dataset.items()},
            "category_counts": {split: dict(counts) for split, counts in categories.items()},
            "approved_literal_exclusions": {"status": "PASS_LITERAL_CONTACT_ONLY", "manifest_path": str(APPROVED_MANIFEST),
                "manifest_sha256": APPROVED_MANIFEST_SHA256, "contacts": contacts,
                "normalization": "NFKC, casefold, punctuation/underscore to spaces, whitespace collapse, complete phrase token boundaries; no stopword removal"},
            "eight_gram_check": "UNVERIFIED_EXACT_STOPWORD_AND_HASHING_ALGORITHM_UNAVAILABLE",
            "eight_gram_collision_count": None,
            "reserved_index_check": "UNVERIFIED_APPROVED_ROW_PARENT_ENTITY_PROMPT_INDICES_UNAVAILABLE",
            "reserved_index_overlap": None, "open_bank_check": "PENDING_FOR_NEW_V2_BYTES",
            "external_exclusions_complete": False, "world_namespace": "BINDV2", "domains": DOMAINS,
            "world_overlap": 0, "entity_overlap": 0, "template_content_overlap": 0, "prompt_overlap": 0,
            "always_unknown_accuracy": 0.5, "person_positive_control_share": 0.25,
            "role_positive_control_share": 0.25, "all_related_views_grouped": True,
            "independent_policy_input_semantics_verified": True,
            "tokenizer_compiled": False, "human_data_review": "NOT_PERFORMED", "quality_claim": False}


def write_dataset(out: Path, dataset: dict[str, list[dict]]) -> dict:
    """Validate before creating any output; preserve partial output on IO failure."""
    validation = audit(dataset)
    require(hashlib.sha256(PARENT_BUILD.read_bytes()).hexdigest() == PARENT_BUILD_SHA256,
            "failed v1 parent build identity changed")
    out = Path(out).absolute()
    require(not out.exists() and not out.is_symlink(), "output directory must be fresh")
    require(".." not in out.parts and not any(p.is_symlink() for p in out.parents), "unsafe output path")
    out.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for split, rows in dataset.items():
        path = out / (split.upper() + ".jsonl")
        with path.open("xb") as stream:
            for row in rows:
                stream.write(canonical(row) + b"\n")
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"schema": 1, "record": "BINDING_SIGNAL_V2_DATA_READINESS", **validation,
                "status": "BUILD_ONLY_BLOCKED_EXTERNAL_COVERAGE", "files_sha256": hashes,
                "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "parent_lineage": {"candidate": str(PARENT_BUILD.parent), "build_sha256": PARENT_BUILD_SHA256,
                    "status": "FAILED_EXTERNAL_EXCLUSIONS_PRESERVED_UNCHANGED", "code_commit": PARENT_CODE_COMMIT,
                    "generator_blob_sha256": PARENT_GENERATOR_BLOB_SHA256},
                "system_contract_sha256": digest(SYSTEM), "license": LICENSE_STATUS,
                "license_basis": "Original self-generated synthetic content; no external corpus read; licensing decision not made.",
                "input_eval_rows_read": 0, "only_external_content_read": "approved exclusion metadata plus parent BUILD metadata", "training_permission": False, "evaluation_permission": False,
                "release_permitted": False, "token_and_dose_budget": "PENDING_NATIVE_TOKENIZER_COMPILATION"}
    with (out / "BUILD.json").open("xb") as stream:
        stream.write(canonical(manifest) + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh name/path beneath binding-signal-prep-20260920")
    parser.add_argument("--train-worlds", type=int, default=64)
    parser.add_argument("--dev-worlds", type=int, default=32)
    args = parser.parse_args()
    out = (args.out if args.out.is_absolute() else ARTIFACT_ROOT / args.out).absolute()
    require(out.resolve().is_relative_to(ARTIFACT_ROOT.resolve()) and out.resolve() != ARTIFACT_ROOT.resolve(),
            "CLI output must be a fresh child of the designated binding-signal records root")
    manifest = write_dataset(out, generate(args.train_worlds, args.dev_worlds))
    print(json.dumps({"status": manifest["status"], "out": str(out), "rows": manifest["rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
