from __future__ import annotations

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from ..hashing import atomic_write_json, canonical_sha256, sha256_file


FAMILIES = (
    "general_reasoning",
    "paraphrase_robustness",
    "counterfactual_reasoning",
    "unanswerable_abstention",
    "long_context_retrieval",
    "lost_in_the_middle",
    "multi_hop_binding",
    "common_sense",
    "mathematics",
    "science",
    "grounded_qa_rag",
    "normal_conversation",
    "instruction_following",
    "short_response",
)


SCIENCE_FACTS = (
    ("Which particle has a negative electric charge?", "electron"),
    ("What gas do plants absorb during photosynthesis?", "carbon dioxide"),
    ("What is the SI unit of electric current?", "ampere"),
    ("Which organelle is the main site of ATP production?", "mitochondrion"),
    ("What force keeps planets in orbit?", "gravity"),
    ("What process changes liquid water into vapor?", "evaporation"),
    ("Which blood cells primarily carry oxygen?", "red blood cells"),
    ("What is the chemical symbol for sodium?", "na"),
    ("Which layer of Earth is liquid and surrounds the inner core?", "outer core"),
    ("What type of bond shares electron pairs?", "covalent bond"),
    ("What is the smallest unit of an element retaining its properties?", "atom"),
    ("Which electromagnetic wave has a longer wavelength: infrared or ultraviolet?", "infrared"),
    ("What is the pH of a neutral aqueous solution near room temperature?", "7"),
    ("Which law relates force, mass, and acceleration?", "newton's second law"),
    ("What molecule carries genetic information in most organisms?", "dna"),
    ("Which planet is known for its prominent ring system?", "saturn"),
    ("What is the phase change from gas directly to solid?", "deposition"),
    ("Which part of a cell controls entry and exit of substances?", "cell membrane"),
    ("What kind of energy is stored in a stretched spring?", "elastic potential energy"),
    ("What is the approximate speed of light in vacuum in meters per second?", "300000000"),
    ("Which enzyme unwinds DNA during replication?", "helicase"),
    ("What is the SI unit of frequency?", "hertz"),
    ("Which subatomic particle determines atomic number?", "proton"),
    ("What is the main gas in Earth's atmosphere?", "nitrogen"),
    ("What instrument measures atmospheric pressure?", "barometer"),
    ("Which vitamin is synthesized in skin exposed to sunlight?", "vitamin d"),
    ("What is the SI unit of energy?", "joule"),
    ("Which planet is closest to the Sun?", "mercury"),
    ("What structure connects muscle to bone?", "tendon"),
    ("What is the process by which cells divide into two identical daughter cells?", "mitosis"),
    ("Which metal is liquid near room temperature?", "mercury"),
    ("What is the SI base unit of temperature?", "kelvin"),
    ("Which part of the brain coordinates balance and movement?", "cerebellum"),
    ("What type of rock forms from cooled magma?", "igneous rock"),
    ("Which gas is most associated with the greenhouse effect from fossil fuels?", "carbon dioxide"),
    ("What is the name for an animal that eats only plants?", "herbivore"),
    ("Which lens corrects nearsightedness?", "concave lens"),
    ("What is the central region of an atom called?", "nucleus"),
    ("Which process converts atmospheric nitrogen into usable compounds?", "nitrogen fixation"),
    ("What is the SI unit of electrical resistance?", "ohm"),
)


COMMON_SENSE = (
    ("A glass falls from a table onto a hard floor. What is most likely to happen?", "it breaks"),
    ("If a wet shirt is left in sunlight, what will most likely happen?", "it dries"),
    ("What should you use to unlock a conventional locked door?", "a key"),
    ("If ice is placed in a warm room, what happens?", "it melts"),
    ("What do people usually carry when rain is expected?", "an umbrella"),
    ("If a phone battery reaches zero percent, what happens?", "the phone turns off"),
    ("Which is safer for touching a hot pan: a bare hand or an oven mitt?", "an oven mitt"),
    ("If you do not water a houseplant for a long time, what may happen?", "it wilts"),
    ("Where would you normally store frozen food?", "a freezer"),
    ("If the traffic light is red, what should a driver do?", "stop"),
    ("What happens to a shadow when the light source is blocked?", "it disappears"),
    ("If a cup has a hole in its bottom, what happens to water poured into it?", "it leaks out"),
    ("Which tool is normally used to cut paper: scissors or a spoon?", "scissors"),
    ("What happens when a book is dropped and nothing supports it?", "it falls"),
    ("Where do people usually sleep at night?", "a bed"),
    ("If a room is dark, what can you switch on to see better?", "a light"),
    ("What should be done with food that smells spoiled?", "discard it"),
    ("If shoes are too small, how will they likely feel?", "tight"),
    ("What happens to paper placed in a flame?", "it burns"),
    ("Which container is suitable for carrying water: a sealed bottle or a net?", "a sealed bottle"),
    ("If someone whispers, is the sound usually loud or quiet?", "quiet"),
    ("What do you use to tell time on your wrist?", "a watch"),
    ("If a bicycle tire is flat, what should be added?", "air"),
    ("What happens to a puddle after several hot, dry days?", "it evaporates"),
    ("What should you do before crossing a busy street?", "look both ways"),
    ("Where should perishable milk normally be stored?", "a refrigerator"),
    ("If a pencil becomes blunt, what tool can make it pointed again?", "a sharpener"),
    ("What happens to an open book when a strong wind blows its pages?", "the pages turn"),
    ("If a lamp is unplugged, will it normally stay lit?", "no"),
    ("What should you wear on your head when riding a bicycle for safety?", "a helmet"),
    ("If water is put in a freezer long enough, what does it become?", "ice"),
    ("Which room in a home is most associated with cooking?", "the kitchen"),
    ("What happens when soap is added while washing greasy hands?", "the grease is removed"),
    ("If a clock is running fast, will it show a later or earlier time than the correct time?", "a later time"),
    ("What should be used to erase a graphite pencil mark?", "an eraser"),
    ("If a sealed balloon is punctured, what happens to its air?", "it escapes"),
    ("Which footwear is more suitable for deep snow: sandals or boots?", "boots"),
    ("What do you normally do with a letter before mailing it?", "put it in an envelope"),
    ("If a mirror is clean, what can you see in it?", "a reflection"),
    ("What happens to a candle flame when covered by an airtight jar?", "it goes out"),
)


def _case(
    *,
    split: str,
    family: str,
    scenario_id: str,
    transformation_id: str,
    prompt: str,
    scorer: dict[str, Any],
    max_new_tokens: int = 24,
    metadata: dict[str, Any] | None = None,
    format_scorer: dict[str, Any] | None = None,
) -> dict[str, Any]:
    identity = canonical_sha256(
        [split, family, scenario_id, transformation_id, prompt]
    )[:16]
    return {
        "schema_version": "1.0",
        "case_id": f"{split}_{family}_{identity}",
        "split": split,
        "family": family,
        "scenario_id": scenario_id,
        "transformation_id": transformation_id,
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "scorer": scorer,
        "max_new_tokens": max_new_tokens,
        "metadata": metadata or {},
        "format_scorer": format_scorer,
    }


def _arithmetic(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    a, b, c = rng.randint(11, 97), rng.randint(3, 19), rng.randint(-40, 40)
    answer = a * b + c
    scenario = f"{split}_math_{index:04d}"
    return [
        _case(
            split=split,
            family="mathematics",
            scenario_id=scenario,
            transformation_id="canonical",
            prompt=f"Compute ({a} × {b}) + ({c}). Return only the integer.",
            scorer={"type": "numeric", "value": answer},
            max_new_tokens=12,
            format_scorer={"type": "regex", "pattern": r"[-+]?\d+"},
        )
    ]


def _paraphrase(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    start, change = rng.randint(20, 90), rng.randint(3, 25)
    answer = start - change
    scenario = f"{split}_para_{index:04d}"
    prompts = (
        f"A box held {start} tokens. {change} were removed. How many remain? Return only the integer.",
        f"Starting with {start} tokens, take away {change}. State only the number left.",
        f"What is {start} minus {change}? Give just the integer answer.",
    )
    return [
        _case(
            split=split,
            family="paraphrase_robustness",
            scenario_id=scenario,
            transformation_id=f"p{variant}",
            prompt=prompt,
            scorer={"type": "numeric", "value": answer},
            max_new_tokens=12,
            format_scorer={"type": "regex", "pattern": r"[-+]?\d+"},
        )
        for variant, prompt in enumerate(prompts)
    ]


def _ordering(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    names = rng.sample(("Ari", "Bex", "Cato", "Dina", "Enzo", "Faye"), 4)
    scenario = f"{split}_reason_{index:04d}"
    prompt = (
        f"{names[0]} is older than {names[1]}. {names[1]} is older than {names[2]}. "
        f"{names[2]} is older than {names[3]}. Who is youngest? Return only the name."
    )
    return [
        _case(
            split=split,
            family="general_reasoning",
            scenario_id=scenario,
            transformation_id="chain",
            prompt=prompt,
            scorer={"type": "keywords", "phrase": names[3]},
            max_new_tokens=12,
            format_scorer={"type": "exact", "answers": [names[3]]},
        )
    ]


def _counterfactual(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    colors = rng.sample(
        (
            "amber",
            "azure",
            "beige",
            "blue",
            "bronze",
            "coral",
            "crimson",
            "green",
            "indigo",
            "ivory",
            "magenta",
            "ochre",
            "silver",
            "teal",
            "violet",
            "yellow",
        ),
        3,
    )
    scenario = f"{split}_counter_{index:04d}"
    prompt = (
        "In a counterfactual world, every object that is "
        f"{colors[0]} is also {colors[1]}, and every {colors[1]} object is {colors[2]}. "
        f"Object N is {colors[0]}. Under those rules, is N {colors[2]}? Answer YES or NO only."
    )
    return [
        _case(
            split=split,
            family="counterfactual_reasoning",
            scenario_id=scenario,
            transformation_id="rule_chain",
            prompt=prompt,
            scorer={"type": "choice", "allowed": ["YES", "NO"], "value": "YES"},
            max_new_tokens=8,
            format_scorer={"type": "exact", "answers": ["YES"]},
        )
    ]


def _unanswerable(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    code = rng.randint(1000, 9999)
    scenario = f"{split}_unknown_{index:04d}"
    prompt = (
        f"Context: The Ralo unit has code {code}. The Tivan unit is stored in cabinet "
        f"{rng.randint(2, 30)}. Question: What is the code of the Merek unit? "
        "If the context does not say, answer exactly UNKNOWN."
    )
    return [
        _case(
            split=split,
            family="unanswerable_abstention",
            scenario_id=scenario,
            transformation_id="missing_fact",
            prompt=prompt,
            scorer={"type": "keywords", "phrase": "UNKNOWN"},
            max_new_tokens=12,
            format_scorer={"type": "exact", "answers": ["UNKNOWN"]},
        )
    ]


def _rag(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    entities = rng.sample(("Ralo", "Tivan", "Merek", "Solen", "Pavo", "Keris"), 4)
    values = rng.sample(range(1200, 9800), 4)
    target = rng.randrange(4)
    context = " ".join(
        f"The {entity} unit has access code {value}."
        for entity, value in zip(entities, values, strict=True)
    )
    scenario = f"{split}_rag_{index:04d}"
    return [
        _case(
            split=split,
            family="grounded_qa_rag",
            scenario_id=scenario,
            transformation_id="grounded",
            prompt=f"Context: {context}\nQuestion: What is the access code of the {entities[target]} unit? Return only the code.",
            scorer={"type": "numeric", "value": values[target]},
            max_new_tokens=12,
            format_scorer={"type": "regex", "pattern": r"\d+"},
        )
    ]


def _binding(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    people = rng.sample(("Ari", "Bex", "Cato", "Dina", "Enzo", "Faye"), 4)
    item = rng.choice(("key", "map", "ring", "token"))
    scenario = f"{split}_binding_{index:04d}"
    prompt = (
        f"{people[0]} gave the {item} to {people[1]}. Then {people[1]} gave it to "
        f"{people[2]}. Finally {people[2]} gave it to {people[3]}. Who has the {item} now? "
        "Return only the name."
    )
    return [
        _case(
            split=split,
            family="multi_hop_binding",
            scenario_id=scenario,
            transformation_id="three_hop",
            prompt=prompt,
            scorer={"type": "keywords", "phrase": people[3]},
            max_new_tokens=12,
            format_scorer={"type": "exact", "answers": [people[3]]},
        )
    ]


def _long_context(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    target_code = rng.randint(100000, 999999)
    scenario = f"{split}_long_{index:04d}"
    positions = (("early", 5), ("middle", 50), ("late", 95))
    rows = []
    for label, position in positions:
        sentences = [
            f"Record {slot:03d}: unit U{rng.randint(100, 999)} has archival value {rng.randint(100000, 999999)}."
            for slot in range(100)
        ]
        sentences[position] = f"Record {position:03d}: the Zorin target has archival value {target_code}."
        prompt = (
            "Use only the records below.\n"
            + "\n".join(sentences)
            + "\nQuestion: What is the archival value of the Zorin target? Return only the integer."
        )
        family = "lost_in_the_middle" if label == "middle" else "long_context_retrieval"
        rows.append(
            _case(
                split=split,
                family=family,
                scenario_id=scenario,
                transformation_id=label,
                prompt=prompt,
                scorer={"type": "numeric", "value": target_code},
                max_new_tokens=12,
                metadata={"target_position": label, "record_count": 100},
                format_scorer={"type": "regex", "pattern": r"\d+"},
            )
        )
    return rows


def _instruction(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    label = f"tag-{rng.randint(100000, 999999)}"
    scenario = f"{split}_instruction_{index:04d}"
    return [
        _case(
            split=split,
            family="instruction_following",
            scenario_id=scenario,
            transformation_id="json",
            prompt=f'Return exactly this JSON object and nothing else: {{"label":"{label}","ok":true}}',
            scorer={"type": "json_exact", "value": {"label": label, "ok": True}},
            max_new_tokens=24,
            format_scorer={"type": "json_exact", "value": {"label": label, "ok": True}},
        )
    ]


def _conversation(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    letters = "abcdefghjkmnpqrstuvwxyz"
    name = "".join(rng.choice(letters) for _ in range(6)).capitalize()
    scenario = f"{split}_chat_{index:04d}"
    constraint = {
        "type": "constraints",
        "minimum_words": 3,
        "maximum_words": 7,
        "contains": [name],
        "forbids": ["!"],
    }
    return [
        _case(
            split=split,
            family="normal_conversation",
            scenario_id=scenario,
            transformation_id="brief_greeting",
            prompt=f"Greet {name} warmly in between three and seven words. Do not use an exclamation mark.",
            scorer=constraint,
            max_new_tokens=20,
            format_scorer=constraint,
        )
    ]


def _short(rng: random.Random, split: str, index: int) -> list[dict[str, Any]]:
    letters = "abcdefghjkmnpqrstuvwxyz"
    word = "".join(rng.choice(letters) for _ in range(7))
    prompt = f"What is the first letter of the string {word.upper()}?"
    answer = word[0]
    scenario = f"{split}_short_{index:04d}"
    return [
        _case(
            split=split,
            family="short_response",
            scenario_id=scenario,
            transformation_id="concise",
            prompt=prompt + " Answer in one or two words.",
            scorer={"type": "exact", "answers": [answer]},
            max_new_tokens=10,
            format_scorer={"type": "exact", "answers": [answer]},
        )
    ]


def _static_fact(
    split: str,
    index: int,
    family: str,
    facts: tuple[tuple[str, str], ...],
    offset: int,
) -> list[dict[str, Any]]:
    prompt, answer = facts[(offset + index) % len(facts)]
    scenario = f"{split}_{family}_{index:04d}"
    return [
        _case(
            split=split,
            family=family,
            scenario_id=scenario,
            transformation_id="fact",
            prompt=prompt + " Answer concisely.",
            scorer={"type": "keywords", "phrase": answer},
            max_new_tokens=16,
            format_scorer={
                "type": "constraints",
                "minimum_words": 1,
                "maximum_words": 8,
            },
        )
    ]


def generate_bank(split: str, seed: int, scenarios_per_family: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    builders: dict[str, Callable[[random.Random, str, int], list[dict[str, Any]]]] = {
        "general_reasoning": _ordering,
        "paraphrase_robustness": _paraphrase,
        "counterfactual_reasoning": _counterfactual,
        "unanswerable_abstention": _unanswerable,
        "long_context_retrieval": _long_context,
        "multi_hop_binding": _binding,
        "mathematics": _arithmetic,
        "grounded_qa_rag": _rag,
        "normal_conversation": _conversation,
        "instruction_following": _instruction,
        "short_response": _short,
    }
    rows: list[dict[str, Any]] = []
    split_offset = {"calibration": 0, "mergedev": 6, "dev_full": 12, "sealed": 24}[split]
    for family, builder in builders.items():
        if family == "long_context_retrieval":
            count = max(1, scenarios_per_family // 3)
        else:
            count = scenarios_per_family
        for index in range(count):
            rows.extend(builder(rng, split, index))
    for index in range(scenarios_per_family):
        rows.extend(_static_fact(split, index, "science", SCIENCE_FACTS, split_offset))
        rows.extend(
            _static_fact(split, index, "common_sense", COMMON_SENSE, split_offset)
        )
    if set(row["family"] for row in rows) != set(FAMILIES):
        missing = set(FAMILIES) - set(row["family"] for row in rows)
        raise AssertionError(f"bank lacks families: {sorted(missing)}")
    case_ids = [row["case_id"] for row in rows]
    if len(case_ids) != len(set(case_ids)):
        raise AssertionError("duplicate case IDs")
    return sorted(rows, key=lambda row: row["case_id"])


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")


def build_banks(
    output_directory: Path,
    sealed_path: Path,
    *,
    specifications: dict[str, tuple[int, int]],
) -> dict[str, Any]:
    output_directory.mkdir(parents=True, exist_ok=True)
    all_rows: dict[str, list[dict[str, Any]]] = {}
    paths: dict[str, Path] = {}
    for split, (seed, scenarios) in specifications.items():
        rows = generate_bank(split, seed, scenarios)
        path = sealed_path if split == "sealed" else output_directory / f"{split}.jsonl"
        write_jsonl(path, rows)
        all_rows[split] = rows
        paths[split] = path

    overlap: dict[str, int] = {}
    splits = list(all_rows)
    for first_index, first in enumerate(splits):
        first_hashes = {row["prompt_sha256"] for row in all_rows[first]}
        for second in splits[first_index + 1 :]:
            second_hashes = {row["prompt_sha256"] for row in all_rows[second]}
            overlap[f"{first}__{second}"] = len(first_hashes & second_hashes)
    if any(overlap.values()):
        raise AssertionError(f"cross-split prompt overlap: {overlap}")
    manifest = {
        "schema_version": "1.0",
        "generator": "procedural_v1",
        "splits": {
            split: {
                "seed": specifications[split][0],
                "scenarios_per_family": specifications[split][1],
                "path": str(paths[split].resolve()) if split != "sealed" else "PRIVATE",
                "sha256": sha256_file(paths[split]),
                "items": len(rows),
                "scenarios": len({row["scenario_id"] for row in rows}),
                "families": dict(sorted(Counter(row["family"] for row in rows).items())),
            }
            for split, rows in all_rows.items()
        },
        "prompt_hash_overlap": overlap,
        "sealed_answers_not_printed": True,
    }
    atomic_write_json(output_directory / "bank_manifest.json", manifest)
    return manifest
