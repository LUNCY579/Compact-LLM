#!/usr/bin/env python3
"""Generate the Rineya Puzzle Reasoning 51K corpus.

All examples are procedurally generated and solved by deterministic symbolic
or graph-search procedures. No public benchmark rows are copied.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import heapq
import itertools
import json
import math
import os
import random
import re
import statistics

import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Sequence

VERSION = "1.0.0"
DEFAULT_SEED = 20260818
SPLIT_COUNTS = {"train": 2720, "validation": 272, "test": 272, "challenge": 136}

NAMES = [
    "Ari", "Bea", "Cato", "Dina", "Eli", "Faye", "Gus", "Hana", "Ivo", "Jia",
    "Kian", "Lena", "Milo", "Nora", "Omar", "Pia", "Quin", "Ravi", "Sora", "Tess",
    "Uma", "Vera", "Wren", "Xena", "Yuri", "Zane", "Asha", "Bram", "Cleo", "Dax",
    "Esme", "Finn", "Gita", "Hugo", "Iris", "Juno", "Kai", "Lila", "Mara", "Niko",
    "Opal", "Pavel", "Rhea", "Sami", "Tara", "Uria", "Vito", "Will", "Xavi", "Yara",
    "Zoya", "Anya", "Ben", "Cora", "Dev", "Evan", "Gia", "Hope", "Ian", "June",
    "Kira", "Leo", "Mina", "Noel", "Orla", "Remy", "Sara", "Theo", "Usha", "Vale",
    "Wade", "Yuna", "Ziv", "Ada", "Bo", "Cyra", "Enzo", "Fia", "Gray", "Indra",
    "Jo", "Ken", "Liv", "Mo", "Nia", "Poe", "Rei", "Sol", "Tia", "Vic",
]
COLORS = ["blue", "red", "green", "yellow", "black", "white", "violet", "orange", "silver", "gold"]
ITEMS = ["apple", "pear", "plum", "orange", "coin", "key", "ring", "marble", "shell", "star", "leaf", "gem"]
PETS = ["cat", "dog", "owl", "fox", "rabbit", "turtle", "parrot", "gecko"]
CITIES = ["Aster", "Birch", "Cedar", "Dover", "Elm", "Frost", "Glen", "Haven"]
PREDICATES = [
    "calm", "bright", "patient", "ready", "kind", "quiet", "swift", "careful",
    "warm", "alert", "steady", "curious", "gentle", "brave", "focused", "tidy",
]
OBJECTS = ["amber token", "blue cube", "copper coin", "green bead", "ivory key", "red disk", "silver ring", "violet card"]
EVENTS = ["briefing", "inspection", "launch", "review", "seminar", "audit", "delivery", "rehearsal", "interview", "workshop"]
LOCATIONS = ["Alder", "Brook", "Cairn", "Delta", "Ember", "Fjord", "Grove", "Harbor", "Iris", "Juniper"]
MALE_NAMES = ["Ari", "Cato", "Eli", "Gus", "Ivo", "Kian", "Milo", "Omar", "Quin", "Ravi", "Zane", "Bram", "Dax", "Finn", "Hugo", "Kai", "Niko", "Pavel", "Sami", "Theo", "Wade", "Xavi", "Ziv", "Ben", "Dev", "Evan", "Ian", "Leo", "Noel", "Remy", "Vic"]
FEMALE_NAMES = ["Bea", "Dina", "Faye", "Hana", "Jia", "Lena", "Nora", "Pia", "Sora", "Tess", "Uma", "Vera", "Xena", "Yara", "Zoya", "Asha", "Cleo", "Esme", "Gita", "Iris", "Juno", "Lila", "Mara", "Opal", "Rhea", "Tara", "Usha", "Yuna", "Ada", "Cyra", "Fia", "Nia"]


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.blake2b(raw, digest_size=8).digest(), "big")


def canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def signature(obj: Any) -> str:
    return hashlib.sha256(canonical(obj).encode("utf-8")).hexdigest()


def unique_permutations(values: Sequence[str]) -> list[tuple[str, ...]]:
    return sorted(set(itertools.permutations(values)))


def oxford(items: Sequence[str]) -> str:
    items = list(items)
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def article(word: str) -> str:
    return "an" if word[0].lower() in "aeiou" else "a"


def cap_first(text: str) -> str:
    return text[:1].upper() + text[1:] if text else text


def count_phrase(n: int, singular: str, plural: str | None = None) -> str:
    """Return a grammatically inflected count phrase."""
    return f"{n} {singular if n == 1 else (plural or singular + 's')}"


def normalize_answer(text: str) -> str:
    text = text.casefold().strip()
    text = text.replace("→", "->")
    text = re.sub(r"[^\w\s\-\>]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def template_index(split: str, rng: random.Random) -> int:
    if split == "train":
        return rng.randrange(0, 3)
    return {"validation": 3, "test": 4, "challenge": 5}[split]


def choose_difficulty(split: str, i: int, rng: random.Random) -> str:
    if split == "challenge":
        return "challenge"
    # Fixed broad ratio while preserving local randomness.
    return "easy" if (i % 20) < 11 else "medium"


def base_row(
    *, family: str, split: str, difficulty: str, template_id: str,
    prompt: str, reasoning_steps: list[str], compact_state: str,
    final_answer: str, accepted_answers: list[str] | None,
    answer_type: str, structure: dict[str, Any], solver: str,
    verification: dict[str, Any], tags: list[str],
) -> dict[str, Any]:
    accepted = accepted_answers or [final_answer]
    if final_answer not in accepted:
        accepted = [final_answer, *accepted]
    return {
        "id": "",  # filled by the orchestrator
        "dataset_version": VERSION,
        "split": split,
        "family": family,
        "difficulty": difficulty,
        "template_id": template_id,
        "prompt": prompt,
        "reasoning_steps": reasoning_steps,
        "reasoning_summary": " ".join(reasoning_steps),
        "compact_state": compact_state,
        "final_answer": final_answer,
        "accepted_answers": sorted(set(accepted)),
        "answer_type": answer_type,
        "answer_normalization": "casefold_whitespace_punctuation",
        "solver": solver,
        "verification": verification,
        "structure": structure,
        "structural_signature": signature({"family": family, "structure": structure}),
        "tags": tags,
    }


# ---------------------------------------------------------------------------
# 1. Dynamic epistemic hat puzzles
# ---------------------------------------------------------------------------

def hat_knows(worlds: Sequence[tuple[str, ...]], world: tuple[str, ...], speaker: int) -> bool:
    visible = world[:speaker]
    local = [w for w in worlds if w[:speaker] == visible]
    return len({w[speaker] for w in local}) == 1


def hat_update(worlds: list[tuple[str, ...]], speaker: int, knows_status: bool) -> list[tuple[str, ...]]:
    return [w for w in worlds if hat_knows(worlds, w, speaker) == knows_status]


def generate_epistemic_hats(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(500):
        if difficulty == "easy":
            n = rng.choice([3, 4])
            num_colors = 2
            max_ann = 2
        elif difficulty == "medium":
            n = rng.choice([4, 5])
            num_colors = 2
            max_ann = 3
        else:
            n = 5
            num_colors = rng.choice([2, 3])
            max_ann = 4
        people = rng.sample(NAMES, n)
        colors = rng.sample(COLORS, num_colors)
        # The stock contains more hats than people, so a person cannot simply
        # infer their own color by subtracting every visible hat from a fully
        # used inventory. This produces genuine public-announcement puzzles.
        stock_total = n + rng.choice([1, 2])
        if num_colors == 2:
            c1 = rng.randint(1, stock_total - 1)
            counts = [c1, stock_total - c1]
        else:
            cuts = sorted(rng.sample(range(1, stock_total), 2))
            counts = [cuts[0], cuts[1] - cuts[0], stock_total - cuts[1]]
            if min(counts) == 0:
                continue
        inventory = [c for c, k in zip(colors, counts) for _ in range(k)]
        stock_counts = collections.Counter(inventory)
        worlds0 = [
            w for w in itertools.product(colors, repeat=n)
            if all(collections.Counter(w)[c] <= stock_counts[c] for c in colors)
        ]
        actual = rng.choice(worlds0)
        current = list(worlds0)
        announcements: list[dict[str, Any]] = []
        speakers = list(range(n - 1, -1, -1))
        for speaker in speakers[:max_ann]:
            status = hat_knows(current, actual, speaker)
            before = len(current)
            nxt = hat_update(current, speaker, status)
            announcements.append({"speaker": speaker, "knows": status, "before": before, "after": len(nxt)})
            current = nxt
            if len(current) <= 1:
                break
        if not announcements or not any(a["after"] < a["before"] for a in announcements):
            continue

        query_options: list[tuple[str, Any, str]] = []
        # Color questions that are forced by public information and not directly stated.
        for target in range(n):
            vals = {w[target] for w in current}
            if len(vals) == 1:
                query_options.append(("color", target, next(iter(vals))))
        # Predict a next knowledge announcement only if public information fixes it.
        used = {a["speaker"] for a in announcements}
        for target in range(n - 1, -1, -1):
            if target in used:
                continue
            statuses = {hat_knows(current, w, target) for w in current}
            if len(statuses) == 1:
                query_options.append(("knowledge", target, next(iter(statuses))))
        if not query_options:
            continue
        qtype, target, qanswer = rng.choice(query_options)
        t = template_index(split, rng)
        counts_text = oxford([f"{inventory.count(c)} {c}" for c in colors])
        order_text = oxford([f"{p} ({'front' if i == 0 else 'back' if i == n-1 else f'position {i+1}'})" for i, p in enumerate(people)])
        statements = []
        for a in announcements:
            who = people[a["speaker"]]
            statements.append(f'{who} says, "I {"know" if a["knows"] else "cannot determine"} my hat color."')
        statement_text = " ".join(statements)
        intros = [
            "A line-of-hats puzzle uses",
            "At a logic club, the available hats are",
            "A group of prisoners knows that the hat inventory is",
            "During an expedition puzzle, the available caps are",
            "A row of robots is assigned hats from",
            "At a logicians' trial, the complete hat inventory is",
        ]
        prompt = (
            f"{intros[t]} {counts_text}. Exactly {n} hats are worn; unused hats may remain. The line from front to back is {order_text}. "
            "Each person sees every hat ahead, sees no hat behind, and cannot see their own hat. "
            f"Everyone knows the complete available stock. Starting from the back, {statement_text} "
        )
        if qtype == "color":
            prompt += f"What color must {people[target]}'s hat be? Give only the color."
            final = str(qanswer).capitalize() + "."
            accepted = [str(qanswer), str(qanswer).capitalize(), final]
            answer_type = "color"
        else:
            prompt += f"Can {people[target]} now determine their own hat color? Answer Yes or No."
            final = "Yes." if qanswer else "No."
            accepted = ["yes" if qanswer else "no", final]
            answer_type = "boolean"

        steps = [f"The inventory permits {len(worlds0)} distinct hat assignments."]
        running = list(worlds0)
        for a in announcements:
            who = people[a["speaker"]]
            phrase = "know" if a["knows"] else "do not know"
            if a["after"] < a["before"]:
                steps.append(f"{who}'s statement that they {phrase} removes inconsistent worlds; possibilities: {a['before']} → {a['after']}.")
            else:
                steps.append(f"{who}'s statement that they {phrase} is compatible with every current world, so {a['after']} remain.")
            running = hat_update(running, a["speaker"], a["knows"])
        if qtype == "color":
            steps.append(f"Every remaining world gives {people[target]} {article(qanswer)} {qanswer} hat.")
            compact = f"Remaining worlds={len(current)}; {people[target]}={qanswer} in all worlds."
        else:
            status_word = "knows" if qanswer else "does not know"
            steps.append(f"Across every remaining world, {people[target]} {status_word} their color from what they see.")
            compact = f"Remaining worlds={len(current)}; knowledge({people[target]})={bool(qanswer)}."
        structure = {
            "people": people,
            "colors": colors,
            "inventory": inventory,
            "actual_world": actual,
            "announcements": [{"speaker": a["speaker"], "knows": a["knows"]} for a in announcements],
            "query": {"type": qtype, "target": target},
        }
        return base_row(
            family="epistemic_hats", split=split, difficulty=difficulty,
            template_id=f"epistemic_hats.t{t}", prompt=prompt,
            reasoning_steps=steps, compact_state=compact, final_answer=final,
            accepted_answers=accepted, answer_type=answer_type, structure=structure,
            solver="public_announcement_world_elimination",
            verification={"initial_worlds": len(worlds0), "remaining_worlds": len(current), "forced": True},
            tags=["epistemic-logic", "theory-of-mind", "hats", "world-elimination"],
        )
    raise RuntimeError("Failed to generate epistemic_hats")


# ---------------------------------------------------------------------------
# 2. Knights and knaves
# ---------------------------------------------------------------------------

def eval_knight_statement(stmt: dict[str, Any], assignment: tuple[bool, ...]) -> bool:
    kind = stmt["kind"]
    a = stmt.get("a")
    b = stmt.get("b")
    if kind == "is_knight": return assignment[a]
    if kind == "is_knave": return not assignment[a]
    if kind == "same": return assignment[a] == assignment[b]
    if kind == "different": return assignment[a] != assignment[b]
    if kind == "both_knights": return assignment[a] and assignment[b]
    if kind == "both_knaves": return (not assignment[a]) and (not assignment[b])
    if kind == "at_least_one_knight": return assignment[a] or assignment[b]
    if kind == "exactly_one_knight": return assignment[a] ^ assignment[b]
    if kind == "all_same":
        inds = stmt["inds"]
        return len({assignment[i] for i in inds}) == 1
    if kind == "exactly_k_knights":
        return sum(assignment[i] for i in stmt["inds"]) == stmt["k"]
    raise ValueError(kind)


def knight_statement_text(stmt: dict[str, Any], people: Sequence[str]) -> str:
    k = stmt["kind"]
    if k == "is_knight": return f"{people[stmt['a']]} is a knight"
    if k == "is_knave": return f"{people[stmt['a']]} is a knave"
    if k == "same": return f"{people[stmt['a']]} and {people[stmt['b']]} are the same type"
    if k == "different": return f"{people[stmt['a']]} and {people[stmt['b']]} are different types"
    if k == "both_knights": return f"{people[stmt['a']]} and {people[stmt['b']]} are both knights"
    if k == "both_knaves": return f"{people[stmt['a']]} and {people[stmt['b']]} are both knaves"
    if k == "at_least_one_knight": return f"at least one of {people[stmt['a']]} and {people[stmt['b']]} is a knight"
    if k == "exactly_one_knight": return f"exactly one of {people[stmt['a']]} and {people[stmt['b']]} is a knight"
    if k == "all_same": return f"{oxford([people[i] for i in stmt['inds']])} are all the same type"
    if k == "exactly_k_knights": return f"exactly {stmt['k']} of {oxford([people[i] for i in stmt['inds']])} are knights"
    raise ValueError(k)


def generate_knights_knaves(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(500):
        n = rng.choice([3, 4]) if difficulty == "easy" else (rng.choice([4, 5]) if difficulty == "medium" else 5)
        people = rng.sample(NAMES, n)
        hidden = tuple(rng.choice([False, True]) for _ in range(n))
        if all(hidden) or not any(hidden):
            continue
        all_assignments = list(itertools.product([False, True], repeat=n))
        pool: list[dict[str, Any]] = []
        for speaker in range(n):
            for a in range(n):
                pool.extend([
                    {"speaker": speaker, "kind": "is_knight", "a": a},
                    {"speaker": speaker, "kind": "is_knave", "a": a},
                ])
            for a, b in itertools.combinations(range(n), 2):
                for kind in ["same", "different", "both_knights", "both_knaves", "at_least_one_knight", "exactly_one_knight"]:
                    pool.append({"speaker": speaker, "kind": kind, "a": a, "b": b})
            if difficulty == "challenge":
                inds = sorted(rng.sample(range(n), rng.choice([3, 4])))
                pool.append({"speaker": speaker, "kind": "all_same", "inds": inds})
                for k in range(1, len(inds)):
                    pool.append({"speaker": speaker, "kind": "exactly_k_knights", "inds": inds, "k": k})
        # Keep only statements whose truth matches the speaker's hidden type.
        pool = [s for s in pool if eval_knight_statement(s, hidden) == hidden[s["speaker"]]]
        rng.shuffle(pool)
        candidates = list(all_assignments)
        chosen: list[dict[str, Any]] = []
        used_text = set()
        used_speakers = collections.Counter()
        trace = []
        for stmt in pool:
            txt = knight_statement_text(stmt, people)
            if txt in used_text or used_speakers[stmt["speaker"]] >= (2 if difficulty == "challenge" else 1):
                continue
            nxt = [a for a in candidates if (eval_knight_statement(stmt, a) == a[stmt["speaker"]])]
            if len(nxt) < len(candidates):
                trace.append((len(candidates), len(nxt)))
                candidates = nxt
                chosen.append(stmt)
                used_text.add(txt)
                used_speakers[stmt["speaker"]] += 1
            if len(candidates) == 1:
                break
        if len(candidates) != 1 or candidates[0] != hidden or len(chosen) < 2:
            continue
        target = rng.randrange(n)
        t = template_index(split, rng)
        intros = [
            "On an island, knights always tell the truth and knaves always lie.",
            "In a logic village, every resident is either a truthful knight or a lying knave.",
            "A puzzle tribunal contains only knights, who speak truth, and knaves, who speak falsely.",
            "At an expedition camp, each speaker is either always truthful or always deceptive.",
            "Among these robots, truth-units always speak truly and lie-units always speak falsely.",
            "In a formal logic trial, every witness is exactly one of two types: truthful knight or lying knave.",
        ]
        sayings = " ".join(f'{people[s["speaker"]]} says, "{cap_first(knight_statement_text(s, people))}."' for s in chosen)
        prompt = f"{intros[t]} {sayings} Is {people[target]} a knight or a knave? Give only the type."
        steps = [f"There are {2**n} possible type assignments."]
        for stmt, (before, after) in zip(chosen, trace):
            who = people[stmt["speaker"]]
            text = knight_statement_text(stmt, people)
            steps.append(f"Requiring {who}'s type to match the truth value of “{text}”; assignments: {before} → {after}.")
        assignment_text = ", ".join(f"{p}={'knight' if v else 'knave'}" for p, v in zip(people, hidden))
        steps.append(f"The only consistent assignment is {assignment_text}.")
        final = "Knight." if hidden[target] else "Knave."
        compact = f"Unique assignment: {assignment_text}."
        structure = {"people": people, "statements": chosen, "query_target": target}
        return base_row(
            family="knights_knaves", split=split, difficulty=difficulty,
            template_id=f"knights_knaves.t{t}", prompt=prompt,
            reasoning_steps=steps, compact_state=compact, final_answer=final,
            accepted_answers=[final, final.rstrip(".").lower()], answer_type="binary_class",
            structure=structure, solver="truth_assignment_enumeration",
            verification={"candidate_assignments": 2**n, "solution_count": 1},
            tags=["truth-liar", "consistency", "deduction", "knights-and-knaves"],
        )
    raise RuntimeError("Failed to generate knights_knaves")


# ---------------------------------------------------------------------------
# 3. Prisoner/suspect alibi-count puzzles
# ---------------------------------------------------------------------------

def eval_alibi(stmt: dict[str, Any], culprit: int) -> bool:
    k = stmt["kind"]
    if k == "culprit": return culprit == stmt["a"]
    if k == "innocent": return culprit != stmt["a"]
    if k == "either": return culprit in (stmt["a"], stmt["b"])
    if k == "neither": return culprit not in (stmt["a"], stmt["b"])
    if k == "one_of_group": return culprit in stmt["inds"]
    raise ValueError(k)


def alibi_text(stmt: dict[str, Any], people: Sequence[str]) -> str:
    k = stmt["kind"]
    if k == "culprit": return f"{people[stmt['a']]} did it"
    if k == "innocent": return f"{people[stmt['a']]} is innocent"
    if k == "either": return f"the culprit is either {people[stmt['a']]} or {people[stmt['b']]}"
    if k == "neither": return f"neither {people[stmt['a']]} nor {people[stmt['b']]} is the culprit"
    if k == "one_of_group": return f"the culprit is one of {oxford([people[i] for i in stmt['inds']])}"
    raise ValueError(k)


def generate_prisoner_alibis(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(500):
        n = rng.choice([3, 4]) if difficulty == "easy" else (rng.choice([4, 5]) if difficulty == "medium" else rng.choice([5, 6]))
        people = rng.sample(NAMES, n)
        m = n if difficulty != "challenge" else n + 1
        statements: list[dict[str, Any]] = []
        for speaker in range(m):
            kind_pool = ["culprit", "innocent", "either", "neither"]
            if difficulty == "challenge": kind_pool.append("one_of_group")
            kind = rng.choice(kind_pool)
            if kind in ("culprit", "innocent"):
                stmt = {"speaker": speaker % n, "kind": kind, "a": rng.randrange(n)}
            elif kind in ("either", "neither"):
                a, b = rng.sample(range(n), 2)
                stmt = {"speaker": speaker % n, "kind": kind, "a": a, "b": b}
            else:
                inds = sorted(rng.sample(range(n), rng.choice([2, 3])))
                stmt = {"speaker": speaker % n, "kind": kind, "inds": inds}
            statements.append(stmt)
        counts = [sum(eval_alibi(s, c) for s in statements) for c in range(n)]
        unique_pairs = [(c, counts[c]) for c in range(n) if counts.count(counts[c]) == 1]
        if not unique_pairs:
            continue
        culprit, k_true = rng.choice(unique_pairs)
        if k_true in (0, len(statements)):
            continue
        t = template_index(split, rng)
        intros = [
            "Exactly one of the following suspects is responsible.",
            "Only one prisoner hid the key.",
            "A single contestant changed the answer sheet.",
            "Exactly one expedition member moved the marker.",
            "Only one robot opened the sealed hatch.",
            "Exactly one witness caused the alarm, and the truth-count condition is exact.",
        ]
        sayings = " ".join(f'{people[s["speaker"]]} says, "{cap_first(alibi_text(s, people))}."' for s in statements)
        prompt = f"{intros[t]} {sayings} Exactly {k_true} of these {len(statements)} statements {'is' if k_true == 1 else 'are'} true. Who is responsible? Give only the name."
        steps = []
        for c in range(n):
            steps.append(f"If {people[c]} were responsible, {count_phrase(counts[c], 'statement')} would be true.")
        steps.append(f"Only {people[culprit]} produces exactly {count_phrase(k_true, 'true statement')}.")
        final = people[culprit] + "."
        compact = "; ".join(f"{people[c]}→{counts[c]} true" for c in range(n))
        structure = {"people": people, "statements": statements, "required_true": k_true}
        return base_row(
            family="prisoner_alibis", split=split, difficulty=difficulty,
            template_id=f"prisoner_alibis.t{t}", prompt=prompt,
            reasoning_steps=steps, compact_state=compact, final_answer=final,
            accepted_answers=[people[culprit], final], answer_type="entity",
            structure=structure, solver="culprit_truth_count_enumeration",
            verification={"candidate_culprits": n, "solution_count": 1, "true_statement_count": k_true},
            tags=["prisoners", "alibis", "truth-count", "case-analysis"],
        )
    raise RuntimeError("Failed to generate prisoner_alibis")


# ---------------------------------------------------------------------------
# 4. Mislabeled containers
# ---------------------------------------------------------------------------

def generate_mislabeled_boxes(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(500):
        n = rng.choice([3, 4]) if difficulty == "easy" else (rng.choice([4, 5]) if difficulty == "medium" else 5)
        box_names = [chr(ord("A") + i) for i in range(n)]
        contents = rng.sample(ITEMS, n)
        labels = list(contents)
        rng.shuffle(labels)
        correctness = 0 if difficulty != "challenge" else rng.choice([0, 1])
        perms = [p for p in itertools.permutations(contents) if sum(p[i] == labels[i] for i in range(n)) == correctness]
        if not perms:
            continue
        hidden = rng.choice(perms)
        candidates = list(perms)
        observations: list[dict[str, Any]] = []
        target = rng.randrange(n)
        # Add observations derived from hidden until target content is forced.
        for _step in range(3 if difficulty != "challenge" else 4):
            if len({p[target] for p in candidates}) == 1:
                break
            b = rng.choice([i for i in range(n) if i != target])
            if rng.random() < 0.65:
                obs = {"kind": "is", "box": b, "item": hidden[b]}
                nxt = [p for p in candidates if p[b] == hidden[b]]
            else:
                wrongs = [x for x in contents if x != hidden[b]]
                item = rng.choice(wrongs)
                obs = {"kind": "not", "box": b, "item": item}
                nxt = [p for p in candidates if p[b] != item]
            if len(nxt) < len(candidates):
                observations.append({**obs, "before": len(candidates), "after": len(nxt)})
                candidates = nxt
        if len({p[target] for p in candidates}) != 1 or not observations:
            continue
        answer = next(iter({p[target] for p in candidates}))
        t = template_index(split, rng)
        intros = [
            "Each sealed box contains one distinct item.",
            "A puzzle shelf has one item in each container.",
            "Each evidence locker holds exactly one different object.",
            "Each expedition crate contains one unique supply.",
            "Each robot bin stores one distinct component.",
            "Each vault drawer contains one unique object.",
        ]
        label_text = "; ".join(f"Box {box_names[i]} is labeled “{labels[i]}”" for i in range(n))
        condition = "Every label is wrong." if correctness == 0 else "Exactly one label is correct."
        obs_texts = []
        for o in observations:
            if o["kind"] == "is":
                obs_texts.append(f"Inspection shows Box {box_names[o['box']]} contains the {o['item']}")
            else:
                obs_texts.append(f"A scan shows Box {box_names[o['box']]} does not contain the {o['item']}")
        prompt = f"{intros[t]} {label_text}. {condition} {'; '.join(obs_texts)}. What is in Box {box_names[target]}? Give only the item."
        steps = [f"The label rule permits {len(perms)} content assignments."]
        for o in observations:
            desc = f"Box {box_names[o['box']]} {'contains' if o['kind']=='is' else 'does not contain'} {o['item']}"
            steps.append(f"Applying “{desc}”; assignments: {o['before']} → {o['after']}.")
        steps.append(f"Every remaining assignment puts the {answer} in Box {box_names[target]}.")
        compact = f"Candidates={len(candidates)}; Box {box_names[target]}={answer}."
        structure = {"boxes": box_names, "contents": contents, "labels": labels, "correct_labels": correctness, "observations": [{k:v for k,v in o.items() if k not in ('before','after')} for o in observations], "query_box": target}
        return base_row(
            family="mislabeled_boxes", split=split, difficulty=difficulty,
            template_id=f"mislabeled_boxes.t{t}", prompt=prompt,
            reasoning_steps=steps, compact_state=compact, final_answer=answer.capitalize()+".",
            accepted_answers=[answer, answer.capitalize(), answer.capitalize()+"."], answer_type="item",
            structure=structure, solver="permutation_constraint_elimination",
            verification={"initial_candidates": len(perms), "remaining_candidates": len(candidates), "forced": True},
            tags=["mislabeled-boxes", "permutations", "constraint-elimination"],
        )
    raise RuntimeError("Failed to generate mislabeled_boxes")

# ---------------------------------------------------------------------------
# 5. Logic-grid constraint puzzles
# ---------------------------------------------------------------------------

def eval_grid_clue(clue: dict[str, Any], candidate: tuple[tuple[str, ...], tuple[str, ...]], people: Sequence[str]) -> bool:
    pet_perm, city_perm = candidate
    k = clue["kind"]
    if k == "person_pet": return pet_perm[clue["person"]] == clue["pet"]
    if k == "person_not_pet": return pet_perm[clue["person"]] != clue["pet"]
    if k == "person_city": return city_perm[clue["person"]] == clue["city"]
    if k == "person_not_city": return city_perm[clue["person"]] != clue["city"]
    if k == "pet_city":
        i = pet_perm.index(clue["pet"])
        return city_perm[i] == clue["city"]
    if k == "pet_not_city":
        i = pet_perm.index(clue["pet"])
        return city_perm[i] != clue["city"]
    raise ValueError(k)


def grid_clue_text(clue: dict[str, Any], people: Sequence[str]) -> str:
    k = clue["kind"]
    if k == "person_pet": return f"{people[clue['person']]} owns the {clue['pet']}"
    if k == "person_not_pet": return f"{people[clue['person']]} does not own the {clue['pet']}"
    if k == "person_city": return f"{people[clue['person']]} lives in {clue['city']}"
    if k == "person_not_city": return f"{people[clue['person']]} does not live in {clue['city']}"
    if k == "pet_city": return f"the {clue['pet']} owner lives in {clue['city']}"
    if k == "pet_not_city": return f"the {clue['pet']} owner does not live in {clue['city']}"
    raise ValueError(k)


def generate_logic_grid(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(500):
        n = 3 if difficulty == "easy" else 4
        people = rng.sample(NAMES, n)
        pets = rng.sample(PETS, n)
        cities = rng.sample(CITIES, n)
        hidden_pet = tuple(rng.sample(pets, n))
        hidden_city = tuple(rng.sample(cities, n))
        hidden = (hidden_pet, hidden_city)
        all_candidates = [(p, c) for p in itertools.permutations(pets) for c in itertools.permutations(cities)]
        pool: list[dict[str, Any]] = []
        for i in range(n):
            pool.append({"kind": "person_pet", "person": i, "pet": hidden_pet[i]})
            pool.append({"kind": "person_city", "person": i, "city": hidden_city[i]})
            for pet in pets:
                if pet != hidden_pet[i]: pool.append({"kind": "person_not_pet", "person": i, "pet": pet})
            for city in cities:
                if city != hidden_city[i]: pool.append({"kind": "person_not_city", "person": i, "city": city})
        for pet in pets:
            i = hidden_pet.index(pet)
            pool.append({"kind": "pet_city", "pet": pet, "city": hidden_city[i]})
            for city in cities:
                if city != hidden_city[i]: pool.append({"kind": "pet_not_city", "pet": pet, "city": city})
        rng.shuffle(pool)
        # Prefer cross-category and negative clues; direct clues remain available.
        pool.sort(key=lambda c: (0 if c["kind"] in ("pet_city", "pet_not_city") else 1 if "not" in c["kind"] else 2, rng.random()))
        candidates = list(all_candidates)
        chosen: list[dict[str, Any]] = []
        trace: list[tuple[int, int]] = []
        direct_count = 0
        for clue in pool:
            if clue["kind"] in ("person_pet", "person_city") and direct_count >= (1 if difficulty != "challenge" else 2):
                continue
            nxt = [c for c in candidates if eval_grid_clue(clue, c, people)]
            if 0 < len(nxt) < len(candidates):
                chosen.append(clue)
                trace.append((len(candidates), len(nxt)))
                candidates = nxt
                if clue["kind"] in ("person_pet", "person_city"):
                    direct_count += 1
            if len(candidates) == 1:
                break
        if len(candidates) != 1 or candidates[0] != hidden or len(chosen) < 4:
            continue
        qkind = rng.choice(["pet_of_person", "city_of_person", "owner_of_pet", "city_of_pet"])
        if qkind == "pet_of_person":
            idx = rng.randrange(n); answer = hidden_pet[idx]
            question = f"What pet does {people[idx]} own?"
            answer_type = "item"
            query = {"kind": qkind, "person": idx}
        elif qkind == "city_of_person":
            idx = rng.randrange(n); answer = hidden_city[idx]
            question = f"Where does {people[idx]} live?"
            answer_type = "location"
            query = {"kind": qkind, "person": idx}
        elif qkind == "owner_of_pet":
            pet = rng.choice(pets); idx = hidden_pet.index(pet); answer = people[idx]
            question = f"Who owns the {pet}?"
            answer_type = "entity"
            query = {"kind": qkind, "pet": pet}
        else:
            pet = rng.choice(pets); idx = hidden_pet.index(pet); answer = hidden_city[idx]
            question = f"In which city does the {pet} owner live?"
            answer_type = "location"
            query = {"kind": qkind, "pet": pet}
        t = template_index(split, rng)
        intros = [
            "Match each person to one pet and one city; every pet and city is used once.",
            "A logic grid pairs each resident with a distinct animal and hometown.",
            "Each researcher has one different pet and lives in one different city.",
            "Each expedition member has a unique companion animal and base city.",
            "Each service robot is assigned a unique mascot and station city.",
            "Solve the bijective logic grid: one person per pet and one person per city.",
        ]
        clues_text = " ".join(f"{i+1}. {cap_first(grid_clue_text(c, people))}." for i, c in enumerate(chosen))
        prompt = f"{intros[t]} People: {oxford(people)}. Pets: {oxford(pets)}. Cities: {oxford(cities)}. Clues: {clues_text} {question} Give only the answer."
        steps = [f"There are {len(all_candidates)} possible person–pet–city matchings."]
        for clue, (before, after) in zip(chosen, trace):
            steps.append(f"Applying “{grid_clue_text(clue, people)}”; matchings: {before} → {after}.")
        solved = "; ".join(f"{people[i]}={hidden_pet[i]}/{hidden_city[i]}" for i in range(n))
        steps.append(f"The unique grid is {solved}.")
        structure = {"people": people, "pets": pets, "cities": cities, "clues": chosen, "query": query}
        return base_row(
            family="logic_grid", split=split, difficulty=difficulty,
            template_id=f"logic_grid.t{t}", prompt=prompt,
            reasoning_steps=steps, compact_state=solved, final_answer=answer.capitalize()+".",
            accepted_answers=[answer, answer.capitalize(), answer.capitalize()+"."], answer_type=answer_type,
            structure=structure, solver="bijective_csp_enumeration",
            verification={"initial_candidates": len(all_candidates), "solution_count": 1},
            tags=["logic-grid", "constraint-satisfaction", "bijection", "deduction"],
        )
    raise RuntimeError("Failed to generate logic_grid")


# ---------------------------------------------------------------------------
# 6. Linear ordering and ranking
# ---------------------------------------------------------------------------
_ORDER_CACHE: dict[int, list[tuple[int, ...]]] = {}

def order_perms(n: int) -> list[tuple[int, ...]]:
    if n not in _ORDER_CACHE:
        _ORDER_CACHE[n] = list(itertools.permutations(range(n)))
    return _ORDER_CACHE[n]


def eval_order_clue(clue: dict[str, Any], perm: tuple[int, ...]) -> bool:
    pos = {x: i for i, x in enumerate(perm)}
    k = clue["kind"]
    a = clue.get("a"); b = clue.get("b")
    if k == "before": return pos[a] < pos[b]
    if k == "immediately_before": return pos[b] == pos[a] + 1
    if k == "gap_before": return pos[b] == pos[a] + clue["gap"]
    if k == "adjacent": return abs(pos[a] - pos[b]) == 1
    if k == "not_adjacent": return abs(pos[a] - pos[b]) != 1
    if k == "at_end": return pos[a] in (0, len(perm)-1)
    if k == "not_at_end": return pos[a] not in (0, len(perm)-1)
    if k == "position": return pos[a] == clue["position"]
    raise ValueError(k)


def order_clue_text(clue: dict[str, Any], names: Sequence[str], context: str) -> str:
    k = clue["kind"]; a = names[clue.get("a", 0)]; b = names[clue.get("b", 0)]
    if k == "before": return f"{a} is before {b}"
    if k == "immediately_before": return f"{a} is immediately before {b}"
    if k == "gap_before": return f"{a} is exactly {clue['gap']} positions before {b}"
    if k == "adjacent": return f"{a} is adjacent to {b}"
    if k == "not_adjacent": return f"{a} is not adjacent to {b}"
    if k == "at_end": return f"{a} is at one end"
    if k == "not_at_end": return f"{a} is not at either end"
    if k == "position": return f"{a} is in position {clue['position']+1}"
    raise ValueError(k)


def generate_linear_order(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(500):
        n = rng.choice([4, 5]) if difficulty == "easy" else (rng.choice([5, 6]) if difficulty == "medium" else 6)
        names = rng.sample(NAMES, n)
        hidden = tuple(rng.sample(range(n), n))
        pos = {x: i for i, x in enumerate(hidden)}
        pool: list[dict[str, Any]] = []
        for a, b in itertools.permutations(range(n), 2):
            if pos[a] < pos[b]: pool.append({"kind": "before", "a": a, "b": b})
            if pos[b] == pos[a] + 1: pool.append({"kind": "immediately_before", "a": a, "b": b})
            gap = pos[b] - pos[a]
            if gap >= 2: pool.append({"kind": "gap_before", "a": a, "b": b, "gap": gap})
        for a, b in itertools.combinations(range(n), 2):
            if abs(pos[a]-pos[b]) == 1: pool.append({"kind": "adjacent", "a": a, "b": b})
            else: pool.append({"kind": "not_adjacent", "a": a, "b": b})
        for a in range(n):
            pool.append({"kind": "at_end" if pos[a] in (0,n-1) else "not_at_end", "a": a})
            pool.append({"kind": "position", "a": a, "position": pos[a]})
        rng.shuffle(pool)
        pool.sort(key=lambda c: (0 if c["kind"] in ("immediately_before", "gap_before") else 1 if c["kind"] in ("adjacent", "before") else 2, rng.random()))
        candidates = list(order_perms(n))
        chosen=[]; trace=[]; direct=0
        for clue in pool:
            if clue["kind"] == "position" and direct >= 1:
                continue
            nxt = [p for p in candidates if eval_order_clue(clue, p)]
            if 0 < len(nxt) < len(candidates):
                chosen.append(clue); trace.append((len(candidates),len(nxt))); candidates=nxt
                if clue["kind"] == "position": direct += 1
            if len(candidates)==1: break
        if len(candidates)!=1 or candidates[0]!=hidden or len(chosen)<3:
            continue
        qkind = rng.choice(["position", "who_at", "first", "last"])
        if qkind == "position":
            x=rng.randrange(n); answer=str(pos[x]+1); question=f"What position is {names[x]} in?"
            answer_type="integer"; query={"kind":qkind,"entity":x}
        elif qkind == "who_at":
            p=rng.randrange(n); answer=names[hidden[p]]; question=f"Who is in position {p+1}?"
            answer_type="entity"; query={"kind":qkind,"position":p}
        elif qkind == "first":
            answer=names[hidden[0]]; question="Who is first?"; answer_type="entity"; query={"kind":qkind}
        else:
            answer=names[hidden[-1]]; question="Who is last?"; answer_type="entity"; query={"kind":qkind}
        t=template_index(split,rng)
        intros=[
            "Arrange the speakers from first to last.",
            "Five runners finish in a strict order with no ties." if n==5 else "The runners finish in a strict order with no ties.",
            "A set of presentations is scheduled in distinct positions.",
            "Expedition teams line up in a single fixed order.",
            "Robots occupy distinct slots from left to right.",
            "Determine the unique total order satisfying every constraint.",
        ]
        clue_text=" ".join(f"{i+1}. {cap_first(order_clue_text(c,names,'order'))}." for i,c in enumerate(chosen))
        prompt=f"{intros[t]} Participants: {oxford(names)}. Clues: {clue_text} {question} Give only the answer."
        steps=[f"There are {math.factorial(n)} possible total orders."]
        for c,(before,after) in zip(chosen,trace):
            steps.append(f"Applying “{order_clue_text(c,names,'order')}”; orders: {before} → {after}.")
        solved=" → ".join(names[i] for i in hidden)
        steps.append(f"The unique order is {solved}.")
        final=answer+"."
        structure={"names":names,"clues":chosen,"query":query}
        return base_row(
            family="linear_order", split=split, difficulty=difficulty,
            template_id=f"linear_order.t{t}", prompt=prompt,
            reasoning_steps=steps, compact_state=solved, final_answer=final,
            accepted_answers=[answer,final], answer_type=answer_type, structure=structure,
            solver="permutation_order_csp",
            verification={"initial_candidates":math.factorial(n),"solution_count":1},
            tags=["ordering","ranking","constraint-satisfaction","deduction"],
        )
    raise RuntimeError("Failed to generate linear_order")


# ---------------------------------------------------------------------------
# 7. Controlled natural-language rule deduction
# ---------------------------------------------------------------------------
Literal = tuple[int, bool]  # (predicate index, polarity)
Rule = tuple[tuple[Literal, ...], Literal]


def rule_closure(facts: dict[tuple[int,int,bool], list[str]], rules: list[Rule], n_entities: int) -> tuple[set[tuple[int,int,bool]], dict[tuple[int,int,bool], list[str]], dict[tuple[int,int,bool], int]]:
    known=set(facts)
    proofs={k:list(v) for k,v in facts.items()}
    depths={k:0 for k in facts}
    changed=True
    while changed:
        changed=False
        for premises, conclusion in rules:
            cp, cpol=conclusion
            for e in range(n_entities):
                prem_keys=[(e,p,pol) for p,pol in premises]
                if all(k in known for k in prem_keys):
                    key=(e,cp,cpol)
                    opposite=(e,cp,not cpol)
                    if opposite in known:
                        continue
                    if key not in known:
                        known.add(key); changed=True
                        depths[key]=1+max((depths[k] for k in prem_keys),default=0)
                        proofs[key]=[f"premise:{k[1]}:{int(k[2])}" for k in prem_keys]
    return known,proofs,depths


def literal_text(lit: Literal, predicates: Sequence[str]) -> str:
    p,pol=lit
    return predicates[p] if pol else f"not {predicates[p]}"


def rule_text(rule: Rule, predicates: Sequence[str]) -> str:
    premises,conclusion=rule
    ps=[literal_text(x,predicates) for x in premises]
    return f"If something is {oxford(ps)}, then it is {literal_text(conclusion,predicates)}"


def generate_rule_deduction(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(500):
        n_entities=rng.choice([3,4]) if difficulty=="easy" else (rng.choice([4,5]) if difficulty=="medium" else 5)
        entities=rng.sample(NAMES,n_entities)
        n_pred=7 if difficulty=="easy" else (10 if difficulty=="medium" else 12)
        predicates=rng.sample(PREDICATES,n_pred)
        # DAG-like rules: premises use earlier predicates, conclusions later predicates.
        rules: list[Rule]=[]
        num_rules=5 if difficulty=="easy" else (8 if difficulty=="medium" else 11)
        for _r in range(num_rules):
            c=rng.randrange(2,n_pred)
            arity=1 if rng.random()<0.65 else 2
            prem_ids=sorted(rng.sample(range(c),arity))
            premises=tuple((p,True if rng.random()<0.9 else False) for p in prem_ids)
            conclusion=(c,True if rng.random()<0.85 else False)
            rules.append((premises,conclusion))
        facts: dict[tuple[int,int,bool],list[str]]={}
        for e in range(n_entities):
            for p in rng.sample(range(min(4,n_pred)),rng.choice([1,2])):
                pol=True if rng.random()<0.85 else False
                if (e,p,not pol) not in facts:
                    facts[(e,p,pol)]=["given"]
        known,proofs,depths=rule_closure(facts,rules,n_entities)
        # Reject contradictions and select a balanced label target.
        if any((e,p,True) in known and (e,p,False) in known for e in range(n_entities) for p in range(n_pred)):
            continue
        label=rng.choice(["yes","no","unknown"])
        if label=="yes":
            opts=[k for k,d in depths.items() if d>=1 and (difficulty!="medium" or d>=2)]
            if not opts: continue
            e,p,pol=rng.choice(opts); query=(e,p,pol)
        elif label=="no":
            opts=[k for k,d in depths.items() if d>=1]
            if not opts: continue
            e,p,derived_pol=rng.choice(opts); query=(e,p,not derived_pol)
            if query in known: continue
        else:
            opts=[]
            for e in range(n_entities):
                for p in range(n_pred):
                    for pol in (True,False):
                        if (e,p,pol) not in known and (e,p,not pol) not in known:
                            opts.append((e,p,pol))
            if not opts: continue
            query=rng.choice(opts)
        e,p,pol=query
        t=template_index(split,rng)
        intros=[
            "Use only the stated facts and rules.",
            "A small rule system follows; do not assume unstated facts.",
            "Reason under an open-world rule set.",
            "An expedition database contains only the following facts and implications.",
            "A robot knowledge base uses the following exact rules.",
            "Apply forward chaining and distinguish proved, disproved, and unknown claims.",
        ]
        fact_sentences=[]
        for (ee,pp,polarity) in sorted(facts):
            fact_sentences.append(f"{entities[ee]} is {'' if polarity else 'not '}{predicates[pp]}")
        rule_sentences=[rule_text(r,predicates) for r in rules]
        query_text=f"{entities[e]} is {'' if pol else 'not '}{predicates[p]}"
        prompt=(f"{intros[t]} Facts: "+"; ".join(fact_sentences)+". Rules: "+"; ".join(rule_sentences)+
                f". Claim: {query_text}. Answer Yes if the claim is forced, No if its negation is forced, otherwise Unknown.")
        # Build a concrete proof chain by replaying rule derivations until query/opposite appears.
        steps=["Begin with the explicit facts and repeatedly apply rules whose premises are known."]
        target_key=query if query in known else (e,p,not pol) if (e,p,not pol) in known else None
        if target_key is not None:
            # Reconstruct a concise derivation by scanning levels.
            for depth in range(1,depths.get(target_key,0)+1):
                derived_at=[k for k,d in depths.items() if d==depth and k[0]==e]
                if derived_at:
                    desc=", ".join(f"{entities[k[0]]} is {'' if k[2] else 'not '}{predicates[k[1]]}" for k in derived_at[:3])
                    steps.append(f"At inference depth {depth}, derive {desc}.")
        if label=="yes":
            steps.append(f"The claim “{query_text}” is derived, so the answer is Yes.")
            final="Yes."
        elif label=="no":
            opp=f"{entities[e]} is {'' if not pol else 'not '}{predicates[p]}"
            steps.append(f"The opposite claim “{opp}” is derived, so the answer is No.")
            final="No."
        else:
            steps.append(f"Neither “{query_text}” nor its negation is derivable, so the answer is Unknown.")
            final="Unknown."
        known_for_e=sorted((predicates[pp] if po else f"not {predicates[pp]}") for ee,pp,po in known if ee==e)
        compact=f"Closure for {entities[e]}: {', '.join(known_for_e)}."
        structure={
            "entities":entities,"predicates":predicates,
            "facts":[{"entity":ee,"predicate":pp,"polarity":po} for ee,pp,po in sorted(facts)],
            "rules":[{"premises":[{"predicate":pp,"polarity":po} for pp,po in pre],"conclusion":{"predicate":co[0],"polarity":co[1]}} for pre,co in rules],
            "query":{"entity":e,"predicate":p,"polarity":pol},
        }
        return base_row(
            family="rule_deduction", split=split, difficulty=difficulty,
            template_id=f"rule_deduction.t{t}", prompt=prompt,
            reasoning_steps=steps, compact_state=compact, final_answer=final,
            accepted_answers=[final,final.rstrip(".").lower()], answer_type="three_way_logic",
            structure=structure, solver="forward_chaining_open_world",
            verification={"closure_size":len(known),"query_label":label,"proof_depth":depths.get(target_key,0) if target_key else None},
            tags=["rule-deduction","forward-chaining","open-world","proof"],
        )
    raise RuntimeError("Failed to generate rule_deduction")

# ---------------------------------------------------------------------------
# 8. Object tracking under transformations
# ---------------------------------------------------------------------------

def apply_tracking_op(state: list[str], op: dict[str, Any]) -> list[str]:
    s=list(state); k=op["kind"]
    if k=="swap":
        i,j=op["i"],op["j"]; s[i],s[j]=s[j],s[i]
    elif k=="reverse":
        i,j=op["i"],op["j"]; s[i:j+1]=reversed(s[i:j+1])
    elif k=="rotate_left":
        q=op["amount"]%len(s); s=s[q:]+s[:q]
    elif k=="rotate_right":
        q=op["amount"]%len(s); s=s[-q:]+s[:-q] if q else s
    elif k=="move":
        i,j=op["i"],op["j"]; x=s.pop(i); s.insert(j,x)
    else: raise ValueError(k)
    return s


def tracking_op_text(op: dict[str, Any]) -> str:
    k=op["kind"]
    if k=="swap": return f"swap positions {op['i']+1} and {op['j']+1}"
    if k=="reverse": return f"reverse positions {op['i']+1} through {op['j']+1}"
    if k=="rotate_left": return f"rotate the whole row left by {op['amount']}"
    if k=="rotate_right": return f"rotate the whole row right by {op['amount']}"
    if k=="move": return f"move the object in position {op['i']+1} to position {op['j']+1}, shifting intervening objects"
    raise ValueError(k)


def generate_object_tracking(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    n=rng.choice([4,5]) if difficulty=="easy" else (rng.choice([5,6,7]) if difficulty=="medium" else 8)
    objects=rng.sample(OBJECTS,n)
    state=list(rng.sample(objects,n)); initial=list(state)
    n_ops=rng.randint(3,5) if difficulty=="easy" else (rng.randint(5,8) if difficulty=="medium" else rng.randint(8,11))
    ops=[]; states=[]
    for _ in range(n_ops):
        kinds=["swap","move","reverse"]
        if n>=5: kinds += ["rotate_left","rotate_right"]
        kind=rng.choice(kinds)
        if kind=="swap":
            i,j=sorted(rng.sample(range(n),2)); op={"kind":kind,"i":i,"j":j}
        elif kind=="move":
            i,j=rng.sample(range(n),2); op={"kind":kind,"i":i,"j":j}
        elif kind=="reverse":
            i,j=sorted(rng.sample(range(n),2)); op={"kind":kind,"i":i,"j":j}
        else:
            op={"kind":kind,"amount":rng.randint(1,n-1)}
        state=apply_tracking_op(state,op); ops.append(op); states.append(list(state))
    qkind=rng.choice(["object_at","position_of"])
    if qkind=="object_at":
        p=rng.randrange(n); answer=state[p]; question=f"Which object is finally in position {p+1}?"; query={"kind":qkind,"position":p}; answer_type="item"
    else:
        obj=rng.choice(objects); p=state.index(obj); answer=str(p+1); question=f"What is the final position of the {obj}?"; query={"kind":qkind,"object":obj}; answer_type="integer"
    t=template_index(split,rng)
    intros=[
        "Track a row of tokens from left to right.",
        "Several cups are rearranged in a fixed sequence.",
        "A card row undergoes deterministic moves.",
        "Expedition markers are reordered from left to right.",
        "A robot sorter transforms a row of components.",
        "Execute every state transition exactly in order.",
    ]
    initial_text="; ".join(f"{i+1}: {x}" for i,x in enumerate(initial))
    op_text=" ".join(f"{i+1}. {cap_first(tracking_op_text(o))}." for i,o in enumerate(ops))
    prompt=f"{intros[t]} Initial row: {initial_text}. Operations: {op_text} {question} Give only the answer."
    steps=[f"Start: {' | '.join(initial)}."]
    for i,(op,st) in enumerate(zip(ops,states),1):
        steps.append(f"After step {i} ({tracking_op_text(op)}): {' | '.join(st)}.")
    steps.append(f"Read the requested value from the final state.")
    final=answer.capitalize()+"." if answer_type=="item" else answer+"."
    structure={"initial":initial,"operations":ops,"query":query}
    return base_row(
        family="object_tracking",split=split,difficulty=difficulty,
        template_id=f"object_tracking.t{t}",prompt=prompt,
        reasoning_steps=steps,compact_state=" | ".join(state),final_answer=final,
        accepted_answers=[answer,answer.capitalize(),final],answer_type=answer_type,
        structure=structure,solver="deterministic_state_simulation",
        verification={"operation_count":n_ops,"final_state":state},
        tags=["state-tracking","simulation","object-permutation","working-memory"],
    )


# ---------------------------------------------------------------------------
# 9. Mastermind-style code breaking
# ---------------------------------------------------------------------------

def code_feedback(secret: tuple[str,...], guess: tuple[str,...]) -> tuple[int,int]:
    exact=sum(a==b for a,b in zip(secret,guess))
    cs=collections.Counter(secret); cg=collections.Counter(guess)
    common=sum(min(cs[d],cg[d]) for d in cs)
    return exact,common-exact


_CODE_CACHE: dict[tuple[int, tuple[str, ...], bool], tuple[list[tuple[str, ...]], np.ndarray]] = {}


def code_space(length: int, alphabet: list[str], repeats: bool) -> tuple[list[tuple[str, ...]], np.ndarray]:
    key = (length, tuple(alphabet), repeats)
    if key in _CODE_CACHE:
        return _CODE_CACHE[key]
    candidates = list(itertools.product(alphabet, repeat=length)) if repeats else list(itertools.permutations(alphabet, length))
    n = len(candidates)
    matrix = np.empty((n, n), dtype=np.uint8)
    # Feedback is symmetric, so compute only one triangle.
    for i, a in enumerate(candidates):
        matrix[i, i] = length * 10
        for j in range(i + 1, n):
            exact, misplaced = code_feedback(a, candidates[j])
            code = exact * 10 + misplaced
            matrix[i, j] = code
            matrix[j, i] = code
    _CODE_CACHE[key] = (candidates, matrix)
    return candidates, matrix


def generate_code_breaking(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(300):
        if difficulty=="easy":
            length=3; alphabet=list("012345"); repeats=False; max_clues=4
        elif difficulty=="medium":
            length=4; alphabet=list("0123456"); repeats=False; max_clues=5
        else:
            length=4; alphabet=list("012345"); repeats=True; max_clues=6
        candidates, feedback_matrix = code_space(length, alphabet, repeats)
        n_candidates = len(candidates)
        secret_idx = rng.randrange(n_candidates)
        secret = candidates[secret_idx]
        current = np.arange(n_candidates, dtype=np.int32)
        pool = rng.sample(range(n_candidates), min(n_candidates, 96))
        if secret_idx in pool:
            pool.remove(secret_idx)
        chosen=[]
        for _step in range(max_clues):
            if not pool:
                break
            trial = rng.sample(pool, min(len(pool), 24))
            scored=[]
            for g in trial:
                fb = int(feedback_matrix[g, secret_idx])
                mask = feedback_matrix[g, current] == fb
                count = int(mask.sum())
                if 0 < count < len(current):
                    scored.append((count, rng.random(), g, fb, mask))
            if not scored:
                # Expand the search once when the random trial is unhelpful.
                for g in pool[:min(len(pool), 64)]:
                    fb = int(feedback_matrix[g, secret_idx])
                    mask = feedback_matrix[g, current] == fb
                    count = int(mask.sum())
                    if 0 < count < len(current):
                        scored.append((count, rng.random(), g, fb, mask))
            if not scored:
                break
            scored.sort(key=lambda x:(x[0],x[1]))
            count,_,g,fb,mask = rng.choice(scored[:min(4,len(scored))])
            before=len(current)
            current=current[mask]
            chosen.append({"guess":candidates[g],"exact":fb//10,"misplaced":fb%10,"before":before,"after":len(current)})
            pool.remove(g)
            if len(current)==1:
                break
        if len(current)!=1 or int(current[0])!=secret_idx or len(chosen)<2:
            continue
        t=template_index(split,rng)
        intros=[
            "Crack a secret digit code.",
            "A lock uses a hidden code.",
            "A laboratory keypad has one valid code.",
            "An expedition vault is protected by a numeric code.",
            "A robot access panel stores a hidden sequence.",
            "Solve the exact constraint system for the code.",
        ]
        repeat_text="Digits may repeat." if repeats else "No digit repeats."
        clue_text=" ".join(f"Guess {''.join(c['guess'])}: {c['exact']} exact-position and {c['misplaced']} correct-digit-wrong-position." for c in chosen)
        prompt=(f"{intros[t]} It has {length} digits chosen from {oxford(alphabet)}. {repeat_text} "
                f"For each clue, 'exact-position' and 'wrong-position' counts are both exact. {clue_text} What is the code? Give only the code.")
        steps=[f"Initially, {n_candidates} codes satisfy the length and repetition rule."]
        for c in chosen:
            steps.append(f"Feedback for {''.join(c['guess'])} ({c['exact']} exact, {c['misplaced']} misplaced); codes: {c['before']} → {c['after']}.")
        code="".join(secret); steps.append(f"Only {code} remains.")
        final="-".join(secret)+"."
        structure={"alphabet":alphabet,"length":length,"repeats":repeats,"clues":[{"guess":c["guess"],"exact":c["exact"],"misplaced":c["misplaced"]} for c in chosen]}
        return base_row(
            family="code_breaking",split=split,difficulty=difficulty,
            template_id=f"code_breaking.t{t}",prompt=prompt,
            reasoning_steps=steps,compact_state=f"Unique code={code}",final_answer=final,
            accepted_answers=[code,"-".join(secret),final],answer_type="code",
            structure=structure,solver="mastermind_constraint_filter",
            verification={"initial_candidates":n_candidates,"solution_count":1,"clue_count":len(chosen)},
            tags=["code-breaking","mastermind","constraint-filtering","combinatorics"],
        )
    raise RuntimeError("Failed to generate code_breaking")


# ---------------------------------------------------------------------------
# 10. Calendar and temporal offset reasoning
# ---------------------------------------------------------------------------
WEEKDAYS=["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]

def generate_calendar_reasoning(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(300):
        n=rng.choice([3,4]) if difficulty=="easy" else (rng.choice([5,6]) if difficulty=="medium" else 7)
        events=rng.sample(EVENTS,n)
        root=0; root_day=rng.randint(6,22)
        days=[None]*n; days[root]=root_day
        depths=[0]*n
        edges=[]
        order=list(range(1,n)); rng.shuffle(order)
        placed=[0]
        ok=True
        for child in order:
            parent=rng.choice(placed)
            possible=[]
            for delta in list(range(-7,0))+list(range(1,8)):
                d=days[parent]+delta
                if 1<=d<=28 and d not in days:
                    possible.append(delta)
            if not possible: ok=False; break
            delta=rng.choice(possible); days[child]=days[parent]+delta
            edges.append({"a":child,"b":parent,"delta":delta}) # day[a]=day[b]+delta
            depths[child]=depths[parent]+1
            placed.append(child)
        if not ok: continue
        min_depth = 1 if difficulty=="easy" else (2 if difficulty=="medium" else 3)
        target_options=[i for i in range(1,n) if depths[i]>=min_depth]
        if not target_options: continue
        target=rng.choice(target_options)
        # Add one or two redundant true relations as distractors, but do not
        # create a shortcut directly involving the queried event.
        extra_count=0 if difficulty=="easy" else (1 if difficulty=="medium" else 2)
        pairs=list(itertools.permutations(range(n),2)); rng.shuffle(pairs)
        for a,b in pairs:
            if extra_count<=0: break
            if target in (a,b): continue
            if any((e["a"],e["b"])==(a,b) for e in edges): continue
            delta=days[a]-days[b]
            if abs(delta)<=12 and delta!=0:
                edges.append({"a":a,"b":b,"delta":delta,"redundant":True}); extra_count-=1
        qkind=rng.choice(["day_number","weekday"])
        start_weekday=rng.randrange(7)
        if qkind=="day_number":
            answer=f"Day {days[target]}"; question=f"On which numbered day does the {events[target]} occur?"; answer_type="date_number"
        else:
            wd=WEEKDAYS[(start_weekday+days[target]-1)%7]; answer=wd; question=f"On which weekday does the {events[target]} occur?"; answer_type="weekday"
        t=template_index(split,rng)
        intros=[
            "A 28-day schedule contains the following events.",
            "Use the exact day offsets in this monthly plan.",
            "A project calendar has one occurrence of each event.",
            "An expedition uses a 28-day operations calendar.",
            "A robot lab schedules each task once during a 28-day cycle.",
            "Solve the temporal constraint graph exactly.",
        ]
        clue_text=[f"The {events[root]} is on day {root_day}"]
        for e in edges:
            if e["delta"]>0:
                unit="day" if e["delta"]==1 else "days"
                clue_text.append(f"the {events[e['a']]} is {e['delta']} {unit} after the {events[e['b']]} ")
            else:
                amount=-e["delta"]; unit="day" if amount==1 else "days"
                clue_text.append(f"the {events[e['a']]} is {amount} {unit} before the {events[e['b']]} ")
        prompt=(f"{intros[t]} Day 1 is {WEEKDAYS[start_weekday]}. Clues: "+"; ".join(x.strip() for x in clue_text)+
                f". {question} Give only the answer.")
        # Find a path from root to target using the tree portion and propagate dates.
        adj=collections.defaultdict(list)
        for e in edges:
            adj[e["b"]].append((e["a"],e["delta"])); adj[e["a"]].append((e["b"],-e["delta"]))
        prev={root:(None,0)}; dq=collections.deque([root])
        while dq:
            u=dq.popleft()
            if u==target: break
            for v,d in adj[u]:
                if v not in prev: prev[v]=(u,d); dq.append(v)
        path=[]; cur=target
        while cur!=root:
            p,d=prev[cur]; path.append((p,cur,d)); cur=p
        path.reverse(); running=root_day
        steps=[f"Anchor {events[root]} at day {root_day}."]
        for u,v,d in path:
            running+=d
            relation=f"+{d}" if d>0 else str(d)
            unit="day" if abs(d)==1 else "days"
            steps.append(f"From {events[u]} to {events[v]} apply {relation} {unit}, giving day {running}.")
        if qkind=="weekday":
            wd=WEEKDAYS[(start_weekday+days[target]-1)%7]
            steps.append(f"Day {days[target]} is {wd} because day 1 is {WEEKDAYS[start_weekday]}.")
        else:
            steps.append(f"Therefore the {events[target]} is on day {days[target]}.")
        compact="; ".join(f"{events[i]}=day {days[i]}" for i in range(n))
        final=answer+"."
        structure={"events":events,"root":root,"root_day":root_day,"edges":edges,"start_weekday":start_weekday,"query":{"kind":qkind,"target":target}}
        return base_row(
            family="calendar_reasoning",split=split,difficulty=difficulty,
            template_id=f"calendar_reasoning.t{t}",prompt=prompt,
            reasoning_steps=steps,compact_state=compact,final_answer=final,
            accepted_answers=[answer,final],answer_type=answer_type,
            structure=structure,solver="temporal_offset_graph_propagation",
            verification={"event_count":n,"target_day":days[target]},
            tags=["temporal-reasoning","calendar","offset-arithmetic","graph-propagation"],
        )
    raise RuntimeError("Failed to generate calendar_reasoning")

# ---------------------------------------------------------------------------
# 11. Family-relation graph reasoning
# ---------------------------------------------------------------------------

def generate_family_relations(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(300):
        used=set()
        def pick(gender: str) -> str:
            pool=MALE_NAMES if gender=="M" else FEMALE_NAMES
            choices=[x for x in pool if x not in used]
            if not choices: raise RuntimeError("name pool exhausted")
            x=rng.choice(choices); used.add(x); return x
        gender: dict[str,str]={}
        parents: dict[str,set[str]]=collections.defaultdict(set)
        # Two grandparents, two or three adult siblings, their partners, and children.
        gpa=pick("M"); gma=pick("F"); gender[gpa]="M"; gender[gma]="F"
        adult_count=2 if difficulty=="easy" else 3
        adults=[]
        for _i in range(adult_count):
            g=rng.choice(["M","F"]); a=pick(g); gender[a]=g; adults.append(a)
            parents[a].update([gpa,gma])
        children_by_adult={}
        partners=[]
        for a in adults:
            pg="F" if gender[a]=="M" else "M"; partner=pick(pg); gender[partner]=pg; partners.append(partner)
            child_count=1 if difficulty=="easy" else rng.choice([1,2])
            kids=[]
            for _j in range(child_count):
                kg=rng.choice(["M","F"]); kid=pick(kg); gender[kid]=kg; kids.append(kid)
                parents[kid].update([a,partner])
            children_by_adult[a]=kids
        people=list(gender)

        def siblings(x:str,y:str)->bool:
            return x!=y and bool(parents.get(x,set()) & parents.get(y,set()))
        def grandparents_of(y:str)->set[str]:
            return {gp for p in parents.get(y,set()) for gp in parents.get(p,set())}
        def relation(x:str,y:str)->str|None:
            # x relative to y
            if x in parents.get(y,set()): return "father" if gender[x]=="M" else "mother"
            if y in parents.get(x,set()): return "son" if gender[x]=="M" else "daughter"
            if siblings(x,y): return "brother" if gender[x]=="M" else "sister"
            if x in grandparents_of(y): return "grandfather" if gender[x]=="M" else "grandmother"
            if y in grandparents_of(x): return "grandson" if gender[x]=="M" else "granddaughter"
            # aunt/uncle: x sibling of one of y's parents
            if any(siblings(x,p) for p in parents.get(y,set())): return "uncle" if gender[x]=="M" else "aunt"
            # niece/nephew: y sibling of one of x's parents
            if any(siblings(y,p) for p in parents.get(x,set())): return "nephew" if gender[x]=="M" else "niece"
            # first cousins: at least one parent of each are siblings
            if any(siblings(px,py) for px in parents.get(x,set()) for py in parents.get(y,set())):
                return "first cousin"
            return None
        pairs=[(x,y,relation(x,y)) for x in people for y in people if x!=y and relation(x,y)]
        if not pairs: continue
        # Bias toward multi-hop relations in medium/challenge.
        if difficulty!="easy":
            hard=[p for p in pairs if p[2] in {"grandfather","grandmother","grandson","granddaughter","aunt","uncle","niece","nephew","first cousin"}]
            if hard: pairs=hard
        x,y,rel=rng.choice(pairs)
        # Emit all parent facts; they are compact and make inference fully explicit.
        facts=[]
        for child,ps in parents.items():
            for p in sorted(ps):
                term="father" if gender[p]=="M" else "mother"
                facts.append(f"{p} is the {term} of {child}")
        rng.shuffle(facts)
        t=template_index(split,rng)
        intros=[
            "Use the stated biological family relationships.",
            "A family-tree puzzle gives these parent facts.",
            "Determine the kinship relation from the family graph.",
            "An expedition registry records the following family links.",
            "A genealogy robot stores these exact parent relationships.",
            "Infer the requested relationship without adding unstated family links.",
        ]
        gender_note = f"{x} is {'male' if gender[x]=='M' else 'female'}"
        prompt=f"{intros[t]} Facts: {'; '.join(facts)}. Gender note: {gender_note}. How is {x} related to {y}? Give only the relationship."
        steps=[]
        if rel in ("father","mother"):
            steps.append(f"{x} is listed directly as a parent of {y}.")
        elif rel in ("son","daughter"):
            steps.append(f"{y} is listed as a parent of {x}.")
        elif rel in ("brother","sister"):
            shared=next(iter(parents[x]&parents[y])); steps.append(f"{x} and {y} share the parent {shared}, so they are siblings.")
        elif rel.startswith("grand"):
            if x in grandparents_of(y):
                mid=next(p for p in parents[y] if x in parents.get(p,set())); steps.append(f"{x} is parent of {mid}, who is parent of {y}.")
            else:
                mid=next(p for p in parents[x] if y in parents.get(p,set())); steps.append(f"{y} is parent of {mid}, who is parent of {x}.")
        elif rel in ("aunt","uncle"):
            py=next(p for p in parents[y] if siblings(x,p)); shared=next(iter(parents[x]&parents[py])); steps.append(f"{x} and {py} share parent {shared}; {py} is a parent of {y}.")
        elif rel in ("niece","nephew"):
            px=next(p for p in parents[x] if siblings(y,p)); shared=next(iter(parents[y]&parents[px])); steps.append(f"{y} and {px} share parent {shared}; {px} is a parent of {x}.")
        else:
            px,py=next((px,py) for px in parents[x] for py in parents[y] if siblings(px,py)); shared=next(iter(parents[px]&parents[py])); steps.append(f"{px} and {py} are siblings through {shared}; they are parents of {x} and {y}.")
        steps.append(f"Using {x}'s recorded gender, the precise relationship is {rel}.")
        final=rel.capitalize()+"."
        structure={"gender":gender,"parents":{k:sorted(v) for k,v in parents.items()},"query":{"x":x,"y":y}}
        return base_row(
            family="family_relations",split=split,difficulty=difficulty,
            template_id=f"family_relations.t{t}",prompt=prompt,
            reasoning_steps=steps,compact_state=f"relation({x},{y})={rel}",final_answer=final,
            accepted_answers=[rel,rel.capitalize(),final],answer_type="relationship",
            structure=structure,solver="kinship_graph_inference",
            verification={"relation":rel,"people":len(people)},
            tags=["family-tree","relations","graph-reasoning","multi-hop"],
        )
    raise RuntimeError("Failed to generate family_relations")


# ---------------------------------------------------------------------------
# 12. Unique shortest-path reasoning
# ---------------------------------------------------------------------------

def dijkstra_unique(n:int, edges:list[tuple[int,int,int]], start:int, directed:bool=False):
    adj=[[] for _ in range(n)]
    for u,v,w in edges:
        adj[u].append((v,w))
        if not directed: adj[v].append((u,w))
    inf=10**18; dist=[inf]*n; count=[0]*n; prev=[None]*n
    dist[start]=0; count[start]=1; pq=[(0,start)]; settled=[]
    while pq:
        d,u=heapq.heappop(pq)
        if d!=dist[u]: continue
        settled.append((u,d))
        for v,w in adj[u]:
            nd=d+w
            if nd<dist[v]:
                dist[v]=nd; count[v]=count[u]; prev[v]=u; heapq.heappush(pq,(nd,v))
            elif nd==dist[v]:
                count[v]+=count[u]; prev[v]=None
    return dist,count,prev,settled,adj


def generate_shortest_path(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(500):
        n=rng.choice([5,6]) if difficulty=="easy" else (rng.choice([6,7]) if difficulty=="medium" else 8)
        nodes=rng.sample(LOCATIONS,n)
        directed=difficulty=="challenge" and rng.random()<0.5
        # Start with a connected undirected tree (or strongly reachable directed backbone).
        edges=[]
        for v in range(1,n):
            u=rng.randrange(v); w=rng.randint(1,9)
            edges.append((u,v,w))
            if directed and rng.random()<0.4:
                edges[-1]=(v,u,w)
                edges.append((u,v,w+rng.randint(1,4)))
        extra=rng.randint(n//2,n) if difficulty!="easy" else rng.randint(2,4)
        existing={(u,v) for u,v,_ in edges}
        for _e in range(extra*3):
            if extra<=0: break
            u,v=rng.sample(range(n),2)
            key=(u,v) if directed else tuple(sorted((u,v)))
            existing_norm={(a,b) if directed else tuple(sorted((a,b))) for a,b,_ in edges}
            if key in existing_norm: continue
            edges.append((u,v,rng.randint(1,12))); extra-=1
        s,tgt=rng.sample(range(n),2)
        dist,count,prev,settled,adj=dijkstra_unique(n,edges,s,directed)
        if dist[tgt]>=10**18 or count[tgt]!=1: continue
        path=[]; cur=tgt
        while cur is not None:
            path.append(cur)
            if cur==s: break
            cur=prev[cur]
        if not path or path[-1]!=s: continue
        path=path[::-1]
        min_hops = 1 if difficulty=="easy" else (2 if difficulty=="medium" else 3)
        if len(path)-1 < min_hops: continue
        qkind=rng.choice(["distance","route","first_hop"])
        if qkind=="distance":
            answer=str(dist[tgt]); question=f"What is the minimum total cost from {nodes[s]} to {nodes[tgt]}?"; answer_type="integer"
        elif qkind=="route":
            answer=" → ".join(nodes[i] for i in path); question=f"What is the unique cheapest route from {nodes[s]} to {nodes[tgt]}?"; answer_type="path"
        else:
            answer=nodes[path[1]]; question=f"Which location should be visited immediately after {nodes[s]} on the unique cheapest route to {nodes[tgt]}?"; answer_type="location"
        t=template_index(split,rng)
        intros=[
            "Find a cheapest route in this weighted map.",
            "A courier network has the following travel costs.",
            "A set of rooms is connected by weighted corridors.",
            "An expedition map lists travel-energy costs.",
            "A robot navigation graph uses these edge costs.",
            "Solve the unique shortest-path problem exactly.",
        ]
        edge_text=[]
        for u,v,w in edges:
            arrow="→" if directed else "—"
            edge_text.append(f"{nodes[u]} {arrow} {nodes[v]} costs {w}")
        prompt=f"{intros[t]} Edges: {'; '.join(edge_text)}. {question} Give only the answer."
        settled_to_target=[]
        for u,d in settled:
            settled_to_target.append(f"{nodes[u]}={d}")
            if u==tgt: break
        steps=[f"Run Dijkstra's algorithm from {nodes[s]}. Finalized distances through the target are {', '.join(settled_to_target)}."]
        steps.append(f"Backtracking unique predecessors gives {' → '.join(nodes[i] for i in path)} with total cost {dist[tgt]}.")
        final=answer+"."
        accepted=[answer,final]
        if qkind=="route": accepted.append(" -> ".join(nodes[i] for i in path))
        structure={"nodes":nodes,"edges":edges,"directed":directed,"query":{"kind":qkind,"start":s,"target":tgt}}
        return base_row(
            family="shortest_path",split=split,difficulty=difficulty,
            template_id=f"shortest_path.t{t}",prompt=prompt,
            reasoning_steps=steps,compact_state=f"path={answer if qkind=='route' else ' → '.join(nodes[i] for i in path)}; cost={dist[tgt]}",final_answer=final,
            accepted_answers=accepted,answer_type=answer_type,structure=structure,
            solver="dijkstra_unique_shortest_path",
            verification={"path_count":count[tgt],"distance":dist[tgt],"path":[nodes[i] for i in path]},
            tags=["graph","shortest-path","planning","dijkstra"],
        )
    raise RuntimeError("Failed to generate shortest_path")


# ---------------------------------------------------------------------------
# 13. Arithmetic constraint word problems
# ---------------------------------------------------------------------------

def generate_arithmetic_constraints(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    t=template_index(split,rng)
    template_pool=["animals","tickets","coins","ages","packages"]
    kind="three_types" if difficulty=="challenge" else rng.choice(template_pool)
    if kind=="animals":
        x=rng.randint(4,35); y=rng.randint(3,25); total=x+y; legs=2*x+4*y
        ask=rng.choice(["x","y"]); answer=x if ask=="x" else y; label="birds" if ask=="x" else "rabbits"
        prompt=f"A pen contains only birds and rabbits. There are {total} animals and {legs} legs in total. How many {label} are there? Give only the number."
        steps=[f"Let b be birds and r be rabbits: b+r={total} and 2b+4r={legs}.",f"Subtract 2(b+r)={2*total} from the leg equation: 2r={legs-2*total}, so r={y}.",f"Then b={total}-{y}={x}."]
        structure={"equations":[[1,1,total],[2,4,legs]],"variables":["birds","rabbits"],"query":ask}
    elif kind=="tickets":
        adult=rng.randint(5,30); child=rng.randint(5,35); pa=rng.randint(7,15); pc=rng.randint(2,6)
        while pa==pc: pa=rng.randint(7,15)
        total=adult+child; revenue=adult*pa+child*pc; ask=rng.choice(["adult","child"]); answer=adult if ask=="adult" else child
        prompt=f"A venue sold {total} tickets. Adult tickets cost ${pa} and child tickets cost ${pc}. Total revenue was ${revenue}. How many {ask} tickets were sold? Give only the number."
        steps=[f"Let a and c be adult and child tickets: a+c={total}, {pa}a+{pc}c={revenue}.",f"Substitute c={total}-a: ({pa}-{pc})a={revenue-pc*total}, so a={adult}.",f"Then c={total}-{adult}={child}."]
        structure={"equations":[[1,1,total],[pa,pc,revenue]],"variables":["adult","child"],"query":ask}
    elif kind=="coins":
        pennies=rng.randint(5,30); nickels=rng.randint(4,25); total=pennies+nickels; value=pennies+5*nickels
        ask=rng.choice(["pennies","nickels"]); answer=pennies if ask=="pennies" else nickels
        prompt=f"A jar contains only pennies and nickels. It has {total} coins worth {value} cents. How many {ask} are in the jar? Give only the number."
        steps=[f"Let p and n be counts: p+n={total}, p+5n={value}.",f"Subtract the count equation: 4n={value-total}, so n={nickels}.",f"Then p={total}-{nickels}={pennies}."]
        structure={"equations":[[1,1,total],[1,5,value]],"variables":["pennies","nickels"],"query":ask}
    elif kind=="ages":
        young=rng.randint(8,30); diff=rng.randint(4,25); old=young+diff; total=old+young
        ask=rng.choice(["older","younger"]); answer=old if ask=="older" else young
        prompt=f"Two people have ages summing to {total}. The older person is {diff} years older than the younger person. What is the {ask} person's age? Give only the number."
        steps=[f"Let o+y={total} and o-y={diff}.",f"Adding gives 2o={total+diff}, so o={old}.",f"Then y={total}-{old}={young}."]
        structure={"equations":[[1,1,total],[1,-1,diff]],"variables":["older","younger"],"query":ask}
    elif kind=="packages":
        light=rng.randint(4,30); heavy=rng.randint(3,20); wl=rng.randint(2,5); wh=rng.randint(6,12)
        total=light+heavy; weight=light*wl+heavy*wh; ask=rng.choice(["light","heavy"]); answer=light if ask=="light" else heavy
        prompt=f"A shipment has {total} packages. Light packages weigh {wl} kg and heavy packages weigh {wh} kg. The total is {weight} kg. How many {ask} packages are there? Give only the number."
        steps=[f"Let l+h={total} and {wl}l+{wh}h={weight}.",f"Substitute l={total}-h: ({wh}-{wl})h={weight-wl*total}, so h={heavy}.",f"Then l={total}-{heavy}={light}."]
        structure={"equations":[[1,1,total],[wl,wh,weight]],"variables":["light","heavy"],"query":ask}
    else:
        x=rng.randint(2,15); y=rng.randint(2,15); z=rng.randint(2,15)
        total=x+y+z
        weights=rng.sample(range(2,9),3)
        while weights[0] + weights[1] == 2 * weights[2]:
            weights=rng.sample(range(2,9),3)
        weighted=weights[0]*x+weights[1]*y+weights[2]*z; diff=x-y
        ask=rng.choice([0,1,2]); labels=["red","blue","green"]; answer=[x,y,z][ask]
        prompt=(f"A box has red, blue, and green tokens: {total} total. Their weighted score is {weights[0]}R+{weights[1]}B+{weights[2]}G={weighted}, "
                f"and R−B={diff}. How many {labels[ask]} tokens are there? Give only the number.")
        steps=[f"Use R+B+G={total}, {weights[0]}R+{weights[1]}B+{weights[2]}G={weighted}, and R−B={diff}.",f"Solving the three independent equations gives R={x}, B={y}, G={z}."]
        structure={"equations":[[1,1,1,total],[*weights,weighted],[1,-1,0,diff]],"variables":labels,"query":ask}
    intros=["Solve the count constraints.","Use exact integer arithmetic.","A small algebra puzzle follows.","An expedition ledger gives these totals.","A robot inventory system reports exact totals.","Solve the simultaneous constraints exactly."]
    prompt=f"{intros[t]} {prompt}"
    final=f"{answer}."
    return base_row(
        family="arithmetic_constraints",split=split,difficulty=difficulty,
        template_id=f"arithmetic_constraints.t{t}",prompt=prompt,
        reasoning_steps=steps,compact_state=f"solution={answer}",final_answer=final,
        accepted_answers=[str(answer),final],answer_type="integer",structure={"kind":kind,**structure},
        solver="integer_linear_equation_solution",
        verification={"integer_solution":True,"answer":answer},
        tags=["arithmetic","word-problem","linear-equations","integer-reasoning"],
    )


# ---------------------------------------------------------------------------
# 14. Boolean switch and circuit logic
# ---------------------------------------------------------------------------

def make_bool_expr(rng: random.Random, variables:list[str], depth:int) -> dict[str,Any]:
    if depth<=0 or (depth>1 and rng.random()<0.25):
        return {"op":"var","name":rng.choice(variables)}
    op=rng.choice(["and","or","xor","not"] if depth<4 else ["and","or","xor","not","implies"])
    if op=="not": return {"op":"not","x":make_bool_expr(rng,variables,depth-1)}
    return {"op":op,"a":make_bool_expr(rng,variables,depth-1),"b":make_bool_expr(rng,variables,depth-1)}


def bool_vars(expr:dict[str,Any])->set[str]:
    if expr["op"]=="var": return {expr["name"]}
    if expr["op"]=="not": return bool_vars(expr["x"])
    return bool_vars(expr["a"])|bool_vars(expr["b"])


def eval_bool_expr(expr:dict[str,Any],assignment:dict[str,bool],trace:list[str]|None=None)->bool:
    op=expr["op"]
    if op=="var": return assignment[expr["name"]]
    if op=="not":
        x=eval_bool_expr(expr["x"],assignment,trace); v=not x
    else:
        a=eval_bool_expr(expr["a"],assignment,trace); b=eval_bool_expr(expr["b"],assignment,trace)
        if op=="and": v=a and b
        elif op=="or": v=a or b
        elif op=="xor": v=a^b
        elif op=="implies": v=(not a) or b
        else: raise ValueError(op)
    if trace is not None: trace.append(f"{bool_expr_text(expr)} = {'On' if v else 'Off'}")
    return v


def bool_expr_text(expr:dict[str,Any])->str:
    op=expr["op"]
    if op=="var": return expr["name"]
    if op=="not": return f"NOT ({bool_expr_text(expr['x'])})"
    names={"and":"AND","or":"OR","xor":"XOR","implies":"IMPLIES"}
    return f"({bool_expr_text(expr['a'])} {names[op]} {bool_expr_text(expr['b'])})"


def generate_boolean_switches(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(300):
        n=rng.choice([3,4]) if difficulty=="easy" else (rng.choice([4,5]) if difficulty=="medium" else 6)
        variables=[f"S{i+1}" for i in range(n)]
        depth=2 if difficulty=="easy" else (3 if difficulty=="medium" else 4)
        expr=make_bool_expr(rng,variables,depth)
        used=sorted(bool_vars(expr))
        if len(used)<2: continue
        assignment={v:rng.choice([False,True]) for v in variables}
        qkind=rng.choice(["output","infer"])
        if qkind=="infer":
            sensitive=[]
            for v in used:
                a0=dict(assignment); a1=dict(assignment); a0[v]=False; a1[v]=True
                if eval_bool_expr(expr,a0)!=eval_bool_expr(expr,a1): sensitive.append(v)
            if not sensitive: continue
            unknown=rng.choice(sensitive); output=eval_bool_expr(expr,assignment)
            known={v:x for v,x in assignment.items() if v!=unknown}
            answer="On" if assignment[unknown] else "Off"
            question=f"The lamp is {'On' if output else 'Off'}. What must {unknown} be?"
            query={"kind":qkind,"unknown":unknown,"output":output}
        else:
            known=assignment; output=eval_bool_expr(expr,assignment); answer="On" if output else "Off"
            question="Is the lamp On or Off?"; query={"kind":qkind}
        t=template_index(split,rng)
        intros=["Evaluate a Boolean switch circuit.","A lamp follows an exact logic rule.","A control panel uses Boolean operations.","An expedition beacon uses a logic circuit.","A robot indicator follows the stated gate expression.","Solve the Boolean expression without assuming unstated states."]
        states="; ".join(f"{v}={'On' if x else 'Off'}" for v,x in known.items())
        prompt=(f"{intros[t]} Rule: the lamp is On exactly when {bool_expr_text(expr)} is true. Known switch states: {states}. "
                "AND requires both; OR requires at least one; XOR requires exactly one; IMPLIES is false only when its left side is true and right side false. "
                f"{question} Give only On or Off.")
        trace=[]; _=eval_bool_expr(expr,assignment,trace)
        steps=[f"Substitute the known switch states into {bool_expr_text(expr)}."]+trace[-min(6,len(trace)):]
        if qkind=="infer":
            a0=dict(assignment);a1=dict(assignment);a0[unknown]=False;a1[unknown]=True
            steps.append(f"With {unknown}=Off the output is {'On' if eval_bool_expr(expr,a0) else 'Off'}; with {unknown}=On it is {'On' if eval_bool_expr(expr,a1) else 'Off'}. Only {answer} matches.")
        else: steps.append(f"The final circuit output is {answer}.")
        final=answer+"."
        structure={"variables":variables,"expression":expr,"known":known,"query":query}
        return base_row(
            family="boolean_switches",split=split,difficulty=difficulty,
            template_id=f"boolean_switches.t{t}",prompt=prompt,
            reasoning_steps=steps,compact_state=f"expression={bool_expr_text(expr)}; answer={answer}",final_answer=final,
            accepted_answers=[answer,answer.lower(),final],answer_type="boolean_state",structure=structure,
            solver="boolean_expression_evaluation",
            verification={"used_variables":used,"answer":answer},
            tags=["boolean-logic","switches","circuits","symbolic-evaluation"],
        )
    raise RuntimeError("Failed to generate boolean_switches")


# ---------------------------------------------------------------------------
# 15. Tower-of-Hanoi state-space planning
# ---------------------------------------------------------------------------
_HANOI_ADJ: dict[int,dict[tuple[int,...],list[tuple[int,...]]]]={}
_HANOI_DIST: dict[tuple[int,tuple[int,...]],dict[tuple[int,...],int]]={}

def hanoi_neighbors(state:tuple[int,...])->list[tuple[int,...]]:
    n=len(state); tops={}
    for disk in range(n): # disk 0 is smallest; first found on each peg is top
        peg=state[disk]
        if peg not in tops: tops[peg]=disk
    out=[]
    for src,disk in tops.items():
        for dst in range(3):
            if dst==src: continue
            dst_top=tops.get(dst)
            if dst_top is None or disk<dst_top:
                s=list(state);s[disk]=dst;out.append(tuple(s))
    return out

def hanoi_adj(n:int):
    if n not in _HANOI_ADJ:
        _HANOI_ADJ[n]={s:hanoi_neighbors(s) for s in itertools.product(range(3),repeat=n)}
    return _HANOI_ADJ[n]

def hanoi_distances(n:int,start:tuple[int,...])->dict[tuple[int,...],int]:
    key=(n,start)
    if key in _HANOI_DIST: return _HANOI_DIST[key]
    adj=hanoi_adj(n); dist={start:0};dq=collections.deque([start])
    while dq:
        u=dq.popleft()
        for v in adj[u]:
            if v not in dist: dist[v]=dist[u]+1;dq.append(v)
    _HANOI_DIST[key]=dist;return dist

def hanoi_state_text(state:tuple[int,...],peg_names:Sequence[str])->str:
    parts=[]
    for p,name in enumerate(peg_names):
        disks=[d+1 for d in range(len(state)-1,-1,-1) if state[d]==p]
        parts.append(f"{name}=[{', '.join(map(str,disks)) if disks else 'empty'}] bottom-to-top")
    return "; ".join(parts)

def hanoi_move(a:tuple[int,...],b:tuple[int,...],peg_names:Sequence[str])->str:
    disk=next(i for i,(x,y) in enumerate(zip(a,b)) if x!=y)
    return f"disk {disk+1}: {peg_names[a[disk]]}→{peg_names[b[disk]]}"

def generate_hanoi_planning(rng: random.Random, split: str, difficulty: str) -> dict[str, Any]:
    for _ in range(500):
        n=rng.choice([3,4]) if difficulty=="easy" else (rng.choice([4,5]) if difficulty=="medium" else rng.choice([5,6]))
        states=list(itertools.product(range(3),repeat=n))
        start=rng.choice(states); goal=rng.choice(states)
        if start==goal: continue
        dist_to_goal=hanoi_distances(n,goal); d=dist_to_goal[start]
        lo,hi=(2,8) if difficulty=="easy" else ((5,14) if difficulty=="medium" else (10,22))
        if not (lo<=d<=hi): continue
        adj=hanoi_adj(n)
        path=[start];cur=start
        while cur!=goal:
            opts=[v for v in adj[cur] if dist_to_goal[v]==dist_to_goal[cur]-1]
            if not opts: break
            cur=rng.choice(opts);path.append(cur)
        if path[-1]!=goal: continue
        optimal_first=[v for v in adj[start] if dist_to_goal[v]==d-1]
        qkind=rng.choice(["min_moves","first_move"] if len(optimal_first)==1 else ["min_moves"])
        peg_names=list(rng.choice([["A","B","C"],["Left","Middle","Right"],["North","Center","South"]]))
        if qkind=="min_moves":
            answer=f"{d} moves"; question="What is the minimum number of legal moves needed?"; answer_type="move_count"
        else:
            answer=hanoi_move(start,optimal_first[0],peg_names); question="What is the uniquely optimal first move?"; answer_type="move"
        t=template_index(split,rng)
        intros=["Solve a Tower-of-Hanoi state transition.","Plan legal disk moves between three pegs.","A disk-transfer puzzle begins in the following state.","An expedition puzzle uses numbered disks and three posts.","A robot arm must rearrange a legal stack configuration.","Find an optimal path in the legal Hanoi state graph."]
        prompt=(f"{intros[t]} Disk 1 is smallest; a larger disk may never be placed on a smaller disk. Move one top disk at a time. "
                f"Start: {hanoi_state_text(start,peg_names)}. Goal: {hanoi_state_text(goal,peg_names)}. {question} Give only the answer.")
        moves=[hanoi_move(a,b,peg_names) for a,b in zip(path,path[1:])]
        steps=[f"Represent each legal configuration by the peg holding each disk; the start-to-goal graph distance is {d}."]
        if len(moves)<=14:
            steps.append("One shortest sequence is: "+"; ".join(moves)+".")
        else:
            steps.append("A shortest sequence begins "+"; ".join(moves[:6])+" and ends "+"; ".join(moves[-4:])+f", totaling {d} moves.")
        steps.append(f"Therefore the requested optimum is {answer}.")
        final=cap_first(answer)+"."
        accepted=[answer,cap_first(answer),final]
        structure={"n_disks":n,"peg_names":peg_names,"start":start,"goal":goal,"query":{"kind":qkind}}
        return base_row(
            family="hanoi_planning",split=split,difficulty=difficulty,
            template_id=f"hanoi_planning.t{t}",prompt=prompt,
            reasoning_steps=steps,compact_state=f"distance={d}; first={moves[0]}",final_answer=final,
            accepted_answers=accepted,answer_type=answer_type,structure=structure,
            solver="breadth_first_search_hanoi_graph",
            verification={"shortest_distance":d,"shortest_path":moves,"unique_optimal_first":len(optimal_first)==1},
            tags=["planning","tower-of-hanoi","state-space","shortest-path"],
        )
    raise RuntimeError("Failed to generate hanoi_planning")

FAMILY_GENERATORS: dict[str, Callable[[random.Random,str,str],dict[str,Any]]] = {
    "epistemic_hats": generate_epistemic_hats,
    "knights_knaves": generate_knights_knaves,
    "prisoner_alibis": generate_prisoner_alibis,
    "mislabeled_boxes": generate_mislabeled_boxes,
    "logic_grid": generate_logic_grid,
    "linear_order": generate_linear_order,
    "rule_deduction": generate_rule_deduction,
    "object_tracking": generate_object_tracking,
    "code_breaking": generate_code_breaking,
    "calendar_reasoning": generate_calendar_reasoning,
    "family_relations": generate_family_relations,
    "shortest_path": generate_shortest_path,
    "arithmetic_constraints": generate_arithmetic_constraints,
    "boolean_switches": generate_boolean_switches,
    "hanoi_planning": generate_hanoi_planning,
}
FAMILY_CODES = {
    "epistemic_hats":"hat","knights_knaves":"knk","prisoner_alibis":"alb","mislabeled_boxes":"box",
    "logic_grid":"grid","linear_order":"ord","rule_deduction":"rule","object_tracking":"track",
    "code_breaking":"code","calendar_reasoning":"time","family_relations":"kin","shortest_path":"path",
    "arithmetic_constraints":"arith","boolean_switches":"bool","hanoi_planning":"hanoi",
}


def percentile(values:list[float],p:float)->float:
    if not values: return 0.0
    vals=sorted(values); idx=min(len(vals)-1,max(0,math.ceil(p*len(vals))-1)); return vals[idx]


def write_jsonl(path:Path, rows:list[dict[str,Any]])->None:
    with path.open("w",encoding="utf-8",newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")


def sha256_file(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024),b""): h.update(chunk)
    return h.hexdigest()


def generate_corpus(output_dir:Path, seed:int, counts:dict[str,int])->dict[str,Any]:
    output_dir.mkdir(parents=True,exist_ok=True)
    data_dir=output_dir/"data"; examples_dir=output_dir/"examples"; reports_dir=output_dir/"reports"
    for d in (data_dir,examples_dir,reports_dir): d.mkdir(parents=True,exist_ok=True)
    rows_by_split={s:[] for s in counts}
    seen_prompts=set(); seen_structures=set()
    family_counter=collections.Counter(); difficulty_counter=collections.Counter(); answer_type_counter=collections.Counter()

    for fidx,(family,generator) in enumerate(FAMILY_GENERATORS.items()):
        print(f"Generating {family}...",flush=True)
        for split,count in counts.items():
            produced=0; outer_attempt=0
            while produced<count:
                outer_attempt+=1
                rng=random.Random(stable_seed(seed,family,split,produced,outer_attempt))
                difficulty=choose_difficulty(split,produced,rng)
                row=generator(rng,split,difficulty)
                ph=signature(row["prompt"])
                sh=row["structural_signature"]
                if ph in seen_prompts or sh in seen_structures:
                    if outer_attempt>1000:
                        raise RuntimeError(f"Deduplication stalled for {family}/{split}/{produced}")
                    continue
                code=FAMILY_CODES[family]
                row["id"]=f"rineya-pr51k-{code}-{split[:3]}-{produced:05d}"
                row["prompt_sha256"]=ph
                # Compact quality assertions.
                assert row["reasoning_steps"] and row["final_answer"].strip()
                assert any(normalize_answer(row["final_answer"])==normalize_answer(a) for a in row["accepted_answers"])
                assert len(row["final_answer"].split())<=16
                seen_prompts.add(ph);seen_structures.add(sh)
                rows_by_split[split].append(row)
                family_counter[(split,family)]+=1;difficulty_counter[(split,difficulty)]+=1;answer_type_counter[row["answer_type"]]+=1
                produced+=1;outer_attempt=0
        print(f"  completed {sum(counts.values())} rows",flush=True)

    # Shuffle deterministically so families are interleaved during training.
    for split,rows in rows_by_split.items():
        random.Random(stable_seed(seed,"shuffle",split)).shuffle(rows)
        write_jsonl(data_dir/f"{split}.jsonl",rows)

    all_rows=[r for rows in rows_by_split.values() for r in rows]
    sample=[]
    for split in rows_by_split:
        by_family=collections.defaultdict(list)
        for r in rows_by_split[split]: by_family[r["family"]].append(r)
        for family in FAMILY_GENERATORS:
            sample.extend(by_family[family][:2 if split=="train" else 1])
    write_jsonl(examples_dir/"sample_75.jsonl",sample)

    prompt_chars=[len(r["prompt"]) for r in all_rows]
    reasoning_counts=[len(r["reasoning_steps"]) for r in all_rows]
    answer_words=[len(r["final_answer"].split()) for r in all_rows]
    stats={
        "dataset":"Rineya Puzzle Reasoning 51K",
        "version":VERSION,
        "seed":seed,
        "total_rows":len(all_rows),
        "split_counts":{s:len(v) for s,v in rows_by_split.items()},
        "family_counts":{s:{f:family_counter[(s,f)] for f in FAMILY_GENERATORS} for s in counts},
        "difficulty_counts":{s:{d:difficulty_counter[(s,d)] for d in ["easy","medium","challenge"] if difficulty_counter[(s,d)]} for s in counts},
        "answer_type_counts":dict(sorted(answer_type_counter.items())),
        "prompt_chars":{"mean":round(statistics.mean(prompt_chars),2),"median":statistics.median(prompt_chars),"p95":percentile(prompt_chars,.95),"max":max(prompt_chars)},
        "reasoning_steps":{"mean":round(statistics.mean(reasoning_counts),2),"median":statistics.median(reasoning_counts),"p95":percentile(reasoning_counts,.95),"max":max(reasoning_counts)},
        "final_answer_words":{"mean":round(statistics.mean(answer_words),2),"median":statistics.median(answer_words),"p95":percentile(answer_words,.95),"max":max(answer_words)},
        "duplicate_prompt_count":len(all_rows)-len({r["prompt_sha256"] for r in all_rows}),
        "duplicate_structure_count":len(all_rows)-len({r["structural_signature"] for r in all_rows}),
        "template_sets":{s:sorted({r["template_id"] for r in rows_by_split[s]}) for s in counts},
    }
    (reports_dir/"dataset_stats.json").write_text(json.dumps(stats,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    manifest={}
    for path in sorted(data_dir.glob("*.jsonl")):
        manifest[path.name]={"bytes":path.stat().st_size,"sha256":sha256_file(path),"rows":sum(1 for _ in path.open(encoding="utf-8"))}
    manifest["sample_75.jsonl"]={"bytes":(examples_dir/"sample_75.jsonl").stat().st_size,"sha256":sha256_file(examples_dir/"sample_75.jsonl"),"rows":len(sample)}
    (data_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n",encoding="utf-8")
    return stats


def parse_args()->argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--output-dir",type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument("--seed",type=int,default=DEFAULT_SEED)
    p.add_argument("--tiny",action="store_true",help="Generate 2 rows per family per split for smoke testing")
    p.add_argument("--count-per-family",type=int,default=None,help="Override every split count per family")
    return p.parse_args()


def main()->None:
    args=parse_args()
    if args.tiny: counts={s:2 for s in SPLIT_COUNTS}
    elif args.count_per_family is not None: counts={s:args.count_per_family for s in SPLIT_COUNTS}
    else: counts=dict(SPLIT_COUNTS)
    stats=generate_corpus(args.output_dir,args.seed,counts)
    print(json.dumps({"total_rows":stats["total_rows"],"split_counts":stats["split_counts"]},indent=2))

if __name__=="__main__":
    main()
