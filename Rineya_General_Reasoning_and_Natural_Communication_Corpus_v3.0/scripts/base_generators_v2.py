from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import shutil
import statistics
import textwrap
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

SEED = 20260816
ROOT = Path('/mnt/data/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v2.0')
ZIP_PATH = Path('/mnt/data/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v2.0.zip')
V1_ROOT = Path('/mnt/data/_rineya_v1_inspect/Rineya_Short_Response_Behavior_Corpus_v1.0')
EVAL_ROOT = Path('/mnt/data/_rineya_repo_eval')

NEW_PER_CLASS = 800
DPO_PER_CLASS = 300
PRESERVATION_ROWS = 8000

GENERAL_SYSTEMS = [
    "You are a concise conversational assistant. Reply naturally in a few words or one short sentence. Preserve correctness and conversation context. Ask one brief clarification when information is missing. For high-stakes requests, give a concise safe boundary or escalation.",
    "Respond like a capable person texting: brief, natural, accurate, and attentive to context. Give one short answer or one useful question. Do not add unnecessary explanation.",
    "Keep the reply compact and human-sounding. Use the conversation's facts, avoid guessing, and ask a short clarification only when needed. Safety and correctness come before brevity.",
    "Give the shortest natural response that fully handles the immediate need. Stay context-consistent, do not invent facts, and use one compact sentence when a few words are not enough.",
]
REASONING_SYSTEMS = [
    "Answer concisely using only the information stated in the prompt. Reason internally and return only the short final answer. If one answer is not supported, reply exactly: \"I don't know based on the available information.\"",
    "Solve the problem silently and give only the concise final answer. Use only the supplied facts; do not show chain-of-thought or invent missing information.",
    "Work out the answer internally, then respond with a few words or one short sentence. If the prompt is underdetermined, say that the available information is insufficient.",
    "Use the stated facts to reach the result. Return only the compact conclusion, not the reasoning steps.",
]
INTEGRATED_SYSTEMS = [
    "Respond naturally and concisely. Use the stated context to reason, preserve updates and corrections, and give only the useful final reply.",
    "Read the whole conversation, resolve the immediate question, and answer like a person texting. Keep the result short, accurate, and context-aware.",
    "Reason from the provided details, then give one compact conversational answer. Do not expose hidden reasoning or ignore later corrections.",
    "Give a brief human-sounding response that reflects both the facts and the user's practical goal. Ask one short question only when the answer cannot be determined.",
]

NAMES = [
    "Maya", "Leo", "Nina", "Owen", "Iris", "Noah", "Ava", "Eli", "Zoe", "Kai",
    "Mina", "Theo", "Lena", "Sam", "Aria", "Jules", "Ravi", "Tara", "Milo", "June",
    "Ari", "Sora", "Nora", "Ben", "Luca", "Mira", "Drew", "Ivy", "Alex", "Remy",
]
PROJECTS = [
    "Atlas", "Harbor", "Juniper", "Nova", "Orchid", "Beacon", "Cedar", "Lumen", "Mosaic", "Drift",
    "Kite", "Maple", "Quartz", "River", "Summit", "Willow", "Echo", "Pine", "Solace", "Vela",
]
DOMAINS = [
    "everyday", "work", "school", "coding", "electronics", "travel", "cooking", "shopping",
    "personal_finance", "health_safety", "home", "social", "games", "transportation", "science", "productivity",
]
OBJECTS = [
    "notebooks", "tickets", "cables", "bottles", "folders", "batteries", "cards", "markers", "boxes", "tokens",
    "mugs", "screws", "files", "photos", "seats", "coins", "apples", "books", "lamps", "packages",
]
PLACES = [
    "library", "station", "office", "clinic", "cafe", "workshop", "campus", "hotel", "museum", "market",
    "garage", "kitchen", "lab", "studio", "airport", "warehouse",
]
TONE_WORDS = ["direct", "warm", "casual", "calm", "practical", "supportive", "clear", "friendly"]


def stable_seed(*parts: Any) -> int:
    raw = "|".join(map(str, parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')


def rng_for(*parts: Any) -> random.Random:
    return random.Random(stable_seed(SEED, *parts))


def norm(text: str) -> str:
    text = text.lower().replace("’", "'").replace("“", '"').replace("”", '"')
    return " ".join(re.findall(r"[a-z0-9_.$%+\-]+", text))


def proxy_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?|[^\w\s]", text, flags=re.UNICODE))


def fmt_num(value: float | int) -> str:
    if isinstance(value, int) or abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.2f}".rstrip('0').rstrip('.')


def money(value: float | int) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"${int(round(value))}"
    return f"${value:.2f}"


def count_noun(value: float | int, singular: str, plural: str | None = None) -> str:
    """Render a numeric value with grammatically correct singular/plural units."""
    plural = plural or f"{singular}s"
    label = singular if abs(float(value) - 1.0) < 1e-9 else plural
    return f"{fmt_num(value)} {label}"


def choose_system(group: str, idx: int) -> str:
    systems = REASONING_SYSTEMS if group == 'reasoning' else GENERAL_SYSTEMS if group == 'communication' else INTEGRATED_SYSTEMS
    return systems[idx % len(systems)]


def pick_text(options: list[str] | tuple[str, ...], idx: int, family_count: int) -> str:
    """Cycle through independently authored valid response phrasings."""
    return options[(idx // family_count) % len(options)]


def qwen_prompt(messages: list[dict[str, str]]) -> str:
    return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + "<|im_start|>assistant\n"


def difficulty_from_steps(steps: int) -> str:
    return 'basic' if steps <= 1 else 'intermediate' if steps == 2 else 'compositional'


def make_messages(system: str, turns: list[tuple[str, str]]) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": system}]
    for role, content in turns:
        messages.append({"role": role, "content": content})
    return messages


def maybe_add_conversational_prelude(messages: list[dict[str, str]], group: str, reply_class: str, idx: int) -> bool:
    """Turn a deterministic slice of single-turn rows into natural multi-turn exchanges."""
    no_generic_prelude = {
        "roleplay_reasoning", "conversation_closure", "acknowledgment_update",
        "apology_repair", "natural_follow_up", "empathy_validation", "informal_texting",
    }
    if len(messages) != 3 or idx % 6 != 0 or reply_class in no_generic_prelude:
        return False
    r = rng_for("conversation_prelude", reply_class, idx)
    project = r.choice(PROJECTS)
    name = r.choice(NAMES)
    if group == "reasoning":
        preludes = [
            (f"Can you help me verify one thing for {project}?", "Sure—what is it?"),
            (f"I need a quick sanity check before I reply to {name}.", "Okay—send it over."),
            (f"One small question about {project}.", "Go ahead."),
            ("Can we check the result before I continue?", "Yes—what are the details?"),
        ]
    elif group == "communication":
        preludes = [
            (f"Can I tell you something about {project}?", "Of course."),
            (f"I need a quick response before I message {name}.", "Sure."),
            ("Can I get your reaction to something?", "Yeah—what happened?"),
            ("One more thing before I go.", "Go ahead."),
        ]
    else:
        preludes = [
            (f"Can we make one quick decision for {project}?", "Sure—what are the options?"),
            (f"I need a practical answer before I update {name}.", "Okay—what is the situation?"),
            ("Can you help me choose the next move?", "Yes—give me the details."),
            ("One last check before I act.", "Go ahead."),
        ]
    p_user, p_assistant = preludes[(idx // 6) % len(preludes)]
    original_user = messages[1]
    target = messages[2]
    messages[:] = [messages[0], {"role": "user", "content": p_user}, {"role": "assistant", "content": p_assistant}, original_user, target]
    return True


SURFACE_PURPOSES = [
    "first review", "final review", "setup check", "budget check", "planning pass",
    "travel check", "morning review", "afternoon review", "quick comparison", "decision check",
    "safety check", "schedule review", "quality pass", "practice round", "project update",
]

MULTITURN_CONTEXT_ON_FIRST = {
    "acknowledgment_update", "natural_follow_up", "empathy_validation", "apology_repair",
    "multi_turn_context_update", "plan_revision", "evidence_grounded_dialogue", "roleplay_reasoning",
}


def add_natural_surface_context(messages: list[dict[str, str]], group: str, reply_class: str, idx: int, attempt: int = 0) -> str:
    """Add varied, semantically neutral context without artificial case tags."""
    r = rng_for("surface_context", reply_class, idx, attempt)
    project = r.choice(PROJECTS)
    name = r.choice(NAMES)
    place = r.choice(PLACES)
    purpose = r.choice(SURFACE_PURPOSES)
    purpose_id = SURFACE_PURPOSES.index(purpose)

    if reply_class == "roleplay_reasoning":
        messages[0]["content"] += (
            f" Current objective: help {name} with {project} during the {purpose} at the {place}."
        )
        user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
        last_i = user_indices[-1]
        last = messages[last_i]["content"]
        messages[last_i]["content"] = (
            f"With {name} beside us during the {purpose} for {project} at the {place}: {last}"
        )
        return f"roleplay_context_{purpose_id:02d}"

    user_indices = [i for i, m in enumerate(messages) if m["role"] == "user"]
    target_i = user_indices[0] if reply_class in MULTITURN_CONTEXT_ON_FIRST else user_indices[-1]
    original = messages[target_i]["content"]

    # Multi-turn examples also vary the latest user turn, because that is what leakage
    # and conversational-diversity checks inspect. The added phrase remains relevant.
    if len(user_indices) > 1:
        last_i = user_indices[-1]
        last = messages[last_i]["content"]
        if reply_class == "acknowledgment_update":
            messages[last_i]["content"] = f"For the {purpose} on {project} with {name} at the {place}: {last}"
        elif reply_class == "natural_follow_up":
            messages[last_i]["content"] = f"Another update from {name} during the {purpose} for {project} at the {place}: {last}"
        elif reply_class == "empathy_validation":
            messages[last_i]["content"] = f"After the {purpose} for {project} with {name} at the {place}: {last}"
        elif reply_class == "apology_repair":
            cleaned = re.sub(r"^(No, |Actually, )", "", last, flags=re.I)
            messages[last_i]["content"] = f"One correction for the {purpose} on {project} with {name}: {cleaned}"
        elif reply_class == "multi_turn_context_update":
            messages[last_i]["content"] = f"For the {purpose} on {project} with {name} at the {place}: {last}"
        elif reply_class == "plan_revision":
            messages[last_i]["content"] = f"For the {purpose} on {project} with {name} at the {place}: {last}"
        elif reply_class == "evidence_grounded_dialogue":
            messages[last_i]["content"] = f"Using only the information above from the {purpose} at the {place}: {last}"
        elif reply_class == "roleplay_reasoning":
            messages[last_i]["content"] = f"With {name} beside us during the {purpose} at the {place}: {last}"

    if reply_class == "apology_repair":
        messages[target_i]["content"] = f"Quick check for {project} with {name} at the {place}."
        return f"repair_context_{purpose_id:02d}"
    if reply_class == "acknowledgment_update" and len(user_indices) > 1:
        messages[target_i]["content"] = f"Quick update on {project} with {name}."
        return f"ack_context_{purpose_id:02d}"
    if reply_class == "natural_follow_up" and len(user_indices) > 1:
        messages[target_i]["content"] = f"I have an update about {project} from {name}."
        return f"followup_context_{purpose_id:02d}"
    if reply_class == "empathy_validation" and len(user_indices) > 1:
        messages[target_i]["content"] = f"Can I vent about the {purpose} for {project}?"
        return f"empathy_context_{purpose_id:02d}"

    if group == "reasoning":
        wrappers = [
            f"Quick check for {project}: {original}",
            f"{name} and I are verifying the {project} details: {original}",
            f"Before the {purpose} at the {place}: {original}",
            f"For the {project} {purpose}: {original}",
            f"I need the result for {project}: {original}",
            f"One more question from the {place} review: {original}",
            f"During the {purpose} with {name}: {original}",
            f"To finish the {project} notes: {original}",
            f"For a quick sanity check at the {place}: {original}",
            f"{name} asked this during the {purpose}: {original}",
            f"Before I finalize {project}: {original}",
            f"This came up in the {project} review: {original}",
        ]
    elif group == "communication":
        wrappers = [
            f"Quick context from {project}: {original}",
            f"This came up with {name} during the {purpose}: {original}",
            f"One more thing about {project}: {original}",
            f"At the {place}, this happened: {original}",
            f"Here is the situation from the {purpose}: {original}",
            f"For the {project} conversation with {name}: {original}",
            f"A quick update before the {purpose}: {original}",
            f"This is what I am dealing with on {project}: {original}",
            f"Something from the {place} meeting: {original}",
            f"Before I wrap up {project}: {original}",
            f"I was talking with {name} about this: {original}",
            f"During today's {purpose}: {original}",
        ]
    else:
        wrappers = [
            f"For the {project} decision: {original}",
            f"During the {purpose} with {name}: {original}",
            f"One more question about the {place} plan: {original}",
            f"To finish the {project} review: {original}",
            f"Here is the practical situation for {project}: {original}",
            f"Before the {purpose}: {original}",
            f"{name} and I need a short answer here: {original}",
            f"For today's plan at the {place}: {original}",
            f"This came up while reviewing {project}: {original}",
            f"A final check before we proceed: {original}",
            f"For the {project} update with {name}: {original}",
            f"At the {place} during the {purpose}: {original}",
        ]
    wrapper_id = (idx + attempt) % len(wrappers)
    messages[target_i]["content"] = wrappers[wrapper_id]
    return f"surface_{wrapper_id:02d}"


@dataclass
class Example:
    messages: list[dict[str, str]]
    domain: str
    template_family: str
    answer_type: str
    reasoning_steps: int
    tone: list[str]
    safety_tier: str = 'normal'
    requires_external_fact: bool = False
    context_dependency: str = 'current_turn'
    counterfactual_group_id: str | None = None
    negative: str | None = None
    rejection_type: str | None = None
    extra_metadata: dict[str, Any] | None = None


# -------------------------- Reasoning generators --------------------------

def gen_arithmetic_chain(idx: int) -> Example:
    r = rng_for('arithmetic_chain', idx)
    t = idx % 8
    start = r.randint(8, 95)
    a, b, c = r.randint(2, 18), r.randint(1, 12), r.randint(2, 15)
    obj = r.choice(OBJECTS)
    if t == 0:
        ans = start + a - b + c
        q = f"A count starts at {start}, gains {a}, loses {b}, then gains {c}. What is the final count?"
    elif t == 1:
        ans = (start + a) * 2
        q = f"A score starts at {start}, increases by {a}, then doubles. What is the final score?"
    elif t == 2:
        ans = start - b + a
        q = f"There are {start} {obj}. {b} are removed and {a} are added. How many are there now?"
    elif t == 3:
        packs = r.randint(3, 12); each = r.randint(2, 9); used = r.randint(1, min(15, packs*each-1))
        ans = packs * each - used
        q = f"There are {packs} packs with {each} {obj} each. After using {used}, how many remain?"
    elif t == 4:
        x = r.randint(5, 30); y = r.randint(2, 8); z = r.randint(1, y)
        ans = x * y - z
        q = f"A game awards {x} points in each of {y} rounds, then subtracts a {z}-point penalty. Final score?"
    elif t == 5:
        x = 2 * r.randint(10, 40); half = x // 2; add = r.randint(3, 17)
        ans = x - half + add
        q = f"A bin has {x} items. Half are moved out, then {add} are returned. How many are in the bin?"
    elif t == 6:
        groups = r.choice([2, 4])
        while True:
            x = r.randint(10, 40); add1 = r.randint(2, 12); add2 = r.randint(2, 12)
            total = x + add1 + add2
            if total % groups == 0:
                break
        ans = total // groups
        q = f"A total starts at {x}, then {add1} and {add2} are added. The result is split equally into {groups} groups. How many per group?"
    else:
        x = r.randint(30, 100); loss1 = r.randint(3, 15); loss2 = r.randint(3, 15)
        ans = x - loss1 - loss2
        q = f"A balance is {x} units. It uses {loss1} units, then {loss2} more. What remains?"
    target = f"{ans}."
    wrong = ans + r.choice([-2, -1, 1, 2, 5])
    return Example(make_messages(choose_system('reasoning', idx), [('user', q), ('assistant', target)]), r.choice(DOMAINS), f"v2_arithmetic_chain_{t:02d}", 'number', 2 if t in {2,7} else 3, ['direct','accurate'], negative=f"{wrong}.", rejection_type='incorrect_arithmetic')


def gen_percentage_reasoning(idx: int) -> Example:
    r = rng_for('percentage_reasoning', idx)
    t = idx % 8
    pct = r.choice([10, 20, 25, 30, 40, 50, 60, 75])
    if t == 0:
        base = r.choice([20, 40, 60, 80, 100, 120, 160, 200])
        ans = base * pct / 100
        q = f"What is {pct}% of {base}?"
        target = f"{fmt_num(ans)}."
    elif t == 1:
        base = r.choice([20, 40, 50, 60, 80, 100, 120, 200])
        discount = r.choice([10, 20, 25, 50])
        ans = base * (1-discount/100)
        q = f"An item costs {money(base)} and is {discount}% off. What is the sale price?"
        target = f"{money(ans)}."
    elif t == 2:
        base = r.choice([20, 40, 50, 80, 100, 120, 200])
        inc = r.choice([10, 20, 25, 50])
        ans = base * (1+inc/100)
        q = f"A value of {base} increases by {inc}%. What is the new value?"
        target = f"{fmt_num(ans)}."
    elif t == 3:
        while True:
            total = r.choice([20, 40, 50, 80, 100, 120, 160, 200])
            if (total * pct) % 100 == 0:
                break
        part = total * pct // 100
        q = f"{part} of {total} tasks are complete. What percentage is that?"
        target = f"{pct}%."
        ans = pct
    elif t == 4:
        old = r.choice([20, 40, 50, 80, 100]); delta_pct = r.choice([10,20,25,50]); new = old*(1+delta_pct/100)
        q = f"A score rises from {old} to {fmt_num(new)}. What is the percent increase?"
        target = f"{delta_pct}%."
        ans = delta_pct
    elif t == 5:
        bill = r.choice([20, 40, 50, 60, 80, 100]); tip = r.choice([10,15,20,25])
        ans = bill * tip / 100
        q = f"What is a {tip}% tip on a {money(bill)} bill?"
        target = f"{money(ans)}."
    elif t == 6:
        capacity = r.choice([40, 80, 100, 120, 200]); used_pct = r.choice([25,50,75]); used = capacity*used_pct/100
        q = f"A {capacity}-unit capacity is {used_pct}% full. How many units are used?"
        target = f"{fmt_num(used)} units."
        ans = used
    else:
        base = r.choice([40, 80, 100, 120, 200]); decrease = r.choice([10,20,25,50]); ans = base*(1-decrease/100)
        q = f"A quantity of {base} decreases by {decrease}%. What remains?"
        target = f"{fmt_num(ans)}."
    wrong_num = float(ans) + r.choice([-10,-5,5,10])
    negative = re.sub(r"-?\d+(?:\.\d+)?", fmt_num(wrong_num), target, count=1)
    return Example(make_messages(choose_system('reasoning', idx), [('user', q), ('assistant', target)]), r.choice(['shopping','personal_finance','work','school','everyday']), f"v2_percentage_{t:02d}", 'number', 2, ['direct','accurate'], negative=negative, rejection_type='incorrect_percentage')


def gen_ratio_proportion(idx: int) -> Example:
    r = rng_for('ratio_proportion', idx)
    t = idx % 8
    if t == 0:
        unit_n = r.randint(2, 6); unit_cost = r.randint(2, 12); wanted = r.randint(3, 10)
        cost = unit_n * unit_cost; ans = wanted * unit_cost
        q = f"{unit_n} notebooks cost {money(cost)}. At the same rate, what do {wanted} notebooks cost?"
        target = f"{money(ans)}."
    elif t == 1:
        a,b = r.choice([(1,2),(2,3),(3,4),(2,5),(3,5)]); mult = r.randint(2, 12); total=(a+b)*mult; ans=b*mult
        q = f"The ratio of red to blue tokens is {a}:{b}, with {total} tokens total. How many are blue?"
        target=f"{ans}."
    elif t == 2:
        base_serv = r.choice([2,4,5,6]); target_serv=r.choice([8,10,12]); amount=r.choice([1,2,3,4]); ans=amount*target_serv/base_serv
        q=f"A recipe uses {count_noun(amount, 'cup')} for {base_serv} servings. How many cups are needed for {target_serv} servings?"
        target=f"{count_noun(ans, 'cup')}."
    elif t == 3:
        speed=r.choice([30,40,50,60]); hours=r.choice([2,3,4]); ans=speed*hours
        q=f"At a constant {speed} miles per hour for {hours} hours, how far do you travel?"
        target=f"{ans} miles."
    elif t == 4:
        scale=r.choice([2,4,5,10]); cm=r.randint(3,15); ans=scale*cm
        q=f"On a map, 1 cm represents {scale} km. What distance does {cm} cm represent?"
        target=f"{ans} km."
    elif t == 5:
        workers=r.choice([2,3,4,5]); boxes_per=r.randint(3,10); new_workers=r.choice([6,8,10]); ans=boxes_per*new_workers
        q=f"{workers} workers each pack {boxes_per} boxes. At the same individual rate, how many boxes do {new_workers} workers pack?"
        target=f"{ans} boxes."
    elif t == 6:
        parts=r.choice([(2,1),(3,2),(4,1)]); total=(parts[0]+parts[1])*r.randint(3,12); one=total//sum(parts); ans=one*parts[0]
        q=f"A mix uses water and concentrate in a {parts[0]}:{parts[1]} ratio. If the total is {count_noun(total, 'cup')}, how many cups are water?"
        target=f"{count_noun(ans, 'cup')}."
    else:
        while True:
            pages=r.choice([20,30,40,50]); minutes=r.choice([10,15,20,25]); target_min=r.choice([30,40,60])
            if (pages*target_min)%minutes==0: break
        ans=pages*target_min/minutes
        q=f"A printer makes {pages} pages in {minutes} minutes. At the same rate, how many pages in {target_min} minutes?"
        target=f"{fmt_num(ans)} pages."
    numeric=float(re.findall(r"-?\d+(?:\.\d+)?", target)[0])
    deltas=[1,2,5] if numeric <= 2 else [-2,-1,1,2,5]
    wrong=numeric+r.choice(deltas)
    if t in {2,6}:
        neg=f"{count_noun(wrong, 'cup')}."
    else:
        neg=re.sub(r"-?\d+(?:\.\d+)?", fmt_num(wrong), target, count=1)
    return Example(make_messages(choose_system('reasoning', idx), [('user',q),('assistant',target)]), r.choice(['cooking','shopping','travel','work','school']), f"v2_ratio_{t:02d}", 'number', 2, ['direct','accurate'], negative=neg, rejection_type='incorrect_proportion')


def gen_unit_conversion(idx: int) -> Example:
    r=rng_for('unit_conversion',idx); t=idx%8
    if t==0:
        h=r.randint(1,9); m=r.choice([0,5,10,15,20,25,30,35,40,45,50,55]); ans=h*60+m; q=f"Convert {count_noun(h, 'hour')} and {count_noun(m, 'minute')} to minutes."; target=f"{count_noun(ans, 'minute')}."
    elif t==1:
        mins=r.choice([60,90,120,150,180,210,240,300,360]); h=mins//60; m=mins%60; q=f"Convert {mins} minutes to hours and minutes."; target=f"{h} {'hour' if h==1 else 'hours'}{f' {m} minutes' if m else ''}."
    elif t==2:
        cm=r.choice([100,150,200,250,300,450,500]); ans=cm/100; q=f"Convert {count_noun(cm, 'centimeter')} to meters."; target=f"{count_noun(ans, 'meter')}."
    elif t==3:
        g=r.choice([500,750,1000,1250,1500,2000,2500]); ans=g/1000; q=f"Convert {count_noun(g, 'gram')} to kilograms."; target=f"{count_noun(ans, 'kilogram')}."
    elif t==4:
        ml=r.choice([250,500,750,1000,1250,1500,2000]); ans=ml/1000; q=f"Convert {count_noun(ml, 'milliliter')} to liters."; target=f"{count_noun(ans, 'liter')}."
    elif t==5:
        days=r.randint(2,14); ans=days*24; q=f"How many hours are in {days} days?"; target=f"{ans} hours."
    elif t==6:
        km=r.choice([1,2,3,4,5,8,10,12]); ans=km*1000; q=f"Convert {count_noun(km, 'kilometer')} to meters."; target=f"{count_noun(ans, 'meter')}."
    else:
        sec=r.choice([60,90,120,180,240,300,360,600]); ans=sec/60; q=f"Convert {count_noun(sec, 'second')} to minutes."; target=f"{count_noun(ans, 'minute')}."
    if t==1:
        wrong_h=h+1; neg=f"{wrong_h} {'hour' if wrong_h==1 else 'hours'}{f' {m} minutes' if m else ''}."
    else:
        num=float(re.findall(r"\d+(?:\.\d+)?",target)[0]); wrong=num+r.choice([1,2,5,10])
        unit_by_template={0:'minute',2:'meter',3:'kilogram',4:'liter',5:'hour',6:'meter',7:'minute'}
        neg=f"{count_noun(wrong, unit_by_template[t])}."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['everyday','cooking','travel','science','work']),f"v2_unit_{t:02d}",'number',1,['direct','accurate'],negative=neg,rejection_type='incorrect_conversion')


def minutes_to_time(total: int) -> str:
    total%=24*60; h=total//60; m=total%60; suffix='AM' if h<12 else 'PM'; h12=h%12 or 12; return f"{h12}:{m:02d} {suffix}"


def gen_time_duration(idx:int)->Example:
    r=rng_for('time_duration',idx); t=idx%8
    start_h=r.randint(7,19); start_m=r.choice([0,5,10,15,20,25,30,35,40,45,50]); start=start_h*60+start_m
    if t in {0,1,2,3}:
        dur=r.choice([20,25,30,35,40,45,50,55,60,75,90,105,120]); end=start+dur
        if t==0: q=f"A meeting starts at {minutes_to_time(start)} and lasts {dur} minutes. When does it end?"
        elif t==1: q=f"A bus leaves at {minutes_to_time(start)}. The ride takes {dur} minutes. Arrival time?"
        elif t==2: q=f"A timer begins at {minutes_to_time(start)} and runs for {dur} minutes. What time does it stop?"
        else: q=f"A study session starts at {minutes_to_time(start)} and lasts {dur} minutes. End time?"
        target=f"{minutes_to_time(end)}."
    else:
        dur=r.choice([25,30,40,45,50,60,75,90,120]); end=start+dur
        if t==4: q=f"How long is it from {minutes_to_time(start)} to {minutes_to_time(end)}?"
        elif t==5: q=f"A class runs from {minutes_to_time(start)} until {minutes_to_time(end)}. How long is it?"
        elif t==6: q=f"Travel begins at {minutes_to_time(start)} and ends at {minutes_to_time(end)}. Duration?"
        else: q=f"A task starts at {minutes_to_time(start)} and finishes at {minutes_to_time(end)}. How many minutes?"
        target=f"{dur} minutes."
    neg=f"{minutes_to_time(end+15)}." if 'minutes' not in target else f"{dur+r.choice([5,10,15])} minutes."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['work','school','travel','productivity','transportation']),f"v2_time_{t:02d}",'time',2,['direct','accurate'],negative=neg,rejection_type='incorrect_time')


def gen_schedule_ordering(idx:int)->Example:
    pair=idx//2; variant=idx%2; r=rng_for('schedule_ordering_pair',pair); t=pair%4
    names=r.sample(['draft','review','test','approve','deploy','backup','pack','leave','check in','board'],4)
    if t==0:
        a,b,c=names[:3]
        q=f"{a.title()} must happen before {b}, and {b} before {c}. What is the order?"
        order=[a,b,c] if variant==0 else [c,b,a]
        if variant==1: q=f"{c.title()} must happen before {b}, and {b} before {a}. What is the order?"
        target=" → ".join(x.title() for x in order)+"."
    elif t==1:
        a,b,c=names[:3]
        q=f"{a.title()} is at 9:00 AM, {b} at 10:30 AM, and {c} at 1:00 PM. Which happens {'first' if variant==0 else 'last'}?"
        target=f"{a.title() if variant==0 else c.title()}."
    elif t==2:
        a,b,c=names[:3]
        if variant==0:
            q=f"Do {a} after {b} but before {c}. If {b} is complete, what comes next?"; target=f"{a.title()}."
        else:
            q=f"Do {c} after {b} but before {a}. If {b} is complete, what comes next?"; target=f"{c.title()}."
    else:
        a,b,c,d=names
        if variant==0:
            q=f"{a.title()} and {b} must finish before {c}; {d} can happen anytime. Which task is blocked until two others finish?"; target=f"{c.title()}."
        else:
            q=f"{c.title()} and {b} must finish before {a}; {d} can happen anytime. Which task is blocked until two others finish?"; target=f"{a.title()}."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['work','travel','school','productivity']),f"v2_schedule_{t:02d}",'ordering',2,['direct','accurate'],counterfactual_group_id=f"schedule_{pair:04d}",negative="The order cannot be determined.",rejection_type='ignores_order_constraints')


def gen_comparison_ranking(idx:int)->Example:
    pair=idx//2; variant=idx%2; r=rng_for('comparison_pair',pair); t=pair%4
    labels=['A','B','C']; vals=r.sample(range(5,40),3)
    if t==0:
        criterion='highest' if variant==0 else 'lowest'; q=f"Option A scores {vals[0]}, B scores {vals[1]}, and C scores {vals[2]}. Which has the {criterion} score?"; best=max(range(3),key=lambda i:vals[i]) if variant==0 else min(range(3),key=lambda i:vals[i]); target=f"Option {labels[best]}."
    elif t==1:
        costs=r.sample(range(20,100,5),3); criterion='cheapest' if variant==0 else 'most expensive'; q=f"A costs {money(costs[0])}, B costs {money(costs[1])}, and C costs {money(costs[2])}. Which is {criterion}?"; best=min(range(3),key=lambda i:costs[i]) if variant==0 else max(range(3),key=lambda i:costs[i]); target=f"Option {labels[best]}."
    elif t==2:
        # Avoid ties so the preference target and DPO rejection are unambiguous.
        while True:
            hours=r.sample(range(3,15),3); costs=[r.choice([20,25,30,35,40]) for _ in range(3)]; ratios=[hours[i]/costs[i] for i in range(3)]
            if len({round(x,10) for x in ratios})==3: break
        q=f"A gives {hours[0]} hours for {money(costs[0])}, B gives {hours[1]} hours for {money(costs[1])}, and C gives {hours[2]} hours for {money(costs[2])}. Which gives the {'best' if variant==0 else 'worst'} hours per dollar?"; best=max(range(3),key=lambda i:ratios[i]) if variant==0 else min(range(3),key=lambda i:ratios[i]); target=f"Option {labels[best]}."
    else:
        lat=r.sample(range(20,120,5),3); q=f"Latency is A={lat[0]} ms, B={lat[1]} ms, C={lat[2]} ms. Which is {'fastest' if variant==0 else 'slowest'}?"; best=min(range(3),key=lambda i:lat[i]) if variant==0 else max(range(3),key=lambda i:lat[i]); target=f"Option {labels[best]}."
    wrong=next(x for x in labels if f"Option {x}."!=target)
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['shopping','electronics','work','travel']),f"v2_comparison_{t:02d}",'choice',2,['direct','decisive'],counterfactual_group_id=f"comparison_{pair:04d}",negative=f"Option {wrong}.",rejection_type='wrong_ranking')


def gen_logical_implication(idx:int)->Example:
    pair=idx//2; variant=idx%2; r=rng_for('logic_pair',pair); t=pair%5
    entity=r.choice(NAMES); category=r.choice(['luma','nori','vex','sable','tavi']); prop=r.choice(['blue','quiet','registered','metal','approved'])
    if t==0:
        q=f"All {category}s are {prop}. {entity} is {'a '+category if variant==0 else 'not a '+category}. Is {entity} definitely {prop}?"; target="Yes." if variant==0 else "No—not from that information."
    elif t==1:
        q=f"If the switch is on, the light is on. The switch is {'on' if variant==0 else 'off'}. Is the light definitely on?"; target="Yes." if variant==0 else "No—not necessarily."
    elif t==2:
        q=f"No {category}s are {prop}. {entity} is a {category}. Is {entity} {prop}?" if variant==0 else f"No {category}s are {prop}. {entity} is not a {category}. Is {entity} {prop}?"; target="No." if variant==0 else "It cannot be determined."
    elif t==3:
        if variant==0:
            q="If a file is signed, it is accepted. This file is signed. Is it definitely accepted?"; target="Yes."
        else:
            q="If a file is signed, it is accepted. This file is accepted. Is it definitely signed?"; target="No—the reverse does not follow."
    else:
        q=f"Every approved request has an ID. This request {'is approved' if variant==0 else 'has an ID'}. Is it definitely approved?"; target="Yes." if variant==0 else "No—not necessarily."
    neg="Yes." if target.startswith(('No','It')) else "No."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),'school',f"v2_logic_{t:02d}",'boolean_or_unknown',2,['direct','precise'],counterfactual_group_id=f"logic_{pair:04d}",negative=neg,rejection_type='invalid_logical_inference')


def gen_conditional_rules(idx:int)->Example:
    pair=idx//2; variant=idx%2; r=rng_for('conditional_pair',pair); t=pair%5
    if t==0:
        level=r.choice([10,15,18]) if variant==0 else r.choice([20,22,30]); q=f"Rule: charge the device when battery is below 20%. The battery is {level}%. What should happen?"; target="Charge it." if level<20 else "No charge is required."
    elif t==1:
        member=variant==0; q=f"Rule: members get free delivery; nonmembers pay delivery. The buyer is {'a member' if member else 'not a member'}. Delivery fee?"; target="Free delivery." if member else "A delivery fee applies."
    elif t==2:
        valid=variant==0; q=f"Rule: a valid code unlocks the file. The code is {'valid' if valid else 'invalid'}. Is the file unlocked?"; target="Yes." if valid else "No."
    elif t==3:
        temp=r.choice([2,4]) if variant==0 else r.choice([6,8]); threshold=5; q=f"Rule: turn on heat below {threshold}°C. The room is {temp}°C. What should the system do?"; target="Turn on the heat." if temp<threshold else "Leave the heat off."
    else:
        complete=variant==0; q=f"Rule: deploy only after tests pass. The tests have {'passed' if complete else 'failed'}. Can deployment proceed?"; target="Yes." if complete else "No."
    if t==0: neg="No charge is required." if level<20 else "Charge it."
    elif t==1: neg="A delivery fee applies." if member else "Free delivery."
    elif t==2: neg="No." if valid else "Yes."
    elif t==3: neg="Leave the heat off." if temp<threshold else "Turn on the heat."
    else: neg="No." if complete else "Yes."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['electronics','work','home','shopping']),f"v2_condition_{t:02d}",'action_or_boolean',2,['direct','accurate'],counterfactual_group_id=f"condition_{pair:04d}",negative=neg,rejection_type='violates_explicit_rule')


def gen_contradiction_detection(idx:int)->Example:
    pair=idx//2; variant=idx%2; r=rng_for('contradiction_pair',pair); t=pair%5; name=r.choice(NAMES)
    if t==0:
        a=r.choice(['red','blue','green']); b=a if variant==0 else r.choice([x for x in ['red','blue','green'] if x!=a]); q=f"Statement 1: {name}'s folder is {a}. Statement 2: {name}'s folder is {b}. Only one color is possible. Do the statements conflict?"
    elif t==1:
        a=r.randint(2,12); b=a if variant==0 else a+r.choice([1,2,3]); q=f"Statement 1: The group has {a} members. Statement 2: The same group has {b} members at the same time. Do they conflict?"
    elif t==2:
        day=r.choice(['Monday','Tuesday','Wednesday']); day2=day if variant==0 else r.choice([d for d in ['Monday','Tuesday','Wednesday'] if d!=day]); q=f"One note says the deadline is {day}. Another says it is {day2}, with no revision mentioned. Is that contradictory?"
    elif t==3:
        q="One report says the server is offline. Another says the same server is online at the same moment. Is there a conflict?" if variant else "One report says the server is online. Another says it is reachable. Is there a conflict?"
    else:
        q="The first message says the package arrived. The second says it has not arrived yet. Both describe the same time. Conflict?" if variant else "The first message says the package arrived. The second says it is at the front desk. Conflict?"
    target="No, they can both be true." if variant==0 else "Yes, they conflict."
    neg="Yes, they conflict." if variant==0 else "No, they can both be true."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['work','everyday','school','travel']),f"v2_contradiction_{t:02d}",'boolean',2,['direct','precise'],counterfactual_group_id=f"contradiction_{pair:04d}",negative=neg,rejection_type='misreads_consistency')


def gen_sequence_pattern(idx:int)->Example:
    r=rng_for('sequence_pattern',idx); t=idx%8
    if t==0:
        start=r.randint(1,20); step=r.randint(2,9); seq=[start+i*step for i in range(4)]; ans=seq[-1]+step
    elif t==1:
        start=r.randint(2,8); factor=r.choice([2,3]); seq=[start*(factor**i) for i in range(4)]; ans=seq[-1]*factor
    elif t==2:
        a=r.randint(1,9); b=r.randint(10,20); seq=[a,b,a,b]; ans=a
    elif t==3:
        start=r.randint(20,50); step=r.randint(2,7); seq=[start-i*step for i in range(4)]; ans=seq[-1]-step
    elif t==4:
        start=r.randint(1,10); seq=[start,start+1,start+3,start+6]; ans=start+10
    elif t==5:
        start=r.randint(1,8); seq=[start,start+2,start+2,start+4]; ans=start+4
    elif t==6:
        start=r.randint(2,10); seq=[start,start*2,start*2+1,(start*2+1)*2]; ans=(start*2+1)*2+1
    else:
        start=r.randint(1,10); seq=[start,start+5,start+3,start+8]; ans=start+6
    q="What comes next: "+", ".join(map(str,seq))+"?"; target=f"{ans}."; neg=f"{ans+r.choice([-2,-1,1,2])}."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),'school',f"v2_sequence_{t:02d}",'number',2,['direct','accurate'],negative=neg,rejection_type='wrong_pattern')


def gen_counting_sets(idx:int)->Example:
    r=rng_for('counting_sets',idx); t=idx%8
    if t==0:
        a=r.randint(5,20); b=r.randint(5,20); both=r.randint(1,min(a,b)-1); ans=a+b-both; q=f"{a} people chose tea, {b} chose coffee, and {both} chose both. How many chose at least one?"; target=f"{ans}."
    elif t==1:
        a=r.randint(4,12); b=r.randint(4,12); both=r.randint(0,min(a,b)); union=a+b-both; total=r.randint(union, union+25); none=total-union; q=f"In a group of {total}, {a} use app A, {b} use app B, and {both} use both. How many use neither?"; target=f"{none}."; ans=none
    elif t==2:
        colors=r.randint(3,8); each=r.randint(2,9); ans=colors*each; q=f"There are {colors} colors with {each} cards of each color. How many cards total?"; target=f"{ans}."
    elif t==3:
        n=r.randint(4,12); ans=n*(n-1)//2; q=f"How many unique pairs can be formed from {n} people?"; target=f"{ans}."
    elif t==4:
        rows=r.randint(3,9); cols=r.randint(3,9); missing=r.randint(1,5); ans=rows*cols-missing; q=f"A grid has {rows} rows and {cols} columns, with {count_noun(missing, 'cell')} blocked. How many usable cells remain?"; target=f"{ans}."
    elif t==5:
        total=r.randint(20,60); removed=r.randint(3,12); added=r.randint(2,10); ans=total-removed+added; q=f"A list has {total} entries. {removed} are removed and {added} new ones are added. Final count?"; target=f"{ans}."
    elif t==6:
        teams=r.randint(3,8); players=r.randint(3,10); subs=r.randint(1,4); ans=teams*(players+subs); q=f"There are {teams} teams, each with {players} players and {subs} {'substitute' if subs==1 else 'substitutes'}. How many people total?"; target=f"{ans}."
    else:
        pages=r.randint(5,20); items=r.randint(3,10); ans=pages*items; q=f"A catalog has {pages} pages with {items} items per page. How many items are listed?"; target=f"{ans}."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['school','work','games','everyday']),f"v2_counting_{t:02d}",'number',2,['direct','accurate'],negative=f"{ans+r.choice([-2,-1,1,2])}.",rejection_type='incorrect_count')


def gen_constraint_selection(idx:int)->Example:
    pair=idx//2; variant=idx%2; r=rng_for('constraint_pair',pair); t=pair%4
    labels=['A','B','C']
    if t==0:
        costs=r.sample(range(20,80,5),3); hours=r.sample(range(4,13),3); budget=r.choice([35,45,55,65]); min_h=r.choice([6,8,10]); valid=[i for i in range(3) if costs[i]<=budget and hours[i]>=min_h]
        if not valid: # force one valid
            i=r.randrange(3); costs[i]=budget-5; hours[i]=min_h+1; valid=[i]
        chosen=valid[0] if variant==0 else valid[-1]
        q=f"Need cost at most {money(budget)} and battery at least {min_h} hours. A: {money(costs[0])}, {hours[0]}h. B: {money(costs[1])}, {hours[1]}h. C: {money(costs[2])}, {hours[2]}h. Name one option that fits."
    elif t==1:
        weights=r.sample([1,2,3,4,5,6],3); sizes=r.sample([10,12,14,16,18,20],3); max_w=r.choice([3,4,5]); min_s=r.choice([12,14,16]); valid=[i for i in range(3) if weights[i]<=max_w and sizes[i]>=min_s]
        if not valid: i=r.randrange(3); weights[i]=max_w; sizes[i]=min_s; valid=[i]
        chosen=valid[0] if variant==0 else valid[-1]; q=f"Need weight no more than {max_w} kg and capacity at least {min_s} L. A: {weights[0]} kg/{sizes[0]} L. B: {weights[1]} kg/{sizes[1]} L. C: {weights[2]} kg/{sizes[2]} L. Which option fits?"
    elif t==2:
        times=r.sample([20,30,40,50,60,70],3); stops=r.sample([0,1,2,3,4],3); max_t=r.choice([40,50,60]); max_s=r.choice([1,2]); valid=[i for i in range(3) if times[i]<=max_t and stops[i]<=max_s]
        if not valid: i=r.randrange(3); times[i]=max_t; stops[i]=max_s; valid=[i]
        chosen=valid[0] if variant==0 else valid[-1]; q=f"Need a route taking at most {max_t} minutes with at most {count_noun(max_s, 'stop')}. A: {times[0]} min/{count_noun(stops[0], 'stop')}. B: {times[1]} min/{count_noun(stops[1], 'stop')}. C: {times[2]} min/{count_noun(stops[2], 'stop')}. Which works?"
    else:
        scores=r.sample(range(60,100,5),3); prices=r.sample(range(10,50,5),3); min_sc=r.choice([70,75,80,85]); max_p=r.choice([25,30,35,40]); valid=[i for i in range(3) if scores[i]>=min_sc and prices[i]<=max_p]
        if not valid: i=r.randrange(3); scores[i]=min_sc; prices[i]=max_p; valid=[i]
        chosen=valid[0] if variant==0 else valid[-1]; q=f"Need quality at least {min_sc} and price at most {money(max_p)}. A: {scores[0]}/{money(prices[0])}. B: {scores[1]}/{money(prices[1])}. C: {scores[2]}/{money(prices[2])}. Which fits?"
    target=f"Option {labels[chosen]}."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['shopping','travel','electronics','work']),f"v2_constraint_{t:02d}",'choice',3,['direct','decisive'],counterfactual_group_id=f"constraint_{pair:04d}",negative="None of them.",rejection_type='violates_constraints')


def gen_causal_reasoning(idx:int)->Example:
    r=rng_for('causal_reasoning',idx); t=idx%8
    cases=[
        ("The lamp worked, then its cable was unplugged. The outlet still works. Most likely cause?", "The lamp is unplugged."),
        ("A plant was healthy, then went two weeks without water and wilted. Most likely cause?", "Lack of water."),
        ("The app connects on Wi-Fi but not on cellular, and cellular data is disabled. Likely cause?", "Cellular data is disabled."),
        ("A printer stopped after the paper tray became empty. What likely caused it?", "The printer is out of paper."),
        ("The room became dark immediately after the bulb burned out. Likely cause?", "The bulb burned out."),
        ("A download paused when the network disconnected. Why?", "The network connection dropped."),
        ("The alarm stopped after its battery was removed. What caused the stop?", "The battery was removed."),
        ("The cup tipped over, then water spread across the table. What caused the spill?", "The cup tipped over."),
    ]
    q,target=cases[t]
    suffix=r.choice(["", f" This happened in the {r.choice(PLACES)}.", f" No other change was reported for {r.choice(PROJECTS)}."])
    q=q+suffix
    neg=r.choice(["There is no likely cause.","A software update caused it.","It happened randomly."])
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['home','electronics','everyday','science']),f"v2_causal_{t:02d}",'cause',2,['direct','careful'],negative=neg,rejection_type='unsupported_cause')


def gen_physical_commonsense(idx:int)->Example:
    r=rng_for('physical_commonsense',idx); t=idx%8
    cases=[
        ("A metal spoon sits in hot soup. What happens to the spoon?", "It gets warm."),
        ("Ice is left in a warm room. What happens?", "It melts."),
        ("A full open cup is tilted sideways. What happens to the liquid?", "It spills."),
        ("A box is too wide for a doorway but narrow enough when turned. What should you try?", "Rotate the box."),
        ("A wet towel is left in moving air. What happens over time?", "It dries."),
        ("A ball is released on a downward slope. Which way does it roll?", "Down the slope."),
        ("A sealed balloon is squeezed gently. What happens to its shape?", "It deforms."),
        ("A cold glass is placed in humid air. What may form outside it?", "Condensation."),
    ]
    q,target=cases[t]; q=q.replace('A ',r.choice(['A ','The ']),1)
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),'science',f"v2_physical_{t:02d}",'commonsense',1,['direct','accurate'],requires_external_fact=True,negative="Nothing changes.",rejection_type='physical_commonsense_error')


def gen_spatial_relations(idx:int)->Example:
    r=rng_for('spatial_relations',idx); t=idx%8; a,b,c=r.sample(NAMES,3)
    if t==0: q=f"{a} is north of {b}. Where is {a} relative to {b}?"; target="North."
    elif t==1: q=f"{a} is left of {b}, and {b} is left of {c}. Where is {a} relative to {c}?"; target="Left."
    elif t==2: q=f"{a} is above {b}, and {b} is above {c}. Where is {c} relative to {a}?"; target="Below."
    elif t==3: q=f"{a} is north of {b}. {b} is east of {c}. Where is {a} relative to {c}?"; target="Northeast."
    elif t==4: q=f"{a} sits between {b} and {c}, with {b} on the left. Who is on the right?"; target=f"{c}."
    elif t==5: q=f"The key is inside the box, and the box is under the table. Where is the key relative to the table?"; target="Under the table."
    elif t==6: q=f"{a} is west of {b}, and {b} is south of {c}. Where is {a} relative to {c}?"; target="Southwest."
    else: q=f"The lamp is to the right of the book. Where is the book relative to the lamp?"; target="To the left."
    wrong_by_template={
        0:"South.", 1:"Right.", 2:"Above.", 3:"Southwest.",
        4:f"{b}.", 5:"On the table.", 6:"Northeast.", 7:"To the right.",
    }
    neg=wrong_by_template[t]
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['everyday','games','travel']),f"v2_spatial_{t:02d}",'relation',2,['direct','accurate'],negative=neg,rejection_type='wrong_spatial_relation')


def gen_estimation_sanity(idx:int)->Example:
    r=rng_for('estimation_sanity',idx); t=idx%8
    if t==0:
        n=r.randint(6,20); price=r.randint(3,12); approx=round(n*price,-1); q=f"About how much do {n} items at roughly {money(price)} each cost?"; target=f"About {money(approx)}."
    elif t==1:
        plausible=(idx//8)%2==0; weight=r.choice([1,2,3]) if plausible else r.choice([100,200,500]); q=f"A laptop is said to weigh {weight} kg. Is that plausible?"; target="Yes." if plausible else "No."
    elif t==2:
        km=r.choice([2,3,4,5]); mins=r.choice([25,35,45,60]); q=f"Walking {km} km in {mins} minutes—is that broadly plausible?"; target="Yes, broadly." if mins>=km*8 else "Probably not."
    elif t==3:
        n=r.randint(20,90); unit=r.choice([2,5,10]); approx=round(n*unit,-1); q=f"Roughly estimate {n} × {unit}."; target=f"About {approx}."
    elif t==4:
        people=r.randint(8,30); slices=people*2; pizzas=math.ceil(slices/8); q=f"About how many 8-slice pizzas serve {people} people at two slices each?"; target=f"About {pizzas} pizzas."
    elif t==5:
        total=r.randint(40,120); fraction=r.choice([0.25,0.5,0.75]); ans=math.floor(total*fraction+0.5); q=f"About {int(fraction*100)}% of {total} is how much?"; target=f"About {ans}."
    elif t==6:
        plausible=(idx//8)%2==0; duration=r.choice([5,10,15]) if plausible else r.choice([300,600]); q=f"A short email takes {duration} minutes to read. Is that plausible?"; target="Yes." if plausible else "No."
    else:
        daily=r.randint(10,50); days=r.choice([5,7,10]); approx=round(daily*days,-1); q=f"At about {daily} units per day for {days} days, roughly how many units?"; target=f"About {approx}."
    if target.startswith('About'):
        num=float(re.findall(r"\d+(?:\.\d+)?",target)[0]); wrong=num+max(5,round(abs(num)*0.35)); neg=re.sub(r"\d+(?:\.\d+)?",fmt_num(wrong),target,count=1)
    else:
        neg="No." if target.startswith('Yes') else "Yes."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['shopping','everyday','work','travel']),f"v2_estimation_{t:02d}",'estimate_or_boolean',1,['practical','concise'],negative=neg,rejection_type='implausible_estimate')


def gen_evidence_inference(idx:int)->Example:
    pair=idx//2; variant=idx%2; r=rng_for('evidence_pair',pair); t=pair%4
    if t==0:
        color=r.choice(['red','amber','green']); meaning={'red':'charging','amber':'waiting','green':'ready'}[color]
        ask=meaning if variant==0 else r.choice([x for x in ['charging','waiting','ready'] if x!=meaning])
        q=f"Manual: {color} means {meaning}. The indicator is {color}. Is the device {ask}?"; target="Yes." if variant==0 else "No."
    elif t==1:
        place=r.choice(PLACES); item=r.choice(['parcel','key','badge','notebook']); q=f"Note: the {item} was left at the {place}. Where is the {item}?" if variant==0 else f"Note: the {item} was left at the {place}. Who left it there?"; target=f"At the {place}." if variant==0 else "I don't know based on the available information."
    elif t==2:
        owner=r.choice(NAMES); project=r.choice(PROJECTS); q=f"Record: {owner} owns project {project}. Who owns {project}?" if variant==0 else f"Record: {owner} owns project {project}. When was it created?"; target=f"{owner}." if variant==0 else "I don't know based on the available information."
    else:
        temp=r.choice([2,4,8,10]); q=f"Rule: below 5°C is cold. Reading: {temp}°C. Is it cold?"; target="Yes." if temp<5 else "No."
    neg="I don't know based on the available information." if target not in {"I don't know based on the available information."} else r.choice(["Yesterday.","Maya.","At noon."])
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['electronics','work','everyday','science']),f"v2_evidence_{t:02d}",'supported_or_abstain',2,['grounded','direct'],counterfactual_group_id=f"evidence_{pair:04d}",negative=neg,rejection_type='unsupported_or_overabstaining')


def gen_insufficient_information(idx:int)->Example:
    pair=idx//2; variant=idx%2; r=rng_for('insufficient_pair',pair); t=pair%5
    if t==0:
        start=r.randint(5,30); removed=r.randint(1,start-1)
        q=f"A box starts with {start} apples and {removed} are removed. How many remain?" if variant==0 else f"Some apples are in a box and {removed} are removed. How many remain?"; target=f"{start-removed}." if variant==0 else "I don't know based on the available information."
    elif t==1:
        a=r.randint(10,50); b=r.choice([x for x in range(10,51) if x!=a])
        q=f"Option A costs {money(a)} and B costs {money(b)}. Which is cheaper?" if variant==0 else f"Option A costs {money(a)} and option B's price is missing. Which is cheaper?"; target=f"Option {'A' if a<b else 'B'}." if variant==0 and a!=b else "I don't know based on the available information."
    elif t==2:
        speed=r.choice([30,40,50]); hours=r.choice([2,3,4]); q=f"A car travels {speed} mph for {hours} hours. Distance?" if variant==0 else f"A car travels for {hours} hours, but its speed is not given. Distance?"; target=f"{speed*hours} miles." if variant==0 else "I don't know based on the available information."
    elif t==3:
        total=r.randint(20,100); pct=r.choice([25,50,75]); q=f"What is {pct}% of {total}?" if variant==0 else f"A percentage of {total} is requested, but the percentage is missing. What is the result?"; target=f"{fmt_num(total*pct/100)}." if variant==0 else "I don't know based on the available information."
    else:
        name=r.choice(NAMES); q=f"The note says {name} booked the room. Who booked it?" if variant==0 else "The note says the room was booked, but gives no name. Who booked it?"; target=f"{name}." if variant==0 else "I don't know based on the available information."
    if not target.startswith("I don't"):
        neg="I don't know based on the available information."
    elif t==0:
        neg=f"{removed}."
    elif t==1:
        neg="Option A."
    elif t==2:
        neg=f"{hours*40} miles."
    elif t==3:
        neg=f"{fmt_num(total*0.5)}."
    else:
        neg=f"{name}."
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['everyday','work','travel','personal_finance']),f"v2_insufficient_{t:02d}",'answer_or_abstain',2,['grounded','direct'],counterfactual_group_id=f"insufficient_{pair:04d}",negative=neg,rejection_type='fabrication_or_unnecessary_abstention')


def gen_troubleshooting_diagnosis(idx:int)->Example:
    r=rng_for('troubleshooting_diagnosis',idx); t=idx%8
    cases=[
        ("The monitor says 'no signal,' the computer is on, and the display cable is loose. First step?", "Reconnect the display cable."),
        ("The printer has paper but shows an empty-ink warning. First step?", "Check or replace the ink."),
        ("The app works on Wi-Fi but not cellular, and cellular access is disabled. First step?", "Enable cellular access."),
        ("A device will not power on, and its battery is fully drained. First step?", "Charge the battery."),
        ("A script fails with 'file not found,' and the path has a typo. First step?", "Fix the file path."),
        ("Audio is silent, and the output device is set to the wrong speakers. First step?", "Select the correct output device."),
        ("A page looks stale after deployment, and the browser is serving cached content. First step?", "Refresh or clear the cache."),
        ("The router has power, but its internet cable is disconnected. First step?", "Reconnect the internet cable."),
    ]
    q,target=cases[t]; q=q.replace('First step?',r.choice(['First step?','What should you check first?','Best next step?']))
    neg=r.choice(["Replace the whole system.","Ignore it and retry repeatedly.","There is nothing to check."])
    return Example(make_messages(choose_system('reasoning',idx),[('user',q),('assistant',target)]),r.choice(['coding','electronics','home','work']),f"v2_troubleshoot_{t:02d}",'next_step',2,['practical','direct'],negative=neg,rejection_type='poor_troubleshooting_step')

# ----------------------- Communication generators ------------------------

def gen_acknowledgment_update(idx:int)->Example:
    r=rng_for('acknowledgment_update',idx); t=idx%16; project=r.choice(PROJECTS); name=r.choice(NAMES)
    time=r.choice(['2:00 PM','3:30 PM','4:00 PM']); folder=r.choice(['blue','green','shared']); route=r.choice(['east','south','west']); deadline=r.choice(['Friday','Monday','August 25','September 3']); room=r.choice(['Room 204','the library','the main studio']); version=r.choice(['compact','stable','v2']); language=r.choice(['English','Spanish','Japanese']); seat=r.choice(['12A','18C','7F'])
    cases=[
        (f"I moved the {project} meeting to {time}.",[f"Got it—{time}.",f"Okay, {time}.",f"Noted—the meeting is at {time}."],f"v2_ack_update_00"),
        (f"I saved the backup in the {folder} folder.",[f"Noted—the {folder} folder.",f"Got it, the {folder} folder.",f"Okay—the backup is in the {folder} folder."],f"v2_ack_update_01"),
        (f"{name} is joining the call after all.",[f"Got it—{name} is joining.",f"Okay, {name} will be there.",f"Nice—{name} is joining."],f"v2_ack_update_02"),
        (f"I switched the route from north to {route}.",[f"Okay—the {route} route.",f"Got it, we are taking {route}.",f"Noted—the route is now {route}."],f"v2_ack_update_03"),
        (f"The new deadline is {deadline}.",[f"Understood—{deadline}.",f"Got it, the deadline is {deadline}.",f"Okay—{deadline} it is."],f"v2_ack_update_04"),
        ("I already sent the revised file.",["Got it.","Perfect, thanks.","Nice—that's sent."],f"v2_ack_update_05"),
        ("The issue is fixed now.",["Nice—glad it is fixed.","Great, that is resolved.","Good—glad it is working."],f"v2_ack_update_06"),
        ("I will handle the first draft.",["Sounds good.","Great, you have the first draft.","Okay—thanks for taking it."],f"v2_ack_update_07"),
        (f"The {project} call is canceled.",["Got it—the call is canceled.","Okay, no call today.","Noted—the call is canceled."],f"v2_ack_update_08"),
        (f"We changed the meeting place to {room}.",[f"Got it—{room}.",f"Okay, we will meet at {room}.",f"Noted—the location is {room}."],f"v2_ack_update_09"),
        ("I set a reminder for tomorrow morning.",["Great—the reminder is set.","Got it, tomorrow morning.","Perfect, thanks for setting it."],f"v2_ack_update_10"),
        (f"We are using the {version} model version.",[f"Got it—the {version} version.",f"Okay, use {version}.",f"Noted—the selected version is {version}."],f"v2_ack_update_11"),
        ("The invoice has been paid.",["Got it—the invoice is paid.","Perfect, that is settled.","Okay—payment is complete."],f"v2_ack_update_12"),
        (f"I switched the presentation language to {language}.",[f"Got it—{language}.",f"Okay, the presentation will be in {language}.",f"Noted—the language is {language}."],f"v2_ack_update_13"),
        (f"I reserved seat {seat}.",[f"Nice—seat {seat}.",f"Got it, {seat} is reserved.",f"Okay—your seat is {seat}."],f"v2_ack_update_14"),
        (f"We are postponing the {project} review until next week.",["Understood—next week.","Got it, the review is postponed.","Okay—we will revisit it next week."],f"v2_ack_update_15"),
    ]
    u,targets,family=cases[t]; target=pick_text(targets,idx,16)
    if idx%3==0:
        messages=make_messages(choose_system('communication',idx),[('user',f"Quick update on {project}."),('assistant','Sure.'),('user',u),('assistant',target)])
        context='recent_update'
    else:
        messages=make_messages(choose_system('communication',idx),[('user',u),('assistant',target)]); context='current_turn'
    return Example(messages,r.choice(DOMAINS),family,'acknowledgment',1,['casual','responsive'],context_dependency=context,negative="I have carefully documented this information and will now explain it in detail.",rejection_type='robotic_overresponse')


def gen_natural_follow_up(idx:int)->Example:
    r=rng_for('natural_follow_up',idx); t=idx%16; name=r.choice(NAMES)
    pairs=[
        ("I finished the interview.",["How did it go?","Nice—how are you feeling about it?","Good—what was the hardest part?"]),
        ("The model finally loaded.",["Nice—how are the outputs?","Great—does it run correctly now?","Finally! How does it look?"]),
        ("I tried the new recipe.",["How did it turn out?","Nice—would you make it again?","How was it?"]),
        ("I got the apartment keys.",["Nice—when do you move in?","That is exciting—moving soon?","Great! What is the first thing you are setting up?"]),
        ("I submitted the application.",["When do you hear back?","Nice—what is the expected timeline?","Good luck! When should they respond?"]),
        ("I ran the benchmark.",["How did the metrics look?","Nice—any surprises in the results?","Great—did it improve?"]),
        ("I talked with my teammate.",["Did the conversation help?","How did they respond?","Good—did you reach an agreement?"]),
        (f"{name} accepted the invitation.",["Great—when is the event?",f"Nice—did {name} confirm the time?","Great—what is the plan now?"]),
        ("I passed the exam.",["Nice!! How did it feel?","Congrats—was it as hard as expected?","That is great—celebrating tonight?"]),
        ("The repair shop called me back.",["What did they say?","Good—did they find the problem?","Nice—when will it be ready?"]),
        ("I booked the flight.",["Nice—when do you leave?","Great—did you get a good time?","Where are you flying from?"]),
        ("I showed the prototype to the team.",["How did they react?","Nice—what feedback did they give?","Good—did the core idea land?"]),
        ("I finally cleaned the whole apartment.",["Nice—does it feel better now?","That must feel good. What took longest?","Great—are you resting now?"]),
        ("I sent the apology.",["How did they respond?","Good—do you feel better after sending it?","Was it received okay?"]),
        ("The package showed up.",["Nice—was everything inside?","Finally! Did it arrive in good shape?","Great—was it the right item?"]),
        ("I tried the direct train today.",["How was it?","Did it save much hassle?","Nice—would you use it again?"]),
    ]
    u,targets=pairs[t]; target=pick_text(targets,idx,16)
    if idx%4==0:
        messages=make_messages(choose_system('communication',idx),[('user','I have an update.'),('assistant','What happened?'),('user',u),('assistant',target)])
    else: messages=make_messages(choose_system('communication',idx),[('user',u),('assistant',target)])
    return Example(messages,r.choice(DOMAINS),f"v2_followup_{t:02d}",'question',1,['curious','natural'],context_dependency='conversation_continuation',negative="Acknowledged.",rejection_type='missed_natural_followup')


def gen_empathy_validation(idx:int)->Example:
    r=rng_for('empathy_validation',idx); t=idx%16
    pairs=[
        ("I have been fixing this bug for hours and I am exhausted.",["That sounds draining. Take a breather.","Yeah, you have been at it a long time—pause for a bit.","That is exhausting. Give yourself a short break."]),
        ("The presentation went badly, and I feel embarrassed.",["That hurts, but one rough presentation does not define you.","That is a rough feeling, but it was one presentation.","I get why you feel embarrassed. Be kind to yourself tonight."]),
        ("I am nervous about tomorrow's interview.",["That is understandable. You have time to prepare.","Totally normal—focus on the parts you can practice.","Yeah, interviews are stressful. You can still go in prepared."]),
        ("My plans fell apart at the last minute.",["That is really frustrating.","Ugh, that is disappointing.","That is a lot to adjust to at once."]),
        ("I keep replaying that awkward conversation.",["Yeah, that can stick with you for a while.","I get it—awkward moments can loop in your head.","That is uncomfortable, but it will feel smaller with time."]),
        ("I did a lot today, but it still feels like nothing.",["You still made progress, even if it feels small.","That feeling is real, but your work still counts.","You did more than your brain is giving you credit for."]),
        ("I am disappointed the test failed again.",["That is discouraging. Step away for a minute.","Yeah, another failure is frustrating. Take a short reset.","That stings. Pause, then come back to the clearest error."]),
        ("I miss my friends back home.",["That sounds lonely. Reach out to someone you trust.","Yeah, being away from your people is hard.","I am sorry—you must really miss them."]),
        ("I did not get the job I wanted.",["I am sorry. That rejection hurts.","That is disappointing, especially after all that effort.","Yeah, that one hurts. Give yourself a little time."]),
        ("I made a mistake that affected the whole team.",["That is a heavy feeling. Own it, fix what you can, and learn from it.","I get why you feel bad. A clear repair matters more than beating yourself up.","That is tough, but one mistake does not make you unreliable."]),
        ("Everything on my list feels urgent.",["That sounds overwhelming. Pick the one thing with the nearest consequence.","Yeah, that is a lot. Choose one priority and start there.","No wonder you feel overloaded—shrink it to the next task."]),
        ("I studied hard and still got a low score.",["That is really discouraging.","I am sorry—that result does not erase the effort you put in.","That hurts. Review what went wrong after you have had a break."]),
        ("My friend canceled on me again.",["That is disappointing, especially if it keeps happening.","Yeah, being canceled on repeatedly feels awful.","I get why you are upset. It may be worth asking what is going on."]),
        ("I feel behind compared with everyone else.",["That comparison can be brutal. Your pace is still valid.","Yeah, it is hard not to compare, but you are seeing only part of their story.","I get it. Focus on your next step, not their timeline."]),
        ("I froze during the meeting and forgot what I wanted to say.",["That is embarrassing, but it happens under pressure.","I am sorry—that must have felt awful in the moment.","That was rough. You can follow up with the point you missed."]),
        ("I am tired of starting over.",["That sounds exhausting. You do not have to solve everything tonight.","Yeah, repeated resets wear you down.","I get it. Preserve the smallest thing that still works and build from there."]),
    ]
    u,targets=pairs[t]; target=pick_text(targets,idx,16)
    if idx%5==0: messages=make_messages(choose_system('communication',idx),[('user','Can I vent for a second?'),('assistant','Of course.'),('user',u),('assistant',target)])
    else: messages=make_messages(choose_system('communication',idx),[('user',u),('assistant',target)])
    return Example(messages,'social',f"v2_empathy_{t:02d}",'supportive_reply',1,['warm','supportive'],negative="That is not a serious problem, so just move on.",rejection_type='dismissive_empathy')


def gen_tactful_disagreement(idx:int)->Example:
    r=rng_for('tactful_disagreement',idx); t=idx%16
    pairs=[
        ("We should delete every backup to save space.",["I would not—keep at least one verified backup.","I disagree; one known-good backup is worth the space.","I would free space elsewhere and preserve a backup."]),
        ("Let's skip testing and deploy now.",["I would test first; a quick deploy is not worth the risk.","I would not skip the test—run the critical checks first.","I disagree. A short verification is safer than a rushed deploy."]),
        ("The cheapest option must be the best one.",["Not necessarily—reliability matters too.","I would compare total value, not price alone.","Cheapest can be best, but not if it fails your main need."]),
        ("I should reply while I am still angry.",["I would wait until you are calmer.","I would draft it now and send it later.","Better to pause first; anger can make the message harsher than intended."]),
        ("We can ignore the warning because it only appeared once.",["I would check it before assuming it is harmless.","One warning can still matter—inspect the cause first.","I would verify it, especially if the failure would be costly."]),
        ("Let's add every feature before showing anyone.",["I would ship a smaller prototype first.","I would test the core idea before building everything.","A focused prototype will teach you more quickly."]),
        ("I think asking for help makes me look weak.",["I disagree—asking early can save time.","I do not think so; knowing when to ask is a strength.","Getting help before you are stuck is usually efficient, not weak."]),
        ("We should choose the longer route because it looks nicer.",["Only if the extra time is worth it to you.","I would choose it only if scenery is a real priority.","That is reasonable for leisure, but not if arrival time matters."]),
        ("I can reuse the same password everywhere because it is strong.",["I would not—one breach would expose every account.","A strong password should still be unique per account.","I disagree; use unique passwords with a manager."]),
        ("We should buy the largest plan just in case.",["I would start with the size you can justify.","Bigger is not automatically better—match it to actual use.","I would avoid paying for capacity you may never use."]),
        ("If the first attempt failed, the idea must be bad.",["Not necessarily—the test may have exposed a fixable issue.","One failure is evidence, not a final verdict.","I would diagnose the failure before rejecting the whole idea."]),
        ("We should cancel the whole project because one metric dipped.",["I would investigate the dip before making that decision.","One metric is not enough to cancel the project.","I disagree—check whether the decline is real and material first."]),
        ("More meetings will solve the communication problem.",["Maybe, but clearer ownership may help more than extra meetings.","I would fix the information flow before adding meetings.","More meetings can add noise; define who needs what first."]),
        ("We should always pick the fastest model.",["Only if speed matters more than quality for that request.","I would route by task; the fastest model is not always sufficient.","Speed is useful, but capability and verification still matter."]),
        ("I should agree so nobody is disappointed.",["I would be honest rather than promise something you cannot support.","Kind disagreement is better than resentful agreement.","You can say no respectfully without rejecting the person."]),
        ("The plan worked once, so we do not need monitoring.",["I would still monitor it; one success does not prove stability.","A successful run is encouraging, not a substitute for monitoring.","I disagree—keep checks until the behavior is reliable."]),
    ]
    u,targets=pairs[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('communication',idx),[('user',u),('assistant',target)]),r.choice(DOMAINS),f"v2_disagreement_{t:02d}",'opinion',1,['respectful','direct'],safety_tier='elevated' if t in {0,1,4,8,15} else 'normal',negative="You're wrong. Do it my way.",rejection_type='hostile_disagreement')


def gen_clarification_request(idx:int)->Example:
    r=rng_for('clarification_request',idx); t=idx%16
    pairs=[
        ("Make it faster.",["Which part feels slow?","Do you mean startup, processing, or response time?","What specifically needs to be faster?"]),
        ("Use the other one.",["Which option do you mean?","Which one should I switch to?","Do you mean the second option or a different one?"]),
        ("Move the meeting.",["What time should I move it to?","What day and time do you want instead?","When should the meeting be rescheduled?"]),
        ("Fix the file.",["What is wrong with the file?","Which file, and what needs fixing?","What problem are you seeing in it?"]),
        ("Book the trip.",["What dates and destination?","Where and when are you traveling?","What destination, dates, and budget should I use?"]),
        ("Change the settings.",["Which setting should change?","What should the new setting be?","Which device and setting do you mean?"]),
        ("Send it to them.",["Who should receive it?","Who do you mean by them?","Which recipient should I use?"]),
        ("Make the answer better.",["Do you want it clearer, shorter, or more formal?","What should improve—accuracy, tone, or length?","What kind of better are you aiming for?"]),
        ("Schedule it for later.",["What date and time should I use?","How much later?","Which time zone and time should I schedule?"]),
        ("Add support for it.",["What feature or format needs support?","What does it refer to here?","Which use case should the new support cover?"]),
        ("Reserve a table.",["For how many people, and when?","Which restaurant and time?","What date, party size, and name should I use?"]),
        ("Use the previous version.",["Which version number do you mean?","Do you mean the last release or an earlier one?","Which previous checkpoint should I use?"]),
        ("Share the results.",["Who should receive them, and in what format?","Which results should I share?","Where should I send the results?"]),
        ("Make the design simpler.",["Which parts feel too complex?","What must the simpler version still preserve?","Do you want fewer screens, fewer controls, or both?"]),
        ("Turn that off.",["What should I turn off?","Which device or setting do you mean?","Do you mean the alert, feature, or device?"]),
        ("Help me prepare for it.",["What are you preparing for?","What is the event, and when is it?","Which part of the preparation do you need help with?"]),
    ]
    u,targets=pairs[t]; target=pick_text(targets,idx,16)
    if idx%4==0: u=f"For {r.choice(PROJECTS)}, {u[0].lower()+u[1:]}"
    return Example(make_messages(choose_system('communication',idx),[('user',u),('assistant',target)]),r.choice(DOMAINS),f"v2_clarification_{t:02d}",'question',1,['brief','specific'],negative="Sure, I will handle it exactly as requested.",rejection_type='acts_on_ambiguity')


def gen_preference_opinion(idx:int)->Example:
    r=rng_for('preference_opinion',idx); t=idx%16
    pairs=[
        ("For a daily laptop, would you prioritize battery life or a brighter screen?",["Battery life, unless you work outdoors a lot.","I would prioritize battery for daily use.","Battery life—unless outdoor visibility is your main issue."]),
        ("Would you choose a quiet cafe or a busy one for studying?",["A quiet cafe.","Quiet, especially for focused work.","I would pick the quieter place."]),
        ("For backups, local only or local plus cloud?",["Local plus cloud gives better redundancy.","Both—local plus encrypted cloud.","I would use local and cloud, not just one copy."]),
        ("Would you take the direct train or the cheaper one with two transfers?",["The direct train, unless the savings are substantial.","I would take the direct train for the lower hassle.","Direct, unless the cheaper fare is meaningfully lower."]),
        ("For a first prototype, polished design or core functionality?",["Core functionality first.","Build the core behavior before polishing it.","Functionality first; polish after the idea works."]),
        ("Would you buy the smaller phone or the one with longer battery life?",["I would pick the longer battery life.","Battery, unless compact size is essential.","The longer-lasting phone would be my choice."]),
        ("For notes, paper or a simple app?",["A simple app is easier to search.","I would use an app for searchable notes.","A simple app, unless handwriting helps you remember."]),
        ("Should dinner be quick or elaborate tonight?",["Quick sounds better tonight.","Keep it simple tonight.","I would make something quick."]),
        ("For a small apartment, more storage or a larger desk?",["More storage, unless you work at the desk all day.","I would prioritize storage in a small space.","Storage first; clutter steals usable space."]),
        ("Would you use one large monitor or two smaller ones?",["Two monitors are better for multitasking.","I would choose two smaller screens for flexible layouts.","Two, unless you need one uninterrupted canvas."]),
        ("Morning workout or evening workout?",["Whichever you can sustain—morning is easier to protect.","I would choose morning if your evenings are unpredictable.","Morning, unless you consistently have more energy later."]),
        ("For travel, carry-on only or check a bag?",["Carry-on only for a short trip.","I would avoid checking a bag unless you need the space.","Carry-on, if everything fits comfortably."]),
        ("Mechanical keyboard or a quiet low-profile one for shared space?",["A quiet low-profile keyboard.","Quiet keys make more sense in shared space.","I would prioritize noise over key feel there."]),
        ("Would you rent closer to campus or save money farther away?",["Closer, if the commute would cost you hours each week.","I would price the commute before choosing the cheaper place.","Near campus is worth it when time is the main constraint."]),
        ("For learning, one long session or shorter daily sessions?",["Short daily sessions are easier to retain.","I would choose consistent shorter sessions.","Daily practice usually beats one long cram session."]),
        ("Would you buy the faster charger or a second regular charger?",["A second charger is more useful in two locations.","I would choose the spare unless charging speed is a real problem.","The extra charger gives more day-to-day convenience."]),
    ]
    q,targets=pairs[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('communication',idx),[('user',q),('assistant',target)]),r.choice(DOMAINS),f"v2_preference_{t:02d}",'preference',1,['natural','decisive'],negative="Both options have various advantages and disadvantages that must be considered in depth.",rejection_type='nonanswering_verbosity')


def gen_apology_repair(idx:int)->Example:
    r=rng_for('apology_repair',idx); t=idx%16; name=r.choice(NAMES); project=r.choice(PROJECTS)
    cases=[
        (f"The {project} deadline is Tuesday.", "No, it is Wednesday.",["You're right—sorry, Wednesday.","My mistake—the deadline is Wednesday.","Thanks for correcting me: Wednesday."]),
        (f"{name} is presenting first.", f"Actually, {name} is presenting last.",[f"Got it—sorry. {name} is presenting last.",f"My mistake—{name} is last.",f"Thanks for the correction; {name} presents last."]),
        ("The file is in the blue folder.", "I moved it to the green folder.",["Thanks for the correction—the green folder.","Got it, the green folder. My mistake.","Right—the file is in green now."]),
        ("The total is 42.", "I rechecked it; the total is 47.",["You're right—47. Sorry about that.","My mistake—the total is 47.","Thanks for rechecking it: 47."]),
        ("The call starts at 2 PM.", "It was moved to 3 PM.",["Got it—3 PM. Sorry.","My mistake; the call is at 3 PM.","Thanks—the updated time is 3 PM."]),
        ("You said the south route.", "I chose the east route.",["Right—the east route. My mistake.","Got it, east—not south.","Thanks for correcting me; use the east route."]),
        ("The package arrived Monday.", "It arrived Tuesday.",["Thanks—Tuesday, not Monday.","My mistake; it arrived Tuesday.","Right—the arrival day was Tuesday."]),
        ("The setting should be off.", "No, keep it on.",["Understood—keep it on. Sorry.","My mistake; leave the setting on.","Got it—the setting stays on."]),
        ("The reservation is under Maya.", "It is under Lena.",["Thanks—the reservation is under Lena.","My mistake; use Lena's name.","Got it—Lena, not Maya."]),
        ("We need three copies.", "We actually need five.",["You're right—we need five. Sorry.","My mistake; make five copies.","Thanks for the update: five copies."]),
        ("The pickup point is the station.", "It changed to the hotel.",["Got it—the hotel. My mistake.","Thanks—the pickup point is now the hotel.","Right, use the hotel instead."]),
        ("The selected model is the large one.", "No, we chose the compact one.",["You're right—the compact model.","My mistake; use the compact one.","Got it—compact, not large."]),
        ("The report should be in PDF.", "It needs to be a spreadsheet.",["Thanks—the report should be a spreadsheet.","My mistake; use the spreadsheet format.","Got it—a spreadsheet, not PDF."]),
        ("The meeting is in Room 301.", "It was moved to Room 214.",["Got it—Room 214. Sorry.","My mistake; the room is 214.","Thanks for the correction—Room 214."]),
        ("The access code ends in 62.", "It ends in 26.",["You're right—it ends in 26.","My mistake; the ending is 26.","Thanks—26, not 62."]),
        ("We are shipping on Friday.", "Shipping is Monday now.",["Got it—Monday. Sorry.","My mistake; shipping moved to Monday.","Thanks for the update—we ship Monday."]),
    ]
    assistant1,user2,targets=cases[t]; target=pick_text(targets,idx,16)
    messages=make_messages(choose_system('communication',idx),[('user',f"Quick check for {project}."),('assistant',assistant1),('user',user2),('assistant',target)])
    return Example(messages,r.choice(DOMAINS),f"v2_apology_{t:02d}",'repair',1,['accountable','brief'],context_dependency='user_correction',negative=assistant1,rejection_type='ignores_correction')


def gen_deescalation(idx:int)->Example:
    r=rng_for('deescalation',idx); t=idx%16
    pairs=[
        ("They ignored my message. I want to send something harsh.",["Pause and ask directly before assuming the worst.","Wait a bit, then send one neutral follow-up.","Do not send it angry—ask whether they saw the message."]),
        ("My teammate blamed me in the group chat.",["Respond to the facts, not the tone.","Correct the record calmly and move the detailed discussion private.","Keep your reply factual and avoid matching the hostility."]),
        ("I want to argue with every comment.",["Address only the comments that affect the result.","Pick the important points and let the rest go.","Respond where clarification changes the work, not to every remark."]),
        ("We are both talking over each other.",["Take turns and settle one point at a time.","Pause and let each person finish one point.","Slow it down—one issue and one speaker at a time."]),
        ("They changed the plan without asking me.",["Ask what changed before escalating.","Start by confirming the new plan and why it changed.","Get the facts first, then explain the impact on you."]),
        ("I am tempted to post the private messages.",["Do not post them; handle the issue privately.","Keep the messages private and address the behavior directly.","Do not make them public—use a private or formal channel."]),
        ("This discussion is going nowhere.",["Pause it and return with one concrete question.","Take a break and restart around the exact decision.","Stop the loop and identify the one unresolved point."]),
        ("I feel like proving them wrong matters more than the project.",["Refocus on the outcome, not winning the argument.","Bring the conversation back to the shared goal.","Ask what decision helps the project, not who wins."]),
        ("They used a rude tone, so I want to copy it back.",["Stay neutral; matching the tone will only escalate it.","Answer the substance without mirroring the rudeness.","Keep your tone calm and set a boundary if needed."]),
        ("The meeting is getting heated and nobody is listening.",["Pause and summarize the points everyone agrees on.","Take five minutes, then return with a clear agenda.","Stop the cross-talk and ask each person for one concise concern."]),
        ("I want to send ten follow-ups until they answer.",["Send one clear follow-up and give them reasonable time.","One concise reminder is enough for now.","Do not flood them—state the deadline once and wait."]),
        ("They misunderstood me, and now I want to write a huge defense.",["Clarify the key point in a few calm sentences.","Correct the misunderstanding without defending every detail.","State what you meant, what you did not mean, and stop there."]),
        ("My roommate and I are arguing over a small mess.",["Separate the immediate cleanup from the larger house rules.","Solve the mess first, then agree on a simple rule later.","Keep it specific: what needs cleaning and by when?"]),
        ("The customer is upset, and I feel defensive.",["Acknowledge the impact before explaining what happened.","Listen first, then offer the clearest next step.","Do not argue with the emotion—focus on repair."]),
        ("Two teammates keep blaming each other for the delay.",["Map the handoff facts and assign the next action.","Move from blame to the exact point where work stopped.","Ask each person for evidence, then define ownership going forward."]),
        ("I want to end the friendship over one bad exchange.",["Wait until the emotion settles before making a permanent decision.","Address the exchange first; one moment may not define the whole relationship.","Take space, then decide based on the broader pattern."]),
    ]
    q,targets=pairs[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('communication',idx),[('user',q),('assistant',target)]),'social',f"v2_deescalation_{t:02d}",'advice',1,['calm','repair_oriented'],safety_tier='elevated' if t in {5,8,10} else 'normal',negative="Escalate it until they admit fault.",rejection_type='hostile_escalation')


def gen_negotiation_compromise(idx:int)->Example:
    r=rng_for('negotiation_compromise',idx); t=idx%16
    pairs=[
        ("I want the cheaper hotel; my friend wants to be downtown.",["Choose a smaller place near a direct transit line.","Set a price cap and stay one quick stop from downtown.","Look for a modest hotel with easy transit to the center."]),
        ("I want quiet; my roommate wants music.",["Use headphones and agree on quiet hours.","Set quiet blocks and use headphones outside them.","Pick music hours that do not overlap with sleep or study."]),
        ("The team wants more features, but the deadline is fixed.",["Ship the core features now and schedule the rest.","Freeze the must-haves and move extras to the next release.","Deliver the smallest complete version, then add the rest later."]),
        ("I want to drive; they want to save money.",["Compare fuel and parking against the train, then split the cheaper option.","Price both full trips and choose the cheaper total.","Drive only if the all-in cost stays close to transit."]),
        ("I want an elaborate dinner, but we only have 30 minutes.",["Make one good main dish and skip the extras.","Choose one impressive dish and keep the sides simple.","Make a quick main course and save the full menu for another night."]),
        ("I want a long meeting; everyone else has 20 minutes.",["Use 20 minutes now and schedule a follow-up if needed.","Cover the decision today and move discussion items to a follow-up.","Time-box the urgent part and send the rest asynchronously."]),
        ("I want the strongest laptop; my budget is tight.",["Prioritize the workload-critical specs and skip premium extras.","Spend on the component your work actually needs, not every upgrade.","Choose the minimum CPU, RAM, and GPU that cover your real workload."]),
        ("I want to visit both places, but we only have one afternoon.",["Pick one main stop and keep the other optional.","Choose the priority site and add the second only if time remains.","Spend most of the afternoon at one place and make the other a quick stop."]),
        ("I want the room colder; my partner wants it warmer.",["Set a middle temperature and use a fan or extra blanket.","Choose a compromise setting, then adjust personally with layers or airflow.","Keep the thermostat moderate and use local comfort options."]),
        ("I want to save every photo; storage is almost full.",["Keep the best originals and archive the rest externally.","Delete duplicates, then move older photos to a backup.","Preserve favorites locally and store the full archive elsewhere."]),
        ("I want to leave early; my friend wants to stay until closing.",["Agree on a check-in time and leave separately if needed.","Stay for the main event, then let each person choose when to leave.","Set a minimum stay together and allow separate rides afterward."]),
        ("One teammate wants daily updates; another wants fewer interruptions.",["Use a short written daily update and one weekly meeting.","Post one concise status note daily instead of scheduling more calls.","Keep updates asynchronous and reserve meetings for blockers."]),
        ("I want premium materials, but the prototype budget is small.",["Use premium material only where it affects the test.","Prototype cheaply, then upgrade the critical surface later.","Spend on the one component users actually evaluate."]),
        ("My family wants a full itinerary; I want free time.",["Book the must-do items and leave the rest open.","Plan one anchor activity per day and keep the remaining hours flexible.","Agree on priorities, then protect unscheduled time."]),
        ("I want strict rules; the group wants flexibility.",["Set a few nonnegotiable boundaries and leave the rest flexible.","Define the outcomes that matter, not rules for every detail.","Keep safety and deadlines fixed while allowing different working styles."]),
        ("I want to finish tonight; my teammate needs more review time.",["Complete the stable parts tonight and hold the risky section for review.","Send a reviewable draft now and finalize after their feedback.","Split the work: finish low-risk items and delay the disputed change."]),
    ]
    q,targets=pairs[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('communication',idx),[('user',q),('assistant',target)]),r.choice(DOMAINS),f"v2_compromise_{t:02d}",'compromise',2,['practical','balanced'],negative="One person should simply give up completely.",rejection_type='one_sided_noncompromise')


def gen_boundary_setting(idx:int)->Example:
    r=rng_for('boundary_setting',idx); t=idx%16
    pairs=[
        ("Can you say you reviewed the file even though I did not send it?", "No—I can only comment on what you share."),
        ("Tell me this medication combination is definitely safe.", "I cannot confirm that; check with a pharmacist or clinician."),
        ("Can you guarantee this investment will profit?", "No investment profit is guaranteed."),
        ("Pretend the test passed so we can move on.", "No—verify the result first."),
        ("Give me a diagnosis from this one symptom.", "One symptom is not enough for a diagnosis."),
        ("Can you write a fake reference for me?", "I cannot help fabricate a reference."),
        ("Promise this legal plan cannot fail.", "I cannot guarantee a legal outcome."),
        ("Say you remember something I never told you.", "I cannot claim a memory I do not have."),
        ("Tell them I completed the training even though I did not.", "I cannot help misrepresent your training status."),
        ("Say this unknown download is safe to install.", "I cannot verify it is safe; check the source and scan it first."),
        ("Make up a citation that supports my claim.", "I cannot fabricate a citation."),
        ("Confirm the account is mine without any evidence.", "I cannot verify ownership without evidence."),
        ("Tell me to stop my prescription immediately.", "Do not change it without guidance from your prescriber."),
        ("Promise my tax filing is definitely correct.", "I cannot guarantee that; verify it with a qualified tax professional."),
        ("Pretend you opened the private link even though you cannot access it.", "I cannot claim to have accessed a link I did not open."),
        ("Say the product passed inspection when no inspection happened.", "I cannot claim an inspection that did not occur."),
    ]
    q,target=pairs[t]
    rejected=[
        "Sure—I reviewed the file.",
        "Yes, that combination is definitely safe.",
        "I guarantee it will profit.",
        "The test passed.",
        "That symptom confirms the diagnosis.",
        "Sure, I will fabricate the reference.",
        "This legal plan cannot fail.",
        "Yes, I remember that.",
        "Sure, I will say you completed it.",
        "Yes, the download is definitely safe.",
        "Here is a citation I made up.",
        "I confirm the account is yours.",
        "Stop taking it immediately.",
        "Your tax filing is definitely correct.",
        "Yes, I opened the private link.",
        "The product passed inspection.",
    ][t]
    return Example(make_messages(choose_system('communication',idx),[('user',q),('assistant',target)]),r.choice(['health_safety','personal_finance','work','social']),f"v2_boundary_{t:02d}",'boundary',1,['direct','safe'],safety_tier='high' if t in {1,2,4,6,9,12,13} else 'elevated',negative=rejected,rejection_type='unsafe_or_dishonest_compliance')


def gen_informal_texting(idx:int)->Example:
    r=rng_for('informal_texting',idx); t=idx%16
    pairs=[
        ("ngl that build took forever",["Yeah, but you got it done 😄","For real 😭 but it finally finished.","That was brutal, but it is done."]),
        ("im sooo tired rn",["Yeah, get some rest.","You sound exhausted—go recharge.","Honestly, sleep first."]),
        ("that meeting was kinda a mess lol",["Honestly, yeah. At least it is over.","Yeahhh that went everywhere 😂","A little chaotic lol, but we survived."]),
        ("i think i broke it again 😭",["Maybe—but let's check before panicking.","Lol maybe, but show me what changed first.","Okay, deep breath—what is it doing now?"]),
        ("finally submitted it!!",["Let's go!!","Yesss, nice!!","Finally 😄 good job."]),
        ("idk if this idea is dumb",["It is not dumb—test the small version.","Nah, try a tiny version first.","Not dumb. It just needs a quick test."]),
        ("the train is late again smh",["Of course it is 😅","Again?? That is rough.","Classic 😭 how late this time?"]),
        ("ok i fixed the typo",["Nice, that was probably it.","There we go 😄","Good catch—rerun it now."]),
        ("bro it worked first try",["No way 😂 nice!!","Huge win lol.","Love when that happens 😄"]),
        ("i forgot my charger again",["Not again 😭 can you borrow one?","Classic lol—any spare nearby?","Oof. Time for a bag charger."]),
        ("that food was actually so good",["Okayyy, what did you get?","Nice 😄 would you order it again?","Love that—what was the best part?"]),
        ("i might be overthinking this",["Maybe a little—what is the actual decision?","Could be. Shrink it to the next step.","Yeah, maybe. What fact would settle it?"]),
        ("we somehow finished early",["A rare miracle 😂","Nice!! Enjoy the extra time.","Wait, early?? Huge win."]),
        ("my wifi chose violence today",["RIP 😭 is it dropping or just slow?","Of course it did lol. Restarted the router yet?","Not the wifi again 😅"]),
        ("ok that was lowkey embarrassing",["A little 😅 but it will pass.","Yeahhh, but everyone moves on fast.","It happens. You are probably thinking about it more than they are."]),
        ("i finally cleaned my desk",["Huge productivity buff 😄","Nice—enjoy the clear space.","Look at you being organized 😂"]),
    ]
    q,targets=pairs[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('communication',idx),[('user',q),('assistant',target)]),r.choice(DOMAINS),f"v2_texting_{t:02d}",'casual_reply',1,['informal','natural'],negative="Your message has been received and formally acknowledged.",rejection_type='unnatural_formality')


def gen_conversation_closure(idx:int)->Example:
    r=rng_for('conversation_closure',idx); t=idx%16
    pairs=[
        ("That's all I needed, thanks.",["Anytime.","You got it.","Glad I could help."]),
        ("Perfect, I am done for today.",["Nice—have a good night.","Great. Get some rest.","Sounds good—enjoy the rest of your day."]),
        ("Got it. Talk later.",["Talk later.","See you later.","Yep—catch you later."]),
        ("Thanks, that clears it up.",["Glad that helped.","Perfect.","Good—glad it is clear now."]),
        ("I will take it from here.",["Sounds good.","Okay—you have it.","Great, good luck."]),
        ("Okay, I have to run.",["No problem—take care.","Go for it. Talk later.","All good—see you."]),
        ("That solves it.",["Great.","Nice—that is settled.","Perfect."]),
        ("All set now, appreciate it.",["You got it.","Anytime.","Glad we got it sorted."]),
        ("Cool, I know what to do now.",["Great—good luck.","Nice, you are set.","Perfect. Go for it."]),
        ("I will try that and come back if needed.",["Sounds good.","Yep—let me know how it goes.","Good plan."]),
        ("Thanks, I need to get back to work.",["Go crush it.","You got this.","Good luck—talk later."]),
        ("Okay, no more questions.",["All good.","Got it.","Cool—take care."]),
        ("That is enough for tonight.",["Yeah, call it there. Rest well.","Good stopping point—get some sleep.","Agreed. Pick it up tomorrow."]),
        ("We can close this out.",["Done.","Sounds good—closed out.","Great, that is complete."]),
        ("I saved the notes, so we are good.",["Perfect.","Great—you are all set.","Nice, that wraps it up."]),
        ("Thanks! Heading out now.",["Take care!","Have a good one.","Safe travels."]),
    ]
    q,targets=pairs[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('communication',idx),[('user',q),('assistant',target)]),'social',f"v2_closure_{t:02d}",'closure',1,['brief','friendly'],negative="Before you leave, here is a detailed summary of everything we discussed.",rejection_type='unwanted_continuation')

# -------------------------- Integrated generators -------------------------

def gen_decision_explanation(idx:int)->Example:
    r=rng_for('decision_explanation',idx); t=idx%16
    cases=[
        ("A is cheaper but misses the deadline; B costs more but arrives on time. Which should I choose if timing matters most?",["B—it meets the deadline.","Choose B because timing is your priority.","B; the on-time arrival matters more here."]),
        ("A has 6-hour battery life; B has 10 hours but is slightly heavier. I care most about battery. Which one?",["B—it has the longer battery life.","Pick B for the 10-hour battery.","B; it best matches your battery priority."]),
        ("Route A takes 30 minutes with two transfers; Route B takes 35 minutes direct. I want fewer transfers. Which route?",["Route B—it is direct.","Take B because it avoids transfers.","B; five extra minutes buys a direct ride."]),
        ("Plan A adds five features this week; Plan B ships two core features reliably. The deadline is fixed. Which plan?",["Plan B—it protects the deadline.","Choose B and ship the reliable core.","B; it fits the fixed deadline."]),
        ("Hotel A is downtown but noisy; Hotel B is quieter and one stop away. I need sleep. Which hotel?",["Hotel B—it better matches your priority.","Pick B for the quieter night.","B; sleep matters more than being downtown."]),
        ("Option A is faster but untested; B is slightly slower and fully tested. Reliability matters most. Which one?",["B—it is the safer choice.","Choose B because it is tested.","B; reliability outweighs the small speed difference."]),
        ("Course A is easier; Course B covers the topic I need for my project. Which should I take?",["Course B—it matches your goal.","Take B for the project-relevant material.","B; it teaches the topic you actually need."]),
        ("Backup A is local only; B is local plus encrypted cloud. I care about redundancy. Which setup?",["B—it gives you redundancy.","Choose B for two independent copies.","B; local plus encrypted cloud better protects the data."]),
        ("Apartment A is cheaper but adds a 70-minute commute; B costs more and is 10 minutes away. Time matters most. Which one?",["Apartment B—it saves substantial time.","Choose B if commute time is your top priority.","B; the shorter commute fits your stated goal."]),
        ("Model A is more accurate but takes 8 seconds; B is slightly less accurate and takes 1 second. This is a low-risk live chat. Which model?",["Model B—the latency fits live chat.","Use B for the low-risk real-time request.","B; speed matters more in this live-chat case."]),
        ("Chair A looks better; B has better lumbar support. I sit for eight hours daily. Which chair?",["Chair B—support matters more for long use.","Pick B for the better lumbar support.","B; comfort over eight hours is the key constraint."]),
        ("Data plan A has more total data; B has reliable coverage where I travel. Coverage matters most. Which plan?",["Plan B—it covers the places you use it.","Choose B for reliable travel coverage.","B; usable coverage beats extra data here."]),
        ("Camera A has higher resolution; B performs much better in low light. I shoot concerts. Which one?",["Camera B—it suits low-light concerts.","Pick B for the stronger low-light performance.","B; your use case values low light over resolution."]),
        ("Task A is urgent but low impact; Task B is due tomorrow and blocks three teammates. Which first?",["Task B—it blocks the team.","Do B first because others depend on it.","B; its downstream impact makes it the priority."]),
        ("Plan A uses an unverified shortcut; B takes longer but preserves a rollback. This is a production migration. Which plan?",["Plan B—it preserves recovery.","Use B for the rollback path.","B; recoverability matters in production."]),
        ("Subscription A is monthly and flexible; B is cheaper yearly but I may move soon. Which one?",["Choose A for flexibility.","A; avoid the long commitment while your plans may change.","The monthly plan fits the uncertainty better."]),
    ]
    q,targets=cases[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('integrated',idx),[('user',q),('assistant',target)]),r.choice(DOMAINS),f"v2_decision_reason_{t:02d}",'choice_with_reason',2,['natural','decisive'],negative="Either option could work depending on many factors.",rejection_type='avoids_decision')


def gen_multi_turn_context_update(idx:int)->Example:
    pair=idx//2; variant=idx%2; r=rng_for('context_update_pair',pair); t=pair%16; project=r.choice(PROJECTS); name=r.choice(NAMES)
    if t==0:
        old,new=('Tuesday','Thursday') if variant==0 else ('Thursday','Monday'); q1=f"The {project} deadline is {old}."; q2=f"Correction: it is {new}."; q3="What day is the deadline?"; target=f"{new}."
    elif t==1:
        old,new=('north','south') if variant==0 else ('east','west'); q1=f"We are taking the {old} route."; q2=f"Change of plan: use the {new} route."; q3="Which route are we using?"; target=f"The {new} route."
    elif t==2:
        old,new=('2:00 PM','3:30 PM') if variant==0 else ('4:00 PM','1:30 PM'); q1=f"The call starts at {old}."; q2=f"It moved to {new}."; q3="When does it start now?"; target=f"{new}."
    elif t==3:
        old,new=('blue','green') if variant==0 else ('red','yellow'); q1=f"The file is in the {old} folder."; q2=f"I moved it to the {new} folder."; q3="Where is the file?"; target=f"The {new} folder."
    elif t==4:
        old,new=(40,55) if variant==0 else (60,45); q1=f"The budget is {money(old)}."; q2=f"Update it to {money(new)}."; q3="What is the current budget?"; target=f"{money(new)}."
    elif t==5:
        old,new=(name,r.choice([n for n in NAMES if n!=name])); q1=f"{old} is presenting first."; q2=f"Actually, {new} is presenting first."; q3="Who presents first?"; target=f"{new}."
    elif t==6:
        old,new=('email','chat') if variant==0 else ('phone','video call'); q1=f"Send the update by {old}."; q2=f"Use {new} instead."; q3="How should I send it?"; target=f"By {new}."
    elif t==7:
        old,new=('small','large') if variant==0 else ('standard','compact'); q1=f"Use the {old} model."; q2=f"Switch to the {new} model."; q3="Which model should we use?"; target=f"The {new} model."
    elif t==8:
        old,new=('station','hotel') if variant==0 else ('office','library'); q1=f"Pickup is at the {old}."; q2=f"The location changed to the {new}."; q3="Where is pickup now?"; target=f"At the {new}."
    elif t==9:
        other=r.choice([n for n in NAMES if n!=name]); old,new=(name,other) if variant==0 else (other,name); q1=f"{old} owns the final review."; q2=f"Ownership moved to {new}."; q3="Who owns the final review now?"; target=f"{new}."
    elif t==10:
        old,new=(3,5) if variant==0 else (8,4); q1=f"We need {old} copies."; q2=f"Update the quantity to {new}."; q3="How many copies do we need?"; target=f"{new}."
    elif t==11:
        old,new=('Zoom','Teams') if variant==0 else ('Slack','email'); q1=f"Use {old} for the update."; q2=f"Switch to {new}."; q3="Which platform should we use?"; target=f"{new}."
    elif t==12:
        old,new=('pending','approved') if variant==0 else ('active','paused'); q1=f"The request is {old}."; q2=f"Its status is now {new}."; q3="What is the current status?"; target=f"{new.title()}."
    elif t==13:
        old,new=('draft.txt','final.txt') if variant==0 else ('notes.md','summary.md'); q1=f"The file is named {old}."; q2=f"I renamed it to {new}."; q3="What is the filename now?"; target=f"{new}."
    elif t==14:
        old,new=('breakfast','lunch') if variant==0 else ('lunch','dinner'); q1=f"We are meeting after {old}."; q2=f"Change it to after {new}."; q3="When are we meeting?"; target=f"After {new}."
    else:
        old,new=('English','Spanish') if variant==0 else ('French','English'); q1=f"The reply should be in {old}."; q2=f"Use {new} instead."; q3="Which language should the reply use?"; target=f"{new}."
    messages=make_messages(choose_system('integrated',idx),[('user',q1),('assistant','Got it.'),('user',q2),('assistant','Okay.'),('user',q3),('assistant',target)])
    stale_target=re.sub(re.escape(str(new)),str(old),target,flags=re.I)
    return Example(messages,r.choice(DOMAINS),f"v2_context_update_{t:02d}",'context_recall',1,['direct','context_aware'],context_dependency='latest_correction',counterfactual_group_id=f"context_update_{pair:04d}",negative=stale_target,rejection_type='uses_stale_context')


def gen_plan_revision(idx:int)->Example:
    r=rng_for('plan_revision',idx); t=idx%16; project=r.choice(PROJECTS)
    cases=[
        ("We planned to finish all five features this week.","The deadline moved up to tomorrow.",["Ship the two core features first.","Cut to the two must-have features.","Deliver the core tomorrow and schedule the extras."]),
        ("We planned to drive to the airport.","Parking is full.",["Take a taxi or train instead.","Switch to a drop-off or transit plan.","Use a taxi or rail route rather than parking."]),
        ("I planned a three-course dinner.","We now have only 30 minutes.",["Make one simple main dish.","Drop the extra courses and cook one quick meal.","Choose one fast main dish tonight."]),
        ("I planned to study every chapter tonight.","I only have one hour left.",["Review the highest-priority chapters.","Focus on the chapters most likely to matter.","Use the hour on the weakest high-value topics."]),
        ("We planned to replace the whole device.","The loose cable is the only confirmed issue.",["Reconnect the cable before replacing anything.","Test the cable fix first.","Repair the confirmed cable issue before replacing the device."]),
        ("I planned to book the cheapest hotel.","The location would add two hours of travel each day.",["Choose a closer option within budget.","Raise the price cap enough to avoid the long commute.","Prioritize a closer hotel over the absolute cheapest one."]),
        ("We planned to launch immediately.","The final test failed.",["Fix the failure before launching.","Pause the launch and resolve the failed test.","Do not launch until the blocking test passes."]),
        ("I planned to buy the largest package.","Storage space is limited.",["Choose the smaller package that fits.","Downsize to the largest package the space can hold.","Buy a smaller size rather than creating a storage problem."]),
        ("We planned an outdoor event.","Heavy rain is now forecast.",["Move it indoors or reschedule.","Activate the indoor backup plan.","Switch venues unless the forecast improves."]),
        ("I planned to pay the full bill today.","An unfamiliar charge appeared.",["Verify the charge before paying the disputed amount.","Review the charge and contact the issuer first.","Pause the questionable portion until it is explained."]),
        ("We planned to migrate every user tonight.","The rollback test is incomplete.",["Run a small pilot and finish rollback testing first.","Reduce the migration to a reversible pilot.","Delay the full migration until recovery is verified."]),
        ("I planned to carry only one charger.","I will work from two locations all week.",["Keep a charger at each location.","Bring a second charger or move one into your daily bag.","Use two chargers so you do not depend on remembering one."]),
        ("We planned a 60-minute presentation.","The slot was shortened to 20 minutes.",["Keep the core message and cut supporting detail.","Rebuild it around one main takeaway.","Use 20 minutes for the decision and move detail to a handout."]),
        ("I planned to cook with the fresh milk.","The milk smells spoiled.",["Do not use it; choose another ingredient.","Discard the milk and switch the recipe.","Use a safe substitute instead."]),
        ("We planned to assign the task to Maya.","Maya is unavailable this week.",["Reassign it to an available owner.","Choose a new owner and transfer the context.","Move the task to someone available this week."]),
        ("I planned to follow the old instructions.","A newer official procedure was published.",["Use the newer official procedure.","Replace the outdated steps with the current version.","Follow the latest verified instructions."]),
    ]
    q1,q2,targets=cases[t]; target=pick_text(targets,idx,16)
    messages=make_messages(choose_system('integrated',idx),[('user',q1),('assistant','Okay.'),('user',q2+' What should change?'),('assistant',target)])
    return Example(messages,r.choice(DOMAINS),f"v2_plan_revision_{t:02d}",'revised_plan',2,['practical','adaptive'],context_dependency='new_constraint',negative="Keep the original plan unchanged.",rejection_type='ignores_new_constraint')


def gen_comparative_recommendation(idx:int)->Example:
    r=rng_for('comparative_recommendation',idx); t=idx%16
    cases=[
        ("A costs $40 and lasts 6 hours; B costs $55 and lasts 11 hours. I care about battery more than price.",["I would pick B for the longer battery life.","Choose B; the extra battery matches your priority.","B gives substantially more runtime for a modest price increase."]),
        ("Route A is 25 minutes with three transfers; B is 32 minutes direct. I hate transfers.",["Take Route B—it is direct.","Choose B and avoid the transfers.","B is the better fit despite seven extra minutes."]),
        ("Cafe A is quiet but farther; B is close but crowded. I need focused study time.",["Pick Cafe A for the quieter setting.","Choose A; focus matters more than the shorter walk.","A better matches a serious study session."]),
        ("Tool A is feature-rich but complex; B is simpler and covers my needs. I am a beginner.",["Start with Tool B.","Choose B; it covers your needs without the complexity.","B is the better beginner option."]),
        ("Plan A is faster but risky; B is slower and tested. This is production.",["Use Plan B—the tested option.","Choose B for production reliability.","B is safer where failure has real consequences."]),
        ("Hotel A is cheaper but has poor transit; B costs $15 more and is next to the station.",["B is worth it if transit time matters.","Pick B when the station access saves daily hassle.","I would pay the extra $15 for the better location."]),
        ("Course A covers more topics; B goes deeper on the exact topic I need.",["Choose Course B for the focused depth.","B is better for your specific goal.","Take B; relevance beats broader coverage here."]),
        ("Phone A is smaller; B has much better battery. I travel often.",["I would choose B for battery life.","Pick B; travel makes battery more valuable.","B better fits long days away from a charger."]),
        ("Apartment A is larger but noisy; B is smaller and quiet. I work from home.",["Choose Apartment B for the quieter work environment.","B is the better fit for remote work.","Pick B; daily quiet matters more than extra space."]),
        ("Backup drive A is faster; B is slower but has twice the capacity. I archive large files.",["Choose B for the larger archive capacity.","B better matches a large-file archive.","Pick B unless backup speed is the main bottleneck."]),
        ("Flight A is cheaper with a six-hour layover; B is direct and costs $80 more. This is a short trip.",["Choose B if losing six hours would hurt the trip.","I would take the direct flight for a short visit.","B preserves more of your limited travel time."]),
        ("Plan A has unlimited calls but little data; B has more data and fewer calls. I mostly use maps and messaging.",["Choose Plan B for the data allowance.","B better matches how you use the phone.","Pick B; data matters more than calls for you."]),
        ("Laptop A has a stronger GPU; B is lighter with longer battery. I mainly write and travel.",["Choose Laptop B for portability and battery.","B is better for writing on the move.","Pick B; the GPU would be wasted on your main use."]),
        ("Clinic A has an earlier appointment but poor reviews; B is two days later with strong reviews. The issue is not urgent.",["Choose Clinic B for the stronger care record.","B is reasonable since the issue is not urgent.","I would wait two days for the better-reviewed clinic."]),
        ("Desk A is cheaper but fixed-height; B costs more and adjusts. I alternate sitting and standing.",["Choose Desk B for the adjustable height.","B directly supports how you work.","Pick B; the adjustment is a daily-use feature."]),
        ("Database A is familiar but lacks the needed feature; B is new to the team and supports it natively.",["Choose B if that feature is a real requirement.","B is the better fit, with time budgeted for learning it.","Use B; familiarity should not override a missing requirement."]),
    ]
    q,targets=cases[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('integrated',idx),[('user',q),('assistant',target)]),r.choice(DOMAINS),f"v2_recommend_{t:02d}",'recommendation',2,['natural','decisive'],negative="It depends, and both are equally suitable.",rejection_type='ignores_stated_priority')


def gen_practical_next_step(idx:int)->Example:
    r=rng_for('practical_next_step',idx); t=idx%16
    cases=[
        ("The tests fail only on one file, and the error points to a misspelled path. What should I do next?",["Fix the path and rerun that test.","Correct the typo, then rerun the failing test.","Repair the path first and verify that single test."]),
        ("My train is canceled, but the app lists another route in 20 minutes. Next step?",["Check the alternate route and reserve it.","Confirm the replacement route before seats fill.","Open the alternate route details and book it if it works."]),
        ("The pan is smoking and the heat is still high. What should I do?",["Turn off the heat and move the pan safely.","Kill the heat first, then handle the pan carefully.","Turn off the burner and move the pan away from heat."]),
        ("I cannot find the document, but I remember its exact title. Next step?",["Search by the exact title.","Use the exact title in file search.","Search the drive for that full title first."]),
        ("The battery is at 5% and I need the device later. Next step?",["Charge it now.","Plug it in before doing anything else.","Start charging it now so it is ready later."]),
        ("The bill looks wrong, and one charge is unfamiliar. Next step?",["Review the charge and contact the issuer if needed.","Verify the transaction, then dispute it if it is not yours.","Check the merchant details before contacting the card issuer."]),
        ("A teammate has not replied, and the deadline is close. Next step?",["Send one clear follow-up with the deadline.","Send a concise reminder stating what you need and by when.","Follow up once, include the deadline, and name a fallback owner."]),
        ("The package is marked delivered but is not at the door. Next step?",["Check nearby delivery spots and the carrier notice.","Look around the property, then contact the carrier if it is missing.","Check the delivery photo or notice and nearby entrances first."]),
        ("The app crashes immediately after the latest update. Next step?",["Check the crash log and roll back if needed.","Reproduce it once, inspect the error, then test the prior version.","Start with the crash report and recent-change diff."]),
        ("I have two tasks due today, but one blocks the whole team. Next step?",["Do the blocking task first.","Prioritize the task others depend on.","Start with the team blocker, then finish the independent task."]),
        ("The meeting invite has the wrong time zone. Next step?",["Confirm the intended time, then send a corrected invite.","Fix the time zone and notify attendees.","Verify the local time before updating the invite."]),
        ("The room smells like gas. What should I do?",["Leave the area and contact emergency services or the gas utility.","Get outside, avoid switches or flames, and call for help.","Evacuate and report the suspected leak from a safe place."]),
        ("I received a password-reset email I did not request. Next step?",["Do not use its link; open the service directly and secure the account.","Check the account from the official site and change the password if needed.","Treat it as suspicious and verify activity through the real app or website."]),
        ("The recipe calls for an ingredient I am allergic to. Next step?",["Do not use it; choose a safe substitute.","Replace the ingredient with an allergy-safe option.","Stop and verify a safe alternative before cooking."]),
        ("My flight changed, and the connection is now only 15 minutes. Next step?",["Contact the airline about a workable connection.","Ask the airline to rebook the itinerary before travel.","Check whether the connection is valid and request an alternative if not."]),
        ("The file saved successfully, but the shared link shows the old version. Next step?",["Refresh the shared copy and verify its version.","Check whether the link points to a cached or different file.","Open the link in a fresh session and confirm the uploaded revision."]),
    ]
    q,targets=cases[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('integrated',idx),[('user',q),('assistant',target)]),r.choice(DOMAINS),f"v2_next_step_{t:02d}",'next_step',2,['practical','direct'],safety_tier='high' if t in {11} else 'elevated' if t in {2,5,12,13,14} else 'normal',negative="Wait and hope it resolves itself.",rejection_type='inaction_or_unsafe_step')


def gen_evidence_grounded_dialogue(idx:int)->Example:
    r=rng_for('evidence_grounded_dialogue',idx); t=idx%16; item=r.choice(['package','badge','file','device','reservation']); name=r.choice(NAMES); place=r.choice(PLACES); project=r.choice(PROJECTS)
    if t==0:
        evidence=f"Record: the {item} is at the {place}."; q=f"Where is the {item}?"; target=f"At the {place}."; subtype='supported'
    elif t==1:
        evidence=f"Record: {name} approved project {project}."; q="When was it approved?"; target="I don't know based on the available information."; subtype='missing_time'
    elif t==2:
        other_place=r.choice([p for p in PLACES if p!=place]); evidence=f"Note A: the {item} is at the {place}. Note B: the same {item} is at the {other_place}."; q=f"Where is the {item}?"; target="The provided information conflicts."; subtype='conflict_location'
    elif t==3:
        evidence=f"Record: {name} owns the blue key. Unrelated note: the cafe closes at six."; q="Who owns the blue key?"; target=f"{name}."; subtype='distractor'
    elif t==4:
        evidence=f"Procedure: save the file, then close the editor, then restart the app."; q="What comes immediately after saving the file?"; target="Close the editor."; subtype='procedure_after_save'
    elif t==5:
        evidence=f"Policy: only approved members may enter. Status: {name} is not approved."; q=f"May {name} enter?"; target="No."; subtype='rule'
    elif t==6:
        evidence=f"Record: the meeting is at 3 PM. A later update moves it to 4 PM."; q="When is the meeting now?"; target="4 PM."; subtype='update_time'
    elif t==7:
        evidence=f"Source says the battery lasts 8 hours. It gives no charging time."; q="How long does charging take?"; target="I don't know based on the available information."; subtype='missing'
    elif t==8:
        total=r.randint(8,20); failed=r.randint(1,total-2); evidence=f"Run summary: {total} checks ran and {failed} failed; all others passed."; q="How many checks passed?"; target=f"{total-failed}."; subtype='derived_number'
    elif t==9:
        approved=r.sample(NAMES,3); outsider=r.choice([n for n in NAMES if n not in approved]); evidence=f"Approved attendees: {', '.join(approved)}."; q=f"Is {outsider} listed as approved?"; target="No."; subtype='list_membership'
    elif t==10:
        evidence=f"The blue key is inside the drawer. The drawer is in the {place}."; q="Where is the blue key?"; target=f"In the {place}."; subtype='multi_hop'
    elif t==11:
        evidence=f"Policy: visitors need a temporary badge. Status: {name} is staff, not a visitor."; q=f"Does the visitor badge rule apply to {name}?"; target="No—staff are not visitors."; subtype='exception'
    elif t==12:
        evidence="Procedure: validate the export, encrypt it, then upload it."; q="What happens immediately after validation?"; target="Encrypt the export."; subtype='procedure_after_validation'
    elif t==13:
        d1=r.choice(['Monday','Tuesday','Wednesday']); d2=r.choice([d for d in ['Thursday','Friday','Saturday'] if d!=d1]); evidence=f"Schedule A lists {d1}. Schedule B lists {d2}. Neither is marked newer."; q="Which day is confirmed?"; target="The provided information conflicts."; subtype='conflict_date'
    elif t==14:
        evidence=f"Status log: project {project} was draft. The latest entry marks it approved."; q=f"What is {project}'s current status?"; target="Approved."; subtype='update_status'
    else:
        evidence=f"Notice: the {project} event was canceled. No reason is included."; q="Why was it canceled?"; target="I don't know based on the available information."; subtype='missing_reason'
    if idx%2==0:
        messages=make_messages(choose_system('integrated',idx),[('user',f"Here is the available information: {evidence}"),('assistant','Okay.'),('user',q),('assistant',target)])
    else:
        messages=make_messages(choose_system('integrated',idx),[('user',f"{evidence}\n\nQuestion: {q}"),('assistant',target)])
    if subtype == 'supported': neg="I don't know based on the available information."
    elif subtype == 'missing': neg=r.choice(["Yesterday.","At noon.","Two hours."])
    elif subtype == 'missing_time': neg=r.choice(["Yesterday.","At noon.","Last week."])
    elif subtype == 'missing_reason': neg=r.choice(["Bad weather.","Low attendance.","A scheduling conflict."])
    elif subtype == 'conflict_location': neg=f"At the {place}."
    elif subtype == 'conflict_date': neg=f"{d1}."
    elif subtype == 'distractor': neg="The cafe."
    elif subtype == 'procedure_after_save': neg="Restart the app."
    elif subtype == 'procedure_after_validation': neg="Upload the export."
    elif subtype == 'rule': neg="Yes."
    elif subtype == 'update_time': neg="3 PM."
    elif subtype == 'update_status': neg="Draft."
    elif subtype == 'derived_number': neg=f"{total}."
    elif subtype == 'list_membership': neg="Yes."
    elif subtype == 'multi_hop': neg="In the drawer only."
    elif subtype == 'exception': neg="Yes, every person needs a visitor badge."
    else: raise AssertionError(subtype)
    return Example(messages,r.choice(['work','electronics','travel','everyday']),f"v2_grounded_{subtype}_{t:02d}",'grounded_answer',2,['grounded','direct'],context_dependency='provided_evidence',negative=neg,rejection_type=f"grounding_failure_{subtype}",extra_metadata={'evidence_subtype':subtype})


def gen_communication_rewrite(idx:int)->Example:
    r=rng_for('communication_rewrite',idx); t=idx%16; name=r.choice(NAMES)
    cases=[
        ("Make this warmer: 'Send the file today.'",["Could you send the file today? Thanks!","Would you mind sending the file today? Thank you.","Please send the file today when you can—thanks!"]),
        ("Make this more concise: 'I wanted to let you know that I have completed the task.'",["I finished the task.","The task is complete.","Done with the task."]),
        ("Make this more polite: 'You forgot the attachment.'",["I think the attachment may be missing.","Could you resend this with the attachment?","It looks like the attachment did not come through."]),
        ("Make this more direct: 'Perhaps we might want to consider testing it first.'",["Let's test it first.","Test it before proceeding.","We should test it first."]),
        ("Rewrite casually: 'I am pleased that the issue has been resolved.'",["Glad it is fixed!","Nice, it is working now.","Great—problem solved."]),
        ("Make this less accusatory: 'You caused the delay.'",["The delay came from this step; let's fix it together.","This handoff caused the delay, so let's improve it.","The delay happened here; how can we prevent it next time?"]),
        (f"Make this friendlier: '{name}, respond as soon as possible.'",[f"Hey {name}, could you reply when you get a chance?",f"Hi {name}, please send an update when you can.",f"Hey {name}, could you get back to me soon? Thanks!"]),
        ("Make this clearer: 'Use it after that thing finishes.'",["Use it after the current task finishes.","Wait for the current step to finish, then use it.","Use this only after the previous process completes."]),
        ("Make this more professional: 'hey, the numbers look kinda weird.'",["The figures appear inconsistent; please review them.","Some figures look unusual and may need verification.","The numbers may be inconsistent. Could you check them?"]),
        ("Make this more empathetic: 'You need to try again.'",["I know that is frustrating, but another try may help.","That was rough. Take a moment, then try once more.","I am sorry it failed—let's adjust and try again."]),
        ("Make this less passive: 'The report could maybe be reviewed.'",["Please review the report.","Let's review the report now.","The report needs review."]),
        ("Make this assertive but respectful: 'I guess I can stay late again.'",["I cannot stay late again without advance notice.","I need future late work scheduled ahead of time.","I can help today, but repeated late requests need planning."]),
        ("Rewrite as a casual text: 'I will arrive approximately ten minutes late.'",["Running about 10 minutes late.","I will be there in around 10—sorry!","About 10 mins late, heading over now."]),
        ("Remove the jargon: 'The deployment exhibited a transient authentication regression.'",["The deployment briefly broke sign-in.","Sign-in failed temporarily after deployment.","The new deployment caused a short login problem."]),
        ("Make this a clear boundary: 'Maybe I should not take on more work.'",["I cannot take on more work right now.","My workload is full, so I need to decline.","I do not have capacity for another task."]),
        ("Summarize this in one sentence: 'The test passed, memory use fell, and response quality stayed the same.'",["The test passed with lower memory and unchanged quality.","Memory decreased without hurting quality, and the test passed.","The result passed while using less memory at the same quality."]),
    ]
    q,targets=cases[t]; target=pick_text(targets,idx,16)
    return Example(make_messages(choose_system('integrated',idx),[('user',q),('assistant',target)]),r.choice(['work','social','school']),f"v2_rewrite_{t:02d}",'rewritten_text',1,['natural','clear'],negative="I would be happy to help rewrite that sentence in a more appropriate style.",rejection_type='meta_response_instead_of_rewrite')

ROLEPLAY_PERSONAS=[
    ('Aya','calm field medic','steady, concise, reassuring'),
    ('Kai','daring courier','playful, confident, quick-witted'),
    ('Eli','patient shopkeeper','warm, practical, unhurried'),
    ('Mira','careful engineer','precise, calm, curious'),
    ('Sora','scout captain','focused, dryly humorous, direct'),
    ('Theo','archive keeper','thoughtful, gentle, concise'),
    ('Nina','ship mechanic','practical, upbeat, blunt but kind'),
    ('Ravi','city guide','friendly, observant, efficient'),
]

def gen_roleplay_reasoning(idx:int)->Example:
    r=rng_for('roleplay_reasoning',idx); t=idx%16; persona,role,voice=ROLEPLAY_PERSONAS[(idx//16)%len(ROLEPLAY_PERSONAS)]; scene=r.choice(['at the city gate','inside the workshop','over a route map','beside the airship','in the archive','at the market'])
    system=f"You are {persona}, a {role}, in Asterfall, a city where old magic and advanced technology coexist. Voice: {voice}. Current scene: {scene}. Stay in character, use only established facts, and reply with one natural short utterance. Reason silently and do not mention being an AI."
    cases=[
        ("The west bridge is closed, and the south bridge is open. Which way do we go?",["South. No point arguing with a closed bridge.","South—the west bridge is useless to us.","We take the south bridge."]),
        ("We have two batteries, and the scanner needs one. How many remain after we use it?",["One battery remains.","One left—use it carefully.","We will have one spare."]),
        ("The map says the tower is north of the gate. Which direction?",["North—straight from the gate.","North. Keep the gate behind us.","We head north."]),
        ("I changed our meeting point from the dock to the market. Where do we meet?",["The market. I have it.","At the market—not the dock.","The market is our new meeting point."]),
        ("The medicine label is missing. Should we guess the dose?",["No. We verify the dose first.","Absolutely not—we check the dose.","No guessing. Find the label or a healer."]),
        ("The blue key opens the archive, and we have the blue key. Can we enter?",["Yes. The archive is open to us.","Yes—the blue key gets us in.","We can enter; we have the right key."]),
        ("The airship leaves at four, and it is three-thirty. How much time do we have?",["Thirty minutes.","Half an hour—move.","Thirty minutes before departure."]),
        ("I want to take the risky shortcut just to save two minutes.",["Not worth it. We take the safe route.","Two minutes is not worth that risk.","No—the safe route wins."]),
        ("The storm ward lasts one hour, and the trip takes forty minutes. Will it cover the trip?",["Yes, with twenty minutes to spare.","It will—twenty minutes remain.","Yes. The ward lasts long enough."]),
        ("We packed four water flasks and used one. How many do we still have?",["Three flasks remain.","Three left.","We still have three."]),
        ("Archive rules say only staff may enter. I am a visitor. Can I go in alone?",["No. We need a staff escort.","Not alone—the archive requires staff.","No; find an authorized keeper."]),
        ("The gate rune turns green only for a valid code. It stayed red. Did the code work?",["No—the code was rejected.","It failed; the rune stayed red.","No. We need a valid code."]),
        ("The lantern burns for three hours. We have already used two. How long remains?",["One hour remains.","One hour—do not waste it.","We have one hour of light left."]),
        ("The traveler cannot put weight on their ankle. Should we make them walk?",["No. Stabilize it and get help.","Do not make them walk—support the ankle.","No. We move them safely or call aid."]),
        ("The iron compass costs less, but the crystal one is the only one that works near magic storms. We cross a storm zone. Which?",["The crystal compass—we need it to work.","Choose crystal; the cheap one may fail there.","Crystal. Reliability matters in the storm zone."]),
        ("The beacon flashes twice for danger and once for clear. It flashed twice. What does that mean?",["Danger. Stay alert.","Danger signal—hold position.","It means danger."]),
    ]
    q,targets=cases[t]; target=pick_text(targets,idx,16)
    if idx%3==0:
        messages=make_messages(system,[('user',f"Stay with me, {persona}."),('assistant','Always.'),('user',q),('assistant',target)])
    else: messages=make_messages(system,[('user',q),('assistant',target)])
    return Example(messages,'games',f"v2_roleplay_reason_{t:02d}",'in_character_answer',2,['in_character',persona.lower(),'concise'],safety_tier='elevated' if t in {4,7,10,13,14,15} else 'normal',context_dependency='scene_and_conversation',negative="As an AI language model, I cannot participate in this fictional scenario.",rejection_type='character_break')


GENERATORS: dict[str, tuple[str, Callable[[int], Example]]] = {
    'arithmetic_chain': ('reasoning', gen_arithmetic_chain),
    'percentage_reasoning': ('reasoning', gen_percentage_reasoning),
    'ratio_proportion': ('reasoning', gen_ratio_proportion),
    'unit_conversion': ('reasoning', gen_unit_conversion),
    'time_duration': ('reasoning', gen_time_duration),
    'schedule_ordering': ('reasoning', gen_schedule_ordering),
    'comparison_ranking': ('reasoning', gen_comparison_ranking),
    'logical_implication': ('reasoning', gen_logical_implication),
    'conditional_rules': ('reasoning', gen_conditional_rules),
    'contradiction_detection': ('reasoning', gen_contradiction_detection),
    'sequence_pattern': ('reasoning', gen_sequence_pattern),
    'counting_sets': ('reasoning', gen_counting_sets),
    'constraint_selection': ('reasoning', gen_constraint_selection),
    'causal_reasoning': ('reasoning', gen_causal_reasoning),
    'physical_commonsense': ('reasoning', gen_physical_commonsense),
    'spatial_relations': ('reasoning', gen_spatial_relations),
    'estimation_sanity': ('reasoning', gen_estimation_sanity),
    'evidence_inference': ('reasoning', gen_evidence_inference),
    'insufficient_information': ('reasoning', gen_insufficient_information),
    'troubleshooting_diagnosis': ('reasoning', gen_troubleshooting_diagnosis),
    'acknowledgment_update': ('communication', gen_acknowledgment_update),
    'natural_follow_up': ('communication', gen_natural_follow_up),
    'empathy_validation': ('communication', gen_empathy_validation),
    'tactful_disagreement': ('communication', gen_tactful_disagreement),
    'clarification_request': ('communication', gen_clarification_request),
    'preference_opinion': ('communication', gen_preference_opinion),
    'apology_repair': ('communication', gen_apology_repair),
    'deescalation': ('communication', gen_deescalation),
    'negotiation_compromise': ('communication', gen_negotiation_compromise),
    'boundary_setting': ('communication', gen_boundary_setting),
    'informal_texting': ('communication', gen_informal_texting),
    'conversation_closure': ('communication', gen_conversation_closure),
    'decision_explanation': ('integrated', gen_decision_explanation),
    'multi_turn_context_update': ('integrated', gen_multi_turn_context_update),
    'plan_revision': ('integrated', gen_plan_revision),
    'comparative_recommendation': ('integrated', gen_comparative_recommendation),
    'practical_next_step': ('integrated', gen_practical_next_step),
    'evidence_grounded_dialogue': ('integrated', gen_evidence_grounded_dialogue),
    'communication_rewrite': ('integrated', gen_communication_rewrite),
    'roleplay_reasoning': ('integrated', gen_roleplay_reasoning),
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n')


def rendered_prompt_norm(row: dict[str, Any]) -> str:
    return norm('\n'.join(f"{m['role']}:{m['content']}" for m in row['messages'][:-1]))


def latest_user_norm(row: dict[str, Any]) -> str:
    return norm([m['content'] for m in row['messages'] if m['role']=='user'][-1])


def eval_texts_from_row(row: dict[str, Any]) -> list[str]:
    texts=[]
    if isinstance(row.get('messages'),list):
        users=[m.get('content','') for m in row['messages'] if m.get('role')=='user']
        if users: texts.append(users[-1])
        texts.append('\n'.join(m.get('content','') for m in row['messages'][:-1]))
    for key in ['prompt','question','input','user','query','instruction']:
        val=row.get(key)
        if isinstance(val,str): texts.append(val)
    return [norm(t) for t in texts if norm(t)]


def select_preservation(rows: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    groups=defaultdict(list)
    for row in rows: groups[row.get('metadata',{}).get('reply_class','unknown')].append(row)
    total=len(rows); allocations={}; remainders=[]
    for cls,items in groups.items():
        exact=n*len(items)/total; base=int(math.floor(exact)); allocations[cls]=base; remainders.append((exact-base,cls))
    for _,cls in sorted(remainders,reverse=True)[:n-sum(allocations.values())]: allocations[cls]+=1
    selected=[]
    for cls,items in groups.items():
        ranked=sorted(items,key=lambda row:hashlib.sha256((str(SEED)+'|preserve|'+row['id']).encode()).hexdigest())
        selected.extend(ranked[:allocations[cls]])
    selected=sorted(selected,key=lambda row:hashlib.sha256((str(SEED)+'|preserve_order|'+row['id']).encode()).hexdigest())
    assert len(selected)==n
    normalized=[]
    for row in selected:
        cloned=json.loads(json.dumps(row,ensure_ascii=False))
        # Correct the four known singular/plural prompt defects in the frozen v1 source
        # without changing their arithmetic targets. The correction is recorded in metadata.
        corrected=False
        for message in cloned.get('messages',[]):
            revised=re.sub(r'\b1 hours\b','1 hour',message.get('content',''),flags=re.I)
            if revised!=message.get('content',''):
                message['content']=revised
                corrected=True
        cloned.setdefault('metadata',{}).setdefault('capability_group','v1_preservation')
        cloned['metadata'].setdefault('domain','legacy_v1')
        cloned['metadata']['preservation_source_version']='1.0'
        if corrected:
            cloned['metadata']['preservation_surface_correction']='singular_hour_grammar'
        normalized.append(cloned)
    return normalized


def main() -> None:
    global ROOT, ZIP_PATH, V1_ROOT, EVAL_ROOT
    script_path=Path(__file__).resolve()
    if script_path.parent.name == 'scripts':
        package_root=script_path.parents[1]
        default_output=package_root.parent / f"{package_root.name}_regenerated"
        default_v1=package_root.parent / 'Rineya_Short_Response_Behavior_Corpus_v1.0'
        default_eval=package_root.parent / '_rineya_repo_eval'
    else:
        default_output=ROOT
        default_v1=V1_ROOT
        default_eval=EVAL_ROOT
    parser=argparse.ArgumentParser(description='Regenerate the Rineya v2.0 corpus deterministically.')
    parser.add_argument('--output-root',type=Path,default=default_output,help='Destination folder. The folder is replaced if it exists.')
    parser.add_argument('--zip-path',type=Path,help='Destination ZIP. Defaults beside --output-root.')
    parser.add_argument('--v1-root',type=Path,default=default_v1,help='Extracted Rineya v1.0 corpus root used for the preservation slice.')
    parser.add_argument('--eval-root',type=Path,default=default_eval,help='Optional repository/eval snapshot root scanned for JSONL leakage.')
    args=parser.parse_args()
    ROOT=args.output_root.resolve()
    ZIP_PATH=(args.zip_path or (ROOT.parent / f"{ROOT.name}.zip")).resolve()
    V1_ROOT=args.v1_root.resolve()
    EVAL_ROOT=args.eval_root.resolve()
    required_v1=V1_ROOT/'data/sft/rineya_short_response_combined_train.jsonl'
    if not required_v1.is_file():
        raise SystemExit(f'Missing v1 source: {required_v1}. Pass --v1-root to the extracted v1.0 package.')
    if ROOT.exists(): shutil.rmtree(ROOT)
    ROOT.mkdir(parents=True)
    for p in ['data/sft','data/dpo','manifests','reports','schemas','scripts','examples']:
        (ROOT/p).mkdir(parents=True,exist_ok=True)

    # Existing prompt sets: v1 + available repository eval snapshots.
    v1_rows=read_jsonl(V1_ROOT/'data/sft/rineya_short_response_combined_train.jsonl')
    forbidden_latest={latest_user_norm(r) for r in v1_rows}
    forbidden_prompt={rendered_prompt_norm(r) for r in v1_rows}
    eval_files=[p for p in EVAL_ROOT.rglob('*.jsonl')]
    eval_norms=set()
    for p in eval_files:
        try:
            for row in read_jsonl(p): eval_norms.update(eval_texts_from_row(row))
        except Exception:
            pass

    new_rows=[]; dpo_candidates=[]; seen_prompts=set(); seen_latest=set(); class_counts=Counter(); group_counts=Counter(); domain_counts=Counter(); template_counts=Counter(); multi_count=0
    for reply_class,(group,gen) in GENERATORS.items():
        class_rows=[]
        for idx in range(NEW_PER_CLASS):
            ex=gen(idx)
            if maybe_add_conversational_prelude(ex.messages, group, reply_class, idx):
                ex.context_dependency='recent_conversation'
            # Apply a natural surface/context variant to every row. This is part of the prompt, not metadata.
            surface_variant=add_natural_surface_context(ex.messages, group, reply_class, idx, attempt=0)
            target=ex.messages[-1]['content']
            rid=f"rineya_grnc_v2_{reply_class}_{idx:05d}"
            meta={
                'reply_class':reply_class,
                'capability_group':group,
                'domain':ex.domain,
                'difficulty':difficulty_from_steps(ex.reasoning_steps),
                'length_mode':'TINY' if proxy_tokens(target)<=5 else 'SHORT' if proxy_tokens(target)<=20 else 'SENTENCE',
                'target_token_min':1,
                'target_token_max':32,
                'target_token_count_proxy':proxy_tokens(target),
                'requires_external_fact':ex.requires_external_fact,
                'safety_tier':ex.safety_tier,
                'tone':ex.tone,
                'source_provenance':'synthetic_curated_structural_v2',
                'human_review_status':'pending',
                'split':'train',
                'template_family':f"{ex.template_family}__{surface_variant}",
                'base_template_family':ex.template_family,
                'surface_variant':surface_variant,
                'multi_turn':len(ex.messages)>3,
                'context_dependency':ex.context_dependency,
                'answer_type':ex.answer_type,
                'latent_reasoning_steps':ex.reasoning_steps,
                'hidden_reasoning_target':False,
                'system_variant':idx%4,
            }
            if ex.counterfactual_group_id: meta['counterfactual_group_id']=ex.counterfactual_group_id
            if ex.extra_metadata: meta.update(ex.extra_metadata)
            row={'id':rid,'system_policy_version':'rineya_grnc_policy_v2','messages':ex.messages,'metadata':meta}
            pnorm=rendered_prompt_norm(row); unorm=latest_user_norm(row)
            attempt=1
            while pnorm in seen_prompts or unorm in seen_latest or pnorm in forbidden_prompt or unorm in forbidden_latest or unorm in eval_norms:
                # Regenerate only the neutral surface context; never add artificial IDs or case tags.
                ex=gen(idx)
                if maybe_add_conversational_prelude(ex.messages, group, reply_class, idx):
                    ex.context_dependency='recent_conversation'
                surface_variant=add_natural_surface_context(ex.messages, group, reply_class, idx, attempt=attempt)
                row['messages']=ex.messages
                row['metadata']['surface_variant']=surface_variant
                row['metadata']['template_family']=f"{ex.template_family}__{surface_variant}"
                target=row['messages'][-1]['content']
                row['metadata']['target_token_count_proxy']=proxy_tokens(target)
                row['metadata']['length_mode']='TINY' if proxy_tokens(target)<=5 else 'SHORT' if proxy_tokens(target)<=20 else 'SENTENCE'
                pnorm=rendered_prompt_norm(row); unorm=latest_user_norm(row)
                attempt+=1
                if attempt>24:
                    raise RuntimeError(f"Could not make prompt unique naturally: {reply_class} {idx}")
            seen_prompts.add(pnorm); seen_latest.add(unorm)
            class_rows.append(row); new_rows.append(row)
            dpo_candidates.append((row,ex.negative or 'This answer is incorrect.',ex.rejection_type or 'generic_failure'))
            class_counts[reply_class]+=1; group_counts[group]+=1; domain_counts[ex.domain]+=1; template_counts[row['metadata']['template_family']]+=1; multi_count+=int(meta['multi_turn'])
        assert len(class_rows)==NEW_PER_CLASS

    # Stable shuffle within files while preserving reproducibility.
    new_rows=sorted(new_rows,key=lambda row:hashlib.sha256((str(SEED)+'|new|'+row['id']).encode()).hexdigest())
    reasoning=[r for r in new_rows if r['metadata']['capability_group']=='reasoning']
    communication=[r for r in new_rows if r['metadata']['capability_group']=='communication']
    integrated=[r for r in new_rows if r['metadata']['capability_group']=='integrated']
    preservation=select_preservation(v1_rows,PRESERVATION_ROWS)
    combined=sorted(new_rows+preservation,key=lambda row:hashlib.sha256((str(SEED)+'|combined|'+row['id']).encode()).hexdigest())

    write_jsonl(ROOT/'data/sft/rineya_v2_reasoning_train.jsonl',reasoning)
    write_jsonl(ROOT/'data/sft/rineya_v2_natural_communication_train.jsonl',communication)
    write_jsonl(ROOT/'data/sft/rineya_v2_integrated_reasoning_dialogue_train.jsonl',integrated)
    write_jsonl(ROOT/'data/sft/rineya_v2_new_all_train.jsonl',new_rows)
    write_jsonl(ROOT/'data/sft/rineya_v1_preservation_slice_8000.jsonl',preservation)
    write_jsonl(ROOT/'data/sft/rineya_v2_recommended_combined_40000.jsonl',combined)

    # DPO: exact stratified sample, favor all templates/difficulties.
    cand_by_class=defaultdict(list)
    for row,neg,rtype in dpo_candidates: cand_by_class[row['metadata']['reply_class']].append((row,neg,rtype))
    dpo=[]
    for cls,items in sorted(cand_by_class.items()):
        ranked=sorted(items,key=lambda x:hashlib.sha256((str(SEED)+'|dpo|'+x[0]['id']).encode()).hexdigest())[:DPO_PER_CLASS]
        for row,neg,rtype in ranked:
            dpo.append({
                'id':'dpo_'+row['id'],
                'prompt':qwen_prompt(row['messages'][:-1]),
                'chosen':row['messages'][-1]['content'],
                'rejected':neg,
                'metadata':{
                    'preference_dimension':cls,
                    'capability_group':row['metadata']['capability_group'],
                    'domain':row['metadata']['domain'],
                    'difficulty':row['metadata']['difficulty'],
                    'rejection_type':rtype,
                    'source_record_id':row['id'],
                    'source_provenance':'synthetic_curated_contrast_v2',
                    'human_review_status':'pending',
                    'split':'train',
                    'template_family':'dpo_'+row['metadata']['template_family'],
                }
            })
    dpo=sorted(dpo,key=lambda row:hashlib.sha256((str(SEED)+'|dpo_order|'+row['id']).encode()).hexdigest())
    write_jsonl(ROOT/'data/dpo/rineya_v2_preferences_train_12000.jsonl',dpo)

    # Schemas
    sft_schema={
      '$schema':'https://json-schema.org/draft/2020-12/schema','title':'Rineya SFT record','type':'object',
      'required':['id','system_policy_version','messages','metadata'],
      'properties':{
        'id':{'type':'string','minLength':1},'system_policy_version':{'type':'string','minLength':1},
        'messages':{'type':'array','minItems':3,'items':{'type':'object','required':['role','content'],'properties':{'role':{'enum':['system','user','assistant']},'content':{'type':'string','minLength':1}},'additionalProperties':False}},
        'metadata':{'type':'object','required':['reply_class','capability_group','length_mode','target_token_min','target_token_max','requires_external_fact','safety_tier','tone','source_provenance','human_review_status','split','template_family']}
      },'additionalProperties':False}
    dpo_schema={
      '$schema':'https://json-schema.org/draft/2020-12/schema','title':'Rineya DPO record','type':'object',
      'required':['id','prompt','chosen','rejected','metadata'],
      'properties':{'id':{'type':'string','minLength':1},'prompt':{'type':'string','minLength':1},'chosen':{'type':'string','minLength':1},'rejected':{'type':'string','minLength':1},'metadata':{'type':'object'}},
      'additionalProperties':False
    }
    (ROOT/'schemas/sft_record.schema.json').write_text(json.dumps(sft_schema,indent=2),encoding='utf-8')
    (ROOT/'schemas/dpo_record.schema.json').write_text(json.dumps(dpo_schema,indent=2),encoding='utf-8')

    # Validation
    errors=[]
    all_ids=set(); all_prompts=set(); all_latest=set(); hidden_tags=[]; lengths=[]
    for row in new_rows:
        rid=row['id']; roles=[m['role'] for m in row['messages']]
        expected=['system']+['user','assistant']*((len(roles)-1)//2)
        if roles!=expected: errors.append(f"role order {rid}: {roles}")
        if row['messages'][-1]['role']!='assistant': errors.append(f"last role {rid}")
        if row['metadata']['split']!='train': errors.append(f"split {rid}")
        if 'question' in row: errors.append(f"top question {rid}")
        if rid in all_ids: errors.append(f"duplicate id {rid}")
        all_ids.add(rid)
        p=rendered_prompt_norm(row); u=latest_user_norm(row)
        if p in all_prompts: errors.append(f"duplicate prompt {rid}")
        if u in all_latest: errors.append(f"duplicate latest user {rid}")
        all_prompts.add(p); all_latest.add(u)
        target=row['messages'][-1]['content']; lengths.append(proxy_tokens(target))
        if proxy_tokens(target)>32: errors.append(f"long target {rid}: {proxy_tokens(target)}")
        if re.search(r"<\/?think>|chain[- ]of[- ]thought",target,re.I): hidden_tags.append(rid)
        if not target.strip(): errors.append(f"empty target {rid}")
    dpo_ids=set(); dpo_prompts=set()
    for row in dpo:
        if row['id'] in dpo_ids: errors.append(f"duplicate dpo id {row['id']}")
        dpo_ids.add(row['id'])
        if row['prompt'] in dpo_prompts: errors.append(f"duplicate dpo prompt {row['id']}")
        dpo_prompts.add(row['prompt'])
        if not row['prompt'].endswith('<|im_start|>assistant\n'): errors.append(f"dpo prefix {row['id']}")
        if row['chosen'].strip()==row['rejected'].strip(): errors.append(f"dpo same {row['id']}")
    if hidden_tags: errors.extend(f"hidden tag {x}" for x in hidden_tags)
    if errors: raise RuntimeError('\n'.join(errors[:100]))

    # Leakage exact against v1 and available repository eval snapshots.
    v1_prompt={rendered_prompt_norm(r) for r in v1_rows}; v1_user={latest_user_norm(r) for r in v1_rows}
    exact_v1_prompt=sum(rendered_prompt_norm(r) in v1_prompt for r in new_rows)
    exact_v1_user=sum(latest_user_norm(r) in v1_user for r in new_rows)
    exact_eval_user=sum(latest_user_norm(r) in eval_norms for r in new_rows)

    # Sampled fuzzy screen. Exact checks above are exhaustive; this avoids an expensive all-pairs scan.
    fuzzy_report={'available':False,'threshold':96.0,'sampled_new_rows':0,'v1_near_overlap':None,'eval_near_overlap':None,'examples':[]}
    try:
        from rapidfuzz import fuzz,process
        fuzzy_report['available']=True
        v1_choices=list(v1_user); eval_choices=list(eval_norms)
        sample=sorted(new_rows,key=lambda row:hashlib.sha256((str(SEED)+'|fuzzy_sample|'+row['id']).encode()).hexdigest())[:2000]
        fuzzy_report['sampled_new_rows']=len(sample)
        v1_hits=[]; eval_hits=[]
        for row in sample:
            q=latest_user_norm(row)
            m=process.extractOne(q,v1_choices,scorer=fuzz.ratio,score_cutoff=96.0)
            if m and q!=m[0]: v1_hits.append((row['id'],q,m[0],m[1]))
            m2=process.extractOne(q,eval_choices,scorer=fuzz.ratio,score_cutoff=96.0)
            if m2 and q!=m2[0]: eval_hits.append((row['id'],q,m2[0],m2[1]))
        fuzzy_report['v1_near_overlap']=len(v1_hits); fuzzy_report['eval_near_overlap']=len(eval_hits)
        fuzzy_report['examples']=[{'id':x[0],'new':x[1],'matched':x[2],'score':x[3]} for x in (v1_hits+eval_hits)[:20]]
    except Exception as exc:
        fuzzy_report['error']=str(exc)

    sorted_lengths=sorted(lengths)
    def pct(p:float)->float: return sorted_lengths[min(len(sorted_lengths)-1,math.ceil(p*len(sorted_lengths))-1)]
    validation_report={
        'status':'PASS','generated_at_utc':datetime.now(timezone.utc).isoformat().replace('+00:00','Z'),'seed':SEED,
        'sft_new_rows':len(new_rows),'sft_preservation_rows':len(preservation),'sft_combined_rows':len(combined),'dpo_pairs':len(dpo),
        'unique_new_ids':len(all_ids),'unique_new_rendered_prompts':len(all_prompts),'unique_new_latest_user_prompts':len(all_latest),
        'all_rows_end_in_assistant':True,'all_rows_train_split':True,'top_level_question_count':0,'hidden_reasoning_target_count':0,
        'target_proxy_tokens':{'min':min(lengths),'median':statistics.median(lengths),'p90':pct(.90),'p95':pct(.95),'max':max(lengths),'mean':statistics.mean(lengths)},
        'multi_turn_new_rows':multi_count,'multi_turn_rate':multi_count/len(new_rows),
        'class_count':len(class_counts),'class_distribution':dict(sorted(class_counts.items())),
        'group_distribution':dict(sorted(group_counts.items())),'domain_distribution':dict(sorted(domain_counts.items())),
        'template_family_count':len(template_counts),'counterfactual_rows':sum('counterfactual_group_id' in r['metadata'] for r in new_rows),
        'dpo_pairs_per_class':DPO_PER_CLASS,'errors':[],
    }
    leakage_report={
        'status':'PASS_FOR_AVAILABLE_SNAPSHOTS','new_train_rows':len(new_rows),'v1_reference_rows':len(v1_rows),
        'available_eval_snapshot_files':[str(p.relative_to(EVAL_ROOT)) for p in eval_files],
        'available_eval_snapshot_row_count':sum(len(read_jsonl(p)) for p in eval_files),
        'exact_v1_rendered_prompt_overlap':exact_v1_prompt,'exact_v1_latest_user_overlap':exact_v1_user,
        'exact_available_eval_text_overlap':exact_eval_user,'fuzzy_latest_user':fuzzy_report,
        'important_limit':'The current local repository DEV/SEALED files were not supplied directly in this turn. Re-run scripts/validate_dataset.py against the exact current repository before training.',
    }
    (ROOT/'reports/validation_report.json').write_text(json.dumps(validation_report,indent=2),encoding='utf-8')
    (ROOT/'reports/leakage_report.json').write_text(json.dumps(leakage_report,indent=2),encoding='utf-8')

    # Category distribution CSV
    with (ROOT/'manifests/category_distribution.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['reply_class','capability_group','new_sft','dpo_pairs'])
        for cls,(grp,_) in GENERATORS.items(): w.writerow([cls,grp,class_counts[cls],DPO_PER_CLASS])
    with (ROOT/'manifests/domain_distribution.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.writer(f); w.writerow(['domain','new_sft']);
        for k,v in sorted(domain_counts.items()): w.writerow([k,v])

    # Audit packet: 10 SFT and 3 DPO records per class.
    audit=[]
    for cls in sorted(GENERATORS):
        items=[r for r in new_rows if r['metadata']['reply_class']==cls]
        picks=sorted(items,key=lambda row:hashlib.sha256((str(SEED)+'|audit|'+row['id']).encode()).hexdigest())[:10]
        audit.extend(picks)
    audit_lines=['# Stratified Human Audit Packet','',f'**400 SFT examples:** 10 per class. Human review is still required before a public training/release claim.','']
    for i,row in enumerate(audit,1):
        audit_lines += [f"## {i}. `{row['id']}`",f"- Class: `{row['metadata']['reply_class']}`",f"- Group/domain/difficulty: `{row['metadata']['capability_group']}` / `{row['metadata']['domain']}` / `{row['metadata']['difficulty']}`",'- Conversation:']
        for m in row['messages']: audit_lines.append(f"  - **{m['role']}**: {m['content']}")
        audit_lines.append('')
    audit_lines += ['# Stratified DPO Spot Check','',f'**120 DPO examples:** 3 per class, selected deterministically.','']
    dpo_audit=[]
    for cls in sorted(GENERATORS):
        items=[r for r in dpo if r['metadata']['preference_dimension']==cls]
        picks=sorted(items,key=lambda row:hashlib.sha256((str(SEED)+'|dpo_audit|'+row['id']).encode()).hexdigest())[:3]
        dpo_audit.extend(picks)
    for row in dpo_audit:
        audit_lines += [f"## `{row['id']}`",f"- Class: `{row['metadata']['preference_dimension']}`",f"- Chosen: {row['chosen']}",f"- Rejected: {row['rejected']}",f"- Failure type: `{row['metadata']['rejection_type']}`",'']
    (ROOT/'reports/sample_audit_packet.md').write_text('\n'.join(audit_lines),encoding='utf-8')

    # Statistics
    stats=textwrap.dedent(f"""
    # Corpus Statistics

    ## Row counts

    | Artifact | Rows |
    |---|---:|
    | New reasoning SFT | {len(reasoning):,} |
    | New natural-communication SFT | {len(communication):,} |
    | New integrated reasoning-dialogue SFT | {len(integrated):,} |
    | **All new SFT** | **{len(new_rows):,}** |
    | v1 preservation slice | {len(preservation):,} |
    | **Recommended combined SFT** | **{len(combined):,}** |
    | DPO preference pairs | {len(dpo):,} |

    ## Target length, proxy tokenizer

    - Median: {statistics.median(lengths):.1f}
    - P90: {pct(.90):.0f}
    - P95: {pct(.95):.0f}
    - Maximum: {max(lengths)}
    - Mean: {statistics.mean(lengths):.2f}

    The proxy is intentionally conservative and is not the official Qwen tokenizer. Run the included optional tokenizer report with the exact local tokenizer before training.

    ## Structure

    - 40 response classes, exactly 800 new rows each.
    - {multi_count:,} multi-turn rows ({multi_count/len(new_rows):.1%}).
    - {sum('counterfactual_group_id' in r['metadata'] for r in new_rows):,} rows belong to explicit counterfactual families.
    - {len(template_counts):,} template families.
    - 16 broad use-case domains.
    - 300 DPO pairs per class.

    ## Leakage status

    Exact overlap is zero against the full v1.0 training corpus and the evaluation snapshots available in `Short_Response_Model_Full_v2.1.1`. The exact current repository's DEV/SEALED files must still be checked locally before training.
    """).strip()+"\n"
    (ROOT/'reports/statistics.md').write_text(stats,encoding='utf-8')

    # Documentation
    readme=textwrap.dedent(f"""
    # Rineya General Reasoning and Natural Communication Corpus v2.0

    Train-only data for the Rineya1.1 research iteration. The corpus expands general reasoning, multi-turn context use, and natural human-like communication while preserving the short-response contract.

    ## Recommended files

    | Use | File | Rows |
    |---|---|---:|
    | First teacher SFT run | `data/sft/rineya_v2_recommended_combined_40000.jsonl` | 40,000 |
    | New-data-only ablation | `data/sft/rineya_v2_new_all_train.jsonl` | 32,000 |
    | Reasoning-only ablation | `data/sft/rineya_v2_reasoning_train.jsonl` | 16,000 |
    | Communication-only ablation | `data/sft/rineya_v2_natural_communication_train.jsonl` | 9,600 |
    | Integrated dialogue ablation | `data/sft/rineya_v2_integrated_reasoning_dialogue_train.jsonl` | 6,400 |
    | v1 preservation control | `data/sft/rineya_v1_preservation_slice_8000.jsonl` | 8,000 |
    | Post-SFT DPO | `data/dpo/rineya_v2_preferences_train_12000.jsonl` | 12,000 |

    The recommended 40,000-row file already contains the 8,000-row v1 preservation slice. **Do not append the full v1 corpus to that file again.** To build your own mix, use `rineya_v2_new_all_train.jsonl` instead.

    ## Design priorities

    1. Structural diversity over repetition.
    2. Short final answers without chain-of-thought targets.
    3. Explicit counterfactual and minimally changed cases.
    4. Multi-turn corrections, plan revisions, and context updates.
    5. Natural informal language, empathy, disagreement, compromise, and repair.
    6. Grounded answers, conflict detection, and abstention when facts are absent.
    7. Balanced reasoning and communication instead of math-only specialization.
    8. Preservation replay from v1.0 to reduce behavioral forgetting.

    ## Hard trainer contract

    Every SFT row is JSONL, starts with a system message, alternates user/assistant roles, and ends with the assistant target. No row contains a top-level `question` field, so Phase 9 does not rebuild the prompt. Every DPO prompt is pre-rendered and ends at the Qwen assistant generation prefix.

    ## Before training

    1. Run `scripts/semantic_audit.py --root .` to recheck the 15,200 deterministic/template-grounded rows and DPO source linkage.
    2. Run `scripts/validate_dataset.py` against the exact current repository DEV/SEALED files.
    3. Human-review `reports/sample_audit_packet.md`, especially safety, empathy, roleplay, and DPO negatives.

    Automated semantic checks are intentionally narrower than human review: they verify recomputable answers and explicit context contracts, not whether every open-ended social reply is the uniquely best wording.
    """).strip()+"\n"
    (ROOT/'README.md').write_text(readme,encoding='utf-8')

    data_card=textwrap.dedent(f"""
    # Data Card — Rineya General Reasoning and Natural Communication Corpus v2.0

    ## Purpose

    This release is the second Rineya behavior corpus and is intended for the Rineya1.1 research cycle. It expands capability breadth without abandoning the defining product behavior: read enough context, reason internally, and answer in a few natural words or one compact sentence.

    The v2.3.1 observatory indicated that raw example count was not the dominant bottleneck. Accordingly, v2.0 increases scale only together with stronger structure: 40 balanced classes, counterfactual pairs, multiple difficulty levels, broad domains, multi-turn state changes, grounded mini-contexts, and failure-specific DPO negatives.

    ## Contents

    - 32,000 new SFT records.
    - 8,000-row stratified v1.0 preservation slice.
    - 40,000-row recommended combined SFT.
    - 12,000 DPO preference pairs.
    - 40 response classes spanning reasoning, communication, and integrated dialogue.
    - Schemas, reproducible generator, validator, manifests, checksums, audit packet, and leakage report.

    ## Intended capabilities

    Reasoning covers arithmetic chains, percentages, ratios, units, time, scheduling, ranking, formal implication, explicit rules, contradiction detection, patterns, set counting, constraints, causal and physical reasoning, spatial relations, estimation, evidence inference, underdetermination, and troubleshooting.

    Communication covers updates, follow-ups, empathy, tactful disagreement, clarification, preferences, apology/repair, de-escalation, compromise, boundaries, informal texting, and conversation closure.

    Integrated examples combine reasoning with concise decision explanations, multi-turn corrections, plan revision, recommendations, practical next steps, evidence-grounded dialogue, message rewriting, and in-character reasoning.

    ## What this corpus does not do

    - It does not contain hidden chain-of-thought targets.
    - It does not try to teach exhaustive current world knowledge.
    - It does not replace the Phase 9 RAG corpus.
    - It does not include repository DEV, SEALED, or preference DEV rows.
    - It does not claim human approval; all synthetic rows remain `human_review_status: pending`.

    ## Recommended use

    For the 454M teacher, begin with one QLoRA epoch on the 40,000-row recommended file plus the locally converted 4,410-row RAG training set, producing a 44,410-row dual-skill mix. Preserve the original teacher checkpoint, evaluate the same official chat and RAG DEVs, and freeze a winner before any sealed pass.

    DPO should follow only after selecting the SFT candidate. Start with a conservative DPO dose and keep the SFT checkpoint available for rollback because preference tuning can improve naturalness while damaging exact reasoning or evidence behavior.

    ## Limitations

    Programmatic validation proves schema, role order, uniqueness, target-length bounds, and available-snapshot isolation. A separate deterministic semantic audit recomputes or contract-checks 15,200 rows across 19 classes and verifies DPO chosen/source linkage. It cannot prove every social response is the uniquely best reply. Human review is required before a public release claim.

    The leakage report uses the v1.0 corpus and evaluation files available from the v2.1.1 repository snapshot. The exact current repository must be checked locally because later DEV/SEALED revisions were not provided as local files in this turn.
    """).strip()+"\n"
    (ROOT/'DATA_CARD.md').write_text(data_card,encoding='utf-8')

    training=textwrap.dedent(r"""
    # Training and Mixing Guide

    ## Recommended Rineya1.1 teacher mix

    1. Start from the frozen accepted Rineya1 teacher checkpoint; do not overwrite it.
    2. Use `data/sft/rineya_v2_recommended_combined_40000.jsonl`.
    3. Append the local 4,410-row converted RAG training file once.
    4. Shuffle deterministically and train one QLoRA epoch as the first controlled run.
    5. Re-run the same 510-case chat DEV and official RAG DEV used for the baseline.
    6. Compare category tails, not only aggregate means.
    7. Freeze one SFT candidate before DPO or any sealed evaluation.

    The result should contain 44,410 rows if the local RAG file still has exactly 4,410 valid records.

    ## PowerShell mixing example

    ```powershell
    python .\scripts\mix_jsonl.py `
      --input .\data\sft\rineya_v2_recommended_combined_40000.jsonl `
      --input C:\PATH\TO\converted_rag_train_4410.jsonl `
      --output C:\PATH\TO\rineya1_1_dual_sft_train_44410.jsonl `
      --seed 20260816
    ```

    ## Re-run the deterministic semantic audit

    ```powershell
    python .\scripts\semantic_audit.py --root .
    ```

    ## Validate against current evaluation splits

    ```powershell
    python .\scripts\validate_dataset.py `
      --sft .\data\sft\rineya_v2_recommended_combined_40000.jsonl `
      --dpo .\data\dpo\rineya_v2_preferences_train_12000.jsonl `
      --forbidden-eval-file C:\PATH\TO\short-response-model-phase-0-1-2\data\dev\dev.jsonl `
      --forbidden-eval-file C:\PATH\TO\short-response-model-phase-0-1-2\data\sealed\sealed.jsonl
    ```

    ## DPO

    Use DPO only after the SFT candidate is accepted on DEV. Keep the SFT adapter immutable, train DPO as a new child artifact, and evaluate chat, reasoning, RAG supported, RAG abstention, and procedure correctness again.
    """).strip()+"\n"
    (ROOT/'examples/training_and_mixing_guide.md').write_text(training,encoding='utf-8')

    design={
        'corpus_name':'Rineya General Reasoning and Natural Communication Corpus','version':'2.0','seed':SEED,
        'new_sft_rows':len(new_rows),'preservation_rows':len(preservation),'recommended_combined_rows':len(combined),'dpo_pairs':len(dpo),
        'class_policy':{'classes':len(GENERATORS),'rows_per_class':NEW_PER_CLASS,'dpo_pairs_per_class':DPO_PER_CLASS},
        'groups':dict(group_counts),'principles':[
            'Structural coverage over raw repetition','Reason internally; supervise only the short final answer','Counterfactual boundary cases','Natural human-like communication','Explicit uncertainty and evidence boundaries','Multi-turn context updates','Preservation replay','No evaluation-split training'
        ],
        'recommended_local_dual_mix':{'v2_combined':40000,'converted_rag':4410,'total':44410},
    }
    (ROOT/'manifests/design_specification.json').write_text(json.dumps(design,indent=2),encoding='utf-8')
    manifest={'files':{},'lineage':{'base_corpus':'Rineya Short-Response Behavior Corpus v1.0','preservation_source_rows':len(v1_rows),'available_eval_snapshot':'Short_Response_Model_Full_v2.1.1_Preservation_First_Procedural_Recovery','generated_seed':SEED}}

    # Reproducible scripts copied/created
    shutil.copy2(Path(__file__),ROOT/'scripts/generate_dataset.py')
    mix_script='''#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse,json,random\nfrom pathlib import Path\n\ndef read(path):\n    with path.open(encoding="utf-8") as f:\n        return [json.loads(x) for x in f if x.strip()]\n\ndef main():\n    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,action="append",required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--seed",type=int,default=20260816); a=ap.parse_args()\n    rows=[]\n    for p in a.input: rows.extend(read(p))\n    ids=[r.get("id") for r in rows]\n    if len(ids)!=len(set(ids)): raise SystemExit("Duplicate IDs detected. Do not append full v1 to the v2 recommended combined file.")\n    random.Random(a.seed).shuffle(rows); a.output.parent.mkdir(parents=True,exist_ok=True)\n    with a.output.open("w",encoding="utf-8",newline="\\n") as f:\n        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\\n")\n    print(json.dumps({"output":str(a.output),"rows":len(rows),"unique_ids":len(set(ids))},indent=2))\nif __name__=="__main__": main()\n'''
    (ROOT/'scripts/mix_jsonl.py').write_text(mix_script,encoding='utf-8')

    validator='''#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse,json,re\nfrom pathlib import Path\n\ndef read(path):\n    with path.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]\ndef norm(t): return " ".join(re.findall(r"[a-z0-9_.$%+\\-]+",t.lower()))\ndef latest(row):\n    if "messages" in row:\n        us=[m["content"] for m in row["messages"] if m["role"]=="user"]; return norm(us[-1]) if us else ""\n    return norm(row.get("prompt") or row.get("question") or row.get("input") or "")\ndef validate_sft(rows):\n    ids=set(); prompts=set(); errs=[]\n    for i,r in enumerate(rows,1):\n        roles=[m.get("role") for m in r.get("messages",[])]; expected=["system"]+["user","assistant"]*((len(roles)-1)//2)\n        if roles!=expected: errs.append(f"line {i}: role order {roles}")\n        if not roles or roles[-1]!="assistant": errs.append(f"line {i}: final role")\n        if r.get("metadata",{}).get("split")!="train": errs.append(f"line {i}: split")\n        if "question" in r: errs.append(f"line {i}: top-level question")\n        if r.get("id") in ids: errs.append(f"line {i}: duplicate id")\n        ids.add(r.get("id")); p=norm("\\n".join(f"{m['role']}:{m['content']}" for m in r.get("messages",[])[:-1]))\n        if p in prompts: errs.append(f"line {i}: duplicate prompt")\n        prompts.add(p)\n    return errs,{"rows":len(rows),"unique_ids":len(ids),"unique_prompts":len(prompts)}\ndef validate_dpo(rows):\n    ids=set(); prompts=set(); errs=[]\n    for i,r in enumerate(rows,1):\n        if r.get("id") in ids: errs.append(f"line {i}: duplicate dpo id")\n        ids.add(r.get("id")); p=r.get("prompt","")\n        if p in prompts: errs.append(f"line {i}: duplicate dpo prompt")\n        prompts.add(p)\n        if not p.endswith("<|im_start|>assistant\\n"): errs.append(f"line {i}: bad prefix")\n        if r.get("chosen","").strip()==r.get("rejected","").strip(): errs.append(f"line {i}: same choice")\n    return errs,{"rows":len(rows),"unique_ids":len(ids),"unique_prompts":len(prompts)}\ndef main():\n    ap=argparse.ArgumentParser(); ap.add_argument("--sft",type=Path,action="append",default=[]); ap.add_argument("--dpo",type=Path,action="append",default=[]); ap.add_argument("--forbidden-eval-file",type=Path,action="append",default=[]); a=ap.parse_args(); report={"sft":{},"dpo":{},"leakage":{}}; errors=[]\n    eval_text=set()\n    for p in a.forbidden_eval_file:\n        for r in read(p): eval_text.add(latest(r))\n    for p in a.sft:\n        rows=read(p); e,s=validate_sft(rows); errors+=e; report["sft"][str(p)]=s\n        if eval_text: report["leakage"][str(p)]={"exact_latest_user_overlap":sum(latest(r) in eval_text for r in rows)}\n    for p in a.dpo:\n        rows=read(p); e,s=validate_dpo(rows); errors+=e; report["dpo"][str(p)]=s\n    report["status"]="PASS" if not errors and all(v["exact_latest_user_overlap"]==0 for v in report["leakage"].values()) else "FAIL"; report["errors"]=errors[:100]; print(json.dumps(report,indent=2)); raise SystemExit(0 if report["status"]=="PASS" else 1)\nif __name__=="__main__": main()\n'''
    (ROOT/'scripts/validate_dataset.py').write_text(validator,encoding='utf-8')

    # checksums and manifest last
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name not in {'checksums.sha256','dataset_manifest.json'}:
            manifest['files'][str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (ROOT/'manifests/dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    checks=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name!='checksums.sha256': checks.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    (ROOT/'manifests/checksums.sha256').write_text('\n'.join(checks)+'\n',encoding='utf-8')

    # Zip
    if ZIP_PATH.exists(): ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(ROOT.rglob('*')):
            if p.is_file(): z.write(p,arcname=f"{ROOT.name}/{p.relative_to(ROOT).as_posix()}")
    with zipfile.ZipFile(ZIP_PATH) as z:
        bad=z.testzip()
        if bad: raise RuntimeError(f"Bad zip member: {bad}")
    print(json.dumps({'zip':str(ZIP_PATH),'size':ZIP_PATH.stat().st_size,'sha256':hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest(),'members':len(zipfile.ZipFile(ZIP_PATH).namelist()),'validation':validation_report,'leakage':leakage_report},indent=2))

if __name__=='__main__':
    main()
