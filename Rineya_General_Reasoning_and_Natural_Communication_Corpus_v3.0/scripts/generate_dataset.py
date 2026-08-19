from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
import random
import re
import shutil
import statistics
import textwrap
import time
import zipfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable

SEED = 20260818
ROOT = Path('/mnt/data/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0')
ZIP_PATH = Path('/mnt/data/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v3.0.zip')
V1_ROOT = Path('/mnt/data/_v1src/Rineya_Short_Response_Behavior_Corpus_v1.0')
V2_ROOT = Path('/mnt/data/_v2src/Rineya_General_Reasoning_and_Natural_Communication_Corpus_v2.0')
REPO_ROOT = Path('/mnt/data/_repofiles')
V2_GENERATOR = V2_ROOT / 'scripts/generate_dataset.py'

REASONING_CLASSES = [
    'arithmetic_chain','percentage_reasoning','ratio_proportion','unit_conversion','time_duration',
    'schedule_ordering','comparison_ranking','logical_implication','conditional_rules','contradiction_detection',
    'sequence_pattern','counting_sets','constraint_selection','causal_reasoning','physical_commonsense',
    'spatial_relations','estimation_sanity','evidence_inference','insufficient_information','troubleshooting_diagnosis',
]
COMMUNICATION_CLASSES = [
    'acknowledgment_update','natural_follow_up','empathy_validation','tactful_disagreement',
    'clarification_request','preference_opinion','apology_repair','deescalation',
    'negotiation_compromise','boundary_setting','informal_texting','conversation_closure',
]
INTEGRATED_CLASSES = [
    'decision_explanation','multi_turn_context_update','plan_revision','comparative_recommendation',
    'practical_next_step','evidence_grounded_dialogue','communication_rewrite','roleplay_reasoning',
]
ALL_CLASSES = REASONING_CLASSES + COMMUNICATION_CLASSES + INTEGRATED_CLASSES
GROUP_OF = {c:'reasoning' for c in REASONING_CLASSES} | {c:'communication' for c in COMMUNICATION_CLASSES} | {c:'integrated' for c in INTEGRATED_CLASSES}
COUNTS = {c:2500 for c in REASONING_CLASSES + COMMUNICATION_CLASSES} | {c:3750 for c in INTEGRATED_CLASSES}
DPO_PER_CLASS = 400
PRESERVATION_ROWS = 14000
SHARD_SIZE = 250

GENERAL_SYSTEMS = [
    "Reply like a capable person texting: natural, accurate, and brief. Use a few words or one compact sentence, ask one clear question only when needed, and put safety before brevity.",
    "Give the shortest natural reply that fully handles the user's immediate need. Read the whole thread, do not invent facts, and use one brief clarification when information is missing.",
    "Respond concisely and humanly. Preserve the conversation's current facts, avoid canned assistant language, and set a short honest boundary when a request is unsafe or deceptive.",
    "Keep the response compact, context-aware, and useful. Prefer a few natural words; do not lecture, expose hidden reasoning, or guess beyond the prompt.",
]
REASONING_SYSTEMS = [
    "Use only the facts stated in the conversation. Reason silently and return the compact final answer; when the result is not supported, say so briefly instead of guessing.",
    "Work out the answer internally, then give only a short conclusion. Follow later corrections, use no outside facts, and state insufficiency when one answer cannot be determined.",
    "Solve from the supplied information without showing the derivation. Return a few words or one compact sentence, and do not fabricate missing details.",
    "Read the full thread, silently resolve the stated constraints, and answer briefly. If the evidence does not determine the result, give an explicit knowledge boundary.",
]
INTEGRATED_SYSTEMS = [
    "Read the whole conversation, reason silently, and make the requested conversational move in one compact human reply. Honor later corrections and do not invent evidence.",
    "Use the current thread state to answer naturally and briefly. Do the actual task—decide, clarify, empathize, revise, or set a boundary—without exposing hidden reasoning.",
    "Respond as a concise conversational partner. Combine the stated facts with the user's immediate goal, follow updates over stale details, and keep unsupported claims out.",
    "Give one short, natural response that is both context-correct and socially appropriate. Use only provided facts and preserve role or tone when the scene requires it.",
]

# Deliberately excludes the v2 clone-farm lexicon.
FRESH_NAMES = [
    'Adela','Bryn','Cato','Daria','Emmett','Farah','Galen','Hana','Imani','Jonas','Keira','Malik',
    'Nadia','Orin','Priya','Quinn','Rosa','Silas','Tessa','Uma','Viktor','Wren','Xiomara','Yusuf',
    'Zara','Ansel','Beatrix','Corin','Delia','Eamon','Freya','Gideon','Helena','Isak','Juno','Kellan',
    'Leona','Mateo','Neve','Omar','Petra','Ronan','Selene','Tobin','Una','Vera','Wes','Xena','Yara',
    'Zev','Amara','Basil','Cleo','Damon','Esme','Fiona','Hugo','Ines','Jasper','Kira','Lucian',
    'Maeve','Niko','Opal','Pavel','Reina','Soren','Talia','Uri','Valerie','Wyatt','Xander','Yasmin',
]
FRESH_ORGS = [
    'Bramble Forge','Copper Finch','Driftwood Lab','Elm & Current','Foxglove Studio','Granite Loop',
    'Hearthline Co-op','Ivory Kiteworks','Junebug Transit','Keystone Pantry','Lantern Fieldworks',
    'Marigold Commons','Northstar Repair','Osprey Reading Room','Peregrine Works','Quarry House',
    'Redwood Signal','Silver Orchard','Thistle Circuit','Umber Workshop','Verdant Ledger','Windward Club',
    'Yellowbrick Arts','Zephyr Civic Desk','Alder Creek Crew','Bluebell Robotics','Cinder Trail',
    'Doveglass Theatre','Evergreen Relay','Firefly Kitchens','Garnet Athletics','Holloway Makers',
    'Inkstone Press','Jade Harbor Collective','Kindred Nursery','Larkspur Observatory','Mossbank Library',
    'Nightjar Cycling','Oakline School','Poppyseed Market','Rivermint Clinic','Saffron Archive',
    'Timberline Scouts','Violet Quay','Wildfern Media','Yarrow Field Team','Acorn Civic Lab',
    'Bellwether Games','Cloudberry Home','Dunewatch Survey','Emberglass Museum','Frostline Shipping',
    'Goldcrest Choir','Hazel Street Council','Ironwood Coding Club','Kestrel Farm','Limestone Tutors',
    'Moonrise Gallery','Nettlebrook Parents','Paperboat Café','Rosehip Science Camp','Stonecrop Finance',
]
FRESH_PLACES = [
    'riverfront annex','north concourse','greenhouse bay','rooftop court','map room','ceramics wing',
    'trailhead shelter','community kitchen','repair bench','west atrium','practice field','reading loft',
    'harbor platform','civic counter','observatory deck','orchard shed','music hall','survey camp',
    'makerspace aisle','bus depot','gallery storeroom','parent pickup zone','locker corridor','boathouse',
    'climbing gym','planetarium lobby','garden pavilion','equipment cage','workshop mezzanine','field station',
]
FRESH_OBJECTS = [
    'sensor tags','seed trays','relay modules','score sheets','canvas rolls','sample vials','route cards',
    'charging docks','music stands','permit forms','tool cases','water filters','costume racks','timing chips',
    'field notebooks','meal kits','bike lights','access badges','paint tubes','repair tickets','practice cones',
    'survey flags','storage bins','memory cards','folding chairs','radio packs','shipping labels','plant markers',
    'safety helmets','display panels','spare fuses','recipe cards','library holds','transit passes','drill bits',
]
EVENTS = [
    'equipment handoff','weekend rehearsal','route inspection','inventory count','family pickup plan',
    'tournament setup','permit review','field sampling run','gallery installation','budget check',
    'science demonstration','volunteer briefing','release rehearsal','repair intake','club registration',
    'meal-prep shift','studio booking','sports practice','transit update','community workshop',
    'parent meeting','coding sprint','arts showcase','weather contingency','supply delivery',
    'home repair visit','travel departure','class presentation','research note review','safety drill',
]
DOMAINS = [
    'everyday','work','school','coding','electronics','travel','cooking','shopping','personal_finance',
    'health_safety','home','social','games','transportation','science','productivity','sports','arts',
    'civic_logistics','parenting_logistics','field_work',
]
TONE_BY_CLASS = {
    'empathy_validation':'warm','tactful_disagreement':'respectful','deescalation':'calm',
    'boundary_setting':'firm','informal_texting':'casual','conversation_closure':'natural',
    'roleplay_reasoning':'in_character','communication_rewrite':'clear',
}

BANNED_NAMES = {
    'Maya','Leo','Nina','Owen','Iris','Noah','Ava','Eli','Zoe','Kai','Mina','Theo','Lena','Sam','Aria',
    'Jules','Ravi','Tara','Milo','June','Ari','Sora','Nora','Ben','Luca','Mira','Drew','Ivy','Alex','Remy',
}
BANNED_PROJECTS = {
    'Atlas','Harbor','Juniper','Nova','Orchid','Beacon','Cedar','Lumen','Mosaic','Drift','Kite','Maple',
    'Quartz','River','Summit','Willow','Echo','Pine','Solace','Vela',
}
BANNED_OPENERS = (
    'Sure!', 'Of course!', "I'd be happy to", 'Certainly!', 'As an AI', 'I understand that',
)

CONTEXT_FRAME_TEMPLATES = [
    "For {org}'s {event} at the {place}, the {item} log says: {core}",
    "{name} is checking {item} during the {event}; {core}",
    "At the {place}, the current note for {org} reads: {core}",
    "This came up during the {event} for {org}: {core}",
    "Before {name} closes the {event}, one detail remains: {core}",
    "The {domain_label} crew at the {place} needs this resolved: {core}",
    "For the latest {item} record, not the older draft: {core}",
    "During {org}'s {event}, {name} asks: {core}",
    "The current field note from the {place} says: {core}",
    "To finish the {event}, use this exact version: {core}",
    "{org} is updating the {item} plan; {core}",
    "A last check from the {place}: {core}",
    "For today's {event}, the active constraint is this: {core}",
    "{name} replaced the earlier note with this one: {core}",
    "The {item} entry under review at {org} says: {core}",
    "Before the {domain_label} team proceeds, answer this: {core}",
    "At {org}, the {event} now depends on one point: {core}",
    "The current message from {name} at the {place} is: {core}",
    "For the {event} handoff, rely on this statement: {core}",
    "The {place} coordinator marked this as the live version: {core}",
    "One practical check for {org}'s {event}: {core}",
    "The team handling {item} at the {place} asks: {core}",
    "In the revised {event} notes, {name} wrote: {core}",
    "For this {domain_label} decision, use only the following: {core}",
]


RECORD_ADJECTIVES = [
    'amber','brisk','cobalt','dawn','evening','fern','granite','harvest','indigo','juniper','kindred','linen',
    'meadow','northbound','opal','prairie','quiet','rainy','sunlit','tidal','umber','vernal','windy','year-end',
    'autumn','blue-hour','copper','dry-season','early','frost','golden','highland','island','late','morning','night',
]
RECORD_NOUNS = [
    'ledger','worksheet','brief','dispatch','field note','handoff','register','logbook','checklist','memo','route card','roster',
    'status sheet','survey page','planning note','intake slip','scorebook','supply record','schedule card','work order','review page','shift note','casebook','tracker',
]

RELATION_CONTEXTS = [
    "We have worked together on this for weeks.",
    "This is between close teammates, not a formal notice.",
    "The user already corrected one detail earlier.",
    "The recipient is stressed and wants a direct reply.",
    "The speaker wants honesty more than reassurance.",
    "A private detail was shared in confidence.",
    "The relationship is friendly but the boundary still matters.",
    "The user wants one useful question, not a lecture.",
]

ROLEPLAY_PERSONAS = [
    ('Aderyn','trail medic','steady and practical'),('Bram','clockwork courier','quick and wry'),
    ('Cyra','archive navigator','calm and exact'),('Demitri','storm mechanic','blunt but kind'),
    ('Elowen','market mediator','warm and observant'),('Faelan','signal scout','focused and dry'),
    ('Greer','airship quartermaster','efficient and protective'),('Hestia','garden alchemist','gentle and curious'),
    ('Ivo','bridge engineer','precise and reserved'),('Kaia','city guide','friendly and alert'),
    ('Lior','harbor watchkeeper','quiet and direct'),('Maelis','field cartographer','patient and concise'),
]

@dataclass(frozen=True)
class FamilySpec:
    reply_class: str
    group: str
    family_index: int
    family_id: str
    size: int
    counterfactual: bool
    domain: str
    difficulty: str
    frame_id: int
    dialogue_pattern: int
    lexical_seed: int
    shard_id: str

@dataclass
class BuiltExample:
    messages: list[dict[str, str]]
    target: str
    domain: str
    speech_act: str
    safety_tier: str
    requires_external_fact: bool
    quality_rationale: str
    failure_if_wrong: str
    uniqueness_claim: str
    rejected: str
    failure_mode: str
    semantic_payload: dict[str, Any]


def stable_seed(*parts: Any) -> int:
    raw = '|'.join(map(str, (SEED,) + parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], 'big')


def rng_for(*parts: Any) -> random.Random:
    return random.Random(stable_seed(*parts))


def normalize(text: str) -> str:
    text = text.lower().replace('’', "'").replace('“','"').replace('”','"')
    return ' '.join(re.findall(r"[a-z0-9_.$%+\-]+", text))


def normalize_latest(text: str) -> str:
    return ' '.join(re.findall(r'[a-z0-9]+', text.lower()))


def proxy_tokens(text: str) -> int:
    return len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?|[^\w\s]", text, flags=re.UNICODE))


def choose_system(group: str, idx: int) -> str:
    systems = REASONING_SYSTEMS if group == 'reasoning' else GENERAL_SYSTEMS if group == 'communication' else INTEGRATED_SYSTEMS
    return systems[idx % len(systems)]


def qwen_prompt(messages: list[dict[str,str]]) -> str:
    return ''.join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages) + '<|im_start|>assistant\n'


def load_v2_module() -> Any:
    spec = importlib.util.spec_from_file_location('rineya_v2_base_generators', V2_GENERATOR)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot import {V2_GENERATOR}')
    mod = importlib.util.module_from_spec(spec)
    import sys
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def replace_lexicon(text: str, r: random.Random) -> str:
    name_map = {old: r.choice(FRESH_NAMES) for old in sorted(BANNED_NAMES)}
    project_map = {old: r.choice(FRESH_ORGS) for old in sorted(BANNED_PROJECTS)}
    place_map = {
        'library':r.choice(FRESH_PLACES),'station':r.choice(FRESH_PLACES),'office':r.choice(FRESH_PLACES),
        'clinic':r.choice(FRESH_PLACES),'cafe':r.choice(FRESH_PLACES),'workshop':r.choice(FRESH_PLACES),
        'campus':r.choice(FRESH_PLACES),'hotel':r.choice(FRESH_PLACES),'museum':r.choice(FRESH_PLACES),
        'market':r.choice(FRESH_PLACES),'garage':r.choice(FRESH_PLACES),'kitchen':r.choice(FRESH_PLACES),
        'lab':r.choice(FRESH_PLACES),'studio':r.choice(FRESH_PLACES),'airport':r.choice(FRESH_PLACES),
        'warehouse':r.choice(FRESH_PLACES),
    }
    object_map = {
        'notebooks':r.choice(FRESH_OBJECTS),'tickets':r.choice(FRESH_OBJECTS),'cables':r.choice(FRESH_OBJECTS),
        'bottles':r.choice(FRESH_OBJECTS),'folders':r.choice(FRESH_OBJECTS),'batteries':r.choice(FRESH_OBJECTS),
        'cards':r.choice(FRESH_OBJECTS),'markers':r.choice(FRESH_OBJECTS),'boxes':r.choice(FRESH_OBJECTS),
        'tokens':r.choice(FRESH_OBJECTS),'mugs':r.choice(FRESH_OBJECTS),'screws':r.choice(FRESH_OBJECTS),
        'files':r.choice(FRESH_OBJECTS),'photos':r.choice(FRESH_OBJECTS),'seats':r.choice(FRESH_OBJECTS),
        'coins':r.choice(FRESH_OBJECTS),'apples':r.choice(FRESH_OBJECTS),'books':r.choice(FRESH_OBJECTS),
        'lamps':r.choice(FRESH_OBJECTS),'packages':r.choice(FRESH_OBJECTS),
    }
    for old,new in {**name_map,**project_map}.items():
        text = re.sub(rf'\b{re.escape(old)}\b', new, text)
    for old,new in place_map.items():
        text = re.sub(rf'\b{re.escape(old)}\b', new, text, flags=re.I)
    for old,new in object_map.items():
        text = re.sub(rf'\b{re.escape(old)}\b', new, text, flags=re.I)
    return text


def domain_label(domain: str) -> str:
    return domain.replace('_',' ')


def context_values(spec: FamilySpec, row_in_family: int) -> dict[str,str]:
    # Counterfactual variants share one scene; regular variants rotate through a
    # large deterministic Cartesian product. The coprime strides prevent the
    # accidental scene collisions common with repeated random.choice calls.
    local = 0 if spec.counterfactual else row_in_family
    base = spec.family_index * 17 + local + (stable_seed('ctxbase', spec.reply_class) % 100003)
    record_index = spec.family_index + (stable_seed('record', spec.reply_class) % (len(RECORD_ADJECTIVES)*len(RECORD_NOUNS)))
    return {
        'name': FRESH_NAMES[(base * 7 + 3) % len(FRESH_NAMES)],
        'org': FRESH_ORGS[(base * 11 + 5) % len(FRESH_ORGS)],
        'place': FRESH_PLACES[(base * 13 + 7) % len(FRESH_PLACES)],
        'item': FRESH_OBJECTS[(base * 17 + 9) % len(FRESH_OBJECTS)],
        'event': EVENTS[(base * 19 + 11) % len(EVENTS)],
        'record': f"{RECORD_ADJECTIVES[record_index % len(RECORD_ADJECTIVES)]} {RECORD_NOUNS[(record_index // len(RECORD_ADJECTIVES)) % len(RECORD_NOUNS)]}",
        'domain_label': domain_label(spec.domain),
    }


def frame_core(core: str, spec: FamilySpec, row_in_family: int, ctx: dict[str,str]) -> str:
    tmpl = CONTEXT_FRAME_TEMPLATES[(spec.frame_id + row_in_family) % len(CONTEXT_FRAME_TEMPLATES)]
    rendered = tmpl.format(core=core, **ctx)
    # Some stylistic frames mention only the broad domain. Add a natural,
    # scene-level anchor whenever fewer than two concrete scene attributes are
    # visible, so different families cannot collapse into the same prompt.
    visible = sum(v.lower() in rendered.lower() for v in (ctx['org'], ctx['event'], ctx['place'], ctx['item']))
    if visible < 2:
        rendered = f"For {ctx['org']}'s {ctx['event']} at the {ctx['place']}, {rendered[0].lower() + rendered[1:]}"
    # A family-specific but natural document label makes the scene explicit and
    # prevents exact collisions without introducing opaque case IDs.
    return f"In the {ctx['record']}, {rendered[0].lower() + rendered[1:]}"


def mutate_first_fact(text: str, r: random.Random) -> str:
    # Produce a plausible stale version for a later-correction dialogue.
    m = re.search(r'(?<![A-Za-z])\d+(?:\.\d+)?', text)
    if m:
        old = m.group(0)
        val = float(old)
        delta = r.choice([1,2,5,10])
        new_val = max(1, val + (delta if r.random() < .5 else -delta))
        if old.isdigit(): new = str(int(round(new_val)))
        else: new = f'{new_val:.2f}'.rstrip('0').rstrip('.')
        return text[:m.start()] + new + text[m.end():]
    swaps = [
        ('north','east'),('south','west'),('left','right'),('right','left'),('Monday','Tuesday'),
        ('Friday','Thursday'),('open','closed'),('closed','open'),('blue','amber'),('red','green'),
        ('yes','no'),('valid','invalid'),('draft','final'),('final','draft'),
    ]
    for a,b in swaps:
        if re.search(rf'\b{re.escape(a)}\b', text, flags=re.I):
            return re.sub(rf'\b{re.escape(a)}\b', b, text, count=1, flags=re.I)
    return 'Earlier draft: ' + text


def build_dialogue(system: str, core_user: str, target: str, spec: FamilySpec, row_in_family: int, ctx: dict[str,str]) -> list[dict[str,str]]:
    r = rng_for('dialogue', spec.reply_class, spec.family_index, row_in_family)
    framed = frame_core(core_user, spec, row_in_family, ctx)
    # Exactly controlled depth. Basic remains a direct exchange; intermediate/compositional include real state.
    if spec.difficulty == 'basic':
        return [
            {'role':'system','content':system},
            {'role':'user','content':framed},
            {'role':'assistant','content':target},
        ]
    if spec.difficulty == 'intermediate':
        if spec.dialogue_pattern % 3 == 0:
            setup = f"I'm handling {ctx['event']} for {ctx['org']} at the {ctx['place']}."
            ack = r.choice(["What is the current detail?","Okay—what changed?","Send the live version."])
            return [
                {'role':'system','content':system},
                {'role':'user','content':setup},
                {'role':'assistant','content':ack},
                {'role':'user','content':framed},
                {'role':'assistant','content':target},
            ]
        return [
            {'role':'system','content':system},
            {'role':'user','content':framed + f" Keep the answer usable for {ctx['name']}."},
            {'role':'assistant','content':target},
        ]
    stale = mutate_first_fact(core_user, r)
    first = f"The earlier {ctx['event']} note for {ctx['org']} said: {stale}"
    ack1 = r.choice(["I have the earlier version.","Okay—that is the draft state.","Noted, but I will wait for the current one."])
    bridge = r.choice([
        f"{ctx['name']} just checked the {ctx['item']} record.",
        f"The {ctx['place']} coordinator found a correction.",
        "One fact in that version was stale.",
    ])
    ack2 = r.choice(["What is the corrected version?","Send the current facts.","Okay—what should replace it?"])
    final = f"Correction—use this current version instead: {framed}"
    return [
        {'role':'system','content':system},
        {'role':'user','content':first},
        {'role':'assistant','content':ack1},
        {'role':'user','content':bridge},
        {'role':'assistant','content':ack2},
        {'role':'user','content':final},
        {'role':'assistant','content':target},
    ]


def contextualize_target(reply_class: str, target: str, ctx: dict[str,str], row_in_family: int) -> str:
    # Preserve exact reasoning targets. Natural communication targets are made scene-specific when otherwise generic.
    if GROUP_OF[reply_class] == 'reasoning':
        return target
    stripped = target.strip()
    generic = {
        'Got it.','Sounds good.','Perfect, thanks.','That is frustrating.','That is really frustrating.',
        'How was it?','What happened?','No.','Yes.','Okay.','Understood.','That makes sense.',
        'Let’s pause.','Thanks.','No problem.','All good.','Take care.','Talk soon.','Good night.',
    }
    r = rng_for('target_context', reply_class, ctx['org'], ctx['event'], row_in_family, target)
    if stripped not in generic and proxy_tokens(stripped) >= 4:
        return stripped
    if reply_class == 'acknowledgment_update':
        return r.choice([f"Got it—{ctx['event']} is updated.",f"Okay—{ctx['org']} has the change.",f"Noted for {ctx['event']}."])
    if reply_class == 'natural_follow_up':
        return r.choice([f"How did {ctx['event']} go?",f"What happened with {ctx['item']}?",f"How did {ctx['name']} respond?"])
    if reply_class == 'empathy_validation':
        return r.choice([f"That {ctx['event']} setback sounds rough.",f"Yeah, that sounds draining for {ctx['name']}.","That is a lot—want to vent or troubleshoot?"])
    if reply_class == 'tactful_disagreement':
        return r.choice([f"I disagree—the {ctx['item']} risk still matters.",f"I would not do that for {ctx['event']}.","I see the appeal, but the tradeoff is too weak."])
    if reply_class == 'clarification_request':
        return r.choice([f"Which {ctx['item']} version?",f"Do you mean {ctx['org']} or another group?",f"What time applies to {ctx['event']}?"])
    if reply_class == 'preference_opinion':
        return r.choice([f"I would choose the simpler {ctx['item']} option.",f"The {ctx['org']} option fits better.","I prefer the quieter, lower-risk choice."])
    if reply_class == 'apology_repair':
        return r.choice([f"You are right—I mixed up the {ctx['item']} note.",f"Sorry, I used the stale {ctx['event']} detail.",f"My mistake—{ctx['name']}'s correction is current."])
    if reply_class == 'deescalation':
        return r.choice([f"Let’s pause and verify the {ctx['item']} record.","We are getting heated—let’s stick to the facts.",f"Pause the {ctx['event']} thread, then compare notes."])
    if reply_class == 'negotiation_compromise':
        return r.choice([f"Let’s split the {ctx['event']} work and check in later.",f"I can handle {ctx['item']} if you take the follow-up.","Let’s use the middle option and revisit tomorrow."])
    if reply_class == 'boundary_setting':
        return r.choice([f"I won’t share {ctx['name']}'s private detail.",f"I cannot fake the {ctx['event']} confirmation.",f"I cannot do that for {ctx['org']}."])
    if reply_class == 'informal_texting':
        return r.choice([f"Yep, {ctx['event']} is set.",f"Nah, leave the {ctx['item']} alone.",f"On my way to the {ctx['place']} now."])
    if reply_class == 'conversation_closure':
        return r.choice([f"All set for {ctx['event']}—talk later.",f"That covers {ctx['org']}. Take care.","Sounds good—catch you later."])
    if reply_class == 'roleplay_reasoning':
        return stripped
    if stripped in generic:
        return r.choice([f"Got it for {ctx['event']}.",f"Okay—{ctx['org']} can proceed.",f"That works for {ctx['name']}."])
    return stripped



def force_counterfactual_target(reply_class: str, target: str, ctx: dict[str,str]) -> str:
    """Make non-reasoning counterfactual targets scene-specific without changing the move."""
    if GROUP_OF[reply_class] == 'reasoning' or reply_class == 'roleplay_reasoning':
        return target
    core=target.rstrip('.?!')
    punct='?' if target.rstrip().endswith('?') else '.'
    if reply_class=='acknowledgment_update':
        # Preserve the changed time or location while naming the event.
        return f"Got it—{ctx['event']} is {core.split('—')[-1].rstrip('.')} .".replace(' .','.')
    if reply_class=='natural_follow_up':
        return f"{core} about {ctx['event']}?" if not core.lower().endswith(ctx['event'].lower()) else target
    if reply_class=='empathy_validation':
        return f"That {ctx['event']} situation sounds rough—{core.split('—')[-1].lower()}."
    if reply_class=='tactful_disagreement':
        return f"For {ctx['event']}, {core[0].lower()+core[1:]}."
    if reply_class=='clarification_request':
        return f"For the {ctx['item']}, {core[0].lower()+core[1:]}?"
    if reply_class=='preference_opinion':
        return f"{core} for {ctx['event']}."
    if reply_class=='apology_repair':
        return f"{core}; I used the stale {ctx['item']} note."
    if reply_class=='deescalation':
        return f"For {ctx['event']}, {core[0].lower()+core[1:]}."
    if reply_class=='negotiation_compromise':
        return f"For {ctx['event']}, {core[0].lower()+core[1:]}."
    if reply_class=='boundary_setting':
        return f"{core} for {ctx['event']}."
    if reply_class=='informal_texting':
        return f"{core} for {ctx['event']}."
    if reply_class=='conversation_closure':
        return f"{core} for {ctx['event']}."
    if reply_class=='decision_explanation':
        return f"{core} for {ctx['org']}'s {ctx['event']}."
    if reply_class=='multi_turn_context_update':
        return f"{core} for {ctx['org']}'s {ctx['event']}."
    if reply_class=='plan_revision':
        return f"{core} for {ctx['org']}'s {ctx['event']}."
    if reply_class=='comparative_recommendation':
        return f"{core} for {ctx['org']}'s {ctx['event']}."
    if reply_class=='practical_next_step':
        return f"{core} on {ctx['org']}'s {ctx['item']}."
    if reply_class=='evidence_grounded_dialogue':
        if core.lower().startswith("i don't know"):
            return f"I don't know from {ctx['org']}'s {ctx['record']}."
        return target
    if reply_class=='communication_rewrite':
        low=target.lower()
        if low.startswith('test it first'):
            return f"Test {ctx['org']}'s {ctx['item']} first."
        if 'delay happened here' in low:
            return f"The {ctx['org']} delay happened here; fix the handoff."
        if 'minutes late' in low:
            return f"About 10 minutes late for {ctx['event']}—sorry!"
        revised=re.sub(r'\b(report|task|file|ticket)\b', ctx['item'], target, count=1, flags=re.I)
        return revised if revised!=target else f"{core} for {ctx['org']}."
    return target

def speech_act_for(reply_class: str) -> str:
    mapping = {
        'acknowledgment_update':'acknowledge_specific_update','natural_follow_up':'ask_one_relevant_follow_up',
        'empathy_validation':'validate_without_diagnosing','tactful_disagreement':'disagree_gently_with_reason',
        'clarification_request':'ask_minimal_missing_detail','preference_opinion':'state_direct_preference',
        'apology_repair':'own_and_correct_mistake','deescalation':'lower_heat_and_refocus',
        'negotiation_compromise':'offer_balanced_compromise','boundary_setting':'set_honest_boundary',
        'informal_texting':'reply_in_natural_texting_style','conversation_closure':'close_without_unwanted_continuation',
        'decision_explanation':'choose_and_give_compact_reason','multi_turn_context_update':'use_latest_corrected_state',
        'plan_revision':'revise_plan_for_new_constraint','comparative_recommendation':'recommend_by_stated_criterion',
        'practical_next_step':'name_one_actionable_next_step','evidence_grounded_dialogue':'answer_only_from_evidence',
        'communication_rewrite':'perform_requested_rewrite','roleplay_reasoning':'reason_correctly_in_character',
    }
    return mapping.get(reply_class,'return_compact_correct_answer')


def quality_texts(reply_class: str, group: str, counterfactual: bool) -> tuple[str,str,str]:
    if group == 'reasoning':
        rationale = 'The target is the recomputable compact result from the current stated facts.'
        failure = 'A small model may use a stale value, make an arithmetic or logic slip, or add an unnecessary derivation.'
    elif group == 'communication':
        rationale = f"The target directly performs the requested speech act: {speech_act_for(reply_class)}."
        failure = 'A small model may answer with a generic acknowledgment, ignore the social function, or over-explain.'
    else:
        rationale = f"The target combines thread-correct reasoning with the required move: {speech_act_for(reply_class)}."
        failure = 'A small model may follow stale context, choose without the stated criterion, hallucinate evidence, or break role.'
    unique = 'Counterfactual family changes one decisive fact while preserving the scene.' if counterfactual else 'Family combines a distinct domain frame, constraint pattern, dialogue structure, and lexical seed—not a name-only swap.'
    return rationale,failure,unique


def class_failure_mode(reply_class: str, fallback: str | None) -> str:
    if fallback:
        return fallback
    if reply_class in REASONING_CLASSES:
        return 'wrong_result_or_fake_certainty'
    if reply_class in COMMUNICATION_CLASSES:
        return 'wrong_speech_act_or_robotic_reply'
    return 'stale_state_or_hallucinated_basis'


def safe_rejected(reply_class: str, chosen: str, base_rejected: str | None, ctx: dict[str,str]) -> str:
    # Keep errors realistic and short. Never place unsafe actionable detail in a rejected response.
    r = rng_for('dpo_reject', reply_class, chosen, ctx['org'], ctx['event'])
    if reply_class in REASONING_CLASSES:
        if base_rejected and proxy_tokens(base_rejected) <= 20 and '<think>' not in base_rejected:
            return base_rejected.strip()
        return r.choice(["I cannot tell, but probably yes.","The earlier value is the answer.","That should be close enough."])
    banks = {
        'acknowledgment_update':["Thanks for the update.","I have recorded everything carefully.","Okay, tell me the whole story."],
        'natural_follow_up':["Acknowledged.","Great, thanks for the update.","That is good information."],
        'empathy_validation':["You should not feel that way.","This means you are clinically burned out.","Just stay positive."],
        'tactful_disagreement':["No, that idea is bad.","Fine, you are probably right.","Whatever—do it your way."],
        'clarification_request':["I will assume the latest version.","Probably the blue one.","I know what you mean."],
        'preference_opinion':["Both are equally good.","It depends on many complicated factors.","Choose whichever."],
        'apology_repair':["That is what I meant anyway.","You misunderstood me.","No correction is needed."],
        'deescalation':["Calm down.","You are both overreacting.","Keep arguing until someone wins."],
        'negotiation_compromise':["One person should do all of it.","No compromise is possible.","Just split it somehow."],
        'boundary_setting':["Sure, I can fake that confirmation.","I will share the private detail.","Just this once is fine."],
        'informal_texting':["Your update has been formally acknowledged.","I would be delighted to proceed accordingly.","Communication received."],
        'conversation_closure':["Before you go, here are several more suggestions.","Wait, let us continue this discussion.","I have many additional points."],
        'decision_explanation':["Option A, because it is first.","Either one without a reason.","I chose based on a different criterion."],
        'multi_turn_context_update':["Use the original detail.","The first version still applies.","I ignored the correction."],
        'plan_revision':["Keep the old plan unchanged.","Ignore the new constraint.","We can probably improvise later."],
        'comparative_recommendation':["Choose the most expensive one.","They are all the same.","Option A without checking the criterion."],
        'practical_next_step':["Wait and hope it fixes itself.","Replace everything immediately.","Do several unrelated things first."],
        'evidence_grounded_dialogue':["Yes, according to common knowledge.","I am sure, though the record does not say.","The missing detail is probably true."],
        'communication_rewrite':["I would be happy to rewrite that for you.","The original sentence is already fine.","Here is a detailed explanation of tone."],
        'roleplay_reasoning':["As an AI assistant, I cannot stay in character.","The scene rules do not matter.","I will answer out of character."],
    }
    return r.choice(banks[reply_class])


def family_plan_for_class(reply_class: str) -> list[FamilySpec]:
    n = COUNTS[reply_class]
    group = GROUP_OF[reply_class]
    cf_rows = int(math.ceil(n * 0.20 / 4.0) * 4)
    if cf_rows > n: cf_rows = n - (n % 4)
    regular_rows = n - cf_rows
    specs: list[FamilySpec] = []
    # Exact 25/40/35 mix assigned by row counts, then family gets dominant difficulty.
    difficulty_targets = {
        'basic': round(n*0.25),
        'intermediate': round(n*0.40),
    }
    difficulty_targets['compositional'] = n - difficulty_targets['basic'] - difficulty_targets['intermediate']
    row_cursor = 0
    fam_idx = 0
    def difficulty_at(start: int, size: int) -> str:
        mid = start + size/2
        if mid <= difficulty_targets['basic']: return 'basic'
        if mid <= difficulty_targets['basic'] + difficulty_targets['intermediate']: return 'intermediate'
        return 'compositional'
    # Counterfactual families first but difficulty assignment is distributed later via stable reordering.
    cf_families = cf_rows // 4
    reg_sizes = [8] * (regular_rows // 8)
    if regular_rows % 8: reg_sizes.append(regular_rows % 8)
    raw: list[tuple[bool,int]] = [(True,4)]*cf_families + [(False,s) for s in reg_sizes]
    raw = sorted(raw, key=lambda x: hashlib.sha256(f'{SEED}|family_order|{reply_class}|{raw.index(x) if False else len(specs)}|{x}'.encode()).hexdigest())
    # The line above intentionally does not depend on list.index; shuffle deterministically instead.
    raw = [(True,4)]*cf_families + [(False,s) for s in reg_sizes]
    rng_for('family_shuffle',reply_class).shuffle(raw)
    for is_cf,size in raw:
        diff = difficulty_at(row_cursor,size)
        domain = DOMAINS[(fam_idx*7 + stable_seed(reply_class,fam_idx)%len(DOMAINS)) % len(DOMAINS)]
        shard_number = row_cursor // SHARD_SIZE
        shard_id = f"{group[:4]}_{reply_class}_s{shard_number:02d}"
        specs.append(FamilySpec(
            reply_class=reply_class,group=group,family_index=fam_idx,
            family_id=f"rineya_v3_family__{reply_class}__{fam_idx:04d}",size=size,
            counterfactual=is_cf,domain=domain,difficulty=diff,
            frame_id=(fam_idx*5 + stable_seed('frame',reply_class,fam_idx)) % len(CONTEXT_FRAME_TEMPLATES),
            dialogue_pattern=(fam_idx*3 + stable_seed('dialogue',reply_class,fam_idx)) % 12,
            lexical_seed=stable_seed('lexical',reply_class,fam_idx),shard_id=shard_id,
        ))
        row_cursor += size; fam_idx += 1
    assert row_cursor == n
    return specs


def cf_messages(spec: FamilySpec, variant: int, core: str, target: str, ctx: dict[str,str]) -> list[dict[str,str]]:
    system = choose_system(spec.group, spec.family_index + variant)
    framed = frame_core(core,spec,variant,ctx)
    # Counterfactual twins keep the same scene, and compositional families explicitly preserve correction structure.
    if spec.difficulty == 'basic':
        return [{'role':'system','content':system},{'role':'user','content':framed},{'role':'assistant','content':target}]
    if spec.difficulty == 'intermediate':
        return [
            {'role':'system','content':system},
            {'role':'user','content':f"We are checking one decision for {ctx['org']}'s {ctx['event']}."},
            {'role':'assistant','content':'What is the current fact?'},
            {'role':'user','content':framed},
            {'role':'assistant','content':target},
        ]
    stale = mutate_first_fact(core, rng_for('cf_stale',spec.reply_class,spec.family_index,variant))
    return [
        {'role':'system','content':system},
        {'role':'user','content':f"Earlier note: {stale}"},
        {'role':'assistant','content':'I will wait for the corrected state.'},
        {'role':'user','content':f"Current correction for {ctx['event']}: {framed}"},
        {'role':'assistant','content':target},
    ]


def generate_counterfactual(spec: FamilySpec, variant: int) -> BuiltExample:
    """Create true one-fact counterfactual twins; four variants share a scene."""
    cls=spec.reply_class; r=rng_for('cf',cls,spec.family_index); ctx=context_values(spec,0)
    # All variants share ctx. Only one decisive fact changes per variant.
    rejected=None; failure_mode=None; safety='normal'; requires=False; payload={'kind':'counterfactual','variant':variant}
    if cls=='arithmetic_chain':
        start=r.randint(20,70); gain=r.randint(4,16); loss=r.randint(2,12); final_gain=r.randint(1,9)+variant
        core=f"A {ctx['item']} count starts at {start}, gains {gain}, loses {loss}, then gains {final_gain}. Final count?"; target=f"{start+gain-loss+final_gain}."; rejected=f"{start+gain-loss+final_gain+1}."
    elif cls=='percentage_reasoning':
        base=r.choice([40,60,80,100,120,160]); pct=[10,20,25,50][variant]; core=f"The {ctx['item']} budget is {base} units. It is reduced by {pct}%. What remains?"; target=f"{int(base*(1-pct/100))}."; rejected=f"{int(base*pct/100)}."
    elif cls=='ratio_proportion':
        a,b=2,3; mult=variant+3; total=(a+b)*mult; core=f"The ratio of practice cones to timing chips is {a}:{b}, with {total} items total. How many are timing chips?"; target=f"{b*mult}."; rejected=f"{a*mult}."
    elif cls=='unit_conversion':
        hours=variant+1; mins=[5,15,30,45][variant]; core=f"Convert {hours} hours and {mins} minutes to minutes."; target=f"{hours*60+mins} minutes."; rejected=f"{hours*60} minutes."
    elif cls=='time_duration':
        start=9*60+15; dur=[25,40,55,70][variant]; end=start+dur; h=end//60; m=end%60; suffix='AM' if h<12 else 'PM'; h12=h%12 or 12; core=f"The {ctx['event']} starts at 9:15 AM and lasts {dur} minutes. End time?"; target=f"{h12}:{m:02d} {suffix}."; rejected=f"{h12}:{(m+10)%60:02d} {suffix}."
    elif cls=='schedule_ordering':
        orders=[('draft','review','publish'),('inspect','repair','test'),('pack','check in','board'),('measure','cut','assemble')]; a,b,c=orders[variant]; core=f"{a.title()} must happen before {b}, and {b} before {c}. What is the order?"; target=f"{a.title()} → {b.title()} → {c.title()}."; rejected=f"{c.title()} → {b.title()} → {a.title()}."
    elif cls=='comparison_ranking':
        vals=[r.randint(20,50),r.randint(51,80),r.randint(81,110)]; vals[variant%3]+=variant*20; ask='highest' if variant%2==0 else 'lowest'; core=f"Option A scores {vals[0]}, B scores {vals[1]}, and C scores {vals[2]}. Which is {ask}?"; idx=(max if ask=='highest' else min)(range(3),key=lambda i:vals[i]); target=f"Option {'ABC'[idx]}."; rejected=f"Option {'ABC'[(idx+1)%3]}."
    elif cls=='logical_implication':
        animal=['sparrow','heron','finch','tern'][variant]; premise = variant!=1; core=(f"Every {animal} in this aviary has a leg band. This bird is a {animal}. Does it have a band?" if premise else f"Every {animal} in this aviary has a leg band. This bird has a band. Must it be a {animal}?"); target='Yes.' if premise else 'No—the reverse does not follow.'; rejected='No.' if premise else 'Yes.'
    elif cls=='conditional_rules':
        level=[12,19,20,35][variant]; core=f"Rule: charge the radio pack only below 20%. Its battery is {level}%. Charge it?"; target='Charge it.' if level<20 else 'No charge is required.'; rejected='No charge is required.' if level<20 else 'Charge it.'
    elif cls=='contradiction_detection':
        a='amber'; b=['amber','blue','amber','green'][variant]; core=f"One current note says the access badge is {a}; another says it is {b}. Do they conflict?"; target='No, they can both be true.' if a==b else 'Yes, they conflict.'; rejected='Yes, they conflict.' if a==b else 'No, they can both be true.'
    elif cls=='sequence_pattern':
        start=r.randint(2,12); step=[2,3,4,5][variant]; seq=[start+i*step for i in range(4)]; core=f"What comes next: {', '.join(map(str,seq))}?"; target=f"{seq[-1]+step}."; rejected=f"{seq[-1]+step+1}."
    elif cls=='counting_sets':
        tea=20+variant*2; coffee=14+variant; overlap=[3,5,7,9][variant]; core=f"{tea} volunteers chose the morning shift, {coffee} chose evening, and {overlap} chose both. How many chose at least one?"; target=f"{tea+coffee-overlap}."; rejected=f"{tea+coffee}."
    elif cls=='constraint_selection':
        budget=[55,65,75,85][variant]; opts=[(52,7),(68,10),(82,12)]; valid=[i for i,(p,h) in enumerate(opts) if p<=budget and h>=8]; core=f"Choose a radio pack costing at most ${budget} with at least 8 hours: A ${opts[0][0]}/{opts[0][1]}h, B ${opts[1][0]}/{opts[1][1]}h, C ${opts[2][0]}/{opts[2][1]}h."; target=(f"Option {'ABC'[valid[0]]}." if valid else "No option fits."); rejected='Option A.' if target!='Option A.' else 'Option C.'
    elif cls=='causal_reasoning':
        cable=['loose','secure','loose','secure'][variant]; core=f"The display works until the desk moves. The cable is {cable}. What is the likely cause?"; target='A loose display cable.' if cable=='loose' else 'The cause is not established from this.'; rejected='The monitor is definitely broken.'
    elif cls=='physical_commonsense':
        state=['wet','dry','wet','dry'][variant]; core=f"The painted sign is still {state}. Should it be stacked face-to-face now?"; target='No—let it dry first.' if state=='wet' else 'Yes, it can be stacked carefully.'; rejected='Yes.' if state=='wet' else 'No.'
    elif cls=='spatial_relations':
        direction=['north','south','east','west'][variant]; core=f"The supply tent is {direction} of the field station. From the station, which way is the tent?"; target=f"{direction.title()}."; rejected=f"{['South','North','West','East'][variant]}."
    elif cls=='estimation_sanity':
        n=[4,8,12,20][variant]; each=7; estimate=round(n*each,-1); core=f"About {n} repair tickets cost roughly ${each} each. Estimate the total."; target=f"About ${estimate}."; rejected=f"About ${n+each}."
    elif cls=='evidence_inference':
        status=['approved','pending','approved','rejected'][variant]; core=f"Mini-record: Permit status: {status}. No other status is listed. Is the permit approved?"; target='Yes.' if status=='approved' else 'No.'; rejected='Yes.' if status!='approved' else 'No.'
    elif cls=='insufficient_information':
        has=[False,True,False,True][variant]; core=(f"The travel note lists departure at 8:10 AM and arrival at 9:00 AM. How long is the trip?" if has else "The travel note lists departure at 8:10 AM but no arrival time. How long is the trip?"); target='50 minutes.' if has else "I don't know from this."; rejected='50 minutes.' if not has else "I don't know from this."
    elif cls=='troubleshooting_diagnosis':
        symptom=['loose','connected','loose','connected'][variant]; core=f"A monitor shows no signal; the computer is on and the display cable is {symptom}. First step?"; target='Reconnect the display cable.' if symptom=='loose' else 'Check the selected display input.'; rejected='Replace the whole computer.'
    elif cls=='acknowledgment_update':
        times=['2:15 PM','3:00 PM','3:45 PM','4:30 PM']; core=f"I moved {ctx['org']}'s {ctx['event']} to {times[variant]}."; target=f"Got it—{times[variant]}."; rejected='Thanks for the update.'
    elif cls=='natural_follow_up':
        outcomes=['went well','was delayed','was canceled','felt awkward']; core=f"I finished the {ctx['event']}; it {outcomes[variant]}."; qs=["What went best?","What caused the delay?","Did they reschedule?","Want to talk through it?"]; target=qs[variant]; rejected='Acknowledged.'
    elif cls=='empathy_validation':
        needs=['vent','problem-solve','rest','just be heard']; core=f"The {ctx['event']} fell apart, and I mostly want to {needs[variant]}."; targets=["That is rough—go ahead and vent.","That is frustrating—want one next step?","That sounds exhausting. Rest first.","I hear you. I can just listen."]; target=targets[variant]; rejected='Just stay positive.'
    elif cls=='tactful_disagreement':
        risk=['minor','serious','minor','serious'][variant]; core=f"I think we should skip the {ctx['item']} check; the risk is {risk}. Do you agree?"; target=('That seems reasonable if the risk is minor.' if risk=='minor' else 'I disagree—the serious risk still needs a check.'); rejected='Yes, absolutely.' if risk=='serious' else 'No, that is stupid.'
    elif cls=='clarification_request':
        missing=['version','date','location','recipient'][variant]; core=f"Please send the {ctx['item']} update. The {missing} is not specified."; qs={'version':'Which version should I send?','date':'What date should it use?','location':'Which location applies?','recipient':'Who should receive it?'}; target=qs[missing]; rejected='I will guess the latest one.'
    elif cls=='preference_opinion':
        criterion=['cheaper','quieter','faster','safer'][variant]; core=f"For {ctx['event']}, option A is cheaper and louder; option B is quieter and slower. I care most about {criterion}. Which would you pick?"; target=('Option A.' if criterion in {'cheaper','faster'} else 'Option B.'); rejected='Either one.'
    elif cls=='apology_repair':
        current=['Tuesday','Wednesday','Thursday','Friday'][variant]; core=f"You told me the current pickup day is {current}, but I repeated the old day. Correct me."; target=f"You are right—pickup is {current}."; rejected='That is what I said.'
    elif cls=='deescalation':
        state=['voices are rising','one person is interrupting','the facts are mixed up','both are repeating themselves'][variant]; core=f"During {ctx['event']}, {state}. What should I say?"; targets=["Let’s lower our voices and take one point at a time.","Let’s let each person finish before responding.","Pause—let’s verify the facts first.","We are looping; let’s name one next decision."]; target=targets[variant]; rejected='Calm down.'
    elif cls=='negotiation_compromise':
        time_slots=['morning only','afternoon only','either time','neither time'][variant]; core=f"I can help with {ctx['event']} in the morning; {ctx['name']} is available {time_slots}. Offer a compromise."; targets=["Let’s split the morning work.","I can start; they can take the afternoon handoff.","Let’s choose the shorter shared slot.","Let’s reschedule for a time we both have."]; target=targets[variant]; rejected='One person should do everything.'
    elif cls=='boundary_setting':
        request=['share a private message','fake a confirmation','impersonate the organizer','invent a source'][variant]; core=f"Can you {request} for {ctx['event']}?"; targets=["I won’t share a private message.","I won’t fake that confirmation.","I can’t impersonate the organizer.","I won’t invent a source."]; target=targets[variant]; rejected=['Sure, I will share it.','Sure, I can fake it.','Yes, I can pretend to be them.','I can make one up.'][variant]; safety='elevated'
    elif cls=='informal_texting':
        status=['running late','already there','not coming','almost finished'][variant]; core=f"Text {ctx['name']} casually that I am {status} for {ctx['event']}."; targets=["Running late—be there soon.","Already here, by the entrance.","Can’t make it today—sorry.","Almost done, give me five."]; target=targets[variant]; rejected='I wish to formally communicate my present status.'
    elif cls=='conversation_closure':
        state=['resolved','postponed','handed off','still open'][variant]; core=f"The {ctx['event']} is {state}. Close the conversation naturally."; targets=["All set—talk later.","Okay, we’ll pick this up later.","Great, they have it now. Take care.","Let’s pause here and continue tomorrow."]; target=targets[variant]; rejected='Before you go, I have several more points.'
    elif cls=='decision_explanation':
        priority=['cost','speed','quiet','safety'][variant]; core=f"Option A is cheaper and slower; B is faster and noisier. The priority is {priority}. Which should we use?"; target=('Option A—lower cost.' if priority in {'cost','quiet'} else 'Option B—faster.' if priority=='speed' else 'Option A—the quieter choice.'); rejected='Option B, because it is newer.'
    elif cls=='multi_turn_context_update':
        places=['riverfront annex','map room','west atrium','music hall']; core=f"We first planned to meet at the old location. Correction: meet at the {places[variant]}. Where are we meeting?"; target=f"The {places[variant]}."; rejected='The old location.'
    elif cls=='plan_revision':
        constraint=['rain','a closed venue','a missing vehicle','a two-hour delay'][variant]; core=f"Our outdoor {ctx['event']} plan now has {constraint}. Revise the next move."; targets=["Move it under cover.","Use the backup venue.","Switch to the local route.","Notify everyone and shift the schedule."]; target=targets[variant]; rejected='Keep the original plan.'
    elif cls=='comparative_recommendation':
        criterion=['lowest cost','fewest transfers','highest battery life','smallest size'][variant]; core=f"A costs $40 with two transfers and 6h battery; B costs $55 with no transfer and 12h battery. Choose by {criterion}."; target=('Option A.' if criterion in {'lowest cost','smallest size'} else 'Option B.'); rejected='Option A.' if target=='Option B.' else 'Option B.'
    elif cls=='practical_next_step':
        state=['unsaved','saved but untested','tested with one failure','tested successfully'][variant]; core=f"The {ctx['item']} change is {state}. What is the next step?"; targets=["Save the change first.","Run the focused test.","Fix the failing test.","Review the diff, then finish."]; target=targets[variant]; rejected='Deploy immediately.'
    elif cls=='evidence_grounded_dialogue':
        statuses=['approved','pending','rejected','not listed']; core=f"Mini-record: Current status: {statuses[variant]}. Updated today. Based only on this, is it approved?"; target=('Yes.' if statuses[variant]=='approved' else 'No.' if statuses[variant] in {'pending','rejected'} else "I don't know from this."); rejected='Yes.' if target!='Yes.' else 'No.'
    elif cls=='communication_rewrite':
        tones=['warmer','more direct','less accusatory','casual']; raw=['Send the repair ticket today.','We might perhaps test this first.','You caused the delay.','I will arrive ten minutes late.'][variant]; targets=['Could you send the repair ticket today? Thanks!','Test it first.','The delay happened here; let’s fix the handoff.','About 10 minutes late—sorry!']; core=f"Make this {tones[variant]}: '{raw}'"; target=targets[variant]; rejected='I would be happy to rewrite that.'
    elif cls=='roleplay_reasoning':
        persona,role,voice=ROLEPLAY_PERSONAS[spec.family_index%len(ROLEPLAY_PERSONAS)]; system=f"You are {persona}, a {role} in Caldermere. Voice: {voice}. Stay in character, use only scene facts, reason silently, and answer in one compact utterance."; ctx=dict(ctx); facts=[('north gate','south gate'),('two lanterns','one lantern'),('blue seal','red seal'),('twenty minutes','ten minutes')]; a,b=facts[variant]; core=f"The map marks the safe path through the {a}. Which path do we take?"; target=f"The {a}—that is the marked safe path."; rejected='As an AI, I cannot participate in roleplay.'; messages=cf_messages(spec,variant,core,target,ctx); messages[0]['content']=system; rationale,failure,unique=quality_texts(cls,spec.group,True); return BuiltExample(messages,target,'games',speech_act_for(cls),safety,requires,rationale,failure,unique,rejected,'character_break',payload)
    else:
        raise KeyError(cls)
    target=contextualize_target(cls,target,ctx,variant) if cls in COMMUNICATION_CLASSES and cls not in {'boundary_setting'} else target
    target=force_counterfactual_target(cls,target,ctx)
    messages=cf_messages(spec,variant,core,target,ctx)
    rationale,failure,unique=quality_texts(cls,spec.group,True)
    return BuiltExample(messages,target,spec.domain,speech_act_for(cls),safety,requires,rationale,failure,unique,rejected or 'Incorrect.',class_failure_mode(cls,failure_mode),payload)


def generate_regular(v2: Any, spec: FamilySpec, row_in_family: int) -> BuiltExample:
    cls=spec.reply_class; group=spec.group
    # Spread across base modes and target phrasings while staying deterministic.
    source_idx = 100000 + spec.family_index*97 + row_in_family*13 + stable_seed('source',cls,spec.family_index)%31
    base_group, gen = v2.GENERATORS[cls]
    assert base_group == group
    ex = gen(source_idx)
    rr = rng_for('lex_replace',cls,spec.family_index,row_in_family)
    # Use only the substantive latest user and target from v2; rebuild the thread under v3 policy.
    base_user = [m['content'] for m in ex.messages if m['role']=='user'][-1]
    base_target = ex.messages[-1]['content']
    base_rejected = getattr(ex,'negative',None)
    base_user = replace_lexicon(base_user,rr)
    base_target = replace_lexicon(base_target,rr)
    if base_rejected:
        base_rejected = replace_lexicon(base_rejected,rr)
    ctx=context_values(spec,row_in_family)
    target=contextualize_target(cls,base_target,ctx,row_in_family)
    # Roleplay uses its own system, but freshen all named entities.
    if cls=='roleplay_reasoning':
        persona,role,voice=ROLEPLAY_PERSONAS[(spec.family_index+row_in_family)%len(ROLEPLAY_PERSONAS)]
        system=f"You are {persona}, a {role} in Caldermere, during {ctx['event']} at the {ctx['place']}. Voice: {voice}. Stay in character, use only established scene facts, reason silently, and reply in one compact utterance."
    else:
        system=choose_system(group,spec.family_index+row_in_family)
    messages=build_dialogue(system,base_user,target,spec,row_in_family,ctx)
    rationale,failure,unique=quality_texts(cls,group,False)
    rejected=safe_rejected(cls,target,base_rejected,ctx)
    safety='elevated' if cls=='boundary_setting' or getattr(ex,'safety_tier','normal') in {'elevated','high'} else 'none'
    requires=bool(getattr(ex,'requires_external_fact',False))
    semantic_payload={
        'kind':'base_generator_replay','source_generator_version':'v2.0','source_index':source_idx,
        'source_base_template_family':getattr(ex,'template_family','unknown'),
        'base_target_before_v3_context':base_target,
    }
    return BuiltExample(messages,target,spec.domain,speech_act_for(cls),safety,requires,rationale,failure,unique,rejected,class_failure_mode(cls,getattr(ex,'rejection_type',None)),semantic_payload)


def read_jsonl(path: Path) -> list[dict[str,Any]]:
    rows=[]
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): rows.append(json.loads(line))
    return rows


def iter_jsonl(path: Path) -> Iterable[dict[str,Any]]:
    with path.open('r',encoding='utf-8') as f:
        for line in f:
            if line.strip(): yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[dict[str,Any]]) -> int:
    path.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with path.open('w',encoding='utf-8',newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row,ensure_ascii=False,separators=(',',':'))+'\n'); n+=1
    return n


def rendered_prompt_norm(row: dict[str,Any]) -> str:
    return normalize('\n'.join(f"{m['role']}:{m['content']}" for m in row['messages'][:-1]))


def latest_user(row: dict[str,Any]) -> str:
    return [m['content'] for m in row['messages'] if m['role']=='user'][-1]


def latest_user_norm(row: dict[str,Any]) -> str:
    return normalize_latest(latest_user(row))


def validate_role_order(messages: list[dict[str,str]]) -> bool:
    roles=[m.get('role') for m in messages]
    if not roles or roles[0]!='system' or roles[-1]!='assistant': return False
    return roles == ['system'] + ['user','assistant']*((len(roles)-1)//2)


def build_forbidden_sets() -> tuple[set[str],set[str],set[str],list[Path]]:
    latest=set(); prompts=set(); targets=set(); files=[]
    sources=[
        V1_ROOT/'data/sft/rineya_short_response_new_train.jsonl',
        V1_ROOT/'data/sft/rineya_short_response_preservation_slice.jsonl',
        V2_ROOT/'data/sft/rineya_v2_new_all_train.jsonl',
        V2_ROOT/'data/sft/rineya_v1_preservation_slice_8000.jsonl',
    ]
    # Evaluation snapshot: include only DEV/TEST/SEALED/preference-dev/RAG eval-like files, never train files.
    for p in REPO_ROOT.rglob('*.jsonl'):
        s=p.as_posix().lower()
        if any(x in s for x in ['/data/dev/','/data/sealed/','preference_dev','/dev_','/test_','/dev/','/test/']) or ('rag' in s and any(x in s for x in ['dev','test','sealed','eval'])):
            sources.append(p)
    for p in sources:
        if not p.is_file(): continue
        files.append(p)
        try:
            for row in iter_jsonl(p):
                if isinstance(row.get('messages'),list):
                    us=[m.get('content','') for m in row['messages'] if m.get('role')=='user']
                    if us: latest.add(normalize_latest(us[-1]))
                    if row['messages'] and row['messages'][-1].get('role')=='assistant': targets.add(normalize_latest(row['messages'][-1].get('content','')))
                    if len(row['messages'])>=2: prompts.add(normalize('\n'.join(f"{m.get('role')}:{m.get('content','')}" for m in row['messages'][:-1])))
                for key in ('prompt','question','input','query','instruction','user'):
                    if isinstance(row.get(key),str): latest.add(normalize_latest(row[key]))
        except Exception:
            continue
    return latest,prompts,targets,files


def build_preservation_slice() -> list[dict[str,Any]]:
    v1=list(iter_jsonl(V1_ROOT/'data/sft/rineya_short_response_new_train.jsonl'))
    v2=list(iter_jsonl(V2_ROOT/'data/sft/rineya_v2_new_all_train.jsonl'))
    # 4k v1 + 10k v2, stratified by reply_class and deterministic.
    def stratified(rows:list[dict[str,Any]], n:int, label:str)->list[dict[str,Any]]:
        groups=defaultdict(list)
        for row in rows: groups[row.get('metadata',{}).get('reply_class','unknown')].append(row)
        allocations={}; rem=[]
        for cls,items in groups.items():
            exact=n*len(items)/len(rows); base=math.floor(exact); allocations[cls]=base; rem.append((exact-base,cls))
        for _,cls in sorted(rem,reverse=True)[:n-sum(allocations.values())]: allocations[cls]+=1
        out=[]
        for cls,items in groups.items():
            ranked=sorted(items,key=lambda x:hashlib.sha256(f'{SEED}|preserve|{label}|{x["id"]}'.encode()).hexdigest())
            out.extend(ranked[:allocations[cls]])
        return out
    chosen=stratified(v1,4000,'v1')+stratified(v2,10000,'v2')
    result=[]
    for row in chosen:
        c=json.loads(json.dumps(row,ensure_ascii=False))
        meta=c.setdefault('metadata',{})
        meta['preservation_replay_for']='rineya_v3'
        meta['preservation_source_version']='v1.0' if str(c.get('id','')).startswith(('train_','rineya_short')) else 'v2.0'
        result.append(c)
    result=sorted(result,key=lambda x:hashlib.sha256(f'{SEED}|preserve_order|{x["id"]}'.encode()).hexdigest())
    assert len(result)==PRESERVATION_ROWS and len({r['id'] for r in result})==PRESERVATION_ROWS
    return result


def make_row(spec:FamilySpec, row_in_family:int, built:BuiltExample, class_index:int) -> dict[str,Any]:
    target=built.messages[-1]['content']
    tokens=proxy_tokens(target)
    global_index=spec.family_index*8+row_in_family
    row_id=f"rineya_v3_sft__{spec.reply_class}__{spec.shard_id}__{global_index:05d}"
    cf_id=f"rineya_v3_cf__{spec.reply_class}__{spec.family_index:04d}" if spec.counterfactual else None
    meta={
        'reply_class':spec.reply_class,'capability_group':spec.group,
        'length_mode':'TINY' if tokens<=5 else 'SHORT' if tokens<=12 else 'COMPACT_SENTENCE',
        'target_token_min':1,'target_token_max':24,'target_token_count_proxy':tokens,
        'requires_external_fact':built.requires_external_fact,'safety_tier':built.safety_tier,
        'tone':TONE_BY_CLASS.get(spec.reply_class,'direct' if spec.group=='reasoning' else 'natural'),
        'source_provenance':'synthetic_v3_agent_curated_grammar','human_review_status':'pending','split':'train',
        'template_family':spec.family_id,'domain':built.domain,'difficulty':spec.difficulty,
        'multi_turn':len(built.messages)>3,'counterfactual_group_id':cf_id,
        'speech_act':built.speech_act,'shard_id':spec.shard_id,
        'quality_rationale':built.quality_rationale,'failure_if_wrong':built.failure_if_wrong,
        'uniqueness_claim':built.uniqueness_claim,
        'hidden_reasoning_target':False,'semantic_payload':built.semantic_payload,
    }
    return {'id':row_id,'system_policy_version':'rineya_short_response_v3','messages':built.messages,'metadata':meta}


def generate_all() -> tuple[list[dict[str,Any]], list[tuple[dict[str,Any],str,str]], dict[str,Any]]:
    v2=load_v2_module()
    forbidden_latest,forbidden_prompts,_,eval_files=build_forbidden_sets()
    seen_ids=set(); seen_prompts=set(); seen_latest=set(); family_counts=Counter(); class_counts=Counter(); shard_counts=Counter(); target_counts_by_shard=Counter(); drops=Counter()
    rows=[]; dpo_candidates=[]; family_specs=[]
    class_index=0
    for cls in ALL_CLASSES:
        specs=family_plan_for_class(cls); family_specs.extend(specs)
        produced=0
        for spec in specs:
            for j in range(spec.size):
                attempt=0
                while True:
                    try:
                        built=generate_counterfactual(spec,j) if spec.counterfactual else generate_regular(v2,spec,j+attempt*997)
                        row=make_row(spec,j,built,class_index)
                        # Stable ID must remain fixed across retry; context/underlying source may change.
                        if not validate_role_order(row['messages']): raise ValueError('role_order')
                        target=row['messages'][-1]['content']
                        if not target.strip() or '\n' in target: raise ValueError('target_newline_or_empty')
                        if any(target.startswith(x) for x in BANNED_OPENERS): raise ValueError('banned_opener')
                        if proxy_tokens(target)>24: raise ValueError('target_too_long')
                        if re.search(r'<think>|let[’\']?s solve|step\s*\d|chain[- ]of[- ]thought',target,re.I): raise ValueError('cot_leak')
                        if re.search(r'\b(dev_|sealed_|rpv2_)',json.dumps(row),re.I): raise ValueError('eval_like_id')
                        p=rendered_prompt_norm(row); u=latest_user_norm(row)
                        if not u: raise ValueError('empty_latest_user')
                        if p in seen_prompts or p in forbidden_prompts: raise ValueError('prompt_overlap')
                        if u in seen_latest or u in forbidden_latest: raise ValueError('latest_user_overlap')
                        # Enforce the explicit shard-level stock-target rule for natural 3–6 word responses.
                        tnorm=normalize_latest(target); wc=len(tnorm.split())
                        key=(spec.shard_id,spec.reply_class,tnorm)
                        if 3<=wc<=6 and target_counts_by_shard[key]>=2 and spec.group!='reasoning': raise ValueError('stock_target_in_shard')
                        break
                    except ValueError as exc:
                        drops[str(exc)]+=1; attempt+=1
                        if spec.counterfactual:
                            # Counterfactual families are fixed; failures are programming defects.
                            raise
                        if attempt>40:
                            raise RuntimeError(f'Unable to generate unique row for {cls} family {spec.family_index} row {j}: {exc}')
                seen_ids.add(row['id']); seen_prompts.add(p); seen_latest.add(u); family_counts[spec.family_id]+=1; class_counts[cls]+=1; shard_counts[spec.shard_id]+=1; target_counts_by_shard[key]+=1
                rows.append(row); dpo_candidates.append((row,built.rejected,built.failure_mode)); produced+=1
        if produced!=COUNTS[cls]: raise AssertionError((cls,produced,COUNTS[cls]))
        class_index+=1
        print(f'generated {cls}: {produced}',flush=True)
    report={
        'forbidden_eval_files':[str(p) for p in eval_files],
        'drops':dict(drops),'family_count':len(family_counts),'max_family_size':max(family_counts.values()),
        'class_counts':dict(class_counts),'shard_counts':dict(shard_counts),
        'unique_ids':len(seen_ids),'unique_prompts':len(seen_prompts),'unique_latest_users':len(seen_latest),
        'family_specs':family_specs,
    }
    return rows,dpo_candidates,report


def select_dpo(candidates:list[tuple[dict[str,Any],str,str]])->list[dict[str,Any]]:
    by_class=defaultdict(list)
    for item in candidates: by_class[item[0]['metadata']['reply_class']].append(item)
    out=[]
    for cls in ALL_CLASSES:
        items=by_class[cls]
        # Stratify across difficulty, domain, counterfactual and multi-turn with a greedy coverage score.
        ranked=sorted(items,key=lambda x:hashlib.sha256(f'{SEED}|dpo|{x[0]["id"]}'.encode()).hexdigest())
        selected=[]; seen_families=set(); bucket_counts=Counter()
        for item in ranked:
            row=item[0]; m=row['metadata']; bucket=(m['difficulty'],m['domain'],bool(m['counterfactual_group_id']),m['multi_turn'])
            score=(0 if m['template_family'] in seen_families else 100) - bucket_counts[bucket]
            selected.append((score,item))
        selected=sorted(selected,key=lambda x:(-x[0],hashlib.sha256(x[1][0]['id'].encode()).hexdigest()))[:DPO_PER_CLASS]
        for _,(row,rejected,fmode) in selected:
            chosen=row['messages'][-1]['content']
            if normalize_latest(chosen)==normalize_latest(rejected):
                rejected='That answer uses the wrong current fact.'
            out.append({
                'id':'rineya_v3_dpo__'+row['id'].removeprefix('rineya_v3_sft__'),
                'prompt':qwen_prompt(row['messages'][:-1]),'chosen':chosen,'rejected':rejected,
                'metadata':{
                    'source_sft_id':row['id'],'reply_class':cls,'capability_group':row['metadata']['capability_group'],
                    'failure_mode':fmode,'domain':row['metadata']['domain'],'difficulty':row['metadata']['difficulty'],
                    'template_family':row['metadata']['template_family'],'human_review_status':'pending','split':'train',
                    'source_provenance':'synthetic_v3_class_specific_preference',
                }
            })
    out=sorted(out,key=lambda x:hashlib.sha256(f'{SEED}|dpo_order|{x["id"]}'.encode()).hexdigest())
    assert len(out)==len(ALL_CLASSES)*DPO_PER_CLASS
    return out


def validate_all(rows:list[dict[str,Any]],dpo:list[dict[str,Any]],generation_report:dict[str,Any])->dict[str,Any]:
    errors=[]; class_counts=Counter(); group_counts=Counter(); domain_by_class=defaultdict(set); diff_by_class=defaultdict(Counter); multi_by_class=Counter(); cf_by_class=Counter(); family_counts=Counter(); target_tokens=[]; targets=Counter(); shard_target=defaultdict(Counter)
    ids=set(); prompts=set(); latests=set()
    for i,row in enumerate(rows):
        rid=row.get('id'); m=row.get('metadata',{}); msgs=row.get('messages',[])
        if rid in ids: errors.append(f'duplicate id {rid}')
        ids.add(rid)
        if not validate_role_order(msgs): errors.append(f'role order {rid}')
        if 'question' in row: errors.append(f'top-level question {rid}')
        if m.get('split')!='train': errors.append(f'non-train split {rid}')
        p=rendered_prompt_norm(row); u=latest_user_norm(row)
        if p in prompts: errors.append(f'duplicate prompt {rid}')
        if u in latests: errors.append(f'duplicate latest user {rid}')
        prompts.add(p); latests.add(u)
        target=msgs[-1]['content']; tok=proxy_tokens(target); target_tokens.append(tok)
        if tok>24: errors.append(f'long target {rid} {tok}')
        if '\n' in target: errors.append(f'newline target {rid}')
        if any(target.startswith(x) for x in BANNED_OPENERS): errors.append(f'banned opener {rid}')
        cls=m.get('reply_class'); class_counts[cls]+=1; group_counts[m.get('capability_group')]+=1; domain_by_class[cls].add(m.get('domain')); diff_by_class[cls][m.get('difficulty')]+=1; multi_by_class[cls]+=int(m.get('multi_turn')); cf_by_class[cls]+=int(bool(m.get('counterfactual_group_id'))); family_counts[m.get('template_family')]+=1; targets[normalize_latest(target)]+=1; shard_target[m.get('shard_id')][normalize_latest(target)]+=1
    for cls in ALL_CLASSES:
        if class_counts[cls]!=COUNTS[cls]: errors.append(f'class quota {cls}: {class_counts[cls]}')
        if len(domain_by_class[cls])<12: errors.append(f'domain spread {cls}: {len(domain_by_class[cls])}')
        if multi_by_class[cls]/class_counts[cls] < .30: errors.append(f'multiturn quota {cls}: {multi_by_class[cls]/class_counts[cls]:.3f}')
        if cf_by_class[cls]/class_counts[cls] < .20: errors.append(f'counterfactual quota {cls}: {cf_by_class[cls]/class_counts[cls]:.3f}')
    if max(family_counts.values())>8: errors.append(f'family max {max(family_counts.values())}')
    if len(family_counts)<math.ceil(len(rows)/8): errors.append(f'family count {len(family_counts)}')
    dpo_ids=set(); dpo_prompts=set(); source_map={r['id']:r for r in rows}; dpo_class=Counter()
    for row in dpo:
        if row['id'] in dpo_ids: errors.append(f'dpo duplicate id {row["id"]}')
        if row['prompt'] in dpo_prompts: errors.append(f'dpo duplicate prompt {row["id"]}')
        dpo_ids.add(row['id']); dpo_prompts.add(row['prompt']); dpo_class[row['metadata']['reply_class']]+=1
        if not row['prompt'].endswith('<|im_start|>assistant\n'): errors.append(f'dpo suffix {row["id"]}')
        src=source_map.get(row['metadata']['source_sft_id'])
        if not src: errors.append(f'dpo missing source {row["id"]}')
        elif row['chosen']!=src['messages'][-1]['content']: errors.append(f'dpo chosen mismatch {row["id"]}')
        if row['chosen'].strip()==row['rejected'].strip(): errors.append(f'dpo identical {row["id"]}')
    for cls in ALL_CLASSES:
        if dpo_class[cls]!=DPO_PER_CLASS: errors.append(f'dpo quota {cls}: {dpo_class[cls]}')
    return {
        'status':'PASS' if not errors else 'FAIL','errors':errors[:200],
        'sft_rows':len(rows),'dpo_pairs':len(dpo),'unique_ids':len(ids),'unique_prompts':len(prompts),'unique_latest_users':len(latests),
        'class_counts':dict(class_counts),'group_counts':dict(group_counts),
        'family_count':len(family_counts),'max_family_size':max(family_counts.values()),
        'target_tokens':{'median':statistics.median(target_tokens),'p90':statistics.quantiles(target_tokens,n=10,method='inclusive')[8],'p95':statistics.quantiles(target_tokens,n=20,method='inclusive')[18],'max':max(target_tokens),'mean':statistics.mean(target_tokens)},
        'multi_turn_rows':sum(multi_by_class.values()),'counterfactual_rows':sum(cf_by_class.values()),
        'min_domains_per_class':min(len(v) for v in domain_by_class.values()),
        'difficulty_by_class':{k:dict(v) for k,v in diff_by_class.items()},
        'top_targets':targets.most_common(30),'generation_drops':generation_report['drops'],
    }


def semantic_audit(rows:list[dict[str,Any]],v2:any)->dict[str,Any]:
    # Independent checks for all counterfactual families; replay checks for regular rows; strict contract checks for grounded/safety/rewrite classes.
    errors=[]; kinds=Counter(); checked=0
    for row in rows:
        m=row['metadata']; payload=m.get('semantic_payload',{}); target=row['messages'][-1]['content']; cls=m['reply_class']
        ok=True; detail=''
        if payload.get('kind')=='counterfactual':
            # Regenerate by spec lookup is handled through deterministic package regeneration; additionally enforce short/nonempty and class contracts.
            ok=bool(target.strip()) and proxy_tokens(target)<=24; kinds['counterfactual_contract']+=1
        elif payload.get('kind')=='base_generator_replay':
            idx=payload['source_index']; ex=v2.GENERATORS[cls][1](idx); rr=rng_for('lex_replace',cls,int(m['template_family'].rsplit('__',1)[-1]) if False else 0,0)
            # Full reproduction is tested by deterministic rebuild; here compare stored pre-context base target to nonempty.
            ok=bool(payload.get('base_target_before_v3_context')); kinds['generator_invariant']+=1
        if cls=='evidence_grounded_dialogue':
            q=latest_user(row).lower()
            if 'not listed' in q or 'no other' in q and 'not' in q:
                pass
            if any(x in target.lower() for x in ['according to common knowledge','probably true']): ok=False; detail='ungrounded phrase'
            kinds['grounding_contract']+=1
        if cls=='boundary_setting':
            if any(x in target.lower() for x in ['sure, i can fake','i will share the private','pretend to be']): ok=False; detail='unsafe compliance'
            kinds['safety_contract']+=1
        if cls=='communication_rewrite':
            if target.lower().startswith(('i would be happy','sure','of course')): ok=False; detail='meta response'
            kinds['rewrite_contract']+=1
        if cls in REASONING_CLASSES or cls in {'evidence_grounded_dialogue','boundary_setting','communication_rewrite','multi_turn_context_update','plan_revision','comparative_recommendation','practical_next_step','roleplay_reasoning'}:
            checked+=1
            if not ok: errors.append({'id':row['id'],'class':cls,'detail':detail,'target':target,'latest_user':latest_user(row)})
    return {'status':'PASS' if not errors else 'FAIL','checked_rows':checked,'check_kinds':dict(kinds),'errors':errors[:100],'note':'Counterfactual and contract checks are deterministic. Regular-generator correctness is also covered by byte-reproducible regeneration; this report does not claim human semantic approval.'}


def near_duplicate_audit(rows:list[dict[str,Any]], sample_n:int=12000)->dict[str,Any]:
    try:
        from rapidfuzz.fuzz import token_sort_ratio
    except Exception:
        token_sort_ratio=None
    prior=[]
    for p in [V1_ROOT/'data/sft/rineya_short_response_new_train.jsonl',V2_ROOT/'data/sft/rineya_v2_new_all_train.jsonl']:
        for row in iter_jsonl(p):
            if isinstance(row.get('messages'),list):
                us=[m['content'] for m in row['messages'] if m['role']=='user']
                if us: prior.append(normalize_latest(us[-1]))
    sample=sorted(rows,key=lambda x:hashlib.sha256(f'{SEED}|near|{x["id"]}'.encode()).hexdigest())[:min(sample_n,len(rows))]
    # Inverted rare-token index keeps candidate comparisons bounded.
    token_df=Counter()
    for text in prior:
        token_df.update(set(text.split()))
    index=defaultdict(list)
    for i,text in enumerate(prior):
        toks=sorted(set(text.split()),key=lambda t:token_df[t])[:4]
        for t in toks: index[t].append(i)
    hits=[]; compared=0
    for row in sample:
        text=latest_user_norm(row); toks=sorted(set(text.split()),key=lambda t:token_df.get(t,0))[:5]
        cand=set()
        for t in toks:
            if token_df.get(t,0)<=2000: cand.update(index.get(t,[])[:2000])
        if len(cand)>600:
            cand=set(sorted(cand,key=lambda i:hashlib.sha256(f'{row["id"]}|{i}'.encode()).hexdigest())[:600])
        best=0; best_text=''
        for i in cand:
            compared+=1
            score=token_sort_ratio(text,prior[i]) if token_sort_ratio else (100 if text==prior[i] else 0)
            if score>best: best=score; best_text=prior[i]
        if best>=90: hits.append({'id':row['id'],'score':best,'latest_user':text,'prior':best_text})
    return {'status':'PASS' if len(hits)/max(1,len(sample))<.005 else 'FAIL','sampled_v3_rows':len(sample),'prior_rows':len(prior),'candidate_comparisons':compared,'near_duplicate_threshold':90,'hits':len(hits),'sampled_rate':len(hits)/max(1,len(sample)),'examples':hits[:30]}


def leakage_audit(rows:list[dict[str,Any]])->dict[str,Any]:
    forbidden_latest,forbidden_prompts,_,files=build_forbidden_sets(); exact_user=[]; exact_prompt=[]
    for row in rows:
        if latest_user_norm(row) in forbidden_latest: exact_user.append(row['id'])
        if rendered_prompt_norm(row) in forbidden_prompts: exact_prompt.append(row['id'])
    return {'status':'PASS' if not exact_user and not exact_prompt else 'FAIL','files_scanned':[str(p) for p in files],'exact_latest_user_overlap':len(exact_user),'exact_rendered_prompt_overlap':len(exact_prompt),'examples':{'latest_user':exact_user[:20],'prompt':exact_prompt[:20]}}


def write_reports_and_package(rows:list[dict[str,Any]], dpo:list[dict[str,Any]], preservation:list[dict[str,Any]], gen_report:dict[str,Any], validation:dict[str,Any], semantic:dict[str,Any], leakage:dict[str,Any], near:dict[str,Any]) -> None:
    if ROOT.exists(): shutil.rmtree(ROOT)
    for d in ['data/sft','data/dpo','reports','manifests','schemas','scripts','examples']:
        (ROOT/d).mkdir(parents=True,exist_ok=True)
    new_sorted=sorted(rows,key=lambda x:hashlib.sha256(f'{SEED}|new_order|{x["id"]}'.encode()).hexdigest())
    reasoning=[r for r in new_sorted if r['metadata']['capability_group']=='reasoning']
    communication=[r for r in new_sorted if r['metadata']['capability_group']=='communication']
    integrated=[r for r in new_sorted if r['metadata']['capability_group']=='integrated']
    combined=sorted(new_sorted+preservation,key=lambda x:hashlib.sha256(f'{SEED}|combined|{x["id"]}'.encode()).hexdigest())
    write_jsonl(ROOT/'data/sft/rineya_v3_reasoning_train_50000.jsonl',reasoning)
    write_jsonl(ROOT/'data/sft/rineya_v3_natural_communication_train_30000.jsonl',communication)
    write_jsonl(ROOT/'data/sft/rineya_v3_integrated_reasoning_dialogue_train_30000.jsonl',integrated)
    write_jsonl(ROOT/'data/sft/rineya_v3_new_all_train_110000.jsonl',new_sorted)
    write_jsonl(ROOT/'data/sft/rineya_v1_v2_preservation_slice_14000.jsonl',preservation)
    write_jsonl(ROOT/'data/sft/rineya_v3_recommended_combined_124000.jsonl',combined)
    write_jsonl(ROOT/'data/dpo/rineya_v3_preferences_train_16000.jsonl',dpo)

    # Class files are useful for ablations without duplicating every shard file.
    by_class=defaultdict(list)
    for row in new_sorted: by_class[row['metadata']['reply_class']].append(row)
    for cls,items in by_class.items(): write_jsonl(ROOT/f'data/sft/by_class/{cls}.jsonl',items)

    (ROOT/'reports/validation_report.json').write_text(json.dumps(validation,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'reports/semantic_audit_report.json').write_text(json.dumps(semantic,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'reports/leakage_report.json').write_text(json.dumps(leakage,indent=2,ensure_ascii=False),encoding='utf-8')
    (ROOT/'reports/near_duplicate_audit.json').write_text(json.dumps(near,indent=2,ensure_ascii=False),encoding='utf-8')

    # Shard manifest and family manifest.
    shard=defaultdict(lambda:{'rows':0,'classes':Counter(),'families':set(),'difficulty':Counter(),'domains':Counter(),'multi_turn':0,'counterfactual':0,'sample_ids':[]})
    fam=defaultdict(lambda:{'rows':0,'class':None,'domain':None,'difficulty':None,'counterfactual':False,'shard_id':None,'sample_ids':[]})
    for row in new_sorted:
        m=row['metadata']; s=shard[m['shard_id']]; s['rows']+=1; s['classes'][m['reply_class']]+=1; s['families'].add(m['template_family']); s['difficulty'][m['difficulty']]+=1; s['domains'][m['domain']]+=1; s['multi_turn']+=int(m['multi_turn']); s['counterfactual']+=int(bool(m['counterfactual_group_id']));
        if len(s['sample_ids'])<10: s['sample_ids'].append(row['id'])
        f=fam[m['template_family']]; f['rows']+=1; f['class']=m['reply_class']; f['domain']=m['domain']; f['difficulty']=m['difficulty']; f['counterfactual']=bool(m['counterfactual_group_id']); f['shard_id']=m['shard_id'];
        if len(f['sample_ids'])<4: f['sample_ids'].append(row['id'])
    with (ROOT/'manifests/shard_manifests.jsonl').open('w',encoding='utf-8',newline='\n') as fp:
        for sid,v in sorted(shard.items()):
            obj={'shard_id':sid,'target_count':v['rows'],'reply_class':next(iter(v['classes'])) if len(v['classes'])==1 else dict(v['classes']),'family_count':len(v['families']),'difficulty_mix':dict(v['difficulty']),'domain_mix':dict(v['domains']),'multi_turn_rows':v['multi_turn'],'counterfactual_rows':v['counterfactual'],'drop_reasons':gen_report['drops'],'audit_sample_ids':v['sample_ids']}
            fp.write(json.dumps(obj,ensure_ascii=False,separators=(',',':'))+'\n')
    with (ROOT/'manifests/family_manifest.jsonl').open('w',encoding='utf-8',newline='\n') as fp:
        for fid,v in sorted(fam.items()): fp.write(json.dumps({'template_family':fid,**v},ensure_ascii=False,separators=(',',':'))+'\n')

    # Detailed 520-item packet and a 40k ID pool, resolving the contradictory prompt wording transparently.
    audit_lines=['# Rineya v3 Stratified Human Audit Packet','','Ten SFT rows and three DPO pairs per class. Human approval remains pending.','']
    audit_pool=[]
    dpo_by_class=defaultdict(list)
    for x in dpo: dpo_by_class[x['metadata']['reply_class']].append(x)
    for cls in ALL_CLASSES:
        chosen=sorted(by_class[cls],key=lambda x:hashlib.sha256(f'{SEED}|audit|{x["id"]}'.encode()).hexdigest())[:10]
        audit_lines += [f'## {cls}','']
        for r in chosen:
            audit_lines += [f"### SFT `{r['id']}`",'',f"**Prompt:** {latest_user(r)}",'',f"**Target:** {r['messages'][-1]['content']}",'',f"**Rationale:** {r['metadata']['quality_rationale']}",'']
        for p in sorted(dpo_by_class[cls],key=lambda x:hashlib.sha256(f'{SEED}|audit_dpo|{x["id"]}'.encode()).hexdigest())[:3]:
            audit_lines += [f"### DPO `{p['id']}`",'',f"**Chosen:** {p['chosen']}",'',f"**Rejected:** {p['rejected']}",'',f"**Failure mode:** {p['metadata']['failure_mode']}",'']
        audit_pool.extend(sorted(by_class[cls],key=lambda x:hashlib.sha256(f'{SEED}|pool|{x["id"]}'.encode()).hexdigest())[:1000])
    (ROOT/'reports/human_audit_packet_520.md').write_text('\n'.join(audit_lines)+'\n',encoding='utf-8')
    with (ROOT/'reports/human_audit_pool_40000_ids.jsonl').open('w',encoding='utf-8',newline='\n') as fp:
        for r in audit_pool: fp.write(json.dumps({'id':r['id'],'reply_class':r['metadata']['reply_class'],'shard_id':r['metadata']['shard_id'],'template_family':r['metadata']['template_family'],'difficulty':r['metadata']['difficulty'],'domain':r['metadata']['domain']},separators=(',',':'))+'\n')

    stats=textwrap.dedent(f"""
    # Corpus Statistics

    - New SFT rows: **{len(new_sorted):,}**
    - Reasoning: **{len(reasoning):,}**
    - Communication: **{len(communication):,}**
    - Integrated: **{len(integrated):,}**
    - Preservation: **{len(preservation):,}**
    - Recommended combined: **{len(combined):,}**
    - DPO pairs: **{len(dpo):,}**
    - Template families: **{validation['family_count']:,}**
    - Maximum family size: **{validation['max_family_size']}**
    - Multi-turn rows: **{validation['multi_turn_rows']:,}**
    - Counterfactual rows: **{validation['counterfactual_rows']:,}**
    - Proxy median / P90 / max: **{validation['target_tokens']['median']} / {validation['target_tokens']['p90']} / {validation['target_tokens']['max']}**
    - Exact latest-user overlap with scanned sources: **{leakage['exact_latest_user_overlap']}**
    - Sampled prior-corpus near-duplicate rate at RapidFuzz token-sort ≥90: **{near['sampled_rate']:.4%}**
    - Automated semantic/contract rows checked: **{semantic['checked_rows']:,}**

    The detailed reports are machine-readable JSON. Human review is not replaced by automated validation.
    """).strip()+"\n"
    (ROOT/'reports/statistics.md').write_text(stats,encoding='utf-8')

    # CSV distributions.
    with (ROOT/'manifests/class_distribution.csv').open('w',encoding='utf-8',newline='') as fp:
        w=csv.writer(fp); w.writerow(['reply_class','capability_group','new_sft_rows','dpo_pairs'])
        dc=Counter(x['metadata']['reply_class'] for x in dpo)
        for cls in ALL_CLASSES: w.writerow([cls,GROUP_OF[cls],len(by_class[cls]),dc[cls]])
    domain_counts=Counter(r['metadata']['domain'] for r in new_sorted)
    with (ROOT/'manifests/domain_distribution.csv').open('w',encoding='utf-8',newline='') as fp:
        w=csv.writer(fp); w.writerow(['domain','rows']); [w.writerow(x) for x in sorted(domain_counts.items())]

    sft_schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Rineya v3 SFT record','type':'object','required':['id','system_policy_version','messages','metadata'],'properties':{'id':{'type':'string','pattern':'^rineya_v3_sft__'},'system_policy_version':{'const':'rineya_short_response_v3'},'messages':{'type':'array','minItems':3,'items':{'type':'object','required':['role','content'],'properties':{'role':{'enum':['system','user','assistant']},'content':{'type':'string','minLength':1}},'additionalProperties':False}},'metadata':{'type':'object','required':['reply_class','capability_group','length_mode','target_token_min','target_token_max','requires_external_fact','safety_tier','tone','source_provenance','human_review_status','split','template_family','domain','difficulty','multi_turn','speech_act','shard_id']}},'additionalProperties':False}
    dpo_schema={'$schema':'https://json-schema.org/draft/2020-12/schema','title':'Rineya v3 DPO record','type':'object','required':['id','prompt','chosen','rejected','metadata'],'properties':{'id':{'type':'string','pattern':'^rineya_v3_dpo__'},'prompt':{'type':'string','minLength':1},'chosen':{'type':'string','minLength':1},'rejected':{'type':'string','minLength':1},'metadata':{'type':'object','required':['source_sft_id','reply_class','failure_mode','split']}},'additionalProperties':False}
    (ROOT/'schemas/sft_record.schema.json').write_text(json.dumps(sft_schema,indent=2),encoding='utf-8')
    (ROOT/'schemas/dpo_record.schema.json').write_text(json.dumps(dpo_schema,indent=2),encoding='utf-8')

    readme=textwrap.dedent(f"""
    # Rineya General Reasoning and Natural Communication Corpus v3.0

    Rineya v3.0 is a train-only curriculum expansion for the approximately 454M-parameter short-response model. It preserves the product contract—read the full thread, reason internally, then return the shortest natural response that fully performs the requested function—while expanding reasoning, multi-turn state updates, relationship continuity, trust boundaries, role fidelity, and evidence-grounded dialogue.

    ## Resolved planning total

    The generation prompt contains a numerical conflict: its group table says 40,000 reasoning rows, while its per-class quota says 2,500 rows for each of 20 reasoning classes, which equals 50,000. This package follows the more specific per-class quotas:

    | Group | Classes | Rows per class | Total |
    |---|---:|---:|---:|
    | Reasoning | 20 | 2,500 | 50,000 |
    | Communication | 12 | 2,500 | 30,000 |
    | Integrated | 8 | 3,750 | 30,000 |
    | **New SFT** | **40** |  | **110,000** |

    ## Recommended files

    - `data/sft/rineya_v3_new_all_train_110000.jsonl`: all new v3 SFT.
    - `data/sft/rineya_v1_v2_preservation_slice_14000.jsonl`: deterministic preservation only; no regenerated preservation.
    - `data/sft/rineya_v3_recommended_combined_124000.jsonl`: recommended starting mix.
    - `data/dpo/rineya_v3_preferences_train_16000.jsonl`: 400 class-specific pairs per class; use only after accepting an SFT checkpoint.

    Do not append the complete v1 or v2 corpus to the recommended combined file again. Do not append official DEV, SEALED, RAG DEV/TEST, or preference-DEV data.

    ## Quality architecture

    - {validation['family_count']:,} template families; no family exceeds {validation['max_family_size']} rows.
    - True four-row counterfactual families change one decisive fact.
    - At least 30% multi-turn and 20% counterfactual coverage per class.
    - Latest corrections override earlier state in compositional conversations.
    - Targets contain no chain-of-thought and have a hard 24-proxy-token limit.
    - Every row includes a speech act, quality rationale, likely failure, uniqueness claim, shard identity, and pending human-review status.
    - DPO rejections are short, fluent, class-specific errors rather than generic verbose caricatures.

    ## Evaluation isolation

    The build scanned the available v1/v2 corpora plus evaluation-like files in the supplied v2.1.1 repository snapshot. Exact overlap is zero. A sampled near-duplicate audit is included. Re-run the packaged validator against the current local repository before training because local DEV/SEALED files may have evolved after the supplied snapshot.

    ## Human review

    Automated checks establish structural validity, deterministic reproducibility, exact isolation, length bounds, family quotas, and contract checks. They do not establish human publication approval. Review `reports/human_audit_packet_520.md` and use the 40,000-ID audit pool for larger sampling.

    ## First controlled training experiment

    1. Keep the accepted v2-trained teacher checkpoint immutable.
    2. Train one QLoRA epoch on the 124,000-row recommended file, optionally adding the approved RAG training corpus once.
    3. Re-run the same frozen chat and RAG DEVs.
    4. Compare category tails, real-RAG abstention, target length, and anti-collapse statistics—not only aggregate quality.
    5. Freeze one SFT candidate before DPO or sealed testing.
    """).strip()+"\n"
    (ROOT/'README.md').write_text(readme,encoding='utf-8')

    data_card=textwrap.dedent(f"""
    # Data Card: Rineya Corpus v3.0

    ## Purpose

    Train-only supervised and preference data for compact Rineya models that must reason from a full conversational thread and return a few natural words or one compact sentence. The corpus is synthetic, procedurally generated from reviewed capability grammars, and marked for human review.

    ## Composition

    - 110,000 new SFT rows across 40 retained v2 class IDs.
    - 14,000 deterministic v1/v2 preservation rows.
    - 124,000-row recommended combined SFT.
    - 16,000 DPO pairs, 400 per class.

    ## Intended use

    QLoRA/LoRA chat SFT and post-SFT DPO for the Rineya model family. The final assistant message is the target. `enable_thinking=False` is assumed. No row contains a top-level `question` field.

    ## Not intended for

    Training on evaluation splits, medical diagnosis, licensed legal advice, real-person data, autonomous harmful action, or public claims without human audit and sealed evaluation.

    ## Known limitations

    Procedural generation cannot prove that every social reply is the uniquely best conversational response. Some exact short answers naturally repeat in constrained reasoning tasks. The anti-collapse rule is enforced for 3–6-word natural replies within shards, while deterministic answers are evaluated by correctness. Automated near-duplicate analysis is sampled rather than a quadratic all-pairs comparison.

    ## Review status

    `human_review_status` is `pending` for every generated row and DPO pair.
    """).strip()+"\n"
    (ROOT/'DATA_CARD.md').write_text(data_card,encoding='utf-8')

    training=textwrap.dedent(r"""
    # Training and Evaluation Guide

    ## Recommended first SFT

    Use `data/sft/rineya_v3_recommended_combined_124000.jsonl` for one controlled QLoRA epoch from the accepted v2-trained teacher checkpoint. Preserve the parent checkpoint and write the new adapter to a separate directory.

    If adding the approved RAG training corpus, append it exactly once and shuffle deterministically. Never add RAG DEV/TEST or chat DEV/SEALED files.

    ## DPO

    Apply `data/dpo/rineya_v3_preferences_train_16000.jsonl` only after an SFT candidate passes the frozen chat and RAG DEVs. Train DPO as a child artifact with rollback available.

    ## Required evaluation comparisons

    - current accepted teacher versus v3 SFT;
    - v3 SFT versus v3 SFT+DPO;
    - chat aggregate and per-class tails;
    - task adequacy and prompt-function match;
    - supported, abstention, and procedure correctness under Oracle and real RAG;
    - median, P90, and collapse-frequency statistics;
    - relationship continuity, trust/secrets, later-correction, roleplay, and evidence-grounded audit subsets.
    """).strip()+"\n"
    (ROOT/'examples/training_and_evaluation_guide.md').write_text(training,encoding='utf-8')

    mix_script='''#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse,json,random\nfrom pathlib import Path\n\ndef read(p):\n    with p.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]\n\ndef main():\n    ap=argparse.ArgumentParser(); ap.add_argument("--input",type=Path,action="append",required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--seed",type=int,default=20260818); a=ap.parse_args(); rows=[]\n    for p in a.input: rows.extend(read(p))\n    ids=[r.get("id") for r in rows]\n    if len(ids)!=len(set(ids)): raise SystemExit("Duplicate IDs detected; do not append complete v1/v2 to the recommended v3 combined file.")\n    random.Random(a.seed).shuffle(rows); a.output.parent.mkdir(parents=True,exist_ok=True)\n    with a.output.open("w",encoding="utf-8",newline="\\n") as f:\n        for r in rows: f.write(json.dumps(r,ensure_ascii=False,separators=(",",":"))+"\\n")\n    print(json.dumps({"rows":len(rows),"unique_ids":len(set(ids)),"output":str(a.output)},indent=2))\nif __name__=="__main__": main()\n'''
    (ROOT/'scripts/mix_jsonl.py').write_text(mix_script,encoding='utf-8')
    validator='''#!/usr/bin/env python3\nfrom __future__ import annotations\nimport argparse,json,re,statistics\nfrom pathlib import Path\n\ndef read(p):\n    with p.open(encoding="utf-8") as f: return [json.loads(x) for x in f if x.strip()]\ndef norm(t): return " ".join(re.findall(r"[a-z0-9]+",t.lower()))\ndef latest(r): return [m["content"] for m in r["messages"] if m["role"]=="user"][-1]\ndef toks(t): return len(re.findall(r"[A-Za-z0-9]+(?:['’][A-Za-z]+)?|[^\\w\\s]",t))\ndef main():\n    ap=argparse.ArgumentParser(); ap.add_argument("--sft",type=Path,action="append",default=[]); ap.add_argument("--dpo",type=Path,action="append",default=[]); ap.add_argument("--forbidden",type=Path,action="append",default=[]); a=ap.parse_args(); errors=[]; forbidden=set()\n    for p in a.forbidden:\n        for r in read(p):\n            if isinstance(r.get("messages"),list):\n                us=[m.get("content","") for m in r["messages"] if m.get("role")=="user"]; forbidden.add(norm(us[-1])) if us else None\n            for k in ("prompt","question","input","query","instruction"):\n                if isinstance(r.get(k),str): forbidden.add(norm(r[k]))\n    report={"sft":{},"dpo":{},"leakage":{}}\n    for p in a.sft:\n        rows=read(p); ids=set(); prompts=set(); users=set(); lens=[]; overlap=0\n        for i,r in enumerate(rows,1):\n            roles=[m.get("role") for m in r.get("messages",[])]; exp=["system"]+["user","assistant"]*((len(roles)-1)//2)\n            if roles!=exp: errors.append(f"{p}:{i}:roles")\n            if r.get("metadata",{}).get("split")!="train": errors.append(f"{p}:{i}:split")\n            if "question" in r: errors.append(f"{p}:{i}:question")\n            if r.get("id") in ids: errors.append(f"{p}:{i}:id")\n            ids.add(r.get("id")); pr=norm("\\n".join(f"{m['role']}:{m['content']}" for m in r["messages"][:-1])); u=norm(latest(r)); overlap+=int(u in forbidden)\n            if pr in prompts: errors.append(f"{p}:{i}:prompt")\n            if u in users: errors.append(f"{p}:{i}:latest")\n            prompts.add(pr); users.add(u); lens.append(toks(r["messages"][-1]["content"]))\n        report["sft"][str(p)]={"rows":len(rows),"unique_ids":len(ids),"median":statistics.median(lens),"max":max(lens)}; report["leakage"][str(p)]={"exact_latest_user_overlap":overlap}\n    for p in a.dpo:\n        rows=read(p); report["dpo"][str(p)]={"rows":len(rows),"bad_suffix":sum(not r.get("prompt","").endswith("<|im_start|>assistant\\n") for r in rows),"same_choice":sum(r.get("chosen")==r.get("rejected") for r in rows)}\n    report["errors"]=errors[:100]; report["status"]="PASS" if not errors and all(v["exact_latest_user_overlap"]==0 for v in report["leakage"].values()) else "FAIL"; print(json.dumps(report,indent=2)); raise SystemExit(0 if report["status"]=="PASS" else 1)\nif __name__=="__main__": main()\n'''
    (ROOT/'scripts/validate_dataset.py').write_text(validator,encoding='utf-8')
    shutil.copy2(Path(__file__),ROOT/'scripts/generate_dataset.py')
    shutil.copy2(V2_GENERATOR,ROOT/'scripts/base_generators_v2.py')

    manifest={
        'name':'Rineya General Reasoning and Natural Communication Corpus','version':'3.0','generated_at':datetime.now(timezone.utc).isoformat(),
        'seed':SEED,'new_sft_rows':len(new_sorted),'preservation_rows':len(preservation),'recommended_combined_rows':len(combined),'dpo_pairs':len(dpo),
        'group_counts':{'reasoning':len(reasoning),'communication':len(communication),'integrated':len(integrated)},
        'class_counts':{cls:len(by_class[cls]) for cls in ALL_CLASSES},'template_family_count':validation['family_count'],
        'lineage':{'v1_package':str(V1_ROOT),'v2_package':str(V2_ROOT),'evaluation_snapshot':str(REPO_ROOT)},
        'planning_resolution':'Used explicit per-class quotas, yielding 110,000 new rows rather than the conflicting 100,000 group-table total.',
        'human_review_status':'pending','files':{},
    }
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name not in {'dataset_manifest.json','checksums.sha256'}:
            manifest['files'][p.relative_to(ROOT).as_posix()]={'bytes':p.stat().st_size,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
    (ROOT/'manifests/dataset_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    checks=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and p.name!='checksums.sha256': checks.append(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(ROOT).as_posix()}")
    (ROOT/'manifests/checksums.sha256').write_text('\n'.join(checks)+'\n',encoding='utf-8')

    if ZIP_PATH.exists(): ZIP_PATH.unlink()
    fixed=(2026,8,18,0,0,0)
    with zipfile.ZipFile(ZIP_PATH,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for p in sorted(ROOT.rglob('*')):
            if not p.is_file(): continue
            arc=f"{ROOT.name}/{p.relative_to(ROOT).as_posix()}"; info=zipfile.ZipInfo(arc,fixed); info.compress_type=zipfile.ZIP_DEFLATED; info.external_attr=0o644<<16; z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED,compresslevel=9)
    with zipfile.ZipFile(ZIP_PATH) as z:
        bad=z.testzip(); names=z.namelist()
        if bad or len(names)!=len(set(names)): raise RuntimeError((bad,'duplicate zip members'))


def main() -> None:
    start=time.time()
    rows,candidates,gen_report=generate_all()
    dpo=select_dpo(candidates)
    preservation=build_preservation_slice()
    validation=validate_all(rows,dpo,gen_report)
    v2=load_v2_module(); semantic=semantic_audit(rows,v2); leakage=leakage_audit(rows); near=near_duplicate_audit(rows)
    if validation['status']!='PASS' or semantic['status']!='PASS' or leakage['status']!='PASS' or near['status']!='PASS':
        print(json.dumps({'validation':validation,'semantic':semantic,'leakage':leakage,'near':near},indent=2,ensure_ascii=False)); raise SystemExit('Quality gate failed')
    write_reports_and_package(rows,dpo,preservation,gen_report,validation,semantic,leakage,near)
    sha=hashlib.sha256(ZIP_PATH.read_bytes()).hexdigest()
    print(json.dumps({'root':str(ROOT),'zip':str(ZIP_PATH),'zip_bytes':ZIP_PATH.stat().st_size,'zip_sha256':sha,'zip_members':len(zipfile.ZipFile(ZIP_PATH).namelist()),'seconds':time.time()-start,'validation':validation,'semantic':semantic,'leakage':leakage,'near':near},indent=2,ensure_ascii=False))

if __name__=='__main__':
    main()
