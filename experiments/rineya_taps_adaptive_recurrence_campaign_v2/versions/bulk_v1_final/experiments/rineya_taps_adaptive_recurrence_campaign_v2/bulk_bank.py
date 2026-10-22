"""Prospective BULK-TAPS-001 preparation: one nonce, no models/tokenizers/data reads."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import secrets

from .open_bank import normalize_output

VERSION = "BULK-TAPS-001"
PACKAGE = Path(__file__).resolve().parent
CATEGORIES = (
    "signed_ledgers", "deduction_composition", "program_state", "dialogue_memory",
    "document_retrieval_reconciliation", "answerability_source_conflict",
)
FAMILIES = {
    "signed_ledgers": ("signed_entry_list", "credit_debit_groups"),
    "deduction_composition": ("implication_paths", "conjunctive_paths"),
    "program_state": ("copy_add_registers", "multiply_branch_registers"),
    "dialogue_memory": ("numbered_speaker_turns", "numbered_memory_updates"),
    "document_retrieval_reconciliation": ("approved_consensus", "latest_approved_consensus"),
    "answerability_source_conflict": ("approved_evidence_transfer", "latest_evidence_transfer"),
}


def local_rng(nonce, seed, *parts):
    raw = json.dumps([VERSION, nonce, seed, *parts], separators=(",", ":")).encode()
    return random.Random(int.from_bytes(hashlib.sha256(raw).digest(), "big"))


def symbols(rng, count):
    result = []
    while len(result) < count:
        name = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(3))
        if name not in result and name not in ("YES", "NOT", "AND", "SET"):
            result.append(name)
    return result


def make_world(nonce, seed, category, index):
    split = "DEVELOPMENT" if index < 16 else "OPEN_VALIDATION"
    family = FAMILIES[category][split == "OPEN_VALIDATION"]
    level = "easy" if (index // 4) % 2 == 0 else "hard"
    rng = local_rng(nonce, seed, category, index)
    ids = symbols(rng, 24)
    role = index % 4
    world = {"category": category, "index": index, "split": split, "family": family,
             "level": level, "ids": ids, "query_role": role}
    if category == "signed_ledgers":
        accounts, answers = [], {}
        for slot, name in enumerate(ids[:4]):
            sign = 1 if (slot + index // 4) % 2 == 0 else -1
            start = rng.randint(10, 20) if sign > 0 else rng.randint(3, 7)
            changes = [sign*rng.randint(50, 60), -rng.randint(1, 5), sign*rng.randint(40, 50)]
            if level == "hard":
                changes.extend([sign*rng.randint(7, 12), -sign*rng.randint(2, 4),
                                sign*rng.randint(7, 12), -sign*rng.randint(2, 4)])
            accounts.append((name, start, changes))
            answers[name] = start + sum(changes)  # construction, not parser
        world.update(accounts=accounts, query=ids[role], gold=str(answers[ids[role]]),
                     kind="integer", status="ANSWER", factors={"entries_per_account": len(accounts[0][2]), "accounts": 4})
    elif category == "deduction_composition":
        depth = 2 if level == "easy" else 4
        needed = 4*(depth + (2 if split == "OPEN_VALIDATION" else 1))
        if needed > len(ids):
            ids = symbols(rng, needed)
            world["ids"] = ids
        rules, facts, ends = [], [], []
        cursor = 0
        for branch in range(4):
            width = depth + (2 if split == "OPEN_VALIDATION" else 1)
            chain = ids[cursor:cursor+width]
            cursor += width
            if split == "OPEN_VALIDATION":
                rules.append((chain[:2], chain[2]))
                for left, right in zip(chain[2:-1], chain[3:]):
                    rules.append(([left], right))
                facts.extend(chain[:2] if branch < 2 else chain[:1])
            else:
                rules.extend(([left], right) for left, right in zip(chain[:-1], chain[1:]))
                if branch < 2:
                    facts.append(chain[0])
            ends.append(chain[-1])
        world.update(rules=rules, facts=facts, query=ends[role],
                     gold="YES" if role < 2 else "NO", kind="choice", status="ANSWER",
                     factors={"inference_depth": depth, "branches": 4, "conjunction": split == "OPEN_VALIDATION"})
    elif category == "program_state":
        a, b, c, d = ids[:4]
        initial = {name: rng.randint(1, 5) for name in (a,b,c,d)}
        if split == "DEVELOPMENT":
            u,v,w,z = rng.randint(90,120),rng.randint(110,140),rng.randint(130,160),rng.randint(10,30)
            instructions = [f"{a}={b}+{u}",f"{b}={c}+{v}",f"{c}={d}+{w}",f"{d}={a}+{z}"]
            av,bv,cv,dv = initial[b]+u,initial[c]+v,initial[d]+w,initial[b]+u+z
            if level == "hard":
                x,y,s,t = rng.randint(3,8),rng.randint(9,18),rng.randint(12,25),rng.randint(2,6)
                instructions += [f"{a}={c}-{x}",f"{b}={d}+{y}",f"{c}={a}+{s}",f"{d}={b}-{t}"]
                av,bv = cv-x,dv+y
                cv,dv = av+s,bv-t
            branch_truth = None
        else:
            m,u,v,w = rng.randint(5,9),rng.randint(40,70),rng.randint(30,60),rng.randint(3,8)
            instructions = [f"{d}={a}*{m}",f"{b}={d}+{u}",f"{a}={b}+{v}",f"{c}={a}-{w}"]
            dv = initial[a]*m
            bv = dv+u
            av,cv = bv+v,bv+v-w
            branch_truth = None
            if level == "hard":
                # Freeze balanced branch truth before model outputs; threshold
                # is a real numeric comparison, not a label in the prompt.
                threshold = cv + (-rng.randint(3,9) if index%2==0 else rng.randint(3,9))
                z,q,s,t,r = (rng.randint(8,15) for _ in range(5))
                instructions += [f"IF {c}>{threshold} THEN {d}={b}+{z} ELSE {d}={a}-{q}",
                                 f"{a}={d}+{r}",f"{b}={c}+{s}",f"{c}={a}-{t}"]
                branch_truth = cv > threshold
                dv = bv+z if branch_truth else av-q
                av,bv,cv = dv+r,cv+s,dv+r-t
        values = dict(zip((a,b,c,d),(av,bv,cv,dv)))
        world.update(initial=initial, instructions=instructions, query=ids[role], gold=str(values[ids[role]]),
                     kind="integer", status="ANSWER", factors={"steps": len(instructions), "registers":4, "branch_truth":branch_truth})
    elif category == "dialogue_memory":
        people = ids[:4]
        codes = rng.sample(range(101,9900),12)  # magnitude independent of event age
        old,middle,latest = codes[:4],codes[4:8],codes[8:]
        events, clock = [], 0
        histories=[]
        for slot in range(4):
            if level=="easy":
                history=[("CLEAR",None),("SET",middle[slot])] if slot<2 else [("SET",old[slot]),("CLEAR",None)]
            elif slot<2:
                history=([("CLEAR",None),("SET",middle[slot]),("QUOTE",old[slot]),("SET",latest[slot])]
                         if (slot+index//4)%2==0 else
                         [("SET",old[slot]),("CLEAR",None),("SET",latest[slot]),("QUOTE",old[slot])])
            else:
                history=([("CLEAR",None),("SET",middle[slot]),("QUOTE",old[slot]),("CLEAR",None)]
                         if (slot+index//4)%2==0 else
                         [("SET",old[slot]),("QUOTE",old[slot]),("SET",latest[slot]),("CLEAR",None)])
            histories.append(history)
        for stage in range(4 if level == "hard" else 2):
            order = list(range(4))
            rng.shuffle(order)
            for slot in order:
                clock += 1
                action,value=histories[slot][stage]
                events.append((clock,people[slot],action,value))
        query = people[role]
        gold = str((latest if level == "hard" else middle)[role]) if role < 2 else "UNKNOWN"
        world.update(events=events, query=query, gold=gold, kind="integer_or_unknown",
                     status="ANSWER" if role < 2 else "MISSING",
                     factors={"events": len(events), "people":4, "quotation_noise":level=="hard",
                              "code_magnitude_age_independent":True,"clear_redeclaration_present":True})
    else:
        # All entities receive bundles before query selection. Pair interventions
        # exchange complete bundles, leaving global authority/status counts fixed.
        people = ids[:4]
        values = rng.sample(range(101,999),12)
        statuses = ["ANSWER","ANSWER","CONFLICT","MISSING"]
        if (index // 4) % 2:
            statuses[2],statuses[3] = statuses[3],statuses[2]
        bundles = {}
        for slot,name in enumerate(people):
            value,other,noise = values[3*slot:3*slot+3]
            status = statuses[slot]
            approved = "D" if status == "MISSING" else "A"
            now = [(approved,2,value),(approved,2,other if status=="CONFLICT" else value)]
            if level == "hard":
                now.append(("A" if split=="OPEN_VALIDATION" and status!="MISSING" else "D",1,noise))
            bundles[name] = now
        if category == "answerability_source_conflict":
            good = people[index % 2]
            bad_status = "MISSING" if (index//2)%2 == 0 else "CONFLICT"
            bad = people[statuses.index(bad_status)]
            query = people[role]
            # Rename the already-built bundles symmetrically so the selected
            # query starts answerable. Both displayed contexts contain all roles.
            if query != good:
                bundles[query],bundles[good] = bundles[good],bundles[query]
                if bad == query:
                    bad = good
            gold = str(next(v for flag,date,v in bundles[query] if flag=="A" and date==2))
            world.update(bundles=bundles,query=query,swap=bad,gold=gold,kind="integer_or_unknown",
                         status="ANSWER",pair_status=bad_status,
                         factors={"records":sum(map(len,bundles.values())),"entities":4,"intervention":"evidence_bundle_exchange"})
        else:
            query = people[role]
            status = statuses[role]
            gold = str(values[3*role]) if status=="ANSWER" else "UNKNOWN"
            world.update(bundles=bundles,query=query,gold=gold,kind="integer_or_unknown",status=status,
                         factors={"records":sum(map(len,bundles.values())),"entities":4,"latest_rule":split=="OPEN_VALIDATION"})
    return world


def render(world, nonce, seed, variant):
    rng = local_rng(nonce,seed,world["category"],world["index"],"render",variant)
    style = (world["index"]+variant)%2
    query = world["query"]
    category = world["category"]
    gold,status = world["gold"],world["status"]
    if category == "signed_ledgers":
        rows=[]
        for name,start,changes in world["accounts"]:
            items=list(changes)
            rng.shuffle(items)
            if world["split"]=="DEVELOPMENT":
                rows.append(f"{name} open {start} entries " + ",".join(f"{v:+}" for v in items))
            else:
                rows.append(f"{name} open {start} credits " + ",".join(str(v) for v in items if v>0) +
                            " debits " + ",".join(str(-v) for v in items if v<0))
        rng.shuffle(rows)
        body="LEDGERS: add signed entries, or credits minus debits, to opening values.\n"+";".join(rows)+"."
    elif category == "deduction_composition":
        rules=["&".join(left)+">"+right for left,right in world["rules"]]
        facts=list(world["facts"])
        rng.shuffle(rules);rng.shuffle(facts)
        body="RULES: infer right symbols only when every left symbol is known. "+",".join(rules)+".\nFACTS: "+",".join(facts)+"."
    elif category == "program_state":
        instructions=[f"{i+1} {text}" for i,text in enumerate(world["instructions"])]
        rng.shuffle(instructions)
        initial=list(world["initial"].items());rng.shuffle(initial)
        body="PROGRAM: execute numbered steps in order using current registers.\nINIT: "+",".join(f"{n}={v}" for n,v in initial)+".\nSTEPS: "+";".join(instructions)+"."
    elif category == "dialogue_memory":
        events=list(world["events"])
        rng.shuffle(events)
        if world["split"]=="DEVELOPMENT":
            pieces=[f"Turn {n} {person} "+("says "+str(value) if action=="SET" else "withdraws" if action=="CLEAR" else "quotes "+str(value))
                    for n,person,action,value in events]
        else:
            pieces=[f"Update {n} {person} {action}"+(" "+str(value) if value is not None else "")
                    for n,person,action,value in events]
        body="MEMORY: follow numbered turns/updates. New declarations replace old codes; withdrawal/CLEAR removes them. QUOTE changes nothing.\n"+";".join(pieces)+"."
    else:
        bundles={name:list(records) for name,records in world["bundles"].items()}
        if category=="answerability_source_conflict" and variant==1:
            bundles[query],bundles[world["swap"]]=bundles[world["swap"]],bundles[query]
            gold,status="UNKNOWN",world["pair_status"]
        records=[]
        for name,bundle in bundles.items():
            records.extend(f"{name} {flag}"+(f" d{date}" if world["split"]=="OPEN_VALIDATION" else "")+f" {value}"
                           for flag,date,value in bundle)
        rng.shuffle(records)
        body=("DOCUMENTS: use only A-approved records"+("; latest approved date per entity" if world["split"]=="OPEN_VALIDATION" else "")+
              ". Equal approved values give that value; absent/conflicting values give UNKNOWN.\n"+";".join(records)+".")
    instruction = "Reply only YES or NO." if world["kind"]=="choice" else "Reply one integer." if world["kind"]=="integer" else "Reply one integer or UNKNOWN."
    question = (f"\nQuestion: {query}? {instruction}" if style==0 else f"\nQuery: {query}. {instruction}")
    return body+question,gold,status,style


def generate(nonce,seed=17):
    rows=[]
    for category in CATEGORIES:
        for index in range(24):
            world=make_world(nonce,seed,category,index)
            parent=f"{VERSION}:{nonce}:{category}:{index}"
            for variant in (0,1):
                prompt,gold,status,style=render(world,nonce,seed,variant)
                identifier=f"{parent}:v{variant}"
                pair_kind="EVIDENCE_TRANSFER" if category=="answerability_source_conflict" else "SAME_GOLD_PROMPT_VARIANTS"
                rows.append({"id":identifier,"case_id":identifier,"parent_id":parent,"scenario_id":parent,
                    "split":world["split"],"category":category,"family":category,
                    "generator_family":category,"generator_variant":world["family"],
                    "variant_id":variant,"transformation_id":f"v{variant}",
                    "pair_kind":pair_kind,"difficulty_stratum":world["level"],"difficulty_factors":world["factors"],
                    "query_role":world["query_role"],"wording_style":style,"prompt":prompt,
                    "gold":gold,"gold_answer":gold,"answer_kind":world["kind"],
                    "answerability":"unanswerable" if gold=="UNKNOWN" else "answerable","source_status":status,
                    "expected_response_change":pair_kind=="EVIDENCE_TRANSFER",
                    "shape":{"target_max_input_tokens":350,"max_output_tokens":8,"native_tokenizer_binding":"NOT_RUN"},
                    "provenance":{"generator_version":VERSION,"nonce":nonce,"seed":seed,"historical_parents":[],
                        "construction":"independently seeded synthetic parent; shared authored generator/template",
                        "human_review":"NOT_REVIEWED"},
                    "exposure":"OPEN_SOURCE_VISIBLE_NO_NATIVE_EVALUATION",
                    "independence":"144 randomized parents are not144 independently authored generators; primary validation48parents",
                    "prompt_sha256":hashlib.sha256(prompt.encode()).hexdigest()})
    return rows


def parse_prompt(prompt):
    """Verifier parses visible text, never construction/gold/family metadata."""
    if not isinstance(prompt,str):
        raise TypeError("raw prompt string required")
    body,question=re.split(r"\n(?:Question|Query): ",prompt)
    query=re.match(r"([A-Z]{3})[?.] ",question)[1]
    if body.startswith("LEDGERS:"):
        values={}
        for text in body.split("\n",1)[1].rstrip(".").split(";"):
            signed=re.fullmatch(r"([A-Z]{3}) open (\d+) entries ([-+\d,]+)",text)
            grouped=re.fullmatch(r"([A-Z]{3}) open (\d+) credits ([\d,]*) debits ([\d,]*)",text)
            if signed:
                name,start,entries=signed.groups()
                current=int(start)
                for amount in entries.split(","):current+=int(amount)
            elif grouped:
                name,start,credits,debits=grouped.groups()
                current=int(start)
                for amount in credits.split(","):
                    if amount:current+=int(amount)
                for amount in debits.split(","):
                    if amount:current-=int(amount)
            else:raise ValueError("invalid ledger")
            if name in values:raise ValueError("duplicate ledger")
            values[name]=current
        return str(values[query]),"ANSWER"
    if body.startswith("RULES:"):
        ruletext,facttext=body.split("\nFACTS: ")
        ruletext=ruletext.split(". ",1)[1].rstrip(".")
        rules=[(set(x.split(">")[0].split("&")),x.split(">")[1]) for x in ruletext.split(",")]
        known=set(facttext.rstrip(".").split(","))
        while True:
            added={right for left,right in rules if left<=known}-known
            if not added:break
            known|=added
        return ("YES" if query in known else "NO"),"ANSWER"
    if body.startswith("PROGRAM:"):
        init,steps=body.split("\nINIT: ")[1].split(".\nSTEPS: ")
        state={name:int(value) for name,value in (item.split("=") for item in init.split(","))}
        instructions=[]
        for line in steps.rstrip(".").split(";"):
            step,text=line.split(" ",1);instructions.append((int(step),text))
        if len(set(n for n,_ in instructions))!=len(instructions):raise ValueError("duplicate step")
        def assignment(text):
            dest,src,operator,amount=re.fullmatch(r"([A-Z]{3})=([A-Z]{3})([-+*])(\d+)",text).groups()
            value=state[src];number=int(amount)
            state[dest]=value+number if operator=="+" else value-number if operator=="-" else value*number
        for _,text in sorted(instructions):
            if text.startswith("IF "):
                name,threshold,yes,no=re.fullmatch(r"IF ([A-Z]{3})>(\d+) THEN (.*?) ELSE (.*)",text).groups()
                assignment(yes if state[name]>int(threshold) else no)
            else:assignment(text)
        return str(state[query]),"ANSWER"
    if body.startswith("MEMORY:"):
        events=body.split("\n",1)[1].rstrip(".").split(";")
        parsed=[]
        for text in events:
            turn=re.fullmatch(r"Turn (\d+) ([A-Z]{3}) (says|withdraws|quotes)(?: (\d+))?",text)
            update=re.fullmatch(r"Update (\d+) ([A-Z]{3}) (SET|CLEAR|QUOTE)(?: (\d+))?",text)
            matched=turn or update
            if not matched:raise ValueError("invalid memory event")
            number,name,action,value=matched.groups()
            parsed.append((int(number),name,action,value))
        if len({x[0] for x in parsed})!=len(parsed):raise ValueError("duplicate memory number")
        codes={}
        for _,name,action,value in sorted(parsed):
            if action in ("says","SET"):codes[name]=str(int(value))
            elif action in ("withdraws","CLEAR"):codes.pop(name,None)
        return (codes[query],"ANSWER") if query in codes else ("UNKNOWN","MISSING")
    if body.startswith("DOCUMENTS:"):
        latest="latest approved date" in body.split("\n",1)[0]
        evidence=defaultdict(list)
        for record in body.split("\n",1)[1].rstrip(".").split(";"):
            name,flag,date,value=re.fullmatch(r"([A-Z]{3}) ([AD]) (?:d(\d+) )?(\d+)",record).groups()
            if flag=="A":evidence[name].append((int(date or 0),int(value)))
        records=evidence[query]
        if not records:return "UNKNOWN","MISSING"
        max_date=max(date for date,_ in records)
        values={value for date,value in records if not latest or date==max_date}
        return (str(next(iter(values))),"ANSWER") if len(values)==1 else ("UNKNOWN","CONFLICT")
    raise ValueError("unsupported prompt")


def solve_prompt(prompt):
    return parse_prompt(prompt)[0]


def score_response(response,case):
    actual=normalize_output(response)
    integer=actual is not None and re.fullmatch(r"0|-?[1-9]\d*",actual) is not None
    kind=case["answer_kind"]
    valid=actual in ("YES","NO") if kind=="choice" else (integer or actual=="UNKNOWN") if kind=="integer_or_unknown" else integer
    expected=case.get("gold",case.get("gold_answer"))
    canonical_abstention=actual=="UNKNOWN"
    return {"score":float(bool(valid and actual==expected)),"correct":bool(valid and actual==expected),
            "format_valid":bool(valid),"canonical_output":actual,
            "literal_UNKNOWN":response.strip()=="UNKNOWN" if isinstance(response,str) else False,
            "semantic_abstention_status":"CANONICAL_UNKNOWN" if canonical_abstention else "CANONICAL_NONABSTENTION" if valid else "NATURAL_LANGUAGE_NOT_ASSESSED",
            "false_UNKNOWN":canonical_abstention and expected!="UNKNOWN",
            "unsupported_canonical_answer":bool(valid and not canonical_abstention and expected=="UNKNOWN")}


def blind_predictions(prompt):
    body,question=re.split(r"\n(?:Question|Query): ",prompt)
    names=[x for x in re.findall(r"\b[A-Z]{3}\b",body) if x not in ("YES","AND","NOT","SET")]
    nums=re.findall(r"[-+]?\d+",body)
    def blind_query(name):return solve_prompt(body+"\nQuestion: "+re.sub(r"[A-Z]{3}",name,question,count=1))
    query=re.match(r"([A-Z]{3})[?.] ",question)[1]
    query_lines=[line for line in re.split(r"[;\n]",body) if re.search(r"\b"+query+r"\b",line)]
    last_matches=re.findall(r"[-+]?\d+",query_lines[-1]) if query_lines else []
    predictions={"constant_YES":"YES","constant_NO":"NO","constant_UNKNOWN":"UNKNOWN",
        "constant_0":"0","constant_1":"1","copy_full_prompt":prompt,
        "copy_first_number":nums[0] if nums else "","copy_last_number":nums[-1] if nums else "",
        "copy_last_entity":names[-1] if names else "",
        "query_blind_first_entity_solver":blind_query(names[0]),
        "query_blind_last_entity_solver":blind_query(names[-1]),
        "last_printed_query_number":last_matches[-1] if last_matches else ""}
    memory_actions=[]
    if body.startswith("MEMORY:"):
        for line in body.split("\n",1)[1].rstrip(".").split(";"):
            matched=(re.fullmatch(r"Turn (\d+) ([A-Z]{3}) (says|withdraws|quotes)(?: (\d+))?",line)
                     or re.fullmatch(r"Update (\d+) ([A-Z]{3}) (SET|CLEAR|QUOTE)(?: (\d+))?",line))
            if matched and matched[2]==query:memory_actions.append((matched[3],matched[4]))
    values=[int(v) for _,v in memory_actions if v is not None]
    predictions["memory_any_clear_else_max_skeleton"]=(
        "UNKNOWN" if any(a in ("CLEAR","withdraws") for a,_ in memory_actions)
        else str(max(values)) if values else "UNKNOWN") if memory_actions else ""
    predictions["memory_max_query_value_ignore_state"]=str(max(values)) if values else ""
    if body.startswith("RULES:"):
        predictions["rule_rhs_presence_skeleton"]="YES" if query in body.split("\nFACTS:")[0] else "NO"
    else:predictions["rule_rhs_presence_skeleton"]=""
    return predictions


def audit(rows):
    groups=defaultdict(list);failures=[];positive=[];floors=defaultdict(Counter);denominators=Counter()
    for case in rows:
        groups[case["parent_id"]].append(case)
        try:
            prediction,status=parse_prompt(case["prompt"])
            passed=prediction==case["gold"] and status==case["source_status"]
        except Exception as error:
            prediction,status,passed=None,"PARSER_ERROR",False
            failures.append({"id":case["id"],"kind":"parser","error":str(error)})
        positive.append({"id":case["id"],"prediction":prediction,"source_status":status,"correct":passed})
        if not passed:failures.append({"id":case["id"],"kind":"gold_or_status"})
        for granularity,key in (
            ("all","all"),("split",case["split"]),("category",case["category"]),
            ("family",case["generator_family"]),("generator_variant",case["generator_variant"]),
            ("difficulty",case["difficulty_stratum"]),
            ("variant",str(case["variant_id"])),
            ("category_split_level",case["category"]+"|"+case["split"]+"|"+case["difficulty_stratum"]),
        ):
            group=granularity+":"+key;denominators[group]+=1
            try:
                for name,value in blind_predictions(case["prompt"]).items():
                    floors[group][name]+=score_response(value,case)["correct"]
            except Exception as error:failures.append({"id":case["id"],"kind":"blind_parser","error":str(error)})
    counts=Counter(x["split"] for x in rows)
    if len(rows)!=288 or len({x["id"] for x in rows})!=288 or len(groups)!=144:failures.append({"kind":"size_or_unique_id"})
    if counts!={"DEVELOPMENT":192,"OPEN_VALIDATION":96}:failures.append({"kind":"split_counts"})
    if len({x["prompt"] for x in rows})!=288:failures.append({"kind":"duplicate_prompt"})
    parent_splits={key:{x["split"] for x in cases} for key,cases in groups.items()}
    if any(len(v)!=1 for v in parent_splits.values()):failures.append({"kind":"parent_split_overlap"})
    for parent,cases in groups.items():
        if len(cases)!=2 or {x["variant_id"] for x in cases}!={0,1}:failures.append({"kind":"pair", "parent":parent})
        elif cases[0]["pair_kind"]=="SAME_GOLD_PROMPT_VARIANTS" and len({x["gold"] for x in cases})!=1:
            failures.append({"kind":"same_gold_pair","parent":parent})
        elif cases[0]["pair_kind"]=="EVIDENCE_TRANSFER" and (cases[0]["gold"]=="UNKNOWN" or cases[1]["gold"]!="UNKNOWN"):
            failures.append({"kind":"evidence_transfer","parent":parent})
    family_splits=defaultdict(set)
    for case in rows:family_splits[case["generator_family"]].add(case["split"])
    variant_splits=defaultdict(set)
    for case in rows:variant_splits[case["generator_variant"]].add(case["split"])
    # Prospective gross-shortcut diagnostics, not a certificate against all attacks.
    if max(floors["all:all"].values(),default=0)>144:failures.append({"kind":"global_blind_above_half"})
    for category in CATEGORIES:
        label="category:"+category
        if max(floors[label].values(),default=0)>=48:failures.append({"kind":"category_perfect_blind","category":category})
    style_labels=Counter((x["split"],x["difficulty_stratum"],x["wording_style"],x["answerability"])
                         for x in rows if x["category"]=="answerability_source_conflict")
    metadata_ceiling={str(v):sum(x["gold"]=="UNKNOWN" for x in rows if x["category"]=="answerability_source_conflict" and x["variant_id"]==v)
                      for v in (0,1)}
    result={"status":"PASS_PREPARATION_ONLY" if not failures else "FAILED_PACKET_RETAINED_NOT_ADMITTED",
        "cases":len(rows),"world_parents":len(groups),"DEV_parent_units":sum(v=={"DEVELOPMENT"} for v in parent_splits.values()),
        "validation_parent_units":sum(v=={"OPEN_VALIDATION"} for v in parent_splits.values()),
        "split_case_counts":dict(counts),"category_case_counts":dict(Counter(x["category"] for x in rows)),
        "positive_control_correct":sum(x["correct"] for x in positive),"failures":failures,
        "floor_tables":{key:{"cases":denominators[key],"correct_counts":dict(value)} for key,value in floors.items()},
        "answerability_style_balance": [{"split":s,"difficulty":d,"style":st,"answerability":a,"cases":n}
                                       for (s,d,st,a),n in sorted(style_labels.items())],
        "metadata_known_variant_UNKNOWN_correct":metadata_ceiling,
        "metadata_floor_caution":"Evidence-transfer variant1 is allUNKNOWN; variant/IDs never enter modelprompt; styles are counterbalanced.",
        "shared_generator_families_across_splits":[key for key,value in family_splits.items() if len(value)>1],
        "named_operator_representation_variants_disjoint":all(len(v)==1 for v in variant_splits.values()),
        "max_ascii_prompt_bytes":max(len(x["prompt"].encode("ascii")) for x in rows),
        "actual_native_token_counts":"NOT_RUN;350 planned ceiling must bind without truncation",
        "independence":"Primary validation48 randomized parents; matched cases not independent; common authors/templates/families remain.",
        "human_review":"NOT_REVIEWED","native_calls":0}
    return result,positive


def run(out,nonce,seed):
    out=Path(out).resolve()
    if PACKAGE not in out.parents:raise PermissionError("exclusive campaign-local output required")
    out.mkdir(parents=True,exist_ok=False)
    rows=generate(nonce,seed)
    def write(name,value):
        with (out/name).open("x",encoding="utf-8") as stream:
            json.dump(value,stream,indent=2,sort_keys=True);stream.write("\n")
    write("bank.json",rows)  # retain every world before audit; never search/drop.
    report,positive=audit(rows)
    write("audit.json",report);write("positive_controls.json",positive)
    source=Path(__file__)
    manifest={"generator_version":VERSION,"status":report["status"],"nonce":nonce,"seed":seed,
        "source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
        "normalizer_source_sha256":hashlib.sha256((PACKAGE/"open_bank.py").read_bytes()).hexdigest(),
        "files":{name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ("bank.json","audit.json","positive_controls.json")},
        "training":False,"model_loads":0,"native_seconds":0,"tokenizers_loaded":0,
        "packet_policy":"exactlyone nonce/seed17; all failures retained; no model-outcome selection",
        "next_gate":"independent code/data review,exact tokenizer<=350/cap8 binding,newcomplete-matrix registration and sole root admission"}
    write("manifest.json",manifest)
    print(json.dumps({k:v for k,v in report.items() if k not in ("floor_tables","answerability_style_balance")},indent=2))
    return 0 if not report["failures"] else 2


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out",required=True);parser.add_argument("--nonce");parser.add_argument("--seed",type=int,default=17)
    args=parser.parse_args()
    raise SystemExit(run(args.out,args.nonce or secrets.token_hex(6),args.seed))
