"""288-case construction/parser/scorer apparatus; no tokenizer/model imports."""
from collections import Counter, defaultdict
from copy import deepcopy
import re
import pytest

from experiments.rineya_taps_adaptive_recurrence_campaign_v2 import bulk_bank as bank


@pytest.fixture(scope="module")
def cases():
    return bank.generate("CPU_FIXTURE_NOT_EXPORTED",17)


def test_exact_parent_case_split_counts_and_provenance(cases):
    assert len(cases)==288 and len({x["id"] for x in cases})==288
    assert len({x["parent_id"] for x in cases})==144
    assert Counter(x["split"] for x in cases)=={"DEVELOPMENT":192,"OPEN_VALIDATION":96}
    by_parent=defaultdict(list)
    for x in cases:
        by_parent[x["parent_id"]].append(x)
        assert x["id"]==x["case_id"] and x["parent_id"]==x["scenario_id"]
        assert x["gold"]==x["gold_answer"]
        assert x["provenance"]["historical_parents"]==[]
        assert x["shape"]["native_tokenizer_binding"]=="NOT_RUN"
    assert all(len(v)==2 and len({x["split"] for x in v})==1 for v in by_parent.values())
    assert sum(v[0]["split"]=="DEVELOPMENT" for v in by_parent.values())==96
    assert sum(v[0]["split"]=="OPEN_VALIDATION" for v in by_parent.values())==48


def test_difficulty_query_role_and_label_balance_before_outcomes(cases):
    for category in bank.CATEGORIES:
        selected=[x for x in cases if x["category"]==category]
        assert len(selected)==48
        for split,parent_count in (("DEVELOPMENT",16),("OPEN_VALIDATION",8)):
            unique={x["parent_id"]:x for x in selected if x["split"]==split}
            assert len(unique)==parent_count
            assert Counter(x["difficulty_stratum"] for x in unique.values())=={"easy":parent_count//2,"hard":parent_count//2}
            for level in ("easy","hard"):
                strata=[x for x in unique.values() if x["difficulty_stratum"]==level]
                assert Counter(x["query_role"] for x in strata)=={i:parent_count//8 for i in range(4)}
        if category=="deduction_composition":
            assert Counter(x["gold"] for x in selected)=={"YES":24,"NO":24}
        elif category in ("dialogue_memory","document_retrieval_reconciliation","answerability_source_conflict"):
            assert Counter(x["answerability"] for x in selected)=={"answerable":24,"unanswerable":24}
        elif category=="signed_ledgers":
            assert sum(x["gold"].startswith("-") for x in selected)==24
    branch_parents={x["parent_id"]:x for x in cases if x["category"]=="program_state" and x["split"]=="OPEN_VALIDATION" and x["difficulty_stratum"]=="hard"}
    assert Counter(x["difficulty_factors"]["branch_truth"] for x in branch_parents.values())=={True:2,False:2}


def test_independent_prompt_parser_all_gold_and_source_status(cases,monkeypatch):
    expected=[(x["prompt"],x["gold"],x["source_status"]) for x in cases]
    monkeypatch.setattr(bank,"make_world",lambda *a,**k: (_ for _ in ()).throw(AssertionError("constructor unavailable")))
    for prompt,gold,status in expected:
        assert bank.parse_prompt(prompt)==(gold,status)
    with pytest.raises(TypeError):
        bank.solve_prompt({"prompt":"ignored","gold":"SECRET"})


def test_same_gold_pairs_and_explicit_semantic_evidence_interventions(cases):
    parents=defaultdict(list)
    for x in cases:parents[x["parent_id"]].append(x)
    for pair in parents.values():
        first,second=sorted(pair,key=lambda x:x["variant_id"])
        if first["category"]!="answerability_source_conflict":
            assert first["gold"]==second["gold"] and not first["expected_response_change"]
        else:
            assert first["gold"]!="UNKNOWN" and second["gold"]=="UNKNOWN"
            assert second["source_status"] in ("MISSING","CONFLICT")
            # Bundle exchange preserves all global flag/date/value fields; only
            # the association with the queried entity changes.
            def content_without_names(prompt):
                body=re.split(r"\n(?:Question|Query): ",prompt)[0].split("\n",1)[1]
                records=body.rstrip(".").split(";")
                return sorted(re.sub(r"^[A-Z]{3} ","",x) for x in records)
            assert content_without_names(first["prompt"])==content_without_names(second["prompt"])


def test_evidence_style_label_balance_and_metadata_not_prompt(cases):
    for split in ("DEVELOPMENT","OPEN_VALIDATION"):
        for level in ("easy","hard"):
            selected=[x for x in cases if x["category"]=="answerability_source_conflict" and x["split"]==split and x["difficulty_stratum"]==level]
            counts=Counter((x["wording_style"],x["answerability"]) for x in selected)
            assert len(set(counts.values()))==1
    for x in cases:
        assert x["id"] not in x["prompt"] and x["parent_id"] not in x["prompt"]
        assert "variant" not in x["prompt"].lower() and "target" not in x["prompt"].lower()


@pytest.mark.parametrize("text",["not 123","123 then 456","YES NO","UNKNOWN.","+12","012","-0",'{"answer":12}',""])
def test_strict_scorer_no_number_choice_keyword_forgiveness(text,cases):
    assert bank.normalize_output(text) is None
    assert not bank.score_response(text,cases[0])["format_valid"]
    assert bank.score_response(text,cases[0])["semantic_abstention_status"]=="NATURAL_LANGUAGE_NOT_ASSESSED"


def test_false_unknown_and_unsupported_answer_flags(cases):
    answerable=next(x for x in cases if x["answerability"]=="answerable")
    missing=next(x for x in cases if x["answerability"]=="unanswerable")
    assert bank.score_response("UNKNOWN",answerable)["false_UNKNOWN"]
    assert bank.score_response("123",missing)["unsupported_canonical_answer"]
    assert not bank.score_response("I do not know.",missing)["correct"]
    assert bank.score_response("unknown",missing)["semantic_abstention_status"]=="CANONICAL_UNKNOWN"
    assert not bank.score_response("unknown",missing)["literal_UNKNOWN"]


def test_memory_number_order_quotes_and_withdrawal():
    head="MEMORY: follow numbered turns/updates. New declarations replace old codes; withdrawal/CLEAR removes them. QUOTE changes nothing.\n"
    prompt=head+"Turn 3 AAA quotes 777;Turn 2 AAA withdraws;Turn 1 AAA says 123.\nQuestion: AAA? Reply one integer or UNKNOWN."
    assert bank.parse_prompt(prompt)==("UNKNOWN","MISSING")
    prompt=head+"Update 3 AAA QUOTE 777;Update 2 AAA SET 456;Update 1 AAA SET 123.\nQuery: AAA. Reply one integer or UNKNOWN."
    assert bank.parse_prompt(prompt)==("456","ANSWER")


def test_latest_authority_conflict_and_draft_filtering():
    head="DOCUMENTS: use only A-approved records; latest approved date per entity. Equal approved values give that value; absent/conflicting values give UNKNOWN.\n"
    suffix=".\nQuestion: AAA? Reply one integer or UNKNOWN."
    assert bank.parse_prompt(head+"AAA A d1 123;AAA A d2 456;AAA D d3 999"+suffix)==("456","ANSWER")
    assert bank.parse_prompt(head+"AAA A d2 456;AAA A d2 457"+suffix)==("UNKNOWN","CONFLICT")
    assert bank.parse_prompt(head+"AAA D d2 456;BBB A d2 457"+suffix)==("UNKNOWN","MISSING")


def test_program_branch_interpreter_uses_live_state():
    prompt="PROGRAM: execute numbered steps in order using current registers.\nINIT: AAA=3,BBB=1,CCC=2,DDD=4.\nSTEPS: 3 IF BBB>11 THEN CCC=BBB+7 ELSE CCC=AAA+9;1 AAA=DDD*3;2 BBB=AAA+1.\nQuestion: CCC? Reply one integer."
    assert bank.parse_prompt(prompt)==("20","ANSWER")


def test_floor_tables_all_granularities_and_shared_generator_limits(cases):
    report,positive=bank.audit(cases)
    assert report["positive_control_correct"]==288 and all(x["correct"] for x in positive)
    assert not report["failures"]
    assert len(report["shared_generator_families_across_splits"])==6
    assert report["named_operator_representation_variants_disjoint"]
    assert {"all:all","split:DEVELOPMENT","difficulty:hard","variant:1"}<=report["floor_tables"].keys()
    assert "category_split_level:signed_ledgers|OPEN_VALIDATION|easy" in report["floor_tables"]
    for name,value in report["floor_tables"]["all:all"]["correct_counts"].items():
        assert value==sum(bank.score_response(bank.blind_predictions(x["prompt"])[name],x)["correct"] for x in cases)


def test_queryblind_solvers_ignore_original_requested_entity(cases):
    for case in cases:
        body,question=re.split(r"\n(?:Question|Query): ",case["prompt"])
        names=[x for x in re.findall(r"\b[A-Z]{3}\b",body) if x not in ("YES","NOT","AND","SET")]
        other=body+"\nQuestion: "+re.sub(r"[A-Z]{3}",names[-1],question,count=1)
        before,after=bank.blind_predictions(case["prompt"]),bank.blind_predictions(other)
        assert before["query_blind_first_entity_solver"]==after["query_blind_first_entity_solver"]
        assert before["query_blind_last_entity_solver"]==after["query_blind_last_entity_solver"]


def test_duplicates_and_parent_firewall_fail_and_are_not_dropped(cases):
    mutated=deepcopy(cases);mutated[-1]=deepcopy(mutated[0])
    result,_=bank.audit(mutated)
    assert result["status"]=="FAILED_PACKET_RETAINED_NOT_ADMITTED" and len(mutated)==288
    assert any(x["kind"]=="size_or_unique_id" for x in result["failures"])
    mutated=deepcopy(cases);mutated[0]["split"]="OPEN_VALIDATION"
    result,_=bank.audit(mutated)
    assert any(x["kind"]=="parent_split_overlap" for x in result["failures"])
