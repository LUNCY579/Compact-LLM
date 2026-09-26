from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from run_fixed_recurrence import development_result


def row(case, family, score):
    return {"case_id":case,"family":family,"score":score,"invalid":False,
            "safety_stopped":False,"logit_cosine":.99,"kl_base_to_arm":.1,"wall_seconds":1.}


def test_neutral_fixed_does_not_license_router():
    rows=[row("a","qa",1),row("b","math",0)]
    r=development_result(rows,rows)
    assert r["preservation_pass"]
    assert not r["router_scaffold_licensed"]


def test_global_tie_with_family_regression_blocks_router():
    base=[row("a","qa",1),row("b","math",0)]
    value=[row("a","qa",0),row("b","math",1)]
    r=development_result(value,base)
    assert r["checks"]["overall_no_loss"]
    assert not r["preservation_pass"] and not r["router_scaffold_licensed"]


def test_single_paired_correction_with_preservation_licenses_scaffold_only():
    base=[row("a","qa",1),row("b","math",0)]
    value=[row("a","qa",1),row("b","math",1)]
    r=development_result(value,base)
    assert r["router_scaffold_licensed"] and r["wins"]==["b"] and not r["losses"]


def test_timed_out_baseline_cannot_license_an_apparent_gain():
    base=[{**row("a","qa",0),"safety_stopped":True}]
    value=[row("a","qa",1)]
    r=development_result(value,base)
    assert not r["comparison_valid"]
    assert not r["preservation_pass"] and not r["router_scaffold_licensed"]


def test_empty_evaluation_cannot_pass_preservation():
    assert not development_result([],[])["preservation_pass"]
