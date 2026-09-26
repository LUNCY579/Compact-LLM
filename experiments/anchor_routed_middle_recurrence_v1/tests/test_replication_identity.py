from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from audit_budget_replication import LOGIT_FIELDS, compare_case


def row(ids,termination="eos"):
    return {"arm":"A1_dev","case_id":"a","score":1.,"token_ids":ids,
            "termination":termination,**{k:0. for k in LOGIT_FIELDS}}


def test_non_censored_output_must_match_entire_sequence():
    assert compare_case(row([1,2]),row([1,2]))["token_identity"]
    assert not compare_case(row([1,2]),row([1,2,3]))["token_identity"]


def test_timed_out_prefix_must_be_preserved():
    assert compare_case(row([1,2],"timeout"),row([1,2,3]))["token_identity"]
    assert not compare_case(row([1,2],"timeout"),row([1,3,4]))["token_identity"]


def test_logit_drift_is_not_hidden_by_output_identity():
    old=row([1,2]); new={**old,"logit_cosine":.01}
    result=compare_case(old,new)
    assert result["token_identity"] and not result["logit_metric_identity"]
