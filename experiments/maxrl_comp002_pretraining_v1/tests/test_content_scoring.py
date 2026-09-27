from __future__ import annotations

import pytest

from experiments.maxrl_comp002_pretraining_v1.content_scoring import score


SPEC = {"initial": [1, 2, 3, 4, 5, 6], "lookup": {"requested": ["x"], "links": {"x": 1}}}
IMPOSSIBLE = {"initial": [1, 2, 3, 4, 5, 6], "lookup": {"requested": ["x"],
              "links": {"x": "y", "y": "x"}}}


@pytest.mark.parametrize("response,category", [
    ('{"answer":[1]}', "answer_object"),
    ('[1]', "bare_list"),
    ('{"answer":[1]}.', "trailing_period+answer_object"),
    ('```json\n{"answer":[1]}\n```', "code_fence+answer_object"),
    ('{"answer":[1],"reason":"ok"}', "extra_object_fields+answer_object"),
])
def test_one_correct_certificate_under_clean_wrappers(response, category):
    result = score(SPEC, response)
    assert result["content_pass"] and result["normalization"] == category
    assert result["strict_pass"] == (response == '{"answer":[1]}')


def test_unknown_is_checked_against_feasibility():
    assert score(IMPOSSIBLE, "UNKNOWN")["content_pass"]
    assert not score(SPEC, "UNKNOWN")["content_pass"]


@pytest.mark.parametrize("response", [
    'Here is the answer: [1]', '[1] or [2]', '{"answer":[1],"answer":[2]}',
    '{"answer":[1],"x":NaN}', '{"answer":[true]}', '{"answer":[2]}',
    '```json\n{"answer":[1]}\n``` extra',
])
def test_wrong_or_ambiguous_payload_never_passes_content(response):
    result = score(SPEC, response)
    assert not result["content_pass"] and not result["format_only_failure"]
