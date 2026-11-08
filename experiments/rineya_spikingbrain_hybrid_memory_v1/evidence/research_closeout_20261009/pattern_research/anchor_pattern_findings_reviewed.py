"""Additive reviewed interpretation; preserve all earlier files and results."""
import json
from pathlib import Path

from anchor_pattern_findings import ROOT, findings, sha, digest


def reviewed_findings():
    result = findings()
    result['observations']['field_binding_on_absence']['interpretation'] = (
        'All absent outputs match visible facts, chiefly the requested subject with '
        'the wrong field. This is consistent with insufficient subject-and-field '
        'binding; it does not establish the internal retrieval algorithm.')
    result['observations']['revision_key_vs_payload']['interpretation'] = (
        'On comparison-key conflict rows, 85 outputs match the greatest code value '
        'and 66 match the payload of the greatest revision. Forty-three outputs '
        'match revision numerals rather than code payloads. These observations '
        'support competing selection/return-field hypotheses, not a proven '
        'payload-comparison algorithm.')
    result['reviewed_interpretation'] = {
        'change': 'Soften two causal interpretations; all raw counts, extraction rules and original correctness metrics remain unchanged.',
        'preserved_prior_findings': {'path': str(ROOT/'anchor-pattern-findings.json'),
                                    'sha256': sha(ROOT/'anchor-pattern-findings.json')},
        'independent_reproduction': {'path': str(ROOT/'anchor-patterns-independent.json'),
                                     'sha256': sha(ROOT/'anchor-patterns-independent.json')},
        'independent_reproduction_results_sha256': json.loads((ROOT/'anchor-patterns-independent.json').read_text())['results_sha256'],
        'causal_mechanism_established': False,
        'observed_gain_from_intervention': False,
    }
    assert result['reviewed_interpretation']['independent_reproduction_results_sha256'] == result['input_receipt']['results_sha256']
    result['reviewed_findings_analyzer_sha256'] = sha(__file__)
    return result


if __name__=='__main__':
    output = ROOT/'anchor-pattern-findings-reviewed.json'
    value = reviewed_findings()
    with output.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status':value['status'],'output':str(output),
        'observations_sha256':digest(value['observations'])},sort_keys=True))
