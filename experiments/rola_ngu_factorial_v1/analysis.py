"""Complete paired blocks only; no pooled row-by-seed significance tests."""
from __future__ import annotations
from dataclasses import dataclass
import statistics
from collections import defaultdict

ARMS = ('A', 'B', 'C', 'D')


@dataclass(frozen=True)
class Outcome:
    draw: int
    arm: str
    sparse: bool
    row_id: str
    family: str
    source_correct: bool
    correct: bool


def contrasts(scores: dict[str, float]) -> dict[str, float]:
    a, b, c, d = (scores[x] for x in ARMS)
    return {'rola': ((c - a) + (d - b)) / 2,
            'ngu': ((b - a) + (d - c)) / 2,
            'interaction': (d - c) - (b - a)}


def summarize(records: list[Outcome], *, minimum_draws: int = 3) -> dict:
    if type(minimum_draws) is not int or minimum_draws < 3:
        raise ValueError('at least three independent draws required')
    cells = defaultdict(dict)
    identities = {}
    for r in records:
        if r.arm not in ARMS or not r.row_id or not r.family or type(r.draw) is not int:
            raise ValueError('invalid outcome identity')
        if any(type(x) is not bool for x in (r.sparse, r.correct, r.source_correct)):
            raise ValueError('boolean outcomes required')
        key = (r.draw, r.arm, r.sparse)
        if r.row_id in cells[key]:
            raise ValueError('duplicate outcome')
        identity = (r.family, r.source_correct)
        if r.row_id in identities and identities[r.row_id] != identity:
            raise ValueError('source denominator or family changed')
        identities[r.row_id] = identity
        cells[key][r.row_id] = r
    draws = sorted({key[0] for key in cells})
    if len(draws) < minimum_draws:
        raise ValueError('incomplete independent-draw block')
    has_sparse = any(key[2] for key in cells)
    expected = {(s, a, k) for s in draws for a in ARMS for k in ([False, True] if has_sparse else [False])}
    if set(cells) != expected:
        raise ValueError('incomplete factorial/sparsity block; no selective cell reporting')
    row_set = set(identities)
    if not row_set or any(set(cell) != row_set for cell in cells.values()):
        raise ValueError('evaluation row sets differ across cells/draws')
    by_draw = []
    for draw in draws:
        scores, retention, pruning = {}, {}, {}
        for arm in ARMS:
            rows = list(cells[draw, arm, False].values())
            scores[arm] = sum(r.correct for r in rows) / len(rows)
            retention[arm] = {'source_correct': sum(r.source_correct for r in rows),
                              'retained': sum(r.correct and r.source_correct for r in rows),
                              'lost': sum(not r.correct and r.source_correct for r in rows),
                              'gained': sum(r.correct and not r.source_correct for r in rows)}
            if has_sparse:
                pruning[arm] = sum(cells[draw, arm, True][i].correct - cells[draw, arm, False][i].correct
                                   for i in row_set) / len(row_set)
        family_effects = {}
        for family in sorted({x[0] for x in identities.values()}):
            ids = [i for i in row_set if identities[i][0] == family]
            fs = {a: sum(cells[draw, a, False][i].correct for i in ids) / len(ids) for a in ARMS}
            family_effects[family] = contrasts(fs)
        by_draw.append({'draw': draw, 'scores': scores, 'retention': retention,
                        'primary': contrasts(scores), 'family_effects': family_effects,
                        'paired_pruning_effects': pruning})
    aggregate = {name: {'mean': statistics.mean(r['primary'][name] for r in by_draw),
                        'sample_sd': statistics.stdev(r['primary'][name] for r in by_draw),
                        'min': min(r['primary'][name] for r in by_draw),
                        'max': max(r['primary'][name] for r in by_draw)}
                 for name in ('rola', 'ngu', 'interaction')}
    return {'evidence_class': 'DESCRIPTIVE_PAIRED_DEVELOPMENT', 'by_draw': by_draw,
            'aggregate': aggregate, 'draws': len(draws), 'rows_per_cell': len(row_set),
            'p_values': None, 'confirmation_claim': False, 'release_permitted': False}


def paired_retention(source: dict[str,bool], candidate: dict[str,bool]) -> dict:
    """Loss rates condition on source-correct rows, never all subgroup rows."""
    if not source or source.keys()!=candidate.keys() or any(type(v) is not bool for d in (source,candidate) for v in d.values()):
        raise ValueError('complete, aligned boolean endpoint records required')
    retained=sum(source[k] and candidate[k] for k in source)
    lost=sum(source[k] and not candidate[k] for k in source)
    gained=sum(not source[k] and candidate[k] for k in source)
    denominator=sum(source.values())
    return {'n':len(source),'source_correct':denominator,'candidate_correct':sum(candidate.values()),
            'retained':retained,'lost':lost,'gained':gained,'net':gained-lost,
            'loss_rate_given_source_correct':lost/denominator if denominator else None,
            'no_observed_losses':lost==0,'no_forgetting_established':False}


def abstention_endpoints(source: dict[str,bool], candidate: dict[str,bool], answerable: dict[str,bool]) -> dict:
    """Inputs denote literal UNKNOWN, not semantic abstention review."""
    if source.keys()!=candidate.keys() or source.keys()!=answerable.keys() or any(type(v) is not bool for d in (source,candidate,answerable) for v in d.values()):
        raise ValueError('aligned sentinel and answerability records required')
    unknown={k:source[k] for k in source if not answerable[k]}
    result=paired_retention(unknown,{k:candidate[k] for k in unknown})
    return {'unanswerable':result,'answerable_n':sum(answerable.values()),
            'source_false_abstentions':sum(source[k] and answerable[k] for k in source),
            'candidate_false_abstentions':sum(candidate[k] and answerable[k] for k in source),
            'semantic_review':'NOT_PERFORMED','confirmation_claim':False}
