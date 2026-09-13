"""Typed answer scoring and the training-time retention calculation.

Why this module exists. Every construction stage in this ladder is satisfied by a model that
emits token salad, and `G1_QUALITY` -- the fix for that -- scores initialization only. Nothing
stopped an optimizer trading capability away during a run, and the measurements say no
likelihood-shaped quantity can notice: at a token-matched 500k dose a converted arm reached a
held-out CE 5.1055 nats *better* than the frozen source while answering 2 of 60 held-out
questions the source answers 20 of 60. See `amendments/HOTFIX_005.md`.

Two pure pieces live here. No torch, no model, no filesystem, no clock: everything is a
function of its arguments, so the whole thing is testable in milliseconds and the same code
scores the source, the initialization and every checkpoint.

1. A typed answer grammar. The exploration scorer
   (`experiments/rab001_exploration/scoring.py`) compares the sequence of numbers in a string.
   That is format-tolerant in the right way and type-blind in the wrong way: it accepts
   `10:25 PM` for `10:25 AM`, `5 hours` for `5 minutes` and `not 5` for `5`, and it rejects
   `60 minutes` for `1 hour`. A retention verdict may not rest on it. The grammar here parses
   the WHOLE answer into a declared type and compares exact rationals -- never a float, never
   a number scraped out of surrounding prose.

2. The retention arithmetic: per-class loss counts against an immutable reference set, and a
   one-sided exact binomial upper bound on the conditional forgetting probability.

The specification is `TYPED_SCORING_CONTRACT.md`, whose sha256 is bound into the policy block
so the prose and the code cannot drift apart unnoticed. Every row of that document's
counterexample table is a test case in `tests/test_retention.py`.

Deliberate scope limit: this grammar accepts exact answers in a bounded declared format. It
rejects valid free-form English that means the right thing ("a bit under an hour"). That is
why a source baseline measured with the legacy scorer cannot be carried over -- the source
must be re-measured under this grammar before any threshold is applied to a candidate.
"""
from __future__ import annotations

import math
import re
from fractions import Fraction

# --------------------------------------------------------------------------------------
# Answer grammar
# --------------------------------------------------------------------------------------

#: Target kinds. An unknown kind fails closed rather than defaulting to numeric comparison.
KINDS = ('number', 'quantity', 'count', 'money', 'percent',
         'distance_km', 'volume_cups', 'duration', 'clock')

#: The only sentence openers stripped before parsing. Nothing else is removed: an answer that
#: needs prose removed to become correct is outside the declared format.
PREFIXES = ('answer:', 'the answer is', 'it is', 'it will be')
CLOCK_PREFIX = 'at'

MERIDIEM_AM = ('am', 'a.m.', 'a.m')
MERIDIEM_PM = ('pm', 'p.m.', 'p.m')

MONEY_UNITS = ('dollar', 'dollars')
PERCENT_UNITS = ('%', 'percent', 'per cent')
KM_UNITS = ('km', 'kilometer', 'kilometers', 'kilometre', 'kilometres')
METRE_UNITS = ('m', 'meter', 'meters', 'metre', 'metres')
CUP_UNITS = ('cup', 'cups')

#: Canonical duration unit is the second, so a compound answer sums exactly.
DURATION_UNITS = {
    'second': 1, 'seconds': 1, 'sec': 1, 'secs': 1, 's': 1,
    'minute': 60, 'minutes': 60, 'min': 60, 'mins': 60,
    'hour': 3600, 'hours': 3600, 'hr': 3600, 'hrs': 3600, 'h': 3600,
}
#: Which canonical unit a bare duration number means, when the row permits a bare number.
DURATION_BARE_UNITS = ('second', 'minute', 'hour')

_SIGN = r'[+-]?'
_UNGROUPED = r'\d+'
_GROUPED = r'\d{1,3}(?:,\d{3})+'
_INTBODY = f'(?:{_GROUPED}|{_UNGROUPED})'
_INT_RE = re.compile(f'^{_SIGN}{_INTBODY}$')
# A trailing bare '.' is not a decimal here: the single optional sentence period is removed
# before parsing, so `3.` has already become `3` and `3..` must fail.
_DEC_RE = re.compile(f'^{_SIGN}(?:{_INTBODY}\\.\\d+|\\.\\d+)$')
# Grabs the leading numeric run so a unit suffix can be inspected separately. Deliberately
# permissive -- `parse_rational` re-validates, so `10,00` is captured and then rejected.
_LEAD_NUM_RE = re.compile(r'^[+-]?(?:\d[\d,]*(?:\.\d+)?|\.\d+)(?:\s*/\s*[+-]?\d+)?')
_CLOCK_RE = re.compile(r'^(\d{1,2}):(\d{2})$')
_BARE_HOUR_RE = re.compile(r'^(\d{1,2})$')


class ParseError(ValueError):
    """A prediction that does not parse. This is an ordinary wrong answer, not an invalid run."""


class TargetError(ValueError):
    """A malformed or unknown target specification. This invalidates the evaluation itself."""


def normalise(text: str) -> str:
    """Collapse whitespace and case. Nothing else -- no punctuation stripping happens here."""
    return ' '.join((text or '').strip().split()).casefold()


def parse_rational(token: str) -> Fraction:
    """Exact rational, or raise. Integers, finite decimals, or one integer/positive-integer
    fraction. No floats are constructed at any point.

    `Fraction` itself accepts scientific notation (`Fraction('5e0')` is 5), so the regexes
    below gate it rather than trusting the constructor.
    """
    t = token.strip()
    if not t:
        raise ParseError('empty number')
    if '/' in t:
        parts = t.split('/')
        if len(parts) != 2:
            raise ParseError(f'not a single fraction: {token!r}')
        num, den = parts[0].strip(), parts[1].strip()
        if not _INT_RE.match(num) or not _INT_RE.match(den):
            raise ParseError(f'fraction parts must be integers: {token!r}')
        denominator = int(den.replace(',', ''))
        if denominator <= 0:
            # Covers `1/0` and a negated denominator, which the contract does not admit.
            raise ParseError(f'non-positive denominator: {token!r}')
        return Fraction(int(num.replace(',', '')), denominator)
    if _INT_RE.match(t) or _DEC_RE.match(t):
        return Fraction(t.replace(',', ''))
    raise ParseError(f'not an exact rational: {token!r}')


def _strip_one_period(text: str) -> str:
    """Remove at most one closing sentence period. `5...` keeps two and will fail to parse."""
    return text[:-1] if text.endswith('.') else text


def _strip_prefix(text: str, allow_at: bool) -> str:
    candidates = list(PREFIXES) + ([CLOCK_PREFIX] if allow_at else [])
    for prefix in sorted(candidates, key=len, reverse=True):
        if text == prefix:
            raise ParseError('prefix with no answer')
        if text.startswith(prefix + ' '):
            return text[len(prefix) + 1:].strip()
        if prefix.endswith(':') and text.startswith(prefix):
            return text[len(prefix):].strip()
    return text


def _split_number(text: str) -> tuple[Fraction, str]:
    """Leading exact rational plus the untouched remainder."""
    match = _LEAD_NUM_RE.match(text)
    if not match:
        raise ParseError(f'no leading number in {text!r}')
    return parse_rational(match.group(0)), text[match.end():].strip()


def _require_empty(rest: str) -> None:
    if rest:
        raise ParseError(f'unconsumed text {rest!r}')


def _aliases(target: dict, key: str) -> tuple[str, ...]:
    return tuple(normalise(a) for a in target.get(key) or ())


def _bare_allowed(target: dict) -> bool:
    return bool(target['bare_allowed'])


def _target_value(target: dict) -> Fraction:
    return parse_rational(str(target['value']))


#: Keys every target may carry, plus the ones each kind requires. Validation is a whitelist so
#: a typo (`bare_allow`) is a loud TargetError rather than a silently missing permission.
_COMMON_KEYS = {'kind', 'value', 'bare_allowed', 'qualifier'}
_KIND_KEYS = {
    'number': set(), 'money': set(), 'percent': set(), 'distance_km': set(),
    'volume_cups': set(), 'quantity': {'unit'}, 'count': {'unit_aliases'},
    'duration': {'bare_unit'}, 'clock': {'minute_of_day'},
}


def validate_target(target: dict) -> None:
    """Reject a malformed target specification before any parsing happens.

    This runs unconditionally rather than lazily inside the parse, because lazy validation
    fails open: a `quantity` target with no `unit` was scored as an ordinary wrong answer
    whenever the prediction happened to be a bare number, so a broken specification would have
    been recorded as a capability loss. A bad target invalidates the evaluation; it is never a
    model failure.
    """
    if not isinstance(target, dict):
        raise TargetError('target must be a mapping')
    kind = target.get('kind')
    if kind not in KINDS:
        raise TargetError(f'unknown target kind: {kind!r}')
    allowed = _COMMON_KEYS | _KIND_KEYS[kind]
    unknown = set(target) - allowed
    if unknown:
        raise TargetError(f'unknown target keys for {kind}: {sorted(unknown)}')
    qualifier = target.get('qualifier')
    if qualifier is not None and (not isinstance(qualifier, str) or not qualifier.strip()):
        raise TargetError('qualifier must be a non-empty string when present')

    if kind == 'clock':
        minute = target.get('minute_of_day')
        if isinstance(minute, bool) or not isinstance(minute, int) or not 0 <= minute <= 1439:
            raise TargetError('clock target needs an integer minute_of_day in 0..1439')
        if 'value' in target or 'bare_allowed' in target:
            raise TargetError('clock targets carry minute_of_day, not value/bare_allowed')
        return

    if 'value' not in target:
        raise TargetError('target has no value')
    try:
        value = parse_rational(str(target['value']))
    except ParseError as exc:
        raise TargetError(f'target value is not an exact rational: {target["value"]!r}') from exc
    if not isinstance(target.get('bare_allowed'), bool):
        raise TargetError('bare_allowed must be an explicit boolean')

    if kind == 'count':
        if value < 0 or value.denominator != 1:
            raise TargetError('a count target must be a non-negative integer')
        aliases = target.get('unit_aliases')
        if (not isinstance(aliases, (list, tuple)) or not aliases
                or not all(isinstance(a, str) and a.strip() for a in aliases)):
            raise TargetError('count target needs a non-empty list of unit_aliases')
    elif kind == 'quantity':
        unit = target.get('unit')
        if not isinstance(unit, str) or not unit.strip():
            raise TargetError('quantity target needs a non-empty unit')
    elif kind == 'duration':
        if value < 0:
            raise TargetError('a duration target must be non-negative')
        if target['bare_allowed'] and target.get('bare_unit') not in DURATION_BARE_UNITS:
            raise TargetError(f'bare_unit must be one of {DURATION_BARE_UNITS}')
        if not target['bare_allowed'] and 'bare_unit' in target:
            raise TargetError('bare_unit is meaningless when a bare value is not permitted')


def _match_qualified_noun(rest: str, target: dict) -> None:
    """`<qualifier>? <noun-alias>` for a counted object.

    Omitting the question's qualifier is accepted; naming a different one is not. The
    qualifier must precede the whole alias, which is the only position the contract declares.
    """
    aliases = _aliases(target, 'unit_aliases')
    qualifier = target.get('qualifier')
    if qualifier:
        qualifier = normalise(qualifier)
        if rest.startswith(qualifier + ' '):
            rest = rest[len(qualifier) + 1:].strip()
    if rest not in aliases:
        raise ParseError(f'noun {rest!r} is not a declared alias for this target')


def _parse_duration_seconds(rest: str, target: dict) -> Fraction:
    """One quantity/unit, or a compound of DISTINCT units summed exactly."""
    text = rest.replace(',', ' ')
    tokens = [t for t in text.split() if t and t != 'and']
    if not tokens:
        raise ParseError('empty duration')
    if len(tokens) % 2:
        raise ParseError(f'duration is not number/unit pairs: {rest!r}')
    total, seen = Fraction(0), set()
    for number, unit in zip(tokens[0::2], tokens[1::2]):
        if unit not in DURATION_UNITS:
            raise ParseError(f'unknown duration unit {unit!r}')
        scale = DURATION_UNITS[unit]
        if scale in seen:
            raise ParseError(f'repeated duration unit {unit!r}')
        seen.add(scale)
        value = parse_rational(number)
        if value < 0:
            raise ParseError('negative duration')
        total += value * scale
    return total


def _parse_clock_minutes(text: str) -> int:
    """Minutes since midnight in a 24-hour day.

    A missing meridiem is read as 24-hour time, never guessed: `4:20` is 04:20, so it does not
    match a 4:20 PM target. `13:20 pm`, `24:00` and a naked hour without a meridiem all fail.
    """
    if text == 'midnight':
        return 0
    if text == 'noon':
        return 12 * 60
    meridiem = None
    for suffix in MERIDIEM_AM + MERIDIEM_PM:
        if text.endswith(' ' + suffix):
            meridiem = 'am' if suffix in MERIDIEM_AM else 'pm'
            text = text[: -(len(suffix) + 1)].strip()
            break
    match = _CLOCK_RE.match(text)
    if match:
        hour, minute = int(match.group(1)), int(match.group(2))
    elif meridiem and _BARE_HOUR_RE.match(text):
        # An explicitly suffixed whole hour: `7 PM`. A naked `7` stays rejected.
        hour, minute = int(text), 0
    else:
        raise ParseError(f'not a clock time: {text!r}')
    if not 0 <= minute <= 59:
        raise ParseError('minute out of range')
    if meridiem is None:
        if not 0 <= hour <= 23:
            raise ParseError('24-hour clock out of range')
        return hour * 60 + minute
    if not 1 <= hour <= 12:
        raise ParseError('12-hour clock out of range')
    if meridiem == 'am':
        hour = 0 if hour == 12 else hour
    else:
        hour = 12 if hour == 12 else hour + 12
    return hour * 60 + minute


def parse_prediction(text: str, target: dict) -> Fraction | int:
    """Parse a prediction under the target's declared kind. Raises ParseError on any mismatch.

    The prediction is parsed AS the target's kind rather than sniffed, so a value that happens
    to share digits with the target but carries a different type or unit cannot pass.
    """
    validate_target(target)
    kind = target['kind']
    body = _strip_prefix(_strip_one_period(normalise(text)), allow_at=(kind == 'clock'))
    if not body:
        raise ParseError('empty prediction')

    if kind == 'clock':
        return _parse_clock_minutes(body)

    if kind == 'money' and body.startswith('$'):
        value, rest = _split_number(body[1:].strip())
        _require_empty(rest)
        return value

    if kind == 'duration':
        try:
            value, rest = _split_number(body)
        except ParseError:
            raise
        if not rest:
            if not _bare_allowed(target):
                raise ParseError('bare duration is ambiguous and not permitted for this row')
            unit = target['bare_unit']
            if value < 0:
                raise ParseError('negative duration')
            return value * DURATION_UNITS[unit]
        return _parse_duration_seconds(body, target)

    value, rest = _split_number(body)

    if kind == 'number':
        # Intrinsically unitless: any suffix at all is a type disagreement.
        _require_empty(rest)
        return value

    if not rest:
        if not _bare_allowed(target):
            raise ParseError('a bare value is not meaningful for this row')
        return value

    if kind == 'quantity':
        unit = normalise(target['unit'])
        if rest not in (unit, unit + 's'):
            raise ParseError(f'unit {rest!r} does not match {unit!r}')
        return value
    if kind == 'count':
        _match_qualified_noun(rest, target)
        return value
    if kind == 'money':
        if rest not in MONEY_UNITS:
            raise ParseError(f'{rest!r} is not a declared money unit')
        return value
    if kind == 'percent':
        if rest not in PERCENT_UNITS:
            raise ParseError(f'{rest!r} is not a percent unit')
        return value
    if kind == 'distance_km':
        if rest in KM_UNITS:
            return value
        if rest in METRE_UNITS:
            return value / 1000          # exact: Fraction division, never a float
        raise ParseError(f'{rest!r} is not a declared distance unit')
    if kind == 'volume_cups':
        qualifier = target.get('qualifier')
        if qualifier:
            suffix = ' of ' + normalise(qualifier)
            if rest.endswith(suffix):
                rest = rest[: -len(suffix)].strip()
            elif ' of ' in rest:
                raise ParseError('explicit volume qualifier disagrees with the question')
        if rest not in CUP_UNITS:
            raise ParseError(f'{rest!r} is not a cup unit')
        return value
    raise TargetError(f'unhandled kind {kind!r}')          # pragma: no cover - KINDS guard


def score_prediction(text: str, terminated: bool, target: dict) -> dict:
    """Full scoring record for one row.

    `correct` requires BOTH EOS termination and an exact typed match. A caller cannot supply
    correctness -- every consumer recomputes from this, which is why the raw text and the
    parse reason are returned alongside the verdict rather than being dropped.
    """
    validate_target(target)
    kind = target['kind']
    want = target['minute_of_day'] if kind == 'clock' else _target_value(target)
    record = {'raw': text, 'terminated': bool(terminated), 'kind': kind,
              'target': str(want), 'parsed': None, 'reason': None, 'correct': False}
    if not terminated:
        record['reason'] = 'did not terminate'
        return record
    try:
        got = parse_prediction(text, target)
    except ParseError as exc:
        record['reason'] = str(exc)
        return record
    record['parsed'] = str(got)
    record['correct'] = (got == want)
    if not record['correct']:
        record['reason'] = 'value mismatch'
    return record


# --------------------------------------------------------------------------------------
# Target overlay: the reviewed label inventory, adapted to the scorer's schema
# --------------------------------------------------------------------------------------

#: `evidence/retention-design-v1/PROBE_LABEL_AUDIT.json` was specified independently of this
#: scorer and uses its own vocabulary: `type` not `kind`, a nested {numerator, denominator}
#: value, `minutes_since_midnight` not `minute_of_day`, and no bare-value permission at all.
#: The two schemas are reconciled here, once and in the open, rather than by hand-editing 150
#: targets -- and the mismatch is the reason this adapter exists at all.
AUDIT_KINDS = {'clock': 'clock', 'count': 'count', 'distance': 'distance_km',
               'duration': 'duration', 'money': 'money', 'number': 'number',
               'percent': 'percent', 'quantity': 'quantity', 'volume': 'volume_cups'}

#: Noun aliases per counted entity, transcribed from TYPED_SCORING_CONTRACT.md. Frozen before
#: any source or candidate measurement; new aliases must not be invented from model outputs.
COUNT_ALIASES = {
    'sensor tags': ('sensor tag', 'sensor tags', 'tag', 'tags'),
    'timing chips': ('timing chip', 'timing chips', 'chip', 'chips'),
    'shipping labels': ('shipping label', 'shipping labels', 'label', 'labels'),
    'folding chairs': ('folding chair', 'folding chairs', 'chair', 'chairs'),
    'drill bits': ('drill bit', 'drill bits', 'bit', 'bits'),
    'page': ('page', 'pages'),
    'access badges': ('access badge', 'access badges', 'badge', 'badges'),
    'safety helmets': ('safety helmet', 'safety helmets', 'helmet', 'helmets'),
}

#: The only duration prompts that name the requested unit, so the only ones where a bare number
#: identifies a value. For the other 16, `90` cannot distinguish minutes from hours. Fixed by
#: the contract, not by the bare policy below.
DURATION_BARE_ROW_SUFFIXES = ('time_duration_s04__01221',
                              'time_duration_s07__02229',
                              'time_duration_s07__02318')

BARE_POLICIES = ('permissive', 'strict')


def target_from_audit(row: dict, bare_policy: str) -> dict:
    """Build one scorer target from one audited label row.

    `bare_policy` is the single reviewable axis: whether a bare number is accepted where the
    prompt arguably fixes the unit. The audit deliberately left this pending, so it is a named
    parameter rather than a silent default, and both settings are measurable.
    """
    if bare_policy not in BARE_POLICIES:
        raise TargetError(f'bare_policy must be one of {BARE_POLICIES}')
    audit = row['proposed_typed_target']
    kind = AUDIT_KINDS.get(audit.get('type'))
    if kind is None:
        raise TargetError(f'unmapped audit target type: {audit.get("type")!r}')

    if kind == 'clock':
        target = {'kind': 'clock', 'minute_of_day': audit['minutes_since_midnight']}
        validate_target(target)
        return target

    if kind == 'duration':
        # The audit carries both the display unit and an exact seconds field; seconds is the
        # canonical representation, so no unit conversion happens here.
        bare = row['id'].endswith(DURATION_BARE_ROW_SUFFIXES)
        target = {'kind': 'duration', 'value': audit['seconds']['rational'],
                  'bare_allowed': bare}
        if bare:
            target['bare_unit'] = 'minute'
        validate_target(target)
        return target

    bare = (bare_policy == 'permissive')
    target = {'kind': kind, 'value': audit['value']['rational'],
              # A unitless number has no unit to withhold, so a bare value is its only form.
              'bare_allowed': True if kind == 'number' else bare}
    if kind == 'quantity':
        target['unit'] = audit['unit']
    elif kind == 'count':
        entity = audit['unit']
        if entity not in COUNT_ALIASES:
            raise TargetError(f'no frozen alias list for counted entity {entity!r}')
        target['unit_aliases'] = list(COUNT_ALIASES[entity])
    if audit.get('qualifier') and kind in ('count', 'volume_cups'):
        target['qualifier'] = audit['qualifier']
    validate_target(target)
    return target


def build_overlay(audit: dict, bare_policy: str) -> dict:
    """Every audited row converted to a scorer target, keyed by row id.

    Returns targets plus the provenance a receipt has to bind: which audit produced them, which
    reviewable policy was applied, and that human review of the labels is still pending.
    """
    rows = audit['rows']
    targets = {}
    for row in rows:
        if row['id'] in targets:
            raise TargetError(f'duplicate row id in audit: {row["id"]}')
        targets[row['id']] = target_from_audit(row, bare_policy)
    return {'schema': 1, 'bare_policy': bare_policy, 'targets': targets,
            'row_count': len(targets),
            'audit_row_count': audit['row_count'],
            'audit_source_sha256': audit['source_files'][0]['sha256'],
            'numeric_label_mismatch_count': audit['numeric_mismatch_count'],
            'unit_or_entity_label_mismatch_count': audit['unit_or_entity_mismatch_count'],
            'label_human_review_status': audit['human_review_status'],
            'duration_bare_rows': list(DURATION_BARE_ROW_SUFFIXES)}


# --------------------------------------------------------------------------------------
# Retention arithmetic
# --------------------------------------------------------------------------------------

def _log_binom_cdf(k: int, n: int, p: float) -> float:
    """log P(X <= k) for X ~ Binomial(n, p), summed in log space so small tails survive."""
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 0.0 if k >= n else -math.inf
    terms = []
    for i in range(k + 1):
        terms.append(math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
                     + i * math.log(p) + (n - i) * math.log1p(-p))
    top = max(terms)
    if top == -math.inf:
        return -math.inf
    return top + math.log(sum(math.exp(t - top) for t in terms))


def exact_binomial_upper_bound(losses: int, n: int, alpha: float) -> float:
    """One-sided exact (Clopper-Pearson) upper bound on the loss probability.

    Solves P(X <= losses | n, p) = alpha for p. At losses == 0 this is the closed form
    1 - alpha**(1/n); the solver is checked against that in the tests rather than trusted.
    """
    if n <= 0:
        raise ValueError('n must be positive')
    if not 0 <= losses <= n:
        raise ValueError('losses must be in 0..n')
    if not 0.0 < alpha < 1.0:
        raise ValueError('alpha must be in (0, 1)')
    if losses == n:
        return 1.0
    target = math.log(alpha)
    lo, hi = 0.0, 1.0
    for _ in range(200):                       # bisection to ~1e-60; the CDF is monotone in p
        mid = (lo + hi) / 2
        if _log_binom_cdf(losses, n, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def monitor_counts(reference_ids, candidate_correct_ids) -> dict:
    """Retained / lost counts against an immutable reference set of item ids.

    A gain cannot offset a loss, so gains are reported but never subtracted. `reference_ids`
    is fixed before training and must not be recomputed from the candidate.
    """
    reference = set(reference_ids)
    correct = set(candidate_correct_ids)
    retained = reference & correct
    return {'reference_n': len(reference), 'retained': len(retained),
            'lost': len(reference - correct), 'gained': len(correct - reference),
            'lost_ids': sorted(reference - correct)}


def monitor_verdict(per_class: dict, policy: dict) -> dict:
    """Apply the fixed monitor rules to already-computed per-class counts.

    `per_class[cls]` supplies reference_n, lost, source_correct_terminated and
    termination_rate. Every rule is a policy constant; none is computed from the candidate.
    """
    max_lost = policy['max_correct_to_incorrect_per_class']
    min_source = policy['min_source_correct_terminated_per_class']
    floor = policy['min_termination_rate_per_class']
    min_term = Fraction(floor['numerator'], floor['denominator'])
    classes, insufficient, failed = {}, [], []
    for name, counts in sorted(per_class.items()):
        rate = Fraction(counts['terminated'], counts['rows']) if counts['rows'] else Fraction(0)
        informative = counts['source_correct_terminated'] >= min_source
        within = counts['lost'] <= max_lost
        terminates = rate >= min_term
        classes[name] = {**counts, 'termination_rate': str(rate),
                         'source_informative': informative,
                         'losses_within_limit': within, 'terminates': terminates}
        if not informative:
            insufficient.append(name)
        if not within or not terminates:
            failed.append(name)
    if insufficient:
        status = 'INSUFFICIENT'
    elif failed:
        status = 'FAIL'
    else:
        status = 'PASS'
    return {'status': status, 'classes': classes,
            'failed_classes': sorted(failed), 'uninformative_classes': sorted(insufficient),
            'permits_next_block': status == 'PASS'}


def confirmation_verdict(per_class_seed: dict, policy: dict) -> dict:
    """Terminal confirmation: an exact binomial bound per seed and class, all required to pass.

    `per_class_seed[(seed, cls)]` supplies source_successes and losses. Bounds are computed at
    the registered per-statement alpha; results are never pooled across seeds, because the same
    families are reused and repeated families are not independent observations.
    """
    alpha_spec = policy['alpha_per_statement']
    alpha = alpha_spec['numerator'] / alpha_spec['denominator']
    cap_spec = policy['max_forgetting_probability_upper_bound']
    cap = Fraction(cap_spec['numerator'], cap_spec['denominator'])
    min_m = policy['min_source_successes_per_class']
    retention_spec = policy['min_retention']
    min_retention = Fraction(retention_spec['numerator'], retention_spec['denominator'])
    statements, insufficient, failed = {}, [], []
    for key in sorted(per_class_seed, key=lambda k: (str(k[0]), str(k[1]))):
        counts = per_class_seed[key]
        m, lost = counts['source_successes'], counts['losses']
        label = f'seed{key[0]}:{key[1]}'
        if m < min_m:
            statements[label] = {**counts, 'status': 'INSUFFICIENT',
                                 'reason': f'{m} source successes below the registered {min_m}'}
            insufficient.append(label)
            continue
        bound = exact_binomial_upper_bound(lost, m, alpha)
        retention = Fraction(m - lost, m)
        ok = bound <= float(cap) and retention >= min_retention
        statements[label] = {**counts, 'forgetting_upper_bound': bound,
                             'retention': str(retention),
                             'status': 'PASS' if ok else 'FAIL'}
        if not ok:
            failed.append(label)
    if len(statements) != policy['simultaneous_statement_count']:
        # The alpha allocation buys simultaneous coverage for exactly the registered number of
        # statements. A different count is a different multiplicity problem, not a verdict.
        return {'status': 'INVALID', 'statements': statements,
                'reason': f'{len(statements)} statements against a registered '
                          f'{policy["simultaneous_statement_count"]}'}
    if insufficient:
        status = 'INSUFFICIENT'
    elif failed:
        status = 'FAIL'
    else:
        status = 'PASS'
    return {'status': status, 'statements': statements,
            'failed_statements': sorted(failed),
            'insufficient_statements': sorted(insufficient),
            'alpha_per_statement': alpha,
            'family_wise_alpha': policy['family_wise_alpha']}
