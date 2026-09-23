"""Value-presence extractors, built only from retention.py's own parsers (attempt 2, fresh).

Every function takes the scorer module R, a raw text and a scorer target dict.

LOOSE  (primary)  some span of the text reads, under the target's kind, as EXACTLY the target
                  value; units and surrounding prose are ignored.
    numeric kinds: every maximal number atom read by R.parse_rational (digits that belong to an
                   h:mm clock are excluded), plus every `a / b` read as one rational; for
                   distance_km also `<n> m|meter|metre(s)` read as n/1000 (the scorer's rule).
    clock:         `h:mm [meridiem]`, `h meridiem`, `noon`, `midnight`, each read by
                   R._parse_clock_minutes (so `10:45` with no meridiem is 24-hour 10:45).
    duration:      MAXIMAL runs of `<n> <duration-unit>` pairs joined by a separator the scorer
                   itself admits (R._DUR_SEP_RE), summed by R._parse_duration_seconds. A sub-run
                   of a compound is NOT a separate value (`2 hours 15 minutes` is 8100 s only).
                   On the 3 rows where the scorer permits a bare minute, a bare number x60.
TYPED  (strictest) some whitespace-token span, optionally with edge punctuation stripped, is
                  ACCEPTED by R.parse_prediction under the given target and equals it -- i.e. the
                  scorer would have marked it correct had everything else been deleted.
FINAL             the LAST maximal LOOSE value (by position) equals the target.
Sensitivities (never in the primary):
    CARRY         a clock `h:mm` with mm >= 60, normalised by carrying into the hour, equals the
                  target (e.g. `9:70 am` -> 10:10 am). The scorer rejects these as out of range.
    BARE_DISPLAY  on a duration row where a bare number is NOT permitted, a bare number times
                  the audit's display unit equals the target.
"""
import re
from fractions import Fraction

NUMERIC_KINDS = ('number', 'quantity', 'count', 'money', 'percent', 'distance_km', 'volume_cups')

_NUM = r'(?:\d[\d,]*(?:\.\d+)?|\.\d+)'
ATOM_RE = re.compile(r'(?<![\d.,:/])(' + _NUM + r')(?![\d]|:\d)')
FRAC_RE = re.compile(r'(?<![\d.,:/])(\d[\d,]*)\s*/\s*(\d[\d,]*)(?![\d.]|:\d)')
METRE_RE = re.compile(r'(?<![\d.,:/])(' + _NUM + r')\s*(m|meter|meters|metre|metres)(?![a-z])')
MER = r'(a\.m\.?|p\.m\.?|am|pm)'
CLOCK_RE = re.compile(r'(?<![\d:.])(\d{1,2}):(\d{2})(?![\d])(?:\s*' + MER + r'(?![a-z]))?')
HOURMER_RE = re.compile(r'(?<![\d:.])(\d{1,2})\s*' + MER + r'(?![a-z])')
WORDCLOCK_RE = re.compile(r'(?<![a-z])(noon|midnight)(?![a-z])')
DUR_PAIR_RE = re.compile(r'(?<![\d.,:/])(' + _NUM + r'(?:\s*/\s*\d+)?)\s*-?\s*([a-z]+)')


def want_of(t, R):
    return t['minute_of_day'] if t['kind'] == 'clock' else R.parse_rational(str(t['value']))


def _rat(R, s):
    try:
        return R.parse_rational(s.strip())
    except R.ParseError:
        return None


def _numeric_values(body, t, R):
    """(start, end, value, maximal) for every number the text states."""
    out = []
    frac_spans = []
    for m in FRAC_RE.finditer(body):
        v = _rat(R, f'{m.group(1)}/{m.group(2)}')
        if v is not None:
            out.append((m.start(), m.end(), v, True))
            frac_spans.append((m.start(), m.end()))
    for m in ATOM_RE.finditer(body):
        tok = m.group(1)
        inside = any(a <= m.start() and m.end() <= b for a, b in frac_spans)
        v = _rat(R, tok)
        if v is not None:
            out.append((m.start(), m.end(), v, not inside))
        elif ',' in tok:          # `10,00` is not a grouped number; read the pieces
            for part in tok.split(','):
                pv = _rat(R, part)
                if pv is not None:
                    out.append((m.start(), m.end(), pv, not inside))
    if t['kind'] == 'distance_km':
        for m in METRE_RE.finditer(body):
            v = _rat(R, m.group(1))
            if v is not None:
                out.append((m.start(), m.end(), v / 1000, True))
    return out


def _clock_values(body, R, carry=False):
    out = []
    for m in CLOCK_RE.finditer(body):
        h, mm, mer = int(m.group(1)), int(m.group(2)), m.group(3)
        if carry:
            if mm < 60:
                continue
            base = f'{h}:00' + (f' {mer}' if mer else '')
            try:          # the hour is read by the scorer's parser; the overflow is carried
                out.append((m.start(), m.end(),
                            (R._parse_clock_minutes(base) + mm) % 1440, True))
            except R.ParseError:
                pass
            continue
        s = f'{h}:{mm:02d}' + (f' {mer}' if mer else '')
        try:
            out.append((m.start(), m.end(), R._parse_clock_minutes(s), True))
        except R.ParseError:
            pass
    if carry:
        return out
    for m in HOURMER_RE.finditer(body):
        try:
            out.append((m.start(), m.end(),
                        R._parse_clock_minutes(f'{m.group(1)} {m.group(2)}'), True))
        except R.ParseError:
            pass
    for m in WORDCLOCK_RE.finditer(body):
        out.append((m.start(), m.end(), R._parse_clock_minutes(m.group(1)), True))
    return out


def _duration_values(body, t, R, bare_scale=None):
    out = []
    pairs = [m for m in DUR_PAIR_RE.finditer(body) if m.group(2) in R.DURATION_UNITS]
    groups, cur = [], []
    for m in pairs:
        if cur:
            gap = body[cur[-1].end():m.start()]
            sep = R._DUR_SEP_RE.fullmatch(gap)
            if sep is not None and gap:
                cur.append(m)
                continue
            groups.append(cur)
        cur = [m]
    if cur:
        groups.append(cur)
    for g in groups:
        s = ' '.join(f'{m.group(1)} {m.group(2)}' for m in g)
        try:
            out.append((g[0].start(), g[-1].end(), R._parse_duration_seconds(s, t), True))
        except R.ParseError:
            for m in g:           # repeated unit etc.: read each pair on its own
                try:
                    out.append((m.start(), m.end(),
                                R._parse_duration_seconds(f'{m.group(1)} {m.group(2)}', t), True))
                except R.ParseError:
                    pass
    if bare_scale is not None:
        for m in ATOM_RE.finditer(body):
            tail = body[m.end():].lstrip()
            if re.match(r'[a-z%$]', tail) or body[max(0, m.start() - 1):m.start()] == '$':
                continue
            v = _rat(R, m.group(1))
            if v is not None:
                out.append((m.start(), m.end(), v * bare_scale, True))
    return out


def loose_values(text, t, R, bare_display_scale=None):
    body = R.normalise(text)
    k = t['kind']
    if k in NUMERIC_KINDS:
        vals = _numeric_values(body, t, R)
    elif k == 'clock':
        vals = _clock_values(body, R)
    elif k == 'duration':
        scale = None
        if t.get('bare_allowed'):
            scale = R.DURATION_UNITS[t['bare_unit']]
        if bare_display_scale is not None:
            scale = bare_display_scale
        vals = _duration_values(body, t, R, bare_scale=scale)
    else:
        raise ValueError(k)
    return sorted(vals, key=lambda x: (x[0], -x[1]))


EDGE = '([{"\'`*'
TAIL = ')]}"\'`*,;:!?'


def _span_values(toks, i, j, t, R):
    """Every value the scorer's parser reads from token span i..j under edge-strip variants."""
    span = ' '.join(toks[i:j + 1])
    got = set()
    for v in {span, span.lstrip(EDGE), span.rstrip(TAIL), span.lstrip(EDGE).rstrip(TAIL),
              span.lstrip(EDGE).rstrip(TAIL).rstrip('.').rstrip(TAIL)}:
        if not v:
            continue
        try:
            got.add(R.parse_prediction(v, t))
        except R.ParseError:
            continue
    return got


def typed_present(text, t, R):
    """A span the scorer accepts equals the target AND the span is not a fragment of a larger
    accepted span with a DIFFERENT value (`1 hour` inside `1 hour and 40 minutes`, `10:45`
    inside `10:45 pm`). Extensions of up to 3 tokens on either side are checked."""
    toks = R.normalise(text).split(' ')
    want = want_of(t, R)
    n = len(toks)
    for i in range(n):
        for j in range(i, min(n, i + 8)):
            if want not in _span_values(toks, i, j, t, R):
                continue
            fragment = False
            for a in range(max(0, i - 3), i + 1):
                for b in range(j, min(n, j + 4)):
                    if (a, b) == (i, j):
                        continue
                    if _span_values(toks, a, b, t, R) - {want}:
                        fragment = True
                        break
                if fragment:
                    break
            if not fragment:
                return True
    return False


def presence(text, t, R, display_scale=None):
    want = want_of(t, R)
    vals = loose_values(text, t, R)
    maximal = [v for v in vals if v[3]]
    rec = {'loose': any(v[2] == want for v in vals),
           'final': bool(maximal) and max(maximal, key=lambda x: x[0])[2] == want,
           'typed': typed_present(text, t, R),
           'n_values': len(maximal),
           'values': [str(v[2]) for v in maximal]}
    rec['carry'] = (t['kind'] == 'clock'
                    and any(v[2] == want for v in _clock_values(R.normalise(text), R, carry=True)))
    rec['subrun'] = False          # sensitivity: ONE pair inside a duration compound matches
    if t['kind'] == 'duration':
        body = R.normalise(text)
        for m in DUR_PAIR_RE.finditer(body):
            if m.group(2) in R.DURATION_UNITS:
                try:
                    if R._parse_duration_seconds(f'{m.group(1)} {m.group(2)}', t) == want:
                        rec['subrun'] = True
                except R.ParseError:
                    pass
    rec['bare_display'] = False
    if t['kind'] == 'duration' and not t.get('bare_allowed') and display_scale:
        rec['bare_display'] = any(v[2] == want for v in
                                  loose_values(text, t, R, bare_display_scale=display_scale))
    return rec
