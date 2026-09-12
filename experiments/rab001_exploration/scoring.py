"""One canonical answer scorer, imported everywhere.

This module exists because the scorer was duplicated across scripts and the copies drifted.
Two separate bugs came out of that drift:

1. `recovery_pilot.py` used full-string equality, so a model answering "3." instead of "3"
   scored 0/6 while being entirely correct. That inverted a published conclusion.
2. `baseline_scan.py` and `rigorous_pilot.py` then used leading-integer matching, which was
   too lenient in the other direction: 31 of 50 `time_duration` answers are clock times like
   "10:25 AM.", so "10:55 AM" matched on the hour alone, and decimals collapsed the same way
   ("62.5" vs "62.9" both read as 62).

The rule below is format-tolerant but numerically strict, and there is now exactly one copy.
"""
import re

_NUM = re.compile(r'-?\d+(?:\.\d+)?')


def numbers(text: str) -> list[str]:
    """Every number in the string, decimals included."""
    return _NUM.findall(text or '')


def normalise(text: str) -> str:
    """Trailing-period and case/whitespace insensitive."""
    return ' '.join((text or '').strip().rstrip('.').casefold().split())


def answer_matches(got: str, want: str) -> bool:
    """Accept a different FORMAT of the same answer; reject a different VALUE.

    "128 units." matches "128." because the corpus style varies, but "10:55 AM" does not
    match "10:25 AM.", and "62.9" does not match "62.5".
    """
    if normalise(got) == normalise(want):
        return True
    a, b = numbers(got), numbers(want)
    return bool(b) and a == b


def strict_matches(got: str, want: str) -> bool:
    """Raw equality, reported alongside so format drift stays visible."""
    return (got or '').strip() == (want or '').strip()


SELF_TEST = [
    ('3.', '3', True), ('128 units.', '128.', True), ('$45.', '$45.', True),
    ('10:25 AM.', '10:25 AM.', True), ('10:55 AM', '10:25 AM.', False),
    ('62.9', '62.5.', False), ('62.5', '62.5.', True),
    ('3.7 cups', '3.2 cups.', False), ('30 units.', '40.', False),
    ('Yes.', '1000 meters.', False), ('', '5.', False),
]


def self_test() -> None:
    bad = [(g, w, e) for g, w, e in SELF_TEST if answer_matches(g, w) != e]
    if bad:
        raise AssertionError(f'scorer self-test failed: {bad}')


if __name__ == '__main__':
    self_test()
    print(f'scorer self-test: all {len(SELF_TEST)} cases pass')
