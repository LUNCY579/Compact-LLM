from __future__ import annotations

"""Observable defect detection and explicit response-contract validation.

Evaluator v4.3 avoids catch-all semantic verdicts and adds prompt-grounded continuous functional scoring.  This module
only emits scored defect labels when the generated text contains a concrete,
reproducible problem or violates an explicit benchmark requirement.  Broader
signals such as low lexical overlap or unusual vocabulary are review-only and
never change the official quality score.
"""

import re
import statistics
import unicodedata
from collections import Counter
from typing import Any, Iterable

QUESTION_STARTERS = {
    "who", "what", "which", "where", "when", "why", "how", "can", "could",
    "would", "should", "do", "does", "did", "is", "are", "was", "were",
    "will", "have", "has",
}
FUNCTION_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "here", "him", "his", "how", "i", "if", "in", "is", "it", "its",
    "just", "me", "my", "no", "not", "of", "on", "or", "our", "she", "so",
    "that", "the", "their", "them", "then", "there", "they", "this", "to",
    "up", "was", "we", "were", "what", "when", "which", "who", "why", "will",
    "with", "would", "you", "your", "yes", "yeah", "sure", "okay", "ok",
}

SEVERITY_COST = {"critical": 1.0, "major": 0.35, "minor": 0.10}


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", str(text)).lower()
    text = text.replace("’", "'").replace("–", "-").replace("—", "-")
    return re.sub(r"\s+", " ", text).strip()


def words(text: str) -> list[str]:
    return re.findall(r"[^\W_]+(?:['-][^\W_]+)*|\d+(?:[._-]\d+)*", normalize(text), flags=re.UNICODE)


def is_question(text: str) -> bool:
    token_list = words(text)
    return "?" in text or "？" in text or bool(token_list and token_list[0] in QUESTION_STARTERS)


def _script_name(char: str) -> str | None:
    if not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    for candidate in (
        "LATIN", "CYRILLIC", "GREEK", "HEBREW", "ARABIC", "HIRAGANA",
        "KATAKANA", "HANGUL", "CJK", "THAI", "DEVANAGARI",
    ):
        if candidate in name:
            return candidate
    return "OTHER"


def _longest_run(token_list: list[str]) -> int:
    longest = current = 0
    previous: str | None = None
    for token in token_list:
        if token == previous:
            current += 1
        else:
            current = 1
            previous = token
        longest = max(longest, current)
    return longest


def _repeat_fraction(token_list: list[str], n: int) -> float:
    if len(token_list) < n * 2:
        return 0.0
    grams = [tuple(token_list[index:index + n]) for index in range(len(token_list) - n + 1)]
    counts = Counter(grams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / max(1, len(grams))


def _issue(label: str, severity: str, evidence: str, definition: str, source: str) -> dict[str, str]:
    return {
        "label": label,
        "severity": severity,
        "evidence": evidence,
        "definition": definition,
        "source": source,
    }


def _delimiter_balance(text: str) -> tuple[bool, int]:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    mismatches = 0
    for char in text:
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                mismatches += 1
    return not stack and mismatches == 0, len(stack) + mismatches


def _version_or_measurement_token(token: str) -> bool:
    cleaned = token.strip(".,!?;:'\"()[]{}<>|/\\-_+=*#@~`“”‘’")
    if re.fullmatch(r"[A-Za-z]:[\\/].+", token) or ("\\" in token and any(char.isalpha() for char in token)):
        return True
    return bool(re.fullmatch(
        r"(?:v?\d+(?:\.\d+)+|[A-Za-z]{1,16}-?v?\d+(?:\.\d+)*|"
        r"\d+(?:\.\d+)?(?:gb|mb|kb|tb|ms|s|%)|c\+\+\d*|c#\d*|-?o\d)",
        cleaned,
        flags=re.IGNORECASE,
    ))


def _normalized_short_reply(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", normalize(text))).strip()


COMMON_VALID_SHORT_REPLIES = {
    "yes", "no", "okay", "ok", "sure", "perfect", "nice", "great", "thanks",
    "thank you", "got it", "noted", "understood", "maybe", "probably", "exactly",
    "agreed", "congrats", "sorry", "fine", "good", "well done", "right", "yep",
    "nope", "absolutely", "correct", "done", "ready", "fair", "true", "same",
    "wait", "why", "how", "who", "what", "where", "when",
}


def detect_observable_defects(
    reply: str,
    *,
    visible_model_tokens: int | None = None,
    expected_language: str | None = None,
) -> dict[str, Any]:
    """Detect concrete mechanical, decoding, or surface-structure corruption.

    v4.3 does not use category, reference similarity, uncommon vocabulary, or a
    semantic catch-all label.  A severe corruption result must be supported by
    independently measurable surface evidence such as fragmented layout,
    digit-letter fusion, abnormal tokenizer fragmentation, punctuation damage,
    repeated tokens, or script islands inside an expected-English response.

    Multilingual text remains valid by itself.  Script mixing contributes to a
    defect only when it appears together with other objective fragmentation
    signals.  Version numbers, measurements, model identifiers, emoji, code
    tokens, and ordinary technical punctuation are explicitly protected.
    """

    normalized = normalize(reply)
    token_list = words(reply)
    chars = [char for char in reply if not char.isspace()]
    letters = [char for char in chars if char.isalpha()]
    digits = [char for char in chars if char.isdigit()]
    symbols = [char for char in chars if not char.isalnum()]
    counts = Counter(token_list)
    raw_lines = reply.splitlines()
    lines = [line.strip() for line in raw_lines if line.strip()]
    raw_segments = re.findall(r"\S+", reply)
    scripts = Counter(filter(None, (_script_name(char) for char in letters)))
    active_scripts = {key: value for key, value in scripts.items() if value >= 1}
    latin_letters = int(scripts.get("LATIN", 0))
    non_latin_letters = sum(value for key, value in scripts.items() if key != "LATIN")
    replacement_count = reply.count("�") + reply.count("\ufffd")
    invalid_controls = [
        f"U+{ord(char):04X}" for char in reply
        if unicodedata.category(char) in {"Cc", "Cs", "Co"} and char not in "\n\r\t"
    ]
    longest_run = _longest_run(token_list)
    bigram_repeat = _repeat_fraction(token_list, 2)
    trigram_repeat = _repeat_fraction(token_list, 3)
    max_token_share = max(counts.values(), default=0) / max(1, len(token_list))
    symbol_ratio = len(symbols) / max(1, len(chars))
    digit_ratio = len(digits) / max(1, len(chars))
    median_line_length = statistics.median(
        sum(char.isalnum() for char in line) for line in lines
    ) if lines else 0.0
    short_line_count = sum(1 for line in lines if sum(char.isalnum() for char in line) <= 5)
    micro_line_count = sum(1 for line in lines if sum(char.isalnum() for char in line) <= 2)
    blank_line_ratio = (len(raw_lines) - len(lines)) / max(1, len(raw_lines))
    model_tokens_per_word = (
        float(visible_model_tokens) / max(1, len(token_list))
        if visible_model_tokens is not None else None
    )

    script_switches = 0
    previous_script: str | None = None
    for char in reply:
        current_script = _script_name(char)
        if current_script and previous_script and current_script != previous_script:
            script_switches += 1
        if current_script:
            previous_script = current_script

    mixed_alphanumeric_tokens: list[str] = []
    internal_punctuation_tokens: list[str] = []
    abnormal_camel_tokens: list[str] = []
    for segment in raw_segments:
        cleaned = segment.strip(".,!?;:'\"()[]{}<>|/\\-_+=*#@~`“”‘’")
        if (
            any(char.isalpha() for char in cleaned)
            and any(char.isdigit() for char in cleaned)
            and not _version_or_measurement_token(segment)
        ):
            mixed_alphanumeric_tokens.append(cleaned)
        if (
            re.search(r"[A-Za-z](?:[:;|{}\[\]<>]{1,3})[A-Za-z]", segment)
            and not re.search(r"\b(?:real-time|wall-clock|a/b|c\+\+|foo_bar)\b", segment, flags=re.IGNORECASE)
        ):
            internal_punctuation_tokens.append(segment)
        # CamelCase, acronyms, and mixed-case technical identifiers are valid
        # language surfaces (for example LoRA, QLoRA, OpenAI, and PowerShell).
        # They must never be treated as decoding corruption by themselves.

    punctuation_clusters = re.findall(r"[^\w\s]{3,}", reply, flags=re.UNICODE)
    long_digit_runs = re.findall(r"\d{4,}", reply)
    numeric_only_lines = sum(1 for line in lines if re.fullmatch(r"[\d\s.,:+-]+", line))
    delimiters_balanced, delimiter_error_count = _delimiter_balance(reply)

    # Independent, explainable corruption signals.  None of these is a
    # semantic opinion about whether the answer is relevant.
    layout_fragmentation = bool(
        (len(lines) >= 5 and (median_line_length <= 12 or short_line_count >= 2 or blank_line_ratio >= 0.35))
        or (len(lines) >= 4 and blank_line_ratio >= 0.35)
        or (len(lines) == 4 and 2 < median_line_length <= 12 and short_line_count >= 1)
    )
    # A high ratio is strong enough for a standalone defect.  A milder ratio
    # is retained only as corroborating evidence and never fails an otherwise
    # clean technical or multilingual reply by itself.
    soft_tokenizer_fragmentation = bool(
        model_tokens_per_word is not None
        and len(token_list) >= 4
        and model_tokens_per_word >= 2.00
    )
    tokenizer_fragmentation = bool(
        model_tokens_per_word is not None
        and (
            (len(token_list) >= 4 and model_tokens_per_word >= 2.75)
            or (len(token_list) <= 3 and len(lines) >= 2 and model_tokens_per_word >= 2.40)
        )
    )
    mixed_alphanumeric_fragmentation = bool(
        mixed_alphanumeric_tokens
        and len(mixed_alphanumeric_tokens) / max(1, len(raw_segments)) >= 0.12
    )
    numeric_segments = [segment for segment in raw_segments if any(char.isdigit() for char in segment)]
    math_expression = bool(re.fullmatch(r"[\d\s+\-*/=().]+", reply.strip()))
    unexplained_numeric_segments = [segment for segment in numeric_segments if not _version_or_measurement_token(segment)]
    numeric_fragment_noise = bool(
        long_digit_runs
        or numeric_only_lines >= 2
        or (not math_expression and len(unexplained_numeric_segments) >= 3 and digit_ratio >= 0.10)
    )
    expected_english = bool(expected_language and str(expected_language).lower().startswith("en"))
    script_island_fragmentation = bool(
        expected_english and non_latin_letters >= 1 and latin_letters >= 4 and script_switches >= 2
    )
    technical_surface = any(_version_or_measurement_token(segment) for segment in raw_segments)
    punctuation_threshold = 0.32 if technical_surface else 0.18
    punctuation_fragmentation = bool(
        (punctuation_clusters and not technical_surface)
        or not delimiters_balanced
        or (len(chars) >= 12 and symbol_ratio >= punctuation_threshold)
        or (len(lines) >= 2 and symbol_ratio >= punctuation_threshold)
    )
    micro_fragmentation = bool(len(lines) >= 3 and micro_line_count >= 2)
    internal_token_fragmentation = bool(internal_punctuation_tokens or abnormal_camel_tokens)
    dominant_token_repetition = bool(
        len(token_list) >= 8
        and max(counts.values(), default=0) >= 4
        and max_token_share >= 0.25
    )
    isolated_fragment = bool(
        expected_english
        and len(token_list) <= 2
        and _normalized_short_reply(reply) not in COMMON_VALID_SHORT_REPLIES
        and (
            len(lines) >= 2
            or (bool(reply.strip()) and not reply.strip()[0].isalnum())
            or script_island_fragmentation
            or (tokenizer_fragmentation and len(lines) >= 2)
        )
    )

    defects: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []

    if not normalized or not token_list:
        defects.append(_issue(
            "empty_output", "critical", "No linguistic token was produced.",
            "The generated response is empty or contains no readable linguistic content.", "mechanical",
        ))
    if replacement_count:
        defects.append(_issue(
            "encoding_replacement_character", "critical", f"Found {replacement_count} U+FFFD replacement character(s).",
            "The output contains Unicode replacement characters, indicating broken decoding or text corruption.", "mechanical",
        ))
    if invalid_controls:
        defects.append(_issue(
            "invalid_control_character", "critical", f"Unexpected controls: {', '.join(invalid_controls[:8])}.",
            "The output contains non-printing control/private-use characters outside normal whitespace.", "mechanical",
        ))
    if longest_run >= 4:
        defects.append(_issue(
            "repeated_token_loop", "critical", f"The same token repeats {longest_run} consecutive times.",
            "A token is repeated in a loop that makes the utterance unusable.", "mechanical",
        ))
    if len(token_list) >= 8 and (max_token_share >= 0.45 or dominant_token_repetition):
        token, count = counts.most_common(1)[0]
        defects.append(_issue(
            "single_token_loop", "critical", f"Token {token!r} occupies {count}/{len(token_list)} words.",
            "One token dominates a multiword response, indicating repetition collapse.", "mechanical",
        ))
    if len(token_list) >= 8 and bigram_repeat >= 0.55:
        defects.append(_issue(
            "repeated_phrase_loop", "critical", f"Repeated-bigram fraction is {bigram_repeat:.3f}.",
            "A short phrase repeats often enough to form a generation loop.", "mechanical",
        ))
    elif len(token_list) >= 10 and trigram_repeat >= 0.30:
        defects.append(_issue(
            "repeated_clause_loop", "major", f"Repeated-trigram fraction is {trigram_repeat:.3f}.",
            "A clause-like token sequence repeats and materially harms readability.", "mechanical",
        ))
    if len(chars) >= 12 and symbol_ratio > 0.48:
        defects.append(_issue(
            "excessive_symbol_fragment", "critical", f"Non-alphanumeric symbol ratio is {symbol_ratio:.3f}.",
            "The response is dominated by symbols rather than readable text.", "mechanical",
        ))
    if layout_fragmentation:
        defects.append(_issue(
            "fragmented_line_output", "major",
            f"{len(lines)} nonempty lines; median alphanumeric line length {median_line_length:.1f}; blank-line ratio {blank_line_ratio:.2f}.",
            "The response is split into many short or widely separated fragments instead of a normal utterance.", "mechanical",
        ))
    elif len(lines) >= 3 and median_line_length <= 2:
        defects.append(_issue(
            "micro_fragment_sequence", "major", f"{len(lines)} micro-lines; median alphanumeric length {median_line_length:.1f}.",
            "The response consists of several mechanically fragmented micro-lines.", "mechanical",
        ))
    if tokenizer_fragmentation:
        defects.append(_issue(
            "tokenization_fragmentation", "major",
            f"{model_tokens_per_word:.2f} model tokens per visible linguistic token.",
            "The decoded output is abnormally fragmented relative to its visible linguistic-token count.", "mechanical",
        ))
    if mixed_alphanumeric_fragmentation:
        defects.append(_issue(
            "mixed_alphanumeric_fragmentation", "major",
            f"Digit-letter fusion appears in: {mixed_alphanumeric_tokens[:6]}.",
            "Several decoded fragments fuse letters and digits in forms that are not versions, measurements, or normal identifiers.", "mechanical",
        ))
    if numeric_fragment_noise:
        defects.append(_issue(
            "numeric_fragment_noise", "major",
            f"Digit ratio={digit_ratio:.3f}; long digit runs={len(long_digit_runs)}; numeric-only lines={numeric_only_lines}.",
            "Unexplained numeric runs or numeric-only fragments materially disrupt the response surface.", "mechanical",
        ))
    if script_island_fragmentation:
        diagnostics.append({
            "label": "multiscript_output_review",
            "evidence": f"Scripts present: {dict(sorted(active_scripts.items()))}; script switches={script_switches}.",
            "definition": "Multiple writing systems appear. This alone is not a defect and has zero score cost.",
        })
    elif len(active_scripts) >= 2:
        diagnostics.append({
            "label": "multiscript_output_review",
            "evidence": f"Scripts present: {dict(sorted(active_scripts.items()))}.",
            "definition": "Multiple writing systems appear. This alone is not a defect and has zero score cost.",
        })
    if punctuation_fragmentation:
        defects.append(_issue(
            "punctuation_fragment_sequence", "major",
            f"Symbol ratio={symbol_ratio:.3f}; punctuation clusters={len(punctuation_clusters)}; delimiter errors={delimiter_error_count}.",
            "Punctuation or delimiter fragments materially interrupt the decoded text surface.", "mechanical",
        ))
    if internal_token_fragmentation:
        defects.append(_issue(
            "internal_token_fragmentation", "major",
            f"Abnormal internal token boundaries: {(internal_punctuation_tokens + abnormal_camel_tokens)[:6]}.",
            "Decoded token fragments contain abnormal internal punctuation or casing boundaries.", "mechanical",
        ))
    if isolated_fragment:
        defects.append(_issue(
            "isolated_lexical_fragment", "critical", f"Only {len(token_list)} linguistic token(s) across {len(lines)} line(s): {reply!r}.",
            "A lone decoded fragment replaces a complete short utterance; protected ordinary one-word replies are excluded.", "mechanical",
        ))

    # Concrete malformed clause patterns from earlier real failures.
    malformed_patterns = {
        "you_can_you": r"\byou\s+can\s+you\b",
        "bad_second_person_verb": r"\byou\s+(?:is|was|gives|creates)\b",
        "duplicated_negation": r"\bnot\s+not\b",
        "orphan_possessive": r"\byou['’]s\b",
        "repeated_subject_chain": r"\b(?:i|you|we|they|he|she|it)(?:\s+(?:i|you|we|they|he|she|it)){2,}\b",
        "article_copula_fragment": r"\b(?:the|a|an)\s+(?:is|are|was|were)(?:\s|[.,;!?]|$)",
    }
    matched = [name for name, pattern in malformed_patterns.items() if re.search(pattern, normalized)]
    tiny_clauses = [
        clause.strip() for clause in re.split(r"[.!?;\n]+", normalized)
        if clause.strip() and len(words(clause)) <= 2
    ]
    if len(matched) >= 2:
        defects.append(_issue(
            "broken_clause_sequence", "critical", f"Matched malformed patterns: {', '.join(matched)}.",
            "Multiple concrete clause-construction errors make the intended sentence structure unrecoverable.", "structural",
        ))
    elif matched:
        defects.append(_issue(
            "malformed_clause_pattern", "major", f"Matched malformed pattern: {matched[0]}.",
            "A concrete clause-construction pattern is malformed and materially harms readability.", "structural",
        ))
    if len(token_list) >= 8 and len(tiny_clauses) >= 3:
        defects.append(_issue(
            "broken_clause_sequence", "major", f"Found {len(tiny_clauses)} one- or two-word clause fragments.",
            "The utterance contains a sequence of tiny broken clauses instead of a recoverable sentence.", "structural",
        ))

    signal_names = [
        name for name, active in {
            "fragmented layout": layout_fragmentation,
            "tokenizer fragmentation": tokenizer_fragmentation,
            "soft tokenizer fragmentation": soft_tokenizer_fragmentation,
            "digit-letter fusion": mixed_alphanumeric_fragmentation,
            "numeric noise": numeric_fragment_noise,
            "script islands": script_island_fragmentation,
            "punctuation fragmentation": punctuation_fragmentation,
            "micro-lines": micro_fragmentation,
            "internal token fragmentation": internal_token_fragmentation,
        }.items() if active
    ]
    compound_corruption = bool(
        (soft_tokenizer_fragmentation and (
            layout_fragmentation or mixed_alphanumeric_fragmentation or script_island_fragmentation
            or numeric_fragment_noise or internal_token_fragmentation or punctuation_fragmentation
        ))
        or (tokenizer_fragmentation and (
            layout_fragmentation or mixed_alphanumeric_fragmentation or script_island_fragmentation
            or numeric_fragment_noise or internal_token_fragmentation
            or (punctuation_fragmentation and (len(lines) >= 2 or symbol_ratio > 0.60))
        ))
        or len(signal_names) >= 3
        or (layout_fragmentation and (
            mixed_alphanumeric_fragmentation or numeric_fragment_noise
            or script_island_fragmentation or internal_token_fragmentation
        ))
        or (mixed_alphanumeric_fragmentation and script_island_fragmentation)
        or (script_island_fragmentation and internal_token_fragmentation and len(lines) >= 2)
        or (tokenizer_fragmentation and len(token_list) >= 10 and internal_token_fragmentation)
    )
    if compound_corruption:
        defects.append(_issue(
            "compound_decoding_corruption", "critical",
            "Independent surface signals: " + ", ".join(signal_names) + ".",
            "Multiple independent mechanical decoding anomalies jointly make the visible response unusable.", "mechanical",
        ))

    if expected_language and str(expected_language).lower().startswith("en") and active_scripts and "LATIN" not in active_scripts:
        diagnostics.append({
            "label": "expected_language_review",
            "evidence": f"Expected {expected_language}; scripts present: {dict(sorted(active_scripts.items()))}.",
            "definition": "The declared benchmark language and observed script may differ; this is review-only unless other corruption is present.",
        })

    severity_rank = {"minor": 1, "major": 2, "critical": 3}
    by_label: dict[str, dict[str, str]] = {}
    for item in defects:
        prior = by_label.get(item["label"])
        if prior is None or severity_rank[item["severity"]] > severity_rank[prior["severity"]]:
            by_label[item["label"]] = item
    defects = sorted(by_label.values(), key=lambda item: (-severity_rank[item["severity"]], item["label"]))

    penalty = min(1.0, sum(SEVERITY_COST[item["severity"]] for item in defects))
    return {
        "defects": defects,
        "review_diagnostics": diagnostics,
        "mechanical_integrity_score": max(0.0, 1.0 - penalty),
        "metrics": {
            "word_count": len(token_list),
            "unique_word_ratio": len(set(token_list)) / max(1, len(token_list)),
            "consecutive_token_run": longest_run,
            "max_token_share": max_token_share,
            "repeated_bigram_fraction": bigram_repeat,
            "repeated_trigram_fraction": trigram_repeat,
            "replacement_character_count": replacement_count,
            "invalid_control_characters": invalid_controls,
            "symbol_ratio": symbol_ratio,
            "digit_ratio": digit_ratio,
            "model_tokens_per_word": model_tokens_per_word,
            "soft_tokenizer_fragmentation": soft_tokenizer_fragmentation,
            "nonempty_line_count": len(lines),
            "blank_line_ratio": blank_line_ratio,
            "median_alphanumeric_line_length": median_line_length,
            "short_line_count": short_line_count,
            "micro_line_count": micro_line_count,
            "script_counts": dict(sorted(active_scripts.items())),
            "script_switch_count": script_switches,
            "mixed_alphanumeric_tokens": mixed_alphanumeric_tokens,
            "internal_punctuation_tokens": internal_punctuation_tokens,
            "abnormal_camel_tokens": abnormal_camel_tokens,
            "punctuation_cluster_count": len(punctuation_clusters),
            "long_digit_run_count": len(long_digit_runs),
            "delimiter_error_count": delimiter_error_count,
            "malformed_clause_patterns": matched,
            "tiny_clause_count": len(tiny_clauses),
            "compound_corruption_signals": signal_names,
        },
    }

def _set_confidence(scores: dict[str, float], mode: str, value: float) -> None:
    scores[mode] = max(scores.get(mode, 0.0), max(0.0, min(1.0, float(value))))


def detect_response_mode_confidences(reply: str) -> dict[str, float]:
    """Estimate broad conversational functions without emitting defect labels.

    These scores are intentionally coarse and transparent. They measure whether
    a reply performs one of the response modes explicitly authored in the case
    contract. They are not semantic-failure labels and never create a binary
    veto. A benchmark may accept several modes, and the best matching mode is
    used as continuous functional evidence.
    """

    text = normalize(reply)
    token_set = set(words(reply))
    scores: dict[str, float] = {}

    if is_question(reply):
        _set_confidence(scores, "question", 1.0)
        _set_confidence(scores, "clarification_or_follow_up", 1.0)
    elif re.search(r"\b(?:i need|show|share|send|provide)\b.*\b(?:source|metadata|artifact|details?|evidence|identity|name|description|test|result|context)\b", text):
        _set_confidence(scores, "clarification_or_follow_up", 0.90)

    if re.search(r"\b(?:got it|noted|understood|acknowledged|good call|makes sense|checkpointed|east hall it is|perfect|yep)\b", text):
        _set_confidence(scores, "acknowledgment", 1.0)
    elif re.search(r"\b(?:okay|ok|right|all right|sounds good|that works|that is good|that's good|ready)\b", text):
        _set_confidence(scores, "acknowledgment", 0.76)

    if re.search(r"\b(?:nice|great|excellent|awesome|congratulations|congrats|impressive|finally|well done|love that|approved)\b", text):
        _set_confidence(scores, "celebration", 1.0)
    elif re.search(r"\b(?:good|promising|ready to (?:merge|evaluate|begin|ship))\b", text):
        _set_confidence(scores, "celebration", 0.68)

    if re.search(r"\b(?:i am sorry|i'm sorry|sorry|that sounds|sounds rough|painful|rough|frustrating|exhausting|annoying|hurt|sad|disappoint(?:ed|ing)|depressing|i know you wanted|not easy|that is hard|that's hard|harder to trust|below what i expected|i expected)\b", text):
        _set_confidence(scores, "empathy", 1.0)
    elif re.search(r"\b(?:understandable|i understand|that makes sense emotionally)\b", text):
        _set_confidence(scores, "empathy", 0.72)

    # Generic negation is weak evidence. A concrete boundary or recommendation
    # is stronger than phrases such as "not enough" with no actionable meaning.
    if re.search(r"^(?:no|do not|don't|not without|cannot|can't)\b", text) or re.search(
        r"\b(?:i would wait|wait first|wait a little|wait until|do not|don't|cannot|can't|avoid|stop|not acceptable|not a good way|not the way|draft it|take a moment|not something i can take (?:away|lightly)|accept the apology|boundary respected)\b", text
    ):
        _set_confidence(scores, "boundary_or_disagreement", 0.95)
    elif re.search(r"\b(?:not enough|not good|not perfect|not something i would want)\b", text):
        _set_confidence(scores, "boundary_or_disagreement", 0.45)

    if re.search(r"\b(?:not sure|uncertain|maybe|probably|depends|not guaranteed|not definitive|cannot verify|can't verify|cannot tell|can't tell|cannot know|can't know|need (?:the|a|more|specific)|only know|only report|confirmed information|without (?:a|the) (?:source|location|device|signal|evidence|data))\b", text):
        _set_confidence(scores, "uncertainty_or_knowledge_limit", 1.0)

    if re.search(r"\b(?:keep it private|will not tell|won't tell|not share|not disclose|safe with me|your .* is safe|you decide|you choose)\b", text):
        _set_confidence(scores, "privacy_assurance", 1.0)

    if re.search(r"\b(?:you can|i meant it|promise still stands|still stands|door is still open|i will|i'll|will stop|won't|will not|do not repeat|don't repeat|not repeat|promise)\b", text):
        _set_confidence(scores, "commitment_or_reassurance", 1.0)
    elif re.search(r"\b(?:that is important|that's important|exactly what i promised|exactly what i need)\b", text):
        _set_confidence(scores, "commitment_or_reassurance", 0.45)

    if re.search(r"\b(?:means|is a|are a|stores?|caches?|reuses?|trains?|learns?|updates?|adapts?|reduces?|removes?|compresses?)\b", text):
        _set_confidence(scores, "definition_or_explanation", 0.90)

    if re.search(r"\b(?:write|draft|start with|begin with|take a deep breath|take a breath|first sentence|opening line|small step|call when|try this|give the conversation|small doorway)\b", text):
        _set_confidence(scores, "supportive_next_step", 1.0)

    if re.search(r"\b(?:i think|i would|i'd|worth it|leave it|pick|choose|prefer|better|worse|typical|normal|admirable|impressive|exhausting|advantage|clear job)\b", text):
        _set_confidence(scores, "brief_opinion_or_comment", 0.88)

    playful_terms = {
        "heroic", "mud", "miracle", "opera", "verse", "bard", "bells", "door",
        "adventurous", "heavy", "pocket", "audience", "tune", "magic", "angle",
        "smoke", "water", "shutters", "room", "window",
    }
    if token_set & playful_terms or re.search(r"\b(?:one minor miracle|try not to get lost|the bells go quiet|follow the water|one line, not an opera)\b", text):
        _set_confidence(scores, "playful_or_in_character", 1.0)
    elif re.match(r"^(?:ah|mm|oh)[, .!-]", text):
        _set_confidence(scores, "playful_or_in_character", 0.45)

    return dict(sorted(scores.items()))


def detect_response_modes(reply: str) -> set[str]:
    """Return response modes with non-zero evidence for compatibility."""

    return {mode for mode, score in detect_response_mode_confidences(reply).items() if score > 0.0}



def infer_prompt_mode_expectations(record: dict[str, Any]) -> dict[str, Any]:
    """Infer broad acceptable conversational functions from the actual prompt.

    Category and reference text are deliberately ignored.  The result is a
    permissive, continuous hint rather than a scored defect contract.  Several
    modes are accepted for each obvious prompt shape so alternate natural
    strategies remain valid.
    """

    messages = record.get("messages") or []
    user = next((str(item.get("content", "")) for item in reversed(messages) if item.get("role") == "user"), "")
    system = " ".join(str(item.get("content", "")) for item in messages if item.get("role") == "system")
    text = normalize(user)
    system_text = normalize(system)
    expected: set[str] = set()
    reasons: list[str] = []

    roleplay = any(cue in system_text for cue in ("stay in character", "you are ori", "you are lumi", "current scene", "relationship to the player"))
    if roleplay:
        expected.add("playful_or_in_character")
        reasons.append("system explicitly requests in-character behavior")

    negative_emotion = re.search(
        r"\b(?:frustrat|exhaust|tired|sad|upset|angry|nervous|scared|afraid|disappoint|rejection|froze|hurt|hate|mock|made fun|inconsistent)\w*\b",
        text,
    )
    positive_update = re.search(
        r"\b(?:accepted|approved|beat the base|got .* to train|completed|finished|passed|worked|succeeded|success|first adapter|first fine[- ]?tune)\b",
        text,
    )
    neutral_update = re.search(
        r"\b(?:backed up|created a fresh branch|moved the meeting|reduced the context|changed the dataset|tried training|quick update|just to let you know|that is where i am now)\b",
        text,
    )
    apology_or_boundary = re.search(r"\b(?:made fun|apolog|sorry|hate when|boundary|do not joke|don't joke)\b", text)
    relationship_memory = re.search(r"\b(?:you said i could|promise|still stands|come to you when)\b", text)

    if negative_emotion:
        expected.update({"empathy", "acknowledgment", "clarification_or_follow_up", "supportive_next_step"})
        reasons.append("prompt contains a negative emotional or setback cue")
    if positive_update:
        expected.update({"celebration", "acknowledgment", "clarification_or_follow_up", "brief_opinion_or_comment"})
        reasons.append("prompt reports an achievement or positive result")
    if neutral_update:
        expected.update({"acknowledgment", "clarification_or_follow_up", "brief_opinion_or_comment"})
        reasons.append("prompt reports a neutral project or context update")
    if apology_or_boundary:
        expected.update({"empathy", "boundary_or_disagreement", "commitment_or_reassurance", "acknowledgment"})
        reasons.append("prompt raises apology or interpersonal-boundary content")
    if relationship_memory:
        expected.update({"commitment_or_reassurance", "acknowledgment", "empathy"})
        reasons.append("prompt asks about a prior relational commitment")

    if is_question(user):
        if re.search(r"\b(?:what is|what does|define|explain|why is|why does)\b", text):
            expected.update({"definition_or_explanation", "clarification_or_follow_up", "uncertainty_or_knowledge_limit"})
            reasons.append("prompt asks for an explanation or definition")
        elif re.search(r"\b(?:can i|should i|would it|is it|do i|will)\b", text):
            expected.update({"brief_opinion_or_comment", "boundary_or_disagreement", "uncertainty_or_knowledge_limit", "clarification_or_follow_up"})
            reasons.append("prompt asks for a judgment, prediction, or permission")
        else:
            expected.update({"brief_opinion_or_comment", "clarification_or_follow_up", "uncertainty_or_knowledge_limit", "acknowledgment"})
            reasons.append("prompt is interrogative")

    if re.search(r"\b(?:write|draft|help me|give me|start|call them|what should i say)\b", text):
        expected.update({"supportive_next_step", "clarification_or_follow_up", "brief_opinion_or_comment"})
        reasons.append("prompt requests an action or next step")

    # A direct privacy assurance is a legitimate function when the prompt
    # mentions an undisclosed private state.  Objective privacy requirements,
    # when needed, still belong in the explicit case contract.
    if re.search(r"\b(?:private|secret|not told|do not tell|don't tell)\b", text):
        expected.update({"privacy_assurance", "empathy", "acknowledgment"})
        reasons.append("prompt contains a privacy or disclosure cue")

    return {
        "expected_modes": sorted(expected),
        "reasons": reasons,
        "category_used": False,
        "reference_used": False,
    }


def prompt_function_alignment(
    record: dict[str, Any],
    reply: str,
    *,
    mechanical_integrity_score: float,
    contract: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return continuous prompt-function alignment without a defect label.

    Evaluator v4.3 first uses an optional human-authored ``prompt_adequacy``
    rule stored in the individual response contract.  That rule is based on the
    actual prompt/context and can provide task, factual, context, and safety
    dimension scores.  Category and reference text are never consulted.

    When no authored rule exists, the broad prompt-shape heuristic remains as a
    permissive fallback.
    """

    inferred = infer_prompt_mode_expectations(record)
    expected = inferred["expected_modes"]
    authored = (contract or {}).get("prompt_adequacy") or {}
    if mechanical_integrity_score < 0.75:
        confidences: dict[str, float] = {}
        score = 0.0
        dimension_scores: dict[str, float] = {}
        source = "mechanically_gated"
    else:
        confidences = detect_response_mode_confidences(reply)
        dimension_scores = {}
        if authored:
            task_rule = authored.get("task") or {}
            score = _score_rule(reply, task_rule) if task_rule else 0.80
            for name in ("factual", "context", "safety"):
                rule = authored.get(name)
                if rule:
                    dimension_scores[name] = _score_rule(reply, rule)
            source = "explicit_case_prompt_adequacy"
        elif expected:
            score = max((confidences.get(mode, 0.0) for mode in expected), default=0.0)
            source = "prompt_shape_fallback"
        else:
            # No clear prompt function was inferred.  Keep the score neutral
            # rather than pretending a perfect semantic judgment.
            score = 0.80
            source = "neutral_fallback"
    return {
        **inferred,
        "detected_mode_confidences": confidences,
        "detected_modes": sorted(confidences),
        "score": max(0.0, min(1.0, float(score))),
        "dimension_scores": {key: max(0.0, min(1.0, float(value))) for key, value in dimension_scores.items()},
        "alignment_source": source,
        "authored_prompt_adequacy_used": bool(authored),
    }

def _matches_rule(reply: str, rule: dict[str, Any]) -> bool:
    return _score_rule(reply, rule) >= float(rule.get("pass_score", 1.0))


def _score_rule(reply: str, rule: dict[str, Any]) -> float:
    """Score a transparent lexical/mode evidence rule in ``[0, 1]``.

    Supported forms include phrase/regex/term/mode/question checks, partial
    concept groups, weighted components, fixed-score evidence options, a
    ``default_score`` for unmatched but readable replies, and explicit
    ``score_caps`` for prompt-grounded contradictions or subject reversals.
    """

    text = normalize(reply)
    token_set = set(words(reply))
    mode_scores = detect_response_mode_confidences(reply)

    if rule.get("evidence_options"):
        base = float(rule.get("default_score", 0.0))
        for option in rule.get("evidence_options") or []:
            option_rule = option.get("match") or {}
            if _score_rule(reply, option_rule) >= float(option.get("minimum_match", 1.0)):
                base = max(base, float(option.get("score", 1.0)))
    elif rule.get("all_components"):
        weighted = 0.0
        total = 0.0
        for component in rule.get("all_components") or []:
            weight = max(0.0, float(component.get("weight", 1.0)))
            weighted += weight * _score_rule(reply, component.get("match") or {})
            total += weight
        base = weighted / total if total else float(rule.get("default_score", 0.0))
    else:
        checks: list[float] = []
        if "any_phrases" in rule:
            checks.append(1.0 if any(normalize(phrase) in text for phrase in rule.get("any_phrases") or []) else 0.0)
        if "all_phrases" in rule:
            checks.append(1.0 if all(normalize(phrase) in text for phrase in rule.get("all_phrases") or []) else 0.0)
        if "any_regex" in rule:
            checks.append(1.0 if any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in rule.get("any_regex") or []) else 0.0)
        if "all_regex" in rule:
            checks.append(1.0 if all(re.search(pattern, text, flags=re.IGNORECASE) for pattern in rule.get("all_regex") or []) else 0.0)
        if "any_terms" in rule:
            checks.append(1.0 if token_set & {normalize(term) for term in rule.get("any_terms") or []} else 0.0)
        if "all_term_groups" in rule:
            groups = rule.get("all_term_groups") or []
            checks.append(sum(1.0 for group in groups if token_set & {normalize(term) for term in group}) / max(1, len(groups)))
        if "any_modes" in rule:
            checks.append(max((mode_scores.get(str(mode), 0.0) for mode in rule.get("any_modes") or []), default=0.0))
        if "question" in rule:
            checks.append(1.0 if is_question(reply) is bool(rule["question"]) else 0.0)
        if "concept_groups" in rule:
            groups = rule.get("concept_groups") or []
            group_scores: list[float] = []
            for group in groups:
                group_checks: list[bool] = []
                if group.get("terms"):
                    group_checks.append(bool(token_set & {normalize(term) for term in group.get("terms") or []}))
                if group.get("phrases"):
                    group_checks.append(any(normalize(phrase) in text for phrase in group.get("phrases") or []))
                if group.get("regex"):
                    group_checks.append(any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in group.get("regex") or []))
                if group.get("modes"):
                    group_checks.append(max((mode_scores.get(str(mode), 0.0) for mode in group.get("modes") or []), default=0.0) >= float(group.get("mode_threshold", 0.65)))
                group_scores.append(1.0 if any(group_checks) else 0.0)
            checks.append(sum(group_scores) / max(1, len(group_scores)))

        if not checks:
            base = float(rule.get("default_score", 0.0))
        else:
            combine = str(rule.get("combine", "all"))
            if combine == "mean":
                base = sum(checks) / len(checks)
            elif combine == "any":
                base = max(checks)
            else:
                base = min(checks)

    for cap_rule in rule.get("score_caps") or []:
        if _score_rule(reply, cap_rule.get("match") or {}) >= float(cap_rule.get("minimum_match", 1.0)):
            base = min(base, float(cap_rule.get("cap", 1.0)))
    return max(0.0, min(1.0, base))


def _requirement_defect(requirement: dict[str, Any], score: float) -> dict[str, str]:
    severity = str(requirement.get("severity", "major"))
    partial_threshold = float(requirement.get("partial_credit_minor_threshold", 0.45))
    if severity == "major" and score >= partial_threshold and not requirement.get("strict_severity"):
        severity = str(requirement.get("partial_severity", "minor"))
    return _issue(
        str(requirement["label"]),
        severity,
        str(requirement.get("failure_evidence") or requirement.get("description") or requirement.get("id") or "Explicit contract requirement was not fully met."),
        str(requirement.get("description") or "An explicit benchmark response requirement was not fully met."),
        "response_contract",
    )


def validate_response_contract(contract: dict[str, Any], reply: str, *, mechanical_integrity_score: float = 1.0) -> dict[str, Any]:
    """Validate explicit requirements and score authored functional modes."""

    defects: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []
    requirement_results: list[dict[str, Any]] = []

    for requirement in contract.get("requirements") or []:
        match_rule = requirement.get("match") or {}
        score = _score_rule(reply, match_rule)
        required_score = float(requirement.get("minimum_score", match_rule.get("pass_score", 1.0)))
        passed = score >= required_score
        result = {
            "id": str(requirement.get("id") or requirement.get("label")),
            "label": str(requirement.get("label")),
            "passed": passed,
            "score": score,
            "minimum_score": required_score,
            "weight": float(requirement.get("weight", 1.0)),
            "severity": str(requirement.get("severity", "major")),
            "description": str(requirement.get("description", "")),
        }
        requirement_results.append(result)
        if not passed:
            defects.append(_requirement_defect(requirement, score))

    for prohibition in contract.get("prohibitions") or []:
        violation_score = _score_rule(reply, prohibition.get("match") or {})
        violated = violation_score >= float(prohibition.get("minimum_score", 1.0))
        result = {
            "id": str(prohibition.get("id") or prohibition.get("label")),
            "label": str(prohibition.get("label")),
            "passed": not violated,
            "score": 0.0 if violated else 1.0,
            "minimum_score": 1.0,
            "weight": float(prohibition.get("weight", 1.0)),
            "severity": str(prohibition.get("severity", "major")),
            "description": str(prohibition.get("description", "")),
        }
        requirement_results.append(result)
        if violated:
            defects.append(_requirement_defect(prohibition, 0.0))

    if contract.get("question_required"):
        passed = is_question(reply)
        label = str(contract.get("question_missing_label", "required_question_missing"))
        requirement_results.append({
            "id": "question_required", "label": label, "passed": passed,
            "score": 1.0 if passed else 0.0, "minimum_score": 1.0,
            "weight": float(contract.get("question_weight", 1.0)), "severity": "major",
        })
        if not passed:
            defects.append(_issue(
                label, "major", "The explicit response contract requires a question, but the reply is not interrogative.",
                "The benchmark explicitly requires a clarification or follow-up question.", "response_contract",
            ))

    if contract.get("question_forbidden"):
        passed = not is_question(reply)
        label = str(contract.get("question_forbidden_label", "forbidden_question_present"))
        requirement_results.append({
            "id": "question_forbidden", "label": label, "passed": passed,
            "score": 1.0 if passed else 0.0, "minimum_score": 1.0,
            "weight": float(contract.get("question_weight", 1.0)), "severity": "major",
        })
        if not passed:
            defects.append(_issue(
                label, "major", "The explicit response contract forbids a question, but the reply is interrogative.",
                "The benchmark explicitly forbids a follow-up question for this case.", "response_contract",
            ))

    # v4.3 does not score or restrict replies using category-authored
    # ``expected_modes``.  Legacy fields are ignored.  Prompt-function scoring
    # is computed separately from the actual messages.
    mode_confidences = detect_response_mode_confidences(reply) if mechanical_integrity_score >= 0.75 else {}
    expected_modes: list[str] = []
    mode_fulfillment = 1.0
    if contract.get("expected_modes") or contract.get("legacy_expected_modes"):
        diagnostics.append({
            "label": "legacy_expected_modes_ignored",
            "evidence": "The record contains an old expected_modes field.",
            "definition": "Evaluator v4.3 ignores category-authored response-mode restrictions; this diagnostic has zero score cost.",
        })

    for item in contract.get("recommended") or []:
        if _score_rule(reply, item.get("match") or {}) < float(item.get("minimum_score", 1.0)):
            diagnostics.append({
                "label": str(item.get("label", "recommended_content_missing")),
                "evidence": str(item.get("description") or "Recommended, non-required content was absent."),
                "definition": "A non-mandatory benchmark preference was not observed; this has zero issue cost.",
            })

    total_weight = sum(float(item.get("weight", 1.0)) for item in requirement_results)
    fulfillment = (
        sum(float(item.get("weight", 1.0)) * float(item.get("score", 0.0)) for item in requirement_results) / total_weight
        if total_weight else 1.0
    )

    domains = contract.get("domains") or {}
    domain_scores: dict[str, float] = {}
    for domain, requirement_ids in domains.items():
        ids = set(requirement_ids or [])
        selected = [item for item in requirement_results if item.get("id") in ids]
        weight = sum(float(item.get("weight", 1.0)) for item in selected)
        domain_scores[str(domain)] = (
            sum(float(item.get("weight", 1.0)) * float(item.get("score", 0.0)) for item in selected) / weight
            if weight else 1.0
        )

    return {
        "defects": defects,
        "review_diagnostics": diagnostics,
        "requirement_results": requirement_results,
        "contract_fulfillment_score": fulfillment,
        "response_mode_fulfillment_score": mode_fulfillment,
        "domain_scores": domain_scores,
        "detected_modes": sorted(mode_confidences),
        "detected_mode_confidences": mode_confidences,
    }


def issue_labels(defects: Iterable[dict[str, str]], severity: str | None = None) -> list[str]:
    return sorted({
        item["label"] for item in defects
        if severity is None or item.get("severity") == severity
    })
