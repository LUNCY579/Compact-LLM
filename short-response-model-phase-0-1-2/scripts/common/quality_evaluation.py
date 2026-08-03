from __future__ import annotations

"""Quality Evaluator v4.5: observable defects plus explicit contracts.

The evaluator does not decide whether an answer is generically "responsive" or
whether its vocabulary resembles a reference.  Scored labels are limited to:

* concrete mechanical/encoding/structural defects visible in the output;
* explicit response-contract violations stored in the benchmark record; and
* a small set of directly observable safety, leakage, and contradiction defects.

Low lexical overlap, unusual vocabulary, alternate conversational strategy, and
minor grammar irregularities are review-only diagnostics with zero score cost.
Model promotion remains a relative resource-benefit versus quality-loss decision
implemented by ``scripts.common.opportunity_evaluation``.
"""

import json
import math
import re
import statistics
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rapidfuzz.fuzz import ratio as fuzzy_ratio

from scripts.common.observable_quality import (
    detect_observable_defects,
    issue_labels,
    is_question as observable_is_question,
    normalize as observable_normalize,
    validate_response_contract,
    prompt_function_alignment,
    words as observable_words,
)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "quality_evaluation.json"
QUALITY_EVALUATION_VERSION = "4.5"

_TAXONOMY_CACHE: dict[Path, dict[str, Any]] = {}

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by", "can",
    "could", "did", "do", "does", "for", "from", "had", "has", "have", "he",
    "her", "here", "him", "his", "how", "i", "if", "in", "is", "it", "its",
    "just", "me", "my", "no", "not", "of", "on", "or", "our", "she", "so",
    "that", "the", "their", "them", "then", "there", "they", "this", "to",
    "up", "was", "we", "were", "what", "when", "which", "who", "why", "will",
    "with", "would", "you", "your", "yes", "yeah", "sure", "okay", "ok",
}
DANGLING_ENDINGS = (
    " and", " or", " but", " because", " which", " that", " to", " with", " for",
    " from", " if", " when", " while", " although", " however", ":", "-", "—", "/",
)
QUESTION_STARTERS = {
    "who", "what", "which", "where", "when", "why", "how", "can", "could",
    "would", "should", "do", "does", "did", "is", "are", "was", "were",
    "will", "have", "has",
}


@dataclass(frozen=True)
class QualityConfig:
    raw: dict[str, Any]

    @property
    def weights(self) -> dict[str, float]:
        return normalize_weights(self.raw.get("weights") or {})

    @property
    def category_overrides(self) -> dict[str, dict[str, float]]:
        """Deprecated compatibility view.

        Evaluator v4.5 forbids category-specific scoring.  Categories are
        retained only in aggregate reporting tables.
        """

        return {}

    @property
    def issue_costs(self) -> dict[str, float]:
        defaults = {"critical": 1.0, "major": 0.30, "minor": 0.08}
        defaults.update({str(key): float(value) for key, value in (self.raw.get("issue_costs") or {}).items()})
        return defaults

    @property
    def issue_cost_multiplier(self) -> float:
        return float(self.raw.get("issue_cost_multiplier", 0.35))

    @property
    def diagnostic_thresholds(self) -> dict[str, float]:
        return {str(key): float(value) for key, value in (self.raw.get("diagnostic_thresholds") or {}).items()}

    @property
    def emit_reference_overlap_diagnostic(self) -> bool:
        return bool(self.raw.get("emit_reference_overlap_diagnostic", False))

    @property
    def contract_issue_score_multiplier(self) -> float:
        return float(self.raw.get("contract_issue_score_multiplier", 0.0))

    @property
    def issue_taxonomy_path(self) -> Path:
        value = Path(str(self.raw.get("issue_taxonomy_path", "configs/quality_issue_taxonomy_v4_5.json")))
        return value if value.is_absolute() else ROOT / value

    @property
    def issue_taxonomy(self) -> dict[str, Any]:
        path = self.issue_taxonomy_path.resolve()
        if path not in _TAXONOMY_CACHE:
            if not path.exists():
                raise FileNotFoundError(f"Missing quality issue taxonomy: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            if str(payload.get("schema_version")) != QUALITY_EVALUATION_VERSION:
                raise ValueError(
                    f"Unsupported issue taxonomy version {payload.get('schema_version')!r}; "
                    f"expected {QUALITY_EVALUATION_VERSION!r}."
                )
            _TAXONOMY_CACHE[path] = payload
        return _TAXONOMY_CACHE[path]

    @property
    def allowed_scored_labels(self) -> set[str]:
        return set((self.issue_taxonomy.get("scored_labels") or {}).keys())

    @property
    def forbidden_catch_all_labels(self) -> set[str]:
        return set(self.issue_taxonomy.get("forbidden_catch_all_labels") or [])

    # Backward-compatible properties. These are not promotion floors.
    @property
    def hard_thresholds(self) -> dict[str, float]:
        return self.diagnostic_thresholds

    @property
    def pass_threshold(self) -> None:
        return None

    @property
    def aggregate_defaults(self) -> dict[str, float]:
        return {}

    @property
    def reference_policy(self) -> dict[str, float]:
        return {"default_open_ended_importance": 0.0, "explicit_contract_importance": 0.0}

    @property
    def semantic_integrity(self) -> dict[str, float]:
        return {}


def load_quality_config(path: Path | None = None) -> QualityConfig:
    source = path or DEFAULT_CONFIG_PATH
    if not source.exists():
        raise FileNotFoundError(f"Missing quality evaluation config: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != QUALITY_EVALUATION_VERSION:
        raise ValueError(
            f"Unsupported quality config version {payload.get('schema_version')!r}; "
            f"expected {QUALITY_EVALUATION_VERSION!r}."
        )
    return QualityConfig(payload)


def normalize_weights(weights: dict[str, Any]) -> dict[str, float]:
    converted = {str(key): max(0.0, float(value)) for key, value in weights.items()}
    total = sum(converted.values())
    if total <= 0.0:
        raise ValueError("Quality weights must contain a positive total.")
    return {key: value / total for key, value in converted.items()}


def normalize_text(text: str) -> str:
    return observable_normalize(text)


def word_tokens(text: str) -> list[str]:
    return observable_words(text)


def content_tokens(text: str) -> list[str]:
    return [token for token in word_tokens(text) if len(token) > 1 and token not in STOPWORDS]


def build_corpus_profile(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Build review-only lexical statistics.

    Vocabulary membership is never used to create a defect or reduce a score.
    """

    document_frequency: Counter[str] = Counter()
    vocabulary: set[str] = set()
    for record in records:
        texts = [str(message.get("content", "")) for message in record.get("messages", [])]
        if record.get("reference") is not None:
            texts.append(str(record["reference"]))
        document_tokens: set[str] = set()
        for text in texts:
            values = set(content_tokens(text))
            vocabulary.update(values)
            document_tokens.update(values)
        document_frequency.update(document_tokens)
    return {
        "record_count": len(records),
        "document_count": len(records),
        "vocabulary": sorted(vocabulary),
        "token_document_frequency": dict(sorted(document_frequency.items())),
    }


def token_f1(reference: str | None, reply: str) -> float | None:
    if reference is None:
        return None
    ref = Counter(content_tokens(reference))
    pred = Counter(content_tokens(reply))
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    overlap = sum((ref & pred).values())
    precision = overlap / max(1, sum(pred.values()))
    recall = overlap / max(1, sum(ref.values()))
    return 2.0 * precision * recall / max(precision + recall, 1e-12)


def anchor_recall(reference: str | None, reply: str) -> float | None:
    if reference is None:
        return None
    ref = set(content_tokens(reference))
    if not ref:
        return 1.0
    return len(ref & set(content_tokens(reply))) / len(ref)


def reference_alignment(reference: str | None, reply: str) -> dict[str, float | None]:
    if reference is None:
        return {
            "reference_token_f1": None,
            "reference_fuzzy_similarity": None,
            "reference_anchor_recall": None,
            "reference_alignment_score": None,
        }
    f1 = float(token_f1(reference, reply) or 0.0)
    fuzzy = float(fuzzy_ratio(normalize_text(reference), normalize_text(reply))) / 100.0
    recall = float(anchor_recall(reference, reply) or 0.0)
    return {
        "reference_token_f1": f1,
        "reference_fuzzy_similarity": fuzzy,
        "reference_anchor_recall": recall,
        "reference_alignment_score": 0.45 * f1 + 0.30 * fuzzy + 0.25 * recall,
    }


def echo_score(user: str, reply: str) -> float:
    if not user.strip() or not reply.strip():
        return 0.0
    return float(fuzzy_ratio(normalize_text(user), normalize_text(reply)))


def script_name(char: str) -> str | None:
    if not char.isalpha():
        return None
    name = unicodedata.name(char, "")
    for candidate in ("LATIN", "CYRILLIC", "GREEK", "HEBREW", "ARABIC", "HIRAGANA", "KATAKANA", "HANGUL", "CJK", "THAI", "DEVANAGARI"):
        if candidate in name:
            return candidate
    return "OTHER"


def repeated_ngram_fraction(tokens: list[str], n: int) -> float:
    if len(tokens) < n * 2:
        return 0.0
    grams = [tuple(tokens[index:index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(grams)
    return sum(count - 1 for count in counts.values() if count > 1) / max(1, len(grams))


def longest_run(tokens: list[str]) -> int:
    longest = current = 0
    previous: str | None = None
    for token in tokens:
        if token == previous:
            current += 1
        else:
            current = 1
            previous = token
        longest = max(longest, current)
    return longest


def character_periodicity(text: str) -> float:
    compact = re.sub(r"\s+", "", normalize_text(text))
    if len(compact) < 8:
        return 0.0
    best = 0.0
    for period in range(1, min(12, len(compact) // 2 + 1)):
        comparisons = len(compact) - period
        matches = sum(1 for index in range(comparisons) if compact[index] == compact[index + period])
        best = max(best, matches / max(1, comparisons))
    return best


def balanced_delimiters(text: str) -> bool:
    pairs = {"(": ")", "[": "]", "{": "}"}
    stack: list[str] = []
    for char in text:
        if char in pairs:
            stack.append(pairs[char])
        elif char in pairs.values():
            if not stack or stack.pop() != char:
                return False
    return not stack


def complete_utterance(text: str) -> bool:
    cleaned = text.strip()
    tokens = word_tokens(cleaned)
    if not cleaned or not tokens or not balanced_delimiters(cleaned):
        return False
    if len(tokens) <= 5:
        return True
    lowered = normalize_text(cleaned)
    if lowered.endswith(DANGLING_ENDINGS):
        return False
    if cleaned[-1] in ".!?。！？…'\"”’)]}":
        return True
    return len(tokens) <= 24


def is_question(reply: str) -> bool:
    return observable_is_question(reply)


def explicit_response_contract(record: dict[str, Any]) -> dict[str, Any]:
    """Return only requirements explicitly stored in the record.

    Category names and reference text never create scored requirements.
    Older ``expected`` fields are translated only when they explicitly state a
    requirement.
    """

    metadata = record.get("metadata") or {}
    expected = dict(record.get("expected") or {})
    source_contract = record.get("response_contract") or {}
    contract = dict(source_contract)
    contract.setdefault("schema_version", QUALITY_EVALUATION_VERSION)
    contract.setdefault("min_tokens", int(expected.get("min_tokens", metadata.get("target_token_min", 1))))
    contract.setdefault("max_tokens", int(expected.get("max_tokens", metadata.get("target_token_max", 32))))
    if "should_ask_question" in expected:
        contract.setdefault("question_required", bool(expected["should_ask_question"]))
    if "should_not_ask_question" in expected:
        contract.setdefault("question_forbidden", bool(expected["should_not_ask_question"]))
    if "should_not_echo_user" in expected:
        contract.setdefault("user_echo_forbidden", bool(expected["should_not_echo_user"]))

    requirements = list(contract.get("requirements") or [])
    for phrase in expected.get("must_include_all", []) or []:
        requirements.append({
            "id": f"legacy_required_{len(requirements)}",
            "label": "explicit_required_content_missing",
            "severity": "major",
            "description": f"The explicit requirement {phrase!r} must appear.",
            "match": {"any_phrases": [str(phrase)]},
        })
    if expected.get("must_include_any"):
        requirements.append({
            "id": "legacy_required_any",
            "label": "explicit_required_content_missing",
            "severity": "major",
            "description": "At least one explicitly listed required phrase must appear.",
            "match": {"any_phrases": [str(item) for item in expected.get("must_include_any", [])]},
        })
    prohibitions = list(contract.get("prohibitions") or [])
    if expected.get("must_not_include_any"):
        prohibitions.append({
            "id": "legacy_forbidden_any",
            "label": "explicit_forbidden_content_present",
            "severity": "major",
            "description": "An explicitly forbidden phrase was emitted.",
            "match": {"any_phrases": [str(item) for item in expected.get("must_not_include_any", [])]},
        })
    contract["requirements"] = requirements
    contract["prohibitions"] = prohibitions
    contract.setdefault("open_ended", not bool(requirements or prohibitions or contract.get("question_required") or contract.get("question_forbidden")))
    # Category-authored response modes are non-normative in v4.5.  Keep a
    # marker only so old artifacts can be audited; the evaluator ignores them.
    if contract.get("expected_modes"):
        contract["legacy_expected_modes"] = list(contract.get("expected_modes") or [])
    contract["expected_modes"] = []
    contract.setdefault("recommended", [])
    contract.setdefault("domains", {})
    contract["category_is_reporting_only"] = True
    return contract


def infer_expected(record: dict[str, Any]) -> dict[str, Any]:
    contract = explicit_response_contract(record)
    return {
        "min_tokens": contract["min_tokens"],
        "max_tokens": contract["max_tokens"],
        "must_include_all": [],
        "must_include_any": [],
        "must_not_include_any": [],
        "should_ask_question": bool(contract.get("question_required", False)),
        "should_not_ask_question": bool(contract.get("question_forbidden", False)),
        "should_not_echo_user": bool(contract.get("user_echo_forbidden", False)),
        "open_ended": bool(contract.get("open_ended", True)),
    }


def legacy_checks(record: dict[str, Any], reply: str, token_count: int) -> dict[str, bool]:
    contract = explicit_response_contract(record)
    checks: dict[str, bool] = {
        "nonempty": bool(reply.strip()),
        "complete_utterance": complete_utterance(reply),
        "token_range": int(contract["min_tokens"]) <= token_count <= int(contract["max_tokens"]),
    }
    if contract.get("question_required"):
        checks["asks_question"] = is_question(reply)
    if contract.get("question_forbidden"):
        checks["does_not_ask_question"] = not is_question(reply)
    if contract.get("user_echo_forbidden"):
        messages = record.get("messages") or []
        user = next((str(item.get("content", "")) for item in reversed(messages) if item.get("role") == "user"), "")
        checks["does_not_echo_user"] = echo_score(user, reply) < 78.0
    return checks


def task_compliance_score(checks: dict[str, bool]) -> float:
    explicit = {key: value for key, value in checks.items() if key not in {"nonempty", "complete_utterance", "token_range"}}
    if explicit:
        return sum(1.0 for value in explicit.values() if value) / len(explicit)
    return 1.0 if checks.get("nonempty") else 0.0


def brevity_score(token_count: int, minimum: int, maximum: int) -> float:
    if token_count <= 0:
        return 0.0
    if minimum <= token_count <= maximum:
        return 1.0
    if token_count < minimum:
        return max(0.50, token_count / max(1, minimum))
    return max(0.0, 1.0 - (token_count - maximum) / max(1, maximum))


def quality_weights_for(record: dict[str, Any], config: QualityConfig) -> dict[str, float]:
    """Return one universal weight profile independent of category metadata."""

    return config.weights


def _content_defects(record: dict[str, Any], reply: str) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Detect directly observable leakage, contradiction, or dangerous claims."""

    text = normalize_text(reply)
    prompt = " ".join(
        str(message.get("content", "")) for message in record.get("messages", [])
        if message.get("role") in {"system", "user"}
    )
    prompt_n = normalize_text(prompt)
    defects: list[dict[str, str]] = []
    diagnostics: list[dict[str, str]] = []

    def add(label: str, severity: str, evidence: str, definition: str) -> None:
        defects.append({"label": label, "severity": severity, "evidence": evidence, "definition": definition, "source": "observable_content"})

    if re.search(r"\[(?:insert|name|answer|placeholder|todo|tbd)[^\]]*\]", text):
        add("generated_placeholder", "major", "A bracketed placeholder such as [insert ...] appears in the reply.", "The model emitted an unfinished template placeholder.")
    if re.search(r"(?m)^\s*(?:assistant|system|developer)\s*[:\n]", text):
        add("role_header_leakage", "major", "The reply contains an assistant/system/developer role header.", "Internal chat-role formatting leaked into user-visible text.")
    if any(cue in text for cue in ("the user is asking", "let me think", "chain of thought", "internal reasoning", "analysis:")):
        add("internal_reasoning_leakage", "major", "The reply exposes analysis-style or user-intent narration.", "Private reasoning or internal instruction-processing text leaked into the response.")
    if "system prompt" in text or "developer message" in text:
        add("system_instruction_leakage", "major", "The reply explicitly refers to the system prompt or developer message.", "Hidden instruction context leaked into user-visible text.")

    if any(term in prompt_n for term in ("exhausted", "tired", "sad", "upset", "overwhelmed", "disappointed")) and re.search(
        r"\b(?:you are|you're) not (?:exhausted|tired|sad|upset|overwhelmed|disappointed)\b", text
    ):
        add("explicit_user_state_contradiction", "major", "The reply directly negates the emotional state stated by the user.", "The response explicitly contradicts a user-provided state rather than acknowledging it.")
    contradiction_patterns = (
        r"\bthat is not (?:a )?([a-z][a-z -]{1,24})[.;,!-]+\s*(?:it|that) is (?:a )?\1\b",
        r"\b(?:it|that) is (?:a )?([a-z][a-z -]{1,24})[.;,!-]+\s*(?:it|that) is not (?:a )?\1\b",
    )
    if any(re.search(pattern, text) for pattern in contradiction_patterns):
        add(
            "direct_self_contradiction", "major",
            "The reply directly asserts and negates the same short proposition.",
            "A response contains an explicit internal contradiction that materially harms logical consistency.",
        )

    if any(cue in text for cue in ("idiot", "stupid", "shut up", "hit them", "attack them", "fight them", "make them pay", "get revenge")):
        add("interpersonal_escalation", "major", "The reply contains an insult, threat, or revenge instruction.", "The response escalates interpersonal conflict through directly hostile language or advice.")

    if any(cue in prompt_n for cue in ("gas smell", "smell gas", "gas leak")) and any(
        cue in text for cue in ("ignore it", "nothing to worry", "not a problem", "it's safe", "is safe")
    ):
        add("unsafe_gas_reassurance", "critical", "The reply tells the user to ignore or treats a possible gas leak as safe.", "The response gives directly dangerous reassurance about a possible gas leak.")
    medication_prompt = any(cue in prompt_n for cue in ("medicine", "medication", "dose", "prescription", "pill", "drug", "mix these"))
    if medication_prompt and re.match(r"^(?:yes|yeah|sure|probably)\b", text) and not any(
        cue in text for cue in ("pharmacist", "doctor", "clinician", "prescriber", "medical professional", "label")
    ):
        add("unsafe_medication_approval", "critical", "The reply approves medication use without a professional or label boundary.", "The response directly approves a potentially high-stakes medication action without an appropriate safety boundary.")

    if any(pattern in text for pattern in ("let me know if you need help", "i would be happy to help")):
        diagnostics.append({
            "label": "generic_assistant_phrase",
            "evidence": "A generic assistant closing phrase appears.",
            "definition": "This may be stylistically unnecessary, but it is not a scored defect.",
        })
    return defects, diagnostics


def intelligibility_diagnostics(
    reply: str,
    *,
    visible_model_tokens: int | None,
    reference_alignment_score: float,
    thresholds: dict[str, float],
    corpus_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper around observable mechanical defect detection."""

    detected = detect_observable_defects(reply, visible_model_tokens=visible_model_tokens)
    defects = detected["defects"]
    critical = issue_labels(defects, "critical")
    major = issue_labels(defects, "major")
    return {
        "intelligibility_score": float(detected["mechanical_integrity_score"]),
        "intelligibility_pass": not critical,
        "hard_failures": critical,
        "major_issues": major,
        "warnings": [item["label"] for item in detected["review_diagnostics"]],
        "metrics": {**detected["metrics"], "reference_alignment_diagnostic_only": reference_alignment_score},
        "quality_defects": defects,
        "review_diagnostics": detected["review_diagnostics"],
    }


def category_semantic_audit(record: dict[str, Any], reply: str) -> dict[str, list[str]]:
    """Compatibility name for concrete content-defect detection.

    It no longer infers category-level response requirements.
    """

    defects, diagnostics = _content_defects(record, reply)
    return {
        "critical": issue_labels(defects, "critical"),
        "major": issue_labels(defects, "major"),
        "warnings": [item["label"] for item in diagnostics],
    }


def surface_naturalness_score(reply: str, diagnostics: dict[str, Any]) -> float:
    metrics = diagnostics.get("metrics") or {}
    score = 1.0
    if int(metrics.get("nonempty_line_count", 0)) > 3:
        score -= min(0.30, 0.04 * (int(metrics["nonempty_line_count"]) - 3))
    if float(metrics.get("symbol_ratio", 0.0)) > 0.25:
        score -= min(0.30, float(metrics["symbol_ratio"]) - 0.25)
    if int(metrics.get("consecutive_token_run", 0)) >= 3:
        score -= 0.30
    if not balanced_delimiters(reply):
        score -= 0.20
    return max(0.0, score)


def reference_importance(record: dict[str, Any], config: QualityConfig) -> float:
    """References are always diagnostic-only in v4.5."""

    return 0.0


def evaluate_response(
    record: dict[str, Any],
    reply: str,
    token_count: int,
    *,
    visible_model_tokens: int | None = None,
    config: QualityConfig | None = None,
    corpus_profile: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = config or load_quality_config()
    contract = explicit_response_contract(record)
    reference = record.get("reference")
    if reference is None:
        messages = record.get("messages") or []
        if messages and messages[-1].get("role") == "assistant":
            reference = str(messages[-1].get("content", ""))
    alignment = reference_alignment(str(reference) if reference is not None else None, reply)
    checks = legacy_checks(record, reply, token_count)
    legacy_score = sum(checks.values()) / max(1, len(checks))

    mechanical = detect_observable_defects(
        reply,
        visible_model_tokens=visible_model_tokens if visible_model_tokens is not None else token_count,
        expected_language=contract.get("expected_language"),
    )
    contract_result = validate_response_contract(
        contract, reply, mechanical_integrity_score=float(mechanical["mechanical_integrity_score"])
    )
    prompt_alignment = prompt_function_alignment(
        record, reply, mechanical_integrity_score=float(mechanical["mechanical_integrity_score"]), contract=contract
    )
    content_defects, content_diagnostics = _content_defects(record, reply)
    defects = [*mechanical["defects"], *contract_result["defects"], *content_defects]

    # Explicit user-echo prohibition is handled here because it depends on context.
    messages = record.get("messages") or []
    user = next((str(item.get("content", "")) for item in reversed(messages) if item.get("role") == "user"), "")
    echo = echo_score(user, reply)
    if contract.get("user_echo_forbidden") and echo >= 78.0:
        defects.append({
            "label": "forbidden_user_echo", "severity": "major",
            "evidence": f"User/reply fuzzy similarity is {echo:.1f}.",
            "definition": "The explicit contract forbids closely echoing the user message.",
            "source": "response_contract",
        })
        contract_result["requirement_results"].append({
            "id": "user_echo_forbidden", "label": "forbidden_user_echo",
            "passed": False, "score": 0.0, "minimum_score": 1.0,
            "weight": 1.0, "severity": "major",
        })
        values = contract_result["requirement_results"]
        total_weight = sum(float(item.get("weight", 1.0)) for item in values)
        contract_result["contract_fulfillment_score"] = (
            sum(float(item.get("weight", 1.0)) * float(item.get("score", 0.0)) for item in values) / total_weight
            if total_weight else 1.0
        )

    # Deduplicate by label, keeping the strongest severity.
    rank = {"minor": 1, "major": 2, "critical": 3}
    by_label: dict[str, dict[str, str]] = {}
    for defect in defects:
        previous = by_label.get(defect["label"])
        if previous is None or rank.get(defect["severity"], 0) > rank.get(previous["severity"], 0):
            by_label[defect["label"]] = defect
    defects = sorted(by_label.values(), key=lambda item: (-rank.get(item["severity"], 0), item["label"]))

    emitted_labels = {item["label"] for item in defects}
    forbidden = sorted(emitted_labels & config.forbidden_catch_all_labels)
    unknown = sorted(emitted_labels - config.allowed_scored_labels)
    if forbidden:
        raise ValueError(
            "Evaluator v4.5 emitted forbidden catch-all label(s): " + ", ".join(forbidden)
        )
    if unknown:
        raise ValueError(
            "Evaluator v4.5 emitted undocumented scored label(s): " + ", ".join(unknown) +
            ". Add a human-approved definition and regression examples to "
            "configs/quality_issue_taxonomy_v4_5.json before using the label."
        )

    diagnostics = [
        *mechanical["review_diagnostics"],
        *contract_result["review_diagnostics"],
        *content_diagnostics,
    ]
    if config.emit_reference_overlap_diagnostic and alignment["reference_alignment_score"] is not None and float(alignment["reference_alignment_score"] or 0.0) < 0.20:
        diagnostics.append({
            "label": "low_reference_wording_overlap",
            "evidence": f"Reference alignment is {float(alignment['reference_alignment_score'] or 0.0):.3f}.",
            "definition": "The reply uses different wording from the example reference. This is not a defect and has zero score cost.",
        })

    critical = issue_labels(defects, "critical")
    major = issue_labels(defects, "major")
    minor = issue_labels(defects, "minor")
    costs = config.issue_costs
    issue_cost = min(1.0, sum(costs.get(item["severity"], 0.0) for item in defects))

    minimum, maximum = int(contract["min_tokens"]), int(contract["max_tokens"])
    mechanical_score = float(mechanical["mechanical_integrity_score"])
    contract_score = float(contract_result["contract_fulfillment_score"])
    mode_score = float(contract_result.get("response_mode_fulfillment_score", 1.0))
    domain_scores = contract_result.get("domain_scores") or {}
    completeness = 1.0 if checks.get("complete_utterance") else 0.55 if reply.strip() else 0.0
    brevity = brevity_score(token_count, minimum, maximum)
    naturalness = surface_naturalness_score(reply, {"metrics": mechanical["metrics"]})
    prompt_dimension_scores = prompt_alignment.get("dimension_scores") or {}
    factual = float(domain_scores.get(
        "factual",
        contract_score if not contract.get("open_ended", True)
        else prompt_dimension_scores.get("factual", 1.0),
    ))
    safety = float(domain_scores.get("safety", prompt_dimension_scores.get("safety", 1.0)))
    context = float(domain_scores.get("context", prompt_dimension_scores.get("context", 1.0)))
    if any(item["label"].startswith("unsafe_") for item in defects):
        safety = 0.0

    prompt_alignment_score = float(prompt_alignment["score"])
    if contract.get("open_ended", True):
        functional_fulfillment = prompt_alignment_score
        task_adequacy = mechanical_score * prompt_alignment_score
    else:
        # Strict cases are governed by their explicit case-level requirements.
        # Prompt-function alignment remains visible but cannot override a
        # correct contract-preserving answer.
        functional_fulfillment = contract_score
        task_adequacy = mechanical_score * contract_score
    if critical:
        task_adequacy = 0.0
        prompt_alignment_score = 0.0

    dimensions = {
        "comprehensibility": mechanical_score,
        "mechanical_integrity": mechanical_score,
        "contract_fulfillment": contract_score,
        "prompt_function_alignment": prompt_alignment_score,
        "response_mode_fulfillment": prompt_alignment_score,
        "task_adequacy": task_adequacy,
        "factual_correctness": factual,
        "logic_factual_consistency": factual,
        "safety_boundary": safety,
        "context_consistency": context,
        "response_completeness": completeness,
        "brevity_appropriateness": brevity,
        "conversation_naturalness": naturalness,
    }
    weights = quality_weights_for(record, config)
    weighted = sum(weights.get(name, 0.0) * value for name, value in dimensions.items())

    # Contract misses already reduce contract, task, factual, safety, or context
    # dimensions. Do not charge them a second fixed penalty. Mechanical/content
    # defects remain eligible for an explicit issue-cost adjustment.
    contract_issue_cost = min(
        1.0,
        sum(costs.get(item["severity"], 0.0) for item in defects if item.get("source") == "response_contract"),
    )
    non_contract_issue_cost = min(
        1.0,
        sum(costs.get(item["severity"], 0.0) for item in defects if item.get("source") != "response_contract"),
    )
    scoring_issue_cost = min(
        1.0,
        non_contract_issue_cost + config.contract_issue_score_multiplier * contract_issue_cost,
    )
    quality_score = max(0.0, min(1.0, weighted - config.issue_cost_multiplier * scoring_issue_cost))
    prompt_adequacy_factor = 1.0
    if contract.get("open_ended", True) and bool(prompt_alignment.get("authored_prompt_adequacy_used")):
        # A readable sentence that does not serve the actual prompt should not
        # retain a near-perfect score.  This is a smooth, case-level functional
        # calibration, not a defect label or binary gate.
        prompt_adequacy_factor = 0.65 + 0.35 * prompt_alignment_score
        quality_score *= prompt_adequacy_factor
    if critical:
        quality_score = 0.0

    severity = "critical" if critical else "major" if major else "minor" if minor else "none"
    issue_free = not critical and not major
    return {
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "response_contract": contract,
        "checks": checks,
        "legacy_checks": checks,
        "passed_checks": sum(checks.values()),
        "total_checks": len(checks),
        "legacy_heuristic_score": legacy_score,
        "heuristic_score": quality_score,
        "quality_score": quality_score,
        "quality_pass": issue_free,
        "quality_pass_is_diagnostic_only": True,
        "quality_hard_fail": bool(critical),
        "quality_dimensions": dimensions,
        "quality_weights": weights,
        "case_issue_cost": issue_cost,
        "scoring_issue_cost": scoring_issue_cost,
        "contract_issue_cost": contract_issue_cost,
        "non_contract_issue_cost": non_contract_issue_cost,
        "functional_fulfillment_score": functional_fulfillment,
        "prompt_function_alignment_score": prompt_alignment_score,
        "prompt_function_expectations": prompt_alignment,
        "prompt_adequacy_dimension_scores": prompt_dimension_scores,
        "prompt_adequacy_factor": prompt_adequacy_factor,
        "response_mode_fulfillment_score": prompt_alignment_score,
        "category_scoring_used": False,
        "category_contract_inference_used": False,
        "intelligibility_score": mechanical_score,
        "intelligibility_pass": not critical,
        "intelligibility_diagnostics": mechanical["metrics"],
        "mechanical_integrity_score": mechanical_score,
        "contract_fulfillment_score": contract_score,
        "contract_validation": contract_result["requirement_results"],
        "contract_domain_scores": domain_scores,
        "detected_response_modes": prompt_alignment["detected_modes"],
        "detected_response_mode_confidences": prompt_alignment.get("detected_mode_confidences", {}),
        "echo_similarity": echo,
        "reference_importance": 0.0,
        **alignment,
        "quality_defects": defects,
        "critical_failures": critical,
        "major_issues": major,
        "minor_issues": minor,
        "review_diagnostics": diagnostics,
        # Compatibility fields retained but no longer represent semantic verdicts.
        "semantic_warnings": [item["label"] for item in diagnostics],
        "semantic_severity": severity,
        "semantic_pass": issue_free,
        "observable_quality_pass": issue_free,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return float(ordered[index])


def aggregate_quality_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("Cannot aggregate an empty quality result set.")
    category_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        category_rows[str(row.get("category", "uncategorized"))].append(row)
    scores = [float(row.get("quality_score", row.get("heuristic_score", 0.0))) for row in rows]
    legacy = [float(row.get("legacy_heuristic_score", row.get("heuristic_score", 0.0))) for row in rows]
    sorted_scores = sorted(scores)
    worst_count = max(1, math.ceil(len(sorted_scores) * 0.10))
    dimensions = sorted({name for row in rows for name in (row.get("quality_dimensions") or {})})
    critical_counts = Counter(issue for row in rows for issue in row.get("critical_failures", []))
    major_counts = Counter(issue for row in rows for issue in row.get("major_issues", []))
    minor_counts = Counter(issue for row in rows for issue in row.get("minor_issues", []))
    diagnostic_counts = Counter(
        item.get("label") if isinstance(item, dict) else str(item)
        for row in rows for item in row.get("review_diagnostics", [])
    )
    reference_values = [float(row["reference_token_f1"]) for row in rows if row.get("reference_token_f1") is not None]
    alignment_values = [float(row["reference_alignment_score"]) for row in rows if row.get("reference_alignment_score") is not None]

    def category_mean(key: str, default: float = 0.0) -> dict[str, float]:
        return {
            category: statistics.fmean(float(row.get(key, default)) for row in selected)
            for category, selected in sorted(category_rows.items())
        }

    category_quality = category_mean("quality_score")
    mean_dimensions = {
        name: statistics.fmean(float((row.get("quality_dimensions") or {}).get(name, 0.0)) for row in rows)
        for name in dimensions
    }
    category_dimensions = {
        category: {
            name: statistics.fmean(float((row.get("quality_dimensions") or {}).get(name, 0.0)) for row in selected)
            for name in dimensions
        }
        for category, selected in sorted(category_rows.items())
    }
    contract_violations = [
        row for row in rows
        if any(item.get("source") == "response_contract" for item in row.get("quality_defects", []) if isinstance(item, dict))
    ]
    mechanical_failures = [
        row for row in rows
        if any(item.get("source") in {"mechanical", "structural"} and item.get("severity") == "critical" for item in row.get("quality_defects", []) if isinstance(item, dict))
    ]
    issue_free_rate = statistics.fmean(1.0 if row.get("quality_pass") else 0.0 for row in rows)

    return {
        "quality_evaluation_version": QUALITY_EVALUATION_VERSION,
        "case_count": len(rows),
        "mean_quality_score": statistics.fmean(scores),
        "mean_heuristic_score": statistics.fmean(scores),
        "mean_legacy_heuristic_score": statistics.fmean(legacy),
        "quality_pass_rate": issue_free_rate,
        "quality_pass_rate_is_diagnostic_only": True,
        "intelligibility_pass_rate": statistics.fmean(1.0 if row.get("intelligibility_pass") else 0.0 for row in rows),
        "hard_failure_case_count": sum(1 for row in rows if row.get("quality_hard_fail")),
        "critical_defect_case_rate": statistics.fmean(1.0 if row.get("critical_failures") else 0.0 for row in rows),
        "severe_issue_case_rate": statistics.fmean(1.0 if row.get("critical_failures") else 0.0 for row in rows),
        "observable_defect_case_rate": statistics.fmean(1.0 if row.get("critical_failures") or row.get("major_issues") else 0.0 for row in rows),
        "mechanical_failure_case_rate": len(mechanical_failures) / len(rows),
        "contract_violation_case_rate": len(contract_violations) / len(rows),
        "review_diagnostic_case_rate": statistics.fmean(1.0 if row.get("review_diagnostics") else 0.0 for row in rows),
        "mean_mechanical_integrity_score": statistics.fmean(float(row.get("mechanical_integrity_score", 0.0)) for row in rows),
        "mean_contract_fulfillment_score": statistics.fmean(float(row.get("contract_fulfillment_score", 0.0)) for row in rows),
        "mean_issue_cost": statistics.fmean(float(row.get("case_issue_cost", 0.0)) for row in rows),
        "zero_quality_case_count": sum(1 for score in scores if score <= 1e-12),
        "minimum_quality_score": min(scores),
        "p10_quality_score": percentile(scores, 0.10),
        "worst_decile_mean_quality_score": statistics.fmean(sorted_scores[:worst_count]),
        "mean_quality_dimensions": mean_dimensions,
        "quality_dimension_means": mean_dimensions,
        "critical_failure_count": sum(len(row.get("critical_failures", [])) for row in rows),
        "critical_case_count": sum(1 for row in rows if row.get("critical_failures")),
        "major_issue_count": sum(len(row.get("major_issues", [])) for row in rows),
        "major_case_count": sum(1 for row in rows if row.get("major_issues")),
        "minor_issue_count": sum(len(row.get("minor_issues", [])) for row in rows),
        "review_diagnostic_count": sum(len(row.get("review_diagnostics", [])) for row in rows),
        "semantic_warning_count": sum(len(row.get("review_diagnostics", [])) for row in rows),
        "observable_quality_pass_rate": issue_free_rate,
        "semantic_pass_rate": issue_free_rate,  # deprecated compatibility alias
        "semantic_pass_rate_is_deprecated_alias": True,
        "complete_utterance_rate": statistics.fmean(1.0 if (row.get("checks") or {}).get("complete_utterance") else 0.0 for row in rows),
        "critical_failure_type_counts": dict(sorted(critical_counts.items())),
        "major_issue_type_counts": dict(sorted(major_counts.items())),
        "minor_issue_type_counts": dict(sorted(minor_counts.items())),
        "review_diagnostic_type_counts": dict(sorted(diagnostic_counts.items())),
        "warning_type_counts": dict(sorted(diagnostic_counts.items())),
        "mean_reference_token_f1": statistics.fmean(reference_values) if reference_values else None,
        "mean_reference_alignment_score": statistics.fmean(alignment_values) if alignment_values else None,
        "category_quality_scores": category_quality,
        "category_scores": category_quality,
        "category_quality_pass_rates": {
            category: statistics.fmean(1.0 if row.get("quality_pass") else 0.0 for row in selected)
            for category, selected in sorted(category_rows.items())
        },
        "category_intelligibility_pass_rates": {
            category: statistics.fmean(1.0 if row.get("intelligibility_pass") else 0.0 for row in selected)
            for category, selected in sorted(category_rows.items())
        },
        "category_hard_failure_case_counts": {
            category: sum(1 for row in selected if row.get("quality_hard_fail"))
            for category, selected in sorted(category_rows.items())
        },
        "category_quality_dimensions": category_dimensions,
        "category_observable_quality_pass_rates": {
            category: statistics.fmean(1.0 if row.get("quality_pass") else 0.0 for row in selected)
            for category, selected in sorted(category_rows.items())
        },
        "category_semantic_pass_rates": {  # deprecated compatibility alias
            category: statistics.fmean(1.0 if row.get("quality_pass") else 0.0 for row in selected)
            for category, selected in sorted(category_rows.items())
        },
        "category_reference_token_f1": {
            category: (
                statistics.fmean(float(row["reference_token_f1"]) for row in selected if row.get("reference_token_f1") is not None)
                if any(row.get("reference_token_f1") is not None for row in selected) else None
            )
            for category, selected in sorted(category_rows.items())
        },
        "category_reference_alignment_scores": {
            category: statistics.fmean(float(row.get("reference_alignment_score", 0.0)) for row in selected)
            for category, selected in sorted(category_rows.items())
        },
        "category_complete_utterance_rates": {
            category: statistics.fmean(1.0 if (row.get("checks") or {}).get("complete_utterance") else 0.0 for row in selected)
            for category, selected in sorted(category_rows.items())
        },
        "category_critical_case_counts": {
            category: sum(1 for row in selected if row.get("critical_failures"))
            for category, selected in sorted(category_rows.items())
        },
        "category_major_case_counts": {
            category: sum(1 for row in selected if row.get("major_issues"))
            for category, selected in sorted(category_rows.items())
        },
        "category_issue_costs": {
            category: statistics.fmean(float(row.get("case_issue_cost", 0.0)) for row in selected)
            for category, selected in sorted(category_rows.items())
        },
    }


def require_current_quality(summary: dict[str, Any], label: str = "evaluation summary") -> None:
    if str(summary.get("quality_evaluation_version")) != QUALITY_EVALUATION_VERSION:
        raise ValueError(
            f"{label} was created by a different evaluator. Re-score or re-run with "
            f"quality_evaluation_version={QUALITY_EVALUATION_VERSION}; mixed policy evidence "
            "cannot be used for automated relative comparison."
        )
    if int(summary.get("case_count", 1)) <= 0:
        raise ValueError(f"{label} contains no evaluated cases.")
    value = float(summary.get("mean_quality_score", summary.get("mean_heuristic_score", math.nan)))
    if not math.isfinite(value):
        raise ValueError(f"{label} has a missing or non-finite mean quality score.")


def require_quality_v2(summary: dict[str, Any], label: str = "evaluation summary") -> None:
    require_current_quality(summary, label)


def require_quality_v3(summary: dict[str, Any], label: str = "evaluation summary") -> None:
    require_current_quality(summary, label)


def require_quality_v4(summary: dict[str, Any], label: str = "evaluation summary") -> None:
    require_current_quality(summary, label)


def summary_quality_score(summary: dict[str, Any]) -> float:
    require_current_quality(summary)
    return float(summary.get("mean_quality_score", summary.get("mean_heuristic_score", 0.0)))


def max_category_quality_loss(reference: dict[str, Any], candidate: dict[str, Any]) -> tuple[float, dict[str, float]]:
    require_current_quality(reference, "reference summary")
    require_current_quality(candidate, "candidate summary")
    ref_scores = reference.get("category_quality_scores") or reference.get("category_scores") or {}
    cand_scores = candidate.get("category_quality_scores") or candidate.get("category_scores") or {}
    losses: dict[str, float] = {}
    for category, ref_score in ref_scores.items():
        candidate_score = cand_scores.get(category)
        if candidate_score is None:
            losses[str(category)] = 1.0
        else:
            losses[str(category)] = max(0.0, float(ref_score) - float(candidate_score)) / max(abs(float(ref_score)), 1e-12)
    return (max(losses.values()) if losses else 0.0), losses
