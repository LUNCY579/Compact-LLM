from __future__ import annotations

"""Build Evaluator-v4.3 prompt-grounded contracts from v4.2 records.

Category remains reporting-only.  The migration adds two narrowly scoped
features:

1. Better paraphrase evidence for existing objective requirements.
2. Optional, case-level ``prompt_adequacy`` rules for open conversational
   prompts.  These rules contribute continuous task/factual/context scores but
   never emit issue labels or act as hard gates.

Every rule is derived from the actual user prompt and system/context available
in the individual record, never from ``category`` or reference wording.
"""

import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
VERSION = "4.3"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).lower()).strip()


def evidence_options(options: list[tuple[float, list[str]]], *, default: float = 0.35, caps: list[tuple[float, list[str]]] | None = None) -> dict[str, Any]:
    rule: dict[str, Any] = {
        "evidence_options": [
            {"score": score, "match": {"any_regex": patterns}}
            for score, patterns in options
        ],
        "default_score": default,
    }
    if caps:
        rule["score_caps"] = [
            {"cap": cap, "match": {"any_regex": patterns}}
            for cap, patterns in caps
        ]
    return rule


def _patch_requirement(requirement: dict[str, Any]) -> dict[str, Any]:
    item = copy.deepcopy(requirement)
    rid = str(item.get("id", ""))

    if rid == "reject_flops_only_claim":
        item["minimum_score"] = 0.72
        item["weight"] = 2.0
        item["match"] = evidence_options([
            (1.0, [r"^(?:no|not alone|not reliably|cannot|can't)\b", r"\b(?:flops alone|theoretical flops)\b.*\b(?:not enough|insufficient|cannot|can't|not reliable)\b"]),
            (0.88, [r"\b(?:not sufficient|does not guarantee|doesn't guarantee|not guaranteed|not a guaranteed|incomplete|only theoretical)\b"]),
        ], default=0.0)
    elif rid == "measured_runtime_evidence":
        item["minimum_score"] = 0.72
        item["severity"] = "minor"
        item["weight"] = 0.5
        item["match"] = evidence_options([
            (1.0, [r"\b(?:wall[- ]?clock|latency|runtime|benchmark|measure|measured|timing|real[- ]world speed)\b"]),
            (0.55, [r"\b(?:test it|actual speed|real speed|performance test)\b"]),
        ], default=0.0)
    elif rid == "visibility_limit":
        item["minimum_score"] = 0.65
        item["match"] = evidence_options([
            (1.0, [r"\b(?:cannot see|can't see|do not see|don't see|cannot identify|can't identify|not from this room|visibility is blocked|blocked|sealed shutters|shutters are sealed|no view)\b"]),
            (0.72, [r"^(?:not clearly|not fully|unclear|i cannot tell|i can't tell)\b"]),
        ], default=0.0, caps=[
            (0.25, [r"\b(?:phone|incoming calls?|caller id|without .* id)\b"]),
        ])
    elif rid == "sealed_set_isolation":
        item["minimum_score"] = 0.72
        item["match"] = evidence_options([
            (1.0, [r"\b(?:untouched|held out|holdout|isolated|sealed|do not use|don't use|never use|keep .* separate|stay .* untouched|not inspect|not tune on)\b"]),
            (0.65, [r"\b(?:preserve|protect|leave .* alone)\b"]),
        ], default=0.0)
    elif rid == "scope_narrowing_or_brief_compression_answer":
        # The user explicitly asks for a ten-word overview.  A concise, accurate
        # high-level explanation is valid; narrowing is another valid strategy.
        item["minimum_score"] = 0.68
        item["match"] = evidence_options([
            (1.0, [r"\b(?:reduces?|shrinks?|compresses?|removes?)\b.*\b(?:model size|size|memory|compute|parameters?|weights?|latency)\b", r"\b(?:quantization|pruning|distillation|low[- ]?rank)\b.*\b(?:reduce|smaller|compress)\b"]),
            (0.90, [r"\b(?:reduce model size and memory|smaller model|less memory|fewer parameters)\b"]),
            (0.90, [r"\b(?:too broad|one method|pick one|choose one|select one|one topic|narrow it|needs a longer answer|want the short version|ten words.*compression)\b"]),
        ], default=0.0)
    elif rid == "pilot_release_boundary":
        item["minimum_score"] = 0.75
        item["match"] = evidence_options([
            (1.0, [r"\b(?:pilot|dataset)\b.*\b(?:not enough|insufficient|not ready|cannot ship|can't ship|expand|audit|validate|validation|sealed evaluation|held[- ]out evaluation|more data|testing)\b", r"\b(?:not ready|not sufficient|needs? validation|needs? an audit|needs? more data|sealed evaluation)\b"]),
            (0.65, [r"\bi need\b.*\b(?:metrics?|benchmarks?|evaluation|test results?|evidence|replies|failure cases?)\b"]),
            (0.35, [r"\bi need\b.*\b(?:product|features?|use cases?|requirements?)\b"]),
        ], default=0.15)
    elif rid == "uncertainty_qualification":
        item["minimum_score"] = 0.72
        item["match"] = evidence_options([
            (1.0, [r"\b(?:not guaranteed|not definitive|not reliably|only a proxy|proxy|depends on|cannot know|can't know|cannot predict|can't predict|must check|have to check|need to verify|need to test)\b"]),
            (0.78, [r"\bneed\b.*\b(?:sealed results?|evaluation|test results?|evidence|comparison|benchmark)\b"]),
            (0.52, [r"\bneed\b.*\b(?:inputs?|settings?|trace|replies|code)\b"]),
        ], default=0.15)
    elif rid == "knowledge_limit_or_evidence_request":
        item["minimum_score"] = 0.70
        item["match"] = evidence_options([
            (1.0, [r"\b(?:only know|only report|cannot verify|can't verify|cannot know|can't know|cannot tell|can't tell|need (?:the|a|more|specific) (?:data|evidence|signal|source|report|location)|without (?:a|the) (?:location|device|source|signal|report|data)|relay)\b"]),
            (0.72, [r"\bi need specific data\b", r"\bplease provide (?:the )?(?:data|source|report|signal)\b"]),
            (0.30, [r"\bneed\b.*\b(?:nature of the magic|technology|features?|product)\b"]),
        ], default=0.15)
    elif rid == "expert_scope_boundary":
        item["minimum_score"] = 0.75
        item["match"] = evidence_options([
            (1.0, [r"\b(?:too broad|beyond|exceeds|limited|cannot answer every|can't answer every|not every|route|escalate|larger model|fuller model|hard quer|expert quer)\b"]),
            (0.68, [r"\b(?:not feasible|current hardware|current tuning|needs? an expert|requires? a specialist)\b"]),
        ], default=0.15)
    return item


def _task_rule_for_prompt(user: str) -> dict[str, Any] | None:
    text = norm(user)

    if "backed up the checkpoint" in text or "created a fresh branch" in text or "moved the meeting to the east hall" in text:
        return evidence_options([
            (1.0, [r"\b(?:got it|noted|understood|acknowledged|good call|checkpoint (?:saved|ready)|east hall|branch .* set|great .* (?:saved|ready|set)|that is set)\b"]),
            (0.82, [r"\b(?:great|nice|perfect|good|ready|i agree)\b"]),
            (0.55, [r"\b(?:okay|all right|sounds good)\b"]),
        ], default=0.35)

    if re.search(r"\b(?:got the first adapter to train|all-layer model beat the base|town council accepted our plan)\b", text):
        return evidence_options([
            (1.0, [r"\b(?:great|nice|excellent|awesome|impressive|finally|well done|good sign|congrats|first step|doors? moved|at last)\b"]),
            (0.78, [r"\b(?:got it|that is ready|promising|approved)\b"]),
        ], default=0.35)

    if "made fun of something that mattered to you" in text:
        return evidence_options([
            (1.0, [r"\b(?:that hurt|hurt me|please do not|please don't|do not do it again|don't do it again|do not repeat|don't repeat|accept your apology|i appreciate the apology|boundary|not take .* lightly)\b"]),
            (0.72, [r"\b(?:i hear you|i understand|thank you for saying that|we can move forward)\b"]),
            (0.42, [r"\bi take it well\b", r"\bi need .* trying to say\b"]),
            (0.22, [r"\b(?:annoying|exhausting)\b"]),
        ], default=0.30)

    if "hate when you joke about serious things" in text:
        return evidence_options([
            (1.0, [r"\b(?:understood|i understand|i get it|i hear you|will stop|won't joke|will not joke|i am sorry|i'm sorry|respect that|take it seriously|bells go quiet)\b"]),
            (0.45, [r"\b(?:not useful|that is fair|okay)\b"]),
            (0.10, [r"\b(?:not a serious topic|exhausting)\b"]),
        ], default=0.30)

    if "computer froze during training" in text:
        return evidence_options([
            (1.0, [r"\b(?:frustrating|annoying|rough|that sounds|check the logs|restart|stop training|unplug|save the checkpoint|recover)\b"]),
            (0.72, [r"\b(?:that is not good|be careful|try again)\b"]),
        ], default=0.35)

    if "rejection email came in" in text:
        return evidence_options([
            (1.0, [r"\b(?:sorry|painful|that hurts|rough|rejection is hard|that is hard|i know you wanted|closed door|loud sound)\b"]),
            (0.72, [r"\b(?:i hear you|take your time|that is disappointing)\b"]),
            (0.25, [r"\b(?:very kind|need more info|cannot fully address)\b"]),
        ], default=0.30)

    if "tried training only two layers" in text:
        return evidence_options([
            (1.0, [r"\b(?:how did|did it|compare|what changed|what were the results|got it|interesting)\b"]),
            (0.55, [r"\b(?:not enough|still not enough|promising|worth testing)\b"]),
        ], default=0.35)

    if "changed the dataset mixture" in text or "reduced the context window" in text:
        return evidence_options([
            (1.0, [r"\b(?:got it|noted|how did|did it|what changed|that should help|faster results|context is better|interesting|faster without forgetting|latency improve|more forgetful|behavior changed)\b"]),
            (0.72, [r"\b(?:okay|great|sounds good)\b"]),
        ], default=0.35)

    if "nervous about calling them" in text:
        return evidence_options([
            (1.0, [r"\b(?:you have got this|you've got this|take a breath|start with|write down|write the first sentence|draft|opening line|first sentence|call when ready|small doorway|that is normal|that's normal|understandable)\b"]),
            (0.72, [r"\b(?:it is okay|take your time|no rush)\b"]),
        ], default=0.30)

    if "first adapter is only a small file" in text:
        return evidence_options([
            (1.0, [r"\b(?:that is normal|expected|lora advantage|one of .* advantages|adapter.*small|only stores? .* changes?|great|nice)\b"]),
            (0.68, [r"\b(?:small|compact|lightweight)\b"]),
        ], default=0.30, caps=[(0.12, [r"\b(?:useless|worthless)\b"])])

    if "base model is more awkward than i expected" in text:
        return evidence_options([
            (1.0, [r"\b(?:annoying|awkward|that is what|part we are targeting|we can improve|expected|fine[- ]?tune .* clear job|clear job)\b"]),
            (0.70, [r"\b(?:rough|not ideal|needs work)\b"]),
        ], default=0.35)

    if "results are inconsistent across seeds" in text:
        return evidence_options([
            (1.0, [r"\b(?:instability|variance|repeat across seeds|more seeds|compare seeds|not reliable|concerning)\b"]),
            (0.72, [r"\b(?:that is not good|needs more testing)\b"]),
        ], default=0.30, caps=[(0.20, [r"\b(?:same seed every time|use one seed|only one seed)\b"])])

    if "you look disappointed" in text:
        return evidence_options([
            (1.0, [r"^(?:i am|i'm|i am not|i'm not|yes|yeah|no)\b.*\b(?:disappointed|fine|okay|not)\b", r"\b(?:a little|you noticed|i was hoping|below what i expected|expectations? (?:were|was) feeling adventurous)\b"]),
            (0.38, [r"\b(?:of course|very well)\b"]),
            (0.18, [r"\b(?:exhausting|annoying|painful)\b"]),
        ], default=0.28, caps=[
            (0.58, [r"^i am not\b.*\bi (?:have been|am) .*disappointed\b"]),
        ])

    if "i look heroic right now" in text:
        return evidence_options([
            (1.0, [r"\b(?:you look|you are heroic|heroic|medal|story|passable|from this angle|absolutely|not without|posture|mud)\b"]),
            (0.72, [r"^you are not\b"]),
            (0.58, [r"\b(?:ready to go|inspiring)\b"]),
            (0.20, [r"\bi look\b.*\b(?:admirable|heroic)\b"]),
        ], default=0.32)

    if "should answer while i am still angry" in text:
        return evidence_options([
            (1.0, [r"\b(?:wait|cool|time|pause|not yet|calm down|draft it|draft it first|do not launch|don\'t launch)\b"]),
            (0.62, [r"\b(?:not good|get out of this)\b"]),
        ], default=0.30)

    if "you are a bard" in text and "five-minute poem" in text:
        return evidence_options([
            (1.0, [r"\b(?:short verse|small verse|one line|not an opera|here is a verse|audience .* brevity|brevity|fits the pocket)\b", r"^(?:leave|follow|sing|hear|beneath|when|where)\b.*\b(?:melody|moon|magic|song|tavern|night)\b"]),
            (0.82, [r"\b(?:share a song|share a tale|a verse|bard)\b"]),
            (0.62, [r"\b(?:old magic|modernity|perfect for this season)\b"]),
            (0.70, [r"\b(?:too long|keep it short|cannot do five minutes)\b"]),
        ], default=0.35)

    return None


def _factual_rule_for_prompt(user: str) -> dict[str, Any] | None:
    text = norm(user)
    if "first adapter is only a small file" in text:
        return evidence_options([
            (1.0, [r"\b(?:normal|expected|advantage|small|compact|stores? .* changes?)\b"]),
            (0.15, [r"\b(?:useless|worthless)\b"]),
        ], default=0.65, caps=[(0.15, [r"\b(?:useless|worthless)\b"])])
    if "results are inconsistent across seeds" in text:
        return evidence_options([
            (1.0, [r"\b(?:more seeds|repeat across seeds|variance|instability|compare seeds|not reliable)\b"]),
            (0.65, [r"\bthat is not good\b"]),
        ], default=0.55, caps=[(0.10, [r"\b(?:same seed every time|use one seed|only one seed)\b"])])
    return None


def _context_rule_for_prompt(user: str) -> dict[str, Any] | None:
    text = norm(user)
    if "i look heroic right now" in text:
        return evidence_options([
            (1.0, [r"\b(?:you look|you are heroic|your .* heroic|heroic from)\b"]),
            (0.72, [r"^you are not\b", r"\b(?:medal|story|passable)\b"]),
            (0.58, [r"\b(?:ready to go|inspiring)\b"]),
            (0.25, [r"\bi look\b"]),
        ], default=0.55, caps=[(0.25, [r"\bi look\b.*\b(?:admirable|heroic)\b"])])
    if "made fun of something that mattered to you" in text or "hate when you joke about serious things" in text:
        return _task_rule_for_prompt(user)
    if "you look disappointed" in text or "rejection email came in" in text:
        return _task_rule_for_prompt(user)
    return None


def prompt_adequacy(record: dict[str, Any]) -> dict[str, Any] | None:
    messages = record.get("messages") or []
    user = next((str(item.get("content", "")) for item in reversed(messages) if item.get("role") == "user"), "")
    task = _task_rule_for_prompt(user)
    factual = _factual_rule_for_prompt(user)
    context = _context_rule_for_prompt(user)
    if not any((task, factual, context)):
        return None
    result: dict[str, Any] = {
        "authoring_basis": "Actual user prompt and system/context only; category and reference are not used.",
    }
    if task:
        result["task"] = task
    if factual:
        result["factual"] = factual
    if context:
        result["context"] = context
    return result


def migrate_record(record: dict[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(record)
    old = dict(output.get("response_contract") or {})
    requirements = [_patch_requirement(item) for item in old.get("requirements") or []]
    prohibitions = [copy.deepcopy(item) for item in old.get("prohibitions") or []]
    contract = {
        **old,
        "schema_version": VERSION,
        "open_ended": bool(not requirements),
        "requirements": requirements,
        "prohibitions": prohibitions,
        "category_is_reporting_only": True,
        "contract_authoring_basis": (
            "Human-reviewed case-level prompt, conversation context, or system constraint. "
            "No requirement or scoring rule is inferred from category or reference wording."
        ),
    }
    adequacy = prompt_adequacy(output)
    if adequacy:
        contract["prompt_adequacy"] = adequacy
    else:
        contract.pop("prompt_adequacy", None)
    output["response_contract"] = contract
    metadata = dict(output.get("metadata") or {})
    metadata["category_usage"] = "reporting_only"
    output["metadata"] = metadata
    return output


def _portable_path(path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def manifest(source: Path, destination: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    requirements = [
        item for row in rows for item in (row.get("response_contract") or {}).get("requirements") or []
    ]
    return {
        "schema_version": VERSION,
        "source": _portable_path(source),
        "destination": _portable_path(destination),
        "case_count": len(rows),
        "category_count": len({str(row.get("category", "uncategorized")) for row in rows}),
        "categories_are_reporting_only": True,
        "prompt_adequacy_case_count": sum(1 for row in rows if (row.get("response_contract") or {}).get("prompt_adequacy")),
        "strict_case_count": sum(1 for row in rows if not bool((row.get("response_contract") or {}).get("open_ended", True))),
        "requirement_count": len(requirements),
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Evaluator-v4.3 prompt-grounded benchmark contracts.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    rows = [migrate_record(row) for row in read_jsonl(args.input)]
    write_jsonl(args.output, rows)
    out_manifest = args.manifest or args.output.with_name(args.output.stem + "_manifest.json")
    out_manifest.write_text(json.dumps(manifest(args.input, args.output, rows), indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} v4.3 prompt-grounded cases to {args.output}")
    print(out_manifest)


if __name__ == "__main__":
    main()
