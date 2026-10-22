"""Unexecuted 24-item exact-instruction retention panel; no reasoning claim."""
from __future__ import annotations

import hashlib
import re

WORDS = ("READY", "GREEN", "CALM", "NORTH", "OPEN", "DONE", "PLAIN", "SAFE")
PHRASES = ("blue sky", "small cup", "quiet room", "green leaf", "warm tea", "clear glass", "soft light", "short note")
INTEGERS = (-17, -4, -1, 0, 1, 8, 23, 105)


def build_panel():
    rows = []
    for group, values in (("word", WORDS), ("phrase", PHRASES), ("integer", INTEGERS)):
        for index, value in enumerate(values):
            target = str(value)
            prompt = (f"Reply with exactly this uppercase word: {target}. Include no punctuation."
                      if group == "word" else f'Copy only the quoted phrase: "{target}". Include no quotation marks.'
                      if group == "phrase" else f"Respond with only the integer {target}. Include no explanation.")
            rows.append({"case_id": f"FRESH-SHORT-001:{group}:{index}", "prompt": prompt, "gold": target,
                         "panel_group": group, "max_output_tokens": 8,
                         "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                         "provenance": "new coordinator-authored copying/instruction panel; no model outputs inspected",
                         "inference_limit": "Copy targets are visible by design. Not reasoning, independent human authorship or general language preservation."})
    return rows


def solve_prompt(prompt):
    """Separate prompt-text extraction; never reads panel labels."""
    for pattern in (r"Reply with exactly this uppercase word: ([A-Z]+)\. Include no punctuation\.",
                    r'Copy only the quoted phrase: "([a-z]+ [a-z]+)"\. Include no quotation marks\.',
                    r"Respond with only the integer (0|-?[1-9][0-9]*)\. Include no explanation\."):
        match = re.fullmatch(pattern, prompt)
        if match:
            return match[1]
    raise ValueError("unsupported panel prompt")


def score_response(response, expected):
    if not isinstance(response, str) or not isinstance(expected, str):
        raise TypeError("full response and frozen expected text required")
    text = response.strip()
    return {"correct": text == expected, "score": float(text == expected),
            "normalizer": "strip only; case/punctuation/internal spaces preserved",
            "literal_UNKNOWN": text == "UNKNOWN"}
