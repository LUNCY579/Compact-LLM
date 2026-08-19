#!/usr/bin/env python3
"""Build a self-contained HTML showcase from model predictions."""

from __future__ import annotations

import argparse
import html
import json
import re
from pathlib import Path
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"^\s*(final\s+answer|answer)\s*:\s*", "", text)
    text = re.sub(r"[“”\"'`]", "", text)
    text = re.sub(r"[^a-z0-9\s-]", " ", text)
    text = re.sub(r"\b(the|a|an)\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def prediction_text(row: dict[str, Any]) -> str:
    for field in ("prediction", "response", "output", "generated_text", "answer"):
        if field in row:
            return str(row[field])
    return ""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", default="showcase")
    parser.add_argument("--variant", default="base")
    parser.add_argument("--model-name", default="Rineya")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    full_rows = load_jsonl(args.full)
    predictions = {
        str(row["id"]): prediction_text(row)
        for row in load_jsonl(args.predictions)
        if "id" in row
    }

    selected = [
        row
        for row in full_rows
        if row.get("split") == args.split and row.get("variant") == args.variant
    ]
    if args.limit is not None:
        selected = selected[: args.limit]

    cards: list[str] = []
    correct_count = 0
    covered_count = 0

    for row in selected:
        prediction = predictions.get(str(row["id"]))
        covered = prediction is not None
        accepted = {normalize(value) for value in row["accepted_answers"]}
        correct = covered and normalize(prediction) in accepted
        covered_count += int(covered)
        correct_count += int(correct)

        status = "Correct" if correct else ("Missing" if not covered else "Incorrect")
        status_class = "correct" if correct else ("missing" if not covered else "incorrect")
        context_html = "<br><br>".join(
            html.escape(paragraph) for paragraph in row["context"].split("\n\n")
        )
        cards.append(
            f"""
            <article class="card">
              <div class="card-head">
                <div>
                  <h2>{html.escape(row['title'])}</h2>
                  <p class="meta">{html.escape(row['id'])} · {html.escape(row['domain_label'])}</p>
                </div>
                <span class="status {status_class}">{status}</span>
              </div>
              <div class="badges">
                <span>{html.escape(row['task_type'])}</span>
                <span>{html.escape(row['difficulty'])}</span>
                <span>{html.escape(row['needle_profile'])} evidence</span>
                <span>{row['context_words']} words</span>
              </div>
              <details>
                <summary>Read the full context</summary>
                <div class="context">{context_html}</div>
              </details>
              <section class="qa">
                <div>
                  <h3>Question</h3>
                  <p>{html.escape(row['question'])}</p>
                </div>
                <div>
                  <h3>{html.escape(args.model_name)} answer</h3>
                  <p class="answer">{html.escape(prediction if covered else 'No prediction')}</p>
                </div>
                <div>
                  <h3>Expected answer</h3>
                  <p class="answer">{html.escape(row['answer'])}</p>
                </div>
              </section>
            </article>
            """
        )

    accuracy = correct_count / covered_count if covered_count else 0.0
    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(args.model_name)} Long-Context Showcase</title>
<style>
:root {{
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #171717;
  background: #f5f5f4;
}}
body {{ margin: 0; }}
header {{
  padding: 48px max(24px, calc((100vw - 1100px) / 2));
  background: #111827;
  color: white;
}}
header h1 {{ margin: 0 0 12px; font-size: clamp(2rem, 5vw, 4rem); }}
header p {{ margin: 8px 0; max-width: 800px; line-height: 1.6; }}
.summary {{ font-size: 1.1rem; font-weight: 700; }}
main {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px 64px; }}
.card {{
  background: white;
  border: 1px solid #dedede;
  border-radius: 18px;
  padding: 24px;
  margin: 0 0 24px;
  box-shadow: 0 8px 30px rgba(0,0,0,.05);
}}
.card-head {{ display: flex; justify-content: space-between; gap: 16px; align-items: start; }}
h2 {{ margin: 0; }}
.meta {{ color: #666; }}
.status {{ padding: 7px 11px; border-radius: 999px; font-weight: 700; white-space: nowrap; }}
.correct {{ background: #dcfce7; color: #166534; }}
.incorrect {{ background: #fee2e2; color: #991b1b; }}
.missing {{ background: #e5e7eb; color: #374151; }}
.badges {{ display: flex; flex-wrap: wrap; gap: 8px; margin: 16px 0; }}
.badges span {{ background: #f3f4f6; border-radius: 999px; padding: 6px 10px; font-size: .85rem; }}
details {{ border-top: 1px solid #e5e7eb; border-bottom: 1px solid #e5e7eb; padding: 14px 0; }}
summary {{ cursor: pointer; font-weight: 700; }}
.context {{ margin-top: 16px; line-height: 1.72; color: #303030; }}
.qa {{ display: grid; grid-template-columns: 1.4fr 1fr 1fr; gap: 18px; margin-top: 20px; }}
.qa h3 {{ font-size: .82rem; text-transform: uppercase; letter-spacing: .05em; color: #6b7280; }}
.answer {{ font-size: 1.1rem; font-weight: 750; }}
footer {{ text-align: center; padding: 24px; color: #666; }}
@media (max-width: 760px) {{
  .qa {{ grid-template-columns: 1fr; }}
  .card-head {{ flex-direction: column; }}
}}
</style>
</head>
<body>
<header>
  <h1>{html.escape(args.model_name)} Long-Context Showcase</h1>
  <p>Original closed-context stories with short exact answers. The contexts use invented entities so the result cannot be explained by memorizing a familiar franchise.</p>
  <p class="summary">Covered: {covered_count}/{len(selected)} · Correct: {correct_count}/{covered_count if covered_count else 0} · Exact accuracy: {accuracy:.1%}</p>
</header>
<main>
{''.join(cards)}
</main>
<footer>Rineya Long-Context Demonstration Suite</footer>
</body>
</html>
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(page, encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
