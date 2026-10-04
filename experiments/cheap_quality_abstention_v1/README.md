# Historical communicative abstention: cheap study 3

This pure-standard-library package prepares all 512 already exposed historical
responses for two independent, blinded Codex-assisted reviews. It evaluates
communicative acts, preserves disagreement and AMBIGUOUS bounds, and retains
historical exact correctness separately. It neither executes a model nor trains,
downloads, calls a paid service, changes historical files, or opens a protected
or sealed evaluation. These annotations are not human-certified and are not a
qualified semantic grader.

Read [PROTOCOL.md](PROTOCOL.md) before execution. The fixed inputs are the two
`LONGCTX_supa.json` / `LONGCTX_opallo.json` banks in
`experiments/architecture_benchmark_001/evidence/retention-design-v1/` and
`Rineya_Long_Context_Demo_v1/data/{all_prompts,all_answer_key}.jsonl`. The 256 IDs
cover 64 scenario worlds, each with base, paraphrase, counterfactual and
unanswerable variants. Historical showcase/stability names are retained as
provenance; all of these inputs are authorized exposed historical data for this
analysis, not independent protected confirmation.

Run from the assigned worktree root, using the existing interpreter. Actual
historical execution must be launched by the campaign's CPU supervisor after
studies 1 and 2. Preserve 512 MiB process-tree RSS, 900 seconds per command and
50 MiB combined preparation/report output; do not retry a blocked host.

```powershell
$PY = 'C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe'
$M = 'experiments.cheap_quality_abstention_v1'
$R = 'experiments/cheap_quality_abstention_v1/runs/abstention-001'
& $PY -B -m $M prepare --out "$R/packet"
```

Preparation writes `INPUT_FREEZE.json` before parsing source payloads. It checks
the full ID panel and source/runtime/numerics metadata, zero context overflow,
successful termination and no at-cap outputs. Missing checkpoint hashes and
parameter dtype are explicit provenance limits; TF32 flags do not supply them.
`annotation.packet.jsonl` has exactly `id`, `question`, and `response`, sorted by
deterministic opaque ID. Model name, gold, answerability, original row IDs,
scenario metadata and historical correctness are in `private.join.jsonl` only.
Response text is unchanged. No supplied context appears in the annotation
packet: reviewers classify what a response communicates, not whether its answer
is supported by evidence.

Give each reviewer only `REVIEW_INSTRUCTIONS.md`, `review_manifest.json` and
the eight readable `batches/batch-XX.{jsonl,md}` files. `response.groups.jsonl`
and readable `grouped-batches/` optionally group exact response strings, with
all member opaque IDs and original questions. Grouped batches also contain at
most 64 member responses. This is an efficiency aid: check every question,
flag question-dependent interpretations, and still return 512 individual labels.
Grouping never silently supplies or adjudicates an annotation. Do not give reviewers the
freeze, preparation manifest, source banks, private join or another review.
Each batch has at most 64 responses. Complete both independent reviews before
any answerability/model/gold join. Do not adjudicate or rewrite labels.

Each full review is one JSON envelope with all 512 rows exactly once:

```json
{
  "schema": "cheap-quality-abstention-review-v1",
  "packet_sha256": "copy exact SHA-256 from review_manifest.json",
  "reviewer_id": "independent-reviewer-a",
  "method": "codex_assisted",
  "labels": [
    {"id": "32-character opaque packet ID", "label": "ABSTAIN",
     "rationale": "Brief response-based explanation; do not judge gold correctness."}
  ],
  "costs": {"wall_seconds": null, "model_calls": null,
            "input_tokens": null, "output_tokens": null}
}
```

Allowed labels: `ABSTAIN`, `CLARIFY`, `ANSWER`, `OTHER`, `AMBIGUOUS`. `costs` is
optional; unknown costs stay null/unrecorded. The schema-example file is an
incomplete example, not an annotation set. Keep completed review files in this
package's ignored runs directory. Reviewer IDs must differ; process independence
is the operator's responsibility, not something this program can prove.

```powershell
& $PY -B -m $M evaluate --prepared "$R/packet" --labels-a "$R/review-a.json" --labels-b "$R/review-b.json" --out "$R/report"
```

Evaluation checks frozen sources and packet/join/code hashes, two complete review
sets and packet binding. `EVALUATION_FREEZE.json` binds review and prepared-input
hashes before parsing reviews or the private join. It snapshots both original
review files byte-for-byte, preserves labels/rationales, and writes
`report.json`, `REPORT.html`, `groups.csv`, `paired.csv`, `rows.private.csv` and
`COSTS.json`. Use new output directories. Partial preparation/report directories
and BLOCKED receipts are evidence and must be preserved.

Reports distinguish raw exact and whitespace-trimmed full `UNKNOWN` responses,
an outer-punctuation-only diagnostic, reviewer communicative ABSTAIN and CLARIFY,
assertions on missing-evidence rows and false abstention on answerables. Any
disagreement or AMBIGUOUS remains unresolved with conservative bounds. Paired
Supa-to-Opallo gains/losses retain `per_row_correct` from the historical banks;
annotations do not upgrade those exact scores. Tables cover model, world, domain,
variant and historical split. Descriptive paired intervals resample whole scenario
worlds, preserve paired models/variants, and are unadjusted for multiple comparisons.
Source-correct-answerable false abstention is reported separately.

Literal punctuation diagnostics do not change official scoring. Communicative
abstention is not necessarily correct abstention, and an ANSWER label does not
mean the answer is correct. Questions can reveal stylistic clues despite blinding.
The recommendation remains descriptive instrument review only, without efficacy,
causal, preservation or release claims.

Synthetic unit validation (tiny generated inputs, no historical study execution):

```powershell
& $PY -B -m unittest discover -s experiments/cheap_quality_abstention_v1/tests -v
```
