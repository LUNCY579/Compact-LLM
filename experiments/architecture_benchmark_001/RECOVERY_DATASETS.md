# Recovery datasets: what to use, obtain and freeze

## Recommendation for this particular experiment

First repair/qualify execution; then test recovery on a **small, reviewed, broad corpus with a common frozen official teacher and matched ordinary controls**. Do not begin by flooding only C40 with a large reasoning instruction set. Improvement then mixes architecture recovery, domain adaptation and extra training budget.

Recovery after structural changes is an empirically supported strategy in Minitron [1], but that is not evidence that this specific CED transplant will recover in a given number of local GPU hours or tokens. A 1–2M-token initial campaign is a **proposed starting budget**, not a validated optimum or an enabled G3 command. Freeze any dose ladder before inspecting held-out student outcomes. The delivered optimizer commands are exactly 25-step engineering checks only.

## What is already included

`data/calibration.jsonl` has 156 rows and `data/train-smoke.jsonl` has 25 rows. The original provenance says they come from the user's synthetic, procedural Rineya corpus, with human review pending. Keep them for bounded engineering and review their source/provenance before scientific use. They are not a substitute for broad language recovery, instruction evaluation or independent held-out tasks. `pilot.jsonl` is an open engineering diagnostic; never relabel it as training data.

**You do not need another dataset to rerun the arms.** For a serious recovery campaign, a reviewed broad training pool is needed. It can be your own appropriately licensed source-distribution material; an external download is not mandatory when such material already exists locally.

## Source priorities

**First choice: reviewed material matching the source model's existing behavior.** Use training-permitted general prose and reviewed natural-communication examples with the official formatting. This is a proposed distribution-preservation choice, not a claim that the original Supa pretraining corpus is available in this ZIP. Retain broad language and instruction-following regression checks, separately held out. Keep each source document, repository, template family and generated sibling family together.

**Useful external general-text source: FineWeb-Edu.** Its card describes educational English web text, with document identifiers and URL metadata; smaller sample configurations can be streamed. It declares ODC-By and CommonCrawl terms. Code is underrepresented, so it is not a complete code-retention corpus. Use a bounded locally saved subset, not a whole multi-billion-token download. Dataset-level licensing is not blanket clearance for every underlying page; preserve attribution/source metadata and review applicable terms [2].

**Optional synthetic supplement: SmolLM-Corpus / Cosmopedia-v2.** This provides generated educational prose and generation-prompt metadata. Treat it as a supplement, not guaranteed ground truth. Verify numerical and factual material; its published token_length uses a different tokenizer, so calculate budgets using the official Supa tokenizer. Group related seed documents/prompts rather than assigning an independent group to every generated row [3].

**Code supplement later:** start with a small reviewed training-permitted corpus with repository-level groups. Do not assume the SmolLM `python-edu` subset is immediately usable source text: its listed fields identify code blobs and repositories, and the card's example retrieves contents separately from Software Heritage's S3 infrastructure. That introduces separate retrieval and license/provenance work [3]. It is not the lowest-friction first recovery source.

I would initially favor reviewed source-distribution/general text, then add a modest explicitly controlled synthetic or task-specific component. There is no established optimal mixture for C28/C40, so this package does not invent one. Keep mixture selection separate from the architecture comparison and charge every tuning run to the same search budget.

## Where to obtain public data

- FineWeb-Edu: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu
- SmolLM-Corpus, including `cosmopedia-v2` and `fineweb-edu-dedup`: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

The code package performs **no network download**. Dataset acquisition belongs in a separate environment so it does not change the pinned qualification runtime. The following is an example for that environment, not a command already run or a bundled dataset. Review the card/terms first, install the official `datasets` and `huggingface_hub` libraries, and record their exact resolved versions. The dataset API is documented in [4].

```python
# Run outside the pinned experiment environment. This downloads a bounded sample.
# It does NOT grant review approval, establish independence, or launch training.
import hashlib
import json
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

repo = "HuggingFaceFW/fineweb-edu"
config = "sample-10BT"
revision = HfApi().dataset_info(repo).sha  # resolve once, then freeze this commit
if not revision:
    raise RuntimeError("Could not resolve an immutable dataset revision")
output = Path("fineweb_edu_raw_pool.jsonl")
if output.exists():
    raise FileExistsError(output)
stream = load_dataset(repo, name=config, split="train", streaming=True, revision=revision)
# Seeded buffer shuffle gives a deterministic bounded stream at a fixed revision;
# it is not a uniform sample over all documents.
stream = stream.shuffle(seed=17, buffer_size=10000)
written = 0
with output.open("x", encoding="utf-8", newline="\n") as f:
    for row in stream:
        text, doc_id, url = row.get("text"), row.get("id"), row.get("url")
        if not isinstance(text, str) or not text.strip() or not doc_id or not url:
            continue
        # Starting document-level grouping only. Review/cluster related URLs and
        # near-duplicates before making a final scientific group assignment.
        group = "url-" + hashlib.sha256(str(url).encode()).hexdigest()
        record = {
            "id": "fineweb-" + str(doc_id), "group_id": group, "split": "train",
            "source": repo, "source_revision": revision,
            "license": "ODC-By-1.0; CommonCrawl terms; underlying-source review required",
            "source_url": str(url), "review_status": "pending", "text": text,
        }
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
        if written >= 2000:  # a bounded acquisition example, not a token budget
            break
if written == 0:
    raise RuntimeError("No eligible documents were acquired")
Path("fineweb_edu_acquisition.json").write_text(json.dumps({
    "repository": repo, "config": config, "revision": revision, "rows": written,
    "seed": 17, "buffer_size": 10000,
    "raw_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    "review_status": "pending", "licenses_training": False,
}, indent=2) + "\n", encoding="utf-8")
```

Inspect the acquired records, remove unwanted/private/evaluation material, preserve complete source information, group near-duplicates and related families, and document approval before using a scientific recovery pool. Do not change `review_status` to approved merely to silence a validator. Dataset card text is not a substitute for content review.

For Cosmopedia, change the repository/config explicitly and build a schema-aware exporter for its fields; do not pretend it has FineWeb's `id`/`url` fields or treat a source-category label as an independent family identifier. Save generation prompts and source/seed relationships where available. The provided exporter is intentionally not a generic silent fallback.

## Local raw JSONL schema accepted by prepare-data

```json
{"id":"doc-001","group_id":"source-document-family-001","split":"train","source":"your-source-identifier","source_revision":"immutable-source-revision-or-content-hash","license":"actual-license-and-review-notes","review_status":"approved","text":"A complete reviewed training-permitted text."}
```

Every record needs nonempty id, group_id, source, source_revision, license and text. The runner treats review/license values as declarations, not verified legal facts. It refuses to relabel supplied evaluation/calibration rows into a training pool. Use an upstream immutable acquisition manifest to retain metadata beyond the minimal required fields.

`prepare-data` assigns **independent groups before windowing**, with expected 90% train / 5% calibration / 5% open DEV under a stable hash and seed. Small pools can produce empty splits; those are explicitly reported, not fabricated. This split is a preparation default, not a complete preregistered benchmark. No sealed TEST is read or produced.

Each document is tokenized with the exact official tokenizer, receives its document EOS, and is windowed at <=512 tokens. One-token overlap initially represents each within-document adjacent target once. Exact duplicate windows are then removed deterministically, keeping open DEV before calibration before training; dropped windows/target counts are reported. This can remove recurring short tails rather than leaking them across splits. Exact duplicate detection is not semantic contamination detection.

Saved input_ids, not decode-and-retokenize text, are authoritative. Tokenizer hash and a tokenization schema marker are required. The text remains for inspection. Counts distinguish input tokens from supervised shifted targets. No cross-document packing or silent truncation occurs. Windows do not preserve all context preceding their document slice; this is explicitly a short-context preparation recipe.

## Teacher targets and CED-specific recovery

Use the exact official Supa source as the common frozen teacher for initial controlled comparisons. Do not silently switch to a larger teacher for C40 alone. Cache targets from the exact same token IDs, masks, splits and source hash used for student training. The delivered feature cache reconstructs full teacher vocabulary probabilities; arbitrary top-K targets are not accepted.

Same teacher + same training rows + same 25-step recipe is enforced in the engineering controls. A scientific comparison additionally needs frozen token/search budgets, continuation controls, a recovered ordinary donor plan, three seeds, regression bounds and held-out endpoints. Width pruning, depth expansion, topology replacement and continued training are distinct factors: matching total parameters alone does not separate them.

Long-context retrieval/interference and multi-step reasoning are useful **later evaluation hypotheses**, not implied successes. The current runner is registered for <=512 tokens, not 8K/32K. Do not simply increase a CLI context limit and claim the same qualification. Train on independently generated/curated families; never use sealed evaluation questions, answers or templates for recovery.

## What is intentionally not supplied

No full external corpus, paid data access, Hugging Face credentials, official source weights, teacher cache, multi-million-token trained checkpoint, or G3 campaign runner is in the ZIP. `recovery_plan.example.json` leaves scientific/resource choices null rather than inventing approvals. Source hashes and program checks reduce accidental drift; they are not cryptographic attestations or a substitute for human data review.

## References

[1] Muralidharan et al., Compact Language Models via Pruning and Knowledge Distillation: https://arxiv.org/abs/2407.14679

[2] FineWeb-Edu dataset card, especially loading, limitations and licensing: https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu

[3] SmolLM-Corpus dataset card, especially Cosmopedia-v2 fields and Python-Edu content retrieval: https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

[4] Hugging Face Datasets loading and streaming documentation: https://huggingface.co/docs/datasets/loading and https://huggingface.co/docs/datasets/stream

Public cards/documentation consulted September 10, 2026 (America/Chicago). Freeze an immutable revision and the locally acquired file hash for your actual run; this report does not pin or download one on your behalf.
