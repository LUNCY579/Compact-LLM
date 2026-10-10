# Durable registry and source intake

Current registry: `registry/`. Current anchored tail is recorded in the cycle
status and source-lineage correction receipt. All snapshots are immutable and
hash chained. Existing pilot registration is retrospective, not a new claim of
prospective construction.

## What it now records

72 pilot parents, their 72 old-label aliases, 72 rendered rows and native prompt
fingerprints, one reviewed release, and 30 scoped structural clusters. Four
legacy TRAIN file containers and an external exclusion inventory are metadata
only; zero legacy examples have been admitted. Content-source families and their
ancestry inherit one split. Shared generator/domain metadata may support multiple
independent worlds and is explicitly distinct from content ancestry.

An independent registry review passed 72 adversarial probes and its unit suite
passed 60 tests. A separate adapter review passed 78 checks after the correction
below. Eight additional fingerprint/anchor checks passed. These are different
scopes; do not combine their counts into a row-quality or model-intelligence score.

## Corrected source adapter

The first seed adapter grouped sources by intake measurement source_id. The
grounding chat derivative has a different source_id from its canonical source,
so the adapter missed their ancestry. Source hashes were correct and all records
were UNQUALIFIED; no imported rows were created. Preserve that script as a
historical attempt rather than a reusable future intake adapter.

The current catalog is
`source_lineage_correction_v2/CURRENT_SOURCE_CATALOG.json`. It selects
`legacy-grounded-chat-lineage-corrected-v1` and supersedes the descriptor
`legacy-train-file-2` for current intake. The corrected descriptor retains its
original byte-family label/hash and declares both canonical and previous
descriptor ancestry. This joins split inheritance through the ancestry graph.
The first attempted correction tried changing the byte-family label and was
correctly rejected with no commit; its failure and source bytes remain evidence.

Snapshot 2 adds exactly one source descriptor. It changes no parent, actor row,
target, role, release or original source bytes. Do not count the corrected
descriptor as a new independent training example or additional corpus.

## Verify before future work

From the foundry root:

```powershell
python cycles/002-source-registry/verify_registry.py --registry registry --anchors cycles/002-source-registry/registry_seed/receipt.json cycles/002-source-registry/source_lineage_correction_v2/receipt.json
python -m unittest registry_tools.test_registry -q
python -m unittest discover -s cycles/002-source-registry -p "test_*.py" -v
```

Use the pinned Python runtime documented in README. Full replay validates
snapshot bytes, schemas, chronology and ancestry. External local receipts anchor
known commits, so a valid shorter replay cannot silently hide loss of the known
tail. A filesystem owner can still rewrite all local history/receipts; these are
ordinary drift controls, not execution or identity authentication.

## Reserve new work

Use `make_source`, `make_structure` and `make_parent` to stage complete unsolved
construction specifications. Source content ancestry must be TRAIN-safe within
the intended data-research role. Structure descriptors must encode actual
inference structure, not labels or incidental prose. Parent specifications have
no target, completed solution, rendered messages or student outcome.

Reserve the parent in an earlier registry transaction than its actor rows.
Then independently render/solve, review every final public row, measure exact
tokenizer/label/EOS boundaries, and register rows/releases using the existing
reserved parent. The transaction order proves local registration order only;
orchestration/source review must establish that query solving/scoring happened
after reservation. Demo/source-fact construction calculations need an explicit
definition in each mechanism contract.

All declared mutations, renamings, length views and other derivatives retain their
role, ancestry and canonical grouping. Same-parent alternate targets may be
legitimate QA failures or corrupt data; the generic registry does not verify
answer semantics. Task verifiers and row reviews remain mandatory.

## Boundaries and future qualification

The external v1 inventory has unknown cross-schema fingerprint identity and is
registered UNQUALIFIED with no active known-match identities. No historical
comparison or clearance was performed. Normalization and graph fingerprints have
declared coverage; they do not prove generic semantic or pretraining exclusion.

Raw-source safety is immutable. Do not turn an UNQUALIFIED/EXCLUDED source into
TRAIN_SAFE by a new ID, family name or revision. Later genuine qualification needs
an explicitly designed append-only clearance/assessment record bound to the same
source content and ancestry, with its required correctness/provenance/native
evidence. This version has no implicit clearance upgrade. A future extension must
retain the old producer/schema bytes and verify old snapshots rather than editing
their meanings. Until then, work on original qualified research sources and leave
legacy corpora unadmitted.

The next mechanism plan is DESIGN_ONLY. A frozen contract and independent design
review must precede world construction, prospective reservation and any new row.
