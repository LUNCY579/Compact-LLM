# Rineya retention-first quality queue

This is an offline evidence index and proposal review tool. It never loads models,
tokenizers or weights, discovers task banks, dispatches owner workers, trains,
evaluates native models, promotes candidates or changes serving policy.

It preserves the five programme priorities: CE8 qualification, one-donor pruning
versus acquisition-matched scalar attenuation, matched MLP at the actual live
bias3.75, reviewed data/ordinary-SFT parent, then the RSD three-arm step2 screen.
Existing lane owners and their registrations remain authoritative. RSD, DSD and
RDD are distinct experiments. SRCL and twelve-draw pruning expansion are deferred.

## Build and inspect

From repository root, use an existing interpreter; no dependencies are needed:

```powershell
$py = "C:/Users/clclc/BusinessProject/.venv-mq001/Scripts/python.exe"
& $py -B -S -m experiments.rineya_quality_queue_v1 build --output "C:/Users/clclc/BusinessProject/rineya-quality-queue-NEW/packet"
& $py -B -S -m experiments.rineya_quality_queue_v1 verify --packet "C:/Users/clclc/BusinessProject/rineya-quality-queue-NEW/packet"
```

Build is create-only and rejects output inside any tracked checkout or `.codex`.
Inputs are explicitly listed source/protocol/receipt/proposal metadata, not weight
payloads or task banks. Existing source worktrees under `.codex` may be inspected
read-only; every new output belongs under BusinessProject outside tracked source.
Run files and logs remain immutable. Choose a new packet after source drift.

`QUEUE.json` fixes priority, owner, blockers and a bounded next proposal for each
lane. Five `*.quality-card.json` files carry exact file hashes, source/runtime and
candidate status, historical measurements with denominators, exposure, registered
rules, and current preservation gates. `REVIEW.txt` summarizes the queue.
`evidence/` contains byte-identical listed receipts/protocols; `specs/` freezes the
card definitions and `software/` preserves the queue implementation. `MANIFEST.json`
binds all packet files. Models, tokenizers and independently supplied raw-bank or
response files are refused; source receipts may retain their historical diagnostics.

Verification checks both immutable capsules and current source bytes by default.
It replays numeric assertions and asset metadata against capsules, compares cards
with their frozen specifications, and checks every path and nested permission flag.
`--capsules-only` can audit old packet integrity, while explicitly making no claim
that the current source still matches. Hashes prove byte equality, not the truth
of source assertions or that a model executed.

## Evidence and advancement contract

Fixture/preparation evidence cannot support measured model-quality numbers.
Absent measurements require null numerator/denominator and `NOT_MEASURED`;
unknown is not zero. JSON assertions resolve an exact pointer; text assertions
retain the exact supporting excerpt and denominator. Historical outcomes stay
historical and never fill the new candidate's gates.
Prose carrying only evidence IDs is an indexed interpretation, rather than a
mechanically checked statement or semantic certification.

Every required gate and every extra registered preservation gate is mandatory.
Acquisition, retention, reliable answering, termination, response length, exact
native qualification and independent replication remain separate. Missing or
failed evidence yields `HOLD`; gains in one axis cannot compensate for another.
Even all-PASS fixture gate metadata yields only eligibility for a separate review,
with native/release/promotion permissions false. The built programme snapshot
keeps every current proposed candidate on HOLD.

Source-specific retention, original source-correct cohorts, independent replay,
family/parent split closure, reference review, complete target+EOS, no silent
truncation, independent training draws, stopped/incomplete attempts and uncertain
spending retain their original contracts. Do not relabel a spent bank as fresh,
an unfinished E1/SYNTH run as an SFT parent, or proposed resource ceilings as
measured prices. Sealed/CHECK/FINAL access and serving changes stay separately gated.

## Tests

```powershell
& $py -B -m unittest discover -s experiments/rineya_quality_queue_v1/tests -v
```

Tests use synthetic metadata/opaque bytes and temporary directories under
BusinessProject. They verify contracts and source binding only; no Rineya
capability, memory fit, throughput or native qualification follows from them.
