# Reproducibility Checklist

## Before the run

- [ ] Confirm Anchor and Greedy are full floating-point checkpoints from the same 12-layer lineage.
- [ ] Confirm the tokenizer and chat template are identical.
- [ ] Freeze the development and sealed JSONL files and record their hashes.
- [ ] Predeclare the objective, alpha grids, layer set, search budget, seeds, protocols, and finalist strategy.
- [ ] Create a new run directory; never recycle a completed run for changed settings.
- [ ] Keep the sealed split inaccessible to tuning decisions.

## During development selection

- [ ] Inspect `manifests/compatibility.json` and resolve every error.
- [ ] Inspect endpoint raw replies and endpoint complementarity before interpreting a soup gain.
- [ ] Preserve the full global interpolation curve, not only its winner.
- [ ] Preserve every intervention row and coordinate-search decision.
- [ ] Treat bootstrap probability as uncertainty evidence, not proof against adaptive-search bias.
- [ ] Review category/tail regressions and the generated human-review packet.

## Before sealed evaluation

- [ ] Verify `selection/selection_freeze.json` exists.
- [ ] Confirm the frozen recipe, checkpoint digest, evaluator hash, objective, and development hash.
- [ ] Confirm no coefficient or finalist decision uses test evidence.
- [ ] Commit or archive the entire run directory.

## After sealed evaluation

- [ ] Do not alter coefficients in the same experiment.
- [ ] Report endpoints, uniform, global, and layer-wise controls together.
- [ ] Report negative and null results.
- [ ] Replicate on a fresh split/seed before making a strong layer-mechanism claim.
- [ ] Keep automated scores separate from human review and deployment approval.
