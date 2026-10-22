"""Deterministic, end-to-end random-fixture demonstration with four interface-only updates."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import torch
from torch import nn

from .alignment import AffineFit, fit_bridge
from .artifacts import new_directory, save_bridge, save_capture, write_json
from .contracts import Capture, RowKey, Site, digest, require
from .evaluation import Outcome, Panel, paired_report
from .fixtures import decoder
from .modules import (Bridge, FFNGraft, GraftedDecoder, StitchedDecoder, fingerprint,
                      interface_parameters, masked_next_token_loss, composition_descriptor)
from .planning import plan


def run(output: Path) -> dict:
    output = new_directory(output)
    small, large = decoder(), decoder(12, 6, 29, heads=6, kv_heads=2, head_dim=4)
    before = [fingerprint(small), fingerprint(large)]
    generator = torch.Generator().manual_seed(1701)
    registration = digest("XSG-001 random-fixture demonstration, not a reviewed native bank")
    banks = {}
    for role, count in (("FIT", 12), ("MAP_CHECK", 4)):
        keys, sx, lx = [], [], []
        for i in range(count):
            ids = torch.randint(0, 23, (1, 8), generator=generator)
            label = f"{role}-{i}"
            s = small.capture(ids, 1, "block_output_raw")[0]
            l = large.capture(ids, 2, "block_output_raw")[0]
            sx.append(s)
            lx.append(l)
            keys.extend(RowKey(label, label, digest([label, "prompt"]), digest(ids.tolist()), j, int(ids[0, j]))
                        for j in range(ids.shape[1]))
        banks["small_" + role] = Capture(small.identity, Site(1, "block_output_raw", 8), role,
                                          registration, tuple(keys), torch.cat(sx))
        banks["large_" + role] = Capture(large.identity, Site(2, "block_output_raw", 12), role,
                                          registration, tuple(keys), torch.cat(lx))
    paths = {}
    for name, bank in banks.items():
        paths[name] = {"directory": name, "manifest_sha256": save_capture(bank, output / name)}
    source, target, cs, ct = banks["large_FIT"], banks["small_FIT"], banks["large_MAP_CHECK"], banks["small_MAP_CHECK"]
    l_to_s, ls_report = fit_bridge(source, target, cs, ct)
    s_to_l, sl_report = fit_bridge(target, source, ct, cs)
    save_bridge(l_to_s, ls_report, source, target, output / "large_to_small")
    save_bridge(s_to_l, sl_report, target, source, output / "small_to_large")
    write_json(output / "fit_request.json", {"source": paths["large_FIT"], "target": paths["small_FIT"],
                                             "check_source": paths["large_MAP_CHECK"], "check_target": paths["small_MAP_CHECK"]})
    ids = torch.tensor([[1, 4, 3, 7, 2]], dtype=torch.long)
    forward = StitchedDecoder(large, small, 3, 2, Bridge.from_fit(l_to_s, dtype=torch.float64))
    reverse = StitchedDecoder(small, large, 2, 3, Bridge.from_fit(s_to_l, dtype=torch.float64))
    stitch_shapes = [list(forward(ids).shape), list(reverse(ids).shape)]
    # Graft uses independently collected RAW POST-ATTENTION sites, not the stitch maps.
    graft_banks = {}
    for name, bank in banks.items():
        model = small if name.startswith("small") else large
        block_index = 1 if name.startswith("small") else 2
        features = []
        for start in range(0, len(bank.rows), 8):
            rowset = bank.rows[start:start + 8]
            seq = torch.tensor([[r.token_id for r in rowset]], dtype=torch.long)
            features.append(model.capture(seq, block_index, "post_attention_raw")[0])
        graft_banks[name] = Capture(bank.identity, Site(block_index, "post_attention_raw", bank.site.width),
                                   bank.role, bank.registration_sha256, bank.rows, torch.cat(features))
    sf, lf, sc, lc = (graft_banks[k] for k in ("small_FIT", "large_FIT", "small_MAP_CHECK", "large_MAP_CHECK"))
    up, _ = fit_bridge(sf, lf, sc, lc)
    down, _ = fit_bridge(lf, sf, lc, sc)
    graft = FFNGraft(large.blocks[2].norm2, large.blocks[2].mlp,
                     Bridge.from_fit(up, dtype=torch.float64), Bridge.from_fit(down, delta=True, dtype=torch.float64),
                     donor_identity=large.identity, donor_block_index=2, alpha=0.05)
    candidate = GraftedDecoder(small, 1, graft)
    bypass = GraftedDecoder(small, 1, graft, bypass=True)
    require(torch.equal(small(ids), bypass(ids)), "fixture bypass changed host")
    require(not torch.equal(small(ids), candidate(ids)), "nonzero fixture graft did not affect logits")
    params = interface_parameters(candidate, [*graft.input_bridge.parameters(), *graft.return_bridge.parameters()])
    optimizer = torch.optim.AdamW(params, lr=1e-3, weight_decay=0)
    target_mask = torch.tensor([[False, False, True, True, True]])
    losses = []
    for _ in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_next_token_loss(candidate(ids), ids, target_mask)
        loss.backward()
        require(all(p.grad is not None and torch.isfinite(p.grad).all().item() for p in params), "missing/nonfinite interface gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    after = [fingerprint(small), fingerprint(large)]
    require(before == after, "fixture source tensors changed")
    rows_a, rows_b = [], []
    for i in range(8):
        common = dict(example_id=str(i), group_id=str(i // 2), family="fictional-smoke", prompt_sha256=digest(i), input_ids_sha256=digest([i, "ids"]),
                      format_ok=True, unknown_expected=i == 7, literal_unknown_ok=True)
        rows_a.append(Outcome(**common, correct=i < 4))
        rows_b.append(Outcome(**common, correct=i < 5))
    panel = dict(registration_sha256=registration, scorer_sha256=digest("fixture-scorer"),
                 decoding_sha256=digest("fixture-decode"), role="SELECT", expected_ids=tuple(str(i) for i in range(8)))
    scoring = paired_report(Panel(small.identity, **panel, outcomes=tuple(rows_a)),
                            Panel(large.identity, **panel, outcomes=tuple(rows_b)))
    result = {"study": "XSG-001", "evidence_class": "RANDOM_FIXTURE_MECHANICS_ONLY",
              "pretrained_model_calls": 0, "native_execution": False,
              "backbone_tensors_unchanged": before == after, "exact_bypass": True,
              "nonzero_graft_changes_logits": True, "bidirectional_stitch_shapes": stitch_shapes,
              "composition_descriptors": [composition_descriptor(m) for m in (candidate, forward, reverse)],
              "fit_reports": {"large_to_small": ls_report, "small_to_large": sl_report},
              "fixture_optimizer_updates": 4, "fixture_losses_not_quality_results": losses,
              "fictional_outcome_scoring": scoring, "plan": plan(),
              "native_qualification": False, "efficacy_claim": False}
    write_json(output / "DEMO_RECEIPT.json", result)
    return result
