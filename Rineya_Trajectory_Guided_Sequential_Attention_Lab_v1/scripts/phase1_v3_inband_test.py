"""Separate memorization failure from extrapolation failure.

v3's dev/test splits differ from train in TWO ways at once: unseen instances
AND harder registered ranges. So "train 0.24, dev 0.04" is ambiguous between:

  (a) memorization  - the model cannot solve a NEW instance even at the training
                      difficulty, so it memorised pairs rather than an algorithm;
  (b) extrapolation - the model solves new in-band instances fine but cannot
                      handle the harder dev/test bands.

The fix differs completely: (a) needs vastly more training data, (b) needs
curriculum or a narrower evaluation band.

This generates a held-out set from the EXACT train-band generator settings with
a different dataset seed, so difficulty is held fixed and only instance novelty
varies.
"""

from __future__ import annotations

import json
import pathlib
import statistics
import sys
from collections import Counter

LAB = pathlib.Path(r"C:\Users\clclc\BusinessProject\Chat-Bubble-tgsa-launch-audit\Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1")
sys.path.insert(0, str(LAB))

import torch

from tgsa.phase1_data import SyntheticVocabulary, fixed_batches, generate_split, load_examples, validate_dataset_manifest
from tgsa.phase1_model import answer_logits, build_model
from tgsa.phase1_runner import _model_config_from_study, load_study_config, load_study_examples

ROOT = pathlib.Path(r"C:\Users\clclc\AppData\Local\Temp\claude\C--Users-clclc-BusinessProject-Chat-Bubble\8b037530-92dc-40fa-93b0-7d2d5ea2749e\scratchpad\v3full")
STUDY, DATA = ROOT / "study_gpu", ROOT / "dataset"
CONFIG = load_study_config(LAB / "configs" / "phase1_identifiability_v3.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = ("cascade", "history_control", "learned_delta")
SEEDS = CONFIG["seeds"][:4]


def load_arm(arm: str, seed: int):
    run = STUDY / "runs" / arm / f"seed_{seed}"
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    dm = validate_dataset_manifest(DATA / "dataset_manifest.json")
    max_len = max(int(e["max_tokens"]) - 1 for e in dm["splits"].values())
    config = _model_config_from_study(CONFIG, arm=arm, vocabulary=dm["vocabulary"], max_sequence_length=max_len)
    model = build_model(config, initialization_seed=int(manifest["initialization_seed"])).to(DEVICE)
    payload = torch.load(run / "checkpoint_final.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval()


@torch.no_grad()
def accuracy(model, examples, max_len: int) -> float:
    usable = [e for e in examples if len(e.tokens) - 1 <= max_len]
    correct = total = 0
    for batch in fixed_batches(usable, batch_size=128):
        batch = batch.to(DEVICE)
        out = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
        logits = answer_logits(out.logits, batch.answer_input_indices)
        correct += int((logits.argmax(-1) == batch.answer_tokens).sum())
        total += len(batch.answer_tokens)
    return correct / max(total, 1)


def main() -> None:
    dataset_config = CONFIG["dataset"]
    dm = validate_dataset_manifest(DATA / "dataset_manifest.json")
    max_len = max(int(e["max_tokens"]) - 1 for e in dm["splits"].values())
    vocabulary = dm["vocabulary"]
    vocab = SyntheticVocabulary(
        domain_size=int(vocabulary["domain_size"]),
        max_hops=int(vocabulary["max_hops"]),
        max_abs_step=int(vocabulary["max_abs_step"]),
    )
    splits = load_study_examples(DATA)
    train_all = load_examples(DATA / "train.jsonl")
    train_signatures = {e.latent_signature for e in train_all}

    # Held-out at TRAIN difficulty: same family settings, different dataset seed.
    inband_spec = {
        "families": dataset_config["families"],
        # deliberately no split_overrides -> the train band
    }
    inband = generate_split(
        split="train",
        count_by_family={"direct_lookup": 300, "pointer_chase": 600, "state_update": 600},
        dataset_seed=987654321,
        vocab=vocab,
        generation_spec=inband_spec,
        forbidden_signatures=train_signatures,
    )
    overlap = len({e.latent_signature for e in inband} & train_signatures)
    print(f"in-band held-out generated: {len(inband)} examples, overlap with train = {overlap}")
    print(f"train band: pointer hops {dataset_config['families']['pointer_chase']['min_hops']}-"
          f"{dataset_config['families']['pointer_chase']['max_hops']}, "
          f"nodes {dataset_config['families']['pointer_chase']['min_nodes']}-"
          f"{dataset_config['families']['pointer_chase']['max_nodes']}")
    ov = dataset_config["split_overrides"]["test"]["pointer_chase"]
    print(f"test  band: pointer hops {ov['min_hops']}-{ov['max_hops']}, nodes {ov['min_nodes']}-{ov['max_nodes']}")
    print()

    answers = [e.answer_token for e in inband]
    print(f"in-band baselines: uniform={1/vocab.domain_size:.4f} "
          f"majority={Counter(answers).most_common(1)[0][1]/len(answers):.4f}")
    print()

    print("=== THE DISCRIMINATOR ===")
    print(f"{'arm':16s} {'train(seen)':>12s} {'inband(new)':>12s} {'dev(harder)':>12s} {'test(hardest)':>14s}")
    for arm in ARMS:
        rows = {"train": [], "inband": [], "dev": [], "test": []}
        for seed in SEEDS:
            model = load_arm(arm, seed)
            rows["train"].append(accuracy(model, train_all[:1200], max_len))
            rows["inband"].append(accuracy(model, inband, max_len))
            rows["dev"].append(accuracy(model, splits["dev"], max_len))
            rows["test"].append(accuracy(model, splits["test"], max_len))
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        print(f"{arm:16s} "
              f"{statistics.mean(rows['train']):12.4f} "
              f"{statistics.mean(rows['inband']):12.4f} "
              f"{statistics.mean(rows['dev']):12.4f} "
              f"{statistics.mean(rows['test']):14.4f}")

    print()
    print("Reading: train >> in-band  => memorization (needs vastly more data).")
    print("         train ~ in-band >> dev/test => extrapolation (needs curriculum/band change).")


if __name__ == "__main__":
    main()
