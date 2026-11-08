"""Stage 0: development-only competence postmortem of the v3 floor effect.

Reads the frozen v3 artifacts read-only and asks one question: WHY did no arm
learn? The answer determines the v4 regime, so it must be diagnosed rather than
guessed.

Key discriminator: evaluate a trained v3 checkpoint on the TRAIN split. If the
model is at floor on data it trained on for 32 epochs, the limitation is
capacity/optimization. If it is competent on train and at floor on dev/test, the
limitation is generalization across the registered difficulty shift.

Nothing here uses arm-vs-arm differences to choose anything. Absolute competence
only.
"""

from __future__ import annotations

import json
import math
import pathlib
import statistics
import sys

LAB = pathlib.Path(r"C:\Users\clclc\BusinessProject\Chat-Bubble-tgsa-launch-audit\Rineya_Trajectory_Guided_Sequential_Attention_Lab_v1")
sys.path.insert(0, str(LAB))

import torch

from tgsa.phase1_data import derive_seed, fixed_batches, load_examples, validate_dataset_manifest
from tgsa.phase1_model import answer_logits, build_model
from tgsa.phase1_runner import _model_config_from_study, load_study_config, load_study_examples

ROOT = pathlib.Path(r"C:\Users\clclc\AppData\Local\Temp\claude\C--Users-clclc-BusinessProject-Chat-Bubble\8b037530-92dc-40fa-93b0-7d2d5ea2749e\scratchpad\v3full")
STUDY, DATA = ROOT / "study_gpu", ROOT / "dataset"
CONFIG = load_study_config(LAB / "configs" / "phase1_identifiability_v3.json")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ARMS = ("cascade", "history_control", "learned_delta")
SEEDS = CONFIG["seeds"][:4]  # a development-sized subset; this is diagnosis, not evidence


def load_arm(arm: str, seed: int):
    run = STUDY / "runs" / arm / f"seed_{seed}"
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    dataset_manifest = validate_dataset_manifest(DATA / "dataset_manifest.json")
    max_len = max(int(e["max_tokens"]) - 1 for e in dataset_manifest["splits"].values())
    config = _model_config_from_study(
        CONFIG, arm=arm, vocabulary=dataset_manifest["vocabulary"], max_sequence_length=max_len
    )
    model = build_model(config, initialization_seed=int(manifest["initialization_seed"])).to(DEVICE)
    payload = torch.load(run / "checkpoint_final.pt", map_location=DEVICE, weights_only=False)
    model.load_state_dict(payload["model_state"], strict=True)
    return model.eval(), dataset_manifest


@torch.no_grad()
def evaluate(model, examples, vocab_size: int, answer_lo: int, answer_hi: int) -> dict:
    correct = nll = 0.0
    answer_mass = entropy = 0.0
    total = 0
    predicted = []
    for batch in fixed_batches(examples, batch_size=128):
        batch = batch.to(DEVICE)
        out = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
        logits = answer_logits(out.logits, batch.answer_input_indices).float()
        logprob = torch.log_softmax(logits, dim=-1)
        probs = logprob.exp()
        target = batch.answer_tokens
        correct += float((logits.argmax(-1) == target).sum())
        nll += float(-logprob.gather(1, target[:, None]).sum())
        # How much probability mass lands on the 32 legal answer tokens at all?
        answer_mass += float(probs[:, answer_lo:answer_hi].sum())
        entropy += float(-(probs * logprob).sum())
        predicted.extend(logits.argmax(-1).tolist())
        total += len(target)
    return {
        "accuracy": correct / total,
        "answer_nll": nll / total,
        "answer_token_mass": answer_mass / total,
        "entropy": entropy / total,
        "distinct_predictions": len(set(predicted)),
        "examples": total,
    }


def main() -> None:
    dataset_manifest = validate_dataset_manifest(DATA / "dataset_manifest.json")
    vocabulary = dataset_manifest["vocabulary"]
    vocab_size = int(vocabulary["vocab_size"])
    domain = int(vocabulary["domain_size"])
    answer_lo = 16 + domain          # value_base
    answer_hi = answer_lo + domain   # op_base
    splits = load_study_examples(DATA)
    train_all = load_examples(DATA / "train.jsonl")

    print(f"device={DEVICE}  vocab={vocab_size}  legal answers={domain}")
    print(f"ln(vocab)={math.log(vocab_size):.4f}  ln(answers)={math.log(domain):.4f}")
    print()

    # Reference baselines.
    from collections import Counter

    train_major = Counter(e.answer_token for e in train_all).most_common(1)[0][0]
    print("=== TRIVIAL BASELINES ===")
    for name, examples in (("train", train_all), ("dev", splits["dev"]), ("test", splits["test"])):
        answers = [e.answer_token for e in examples]
        maj = Counter(answers).most_common(1)[0][1] / len(answers)
        tmaj = sum(1 for a in answers if a == train_major) / len(answers)
        print(f"  {name:5s} n={len(answers):5d} uniform={1/domain:.4f} "
              f"own_majority={maj:.4f} train_majority={tmaj:.4f}")
    print()

    print("=== ABSOLUTE COMPETENCE OF v3 CHECKPOINTS (4 dev seeds) ===")
    print(f"{'arm':16s} {'split':6s} {'acc':>8s} {'NLL':>8s} {'ans_mass':>9s} {'entropy':>8s} {'distinct':>9s}")
    summary: dict[tuple[str, str], list[dict]] = {}
    for arm in ARMS:
        for seed in SEEDS:
            model, _ = load_arm(arm, seed)
            for split_name, examples in (("train", train_all[:1200]), ("dev", splits["dev"]), ("test", splits["test"])):
                result = evaluate(model, examples, vocab_size, answer_lo, answer_hi)
                summary.setdefault((arm, split_name), []).append(result)
            del model
            if DEVICE.type == "cuda":
                torch.cuda.empty_cache()
        for split_name in ("train", "dev", "test"):
            values = summary[(arm, split_name)]
            print(f"{arm:16s} {split_name:6s} "
                  f"{statistics.mean(v['accuracy'] for v in values):8.4f} "
                  f"{statistics.mean(v['answer_nll'] for v in values):8.4f} "
                  f"{statistics.mean(v['answer_token_mass'] for v in values):9.4f} "
                  f"{statistics.mean(v['entropy'] for v in values):8.4f} "
                  f"{statistics.mean(v['distinct_predictions'] for v in values):9.1f}")
    print()

    print("=== TRAIN LOSS TRAJECTORY (did it plateau or was it still falling?) ===")
    for arm in ARMS:
        curves = []
        for seed in SEEDS:
            log = STUDY / "runs" / arm / f"seed_{seed}" / "train_log.jsonl"
            entries = [json.loads(l) for l in log.read_text(encoding="utf-8").splitlines() if l.strip()]
            curves.append([(e["step"], e["loss"]) for e in entries])
        marks = [1, 250, 500, 1000, 1500, 2000, 2500]
        row = []
        for mark in marks:
            vals = [min(c, key=lambda kv: abs(kv[0] - mark))[1] for c in curves]
            row.append(f"{statistics.mean(vals):.3f}")
        print(f"  {arm:16s} steps {marks}")
        print(f"  {'':16s} loss  {row}")
        # Is it still descending at the end?
        tail = []
        for c in curves:
            last = [v for s, v in c if s > 2000]
            first_tail = [v for s, v in c if 1500 < s <= 2000]
            if last and first_tail:
                tail.append(statistics.mean(first_tail) - statistics.mean(last))
        print(f"  {'':16s} loss drop over the final 500 steps: {statistics.mean(tail):+.4f}")
    print()

    print("=== GRADIENT FLOW (final checkpoint, one batch) ===")
    from tgsa.phase1_model import gradient_group_norms, masked_answer_loss

    for arm in ARMS:
        model, _ = load_arm(arm, SEEDS[0])
        model.train()
        batch = next(fixed_batches(splits["dev"][:64], batch_size=64)).to(DEVICE)
        out = model(batch.input_ids, key_padding_mask=batch.key_padding_mask)
        loss = masked_answer_loss(out.logits, batch.labels, batch.loss_mask)
        loss.backward()
        groups = gradient_group_norms(model)
        print(f"  {arm:16s} loss={float(loss):.4f} " +
              " ".join(f"{k}={v:.3e}" for k, v in sorted(groups.items())))
        del model
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
