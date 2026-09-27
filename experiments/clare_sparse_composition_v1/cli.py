"""Offline qualification and explicitly gated native smoke; no full-training switch."""
import argparse
import csv
import itertools
import json
from pathlib import Path
from . import EXPERIMENT, PARENT_COMMIT
from .artifacts import save_checkpoint, tensor_hash, write_new_json
from .config import Config
from .data import TrainingRow, digest_json, read_json


def csv_new(path, fields, rows):
    with Path(path).open("x", newline="", encoding="utf-8") as out:
        writer = csv.DictWriter(out, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def synthetic(out, *, source_commit):
    import torch
    from dataclasses import replace
    from .training import SkillRunner
    from .metrics import layer_allocation, pair_geometry
    from .native import runtime
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    schema_digest = digest_json({"fixture": "four regression stand-ins, NOT R/U/S/P corpus"})
    rows = {s: [TrainingRow("toy-"+s, "world-"+s, s, "train", schema_digest,
                            json.dumps([{"role": "user", "content": "mechanics fixture "+s}]), "{}")]
            for s in "RUSP"}
    inputs = torch.eye(4)
    def loss(model, row):
        i = "RUSP".index(row.skill)
        return (model(inputs[i:i+1]) - (i+1)/4).square().mean()
    summaries, layers, geometry = [], [], []
    for arm in ("dense_sequential", "independent_add", "magnitude_sparse", "random_disjoint", "learned_disjoint"):
        torch.manual_seed(20260927)
        model = torch.nn.Sequential(torch.nn.Linear(4, 4, bias=False), torch.nn.Tanh(), torch.nn.Linear(4, 1, bias=False))
        cfg = replace(Config(), arm=arm, lr=0.01)
        runner = SkillRunner(model, ("0.weight", "2.weight"), cfg, data_digest=schema_digest)
        identity = {"model_id": "synthetic-20-coordinate-MLP-NOT-Rineya", "tokenizer_id": "none-synthetic",
                    "model_sha256": tensor_hash(runner.base), "tokenizer_sha256": digest_json("no tokenizer"),
                    "git_commit": source_commit, "runtime": runtime()}
        for skill in cfg.order:
            free = runner.allocation.free if "disjoint" in arm else torch.ones(runner.coords.size, dtype=torch.bool)
            rec = runner.train_skill(skill, rows[skill], loss)
            layers += [{"arm": arm, "skill": skill, **r} for r in layer_allocation(runner.coords, free, rec["mask"], runner.deltas[skill])]
        manifest = save_checkpoint(runner, out / arm, identity)
        for a, b in itertools.combinations(cfg.order, 2):
            geometry.append({"arm": arm, "skill_a": a, "skill_b": b, **pair_geometry(runner.deltas[a], runner.deltas[b])})
        summaries.append({"arm": arm, "skills_completed": len(runner.records),
                          "all_resets_exact": all(r["reset_exact"] for r in runner.records),
                          "budgets": [r["capacity_budget"] for r in runner.records],
                          "selected": [r["selected_coordinates"] for r in runner.records],
                          "state_sha256": manifest["state_sha256"]})
    report = {"experiment": EXPERIMENT, "status": "SYNTHETIC_MECHANICS_PASS_ONLY",
              "source_commit": source_commit, "parent_commit": PARENT_COMMIT, "runtime": runtime(),
              "measured": summaries, "native_model_loads": 0, "native_training_steps": 0,
              "rineya_acquisition": None, "rineya_retention": None, "rineya_composition": None,
              "geometry_is_not_functional_evidence": True}
    write_new_json(out / "results.json", report)
    csv_new(out / "allocation_by_layer.csv", list(layers[0]), layers)
    csv_new(out / "delta_geometry.csv", list(geometry[0]), geometry)
    csv_new(out / "retention_matrix.csv", ["arm", "seed", "skill", "base", "after_R", "after_U", "after_S", "after_P"], [])
    csv_new(out / "composition_matrix.csv", ["arm", "seed", "signature", "correct", "total", "parent_count"], [])
    (out / "RESULTS.md").write_text("# Synthetic qualification only\n\nAll five arms ran on a 20-coordinate MLP. No Rineya model, tokenizer or corpus was loaded. Empty retention/composition CSVs mean NOT RUN, not zero accuracy. Geometry and reset checks are mechanics, not efficacy.\n", encoding="utf-8")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("validate-config")
    p.add_argument("--config", type=Path, required=True)
    p = sub.add_parser("synthetic")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--source-commit", required=True, help="actual git commit whose source bytes were executed")
    for name in ("preflight", "native-smoke"):
        p = sub.add_parser(name)
        p.add_argument("--binding", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "native-smoke":
            p.add_argument("--config", type=Path, required=True)
            p.add_argument("--authorize-smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command == "synthetic":
            report = synthetic(args.out, source_commit=args.source_commit)
        elif args.command == "validate-config":
            cfg = Config.from_dict(read_json(args.config))
            report = {"status": "CONFIG_ONLY_PASS", "config_sha256": cfg.digest, "native_authorized": False}
        else:
            from .native import preflight, smoke
            if args.out.exists():
                raise ValueError("refusing output overwrite before any native work")
            binding = read_json(args.binding)
            report = preflight(binding) if args.command == "preflight" else smoke(binding,
                     Config.from_dict(read_json(args.config)), authorized=args.authorize_smoke)
            write_new_json(args.out, report)
        print(json.dumps(report, indent=2, allow_nan=False))
        return 2 if report["status"] == "BLOCKED" else 0
    except (ValueError, OSError, RuntimeError, KeyError, TypeError) as exc:
        report = {"status": "REFUSED", "error": str(exc), "native_model_loads": None,
                  "native_training_steps": None, "full_training_authorized": False}
        if args.command in ("preflight", "native-smoke") and not args.out.exists():
            write_new_json(args.out, report)
        print(json.dumps(report, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
