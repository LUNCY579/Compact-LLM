"""Run from this experiment directory: python -m rgi --help."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from . import data, metrics, native, protocol, importer
from .io import artifact, digest, load, require, score_rows, seal, write_new


def demo(out: str) -> dict:
    root = Path(out)
    root.mkdir(parents=True, exist_ok=False)
    bank = data.encode_bank(data.mini_bank(), data.FixtureTokenizer(), digest("fixture-char-v1"))
    values = [[2 + (int(digest([x["item_id"], i])[:8], 16) % 100) / 50
               for i, role in enumerate(x["roles"]) if role != "prompt"] for x in bank["items"]]
    base_rows = score_rows(bank, values)
    host, ca, cb = digest("fixture-host"), digest("fixture-A"), digest("fixture-B")
    arms = {}
    for name in ("base", "uniform", "scaled", "A", "B", "AB"):
        rows = []
        for row in base_rows:
            a = -0.6 if row["family"] == "ledger_mini" and row["role"] == "content" else 0
            b = -0.4 if row["family"] == "register_mini" and row["role"] == "content" else 0
            harmful = 0.3 if row["family"] == "prerequisite_mini" else 0
            value = {"base": row["nll"], "uniform": row["nll"] - 0.1, "scaled": row["nll"] * 2,
                     "A": row["nll"] + a, "B": row["nll"] + b, "AB": row["nll"] + a + b + harmful}[name]
            rows.append({**row, "nll": value})
        components = {"A": {ca: 1.0}, "B": {cb: 0.25}, "AB": {ca: 1.0, cb: 0.25}}.get(name, {})
        arms[name] = artifact(bank, rows, name, host, components)
        write_new(root / f"{name}.json", arms[name])
    write_new(root / "bank.json", bank)
    reports = {name: metrics.compare(arms["base"], arms[name], repeats=500) for name in ("uniform", "scaled", "A", "AB")}
    reports["interaction"] = metrics.interaction(arms["base"], arms["A"], arms["B"], arms["AB"], repeats=500)
    checks = {"constant_shift_residual_zero": reports["uniform"]["token_profile"]["residual_rms"] < 1e-12,
              "correlation_is_not_invariance": reports["scaled"]["token_profile"]["pearson"] > 0.999
                  and reports["scaled"]["token_profile"]["centered_ccc"] < 0.9,
              "interaction_is_detected": reports["interaction"]["interaction_rms"] > 0.1}
    require(all(checks.values()), "Synthetic diagnostic canary failed")
    result = seal({"kind": "synthetic_validation", "evidence_class": "synthetic_fixture", "checks": checks,
                   "reports": reports, "native_execution": False, "quality_improvement_established": False})
    write_new(root / "demo.json", result)
    (root / "REPORT.md").write_text(protocol.render_report(reports["AB"]), encoding="utf-8")
    return {"directory": str(root), "checks": checks, "evidence_class": "synthetic_fixture"}


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="RGI-001: evaluate loss profiles without conflating them with skills")
    sub = p.add_subparsers(dest="command", required=True)
    def command(name, *fields):
        q = sub.add_parser(name)
        for field in fields:
            q.add_argument("--" + field.replace("_", "-"), required=True)
        return q
    command("demo", "out")
    q = command("mini-bank", "out")
    q.add_argument("--groups", type=int, default=16)
    q.add_argument("--seed", type=int, default=17)
    q.add_argument("--split", choices=["train", "development", "confirmation"], default="development")
    q = command("fingerprint", "path", "kind")
    q = command("encode", "raw", "out")
    q.add_argument("--spec")
    q.add_argument("--fixture", action="store_true")
    command("import-losses", "bank", "rows", "receipt", "out")
    q = command("match-controls", "bank", "out")
    q.add_argument("--pairs", type=int, required=True)
    q.add_argument("--seed", type=int, default=17)
    q = command("compare", "reference", "candidate", "out")
    q.add_argument("--repeats", type=int, default=1000)
    q.add_argument("--seed", type=int, default=17)
    command("interaction", "base", "a", "b", "ab", "out")
    command("preflight", "spec", "bank", "out")
    q = command("collect", "spec", "bank", "out")
    q.add_argument("--execute", action="store_true")
    q = command("control-plan-matrix", "targeted", "control", "spec", "out")
    q.add_argument("--seeds", type=int, nargs="+", default=[17, 23, 29])
    command("train-plan", "spec", "bank", "out")
    q = command("train", "spec", "bank", "plan", "out")
    q.add_argument("--execute", action="store_true")
    command("freeze", "inputs", "settings", "ledger", "out")
    q = command("collect-frozen", "freeze", "out")
    q.add_argument("--execute", action="store_true")
    q = command("confirm", "freeze", "reference", "candidate", "out")
    q.add_argument("--reference-outcomes")
    q.add_argument("--candidate-outcomes")
    return p


def run(args) -> dict:
    c = args.command
    if c == "demo":
        return demo(args.out)
    if c == "fingerprint":
        return {"sha256": native.fingerprint(args.path, args.kind)}
    if c == "mini-bank":
        result = data.mini_bank(args.groups, args.seed, args.split)
    elif c == "encode":
        raw = load(args.raw)
        require(bool(args.fixture) != bool(args.spec), "Choose exactly one of --fixture or --spec")
        if args.fixture:
            require(raw["evidence_class"] == "synthetic_fixture", "Fixture tokenizer cannot encode native evidence")
            result = data.encode_bank(raw, data.FixtureTokenizer(), digest("fixture-char-v1"))
        else:
            spec = load(args.spec)
            require(raw["evidence_class"] == spec["evidence_class"], "Encoding evidence class mismatch")
            result = data.encode_bank(raw, native.local_tokenizer(spec), spec["tokenizer_sha256"], spec["max_length"])
    elif c == "import-losses":
        result = importer.import_losses(load(args.bank), args.rows, load(args.receipt))
    elif c == "match-controls":
        result = data.match_controls(load(args.bank), args.pairs, args.seed)
        root = Path(args.out)
        root.mkdir(parents=True, exist_ok=False)
        for name, value in result.items():
            write_new(root / f"{name}.json", value)
        return {"directory": str(root), **result["receipt"]}
    elif c in {"compare", "interaction"}:
        paths = [args.reference, args.candidate] if c == "compare" else [args.base, args.a, args.b, args.ab]
        models = [load(path) for path in paths]
        require(all(all(r["split"] == "development" for r in a["rows"]) for a in models),
                "Confirmation requires the frozen, one-read confirm command")
        result = metrics.compare(*models, args.repeats, args.seed) if c == "compare" else metrics.interaction(*models)
    elif c == "control-plan-matrix":
        result = native.control_plan_matrix(load(args.targeted), load(args.control), load(args.spec), args.seeds)
        root = Path(args.out)
        root.mkdir(parents=True, exist_ok=False)
        for name, arm in result["arms"].items():
            write_new(root / (name + ".spec.json"), arm["spec"])
            write_new(root / (name + ".plan.json"), arm["plan"])
        write_new(root / "matrix.json", result)
        return {"directory": str(root), "arms": list(result["arms"]), "training_executed": False}
    elif c in {"preflight", "collect", "train-plan", "train"}:
        bank, spec = load(args.bank), load(args.spec)
        if c == "preflight":
            native.check_bank_spec(bank, spec)
            result = {"inputs": native.inspect_spec(spec), "native_execution": False}
            native.resource_guard(spec)
        elif c == "collect":
            require(args.execute, "Collection requires --execute; no implicit model/GPU execution")
            require(all(x["split"] == "development" for x in bank["items"]),
                    "Use collect-frozen for confirmation; do not probe its bank interactively")
            result = native.collect(bank, spec)
        elif c == "train-plan":
            result = native.training_plan(bank, spec)
        else:
            return native.train(bank, spec, load(args.plan), args.out, args.execute)
    elif c == "freeze":
        result = protocol.freeze(load(args.inputs), load(args.settings), args.ledger)
    elif c == "collect-frozen":
        return protocol.collect_frozen(load(args.freeze), args.out, args.execute)
    elif c == "confirm":
        result = protocol.confirm(load(args.freeze), args.reference, args.candidate,
                                  args.reference_outcomes, args.candidate_outcomes)
    else:
        raise ValueError("Unknown command")
    write_new(args.out, result)
    return {"output": args.out, "kind": result.get("kind"), "verdict": result.get("verdict")}


def main(argv=None) -> int:
    args = parser().parse_args(argv)
    try:
        print(json.dumps(run(args), indent=2, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError, KeyError, TypeError) as exc:
        print(f"RGI-001 refused: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
