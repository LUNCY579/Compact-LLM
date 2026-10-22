"""Explicit command line: plan/build-data/smoke/qualify/train/evaluate/compare."""
import argparse
from dataclasses import replace
import json
from pathlib import Path

from .config import Plan, screening_plans, factorial_plans
from .data import audit_disjoint, make_bank, read_bank, write_bank
from .provenance import write_json_new, exclusive_run


def main(argv=None):
    parser = argparse.ArgumentParser(description="LGR-001, opt-in research only")
    sub = parser.add_subparsers(dest="command", required=True)
    plans = sub.add_parser("plan", help="Print configuration only; no model imports")
    plans.add_argument("--factorial-decay", type=float)
    build = sub.add_parser("build-data", help="Synthetic diagnostics, not a published benchmark")
    build.add_argument("--output", required=True)
    build.add_argument("--split", choices=["train", "dev", "confirmation"], required=True)
    build.add_argument("--seed", type=int, default=101)
    build.add_argument("--parents", type=int, default=4)
    build.add_argument("--lengths", type=int, nargs="+", default=[1024, 2048, 4096, 8192])
    build.add_argument("--checkpoint", help="Local tokenizer directory; no checkpoint weights are loaded")
    build.add_argument("--fixture", action="store_true")
    smoke = sub.add_parser("smoke", help="Tiny CPU mechanics; never native Rineya evidence")
    smoke.add_argument("--output", required=True)
    for name in ("qualify", "train", "evaluate"):
        native = sub.add_parser(name)
        native.add_argument("--checkpoint", required=True)
        native.add_argument("--plan", required=True)
        native.add_argument("--bank", required=True)
        native.add_argument("--eligibility", required=True)
        native.add_argument("--output", required=True)
        native.add_argument("--allow-native-load", action="store_true")
        native.add_argument("--allow-training", action="store_true")
        native.add_argument("--qualification")
        native.add_argument("--adapter")
        native.add_argument("--allow-confirmation", action="store_true")
    compare_parser = sub.add_parser("compare")
    compare_parser.add_argument("--reference", nargs="+", required=True)
    compare_parser.add_argument("--candidate", nargs="+", required=True)
    compare_parser.add_argument("--output", required=True)
    audit = sub.add_parser("audit-data")
    audit.add_argument("banks", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "plan":
        base = Plan(mode="vpad", probability=0.1)
        result = factorial_plans(base, args.factorial_decay) if args.factorial_decay is not None else screening_plans()
        print(json.dumps({k: p.to_dict() for k, p in result.items()}, indent=2))
    elif args.command == "build-data":
        if args.fixture == bool(args.checkpoint):
            parser.error("Choose exactly one of --fixture or --checkpoint")
        if args.fixture:
            from .fixtures import ByteTokenizer
            tokenizer = ByteTokenizer()
        else:
            from transformers import AutoTokenizer
            if not Path(args.checkpoint).is_dir():
                parser.error("Tokenizer must be an existing local directory")
            tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
        rows = make_bank(tokenizer, split=args.split, seed=args.seed, parents=args.parents,
                         lengths=tuple(args.lengths), fixture=args.fixture)
        write_bank(args.output, rows)
        print(json.dumps({"cases": len(rows), "fixture": args.fixture, "synthetic": True}))
    elif args.command == "audit-data":
        print(json.dumps(audit_disjoint(*(read_bank(p) for p in args.banks))))
    elif args.command == "smoke":
        from .fixtures import ByteTokenizer, TinyLM
        from .adapters import inject_lora
        from .training import train_bounded
        from .evaluation import collect, compare
        import torch
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            tokenizer = ByteTokenizer()
            train = make_bank(tokenizer, split="train", seed=71, parents=1, lengths=(256,), depths=(0.5,), fixture=True)
            dev = make_bank(tokenizer, split="dev", seed=73, parents=1, lengths=(256,), depths=(0.5,), fixture=True)
            audit_disjoint(train, dev)
            base = Plan(expected_layers=2, train_context=512, max_tokens=2048, max_steps=2,
                        accumulation=2, rank=2, lora_alpha=4, dtype="float32", gradient_checkpointing=False,
                        eval_lengths=(256,), max_new_tokens=3)
            outputs, reports = {}, {}
            with exclusive_run(args.output) as output:
                for name, plan in screening_plans(base).items():
                    torch.manual_seed(41)
                    model = TinyLM()
                    inject_lora(model, plan)
                    reports[name] = train_bounded(model, train, plan)
                    outputs[name] = collect(model, tokenizer, dev, seed=plan.seed, context_limit=512,
                                            max_new_tokens=3)
                report = {"evidence_level": "CPU_FIXTURE_ONLY", "native_checkpoint_loaded": False,
                          "training": reports, "comparison": compare(outputs["A_none"], outputs["C_vpad"], resamples=100)}
                write_json_new(output / "smoke.json", report)
                print(json.dumps({"status": "CPU_FIXTURE_COMPLETE", "output": str(output),
                                  "native_checkpoint_loaded": False}))
        finally:
            torch.set_num_threads(previous_threads)
    elif args.command == "compare":
        from .evaluation import compare
        contracts = set()
        def records(paths):
            data = []
            for path in paths:
                item = json.loads(Path(path).read_text(encoding="utf-8"))
                if not isinstance(item, dict) or "evaluation_contract" not in item:
                    raise ValueError("CLI comparison requires bound native evaluation reports")
                contracts.add(item["evaluation_contract"])
                data.extend(item["records"])
            return data
        before, after = records(args.reference), records(args.candidate)
        if len(contracts) != 1:
            raise ValueError("Mismatched evaluation contracts")
        write_json_new(args.output, compare(before, after))
    else:
        # Permission checks occur before importing optional native dependencies.
        if not args.allow_native_load:
            parser.error("Native loading requires --allow-native-load")
        if args.command == "train" and (not args.allow_training or not args.qualification):
            parser.error("Training requires --allow-training and --qualification")
        from .native import native_run
        plan = Plan(**json.loads(Path(args.plan).read_text(encoding="utf-8")))
        native_run(args.command, checkpoint=args.checkpoint, rows=read_bank(args.bank), bank_path=args.bank,
                   plan=plan, eligibility=args.eligibility, output_dir=args.output,
                   allow_native_load=args.allow_native_load, allow_training=args.allow_training,
                   qualification=args.qualification, adapter_path=args.adapter,
                   allow_confirmation=args.allow_confirmation)


if __name__ == "__main__":
    main()
