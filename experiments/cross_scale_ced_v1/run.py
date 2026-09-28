"""Explicit local-only qualification and separately authorized engineering CLI."""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path
import torch
from .config import official_config, tiny_config, cross_budget
from .model import CrossScale
from .checks import qualify_model, native_parity
from .source import audit_source, load_reference, native_from_reference, tokenizer_pair
from .guards import (PACKAGE, protocol, permissions, runtime, checkout, verify_dependencies,
                     require_resources, new_directory, write_json_new, read_json,
                     sha256_file, seal_receipt, validate_p1)
from .training import paired_smoke


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("inspect", "tiny-qualify", "tiny-smoke", "qualify", "smoke"))
    parser.add_argument("--encoder", type=Path)
    parser.add_argument("--decoder", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--p1-receipt", type=Path)
    parser.add_argument("--confirm-engineering-only", action="store_true")
    args = parser.parse_args(argv)
    output = None
    try:
        verify_dependencies()
        if args.command == "inspect":
            result = dict(experiment="XSC-001", config=official_config().to_dict(),
                          budget=cross_budget(official_config()), runtime=runtime(), permissions=permissions(),
                          official_execution="not run; requires exact sources and fresh P1")
            print(json.dumps(result, indent=2)); return 0
        if args.output is None:
            raise ValueError("--output must name a new run directory")
        output = new_directory(args.output, scoped=True)
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(protocol()["seed"])
        if args.command.startswith("tiny-"):
            model = CrossScale(tiny_config())
            binding = {"scope": "synthetic_random_weights", "config": model.config.to_dict(), "runtime": runtime()}
            if args.command == "tiny-qualify":
                results = {}
                for mode in ("H", "P"):
                    model = CrossScale(tiny_config(mode))
                    results[mode] = qualify_model(model, checkpoint_parent=output, binding=binding)
                write_json_new(output/"qualification.json", dict(status="SYNTHETIC_PASS_ONLY", results=results,
                                                                 permissions=permissions(), licenses="nothing"))
            else:
                rows = [torch.tensor([[3, 11, 8, 7, 19, 2, 5, 13]])]
                paired_smoke(model.encoder, model.decoder, rows, output, binding, synthetic=True)
            return 0
        if args.encoder is None or args.decoder is None:
            raise ValueError("both --encoder and --decoder local source directories are required")
        if args.command == "smoke" and (not args.confirm_engineering_only or args.p1_receipt is None):
            raise ValueError("smoke needs --p1-receipt and separately authorized --confirm-engineering-only")
        # All authorization/resource checks precede model allocation; no network/download.
        binding = dict(checkout=checkout(), runtime=runtime(require=True),
                       protocol_sha256=sha256_file(PACKAGE/"protocol.json"),
                       sources_lock_sha256=sha256_file(PACKAGE/"sources.json"))
        resources = require_resources("qualify" if args.command == "qualify" else "smoke")
        binding["encoder"] = audit_source(args.encoder, "encoder")
        binding["decoder"] = audit_source(args.decoder, "decoder")
        if args.command == "smoke":
            validate_p1(args.p1_receipt, binding)
        write_json_new(output/"binding.json", dict(binding=binding, resources=resources))
        tok = tokenizer_pair(args.encoder, args.decoder)
        encoder = load_reference(args.encoder, "encoder")
        native = native_from_reference(encoder, args.encoder)
        ne = native_parity(encoder, native)
        del native; gc.collect()
        decoder = load_reference(args.decoder, "decoder")
        native = native_from_reference(decoder, args.decoder)
        nd = native_parity(decoder, native)
        del native; gc.collect()
        if args.command == "qualify":
            results, inventory = {}, []
            for mode in ("H", "P"):
                torch.manual_seed(protocol()["seed"])
                model = CrossScale.from_sources(encoder, decoder, mode)
                model.freeze_backbones(); model.verify_budget()
                inventory.append([(n, list(p.shape)) for n, p in model.named_parameters() if p.requires_grad])
                results[mode] = qualify_model(model, checkpoint_parent=output, binding=binding)
                del model; gc.collect()
            if inventory[0] != inventory[1]:
                raise AssertionError("H/P inventory mismatch")
            write_json_new(output/"qualification.json", dict(schema=1, experiment="XSC-001", phase="P1",
                                                            binding=binding, matched_inventory=True,
                                                            results=results, native_encoder=ne,
                                                            native_decoder=nd, resources=resources))
            names = ["qualification.json", "binding.json"]
            for mode in ("H", "P"):
                names.extend([f"{mode}_active_interface/manifest.json", f"{mode}_active_interface/interface.safetensors"])
            body = dict(schema=1, experiment="XSC-001", phase="P1", status="PASS", binding=binding,
                        checks={k: True for k in protocol()["P1_checks"]}, permissions=permissions(),
                        licenses=protocol()["P1_licenses"], artifacts={name: sha256_file(output/name) for name in names})
            write_json_new(output/"receipt.json", seal_receipt(body))
        else:
            rows = [torch.tensor([tok.encode(text, add_special_tokens=False)[:protocol()["smoke"]["max_tokens"]]])
                    for text in read_json(PACKAGE/"fixtures.json")]
            paired_smoke(encoder, decoder, rows, output, binding)
        return 0
    except Exception as exc:
        if output is not None:
            write_json_new(output/"failure.json", dict(status="FAIL_OR_BLOCKED", command=args.command,
                                                       error_type=type(exc).__name__, error=str(exc),
                                                       permissions=permissions(), licenses="nothing"))
        print(f"{type(exc).__name__}: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
