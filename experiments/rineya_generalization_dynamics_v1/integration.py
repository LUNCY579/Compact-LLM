"""Record completed trainer exports without changing any trainer or model state."""
from pathlib import Path
from .common import require, signed, write_new
from .registry import snapshot


def record_snapshot(receipt_directory: str, checkpoint_directory: str, step: int, kind: str = "peft") -> dict:
    """Call AFTER save_pretrained/save_model completes, outside an optimizer step.

    Records are ancillary provenance. Register the complete trajectory using `init`
    before running any diagnostics. This hook performs no evaluation or training.
    """
    require(type(step) is int and step > 0, "A trained checkpoint needs a positive step")
    require(kind in {"hf", "peft"}, "Only HF or LoRA exports can be recorded")
    asset = snapshot(checkpoint_directory, kind)
    entry = {"id": f"step_{step:08d}", "step": step, "kind": kind, "path": asset["path"]}
    receipt = signed({"entry": entry, "asset": asset, "evaluated": False})
    write_new(Path(receipt_directory) / (entry["id"] + ".json"), receipt)
    return receipt
