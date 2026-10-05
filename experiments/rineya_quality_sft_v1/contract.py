"""Fixed RQSFT-001 recipe and bounds; no user-overridable native sweep."""
from dataclasses import asdict, dataclass
from pathlib import Path

from . import EXPERIMENT, SOURCE_CORE, SOURCE_WEIGHT_SHA256

TRAIN_COUNTS = {"grounded": 80, "solving": 60, "preservation_dialogue": 40, "output_contract": 20}
DEV_COUNTS = {"acquisition": 128, "harder_transfer": 128, "grounding_contracts": 128,
              "preservation_dialogue": 128}
ARMS = {"whole_1e-5": ("whole", 1e-5), "response_1e-5": ("response", 1e-5)}
SOURCE_PATH = "C:/Users/clclc/BusinessProject/Chat-Bubble/Rineya_Published_Cores/Rineya-Supa-1.7.0-devfreeze"
DATASET_OWNER = {"thread_id": "01a109dd-a8bc-73d2-bf9c-5ebc09a00cde",
                 "title": "Plan Rineya dataset pipeline",
                 "root": "C:/Users/clclc/BusinessProject/Rineya-Dataset/Rineya_Intelligence_Quality_Foundry_v1"}


@dataclass(frozen=True)
class Design:
    processed_budget: int = 25000
    max_updates: int = 32
    max_sequence: int = 512
    eval_sequence: int = 2048
    update_tokens: int = 2048
    max_dev_rows: int = 512
    max_new_tokens: int = 64
    selection_seed: int = 17
    order_seed: int = 17
    trainable_tail: int = 20
    ce_chunk: int = 64
    warmup_fraction: float = .20
    decay_fraction: float = .20
    minimum_lr_fraction: float = .10
    beta1: float = .9
    beta2: float = .95
    epsilon: float = 1e-8
    weight_decay: float = 0.
    clip_norm: float = 1.
    checkpoint_every: int = 4
    timeout_seconds: int = 3600

    def as_dict(self):
        return asdict(self)


def validate_design(value):
    expected = Design().as_dict()
    if not isinstance(value, dict) or value != expected or any(type(value[k]) is not type(v) for k, v in expected.items()):
        raise ValueError("Only the reviewed 200-row/two-arm fixed RQSFT design is supported")
    return Design()


def external_output(path):
    target = Path(path).resolve()
    business = Path("C:/Users/clclc/BusinessProject").resolve()
    source = Path(SOURCE_PATH).resolve()
    if (target == business or not target.is_relative_to(business)
            or target.is_relative_to(source) or any(p.casefold() == ".codex" for p in target.parts)
            or any((p / ".git").exists() for p in (target, *target.parents))):
        raise ValueError("Outputs/caches must be under BusinessProject outside source/Git/.codex")
    return target
