import pytest
from rgi.io import digest, seal, artifact, score_rows


def make_bank(n=24, split="development", evidence="synthetic_fixture"):
    items = []
    for i in range(n):
        prefix = f"{split}-{i}"
        items.append({"item_id": prefix, "parent_id": prefix, "lineage_id": prefix,
            "payload_sha256": digest(prefix), "family": f"f{i % 3}", "operation": "transition",
            "rule_case": "conditional" if i % 3 == 0 else "ordinary", "suite": "skill" if i % 2 else "protected",
            "format_id": "fixture-v1", "split": split, "horizon": 1, "target_rule": i % 3 == 0,
            "input_ids": [10, 11, 12, 13], "roles": ["prompt", "prompt", "content", "format"]})
    return seal({"schema_version": 1, "kind": "encoded_bank", "tokenizer_sha256": digest("fixture"),
                 "evidence_class": evidence, "items": items})


def make_artifact(bank, offset=0, model_id="fixture", components=None):
    values = [[2 + i / 50 + offset, 0.5 + i / 100 + offset] for i, _ in enumerate(bank["items"])]
    return artifact(bank, score_rows(bank, values), model_id, digest("host"), components)


@pytest.fixture
def bank():
    return make_bank()


@pytest.fixture
def reference(bank):
    return make_artifact(bank)
