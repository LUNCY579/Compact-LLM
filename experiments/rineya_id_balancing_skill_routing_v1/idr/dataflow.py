"""Collect costly counterfactual labels, fit on TRAIN, predict without outcomes."""
import copy
from pathlib import Path
import time
import torch
from .artifacts import validate_rows, sha256_file
from .control import digest
from .quality import QualityHead, fit_quality
from .experiment import build_pool, exact_state


def capture(spec, rows, backend):
    """All-expert labels for diagnosis/TRAIN quality fitting, not free routing.

    Reserved confirmation banks cannot be used for counterfactual capture.
    Budget accounting includes EVERY executed eligible expert, not just a winner.
    """
    validate_rows(rows)
    if not rows or any(r["split"] == "confirm" for r in rows):
        raise ValueError("confirmation counterfactual capture is prohibited")
    pool = build_pool(spec)
    captured = []
    for row in rows:
        if row["family"] != pool.experts[0].operation:
            raise ValueError("cross-operation capture prohibited")
        record = copy.deepcopy(row)
        record.update({"expert_order": [e.name for e in pool.experts],
                       "capture_pool_sha256": digest(spec["experts"]),
                       "outcomes": [], "observed": [], "observed_cost": [], "responses": []})
        for expert in pool.experts:
            observed = expert.qualified and expert.prior > 0
            response = backend(expert.name, expert.strength, {"prompt": row["payload"]["prompt"],
                               "max_new_tokens": spec.get("max_new_tokens", 128)}) if observed else None
            record["observed"].append(observed)
            record["outcomes"].append(exact_state(response["text"], row["expected_state"]) if observed else 0.0)
            record["observed_cost"].append(response["elapsed_seconds"] if observed else 0.0)
            record["responses"].append(response)
        record["counterfactual_calls"] = sum(record["observed"])
        record["counterfactual_seconds"] = sum(record["observed_cost"])
        captured.append(record)
    return captured


def save_quality(spec, rows, path, epochs=100, seed=0):
    if Path(path).exists():
        raise FileExistsError("quality checkpoint already exists")
    order = [e["name"] for e in spec["experts"]]
    if any(r["expert_order"] != order or r["capture_pool_sha256"] != digest(spec["experts"]) for r in rows):
        raise ValueError("quality labels refer to another expert bank/order")
    head, report = fit_quality(rows, epochs=epochs, seed=seed)
    saved = {"schema": 1, "state": head.state_dict(), "features": head.linear.in_features,
             "experts": head.linear.out_features, "pool_sha256": digest(spec["experts"]),
             "train_sha256": digest(rows), "report": report,
             "identities": [{"family": r["family"], "parent_id": r["parent_id"],
                              "payload_sha256": digest(r["payload"])} for r in rows]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("xb") as f:
        torch.save(saved, f)
    return {**report, "checkpoint_sha256": sha256_file(path)}


def predict_quality(spec, rows, path, expected_sha256):
    validate_rows(rows)
    if sha256_file(path) != expected_sha256:
        raise ValueError("quality checkpoint hash mismatch")
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved["schema"] != 1 or saved["pool_sha256"] != digest(spec["experts"]):
        raise ValueError("quality checkpoint bank mismatch")
    parents = {(r["family"], r["parent_id"]) for r in saved["identities"]}
    payloads = {r["payload_sha256"] for r in saved["identities"]}
    if any(r["split"] == "train" or (r["family"], r["parent_id"]) in parents
           or digest(r["payload"]) in payloads for r in rows):
        raise ValueError("quality prediction input overlaps TRAIN")
    head = QualityHead(saved["features"], saved["experts"])
    head.load_state_dict(saved["state"])
    head.eval().requires_grad_(False)
    predictions = []
    for row in rows:
        record = copy.deepcopy(row)
        start = time.perf_counter()
        features = torch.tensor(row["features"], dtype=torch.float32)
        if features.shape != (saved["features"],) or not torch.isfinite(features).all():
            raise ValueError("invalid input-only routing features")
        with torch.inference_mode():
            record["quality"] = head(features).tolist()
        record["quality_feature_seconds"] = time.perf_counter() - start
        record["quality_source"] = {"checkpoint_sha256": expected_sha256,
                                    "train_sha256": saved["train_sha256"]}
        # Cost-unaware default. Do not use observed counterfactual costs as predictions.
        record["predicted_cost"] = row.get("predicted_cost", [1.0] * saved["experts"])
        predictions.append(record)
    return predictions
