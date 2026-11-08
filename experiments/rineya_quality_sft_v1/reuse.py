"""Exact request-identity reuse of public predictions; never qualifications."""
from .integrity import digest

REQUEST_FIELDS = ("input_ids", "prefix_ids", "eos_ids", "source_assets_sha256",
    "source_parameters_sha256", "runtime", "generation_recipe", "context_limit", "output_limit")
BARRED_ROLES = {"SEALED", "PROTECTED", "CONFIRMATION", "EXCLUDED", "SPENT", "ANY_READ"}


def reuse_prediction(historical, request):
    if historical.get("material_role") != "OPEN_DEV" or any(role in BARRED_ROLES for role in historical.get("ancestry_roles", [])):
        raise PermissionError("Historical protected/excluded/spent material cannot be read or relabeled")
    if historical.get("completed") is not True or historical.get("source_unchanged") is not True:
        raise ValueError("Reuse requires actual completed unchanged-source measurement")
    old = historical.get("request")
    if not isinstance(old, dict) or any(k not in old or k not in request or old[k] != request[k] for k in REQUEST_FIELDS):
        raise ValueError("Changed input/prefix/EOS/source/runtime/generation requires fresh measurement")
    prediction = historical.get("prediction")
    if not isinstance(prediction, dict) or not historical.get("artifact_sha256"):
        raise ValueError("Actual prediction and retained artifact binding required")
    raw = {key: value for key, value in prediction.items() if key in {
        "text", "prediction", "generated_ids", "stop_reason", "terminated", "context_tokens"}}
    if not any(isinstance(raw.get(key), str) for key in ("text", "prediction")):
        raise ValueError("Reusable raw prediction text required; old judgments are never reused")
    return {"prediction": raw, "origin": "REUSED_COMPLETED_MEASUREMENT",
        "historical_artifact_sha256": historical["artifact_sha256"], "request_sha256": digest(request),
        "new_model_calls": 0, "cost_equivalence_claimed": False, "qualification_transferred": False,
        "judgments_reused": False, "fresh_current_gold_and_scorer_required": True}
