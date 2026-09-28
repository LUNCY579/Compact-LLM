"""CPU estimator contracts and exact small-domain tests; NOT a native trainer."""
from __future__ import annotations
import math
from .specs import require

def advantages(rewards: list[int], method: str = "maxrl", epsilon: float = 1e-6) -> list[float]:
    require(type(rewards) is list and len(rewards) >= 2 and all(type(r) is int and r in (0, 1) for r in rewards), "need >=2 binary outcomes from ONE prompt")
    require(method in ("maxrl", "grpo", "reinforce_centered"), "unknown estimator")
    require(type(epsilon) in (int, float) and math.isfinite(epsilon) and epsilon >= 0, "bad epsilon")
    mean = sum(rewards)/len(rewards)
    if mean in (0.0, 1.0): return [0.0]*len(rewards)
    if method == "maxrl": denominator = mean+epsilon
    elif method == "grpo":
        # Match the official code's sample standard deviation, ddof=1.
        denominator = math.sqrt(sum((r-mean)**2 for r in rewards)/(len(rewards)-1))+epsilon
    else: denominator = 1.0
    return [(r-mean)/denominator for r in rewards]

def checked_group(records: list[dict], n: int, policy_id: str) -> list[float]:
    """Reject mixed prompts, duplicate trajectories, partial groups and stale samples."""
    require(type(records) is list and type(n) is int and n >= 2 and len(records) == n, "incomplete group")
    required = {"prompt_id", "rollout_id", "policy_id", "reward", "generated_tokens"}
    require(all(type(r) is dict and set(r) == required for r in records), "invalid rollout metadata")
    require(all(type(r["prompt_id"]) is str and r["prompt_id"] and type(r["rollout_id"]) is str and r["rollout_id"] for r in records), "missing identities")
    require(len({r["prompt_id"] for r in records}) == 1 and len({r["rollout_id"] for r in records}) == n, "group identity/collision")
    require(type(policy_id) is str and policy_id and all(r["policy_id"] == policy_id for r in records), "stale/ambiguous policy")
    require(all(type(r["generated_tokens"]) is int and r["generated_tokens"] > 0 for r in records), "missing token costs")
    return advantages([r["reward"] for r in records])

def pass_at_k(n: int, c: int, k: int) -> float:
    require(all(type(x) is int for x in (n, c, k)) and 0 <= c <= n and 1 <= k <= n, "invalid pass@k counts")
    return 1.0 if n-c < k else 1-math.comb(n-c, k)/math.comb(n, k)

def signal_probability(p: float, n: int) -> dict[str, float]:
    require(0 <= p <= 1 and type(n) is int and n >= 2, "bad sampling inputs")
    return {"all_failure": (1-p)**n, "mixed": 1-(1-p)**n-p**n, "all_success": p**n}
