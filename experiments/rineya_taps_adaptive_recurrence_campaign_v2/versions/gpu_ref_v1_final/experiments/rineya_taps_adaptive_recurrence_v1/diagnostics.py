"""Keep label-conditioned diagnostics out of the causal recurrence controller."""
from types import FunctionType


def generation_only_evaluator(evaluator):
    """Reuse the pinned ARMR generation/scoring body, without optional gold NLL.

    The inherited optional NLL sends a whole gold continuation through one model
    call. Sequence-wide adaptive statistics would let later answer tokens affect
    earlier logits. This lane therefore disables that diagnostic, rather than
    calling it a causal LM loss or adding its alphas to the DEV random pool.
    A fresh function-global dictionary changes only canonical_answer; neither
    the inherited module nor the scorer/generation bytecode is mutated.
    """
    if not isinstance(evaluator, FunctionType) or evaluator.__closure__ is not None:
        raise TypeError("a pinned non-closure evaluator function is required")
    if "canonical_answer" not in evaluator.__code__.co_names or not callable(evaluator.__globals__.get("canonical_answer")):
        raise ValueError("inherited canonical-answer diagnostic boundary changed")
    namespace = {**evaluator.__globals__, "canonical_answer": lambda _case: None}
    result = FunctionType(evaluator.__code__, namespace, evaluator.__name__, evaluator.__defaults__)
    result.__kwdefaults__ = evaluator.__kwdefaults__
    return result
