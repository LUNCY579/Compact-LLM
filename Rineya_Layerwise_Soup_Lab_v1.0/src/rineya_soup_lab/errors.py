from __future__ import annotations


class RineyaSoupError(RuntimeError):
    """Base exception for expected research-pipeline failures."""


class ConfigurationError(RineyaSoupError):
    """Raised when a configuration is invalid or unsafe."""


class CompatibilityError(RineyaSoupError):
    """Raised when the two checkpoints cannot be merged safely."""


class EvaluationError(RineyaSoupError):
    """Raised when an evaluator fails or produces invalid evidence."""


class SealedTestError(RineyaSoupError):
    """Raised when a sealed-test access policy would be violated."""
