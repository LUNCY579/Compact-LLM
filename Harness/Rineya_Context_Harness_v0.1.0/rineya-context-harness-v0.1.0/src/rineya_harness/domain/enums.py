from __future__ import annotations

from enum import StrEnum


class ActorType(StrEnum):
    MODEL = "model"
    SYSTEM = "system"
    USER = "user"
    ADMIN = "admin"
    IMPORTER = "importer"


class RunMode(StrEnum):
    AUTO = "auto"
    DIRECT = "direct"
    GROUNDED = "grounded"
    RESEARCH = "research"


class RunStatus(StrEnum):
    CREATED = "created"
    RUNNING = "running"
    WAITING = "waiting"
    VERIFYING = "verifying"
    COMPLETED = "completed"
    ABSTAINED = "abstained"
    FAILED = "failed"
    CANCELLED = "cancelled"


class MemoryKind(StrEnum):
    GLOBAL_INVARIANT = "global_invariant"
    CURRENT_STATE = "current_state"
    OPEN_GOAL = "open_goal"
    ACTIVE_EVIDENCE = "active_evidence"
    RESOLVED_EVIDENCE = "resolved_evidence"
    TRANSIENT_OBSERVATION = "transient_observation"
    COMPLETED_CHECKPOINT = "completed_checkpoint"
    SUPERSEDED_STATE = "superseded_state"
    CONFLICT = "conflict"
    USER_PREFERENCE = "user_preference"
    EVENT_HISTORY = "event_history"


class MemoryStatus(StrEnum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    EVICTED = "evicted"
    SUPERSEDED = "superseded"
    PURGED = "purged"


class MemoryTemperature(StrEnum):
    PINNED = "pinned"
    HOT = "hot"
    WARM = "warm"
    COLD = "cold"


class MemoryOperationType(StrEnum):
    ADD = "add"
    UPDATE = "update"
    MERGE = "merge"
    SUPERSEDE = "supersede"
    ARCHIVE = "archive"
    EVICT = "evict"
    RESTORE = "restore"
    PURGE = "purge"
    PIN = "pin"
    UNPIN = "unpin"


class DeleteScope(StrEnum):
    ACTIVE_CONTEXT = "active_context"
    ARCHIVE = "archive"
    PERSISTENT_PURGE = "persistent_purge"


class ReasonCode(StrEnum):
    COMPLETED_NO_FUTURE_DEPENDENCY = "completed_no_future_dependency"
    SUPERSEDED_BY_CURRENT_STATE = "superseded_by_current_state"
    DUPLICATE = "duplicate"
    IRRELEVANT_TO_OPEN_GOALS = "irrelevant_to_open_goals"
    SOURCE_REDUCED_TO_CLAIM = "source_reduced_to_claim"
    EXPIRED = "expired"
    USER_REQUEST = "user_request"
    POLICY_RETENTION = "policy_retention"
    OTHER = "other"


class SupportStatus(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    CONFLICTED = "conflicted"
    UNSUPPORTED = "unsupported"
    STALE = "stale"


class RouteAction(StrEnum):
    DIRECT = "direct"
    CLARIFY = "clarify"
    SEARCH = "search"
    ABSTAIN = "abstain"


class FinishAction(StrEnum):
    ANSWER = "answer"
    SEARCH_MORE = "search_more"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"


class WorkflowNode(StrEnum):
    INTAKE = "intake"
    ROUTE = "route"
    SEARCH = "search"
    FETCH = "fetch"
    EXTRACT = "extract"
    RECONSTRUCT = "reconstruct"
    COMPACT = "compact"
    VERIFY = "verify"
    ANSWER = "answer"
    CLARIFY = "clarify"
    ABSTAIN = "abstain"
    DONE = "done"
    ERROR = "error"


class TraceEventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_FINISHED = "run_finished"
    NODE_ENTERED = "node_entered"
    NODE_EXITED = "node_exited"
    TOOL_STARTED = "tool_started"
    TOOL_FINISHED = "tool_finished"
    TOOL_FAILED = "tool_failed"
    MODEL_STARTED = "model_started"
    MODEL_FINISHED = "model_finished"
    STATE_PATCH_COMMITTED = "state_patch_committed"
    CONTEXT_REBUILT = "context_rebuilt"
    ANSWER_VERIFIED = "answer_verified"
    SECURITY_REJECTION = "security_rejection"
    BUDGET_UPDATED = "budget_updated"
    ERROR = "error"


class ToolRisk(StrEnum):
    READ_ONLY = "read_only"
    MUTATING_REVERSIBLE = "mutating_reversible"
    MUTATING_IRREVERSIBLE = "mutating_irreversible"
