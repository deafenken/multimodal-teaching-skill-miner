"""Agent Harness: a recoverable, provider-neutral coding-agent runtime."""

__version__ = "2.4.0"

from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .context import ContextCompactionPlan, ContextCompactionResult
from .runner import AgentRunner, TurnOutcome
from .session import (
    CONTEXT_COMPACTION_SCHEMA,
    SESSION_SCHEMA,
    SessionStore,
    SessionStoreError,
)

__all__ = [
    *_core_all,
    "AgentRunner",
    "CONTEXT_COMPACTION_SCHEMA",
    "ContextCompactionPlan",
    "ContextCompactionResult",
    "SESSION_SCHEMA",
    "SessionStore",
    "SessionStoreError",
    "TurnOutcome",
    "__version__",
]
