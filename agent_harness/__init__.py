"""Agent Harness: a recoverable, provider-neutral coding-agent runtime."""

__version__ = "2.7.0"

from .attachments import (
    ATTACHMENT_SCHEMA,
    AttachmentDescriptor,
    AttachmentError,
    AttachmentStore,
)
from .core import *  # noqa: F401,F403
from .core import __all__ as _core_all
from .context import ContextCompactionPlan, ContextCompactionResult
from .runner import AgentRunner, TurnOutcome
from .sdk import (
    SDK_RUN_RESULT_SCHEMA,
    AsyncHarnessClient,
    AsyncHarnessEventStream,
    AsyncHarnessThread,
    EventCallback,
    HarnessClient,
    HarnessClientOptions,
    HarnessEvent,
    HarnessEventStream,
    HarnessRunOptions,
    HarnessRunResult,
    HarnessSdkContractError,
    HarnessThread,
    HarnessThreadOptions,
    PathInput,
    PermissionMode,
)
from .session import (
    CONTEXT_COMPACTION_SCHEMA,
    SESSION_SCHEMA,
    SessionStore,
    SessionStoreError,
)
from .subagents import (
    SUBAGENT_BATCH_SCHEMA,
    SubagentBudgetLedger,
    SubagentLimits,
    SubagentLineage,
    SubagentResult,
    SubagentScheduler,
    SubagentTask,
)
from .worktrees import (
    WORKTREE_RECORD_SCHEMA,
    WorktreeCleanupResult,
    WorktreeError,
    WorktreeManager,
    WorktreeRecord,
)

__all__ = [
    *_core_all,
    "ATTACHMENT_SCHEMA",
    "AgentRunner",
    "AsyncHarnessClient",
    "AsyncHarnessEventStream",
    "AsyncHarnessThread",
    "AttachmentDescriptor",
    "AttachmentError",
    "AttachmentStore",
    "CONTEXT_COMPACTION_SCHEMA",
    "ContextCompactionPlan",
    "ContextCompactionResult",
    "EventCallback",
    "HarnessClient",
    "HarnessClientOptions",
    "HarnessEvent",
    "HarnessEventStream",
    "HarnessRunOptions",
    "HarnessRunResult",
    "HarnessSdkContractError",
    "HarnessThread",
    "HarnessThreadOptions",
    "PathInput",
    "PermissionMode",
    "SDK_RUN_RESULT_SCHEMA",
    "SESSION_SCHEMA",
    "SUBAGENT_BATCH_SCHEMA",
    "SessionStore",
    "SessionStoreError",
    "SubagentBudgetLedger",
    "SubagentLimits",
    "SubagentLineage",
    "SubagentResult",
    "SubagentScheduler",
    "SubagentTask",
    "TurnOutcome",
    "WORKTREE_RECORD_SCHEMA",
    "WorktreeCleanupResult",
    "WorktreeError",
    "WorktreeManager",
    "WorktreeRecord",
    "__version__",
]
