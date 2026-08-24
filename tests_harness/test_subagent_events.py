from __future__ import annotations

import pytest

from agent_harness.core import HarnessContractError
from agent_harness.core.events import HarnessEventEmitter, public_event_projection


def _progress_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "call_id": "delegate_1",
        "tool_name": "agent.delegate",
        "progress_kind": "subagent.child_finished",
        "progress": {
            "agent_id": "agent_123",
            "depth": 1,
            "ordinal": 0,
            "status": "completed",
        },
    }
    payload.update(overrides)
    return payload


def test_subagent_progress_has_typed_content_free_public_projection() -> None:
    emitter = HarnessEventEmitter(run_id="run_agents", turn_id="turn_agents")
    event = emitter.emit("tool.progress", _progress_payload())

    public = public_event_projection(event)

    assert public["payload"] == {
        "call_id": "delegate_1",
        "tool_name": "agent.delegate",
        "progress_kind": "subagent.child_finished",
        "progress": {
            "agent_id": "agent_123",
            "depth": 1,
            "ordinal": 0,
            "status": "completed",
        },
    }


@pytest.mark.parametrize(
    "payload",
    (
        _progress_payload(tool_name="workspace.read"),
        _progress_payload(progress={"agent_id": "agent_123", "depth": 1}),
        _progress_payload(
            progress={
                "agent_id": "agent_123",
                "depth": 1,
                "ordinal": 0,
                "status": "completed",
                "prompt": "must-not-enter-lifecycle-events",
            }
        ),
        _progress_payload(
            progress={
                "agent_id": "../escape",
                "depth": 1,
                "ordinal": 0,
                "status": "completed",
            }
        ),
        _progress_payload(
            progress={
                "agent_id": "agent_123",
                "depth": 1,
                "ordinal": 0,
                "status": "running",
            }
        ),
    ),
)
def test_subagent_progress_rejects_ambiguous_or_content_bearing_metadata(
    payload: dict[str, object],
) -> None:
    emitter = HarnessEventEmitter(run_id="run_agents", turn_id="turn_agents")

    with pytest.raises(HarnessContractError):
        emitter.emit("tool.progress", payload)

    assert emitter.next_sequence == 1
    assert emitter.events == []
