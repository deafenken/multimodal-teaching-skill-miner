from __future__ import annotations

from pathlib import Path

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent import advance_teacher_agent_session, start_teacher_agent_session
from teaching_skill_miner.teacher_agent_live import (
    LiveAgentOptions,
    advance_live_teacher_agent_session,
    start_live_teacher_agent_session,
)


ROOT = Path(__file__).resolve().parents[1]


def _fixtures() -> tuple[dict, dict]:
    return (
        read_json(ROOT / "data/teacher_agent_demo_input.json"),
        read_json(ROOT / "data/teacher_agent_skill_library_v2.json"),
    )


def test_system_fallback_does_not_count_as_three_no_progress_turns() -> None:
    demo, library = _fixtures()
    session = start_teacher_agent_session(
        demo["goal"], demo["student_profile"], library
    )
    # Establish two genuine no-progress observations first.
    for signal in ("confused", "no_response"):
        session = advance_teacher_agent_session(
            session,
            learner_response="不知道",
            signal=signal,
        )
    assert session["control"]["consecutive_no_progress"] == 2

    # A provider/assessment fallback is explicitly not learner evidence.  It
    # must consume turns without pushing the no-progress counter over the
    # terminal threshold.
    for _ in range(3):
        session = advance_teacher_agent_session(
            session,
            learner_response="答案暂时无法核验",
            signal="partial",
            signal_confidence=0.0,
            count_as_no_progress=False,
        )

    assert session["status"] == "active"
    assert session["control"]["consecutive_no_progress"] == 2


def test_live_fallback_report_separates_loop_planner_action_and_assessment_failures() -> None:
    demo, library = _fixtures()

    def unavailable_transport(*_args: object, **_kwargs: object) -> tuple[int, bytes]:
        return 503, b"provider unavailable"

    client = DeepSeekClient(
        DeepSeekConfig(allow_remote_student_data=True, max_retries=0),
        api_key="fixture-key",
        transport=unavailable_transport,
    )
    options = LiveAgentOptions(fallback_to_rules=True)
    session = start_live_teacher_agent_session(
        demo["goal"], demo["student_profile"], library, client, options=options
    )
    for _ in range(3):
        session = advance_live_teacher_agent_session(
            session,
            learner_response="我还不能确认这一步。",
            client=client,
            options=options,
        )

    runtime = session["agent_runtime"]
    assert session["status"] == "active"
    assert session["control"]["consecutive_no_progress"] == 0
    assert runtime["fallback_count"] == 4
    assert runtime["planner_fallback_count"] == 4
    assert runtime["action_fallback_count"] == 4
    assert runtime["assessment_failure_count"] == 3
    assert runtime["consecutive_assessment_failures"] == 3
