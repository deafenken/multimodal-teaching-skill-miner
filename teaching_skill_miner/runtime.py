from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


VALID_SIGNALS = {"achieved", "not_achieved"}


def _fill(value: str, parameters: dict[str, str]) -> str:
    rendered = value
    for key, replacement in parameters.items():
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


@dataclass
class SkillRuntime:
    """A deterministic state machine that another teaching Agent can drive."""

    skill: dict[str, Any]
    concept: str
    learner_level: str = "beginner"
    phase: str = "procedure"
    procedure_index: int = 0
    verification_index: int = 0
    attempts: int = 0
    fallback_count: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)

    @property
    def parameters(self) -> dict[str, str]:
        return {"concept": self.concept, "learner_level": self.learner_level}

    @property
    def completed(self) -> bool:
        return self.phase == "completed"

    def current_turn(self) -> dict[str, Any]:
        if self.phase == "completed":
            return {
                "phase": "completed",
                "teacher_action": "summarize",
                "teacher_message": f"{self.concept} 的本轮教学与验证已完成。",
                "expected_signal": "session_complete",
            }
        if self.phase == "procedure":
            step = self.skill["procedure"][self.procedure_index]
            return {
                "phase": "procedure",
                "index": self.procedure_index,
                "step": step["step"],
                "teacher_action": step["teacher_action"],
                "teacher_message": _fill(step["instruction"], self.parameters),
                "expected_signal": _fill(step["expected_signal"], self.parameters),
                "fallback": _fill(step["fallback"], self.parameters),
                "attempt": self.attempts + 1,
            }
        check = self.skill["verification"][self.verification_index]
        return {
            "phase": "verification",
            "index": self.verification_index,
            "teacher_action": "check_understanding",
            "teacher_message": _fill(check["prompt"], self.parameters),
            "expected_signal": _fill(check["pass_condition"], self.parameters),
            "verification_type": check["type"],
            "attempt": self.attempts + 1,
        }

    def observe(self, learner_response: str, signal: str) -> dict[str, Any]:
        if signal not in VALID_SIGNALS:
            raise ValueError(f"signal must be one of {sorted(VALID_SIGNALS)}")
        if self.completed:
            raise RuntimeError("cannot observe after the session is completed")
        before = self.current_turn()
        achieved = signal == "achieved"
        event = {
            "phase": self.phase,
            "index": before.get("index"),
            "teacher_message": before["teacher_message"],
            "expected_signal": before["expected_signal"],
            "learner_response": learner_response,
            "signal": signal,
            "attempt": before["attempt"],
        }
        if achieved:
            self.attempts = 0
            if self.phase == "procedure":
                self.procedure_index += 1
                if self.procedure_index >= len(self.skill["procedure"]):
                    self.phase = "verification"
            else:
                self.verification_index += 1
                if self.verification_index >= len(self.skill["verification"]):
                    self.phase = "completed"
            event["transition"] = "advance"
        else:
            self.attempts += 1
            self.fallback_count += 1
            if self.phase == "procedure":
                fallback = before["fallback"]
            else:
                fallback = "先指出答案中与通过条件不一致的一处，再提供分层提示并要求重新作答。"
            if self.attempts >= 2:
                fallback += " 已连续两次未达标：降低任务复杂度，并回查必要前置知识。"
            event["transition"] = "retry"
            event["fallback_message"] = fallback
        self.history.append(event)
        return {"event": event, "next_turn": self.current_turn(), "state": self.snapshot()}

    def snapshot(self) -> dict[str, Any]:
        total = len(self.skill.get("procedure", [])) + len(self.skill.get("verification", []))
        completed = self.procedure_index + self.verification_index
        if self.phase == "completed":
            completed = total
        return {
            "skill_id": self.skill.get("skill_id"),
            "concept": self.concept,
            "learner_level": self.learner_level,
            "phase": self.phase,
            "procedure_index": self.procedure_index,
            "verification_index": self.verification_index,
            "attempts_on_current_turn": self.attempts,
            "fallback_count": self.fallback_count,
            "completed": self.completed,
            "completion_ratio": round(completed / total, 3) if total else 0.0,
            "history": self.history,
        }


def run_scripted_session(
    skill: dict[str, Any],
    *,
    concept: str,
    responses: list[dict[str, str]],
    learner_level: str = "beginner",
) -> dict[str, Any]:
    runtime = SkillRuntime(skill, concept=concept, learner_level=learner_level)
    initial_turn = runtime.current_turn()
    for item in responses:
        if runtime.completed:
            break
        runtime.observe(item.get("response", ""), item["signal"])
    return {
        "session_mode": "scripted_state_machine_demo",
        "response_source": "provided_script",
        "learning_effectiveness_established": False,
        "completion_semantics": "state-machine path completion only",
        "initial_turn": initial_turn,
        **runtime.snapshot(),
    }
