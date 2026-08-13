from __future__ import annotations

from copy import deepcopy

import pytest

from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent import (
    classify_lesson_clarification_text,
    is_lesson_clarification_response,
    start_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_context import (
    _is_explicit_learner_question_text,
)
from teaching_skill_miner.teacher_agent_discourse import (
    classify_learner_discourse_text,
)
from teaching_skill_miner.teacher_agent_live import (
    _explicit_confusion,
    _is_question_response,
)
from teaching_skill_miner.teacher_agent_memory import _is_question_text


@pytest.mark.parametrize(
    ("text", "kind"),
    (
        ("递推关系到底拆成几个东西讲？", "composition"),
        ("你能展开一下这里吗", "definition"),
        ("Could you unpack what the recurrence means?", "definition"),
        ("What are the parts of a recurrence?", "composition"),
        ("Why do we standardize the features?", "rationale"),
        ("How does photosynthesis work", "procedure"),
        ("What is the difference between mitosis and meiosis?", "comparison"),
        ("Could you give an example", "example_request"),
    ),
)
def test_shared_clarification_classifier_generalizes_across_language_and_domain(
    text: str, kind: str
) -> None:
    discourse = classify_learner_discourse_text(text)
    assert discourse.clarification_kind == kind
    assert discourse.is_question is True
    assert discourse.explicit_confusion is False
    assert discourse.solution_risk == "none"
    assert classify_lesson_clarification_text(text) == kind
    assert _is_question_response(text) is True
    assert _is_question_text(text, legacy_detection=False) is True
    assert _is_explicit_learner_question_text(text) is True


@pytest.mark.parametrize(
    "text",
    (
        "这个我有点跟不上",
        "这里我没跟上",
        "I am lost here",
        "I don't understand this transition",
        "This went over my head",
    ),
)
def test_shared_explicit_confusion_floor_survives_missing_question_mark(
    text: str,
) -> None:
    discourse = classify_learner_discourse_text(text)
    assert discourse.explicit_confusion is True
    assert discourse.clarification_kind is None
    assert _explicit_confusion(text) is True


@pytest.mark.parametrize(
    "text",
    (
        "我知道为什么要用递推",
        "递推关系包括初始条件和递推规则",
        "I know what recurrence means",
        "This variable means the accumulated cost",
        "我不是跟不上，我的意思是状态不等于答案",
        "I am not confused; the state stores the best prefix cost",
    ),
)
def test_shared_classifier_does_not_erase_substantive_claims(text: str) -> None:
    discourse = classify_learner_discourse_text(text)
    assert discourse.clarification_kind is None
    assert discourse.explicit_confusion is False
    assert discourse.is_question is False


@pytest.mark.parametrize(
    "text",
    (
        "直接告诉我这道题的完整答案",
        "Can you give me the final answer?",
        "Solve this exercise for me",
    ),
)
def test_solution_requests_remain_questions_but_not_conceptual_clarifications(
    text: str,
) -> None:
    discourse = classify_learner_discourse_text(text)
    assert discourse.is_question is True
    assert discourse.solution_risk == "final_solution"
    assert discourse.clarification_kind is None
    assert classify_lesson_clarification_text(text) is None


def test_current_step_scope_is_separate_from_conceptual_answer_first() -> None:
    text = "Why does this step compare every predecessor?"
    discourse = classify_learner_discourse_text(text)
    assert discourse.clarification_kind == "rationale"
    assert discourse.solution_risk == "current_step"

    demo = read_json("data/teacher_agent_demo_input.json")
    library = read_json("data/teacher_agent_skill_library_v2.json")
    goal = deepcopy(demo["goal"])
    goal["learning_intent"] = "task_first"
    session = start_teacher_agent_session(goal, demo["student_profile"], library)
    assert session["lesson_state"]["lesson_phase"] == "guided_practice"
    assert is_lesson_clarification_response(session, text) is False
