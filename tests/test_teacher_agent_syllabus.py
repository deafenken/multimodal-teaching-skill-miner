from __future__ import annotations

from collections import deque
from copy import deepcopy
from hashlib import sha256
import http.client
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from urllib.parse import urlsplit

import jsonschema
import pytest

from teaching_skill_miner.deepseek_client import (
    DeepSeekClient,
    DeepSeekClientError,
    DeepSeekConfig,
)
from teaching_skill_miner.io_utils import project_root, read_json
from teaching_skill_miner.teacher_agent import start_teacher_agent_session
from teaching_skill_miner.teacher_agent_context import build_layered_context
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
)
from teaching_skill_miner.teacher_agent_syllabus import (
    TEACHING_SYLLABUS_AUXILIARY_SKILL,
    TeachingSyllabusError,
    TeachingSyllabusStore,
    _seal_generated_syllabus,
    generate_teaching_syllabus,
    syllabus_lesson_start_payload,
    teaching_syllabus_editable_draft,
    validate_teaching_syllabus,
)
from teaching_skill_miner.teacher_agent_live import (
    advance_live_teacher_agent_session,
    start_live_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_syllabus_versions import (
    TeachingSyllabusVersionError,
    TeachingSyllabusVersionStore,
)


def _draft() -> dict:
    return {
        "title": "机器学习入门",
        "description": "从监督学习的基本任务走向新情境迁移。",
        "audience": "零基础大学生",
        "estimated_duration_minutes": 90,
        "learning_objectives": ["区分输入、标签与模型输出", "把学习框架迁移到新任务"],
        "prerequisites": ["能阅读简单表格"],
        "modules": [
            {
                "title": "监督学习基础",
                "description": "建立数据、标签和预测之间的关系。",
                "lessons": [
                    {
                        "title": "输入、标签与预测",
                        "objective": "能解释监督学习任务的输入和输出。",
                        "summary": "用有标签样本学习从输入到输出的关系。",
                        "duration_minutes": 90,
                        "knowledge_components": ["输入特征", "标签", "预测"],
                        "materials": {
                            "example": "以历史房屋信息预测价格，辨认输入与标签。",
                            "practice": "给出一个分类场景并标出输入和标签。",
                            "transfer_task": "把同一框架迁移到垃圾邮件分类。",
                        },
                    }
                ],
            }
        ],
    }


class _SyllabusClient:
    def __init__(
        self,
        response: dict | None = None,
        *,
        responses: list[dict | Exception] | None = None,
    ) -> None:
        self.response = deepcopy(response or _draft())
        self.responses = list(responses or [])
        self.calls: list[dict] = []

    def public_status(self) -> dict:
        return {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_origin": "https://api.deepseek.example",
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "configured": True,
            "remote_student_data_opt_in": True,
            "api_key_exposed": False,
        }

    def chat_json(self, messages, *, request_kind, require_remote_consent=True):
        self.calls.append(
            {
                "messages": deepcopy(messages),
                "request_kind": request_kind,
                "require_remote_consent": require_remote_consent,
            }
        )
        response = self.responses.pop(0) if self.responses else self.response
        if isinstance(response, Exception):
            raise response
        return deepcopy(response), {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "latency_ms": 1.2,
            "attempt_count": 1,
            "http_status": 200,
            "usage": {"total_tokens": 120},
            "credential_logged": False,
            "api_key": "must-not-escape",
        }


def _live_plan(
    *,
    signal: str,
    confidence: float,
    skill_id: str,
    action_type: str,
    message: str,
    evidence_excerpt: str = "",
) -> dict:
    alignment = {
        "not_observed": "not_applicable",
        "correct": "aligned",
        "partial": "partially_aligned",
    }[signal]
    return {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": signal,
            "confidence": confidence,
            "answer_alignment": alignment,
            "matched_concepts": [],
            "missing_concepts": [],
            "diagnosis_reason": "模型候选判断，仅供服务端门禁核验。",
            "evidence_excerpt": evidence_excerpt,
            "misconception_tag": None,
            "misconception_description": "",
            "resolved_misconception_tags": [],
            "response_quality": "empty" if signal == "not_observed" else "partial",
            "engagement_level": "unknown" if signal == "not_observed" else "medium",
            "needs_human_review": False,
        },
        "decision": {
            "primary_skill_id": skill_id,
            "supporting_skill_ids": [],
            "selection_reason": "按当前课堂阶段选择教学动作。",
            "next_focus": "conceptual",
        },
        "teacher_action": {
            "type": action_type,
            "message": message,
            "expected_signal": "学生给出一个可核验的当前回答。",
            "question_contract": {
                "answer_type": "explanation",
                "target_concepts": ["当前课节知识点"],
                "accepted_aliases": [],
                "success_criteria": ["直接回答当前问题并说明依据"],
            },
        },
        "stop_recommendation": {"should_stop": False, "reason": ""},
    }


class _LivePlanClient:
    def __init__(self, plans: list[dict]) -> None:
        self.plans = deque(deepcopy(plans))

    def public_status(self) -> dict:
        return {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_origin": "https://api.deepseek.example",
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "configured": True,
            "remote_student_data_opt_in": True,
            "api_key_exposed": False,
        }

    def chat_json(self, _messages, *, request_kind, require_remote_consent=True):
        assert request_kind in {"teacher_agent_initial", "teacher_agent_turn"}
        assert require_remote_consent is True
        return self.plans.popleft(), {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "latency_ms": 1.0,
            "attempt_count": 1,
            "http_status": 200,
            "usage": {"total_tokens": 80},
            "credential_logged": False,
        }


def _lesson_live_plans(final_plan: dict) -> list[dict]:
    orientation = _live_plan(
        signal="not_observed",
        confidence=0.0,
        skill_id="skill_contextual_problem_setup",
        action_type="establish_problem_context",
        message=(
            "先由我做本节导入：本阶段不考前置知识。"
            "本节教学路径是先讲解、再示范、随后带练与核验。"
            "现在只需回复继续。"
        ),
    )
    orientation["teacher_action"]["question_contract"] = {
        "answer_type": "reflection",
        "target_concepts": ["继续或选择教学方式"],
        "accepted_aliases": ["继续", "先看例子", "按步骤讲"],
        "success_criteria": ["学生确认继续或选择一种教学方式"],
    }
    return [
        orientation,
        _live_plan(
            signal="partial",
            confidence=0.7,
            skill_id="skill_concrete_example_bridge",
            action_type="present_minimal_example",
            message="先讲清当前知识点的对象、条件与关系；理解路径后回复继续。",
            evidence_excerpt="继续",
        ),
        _live_plan(
            signal="partial",
            confidence=0.7,
            skill_id="skill_concrete_example_bridge",
            action_type="present_minimal_example",
            message="完整示范一个最小例子的输入、关系与结果；回复继续进入带练。",
            evidence_excerpt="继续",
        ),
        _live_plan(
            signal="partial",
            confidence=0.7,
            skill_id="skill_stepwise_scaffolding",
            action_type="guide_one_micro_step",
            message="现在一起完成一个可检查的第一步，并说明这一步的依据。",
            evidence_excerpt="继续",
        ),
        final_plan,
    ]


def _advance_to_guided_practice(goal: dict, profile: dict, library: dict, client):
    session = start_live_teacher_agent_session(goal, profile, library, client)
    for _ in range(3):
        session = advance_live_teacher_agent_session(
            session,
            learner_response="继续",
            client=client,
        )
    assert session["lesson_state"]["lesson_phase"] == "guided_practice"
    return session


def _sealed() -> dict:
    return _seal_generated_syllabus(
        _draft(),
        model="deepseek-v4-flash",
        source_resource_ids=[],
        created_at="2026-08-11T12:00:00Z",
    )


def _post(base_url: str, route: str, body: dict) -> tuple[int, dict, dict]:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    path = parsed.path + route
    encoded = json.dumps(body, ensure_ascii=False).encode("utf-8")
    connection.request(
        "POST",
        path,
        body=encoded,
        headers={
            "Content-Type": "application/json",
            "Content-Length": str(len(encoded)),
        },
    )
    response = connection.getresponse()
    payload = json.loads(response.read())
    headers = dict(response.getheaders())
    connection.close()
    return response.status, payload, headers


def _get(base_url: str, route: str) -> tuple[int, dict, dict]:
    parsed = urlsplit(base_url)
    connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
    connection.request("GET", parsed.path + route)
    response = connection.getresponse()
    payload = json.loads(response.read())
    headers = dict(response.getheaders())
    connection.close()
    return response.status, payload, headers


def test_strict_syllabus_validates_in_python_and_json_schema() -> None:
    syllabus = _sealed()
    validate_teaching_syllabus(syllabus)
    schema = read_json(project_root() / "schema/teaching_syllabus.schema.json")
    jsonschema.Draft202012Validator.check_schema(schema)
    jsonschema.Draft202012Validator(schema).validate(syllabus)
    assert syllabus["schema"] == "teaching_syllabus.v1"
    assert syllabus["claim_boundary"]["contains_gold_answers"] is False
    assert (
        syllabus["modules"][0]["lessons"][0]["teaching_goal"]["learning_intent"]
        == "teach_first"
    )
    generated_spec = syllabus["modules"][0]["lessons"][0]["teaching_goal"][
        "knowledge_spec"
    ]
    assert generated_spec["rubric_criteria"]
    assert generated_spec["canonical_claims"] == []
    assert generated_spec["authority"] == {
        "status": "unvalidated_model_generated",
        "authoring_origin": "teaching_syllabus_generator",
        "validated_by": [],
        "validation_receipts": [],
        "authoritative_for_runtime_grading": False,
    }


def test_generation_rejects_extra_fields_and_incomplete_teaching_materials() -> None:
    extra = _draft()
    extra["unexpected"] = "unsafe"
    with pytest.raises(TeachingSyllabusError, match="extra=unexpected"):
        generate_teaching_syllabus(
            _SyllabusClient(extra),
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=90,
            source_resources=[],
        )

    missing_material = _draft()
    del missing_material["modules"][0]["lessons"][0]["materials"]["transfer_task"]
    with pytest.raises(TeachingSyllabusError, match="must contain exactly"):
        generate_teaching_syllabus(
            _SyllabusClient(missing_material),
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=90,
            source_resources=[],
        )

    gold_material = _draft()
    gold_material["modules"][0]["lessons"][0]["materials"]["practice"] = (
        "标准答案：输入是面积，输出是价格。"
    )
    with pytest.raises(TeachingSyllabusError, match="gold, or rubric"):
        generate_teaching_syllabus(
            _SyllabusClient(gold_material),
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=90,
            source_resources=[],
        )

    hidden_gold = _draft()
    hidden_gold["modules"][0]["lessons"][0]["summary"] = (
        "本课标准答案是把面积作为输入。"
    )
    with pytest.raises(TeachingSyllabusError, match="gold, or rubric"):
        generate_teaching_syllabus(
            _SyllabusClient(hidden_gold),
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=90,
            source_resources=[],
        )


@pytest.mark.parametrize(
    ("field", "instruction_only"),
    [
        (
            "summary",
            "本课将通过生活实例解释什么是机器学习，再让学生回复继续。",
        ),
        (
            "example",
            "教师应展示垃圾邮件过滤案例，让学生辨认输入与标签。",
        ),
    ],
)
def test_generation_repairs_teacher_instructions_into_learner_facing_content(
    field: str,
    instruction_only: str,
) -> None:
    invalid = _draft()
    lesson = invalid["modules"][0]["lessons"][0]
    if field == "summary":
        lesson["summary"] = instruction_only
    else:
        lesson["materials"]["example"] = instruction_only
    client = _SyllabusClient(responses=[invalid, _draft()])

    syllabus, trace = generate_teaching_syllabus(
        client,
        topic="机器学习",
        audience="大学生",
        objectives=[],
        duration_minutes=90,
        source_resources=[],
    )

    repaired_lesson = syllabus["modules"][0]["lessons"][0]
    repaired_text = (
        repaired_lesson["summary"]
        if field == "summary"
        else repaired_lesson["materials"]["example"]
    )
    assert repaired_text != instruction_only
    assert len(client.calls) == 2
    assert client.calls[1]["request_kind"] == "teaching_syllabus_generation_repair"
    assert "must be learner-facing subject content" in client.calls[1]["messages"][1][
        "content"
    ]
    assert "不能只换一种方式描述教师要做什么" in client.calls[1]["messages"][0][
        "content"
    ]
    assert trace["repair_reasons"] == ["strict_validation_failed"]

    sealed = _sealed()
    sealed_lesson = sealed["modules"][0]["lessons"][0]
    if field == "summary":
        sealed_lesson["summary"] = instruction_only
    else:
        sealed_lesson["materials"]["example"] = instruction_only
    with pytest.raises(TeachingSyllabusError, match="learner-facing subject content"):
        validate_teaching_syllabus(sealed)


def test_generation_accepts_explanatory_content_that_is_not_a_teaching_plan() -> None:
    direct = _draft()
    direct["modules"][0]["lessons"][0]["summary"] = (
        "通过比较已标注邮件与新邮件，可以看出模型先从数据归纳规律，再用于预测。"
    )

    syllabus, trace = generate_teaching_syllabus(
        _SyllabusClient(direct),
        topic="机器学习",
        audience="大学生",
        objectives=[],
        duration_minutes=90,
        source_resources=[],
    )

    assert syllabus["modules"][0]["lessons"][0]["summary"] == direct["modules"][0][
        "lessons"
    ][0]["summary"]
    assert trace["repair_attempted"] is False


def test_generation_bounds_resources_before_remote_call_and_truncates_context() -> None:
    client = _SyllabusClient()
    resources = [
        {
            "resource_id": f"res_{index}",
            "display_name": f"资源 {index}",
            "extracted_text": "材料" * 20_000,
        }
        for index in range(7)
    ]
    with pytest.raises(TeachingSyllabusError, match="at most six"):
        generate_teaching_syllabus(
            client,
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=90,
            source_resources=resources,
        )
    assert client.calls == []

    injected = deepcopy(resources[0])
    injected["resource_id"] = "res_safe\nINJECT"
    with pytest.raises(TeachingSyllabusError, match="resource_id is unsafe"):
        generate_teaching_syllabus(
            client,
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=90,
            source_resources=[injected],
        )
    assert client.calls == []

    syllabus, _ = generate_teaching_syllabus(
        client,
        topic="机器学习",
        audience="大学生",
        objectives=[],
        duration_minutes=90,
        source_resources=resources[:2],
    )
    sent = json.loads(client.calls[-1]["messages"][-1]["content"])
    assert (
        sum(len(item["extracted_text"]) for item in sent["teacher_resources"]) <= 30_000
    )
    assert syllabus["source"]["resource_ids"] == ["res_0", "res_1"]


def test_generation_is_original_only_and_never_persists_credentials() -> None:
    client = _SyllabusClient()
    client.response["modules"][0]["lessons"][0]["materials"]["practice"] = (
        "请独立作答，不要查看答案，再说明你的思路。"
    )
    syllabus, trace = generate_teaching_syllabus(
        client,
        topic="机器学习",
        audience="大学生",
        objectives=["理解监督学习"],
        duration_minutes=90,
        source_resources=[],
    )
    assert client.calls[0]["request_kind"] == "teaching_syllabus_generation"
    system = client.calls[0]["messages"][0]["content"]
    assert "不得复制 GitHub" in system
    assert "不得输出标准答案" in system
    assert "lesson.summary 与 materials.example 会原样展示给学习者" in system
    assert "本阶段不考" in system
    assert trace["credential_logged"] is False
    assert "api_key" not in trace
    assert "must-not-escape" not in json.dumps(syllabus, ensure_ascii=False)


def test_generation_binds_topic_and_duration_without_overfitting_level_wording() -> (
    None
):
    reasonable = _draft()
    reasonable["title"] = "机器学习基础"
    syllabus, _ = generate_teaching_syllabus(
        _SyllabusClient(reasonable),
        topic="机器学习入门",
        audience="大学生",
        objectives=[],
        duration_minutes=90,
        source_resources=[],
    )
    assert syllabus["title"] == "机器学习基础"

    english = _draft()
    english["title"] = "Machine Learning Foundations"
    syllabus, _ = generate_teaching_syllabus(
        _SyllabusClient(english),
        topic="Design an introduction to machine learning for beginners",
        audience="undergraduates",
        objectives=[],
        duration_minutes=90,
        source_resources=[],
    )
    assert syllabus["title"] == "Machine Learning Foundations"

    syllabus, _ = generate_teaching_syllabus(
        _SyllabusClient(_draft()),
        topic="请帮我为零基础大学生设计一份机器学习入门教学大纲",
        audience="零基础大学生",
        objectives=[],
        duration_minutes=90,
        source_resources=[],
    )
    assert syllabus["title"] == "机器学习入门"

    unrelated = _draft()
    unrelated.update(
        {
            "title": "法式烘焙",
            "description": "学习面团发酵与烤箱温度。",
            "estimated_duration_minutes": 20_000,
            "learning_objectives": ["制作法棍"],
        }
    )
    lesson = unrelated["modules"][0]["lessons"][0]
    unrelated["modules"][0]["title"] = "烘焙基础"
    unrelated["modules"][0]["description"] = "认识面粉和酵母。"
    lesson.update(
        {
            "title": "揉面",
            "objective": "掌握基础揉面动作。",
            "summary": "观察面团状态。",
            "duration_minutes": 5,
            "knowledge_components": ["面粉", "酵母"],
        }
    )
    with pytest.raises(TeachingSyllabusError, match="requested topic"):
        generate_teaching_syllabus(
            _SyllabusClient(unrelated),
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=60,
            source_resources=[],
        )

    wrong_duration = _draft()
    wrong_duration["estimated_duration_minutes"] = 20_000
    wrong_duration["modules"][0]["lessons"][0]["duration_minutes"] = 5
    with pytest.raises(TeachingSyllabusError, match="estimated duration"):
        generate_teaching_syllabus(
            _SyllabusClient(wrong_duration),
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=60,
            source_resources=[],
        )


def test_generation_repairs_one_invalid_shape_without_weakening_final_validation() -> (
    None
):
    invalid = _draft()
    invalid["prerequisites"] = "能阅读简单表格"
    client = _SyllabusClient(responses=[invalid, _draft()])

    syllabus, trace = generate_teaching_syllabus(
        client,
        topic="机器学习",
        audience="大学生",
        objectives=[],
        duration_minutes=90,
        source_resources=[],
    )

    assert syllabus["prerequisites"] == ["能阅读简单表格"]
    assert len(client.calls) == 2
    assert client.calls[1]["request_kind"] == "teaching_syllabus_generation_repair"
    assert (
        "prerequisites must contain between 0 and 16 strings"
        in client.calls[1]["messages"][1]["content"]
    )
    assert trace["generation_attempt_count"] == 2
    assert trace["repair_attempted"] is True
    assert trace["repair_succeeded"] is True

    sealed = _sealed()
    sealed["prerequisites"] = "能阅读简单表格"
    with pytest.raises(TeachingSyllabusError, match="between 0 and 16 strings"):
        validate_teaching_syllabus(sealed)


def test_generation_retries_one_malformed_structured_response() -> None:
    client = _SyllabusClient(
        responses=[
            DeepSeekClientError("DeepSeek returned malformed structured output"),
            _draft(),
        ]
    )

    syllabus, trace = generate_teaching_syllabus(
        client,
        topic="机器学习",
        audience="大学生",
        objectives=[],
        duration_minutes=90,
        source_resources=[],
    )

    assert syllabus["title"] == "机器学习入门"
    assert len(client.calls) == 2
    assert (
        "DeepSeek returned malformed structured output"
        in client.calls[1]["messages"][1]["content"]
    )
    assert trace["generation_attempt_count"] == 2
    assert trace["repair_reasons"] == ["malformed_structured_output"]


def test_generation_stops_after_three_invalid_model_outputs() -> None:
    invalid = _draft()
    invalid["prerequisites"] = "能阅读简单表格"
    client = _SyllabusClient(invalid)

    with pytest.raises(TeachingSyllabusError, match="between 0 and 16 strings"):
        generate_teaching_syllabus(
            client,
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=90,
            source_resources=[],
        )

    assert len(client.calls) == 3
    assert all(
        call["request_kind"] == "teaching_syllabus_generation_repair"
        for call in client.calls[1:]
    )


def test_generation_uses_a_syllabus_only_output_budget() -> None:
    captured: dict[str, object] = {}

    def transport(_url: str, _headers: dict, payload: bytes, _timeout: float):
        captured["payload"] = payload
        return 200, json.dumps(
            {
                "id": "syllabus-response",
                "choices": [
                    {"message": {"content": json.dumps(_draft(), ensure_ascii=False)}}
                ],
                "usage": {"prompt_tokens": 10, "completion_tokens": 20},
            },
            ensure_ascii=False,
        ).encode()

    client = DeepSeekClient(
        DeepSeekConfig(allow_remote_student_data=True),
        api_key="secret-test-key",
        transport=transport,
    )
    generate_teaching_syllabus(
        client,
        topic="机器学习",
        audience="大学生",
        objectives=[],
        duration_minutes=90,
        source_resources=[],
    )

    request_body = json.loads(captured["payload"])
    assert request_body["max_tokens"] == 8_192
    assert client.config.max_tokens == 1_800


def test_syllabus_identity_binds_provenance_while_outline_hash_stays_structural() -> (
    None
):
    first = _seal_generated_syllabus(
        _draft(),
        model="deepseek-v4-flash",
        source_resource_ids=["res_alpha"],
        created_at="2026-08-11T12:00:00Z",
    )
    later = _seal_generated_syllabus(
        _draft(),
        model="deepseek-v4-flash",
        source_resource_ids=["res_alpha"],
        created_at="2026-08-11T12:00:01Z",
    )
    other_model = _seal_generated_syllabus(
        _draft(),
        model="deepseek-v4-flash-revalidated",
        source_resource_ids=["res_alpha"],
        created_at="2026-08-11T12:00:00Z",
    )
    other_resource = _seal_generated_syllabus(
        _draft(),
        model="deepseek-v4-flash",
        source_resource_ids=["res_beta"],
        created_at="2026-08-11T12:00:00Z",
    )
    assert (
        len(
            {
                first["syllabus_id"],
                later["syllabus_id"],
                other_model["syllabus_id"],
                other_resource["syllabus_id"],
            }
        )
        == 4
    )
    assert {
        first["integrity"]["outline_sha256"],
        later["integrity"]["outline_sha256"],
        other_model["integrity"]["outline_sha256"],
        other_resource["integrity"]["outline_sha256"],
    } == {first["integrity"]["outline_sha256"]}
    assert (
        first["modules"][0]["lessons"][0]["teaching_goal"]["syllabus_ref"][
            "content_sha256"
        ]
        == first["integrity"]["outline_sha256"]
    )


def test_public_generator_defaults_match_auxiliary_optional_contract() -> None:
    draft = _draft()
    draft["estimated_duration_minutes"] = 120
    draft["modules"][0]["lessons"][0]["duration_minutes"] = 120
    syllabus, _ = generate_teaching_syllabus(_SyllabusClient(draft), topic="机器学习")
    assert syllabus["audience"] == draft["audience"]
    assert syllabus["estimated_duration_minutes"] == 120


def test_timestamp_and_optional_array_types_fail_closed_before_remote_call() -> None:
    invalid_timestamp = _sealed()
    invalid_timestamp["created_at"] = "2026-99-99T99:99:99Z"
    with pytest.raises(TeachingSyllabusError, match="not a real UTC timestamp"):
        validate_teaching_syllabus(invalid_timestamp)

    client = _SyllabusClient()
    with pytest.raises(TeachingSyllabusError, match="objectives must be an array"):
        generate_teaching_syllabus(
            client,
            topic="机器学习",
            audience="大学生",
            objectives=None,  # type: ignore[arg-type]
            duration_minutes=90,
            source_resources=[],
        )
    with pytest.raises(
        TeachingSyllabusError, match="source_resources must be an array"
    ):
        generate_teaching_syllabus(
            client,
            topic="机器学习",
            audience="大学生",
            objectives=[],
            duration_minutes=90,
            source_resources=None,  # type: ignore[arg-type]
        )
    assert client.calls == []


def test_store_writes_one_atomic_json_file_and_lists_full_documents() -> None:
    syllabus = _sealed()
    with TemporaryDirectory() as directory:
        store = TeachingSyllabusStore(directory)
        assert store.save(syllabus) is True
        assert store.save(syllabus) is False
        files = list(Path(directory).iterdir())
        assert [path.name for path in files] == [f"{syllabus['syllabus_id']}.json"]
        listed = store.list()
        assert listed[0]["modules"][0]["lessons"][0]["materials"]["example"]
        assert store.read(syllabus["syllabus_id"]) == syllabus

        corrupted = deepcopy(syllabus)
        corrupted["title"] = "被篡改"
        Path(files[0]).write_text(json.dumps(corrupted), encoding="utf-8")
        with pytest.raises(TeachingSyllabusError, match="syllabus_id does not match"):
            store.read(syllabus["syllabus_id"])


def test_lesson_start_payload_and_session_keep_auditable_syllabus_ref() -> None:
    root = project_root()
    syllabus = _sealed()
    payload = syllabus_lesson_start_payload(syllabus, "lesson_01_01")
    assert payload["goal"] == payload["teaching_goal"]
    assert payload["syllabus_ref"] == payload["goal"]["syllabus_ref"]
    assert (
        payload["goal"]["syllabus_ref"]["content_sha256"]
        == syllabus["integrity"]["outline_sha256"]
    )
    library = read_json(root / "data/teacher_agent_skill_library_v2.json")
    profile = read_json(root / "data/teacher_agent_demo_input.json")["student_profile"]
    session = start_teacher_agent_session(payload["goal"], profile, library)
    assert session["goal"]["syllabus_ref"] == payload["goal"]["syllabus_ref"]
    assert session["goal"]["knowledge_spec"]["status"] == "generated_unvalidated"
    assert session["goal"]["knowledge_spec"]["rubric_criteria"]
    assert (
        session["goal"]["knowledge_spec"]["claim_boundary"][
            "authoritative_for_runtime_grading"
        ]
        is False
    )
    context = build_layered_context(session, None)
    assert (
        context["fixed_context"]["teaching_goal"]["syllabus_ref"]
        == payload["goal"]["syllabus_ref"]
    )


def test_legacy_syllabus_projects_non_authoritative_rubric_without_rewriting() -> None:
    syllabus = _sealed()
    stored_goal = syllabus["modules"][0]["lessons"][0]["teaching_goal"]
    stored_goal.pop("knowledge_spec")
    material = deepcopy(syllabus)
    material.pop("integrity")
    syllabus["integrity"]["content_sha256"] = sha256(
        json.dumps(
            material,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    before = deepcopy(syllabus)

    validate_teaching_syllabus(syllabus)
    payload = syllabus_lesson_start_payload(syllabus, "lesson_01_01")

    assert syllabus == before
    assert "knowledge_spec" not in stored_goal
    assert payload["goal"]["knowledge_spec"]["authority"] == {
        "status": "unvalidated_model_generated",
        "authoring_origin": "teaching_syllabus_generator",
        "validated_by": [],
        "validation_receipts": [],
        "authoritative_for_runtime_grading": False,
    }


def test_syllabus_to_teach_model_quote_cannot_self_authorize_mastery() -> None:
    root = project_root()
    payload = syllabus_lesson_start_payload(_sealed(), "lesson_01_01")
    absurd = "因为月亮引力会把两个状态吸在一起，所以必须取最小。"
    final = _live_plan(
        signal="correct",
        confidence=0.97,
        skill_id="skill_socratic_understanding_check",
        action_type="socratic_comprehension_probe",
        message="请再说明一次你使用的依据。",
        evidence_excerpt=absurd,
    )
    client = _LivePlanClient(_lesson_live_plans(final))
    library = read_json(root / "data/teacher_agent_skill_library_v2.json")
    profile = read_json(root / "data/teacher_agent_demo_input.json")["student_profile"]
    session = _advance_to_guided_practice(payload["goal"], profile, library, client)
    mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])
    phase_before = session["lesson_state"]["lesson_phase"]
    status_before = session["status"]
    termination_before = session["control"]["termination_reason"]

    updated = advance_live_teacher_agent_session(
        session,
        learner_response=absurd,
        client=client,
    )

    assessment = updated["history"][-1]["deepseek_assessment"]
    assert assessment["model_raw_signal"] == "correct"
    assert assessment["signal"] == "partial"
    assert assessment["assessment_observation_status"] == "not_observed"
    assert assessment["needs_human_review"] is True
    assert assessment["evidence_binding_source"] == (
        "model_excerpt_current_response_substring"
    )
    assert assessment["evidence_semantics"] == "learner_text_provenance_only"
    assert assessment["semantic_entailment_established"] is False
    assert assessment["teacher_grading_authority_available"] is False
    assert (
        "high_impact_diagnosis_without_authoritative_entailment_downgraded"
        in assessment["normalization_reasons"]
    )
    assert updated["student_state"]["knowledge_mastery"] == mastery_before
    assert updated["lesson_state"]["lesson_phase"] == phase_before
    assert updated["status"] == status_before
    assert updated["control"]["termination_reason"] == termination_before
    assert updated["history"][-1]["structured_signal"]["applied_to_mastery"] is False


def test_ad_hoc_cross_domain_spec_is_observed_but_never_grants_progress() -> None:
    root = project_root()
    biology = _draft()
    biology.update(
        {
            "title": "植物生物学入门",
            "description": "理解植物如何利用光能合成有机物。",
            "learning_objectives": ["解释光合作用的输入、能量来源与产物"],
        }
    )
    module = biology["modules"][0]
    module.update({"title": "光合作用", "description": "建立物质与能量关系。"})
    lesson = module["lessons"][0]
    lesson.update(
        {
            "title": "光合作用的物质与能量",
            "objective": "能说明光合作用利用光能把二氧化碳和水转化为有机物。",
            "summary": "区分反应输入、能量来源和主要产物。",
            "knowledge_components": ["光合作用物质与能量关系"],
            "materials": {
                "example": "观察叶片在光照条件下合成有机物的情境。",
                "practice": "标出反应的输入、能量来源和产物。",
                "transfer_task": "解释缺少光照时合成过程为何受限。",
            },
        }
    )
    payload = syllabus_lesson_start_payload(
        _seal_generated_syllabus(
            biology,
            model="deepseek-v4-flash",
            source_resource_ids=[],
            created_at="2026-08-11T12:00:02Z",
        ),
        "lesson_01_01",
    )
    teacher_claim = "光合作用利用光能把二氧化碳和水转化为有机物。"
    payload["goal"]["knowledge_spec"] = {
        "canonical_claims": [
            {
                "claim_id": "claim_photosynthesis_inputs",
                "statement": teacher_claim,
                "knowledge_components": ["光合作用物质与能量关系"],
                "source_ids": ["teacher_biology_note"],
            }
        ],
        "rubric_criteria": [
            {
                "criterion_id": "criterion_photosynthesis_relation",
                "description": "同时说明光能、二氧化碳、水和有机物之间的关系",
                "knowledge_component": "光合作用物质与能量关系",
                "acceptable_evidence": [teacher_claim],
            }
        ],
        "sources": [
            {
                "source_id": "teacher_biology_note",
                "title": "教师审定的生物学课节说明",
                "citation": "教师本地课程说明第 1 节",
            }
        ],
    }
    conservative = _live_plan(
        signal="partial",
        confidence=0.45,
        skill_id="skill_socratic_understanding_check",
        action_type="socratic_comprehension_probe",
        message="请说明输入、能量来源和产物之间的关系。",
        evidence_excerpt=teacher_claim,
    )
    client = _LivePlanClient(_lesson_live_plans(conservative))
    library = read_json(root / "data/teacher_agent_skill_library_v2.json")
    profile = read_json(root / "data/teacher_agent_demo_input.json")["student_profile"]
    session = _advance_to_guided_practice(payload["goal"], profile, library, client)
    mastery_before = deepcopy(session["student_state"]["knowledge_mastery"])

    updated = advance_live_teacher_agent_session(
        session,
        learner_response=teacher_claim,
        client=client,
    )

    assessment = updated["history"][-1]["deepseek_assessment"]
    assert assessment["signal"] == "correct"
    assert assessment["assessment_source"] == "teacher_knowledge_spec_exact_match"
    assert assessment["semantic_entailment_established"] is True
    # A browser/local caller can still supply reference text for deterministic
    # observation, but only the server-injected sealed curriculum projection may
    # authorize a mastery mutation.
    assert assessment["teacher_grading_authority_available"] is False
    assert assessment["needs_human_review"] is False
    assert updated["lesson_state"]["lesson_phase"] == "guided_practice"
    assert updated["student_state"]["knowledge_mastery"] == mastery_before
    assert updated["history"][-1]["structured_signal"]["applied_to_mastery"] is False


def test_dashboard_http_generate_list_download_import_and_start_payload() -> None:
    root = project_root()
    with TemporaryDirectory() as directory:
        snapshot = build_teacher_agent_dashboard_snapshot(
            root / "data/teacher_agent_skill_library_v2.json",
            root / "data/teacher_agent_demo_input.json",
            root / "data/teacher_agent_evaluation_cases.json",
            client=_SyllabusClient(),
            syllabus_store_path=directory,
            consent_store_path=Path(directory) / "remote-consent.json",
            consent_signing_secret=(
                b"syllabus-test-consent-signing-secret-material-32-bytes"
            ),
        )
        consent = snapshot.grant_remote_consent(
            {
                "purpose": "remote_syllabus_generation",
                "validity_days": 1,
                "likely_minor": False,
                "guardian_or_school_policy": "not_required",
            }
        )["receipt"]
        bootstrap = snapshot.bootstrap()
        assert (
            bootstrap["auxiliary_skills"][0]["skill_id"]
            == (TEACHING_SYLLABUS_AUXILIARY_SKILL["skill_id"])
        )
        assert len(bootstrap["skills"]) == len(snapshot.library["skills"])
        assert bootstrap["interaction_contract"]["teaching_syllabus_enabled"] is True

        server, base_url = create_teacher_agent_dashboard_server(snapshot)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, generated, _ = _post(
                base_url,
                "api/syllabi/generate",
                {
                    "topic": "机器学习",
                    "audience": "大学生",
                    "objectives": ["理解监督学习"],
                    "duration_minutes": 90,
                    "source_resource_ids": [],
                    "remote_consent_id": consent["consent_id"],
                },
            )
            assert status == 200
            syllabus = generated["syllabus"]
            syllabus_id = syllabus["syllabus_id"]

            status, listed, _ = _get(base_url, "api/syllabi")
            assert status == 200
            assert listed["syllabi"][0]["modules"]

            status, downloaded, headers = _get(
                base_url, f"api/syllabi/{syllabus_id}/download"
            )
            assert status == 200
            assert downloaded == syllabus
            assert "attachment" in headers["Content-Disposition"]

            status, imported, _ = _post(
                base_url, "api/syllabi/import", {"syllabus": downloaded}
            )
            assert status == 200
            assert imported["created"] is False

            status, start_payload, _ = _get(
                base_url,
                f"api/syllabi/{syllabus_id}/lessons/lesson_01_01/start-payload",
            )
            assert status == 200
            assert start_payload["goal"]["syllabus_ref"]["syllabus_id"] == syllabus_id
            assert (
                snapshot._bind_start_goal_to_syllabus(
                    {
                        "syllabus_ref": start_payload["goal"]["syllabus_ref"],
                    },
                    start_payload["goal"],
                )
                == start_payload["goal"]
            )

            editable = teaching_syllabus_editable_draft(syllabus)
            editable["title"] = "机器学习入门（教师修订）"
            revision_request = {
                "editable_draft": editable,
                "change_summary": "明确教师修订标题",
                "expected_version": generated["version_family"]["version"],
                "idempotency_key": "http-create-syllabus-revision",
            }
            status, revision, _ = _post(
                base_url,
                f"api/syllabi/{syllabus_id}/revisions",
                revision_request,
            )
            assert status == 200
            revised = revision["syllabus"]
            revised_id = revised["syllabus_id"]
            revision_row = revision["version_family"]["revisions"][-1]
            assert revision_row["status"] == "draft"
            status, replayed, _ = _post(
                base_url,
                f"api/syllabi/{syllabus_id}/revisions",
                revision_request,
            )
            assert status == 200
            assert replayed["syllabus"] == revision["syllabus"]
            assert replayed["version_family"] == revision["version_family"]
            assert replayed["created"] is False
            status, stale, _ = _post(
                base_url,
                f"api/syllabi/{syllabus_id}/publish",
                {
                    "revision_id": revision_row["revision_id"],
                    "expected_version": generated["version_family"]["version"],
                    "idempotency_key": "http-stale-publish-syllabus-revision",
                },
            )
            assert status == 409
            assert "version conflict" in stale["error"]
            status, _, _ = _get(
                base_url,
                f"api/syllabi/{revised_id}/lessons/lesson_01_01/start-payload",
            )
            assert status == 400

            status, published, _ = _post(
                base_url,
                f"api/syllabi/{syllabus_id}/publish",
                {
                    "revision_id": revision_row["revision_id"],
                    "expected_version": revision["version_family"]["version"],
                    "idempotency_key": "http-publish-syllabus-revision",
                },
            )
            assert status == 200
            assert published["syllabus"]["syllabus_id"] == revised_id
            status, _, _ = _get(
                base_url,
                f"api/syllabi/{syllabus_id}/lessons/lesson_01_01/start-payload",
            )
            assert status == 400
            status, _, _ = _get(
                base_url,
                f"api/syllabi/{revised_id}/lessons/lesson_01_01/start-payload",
            )
            assert status == 200

            status, versions, _ = _get(base_url, f"api/syllabi/{revised_id}/versions")
            assert status == 200
            assert len(versions["version_family"]["revisions"]) == 2
            assert len(versions["syllabi"]) == 2
            status, curriculum, _ = _get(
                base_url,
                f"api/syllabi/{revised_id}/curriculum-blueprint",
            )
            assert status == 200
            assert curriculum["authoritative_for_runtime_grading"] is False
            assert curriculum["curriculum_blueprint"]["authority"] == {
                "status": "generated_unvalidated",
                "authority": False,
                "authoritative_for_runtime_grading": False,
                "receipt": None,
            }
            status, rolled_back, _ = _post(
                base_url,
                f"api/syllabi/{revised_id}/rollback",
                {
                    "revision_id": generated["version_family"]["revisions"][0][
                        "revision_id"
                    ],
                    "expected_version": published["version_family"]["version"],
                    "idempotency_key": "http-rollback-syllabus-revision",
                },
            )
            assert status == 200
            assert rolled_back["syllabus"]["syllabus_id"] == syllabus_id
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
        # Exercise the exact frontend hand-off against a deterministic backend:
        # GET start-payload -> POST start, with the complete top-level ref.
        offline = build_teacher_agent_dashboard_snapshot(
            root / "data/teacher_agent_skill_library_v2.json",
            root / "data/teacher_agent_demo_input.json",
            root / "data/teacher_agent_evaluation_cases.json",
            syllabus_store_path=directory,
        )
        server, base_url = create_teacher_agent_dashboard_server(offline)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            status, authoritative, _ = _get(
                base_url,
                f"api/syllabi/{syllabus_id}/lessons/lesson_01_01/start-payload",
            )
            assert status == 200
            demo = read_json(root / "data/teacher_agent_demo_input.json")
            status, started, _ = _post(
                base_url,
                "api/start",
                {
                    "goal": authoritative["goal"],
                    "syllabus_ref": authoritative["syllabus_ref"],
                    "student_profile": demo["student_profile"],
                    "start_idempotency_key": "syllabus-start-test",
                },
            )
            assert status == 200
            assert (
                started["setup_snapshot"]["goal"]["syllabus_ref"]
                == authoritative["syllabus_ref"]
            )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


def test_syllabus_import_reconciles_version_ledger_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = project_root()
    snapshot = build_teacher_agent_dashboard_snapshot(
        root / "data/teacher_agent_skill_library_v2.json",
        root / "data/teacher_agent_demo_input.json",
        root / "data/teacher_agent_evaluation_cases.json",
        syllabus_store_path=tmp_path,
    )
    first = _seal_generated_syllabus(
        _draft(),
        model="deepseek-v4-flash",
        source_resource_ids=[],
        created_at="2026-08-12T03:00:00Z",
    )

    def fail_before_commit(*args, **kwargs):
        del args, kwargs
        raise TeachingSyllabusVersionError("injected version failure")

    monkeypatch.setattr(TeachingSyllabusVersionStore, "register", fail_before_commit)
    with pytest.raises(TeacherAgentDashboardError, match="injected version failure"):
        snapshot.import_syllabus({"syllabus": first})
    assert not snapshot.syllabus_store._path(first["syllabus_id"]).exists()

    monkeypatch.undo()
    second = _seal_generated_syllabus(
        _draft(),
        model="deepseek-v4-flash",
        source_resource_ids=[],
        created_at="2026-08-12T03:00:01Z",
    )
    original = TeachingSyllabusVersionStore.register

    def fail_after_commit(store, *args, **kwargs):
        original(store, *args, **kwargs)
        raise OSError("injected acknowledgement loss")

    monkeypatch.setattr(TeachingSyllabusVersionStore, "register", fail_after_commit)
    with pytest.raises(OSError, match="acknowledgement loss"):
        snapshot.import_syllabus({"syllabus": second})
    assert snapshot.syllabus_store.read(second["syllabus_id"]) == second
    assert (
        snapshot.syllabus_version_store.read_family(second["syllabus_id"])["revisions"][
            0
        ]["syllabus_id"]
        == second["syllabus_id"]
    )
