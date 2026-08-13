"""Deterministic, stateful Teaching Agent for task two.

The module intentionally separates three responsibilities:

* a structured learner signal is supplied by a teacher or external judge;
* the Agent updates an explicit student model;
* a deterministic policy selects exactly one next teaching action.

It does not claim to grade arbitrary free-form answers or establish real learner
effects.  Those boundaries are written into every persisted session and report.
"""

from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
import math
import re
from typing import Any, Mapping

from .teacher_agent_discourse import classify_learner_discourse_text
from .student_model import (
    LEGACY_STUDENT_MODEL_SCHEMA,
    STUDENT_MODEL_SCHEMA,
    StudentModelError,
    student_model_mastery_readiness,
    stable_knowledge_component_id,
    validate_student_model,
)


SESSION_SCHEMA = "teaching_skill_miner.teacher_agent_session.v1"
LIBRARY_SCHEMA = "teaching_skill_miner.teacher_agent_skill_library.v1"
LIBRARY_SCHEMA_V2 = "teaching_skill_miner.teacher_agent_skill_library.v2"
LIBRARY_SCHEMAS = frozenset({LIBRARY_SCHEMA, LIBRARY_SCHEMA_V2})
EVALUATION_SCHEMA = "teaching_skill_miner.teacher_agent_evaluation.v1"
POLICIES = {"adaptive_skill_library", "fixed_single_skill_baseline"}
SIGNALS = {"correct", "partial", "misconception", "confused", "no_response"}
MASTERY_DIMENSIONS = ("prerequisite", "conceptual", "procedural", "transfer")
TERMINAL_STATUSES = {"succeeded", "terminated_unable"}
ADAPTIVE_PROFILE_SCHEMA = "teaching_skill_miner.adaptive_student_profile_summary.v1"
ADAPTIVE_OBSERVATION_LIMIT = 12
ADAPTIVE_OBSERVATION_SOURCE = "deepseek_v4_flash_validated_diagnosis"
ADAPTIVE_OBSERVATION_STATUS = "candidate_unconfirmed"
KNOWLEDGE_SPEC_SCHEMA = "teaching_skill_miner.teacher_goal_knowledge_spec.v1"
KNOWLEDGE_SPEC_MAX_ITEMS = 24
CURRICULUM_RUNTIME_AUTHORITY_SCHEMA = (
    "teaching_skill_miner.curriculum_runtime_authority.v1"
)
PRIMARY_ROLES = {
    "diagnostic",
    "context",
    "example",
    "concept_mapping",
    "scaffolding",
    "assessment",
    "correction",
    "practice",
    "review",
    "metacognition",
    "engagement",
    "transfer",
    "summary",
}

LEARNING_INTENTS = {
    "teach_first",
    "diagnostic_first",
    "task_first",
    "legacy_diagnostic_first",
}
LESSON_STATE_SCHEMA = "teaching_skill_miner.teacher_agent_lesson_state.v1"
LESSON_PHASES = (
    "orientation",
    "explanation",
    "worked_example",
    "guided_practice",
    "verification",
    "transfer",
)
LESSON_PHASE_LABELS = {
    "orientation": "导入",
    "explanation": "讲解",
    "worked_example": "示范",
    "guided_practice": "带练",
    "verification": "核验",
    "transfer": "迁移",
}
LESSON_PHASE_REASONS = {
    "orientation": "先明确学习目标和路径，不要求回忆尚未讲解的内容。",
    "explanation": "先讲清核心概念、条件和关键关系。",
    "worked_example": "由教师完整示范一个最小例子，再观察关键结构。",
    "guided_practice": "师生一起完成一个可检查的步骤。",
    "verification": "只根据学生自己的解释或作答核验理解。",
    "transfer": "把已核验的方法应用到一个新情境。",
}
# The teach-first closing sequence reserves one real summary attempt plus at
# most two bounded recovery responses (live mode uses scaffold then retry).  It
# is deliberately separate from goal.max_rounds so a transfer accepted on the
# last regular round cannot create an obligation the learner cannot meet.
SUMMARY_CLOSURE_ROUND_LIMIT = 3
LESSON_EXPOSURE_STAGES = {
    "unseen": 0,
    "explained": 1,
    "example_seen": 2,
    "guided_practice_completed": 3,
    "learner_verified": 4,
}
_LESSON_NAVIGATION_RE = re.compile(
    r"^(?:继续|下一步|开始吧?|可以|好的?|好呀|行|嗯+|收到|明白了?|懂了|"
    r"请继续|接着讲|往下讲|看例子|看示范|开始带练|开始练习|进入下一步)[！!。.]?$",
    re.IGNORECASE,
)
_LESSON_CLARIFICATION_KIND_PATTERNS = (
    (
        "symbol_meaning",
        re.compile(
            r"(?:表示|代表)(?:的)?(?:是)?什么|"
            r"(?:符号|字母|变量|参数|下标).{0,12}"
            r"(?:含义|作用)(?:是)?什么|"
            r"(?:符号|字母|变量|参数|下标).{0,12}"
            r"(?:怎么|如何)读",
            re.IGNORECASE,
        ),
    ),
    (
        "definition",
        re.compile(
            r"(?:什么是|是什么意思|是什么概念|什么意思|"
            r"定义(?:是)?什么|含义是什么|指(?:的)?是什么|"
            r"(?:怎么|如何)理解|"
            r"(?:原理|机制|作用)(?:是)?什么)",
            re.IGNORECASE,
        ),
    ),
    (
        "rationale",
        re.compile(r"(?:为什么|为何|原因(?:是)?什么)", re.IGNORECASE),
    ),
    (
        "procedure",
        re.compile(
            r"(?:分|有|包括|包含).{0,6}(?:哪几|哪些|几)(?:个)?"
            r"(?:步|步骤|阶段|环节)|"
            r"(?:步骤|流程)(?:是什么|有哪些|有哪几|有几)|"
            r"(?:怎么|如何)(?:做|操作|计算|推导|判断|选择|设置|"
            r"使用|开始|分解|实现|运行|工作)",
            re.IGNORECASE,
        ),
    ),
    (
        "composition",
        re.compile(
            r"(?:分|有|包括|包含|由).{0,8}(?:哪几|哪些|几)(?:个)?"
            r"(?:种|部分|方面|类|要素|成分|内容)|"
            r"(?:哪几|哪些|几)(?:个)?(?:种|部分|方面|类|要素|成分)|"
            r"由(?:哪些|什么).{0,12}(?:组成|构成)|"
            r"(?:包括|包含)(?:哪些|什么)",
            re.IGNORECASE,
        ),
    ),
    (
        "comparison",
        re.compile(
            r"(?:有什么|有何)(?:区别|差别|不同)|"
            r"(?:区别|差别|不同)(?:是什么|在哪|有哪些)|"
            r"(?:怎么|如何)区分",
            re.IGNORECASE,
        ),
    ),
    (
        "example_request",
        re.compile(
            r"(?:举|给).{0,6}(?:个|一个)?(?:例子|示例)|"
            r"(?:例子|示例)(?:是什么|有哪些)",
            re.IGNORECASE,
        ),
    ),
)
_LESSON_FINAL_ANSWER_REQUEST_RE = re.compile(
    r"(?:直接)?(?:告诉|给|说出|公布).{0,8}(?:答案|解法|结果)|"
    r"(?:最终|完整|标准|正确)(?:答案|解法|结果)|"
    r"(?:这题|这道题|当前题|这个练习|当前任务).{0,10}"
    r"(?:怎么做|如何做|解一下|做完|写完)|"
    r"(?:下一步|这一步).{0,8}(?:怎么做|怎么写|直接写)|"
    r"(?:把|帮我).{0,12}(?:答案|解法|代码|题).{0,8}(?:写出|写完|做完)",
    re.IGNORECASE,
)
_LESSON_CLARIFICATION_DECLARATION_RE = re.compile(
    r"^(?:我|我们)(?:已经|现在|基本)?"
    r"(?:知道|明白|理解|能(?:够)?解释|可以解释).{0,120}"
    r"(?:为什么|什么意思|表示什么|哪几|几(?:个|种|部分)|"
    r"由什么|包括哪些)"
    r"|^(?:我|我们).{0,40}(?:把|将).{0,40}(?:分成|分为).{0,20}"
    r"(?:个|部分)(?:了)?[！!。.]*$",
    re.IGNORECASE,
)
_LESSON_CONTRAST_QUESTION_RE = re.compile(
    r"(?:但|但是|不过|可是).{0,80}"
    r"(?:为什么|为何|什么意思|表示什么|代表什么|"
    r"由什么组成|包括哪些|怎么|如何)",
    re.IGNORECASE,
)
_LESSON_TASK_SOLUTION_SCOPE_RE = re.compile(
    r"(?:这题|这道题|当前题|这个练习|当前任务|"
    r"这一步|下一步|这里|此处|这样|这么|这个式子|"
    r"这段推导|这段证明|我的答案|这段代码)",
    re.IGNORECASE,
)
_LESSON_REPETITION_COMPLAINT_RE = re.compile(
    r"(?:我)?(?:之前|刚才)不是(?:已经)?说.{0,32}(?:吗|过)|"
    r"(?:怎么|为什么).{0,16}(?:又问|重复|还问)|"
    r"(?:同一个问题|刚才的问题).{0,16}(?:又|重复|再).{0,8}(?:问|说)|"
    r"(?:再说一遍|重复一遍).{0,8}(?:是什么意思|干什么|有意义吗)",
    re.IGNORECASE,
)


class TeacherAgentError(ValueError):
    """Raised when a session or public fixture violates the task-two contract."""


class _TemplateValues(dict[str, str]):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def _canonical_short_identifier(value: Any) -> str:
    """Normalize a bounded taxonomy identifier for collision checks."""

    return re.sub(r"[\s\W_]+", "", str(value)).casefold()


def _finite_probability(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TeacherAgentError(f"{field} must be a number in [0, 1]")
    result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise TeacherAgentError(f"{field} must be a finite number in [0, 1]")
    return result


def _nonempty_string(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TeacherAgentError(f"{field} must be a non-empty string")
    return value.strip()


def _string_list(value: Any, *, field: str, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value):
        qualifier = "a list" if allow_empty else "a non-empty list"
        raise TeacherAgentError(f"{field} must be {qualifier}")
    result = [_nonempty_string(item, field=f"{field}[]") for item in value]
    if len(result) != len(set(result)):
        raise TeacherAgentError(f"{field} must not contain duplicates")
    return result


def _optional_string_list(
    value: Any,
    *,
    field: str,
    maximum_items: int = KNOWLEDGE_SPEC_MAX_ITEMS,
) -> list[str]:
    """Validate an optional bounded string list without inventing content."""

    if value is None:
        return []
    result = _string_list(value, field=field, allow_empty=True)
    if len(result) > maximum_items:
        raise TeacherAgentError(f"{field} must contain at most {maximum_items} items")
    return result


def _strict_boolean(value: Any, *, field: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise TeacherAgentError(f"{field} must be a boolean")
    return value


def _bounded_identifier(value: Any, *, field: str, fallback: str) -> str:
    identifier = str(value or fallback).strip()
    if not identifier or len(identifier) > 120:
        raise TeacherAgentError(f"{field} must be a non-empty identifier <= 120 chars")
    return identifier


def _normalize_knowledge_spec(
    value: Any,
    *,
    knowledge_components: list[str],
) -> dict[str, Any]:
    """Normalize teacher-authored domain truth and criterion-level grading aids.

    The structure is deliberately optional.  When it is absent the runtime says
    so explicitly instead of pretending that model memory is an authoritative
    answer key.  When present, every item retains teacher/import provenance; the
    system still does not claim that the material has been independently
    verified.
    """

    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise TeacherAgentError("goal.knowledge_spec must be an object")

    def raw_items(field: str) -> list[Any]:
        raw = value.get(field, [])
        if not isinstance(raw, list):
            raise TeacherAgentError(f"goal.knowledge_spec.{field} must be a list")
        if len(raw) > KNOWLEDGE_SPEC_MAX_ITEMS:
            raise TeacherAgentError(
                f"goal.knowledge_spec.{field} must contain at most "
                f"{KNOWLEDGE_SPEC_MAX_ITEMS} items"
            )
        return raw

    def ensure_unique(rows: list[dict[str, Any]], key: str, field: str) -> None:
        identifiers = [str(row[key]) for row in rows]
        if len(identifiers) != len(set(identifiers)):
            raise TeacherAgentError(
                f"goal.knowledge_spec.{field} identifiers must be unique"
            )

    canonical_claims: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items("canonical_claims")):
        if isinstance(item, str):
            item = {"statement": item}
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"goal.knowledge_spec.canonical_claims[{index}] must be an object"
            )
        components = _optional_string_list(
            item.get("knowledge_components", []),
            field=(
                f"goal.knowledge_spec.canonical_claims[{index}].knowledge_components"
            ),
            maximum_items=12,
        )
        unknown = set(components) - set(knowledge_components)
        if unknown:
            raise TeacherAgentError(
                "goal.knowledge_spec.canonical_claims"
                f"[{index}] references unknown knowledge components: {sorted(unknown)}"
            )
        canonical_claims.append(
            {
                "claim_id": _bounded_identifier(
                    item.get("claim_id"),
                    field=f"goal.knowledge_spec.canonical_claims[{index}].claim_id",
                    fallback=f"claim_{index + 1:02d}",
                ),
                "statement": _nonempty_string(
                    item.get("statement", item.get("claim")),
                    field=f"goal.knowledge_spec.canonical_claims[{index}].statement",
                )[:1200],
                "knowledge_components": components,
                "required": _strict_boolean(
                    item.get("required"),
                    field=f"goal.knowledge_spec.canonical_claims[{index}].required",
                    default=True,
                ),
                "source_ids": _optional_string_list(
                    item.get("source_ids", []),
                    field=f"goal.knowledge_spec.canonical_claims[{index}].source_ids",
                    maximum_items=8,
                ),
            }
        )
    ensure_unique(canonical_claims, "claim_id", "canonical_claims")

    rubric_criteria: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items("rubric_criteria")):
        if isinstance(item, str):
            item = {"description": item}
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"goal.knowledge_spec.rubric_criteria[{index}] must be an object"
            )
        component = str(item.get("knowledge_component", "")).strip()
        if component and component not in knowledge_components:
            raise TeacherAgentError(
                "goal.knowledge_spec.rubric_criteria"
                f"[{index}] references unknown knowledge_component: {component}"
            )
        rubric_criteria.append(
            {
                "criterion_id": _bounded_identifier(
                    item.get("criterion_id"),
                    field=f"goal.knowledge_spec.rubric_criteria[{index}].criterion_id",
                    fallback=f"criterion_{index + 1:02d}",
                ),
                "description": _nonempty_string(
                    item.get("description", item.get("criterion")),
                    field=f"goal.knowledge_spec.rubric_criteria[{index}].description",
                )[:800],
                "knowledge_component": component or None,
                "required": _strict_boolean(
                    item.get("required"),
                    field=f"goal.knowledge_spec.rubric_criteria[{index}].required",
                    default=True,
                ),
                "acceptable_evidence": _optional_string_list(
                    item.get("acceptable_evidence", []),
                    field=(
                        "goal.knowledge_spec.rubric_criteria"
                        f"[{index}].acceptable_evidence"
                    ),
                    maximum_items=12,
                ),
            }
        )
    ensure_unique(rubric_criteria, "criterion_id", "rubric_criteria")

    accepted_alternatives: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items("accepted_alternatives")):
        if isinstance(item, str):
            item = {"description": item}
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"goal.knowledge_spec.accepted_alternatives[{index}] must be an object"
            )
        accepted_alternatives.append(
            {
                "alternative_id": _bounded_identifier(
                    item.get("alternative_id"),
                    field=(
                        "goal.knowledge_spec.accepted_alternatives"
                        f"[{index}].alternative_id"
                    ),
                    fallback=f"alternative_{index + 1:02d}",
                ),
                "description": _nonempty_string(
                    item.get("description", item.get("alternative")),
                    field=(
                        "goal.knowledge_spec.accepted_alternatives"
                        f"[{index}].description"
                    ),
                )[:800],
                "equivalent_claim_ids": _optional_string_list(
                    item.get("equivalent_claim_ids", []),
                    field=(
                        "goal.knowledge_spec.accepted_alternatives"
                        f"[{index}].equivalent_claim_ids"
                    ),
                    maximum_items=12,
                ),
                "conditions": _optional_string_list(
                    item.get("conditions", []),
                    field=(
                        f"goal.knowledge_spec.accepted_alternatives[{index}].conditions"
                    ),
                    maximum_items=8,
                ),
            }
        )
    ensure_unique(accepted_alternatives, "alternative_id", "accepted_alternatives")

    reference_steps: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items("reference_steps")):
        if isinstance(item, str):
            item = {"description": item}
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"goal.knowledge_spec.reference_steps[{index}] must be an object"
            )
        components = _optional_string_list(
            item.get("knowledge_components", []),
            field=(
                f"goal.knowledge_spec.reference_steps[{index}].knowledge_components"
            ),
            maximum_items=12,
        )
        unknown = set(components) - set(knowledge_components)
        if unknown:
            raise TeacherAgentError(
                "goal.knowledge_spec.reference_steps"
                f"[{index}] references unknown knowledge components: {sorted(unknown)}"
            )
        reference_steps.append(
            {
                "step_id": _bounded_identifier(
                    item.get("step_id"),
                    field=f"goal.knowledge_spec.reference_steps[{index}].step_id",
                    fallback=f"reference_step_{index + 1:02d}",
                ),
                "description": _nonempty_string(
                    item.get("description", item.get("step")),
                    field=f"goal.knowledge_spec.reference_steps[{index}].description",
                )[:800],
                "knowledge_components": components,
                "depends_on": _optional_string_list(
                    item.get("depends_on", []),
                    field=f"goal.knowledge_spec.reference_steps[{index}].depends_on",
                    maximum_items=12,
                ),
            }
        )
    ensure_unique(reference_steps, "step_id", "reference_steps")

    misconception_catalog: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items("misconception_catalog")):
        if isinstance(item, str):
            item = {"description": item}
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"goal.knowledge_spec.misconception_catalog[{index}] must be an object"
            )
        misconception_catalog.append(
            {
                "tag": _bounded_identifier(
                    item.get("tag"),
                    field=f"goal.knowledge_spec.misconception_catalog[{index}].tag",
                    fallback=f"misconception_{index + 1:02d}",
                ),
                "description": _nonempty_string(
                    item.get("description"),
                    field=(
                        "goal.knowledge_spec.misconception_catalog"
                        f"[{index}].description"
                    ),
                )[:800],
                "contradicts_claim_ids": _optional_string_list(
                    item.get("contradicts_claim_ids", []),
                    field=(
                        "goal.knowledge_spec.misconception_catalog"
                        f"[{index}].contradicts_claim_ids"
                    ),
                    maximum_items=12,
                ),
                "aliases": _optional_string_list(
                    item.get("aliases", []),
                    field=(
                        f"goal.knowledge_spec.misconception_catalog[{index}].aliases"
                    ),
                    maximum_items=8,
                ),
                "corrective_principle": str(
                    item.get("corrective_principle", "")
                ).strip()[:800],
            }
        )
    ensure_unique(misconception_catalog, "tag", "misconception_catalog")

    sources: list[dict[str, Any]] = []
    for index, item in enumerate(raw_items("sources")):
        if isinstance(item, str):
            item = {"title": item, "citation": item}
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"goal.knowledge_spec.sources[{index}] must be an object"
            )
        sources.append(
            {
                "source_id": _bounded_identifier(
                    item.get("source_id"),
                    field=f"goal.knowledge_spec.sources[{index}].source_id",
                    fallback=f"source_{index + 1:02d}",
                ),
                "title": _nonempty_string(
                    item.get("title", item.get("citation")),
                    field=f"goal.knowledge_spec.sources[{index}].title",
                )[:500],
                "citation": _nonempty_string(
                    item.get("citation", item.get("title")),
                    field=f"goal.knowledge_spec.sources[{index}].citation",
                )[:1000],
                "kind": str(item.get("kind", "teacher_material")).strip()[:80]
                or "teacher_material",
            }
        )
    ensure_unique(sources, "source_id", "sources")

    claim_ids = {item["claim_id"] for item in canonical_claims}
    source_ids = {item["source_id"] for item in sources}
    step_ids = {item["step_id"] for item in reference_steps}
    for row in canonical_claims:
        unknown = set(row["source_ids"]) - source_ids
        if unknown:
            raise TeacherAgentError(
                f"canonical claim {row['claim_id']} references unknown sources: "
                f"{sorted(unknown)}"
            )
    for row in accepted_alternatives:
        unknown = set(row["equivalent_claim_ids"]) - claim_ids
        if unknown:
            raise TeacherAgentError(
                f"accepted alternative {row['alternative_id']} references unknown "
                f"claims: {sorted(unknown)}"
            )
    for row in misconception_catalog:
        unknown = set(row["contradicts_claim_ids"]) - claim_ids
        if unknown:
            raise TeacherAgentError(
                f"misconception {row['tag']} references unknown claims: {sorted(unknown)}"
            )
    misconception_aliases: dict[str, str] = {}
    for row in misconception_catalog:
        for alias in [row["tag"], *row.get("aliases", [])]:
            key = _canonical_short_identifier(alias)
            owner = misconception_aliases.get(key)
            if owner is not None and owner != row["tag"]:
                raise TeacherAgentError(
                    "misconception catalog tags/aliases must be unambiguous"
                )
            misconception_aliases[key] = row["tag"]
    for row in reference_steps:
        unknown = set(row["depends_on"]) - step_ids
        if unknown:
            raise TeacherAgentError(
                f"reference step {row['step_id']} references unknown steps: "
                f"{sorted(unknown)}"
            )

    provided = any(
        (
            canonical_claims,
            rubric_criteria,
            accepted_alternatives,
            reference_steps,
            misconception_catalog,
            sources,
        )
    )
    raw_authority = value.get("authority")
    generated_unvalidated = raw_authority is not None
    if generated_unvalidated:
        if not isinstance(raw_authority, Mapping) or set(raw_authority) != {
            "status",
            "authoring_origin",
            "validated_by",
            "validation_receipts",
            "authoritative_for_runtime_grading",
        }:
            raise TeacherAgentError(
                "goal.knowledge_spec.authority must be a strict provenance receipt"
            )
        if (
            raw_authority.get("status") != "unvalidated_model_generated"
            or raw_authority.get("authoring_origin") != "teaching_syllabus_generator"
            or raw_authority.get("validated_by") != []
            or raw_authority.get("validation_receipts") != []
            or raw_authority.get("authoritative_for_runtime_grading") is not False
        ):
            raise TeacherAgentError(
                "model-generated knowledge specifications must remain unvalidated "
                "and non-authoritative"
            )
    authoritative = bool(provided and not generated_unvalidated)
    return {
        "schema": KNOWLEDGE_SPEC_SCHEMA,
        "status": (
            "generated_unvalidated"
            if generated_unvalidated
            else "teacher_provided"
            if provided
            else "not_provided"
        ),
        "canonical_claims": canonical_claims,
        "rubric_criteria": rubric_criteria,
        "accepted_alternatives": accepted_alternatives,
        "reference_steps": reference_steps,
        "misconception_catalog": misconception_catalog,
        "sources": sources,
        "authority": (
            {
                "status": "unvalidated_model_generated",
                "authoring_origin": "teaching_syllabus_generator",
                "validated_by": [],
                "validation_receipts": [],
                "authoritative_for_runtime_grading": False,
            }
            if generated_unvalidated
            else {
                "status": "teacher_asserted" if provided else "not_provided",
                "authoring_origin": "teacher_goal_input" if provided else "none",
                "validated_by": ["teacher_input_boundary"] if provided else [],
                "validation_receipts": [],
                "authoritative_for_runtime_grading": authoritative,
            }
        ),
        "claim_boundary": {
            "authoritative_for_runtime_grading": authoritative,
            "teacher_authored_or_imported": authoritative,
            "independently_verified_by_system": False,
            "model_memory_is_authoritative_when_absent": False,
        },
    }


def validate_skill_library(library: Mapping[str, Any]) -> None:
    if not isinstance(library, Mapping) or library.get("schema") not in LIBRARY_SCHEMAS:
        raise TeacherAgentError(
            f"skill library schema must be one of {sorted(LIBRARY_SCHEMAS)}"
        )
    skills = library.get("skills")
    if not isinstance(skills, list) or len(skills) < 5:
        raise TeacherAgentError("skill library must contain at least five Skills")
    identifiers: set[str] = set()
    primary_count = 0
    is_v2 = library.get("schema") == LIBRARY_SCHEMA_V2
    if is_v2:
        derivation = library.get("derivation")
        if (
            not isinstance(derivation, Mapping)
            or derivation.get("general_skill_id")
            != "evidence_grounded_multimodal_teaching_neural_v1"
        ):
            raise TeacherAgentError("v2 library must bind the neural-v1 general Skill")
        boundaries = library.get("claim_boundary")
        if (
            not isinstance(boundaries, Mapping)
            or boundaries.get("neural_v1_materialization_gate_passed") is not False
        ):
            raise TeacherAgentError(
                "v2 library must preserve the failed neural-v1 materialization gate"
            )
    for index, raw in enumerate(skills):
        if not isinstance(raw, Mapping):
            raise TeacherAgentError(f"skills[{index}] must be an object")
        skill_id = _nonempty_string(
            raw.get("skill_id"), field=f"skills[{index}].skill_id"
        )
        if skill_id in identifiers:
            raise TeacherAgentError(f"duplicate skill_id: {skill_id}")
        identifiers.add(skill_id)
        role = _nonempty_string(raw.get("role"), field=f"skills[{index}].role")
        if role in PRIMARY_ROLES:
            primary_count += 1
        elif role != "support":
            raise TeacherAgentError(f"unsupported Skill role: {role}")
        _nonempty_string(raw.get("name"), field=f"skills[{index}].name")
        _nonempty_string(
            raw.get("selection_rationale"), field=f"skills[{index}].selection_rationale"
        )
        _nonempty_string(
            raw.get("message_template"), field=f"skills[{index}].message_template"
        )
        _nonempty_string(
            raw.get("expected_signal"), field=f"skills[{index}].expected_signal"
        )
        focus = _nonempty_string(
            raw.get("focus_dimension"), field=f"skills[{index}].focus_dimension"
        )
        if focus not in MASTERY_DIMENSIONS:
            raise TeacherAgentError(f"unsupported focus_dimension: {focus}")
        signals = _string_list(
            raw.get("applicable_signals", []),
            field=f"skills[{index}].applicable_signals",
            allow_empty=role == "support",
        )
        if not set(signals) <= SIGNALS | {"not_observed"}:
            raise TeacherAgentError(f"skills[{index}] has unsupported signal")
        _string_list(
            raw.get("supporting_skill_ids", []),
            field=f"skills[{index}].supporting_skill_ids",
            allow_empty=True,
        )
        source = raw.get("source")
        if not isinstance(source, Mapping):
            raise TeacherAgentError(f"skills[{index}].source must be an object")
        _nonempty_string(source.get("origin"), field=f"skills[{index}].source.origin")
        if is_v2:
            if source.get("general_skill_id") != (
                "evidence_grounded_multimodal_teaching_neural_v1"
            ):
                raise TeacherAgentError(
                    f"skills[{index}] must bind the neural-v1 general Skill"
                )
            contract = raw.get("execution_contract")
            if not isinstance(contract, Mapping):
                raise TeacherAgentError(
                    f"skills[{index}].execution_contract must be an object"
                )
            for contract_field in (
                "applicable_when",
                "failure_transition",
                "addition_reason",
            ):
                _nonempty_string(
                    contract.get(contract_field),
                    field=f"skills[{index}].execution_contract.{contract_field}",
                )
            for contract_field in (
                "contraindications",
                "preconditions",
                "postconditions",
            ):
                _string_list(
                    contract.get(contract_field, []),
                    field=f"skills[{index}].execution_contract.{contract_field}",
                    allow_empty=contract_field == "contraindications",
                )
            max_repeat = contract.get("max_repeat")
            if (
                isinstance(max_repeat, bool)
                or not isinstance(max_repeat, int)
                or not 1 <= max_repeat <= 50
            ):
                raise TeacherAgentError(
                    f"skills[{index}].execution_contract.max_repeat is invalid"
                )
    if primary_count < 5:
        raise TeacherAgentError(
            "skill library must contain at least five primary Skills"
        )
    for raw in skills:
        unknown = set(raw.get("supporting_skill_ids", [])) - identifiers
        if unknown:
            raise TeacherAgentError(
                f"{raw['skill_id']} references unknown supporting Skills: {sorted(unknown)}"
            )
    declared = library.get("content_sha256")
    if declared is not None:
        material = dict(library)
        material.pop("content_sha256", None)
        if declared != canonical_sha256(material):
            raise TeacherAgentError("skill library content_sha256 mismatch")


def _trusted_curriculum_authority(
    value: Mapping[str, Any] | None,
    *,
    goal: Mapping[str, Any],
    knowledge_components: list[str],
    knowledge_spec: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Normalize an out-of-band server-verified curriculum projection.

    This value is deliberately a separate function argument to the session
    constructor.  A browser field with the same JSON cannot grant authority;
    the dashboard supplies it only after revalidating the durable seal and the
    currently published syllabus revision.
    """

    if value is None:
        if "curriculum_authority" in goal:
            raise TeacherAgentError(
                "goal.curriculum_authority requires a trusted server verifier"
            )
        return None
    expected = {
        "schema",
        "authority",
        "authoritative_for_runtime_grading",
        "curriculum_id",
        "curriculum_content_sha256",
        "receipt_id",
        "receipt_sha256",
        "signing_key_id",
        "family_id",
        "published_revision_id",
        "published_syllabus_id",
        "published_syllabus_sha256",
        "authority_version",
        "lesson_id",
        "legacy_lesson_id",
        "objective_ids",
        "kc_ids",
        "factual_claim_ids",
        "rubric_ids",
        "item_blueprint_ids",
        "projection_sha256",
    }
    if not isinstance(value, Mapping) or set(value) != expected:
        raise TeacherAgentError("trusted curriculum authority shape is invalid")
    material = deepcopy(dict(value))
    declared = material.pop("projection_sha256", None)
    if (
        value.get("schema") != CURRICULUM_RUNTIME_AUTHORITY_SCHEMA
        or value.get("authority") is not True
        or value.get("authoritative_for_runtime_grading") is not True
        or not isinstance(declared, str)
        or not re.fullmatch(r"[0-9a-f]{64}", declared)
        or canonical_sha256(material) != declared
    ):
        raise TeacherAgentError("trusted curriculum authority integrity is invalid")
    patterns = {
        "curriculum_id": r"cur_[0-9a-f]{24}",
        "curriculum_content_sha256": r"[0-9a-f]{64}",
        "receipt_id": r"receipt_[0-9a-f]{20}",
        "receipt_sha256": r"[0-9a-f]{64}",
        "signing_key_id": r"[A-Za-z][A-Za-z0-9_.:-]{0,119}",
        "family_id": r"syf_[0-9a-f]{24}",
        "published_revision_id": r"syr_[0-9a-f]{24}",
        "published_syllabus_id": r"syl_[0-9a-f]{24}",
        "published_syllabus_sha256": r"[0-9a-f]{64}",
        "lesson_id": r"lsn_[0-9a-f]{20}",
        "legacy_lesson_id": r"lesson_[0-9]{2}_[0-9]{2}",
    }
    if any(
        not isinstance(value.get(field), str)
        or re.fullmatch(pattern, str(value[field])) is None
        for field, pattern in patterns.items()
    ):
        raise TeacherAgentError("trusted curriculum authority identity is invalid")
    if (
        isinstance(value.get("authority_version"), bool)
        or not isinstance(value.get("authority_version"), int)
        or int(value["authority_version"]) < 1
    ):
        raise TeacherAgentError("trusted curriculum authority version is invalid")
    id_patterns = {
        "objective_ids": r"obj_[0-9a-f]{20}",
        "kc_ids": r"kc_[a-z0-9][a-z0-9_-]{2,80}",
        "factual_claim_ids": r"claim_[0-9a-f]{20}",
        "rubric_ids": r"rubric_[0-9a-f]{20}",
        "item_blueprint_ids": r"item_[0-9a-f]{20}",
    }
    for field, pattern in id_patterns.items():
        rows = value.get(field)
        if (
            not isinstance(rows, list)
            or (not rows and field != "factual_claim_ids")
            or len(rows) != len(set(rows))
            or any(
                not isinstance(item, str) or re.fullmatch(pattern, item) is None
                for item in rows
            )
        ):
            raise TeacherAgentError(
                f"trusted curriculum authority {field} is invalid"
            )
    syllabus_ref = goal.get("syllabus_ref")
    if (
        not isinstance(syllabus_ref, Mapping)
        or syllabus_ref.get("syllabus_id") != value["published_syllabus_id"]
        or syllabus_ref.get("lesson_id") != value["legacy_lesson_id"]
    ):
        raise TeacherAgentError(
            "trusted curriculum authority does not bind this syllabus lesson"
        )
    expected_kcs = [stable_knowledge_component_id(item) for item in knowledge_components]
    if value["kc_ids"] != expected_kcs:
        raise TeacherAgentError(
            "trusted curriculum authority does not bind the normalized KCs"
        )
    criterion_ids = [
        str(row.get("criterion_id", ""))
        for row in knowledge_spec.get("rubric_criteria", [])
        if isinstance(row, Mapping)
    ]
    if criterion_ids != value["rubric_ids"]:
        raise TeacherAgentError(
            "trusted curriculum authority does not bind the normalized rubrics"
        )
    claim_ids = [
        str(row.get("claim_id", ""))
        for row in knowledge_spec.get("canonical_claims", [])
        if isinstance(row, Mapping)
    ]
    if claim_ids != value["factual_claim_ids"]:
        raise TeacherAgentError(
            "trusted curriculum authority does not bind the normalized claims"
        )
    return deepcopy(dict(value))


def _normalized_goal(
    goal: Mapping[str, Any],
    *,
    trusted_curriculum_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(goal, Mapping):
        raise TeacherAgentError("goal must be an object")
    concept = _nonempty_string(goal.get("concept"), field="goal.concept")
    objective = _nonempty_string(goal.get("objective"), field="goal.objective")
    thresholds_raw = goal.get("success_thresholds", {})
    if not isinstance(thresholds_raw, Mapping):
        raise TeacherAgentError("goal.success_thresholds must be an object")
    default_thresholds = {
        "prerequisite": 0.60,
        "conceptual": 0.65,
        "procedural": 0.60,
        "transfer": 0.55,
    }
    thresholds = {
        dimension: _finite_probability(
            thresholds_raw.get(dimension, default),
            field=f"goal.success_thresholds.{dimension}",
        )
        for dimension, default in default_thresholds.items()
    }
    max_rounds = goal.get("max_rounds", 12)
    if (
        isinstance(max_rounds, bool)
        or not isinstance(max_rounds, int)
        or not 3 <= max_rounds <= 50
    ):
        raise TeacherAgentError("goal.max_rounds must be an integer in [3, 50]")
    materials = goal.get("materials", {})
    if not isinstance(materials, Mapping):
        raise TeacherAgentError("goal.materials must be an object")
    safe_materials: dict[str, str] = {}
    for key, value in materials.items():
        safe_materials[_nonempty_string(key, field="goal.materials key")] = (
            _nonempty_string(value, field=f"goal.materials.{key}")
        )
    raw_components = goal.get("knowledge_components")
    if raw_components is None:
        # The teacher-provided concept is the minimum faithful retrieval anchor;
        # this is not a model-inferred curriculum decomposition.
        knowledge_components = [concept]
    else:
        knowledge_components = _string_list(
            raw_components,
            field="goal.knowledge_components",
        )
        if len(knowledge_components) > 12:
            raise TeacherAgentError(
                "goal.knowledge_components must contain at most 12 items"
            )
    knowledge_spec = _normalize_knowledge_spec(
        goal.get("knowledge_spec"),
        knowledge_components=knowledge_components,
    )
    raw_learning_intent = goal.get("learning_intent")
    if raw_learning_intent is None:
        # Direct library/benchmark callers keep the historical first-turn
        # diagnostic contract.  Product surfaces opt into the new lesson flow
        # explicitly so old evaluation fingerprints do not silently change.
        learning_intent = "legacy_diagnostic_first"
    else:
        learning_intent = _nonempty_string(
            raw_learning_intent, field="goal.learning_intent"
        )
        if learning_intent not in LEARNING_INTENTS - {"legacy_diagnostic_first"}:
            raise TeacherAgentError(
                "goal.learning_intent must be teach_first, diagnostic_first, or task_first"
            )
    result = {
        "concept": concept,
        "objective": objective,
        "knowledge_components": knowledge_components,
        "knowledge_spec": knowledge_spec,
        "success_thresholds": thresholds,
        "max_rounds": max_rounds,
        "materials": safe_materials,
    }
    if raw_learning_intent is not None:
        result["learning_intent"] = learning_intent
    raw_syllabus_ref = goal.get("syllabus_ref")
    if raw_syllabus_ref is not None:
        if not isinstance(raw_syllabus_ref, Mapping) or set(raw_syllabus_ref) != {
            "syllabus_id",
            "module_id",
            "lesson_id",
            "content_sha256",
        }:
            raise TeacherAgentError(
                "goal.syllabus_ref must be a strict syllabus lesson reference"
            )
        syllabus_ref = {
            key: _nonempty_string(
                raw_syllabus_ref.get(key), field=f"goal.syllabus_ref.{key}"
            )
            for key in (
                "syllabus_id",
                "module_id",
                "lesson_id",
                "content_sha256",
            )
        }
        if not re.fullmatch(r"syl_[0-9a-f]{24}", syllabus_ref["syllabus_id"]):
            raise TeacherAgentError("goal.syllabus_ref.syllabus_id is invalid")
        if not re.fullmatch(r"module_[0-9]{2}", syllabus_ref["module_id"]):
            raise TeacherAgentError("goal.syllabus_ref.module_id is invalid")
        if not re.fullmatch(r"lesson_[0-9]{2}_[0-9]{2}", syllabus_ref["lesson_id"]):
            raise TeacherAgentError("goal.syllabus_ref.lesson_id is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", syllabus_ref["content_sha256"]):
            raise TeacherAgentError("goal.syllabus_ref.content_sha256 is invalid")
        result["syllabus_ref"] = syllabus_ref
    trusted = _trusted_curriculum_authority(
        trusted_curriculum_authority,
        goal=goal,
        knowledge_components=knowledge_components,
        knowledge_spec=knowledge_spec,
    )
    if trusted is not None:
        knowledge_spec["status"] = "sealed_teacher_curriculum"
        knowledge_spec["authority"] = {
            "status": "sealed_teacher_curriculum",
            "authoring_origin": "authenticated_curriculum_review",
            "validated_by": [
                "deployment_trusted_ed25519_receipt",
                "published_syllabus_revision_binding",
            ],
            "validation_receipts": [deepcopy(trusted)],
            "authoritative_for_runtime_grading": True,
        }
        knowledge_spec["claim_boundary"] = {
            "authoritative_for_runtime_grading": True,
            "teacher_authored_or_imported": True,
            "independently_verified_by_system": True,
            "model_memory_is_authoritative_when_absent": False,
        }
        result["curriculum_authority"] = trusted
    return result


def _lesson_phase_index(phase: str) -> int:
    try:
        return LESSON_PHASES.index(phase) + 1
    except ValueError:
        return 1


def _initial_lesson_phase(intent: str) -> str:
    if intent == "teach_first":
        # Orientation is an internal planning concern, not a learner turn.
        # A teach-first lesson must open with useful teaching content instead
        # of asking the learner to acknowledge the route before anything has
        # been explained.
        return "explanation"
    if intent == "task_first":
        return "guided_practice"
    return "verification"


def _new_lesson_state(goal: Mapping[str, Any]) -> dict[str, Any]:
    intent = str(goal.get("learning_intent", "legacy_diagnostic_first"))
    if intent not in LEARNING_INTENTS:
        intent = "legacy_diagnostic_first"
    phase = _initial_lesson_phase(intent)
    components = [
        str(item) for item in goal.get("knowledge_components", []) if str(item).strip()
    ] or [str(goal.get("concept", "当前概念"))]
    return {
        "schema": LESSON_STATE_SCHEMA,
        "intent": intent,
        "intent_source": (
            "explicit_product_request"
            if intent != "legacy_diagnostic_first"
            else "legacy_compatibility_default"
        ),
        "lesson_phase": phase,
        "phase_iteration": 1,
        # ``summary`` remains a closing sub-step of the learner-facing transfer
        # phase so existing six-stage UI contracts stay stable.  These fields
        # are optional in persisted v1 sessions for backward compatibility;
        # new sessions always make the closing obligation explicit.
        "summary_required": False,
        "summary_completed": False,
        "summary_closure_round_limit": SUMMARY_CLOSURE_ROUND_LIMIT,
        "summary_closure_rounds_used": 0,
        "last_transition": {
            "from": None,
            "to": phase,
            "round": 0,
            "reason": "lesson_initialized",
            "source": "deterministic_lesson_policy",
        },
        "knowledge_exposure": {
            component: {
                "status": "unseen",
                "stage_index": 0,
                "source": "no_teacher_exposure_recorded",
                "last_round": None,
                "evidence_refs": [],
            }
            for component in dict.fromkeys(components)
        },
        "claim_boundary": {
            "exposure_is_mastery": False,
            "teacher_action_is_learner_evidence": False,
            "phase_progress_is_learning_effect": False,
            "phase_may_regress": True,
            "benchmark_gold_used": False,
        },
    }


def _ensure_lesson_state(session: dict[str, Any]) -> dict[str, Any]:
    state = session.get("lesson_state")
    if not isinstance(state, dict) or state.get("schema") != LESSON_STATE_SCHEMA:
        state = _new_lesson_state(session.get("goal", {}))
        session["lesson_state"] = state
    # Older v1 sessions predate the transfer-closing flags.  Missing means
    # ``false``; never infer completion from phase, mastery, or teacher output.
    state.setdefault("summary_required", False)
    state.setdefault("summary_completed", False)
    state.setdefault("summary_closure_round_limit", SUMMARY_CLOSURE_ROUND_LIMIT)
    state.setdefault("summary_closure_rounds_used", 0)
    return state


def lesson_progress_view(session: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the small, learner-facing projection of the lesson state."""

    state = session.get("lesson_state")
    if not isinstance(state, Mapping) or state.get("schema") != LESSON_STATE_SCHEMA:
        return None
    phase = str(state.get("lesson_phase", ""))
    if phase not in LESSON_PHASES:
        return None
    goal = session.get("goal", {})
    concept = str(goal.get("concept", "")).strip() if isinstance(goal, Mapping) else ""
    summary_pending = bool(
        phase == "transfer"
        and state.get("intent") == "teach_first"
        and state.get("summary_required") is True
        and state.get("summary_completed") is not True
    )
    return {
        "phase": phase,
        "phase_label": "总结" if summary_pending else LESSON_PHASE_LABELS[phase],
        "phase_index": _lesson_phase_index(phase),
        "phase_count": len(LESSON_PHASES),
        "phase_reason": (
            "迁移已经完成，正在用自己的话总结适用条件、关键关系和失效边界。"
            if summary_pending
            else LESSON_PHASE_REASONS[phase]
        ),
        "current_concept": concept[:120],
    }


def is_lesson_navigation_response(
    session: Mapping[str, Any], learner_response: str
) -> bool:
    state = session.get("lesson_state")
    if not isinstance(state, Mapping) or state.get("intent") not in {
        "teach_first",
        "task_first",
        "diagnostic_first",
    }:
        return False
    response = str(learner_response).strip()
    if _LESSON_NAVIGATION_RE.fullmatch(response):
        return True

    # An early teaching action may ask the learner to choose the *way teaching
    # should continue* (for example ``具体例子`` or ``关键词``).  That exact
    # option is navigation, not proof that the learner answered the academic
    # question.  Keep this exception action- and phase-scoped so aliases on a
    # real assessment question can never bypass grading.
    action = session.get("current_action", {})
    primary = action.get("primary_skill", {}) if isinstance(action, Mapping) else {}
    teacher_action = (
        action.get("teacher_action", {}) if isinstance(action, Mapping) else {}
    )
    if isinstance(primary, Mapping) and isinstance(teacher_action, Mapping):
        contract = teacher_action.get("question_contract", {})
        phase = str(state.get("lesson_phase", ""))
        policy = action.get("learning_evidence_policy", {})
        recovery_choice = (
            primary.get("skill_id") == "skill_engagement_recovery"
            and teacher_action.get("type") == "refocus_and_restore_engagement"
        )
        formative_choice = (
            phase in {"orientation", "explanation", "worked_example"}
            and isinstance(policy, Mapping)
            and policy.get("mastery_gain_allowed") is False
            and isinstance(contract, Mapping)
            and contract.get("answer_type") == "reflection"
        )
        aliases = (
            contract.get("accepted_aliases", [])
            if isinstance(contract, Mapping)
            else []
        )
        exact_alias_choice = isinstance(aliases, list) and any(
            response.casefold() == str(alias).strip().casefold()
            for alias in aliases
            if str(alias).strip()
        )
        entry_mode_choice = formative_choice and bool(
            re.fullmatch(
                r"(?:(?:从)?(?:具体)?例子(?:开始)?|"
                r"(?:从)?关键词(?:开始)?)[！!。.]?",
                response,
                re.IGNORECASE,
            )
        )
        if (recovery_choice or formative_choice) and (
            exact_alias_choice or entry_mode_choice
        ):
            return True
    return False


def is_lesson_clarification_response(
    session: Mapping[str, Any], learner_response: str
) -> bool:
    """Return whether the learner asks for missing explanatory prerequisites.

    This is intentionally narrower than a generic question detector.  It
    covers definitions, composition, and reasons needed to understand the
    current teaching move, while excluding requests for a final answer or a
    complete solution.  A clarification controls the next teaching action; it
    is never evidence that the learner has mastered or failed the concept.
    """

    return lesson_clarification_kind(session, learner_response) is not None


def lesson_clarification_kind(
    session: Mapping[str, Any], learner_response: str
) -> str | None:
    """Classify a bounded learner request for explanatory prerequisites."""

    state = session.get("lesson_state")
    if not isinstance(state, Mapping) or state.get("intent") not in {
        "teach_first",
        "task_first",
        "diagnostic_first",
    }:
        return None
    response = str(learner_response).strip()
    kind = classify_lesson_clarification_text(response)
    if kind is None:
        return None
    phase = str(state.get("lesson_phase", ""))
    discourse = classify_learner_discourse_text(response)
    if phase in {"guided_practice", "verification", "transfer"} and (
        kind in {"procedure", "rationale"}
        and (
            discourse.solution_risk == "current_step"
            or _LESSON_TASK_SOLUTION_SCOPE_RE.search(response)
        )
    ):
        # A request about the active exercise is a scaffolding request, not a
        # conceptual clarification.  Keeping it out of answer-first prevents
        # the protocol from becoming a route around the final-solution guard.
        return None
    return kind


def classify_lesson_clarification_text(learner_response: str) -> str | None:
    """Classify a bounded clarification without depending on session state."""

    response = str(learner_response).strip()
    if not response or len(response) > 240:
        return None
    if _LESSON_REPETITION_COMPLAINT_RE.search(response):
        return None
    if _LESSON_FINAL_ANSWER_REQUEST_RE.search(response):
        return None
    if _LESSON_CLARIFICATION_DECLARATION_RE.search(
        response
    ) and not _LESSON_CONTRAST_QUESTION_RE.search(response):
        return None
    shared = classify_learner_discourse_text(response)
    if shared.solution_risk == "final_solution" or shared.explicit_confusion:
        return None
    if shared.clarification_kind is not None:
        return shared.clarification_kind
    return next(
        (
            kind
            for kind, pattern in _LESSON_CLARIFICATION_KIND_PATTERNS
            if pattern.search(response)
        ),
        None,
    )


def _lesson_library_for_goal(
    library: Mapping[str, Any], goal: Mapping[str, Any]
) -> dict[str, Any]:
    """Create a session-local Skill contract for an explicit lesson intent.

    The canonical library and legacy benchmark remain byte-for-byte unchanged.
    Only the persisted library inside an opted-in session gains the narrow
    signal applicability needed to teach before assessing.
    """

    result = deepcopy(dict(library))
    intent = str(goal.get("learning_intent", "legacy_diagnostic_first"))
    additions: dict[str, set[str]] = {}
    if intent == "teach_first":
        additions = {
            "skill_contextual_problem_setup": {"not_observed", "no_response"},
            "skill_concrete_example_bridge": {"not_observed", "correct"},
        }
    elif intent == "task_first":
        additions = {
            "skill_stepwise_scaffolding": {"not_observed", "no_response"},
            "skill_concrete_example_bridge": {"not_observed", "correct"},
        }
    if not additions:
        return result
    for skill in result.get("skills", []):
        if not isinstance(skill, dict) or skill.get("skill_id") not in additions:
            continue
        skill_id = str(skill["skill_id"])
        signals = list(skill.get("applicable_signals", []))
        skill["applicable_signals"] = list(
            dict.fromkeys([*signals, *sorted(additions[skill_id])])
        )
        if intent == "teach_first" and skill_id == "skill_contextual_problem_setup":
            skill["message_template"] = (
                "先不考前置知识。我们把 {concept} 放进一条清楚的学习路径："
                "先讲清它解决的问题和核心关系，再看完整示范、一起练一步，"
                "最后核验并迁移。你可以回复“继续”，或告诉我最想先弄清哪一部分。"
            )
            skill["expected_signal"] = (
                "学生选择一个关注点，或仅用“继续”确认进入概念讲解；"
                "该确认不作为掌握度证据。"
            )
        elif intent == "task_first" and skill_id == "skill_stepwise_scaffolding":
            skill["message_template"] = (
                "我会先给出完成 {practice} 所需的最小解释，再与你一起只处理第一步；"
                "请先指出题目给了什么、要求什么，不需要直接给最终答案。"
            )
    result.pop("content_sha256", None)
    return result


def _normalized_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(profile, Mapping):
        raise TeacherAgentError("student_profile must be an object")
    learner_level = _nonempty_string(
        profile.get("learner_level", "beginner"), field="student_profile.learner_level"
    )
    initial = profile.get("initial_mastery", {})
    if not isinstance(initial, Mapping):
        raise TeacherAgentError("student_profile.initial_mastery must be an object")
    mastery = {
        dimension: _finite_probability(
            initial.get(dimension, 0.0),
            field=f"student_profile.initial_mastery.{dimension}",
        )
        for dimension in MASTERY_DIMENSIONS
    }
    preferences = profile.get("preferences", [])
    if not isinstance(preferences, list):
        raise TeacherAgentError("student_profile.preferences must be a list")
    preferences = [
        _nonempty_string(item, field="student_profile.preferences[]")
        for item in preferences
    ]
    misconceptions_raw = profile.get("known_misconceptions", [])
    if not isinstance(misconceptions_raw, list):
        raise TeacherAgentError("student_profile.known_misconceptions must be a list")
    misconceptions: list[dict[str, Any]] = []
    for index, item in enumerate(misconceptions_raw):
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"student_profile.known_misconceptions[{index}] must be an object"
            )
        misconceptions.append(
            {
                "tag": _nonempty_string(
                    item.get("tag"),
                    field=f"student_profile.known_misconceptions[{index}].tag",
                ),
                "description": _nonempty_string(
                    item.get("description", item.get("tag")),
                    field=f"student_profile.known_misconceptions[{index}].description",
                ),
                "confidence": _finite_probability(
                    item.get("confidence", 0.8),
                    field=f"student_profile.known_misconceptions[{index}].confidence",
                ),
                "status": "active",
                "first_observed_round": 0,
                "last_observed_round": 0,
            }
        )
    history = profile.get("conversation_history", [])
    if not isinstance(history, list):
        raise TeacherAgentError("student_profile.conversation_history must be a list")
    safe_history: list[dict[str, Any]] = []
    for index, item in enumerate(history):
        if not isinstance(item, Mapping):
            raise TeacherAgentError(
                f"student_profile.conversation_history[{index}] must be an object"
            )
        signal = _nonempty_string(
            item.get("signal"),
            field=f"student_profile.conversation_history[{index}].signal",
        )
        if signal not in SIGNALS:
            raise TeacherAgentError(f"unsupported history signal: {signal}")
        focus = str(item.get("focus_dimension", "conceptual"))
        if focus not in MASTERY_DIMENSIONS:
            raise TeacherAgentError(f"unsupported history focus_dimension: {focus}")
        safe_history.append(
            {
                "response": str(item.get("response", ""))[:500],
                "signal": signal,
                "focus_dimension": focus,
            }
        )
    background_history_raw = profile.get("background_history", [])
    if not isinstance(background_history_raw, list):
        raise TeacherAgentError("student_profile.background_history must be a list")
    background_history = [
        str(item).strip()[:500]
        for item in background_history_raw[:12]
        if str(item).strip()
    ]
    return {
        "profile_ref": str(profile.get("profile_ref", "anonymous_student_profile"))[
            :80
        ],
        "learner_level": learner_level,
        "preferences": preferences,
        "accessibility_needs": [
            _nonempty_string(item, field="student_profile.accessibility_needs[]")
            for item in profile.get("accessibility_needs", [])
        ]
        if isinstance(profile.get("accessibility_needs", []), list)
        else [],
        "initial_mastery": mastery,
        "known_misconceptions": misconceptions,
        "conversation_history": safe_history,
        "background_history": background_history,
        "contains_direct_identity": bool(
            profile.get("contains_direct_identity", False)
        ),
        "identity_assessment": (
            "user_declared_present"
            if bool(profile.get("contains_direct_identity", False))
            else "not_asserted_present_not_independently_verified"
        ),
    }


def _skill_index(library: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["skill_id"]): dict(item) for item in library["skills"]}


def _active_misconceptions(state: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        dict(item)
        for item in state.get("misconceptions", [])
        if item.get("status") == "active" and float(item.get("confidence", 0)) >= 0.5
    ]


def _lowest_mastery_dimension(
    mastery: Mapping[str, Any], thresholds: Mapping[str, Any]
) -> str:
    return min(
        MASTERY_DIMENSIONS,
        key=lambda dimension: (
            float(mastery[dimension]) / max(float(thresholds[dimension]), 1e-9),
            MASTERY_DIMENSIONS.index(dimension),
        ),
    )


def _curriculum_focus(session: Mapping[str, Any]) -> str:
    """Choose the first unmet prerequisite in the learning dependency chain."""

    state = session["student_state"]
    mastery = state["knowledge_mastery"]
    thresholds = session["goal"]["success_thresholds"]
    if _active_misconceptions(state):
        return "conceptual"
    for dimension in MASTERY_DIMENSIONS:
        if float(mastery[dimension]) < float(thresholds[dimension]):
            return dimension
    return _lowest_mastery_dimension(mastery, thresholds)


def _active_knowledge_components(
    session: Mapping[str, Any],
    *,
    focus_dimension: str,
    action_text: str = "",
) -> list[str]:
    """Return the small ordered KC slice actually addressed by one action.

    Goal knowledge components are teacher-authored in dependency order.  A
    literal mention in the action wins; otherwise the current mastery phase is
    mapped to its corresponding slice.  This prevents every history turn from
    being tagged with the whole goal and makes same-KC retrieval meaningful.
    """

    goal = session.get("goal", {})
    raw_components = (
        goal.get("knowledge_components", []) if isinstance(goal, Mapping) else []
    )
    components = [
        str(item).strip()
        for item in raw_components
        if isinstance(item, str) and str(item).strip()
    ]
    if not components:
        concept = (
            str(goal.get("concept", "")).strip() if isinstance(goal, Mapping) else ""
        )
        return [concept] if concept else []

    folded_text = action_text.casefold()
    literal_matches = [
        component for component in components if component.casefold() in folded_text
    ]
    if literal_matches:
        return literal_matches[:3]

    try:
        phase_index = MASTERY_DIMENSIONS.index(focus_dimension)
    except ValueError:
        phase_index = 0
    start = phase_index * len(components) // len(MASTERY_DIMENSIONS)
    end = (phase_index + 1) * len(components) // len(MASTERY_DIMENSIONS)
    if end <= start:
        nearest = min(
            len(components) - 1,
            (phase_index * (len(components) - 1) + 1) // (len(MASTERY_DIMENSIONS) - 1),
        )
        return [components[nearest]]
    return components[start : min(end, start + 3)]


_LESSON_PHASE_ROLE_ORDER: dict[str, tuple[str, ...]] = {
    "orientation": ("context",),
    "explanation": ("example",),
    "worked_example": ("example", "concept_mapping"),
    "guided_practice": ("scaffolding", "practice"),
    "verification": ("assessment", "metacognition", "review"),
    "transfer": ("transfer", "summary"),
}


def lesson_required_primary_roles(
    session: Mapping[str, Any],
    *,
    lesson_phase: str | None = None,
    summary_required: bool | None = None,
    summary_completed: bool | None = None,
) -> tuple[str, ...]:
    """Return the server-owned primary roles for the current lesson step.

    Context construction, deterministic selection, and live projected routing
    must share this exact policy.  During teach-first transfer, a completed
    transfer attempt opens one mandatory learner-summary sub-step without
    adding a seventh learner-facing phase.
    """

    state = session.get("lesson_state")
    if not isinstance(state, Mapping) or state.get("intent") not in {
        "teach_first",
        "task_first",
    }:
        return ()
    phase = (
        str(state.get("lesson_phase", ""))
        if lesson_phase is None
        else str(lesson_phase)
    )
    roles = _LESSON_PHASE_ROLE_ORDER.get(phase, ())
    if phase != "transfer" or state.get("intent") != "teach_first":
        return roles
    required = (
        bool(state.get("summary_required", False))
        if summary_required is None
        else summary_required
    )
    completed = (
        bool(state.get("summary_completed", False))
        if summary_completed is None
        else summary_completed
    )
    return ("summary",) if required and not completed else ("transfer",)


def _lesson_role_order(session: Mapping[str, Any]) -> tuple[str, ...]:
    return lesson_required_primary_roles(session)


def _update_exposure(
    session: dict[str, Any],
    *,
    knowledge_components: list[str],
    status: str,
    source: str,
    evidence_ref: str,
    round_number: int | None = None,
) -> None:
    if status not in LESSON_EXPOSURE_STAGES:
        return
    state = _ensure_lesson_state(session)
    ledger = state.get("knowledge_exposure")
    if not isinstance(ledger, dict):
        return
    target_stage = LESSON_EXPOSURE_STAGES[status]
    for component in knowledge_components:
        key = str(component).strip()
        if not key:
            continue
        record = ledger.setdefault(
            key,
            {
                "status": "unseen",
                "stage_index": 0,
                "source": "no_teacher_exposure_recorded",
                "last_round": None,
                "evidence_refs": [],
            },
        )
        if not isinstance(record, dict):
            continue
        current_stage = int(record.get("stage_index", 0))
        refs = record.setdefault("evidence_refs", [])
        if isinstance(refs, list) and evidence_ref not in refs:
            refs.append(evidence_ref)
            del refs[:-8]
        if target_stage >= current_stage:
            record.update(
                {
                    "status": status,
                    "stage_index": target_stage,
                    "source": source,
                    "last_round": (
                        int(round_number)
                        if round_number is not None
                        else int(session.get("round", 0))
                    ),
                }
            )


def _bind_lesson_phase_to_action(
    session: dict[str, Any], action: dict[str, Any]
) -> dict[str, Any]:
    if "lesson_state" not in session and "learning_intent" not in session.get(
        "goal", {}
    ):
        return action
    state = _ensure_lesson_state(session)
    phase = str(state.get("lesson_phase", "orientation"))
    if phase not in LESSON_PHASES:
        phase = "orientation"
        state["lesson_phase"] = phase
    action["lesson_phase"] = lesson_progress_view(session)
    mastery_allowed = phase in {"guided_practice", "verification", "transfer"}
    action["learning_evidence_policy"] = {
        "scope": (
            "navigation_only"
            if phase == "orientation"
            else "formative_observation_no_mastery"
            if phase in {"explanation", "worked_example"}
            else "guided_attempt_evidence"
            if phase == "guided_practice"
            else "independent_learner_evidence"
        ),
        "mastery_gain_allowed": mastery_allowed,
        "teacher_action_is_learner_evidence": False,
    }
    components = [
        str(item)
        for item in action.get("knowledge_components", [])
        if str(item).strip()
    ]
    action_id = str(action.get("action_id", "current_action"))
    if phase == "explanation":
        _update_exposure(
            session,
            knowledge_components=components,
            status="explained",
            source="teacher_explanation_presented",
            evidence_ref=f"action:{action_id}:teacher_action",
            round_number=int(action.get("round", session.get("round", 0))),
        )
    elif phase == "worked_example":
        _update_exposure(
            session,
            knowledge_components=components,
            status="example_seen",
            source="teacher_worked_example_presented",
            evidence_ref=f"action:{action_id}:teacher_action",
            round_number=int(action.get("round", session.get("round", 0))),
        )
    return action


def lesson_response_evidence_eligible(
    session: Mapping[str, Any], learner_response: str
) -> bool:
    if is_lesson_navigation_response(
        session, learner_response
    ) or is_lesson_clarification_response(session, learner_response):
        return False
    action = session.get("current_action")
    if not isinstance(action, Mapping):
        return True
    policy = action.get("learning_evidence_policy")
    if not isinstance(policy, Mapping):
        return True
    return policy.get("mastery_gain_allowed") is True


def _advance_lesson_state_after_response(
    session: dict[str, Any],
    *,
    action_before: Mapping[str, Any],
    learner_response: str,
    signal: str,
    confidence: float,
    answer_alignment: str | None,
    needs_human_review: bool,
    evidence_eligible: bool,
) -> dict[str, Any]:
    state = _ensure_lesson_state(session)
    intent = str(state.get("intent", "legacy_diagnostic_first"))
    current = str(state.get("lesson_phase", _initial_lesson_phase(intent)))
    if current not in LESSON_PHASES:
        current = _initial_lesson_phase(intent)
    navigation = is_lesson_navigation_response(session, learner_response)
    clarification = is_lesson_clarification_response(session, learner_response)
    primary = action_before.get("primary_skill", {})
    primary_role = str(primary.get("role", "")) if isinstance(primary, Mapping) else ""
    provenance = action_before.get("action_provenance", {})
    execution_deferred = (
        isinstance(provenance, Mapping)
        and provenance.get("primary_skill_execution_deferred") is True
    )
    strong_closing_evidence = bool(
        evidence_eligible
        and signal == "correct"
        and confidence >= 0.5
        and answer_alignment == "aligned"
        and not needs_human_review
        and not execution_deferred
    )
    next_phase = current
    transition_reason = "phase_held_for_more_evidence"
    if intent in {"teach_first", "task_first"}:
        current_index = LESSON_PHASES.index(current)
        summary_pending = bool(
            intent == "teach_first"
            and current == "transfer"
            and state.get("summary_required", False)
            and not state.get("summary_completed", False)
        )
        if summary_pending:
            # Once a real transfer attempt opens the closing obligation, the
            # six-stage lesson stays in transfer until a real summary action
            # receives strong learner evidence.  Navigation, confusion, and a
            # deferred executor may change the next teaching tactic but may
            # not silently erase or satisfy this obligation.
            next_phase = current
            if primary_role == "summary" and strong_closing_evidence:
                state["summary_completed"] = True
                transition_reason = "learner_summary_completed"
            else:
                transition_reason = "learner_summary_still_required"
            state["summary_closure_rounds_used"] = (
                int(state.get("summary_closure_rounds_used", 0)) + 1
            )
        elif clarification:
            next_phase = current
            transition_reason = "learner_requested_explanatory_prerequisite"
        elif navigation:
            if current in {"orientation", "explanation", "worked_example"}:
                next_phase = LESSON_PHASES[current_index + 1]
                transition_reason = "learner_requested_next_teaching_step"
            else:
                next_phase = current
                transition_reason = "learner_attempt_required_before_phase_advance"
        elif signal in {"confused", "no_response"}:
            recovery = {
                "orientation": "explanation",
                "explanation": "explanation",
                "worked_example": "explanation",
                "guided_practice": "worked_example",
                "verification": "worked_example",
                "transfer": "guided_practice",
            }
            next_phase = recovery[current]
            transition_reason = "confusion_triggered_teaching_recovery"
        elif current == "orientation":
            next_phase = "explanation"
            transition_reason = "orientation_completed"
        else:
            aligned = answer_alignment in {None, "aligned", "partially_aligned"}
            strong = (
                signal == "correct"
                and confidence >= 0.5
                and aligned
                and not needs_human_review
            )
            partial_progress = (
                signal == "partial"
                and confidence >= 0.5
                and aligned
                and not needs_human_review
            )
            if current == "explanation" and (strong or partial_progress):
                next_phase = "worked_example"
                transition_reason = "concept_representation_ready_for_modeling"
            elif current == "worked_example" and strong:
                next_phase = "guided_practice"
                transition_reason = "worked_example_ready_for_guided_attempt"
            elif current == "guided_practice" and strong:
                next_phase = "verification"
                transition_reason = "guided_attempt_ready_for_independent_check"
            elif current == "verification" and strong:
                next_phase = "transfer"
                transition_reason = "independent_check_ready_for_transfer"
            elif (
                intent == "teach_first"
                and current == "transfer"
                and primary_role == "transfer"
                and strong_closing_evidence
                and state.get("summary_completed") is not True
            ):
                state["summary_required"] = True
                state["summary_completed"] = False
                state["summary_closure_round_limit"] = SUMMARY_CLOSURE_ROUND_LIMIT
                state["summary_closure_rounds_used"] = 0
                next_phase = "transfer"
                transition_reason = "transfer_evidence_requires_learner_summary"
    if next_phase != current:
        state["lesson_phase"] = next_phase
        state["phase_iteration"] = 1
    else:
        state["phase_iteration"] = int(state.get("phase_iteration", 1)) + 1
    transition = {
        "from": current,
        "to": next_phase,
        "round": int(session.get("round", 0)) + 1,
        "reason": transition_reason,
        "source": "deterministic_lesson_policy",
        "navigation_only": navigation,
        "learner_evidence_applied": evidence_eligible,
    }
    state["last_transition"] = transition

    action_phase = action_before.get("lesson_phase", {})
    prior_phase = (
        str(action_phase.get("phase", ""))
        if isinstance(action_phase, Mapping)
        else current
    )
    action_components = [
        str(item)
        for item in action_before.get("knowledge_components", [])
        if str(item).strip()
    ]
    aligned_for_exposure = answer_alignment in {None, "aligned", "partially_aligned"}
    if (
        evidence_eligible
        and aligned_for_exposure
        and not needs_human_review
        and signal in {"correct", "partial"}
        and confidence >= 0.5
        and prior_phase == "guided_practice"
    ):
        _update_exposure(
            session,
            knowledge_components=action_components,
            status="guided_practice_completed",
            source="learner_completed_guided_attempt",
            evidence_ref=f"history:r{int(session.get('round', 0)) + 1}:structured_signal",
            round_number=int(session.get("round", 0)) + 1,
        )
    if (
        evidence_eligible
        and answer_alignment in {None, "aligned"}
        and not needs_human_review
        and signal == "correct"
        and confidence >= 0.5
        and prior_phase == "verification"
    ):
        _update_exposure(
            session,
            knowledge_components=action_components,
            status="learner_verified",
            source="learner_independent_verification",
            evidence_ref=f"history:r{int(session.get('round', 0)) + 1}:structured_signal",
            round_number=int(session.get("round", 0)) + 1,
        )
    return transition


def _projected_lesson_state(
    session: Mapping[str, Any],
    *,
    learner_response: str,
    signal: str,
    confidence: float,
    answer_alignment: str | None,
    needs_human_review: bool,
) -> dict[str, Any] | None:
    """Project the server-owned lesson state without mutating the session."""

    material = deepcopy(dict(session))
    state = material.get("lesson_state")
    if not isinstance(state, Mapping):
        return None
    action = material.get("current_action", {})
    if not isinstance(action, Mapping):
        action = {}
    _advance_lesson_state_after_response(
        material,
        action_before=action,
        learner_response=learner_response,
        signal=signal,
        confidence=confidence,
        answer_alignment=answer_alignment,
        needs_human_review=needs_human_review,
        evidence_eligible=lesson_response_evidence_eligible(material, learner_response),
    )
    projected = material.get("lesson_state", {})
    return deepcopy(dict(projected)) if isinstance(projected, Mapping) else None


def projected_lesson_phase(
    session: Mapping[str, Any],
    *,
    learner_response: str,
    signal: str,
    confidence: float,
    answer_alignment: str | None,
    needs_human_review: bool,
) -> str | None:
    """Project the next classroom phase without mutating the live session."""

    projected = _projected_lesson_state(
        session,
        learner_response=learner_response,
        signal=signal,
        confidence=confidence,
        answer_alignment=answer_alignment,
        needs_human_review=needs_human_review,
    )
    phase = (
        str(projected.get("lesson_phase", "")) if isinstance(projected, Mapping) else ""
    )
    return phase if phase in LESSON_PHASES else None


def projected_lesson_closure(
    session: Mapping[str, Any],
    *,
    learner_response: str,
    signal: str,
    confidence: float,
    answer_alignment: str | None,
    needs_human_review: bool,
) -> dict[str, bool] | None:
    """Project transfer-summary flags for live planning before state commit."""

    projected = _projected_lesson_state(
        session,
        learner_response=learner_response,
        signal=signal,
        confidence=confidence,
        answer_alignment=answer_alignment,
        needs_human_review=needs_human_review,
    )
    if projected is None:
        return None
    return {
        "summary_required": bool(projected.get("summary_required", False)),
        "summary_completed": bool(projected.get("summary_completed", False)),
    }


def _selection_scores(
    session: Mapping[str, Any], *, policy: str, fixed_skill_id: str | None
) -> list[dict[str, Any]]:
    library = session["skill_library"]
    state = session["student_state"]
    round_index = int(session["round"])
    history_present = bool(session["student_profile"]["conversation_history"])
    last_signal = str(state["understanding_signal"]["label"])
    active_misconceptions = _active_misconceptions(state)
    focus = _curriculum_focus(session)
    used_roles = [
        str(event.get("action", {}).get("primary_skill", {}).get("role", ""))
        for event in session.get("history", [])
    ]
    no_progress = int(session.get("control", {}).get("consecutive_no_progress", 0))
    engagement = str(
        state.get("interaction_statistics", {}).get("engagement_level", "unknown")
    )
    previous_id = (
        session.get("current_action", {}).get("primary_skill", {}).get("skill_id")
    )
    goal_intent = str(
        session.get("goal", {}).get("learning_intent", "legacy_diagnostic_first")
    )
    lesson_roles = _lesson_role_order(session)
    rows: list[dict[str, Any]] = []
    for skill in library["skills"]:
        if skill["role"] not in PRIMARY_ROLES:
            continue
        score = float(skill.get("base_priority", 0))
        reasons: list[str] = []
        if policy == "fixed_single_skill_baseline":
            score = 1000.0 if skill["skill_id"] == fixed_skill_id else -1000.0
            reasons.append("固定单一 Skill 基线不根据学生状态切换")
        else:
            if (
                round_index == 0
                and not history_present
                and goal_intent in {"diagnostic_first", "legacy_diagnostic_first"}
                and skill["role"] == "diagnostic"
            ):
                score += 200
                reasons.append("首次教学先诊断前置知识")
            if lesson_roles:
                if skill["role"] in lesson_roles:
                    role_rank = lesson_roles.index(skill["role"])
                    score += 260 - role_rank * 45
                    reasons.append(f"当前课堂阶段优先执行 {skill['role']} 教学行为")
                else:
                    score -= 140
                if (
                    round_index == 0
                    and goal_intent == "teach_first"
                    and skill["role"] == "diagnostic"
                ):
                    score -= 260
                    reasons.append("先教后验：首轮禁止要求回忆未讲内容")
            if last_signal in skill["applicable_signals"]:
                score += 35
                reasons.append(f"适配当前理解信号 {last_signal}")
            if skill["focus_dimension"] == focus:
                score += 120
                reasons.append(f"按学习依赖顺序，当前先补 {focus}")
            else:
                score -= 25
            if active_misconceptions and skill["role"] == "correction":
                score += 180
                reasons.append("存在高置信度活跃误解，优先对比纠错")
            elif not active_misconceptions and skill["role"] == "correction":
                score -= 80
            if skill["role"] == "transfer":
                mastery = state["knowledge_mastery"]
                thresholds = session["goal"]["success_thresholds"]
                ready = (
                    mastery["conceptual"] >= thresholds["conceptual"] * 0.75
                    and mastery["procedural"] >= thresholds["procedural"] * 0.75
                )
                score += 70 if ready else -90
                reasons.append(
                    "概念与过程达到迁移准备线" if ready else "尚未达到迁移准备线"
                )
            if skill["role"] == "summary":
                score -= 40
            if skill["role"] == "diagnostic" and round_index > 0:
                if (
                    state["knowledge_mastery"]["prerequisite"]
                    < session["goal"]["success_thresholds"]["prerequisite"]
                ):
                    score += 35
                    reasons.append("前置知识仍低于成功门槛")
                else:
                    score -= 45
            if focus == "conceptual":
                if skill["role"] == "context" and "context" not in used_roles:
                    score += 48
                    reasons.append("已具备最低前置知识，先建立问题情境")
                if skill["role"] == "example" and last_signal in {
                    "not_observed",
                    "confused",
                    "no_response",
                }:
                    score += 55
                    reasons.append("先用直观例子建立概念表征")
                if skill["role"] == "concept_mapping" and last_signal in {
                    "correct",
                    "partial",
                }:
                    score += 50
                    reasons.append("已有初步表征，转入例子—定义映射")
                if skill["role"] == "assessment" and "concept_mapping" in used_roles:
                    score += 42
                    reasons.append("已有概念映射，进一步检查理由和边界")
            if focus == "procedural":
                if skill["role"] == "scaffolding" and "scaffolding" not in used_roles:
                    score += 65
                    reasons.append("先完成一次逐步支架示范")
                if skill["role"] == "practice" and "scaffolding" in used_roles:
                    score += 65
                    reasons.append("已有支架经历，转入独立练习反馈")
            if focus == "transfer" and skill["role"] == "transfer":
                score += 100
                reasons.append("前置、概念和过程已达标，开始变式迁移")
            if skill["role"] == "review" and last_signal in {
                "confused",
                "no_response",
            }:
                score += 45
                reasons.append("疑似回忆失败，优先主动检索而非完整重讲")
            if skill["role"] == "engagement":
                if no_progress >= 2 or engagement == "low":
                    score += 105
                    reasons.append("连续低进展或低参与，先恢复最小参与")
                else:
                    score -= 35
            if skill["role"] == "metacognition" and last_signal in {
                "correct",
                "partial",
            }:
                score += 28
                reasons.append("答案已有进展，要求学生解释依据与监控点")
            if previous_id == skill["skill_id"] and last_signal == "correct":
                score -= 25
                reasons.append("上一轮已正确，降低原 Skill 重复优先级")
        rows.append(
            {
                "skill_id": skill["skill_id"],
                "score": round(score, 3),
                "reasons": reasons or [skill["selection_rationale"]],
            }
        )
    return sorted(rows, key=lambda row: (-row["score"], row["skill_id"]))


def _render_action(
    session: dict[str, Any], *, policy: str, fixed_skill_id: str | None
) -> dict[str, Any]:
    scores = _selection_scores(session, policy=policy, fixed_skill_id=fixed_skill_id)
    selected_id = scores[0]["skill_id"]
    skills = _skill_index(session["skill_library"])
    selected = skills[selected_id]
    support_ids = list(selected.get("supporting_skill_ids", []))
    values = _TemplateValues(
        concept=session["goal"]["concept"],
        objective=session["goal"]["objective"],
        learner_level=session["student_profile"]["learner_level"],
        misconception=(
            _active_misconceptions(session["student_state"])[0]["description"]
            if _active_misconceptions(session["student_state"])
            else "当前尚未确认的误解"
        ),
        **session["goal"]["materials"],
    )
    message = str(selected["message_template"]).format_map(values)
    expected = str(selected["expected_signal"]).format_map(values)
    active_knowledge_components = _active_knowledge_components(
        session,
        focus_dimension=str(selected["focus_dimension"]),
        action_text=f"{message}\n{expected}",
    )
    previous_id = (
        session.get("current_action", {}).get("primary_skill", {}).get("skill_id")
    )
    switched = previous_id is not None and previous_id != selected_id
    reason = "；".join(scores[0]["reasons"] + [selected["selection_rationale"]])
    action = {
        "action_id": f"turn_{int(session['round']) + 1:03d}",
        "round": int(session["round"]) + 1,
        "primary_skill": {
            "skill_id": selected_id,
            "name": selected["name"],
            "role": selected["role"],
            "focus_dimension": selected["focus_dimension"],
            "knowledge_components": deepcopy(active_knowledge_components),
            "source": deepcopy(selected["source"]),
        },
        "supporting_skills": [
            {"skill_id": skill_id, "name": skills[skill_id]["name"]}
            for skill_id in support_ids
        ],
        "selection_reason": reason,
        "candidate_ranking": scores,
        "skill_switched": switched,
        "previous_primary_skill_id": previous_id,
        "teacher_action": {
            "type": selected["action_type"],
            "message": message,
            "expected_signal": expected,
            "direct_answer_prohibited": bool(
                selected.get("direct_answer_prohibited", True)
            ),
            "wait_for_student_before_next_action": True,
        },
        "knowledge_components": deepcopy(active_knowledge_components),
    }
    session["student_state"]["next_focus"] = {
        "dimension": selected["focus_dimension"],
        "reason": reason,
        "selected_skill_id": selected_id,
        "knowledge_components": deepcopy(active_knowledge_components),
    }
    return _bind_lesson_phase_to_action(session, action)


def _session_material(session: Mapping[str, Any]) -> dict[str, Any]:
    material = deepcopy(dict(session))
    material.pop("integrity", None)
    return material


def _refresh_integrity(session: dict[str, Any]) -> dict[str, Any]:
    session["integrity"] = {
        "algorithm": "sha256_canonical_json_without_integrity",
        "content_sha256": canonical_sha256(_session_material(session)),
    }
    return session


def _validate_adaptive_student_profile(
    profile: Mapping[str, Any], *, required: bool
) -> None:
    observations = profile.get("adaptive_observations")
    summary = profile.get("adaptive_summary")
    if observations is None and summary is None and not required:
        return
    if not isinstance(observations, list) or not isinstance(summary, Mapping):
        raise TeacherAgentError(
            "adaptive student profile requires observations and summary"
        )
    if len(observations) > ADAPTIVE_OBSERVATION_LIMIT:
        raise TeacherAgentError("adaptive_observations exceeds its retention limit")
    allowed_review_reasons = {
        "low_confidence",
        "model_requested_review",
        "deterministic_contract_normalization",
        "no_grounded_excerpt",
    }
    allowed_assessment_sources = {
        "deepseek_v4_flash",
        "deepseek_v4_flash_constrained_by_deterministic_contract",
        "active_question_contract_exact_match",
        "teacher_knowledge_spec_exact_match",
        "teacher_goal_knowledge_component_bounded_match",
    }
    for index, observation in enumerate(observations):
        if not isinstance(observation, Mapping) or set(observation) != {
            "round",
            "source",
            "status",
            "candidate",
            "evidence",
        }:
            raise TeacherAgentError(f"adaptive_observations[{index}] is invalid")
        round_number = observation["round"]
        if (
            isinstance(round_number, bool)
            or not isinstance(round_number, int)
            or round_number < 1
        ):
            raise TeacherAgentError(f"adaptive_observations[{index}].round is invalid")
        if (
            observation["source"] != ADAPTIVE_OBSERVATION_SOURCE
            or observation["status"] != ADAPTIVE_OBSERVATION_STATUS
        ):
            raise TeacherAgentError(
                f"adaptive_observations[{index}] provenance is invalid"
            )
        candidate = observation["candidate"]
        if not isinstance(candidate, Mapping) or set(candidate) != {
            "response_quality",
            "engagement_level",
            "misconception_tag",
            "next_focus",
        }:
            raise TeacherAgentError(
                f"adaptive_observations[{index}].candidate is invalid"
            )
        if candidate["response_quality"] not in {
            "complete",
            "partial",
            "minimal",
            "off_topic",
            "empty",
        } or candidate["engagement_level"] not in {
            "high",
            "medium",
            "low",
            "unknown",
        }:
            raise TeacherAgentError(
                f"adaptive_observations[{index}] candidate labels are invalid"
            )
        tag = candidate["misconception_tag"]
        if tag is not None and (not isinstance(tag, str) or len(tag) > 120):
            raise TeacherAgentError(
                f"adaptive_observations[{index}].misconception_tag is invalid"
            )
        if candidate["next_focus"] not in MASTERY_DIMENSIONS:
            raise TeacherAgentError(
                f"adaptive_observations[{index}].next_focus is invalid"
            )
        evidence = observation["evidence"]
        if not isinstance(evidence, Mapping) or set(evidence) != {
            "excerpt",
            "confidence",
            "assessment_source",
            "model_raw_signal",
            "final_signal",
            "normalization_reasons",
            "grounding",
            "needs_human_review",
            "review_reasons",
            "privacy_status",
        }:
            raise TeacherAgentError(
                f"adaptive_observations[{index}].evidence is invalid"
            )
        if not isinstance(evidence["excerpt"], str) or len(evidence["excerpt"]) > 240:
            raise TeacherAgentError(
                f"adaptive_observations[{index}].evidence.excerpt is invalid"
            )
        _finite_probability(
            evidence["confidence"],
            field=f"adaptive_observations[{index}].evidence.confidence",
        )
        normalization_reasons = evidence["normalization_reasons"]
        if (
            evidence["assessment_source"] not in allowed_assessment_sources
            # The provider may emit ``not_observed`` on a non-initial OCR-only
            # turn.  The live controller records that raw label for audit but
            # deterministically normalizes the final signal to a permitted
            # outcome before updating student state.
            or evidence["model_raw_signal"] not in SIGNALS | {"not_observed"}
            or evidence["final_signal"] not in SIGNALS
            or not isinstance(normalization_reasons, list)
            or len(normalization_reasons) > 12
            or len(normalization_reasons) != len(set(normalization_reasons))
            or any(
                not isinstance(reason, str) or not reason or len(reason) > 120
                for reason in normalization_reasons
            )
        ):
            raise TeacherAgentError(
                f"adaptive_observations[{index}].evidence assessment trace is invalid"
            )
        if (
            evidence["grounding"]
            not in {
                "verified_current_response_substring",
                "empty_response_no_excerpt",
                "no_grounded_excerpt",
            }
            or evidence["privacy_status"] != "redacted_before_candidate_storage"
        ):
            raise TeacherAgentError(
                f"adaptive_observations[{index}].evidence provenance is invalid"
            )
        reasons = evidence["review_reasons"]
        grounding = evidence["grounding"]
        grounding_review_consistent = isinstance(reasons, list) and (
            (grounding == "no_grounded_excerpt") == ("no_grounded_excerpt" in reasons)
        )
        if (
            not isinstance(evidence["needs_human_review"], bool)
            or not isinstance(reasons, list)
            or len(reasons) != len(set(reasons))
            or not set(reasons) <= allowed_review_reasons
            or bool(reasons) != evidence["needs_human_review"]
            or not grounding_review_consistent
        ):
            raise TeacherAgentError(
                f"adaptive_observations[{index}].evidence review state is invalid"
            )
    expected_summary_fields = {
        "schema",
        "status",
        "source",
        "observation_limit",
        "total_observation_count",
        "retained_observation_count",
        "latest_round",
        "latest_response_quality",
        "latest_engagement_level",
        "candidate_misconception_tags",
        "latest_next_focus",
        "needs_human_review",
        "teacher_provided_fields_overwritten",
    }
    if set(summary) != expected_summary_fields:
        raise TeacherAgentError("adaptive_summary fields are invalid")
    total = summary["total_observation_count"]
    if (
        summary["schema"] != ADAPTIVE_PROFILE_SCHEMA
        or summary["status"] != ADAPTIVE_OBSERVATION_STATUS
        or summary["source"] != "validated_deepseek_diagnoses_only"
        or summary["observation_limit"] != ADAPTIVE_OBSERVATION_LIMIT
        or summary["retained_observation_count"] != len(observations)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or total < len(observations)
        or not isinstance(summary["needs_human_review"], bool)
        or summary["teacher_provided_fields_overwritten"] is not False
    ):
        raise TeacherAgentError("adaptive_summary metadata is invalid")
    latest = observations[-1] if observations else None
    if latest is None:
        if any(
            summary[field] is not None
            for field in (
                "latest_round",
                "latest_response_quality",
                "latest_engagement_level",
                "latest_next_focus",
            )
        ) or (
            summary["candidate_misconception_tags"] != []
            or summary["total_observation_count"] != 0
            or summary["needs_human_review"]
        ):
            raise TeacherAgentError("empty adaptive_summary is inconsistent")
        return
    candidate = latest["candidate"]
    if (
        summary["latest_round"] != latest["round"]
        or summary["latest_response_quality"] != candidate["response_quality"]
        or summary["latest_engagement_level"] != candidate["engagement_level"]
        or summary["latest_next_focus"] != candidate["next_focus"]
        or summary["needs_human_review"]
        != any(item["evidence"]["needs_human_review"] for item in observations)
    ):
        raise TeacherAgentError("adaptive_summary latest candidate is inconsistent")
    expected_tags = list(
        dict.fromkeys(
            item["candidate"]["misconception_tag"]
            for item in observations
            if item["candidate"]["misconception_tag"]
        )
    )[-8:]
    if summary["candidate_misconception_tags"] != expected_tags:
        raise TeacherAgentError("adaptive_summary misconception tags are inconsistent")


def _validate_lesson_state(state: Mapping[str, Any]) -> None:
    if state.get("schema") != LESSON_STATE_SCHEMA:
        raise TeacherAgentError("lesson_state schema is invalid")
    if state.get("intent") not in LEARNING_INTENTS:
        raise TeacherAgentError("lesson_state intent is invalid")
    if state.get("lesson_phase") not in LESSON_PHASES:
        raise TeacherAgentError("lesson_state lesson_phase is invalid")
    iteration = state.get("phase_iteration")
    if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 1:
        raise TeacherAgentError("lesson_state phase_iteration is invalid")
    for field in ("summary_required", "summary_completed"):
        if field in state and not isinstance(state[field], bool):
            raise TeacherAgentError(f"lesson_state {field} is invalid")
    closure_limit = state.get(
        "summary_closure_round_limit", SUMMARY_CLOSURE_ROUND_LIMIT
    )
    closure_used = state.get("summary_closure_rounds_used", 0)
    if (
        isinstance(closure_limit, bool)
        or not isinstance(closure_limit, int)
        or closure_limit != SUMMARY_CLOSURE_ROUND_LIMIT
    ):
        raise TeacherAgentError("lesson_state summary closure round limit is invalid")
    if (
        isinstance(closure_used, bool)
        or not isinstance(closure_used, int)
        or not 0 <= closure_used <= closure_limit
    ):
        raise TeacherAgentError("lesson_state summary closure rounds used is invalid")
    summary_required = bool(state.get("summary_required", False))
    summary_completed = bool(state.get("summary_completed", False))
    if summary_completed and not summary_required:
        raise TeacherAgentError("lesson_state summary completion is inconsistent")
    if (summary_required or summary_completed) and (
        state.get("intent") != "teach_first" or state.get("lesson_phase") != "transfer"
    ):
        raise TeacherAgentError("lesson_state summary flags are outside transfer")
    if closure_used and not summary_required:
        raise TeacherAgentError("lesson_state summary closure usage is inconsistent")
    transition = state.get("last_transition")
    if not isinstance(transition, Mapping):
        raise TeacherAgentError("lesson_state last_transition is invalid")
    if transition.get("to") not in LESSON_PHASES or transition.get("from") not in {
        None,
        *LESSON_PHASES,
    }:
        raise TeacherAgentError("lesson_state transition phases are invalid")
    ledger = state.get("knowledge_exposure")
    if not isinstance(ledger, Mapping) or not 1 <= len(ledger) <= 12:
        raise TeacherAgentError("lesson_state knowledge_exposure is invalid")
    for component, record in ledger.items():
        if not str(component).strip() or not isinstance(record, Mapping):
            raise TeacherAgentError("lesson_state exposure record is invalid")
        status = str(record.get("status", ""))
        if (
            status not in LESSON_EXPOSURE_STAGES
            or record.get("stage_index") != LESSON_EXPOSURE_STAGES[status]
        ):
            raise TeacherAgentError("lesson_state exposure stage is inconsistent")
        refs = record.get("evidence_refs")
        if (
            not isinstance(refs, list)
            or len(refs) > 8
            or any(not isinstance(item, str) or not item for item in refs)
        ):
            raise TeacherAgentError("lesson_state exposure evidence_refs are invalid")
    boundary = state.get("claim_boundary")
    if (
        not isinstance(boundary, Mapping)
        or boundary.get("exposure_is_mastery") is not False
        or boundary.get("teacher_action_is_learner_evidence") is not False
        or boundary.get("phase_progress_is_learning_effect") is not False
        or boundary.get("phase_may_regress") is not True
        or boundary.get("benchmark_gold_used") is not False
    ):
        raise TeacherAgentError("lesson_state claim boundary is invalid")


def validate_session(session: Mapping[str, Any]) -> None:
    if not isinstance(session, Mapping) or session.get("schema") != SESSION_SCHEMA:
        raise TeacherAgentError(f"session schema must be {SESSION_SCHEMA}")
    if session.get("policy") not in POLICIES:
        raise TeacherAgentError("session policy is unsupported")
    validate_skill_library(session.get("skill_library", {}))
    if session.get("skill_library_fingerprint") != canonical_sha256(
        session.get("skill_library", {})
    ):
        raise TeacherAgentError("session skill_library_fingerprint mismatch")
    integrity = session.get("integrity")
    if not isinstance(integrity, Mapping) or integrity.get(
        "content_sha256"
    ) != canonical_sha256(_session_material(session)):
        raise TeacherAgentError("session integrity check failed")
    state = session.get("student_state")
    if not isinstance(state, Mapping):
        raise TeacherAgentError("session student_state is missing")
    if set(state.get("knowledge_mastery", {})) != set(MASTERY_DIMENSIONS):
        raise TeacherAgentError(
            "student_state knowledge_mastery dimensions are incomplete"
        )
    for dimension, value in state["knowledge_mastery"].items():
        _finite_probability(value, field=f"student_state.knowledge_mastery.{dimension}")
    raw_student_model = state.get("student_model")
    if isinstance(raw_student_model, Mapping):
        # Durable v1 live sessions remain readable long enough for the live
        # boundary to migrate them.  All newly persisted v2 models are fully
        # validated here rather than accepted as an opaque extension.
        if raw_student_model.get("schema") == STUDENT_MODEL_SCHEMA:
            try:
                validate_student_model(raw_student_model)
            except StudentModelError as exc:
                raise TeacherAgentError(
                    "student_state.student_model is invalid"
                ) from exc
        elif raw_student_model.get("schema") != LEGACY_STUDENT_MODEL_SCHEMA:
            raise TeacherAgentError("student_state.student_model schema is unsupported")
    for field in ("misconceptions", "understanding_signal", "next_focus"):
        if field not in state:
            raise TeacherAgentError(f"student_state.{field} is required")
    profile = session.get("student_profile")
    if not isinstance(profile, Mapping):
        raise TeacherAgentError("session student_profile is missing")
    _validate_adaptive_student_profile(
        profile,
        required=session.get("artifact_kind")
        == "real_time_deepseek_teaching_agent_session",
    )
    adaptive_rounds = [
        item["round"] for item in profile.get("adaptive_observations", [])
    ]
    if adaptive_rounds != sorted(set(adaptive_rounds)) or any(
        round_number > int(session.get("round", -1)) for round_number in adaptive_rounds
    ):
        raise TeacherAgentError("adaptive_observations rounds are inconsistent")
    status = session.get("status")
    if status not in {"active", *TERMINAL_STATUSES}:
        raise TeacherAgentError("session status is unsupported")
    if status == "active" and not isinstance(session.get("current_action"), Mapping):
        raise TeacherAgentError("active session requires current_action")
    lesson_state = session.get("lesson_state")
    if lesson_state is not None:
        if not isinstance(lesson_state, Mapping):
            raise TeacherAgentError("session lesson_state is invalid")
        _validate_lesson_state(lesson_state)
    if "teaching_resources" in session:
        from .teacher_agent_resources import (  # noqa: PLC0415
            TeachingResourceError,
            validate_teaching_resources,
        )

        try:
            validate_teaching_resources(session.get("teaching_resources"))
        except (TeachingResourceError, TypeError, ValueError) as exc:
            raise TeacherAgentError("session teaching_resources are invalid") from exc
    if session.get("artifact_kind") == "real_time_deepseek_teaching_agent_session":
        # Local import keeps the deterministic core reusable while making a
        # live session fail closed on malformed or over-budget model context.
        from .teacher_agent_context import (  # noqa: PLC0415
            LAYERED_CONTEXT_SCHEMA,
            validate_layered_context,
        )
        from .teacher_agent_memory import (  # noqa: PLC0415
            validate_teaching_memory_checkpoint,
        )

        try:
            validate_layered_context(session.get("context_memory", {}))
        except (TypeError, ValueError) as exc:
            raise TeacherAgentError("live session context_memory is invalid") from exc
        try:
            memory = session.get("teaching_memory", {})
            validate_teaching_memory_checkpoint(
                memory,
                goal=session["goal"],
                profile=profile,
                history=session.get("history", []),
                expected_round=session.get("round", -1),
            )
        except (TypeError, ValueError) as exc:
            raise TeacherAgentError("live session teaching_memory is invalid") from exc
        runtime = session.get("agent_runtime", {})
        trace = (
            runtime.get("last_context_trace") if isinstance(runtime, Mapping) else None
        )
        if (
            not isinstance(trace, Mapping)
            or trace.get("schema") != LAYERED_CONTEXT_SCHEMA
            or trace.get("content_sha256")
            != canonical_sha256(session["context_memory"])
            or trace.get("serialized_chars")
            != session["context_memory"]["budget"]["serialized_chars"]
            or trace.get("model_generated_memory_written") is not False
        ):
            raise TeacherAgentError("live session context trace is invalid")


def start_teacher_agent_session(
    goal: Mapping[str, Any],
    student_profile: Mapping[str, Any],
    skill_library: Mapping[str, Any],
    *,
    policy: str = "adaptive_skill_library",
    fixed_skill_id: str | None = None,
    trusted_curriculum_authority: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a session and emit only the first next action."""

    if policy not in POLICIES:
        raise TeacherAgentError(f"policy must be one of {sorted(POLICIES)}")
    goal_value = _normalized_goal(
        goal,
        trusted_curriculum_authority=trusted_curriculum_authority,
    )
    library = _lesson_library_for_goal(skill_library, goal_value)
    validate_skill_library(library)
    profile = _normalized_profile(student_profile)
    skills = _skill_index(library)
    if policy == "fixed_single_skill_baseline":
        fixed_skill_id = fixed_skill_id or "skill_stepwise_scaffolding"
        if (
            fixed_skill_id not in skills
            or skills[fixed_skill_id]["role"] not in PRIMARY_ROLES
        ):
            raise TeacherAgentError("fixed baseline Skill is missing or not primary")
    mastery = deepcopy(profile["initial_mastery"])
    signal = {
        "label": "not_observed",
        "confidence": 0.0,
        "response_excerpt": "",
        "source": "no_current_turn_observation",
    }
    for prior in profile["conversation_history"]:
        signal = {
            "label": prior["signal"],
            "confidence": 1.0,
            "response_excerpt": prior["response"][:160],
            "source": "provided_conversation_history",
        }
        if prior["signal"] == "correct":
            mastery[prior["focus_dimension"]] = min(
                1.0, mastery[prior["focus_dimension"]] + 0.08
            )
    session: dict[str, Any] = {
        "schema": SESSION_SCHEMA,
        "artifact_kind": "real_time_single_turn_teaching_agent_session",
        "status": "active",
        "policy": policy,
        "fixed_skill_id": fixed_skill_id,
        "goal": goal_value,
        "student_profile": profile,
        "skill_library": library,
        "skill_library_fingerprint": canonical_sha256(library),
        "round": 0,
        "student_state": {
            "knowledge_mastery": mastery,
            "misconceptions": deepcopy(profile["known_misconceptions"]),
            "understanding_signal": signal,
            "next_focus": {
                "dimension": "prerequisite",
                "reason": "等待首次 Skill 选择",
                "selected_skill_id": None,
            },
            "assessment_basis": "provided structured signals and deterministic updates",
        },
        "current_action": {},
        "history": [],
        "control": {
            "consecutive_no_progress": 0,
            "skill_switch_count": 0,
            "termination_reason": None,
        },
        "claim_boundary": {
            "free_text_answer_grading_established": False,
            "structured_signal_source": "teacher_or_external_judge",
            "student_identity_required": False,
            "simulated_evaluation_is_real_learning_effect": False,
            "real_learning_effectiveness_established": False,
            "deployment_quality_established": False,
        },
    }
    if "learning_intent" in goal_value:
        session["lesson_state"] = _new_lesson_state(goal_value)
    session["current_action"] = _render_action(
        session, policy=policy, fixed_skill_id=fixed_skill_id
    )
    return _refresh_integrity(session)


def _update_misconceptions(
    session: dict[str, Any],
    *,
    signal: str,
    response: str,
    misconception_tag: str | None,
    confidence: float,
    resolve_all_on_correction: bool,
) -> None:
    state = session["student_state"]
    action = session["current_action"]
    if signal == "misconception":
        tag = (misconception_tag or "unclassified_misconception").strip()
        if not tag:
            tag = "unclassified_misconception"
        existing = next(
            (item for item in state["misconceptions"] if item["tag"] == tag), None
        )
        if existing is None:
            state["misconceptions"].append(
                {
                    "tag": tag[:120],
                    "description": response[:240] or tag[:120],
                    "confidence": confidence,
                    "status": "active",
                    "first_observed_round": int(session["round"]) + 1,
                    "last_observed_round": int(session["round"]) + 1,
                }
            )
        else:
            existing["status"] = "active"
            existing["confidence"] = max(float(existing["confidence"]), confidence)
            existing["last_observed_round"] = int(session["round"]) + 1
    elif (
        resolve_all_on_correction
        and signal == "correct"
        and action["primary_skill"]["role"] == "correction"
    ):
        for item in state["misconceptions"]:
            if item["status"] == "active":
                item["status"] = "resolved"
                item["confidence"] = round(
                    max(0.0, float(item["confidence"]) - 0.55), 3
                )
                item["resolved_round"] = int(session["round"]) + 1


def _apply_student_signal(
    session: dict[str, Any],
    *,
    response: str,
    signal: str,
    misconception_tag: str | None,
    confidence: float,
    resolve_all_on_correction: bool,
    answer_alignment: str | None = None,
    needs_human_review: bool = False,
    count_as_no_progress: bool = True,
    assessment_eligible: bool = True,
) -> None:
    action = session["current_action"]
    focus = action["primary_skill"]["focus_dimension"]
    state = session["student_state"]
    if not assessment_eligible:
        state["understanding_signal"] = {
            "label": "not_observed",
            "confidence": 0.0,
            "response_excerpt": response[:240],
            "source": "lesson_navigation_or_formative_exposure_not_assessed",
        }
        return
    increments = {
        "correct": 0.28,
        "partial": 0.10,
        "misconception": 0.0,
        "confused": 0.0,
        "no_response": 0.0,
    }
    # A non-empty answer is not automatically evidence for the active
    # question.  Live mode supplies the final alignment after the server-side
    # contract gate; related/off-topic/ambiguous answers and any response that
    # still requires human review must not increase mastery.  The optional
    # arguments preserve the legacy deterministic API for callers that only
    # provide a structured signal.
    alignment_allows_gain = answer_alignment is None or answer_alignment in {
        "aligned",
        "partially_aligned",
    }
    if needs_human_review or not alignment_allows_gain:
        delta = 0.0
    else:
        delta = increments[signal] * confidence
    state["knowledge_mastery"][focus] = round(
        min(1.0, float(state["knowledge_mastery"][focus]) + delta), 3
    )
    state["understanding_signal"] = {
        "label": signal,
        "confidence": confidence,
        "response_excerpt": response[:240],
        "source": "teacher_or_external_judge",
    }
    _update_misconceptions(
        session,
        signal=signal,
        response=response,
        misconception_tag=misconception_tag,
        confidence=confidence,
        resolve_all_on_correction=resolve_all_on_correction,
    )
    # A model/transport failure is not evidence about the learner.  Live mode
    # passes ``count_as_no_progress=False`` for its deterministic safety
    # fallback so repeated provider failures cannot masquerade as three
    # consecutive unproductive learner turns.  Keep the legacy default
    # unchanged for callers that supply an actual structured observation.
    if not count_as_no_progress:
        return
    if signal in {"correct", "partial"} and confidence >= 0.5:
        session["control"]["consecutive_no_progress"] = 0
    else:
        session["control"]["consecutive_no_progress"] += 1


def _success_readiness(session: Mapping[str, Any]) -> dict[str, Any]:
    reasons: list[str] = []
    if int(session["round"]) < 3:
        reasons.append("minimum_observation_rounds_not_reached")
    lesson_state = session.get("lesson_state")
    if (
        isinstance(lesson_state, Mapping)
        and lesson_state.get("intent") in {"teach_first", "task_first"}
        and lesson_state.get("lesson_phase") != "transfer"
    ):
        reasons.append("required_lesson_phase_not_reached")
    if (
        isinstance(lesson_state, Mapping)
        and lesson_state.get("intent") == "teach_first"
        and not (
            lesson_state.get("summary_required") is True
            and lesson_state.get("summary_completed") is True
        )
    ):
        reasons.append("teach_first_transfer_summary_closure_incomplete")
    state = session["student_state"]
    if _active_misconceptions(state):
        reasons.append("active_high_confidence_misconception")
    understanding = state["understanding_signal"]
    if (
        understanding["label"] != "correct"
        or float(understanding.get("confidence", 0.0)) < 0.5
    ):
        reasons.append("latest_structured_signal_is_not_strong_correct_evidence")

    thresholds = session["goal"]["success_thresholds"]
    if session.get("artifact_kind") == "real_time_deepseek_teaching_agent_session":
        raw_model = state.get("student_model")
        if not isinstance(raw_model, Mapping):
            model_readiness: dict[str, Any] = {
                "schema": "teaching_skill_miner.student_model_mastery_readiness.v1",
                "eligible": False,
                "source": "missing_live_student_model",
                "runtime_authoritative": False,
                "component_coverage": [],
                "dimensions": {},
                "reasons": ["live_student_model_missing"],
                "claim_boundary": {
                    "unobserved_prior_counts_as_mastery_evidence": False,
                    "model_confidence_alone_is_mastery_evidence": False,
                    "external_calibration_established": False,
                },
            }
        else:
            try:
                model_readiness = student_model_mastery_readiness(raw_model, thresholds)
            except StudentModelError:
                model_readiness = {
                    "schema": "teaching_skill_miner.student_model_mastery_readiness.v1",
                    "eligible": False,
                    "source": "invalid_live_student_model",
                    "runtime_authoritative": False,
                    "component_coverage": [],
                    "dimensions": {},
                    "reasons": ["live_student_model_invalid"],
                    "claim_boundary": {
                        "unobserved_prior_counts_as_mastery_evidence": False,
                        "model_confidence_alone_is_mastery_evidence": False,
                        "external_calibration_established": False,
                    },
                }
        if model_readiness["eligible"] is not True:
            reasons.extend(str(item) for item in model_readiness["reasons"])
        source = "kc_model_active_authoritative_evidence"
    else:
        mastery = state["knowledge_mastery"]
        below = [
            key
            for key in MASTERY_DIMENSIONS
            if float(mastery[key]) < float(thresholds[key])
        ]
        if below:
            reasons.append("legacy_mastery_threshold_not_reached")
        model_readiness = None
        source = "legacy_structured_signal_mastery"

    return {
        "schema": "teaching_skill_miner.teacher_session_success_readiness.v1",
        "eligible": not reasons,
        "source": source,
        "reasons": list(dict.fromkeys(reasons)),
        "student_model_readiness": model_readiness,
    }


def _success_ready(session: Mapping[str, Any]) -> bool:
    return bool(_success_readiness(session)["eligible"])


def _apply_success_terminal(
    current: dict[str, Any], readiness: Mapping[str, Any]
) -> None:
    """Apply the single success terminal projection after a readiness receipt."""

    if readiness.get("eligible") is not True:
        raise TeacherAgentError("cannot apply success without an eligible receipt")
    current["control"]["mastery_readiness"] = deepcopy(dict(readiness))
    current["status"] = "succeeded"
    lesson_state = current.get("lesson_state", {})
    teach_first_closed = bool(
        isinstance(lesson_state, Mapping)
        and lesson_state.get("intent") == "teach_first"
        and lesson_state.get("summary_required") is True
        and lesson_state.get("summary_completed") is True
    )
    evidence_clause = (
        ", using active authoritative per-knowledge-component evidence"
        if readiness.get("source") == "kc_model_active_authoritative_evidence"
        else ""
    )
    current["control"]["termination_reason"] = (
        "all mastery thresholds reached"
        + evidence_clause
        + ", transfer evidence and learner summary completed, and no active "
        "high-confidence misconception remains"
        if teach_first_closed
        else (
            "all mastery thresholds reached"
            + evidence_clause
            + ", no active high-confidence misconception, and the latest "
            "structured signal is correct"
        )
    )
    current["current_action"] = {
        "action_id": f"terminal_{current['round']:03d}",
        "round": current["round"],
        "type": "terminate_success",
        "teacher_action": {
            "type": "summarize_and_stop",
            "message": (
                f"本轮关于 {current['goal']['concept']} 的迁移与学习者总结均已完成。"
                "建议安排一次延迟复习。"
                if teach_first_closed
                else (
                    f"本轮关于 {current['goal']['concept']} 的目标已达到。"
                    "请学习者用自己的话总结关键条件，并安排一次延迟复习。"
                )
            ),
            "wait_for_student_before_next_action": False,
        },
        "termination_reason": current["control"]["termination_reason"],
    }


def _summary_closure_budget_state(session: Mapping[str, Any]) -> tuple[bool, bool]:
    """Return (budget_available, budget_exhausted) for teach-first closure."""

    state = session.get("lesson_state")
    pending = bool(
        isinstance(state, Mapping)
        and state.get("intent") == "teach_first"
        and state.get("lesson_phase") == "transfer"
        and state.get("summary_required") is True
        and state.get("summary_completed") is not True
    )
    if not pending:
        return False, False
    limit = int(state.get("summary_closure_round_limit", SUMMARY_CLOSURE_ROUND_LIMIT))
    used = int(state.get("summary_closure_rounds_used", 0))
    return used < limit, used >= limit


def advance_teacher_agent_session(
    session: Mapping[str, Any],
    *,
    learner_response: str,
    signal: str,
    misconception_tag: str | None = None,
    signal_confidence: float = 1.0,
    resolve_all_on_correction: bool = True,
    resolved_misconception_tags: list[str] | None = None,
    answer_alignment: str | None = None,
    needs_human_review: bool = False,
    count_as_no_progress: bool = True,
) -> dict[str, Any]:
    """Consume one response and emit exactly one subsequent action or termination."""

    current = deepcopy(dict(session))
    validate_session(current)
    if "lesson_state" in current or "learning_intent" in current.get("goal", {}):
        _ensure_lesson_state(current)
    if current["status"] != "active":
        raise TeacherAgentError("cannot advance a terminal session")
    if signal not in SIGNALS:
        raise TeacherAgentError(f"signal must be one of {sorted(SIGNALS)}")
    if answer_alignment is not None and answer_alignment not in {
        "not_applicable",
        "aligned",
        "partially_aligned",
        "related_but_not_answer",
        "contradicted",
        "ambiguous",
        "no_response",
    }:
        raise TeacherAgentError("answer_alignment is invalid")
    if not isinstance(needs_human_review, bool):
        raise TeacherAgentError("needs_human_review must be a boolean")
    if not isinstance(count_as_no_progress, bool):
        raise TeacherAgentError("count_as_no_progress must be a boolean")
    confidence = _finite_probability(signal_confidence, field="signal_confidence")
    response = str(learner_response).strip()
    action_before = deepcopy(current["current_action"])
    state_before = deepcopy(current["student_state"])
    lesson_state_before = deepcopy(current.get("lesson_state", {}))
    evidence_eligible = lesson_response_evidence_eligible(current, response)
    _apply_student_signal(
        current,
        response=response,
        signal=signal,
        misconception_tag=misconception_tag,
        confidence=confidence,
        resolve_all_on_correction=resolve_all_on_correction,
        answer_alignment=answer_alignment,
        needs_human_review=needs_human_review,
        count_as_no_progress=count_as_no_progress,
        assessment_eligible=evidence_eligible,
    )
    lesson_transition = (
        _advance_lesson_state_after_response(
            current,
            action_before=action_before,
            learner_response=response,
            signal=signal,
            confidence=confidence,
            answer_alignment=answer_alignment,
            needs_human_review=needs_human_review,
            evidence_eligible=evidence_eligible,
        )
        if "lesson_state" in current
        else None
    )
    resolved_tags = resolved_misconception_tags or []
    if (
        not isinstance(resolved_tags, list)
        or len(resolved_tags) > 4
        or any(
            not isinstance(tag, str) or not tag or len(tag) > 120
            for tag in resolved_tags
        )
    ):
        raise TeacherAgentError("resolved_misconception_tags is invalid")
    resolved_tag_set = set(resolved_tags)
    if resolved_tag_set:
        primary = action_before.get("primary_skill", {})
        targets = action_before.get("target_misconception_tags", [])
        active_tags = {
            str(item.get("tag"))
            for item in current["student_state"]["misconceptions"]
            if isinstance(item, Mapping) and item.get("status") == "active"
        }
        target_binding = str(action_before.get("target_misconception_binding", "none"))
        correction_chain = (
            isinstance(primary, Mapping)
            and primary.get("role") in {"assessment", "metacognition", "review"}
            and target_binding == "prior_correction_chain"
        )
        if (
            signal != "correct"
            or not isinstance(primary, Mapping)
            or (primary.get("role") != "correction" and not correction_chain)
            or not isinstance(targets, list)
            or not resolved_tag_set <= set(str(item) for item in targets)
            or not resolved_tag_set <= active_tags
        ):
            raise TeacherAgentError(
                "resolved_misconception_tags requires a correct, targeted correction turn"
            )
    for item in current["student_state"]["misconceptions"]:
        if item.get("tag") in resolved_tag_set and item.get("status") == "active":
            item["status"] = "resolved"
            item["resolved_round"] = int(current["round"]) + 1
    current["round"] += 1
    event = {
        "round": current["round"],
        "action": action_before,
        "learner_response": response[:1000],
        "structured_signal": {
            "label": signal,
            "confidence": confidence,
            "source": "teacher_or_external_judge",
            "counted_as_no_progress": count_as_no_progress,
        },
        "student_state_before": state_before,
        "student_state_after_observation": deepcopy(current["student_state"]),
    }
    if lesson_transition is not None:
        event["structured_signal"].update(
            {
                "counted_as_no_progress": count_as_no_progress and evidence_eligible,
                "assessment_eligible": evidence_eligible,
                "applied_to_mastery": bool(
                    evidence_eligible
                    and signal in {"correct", "partial"}
                    and confidence > 0
                    and not needs_human_review
                    and answer_alignment in {None, "aligned", "partially_aligned"}
                ),
            }
        )
        event.update(
            {
                "lesson_state_before": lesson_state_before,
                "lesson_state_after": deepcopy(current.get("lesson_state", {})),
                "lesson_transition": deepcopy(lesson_transition),
            }
        )
    current["history"].append(event)

    readiness = _success_readiness(current)
    current["control"]["mastery_readiness"] = deepcopy(readiness)
    if readiness["eligible"]:
        _apply_success_terminal(current, readiness)
    else:
        summary_budget_available, summary_budget_exhausted = (
            _summary_closure_budget_state(current)
        )
    if current["status"] == "active" and (
        summary_budget_exhausted
        or (
            not summary_budget_available
            and (
                current["round"] >= current["goal"]["max_rounds"]
                or current["control"]["consecutive_no_progress"] >= 3
            )
        )
    ):
        current["status"] = "terminated_unable"
        reason = (
            "summary closure round limit reached"
            if summary_budget_exhausted
            else (
                "maximum teaching rounds reached"
                if current["round"] >= current["goal"]["max_rounds"]
                else "three consecutive rounds without observable progress"
            )
        )
        current["control"]["termination_reason"] = reason
        current["current_action"] = {
            "action_id": f"terminal_{current['round']:03d}",
            "round": current["round"],
            "type": "terminate_unable",
            "teacher_action": {
                "type": "stop_and_escalate",
                "message": (
                    "当前证据不足以安全继续。停止增加新内容，记录未掌握点，"
                    "建议教师人工诊断或更换前置材料。"
                ),
                "wait_for_student_before_next_action": False,
            },
            "termination_reason": reason,
        }
    elif current["status"] == "active":
        next_action = _render_action(
            current,
            policy=current["policy"],
            fixed_skill_id=current.get("fixed_skill_id"),
        )
        if next_action["skill_switched"]:
            current["control"]["skill_switch_count"] += 1
        current["current_action"] = next_action
    return _refresh_integrity(current)


def session_turn_summary(session: Mapping[str, Any]) -> dict[str, Any]:
    """Return a concise, UI-safe view of the next action and explicit state."""

    validate_session(session)
    result = {
        "schema": session["schema"],
        "status": session["status"],
        "rounds_completed": session["round"],
        "goal": deepcopy(session["goal"]),
        "student_state": deepcopy(session["student_state"]),
        "next_action": deepcopy(session["current_action"]),
        "skill_switch_count": session["control"]["skill_switch_count"],
        "termination_reason": session["control"]["termination_reason"],
        "claim_boundary": deepcopy(session["claim_boundary"]),
    }
    progress = lesson_progress_view(session)
    if progress is not None:
        result["lesson_progress"] = progress
        result["lesson_state"] = deepcopy(session["lesson_state"])
    return result


def _mastery_score(session: Mapping[str, Any]) -> float:
    mastery = session["student_state"]["knowledge_mastery"]
    return round(
        100
        * sum(float(mastery[key]) for key in ("conceptual", "procedural", "transfer"))
        / 3,
        3,
    )


def _behavior_checks(action: Mapping[str, Any], concept: str) -> dict[str, bool]:
    teacher = action.get("teacher_action", {})
    message = str(teacher.get("message", ""))
    return {
        "message_nonempty": bool(message.strip()),
        "goal_targeted": concept in message,
        "selection_reason_present": bool(
            str(action.get("selection_reason", "")).strip()
        ),
        "expected_signal_present": bool(
            str(teacher.get("expected_signal", "")).strip()
        ),
        "waits_for_student": teacher.get("wait_for_student_before_next_action") is True,
        "does_not_directly_dump_answer": teacher.get("direct_answer_prohibited")
        is True,
    }


def _run_evaluation_policy(
    library: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    policy: str,
    fixed_skill_id: str | None = None,
) -> dict[str, Any]:
    session = start_teacher_agent_session(
        case["goal"],
        case["student_profile"],
        library,
        policy=policy,
        fixed_skill_id=fixed_skill_id,
    )
    pretest = _mastery_score(session)
    decisions: list[bool] = []
    state_checks: list[bool] = []
    behavior_checks: list[bool] = []
    selected: list[str] = []
    timeline: list[dict[str, Any]] = []
    for index, feedback in enumerate(case["feedback_sequence"]):
        if session["status"] != "active":
            break
        action = session["current_action"]
        selected_id = action["primary_skill"]["skill_id"]
        selected.append(selected_id)
        expected = feedback.get("expected_skill_ids", [])
        if expected:
            decisions.append(selected_id in expected)
        checks = _behavior_checks(action, session["goal"]["concept"])
        behavior_checks.extend(checks.values())
        state_before = deepcopy(session["student_state"])
        session = advance_teacher_agent_session(
            session,
            learner_response=str(feedback.get("response", "")),
            signal=str(feedback["signal"]),
            misconception_tag=feedback.get("misconception_tag"),
            signal_confidence=float(feedback.get("confidence", 1.0)),
        )
        timeline.append(
            {
                "round": index + 1,
                "teacher_action": {
                    "primary_skill_id": selected_id,
                    "primary_skill_name": action["primary_skill"]["name"],
                    "selection_reason": action["selection_reason"],
                    "message": action["teacher_action"]["message"],
                    "expected_signal": action["teacher_action"]["expected_signal"],
                },
                "learner_feedback": {
                    "response": str(feedback.get("response", ""))[:1000],
                    "declared_signal": feedback["signal"],
                    "signal_confidence": float(feedback.get("confidence", 1.0)),
                },
                "student_state_before": state_before,
                "student_state_after": deepcopy(session["student_state"]),
                "status_after": session["status"],
            }
        )
        observed = session["student_state"]["understanding_signal"]
        state_checks.append(observed["label"] == feedback["signal"])
        expected_focus = feedback.get("expected_next_focus")
        if expected_focus and session["status"] == "active":
            state_checks.append(
                session["student_state"]["next_focus"]["dimension"] == expected_focus
            )
    posttest = _mastery_score(session)
    transfer_threshold = float(session["goal"]["success_thresholds"]["transfer"])
    return {
        "policy": policy,
        "status": session["status"],
        "termination_reason": session["control"]["termination_reason"],
        "selected_skill_ids": selected,
        "unique_skill_count": len(set(selected)),
        "skill_switch_count": session["control"]["skill_switch_count"],
        "timeline": timeline,
        "structured_state_check_rate": (
            sum(state_checks) / len(state_checks) if state_checks else 0.0
        ),
        "decision_match_rate": (sum(decisions) / len(decisions) if decisions else None),
        "behavior_quality_rate": (
            sum(behavior_checks) / len(behavior_checks) if behavior_checks else 0.0
        ),
        "simulated_pretest_score": pretest,
        "simulated_posttest_score": posttest,
        "simulated_gain": round(posttest - pretest, 3),
        "simulated_transfer_passed": (
            session["student_state"]["knowledge_mastery"]["transfer"]
            >= transfer_threshold
        ),
        "final_student_state": deepcopy(session["student_state"]),
        "session_content_sha256": session["integrity"]["content_sha256"],
    }


def evaluate_teacher_agent(
    skill_library: Mapping[str, Any], evaluation_cases: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate adaptive decisions against a fixed single-Skill baseline.

    The fixtures use supplied structured signals.  The reported pre/post and
    transfer values are deterministic simulation diagnostics, never real
    learner-effect evidence.
    """

    library = deepcopy(dict(skill_library))
    validate_skill_library(library)
    cases = (
        evaluation_cases.get("cases") if isinstance(evaluation_cases, Mapping) else None
    )
    if not isinstance(cases, list) or not cases:
        raise TeacherAgentError("evaluation cases must contain a non-empty cases list")
    results: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        if not isinstance(case, Mapping):
            raise TeacherAgentError(f"cases[{index}] must be an object")
        case_id = _nonempty_string(case.get("case_id"), field=f"cases[{index}].case_id")
        expected_terminal_status = _nonempty_string(
            case.get("expected_terminal_status"),
            field=f"cases[{index}].expected_terminal_status",
        )
        if expected_terminal_status not in TERMINAL_STATUSES:
            raise TeacherAgentError(
                f"cases[{index}].expected_terminal_status must be one of "
                f"{sorted(TERMINAL_STATUSES)}"
            )
        feedback = case.get("feedback_sequence")
        if not isinstance(feedback, list) or not feedback:
            raise TeacherAgentError(
                f"cases[{index}].feedback_sequence must be non-empty"
            )
        adaptive = _run_evaluation_policy(
            library, case, policy="adaptive_skill_library"
        )
        baseline = _run_evaluation_policy(
            library,
            case,
            policy="fixed_single_skill_baseline",
            fixed_skill_id="skill_stepwise_scaffolding",
        )
        results.append(
            {
                "case_id": case_id,
                "expected_terminal_status": expected_terminal_status,
                "adaptive_agent": adaptive,
                "fixed_single_skill_baseline": baseline,
                "terminal_decision_match": (
                    adaptive["status"] == expected_terminal_status
                ),
                "simulated_gain_delta": round(
                    adaptive["simulated_gain"] - baseline["simulated_gain"], 3
                ),
                "adaptive_used_multiple_skills": adaptive["unique_skill_count"] >= 2,
            }
        )
    adaptive_rows = [row["adaptive_agent"] for row in results]
    baseline_rows = [row["fixed_single_skill_baseline"] for row in results]

    def mean(rows: list[dict[str, Any]], key: str) -> float:
        return round(sum(float(row[key]) for row in rows) / len(rows), 6)

    decision_values = [
        float(row["decision_match_rate"])
        for row in adaptive_rows
        if row["decision_match_rate"] is not None
    ]
    aggregate = {
        "case_count": len(results),
        "expected_success_case_count": sum(
            row["expected_terminal_status"] == "succeeded" for row in results
        ),
        "expected_unable_case_count": sum(
            row["expected_terminal_status"] == "terminated_unable" for row in results
        ),
        "terminal_decision_match_rate": round(
            sum(row["terminal_decision_match"] for row in results) / len(results),
            6,
        ),
        "student_state_judgement_rate": mean(
            adaptive_rows, "structured_state_check_rate"
        ),
        "teaching_decision_match_rate": round(
            sum(decision_values) / len(decision_values), 6
        )
        if decision_values
        else 0.0,
        "teaching_behavior_quality_rate": mean(adaptive_rows, "behavior_quality_rate"),
        "adaptive_multi_skill_case_rate": round(
            sum(row["adaptive_used_multiple_skills"] for row in results) / len(results),
            6,
        ),
        "adaptive_simulated_mean_gain": mean(adaptive_rows, "simulated_gain"),
        "baseline_simulated_mean_gain": mean(baseline_rows, "simulated_gain"),
        "simulated_mean_gain_delta": round(
            mean(adaptive_rows, "simulated_gain")
            - mean(baseline_rows, "simulated_gain"),
            6,
        ),
        "adaptive_simulated_transfer_pass_rate": round(
            sum(row["simulated_transfer_passed"] for row in adaptive_rows)
            / len(adaptive_rows),
            6,
        ),
        "baseline_simulated_transfer_pass_rate": round(
            sum(row["simulated_transfer_passed"] for row in baseline_rows)
            / len(baseline_rows),
            6,
        ),
    }
    gates = {
        "student_state_judgement": aggregate["student_state_judgement_rate"] >= 0.90,
        "teaching_decision_quality": aggregate["teaching_decision_match_rate"] >= 0.75,
        "teaching_behavior_quality": aggregate["teaching_behavior_quality_rate"]
        >= 0.95,
        "termination_quality": aggregate["terminal_decision_match_rate"] >= 0.95,
        "multi_skill_selection_and_switching": aggregate[
            "adaptive_multi_skill_case_rate"
        ]
        >= 0.80,
        "adaptive_beats_fixed_skill_in_simulation": aggregate[
            "simulated_mean_gain_delta"
        ]
        > 0,
    }
    report = {
        "schema": EVALUATION_SCHEMA,
        "artifact_kind": "reproducible_structured_fixture_teacher_agent_evaluation",
        "skill_library_fingerprint": canonical_sha256(library),
        "evaluation_cases_fingerprint": canonical_sha256(dict(evaluation_cases)),
        "baseline": {
            "policy": "fixed_single_skill_baseline",
            "skill_id": "skill_stepwise_scaffolding",
        },
        "aggregate": aggregate,
        "gates": gates,
        "passed": all(gates.values()),
        "cases": results,
        "claim_boundary": {
            "fixtures_are_synthetic": True,
            "responses_are_structured_test_inputs": True,
            "free_text_answer_grading_established": False,
            "simulated_pre_post_is_real_learning_effect": False,
            "real_learner_effectiveness_established": False,
            "expert_teaching_quality_established": False,
            "deployment_quality_established": False,
        },
        "metric_semantics": {
            "student_state_judgement_rate": "agreement with declared structured fixture state, not free-text diagnostic accuracy",
            "teaching_decision_match_rate": "match to fixture-allowed Skill choices",
            "teaching_behavior_quality_rate": "deterministic safety and completeness checks, not expert rating",
            "terminal_decision_match_rate": "agreement with each fixture's declared success-or-escalate outcome",
            "simulated_gain": "change in deterministic mastery state, not observed learner gain",
        },
    }
    report["content_sha256"] = canonical_sha256(report)
    return report
