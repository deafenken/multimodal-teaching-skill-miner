"""Evidence-linked long-horizon memory for the live Teaching Agent.

The live model already receives recent verbatim turns.  This module preserves
the few cross-turn facts that are easy to lose after bounded context
compaction: explicit learner preferences, learner-authored questions, teacher
commitments, and named alternatives used by later references such as
``第二种呢``.  It never invents a narrative summary and every retained item is
linked to an observable utterance.
"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Any, Mapping, Sequence

from .teacher_agent import canonical_sha256


TEACHING_MEMORY_SCHEMA = "teaching_skill_miner.teaching_memory.v1"
TEACHING_MEMORY_PROJECTION_SCHEMA = (
    "teaching_skill_miner.teaching_memory_projection.v1"
)
MAX_PREFERENCES = 12
MAX_OPEN_QUESTIONS = 12
MAX_COMMITMENTS = 8
MAX_REFERENTS = 8
_DYNAMIC_PROFILE_FIELDS = frozenset(
    {
        "adaptive_observations",
        "adaptive_summary",
    }
)

_QUESTION_RE = re.compile(
    r"(?:[?？]\s*$|^(?:为什么|为何|怎么|怎样|如何|什么|哪(?:个|种)|能不能|可不可以|"
    r"是不是|是否|请问)|(?:吗|呢|么)[?？。！!\s]*$)",
    re.IGNORECASE,
)
_POSITIVE_ACK_RE = re.compile(
    r"(?:明白了|懂了|理解了|解决了|知道了|清楚了|原来如此|got\s+it|makes\s+sense)",
    re.IGNORECASE,
)
_REOPEN_RE = re.compile(
    r"(?:还(?:是)?不(?:懂|明白|理解)|没有回答|还没回答|没有解决|还没解决|仍然不清楚)",
    re.IGNORECASE,
)
_COMMITMENT_RE = re.compile(
    r"(?:接下来(?:我)?会|下一步(?:我)?会|稍后(?:我)?会|等你.+(?:后|之后).*(?:再|会)|"
    r"之后(?:我)?会)",
    re.IGNORECASE,
)
_REFERENT_MARKER_RE = re.compile(
    r"(?:第一种|第二种|第三种|方法一|方法二|方法三|方案\s*[ABC一二三]|"
    r"路径\s*[ABC一二三]|(?:^|[，,；;。\n])\s*[123][.、)])",
    re.IGNORECASE,
)
_PREFERENCE_CUE_RE = re.compile(
    r"(?:我(?:更|比较)?喜欢|我偏好|我希望|我想要|我想(?:先|用|看|听)|"
    r"请(?:先|不要|别|用|给)|能不能(?:先|不要|用)|可不可以(?:先|不要|用)|"
    r"不要|别)",
    re.IGNORECASE,
)
_SPACE_RE = re.compile(r"\s+")


class TeachingMemoryError(ValueError):
    """Raised when long-horizon memory violates its evidence contract."""


def _text(value: Any, maximum: int = 400) -> str:
    return _SPACE_RE.sub(" ", str(value or "").strip())[:maximum]


def _memory_id(kind: str, round_number: int, text: str) -> str:
    return f"{kind}_r{round_number:03d}_{canonical_sha256(text)[:12]}"


def _preference_kind(text: str) -> str:
    folded = text.casefold()
    if re.search(r"(?:不要|别).{0,8}(?:公式|推导)", text):
        return "avoid_formula_first"
    if "例" in text:
        return "prefer_examples"
    if any(term in text for term in ("分步", "一步一步", "一步步")):
        return "prefer_stepwise"
    if any(term in text for term in ("图示", "图像", "可视化", "画图")):
        return "prefer_visual_explanation"
    if any(term in text for term in ("简洁", "简短", "短一点")):
        return "prefer_concise"
    if any(term in text for term in ("详细", "讲细", "展开讲")):
        return "prefer_detailed"
    if "example" in folded:
        return "prefer_examples"
    return "explicit_instruction"


def _profile_preferences(profile: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = profile.get("preferences", [])
    if not isinstance(raw, list):
        return []
    result: list[dict[str, Any]] = []
    for index, value in enumerate(raw[:MAX_PREFERENCES]):
        statement = _text(value, 300)
        if not statement:
            continue
        result.append(
            {
                "memory_id": f"profile_preference_{index + 1:02d}",
                "kind": _preference_kind(statement),
                "statement": statement,
                "status": "confirmed_teacher_profile",
                "first_observed_round": 0,
                "last_confirmed_round": 0,
                "evidence_refs": [f"teacher_profile:preference:{index}"],
            }
        )
    return result


def _immutable_profile_context(profile: Mapping[str, Any]) -> dict[str, Any]:
    """Return all fixed profile fields while excluding rollout-derived state."""

    if not isinstance(profile, Mapping):
        raise TeachingMemoryError("student profile must be an object")
    return {
        str(key): deepcopy(value)
        for key, value in profile.items()
        if str(key) not in _DYNAMIC_PROFILE_FIELDS
    }


def teaching_memory_context_fingerprint(
    goal: Mapping[str, Any], profile: Mapping[str, Any]
) -> str:
    """Bind memory to the complete immutable goal/profile setup contract."""

    if not isinstance(goal, Mapping):
        raise TeachingMemoryError("teaching goal must be an object")
    return canonical_sha256(
        {
            "goal": deepcopy(dict(goal)),
            "student_profile": _immutable_profile_context(profile),
        }
    )


def initialize_teaching_memory(
    goal: Mapping[str, Any], profile: Mapping[str, Any]
) -> dict[str, Any]:
    """Create an empty, versioned memory bound to immutable setup context."""

    memory = {
        "schema": TEACHING_MEMORY_SCHEMA,
        "history_version": 0,
        "compaction_generation": 0,
        "fixed_context_fingerprint": teaching_memory_context_fingerprint(
            goal, profile
        ),
        "last_observed_round": 0,
        "preferences": _profile_preferences(profile),
        "open_questions": [],
        "commitments": [],
        "referents": [],
        "claim_boundary": {
            "model_generated_summary": False,
            "items_require_utterance_evidence": True,
            "question_resolution_requires_learner_confirmation": True,
            "teacher_profile_preferences_are_confirmed": True,
            "learner_utterance_preferences_are_explicit_unverified": True,
        },
    }
    validate_teaching_memory(memory)
    return memory


def _upsert_preference(
    preferences: list[dict[str, Any]], *, statement: str, round_number: int
) -> None:
    kind = _preference_kind(statement)
    existing = next(
        (
            item
            for item in reversed(preferences)
            if item.get("kind") == kind and item.get("status") != "superseded"
        ),
        None,
    )
    evidence_ref = f"session_history:r{round_number}:learner_response"
    if existing is not None:
        existing["statement"] = statement
        existing["last_confirmed_round"] = round_number
        existing["status"] = "explicit_learner_instruction"
        existing["evidence_refs"] = list(
            dict.fromkeys([*existing.get("evidence_refs", []), evidence_ref])
        )[-4:]
        return
    preferences.append(
        {
            "memory_id": _memory_id("preference", round_number, statement),
            "kind": kind,
            "statement": statement,
            "status": "explicit_learner_instruction",
            "first_observed_round": round_number,
            "last_confirmed_round": round_number,
            "evidence_refs": [evidence_ref],
        }
    )
    del preferences[:-MAX_PREFERENCES]


def _append_question(
    questions: list[dict[str, Any]], *, question: str, round_number: int
) -> dict[str, Any]:
    canonical = re.sub(r"[?？。！!\s]", "", question).casefold()
    existing = next(
        (
            item
            for item in reversed(questions)
            if re.sub(r"[?？。！!\s]", "", str(item.get("question", ""))).casefold()
            == canonical
            and item.get("status") != "resolved"
        ),
        None,
    )
    evidence_ref = f"session_history:r{round_number}:learner_response"
    if existing is not None:
        existing["last_raised_round"] = round_number
        existing["status"] = "open"
        existing["evidence_refs"] = list(
            dict.fromkeys([*existing.get("evidence_refs", []), evidence_ref])
        )[-4:]
        return existing
    item = {
        "memory_id": _memory_id("question", round_number, question),
        "question": question,
        "status": "open",
        "first_raised_round": round_number,
        "last_raised_round": round_number,
        "addressed_round": None,
        "resolved_round": None,
        "evidence_refs": [evidence_ref],
        "answer_evidence_refs": [],
    }
    questions.append(item)
    del questions[:-MAX_OPEN_QUESTIONS]
    return item


def _resolve_or_reopen_questions(
    questions: list[dict[str, Any]], *, learner_text: str, round_number: int
) -> None:
    if _REOPEN_RE.search(learner_text):
        candidate = next(
            (
                item
                for item in reversed(questions)
                if item.get("status") in {"addressed_pending_confirmation", "resolved"}
            ),
            None,
        )
        if candidate is not None:
            candidate["status"] = "open"
            candidate["resolved_round"] = None
            candidate["last_raised_round"] = round_number
            candidate["evidence_refs"] = list(
                dict.fromkeys(
                    [
                        *candidate.get("evidence_refs", []),
                        f"session_history:r{round_number}:learner_response",
                    ]
                )
            )[-4:]
        return
    if not _POSITIVE_ACK_RE.search(learner_text):
        return
    candidate = next(
        (
            item
            for item in reversed(questions)
            if item.get("status") == "addressed_pending_confirmation"
        ),
        None,
    )
    if candidate is not None:
        candidate["status"] = "resolved"
        candidate["resolved_round"] = round_number
        candidate["answer_evidence_refs"] = list(
            dict.fromkeys(
                [
                    *candidate.get("answer_evidence_refs", []),
                    f"session_history:r{round_number}:learner_response",
                ]
            )
        )[-4:]


def _record_teacher_action(
    memory: dict[str, Any], *, round_number: int, teacher_action: Mapping[str, Any]
) -> None:
    message = _text(teacher_action.get("message", ""), 700)
    if not message:
        return
    action_id = _text(teacher_action.get("action_id", ""), 120)
    teacher_ref = f"session_history:r{round_number}:teacher_action"
    open_question = next(
        (
            item
            for item in reversed(memory["open_questions"])
            if item.get("status") == "open"
        ),
        None,
    )
    if open_question is not None:
        open_question["status"] = "addressed_pending_confirmation"
        open_question["addressed_round"] = round_number
        open_question["answer_evidence_refs"] = list(
            dict.fromkeys(
                [*open_question.get("answer_evidence_refs", []), teacher_ref]
            )
        )[-4:]

    if _COMMITMENT_RE.search(message):
        memory["commitments"].append(
            {
                "memory_id": _memory_id("commitment", round_number, message),
                "statement": message,
                "status": "pending",
                "created_round": round_number,
                "action_id": action_id or None,
                "evidence_refs": [teacher_ref],
            }
        )
        del memory["commitments"][:-MAX_COMMITMENTS]

    if len(_REFERENT_MARKER_RE.findall(message)) >= 2:
        memory["referents"].append(
            {
                "memory_id": _memory_id("referents", round_number, message),
                "round": round_number,
                "description": message,
                "status": "active",
                "evidence_refs": [teacher_ref],
            }
        )
        del memory["referents"][:-MAX_REFERENTS]


def commit_teaching_memory_turn(
    memory: Mapping[str, Any],
    *,
    round_number: int,
    learner_text: str,
    teacher_action: Mapping[str, Any],
) -> dict[str, Any]:
    """Commit one completed learner→teacher turn to evidence-linked memory."""

    validate_teaching_memory(memory)
    if isinstance(round_number, bool) or not isinstance(round_number, int) or round_number < 1:
        raise TeachingMemoryError("round_number must be a positive integer")
    if round_number <= int(memory["last_observed_round"]):
        raise TeachingMemoryError("teaching memory turns must commit monotonically")

    updated = deepcopy(dict(memory))
    response = _text(learner_text, 700)
    _resolve_or_reopen_questions(
        updated["open_questions"],
        learner_text=response,
        round_number=round_number,
    )
    if response and _PREFERENCE_CUE_RE.search(response):
        _upsert_preference(
            updated["preferences"], statement=response, round_number=round_number
        )
    if response and _QUESTION_RE.search(response):
        _append_question(
            updated["open_questions"],
            question=response,
            round_number=round_number,
        )
    _record_teacher_action(
        updated,
        round_number=round_number,
        teacher_action=teacher_action,
    )
    updated["history_version"] = int(updated["history_version"]) + 1
    updated["compaction_generation"] = max(
        int(updated["compaction_generation"]), round_number // 8
    )
    updated["last_observed_round"] = round_number
    validate_teaching_memory(updated)
    return updated


def _rollout_learner_text(event: Mapping[str, Any]) -> str:
    """Use locally retained raw text, not a model/OCR transport envelope."""

    if "learner_text" in event:
        return str(event.get("learner_text") or "")
    return str(event.get("learner_response") or "")


def _rollout_teacher_action(event: Mapping[str, Any]) -> dict[str, Any]:
    """Return the teacher action that the learner actually answered."""

    action = event.get("action")
    if not isinstance(action, Mapping):
        raise TeachingMemoryError("session history action is invalid")
    teacher_action = action.get("teacher_action")
    if not isinstance(teacher_action, Mapping):
        raise TeachingMemoryError("session history teacher_action is invalid")
    return {
        **deepcopy(dict(teacher_action)),
        "action_id": action.get("action_id"),
    }


def _validate_rollout_checkpoint_trace(
    event: Mapping[str, Any], memory: Mapping[str, Any]
) -> None:
    trace = event.get("teaching_memory_trace")
    expected = {
        "history_version": memory["history_version"],
        "compaction_generation": memory["compaction_generation"],
        "fixed_context_fingerprint": memory["fixed_context_fingerprint"],
        "content_sha256": canonical_sha256(memory),
        "source": "deterministic_evidence_linked_rollout_projection",
        "model_generated_summary": False,
    }
    if not isinstance(trace, Mapping) or dict(trace) != expected:
        raise TeachingMemoryError(
            "session history teaching_memory_trace does not match canonical replay"
        )


def rebuild_teaching_memory_from_rollout(
    goal: Mapping[str, Any],
    profile: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    *,
    validate_checkpoint_traces: bool = False,
) -> dict[str, Any]:
    """Reconstruct memory solely from immutable setup and durable turn events.

    A stored memory object is deliberately not an input.  Each event must be
    the next contiguous round and contributes only its retained learner text
    plus the teacher action embedded in that same event.  Consequently every
    ``session_history:rNNN:*`` pointer is reproducible from its referenced row.
    """

    if not isinstance(history, Sequence) or isinstance(
        history, (str, bytes, bytearray)
    ):
        raise TeachingMemoryError("session history must be a sequence")
    memory = initialize_teaching_memory(goal, profile)
    for expected_round, event in enumerate(history, 1):
        if not isinstance(event, Mapping):
            raise TeachingMemoryError("session history event is invalid")
        round_number = event.get("round")
        if (
            isinstance(round_number, bool)
            or not isinstance(round_number, int)
            or round_number != expected_round
        ):
            raise TeachingMemoryError(
                "session history rounds must be contiguous and one-based"
            )
        memory = commit_teaching_memory_turn(
            memory,
            round_number=round_number,
            learner_text=_rollout_learner_text(event),
            teacher_action=_rollout_teacher_action(event),
        )
        if validate_checkpoint_traces:
            _validate_rollout_checkpoint_trace(event, memory)
    return memory


def validate_teaching_memory_checkpoint(
    memory: Mapping[str, Any],
    *,
    goal: Mapping[str, Any],
    profile: Mapping[str, Any],
    history: Sequence[Mapping[str, Any]],
    expected_round: int,
) -> dict[str, Any]:
    """Require a stored checkpoint to equal deterministic history replay."""

    validate_teaching_memory(memory)
    if (
        isinstance(expected_round, bool)
        or not isinstance(expected_round, int)
        or expected_round < 0
        or len(history) != expected_round
    ):
        raise TeachingMemoryError(
            "session round does not match durable teaching-memory history"
        )
    rebuilt = rebuild_teaching_memory_from_rollout(
        goal,
        profile,
        history,
        validate_checkpoint_traces=True,
    )
    if dict(memory) != rebuilt or canonical_sha256(memory) != canonical_sha256(
        rebuilt
    ):
        raise TeachingMemoryError(
            "teaching memory checkpoint differs from canonical history replay"
        )
    return rebuilt


def project_teaching_memory(
    memory: Mapping[str, Any], *, content_limit: int = 240
) -> dict[str, Any]:
    """Return the bounded model-visible projection, preserving evidence refs."""

    validate_teaching_memory(memory)
    limit = max(48, min(int(content_limit), 400))
    preferences = [
        {
            **deepcopy(item),
            "statement": _text(item.get("statement", ""), limit),
        }
        for item in memory["preferences"][-6:]
        if item.get("status") != "superseded"
    ]
    questions = [
        {
            **deepcopy(item),
            "question": _text(item.get("question", ""), limit),
        }
        for item in memory["open_questions"]
        if item.get("status") != "resolved"
    ][-6:]
    commitments = [
        {
            **deepcopy(item),
            "statement": _text(item.get("statement", ""), limit),
        }
        for item in memory["commitments"]
        if item.get("status") == "pending"
    ][-4:]
    referents = [
        {
            **deepcopy(item),
            "description": _text(item.get("description", ""), limit),
        }
        for item in memory["referents"]
        if item.get("status") == "active"
    ][-4:]
    return {
        "schema": TEACHING_MEMORY_PROJECTION_SCHEMA,
        "history_version": memory["history_version"],
        "compaction_generation": memory["compaction_generation"],
        "fixed_context_fingerprint": memory["fixed_context_fingerprint"],
        "active_preferences": preferences,
        "unresolved_questions": questions,
        "pending_teacher_commitments": commitments,
        "active_referents": referents,
        "source": "deterministic_evidence_linked_rollout_projection",
        "narrative_inference_added": False,
        "model_may_mutate": False,
    }


def validate_teaching_memory(memory: Mapping[str, Any]) -> None:
    if not isinstance(memory, Mapping) or memory.get("schema") != TEACHING_MEMORY_SCHEMA:
        raise TeachingMemoryError(f"teaching memory schema must be {TEACHING_MEMORY_SCHEMA}")
    for field in ("history_version", "compaction_generation", "last_observed_round"):
        value = memory.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise TeachingMemoryError(f"teaching memory {field} is invalid")
    fingerprint = memory.get("fixed_context_fingerprint")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-f]{64}", fingerprint):
        raise TeachingMemoryError("teaching memory fixed_context_fingerprint is invalid")
    limits = {
        "preferences": MAX_PREFERENCES,
        "open_questions": MAX_OPEN_QUESTIONS,
        "commitments": MAX_COMMITMENTS,
        "referents": MAX_REFERENTS,
    }
    for field, maximum in limits.items():
        rows = memory.get(field)
        if not isinstance(rows, list) or len(rows) > maximum:
            raise TeachingMemoryError(f"teaching memory {field} is invalid")
        identifiers = []
        for item in rows:
            if not isinstance(item, Mapping):
                raise TeachingMemoryError(f"teaching memory {field} item is invalid")
            memory_id = item.get("memory_id")
            evidence_refs = item.get("evidence_refs")
            if (
                not isinstance(memory_id, str)
                or not memory_id
                or not isinstance(evidence_refs, list)
                or not evidence_refs
                or any(not isinstance(ref, str) or not ref for ref in evidence_refs)
            ):
                raise TeachingMemoryError(
                    f"teaching memory {field} provenance is invalid"
                )
            identifiers.append(memory_id)
        if len(identifiers) != len(set(identifiers)):
            raise TeachingMemoryError(f"teaching memory {field} IDs must be unique")
    boundary = memory.get("claim_boundary")
    if not isinstance(boundary, Mapping) or any(
        boundary.get(key) is not expected
        for key, expected in {
            "model_generated_summary": False,
            "items_require_utterance_evidence": True,
            "question_resolution_requires_learner_confirmation": True,
            "teacher_profile_preferences_are_confirmed": True,
            "learner_utterance_preferences_are_explicit_unverified": True,
        }.items()
    ):
        raise TeachingMemoryError("teaching memory claim boundary is invalid")
