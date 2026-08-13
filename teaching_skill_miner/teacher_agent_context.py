"""Privacy-safe context construction for the task-two Teaching Agent.

The functions in this module are deliberately deterministic and dependency
free.  They prepare the smallest useful learner context for a remote language
model, but they do not perform a network request or claim that a model-derived
student state is correct.
"""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import json
import re
from typing import Any, Mapping, Sequence

from .teacher_agent_discourse import classify_learner_discourse_text
from .teacher_agent import lesson_required_primary_roles
from .teacher_agent_memory import (
    TEACHING_MEMORY_PROJECTION_SCHEMA,
    project_teaching_memory,
    validate_teaching_memory,
)
from .student_model import project_student_model


CONTEXT_SCHEMA = "teaching_skill_miner.teacher_agent_context.v1"
LAYERED_CONTEXT_SCHEMA = "teaching_skill_miner.layered_teacher_context.v1"
GOAL_PLAN_SCHEMA = "teaching_skill_miner.teacher_agent_goal_plan.v1"
CONTINUITY_RECALL_SCHEMA = "teaching_skill_miner.teacher_agent_continuity_recall.v1"

DEFAULT_LAYERED_CONTEXT_CHARS = 14_000
MINIMUM_LAYERED_CONTEXT_CHARS = 6_000

_REPLACEMENTS = {
    "email": "[REDACTED_EMAIL]",
    "phone": "[REDACTED_PHONE]",
    "cn_id": "[REDACTED_CN_ID]",
    "url": "[REDACTED_URL]",
    "local_path": "[REDACTED_LOCAL_PATH]",
}

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "url",
        re.compile(
            r"(?i)(?<![\w@])(?:https?://|ftp://|file://|www\.)"
            r"[^\s<>\"'，。！？；、（）【】《》]+"
        ),
    ),
    (
        "email",
        re.compile(
            r"(?i)(?<![A-Z0-9._%+\-])"
            r"[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}"
            r"(?![A-Z0-9._%+\-])"
        ),
    ),
    (
        "cn_id",
        re.compile(r"(?<!\d)\d{17}[0-9Xx](?![0-9Xx])"),
    ),
    (
        "phone",
        re.compile(
            r"(?<!\d)(?:(?:\+?86[\s.-]?)?1[3-9]\d(?:[\s.-]?\d){8}"
            r"|(?:\+?1[\s.-]?)?(?:\([2-9]\d{2}\)|[2-9]\d{2})"
            r"[\s.-]?\d{3}[\s.-]?\d{4})(?!\d)"
        ),
    ),
    (
        "local_path",
        re.compile(
            r"(?<![A-Za-z0-9])(?:"
            r"[A-Za-z]:\\(?:[^\\\r\n<>:\"|?*]+\\)*[^\\\r\n<>:\"|?*]*"
            r"|/(?!/)(?:[^/\s<>:\"|?*]+/)*[^/\s<>:\"|?*]+"
            r")"
        ),
    ),
)

_TRAILING_PATH_PUNCTUATION = ".,;!?)]}，。；！？）》】」』"
_TEXT_TRUNCATION_MARKER = "…[truncated]"
_MIDDLE_TRUNCATION_MARKER = "…[middle truncated]…"
_COMMON_POSIX_PATH_PREFIXES = (
    "/Applications/",
    "/Library/",
    "/System/",
    "/Users/",
    "/Volumes/",
    "/data/",
    "/etc/",
    "/home/",
    "/mnt/",
    "/opt/",
    "/private/",
    "/s1home/",
    "/tmp/",
    "/usr/",
    "/var/",
    "/work/",
    "/workspace/",
)
_EXPLICIT_LEARNER_QUESTION_RE = re.compile(
    r"(?:[?？]\s*$|(?:为什么|怎么|如何|哪一步|哪里|是否|能否|是不是|会不会)|"
    r"分(?:成|为)?(?:哪几|哪些|几)(?:个)?部分|"
    r"由(?:什么|哪些|哪几(?:个)?).{0,12}(?:组成|构成)|"
    r"(?:包括|包含)(?:什么|哪些|哪几(?:个)?)(?:部分|要素|成分|内容)?|"
    r"(?:有|涉及)(?:哪几|几)(?:个)?(?:部分|要素|成分|内容))"
)
_EXPLICIT_LEARNER_QUESTION_DECLARATION_RE = re.compile(
    r"^(?:我|我们).{0,40}(?:把|将).{0,40}(?:分成|分为).{0,20}"
    r"(?:个|部分)(?:了)?[！!。.]*$"
)


def _is_explicit_learner_question_text(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    if classify_learner_discourse_text(text).is_question:
        return True
    return bool(_EXPLICIT_LEARNER_QUESTION_RE.search(text)) and not bool(
        _EXPLICIT_LEARNER_QUESTION_DECLARATION_RE.search(text)
    )


_EXPLICIT_LEARNER_REQUEST_RE = re.compile(
    r"(?:^|[，。！？；,.!?;\s])(?:请|能不能|可不可以|我(?:希望|想要|更喜欢|需要)|不要|别)"
)
_TEACHER_NEXT_STEP_RE = re.compile(
    r"(?:下一步|接下来|随后|之后(?:我|我们)?|然后(?:我|我们)?|我会|我们会|我们将)"
)
_NAMED_ALTERNATIVE_MARKER_RE = re.compile(
    r"(?:第\s*[一二三四五六七八九十\d]+\s*种|"
    r"方法\s*[一二三四五六七八九十\dABC]|"
    r"方案\s*[一二三四五六七八九十\dABC]|"
    r"路径\s*[一二三四五六七八九十\dABC]|"
    r"(?:^|[，,；;。\n])\s*[123]\s*[.、)])",
    re.IGNORECASE,
)
_ORDINAL_REFERENCE_CUE_RE = re.compile(
    r"(?:第\s*[一二三四五六七八九十\d]+\s*种|"
    r"方法\s*[一二三四五六七八九十\dABC]|"
    r"方案\s*[一二三四五六七八九十\dABC]|"
    r"前一种|后一种|另一个(?:方法|方案)?|前一个|后一个)",
    re.IGNORECASE,
)
_EARLIEST_INSTRUCTION_CUE_RE = re.compile(
    r"(?:按|照|继续用|还是用).{0,12}(?:最开始|一开始|最初).{0,12}"
    r"(?:说|提|要求|希望|方式|方法)",
    re.IGNORECASE,
)
_EARLIER_QUESTION_CUE_RE = re.compile(
    r"(?:回到|继续|回答).{0,12}(?:一开始|最开始|最初|前面|之前|刚才).{0,12}"
    r"(?:问题|疑问)|(?:一开始|最开始|最初|前面|之前|刚才).{0,12}"
    r"(?:问题|疑问).{0,12}(?:呢|继续|回答|回去)",
    re.IGNORECASE,
)
_PRIOR_AGREEMENT_CUE_RE = re.compile(
    r"(?:按|照).{0,12}(?:刚才|之前|前面|先前).{0,12}"
    r"(?:约定|说好|说的|安排).{0,12}(?:继续|来|做|讲|走)?|"
    r"(?:刚才|之前|前面|先前).{0,12}(?:约定|说好|安排).{0,12}"
    r"(?:继续|下一步|往下)",
    re.IGNORECASE,
)
_CONTINUITY_COMPLETION_STATUS_CUE_RE = re.compile(
    r"(?:未完成|还没完成|尚未|待完成|还差|下一步|还需要|遗漏|"
    r"哪里.{0,8}(?:完成|遗漏|还差|需要))",
    re.IGNORECASE,
)
_ROUND_REFERENCE_CUE_RE = re.compile(
    r"第\s*([\d零〇一二两三四五六七八九十百]+)\s*轮",
    re.IGNORECASE,
)
_SEMANTIC_HISTORY_CUE_RE = re.compile(
    r"(?:回到|回顾|重新(?:讲|解释|看|说|做)|再(?:讲|解释|看|说一下)|"
    r"之前.{0,16}(?:讲|提|说|学|问|讨论)|"
    r"前面.{0,16}(?:讲|提|说|学|问|讨论)|"
    r"刚才.{0,16}(?:讲|提|说|学|问|讨论|个))",
    re.IGNORECASE,
)
_CURRENT_ACTION_DEICTIC_CUE_RE = re.compile(
    r"(?:刚才|刚刚|方才|前面)\s*(?:的)?\s*(?:这|那)?(?:一)?个"
    r"(?:问题|概念|例子|步骤|部分|内容|说法|东西|点)?|"
    r"(?:刚才|刚刚|方才)\s*(?:问|说|讲|提)(?:到)?(?:的)?\s*"
    r"(?:这个|那个|内容|问题)",
    re.IGNORECASE,
)
_RECENT_REPETITION_COMPLAINT_CUE_RE = re.compile(
    r"(?:之前|刚才|前面).{0,24}(?:不是)?(?:说|讲|表示|告诉).{0,12}"
    r"(?:不会|不懂|没懂|不知道|不明白|没明白)|"
    r"(?:我都|我已经|已经).{0,16}(?:说|表示|告诉).{0,10}"
    r"(?:不会|不懂|没懂|不知道|不明白|没明白)|"
    r"(?:怎么|为什么|为何).{0,16}(?:又|还|重复|再).{0,12}"
    r"(?:问|让我答|让我说|让我解释)",
    re.IGNORECASE,
)
_LEARNER_DIFFICULTY_SIGNAL_RE = re.compile(
    r"^(?:我)?(?:还是|仍然|真的|完全|确实|也)?"
    r"(?:不会|不懂|没懂|不知道|不明白|没明白|答不出|说不出|看不懂)"
    r"(?:了|啊|呀|呢|吧|。|！|!|？|\?|\s)*$",
    re.IGNORECASE,
)
_LEARNER_FUTURE_AGENDA_RE = re.compile(
    r"(?:之后|稍后|等会(?:儿)?|待会(?:儿)?|下一步|然后).{0,18}"
    r"(?:再|回到|继续|讲|看|做|处理|讨论)|"
    r"(?:先).{1,120}(?:，|,|；|;|。)?\s*(?:再|然后|之后).{1,120}",
    re.IGNORECASE,
)

_QUERY_STOP_TERMS = {
    "一下",
    "之前",
    "前面",
    "刚才",
    "回到",
    "回顾",
    "重新",
    "解释",
    "讲的",
    "问题",
    "关系",
    "这个",
    "那个",
    "第一",
    "第二",
    "第三",
}


def _trim_candidate(kind: str, text: str, start: int, end: int) -> tuple[int, int]:
    if kind in {"url", "local_path"}:
        while end > start and text[end - 1] in _TRAILING_PATH_PUNCTUATION:
            end -= 1
    return start, end


def _looks_like_local_path(value: str) -> bool:
    """Reject division/formula fragments without weakening real path redaction."""

    if re.match(r"^[A-Za-z]:\\", value):
        return True
    if not value.startswith("/"):
        return False
    if value.startswith(_COMMON_POSIX_PATH_PREFIXES):
        return True
    # Expressions such as ``dp[i]/dp[i-1]`` and ``f(n)/g(n)`` used to lose
    # their denominator because ``/dp[i-1]`` looked like a one-segment path.
    if any(character in value for character in "[](){}=+*^"):
        return False
    basename = value.rsplit("/", 1)[-1]
    has_file_extension = bool(re.search(r"\.[A-Za-z0-9]{1,12}$", basename))
    return value.count("/") >= 2 or has_file_extension


def redact_remote_text(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Redact common direct identifiers before text leaves the machine.

    Findings contain only type, span and replacement metadata; the sensitive
    source value is intentionally never copied into the returned audit list.
    Overlapping matches are resolved by occurrence, then by the declared
    pattern priority (URL before path, for example).
    """

    if not isinstance(text, str):
        raise TypeError("text must be a string")

    candidates: list[tuple[int, int, int, str]] = []
    for priority, (kind, pattern) in enumerate(_PATTERNS):
        for match in pattern.finditer(text):
            start, end = _trim_candidate(kind, text, match.start(), match.end())
            candidate = text[start:end]
            if end > start and (
                kind != "local_path" or _looks_like_local_path(candidate)
            ):
                candidates.append((start, end, priority, kind))

    candidates.sort(key=lambda row: (row[0], row[2], -(row[1] - row[0])))
    accepted: list[tuple[int, int, str]] = []
    occupied_until = -1
    for start, end, _priority, kind in candidates:
        if start < occupied_until:
            continue
        accepted.append((start, end, kind))
        occupied_until = end

    if not accepted:
        return text, []

    parts: list[str] = []
    findings: list[dict[str, Any]] = []
    cursor = 0
    for start, end, kind in accepted:
        replacement = _REPLACEMENTS[kind]
        parts.extend((text[cursor:start], replacement))
        findings.append(
            {
                "kind": kind,
                "start": start,
                "end": end,
                "replacement": replacement,
            }
        )
        cursor = end
    parts.append(text[cursor:])
    return "".join(parts), findings


def _as_nonempty_string(value: Any) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _string_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, Mapping):
        result: list[str] = []
        for key, item in value.items():
            if isinstance(item, bool) and not item:
                continue
            candidate = _as_nonempty_string(key)
            if candidate:
                result.append(candidate)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        result = []
        for item in value:
            if isinstance(item, Mapping):
                candidate = _as_nonempty_string(
                    item.get("knowledge_component")
                    or item.get("kc_id")
                    or item.get("id")
                    or item.get("name")
                )
            else:
                candidate = _as_nonempty_string(item)
            if candidate:
                result.append(candidate)
        return result
    return []


def _knowledge_components(value: Mapping[str, Any]) -> list[str]:
    result: list[str] = []
    keys = (
        "knowledge_component",
        "knowledge_components",
        "kc",
        "kc_id",
        "kc_ids",
        "target_knowledge_components",
    )
    for key in keys:
        result.extend(_string_values(value.get(key)))
    return list(dict.fromkeys(result))


def _current_focus_and_kcs(session: Mapping[str, Any]) -> tuple[str | None, list[str]]:
    state = session.get("student_state", {})
    next_focus = state.get("next_focus", {}) if isinstance(state, Mapping) else {}
    action = session.get("current_action", {})
    primary = action.get("primary_skill", {}) if isinstance(action, Mapping) else {}
    goal = session.get("goal", {})

    focus = None
    for container in (next_focus, primary, action):
        if isinstance(container, Mapping):
            focus = _as_nonempty_string(
                container.get("dimension") or container.get("focus_dimension")
            )
            if focus:
                break

    kcs: list[str] = []
    for container in (next_focus, primary, action):
        if isinstance(container, Mapping):
            kcs.extend(_knowledge_components(container))
    if not kcs and isinstance(goal, Mapping):
        # Legacy sessions may not carry action-level KC tags.  Only then use
        # the complete goal as a conservative retrieval fallback.
        kcs.extend(_knowledge_components(goal))
    return focus, list(dict.fromkeys(kcs))


def _event_view(event: Mapping[str, Any], fallback_round: int) -> dict[str, Any]:
    action = event.get("action", {})
    primary = action.get("primary_skill", {}) if isinstance(action, Mapping) else {}
    teacher = action.get("teacher_action", {}) if isinstance(action, Mapping) else {}
    mastery_signal = event.get("structured_signal", {})
    if not isinstance(mastery_signal, Mapping):
        mastery_signal = {}
    pedagogical_signal = event.get("pedagogical_signal", {})
    signal = (
        pedagogical_signal
        if isinstance(pedagogical_signal, Mapping)
        and _as_nonempty_string(pedagogical_signal.get("label"))
        else mastery_signal
    )
    focus = None
    if isinstance(primary, Mapping):
        focus = _as_nonempty_string(primary.get("focus_dimension"))
    if not focus:
        focus = _as_nonempty_string(event.get("focus_dimension"))

    kcs: list[str] = []
    for container in (event, action, primary):
        if isinstance(container, Mapping):
            kcs.extend(_knowledge_components(container))

    return {
        "round": event.get("round", fallback_round),
        "focus_dimension": focus,
        "knowledge_components": list(dict.fromkeys(kcs)),
        "skill_id": (
            _as_nonempty_string(primary.get("skill_id"))
            if isinstance(primary, Mapping)
            else None
        ),
        "teacher_message": (
            str(teacher.get("message", "")) if isinstance(teacher, Mapping) else ""
        ),
        "learner_response": str(event.get("learner_response", "")),
        "signal": _as_nonempty_string(signal.get("label")),
        "signal_confidence": signal.get("confidence"),
        "signal_source": _as_nonempty_string(signal.get("source")),
    }


def _named_alternative_marker_count(value: str) -> int:
    markers = {
        re.sub(r"[\s，,；;。\n.、)]", "", match.group(0)).casefold()
        for match in _NAMED_ALTERNATIVE_MARKER_RE.finditer(str(value))
    }
    return len({item for item in markers if item})


def _parse_positive_ordinal(value: str) -> int | None:
    token = str(value).strip()
    if not token:
        return None
    if token.isdigit():
        number = int(token)
        return number if number > 0 else None
    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    if token in digits:
        return digits[token] or None
    if token == "十":
        return 10
    if "百" in token:
        left, _, right = token.partition("百")
        hundreds = digits.get(left, 1 if not left else -1)
        if hundreds < 0:
            return None
        remainder = _parse_positive_ordinal(right) if right else 0
        if remainder is None:
            return None
        result = hundreds * 100 + remainder
        return result if result > 0 else None
    if "十" in token:
        left, _, right = token.partition("十")
        tens = digits.get(left, 1 if not left else -1)
        ones = digits.get(right, 0 if not right else -1)
        if tens < 0 or ones < 0:
            return None
        result = tens * 10 + ones
        return result if result > 0 else None
    return None


def _requested_round(value: str) -> int | None:
    match = _ROUND_REFERENCE_CUE_RE.search(str(value))
    return _parse_positive_ordinal(match.group(1)) if match else None


def _query_terms(value: str) -> set[str]:
    """Return small deterministic lexical anchors without a tokenizer/model."""

    terms: set[str] = set()
    for token in re.findall(
        r"[A-Za-z][A-Za-z0-9_\-]{1,}|[\u4e00-\u9fff]{2,}", str(value)
    ):
        normalized = token.casefold()
        if re.fullmatch(r"[\u4e00-\u9fff]+", normalized):
            for size in (2, 3, 4):
                for index in range(max(0, len(normalized) - size + 1)):
                    term = normalized[index : index + size]
                    if term not in _QUERY_STOP_TERMS:
                        terms.add(term)
        elif normalized not in _QUERY_STOP_TERMS:
            terms.add(normalized)
    return terms


def _event_query_score(
    event: Mapping[str, Any],
    learner_response: str,
    *,
    requested_round: int | None = None,
    query_terms: set[str] | None = None,
) -> tuple[int, int, int]:
    if requested_round is None:
        requested_round = _requested_round(learner_response)
    if query_terms is None:
        query_terms = _query_terms(learner_response)
    round_number = _safe_round(event.get("round"), 0)
    explicit_round_match = int(
        requested_round is not None and round_number == requested_round
    )
    query_folded = learner_response.casefold()
    mentioned_kcs = sum(
        1
        for item in event.get("knowledge_components", [])
        if str(item).strip() and str(item).strip().casefold() in query_folded
    )
    searchable = "\n".join(
        [
            str(event.get("teacher_message", "")),
            str(event.get("learner_response", "")),
            " ".join(str(item) for item in event.get("knowledge_components", [])),
        ]
    )
    lexical_overlap = len(query_terms & _query_terms(searchable))
    return explicit_round_match, mentioned_kcs, lexical_overlap


def _continuity_cue_kind(value: str) -> str | None:
    """Classify only explicit backward-looking learner references.

    A turn that defines two alternatives is not itself a recall request.  This
    prevents ``第一种...第二种...`` from being treated as missing history before
    the just-written alternatives have been committed.
    """

    text = str(value).strip()
    if not text:
        return None
    if _ROUND_REFERENCE_CUE_RE.search(text):
        return "explicit_round_reference"
    if _EARLIER_QUESTION_CUE_RE.search(text):
        return "earlier_unresolved_question"
    if _EARLIEST_INSTRUCTION_CUE_RE.search(text):
        return "earliest_learner_instruction"
    if _PRIOR_AGREEMENT_CUE_RE.search(text):
        return "prior_agreement_or_agenda"
    if _RECENT_REPETITION_COMPLAINT_CUE_RE.search(text):
        return "recent_repetition_complaint"
    if _named_alternative_marker_count(text) < 2 and _ORDINAL_REFERENCE_CUE_RE.search(
        text
    ):
        return "ordinal_reference"
    if _SEMANTIC_HISTORY_CUE_RE.search(text):
        return "semantic_topic_reference"
    return None


def _continuity_completion_status_requested(value: str) -> bool:
    """Return whether a prior-agreement cue also asks for open-work status.

    This is intentionally a narrow lexical gate.  It is only used after the
    caller has classified the utterance as ``prior_agreement_or_agenda`` so a
    standalone question containing ``下一步`` cannot steal an unrelated
    continuity target.
    """

    return bool(_CONTINUITY_COMPLETION_STATUS_CUE_RE.search(str(value)))


def _history_continuity_candidates(
    session: Mapping[str, Any],
) -> list[dict[str, Any]]:
    history = session.get("history", [])
    if not isinstance(history, list):
        return []
    candidates: list[dict[str, Any]] = []
    for fallback_round, raw_event in enumerate(history, 1):
        if not isinstance(raw_event, Mapping):
            continue
        round_number = _safe_round(raw_event.get("round"), fallback_round)
        learner_text = str(
            raw_event.get("learner_text", raw_event.get("learner_response", "")) or ""
        ).strip()
        action = raw_event.get("action", {})
        teacher_action = (
            action.get("teacher_action", {}) if isinstance(action, Mapping) else {}
        )
        teacher_text = (
            str(teacher_action.get("message", "") or "").strip()
            if isinstance(teacher_action, Mapping)
            else ""
        )
        event_view = _event_view(raw_event, fallback_round)
        generic_excerpt = teacher_text or learner_text
        if generic_excerpt:
            generic_field = (
                "action.teacher_action.message" if teacher_text else "learner_response"
            )
            generic_speaker = "teacher" if teacher_text else "learner"
            candidates.append(
                {
                    "kind": "historical_turn",
                    "speaker": generic_speaker,
                    "round": round_number,
                    "excerpt": generic_excerpt,
                    "evidence_refs": [
                        (
                            f"session_history:r{round_number}:teacher_action"
                            if teacher_text
                            else f"session_history:r{round_number}:learner_response"
                        )
                    ],
                    "source": "session_history",
                    "field": generic_field,
                    "knowledge_components": event_view.get("knowledge_components", []),
                    "teacher_message": teacher_text,
                    "learner_response": learner_text,
                }
            )
        if learner_text and _named_alternative_marker_count(learner_text) >= 2:
            candidates.append(
                {
                    "kind": "learner_named_alternatives",
                    "speaker": "learner",
                    "round": round_number,
                    "excerpt": learner_text,
                    "evidence_refs": [
                        f"session_history:r{round_number}:learner_response"
                    ],
                    "source": "learner_defined_alternatives",
                    "field": "learner_response",
                }
            )
        if learner_text and _LEARNER_DIFFICULTY_SIGNAL_RE.search(learner_text):
            candidates.append(
                {
                    "kind": "learner_difficulty_signal",
                    "speaker": "learner",
                    "round": round_number,
                    "excerpt": learner_text,
                    "evidence_refs": [
                        f"session_history:r{round_number}:learner_response"
                    ],
                    "source": "learner_explicit_difficulty",
                    "field": "learner_response",
                }
            )
        if learner_text and _LEARNER_FUTURE_AGENDA_RE.search(learner_text):
            candidates.append(
                {
                    "kind": "learner_future_agenda",
                    "speaker": "learner",
                    "round": round_number,
                    "excerpt": learner_text,
                    "evidence_refs": [
                        f"session_history:r{round_number}:learner_response"
                    ],
                    "source": "learner_requested_future_agenda",
                    "field": "learner_response",
                }
            )
        if teacher_text and _named_alternative_marker_count(teacher_text) >= 2:
            candidates.append(
                {
                    "kind": "teacher_named_alternatives",
                    "speaker": "teacher",
                    "round": round_number,
                    "excerpt": teacher_text,
                    "evidence_refs": [
                        f"session_history:r{round_number}:teacher_action"
                    ],
                    "source": "teacher_named_alternatives",
                    "field": "action.teacher_action.message",
                }
            )
        if (
            teacher_text
            and isinstance(teacher_action, Mapping)
            and (
                teacher_action.get("question_id")
                or _EXPLICIT_LEARNER_QUESTION_RE.search(teacher_text)
            )
        ):
            candidates.append(
                {
                    "kind": "teacher_question",
                    "speaker": "teacher",
                    "round": round_number,
                    "excerpt": teacher_text,
                    "evidence_refs": [
                        f"session_history:r{round_number}:teacher_action"
                    ],
                    "source": "teacher_question",
                    "field": "action.teacher_action.message",
                }
            )
    return candidates


def _current_action_continuity_candidates(
    session: Mapping[str, Any],
) -> list[dict[str, Any]]:
    action = session.get("current_action", {})
    if not isinstance(action, Mapping):
        return []
    teacher_action = action.get("teacher_action", {})
    if not isinstance(teacher_action, Mapping):
        return []
    message = str(teacher_action.get("message", "") or "").strip()
    if not message:
        return []
    round_number = _safe_round(action.get("round"), int(session.get("round", 0)) + 1)
    evidence_ref = f"current_action:r{round_number}:teacher_action"
    # The action currently visible to the learner is the first referent for
    # phrases such as ``刚才那个``.  It has not been committed to history yet,
    # so expose one generic, evidence-linked candidate in addition to the
    # narrower question/alternative/commitment projections below.
    candidates: list[dict[str, Any]] = [
        {
            "kind": "current_teacher_action",
            "speaker": "teacher",
            "round": round_number,
            "excerpt": message,
            "evidence_refs": [evidence_ref],
            "source": "current_validated_teacher_action",
            "field": "current_action.teacher_action.message",
        }
    ]
    if _named_alternative_marker_count(message) >= 2:
        candidates.append(
            {
                "kind": "teacher_named_alternatives",
                "speaker": "teacher",
                "round": round_number,
                "excerpt": message,
                "evidence_refs": [evidence_ref],
                "source": "current_validated_teacher_action",
                "field": "current_action.teacher_action.message",
            }
        )
    if _TEACHER_NEXT_STEP_RE.search(message):
        candidates.append(
            {
                "kind": "teacher_commitment",
                "speaker": "teacher",
                "round": round_number,
                "excerpt": message,
                "evidence_refs": [evidence_ref],
                "source": "current_validated_teacher_action",
                "field": "current_action.teacher_action.message",
            }
        )
    if teacher_action.get("question_id") or _EXPLICIT_LEARNER_QUESTION_RE.search(
        message
    ):
        candidates.append(
            {
                "kind": "teacher_question",
                "speaker": "teacher",
                "round": round_number,
                "excerpt": message,
                "evidence_refs": [evidence_ref],
                "source": "current_validated_teacher_action",
                "field": "current_action.teacher_action.message",
            }
        )
    return candidates


def _memory_continuity_candidates(
    session: Mapping[str, Any],
) -> list[dict[str, Any]]:
    memory = session.get("teaching_memory")
    if not isinstance(memory, Mapping):
        return []
    candidates: list[dict[str, Any]] = []
    for item in memory.get("preferences", []):
        if not isinstance(item, Mapping) or item.get("status") == "superseded":
            continue
        status = str(item.get("status", ""))
        candidates.append(
            {
                "kind": "learner_instruction",
                "speaker": (
                    "teacher_profile"
                    if status == "confirmed_teacher_profile"
                    else "learner"
                ),
                "round": _safe_round(item.get("first_observed_round"), 0),
                "excerpt": str(item.get("statement", "")),
                "evidence_refs": list(item.get("evidence_refs", [])),
                "source": (
                    "teacher_provided_student_profile"
                    if status == "confirmed_teacher_profile"
                    else "learner_explicit_preference"
                ),
                "field": "teaching_memory.preferences.statement",
            }
        )
    for item in memory.get("open_questions", []):
        if not isinstance(item, Mapping) or item.get("status") == "resolved":
            continue
        candidates.append(
            {
                "kind": "unresolved_learner_question",
                "speaker": "learner",
                "round": _safe_round(item.get("first_raised_round"), 0),
                "excerpt": str(item.get("question", "")),
                "evidence_refs": list(item.get("evidence_refs", [])),
                "source": "learner_question",
                "field": "teaching_memory.open_questions.question",
            }
        )
    for item in memory.get("commitments", []):
        if not isinstance(item, Mapping) or item.get("status") != "pending":
            continue
        candidates.append(
            {
                "kind": "teacher_commitment",
                "speaker": "teacher",
                "round": _safe_round(item.get("created_round"), 0),
                "excerpt": str(item.get("statement", "")),
                "evidence_refs": list(item.get("evidence_refs", [])),
                "source": "teacher_commitment",
                "field": "teaching_memory.commitments.statement",
            }
        )
    for item in memory.get("referents", []):
        if not isinstance(item, Mapping) or item.get("status") != "active":
            continue
        candidates.append(
            {
                "kind": "teacher_named_alternatives",
                "speaker": "teacher",
                "round": _safe_round(item.get("round"), 0),
                "excerpt": str(item.get("description", "")),
                "evidence_refs": list(item.get("evidence_refs", [])),
                "source": "teacher_named_alternatives",
                "field": "teaching_memory.referents.description",
            }
        )
    return candidates


def _select_continuity_target(
    cue_kind: str,
    candidates: Sequence[Mapping[str, Any]],
    learner_response: str,
) -> Mapping[str, Any] | None:
    if cue_kind == "explicit_round_reference":
        requested_round = _requested_round(learner_response)
        if requested_round is None:
            return None
        return next(
            (
                item
                for item in candidates
                if item.get("kind") == "historical_turn"
                and _safe_round(item.get("round"), 0) == requested_round
            ),
            None,
        )
    if cue_kind == "semantic_topic_reference":
        if _CURRENT_ACTION_DEICTIC_CUE_RE.search(learner_response):
            current_targets = [
                item
                for item in candidates
                if item.get("kind") == "current_teacher_action"
                and item.get("source") == "current_validated_teacher_action"
            ]
            if current_targets:
                return max(
                    current_targets,
                    key=lambda item: _safe_round(item.get("round"), 0),
                )
        eligible = [
            item for item in candidates if item.get("kind") == "historical_turn"
        ]
        requested_round = _requested_round(learner_response)
        query_terms = _query_terms(learner_response)
        scored = [
            (
                _event_query_score(
                    item,
                    learner_response,
                    requested_round=requested_round,
                    query_terms=query_terms,
                ),
                item,
            )
            for item in eligible
        ]
        scored = [
            (score, item)
            for score, item in scored
            if score[0] or score[1] or score[2] >= 2
        ]
        return (
            max(
                scored,
                key=lambda row: (
                    row[0],
                    _safe_round(row[1].get("round"), 0),
                ),
            )[1]
            if scored
            else None
        )
    if cue_kind == "ordinal_reference":
        eligible = [
            item
            for item in candidates
            if item.get("kind")
            in {"learner_named_alternatives", "teacher_named_alternatives"}
        ]
        return (
            max(
                eligible,
                key=lambda item: (
                    _safe_round(item.get("round"), 0),
                    item.get("speaker") == "learner",
                ),
            )
            if eligible
            else None
        )
    if cue_kind == "earliest_learner_instruction":
        learner_items = [
            item
            for item in candidates
            if item.get("kind") == "learner_instruction"
            and item.get("speaker") == "learner"
        ]
        fallback_items = [
            item for item in candidates if item.get("kind") == "learner_instruction"
        ]
        eligible = learner_items or fallback_items
        return (
            min(eligible, key=lambda item: _safe_round(item.get("round"), 0))
            if eligible
            else None
        )
    if cue_kind == "earlier_unresolved_question":
        learner_questions = [
            item
            for item in candidates
            if item.get("kind") == "unresolved_learner_question"
        ]
        teacher_questions = [
            item for item in candidates if item.get("kind") == "teacher_question"
        ]
        eligible = learner_questions or teacher_questions
        return (
            min(eligible, key=lambda item: _safe_round(item.get("round"), 0))
            if eligible
            else None
        )
    if cue_kind == "prior_agreement_or_agenda":
        if _continuity_completion_status_requested(learner_response):
            # A request to be reminded what is unfinished should resolve to an
            # evidence-linked open question or pending teacher commitment
            # before falling back to a learner's old agenda/preference.  The
            # latter describes how to teach, but not what remains to finish.
            status_candidates = [
                item
                for item in candidates
                if item.get("kind")
                in {"unresolved_learner_question", "teacher_commitment"}
            ]
            if status_candidates:
                return max(
                    status_candidates,
                    key=lambda item: (
                        _safe_round(item.get("round"), 0),
                        item.get("kind") == "unresolved_learner_question",
                    ),
                )
        eligible = [
            item
            for item in candidates
            if item.get("kind") in {"learner_future_agenda", "teacher_commitment"}
        ]
        return (
            max(
                eligible,
                key=lambda item: (
                    _safe_round(item.get("round"), 0),
                    item.get("kind") == "learner_future_agenda",
                ),
            )
            if eligible
            else None
        )
    if cue_kind == "recent_repetition_complaint":
        eligible = [
            item
            for item in candidates
            if item.get("kind") == "learner_difficulty_signal"
        ]
        return (
            max(eligible, key=lambda item: _safe_round(item.get("round"), 0))
            if eligible
            else None
        )
    return None


def _continuity_recall_layer(
    session: Mapping[str, Any],
    learner_response: str | None,
    *,
    current_response_id: str | None,
    content_limit: int,
    evidence_excerpt_limit: int,
    evidence: list[dict[str, Any]],
) -> dict[str, Any] | None:
    if learner_response is None or current_response_id is None:
        return None
    cue_kind = _continuity_cue_kind(learner_response)
    if cue_kind is None:
        return None
    candidates = [
        *_current_action_continuity_candidates(session),
        *_history_continuity_candidates(session),
        *_memory_continuity_candidates(session),
    ]
    target = _select_continuity_target(cue_kind, candidates, learner_response)
    cue_excerpt = _bounded_text(learner_response, min(240, content_limit))
    if target is None:
        return {
            "schema": CONTINUITY_RECALL_SCHEMA,
            "cue_kind": cue_kind,
            "cue_excerpt": cue_excerpt,
            "cue_evidence_refs": [current_response_id],
            "status": "unresolved_no_matching_evidence",
            "target": None,
            "instruction": (
                "不得声称记得不存在的方案、约定或问题；明确说明没有找到匹配记录，"
                "并请学生用一句话重述所指内容。"
            ),
            "selection_policy": "fail_closed_when_no_evidence_linked_target",
            "must_not_invent": True,
        }

    excerpt = _bounded_text(target.get("excerpt", ""), min(400, content_limit))
    refs = [str(ref) for ref in target.get("evidence_refs", []) if str(ref)]
    round_number = _safe_round(target.get("round"), 0)
    for evidence_id in refs:
        evidence.append(
            _evidence_record(
                evidence_id,
                source=str(target.get("source", "session_history")),
                field=str(target.get("field", "utterance")),
                round_number=round_number,
                excerpt=_bounded_text(excerpt, evidence_excerpt_limit),
            )
        )
    instruction = (
        "学生已明确表达过不会或不懂；先承认刚才重复了同一要求，"
        "不得再次索取同一答案。改为教师先示范、换一种表征或给出更低负荷的"
        "识别步骤，再只检查一个更小的新点。"
        if cue_kind == "recent_repetition_complaint"
        else (
            "先按目标证据消解当前指代，再生成本轮动作；只使用所引证的原话，"
            "不得补写未命名的方案、约定、偏好或问题。"
        )
    )
    return {
        "schema": CONTINUITY_RECALL_SCHEMA,
        "cue_kind": cue_kind,
        "cue_excerpt": cue_excerpt,
        "cue_evidence_refs": [current_response_id],
        "status": "resolved_evidence_linked",
        "target": {
            "kind": str(target.get("kind", "")),
            "speaker": str(target.get("speaker", "")),
            "source_round": round_number,
            "excerpt": excerpt,
            "evidence_refs": refs,
        },
        "instruction": instruction,
        "selection_policy": "cue_specific_evidence_linked_target",
        "must_not_invent": True,
    }


def _redact_value(value: str, counts: Counter[str]) -> str:
    redacted, findings = redact_remote_text(value)
    counts.update(item["kind"] for item in findings)
    return redacted


def _redacted_event(event: dict[str, Any], counts: Counter[str]) -> dict[str, Any]:
    result = deepcopy(event)
    for field in (
        "focus_dimension",
        "skill_id",
        "teacher_message",
        "learner_response",
        "signal",
        "signal_source",
    ):
        value = result.get(field)
        if isinstance(value, str):
            result[field] = _redact_value(value, counts)
    result["knowledge_components"] = [
        _redact_value(str(item), counts)
        for item in result.get("knowledge_components", [])
    ]
    return result


def _history_summary(events: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rounds = [item["round"] for item in events if isinstance(item.get("round"), int)]
    signals = Counter(item.get("signal") or "not_recorded" for item in events)
    focuses = Counter(item.get("focus_dimension") or "not_recorded" for item in events)
    skills = Counter(item.get("skill_id") or "not_recorded" for item in events)
    sources = Counter(item.get("signal_source") or "not_recorded" for item in events)
    return {
        "turn_count": len(events),
        "round_range": [min(rounds), max(rounds)] if rounds else None,
        "signal_counts": dict(sorted(signals.items())),
        "focus_counts": dict(sorted(focuses.items())),
        "skill_counts": dict(sorted(skills.items())),
        "signal_source_counts": dict(sorted(sources.items())),
    }


def _serialized_length(value: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    )


def _truncate_text(value: str, limit: int) -> tuple[str, bool]:
    if len(value) <= limit:
        return value, False
    if limit <= len(_TEXT_TRUNCATION_MARKER):
        return _TEXT_TRUNCATION_MARKER[:limit], True
    return value[: limit - len(_TEXT_TRUNCATION_MARKER)] + _TEXT_TRUNCATION_MARKER, True


def _truncate_current_response(value: str, limit: int) -> tuple[str, bool]:
    """Preserve both the setup and final conclusion of the current answer."""

    if len(value) <= limit:
        return value, False
    if limit <= len(_MIDDLE_TRUNCATION_MARKER):
        return _MIDDLE_TRUNCATION_MARKER[:limit], True
    remaining = limit - len(_MIDDLE_TRUNCATION_MARKER)
    head = max(1, (remaining * 3) // 5)
    tail = remaining - head
    return (
        value[:head] + _MIDDLE_TRUNCATION_MARKER + (value[-tail:] if tail else ""),
        True,
    )


def _rebuild_context(
    *,
    learner_response: str,
    selected: Sequence[dict[str, Any]],
    omitted: Sequence[dict[str, Any]],
    current_focus: str | None,
    current_kcs: Sequence[str],
    finding_counts: Counter[str],
    history_count: int,
    max_recent_turns: int,
    max_chars: int,
    response_limit: int,
    text_limit: int,
    truncated: bool,
) -> dict[str, Any]:
    response, response_cut = _truncate_current_response(
        learner_response, response_limit
    )
    turns: list[dict[str, Any]] = []
    any_cut = response_cut
    for item in selected:
        row = deepcopy(item)
        for field in ("teacher_message", "learner_response"):
            row[field], cut = _truncate_text(str(row.get(field, "")), text_limit)
            any_cut = any_cut or cut
        turns.append(row)
    return {
        "schema": CONTEXT_SCHEMA,
        "current_focus": current_focus,
        "current_knowledge_components": list(current_kcs),
        "current_learner_response": response,
        "recent_turns": turns,
        "earlier_summary": _history_summary(omitted),
        "privacy": {
            "remote_text_redacted": bool(finding_counts),
            "finding_counts": dict(sorted(finding_counts.items())),
            "original_values_retained": False,
            "known_pattern_identifiers_redacted": bool(finding_counts),
            "raw_identity_fields_sent": "not_established",
            "residual_identity_risk": True,
        },
        "selection": {
            "priority": "query_round_kc_lexical_then_active_focus_recency",
            "history_turn_count": history_count,
            "selected_turn_count": len(turns),
            "max_recent_turns": max_recent_turns,
            "max_chars": max_chars,
            "truncated": truncated or any_cut,
        },
    }


def build_relevant_history(
    session: Mapping[str, Any],
    learner_response: str,
    max_recent_turns: int = 6,
    max_chars: int = 8000,
) -> dict[str, Any]:
    """Build bounded, redacted history for one remote learner assessment.

    Explicit round references and current-query matches rank first, followed
    by the active knowledge component, focus and recency.  At least the latest
    two turns are retained when capacity permits; all omitted turns are
    represented only by aggregate statistics.
    """

    if not isinstance(session, Mapping):
        raise TypeError("session must be a mapping")
    if not isinstance(learner_response, str):
        raise TypeError("learner_response must be a string")
    if isinstance(max_recent_turns, bool) or not isinstance(max_recent_turns, int):
        raise TypeError("max_recent_turns must be an integer")
    if not 0 <= max_recent_turns <= 100:
        raise ValueError("max_recent_turns must be in [0, 100]")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars < 640:
        raise ValueError("max_chars must be at least 640")

    raw_history = session.get("history", [])
    if not isinstance(raw_history, list):
        raise ValueError("session.history must be a list")
    events: list[dict[str, Any]] = []
    for index, value in enumerate(raw_history):
        if not isinstance(value, Mapping):
            raise ValueError(f"session.history[{index}] must be an object")
        events.append(_event_view(value, index + 1))

    current_focus, current_kcs = _current_focus_and_kcs(session)
    normalized_kcs = {item.casefold() for item in current_kcs}
    requested_round = _requested_round(learner_response)
    response_query_terms = _query_terms(learner_response)

    latest_indices: set[int] = set()
    if max_recent_turns:
        latest_count = min(2, max_recent_turns, len(events))
        latest_indices.update(range(len(events) - latest_count, len(events)))

    def relevance(index: int) -> tuple[int, int, int, int, int, int]:
        event = events[index]
        query_round_match, query_kc_matches, query_lexical_overlap = _event_query_score(
            event,
            learner_response,
            requested_round=requested_round,
            query_terms=response_query_terms,
        )
        event_kcs = {
            str(item).casefold() for item in event.get("knowledge_components", [])
        }
        kc_match = bool(normalized_kcs and normalized_kcs & event_kcs)
        focus_match = bool(
            current_focus and event.get("focus_dimension") == current_focus
        )
        return (
            query_round_match,
            query_kc_matches,
            query_lexical_overlap,
            int(kc_match),
            int(focus_match),
            index,
        )

    available = [index for index in range(len(events)) if index not in latest_indices]
    available.sort(key=relevance, reverse=True)
    remaining = max(0, max_recent_turns - len(latest_indices))
    selected_indices = latest_indices | set(available[:remaining])
    selected = [events[index] for index in sorted(selected_indices)]
    omitted = [
        events[index] for index in range(len(events)) if index not in selected_indices
    ]

    finding_counts: Counter[str] = Counter()
    redacted_response = _redact_value(learner_response, finding_counts)
    selected = [_redacted_event(item, finding_counts) for item in selected]
    omitted = [_redacted_event(item, finding_counts) for item in omitted]
    if current_focus:
        current_focus = _redact_value(current_focus, finding_counts)
    current_kcs = [_redact_value(item, finding_counts) for item in current_kcs]

    # The answer being assessed is more valuable than already-recorded turns.
    # Keep up to 2,400 characters (with a balanced head/tail cut) while first
    # shrinking and then dropping older history.  Only after the history layer
    # is exhausted may the current answer itself be reduced.
    response_limits = (2400, 1800, 1200, 800, 480, 280, 160, 80, 32, 0)
    text_limits = (1200, 800, 480, 280, 160, 80, 32, 0)
    context: dict[str, Any] | None = None
    for response_limit in response_limits:
        retained = list(selected)
        additionally_omitted = list(omitted)
        while True:
            for text_limit in text_limits:
                context = _rebuild_context(
                    learner_response=redacted_response,
                    selected=retained,
                    omitted=additionally_omitted,
                    current_focus=current_focus,
                    current_kcs=current_kcs,
                    finding_counts=finding_counts,
                    history_count=len(events),
                    max_recent_turns=max_recent_turns,
                    max_chars=max_chars,
                    response_limit=response_limit,
                    text_limit=text_limit,
                    truncated=(
                        response_limit < response_limits[0]
                        or text_limit < text_limits[0]
                        or len(retained) < len(selected)
                    ),
                )
                if _serialized_length(context) <= max_chars:
                    return context
            if not retained:
                break
            additionally_omitted.append(retained.pop(0))
            additionally_omitted.sort(key=lambda item: int(item.get("round", 0)))

    minimal = {
        "schema": CONTEXT_SCHEMA,
        "current_focus": current_focus,
        "current_knowledge_components": [],
        "current_learner_response": "",
        "recent_turns": [],
        "earlier_summary": {
            "turn_count": len(events),
            "signal_counts": _history_summary(events)["signal_counts"],
        },
        "privacy": {
            "remote_text_redacted": bool(finding_counts),
            "finding_counts": dict(sorted(finding_counts.items())),
            "known_pattern_identifiers_redacted": bool(finding_counts),
            "raw_identity_fields_sent": "not_established",
            "residual_identity_risk": True,
        },
        "selection": {
            "priority": "query_round_kc_lexical_then_active_focus_recency",
            "history_turn_count": len(events),
            "selected_turn_count": 0,
            "max_recent_turns": max_recent_turns,
            "max_chars": max_chars,
            "truncated": True,
        },
    }
    if _serialized_length(minimal) > max_chars:
        raise ValueError("max_chars is too small for the minimum context envelope")
    return minimal


def _bounded_text(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if not text or limit <= 0:
        return ""
    return _truncate_text(text, limit)[0]


def _bounded_current_response(value: Any, limit: int) -> str:
    text = str(value or "").strip()
    if not text or limit <= 0:
        return ""
    return _truncate_current_response(text, limit)[0]


def _redact_structure(value: Any, counts: Counter[str]) -> Any:
    """Deep-copy and redact every textual key and leaf in a JSON value.

    Teacher-authored mappings can place identifiers in field names as well as
    values.  Key collisions after redaction are resolved with an opaque,
    deterministic suffix so no source key is leaked and no sibling value is
    silently overwritten.
    """

    if isinstance(value, str):
        return _redact_value(value, counts)
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = _redact_value(str(key), counts)
            candidate = safe_key
            collision_index = 2
            while candidate in result:
                candidate = f"{safe_key}__{collision_index}"
                collision_index += 1
            result[candidate] = _redact_structure(item, counts)
        return result
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_redact_structure(item, counts) for item in value]
    return deepcopy(value)


def _safe_round(value: Any, fallback: int) -> int:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return fallback


def _evidence_record(
    evidence_id: str,
    *,
    source: str,
    field: str,
    round_number: int | None = None,
    excerpt: str = "",
    confidence: float | int | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "evidence_id": evidence_id,
        "source": source,
        "field": field,
        "round": round_number,
        "excerpt": excerpt,
    }
    if confidence is not None and not isinstance(confidence, bool):
        try:
            numeric = float(confidence)
        except (TypeError, ValueError):
            numeric = 0.0
        result["confidence"] = round(min(1.0, max(0.0, numeric)), 4)
    return result


def _teacher_profile_layer(
    profile: Mapping[str, Any],
    *,
    content_limit: int,
    prior_limit: int,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    initial_mastery = profile.get("initial_mastery", {})
    if not isinstance(initial_mastery, Mapping):
        initial_mastery = {}
    known = profile.get("known_misconceptions", [])
    if not isinstance(known, list):
        known = []
    known_view: list[dict[str, Any]] = []
    for index, item in enumerate(known[:8]):
        if not isinstance(item, Mapping):
            continue
        evidence_id = f"teacher_profile:known_misconception:{index}"
        description = _bounded_text(
            item.get("description", ""), min(300, content_limit)
        )
        known_view.append(
            {
                "tag": _bounded_text(item.get("tag", ""), 120),
                "description": description,
                "confidence": item.get("confidence"),
                "evidence_refs": [evidence_id],
            }
        )
        evidence.append(
            _evidence_record(
                evidence_id,
                source="teacher_provided_student_profile",
                field=f"known_misconceptions[{index}]",
                excerpt=description,
                confidence=item.get("confidence"),
            )
        )

    prior = profile.get("conversation_history", [])
    if not isinstance(prior, list):
        prior = []
    retained_start = max(0, len(prior) - prior_limit)
    prior_view: list[dict[str, Any]] = []
    for index, item in enumerate(prior[retained_start:], retained_start):
        if not isinstance(item, Mapping):
            continue
        evidence_id = f"teacher_profile:conversation_history:{index}"
        excerpt = _bounded_text(item.get("response", ""), min(400, content_limit))
        prior_view.append(
            {
                "index": index,
                "response_excerpt": excerpt,
                "signal": item.get("signal"),
                "focus_dimension": item.get("focus_dimension"),
                "evidence_refs": [evidence_id],
            }
        )
        evidence.append(
            _evidence_record(
                evidence_id,
                source="teacher_provided_student_profile",
                field=f"conversation_history[{index}]",
                excerpt=excerpt,
            )
        )

    background = profile.get("background_history", [])
    if not isinstance(background, list):
        background = []
    background_start = max(0, len(background) - prior_limit)
    background_view: list[dict[str, Any]] = []
    for index, item in enumerate(background[background_start:], background_start):
        excerpt = _bounded_text(item, min(400, content_limit))
        if not excerpt:
            continue
        evidence_id = f"teacher_profile:background_history:{index}"
        background_view.append(
            {
                "index": index,
                "response_excerpt": excerpt,
                "label_status": "unlabeled_background_not_state_evidence",
                "evidence_refs": [evidence_id],
            }
        )
        evidence.append(
            _evidence_record(
                evidence_id,
                source="teacher_provided_unlabeled_background",
                field=f"background_history[{index}]",
                excerpt=excerpt,
            )
        )

    for dimension, value in initial_mastery.items():
        evidence.append(
            _evidence_record(
                f"teacher_profile:initial_mastery:{dimension}",
                source="teacher_provided_student_profile",
                field=f"initial_mastery.{dimension}",
            )
        )
    preferences = profile.get("preferences", [])
    accessibility = profile.get("accessibility_needs", [])
    return {
        "learner_level": _bounded_text(profile.get("learner_level", ""), 120),
        "preferences": [
            _bounded_text(item, min(200, content_limit))
            for item in preferences[:8]
            if _bounded_text(item, min(200, content_limit))
        ]
        if isinstance(preferences, list)
        else [],
        "accessibility_needs": [
            _bounded_text(item, min(200, content_limit))
            for item in accessibility[:8]
            if _bounded_text(item, min(200, content_limit))
        ]
        if isinstance(accessibility, list)
        else [],
        "initial_mastery": dict(initial_mastery),
        "declared_known_misconceptions": known_view,
        "provided_prior_context": {
            "total_turn_count": len(prior) + len(background),
            "retained_turn_count": len(prior_view) + len(background_view),
            "turns": prior_view,
            "unlabeled_notes": background_view,
            "unlabeled_notes_may_update_state": False,
        },
        "source": "teacher_input_normalized_locally",
        "mutable_by_model": False,
    }


def _bounded_knowledge_spec(value: Any, *, content_limit: int) -> dict[str, Any]:
    """Project teacher-provided domain truth without claiming independent review."""

    if not isinstance(value, Mapping) or value.get("status") == "not_provided":
        return {
            "status": "not_provided",
            "claim_boundary": {
                "authoritative_for_runtime_grading": False,
                "model_memory_is_authoritative_when_absent": False,
            },
        }
    limit = max(48, min(content_limit, 320))

    def rows(
        field: str, text_fields: tuple[str, ...], maximum: int
    ) -> list[dict[str, Any]]:
        raw = value.get(field, [])
        if not isinstance(raw, list):
            return []
        result: list[dict[str, Any]] = []
        for item in raw[:maximum]:
            if not isinstance(item, Mapping):
                continue
            row = deepcopy(dict(item))
            for text_field in text_fields:
                if text_field in row:
                    row[text_field] = _bounded_text(row[text_field], limit)
            result.append(row)
        return result

    return {
        "schema": value.get("schema"),
        "status": value.get("status", "not_provided"),
        "canonical_claims": rows("canonical_claims", ("statement",), 8),
        "rubric_criteria": rows("rubric_criteria", ("description",), 8),
        "accepted_alternatives": rows("accepted_alternatives", ("description",), 6),
        "reference_steps": rows("reference_steps", ("description",), 8),
        "misconception_catalog": rows(
            "misconception_catalog", ("description", "corrective_principle"), 8
        ),
        "sources": rows("sources", ("title", "citation"), 6),
        "claim_boundary": deepcopy(value.get("claim_boundary", {})),
    }


def _minimal_knowledge_spec(
    value: Any,
    *,
    current_components: Sequence[str],
    content_limit: int,
) -> dict[str, Any] | None:
    """Keep only the grading evidence relevant to the active knowledge step.

    The normal layered context carries the complete teacher-authored knowledge
    specification.  The 6k emergency envelope cannot do that safely, so this
    projection keeps the active claims, criteria, equivalent method, reference
    step, and misconception cue without changing their teacher provenance.
    """

    if not isinstance(value, Mapping) or value.get("status") != "teacher_provided":
        return None
    limit = max(32, min(int(content_limit), 96))
    active = {str(item) for item in current_components if str(item).strip()}

    def mappings(field: str) -> list[Mapping[str, Any]]:
        rows = value.get(field, [])
        if not isinstance(rows, list):
            return []
        return [item for item in rows if isinstance(item, Mapping)]

    def components_for(item: Mapping[str, Any]) -> set[str]:
        raw = item.get("knowledge_components")
        if isinstance(raw, list):
            return {str(component) for component in raw}
        component = item.get("knowledge_component")
        return {str(component)} if component else set()

    def relevant(
        rows: list[Mapping[str, Any]], maximum: int
    ) -> list[Mapping[str, Any]]:
        selected = [item for item in rows if components_for(item) & active]
        if not selected:
            selected = [item for item in rows if not components_for(item)]
        if not selected and rows:
            selected = rows[:1]
        return selected[:maximum]

    claim_rows = relevant(mappings("canonical_claims"), 2)
    claim_ids = {
        str(item.get("claim_id")) for item in claim_rows if item.get("claim_id")
    }
    rubric_rows = relevant(mappings("rubric_criteria"), 2)
    step_rows = relevant(mappings("reference_steps"), 1)
    alternative_rows = [
        item
        for item in mappings("accepted_alternatives")
        if claim_ids
        & {
            str(claim_id)
            for claim_id in item.get("equivalent_claim_ids", [])
            if isinstance(item.get("equivalent_claim_ids"), list)
        }
    ][:1]
    misconception_rows = [
        item
        for item in mappings("misconception_catalog")
        if claim_ids
        & {
            str(claim_id)
            for claim_id in item.get("contradicts_claim_ids", [])
            if isinstance(item.get("contradicts_claim_ids"), list)
        }
    ][:1]
    boundary = value.get("claim_boundary", {})
    if not isinstance(boundary, Mapping):
        boundary = {}
    return {
        "status": "teacher_provided",
        "active_knowledge_components": list(active)[:4],
        "canonical_claims": [
            {
                "claim_id": item.get("claim_id"),
                "statement": _bounded_text(item.get("statement", ""), limit),
                "required": item.get("required") is True,
            }
            for item in claim_rows
        ],
        "rubric_criteria": [
            {
                "criterion_id": item.get("criterion_id"),
                "description": _bounded_text(item.get("description", ""), limit),
                "required": item.get("required") is True,
                "acceptable_evidence": [
                    _bounded_text(candidate, 48)
                    for candidate in item.get("acceptable_evidence", [])[:2]
                    if _bounded_text(candidate, 48)
                ]
                if isinstance(item.get("acceptable_evidence"), list)
                else [],
            }
            for item in rubric_rows
        ],
        "accepted_alternatives": [
            {"description": _bounded_text(item.get("description", ""), limit)}
            for item in alternative_rows
        ],
        "reference_steps": [
            {"description": _bounded_text(item.get("description", ""), limit)}
            for item in step_rows
        ],
        "misconception_catalog": [
            {
                "tag": _bounded_text(item.get("tag", ""), 48),
                "aliases": [
                    _bounded_text(alias, 48)
                    for alias in item.get("aliases", [])[:4]
                    if _bounded_text(alias, 48)
                ]
                if isinstance(item.get("aliases"), list)
                else [],
                "description": _bounded_text(item.get("description", ""), limit),
                "corrective_principle": _bounded_text(
                    item.get("corrective_principle", ""), limit
                ),
            }
            for item in misconception_rows
        ],
        "claim_boundary": {
            "authoritative_for_runtime_grading": boundary.get(
                "authoritative_for_runtime_grading"
            )
            is True,
            "independently_verified_by_system": False,
            "model_memory_is_authoritative_when_absent": False,
        },
    }


def _teaching_memory_layer(
    session: Mapping[str, Any],
    *,
    content_limit: int,
    evidence_excerpt_limit: int,
    evidence: list[dict[str, Any]],
    compact: bool = False,
) -> dict[str, Any] | None:
    """Return the bounded long-horizon memory and register every evidence ref."""

    memory = session.get("teaching_memory")
    if not isinstance(memory, Mapping):
        return None
    validate_teaching_memory(memory)
    projection = project_teaching_memory(memory, content_limit=content_limit)
    if compact:
        compact_groups: dict[str, list[dict[str, Any]]] = {}
        for group, text_field, maximum in (
            ("active_preferences", "statement", 3),
            ("unresolved_questions", "question", 3),
            ("pending_teacher_commitments", "statement", 2),
            ("active_referents", "description", 2),
        ):
            rows = projection.get(group, [])
            if not isinstance(rows, list):
                rows = []
            compact_rows: list[dict[str, Any]] = []
            for item in rows:
                if not isinstance(item, Mapping):
                    continue
                if (
                    group == "active_preferences"
                    and item.get("status") == "confirmed_teacher_profile"
                ):
                    # The immutable profile already carries these preferences.
                    continue
                compact_item = {
                    "memory_id": item.get("memory_id"),
                    text_field: _bounded_text(item.get(text_field, ""), content_limit),
                    "status": item.get("status"),
                    "evidence_refs": list(item.get("evidence_refs", []))[-2:],
                }
                if group == "active_preferences":
                    compact_item["kind"] = item.get("kind")
                answer_refs = item.get("answer_evidence_refs")
                if isinstance(answer_refs, list) and answer_refs:
                    compact_item["answer_evidence_refs"] = list(answer_refs)[-2:]
                compact_rows.append(compact_item)
            compact_groups[group] = compact_rows[-maximum:]
        projection = {
            "schema": TEACHING_MEMORY_PROJECTION_SCHEMA,
            "history_version": memory["history_version"],
            "compaction_generation": memory["compaction_generation"],
            "fixed_context_fingerprint": memory["fixed_context_fingerprint"],
            **compact_groups,
            "source": "deterministic_evidence_linked_rollout_projection",
            "narrative_inference_added": False,
            "model_may_mutate": False,
        }
        if int(memory["history_version"]) == 0 and not any(compact_groups.values()):
            return None
    groups = {
        "active_preferences": ("student_preference", "statement"),
        "unresolved_questions": ("learner_question", "question"),
        "pending_teacher_commitments": ("teacher_commitment", "statement"),
        "active_referents": ("teacher_named_alternatives", "description"),
    }
    for group, (source, text_field) in groups.items():
        rows = projection.get(group, [])
        if not isinstance(rows, list):
            continue
        for item in rows:
            if not isinstance(item, Mapping):
                continue
            excerpt = _bounded_text(item.get(text_field, ""), evidence_excerpt_limit)
            refs = item.get("evidence_refs", [])
            if not isinstance(refs, list):
                continue
            for evidence_id in refs:
                evidence_id = str(evidence_id)
                if not evidence_id:
                    continue
                round_match = re.search(r":r(\d+):", evidence_id)
                if round_match is None:
                    round_match = re.search(r"_r(\d+)_", str(item.get("memory_id", "")))
                round_number = int(round_match.group(1)) if round_match else 0
                evidence.append(
                    _evidence_record(
                        evidence_id,
                        source=source,
                        field=f"teaching_memory.{group}.{text_field}",
                        round_number=round_number,
                        excerpt=excerpt,
                    )
                )
            answer_refs = item.get("answer_evidence_refs", [])
            if isinstance(answer_refs, list):
                for evidence_id in answer_refs:
                    evidence_id = str(evidence_id)
                    if not evidence_id:
                        continue
                    round_match = re.search(r":r(\d+):", evidence_id)
                    evidence.append(
                        _evidence_record(
                            evidence_id,
                            source="teacher_or_learner_resolution_evidence",
                            field="teaching_memory.question_resolution",
                            round_number=(
                                int(round_match.group(1)) if round_match else 0
                            ),
                            excerpt="",
                        )
                    )
    return projection


def _bounded_question_contract(value: Any, *, content_limit: int) -> dict[str, Any]:
    """Keep the current grading contract useful without letting it own the budget."""

    if not isinstance(value, Mapping) or not value:
        return {}
    item_limit = max(24, min(content_limit, 120))
    criterion_limit = max(32, min(content_limit, 160))

    def strings(field: str, maximum_items: int, maximum_chars: int) -> list[str]:
        raw = value.get(field, [])
        if not isinstance(raw, list):
            return []
        return [
            bounded
            for item in raw[:maximum_items]
            if (bounded := _bounded_text(item, maximum_chars))
        ]

    result = {
        "answer_type": _bounded_text(value.get("answer_type", "open"), 32),
        "target_concepts": strings("target_concepts", 4, item_limit),
        "accepted_aliases": strings("accepted_aliases", 8, item_limit),
        "success_criteria": strings("success_criteria", 4, criterion_limit),
        "grading_scope": _bounded_text(
            value.get("grading_scope", "current_question_only"), 40
        ),
    }
    if content_limit <= 48:
        source_counts = {
            field: len(value.get(field, []))
            if isinstance(value.get(field), list)
            else 0
            for field in (
                "target_concepts",
                "accepted_aliases",
                "success_criteria",
            )
        }
        result["context_projection"] = {
            "text_truncated": any(
                len(str(source)) > len(str(projected))
                for field in source_counts
                for source, projected in zip(
                    value.get(field, []), result[field], strict=False
                )
            ),
            "source_item_counts": source_counts,
        }
    return result


def _current_plan_layer(
    session: Mapping[str, Any], *, content_limit: int
) -> dict[str, Any]:
    plan = session.get("goal_plan", {})
    if not isinstance(plan, Mapping):
        plan = {}
    steps = plan.get("intermediate_objectives", [])
    if not isinstance(steps, list):
        steps = []
    active_id = plan.get("active_step")
    active = next(
        (
            item
            for item in steps
            if isinstance(item, Mapping) and item.get("step_id") == active_id
        ),
        None,
    )
    active_view: dict[str, Any] | None = None
    if isinstance(active, Mapping):
        verification = active.get("verification", {})
        if not isinstance(verification, Mapping):
            verification = {}
        active_view = {
            "step_id": active.get("step_id"),
            "dimension": active.get("dimension"),
            "objective": _bounded_text(active.get("objective", ""), content_limit),
            "success_criterion": _bounded_text(
                verification.get("success_criterion", ""), content_limit
            ),
            "mastery_threshold": verification.get("mastery_threshold"),
            "status": active.get("status"),
        }
    action = session.get("current_action", {})
    if not isinstance(action, Mapping):
        action = {}
    primary = action.get("primary_skill", {})
    teacher_action = action.get("teacher_action", {})
    supporting = action.get("supporting_skills", [])
    if not isinstance(primary, Mapping):
        primary = {}
    if not isinstance(teacher_action, Mapping):
        teacher_action = {}
    if not isinstance(supporting, list):
        supporting = []
    return {
        "plan_schema": plan.get("schema"),
        "plan_status": plan.get("status"),
        "progress": deepcopy(plan.get("progress", {})),
        "active_step": active_view,
        "step_statuses": [
            {
                "step_id": item.get("step_id"),
                "dimension": item.get("dimension"),
                "status": item.get("status"),
            }
            for item in steps
            if isinstance(item, Mapping)
        ],
        "current_action": {
            "action_id": action.get("action_id"),
            "primary_skill_id": primary.get("skill_id"),
            "primary_skill_name": primary.get("name"),
            "focus_dimension": primary.get("focus_dimension"),
            "knowledge_components": deepcopy(
                primary.get(
                    "knowledge_components",
                    action.get("knowledge_components", []),
                )
            ),
            "supporting_skill_ids": [
                item.get("skill_id")
                for item in supporting
                if isinstance(item, Mapping) and item.get("skill_id")
            ],
            "target_misconception_tags": [
                _bounded_text(item, 120)
                for item in action.get("target_misconception_tags", [])[:4]
                if _bounded_text(item, 120)
            ]
            if isinstance(action.get("target_misconception_tags"), list)
            else [],
            "selection_reason": _bounded_text(
                action.get("selection_reason", ""), content_limit
            ),
            "teacher_message": _bounded_text(
                teacher_action.get("message", ""), content_limit
            ),
            "expected_signal": _bounded_text(
                teacher_action.get("expected_signal", ""), content_limit
            ),
            "question_id": teacher_action.get("question_id"),
            "question_contract": _bounded_question_contract(
                teacher_action.get("question_contract", {}),
                content_limit=content_limit,
            ),
        },
        "source": "deterministic_session_plan_and_last_validated_action",
    }


def _latest_unresolved_events(events: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return unresolved observations without inventing a semantic diagnosis."""

    unresolved: list[dict[str, Any]] = []
    for event in events:
        focus = str(event.get("focus_dimension") or "unspecified")
        event_kcs = {
            str(item).casefold()
            for item in event.get("knowledge_components", [])
            if str(item).strip()
        }

        def same_issue(candidate: Mapping[str, Any]) -> bool:
            candidate_focus = str(candidate.get("focus_dimension") or "unspecified")
            candidate_kcs = {
                str(item).casefold()
                for item in candidate.get("knowledge_components", [])
                if str(item).strip()
            }
            if event_kcs and candidate_kcs:
                return bool(event_kcs & candidate_kcs)
            return candidate_focus == focus

        signal = event.get("signal")
        if signal == "correct":
            unresolved = [item for item in unresolved if not same_issue(item)]
        elif signal in {"partial", "misconception", "confused", "no_response"}:
            unresolved = [item for item in unresolved if not same_issue(item)]
            unresolved.append(event)
    return sorted(
        unresolved,
        key=lambda item: _safe_round(item.get("round"), 0),
    )[-4:]


def _knowledge_state_layer(
    session: Mapping[str, Any],
    *,
    events: Sequence[dict[str, Any]],
    content_limit: int,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    state = session.get("student_state", {})
    if not isinstance(state, Mapping):
        state = {}
    mastery = state.get("knowledge_mastery", {})
    if not isinstance(mastery, Mapping):
        mastery = {}
    thresholds = session.get("goal", {}).get("success_thresholds", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    calibrated_focus: dict[str, Any] | None = None
    raw_estimate = state.get("student_model_projection") or state.get("student_model")
    if isinstance(raw_estimate, Mapping):
        try:
            projected = project_student_model(raw_estimate)
            if int(projected.get("overall", {}).get("evidence_count", 0)) > 0:
                calibrated_focus = {
                    "dimension": projected.get("recommended_focus", {}).get(
                        "dimension"
                    ),
                    "confidence": projected.get("recommended_focus", {}).get(
                        "confidence", 0.0
                    ),
                    "evidence_refs": list(
                        projected.get("recommended_focus", {}).get("evidence_refs", [])
                    )[:2],
                    "source": "deterministic_evidence_weighted_estimator",
                }
        except (TypeError, ValueError):
            calibrated_focus = None
    mastery_view: list[dict[str, Any]] = []
    for dimension, value in mastery.items():
        matching = [item for item in events if item.get("focus_dimension") == dimension]
        if matching:
            latest_match = matching[-1]
            latest_round = _safe_round(latest_match.get("round"), 0)
            ref = f"session_history:r{latest_round}:structured_signal"
            evidence.append(
                _evidence_record(
                    ref,
                    source=latest_match.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=latest_round,
                    confidence=latest_match.get("signal_confidence"),
                )
            )
        else:
            ref = f"teacher_profile:initial_mastery:{dimension}"
        mastery_view.append(
            {
                "dimension": dimension,
                "value": value,
                "success_threshold": thresholds.get(dimension),
                "evidence_refs": [ref],
                "interpretation": "deterministic_state_estimate_not_ground_truth",
            }
        )

    misconceptions_raw = state.get("misconceptions", [])
    if not isinstance(misconceptions_raw, list):
        misconceptions_raw = []
    misconceptions: list[dict[str, Any]] = []
    for index, item in enumerate(misconceptions_raw[:12]):
        if not isinstance(item, Mapping):
            continue
        last_round = _safe_round(item.get("last_observed_round"), 0)
        ref = (
            f"session_history:r{last_round}:structured_signal"
            if last_round > 0
            else f"teacher_profile:known_misconception:{index}"
        )
        if last_round > 0:
            matching_event = next(
                (
                    event
                    for event in reversed(events)
                    if _safe_round(event.get("round"), 0) == last_round
                ),
                {},
            )
            evidence.append(
                _evidence_record(
                    ref,
                    source=matching_event.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=last_round,
                    confidence=matching_event.get("signal_confidence"),
                )
            )
        misconceptions.append(
            {
                "tag": _bounded_text(item.get("tag", ""), 120),
                "description": _bounded_text(
                    item.get("description", ""), min(300, content_limit)
                ),
                "confidence": item.get("confidence"),
                "status": item.get("status"),
                "first_observed_round": item.get("first_observed_round"),
                "last_observed_round": item.get("last_observed_round"),
                "resolved_round": item.get("resolved_round"),
                "evidence_refs": [ref],
            }
        )

    unresolved: list[dict[str, Any]] = []
    for event in _latest_unresolved_events(events):
        round_number = _safe_round(event.get("round"), 0)
        response_id = f"session_history:r{round_number}:learner_response"
        signal_id = f"session_history:r{round_number}:structured_signal"
        excerpt = _bounded_text(
            event.get("learner_response", ""), min(240, content_limit)
        )
        unresolved.append(
            {
                "issue_kind": "unresolved_observed_response",
                "focus_dimension": event.get("focus_dimension"),
                "knowledge_components": deepcopy(event.get("knowledge_components", [])),
                "observed_signal": event.get("signal"),
                "observation_source": event.get("signal_source") or "not_recorded",
                "learner_response_excerpt": excerpt,
                "evidence_refs": [response_id, signal_id],
            }
        )
        evidence.extend(
            [
                _evidence_record(
                    response_id,
                    source="session_history",
                    field="learner_response",
                    round_number=round_number,
                    excerpt=excerpt,
                ),
                _evidence_record(
                    signal_id,
                    source=event.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=round_number,
                    confidence=event.get("signal_confidence"),
                ),
            ]
        )
    result = {
        "concept_mastery": mastery_view,
        "misconceptions": misconceptions,
        "unresolved_issues": unresolved,
        "current_understanding_signal": deepcopy(state.get("understanding_signal", {})),
        "next_focus": deepcopy(state.get("next_focus", {})),
        "assessment_evidence": deepcopy(state.get("assessment_evidence", {})),
        "source": "deterministic_state_machine_over_labeled_observations",
    }
    if calibrated_focus is not None:
        result["calibrated_focus"] = calibrated_focus
    return result


def _candidate_memory_layer(
    profile: Mapping[str, Any],
    *,
    observation_limit: int,
    content_limit: int,
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    observations = profile.get("adaptive_observations", [])
    if not isinstance(observations, list):
        observations = []
    retained = observations[-observation_limit:] if observation_limit else []
    views: list[dict[str, Any]] = []
    for item in retained:
        if not isinstance(item, Mapping):
            continue
        round_number = _safe_round(item.get("round"), 0)
        candidate = item.get("candidate", {})
        evidence_value = item.get("evidence", {})
        if not isinstance(candidate, Mapping) or not isinstance(
            evidence_value, Mapping
        ):
            continue
        evidence_id = f"adaptive_profile:r{round_number}:validated_diagnosis"
        excerpt = _bounded_text(
            evidence_value.get("excerpt", ""), min(240, content_limit)
        )
        views.append(
            {
                "round": round_number,
                "status": item.get("status"),
                "candidate": deepcopy(dict(candidate)),
                "confidence": evidence_value.get("confidence"),
                "assessment_source": evidence_value.get("assessment_source"),
                "model_raw_signal": evidence_value.get("model_raw_signal"),
                "final_signal": evidence_value.get("final_signal"),
                "normalization_reasons": deepcopy(
                    evidence_value.get("normalization_reasons", [])
                ),
                "needs_human_review": evidence_value.get("needs_human_review"),
                "evidence_refs": [evidence_id],
            }
        )
        evidence.append(
            _evidence_record(
                evidence_id,
                source=str(
                    evidence_value.get("assessment_source")
                    or item.get("source")
                    or "candidate_profile"
                ),
                field="adaptive_observation.evidence",
                round_number=round_number,
                excerpt=excerpt,
                confidence=evidence_value.get("confidence"),
            )
        )
    return {
        "status": "candidate_unconfirmed",
        "summary": deepcopy(profile.get("adaptive_summary", {})),
        "retained_for_context": len(views),
        "observations": views,
        "may_override_teacher_profile": False,
        "eligible_sources": ["deepseek_v4_flash_validated_diagnosis"],
    }


def _deduplicate_evidence(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in records:
        evidence_id = str(item.get("evidence_id", ""))
        if not evidence_id or evidence_id in seen:
            continue
        seen.add(evidence_id)
        result.append(item)
    return result


def _teaching_checkpoints(
    events: Sequence[dict[str, Any]],
    *,
    selected_rounds: set[int],
    content_limit: int,
    evidence_excerpt_limit: int,
    evidence: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Extract a bounded set of audit-friendly facts from omitted turns.

    These checkpoints are deliberately narrower than a narrative summary.  A
    checkpoint is emitted only from an explicit learner/teacher utterance or a
    recorded structured signal.  Unknown resolution/completion states remain
    unknown instead of being inferred by a language model.
    """

    omitted = [
        event
        for event in events
        if _safe_round(event.get("round"), 0) not in selected_rounds
    ]
    if not omitted:
        return []

    def component_keys(event: Mapping[str, Any]) -> list[str]:
        components = [
            str(item).strip()
            for item in event.get("knowledge_components", [])
            if str(item).strip()
        ]
        if components:
            return list(dict.fromkeys(components))
        focus = str(event.get("focus_dimension") or "unspecified").strip()
        return [f"focus:{focus}"]

    unresolved_by_component: dict[str, dict[str, Any]] = {}
    verified_prerequisite_by_component: dict[str, dict[str, Any]] = {}
    for event in events:
        signal = str(event.get("signal") or "not_recorded")
        for key in component_keys(event):
            if signal in {"partial", "confused", "misconception"}:
                unresolved_by_component[key] = event
            elif signal == "correct":
                unresolved_by_component.pop(key, None)
                if event.get("focus_dimension") == "prerequisite":
                    verified_prerequisite_by_component[key] = event

    candidates: list[tuple[int, int, dict[str, Any]]] = []

    def append_candidate(
        *,
        priority: int,
        kind: str,
        status: str,
        event: Mapping[str, Any],
        text_field: str,
        include_signal: bool,
        derivation: str,
    ) -> None:
        round_number = _safe_round(event.get("round"), 0)
        if round_number in selected_rounds:
            return
        if text_field == "teacher_message":
            evidence_id = f"session_history:r{round_number}:teacher_action"
            ledger_field = "action.teacher_action.message"
        else:
            evidence_id = f"session_history:r{round_number}:learner_response"
            ledger_field = "learner_response"
        excerpt = _bounded_text(event.get(text_field, ""), min(140, content_limit))
        refs = [evidence_id]
        evidence.append(
            _evidence_record(
                evidence_id,
                source="session_history",
                field=ledger_field,
                round_number=round_number,
                excerpt=_bounded_text(excerpt, evidence_excerpt_limit),
            )
        )
        if include_signal:
            signal_id = f"session_history:r{round_number}:structured_signal"
            refs.append(signal_id)
            evidence.append(
                _evidence_record(
                    signal_id,
                    source=event.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=round_number,
                    confidence=event.get("signal_confidence"),
                )
            )
        candidates.append(
            (
                priority,
                round_number,
                {
                    "checkpoint_version": 1,
                    "kind": kind,
                    "status": status,
                    "round": round_number,
                    "focus_dimension": event.get("focus_dimension"),
                    "knowledge_components": deepcopy(
                        event.get("knowledge_components", [])
                    ),
                    "signal": event.get("signal") if include_signal else None,
                    "excerpt": excerpt,
                    "evidence_refs": refs,
                    "derivation": derivation,
                },
            )
        )

    unresolved_events = sorted(
        {
            _safe_round(event.get("round"), 0): event
            for event in unresolved_by_component.values()
            if _safe_round(event.get("round"), 0) not in selected_rounds
        }.values(),
        key=lambda event: _safe_round(event.get("round"), 0),
        reverse=True,
    )[:2]
    for event in unresolved_events:
        append_candidate(
            priority=0,
            kind="unresolved_learning_signal",
            status="not_cleared_by_later_correct_signal",
            event=event,
            text_field="learner_response",
            include_signal=True,
            derivation="structured_signal_and_later_same_component_scan",
        )

    explicit_kinds = (
        (
            1,
            "explicit_learner_question",
            "resolution_not_established",
            "learner_response",
            _EXPLICIT_LEARNER_QUESTION_RE,
            "explicit_question_pattern",
        ),
        (
            2,
            "explicit_learner_preference_or_constraint",
            "candidate_until_confirmed",
            "learner_response",
            _EXPLICIT_LEARNER_REQUEST_RE,
            "explicit_request_or_constraint_pattern",
        ),
        (
            4,
            "teacher_next_step_statement",
            "completion_not_established",
            "teacher_message",
            _TEACHER_NEXT_STEP_RE,
            "explicit_teacher_next_step_pattern",
        ),
    )
    for priority, kind, status, text_field, pattern, derivation in explicit_kinds:
        latest = next(
            (
                event
                for event in reversed(omitted)
                if (
                    _is_explicit_learner_question_text(event.get(text_field, ""))
                    if kind == "explicit_learner_question"
                    else bool(pattern.search(str(event.get(text_field, ""))))
                )
            ),
            None,
        )
        if latest is not None:
            append_candidate(
                priority=priority,
                kind=kind,
                status=status,
                event=latest,
                text_field=text_field,
                include_signal=False,
                derivation=derivation,
            )

    verified_events = sorted(
        {
            _safe_round(event.get("round"), 0): event
            for event in verified_prerequisite_by_component.values()
            if _safe_round(event.get("round"), 0) not in selected_rounds
        }.values(),
        key=lambda event: _safe_round(event.get("round"), 0),
        reverse=True,
    )[:1]
    for event in verified_events:
        append_candidate(
            priority=3,
            kind="verified_prerequisite",
            status="verified_by_recorded_correct_signal",
            event=event,
            text_field="learner_response",
            include_signal=True,
            derivation="recorded_correct_signal_on_prerequisite_focus",
        )

    selected = sorted(candidates, key=lambda item: (item[0], -item[1]))[:6]
    selected.sort(key=lambda item: (item[1], item[0]))
    result: list[dict[str, Any]] = []
    for index, (_priority, round_number, checkpoint) in enumerate(selected, 1):
        checkpoint["checkpoint_id"] = (
            f"omitted:r{round_number}:{checkpoint['kind']}:{index}"
        )
        result.append(checkpoint)
    return result


def _set_serialized_size(context: dict[str, Any]) -> int:
    """Reach the stable character count after writing the count itself."""

    for _ in range(8):
        measured = _serialized_length(context)
        if context["budget"].get("serialized_chars") == measured:
            return measured
        context["budget"]["serialized_chars"] = measured
    return _serialized_length(context)


def _layered_context_candidate(
    session: Mapping[str, Any],
    learner_response: str | None,
    *,
    max_chars: int,
    max_recent_turns: int,
    prior_limit: int,
    adaptive_limit: int,
    content_limit: int,
    evidence_excerpt_limit: int,
    degraded: bool,
) -> dict[str, Any]:
    evidence: list[dict[str, Any]] = []
    history_budget = max(640, min(8_000, max_chars // 2))
    relevant = build_relevant_history(
        session,
        learner_response or "",
        max_recent_turns=max_recent_turns,
        max_chars=history_budget,
    )
    raw_history = session.get("history", [])
    events = (
        [
            _event_view(item, index + 1)
            for index, item in enumerate(raw_history)
            if isinstance(item, Mapping)
        ]
        if isinstance(raw_history, list)
        else []
    )

    recent_turns: list[dict[str, Any]] = []
    for item in relevant["recent_turns"]:
        round_number = _safe_round(item.get("round"), 0)
        teacher_id = f"session_history:r{round_number}:teacher_action"
        response_id = f"session_history:r{round_number}:learner_response"
        signal_id = f"session_history:r{round_number}:structured_signal"
        row = deepcopy(item)
        row["evidence_refs"] = [teacher_id, response_id, signal_id]
        recent_turns.append(row)
        evidence.extend(
            [
                _evidence_record(
                    teacher_id,
                    source="session_history",
                    field="action.teacher_action.message",
                    round_number=round_number,
                    excerpt=_bounded_text(
                        item.get("teacher_message", ""), evidence_excerpt_limit
                    ),
                ),
                _evidence_record(
                    response_id,
                    source="session_history",
                    field="learner_response",
                    round_number=round_number,
                    excerpt=_bounded_text(
                        item.get("learner_response", ""), evidence_excerpt_limit
                    ),
                ),
                _evidence_record(
                    signal_id,
                    source=item.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=round_number,
                    confidence=item.get("signal_confidence"),
                ),
            ]
        )

    current_response_id: str | None = None
    if learner_response is not None:
        current_response_id = f"current_response:r{int(session.get('round', 0)) + 1}"
        evidence.append(
            _evidence_record(
                current_response_id,
                source="current_learner_input",
                field="learner_response",
                round_number=int(session.get("round", 0)) + 1,
                # The actual current answer lives in working_memory exactly
                # once.  The ledger is a pointer, not a second text copy.
                excerpt="",
            )
        )

    selected_rounds = {_safe_round(item.get("round"), 0) for item in recent_turns}
    checkpoint_by_focus: dict[str, dict[str, Any]] = {}
    for event in events:
        round_number = _safe_round(event.get("round"), 0)
        if round_number in selected_rounds:
            continue
        focus = str(event.get("focus_dimension") or "unspecified")
        checkpoint_by_focus[focus] = event
    focus_checkpoints: list[dict[str, Any]] = []
    checkpoint_items = sorted(
        checkpoint_by_focus.items(),
        key=lambda item: _safe_round(item[1].get("round"), 0),
    )[-4:]
    for focus, event in checkpoint_items:
        round_number = _safe_round(event.get("round"), 0)
        response_id = f"session_history:r{round_number}:learner_response"
        signal_id = f"session_history:r{round_number}:structured_signal"
        excerpt = _bounded_text(
            event.get("learner_response", ""), min(160, content_limit)
        )
        focus_checkpoints.append(
            {
                "focus_dimension": focus,
                "latest_omitted_round": round_number,
                "signal": event.get("signal"),
                "signal_source": event.get("signal_source") or "not_recorded",
                "skill_id": event.get("skill_id"),
                "knowledge_components": deepcopy(event.get("knowledge_components", [])),
                "learner_response_excerpt": excerpt,
                "evidence_refs": [response_id, signal_id],
            }
        )
        evidence.extend(
            [
                _evidence_record(
                    response_id,
                    source="session_history",
                    field="learner_response",
                    round_number=round_number,
                    excerpt=_bounded_text(excerpt, evidence_excerpt_limit),
                ),
                _evidence_record(
                    signal_id,
                    source=event.get("signal_source") or "not_recorded",
                    field="structured_signal",
                    round_number=round_number,
                    confidence=event.get("signal_confidence"),
                ),
            ]
        )

    teaching_checkpoints = _teaching_checkpoints(
        events,
        selected_rounds=selected_rounds,
        content_limit=content_limit,
        evidence_excerpt_limit=evidence_excerpt_limit,
        evidence=evidence,
    )

    profile = session.get("student_profile", {})
    if not isinstance(profile, Mapping):
        profile = {}
    goal = session.get("goal", {})
    if not isinstance(goal, Mapping):
        goal = {}
    lesson_state = session.get("lesson_state", {})
    if not isinstance(lesson_state, Mapping):
        lesson_state = {}
    exposure = lesson_state.get("knowledge_exposure", {})
    lesson_intent = str(
        lesson_state.get(
            "intent", goal.get("learning_intent", "legacy_diagnostic_first")
        )
    )
    lesson_phase = str(lesson_state.get("lesson_phase", ""))
    lesson_roles = list(lesson_required_primary_roles(session))
    lesson_contract = (
        {
            "intent": lesson_state.get(
                "intent", goal.get("learning_intent", "legacy_diagnostic_first")
            ),
            "current_lesson_phase": lesson_phase,
            "required_primary_roles": lesson_roles,
            "summary_required": bool(lesson_state.get("summary_required", False)),
            "summary_completed": bool(lesson_state.get("summary_completed", False)),
            "phase_sequence": [
                "orientation",
                "explanation",
                "worked_example",
                "guided_practice",
                "verification",
                "transfer",
            ],
            "knowledge_exposure": [
                {
                    "knowledge_component": _bounded_text(
                        component, min(240, content_limit)
                    ),
                    "status": record.get("status"),
                    "source": record.get("source"),
                }
                for component, record in list(exposure.items())[:12]
                if isinstance(record, Mapping)
            ]
            if isinstance(exposure, Mapping)
            else [],
            "unseen_independent_recall_prohibited": lesson_state.get("intent")
            == "teach_first",
            "exposure_is_mastery": False,
            "mutable_by_model": False,
        }
        if lesson_intent != "legacy_diagnostic_first"
        else None
    )
    teaching_memory = _teaching_memory_layer(
        session,
        content_limit=content_limit,
        evidence_excerpt_limit=evidence_excerpt_limit,
        evidence=evidence,
    )
    continuity_recall = _continuity_recall_layer(
        session,
        learner_response,
        current_response_id=current_response_id,
        content_limit=content_limit,
        evidence_excerpt_limit=evidence_excerpt_limit,
        evidence=evidence,
    )
    teaching_resources = []
    raw_resources = session.get("teaching_resources", [])
    if isinstance(raw_resources, list):
        for resource in raw_resources[:6]:
            if not isinstance(resource, Mapping):
                continue
            extracted_text = _bounded_text(
                resource.get("extracted_text", ""), content_limit
            )
            if not extracted_text:
                continue
            teaching_resources.append(
                {
                    "resource_id": resource.get("resource_id"),
                    "display_name": _bounded_text(
                        resource.get("display_name", ""), 120
                    ),
                    "resource_type": resource.get("resource_type"),
                    "content_sha256": resource.get("content_sha256"),
                    "page_count": resource.get("page_count"),
                    "truncated": resource.get("truncated") is True,
                    "needs_review": resource.get("needs_review") is not False,
                    "text": extracted_text,
                    "source": "teacher_imported_resource_local_extraction",
                    "mutable_by_model": False,
                }
            )
    fixed_context = {
        "teaching_goal": {
            "concept": _bounded_text(goal.get("concept", ""), min(240, content_limit)),
            "objective": _bounded_text(goal.get("objective", ""), content_limit),
            **(
                {
                    "learning_intent": lesson_intent,
                    "lesson_contract": lesson_contract,
                }
                if lesson_contract is not None
                else {}
            ),
            "knowledge_components": [
                _bounded_text(item, min(240, content_limit))
                for item in goal.get("knowledge_components", [])[:12]
                if _bounded_text(item, min(240, content_limit))
            ]
            if isinstance(goal.get("knowledge_components", []), list)
            else [],
            "success_thresholds": deepcopy(goal.get("success_thresholds", {})),
            "max_rounds": goal.get("max_rounds"),
            **(
                {"syllabus_ref": deepcopy(goal.get("syllabus_ref"))}
                if isinstance(goal.get("syllabus_ref"), Mapping)
                else {}
            ),
            "materials": {
                str(key): _bounded_text(value, content_limit)
                for key, value in list(goal.get("materials", {}).items())[:10]
            }
            if isinstance(goal.get("materials", {}), Mapping)
            else {},
            "teaching_resources": teaching_resources,
            **(
                {
                    "knowledge_spec": _bounded_knowledge_spec(
                        goal.get("knowledge_spec"), content_limit=content_limit
                    )
                }
                if isinstance(goal.get("knowledge_spec"), Mapping)
                and goal.get("knowledge_spec", {}).get("status") == "teacher_provided"
                else {}
            ),
            "source": "teacher_input_normalized_locally",
            "mutable_by_model": False,
        },
        "teacher_provided_student_profile": _teacher_profile_layer(
            profile,
            content_limit=content_limit,
            prior_limit=prior_limit,
            evidence=evidence,
        ),
    }
    knowledge_state = _knowledge_state_layer(
        session,
        events=events,
        content_limit=content_limit,
        evidence=evidence,
    )
    candidate_memory = _candidate_memory_layer(
        profile,
        observation_limit=adaptive_limit,
        content_limit=content_limit,
        evidence=evidence,
    )
    context: dict[str, Any] = {
        "schema": LAYERED_CONTEXT_SCHEMA,
        "snapshot": {
            "kind": "bounded_outbound_model_request_context",
            "session_round_before_request": int(session.get("round", 0)),
            "operation": (
                "initial_action" if learner_response is None else "assess_and_act"
            ),
            "current_response_present": learner_response is not None,
        },
        "fixed_context": fixed_context,
        "current_plan": _current_plan_layer(session, content_limit=content_limit),
        "working_memory": {
            "current_focus": relevant.get("current_focus"),
            "current_knowledge_components": deepcopy(
                relevant.get("current_knowledge_components", [])
            ),
            "current_learner_response": relevant.get("current_learner_response", ""),
            "current_response_evidence_id": current_response_id,
            "recent_turns": recent_turns,
        },
        "semantic_summary": {
            **deepcopy(relevant.get("earlier_summary", {})),
            "focus_checkpoints": focus_checkpoints,
            "teaching_checkpoints": teaching_checkpoints,
            **({"teaching_memory": teaching_memory} if teaching_memory else {}),
            **({"continuity_recall": continuity_recall} if continuity_recall else {}),
            "compression_method": (
                "deterministic_aggregate_and_extractive_checkpoints_no_model_generation"
            ),
            "source": "omitted_local_session_history",
            "narrative_inference_added": False,
        },
        "knowledge_state": knowledge_state,
        "candidate_long_term_memory": candidate_memory,
        "evidence_ledger": _deduplicate_evidence(evidence),
        "retrieval": {
            **deepcopy(relevant.get("selection", {})),
            "selected_rounds": [item.get("round") for item in recent_turns],
            "fixed_context_always_included": True,
            "current_plan_always_included": True,
            "omitted_turns_accounted_for_statistically": True,
            "teaching_checkpoints_are_selective_extracts": True,
            "candidate_memory_policy": "latest_validated_candidates_only",
        },
        "privacy": {
            **deepcopy(relevant.get("privacy", {})),
            "known_pattern_identifiers_redacted": bool(
                relevant.get("privacy", {}).get("known_pattern_identifiers_redacted")
            ),
            "raw_identity_fields_sent": "not_established",
            "residual_identity_risk": True,
            "original_values_retained_in_context": False,
        },
        "budget": {
            "policy": "hard_serialized_character_cap",
            "max_chars": max_chars,
            "serialized_chars": 0,
            "max_recent_turns": max_recent_turns,
            "retained_recent_turns": len(recent_turns),
            "retained_teacher_prior_turns": fixed_context[
                "teacher_provided_student_profile"
            ]["provided_prior_context"]["retained_turn_count"],
            "retained_candidate_observations": candidate_memory["retained_for_context"],
            "truncated": bool(degraded or relevant["selection"]["truncated"]),
        },
        "claim_boundary": {
            "model_generated_history_summary": False,
            "candidate_long_term_memory_is_confirmed": False,
            "fallback_observations_are_labeled_by_actual_source": True,
            "context_is_evidence_not_learner_ground_truth": True,
            "model_may_not_mutate_fixed_context": True,
            "omitted_turn_semantics_are_exhaustive": False,
        },
    }
    counts: Counter[str] = Counter()
    redacted = _redact_structure(context, counts)
    redacted["privacy"]["remote_text_redacted"] = bool(
        counts or relevant.get("privacy", {}).get("remote_text_redacted")
    )
    combined_counts = Counter(relevant.get("privacy", {}).get("finding_counts", {}))
    combined_counts.update(counts)
    redacted["privacy"]["finding_counts"] = dict(sorted(combined_counts.items()))
    redacted["privacy"]["known_pattern_identifiers_redacted"] = bool(combined_counts)
    _set_serialized_size(redacted)
    return redacted


def build_minimal_layered_context(
    session: Mapping[str, Any],
    learner_response: str | None,
    *,
    max_chars: int = MINIMUM_LAYERED_CONTEXT_CHARS,
) -> dict[str, Any]:
    """Build the smallest schema-valid context while retaining current input.

    This path is deterministic and intentionally extractive.  It is used as
    the final budget degradation profile and as the local safety envelope when
    construction of the richer retrieval snapshot fails unexpectedly.
    """

    if not isinstance(session, Mapping):
        raise TypeError("session must be a mapping")
    if learner_response is not None and not isinstance(learner_response, str):
        raise TypeError("learner_response must be a string or None")
    if (
        isinstance(max_chars, bool)
        or not isinstance(max_chars, int)
        or max_chars < MINIMUM_LAYERED_CONTEXT_CHARS
    ):
        raise ValueError(f"max_chars must be at least {MINIMUM_LAYERED_CONTEXT_CHARS}")

    goal = session.get("goal", {})
    if not isinstance(goal, Mapping):
        goal = {}
    lesson_state = session.get("lesson_state", {})
    if not isinstance(lesson_state, Mapping):
        lesson_state = {}
    profile = session.get("student_profile", {})
    if not isinstance(profile, Mapping):
        profile = {}
    state = session.get("student_state", {})
    if not isinstance(state, Mapping):
        state = {}
    plan = session.get("goal_plan", {})
    if not isinstance(plan, Mapping):
        plan = {}
    action = session.get("current_action", {})
    if not isinstance(action, Mapping):
        action = {}
    primary = action.get("primary_skill", {})
    if not isinstance(primary, Mapping):
        primary = {}
    teacher_action = action.get("teacher_action", {})
    if not isinstance(teacher_action, Mapping):
        teacher_action = {}
    mastery = state.get("knowledge_mastery", {})
    if not isinstance(mastery, Mapping):
        mastery = {}
    thresholds = goal.get("success_thresholds", {})
    if not isinstance(thresholds, Mapping):
        thresholds = {}
    history = session.get("history", [])
    history_count = len(history) if isinstance(history, list) else 0

    raw_misconceptions = state.get("misconceptions", [])
    if not isinstance(raw_misconceptions, list):
        raw_misconceptions = []
    retained_misconception = next(
        (
            item
            for item in reversed(raw_misconceptions)
            if isinstance(item, Mapping) and item.get("status") == "active"
        ),
        None,
    )
    adaptive = profile.get("adaptive_observations", [])
    if not isinstance(adaptive, list):
        adaptive = []
    adaptive_summary = profile.get("adaptive_summary", {})
    if not isinstance(adaptive_summary, Mapping):
        adaptive_summary = {}

    for response_limit in (
        4096,
        3072,
        2400,
        2048,
        1536,
        1024,
        768,
        512,
        256,
        128,
        64,
        48,
    ):
        evidence: list[dict[str, Any]] = []
        current_response_id: str | None = None
        current_response = _bounded_current_response(
            learner_response or "", response_limit
        )
        if learner_response is not None:
            current_response_id = (
                f"current_response:r{int(session.get('round', 0)) + 1}"
            )
            evidence.append(
                _evidence_record(
                    current_response_id,
                    source="current_learner_input",
                    field="learner_response",
                    round_number=int(session.get("round", 0)) + 1,
                    excerpt="",
                )
            )

        mastery_view: list[dict[str, Any]] = []
        for dimension, value in list(mastery.items())[:4]:
            mastery_view.append(
                {
                    "dimension": dimension,
                    "value": value,
                    "success_threshold": thresholds.get(dimension),
                }
            )

        misconception_view: list[dict[str, Any]] = []
        unresolved_view: list[dict[str, Any]] = []
        if isinstance(retained_misconception, Mapping):
            evidence_id = "student_state:latest_active_misconception"
            misconception_view.append(
                {
                    "tag": _bounded_text(retained_misconception.get("tag", ""), 64),
                    "description": _bounded_text(
                        retained_misconception.get("description", ""), 64
                    ),
                    "confidence": retained_misconception.get("confidence"),
                    "status": "active",
                    "evidence_refs": [evidence_id],
                }
            )
            unresolved_view.append(
                {
                    "issue_kind": "active_misconception",
                    "focus_dimension": "conceptual",
                    "observed_signal": "misconception",
                    "observation_source": "deterministic_session_state",
                    "evidence_refs": [evidence_id],
                }
            )
            evidence.append(
                _evidence_record(
                    evidence_id,
                    source="deterministic_session_state",
                    field="student_state.misconceptions.latest_active",
                    round_number=_safe_round(
                        retained_misconception.get("last_observed_round"), 0
                    ),
                )
            )

        signal = state.get("understanding_signal", {})
        if not isinstance(signal, Mapping):
            signal = {}
        assessment = state.get("assessment_evidence", {})
        if not isinstance(assessment, Mapping):
            assessment = {}
        next_focus = state.get("next_focus", {})
        if not isinstance(next_focus, Mapping):
            next_focus = {}
        provided_history = profile.get("conversation_history", [])
        background_history = profile.get("background_history", [])
        known = profile.get("known_misconceptions", [])
        preferences = profile.get("preferences", [])
        accessibility = profile.get("accessibility_needs", [])
        components = goal.get("knowledge_components", [])
        if not isinstance(components, list):
            components = []
        current_focus, raw_current_components = _current_focus_and_kcs(session)
        current_components = [
            _bounded_text(item, 24)
            for item in raw_current_components[:4]
            if _bounded_text(item, 24)
        ]
        if not current_components:
            current_components = [
                item for item in [_bounded_text(goal.get("concept", ""), 32)] if item
            ]

        teaching_memory = _teaching_memory_layer(
            session,
            content_limit=48,
            evidence_excerpt_limit=48,
            evidence=evidence,
            compact=True,
        )
        continuity_recall = _continuity_recall_layer(
            session,
            learner_response,
            current_response_id=current_response_id,
            content_limit=96,
            evidence_excerpt_limit=48,
            evidence=evidence,
        )
        minimal_knowledge_spec = _minimal_knowledge_spec(
            goal.get("knowledge_spec"),
            current_components=current_components,
            content_limit=56,
        )
        teaching_resources = []
        raw_resources = session.get("teaching_resources", [])
        if isinstance(raw_resources, list):
            for resource in raw_resources[:6]:
                if not isinstance(resource, Mapping):
                    continue
                extracted_text = _bounded_text(resource.get("extracted_text", ""), 96)
                if extracted_text:
                    teaching_resources.append(
                        {
                            "resource_id": resource.get("resource_id"),
                            "display_name": _bounded_text(
                                resource.get("display_name", ""), 48
                            ),
                            "resource_type": resource.get("resource_type"),
                            "content_sha256": resource.get("content_sha256"),
                            "needs_review": resource.get("needs_review") is not False,
                            "text": extracted_text,
                            "source": "teacher_imported_resource_local_extraction",
                            "mutable_by_model": False,
                        }
                    )

        context: dict[str, Any] = {
            "schema": LAYERED_CONTEXT_SCHEMA,
            "snapshot": {
                "kind": "bounded_outbound_model_request_context",
                "session_round_before_request": int(session.get("round", 0)),
                "operation": (
                    "initial_action" if learner_response is None else "assess_and_act"
                ),
                "current_response_present": learner_response is not None,
            },
            "fixed_context": {
                "teaching_goal": {
                    "concept": _bounded_text(goal.get("concept", ""), 96),
                    "objective": _bounded_text(goal.get("objective", ""), 64),
                    **(
                        {
                            "learning_intent": lesson_state.get("intent"),
                            "lesson_contract": {
                                "intent": lesson_state.get("intent"),
                                "current_lesson_phase": lesson_state.get(
                                    "lesson_phase"
                                ),
                                "required_primary_roles": list(
                                    lesson_required_primary_roles(session)
                                ),
                                "summary_required": bool(
                                    lesson_state.get("summary_required", False)
                                ),
                                "summary_completed": bool(
                                    lesson_state.get("summary_completed", False)
                                ),
                                "phase_sequence": [
                                    "orientation",
                                    "explanation",
                                    "worked_example",
                                    "guided_practice",
                                    "verification",
                                    "transfer",
                                ],
                                "unseen_independent_recall_prohibited": lesson_state.get(
                                    "intent"
                                )
                                == "teach_first",
                                "exposure_is_mastery": False,
                                "mutable_by_model": False,
                            },
                        }
                        if lesson_state.get("intent") != "legacy_diagnostic_first"
                        else {}
                    ),
                    "knowledge_components": current_components,
                    "total_knowledge_component_count": len(components),
                    "success_thresholds": dict(thresholds),
                    "max_rounds": goal.get("max_rounds"),
                    **(
                        {"syllabus_ref": deepcopy(goal.get("syllabus_ref"))}
                        if isinstance(goal.get("syllabus_ref"), Mapping)
                        else {}
                    ),
                    "teaching_resources": teaching_resources,
                    **(
                        {"knowledge_spec": minimal_knowledge_spec}
                        if minimal_knowledge_spec is not None
                        else {}
                    ),
                    "source": "teacher_input_normalized_locally",
                    "mutable_by_model": False,
                },
                "teacher_provided_student_profile": {
                    "learner_level": _bounded_text(
                        profile.get("learner_level", ""), 64
                    ),
                    "preferences": [
                        _bounded_text(item, 48)
                        for item in preferences[:3]
                        if _bounded_text(item, 48)
                    ]
                    if isinstance(preferences, list)
                    else [],
                    "accessibility_needs": [
                        _bounded_text(item, 48)
                        for item in accessibility[:2]
                        if _bounded_text(item, 48)
                    ]
                    if isinstance(accessibility, list)
                    else [],
                    "declared_known_misconception_count": (
                        len(known) if isinstance(known, list) else 0
                    ),
                    "provided_prior_turn_count": (
                        len(provided_history)
                        if isinstance(provided_history, list)
                        else 0
                    )
                    + (
                        len(background_history)
                        if isinstance(background_history, list)
                        else 0
                    ),
                    "source": "teacher_input_normalized_locally",
                    "mutable_by_model": False,
                },
            },
            "current_plan": {
                "plan_status": plan.get("status"),
                "active_step": plan.get("active_step"),
                "current_action": {
                    "action_id": action.get("action_id"),
                    "primary_skill_id": primary.get("skill_id"),
                    "focus_dimension": primary.get("focus_dimension"),
                    "knowledge_components": current_components,
                    "target_misconception_tags": [
                        _bounded_text(item, 120)
                        for item in action.get("target_misconception_tags", [])[:4]
                        if _bounded_text(item, 120)
                    ]
                    if isinstance(action.get("target_misconception_tags"), list)
                    else [],
                    "teacher_message": _bounded_text(
                        teacher_action.get("message", ""), 160
                    ),
                    "expected_signal": _bounded_text(
                        teacher_action.get("expected_signal", ""), 160
                    ),
                    "question_id": teacher_action.get("question_id"),
                    "question_contract": _bounded_question_contract(
                        teacher_action.get("question_contract", {}),
                        content_limit=24,
                    ),
                },
                "source": "deterministic_session_plan_and_last_validated_action",
            },
            "working_memory": {
                "current_focus": current_focus,
                "current_knowledge_components": current_components,
                "current_learner_response": current_response,
                "current_response_evidence_id": current_response_id,
                "recent_turns": [],
            },
            "semantic_summary": {
                "turn_count": history_count,
                "focus_checkpoints": [],
                "teaching_checkpoints": [],
                **({"teaching_memory": teaching_memory} if teaching_memory else {}),
                **(
                    {"continuity_recall": continuity_recall}
                    if continuity_recall
                    else {}
                ),
                "compression_method": (
                    "deterministic_aggregate_and_extractive_checkpoints_"
                    "no_model_generation"
                ),
                "source": "omitted_local_session_history",
                "narrative_inference_added": False,
            },
            "knowledge_state": {
                "concept_mastery": mastery_view,
                "misconceptions": misconception_view,
                "unresolved_issues": unresolved_view,
                "current_understanding_signal": {
                    "label": signal.get("label"),
                    "confidence": signal.get("confidence"),
                },
                "next_focus": {
                    "dimension": next_focus.get("dimension"),
                    "selected_skill_id": next_focus.get("selected_skill_id"),
                },
                "assessment_evidence": {
                    "needs_human_review": assessment.get("needs_human_review"),
                },
                "source": "deterministic_state_machine_over_labeled_observations",
            },
            "candidate_long_term_memory": {
                "status": "candidate_unconfirmed",
                "summary": {
                    "total_observation_count": adaptive_summary.get(
                        "total_observation_count", len(adaptive)
                    ),
                    "retained_observation_count": len(adaptive),
                    "needs_human_review": adaptive_summary.get(
                        "needs_human_review", False
                    ),
                },
                "retained_for_context": 0,
                "observations": [],
                "may_override_teacher_profile": False,
                "eligible_sources": ["deepseek_v4_flash_validated_diagnosis"],
            },
            "evidence_ledger": _deduplicate_evidence(evidence),
            "retrieval": {
                "priority": "minimum_safety_envelope_after_budget_degradation",
                "history_turn_count": history_count,
                "selected_turn_count": 0,
                "max_recent_turns": 0,
                "max_chars": max_chars,
                "truncated": True,
                "selected_rounds": [],
                "fixed_context_always_included": True,
                "current_plan_always_included": True,
                "omitted_turns_accounted_for_statistically": True,
                "teaching_checkpoints_are_selective_extracts": True,
                "candidate_memory_policy": "latest_validated_candidates_only",
            },
            "privacy": {
                "remote_text_redacted": False,
                "finding_counts": {},
                "known_pattern_identifiers_redacted": False,
                "raw_identity_fields_sent": "not_established",
                "residual_identity_risk": True,
                "original_values_retained_in_context": False,
            },
            "budget": {
                "policy": "hard_serialized_character_cap",
                "max_chars": max_chars,
                "serialized_chars": 0,
                "max_recent_turns": 0,
                "retained_recent_turns": 0,
                "retained_teacher_prior_turns": 0,
                "retained_candidate_observations": 0,
                "truncated": True,
            },
            "claim_boundary": {
                "model_generated_history_summary": False,
                "candidate_long_term_memory_is_confirmed": False,
                "fallback_observations_are_labeled_by_actual_source": True,
                "context_is_evidence_not_learner_ground_truth": True,
                "model_may_not_mutate_fixed_context": True,
                "omitted_turn_semantics_are_exhaustive": False,
            },
        }
        counts: Counter[str] = Counter()
        redacted = _redact_structure(context, counts)
        redacted["privacy"]["remote_text_redacted"] = bool(counts)
        redacted["privacy"]["finding_counts"] = dict(sorted(counts.items()))
        redacted["privacy"]["known_pattern_identifiers_redacted"] = bool(counts)
        _set_serialized_size(redacted)
        if _serialized_length(redacted) <= max_chars:
            validate_layered_context(redacted)
            return redacted
    raise ValueError("minimum layered context exceeds max_chars")


def build_layered_context(
    session: Mapping[str, Any],
    learner_response: str | None,
    *,
    max_chars: int = DEFAULT_LAYERED_CONTEXT_CHARS,
    max_recent_turns: int = 6,
) -> dict[str, Any]:
    """Build the single bounded, evidence-linked context sent to DeepSeek.

    The snapshot separates immutable teacher input, the current plan, recent
    working memory, deterministic older-history compression, explicit learner
    state, and unconfirmed long-term profile candidates.  No language model is
    called to create or repair memory.  When the full representation would
    exceed ``max_chars``, deterministic retention profiles are tried in order.
    """

    if not isinstance(session, Mapping):
        raise TypeError("session must be a mapping")
    if learner_response is not None and not isinstance(learner_response, str):
        raise TypeError("learner_response must be a string or None")
    if isinstance(max_chars, bool) or not isinstance(max_chars, int):
        raise TypeError("max_chars must be an integer")
    if max_chars < MINIMUM_LAYERED_CONTEXT_CHARS:
        raise ValueError(f"max_chars must be at least {MINIMUM_LAYERED_CONTEXT_CHARS}")
    if isinstance(max_recent_turns, bool) or not isinstance(max_recent_turns, int):
        raise TypeError("max_recent_turns must be an integer")
    if not 0 <= max_recent_turns <= 12:
        raise ValueError("max_recent_turns must be in [0, 12]")

    profiles = (
        (max_recent_turns, 3, 4, 800, 240),
        (min(max_recent_turns, 4), 2, 3, 480, 160),
        (min(max_recent_turns, 3), 1, 2, 300, 100),
        (min(max_recent_turns, 2), 0, 1, 180, 60),
        (min(max_recent_turns, 1), 0, 0, 100, 0),
        (0, 0, 0, 64, 0),
    )
    last: dict[str, Any] | None = None
    for index, (recent, prior, adaptive, content, excerpt) in enumerate(profiles):
        last = _layered_context_candidate(
            session,
            learner_response,
            max_chars=max_chars,
            max_recent_turns=recent,
            prior_limit=prior,
            adaptive_limit=adaptive,
            content_limit=content,
            evidence_excerpt_limit=excerpt,
            degraded=index > 0,
        )
        if _serialized_length(last) <= max_chars:
            validate_layered_context(last)
            return last
    return build_minimal_layered_context(
        session,
        learner_response,
        max_chars=max_chars,
    )


def _collect_evidence_refs(value: Any) -> list[str]:
    refs: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key == "evidence_refs" and isinstance(item, list):
                refs.extend(str(ref) for ref in item)
            elif key == "current_response_evidence_id" and isinstance(item, str):
                refs.append(item)
            else:
                refs.extend(_collect_evidence_refs(item))
    elif isinstance(value, list):
        for item in value:
            refs.extend(_collect_evidence_refs(item))
    return refs


def validate_layered_context(context: Mapping[str, Any]) -> None:
    """Fail closed on malformed budgets, provenance, or dangling evidence."""

    if (
        not isinstance(context, Mapping)
        or context.get("schema") != LAYERED_CONTEXT_SCHEMA
    ):
        raise ValueError(f"context schema must be {LAYERED_CONTEXT_SCHEMA}")
    required = {
        "snapshot",
        "fixed_context",
        "current_plan",
        "working_memory",
        "semantic_summary",
        "knowledge_state",
        "candidate_long_term_memory",
        "evidence_ledger",
        "retrieval",
        "privacy",
        "budget",
        "claim_boundary",
    }
    if not required <= set(context):
        raise ValueError("layered context sections are incomplete")
    snapshot = context.get("snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("kind") != "bounded_outbound_model_request_context"
        or snapshot.get("operation") not in {"initial_action", "assess_and_act"}
        or not isinstance(snapshot.get("current_response_present"), bool)
    ):
        raise ValueError("layered context snapshot metadata is invalid")
    fixed = context.get("fixed_context")
    if not isinstance(fixed, Mapping) or set(fixed) != {
        "teaching_goal",
        "teacher_provided_student_profile",
    }:
        raise ValueError("layered context fixed context is invalid")
    fixed_goal = fixed.get("teaching_goal", {})
    fixed_profile = fixed.get("teacher_provided_student_profile", {})
    if (
        not isinstance(fixed_goal, Mapping)
        or fixed_goal.get("source") != "teacher_input_normalized_locally"
        or fixed_goal.get("mutable_by_model") is not False
        or not isinstance(fixed_profile, Mapping)
        or fixed_profile.get("source") != "teacher_input_normalized_locally"
        or fixed_profile.get("mutable_by_model") is not False
    ):
        raise ValueError("layered context immutable teacher input is invalid")
    current_plan = context.get("current_plan")
    if (
        not isinstance(current_plan, Mapping)
        or current_plan.get("source")
        != "deterministic_session_plan_and_last_validated_action"
    ):
        raise ValueError("layered context current plan provenance is invalid")
    working = context.get("working_memory")
    if (
        not isinstance(working, Mapping)
        or not isinstance(working.get("current_learner_response"), str)
        or not isinstance(working.get("recent_turns"), list)
        or not isinstance(working.get("current_knowledge_components"), list)
    ):
        raise ValueError("layered context working memory is invalid")
    response_id = working.get("current_response_evidence_id")
    if snapshot["current_response_present"] != isinstance(response_id, str):
        raise ValueError("layered context current-response provenance is inconsistent")
    semantic = context.get("semantic_summary")
    if (
        not isinstance(semantic, Mapping)
        or semantic.get("compression_method")
        != "deterministic_aggregate_and_extractive_checkpoints_no_model_generation"
        or semantic.get("source") != "omitted_local_session_history"
        or semantic.get("narrative_inference_added") is not False
        or not isinstance(semantic.get("focus_checkpoints"), list)
        or len(semantic["focus_checkpoints"]) > 4
    ):
        raise ValueError("layered context semantic summary is invalid")
    teaching_memory = semantic.get("teaching_memory")
    if teaching_memory is not None:
        if (
            not isinstance(teaching_memory, Mapping)
            or teaching_memory.get("schema") != TEACHING_MEMORY_PROJECTION_SCHEMA
            or teaching_memory.get("source")
            != "deterministic_evidence_linked_rollout_projection"
            or teaching_memory.get("narrative_inference_added") is not False
            or teaching_memory.get("model_may_mutate") is not False
            or isinstance(teaching_memory.get("history_version"), bool)
            or not isinstance(teaching_memory.get("history_version"), int)
            or teaching_memory.get("history_version", -1) < 0
            or any(
                not isinstance(teaching_memory.get(field), list)
                for field in (
                    "active_preferences",
                    "unresolved_questions",
                    "pending_teacher_commitments",
                    "active_referents",
                )
            )
            or len(teaching_memory.get("active_preferences", [])) > 6
            or len(teaching_memory.get("unresolved_questions", [])) > 6
            or len(teaching_memory.get("pending_teacher_commitments", [])) > 4
            or len(teaching_memory.get("active_referents", [])) > 4
        ):
            raise ValueError("layered context teaching memory is invalid")
    continuity_recall = semantic.get("continuity_recall")
    if continuity_recall is not None:
        if (
            not isinstance(continuity_recall, Mapping)
            or continuity_recall.get("schema") != CONTINUITY_RECALL_SCHEMA
            or continuity_recall.get("cue_kind")
            not in {
                "explicit_round_reference",
                "semantic_topic_reference",
                "ordinal_reference",
                "earliest_learner_instruction",
                "earlier_unresolved_question",
                "prior_agreement_or_agenda",
                "recent_repetition_complaint",
            }
            or not isinstance(continuity_recall.get("cue_excerpt"), str)
            or not continuity_recall.get("cue_excerpt")
            or not isinstance(continuity_recall.get("cue_evidence_refs"), list)
            or not continuity_recall.get("cue_evidence_refs")
            or continuity_recall.get("status")
            not in {
                "resolved_evidence_linked",
                "unresolved_no_matching_evidence",
            }
            or not isinstance(continuity_recall.get("instruction"), str)
            or not continuity_recall.get("instruction")
            or not isinstance(continuity_recall.get("selection_policy"), str)
            or not continuity_recall.get("selection_policy")
            or continuity_recall.get("must_not_invent") is not True
        ):
            raise ValueError("layered context continuity recall is invalid")
        recall_target = continuity_recall.get("target")
        if continuity_recall["status"] == "unresolved_no_matching_evidence":
            if recall_target is not None:
                raise ValueError(
                    "unresolved continuity recall must not contain a target"
                )
        elif (
            not isinstance(recall_target, Mapping)
            or recall_target.get("kind")
            not in {
                "historical_turn",
                "learner_named_alternatives",
                "teacher_named_alternatives",
                "learner_instruction",
                "unresolved_learner_question",
                "learner_future_agenda",
                "teacher_commitment",
                "teacher_question",
                "learner_difficulty_signal",
                "current_teacher_action",
            }
            or recall_target.get("speaker")
            not in {"learner", "teacher", "teacher_profile"}
            or isinstance(recall_target.get("source_round"), bool)
            or not isinstance(recall_target.get("source_round"), int)
            or recall_target.get("source_round", -1) < 0
            or not isinstance(recall_target.get("excerpt"), str)
            or not recall_target.get("excerpt")
            or not isinstance(recall_target.get("evidence_refs"), list)
            or not recall_target.get("evidence_refs")
        ):
            raise ValueError("resolved continuity recall target is invalid")
    teaching_checkpoints = semantic.get("teaching_checkpoints")
    allowed_checkpoint_kinds = {
        "unresolved_learning_signal",
        "explicit_learner_question",
        "explicit_learner_preference_or_constraint",
        "verified_prerequisite",
        "teacher_next_step_statement",
    }
    if (
        not isinstance(teaching_checkpoints, list)
        or len(teaching_checkpoints) > 6
        or any(
            not isinstance(item, Mapping)
            or item.get("checkpoint_version") != 1
            or not isinstance(item.get("checkpoint_id"), str)
            or not item.get("checkpoint_id")
            or item.get("kind") not in allowed_checkpoint_kinds
            or not isinstance(item.get("status"), str)
            or not item.get("status")
            or isinstance(item.get("round"), bool)
            or not isinstance(item.get("round"), int)
            or item.get("round", -1) < 0
            or not isinstance(item.get("knowledge_components"), list)
            or not isinstance(item.get("excerpt"), str)
            or not isinstance(item.get("evidence_refs"), list)
            or not item.get("evidence_refs")
            or not isinstance(item.get("derivation"), str)
            or not item.get("derivation")
            for item in teaching_checkpoints
        )
        or len(
            {
                str(item.get("checkpoint_id"))
                for item in teaching_checkpoints
                if isinstance(item, Mapping)
            }
        )
        != len(teaching_checkpoints)
    ):
        raise ValueError("layered context teaching checkpoints are invalid")
    knowledge = context.get("knowledge_state")
    if (
        not isinstance(knowledge, Mapping)
        or knowledge.get("source")
        != "deterministic_state_machine_over_labeled_observations"
        or not isinstance(knowledge.get("concept_mastery"), list)
        or not isinstance(knowledge.get("misconceptions"), list)
        or not isinstance(knowledge.get("unresolved_issues"), list)
    ):
        raise ValueError("layered context knowledge state is invalid")
    candidate = context.get("candidate_long_term_memory")
    if (
        not isinstance(candidate, Mapping)
        or candidate.get("status") != "candidate_unconfirmed"
        or candidate.get("may_override_teacher_profile") is not False
        or candidate.get("eligible_sources")
        != ["deepseek_v4_flash_validated_diagnosis"]
        or not isinstance(candidate.get("observations"), list)
        or len(candidate["observations"]) > 4
        or any(
            not isinstance(item, Mapping)
            or item.get("status") != "candidate_unconfirmed"
            for item in candidate["observations"]
        )
    ):
        raise ValueError("layered context candidate memory is invalid")
    retrieval = context.get("retrieval")
    if (
        not isinstance(retrieval, Mapping)
        or retrieval.get("fixed_context_always_included") is not True
        or retrieval.get("current_plan_always_included") is not True
        or retrieval.get("omitted_turns_accounted_for_statistically") is not True
        or retrieval.get("teaching_checkpoints_are_selective_extracts") is not True
        or "semantic_summary_covers_omitted_turns" in retrieval
        or retrieval.get("candidate_memory_policy")
        != "latest_validated_candidates_only"
    ):
        raise ValueError("layered context retrieval metadata is invalid")
    privacy = context.get("privacy")
    if (
        not isinstance(privacy, Mapping)
        or not isinstance(privacy.get("remote_text_redacted"), bool)
        or not isinstance(privacy.get("known_pattern_identifiers_redacted"), bool)
        or privacy.get("raw_identity_fields_sent") != "not_established"
        or privacy.get("residual_identity_risk") is not True
        or privacy.get("original_values_retained_in_context") is not False
    ):
        raise ValueError("layered context privacy metadata is invalid")
    budget = context.get("budget")
    if not isinstance(budget, Mapping):
        raise ValueError("layered context budget is missing")
    maximum = budget.get("max_chars")
    measured = _serialized_length(context)
    if (
        isinstance(maximum, bool)
        or not isinstance(maximum, int)
        or maximum < MINIMUM_LAYERED_CONTEXT_CHARS
        or measured > maximum
        or budget.get("serialized_chars") != measured
        or budget.get("policy") != "hard_serialized_character_cap"
        or budget.get("retained_recent_turns") != len(working["recent_turns"])
        or budget.get("retained_candidate_observations")
        != len(candidate["observations"])
        or not isinstance(budget.get("truncated"), bool)
    ):
        raise ValueError("layered context character budget is invalid")
    ledger = context.get("evidence_ledger")
    if not isinstance(ledger, list) or not all(
        isinstance(item, Mapping) for item in ledger
    ):
        raise ValueError("layered context evidence ledger is invalid")
    identifiers = [str(item.get("evidence_id", "")) for item in ledger]
    if any(not item for item in identifiers) or len(identifiers) != len(
        set(identifiers)
    ):
        raise ValueError("layered context evidence IDs must be unique and non-empty")
    dangling = set(_collect_evidence_refs(context)) - set(identifiers)
    if dangling:
        raise ValueError(
            f"layered context has dangling evidence refs: {sorted(dangling)}"
        )
    boundary = context.get("claim_boundary", {})
    if not isinstance(boundary, Mapping) or any(
        boundary.get(field) is not expected
        for field, expected in {
            "model_generated_history_summary": False,
            "candidate_long_term_memory_is_confirmed": False,
            "fallback_observations_are_labeled_by_actual_source": True,
            "context_is_evidence_not_learner_ground_truth": True,
            "model_may_not_mutate_fixed_context": True,
            "omitted_turn_semantics_are_exhaustive": False,
        }.items()
    ):
        raise ValueError("layered context claim boundary is invalid")


def _goal_text(goal: Mapping[str, Any], key: str) -> str:
    value = goal.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"goal.{key} must be a non-empty string")
    return value.strip()


def _threshold(goal: Mapping[str, Any], dimension: str, default: float) -> float:
    thresholds = goal.get("success_thresholds", {})
    if not isinstance(thresholds, Mapping):
        raise ValueError("goal.success_thresholds must be an object")
    value = thresholds.get(dimension, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"goal.success_thresholds.{dimension} must be in [0, 1]")
    result = float(value)
    if not 0 <= result <= 1:
        raise ValueError(f"goal.success_thresholds.{dimension} must be in [0, 1]")
    return result


def build_goal_plan(goal: Mapping[str, Any]) -> dict[str, Any]:
    """Turn a teaching goal into four observable intermediate objectives."""

    if not isinstance(goal, Mapping):
        raise TypeError("goal must be a mapping")
    concept = _goal_text(goal, "concept")
    objective = _goal_text(goal, "objective")
    thresholds = {
        "prerequisite": _threshold(goal, "prerequisite", 0.60),
        "conceptual": _threshold(goal, "conceptual", 0.65),
        "procedural": _threshold(goal, "procedural", 0.60),
        "transfer": _threshold(goal, "transfer", 0.55),
    }
    materials = goal.get("materials", {})
    if not isinstance(materials, Mapping):
        raise ValueError("goal.materials must be an object")
    practice = _as_nonempty_string(materials.get("practice"))
    transfer_task = _as_nonempty_string(materials.get("transfer_task"))

    specifications = (
        (
            "prerequisite",
            f"确认理解 {concept} 所需的前置知识",
            f"学习者能独立说出至少一个必要前置概念，并解释它与 {concept} 的关系。",
        ),
        (
            "conceptual",
            f"建立 {concept} 的概念表征与适用边界",
            f"学习者能用自己的话定义 {concept}，并给出一个适用例和一个不适用情形。",
        ),
        (
            "procedural",
            f"完成一次 {concept} 的关键操作或推导",
            (
                f"学习者能完成练习“{practice}”的关键步骤并逐步说明理由。"
                if practice
                else f"学习者能完成一道只考查 {concept} 的代表性练习，并逐步说明理由。"
            ),
        ),
        (
            "transfer",
            f"把 {concept} 迁移到表面不同的新情境",
            (
                f"学习者能处理迁移任务“{transfer_task}”，并说明可迁移与不可迁移的条件。"
                if transfer_task
                else f"学习者能在一个新情境中判断是否应使用 {concept}，并说明判断依据。"
            ),
        ),
    )

    steps: list[dict[str, Any]] = []
    for index, (dimension, step_objective, criterion) in enumerate(specifications, 1):
        steps.append(
            {
                "step_id": f"goal_step_{index:02d}",
                "order": index,
                "dimension": dimension,
                "objective": step_objective,
                "verification": {
                    "method": "observable_learner_response_or_task_performance",
                    "success_criterion": criterion,
                    "mastery_threshold": thresholds[dimension],
                },
                "status": "active" if index == 1 else "pending",
            }
        )

    return {
        "schema": GOAL_PLAN_SCHEMA,
        "goal": {"concept": concept, "objective": objective},
        "status": "active",
        "intermediate_objectives": steps,
        "active_step": steps[0]["step_id"],
        "progress": {
            "completed_steps": 0,
            "total_steps": len(steps),
            "fraction": 0.0,
        },
        "claim_boundary": {
            "objectives_are_deterministic_plan_scaffolds": True,
            "objective_completion_requires_observation": True,
            "plan_is_learning_effect_evidence": False,
        },
    }
