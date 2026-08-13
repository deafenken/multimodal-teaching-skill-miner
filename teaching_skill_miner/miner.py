from __future__ import annotations

import hashlib
import re
from collections import Counter
from typing import Any

from .models import validate_skill, validate_transcript
from .teaching_phases import (
    detect_teaching_phases,
    ordered_phase_plan,
)


STRATEGY_PATTERNS: dict[str, tuple[str, ...]] = {
    "concrete_example": ("example", "suppose", "consider", "for instance", "例如", "比如", "假设"),
    "step_by_step": ("first", "then", "next", "step", "algorithm", "逐步", "首先", "然后", "接着"),
    "question_and_wait": ("?", "predict", "what happens", "can you", "why", "想一想", "为什么", "会怎样"),
    "socratic_followup": ("why", "what if", "how do we know", "why not", "追问", "如果", "怎么知道"),
    "error_correction": ("wrong", "mistake", "error", "fail", "debug", "错误", "失败", "调试", "纠正"),
    "contrastive_learning": ("compare", "difference", "versus", "contrast", "instead", "区别", "对比", "而不是"),
    "intuition_to_formalization": ("picture", "intuitive", "visual", "formal", "definition", "geometry", "直观", "图像", "定义", "形式化"),
    "whole_to_parts": ("roadmap", "big picture", "overview", "whole program", "break down", "break the", "整体", "路线图", "分解"),
    "line_by_line_explanation": ("code", "line", "run", "output", "formula", "equation", "代码", "逐行", "公式", "运行"),
    "practice_feedback_loop": ("try", "exercise", "practice", "check", "feedback", "练习", "检查", "反馈"),
    "difficulty_progression": ("simple", "harder", "two", "three", "general", "scale", "简单", "更难", "推广", "一般"),
    "review_and_spaced_recall": ("review", "remember", "recall", "previous", "回顾", "复习", "还记得"),
    "summary_and_transfer": ("summary", "generalize", "new case", "transfer", "takeaway", "总结", "迁移", "新情境"),
    "adaptive_teaching": ("if you", "depending", "if not", "slow down", "如果学生", "取决于", "调整"),
}

STRATEGY_NAMES = {
    "concrete_example": "具体例子驱动",
    "step_by_step": "逐步拆解",
    "question_and_wait": "提问与等待",
    "socratic_followup": "苏格拉底式追问",
    "error_correction": "错误示范与纠错",
    "contrastive_learning": "对比学习",
    "intuition_to_formalization": "先直觉后形式化",
    "whole_to_parts": "先整体后局部",
    "line_by_line_explanation": "代码或公式逐行解释",
    "practice_feedback_loop": "练习—反馈—再练习",
    "difficulty_progression": "难度递进",
    "review_and_spaced_recall": "知识回顾与间隔复习",
    "summary_and_transfer": "总结与迁移",
    "adaptive_teaching": "根据学生反应动态调整",
}

ACTION_PATTERNS: dict[str, tuple[str, ...]] = {
    "ask": ("?", "ask", "predict", "why", "can you", "问题", "为什么"),
    "explain": ("explain", "means", "definition", "because", "解释", "定义"),
    "demonstrate": ("example", "show", "draw", "run", "suppose", "演示", "例如"),
    "contrast": ("compare", "difference", "versus", "instead", "对比", "区别"),
    "check_understanding": ("check", "try", "exercise", "verify", "检查", "练习"),
    "give_feedback": ("feedback", "correct", "mistake", "error", "反馈", "纠正"),
    "summarize": ("summary", "takeaway", "generalize", "总结", "推广"),
    "adapt": ("if not", "depending", "adjust", "slow down", "调整", "如果学生"),
}


def _text(transcript: dict[str, Any]) -> str:
    return " ".join(segment["text"] for segment in transcript.get("segments", [])).lower()


def _slug(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if normalized:
        return normalized[:54]
    return "skill_" + hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]


def detect_strategies(transcript: dict[str, Any]) -> list[dict[str, Any]]:
    text = _text(transcript)
    multimodal_counts: Counter[str] = Counter()
    for event in transcript.get("multimodal", {}).get("events", []):
        if not isinstance(event, dict):
            continue
        for strategy in event.get("supports_strategies", []):
            if strategy in STRATEGY_PATTERNS:
                multimodal_counts[strategy] += 1
    detected: list[dict[str, Any]] = []
    for strategy, patterns in STRATEGY_PATTERNS.items():
        text_matches = sum(text.count(pattern.lower()) for pattern in patterns)
        multimodal_matches = multimodal_counts[strategy]
        weighted_matches = text_matches + 2 * multimodal_matches
        if weighted_matches:
            detected.append(
                {
                    "id": strategy,
                    "name": STRATEGY_NAMES[strategy],
                    "evidence_count": text_matches + multimodal_matches,
                    "text_evidence_count": text_matches,
                    "multimodal_evidence_count": multimodal_matches,
                    "confidence": round(min(0.98, 0.52 + 0.08 * weighted_matches), 2),
                    "origin": "observed_method",
                }
            )
    detected.sort(key=lambda item: (-item["evidence_count"], item["id"]))
    if not detected:
        detected.append(
            {
                "id": "step_by_step",
                "name": STRATEGY_NAMES["step_by_step"],
                "evidence_count": 0,
                "text_evidence_count": 0,
                "multimodal_evidence_count": 0,
                "confidence": 0.35,
                "origin": "recommended_enrichment",
            }
        )
    return detected[:8]


def detect_actions(transcript: dict[str, Any]) -> list[str]:
    text = _text(transcript)
    scores = Counter(
        {
            action: sum(text.count(pattern.lower()) for pattern in patterns)
            for action, patterns in ACTION_PATTERNS.items()
        }
    )
    actions = [name for name, score in scores.most_common() if score > 0]
    baseline = ["ask", "explain", "demonstrate", "check_understanding", "give_feedback", "summarize", "adapt"]
    for action in baseline:
        if action not in actions:
            actions.append(action)
    return actions[:8]


def infer_bloom(title: str, strategies: list[dict[str, Any]]) -> str:
    lowered = title.lower()
    ids = {item["id"] for item in strategies}
    if any(word in lowered for word in ("debug", "efficiency", "compare", "inverse", "decomposition", "aliasing")):
        return "analyze"
    if ids & {"line_by_line_explanation", "practice_feedback_loop", "step_by_step"}:
        return "apply"
    return "understand"


def _stable_evidence_id(
    transcript: dict[str, Any], segment_index: int, segment: dict[str, Any]
) -> str:
    material = "\x1f".join(
        (
            str(transcript.get("video_id", "")),
            str(segment_index),
            str(segment.get("start", "")),
            str(segment.get("end", "")),
            str(segment.get("text", "")),
        )
    )
    return "evi_" + hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _parent_segment_index(segment: dict[str, Any], local_index: int) -> int:
    """Return the immutable index in the parent transcript when available.

    Episode mining passes segments through a provenance-preserving slice.  The
    slice is intentionally allowed to be re-indexed for the phase detector,
    but evidence IDs must continue to use the parent's index.  Keeping this
    fallback local-index based preserves the exact behaviour for ordinary
    (unsliced) transcripts and for older callers.
    """

    value = segment.get("_parent_segment_index")
    if value is None:
        return local_index
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return local_index
    return parsed if parsed >= 0 else local_index


#: Quotes are truncated for readability; strategy support is therefore matched
#: against the truncated quote, never against the full segment, so that what a
#: reader can verify in the Skill is exactly what the evaluator checked.
QUOTE_LIMIT = 280

#: Per observed phase, how many backing segments are cited.  Long real captions
#: would otherwise push hundreds of segments into a single Skill document.
MAX_EVIDENCE_PER_PHASE = 3

#: Strongest strategy-matched segments kept in addition to the phase evidence.
MAX_STRATEGY_EVIDENCE = 4


def _quote_of(segment: dict[str, Any]) -> str:
    return str(segment.get("text", ""))[:QUOTE_LIMIT]


def _supported_strategies(quote: str, strategy_ids: set[str]) -> list[str]:
    lowered = quote.lower()
    return sorted(
        strategy
        for strategy in strategy_ids
        if any(pattern.lower() in lowered for pattern in STRATEGY_PATTERNS[strategy])
    )


def select_evidence(
    transcript: dict[str, Any],
    strategy_ids: set[str],
    phase_detection: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Pick the transcript segments the Skill will cite.

    Segments backing an observed teaching phase are kept unconditionally,
    because the distilled procedure references them by evidence id.  The
    strongest strategy-matched segments are added on top, and the result is
    emitted in timeline order so a reviewer can read it against the video.
    """

    segments = transcript.get("segments", [])
    keep: set[int] = set()

    if phase_detection:
        for record in phase_detection.get("observed_phases", []):
            for index in record["segment_indices"][:MAX_EVIDENCE_PER_PHASE]:
                keep.add(index)

    scored: list[tuple[int, int, int]] = []
    for index, segment in enumerate(segments):
        matched = _supported_strategies(_quote_of(segment), strategy_ids)
        scored.append((len(matched), -index, index))
    scored.sort(reverse=True, key=lambda item: (item[0], item[1]))
    strategy_ranked = [item[2] for item in scored if item[0] > 0][:MAX_STRATEGY_EVIDENCE]
    if not strategy_ranked and not keep:
        strategy_ranked = [item[2] for item in scored[:2]]
    keep.update(strategy_ranked)

    evidence: list[dict[str, Any]] = []
    for index in sorted(keep):
        segment = segments[index]
        quote = _quote_of(segment)
        supports = _supported_strategies(quote, strategy_ids)
        parent_index = _parent_segment_index(segment, index)
        item = {
            "evidence_id": _stable_evidence_id(transcript, parent_index, segment),
            "start": segment["start"],
            "end": segment["end"],
            "quote": quote,
            "supports": supports or ["teaching_sequence"],
            # ``segment_index`` is the parent/global index for an episode
            # slice, while ordinary transcripts retain their historical local
            # index semantics.
            "segment_index": parent_index,
        }
        if "_parent_segment_index" in segment:
            # The local index is retained solely so phase observations on a
            # slice can be joined back to evidence records without changing
            # the stable parent ID namespace.
            item["local_segment_index"] = index
        evidence.append(item)
    return evidence


def select_multimodal_evidence(transcript: dict[str, Any], strategy_ids: set[str]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for event in transcript.get("multimodal", {}).get("events", []):
        if not isinstance(event, dict):
            continue
        supports = set(event.get("supports_strategies", []))
        overlap = sorted(supports & strategy_ids)
        if not overlap:
            continue
        candidates.append(
            {
                "event_id": event.get("event_id"),
                "type": event.get("type"),
                "start": event.get("start"),
                "end": event.get("end"),
                "modalities": event.get("modalities", []),
                "supports": overlap,
                "evidence": event.get("evidence", {}),
                "confidence": event.get("confidence"),
            }
        )
    candidates.sort(key=lambda item: (-len(item["modalities"]), -float(item.get("confidence") or 0), float(item.get("start") or 0)))
    return candidates[:6]


def _clock(value: Any) -> str:
    try:
        seconds = int(float(value))
    except (TypeError, ValueError):
        return "--:--"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _phase_strategy_id(
    phase: dict[str, Any], mined_strategy_ids: set[str], primary: str
) -> str:
    """Bind a phase to a strategy the miner actually observed.

    ``validate_skill`` requires an ``observed_method`` step to name a strategy
    that exists on the Skill, so a phase whose natural strategy was not mined
    falls back to the lesson's primary strategy.
    """

    for candidate in phase["strategies"]:
        if candidate in mined_strategy_ids:
            return candidate
    return primary


def build_procedure(
    phase_detection: dict[str, Any],
    evidence_id_by_segment: dict[int, str],
    *,
    mined_strategy_ids: set[str],
    primary: str,
) -> list[dict[str, Any]]:
    """Turn the observed teaching flow into an executable procedure.

    Phases the teacher actually used keep their observed order, timestamps and
    backing evidence, and are marked ``observed_method``.  The remaining
    canonical phases are emitted as ``recommended_enrichment`` scaffolds at
    their canonical position so the Skill still runs end to end; they carry no
    evidence and never count as video method evidence.
    """

    steps: list[dict[str, Any]] = []
    for number, entry in enumerate(ordered_phase_plan(phase_detection), 1):
        phase = entry["phase"]
        record = entry["observation"]
        evidence_ids: list[str] = []
        if record is not None:
            evidence_ids = list(
                dict.fromkeys(
                    evidence_id_by_segment[index]
                    for index in record["segment_indices"]
                    if index in evidence_id_by_segment
                )
            )
        observed = bool(evidence_ids)
        matched_cues = list(record["matched_cues"][:4]) if record is not None else []

        if observed:
            span = f"{_clock(record['start'])}–{_clock(record['end'])}"
            cues = "、".join(matched_cues) if matched_cues else "教学流程"
            instruction = (
                f"{phase['instruction']}"
                f"（视频 {span} 观察到该环节；触发线索：{cues}）"
            )
            origin = "observed_method"
            strategy_id = _phase_strategy_id(phase, mined_strategy_ids, primary)
            derivation = "observed_teaching_phase_from_timeline"
        else:
            instruction = f"{phase['instruction']}（视频中未观察到该环节，作为可执行教学补充）"
            origin = "recommended_enrichment"
            strategy_id = None
            derivation = "canonical_phase_scaffold"

        steps.append(
            {
                "step": number,
                "teaching_phase": phase["id"],
                "teaching_phase_name": phase["name"],
                "canonical_phase_rank": phase["rank"],
                "teacher_action": phase["teacher_action"],
                "instruction": instruction,
                "expected_signal": phase["expected_signal"],
                "fallback": phase["fallback"],
                "origin": origin,
                "evidence_ids": evidence_ids,
                "observed_span": (
                    {"start": record["start"], "end": record["end"]} if observed else None
                ),
                "matched_cues": matched_cues if observed else [],
                "provenance": {
                    "origin": origin,
                    "strategy_id": strategy_id,
                    "evidence_ids": evidence_ids,
                    "derivation": derivation,
                },
            }
        )
    return steps


def mine_skill(transcript: dict[str, Any]) -> dict[str, Any]:
    transcript_result = validate_transcript(transcript)
    if not transcript_result.valid:
        raise ValueError("invalid transcript: " + "; ".join(transcript_result.errors))

    title = str(transcript["title"])
    topic = re.sub(r"^lecture\s*\d+\s*:\s*", "", title, flags=re.IGNORECASE).strip()
    strategies = detect_strategies(transcript)
    primary = strategies[0]["id"]
    bloom = infer_bloom(title, strategies)
    actions = detect_actions(transcript)
    phase_detection = detect_teaching_phases(transcript)
    text_evidence = select_evidence(
        transcript,
        {item["id"] for item in strategies[:4]},
        phase_detection,
    )
    multimodal_evidence = select_multimodal_evidence(
        transcript, {item["id"] for item in strategies}
    )
    evidence_id_by_segment = {
        int(item.get("local_segment_index", item["segment_index"])): str(item["evidence_id"])
        for item in text_evidence
        if item.get("segment_index") is not None
    }
    mined_strategy_ids = {item["id"] for item in strategies}
    procedure = build_procedure(
        phase_detection,
        evidence_id_by_segment,
        mined_strategy_ids=mined_strategy_ids,
        primary=primary,
    )
    procedure_actions = [step["teacher_action"] for step in procedure]
    for action in procedure_actions:
        if action not in actions:
            actions.append(action)
    source = {
        "video_id": transcript["video_id"],
        "course_id": transcript["course_id"],
        "title": title,
        "source_url": transcript["source_url"],
        "transcript_kind": transcript.get("transcript_kind", "unknown"),
        "language": transcript.get("language", "unknown"),
        "evidence": text_evidence,
    }
    if transcript.get("multimodal"):
        source["modalities_available"] = transcript["multimodal"].get("modalities_available", ["transcript"])
        source["multimodal_evidence"] = multimodal_evidence
    if transcript.get("transcript_url"):
        source["transcript_url"] = transcript["transcript_url"]
    if transcript.get("provenance"):
        source["provenance"] = transcript["provenance"]

    skill: dict[str, Any] = {
        "skill_id": f"{_slug(primary)}_{_slug(str(transcript['video_id']))}_v1",
        "name": f"{STRATEGY_NAMES[primary]}：{topic}",
        "version": "1.0",
        "mining_metadata": {
            "method": "interpretable_heuristic_v2_phase_sequence",
            "segment_count": len(transcript.get("segments", [])),
            "evidence_policy": "exact_substring_with_timestamp",
            "multimodal_event_count": len(transcript.get("multimodal", {}).get("events", [])),
            "multimodal_evidence_count": len(multimodal_evidence),
            "observed_strategy_count": sum(
                item.get("origin") == "observed_method" for item in strategies
            ),
            "recommended_strategy_count": sum(
                item.get("origin") == "recommended_enrichment"
                for item in strategies
            ),
            "observed_procedure_step_count": sum(
                step.get("origin") == "observed_method" for step in procedure
            ),
            "recommended_procedure_step_count": sum(
                step.get("origin") == "recommended_enrichment"
                for step in procedure
            ),
            "teaching_phase_analysis": {
                "detector": phase_detection["detector"],
                "detector_semantics": phase_detection["detector_semantics"],
                "canonical_phase_count": phase_detection["canonical_phase_count"],
                "observed_phase_count": phase_detection["observed_phase_count"],
                "observed_phase_sequence": phase_detection["observed_phase_sequence"],
                "unused_phase_ids": [
                    item["phase_id"] for item in phase_detection["unused_phases"]
                ],
                "phase_labelled_segment_count": phase_detection[
                    "phase_labelled_segment_count"
                ],
                "segment_coverage": phase_detection["segment_coverage"],
            },
        },
        "source": source,
        "learning_objective": {
            "statement": f"学习者能够解释 {topic} 的关键结构，并在一个新问题中正确应用。",
            "bloom_level": bloom,
            "observable": True,
            "assessment": "通过口头解释、边界判断和一道迁移题观察。",
        },
        "trigger": [
            f"学习者第一次系统学习 {topic}",
            "学习者会模仿步骤，但不能解释步骤与概念之间的关系",
        ],
        "preconditions": [
            "已用一个诊断问题检查必要前置知识",
            "具备完成最小实例所需的语言、符号或操作基础",
        ],
        "goal": f"通过“{STRATEGY_NAMES[primary]}”帮助学习者建立 {{concept}} 的可解释心智模型，并能迁移使用。",
        "parameters": {
            "concept": {"type": "string", "required": True, "default": topic},
            "learner_level": {"type": "string", "required": False, "default": "beginner"},
        },
        "strategies": strategies,
        "procedure": procedure,
        "teacher_actions": actions,
        "student_signals": [
            "能用自己的话解释关键结构",
            "能指出正例与反例的决定性差异",
            "能在只提供分层提示的情况下解决新问题",
        ],
        "success_criteria": [
            "迁移题关键步骤正确率达到 80%",
            "反例判断正确且理由引用了适用条件",
            "学习者总结同时包含概念、条件和用途",
        ],
        "failure_modes": [
            {"mode": "实例复杂度过高", "mitigation": "减少变量，只保留一个待观察结构。"},
            {"mode": "直接给出形式定义而缺少映射", "mitigation": "返回具体实例并逐项建立对应。"},
            {"mode": "只检查答案不检查理由", "mitigation": "追加“为什么”与反例判断。"},
            {"mode": "所有学习者使用同一节奏", "mitigation": "依据预期信号选择 fallback 或进阶题。"},
        ],
        "verification": [
            {
                "type": "near_transfer",
                "prompt": "换一个表面情境，使用 {concept} 完成关键步骤并解释依据。",
                "pass_condition": "关键步骤正确，且解释至少引用一个适用条件。",
            },
            {
                "type": "counterexample",
                "prompt": "判断一个边界案例是否属于 {concept}，若不是请指出失效条件。",
                "pass_condition": "判断正确，并明确指出决定性的条件。",
            },
        ],
    }
    result = validate_skill(skill)
    if not result.valid:
        raise AssertionError("generated invalid skill: " + "; ".join(result.errors))
    return skill
