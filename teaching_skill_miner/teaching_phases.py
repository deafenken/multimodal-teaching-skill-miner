"""Distillation of the teacher's observed teaching flow.

The competition statement (§4.2 教学步骤) asks the system to identify the
sequence of teaching phases a teacher actually used, rather than to emit a
generic lesson template.  This module maps transcript segments onto the nine
canonical phases named in that section, keeps the teacher's real temporal
order, and reports which phases were never observed so the generated Skill can
mark them as recommended scaffolding instead of claiming them as video
evidence.

The detector is deliberately interpretable: every observed phase is backed by
the exact transcript segments that triggered it, together with the cue tokens
that matched.  Nothing here predicts learning outcomes; it only records which
teaching moves are visible on the timeline.
"""

from __future__ import annotations

from typing import Any


#: The nine canonical phases listed in the competition statement, in the order
#: the statement presents them.  ``rank`` is that canonical order; it is *not*
#: the order a given teacher used.
CANONICAL_PHASES: tuple[dict[str, Any], ...] = (
    {
        "id": "prior_knowledge_review",
        "name": "复习前置知识",
        "rank": 1,
        "teacher_action": "ask",
        "strategies": ("review_and_spaced_recall", "question_and_wait"),
        "position_prior": "early",
        "cues": (
            "review",
            "recall",
            "remember",
            "previous lecture",
            "previous",
            "last lecture",
            "already know",
            "earlier we",
            "回顾",
            "复习",
            "还记得",
            "上一讲",
            "前置",
        ),
        "instruction": (
            "先用一个诊断问题回顾 {concept} 所需的前置知识，等待学习者作答后再进入新内容。"
        ),
        "expected_signal": "能正确说明前置概念并举出一个例子",
        "fallback": "若回答不完整，给出一个最小提示，并换一道更简单的前置题重新确认。",
    },
    {
        "id": "problem_or_context_setup",
        "name": "提出问题或情境",
        "rank": 2,
        "teacher_action": "prompt",
        "strategies": ("question_and_wait", "whole_to_parts", "socratic_followup"),
        "position_prior": "early",
        "cues": (
            "roadmap",
            "big picture",
            "overview",
            "ask what",
            "ask how",
            "ask where",
            "ask why",
            "ask students",
            "present the whole",
            "begin with the big",
            "问题",
            "情境",
            "路线图",
            "整体",
            "先提出",
        ),
        "instruction": (
            "提出一个 {concept} 尚不能解决的问题或情境，明确本轮要回答什么，并留出思考时间。"
        ),
        "expected_signal": "能复述待解决的问题，并说出自己的初步猜想",
        "fallback": "把问题缩小到一个更具体的场景，并给出两个候选答案让学习者选择。",
    },
    {
        "id": "intuitive_example",
        "name": "给出直观例子",
        "rank": 3,
        "teacher_action": "demonstrate",
        "strategies": ("concrete_example", "intuition_to_formalization"),
        "cues": (
            "concrete example",
            "for instance",
            "familiar",
            "simple example",
            "small example",
            "example",
            "suppose",
            "draw",
            "picture",
            "visual",
            "intuition",
            "intuitive",
            "具体例子",
            "例如",
            "比如",
            "直观",
            "图像",
        ),
        "instruction": (
            "给出一个最小、可手算的 {concept} 直观例子，先让学习者描述其中的已知量与目标，暂不给定义。"
        ),
        "expected_signal": "能用非术语语言指出例子中的关键结构",
        "fallback": "进一步减少变量，只保留一个待观察的结构，并示范第一个微步骤。",
    },
    {
        "id": "abstract_concept_building",
        "name": "建立抽象概念",
        "rank": 4,
        "teacher_action": "explain",
        "strategies": ("intuition_to_formalization", "step_by_step"),
        "cues": (
            "definition",
            "define",
            "formal",
            "abstract",
            "notation",
            "specification",
            "general rule",
            "generalize",
            "the formal",
            "定义",
            "形式化",
            "抽象",
            "记号",
            "一般化",
        ),
        "instruction": (
            "把刚才例子中的共同结构映射到 {concept} 的正式定义、记号或规则，逐项建立对应关系。"
        ),
        "expected_signal": "能逐项说明例子与形式定义之间的对应",
        "fallback": "回到例子，用逐项对照表重新建立映射，再复述一次定义。",
    },
    {
        "id": "derivation_or_operation_walkthrough",
        "name": "展示推导或操作",
        "rank": 5,
        "teacher_action": "demonstrate",
        "strategies": ("line_by_line_explanation", "step_by_step", "difficulty_progression"),
        "cues": (
            "step by step",
            "line by line",
            "walk through",
            "trace",
            "derive",
            "derivation",
            "run the",
            "run it",
            "compute",
            "multiply",
            "eliminate",
            "elimination",
            "first choose",
            "then multiply",
            "algorithm",
            "逐行",
            "逐步",
            "推导",
            "演算",
            "运行",
        ),
        "instruction": (
            "逐行或逐步演示 {concept} 的推导与操作，每完成一步先让学习者预测下一步的结果再公布。"
        ),
        "expected_signal": "能说出每一步的输入、操作与输出",
        "fallback": "只演示一个微步骤，标注变化的量，再让学习者接着做下一步。",
    },
    {
        "id": "understanding_check",
        "name": "检查学生理解",
        "rank": 6,
        "teacher_action": "check_understanding",
        "strategies": ("question_and_wait", "practice_feedback_loop", "socratic_followup"),
        "cues": (
            "check",
            "checking",
            "predict",
            "verify",
            "ask what happens",
            "ask which",
            "ask students to predict",
            "invite students to predict",
            "检查",
            "预测",
            "验证",
            "追问",
        ),
        "instruction": (
            "用一道表面不同但结构相同的问题检查 {concept} 的理解，要求学习者口述判断依据而不只给答案。"
        ),
        "expected_signal": "能在新表述下完成关键步骤，并说明理由",
        "fallback": "改为分层提示：先指出目标，再提示适用规则，最后只示范第一步。",
    },
    {
        "id": "error_diagnosis_and_correction",
        "name": "纠正常见错误",
        "rank": 7,
        "teacher_action": "contrast",
        "strategies": ("error_correction", "contrastive_learning"),
        "cues": (
            "wrong",
            "error",
            "mistake",
            "fails",
            "failed",
            "fail",
            "incorrect",
            "debug",
            "goes wrong",
            "correct it",
            "surprising error",
            "错误",
            "纠正",
            "调试",
            "失败",
        ),
        "instruction": (
            "展示 {concept} 的一个典型错误或边界失效情形，与正确做法对比，追问是哪一个条件发生了变化。"
        ),
        "expected_signal": "能定位第一处出错的步骤，并说出失效条件",
        "fallback": "一次只改变一个条件，让学习者重新比较正确与错误两个版本。",
    },
    {
        "id": "practice_and_feedback",
        "name": "练习与反馈",
        "rank": 8,
        "teacher_action": "give_feedback",
        "strategies": ("practice_feedback_loop", "difficulty_progression"),
        "position_prior": "late",
        "cues": (
            # Deliberately not a bare "practice": a roadmap sentence such as
            # "problem solving through repeated practice" is not a practice phase.
            "practice on",
            "practice finding",
            "practice predicting",
            "practice with",
            "exercise",
            "try again",
            "feedback",
            "give a short",
            "revise",
            "练习",
            "反馈",
            "再试",
            "重做",
        ),
        "instruction": (
            "给出一道只考查 {concept} 单一关键点的练习，针对学习者的过程给出具体反馈，再让其修改后重做一次。"
        ),
        "expected_signal": "能根据反馈定位自己的问题并完成第二次尝试",
        "fallback": "把练习拆成两问，先只检查第一问，通过后再给第二问。",
    },
    {
        "id": "summary_and_transfer",
        "name": "总结和迁移",
        "rank": 9,
        "teacher_action": "summarize",
        "strategies": ("summary_and_transfer", "difficulty_progression"),
        "position_prior": "late",
        "cues": (
            "summarize",
            "summary",
            "takeaway",
            "transfer",
            "generalize the",
            "generalize",
            "new case",
            "new right-hand",
            "everyday task",
            "总结",
            "迁移",
            "推广",
            "小结",
        ),
        "instruction": (
            "请学习者用自己的话总结 {concept}、适用条件和一个不适用的情形，再把方法迁移到一个新情境。"
        ),
        "expected_signal": "总结同时包含定义、适用条件、反例与迁移用途",
        "fallback": "提供“它是什么—何时用—如何检验—何时失效”的句式框架，逐条补全。",
    },
)

PHASE_BY_ID: dict[str, dict[str, Any]] = {phase["id"]: phase for phase in CANONICAL_PHASES}
PHASE_IDS: tuple[str, ...] = tuple(phase["id"] for phase in CANONICAL_PHASES)


#: A phase whose canonical place in a lesson is violated needs corroboration
#: from more than one cue before it is accepted.  This is what stops
#: "review the proposed interfaces" in a closing feedback segment from being
#: read as 复习前置知识.
POSITION_PRIOR_WINDOW = 0.5
CORROBORATION_CUES_OUTSIDE_PRIOR = 2


def _matched_cues(text: str, phase: dict[str, Any]) -> list[str]:
    lowered = text.lower()
    return [cue for cue in phase["cues"] if cue.lower() in lowered]


def _violates_position_prior(phase: dict[str, Any], position: float | None) -> bool:
    prior = phase.get("position_prior")
    if prior is None or position is None:
        return False
    if prior == "early":
        return position > POSITION_PRIOR_WINDOW
    if prior == "late":
        return position < 1.0 - POSITION_PRIOR_WINDOW
    return False


def classify_segment(text: str, position: float | None = None) -> list[dict[str, Any]]:
    """Return every canonical phase whose cues appear in ``text``.

    ``position`` is the segment's normalised place in the lesson (0.0 at the
    first segment, 1.0 at the last).  Phases with a canonical position prior
    need corroborating cues when they appear outside that window.

    Results are ordered strongest-first.  A segment may legitimately evidence
    more than one phase (a teacher often demonstrates and checks in the same
    breath), so callers decide how many to keep.
    """

    hits: list[dict[str, Any]] = []
    for phase in CANONICAL_PHASES:
        cues = _matched_cues(text, phase)
        if not cues:
            continue
        outside_prior = _violates_position_prior(phase, position)
        if outside_prior and len(cues) < CORROBORATION_CUES_OUTSIDE_PRIOR:
            continue
        # Longer cues are more specific; use total matched cue length as a
        # tie-breaker so "line by line" outranks a bare "run the".
        specificity = sum(len(cue) for cue in cues)
        hits.append(
            {
                "phase_id": phase["id"],
                "matched_cues": cues,
                "hit_count": len(cues),
                "specificity": specificity,
                "rank": phase["rank"],
                "outside_position_prior": outside_prior,
            }
        )
    hits.sort(key=lambda item: (-item["hit_count"], -item["specificity"], item["rank"]))
    return hits


def detect_teaching_phases(
    transcript: dict[str, Any], *, max_phases_per_segment: int = 2
) -> dict[str, Any]:
    """Map a transcript's timeline onto the nine canonical teaching phases.

    Returns the observed phases in the teacher's actual temporal order, each
    bound to the segment indices, timestamps and cue tokens that produced it,
    plus the canonical phases that were never observed.
    """

    segments = transcript.get("segments", [])
    observed: dict[str, dict[str, Any]] = {}
    segment_assignments: list[dict[str, Any]] = []

    last_index = len(segments) - 1
    for index, segment in enumerate(segments):
        text = str(segment.get("text", ""))
        position = (index / last_index) if last_index > 0 else 0.0
        hits = classify_segment(text, position)[:max_phases_per_segment]
        segment_assignments.append(
            {
                "segment_index": index,
                "start": segment.get("start"),
                "end": segment.get("end"),
                "phase_ids": [hit["phase_id"] for hit in hits],
            }
        )
        for hit in hits:
            record = observed.setdefault(
                hit["phase_id"],
                {
                    "phase_id": hit["phase_id"],
                    "name": PHASE_BY_ID[hit["phase_id"]]["name"],
                    "rank": PHASE_BY_ID[hit["phase_id"]]["rank"],
                    "segment_indices": [],
                    "matched_cues": [],
                    "start": None,
                    "end": None,
                    "hit_count": 0,
                },
            )
            record["segment_indices"].append(index)
            record["hit_count"] += hit["hit_count"]
            for cue in hit["matched_cues"]:
                if cue not in record["matched_cues"]:
                    record["matched_cues"].append(cue)
            try:
                start = float(segment.get("start"))
                end = float(segment.get("end"))
            except (TypeError, ValueError):
                continue
            record["start"] = start if record["start"] is None else min(record["start"], start)
            record["end"] = end if record["end"] is None else max(record["end"], end)

    # Teacher order: first moment the phase appears on the timeline.  Ties fall
    # back to the canonical order so the result stays deterministic.
    def _first_start(record: dict[str, Any]) -> float:
        return float(record["start"]) if record["start"] is not None else float("inf")

    ordered = sorted(observed.values(), key=lambda item: (_first_start(item), item["rank"]))
    for position, record in enumerate(ordered, 1):
        record["observed_position"] = position

    unused = [
        {"phase_id": phase["id"], "name": phase["name"], "rank": phase["rank"]}
        for phase in CANONICAL_PHASES
        if phase["id"] not in observed
    ]

    total_segments = len(segments)
    covered_segments = sum(1 for item in segment_assignments if item["phase_ids"])
    return {
        "canonical_phase_count": len(CANONICAL_PHASES),
        "observed_phase_count": len(ordered),
        "observed_phase_sequence": [record["phase_id"] for record in ordered],
        "observed_phases": ordered,
        "unused_phases": unused,
        "segment_assignments": segment_assignments,
        "segment_count": total_segments,
        "phase_labelled_segment_count": covered_segments,
        "segment_coverage": (
            round(covered_segments / total_segments, 3) if total_segments else 0.0
        ),
        "detector": "canonical_phase_cue_v1",
        "detector_semantics": (
            "Cue-based mapping of transcript segments onto the nine canonical "
            "teaching phases; it records which teaching moves are visible on the "
            "timeline and does not establish recognition accuracy."
        ),
    }


def ordered_phase_plan(detection: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge observed phases with scaffolds for the phases never observed.

    Observed phases keep the teacher's temporal order.  Unobserved phases are
    slotted at their canonical position relative to that order rather than
    dumped at the end, so the emitted procedure stays executable end to end.
    """

    observed = {record["phase_id"]: record for record in detection.get("observed_phases", [])}
    observed_count = max(len(observed), 1)

    plan: list[tuple[float, int, dict[str, Any]]] = []
    for phase in CANONICAL_PHASES:
        record = observed.get(phase["id"])
        if record is not None:
            # Normalised position within the teacher's own sequence.
            key = (record["observed_position"] - 1) / observed_count
            entry = {"phase": phase, "origin": "observed_method", "observation": record}
        else:
            key = (phase["rank"] - 1) / len(CANONICAL_PHASES)
            entry = {"phase": phase, "origin": "recommended_enrichment", "observation": None}
        plan.append((key, phase["rank"], entry))

    plan.sort(key=lambda item: (item[0], item[1]))
    return [item[2] for item in plan]
