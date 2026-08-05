"""Cross-lecture distillation for a transferable Teaching Skill candidate.

This module deliberately uses a schema separate from the per-video Skill
schema.  A per-video Skill can cite transcript and multimodal evidence.  A
cross-lecture Skill must not pretend that those evidence identifiers belong to
one new video, so it retains only content-free hashes and balanced binary vote
counts.  No source URL, filesystem path, transcript quote, OCR text, or frame
payload is copied into the resulting artifact.

The resulting Skill is executable on a new concept, but remains a provisional
within-corpus consensus.  Its internal evaluation is a structural/support
audit; it is explicitly not recognition Accuracy or evidence of teaching
effectiveness.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from typing import Any, Iterable

from .executor import execute_skill
from .miner import STRATEGY_NAMES
from .models import ALLOWED_ACTIONS, ValidationResult, validate_skill
from .teaching_phases import CANONICAL_PHASES


GENERAL_SKILL_SCHEMA = "teaching_skill_miner.general_teaching_skill.v1"
GENERAL_SKILL_ARTIFACT_KIND = "cross_lesson_evidence_weighted_general_teaching_skill"
GENERAL_SKILL_STATUS = "heuristic_provisional"
CONSENSUS_ORIGIN = "cross_lecture_observed_consensus"
RECOMMENDED_ORIGIN = "recommended_enrichment"

DEFAULT_OVERALL_SUPPORT_THRESHOLD = 0.8
DEFAULT_PER_COURSE_SUPPORT_THRESHOLD = 0.6
DEFAULT_MINIMUM_COURSE_COUNT = 2
DEFAULT_MINIMUM_SKILLS_PER_COURSE = 5

_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_SAFE_STRATEGY_ID_RE = re.compile(r"[a-z0-9][a-z0-9_-]*")
_COURSE_REF_RE = re.compile(r"crs_[0-9a-f]{16}")
_LESSON_REF_RE = re.compile(r"lsn_[0-9a-f]{16}")
_PATH_OR_URL_RE = re.compile(
    r"(?:https?://|file://|(?:^|\s)/(?:Users|Volumes|home|data|tmp)/|[A-Za-z]:[\\/])"
)
_FORBIDDEN_CONTENT_KEYS = {
    "source_url",
    "quote",
    "ocr_text",
    "speech_quote",
    "frame_path",
    "before_frame",
    "after_frame",
    "anonymized_note",
    "transcript",
    "segments",
    "multimodal_evidence",
    "evidence_ids",
}

_TOP_LEVEL_FIELDS = {
    "schema",
    "artifact_kind",
    "status",
    "skill",
    "distillation",
    "claim_boundary",
}
_DISTILLATION_FIELDS = {
    "algorithm",
    "vote_unit",
    "eligible_origin",
    "input_skill_count",
    "input_course_count",
    "course_skill_counts",
    "thresholds",
    "source_skills",
    "source_collection_sha256",
    "execution_order_policy",
    "consensus_phase_count",
    "consensus_strategy_count",
}
_SOURCE_RECORD_FIELDS = {"course_ref", "lesson_ref", "skill_sha256"}
_THRESHOLD_FIELDS = {
    "overall_support_fraction",
    "per_course_support_fraction",
    "minimum_course_count",
    "minimum_skills_per_course",
}
_SUPPORT_FIELDS = {
    "observed_skill_count",
    "observed_fraction",
    "observed_course_count",
    "per_course_support",
    "source_skill_fingerprints",
}
_PER_COURSE_SUPPORT_FIELDS = {
    "observed_skill_count",
    "total_skill_count",
    "observed_fraction",
}
_SKILL_FIELDS = {
    "skill_id",
    "name",
    "version",
    "parameters",
    "learning_objective",
    "trigger",
    "preconditions",
    "goal",
    "strategies",
    "procedure",
    "teacher_actions",
    "student_signals",
    "success_criteria",
    "failure_modes",
    "verification",
}
_STRATEGY_FIELDS = {"id", "name", "origin"} | _SUPPORT_FIELDS
_PROCEDURE_FIELDS = {
    "step",
    "teaching_phase",
    "teaching_phase_name",
    "canonical_phase_rank",
    "teacher_action",
    "instruction",
    "expected_signal",
    "fallback",
    "normative_content_origin",
    "origin",
} | _SUPPORT_FIELDS
_CLAIM_BOUNDARY = {
    "neural_end_to_end_model_trained": False,
    "independent_phase_or_strategy_gold_used": False,
    "recognition_accuracy_established": False,
    "cross_course_generality_established": False,
    "population_generalization_established": False,
    "expert_consensus_established": False,
    "deployment_accuracy_established": False,
    "teaching_effectiveness_established": False,
    "recommended_counts_as_observed": False,
    "internal_score_is_accuracy": False,
}

_EVALUATION_WEIGHTS = {
    "schema_integrity": 0.20,
    "source_diversity": 0.15,
    "phase_consensus_support": 0.20,
    "strategy_consensus_support": 0.15,
    "canonical_executability": 0.20,
    "privacy_and_provenance": 0.10,
}


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _stable_ref(prefix: str, *values: str) -> str:
    payload = "\x00".join(values).encode("utf-8")
    return prefix + hashlib.sha256(payload).hexdigest()[:16]


def _fraction(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def _is_threshold_passed(
    observed_count: int,
    total_count: int,
    per_course_observed: dict[str, int],
    course_counts: dict[str, int],
    *,
    overall_threshold: float,
    per_course_threshold: float,
) -> bool:
    if not total_count or not course_counts:
        return False
    overall_passed = observed_count / total_count + 1e-12 >= overall_threshold
    course_passed = all(
        per_course_observed.get(course_ref, 0) / course_total + 1e-12
        >= per_course_threshold
        for course_ref, course_total in course_counts.items()
    )
    return overall_passed and course_passed


def _support_payload(
    supporting_records: list[dict[str, str]],
    all_records: list[dict[str, str]],
    course_counts: dict[str, int],
) -> dict[str, Any]:
    supporting_hashes = sorted(record["skill_sha256"] for record in supporting_records)
    per_course_observed = Counter(record["course_ref"] for record in supporting_records)
    return {
        "observed_skill_count": len(supporting_hashes),
        "observed_fraction": _fraction(len(supporting_hashes), len(all_records)),
        "observed_course_count": sum(
            per_course_observed.get(course_ref, 0) > 0 for course_ref in course_counts
        ),
        "per_course_support": {
            course_ref: {
                "observed_skill_count": per_course_observed.get(course_ref, 0),
                "total_skill_count": course_total,
                "observed_fraction": _fraction(
                    per_course_observed.get(course_ref, 0), course_total
                ),
            }
            for course_ref, course_total in sorted(course_counts.items())
        },
        "source_skill_fingerprints": supporting_hashes,
    }


def _validate_thresholds(
    *,
    overall_support_threshold: float,
    per_course_support_threshold: float,
    minimum_course_count: int,
    minimum_skills_per_course: int,
) -> None:
    for name, value in (
        ("overall_support_threshold", overall_support_threshold),
        ("per_course_support_threshold", per_course_support_threshold),
    ):
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0 < float(value) <= 1
        ):
            raise ValueError(f"{name} must be a finite number in (0, 1]")
    for name, value in (
        ("minimum_course_count", minimum_course_count),
        ("minimum_skills_per_course", minimum_skills_per_course),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be a positive integer")


def distill_general_skill(
    skills: Iterable[dict[str, Any]],
    *,
    overall_support_threshold: float = DEFAULT_OVERALL_SUPPORT_THRESHOLD,
    per_course_support_threshold: float = DEFAULT_PER_COURSE_SUPPORT_THRESHOLD,
    minimum_course_count: int = DEFAULT_MINIMUM_COURSE_COUNT,
    minimum_skills_per_course: int = DEFAULT_MINIMUM_SKILLS_PER_COURSE,
) -> dict[str, Any]:
    """Distil balanced cross-lecture consensus into one executable Skill.

    Every valid input lecture contributes at most one vote to a strategy or a
    canonical phase.  A vote is eligible only when the corresponding item is
    marked ``observed_method``.  Consensus requires both the overall threshold
    and the per-course threshold for *every* input course, preventing one
    course with many lectures from dominating the result.

    The function is deterministic with respect to input order.  It raises
    ``ValueError`` on invalid Skills, duplicate lessons, insufficient course
    coverage, or unsafe/unsupported strategy identifiers.
    """

    _validate_thresholds(
        overall_support_threshold=overall_support_threshold,
        per_course_support_threshold=per_course_support_threshold,
        minimum_course_count=minimum_course_count,
        minimum_skills_per_course=minimum_skills_per_course,
    )
    materialized = list(skills)
    if not materialized:
        raise ValueError("at least one per-video Skill is required")

    prepared: list[dict[str, Any]] = []
    seen_lessons: set[tuple[str, str]] = set()
    seen_fingerprints: set[str] = set()
    all_strategy_ids: set[str] = set()
    for index, skill in enumerate(materialized):
        validation = validate_skill(skill)
        if not validation.valid:
            details = "; ".join(validation.errors[:5])
            raise ValueError(f"skills[{index}] is invalid: {details}")
        source = skill["source"]
        course_id = str(source["course_id"])
        video_id = str(source["video_id"])
        lesson_key = (course_id, video_id)
        if lesson_key in seen_lessons:
            raise ValueError("duplicate course/video lesson in distillation input")
        seen_lessons.add(lesson_key)

        fingerprint = _sha256_json(skill)
        if fingerprint in seen_fingerprints:
            raise ValueError("duplicate Skill payload in distillation input")
        seen_fingerprints.add(fingerprint)
        course_ref = _stable_ref("crs_", course_id)
        lesson_ref = _stable_ref("lsn_", course_id, video_id)

        candidate_strategy_ids: set[str] = set()
        observed_strategy_ids: set[str] = set()
        for strategy in skill.get("strategies", []):
            strategy_id = str(strategy.get("id", ""))
            if not _SAFE_STRATEGY_ID_RE.fullmatch(strategy_id):
                raise ValueError(
                    f"skills[{index}] contains an unsafe strategy identifier"
                )
            candidate_strategy_ids.add(strategy_id)
            if strategy.get("origin") == "observed_method":
                observed_strategy_ids.add(strategy_id)
        all_strategy_ids.update(candidate_strategy_ids)

        observed_phase_ids = {
            str(step.get("teaching_phase"))
            for step in skill.get("procedure", [])
            if isinstance(step, dict) and step.get("origin") == "observed_method"
        }
        unknown_phase_ids = observed_phase_ids - {
            str(phase["id"]) for phase in CANONICAL_PHASES
        }
        if unknown_phase_ids:
            raise ValueError(
                "observed procedure contains unknown canonical phase: "
                + ", ".join(sorted(unknown_phase_ids))
            )

        prepared.append(
            {
                "course_ref": course_ref,
                "lesson_ref": lesson_ref,
                "skill_sha256": fingerprint,
                "observed_strategy_ids": observed_strategy_ids,
                "observed_phase_ids": observed_phase_ids,
            }
        )

    prepared.sort(
        key=lambda record: (
            record["course_ref"],
            record["lesson_ref"],
            record["skill_sha256"],
        )
    )
    course_counts = dict(
        sorted(Counter(record["course_ref"] for record in prepared).items())
    )
    if len(course_counts) < minimum_course_count:
        raise ValueError(
            f"distillation requires at least {minimum_course_count} courses"
        )
    undersized = {
        course_ref: count
        for course_ref, count in course_counts.items()
        if count < minimum_skills_per_course
    }
    if undersized:
        raise ValueError(
            "every input course must have at least "
            f"{minimum_skills_per_course} unique lectures"
        )

    source_records = [
        {
            "course_ref": record["course_ref"],
            "lesson_ref": record["lesson_ref"],
            "skill_sha256": record["skill_sha256"],
        }
        for record in prepared
    ]

    strategies: list[dict[str, Any]] = []
    for strategy_id in sorted(all_strategy_ids):
        supporting = [
            record
            for record in prepared
            if strategy_id in record["observed_strategy_ids"]
        ]
        support = _support_payload(supporting, prepared, course_counts)
        per_course_observed = {
            course_ref: values["observed_skill_count"]
            for course_ref, values in support["per_course_support"].items()
        }
        consensus = _is_threshold_passed(
            support["observed_skill_count"],
            len(prepared),
            per_course_observed,
            course_counts,
            overall_threshold=float(overall_support_threshold),
            per_course_threshold=float(per_course_support_threshold),
        )
        # The executable general method contains only techniques that passed
        # both balance gates.  Non-consensus candidates are not silently
        # promoted to generic recommendations; nine-phase procedure scaffolds
        # carry the separate recommended_enrichment semantics where needed.
        if consensus:
            strategies.append(
                {
                    "id": strategy_id,
                    "name": STRATEGY_NAMES.get(
                        strategy_id, strategy_id.replace("_", " ").strip().title()
                    ),
                    "origin": CONSENSUS_ORIGIN,
                    **support,
                }
            )

    procedure: list[dict[str, Any]] = []
    for phase in CANONICAL_PHASES:
        phase_id = str(phase["id"])
        supporting = [
            record for record in prepared if phase_id in record["observed_phase_ids"]
        ]
        support = _support_payload(supporting, prepared, course_counts)
        per_course_observed = {
            course_ref: values["observed_skill_count"]
            for course_ref, values in support["per_course_support"].items()
        }
        consensus = _is_threshold_passed(
            support["observed_skill_count"],
            len(prepared),
            per_course_observed,
            course_counts,
            overall_threshold=float(overall_support_threshold),
            per_course_threshold=float(per_course_support_threshold),
        )
        procedure.append(
            {
                "step": int(phase["rank"]),
                "teaching_phase": phase_id,
                "teaching_phase_name": str(phase["name"]),
                "canonical_phase_rank": int(phase["rank"]),
                "teacher_action": str(phase["teacher_action"]),
                "instruction": str(phase["instruction"]),
                "expected_signal": str(phase["expected_signal"]),
                "fallback": str(phase["fallback"]),
                "normative_content_origin": "CANONICAL_PHASES",
                "origin": CONSENSUS_ORIGIN if consensus else RECOMMENDED_ORIGIN,
                **support,
            }
        )

    teacher_actions: list[str] = []
    for phase in CANONICAL_PHASES:
        action = str(phase["teacher_action"])
        if action not in teacher_actions:
            teacher_actions.append(action)
    if "adapt" not in teacher_actions:
        teacher_actions.append("adapt")

    skill_payload: dict[str, Any] = {
        "skill_id": "evidence_grounded_adaptive_concept_teaching_v0",
        "name": "跨讲次证据共识驱动的自适应概念教学",
        "version": "0.1",
        "parameters": {
            "concept": "待教学的新概念",
            "learner_level": "beginner",
        },
        "learning_objective": {
            "statement": "学习者能够解释 {concept} 的关键结构，并迁移到一个新问题。",
            "bloom_level": "apply",
            "observable": True,
            "assessment": "通过口头解释、边界判断和近迁移任务观察。",
        },
        "trigger": [
            "需要为一个新概念生成可执行且可检查的教学过程",
            "学习者会模仿步骤，但尚不能解释步骤与概念之间的关系",
        ],
        "preconditions": [
            "已明确本轮待教学的 {concept} 与学习者水平",
            "能够提出一个检查必要前置知识的最小诊断问题",
        ],
        "goal": "帮助学习者建立 {concept} 的可解释心智模型，并能识别边界、完成近迁移。",
        "strategies": strategies,
        "procedure": procedure,
        "teacher_actions": teacher_actions,
        "student_signals": [
            "能用自己的话解释关键结构",
            "能指出正例与反例的决定性差异",
            "能在只提供分层提示时完成新问题的关键步骤",
        ],
        "success_criteria": [
            "学习者能解释关键步骤所依据的条件",
            "近迁移任务的关键步骤正确率达到 80%",
            "学习者能识别至少一个不适用情形并说明原因",
        ],
        "failure_modes": [
            {
                "mode": "实例复杂度过高",
                "mitigation": "减少变量，只保留一个待观察结构。",
            },
            {
                "mode": "直接给出形式定义而缺少直观映射",
                "mitigation": "返回最小具体实例并逐项建立对应。",
            },
            {
                "mode": "只检查答案而不检查理由",
                "mitigation": "追加依据追问与边界案例判断。",
            },
            {
                "mode": "学习者未达预期信号仍继续推进",
                "mitigation": "执行当前步骤的 fallback，降低难度后再次检查。",
            },
        ],
        "verification": [
            {
                "type": "near_transfer",
                "prompt": "换一个表面情境，使用 {concept} 完成关键步骤并解释依据。",
                "pass_condition": "关键步骤正确，且解释至少引用一个适用条件。",
            },
            {
                "type": "counterexample",
                "prompt": "判断一个边界案例是否适用 {concept}，若不适用请指出失效条件。",
                "pass_condition": "判断正确，并明确指出决定性的条件。",
            },
        ],
    }
    thresholds = {
        "overall_support_fraction": float(overall_support_threshold),
        "per_course_support_fraction": float(per_course_support_threshold),
        "minimum_course_count": minimum_course_count,
        "minimum_skills_per_course": minimum_skills_per_course,
    }
    artifact = {
        "schema": GENERAL_SKILL_SCHEMA,
        "artifact_kind": GENERAL_SKILL_ARTIFACT_KIND,
        "status": GENERAL_SKILL_STATUS,
        "skill": skill_payload,
        "distillation": {
            "algorithm": "balanced_binary_lecture_vote_v1",
            "vote_unit": "one_binary_vote_per_unique_course_lesson",
            "eligible_origin": "observed_method",
            "input_skill_count": len(prepared),
            "input_course_count": len(course_counts),
            "course_skill_counts": course_counts,
            "thresholds": thresholds,
            "source_skills": source_records,
            "source_collection_sha256": _sha256_json(source_records),
            "execution_order_policy": (
                "canonical_nine_phase_order_for_transfer; this order is a control "
                "policy and is not claimed to be the observed order of every lecture"
            ),
            "consensus_phase_count": sum(
                item["origin"] == CONSENSUS_ORIGIN for item in procedure
            ),
            "consensus_strategy_count": sum(
                item["origin"] == CONSENSUS_ORIGIN for item in strategies
            ),
        },
        "claim_boundary": dict(_CLAIM_BOUNDARY),
    }
    validation = validate_general_skill(artifact)
    if not validation.valid:
        raise RuntimeError(
            "generated general Skill failed validation: " + "; ".join(validation.errors)
        )
    return artifact


def _walk_for_privacy_issues(value: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(value, dict):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            if key in _FORBIDDEN_CONTENT_KEYS:
                errors.append(f"forbidden source-content field: {child_path}")
            errors.extend(_walk_for_privacy_issues(child, child_path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            errors.extend(_walk_for_privacy_issues(child, f"{path}[{index}]"))
    elif isinstance(value, str) and _PATH_OR_URL_RE.search(value):
        errors.append(f"path or URL disclosure at {path}")
    return errors


def _check_exact_fields(
    value: Any,
    expected: set[str],
    label: str,
    errors: list[str],
) -> bool:
    if not isinstance(value, dict):
        errors.append(f"{label} must be an object")
        return False
    actual = set(value)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing:
            errors.append(f"{label} missing fields: {', '.join(missing)}")
        if extra:
            errors.append(f"{label} has unsupported fields: {', '.join(extra)}")
        return False
    return True


def _validate_string_list(value: Any, label: str, errors: list[str]) -> None:
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item.strip() for item in value)
    ):
        errors.append(f"{label} must be a non-empty list of non-empty strings")


def _validate_support_fields(
    item: dict[str, Any],
    label: str,
    *,
    input_skill_count: int,
    course_counts: dict[str, int],
    fingerprint_to_course: dict[str, str],
    errors: list[str],
) -> bool:
    observed = item.get("observed_skill_count")
    if (
        isinstance(observed, bool)
        or not isinstance(observed, int)
        or not 0 <= observed <= input_skill_count
    ):
        errors.append(f"{label}.observed_skill_count is invalid")
        observed = 0
    observed_fraction = item.get("observed_fraction")
    expected_fraction = _fraction(observed, input_skill_count)
    if (
        isinstance(observed_fraction, bool)
        or not isinstance(observed_fraction, (int, float))
        or not math.isfinite(float(observed_fraction))
        or abs(float(observed_fraction) - expected_fraction) > 1e-9
    ):
        errors.append(f"{label}.observed_fraction does not match its count")

    fingerprints = item.get("source_skill_fingerprints")
    if (
        not isinstance(fingerprints, list)
        or len(fingerprints) != len(set(fingerprints))
        or fingerprints != sorted(fingerprints)
        or any(
            not isinstance(value, str) or not _SHA256_RE.fullmatch(value)
            for value in fingerprints
        )
        or any(value not in fingerprint_to_course for value in fingerprints)
    ):
        errors.append(f"{label}.source_skill_fingerprints is invalid")
        fingerprints = []
    if len(fingerprints) != observed:
        errors.append(f"{label}.source_skill_fingerprints count does not match support")

    per_course = item.get("per_course_support")
    per_course_observed: dict[str, int] = {}
    if not isinstance(per_course, dict) or set(per_course) != set(course_counts):
        errors.append(f"{label}.per_course_support must cover every input course")
        per_course = {}
    for course_ref, total in course_counts.items():
        record = per_course.get(course_ref)
        course_label = f"{label}.per_course_support.{course_ref}"
        if not _check_exact_fields(
            record, _PER_COURSE_SUPPORT_FIELDS, course_label, errors
        ):
            continue
        course_observed = record.get("observed_skill_count")
        if (
            isinstance(course_observed, bool)
            or not isinstance(course_observed, int)
            or not 0 <= course_observed <= total
        ):
            errors.append(f"{course_label}.observed_skill_count is invalid")
            continue
        per_course_observed[course_ref] = course_observed
        if record.get("total_skill_count") != total:
            errors.append(f"{course_label}.total_skill_count is inconsistent")
        expected_course_fraction = _fraction(course_observed, total)
        value = record.get("observed_fraction")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or abs(float(value) - expected_course_fraction) > 1e-9
        ):
            errors.append(f"{course_label}.observed_fraction is inconsistent")

        actual_from_hashes = sum(
            fingerprint_to_course.get(fingerprint) == course_ref
            for fingerprint in fingerprints
        )
        if actual_from_hashes != course_observed:
            errors.append(f"{course_label} does not match supporting fingerprints")

    observed_course_count = item.get("observed_course_count")
    expected_course_count = sum(value > 0 for value in per_course_observed.values())
    if observed_course_count != expected_course_count:
        errors.append(f"{label}.observed_course_count is inconsistent")
    if sum(per_course_observed.values()) != observed:
        errors.append(
            f"{label}.per_course_support counts do not sum to overall support"
        )
    return not any(error.startswith(label) for error in errors)


def validate_general_skill(value: dict[str, Any]) -> ValidationResult:
    """Validate the independent, content-free general Skill wrapper."""

    errors: list[str] = []
    warnings: list[str] = []
    if not _check_exact_fields(value, _TOP_LEVEL_FIELDS, "general_skill", errors):
        if not isinstance(value, dict):
            return ValidationResult(False, errors, warnings)

    if value.get("schema") != GENERAL_SKILL_SCHEMA:
        errors.append("general_skill.schema is unsupported")
    if value.get("artifact_kind") != GENERAL_SKILL_ARTIFACT_KIND:
        errors.append("general_skill.artifact_kind is unsupported")
    if value.get("status") != GENERAL_SKILL_STATUS:
        errors.append("general_skill.status must remain heuristic_provisional")

    distillation = value.get("distillation")
    if not _check_exact_fields(
        distillation, _DISTILLATION_FIELDS, "distillation", errors
    ):
        distillation = distillation if isinstance(distillation, dict) else {}
    if distillation.get("algorithm") != "balanced_binary_lecture_vote_v1":
        errors.append("distillation.algorithm is unsupported")
    if distillation.get("vote_unit") != "one_binary_vote_per_unique_course_lesson":
        errors.append("distillation.vote_unit must be per unique lecture")
    if distillation.get("eligible_origin") != "observed_method":
        errors.append("distillation.eligible_origin must be observed_method")
    if distillation.get("execution_order_policy") != (
        "canonical_nine_phase_order_for_transfer; this order is a control "
        "policy and is not claimed to be the observed order of every lecture"
    ):
        errors.append("distillation.execution_order_policy is invalid")

    input_skill_count = distillation.get("input_skill_count")
    input_course_count = distillation.get("input_course_count")
    if (
        isinstance(input_skill_count, bool)
        or not isinstance(input_skill_count, int)
        or input_skill_count < 1
    ):
        errors.append("distillation.input_skill_count is invalid")
        input_skill_count = 0
    if (
        isinstance(input_course_count, bool)
        or not isinstance(input_course_count, int)
        or input_course_count < 1
    ):
        errors.append("distillation.input_course_count is invalid")
        input_course_count = 0

    course_counts = distillation.get("course_skill_counts")
    if (
        not isinstance(course_counts, dict)
        or not course_counts
        or any(
            not isinstance(key, str)
            or not _COURSE_REF_RE.fullmatch(key)
            or isinstance(count, bool)
            or not isinstance(count, int)
            or count < 1
            for key, count in course_counts.items()
        )
    ):
        errors.append("distillation.course_skill_counts is invalid")
        course_counts = {}
    else:
        course_counts = dict(sorted(course_counts.items()))
        if len(course_counts) != input_course_count:
            errors.append("distillation.input_course_count is inconsistent")
        if sum(course_counts.values()) != input_skill_count:
            errors.append("distillation.input_skill_count is inconsistent")

    thresholds = distillation.get("thresholds")
    if not _check_exact_fields(
        thresholds, _THRESHOLD_FIELDS, "distillation.thresholds", errors
    ):
        thresholds = thresholds if isinstance(thresholds, dict) else {}
    try:
        _validate_thresholds(
            overall_support_threshold=thresholds.get("overall_support_fraction"),
            per_course_support_threshold=thresholds.get("per_course_support_fraction"),
            minimum_course_count=thresholds.get("minimum_course_count"),
            minimum_skills_per_course=thresholds.get("minimum_skills_per_course"),
        )
    except ValueError as exc:
        errors.append(f"distillation.thresholds invalid: {exc}")
    if (
        isinstance(thresholds.get("minimum_course_count"), int)
        and input_course_count < thresholds["minimum_course_count"]
    ):
        errors.append("distillation has fewer courses than its declared minimum")
    if isinstance(thresholds.get("minimum_skills_per_course"), int) and any(
        count < thresholds["minimum_skills_per_course"]
        for count in course_counts.values()
    ):
        errors.append("distillation has an undersized course")

    source_records = distillation.get("source_skills")
    fingerprint_to_course: dict[str, str] = {}
    if not isinstance(source_records, list) or len(source_records) != input_skill_count:
        errors.append("distillation.source_skills count is inconsistent")
        source_records = []
    expected_source_sort: list[tuple[str, str, str]] = []
    seen_lessons: set[tuple[str, str]] = set()
    for index, record in enumerate(source_records):
        label = f"distillation.source_skills[{index}]"
        if not _check_exact_fields(record, _SOURCE_RECORD_FIELDS, label, errors):
            continue
        course_ref = record.get("course_ref")
        lesson_ref = record.get("lesson_ref")
        fingerprint = record.get("skill_sha256")
        if (
            not isinstance(course_ref, str)
            or not _COURSE_REF_RE.fullmatch(course_ref)
            or course_ref not in course_counts
        ):
            errors.append(f"{label}.course_ref is invalid")
        if not isinstance(lesson_ref, str) or not _LESSON_REF_RE.fullmatch(lesson_ref):
            errors.append(f"{label}.lesson_ref is invalid")
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            errors.append(f"{label}.skill_sha256 is invalid")
            continue
        if fingerprint in fingerprint_to_course:
            errors.append(f"{label}.skill_sha256 is duplicated")
        else:
            fingerprint_to_course[fingerprint] = str(course_ref)
        lesson_key = (str(course_ref), str(lesson_ref))
        if lesson_key in seen_lessons:
            errors.append(f"{label} duplicates a lesson")
        seen_lessons.add(lesson_key)
        expected_source_sort.append((str(course_ref), str(lesson_ref), fingerprint))
    if expected_source_sort != sorted(expected_source_sort):
        errors.append("distillation.source_skills must use deterministic order")
    if source_records and distillation.get("source_collection_sha256") != _sha256_json(
        source_records
    ):
        errors.append(
            "distillation.source_collection_sha256 does not match source_skills"
        )

    skill = value.get("skill")
    if not _check_exact_fields(skill, _SKILL_FIELDS, "skill", errors):
        skill = skill if isinstance(skill, dict) else {}
    if skill.get("skill_id") != "evidence_grounded_adaptive_concept_teaching_v0":
        errors.append("skill.skill_id is invalid")
    if skill.get("name") != "跨讲次证据共识驱动的自适应概念教学":
        errors.append("skill.name is invalid")
    if skill.get("version") != "0.1":
        errors.append("skill.version is invalid")
    parameters = skill.get("parameters")
    if parameters != {"concept": "待教学的新概念", "learner_level": "beginner"}:
        errors.append("skill.parameters is invalid")
    objective = skill.get("learning_objective")
    if not isinstance(objective, dict) or set(objective) != {
        "statement",
        "bloom_level",
        "observable",
        "assessment",
    }:
        errors.append("skill.learning_objective is invalid")
    elif (
        objective.get("observable") is not True
        or objective.get("bloom_level") != "apply"
    ):
        errors.append("skill.learning_objective must be observable at apply level")
    for field in ("trigger", "preconditions", "student_signals", "success_criteria"):
        _validate_string_list(skill.get(field), f"skill.{field}", errors)
    if not isinstance(skill.get("goal"), str) or "{concept}" not in skill.get(
        "goal", ""
    ):
        errors.append("skill.goal must be parameterized by {concept}")

    strategies = skill.get("strategies")
    if not isinstance(strategies, list) or not strategies:
        errors.append("skill.strategies must be a non-empty list")
        strategies = []
    strategy_ids: list[str] = []
    consensus_strategy_count = 0
    for index, strategy in enumerate(strategies):
        label = f"skill.strategies[{index}]"
        if not _check_exact_fields(strategy, _STRATEGY_FIELDS, label, errors):
            continue
        strategy_id = strategy.get("id")
        if not isinstance(strategy_id, str) or not _SAFE_STRATEGY_ID_RE.fullmatch(
            strategy_id
        ):
            errors.append(f"{label}.id is invalid")
        else:
            strategy_ids.append(strategy_id)
        if not isinstance(strategy.get("name"), str) or not strategy["name"].strip():
            errors.append(f"{label}.name is invalid")
        _validate_support_fields(
            strategy,
            label,
            input_skill_count=input_skill_count,
            course_counts=course_counts,
            fingerprint_to_course=fingerprint_to_course,
            errors=errors,
        )
        threshold_passed = _support_passes_from_artifact(
            strategy, course_counts, thresholds
        )
        expected_origin = CONSENSUS_ORIGIN if threshold_passed else RECOMMENDED_ORIGIN
        if strategy.get("origin") != expected_origin:
            errors.append(f"{label}.origin does not match balanced support")
        if expected_origin == CONSENSUS_ORIGIN:
            consensus_strategy_count += 1
    if strategy_ids != sorted(strategy_ids) or len(strategy_ids) != len(
        set(strategy_ids)
    ):
        errors.append("skill.strategies must have unique ids in deterministic order")

    procedure = skill.get("procedure")
    if not isinstance(procedure, list) or len(procedure) != len(CANONICAL_PHASES):
        errors.append("skill.procedure must contain all nine canonical phases")
        procedure = []
    consensus_phase_count = 0
    for index, phase in enumerate(CANONICAL_PHASES):
        if index >= len(procedure):
            break
        step = procedure[index]
        label = f"skill.procedure[{index}]"
        if not _check_exact_fields(step, _PROCEDURE_FIELDS, label, errors):
            continue
        expected_static = {
            "step": int(phase["rank"]),
            "teaching_phase": str(phase["id"]),
            "teaching_phase_name": str(phase["name"]),
            "canonical_phase_rank": int(phase["rank"]),
            "teacher_action": str(phase["teacher_action"]),
            "instruction": str(phase["instruction"]),
            "expected_signal": str(phase["expected_signal"]),
            "fallback": str(phase["fallback"]),
            "normative_content_origin": "CANONICAL_PHASES",
        }
        for field, expected in expected_static.items():
            if step.get(field) != expected:
                errors.append(f"{label}.{field} does not match canonical phase")
        _validate_support_fields(
            step,
            label,
            input_skill_count=input_skill_count,
            course_counts=course_counts,
            fingerprint_to_course=fingerprint_to_course,
            errors=errors,
        )
        threshold_passed = _support_passes_from_artifact(
            step, course_counts, thresholds
        )
        expected_origin = CONSENSUS_ORIGIN if threshold_passed else RECOMMENDED_ORIGIN
        if step.get("origin") != expected_origin:
            errors.append(f"{label}.origin does not match balanced support")
        if expected_origin == CONSENSUS_ORIGIN:
            consensus_phase_count += 1

    if distillation.get("consensus_phase_count") != consensus_phase_count:
        errors.append("distillation.consensus_phase_count is inconsistent")
    if distillation.get("consensus_strategy_count") != consensus_strategy_count:
        errors.append("distillation.consensus_strategy_count is inconsistent")

    actions = skill.get("teacher_actions")
    if (
        not isinstance(actions, list)
        or not actions
        or len(actions) != len(set(actions))
        or any(action not in ALLOWED_ACTIONS for action in actions)
    ):
        errors.append("skill.teacher_actions is invalid")
    failure_modes = skill.get("failure_modes")
    if (
        not isinstance(failure_modes, list)
        or not failure_modes
        or any(
            not isinstance(item, dict)
            or set(item) != {"mode", "mitigation"}
            or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in item
            )
            for item in failure_modes
        )
    ):
        errors.append("skill.failure_modes is invalid")
    verification = skill.get("verification")
    if (
        not isinstance(verification, list)
        or len(verification) < 2
        or any(
            not isinstance(item, dict)
            or set(item) != {"type", "prompt", "pass_condition"}
            or any(
                not isinstance(item.get(key), str) or not item[key].strip()
                for key in item
            )
            for item in verification
        )
    ):
        errors.append("skill.verification is invalid")

    if value.get("claim_boundary") != _CLAIM_BOUNDARY:
        errors.append("claim_boundary must retain every negative claim boundary")
    errors.extend(_walk_for_privacy_issues(value))
    if consensus_phase_count < len(CANONICAL_PHASES):
        warnings.append(
            "some canonical phases are recommended enrichment, not observed consensus"
        )
    if consensus_strategy_count == 0:
        warnings.append("no strategy passed balanced cross-lecture consensus")
    return ValidationResult(not errors, errors, warnings)


def _support_passes_from_artifact(
    item: dict[str, Any],
    course_counts: dict[str, int],
    thresholds: dict[str, Any],
) -> bool:
    try:
        per_course_observed = {
            course_ref: int(
                item["per_course_support"][course_ref]["observed_skill_count"]
            )
            for course_ref in course_counts
        }
        return _is_threshold_passed(
            int(item["observed_skill_count"]),
            sum(course_counts.values()),
            per_course_observed,
            course_counts,
            overall_threshold=float(thresholds["overall_support_fraction"]),
            per_course_threshold=float(thresholds["per_course_support_fraction"]),
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return False


def evaluate_general_skill(value: dict[str, Any]) -> dict[str, Any]:
    """Return an internal support/executability audit, never ``Accuracy``."""

    validation = validate_general_skill(value)
    skill = value.get("skill", {}) if isinstance(value, dict) else {}
    distillation = value.get("distillation", {}) if isinstance(value, dict) else {}
    procedure = skill.get("procedure", []) if isinstance(skill, dict) else []
    strategies = skill.get("strategies", []) if isinstance(skill, dict) else []
    phase_consensus_count = sum(
        isinstance(item, dict) and item.get("origin") == CONSENSUS_ORIGIN
        for item in procedure
    )
    strategy_consensus_count = sum(
        isinstance(item, dict) and item.get("origin") == CONSENSUS_ORIGIN
        for item in strategies
    )

    thresholds = (
        distillation.get("thresholds", {}) if isinstance(distillation, dict) else {}
    )
    course_counts = (
        distillation.get("course_skill_counts", {})
        if isinstance(distillation, dict)
        else {}
    )
    course_count = (
        distillation.get("input_course_count", 0)
        if isinstance(distillation, dict)
        else 0
    )
    skill_count = (
        distillation.get("input_skill_count", 0)
        if isinstance(distillation, dict)
        else 0
    )
    minimum_courses = thresholds.get(
        "minimum_course_count", DEFAULT_MINIMUM_COURSE_COUNT
    )
    minimum_per_course = thresholds.get(
        "minimum_skills_per_course", DEFAULT_MINIMUM_SKILLS_PER_COURSE
    )

    source_diversity_checks = [
        isinstance(course_count, int) and course_count >= minimum_courses,
        isinstance(skill_count, int)
        and skill_count >= minimum_courses * minimum_per_course,
        isinstance(course_counts, dict)
        and bool(course_counts)
        and all(
            isinstance(count, int) and count >= minimum_per_course
            for count in course_counts.values()
        ),
    ]
    phase_support_score = round(100 * phase_consensus_count / len(CANONICAL_PHASES), 1)
    strategy_support_score = (
        round(
            100
            * sum(float(item.get("observed_fraction", 0.0)) for item in strategies)
            / len(strategies),
            1,
        )
        if strategies
        else 0.0
    )
    canonical_checks = [
        len(procedure) == len(CANONICAL_PHASES),
        all(
            isinstance(step, dict)
            and step.get("step") == index
            and step.get("teaching_phase") == CANONICAL_PHASES[index - 1]["id"]
            and all(
                isinstance(step.get(field), str) and step[field].strip()
                for field in ("instruction", "expected_signal", "fallback")
            )
            for index, step in enumerate(procedure, 1)
        ),
        isinstance(skill, dict) and "{concept}" in str(skill.get("goal", "")),
        isinstance(skill, dict) and len(skill.get("verification", [])) >= 2,
    ]
    source_records = (
        distillation.get("source_skills", []) if isinstance(distillation, dict) else []
    )
    provenance_checks = [
        isinstance(source_records, list)
        and bool(source_records)
        and distillation.get("source_collection_sha256")
        == _sha256_json(source_records),
        not _walk_for_privacy_issues(value),
        value.get("claim_boundary") == _CLAIM_BOUNDARY
        if isinstance(value, dict)
        else False,
    ]
    dimensions = {
        "schema_integrity": 100.0 if validation.valid else 0.0,
        "source_diversity": round(
            100 * sum(source_diversity_checks) / len(source_diversity_checks), 1
        ),
        "phase_consensus_support": phase_support_score,
        "strategy_consensus_support": strategy_support_score,
        "canonical_executability": round(
            100 * sum(canonical_checks) / len(canonical_checks), 1
        ),
        "privacy_and_provenance": round(
            100 * sum(provenance_checks) / len(provenance_checks), 1
        ),
    }
    overall_score = round(
        sum(dimensions[name] * weight for name, weight in _EVALUATION_WEIGHTS.items()),
        1,
    )
    gates = {
        "schema_valid": validation.valid,
        "minimum_source_diversity_met": all(source_diversity_checks),
        "at_least_three_consensus_phases": phase_consensus_count >= 3,
        "at_least_three_consensus_strategies": strategy_consensus_count >= 3,
        "all_nine_phases_executable": all(canonical_checks),
        "content_free_provenance_bound": all(provenance_checks),
        "parameterized_for_new_concept": isinstance(skill, dict)
        and "{concept}" in str(skill.get("goal", "")),
    }
    return {
        "schema": "teaching_skill_miner.general_skill_evaluation.v1",
        "evaluation_kind": "internal_structural_support_and_executability_audit",
        "general_skill_sha256": _sha256_json(value),
        "metric_name": "internal_general_skill_readiness_score",
        "overall_score": overall_score,
        "score_semantics": (
            "This score audits schema integrity, balanced within-corpus lecture support, "
            "canonical executability, privacy, and hash provenance. It is not recognition "
            "Accuracy, deployment accuracy, external generalization, or a learning-effect estimate."
        ),
        "dimensions": dimensions,
        "dimension_weights": dict(_EVALUATION_WEIGHTS),
        "consensus_phase_count": phase_consensus_count,
        "recommended_phase_count": max(
            0, len(CANONICAL_PHASES) - phase_consensus_count
        ),
        "consensus_strategy_count": strategy_consensus_count,
        "recommended_strategy_count": max(
            0, len(strategies) - strategy_consensus_count
        ),
        "gates": gates,
        "validation": validation.as_dict(),
        "claim_boundary": dict(_CLAIM_BOUNDARY),
        "passed": all(gates.values()),
    }


def execute_general_skill(
    value: dict[str, Any],
    *,
    concept: str,
    learner_level: str = "beginner",
) -> str:
    """Validate and execute the distilled Skill for a new concept."""

    validation = validate_general_skill(value)
    if not validation.valid:
        raise ValueError("invalid general Skill: " + "; ".join(validation.errors))
    if not isinstance(concept, str) or not concept.strip():
        raise ValueError("concept must be a non-empty string")
    if not isinstance(learner_level, str) or not learner_level.strip():
        raise ValueError("learner_level must be a non-empty string")
    rendered = execute_skill(
        value["skill"],
        concept=concept.strip(),
        learner_level=learner_level.strip(),
    )
    for step in value["skill"]["procedure"]:
        machine_heading = f"### {step['step']}. {step['teacher_action']}"
        readable_heading = (
            f"### {step['step']}. {step['teaching_phase_name']}"
            f"（{step['teacher_action']}）"
        )
        rendered = rendered.replace(machine_heading, readable_heading, 1)
    boundary = (
        "\n## 证据边界\n\n"
        "- 共识标签只表示这些方法在输入讲次中达到整体与逐课程门槛。\n"
        "- 补充步骤未达到双门槛，其措辞来自规范模板，不能称为跨课观察共识。\n"
        "- 本过程尚未证明外部泛化、部署准确率或学习效果。\n"
    )
    return rendered.rstrip() + "\n" + boundary


def render_general_skill_summary(value: dict[str, Any]) -> str:
    """Render a content-free, human-readable audit summary."""

    validation = validate_general_skill(value)
    if not validation.valid:
        raise ValueError("invalid general Skill: " + "; ".join(validation.errors))
    skill = value["skill"]
    distillation = value["distillation"]
    thresholds = distillation["thresholds"]
    evaluation = evaluate_general_skill(value)
    lines = [
        "# 通用 Teaching Skill 摘要",
        "",
        f"- Skill：`{skill['skill_id']}`（{skill['name']}）",
        f"- 状态：`{value['status']}`",
        (
            "- 输入范围："
            f"{distillation['input_course_count']} 门课程 / "
            f"{distillation['input_skill_count']} 个独立讲次"
        ),
        (
            "- 共识门槛：整体支持率 ≥ "
            f"{thresholds['overall_support_fraction']:.0%}，且每门课支持率 ≥ "
            f"{thresholds['per_course_support_fraction']:.0%}"
        ),
        (
            "- 内部 readiness："
            f"{evaluation['overall_score']:.1f} / 100"
            f"（{'通过' if evaluation['passed'] else '未通过'}；"
            "不是 Accuracy）"
        ),
        "",
        "## 跨课程共识策略",
        "",
        "| 策略 | 讲次支持 | 课程覆盖 |",
        "|---|---:|---:|",
    ]
    for strategy in skill["strategies"]:
        lines.append(
            f"| {strategy['name']} (`{strategy['id']}`) | "
            f"{strategy['observed_skill_count']}/{distillation['input_skill_count']} | "
            f"{strategy['observed_course_count']}/{distillation['input_course_count']} |"
        )
    lines.extend(
        [
            "",
            "## 九阶段执行策略",
            "",
            "| 步骤 | 教学环节 | 讲次支持 | 来源语义 |",
            "|---:|---|---:|---|",
        ]
    )
    for step in skill["procedure"]:
        origin = (
            "跨讲次观察共识"
            if step["origin"] == CONSENSUS_ORIGIN
            else "规范补充（未达双门槛）"
        )
        lines.append(
            f"| {step['step']} | {step['teaching_phase_name']} | "
            f"{step['observed_skill_count']}/{distillation['input_skill_count']} | {origin} |"
        )
    lines.extend(
        [
            "",
            "策略是讲次内可多次出现的局部方法线索，阶段是具有更严格线索与位置约束的教学环节；"
            "两者标签粒度不同，不能互换票数。",
            "",
            "九阶段顺序来自规范执行策略，不声称十个讲次采用了同一顺序；"
            "每讲的真实观察顺序仍保存在对应单讲 Skill 中。",
            "",
            "## 结论边界",
            "",
            "- 当前产物是可执行的启发式跨讲次候选，不是已训练的神经端到端模型。",
            "- 跨讲次支持度只描述本次输入语料，不是识别准确率、专家共识或总体泛化证明。",
            "- 自动 readiness 只审计结构、支持、可执行性、隐私与哈希溯源，不证明学习效果。",
            "- 独立核对来源内容时仍须持有十份原始单讲 Skill，并重新计算各自指纹。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def build_general_skill_receipt(
    general_skill: dict[str, Any],
    evaluation: dict[str, Any],
) -> dict[str, Any]:
    """Build a deterministic, content-free receipt for public verification."""

    validation = validate_general_skill(general_skill)
    expected_evaluation = evaluate_general_skill(general_skill)
    if evaluation != expected_evaluation:
        raise ValueError("evaluation does not match the supplied general Skill")
    return {
        "schema": "teaching_skill_miner.general_skill_receipt.v1",
        "artifact_kind": "content_free_general_skill_distillation_receipt",
        "general_skill_sha256": _sha256_json(general_skill),
        "evaluation_sha256": _sha256_json(evaluation),
        "source_collection_sha256": general_skill.get("distillation", {}).get(
            "source_collection_sha256"
        ),
        "input_skill_count": general_skill.get("distillation", {}).get(
            "input_skill_count"
        ),
        "input_course_count": general_skill.get("distillation", {}).get(
            "input_course_count"
        ),
        "consensus_phase_count": evaluation.get("consensus_phase_count"),
        "consensus_strategy_count": evaluation.get("consensus_strategy_count"),
        "validation_passed": validation.valid,
        "evaluation_passed": evaluation.get("passed") is True,
        "metric_name": "internal_general_skill_readiness_score",
        "metric_value": evaluation.get("overall_score"),
        "metric_is_accuracy": False,
        "neural_end_to_end_model_trained": False,
        "recognition_accuracy_established": False,
        "external_generality_established": False,
        "teaching_effectiveness_established": False,
        "recommended_counts_as_observed": False,
    }


__all__ = [
    "CONSENSUS_ORIGIN",
    "DEFAULT_MINIMUM_COURSE_COUNT",
    "DEFAULT_MINIMUM_SKILLS_PER_COURSE",
    "DEFAULT_OVERALL_SUPPORT_THRESHOLD",
    "DEFAULT_PER_COURSE_SUPPORT_THRESHOLD",
    "GENERAL_SKILL_SCHEMA",
    "RECOMMENDED_ORIGIN",
    "build_general_skill_receipt",
    "distill_general_skill",
    "evaluate_general_skill",
    "execute_general_skill",
    "render_general_skill_summary",
    "validate_general_skill",
]
