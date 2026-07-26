from __future__ import annotations

from statistics import mean
from typing import Any

from .executor import execute_skill
from .runtime import SkillRuntime


CAPABILITY_WEIGHTS = {
    "diagnostic_check": 1.0,
    "concrete_or_visual_example": 1.0,
    "explicit_formal_mapping": 1.0,
    "boundary_contrast": 1.0,
    "adaptive_fallback": 1.0,
    "near_transfer": 1.0,
    "learner_summary": 1.0,
    "signal_gating": 1.0,
}

BASELINE_CAPABILITIES = {
    "concrete_or_visual_example",
    "explicit_formal_mapping",
    "learner_summary",
}


def skill_capabilities(skill: dict[str, Any]) -> set[str]:
    procedure = skill.get("procedure", [])
    actions = {step.get("teacher_action") for step in procedure if isinstance(step, dict)}
    instructions = " ".join(str(step.get("instruction", "")) for step in procedure if isinstance(step, dict))
    capabilities: set[str] = set()
    if procedure and procedure[0].get("teacher_action") == "ask":
        capabilities.add("diagnostic_check")
    if "demonstrate" in actions or "实例" in instructions or "例子" in instructions:
        capabilities.add("concrete_or_visual_example")
    if "explain" in actions and "映射" in instructions:
        capabilities.add("explicit_formal_mapping")
    if "contrast" in actions and ("边界" in instructions or "错误" in instructions):
        capabilities.add("boundary_contrast")
    if procedure and all(step.get("fallback") for step in procedure if isinstance(step, dict)):
        capabilities.add("adaptive_fallback")
    if any(item.get("type") == "near_transfer" for item in skill.get("verification", []) if isinstance(item, dict)):
        capabilities.add("near_transfer")
    if "summarize" in actions:
        capabilities.add("learner_summary")
    if procedure and all(step.get("expected_signal") for step in procedure if isinstance(step, dict)):
        capabilities.add("signal_gating")
    return capabilities


def _score(required: list[str], available: set[str]) -> float:
    denominator = sum(CAPABILITY_WEIGHTS.get(item, 1.0) for item in required)
    numerator = sum(CAPABILITY_WEIGHTS.get(item, 1.0) for item in required if item in available)
    return round(100 * numerator / denominator, 1) if denominator else 0.0


def _choose_skill(skills: list[dict[str, Any]], preferred_strategy: str) -> dict[str, Any]:
    ranked: list[tuple[int, float, dict[str, Any]]] = []
    for skill in skills:
        confidence = max(
            (
                float(strategy.get("confidence", 0))
                for strategy in skill.get("strategies", [])
                if strategy.get("id") == preferred_strategy
            ),
            default=-1.0,
        )
        primary_match = int(bool(skill.get("strategies")) and skill["strategies"][0].get("id") == preferred_strategy)
        ranked.append((primary_match, confidence, skill))
    ranked.sort(key=lambda item: (-item[0], -item[1], str(item[2].get("skill_id", ""))))
    return ranked[0][2]


def benchmark_transfer(skills: list[dict[str, Any]], cases: list[dict[str, Any]]) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for case in cases:
        skill = _choose_skill(skills, case.get("preferred_strategy", ""))
        required = case.get("required_capabilities", list(CAPABILITY_WEIGHTS))
        capabilities = skill_capabilities(skill)
        capability_score = _score(required, capabilities)
        strategy_aligned = any(
            item.get("id") == case.get("preferred_strategy")
            for item in skill.get("strategies", [])
            if isinstance(item, dict)
        )
        skill_score = round(0.85 * capability_score + 15 * strategy_aligned, 1)
        baseline_score = _score(required, BASELINE_CAPABILITIES)
        lesson = execute_skill(
            skill,
            concept=case["concept"],
            learner_level=case.get("learner_level", "beginner"),
        )
        runtime = SkillRuntime(skill, concept=case["concept"], learner_level=case.get("learner_level", "beginner"))
        retry = runtime.observe(case.get("weak_response", "我不确定。"), "not_achieved")
        recovery = runtime.observe(case.get("recovered_response", "我能说明理由并给出例子。"), "achieved")
        source_title = str(skill.get("source", {}).get("title", "")).lower()
        topic_leakage = case["concept"].lower() in source_title
        results.append(
            {
                "case_id": case["case_id"],
                "domain": case.get("domain"),
                "concept": case["concept"],
                "selected_skill_id": skill["skill_id"],
                "preferred_strategy": case.get("preferred_strategy"),
                "required_capabilities": required,
                "detected_capabilities": sorted(capabilities),
                "strategy_aligned": strategy_aligned,
                "skill_score": skill_score,
                "static_baseline_score": baseline_score,
                "delta": round(skill_score - baseline_score, 1),
                "concept_parameterized": case["concept"] in lesson and "{concept}" not in lesson,
                "source_topic_leakage": topic_leakage,
                "fallback_triggered": retry["event"]["transition"] == "retry" and bool(retry["event"].get("fallback_message")),
                "recovered_and_advanced": recovery["event"]["transition"] == "advance",
            }
        )
    gates = {
        "all_cases_parameterized": all(item["concept_parameterized"] for item in results),
        "all_preferred_strategies_aligned": all(item["strategy_aligned"] for item in results),
        "no_source_topic_leakage": not any(item["source_topic_leakage"] for item in results),
        "all_cases_trigger_fallback": all(item["fallback_triggered"] for item in results),
        "all_cases_recover": all(item["recovered_and_advanced"] for item in results),
        "mean_skill_score_at_least_80": bool(results) and mean(item["skill_score"] for item in results) >= 80,
    }
    skill_mean = round(mean(item["skill_score"] for item in results), 1) if results else 0.0
    baseline_mean = round(mean(item["static_baseline_score"] for item in results), 1) if results else 0.0
    return {
        "benchmark_type": "deterministic_capability_and_runtime_transfer",
        "interpretation_limit": "该分数衡量可执行能力覆盖和状态机行为，不代表真实学生学习增益。",
        "case_count": len(results),
        "skill_mean": skill_mean,
        "static_baseline_mean": baseline_mean,
        "mean_delta": round(skill_mean - baseline_mean, 1),
        "gates": gates,
        "passed": all(gates.values()),
        "cases": results,
    }
