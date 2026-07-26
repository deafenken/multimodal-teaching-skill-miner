from __future__ import annotations

from typing import Any


def _fill(value: str, parameters: dict[str, str]) -> str:
    rendered = value
    for key, replacement in parameters.items():
        rendered = rendered.replace("{" + key + "}", replacement)
    return rendered


def execute_skill(skill: dict[str, Any], *, concept: str, learner_level: str = "beginner") -> str:
    parameters = {"concept": concept, "learner_level": learner_level}
    lines = [
        f"# 教学过程：{concept}",
        "",
        f"- 使用 Skill：`{skill['skill_id']}`（{skill['name']}）",
        f"- 学习者水平：{learner_level}",
        f"- 本轮目标：{_fill(skill['goal'], parameters)}",
        "",
        "## 执行步骤",
        "",
    ]
    for step in skill.get("procedure", []):
        lines.extend(
            [
                f"### {step['step']}. {step['teacher_action']}",
                "",
                _fill(step["instruction"], parameters),
                "",
                f"观察信号：{_fill(step['expected_signal'], parameters)}",
                "",
                f"动态分支：{_fill(step['fallback'], parameters)}",
                "",
            ]
        )
    lines.extend(["## 当堂验证", ""])
    for index, check in enumerate(skill.get("verification", []), 1):
        lines.extend(
            [
                f"{index}. **{check['type']}**：{_fill(check['prompt'], parameters)}",
                f"   - 通过条件：{_fill(check['pass_condition'], parameters)}",
            ]
        )
    lines.extend(
        [
            "",
            "## Agent 状态更新规则",
            "",
            "- 命中观察信号：进入下一步；验证通过后结束本轮并安排延迟复习。",
            "- 未命中观察信号：只执行当前步骤的动态分支，再次检查，不直接泄露完整答案。",
            "- 连续两次未命中：降低实例复杂度，并重新检查前置知识。",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"

