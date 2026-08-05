"""Shared, auditable diagnosis semantics for live teaching and benchmarks."""

from __future__ import annotations


def diagnosis_taxonomy_prompt(*, extended: bool) -> str:
    """Return one canonical grading rubric shared by product and benchmark."""

    lines = [
        "请按当前教师问题和明确判据区分回答：",
        "- correct：直接满足当前判据且无实质错误；满足约束的替代路径也属于正确；",
        "- partial：方向相关或含有正确信息，但缺少关键条件、依据、步骤，或没有直接回答本问；",
        "- misconception：学生明确陈述了可定位、可证伪的错误规则或错误观念；",
        "- confused：学生明确表示无法区分、不知道或不理解，但没有形成具体错误主张；",
        "- no_response：回答为空。",
    ]
    if extended:
        lines.extend(
            [
                "- off_topic：回答与当前教学问题无关；",
                "- valid_alternative：路径不同，但直接满足目标约束。",
            ]
        )
    lines.append(
        "不得把短回答、相邻概念或未命中预设词自动判为 confused；证据不足时保留不确定性并请求澄清。"
    )
    return "\n".join(lines)
