from __future__ import annotations

import hashlib

import pytest

from teaching_skill_miner.teacher_agent_accessibility import (
    AccessibilityError,
    accessible_teacher_rendering,
    build_accessibility_contract,
    repair_accessible_teacher_message,
)


def test_paired_profiles_enable_constraints_without_lowering_mastery_standard() -> None:
    baseline = build_accessibility_contract({"accessibility_needs": []})
    supported = build_accessibility_contract(
        {
            "accessibility_needs": [
                "请用短句并一次一问",
                "使用屏幕阅读器 VoiceOver",
                "色弱：不依赖颜色",
                "听障：需要文字稿",
                "第二语言学习者，请解释术语",
            ]
        }
    )

    assert not any(baseline["enabled"].values())
    assert all(supported["enabled"].values())
    assert supported["mastery_standard_lowered"] is False
    assert supported["assessment_construct_changed"] is False
    assert "at_most_one_question" in supported["constraints"]
    assert "audio_requires_text_equivalent" in supported["constraints"]


def test_accessible_projection_is_lossless_linear_and_machine_verifiable() -> None:
    contract = build_accessibility_contract(
        {"accessibility_needs": ["一次一问", "屏幕阅读器"]}
    )
    message = "递推关系把较小子问题的结果连接起来。现在只看一个点：边界条件负责什么？"

    rendering = accessible_teacher_rendering(message, contract)

    assert rendering["compliant"] is True
    assert rendering["question_count"] == 1
    assert rendering["authoritative_message_preserved"] is True
    assert rendering["screen_reader_order"] == [1, 2]
    assert rendering["segments"][1]["role"] == "question"
    assert rendering["mastery_standard_lowered"] is False

    repeated_punctuation = accessible_teacher_rendering(
        "先看例子。。现在继续？", contract
    )
    assert repeated_punctuation["authoritative_message_preserved"] is True
    assert (
        "".join(segment["text"] for segment in repeated_punctuation["segments"])
        == "先看例子。。现在继续？"
    )


def test_accessibility_validator_rejects_multi_question_color_only_and_audio_only() -> (
    None
):
    contract = build_accessibility_contract(
        {
            "accessibility_needs": [
                "一次一问",
                "色弱",
                "听障",
                "屏幕阅读器",
            ]
        }
    )
    rendering = accessible_teacher_rendering(
        "看上图：红色对吗？绿色为什么？请听录音。",
        contract,
    )

    assert rendering["compliant"] is False
    assert set(rendering["violations"]) == {
        "more_than_one_question",
        "color_is_sole_cue",
        "audio_has_no_text_equivalent",
        "visual_position_has_no_text_equivalent",
    }


def test_paired_domain_and_language_profiles_repair_only_declared_needs() -> None:
    cases = [
        (
            "数学-中文",
            "看上图，红色曲线代表递推结果。这个结论是什么？为什么成立？请听录音。"
            "这里继续补充一段较长但仍属于原回答的说明，用来验证系统会加短分段而不会删除数学讲解内容。"
            "“教师原句？递推只依赖已定义的前项。”",
            "教师原句？递推只依赖已定义的前项。",
            ["一次一问并使用短句", "屏幕阅读器", "色弱", "听障需要字幕"],
        ),
        (
            "科学-English",
            "See above. The red curve marks the chosen result in a deliberately extended "
            "explanation that keeps every scientific claim available for later audit. "
            "Which force follows? Why does it follow? Listen to audio. "
            '"Teacher source? Net force changes motion."',
            "Teacher source? Net force changes motion.",
            [
                "one question and short sentences",
                "screen reader",
                "color blind",
                "hearing transcript",
            ],
        ),
        (
            "人文-中文",
            "看下图，红色段落代表作者立场。这一立场是什么？用了什么依据？请播放音频。"
            "这段说明保留论证、证据和适用范围，修复只能调整呈现而不能删去原有的人文分析。"
            "“教师材料？作者先陈述主张，再给史料。”",
            "教师材料？作者先陈述主张，再给史料。",
            ["一次一问并分小步", "VoiceOver 读屏", "不依赖颜色", "听障文字稿"],
        ),
        (
            "编程-English",
            "See below. The blue branch marks the chosen result in a deliberately extended "
            "explanation that preserves every programming claim for audit. What returns? "
            "Why does it return? Listen to audio. "
            '"Teacher source? The base case returns one."',
            "Teacher source? The base case returns one.",
            [
                "one question and cognitive load",
                "NVDA screen reader",
                "color independent",
                "captions for hearing",
            ],
        ),
    ]

    observed_repairs: set[str] = set()
    for domain, message, excerpt, needs in cases:
        baseline = repair_accessible_teacher_message(
            message,
            build_accessibility_contract({"accessibility_needs": []}),
            protected_verbatim_excerpts=[excerpt],
        )
        assert baseline["status"] == "already_compliant", domain
        assert baseline["observable_message"] == message, domain

        excerpt_hash = hashlib.sha256(excerpt.encode("utf-8")).hexdigest()
        supported = repair_accessible_teacher_message(
            message,
            build_accessibility_contract({"accessibility_needs": needs}),
            protected_verbatim_excerpts=[
                {
                    "ref": f"teacher:{domain}",
                    "excerpt": excerpt,
                    "excerpt_sha256": excerpt_hash,
                }
            ],
        )

        assert supported["status"] == "repaired", domain
        assert supported["rendering"]["compliant"] is True, domain
        assert supported["rendering"]["question_count"] == 1, domain
        assert supported["authoritative_message"] == message, domain
        assert excerpt in supported["observable_message"], domain
        assert supported["verbatim_excerpts"] == [
            {
                "ref": f"teacher:{domain}",
                "excerpt_sha256": excerpt_hash,
                "occurrences_before": 1,
                "occurrences_after": 1,
                "preserved_verbatim": True,
            }
        ]
        assert supported["answer_content_deleted"] is False, domain
        assert supported["scoring_contract_changed"] is False, domain
        observed_repairs.update(supported["repair_codes"])
        assert {
            "at_most_one_interactive_question",
            "non_positional_visual_equivalent",
            "color_redundant_cue",
            "audio_text_equivalent_fail_closed",
        } <= set(supported["repair_codes"]), domain
        assert (
            max(segment["char_count"] for segment in supported["rendering"]["segments"])
            <= 80
        ), domain
    assert "short_linear_segments" in observed_repairs


def test_repair_rejects_a_forged_verbatim_excerpt_hash() -> None:
    contract = build_accessibility_contract({"accessibility_needs": ["一次一问"]})
    with pytest.raises(AccessibilityError, match="does not match"):
        repair_accessible_teacher_message(
            "依据：“教师原句。”现在继续？",
            contract,
            protected_verbatim_excerpts=[
                {
                    "ref": "teacher:claim",
                    "excerpt": "教师原句。",
                    "excerpt_sha256": "a" * 64,
                }
            ],
        )
