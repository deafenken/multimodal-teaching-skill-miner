"""Executable accessibility projections and final-action repairs.

Accessibility preferences are not decorative profile metadata.  This module
compiles bounded learner needs into a deterministic contract, validates the
observable teacher message, and can make bounded presentation-only repairs.
The pre-repair message and every declared verbatim source excerpt remain
byte-for-byte auditable; assessment contracts are deliberately out of scope.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Mapping, Sequence


ACCESSIBILITY_CONTRACT_SCHEMA = "teaching_skill_miner.accessibility_contract.v1"
ACCESSIBILITY_RENDERING_SCHEMA = "teaching_skill_miner.accessible_rendering.v1"
ACCESSIBILITY_REPAIR_SCHEMA = "teaching_skill_miner.accessibility_repair.v1"
MAX_ACCESSIBILITY_NEEDS = 16
MAX_ACCESSIBILITY_MESSAGE_CHARS = 12_000
MAX_ACCESSIBILITY_REPAIRED_CHARS = 14_000
SHORT_SEGMENT_TARGET_CHARS = 80

_QUESTION = re.compile(r"[?？]")
_SENTENCE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]*|[。！？!?；;]+|\n")
_COLOR_WORD = re.compile(
    r"(?:红色|绿色|蓝色|黄色|橙色|紫色|颜色)|"
    r"\b(?:red|green|blue|yellow|orange|purple|colou?r)\b",
    re.IGNORECASE,
)
_NON_COLOR_CUE = re.compile(
    r"(?:标签|文字|形状|线型|编号|位置|上方|下方|左侧|右侧)|"
    r"\b(?:label|text|shape|line|number|position)\b",
    re.IGNORECASE,
)
_AUDIO_CUE = re.compile(
    r"(?:听录音|听音频|播放音频)|\b(?:listen to|audio)\b", re.IGNORECASE
)
_TEXT_EQUIVALENT = re.compile(
    r"(?:文字稿|字幕|转写|文本)|\b(?:transcript|captions?|text)\b", re.IGNORECASE
)
_VISUAL_POSITION_CUE = re.compile(
    r"(?:看上图|看下图|图中颜色|上面的图|下面的图|"
    r"\bsee above\b|\bsee below\b|\bfigure above\b|\bfigure below\b)",
    re.IGNORECASE,
)
_PROTECTED_LITERAL = re.compile(
    r"```[\s\S]*?```|`[^`\n]+`|\$\$[\s\S]*?\$\$|\$[^$\n]+\$|"
    r"https?://[^\s<>()]+|“[^”]*”|「[^」]*」|『[^』]*』",
    re.IGNORECASE,
)

_ALIASES: dict[str, tuple[str, ...]] = {
    "reduced_cognitive_load": (
        "一次一问",
        "认知负荷",
        "分小步",
        "简短",
        "short sentences",
        "one question",
        "cognitive load",
    ),
    "screen_reader_linear": (
        "读屏",
        "屏幕阅读器",
        "voiceover",
        "nvda",
        "screen reader",
    ),
    "color_independent": (
        "色弱",
        "色盲",
        "不依赖颜色",
        "color blind",
        "colour blind",
        "color independent",
    ),
    "text_equivalent_for_audio": (
        "听障",
        "字幕",
        "文字稿",
        "hearing",
        "captions",
        "transcript",
    ),
    "language_scaffold": (
        "语言学习者",
        "第二语言",
        "术语解释",
        "language learner",
        "second language",
        "explain terms",
    ),
}


class AccessibilityError(ValueError):
    """Raised when an accessibility contract or rendering is invalid."""


def _normalized_needs(profile: Mapping[str, Any] | None) -> list[str]:
    raw = profile.get("accessibility_needs", []) if isinstance(profile, Mapping) else []
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise AccessibilityError("accessibility_needs must be an array")
    if len(raw) > MAX_ACCESSIBILITY_NEEDS:
        raise AccessibilityError("too many accessibility needs")
    result: list[str] = []
    for item in raw:
        if not isinstance(item, str) or not item.strip() or len(item) > 160:
            raise AccessibilityError("accessibility need is invalid")
        value = re.sub(r"\s+", " ", item).strip()
        if value not in result:
            result.append(value)
    return result


def build_accessibility_contract(
    profile: Mapping[str, Any] | None,
) -> dict[str, Any]:
    needs = _normalized_needs(profile)
    joined = "\n".join(needs).casefold()
    enabled = {
        capability: any(alias.casefold() in joined for alias in aliases)
        for capability, aliases in _ALIASES.items()
    }
    constraints: list[str] = []
    if enabled["reduced_cognitive_load"]:
        constraints.extend(
            ["at_most_one_question", "linear_segments", "short_segment_target_80"]
        )
    if enabled["screen_reader_linear"]:
        constraints.extend(
            ["linear_segments", "semantic_segment_labels", "no_visual_position_only"]
        )
    if enabled["color_independent"]:
        constraints.append("color_never_sole_cue")
    if enabled["text_equivalent_for_audio"]:
        constraints.append("audio_requires_text_equivalent")
    if enabled["language_scaffold"]:
        constraints.extend(["plain_language", "term_glossary_when_needed"])
    return {
        "schema": ACCESSIBILITY_CONTRACT_SCHEMA,
        "source": "learner_profile.accessibility_needs",
        "declared_needs": needs,
        "enabled": enabled,
        "constraints": sorted(set(constraints)),
        "mastery_standard_lowered": False,
        "assessment_construct_changed": False,
    }


def _segments(message: str) -> list[dict[str, Any]]:
    parts = [match.group(0) for match in _SENTENCE.finditer(message)]
    if "".join(parts) != message:
        raise AccessibilityError("accessible sentence segmentation is not lossless")
    result: list[dict[str, Any]] = []
    for part in parts:
        if part == "\n":
            continue
        content = part.strip()
        if not content:
            continue
        result.append(
            {
                "index": len(result) + 1,
                "role": "question" if _QUESTION.search(content) else "explanation",
                "text": content,
                "char_count": len(content),
            }
        )
    return result


def _normalized_verbatim_excerpts(
    message: str,
    excerpts: Sequence[Mapping[str, Any] | str],
) -> list[dict[str, str]]:
    if isinstance(excerpts, (str, bytes)):
        raise AccessibilityError("verbatim excerpts must be an array")
    normalized: list[dict[str, str]] = []
    for index, item in enumerate(excerpts):
        if isinstance(item, Mapping):
            text = str(item.get("excerpt", item.get("text", "")))
            reference = str(item.get("ref", f"verbatim:{index + 1}"))
            declared_hash = str(item.get("excerpt_sha256", ""))
        else:
            text = str(item)
            reference = f"verbatim:{index + 1}"
            declared_hash = ""
        if not text or text not in message:
            raise AccessibilityError(
                "verbatim excerpt is not present in teacher message"
            )
        if declared_hash and not re.fullmatch(r"[0-9a-f]{64}", declared_hash):
            raise AccessibilityError("verbatim excerpt hash is invalid")
        raw_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        canonical_string_hash = hashlib.sha256(
            json.dumps(
                text,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
        ).hexdigest()
        if declared_hash and declared_hash not in {raw_hash, canonical_string_hash}:
            raise AccessibilityError("verbatim excerpt hash does not match excerpt")
        normalized.append(
            {
                "excerpt": text,
                "ref": reference,
                "excerpt_sha256": declared_hash or raw_hash,
            }
        )
    return normalized


def _protected_ranges(
    message: str,
    excerpts: Sequence[Mapping[str, str]],
) -> list[tuple[int, int]]:
    ranges = [
        (match.start(), match.end()) for match in _PROTECTED_LITERAL.finditer(message)
    ]
    for item in excerpts:
        excerpt = item["excerpt"]
        start = 0
        while (found := message.find(excerpt, start)) >= 0:
            ranges.append((found, found + len(excerpt)))
            start = found + len(excerpt)
    if not ranges:
        return []
    merged: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _chunks(
    message: str,
    excerpts: Sequence[Mapping[str, str]],
) -> list[tuple[str, bool]]:
    result: list[tuple[str, bool]] = []
    cursor = 0
    for start, end in _protected_ranges(message, excerpts):
        if cursor < start:
            result.append((message[cursor:start], False))
        result.append((message[start:end], True))
        cursor = end
    if cursor < len(message):
        result.append((message[cursor:], False))
    return result or [(message, False)]


def _validation_surface(
    message: str,
    excerpts: Sequence[Mapping[str, str]],
) -> str:
    """Replace quoted/code source spans with line boundaries for validation.

    A question mark inside a cited teacher excerpt is evidence being read, not
    a second prompt for the learner.  A long hash-bound excerpt also cannot be
    line-wrapped without breaking its provenance.  The original and observable
    message still retain every byte of each excerpt.
    """

    return "".join(
        "\n" if protected else text for text, protected in _chunks(message, excerpts)
    )


def _soft_wrap(text: str, *, target: int = SHORT_SEGMENT_TARGET_CHARS) -> str:
    """Insert line boundaries without deleting or reordering any character."""

    output: list[str] = []
    for line_index, line in enumerate(text.split("\n")):
        remaining = line
        while len(remaining) > target:
            search_floor = max(1, target // 2)
            candidates = [
                position + 1
                for position, character in enumerate(remaining[:target])
                if position + 1 >= search_floor
                and character in "。！？!?；;，,、：: \t"
            ]
            cut = candidates[-1] if candidates else target
            output.append(remaining[:cut])
            output.append("\n")
            remaining = remaining[cut:]
        output.append(remaining)
        if line_index < len(text.split("\n")) - 1:
            output.append("\n")
    return "".join(output)


def _is_chinese(message: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", message))


def _neutralize_earlier_questions(
    chunks: Sequence[tuple[str, bool]],
) -> tuple[list[tuple[str, bool]], int]:
    question_count = sum(
        len(_QUESTION.findall(text)) for text, protected in chunks if not protected
    )
    remaining = max(0, question_count - 1)
    repaired: list[tuple[str, bool]] = []
    changed = 0
    for text, protected in chunks:
        if protected or remaining == 0:
            repaired.append((text, protected))
            continue

        def replace(match: re.Match[str]) -> str:
            nonlocal remaining, changed
            if remaining <= 0:
                return match.group(0)
            remaining -= 1
            changed += 1
            return "。" if match.group(0) == "？" else "."

        repaired.append((_QUESTION.sub(replace, text), False))
    return repaired, changed


def accessible_teacher_rendering(
    message: str,
    contract: Mapping[str, Any],
    *,
    protected_verbatim_excerpts: Sequence[Mapping[str, Any] | str] = (),
) -> dict[str, Any]:
    """Return a lossless parallel rendering and concrete violation list."""

    if contract.get("schema") != ACCESSIBILITY_CONTRACT_SCHEMA:
        raise AccessibilityError("accessibility contract schema is invalid")
    if (
        not isinstance(message, str)
        or not message.strip()
        or len(message) > MAX_ACCESSIBILITY_MESSAGE_CHARS
    ):
        raise AccessibilityError("teacher message is empty or too long")
    enabled = contract.get("enabled")
    if not isinstance(enabled, Mapping):
        raise AccessibilityError("accessibility contract capabilities are invalid")
    normalized_excerpts = _normalized_verbatim_excerpts(
        message, protected_verbatim_excerpts
    )
    segments = _segments(message)
    validation_surface = _validation_surface(message, normalized_excerpts)
    validation_segments = _segments(validation_surface)
    question_count = len(_QUESTION.findall(validation_surface))
    raw_question_mark_count = len(_QUESTION.findall(message))
    violations: list[str] = []
    if enabled.get("reduced_cognitive_load") is True and question_count > 1:
        violations.append("more_than_one_question")
    if enabled.get("reduced_cognitive_load") is True and any(
        segment["char_count"] > SHORT_SEGMENT_TARGET_CHARS
        for segment in validation_segments
    ):
        violations.append("long_unbroken_segment")
    if (
        enabled.get("color_independent") is True
        and _COLOR_WORD.search(message)
        and not _NON_COLOR_CUE.search(message)
    ):
        violations.append("color_is_sole_cue")
    if (
        enabled.get("text_equivalent_for_audio") is True
        and _AUDIO_CUE.search(message)
        and not _TEXT_EQUIVALENT.search(message)
    ):
        violations.append("audio_has_no_text_equivalent")
    if (
        enabled.get("screen_reader_linear") is True
        and _VISUAL_POSITION_CUE.search(message)
        and not _TEXT_EQUIVALENT.search(message)
    ):
        violations.append("visual_position_has_no_text_equivalent")
    return {
        "schema": ACCESSIBILITY_RENDERING_SCHEMA,
        "message_sha256": hashlib.sha256(message.encode("utf-8")).hexdigest(),
        "authoritative_message_preserved": True,
        "segments": segments,
        "question_count": question_count,
        "raw_question_mark_count": raw_question_mark_count,
        "screen_reader_order": [segment["index"] for segment in segments],
        "verbatim_excerpt_count": len(normalized_excerpts),
        "verbatim_long_segment_exemptions": sum(
            len(item["excerpt"]) > SHORT_SEGMENT_TARGET_CHARS
            for item in normalized_excerpts
        ),
        "violations": violations,
        "compliant": not violations,
        "mastery_standard_lowered": False,
    }


def repair_accessible_teacher_message(
    message: str,
    contract: Mapping[str, Any],
    *,
    protected_verbatim_excerpts: Sequence[Mapping[str, Any] | str] = (),
) -> dict[str, Any]:
    """Return a deterministic, presentation-only repair and audit receipt.

    Repairs may add accessibility notes and line boundaries.  The only
    in-place substitution is neutralising earlier interactive question marks;
    all lexical answer content remains in order.  Declared verbatim excerpts,
    code, maths, URLs, and quoted evidence are never modified.
    """

    normalized_excerpts = _normalized_verbatim_excerpts(
        message, protected_verbatim_excerpts
    )
    before = accessible_teacher_rendering(
        message,
        contract,
        protected_verbatim_excerpts=normalized_excerpts,
    )
    enabled = contract.get("enabled", {})
    if not isinstance(enabled, Mapping):
        raise AccessibilityError("accessibility contract capabilities are invalid")

    message_chunks = _chunks(message, normalized_excerpts)
    repair_codes: list[str] = []
    question_marks_neutralized = 0
    if enabled.get("reduced_cognitive_load") is True and before["question_count"] > 1:
        message_chunks, question_marks_neutralized = _neutralize_earlier_questions(
            message_chunks
        )
        repair_codes.append("at_most_one_interactive_question")

    if enabled.get("reduced_cognitive_load") is True:
        wrapped: list[tuple[str, bool]] = []
        for text, protected in message_chunks:
            wrapped.append((text if protected else _soft_wrap(text), protected))
        if any(
            text != new_text
            for (text, _), (new_text, _) in zip(message_chunks, wrapped)
        ):
            repair_codes.append("short_linear_segments")
        message_chunks = wrapped

    repaired_body = "".join(text for text, _protected in message_chunks)
    notes: list[str] = []
    chinese = _is_chinese(message)
    if question_marks_neutralized:
        notes.append(
            "作答顺序：本轮只回答最后一个问题；前面的问句仅作讲解提示。"
            if chinese
            else "Response order: answer only the final question; earlier questions are explanatory prompts."
        )
    if (
        enabled.get("screen_reader_linear") is True
        and "visual_position_has_no_text_equivalent" in before["violations"]
    ):
        notes.append(
            "读屏说明：位置词对应的内容须同时按标题、编号和文字说明给出；未提供时，本轮不依赖该位置指示。"
            if chinese
            else "Screen-reader note: positional references require a named, numbered text description; this turn does not rely on them until provided."
        )
        repair_codes.append("non_positional_visual_equivalent")
    if (
        enabled.get("color_independent") is True
        and "color_is_sole_cue" in before["violations"]
    ):
        notes.append(
            "颜色不是唯一线索：请同时按文字标签、形状或编号判断。"
            if chinese
            else "Color is not the only cue: use the text label, shape, or number as well."
        )
        repair_codes.append("color_redundant_cue")
    if (
        enabled.get("text_equivalent_for_audio") is True
        and "audio_has_no_text_equivalent" in before["violations"]
    ):
        notes.append(
            "文字稿状态：尚未提供；提供前，本轮不依赖音频内容，也不据此评分。"
            if chinese
            else "Transcript status: not yet provided; this turn does not rely on or grade the audio until text is available."
        )
        repair_codes.append("audio_text_equivalent_fail_closed")

    if enabled.get("reduced_cognitive_load") is True:
        notes = [_soft_wrap(note) for note in notes]
    observable_message = "\n".join([*notes, repaired_body]) if notes else repaired_body
    if len(observable_message) > MAX_ACCESSIBILITY_REPAIRED_CHARS:
        raise AccessibilityError("accessible teacher message exceeds repair budget")
    after = accessible_teacher_rendering(
        observable_message,
        contract,
        protected_verbatim_excerpts=normalized_excerpts,
    )
    if after["violations"]:
        raise AccessibilityError(
            "accessible teacher message remains non-compliant after repair: "
            + ", ".join(after["violations"])
        )

    excerpt_receipts: list[dict[str, Any]] = []
    for item in normalized_excerpts:
        before_count = message.count(item["excerpt"])
        after_count = observable_message.count(item["excerpt"])
        excerpt_receipts.append(
            {
                "ref": item["ref"],
                "excerpt_sha256": item["excerpt_sha256"],
                "occurrences_before": before_count,
                "occurrences_after": after_count,
                "preserved_verbatim": before_count == after_count and before_count > 0,
            }
        )
    if not all(item["preserved_verbatim"] for item in excerpt_receipts):
        raise AccessibilityError(
            "verbatim grounding changed during accessibility repair"
        )

    return {
        "schema": ACCESSIBILITY_REPAIR_SCHEMA,
        "status": "repaired" if repair_codes else "already_compliant",
        "authoritative_message": message,
        "authoritative_message_sha256": hashlib.sha256(
            message.encode("utf-8")
        ).hexdigest(),
        "observable_message": observable_message,
        "observable_message_sha256": after["message_sha256"],
        "repair_codes": repair_codes,
        "question_marks_neutralized": question_marks_neutralized,
        "verbatim_excerpts": excerpt_receipts,
        "verbatim_excerpts_preserved": all(
            item["preserved_verbatim"] for item in excerpt_receipts
        ),
        "authoritative_message_stored_verbatim": True,
        "answer_content_deleted": False,
        "answer_content_reordered": False,
        "scoring_contract_changed": False,
        "mastery_standard_lowered": False,
        "rendering": after,
    }


__all__ = [
    "ACCESSIBILITY_CONTRACT_SCHEMA",
    "ACCESSIBILITY_REPAIR_SCHEMA",
    "ACCESSIBILITY_RENDERING_SCHEMA",
    "AccessibilityError",
    "accessible_teacher_rendering",
    "build_accessibility_contract",
    "repair_accessible_teacher_message",
]
