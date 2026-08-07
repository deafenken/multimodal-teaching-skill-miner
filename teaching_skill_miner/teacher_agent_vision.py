"""Local, auditable visual-evidence extraction for learner answer images.

DeepSeek's current ``deepseek-v4-flash`` Chat Completions endpoint accepts text
content only.  This module therefore keeps the original image local, extracts a
bounded OCR observation with Apple Vision on supported macOS hosts or the
system Tesseract binary elsewhere, and returns a provenance-rich text record
that the Teaching Agent can reason over.  Raw media is never returned or
retained by this module.
"""

from __future__ import annotations

import csv
from hashlib import sha256
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence
import unicodedata


VISUAL_EVIDENCE_SCHEMA = "teaching_skill_miner.local_visual_evidence.v1"
SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_RECOGNIZED_TEXT_CHARS = 2_400
MINIMUM_TRUSTED_OCR_CONFIDENCE = 0.58
MINIMUM_CORROBORATED_OCR_CONFIDENCE = 0.72
MINIMUM_SINGLE_ENGINE_FORMULA_CONFIDENCE = 0.90
MAX_DECODED_IMAGE_PIXELS = 20_000_000
_LANGUAGE_RE = re.compile(r"[A-Za-z0-9_+.-]{1,80}")
_APPLE_VISION_CONTROL_VALUES = frozenset(
    {"auto", "on", "off", "true", "false", "1", "0", "yes", "no"}
)
_FORMULA_TEXT_RE = re.compile(
    r"""
    [=＝±√∫∑∏∂∞≈≠≤≥<>×÷∈∉∩∪→←↔⇒⇔∧∨¬^_²³⁰¹]
    |(?:\+\+|--|&&|\|\||::|:=|=>|->|<<|>>)
    |[A-Za-z0-9_)\]]\s*[+＋*/×÷%]\s*[A-Za-z0-9_(\[]
    |(?:[A-Za-z0-9_)\]]\s*[-−]\s*\d|\d\s*[-−]\s*[A-Za-z0-9_(\[])
    |\b[A-Za-z]\s*[-−]\s*[A-Za-z]\b
    |\b\d+(?:\.\d+)?\s*[A-Za-z]\b
    |\b[A-Za-z_][A-Za-z0-9_]*\[[^\]\r\n]{1,80}\]
    |\b[A-Za-z_][A-Za-z0-9_]*\([^()\r\n]{0,80}\)
    |\b(?:if|else|for|while|return|def|class|lambda|print|sqrt|integral|matrix|det)\b
    """,
    re.IGNORECASE | re.VERBOSE,
)
_APPLE_VISION_OBJC_SOURCE = r"""#import <AppKit/AppKit.h>
#import <Foundation/Foundation.h>
#import <Vision/Vision.h>

int main(int argc, const char *argv[]) {
    @autoreleasepool {
        if (argc != 2) { return 2; }
        NSString *path = [NSString stringWithUTF8String:argv[1]];
        NSImage *image = [[NSImage alloc] initWithContentsOfFile:path];
        if (image == nil) { return 3; }
        CGRect rect = CGRectMake(0, 0, image.size.width, image.size.height);
        CGImageRef cgImage = [image CGImageForProposedRect:&rect context:nil hints:nil];
        if (cgImage == nil) { return 4; }

        VNRecognizeTextRequest *request = [[VNRecognizeTextRequest alloc] init];
        request.recognitionLevel = VNRequestTextRecognitionLevelAccurate;
        request.usesLanguageCorrection = YES;
        request.minimumTextHeight = 0.012;
        if (@available(macOS 13.0, *)) {
            request.automaticallyDetectsLanguage = YES;
        } else {
            request.recognitionLanguages = @[@"zh-Hans", @"en-US"];
        }
        VNImageRequestHandler *handler = [[VNImageRequestHandler alloc]
            initWithCGImage:cgImage options:@{}];
        NSError *error = nil;
        if (![handler performRequests:@[request] error:&error]) { return 5; }

        NSArray<VNRecognizedTextObservation *> *observations =
            [request.results sortedArrayUsingComparator:^NSComparisonResult(
                VNRecognizedTextObservation *left,
                VNRecognizedTextObservation *right
            ) {
                CGFloat vertical = CGRectGetMidY(left.boundingBox)
                    - CGRectGetMidY(right.boundingBox);
                if (fabs(vertical) > 0.015) {
                    return vertical > 0 ? NSOrderedAscending : NSOrderedDescending;
                }
                CGFloat horizontal = CGRectGetMinX(left.boundingBox)
                    - CGRectGetMinX(right.boundingBox);
                return horizontal < 0 ? NSOrderedAscending : NSOrderedDescending;
            }];
        NSMutableArray<NSString *> *lines = [NSMutableArray array];
        double confidenceTotal = 0.0;
        for (VNRecognizedTextObservation *observation in observations) {
            VNRecognizedText *candidate = [observation topCandidates:1].firstObject;
            if (candidate == nil) { continue; }
            [lines addObject:candidate.string];
            confidenceTotal += candidate.confidence;
        }
        NSString *text = [lines componentsJoinedByString:@"\n"];
        if (text.length > 2400) { text = [text substringToIndex:2400]; }
        double confidence = lines.count == 0
            ? 0.0
            : confidenceTotal / lines.count;
        NSDictionary *payload = @{
            @"text": text,
            @"confidence": @(confidence),
        };
        NSData *encoded = [NSJSONSerialization
            dataWithJSONObject:payload options:0 error:&error];
        if (encoded == nil) { return 6; }
        fwrite(encoded.bytes, 1, encoded.length, stdout);
        fputc('\n', stdout);
        return 0;
    }
}
"""


class LocalVisualEvidenceError(RuntimeError):
    """Raised when learner media cannot be handled within the local contract."""


def contains_formula_like_text(value: str) -> bool:
    """Return whether text contains a formula, code, or symbolic expression.

    The same conservative detector is shared by local OCR auditing and the live
    Agent's typed-answer gate so keyboard-entered ``x+1`` is not treated like
    uncertain OCR merely because an image was attached to the same turn.
    """

    return bool(_FORMULA_TEXT_RE.search(str(value)))


def _bounded_confidence(value: Any) -> float:
    if (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(float(value))
    ):
        return max(0.0, min(1.0, float(value)))
    return 0.0


def _canonical_transcription(value: str) -> str:
    """Normalize only OCR-formatting differences, not mathematical meaning."""

    normalized = unicodedata.normalize("NFKC", str(value)).casefold()
    normalized = normalized.translate(
        str.maketrans(
            {
                "−": "-",
                "–": "-",
                "—": "-",
                "﹣": "-",
                "×": "*",
                "÷": "/",
            }
        )
    )
    return re.sub(r"\s+", "", normalized).strip()


def _answer_variants(value: str) -> set[str]:
    """Return conservative full-answer variants for deterministic alignment."""

    text = str(value).strip()
    values = {_canonical_transcription(text)} if text else set()
    terminal_stripped = text.rstrip("。.!！?？;；")
    if terminal_stripped != text:
        values.add(_canonical_transcription(terminal_stripped))
    for prefix in (
        "答案是",
        "答案",
        "结果是",
        "结果",
        "我认为是",
        "我觉得是",
        "应该是",
        "answer is",
        "answer",
    ):
        match = re.match(
            rf"^\s*{re.escape(prefix)}\s*[:：=]?\s*(?P<answer>.+?)\s*$",
            text,
            re.IGNORECASE | re.DOTALL,
        )
        if match:
            answer = match.group("answer").rstrip("。.!！?？;；")
            values.add(_canonical_transcription(answer))
    values.discard("")
    return values


def assess_typed_visual_consistency(
    learner_text: str,
    evidence: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    """Flag a typed/OCR disagreement without pretending to understand the image.

    This comparison is deliberately lexical.  It can establish that two local
    transcriptions agree, but never that either answer is pedagogically correct.
    """

    typed = str(learner_text or "").strip()
    recognized = [
        str(item.get("recognized_text", "")).strip()
        for item in evidence
        if str(item.get("recognized_text", "")).strip()
        and (
            item.get("status") == "recognized"
            or item.get("student_confirmed_recognized_text") is True
        )
        and not bool(item.get("needs_student_confirmation"))
    ]
    if not typed or not recognized:
        return {
            "relation": "not_comparable",
            "possible_conflict": False,
            "needs_student_confirmation": False,
        }
    typed_variants = _answer_variants(typed)
    per_attachment_relations: list[str] = []
    for recognized_text in recognized:
        visual_variants = _answer_variants(recognized_text)
        if typed_variants & visual_variants:
            per_attachment_relations.append("exact_agreement")
        elif any(
            len(left) >= 3 and len(right) >= 3 and (left in right or right in left)
            for left in typed_variants
            for right in visual_variants
        ):
            per_attachment_relations.append("compatible_overlap")
        else:
            per_attachment_relations.append("possible_conflict")
    if "possible_conflict" in per_attachment_relations:
        relation = "possible_conflict"
        conflict = True
    elif set(per_attachment_relations) == {"exact_agreement"}:
        relation = "exact_agreement"
        conflict = False
    else:
        relation = "compatible_overlap"
        conflict = False
    return {
        "relation": relation,
        "possible_conflict": conflict,
        "needs_student_confirmation": conflict,
    }


def align_ocr_text_to_answer_references(
    recognized_text: str,
    question_contract: Mapping[str, Any] | None,
    knowledge_spec: Mapping[str, Any] | None = None,
    active_knowledge_components: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Deterministically align OCR text to teacher-owned answer references.

    Exact alignment is intentionally narrow.  It is suitable for repairing a
    model false negative after the OCR transcription itself has passed the
    local reliability gate.  It is not semantic grading and it never promotes
    a merely related phrase to a correct answer.
    """

    text_variants = _answer_variants(recognized_text)
    contract = question_contract if isinstance(question_contract, Mapping) else {}
    answer_type = str(contract.get("answer_type", "open"))
    raw_contract_references: list[tuple[str, str]] = []
    for field in ("target_concepts", "accepted_aliases"):
        values = contract.get(field, [])
        if isinstance(values, Sequence) and not isinstance(values, (str, bytes)):
            raw_contract_references.extend(
                (field, str(value).strip()) for value in values if str(value).strip()
            )

    def exact_match(
        references: Iterable[tuple[str, str]],
    ) -> tuple[str, str] | None:
        for source, reference in references:
            if _canonical_transcription(reference) in text_variants:
                return source, reference
        return None

    contract_match = exact_match(raw_contract_references)
    if contract_match is not None:
        source, reference = contract_match
        formula_or_numeric = contains_formula_like_text(reference) or bool(
            re.fullmatch(r"[-+]?\d+(?:\.\d+)?", reference.strip())
        )
        correctness_established = answer_type == "short_concept" or (
            answer_type == "worked_step" and formula_or_numeric
        )
        return {
            "alignment": "exact_contract_match",
            "matched_source": f"question_contract.{source}",
            "matched_reference": reference,
            "deterministic_correctness_established": correctness_established,
            "requires_reliable_transcription": True,
        }

    active_components = {
        str(item).strip()
        for item in (active_knowledge_components or [])
        if str(item).strip()
    }

    def claim_is_in_scope(item: Mapping[str, Any]) -> bool:
        """Only use teacher claims explicitly attached to this action's topic.

        An unscoped claim is useful as model context, but it is never an
        authoritative deterministic answer key.  This prevents a correct
        statement about a neighbouring concept from being credited for the
        question currently on screen.
        """

        components = {
            str(component).strip()
            for component in (item.get("knowledge_components", []) or [])
            if str(component).strip()
        }
        return bool(active_components and components and active_components & components)

    teacher_references: list[tuple[str, str, bool]] = []
    spec = knowledge_spec if isinstance(knowledge_spec, Mapping) else {}
    for item in spec.get("canonical_claims", []) or []:
        if (
            isinstance(item, Mapping)
            and claim_is_in_scope(item)
            and str(item.get("statement", "")).strip()
        ):
            teacher_references.append(
                (
                    f"knowledge_spec.canonical_claims.{item.get('claim_id', 'unknown')}",
                    str(item["statement"]).strip(),
                    True,
                )
            )
    for item in spec.get("rubric_criteria", []) or []:
        if not isinstance(item, Mapping):
            continue
        criterion_component = str(item.get("knowledge_component", "")).strip()
        if not active_components or not criterion_component or criterion_component not in active_components:
            continue
        criterion_id = str(item.get("criterion_id", "unknown"))
        for accepted in item.get("acceptable_evidence", []) or []:
            if str(accepted).strip():
                teacher_references.append(
                    (
                        f"knowledge_spec.rubric_criteria.{criterion_id}",
                        str(accepted).strip(),
                        False,
                    )
                )
    teacher_match: tuple[str, str, bool] | None = None
    for source, reference, establishes_correctness in teacher_references:
        if _canonical_transcription(reference) in text_variants:
            teacher_match = (source, reference, establishes_correctness)
            break
    if teacher_match is not None:
        source, reference, establishes_correctness = teacher_match
        return {
            "alignment": "exact_teacher_reference_match",
            "matched_source": source,
            "matched_reference": reference,
            "deterministic_correctness_established": establishes_correctness,
            "requires_reliable_transcription": True,
        }
    return {
        "alignment": "not_established",
        "matched_source": None,
        "matched_reference": None,
        "deterministic_correctness_established": False,
        "requires_reliable_transcription": True,
    }


def _detected_mime_type(image_bytes: bytes) -> str | None:
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if (
        len(image_bytes) >= 12
        and image_bytes[:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "image/webp"
    return None


def validate_local_image(image_bytes: bytes, mime_type: str) -> str:
    """Validate a bounded browser-normalized image and return its MIME type."""

    if not isinstance(image_bytes, bytes):
        raise TypeError("image_bytes must be bytes")
    if not image_bytes or len(image_bytes) > MAX_IMAGE_BYTES:
        raise LocalVisualEvidenceError(
            f"image byte size must be in [1, {MAX_IMAGE_BYTES}]"
        )
    declared = str(mime_type or "").strip().casefold()
    if declared not in SUPPORTED_IMAGE_MIME_TYPES:
        raise LocalVisualEvidenceError("image MIME type must be JPEG, PNG, or WebP")
    detected = _detected_mime_type(image_bytes)
    if detected is None or detected != declared:
        raise LocalVisualEvidenceError(
            "image bytes do not match the declared MIME type"
        )
    return detected


def _ocr_languages() -> str:
    configured = os.getenv("TSM_TESSERACT_LANGUAGES", "eng+snum").strip()
    if not _LANGUAGE_RE.fullmatch(configured):
        raise LocalVisualEvidenceError(
            "TSM_TESSERACT_LANGUAGES contains unsupported characters"
        )
    return configured


def _tesseract_command() -> str | None:
    configured = os.getenv("TSM_TESSERACT", "tesseract").strip()
    if not configured:
        return None
    if Path(configured).is_absolute():
        return configured if Path(configured).is_file() else None
    return shutil.which(configured)


def _apple_vision_command() -> str | None:
    """Return the local compiler for optional macOS Apple Vision OCR."""

    configured_mode = os.getenv("TSM_APPLE_VISION_OCR", "auto").strip().casefold()
    if configured_mode not in _APPLE_VISION_CONTROL_VALUES:
        raise LocalVisualEvidenceError(
            "TSM_APPLE_VISION_OCR must be auto, on/off, true/false, yes/no, or 1/0"
        )
    if configured_mode in {"off", "false", "no", "0"} or sys.platform != "darwin":
        return None
    configured = os.getenv("TSM_APPLE_VISION_CLANG", "/usr/bin/clang").strip()
    if not configured:
        return None
    if Path(configured).is_absolute():
        return configured if Path(configured).is_file() else None
    return shutil.which(configured)


def _apple_vision_environment(working_dir: Path) -> dict[str, str]:
    """Build a minimal subprocess environment without forwarding API secrets."""

    environment = {
        key: os.environ[key]
        for key in ("PATH", "LANG", "LC_ALL", "SDKROOT", "DEVELOPER_DIR")
        if key in os.environ
    }
    clang_cache = working_dir / "clang-module-cache"
    clang_cache.mkdir()
    environment.update(
        {
            "TMPDIR": str(working_dir),
            "CLANG_MODULE_CACHE_PATH": str(clang_cache),
        }
    )
    return environment


def _tesseract_environment(working_dir: Path) -> dict[str, str]:
    """Build a minimal OCR environment without forwarding process secrets."""

    environment = {
        key: os.environ[key]
        for key in (
            "PATH",
            "LANG",
            "LC_ALL",
            "LC_CTYPE",
            "TESSDATA_PREFIX",
            "SYSTEMROOT",
            "WINDIR",
        )
        if key in os.environ
    }
    environment["TMPDIR"] = str(working_dir)
    return environment


def _run_apple_vision(
    executable: str,
    image_path: Path,
    *,
    working_dir: Path,
) -> tuple[str, float]:
    """Run the on-device Vision framework and parse its bounded JSON result."""

    source_path = working_dir / "apple_vision_ocr.m"
    helper_path = working_dir / "apple_vision_ocr"
    source_path.write_text(_APPLE_VISION_OBJC_SOURCE, encoding="utf-8")
    environment = _apple_vision_environment(working_dir)
    try:
        compiled = subprocess.run(
            [
                executable,
                "-fobjc-arc",
                "-fblocks",
                "-framework",
                "Foundation",
                "-framework",
                "AppKit",
                "-framework",
                "Vision",
                str(source_path),
                "-o",
                str(helper_path),
            ],
            check=False,
            capture_output=True,
            timeout=12,
            env=environment,
        )
        if compiled.returncode != 0 or not helper_path.is_file():
            raise LocalVisualEvidenceError("Apple Vision OCR helper build failed")
        completed = subprocess.run(
            [str(helper_path), str(image_path)],
            check=False,
            capture_output=True,
            timeout=8,
            env=environment,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalVisualEvidenceError("Apple Vision OCR process failed") from exc
    if completed.returncode != 0:
        raise LocalVisualEvidenceError("Apple Vision OCR process failed")
    try:
        payload = json.loads(completed.stdout.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise LocalVisualEvidenceError(
            "Apple Vision OCR returned invalid JSON"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("text"), str):
        raise LocalVisualEvidenceError("Apple Vision OCR returned an invalid payload")
    confidence = payload.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
    ):
        raise LocalVisualEvidenceError(
            "Apple Vision OCR returned an invalid confidence"
        )
    return payload["text"].strip(), max(0.0, min(1.0, float(confidence)))


def _line_text(rows: Iterable[dict[str, str]]) -> tuple[str, float]:
    lines: dict[tuple[str, str, str, str], list[str]] = {}
    confidences: list[float] = []
    for row in rows:
        token = str(row.get("text", "")).strip()
        if not token:
            continue
        try:
            confidence = float(row.get("conf", "-1"))
        except ValueError:
            confidence = -1.0
        if confidence >= 0:
            confidences.append(confidence)
        key = (
            str(row.get("page_num", "0")),
            str(row.get("block_num", "0")),
            str(row.get("par_num", "0")),
            str(row.get("line_num", "0")),
        )
        lines.setdefault(key, []).append(token)
    text = "\n".join(" ".join(tokens) for tokens in lines.values()).strip()
    confidence = sum(confidences) / len(confidences) / 100.0 if confidences else 0.0
    return text, max(0.0, min(1.0, confidence))


def _candidate_score(
    text: str, confidence: float
) -> tuple[int, float, float, float, int]:
    """Rank usable OCR candidates by confidence, then bounded text quality.

    Page-segmentation modes often trade a short, accurate answer against a much
    longer low-confidence token stream.  Length is therefore only a tie-breaker
    after confidence; it must never compensate for a weaker OCR confidence.
    """

    stripped = text.strip()
    bounded_confidence = (
        max(0.0, min(1.0, float(confidence)))
        if isinstance(confidence, (int, float))
        and not isinstance(confidence, bool)
        and math.isfinite(float(confidence))
        else 0.0
    )
    characters = [character for character in stripped if not character.isspace()]
    accepted_symbols = frozenset("+-*/%=<>[](){}_^.,:;!?，。！？；：、'\"")
    readable_count = sum(
        character.isalnum() or character in accepted_symbols for character in characters
    )
    readable_ratio = readable_count / len(characters) if characters else 0.0
    diversity_ratio = (
        min(1.0, len(set(characters)) / min(len(characters), 32)) if characters else 0.0
    )
    return (
        int(bool(stripped)),
        bounded_confidence,
        round(readable_ratio, 6),
        round(diversity_ratio, 6),
        min(len(stripped), MAX_RECOGNIZED_TEXT_CHARS),
    )


def _otsu_threshold(histogram: Sequence[int]) -> int:
    """Return a deterministic Otsu threshold for an 8-bit histogram."""

    total = sum(int(value) for value in histogram[:256])
    if total <= 0:
        return 127
    weighted_total = sum(
        index * int(value) for index, value in enumerate(histogram[:256])
    )
    background_weight = 0
    background_sum = 0.0
    best_threshold = 127
    best_variance = -1.0
    for threshold, count_raw in enumerate(histogram[:256]):
        count = int(count_raw)
        background_weight += count
        if background_weight == 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight == 0:
            break
        background_sum += threshold * count
        background_mean = background_sum / background_weight
        foreground_mean = (weighted_total - background_sum) / foreground_weight
        between_variance = (
            background_weight
            * foreground_weight
            * (background_mean - foreground_mean) ** 2
        )
        if between_variance > best_variance:
            best_variance = between_variance
            best_threshold = threshold
    return best_threshold


def _prepare_ocr_variants(
    image_path: Path,
    *,
    working_dir: Path,
) -> list[dict[str, Any]]:
    """Create bounded local OCR variants; failure leaves the original usable.

    Pillow is optional at runtime.  When available, rotation normalization,
    upscaling, grayscale autocontrast, and binarization provide genuinely
    different OCR routes.  Every derivative stays inside the ephemeral working
    directory and is deleted with the original attachment.
    """

    original_digest = sha256(image_path.read_bytes()).hexdigest()
    variants: list[dict[str, Any]] = [
        {
            "name": "original",
            "path": image_path,
            "content_sha256": original_digest,
        }
    ]
    try:
        from PIL import Image, ImageOps
    except ImportError:
        return variants
    try:
        with Image.open(image_path) as source:
            width, height = source.size
            if width <= 0 or height <= 0 or width * height > MAX_DECODED_IMAGE_PIXELS:
                return variants
            normalized = ImageOps.exif_transpose(source).convert("L")
            longest_side = max(normalized.size)
            if longest_side < 1_600:
                scale = min(3.0, 1_600 / max(1, longest_side))
                normalized = normalized.resize(
                    (
                        max(1, round(normalized.width * scale)),
                        max(1, round(normalized.height * scale)),
                    ),
                    Image.Resampling.LANCZOS,
                )
            enhanced = ImageOps.autocontrast(normalized, cutoff=1)
            enhanced_path = working_dir / "answer-grayscale-autocontrast.png"
            enhanced.save(enhanced_path, format="PNG", optimize=False)
            threshold = _otsu_threshold(enhanced.histogram())
            binary = enhanced.point(
                lambda pixel: 255 if pixel > threshold else 0,
                mode="1",
            )
            binary_path = working_dir / "answer-high-contrast-binary.png"
            binary.save(binary_path, format="PNG", optimize=False)
    except (OSError, ValueError, RuntimeError):
        return variants
    seen_digests = {original_digest}
    for name, path in (
        ("grayscale_autocontrast", enhanced_path),
        ("high_contrast_binary", binary_path),
    ):
        digest = sha256(path.read_bytes()).hexdigest()
        if digest in seen_digests:
            continue
        seen_digests.add(digest)
        variants.append({"name": name, "path": path, "content_sha256": digest})
    return variants


def _ocr_candidate(
    *,
    text: str,
    confidence: float,
    engine: str,
    preprocessing: str,
    preprocessing_digest: str,
    page_segmentation_mode: int | None,
) -> dict[str, Any]:
    return {
        "text": str(text).strip()[:MAX_RECOGNIZED_TEXT_CHARS],
        "confidence": _bounded_confidence(confidence),
        "engine": engine,
        "engine_family": (
            "apple_vision" if engine.startswith("apple_vision") else "tesseract"
        ),
        "preprocessing": preprocessing,
        "preprocessing_digest": preprocessing_digest,
        "page_segmentation_mode": page_segmentation_mode,
    }


def _select_ocr_consensus(candidates: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Select a transcription and summarize independent local corroboration."""

    usable = [
        dict(candidate)
        for candidate in candidates
        if str(candidate.get("text", "")).strip()
    ]
    if not usable:
        return {
            "text": "",
            "confidence": 0.0,
            "transcription_confidence": 0.0,
            "engine": "unavailable",
            "candidate_count": 0,
            "agreement_count": 0,
            "engine_count": 0,
            "preprocessing_count": 0,
            "transcription_corroborated": False,
            "material_disagreement": False,
        }
    groups: dict[str, list[dict[str, Any]]] = {}
    for candidate in usable:
        canonical = _canonical_transcription(str(candidate["text"]))
        if canonical:
            groups.setdefault(canonical, []).append(candidate)

    def group_summary(group: list[dict[str, Any]]) -> dict[str, Any]:
        confidences = [_bounded_confidence(item.get("confidence")) for item in group]
        engine_families = {str(item.get("engine_family")) for item in group}
        preprocessing_digests = {
            str(item.get("preprocessing_digest")) for item in group
        }
        average = sum(confidences) / len(confidences)
        cross_engine = len(engine_families) >= 2
        apple_supported = "apple_vision" in engine_families
        route_score = (
            average
            + 0.10 * int(cross_engine)
            + 0.04 * min(3, len(preprocessing_digests) - 1)
            + 0.03 * int(apple_supported)
        )
        best = max(
            group,
            key=lambda item: _candidate_score(
                str(item.get("text", "")), _bounded_confidence(item.get("confidence"))
            ),
        )
        corroborated = bool(
            (cross_engine and average >= MINIMUM_CORROBORATED_OCR_CONFIDENCE)
            or (
                len(group) >= 3
                and len(preprocessing_digests) >= 3
                and average >= MINIMUM_SINGLE_ENGINE_FORMULA_CONFIDENCE
            )
        )
        return {
            "group": group,
            "best": best,
            "average": average,
            "route_score": route_score,
            "engine_count": len(engine_families),
            "preprocessing_count": len(preprocessing_digests),
            "corroborated": corroborated,
        }

    summaries = [group_summary(group) for group in groups.values()]
    selected = max(
        summaries,
        key=lambda item: (
            float(item["route_score"]),
            _candidate_score(
                str(item["best"].get("text", "")),
                _bounded_confidence(item["best"].get("confidence")),
            ),
        ),
    )
    competing = [
        item
        for item in summaries
        if item is not selected
        and max(
            _bounded_confidence(candidate.get("confidence"))
            for candidate in item["group"]
        )
        >= MINIMUM_CORROBORATED_OCR_CONFIDENCE
    ]
    # A strong conflicting transcription must never disappear merely because
    # several variants from the selected engine agree with one another.  In
    # particular, three Tesseract preprocessing routes do not overrule a
    # high-confidence Apple Vision result that says something materially
    # different.  The caller will abstain and ask the learner to confirm.
    material_disagreement = bool(competing)
    transcription_confidence = float(selected["average"])
    if selected["corroborated"]:
        transcription_confidence += 0.08
    elif len(selected["group"]) >= 2:
        transcription_confidence += 0.03
    if material_disagreement:
        transcription_confidence -= 0.20
    best = selected["best"]
    return {
        "text": str(best["text"]),
        "confidence": _bounded_confidence(best.get("confidence")),
        "transcription_confidence": max(0.0, min(1.0, transcription_confidence)),
        "engine": str(best.get("engine", "unavailable")),
        "candidate_count": len(usable),
        "agreement_count": len(selected["group"]),
        "engine_count": int(selected["engine_count"]),
        "preprocessing_count": int(selected["preprocessing_count"]),
        "transcription_corroborated": bool(selected["corroborated"]),
        "material_disagreement": material_disagreement,
    }


def _run_tesseract(
    executable: str, image_path: Path, *, page_segmentation_mode: int
) -> tuple[str, float]:
    try:
        completed = subprocess.run(
            [
                executable,
                str(image_path),
                "stdout",
                "--psm",
                str(page_segmentation_mode),
                "-l",
                _ocr_languages(),
                "tsv",
            ],
            check=False,
            capture_output=True,
            timeout=25,
            env=_tesseract_environment(image_path.parent),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise LocalVisualEvidenceError("local OCR process failed") from exc
    if completed.returncode != 0:
        return "", 0.0
    try:
        decoded = completed.stdout.decode("utf-8", errors="replace")
        rows = csv.DictReader(io.StringIO(decoded), delimiter="\t")
        return _line_text(rows)
    except (csv.Error, UnicodeError):
        return "", 0.0


def extract_local_visual_evidence(
    image_bytes: bytes,
    mime_type: str,
    *,
    display_name: str = "learner-answer-image",
) -> dict[str, Any]:
    """Return bounded local OCR evidence without retaining the source image."""

    detected_mime = validate_local_image(image_bytes, mime_type)
    content_sha256 = sha256(image_bytes).hexdigest()
    apple_vision = _apple_vision_command()
    tesseract = _tesseract_command()
    candidates: list[dict[str, Any]] = []
    preprocessing_routes: list[str] = []
    apple_attempted = False
    apple_failed = False
    tesseract_attempted = False
    if apple_vision is not None or tesseract is not None:
        suffix = {
            "image/jpeg": ".jpg",
            "image/png": ".png",
            "image/webp": ".webp",
        }[detected_mime]
        with tempfile.TemporaryDirectory(prefix="tsm-learner-image-") as tmp_dir:
            working_dir = Path(tmp_dir)
            image_path = working_dir / f"answer{suffix}"
            image_path.write_bytes(image_bytes)
            variants = _prepare_ocr_variants(
                image_path,
                working_dir=working_dir,
            )
            preprocessing_routes = [str(item["name"]) for item in variants]
            if apple_vision is not None:
                apple_attempted = True
                try:
                    apple_text, apple_confidence = _run_apple_vision(
                        apple_vision,
                        image_path,
                        working_dir=working_dir,
                    )
                except LocalVisualEvidenceError:
                    apple_failed = True
                else:
                    candidates.append(
                        _ocr_candidate(
                            text=apple_text,
                            confidence=apple_confidence,
                            engine="apple_vision_local_objc_cli",
                            preprocessing="original",
                            preprocessing_digest=str(variants[0]["content_sha256"]),
                            page_segmentation_mode=None,
                        )
                    )
            apple_candidate = candidates[0] if candidates else None
            apple_needs_crosscheck = bool(
                apple_candidate is None
                or not str(apple_candidate.get("text", "")).strip()
                or _bounded_confidence(apple_candidate.get("confidence"))
                < MINIMUM_TRUSTED_OCR_CONFIDENCE
                or contains_formula_like_text(str(apple_candidate.get("text", "")))
            )
            if tesseract is not None and apple_needs_crosscheck:
                tesseract_attempted = True
                for variant in variants:
                    modes = (6, 11) if variant["name"] == "original" else (6, 7)
                    for page_segmentation_mode in modes:
                        text, confidence = _run_tesseract(
                            tesseract,
                            Path(variant["path"]),
                            page_segmentation_mode=page_segmentation_mode,
                        )
                        candidates.append(
                            _ocr_candidate(
                                text=text,
                                confidence=confidence,
                                engine="tesseract_local_cli",
                                preprocessing=str(variant["name"]),
                                preprocessing_digest=str(variant["content_sha256"]),
                                page_segmentation_mode=page_segmentation_mode,
                            )
                        )
    consensus = _select_ocr_consensus(candidates)
    recognized_text = str(consensus["text"])[:MAX_RECOGNIZED_TEXT_CHARS].strip()
    confidence = float(consensus["confidence"])
    transcription_confidence = float(consensus["transcription_confidence"])
    engine = str(consensus["engine"])
    if engine == "tesseract_local_cli" and apple_attempted:
        engine = (
            "tesseract_local_cli_fallback"
            if apple_failed
            else "tesseract_local_cli_crosscheck"
        )
    if apple_vision is None and tesseract is None:
        status = "extractor_unavailable"
    elif not recognized_text:
        status = (
            "extractor_failed"
            if apple_failed and not tesseract_attempted
            else "no_text_recognized"
        )
    elif consensus["material_disagreement"]:
        status = "conflicting_recognition"
    elif transcription_confidence < MINIMUM_TRUSTED_OCR_CONFIDENCE:
        status = "low_confidence"
    else:
        status = "recognized"
    formula_like_text = contains_formula_like_text(recognized_text)
    formula_transcription_established = bool(
        formula_like_text
        and status == "recognized"
        and consensus["transcription_corroborated"]
        and not consensus["material_disagreement"]
    )
    needs_confirmation = status != "recognized" or (
        formula_like_text and not formula_transcription_established
    )
    if status != "recognized":
        recognition_reliability = status
    elif consensus["transcription_corroborated"]:
        recognition_reliability = "corroborated_transcription"
    else:
        recognition_reliability = "single_route_transcription"
    safe_name = Path(str(display_name or "learner-answer-image")).name[:120]
    return {
        "schema": VISUAL_EVIDENCE_SCHEMA,
        "source_modality": "image",
        "source_kind": "learner_answer_attachment",
        "display_name": safe_name or "learner-answer-image",
        "mime_type": detected_mime,
        "byte_size": len(image_bytes),
        "content_sha256": content_sha256,
        "engine": engine,
        "status": status,
        "recognized_text": recognized_text,
        "confidence": round(confidence, 4),
        "confidence_semantics": (
            "selected_engine_native_ocr_heuristic_not_formula_correctness"
        ),
        "transcription_confidence": round(transcription_confidence, 4),
        "transcription_confidence_semantics": (
            "local_cross_route_agreement_heuristic_not_answer_correctness"
        ),
        "recognition_reliability": recognition_reliability,
        "ocr_candidate_count": int(consensus["candidate_count"]),
        "ocr_agreement_count": int(consensus["agreement_count"]),
        "ocr_independent_engine_count": int(consensus["engine_count"]),
        "ocr_preprocessing_count": int(consensus["preprocessing_count"]),
        "ocr_preprocessing_routes": preprocessing_routes,
        "ocr_transcription_corroborated": bool(consensus["transcription_corroborated"]),
        "ocr_material_disagreement": bool(consensus["material_disagreement"]),
        "formula_like_text_detected": formula_like_text,
        "formula_accuracy_established": False,
        "formula_transcription_established": formula_transcription_established,
        "content_style_assessment": "not_classified",
        "handwriting_recognition_established": False,
        "extractor_fallback_used": apple_failed and tesseract_attempted,
        "needs_student_confirmation": needs_confirmation,
        "raw_media_retained": False,
        "remote_media_sent": False,
        "remote_representation": "bounded_redacted_ocr_text_only",
    }


def compose_visual_evidence_text(
    learner_text: str, evidence: Iterable[dict[str, Any]]
) -> str:
    """Compose the exact auditable text representation sent to the Agent."""

    typed = str(learner_text or "").strip()
    evidence_items = list(evidence)
    # Preserve the original text-only contract byte-for-byte (apart from the
    # established outer whitespace trim).  The visual envelope is provenance
    # for an attached image, not a generic wrapper for every learner turn;
    # adding it to text-only answers breaks exact short-answer matching and
    # question/no-progress heuristics downstream.
    if not evidence_items:
        return typed
    parts: list[str] = []
    if typed:
        parts.append("[STUDENT_TYPED_TEXT]\n" + typed)
    consistency = assess_typed_visual_consistency(typed, evidence_items)
    for index, raw in enumerate(evidence_items, 1):
        text = str(raw.get("recognized_text", "")).strip()
        status = str(raw.get("status", "unavailable"))
        confidence = raw.get("confidence")
        confidence_text = (
            f"{float(confidence):.2f}"
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            else "unknown"
        )
        transcription_confidence = raw.get("transcription_confidence")
        transcription_confidence_text = (
            f"{float(transcription_confidence):.2f}"
            if isinstance(transcription_confidence, (int, float))
            and not isinstance(transcription_confidence, bool)
            else "unknown"
        )
        parts.append(
            "\n".join(
                [
                    f"[LOCAL_VISUAL_EVIDENCE {index}]",
                    "原图未发送给远程模型；以下内容由本机 OCR 提取，可能存在识别误差。",
                    f"status={status}; confidence={confidence_text}; "
                    f"transcription_confidence={transcription_confidence_text}; "
                    f"corroborated={str(bool(raw.get('ocr_transcription_corroborated'))).lower()}; "
                    "student_confirmed_transcription="
                    f"{str(bool(raw.get('student_confirmed_recognized_text'))).lower()}; "
                    f"needs_confirmation={str(bool(raw.get('needs_student_confirmation'))).lower()}",
                    "recognized_text:",
                    text or "（未识别到可靠文字）",
                ]
            )
        )
    if consistency["relation"] != "not_comparable":
        consistency_lines = [
            "[TYPED_VISUAL_CONSISTENCY]",
            f"relation={consistency['relation']}; "
            f"needs_confirmation={str(bool(consistency['needs_student_confirmation'])).lower()}",
        ]
        if consistency["possible_conflict"]:
            consistency_lines.append(
                "学生键入文本与本机 OCR 文字不一致；不得把两者合并成一个答案，"
                "也不得据此判对或判错，应先请学生确认。"
            )
        parts.append("\n".join(consistency_lines))
    return "\n\n".join(parts).strip()
