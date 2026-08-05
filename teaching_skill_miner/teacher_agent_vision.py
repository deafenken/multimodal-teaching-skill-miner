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
from typing import Any, Iterable


VISUAL_EVIDENCE_SCHEMA = "teaching_skill_miner.local_visual_evidence.v1"
SUPPORTED_IMAGE_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
MAX_IMAGE_BYTES = 4 * 1024 * 1024
MAX_RECOGNIZED_TEXT_CHARS = 2_400
MINIMUM_TRUSTED_OCR_CONFIDENCE = 0.58
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
        character.isalnum() or character in accepted_symbols
        for character in characters
    )
    readable_ratio = readable_count / len(characters) if characters else 0.0
    diversity_ratio = (
        min(1.0, len(set(characters)) / min(len(characters), 32))
        if characters
        else 0.0
    )
    return (
        int(bool(stripped)),
        bounded_confidence,
        round(readable_ratio, 6),
        round(diversity_ratio, 6),
        min(len(stripped), MAX_RECOGNIZED_TEXT_CHARS),
    )


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
    recognized_text = ""
    confidence = 0.0
    engine = "unavailable"
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
            if apple_vision is not None:
                apple_attempted = True
                try:
                    recognized_text, confidence = _run_apple_vision(
                        apple_vision,
                        image_path,
                        working_dir=working_dir,
                    )
                except LocalVisualEvidenceError:
                    apple_failed = True
                    engine = "apple_vision_local_objc_cli_failed"
                else:
                    engine = "apple_vision_local_objc_cli"
            if not recognized_text and tesseract is not None:
                tesseract_attempted = True
                candidates = [
                    _run_tesseract(
                        tesseract,
                        image_path,
                        page_segmentation_mode=6,
                    ),
                    _run_tesseract(
                        tesseract,
                        image_path,
                        page_segmentation_mode=11,
                    ),
                ]
                recognized_text, confidence = max(
                    candidates,
                    key=lambda candidate: _candidate_score(candidate[0], candidate[1]),
                )
                engine = (
                    "tesseract_local_cli_fallback"
                    if apple_attempted
                    else "tesseract_local_cli"
                )
    recognized_text = recognized_text[:MAX_RECOGNIZED_TEXT_CHARS].strip()
    if apple_vision is None and tesseract is None:
        status = "extractor_unavailable"
    elif not recognized_text:
        status = (
            "extractor_failed"
            if apple_failed and not tesseract_attempted
            else "no_text_recognized"
        )
    elif confidence < MINIMUM_TRUSTED_OCR_CONFIDENCE:
        status = "low_confidence"
    else:
        status = "recognized"
    formula_like_text = contains_formula_like_text(recognized_text)
    needs_confirmation = status != "recognized" or formula_like_text
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
        "confidence_semantics": ("engine_native_ocr_heuristic_not_formula_correctness"),
        "formula_like_text_detected": formula_like_text,
        "formula_accuracy_established": False,
        "extractor_fallback_used": engine == "tesseract_local_cli_fallback",
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
    for index, raw in enumerate(evidence_items, 1):
        text = str(raw.get("recognized_text", "")).strip()
        status = str(raw.get("status", "unavailable"))
        confidence = raw.get("confidence")
        confidence_text = (
            f"{float(confidence):.2f}"
            if isinstance(confidence, (int, float)) and not isinstance(confidence, bool)
            else "unknown"
        )
        parts.append(
            "\n".join(
                [
                    f"[LOCAL_VISUAL_EVIDENCE {index}]",
                    "原图未发送给远程模型；以下内容由本机 OCR 提取，可能存在识别误差。",
                    f"status={status}; confidence={confidence_text}; "
                    f"needs_confirmation={str(bool(raw.get('needs_student_confirmation'))).lower()}",
                    "recognized_text:",
                    text or "（未识别到可靠文字）",
                ]
            )
        )
    return "\n\n".join(parts).strip()
