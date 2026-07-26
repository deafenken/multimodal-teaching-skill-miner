"""Private TeachObs source-caption retrieval and released-text auditing.

This module deliberately separates three kinds of evidence:

* yt-dlp's current description of a lesson's own YouTube caption tracks;
* timing and integrity properties measured from a retrieved VTT file; and
* automatic quality diagnostics for TeachObs' released 15-second text files.

None of those establishes caption content accuracy, word error rate, human
authorship, or independent double review.  Full caption text, source URLs,
lesson/video identifiers, and window-level results remain private.  The public
receipt builder emits aggregate counts only.
"""

from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from hashlib import sha256
import html
import importlib.util
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile
from typing import Any, Callable, Sequence
import unicodedata

from .full_video_dataset import file_sha256
from .io_utils import (
    ensure_private_directory,
    ensure_private_file,
    read_json,
    write_json,
)
from .preprocess import parse_srt_or_vtt
from .teachobs_media import (
    DATASET_ID,
    PINNED_REPOSITORY_COMMIT,
    PLAN_SCHEMA,
    build_teachobs_media_plan,
    validate_teachobs_cookies_from_browser,
    validate_teachobs_ytdlp_transport,
)


PRIVATE_AUDIT_SCHEMA = "teaching_skill_miner.teachobs_private_caption_audit.v1"
PUBLIC_RECEIPT_SCHEMA = "teaching_skill_miner.teachobs_caption_receipt.v1"
SCENE_SECONDS = 15.0
_LANGUAGE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_LESSON_ID_RE = re.compile(r"^S(?:[1-9]|[12][0-9]|30)$")
_TOKEN_RE = re.compile(r"[^\W_]+(?:['’][^\W_]+)?", re.UNICODE)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_YTDLP_YOUTUBE_CLIENTS = frozenset({"android_vr"})
_PUBLIC_AGGREGATE_FIELDS = (
    "lesson_count",
    "caption_timeline_audited_lesson_count",
    "caption_unavailable_or_failed_lesson_count",
    "reused_hash_verified_lesson_count",
    "retrieved_now_lesson_count",
    "caption_track_count",
    "manual_creator_provided_track_count",
    "youtube_automatic_caption_track_count",
    "original_language_role_track_count",
    "english_role_track_count",
    "timeline_span_coverage_mean_across_tracks",
    "timeline_span_coverage_ge_0_90_track_count",
    "released_transcript_expected_file_count",
    "released_transcript_readable_utf8_file_count",
    "released_transcript_file_coverage_fraction",
    "released_transcript_nonempty_file_count",
    "released_transcript_nonempty_file_fraction",
    "released_transcript_character_count",
    "released_transcript_nonwhitespace_character_count",
    "released_transcript_nonwhitespace_character_fraction",
    "released_transcript_abnormal_repetition_file_count",
    "released_transcript_abnormal_repetition_file_fraction",
    "track_level_normalized_token_overlap_mean",
    "track_level_normalized_character_bigram_overlap_mean",
    "formal_caption_timeline_audit_completed",
)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _sha256_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _mean(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _safe_lesson_id(value: Any) -> str:
    lesson_id = str(value).strip()
    if not _LESSON_ID_RE.fullmatch(lesson_id):
        raise ValueError(f"unsafe TeachObs lesson id: {value!r}")
    return lesson_id


def _safe_transcript_filename(value: Any) -> str:
    text = str(value).strip()
    path = PurePosixPath(text)
    if (
        not text
        or path.is_absolute()
        or len(path.parts) != 1
        or path.name in {"", ".", ".."}
        or "\\" in text
        or "\x00" in text
        or not text.endswith(".txt")
    ):
        raise ValueError(f"unsafe TeachObs transcript filename: {value!r}")
    return text


def _safe_language(value: Any) -> str | None:
    language = str(value or "").strip()
    return language if _LANGUAGE_RE.fullmatch(language) else None


def _validate_ytdlp_youtube_client(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _YTDLP_YOUTUBE_CLIENTS:
        raise ValueError("yt_dlp_youtube_client uses an unsupported target")
    return value


def _command_prefix(command: Sequence[str] | str | None) -> list[str]:
    if command is None:
        executable = shutil.which("yt-dlp")
        if executable:
            return [executable]
        if importlib.util.find_spec("yt_dlp") is not None:
            return [os.fspath(Path(os.sys.executable)), "-m", "yt_dlp"]
        raise RuntimeError(
            "TeachObs caption retrieval requires yt-dlp; install it privately "
            "or pass yt_dlp_command=(python, '-m', 'yt_dlp')"
        )
    parts = [command] if isinstance(command, str) else list(command)
    if not parts or any(not isinstance(part, str) or not part for part in parts):
        raise ValueError("yt_dlp_command must contain non-empty strings")
    executable = shutil.which(parts[0])
    if executable is None:
        raise RuntimeError(f"TeachObs caption retrieval cannot find `{parts[0]}`")
    parts[0] = executable
    return parts


def _atomic_write_bytes(path: Path, payload: bytes) -> Path:
    ensure_private_directory(path.parent)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return ensure_private_file(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _normalize_text(value: str) -> str:
    text = html.unescape(value)
    text = unicodedata.normalize("NFKC", text).casefold()
    return " ".join(_TOKEN_RE.findall(text))


def _normalized_tokens(value: str) -> list[str]:
    return _normalize_text(value).split()


def _normalized_characters(value: str) -> list[str]:
    normalized = unicodedata.normalize("NFKC", html.unescape(value)).casefold()
    return [character for character in normalized if character.isalnum()]


def _multiset_overlap(left: Sequence[str], right: Sequence[str]) -> float | None:
    """Return symmetric multiset overlap in [0, 1], or None for two empties."""

    if not left and not right:
        return None
    if not left or not right:
        return 0.0
    intersection = sum((Counter(left) & Counter(right)).values())
    return round((2.0 * intersection) / (len(left) + len(right)), 6)


def normalized_token_overlap(left: str, right: str) -> float | None:
    """Symmetric normalized token overlap; this is explicitly not WER."""

    return _multiset_overlap(_normalized_tokens(left), _normalized_tokens(right))


def normalized_character_overlap(left: str, right: str) -> float | None:
    """Symmetric normalized character-bigram overlap; this is not accuracy."""

    left_characters = _normalized_characters(left)
    right_characters = _normalized_characters(right)
    size = 2 if len(left_characters) >= 2 and len(right_characters) >= 2 else 1
    left_ngrams = [
        "".join(left_characters[index : index + size])
        for index in range(max(0, len(left_characters) - size + 1))
    ]
    right_ngrams = [
        "".join(right_characters[index : index + size])
        for index in range(max(0, len(right_characters) - size + 1))
    ]
    return _multiset_overlap(left_ngrams, right_ngrams)


def repetition_diagnostics(text: str) -> dict[str, Any]:
    """Flag obvious repetition using declared, deterministic heuristics."""

    tokens = _normalized_tokens(text)
    token_count = len(tokens)
    counts = Counter(tokens)
    dominant_fraction = max(counts.values(), default=0) / max(1, token_count)
    unique_fraction = len(counts) / max(1, token_count)
    longest_run = 0
    current_run = 0
    previous: str | None = None
    for token in tokens:
        current_run = current_run + 1 if token == previous else 1
        longest_run = max(longest_run, current_run)
        previous = token
    raw = text.encode("utf-8")
    # A small standard-library compression probe catches repeated multi-token
    # phrases without retaining or exposing the phrase itself.
    import zlib

    compression_ratio = len(zlib.compress(raw, level=9)) / max(1, len(raw))
    reasons: list[str] = []
    if token_count >= 20 and longest_run >= 8:
        reasons.append("identical_token_run_ge_8")
    if token_count >= 32 and dominant_fraction >= 0.45:
        reasons.append("dominant_token_fraction_ge_0_45")
    if (
        token_count >= 50
        and unique_fraction <= 0.12
        and compression_ratio <= 0.30
    ):
        reasons.append("low_lexical_diversity_and_high_compressibility")
    return {
        "heuristic_version": "teachobs_repetition_v1",
        "token_count": token_count,
        "unique_token_fraction": round(unique_fraction, 6),
        "dominant_token_fraction": round(dominant_fraction, 6),
        "longest_identical_token_run": longest_run,
        "utf8_compression_ratio": round(compression_ratio, 6),
        "abnormal_repetition_flag": bool(reasons),
        "reasons": reasons,
    }


def _load_released_scene_rows(
    repository_root: Path,
    lesson: dict[str, Any],
) -> list[dict[str, Any]]:
    lesson_id = _safe_lesson_id(lesson.get("lesson_id"))
    manifest_path = (
        repository_root / "data" / "scenes" / lesson_id / "manifest.jsonl"
    )
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError(f"TeachObs scene manifest is missing for {lesson_id}")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8-sig").splitlines(), 1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"invalid TeachObs scene JSON for {lesson_id}, line {line_number}"
            ) from exc
        expected_scene_no = len(rows) + 1
        if (
            not isinstance(row, dict)
            or row.get("id") != lesson_id
            or row.get("scene_no") != expected_scene_no
        ):
            raise ValueError(f"invalid TeachObs scene identity for {lesson_id}")
        transcript_name = _safe_transcript_filename(row.get("transcript_file"))
        transcript_path = manifest_path.parent / transcript_name
        if transcript_path.is_symlink() or not transcript_path.is_file():
            raise ValueError(f"TeachObs scene transcript is missing for {lesson_id}")
        try:
            start = float(row.get("start"))
            end = float(row.get("end"))
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"invalid TeachObs scene timing for {lesson_id}") from exc
        if (
            not math.isclose(start, (expected_scene_no - 1) * SCENE_SECONDS)
            or not math.isclose(end, expected_scene_no * SCENE_SECONDS)
        ):
            raise ValueError(f"TeachObs scene timing is not fixed at 15 seconds: {lesson_id}")
        try:
            text = transcript_path.read_text(encoding="utf-8-sig")
        except UnicodeDecodeError as exc:
            raise ValueError(
                f"TeachObs scene transcript is not valid UTF-8 for {lesson_id}"
            ) from exc
        rows.append(
            {
                "scene_no": expected_scene_no,
                "start": start,
                "end": end,
                "transcript_sha256": file_sha256(transcript_path),
                "text": text,
                "repetition": repetition_diagnostics(text),
            }
        )
    if len(rows) != lesson.get("scene_count"):
        raise ValueError(f"TeachObs scene count differs from media plan for {lesson_id}")
    return rows


def _audit_released_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    characters = [len(row["text"]) for row in rows]
    nonwhitespace = [
        sum(not character.isspace() for character in row["text"]) for row in rows
    ]
    nonempty = [count > 0 for count in nonwhitespace]
    abnormal = [
        bool(row["repetition"]["abnormal_repetition_flag"]) for row in rows
    ]
    return {
        "expected_file_count": len(rows),
        "readable_utf8_file_count": len(rows),
        "file_coverage_fraction": 1.0,
        "nonempty_file_count": sum(nonempty),
        "nonempty_file_fraction": round(sum(nonempty) / max(1, len(rows)), 6),
        "character_count": sum(characters),
        "nonwhitespace_character_count": sum(nonwhitespace),
        "nonwhitespace_character_fraction": round(
            sum(nonwhitespace) / max(1, sum(characters)), 6
        ),
        "abnormal_repetition_file_count": sum(abnormal),
        "abnormal_repetition_file_fraction": round(
            sum(abnormal) / max(1, len(rows)), 6
        ),
        "repetition_heuristic_version": "teachobs_repetition_v1",
        "content_accuracy_established": False,
        "human_review_established": False,
    }


def _cue_union_duration(segments: list[dict[str, Any]]) -> float:
    intervals = sorted((float(row["start"]), float(row["end"])) for row in segments)
    total = 0.0
    current_start: float | None = None
    current_end: float | None = None
    for start, end in intervals:
        if current_start is None:
            current_start, current_end = start, end
        elif start <= float(current_end):
            current_end = max(float(current_end), end)
        else:
            total += float(current_end) - current_start
            current_start, current_end = start, end
    if current_start is not None:
        total += float(current_end) - current_start
    return total


def _audit_vtt(payload: bytes, *, reference_duration_seconds: float) -> dict[str, Any]:
    try:
        text = payload.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("retrieved TeachObs caption is not UTF-8") from exc
    segments = parse_srt_or_vtt(text)
    if not segments:
        raise ValueError("retrieved TeachObs caption contains no timed cues")
    previous_start = -1.0
    input_order_monotonic = True
    for segment in segments:
        start = float(segment["start"])
        end = float(segment["end"])
        if (
            not math.isfinite(start)
            or not math.isfinite(end)
            or start < 0
            or end <= start
        ):
            raise ValueError("retrieved TeachObs caption has invalid cue timing")
        input_order_monotonic = input_order_monotonic and start >= previous_start
        previous_start = start
    # YouTube can serialize separate VTT regions in non-monotonic block order.
    # Sorting valid timed cues is deterministic and does not change their text
    # or timestamps; the receipt still records that normalization occurred.
    segments.sort(key=lambda row: (float(row["start"]), float(row["end"])))
    first = min(float(row["start"]) for row in segments)
    last = max(float(row["end"]) for row in segments)
    span = max(0.0, last - first)
    union = _cue_union_duration(segments)
    return {
        "segments": segments,
        "cue_count": len(segments),
        "first_cue_start_seconds": round(first, 3),
        "last_cue_end_seconds": round(last, 3),
        "timeline_span_seconds": round(span, 3),
        "timeline_span_coverage_fraction": round(
            min(1.0, span / reference_duration_seconds), 6
        ),
        "cue_union_duration_seconds": round(union, 3),
        "cue_union_coverage_fraction": round(
            min(1.0, union / reference_duration_seconds), 6
        ),
        "endpoint_within_reference_tolerance": bool(
            last <= reference_duration_seconds + SCENE_SECONDS
        ),
        "input_cue_order_monotonic": input_order_monotonic,
        "cue_order_normalized_for_alignment": not input_order_monotonic,
        "timeline_audit_completed": True,
    }


def _caption_text_for_window(
    segments: list[dict[str, Any]], *, start: float, end: float
) -> str:
    values: list[str] = []
    seen: set[str] = set()
    for segment in segments:
        if float(segment["start"]) >= end or float(segment["end"]) <= start:
            continue
        value = str(segment["text"]).strip()
        normalized = _normalize_text(value)
        if value and normalized not in seen:
            seen.add(normalized)
            values.append(value)
    return " ".join(values)


def align_caption_to_released_scenes(
    rows: list[dict[str, Any]],
    segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare caption cues with released 15-second text without calling it WER."""

    per_scene: list[dict[str, Any]] = []
    token_values: list[float] = []
    character_values: list[float] = []
    caption_nonempty = 0
    paired_nonempty = 0
    for row in rows:
        caption_text = _caption_text_for_window(
            segments, start=float(row["start"]), end=float(row["end"])
        )
        released_text = str(row["text"])
        caption_has_text = bool(_normalized_characters(caption_text))
        released_has_text = bool(_normalized_characters(released_text))
        caption_nonempty += caption_has_text
        paired_nonempty += caption_has_text and released_has_text
        token_overlap = normalized_token_overlap(released_text, caption_text)
        character_overlap = normalized_character_overlap(released_text, caption_text)
        if caption_has_text and released_has_text:
            if token_overlap is not None:
                token_values.append(token_overlap)
            if character_overlap is not None:
                character_values.append(character_overlap)
        per_scene.append(
            {
                "scene_no": row["scene_no"],
                "caption_nonempty": caption_has_text,
                "released_nonempty": released_has_text,
                "normalized_token_overlap": token_overlap,
                "normalized_character_bigram_overlap": character_overlap,
                "overlap_is_wer": False,
                "overlap_is_content_accuracy": False,
            }
        )
    return {
        "scene_count": len(rows),
        "caption_nonempty_scene_count": caption_nonempty,
        "caption_nonempty_scene_fraction": round(
            caption_nonempty / max(1, len(rows)), 6
        ),
        "paired_nonempty_scene_count": paired_nonempty,
        "normalized_token_overlap_mean_on_paired_nonempty": _mean(token_values),
        "normalized_character_bigram_overlap_mean_on_paired_nonempty": _mean(
            character_values
        ),
        "metric_interpretation": (
            "symmetric normalized multiset overlap after 15-second cue/window "
            "alignment; not edit-distance WER and not caption content accuracy"
        ),
        "word_error_rate_established": False,
        "content_accuracy_established": False,
        "per_scene": per_scene,
    }


def _caption_maps(metadata: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manual = metadata.get("subtitles")
    automatic = metadata.get("automatic_captions")
    return (
        manual if isinstance(manual, dict) else {},
        automatic if isinstance(automatic, dict) else {},
    )


def _available_languages(mapping: dict[str, Any]) -> list[str]:
    return sorted(
        language
        for language, formats in mapping.items()
        if _safe_language(language) is not None
        and isinstance(formats, list)
        and bool(formats)
    )


def _choose_language(
    candidates: Sequence[str], preferred: str | None, *, english: bool
) -> str | None:
    if preferred and preferred in candidates:
        return preferred
    if preferred:
        prefix = preferred.split("-", 1)[0].casefold()
        matches = [item for item in candidates if item.casefold().split("-", 1)[0] == prefix]
        if matches:
            return sorted(matches, key=lambda value: (len(value), value))[0]
    if english:
        matches = [item for item in candidates if item.casefold().split("-", 1)[0] == "en"]
        if "en" in matches:
            return "en"
        if matches:
            return sorted(matches, key=lambda value: (len(value), value))[0]
    return None


def _select_caption_tracks(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    manual, automatic = _caption_maps(metadata)
    manual_languages = _available_languages(manual)
    automatic_languages = _available_languages(automatic)
    declared_original = _safe_language(metadata.get("language"))

    def choose(preferred: str | None, *, english: bool) -> tuple[str, str] | None:
        language = _choose_language(manual_languages, preferred, english=english)
        if language:
            return language, "manual_creator_provided"
        language = _choose_language(automatic_languages, preferred, english=english)
        if language:
            return language, "youtube_automatic_caption"
        return None

    selected: dict[str, dict[str, Any]] = {}
    original = choose(declared_original, english=False)
    if original is None and declared_original is None:
        # Metadata occasionally omits the original language.  Prefer a manual
        # track, then an automatic track, while recording that the role was
        # inferred rather than declared.
        fallback_languages = manual_languages or automatic_languages
        if fallback_languages:
            first = fallback_languages[0]
            source = (
                "manual_creator_provided"
                if first in manual_languages
                else "youtube_automatic_caption"
            )
            original = (first, source)
    if original:
        language, track_type = original
        selected[language] = {
            "language": language,
            "roles": ["original_language"],
            "original_language_declared_by_youtube": bool(
                declared_original
                and language.casefold().split("-", 1)[0]
                == declared_original.casefold().split("-", 1)[0]
            ),
            "track_type": track_type,
        }
    english = choose("en", english=True)
    if english:
        language, track_type = english
        if language in selected:
            selected[language]["roles"].append("english")
        else:
            selected[language] = {
                "language": language,
                "roles": ["english"],
                "original_language_declared_by_youtube": False,
                "track_type": track_type,
            }
    return [selected[key] for key in sorted(selected)]


def _append_ytdlp_transport_options(
    command: list[str],
    *,
    cookies_from_browser: str | None,
    yt_dlp_direct: bool,
    yt_dlp_impersonate: str | None,
) -> None:
    """Append only the validated, ephemeral yt-dlp transport opt-ins.

    The browser selector is passed directly to the child process.  It is never
    converted to a cookie jar or returned from this helper for persistence.
    """

    if cookies_from_browser is not None:
        command.extend(["--cookies-from-browser", cookies_from_browser])
    if yt_dlp_direct:
        # An explicit empty proxy bypasses inherited HTTP(S)_PROXY variables.
        # This interface never accepts a proxy URL.
        command.extend(["--proxy", ""])
    if yt_dlp_impersonate is not None:
        command.extend(["--impersonate", yt_dlp_impersonate])


def _metadata_command(
    prefix: Sequence[str],
    *,
    source_url: str,
    js_runtime: str | None,
    cookies_from_browser: str | None = None,
    yt_dlp_direct: bool = False,
    yt_dlp_impersonate: str | None = None,
    yt_dlp_youtube_client: str | None = None,
) -> list[str]:
    youtube_client = _validate_ytdlp_youtube_client(yt_dlp_youtube_client)
    command = [
        *prefix,
        "--no-playlist",
        "--skip-download",
        "--dump-single-json",
        "--no-warnings",
        "--no-cache-dir",
        "--no-remote-components",
        "--retries",
        "5",
        "--retry-sleep",
        "http:linear=2::10",
        "--sleep-requests",
        "1",
    ]
    if js_runtime:
        command.extend(["--js-runtimes", js_runtime])
    if youtube_client is not None:
        command.extend(
            [
                "--extractor-args",
                f"youtube:player_client={youtube_client}",
            ]
        )
    _append_ytdlp_transport_options(
        command,
        cookies_from_browser=cookies_from_browser,
        yt_dlp_direct=yt_dlp_direct,
        yt_dlp_impersonate=yt_dlp_impersonate,
    )
    command.append(source_url)
    return command


def _download_command(
    prefix: Sequence[str],
    *,
    source_url: str,
    lesson_id: str,
    languages: Sequence[str],
    staging_directory: Path,
    js_runtime: str | None,
    cookies_from_browser: str | None = None,
    yt_dlp_direct: bool = False,
    yt_dlp_impersonate: str | None = None,
    yt_dlp_youtube_client: str | None = None,
) -> list[str]:
    youtube_client = _validate_ytdlp_youtube_client(yt_dlp_youtube_client)
    language_pattern = ",".join(f"^{re.escape(value)}$" for value in languages)
    command = [
        *prefix,
        "--no-playlist",
        "--skip-download",
        "--write-subs",
        "--write-auto-subs",
        "--sub-format",
        "vtt",
        "--sub-langs",
        language_pattern,
        "--no-write-info-json",
        "--no-write-thumbnail",
        "--no-write-comments",
        "--no-cache-dir",
        "--no-remote-components",
        "--retries",
        "5",
        "--retry-sleep",
        "http:linear=2::10",
        "--sleep-requests",
        "1",
        "--sleep-subtitles",
        "2",
        "--paths",
        str(staging_directory),
        "--output",
        # yt-dlp itself inserts the selected subtitle language before the
        # extension.  Adding %(language)s here would duplicate or mislabel it
        # with the video's metadata language for translated tracks.
        f"{lesson_id}.%(ext)s",
    ]
    if js_runtime:
        command.extend(["--js-runtimes", js_runtime])
    if youtube_client is not None:
        command.extend(
            [
                "--extractor-args",
                f"youtube:player_client={youtube_client}",
            ]
        )
    _append_ytdlp_transport_options(
        command,
        cookies_from_browser=cookies_from_browser,
        yt_dlp_direct=yt_dlp_direct,
        yt_dlp_impersonate=yt_dlp_impersonate,
    )
    command.append(source_url)
    return command


def _run_command(
    command: list[str],
    *,
    runner: Callable[..., Any],
    timeout_seconds: int,
    purpose: str,
) -> Any:
    try:
        result = runner(
            command,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"yt-dlp timed out during {purpose}") from exc
    if result.returncode != 0:
        # Child diagnostics can echo a browser profile, account identifier,
        # source URL, proxy URL, or local path.  They are intentionally not
        # placed in an exception that is later persisted in the private audit.
        diagnostic = str(result.stderr or result.stdout or "").casefold()
        if "429" in diagnostic or "too many requests" in diagnostic:
            failure_class = "http_rate_limited"
        elif "unavailable" in diagnostic or "private video" in diagnostic:
            failure_class = "source_unavailable"
        elif "subtitle" in diagnostic or "caption" in diagnostic:
            failure_class = "caption_transport_failure"
        else:
            failure_class = "yt_dlp_nonzero_exit"
        raise RuntimeError(
            f"yt-dlp failed during {purpose}; failure_class={failure_class}; "
            "raw child diagnostics were not retained"
        )
    return result


def _metadata_from_result(result: Any) -> dict[str, Any]:
    lines = [line for line in str(result.stdout or "").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("yt-dlp returned no caption metadata JSON")
    try:
        metadata = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError("yt-dlp returned invalid caption metadata JSON") from exc
    if not isinstance(metadata, dict):
        raise RuntimeError("yt-dlp caption metadata is not an object")
    return metadata


def _caption_path_for_language(
    staging_directory: Path, *, lesson_id: str, language: str
) -> Path:
    exact = staging_directory / f"{lesson_id}.{language}.vtt"
    if exact.is_file() and not exact.is_symlink():
        return exact
    matches = [
        path
        for path in staging_directory.glob(f"{lesson_id}.{language}.*")
        if path.is_file() and not path.is_symlink() and path.suffix.casefold() == ".vtt"
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"yt-dlp did not produce exactly one VTT for requested language {language}"
        )
    return matches[0]


def _private_error(exc: BaseException) -> str:
    """Return an allowlisted failure class without child text or local paths."""

    value = str(exc).casefold()
    if "http_rate_limited" in value:
        failure_class = "http_rate_limited"
    elif "source_unavailable" in value:
        failure_class = "source_unavailable"
    elif "caption_transport_failure" in value:
        failure_class = "caption_transport_failure"
    elif "timed out" in value:
        failure_class = "yt_dlp_timeout"
    elif "metadata" in value:
        failure_class = "caption_metadata_invalid_or_missing"
    elif "vtt" in value or "subtitle" in value:
        failure_class = "caption_file_invalid_or_missing"
    elif isinstance(exc, OSError):
        failure_class = "local_io_failure"
    elif isinstance(exc, (ValueError, json.JSONDecodeError)):
        failure_class = "caption_validation_failure"
    else:
        failure_class = "caption_retrieval_failure"
    return f"failure_class={failure_class}; raw details redacted"


def _network_transport_record(
    *,
    browser_family: str | None,
    yt_dlp_direct: bool,
    yt_dlp_impersonate: str | None,
    yt_dlp_youtube_client: str | None,
) -> dict[str, Any]:
    """Build the only transport metadata allowed into persistent artifacts."""

    return {
        "transport": "yt-dlp_https_youtube",
        "transport_mode": (
            "direct_environment_proxy_bypass"
            if yt_dlp_direct
            else "inherited_environment_proxy_policy"
        ),
        "network_retrieval_performed": True,
        "direct_environment_proxy_bypass_enabled": yt_dlp_direct,
        "environment_proxy_policy": (
            "explicit_bypass" if yt_dlp_direct else "inherited_environment_default"
        ),
        "proxy_url_recorded": False,
        "credentials_used": browser_family is not None,
        "browser_family": browser_family,
        "browser_profile_recorded": False,
        "account_identifier_recorded": False,
        "cookie_file_export_requested": False,
        "cookie_material_recorded": False,
        "http_impersonation_requested": yt_dlp_impersonate is not None,
        "http_impersonation_target": yt_dlp_impersonate,
        "youtube_player_client_requested": yt_dlp_youtube_client is not None,
        "youtube_player_client": yt_dlp_youtube_client,
    }


def _local_reuse_transport_record() -> dict[str, Any]:
    return {
        "transport": "local_hash_verified_caption_reuse",
        "transport_mode": "local_hash_verified_no_network",
        "network_retrieval_performed": False,
        "direct_environment_proxy_bypass_enabled": False,
        "environment_proxy_policy": "not_applicable_no_network",
        "proxy_url_recorded": False,
        "credentials_used": False,
        "browser_family": None,
        "browser_profile_recorded": False,
        "account_identifier_recorded": False,
        "cookie_file_export_requested": False,
        "cookie_material_recorded": False,
        "http_impersonation_requested": False,
        "http_impersonation_target": None,
        "youtube_player_client_requested": False,
        "youtube_player_client": None,
    }


def _load_reusable_lesson_record(
    output_directory: Path,
    *,
    lesson: dict[str, Any],
    released_audit: dict[str, Any],
    verified_at_utc: str,
) -> dict[str, Any] | None:
    """Reuse only a fully hash-verified successful private caption record."""

    lesson_id = _safe_lesson_id(lesson.get("lesson_id"))
    record_path = output_directory / "lessons" / f"{lesson_id}.json"
    if record_path.is_symlink() or not record_path.is_file():
        return None
    try:
        record = read_json(record_path)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(record, dict) or record.get("status") != "caption_timeline_audited":
        return None
    expected_url_digest = _sha256_bytes(str(lesson["source_url"]).encode("utf-8"))
    if (
        record.get("lesson_id") != lesson_id
        or record.get("source_url_sha256") != expected_url_digest
        or record.get("scene_manifest_sha256") != lesson["scene_manifest_sha256"]
        or record.get("reference_duration_seconds")
        != lesson["reference_duration_seconds"]
    ):
        return None
    tracks = record.get("tracks")
    if (
        not isinstance(tracks, list)
        or not tracks
        or record.get("track_count") != len(tracks)
    ):
        return None
    for track in tracks:
        if not isinstance(track, dict):
            return None
        relative = PurePosixPath(str(track.get("private_caption_relative_path", "")))
        if (
            relative.is_absolute()
            or len(relative.parts) != 3
            or relative.parts[:2] != ("raw", lesson_id)
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            return None
        caption_path = output_directory.joinpath(*relative.parts)
        expected_digest = str(track.get("caption_sha256", ""))
        timeline = track.get("timeline")
        if (
            caption_path.is_symlink()
            or not caption_path.is_file()
            or not _SHA256_RE.fullmatch(expected_digest)
            or file_sha256(caption_path) != expected_digest
            or not isinstance(timeline, dict)
            or timeline.get("timeline_audit_completed") is not True
        ):
            return None
    record["released_transcript_audit"] = released_audit
    record["retrieval_state"] = "reused_hash_verified"
    record["last_local_verification_at_utc"] = verified_at_utc
    record["whisper_model_downloaded"] = False
    record["whisper_executed"] = False
    # Never carry transport strings from an older mutable record into a new
    # aggregate.  This execution performed only a local, hash-verified reuse.
    record["retrieval_transport"] = _local_reuse_transport_record()
    return record


def _retrieve_one_lesson(
    lesson: dict[str, Any],
    *,
    repository_root: Path,
    output_directory: Path,
    prefix: Sequence[str],
    js_runtime: str | None,
    cookies_from_browser: str | None,
    browser_family: str | None,
    yt_dlp_direct: bool,
    yt_dlp_impersonate: str | None,
    yt_dlp_youtube_client: str | None,
    timeout_seconds: int,
    runner: Callable[..., Any],
    retrieved_at_utc: str,
) -> dict[str, Any]:
    lesson_id = _safe_lesson_id(lesson.get("lesson_id"))
    source_url = str(lesson["source_url"])
    source_url_sha256 = _sha256_bytes(source_url.encode("utf-8"))
    released_rows = _load_released_scene_rows(repository_root, lesson)
    released_audit = _audit_released_rows(released_rows)
    common: dict[str, Any] = {
        "lesson_id": lesson_id,
        "split": lesson["split"],
        "source_url": source_url,
        "source_url_sha256": source_url_sha256,
        "scene_manifest_sha256": lesson["scene_manifest_sha256"],
        "reference_duration_seconds": lesson["reference_duration_seconds"],
        "released_transcript_audit": released_audit,
        "retrieved_at_utc": retrieved_at_utc,
        "whisper_model_downloaded": False,
        "whisper_executed": False,
        "caption_content_accuracy_established": False,
        "human_caption_authorship_independently_verified": False,
        "independent_human_review_established": False,
        "word_error_rate_established": False,
        "retrieval_state": "retrieved_now",
        "retrieval_transport": _network_transport_record(
            browser_family=browser_family,
            yt_dlp_direct=yt_dlp_direct,
            yt_dlp_impersonate=yt_dlp_impersonate,
            yt_dlp_youtube_client=yt_dlp_youtube_client,
        ),
    }
    reusable = _load_reusable_lesson_record(
        output_directory,
        lesson=lesson,
        released_audit=released_audit,
        verified_at_utc=retrieved_at_utc,
    )
    if reusable is not None:
        return reusable
    try:
        metadata_result = _run_command(
            _metadata_command(
                prefix,
                source_url=source_url,
                js_runtime=js_runtime,
                cookies_from_browser=cookies_from_browser,
                yt_dlp_direct=yt_dlp_direct,
                yt_dlp_impersonate=yt_dlp_impersonate,
                yt_dlp_youtube_client=yt_dlp_youtube_client,
            ),
            runner=runner,
            timeout_seconds=timeout_seconds,
            purpose="caption metadata retrieval",
        )
        metadata = _metadata_from_result(metadata_result)
        tracks = _select_caption_tracks(metadata)
        if not tracks:
            return {
                **common,
                "status": "caption_unavailable_fallback_pending",
                "track_count": 0,
                "tracks": [],
                "fallback": {
                    "status": "pending",
                    "required_next_step": (
                        "manual source review or separately authorized audited ASR"
                    ),
                    "whisper_fallback_allowed_in_this_workflow": False,
                },
            }
        lesson_raw = ensure_private_directory(output_directory / "raw" / lesson_id)
        with tempfile.TemporaryDirectory(
            prefix=f".{lesson_id}-captions-", dir=output_directory
        ) as temporary_directory:
            staging = Path(temporary_directory)
            command = _download_command(
                prefix,
                source_url=source_url,
                lesson_id=lesson_id,
                languages=[track["language"] for track in tracks],
                staging_directory=staging,
                js_runtime=js_runtime,
                cookies_from_browser=cookies_from_browser,
                yt_dlp_direct=yt_dlp_direct,
                yt_dlp_impersonate=yt_dlp_impersonate,
                yt_dlp_youtube_client=yt_dlp_youtube_client,
            )
            _run_command(
                command,
                runner=runner,
                timeout_seconds=timeout_seconds,
                purpose="caption track download",
            )
            audited_tracks: list[dict[str, Any]] = []
            for track in tracks:
                language = track["language"]
                staged = _caption_path_for_language(
                    staging, lesson_id=lesson_id, language=language
                )
                payload = staged.read_bytes()
                vtt_audit = _audit_vtt(
                    payload,
                    reference_duration_seconds=float(
                        lesson["reference_duration_seconds"]
                    ),
                )
                alignment = align_caption_to_released_scenes(
                    released_rows, vtt_audit.pop("segments")
                )
                target = lesson_raw / f"{language}.vtt"
                _atomic_write_bytes(target, payload)
                audited_tracks.append(
                    {
                        **track,
                        "human_authorship_independently_verified": False,
                        "caption_sha256": file_sha256(target),
                        "caption_size_bytes": target.stat().st_size,
                        "private_caption_relative_path": (
                            Path("raw") / lesson_id / target.name
                        ).as_posix(),
                        "timeline": vtt_audit,
                        "released_window_alignment": alignment,
                    }
                )
        return {
            **common,
            "status": "caption_timeline_audited",
            "track_count": len(audited_tracks),
            "tracks": audited_tracks,
            "fallback": {"status": "not_used", "whisper_executed": False},
        }
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        return {
            **common,
            "status": "caption_retrieval_failed_fallback_pending",
            "track_count": 0,
            "tracks": [],
            "private_failure_detail": _private_error(exc),
            "fallback": {
                "status": "pending",
                "required_next_step": (
                    "manual source review or separately authorized audited ASR"
                ),
                "whisper_fallback_allowed_in_this_workflow": False,
            },
        }


def _validate_plan_against_repository(
    repository_root: Path,
    media_plan_path: str | Path,
    *,
    lesson_ids: Sequence[str] | None,
    acquisition_receipt_path: str | Path | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    plan_path = Path(media_plan_path)
    if plan_path.is_symlink() or not plan_path.is_file():
        raise ValueError("TeachObs caption audit requires a private media plan")
    plan = read_json(plan_path)
    if not isinstance(plan, dict) or plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("unsupported TeachObs private media plan")
    unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
    if plan.get("plan_sha256") != _canonical_sha256(unsigned):
        raise ValueError("TeachObs private media plan hash mismatch")
    provenance = plan.get("repository_provenance")
    if (
        not isinstance(provenance, dict)
        or provenance.get("repository_commit") != PINNED_REPOSITORY_COMMIT
    ):
        raise ValueError("TeachObs caption audit is not bound to the pinned commit")
    plan_lessons = plan.get("lessons")
    if not isinstance(plan_lessons, list) or not plan_lessons:
        raise ValueError("TeachObs private media plan contains no lessons")
    by_id: dict[str, dict[str, Any]] = {}
    for item in plan_lessons:
        if not isinstance(item, dict):
            raise ValueError("TeachObs private media plan has a non-object lesson")
        lesson_id = _safe_lesson_id(item.get("lesson_id"))
        if lesson_id in by_id:
            raise ValueError("TeachObs private media plan contains duplicate lessons")
        by_id[lesson_id] = item
    if lesson_ids is None:
        selected_ids = list(by_id)
    else:
        selected_ids = [_safe_lesson_id(value) for value in lesson_ids]
        if not selected_ids or len(set(selected_ids)) != len(selected_ids):
            raise ValueError("TeachObs caption lesson selection must be non-empty and unique")
        if not set(selected_ids).issubset(by_id):
            raise ValueError("TeachObs caption lesson selection is absent from media plan")
    expected = build_teachobs_media_plan(
        repository_root,
        lesson_ids=selected_ids,
        acquisition_receipt_path=acquisition_receipt_path,
    )
    expected_by_id = {item["lesson_id"]: item for item in expected["lessons"]}
    fields = (
        "split",
        "source_url",
        "reference_duration_seconds",
        "scene_manifest_sha256",
        "scene_count",
    )
    selected: list[dict[str, Any]] = []
    for lesson_id in sorted(selected_ids, key=lambda value: int(value[1:])):
        actual_item = by_id[lesson_id]
        expected_item = expected_by_id[lesson_id]
        if any(actual_item.get(field) != expected_item.get(field) for field in fields):
            raise ValueError(
                f"TeachObs caption plan differs from pinned repository for {lesson_id}"
            )
        selected.append(actual_item)
    return plan, selected


def _aggregate_private_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    tracks = [track for record in records for track in record["tracks"]]
    released = [record["released_transcript_audit"] for record in records]
    alignments = [
        track["released_window_alignment"]
        for track in tracks
        if isinstance(track.get("released_window_alignment"), dict)
    ]
    token_means = [
        float(value)
        for item in alignments
        if (value := item.get("normalized_token_overlap_mean_on_paired_nonempty"))
        is not None
    ]
    character_means = [
        float(value)
        for item in alignments
        if (
            value := item.get(
                "normalized_character_bigram_overlap_mean_on_paired_nonempty"
            )
        )
        is not None
    ]
    timeline_coverages = [
        float(track["timeline"]["timeline_span_coverage_fraction"])
        for track in tracks
    ]
    audited_count = sum(record["status"] == "caption_timeline_audited" for record in records)
    expected_files = sum(item["expected_file_count"] for item in released)
    readable_files = sum(item["readable_utf8_file_count"] for item in released)
    nonempty_files = sum(item["nonempty_file_count"] for item in released)
    abnormal_files = sum(
        item["abnormal_repetition_file_count"] for item in released
    )
    character_count = sum(item["character_count"] for item in released)
    nonwhitespace_count = sum(
        item["nonwhitespace_character_count"] for item in released
    )
    return {
        "lesson_count": len(records),
        "caption_timeline_audited_lesson_count": audited_count,
        "caption_unavailable_or_failed_lesson_count": len(records) - audited_count,
        "reused_hash_verified_lesson_count": sum(
            record.get("retrieval_state") == "reused_hash_verified"
            for record in records
        ),
        "retrieved_now_lesson_count": sum(
            record.get("retrieval_state") == "retrieved_now" for record in records
        ),
        "caption_track_count": len(tracks),
        "manual_creator_provided_track_count": sum(
            track["track_type"] == "manual_creator_provided" for track in tracks
        ),
        "youtube_automatic_caption_track_count": sum(
            track["track_type"] == "youtube_automatic_caption" for track in tracks
        ),
        "original_language_role_track_count": sum(
            "original_language" in track["roles"] for track in tracks
        ),
        "english_role_track_count": sum("english" in track["roles"] for track in tracks),
        "timeline_span_coverage_mean_across_tracks": _mean(timeline_coverages),
        "timeline_span_coverage_ge_0_90_track_count": sum(
            value >= 0.90 for value in timeline_coverages
        ),
        "released_transcript_expected_file_count": expected_files,
        "released_transcript_readable_utf8_file_count": readable_files,
        "released_transcript_file_coverage_fraction": round(
            readable_files / max(1, expected_files), 6
        ),
        "released_transcript_nonempty_file_count": nonempty_files,
        "released_transcript_nonempty_file_fraction": round(
            nonempty_files / max(1, expected_files), 6
        ),
        "released_transcript_character_count": character_count,
        "released_transcript_nonwhitespace_character_count": nonwhitespace_count,
        "released_transcript_nonwhitespace_character_fraction": round(
            nonwhitespace_count / max(1, character_count), 6
        ),
        "released_transcript_abnormal_repetition_file_count": abnormal_files,
        "released_transcript_abnormal_repetition_file_fraction": round(
            abnormal_files / max(1, expected_files), 6
        ),
        "track_level_normalized_token_overlap_mean": _mean(token_means),
        "track_level_normalized_character_bigram_overlap_mean": _mean(
            character_means
        ),
        "formal_caption_timeline_audit_completed": bool(
            records and audited_count == len(records)
        ),
    }


def retrieve_and_audit_teachobs_captions(
    repository_root: str | Path,
    media_plan_path: str | Path,
    output_directory: str | Path,
    *,
    acknowledge_source_terms: bool,
    acquisition_receipt_path: str | Path | None = None,
    lesson_ids: Sequence[str] | None = None,
    yt_dlp_command: Sequence[str] | str | None = None,
    js_runtime: str | None = None,
    yt_dlp_direct: bool = False,
    yt_dlp_impersonate: str | None = None,
    yt_dlp_youtube_client: str | None = None,
    cookies_from_browser: str | None = None,
    timeout_seconds: int = 600,
    max_workers: int = 4,
    runner: Callable[..., Any] = subprocess.run,
    retrieved_at_utc: str | None = None,
) -> dict[str, Any]:
    """Retrieve selected source captions and audit them against release text.

    The workflow never downloads or executes Whisper.  A missing or failed
    caption is recorded as pending rather than silently replaced by ASR.
    """

    if not acknowledge_source_terms:
        raise ValueError(
            "TeachObs caption retrieval requires explicit acknowledgement of "
            "the source-video/platform terms"
        )
    if timeout_seconds < 1:
        raise ValueError("timeout_seconds must be positive")
    if max_workers < 1 or max_workers > 16:
        raise ValueError("max_workers must be between 1 and 16")
    cookie_spec, browser_family = validate_teachobs_cookies_from_browser(
        cookies_from_browser
    )
    direct, impersonation_target = validate_teachobs_ytdlp_transport(
        direct=yt_dlp_direct,
        impersonate=yt_dlp_impersonate,
    )
    youtube_client = _validate_ytdlp_youtube_client(yt_dlp_youtube_client)
    repository = Path(repository_root).resolve()
    if repository.is_symlink() or not repository.is_dir():
        raise ValueError("TeachObs repository root is missing or unsafe")
    plan, lessons = _validate_plan_against_repository(
        repository,
        media_plan_path,
        lesson_ids=lesson_ids,
        acquisition_receipt_path=acquisition_receipt_path,
    )
    readme_path = repository / "README.md"
    if readme_path.is_symlink() or not readme_path.is_file():
        raise ValueError("TeachObs repository README is missing")
    output = ensure_private_directory(output_directory).resolve()
    ensure_private_directory(output / "raw")
    ensure_private_directory(output / "lessons")
    prefix = _command_prefix(yt_dlp_command)
    timestamp = retrieved_at_utc or _utc_now()
    records: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=min(max_workers, len(lessons))) as executor:
        futures = {
            executor.submit(
                _retrieve_one_lesson,
                lesson,
                repository_root=repository,
                output_directory=output,
                prefix=prefix,
                js_runtime=js_runtime,
                cookies_from_browser=cookie_spec,
                browser_family=browser_family,
                yt_dlp_direct=direct,
                yt_dlp_impersonate=impersonation_target,
                yt_dlp_youtube_client=youtube_client,
                timeout_seconds=timeout_seconds,
                runner=runner,
                retrieved_at_utc=timestamp,
            ): lesson["lesson_id"]
            for lesson in lessons
        }
        for future in as_completed(futures):
            record = future.result()
            write_json(output / "lessons" / f"{record['lesson_id']}.json", record)
            records.append(record)
    records.sort(key=lambda item: int(item["lesson_id"][1:]))
    aggregate = _aggregate_private_records(records)
    audit: dict[str, Any] = {
        "schema": PRIVATE_AUDIT_SCHEMA,
        "dataset_id": DATASET_ID,
        "generated_at_utc": timestamp,
        "private_artifact": True,
        "public_release_authorized": False,
        "repository_commit": PINNED_REPOSITORY_COMMIT,
        "repository_readme_sha256": file_sha256(readme_path),
        "media_plan_sha256": plan["plan_sha256"],
        "source_reported_provenance": {
            "claim": (
                "The pinned repository README reports that each lesson's own "
                "YouTube captions are the recommended source and that released "
                "scene text was built from those captions."
            ),
            "status": "source_reported_not_independently_verified",
            "paper_or_repository_authorship_independently_verified": False,
            "released_text_derivation_reexecuted_by_this_audit": False,
        },
        "workflow": {
            "transport": "yt-dlp_https_youtube",
            "requested_retrieval_transport": _network_transport_record(
                browser_family=browser_family,
                yt_dlp_direct=direct,
                yt_dlp_impersonate=impersonation_target,
                yt_dlp_youtube_client=youtube_client,
            ),
            "metadata_mode": "skip_download_dump_single_json",
            "caption_flags": ["--skip-download", "--write-subs", "--write-auto-subs"],
            "track_selection": (
                "creator-provided preferred over automatic; original-language "
                "and English roles requested when available"
            ),
            "whisper_model_downloaded": False,
            "whisper_executed": False,
            "missing_caption_fallback": "pending_manual_review_or_authorized_audited_asr",
        },
        "aggregate": aggregate,
        "claims": {
            "formal_caption_timeline_audit_completed": aggregate[
                "formal_caption_timeline_audit_completed"
            ],
            "caption_content_accuracy_established": False,
            "word_error_rate_established": False,
            "human_caption_authorship_independently_verified": False,
            "independent_human_transcript_audit_completed": False,
            "double_annotation_reliability_established": False,
        },
        "metric_boundary": (
            "15-second normalized token/character overlap is a descriptive "
            "alignment diagnostic, not WER, recognition accuracy, or human audit"
        ),
        "records": records,
    }
    audit["audit_canonical_sha256"] = _canonical_sha256(audit)
    audit_path = write_json(output / "caption_audit.json", audit)
    return {
        "output_directory": str(output),
        "private_audit_path": str(audit_path.resolve()),
        "audit": audit,
    }


def build_public_teachobs_caption_receipt(
    private_audit: dict[str, Any],
    *,
    private_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a text-, URL-, path-, and per-lesson-identity-free receipt."""

    if (
        not isinstance(private_audit, dict)
        or private_audit.get("schema") != PRIVATE_AUDIT_SCHEMA
    ):
        raise ValueError("unsupported TeachObs private caption audit")
    claimed_digest = private_audit.get("audit_canonical_sha256")
    unsigned = {
        key: value
        for key, value in private_audit.items()
        if key != "audit_canonical_sha256"
    }
    if claimed_digest != _canonical_sha256(unsigned):
        raise ValueError("TeachObs private caption audit hash mismatch")
    aggregate = private_audit.get("aggregate")
    claims = private_audit.get("claims")
    workflow = private_audit.get("workflow")
    if not all(isinstance(value, dict) for value in (aggregate, claims, workflow)):
        raise ValueError("TeachObs private caption audit lacks aggregate evidence")
    assert isinstance(aggregate, dict)
    assert isinstance(claims, dict)
    assert isinstance(workflow, dict)
    missing_aggregate_fields = set(_PUBLIC_AGGREGATE_FIELDS).difference(aggregate)
    if missing_aggregate_fields:
        raise ValueError("TeachObs private caption audit has an incomplete aggregate")
    # Never copy arbitrary private-audit keys into a public artifact.  This
    # allowlist prevents a profile, account, proxy URL, or local path appended
    # to an otherwise valid private aggregate from crossing the release edge.
    public_aggregate = {
        field: aggregate[field] for field in _PUBLIC_AGGREGATE_FIELDS
    }
    private_digest: str | None = None
    if private_audit_path is not None:
        path = Path(private_audit_path)
        if path.is_symlink() or not path.is_file():
            raise ValueError("private TeachObs caption audit path is missing or unsafe")
        private_digest = file_sha256(path)
    receipt: dict[str, Any] = {
        "artifact_kind": "teachobs_aggregate_caption_timeline_audit_receipt",
        "schema": PUBLIC_RECEIPT_SCHEMA,
        "generated_at_utc": private_audit.get("generated_at_utc"),
        "private_audit_sha256": private_digest,
        "source_binding_sha256": private_audit.get("media_plan_sha256"),
        "aggregate": public_aggregate,
        "evidence_status": {
            "formal_caption_timeline_audit_completed": bool(
                claims.get("formal_caption_timeline_audit_completed")
            ),
            "caption_content_accuracy_established": False,
            "word_error_rate_established": False,
            "human_caption_authorship_independently_verified": False,
            "independent_human_transcript_audit_completed": False,
            "double_annotation_reliability_established": False,
            "released_text_provenance_independently_verified": False,
            "whisper_model_downloaded": False,
            "whisper_executed": False,
        },
        "metric_boundary": (
            "Aggregate timing coverage and normalized 15-second overlap are "
            "descriptive diagnostics; they are not WER or recognition accuracy."
        ),
        "content_exclusion": {
            "caption_text_included": False,
            "released_transcript_text_included": False,
            "source_urls_included": False,
            "lesson_or_video_ids_included": False,
            "filesystem_paths_included": False,
            "per_lesson_records_included": False,
            "per_scene_records_included": False,
            "browser_profile_included": False,
            "account_identifier_included": False,
            "proxy_url_included": False,
            "cookie_material_included": False,
        },
    }
    serialized = json.dumps(receipt, ensure_ascii=False)
    if (
        "http://" in serialized.casefold()
        or "https://" in serialized.casefold()
        or re.search(r'(?<![A-Za-z0-9])S(?:[1-9]|[12][0-9]|30)(?![A-Za-z0-9])', serialized)
    ):
        raise ValueError("public TeachObs caption receipt leaked private identity data")
    return receipt


__all__ = [
    "PRIVATE_AUDIT_SCHEMA",
    "PUBLIC_RECEIPT_SCHEMA",
    "align_caption_to_released_scenes",
    "build_public_teachobs_caption_receipt",
    "normalized_character_overlap",
    "normalized_token_overlap",
    "repetition_diagnostics",
    "retrieve_and_audit_teachobs_captions",
]
