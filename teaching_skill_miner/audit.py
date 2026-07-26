from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from statistics import mean
from typing import Any

from .io_utils import read_json
from .models import validate_transcript


RESEARCH_TRANSCRIPT_KINDS = {"caption_import", "automatic_speech_recognition", "human_verified_transcript"}
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _formal_transcript_checks(
    transcript: dict[str, Any],
    *,
    segment_count: int,
    token_count: int,
    duration_seconds: float,
) -> dict[str, bool]:
    """Require completeness and kind-specific provenance, not just a kind label."""

    kind = transcript.get("transcript_kind")
    provenance = transcript.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    coverage = provenance.get("transcript_coverage")
    coverage = coverage if isinstance(coverage, dict) else {}
    try:
        source_duration = float(coverage.get("source_duration_seconds"))
        covered_duration = float(coverage.get("covered_duration_seconds"))
        coverage_fraction = float(coverage.get("coverage_fraction"))
    except (TypeError, ValueError, OverflowError):
        source_duration = covered_duration = coverage_fraction = -1.0
    common = {
        "supported_transcript_kind": kind in RESEARCH_TRANSCRIPT_KINDS,
        "timestamps_are_exact": transcript.get("timestamps_are_approximate") is False,
        "nontrivial_segment_count": segment_count >= 3,
        "nontrivial_duration": duration_seconds >= 60.0,
        "nontrivial_text_volume": token_count >= 100,
        "coverage_attested": coverage.get("completeness_verified") is True,
        "coverage_has_method": isinstance(coverage.get("verification_method"), str)
        and bool(coverage.get("verification_method", "").strip()),
        "coverage_fraction_sufficient": 0.95 <= coverage_fraction <= 1.0,
        "source_duration_sufficient": source_duration >= 60.0,
        "covered_duration_consistent": (
            covered_duration >= 0.95 * source_duration
            and duration_seconds >= 0.90 * source_duration
            and duration_seconds <= source_duration + 5.0
        ),
        "input_content_hash_present": isinstance(provenance.get("input_sha256"), str)
        and bool(SHA256_RE.fullmatch(provenance.get("input_sha256", "").lower())),
    }
    if kind == "caption_import":
        transcript_url = transcript.get("transcript_url")
        common.update(
            {
                "official_caption_source_verified": provenance.get(
                    "caption_source_verified"
                )
                is True,
                "caption_url_is_http": isinstance(transcript_url, str)
                and transcript_url.startswith(("https://", "http://")),
            }
        )
    elif kind == "automatic_speech_recognition":
        asr = provenance.get("asr")
        asr = asr if isinstance(asr, dict) else {}
        quality = provenance.get("asr_quality_audit")
        quality = quality if isinstance(quality, dict) else {}
        try:
            reference_words = int(quality.get("reference_word_count"))
            sampled_fraction = float(quality.get("sampled_duration_fraction"))
            word_error_rate = float(quality.get("word_error_rate"))
            reviewer_count = int(quality.get("independent_reviewer_count"))
        except (TypeError, ValueError, OverflowError):
            reference_words = reviewer_count = -1
            sampled_fraction = word_error_rate = -1.0
        common.update(
            {
                "asr_engine_supported": asr.get("engine") == "openai_whisper_cli",
                "asr_version_recorded": isinstance(asr.get("version"), str)
                and bool(asr.get("version", "").strip()),
                "asr_model_recorded": isinstance(asr.get("model"), str)
                and bool(asr.get("model", "").strip()),
                "asr_model_fingerprint_recorded": isinstance(
                    asr.get("model_sha256"), str
                )
                and bool(SHA256_RE.fullmatch(asr.get("model_sha256", "").lower())),
                "asr_decoding_config_fingerprint_recorded": isinstance(
                    asr.get("decoding_config_sha256"), str
                )
                and bool(
                    SHA256_RE.fullmatch(
                        asr.get("decoding_config_sha256", "").lower()
                    )
                ),
                "asr_independent_quality_audit": quality.get(
                    "independently_reviewed"
                )
                is True,
                "asr_reference_words_sufficient": reference_words >= 200,
                "asr_sample_fraction_sufficient": sampled_fraction >= 0.20,
                "asr_word_error_rate_acceptable": 0.0 <= word_error_rate <= 0.20,
                "asr_reviewer_present": reviewer_count >= 1,
            }
        )
    elif kind == "human_verified_transcript":
        verification = provenance.get("human_verification")
        verification = verification if isinstance(verification, dict) else {}
        try:
            reviewer_count = int(verification.get("independent_reviewer_count"))
        except (TypeError, ValueError, OverflowError):
            reviewer_count = -1
        common.update(
            {
                "human_full_review_completed": verification.get(
                    "full_transcript_review_completed"
                )
                is True,
                "human_reviewer_present": reviewer_count >= 1,
                "human_verification_protocol_recorded": isinstance(
                    verification.get("protocol"), str
                )
                and bool(verification.get("protocol", "").strip()),
            }
        )
    else:
        common["kind_specific_provenance_complete"] = False
    return common


def audit_dataset(manifest: dict[str, Any], root: Path) -> dict[str, Any]:
    videos = manifest.get("videos", [])
    ids = [str(item.get("video_id", "")) for item in videos]
    urls = [str(item.get("source_url", "")) for item in videos]
    course_counts = Counter(str(item.get("course_id", "")) for item in videos)
    errors: list[str] = []
    warnings: list[str] = []
    transcript_stats: list[dict[str, Any]] = []
    declared_transcript_source_count = 0
    multimodal_ready_count = 0
    semantic_pipeline_ready_count = 0
    all_identity_checks_passed = True
    caption_candidates: list[tuple[int, str, str]] = []
    for item in videos:
        relative = item.get("transcript_path")
        path = root / relative if relative else None
        if path is None or not path.exists():
            errors.append(f"missing transcript: {relative}")
            continue
        transcript = read_json(path)
        declared_transcript_source_count += int(bool(transcript.get("transcript_url")))
        result = validate_transcript(transcript)
        if not result.valid:
            errors.extend(f"{item.get('video_id')}: {message}" for message in result.errors)
        identity_checks = {
            f"{field}_matches_manifest": transcript.get(field) == item.get(field)
            for field in ("video_id", "course_id", "title", "source_url")
        }
        for field in ("video_id", "course_id", "title", "source_url"):
            check_name = f"{field}_matches_manifest"
            if not identity_checks[check_name]:
                errors.append(
                    "manifest/transcript "
                    f"{field} mismatch for {item.get('video_id')!r}: "
                    f"manifest={item.get(field)!r}, transcript={transcript.get(field)!r}"
                )
        all_identity_checks_passed = all_identity_checks_passed and all(
            identity_checks.values()
        )
        segments = transcript.get("segments", [])
        duration = max((float(segment["end"]) for segment in segments), default=0.0)
        words = sum(
            len(re.findall(r"[A-Za-z0-9_]+|[\u3400-\u9fff]", str(segment.get("text", ""))))
            for segment in segments
        )
        kind = str(transcript.get("transcript_kind", "unknown"))
        approximate = bool(transcript.get("timestamps_are_approximate"))
        formal_checks = _formal_transcript_checks(
            transcript,
            segment_count=len(segments),
            token_count=words,
            duration_seconds=duration,
        )
        base_research_grade_candidate = all(formal_checks.values()) and all(
            identity_checks.values()
        )
        multimodal = transcript.get("multimodal", {})
        modalities = set(multimodal.get("modalities_available", []))
        caption_alignment = multimodal.get("caption_media_alignment", {})
        source_verified_caption_language = bool(
            "transcript" in modalities
            and kind == "caption_import"
            and all(formal_checks.values())
            and caption_alignment.get(
                "official_caption_timeline_media_binding_verified"
            )
        )
        verified_language_available = bool(
            "speech" in modalities or source_verified_caption_language
        )
        visual = multimodal.get("visual", {})
        visual = visual if isinstance(visual, dict) else {}
        sampling_coverage = visual.get("sampling_coverage", {})
        full_timeline_sampling_passed = bool(
            sampling_coverage.get("full_timeline_sampling_passed")
        )
        multimodal_ready = bool(
            verified_language_available
            and {"audio", "visual"} <= modalities
            and multimodal.get("events")
            and full_timeline_sampling_passed
        )
        multimodal_ready_count += int(multimodal_ready)
        keyframes = [
            frame
            for frame in visual.get("keyframes", [])
            if isinstance(frame, dict)
        ]
        semantic_metadata = visual.get("semantic_features", {})
        semantic_metadata = (
            semantic_metadata if isinstance(semantic_metadata, dict) else {}
        )
        semantic_rows = [
            frame.get("visual_semantics")
            for frame in keyframes
            if isinstance(frame.get("visual_semantics"), dict)
        ]
        embedding_hashes_match = bool(semantic_rows) and all(
            isinstance(row.get("embedding"), list)
            and SHA256_RE.fullmatch(str(row.get("embedding_sha256", "")))
            and _canonical_sha256(row["embedding"])
            == str(row.get("embedding_sha256"))
            for row in semantic_rows
        )

        semantic_relative = str(item.get("semantic_result_path", ""))
        semantic_candidate = (root / semantic_relative).resolve()
        try:
            semantic_candidate.relative_to(root.resolve())
            semantic_path_confined = bool(semantic_relative)
        except ValueError:
            semantic_path_confined = False
        semantic_file_exists = bool(
            semantic_path_confined and semantic_candidate.is_file()
        )
        expected_semantic_file_hash = str(
            item.get("semantic_result_sha256", "")
        )
        semantic_file_hash_matches = bool(
            semantic_file_exists
            and SHA256_RE.fullmatch(expected_semantic_file_hash)
            and _file_sha256(semantic_candidate) == expected_semantic_file_hash
        )
        semantic_payload: dict[str, Any] = {}
        if semantic_file_hash_matches:
            try:
                loaded_semantic_payload = read_json(semantic_candidate)
            except (OSError, ValueError, TypeError):
                loaded_semantic_payload = None
            if isinstance(loaded_semantic_payload, dict):
                semantic_payload = loaded_semantic_payload
        semantic_checks = {
            "status_complete": semantic_metadata.get("status")
            == "complete_hash_bound_inference",
            "keyframes_present": bool(keyframes),
            "all_keyframes_have_semantics": len(semantic_rows) == len(keyframes)
            and bool(keyframes),
            "declared_frame_counts_match": semantic_metadata.get("frame_count")
            == len(keyframes)
            and semantic_metadata.get("covered_frame_count") == len(keyframes),
            "embedding_hashes_match": embedding_hashes_match,
            "scores_explicitly_uncalibrated": all(
                row.get("scores_are_calibrated_probabilities") is False
                for row in semantic_rows
            )
            and bool(semantic_rows),
            "semantic_result_path_is_relative_and_confined": (
                semantic_path_confined and not Path(semantic_relative).is_absolute()
            ),
            "semantic_result_file_exists": semantic_file_exists,
            "semantic_result_file_hash_matches": semantic_file_hash_matches,
            "semantic_result_payload_matches_video": semantic_payload.get(
                "video_id"
            )
            == transcript.get("video_id"),
            "semantic_result_payload_matches_media": semantic_payload.get(
                "media_sha256"
            )
            == multimodal.get("media", {}).get("sha256"),
            "attached_result_hash_matches_payload": bool(semantic_payload)
            and semantic_metadata.get("result_sha256")
            == _canonical_sha256(semantic_payload),
        }
        semantic_pipeline_ready = bool(
            multimodal_ready and all(semantic_checks.values())
        )
        semantic_pipeline_ready_count += int(semantic_pipeline_ready)
        transcript_stats.append(
            {
                "video_id": item.get("video_id"),
                "kind": kind,
                "segment_count": len(segments),
                "word_count": words,
                "duration_seconds": duration,
                "timestamps_approximate": approximate,
                "research_grade": False,
                "manifest_identity_checks": identity_checks,
                "manifest_identity_failures": sorted(
                    name for name, passed in identity_checks.items() if not passed
                ),
                "formal_readiness_checks": formal_checks,
                "formal_readiness_failures": sorted(
                    name for name, passed in formal_checks.items() if not passed
                ),
                "modalities_available": sorted(modalities),
                "verified_language_available": verified_language_available,
                "source_verified_caption_language": (
                    source_verified_caption_language
                ),
                "audio_content_verified": bool(
                    multimodal.get("language", {}).get(
                        "audio_content_verified", False
                    )
                ),
                "full_timeline_sampling_passed": full_timeline_sampling_passed,
                "visual_semantic_features_status": (
                    semantic_metadata.get("status")
                ),
                "visual_semantic_readiness_checks": semantic_checks,
                "visual_semantic_readiness_failures": sorted(
                    name for name, passed in semantic_checks.items() if not passed
                ),
                "visual_semantic_pipeline_ready": semantic_pipeline_ready,
                "multimodal_ready": multimodal_ready,
            }
        )
        if kind == "caption_import" and base_research_grade_candidate:
            caption_candidates.append(
                (
                    len(transcript_stats) - 1,
                    str(transcript.get("transcript_url", "")),
                    str(
                        transcript.get("provenance", {}).get("input_sha256", "")
                    ).lower(),
                )
            )

    caption_urls: dict[str, list[int]] = {}
    caption_hashes: dict[str, list[int]] = {}
    for index, transcript_url, input_sha256 in caption_candidates:
        caption_urls.setdefault(transcript_url, []).append(index)
        caption_hashes.setdefault(input_sha256, []).append(index)

    caption_transcript_urls_unique = all(
        len(indexes) == 1 for indexes in caption_urls.values()
    )
    caption_input_sha256_values_unique = all(
        len(indexes) == 1 for indexes in caption_hashes.values()
    )
    for value, indexes in caption_urls.items():
        if len(indexes) > 1:
            duplicate_ids = [transcript_stats[index]["video_id"] for index in indexes]
            errors.append(
                "duplicate research-grade caption transcript_url "
                f"{value!r}: video_ids={duplicate_ids!r}"
            )
    for value, indexes in caption_hashes.items():
        if len(indexes) > 1:
            duplicate_ids = [transcript_stats[index]["video_id"] for index in indexes]
            errors.append(
                "duplicate research-grade caption input_sha256 "
                f"{value!r}: video_ids={duplicate_ids!r}"
            )
    for index, transcript_url, input_sha256 in caption_candidates:
        formal_checks = transcript_stats[index]["formal_readiness_checks"]
        formal_checks["caption_transcript_url_unique"] = (
            len(caption_urls[transcript_url]) == 1
        )
        formal_checks["caption_input_sha256_unique"] = (
            len(caption_hashes[input_sha256]) == 1
        )

    research_grade_count = 0
    for stat in transcript_stats:
        research_grade = all(stat["formal_readiness_checks"].values()) and all(
            stat["manifest_identity_checks"].values()
        )
        stat["research_grade"] = research_grade
        stat["formal_readiness_failures"] = sorted(
            name
            for name, passed in stat["formal_readiness_checks"].items()
            if not passed
        )
        research_grade_count += int(research_grade)
    if len(set(ids)) != len(ids):
        errors.append("duplicate video_id values found")
    if len(set(urls)) != len(urls):
        errors.append("duplicate source_url values found")
    if research_grade_count < len(videos):
        warnings.append(
            "存在释义节选或近似时间戳；适合工程演示，不足以单独支持正式教学效果结论。"
        )
    structure_checks = {
        "at_least_two_courses": len(course_counts) >= 2,
        "at_least_five_lessons_per_course": bool(course_counts) and min(course_counts.values()) >= 5,
        "all_transcripts_present_and_valid": not errors and len(transcript_stats) == len(videos),
        "unique_video_ids": len(set(ids)) == len(ids),
        "unique_source_urls": len(set(urls)) == len(urls),
        "all_source_urls_are_http": all(url.startswith(("https://", "http://")) for url in urls),
        "all_transcripts_declare_source": declared_transcript_source_count == len(videos),
        "all_transcript_metadata_matches_manifest": all_identity_checks_passed,
    }
    research_grade_integrity_checks = {
        "all_transcript_metadata_matches_manifest": all_identity_checks_passed,
        "caption_transcript_urls_unique": caption_transcript_urls_unique,
        "caption_input_sha256_values_unique": caption_input_sha256_values_unique,
    }
    return {
        "dataset_id": manifest.get("dataset_id"),
        "course_lesson_counts": dict(sorted(course_counts.items())),
        "video_count": len(videos),
        "valid_transcript_count": len(transcript_stats),
        "research_grade_transcript_count": research_grade_count,
        "multimodal_ready_transcript_count": multimodal_ready_count,
        "semantic_feature_pipeline_ready_transcript_count": (
            semantic_pipeline_ready_count
        ),
        "average_segments": round(mean(item["segment_count"] for item in transcript_stats), 1) if transcript_stats else 0.0,
        "average_words": round(mean(item["word_count"] for item in transcript_stats), 1) if transcript_stats else 0.0,
        "structure_checks": structure_checks,
        "research_grade_integrity_checks": research_grade_integrity_checks,
        "dataset_structure_passed": all(structure_checks.values()),
        "formal_empirical_ready": all(structure_checks.values()) and research_grade_count == len(videos),
        "multimodal_empirical_ready": all(structure_checks.values()) and multimodal_ready_count == len(videos),
        "semantic_feature_pipeline_ready": all(structure_checks.values())
        and semantic_pipeline_ready_count == len(videos),
        "errors": errors,
        "warnings": warnings,
        "transcripts": transcript_stats,
    }
