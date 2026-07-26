from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from teaching_skill_miner.audit import audit_dataset
from teaching_skill_miner.io_utils import write_json


def _formal_caption(
    *,
    video_id: str,
    course_id: str = "course-a",
    title: str | None = None,
    source_url: str | None = None,
    transcript_url: str | None = None,
    input_sha256: str = "a" * 64,
) -> dict[str, Any]:
    title = title or video_id
    source_url = source_url or f"https://example.org/videos/{video_id}"
    transcript_url = transcript_url or f"https://example.org/captions/{video_id}.srt"
    text = " ".join(f"word{index}" for index in range(40))
    return {
        "video_id": video_id,
        "course_id": course_id,
        "title": title,
        "source_url": source_url,
        "transcript_url": transcript_url,
        "transcript_kind": "caption_import",
        "timestamps_are_approximate": False,
        "segments": [
            {"start": 0.0, "end": 30.0, "text": text},
            {"start": 30.0, "end": 60.0, "text": text},
            {"start": 60.0, "end": 100.0, "text": text},
        ],
        "provenance": {
            "input_sha256": input_sha256,
            "caption_source_verified": True,
            "transcript_coverage": {
                "source_duration_seconds": 100.0,
                "covered_duration_seconds": 100.0,
                "coverage_fraction": 1.0,
                "completeness_verified": True,
                "verification_method": "matched final caption cue to source duration",
            },
        },
    }


def _manifest_item(transcript: dict[str, Any], path: str) -> dict[str, Any]:
    return {
        "video_id": transcript["video_id"],
        "course_id": transcript["course_id"],
        "title": transcript["title"],
        "source_url": transcript["source_url"],
        "transcript_path": path,
    }


class FormalCaptionAuditTests(unittest.TestCase):
    def test_hash_bound_visual_semantics_are_audited_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = _formal_caption(video_id="lesson-semantic")
            embedding = [0.1, 0.2]
            embedding_sha256 = hashlib.sha256(
                json.dumps(
                    embedding,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            semantic_payload = {
                "schema": "teaching_skill_miner.visual_semantic_results.v1",
                "video_id": transcript["video_id"],
                "media_sha256": "b" * 64,
                "frames": [],
            }
            semantic_payload_sha256 = hashlib.sha256(
                json.dumps(
                    semantic_payload,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            transcript["multimodal"] = {
                "modalities_available": ["transcript", "audio", "visual"],
                "media": {"duration_seconds": 100.0, "sha256": "b" * 64},
                "caption_media_alignment": {
                    "official_caption_timeline_media_binding_verified": True,
                },
                "visual": {
                    "sampling_coverage": {
                        "full_timeline_sampling_passed": True,
                    },
                    "keyframes": [
                        {
                            "path": "frames/10.jpg",
                            "visual_semantics": {
                                "top_label": "presentation_slide",
                                "top_relative_score": 0.8,
                                "score_margin": 0.5,
                                "embedding": embedding,
                                "embedding_sha256": embedding_sha256,
                                "scores_are_calibrated_probabilities": False,
                            },
                        }
                    ],
                    "semantic_features": {
                        "status": "complete_hash_bound_inference",
                        "frame_count": 1,
                        "covered_frame_count": 1,
                        "result_sha256": semantic_payload_sha256,
                    },
                },
                "events": [
                    {
                        "event_id": "mme_0001",
                        "type": "scene_change",
                        "start": 10.0,
                        "end": 10.0,
                        "modalities": ["visual"],
                        "evidence": {"frame_path": "frames/10.jpg"},
                        "supports_strategies": [],
                    }
                ],
            }
            write_json(root / "lesson.json", transcript)
            write_json(root / "semantic.json", semantic_payload)
            semantic_file_sha256 = hashlib.sha256(
                (root / "semantic.json").read_bytes()
            ).hexdigest()
            item = _manifest_item(transcript, "lesson.json")
            item.update(
                {
                    "semantic_result_path": "semantic.json",
                    "semantic_result_sha256": semantic_file_sha256,
                }
            )

            report = audit_dataset(
                {"dataset_id": "semantic", "videos": [item]}, root
            )

            transcript_report = report["transcripts"][0]
            self.assertTrue(transcript_report["visual_semantic_pipeline_ready"])
            self.assertEqual(
                transcript_report["visual_semantic_readiness_failures"], []
            )
            self.assertEqual(
                report["semantic_feature_pipeline_ready_transcript_count"], 1
            )

    def test_verified_caption_alignment_counts_as_multimodal_language(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            transcript = _formal_caption(video_id="lesson-1")
            transcript["multimodal"] = {
                "modalities_available": ["transcript", "audio", "visual"],
                "caption_media_alignment": {
                    "official_caption_timeline_media_binding_verified": True,
                },
                "visual": {
                    "sampling_coverage": {
                        "full_timeline_sampling_passed": True,
                    }
                },
                "events": [
                    {
                        "type": "scene_change",
                        "start": 10.0,
                        "end": 10.0,
                        "modalities": ["visual"],
                        "evidence": {"frame_path": "frames/10.jpg"},
                    }
                ],
            }
            path = "lesson.json"
            write_json(root / path, transcript)

            report = audit_dataset(
                {
                    "dataset_id": "caption-multimodal",
                    "videos": [_manifest_item(transcript, path)],
                },
                root,
            )

            transcript_report = report["transcripts"][0]
            self.assertTrue(transcript_report["source_verified_caption_language"])
            self.assertTrue(transcript_report["verified_language_available"])
            self.assertTrue(transcript_report["multimodal_ready"])
            self.assertEqual(report["multimodal_ready_transcript_count"], 1)

    def test_all_manifest_identity_fields_require_exact_match(self) -> None:
        for field in ("video_id", "course_id", "title", "source_url"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                transcript = _formal_caption(video_id="lesson-1")
                item = _manifest_item(transcript, "lesson.json")
                item[field] = f"different-{field}"
                write_json(root / "lesson.json", transcript)

                report = audit_dataset({"dataset_id": "identity", "videos": [item]}, root)

                check_name = f"{field}_matches_manifest"
                self.assertFalse(
                    report["transcripts"][0]["manifest_identity_checks"][check_name]
                )
                self.assertIn(
                    check_name,
                    report["transcripts"][0]["manifest_identity_failures"],
                )
                self.assertTrue(
                    any(
                        f"manifest/transcript {field} mismatch" in error
                        for error in report["errors"]
                    ),
                    report["errors"],
                )
                self.assertFalse(report["transcripts"][0]["research_grade"])
                self.assertFalse(
                    report["research_grade_integrity_checks"][
                        "all_transcript_metadata_matches_manifest"
                    ]
                )

    def test_duplicate_formal_caption_url_and_hash_disqualify_both(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            duplicate_url = "https://example.org/captions/shared.srt"
            duplicate_hash = "b" * 64
            transcripts = [
                _formal_caption(
                    video_id=f"lesson-{index}",
                    transcript_url=duplicate_url,
                    input_sha256=duplicate_hash,
                )
                for index in (1, 2)
            ]
            videos = []
            for index, transcript in enumerate(transcripts, start=1):
                path = f"lesson-{index}.json"
                write_json(root / path, transcript)
                videos.append(_manifest_item(transcript, path))

            report = audit_dataset({"dataset_id": "duplicates", "videos": videos}, root)

            self.assertEqual(report["research_grade_transcript_count"], 0)
            self.assertFalse(
                report["research_grade_integrity_checks"][
                    "caption_transcript_urls_unique"
                ]
            )
            self.assertFalse(
                report["research_grade_integrity_checks"][
                    "caption_input_sha256_values_unique"
                ]
            )
            for transcript_report in report["transcripts"]:
                self.assertFalse(
                    transcript_report["formal_readiness_checks"][
                        "caption_transcript_url_unique"
                    ]
                )
                self.assertFalse(
                    transcript_report["formal_readiness_checks"][
                        "caption_input_sha256_unique"
                    ]
                )
                self.assertFalse(transcript_report["research_grade"])
            self.assertTrue(
                any(
                    "duplicate research-grade caption transcript_url" in error
                    for error in report["errors"]
                )
            )
            self.assertTrue(
                any(
                    "duplicate research-grade caption input_sha256" in error
                    for error in report["errors"]
                )
            )

    def test_non_candidate_duplicate_does_not_disqualify_formal_caption(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shared_url = "https://example.org/captions/shared.srt"
            shared_hash = "c" * 64
            formal = _formal_caption(
                video_id="formal",
                transcript_url=shared_url,
                input_sha256=shared_hash,
            )
            short = _formal_caption(
                video_id="short",
                transcript_url=shared_url,
                input_sha256=shared_hash,
            )
            short["segments"] = [
                {"start": 0.0, "end": 1.0, "text": "one"},
                {"start": 1.0, "end": 2.0, "text": "two"},
                {"start": 2.0, "end": 3.0, "text": "three"},
            ]
            videos = []
            for transcript in (formal, short):
                path = f"{transcript['video_id']}.json"
                write_json(root / path, transcript)
                videos.append(_manifest_item(transcript, path))

            report = audit_dataset({"dataset_id": "candidate-scope", "videos": videos}, root)

            self.assertTrue(report["transcripts"][0]["research_grade"])
            self.assertFalse(report["transcripts"][1]["research_grade"])
            self.assertTrue(
                report["research_grade_integrity_checks"][
                    "caption_transcript_urls_unique"
                ]
            )
            self.assertTrue(
                report["research_grade_integrity_checks"][
                    "caption_input_sha256_values_unique"
                ]
            )
            self.assertFalse(
                any("duplicate research-grade caption" in error for error in report["errors"])
            )


if __name__ == "__main__":
    unittest.main()
