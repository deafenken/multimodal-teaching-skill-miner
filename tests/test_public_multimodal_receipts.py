from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from teaching_skill_miner.io_utils import write_json
from teaching_skill_miner.public_multimodal_receipts import (
    ARM_ORDER,
    build_public_multimodal_ablation_receipt,
    build_public_multimodal_validation_receipt,
    write_public_multimodal_receipts,
)
from teaching_skill_miner.release_audit import audit_release_path


PRIVATE_FIXTURE_ROOT = "/" + "Volumes/ORICO/private"
PRIVATE_MARKERS = (
    f"{PRIVATE_FIXTURE_ROOT}/video.mp4",
    "SECRET CAPTION SENTENCE",
    "SECRET OCR BOARD TEXT",
    "SECRET EMBEDDING VALUE",
    "private-lecture-alpha",
)


def _summary(
    *,
    media_sha256: str,
    duration: float,
    keyframes: int,
    visual_events: int,
    fused_events: int,
) -> dict:
    return {
        "media_sha256": media_sha256,
        "duration_seconds": duration,
        "keyframe_count": keyframes,
        "nonempty_ocr_frame_count": keyframes - 1,
        "ocr_frame_with_three_or_more_accepted_words_count": keyframes - 2,
        "ocr_accepted_word_count": keyframes * 5,
        "ocr_accuracy_established": False,
        "visual_event_count": visual_events,
        "fused_event_count": fused_events,
        "full_timeline_sampling_passed": True,
        "caption_timeline_media_binding_verified": True,
        "semantic_features_status": "complete_hash_bound_inference",
    }


def _fixtures() -> tuple[dict, dict, dict, dict]:
    summaries = [
        _summary(
            media_sha256="d" * 64,
            duration=100.0,
            keyframes=10,
            visual_events=5,
            fused_events=7,
        ),
        _summary(
            media_sha256="e" * 64,
            duration=120.0,
            keyframes=12,
            visual_events=6,
            fused_events=8,
        ),
    ]
    manifest = {
        "dataset_id": "public-test-dataset",
        "video_count": 2,
        "videos": [
            {
                "video_id": "private-lecture-alpha",
                "title": "SECRET CAPTION SENTENCE",
                "transcript_path": f"{PRIVATE_FIXTURE_ROOT}/video.mp4",
                "summary": summaries[0],
                "semantic_result_path": f"{PRIVATE_FIXTURE_ROOT}/semantics-a.json",
                "semantic_result_sha256": "a" * 64,
            },
            {
                "video_id": "private-lecture-beta",
                "title": "SECRET OCR BOARD TEXT",
                "analysis": {"embedding": ["SECRET EMBEDDING VALUE"]},
                "summary": summaries[1],
                "semantic_result_path": f"{PRIVATE_FIXTURE_ROOT}/semantics-b.json",
                "semantic_result_sha256": "b" * 64,
            },
        ],
        "aggregate": {
            "duration_seconds": 220.0,
            "keyframe_count": 22,
            "nonempty_ocr_frame_count": 20,
            "ocr_frame_with_three_or_more_accepted_words_count": 18,
            "ocr_accepted_word_count": 110,
            "visual_event_count": 11,
            "fused_event_count": 15,
            "full_timeline_sampling_passed_count": 2,
            "caption_timeline_media_binding_verified_count": 2,
            "semantic_feature_complete_count": 2,
        },
        "claim_boundary": {
            "full_video_bytes_downloaded_and_hashed": True,
            "full_timeline_sampling_required": True,
            "official_caption_timeline_alignment_required": True,
            "audio_content_word_level_verified": False,
            "visual_semantic_features_complete": True,
            "recognition_accuracy_established": False,
            "causal_multimodal_gain_established": False,
        },
    }
    audit = {
        "dataset_id": "public-test-dataset",
        "course_lesson_counts": {"private-course": 2},
        "video_count": 2,
        "valid_transcript_count": 2,
        "research_grade_transcript_count": 2,
        "multimodal_ready_transcript_count": 2,
        "average_segments": 25.5,
        "average_words": 220.0,
        "dataset_structure_passed": True,
        "formal_empirical_ready": True,
        "multimodal_empirical_ready": True,
        "errors": [],
        "warnings": ["SECRET CAPTION SENTENCE"],
        "transcripts": [
            {"ocr_text": "SECRET OCR BOARD TEXT"},
            {"caption_text": "SECRET CAPTION SENTENCE"},
        ],
    }
    semantic_batch = {
        "artifact_kind": "private_visual_semantic_batch_receipt",
        "video_count": 2,
        "frame_count": 22,
        "complete": True,
        "model_path": f"{PRIVATE_FIXTURE_ROOT}/model",
        "results": [
            {
                "video_id": "private-lecture-alpha",
                "frame_count": 10,
                "result_path": f"{PRIVATE_FIXTURE_ROOT}/semantics-a.json",
                "result_sha256": "a" * 64,
                "weight_manifest_sha256": "c" * 64,
            },
            {
                "video_id": "private-lecture-beta",
                "frame_count": 12,
                "result_path": f"{PRIVATE_FIXTURE_ROOT}/semantics-b.json",
                "result_sha256": "b" * 64,
                "weight_manifest_sha256": "c" * 64,
            },
        ],
        "claim_boundary": {
            "visual_semantic_features_computed": True,
            "recognition_accuracy_established": False,
            "human_ground_truth_used": False,
        },
    }

    per_arm_events = {
        "transcript_only": 0,
        "transcript_audio": 4,
        "transcript_visual": 6,
        "full": 10,
    }
    aggregate_metrics = {}
    for arm in ARM_ORDER:
        aggregate_metrics[arm] = {
            "lecture_count": 2,
            "mean_internal_overall_score": 99.0,
            "mean_paired_internal_score_delta": 0.0,
            "median_paired_internal_score_delta": 0.0,
            "mean_paired_evidence_grounding_delta": 0.0,
            "lectures_with_positive_internal_score_delta": 0,
            "lectures_with_zero_internal_score_delta": 2,
            "lectures_with_negative_internal_score_delta": 0,
            "lectures_with_retained_events": 0 if arm == "transcript_only" else 2,
            "total_retained_event_count": per_arm_events[arm],
        }
    ablation = {
        "evaluation_kind": "paired_internal_multimodal_pipeline_ablation",
        "design": {
            "unit_of_pairing": "lecture",
            "baseline_arm": "transcript_only",
            "classroom_observations_included": False,
            "model_fitting_performed": False,
            "private_note": "SECRET CAPTION SENTENCE",
        },
        "arm_order": list(ARM_ORDER),
        "paired_lecture_count": 2,
        "all_arms_use_identical_transcript_segments": True,
        "aggregate_internal_metrics": aggregate_metrics,
        "per_lecture": [
            {
                "video_id": "private-lecture-alpha",
                "title": "SECRET CAPTION SENTENCE",
                "source_transcript_fingerprint_sha256": "1" * 64,
                "paired_on_identical_transcript_segments": True,
            },
            {
                "video_id": "private-lecture-beta",
                "title": "SECRET OCR BOARD TEXT",
                "source_transcript_fingerprint_sha256": "2" * 64,
                "paired_on_identical_transcript_segments": True,
            },
        ],
        "claim_boundary": {
            "independent_event_ground_truth_used": False,
            "human_skill_quality_labels_used": False,
            "learner_outcomes_used": False,
            "recognition_accuracy_established": False,
            "recognition_precision_recall_f1_established": False,
            "multimodal_gain_established": False,
            "causal_multimodal_gain_established": False,
            "independent_skill_quality_gain_established": False,
            "teaching_effectiveness_established": False,
            "deployment_accuracy_established": False,
            "recognition_metrics": {
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
            },
        },
    }
    return manifest, audit, semantic_batch, ablation


class PublicMultimodalReceiptTests(unittest.TestCase):
    def test_validation_receipt_is_aggregate_only_and_content_free(self) -> None:
        manifest, audit, semantic_batch, _ = _fixtures()
        receipt = build_public_multimodal_validation_receipt(
            manifest, audit, semantic_batch
        )
        self.assertEqual(receipt["aggregate_validation"]["video_count"], 2)
        self.assertEqual(receipt["aggregate_validation"]["keyframe_count"], 22)
        self.assertFalse(receipt["evidence_status"]["recognition_accuracy_established"])
        self.assertFalse(receipt["content_exclusion"]["per_lecture_records_included"])
        encoded = json.dumps(receipt, ensure_ascii=False)
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, encoded)
        self.assertNotIn("model_path", encoded)
        self.assertNotIn("transcript_path", encoded)

    def test_ablation_receipt_keeps_only_aggregate_internal_metrics(self) -> None:
        manifest, audit, semantic_batch, ablation = _fixtures()
        receipt = build_public_multimodal_ablation_receipt(
            manifest, audit, semantic_batch, ablation
        )
        self.assertEqual(receipt["design"]["arm_order"], list(ARM_ORDER))
        self.assertEqual(
            receipt["aggregate_internal_metrics"]["full"][
                "total_retained_event_count"
            ],
            10,
        )
        self.assertIsNone(receipt["metric_boundary"]["recognition_metrics"]["f1"])
        self.assertFalse(receipt["metric_boundary"]["internal_overall_score_is_accuracy"])
        encoded = json.dumps(receipt, ensure_ascii=False)
        for marker in PRIVATE_MARKERS:
            self.assertNotIn(marker, encoded)
        self.assertNotIn("per_lecture", receipt)

    def test_inconsistent_or_overclaimed_inputs_fail_closed(self) -> None:
        manifest, audit, semantic_batch, ablation = _fixtures()
        broken_batch = copy.deepcopy(semantic_batch)
        broken_batch["results"][0]["result_sha256"] = "f" * 64
        with self.assertRaisesRegex(ValueError, "semantic result hashes differ"):
            build_public_multimodal_validation_receipt(
                manifest, audit, broken_batch
            )

        overclaimed = copy.deepcopy(ablation)
        overclaimed["claim_boundary"]["recognition_accuracy_established"] = True
        with self.assertRaisesRegex(ValueError, "unsupported ablation claim"):
            build_public_multimodal_ablation_receipt(
                manifest, audit, semantic_batch, overclaimed
            )

    def test_path_workflow_outputs_pass_public_release_audit(self) -> None:
        manifest, audit, semantic_batch, ablation = _fixtures()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            private = root / "private-inputs"
            public = root / "public"
            manifest_path = write_json(private / "manifest.json", manifest)
            audit_path = write_json(private / "audit.json", audit)
            semantic_path = write_json(private / "semantic.json", semantic_batch)
            ablation_path = write_json(private / "ablation.json", ablation)
            validation_output = public / "full_multimodal_validation_receipt.json"
            ablation_output = public / "multimodal_ablation_receipt.json"

            write_public_multimodal_receipts(
                manifest_path=manifest_path,
                audit_path=audit_path,
                semantic_batch_path=semantic_path,
                ablation_report_path=ablation_path,
                validation_output_path=validation_output,
                ablation_output_path=ablation_output,
            )

            self.assertTrue(validation_output.is_file())
            self.assertTrue(ablation_output.is_file())
            release_audit = audit_release_path(public)
            self.assertTrue(release_audit["passed"], release_audit["findings"])


if __name__ == "__main__":
    unittest.main()
