from __future__ import annotations

import copy
import unittest

from teaching_skill_miner.multimodal_ablation import (
    ARM_ORDER,
    FULL,
    TRANSCRIPT_AUDIO,
    TRANSCRIPT_ONLY,
    TRANSCRIPT_VISUAL,
    build_ablation_transcript,
    evaluate_multimodal_ablation,
)


def _fixture() -> dict:
    frames = [
        {
            "path": "frames/a.jpg",
            "timestamp": 2.0,
            "ocr_text": "x = 1",
            "visual_semantics": {
                "top_label": "mathematical_formula",
                "top_relative_score": 0.8,
                "score_margin": 0.6,
            },
        },
        {"path": "frames/b.jpg", "timestamp": 4.0, "ocr_text": "x = 1 y = 2"},
    ]
    return {
        "video_id": "lecture-1",
        "course_id": "course-1",
        "title": "Lecture 1: Example",
        "source_url": "https://example.org/lecture-1",
        "transcript_url": "https://example.org/lecture-1.vtt",
        "transcript_kind": "caption_import",
        "language": "en",
        "timestamps_are_approximate": False,
        "segments": [
            {"start": 0.0, "end": 2.0, "text": "First consider this example."},
            {"start": 2.0, "end": 4.0, "text": "Why does the formula work?"},
            {"start": 4.0, "end": 6.0, "text": "Then check the next step."},
        ],
        "multimodal": {
            "modalities_available": [
                "transcript",
                "audio",
                "visual",
                "ocr",
                "classroom_observation",
            ],
            "media": {"duration_seconds": 6.0},
            "audio": {"silences": [{"start": 4.0, "end": 4.8, "duration": 0.8}]},
            "visual": {
                "keyframes": frames,
                "semantic_features": {"status": "complete_hash_bound_inference"},
            },
            "events": [
                {
                    "event_id": "mme_0001",
                    "type": "question_and_wait",
                    "start": 2.0,
                    "end": 4.8,
                    "modalities": ["transcript", "audio"],
                    "evidence": {
                        "speech_quote": "Why does the formula work?",
                        "question_segment": {"start": 2.0, "end": 4.0},
                        "silence": {"start": 4.0, "end": 4.8, "duration": 0.8},
                        "wait_seconds": 0.8,
                    },
                    "supports_strategies": ["question_and_wait"],
                    "confidence": 0.8,
                },
                {
                    "event_id": "mme_0002",
                    "type": "code_or_formula_visible",
                    "start": 2.0,
                    "end": 2.0,
                    "modalities": ["visual", "ocr"],
                    "evidence": {"frame_path": "frames/a.jpg", "ocr_text": "x = 1"},
                    "supports_strategies": [],
                    "confidence": 0.7,
                },
                {
                    "event_id": "mme_0003",
                    "type": "teacher_adjustment",
                    "start": 4.0,
                    "end": 5.0,
                    "modalities": ["classroom_observation"],
                    "evidence": {
                        "anonymized_note": "teacher repeats example",
                        "evidence_origin": "provided_anonymized_annotation",
                    },
                    "supports_strategies": ["adaptive_teaching"],
                    "confidence": 0.9,
                },
            ],
        },
    }


class MultimodalAblationTests(unittest.TestCase):
    def test_arm_views_are_disjoint_and_preserve_identical_segments(self) -> None:
        source = _fixture()
        expected = {
            TRANSCRIPT_ONLY: [],
            TRANSCRIPT_AUDIO: ["mme_0001"],
            TRANSCRIPT_VISUAL: ["mme_0002"],
            FULL: ["mme_0001", "mme_0002"],
        }
        for arm in ARM_ORDER:
            view = build_ablation_transcript(source, arm)
            self.assertEqual(view["segments"], source["segments"])
            event_ids = [
                item["event_id"]
                for item in view.get("multimodal", {}).get("events", [])
            ]
            self.assertEqual(event_ids, expected[arm])
            self.assertNotIn("mme_0003", event_ids)

    def test_report_is_paired_and_refuses_accuracy_claims(self) -> None:
        source = _fixture()
        report = evaluate_multimodal_ablation([source])
        self.assertEqual(report["paired_lecture_count"], 1)
        self.assertTrue(report["all_arms_use_identical_transcript_segments"])
        self.assertFalse(report["claim_boundary"]["recognition_accuracy_established"])
        self.assertFalse(report["claim_boundary"]["causal_multimodal_gain_established"])
        self.assertFalse(report["claim_boundary"]["teaching_effectiveness_established"])
        row = report["per_lecture"][0]
        self.assertEqual(row["arms"][TRANSCRIPT_AUDIO]["analysis_metrics"]["event_count"], 1)
        self.assertEqual(row["arms"][TRANSCRIPT_VISUAL]["analysis_metrics"]["keyframe_count"], 2)
        self.assertEqual(
            row["arms"][TRANSCRIPT_VISUAL]["analysis_metrics"][
                "visual_semantic_keyframe_count"
            ],
            1,
        )
        self.assertEqual(
            row["arms"][TRANSCRIPT_VISUAL]["analysis_metrics"][
                "visual_semantic_top_label_frequency"
            ],
            {"mathematical_formula": 1},
        )
        self.assertEqual(row["arms"][FULL]["analysis_metrics"]["event_count"], 2)
        self.assertIn("_payloads", report)

    def test_duplicate_lecture_ids_fail_closed(self) -> None:
        source = _fixture()
        with self.assertRaisesRegex(ValueError, "duplicate video_id"):
            evaluate_multimodal_ablation([source, copy.deepcopy(source)])

    def test_unknown_arm_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown ablation arm"):
            build_ablation_transcript(_fixture(), "audio_only")


if __name__ == "__main__":
    unittest.main()
