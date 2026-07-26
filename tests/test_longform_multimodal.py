from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import unittest

from teaching_skill_miner.io_utils import write_json
from teaching_skill_miner.longform_multimodal import (
    _ocr_frame_audited,
    _checkpoint_valid,
    _sampling_coverage,
    attach_visual_semantic_results,
    deduplicate_visual_events,
    evaluate_caption_media_alignment,
    plan_chunks,
)
from teaching_skill_miner.models import validate_transcript
from teaching_skill_miner.multimodal import fuse_multimodal_events


class LongformMultimodalTests(unittest.TestCase):
    def test_tesseract_quotes_cannot_absorb_following_tsv_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fake_tesseract = root / "fake-tesseract"
            fake_tesseract.write_text(
                "#!/bin/sh\n"
                "printf 'level\\tpage_num\\tblock_num\\tpar_num\\tline_num\\tword_num\\tleft\\ttop\\twidth\\theight\\tconf\\ttext\\n'\n"
                "printf '5\\t1\\t1\\t1\\t1\\t1\\t0\\t0\\t1\\t1\\t90\\t\"\\n'\n"
                "printf '5\\t1\\t1\\t1\\t1\\t2\\t2\\t0\\t4\\t1\\t80\\tnext\\n'\n",
                encoding="utf-8",
            )
            fake_tesseract.chmod(0o700)
            frame = root / "frame.jpg"
            frame.write_bytes(b"not-used-by-fake-backend")
            previous = os.environ.get("TSM_TESSERACT")
            os.environ["TSM_TESSERACT"] = str(fake_tesseract)
            try:
                text, audit = _ocr_frame_audited(frame, "eng")
            finally:
                if previous is None:
                    os.environ.pop("TSM_TESSERACT", None)
                else:
                    os.environ["TSM_TESSERACT"] = previous

            self.assertEqual(text, '" next')
            self.assertEqual(audit["raw_word_count"], 2)
            self.assertEqual(audit["accepted_word_count"], 2)

    def test_chunk_plan_has_no_nominal_gaps_and_bounded_overlap(self) -> None:
        chunks = plan_chunks(601.0, chunk_seconds=300.0, overlap_seconds=2.0)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["nominal_start"], 0.0)
        self.assertEqual(chunks[-1]["nominal_end"], 601.0)
        for earlier, later in zip(chunks, chunks[1:]):
            self.assertEqual(earlier["nominal_end"], later["nominal_start"])
            self.assertLessEqual(later["decode_start"], later["nominal_start"])
            self.assertGreaterEqual(earlier["decode_end"], earlier["nominal_end"])

    def test_sampling_coverage_requires_beginning_middle_and_end(self) -> None:
        duration = 600.0
        chunks = plan_chunks(duration, chunk_seconds=300.0, overlap_seconds=2.0)
        frames = [
            {"timestamp": float(value), "sampling_source": "uniform"}
            for value in range(0, 601, 15)
        ]
        report = _sampling_coverage(
            frames,
            duration_seconds=duration,
            interval_seconds=15.0,
            chunks=chunks,
        )
        self.assertTrue(report["full_timeline_sampling_passed"])
        truncated = _sampling_coverage(
            frames[:20],
            duration_seconds=duration,
            interval_seconds=15.0,
            chunks=chunks,
        )
        self.assertFalse(truncated["full_timeline_sampling_passed"])
        self.assertFalse(truncated["timeline_endpoints_covered"])

    def test_official_caption_timeline_can_align_without_claiming_asr(self) -> None:
        transcript = {
            "transcript_kind": "caption_import",
            "segments": [
                {"start": 1.0, "end": 9.0, "text": "one"},
                {"start": 9.0, "end": 19.0, "text": "two"},
                {"start": 19.0, "end": 29.0, "text": "three"},
            ],
            "provenance": {
                "caption_source_verified": True,
                "caption_source_verification": {
                    "trusted_source_host": "ocw.mit.edu",
                    "page_declared_media_url": (
                        "http://www.archive.org/download/item/lecture.mp4"
                    ),
                },
                "transcript_coverage": {"source_duration_seconds": 30.0},
            },
        }
        report = evaluate_caption_media_alignment(
            transcript,
            media_duration_seconds=30.0,
            silences=[{"start": 11.0, "end": 12.0, "duration": 1.0}],
            expected_media_url="https://archive.org/download/item/lecture.mp4",
        )
        self.assertTrue(
            report["official_caption_timeline_media_binding_verified"]
        )
        self.assertGreater(report["caption_audio_activity_overlap_fraction"], 0.9)
        self.assertFalse(report["audio_content_verified"])

    def test_continuous_code_tracks_merge_but_distant_tracks_do_not(self) -> None:
        def event(start: float) -> dict:
            return {
                "type": "code_or_formula_visible",
                "start": start,
                "end": start,
                "modalities": ["visual", "ocr"],
                "evidence": {"frame_path": f"{start}.jpg", "ocr_text": "x = 1"},
                "confidence": 0.7,
            }

        result = deduplicate_visual_events(
            [event(10.0), event(20.0), event(80.0)],
            continuous_gap_seconds=25.0,
        )
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["start"], 10.0)
        self.assertEqual(result[0]["end"], 20.0)
        self.assertEqual(result[0]["evidence"]["track_detection_count"], 2)

    def test_verified_checkpoint_detects_frame_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frame = root / "frame.jpg"
            frame.write_bytes(b"frame-one")
            digest = hashlib.sha256(frame.read_bytes()).hexdigest()
            checkpoint = root / "checkpoint.json"
            write_json(
                checkpoint,
                {
                    "status": "completed",
                    "chunk_fingerprint_sha256": "f" * 64,
                    "frames": [{"path": "frame.jpg", "sha256": digest}],
                },
            )
            self.assertIsNotNone(
                _checkpoint_valid(
                    checkpoint,
                    expected_fingerprint="f" * 64,
                    output_root=root,
                )
            )
            frame.write_bytes(b"mutated")
            self.assertIsNone(
                _checkpoint_valid(
                    checkpoint,
                    expected_fingerprint="f" * 64,
                    output_root=root,
                )
            )

    def test_semantics_are_bound_by_media_and_frame_hash(self) -> None:
        frame_digest = "1" * 64
        media_digest = "2" * 64
        transcript = {
            "video_id": "lecture-1",
            "course_id": "course-1",
            "title": "Lecture 1",
            "source_url": "https://example.org/lecture-1",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "one"},
                {"start": 1.0, "end": 2.0, "text": "two"},
                {"start": 2.0, "end": 3.0, "text": "three"},
            ],
            "multimodal": {
                "modalities_available": ["transcript", "visual"],
                "media": {"duration_seconds": 3.0, "sha256": media_digest},
                "visual": {
                    "keyframes": [
                        {
                            "timestamp": 1.0,
                            "path": "frames/a.jpg",
                            "sha256": frame_digest,
                            "ocr_text": "",
                        }
                    ],
                    "events": [],
                },
                "events": [],
            },
        }
        semantic = {
            "schema": "teaching_skill_miner.visual_semantic_results.v1",
            "video_id": "lecture-1",
            "media_sha256": media_digest,
            "backend": "test-clip",
            "model_provenance": {"revision": "fixed"},
            "ontology": {"labels": ["presentation_slide"]},
            "frames": [
                {
                    "path": "frames/a.jpg",
                    "sha256": frame_digest,
                    "top_label": "presentation_slide",
                    "top_relative_score": 0.8,
                    "score_margin": 0.5,
                    "relative_prompt_scores": {"presentation_slide": 0.8},
                    "embedding": [0.1, 0.2],
                    "embedding_sha256": "3" * 64,
                }
            ],
        }
        result = attach_visual_semantic_results(transcript, semantic)
        visual = result["multimodal"]["visual"]
        self.assertEqual(
            visual["keyframes"][0]["visual_semantics"]["top_label"],
            "presentation_slide",
        )
        self.assertEqual(
            visual["semantic_features"]["status"],
            "complete_hash_bound_inference",
        )
        bad = dict(semantic)
        bad["media_sha256"] = "4" * 64
        with self.assertRaisesRegex(ValueError, "media SHA-256"):
            attach_visual_semantic_results(transcript, bad)

    def test_duplicate_question_text_is_disambiguated_by_timestamps(self) -> None:
        transcript = {
            "video_id": "lecture-1",
            "course_id": "course-1",
            "title": "Lecture 1",
            "source_url": "https://example.org/lecture-1",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Right?"},
                {"start": 1.0, "end": 10.0, "text": "Explanation."},
                {"start": 10.0, "end": 11.0, "text": "Right?"},
            ],
        }
        silences = [{"start": 11.0, "end": 11.817, "duration": 0.817}]
        events = fuse_multimodal_events(
            transcript,
            silences,
            [],
            language_modality="transcript",
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["evidence"]["question_segment"]["start"], 10.0)
        transcript["multimodal"] = {
            "modalities_available": ["transcript", "audio"],
            "media": {"duration_seconds": 12.0},
            "audio": {"silences": silences},
            "events": events,
        }
        self.assertTrue(validate_transcript(transcript).valid)


if __name__ == "__main__":
    unittest.main()
