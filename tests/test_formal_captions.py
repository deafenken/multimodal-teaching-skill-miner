from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from teaching_skill_miner.formal_captions import (
    DownloadedResource,
    import_mit_ocw_formal_captions,
)
from teaching_skill_miner.cli import main
from teaching_skill_miner.delivery import verify_delivery
from teaching_skill_miner.io_utils import project_root, read_json


def _caption(video_id: str) -> bytes:
    words = " ".join(f"{video_id}_word_{index}" for index in range(40))
    return (
        "WEBVTT\n\n"
        f"00:00.000 --> 00:30.000\nfirst {words}\n\n"
        f"00:30.000 --> 01:00.000\nsecond {words}\n\n"
        f"01:00.000 --> 01:40.000\nthird {words}\n"
    ).encode()


def _source_manifest() -> tuple[dict, dict[str, bytes]]:
    videos = []
    captions = {}
    for course_index, course_id in enumerate(("course-a", "course-b")):
        for lesson_index in range(1, 6):
            video_id = f"{course_id}-lesson-{lesson_index}"
            page_url = f"https://ocw.mit.edu/courses/test/{video_id}/"
            caption_url = f"https://ocw.mit.edu/courses/test/{video_id}.vtt"
            media_url = (
                "https://archive.org/download/test/"
                f"{course_index}-{lesson_index}.mp4"
            )
            body = _caption(video_id)
            captions[caption_url] = body
            videos.append(
                {
                    "video_id": video_id,
                    "course_id": course_id,
                    "title": f"Lesson {lesson_index}",
                    "source_url": page_url,
                    "caption_url": caption_url,
                    "media_url": media_url,
                    "caption_sha256": sha256(body).hexdigest(),
                    "reference_media_duration_seconds": 100.0,
                    "reference_last_caption_cue_end_seconds": 100.0,
                    "language": "en",
                }
            )
    return (
        {
            "source_kind": "mit_ocw_official_caption_index",
            "dataset_id": "test-official-captions",
            "version": "1.0",
            "publisher": "MIT OpenCourseWare",
            "publisher_url": "https://ocw.mit.edu/",
            "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
            "courses": [
                {"course_id": "course-a", "name": "Course A"},
                {"course_id": "course-b", "name": "Course B"},
            ],
            "videos": videos,
        },
        captions,
    )


def _fake_downloader(source: dict, captions: dict[str, bytes]):
    by_page = {item["source_url"]: item for item in source["videos"]}

    def download(
        url: str, *, timeout_seconds: int, maximum_bytes: int
    ) -> DownloadedResource:
        del timeout_seconds
        if url in by_page:
            item = by_page[url]
            body = (
                '<html><video data-downloadlink="'
                + item["media_url"]
                + '"><track kind="captions" srclang="en" src="'
                + item["caption_url"]
                + '"></video></html>'
            ).encode()
            content_type = "text/html"
        else:
            body = captions[url]
            content_type = "text/vtt"
        if len(body) > maximum_bytes:
            raise AssertionError("test fixture unexpectedly exceeds download limit")
        return DownloadedResource(
            requested_url=url,
            final_url=url,
            content_type=content_type,
            body=body,
        )

    return download


class FormalCaptionImportTests(unittest.TestCase):
    def test_builds_private_two_by_five_formal_dataset_and_text_free_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, captions = _source_manifest()
            source_path = root / "sources.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")

            with patch(
                "teaching_skill_miner.formal_captions._ffprobe_version",
                return_value="ffprobe test-version",
            ):
                result = import_mit_ocw_formal_captions(
                    source_path,
                    root / "private",
                    downloader=_fake_downloader(source, captions),
                    media_duration_probe=lambda *args, **kwargs: 100.0,
                    retrieved_at_utc="2026-07-22T00:00:00Z",
                )

            audit = result["audit"]
            self.assertTrue(audit["formal_empirical_ready"], audit)
            self.assertEqual(audit["research_grade_transcript_count"], 10)
            self.assertEqual(audit["course_lesson_counts"], {"course-a": 5, "course-b": 5})
            output = root / "private"
            self.assertEqual(len(list((output / "raw").glob("*.vtt"))), 10)
            self.assertEqual(len(list((output / "transcripts").glob("*.json"))), 10)
            receipt = read_json(output / "retrieval_receipt.json")
            self.assertFalse(receipt["caption_text_included"])
            self.assertFalse(receipt["raw_media_downloaded"])
            self.assertTrue(receipt["formal_empirical_ready"])
            receipt_text = json.dumps(receipt, ensure_ascii=False)
            self.assertNotIn("course-a-lesson-1_word_0", receipt_text)
            transcript = read_json(output / "transcripts/course-a-lesson-1.json")
            self.assertTrue(transcript["provenance"]["caption_source_verified"])
            self.assertEqual(
                transcript["provenance"]["transcript_coverage"]["coverage_metric"],
                "(last_caption_cue_end-first_caption_cue_start)/media_duration",
            )
            repository = project_root()
            delivery = verify_delivery(
                repository / "data/dataset_manifest.json",
                repository / "data/evaluation_cases.json",
                formal_manifest_path=output / "dataset_manifest.json",
            )
            self.assertTrue(delivery["engineering_delivery_ready"])
            self.assertTrue(
                delivery["external_evidence_checks"][
                    "formal_full_transcripts_or_asr"
                ]
            )
            self.assertTrue(
                delivery["formal_transcript_dataset"][
                    "separate_from_demo_manifest"
                ]
            )
            self.assertFalse(delivery["research_validation_complete"])
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "audit",
                            "--manifest",
                            str(output / "dataset_manifest.json"),
                            "--require-formal",
                        ]
                    ),
                    0,
                )
                self.assertEqual(
                    main(
                        [
                            "audit",
                            "--manifest",
                            str(repository / "data/dataset_manifest.json"),
                            "--require-formal",
                        ]
                    ),
                    2,
                )

    def test_hash_mismatch_fails_before_publishing_dataset(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, captions = _source_manifest()
            source["videos"][0]["caption_sha256"] = "f" * 64
            source_path = root / "sources.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")

            with (
                patch(
                    "teaching_skill_miner.formal_captions._ffprobe_version",
                    return_value="ffprobe test-version",
                ),
                self.assertRaisesRegex(ValueError, "caption SHA-256 mismatch"),
            ):
                import_mit_ocw_formal_captions(
                    source_path,
                    root / "private",
                    downloader=_fake_downloader(source, captions),
                    media_duration_probe=lambda *args, **kwargs: 100.0,
                )
            self.assertFalse((root / "private/dataset_manifest.json").exists())

    def test_source_manifest_rejects_duplicate_caption_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, _ = _source_manifest()
            source["videos"][1]["caption_sha256"] = source["videos"][0][
                "caption_sha256"
            ]
            source_path = root / "sources.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "duplicate caption_sha256"):
                import_mit_ocw_formal_captions(source_path, root / "private")

    def test_insufficient_endpoint_coverage_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, captions = _source_manifest()
            for item in source["videos"]:
                item["reference_media_duration_seconds"] = 200.0
            source_path = root / "sources.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")

            with (
                patch(
                    "teaching_skill_miner.formal_captions._ffprobe_version",
                    return_value="ffprobe test-version",
                ),
                self.assertRaisesRegex(
                    ValueError, "timeline-span coverage is insufficient"
                ),
            ):
                import_mit_ocw_formal_captions(
                    source_path,
                    root / "private",
                    downloader=_fake_downloader(source, captions),
                    media_duration_probe=lambda *args, **kwargs: 200.0,
                )


if __name__ == "__main__":
    unittest.main()
