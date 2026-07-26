from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
import stat
import subprocess
import tempfile
import unittest

from teaching_skill_miner.full_video_dataset import (
    build_public_full_video_receipt,
    build_download_plan,
    curl_download_to_partial,
    download_full_video_dataset,
    normalize_archive_media_url,
    parse_curl_write_out,
    validate_media_probe,
)


def _write_json(path: Path, value: object) -> bytes:
    body = json.dumps(value, ensure_ascii=False, indent=2).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(body)
    return body


def _dataset_fixture(root: Path) -> tuple[Path, Path]:
    source_videos = []
    formal_videos = []
    transcript_values: dict[str, dict] = {}
    for index in range(10):
        video_id = f"lecture_{index + 1:02d}"
        course_id = "course_a" if index < 5 else "course_b"
        source_url = f"https://ocw.mit.edu/courses/test/{video_id}/"
        media_url = (
            f"http://www.archive.org/download/test-course/{video_id}.mp4"
            if index < 5
            else f"https://archive.org/download/test-course/{video_id}.mp4"
        )
        caption_digest = sha256(f"caption-{video_id}".encode()).hexdigest()
        title = f"Lecture {index + 1}"
        source_videos.append(
            {
                "video_id": video_id,
                "course_id": course_id,
                "title": title,
                "source_url": source_url,
                "caption_url": f"https://ocw.mit.edu/courses/test/{video_id}.vtt",
                "media_url": media_url,
                "caption_sha256": caption_digest,
                "reference_media_duration_seconds": 100.0 + index,
                "reference_last_caption_cue_end_seconds": 99.0 + index,
                "language": "en",
            }
        )
        formal_videos.append(
            {
                "video_id": video_id,
                "course_id": course_id,
                "title": title,
                "source_url": source_url,
                "transcript_path": f"transcripts/{video_id}.json",
            }
        )
        transcript_values[video_id] = {
            "video_id": video_id,
            "course_id": course_id,
            "title": title,
            "source_url": source_url,
            "provenance": {
                "caption_source_verified": True,
                "input_sha256": caption_digest,
                "caption_source_verification": {
                    "media_probe_url": normalize_archive_media_url(media_url),
                },
            },
            "segments": [{"start": 0.0, "end": 99.0, "text": "private"}],
        }
    source = {
        "source_kind": "mit_ocw_official_caption_index",
        "dataset_id": "test-formal-video-dataset",
        "version": "1.0",
        "publisher": "MIT OpenCourseWare",
        "publisher_url": "https://ocw.mit.edu/",
        "license_url": "https://creativecommons.org/licenses/by-nc-sa/4.0/",
        "videos": source_videos,
    }
    source_path = root / "formal_caption_sources.json"
    source_body = _write_json(source_path, source)
    formal = {
        "dataset_id": source["dataset_id"],
        "publisher": source["publisher"],
        "license_url": source["license_url"],
        "source_manifest_sha256": sha256(source_body).hexdigest(),
        "videos": formal_videos,
    }
    formal_path = root / "formal" / "dataset_manifest.json"
    _write_json(formal_path, formal)
    for video_id, transcript in transcript_values.items():
        _write_json(formal_path.parent / "transcripts" / f"{video_id}.json", transcript)
    return source_path, formal_path


def _valid_probe(duration: float, size: int) -> dict:
    return {
        "format": {
            "duration": str(duration),
            "format_name": "mov,mp4,m4a,3gp,3g2,mj2",
            "size": str(size),
        },
        "streams": [
            {
                "index": 0,
                "codec_type": "video",
                "codec_name": "h264",
                "width": 640,
                "height": 360,
                "avg_frame_rate": "30/1",
            },
            {
                "index": 1,
                "codec_type": "audio",
                "codec_name": "aac",
                "sample_rate": "44100",
                "channels": 2,
            },
        ],
    }


class FullVideoDatasetTests(unittest.TestCase):
    def test_normalizes_only_safe_archive_download_urls(self) -> None:
        self.assertEqual(
            normalize_archive_media_url(
                "http://www.archive.org/download/course/lecture.mp4"
            ),
            "https://archive.org/download/course/lecture.mp4",
        )
        rejected = [
            "https://archive.org.evil.example/download/x.mp4",
            "https://user@archive.org/download/x.mp4",
            "https://archive.org:444/download/x.mp4",
            "https://archive.org/download/x.mp4?mirror=evil",
            "https://archive.org/download/%2e%2e/x.mp4",
            "ftp://archive.org/download/x.mp4",
            "https://ia.example.org/download/x.mp4",
        ]
        for value in rejected:
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_archive_media_url(value)

    def test_plan_requires_exact_source_hash_and_identity(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            source_path, formal_path = _dataset_fixture(Path(directory))
            source = json.loads(source_path.read_text(encoding="utf-8"))
            formal = json.loads(formal_path.read_text(encoding="utf-8"))
            plan = build_download_plan(
                source,
                formal,
                source_manifest_sha256=sha256(source_path.read_bytes()).hexdigest(),
            )
            self.assertEqual(len(plan), 10)
            self.assertTrue(
                all(item["media_url"].startswith("https://archive.org/") for item in plan)
            )
            formal["videos"][0]["title"] = "Different title"
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                build_download_plan(
                    source,
                    formal,
                    source_manifest_sha256=sha256(source_path.read_bytes()).hexdigest(),
                )

    def test_terms_gate_runs_before_any_output_or_manifest_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "private"
            with self.assertRaisesRegex(ValueError, "acknowledge_source_terms"):
                download_full_video_dataset(
                    Path(directory) / "missing-source.json",
                    Path(directory) / "missing-formal.json",
                    output,
                )
            self.assertFalse(output.exists())

    def test_downloads_all_ten_atomically_and_reuses_verified_files(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, formal_path = _dataset_fixture(root)
            output = root / "private-videos"
            durations = {
                f"lecture_{index + 1:02d}": 100.0 + index for index in range(10)
            }
            calls: list[str] = []

            def fake_download(url: str, partial_path: Path, **kwargs) -> dict:
                del kwargs
                video_id = Path(url).stem
                calls.append(video_id)
                partial_path.write_bytes(f"complete-media-{video_id}".encode())
                return {
                    "effective_url": (
                        "https://ia801.example.us.archive.org/download/"
                        f"test-course/{video_id}.mp4"
                    ),
                    "http_status": 200,
                    "downloaded_bytes_this_attempt": partial_path.stat().st_size,
                    "resumed_from_bytes": 0,
                }

            def fake_probe(path: Path, **kwargs) -> dict:
                del kwargs
                video_id = path.name.split(".mp4", maxsplit=1)[0]
                return _valid_probe(durations[video_id], path.stat().st_size)

            result = download_full_video_dataset(
                source_path,
                formal_path,
                output,
                acknowledge_source_terms=True,
                downloader=fake_download,
                media_probe=fake_probe,
                generated_at_utc="2026-07-22T00:00:00Z",
            )
            self.assertEqual(len(calls), 10)
            self.assertTrue(result["manifest"]["complete"])
            self.assertEqual(result["manifest"]["video_count"], 10)
            self.assertEqual(result["receipt"]["downloaded_count"], 10)
            self.assertFalse(result["receipt"]["publisher_media_hashes_pinned"])
            self.assertFalse(result["receipt"]["public_artifact"])
            self.assertFalse(result["receipt"]["media_content_included"])
            self.assertEqual(len(list((output / "videos").glob("*.mp4"))), 10)
            self.assertEqual(len(list((output / "videos").glob("*.partial"))), 0)
            public_receipt = build_public_full_video_receipt(result["manifest"])
            self.assertEqual(public_receipt["video_count"], 10)
            self.assertFalse(public_receipt["media_content_included"])
            self.assertFalse(public_receipt["caption_text_included"])
            self.assertFalse(public_receipt["frame_content_included"])
            self.assertNotIn("media_path", public_receipt["records"][0])
            if os.name == "posix":
                self.assertEqual(
                    stat.S_IMODE((output / "videos").stat().st_mode), 0o700
                )
                self.assertEqual(
                    stat.S_IMODE(
                        (output / "videos" / "lecture_01.mp4").stat().st_mode
                    ),
                    0o600,
                )
                self.assertEqual(
                    stat.S_IMODE((output / "media_manifest.json").stat().st_mode),
                    0o600,
                )

            def should_not_download(*args, **kwargs):
                del args, kwargs
                raise AssertionError("verified files should be reused")

            second = download_full_video_dataset(
                source_path,
                formal_path,
                output,
                acknowledge_source_terms=True,
                downloader=should_not_download,
                media_probe=fake_probe,
                generated_at_utc="2026-07-22T00:01:00Z",
            )
            self.assertEqual(second["receipt"]["downloaded_count"], 0)
            self.assertEqual(second["receipt"]["reused_verified_count"], 10)
            self.assertTrue(
                all(
                    item["status"] == "reused_verified_local_file"
                    for item in second["manifest"]["videos"]
                )
            )

    def test_formal_transcript_media_binding_is_enforced_before_download(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_path, formal_path = _dataset_fixture(root)
            transcript_path = formal_path.parent / "transcripts/lecture_01.json"
            transcript = json.loads(transcript_path.read_text(encoding="utf-8"))
            transcript["provenance"]["caption_source_verification"][
                "media_probe_url"
            ] = "https://archive.org/download/different/video.mp4"
            _write_json(transcript_path, transcript)
            with self.assertRaisesRegex(ValueError, "media URL mismatch"):
                download_full_video_dataset(
                    source_path,
                    formal_path,
                    root / "output",
                    acknowledge_source_terms=True,
                    downloader=lambda *args, **kwargs: {},
                )
            self.assertFalse((root / "output").exists())

    def test_curl_resume_command_and_effective_host_receipt(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            partial = Path(directory) / "lecture.mp4.partial"
            partial.write_bytes(b"prefix")
            observed: list[str] = []

            def fake_runner(command: list[str], **kwargs):
                del kwargs
                observed.extend(command)
                return subprocess.CompletedProcess(
                    command,
                    0,
                    stdout=(
                        "https://ia801.example.us.archive.org/download/"
                        "course/lecture.mp4\n206\n120\n"
                    ),
                    stderr="",
                )

            result = curl_download_to_partial(
                "https://archive.org/download/course/lecture.mp4",
                partial,
                runner=fake_runner,
            )
            self.assertEqual(result["resumed_from_bytes"], len(b"prefix"))
            resume_index = observed.index("--continue-at")
            self.assertEqual(observed[resume_index + 1], "-")
            self.assertIn("=https", observed)
            with self.assertRaises(ValueError):
                parse_curl_write_out(
                    "https://archive.org.evil.example/x.mp4\n200\n1\n"
                )

    def test_probe_rejects_duration_mismatch_and_missing_audio(self) -> None:
        summary = validate_media_probe(
            _valid_probe(100.0, 50),
            reference_duration_seconds=100.0,
            actual_file_size_bytes=50,
        )
        self.assertEqual(summary["duration_seconds"], 100.0)
        with self.assertRaisesRegex(ValueError, "duration differs"):
            validate_media_probe(
                _valid_probe(110.0, 50),
                reference_duration_seconds=100.0,
                actual_file_size_bytes=50,
            )
        missing_audio = _valid_probe(100.0, 50)
        missing_audio["streams"] = missing_audio["streams"][:1]
        with self.assertRaisesRegex(ValueError, "no audio stream"):
            validate_media_probe(
                missing_audio,
                reference_duration_seconds=100.0,
                actual_file_size_bytes=50,
            )


if __name__ == "__main__":
    unittest.main()
