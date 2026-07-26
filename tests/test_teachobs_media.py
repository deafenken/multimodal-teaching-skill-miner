from __future__ import annotations

import csv
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import stat
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from teaching_skill_miner.cli import build_parser
from teaching_skill_miner.teachobs_media import (
    EXPECTED_TEST_IDS,
    FEATURE_FAILURE_SCHEMA,
    MAXIMUM_TAIL_CLAMP_DELTA_SECONDS,
    PINNED_REPOSITORY_COMMIT,
    SOURCE_OVERRIDE_SCHEMA,
    VISUAL_EVIDENCE_EVENT_COUNTING_POLICY,
    VISUAL_EVIDENCE_EVENT_TYPES,
    VISUAL_EVIDENCE_SCHEMA,
    _download_with_ytdlp,
    _harden_private_staging_tree,
    _run_private_ytdlp,
    build_teachobs_media_plan,
    clip_preflight,
    download_teachobs_media,
    extract_teachobs_audio_statistics,
    extract_teachobs_midpoint_frames,
    extract_teachobs_multimodal_features,
    extract_teachobs_scene_visual_evidence,
    prepare_teachobs_media_dataset,
    validate_teachobs_cookies_from_browser,
)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _showinfo_stderr(timestamps: list[float]) -> str:
    return "\n".join(
        f"[Parsed_showinfo_2] n: {index} pts: {index} pts_time:{timestamp:.6f}"
        for index, timestamp in enumerate(timestamps)
    )


def _write_source_override(
    path: Path,
    plan: dict,
    *,
    lesson_id: str = "S4",
    override_url: str = "https://www.bilibili.com/video/av561484794/",
) -> Path:
    item = next(row for row in plan["lessons"] if row["lesson_id"] == lesson_id)
    reference_duration = float(item["reference_duration_seconds"])
    evidence = {
        "verification_method": "fixture_read_only_metadata_and_id_reference",
        "observed_title": "Private candidate mirror fixture",
        "observed_duration_seconds": reference_duration - 0.046,
        "official_reference_duration_seconds": reference_duration,
        "duration_absolute_difference_seconds": 0.046,
        "canonical_source_identifier_reference_observed": True,
        "canonical_source_identifier_reference": item["source_url"].split("v=", 1)[1],
        "evidence_retrieved_at_utc": "2026-07-23T00:00:00Z",
    }
    override = {
        "lesson_id": lesson_id,
        "canonical_source_url_sha256": sha256(
            item["source_url"].encode("utf-8")
        ).hexdigest(),
        "override_source_url": override_url,
        "override_source_url_sha256": sha256(
            override_url.encode("utf-8")
        ).hexdigest(),
        "reason": "canonical_source_unavailable_explicit_fixture_mirror",
        "evidence_metadata": evidence,
        "evidence_metadata_sha256": _canonical_sha256(evidence),
        "candidate_same_content_mirror": True,
        "publisher_byte_identity_established": False,
    }
    value = {
        "schema": SOURCE_OVERRIDE_SCHEMA,
        "dataset_id": "teachobs_v0_1_human_validated",
        "repository_commit": PINNED_REPOSITORY_COMMIT,
        "private_artifact": True,
        "safe_to_publish": False,
        "terms": {
            "canonical_source_terms_acknowledgement_required": True,
            "override_source_terms_acknowledgement_required": True,
            "override_redistribution_authorized": False,
        },
        "overrides": [override],
    }
    value["manifest_canonical_sha256"] = _canonical_sha256(value)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _tree_digest(root: Path) -> tuple[str, int, int]:
    digest = sha256()
    files = sorted(
        (path for path in root.rglob("*") if path.is_file()),
        key=lambda path: path.relative_to(root).as_posix(),
    )
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(str(size).encode())
        digest.update(b"\0")
        digest.update(_file_sha256(path).encode())
        digest.update(b"\n")
        total += size
    return digest.hexdigest(), len(files), total


def _write_receipt(root: Path) -> Path:
    repository = root / "repository"
    tree_digest, file_count, byte_count = _tree_digest(repository)
    receipt = {
        "repository_identification": {
            "fixed_commit": PINNED_REPOSITORY_COMMIT,
            "identification_status": "test_fixture_fixed_commit",
        },
        "extracted_repository": {
            "sha256_manifest_v1": tree_digest,
            "files": file_count,
            "bytes": byte_count,
        },
        "licenses": {
            "video_redistribution_authorized_by_dataset": False,
        },
    }
    path = root / "acquisition_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    return path


def _repository_fixture(root: Path) -> Path:
    repository = root / "repository"
    scenes_root = repository / "data" / "scenes"
    scenes_root.mkdir(parents=True)
    fields = [
        "id",
        "week",
        "subject",
        "school_level",
        "country",
        "duration",
        "source",
        "youtube_url",
        "split",
    ]
    rows = []
    for index in range(1, 31):
        lesson_id = f"S{index}"
        rows.append(
            {
                "id": lesson_id,
                "week": "1",
                "subject": "Test",
                "school_level": "Test",
                "country": "US",
                "duration": "0:15",
                "source": "Test source",
                "youtube_url": (
                    f"https://www.youtube.com/watch?v=testVideo{index:02d}"
                ),
                "split": "test" if lesson_id in EXPECTED_TEST_IDS else "train",
            }
        )
        lesson_root = scenes_root / lesson_id
        lesson_root.mkdir()
        transcript_name = f"{lesson_id}_scene_0001.txt"
        (lesson_root / transcript_name).write_text("fixture", encoding="utf-8")
        scene = {
            "id": lesson_id,
            "scene_no": 1,
            "start": 0.0,
            "end": 15.0,
            "mid_frame": f"{lesson_id}_scene_0001.jpg",
            "grid": f"{lesson_id}_scene_0001_grid.jpg",
            "transcript_file": transcript_name,
        }
        (lesson_root / "manifest.jsonl").write_text(
            json.dumps(scene) + "\n", encoding="utf-8"
        )
    lessons_path = repository / "data" / "lessons.csv"
    with lessons_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    _write_receipt(root)
    return repository


def _valid_probe(size: int) -> dict:
    return {
        "format": {
            "duration": "15.0",
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


class TeachObsMediaTests(unittest.TestCase):
    def test_plan_is_bound_to_pinned_tree_and_scene_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            plan = build_teachobs_media_plan(repository, lesson_ids=["S2", "S1"])
            self.assertEqual(plan["lesson_count"], 2)
            self.assertEqual(plan["scene_count"], 2)
            self.assertEqual(
                [row["lesson_id"] for row in plan["lessons"]], ["S1", "S2"]
            )
            self.assertEqual(
                plan["repository_provenance"]["repository_commit"],
                PINNED_REPOSITORY_COMMIT,
            )
            self.assertFalse(plan["public_release_authorized"])
            self.assertFalse(
                plan["terms_boundary"]["source_videos_covered_by_annotation_license"]
            )

            manifest = repository / "data" / "scenes" / "S2" / "manifest.jsonl"
            manifest.write_text(manifest.read_text() + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "tree differs"):
                build_teachobs_media_plan(repository, lesson_ids=["S2"])

    def test_terms_gate_precedes_output_and_dry_run_has_no_side_effect(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            output = root / "private-media"
            with self.assertRaisesRegex(ValueError, "acknowledge_source_terms"):
                prepare_teachobs_media_dataset(repository, output, lesson_ids=["S2"])
            self.assertFalse(output.exists())

            result = prepare_teachobs_media_dataset(
                repository,
                output,
                lesson_ids=["S2"],
                dry_run=True,
                yt_dlp_command=sys.executable,
                ffmpeg_command=sys.executable,
                ffprobe_command=sys.executable,
            )
            self.assertEqual(result["mode"], "dry_run")
            self.assertFalse(result["output_created"])
            self.assertFalse(output.exists())
            self.assertTrue(result["acknowledgement_required_for_execution"])

    def test_parallel_download_is_private_atomic_and_resumable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            output = root / "private-media"
            plan = build_teachobs_media_plan(repository, lesson_ids=["S1", "S2"])
            calls: list[str] = []

            def fake_download(url: str, staging: Path, **kwargs) -> dict:
                del kwargs
                calls.append(url)
                media = staging / "media.mp4"
                media.write_bytes(("private-video:" + url).encode())
                return {
                    "download_path": str(media),
                    "format_policy": "youtube_134+139_then_160+139_fallback",
                    "remote_ejs_allowed": False,
                    "partial_resume_enabled": True,
                }

            def fake_probe(path: Path, **kwargs) -> dict:
                del kwargs
                return _valid_probe(path.stat().st_size)

            result = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=fake_download,
                media_probe=fake_probe,
                jobs=2,
                generated_at_utc="2026-07-22T00:00:00Z",
            )
            self.assertEqual(len(calls), 2)
            self.assertTrue(result["manifest"]["selected_complete"])
            self.assertEqual(result["manifest"]["selected_lesson_count"], 2)
            self.assertFalse(result["manifest"]["public_release_authorized"])
            self.assertNotIn("browser_cookie_credentials", result["manifest"])
            self.assertNotIn("download_transport", result["manifest"])
            self.assertFalse(result["manifest"]["claim_boundary"]["publisher_media_hashes_pinned"])
            for lesson_id in ("S1", "S2"):
                media = output / "videos" / f"{lesson_id}.mp4"
                binding = output / "videos" / f"{lesson_id}.media-binding.json"
                self.assertTrue(media.is_file())
                self.assertTrue(binding.is_file())
                if os.name == "posix":
                    self.assertEqual(stat.S_IMODE(media.stat().st_mode), 0o600)
                    self.assertEqual(stat.S_IMODE(binding.stat().st_mode), 0o600)
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE((output / "videos").stat().st_mode), 0o700)

            def should_not_download(*args, **kwargs):
                del args, kwargs
                raise AssertionError("hash-bound media should be reused")

            second = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=should_not_download,
                media_probe=fake_probe,
                jobs=2,
            )
            self.assertTrue(second["manifest"]["selected_complete"])
            self.assertTrue(
                all(
                    row["status"] == "reused_hash_bound_private_media"
                    for row in second["manifest"]["lessons"]
                )
            )

    def test_parallel_failure_still_receipts_success_and_then_recovers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            output = root / "private-media"
            plan = build_teachobs_media_plan(repository, lesson_ids=["S1", "S2"])

            def one_failure(url: str, staging: Path, **kwargs) -> dict:
                del kwargs
                if url.endswith("testVideo01"):
                    raise RuntimeError("simulated independent failure")
                media = staging / "media.mp4"
                media.write_bytes(b"successful-private-video")
                return {"download_path": str(media)}

            def fake_probe(path: Path, **kwargs) -> dict:
                del kwargs
                return _valid_probe(path.stat().st_size)

            partial = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=one_failure,
                media_probe=fake_probe,
                jobs=2,
                fail_on_incomplete=False,
            )
            self.assertFalse(partial["manifest"]["selected_complete"])
            self.assertEqual(
                [row["lesson_id"] for row in partial["manifest"]["lessons"]], ["S2"]
            )
            failures = json.loads((output / "media_failures.json").read_text())
            self.assertEqual(failures["failure_count"], 1)
            self.assertEqual(failures["failures"][0]["lesson_id"], "S1")
            self.assertFalse(failures["privacy"]["contains_raw_exception_messages"])
            self.assertNotIn("simulated independent failure", json.dumps(failures))

            def recovered_download(url: str, staging: Path, **kwargs) -> dict:
                del url, kwargs
                media = staging / "media.mp4"
                media.write_bytes(b"recovered-private-video")
                return {"download_path": str(media)}

            recovered = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=recovered_download,
                media_probe=fake_probe,
                jobs=2,
            )
            self.assertTrue(recovered["manifest"]["selected_complete"])
            self.assertFalse((output / "media_failures.json").exists())

    def test_cookie_and_transport_provenance_is_minimal_and_profile_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            output = root / "private-media"
            plan = build_teachobs_media_plan(repository, lesson_ids=["S1"])
            observed: dict[str, object] = {}

            def fake_download(url: str, staging: Path, **kwargs) -> dict:
                del url
                observed.update(kwargs)
                media = staging / "media.mp4"
                media.write_bytes(b"credentialed-private-video")
                return {
                    "download_path": str(media),
                    "transport": "fixture",
                    "format_policy": "fixture_credentialed_policy",
                    "youtube_player_client": "default,web_safari",
                    "retrieval_host_class": "canonical_youtube",
                    "remote_ejs_allowed": False,
                    "partial_resume_enabled": True,
                    "cookies_from_browser": "chrome:Profile 1",
                    "account": "private-account@example.invalid",
                    "cookie_path": "/private/browser/Cookies",
                }

            def fake_probe(path: Path, **kwargs) -> dict:
                del kwargs
                return _valid_probe(path.stat().st_size)

            result = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=fake_download,
                media_probe=fake_probe,
                cookies_from_browser="chrome:Profile 1",
                yt_dlp_direct=True,
                yt_dlp_impersonate="chrome",
            )
            self.assertEqual(observed["cookies_from_browser"], "chrome:Profile 1")
            self.assertIs(observed["yt_dlp_direct"], True)
            self.assertEqual(observed["yt_dlp_impersonate"], "chrome")
            manifest = result["manifest"]
            receipt = manifest["lessons"][0]["download_receipt"]
            self.assertTrue(receipt["credentials_used"])
            self.assertEqual(receipt["browser_family"], "chrome")
            self.assertEqual(
                receipt["transport_mode"], "direct_environment_proxy_bypass"
            )
            self.assertEqual(receipt["http_impersonation_target"], "chrome")
            self.assertTrue(
                manifest["browser_cookie_credentials"]["credentials_used"]
            )
            self.assertEqual(
                manifest["download_transport"][
                    "direct_proxy_bypass_lesson_count"
                ],
                1,
            )
            serialized = json.dumps(manifest)
            for secret in (
                "Profile 1",
                "private-account",
                "/private/browser/Cookies",
                "cookies_from_browser",
            ):
                self.assertNotIn(secret, serialized)

            def should_not_download(*args, **kwargs):
                del args, kwargs
                raise AssertionError("hash-bound media should be reused")

            reused = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=should_not_download,
                media_probe=fake_probe,
            )
            reused_receipt = reused["manifest"]["lessons"][0][
                "download_receipt"
            ]
            self.assertTrue(reused_receipt["credentials_used"])
            self.assertTrue(reused_receipt["reused"])
            self.assertEqual(
                reused_receipt["transport_mode"],
                "direct_environment_proxy_bypass",
            )
            self.assertEqual(
                reused_receipt["youtube_player_client"],
                "default,web_safari",
            )
            self.assertFalse(reused_receipt["remote_ejs_allowed"])
            sidecar = json.loads(
                (output / "videos" / "S1.media-binding.json").read_text()
            )
            self.assertEqual(
                sidecar["download_policy"]["format_policy"],
                "fixture_credentialed_policy",
            )

    def test_credentialed_parallel_failure_summary_drops_raw_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            output = root / "private-media"
            plan = build_teachobs_media_plan(repository, lesson_ids=["S1"])

            def failed_download(url: str, staging: Path, **kwargs) -> dict:
                del url, staging, kwargs
                raise RuntimeError(
                    "cookie=secret account=private@example.invalid "
                    "profile=Profile 1 path=/private/Cookies"
                )

            result = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=failed_download,
                cookies_from_browser="chrome:Profile 1",
                yt_dlp_direct=True,
                yt_dlp_impersonate="chrome",
                jobs=2,
                fail_on_incomplete=False,
            )
            self.assertFalse(result["manifest"]["selected_complete"])
            failure_text = (output / "media_failures.json").read_text()
            for secret in (
                "secret",
                "private@example.invalid",
                "Profile 1",
                "/private/Cookies",
            ):
                self.assertNotIn(secret, failure_text)
            failures = json.loads(failure_text)
            self.assertFalse(failures["privacy"]["contains_cookie_material"])
            self.assertFalse(failures["privacy"]["contains_proxy_url"])

    def test_explicit_mirror_requires_double_ack_and_preserves_both_sources(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            plan = build_teachobs_media_plan(repository, lesson_ids=["S4"])
            override_path = _write_source_override(root / "override.json", plan)
            output = root / "private-media"

            with self.assertRaisesRegex(
                ValueError, "acknowledge_override_source_terms"
            ):
                download_teachobs_media(
                    plan,
                    output,
                    acknowledge_source_terms=True,
                    source_override_manifest_path=override_path,
                )
            self.assertFalse(output.exists())

            observed_urls: list[str] = []
            created_modes: list[int] = []

            def fake_download(url: str, staging: Path, **kwargs) -> dict:
                del kwargs
                observed_urls.append(url)
                media = staging / "media.mp4"
                media.write_bytes(b"private-mirror-video")
                if os.name == "posix":
                    created_modes.append(stat.S_IMODE(media.stat().st_mode))
                return {"download_path": str(media), "remote_ejs_allowed": False}

            def fake_probe(path: Path, **kwargs) -> dict:
                del kwargs
                return _valid_probe(path.stat().st_size)

            result = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                source_override_manifest_path=override_path,
                acknowledge_override_source_terms=True,
                downloader=fake_download,
                media_probe=fake_probe,
            )
            self.assertEqual(
                observed_urls,
                ["https://www.bilibili.com/video/av561484794/"],
            )
            record = result["manifest"]["lessons"][0]
            self.assertNotEqual(
                record["canonical_source_url_sha256"],
                record["retrieval_source_url_sha256"],
            )
            self.assertTrue(record["candidate_same_content_mirror"])
            self.assertFalse(record["publisher_byte_identity_established"])
            self.assertTrue(record["override_source_terms_acknowledged"])
            self.assertEqual(
                result["manifest"]["claim_boundary"][
                    "candidate_same_content_mirror_count"
                ],
                1,
            )
            sidecar = json.loads(
                (output / "videos" / "S4.media-binding.json").read_text()
            )
            self.assertEqual(
                sidecar["canonical_source_url_sha256"],
                record["canonical_source_url_sha256"],
            )
            self.assertEqual(
                sidecar["retrieval_source_url_sha256"],
                record["retrieval_source_url_sha256"],
            )
            self.assertTrue(sidecar["candidate_same_content_mirror"])
            self.assertFalse(sidecar["publisher_byte_identity_established"])
            if os.name == "posix":
                self.assertEqual(created_modes, [0o600])
                self.assertEqual(
                    stat.S_IMODE((output / "videos" / "S4.mp4").stat().st_mode),
                    0o600,
                )
                self.assertEqual(
                    stat.S_IMODE(
                        (output / "videos" / "S4.media-binding.json").stat().st_mode
                    ),
                    0o600,
                )

    def test_override_tampering_and_partial_file_permissions_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            mirror_plan = build_teachobs_media_plan(repository, lesson_ids=["S4"])
            override_path = _write_source_override(root / "override.json", mirror_plan)
            value = json.loads(override_path.read_text())
            value["overrides"][0]["override_source_url"] += "tampered"
            override_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                download_teachobs_media(
                    mirror_plan,
                    root / "tampered-output",
                    acknowledge_source_terms=True,
                    source_override_manifest_path=override_path,
                    acknowledge_override_source_terms=True,
                )
            self.assertFalse((root / "tampered-output").exists())

            plan = build_teachobs_media_plan(repository, lesson_ids=["S1"])
            output = root / "partial-output"
            observed_mode: list[int] = []

            def failed_download(url: str, staging: Path, **kwargs) -> dict:
                del url, kwargs
                partial = staging / "media.mp4.part"
                partial.write_bytes(b"resumable-private-partial")
                if os.name == "posix":
                    observed_mode.append(stat.S_IMODE(partial.stat().st_mode))
                raise RuntimeError("simulated interrupted transfer")

            original_umask = os.umask(0o027) if os.name == "posix" else None
            try:
                result = download_teachobs_media(
                    plan,
                    output,
                    acknowledge_source_terms=True,
                    downloader=failed_download,
                    jobs=2,
                    fail_on_incomplete=False,
                )
            finally:
                if os.name == "posix":
                    assert original_umask is not None
                    restored_umask = os.umask(original_umask)
                    self.assertEqual(restored_umask, 0o027)
            self.assertFalse(result["manifest"]["selected_complete"])
            partial = output / ".yt-dlp-staging" / "S1" / "media.mp4.part"
            self.assertTrue(partial.is_file())
            if os.name == "posix":
                self.assertEqual(observed_mode, [0o600])
                self.assertEqual(stat.S_IMODE(partial.stat().st_mode), 0o600)

    @unittest.skipUnless(os.name == "posix", "POSIX permissions are required")
    def test_real_subprocess_staging_is_private_and_resume_is_preserved(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "private staging [literal] $"
            staging.mkdir(mode=0o700)
            resume = staging / "media.mp4.part"
            resume.write_bytes(b"existing-resume-prefix")
            os.chmod(resume, 0o644)
            child_code = r"""
import json
import os
from pathlib import Path
import stat
import sys
import time

root = Path(sys.argv[1])
observed_umask = os.umask(0)
os.umask(observed_umask)
resume_preserved = (root / "media.mp4.part").read_bytes()
child_created = root / "child-created.part"
child_created.write_bytes(b"child-created-under-private-umask")
child_created_mode = stat.S_IMODE(child_created.stat().st_mode)
forced_public = root / "forced-0644.part"
forced_public.write_bytes(b"watcher-must-tighten-this")
os.chmod(forced_public, 0o644)
deadline = time.monotonic() + 2.0
while stat.S_IMODE(forced_public.stat().st_mode) != 0o600:
    if time.monotonic() >= deadline:
        raise RuntimeError("permission watcher did not tighten forced-0644.part")
    time.sleep(0.01)
(root / "subprocess-observations.json").write_text(
    json.dumps(
        {
            "observed_umask": observed_umask,
            "resume_preserved": resume_preserved.decode("ascii"),
            "child_created_mode": child_created_mode,
            "forced_mode_after_watcher": stat.S_IMODE(
                forced_public.stat().st_mode
            ),
        }
    ),
    encoding="utf-8",
)
"""
            result = _run_private_ytdlp(
                [sys.executable, "-c", child_code, str(staging)],
                staging_directory=staging,
                runner=subprocess.run,
                timeout_seconds=10,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            observations = json.loads(
                (staging / "subprocess-observations.json").read_text()
            )
            self.assertEqual(observations["observed_umask"], 0o077)
            self.assertEqual(
                observations["resume_preserved"], "existing-resume-prefix"
            )
            self.assertEqual(observations["child_created_mode"], 0o600)
            self.assertEqual(observations["forced_mode_after_watcher"], 0o600)
            for path in staging.rglob("*"):
                expected_mode = 0o700 if path.is_dir() else 0o600
                self.assertEqual(
                    stat.S_IMODE(path.stat().st_mode),
                    expected_mode,
                    str(path),
                )

    def test_download_wrapper_passes_private_timeout_and_returns_media(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "staging"
            staging.mkdir(mode=0o700)
            observed: dict[str, object] = {}

            def fake_runner(command: list[str], **kwargs) -> SimpleNamespace:
                observed["command"] = command
                observed["timeout"] = kwargs.get("timeout")
                (staging / "media.mp4").write_bytes(b"fixture-private-video")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            receipt = _download_with_ytdlp(
                "https://www.youtube.com/watch?v=testVideo01",
                staging,
                yt_dlp_command=sys.executable,
                js_runtime=None,
                timeout_seconds=37,
                runner=fake_runner,
            )

            self.assertEqual(observed["timeout"], 37)
            self.assertTrue(Path(receipt["download_path"]).is_file())
            self.assertEqual(receipt["child_process_umask"], "0077")
            self.assertTrue(receipt["staging_permission_watcher_enabled"])
            command = observed["command"]
            assert isinstance(command, list)
            format_index = command.index("--format")
            self.assertEqual(command[format_index + 1], "134+140/18/160+139")
            extractor_index = command.index("--extractor-args")
            self.assertEqual(
                command[extractor_index + 1],
                "youtube:player_client=android_vr",
            )
            self.assertEqual(receipt["youtube_player_client"], "android_vr")
            self.assertEqual(command.count("--no-remote-components"), 1)
            self.assertEqual(
                receipt["format_policy"],
                "youtube_134+140_then_18_then_160+139_fallback",
            )
            self.assertNotIn("--cookies-from-browser", command)
            self.assertNotIn("--proxy", command)
            self.assertNotIn("--impersonate", command)
            self.assertNotIn("credentials_used", receipt)
            self.assertNotIn("transport_mode", receipt)

    def test_download_wrapper_cookie_direct_and_impersonation_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "staging"
            staging.mkdir(mode=0o700)
            observed: dict[str, object] = {}

            def fake_runner(command: list[str], **kwargs) -> SimpleNamespace:
                del kwargs
                observed["command"] = command
                (staging / "media.mp4").write_bytes(b"fixture-private-video")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            receipt = _download_with_ytdlp(
                "https://www.youtube.com/watch?v=testVideo01",
                staging,
                yt_dlp_command=[sys.executable, "-m", "yt_dlp"],
                js_runtime="node:/private/bin/node",
                timeout_seconds=37,
                runner=fake_runner,
                cookies_from_browser="chrome:Profile 1",
                yt_dlp_direct=True,
                yt_dlp_impersonate="chrome",
            )
            command = observed["command"]
            assert isinstance(command, list)
            cookie_index = command.index("--cookies-from-browser")
            self.assertEqual(command[cookie_index + 1], "chrome:Profile 1")
            proxy_index = command.index("--proxy")
            self.assertEqual(command[proxy_index + 1], "")
            impersonate_index = command.index("--impersonate")
            self.assertEqual(command[impersonate_index + 1], "chrome")
            extractor_index = command.index("--extractor-args")
            self.assertEqual(
                command[extractor_index + 1],
                "youtube:player_client=default,web_safari",
            )
            self.assertNotIn("--cookies", command)
            self.assertTrue(receipt["credentials_used"])
            self.assertEqual(receipt["browser_family"], "chrome")
            self.assertEqual(
                receipt["youtube_player_client"], "default,web_safari"
            )
            self.assertEqual(command.count("--no-remote-components"), 1)
            self.assertFalse(receipt["remote_ejs_allowed"])
            self.assertEqual(receipt["javascript_runtime_family"], "node")
            self.assertFalse(receipt["browser_profile_recorded"])
            self.assertEqual(
                receipt["transport_mode"], "direct_environment_proxy_bypass"
            )
            self.assertEqual(receipt["http_impersonation_target"], "chrome")
            self.assertNotIn("Profile 1", json.dumps(receipt))

    def test_download_wrapper_allows_separate_av_fallback_for_mirrors(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "staging"
            staging.mkdir(mode=0o700)
            observed: dict[str, object] = {}

            def fake_runner(command: list[str], **kwargs) -> SimpleNamespace:
                del kwargs
                observed["command"] = command
                (staging / "media.mp4").write_bytes(b"fixture-private-video")
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            receipt = _download_with_ytdlp(
                "https://web.archive.org/web/20231227092138id_/http://example.test/video",
                staging,
                yt_dlp_command=sys.executable,
                js_runtime=None,
                timeout_seconds=37,
                runner=fake_runner,
            )
            command = observed["command"]
            assert isinstance(command, list)
            format_index = command.index("--format")
            self.assertEqual(
                command[format_index + 1],
                (
                    "bestvideo[height<=480]+bestaudio/"
                    "best[height<=480]/bestvideo+bestaudio/best"
                ),
            )
            self.assertEqual(
                receipt["format_policy"],
                (
                    "explicit_mirror_best_height_lte_480_then_"
                    "separate_av_then_best"
                ),
            )
            self.assertNotIn("--extractor-args", command)

    def test_credentialed_download_failure_discards_private_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "staging"
            staging.mkdir(mode=0o700)

            def fake_runner(command: list[str], **kwargs) -> SimpleNamespace:
                del command, kwargs
                return SimpleNamespace(
                    returncode=1,
                    stdout="private-account@example.test",
                    stderr="chrome:Private Profile secret-cookie-value",
                )

            with self.assertRaises(RuntimeError) as caught:
                _download_with_ytdlp(
                    "https://www.youtube.com/watch?v=testVideo01",
                    staging,
                    yt_dlp_command=sys.executable,
                    js_runtime=None,
                    timeout_seconds=37,
                    runner=fake_runner,
                    cookies_from_browser="chrome:Private Profile",
                )
            rendered = str(caught.exception)
            self.assertIn("browser credentials were enabled", rendered)
            self.assertNotIn("Private Profile", rendered)
            self.assertNotIn("private-account", rendered)
            self.assertNotIn("secret-cookie-value", rendered)

    def test_credentialed_timeout_does_not_chain_private_command(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "staging"
            staging.mkdir(mode=0o700)

            def fake_runner(command: list[str], **kwargs) -> SimpleNamespace:
                del kwargs
                raise subprocess.TimeoutExpired(
                    command,
                    timeout=37,
                    output="private-account@example.test",
                    stderr="secret-cookie-value",
                )

            with self.assertRaises(RuntimeError) as caught:
                _download_with_ytdlp(
                    "https://www.youtube.com/watch?v=testVideo01",
                    staging,
                    yt_dlp_command=sys.executable,
                    js_runtime=None,
                    timeout_seconds=37,
                    runner=fake_runner,
                    cookies_from_browser="chrome:Private Profile",
                )
            self.assertEqual(
                str(caught.exception),
                "yt-dlp timed out; its private partial files were preserved",
            )
            self.assertIsNone(caught.exception.__cause__)
            self.assertIsNone(caught.exception.__context__)

    def test_download_wrapper_rejects_unsafe_cookie_and_transport_specs(self) -> None:
        self.assertEqual(
            validate_teachobs_cookies_from_browser("safari"),
            ("safari", "safari"),
        )
        self.assertEqual(
            validate_teachobs_cookies_from_browser("chrome:Profile 1"),
            ("chrome:Profile 1", "chrome"),
        )
        with tempfile.TemporaryDirectory() as directory:
            staging = Path(directory) / "staging"
            staging.mkdir(mode=0o700)
            for spec in (
                "chrome:",
                "chrome:../Default",
                "chrome:/tmp/profile",
                "chrome::container",
                "chrome+keyring",
                "chrome\n--proxy",
                "unknown",
            ):
                with self.subTest(spec=spec), self.assertRaises(ValueError):
                    _download_with_ytdlp(
                        "https://www.youtube.com/watch?v=testVideo01",
                        staging,
                        yt_dlp_command=sys.executable,
                        js_runtime=None,
                        timeout_seconds=37,
                        runner=subprocess.run,
                        cookies_from_browser=spec,
                    )
            with self.assertRaisesRegex(ValueError, "unsupported target"):
                _download_with_ytdlp(
                    "https://www.youtube.com/watch?v=testVideo01",
                    staging,
                    yt_dlp_command=sys.executable,
                    js_runtime=None,
                    timeout_seconds=37,
                    runner=subprocess.run,
                    yt_dlp_impersonate="firefox",
                )

    def test_installed_media_cli_forwards_explicit_private_transport(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "prepare-teachobs-media",
                "--dry-run",
                "--cookies-from-browser",
                "chrome:Profile 1",
                "--yt-dlp-direct",
                "--yt-dlp-impersonate",
                "chrome",
            ]
        )
        fixture = {
            "mode": "dry_run",
            "plan": {
                "lesson_count": 30,
                "scene_count": 5158,
                "reference_duration_seconds": 77334.0,
            },
        }
        with patch(
            "teaching_skill_miner.cli.prepare_teachobs_media_dataset",
            return_value=fixture,
        ) as mocked_prepare, patch("builtins.print"):
            self.assertEqual(args.func(args), 0)
        kwargs = mocked_prepare.call_args.kwargs
        self.assertEqual(kwargs["cookies_from_browser"], "chrome:Profile 1")
        self.assertIs(kwargs["yt_dlp_direct"], True)
        self.assertEqual(kwargs["yt_dlp_impersonate"], "chrome")
        with self.assertRaises(SystemExit):
            parser.parse_args(
                [
                    "prepare-teachobs-media",
                    "--dry-run",
                    "--yt-dlp-impersonate",
                    "firefox",
                ]
            )

    @unittest.skipUnless(os.name == "posix", "POSIX permissions are required")
    def test_staging_hardener_rejects_symlink_without_touching_target(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            staging = root / "staging"
            staging.mkdir(mode=0o700)
            outside = root / "outside-private-boundary.part"
            outside.write_bytes(b"must-not-be-followed")
            os.chmod(outside, 0o644)
            (staging / "escaping.part").symlink_to(outside)

            with self.assertRaisesRegex(ValueError, "symlink"):
                _harden_private_staging_tree(staging)

            self.assertEqual(stat.S_IMODE(outside.stat().st_mode), 0o644)
            self.assertEqual(outside.read_bytes(), b"must-not-be-followed")

    def test_short_tail_uses_explicit_final_scene_timestamp_clamp(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            plan = build_teachobs_media_plan(repository, lesson_ids=["S6"])
            item = plan["lessons"][0]
            item["reference_duration_seconds"] = 2691.0
            item["scene_count"] = 180
            item["last_scene_end_seconds"] = 2700.0
            item["scenes"] = [
                {
                    "scene_no": index,
                    "start": float((index - 1) * 15),
                    "end": float(index * 15),
                    "midpoint": float((index - 0.5) * 15),
                    "declared_mid_frame": f"S6_scene_{index:04d}.jpg",
                    "declared_grid": f"S6_scene_{index:04d}_grid.jpg",
                }
                for index in range(1, 181)
            ]
            plan["scene_count"] = 180
            plan["reference_duration_seconds"] = 2691.0
            unsigned = {key: value for key, value in plan.items() if key != "plan_sha256"}
            plan["plan_sha256"] = _canonical_sha256(unsigned)
            output = root / "private-media"

            def fake_download(url: str, staging: Path, **kwargs) -> dict:
                del url, kwargs
                media = staging / "media.mp4"
                media.write_bytes(b"complete-s6-private-video")
                return {"download_path": str(media)}

            def s6_probe(path: Path, **kwargs) -> dict:
                del kwargs
                probe = _valid_probe(path.stat().st_size)
                probe["format"]["duration"] = "2690.960544"
                return probe

            media_result = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=fake_download,
                media_probe=s6_probe,
            )
            media_record = media_result["manifest"]["lessons"][0]
            self.assertTrue(
                media_record["tail_frame_timing"]["tail_frame_clamp_required"]
            )
            commands: list[list[str]] = []

            def frame_runner(command: list[str], **kwargs) -> SimpleNamespace:
                del kwargs
                commands.append(command)
                output_path = command[-1]
                count = int(command[command.index("-frames:v") + 1])
                for index in range(1, count + 1):
                    Path(output_path.replace("%04d", f"{index:04d}")).write_bytes(
                        b"timestamp-verified-private-frame"
                    )
                timestamps = [
                    7.5 + (index * 15.0) for index in range(count - 1)
                ] + [2690.721367]
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr(timestamps),
                )

            frame_result = extract_teachobs_midpoint_frames(
                output / "videos" / "S6.mp4",
                item,
                output / "frames",
                media_sha256=media_record["media_sha256"],
                media_duration_seconds=2690.960544,
                ffmpeg_command=sys.executable,
                runner=frame_runner,
            )
            self.assertEqual(len(commands), 1)
            self.assertIn("selected_n", commands[0][commands[0].index("-vf") + 1])
            self.assertNotIn("fps=1/15", commands[0][commands[0].index("-vf") + 1])
            frames = frame_result["task"]["frames"]
            self.assertTrue(frames[-1]["timestamp_clamped"])
            self.assertEqual(frames[-1]["requested_timestamp"], 2692.5)
            self.assertAlmostEqual(frames[-1]["timestamp"], 2690.710544, places=6)
            self.assertAlmostEqual(
                frames[-1]["timestamp_clamp_delta_seconds"], 1.789456, places=6
            )
            self.assertLessEqual(
                frames[-1]["timestamp_clamp_delta_seconds"],
                MAXIMUM_TAIL_CLAMP_DELTA_SECONDS,
            )
            self.assertFalse(any(row["timestamp_clamped"] for row in frames[:-1]))
            self.assertTrue(
                all(row["timestamp_verified_within_tolerance"] for row in frames)
            )
            self.assertAlmostEqual(
                frames[-1]["source_frame_timestamp"], 2690.721367, places=6
            )

            with self.assertRaisesRegex(ValueError, "non-tail scene clamp"):
                extract_teachobs_midpoint_frames(
                    output / "videos" / "S6.mp4",
                    {**item, "lesson_id": "S6"},
                    root / "invalid-tail-frames",
                    media_sha256=media_record["media_sha256"],
                    media_duration_seconds=2670.0,
                    ffmpeg_command=sys.executable,
                    runner=frame_runner,
                )

    def test_midpoint_timing_fails_closed_and_legacy_cache_upgrades(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            item = build_teachobs_media_plan(
                repository, lesson_ids=["S2"]
            )["lessons"][0]
            media = root / "private-video.mp4"
            media.write_bytes(b"fixture-private-media")
            media_digest = _file_sha256(media)
            frames_root = root / "frames"

            def bad_timing_runner(command: list[str], **kwargs) -> SimpleNamespace:
                del kwargs
                Path(command[-1].replace("%04d", "0001")).write_bytes(
                    b"wrong-time-frame"
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr([7.75]),
                )

            with self.assertRaisesRegex(RuntimeError, "tolerance exceeded"):
                extract_teachobs_midpoint_frames(
                    media,
                    item,
                    frames_root,
                    media_sha256=media_digest,
                    media_duration_seconds=15.0,
                    ffmpeg_command=sys.executable,
                    runner=bad_timing_runner,
                )
            self.assertFalse((frames_root / "S2").exists())

            def verified_runner(command: list[str], **kwargs) -> SimpleNamespace:
                del kwargs
                Path(command[-1].replace("%04d", "0001")).write_bytes(
                    b"verified-time-frame"
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr([7.52]),
                )

            first = extract_teachobs_midpoint_frames(
                media,
                item,
                frames_root,
                media_sha256=media_digest,
                media_duration_seconds=15.0,
                ffmpeg_command=sys.executable,
                runner=verified_runner,
            )
            task_path = Path(first["task_path"])
            legacy = json.loads(task_path.read_text())
            for field in (
                "extraction_method",
                "timestamp_selection_policy",
                "timestamp_evidence_source",
                "timestamp_tolerance_seconds",
            ):
                legacy.pop(field)
            for field in (
                "source_frame_timestamp",
                "timestamp_selection_error_seconds",
                "timestamp_verified_within_tolerance",
            ):
                legacy["frames"][0].pop(field)
            task_path.write_text(json.dumps(legacy), encoding="utf-8")
            calls = 0

            def replacement_runner(
                command: list[str], **kwargs
            ) -> SimpleNamespace:
                nonlocal calls
                del kwargs
                calls += 1
                Path(command[-1].replace("%04d", "0001")).write_bytes(
                    b"replacement-verified-frame"
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr([7.51]),
                )

            upgraded = extract_teachobs_midpoint_frames(
                media,
                item,
                frames_root,
                media_sha256=media_digest,
                media_duration_seconds=15.0,
                ffmpeg_command=sys.executable,
                runner=replacement_runner,
            )
            self.assertFalse(upgraded["reused"])
            self.assertEqual(calls, 1)
            self.assertTrue(
                upgraded["task"]["frames"][0][
                    "timestamp_verified_within_tolerance"
                ]
            )
            self.assertEqual(
                (frames_root / "S2" / "frame_0001.jpg").read_bytes(),
                b"replacement-verified-frame",
            )

    def test_audio_tail_padding_records_real_coverage_and_excludes_padding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "private.mp4"
            media.write_bytes(b"private-media")
            output = root / "features" / "S6.json"
            item = {
                "lesson_id": "S6",
                "scene_manifest_sha256": "a" * 64,
                "scene_count": 2,
                "scenes": [
                    {"scene_no": 1, "start": 0.0, "end": 15.0},
                    {"scene_no": 2, "start": 15.0, "end": 30.0},
                ],
            }

            def tail_audio_runner(command: list[str], **kwargs):
                del kwargs
                import numpy as np

                # The media ends five seconds into the final 15-second scene.
                samples = np.full(20 * 16_000, 1000, dtype="<i2")
                Path(command[-1]).write_bytes(samples.tobytes())
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            result = extract_teachobs_audio_statistics(
                media,
                item,
                output,
                media_sha256=_file_sha256(media),
                media_duration_seconds=20.0,
                ffmpeg_command=sys.executable,
                runner=tail_audio_runner,
            )
            self.assertFalse(result["reused"])
            audio = result["features"]
            self.assertEqual(audio["tail_padding_scene_count"], 1)
            self.assertEqual(audio["tail_padding_duration_seconds"], 10.0)
            self.assertTrue(audio["statistics_exclude_padding"])
            first, tail = audio["scenes"]
            self.assertFalse(first["padding_applied"])
            self.assertEqual(first["coverage_fraction"], 1.0)
            self.assertTrue(tail["padding_applied"])
            self.assertEqual(tail["observed_duration_seconds"], 5.0)
            self.assertEqual(tail["missing_duration_seconds"], 10.0)
            self.assertEqual(tail["padding_duration_seconds"], 10.0)
            self.assertEqual(tail["coverage_fraction"], 0.33333333)
            self.assertEqual(tail["statistics_sample_count"], 5 * 16_000)
            self.assertEqual(tail["analysis_window_sample_count"], 15 * 16_000)
            self.assertFalse(tail["minimum_scene_coverage_met"])
            self.assertTrue(tail["tail_eof_exception_applied"])
            # If ten padded seconds were included, these values would be diluted
            # and the silence fraction would be 2/3 rather than zero.
            self.assertEqual(tail["rms_amplitude"], 0.03051758)
            self.assertEqual(tail["mean_absolute_amplitude"], 0.03051758)
            self.assertEqual(tail["silence_fraction"], 0.0)

            reused = extract_teachobs_audio_statistics(
                media,
                item,
                output,
                media_sha256=_file_sha256(media),
                media_duration_seconds=20.0,
                ffmpeg_command=sys.executable,
                runner=lambda *args, **kwargs: self.fail("audio cache missed"),
            )
            self.assertTrue(reused["reused"])

            tampered = json.loads(output.read_text(encoding="utf-8"))
            tampered["scenes"][-1]["statistics_sample_count"] = 15 * 16_000
            output.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "audio tail audit"):
                extract_teachobs_audio_statistics(
                    media,
                    item,
                    output,
                    media_sha256=_file_sha256(media),
                    media_duration_seconds=20.0,
                    ffmpeg_command=sys.executable,
                    runner=lambda *args, **kwargs: self.fail("tampered cache used"),
                )

    def test_audio_non_tail_gap_fails_closed_even_at_media_eof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "private.mp4"
            media.write_bytes(b"private-media")
            output = root / "features" / "S6.json"
            item = {
                "lesson_id": "S6",
                "scene_manifest_sha256": "a" * 64,
                "scene_count": 3,
                "scenes": [
                    {"scene_no": 1, "start": 0.0, "end": 15.0},
                    {"scene_no": 2, "start": 15.0, "end": 30.0},
                    {"scene_no": 3, "start": 30.0, "end": 45.0},
                ],
            }

            def short_audio_runner(command: list[str], **kwargs):
                del kwargs
                Path(command[-1]).write_bytes(b"\0\0" * (10 * 16_000))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with self.assertRaisesRegex(RuntimeError, "non-tail gap"):
                extract_teachobs_audio_statistics(
                    media,
                    item,
                    output,
                    media_sha256=_file_sha256(media),
                    media_duration_seconds=10.0,
                    ffmpeg_command=sys.executable,
                    runner=short_audio_runner,
                )
            self.assertFalse(output.exists())

    def test_audio_tail_must_align_with_media_eof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "private.mp4"
            media.write_bytes(b"private-media")
            output = root / "features" / "S6.json"
            item = {
                "lesson_id": "S6",
                "scene_manifest_sha256": "a" * 64,
                "scene_count": 2,
                "scenes": [
                    {"scene_no": 1, "start": 0.0, "end": 15.0},
                    {"scene_no": 2, "start": 15.0, "end": 30.0},
                ],
            }

            def early_audio_eof_runner(command: list[str], **kwargs):
                del kwargs
                Path(command[-1]).write_bytes(b"\0\0" * (20 * 16_000))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with self.assertRaisesRegex(RuntimeError, "does not align with media EOF"):
                extract_teachobs_audio_statistics(
                    media,
                    item,
                    output,
                    media_sha256=_file_sha256(media),
                    media_duration_seconds=30.0,
                    ffmpeg_command=sys.executable,
                    runner=early_audio_eof_runner,
                )
            self.assertFalse(output.exists())

    def test_frames_and_audio_cover_every_scene_and_reuse_cache(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            output = root / "private-media"
            plan = build_teachobs_media_plan(repository, lesson_ids=["S2"])

            def fake_download(url: str, staging: Path, **kwargs) -> dict:
                del url, kwargs
                media = staging / "media.mp4"
                media.write_bytes(b"private-complete-video")
                return {"download_path": str(media), "partial_resume_enabled": True}

            def fake_probe(path: Path, **kwargs) -> dict:
                del kwargs
                return _valid_probe(path.stat().st_size)

            media_result = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=fake_download,
                media_probe=fake_probe,
            )
            frame_calls = 0
            audio_calls = 0

            def fake_frame_runner(command: list[str], **kwargs):
                nonlocal frame_calls
                del kwargs
                frame_calls += 1
                pattern = command[-1]
                Path(pattern.replace("%04d", "0001")).write_bytes(b"jpeg-frame")
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr([7.5]),
                )

            def fake_audio_runner(command: list[str], **kwargs):
                nonlocal audio_calls
                del kwargs
                audio_calls += 1
                # One complete 15-second, 16-kHz, little-endian mono scene.
                import numpy as np

                samples = np.full(15 * 16_000, 1000, dtype="<i2")
                Path(command[-1]).write_bytes(samples.tobytes())
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def fake_image_metrics(path: Path) -> dict:
                del path
                return {
                    "backend": "fixture",
                    "dhash64": "0000000000000000",
                    "edge_difference_mean": 1.0,
                }

            def fake_ocr(path: Path, language: str, **kwargs):
                del path, kwargs
                return "fixture board text", {
                    "status": "completed",
                    "language": language,
                    "accepted_word_count": 3,
                    "confidence_is_calibrated_probability": False,
                }

            features = extract_teachobs_multimodal_features(
                plan,
                media_result["manifest"],
                output,
                acknowledge_source_terms=True,
                ffmpeg_command=sys.executable,
                frame_runner=fake_frame_runner,
                audio_runner=fake_audio_runner,
                image_metric_extractor=fake_image_metrics,
                ocr_extractor=fake_ocr,
                generated_at_utc="2026-07-22T00:00:00Z",
            )
            manifest = features["manifest"]
            self.assertTrue(manifest["complete"])
            self.assertEqual(
                manifest["media_manifest_sha256"],
                _canonical_sha256(media_result["manifest"]),
            )
            self.assertEqual(manifest["scene_count"], 1)
            self.assertTrue(manifest["claim_boundary"]["full_scene_midpoint_frames_extracted"])
            self.assertFalse(manifest["claim_boundary"]["multimodal_gain_established"])
            audio = json.loads((output / "features" / "audio" / "S2.json").read_text())
            self.assertEqual(len(audio["scenes"]), 1)
            self.assertEqual(audio["scenes"][0]["coverage_fraction"], 1.0)
            self.assertEqual(audio["scenes"][0]["zero_crossing_rate"], 0.0)
            self.assertFalse(audio["claim_boundary"]["speech_content_transcribed"])
            task = json.loads((output / "frames" / "S2" / "task.json").read_text())
            self.assertEqual(task["frame_count"], 1)
            self.assertEqual(task["frames"][0]["timestamp"], 7.5)
            self.assertFalse(manifest["privacy"]["contains_transcript_text"])
            self.assertNotIn("transcript_text", manifest["lessons"][0])
            evidence = json.loads(
                (output / "features" / "visual_evidence" / "S2.json").read_text()
            )
            self.assertEqual(evidence["scene_count"], 1)
            self.assertEqual(evidence["ocr_status_counts"], {"completed": 1})
            self.assertFalse(evidence["privacy"]["safe_to_publish"])

            second = extract_teachobs_multimodal_features(
                plan,
                media_result["manifest"],
                output,
                acknowledge_source_terms=True,
                ffmpeg_command=sys.executable,
                frame_runner=lambda *args, **kwargs: self.fail("frame cache missed"),
                audio_runner=lambda *args, **kwargs: self.fail("audio cache missed"),
            )
            self.assertTrue(second["manifest"]["complete"])
            self.assertEqual(frame_calls, 1)
            self.assertEqual(audio_calls, 1)

            evidence_path = output / "features" / "visual_evidence" / "S2.json"
            legacy_evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
            legacy_evidence["schema"] = (
                "teaching_skill_miner."
                "teachobs_private_scene_visual_evidence.v1"
            )
            legacy_evidence["configuration"].pop("event_type_counting_policy")
            legacy_evidence["configuration_sha256"] = _canonical_sha256(
                legacy_evidence["configuration"]
            )
            legacy_evidence["event_type_counts"].pop("code_or_formula_visible")
            evidence_path.write_text(
                json.dumps(legacy_evidence), encoding="utf-8"
            )
            manifest_path = output / "feature_manifest.json"
            legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            legacy_manifest["lessons"][0]["visual_evidence_sha256"] = (
                _file_sha256(evidence_path)
            )
            manifest_path.write_text(
                json.dumps(legacy_manifest), encoding="utf-8"
            )
            upgraded = extract_teachobs_multimodal_features(
                plan,
                media_result["manifest"],
                output,
                acknowledge_source_terms=True,
                ffmpeg_command=sys.executable,
                frame_runner=lambda *args, **kwargs: self.fail(
                    "legacy count upgrade missed frame cache"
                ),
                audio_runner=lambda *args, **kwargs: self.fail(
                    "legacy count upgrade missed audio cache"
                ),
                image_metric_extractor=lambda path: self.fail(
                    f"legacy count upgrade re-extracted {path}"
                ),
                ocr_extractor=lambda *args, **kwargs: self.fail(
                    "legacy count upgrade reran OCR"
                ),
            )["manifest"]
            upgraded_record = upgraded["lessons"][0]
            self.assertTrue(
                upgraded_record["visual_evidence_upgraded_legacy_event_counts"]
            )
            self.assertFalse(upgraded_record["visual_evidence_reused"])
            upgraded_evidence = json.loads(
                evidence_path.read_text(encoding="utf-8")
            )
            self.assertEqual(upgraded_evidence["schema"], VISUAL_EVIDENCE_SCHEMA)
            self.assertEqual(
                sum(upgraded_evidence["event_type_counts"].values()),
                upgraded_evidence["event_count"],
            )

    def test_full_feature_cache_upgrade_rejects_tamper_then_rebuilds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            output = root / "private-media"
            plan = build_teachobs_media_plan(repository, lesson_ids=["S2"])

            def fake_download(url: str, staging: Path, **kwargs) -> dict:
                del url, kwargs
                media = staging / "media.mp4"
                media.write_bytes(b"private-complete-video")
                return {"download_path": str(media)}

            media_result = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=fake_download,
                media_probe=lambda path, **kwargs: _valid_probe(
                    path.stat().st_size
                ),
            )

            def frame_runner(command: list[str], **kwargs) -> SimpleNamespace:
                del kwargs
                Path(command[-1].replace("%04d", "0001")).write_bytes(
                    b"private-frame-before-upgrade"
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr([7.5]),
                )

            def audio_runner(command: list[str], **kwargs) -> SimpleNamespace:
                del kwargs
                Path(command[-1]).write_bytes(b"\0\0" * (15 * 16_000))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            def metrics(path: Path) -> dict:
                del path
                return {
                    "backend": "fixture",
                    "dhash64": "0000000000000000",
                    "edge_difference_mean": 1.0,
                }

            def ocr(path: Path, language: str, **kwargs):
                del path, kwargs
                return "private fixture OCR", {
                    "status": "completed",
                    "language": language,
                    "accepted_word_count": 3,
                    "confidence_is_calibrated_probability": False,
                }

            extract_teachobs_multimodal_features(
                plan,
                media_result["manifest"],
                output,
                acknowledge_source_terms=True,
                frame_runner=frame_runner,
                audio_runner=audio_runner,
                image_metric_extractor=metrics,
                ocr_extractor=ocr,
                ffmpeg_command=sys.executable,
            )
            task_path = output / "frames" / "S2" / "task.json"
            evidence_path = output / "features" / "visual_evidence" / "S2.json"
            audio_path = output / "features" / "audio" / "S2.json"
            feature_manifest_path = output / "feature_manifest.json"

            legacy_task = json.loads(task_path.read_text())
            legacy_task["sampling_method"] = (
                "ffmpeg_scene_midpoints_with_explicit_bounded_tail_clamp_v2"
            )
            for field in (
                "extraction_method",
                "timestamp_selection_policy",
                "timestamp_evidence_source",
                "timestamp_tolerance_seconds",
                "maximum_observed_timestamp_error_seconds",
                "showinfo_timestamp_count",
                "showinfo_timestamps_used",
            ):
                legacy_task.pop(field)
            for field in (
                "source_frame_timestamp",
                "timestamp_selection_error_seconds",
                "timestamp_verified_within_tolerance",
            ):
                legacy_task["frames"][0].pop(field)
            task_path.write_text(json.dumps(legacy_task), encoding="utf-8")

            legacy_evidence = json.loads(evidence_path.read_text())
            legacy_evidence["task_file_sha256"] = _file_sha256(task_path)
            for field in (
                "source_frame_timestamp",
                "timestamp_selection_error_seconds",
                "timestamp_verified_within_tolerance",
            ):
                legacy_evidence["scenes"][0].pop(field)
            evidence_path.write_text(
                json.dumps(legacy_evidence), encoding="utf-8"
            )

            legacy_audio = json.loads(audio_path.read_text())
            legacy_audio["algorithm"] = (
                "teaching_skill_miner.scene_pcm_statistics.v1"
            )
            for field in (
                "media_duration_seconds",
                "decoded_real_sample_count",
                "decoded_real_duration_seconds",
                "media_eof_alignment_tolerance_seconds",
                "minimum_scene_coverage",
                "tail_padding_scene_count",
                "tail_padding_sample_count",
                "tail_padding_duration_seconds",
                "statistics_exclude_padding",
                "cache_upgrade_from_algorithm",
            ):
                legacy_audio.pop(field)
            for field in (
                "requested_sample_count",
                "observed_duration_seconds",
                "real_audio_coverage_fraction",
                "coverage_basis",
                "missing_duration_seconds",
                "padding_applied",
                "padding_sample_count",
                "padding_duration_seconds",
                "padding_value_normalized",
                "analysis_window_sample_count",
                "statistics_sample_count",
                "statistics_exclude_padding",
                "minimum_scene_coverage_met",
                "tail_eof_exception_applied",
                "media_eof_alignment_delta_seconds",
            ):
                legacy_audio["scenes"][0].pop(field)
            audio_path.write_text(json.dumps(legacy_audio), encoding="utf-8")

            legacy_manifest = json.loads(feature_manifest_path.read_text())
            legacy_record = legacy_manifest["lessons"][0]
            legacy_record["frame_task_sha256"] = _file_sha256(task_path)
            legacy_record["visual_evidence_sha256"] = _file_sha256(evidence_path)
            legacy_record["audio_feature_sha256"] = _file_sha256(audio_path)
            feature_manifest_path.write_text(
                json.dumps(legacy_manifest), encoding="utf-8"
            )
            valid_legacy_evidence = evidence_path.read_bytes()
            valid_legacy_audio = audio_path.read_bytes()
            valid_legacy_manifest = feature_manifest_path.read_bytes()
            legacy_task_digest = _file_sha256(task_path)

            tampered_evidence = json.loads(evidence_path.read_text())
            tampered_evidence["task_file_sha256"] = "f" * 64
            evidence_path.write_text(
                json.dumps(tampered_evidence), encoding="utf-8"
            )
            with self.assertRaisesRegex(
                RuntimeError, "successful independent artifacts were retained"
            ):
                extract_teachobs_multimodal_features(
                    plan,
                    media_result["manifest"],
                    output,
                    acknowledge_source_terms=True,
                    frame_runner=lambda *args, **kwargs: self.fail(
                        "tampered visual cache was silently replaced"
                    ),
                    audio_runner=audio_runner,
                    image_metric_extractor=metrics,
                    ocr_extractor=ocr,
                    ffmpeg_command=sys.executable,
                )
            self.assertEqual(_file_sha256(task_path), legacy_task_digest)

            evidence_path.write_bytes(valid_legacy_evidence)
            feature_manifest_path.write_bytes(valid_legacy_manifest)
            tampered_audio = json.loads(audio_path.read_text())
            tampered_audio["media_sha256"] = "f" * 64
            audio_path.write_text(json.dumps(tampered_audio), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "binding differs"):
                extract_teachobs_audio_statistics(
                    output / "videos" / "S2.mp4",
                    plan["lessons"][0],
                    audio_path,
                    media_sha256=media_result["manifest"]["lessons"][0][
                        "media_sha256"
                    ],
                    media_duration_seconds=15.0,
                    ffmpeg_command=sys.executable,
                    runner=lambda *args, **kwargs: self.fail(
                        "tampered audio cache was silently replaced"
                    ),
                )
            self.assertEqual(
                json.loads(audio_path.read_text())["media_sha256"], "f" * 64
            )
            audio_path.write_bytes(valid_legacy_audio)
            upgrade_frame_calls = 0
            upgrade_audio_calls = 0

            def upgrade_frame_runner(
                command: list[str], **kwargs
            ) -> SimpleNamespace:
                nonlocal upgrade_frame_calls
                del kwargs
                upgrade_frame_calls += 1
                Path(command[-1].replace("%04d", "0001")).write_bytes(
                    b"private-frame-after-upgrade"
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr([7.51]),
                )

            def upgrade_audio_runner(
                command: list[str], **kwargs
            ) -> SimpleNamespace:
                nonlocal upgrade_audio_calls
                del kwargs
                upgrade_audio_calls += 1
                Path(command[-1]).write_bytes(b"\0\0" * (15 * 16_000))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            upgraded = extract_teachobs_multimodal_features(
                plan,
                media_result["manifest"],
                output,
                acknowledge_source_terms=True,
                frame_runner=upgrade_frame_runner,
                audio_runner=upgrade_audio_runner,
                image_metric_extractor=metrics,
                ocr_extractor=ocr,
                ffmpeg_command=sys.executable,
            )["manifest"]
            record = upgraded["lessons"][0]
            self.assertEqual(upgrade_frame_calls, 1)
            self.assertEqual(upgrade_audio_calls, 1)
            self.assertFalse(record["frames_reused"])
            self.assertFalse(record["visual_evidence_reused"])
            self.assertTrue(record["visual_evidence_replaced_obsolete_binding"])
            self.assertFalse(record["audio_features_reused"])
            self.assertTrue(record["audio_features_upgraded_legacy_cache"])
            self.assertTrue(
                json.loads(task_path.read_text())["frames"][0][
                    "timestamp_verified_within_tolerance"
                ]
            )
            upgraded_audio = json.loads(audio_path.read_text())
            self.assertEqual(
                upgraded_audio["cache_upgrade_from_algorithm"],
                "teaching_skill_miner.scene_pcm_statistics.v1",
            )
            self.assertEqual(
                json.loads(evidence_path.read_text())["task_file_sha256"],
                _file_sha256(task_path),
            )

    def test_parallel_feature_failure_retains_success_and_resume_reuses_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            output = root / "private-media"
            plan = build_teachobs_media_plan(
                repository, lesson_ids=["S2", "S4"]
            )

            def fake_download(url: str, staging: Path, **kwargs) -> dict:
                del url, kwargs
                media = staging / "media.mp4"
                media.write_bytes(b"private-complete-video")
                return {"download_path": str(media), "partial_resume_enabled": True}

            def fake_probe(path: Path, **kwargs) -> dict:
                del kwargs
                return _valid_probe(path.stat().st_size)

            media_result = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=fake_download,
                media_probe=fake_probe,
                jobs=2,
            )
            first_frame_calls: list[str] = []
            first_audio_calls: list[str] = []

            def first_frame_runner(command: list[str], **kwargs):
                del kwargs
                staging_name = Path(command[-1]).parent.name
                first_frame_calls.append(staging_name)
                if staging_name.startswith(".S4-"):
                    return SimpleNamespace(
                        returncode=2, stdout="", stderr="fixture frame failure"
                    )
                Path(command[-1].replace("%04d", "0001")).write_bytes(
                    b"jpeg-frame"
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr([7.5]),
                )

            def first_audio_runner(command: list[str], **kwargs):
                del kwargs
                first_audio_calls.append(Path(command[-1]).name)
                Path(command[-1]).write_bytes(b"\0\0" * (15 * 16_000))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            with self.assertRaisesRegex(
                RuntimeError, "successful independent artifacts were retained"
            ):
                extract_teachobs_multimodal_features(
                    plan,
                    media_result["manifest"],
                    output,
                    acknowledge_source_terms=True,
                    include_visual_evidence=False,
                    feature_jobs=2,
                    frame_runner=first_frame_runner,
                    audio_runner=first_audio_runner,
                    ffmpeg_command=sys.executable,
                    generated_at_utc="2026-07-23T00:00:00Z",
                )
            partial = json.loads((output / "feature_manifest.json").read_text())
            self.assertFalse(partial["complete"])
            self.assertEqual(partial["lesson_count"], 1)
            self.assertEqual(partial["failed_lesson_count"], 1)
            self.assertEqual(partial["failed_lesson_ids"], ["S4"])
            self.assertEqual(partial["failure_records_path"], "feature_failures.json")
            failure_path = output / partial["failure_records_path"]
            self.assertEqual(partial["failure_records_sha256"], _file_sha256(failure_path))
            failure_text = failure_path.read_text(encoding="utf-8")
            self.assertNotIn("fixture frame failure", failure_text)
            failures = json.loads(failure_text)
            self.assertEqual(failures["schema"], FEATURE_FAILURE_SCHEMA)
            self.assertEqual(failures["failure_count"], 1)
            self.assertEqual(
                failures["failures"],
                [
                    {
                        "lesson_id": "S4",
                        "stage": "frame_extraction",
                        "reason_code": "private_frame_extraction_failed",
                        "safe_reason": (
                            "Private scene-frame extraction did not complete."
                        ),
                        "exception_type": "RuntimeError",
                        "raw_exception_message_persisted": False,
                    }
                ],
            )
            self.assertFalse(failures["privacy"]["contains_exception_messages"])
            if os.name == "posix":
                self.assertEqual(stat.S_IMODE(failure_path.stat().st_mode), 0o600)
            self.assertEqual(partial["lessons"][0]["lesson_id"], "S2")
            self.assertTrue((output / "frames" / "S2" / "task.json").is_file())
            self.assertTrue((output / "features" / "audio" / "S2.json").is_file())
            self.assertEqual(len(first_frame_calls), 2)
            self.assertEqual(len(first_audio_calls), 1)

            resumed_frame_calls: list[str] = []
            resumed_audio_calls: list[str] = []

            def resumed_frame_runner(command: list[str], **kwargs):
                del kwargs
                staging_name = Path(command[-1]).parent.name
                resumed_frame_calls.append(staging_name)
                if staging_name.startswith(".S2-"):
                    self.fail("completed S2 frame artifact was not reused")
                Path(command[-1].replace("%04d", "0001")).write_bytes(
                    b"jpeg-frame"
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr([7.5]),
                )

            def resumed_audio_runner(command: list[str], **kwargs):
                del kwargs
                resumed_audio_calls.append(Path(command[-1]).name)
                Path(command[-1]).write_bytes(b"\0\0" * (15 * 16_000))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            resumed = extract_teachobs_multimodal_features(
                plan,
                media_result["manifest"],
                output,
                acknowledge_source_terms=True,
                include_visual_evidence=False,
                feature_jobs=2,
                frame_runner=resumed_frame_runner,
                audio_runner=resumed_audio_runner,
                ffmpeg_command=sys.executable,
                generated_at_utc="2026-07-23T00:00:00Z",
            )["manifest"]
            self.assertTrue(resumed["complete"])
            self.assertEqual(resumed["lesson_count"], 2)
            self.assertEqual(resumed["failed_lesson_count"], 0)
            self.assertEqual(resumed["failed_lesson_ids"], [])
            self.assertIsNone(resumed["failure_records_path"])
            self.assertIsNone(resumed["failure_records_sha256"])
            self.assertFalse((output / "feature_failures.json").exists())
            self.assertEqual([row["lesson_id"] for row in resumed["lessons"]], ["S2", "S4"])
            s2 = next(row for row in resumed["lessons"] if row["lesson_id"] == "S2")
            self.assertTrue(s2["frames_reused"])
            self.assertTrue(s2["audio_features_reused"])
            self.assertEqual(len(resumed_frame_calls), 1)
            self.assertTrue(resumed_frame_calls[0].startswith(".S4-"))
            self.assertEqual(len(resumed_audio_calls), 1)

    def test_partial_clip_with_s4_hole_is_id_mapped_cached_and_replaced_by_full(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository = _repository_fixture(root)
            output = root / "private-media"
            model = root / "private-clip-model"
            model.mkdir()
            plan = build_teachobs_media_plan(
                repository, lesson_ids=["S2", "S4", "S5"]
            )

            def fake_download(url: str, staging: Path, **kwargs) -> dict:
                del kwargs
                media = staging / "media.mp4"
                media.write_bytes(("private-complete-video:" + url).encode())
                return {"download_path": str(media), "partial_resume_enabled": True}

            full_media = download_teachobs_media(
                plan,
                output,
                acknowledge_source_terms=True,
                downloader=fake_download,
                media_probe=lambda path, **kwargs: _valid_probe(
                    path.stat().st_size
                ),
            )["manifest"]
            partial_media = json.loads(json.dumps(full_media))
            partial_media["lessons"] = [
                row for row in full_media["lessons"] if row["lesson_id"] != "S4"
            ]
            partial_media["downloaded_lesson_count_total"] = 2
            partial_media["selected_complete"] = False

            def frame_runner(command: list[str], **kwargs):
                del kwargs
                Path(command[-1].replace("%04d", "0001")).write_bytes(
                    b"jpeg-frame"
                )
                return SimpleNamespace(
                    returncode=0,
                    stdout="",
                    stderr=_showinfo_stderr([7.5]),
                )

            def audio_runner(command: list[str], **kwargs):
                del kwargs
                Path(command[-1]).write_bytes(b"\0\0" * (15 * 16_000))
                return SimpleNamespace(returncode=0, stdout="", stderr="")

            clip_calls: list[Path] = []

            def fake_clip_run(arguments) -> dict:
                task_path = Path(arguments.tasks)
                clip_calls.append(task_path)
                task = json.loads(task_path.read_text(encoding="utf-8"))
                weight_files = [
                    {
                        "path": "model.safetensors",
                        "size_bytes": 1,
                        "sha256": "a" * 64,
                    }
                ]
                result_frames = []
                for frame in task["frames"]:
                    embedding = [1.0, 0.0]
                    result_frames.append(
                        {
                            "frame_id": frame["frame_id"],
                            "timestamp": frame["timestamp"],
                            "path": frame["path"],
                            "sha256": frame["sha256"],
                            "embedding": embedding,
                            "embedding_sha256": _canonical_sha256(embedding),
                        }
                    )
                return {
                    "schema": (
                        "teaching_skill_miner.visual_semantic_results.v1"
                    ),
                    "video_id": task["video_id"],
                    "media_sha256": task["media_sha256"],
                    "task_manifest_sha256": _canonical_sha256(task),
                    "frame_count": len(result_frames),
                    "model_provenance": {
                        "loaded_model_path": str(model.resolve()),
                        "requested_revision": arguments.source_revision,
                        "weight_manifest": {
                            "files": weight_files,
                            "manifest_sha256": _canonical_sha256(weight_files),
                        },
                        "device": arguments.device,
                        "batch_size": arguments.batch_size,
                    },
                    "privacy": {
                        "face_recognition_performed": False,
                        "identity_recognition_performed": False,
                    },
                    "frames": result_frames,
                    "claim_boundary": {
                        "recognition_accuracy_established": False,
                    },
                }

            ready = {
                "requested": True,
                "ready": True,
                "model_path": str(model.resolve()),
                "weight_file_count": 1,
                "device": "cpu",
            }
            clip_arguments = {
                "clip_model": model,
                "clip_source_revision": "fixture-pinned-revision",
                "clip_device": "cpu",
                "clip_batch_size": 2,
            }
            with (
                patch(
                    "teaching_skill_miner.teachobs_media.clip_preflight",
                    return_value=ready,
                ),
                patch(
                    "teaching_skill_miner.visual_semantics.run_inference",
                    side_effect=fake_clip_run,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "successful independent artifacts were retained"
                ):
                    extract_teachobs_multimodal_features(
                        plan,
                        partial_media,
                        output,
                        acknowledge_source_terms=True,
                        include_visual_evidence=False,
                        frame_runner=frame_runner,
                        audio_runner=audio_runner,
                        ffmpeg_command=sys.executable,
                        generated_at_utc="2026-07-23T00:00:00Z",
                        **clip_arguments,
                    )
                partial = json.loads(
                    (output / "feature_manifest.json").read_text(encoding="utf-8")
                )
                self.assertFalse(partial["complete"])
                self.assertTrue(partial["clip_embeddings_included"])
                self.assertFalse(
                    partial["clip_embeddings_complete_for_all_lessons"]
                )
                self.assertFalse(partial["combined_clip_complete"])
                self.assertEqual(partial["failed_lesson_ids"], ["S4"])
                self.assertEqual(partial["combined_clip_failed_lesson_ids"], ["S4"])
                self.assertEqual(partial["combined_clip_lesson_ids"], ["S2", "S5"])
                failure_path = output / partial["failure_records_path"]
                self.assertTrue(failure_path.is_file())
                self.assertEqual(
                    partial["failure_records_sha256"], _file_sha256(failure_path)
                )
                failure = json.loads(failure_path.read_text(encoding="utf-8"))
                self.assertEqual(failure["failures"][0]["stage"], "media_binding")

                partial_task_path = output / partial["combined_clip_task_path"]
                partial_result_path = output / partial["combined_clip_result_path"]
                partial_task = json.loads(
                    partial_task_path.read_text(encoding="utf-8")
                )
                self.assertFalse(partial_task["complete"])
                self.assertEqual(partial_task["included_lesson_ids"], ["S2", "S5"])
                self.assertEqual(partial_task["failed_lesson_ids"], ["S4"])
                self.assertEqual(
                    [row["lesson_id"] for row in partial_task["media_set"]],
                    ["S2", "S5"],
                )
                self.assertEqual(
                    [row["path"].split("/", 1)[0] for row in partial_task["frames"]],
                    ["S2", "S5"],
                )
                self.assertEqual(len(clip_calls), 1)
                if os.name == "posix":
                    self.assertEqual(
                        stat.S_IMODE(partial_task_path.stat().st_mode), 0o600
                    )
                    self.assertEqual(
                        stat.S_IMODE(partial_result_path.stat().st_mode), 0o600
                    )

                with self.assertRaisesRegex(
                    RuntimeError, "successful independent artifacts were retained"
                ):
                    extract_teachobs_multimodal_features(
                        plan,
                        partial_media,
                        output,
                        acknowledge_source_terms=True,
                        include_visual_evidence=False,
                        frame_runner=lambda *args, **kwargs: self.fail(
                            "cached frames were not reused"
                        ),
                        audio_runner=lambda *args, **kwargs: self.fail(
                            "cached audio was not reused"
                        ),
                        ffmpeg_command=sys.executable,
                        generated_at_utc="2026-07-23T00:00:00Z",
                        **clip_arguments,
                    )
                cached = json.loads(
                    (output / "feature_manifest.json").read_text(encoding="utf-8")
                )
                self.assertTrue(cached["combined_clip_reused"])
                self.assertEqual(len(clip_calls), 1)

                untampered_result = partial_result_path.read_bytes()
                partial_result_path.write_bytes(untampered_result + b" ")
                with self.assertRaisesRegex(ValueError, "result hash differs"):
                    extract_teachobs_multimodal_features(
                        plan,
                        partial_media,
                        output,
                        acknowledge_source_terms=True,
                        include_visual_evidence=False,
                        frame_runner=lambda *args, **kwargs: self.fail(
                            "cached frames were not reused"
                        ),
                        audio_runner=lambda *args, **kwargs: self.fail(
                            "cached audio was not reused"
                        ),
                        ffmpeg_command=sys.executable,
                        generated_at_utc="2026-07-23T00:00:00Z",
                        **clip_arguments,
                    )
                after_tamper = json.loads(
                    (output / "feature_manifest.json").read_text(encoding="utf-8")
                )
                self.assertEqual(after_tamper["failed_lesson_ids"], ["S4"])
                self.assertEqual(
                    after_tamper["failure_records_sha256"],
                    _file_sha256(output / after_tamper["failure_records_path"]),
                )
                partial_result_path.write_bytes(untampered_result)

                full = extract_teachobs_multimodal_features(
                    plan,
                    full_media,
                    output,
                    acknowledge_source_terms=True,
                    include_visual_evidence=False,
                    frame_runner=frame_runner,
                    audio_runner=audio_runner,
                    ffmpeg_command=sys.executable,
                    generated_at_utc="2026-07-23T00:00:00Z",
                    **clip_arguments,
                )["manifest"]
            self.assertTrue(full["complete"])
            self.assertTrue(full["combined_clip_complete"])
            self.assertTrue(full["clip_embeddings_complete_for_all_lessons"])
            self.assertEqual(full["combined_clip_lesson_ids"], ["S2", "S4", "S5"])
            self.assertEqual(full["combined_clip_failed_lesson_ids"], [])
            self.assertEqual(full["failed_lesson_ids"], [])
            self.assertFalse((output / "feature_failures.json").exists())
            full_task_path = output / full["combined_clip_task_path"]
            full_result_path = output / full["combined_clip_result_path"]
            self.assertNotEqual(full_task_path, partial_task_path)
            self.assertNotEqual(full_result_path, partial_result_path)
            self.assertTrue(partial_task_path.is_file())
            self.assertTrue(partial_result_path.is_file())
            self.assertTrue(full_task_path.is_file())
            self.assertTrue(full_result_path.is_file())
            self.assertEqual(len(clip_calls), 2)

    def test_private_ocr_image_metrics_and_adjacent_events(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame_text = {
                "frame_0001.jpg": "x = alpha beta gamma",
                "frame_0002.jpg": "completely different slide content",
                "frame_0003.jpg": (
                    "completely different slide content plus one two three four"
                ),
            }
            dhashes = {
                "frame_0001.jpg": "0000000000000000",
                "frame_0002.jpg": "ffffffffffffffff",
                "frame_0003.jpg": "fffffffffffffffe",
            }
            frames = []
            for index, name in enumerate(frame_text, 1):
                path = root / name
                path.write_bytes(f"private-frame-{index}".encode())
                frames.append(
                    {
                        "frame_id": f"S2_scene_{index:04d}",
                        "scene_no": index,
                        "timestamp": 7.5 + (index - 1) * 15,
                        "path": name,
                        "sha256": _file_sha256(path),
                    }
                )
            task_path = root / "task.json"
            task_path.write_text(
                json.dumps(
                    {
                        "schema": "teaching_skill_miner.visual_semantic_tasks.v1",
                        "video_id": "S2",
                        "media_sha256": "a" * 64,
                        "scene_manifest_sha256": "b" * 64,
                        "frame_count": 3,
                        "frames": frames,
                    }
                ),
                encoding="utf-8",
            )

            def metrics(path: Path) -> dict:
                return {
                    "backend": "fixture",
                    "dhash64": dhashes[path.name],
                    "edge_difference_mean": float(int(path.stem[-1])),
                }

            def ocr(path: Path, language: str, **kwargs):
                del kwargs
                return frame_text[path.name], {
                    "status": "completed",
                    "language": language,
                    "accepted_word_count": len(frame_text[path.name].split()),
                    "confidence_is_calibrated_probability": False,
                }

            output = root / "private-visual-evidence.json"
            result = extract_teachobs_scene_visual_evidence(
                task_path,
                output,
                visual_jobs=2,
                ocr_jobs=2,
                image_metric_extractor=metrics,
                ocr_extractor=ocr,
            )["result"]
            self.assertEqual(result["scene_count"], 3)
            self.assertEqual(result["transition_count"], 2)
            event_types = {event["type"] for event in result["events"]}
            self.assertIn("scene_change", event_types)
            self.assertIn("slide_change", event_types)
            self.assertIn("board_build_up", event_types)
            self.assertIn("code_or_formula_visible", event_types)
            self.assertEqual(result["schema"], VISUAL_EVIDENCE_SCHEMA)
            self.assertEqual(
                set(result["event_type_counts"]),
                set(VISUAL_EVIDENCE_EVENT_TYPES),
            )
            self.assertEqual(
                result["event_type_counts"]["code_or_formula_visible"],
                sum(
                    event["type"] == "code_or_formula_visible"
                    for event in result["events"]
                ),
            )
            self.assertEqual(
                sum(result["event_type_counts"].values()),
                result["event_count"],
            )
            self.assertEqual(result["ocr_status_counts"], {"completed": 3})
            self.assertTrue(result["privacy"]["contains_ocr_text"])
            self.assertFalse(result["privacy"]["safe_to_publish"])
            self.assertFalse(result["claim_boundary"]["event_accuracy_established"])

            legacy = json.loads(output.read_text(encoding="utf-8"))
            legacy["schema"] = (
                "teaching_skill_miner."
                "teachobs_private_scene_visual_evidence.v1"
            )
            legacy["configuration"].pop("event_type_counting_policy")
            legacy["configuration_sha256"] = _canonical_sha256(
                legacy["configuration"]
            )
            legacy["event_type_counts"].pop("code_or_formula_visible")
            output.write_text(json.dumps(legacy), encoding="utf-8")
            upgraded = extract_teachobs_scene_visual_evidence(
                task_path,
                output,
                image_metric_extractor=lambda path: self.fail(
                    f"validated legacy metadata upgrade re-extracted {path}"
                ),
                ocr_extractor=lambda *args, **kwargs: self.fail(
                    "validated legacy metadata upgrade reran OCR"
                ),
            )
            self.assertFalse(upgraded["reused"])
            self.assertTrue(upgraded["replaced_obsolete_binding"])
            self.assertTrue(upgraded["upgraded_legacy_event_counts"])
            self.assertEqual(upgraded["result"]["schema"], VISUAL_EVIDENCE_SCHEMA)
            self.assertTrue(upgraded["result"]["cache_upgrade"]["metadata_only"])
            self.assertEqual(
                upgraded["result"]["configuration"][
                    "event_type_counting_policy"
                ],
                VISUAL_EVIDENCE_EVENT_COUNTING_POLICY,
            )
            self.assertEqual(
                sum(upgraded["result"]["event_type_counts"].values()),
                upgraded["result"]["event_count"],
            )

            tampered = json.loads(output.read_text(encoding="utf-8"))
            tampered["event_type_counts"]["code_or_formula_visible"] = 0
            output.write_text(json.dumps(tampered), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "event type counts differ"):
                extract_teachobs_scene_visual_evidence(
                    task_path,
                    output,
                    image_metric_extractor=lambda path: self.fail(
                        f"tampered cache recomputed: {path}"
                    ),
                    ocr_extractor=lambda *args, **kwargs: self.fail(
                        "tampered cache recomputed"
                    ),
                )

            with patch.dict(os.environ, {"TSM_TESSERACT": "missing-tesseract"}):
                unavailable = extract_teachobs_scene_visual_evidence(
                    task_path,
                    root / "ocr-unavailable.json",
                    image_metric_extractor=metrics,
                )["result"]
            self.assertEqual(unavailable["ocr_status_counts"], {"unavailable": 3})
            self.assertFalse(
                unavailable["claim_boundary"]["ocr_complete_for_every_scene"]
            )

    def test_clip_preflight_fails_explicitly_for_missing_model(self) -> None:
        status = clip_preflight("/definitely/missing/clip-model", device="cpu")
        self.assertTrue(status["requested"])
        self.assertFalse(status["ready"])
        self.assertIn("does not exist", status["reason"])


if __name__ == "__main__":
    unittest.main()
