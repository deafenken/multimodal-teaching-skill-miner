from __future__ import annotations

import csv
from hashlib import sha256
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from teaching_skill_miner.cli import main
from teaching_skill_miner.io_utils import write_json
from teaching_skill_miner.teachobs_captions import (
    align_caption_to_released_scenes,
    build_public_teachobs_caption_receipt,
    normalized_character_overlap,
    normalized_token_overlap,
    repetition_diagnostics,
    retrieve_and_audit_teachobs_captions,
)
from teaching_skill_miner.teachobs_media import (
    EXPECTED_TEST_IDS,
    PINNED_REPOSITORY_COMMIT,
    build_teachobs_media_plan,
)


def _file_sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


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


def _repository_fixture(root: Path) -> tuple[Path, Path, Path]:
    repository = root / "repository"
    scenes_root = repository / "data" / "scenes"
    scenes_root.mkdir(parents=True)
    (repository / "README.md").write_text(
        "Source-reported: lesson captions were used for scene transcripts.\n",
        encoding="utf-8",
    )
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
    lesson_rows = []
    for index in range(1, 31):
        lesson_id = f"S{index}"
        lesson_rows.append(
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
        text = {
            "S1": "hello class we solve an equation",
            "S2": "today we inspect the heart diagram",
            "S3": "caption fallback must remain pending",
        }.get(lesson_id, "ordinary fixture transcript")
        (lesson_root / transcript_name).write_text(text, encoding="utf-8")
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
        writer.writerows(lesson_rows)

    tree_sha256, file_count, byte_count = _tree_digest(repository)
    receipt_path = root / "acquisition_receipt.json"
    write_json(
        receipt_path,
        {
            "repository_identification": {
                "fixed_commit": PINNED_REPOSITORY_COMMIT,
                "identification_status": "test_fixture_fixed_commit",
            },
            "extracted_repository": {
                "sha256_manifest_v1": tree_sha256,
                "files": file_count,
                "bytes": byte_count,
            },
            "licenses": {"video_redistribution_authorized_by_dataset": False},
        },
    )
    plan_path = root / "media_plan.json"
    write_json(
        plan_path,
        build_teachobs_media_plan(
            repository, acquisition_receipt_path=receipt_path
        ),
    )
    return repository, receipt_path, plan_path


def _vtt(text: str) -> str:
    return f"WEBVTT\n\n00:00:00.000 --> 00:00:14.500\n{text}\n"


class _FakeYtDlp:
    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        self.commands.append(command)
        url = command[-1]
        if "--dump-single-json" in command:
            if "testVideo01" in url:
                metadata = {
                    "id": "privateVideoOne",
                    "language": "en",
                    "subtitles": {"en": [{"ext": "vtt"}]},
                    "automatic_captions": {"en": [{"ext": "vtt"}]},
                }
            elif "testVideo02" in url:
                metadata = {
                    "id": "privateVideoTwo",
                    "language": "ja",
                    "subtitles": {"en": [{"ext": "vtt"}]},
                    "automatic_captions": {"ja": [{"ext": "vtt"}]},
                }
            else:
                metadata = {
                    "id": "privateVideoNoCaption",
                    "language": "en",
                    "subtitles": {},
                    "automatic_captions": {},
                }
            return SimpleNamespace(
                returncode=0, stdout=json.dumps(metadata), stderr=""
            )

        staging = Path(command[command.index("--paths") + 1])
        template = command[command.index("--output") + 1]
        lesson_id = template.split(".", 1)[0]
        if lesson_id == "S1":
            (staging / "S1.en.vtt").write_text(
                _vtt("hello class we solve an equation"), encoding="utf-8"
            )
        elif lesson_id == "S2":
            (staging / "S2.ja.vtt").write_text(
                _vtt("心臓の図を見ます"), encoding="utf-8"
            )
            (staging / "S2.en.vtt").write_text(
                _vtt("today we inspect the heart diagram"), encoding="utf-8"
            )
        return SimpleNamespace(returncode=0, stdout="", stderr="")


class _FailingPrivateYtDlp(_FakeYtDlp):
    def __call__(self, command: list[str], **kwargs: object) -> SimpleNamespace:
        if "--dump-single-json" in command:
            return super().__call__(command, **kwargs)
        del kwargs
        self.commands.append(command)
        return SimpleNamespace(
            returncode=1,
            stdout="",
            stderr=(
                "HTTP 429 for https://proxy-secret.invalid/caption "
                "account=private-user@example.invalid "
                "profile=Highly Secret Profile /" + "Users/private/browser"
            ),
        )


class TeachObsCaptionAuditTests(unittest.TestCase):
    def test_cli_primary_and_fetch_alias_have_side_effect_free_help(self) -> None:
        for command in ("audit-teachobs-captions", "fetch-teachobs-captions"):
            with self.subTest(command=command), self.assertRaises(SystemExit) as raised:
                main([command, "--help"])
            self.assertEqual(raised.exception.code, 0)

    def test_overlap_and_repetition_are_bounded_diagnostics_not_wer(self) -> None:
        self.assertEqual(normalized_token_overlap("Hello, WORLD", "hello world"), 1.0)
        self.assertEqual(
            normalized_character_overlap("课 堂", "课堂"),
            1.0,
        )
        repetition = repetition_diagnostics(("subscribe now " * 80).strip())
        self.assertTrue(repetition["abnormal_repetition_flag"])
        self.assertIn(
            "low_lexical_diversity_and_high_compressibility",
            repetition["reasons"],
        )

        aligned = align_caption_to_released_scenes(
            [
                {
                    "scene_no": 1,
                    "start": 0.0,
                    "end": 15.0,
                    "text": "hello class",
                }
            ],
            [{"start": 0.0, "end": 14.0, "text": "hello class"}],
        )
        self.assertEqual(
            aligned["normalized_token_overlap_mean_on_paired_nonempty"], 1.0
        )
        self.assertFalse(aligned["word_error_rate_established"])
        self.assertFalse(aligned["content_accuracy_established"])

    def test_retrieves_manual_original_and_auto_original_without_overclaiming(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, receipt_path, plan_path = _repository_fixture(root)
            runner = _FakeYtDlp()
            result = retrieve_and_audit_teachobs_captions(
                repository,
                plan_path,
                root / "private-captions",
                acknowledge_source_terms=True,
                acquisition_receipt_path=receipt_path,
                lesson_ids=["S1", "S2"],
                yt_dlp_command=os.sys.executable,
                max_workers=2,
                runner=runner,
                retrieved_at_utc="2026-07-23T00:00:00Z",
            )
            audit = result["audit"]
            self.assertEqual(audit["aggregate"]["lesson_count"], 2)
            self.assertEqual(
                audit["aggregate"]["caption_timeline_audited_lesson_count"], 2
            )
            self.assertTrue(
                audit["claims"]["formal_caption_timeline_audit_completed"]
            )
            self.assertFalse(audit["claims"]["caption_content_accuracy_established"])
            self.assertFalse(audit["claims"]["word_error_rate_established"])
            self.assertEqual(
                audit["source_reported_provenance"]["status"],
                "source_reported_not_independently_verified",
            )
            track_types = {
                track["track_type"]
                for record in audit["records"]
                for track in record["tracks"]
            }
            self.assertEqual(
                track_types,
                {"manual_creator_provided", "youtube_automatic_caption"},
            )
            download_commands = [
                command for command in runner.commands if "--write-subs" in command
            ]
            self.assertEqual(len(download_commands), 2)
            for command in download_commands:
                self.assertIn("--skip-download", command)
                self.assertIn("--write-subs", command)
                self.assertIn("--write-auto-subs", command)
            for command in runner.commands:
                self.assertNotIn("--proxy", command)
                self.assertNotIn("--impersonate", command)
                self.assertNotIn("--cookies-from-browser", command)
                self.assertNotIn("--extractor-args", command)

            output = root / "private-captions"
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            self.assertEqual((output / "raw" / "S1" / "en.vtt").stat().st_mode & 0o777, 0o600)
            receipt = build_public_teachobs_caption_receipt(
                audit,
                private_audit_path=result["private_audit_path"],
            )
            serialized = json.dumps(receipt, ensure_ascii=False)
            self.assertNotIn("hello class", serialized)
            self.assertNotIn("testVideo", serialized)
            self.assertNotIn("https://", serialized)
            self.assertNotIn(str(output), serialized)
            self.assertNotIn('"S1"', serialized)
            self.assertFalse(receipt["content_exclusion"]["caption_text_included"])
            self.assertFalse(
                receipt["evidence_status"]["caption_content_accuracy_established"]
            )
            self.assertTrue(
                receipt["evidence_status"][
                    "formal_caption_timeline_audit_completed"
                ]
            )

            retry_runner = _FakeYtDlp()
            second = retrieve_and_audit_teachobs_captions(
                repository,
                plan_path,
                output,
                acknowledge_source_terms=True,
                acquisition_receipt_path=receipt_path,
                lesson_ids=["S1", "S2"],
                yt_dlp_command=os.sys.executable,
                max_workers=2,
                runner=retry_runner,
                retrieved_at_utc="2026-07-23T00:10:00Z",
            )
            self.assertEqual(retry_runner.commands, [])
            self.assertEqual(
                second["audit"]["aggregate"][
                    "reused_hash_verified_lesson_count"
                ],
                2,
            )

    def test_opt_in_transport_is_used_for_metadata_and_subtitles_but_not_persisted(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, receipt_path, plan_path = _repository_fixture(root)
            runner = _FakeYtDlp()
            result = retrieve_and_audit_teachobs_captions(
                repository,
                plan_path,
                root / "private-captions",
                acknowledge_source_terms=True,
                acquisition_receipt_path=receipt_path,
                lesson_ids=["S1"],
                yt_dlp_command=os.sys.executable,
                cookies_from_browser="chrome:Highly Secret Profile",
                yt_dlp_direct=True,
                yt_dlp_impersonate="chrome",
                yt_dlp_youtube_client="android_vr",
                max_workers=1,
                runner=runner,
            )
            self.assertEqual(len(runner.commands), 2)
            for command in runner.commands:
                self.assertEqual(
                    command[command.index("--cookies-from-browser") + 1],
                    "chrome:Highly Secret Profile",
                )
                self.assertEqual(command[command.index("--proxy") + 1], "")
                self.assertEqual(
                    command[command.index("--impersonate") + 1], "chrome"
                )
                self.assertEqual(
                    command[command.index("--extractor-args") + 1],
                    "youtube:player_client=android_vr",
                )
                self.assertNotIn("--cookies", command)

            audit = result["audit"]
            transport = audit["workflow"]["requested_retrieval_transport"]
            self.assertTrue(transport["direct_environment_proxy_bypass_enabled"])
            self.assertTrue(transport["credentials_used"])
            self.assertEqual(transport["browser_family"], "chrome")
            self.assertTrue(transport["http_impersonation_requested"])
            self.assertTrue(transport["youtube_player_client_requested"])
            self.assertEqual(transport["youtube_player_client"], "android_vr")
            self.assertFalse(transport["browser_profile_recorded"])
            self.assertFalse(transport["cookie_file_export_requested"])
            serialized_audit = json.dumps(audit, ensure_ascii=False)
            self.assertNotIn("Highly Secret Profile", serialized_audit)

            receipt = build_public_teachobs_caption_receipt(audit)
            serialized_receipt = json.dumps(receipt, ensure_ascii=False)
            self.assertNotIn("Highly Secret Profile", serialized_receipt)
            self.assertNotIn("private-user", serialized_receipt)
            self.assertFalse(receipt["content_exclusion"]["browser_profile_included"])
            self.assertFalse(receipt["content_exclusion"]["account_identifier_included"])
            self.assertFalse(receipt["content_exclusion"]["proxy_url_included"])
            self.assertFalse(receipt["content_exclusion"]["cookie_material_included"])

            injected = json.loads(json.dumps(audit))
            injected["aggregate"]["debug_browser_profile"] = (
                "Highly Secret Profile /" + "Users/private/browser"
            )
            injected.pop("audit_canonical_sha256")
            canonical = json.dumps(
                injected,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            injected["audit_canonical_sha256"] = sha256(canonical).hexdigest()
            allowlisted_receipt = build_public_teachobs_caption_receipt(injected)
            self.assertNotIn(
                "Highly Secret Profile",
                json.dumps(allowlisted_receipt, ensure_ascii=False),
            )

    def test_transport_validation_rejects_paths_targets_and_non_boolean_direct(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, receipt_path, plan_path = _repository_fixture(root)
            cases = (
                {"cookies_from_browser": "chrome:/" + "Users/private/Profile"},
                {"yt_dlp_impersonate": "safari"},
                {"yt_dlp_youtube_client": "web"},
                {"yt_dlp_direct": "true"},
            )
            for index, options in enumerate(cases):
                output = root / f"rejected-{index}"
                with self.subTest(options=options), self.assertRaises(ValueError):
                    retrieve_and_audit_teachobs_captions(
                        repository,
                        plan_path,
                        output,
                        acknowledge_source_terms=True,
                        acquisition_receipt_path=receipt_path,
                        lesson_ids=["S1"],
                        yt_dlp_command=os.sys.executable,
                        **options,
                    )
                self.assertFalse(output.exists())

    def test_failed_child_diagnostics_are_classified_without_secret_persistence(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, receipt_path, plan_path = _repository_fixture(root)
            result = retrieve_and_audit_teachobs_captions(
                repository,
                plan_path,
                root / "private-captions",
                acknowledge_source_terms=True,
                acquisition_receipt_path=receipt_path,
                lesson_ids=["S1"],
                yt_dlp_command=os.sys.executable,
                cookies_from_browser="chrome:Highly Secret Profile",
                yt_dlp_direct=True,
                yt_dlp_impersonate="chrome",
                yt_dlp_youtube_client="android_vr",
                max_workers=1,
                runner=_FailingPrivateYtDlp(),
            )
            record = result["audit"]["records"][0]
            self.assertEqual(
                record["status"], "caption_retrieval_failed_fallback_pending"
            )
            self.assertIn("failure_class=http_rate_limited", record["private_failure_detail"])
            serialized = json.dumps(result["audit"], ensure_ascii=False)
            for secret in (
                "proxy-secret.invalid",
                "private-user@example.invalid",
                "Highly Secret Profile",
                "/" + "Users/private/browser",
            ):
                self.assertNotIn(secret, serialized)

    def test_missing_caption_is_pending_and_never_runs_whisper(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, receipt_path, plan_path = _repository_fixture(root)
            runner = _FakeYtDlp()
            result = retrieve_and_audit_teachobs_captions(
                repository,
                plan_path,
                root / "private-captions",
                acknowledge_source_terms=True,
                acquisition_receipt_path=receipt_path,
                lesson_ids=["S3"],
                yt_dlp_command=os.sys.executable,
                max_workers=1,
                runner=runner,
            )
            record = result["audit"]["records"][0]
            self.assertEqual(record["status"], "caption_unavailable_fallback_pending")
            self.assertEqual(record["fallback"]["status"], "pending")
            self.assertFalse(record["whisper_model_downloaded"])
            self.assertFalse(record["whisper_executed"])
            self.assertEqual(len(runner.commands), 1)
            receipt = build_public_teachobs_caption_receipt(result["audit"])
            self.assertFalse(
                receipt["evidence_status"][
                    "formal_caption_timeline_audit_completed"
                ]
            )

    def test_terms_gate_and_plan_hash_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repository, receipt_path, plan_path = _repository_fixture(root)
            with self.assertRaisesRegex(ValueError, "explicit acknowledgement"):
                retrieve_and_audit_teachobs_captions(
                    repository,
                    plan_path,
                    root / "no-output",
                    acknowledge_source_terms=False,
                    acquisition_receipt_path=receipt_path,
                    yt_dlp_command=os.sys.executable,
                )
            self.assertFalse((root / "no-output").exists())

            plan = json.loads(plan_path.read_text(encoding="utf-8"))
            plan["lesson_count"] = 29
            write_json(plan_path, plan)
            with self.assertRaisesRegex(ValueError, "plan hash mismatch"):
                retrieve_and_audit_teachobs_captions(
                    repository,
                    plan_path,
                    root / "still-no-output",
                    acknowledge_source_terms=True,
                    acquisition_receipt_path=receipt_path,
                    lesson_ids=["S1"],
                    yt_dlp_command=os.sys.executable,
                )
            self.assertFalse((root / "still-no-output").exists())


if __name__ == "__main__":
    unittest.main()
