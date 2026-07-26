from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from teaching_skill_miner import teachobs_asr_handoff
from teaching_skill_miner.cli import main
from teaching_skill_miner.io_utils import write_json
from teaching_skill_miner.release_audit import audit_release_path
from teaching_skill_miner.teachobs_asr_handoff import (
    COVERAGE_POLICY_NAME,
    JOB_MANIFEST_SCHEMA,
    MEDIA_MANIFEST_SCHEMA,
    RESULT_SCHEMA,
    RUNNER_NAME,
    build_pending_teachobs_asr_receipt,
    build_public_teachobs_asr_receipt,
    build_teachobs_asr_job_manifest,
    import_teachobs_asr_results,
    model_directory_sha256,
    run_teachobs_asr_gpu_jobs,
    validate_teachobs_asr_job_manifest,
    write_teachobs_asr_import,
)
from teaching_skill_miner.teachobs_captions import PRIVATE_AUDIT_SCHEMA
from scripts.check_teachobs_external_study import (
    FULL_23_TRAIN_7_TEST_PROFILE,
    PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
    StageCheckError,
    check_asr,
)


def _canonical(value: object) -> str:
    return sha256(
        json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()


def _sha(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


class TeachObsAsrHandoffTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path, Path]:
        media_root = root / "media"
        videos = media_root / "videos"
        videos.mkdir(parents=True)
        lessons = []
        caption_records = []
        for index in range(1, 31):
            lesson_id = f"S{index}"
            media = videos / f"{lesson_id}.mp4"
            media.write_bytes((f"private-media-{lesson_id}" * 3).encode())
            lessons.append(
                {
                    "lesson_id": lesson_id,
                    "split": "test" if index <= 7 else "train",
                    "media_path": f"videos/{lesson_id}.mp4",
                    "media_size_bytes": media.stat().st_size,
                    "media_sha256": _sha(media),
                    "media_probe": {"duration_seconds": 15.0},
                }
            )
            if lesson_id == "S30":
                caption_records.append(
                    {
                        "lesson_id": lesson_id,
                        "status": "caption_retrieval_failed_fallback_pending",
                        "tracks": [],
                    }
                )
            else:
                caption_records.append(
                    {
                        "lesson_id": lesson_id,
                        "status": "caption_timeline_audited",
                        "tracks": [
                            {
                                "track_type": "manual_creator_provided",
                                "roles": ["original_language"],
                                "language": "en",
                                "caption_sha256": sha256(
                                    f"caption-{lesson_id}".encode()
                                ).hexdigest(),
                                "timeline": {
                                    "timeline_audit_completed": True,
                                    "timeline_span_coverage_fraction": 0.98,
                                },
                            }
                        ],
                    }
                )
        media_manifest = {
            "schema": MEDIA_MANIFEST_SCHEMA,
            "dataset_id": "teachobs_v0_1_human_validated",
            "repository_commit": "a" * 40,
            "selected_lesson_count": 30,
            "selected_complete": True,
            "lessons": lessons,
        }
        media_manifest_path = write_json(media_root / "media_manifest.json", media_manifest)
        caption_audit = {
            "schema": PRIVATE_AUDIT_SCHEMA,
            "dataset_id": "teachobs_v0_1_human_validated",
            "repository_commit": "a" * 40,
            "records": caption_records,
            "aggregate": {"lesson_count": 30},
        }
        caption_audit["audit_canonical_sha256"] = _canonical(caption_audit)
        caption_path = write_json(root / "caption_audit.json", caption_audit)
        return media_manifest_path, media_root, caption_path

    def _manifest(self, root: Path) -> tuple[Path, Path, Path, dict]:
        media_manifest, media_root, caption_audit = self._fixture(root)
        manifest = build_teachobs_asr_job_manifest(
            media_manifest,
            media_root,
            caption_audit,
            model_id="openai/whisper-large-v3-turbo",
            model_revision="b" * 40,
            model_files_sha256="c" * 64,
            faster_whisper_version="1.2.1",
            ctranslate2_version="4.6.0",
            container_image_digest="sha256:" + "e" * 64,
            duration_probe=lambda _: 15.0,
            generated_at_utc="2026-07-23T00:00:00Z",
        )
        manifest_path = write_json(root / "asr_job_manifest.json", manifest)
        return manifest_path, media_manifest, caption_audit, manifest

    def _result(self, job: dict, manifest: dict) -> dict:
        segments = [{"start": 0.0, "end": 15.0, "text": "private fixture words"}]
        coverage = {
            "segment_count": 1,
            "first_segment_start_seconds": 0.0,
            "last_segment_end_seconds": 15.0,
            "timeline_span_seconds": 15.0,
            "timeline_span_fraction": 1.0,
            "speech_segment_union_seconds": 15.0,
            "speech_segment_union_fraction": 1.0,
            "initial_gap_seconds": 0.0,
            "trailing_gap_seconds": 0.0,
            "initial_gap_fraction": 0.0,
            "trailing_gap_fraction": 0.0,
            "endpoint_gap_policy_name": COVERAGE_POLICY_NAME,
            "absolute_endpoint_gap_seconds_gate_applied": False,
            "timeline_policy_passed": True,
            "content_accuracy_established": False,
            "word_error_rate_established": False,
        }
        result = {
            "schema": RESULT_SCHEMA,
            "job_manifest_sha256": manifest["manifest_sha256"],
            "job_sha256": job["job_sha256"],
            "job_id": job["job_id"],
            "lesson_id": job["lesson_id"],
            "transcript_kind": "automatic_speech_recognition",
            "source_tier": "audited_asr_fallback_candidate",
            "official_caption": False,
            "human_content_review_completed": False,
            "content_accuracy_established": False,
            "word_error_rate_established": False,
            "media_binding": job["media_binding"],
            "model": job["model"],
            "model_runtime": {
                "model_snapshot_sha256_v1": job["model"][
                    "model_snapshot_sha256_v1"
                ],
                "local_files_only": True,
                "network_access_enabled": False,
            },
            "decoding_config": job["decoding_config"],
            "segment_postprocessing": {
                "policy": (
                    "deterministic_greedy_positive_word_anchor_with_point_"
                    "anchored_zero_duration_nearest_attachment_v3"
                ),
                "maximum_output_segment_duration_seconds": 30.5,
                "source_segment_count": 1,
                "source_segments_split": 0,
                "output_segment_count": 1,
                "overlong_source_word_timestamp_count": 0,
                "positive_duration_word_anchor_count": 0,
                "attached_zero_duration_word_count": 0,
                "zero_duration_word_attachment_policy": (
                    "point_anchored_zero_duration_word_nearest_positive_attachment"
                ),
                "zero_duration_word_maximum_attachment_distance_seconds": 30.5,
                "maximum_observed_zero_duration_word_attachment_distance_seconds": (
                    0.0
                ),
                "zero_duration_point_anchors_within_source_segment_verified": True,
                "zero_duration_assignment_anchor_indices_monotonic_verified": True,
                "positive_word_boundaries_preserved": True,
                "split_text_trimmed_exact_equivalence_verified": True,
                "reference_or_label_used": False,
                "boundary_selection_uses_text_content": False,
            },
            "language": {
                "requested": "auto",
                "detected": "en",
                "detected_probability": 0.99,
            },
            "runtime": {
                "runner_name": RUNNER_NAME,
                "runner_source_sha256": job["runtime_contract"][
                    "runner_module_sha256"
                ],
                "python_version": "3.12.8",
                "package_versions": {
                    "faster-whisper": "1.2.1",
                    "ctranslate2": "4.6.0",
                },
                "accelerator": {
                    "device_type": "cuda",
                    "device_count": 1,
                    "device_name": "fixture-gpu",
                    "driver_version": "fixture-driver",
                },
                "container_image_digest": "sha256:" + "e" * 64,
                "executed_at_utc": "2026-07-23T00:10:00Z",
                "hostname_recorded": False,
            },
            "segments": segments,
            "timeline_coverage": coverage,
        }
        result["result_sha256"] = _canonical(result)
        return result

    def test_builds_one_fallback_job_with_content_free_hash_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, manifest = self._manifest(root)
            validation = validate_teachobs_asr_job_manifest(manifest)
            self.assertEqual(validation["job_count"], 1)
            self.assertEqual(manifest["jobs"][0]["lesson_id"], "S30")
            self.assertFalse(manifest["contains_media_bytes"])
            self.assertFalse(manifest["contains_transcript_text"])
            self.assertFalse(manifest["jobs"][0]["official_caption_claim_allowed"])
            self.assertEqual(
                manifest["schema"],
                "teaching_skill_miner.teachobs_asr_job_manifest.v2",
            )
            self.assertEqual(manifest["schema"], JOB_MANIFEST_SCHEMA)
            self.assertEqual(
                manifest["output_contract"]["schema"],
                "teaching_skill_miner.teachobs_asr_lesson_result.v4",
            )
            self.assertEqual(
                manifest["runtime_contract"]["runner_name"],
                "teaching_skill_miner.teachobs_asr_gpu_runner.v4",
            )
            self.assertEqual(
                manifest["coverage_policy"],
                {
                    "policy_name": COVERAGE_POLICY_NAME,
                    "input_scope": "hash_bound_full_media_single_pass",
                    "vad_filter_enabled": True,
                    "endpoint_timestamps_mean": (
                        "first_and_last_vad_retained_speech"
                    ),
                    "minimum_timeline_span_fraction": 0.9,
                    "maximum_endpoint_gap_fraction": 0.1,
                    "endpoint_gap_fraction_applied_symmetrically": True,
                    "endpoint_gap_fraction_denominator": (
                        "hash_bound_media_duration_seconds"
                    ),
                    "absolute_endpoint_gap_seconds_gate_enabled": False,
                    "interpretation": (
                        "technical full-media ASR/VAD timeline coverage only; "
                        "endpoint gaps are measured relative to media duration "
                        "because VAD emits speech anchors rather than silence; "
                        "not content accuracy or WER"
                    ),
                },
            )
            self.assertEqual(
                manifest["model"]["model_revision"],
                "b" * 40,
            )
            self.assertEqual(
                manifest["runtime_contract"]["container_image_digest"],
                "sha256:" + "e" * 64,
            )

    def test_full_media_vad_coverage_uses_symmetric_relative_endpoints(self) -> None:
        policy = {
            "policy_name": COVERAGE_POLICY_NAME,
            "minimum_timeline_span_fraction": 0.90,
            "maximum_endpoint_gap_fraction": 0.10,
        }
        s9_like = teachobs_asr_handoff._timeline_coverage(
            [
                {"start": 224.4, "end": 225.0, "text": "x"},
                {"start": 2954.0, "end": 2954.82, "text": "y"},
            ],
            duration=2974.61551,
            maximum_segment_duration=30.5,
            policy=policy,
        )
        self.assertTrue(s9_like["timeline_policy_passed"])
        self.assertEqual(s9_like["timeline_span_fraction"], 0.917907)
        self.assertEqual(s9_like["initial_gap_fraction"], 0.075438)
        self.assertEqual(s9_like["trailing_gap_fraction"], 0.006655)
        self.assertFalse(
            s9_like["absolute_endpoint_gap_seconds_gate_applied"]
        )

        s19_like = teachobs_asr_handoff._timeline_coverage(
            [
                {"start": 60.62, "end": 61.0, "text": "x"},
                {"start": 2781.0, "end": 2781.28, "text": "y"},
            ],
            duration=2822.013968,
            maximum_segment_duration=30.5,
            policy=policy,
        )
        self.assertTrue(s19_like["timeline_policy_passed"])
        self.assertEqual(s19_like["timeline_span_fraction"], 0.964085)
        self.assertEqual(s19_like["initial_gap_fraction"], 0.021481)
        self.assertEqual(s19_like["trailing_gap_fraction"], 0.014434)

        stricter_endpoint_policy = {
            **policy,
            "maximum_endpoint_gap_fraction": 0.05,
        }
        endpoint_failure = teachobs_asr_handoff._timeline_coverage(
            [
                {"start": 180.0, "end": 181.0, "text": "x"},
                {"start": 2999.0, "end": 3000.0, "text": "y"},
            ],
            duration=3000.0,
            maximum_segment_duration=30.5,
            policy=stricter_endpoint_policy,
        )
        self.assertGreaterEqual(
            endpoint_failure["timeline_span_fraction"],
            policy["minimum_timeline_span_fraction"],
        )
        self.assertFalse(endpoint_failure["timeline_policy_passed"])

    def test_relative_endpoint_contract_rejects_legacy_and_inconsistent_policy(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, _, _, manifest = self._manifest(root)
            self.assertNotIn(
                "maximum_initial_gap_seconds",
                manifest["coverage_policy"],
            )
            self.assertNotIn(
                "maximum_trailing_gap_seconds",
                manifest["coverage_policy"],
            )

            legacy = json.loads(json.dumps(manifest))
            legacy["schema"] = "teaching_skill_miner.teachobs_asr_job_manifest.v1"
            with self.assertRaisesRegex(
                ValueError, "unsupported TeachObs ASR job manifest"
            ):
                validate_teachobs_asr_job_manifest(legacy)

            legacy_result = self._result(manifest["jobs"][0], manifest)
            legacy_result["schema"] = (
                "teaching_skill_miner.teachobs_asr_lesson_result.v3"
            )
            with self.assertRaisesRegex(
                ValueError, "unsupported TeachObs ASR result schema"
            ):
                teachobs_asr_handoff.validate_teachobs_asr_result(
                    legacy_result,
                    manifest_sha256=manifest["manifest_sha256"],
                    job=manifest["jobs"][0],
                )

            forged_policy = json.loads(json.dumps(manifest))
            forged_policy["coverage_policy"][
                "absolute_endpoint_gap_seconds_gate_enabled"
            ] = True
            for job in forged_policy["jobs"]:
                job["coverage_policy"] = forged_policy["coverage_policy"]
                job["job_sha256"] = _canonical(
                    {
                        key: value
                        for key, value in job.items()
                        if key != "job_sha256"
                    }
                )
            forged_policy["manifest_sha256"] = _canonical(
                {
                    key: value
                    for key, value in forged_policy.items()
                    if key != "manifest_sha256"
                }
            )
            with self.assertRaisesRegex(
                ValueError, "fixed contract mismatch"
            ):
                validate_teachobs_asr_job_manifest(forged_policy)

            media_manifest, media_root, caption_audit = self._fixture(root / "bad")
            with self.assertRaisesRegex(
                ValueError,
                "no greater than one minus the minimum timeline-span",
            ):
                build_teachobs_asr_job_manifest(
                    media_manifest,
                    media_root,
                    caption_audit,
                    model_id="openai/whisper-large-v3-turbo",
                    model_revision="b" * 40,
                    model_files_sha256="c" * 64,
                    faster_whisper_version="1.2.1",
                    ctranslate2_version="4.6.0",
                    container_image_digest="sha256:" + "e" * 64,
                    min_timeline_span_fraction=0.90,
                    max_endpoint_gap_fraction=0.11,
                    duration_probe=lambda _: 15.0,
                )

    def test_prepare_cli_requires_frozen_container_only_for_nonpending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_manifest, _, caption_audit = self._fixture(root)
            pending_receipt = root / "pending.json"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "prepare-teachobs-asr-handoff",
                            "--media-manifest",
                            str(media_manifest),
                            "--caption-audit",
                            str(caption_audit),
                            "--public-receipt",
                            str(pending_receipt),
                            "--pending-only",
                        ]
                    ),
                    0,
                )
            self.assertTrue(pending_receipt.is_file())

            error = io.StringIO()
            with redirect_stderr(error):
                self.assertEqual(
                    main(
                        [
                            "prepare-teachobs-asr-handoff",
                            "--media-manifest",
                            str(media_manifest),
                            "--caption-audit",
                            str(caption_audit),
                            "--model-id",
                            "openai/whisper-large-v3-turbo",
                            "--model-revision",
                            "b" * 40,
                            "--model-files-sha256",
                            "c" * 64,
                            "--faster-whisper-version",
                            "1.2.1",
                            "--ctranslate2-version",
                            "4.6.0",
                        ]
                    ),
                    1,
                )
            self.assertIn("--container-image-digest", error.getvalue())

    def test_model_tree_hash_rejects_every_symlink_and_nonregular_node(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            baseline = model_directory_sha256(model)
            self.assertRegex(baseline, r"^[0-9a-f]{64}$")

            outside = root / "outside"
            outside.mkdir()
            directory_link = model / "directory-link"
            directory_link.symlink_to(outside, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "symlinks"):
                model_directory_sha256(model)
            directory_link.unlink()

            file_link = model / "file-link"
            file_link.symlink_to(model / "config.json")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                model_directory_sha256(model)
            file_link.unlink()

            broken_link = model / "broken-link"
            broken_link.symlink_to(root / "missing")
            with self.assertRaisesRegex(ValueError, "symlinks"):
                model_directory_sha256(model)
            broken_link.unlink()

            if hasattr(os, "mkfifo"):
                fifo = model / "nonregular"
                os.mkfifo(fifo)
                with self.assertRaisesRegex(ValueError, "regular files"):
                    model_directory_sha256(model)

    def test_import_is_pending_until_valid_gpu_result_then_completes_matrix(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, media_manifest, caption_audit, manifest = self._manifest(root)
            results = root / "results"
            results.mkdir()
            pending = import_teachobs_asr_results(
                manifest_path,
                media_manifest,
                root / "media",
                caption_audit,
                results,
                duration_probe=lambda _: 15.0,
                generated_at_utc="2026-07-23T00:20:00Z",
            )
            self.assertEqual(pending["audit"]["aggregate"]["pending_result_count"], 1)
            self.assertEqual(
                pending["coverage_matrix"]["aggregate"]["covered_lesson_count"],
                29,
            )

            write_json(results / "S30.json", self._result(manifest["jobs"][0], manifest))
            completed = import_teachobs_asr_results(
                manifest_path,
                media_manifest,
                root / "media",
                caption_audit,
                results,
                duration_probe=lambda _: 15.0,
                generated_at_utc="2026-07-23T00:30:00Z",
            )
            self.assertTrue(completed["audit"]["aggregate"]["asr_job_set_complete"])
            aggregate = completed["coverage_matrix"]["aggregate"]
            self.assertEqual(aggregate["audited_asr_fallback_lesson_count"], 1)
            self.assertTrue(aggregate["transcript_source_coverage_complete"])
            row = completed["coverage_matrix"]["rows"][-1]
            self.assertEqual(row["selected_source_tier"], "audited_asr_fallback")
            self.assertFalse(row["asr_is_official_caption"])

    def test_paper_profile_accepts_six_valid_asr_with_only_s4_pending(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media_manifest, media_root, caption_audit = self._fixture(root)

            media_value = json.loads(media_manifest.read_text(encoding="utf-8"))
            media_value["lessons"] = [
                row for row in media_value["lessons"] if row["lesson_id"] != "S4"
            ]
            media_value["selected_complete"] = False
            write_json(media_manifest, media_value)

            fallback_ids = {"S4", "S9", "S15", "S19", "S20", "S25", "S29"}
            caption_value = json.loads(caption_audit.read_text(encoding="utf-8"))
            for row in caption_value["records"]:
                if row["lesson_id"] in fallback_ids:
                    row["status"] = "caption_retrieval_failed_fallback_pending"
                    row["tracks"] = []
                elif row.get("status") != "caption_timeline_audited":
                    lesson_id = row["lesson_id"]
                    row["status"] = "caption_timeline_audited"
                    row["tracks"] = [
                        {
                            "track_type": "manual_creator_provided",
                            "roles": ["original_language"],
                            "language": "en",
                            "caption_sha256": sha256(
                                f"caption-{lesson_id}".encode()
                            ).hexdigest(),
                            "timeline": {
                                "timeline_audit_completed": True,
                                "timeline_span_coverage_fraction": 0.98,
                            },
                        }
                    ]
            unsigned_caption = {
                key: value
                for key, value in caption_value.items()
                if key != "audit_canonical_sha256"
            }
            caption_value["audit_canonical_sha256"] = _canonical(unsigned_caption)
            write_json(caption_audit, caption_value)

            manifest = build_teachobs_asr_job_manifest(
                media_manifest,
                media_root,
                caption_audit,
                model_id="openai/whisper-large-v3-turbo",
                model_revision="b" * 40,
                model_files_sha256="c" * 64,
                faster_whisper_version="1.2.1",
                ctranslate2_version="4.6.0",
                container_image_digest="sha256:" + "e" * 64,
                duration_probe=lambda _: 15.0,
                generated_at_utc="2026-07-23T00:00:00Z",
            )
            self.assertEqual(
                [job["lesson_id"] for job in manifest["jobs"]],
                ["S9", "S15", "S19", "S20", "S25", "S29"],
            )
            self.assertEqual(
                [row["lesson_id"] for row in manifest["media_pending"]],
                ["S4"],
            )
            manifest_path = write_json(root / "asr_job_manifest.json", manifest)
            results = root / "results"
            results.mkdir()
            for job in manifest["jobs"]:
                write_json(
                    results / f"{job['lesson_id']}.json",
                    self._result(job, manifest),
                )
            imported = import_teachobs_asr_results(
                manifest_path,
                media_manifest,
                media_root,
                caption_audit,
                results,
                duration_probe=lambda _: 15.0,
                generated_at_utc="2026-07-23T00:30:00Z",
            )
            outputs = write_teachobs_asr_import(
                imported,
                audit_path=root / "private" / "asr_audit.json",
                coverage_matrix_path=root / "private" / "coverage.json",
            )
            receipt = build_public_teachobs_asr_receipt(
                imported["audit"],
                imported["coverage_matrix"],
                asr_import_audit_path=outputs["audit"],
                coverage_matrix_path=outputs["coverage_matrix"],
                generated_at_utc="2026-07-23T00:40:00Z",
            )
            receipt_path = write_json(root / "public-asr.json", receipt)

            report = check_asr(
                str(receipt_path),
                str(media_manifest),
                str(caption_audit),
                job_manifest_path=str(manifest_path),
                import_audit_path=str(outputs["audit"]),
                coverage_matrix_path=str(outputs["coverage_matrix"]),
                evaluation_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
            )
            self.assertEqual(report["handoff_status"], "pending")
            self.assertEqual(report["covered_lesson_count"], 29)
            self.assertEqual(report["valid_asr_result_count"], 6)
            with self.assertRaisesRegex(
                StageCheckError, "pending lessons differ from the evaluation profile"
            ):
                check_asr(
                    str(receipt_path),
                    str(media_manifest),
                    str(caption_audit),
                    job_manifest_path=str(manifest_path),
                    import_audit_path=str(outputs["audit"]),
                    coverage_matrix_path=str(outputs["coverage_matrix"]),
                    evaluation_profile=FULL_23_TRAIN_7_TEST_PROFILE,
                )

    def test_rejects_forged_official_claim_runtime_and_media_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, media_manifest, caption_audit, manifest = self._manifest(root)
            results = root / "results"
            results.mkdir()
            result = self._result(manifest["jobs"][0], manifest)
            result["official_caption"] = True
            result["result_sha256"] = _canonical(
                {key: value for key, value in result.items() if key != "result_sha256"}
            )
            write_json(results / "S30.json", result)
            with self.assertRaisesRegex(ValueError, "invalid evidence claim"):
                import_teachobs_asr_results(
                    manifest_path,
                    media_manifest,
                    root / "media",
                    caption_audit,
                    results,
                    duration_probe=lambda _: 15.0,
                )

            result = self._result(manifest["jobs"][0], manifest)
            result["runtime"]["package_versions"]["faster-whisper"] = "9.9.9"
            result["result_sha256"] = _canonical(
                {key: value for key, value in result.items() if key != "result_sha256"}
            )
            write_json(results / "S30.json", result)
            with self.assertRaisesRegex(ValueError, "package versions"):
                import_teachobs_asr_results(
                    manifest_path,
                    media_manifest,
                    root / "media",
                    caption_audit,
                    results,
                    duration_probe=lambda _: 15.0,
                )

            result = self._result(manifest["jobs"][0], manifest)
            result["runtime"]["container_image_digest"] = "sha256:" + "d" * 64
            result["result_sha256"] = _canonical(
                {key: value for key, value in result.items() if key != "result_sha256"}
            )
            write_json(results / "S30.json", result)
            with self.assertRaisesRegex(ValueError, "container image differs"):
                import_teachobs_asr_results(
                    manifest_path,
                    media_manifest,
                    root / "media",
                    caption_audit,
                    results,
                    duration_probe=lambda _: 15.0,
                )

            result = self._result(manifest["jobs"][0], manifest)
            result["segment_postprocessing"]["output_segment_count"] = 2
            result["result_sha256"] = _canonical(
                {
                    key: value
                    for key, value in result.items()
                    if key != "result_sha256"
                }
            )
            write_json(results / "S30.json", result)
            with self.assertRaisesRegex(ValueError, "postprocessing"):
                import_teachobs_asr_results(
                    manifest_path,
                    media_manifest,
                    root / "media",
                    caption_audit,
                    results,
                    duration_probe=lambda _: 15.0,
                )

            result = self._result(manifest["jobs"][0], manifest)
            result["segment_postprocessing"][
                "maximum_observed_zero_duration_word_attachment_distance_seconds"
            ] = 30.500001
            result["result_sha256"] = _canonical(
                {
                    key: value
                    for key, value in result.items()
                    if key != "result_sha256"
                }
            )
            write_json(results / "S30.json", result)
            with self.assertRaisesRegex(ValueError, "postprocessing"):
                import_teachobs_asr_results(
                    manifest_path,
                    media_manifest,
                    root / "media",
                    caption_audit,
                    results,
                    duration_probe=lambda _: 15.0,
                )

            result = self._result(manifest["jobs"][0], manifest)
            result["segment_postprocessing"][
                "attached_zero_duration_word_count"
            ] = 1
            result["result_sha256"] = _canonical(
                {
                    key: value
                    for key, value in result.items()
                    if key != "result_sha256"
                }
            )
            write_json(results / "S30.json", result)
            with self.assertRaisesRegex(ValueError, "postprocessing"):
                import_teachobs_asr_results(
                    manifest_path,
                    media_manifest,
                    root / "media",
                    caption_audit,
                    results,
                    duration_probe=lambda _: 15.0,
                )

            (root / "media" / "videos" / "S30.mp4").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                import_teachobs_asr_results(
                    manifest_path,
                    media_manifest,
                    root / "media",
                    caption_audit,
                    results,
                    duration_probe=lambda _: 15.0,
                )

    def test_gpu_runner_rejects_container_digest_before_cuda_execution(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, _, _, manifest = self._manifest(root)
            with patch(
                "teaching_skill_miner.teachobs_asr_handoff.model_directory_sha256",
                return_value="c" * 64,
            ), self.assertRaisesRegex(ValueError, "job manifest contract"):
                run_teachobs_asr_gpu_jobs(
                    manifest_path,
                    root / "media",
                    root / "model",
                    root / "results",
                    container_image_digest="sha256:" + "d" * 64,
                    runner_source_sha256=manifest["runtime_contract"][
                        "runner_module_sha256"
                    ],
                    duration_probe=lambda _: 15.0,
                )

    def test_overlong_engine_segments_split_at_verified_word_boundaries(
        self,
    ) -> None:
        words = [
            SimpleNamespace(start=0.0, end=10.0, word=" alpha"),
            SimpleNamespace(start=10.0, end=20.0, word=" beta"),
            SimpleNamespace(start=20.0, end=31.0, word=" gamma"),
            SimpleNamespace(start=31.0, end=44.18, word=" delta"),
        ]
        source = SimpleNamespace(
            start=0.0,
            end=44.18,
            text=" alpha beta gamma delta",
            words=words,
        )
        segments, provenance = (
            teachobs_asr_handoff._postprocess_whisper_segments(
                [source],
                maximum_segment_duration=30.5,
            )
        )
        self.assertEqual(
            segments,
            [
                {"start": 0.0, "end": 20.0, "text": "alpha beta"},
                {"start": 20.0, "end": 44.18, "text": " gamma delta"},
            ],
        )
        self.assertEqual(provenance["source_segment_count"], 1)
        self.assertEqual(provenance["source_segments_split"], 1)
        self.assertEqual(provenance["output_segment_count"], 2)
        self.assertEqual(provenance["overlong_source_word_timestamp_count"], 4)
        self.assertEqual(provenance["positive_duration_word_anchor_count"], 4)
        self.assertEqual(provenance["attached_zero_duration_word_count"], 0)
        self.assertTrue(
            provenance["split_text_trimmed_exact_equivalence_verified"]
        )
        self.assertEqual(
            "".join(segment["text"] for segment in segments),
            source.text.strip(),
        )
        self.assertTrue(
            all(
                segment["end"] - segment["start"] <= 30.5
                for segment in segments
            )
        )
        repeated, repeated_provenance = (
            teachobs_asr_handoff._postprocess_whisper_segments(
                [source],
                maximum_segment_duration=30.5,
            )
        )
        self.assertEqual(repeated, segments)
        self.assertEqual(repeated_provenance, provenance)

        exact_limit = SimpleNamespace(
            start=0.0,
            end=30.5,
            text="private fixture words",
            words=None,
        )
        exact_segments, exact_provenance = (
            teachobs_asr_handoff._postprocess_whisper_segments(
                [exact_limit],
                maximum_segment_duration=30.5,
            )
        )
        self.assertEqual(len(exact_segments), 1)
        self.assertEqual(exact_provenance["source_segments_split"], 0)

        without_words = SimpleNamespace(
            start=0.0,
            end=30.501,
            text="private fixture words",
            words=None,
        )
        with self.assertRaisesRegex(ValueError, "word timestamps"):
            teachobs_asr_handoff._postprocess_whisper_segments(
                [without_words],
                maximum_segment_duration=30.5,
            )

        mismatched_text = SimpleNamespace(
            start=0.0,
            end=44.18,
            text="different private fixture words",
            words=words,
        )
        with self.assertRaisesRegex(ValueError, "preserve"):
            teachobs_asr_handoff._postprocess_whisper_segments(
                [mismatched_text],
                maximum_segment_duration=30.5,
            )

        invalid_word_cases = (
            [
                SimpleNamespace(start=0.0, end=31.0, word=" one"),
                SimpleNamespace(start=31.0, end=44.18, word=" two"),
            ],
            [
                SimpleNamespace(start=float("nan"), end=10.0, word=" one"),
                SimpleNamespace(start=10.0, end=44.18, word=" two"),
            ],
            [
                SimpleNamespace(start=0.0, end=25.0, word=" one"),
                SimpleNamespace(start=20.0, end=44.18, word=" two"),
            ],
        )
        for invalid_words in invalid_word_cases:
            invalid = SimpleNamespace(
                start=0.0,
                end=44.18,
                text=" one two",
                words=invalid_words,
            )
            with self.subTest(words=invalid_words), self.assertRaises(ValueError):
                teachobs_asr_handoff._postprocess_whisper_segments(
                    [invalid],
                    maximum_segment_duration=30.5,
                )

    def test_zero_duration_point_words_preserve_leading_middle_and_trailing_text(
        self,
    ) -> None:
        def process(
            words: list[SimpleNamespace],
        ) -> tuple[list[dict], dict]:
            source = SimpleNamespace(
                start=0.0,
                end=44.0,
                text="".join(word.word for word in words),
                words=words,
            )
            return teachobs_asr_handoff._postprocess_whisper_segments(
                [source],
                maximum_segment_duration=30.5,
            )

        cases = {
            "trailing": (
                [
                    SimpleNamespace(start=0.0, end=20.0, word=" alpha"),
                    SimpleNamespace(start=20.0, end=44.0, word=" beta"),
                    SimpleNamespace(start=44.0, end=44.0, word=" tail"),
                ],
                [
                    {"start": 0.0, "end": 20.0, "text": "alpha"},
                    {"start": 20.0, "end": 44.0, "text": " beta tail"},
                ],
                1,
            ),
            "leading": (
                [
                    SimpleNamespace(start=0.0, end=0.0, word=" lead"),
                    SimpleNamespace(start=0.0, end=20.0, word=" alpha"),
                    SimpleNamespace(start=20.0, end=44.0, word=" beta"),
                ],
                [
                    {"start": 0.0, "end": 20.0, "text": "lead alpha"},
                    {"start": 20.0, "end": 44.0, "text": " beta"},
                ],
                1,
            ),
            "middle_contiguous": (
                [
                    SimpleNamespace(start=0.0, end=20.0, word=" alpha"),
                    SimpleNamespace(start=20.0, end=20.0, word=" zero-one"),
                    SimpleNamespace(start=20.0, end=20.0, word=" zero-two"),
                    SimpleNamespace(start=20.0, end=44.0, word=" beta"),
                ],
                [
                    {
                        "start": 0.0,
                        "end": 20.0,
                        "text": "alpha zero-one zero-two",
                    },
                    {"start": 20.0, "end": 44.0, "text": " beta"},
                ],
                2,
            ),
        }
        for name, (words, expected, attached_count) in cases.items():
            with self.subTest(name=name):
                segments, provenance = process(words)
                self.assertEqual(segments, expected)
                self.assertEqual(
                    "".join(segment["text"] for segment in segments),
                    "".join(word.word for word in words).strip(),
                )
                self.assertEqual(
                    provenance["attached_zero_duration_word_count"],
                    attached_count,
                )
                self.assertEqual(
                    provenance["overlong_source_word_timestamp_count"],
                    len(words),
                )
                self.assertEqual(
                    provenance["positive_duration_word_anchor_count"],
                    len(words) - attached_count,
                )
                self.assertTrue(provenance["positive_word_boundaries_preserved"])
                self.assertTrue(
                    provenance[
                        "zero_duration_assignment_anchor_indices_monotonic_verified"
                    ]
                )
                repeated = process(words)
                self.assertEqual(repeated, (segments, provenance))

    def test_point_anchored_zero_duration_words_use_nearest_positive_anchor(
        self,
    ) -> None:
        words = [
            SimpleNamespace(start=0.0, end=10.0, word=" alpha"),
            SimpleNamespace(start=10.14, end=10.14, word=" near-previous"),
            SimpleNamespace(start=12.46, end=20.0, word=" beta"),
            SimpleNamespace(start=22.32, end=22.32, word=" tie-previous"),
            SimpleNamespace(start=24.5, end=24.5, word=" near-next"),
            SimpleNamespace(start=24.64, end=44.0, word=" gamma"),
        ]
        source = SimpleNamespace(
            start=0.0,
            end=44.0,
            text="".join(word.word for word in words),
            words=words,
        )
        expected = [
            {
                "start": 0.0,
                "end": 20.0,
                "text": "alpha near-previous beta tie-previous",
            },
            {
                "start": 24.64,
                "end": 44.0,
                "text": " near-next gamma",
            },
        ]
        first = teachobs_asr_handoff._postprocess_whisper_segments(
            [source],
            maximum_segment_duration=30.5,
        )
        second = teachobs_asr_handoff._postprocess_whisper_segments(
            [source],
            maximum_segment_duration=30.5,
        )
        segments, provenance = first
        self.assertEqual(first, second)
        self.assertEqual(segments, expected)
        self.assertEqual(
            "".join(segment["text"] for segment in segments),
            source.text.strip(),
        )
        self.assertEqual(provenance["attached_zero_duration_word_count"], 3)
        self.assertEqual(
            provenance[
                "maximum_observed_zero_duration_word_attachment_distance_seconds"
            ],
            2.32,
        )
        self.assertEqual(
            provenance["zero_duration_word_maximum_attachment_distance_seconds"],
            30.5,
        )
        self.assertTrue(
            provenance[
                "zero_duration_assignment_anchor_indices_monotonic_verified"
            ]
        )
        self.assertTrue(
            provenance["zero_duration_point_anchors_within_source_segment_verified"]
        )

    def test_zero_duration_words_fail_closed_without_valid_point_assignment(
        self,
    ) -> None:
        invalid_cases = {
            "all_zero": [
                SimpleNamespace(start=0.0, end=0.0, word=" all"),
                SimpleNamespace(start=0.0, end=0.0, word=" zero"),
            ],
            "over_maximum_distance": [
                SimpleNamespace(start=0.0, end=1.0, word=" alpha"),
                SimpleNamespace(start=32.0, end=32.0, word=" too-far"),
            ],
            "reverse": [
                SimpleNamespace(start=0.0, end=20.0, word=" alpha"),
                SimpleNamespace(start=21.0, end=20.0, word=" reversed"),
                SimpleNamespace(start=20.0, end=44.0, word=" beta"),
            ],
            "point_outside_segment": [
                SimpleNamespace(start=0.0, end=20.0, word=" alpha"),
                SimpleNamespace(start=44.1, end=44.1, word=" outside"),
            ],
        }
        for name, words in invalid_cases.items():
            source = SimpleNamespace(
                start=0.0,
                end=44.0,
                text="".join(word.word for word in words),
                words=words,
            )
            with self.subTest(name=name), self.assertRaises(ValueError):
                teachobs_asr_handoff._postprocess_whisper_segments(
                    [source],
                    maximum_segment_duration=30.5,
                )

    def test_public_pending_and_completed_receipts_exclude_private_rows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path, media_manifest, caption_audit, manifest = self._manifest(root)
            pending = build_pending_teachobs_asr_receipt(
                media_manifest,
                caption_audit,
                generated_at_utc="2026-07-23T00:00:00Z",
            )
            self.assertEqual(pending["evidence_status"]["handoff_status"], "pending")
            self.assertFalse(pending["evidence_status"]["asr_is_official_caption"])
            pending_path = write_json(root / "public-pending.json", pending)
            self.assertTrue(audit_release_path(pending_path)["passed"])
            pending_check = check_asr(
                str(pending_path),
                str(media_manifest),
                str(caption_audit),
                evaluation_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
            )
            self.assertEqual(pending_check["handoff_status"], "pending")

            results = root / "results"
            results.mkdir()
            write_json(results / "S30.json", self._result(manifest["jobs"][0], manifest))
            imported = import_teachobs_asr_results(
                manifest_path,
                media_manifest,
                root / "media",
                caption_audit,
                results,
                duration_probe=lambda _: 15.0,
                generated_at_utc="2026-07-23T00:30:00Z",
            )
            outputs = write_teachobs_asr_import(
                imported,
                audit_path=root / "private" / "asr_audit.json",
                coverage_matrix_path=root / "private" / "coverage.json",
            )
            receipt = build_public_teachobs_asr_receipt(
                imported["audit"],
                imported["coverage_matrix"],
                asr_import_audit_path=outputs["audit"],
                coverage_matrix_path=outputs["coverage_matrix"],
                generated_at_utc="2026-07-23T00:40:00Z",
            )
            serialized = json.dumps(receipt)
            self.assertNotIn("S30", serialized)
            self.assertNotIn("private fixture words", serialized)
            self.assertFalse(receipt["evidence_status"]["content_accuracy_established"])
            public_path = write_json(root / "public-completed.json", receipt)
            self.assertTrue(audit_release_path(public_path)["passed"])
            completed_check = check_asr(
                str(public_path),
                str(media_manifest),
                str(caption_audit),
                job_manifest_path=str(manifest_path),
                import_audit_path=str(outputs["audit"]),
                coverage_matrix_path=str(outputs["coverage_matrix"]),
            )
            self.assertEqual(completed_check["handoff_status"], "completed")
            self.assertEqual(completed_check["valid_asr_result_count"], 1)

            coverage_payload = outputs["coverage_matrix"].read_bytes()
            with outputs["coverage_matrix"].open("ab") as handle:
                handle.write(b" ")
            with self.assertRaisesRegex(
                StageCheckError, "coverage_matrix_file_sha256"
            ):
                check_asr(
                    str(public_path),
                    str(media_manifest),
                    str(caption_audit),
                    job_manifest_path=str(manifest_path),
                    import_audit_path=str(outputs["audit"]),
                    coverage_matrix_path=str(outputs["coverage_matrix"]),
                )
            outputs["coverage_matrix"].write_bytes(coverage_payload)

            forged = json.loads(json.dumps(receipt))
            forged["evidence_status"]["recognition_accuracy_established"] = True
            forged_path = write_json(root / "public-forged.json", forged)
            with self.assertRaisesRegex(StageCheckError, "unsafe ASR receipt claim"):
                check_asr(
                    str(forged_path),
                    str(media_manifest),
                    str(caption_audit),
                    job_manifest_path=str(manifest_path),
                    import_audit_path=str(outputs["audit"]),
                    coverage_matrix_path=str(outputs["coverage_matrix"]),
                )

            with media_manifest.open("a", encoding="utf-8") as handle:
                handle.write(" ")
            with self.assertRaisesRegex(StageCheckError, "current media manifest"):
                check_asr(
                    str(public_path),
                    str(media_manifest),
                    str(caption_audit),
                    job_manifest_path=str(manifest_path),
                    import_audit_path=str(outputs["audit"]),
                    coverage_matrix_path=str(outputs["coverage_matrix"]),
                )

    def test_installed_gpu_worker_hash_and_run_entrypoints(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_text("{}", encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                self.assertEqual(
                    main(
                        [
                            "hash-teachobs-asr-model",
                            "--model-directory",
                            str(model),
                        ]
                    ),
                    0,
                )
            self.assertRegex(output.getvalue().strip(), r"^[0-9a-f]{64}$")

            expected_runner_sha = sha256(
                Path(teachobs_asr_handoff.__file__).resolve().read_bytes()
            ).hexdigest()
            with patch(
                "teaching_skill_miner.cli.run_teachobs_asr_gpu_jobs",
                return_value={"completed_result_count": 1},
            ) as mocked_run, redirect_stdout(io.StringIO()):
                self.assertEqual(
                    main(
                        [
                            "run-teachobs-asr-gpu",
                            "--job-manifest",
                            str(root / "jobs.json"),
                            "--media-root",
                            str(root / "media"),
                            "--model-directory",
                            str(model),
                            "--output",
                            str(root / "results"),
                            "--container-image-digest",
                            "sha256:" + "a" * 64,
                            "--lesson-id",
                            "S30",
                        ]
                    ),
                    0,
                )
            self.assertEqual(
                mocked_run.call_args.kwargs["runner_source_sha256"],
                expected_runner_sha,
            )
            self.assertEqual(
                mocked_run.call_args.kwargs["container_image_digest"],
                "sha256:" + "a" * 64,
            )
            self.assertEqual(mocked_run.call_args.kwargs["lesson_ids"], ["S30"])


if __name__ == "__main__":
    unittest.main()
