from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from scripts.check_teachobs_external_study import (
    FEATURE_FAILURE_SCHEMA,
    FEATURE_MANIFEST_SCHEMA,
    MEDIA_MANIFEST_SCHEMA,
    MEDIA_PLAN_SCHEMA,
    PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
    StageCheckError,
    build_parser,
    check_benchmark,
    check_features,
    check_human_template_reusable,
    check_media,
)


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_teachobs_external_study.sh"
CHECKER = ROOT / "scripts" / "check_teachobs_external_study.py"
MEDIA_PREPARER = ROOT / "scripts" / "run_teachobs_media_preparation.py"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _media_canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=True,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TeachObsExternalRunnerTests(unittest.TestCase):
    def test_benchmark_checker_cli_requires_transcript_materialization(self) -> None:
        parser = build_parser()
        required = [
            "benchmark",
            "--result",
            "result.json",
            "--receipt",
            "receipt.json",
            "--frozen-model-output",
            "frozen",
            "--feature-manifest",
            "features.json",
        ]
        with self.assertRaises(SystemExit):
            parser.parse_args(required)
        parsed = parser.parse_args(
            [
                *required,
                "--transcript-materialization-manifest",
                "transcripts/manifest.json",
            ]
        )
        self.assertEqual(
            parsed.transcript_materialization_manifest,
            "transcripts/manifest.json",
        )

    def test_shell_syntax_and_help_are_safe_without_authorization(self) -> None:
        syntax = subprocess.run(
            ["sh", "-n", os.fspath(RUNNER)],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(syntax.returncode, 0, syntax.stderr)
        help_run = subprocess.run(
            ["sh", os.fspath(RUNNER), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(help_run.returncode, 0)
        self.assertIn("Resume the real 30-lesson", help_run.stderr)
        self.assertIn("Remote yt-dlp", help_run.stderr)
        self.assertIn("--cookies-from-browser", help_run.stderr)
        self.assertIn("account challenges or suspension", help_run.stderr)
        self.assertIn("--yt-dlp-direct", help_run.stderr)
        self.assertIn("--yt-dlp-impersonate chrome", help_run.stderr)
        self.assertIn("--evaluation-profile PROFILE", help_run.stderr)

        media_help = subprocess.run(
            [sys.executable, os.fspath(MEDIA_PREPARER), "--help"],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(media_help.returncode, 0, media_help.stderr)
        self.assertIn("--cookies-from-browser", media_help.stdout)
        self.assertIn("--yt-dlp-direct", media_help.stdout)
        self.assertIn("--yt-dlp-impersonate {chrome}", media_help.stdout)

    def test_source_and_override_terms_are_independent_fail_closed_gates(
        self,
    ) -> None:
        environment = dict(os.environ)
        environment.pop("TSM_TEACHOBS_SOURCE_TERMS_ACKNOWLEDGED", None)
        no_source_ack = subprocess.run(
            ["sh", os.fspath(RUNNER)],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(no_source_ack.returncode, 2)
        self.assertIn("without --acknowledge-source-terms", no_source_ack.stderr)

        no_override_ack = subprocess.run(
            [
                "sh",
                os.fspath(RUNNER),
                "--acknowledge-source-terms",
                "--source-override-manifest",
                "private-override.json",
            ],
            cwd=ROOT,
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(no_override_ack.returncode, 2)
        self.assertIn("also requires", no_override_ack.stderr)

    def test_media_preparer_rejects_cookie_profile_paths_before_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "must-not-exist"
            rejected = subprocess.run(
                [
                    sys.executable,
                    os.fspath(MEDIA_PREPARER),
                    "--repository",
                    os.fspath(root / "missing-repository"),
                    "--output",
                    os.fspath(output),
                    "--stage",
                    "plan",
                    "--cookies-from-browser",
                    "chrome:/private/browser/profile",
                ],
                cwd=ROOT,
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(rejected.returncode, 0)
            self.assertIn("safe local profile name", rejected.stderr)
            self.assertNotIn("/private/browser/profile", rejected.stderr)
            self.assertFalse(output.exists())

    def test_runner_has_required_stages_without_positive_evidence_shortcuts(
        self,
    ) -> None:
        runner = RUNNER.read_text(encoding="utf-8")
        checker = CHECKER.read_text(encoding="utf-8")
        for command in (
            "fetch-teachobs",
            "audit-teachobs-captions",
            "prepare-teachobs-asr-handoff",
            "run_teachobs_media_preparation.py",
            "benchmark-teachobs-multimodal",
            "prepare-teachobs-double-annotation",
            "prepare-teachobs-lockbox-preregistration",
            "release-audit",
        ):
            self.assertIn(command, runner)
        for gate in (
            "selected_complete",
            "downloaded_lesson_count_total",
            "clip_embeddings_included",
            "private_result_canonical_sha256",
            "human-template-reusable",
            "test_prediction_bitwise_parity_verified",
            "load_teachobs_frozen_bundle",
            "verify_teachobs_lockbox_artifact_files",
            "ASR receipt is not bound to the current media manifest",
            "ASR receipt is not bound to the current caption audit",
        ):
            self.assertIn(gate, runner + checker)
        self.assertIn('"$tsm_checker" asr', runner)
        self.assertIn("TSM_TEACHOBS_HUMAN_RECEIPT", runner)
        self.assertIn("TSM_TEACHOBS_LOCKBOX_DRAFT", runner)
        self.assertIn("private ASR import evidence exists", runner)
        self.assertIn("private ASR job manifest exists", runner)
        self.assertIn('[ -e "$tsm_asr_import_audit" ]', runner)
        self.assertIn('[ -e "$tsm_asr_job_manifest" ]', runner)
        self.assertIn("run_asr_receipt_check full", runner)
        self.assertIn("run_asr_receipt_check job", runner)
        asr_gate = runner.split(
            'echo "[TeachObs ASR]', 1
        )[1].split('echo "[TeachObs 5/10]', 1)[0]
        self.assertLess(
            asr_gate.index('[ -e "$tsm_asr_import_audit" ]'),
            asr_gate.index("run_asr_receipt_check none"),
        )
        self.assertLess(
            asr_gate.index('[ -e "$tsm_asr_job_manifest" ]'),
            asr_gate.index("run_asr_receipt_check none"),
        )
        self.assertIn('--evaluation-profile "$tsm_evaluation_profile"', runner)
        self.assertIn("--pending-only", runner)
        self.assertLess(
            runner.rfind('run_asr_receipt_check "$tsm_asr_receipt_mode"'),
            runner.rfind('release-audit "$tsm_public_root"'),
        )
        self.assertIn("--frozen-model-output", runner)
        self.assertIn("--transcript-materialization-manifest", runner)
        self.assertIn("--transcript-materialization-manifest", checker)
        self.assertIn(
            "--analysis-code teaching_skill_miner/teachobs_lockbox.py",
            runner,
        )
        self.assertIn(
            "TSM_TEACHOBS_EVALUATION_PROFILE:-paper_track1_23_train_6_test}",
            runner,
        )
        self.assertIn('--evaluation-profile "$tsm_evaluation_profile"', runner)
        self.assertIn("unique expected S4 hole", runner)
        self.assertIn("--system-artifact \"$tsm_frozen_model_root/bundle_manifest.json\"", runner)
        for arm in (
            "transcript_only",
            "transcript_audio",
            "transcript_visual",
            "full",
        ):
            self.assertIn(f'--arm-model "{arm}=', runner)
        self.assertIn("TSM_TEACHOBS_SOURCE_OVERRIDE_MANIFEST:-}", runner)
        self.assertIn("TSM_TEACHOBS_JS_RUNTIME:-}", runner)
        self.assertIn("TSM_TEACHOBS_COOKIES_FROM_BROWSER:-}", runner)
        self.assertIn("TSM_TEACHOBS_YT_DLP_DIRECT:-false}", runner)
        self.assertIn("TSM_TEACHOBS_YT_DLP_IMPERSONATE:-}", runner)
        self.assertIn("TSM_TEACHOBS_YT_DLP_YOUTUBE_CLIENT:-}", runner)
        self.assertIn(
            'set -- "$@" --cookies-from-browser "$tsm_cookies_from_browser"',
            runner,
        )
        self.assertIn('set -- "$@" --yt-dlp-direct', runner)
        self.assertIn(
            'set -- "$@" --yt-dlp-impersonate "$tsm_yt_dlp_impersonate"',
            runner,
        )
        caption_stage = runner.split("run_caption_audit() {", 1)[1].split(
            'echo "[TeachObs 3/10]', 1
        )[0]
        self.assertIn(
            'set -- "$@" --cookies-from-browser "$tsm_cookies_from_browser"',
            caption_stage,
        )
        self.assertIn('set -- "$@" --yt-dlp-direct', caption_stage)
        self.assertIn(
            'set -- "$@" --yt-dlp-impersonate "$tsm_yt_dlp_impersonate"',
            caption_stage,
        )
        self.assertIn(
            'set -- "$@" --yt-dlp-youtube-client "$tsm_yt_dlp_youtube_client"',
            caption_stage,
        )
        self.assertNotIn("--remote-components", runner)
        self.assertNotIn("analyze-teachobs-double-annotation", runner)
        self.assertNotIn("analyze-learner-effect-study", runner)
        for claim in (
            "confirmatory_multimodal_gain_established=false",
            "external_lockbox_established=false",
            "deployment_accuracy_established=false",
            "learner_effectiveness_established=false",
        ):
            self.assertIn(claim, runner)
        self.assertIn("frozen_artifact_set_complete=true", runner)

    def test_feature_postcondition_rejects_partial_success(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            media = root / "media_manifest.json"
            media_value = {
                "schema": MEDIA_MANIFEST_SCHEMA,
                "plan_sha256": "a" * 64,
                "unicode_private_metadata_fixture": "真实课堂",
            }
            _write_json(media, media_value)
            feature = root / "feature_manifest.json"
            value = {
                "schema": FEATURE_MANIFEST_SCHEMA,
                "dataset_id": "teachobs_v0_1_human_validated",
                "plan_sha256": "a" * 64,
                "lesson_count": 30,
                "expected_lesson_count": 30,
                "scene_count": 5158,
                "complete": True,
                "failed_lesson_count": 0,
                "media_manifest_sha256": _media_canonical_sha256(media_value),
                "audio_statistics_included": True,
                "visual_evidence_included": True,
                "ocr_requested": True,
                "clip_embeddings_requested": True,
                "clip_embeddings_included": True,
                "claim_boundary": {
                    "recognition_accuracy_established": False,
                    "multimodal_gain_established": False,
                },
            }
            _write_json(feature, value)
            report = check_features(os.fspath(feature), os.fspath(media))
            self.assertTrue(report["complete"])

            media.write_text(
                json.dumps(
                    media_value,
                    ensure_ascii=False,
                    sort_keys=False,
                    separators=(", ", ": "),
                )
                + "\n",
                encoding="utf-8",
            )
            self.assertTrue(
                check_features(os.fspath(feature), os.fspath(media))["complete"]
            )
            _write_json(media, {**media_value, "tampered": True})
            with self.assertRaisesRegex(StageCheckError, "not bound"):
                check_features(os.fspath(feature), os.fspath(media))
            _write_json(media, media_value)

            value["schema"] = "wrong-schema"
            _write_json(feature, value)
            with self.assertRaisesRegex(StageCheckError, "feature manifest schema"):
                check_features(os.fspath(feature), os.fspath(media))
            value["schema"] = FEATURE_MANIFEST_SCHEMA

            value["complete"] = False
            value["failed_lesson_count"] = 1
            _write_json(feature, value)
            with self.assertRaisesRegex(StageCheckError, "incomplete"):
                check_features(os.fspath(feature), os.fspath(media))

    def test_paper_profile_gates_accept_only_the_unique_s4_hole(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plan = root / "media_plan.json"
            plan_value = {
                "schema": MEDIA_PLAN_SCHEMA,
                "repository_provenance": {
                    "repository_commit": (
                        "96c251ae09e79edd06a3a9bbaaa8b8f7fe99a15c"
                    )
                },
                "lesson_count": 30,
                "scene_count": 5158,
            }
            plan_value["plan_sha256"] = _media_canonical_sha256(plan_value)
            _write_json(plan, plan_value)
            selected_ids = [f"S{index}" for index in range(1, 31) if index != 4]
            media = root / "media_manifest.json"
            media_value = {
                    "schema": MEDIA_MANIFEST_SCHEMA,
                    "dataset_id": "teachobs_v0_1_human_validated",
                    "plan_sha256": plan_value["plan_sha256"],
                    "selected_lesson_count": 30,
                    "selected_complete": False,
                    "downloaded_lesson_count_total": 29,
                    "active_source_override_count": 0,
                    "active_source_override_manifest_sha256": None,
                    "override_source_terms_acknowledged": False,
                    "lessons": [
                        {"lesson_id": lesson_id} for lesson_id in selected_ids
                    ],
                    "claim_boundary": {
                        "candidate_same_content_mirror_count": 0,
                        "candidate_same_content_mirror_used": False,
                    },
                }
            _write_json(media, media_value)
            media_report = check_media(
                os.fspath(plan),
                os.fspath(media),
                source_override_manifest=None,
                evaluation_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
            )
            self.assertEqual(media_report["lesson_count"], 29)
            self.assertEqual(media_report["scene_count"], 4945)

            failure = root / "feature_failures.json"
            feature = root / "feature_manifest.json"
            media_value = json.loads(media.read_text(encoding="utf-8"))
            media_digest = _media_canonical_sha256(media_value)
            _write_json(
                failure,
                {
                    "schema": FEATURE_FAILURE_SCHEMA,
                    "dataset_id": "teachobs_v0_1_human_validated",
                    "plan_sha256": plan_value["plan_sha256"],
                    "media_manifest_sha256": media_digest,
                    "failure_count": 1,
                    "failures": [{"lesson_id": "S4"}],
                },
            )
            _write_json(
                feature,
                {
                    "schema": FEATURE_MANIFEST_SCHEMA,
                    "dataset_id": "teachobs_v0_1_human_validated",
                    "plan_sha256": plan_value["plan_sha256"],
                    "lesson_count": 29,
                    "expected_lesson_count": 30,
                    "scene_count": 4945,
                    "complete": False,
                    "failed_lesson_count": 1,
                    "failed_lesson_ids": ["S4"],
                    "failure_records_path": failure.name,
                    "failure_records_sha256": _sha256(failure),
                    "media_manifest_sha256": media_digest,
                    "audio_statistics_included": True,
                    "visual_evidence_included": True,
                    "ocr_requested": True,
                    "clip_embeddings_requested": True,
                    "clip_embeddings_included": True,
                    "clip_embeddings_complete_for_all_lessons": False,
                    "combined_clip_complete": False,
                    "combined_clip_lesson_count": 29,
                    "combined_clip_expected_lesson_count": 30,
                    "combined_clip_lesson_ids": selected_ids,
                    "combined_clip_failed_lesson_ids": ["S4"],
                    "combined_clip_frame_count": 4945,
                    "lessons": [
                        {"lesson_id": lesson_id} for lesson_id in selected_ids
                    ],
                    "claim_boundary": {
                        "full_scene_midpoint_frames_extracted": False,
                        "audio_statistics_computed": False,
                        "image_metrics_computed": False,
                        "adjacent_visual_events_inferred": False,
                        "clip_visual_embeddings_computed": True,
                        "clip_visual_embeddings_complete_for_all_lessons": False,
                        "clip_visual_embeddings_partial_success_only": True,
                        "recognition_accuracy_established": False,
                        "multimodal_gain_established": False,
                    },
                },
            )
            report = check_features(
                os.fspath(feature),
                os.fspath(media),
                evaluation_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
            )
            self.assertEqual(report["lesson_count"], 29)
            self.assertEqual(report["scene_count"], 4945)

            value = json.loads(feature.read_text(encoding="utf-8"))
            value["failed_lesson_ids"] = ["S5"]
            _write_json(feature, value)
            with self.assertRaisesRegex(StageCheckError, "exact profile exception"):
                check_features(
                    os.fspath(feature),
                    os.fspath(media),
                    evaluation_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                )

    def test_pending_human_template_is_never_overwritten_after_any_change(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            assignments: dict[str, dict[str, object]] = {}
            for slot in ("A", "B"):
                path = root / f"assignment_{slot}.csv"
                path.write_text("item_token,label::code\ntoken,\n", encoding="utf-8")
                assignments[slot] = {
                    "template_file": path.name,
                    "template_file_sha256": _sha256(path),
                }
            manifest = root / "assignment_manifest.json"
            _write_json(
                manifest,
                {
                    "human_completion": False,
                    "operational_codebook": {
                        "operational_definitions_complete": False,
                        "annotation_execution_ready": False,
                    },
                    "assignments": assignments,
                },
            )
            report = check_human_template_reusable(os.fspath(manifest))
            self.assertTrue(report["safe_to_regenerate_pending_skeleton"])

            with (root / "assignment_A.csv").open("a", encoding="utf-8") as handle:
                handle.write("human-token,1\n")
            with self.assertRaisesRegex(StageCheckError, "refusing to overwrite"):
                check_human_template_reusable(os.fspath(manifest))

    def test_benchmark_checker_binds_frozen_bundle_and_companion_arrays(
        self,
    ) -> None:
        arms = (
            "transcript_only",
            "transcript_audio",
            "transcript_visual",
            "full",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            feature_manifest = root / "feature_manifest.json"
            _write_json(feature_manifest, {"fixture": True})
            transcript_manifest_path = root / "transcript_manifest.json"
            _write_json(transcript_manifest_path, {"private_fixture": True})
            transcript_manifest = {
                "schema": (
                    "teaching_skill_miner."
                    "teachobs_transcript_materialization_manifest.v2"
                ),
                "profile_id": PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                "manifest_sha256": "d" * 64,
                "materialization_fingerprint_sha256": "e" * 64,
                "ordered_sample_id_sha256": "f" * 64,
                "ordered_scene_text_sha256": "1" * 64,
                "repository_binding": {"repository_commit": "pinned"},
                "input_hashes": {"caption_audit_file_sha256": "2" * 64},
                "selection": {"selected_train_scene_count": 3846},
                "aggregate": {
                    "lesson_count": 29,
                    "scene_count": 4945,
                    "nonempty_scene_count": 2,
                    "empty_scene_count": 4943,
                    "source_tier_lesson_counts": {
                        "platform_creator_provided_caption": 19,
                        "platform_automatic_caption": 4,
                        "audited_asr_fallback": 6,
                    },
                    "source_tier_scene_counts": {
                        "platform_creator_provided_caption": 3000,
                        "platform_automatic_caption": 900,
                        "audited_asr_fallback": 1045,
                    },
                },
            }
            transcript_rows = tuple(
                {
                    "sample_id": f"sample:{index}",
                    "text": (
                        "train transcript"
                        if index == 0
                        else "test transcript"
                        if index == 3846
                        else ""
                    ),
                }
                for index in range(4945)
            )
            transcript_texts = [row["text"] for row in transcript_rows]
            feature_manifest_sha256 = _sha256(feature_manifest)
            transcript_manifest_file_sha256 = _sha256(
                transcript_manifest_path
            )
            benchmark_input_fingerprint = _canonical_sha256(
                {
                    "schema": "teaching_skill_miner.teachobs_benchmark_input.v1",
                    "dataset_profile_fingerprint": "c" * 64,
                    "feature_manifest_sha256": feature_manifest_sha256,
                    "transcript_materialization_manifest_file_sha256": (
                        transcript_manifest_file_sha256
                    ),
                    "transcript_materialization_manifest_sha256": "d" * 64,
                    "transcript_materialization_fingerprint_sha256": "e" * 64,
                    "transcript_ordered_sample_id_sha256": "f" * 64,
                    "transcript_ordered_scene_text_sha256": "1" * 64,
                }
            )
            transcript_audit = {
                "schema": transcript_manifest["schema"],
                "profile_id": PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                "manifest_file_sha256": transcript_manifest_file_sha256,
                "manifest_sha256": "d" * 64,
                "materialization_fingerprint_sha256": "e" * 64,
                "ordered_sample_id_sha256": "f" * 64,
                "ordered_scene_text_sha256": "1" * 64,
                "repository_binding_sha256": _canonical_sha256(
                    transcript_manifest["repository_binding"]
                ),
                "input_hashes_sha256": _canonical_sha256(
                    transcript_manifest["input_hashes"]
                ),
                "selected_train_transcript_order_sha256": _canonical_sha256(
                    transcript_texts[:3846]
                ),
                "selected_test_transcript_order_sha256": _canonical_sha256(
                    transcript_texts[3846:]
                ),
                **transcript_manifest["aggregate"],
                "all_selected_scene_transcripts_materialized": True,
                "sample_order_matches_benchmark_profile": True,
                "released_transcript_fallback_used": False,
                "labels_read_or_used": False,
                "empty_scenes_filled_from_released_transcript": False,
                "released_repository_transcripts_used_for_model_input": False,
                "benchmark_input_fingerprint": benchmark_input_fingerprint,
            }
            frozen = root / "frozen"
            frozen.mkdir()
            bundle_manifest = frozen / "bundle_manifest.json"
            _write_json(bundle_manifest, {"fixture_bundle": True})
            arm_artifacts: dict[str, dict[str, str]] = {}
            for arm in arms:
                arm_root = frozen / arm
                arm_root.mkdir()
                manifest = arm_root / "manifest.json"
                arrays = arm_root / "arrays.npz"
                _write_json(manifest, {"arm": arm})
                arrays.write_bytes(f"arrays:{arm}".encode("ascii"))
                arm_artifacts[arm] = {
                    "model_manifest_path": os.fspath(manifest.resolve()),
                    "model_manifest_file_sha256": _sha256(manifest),
                    "numeric_state_file_sha256": _sha256(arrays),
                }
            result = {
                "benchmark_kind": (
                    "teachobs_official_held_out_four_arm_multimodal_benchmark"
                ),
                "arms": {arm: {} for arm in arms},
                "dataset_audit": {
                    "benchmark_profile": PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                    "dataset_profile_fingerprint": "c" * 64,
                    "benchmark_input_fingerprint": benchmark_input_fingerprint,
                    "lesson_count": 29,
                    "train_lesson_count": 23,
                    "test_lesson_count": 6,
                    "scene_count": 4945,
                    "test_scene_count": 1099,
                    "code_count": 39,
                },
                "private_transcript_audit": transcript_audit,
                "private_feature_audit": {
                    "feature_manifest_sha256": feature_manifest_sha256
                },
                "protocol": {
                    "benchmark_profile": PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                    "same_test_scenes_used_for_all_arms": True,
                    "test_labels_used_for_feature_fitting_training_or_tuning": False,
                    "s4_labels_used_for_fitting_tuning_or_metrics": False,
                    "same_materialized_transcript_scenes_used_for_all_arms": True,
                    "released_repository_transcripts_used_for_model_input": False,
                    "transcript_materialization_manifest_required": True,
                    "benchmark_input_fingerprint": benchmark_input_fingerprint,
                },
                "paired_cluster_bootstrap": {"cluster_count": 6},
                "evidence_scope": {
                    "provisional_result": True,
                    "audited_transcript_materialization_used": True,
                    "released_repository_transcripts_used_for_model_input": False,
                    "transcript_content_accuracy_established": False,
                    "transcript_word_error_rate_established": False,
                    "confirmatory_multimodal_gain_established": False,
                    "external_lockbox_established": False,
                    "deployment_accuracy_established": False,
                    "learning_effectiveness_established": False,
                },
                "frozen_model_export": {
                    "output_directory": os.fspath(frozen.resolve()),
                    "bundle_fingerprint": "a" * 64,
                    "bundle_manifest_file_sha256": _sha256(bundle_manifest),
                    "benchmark_profile": PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                    "dataset_profile_fingerprint": "c" * 64,
                    "benchmark_input_fingerprint": benchmark_input_fingerprint,
                    "transcript_materialization_manifest_file_sha256": (
                        transcript_manifest_file_sha256
                    ),
                    "transcript_materialization_fingerprint_sha256": "e" * 64,
                    "arm_count": 4,
                    "arms": list(arms),
                    "arm_model_artifacts": arm_artifacts,
                    "pickle_used": False,
                    "deterministic_npz": True,
                    "test_prediction_bitwise_parity_verified": True,
                    "test_probability_maximum_absolute_delta": 0.0,
                    "public_test_labels_were_accessible_before_freeze": True,
                    "confirmatory_lockbox_result_established": False,
                    "deployment_accuracy_established": False,
                },
            }
            result_path = root / "result.json"
            _write_json(result_path, result)
            receipt = {
                "receipt_kind": "teachobs_aggregate_four_arm_multimodal_benchmark",
                "private_result_canonical_sha256": _canonical_sha256(result),
                "dataset_aggregate": {
                    "benchmark_profile": PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                    "lesson_count": 29,
                    "test_lesson_count": 6,
                    "test_scene_count": 1099,
                    "dataset_profile_fingerprint": "c" * 64,
                    "benchmark_input_fingerprint": benchmark_input_fingerprint,
                },
                "transcript_aggregate": transcript_audit,
                "claim_boundaries": {
                    "confirmatory_multimodal_gain_established": False,
                    "external_lockbox_established": False,
                    "deployment_accuracy_established": False,
                    "learning_effectiveness_established": False,
                },
            }
            receipt_path = root / "receipt.json"
            _write_json(receipt_path, receipt)

            fake_bundle = SimpleNamespace(
                bundle_fingerprint="a" * 64,
                arms={arm: object() for arm in arms},
                transcript_materialization_fingerprint_sha256="e" * 64,
                benchmark_input_fingerprint=benchmark_input_fingerprint,
            )

            def fake_arm(path: Path, **_: object) -> SimpleNamespace:
                arm = path.parent.name
                return SimpleNamespace(
                    arm=arm,
                    arrays_file_sha256=arm_artifacts[arm][
                        "numeric_state_file_sha256"
                    ],
                    training_provenance={
                        "transcript_materialization": {
                            "materialization_fingerprint_sha256": "e" * 64
                        },
                        "benchmark_input_fingerprint": (
                            benchmark_input_fingerprint
                        ),
                    },
                )

            with (
                patch(
                    "teaching_skill_miner.teachobs_transcript_materialization."
                    "validate_teachobs_transcript_materialization",
                    return_value={
                        "manifest": transcript_manifest,
                        "manifest_path": transcript_manifest_path,
                        "rows": transcript_rows,
                        "text_by_sample_id": {
                            row["sample_id"]: row["text"]
                            for row in transcript_rows
                        },
                    },
                ) as live_validate,
                patch(
                    "teaching_skill_miner.teachobs_frozen_model."
                    "load_teachobs_frozen_bundle",
                    return_value=fake_bundle,
                ) as load_bundle,
                patch(
                    "teaching_skill_miner.teachobs_frozen_model."
                    "load_teachobs_frozen_arm",
                    side_effect=fake_arm,
                ) as load_arm,
            ):
                report = check_benchmark(
                    os.fspath(result_path),
                    os.fspath(receipt_path),
                    os.fspath(frozen),
                    os.fspath(feature_manifest),
                    os.fspath(transcript_manifest_path),
                    evaluation_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                )
                self.assertTrue(report["frozen_bundle_integrity_verified"])
                self.assertTrue(
                    report[
                        "transcript_materialization_live_validation_verified"
                    ]
                )
                self.assertEqual(
                    report["benchmark_input_fingerprint"],
                    benchmark_input_fingerprint,
                )
                live_validate.assert_called_with(
                    os.fspath(transcript_manifest_path),
                    expected_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                )
                self.assertEqual(
                    load_bundle.call_args.kwargs[
                        "expected_transcript_materialization_fingerprint"
                    ],
                    "e" * 64,
                )
                self.assertEqual(
                    load_bundle.call_args.kwargs[
                        "expected_benchmark_input_fingerprint"
                    ],
                    benchmark_input_fingerprint,
                )
                self.assertEqual(load_arm.call_count, 4)
                self.assertTrue(
                    all(
                        call.kwargs[
                            "expected_transcript_materialization_fingerprint"
                        ]
                        == "e" * 64
                        and call.kwargs[
                            "expected_benchmark_input_fingerprint"
                        ]
                        == benchmark_input_fingerprint
                        for call in load_arm.call_args_list
                    )
                )
                relocated_frozen = root / "relocated-frozen-models"
                shutil.copytree(frozen, relocated_frozen)
                relocated_report = check_benchmark(
                    os.fspath(result_path),
                    os.fspath(receipt_path),
                    os.fspath(relocated_frozen),
                    os.fspath(feature_manifest),
                    os.fspath(transcript_manifest_path),
                    evaluation_profile=PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE,
                )
                self.assertTrue(
                    relocated_report["frozen_bundle_integrity_verified"]
                )
                result["private_transcript_audit"] = {
                    **transcript_audit,
                    "manifest_sha256": "9" * 64,
                }
                _write_json(result_path, result)
                receipt["private_result_canonical_sha256"] = _canonical_sha256(
                    result
                )
                receipt["transcript_aggregate"] = result[
                    "private_transcript_audit"
                ]
                _write_json(receipt_path, receipt)
                with self.assertRaisesRegex(
                    StageCheckError,
                    "private transcript audit differs",
                ):
                    check_benchmark(
                        os.fspath(result_path),
                        os.fspath(receipt_path),
                        os.fspath(frozen),
                        os.fspath(feature_manifest),
                        os.fspath(transcript_manifest_path),
                        evaluation_profile=(
                            PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
                        ),
                    )
                result["private_transcript_audit"] = transcript_audit
                _write_json(result_path, result)
                receipt["private_result_canonical_sha256"] = _canonical_sha256(
                    result
                )
                receipt["transcript_aggregate"] = transcript_audit
                _write_json(receipt_path, receipt)
                (frozen / "full" / "arrays.npz").write_bytes(b"tampered")
                with self.assertRaisesRegex(StageCheckError, "arrays hash mismatch"):
                    check_benchmark(
                        os.fspath(result_path),
                        os.fspath(receipt_path),
                        os.fspath(frozen),
                        os.fspath(feature_manifest),
                        os.fspath(transcript_manifest_path),
                        evaluation_profile=(
                            PAPER_TRACK1_23_TRAIN_6_TEST_PROFILE
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
