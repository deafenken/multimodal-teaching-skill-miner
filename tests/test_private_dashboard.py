from __future__ import annotations

from contextlib import redirect_stdout
from hashlib import sha256
from html.parser import HTMLParser
import io
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch
from urllib.parse import urlsplit

from teaching_skill_miner.cli import main
from teaching_skill_miner.private_dashboard import (
    BEHAVIOR_PROJECTION_SCHEMA,
    DEFAULT_PRIVATE_SKILL,
    PRIVATE_DASHBOARD_ARMS,
    PRIVATE_SKILL_EVALUATION_DIMENSIONS,
    PRIVATE_SKILL_EVALUATION_GATES,
    PrivateDashboardConfig,
    PrivateDashboardError,
    PrivateDashboardSnapshot,
    PrivateLessonSnapshot,
    PrivateSkillSnapshot,
    RangeNotSatisfiable,
    create_private_dashboard_server,
    _build_private_skill_snapshots,
    parse_single_byte_range,
    private_dashboard_html_bytes,
    private_dashboard_asset_bytes,
    private_dashboard_template_self_check,
    private_snapshot_summary,
    project_teachobs_behavior_view,
    serve_private_dashboard,
)


class _PrivateDashboardParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.video_count = 0
        self.external_assets: list[str] = []

    def handle_starttag(
        self,
        tag: str,
        attrs: list[tuple[str, str | None]],
    ) -> None:
        if tag == "video":
            self.video_count += 1
        values = dict(attrs)
        for name in ("src", "href"):
            value = values.get(name)
            if value and value.startswith(("http://", "https://", "//")):
                self.external_assets.append(value)


def _digest(payload: bytes) -> str:
    return sha256(payload).hexdigest()


def _make_snapshot(
    root: Path,
    *,
    expected_media_sha256: str | None = None,
    expected_frame_sha256: str | None = None,
) -> PrivateDashboardSnapshot:
    media_payload = b"private-video-test-bytes"
    frame_payload = b"private-frame-test-bytes"
    media_path = root / "opaque-lesson-video.mp4"
    frame_path = root / "opaque-scene-frame.jpg"
    media_path.write_bytes(media_payload)
    frame_path.write_bytes(frame_payload)
    label_rows = [
        {
            "label": "Gesture",
            "group": "visual",
            "predicted": True,
            "score": 0.75,
            "reference": True,
        },
        {
            "label": "Checking",
            "group": "nonvisual",
            "predicted": True,
            "score": 0.62,
            "reference": False,
        },
        {
            "label": "Student raises hand",
            "group": "unknown",
            "predicted": True,
            "score": 0.99,
            "reference": False,
        },
    ]
    scene = {
        "schema_version": "1.1",
        "lesson": "S24",
        "scene_number": 1,
        "scene_count": 1,
        "start_seconds": 0.0,
        "end_seconds": 15.0,
        "transcript": {
            "text": "A materialized transcript row.",
            "source_tier": "audited_asr_fallback",
            "empty": False,
        },
        "visual": {
            "frame_url": "frames/S24/1",
            "frame_kind": "time_aligned_scene_frame",
            "ocr_used_as_frozen_model_input": True,
            "ocr_displayed": False,
        },
        "reference": {
            "kind": "released_consensus_labels",
            "positive_labels": ["Gesture"],
        },
        "predictions": {
            arm: {
                "labels": [dict(row) for row in label_rows],
                "behavior_view": project_teachobs_behavior_view(label_rows),
                "summary": {
                    "predicted_positive": 3,
                    "reference_positive": 1,
                    "true_positive": 1,
                    "false_positive": 2,
                    "false_negative": 0,
                },
            }
            for arm in PRIVATE_DASHBOARD_ARMS
        },
        "provenance": {
            "media_sha256": "a" * 12 + "…" + "a" * 8,
            "scene_manifest_sha256": "b" * 12 + "…" + "b" * 8,
            "frame_sha256": "c" * 12 + "…" + "c" * 8,
            "bundle_fingerprint": "d" * 12 + "…" + "d" * 8,
            "prediction_input_fingerprint": "e" * 12 + "…" + "e" * 8,
        },
        "claim_boundary": {
            "real_video": True,
            "real_materialized_transcript": True,
            "real_ocr": True,
            "ocr_displayed": False,
            "frozen_model_prediction": True,
            "teacher_behavior_view": True,
            "behavior_view_is_frozen_label_projection": True,
            "direct_student_action_recognition": False,
            "pose_tracking_performed": False,
            "live_feature_extraction": False,
            "ocr_accuracy_established": False,
            "action_subgroup_accuracy_established": False,
            "score_is_calibrated_probability": False,
            "deployment_accuracy_established": False,
        },
    }
    lesson = PrivateLessonSnapshot(
        lesson_id="S24",
        catalog_row={
            "id": "S24",
            "subject": "Mathematics",
            "school_level": "Secondary",
            "country": "United States",
            "source": "TeachObs",
            "duration_seconds": 15.0,
            "scene_count": 1,
            "suggested_scene": 1,
            "video": "media/S24",
            "captions": "captions/S24.vtt",
        },
        media_path=media_path,
        expected_media_sha256=(
            expected_media_sha256
            if expected_media_sha256 is not None
            else _digest(media_payload)
        ),
        scene_manifest_sha256="b" * 64,
        scenes=(scene,),
        frame_paths=(frame_path,),
        expected_frame_sha256=(
            expected_frame_sha256
            if expected_frame_sha256 is not None
            else _digest(frame_payload),
        ),
        captions_vtt=b"WEBVTT\n\n1\n00:00:00.000 --> 00:00:15.000\nCaption\n",
    )
    skill_payload = {
        "schema_version": "1.0",
        "kind": "local_private_distilled_skill",
        "video_id": "linear_algebra_l03",
        "skill": {
            "skill_id": "question_and_wait_linear_algebra_l03_v1",
            "name": "提问与等待：矩阵乘法",
            "goal": "帮助学习者建立 {concept} 的可解释心智模型。",
            "learning_objective": {
                "statement": "解释矩阵乘法并完成一道迁移题。",
                "bloom_level": "analyze",
                "assessment": "口头解释与迁移题。",
            },
            "parameters": {
                "concept": {
                    "type": "string",
                    "required": True,
                    "default": "矩阵乘法",
                },
                "learner_level": {
                    "type": "string",
                    "required": False,
                    "default": "beginner",
                },
            },
            "strategies": [],
            "procedure": [
                {
                    "step": 1,
                    "teaching_phase_name": "展示推导或操作",
                    "teacher_action": "demonstrate",
                    "instruction": "逐步展示推导。",
                    "expected_signal": "能说出下一步。",
                    "fallback": "只演示一个微步骤。",
                    "origin": "observed_method",
                    "evidence_ids": ["evi_1234567890abcdef"],
                    "observed_span": {"start": 8.0, "end": 12.0},
                }
            ],
            "verification": [
                {
                    "type": "explanation",
                    "prompt": "解释 {concept}。",
                    "pass_condition": "说明关键关系。",
                }
            ],
            "failure_modes": [],
        },
        "text_evidence": [
            {
                "evidence_id": "evi_1234567890abcdef",
                "start": 8.0,
                "end": 12.0,
                "quote": "A short caption excerpt.",
            }
        ],
        "multimodal_events": [],
        "evaluation": {
            "score_scope": "structural_quality_and_internal_evidence_consistency",
            "internal_overall_score": 94.4,
            "internal_grade": "A",
            "internal_passed": True,
            "internal_threshold": 75.0,
            "dimensions": {
                "structural_completeness": 100.0,
                "evidence_grounding": 100.0,
                "executability": 100.0,
                "method_fidelity": 74.4,
                "pedagogical_quality": 100.0,
                "generalizability": 100.0,
                "traceability": 100.0,
            },
            "weights": {
                "structural_completeness": 0.12,
                "evidence_grounding": 0.18,
                "executability": 0.18,
                "method_fidelity": 0.22,
                "pedagogical_quality": 0.12,
                "generalizability": 0.09,
                "traceability": 0.09,
            },
            "gates": {
                "schema_valid": True,
                "grounded": True,
                "executable": True,
                "testable": True,
                "multimodal_consistent": True,
                "method_distilled_from_video": True,
            },
            "method_fidelity": 74.4,
            "internal_evidence_consistency_score": 100.0,
            "internal_score_is_accuracy": False,
            "teaching_effectiveness_established": False,
            "real_world_recognition_accuracy_established": False,
        },
        "claim_boundary": {
            "procedure_is_text_led": True,
            "event_recognition_accuracy_established": False,
            "teaching_effectiveness_established": False,
        },
    }
    skill = PrivateSkillSnapshot(
        video_id="linear_algebra_l03",
        catalog_row={
            "id": "linear_algebra_l03",
            "course_id": "mit_1806",
            "title": "Lecture 3: Multiplication and Inverse Matrices",
            "duration_seconds": 2808.105,
            "candidate_event_count": 220,
            "selected_multimodal_event_count": 6,
            "observed_phase_count": 5,
            "recommended_phase_count": 4,
            "procedure_step_count": 9,
            "method_fidelity": 74.4,
            "detail": "api/skills/linear_algebra_l03",
        },
        payload=skill_payload,
    )
    return PrivateDashboardSnapshot(
        lessons={"S24": lesson},
        default_lesson="S24",
        bundle_fingerprint="d" * 64,
        transcript_fingerprint="f" * 64,
        prediction_input_fingerprint="e" * 64,
        prediction_matrix_sha256={arm: "1" * 64 for arm in PRIVATE_DASHBOARD_ARMS},
        template=private_dashboard_html_bytes(),
        skills={"linear_algebra_l03": skill},
        default_skill="linear_algebra_l03",
        skill_aggregate={
            "procedure_step_count": 9,
            "procedure_direct_mme_reference_count": 0,
            "observed_method_step_count": 5,
            "recommended_enrichment_step_count": 4,
        },
    )


class PrivateDashboardTemplateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.payload = private_dashboard_html_bytes()
        self.html = self.payload.decode("utf-8")

    def test_template_self_check_passes_without_private_material(self) -> None:
        report = private_dashboard_template_self_check()
        self.assertTrue(report["passed"], report)
        self.assertTrue(report["required_markers_present"], report)
        self.assertTrue(report["forbidden_markers_absent"], report)
        self.assertFalse(report["contains_private_data"])
        self.assertEqual(report["missing_markers"], [])
        self.assertEqual(report["forbidden_matches"], [])
        self.assertTrue(report["same_origin_assets_allowed"], report)
        self.assertEqual(report["size_bytes"], len(self.payload))
        self.assertEqual(report["sha256"], _digest(self.payload))

    def test_template_uses_safe_dynamic_rendering_and_no_external_assets(self) -> None:
        parser = _PrivateDashboardParser()
        parser.feed(self.html)
        self.assertEqual(parser.video_count, 1)
        self.assertEqual(parser.external_assets, [])
        for required in (
            "fetch(",
            ".textContent",
            "replaceChildren(",
            "真实 TeachObs 完整视频",
            "同时间窗字幕",
            "教师动作 / 学生状态",
            "教师侧可观察教学动作",
            "学生动作",
            "当前未建模",
            "冻结标签投影",
            "这不是把 OCR 改名为动作",
            "当前 15 秒场景的四臂预测",
            "TeachObs 行为识别",
            "MIT Skill 蒸馏",
            "九环节可执行 Skill",
            'data-screen-label="01 Behavior recognition"',
            'data-screen-label="02 Skill distillation"',
            'data-screen-label="03 Teaching process generation"',
            'data-screen-label="04 Automatic evaluation"',
            "private_skill_demo.css",
            "private_skill_demo.js",
            'aria-controls="skillOutcome"',
            'data-load-state="loading"',
            "正在加载真实 Skill",
            "用当前 Skill 生成一段可执行教学过程",
            "七维量表 + 六项门槛",
            "结构质量与内部证据一致性量表，不是 Accuracy",
        ):
            self.assertIn(required, self.html)
        for forbidden in (
            "innerHTML",
            "localStorage",
            "sessionStorage",
            "http://",
            "https://",
            'id="ocrText"',
            "同帧 OCR",
        ):
            self.assertNotIn(forbidden, self.html)

        assets = b"\n".join(
            private_dashboard_asset_bytes(name)
            for name in ("private_skill_demo.css", "private_skill_demo.js")
        ).decode("utf-8")
        self.assertIn("procedure-step", assets)
        self.assertIn("requestJson", assets)
        self.assertIn('window.location.hash === "#skill"', assets)
        self.assertIn("点击进入 Skill Demo", assets)
        self.assertIn('button.dataset.loadState = "ready"', assets)
        self.assertIn("generateTeachingProcess", assets)
        self.assertIn("renderEvaluation", assets)
        self.assertIn("fillTemplate", assets)
        self.assertIn("evaluationDimensions", assets)
        self.assertIn("evaluationGates", assets)
        for forbidden in (
            "innerHTML",
            "localStorage",
            "sessionStorage",
            "http://",
            "https://",
            "artifacts/private",
            "/Volumes/",
            "/Users/",
        ):
            self.assertNotIn(forbidden, assets)

    def test_template_states_the_claim_boundaries(self) -> None:
        for marker in (
            "不是实时部署推理",
            "post-test exploratory",
            "NON-CALIBRATED MODEL SCORE",
            "deployment_accuracy_established = false",
            "不提供上传、导出或外网请求",
            "不是新的人体动作模型",
            "这不是把 OCR 改名为动作",
            "没有学生角色跟踪、姿态模型或逐场景学生动作标注",
            "字幕证据主导",
            "直接 mme_* 引用数",
            "尚无独立事件真值、专家 Skill 金标准或学习效果实验",
            "生成在本机浏览器完成，不调用外部服务",
            "学生是否达标由演示者输入",
        ):
            self.assertIn(marker, self.html)

    def test_cli_check_template_is_nonblocking_and_path_free(self) -> None:
        with (
            patch(
                "teaching_skill_miner.cli.build_private_snapshot",
            ) as build,
            redirect_stdout(io.StringIO()) as output,
        ):
            result = main(["dashboard-real", "--check-template"])
        self.assertEqual(result, 0)
        build.assert_not_called()
        report = json.loads(output.getvalue())
        self.assertTrue(report["passed"], report)
        self.assertFalse(report["contains_private_data"])
        self.assertNotIn("/Volumes/", output.getvalue())
        self.assertNotIn("artifacts/private", output.getvalue())

    def test_cli_check_data_wires_both_private_roots(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root)
            teachobs_root = root / "teachobs-input"
            skill_root = root / "skill-input"
            with (
                patch(
                    "teaching_skill_miner.cli.build_private_snapshot",
                    return_value=snapshot,
                ) as build,
                redirect_stdout(io.StringIO()) as output,
            ):
                result = main(
                    [
                        "dashboard-real",
                        "--teachobs-root",
                        str(teachobs_root),
                        "--skill-root",
                        str(skill_root),
                        "--initial-lesson",
                        "S24",
                        "--initial-skill",
                        "linear_algebra_l03",
                        "--check-data",
                    ]
                )
            self.assertEqual(result, 0)
            build.assert_called_once_with(
                PrivateDashboardConfig(
                    root=teachobs_root,
                    initial_lesson="S24",
                    skill_root=skill_root,
                    initial_skill="linear_algebra_l03",
                )
            )
            report = json.loads(output.getvalue())
            self.assertEqual(report["product_count"], 2)
            self.assertTrue(report["real_skill_distillation"])
            self.assertEqual(
                report["file_integrity"]["validated_skill_artifact_count"], 1
            )


class ByteRangeTests(unittest.TestCase):
    def test_absent_and_ordinary_ranges(self) -> None:
        self.assertIsNone(parse_single_byte_range(None, 100))
        self.assertEqual(parse_single_byte_range("bytes=0-9", 100), (0, 9))
        self.assertEqual(parse_single_byte_range(" bytes=90-200 ", 100), (90, 99))

    def test_open_and_suffix_ranges(self) -> None:
        self.assertEqual(parse_single_byte_range("bytes=10-", 100), (10, 99))
        self.assertEqual(parse_single_byte_range("bytes=-10", 100), (90, 99))
        self.assertEqual(parse_single_byte_range("bytes=-500", 100), (0, 99))

    def test_invalid_and_multipart_ranges_fail_closed(self) -> None:
        invalid = (
            "bytes=",
            "items=0-1",
            "bytes=100-",
            "bytes=10-9",
            "bytes=-0",
            "bytes=0-1,4-5",
            "bytes=0-1,",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(RangeNotSatisfiable):
                parse_single_byte_range(value, 100)

    def test_zero_and_negative_size_handling(self) -> None:
        self.assertIsNone(parse_single_byte_range(None, 0))
        with self.assertRaisesRegex(RangeNotSatisfiable, "empty resource"):
            parse_single_byte_range("bytes=0-", 0)
        with self.assertRaisesRegex(ValueError, "nonnegative"):
            parse_single_byte_range(None, -1)


class BehaviorProjectionTests(unittest.TestCase):
    def test_projection_preserves_frozen_rows_without_inventing_actions(self) -> None:
        labels = [
            {
                "label": "Gesture",
                "group": "visual",
                "predicted": True,
                "score": 0.812345,
                "reference": False,
            },
            {
                "label": "Checking",
                "group": "nonvisual",
                "predicted": False,
                "score": 0.3125,
                "reference": True,
            },
            {
                "label": "Student raises hand",
                "group": "unknown",
                "predicted": True,
                "score": 0.999,
                "reference": True,
            },
        ]

        view = project_teachobs_behavior_view(labels)

        self.assertEqual(view["schema"], BEHAVIOR_PROJECTION_SCHEMA)
        self.assertEqual(view["student_actions"], [])
        self.assertEqual(
            view["student_action_status"],
            "not_modeled_no_scene_level_ground_truth",
        )
        self.assertFalse(view["direct_student_action_recognition"])
        self.assertFalse(view["pose_tracking_performed"])
        self.assertFalse(view["action_subgroup_accuracy_established"])
        teacher = view["teacher_visible_actions"]
        self.assertEqual(len(teacher), 1)
        self.assertEqual(teacher[0]["label"], "Gesture")
        self.assertEqual(teacher[0]["display_label"], "手势")
        self.assertEqual(teacher[0]["score"], labels[0]["score"])
        self.assertEqual(teacher[0]["predicted"], labels[0]["predicted"])
        self.assertEqual(teacher[0]["reference"], labels[0]["reference"])
        learner = view["learner_related_teaching_signals"]
        self.assertEqual(len(learner), 1)
        self.assertEqual(learner[0]["label"], "Checking")
        self.assertFalse(learner[0]["direct_student_action"])
        serialized = json.dumps(view, ensure_ascii=False)
        self.assertNotIn("Student raises hand", serialized)

    def test_unknown_labels_do_not_get_guessed_into_actor_groups(self) -> None:
        view = project_teachobs_behavior_view(
            [
                {
                    "label": "Student writing",
                    "group": "visual",
                    "predicted": True,
                    "score": 1.0,
                    "reference": True,
                }
            ]
        )
        self.assertEqual(view["teacher_visible_actions"], [])
        self.assertEqual(view["student_actions"], [])
        self.assertEqual(view["learner_related_teaching_signals"], [])


class PrivateSnapshotTests(unittest.TestCase):
    def test_catalog_scene_and_summary_are_path_free_and_truthful(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            snapshot = _make_snapshot(root)
            catalog = snapshot.catalog()
            scene = snapshot.scene("S24", 1)
            summary = private_snapshot_summary(snapshot)
            serialized = json.dumps(
                {"catalog": catalog, "scene": scene, "summary": summary},
                ensure_ascii=False,
            )

            self.assertNotIn(str(root), serialized)
            self.assertNotIn("opaque-lesson-video.mp4", serialized)
            self.assertNotIn("opaque-scene-frame.jpg", serialized)
            self.assertEqual(catalog["mode"], "local_private_real_data")
            self.assertTrue(catalog["privacy"]["loopback_only"])
            self.assertFalse(catalog["privacy"]["persistent_browser_storage"])
            self.assertFalse(catalog["privacy"]["private_files_copied"])
            self.assertFalse(catalog["inference"]["live_feature_extraction"])
            self.assertFalse(catalog["inference"]["deployment_accuracy_established"])
            self.assertFalse(catalog["inference"]["direct_student_action_recognition"])
            self.assertEqual(len(catalog["products"]), 2)
            self.assertTrue(catalog["skill_distillation"]["available"])
            self.assertEqual(
                catalog["skill_distillation"]["default_skill"],
                "linear_algebra_l03",
            )
            self.assertEqual(
                catalog["skill_distillation"]["relationship_to_recognition"],
                "independent_validation_track_not_direct_prediction_input",
            )
            self.assertTrue(scene["claim_boundary"]["real_video"])
            self.assertTrue(scene["claim_boundary"]["real_ocr"])
            self.assertFalse(scene["claim_boundary"]["ocr_displayed"])
            self.assertTrue(scene["claim_boundary"]["frozen_model_prediction"])
            self.assertTrue(scene["claim_boundary"]["teacher_behavior_view"])
            self.assertFalse(
                scene["claim_boundary"]["direct_student_action_recognition"]
            )
            self.assertFalse(scene["claim_boundary"]["live_feature_extraction"])
            self.assertTrue(scene["visual"]["ocr_used_as_frozen_model_input"])
            self.assertFalse(scene["visual"]["ocr_displayed"])
            self.assertNotIn('"ocr_text"', serialized.casefold())
            self.assertEqual(
                scene["predictions"]["full"]["behavior_view"]["student_actions"],
                [],
            )
            self.assertTrue(summary["private_paths_disclosed"] is False)
            self.assertTrue(summary["frozen_per_scene_predictions"])
            self.assertFalse(summary["ocr_displayed"])
            self.assertTrue(summary["teacher_behavior_projection"])
            self.assertFalse(summary["direct_student_action_recognition"])
            self.assertEqual(summary["product_count"], 2)
            self.assertEqual(summary["skill_lesson_count"], 1)
            self.assertEqual(summary["skill_procedure_step_count"], 9)
            self.assertTrue(summary["real_skill_distillation"])
            self.assertTrue(summary["skill_procedure_is_text_led"])
            self.assertEqual(summary["skill_direct_mme_reference_count"], 0)
            self.assertFalse(summary["expert_skill_gold_established"])
            self.assertFalse(summary["skill_accuracy_established"])
            self.assertFalse(summary["teaching_effectiveness_established"])
            self.assertFalse(summary["deployment_accuracy_established"])
            with self.assertRaises(KeyError):
                snapshot.scene("S24", 2)
            with self.assertRaises(KeyError):
                snapshot.scene("S25", 1)
            self.assertEqual(
                snapshot.skill("linear_algebra_l03")["video_id"],
                "linear_algebra_l03",
            )
            with self.assertRaises(KeyError):
                snapshot.skill("missing_skill")

    def test_media_and_frame_hashes_are_verified_before_release(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _make_snapshot(Path(directory))
            lesson = snapshot.lessons["S24"]
            lesson.verify_media()
            lesson.verify_frame(1)
            self.assertTrue(lesson._verified_media)
            self.assertEqual(lesson._verified_frames, {0})

    def test_media_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _make_snapshot(
                Path(directory),
                expected_media_sha256="0" * 64,
            )
            lesson = snapshot.lessons["S24"]
            with self.assertRaisesRegex(
                PrivateDashboardError, "media integrity mismatch"
            ):
                lesson.verify_media()
            self.assertFalse(lesson._verified_media)

    def test_frame_hash_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _make_snapshot(
                Path(directory),
                expected_frame_sha256="0" * 64,
            )
            lesson = snapshot.lessons["S24"]
            with self.assertRaisesRegex(
                PrivateDashboardError, "frame integrity mismatch"
            ):
                lesson.verify_frame(1)
            self.assertEqual(lesson._verified_frames, set())


class PrivateSkillArtifactTests(unittest.TestCase):
    @unittest.skipUnless(
        Path("artifacts/private/full_multimodal").is_dir(),
        "private long-form artifacts are not present",
    )
    def test_real_ten_lecture_skill_projection_is_complete_and_path_safe(self) -> None:
        skills, default_skill, aggregate = _build_private_skill_snapshots(
            Path("artifacts/private/full_multimodal"),
            initial_skill=DEFAULT_PRIVATE_SKILL,
        )

        self.assertEqual(len(skills), 10)
        self.assertEqual(default_skill, "linear_algebra_l03")
        self.assertEqual(aggregate["duration_hours"], 7.404675)
        self.assertEqual(aggregate["keyframe_count"], 2553)
        self.assertEqual(aggregate["candidate_multimodal_event_count"], 2270)
        self.assertEqual(aggregate["procedure_step_count"], 90)
        self.assertEqual(aggregate["observed_method_step_count"], 62)
        self.assertEqual(aggregate["recommended_enrichment_step_count"], 28)
        self.assertEqual(aggregate["text_evidence_count"], 200)
        self.assertEqual(aggregate["selected_multimodal_event_count"], 59)
        self.assertEqual(aggregate["verification_step_count"], 20)
        self.assertEqual(aggregate["procedure_direct_mme_reference_count"], 0)
        self.assertEqual(
            aggregate["selected_event_type_frequency"],
            {
                "code_formula_walkthrough": 15,
                "question_and_wait": 39,
                "visual_example": 5,
            },
        )
        observed_counts = {
            sum(
                step["origin"] == "observed_method"
                for step in snapshot.payload["skill"]["procedure"]
            )
            for snapshot in skills.values()
        }
        self.assertEqual(observed_counts, {5, 6, 7})
        self.assertTrue(
            all(
                len(snapshot.payload["skill"]["procedure"]) == 9
                for snapshot in skills.values()
            )
        )
        for snapshot in skills.values():
            parameters = snapshot.payload["skill"]["parameters"]
            self.assertEqual(parameters["concept"]["type"], "string")
            self.assertTrue(parameters["concept"]["default"])
            self.assertEqual(parameters["learner_level"]["default"], "beginner")
            evaluation = snapshot.payload["evaluation"]
            self.assertEqual(
                set(evaluation["dimensions"]),
                set(PRIVATE_SKILL_EVALUATION_DIMENSIONS),
            )
            self.assertEqual(
                evaluation["weights"],
                PRIVATE_SKILL_EVALUATION_DIMENSIONS,
            )
            self.assertEqual(
                set(evaluation["gates"]),
                set(PRIVATE_SKILL_EVALUATION_GATES),
            )
            self.assertTrue(all(evaluation["gates"].values()))
            self.assertTrue(evaluation["internal_passed"])
            self.assertEqual(evaluation["internal_threshold"], 75.0)
            weighted_score = round(
                sum(
                    evaluation["dimensions"][name] * weight
                    for name, weight in PRIVATE_SKILL_EVALUATION_DIMENSIONS.items()
                ),
                1,
            )
            self.assertEqual(evaluation["internal_overall_score"], weighted_score)
            self.assertEqual(
                evaluation["method_fidelity"],
                evaluation["dimensions"]["method_fidelity"],
            )
            self.assertFalse(evaluation["internal_score_is_accuracy"])
            self.assertFalse(evaluation["teaching_effectiveness_established"])
            self.assertFalse(evaluation["real_world_recognition_accuracy_established"])
        serialized = json.dumps(
            {
                "aggregate": aggregate,
                "skills": {key: value.payload for key, value in skills.items()},
            },
            ensure_ascii=False,
        )
        for forbidden in (
            "frame_path",
            "ocr_text",
            "source_url",
            "transcript_url",
            "jobs/",
            "artifacts/private",
            "/Volumes/",
            "/Users/",
            '"notes"',
            '"validation"',
            '"transcript_validation"',
            '"step_audit"',
            '"recognition_precision"',
            '"recognition_recall"',
            '"recognition_f1"',
        ):
            self.assertNotIn(forbidden, serialized)

    def test_skill_root_symlink_is_rejected_before_any_artifact_read(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            target = base / "real"
            target.mkdir()
            link = base / "linked"
            link.symlink_to(target, target_is_directory=True)
            with self.assertRaisesRegex(
                PrivateDashboardError,
                "Skill root may not be a symlink",
            ):
                _build_private_skill_snapshots(
                    link,
                    initial_skill=DEFAULT_PRIVATE_SKILL,
                )


class PrivateServerTests(unittest.TestCase):
    def test_live_loopback_server_serves_sanitized_skill_and_generic_assets(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _make_snapshot(Path(directory))
            try:
                server, url = create_private_dashboard_server(
                    snapshot,
                    capability_token=("LiveCapability" + "Token_123456"),
                )
            except PermissionError:
                self.skipTest("sandbox does not permit loopback socket binding")
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            parsed = urlsplit(url)
            connection = http.client.HTTPConnection(
                parsed.hostname,
                parsed.port,
                timeout=5,
            )
            try:
                connection.request(
                    "GET",
                    f"{parsed.path}api/skills/linear_algebra_l03",
                )
                response = connection.getresponse()
                payload = response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.getheader("Cache-Control"), "no-store, max-age=0"
                )
                content_security_policy = response.getheader("Content-Security-Policy")
                self.assertIn("default-src 'none'", content_security_policy)
                self.assertIn(
                    "style-src 'self' 'unsafe-inline'",
                    content_security_policy,
                )
                self.assertIn(
                    "script-src 'self' 'unsafe-inline'",
                    content_security_policy,
                )
                skill = json.loads(payload)
                self.assertEqual(skill["video_id"], "linear_algebra_l03")
                self.assertEqual(
                    skill["skill"]["parameters"]["concept"]["default"],
                    "矩阵乘法",
                )
                self.assertTrue(skill["evaluation"]["internal_passed"])
                self.assertEqual(
                    set(skill["evaluation"]["dimensions"]),
                    set(PRIVATE_SKILL_EVALUATION_DIMENSIONS),
                )
                self.assertEqual(
                    set(skill["evaluation"]["gates"]),
                    set(PRIVATE_SKILL_EVALUATION_GATES),
                )
                self.assertFalse(skill["evaluation"]["internal_score_is_accuracy"])
                self.assertNotIn("opaque-lesson-video.mp4", payload.decode("utf-8"))

                connection.request(
                    "GET",
                    f"{parsed.path}assets/private_skill_demo.js",
                )
                response = connection.getresponse()
                script = response.read()
                self.assertEqual(response.status, 200)
                self.assertEqual(
                    response.getheader("Content-Type"), "text/javascript; charset=utf-8"
                )
                self.assertIn(b"initializeSkillDemo", script)

                connection.request("GET", f"{parsed.path}api/skills/not_present")
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 404)

                connection.request("GET", "/api/skills/linear_algebra_l03")
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 403)

                connection.request("POST", parsed.path)
                response = connection.getresponse()
                response.read()
                self.assertEqual(response.status, 405)
            finally:
                connection.close()
                server.shutdown()
                server.server_close()
                worker.join(timeout=5)

    def test_server_is_fixed_to_loopback_with_random_capability_url(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _make_snapshot(Path(directory))
            server = MagicMock()
            server.server_port = 49152
            capability = "RandomCapabilityToken_123456"
            with (
                patch(
                    "teaching_skill_miner.private_dashboard.secrets.token_urlsafe",
                    return_value=capability,
                ) as random_token,
                patch(
                    "teaching_skill_miner.private_dashboard.ThreadingHTTPServer",
                    return_value=server,
                ) as server_type,
            ):
                actual_server, url = create_private_dashboard_server(snapshot)

            self.assertIs(actual_server, server)
            self.assertEqual(url, f"http://127.0.0.1:49152/{capability}/")
            random_token.assert_called_once_with(24)
            address, handler = server_type.call_args.args
            self.assertEqual(address, ("127.0.0.1", 0))
            self.assertTrue(callable(handler))
            self.assertTrue(server.daemon_threads)

    def test_server_uses_requested_port_and_explicit_capability(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _make_snapshot(Path(directory))
            server = MagicMock()
            server.server_port = 8765
            capability = "ExplicitCapability_12345"
            with (
                patch(
                    "teaching_skill_miner.private_dashboard.ThreadingHTTPServer",
                    return_value=server,
                ) as server_type,
                patch(
                    "teaching_skill_miner.private_dashboard.secrets.token_urlsafe",
                ) as random_token,
            ):
                _, url = create_private_dashboard_server(
                    snapshot,
                    port=8765,
                    capability_token=capability,
                )

            self.assertEqual(server_type.call_args.args[0], ("127.0.0.1", 8765))
            self.assertEqual(url, f"http://127.0.0.1:8765/{capability}/")
            random_token.assert_not_called()

    def test_invalid_port_and_capability_are_rejected_before_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _make_snapshot(Path(directory))
            invalid_ports = (-1, 65536, True, 1.5, "8000")
            with patch(
                "teaching_skill_miner.private_dashboard.ThreadingHTTPServer",
            ) as server_type:
                for port in invalid_ports:
                    with (
                        self.subTest(port=port),
                        self.assertRaisesRegex(
                            PrivateDashboardError,
                            "port is invalid",
                        ),
                    ):
                        create_private_dashboard_server(snapshot, port=port)  # type: ignore[arg-type]
                for token in ("short", "contains spaces and symbols!", "a" * 129):
                    with (
                        self.subTest(token=token),
                        self.assertRaisesRegex(
                            PrivateDashboardError,
                            "capability token is invalid",
                        ),
                    ):
                        create_private_dashboard_server(
                            snapshot,
                            capability_token=token,
                        )
            server_type.assert_not_called()

    def test_serve_lifecycle_is_nonblocking_and_stdout_is_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "do-not-disclose-private-root"
            snapshot = _make_snapshot(Path(directory))
            server = MagicMock()
            server.serve_forever.side_effect = KeyboardInterrupt
            url = "http://127.0.0.1:49152/CapabilityToken_123456/"
            with (
                patch(
                    "teaching_skill_miner.private_dashboard.build_private_snapshot",
                    return_value=snapshot,
                ) as build,
                patch(
                    "teaching_skill_miner.private_dashboard.create_private_dashboard_server",
                    return_value=(server, url),
                ) as create_server,
                patch(
                    "teaching_skill_miner.private_dashboard.webbrowser.open",
                    return_value=True,
                ) as browser,
                redirect_stdout(io.StringIO()) as output,
            ):
                result = serve_private_dashboard(
                    root,
                    initial_lesson="S24",
                    port=0,
                    open_browser=True,
                )

            self.assertEqual(result, 0)
            build.assert_called_once_with(
                PrivateDashboardConfig(root=root, initial_lesson="S24")
            )
            create_server.assert_called_once_with(snapshot, port=0)
            browser.assert_called_once_with(url)
            server.serve_forever.assert_called_once_with(poll_interval=0.25)
            server.shutdown.assert_called_once_with()
            server.server_close.assert_called_once_with()
            status = json.loads(output.getvalue())
            self.assertEqual(status["dashboard_url"], url)
            self.assertTrue(status["loopback_only"])
            self.assertEqual(status["cache_control"], "no-store")
            self.assertTrue(status["browser_open_requested"])
            self.assertNotIn(str(root), output.getvalue())
            self.assertNotIn(str(snapshot.lessons["S24"].media_path), output.getvalue())

    def test_serve_no_browser_skips_browser_and_still_closes_server(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            snapshot = _make_snapshot(Path(directory))
            server = MagicMock()
            url = "http://127.0.0.1:49152/CapabilityToken_123456/"
            with (
                patch(
                    "teaching_skill_miner.private_dashboard.build_private_snapshot",
                    return_value=snapshot,
                ),
                patch(
                    "teaching_skill_miner.private_dashboard.create_private_dashboard_server",
                    return_value=(server, url),
                ),
                patch(
                    "teaching_skill_miner.private_dashboard.webbrowser.open",
                ) as browser,
                redirect_stdout(io.StringIO()),
            ):
                result = serve_private_dashboard(
                    Path(directory),
                    open_browser=False,
                )

            self.assertEqual(result, 0)
            browser.assert_not_called()
            server.serve_forever.assert_called_once_with(poll_interval=0.25)
            server.shutdown.assert_called_once_with()
            server.server_close.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
