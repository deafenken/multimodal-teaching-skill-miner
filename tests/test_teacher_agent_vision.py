from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image, ImageDraw

from teaching_skill_miner import teacher_agent_vision as vision
from teaching_skill_miner.teacher_agent_vision import (
    LocalVisualEvidenceError,
    align_ocr_text_to_answer_references,
    assess_typed_visual_consistency,
    compose_visual_evidence_text,
    extract_local_visual_evidence,
)


_SYNTHETIC_PNG = b"\x89PNG\r\n\x1a\nsynthetic-no-personal-data"


def _printed_formula_png() -> bytes:
    image = Image.new("RGB", (1_600, 320), "white")
    ImageDraw.Draw(image).text(
        (120, 110),
        "dp[i] = dp[i-1] + dp[i-2]",
        fill="black",
        stroke_width=1,
    )
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


class TeacherAgentVisionTests(unittest.TestCase):
    def test_text_only_turn_preserves_plain_learner_text(self) -> None:
        self.assertEqual(
            compose_visual_evidence_text("  答案是：缓存。  ", []),
            "答案是：缓存。",
        )
        self.assertEqual(
            compose_visual_evidence_text("什么是动态规划", iter(())),
            "什么是动态规划",
        )

    def test_visual_envelope_is_added_only_when_evidence_exists(self) -> None:
        composed = compose_visual_evidence_text(
            "我在图片中写了推导",
            [
                {
                    "recognized_text": "f[i] = min(f[j] + 1)",
                    "status": "recognized",
                    "confidence": 0.91,
                    "needs_student_confirmation": False,
                }
            ],
        )

        self.assertIn("[STUDENT_TYPED_TEXT]\n我在图片中写了推导", composed)
        self.assertIn("[LOCAL_VISUAL_EVIDENCE 1]", composed)
        self.assertIn("confidence=0.91", composed)
        self.assertIn("f[i] = min(f[j] + 1)", composed)

    def test_typed_and_visual_conflict_is_explicit_and_never_silently_merged(
        self,
    ) -> None:
        evidence = [
            {
                "recognized_text": "x+1",
                "status": "recognized",
                "confidence": 0.97,
                "transcription_confidence": 0.96,
                "ocr_transcription_corroborated": True,
                "needs_student_confirmation": False,
            }
        ]

        consistency = assess_typed_visual_consistency("x-1", evidence)
        composed = compose_visual_evidence_text("x-1", evidence)

        self.assertEqual(consistency["relation"], "possible_conflict")
        self.assertTrue(consistency["needs_student_confirmation"])
        self.assertIn("[TYPED_VISUAL_CONSISTENCY]", composed)
        self.assertIn("不得把两者合并成一个答案", composed)

    def test_identical_typed_and_visual_answers_are_not_flagged_as_conflict(
        self,
    ) -> None:
        evidence = [
            {
                "recognized_text": "dp[i] = dp[i-1] + dp[i-2]",
                "status": "recognized",
                "needs_student_confirmation": False,
            }
        ]

        consistency = assess_typed_visual_consistency(
            "答案是：dp[i]=dp[i-1]+dp[i-2]",
            evidence,
        )

        self.assertEqual(consistency["relation"], "exact_agreement")
        self.assertFalse(consistency["possible_conflict"])

    def test_one_conflicting_attachment_cannot_be_hidden_by_one_matching_image(
        self,
    ) -> None:
        evidence = [
            {
                "recognized_text": text,
                "status": "recognized",
                "needs_student_confirmation": False,
            }
            for text in ("x+1", "x-1")
        ]

        consistency = assess_typed_visual_consistency("x+1", evidence)

        self.assertEqual(consistency["relation"], "possible_conflict")
        self.assertTrue(consistency["needs_student_confirmation"])

    def test_apple_vision_is_primary_and_raw_image_is_temporary(self) -> None:
        observed: dict[str, Path] = {}

        def fake_apple(
            executable: str,
            image_path: Path,
            *,
            working_dir: Path,
        ) -> tuple[str, float]:
            self.assertEqual(executable, "/usr/bin/clang")
            self.assertEqual(image_path.parent, working_dir)
            self.assertEqual(image_path.read_bytes(), _SYNTHETIC_PNG)
            observed["image_path"] = image_path
            return "牛顿第二定律", 0.96

        with (
            patch.object(
                vision, "_apple_vision_command", return_value="/usr/bin/clang"
            ),
            patch.object(vision, "_tesseract_command", return_value="tesseract"),
            patch.object(vision, "_run_apple_vision", side_effect=fake_apple),
            patch.object(vision, "_run_tesseract") as tesseract,
        ):
            result = extract_local_visual_evidence(
                _SYNTHETIC_PNG,
                "image/png",
                display_name="../student-answer.png",
            )

        tesseract.assert_not_called()
        self.assertFalse(observed["image_path"].exists())
        self.assertEqual(result["engine"], "apple_vision_local_objc_cli")
        self.assertEqual(result["status"], "recognized")
        self.assertEqual(result["recognized_text"], "牛顿第二定律")
        self.assertEqual(result["display_name"], "student-answer.png")
        self.assertFalse(result["raw_media_retained"])
        self.assertFalse(result["remote_media_sent"])
        self.assertNotIn(str(observed["image_path"]), json.dumps(result))

    def test_apple_failure_falls_back_to_tesseract(self) -> None:
        with (
            patch.object(
                vision, "_apple_vision_command", return_value="/usr/bin/clang"
            ),
            patch.object(vision, "_tesseract_command", return_value="tesseract"),
            patch.object(
                vision,
                "_run_apple_vision",
                side_effect=LocalVisualEvidenceError("synthetic failure"),
            ),
            patch.object(
                vision,
                "_run_tesseract",
                side_effect=[("", 0.0), ("状态转移", 0.88)],
            ) as tesseract,
        ):
            result = extract_local_visual_evidence(_SYNTHETIC_PNG, "image/png")

        self.assertEqual(tesseract.call_count, 2)
        self.assertEqual(result["engine"], "tesseract_local_cli_fallback")
        self.assertTrue(result["extractor_fallback_used"])
        self.assertEqual(result["status"], "recognized")
        self.assertEqual(result["recognized_text"], "状态转移")

    def test_clear_printed_formula_is_corroborated_across_local_ocr_routes(
        self,
    ) -> None:
        formula = "dp[i] = dp[i-1] + dp[i-2]"

        def fake_tesseract(
            _executable: str,
            _image_path: Path,
            *,
            page_segmentation_mode: int,
        ) -> tuple[str, float]:
            return formula, 0.94 if page_segmentation_mode != 11 else 0.92

        with (
            patch.object(
                vision, "_apple_vision_command", return_value="/usr/bin/clang"
            ),
            patch.object(vision, "_tesseract_command", return_value="tesseract"),
            patch.object(
                vision,
                "_run_apple_vision",
                return_value=(formula, 0.98),
            ),
            patch.object(
                vision,
                "_run_tesseract",
                side_effect=fake_tesseract,
            ) as tesseract,
        ):
            result = extract_local_visual_evidence(
                _printed_formula_png(),
                "image/png",
            )

        self.assertGreaterEqual(tesseract.call_count, 4)
        self.assertEqual(result["status"], "recognized")
        self.assertEqual(result["recognized_text"], formula)
        self.assertTrue(result["formula_like_text_detected"])
        self.assertTrue(result["ocr_transcription_corroborated"])
        self.assertTrue(result["formula_transcription_established"])
        self.assertFalse(result["formula_accuracy_established"])
        self.assertFalse(result["needs_student_confirmation"])
        self.assertGreaterEqual(result["ocr_independent_engine_count"], 2)
        self.assertGreaterEqual(result["ocr_preprocessing_count"], 2)
        self.assertEqual(result["content_style_assessment"], "not_classified")
        self.assertFalse(result["remote_media_sent"])

    def test_low_confidence_handwritten_like_formula_abstains(self) -> None:
        weak_candidates = iter(
            [
                ("x+?", 0.43),
                ("x+7", 0.38),
                ("", 0.0),
                ("x + l", 0.41),
                ("", 0.0),
                ("x+1", 0.45),
            ]
        )
        with (
            patch.object(
                vision, "_apple_vision_command", return_value="/usr/bin/clang"
            ),
            patch.object(vision, "_tesseract_command", return_value="tesseract"),
            patch.object(
                vision,
                "_run_apple_vision",
                return_value=("x+?", 0.42),
            ),
            patch.object(
                vision,
                "_run_tesseract",
                side_effect=lambda *_args, **_kwargs: next(weak_candidates),
            ),
        ):
            result = extract_local_visual_evidence(
                _printed_formula_png(),
                "image/png",
            )

        self.assertIn(
            result["status"],
            {"low_confidence", "conflicting_recognition"},
        )
        self.assertFalse(result["formula_transcription_established"])
        self.assertFalse(result["handwriting_recognition_established"])
        self.assertTrue(result["needs_student_confirmation"])

    def test_high_confidence_independent_engine_conflict_forces_confirmation(
        self,
    ) -> None:
        tesseract_candidates = iter(
            [
                ("x+1", 0.96),
                ("x+1", 0.96),
                ("x+1", 0.96),
                ("", 0.0),
                ("", 0.0),
                ("", 0.0),
            ]
        )
        with (
            patch.object(
                vision, "_apple_vision_command", return_value="/usr/bin/clang"
            ),
            patch.object(vision, "_tesseract_command", return_value="tesseract"),
            patch.object(
                vision,
                "_run_apple_vision",
                return_value=("x-1", 0.99),
            ),
            patch.object(
                vision,
                "_run_tesseract",
                side_effect=lambda *_args, **_kwargs: next(tesseract_candidates),
            ),
        ):
            result = extract_local_visual_evidence(
                _printed_formula_png(),
                "image/png",
            )

        self.assertEqual(result["status"], "conflicting_recognition")
        self.assertTrue(result["ocr_material_disagreement"])
        self.assertFalse(result["formula_transcription_established"])
        self.assertTrue(result["needs_student_confirmation"])

    def test_formula_like_ocr_always_requires_student_confirmation(self) -> None:
        with (
            patch.object(
                vision, "_apple_vision_command", return_value="/usr/bin/clang"
            ),
            patch.object(vision, "_tesseract_command", return_value=None),
            patch.object(
                vision,
                "_run_apple_vision",
                return_value=("x = (-b + sqrt(b^2 - 4ac)) / (2a)", 0.99),
            ),
        ):
            result = extract_local_visual_evidence(_SYNTHETIC_PNG, "image/png")

        self.assertEqual(result["status"], "recognized")
        self.assertTrue(result["formula_like_text_detected"])
        self.assertFalse(result["formula_accuracy_established"])
        self.assertTrue(result["needs_student_confirmation"])
        self.assertIn("not_formula_correctness", result["confidence_semantics"])

    def test_common_math_and_code_ocr_require_student_confirmation(self) -> None:
        expressions = (
            "x+1",
            "f(n)",
            "dp[i]",
            "a/b",
            "x<y",
            "2x+3",
            "if (x > 0) return x;",
        )
        for expression in expressions:
            with (
                self.subTest(expression=expression),
                patch.object(
                    vision, "_apple_vision_command", return_value="/usr/bin/clang"
                ),
                patch.object(vision, "_tesseract_command", return_value=None),
                patch.object(
                    vision,
                    "_run_apple_vision",
                    return_value=(expression, 0.99),
                ),
            ):
                result = extract_local_visual_evidence(
                    _SYNTHETIC_PNG,
                    "image/png",
                )

            self.assertEqual(result["status"], "recognized")
            self.assertTrue(result["formula_like_text_detected"])
            self.assertFalse(result["formula_accuracy_established"])
            self.assertTrue(result["needs_student_confirmation"])

    def test_image_only_formula_can_match_teacher_question_contract_exactly(
        self,
    ) -> None:
        alignment = align_ocr_text_to_answer_references(
            "dp[i] = dp[i-1] + dp[i-2]",
            {
                "answer_type": "worked_step",
                "target_concepts": ["dp[i]=dp[i-1]+dp[i-2]"],
                "accepted_aliases": [],
                "success_criteria": ["写出状态转移式"],
            },
        )

        self.assertEqual(alignment["alignment"], "exact_contract_match")
        self.assertEqual(
            alignment["matched_source"],
            "question_contract.target_concepts",
        )
        self.assertTrue(alignment["deterministic_correctness_established"])
        self.assertTrue(alignment["requires_reliable_transcription"])

    def test_related_open_target_is_not_promoted_to_deterministically_correct(
        self,
    ) -> None:
        alignment = align_ocr_text_to_answer_references(
            "状态转移",
            {
                "answer_type": "explanation",
                "target_concepts": ["状态转移"],
                "accepted_aliases": [],
                "success_criteria": ["解释两个来源为何相加"],
            },
        )

        self.assertEqual(alignment["alignment"], "exact_contract_match")
        self.assertFalse(alignment["deterministic_correctness_established"])

    def test_rubric_fragment_matches_but_does_not_alone_establish_correctness(
        self,
    ) -> None:
        alignment = align_ocr_text_to_answer_references(
            "dp[0]=1 和 dp[1]=1",
            {
                "answer_type": "open",
                "target_concepts": ["边界条件"],
                "accepted_aliases": [],
                "success_criteria": ["给出正确边界"],
            },
            {
                "rubric_criteria": [
                    {
                        "criterion_id": "boundary",
                        "knowledge_component": "边界条件",
                        "acceptable_evidence": ["dp[0]=1 和 dp[1]=1"],
                    }
                ]
            },
            ["边界条件"],
        )

        self.assertEqual(
            alignment["alignment"],
            "exact_teacher_reference_match",
        )
        self.assertEqual(
            alignment["matched_source"],
            "knowledge_spec.rubric_criteria.boundary",
        )
        self.assertFalse(alignment["deterministic_correctness_established"])

    def test_exact_teacher_canonical_claim_can_ground_image_only_answer(self) -> None:
        claim = "采用 dp[0]=1、dp[1]=1 的约定后，从小到大计算后续状态。"
        alignment = align_ocr_text_to_answer_references(
            claim,
            {
                "answer_type": "open",
                "target_concepts": ["边界条件"],
                "accepted_aliases": [],
                "success_criteria": ["给出边界与计算顺序"],
            },
            {
                "canonical_claims": [
                    {
                        "claim_id": "boundary_claim",
                        "statement": claim,
                        "knowledge_components": ["边界条件"],
                    }
                ]
            },
            ["边界条件"],
        )

        self.assertEqual(
            alignment["alignment"],
            "exact_teacher_reference_match",
        )
        self.assertEqual(
            alignment["matched_source"],
            "knowledge_spec.canonical_claims.boundary_claim",
        )
        self.assertTrue(alignment["deterministic_correctness_established"])

    def test_teacher_claim_from_another_knowledge_component_cannot_ground_answer(
        self,
    ) -> None:
        claim = "状态转移是 dp[i]=dp[i-1]+dp[i-2]。"
        alignment = align_ocr_text_to_answer_references(
            claim,
            {
                "answer_type": "open",
                "target_concepts": ["边界条件"],
                "accepted_aliases": [],
                "success_criteria": ["给出边界与计算顺序"],
            },
            {
                "canonical_claims": [
                    {
                        "claim_id": "transition_claim",
                        "statement": claim,
                        "knowledge_components": ["状态转移"],
                    }
                ]
            },
            ["边界条件"],
        )

        self.assertEqual(alignment["alignment"], "not_established")
        self.assertFalse(alignment["deterministic_correctness_established"])

    def test_unscoped_teacher_claim_cannot_ground_answer(self) -> None:
        claim = "采用 dp[0]=1、dp[1]=1 的约定后，从小到大计算后续状态。"
        alignment = align_ocr_text_to_answer_references(
            claim,
            {"answer_type": "open", "target_concepts": ["边界条件"]},
            {"canonical_claims": [{"claim_id": "unscoped", "statement": claim}]},
            ["边界条件"],
        )

        self.assertEqual(alignment["alignment"], "not_established")
        self.assertFalse(alignment["deterministic_correctness_established"])

    def test_multiline_ocr_with_unrelated_line_does_not_exact_match_claim(self) -> None:
        claim = "采用 dp[0]=1、dp[1]=1 的约定后，从小到大计算后续状态。"
        alignment = align_ocr_text_to_answer_references(
            "与本问无关的状态转移说明。\n" + claim,
            {"answer_type": "open", "target_concepts": ["边界条件"]},
            {
                "canonical_claims": [
                    {
                        "claim_id": "boundary_claim",
                        "statement": claim,
                        "knowledge_components": ["边界条件"],
                    }
                ]
            },
            ["边界条件"],
        )

        self.assertEqual(alignment["alignment"], "not_established")

    def test_tesseract_candidate_selection_is_confidence_first(self) -> None:
        high_confidence_answer = "缓存"
        low_confidence_garbage = "x" * 120
        with (
            patch.object(vision, "_apple_vision_command", return_value=None),
            patch.object(vision, "_tesseract_command", return_value="tesseract"),
            patch.object(
                vision,
                "_run_tesseract",
                side_effect=[
                    (high_confidence_answer, 0.95),
                    (low_confidence_garbage, 0.59),
                ],
            ),
        ):
            result = extract_local_visual_evidence(_SYNTHETIC_PNG, "image/png")

        self.assertEqual(result["recognized_text"], high_confidence_answer)
        self.assertEqual(result["confidence"], 0.95)
        self.assertEqual(result["status"], "recognized")
        self.assertFalse(result["needs_student_confirmation"])

    def test_tesseract_character_quality_breaks_equal_confidence_tie(self) -> None:
        high_quality_answer = "缓存"
        repeated_garbage = "x" * 120
        with (
            patch.object(vision, "_apple_vision_command", return_value=None),
            patch.object(vision, "_tesseract_command", return_value="tesseract"),
            patch.object(
                vision,
                "_run_tesseract",
                side_effect=[
                    (high_quality_answer, 0.9),
                    (repeated_garbage, 0.9),
                ],
            ),
        ):
            result = extract_local_visual_evidence(_SYNTHETIC_PNG, "image/png")

        self.assertEqual(result["recognized_text"], high_quality_answer)
        self.assertEqual(result["confidence"], 0.9)

    def test_tesseract_subprocess_receives_only_minimal_environment(self) -> None:
        captured: dict[str, object] = {}
        tsv = (
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\t"
            "left\ttop\twidth\theight\tconf\ttext\n"
            "5\t1\t1\t1\t1\t1\t0\t0\t10\t10\t94\tcache\n"
        ).encode("utf-8")

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            captured["args"] = args
            captured["env"] = kwargs["env"]
            return subprocess.CompletedProcess(args, 0, stdout=tsv, stderr=b"")

        with tempfile.TemporaryDirectory(prefix="tsm-tesseract-env-test-") as tmp_dir:
            image_path = Path(tmp_dir) / "answer.png"
            image_path.write_bytes(_SYNTHETIC_PNG)
            with (
                patch.dict(
                    vision.os.environ,
                    {
                        "DEEPSEEK_API_KEY": "must-not-be-forwarded",
                        "UNRELATED_SECRET": "must-not-be-forwarded",
                        "TESSDATA_PREFIX": "/opt/tessdata",
                    },
                ),
                patch.object(vision.subprocess, "run", side_effect=fake_run),
            ):
                text, confidence = vision._run_tesseract(
                    "/opt/homebrew/bin/tesseract",
                    image_path,
                    page_segmentation_mode=6,
                )

        environment = captured["env"]
        self.assertIsInstance(environment, dict)
        assert isinstance(environment, dict)
        self.assertNotIn("DEEPSEEK_API_KEY", environment)
        self.assertNotIn("UNRELATED_SECRET", environment)
        self.assertEqual(environment["TESSDATA_PREFIX"], "/opt/tessdata")
        self.assertEqual(environment["TMPDIR"], str(image_path.parent))
        self.assertEqual(text, "cache")
        self.assertEqual(confidence, 0.94)

    def test_swift_subprocess_receives_only_minimal_environment(self) -> None:
        captured: dict[str, object] = {}

        calls: list[list[str]] = []

        def fake_run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
            calls.append(args)
            captured["env"] = kwargs["env"]
            if "-o" in args:
                Path(args[args.index("-o") + 1]).write_bytes(b"synthetic-helper")
                return subprocess.CompletedProcess(args, 0, stdout=b"", stderr=b"")
            payload = json.dumps({"text": "Printed answer", "confidence": 0.91}).encode(
                "utf-8"
            )
            return subprocess.CompletedProcess(args, 0, stdout=payload, stderr=b"")

        with tempfile.TemporaryDirectory(prefix="tsm-vision-test-") as tmp_dir:
            working_dir = Path(tmp_dir)
            image_path = working_dir / "answer.png"
            image_path.write_bytes(_SYNTHETIC_PNG)
            with (
                patch.dict(
                    vision.os.environ,
                    {
                        "DEEPSEEK_API_KEY": "must-not-be-forwarded",
                        "UNRELATED_SECRET": "must-not-be-forwarded",
                    },
                ),
                patch.object(vision.subprocess, "run", side_effect=fake_run),
            ):
                text, confidence = vision._run_apple_vision(
                    "/usr/bin/clang",
                    image_path,
                    working_dir=working_dir,
                )

        environment = captured["env"]
        self.assertIsInstance(environment, dict)
        assert isinstance(environment, dict)
        self.assertNotIn("DEEPSEEK_API_KEY", environment)
        self.assertNotIn("UNRELATED_SECRET", environment)
        self.assertEqual(text, "Printed answer")
        self.assertEqual(confidence, 0.91)
        self.assertEqual(len(calls), 2)
        self.assertNotIn(str(image_path), calls[0])
        self.assertEqual(calls[1][-1], str(image_path))


if __name__ == "__main__":
    unittest.main()
