from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from teaching_skill_miner import teacher_agent_vision as vision
from teaching_skill_miner.teacher_agent_vision import (
    LocalVisualEvidenceError,
    compose_visual_evidence_text,
    extract_local_visual_evidence,
)


_SYNTHETIC_PNG = b"\x89PNG\r\n\x1a\nsynthetic-no-personal-data"


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
