from __future__ import annotations

import tempfile
import unittest
import csv
import copy
from pathlib import Path
from unittest.mock import patch

from teaching_skill_miner.audit import audit_dataset
from teaching_skill_miner.benchmark import benchmark_transfer
from teaching_skill_miner.cli import main
from teaching_skill_miner.evaluator import evaluate_collection, evaluate_skill
from teaching_skill_miner.executor import execute_skill
from teaching_skill_miner.human_eval import (
    skill_review_fingerprint,
    summarize_human_review,
)
from teaching_skill_miner.io_utils import project_root, read_json, write_json
from teaching_skill_miner.miner import mine_skill
from teaching_skill_miner.models import validate_skill, validate_transcript
from teaching_skill_miner.multimodal import fuse_multimodal_events, infer_visual_events, parse_silencedetect
from teaching_skill_miner.multimodal_benchmark import evaluate_multimodal_fixture
from teaching_skill_miner.preprocess import parse_srt_or_vtt, preprocess_file
from teaching_skill_miner.runtime import SkillRuntime, run_scripted_session


class PreprocessTests(unittest.TestCase):
    def test_parse_srt(self) -> None:
        segments = parse_srt_or_vtt(
            "1\n00:00:01,000 --> 00:00:03,500\nWhat happens next?\n\n"
            "2\n00:00:04,000 --> 00:00:06,000\nTry an example.\n"
        )
        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0]["start"], 1.0)
        self.assertEqual(segments[0]["end"], 3.5)

    def test_parse_vtt_timestamp_without_hours(self) -> None:
        segments = parse_srt_or_vtt("WEBVTT\n\n00:01.000 --> 00:03.000\nA short cue.\n")
        self.assertEqual(segments, [{"start": 1.0, "end": 3.0, "text": "A short cue."}])

    def test_plain_text_preprocess(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.txt"
            path.write_text("First, try an example. Then check it. Why does it work?", encoding="utf-8")
            transcript = preprocess_file(
                path,
                video_id="v1",
                course_id="c1",
                title="A lesson",
                source_url="https://example.org/video",
            )
            self.assertTrue(validate_transcript(transcript).valid)
            self.assertEqual(len(transcript["provenance"]["input_sha256"]), 64)

    def test_parse_audio_silences(self) -> None:
        log = "silence_start: 2.0\nsilence_end: 4.5 | silence_duration: 2.5\n"
        self.assertEqual(parse_silencedetect(log, 8.0), [{"start": 2.0, "end": 4.5, "duration": 2.5}])

    def test_visual_event_inference_from_ocr(self) -> None:
        events = infer_visual_events(
            [
                {"timestamp": 0.0, "path": "frames/a.jpg", "sampling_source": "uniform", "ocr_text": "Example x"},
                {"timestamp": 5.0, "path": "frames/b.jpg", "sampling_source": "scene", "ocr_text": "def solve(x): return x"},
            ]
        )
        types = {event["type"] for event in events}
        self.assertIn("scene_change", types)
        self.assertIn("code_or_formula_visible", types)

    def test_question_wait_fuses_speech_and_audio(self) -> None:
        transcript = {
            "segments": [{"start": 0.0, "end": 2.0, "text": "Why does this happen?"}]
        }
        events = fuse_multimodal_events(
            transcript,
            [{"start": 2.0, "end": 4.5, "duration": 2.5}],
            [],
        )
        self.assertEqual(events[0]["type"], "question_and_wait")
        self.assertEqual(events[0]["modalities"], ["speech", "audio"])

    def test_question_wait_counts_only_silence_after_question(self) -> None:
        events = fuse_multimodal_events(
            {"segments": [{"start": 0.0, "end": 2.0, "text": "Why does this happen?"}]},
            [{"start": 0.5, "end": 3.0, "duration": 2.5}],
            [],
        )
        self.assertEqual(events[0]["evidence"]["wait_seconds"], 1.0)
        self.assertEqual(events[0]["evidence"]["wait_interval"], {"start": 2.0, "end": 3.0})

    def test_scene_change_plus_code_speech_is_not_code_walkthrough(self) -> None:
        events = fuse_multimodal_events(
            {"segments": [{"start": 0.0, "end": 3.0, "text": "Read the code line by line."}]},
            [],
            [
                {
                    "type": "scene_change",
                    "start": 1.0,
                    "end": 1.0,
                    "modalities": ["visual"],
                    "evidence": {"frame_path": "frames/a.jpg", "ocr_text": ""},
                    "confidence": 0.75,
                }
            ],
        )
        self.assertNotIn("code_formula_walkthrough", {event["type"] for event in events})

    def test_classroom_observations_reject_identity(self) -> None:
        with self.assertRaises(ValueError):
            fuse_multimodal_events(
                {"segments": []},
                [],
                [],
                [{"type": "student_answer", "start": 1, "face_id": "person-7"}],
            )

    def test_multimodal_validator_rejects_fake_type_and_confidence(self) -> None:
        transcript = {
            "video_id": "v1",
            "course_id": "c1",
            "title": "Synthetic",
            "source_url": "https://example.org/v1",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "First."},
                {"start": 1.0, "end": 2.0, "text": "Second."},
                {"start": 2.0, "end": 3.0, "text": "Third."},
            ],
            "multimodal": {
                "modalities_available": ["visual"],
                "media": {"duration_seconds": 3.0},
                "visual": {"keyframes": []},
                "audio": {"silences": []},
                "events": [
                    {
                        "event_id": "bad",
                        "type": "completely_fake",
                        "start": 0.0,
                        "end": 9.0,
                        "modalities": ["visual"],
                        "evidence": {},
                        "confidence": 999,
                        "supports_strategies": ["adaptive_teaching"],
                    }
                ],
            },
        }
        result = validate_transcript(transcript)
        self.assertFalse(result.valid)
        self.assertTrue(any("unsupported" in error for error in result.errors))

    def test_synthetic_fixture_benchmark_labels_scope(self) -> None:
        analysis = {
            "events": [
                {"type": "question_and_wait", "start": 0.0, "end": 3.5, "modalities": ["speech", "audio"]}
            ],
            "audio": {"silences": [{"start": 1.5, "end": 3.5, "duration": 2.0}]},
            "visual": {"keyframes": [{"timestamp": 0.0, "ocr_text": "Concrete Example"}]},
        }
        ground_truth = {
            "fixture_id": "unit",
            "events": [{"type": "question_and_wait", "start": 0.0, "end": 3.5}],
            "silences": [{"start": 1.5, "end": 3.5}],
            "ocr_checks": [{"timestamp": 0.0, "required_tokens": ["concrete", "example"]}],
        }
        report = evaluate_multimodal_fixture(analysis, ground_truth)
        self.assertTrue(report["passed"])
        self.assertEqual(report["event_metrics"]["f1"], 1.0)
        self.assertFalse(report["real_world_accuracy_established"])


class MiningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = project_root()
        cls.transcript = read_json(cls.root / "data/transcripts/python_l03.json")
        cls.skill = mine_skill(cls.transcript)

    def test_generated_skill_is_valid(self) -> None:
        result = validate_skill(self.skill)
        self.assertTrue(result.valid, result.errors)
        self.assertGreaterEqual(len(self.skill["procedure"]), 4)

    def test_runtime_validation_rejects_legacy_skill_without_provenance(self) -> None:
        legacy = copy.deepcopy(self.skill)
        legacy["procedure"][0].pop("origin")
        legacy["procedure"][0].pop("evidence_ids")
        legacy["procedure"][0].pop("provenance")
        result = validate_skill(legacy)
        self.assertFalse(result.valid)
        self.assertTrue(any("procedure[0]" in error for error in result.errors))

    def test_evidence_is_grounded(self) -> None:
        report = evaluate_skill(self.skill, self.transcript)
        self.assertTrue(report["passed"])
        self.assertGreaterEqual(report["dimensions"]["evidence_grounding"], 60)

    def test_evidence_ids_and_procedure_origins_are_stable(self) -> None:
        repeated = mine_skill(self.transcript)
        evidence_ids = [
            item["evidence_id"] for item in self.skill["source"]["evidence"]
        ]
        self.assertEqual(
            evidence_ids,
            [item["evidence_id"] for item in repeated["source"]["evidence"]],
        )
        self.assertEqual(len(evidence_ids), len(set(evidence_ids)))
        self.assertTrue(all(value.startswith("evi_") for value in evidence_ids))

        known_ids = set(evidence_ids)
        observed_steps = []
        for step in self.skill["procedure"]:
            self.assertIn(
                step["origin"], {"observed_method", "recommended_enrichment"}
            )
            self.assertEqual(step["origin"], step["provenance"]["origin"])
            self.assertEqual(
                step["evidence_ids"], step["provenance"]["evidence_ids"]
            )
            if step["origin"] == "observed_method":
                observed_steps.append(step)
                self.assertTrue(set(step["evidence_ids"]) <= known_ids)
            else:
                self.assertEqual(step["evidence_ids"], [])
        self.assertTrue(observed_steps)

        report = evaluate_skill(self.skill, self.transcript)
        provenance = report["method_provenance"]
        self.assertEqual(
            provenance["observed_method_step_count"], len(observed_steps)
        )
        self.assertEqual(
            provenance["recommended_enrichment_step_count"],
            len(self.skill["procedure"]) - len(observed_steps),
        )
        self.assertEqual(
            provenance["no_evidence_step_count"],
            provenance["recommended_enrichment_step_count"],
        )
        self.assertFalse(
            provenance["recommended_enrichment_counts_as_video_method_evidence"]
        )

    def test_no_evidence_fallback_is_recommended_not_observed(self) -> None:
        transcript = {
            "video_id": "neutral_v1",
            "course_id": "neutral_c1",
            "title": "Neutral material",
            "source_url": "https://example.org/neutral-v1",
            "transcript_kind": "caption_import",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "Alpha material."},
                {"start": 1.0, "end": 2.0, "text": "Beta material."},
                {"start": 2.0, "end": 3.0, "text": "Gamma material."},
            ],
        }
        skill = mine_skill(transcript)
        self.assertEqual(skill["strategies"][0]["id"], "step_by_step")
        self.assertEqual(
            skill["strategies"][0]["origin"], "recommended_enrichment"
        )
        self.assertTrue(
            all(
                step["origin"] == "recommended_enrichment"
                and step["evidence_ids"] == []
                for step in skill["procedure"]
            )
        )
        report = evaluate_skill(skill, transcript)
        provenance = report["method_provenance"]
        self.assertEqual(provenance["observed_method_step_count"], 0)
        self.assertEqual(
            provenance["recommended_enrichment_step_count"],
            len(skill["procedure"]),
        )
        self.assertEqual(
            provenance["no_evidence_step_count"], len(skill["procedure"])
        )
        self.assertFalse(
            provenance["observed_method_established_from_step_evidence"]
        )

    def test_tampered_evidence_fails_grounding_gate(self) -> None:
        skill = {**self.skill, "source": {**self.skill["source"]}}
        skill["source"]["evidence"] = [
            {"start": 0, "end": 1, "quote": "This sentence never occurred.", "supports": ["step_by_step"]}
        ]
        report = evaluate_skill(skill, self.transcript)
        self.assertFalse(report["gates"]["grounded"])
        self.assertFalse(report["passed"])

    def test_mislabeled_evidence_support_fails_grounding(self) -> None:
        skill = {**self.skill, "source": {**self.skill["source"]}}
        evidence = [dict(item) for item in self.skill["source"]["evidence"][:2]]
        for item in evidence:
            item["supports"] = ["review_and_spaced_recall"]
        skill["source"]["evidence"] = evidence
        report = evaluate_skill(skill, self.transcript)
        self.assertEqual(report["grounding_diagnostics"]["fully_matched_count"], 0)
        self.assertFalse(report["gates"]["grounded"])

    def test_multimodal_evidence_is_mined_and_verified(self) -> None:
        transcript = copy.deepcopy(self.transcript)
        transcript["multimodal"] = {
            "modalities_available": ["speech", "visual", "ocr"],
            "media": {"duration_seconds": 900.0},
            "audio": {"silences": []},
            "visual": {
                "keyframes": [
                    {
                        "timestamp": 610.0,
                        "path": "frames/scene_0001.jpg",
                        "sampling_source": "scene",
                        "ocr_text": "def solve(x): return x",
                        "ocr_language": "eng",
                    }
                ]
            },
            "events": [
                {
                    "event_id": "mme_0001",
                    "type": "code_formula_walkthrough",
                    "start": 610.0,
                    "end": 790.0,
                    "modalities": ["speech", "visual", "ocr"],
                    "evidence": {
                        "speech_quote": transcript["segments"][4]["text"],
                        "frame_path": "frames/scene_0001.jpg",
                        "ocr_text": "def solve(x): return x",
                    },
                    "confidence": 0.9,
                    "supports_strategies": ["line_by_line_explanation", "step_by_step"],
                }
            ],
        }
        skill = mine_skill(transcript)
        self.assertTrue(skill["source"]["multimodal_evidence"])
        report = evaluate_skill(skill, transcript)
        self.assertTrue(report["gates"]["multimodal_consistent"])
        self.assertEqual(report["multimodal_evaluation"]["score"], 100.0)
        self.assertEqual(report["multimodal_evaluation"]["metric_name"], "internal_evidence_consistency")
        self.assertIsNone(report["multimodal_evaluation"]["recognition_f1"])

        tampered = copy.deepcopy(skill)
        tampered["source"]["multimodal_evidence"][0]["event_id"] = "fake_event"
        bad_report = evaluate_skill(tampered, transcript)
        self.assertFalse(bad_report["gates"]["multimodal_consistent"])
        self.assertFalse(bad_report["passed"])

        tampered_evidence = copy.deepcopy(skill)
        tampered_evidence["source"]["multimodal_evidence"][0]["evidence"]["ocr_text"] = "fabricated"
        evidence_report = evaluate_skill(tampered_evidence, transcript)
        self.assertFalse(evidence_report["gates"]["multimodal_consistent"])

        inconsistent_modalities = copy.deepcopy(transcript)
        inconsistent_modalities["multimodal"]["modalities_available"] = ["speech"]
        bounded_report = evaluate_skill(skill, inconsistent_modalities)
        self.assertLessEqual(bounded_report["multimodal_evaluation"]["score"], 100.0)
        self.assertLessEqual(bounded_report["overall_score"], 100.0)
        self.assertFalse(bounded_report["passed"])

    def test_execution_substitutes_new_concept(self) -> None:
        lesson = execute_skill(self.skill, concept="牛顿法", learner_level="intermediate")
        self.assertIn("牛顿法", lesson)
        self.assertIn("动态分支", lesson)
        self.assertNotIn("{concept}", lesson)

    def test_manifest_has_required_coverage(self) -> None:
        manifest = read_json(self.root / "data/dataset_manifest.json")
        skills = []
        reports = []
        for item in manifest["videos"]:
            transcript = read_json(self.root / item["transcript_path"])
            skill = mine_skill(transcript)
            skills.append(skill)
            reports.append(evaluate_skill(skill, transcript))
        summary = evaluate_collection(skills, reports, manifest)
        self.assertEqual(summary["course_count"], 2)
        self.assertEqual(summary["video_count"], 10)
        self.assertGreaterEqual(summary["strategy_type_count"], 5)
        self.assertTrue(summary["passed"])

    def test_runtime_retries_then_advances(self) -> None:
        runtime = SkillRuntime(self.skill, concept="递归调用栈")
        retry = runtime.observe("我不知道。", "not_achieved")
        self.assertEqual(retry["event"]["transition"], "retry")
        self.assertIn("fallback_message", retry["event"])
        advance = runtime.observe("我能解释前置知识并给出例子。", "achieved")
        self.assertEqual(advance["event"]["transition"], "advance")
        self.assertEqual(runtime.procedure_index, 1)

    def test_scripted_runtime_can_complete(self) -> None:
        total = len(self.skill["procedure"]) + len(self.skill["verification"])
        result = run_scripted_session(
            self.skill,
            concept="递归调用栈",
            responses=[{"response": "有理由的回答", "signal": "achieved"}] * total,
        )
        self.assertTrue(result["completed"])
        self.assertEqual(result["completion_ratio"], 1.0)
        self.assertEqual(result["session_mode"], "scripted_state_machine_demo")
        self.assertEqual(result["response_source"], "provided_script")
        self.assertFalse(result["learning_effectiveness_established"])
        self.assertEqual(
            result["completion_semantics"], "state-machine path completion only"
        )

    def test_dataset_audit_is_honest_about_demo_data(self) -> None:
        manifest = read_json(self.root / "data/dataset_manifest.json")
        report = audit_dataset(manifest, self.root)
        self.assertTrue(report["dataset_structure_passed"])
        self.assertFalse(report["formal_empirical_ready"])
        self.assertEqual(report["research_grade_transcript_count"], 0)

    def test_short_kind_label_cannot_fake_formal_transcript_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            videos = []
            for course in ("course-a", "course-b"):
                for lesson in range(5):
                    video_id = f"{course}-{lesson}"
                    relative = f"{video_id}.json"
                    write_json(
                        root / relative,
                        {
                            "video_id": video_id,
                            "course_id": course,
                            "title": video_id,
                            "source_url": f"https://example.org/{video_id}",
                            "transcript_url": "not-a-url",
                            "transcript_kind": "human_verified_transcript",
                            "timestamps_are_approximate": False,
                            "segments": [
                                {"start": 0, "end": 1, "text": "one"},
                                {"start": 1, "end": 2, "text": "two"},
                                {"start": 2, "end": 3, "text": "three"},
                            ],
                        },
                    )
                    videos.append(
                        {
                            "video_id": video_id,
                            "course_id": course,
                            "title": video_id,
                            "source_url": f"https://example.org/{video_id}",
                            "transcript_path": relative,
                        }
                    )
            report = audit_dataset({"dataset_id": "short", "videos": videos}, root)
            self.assertTrue(report["dataset_structure_passed"])
            self.assertFalse(report["formal_empirical_ready"])
            self.assertIn(
                "nontrivial_duration",
                report["transcripts"][0]["formal_readiness_failures"],
            )

    def test_held_out_transfer_benchmark(self) -> None:
        manifest = read_json(self.root / "data/dataset_manifest.json")
        skills = [mine_skill(read_json(self.root / item["transcript_path"])) for item in manifest["videos"]]
        cases = read_json(self.root / "data/evaluation_cases.json")["cases"]
        report = benchmark_transfer(skills, cases)
        self.assertTrue(report["passed"])
        self.assertGreater(report["skill_mean"], report["static_baseline_mean"])
        self.assertTrue(report["gates"]["all_preferred_strategies_aligned"])


class HumanEvaluationTests(unittest.TestCase):
    def test_two_reviewer_summary(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            fields = [
                "skill_id", "reviewer_id", "goal_clarity_1_5", "evidence_fidelity_1_5",
                "procedure_executability_1_5", "adaptivity_1_5", "transferability_1_5",
                "harm_or_bias_flag_0_1", "comments",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for reviewer in ("r1", "r2"):
                    writer.writerow(
                        {
                            "skill_id": "s1", "reviewer_id": reviewer,
                            "goal_clarity_1_5": 4, "evidence_fidelity_1_5": 4,
                            "procedure_executability_1_5": 5, "adaptivity_1_5": 4,
                            "transferability_1_5": 4, "harm_or_bias_flag_0_1": 0,
                        }
                    )
            report = summarize_human_review(path, expected_skill_ids={"s1"})
            self.assertTrue(report["passed"])
            self.assertEqual(report["validation_status"], "complete")
            self.assertTrue(report["coverage"]["valid"])
            self.assertEqual(report["quadratic_weighted_kappa"], 1.0)

    def test_blank_review_template_is_explicitly_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            fields = [
                "skill_id", "reviewer_id", "goal_clarity_1_5", "evidence_fidelity_1_5",
                "procedure_executability_1_5", "adaptivity_1_5", "transferability_1_5",
                "harm_or_bias_flag_0_1", "comments",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerow({"skill_id": "s1"})
                writer.writerow({"skill_id": "s1"})
            report = summarize_human_review(path)
            self.assertFalse(report["passed"])
            self.assertEqual(report["validation_status"], "incomplete")
            self.assertEqual(
                report["expected_skill_id_source"], "csv_declared_skill_ids"
            )
            self.assertEqual(report["missing_skill_ids"], ["s1"])
            self.assertEqual(report["insufficient_reviewer_skill_ids"], ["s1"])
            self.assertEqual(report["completed_review_count"], 0)

    def test_human_review_reports_missing_extra_and_duplicate_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            fields = [
                "skill_id", "reviewer_id", "goal_clarity_1_5", "evidence_fidelity_1_5",
                "procedure_executability_1_5", "adaptivity_1_5", "transferability_1_5",
                "harm_or_bias_flag_0_1", "comments",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for skill_id, reviewer_id in (
                    ("s1", "r1"),
                    ("s1", "r1"),
                    ("s1", "r2"),
                    ("s3", "r1"),
                    ("s3", "r2"),
                ):
                    writer.writerow(
                        {
                            "skill_id": skill_id,
                            "reviewer_id": reviewer_id,
                            "goal_clarity_1_5": 4,
                            "evidence_fidelity_1_5": 4,
                            "procedure_executability_1_5": 4,
                            "adaptivity_1_5": 4,
                            "transferability_1_5": 4,
                            "harm_or_bias_flag_0_1": 0,
                        }
                    )
            report = summarize_human_review(
                path, expected_skill_ids={"s1", "s2"}
            )
            self.assertFalse(report["passed"])
            self.assertEqual(report["validation_status"], "invalid")
            self.assertEqual(report["missing_skill_ids"], ["s2"])
            self.assertEqual(report["extra_skill_ids"], ["s3"])
            self.assertEqual(report["insufficient_reviewer_skill_ids"], ["s2"])
            self.assertEqual(
                report["duplicate_reviewer_coverage"],
                [{"skill_id": "s1", "reviewer_id": "r1", "row_count": 2}],
            )

    def test_review_is_bound_to_exact_skill_contents(self) -> None:
        skill = {"skill_id": "s1", "version": "1.0", "procedure": ["step-a"]}
        fingerprint = skill_review_fingerprint(skill)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "reviews.csv"
            fields = [
                "skill_id",
                "skill_fingerprint",
                "reviewer_id",
                "goal_clarity_1_5",
                "evidence_fidelity_1_5",
                "procedure_executability_1_5",
                "adaptivity_1_5",
                "transferability_1_5",
                "harm_or_bias_flag_0_1",
                "comments",
            ]
            with path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for reviewer in ("r1", "r2"):
                    writer.writerow(
                        {
                            "skill_id": "s1",
                            "skill_fingerprint": fingerprint,
                            "reviewer_id": reviewer,
                            "goal_clarity_1_5": 4,
                            "evidence_fidelity_1_5": 4,
                            "procedure_executability_1_5": 4,
                            "adaptivity_1_5": 4,
                            "transferability_1_5": 4,
                            "harm_or_bias_flag_0_1": 0,
                        }
                    )
            report = summarize_human_review(
                path,
                expected_skill_ids={"s1"},
                expected_skill_fingerprints={"s1": fingerprint},
            )
            self.assertTrue(report["passed"])
            self.assertTrue(report["skill_fingerprint_binding_complete"])

            changed_fingerprint = skill_review_fingerprint(
                {**skill, "procedure": ["changed-after-review"]}
            )
            stale = summarize_human_review(
                path,
                expected_skill_ids={"s1"},
                expected_skill_fingerprints={"s1": changed_fingerprint},
            )
            self.assertFalse(stale["passed"])
            self.assertEqual(stale["validation_status"], "invalid")
            self.assertEqual(len(stale["skill_fingerprint_mismatches"]), 2)


class PipelineIntegrationTests(unittest.TestCase):
    def test_text_input_runs_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "lesson.txt"
            source.write_text(
                "First review the prerequisite. Ask why it works? Give a concrete example. "
                "Then explain the formal definition. Compare a correct case with an error. "
                "Try an exercise and check the answer. Provide feedback. Generalize to a new case. "
                "Finally summarize the idea.",
                encoding="utf-8",
            )
            output = root / "out"
            exit_code = main(
                [
                    "pipeline", str(source), "--video-id", "new_v1", "--course-id", "new_c1",
                    "--title", "A new lesson", "--source-url", "https://example.org/new-v1",
                    "--concept", "贝叶斯公式", "--output", str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            self.assertTrue((output / "skill.json").exists())
            summary = read_json(output / "pipeline_summary.json")
            self.assertTrue(summary["interactive_session_completed"])
            self.assertEqual(
                summary["interactive_session_mode"],
                "scripted_state_machine_demo",
            )
            self.assertFalse(summary["learning_effectiveness_established"])

    def test_video_with_provided_transcript_bypasses_asr_transparently(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "lesson.mp4"
            video.write_bytes(b"test-only-video-placeholder")
            transcript = project_root() / "data/transcripts/python_l03.json"
            output = root / "out"
            analysis = {
                "schema_version": "1.0",
                "media": {"duration_seconds": 10.0, "sha256": "a" * 64},
                "language": {
                    "modality": "transcript",
                    "status": "provided_transcript",
                    "transcript_kind": "curated_paraphrase_excerpt",
                    "audio_content_verified": False,
                },
                "modalities_available": ["transcript"],
                "audio": {"status": "unavailable", "silences": [], "silence_ratio": 0.0},
                "visual": {
                    "status": "unavailable",
                    "keyframes": [],
                    "events": [],
                    "ocr_enabled": False,
                },
                "events": [],
                "privacy": {
                    "identity_recognition_performed": False,
                    "classroom_observations_must_be_anonymized": True,
                },
                "provenance": {"pipeline": "test-fixture"},
            }
            with patch("teaching_skill_miner.cli.analyze_video", return_value=analysis):
                exit_code = main(
                    [
                        "pipeline",
                        str(video),
                        "--transcript",
                        str(transcript),
                        "--concept",
                        "递归调用栈",
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 0)
            summary = read_json(output / "pipeline_summary.json")
            self.assertEqual(
                summary["transcript_source_mode"], "provided_transcript_json"
            )
            self.assertTrue(summary["multimodal_analysis_performed"])
            self.assertEqual(summary["language_evidence_status"], "provided_transcript")
            self.assertFalse(summary["audio_content_verified"])


if __name__ == "__main__":
    unittest.main()
