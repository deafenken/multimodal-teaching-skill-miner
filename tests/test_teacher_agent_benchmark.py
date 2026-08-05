from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.cli import main as unified_main
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent_benchmark import (
    PREDICTION_SCHEMA,
    benchmark_exit_code,
    build_blind_case_payload,
    canonical_sha256,
    main,
    run_teacher_agent_benchmark,
    validate_benchmark_dataset,
)


ROOT = Path(__file__).resolve().parents[1]


def _prediction(case: dict) -> dict:
    return {
        "schema": PREDICTION_SCHEMA,
        "signal": case["gold_signal"],
        "confidence": 0.93,
        "misconception_tag": case["gold_misconception_tag"],
        "primary_skill_id": case["allowed_primary_skill_ids"][0],
        "should_terminate": case["should_terminate"],
        "needs_human_review": False,
    }


def _envelope(content: dict, *, response_id: str) -> bytes:
    return json.dumps(
        {
            "id": response_id,
            "choices": [
                {"message": {"content": json.dumps(content, ensure_ascii=False)}}
            ],
            "usage": {
                "prompt_tokens": 120,
                "completion_tokens": 24,
                "debug_echo": "provider body must not persist",
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")


class TeacherAgentFreeTextBenchmarkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.dataset = read_json(
            ROOT / "data/teacher_agent_free_text_benchmark.json"
        )
        cls.library = read_json(ROOT / "data/teacher_agent_skill_library.json")

    def _client(
        self,
        *,
        fail_indices: set[int] | None = None,
        invalid_bool_indices: set[int] | None = None,
    ) -> tuple[DeepSeekClient, list[bytes]]:
        calls: list[bytes] = []
        fail_indices = fail_indices or set()
        invalid_bool_indices = invalid_bool_indices or set()

        def transport(_url: str, headers: dict, payload: bytes, _timeout: float):
            self.assertEqual(headers["Authorization"], "Bearer fake-unit-test-key")
            index = len(calls)
            calls.append(payload)
            if index in fail_indices:
                return 503, b'{"error":"fixture failure"}'
            prediction = _prediction(self.dataset["cases"][index])
            if index in invalid_bool_indices:
                prediction["needs_human_review"] = "false"
            return 200, _envelope(prediction, response_id=f"fake_{index:03d}")

        client = DeepSeekClient(
            DeepSeekConfig(
                allow_remote_student_data=True,
                max_retries=0,
            ),
            api_key="fake-unit-test-key",
            transport=transport,
        )
        return client, calls

    def test_dataset_and_remote_payload_exclude_gold_fields(self) -> None:
        validate_benchmark_dataset(self.dataset, self.library)
        tags = sorted(
            {
                case["gold_misconception_tag"]
                for case in self.dataset["cases"]
                if case["gold_misconception_tag"]
            }
        )
        payload = build_blind_case_payload(
            self.dataset["cases"][0],
            self.library,
            misconception_tags=tags,
        )
        serialized = json.dumps(payload, ensure_ascii=False)
        for forbidden in (
            "case_id",
            "group_id",
            "gold_signal",
            "gold_misconception_tag",
            "allowed_primary_skill_ids",
            "should_switch",
            "should_terminate",
            "provenance",
        ):
            self.assertNotIn(f'"{forbidden}"', serialized)

    def test_online_run_uses_one_fake_call_per_case_and_scores_all_metrics(self) -> None:
        client, calls = self._client()
        report = run_teacher_agent_benchmark(
            self.dataset,
            self.library,
            client=client,
        )
        self.assertEqual(len(calls), 28)
        self.assertEqual(len(report["case_records"]), 28)
        self.assertEqual(report["run_status"], "completed")
        metrics = report["online_deepseek"]["metrics"]
        self.assertEqual(
            metrics["signal"]["end_to_end_all_attempts"]["accuracy"], 1.0
        )
        self.assertEqual(
            metrics["signal"]["end_to_end_all_attempts"]["macro_f1"], 1.0
        )
        self.assertEqual(
            metrics["misconception_tag"]["end_to_end_exact_match_accuracy"],
            1.0,
        )
        self.assertEqual(
            metrics["allowed_primary_skill"]["end_to_end_hit_rate"], 1.0
        )
        self.assertEqual(
            report["online_deepseek"]["operational"]["call_failure_rate"], 0.0
        )
        self.assertIsNone(metrics["needs_human_review"]["accuracy"])
        self.assertFalse(
            metrics["needs_human_review"]["gold_review_label_available"]
        )

        serialized_report = json.dumps(report, ensure_ascii=False)
        private_input = self.dataset["cases"][0]["learner_response"]
        private_prompt = self.dataset["cases"][0]["current_action"]["prompt"]
        self.assertNotIn(private_input, serialized_report)
        self.assertNotIn(private_prompt, serialized_report)
        self.assertNotIn("fake-unit-test-key", serialized_report)
        self.assertNotIn("provider body must not persist", serialized_report)
        material = dict(report)
        content_sha = material.pop("content_sha256")
        self.assertEqual(content_sha, canonical_sha256(material))

        first_request = json.loads(calls[0])
        user_message = first_request["messages"][1]["content"]
        self.assertIn(private_input, user_message)
        self.assertNotIn('"gold_signal"', user_message)
        self.assertNotIn('"allowed_primary_skill_ids"', user_message)

    def test_api_and_strict_validation_failures_are_counted_and_force_review(self) -> None:
        client, _calls = self._client(
            fail_indices={0},
            invalid_bool_indices={1},
        )
        report = run_teacher_agent_benchmark(
            self.dataset,
            self.library,
            client=client,
        )
        operational = report["online_deepseek"]["operational"]
        self.assertEqual(operational["api_call_failure_count"], 1)
        self.assertEqual(operational["prediction_validation_failure_count"], 1)
        self.assertAlmostEqual(
            operational["end_to_end_failure_rate"], 2 / 28, places=6
        )
        records = report["case_records"]
        self.assertEqual(records[0]["failure"]["stage"], "api_call")
        self.assertEqual(
            records[1]["failure"]["stage"], "prediction_validation"
        )
        self.assertTrue(records[0]["needs_human_review"])
        self.assertTrue(records[1]["needs_human_review"])
        self.assertNotIn("fixture failure", json.dumps(report))
        self.assertNotIn("false must be", json.dumps(report))
        self.assertEqual(benchmark_exit_code(report), 2)

    def test_baseline_only_cli_never_requires_a_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            exit_code = main(
                [
                    "--benchmark",
                    str(ROOT / "data/teacher_agent_free_text_benchmark.json"),
                    "--skill-library",
                    str(ROOT / "data/teacher_agent_skill_library.json"),
                    "--output",
                    str(output),
                ]
            )
            self.assertEqual(exit_code, 0)
            report = read_json(output)
            self.assertEqual(report["run_status"], "baseline_only")
            self.assertEqual(
                report["online_deepseek"]["operational"]["attempted_call_count"],
                0,
            )
            self.assertTrue(
                report["baselines"]["deterministic_structured_signal_router"][
                    "uses_gold_structured_signal"
                ]
            )
            self.assertIsNone(
                report["baselines"]["fixed_single_skill"]["signal_accuracy"]
            )
            self.assertFalse(report["claim_boundary"]["expert_validated"])
            self.assertFalse(
                report["claim_boundary"]["deployment_accuracy_established"]
            )

    def test_unified_cli_exposes_safe_baseline_mode_and_hash_bound_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "report.json"
            with redirect_stdout(io.StringIO()) as stdout:
                exit_code = unified_main(
                    [
                        "teacher-agent-benchmark",
                        "--benchmark",
                        str(ROOT / "data/teacher_agent_free_text_benchmark.json"),
                        "--skill-library",
                        str(ROOT / "data/teacher_agent_skill_library_v2.json"),
                        "--output",
                        str(output),
                    ]
                )
            self.assertEqual(exit_code, 0)
            summary = json.loads(stdout.getvalue())
            report = read_json(output)
            self.assertEqual(summary["run_status"], "baseline_only")
            self.assertFalse(summary["input_content_printed"])
            self.assertFalse(summary["api_key_printed"])
            material = dict(report)
            content_sha = material.pop("content_sha256")
            self.assertEqual(content_sha, canonical_sha256(material))

    def test_unified_online_mode_requires_explicit_remote_data_consent(self) -> None:
        with redirect_stderr(io.StringIO()) as stderr:
            exit_code = unified_main(["teacher-agent-benchmark", "--online"])
        self.assertEqual(exit_code, 1)
        self.assertIn("--allow-remote-benchmark-data", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
