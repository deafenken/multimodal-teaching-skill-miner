"""Regression checks for public documentation that mirrors generated evidence."""

from __future__ import annotations

import json
import unittest

from teaching_skill_miner.evaluator import DIMENSION_WEIGHTS
from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.teacher_agent import (
    ADAPTIVE_OBSERVATION_LIMIT,
    ADAPTIVE_OBSERVATION_STATUS,
)
from teaching_skill_miner.teacher_agent_harness import _tool_specs
from teaching_skill_miner.teacher_agent_live import LIVE_PROMPT_VERSION


class DocumentationFreshnessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = project_root()

    def _read(self, relative: str) -> str:
        return (self.root / relative).read_text(encoding="utf-8")

    def test_ablation_scores_match_current_public_receipt(self) -> None:
        receipt = json.loads(
            self._read("artifacts/public/multimodal_ablation_receipt.json")
        )
        metrics = receipt["aggregate_internal_metrics"]
        ordered_arms = (
            "transcript_only",
            "transcript_audio",
            "transcript_visual",
            "full",
        )
        score_sequence = "/".join(
            f"{metrics[arm]['mean_internal_overall_score']:.2f}"
            for arm in ordered_arms
        )
        for relative in (
            "artifacts/public/README.md",
            "docs/project_status.md",
            "docs/requirements_traceability.md",
        ):
            text = self._read(relative)
            self.assertIn(score_sequence, text, relative)
            self.assertNotIn("100.00/100.00/99.92/99.94", text, relative)

        table_labels = {
            "transcript_only": "transcript-only",
            "transcript_audio": "transcript + audio",
            "transcript_visual": "transcript + visual/OCR",
            "full": "full",
        }
        for relative in ("README.md", "docs/multimodal_design.md"):
            text = self._read(relative)
            for arm in ordered_arms:
                expected = (
                    f"| {table_labels[arm]} | "
                    f"{metrics[arm]['mean_internal_overall_score']:.2f} |"
                )
                self.assertIn(expected, text, f"{relative}: {arm}")

    def test_evaluation_dimension_summaries_match_current_weights(self) -> None:
        expected = {
            "structural_completeness": ("结构完整性", 0.12),
            "evidence_grounding": ("证据落地性", 0.18),
            "executability": ("可执行性", 0.18),
            "method_fidelity": ("方法忠实度", 0.22),
            "pedagogical_quality": ("教学质量", 0.12),
            "generalizability": ("可迁移性", 0.09),
            "traceability": ("可追溯性", 0.09),
        }
        self.assertEqual(DIMENSION_WEIGHTS, {key: value[1] for key, value in expected.items()})

        readme = self._read("README.md")
        overview = readme.split("### 1. 视频采集与预处理", 1)[0]
        self.assertIn("方法忠实度", overview)
        for key, (label, weight) in expected.items():
            row = f"| {label} | {weight:.0%} |"
            self.assertIn(row, readme, key)

        design = self._read("docs/multimodal_design.md")
        self.assertIn("七维内部量表", design)
        self.assertIn("`method_fidelity`", design)
        public_readme = self._read("artifacts/public/README.md")
        self.assertIn("seven-dimension internal rubric", public_readme)
        self.assertIn("`method_fidelity`", public_readme)

    def test_validation_report_matches_current_teachobs_receipt(self) -> None:
        receipt = json.loads(
            self._read(
                "artifacts/public/teachobs_multimodal_benchmark_receipt.json"
            )
        )
        report = self._read("docs/validation_report.md")
        for arm in receipt["aggregate_arms"].values():
            for metric in (
                "micro_f1",
                "macro_f1",
                "hamming_accuracy",
                "visual_macro_f1",
            ):
                self.assertIn(f"{arm['metrics'][metric]:.6f}", report)

        comparison = receipt["paired_cluster_bootstrap"]["comparisons"][
            "full_minus_transcript_only"
        ]
        for metric in ("macro_f1", "visual_macro_f1"):
            result = comparison[metric]
            self.assertIn(f"{result['point_delta']:+.6f}", report)
            for bound in result["percentile_95_ci"]:
                self.assertIn(f"{bound:+.6f}", report)

        for claim, value in receipt["claim_boundaries"].items():
            if claim.endswith("_established"):
                self.assertFalse(value, claim)
                self.assertIn(f"`{claim}=false`", report)

    def test_task_two_docs_match_adaptive_profile_candidate_contract(self) -> None:
        schema = json.loads(
            self._read("schema/teacher_agent_live_session.schema.json")
        )
        observation_definition = schema["$defs"]["adaptiveObservations"]
        summary_definition = schema["$defs"]["adaptiveSummary"]
        self.assertEqual(
            observation_definition["maxItems"], ADAPTIVE_OBSERVATION_LIMIT
        )
        self.assertEqual(
            observation_definition["items"]["properties"]["status"]["const"],
            ADAPTIVE_OBSERVATION_STATUS,
        )
        self.assertFalse(
            summary_definition["properties"][
                "teacher_provided_fields_overwritten"
            ]["const"]
        )

        readme = self._read("README.md")
        task_two_readme = readme.split(
            "### 题目二：实时自适应教学 Agent", 1
        )[1].split("### 本机双成果真实演示", 1)[0]
        documents = {
            "README.md task two": task_two_readme,
            "docs/teacher_agent_task2.md": self._read(
                "docs/teacher_agent_task2.md"
            ),
            "docs/teacher_agent_acceptance_matrix.md": self._read(
                "docs/teacher_agent_acceptance_matrix.md"
            ),
            "docs/teacher_agent_defense_guide.md": self._read(
                "docs/teacher_agent_defense_guide.md"
            ),
        }
        for relative, text in documents.items():
            with self.subTest(relative=relative):
                self.assertIn(ADAPTIVE_OBSERVATION_STATUS, text)
                self.assertIn(str(ADAPTIVE_OBSERVATION_LIMIT), text)
                self.assertIn("fallback", text)
                self.assertIn("跨 session", text)
                self.assertTrue(
                    any(
                        marker in text
                        for marker in ("不会覆盖", "不会被覆盖", "不覆盖")
                    ),
                    relative,
                )

        acceptance = documents["docs/teacher_agent_acceptance_matrix.md"]
        self.assertIn("`teacher_provided_fields_overwritten=false`", acceptance)
        self.assertIn("fallback 后候选数不增加", acceptance)

    def test_task_two_docs_match_current_resume_visual_and_multiturn_boundaries(
        self,
    ) -> None:
        readme = self._read("README.md")
        task_two_readme = readme.split(
            "### 题目二：实时自适应教学 Agent", 1
        )[1].split("### 本机双成果真实演示", 1)[0]
        task_two = self._read("docs/teacher_agent_task2.md")
        acceptance = self._read("docs/teacher_agent_acceptance_matrix.md")
        defense = self._read("docs/teacher_agent_defense_guide.md")

        for label, text in {
            "README task two": task_two_readme,
            "task two design": task_two,
            "acceptance matrix": acceptance,
            "defense guide": defense,
        }.items():
            with self.subTest(document=label):
                self.assertIn("20", text)
                self.assertIn("65", text)
                self.assertIn("1 次画像替换操作", text)
                self.assertIn("未经专家复核", text)
                self.assertIn("formula_accuracy_established", text)
                self.assertIn("--session-store", text)

        for text in (task_two_readme, task_two, acceptance):
            self.assertIn("formula_transcription_established", text)

        for text in (task_two_readme, task_two, defense):
            self.assertIn("15ea598c6e7e0914a7ae8c881ac05dacea2f7902", text)

        for text in (task_two_readme, task_two, acceptance):
            self.assertIn("runtime_policy_contract", text)
            self.assertIn("turn_started", text)
            self.assertIn("turn_committed", text)
            self.assertIn("turn_aborted", text)
            self.assertIn("deepseek_safe_generative", text)
            self.assertIn("deterministic_materializer", text)

        self.assertIn("10 个相关回合", task_two_readme)
        self.assertIn("默认最多 10 个相关回合", acceptance)
        self.assertNotIn("公式样或低于共享数值阈值的 OCR 固定要求", task_two)

    def test_task_two_docs_match_v18_routing_continuity_accounting_and_stop_edges(
        self,
    ) -> None:
        readme = self._read("README.md")
        task_two = self._read("docs/teacher_agent_task2.md")
        acceptance = self._read("docs/teacher_agent_acceptance_matrix.md")
        defense = self._read("docs/teacher_agent_defense_guide.md")
        project_status = self._read("docs/project_status.md")
        changelog = self._read("CHANGELOG.md")
        documents = {
            "README.md": readme,
            "docs/teacher_agent_task2.md": task_two,
            "docs/teacher_agent_acceptance_matrix.md": acceptance,
            "docs/teacher_agent_defense_guide.md": defense,
            "docs/project_status.md": project_status,
            "CHANGELOG.md": changelog,
        }
        self.assertEqual(
            LIVE_PROMPT_VERSION,
            "teaching_agent_assess_route_act_v18_direct_teaching_cache_stable_prefix",
        )
        for relative, text in documents.items():
            with self.subTest(document=relative):
                self.assertIn("V18", text)
                self.assertNotIn("V12", text)
                self.assertNotIn("v12", text)
                self.assertIn("state-first", text)
                self.assertIn("continuity_recall", text)
                self.assertIn("action_only_repair", text)
                self.assertIn("completed_committed_turns_only", text)
                self.assertIn("cancel_turn", text)
                self.assertIn("transport_cancellation_supported", text)
        for text in (readme, task_two, acceptance, defense):
            self.assertIn("确认不等于答案正确", text)
            self.assertIn("DeepSeek-Reasonix", text)
            self.assertIn("prompt_cache_hit_tokens", text)
            self.assertIn("prompt_cache_miss_tokens", text)
        self.assertNotIn("teacher-led orientation", changelog)
        self.assertIn(LIVE_PROMPT_VERSION, acceptance)

    def test_modern_console_harness_project_and_recovery_docs_match_current_tree(
        self,
    ) -> None:
        package = json.loads(self._read("apps/console/package.json"))
        self.assertEqual(package["dependencies"]["next"], "15.5.23")
        self.assertNotIn("monaco-editor", package["dependencies"])
        self.assertNotIn("@xterm/xterm", package["dependencies"])

        console_documents = {
            relative: self._read(relative)
            for relative in (
                "README.md",
                "CHANGELOG.md",
                "apps/console/README.md",
                "docs/project_status.md",
                "docs/requirements_traceability.md",
                "docs/teacher_agent_acceptance_matrix.md",
                "docs/teacher_agent_task2.md",
            )
        }
        for relative, text in console_documents.items():
            with self.subTest(document=relative):
                self.assertIn("15.5.23", text)
                self.assertIn("Monaco/xterm", text)
                self.assertTrue(
                    "移除" in text or "removed" in text,
                    relative,
                )

        expected_tools = {
            "inspect_student_state",
            "inspect_recent_history",
            "search_skills",
            "select_skills",
            "set_next_focus",
            "evaluate_termination",
            "retrieve_resources",
        }
        self.assertEqual({spec.name for spec in _tool_specs()}, expected_tools)
        for relative in (
            "README.md",
            "docs/agent_harness_architecture.md",
            "docs/requirements_traceability.md",
            "docs/teacher_agent_acceptance_matrix.md",
            "docs/teacher_agent_task2.md",
        ):
            text = self._read(relative)
            with self.subTest(tool_document=relative):
                for tool_name in expected_tools:
                    self.assertIn(tool_name, text)
                self.assertIn("retrieve_resources", text)

        recovery_documents = {
            relative: self._read(relative)
            for relative in (
                "docs/agent_harness_architecture.md",
                "docs/project_status.md",
                "docs/requirements_traceability.md",
                "docs/teacher_agent_acceptance_matrix.md",
                "docs/teacher_agent_task2.md",
            )
        }
        for relative, text in recovery_documents.items():
            with self.subTest(recovery_document=relative):
                self.assertTrue(
                    "pre-provider" in text or "start_pre_provider_only" in text,
                    relative,
                )
                self.assertIn("handoff", text)
                self.assertIn("domain commit", text)
                self.assertIn("Chat 非终态", text)
                self.assertIn("不重放", text)

        for marker in (
            "学习项目",
            "400 条",
            "scoring gold",
            "Web Search 默认关闭",
            "consent",
        ):
            self.assertIn(marker, self._read("docs/teacher_agent_task2.md"))

        for relative in (
            "README.md",
            "docs/project_status.md",
            "docs/requirements_traceability.md",
            "docs/teacher_agent_acceptance_matrix.md",
            "docs/teacher_agent_task2.md",
        ):
            text = self._read(relative)
            with self.subTest(acceptance_freshness_document=relative):
                self.assertIn("artifacts/release_acceptance_1.2.0.json", text)
                self.assertIn("verification scope", text)
                self.assertIn("build_release_acceptance.sh", text)
                self.assertNotIn("1,119", text)
                self.assertNotIn("1,474", text)


if __name__ == "__main__":
    unittest.main()
