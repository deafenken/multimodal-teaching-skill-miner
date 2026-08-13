from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest

import jsonschema

from teaching_skill_miner.cli import main
from teaching_skill_miner.io_utils import project_root, read_json
from teaching_skill_miner.teacher_agent import (
    TeacherAgentError,
    advance_teacher_agent_session,
    evaluate_teacher_agent,
    session_turn_summary,
    start_teacher_agent_session,
    validate_session,
    validate_skill_library,
)


class TeacherAgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = project_root()
        cls.library = read_json(root / "data/teacher_agent_skill_library.json")
        cls.demo = read_json(root / "data/teacher_agent_demo_input.json")
        cls.cases = read_json(root / "data/teacher_agent_evaluation_cases.json")

    def _start(self, profile: dict | None = None) -> dict:
        return start_teacher_agent_session(
            self.demo["goal"],
            profile or self.demo["student_profile"],
            self.library,
        )

    def test_public_skill_library_is_valid_and_source_bounded(self) -> None:
        validate_skill_library(self.library)
        self.assertEqual(len(self.library["skills"]), 10)
        primary = [item for item in self.library["skills"] if item["role"] != "support"]
        self.assertEqual(len(primary), 8)
        for skill in self.library["skills"]:
            with self.subTest(skill_id=skill["skill_id"]):
                self.assertEqual(
                    skill["source"]["origin"],
                    "operational_wrapper_from_general_skill_v0",
                )
                self.assertEqual(
                    skill["source"]["general_skill_id"],
                    "evidence_grounded_adaptive_concept_teaching_v0",
                )

    def test_neural_v1_runtime_library_and_manifest_are_schema_valid_and_bounded(
        self,
    ) -> None:
        root = project_root()
        library = read_json(root / "data/teacher_agent_skill_library_v2.json")
        manifest = read_json(root / "data/neural_v1_runtime_manifest.json")
        validate_skill_library(library)
        jsonschema.Draft202012Validator(
            read_json(root / "schema/teacher_agent_skill_library_v2.schema.json")
        ).validate(library)
        jsonschema.Draft202012Validator(
            read_json(root / "schema/neural_v1_runtime_manifest.schema.json")
        ).validate(manifest)
        self.assertEqual(library["derivation"]["general_skill_id"], manifest["skill_id"])
        self.assertEqual(len(manifest["canonical_phases"]), 9)
        self.assertFalse(manifest["materialization_gate"]["passed"])
        self.assertFalse(
            library["claim_boundary"]["neural_v1_materialization_gate_passed"]
        )

    def test_free_text_benchmark_is_schema_valid_and_author_constructed(self) -> None:
        root = project_root()
        benchmark = read_json(root / "data/teacher_agent_free_text_benchmark.json")
        jsonschema.Draft202012Validator(
            read_json(root / "schema/teacher_agent_free_text_benchmark.schema.json")
        ).validate(benchmark)
        self.assertGreaterEqual(len(benchmark["cases"]), 24)
        self.assertFalse(benchmark["claim_boundary"]["expert_validated"])
        self.assertFalse(benchmark["claim_boundary"]["real_students_involved"])

    def test_public_fixtures_and_generated_artifacts_match_json_schemas(self) -> None:
        root = project_root()
        jsonschema.Draft202012Validator(
            read_json(root / "schema/teacher_agent_skill_library.schema.json")
        ).validate(self.library)
        jsonschema.Draft202012Validator(
            read_json(root / "schema/teacher_agent_evaluation_cases.schema.json")
        ).validate(self.cases)
        session = self._start()
        jsonschema.Draft202012Validator(
            read_json(root / "schema/teacher_agent_session.schema.json")
        ).validate(session)
        report = evaluate_teacher_agent(self.library, self.cases)
        jsonschema.Draft202012Validator(
            read_json(root / "schema/teacher_agent_evaluation_report.schema.json")
        ).validate(report)

    def test_start_emits_only_one_action_and_explicit_state(self) -> None:
        session = self._start()
        validate_session(session)
        self.assertEqual(session["round"], 0)
        self.assertEqual(session["history"], [])
        self.assertEqual(
            session["current_action"]["primary_skill"]["skill_id"],
            "skill_diagnostic_questioning",
        )
        self.assertTrue(
            session["current_action"]["teacher_action"][
                "wait_for_student_before_next_action"
            ]
        )
        state = session["student_state"]
        self.assertEqual(
            set(state["knowledge_mastery"]),
            {"prerequisite", "conceptual", "procedural", "transfer"},
        )
        self.assertIn("misconceptions", state)
        self.assertIn("understanding_signal", state)
        self.assertIn("next_focus", state)
        self.assertTrue(session["current_action"]["selection_reason"])

    def test_teacher_knowledge_spec_is_normalized_and_bound_to_components(self) -> None:
        goal = deepcopy(self.demo["goal"])
        goal["knowledge_components"] = ["状态定义", "状态转移"]
        goal["knowledge_spec"] = {
            "canonical_claims": [
                {
                    "claim_id": "claim_transition",
                    "statement": "状态转移说明当前状态怎样依赖已解决子问题。",
                    "knowledge_components": ["状态转移"],
                    "source_ids": ["teacher_note"],
                }
            ],
            "rubric_criteria": [
                {
                    "criterion_id": "criterion_dependency",
                    "description": "指出依赖的先前状态及组合方式",
                    "knowledge_component": "状态转移",
                    "acceptable_evidence": ["明确写出依赖关系"],
                }
            ],
            "accepted_alternatives": [
                {
                    "alternative_id": "alternative_recursive",
                    "description": "用记忆化递归表达同一依赖",
                    "equivalent_claim_ids": ["claim_transition"],
                }
            ],
            "reference_steps": [
                {
                    "step_id": "step_define",
                    "description": "先定义状态",
                    "knowledge_components": ["状态定义"],
                },
                {
                    "step_id": "step_transition",
                    "description": "再写转移",
                    "knowledge_components": ["状态转移"],
                    "depends_on": ["step_define"],
                },
            ],
            "misconception_catalog": [
                {
                    "tag": "only_previous",
                    "description": "误以为所有状态只依赖紧邻前一项",
                    "contradicts_claim_ids": ["claim_transition"],
                }
            ],
            "sources": [
                {
                    "source_id": "teacher_note",
                    "title": "教师课程说明",
                    "citation": "本次课程讲义第 2 节",
                }
            ],
        }
        session = start_teacher_agent_session(
            goal, self.demo["student_profile"], self.library
        )
        spec = session["goal"]["knowledge_spec"]
        self.assertEqual(spec["status"], "teacher_provided")
        self.assertTrue(spec["claim_boundary"]["authoritative_for_runtime_grading"])
        self.assertEqual(spec["authority"]["status"], "teacher_asserted")
        self.assertTrue(spec["authority"]["authoritative_for_runtime_grading"])
        self.assertFalse(spec["claim_boundary"]["independently_verified_by_system"])
        self.assertEqual(
            spec["rubric_criteria"][0]["knowledge_component"], "状态转移"
        )

    def test_absent_knowledge_spec_does_not_promote_model_memory_to_answer_key(self) -> None:
        goal = deepcopy(self.demo["goal"])
        goal.pop("knowledge_spec", None)
        session = start_teacher_agent_session(
            goal,
            self.demo["student_profile"],
            self.library,
        )
        spec = session["goal"]["knowledge_spec"]
        self.assertEqual(spec["status"], "not_provided")
        self.assertEqual(spec["authority"]["status"], "not_provided")
        self.assertFalse(spec["claim_boundary"]["authoritative_for_runtime_grading"])
        self.assertFalse(
            spec["claim_boundary"]["model_memory_is_authoritative_when_absent"]
        )

    def test_knowledge_spec_rejects_unknown_component_and_dangling_claim(self) -> None:
        for knowledge_spec in (
            {
                "canonical_claims": [
                    {
                        "statement": "错误组件引用",
                        "knowledge_components": ["不存在的组件"],
                    }
                ]
            },
            {
                "accepted_alternatives": [
                    {
                        "description": "悬空等价路径",
                        "equivalent_claim_ids": ["missing_claim"],
                    }
                ]
            },
        ):
            goal = deepcopy(self.demo["goal"])
            goal["knowledge_components"] = ["状态转移"]
            goal["knowledge_spec"] = knowledge_spec
            with self.subTest(knowledge_spec=knowledge_spec):
                with self.assertRaises(TeacherAgentError):
                    start_teacher_agent_session(
                        goal, self.demo["student_profile"], self.library
                    )

    def test_each_action_tags_only_the_active_goal_knowledge_component(self) -> None:
        goal = deepcopy(self.demo["goal"])
        goal["knowledge_components"] = [
            "递归",
            "状态定义",
            "状态转移",
            "迁移判定",
        ]
        goal.pop("knowledge_spec", None)
        session = start_teacher_agent_session(
            goal,
            self.demo["student_profile"],
            self.library,
        )

        self.assertEqual(
            session["current_action"]["knowledge_components"],
            ["递归"],
        )
        self.assertEqual(
            session["student_state"]["next_focus"]["knowledge_components"],
            ["递归"],
        )
        self.assertNotEqual(
            session["current_action"]["knowledge_components"],
            session["goal"]["knowledge_components"],
        )

    def test_misconception_triggers_correction_and_can_be_resolved(self) -> None:
        session = self._start()
        session = advance_teacher_agent_session(
            session,
            learner_response="状态只看上一步。",
            signal="misconception",
            misconception_tag="missing_transition",
        )
        self.assertEqual(
            session["current_action"]["primary_skill"]["skill_id"],
            "skill_misconception_contrast",
        )
        self.assertEqual(session["student_state"]["misconceptions"][0]["status"], "active")
        session = advance_teacher_agent_session(
            session,
            learner_response="还需要考虑从前两步转移。",
            signal="correct",
        )
        self.assertEqual(
            session["student_state"]["misconceptions"][0]["status"],
            "resolved",
        )

    def test_named_resolution_parameter_cannot_bypass_core_correction_contract(self) -> None:
        session = self._start()
        session = advance_teacher_agent_session(
            session,
            learner_response="状态只看上一步。",
            signal="misconception",
            misconception_tag="missing_transition",
        )
        before = deepcopy(session)

        with self.assertRaisesRegex(
            TeacherAgentError, "correct, targeted correction turn"
        ):
            advance_teacher_agent_session(
                session,
                learner_response="我还不确定。",
                signal="partial",
                resolve_all_on_correction=False,
                resolved_misconception_tags=["missing_transition"],
            )

        self.assertEqual(session, before)

    def test_three_no_progress_signals_stop_and_escalate(self) -> None:
        session = self._start()
        for signal in ("confused", "no_response", "confused"):
            session = advance_teacher_agent_session(
                session,
                learner_response="不知道",
                signal=signal,
            )
        self.assertEqual(session["status"], "terminated_unable")
        self.assertEqual(session["round"], 3)
        self.assertEqual(session["current_action"]["type"], "terminate_unable")
        with self.assertRaisesRegex(TeacherAgentError, "terminal"):
            advance_teacher_agent_session(
                session,
                learner_response="继续",
                signal="correct",
            )

    def test_demo_reaches_success_with_multiple_skill_switches(self) -> None:
        session = self._start()
        for item in self.demo["demo_feedback_sequence"]:
            if session["status"] != "active":
                break
            session = advance_teacher_agent_session(
                session,
                learner_response=item["response"],
                signal=item["signal"],
                misconception_tag=item.get("misconception_tag"),
            )
        self.assertEqual(session["status"], "succeeded")
        self.assertEqual(session["round"], 9)
        self.assertGreaterEqual(session["control"]["skill_switch_count"], 5)
        self.assertTrue(
            all(
                session["student_state"]["knowledge_mastery"][dimension]
                >= session["goal"]["success_thresholds"][dimension]
                for dimension in session["goal"]["success_thresholds"]
            )
        )
        self.assertFalse(
            session["claim_boundary"]["real_learning_effectiveness_established"]
        )

    def test_fixed_baseline_does_not_switch(self) -> None:
        session = start_teacher_agent_session(
            self.demo["goal"],
            self.demo["student_profile"],
            self.library,
            policy="fixed_single_skill_baseline",
            fixed_skill_id="skill_stepwise_scaffolding",
        )
        for signal in ("partial", "correct", "misconception", "correct"):
            self.assertEqual(
                session["current_action"]["primary_skill"]["skill_id"],
                "skill_stepwise_scaffolding",
            )
            session = advance_teacher_agent_session(
                session,
                learner_response="fixture response",
                signal=signal,
                misconception_tag="fixture_error" if signal == "misconception" else None,
            )
        self.assertEqual(session["control"]["skill_switch_count"], 0)

    def test_integrity_rejects_tampered_session(self) -> None:
        session = self._start()
        tampered = deepcopy(session)
        tampered["student_state"]["knowledge_mastery"]["conceptual"] = 1.0
        with self.assertRaisesRegex(TeacherAgentError, "integrity"):
            validate_session(tampered)

    def test_evaluation_is_reproducible_and_beats_fixed_simulation(self) -> None:
        first = evaluate_teacher_agent(self.library, self.cases)
        second = evaluate_teacher_agent(self.library, self.cases)
        self.assertEqual(first, second)
        self.assertTrue(first["passed"], first)
        self.assertEqual(first["aggregate"]["student_state_judgement_rate"], 1.0)
        self.assertEqual(first["aggregate"]["teaching_decision_match_rate"], 1.0)
        self.assertEqual(first["aggregate"]["case_count"], 4)
        self.assertEqual(first["aggregate"]["expected_success_case_count"], 3)
        self.assertEqual(first["aggregate"]["expected_unable_case_count"], 1)
        self.assertEqual(first["aggregate"]["terminal_decision_match_rate"], 1.0)
        failure = next(
            row
            for row in first["cases"]
            if row["case_id"] == "persistent_no_progress_escalation"
        )
        self.assertEqual(failure["adaptive_agent"]["status"], "terminated_unable")
        self.assertEqual(
            failure["adaptive_agent"]["termination_reason"],
            "three consecutive rounds without observable progress",
        )
        self.assertGreater(first["aggregate"]["simulated_mean_gain_delta"], 0)
        self.assertFalse(
            first["claim_boundary"]["real_learner_effectiveness_established"]
        )

    def test_summary_is_ui_safe_and_cli_round_trip_persists_session(self) -> None:
        summary = session_turn_summary(self._start())
        self.assertEqual(summary["rounds_completed"], 0)
        self.assertNotIn("skill_library", summary)
        with tempfile.TemporaryDirectory() as directory:
            session_path = Path(directory) / "session.json"
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "teacher-agent-start",
                        "--input",
                        str(project_root() / "data/teacher_agent_demo_input.json"),
                        "--library",
                        str(project_root() / "data/teacher_agent_skill_library.json"),
                        "--session",
                        str(session_path),
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(session_path.is_file())
            self.assertEqual(session_path.stat().st_mode & 0o777, 0o600)
            with redirect_stdout(io.StringIO()):
                code = main(
                    [
                        "teacher-agent-step",
                        "--session",
                        str(session_path),
                        "--response",
                        "我能说明递归拆分。",
                        "--signal",
                        "partial",
                    ]
                )
            self.assertEqual(code, 0)
            validate_session(json.loads(session_path.read_text(encoding="utf-8")))


if __name__ == "__main__":
    unittest.main()
