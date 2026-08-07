from __future__ import annotations

from collections import deque
from copy import deepcopy
import json
from pathlib import Path

import jsonschema
import pytest

import teaching_skill_miner.teacher_agent_multiturn_benchmark as benchmark_module
from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.teacher_agent_live import LiveAgentOptions
from teaching_skill_miner.teacher_agent_multiturn_benchmark import (
    BENCHMARK_SCHEMA,
    CurrentLiveExecutor,
    EXECUTOR_CURRENT,
    EXECUTOR_SAFE,
    REPORT_SCHEMA,
    SafeGenerativeExecutor,
    TeacherAgentMultiturnBenchmarkError,
    build_blind_episode_payload,
    main,
    run_multiturn_benchmark,
    validate_multiturn_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


def _contains_key(value: object, target: str) -> bool:
    if isinstance(value, dict):
        return target in value or any(_contains_key(item, target) for item in value.values())
    if isinstance(value, list):
        return any(_contains_key(item, target) for item in value)
    return False


class _ScriptedExecutor:
    def __init__(self, name: str, *, good: bool) -> None:
        self.name = name
        self.good = good
        self.execution_source = "scripted_test_double_not_real_deepseek"

    def run_episode(self, dataset: dict, episode: dict) -> dict:
        del dataset
        observations: list[dict] = []
        for turn in episode["turns"]:
            if turn["operation"] != "learner_turn":
                continue
            gold = turn["gold"]
            if self.good:
                terms = [
                    group[0]
                    for group in (
                        gold["must_address_term_groups"]
                        + gold["recall_term_groups"]
                    )
                ]
                message = "；".join(dict.fromkeys(terms))
                message = (message + "。请继续说明你的判断依据。").lstrip("。")
                signal = gold["acceptable_signals"][0]
                skills = gold["allowed_primary_skill_ids"]
                primary = skills[0] if skills else None
                switched = (
                    bool(gold["expected_switch"])
                    if gold["expected_switch"] is not None
                    else False
                )
                needs_review = bool(gold["requires_visual_abstention"])
                terminal = bool(gold["should_stop"])
            else:
                message = "请继续说明你的判断依据。"
                signal = "partial"
                primary = "skill_diagnostic_questioning"
                switched = False
                needs_review = False
                terminal = False
            observations.append(
                {
                    "turn_id": turn["turn_id"],
                    "signal": signal,
                    "needs_human_review": needs_review,
                    "assessment_source": "scripted_test_double",
                    "assessment_rule_fallback": False,
                    "primary_skill_id": primary,
                    "previous_primary_skill_id": None,
                    "skill_switched": switched,
                    "teacher_message": message,
                    "terminal": terminal,
                    "terminal_status": "terminated_unable" if terminal else "active",
                    "latency_ms": 1.0,
                    "generator_used": self.name == EXECUTOR_SAFE,
                    "generator_eligible": self.name == EXECUTOR_SAFE,
                    "generator_fallback": False,
                    "model_request_outcome": "validated_model_plan",
                    "validated_model_plan_count_delta": 1,
                    "validated_plan_count_delta": 1,
                    "action_provenance": {
                        "requested_executor_mode": (
                            "safe_generative"
                            if self.name == EXECUTOR_SAFE
                            else "deterministic_legacy"
                        ),
                        "executor_origin": (
                            "deepseek_safe_generative"
                            if self.name == EXECUTOR_SAFE
                            else "deterministic_materializer"
                        ),
                        "model_teacher_action_used": self.name == EXECUTOR_SAFE,
                    },
                }
            )
        return {
            "episode_id": episode["episode_id"],
            "executor": self.name,
            "execution_source": self.execution_source,
            "observations": observations,
            "final_status": "active",
        }


class _FailingExecutor:
    name = EXECUTOR_CURRENT
    execution_source = "scripted_failure_not_real_deepseek"

    def run_episode(self, dataset: dict, episode: dict) -> dict:
        del dataset, episode
        raise RuntimeError("synthetic failure with no learner content")


class _AllFallbackRealExecutor:
    def __init__(self, name: str = EXECUTOR_CURRENT, **_kwargs: object) -> None:
        self.name = name
        self.execution_source = (
            "real_deepseek_live_agent_integrated_safe_generative"
            if name == EXECUTOR_SAFE
            else "real_deepseek_live_agent_deterministic_legacy"
        )

    def run_episode(self, dataset: dict, episode: dict) -> dict:
        del dataset
        observations = []
        for turn in episode["turns"]:
            if turn["operation"] != "learner_turn":
                continue
            generator_eligible = self.name == EXECUTOR_SAFE
            observations.append(
                {
                    "turn_id": turn["turn_id"],
                    "signal": "partial",
                    "needs_human_review": True,
                    "assessment_source": "deterministic_safety_fallback",
                    "assessment_rule_fallback": True,
                    "primary_skill_id": "skill_diagnostic_questioning",
                    "previous_primary_skill_id": None,
                    "skill_switched": False,
                    "teacher_message": "请继续说明你的判断依据。",
                    "terminal": False,
                    "terminal_status": "active",
                    "latency_ms": 1.0,
                    "model_request_outcome": "deterministic_safety_fallback",
                    "validated_model_plan_count_delta": 0,
                    "validated_plan_count_delta": 0,
                    "generator_used": False,
                    "generator_eligible": generator_eligible,
                    "generator_fallback": generator_eligible,
                    "action_provenance": {
                        "requested_executor_mode": (
                            "safe_generative"
                            if generator_eligible
                            else "deterministic_legacy"
                        ),
                        "executor_origin": "deterministic_safety_fallback",
                        "model_teacher_action_used": False,
                    },
                }
            )
        return {
            "episode_id": episode["episode_id"],
            "executor": self.name,
            "execution_source": self.execution_source,
            "observations": observations,
            "final_status": "active",
        }


def _live_plan(
    *,
    signal: str,
    skill: dict,
    message: str,
) -> dict:
    answer_alignment = {
        "not_observed": "not_applicable",
        "partial": "partially_aligned",
    }[signal]
    return {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": signal,
            "confidence": 0.0 if signal == "not_observed" else 0.8,
            "answer_alignment": answer_alignment,
            "matched_concepts": [],
            "missing_concepts": [],
            "diagnosis_reason": "当前回答提供了可定位的部分证据。",
            "evidence_excerpt": "",
            "misconception_tag": None,
            "misconception_description": "",
            "resolved_misconception_tags": [],
            "response_quality": "empty" if signal == "not_observed" else "partial",
            "engagement_level": "unknown" if signal == "not_observed" else "medium",
            "needs_human_review": False,
        },
        "decision": {
            "primary_skill_id": skill["skill_id"],
            "supporting_skill_ids": [],
            "selection_reason": "根据当前证据选择对应教学 Skill。",
            "next_focus": skill["focus_dimension"],
        },
        "teacher_action": {
            "type": skill["action_type"],
            "message": message,
            "expected_signal": "学生说明一个可核验的关键关系。",
            "question_contract": {
                "answer_type": "explanation",
                "target_concepts": ["当前关键关系"],
                "accepted_aliases": [],
                "success_criteria": ["说明一个可核验的关键关系"],
            },
        },
        "stop_recommendation": {"should_stop": False, "reason": ""},
    }


def _live_client(plans: list[dict], counter: dict[str, int]) -> DeepSeekClient:
    queue = deque(deepcopy(plans))

    def transport(
        _url: str, _headers: dict, _payload: bytes, _timeout: float
    ) -> tuple[int, bytes]:
        counter["calls"] += 1
        content = queue.popleft()
        response = {
            "id": "paired_executor_semantics_test",
            "choices": [
                {"message": {"content": json.dumps(content, ensure_ascii=False)}}
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 20},
        }
        return 200, json.dumps(response, ensure_ascii=False).encode()

    return DeepSeekClient(
        DeepSeekConfig(allow_remote_student_data=True),
        api_key="test-only-key",
        transport=transport,
    )


@pytest.fixture(scope="module")
def dataset() -> dict:
    return read_json(ROOT / "data/teacher_agent_multiturn_benchmark_v1.json")


@pytest.fixture(scope="module")
def library() -> dict:
    return read_json(ROOT / "data/teacher_agent_skill_library_v2.json")


def test_fixture_validates_against_schema_and_runtime_contract(
    dataset: dict, library: dict
) -> None:
    schema = read_json(ROOT / "schema/teacher_agent_multiturn_benchmark.schema.json")
    jsonschema.Draft202012Validator(schema).validate(dataset)
    validate_multiturn_benchmark(dataset, library)

    learner_turns = [
        turn
        for episode in dataset["episodes"]
        for turn in episode["turns"]
        if turn["operation"] == "learner_turn"
    ]
    assert dataset["schema"] == BENCHMARK_SCHEMA
    assert len(dataset["episodes"]) >= 16
    assert len(learner_turns) == 65
    assert len({episode["category"] for episode in dataset["episodes"]}) >= 6
    assert any(len(episode["turns"]) >= 6 for episode in dataset["episodes"])
    replace_profile_turns = [
        turn
        for episode in dataset["episodes"]
        for turn in episode["turns"]
        if turn["operation"] == "replace_profile"
    ]
    assert len(replace_profile_turns) == 1
    assert any(
        turn["operation"] == "replace_profile"
        for episode in dataset["episodes"]
        for turn in episode["turns"]
    )
    misconception = dataset["student_profiles"]["profile_a_formula_averse"][
        "known_misconceptions"
    ][0]
    assert set(misconception) == {"tag", "description", "confidence"}


def test_profile_replacement_episode_starts_with_the_real_live_executor(
    dataset: dict, library: dict
) -> None:
    def invalid_plan_transport(
        _url: str, _headers: dict, _payload: bytes, _timeout: float
    ) -> tuple[int, bytes]:
        return 200, b'{"choices":[]}'

    client = DeepSeekClient(
        DeepSeekConfig(allow_remote_student_data=True),
        api_key="test-only-key",
        transport=invalid_plan_transport,
    )
    executor = CurrentLiveExecutor(client, library)
    episode = next(
        item
        for item in dataset["episodes"]
        if item["episode_id"] == "profile_replacement_isolation"
    )

    result = executor.run_episode(dataset, episode)

    assert result["episode_id"] == "profile_replacement_isolation"
    assert [item["turn_id"] for item in result["observations"]] == [
        "a1",
        "b1",
        "b2",
    ]
    assert result["final_status"] == "active"


def test_paired_live_executors_use_one_integrated_plan_call_and_real_provenance(
    dataset: dict, library: dict
) -> None:
    by_id = {item["skill_id"]: item for item in library["skills"]}
    plans = [
        _live_plan(
            signal="not_observed",
            skill=by_id["skill_diagnostic_questioning"],
            message="请先说出一个必要前置概念，并说明它为什么重要？",
        ),
        _live_plan(
            signal="partial",
            skill=by_id["skill_concrete_example_bridge"],
            message="请用一个两步小例子指出当前关键关系在哪里？",
        ),
        _live_plan(
            signal="partial",
            skill=by_id["skill_concrete_example_bridge"],
            message="沿用这个两步例子，哪一项信息决定下一步？",
        ),
    ]
    episode = {
        "episode_id": "paired_executor_semantics",
        "category": "executor_semantics",
        "goal_ref": "dynamic_programming",
        "student_profile_ref": "neutral_beginner",
        "turns": [
            {
                "turn_id": "t1",
                "operation": "learner_turn",
                "learner_input": {"text": "我能说出一部分关系。", "visual_evidence_refs": []},
            },
            {
                "turn_id": "t2",
                "operation": "learner_turn",
                "learner_input": {"text": "我再补充一个小例子。", "visual_evidence_refs": []},
            },
        ],
    }
    counters = {EXECUTOR_CURRENT: {"calls": 0}, EXECUTOR_SAFE: {"calls": 0}}
    control = CurrentLiveExecutor(
        _live_client(plans, counters[EXECUTOR_CURRENT]), library
    )
    candidate = SafeGenerativeExecutor(
        _live_client(plans, counters[EXECUTOR_SAFE]),
        library,
        options=LiveAgentOptions(
            action_executor_mode="safe_generative",
            action_only_repair_enabled=True,
            state_first_route_adjudication_enabled=False,
        ),
    )

    control_result = control.run_episode(dataset, episode)
    candidate_result = candidate.run_episode(dataset, episode)

    assert control.options.action_executor_mode == "deterministic_legacy"
    assert candidate.options.action_executor_mode == "safe_generative"
    assert counters == {
        EXECUTOR_CURRENT: {"calls": 3},
        EXECUTOR_SAFE: {"calls": 3},
    }
    assert control.execution_source.endswith("deterministic_legacy")
    assert candidate.execution_source.endswith("integrated_safe_generative")
    for observation in control_result["observations"]:
        assert observation["model_request_outcome"] == "validated_model_plan"
        assert observation["validated_model_plan_count_delta"] == 1
        assert observation["validated_plan_count_delta"] == 1
        assert observation["assessment_rule_fallback"] is False
        assert observation["generator_used"] is False
        assert observation["generator_eligible"] is False
        assert observation["generator_fallback"] is False
        assert (
            observation["action_provenance"]["requested_executor_mode"]
            == "deterministic_legacy"
        )
        assert (
            observation["action_provenance"]["executor_origin"]
            == "deterministic_materializer"
        )
    for observation in candidate_result["observations"]:
        assert observation["model_request_outcome"] == "validated_model_plan"
        assert observation["validated_model_plan_count_delta"] == 1
        assert observation["validated_plan_count_delta"] == 1
        assert observation["assessment_rule_fallback"] is False
        assert observation["generator_used"] is True
        assert observation["generator_eligible"] is True
        assert observation["generator_fallback"] is False
        assert (
            observation["action_provenance"]["requested_executor_mode"]
            == "safe_generative"
        )
        assert (
            observation["action_provenance"]["executor_origin"]
            == "deepseek_safe_generative"
        )


def test_safe_executor_default_enables_state_first_route_adjudication(
    library: dict,
) -> None:
    executor = SafeGenerativeExecutor(
        _live_client([], {"calls": 0}),
        library,
    )

    assert executor.options.state_first_route_adjudication_enabled is True


def test_blind_payload_excludes_every_gold_field(
    dataset: dict,
) -> None:
    gold_keys = {
        "gold",
        "acceptable_signals",
        "allowed_primary_skill_ids",
        "must_address_term_groups",
        "recall_term_groups",
        "forbidden_output_terms",
        "direct_answer_terms",
        "profile_leakage_terms",
        "requires_visual_abstention",
        "expected_switch",
        "should_stop",
        "critical",
    }
    for episode in dataset["episodes"]:
        payload = build_blind_episode_payload(dataset, episode)
        for key in gold_keys:
            assert not _contains_key(payload, key)


def test_paired_report_exposes_semantic_gap_without_claiming_real_results(
    dataset: dict, library: dict
) -> None:
    report = run_multiturn_benchmark(
        dataset,
        library,
        executors={
            EXECUTOR_CURRENT: _ScriptedExecutor(EXECUTOR_CURRENT, good=False),
            EXECUTOR_SAFE: _ScriptedExecutor(EXECUTOR_SAFE, good=True),
        },
    )

    assert report["schema"] == REPORT_SCHEMA
    assert report["run_status"] == "completed"
    assert report["privacy"]["gold_sent_to_model"] is False
    assert report["privacy"]["teacher_messages_persisted"] is False
    assert report["claim_boundary"]["deployment_accuracy_established"] is False
    assert report["claim_boundary"]["real_learning_effect_established"] is False
    assert report["run_config"]["second_pass_action_rewrite"] is False
    assert report["run_config"]["safe_executor_prompt_version"] is None
    assert report["run_config"]["executor_semantics"] == {
        EXECUTOR_CURRENT: {
            "action_executor_mode": "deterministic_legacy",
            "request_topology": "integrated_single_plan_request",
            "action_only_repair_enabled": False,
            "state_first_route_adjudication_enabled": False,
            "fallback_to_rules": None,
        },
        EXECUTOR_SAFE: {
            "action_executor_mode": "safe_generative",
            "request_topology": "integrated_single_plan_request",
            "action_only_repair_enabled": False,
            "state_first_route_adjudication_enabled": False,
            "fallback_to_rules": None,
        },
    }
    assert (
        report["executors"][EXECUTOR_CURRENT]["action_executor_mode"]
        == "deterministic_legacy"
    )
    assert (
        report["executors"][EXECUTOR_SAFE]["action_executor_mode"]
        == "safe_generative"
    )
    assert all(
        executor["second_pass_action_rewrite"] is False
        for executor in report["executors"].values()
    )
    current = report["executors"][EXECUTOR_CURRENT]["aggregate"]
    candidate = report["executors"][EXECUTOR_SAFE]["aggregate"]
    assert current["validated_model_plan_rate"] == 1.0
    assert candidate["validated_model_plan_rate"] == 1.0
    assert current["assessment_fallback_rate"] == 0.0
    assert candidate["assessment_fallback_rate"] == 0.0
    assert current["generator_fallback_rate"] is None
    assert current["safe_executor_success_rate"] is None
    assert candidate["generator_fallback_rate"] == 0.0
    assert candidate["safe_executor_success_rate"] == 1.0
    assert candidate["action_specificity"] > current["action_specificity"]
    assert candidate["delayed_recall_accuracy"] > current["delayed_recall_accuracy"]
    assert candidate["generic_action_rate"] < current["generic_action_rate"]
    assert candidate["repetition_rate"] < current["repetition_rate"]
    assert candidate["episode_success_rate"] > current["episode_success_rate"]
    paired = report["paired_comparison"]
    assert paired["candidate_wins"] > 0
    assert paired["promotion_decision"] == "not_automatically_established"
    for executor in report["executors"].values():
        for episode in executor["episodes"]:
            for turn in episode["turn_records"]:
                assert turn["teacher_message_persisted"] is False
                assert "teacher_message" not in turn
                assert turn["model_request_outcome"] == "validated_model_plan"
                assert turn["validated_model_plan_count_delta"] == 1
                assert turn["validated_model_plan"] is True
                assert turn["assessment_rule_fallback"] is False
                assert turn["validated_plan_count_delta"] == 1
                assert isinstance(turn["action_provenance"], dict)


def test_request_accounting_uses_only_completed_committed_turns_and_matches_provenance(
    dataset: dict,
) -> None:
    episode = next(
        item
        for item in dataset["episodes"]
        if sum(turn["operation"] == "learner_turn" for turn in item["turns"]) >= 3
    )
    result = _ScriptedExecutor(EXECUTOR_SAFE, good=True).run_episode(dataset, episode)
    observations = result["observations"][:3]
    result["observations"] = observations

    observations[0].update(
        {
            "validated_plan_request_count_delta": 1,
            "action_repair_request_count_delta": 0,
            "logical_model_request_count_delta": 1,
            "action_only_repair_used": False,
        }
    )
    observations[1].update(
        {
            "validated_plan_request_count_delta": 1,
            "action_repair_request_count_delta": 1,
            "logical_model_request_count_delta": 2,
            "action_only_repair_used": True,
        }
    )
    observations[1]["action_provenance"].update(
        {
            "executor_origin": "deepseek_action_only_repair",
            "model_teacher_action_used": True,
        }
    )
    observations[2].update(
        {
            "validated_plan_request_count_delta": 1,
            "action_repair_request_count_delta": 1,
            "logical_model_request_count_delta": 2,
            "action_only_repair_used": False,
            "generator_used": False,
            "generator_fallback": True,
        }
    )
    observations[2]["action_provenance"].update(
        {
            "executor_origin": "deterministic_materializer",
            "model_teacher_action_used": False,
        }
    )

    scored = benchmark_module._score_episode(episode, result)
    aggregate = benchmark_module._aggregate_executor([scored])
    completed = [
        turn for turn in scored["turn_records"] if turn["status"] == "completed"
    ]

    assert aggregate["request_accounting_scope"] == "completed_committed_turns_only"
    assert aggregate["completed_turn_count"] == 3
    assert aggregate["validated_plan_request_total"] == 3
    assert aggregate["action_repair_request_total"] == 2
    assert aggregate["logical_model_request_total"] == 5
    assert aggregate["action_repair_request_turn_count"] == 2
    assert aggregate["action_repair_adopted_turn_count"] == 1
    assert aggregate["action_repair_adoption_rate"] == 0.5
    assert aggregate["action_only_repair_turn_count"] == 1
    assert aggregate["action_only_repair_rate"] == 0.333333
    assert aggregate["mean_validated_plan_requests_per_completed_turn"] == 1.0
    assert aggregate["mean_action_repair_requests_per_completed_turn"] == 0.666667
    assert aggregate["mean_logical_model_requests_per_completed_turn"] == 1.666667
    assert aggregate["logical_model_request_total"] == (
        aggregate["validated_plan_request_total"]
        + aggregate["action_repair_request_total"]
    )
    assert aggregate["action_repair_adopted_turn_count"] == sum(
        turn["action_provenance"]["executor_origin"]
        == "deepseek_action_only_repair"
        for turn in completed
    )
    assert all(
        turn["logical_model_request_count_delta"]
        == turn["validated_plan_request_count_delta"]
        + turn["action_repair_request_count_delta"]
        for turn in completed
    )


@pytest.mark.parametrize(
    ("repair_requests", "logical_requests", "origin", "repair_used", "match"),
    [
        (0, 1, "deepseek_action_only_repair", True, "without one repair request"),
        (1, 1, "deepseek_action_only_repair", True, "inconsistent logical"),
        (2, 3, "deepseek_action_only_repair", True, "one action-repair request"),
        (1, 2, "deepseek_safe_generative", True, "contradicts provenance"),
    ],
)
def test_request_accounting_fails_closed_on_provenance_contradictions(
    dataset: dict,
    repair_requests: int,
    logical_requests: int,
    origin: str,
    repair_used: bool,
    match: str,
) -> None:
    episode = next(
        item
        for item in dataset["episodes"]
        if any(turn["operation"] == "learner_turn" for turn in item["turns"])
    )
    result = _ScriptedExecutor(EXECUTOR_SAFE, good=True).run_episode(dataset, episode)
    result["observations"] = result["observations"][:1]
    observation = result["observations"][0]
    observation.update(
        {
            "validated_plan_request_count_delta": 1,
            "action_repair_request_count_delta": repair_requests,
            "logical_model_request_count_delta": logical_requests,
            "action_only_repair_used": repair_used,
        }
    )
    observation["action_provenance"]["executor_origin"] = origin

    with pytest.raises(TeacherAgentMultiturnBenchmarkError, match=match):
        benchmark_module._score_episode(episode, result)


def test_executor_failures_are_counted_and_cannot_look_successful(
    dataset: dict, library: dict
) -> None:
    report = run_multiturn_benchmark(
        dataset,
        library,
        executors={EXECUTOR_CURRENT: _FailingExecutor()},
    )
    current = report["executors"][EXECUTOR_CURRENT]
    assert report["run_status"] == "partial_with_failures"
    assert current["episode_failure_count"] == len(dataset["episodes"])
    assert current["aggregate"]["episode_success_rate"] == 0.0
    assert current["aggregate"]["turn_completion_rate"] == 0.0
    assert {
        episode["failure"]["error_type"]
        for episode in current["episodes"]
    } == {"RuntimeError"}
    assert "synthetic failure with no learner content" not in json.dumps(
        report, ensure_ascii=False
    )


def test_prior_110_of_110_style_real_fallback_run_cannot_report_completed(
    dataset: dict, library: dict
) -> None:
    report = run_multiturn_benchmark(
        dataset,
        library,
        executors={
            EXECUTOR_CURRENT: _AllFallbackRealExecutor(EXECUTOR_CURRENT),
            EXECUTOR_SAFE: _AllFallbackRealExecutor(EXECUTOR_SAFE),
        },
    )

    assert report["run_status"] == "insufficient_model_participation"
    for name, executor in report["executors"].items():
        aggregate = executor["aggregate"]
        assert executor["run_status"] == "insufficient_model_participation"
        assert executor["model_participation_gate"]["passed"] is False
        assert "all_assessments_used_rule_fallback" in executor[
            "model_participation_gate"
        ]["failure_reasons"]
        assert "validated_model_plan_rate_below_threshold" in executor[
            "model_participation_gate"
        ]["failure_reasons"]
        assert aggregate["validated_model_plan_rate"] == 0.0
        assert aggregate["assessment_fallback_rate"] == 1.0
        if name == EXECUTOR_CURRENT:
            assert aggregate["generator_fallback_rate"] is None
            assert aggregate["safe_executor_success_rate"] is None
        else:
            assert aggregate["generator_fallback_rate"] == 1.0
            assert aggregate["safe_executor_success_rate"] == 0.0


def test_validation_rejects_overstated_claims_and_gold_shape(
    dataset: dict, library: dict
) -> None:
    overstated = deepcopy(dataset)
    overstated["claim_boundary"]["deployment_accuracy_established"] = True
    with pytest.raises(TeacherAgentMultiturnBenchmarkError, match="claim_boundary"):
        validate_multiturn_benchmark(overstated, library)

    malformed = deepcopy(dataset)
    malformed["episodes"][0]["turns"][0]["gold"][
        "allowed_primary_skill_ids"
    ] = ["unknown_skill"]
    with pytest.raises(TeacherAgentMultiturnBenchmarkError, match="unknown allowed Skills"):
        validate_multiturn_benchmark(malformed, library)

    legacy_profile = deepcopy(dataset)
    legacy_profile["student_profiles"]["profile_a_formula_averse"][
        "known_misconceptions"
    ] = ["把递归和循环完全等同"]
    schema = read_json(ROOT / "schema/teacher_agent_multiturn_benchmark.schema.json")
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(legacy_profile)
    with pytest.raises(
        TeacherAgentMultiturnBenchmarkError,
        match="profile_a_formula_averse.*incompatible with the live runtime",
    ):
        validate_multiturn_benchmark(legacy_profile, library)


def test_validate_only_cli_never_requires_or_prints_a_key(
    capsys: pytest.CaptureFixture[str],
) -> None:
    code = main(
        [
            "--benchmark",
            str(ROOT / "data/teacher_agent_multiturn_benchmark_v1.json"),
            "--skill-library",
            str(ROOT / "data/teacher_agent_skill_library_v2.json"),
            "--validate-only",
        ]
    )
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert code == 0
    assert payload["validated"] is True
    assert payload["gold_sent_to_model"] is False
    assert "api_key" not in captured.out.casefold()


def test_validate_only_cli_honors_output_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    output = tmp_path / "validation.json"
    code = main(
        [
            "--benchmark",
            str(ROOT / "data/teacher_agent_multiturn_benchmark_v1.json"),
            "--skill-library",
            str(ROOT / "data/teacher_agent_skill_library_v2.json"),
            "--validate-only",
            "--output",
            str(output),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    payload = read_json(output)
    assert code == 0
    assert summary["output"] == str(output)
    assert summary["validated"] is True
    assert summary["gold_sent_to_model"] is False
    assert payload["schema"] == BENCHMARK_SCHEMA
    assert payload["episode_count"] == 20
    assert payload["validated"] is True
    assert payload["gold_sent_to_model"] is False


def test_online_cli_requires_explicit_remote_fixture_consent() -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "--benchmark",
                str(ROOT / "data/teacher_agent_multiturn_benchmark_v1.json"),
                "--skill-library",
                str(ROOT / "data/teacher_agent_skill_library_v2.json"),
                "--online",
            ]
        )
    assert exc.value.code == 2


def test_online_cli_returns_nonzero_for_rule_only_real_deepseek_run(
    dataset: dict,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured_options: list[object] = []

    class _ConfigStub:
        @classmethod
        def from_environment(cls, **_kwargs: object) -> object:
            return object()

    monkeypatch.setattr(benchmark_module, "DeepSeekConfig", _ConfigStub)
    monkeypatch.setattr(
        benchmark_module, "DeepSeekClient", lambda _config: object()
    )
    def current_executor_factory(**kwargs: object) -> _AllFallbackRealExecutor:
        captured_options.append(kwargs["options"])
        return _AllFallbackRealExecutor(EXECUTOR_CURRENT)

    monkeypatch.setattr(
        benchmark_module, "CurrentLiveExecutor", current_executor_factory
    )
    output = tmp_path / "rule_only_report.json"

    code = main(
        [
            "--benchmark",
            str(ROOT / "data/teacher_agent_multiturn_benchmark_v1.json"),
            "--skill-library",
            str(ROOT / "data/teacher_agent_skill_library_v2.json"),
            "--online",
            "--allow-remote-benchmark-data",
            "--no-rule-fallback",
            "--mode",
            EXECUTOR_CURRENT,
            "--output",
            str(output),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    report = read_json(output)
    assert len(dataset["episodes"]) == report["executors"][EXECUTOR_CURRENT][
        "aggregate"
    ]["episode_count"]
    assert code == 2
    assert len(captured_options) == 1
    assert getattr(captured_options[0], "fallback_to_rules") is False
    assert summary["run_status"] == "insufficient_model_participation"
    assert report["run_status"] == "insufficient_model_participation"
