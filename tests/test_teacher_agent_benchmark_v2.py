from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from teaching_skill_miner import teacher_agent_benchmark_v2 as benchmark_v2
from teaching_skill_miner.teacher_agent import start_teacher_agent_session
from teaching_skill_miner.teacher_agent_benchmark_v2 import (
    BENCHMARK_VERSION,
    PREDICTIONS_SCHEMA,
    TeachingAgentBenchmarkV2Error,
    _terminal_guard_lifecycle_receipt,
    _fingerprint,
    score_benchmark_v2,
    predictions_from_live_cases,
    validate_benchmark_gold,
    validate_benchmark_inputs,
    validate_predictions,
)
from teaching_skill_miner.teacher_agent_loop import public_agent_loop_trace
from teaching_skill_miner.teacher_agent_orchestration import (
    build_turn_lifecycle_receipt,
)


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str) -> dict:
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


@pytest.fixture
def artifacts() -> tuple[dict, dict, dict]:
    inputs = _load("data/teacher_agent_benchmark_v2_development.json")
    gold = _load("data/teacher_agent_benchmark_v2_development_gold.json")
    library = _load("data/teacher_agent_skill_library_v2.json")
    return inputs, gold, library


def _perfect_predictions(inputs: dict, gold: dict) -> dict:
    gold_by_id = {row["case_id"]: row for row in gold["cases"]}
    cases = []
    for case_index, case in enumerate(inputs["cases"]):
        turns = []
        for turn_gold in gold_by_id[case["case_id"]]["turns"]:
            message_terms = [
                group[0]
                for group in turn_gold["recall_term_groups"]
                if group
            ]
            message = "请先说明你的理由。"
            if message_terms:
                message = "我们按" + "、".join(message_terms) + "继续，请先说下一步。"
            turns.append(
                {
                    "turn_id": turn_gold["turn_id"],
                    "teacher_message": message,
                    "primary_skill_id": turn_gold["allowed_primary_skill_ids"][0],
                    "skill_switched": bool(turn_gold["expected_switch"]),
                    "memory_status": turn_gold["expected_memory_status"],
                    "memory_evidence_turn_ids": (
                        [case["turns"][0]["turn_id"]]
                        if turn_gold["expected_memory_status"]
                        == "resolved_evidence_linked"
                        else []
                    ),
                    "active_misconception_tags": list(
                        turn_gold["expected_active_misconception_tags"]
                    ),
                    "resolved_misconception_tags": list(
                        turn_gold["expected_resolved_misconception_tags"]
                    ),
                    "resolution_evidence_turn_ids": (
                        [turn_gold["turn_id"]]
                        if turn_gold["required_resolution_evidence"]
                        else []
                    ),
                    "prompt_injection_blocked": bool(
                        turn_gold["prompt_injection_blocked"]
                    ),
                    "terminal": bool(turn_gold["should_stop"]),
                    "deterministic_fallback": False,
                }
            )
        cases.append(
            {
                "case_id": case["case_id"],
                "session_instance_id": f"session-{case_index}-{case['case_id']}",
                "turns": turns,
                "final_status": "active",
            }
        )
    return {
        "schema": PREDICTIONS_SCHEMA,
        "benchmark_version": BENCHMARK_VERSION,
        "benchmark_id": inputs["benchmark_id"],
        "input_fingerprint": _fingerprint(inputs),
        "runtime": {
            "provider": "scripted_test_double",
            "model": "none",
            "agent_loop_enabled": True,
        },
        "cases": cases,
    }


def _seal_event(event: dict) -> dict:
    result = deepcopy(event)
    result["event_sha256"] = _fingerprint(result)
    return result


def _sealed_loop_summary(skill_id: str, *, bounded: bool) -> dict:
    result = {
        "schema": "teaching_skill_miner.teacher_agent_loop.v1",
        "status": "route_ready",
        "termination_reason": (
            "max_steps exceeded; using the last validated route"
            if bounded
            else "route ready"
        ),
        "steps": 4,
        "model_call_count": 0,
        "tool_call_count": 0,
        "retry_count": 0,
        "selected_skill_id": skill_id,
        "supporting_skill_ids": [],
        "next_focus": "conceptual",
        "deterministic_fallback": False,
        "bounded_route_completion": bounded,
        "events": [],
    }
    result["trace_sha256"] = _fingerprint(result)
    return result


def _sealed_lifecycle_receipt(turn: dict, loop_summary: dict) -> dict:
    selected = turn["primary_skill_id"]
    initial = "skill_diagnostic_questioning"
    events = [
        _seal_event(
            {
                "sequence": 1,
                "event": "observe",
                "status": "observed",
                "observation_present": True,
                "evidence_count": 1,
                "source": "learner_turn",
            }
        ),
        _seal_event(
            {
                "sequence": 2,
                "event": "assess",
                "status": "assessed",
                "signal": "partial",
                "confidence": 0.8,
                "needs_human_review": False,
                "source": "bounded_test_assessment",
            }
        ),
        _seal_event(
            {
                "sequence": 3,
                "event": "route",
                "status": "routed",
                "selected_skill_id": initial,
                "supporting_skill_ids": [],
                "route_authority": "validated_agent_loop",
                "contract_validated": True,
                "reason_codes": ["route_proposal_rejected"],
            }
        ),
        _seal_event(
            {
                "sequence": 4,
                "event": "route",
                "status": "routed",
                "selected_skill_id": selected,
                "supporting_skill_ids": [],
                "route_authority": "validated_plan",
                "contract_validated": True,
                "reason_codes": ["route_replanned"],
            }
        ),
        _seal_event(
            {
                "sequence": 5,
                "event": "act",
                "status": "materialized",
                "selected_skill_id": selected,
                "action_type": "bounded_test_action",
                "action_materialized": True,
                "action_sha256": _fingerprint({"skill_id": selected}),
            }
        ),
        _seal_event(
            {
                "sequence": 6,
                "event": "commit",
                "status": "committed",
                "committed": True,
                "commit_round": 1,
                "session_round_matches": True,
            }
        ),
    ]
    result = {
        "schema": "teaching_skill_miner.teacher_agent_orchestration.v1",
        "state_version": 1,
        "artifact_kind": "committed_turn_lifecycle_receipt",
        "lifecycle_mode": "event_derived",
        "status": "completed",
        "phase": "commit",
        "turn_outcome": "commit",
        "commit_round": 1,
        "session_fingerprint": _fingerprint({"turn_id": turn["turn_id"]}),
        "cycle": 2,
        "replan_count": 1,
        "replan": {
            "occurred": True,
            "count": 1,
            "route_event_count": 2,
            "route_changed": True,
            "initial_skill_id": initial,
            "final_skill_id": selected,
            "reason_codes": ["route_proposal_rejected", "route_replanned"],
        },
        "route_authority": "validated_plan",
        "route": {
            "authority": "validated_plan",
            "authority_source": "lifecycle_event",
            "selected_skill_id": selected,
            "supporting_skill_ids": [],
            "contract_validated": True,
        },
        "selected_skill_id": selected,
        "supporting_skill_ids": [],
        "next_focus": "conceptual",
        "action_type": "bounded_test_action",
        "action_sha256": _fingerprint({"skill_id": selected}),
        "loop_trace_sha256": loop_summary["trace_sha256"],
        "events": events,
        "phases": [
            {"phase": event["event"], "status": event["status"], "sequence": event["sequence"]}
            for event in events
        ],
        "verification": {
            "status": "verified",
            "checks": ["ordered_observe_assess_route_act_complete"],
            "failures": [],
        },
        "uncertainty": {
            "score": 0.2,
            "level": "low",
            "sources": ["route_replanned"],
            "needs_human_review": False,
        },
        "claim_boundary": {
            "model_reasoning_persisted": False,
            "raw_learner_text_persisted": False,
            "assessment_excerpt_persisted": False,
            "tool_payloads_persisted": False,
            "teacher_message_persisted": False,
            "input_events_persisted_verbatim": False,
            "allowlisted_event_projection_only": True,
            "commit_established": True,
            "learning_effect_established": False,
        },
    }
    result["checkpoint_sha256"] = _fingerprint(result)
    return result


def test_model_trace_accepts_bounded_deepseek_prompt_cache_usage() -> None:
    benchmark_v2._validate_model_trace(
        {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "usage": {
                "prompt_tokens": 300,
                "prompt_cache_hit_tokens": 256,
                "prompt_cache_miss_tokens": 44,
            },
        },
        field="model_trace",
    )


def test_development_fixture_is_input_gold_separated_and_valid(artifacts) -> None:
    inputs, gold, library = artifacts

    validate_benchmark_inputs(inputs, library)
    validate_benchmark_gold(gold, inputs, library)

    assert inputs["split"] == "development"
    assert inputs["claim_boundary"]["held_out_after_prompt_development"] is False
    assert inputs["claim_boundary"]["real_learning_effect_established"] is False
    assert len(inputs["cases"]) == 6
    assert sum(len(case["turns"]) for case in inputs["cases"]) == 20
    assert "allowed_primary_skill_ids" not in json.dumps(
        inputs, ensure_ascii=False
    )


def test_score_exposes_product_dimensions_without_claiming_accuracy(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)

    report = score_benchmark_v2(inputs, gold, predictions, library)

    assert report["metrics"]["long_horizon_memory"]["recall_group_coverage"] == 1.0
    assert report["metrics"]["misconception_resolution"][
        "resolution_exact_rate"
    ] == 1.0
    assert report["metrics"]["skill_switching"]["allowed_skill_hit_rate"] == 1.0
    assert report["metrics"]["skill_switching"]["switch_f1"] == 1.0
    assert report["metrics"]["prompt_injection"]["direct_answer_leak_rate"] == 0.0
    assert report["metrics"]["cross_session_isolation"]["leakage_rate"] == 0.0
    assert report["metrics"]["learning_outcome"]["record_count"] == 3
    assert report["metrics"]["learning_outcome"]["mean_absolute_gain"] > 0
    assert report["metrics"]["learning_outcome"][
        "mean_delayed_retention_ratio"
    ] is not None
    assert report["claim_boundary"]["metrics_are_not_accuracy"] is True
    assert report["claim_boundary"]["real_learning_effect_established"] is False
    assert report["claim_boundary"]["lifecycle_receipts_externally_attested"] is False
    assert "请先说明" not in json.dumps(report, ensure_ascii=False)


def test_score_preserves_live_lifecycle_and_loop_telemetry(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    for case in predictions["cases"]:
        for turn_index, turn in enumerate(case["turns"]):
            turn["loop_summary"] = _sealed_loop_summary(
                turn["primary_skill_id"], bounded=turn_index == 0
            )
            turn["lifecycle_receipt"] = _sealed_lifecycle_receipt(
                turn, turn["loop_summary"]
            )
            turn["fallback_counts"] = {
                "agent_loop_fallback_count": 0,
                "planner_fallback_count": 0,
                "action_fallback_count": 0,
                "assessment_failure_count": 0,
            }
    predictions["runtime"]["fallback_totals"] = {
        "agent_loop_fallback_count": 0,
        "planner_fallback_count": 0,
        "action_fallback_count": 0,
        "assessment_failure_count": 0,
    }

    report = score_benchmark_v2(inputs, gold, predictions, library)
    telemetry = report["metrics"]["diagnostic_telemetry"]

    assert telemetry["lifecycle_receipt_coverage"] == 1.0
    assert telemetry["lifecycle_commit_verification_rate"] == 1.0
    assert telemetry["lifecycle_route_contract_rate"] == 1.0
    assert telemetry["lifecycle_explicit_replan_rate"] == 1.0
    assert telemetry["bounded_route_completion_rate"] == 0.3
    assert report["runtime"]["fallback_telemetry_complete"] is True


def test_missing_new_fallback_telemetry_is_not_materialized_as_zero(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    turn = predictions["cases"][0]["turns"][0]
    turn["loop_summary"] = _sealed_loop_summary(turn["primary_skill_id"], bounded=False)

    report = score_benchmark_v2(inputs, gold, predictions, library)

    assert report["runtime"]["fallback_telemetry_complete"] is False
    assert report["runtime"]["fallback_totals"] is None


def test_score_report_drops_model_authored_loop_reason(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    turn = predictions["cases"][0]["turns"][0]
    loop_summary = _sealed_loop_summary(turn["primary_skill_id"], bounded=False)
    loop_summary["termination_reason"] = "PRIVATE_LOOP_SENTINEL"
    material = dict(loop_summary)
    material.pop("trace_sha256", None)
    loop_summary["trace_sha256"] = _fingerprint(material)
    turn["loop_summary"] = loop_summary

    report = score_benchmark_v2(inputs, gold, predictions, library)

    assert "PRIVATE_LOOP_SENTINEL" not in json.dumps(report, ensure_ascii=False)


def test_fallback_loop_summary_may_have_no_selected_skill(artifacts) -> None:
    inputs, gold, _library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    predictions["cases"][0]["turns"][0]["loop_summary"] = public_agent_loop_trace(
        {
            "status": "action_ready",
            "termination_reason": "model unavailable",
            "steps": 1,
            "deterministic_fallback": True,
            "loop_state": {
                "selected_skill_id": None,
                "supporting_skill_ids": [],
                "next_focus": "conceptual",
            },
            "events": [
                {
                    "type": "model_error",
                    "step": 1,
                    "attempt": 1,
                    "error": "RuntimeError: unavailable",
                },
                {"type": "fallback", "step": 1, "reason": "model unavailable"},
            ],
        }
    )

    validate_predictions(predictions, inputs)


def test_live_benchmark_pads_turns_after_terminal_without_readvance(
    artifacts, monkeypatch
) -> None:
    inputs, _gold, library = artifacts
    advance_calls: list[str] = []

    def fake_session(*_args, **_kwargs):
        return {
            "status": "active",
            "round": 0,
            "skill_library": deepcopy(library),
            "integrity": {"content_sha256": "0" * 64},
            "current_action": {
                "primary_skill": {"skill_id": "skill_diagnostic_questioning"},
                "supporting_skills": [],
                "teacher_action": {
                    "type": "probe_prior_knowledge",
                    "message": "先说出你的思路。",
                },
                "action_provenance": {"route_adjudication": {"changed": False}},
                "decision_origin": "model",
            },
            "student_state": {"misconceptions": []},
            "context_memory": {
                "semantic_summary": {
                    "continuity_recall": {
                        "status": "not_requested",
                        "evidence_refs": [],
                    }
                }
            },
            "history": [],
            "agent_runtime": {
                "agent_loop_fallback_count": 0,
                "planner_fallback_count": 0,
                "action_fallback_count": 0,
                "assessment_failure_count": 0,
                "last_agent_loop": None,
            },
        }

    def fake_advance(session, *, learner_response, client, options):
        advance_calls.append(learner_response)
        updated = deepcopy(session)
        updated["status"] = "succeeded"
        updated["agent_runtime"]["agent_loop_fallback_count"] = 1
        return updated

    class _FakeClient:
        def public_status(self):
            return {"provider": "test", "model": "terminal-guard"}

    monkeypatch.setattr(
        "teaching_skill_miner.teacher_agent_benchmark_v2.start_live_teacher_agent_session",
        fake_session,
    )
    monkeypatch.setattr(
        "teaching_skill_miner.teacher_agent_benchmark_v2.advance_live_teacher_agent_session",
        fake_advance,
    )

    predictions = predictions_from_live_cases(inputs, library, _FakeClient())

    assert len(advance_calls) == len(inputs["cases"])
    assert all(
        len(predicted["turns"]) == len(source["turns"])
        for predicted, source in zip(predictions["cases"], inputs["cases"])
    )
    assert all(
        turn["terminal"]
        and turn["teacher_message"] == ""
        and turn["lifecycle_receipt"]["status"] == "aborted"
        for predicted in predictions["cases"]
        for turn in predicted["turns"][1:]
    )
    assert all(
        turn["fallback_counts"]["agent_loop_fallback_count"] == 0
        for predicted in predictions["cases"]
        for turn in predicted["turns"][1:]
    )
    assert predictions["runtime"]["fallback_totals"]["agent_loop_fallback_count"] == len(
        inputs["cases"]
    )


def test_unsealed_lifecycle_telemetry_is_rejected(artifacts) -> None:
    inputs, gold, _library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    turn = predictions["cases"][0]["turns"][0]
    turn["loop_summary"] = _sealed_loop_summary(turn["primary_skill_id"], bounded=False)
    receipt = _sealed_lifecycle_receipt(turn, turn["loop_summary"])
    receipt["route"]["selected_skill_id"] = "fabricated_skill"
    # The route mutation is intentionally not resealed; a scorer must fail
    # closed instead of counting a forged lifecycle commit.
    turn["lifecycle_receipt"] = receipt

    with pytest.raises(TeachingAgentBenchmarkV2Error, match="integrity check|does not match"):
        validate_predictions(predictions, inputs)


def test_resealed_forged_lifecycle_order_is_rejected(artifacts) -> None:
    """Self-consistent hashes cannot turn an invalid event stream into a commit."""

    inputs, gold, _library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    turn = predictions["cases"][0]["turns"][0]
    turn["loop_summary"] = _sealed_loop_summary(turn["primary_skill_id"], bounded=False)
    receipt = _sealed_lifecycle_receipt(turn, turn["loop_summary"])
    for event in receipt["events"]:
        event["event"] = "commit"
        event["status"] = "committed"
        for key in (
            "observation_present",
            "evidence_count",
            "signal",
            "confidence",
            "needs_human_review",
            "source",
            "selected_skill_id",
            "supporting_skill_ids",
            "route_authority",
            "contract_validated",
            "reason_codes",
            "action_type",
            "action_materialized",
            "action_sha256",
            "commit_round",
            "session_round_matches",
        ):
            event.pop(key, None)
        event["event_sha256"] = _fingerprint(
            {key: value for key, value in event.items() if key != "event_sha256"}
        )
    receipt["phases"] = [
        {
            "phase": event["event"],
            "status": event["status"],
            "sequence": event["sequence"],
        }
        for event in receipt["events"]
    ]
    receipt["checkpoint_sha256"] = _fingerprint(
        {key: value for key, value in receipt.items() if key != "checkpoint_sha256"}
    )
    turn["lifecycle_receipt"] = receipt

    with pytest.raises(
        TeachingAgentBenchmarkV2Error,
        match="commit events|final route|final act|verified|precedes",
    ):
        validate_predictions(predictions, inputs)


def test_resealed_lifecycle_status_retyping_and_post_terminal_event_are_rejected(
    artifacts,
) -> None:
    inputs, gold, _library = artifacts
    base_predictions = _perfect_predictions(inputs, gold)
    turn = base_predictions["cases"][0]["turns"][0]
    turn["loop_summary"] = _sealed_loop_summary(turn["primary_skill_id"], bounded=False)
    base_receipt = _sealed_lifecycle_receipt(turn, turn["loop_summary"])

    def reseal(receipt: dict) -> None:
        for event in receipt["events"]:
            event["event_sha256"] = _fingerprint(
                {key: value for key, value in event.items() if key != "event_sha256"}
            )
        receipt["phases"] = [
            {
                "phase": event["event"],
                "status": event["status"],
                "sequence": event["sequence"],
            }
            for event in receipt["events"]
        ]
        receipt["checkpoint_sha256"] = _fingerprint(
            {key: value for key, value in receipt.items() if key != "checkpoint_sha256"}
        )

    retyped = deepcopy(base_predictions)
    retyped_receipt = deepcopy(base_receipt)
    retyped_receipt["events"][0]["status"] = "committed"
    reseal(retyped_receipt)
    retyped["cases"][0]["turns"][0]["lifecycle_receipt"] = retyped_receipt
    with pytest.raises(TeachingAgentBenchmarkV2Error, match="status does not match"):
        validate_predictions(retyped, inputs)

    post_terminal = deepcopy(base_predictions)
    post_terminal_receipt = deepcopy(base_receipt)
    post_terminal_receipt["events"].append(
        {
            "sequence": 7,
            "event": "observe",
            "observation_present": True,
            "evidence_count": 1,
            "status": "observed",
        }
    )
    reseal(post_terminal_receipt)
    post_terminal["cases"][0]["turns"][0]["lifecycle_receipt"] = post_terminal_receipt
    with pytest.raises(TeachingAgentBenchmarkV2Error, match="after terminal"):
        validate_predictions(post_terminal, inputs)


def test_terminal_flags_and_status_contract_cannot_be_forged(artifacts) -> None:
    inputs, gold, library = artifacts

    def reseal(receipt: dict) -> None:
        for event in receipt["events"]:
            event["event_sha256"] = _fingerprint(
                {key: value for key, value in event.items() if key != "event_sha256"}
            )
        receipt["phases"] = [
            {
                "phase": event["event"],
                "status": event["status"],
                "sequence": event["sequence"],
            }
            for event in receipt["events"]
        ]
        receipt["checkpoint_sha256"] = _fingerprint(
            {key: value for key, value in receipt.items() if key != "checkpoint_sha256"}
        )

    commit_predictions = _perfect_predictions(inputs, gold)
    commit_turn = commit_predictions["cases"][0]["turns"][0]
    commit_turn["loop_summary"] = _sealed_loop_summary(
        commit_turn["primary_skill_id"], bounded=False
    )
    commit_receipt = _sealed_lifecycle_receipt(commit_turn, commit_turn["loop_summary"])
    commit_receipt["events"][-1].pop("committed")
    reseal(commit_receipt)
    commit_turn["lifecycle_receipt"] = commit_receipt
    with pytest.raises(TeachingAgentBenchmarkV2Error, match="committed=true"):
        validate_predictions(commit_predictions, inputs)

    abort_predictions = _perfect_predictions(inputs, gold)
    case = inputs["cases"][0]
    session = start_teacher_agent_session(
        deepcopy(inputs["goals"][case["goal_ref"]]),
        deepcopy(inputs["student_profiles"][case["student_profile_ref"]]),
        deepcopy(library),
    )
    abort_receipt = _terminal_guard_lifecycle_receipt({**session, "status": "succeeded"})
    abort_receipt["events"][-1].pop("aborted")
    reseal(abort_receipt)
    abort_predictions["cases"][0]["turns"][0]["lifecycle_receipt"] = abort_receipt
    with pytest.raises(TeachingAgentBenchmarkV2Error, match="aborted=true"):
        validate_predictions(abort_predictions, inputs)

    blocked_predictions = _perfect_predictions(inputs, gold)
    blocked = build_turn_lifecycle_receipt(
        session,
        loop_trace=None,
        plan=None,
        output_action=None,
        lifecycle_events=[
            {"event": "observe", "observed": False},
            {"event": "assess", "signal": "partial", "confidence": 0.8},
            {
                "event": "route",
                "selected_skill_id": "skill_diagnostic_questioning",
                "route_authority": "validated_agent_loop",
            },
            {
                "event": "act",
                "selected_skill_id": "skill_diagnostic_questioning",
                "action_type": "probe_prior_knowledge",
                "action_materialized": True,
            },
            {"event": "commit", "committed": True, "round": 0},
        ],
        turn_outcome="commit",
        commit_round=0,
    )
    blocked["verification"]["status"] = "fallback"
    reseal(blocked)
    blocked_predictions["cases"][0]["turns"][0]["lifecycle_receipt"] = blocked
    with pytest.raises(TeachingAgentBenchmarkV2Error, match="status/outcome/verification"):
        validate_predictions(blocked_predictions, inputs)


def test_resealed_terminal_polarity_and_status_contract_are_rejected(artifacts) -> None:
    inputs, gold, library = artifacts
    base_predictions = _perfect_predictions(inputs, gold)
    turn = base_predictions["cases"][0]["turns"][0]
    turn["loop_summary"] = _sealed_loop_summary(turn["primary_skill_id"], bounded=False)

    def reseal(receipt: dict) -> None:
        for event in receipt["events"]:
            event["event_sha256"] = _fingerprint(
                {key: value for key, value in event.items() if key != "event_sha256"}
            )
        receipt["phases"] = [
            {
                "phase": event["event"],
                "status": event["status"],
                "sequence": event["sequence"],
            }
            for event in receipt["events"]
        ]
        receipt["checkpoint_sha256"] = _fingerprint(
            {key: value for key, value in receipt.items() if key != "checkpoint_sha256"}
        )

    missing_commit_flag = deepcopy(base_predictions)
    commit_receipt = _sealed_lifecycle_receipt(
        missing_commit_flag["cases"][0]["turns"][0],
        missing_commit_flag["cases"][0]["turns"][0]["loop_summary"],
    )
    commit_receipt["events"][-1].pop("committed", None)
    reseal(commit_receipt)
    missing_commit_flag["cases"][0]["turns"][0]["lifecycle_receipt"] = commit_receipt
    with pytest.raises(TeachingAgentBenchmarkV2Error, match="committed=true"):
        validate_predictions(missing_commit_flag, inputs)

    inconsistent_status = deepcopy(base_predictions)
    inconsistent_receipt = _sealed_lifecycle_receipt(
        inconsistent_status["cases"][0]["turns"][0],
        inconsistent_status["cases"][0]["turns"][0]["loop_summary"],
    )
    inconsistent_receipt["status"] = "blocked"
    inconsistent_receipt["turn_outcome"] = "invalid"
    inconsistent_receipt["verification"]["status"] = "rejected"
    inconsistent_receipt["claim_boundary"]["commit_established"] = False
    reseal(inconsistent_receipt)
    inconsistent_status["cases"][0]["turns"][0]["lifecycle_receipt"] = (
        inconsistent_receipt
    )
    with pytest.raises(TeachingAgentBenchmarkV2Error, match="blocked receipt lacks rejected"):
        validate_predictions(inconsistent_status, inputs)

    missing_abort_flag = deepcopy(base_predictions)
    case = inputs["cases"][0]
    session = start_teacher_agent_session(
        deepcopy(inputs["goals"][case["goal_ref"]]),
        deepcopy(inputs["student_profiles"][case["student_profile_ref"]]),
        deepcopy(library),
    )
    abort_receipt = build_turn_lifecycle_receipt(
        session,
        loop_trace=None,
        plan=None,
        lifecycle_events=[
            {"event": "observe", "observation_present": True},
            {"event": "abort", "aborted": True, "reason_codes": ["cancelled"]},
        ],
        turn_outcome="abort",
    )
    abort_receipt["events"][-1].pop("aborted", None)
    reseal(abort_receipt)
    missing_abort_flag["cases"][0]["turns"][0]["loop_summary"] = None
    missing_abort_flag["cases"][0]["turns"][0]["lifecycle_receipt"] = abort_receipt
    with pytest.raises(TeachingAgentBenchmarkV2Error, match="aborted=true"):
        validate_predictions(missing_abort_flag, inputs)


def test_runtime_fallback_totals_are_bound_to_turn_deltas(artifacts) -> None:
    inputs, gold, _library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    predictions["runtime"]["fallback_totals"] = {
        "agent_loop_fallback_count": 0,
        "planner_fallback_count": 1,
        "action_fallback_count": 0,
        "assessment_failure_count": 0,
    }

    with pytest.raises(TeachingAgentBenchmarkV2Error, match="fallback_totals.*does not match"):
        validate_predictions(predictions, inputs)


def test_live_prediction_receipt_carries_initial_fallback_into_first_turn(
    artifacts, monkeypatch
) -> None:
    inputs, _gold, library = artifacts
    calls = {"start": 0}

    def fake_session(*_args, **_kwargs):
        calls["start"] += 1
        return {
            "integrity": {"content_sha256": "a" * 64},
            "agent_runtime": {
                "agent_loop_fallback_count": 1,
                "planner_fallback_count": 1,
                "action_fallback_count": 1,
                "assessment_failure_count": 1,
                "last_agent_loop": None,
            },
            "current_action": {
                "primary_skill": {"skill_id": "skill_diagnostic_questioning"},
                "teacher_action": {
                    "type": "probe_prior_knowledge",
                    "message": "请说出一个前置概念。",
                    "direct_answer_prohibited": True,
                },
                "action_provenance": {
                    "route_adjudication": {"changed": False}
                },
                "skill_switched": False,
            },
            "student_state": {
                "misconceptions": [],
                "next_focus": {"dimension": "prerequisite"},
            },
            "context_memory": {
                "semantic_summary": {
                    "continuity_recall": {
                        "status": "not_requested",
                        "evidence_refs": [],
                    }
                }
            },
            "history": [],
            "status": "active",
        }

    def fake_advance(session, **_kwargs):
        session["history"].append(
            {
                "deepseek_assessment": {
                    "signal": "partial",
                    "confidence": 0.5,
                    "evidence_excerpt": "学生给出了一部分解释",
                },
                "turn_lifecycle": None,
            }
        )
        return session

    monkeypatch.setattr(benchmark_v2, "start_live_teacher_agent_session", fake_session)
    monkeypatch.setattr(
        benchmark_v2, "advance_live_teacher_agent_session", fake_advance
    )

    class FakeClient:
        @staticmethod
        def public_status():
            return {"provider": "test", "model": "none"}

    predictions = benchmark_v2.predictions_from_live_cases(
        inputs,
        library,
        client=FakeClient(),
        options=benchmark_v2.LiveAgentOptions(agent_loop_enabled=False),
    )
    expected = len(inputs["cases"])
    assert all(
        case["turns"][0]["fallback_counts"]
        == {
            "agent_loop_fallback_count": 1,
            "planner_fallback_count": 1,
            "action_fallback_count": 1,
            "assessment_failure_count": 1,
        }
        for case in predictions["cases"]
    )
    assert predictions["runtime"]["fallback_totals"] == {
        "agent_loop_fallback_count": expected,
        "planner_fallback_count": expected,
        "action_fallback_count": expected,
        "assessment_failure_count": expected,
    }


def test_live_prediction_pads_turns_after_terminal_session(
    artifacts, monkeypatch
) -> None:
    """A model stop must not make later benchmark turns crash the executor."""

    inputs, _gold, library = artifacts

    def fake_session(goal, profile, skill_library, *_args, **_kwargs):
        session = start_teacher_agent_session(
            deepcopy(goal), deepcopy(profile), deepcopy(skill_library)
        )
        session["status"] = "active"
        return session

    def fake_advance(session, **_kwargs):
        session["status"] = "succeeded"
        session.setdefault("history", []).append(
            {
                "deepseek_assessment": {
                    "signal": "correct",
                    "confidence": 0.9,
                    "evidence_excerpt": "bounded test evidence",
                },
                "turn_lifecycle": None,
            }
        )
        return session

    monkeypatch.setattr(
        benchmark_v2, "start_live_teacher_agent_session", fake_session
    )
    monkeypatch.setattr(
        benchmark_v2, "advance_live_teacher_agent_session", fake_advance
    )

    class FakeClient:
        @staticmethod
        def public_status():
            return {"provider": "test", "model": "none"}

    predictions = benchmark_v2.predictions_from_live_cases(
        inputs,
        library,
        client=FakeClient(),
        options=benchmark_v2.LiveAgentOptions(agent_loop_enabled=False),
    )
    for input_case, prediction_case in zip(
        inputs["cases"], predictions["cases"], strict=True
    ):
        assert len(prediction_case["turns"]) == len(input_case["turns"])
        assert prediction_case["turns"][0]["terminal"] is True
        padded = prediction_case["turns"][1:]
        assert padded
        assert all(row["teacher_message"] == "" for row in padded)
        assert all(
            row["assessment_source"] == "terminal_session_no_advance"
            for row in padded
        )
        assert all(
            row["lifecycle_receipt"]["status"] == "aborted"
            and [event["event"] for event in row["lifecycle_receipt"]["events"]]
            == ["observe", "abort"]
            for row in padded
        )
    validate_predictions(predictions, inputs)


def test_event_derived_abort_and_blocked_receipts_remain_valid(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    case = inputs["cases"][0]
    session = start_teacher_agent_session(
        deepcopy(inputs["goals"][case["goal_ref"]]),
        deepcopy(inputs["student_profiles"][case["student_profile_ref"]]),
        deepcopy(library),
    )
    receipt = build_turn_lifecycle_receipt(
        session,
        loop_trace=None,
        plan=None,
        lifecycle_events=[
            {"event": "observe", "observation_present": True},
            {
                "event": "abort",
                "aborted": True,
                "reason_codes": ["client_cancelled"],
            },
        ],
        turn_outcome="abort",
    )
    predictions["cases"][0]["turns"][0]["lifecycle_receipt"] = receipt
    action = {
        "type": "probe_prior_knowledge",
        "primary_skill": {"skill_id": "skill_diagnostic_questioning"},
        "supporting_skills": [],
        "next_focus": "prerequisite",
    }
    blocked = build_turn_lifecycle_receipt(
        session,
        loop_trace=None,
        plan=None,
        output_action=action,
        route_authority="state_first_policy",
        lifecycle_events=[
            {"event": "observe", "observed": False},
            {"event": "assess", "signal": "partial", "confidence": 0.8},
            {
                "event": "route",
                "selected_skill_id": "skill_diagnostic_questioning",
            },
            {
                "event": "act",
                "selected_skill_id": "skill_diagnostic_questioning",
                "action_type": "probe_prior_knowledge",
                "action_materialized": True,
            },
            {"event": "commit", "round": 0},
        ],
        turn_outcome="commit",
        commit_round=0,
    )
    assert blocked["status"] == "blocked"
    assert blocked["events"][0]["failure_code"] == "observation_fact_missing"
    predictions["cases"][0]["turns"][1]["lifecycle_receipt"] = blocked

    validate_predictions(predictions, inputs)


def test_cross_session_leak_and_reused_instance_are_detected(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    by_id = {row["case_id"]: row for row in predictions["cases"]}
    by_id["cross_session_beta"]["session_instance_id"] = by_id[
        "cross_session_alpha"
    ]["session_instance_id"]
    by_id["cross_session_beta"]["turns"][0][
        "teacher_message"
    ] = "我记得 ALPHA_MEMORY_SENTINEL。"

    report = score_benchmark_v2(inputs, gold, predictions, library)

    isolation = report["metrics"]["cross_session_isolation"]
    assert isolation["unique_session_instance_rate"] == 0.0
    assert isolation["leakage_rate"] > 0.0


def test_input_artifact_rejects_embedded_gold(artifacts) -> None:
    inputs, _gold, library = artifacts
    tainted = deepcopy(inputs)
    tainted["cases"][0]["gold"] = {"expected_switch": True}

    with pytest.raises(TeachingAgentBenchmarkV2Error, match="gold-only"):
        validate_benchmark_inputs(tainted, library)


def test_failed_injection_block_is_counted_in_denominator(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    injection_case = next(
        row
        for row in predictions["cases"]
        if row["case_id"] == "prompt_injection_resistance"
    )
    injection_case["turns"][0]["prompt_injection_blocked"] = False

    report = score_benchmark_v2(inputs, gold, predictions, library)

    metric = report["metrics"]["prompt_injection"]
    assert metric["blocked_case_rate"] < 1.0
    assert metric["blocked_case_rate"] == 0.5


def test_missed_misconception_resolution_is_counted_in_denominator(artifacts) -> None:
    inputs, gold, library = artifacts
    predictions = _perfect_predictions(inputs, gold)
    misconception_case = next(
        row for row in predictions["cases"] if row["case_id"] == "misconception_resolution"
    )
    misconception_case["turns"][0]["active_misconception_tags"] = []

    report = score_benchmark_v2(inputs, gold, predictions, library)

    metric = report["metrics"]["misconception_resolution"]
    assert metric["resolution_exact_rate"] < 1.0
    assert metric["resolution_exact_rate"] == 0.666667
