from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

from teaching_skill_miner.deepseek_client import DeepSeekClient, DeepSeekConfig
from teaching_skill_miner.io_utils import read_json
from teaching_skill_miner.student_model import (
    initialize_student_model,
    update_student_model,
)
from teaching_skill_miner.teacher_agent import _refresh_integrity, validate_session
from teaching_skill_miner.teacher_agent_curriculum import (
    build_teacher_owned_curriculum_spec,
    create_teacher_curriculum_authority_receipt,
    seal_teacher_owned_curriculum_blueprint,
)
from teaching_skill_miner.teacher_agent_curriculum_authority_store import (
    curriculum_runtime_authority_projection,
)
from teaching_skill_miner.teacher_agent_learning_records import mint_learner_key
from teaching_skill_miner.teacher_agent_live import (
    LiveTeacherAgentError,
    live_metacognition_prompt_contract,
    pair_live_metacognitive_outcome,
    record_live_metacognitive_prediction,
    start_live_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_metacognition import MetacognitionStore
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardSnapshot,
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


def _client() -> DeepSeekClient:
    plan = {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": "not_observed",
            "confidence": 0.0,
            "answer_alignment": "not_applicable",
            "matched_concepts": [],
            "missing_concepts": [],
            "diagnosis_reason": "initial action",
            "evidence_excerpt": "",
            "misconception_tag": None,
            "misconception_description": "",
            "resolved_misconception_tags": [],
            "response_quality": "empty",
            "engagement_level": "unknown",
            "needs_human_review": False,
        },
        "decision": {
            "primary_skill_id": "skill_diagnostic_questioning",
            "supporting_skill_ids": [],
            "selection_reason": "begin with a diagnostic",
            "next_focus": "prerequisite",
        },
        "teacher_action": {
            "type": "probe_prior_knowledge",
            "message": "请先说明你知道的递归分解。",
            "expected_signal": "说明一个递归分解关系。",
            "question_contract": {
                "answer_type": "explanation",
                "target_concepts": ["递归分解"],
                "accepted_aliases": [],
                "success_criteria": ["说明一个递归分解关系"],
            },
        },
        "stop_recommendation": {"should_stop": False, "reason": ""},
    }

    def transport(_url: str, _headers: dict, _payload: bytes, _timeout: float):
        envelope = {
            "id": "metacognition-live-test",
            "choices": [{"message": {"content": json.dumps(plan, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10},
        }
        return 200, json.dumps(envelope).encode()

    return DeepSeekClient(
        DeepSeekConfig(allow_remote_student_data=True),
        api_key="test-key",
        transport=transport,
    )


def _assessment_session() -> tuple[dict, dict]:
    library = read_json(ROOT / "data/teacher_agent_skill_library_v2.json")
    demo = read_json(ROOT / "data/teacher_agent_demo_input.json")
    signing_key = Ed25519PrivateKey.generate()
    private_pem = signing_key.private_bytes(
        serialization.Encoding.PEM,
        serialization.PrivateFormat.PKCS8,
        serialization.NoEncryption(),
    )
    public_key = signing_key.public_key().public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    key_id = "metacognition-test-teacher-key"
    spec = build_teacher_owned_curriculum_spec(
        title="元认知权威课节",
        lessons=[
            {
                "legacy_lesson_id": "lesson_01_01",
                "title": "状态定义",
                "objective": "解释状态下标与保存值分别表示什么。",
                "knowledge_components": [
                    {
                        "label": "状态定义",
                        "prerequisites": [],
                        "source_resource_ids": ["teacher_source"],
                    }
                ],
            }
        ],
        source_spans=[
            {
                "resource_id": "teacher_source",
                "content_sha256": sha256(b"meta source").hexdigest(),
                "excerpt_sha256": sha256(b"meta excerpt").hexdigest(),
                "locator": {"kind": "page", "start": 1, "end": 1},
            }
        ],
        factual_claims=[
            {
                "statement": "dp[i] 的下标表示问题规模，保存值表示该规模子问题的结果。",
                "knowledge_components": ["状态定义"],
                "source_resource_ids": ["teacher_source"],
            }
        ],
    )
    receipt = create_teacher_curriculum_authority_receipt(
        spec,
        teacher_id_hash=sha256(b"meta teacher").hexdigest(),
        reviewed_at="2026-08-12T02:00:00Z",
        teacher_confirmed_authority=True,
        signing_key_id=key_id,
        private_key_pem=private_pem,
    )
    blueprint = seal_teacher_owned_curriculum_blueprint(
        spec,
        receipt,
        trusted_teacher_public_keys={key_id: public_key},
    )
    syllabus_ref = {
        "syllabus_id": "syl_" + "a" * 24,
        "module_id": "module_01",
        "lesson_id": "lesson_01_01",
        "content_sha256": "b" * 64,
    }
    projection = curriculum_runtime_authority_projection(
        blueprint,
        legacy_lesson_id=syllabus_ref["lesson_id"],
        family_id="syf_" + "c" * 24,
        published_revision_id="syr_" + "d" * 24,
        published_syllabus_id=syllabus_ref["syllabus_id"],
        # syllabus_ref.content_sha256 is the outline digest. Runtime authority
        # independently binds the complete published syllabus document.
        published_syllabus_sha256=sha256(
            b"published metacognition syllabus"
        ).hexdigest(),
        authority_version=1,
        trusted_teacher_public_keys={key_id: public_key},
    )
    assert projection["published_syllabus_sha256"] != syllabus_ref["content_sha256"]
    base_payload = {"goal": {"syllabus_ref": syllabus_ref}}
    sealed_payload = TeacherAgentDashboardSnapshot._sealed_curriculum_lesson_payload(
        base_payload,
        blueprint=blueprint,
        runtime_authority=projection,
    )
    session = start_live_teacher_agent_session(
        sealed_payload["goal"],
        demo["student_profile"],
        library,
        _client(),
        trusted_curriculum_authority=projection,
    )
    assessment_skill = next(
        item
        for item in library["skills"]
        if item["skill_id"] == "skill_socratic_understanding_check"
    )
    action = session["current_action"]
    action["primary_skill"] = deepcopy(assessment_skill)
    action["knowledge_components"] = ["状态定义"]
    action["teacher_action"]["message"] = "请说明 dp[i] 的下标和值分别表示什么。"
    action["teacher_action"]["question_contract"] = {
        "answer_type": "explanation",
        "target_concepts": ["状态定义"],
        "accepted_aliases": [],
        "success_criteria": ["说明 i 对应级数和 dp[i] 对应走法数"],
        "grading_scope": "current_question_only",
    }
    session["student_state"]["assessment_confidence"] = 0.999
    session = _refresh_integrity(session)
    validate_session(session)
    return session, library


def test_live_prompt_record_and_authoritative_pairing_form_reachable_loop() -> None:
    session, _library = _assessment_session()
    prompt = live_metacognition_prompt_contract(session, session_id="session-live-1")
    assert prompt["assessment_confidence_used_as_learner_jol"] is False
    assert prompt["target"]["knowledge_component_label"] == "状态定义"
    assert prompt["attempt"]["question_id"] == session["current_action"]["action_id"]

    evidence_registry: dict[str, dict] = {}
    now = datetime(2026, 8, 12, 8, 0, tzinfo=timezone.utc)
    learner_key = mint_learner_key(
        "student-live-meta",
        tenant_id="school-live-meta",
        secret=b"live-metacognition-test-secret-32bytes",
    )
    with TemporaryDirectory() as directory:
        store = MetacognitionStore(
            Path(directory) / "meta.jsonl",
            authoritative_evidence_resolver=lambda evidence_id: deepcopy(
                evidence_registry[evidence_id]
            ),
            clock=lambda: now,
        )
        prediction = record_live_metacognitive_prediction(
            session,
            store=store,
            learner_key=learner_key,
            session_id="session-live-1",
            question_issued_at_utc=now.isoformat().replace("+00:00", "Z"),
            captured_at_utc=now.isoformat().replace("+00:00", "Z"),
            learner_jol_percent=85,
            strategy_codes=["self_explanation", "checking"],
        )
        assert prediction.applied is True
        assert prediction.event["data"]["learner_jol_percent"] == 85

        target_id = prompt["target"]["knowledge_component_id"]
        model = session["student_state"]["student_model"]
        model = update_student_model(
            model,
            signal="misconception",
            confidence=0.95,
            focus_dimension="conceptual",
            knowledge_component_ids=[target_id],
            answer_alignment="contradicted",
            needs_human_review=False,
            assessment_eligible=True,
            authoritative=True,
            round_number=1,
            evidence_id="evidence.live.meta.1",
            item_id=prompt["attempt"]["item_id"],
            question_id=prompt["attempt"]["question_id"],
            rubric_id=prompt["attempt"]["rubric_id"],
            observed_at="2000-01-01T00:00:01Z",
            time_basis="session_logical",
            source="server_live_test",
        )
        session["student_state"]["student_model"] = model
        session = _refresh_integrity(session)
        validate_session(session)
        evidence = model["knowledge_components"][target_id]["evidence_ledger"][-1]
        evidence_registry[evidence["evidence_id"]] = deepcopy(evidence)
        committed_at = now.isoformat().replace("+00:00", "Z")
        paired = pair_live_metacognitive_outcome(
            session,
            store=store,
            prediction_event_id=prediction.event_id,
            committed_at_utc=committed_at,
            commit_receipt_id="turn_committed:live-meta-1",
        )
        assert paired.event["data"]["authoritative_outcome"] == "incorrect"
        assert paired.event["data"]["calibration_classification"] == "overconfident"
        assert paired.event["data"]["scoring_standard_changed"] is False
        assert paired.event["data"]["mastery_changed_by_metacognition"] is False


def test_dashboard_unsealed_session_rejects_prediction_and_client_outcome() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        snapshot = build_teacher_agent_dashboard_snapshot(
            ROOT / "data/teacher_agent_skill_library_v2.json",
            ROOT / "data/teacher_agent_demo_input.json",
            ROOT / "data/teacher_agent_evaluation_cases.json",
            store_path=root / "sessions.jsonl",
            learning_record_store_path=root / "learning.jsonl",
            metacognition_store_path=root / "metacognition.jsonl",
            learner_key_secret=b"dashboard-metacognition-secret-32bytes",
            learner_tenant_id="school-dashboard-meta",
        )
        profile = deepcopy(snapshot.demo_input["student_profile"])
        profile["profile_ref"] = "student-dashboard-meta"
        started = snapshot.start(
            {
                "goal": snapshot.demo_input["goal"],
                "student_profile": profile,
                "start_idempotency_key": "start.dashboard.meta",
                "profile_revision": "student-dashboard-meta-v1",
                "profile_display_name": "元认知测试学习者",
            }
        )
        record = snapshot.sessions[started["session_id"]]
        with record.lock:
            assessment_skill = next(
                item
                for item in snapshot.library["skills"]
                if item["skill_id"] == "skill_socratic_understanding_check"
            )
            action = record.session["current_action"]
            action["primary_skill"] = deepcopy(assessment_skill)
            action["knowledge_components"] = ["状态定义"]
            action["teacher_action"]["question_contract"] = {
                "answer_type": "explanation",
                "target_concepts": ["状态定义"],
                "accepted_aliases": [],
                "success_criteria": ["说明 i 对应级数和 dp[i] 对应走法数"],
                "grading_scope": "current_question_only",
            }
            record.session["student_state"]["student_model"] = initialize_student_model(
                record.session["student_state"].get("knowledge_mastery", {}),
                goal=record.session["goal"],
            )
            record.session = _refresh_integrity(record.session)
            current = snapshot._response(started["session_id"], record)
        guards = {
            "session_id": current["session_id"],
            "expected_round": current["rounds_completed"],
            "expected_question_id": current["expected_question_id"],
            "expected_context_version": current["context_version"],
            "profile_revision": current["profile_summary"]["profile_revision"],
        }
        with pytest.raises(
            TeacherAgentDashboardError, match="prediction request fields are invalid"
        ):
            snapshot.record_metacognitive_prediction(
                {
                    **guards,
                    "learner_jol_percent": 90,
                    "strategy_codes": ["retrieval"],
                    "review_id": "review_" + "1" * 64,
                    "lease_id": "lease_" + "2" * 64,
                }
            )
        with pytest.raises(
            (TeacherAgentDashboardError, LiveTeacherAgentError),
            match="teacher grading authority|teacher rubric authority",
        ):
            snapshot.record_metacognitive_prediction(
                {
                    **guards,
                    "learner_jol_percent": 90,
                    "strategy_codes": ["retrieval", "checking"],
                }
            )
        pending_projection = snapshot.list_metacognitive_predictions(guards)
        assert pending_projection["contains_learner_answer"] is False
        assert pending_projection["outcome_accepted_from_client"] is False
        assert pending_projection["mastery_changed"] is False
        assert pending_projection["predictions"] == []
        with pytest.raises(
            TeacherAgentDashboardError, match="server-authoritative fields"
        ):
            snapshot.record_metacognitive_prediction(
                {
                    **guards,
                    "learner_jol_percent": 90,
                    "strategy_codes": ["retrieval"],
                    "outcome": "correct",
                }
            )

        with pytest.raises(
            TeacherAgentDashboardError, match="projection request fields"
        ):
            snapshot.list_metacognitive_predictions({**guards, "outcome": "correct"})

        bootstrap = snapshot.bootstrap()
        assert (
            bootstrap["interaction_contract"][
                "metacognition_prediction_and_pairing_enabled"
            ]
            is True
        )
        assert (
            bootstrap["interaction_contract"][
                "metacognition_client_outcome_updates_enabled"
            ]
            is False
        )


def test_dashboard_pairing_uses_real_durable_turn_commit_receipt() -> None:
    session, _library = _assessment_session()
    target_id = next(
        iter(session["student_state"]["student_model"]["knowledge_components"])
    )
    evidence_id = "evidence.durable.meta"
    session["student_state"]["student_model"]["knowledge_components"][target_id][
        "evidence_ledger"
    ] = [{"evidence_id": evidence_id}]
    snapshot = build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library_v2.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
    )
    snapshot.store = SimpleNamespace(  # type: ignore[assignment]
        events=(
            {
                "event_type": "turn_committed",
                "session_id": "session-durable-meta",
                "turn_id": "turn-durable-meta",
                "data": {
                    "record": {"session": session},
                    "commit_receipt_id": "turn_committed:turn-durable-meta",
                },
            },
        )
    )
    assert (
        snapshot._metacognition_turn_commit_receipt(
            session_id="session-durable-meta", evidence_id=evidence_id
        )
        == "turn_committed:turn-durable-meta"
    )
