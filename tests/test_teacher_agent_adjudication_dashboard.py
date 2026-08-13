from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
import io
import base64
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
from pathlib import Path
import threading
import zipfile

import pytest

from teaching_skill_miner.io_utils import project_root
from teaching_skill_miner.cli import main as cli_main
from teaching_skill_miner.student_model import (
    initialize_student_model,
    stable_knowledge_component_id,
    update_student_model,
)
from teaching_skill_miner.teacher_agent import (
    _new_lesson_state,
    _refresh_integrity,
    advance_teacher_agent_session,
)
from teaching_skill_miner.teacher_agent_adjudication import (
    AdjudicationConflictError,
    AdjudicationEvidenceError,
)
from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardSnapshot,
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
)
from teaching_skill_miner.teacher_agent_data_rights import deletion_confirmation
from teaching_skill_miner.teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TEACHER_AUTHORITY_SCHEMA,
    TeacherAuthorityVerifier,
    canonical_bytes,
    canonical_sha256,
)
from teaching_skill_miner.teacher_agent_curriculum import (
    build_teacher_owned_curriculum_spec,
)
from teaching_skill_miner.teacher_agent_curriculum_authority_store import (
    TeachingCurriculumAuthorityStore,
)
from teaching_skill_miner.teacher_agent_curriculum_signing import (
    CurriculumSigningKeyring,
)
from teaching_skill_miner.teacher_agent_syllabus import _seal_generated_syllabus


ROOT = project_root()
LIBRARY = ROOT / "data" / "teacher_agent_skill_library.json"
DEMO = ROOT / "data" / "teacher_agent_demo_input.json"
CASES = ROOT / "data" / "teacher_agent_evaluation_cases.json"


def _snapshot(
    tmp_path: Path,
    *,
    teacher_authority_verifier: TeacherAuthorityVerifier | None = None,
):
    keyring = None
    authority_store = None
    if teacher_authority_verifier is not None:
        keyring = CurriculumSigningKeyring(
            tmp_path / "syllabi" / ".curriculum_signing_keyring.json",
            integrity_key=b"c" * 32,
        )
        authority_store = TeachingCurriculumAuthorityStore(
            tmp_path / "syllabi" / ".curriculum_authority.json",
            scope_id=_AUTHORITY_SCOPE,
            trusted_teacher_public_keys=keyring.trusted_public_keys,
            gateway_receipt_validator=(
                teacher_authority_verifier.verify_verification_receipt
            ),
        )
    snapshot = build_teacher_agent_dashboard_snapshot(
        LIBRARY,
        DEMO,
        CASES,
        store_path=tmp_path / "sessions.jsonl",
        syllabus_store_path=tmp_path / "syllabi",
        project_store_path=tmp_path / "projects",
        resource_index_store_path=tmp_path / "resources",
        adjudication_store_path=tmp_path / "adjudications.jsonl",
        teacher_authority_verifier=teacher_authority_verifier,
        curriculum_authority_store=authority_store,
        curriculum_signing_keyring=keyring,
    )
    if teacher_authority_verifier is not None:
        _ensure_sealed_demo_curriculum(snapshot)
    return snapshot


_AUTHORITY_NOW = datetime(2026, 8, 12, 4, 0, tzinfo=timezone.utc)
_AUTHORITY_KEY = b"t" * 32
_AUTHORITY_SCOPE = "scope_" + "a" * 48


def _authority_verifier(tmp_path: Path) -> TeacherAuthorityVerifier:
    directory = tmp_path / "authority"
    directory.mkdir(mode=0o700, exist_ok=True)
    return TeacherAuthorityVerifier(
        key=_AUTHORITY_KEY,
        scope_id=_AUTHORITY_SCOPE,
        scope_key_version="k1",
        replay_store_path=directory / "replay.jsonl",
        clock=lambda: _AUTHORITY_NOW,
    )


def _authorized_body(body: dict, *, path: str, nonce: str) -> dict:
    idempotency_key = body.get("adjudication_idempotency_key") or body.get(
        "curriculum_authority_idempotency_key"
    )
    assert isinstance(idempotency_key, str)
    envelope = {
        "schema": TEACHER_AUTHORITY_SCHEMA,
        "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
        "assurance": AUTHORITY_ASSURANCE,
        "scope_id": _AUTHORITY_SCOPE,
        "scope_key_version": "k1",
        "actor_principal_sha256": "1" * 64,
        "roles_sha256": "2" * 64,
        "role_policy_sha256": "3" * 64,
        "method": "POST",
        "path": path,
        "body_sha256": canonical_sha256(body),
        "idempotency_key_sha256": sha256(
            idempotency_key.encode("utf-8")
        ).hexdigest(),
        "issued_at": _AUTHORITY_NOW.isoformat(timespec="seconds").replace(
            "+00:00", "Z"
        ),
        "expires_at": (_AUTHORITY_NOW + timedelta(minutes=2))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "nonce": "tan_" + nonce,
    }
    envelope["authority_id"] = "tauth_" + canonical_sha256(envelope)[:24]
    envelope["signature"] = (
        base64.urlsafe_b64encode(
            hmac.new(_AUTHORITY_KEY, canonical_bytes(envelope), sha256).digest()
        )
        .decode("ascii")
        .rstrip("=")
    )
    return {**body, "_teacher_authority": envelope}


def _demo_syllabus(snapshot) -> dict:
    goal = snapshot.demo_input["goal"]
    return _seal_generated_syllabus(
        {
            "title": goal["concept"],
            "description": goal["objective"],
            "audience": "裁决集成测试学习者",
            "estimated_duration_minutes": 45,
            "learning_objectives": [goal["objective"]],
            "prerequisites": [],
            "modules": [
                {
                    "title": goal["concept"],
                    "description": goal["objective"],
                    "lessons": [
                        {
                            "title": goal["concept"],
                            "objective": goal["objective"],
                            "summary": goal["objective"],
                            "duration_minutes": 45,
                            "knowledge_components": list(
                                goal["knowledge_components"]
                            ),
                            "materials": deepcopy(goal["materials"]),
                        }
                    ],
                }
            ],
        },
        model="fixture-teacher",
        source_resource_ids=["source_demo_teacher"],
        created_at="2026-08-12T04:00:00Z",
    )


def _demo_curriculum_spec(snapshot) -> dict:
    goal = snapshot.demo_input["goal"]
    labels = list(goal["knowledge_components"])
    return build_teacher_owned_curriculum_spec(
        title=goal["concept"],
        lessons=[
            {
                "legacy_lesson_id": "lesson_01_01",
                "title": goal["concept"],
                "objective": goal["objective"],
                "knowledge_components": [
                    {
                        "label": label,
                        "prerequisites": [],
                        "source_resource_ids": ["source_demo_teacher"],
                    }
                    for label in labels
                ],
            }
        ],
        source_spans=[
            {
                "resource_id": "source_demo_teacher",
                "content_sha256": sha256(b"demo teacher source").hexdigest(),
                "excerpt_sha256": sha256(b"demo teacher excerpt").hexdigest(),
                "locator": {"kind": "line", "start": 1, "end": 1},
            }
        ],
        factual_claims=[
            {
                "statement": claim["statement"],
                "knowledge_components": list(claim["knowledge_components"]),
                "source_resource_ids": ["source_demo_teacher"],
            }
            for claim in goal["knowledge_spec"]["canonical_claims"]
        ],
    )


def _ensure_sealed_demo_curriculum(snapshot) -> None:
    syllabus = _demo_syllabus(snapshot)
    family = snapshot.import_syllabus({"syllabus": syllabus})["version_family"]
    assert snapshot.curriculum_authority_store is not None
    if snapshot.curriculum_authority_store.read_family(family["family_id"]) is not None:
        return
    review_body = {
        "syllabus_id": syllabus["syllabus_id"],
        "teacher_spec": _demo_curriculum_spec(snapshot),
        "expected_syllabus_version": family["version"],
        "expected_authority_version": 0,
        "curriculum_authority_idempotency_key": "adjudication-curriculum-review-0001",
    }
    reviewed = snapshot.review_curriculum(
        _authorized_body(
            review_body,
            path="api/curriculum/review",
            nonce="a" * 48,
        )
    )
    seal_body = {
        "syllabus_id": syllabus["syllabus_id"],
        "review_id": reviewed["curriculum_authority"]["review"]["review_id"],
        "teacher_confirmed_authority": True,
        "expected_syllabus_version": family["version"],
        "expected_authority_version": 1,
        "curriculum_authority_idempotency_key": "adjudication-curriculum-seal-0001",
    }
    snapshot.seal_curriculum(
        _authorized_body(
            seal_body,
            path="api/curriculum/seal",
            nonce="b" * 48,
        )
    )


def _sealed_lesson_payload(snapshot) -> dict:
    family = snapshot.syllabus_version_store.list_families()[0]
    published = next(
        row
        for row in family["revisions"]
        if row["revision_id"] == family["published_revision_id"]
    )
    return snapshot.syllabus_lesson_payload(
        published["syllabus_id"], "lesson_01_01"
    )


def _install_authoritative_evidence(snapshot, *, suffix: str = "one"):
    lesson_payload = (
        _sealed_lesson_payload(snapshot)
        if snapshot.curriculum_authority_store is not None
        else None
    )
    started = snapshot.start(
        {
            "start_idempotency_key": f"start-adjudication-{suffix}",
            "goal": deepcopy(
                lesson_payload["goal"]
                if lesson_payload is not None
                else snapshot.demo_input["goal"]
            ),
            **(
                {"syllabus_ref": deepcopy(lesson_payload["syllabus_ref"])}
                if lesson_payload is not None
                else {}
            ),
            "student_profile": deepcopy(snapshot.demo_input["student_profile"]),
        }
    )
    session_id = started["session_id"]
    record = snapshot.sessions[session_id]
    initial_model = initialize_student_model(
        record.session["student_profile"]["initial_mastery"],
        goal=record.session["goal"],
    )
    kc_id = stable_knowledge_component_id("状态定义")
    action = deepcopy(record.session["current_action"])
    action_id = str(action["action_id"])
    evidence_id = f"evidence.dashboard.{suffix}"
    criterion = next(
        row
        for row in record.session["goal"]["knowledge_spec"]["rubric_criteria"]
        if row.get("knowledge_component") == "状态定义"
    )
    updated_model = update_student_model(
        initial_model,
        signal="misconception",
        confidence=0.9,
        focus_dimension="conceptual",
        knowledge_component_ids=[kc_id],
        answer_alignment="contradicted",
        needs_human_review=False,
        assessment_eligible=True,
        authoritative=True,
        round_number=1,
        evidence_id=evidence_id,
        item_id=action_id,
        question_id=action_id,
        rubric_id=f"teacher_rubric:{criterion['criterion_id']}",
        difficulty=0.5,
        discrimination=0.8,
        observed_at="2000-01-01T00:00:01Z",
        time_basis="session_logical",
        source="validated_teacher_rubric",
    )
    advanced = advance_teacher_agent_session(
        record.session,
        learner_response="这是不应出现在公开复核 API 的学生原文",
        signal="misconception",
        signal_confidence=0.9,
    )
    advanced["history"][-1]["student_state_before"]["student_model"] = deepcopy(
        initial_model
    )
    advanced["history"][-1]["student_state_after_observation"]["student_model"] = (
        deepcopy(updated_model)
    )
    advanced["student_state"]["student_model"] = deepcopy(updated_model)
    _refresh_integrity(advanced)
    record.session = advanced
    snapshot._persist_events(
        [
            snapshot._checkpoint_specification(
                session_id,
                record,
                idempotency_key=None,
                request_fingerprint=None,
                reason="test_authoritative_evidence_fixture",
            )
        ]
    )
    response = snapshot._response(session_id, record)
    return session_id, record, response, kc_id, evidence_id


def _guard(response: dict) -> dict:
    return {
        "session_id": response["session_id"],
        "expected_round": response["rounds_completed"],
        "expected_question_id": response["expected_question_id"],
        "expected_context_version": response["context_version"],
        "profile_revision": response["profile_summary"]["profile_revision"],
    }


def _guard_ref(response: dict) -> dict:
    return {
        "session_id": response["session_id"],
        "expected_round": response["expected_round"],
        "expected_question_id": response["expected_question_id"],
        "expected_context_version": response["expected_context_version"],
        "profile_revision": response["profile_revision"],
    }


def _enqueue(snapshot, response: dict, kc_id: str, *, suffix: str = "one") -> dict:
    return snapshot.enqueue_adjudication_review(
        {
            **_guard(response),
            "history_round": 1,
            "knowledge_component_id": kc_id,
            "review_reason": "learner_dispute",
            "adjudication_idempotency_key": f"enqueue-dashboard-{suffix}",
        }
    )["item"]


def _claim(snapshot, response: dict, item: dict, *, suffix: str = "one") -> dict:
    return snapshot.claim_adjudication_review(
        {
            **_guard(response),
            "item_id": item["item_id"],
            "expected_version": item["version"],
            "adjudication_idempotency_key": f"claim-dashboard-{suffix}",
        }
    )


def test_abstain_supersedes_exact_kc_persists_and_never_exposes_text(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    session_id, record, response, kc_id, evidence_id = _install_authoritative_evidence(
        snapshot
    )
    item = _enqueue(snapshot, response, kc_id)
    candidates = snapshot.list_adjudication_candidates(_guard(response))
    assert candidates["candidates"] == [
        {
            "history_round": 1,
            "knowledge_component_id": kc_id,
            "knowledge_component_label": "状态定义",
            "question_id": record.session["history"][0]["action"]["action_id"],
            "rubric_id": "teacher_rubric:criterion_state",
            "evidence_locator_sha256": candidates["candidates"][0][
                "evidence_locator_sha256"
            ],
        }
    ]
    assert "学生原文" not in json.dumps(candidates, ensure_ascii=False)
    assert "学生原文" not in json.dumps(item, ensure_ascii=False)
    assert "learner_text" not in json.dumps(item, ensure_ascii=False)
    before_model = deepcopy(record.session["student_state"]["student_model"])
    other_before = {
        candidate_id: deepcopy(component)
        for candidate_id, component in before_model["knowledge_components"].items()
        if candidate_id != kc_id
    }
    claim = _claim(snapshot, response, item)
    result = snapshot.decide_adjudication_review(
        {
            **_guard(response),
            "item_id": item["item_id"],
            "expected_version": claim["item"]["version"],
            "claim_token": claim["claim_token"],
            "decision": "abstain",
            "reason_code": "insufficient_evidence",
            "adjudication_idempotency_key": "decide-dashboard-abstain-one",
        }
    )
    assert result["model_effect"]["status"] == "applied_supersede_replay"
    assert (
        result["session_ref"]["expected_context_version"]
        == response["context_version"] + 1
    )
    public_decision = json.dumps(result, ensure_ascii=False)
    assert "学生原文" not in public_decision
    assert "learner_text" not in public_decision
    assert "model" not in result["model_effect"]
    assert "session" not in result
    revised_model = record.session["student_state"]["student_model"]
    source = next(
        row
        for row in revised_model["knowledge_components"][kc_id]["evidence_ledger"]
        if row["evidence_id"] == evidence_id
    )
    assert source["lifecycle_status"] == "superseded"
    for other_id, component in other_before.items():
        assert revised_model["knowledge_components"][other_id] == component

    replay = snapshot.decide_adjudication_review(
        {
            **_guard_ref(result["session_ref"]),
            "item_id": item["item_id"],
            "expected_version": claim["item"]["version"],
            "claim_token": claim["claim_token"],
            "decision": "abstain",
            "reason_code": "insufficient_evidence",
            "adjudication_idempotency_key": "decide-dashboard-abstain-one",
        }
    )
    assert replay["model_effect"]["status"] == "already_applied_no_change"
    assert replay["session_ref"] == result["session_ref"]

    restarted = _snapshot(tmp_path)
    resumed = restarted.resume({"session_id": session_id})
    listed = restarted.list_adjudication_reviews({**_guard(resumed)})
    assert listed["items"][0]["status"] == "decided"
    assert (
        restarted.sessions[session_id].session["student_state"]["student_model"]
        == revised_model
    )


@pytest.mark.parametrize(
    ("decision", "reason", "correction", "expected_status"),
    [
        ("approve", "assessment_confirmed", None, "approved_no_change"),
        (
            "correct",
            "signal_misclassified",
            {"signal": "partial"},
            "pending_authority_revalidation",
        ),
    ],
)
def test_unauthenticated_approve_and_correct_never_mutate_model_or_context(
    tmp_path: Path,
    decision: str,
    reason: str,
    correction: dict | None,
    expected_status: str,
) -> None:
    snapshot = _snapshot(tmp_path)
    _session_id, record, response, kc_id, _evidence_id = (
        _install_authoritative_evidence(snapshot, suffix=decision)
    )
    before_session = deepcopy(record.session)
    item = _enqueue(snapshot, response, kc_id, suffix=decision)
    claim = _claim(snapshot, response, item, suffix=decision)
    result = snapshot.decide_adjudication_review(
        {
            **_guard(response),
            "item_id": item["item_id"],
            "expected_version": claim["item"]["version"],
            "claim_token": claim["claim_token"],
            "decision": decision,
            "reason_code": reason,
            **({"correction": correction} if correction else {}),
            "adjudication_idempotency_key": f"decide-dashboard-{decision}",
        }
    )
    assert result["model_effect"]["status"] == expected_status
    assert record.session == before_session
    assert (
        result["session_ref"]["expected_context_version"] == response["context_version"]
    )
    assert result["authenticated_teacher"] is False
    if decision == "correct":
        assert result["model_effect"]["pending"]["model_mutated"] is False
        assert result["model_effect"]["pending"]["mastery_update_authorized"] is False


def test_authenticated_teacher_correct_revalidates_rubric_replays_and_restarts(
    tmp_path: Path,
) -> None:
    verifier = _authority_verifier(tmp_path)
    snapshot = _snapshot(tmp_path, teacher_authority_verifier=verifier)
    session_id, record, response, kc_id, evidence_id = _install_authoritative_evidence(
        snapshot, suffix="authenticated-correct"
    )
    item = _enqueue(snapshot, response, kc_id, suffix="authenticated-correct")
    claim_body = {
        **_guard(response),
        "item_id": item["item_id"],
        "expected_version": item["version"],
        "adjudication_idempotency_key": "claim-authenticated-correct",
    }
    claim = snapshot.claim_adjudication_review(
        _authorized_body(
            claim_body,
            path="api/adjudication/claim",
            nonce="1" * 48,
        )
    )
    assert claim["authenticated_teacher"] is True
    assert claim["operator_identity"] == AUTHENTICATED_TEACHER_ACTOR
    lesson_state = _new_lesson_state(record.session["goal"])
    lesson_state["lesson_phase"] = "transfer"
    lesson_state["last_transition"]["to"] = "transfer"
    record.session["lesson_state"] = lesson_state
    _refresh_integrity(record.session)
    model_before = deepcopy(record.session["student_state"]["student_model"])
    other_before = {
        other_id: deepcopy(component)
        for other_id, component in model_before["knowledge_components"].items()
        if other_id != kc_id
    }
    phase_before = record.session["lesson_state"]["lesson_phase"]
    decide_body = {
        **_guard(response),
        "item_id": item["item_id"],
        "expected_version": claim["item"]["version"],
        "claim_token": claim["claim_token"],
        "decision": "correct",
        "reason_code": "signal_misclassified",
        "correction": {
            "signal": "partial",
            "answer_alignment": "partially_aligned",
            "focus_dimension": "conceptual",
            "target_kc_ids": [kc_id],
        },
        "adjudication_idempotency_key": "decide-authenticated-correct",
    }
    decided = snapshot.decide_adjudication_review(
        _authorized_body(
            decide_body,
            path="api/adjudication/decide",
            nonce="2" * 48,
        )
    )
    assert decided["authenticated_teacher"] is True
    assert decided["correct_requires_authority_revalidation"] is False
    assert decided["model_effect"]["status"] == "applied_supersede_replay"
    revision = decided["model_effect"]["revision_receipt"]
    assert revision["decision_kind"] == "correct"
    assert "authority_revalidation_receipt_sha256" in revision
    revised = record.session["student_state"]["student_model"]
    for other_id, component in other_before.items():
        assert revised["knowledge_components"][other_id] == component
    target = revised["knowledge_components"][kc_id]
    source = next(
        row for row in target["evidence_ledger"] if row["evidence_id"] == evidence_id
    )
    assert source["lifecycle_status"] == "superseded"
    corrected = next(
        row
        for row in target["evidence_ledger"]
        if row["source"] == "authenticated_teacher_adjudication_correction"
    )
    assert corrected["signal"] == "partial"
    phase_order = [
        "orientation",
        "explanation",
        "worked_example",
        "guided_practice",
        "verification",
        "transfer",
    ]
    phase_after = record.session["lesson_state"]["lesson_phase"]
    assert phase_order.index(phase_after) <= phase_order.index(phase_before)

    # A new envelope/nonce after a lost response returns the sealed result and
    # does not append another student-model revision.
    retry_body = {**decide_body, **_guard_ref(decided["session_ref"])}
    retried = snapshot.decide_adjudication_review(
        _authorized_body(
            retry_body,
            path="api/adjudication/decide",
            nonce="3" * 48,
        )
    )
    assert retried["model_effect"]["status"] == "already_applied_no_change"
    assert (
        len(record.session["student_state"]["student_model"]["adjudication_revisions"])
        == 1
    )

    restarted = _snapshot(
        tmp_path, teacher_authority_verifier=_authority_verifier(tmp_path)
    )
    resumed = restarted.resume({"session_id": session_id})
    listed = restarted.list_adjudication_reviews(_guard(resumed))
    actor = listed["items"][0]["decision"]["actor"]
    assert actor["identity"] == AUTHENTICATED_TEACHER_ACTOR
    serialized = json.dumps(listed, ensure_ascii=False)
    assert "subject" not in serialized and "tenant" not in serialized


def test_forged_evidence_fields_active_turn_and_concurrent_claim_are_fenced(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    _session_id, record, response, kc_id, _evidence_id = (
        _install_authoritative_evidence(snapshot, suffix="fences")
    )
    with pytest.raises(TeacherAgentDashboardError, match="server-authoritative"):
        snapshot.enqueue_adjudication_review(
            {
                **_guard(response),
                "history_round": 1,
                "knowledge_component_id": kc_id,
                "review_reason": "learner_dispute",
                "adjudication_idempotency_key": "forged-dashboard-enqueue",
                "evidence": {"learner_text": "forged"},
            }
        )
    with pytest.raises(TeacherAgentDashboardError, match="unsupported fields"):
        snapshot.enqueue_adjudication_review(
            {
                **_guard(response),
                "history_round": 1,
                "knowledge_component_id": kc_id,
                "review_reason": "learner_dispute",
                "adjudication_idempotency_key": "unknown-dashboard-enqueue",
                "private_note": "raw learner material has no API field",
            }
        )
    record.active_turn_id = "turn_active"
    with pytest.raises(AdjudicationConflictError, match="active teaching turn"):
        _enqueue(snapshot, response, kc_id, suffix="active")
    record.active_turn_id = None
    item = _enqueue(snapshot, response, kc_id, suffix="fences")

    barrier = threading.Barrier(3)
    outcomes: list[object] = []
    lock = threading.Lock()

    def worker(number: int) -> None:
        barrier.wait()
        try:
            value = snapshot.claim_adjudication_review(
                {
                    **_guard(response),
                    "item_id": item["item_id"],
                    "expected_version": item["version"],
                    "adjudication_idempotency_key": f"claim-race-dashboard-{number}",
                }
            )
        except Exception as exc:  # noqa: BLE001 - intentional race capture.
            value = exc
        with lock:
            outcomes.append(value)

    threads = [threading.Thread(target=worker, args=(number,)) for number in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=3)
        assert not thread.is_alive()
    assert sum(isinstance(value, dict) for value in outcomes) == 1
    assert sum(isinstance(value, AdjudicationConflictError) for value in outcomes) == 1


def test_history_action_and_student_model_evidence_must_share_provenance(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    session_id, record, _response, kc_id, _evidence_id = (
        _install_authoritative_evidence(snapshot, suffix="provenance")
    )
    record.session["history"][0]["action"]["action_id"] = "turn_999"
    _refresh_integrity(record.session)
    snapshot._persist_events(
        [
            snapshot._checkpoint_specification(
                session_id,
                record,
                idempotency_key=None,
                request_fingerprint=None,
                reason="test_mismatched_adjudication_provenance_fixture",
            )
        ]
    )
    response = snapshot._response(session_id, record)

    with pytest.raises(AdjudicationEvidenceError, match="provenance"):
        _enqueue(snapshot, response, kc_id, suffix="provenance")
    assert snapshot._adjudication_queue().list_items() == ()


def test_abstain_with_truncated_ledger_is_rejected_before_terminal_decision(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    session_id, record, _response, kc_id, _evidence_id = (
        _install_authoritative_evidence(snapshot, suffix="truncated")
    )
    truncated_model = deepcopy(record.session["student_state"]["student_model"])
    ledger_state = truncated_model["knowledge_components"][kc_id][
        "evidence_ledger_state"
    ]
    ledger_state["total_recorded_evidence"] += 1
    ledger_state["complete_from_initial_prior"] = False
    record.session["student_state"]["student_model"] = deepcopy(truncated_model)
    record.session["history"][0]["student_state_after_observation"]["student_model"] = (
        deepcopy(truncated_model)
    )
    _refresh_integrity(record.session)
    snapshot._persist_events(
        [
            snapshot._checkpoint_specification(
                session_id,
                record,
                idempotency_key=None,
                request_fingerprint=None,
                reason="test_truncated_adjudication_ledger_fixture",
            )
        ]
    )
    response = snapshot._response(session_id, record)
    item = _enqueue(snapshot, response, kc_id, suffix="truncated")
    claim = _claim(snapshot, response, item, suffix="truncated")
    before_session = deepcopy(record.session)

    with pytest.raises(
        AdjudicationConflictError,
        match="cannot be committed without exact safe replay",
    ):
        snapshot.decide_adjudication_review(
            {
                **_guard(response),
                "item_id": item["item_id"],
                "expected_version": claim["item"]["version"],
                "claim_token": claim["claim_token"],
                "decision": "abstain",
                "reason_code": "insufficient_evidence",
                "adjudication_idempotency_key": "decide-dashboard-truncated",
            }
        )

    assert snapshot._adjudication_queue().get(item["item_id"])["status"] == "claimed"
    assert record.session == before_session
    restarted = _snapshot(tmp_path)
    assert restarted._adjudication_queue().get(item["item_id"])["status"] == "claimed"
    assert restarted.sessions[session_id].session == before_session


def test_project_export_and_purge_cover_adjudication_history_and_tombstone(
    tmp_path: Path,
) -> None:
    snapshot = _snapshot(tmp_path)
    session_id, _record, response, kc_id, evidence_id = _install_authoritative_evidence(
        snapshot, suffix="rights"
    )
    item = _enqueue(snapshot, response, kc_id, suffix="rights")
    project = snapshot.create_project({"title": "裁决数据权利"})["project"]
    snapshot.add_project_reference(
        project["project_id"],
        {"kind": "teaching_session", "reference_id": session_id},
    )

    archive = snapshot.export_project(project["project_id"])
    with zipfile.ZipFile(io.BytesIO(archive.payload)) as exported:
        name = f"adjudications/{item['item_id']}.json"
        assert name in exported.namelist()
        material = exported.read(name).decode("utf-8")
        assert item["item_id"] in material
        assert "学生原文" not in material

    trashed = snapshot.trash_project(project["project_id"], {})
    receipt = snapshot.purge_project(
        project["project_id"],
        {
            "recovery_token": trashed["recovery_token"],
            "confirmation": deletion_confirmation(project["project_id"]),
        },
    )["deletion_receipt"]
    assert receipt["deleted_counts"]["adjudications"] == 1
    assert receipt["deleted_counts"]["adjudication_tombstone_events"] == 1
    assert evidence_id not in json.dumps(receipt)
    queue = snapshot._adjudication_queue()
    assert queue.get(item["item_id"])["status"] == "cancelled"
    recovery = queue.store.recover()
    assert recovery.erasure_tombstones
    restarted = _snapshot(tmp_path)
    assert restarted._adjudication_queue().get(item["item_id"])["status"] == "cancelled"

    passphrase = tmp_path / "backup.passphrase"
    passphrase.write_text("adjudication backup secret", encoding="utf-8")
    passphrase.chmod(0o600)
    backup = tmp_path / "backup.tsm"
    with redirect_stdout(io.StringIO()):
        assert (
            cli_main(
                [
                    "teacher-agent-backup",
                    "--output",
                    str(backup),
                    "--passphrase-file",
                    str(passphrase),
                    "--session-store",
                    str(tmp_path / "sessions.jsonl"),
                    "--syllabus-store",
                    str(tmp_path / "syllabi"),
                    "--project-store",
                    str(tmp_path / "projects"),
                    "--resource-index-store",
                    str(tmp_path / "resources"),
                    "--adjudication-store",
                    str(tmp_path / "adjudications.jsonl"),
                ]
            )
            == 0
        )
        assert (
            cli_main(
                [
                    "teacher-agent-restore-drill",
                    "--backup",
                    str(backup),
                    "--passphrase-file",
                    str(passphrase),
                    "--destination",
                    str(tmp_path / "restore-drill"),
                ]
            )
            == 0
        )
    restored = (
        tmp_path
        / "restore-drill"
        / "restored"
        / "stores"
        / "assessment_adjudication"
        / "adjudications.jsonl"
    )
    assert restored.read_bytes() == (tmp_path / "adjudications.jsonl").read_bytes()


def test_crash_after_queue_decision_replays_session_revision_exactly_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = _snapshot(tmp_path)
    session_id, record, response, kc_id, evidence_id = _install_authoritative_evidence(
        snapshot, suffix="crash"
    )
    item = _enqueue(snapshot, response, kc_id, suffix="crash")
    claim = _claim(snapshot, response, item, suffix="crash")
    original_persist = TeacherAgentDashboardSnapshot._persist_events

    def crash_before_session_checkpoint(self, events):
        if self is snapshot and any(
            event.get("data", {}).get("reason")
            == "assessment_adjudication_supersede_replay"
            for event in events
        ):
            raise TeacherAgentDashboardError("synthetic adjudication checkpoint crash")
        return original_persist(self, events)

    monkeypatch.setattr(
        TeacherAgentDashboardSnapshot,
        "_persist_events",
        crash_before_session_checkpoint,
    )
    request = {
        **_guard(response),
        "item_id": item["item_id"],
        "expected_version": claim["item"]["version"],
        "claim_token": claim["claim_token"],
        "decision": "abstain",
        "reason_code": "insufficient_evidence",
        "adjudication_idempotency_key": "decide-dashboard-crash",
    }
    with pytest.raises(TeacherAgentDashboardError, match="synthetic"):
        snapshot.decide_adjudication_review(request)
    assert snapshot._adjudication_queue().get(item["item_id"])["status"] == "decided"
    source_before = next(
        row
        for row in record.session["student_state"]["student_model"][
            "knowledge_components"
        ][kc_id]["evidence_ledger"]
        if row["evidence_id"] == evidence_id
    )
    assert source_before["lifecycle_status"] == "active"
    with pytest.raises(AdjudicationConflictError, match="different adjudication input"):
        snapshot.decide_adjudication_review(
            {
                **request,
                "decision": "approve",
                "reason_code": "assessment_confirmed",
            }
        )

    monkeypatch.setattr(
        TeacherAgentDashboardSnapshot, "_persist_events", original_persist
    )
    restarted = _snapshot(tmp_path)
    applied_session = restarted.resume({"session_id": session_id})
    applied_model = restarted.sessions[session_id].session["student_state"][
        "student_model"
    ]
    assert len(applied_model["adjudication_revisions"]) == 1
    assert (
        next(
            row
            for row in applied_model["knowledge_components"][kc_id]["evidence_ledger"]
            if row["evidence_id"] == evidence_id
        )["lifecycle_status"]
        == "superseded"
    )
    retried = restarted.decide_adjudication_review(
        {**request, **_guard(applied_session)}
    )
    assert retried["model_effect"]["status"] == "already_applied_no_change"
    assert (
        len(
            restarted.sessions[session_id].session["student_state"]["student_model"][
                "adjudication_revisions"
            ]
        )
        == 1
    )
    second_restart = _snapshot(tmp_path)
    resumed = second_restart.resume({"session_id": session_id})
    assert resumed["context_version"] == applied_session["context_version"]


def test_authenticated_correction_crash_revalidates_sealed_receipt_on_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    verifier = _authority_verifier(tmp_path)
    snapshot = _snapshot(tmp_path, teacher_authority_verifier=verifier)
    session_id, record, response, kc_id, evidence_id = _install_authoritative_evidence(
        snapshot, suffix="auth-crash"
    )
    item = _enqueue(snapshot, response, kc_id, suffix="auth-crash")
    claim_body = {
        **_guard(response),
        "item_id": item["item_id"],
        "expected_version": item["version"],
        "adjudication_idempotency_key": "claim-auth-crash",
    }
    claim = snapshot.claim_adjudication_review(
        _authorized_body(claim_body, path="api/adjudication/claim", nonce="7" * 48)
    )
    lesson_state = _new_lesson_state(record.session["goal"])
    lesson_state["lesson_phase"] = "transfer"
    lesson_state["last_transition"]["to"] = "transfer"
    record.session["lesson_state"] = lesson_state
    _refresh_integrity(record.session)
    snapshot._persist_events(
        [
            snapshot._checkpoint_specification(
                session_id,
                record,
                idempotency_key=None,
                request_fingerprint=None,
                reason="test_authenticated_crash_phase_fixture",
            )
        ]
    )
    original_persist = TeacherAgentDashboardSnapshot._persist_events

    def crash_before_session_checkpoint(self, events):
        if self is snapshot and any(
            event.get("data", {}).get("reason")
            == "assessment_adjudication_supersede_replay"
            for event in events
        ):
            raise TeacherAgentDashboardError("synthetic authenticated checkpoint crash")
        return original_persist(self, events)

    monkeypatch.setattr(
        TeacherAgentDashboardSnapshot,
        "_persist_events",
        crash_before_session_checkpoint,
    )
    decide_body = {
        **_guard(response),
        "item_id": item["item_id"],
        "expected_version": claim["item"]["version"],
        "claim_token": claim["claim_token"],
        "decision": "correct",
        "reason_code": "signal_misclassified",
        "correction": {
            "signal": "partial",
            "answer_alignment": "partially_aligned",
            "focus_dimension": "conceptual",
            "target_kc_ids": [kc_id],
        },
        "adjudication_idempotency_key": "decide-auth-crash",
    }
    with pytest.raises(TeacherAgentDashboardError, match="synthetic authenticated"):
        snapshot.decide_adjudication_review(
            _authorized_body(
                decide_body, path="api/adjudication/decide", nonce="8" * 48
            )
        )
    assert snapshot._adjudication_queue().get(item["item_id"])["status"] == "decided"
    assert (
        next(
            row
            for row in record.session["student_state"]["student_model"][
                "knowledge_components"
            ][kc_id]["evidence_ledger"]
            if row["evidence_id"] == evidence_id
        )["lifecycle_status"]
        == "active"
    )

    monkeypatch.setattr(
        TeacherAgentDashboardSnapshot, "_persist_events", original_persist
    )
    restarted = _snapshot(
        tmp_path, teacher_authority_verifier=_authority_verifier(tmp_path)
    )
    recovered = restarted.resume({"session_id": session_id})
    model = restarted.sessions[session_id].session["student_state"]["student_model"]
    assert len(model["adjudication_revisions"]) == 1
    assert (
        next(
            row
            for row in model["knowledge_components"][kc_id]["evidence_ledger"]
            if row["evidence_id"] == evidence_id
        )["lifecycle_status"]
        == "superseded"
    )
    assert (
        restarted.sessions[session_id].session["lesson_state"]["lesson_phase"]
        == "guided_practice"
    )
    retry = restarted.decide_adjudication_review(
        _authorized_body(
            {**decide_body, **_guard(recovered)},
            path="api/adjudication/decide",
            nonce="9" * 48,
        )
    )
    assert retry["model_effect"]["status"] == "already_applied_no_change"
    assert len(model["adjudication_revisions"]) == 1
