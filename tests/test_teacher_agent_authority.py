from __future__ import annotations

import base64
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from hashlib import sha256
import hmac
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from teaching_skill_miner.teacher_agent_authority import (
    AUTHENTICATED_TEACHER_ACTOR,
    AUTHORITY_ASSURANCE,
    TEACHER_AUTHORITY_SCHEMA,
    TeacherAuthorityError,
    TeacherAuthorityReplayError,
    TeacherAuthorityVerifier,
    authenticated_teacher_actor,
    canonical_bytes,
    canonical_sha256,
)
from teaching_skill_miner.teacher_agent_adjudication import (
    TeacherAgentAdjudicationError,
    authoritative_evidence_sha256,
    authenticated_instruction_authority_basis,
    authorize_authenticated_instruction,
    canonical_sha256 as adjudication_sha256,
)
from teaching_skill_miner.teacher_agent_adjudication_store import (
    DurableTeacherAgentAdjudicationQueue,
)
from teaching_skill_miner.student_model import (
    apply_student_model_adjudication,
    initialize_student_model,
    update_student_model,
)


NOW = datetime(2026, 8, 12, 4, 0, tzinfo=timezone.utc)
SCOPE = "scope_" + "a" * 48
KEY = b"k" * 32


def _signature(key: bytes, value: dict) -> str:
    return (
        base64.urlsafe_b64encode(hmac.new(key, canonical_bytes(value), sha256).digest())
        .decode("ascii")
        .rstrip("=")
    )


def _body(*, key: str = "authority-idempotency-001") -> dict:
    return {
        "item_id": "adj_" + "1" * 24,
        "expected_version": 2,
        "adjudication_idempotency_key": key,
    }


def _signed_request(
    body: dict,
    *,
    key: bytes = KEY,
    scope: str = SCOPE,
    key_version: str = "k1",
    path: str = "api/adjudication/claim",
    issued_at: datetime = NOW,
    nonce_hex: str = "b" * 48,
) -> dict:
    idempotency_key = body.get(
        "resource_review_idempotency_key",
        body.get("adjudication_idempotency_key"),
    )
    assert isinstance(idempotency_key, str)
    envelope = {
        "schema": TEACHER_AUTHORITY_SCHEMA,
        "authority_kind": AUTHENTICATED_TEACHER_ACTOR,
        "assurance": AUTHORITY_ASSURANCE,
        "scope_id": scope,
        "scope_key_version": key_version,
        "actor_principal_sha256": "1" * 64,
        "roles_sha256": "2" * 64,
        "role_policy_sha256": "3" * 64,
        "method": "POST",
        "path": path,
        "body_sha256": canonical_sha256(body),
        "idempotency_key_sha256": sha256(idempotency_key.encode("utf-8")).hexdigest(),
        "issued_at": issued_at.isoformat(timespec="seconds").replace("+00:00", "Z"),
        "expires_at": (issued_at + timedelta(minutes=2))
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "nonce": "tan_" + nonce_hex,
    }
    envelope["authority_id"] = "tauth_" + canonical_sha256(envelope)[:24]
    envelope["signature"] = _signature(key, envelope)
    return {**deepcopy(body), "_teacher_authority": envelope}


def test_resource_review_uses_exact_operation_idempotency_field(
    tmp_path: Path,
) -> None:
    body = {
        "resource_id": "res_" + "1" * 20,
        "resource_review_idempotency_key": "resource-review-key-001",
    }
    request = _signed_request(body, path="api/resource/review")
    receipt = _verifier(tmp_path).verify(
        request, method="POST", path="api/resource/review"
    )
    assert receipt["path"] == "api/resource/review"
    assert (
        receipt["idempotency_key_sha256"]
        == sha256(body["resource_review_idempotency_key"].encode("utf-8")).hexdigest()
    )

    wrong_field = {
        "resource_id": body["resource_id"],
        "adjudication_idempotency_key": "resource-review-key-002",
    }
    with pytest.raises(TeacherAuthorityError, match="idempotency binding"):
        _verifier(tmp_path).verify(
            _signed_request(
                wrong_field, path="api/resource/review", nonce_hex="c" * 48
            ),
            method="POST",
            path="api/resource/review",
        )


@pytest.mark.parametrize(
    "key",
    ["short", " leading-key", "trailing-key ", "bad/key-value", "x" * 161],
)
def test_all_teacher_authority_idempotency_keys_use_the_strict_pattern(
    tmp_path: Path, key: str
) -> None:
    body = _body(key=key)
    with pytest.raises(TeacherAuthorityError, match="idempotency binding"):
        _verifier(tmp_path).verify(
            _signed_request(body, nonce_hex=sha256(key.encode()).hexdigest()[:48]),
            method="POST",
            path="api/adjudication/claim",
        )


def _verifier(tmp_path: Path, **overrides) -> TeacherAuthorityVerifier:
    return TeacherAuthorityVerifier(
        key=overrides.get("key", KEY),
        scope_id=overrides.get("scope_id", SCOPE),
        scope_key_version=overrides.get("scope_key_version", "k1"),
        replay_store_path=tmp_path / "authority-replay.jsonl",
        legacy_scope_bindings=overrides.get("legacy_scope_bindings", ()),
        clock=overrides.get("clock", lambda: NOW),
    )


def test_scope_authority_consumes_nonce_and_survives_restart(tmp_path: Path) -> None:
    request = _signed_request(_body())
    receipt = _verifier(tmp_path).verify(
        request, method="POST", path="api/adjudication/claim"
    )
    assert receipt["service_authorization_signature_verified"] is True
    assert receipt["personal_non_repudiation"] is False
    assert (
        "subject" not in receipt and "tenant" not in receipt and "roles" not in receipt
    )
    actor = authenticated_teacher_actor(receipt)
    assert actor == {
        "identity": AUTHENTICATED_TEACHER_ACTOR,
        "authenticated": True,
        "teacher_identity_claimed": True,
        "principal_sha256": "1" * 64,
        "roles_sha256": "2" * 64,
        "authority_id": request["_teacher_authority"]["authority_id"],
        "assurance": AUTHORITY_ASSURANCE,
    }
    with pytest.raises(TeacherAuthorityReplayError, match="replayed"):
        _verifier(tmp_path).verify(
            request, method="POST", path="api/adjudication/claim"
        )


def test_rotated_verifier_accepts_only_legacy_receipts_not_legacy_requests(
    tmp_path: Path,
) -> None:
    request = _signed_request(_body())
    old_receipt = _verifier(tmp_path).verify(
        request,
        method="POST",
        path="api/adjudication/claim",
        consume=False,
    )
    rotated_scope = "scope_" + "c" * 48
    rotated = _verifier(
        tmp_path,
        scope_id=rotated_scope,
        scope_key_version="k2",
        legacy_scope_bindings=((SCOPE, "k1"),),
    )
    assert rotated.verify_verification_receipt(old_receipt) == old_receipt
    with pytest.raises(TeacherAuthorityError, match="binding"):
        rotated.verify(
            request,
            method="POST",
            path="api/adjudication/claim",
            consume=False,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.update(expected_version=3), "body changed"),
        (
            lambda value: value["_teacher_authority"].update(
                scope_id="scope_" + "c" * 48
            ),
            "binding",
        ),
        (
            lambda value: value["_teacher_authority"].update(scope_key_version="k2"),
            "binding",
        ),
        (
            lambda value: value["_teacher_authority"].update(
                path="api/adjudication/decide"
            ),
            "binding",
        ),
    ],
)
def test_payload_route_scope_and_key_mutations_fail_closed(
    tmp_path: Path, mutation, message: str
) -> None:
    request = _signed_request(_body())
    mutation(request)
    with pytest.raises(TeacherAuthorityError, match=message):
        _verifier(tmp_path).verify(
            request, method="POST", path="api/adjudication/claim"
        )


def test_expired_wrong_key_and_cross_scope_envelopes_fail(tmp_path: Path) -> None:
    expired = _signed_request(_body(), issued_at=NOW - timedelta(minutes=3))
    with pytest.raises(TeacherAuthorityError, match="expired"):
        _verifier(tmp_path).verify(
            expired, method="POST", path="api/adjudication/claim"
        )
    current = _signed_request(_body(), key=b"z" * 32)
    with pytest.raises(TeacherAuthorityError, match="signature"):
        _verifier(tmp_path).verify(
            current, method="POST", path="api/adjudication/claim"
        )
    scoped = _signed_request(_body(), scope="scope_" + "d" * 48)
    with pytest.raises(TeacherAuthorityError, match="binding"):
        _verifier(tmp_path).verify(scoped, method="POST", path="api/adjudication/claim")


def test_revalidation_receipt_binds_rubric_correction_and_instruction(
    tmp_path: Path,
) -> None:
    body = {
        **_body(key="authority-idempotency-002"),
        "evidence_id": "evidence.assessment.1",
        "evidence_sha256": "4" * 64,
    }
    verifier = _verifier(tmp_path)
    verification = verifier.verify(
        _signed_request(
            body,
            path="api/adjudication/decide",
            nonce_hex="e" * 48,
        ),
        method="POST",
        path="api/adjudication/decide",
    )
    receipt = verifier.revalidate(
        verification,
        review_item_id="adj_" + "1" * 24,
        review_item_version_sha256="5" * 64,
        evidence_id="evidence.assessment.1",
        evidence_sha256="4" * 64,
        model_evidence_sha256="4" * 64,
        instruction_authority_basis_sha256="6" * 64,
        correction={"signal": "partial"},
        target_kc_ids=["kc_state_definition"],
        rubric_id="rubric.dp.teacher.1",
        rubric_authority_sha256="7" * 64,
    )
    assert verifier.verify_revalidation(receipt) == receipt
    for field in (
        "correction_sha256",
        "instruction_authority_basis_sha256",
        "rubric_authority_sha256",
        "evidence_sha256",
    ):
        tampered = deepcopy(receipt)
        tampered[field] = "f" * 64
        with pytest.raises(TeacherAuthorityError):
            verifier.verify_revalidation(tampered)


def test_replay_store_permissions_and_symlink_fail_closed(tmp_path: Path) -> None:
    unsafe = tmp_path / "unsafe"
    unsafe.mkdir(mode=0o755)
    verifier = TeacherAuthorityVerifier(
        key=KEY,
        scope_id=SCOPE,
        scope_key_version="k1",
        replay_store_path=unsafe / "replay.jsonl",
        clock=lambda: NOW,
    )
    with pytest.raises(TeacherAuthorityError, match="directory is unsafe"):
        verifier.verify(
            _signed_request(_body()),
            method="POST",
            path="api/adjudication/claim",
        )


def _evidence() -> dict:
    value = {
        "evidence_id": "evidence.assessment.1",
        "source": {
            "session_id": "teach_session_1",
            "round_number": 3,
            "action_id": "action.assess.3",
            "question_id": "question.dp.3",
            "history_event_sha256": "8" * 64,
        },
        "target_kc_ids": ["kc_state_definition"],
        "original_assessment_id": "assessment.dp.3",
        "original_assessment_sha256": "9" * 64,
        "rubric_id": "rubric.dp.teacher.1",
        "rubric_authority_sha256": "a" * 64,
        "before_snapshot": {"version": 1},
        "after_snapshot": {"version": 2},
    }
    value["evidence_sha256"] = authoritative_evidence_sha256(value)
    return value


def _verified_authority(
    verifier: TeacherAuthorityVerifier,
    *,
    path: str,
    idempotency_key: str,
    nonce: str,
) -> dict:
    request = {"adjudication_idempotency_key": idempotency_key}
    return verifier.verify(
        _signed_request(request, path=path, nonce_hex=nonce),
        method="POST",
        path=path,
    )


def test_durable_authenticated_actor_restart_and_fresh_envelope_retry(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    registry = {evidence["evidence_id"]: evidence}
    queue_path = tmp_path / "adjudication.jsonl"
    verifier = _verifier(tmp_path)

    def queue() -> DurableTeacherAgentAdjudicationQueue:
        return DurableTeacherAgentAdjudicationQueue(
            queue_path,
            evidence_resolver=registry.get,
            clock=lambda: NOW,
            token_factory=lambda: "authenticated-claim-token-001",
        )

    pending = queue().enqueue(
        evidence_id=evidence["evidence_id"],
        evidence_sha256=evidence["evidence_sha256"],
        idempotency_key="authority-enqueue-001",
    )
    claim_key = "authority-claim-001"
    first_claim_receipt = _verified_authority(
        verifier,
        path="api/adjudication/claim",
        idempotency_key=claim_key,
        nonce="1" * 48,
    )
    claimed = queue().claim(
        pending["item_id"],
        expected_version=1,
        idempotency_key=claim_key,
        authority_receipt=first_claim_receipt,
        authority_validator=verifier.verify_verification_receipt,
    )
    # A network retry receives a newly signed nonce.  It is consumed but the
    # durable business idempotency binding returns the first sealed response.
    retry_claim_receipt = _verified_authority(
        verifier,
        path="api/adjudication/claim",
        idempotency_key=claim_key,
        nonce="2" * 48,
    )
    assert (
        queue().claim(
            pending["item_id"],
            expected_version=1,
            idempotency_key=claim_key,
            authority_receipt=retry_claim_receipt,
            authority_validator=verifier.verify_verification_receipt,
        )
        == claimed
    )

    decide_key = "authority-decide-001"
    decide_receipt = _verified_authority(
        verifier,
        path="api/adjudication/decide",
        idempotency_key=decide_key,
        nonce="3" * 48,
    )

    def authorize(instruction, item, verification):
        basis_hash = adjudication_sha256(
            authenticated_instruction_authority_basis(instruction)
        )
        receipt = verifier.revalidate(
            verification,
            review_item_id=item["item_id"],
            review_item_version_sha256=item["version_sha256"],
            evidence_id=item["original"]["evidence_id"],
            evidence_sha256=item["original"]["evidence_sha256"],
            model_evidence_sha256=item["original"]["evidence_sha256"],
            instruction_authority_basis_sha256=basis_hash,
            correction=item["decision"]["correction"],
            target_kc_ids=item["target_kc_ids"],
            rubric_id=item["original"]["rubric_id"],
            rubric_authority_sha256=item["original"]["rubric_authority_sha256"],
        )
        return authorize_authenticated_instruction(
            instruction, revalidation_receipt=receipt
        )

    arguments = {
        "expected_version": claimed["item"]["version"],
        "claim_token": claimed["claim_token"],
        "evidence_id": evidence["evidence_id"],
        "evidence_sha256": evidence["evidence_sha256"],
        "idempotency_key": decide_key,
        "decision": "correct",
        "reason_code": "signal_misclassified",
        "correction": {
            "signal": "partial",
            "answer_alignment": "partially_aligned",
            "focus_dimension": "conceptual",
            "target_kc_ids": ["kc_state_definition"],
        },
        "authority_receipt": decide_receipt,
        "authority_validator": verifier.verify_verification_receipt,
        "instruction_authorizer": authorize,
    }
    decided = queue().decide(pending["item_id"], **arguments)
    assert decided["instruction"]["mastery_update_authorized"] is True
    assert decided["instruction"]["authority"]["personal_non_repudiation"] is False
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schema"
            / "teacher_agent_adjudication.schema.json"
        ).read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(decided["item"])
    Draft202012Validator(schema).validate(decided["instruction"])
    assert queue().get(pending["item_id"])["status"] == "decided"
    journal = queue_path.read_text(encoding="utf-8")
    assert "tenant" not in journal and "subject" not in journal
    assert "authenticated_teacher_server_authorized" in journal

    model = initialize_student_model(
        {"conceptual": 0.2},
        goal={
            "concept": "动态规划",
            "knowledge_components": [
                {"kc_id": "kc_state_definition", "label": "状态定义"},
                {"kc_id": "kc_state_transition", "label": "状态转移"},
            ],
            "knowledge_spec": {
                "schema": "teaching_skill_miner.teacher_goal_knowledge_spec.v1",
                "status": "teacher_provided",
                "claim_boundary": {"authoritative_for_runtime_grading": True},
            },
        },
    )
    model = update_student_model(
        model,
        signal="correct",
        confidence=0.9,
        focus_dimension="conceptual",
        knowledge_component_ids=["kc_state_definition"],
        answer_alignment="aligned",
        assessment_eligible=True,
        authoritative=True,
        round_number=3,
        evidence_id=evidence["evidence_id"],
        item_id="item.assessment.1",
        question_id="question.dp.3",
        rubric_id=evidence["rubric_id"],
        observed_at="2026-08-12T04:00:00Z",
        source="validated_teacher_rubric",
        authority_evidence_sha256=evidence["evidence_sha256"],
    )
    model = update_student_model(
        model,
        signal="correct",
        confidence=0.8,
        focus_dimension="conceptual",
        knowledge_component_ids=["kc_state_transition"],
        answer_alignment="aligned",
        assessment_eligible=True,
        authoritative=True,
        round_number=4,
        evidence_id="evidence.other.1",
        item_id="item.other.1",
        question_id="question.other.1",
        rubric_id="rubric.other.1",
        observed_at="2026-08-12T04:01:00Z",
        source="validated_teacher_rubric",
    )
    other_before = deepcopy(model["knowledge_components"]["kc_state_transition"])
    applied = apply_student_model_adjudication(
        model,
        target_kc_id="kc_state_definition",
        source_evidence_id=evidence["evidence_id"],
        instruction=decided["instruction"],
        authority_revalidator=verifier.verify_revalidation,
    )
    assert applied["status"] == "applied_supersede_replay"
    assert applied["revision_receipt"]["decision_kind"] == "correct"
    target = applied["model"]["knowledge_components"]["kc_state_definition"]
    assert (
        sum(
            row["lifecycle_status"] == "superseded" for row in target["evidence_ledger"]
        )
        == 1
    )
    corrected = next(
        row
        for row in target["evidence_ledger"]
        if row["source"] == "authenticated_teacher_adjudication_correction"
    )
    assert corrected["signal"] == "partial"
    assert (
        corrected["authority_evidence_sha256"]
        == decided["instruction"]["authority"]["server_revalidation_receipt"][
            "receipt_sha256"
        ]
    )
    assert (
        applied["model"]["knowledge_components"]["kc_state_transition"] == other_before
    )
    replay = apply_student_model_adjudication(
        applied["model"],
        target_kc_id="kc_state_definition",
        source_evidence_id=evidence["evidence_id"],
        instruction=decided["instruction"],
        authority_revalidator=verifier.verify_revalidation,
    )
    assert replay["status"] == "already_applied_no_change"
    assert replay["model"] == applied["model"]


def test_failed_rubric_authorizer_does_not_commit_terminal_decision(
    tmp_path: Path,
) -> None:
    evidence = _evidence()
    verifier = _verifier(tmp_path)
    queue = DurableTeacherAgentAdjudicationQueue(
        tmp_path / "adjudication.jsonl",
        evidence_resolver={evidence["evidence_id"]: evidence}.get,
        clock=lambda: NOW,
        token_factory=lambda: "authenticated-claim-token-002",
    )
    pending = queue.enqueue(
        evidence_id=evidence["evidence_id"],
        evidence_sha256=evidence["evidence_sha256"],
        idempotency_key="authority-enqueue-002",
    )
    claim_key = "authority-claim-002"
    claimed = queue.claim(
        pending["item_id"],
        expected_version=1,
        idempotency_key=claim_key,
        authority_receipt=_verified_authority(
            verifier,
            path="api/adjudication/claim",
            idempotency_key=claim_key,
            nonce="4" * 48,
        ),
        authority_validator=verifier.verify_verification_receipt,
    )
    decide_key = "authority-decide-002"
    with pytest.raises(TeacherAgentAdjudicationError, match="revalidation failed"):
        queue.decide(
            pending["item_id"],
            expected_version=claimed["item"]["version"],
            claim_token=claimed["claim_token"],
            evidence_id=evidence["evidence_id"],
            evidence_sha256=evidence["evidence_sha256"],
            idempotency_key=decide_key,
            decision="correct",
            reason_code="signal_misclassified",
            correction={"signal": "partial"},
            authority_receipt=_verified_authority(
                verifier,
                path="api/adjudication/decide",
                idempotency_key=decide_key,
                nonce="5" * 48,
            ),
            authority_validator=verifier.verify_verification_receipt,
            instruction_authorizer=lambda *_args: (_ for _ in ()).throw(
                RuntimeError("rubric registry unavailable")
            ),
        )
    assert queue.get(pending["item_id"])["status"] == "claimed"
