from __future__ import annotations

import base64
from copy import deepcopy
from hashlib import sha256
import http.client
import io
import json
from pathlib import Path
import threading
from urllib.parse import urlsplit
import zipfile

import pytest

from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
    create_teacher_agent_dashboard_server,
)
from teaching_skill_miner.teacher_agent_data_rights import (
    deletion_confirmation,
    validate_project_export_archive,
)
from teaching_skill_miner.teacher_agent_multimodal import (
    VISUAL_PROVIDER_RESULT_SCHEMA,
)


ROOT = Path(__file__).resolve().parents[1]
SECRET = b"dashboard-consent-signing-secret-material-32-bytes"
PROVIDER_POLICY = {
    "policy_id": "deepseek-school-processing",
    "policy_version": "2026-08-12",
    "policy_source": "deployment_operator_asserted_external_terms_not_repository_verified",
    "processing_region": "cn_north",
    "provider_retention_days": 7,
    "deletion_status": "outside_service_control_subject_to_provider_policy",
    "documentation_url": "https://provider.example/privacy",
}
PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk"
    "+A8AAQUBAScY42YAAAAASUVORK5CYII="
)


class _Client:
    def __init__(self) -> None:
        self.calls = 0

    def public_status(self):
        return {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "base_origin": "https://api.deepseek.example",
            "configured": True,
            "web_search_supported": True,
            "thinking_mode": "disabled",
            "temperature": 0.0,
            "remote_student_data_opt_in": True,
            "api_key_exposed": False,
        }

    def chat_json(self, messages, *, request_kind, require_remote_consent=True):
        del messages
        assert require_remote_consent is True
        self.calls += 1
        if request_kind == "console_direct_chat":
            return {"message": "服务器收据已核验。"}, {
                "provider": "deepseek",
                "model": "deepseek-v4-flash",
                "usage": {},
            }
        return deepcopy(_live_plan()), {
            "provider": "deepseek",
            "model": "deepseek-v4-flash",
            "request_kind": request_kind,
            "request_sha256": "a" * 64,
            "latency_ms": 1.0,
            "attempt_count": 1,
            "http_status": 200,
            "response_id": "fake",
            "usage": {},
            "credential_logged": False,
        }


def _live_plan():
    return {
        "schema": "teaching_skill_miner.deepseek_turn_plan.v1",
        "diagnosis": {
            "signal": "not_observed",
            "confidence": 0.0,
            "diagnosis_reason": "等待学习者作答。",
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
            "supporting_skill_ids": ["skill_wait_and_elicit"],
            "selection_reason": "先确认一个最小前置概念。",
            "next_focus": "prerequisite",
        },
        "teacher_action": {
            "type": "ask_one_question",
            "message": "请先说出一个相关的前置概念。",
            "expected_signal": "学习者说出一个相关概念。",
        },
        "stop_recommendation": {"should_stop": False, "reason": ""},
    }


class _VisionExtractor:
    def __call__(self, image_bytes, mime_type, *, display_name):
        return {
            "schema": "teaching_skill_miner.local_visual_evidence.v1",
            "source_modality": "image",
            "source_kind": "learner_answer_attachment",
            "display_name": display_name,
            "mime_type": mime_type,
            "byte_size": len(image_bytes),
            "content_sha256": sha256(image_bytes).hexdigest(),
            "engine": "test-local-ocr",
            "status": "recognized",
            "recognized_text": "图中有一个状态转移箭头。",
            "confidence": 0.9,
            "confidence_semantics": "ocr_only",
            "formula_like_text_detected": False,
            "formula_accuracy_established": False,
            "extractor_fallback_used": False,
            "needs_student_confirmation": False,
            "raw_media_retained": False,
            "remote_media_sent": False,
            "remote_representation": "bounded_redacted_ocr_text_only",
        }


class _RemoteVisionProvider:
    provider_id = "visual-lab-v1"
    processing_region = "cn-north-1"
    sends_raw_media_remotely = True

    def __init__(self) -> None:
        self.calls = 0
        self.last_bytes = b""

    def analyze(self, image_bytes, mime_type, *, task_context):
        assert mime_type == "image/png"
        assert task_context
        self.calls += 1
        self.last_bytes = image_bytes
        return {
            "schema": VISUAL_PROVIDER_RESULT_SCHEMA,
            "modality": "diagram",
            "description": "图中有一个由左向右的状态转移箭头。",
            "claims": [
                {
                    "statement": "状态 A 指向状态 B。",
                    "evidence_locator": "diagram/arrow-1",
                    "confidence": 0.9,
                }
            ],
            "uncertainties": ["节点文字仍需核对"],
        }


def _snapshot(tmp_path, *, client=None, remote_vision=None, durable=False):
    return build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
        client=client,
        store_path=tmp_path / "sessions.jsonl" if durable else None,
        project_store_path=tmp_path / "projects" if durable else None,
        consent_store_path=tmp_path / "consent.json",
        consent_signing_secret=SECRET,
        vision_extractor=_VisionExtractor(),
        visual_semantic_provider=remote_vision,
        visual_provider_retention_days=0,
    )


def _grant(snapshot, purpose):
    return snapshot.grant_remote_consent(
        {
            "purpose": purpose,
            "validity_days": 30,
            "likely_minor": False,
            "guardian_or_school_policy": "not_required",
        }
    )["receipt"]


def _start_body(snapshot, consent_id=None, *, key="consent-start-1"):
    body = {
        "goal": snapshot.demo_input["goal"],
        "student_profile": snapshot.demo_input["student_profile"],
        "start_idempotency_key": key,
    }
    if consent_id is not None:
        body["remote_consent_id"] = consent_id
    return body


def _attachment_body(session, *, key, consent_id=None):
    body = {
        "session_id": session["session_id"],
        "expected_round": session["rounds_completed"],
        "expected_question_id": session["expected_question_id"],
        "expected_context_version": session["context_version"],
        "profile_revision": session["profile_summary"]["profile_revision"],
        "attachment_idempotency_key": key,
        "mime_type": "image/png",
        "display_name": "diagram.png",
        "data_base64": base64.b64encode(PNG).decode(),
        "visual_analysis_requested": True,
    }
    if consent_id is not None:
        body["visual_consent_id"] = consent_id
    return body


def test_chat_consent_is_server_owned_revocable_and_restart_durable(tmp_path):
    client = _Client()
    snapshot = _snapshot(tmp_path, client=client)
    policies = snapshot.bootstrap()["remote_consent"]["policies"]
    chat_policy = next(item for item in policies if item["purpose"] == "remote_chat")
    assert chat_policy["provider_id"] == "deepseek"

    with pytest.raises(TeacherAgentDashboardError, match="server-owned"):
        snapshot.grant_remote_consent(
            {"purpose": "remote_chat", "provider_id": "attacker"}
        )
    receipt = _grant(snapshot, "remote_chat")
    body = {
        "messages": [{"role": "user", "content": "LEARNER_PRIVATE_MARKER"}],
        "remote_consent_id": receipt["consent_id"],
    }
    assert snapshot.chat(body)["message"] == "服务器收据已核验。"
    assert client.calls == 1

    forged = {**body, "remote_processing_acknowledged": True}
    with pytest.raises(TeacherAgentDashboardError, match="legacy browser consent"):
        snapshot.chat(forged)
    assert client.calls == 1

    revoked = snapshot.revoke_remote_consent(
        {"consent_id": receipt["consent_id"], "reason_code": "user_revoked"}
    )["receipt"]
    assert revoked["status"] == "revoked"
    with pytest.raises(TeacherAgentDashboardError, match="does not authorize"):
        snapshot.chat(body)
    assert client.calls == 1

    restarted = _snapshot(tmp_path, client=client)
    listing = restarted.list_remote_consents({})
    assert listing["receipts"][0]["status"] == "revoked"
    assert "LEARNER_PRIVATE_MARKER" not in json.dumps(listing, ensure_ascii=False)


def test_authenticated_remote_policy_is_explicit_server_owned_and_stale_safe(
    tmp_path,
):
    common = {
        "client": _Client(),
        "store_path": tmp_path / "sessions.jsonl",
        "learning_record_store_path": tmp_path / "learning.jsonl",
        "learner_key_secret": b"l" * 32,
        "learner_tenant_id": "opaque-scope",
        "trusted_learner_profile_ref": "profile_" + "a" * 64,
        "consent_store_path": tmp_path / "consent.json",
        "consent_signing_secret": SECRET,
        "remote_processing_region": "cn_north",
        "remote_provider_retention_days": 7,
        "remote_provider_policy": PROVIDER_POLICY,
    }
    with pytest.raises(TeacherAgentDashboardError, match="explicit deployment"):
        build_teacher_agent_dashboard_snapshot(
            ROOT / "data/teacher_agent_skill_library.json",
            ROOT / "data/teacher_agent_demo_input.json",
            ROOT / "data/teacher_agent_evaluation_cases.json",
            **common,
        )

    denied_policy = {
        "policy_id": "organization-age-policy",
        "policy_version": "2026-fall",
        "policy_source": "organization_oidc_or_roster_policy",
        "likely_minor": False,
        "guardian_or_school_policy": "not_required",
        "remote_processing_eligible": False,
    }
    denied = build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
        remote_subject_policy=denied_policy,
        **common,
    )
    with pytest.raises(TeacherAgentDashboardError, match="does not authorize"):
        denied.grant_remote_consent(
            {
                "purpose": "remote_chat",
                "validity_days": 1,
                "likely_minor": False,
                "guardian_or_school_policy": "not_required",
            }
        )
    with pytest.raises(TeacherAgentDashboardError, match="server-owned"):
        denied.grant_remote_consent(
            {
                "purpose": "remote_chat",
                "likely_minor": True,
                "guardian_or_school_policy": "verified_guardian",
            }
        )

    allowed_policy = {**denied_policy, "remote_processing_eligible": True}
    allowed = build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
        remote_subject_policy=allowed_policy,
        **{**common, "consent_store_path": tmp_path / "allowed-consent.json"},
    )
    receipt = allowed.grant_remote_consent(
        {"purpose": "remote_chat", "validity_days": 1}
    )["receipt"]
    assert receipt["provider_policy"] == PROVIDER_POLICY
    assert receipt["subject_policy"] == allowed_policy
    allowed.remote_provider_policy = {
        **PROVIDER_POLICY,
        "policy_version": "2026-08-13",
    }
    with pytest.raises(TeacherAgentDashboardError, match="policy is stale"):
        allowed.chat(
            {
                "messages": [{"role": "user", "content": "复习一下状态定义"}],
                "remote_consent_id": receipt["consent_id"],
            }
        )


def test_loopback_consent_grant_list_revoke_routes_expose_no_learner_text(tmp_path):
    snapshot = _snapshot(tmp_path, client=_Client())
    try:
        server, url = create_teacher_agent_dashboard_server(
            snapshot, capability_token="r" * 24
        )
    except PermissionError:
        pytest.skip("sandbox does not permit loopback socket binding")
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    parsed = urlsplit(url)

    def post(route, payload):
        connection = http.client.HTTPConnection(parsed.hostname, parsed.port, timeout=5)
        try:
            connection.request(
                "POST",
                f"{parsed.path}api/consent/{route}",
                body=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    try:
        status, granted = post(
            "grant",
            {
                "purpose": "remote_chat",
                "validity_days": 30,
                "likely_minor": False,
                "guardian_or_school_policy": "not_required",
            },
        )
        assert status == 200
        consent_id = granted["receipt"]["consent_id"]
        status, listed = post("list", {})
        assert status == 200
        assert listed["receipts"][0]["consent_id"] == consent_id
        assert "LEARNER_PRIVATE_MARKER" not in json.dumps(listed, ensure_ascii=False)
        status, revoked = post(
            "revoke",
            {"consent_id": consent_id, "reason_code": "user_revoked"},
        )
        assert status == 200
        assert revoked["receipt"]["status"] == "revoked"
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=5)


def test_remote_effects_fail_closed_without_configured_consent_store(tmp_path):
    client = _Client()
    snapshot = build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
        client=client,
    )
    with pytest.raises(
        TeacherAgentDashboardError, match="remote processing is disabled"
    ):
        snapshot.chat(
            {
                "messages": [{"role": "user", "content": "do not send"}],
                "remote_consent_id": "consent_attacker_supplied_1234",
            }
        )
    with pytest.raises(
        TeacherAgentDashboardError, match="remote processing is disabled"
    ):
        snapshot.start(_start_body(snapshot, "consent_attacker_supplied_1234"))
    assert client.calls == 0


def test_teach_rechecks_active_receipt_each_turn_before_state_mutation(tmp_path):
    client = _Client()
    snapshot = _snapshot(tmp_path, client=client, durable=True)
    receipt = _grant(snapshot, "remote_teaching")
    started = snapshot.start(_start_body(snapshot, receipt["consent_id"]))
    record = snapshot.sessions[started["session_id"]]
    before_session = deepcopy(record.session)
    before_context = record.context_version
    before_store_events = len(snapshot.store.events)
    before_calls = client.calls
    snapshot.revoke_remote_consent(
        {"consent_id": receipt["consent_id"], "reason_code": "user_revoked"}
    )
    with pytest.raises(TeacherAgentDashboardError, match="does not authorize"):
        snapshot.step(
            {
                "session_id": started["session_id"],
                "learner_response": "我不会，请继续讲解。",
                "expected_round": started["rounds_completed"],
                "expected_question_id": started["expected_question_id"],
                "expected_context_version": started["context_version"],
                "profile_revision": started["profile_summary"]["profile_revision"],
                "idempotency_key": "revoked-turn-1",
                "remote_consent_id": receipt["consent_id"],
            }
        )
    assert client.calls == before_calls
    assert record.active_turn_id is None
    assert record.context_version == before_context
    assert record.session == before_session
    assert len(snapshot.store.events) == before_store_events


def test_visual_semantics_sends_raw_image_only_after_exact_receipt(tmp_path):
    provider = _RemoteVisionProvider()
    snapshot = _snapshot(tmp_path, remote_vision=provider)
    session = snapshot.start(_start_body(snapshot))
    with pytest.raises(TeacherAgentDashboardError, match="visual_consent_id"):
        snapshot.upload_attachment(_attachment_body(session, key="visual-no-consent"))
    assert provider.calls == 0

    receipt = _grant(snapshot, "remote_visual_analysis")
    response = snapshot.upload_attachment(
        _attachment_body(
            session,
            key="visual-with-consent",
            consent_id=receipt["consent_id"],
        )
    )
    assert provider.calls == 1
    assert provider.last_bytes == PNG
    evidence = response["attachment"]
    assert evidence["remote_media_sent"] is True
    assert (
        evidence["visual_semantics"]["consent_receipt_sha256"]
        == receipt["receipt_sha256"]
    )
    encoded = json.dumps(response, ensure_ascii=False)
    assert base64.b64encode(PNG).decode() not in encoded

    snapshot.visual_provider_retention_days = 1
    current_session = {**session, "context_version": response["context_version"]}
    with pytest.raises(TeacherAgentDashboardError, match="policy is stale"):
        snapshot.upload_attachment(
            _attachment_body(
                current_session,
                key="visual-stale-policy",
                consent_id=receipt["consent_id"],
            )
        )
    assert provider.calls == 1


def test_local_visual_semantics_needs_no_remote_receipt(tmp_path):
    snapshot = _snapshot(tmp_path)
    session = snapshot.start(_start_body(snapshot))
    response = snapshot.upload_attachment(_attachment_body(session, key="local-visual"))
    evidence = response["attachment"]
    assert evidence["remote_media_sent"] is False
    assert evidence["visual_semantics"]["provider_id"] == "local-ocr-visual-v1"
    assert evidence["visual_semantics"]["consent_receipt_sha256"] is None


def test_project_export_and_purge_keep_subject_consent_as_shared_audit(tmp_path):
    snapshot = _snapshot(tmp_path, client=_Client(), durable=True)
    receipt = _grant(snapshot, "remote_chat")
    project = snapshot.create_project({"title": "同意导出项目"})["project"]
    archive = snapshot.export_project(project["project_id"])
    validate_project_export_archive(archive.payload)
    with zipfile.ZipFile(io.BytesIO(archive.payload)) as exported:
        path = f"consent_receipts/{receipt['consent_id']}.json"
        assert path in exported.namelist()
        exported_receipt = json.loads(exported.read(path))
        assert exported_receipt["receipt_sha256"] == receipt["receipt_sha256"]

    trash = snapshot.trash_project(project["project_id"], {})
    result = snapshot.purge_project(
        project["project_id"],
        {
            "recovery_token": trash["recovery_token"],
            "confirmation": deletion_confirmation(project["project_id"]),
        },
    )["deletion_receipt"]
    assert result["deleted_counts"]["consent_receipts"] == 0
    assert result["retained_shared_counts"]["consent_receipts"] == 1
    assert (
        snapshot.list_remote_consents({})["receipts"][0]["consent_id"]
        == receipt["consent_id"]
    )
