from __future__ import annotations

import base64
import json
from pathlib import Path
import threading
from typing import Any

import pytest

from teaching_skill_miner.teacher_agent_dashboard import (
    TeacherAgentDashboardError,
    build_teacher_agent_dashboard_snapshot,
)


ROOT = Path(__file__).resolve().parents[1]


class _CapturingChatClient:
    native_stream_available = False

    def __init__(self) -> None:
        self.calls = 0
        self.messages: list[Any] = []
        self.lock = threading.Lock()

    def public_status(self) -> dict[str, object]:
        return {
            "provider": "deepseek",
            "model": "fake-chat",
            "base_origin": "https://api.deepseek.example",
            "configured": True,
            "remote_student_data_opt_in": True,
            "api_key_exposed": False,
        }

    def chat_json(
        self,
        messages: object,
        *,
        request_kind: str,
        require_remote_consent: bool = True,
    ) -> tuple[dict[str, object], dict[str, object]]:
        assert request_kind == "console_direct_chat"
        assert require_remote_consent is True
        with self.lock:
            self.calls += 1
            self.messages = json.loads(json.dumps(messages, ensure_ascii=False))
        return (
            {"message": f"已依据本地检索出的有限附件摘录回答（{self.calls}）。"},
            {
                "provider": "deepseek",
                "model": "fake-chat",
                "latency_ms": 1,
                "usage": {},
            },
        )


def _snapshot(tmp_path: Path, client: _CapturingChatClient):
    return build_teacher_agent_dashboard_snapshot(
        ROOT / "data/teacher_agent_skill_library_v2.json",
        ROOT / "data/teacher_agent_demo_input.json",
        ROOT / "data/teacher_agent_evaluation_cases.json",
        client=client,
        project_store_path=tmp_path / "projects",
        resource_index_store_path=tmp_path / "resources",
        consent_store_path=tmp_path / "consent.json",
        consent_signing_secret=b"chat-attachment-consent-signing-secret-material",
    )


def _grant(snapshot) -> dict[str, Any]:
    return snapshot.grant_remote_consent(
        {
            "purpose": "remote_chat",
            "validity_days": 1,
            "likely_minor": False,
            "guardian_or_school_policy": "not_required",
        }
    )["receipt"]


def _project(snapshot, title: str) -> dict[str, Any]:
    return snapshot.create_project({"title": title})["project"]


def _upload(
    snapshot,
    project: dict[str, Any],
    *,
    name: str,
    content: bytes,
    mime_type: str = "text/plain",
) -> tuple[dict[str, Any], dict[str, str]]:
    response = snapshot.upload_resource(
        {
            "resource_idempotency_key": f"resource-{name}",
            "mime_type": mime_type,
            "display_name": name,
            "data_base64": base64.b64encode(content).decode("ascii"),
        }
    )
    resource = response["resource"]
    snapshot.add_project_reference(
        project["project_id"],
        {"kind": "resource", "reference_id": resource["resource_id"]},
    )
    return response, {
        "resource_id": resource["resource_id"],
        "staged_resource_id": resource["staged_resource_id"],
    }


def _chat_body(
    project: dict[str, Any],
    consent_id: str,
    refs: list[dict[str, str]],
    *,
    prompt: str = "动态规划的状态表示是什么？",
    thread: str = "1" * 24,
) -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": prompt}],
        "remote_consent_id": consent_id,
        "project_id": project["project_id"],
        "chat_thread_id": f"chat_{thread}",
        "resource_refs": refs,
    }


def test_chat_attachment_uses_project_owned_bounded_excerpt_and_hash_receipt(
    tmp_path: Path,
) -> None:
    client = _CapturingChatClient()
    snapshot = _snapshot(tmp_path, client)
    policy = next(
        item
        for item in snapshot.bootstrap()["remote_consent"]["policies"]
        if item["purpose"] == "remote_chat"
    )
    assert policy["data_categories"] == [
        "learner_message",
        "teaching_resource_excerpt",
    ]
    project = _project(snapshot, "附件问答")
    private_marker = "PRIVATE_RESOURCE_SENTINEL_7F3A"
    _response, ref = _upload(
        snapshot,
        project,
        name="dp-notes.txt",
        content=(
            f"{private_marker}\n动态规划的状态表示，是对子问题求解所需信息的最小描述。\n"
            + "无关附录。" * 4_000
        ).encode(),
    )
    result = snapshot.chat(_chat_body(project, _grant(snapshot)["consent_id"], [ref]))

    assert client.calls == 1
    provider_payload = json.dumps(client.messages, ensure_ascii=False)
    assert "动态规划的状态表示" in provider_payload
    # Retrieval is query-related: an unrelated sentinel outside the selected
    # excerpt is not included merely because the file was attached.
    assert provider_payload.count("无关附录") < 100
    assert len(provider_payload) < 12_000
    receipt = result["context_receipt"]
    resource_receipt = receipt["resource_context"]
    assert resource_receipt["raw_media_sent"] is False
    assert resource_receipt["full_resource_text_sent"] is False
    assert resource_receipt["context_role"] == "untrusted_quote_only"
    assert resource_receipt["citations"][0]["resource_id"] == ref["resource_id"]
    assert len(resource_receipt["receipt_sha256"]) == 64
    assert len(receipt["system_prompt_sha256"]) == 64
    assert len(receipt["context_receipt_sha256"]) == 64
    assert private_marker not in json.dumps(receipt, ensure_ascii=False)


def test_remote_chat_redacts_identifiers_in_history_query_and_resource_appendix(
    tmp_path: Path,
) -> None:
    client = _CapturingChatClient()
    snapshot = _snapshot(tmp_path, client)
    project = _project(snapshot, "远程最小化")
    identifiers = (
        "student42@example.test",
        "+1 415-555-0199",
        "11010519491231002X",
        "https://private.example/student/42",
        "/" + "Users/student/private/answer.txt",
    )
    resource_text = " 联系方式与本地作业路径：" + " ".join(identifiers)
    _resource, ref = _upload(
        snapshot,
        project,
        name="contact-notes.txt",
        content=resource_text.encode(),
    )
    consent_id = _grant(snapshot)["consent_id"]
    first_body = _chat_body(
        project,
        consent_id,
        [ref],
        prompt="请总结联系方式 " + identifiers[0],
        thread="a" * 24,
    )
    first_body["request_id"] = "chat-redaction-observation-001"
    first = snapshot.chat(first_body)
    first_payload = json.dumps(client.messages, ensure_ascii=False)
    assert all(value not in first_payload for value in identifiers)
    assert "[REDACTED_EMAIL]" in first_payload
    assert "[REDACTED_PHONE]" in first_payload
    assert "[REDACTED_CN_ID]" in first_payload
    assert "[REDACTED_URL]" in first_payload
    assert "[REDACTED_LOCAL_PATH]" in first_payload
    redaction = first["context_receipt"]["remote_redaction"]
    assert redaction["applied"] is True
    assert redaction["finding_count"] >= len(identifiers)
    assert all(value not in json.dumps(redaction) for value in identifiers)

    # The durable local transcript may retain the learner's original text by
    # local policy, but a later remote turn must redact that history again.
    second_body = _chat_body(
        project,
        consent_id,
        [ref],
        prompt="也请检查电话 " + identifiers[1],
        thread="a" * 24,
    )
    second_body["request_id"] = "chat-redaction-observation-002"
    snapshot.chat(second_body)
    second_payload = json.dumps(client.messages, ensure_ascii=False)
    assert all(value not in second_payload for value in identifiers)


def test_forged_other_project_and_mismatched_stage_are_rejected_before_model(
    tmp_path: Path,
) -> None:
    client = _CapturingChatClient()
    snapshot = _snapshot(tmp_path, client)
    first = _project(snapshot, "项目甲")
    second = _project(snapshot, "项目乙")
    _one, first_ref = _upload(
        snapshot,
        first,
        name="first.txt",
        content="动态规划状态由阶段和容量共同确定。".encode(),
    )
    _two, second_ref = _upload(
        snapshot,
        first,
        name="second.txt",
        content="动态规划的转移必须保持子问题闭包。".encode(),
    )
    consent = _grant(snapshot)["consent_id"]
    with pytest.raises(TeacherAgentDashboardError, match="not owned"):
        snapshot.chat(_chat_body(second, consent, [first_ref]))
    forged_pair = {
        "resource_id": first_ref["resource_id"],
        "staged_resource_id": second_ref["staged_resource_id"],
    }
    with pytest.raises(TeacherAgentDashboardError, match="does not match"):
        snapshot.chat(_chat_body(first, consent, [forged_pair], thread="2" * 24))
    assert client.calls == 0


def test_raw_attachment_payload_is_rejected_instead_of_silently_ignored(
    tmp_path: Path,
) -> None:
    client = _CapturingChatClient()
    snapshot = _snapshot(tmp_path, client)
    project = _project(snapshot, "IDs only")
    body = _chat_body(project, _grant(snapshot)["consent_id"], [])
    body["data_base64"] = base64.b64encode(b"private raw bytes").decode()
    with pytest.raises(TeacherAgentDashboardError, match="resource IDs only"):
        snapshot.chat(body)
    assert client.calls == 0


def test_visual_pending_and_cross_layer_abstention_never_reach_model(
    tmp_path: Path,
) -> None:
    client = _CapturingChatClient()
    snapshot = _snapshot(tmp_path, client)
    project = _project(snapshot, "待复核附件")
    pending, pending_ref = _upload(
        snapshot,
        project,
        name="pending.txt",
        content="动态规划状态表示包含阶段和容量。".encode(),
    )
    assert pending["session_use"]["status"] == "eligible_untrusted_context"
    assert snapshot.resource_index_store is not None
    index_path = snapshot.resource_index_store._path(
        pending["resource"]["content_sha256"]
    )
    index_document = json.loads(index_path.read_text(encoding="utf-8"))
    for chunk in index_document["retrieval_index"]["chunks"]:
        chunk["needs_visual_review"] = True
    index_path.write_text(
        json.dumps(index_document, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    index_path.chmod(0o600)
    consent = _grant(snapshot)["consent_id"]
    with pytest.raises(
        TeacherAgentDashboardError, match="visual semantics are pending"
    ):
        snapshot.chat(_chat_body(project, consent, [pending_ref]))

    blocked, blocked_ref = _upload(
        snapshot,
        project,
        name="formula.csv",
        mime_type="text/csv",
        content="项目,数值\n阈值,=1+1\n".encode(),
    )
    assert blocked["session_use"]["status"] == "blocked_abstained"
    with pytest.raises(TeacherAgentDashboardError, match="blocked pending"):
        snapshot.chat(_chat_body(project, consent, [blocked_ref], thread="2" * 24))
    assert client.calls == 0


def test_conflicting_attachment_sources_require_confirmation_before_model(
    tmp_path: Path,
) -> None:
    client = _CapturingChatClient()
    snapshot = _snapshot(tmp_path, client)
    project = _project(snapshot, "冲突附件")
    _first, first_ref = _upload(
        snapshot,
        project,
        name="policy-a.txt",
        content="系统发布阈值为 8，系统发布阈值用于正式部署。".encode(),
    )
    _second, second_ref = _upload(
        snapshot,
        project,
        name="policy-b.txt",
        content="系统发布阈值不是 8，系统发布阈值用于正式部署。".encode(),
    )
    with pytest.raises(TeacherAgentDashboardError, match="conflicting evidence"):
        snapshot.chat(
            _chat_body(
                project,
                _grant(snapshot)["consent_id"],
                [first_ref, second_ref],
                prompt="系统发布阈值是什么？",
            )
        )
    assert client.calls == 0


def test_attachment_consent_category_and_revocation_are_effect_boundary_guards(
    tmp_path: Path,
) -> None:
    client = _CapturingChatClient()
    snapshot = _snapshot(tmp_path, client)
    project = _project(snapshot, "同意边界")
    _resource, ref = _upload(
        snapshot,
        project,
        name="private.txt",
        content="动态规划状态表示包含当前阶段。".encode(),
    )
    assert snapshot.consent_store is not None
    assert snapshot.consent_subject_id is not None
    incomplete = snapshot.consent_store.grant(
        subject_id=snapshot.consent_subject_id,
        purpose="remote_chat",
        provider_id="deepseek",
        processing_region="provider_managed",
        data_categories=["learner_message"],
        provider_retention_days=30,
        validity_days=1,
    )
    with pytest.raises(TeacherAgentDashboardError, match="does not authorize"):
        snapshot.chat(_chat_body(project, incomplete["consent_id"], [ref]))

    revoked = _grant(snapshot)
    snapshot.revoke_remote_consent(
        {"consent_id": revoked["consent_id"], "reason_code": "user_revoked"}
    )
    with pytest.raises(TeacherAgentDashboardError, match="does not authorize"):
        snapshot.chat(
            _chat_body(project, revoked["consent_id"], [ref], thread="2" * 24)
        )
    assert client.calls == 0


def test_attachment_revocation_during_local_retrieval_stops_remote_effect(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = _CapturingChatClient()
    snapshot = _snapshot(tmp_path, client)
    project = _project(snapshot, "effect boundary")
    _resource, ref = _upload(
        snapshot,
        project,
        name="effect-boundary.txt",
        content="动态规划状态表示保存当前子问题所需的最小信息。".encode(),
    )
    receipt = _grant(snapshot)
    snapshot_type = type(snapshot)
    original = snapshot_type._chat_resource_context

    def resolve_then_revoke(self, **kwargs):
        result = original(self, **kwargs)
        self.revoke_remote_consent(
            {"consent_id": receipt["consent_id"], "reason_code": "user_revoked"}
        )
        return result

    monkeypatch.setattr(snapshot_type, "_chat_resource_context", resolve_then_revoke)
    with pytest.raises(TeacherAgentDashboardError, match="does not authorize"):
        snapshot.chat(_chat_body(project, receipt["consent_id"], [ref]))
    assert client.calls == 0


def test_private_resource_text_never_enters_sse_journal_or_result_reference(
    tmp_path: Path,
) -> None:
    client = _CapturingChatClient()
    snapshot = _snapshot(tmp_path, client)
    project = _project(snapshot, "SSE 隐私")
    private_marker = "SSE_PRIVATE_RESOURCE_SENTINEL_A91C"
    _resource, ref = _upload(
        snapshot,
        project,
        name="sse-private.txt",
        content=f"状态表示保存解题所需的最小信息，私有标记为 {private_marker}。".encode(),
    )
    record, _cursor = snapshot.open_harness_stream(
        {
            "operation": "chat",
            "request_id": "chat-attachment-private-sse-001",
            "payload": _chat_body(
                project,
                _grant(snapshot)["consent_id"],
                [ref],
                prompt="状态表示保存什么信息？",
            ),
        }
    )
    assert record.handle.wait(timeout=5)["status"] == "completed"
    assert private_marker in json.dumps(client.messages, ensure_ascii=False)
    events = record.journal.replay()
    serialized = json.dumps(events, ensure_ascii=False)
    assert private_marker not in serialized
    operation_result = next(
        item for item in events if item["type"] == "operation.result"
    )
    chat_ref = operation_result["payload"]["result"]["chat"]
    assert "context_receipt" not in chat_ref
    assert "message" not in chat_ref
    assert "resource_refs" not in serialized
    assert client.calls == 1
