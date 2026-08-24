from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import pytest

from agent_harness.context import estimate_context_tokens, select_compaction_plan
from agent_harness.session import SESSION_SCHEMA, SESSION_SCHEMA_V1, SessionStore, SessionStoreError


def _store(tmp_path: Path) -> SessionStore:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    return SessionStore(workspace, state_home=tmp_path / "state")


def _descriptor(index: int = 0, *, estimated_tokens: int = 37) -> dict[str, Any]:
    return {
        "schema": "agent_harness.attachment.v1",
        "attachment_id": f"att_{index:032x}",
        "kind": "text",
        "media_type": "text/markdown; charset=utf-8",
        "display_name": f"notes-{index}.md",
        "size_bytes": estimated_tokens,
        "sha256": f"{index + 1:064x}",
        "estimated_tokens": estimated_tokens,
    }


def _pdf_descriptor(index: int, *, size_bytes: int) -> dict[str, Any]:
    value = _descriptor(index, estimated_tokens=size_bytes)
    value.update(
        {
            "kind": "pdf",
            "media_type": "application/pdf",
            "display_name": f"document-{index}.pdf",
            "size_bytes": size_bytes,
        }
    )
    return value


def _new_session(store: SessionStore) -> dict[str, Any]:
    return store.create(
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
    )


def _rewrite_schema(store: SessionStore, session_id: str, schema: str) -> None:
    path = store.sessions_directory / f"{session_id}.json"
    material = json.loads(path.read_text(encoding="utf-8"))
    material["schema"] = schema
    path.write_text(json.dumps(material), encoding="utf-8")
    path.chmod(0o600)


def test_v2_message_manifest_is_hashed_and_contains_no_local_payload(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = _new_session(store)
    assert session["schema"] == SESSION_SCHEMA == "agent_harness.session.v2"

    stored = store.append_message(
        session["session_id"],
        role="user",
        content="inspect this",
        attachments=[_descriptor()],
    )
    message = stored["messages"][-1]

    assert message["attachments"] == [_descriptor()]
    assert len(message["content_sha256"]) == 64
    assert len(message["attachment_manifest_sha256"]) == 64
    assert len(message["message_payload_sha256"]) == 64
    assert not ({"path", "blob", "base64", "content"} & set(message["attachments"][0]))

    view = store.context_view(session["session_id"])
    assert view["messages"][-1]["attachments"] == [_descriptor()]
    forked = store.fork(session["session_id"])
    assert forked["messages"][-1]["attachments"] == [_descriptor()]


def test_attachment_integrity_and_batch_identity_are_fail_closed(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = _new_session(store)

    with pytest.raises(SessionStoreError, match="not unique"):
        store.append_message(
            session["session_id"],
            role="user",
            content="duplicates",
            attachments=[_descriptor(), _descriptor()],
        )
    with pytest.raises(SessionStoreError, match="attachments are invalid"):
        store.append_message(
            session["session_id"],
            role="user",
            content="too many",
            attachments=[_descriptor(index) for index in range(9)],
        )
    with pytest.raises(SessionStoreError, match="batch is too large"):
        store.append_message(
            session["session_id"],
            role="user",
            content="too large together",
            attachments=[
                _pdf_descriptor(1, size_bytes=16 * 1024 * 1024),
                _pdf_descriptor(2, size_bytes=16 * 1024 * 1024),
            ],
        )

    stored = store.append_message(
        session["session_id"],
        role="user",
        content="one",
        attachments=[_descriptor()],
    )
    tampered = deepcopy(stored)
    tampered["messages"][-1]["attachments"][0]["display_name"] = "changed.md"
    with pytest.raises(SessionStoreError, match="manifest digest"):
        store.save(tampered)

    tampered = deepcopy(stored)
    tampered["messages"][-1]["message_payload_sha256"] = "0" * 64
    with pytest.raises(SessionStoreError, match="payload digest"):
        store.save(tampered)


def test_v1_text_session_stays_v1_until_a_real_attachment_is_added(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = _new_session(store)
    _rewrite_schema(store, session["session_id"], SESSION_SCHEMA_V1)

    loaded = store.load(session["session_id"])
    assert loaded["schema"] == SESSION_SCHEMA_V1
    text_only = store.append_message(
        session["session_id"], role="user", content="legacy", attachments=()
    )
    assert text_only["schema"] == SESSION_SCHEMA_V1
    upgraded = store.append_message(
        session["session_id"],
        role="assistant",
        content="with evidence",
        attachments=[_descriptor()],
    )
    assert upgraded["schema"] == SESSION_SCHEMA
    assert "message_payload_sha256" not in upgraded["messages"][0]
    assert "message_payload_sha256" in upgraded["messages"][1]


def test_v1_rejects_attachment_fields_but_preserves_legacy_compaction_hashes(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    session = _new_session(store)
    store.append_message(session["session_id"], role="user", content="old question")
    stored = store.append_message(
        session["session_id"], role="assistant", content="old answer"
    )
    _rewrite_schema(store, session["session_id"], SESSION_SCHEMA_V1)
    legacy = store.load(session["session_id"])
    record = store.record_compaction(
        session["session_id"],
        summary="legacy summary",
        source_message_count=2,
        provider="deepseek",
        model="deepseek-test",
        trigger="manual",
    )
    assert store.load(session["session_id"])["compactions"][-1] == record
    legacy_prefix = [
        {
            "message_id": item["message_id"],
            "role": item["role"],
            "content_sha256": item["content_sha256"],
        }
        for item in legacy["messages"]
    ]
    assert record["source_messages_sha256"] == sha256(
        json.dumps(
            legacy_prefix,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()
    assert [item["content_sha256"] for item in legacy["messages"]] == [
        item["content_sha256"] for item in stored["messages"]
    ]

    path = store.sessions_directory / f"{session['session_id']}.json"
    material = json.loads(path.read_text(encoding="utf-8"))
    material["messages"][0]["attachments"] = [_descriptor()]
    path.write_text(json.dumps(material), encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(SessionStoreError, match="unknown fields"):
        store.load(session["session_id"])


def test_begin_run_persists_attachment_manifest_atomically(tmp_path: Path) -> None:
    store = _store(tmp_path)
    session = _new_session(store)
    running = store.begin_run(
        session["session_id"],
        run_id="run_attach_12345678",
        turn_id="turn_attach_12345678",
        user_content="",
        provider="deepseek",
        model="deepseek-test",
        permission_mode="read-only",
        attachments=[_descriptor()],
    )

    assert running["runs"][-1]["status"] == "running"
    assert running["messages"][-1]["content"] == ""
    assert running["messages"][-1]["attachments"] == [_descriptor()]
    assert running["messages"][-1]["run_id"] == "run_attach_12345678"


def test_compaction_stops_before_and_cannot_cross_attachment_boundary(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    session = _new_session(store)
    store.append_message(session["session_id"], role="user", content="u0")
    store.append_message(session["session_id"], role="assistant", content="a0")
    store.append_message(
        session["session_id"],
        role="user",
        content="attached",
        attachments=[_descriptor()],
    )
    store.append_message(session["session_id"], role="assistant", content="a1")
    store.append_message(session["session_id"], role="user", content="u2")
    session = store.append_message(
        session["session_id"], role="assistant", content="a2"
    )

    plan = select_compaction_plan(
        session,
        retain_messages=2,
        max_source_bytes=10_000,
    )
    assert plan is not None
    assert plan.source_message_count == 2
    store.record_compaction(
        session["session_id"],
        summary="safe prefix",
        source_message_count=2,
        provider="deepseek",
        model="deepseek-test",
        trigger="manual",
    )
    with pytest.raises(SessionStoreError, match="attachment boundary"):
        store.record_compaction(
            session["session_id"],
            summary="unsafe prefix",
            source_message_count=4,
            provider="deepseek",
            model="deepseek-test",
            trigger="manual",
        )

    view = store.context_view(session["session_id"])
    assert view["messages"][0]["attachments"] == [_descriptor()]
    assert view["compacted_message_count"] == 2


def test_context_estimate_includes_historical_and_prospective_attachment_tokens() -> None:
    descriptor = _descriptor(estimated_tokens=211)
    baseline = estimate_context_tokens(
        summary="",
        messages=[{"role": "user", "content": "x"}],
    )
    historical = estimate_context_tokens(
        summary="",
        messages=[
            {"role": "user", "content": "x", "attachments": [descriptor]}
        ],
    )
    prospective = estimate_context_tokens(
        summary="",
        messages=[{"role": "user", "content": "x"}],
        prospective_attachments=[descriptor],
    )

    assert historical >= baseline + descriptor["estimated_tokens"]
    assert prospective >= baseline + descriptor["estimated_tokens"]
