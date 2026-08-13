from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json
import threading

import pytest
from jsonschema import Draft202012Validator

from teaching_skill_miner.teacher_agent_projects import (
    LEARNING_PROJECT_SCHEMA,
    LearningProjectError,
    LearningProjectStore,
    new_learning_project,
    validate_learning_project,
)


def _thread(thread_id: str = "chat_0123456789abcdef01234567") -> dict:
    return {
        "thread_id": thread_id,
        "title": "动态规划讨论",
        "created_at": "2026-08-12T09:00:00Z",
        "updated_at": "2026-08-12T09:01:00Z",
        "messages": [
            {
                "message_id": "message_1",
                "role": "user",
                "content": "我不理解状态转移。",
                "status": "completed",
                "created_at": "2026-08-12T09:00:00Z",
                "web_search_used": False,
                "sources": [],
            },
            {
                "message_id": "message_2",
                "role": "assistant",
                "content": "我们先从一个最小例子开始。",
                "status": "completed",
                "created_at": "2026-08-12T09:01:00Z",
                "web_search_used": True,
                "sources": [{"title": "来源", "url": "https://example.test/dp"}],
            },
        ],
    }


def test_project_store_persists_links_chat_notes_and_metadata(tmp_path: Path) -> None:
    store = LearningProjectStore(tmp_path / "projects")
    project = store.create(title="算法课", description="算法基础学习项目")
    assert project["schema"] == LEARNING_PROJECT_SCHEMA
    project_id = project["project_id"]

    store.add_reference(
        project_id, kind="syllabus", reference_id="syl_0123456789abcdef01234567"
    )
    store.add_reference(
        project_id, kind="teaching_session", reference_id="session_dp_1"
    )
    store.add_reference(project_id, kind="resource", reference_id="resource_slides_1")
    store.add_reference(project_id, kind="resource", reference_id="resource_slides_1")
    stored = store._commit_chat_thread(project_id, _thread())
    stored = store.upsert_note(
        project_id,
        {
            "note_id": "note_0123456789abcdef01234567",
            "title": "复习重点",
            "body": "状态定义与边界条件",
            "created_at": "2026-08-12T09:02:00Z",
            "updated_at": "2026-08-12T09:02:00Z",
        },
    )
    stored = store.update_metadata(project_id, title="算法学习", pinned=True)

    assert stored["title"] == "算法学习"
    assert stored["pinned"] is True
    assert stored["resource_ids"] == ["resource_slides_1"]
    assert stored["chat_threads"][0]["messages"][1]["web_search_used"] is True
    assert stored["notes"][0]["title"] == "复习重点"
    assert store.list()[0] == {
        "project_id": project_id,
        "title": "算法学习",
        "description": "算法基础学习项目",
        "status": "active",
        "pinned": True,
        "updated_at": stored["updated_at"],
        "syllabus_count": 1,
        "teaching_session_count": 1,
        "resource_count": 1,
        "chat_thread_count": 1,
        "note_count": 1,
    }


def test_project_claim_boundary_and_strict_fields_fail_closed() -> None:
    project = new_learning_project(title="物理课")
    altered = deepcopy(project)
    altered["claim_boundary"]["project_resources_are_scoring_gold"] = True
    with pytest.raises(LearningProjectError, match="claim_boundary"):
        validate_learning_project(altered)

    altered = deepcopy(project)
    altered["unexpected"] = True
    with pytest.raises(LearningProjectError, match="invalid fields"):
        validate_learning_project(altered)


def test_chat_validation_rejects_duplicate_ids_and_non_http_sources(
    tmp_path: Path,
) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create(title="数学课")
    thread = _thread()
    thread["messages"][1]["message_id"] = "message_1"
    with pytest.raises(LearningProjectError, match="duplicate ids"):
        store._commit_chat_thread(project["project_id"], thread)

    thread = _thread()
    thread["messages"][1]["sources"][0]["url"] = "javascript:alert(1)"
    with pytest.raises(LearningProjectError, match="url is invalid"):
        store._commit_chat_thread(project["project_id"], thread)


def test_project_trash_is_recoverable_and_removes_active_entry(tmp_path: Path) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create(title="化学课")
    receipt = store.trash(project["project_id"])
    assert receipt["project_id"] == project["project_id"]
    assert receipt["recovery_token"].startswith("restore_project_")
    assert store.list() == []
    assert store.list_trash()[0]["recovery_token"] == receipt["recovery_token"]
    with pytest.raises(LearningProjectError, match="not found"):
        store.read(project["project_id"])
    restored = store.restore(receipt["recovery_token"])
    assert restored["project_id"] == project["project_id"]
    assert store.list_trash() == []
    assert store.read(project["project_id"])["title"] == "化学课"


def test_permanent_project_purge_removes_private_operation_receipts(
    tmp_path: Path,
) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create_idempotent(
        operation_id="create:purge-receipts-0001", title="待永久删除"
    )
    receipt = store.trash(project["project_id"])
    assert json.loads((tmp_path / ".operations.json").read_text(encoding="utf-8"))[
        "receipts"
    ]

    assert store.purge_trash(receipt["recovery_token"]) == project["project_id"]
    assert json.loads((tmp_path / ".operations.json").read_text(encoding="utf-8"))[
        "receipts"
    ] == {}


def test_project_restore_rejects_unknown_token_without_affecting_other_projects(
    tmp_path: Path,
) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create(title="地理课")
    receipt = store.trash(project["project_id"])
    with pytest.raises(LearningProjectError, match="invalid"):
        store.restore("not-a-token")
    store.create(title="另一个项目")
    restored = store.restore(receipt["recovery_token"])
    assert restored["title"] == "地理课"


def test_learning_project_matches_public_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    schema = json.loads(
        (root / "schema" / "learning_project.schema.json").read_text(encoding="utf-8")
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(new_learning_project(title="生物课"))


def test_concurrent_reference_updates_do_not_lose_successful_mutations(
    tmp_path: Path,
) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create(title="并发项目")
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def add(kind: str, reference_id: str) -> None:
        try:
            barrier.wait(timeout=3)
            store.add_reference(
                project["project_id"], kind=kind, reference_id=reference_id
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=add, args=("syllabus", "syllabus_a")),
        threading.Thread(target=add, args=("resource", "resource_b")),
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert errors == []
    restored = store.read(project["project_id"])
    assert restored["syllabus_ids"] == ["syllabus_a"]
    assert restored["resource_ids"] == ["resource_b"]


def test_chat_upsert_cannot_truncate_or_rewrite_durable_history(
    tmp_path: Path,
) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create(title="权威对话")
    durable = _thread()
    store._commit_chat_thread(project["project_id"], durable)

    truncated = deepcopy(durable)
    truncated["messages"] = truncated["messages"][:1]
    with pytest.raises(LearningProjectError, match="cannot truncate"):
        store._commit_chat_thread(project["project_id"], truncated)

    rewritten = deepcopy(durable)
    rewritten["messages"][0]["content"] = "被陈旧客户端改写"
    with pytest.raises(LearningProjectError, match="conflicts"):
        store._commit_chat_thread(project["project_id"], rewritten)

    browser_projection = deepcopy(durable)
    browser_projection["messages"][1]["message_id"] = "browser_message_2"
    browser_projection["messages"][1]["web_search_used"] = False
    browser_projection["messages"][1]["sources"] = []
    restored = store._commit_chat_thread(project["project_id"], browser_projection)
    authoritative = restored["chat_threads"][0]["messages"][1]
    assert authoritative["message_id"] == "message_2"
    assert authoritative["web_search_used"] is True
    assert authoritative["sources"] == [
        {"title": "来源", "url": "https://example.test/dp"}
    ]


def test_client_chat_upsert_only_accepts_one_server_normalized_user_message(
    tmp_path: Path,
) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create(title="客户端权限边界")
    thread = _thread()

    with pytest.raises(LearningProjectError, match="only one learner message"):
        store.upsert_client_chat_thread(project["project_id"], thread)

    learner_only = deepcopy(thread)
    learner_only["messages"] = learner_only["messages"][:1]
    learner_only["messages"][0]["message_id"] = "forged_browser_id"
    stored = store.upsert_client_chat_thread(project["project_id"], learner_only)
    message = stored["chat_threads"][0]["messages"][0]
    assert message["role"] == "user"
    assert message["message_id"].startswith("server_")
    assert message["message_id"] != "forged_browser_id"
    assert message["sources"] == []

    forged = deepcopy(learner_only)
    forged["messages"].append(thread["messages"][1])
    with pytest.raises(LearningProjectError, match="assistant or tool"):
        store.upsert_client_chat_thread(project["project_id"], forged)


def test_default_project_and_legacy_migration_are_durable_idempotent(
    tmp_path: Path,
) -> None:
    store = LearningProjectStore(tmp_path)
    first = store.create_default(
        idempotency_key="desktop-install-v1",
        title="我的学习空间",
        description="本地持久空间",
    )
    replay = LearningProjectStore(tmp_path).create_default(
        idempotency_key="desktop-install-v1",
        title="我的学习空间",
        description="本地持久空间",
    )
    assert replay == first
    legacy = _thread()
    migrated = store.migrate_legacy_handles(
        first["project_id"],
        operation_id="migration:browser-cache-v1",
        chat_threads=[legacy],
        teaching_session_ids=["session_legacy_1"],
    )
    assert migrated["teaching_session_ids"] == ["session_legacy_1"]
    assert migrated["chat_threads"][0]["title"].startswith("旧版迁移")
    assert migrated["chat_threads"][0]["messages"][1]["sources"] == []
    assert migrated["chat_threads"][0]["messages"][1]["web_search_used"] is False
    assert "未验证" in migrated["chat_threads"][0]["messages"][1]["content"]
    assert (
        store.migrate_legacy_handles(
            first["project_id"],
            operation_id="migration:browser-cache-v1",
            chat_threads=[legacy],
            teaching_session_ids=["session_legacy_1"],
        )
        == migrated
    )


def test_project_cas_idempotent_notes_and_searchable_cursor_pages(
    tmp_path: Path,
) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create_idempotent(
        operation_id="create:workspace-0001", title="分页项目"
    )
    assert (
        store.create_idempotent(operation_id="create:workspace-0001", title="分页项目")
        == project
    )
    updated = store.upsert_note_content(
        project["project_id"],
        note_id=None,
        title="第一条",
        body="动态规划边界",
        expected_updated_at=project["updated_at"],
        operation_id="note:create-0001",
    )
    replay = store.upsert_note_content(
        project["project_id"],
        note_id=None,
        title="第一条",
        body="动态规划边界",
        expected_updated_at=project["updated_at"],
        operation_id="note:create-0001",
    )
    assert replay == updated
    with pytest.raises(LearningProjectError, match="revision conflict"):
        store.upsert_note_content(
            project["project_id"],
            note_id=updated["notes"][0]["note_id"],
            title="陈旧写入",
            body="不应覆盖",
            expected_updated_at=project["updated_at"],
            operation_id="note:update-stale-0001",
        )
    current = updated
    for index in range(3):
        current = store.upsert_note_content(
            project["project_id"],
            note_id=None,
            title=f"检索笔记 {index}",
            body="状态转移" if index == 2 else "其他内容",
            expected_updated_at=current["updated_at"],
            operation_id=f"note:create-page-{index}",
        )
    first_page = store.browse(project["project_id"], section="notes", limit=2)
    assert len(first_page["items"]) == 2
    assert first_page["next_cursor"]
    second_page = store.browse(
        project["project_id"],
        section="notes",
        limit=2,
        cursor=first_page["next_cursor"],
    )
    assert len(second_page["items"]) == 2
    found = store.browse(project["project_id"], section="notes", query="状态转移")
    assert found["total"] == 1
    with pytest.raises(LearningProjectError, match="stale or invalid"):
        store.browse(
            project["project_id"], section="notes", cursor="cursor_2_deadbeefdeadbeef"
        )


def test_reference_removal_is_cas_guarded_and_replay_safe(tmp_path: Path) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create(title="资源所有权")
    linked = store.add_reference(
        project["project_id"],
        kind="resource",
        reference_id="res_0123456789abcdef0123",
    )
    removed = store.remove_reference(
        project["project_id"],
        kind="resource",
        reference_id="res_0123456789abcdef0123",
        expected_updated_at=linked["updated_at"],
        operation_id="reference:remove-0001",
    )
    assert removed["resource_ids"] == []
    assert (
        store.remove_reference(
            project["project_id"],
            kind="resource",
            reference_id="res_0123456789abcdef0123",
            expected_updated_at=removed["updated_at"],
            operation_id="reference:remove-0001",
        )
        == removed
    )


def test_reference_operation_recovers_after_effect_before_receipt_crash_window(
    tmp_path: Path,
) -> None:
    store = LearningProjectStore(tmp_path)
    project = store.create(title="崩溃恢复")
    linked = store.add_reference(
        project["project_id"],
        kind="resource",
        reference_id="res_0123456789abcdef0123",
        expected_updated_at=project["updated_at"],
        operation_id="reference:add-crash-window-0001",
    )
    (tmp_path / ".operations.json").unlink()

    replay = store.add_reference(
        project["project_id"],
        kind="resource",
        reference_id="res_0123456789abcdef0123",
        # A retry may rebase its CAS after the successful response was lost.
        # The operation identity is the logical set-add, not its first base.
        expected_updated_at=linked["updated_at"],
        operation_id="reference:add-crash-window-0001",
    )

    assert replay == linked
    operations = json.loads(
        (tmp_path / ".operations.json").read_text(encoding="utf-8")
    )
    assert "reference:add-crash-window-0001" in operations["receipts"]
