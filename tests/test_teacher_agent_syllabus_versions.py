from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import validate as jsonschema_validate

from teaching_skill_miner.teacher_agent_syllabus import (
    _seal_generated_syllabus,
    revise_teaching_syllabus,
    teaching_syllabus_editable_draft,
    validate_teaching_syllabus,
)
from teaching_skill_miner.teacher_agent_syllabus_versions import (
    TeachingSyllabusVersionError,
    TeachingSyllabusVersionStore,
)


def _draft(title: str = "动态规划基础") -> dict:
    return {
        "title": title,
        "description": "从状态定义到迁移判断的可执行课程。",
        "audience": "零基础本科生",
        "estimated_duration_minutes": 60,
        "learning_objectives": ["能定义状态并解释状态转移"],
        "prerequisites": ["基本递归"],
        "modules": [
            {
                "title": "状态与转移",
                "description": "用一个最小例子建立状态语义。",
                "lessons": [
                    {
                        "title": "最小状态定义",
                        "objective": "定义状态并说明下标和值的含义。",
                        "summary": "从子问题出发定义状态。",
                        "duration_minutes": 60,
                        "knowledge_components": ["状态定义"],
                        "materials": {
                            "example": "爬楼梯问题",
                            "practice": "定义 dp[i]",
                            "transfer_task": "迁移到硬币问题",
                        },
                    }
                ],
            }
        ],
    }


def _base() -> dict:
    return _seal_generated_syllabus(
        _draft(),
        model="deepseek-test",
        source_resource_ids=[],
        created_at="2026-08-12T10:00:00Z",
    )


def test_teacher_revision_is_immutable_non_authoritative_and_parent_bound() -> None:
    base = _base()
    editable = teaching_syllabus_editable_draft(base)
    editable["title"] = "动态规划基础（教师修订）"
    editable["modules"][0]["lessons"][0]["duration_minutes"] = 55
    editable["estimated_duration_minutes"] = 55

    revised = revise_teaching_syllabus(
        base,
        editable,
        change_summary="缩短示例并明确课节标题",
        created_at="2026-08-12T11:00:00Z",
    )

    validate_teaching_syllabus(revised)
    assert revised["syllabus_id"] != base["syllabus_id"]
    assert revised["source"]["kind"] == "teacher_edited"
    assert revised["source"]["parent_syllabus_id"] == base["syllabus_id"]
    assert (
        revised["source"]["parent_content_sha256"]
        == base["integrity"]["content_sha256"]
    )
    goal = revised["modules"][0]["lessons"][0]["teaching_goal"]
    assert (
        goal["knowledge_spec"]["authority"]["authoritative_for_runtime_grading"]
        is False
    )
    assert base["title"] == "动态规划基础"


def test_version_store_create_publish_and_append_only_rollback(tmp_path: Path) -> None:
    base = _base()
    editable = teaching_syllabus_editable_draft(base)
    editable["title"] = "动态规划基础（版本二）"
    revised = revise_teaching_syllabus(
        base,
        editable,
        change_summary="更新标题",
        created_at="2026-08-12T11:00:00Z",
    )
    store = TeachingSyllabusVersionStore(tmp_path)
    first = store.register(
        base,
        idempotency_key="register-base",
        occurred_at_utc="2026-08-12T10:00:01Z",
    )
    draft = store.create_revision(
        base_syllabus_id=base["syllabus_id"],
        revised_syllabus=revised,
        change_summary="更新标题",
        expected_version=first["version"],
        idempotency_key="create-v2",
        occurred_at_utc="2026-08-12T11:00:01Z",
    )
    revision_two = draft["revisions"][-1]
    assert revision_two["status"] == "draft"
    with pytest.raises(TeachingSyllabusVersionError, match="currently published"):
        store.require_published(revised["syllabus_id"])

    published = store.publish(
        family_id=first["family_id"],
        revision_id=revision_two["revision_id"],
        expected_version=draft["version"],
        idempotency_key="publish-v2",
        occurred_at_utc="2026-08-12T11:05:00Z",
    )
    assert (
        store.require_published(revised["syllabus_id"])["published_revision_id"]
        == revision_two["revision_id"]
    )
    with pytest.raises(TeachingSyllabusVersionError, match="currently published"):
        store.require_published(base["syllabus_id"])

    rolled_back = store.rollback(
        family_id=first["family_id"],
        revision_id=first["revisions"][0]["revision_id"],
        expected_version=published["version"],
        idempotency_key="rollback-v1",
        occurred_at_utc="2026-08-12T11:10:00Z",
    )
    assert rolled_back["published_revision_id"] == first["revisions"][0]["revision_id"]
    assert len(json.loads(store.path.read_text())["events"]) == 4


def test_version_store_cas_idempotency_and_tamper_fail_closed(tmp_path: Path) -> None:
    base = _base()
    editable = teaching_syllabus_editable_draft(base)
    editable["description"] = "教师修订后的描述。"
    revised = revise_teaching_syllabus(
        base,
        editable,
        change_summary="修订描述",
        created_at="2026-08-12T11:00:00Z",
    )
    store = TeachingSyllabusVersionStore(tmp_path)
    family = store.register(base, idempotency_key="register")
    created = store.create_revision(
        base_syllabus_id=base["syllabus_id"],
        revised_syllabus=revised,
        change_summary="修订描述",
        expected_version=family["version"],
        idempotency_key="create",
    )
    replay = store.create_revision(
        base_syllabus_id=base["syllabus_id"],
        revised_syllabus=revised,
        change_summary="修订描述",
        expected_version=family["version"],
        idempotency_key="create",
    )
    assert replay == created
    with pytest.raises(TeachingSyllabusVersionError, match="version conflict"):
        store.publish(
            family_id=family["family_id"],
            revision_id=created["revisions"][-1]["revision_id"],
            expected_version=family["version"],
            idempotency_key="stale-publish",
        )

    state = json.loads(store.path.read_text())
    state["events"][0]["payload"]["change_summary"] = "tampered"
    store.path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(TeachingSyllabusVersionError, match="hash"):
        store.list_families()


def test_revision_rejects_gold_or_unknown_fields() -> None:
    base = _base()
    editable = teaching_syllabus_editable_draft(base)
    editable["answer_key"] = "not allowed"
    with pytest.raises(Exception, match="fields|answer-key|gold"):
        revise_teaching_syllabus(base, editable, change_summary="unsafe")

    clean = teaching_syllabus_editable_draft(base)
    clean["modules"] = deepcopy(clean["modules"])
    clean["modules"][0]["lessons"][0]["materials"]["example"] = "标准答案：42"
    with pytest.raises(Exception, match="answer-key|gold"):
        revise_teaching_syllabus(base, clean, change_summary="unsafe")


def test_version_store_matches_public_schema(tmp_path: Path) -> None:
    store = TeachingSyllabusVersionStore(tmp_path)
    store.register(_base(), idempotency_key="schema-register")
    schema = json.loads(
        (
            Path(__file__).resolve().parents[1]
            / "schema"
            / "teaching_syllabus_version_store.schema.json"
        ).read_text()
    )
    jsonschema_validate(json.loads(store.path.read_text()), schema)


def test_family_purge_compacts_content_and_fences_resurrection(tmp_path: Path) -> None:
    base = _base()
    editable = teaching_syllabus_editable_draft(base)
    editable["title"] = "永久删除哨兵标题"
    revised = revise_teaching_syllabus(
        base,
        editable,
        change_summary="永久删除哨兵修订说明",
        created_at="2026-08-12T11:00:00Z",
    )
    store = TeachingSyllabusVersionStore(tmp_path)
    family = store.register(base, idempotency_key="purge-register")
    store.create_revision(
        base_syllabus_id=base["syllabus_id"],
        revised_syllabus=revised,
        change_summary="永久删除哨兵修订说明",
        expected_version=family["version"],
        idempotency_key="purge-revision",
    )

    result = store.purge_families(
        [family["family_id"]], occurred_at_utc="2026-08-12T12:00:00Z"
    )
    assert result == {"families": 1, "events": 2, "tombstones": 1}
    raw = store.path.read_text()
    assert "永久删除哨兵" not in raw
    assert base["syllabus_id"] not in raw
    assert family["family_id"] not in raw
    assert json.loads(raw)["events"] == []
    with pytest.raises(TeachingSyllabusVersionError, match="permanently erased"):
        store.register(base, idempotency_key="resurrection")
    assert store.purge_families([family["family_id"]])["events"] == 0
